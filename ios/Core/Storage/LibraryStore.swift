import Foundation

/// One indexed photo (mirrors the prototype's `photos` table; `id` is a local rowid, `localID` the PHAsset id).
struct PhotoRecord: Sendable, Hashable, Identifiable {
    let id: Int64
    let localID: String
    let taken: Date?
    let lat: Double?
    let lon: Double?
    let place: String?
    let ocr: String?
}

/// Metadata the indexer reads from PhotoKit for each asset.
struct AssetInfo: Sendable {
    let localID: String
    let taken: Date?
    let modified: Date?
    let lat: Double?
    let lon: Double?
    let width: Int
    let height: Int
}

/// Everything the in-memory search index needs, loaded in one pass.
struct VectorSnapshot: Sendable {
    var ids: [Int64] = []
    var localIDs: [String] = []
    var matrix: [Float] = []       // ids.count × dim, unit rows
    var taken: [Double?] = []
    var lat: [Double?] = []
    var lon: [Double?] = []
    var place: [String?] = []
}

/// The on-device index: SQLite (metadata, OCR, FTS5 full text, Float16 embeddings) under Application Support,
/// excluded from iCloud/iTunes backup. Nothing here touches the network.
actor LibraryStore {
    static let modelID = "siglip2-b16-384/coreml-fp16"
    let directory: URL
    private let db: SQLite
    private(set) var hasFTS = true

    static func defaultDirectory() throws -> URL {
        let base = try FileManager.default.url(for: .applicationSupportDirectory, in: .userDomainMask, appropriateFor: nil, create: true)
        return base.appendingPathComponent("PhotoIndex", isDirectory: true)
    }

    init(directory: URL? = nil) throws {
        var dir = try directory ?? Self.defaultDirectory()
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        var values = URLResourceValues()
        values.isExcludedFromBackup = true  // derived data: rebuilt from the library, never backed up
        try dir.setResourceValues(values)
        self.directory = dir
        db = try SQLite(path: dir.appendingPathComponent("library.sqlite").path)
        try db.execute("PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;")
        try db.execute("""
            CREATE TABLE IF NOT EXISTS photos (
              id INTEGER PRIMARY KEY, local_id TEXT UNIQUE NOT NULL, taken REAL, modified REAL, lat REAL, lon REAL,
              place TEXT, place_done INT DEFAULT 0, width INT, height INT, ocr TEXT, ocr_done INT DEFAULT 0,
              embedding BLOB, failed INT DEFAULT 0);
            CREATE INDEX IF NOT EXISTS photos_taken ON photos(taken);
            CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
            """)
        do {
            try db.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS fts USING fts5(ocr, place, content='photos', content_rowid='id', tokenize='porter unicode61');
                CREATE TRIGGER IF NOT EXISTS photos_ai AFTER INSERT ON photos BEGIN
                  INSERT INTO fts(rowid, ocr, place) VALUES (new.id, new.ocr, new.place); END;
                CREATE TRIGGER IF NOT EXISTS photos_ad AFTER DELETE ON photos BEGIN
                  INSERT INTO fts(fts, rowid, ocr, place) VALUES('delete', old.id, old.ocr, old.place); END;
                CREATE TRIGGER IF NOT EXISTS photos_au AFTER UPDATE OF ocr, place ON photos BEGIN
                  INSERT INTO fts(fts, rowid, ocr, place) VALUES('delete', old.id, old.ocr, old.place);
                  INSERT INTO fts(rowid, ocr, place) VALUES (new.id, new.ocr, new.place); END;
                """)
        } catch {
            hasFTS = false  // FTS5 missing: fall back to LIKE matching
        }
        // A different model means different vectors: re-embed everything (OCR stays).
        if try db.query("SELECT value FROM meta WHERE key='model'").first?.first?.string != Self.modelID {
            try db.run("UPDATE photos SET embedding=NULL, failed=0")
            try db.run("INSERT OR REPLACE INTO meta(key, value) VALUES ('model', ?)", [.text(Self.modelID)])
        }
    }

    // MARK: - Sync with PhotoKit

    /// Upserts the current library listing; removes rows for assets that are gone. Edited photos are re-indexed.
    @discardableResult
    func sync(_ assets: [AssetInfo]) throws -> (added: Int, removed: [Int64]) {
        var known: [String: (id: Int64, modified: Double?)] = [:]
        try db.each("SELECT id, local_id, modified FROM photos") { r in
            known[r[1].string!] = (r[0].int!, r[2].double)
        }
        var added = 0
        let present = Set(assets.map(\.localID))
        let removed = known.filter { !present.contains($0.key) }.map(\.value.id)
        try db.transaction {
            for a in assets {
                let modified = a.modified?.timeIntervalSince1970
                if let k = known[a.localID] {
                    if let m = modified, let old = k.modified, m > old + 1 {
                        try db.run("UPDATE photos SET modified=?, taken=?, lat=?, lon=?, embedding=NULL, ocr=NULL, ocr_done=0, failed=0 WHERE id=?",
                                   [.init(modified), .init(a.taken?.timeIntervalSince1970), .init(a.lat), .init(a.lon), .int(k.id)])
                    }
                    continue
                }
                try db.run("INSERT INTO photos(local_id, taken, modified, lat, lon, width, height) VALUES (?,?,?,?,?,?,?)",
                           [.text(a.localID), .init(a.taken?.timeIntervalSince1970), .init(modified), .init(a.lat), .init(a.lon),
                            .int(Int64(a.width)), .int(Int64(a.height))])
                added += 1
            }
            for id in removed { try db.run("DELETE FROM photos WHERE id=?", [.int(id)]) }
        }
        return (added, removed)
    }

    // MARK: - Indexing work queue

    struct Pending: Sendable { let id: Int64; let localID: String; let needsEmbedding: Bool; let needsOCR: Bool }

    /// Newest photos first, so recent memories become searchable soonest.
    func pending(limit: Int) throws -> [Pending] {
        try db.query("""
            SELECT id, local_id, embedding IS NULL, ocr_done = 0 FROM photos
            WHERE failed = 0 AND (embedding IS NULL OR ocr_done = 0) ORDER BY taken DESC LIMIT ?
            """, [.int(Int64(limit))]).map { Pending(id: $0[0].int!, localID: $0[1].string!, needsEmbedding: $0[2].int == 1, needsOCR: $0[3].int == 1) }
    }

    struct Result: Sendable { let id: Int64; var embedding: [Float]?; var ocr: String?; var ocrDone: Bool; var failed: Bool }

    func save(_ results: [Result]) throws {
        try db.transaction {
            for r in results {
                if r.failed {
                    try db.run("UPDATE photos SET failed=1 WHERE id=?", [.int(r.id)])
                    continue
                }
                if let e = r.embedding {
                    try db.run("UPDATE photos SET embedding=? WHERE id=?", [.blob(SearchMath.halfData(from: e)), .int(r.id)])
                }
                if r.ocrDone {
                    try db.run("UPDATE photos SET ocr=?, ocr_done=1 WHERE id=?", [.init(r.ocr), .int(r.id)])
                }
            }
        }
    }

    func counts() throws -> (total: Int, embedded: Int, failed: Int) {
        let r = try db.query("SELECT count(*), count(embedding), sum(failed) FROM photos").first ?? []
        return (Int(r[0].int ?? 0), Int(r[1].int ?? 0), Int(r[2].int ?? 0))
    }

    // MARK: - Places (opt-in reverse geocoding)

    func pendingPlaces(limit: Int) throws -> [(id: Int64, lat: Double, lon: Double)] {
        try db.query("SELECT id, lat, lon FROM photos WHERE lat IS NOT NULL AND place_done = 0 ORDER BY taken DESC LIMIT ?", [.int(Int64(limit))])
            .map { ($0[0].int!, $0[1].double!, $0[2].double!) }
    }

    func setPlace(_ place: String?, for id: Int64) throws {
        try db.run("UPDATE photos SET place=?, place_done=1 WHERE id=?", [.init(place), .int(id)])
    }

    /// Distinct place names, for matching words in a query against them.
    func placeNames() throws -> [String] {
        try db.query("SELECT DISTINCT place FROM photos WHERE place IS NOT NULL").compactMap { $0[0].string }
    }

    // MARK: - Reads

    func snapshot(dim: Int) throws -> VectorSnapshot {
        var s = VectorSnapshot()
        try db.each("SELECT id, local_id, taken, lat, lon, place, embedding FROM photos WHERE embedding IS NOT NULL") { r in
            guard let blob = r[6].data, blob.count == dim * 2 else { return }
            s.ids.append(r[0].int!)
            s.localIDs.append(r[1].string!)
            s.taken.append(r[2].double)
            s.lat.append(r[3].double)
            s.lon.append(r[4].double)
            s.place.append(r[5].string)
            s.matrix += SearchMath.floats(fromHalf: blob)
        }
        return s
    }

    func records(_ ids: [Int64]) throws -> [PhotoRecord] {
        guard !ids.isEmpty else { return [] }
        let marks = Array(repeating: "?", count: ids.count).joined(separator: ",")
        var by: [Int64: PhotoRecord] = [:]
        try db.each("SELECT id, local_id, taken, lat, lon, place, ocr FROM photos WHERE id IN (\(marks))", ids.map { .int($0) }) { r in
            by[r[0].int!] = PhotoRecord(id: r[0].int!, localID: r[1].string!, taken: r[2].double.map(Date.init(timeIntervalSince1970:)),
                                        lat: r[3].double, lon: r[4].double, place: r[5].string, ocr: r[6].string)
        }
        return ids.compactMap { by[$0] }
    }

    func idsWithText() throws -> [Int64] {
        try db.query("SELECT id FROM photos WHERE ocr IS NOT NULL").compactMap { $0[0].int }
    }

    func id(forLocalID localID: String) throws -> Int64? {
        try db.query("SELECT id FROM photos WHERE local_id=?", [.text(localID)]).first?.first?.int
    }

    /// BM25 over OCR text + place names (OCR weighted 1, place 2 as in the prototype). Words are OR'ed.
    func fulltext(_ text: String, limit: Int = 300) throws -> [Int64] {
        let words = Self.words(text)
        guard !words.isEmpty else { return [] }
        if hasFTS {
            let q = words.map { "\"\($0)\"" }.joined(separator: " OR ")
            return (try? db.query("SELECT rowid FROM fts WHERE fts MATCH ? ORDER BY bm25(fts, 1.0, 2.0) LIMIT ?", [.text(q), .int(Int64(limit))])
                .compactMap { $0[0].int }) ?? []
        }
        let clause = words.map { _ in "(ocr LIKE ? OR place LIKE ?)" }.joined(separator: " OR ")
        return try db.query("SELECT id FROM photos WHERE \(clause) LIMIT ?", words.flatMap { [.text("%\($0)%"), .text("%\($0)%")] } + [.int(Int64(limit))])
            .compactMap { $0[0].int }
    }

    /// Query words worth matching literally: lowercase, ≥ 3 letters (as `fulltext` in search.py).
    static func words(_ text: String) -> [String] {
        let tokens = text.lowercased().split { !($0.isLetter || $0.isNumber) }.map(String.init)
        var seen = Set<String>()
        return tokens.filter { $0.count > 2 && seen.insert($0).inserted }
    }

    /// Wipes the index (Settings → "Rebuild index").
    func reset() throws {
        try db.execute("DELETE FROM photos")  // the delete trigger keeps fts in step
    }
}
