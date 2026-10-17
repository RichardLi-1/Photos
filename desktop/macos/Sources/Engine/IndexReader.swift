import Foundation

/// A second, read-mostly connection to the shared index (`LibraryStore`'s library.sqlite, WAL mode) for the few
/// queries the Mac app needs that the shared store doesn't expose: what's already indexed (so unchanged files aren't
/// re-read on every scan) and the newest-first browse wall.
actor IndexReader {
    private let db: SQLite

    init(directory: URL) throws {
        db = try SQLite(path: directory.appendingPathComponent("library.sqlite").path)
        try db.execute("PRAGMA query_only=1;")
    }

    struct Known: Sendable { let modified: Double?; let info: AssetInfo }

    /// Rows already in the index, keyed by localID, rebuilt as `AssetInfo` (no file access needed).
    func known() throws -> [String: Known] {
        var out: [String: Known] = [:]
        try db.each("SELECT local_id, modified, taken, lat, lon, width, height FROM photos") { r in
            guard let id = r[0].string else { return }
            let modified = r[1].double
            out[id] = Known(modified: modified, info: AssetInfo(
                localID: id, taken: r[2].double.map(Date.init(timeIntervalSince1970:)),
                modified: modified.map(Date.init(timeIntervalSince1970:)), lat: r[3].double, lon: r[4].double,
                width: Int(r[5].int ?? 0), height: Int(r[6].int ?? 0)))
        }
        return out
    }

    /// Indexed photos, newest first (undated last, then by insertion).
    func browse(limit: Int = 5000) throws -> [SearchHit] {
        try db.query("""
            SELECT id, local_id FROM photos WHERE embedding IS NOT NULL
            ORDER BY taken IS NULL, taken DESC, id DESC LIMIT ?
            """, [.int(Int64(limit))]).compactMap { r in
            guard let id = r[0].int, let local = r[1].string else { return nil }
            return SearchHit(id: id, localID: local, score: nil)
        }
    }

    func localIDs(_ ids: [Int64]) throws -> [Int64: String] {
        guard !ids.isEmpty else { return [:] }
        let marks = Array(repeating: "?", count: ids.count).joined(separator: ",")
        var out: [Int64: String] = [:]
        try db.each("SELECT id, local_id FROM photos WHERE id IN (\(marks))", ids.map { .int($0) }) { r in
            if let id = r[0].int, let s = r[1].string { out[id] = s }
        }
        return out
    }
}
