import Foundation

struct SearchFilters: Sendable, Hashable {
    var after: Date?
    var before: Date?
    var place: String?
    var hasText = false

    var isActive: Bool { after != nil || before != nil || place != nil || hasText }
}

struct SearchRequest: Sendable, Hashable {
    var text: String?
    var imageVector: [Float]?      // a photo picked or taken by the user (not necessarily in the library)
    var similarTo: Int64?          // "More like this" from an indexed photo
    var textWeight: Float = 0.5    // with an image + text: how much the text steers
    var filters = SearchFilters()
    var limit = 120
}

struct SearchHit: Sendable, Hashable, Identifiable {
    let id: Int64
    let localID: String
    let score: Float?
}

struct SearchResponse: Sendable {
    var hits: [SearchHit]
    var queryTags: [String] = []   // concepts in the user's query photo (for match cards)
    var elapsed: Duration = .zero
}

/// Why a photo matched, in a few words (the web prototype's /api/explain).
struct Explanation: Sendable, Equatable {
    enum Mode: Sendable { case text, similar, photo, browse }
    var mode: Mode
    var shared: [String]
    var overall = false            // nothing specific matched: these are the photo's own top concepts
    var textHits: [String] = []    // query words written in the photo (OCR)
    var placeHits: [String] = []
    var place: String?
    var date: Date?
}

enum MatchContext: Sendable, Hashable {
    case browse
    case text(String)
    case similar(Int64)
    case photo(tags: [String])
}

/// Hybrid retrieval: brute-force cosine over an in-memory Float32 matrix (vDSP), fused (RRF) with FTS5 hits on OCR text
/// and place names, plus date/place filters. Mirrors search/photosearch/search.py.
actor SearchEngine {
    enum Failure: Error { case modelsUnavailable, notIndexed }

    private let store: LibraryStore
    private var encoder: SigLIPEncoder?
    private var explainer: ConceptExplainer?
    private var snap = VectorSnapshot()
    private var row: [Int64: Int] = [:]
    private var sessions: [Int] = []
    private var sessionsDirty = true
    private var textCache: [String: [Float]] = [:]
    private let dim = SigLIPEncoder.dim

    init(store: LibraryStore) { self.store = store }

    var count: Int { snap.ids.count }

    func attach(encoder: SigLIPEncoder, explainer: ConceptExplainer?) {
        self.encoder = encoder
        self.explainer = explainer
    }

    func reload() async throws {
        snap = try await store.snapshot(dim: dim)
        row = Dictionary(uniqueKeysWithValues: snap.ids.enumerated().map { ($1, $0) })
        sessionsDirty = true
    }

    /// Adds freshly indexed photos without reloading everything.
    func upsert(_ items: [(record: PhotoRecord, vector: [Float])]) {
        for (r, v) in items where v.count == dim {
            if let i = row[r.id] {
                snap.matrix.replaceSubrange(i * dim..<(i + 1) * dim, with: v)
                continue
            }
            row[r.id] = snap.ids.count
            snap.ids.append(r.id); snap.localIDs.append(r.localID)
            snap.taken.append(r.taken?.timeIntervalSince1970); snap.lat.append(r.lat); snap.lon.append(r.lon)
            snap.place.append(r.place)
            snap.matrix += v
        }
        sessionsDirty = true
    }

    func remove(_ ids: [Int64]) async throws {
        guard ids.contains(where: { row[$0] != nil }) else { return }
        try await reload()
    }

    func updatePlace(_ place: String?, for id: Int64) {
        if let i = row[id] { snap.place[i] = place }
    }

    func vector(for id: Int64) -> [Float]? {
        row[id].map { Array(snap.matrix[$0 * dim..<($0 + 1) * dim]) }
    }

    func embed(text: String) throws -> [Float] {
        let key = text.lowercased()
        if let v = textCache[key] { return v }
        guard let encoder else { throw Failure.modelsUnavailable }
        let v = try encoder.embed(text: text)
        if textCache.count > 64 { textCache.removeAll() }
        textCache[key] = v
        return v
    }

    func tags(for vector: [Float], k: Int = 12, floor: Float = 0.01) -> [String] {
        explainer?.tags(vector, k: k, floor: floor) ?? []
    }

    // MARK: - Search

    func search(_ q: SearchRequest) async throws -> SearchResponse {
        let clock = ContinuousClock(), start = clock.now
        let text = q.text?.trimmingCharacters(in: .whitespacesAndNewlines).nilIfEmpty
        let allowed = try await mask(q.filters)

        // Filters only ("photos from july 2023"): newest first.
        if text == nil && q.imageVector == nil && q.similarTo == nil {
            guard q.filters.isActive else { return SearchResponse(hits: []) }
            let rows = snap.ids.indices.filter { allowed[$0] }.sorted { (snap.taken[$0] ?? 0) > (snap.taken[$1] ?? 0) }
            return SearchResponse(hits: rows.prefix(q.limit).map { hit($0, nil) }, elapsed: clock.now - start)
        }

        var image = q.imageVector
        if let ref = q.similarTo {
            guard let v = vector(for: ref) else { throw Failure.notIndexed }
            image = v
        }
        var query = [Float](repeating: 0, count: dim)
        if let image { query = image }
        if let text {
            let t = try embed(text: text)
            let w: Float = image == nil ? 1 : q.textWeight / (1 - q.textWeight)
            for i in 0..<dim { query[i] += w * t[i] }
        }
        query = SearchMath.normalized(query)

        var sim = SearchMath.scores(matrix: snap.matrix, rows: snap.ids.count, dim: dim, query: query)
        if image != nil && text == nil {
            if sessionsDirty {
                sessions = SearchMath.sessions(taken: snap.taken, lat: snap.lat, lon: snap.lon)
                sessionsDirty = false
            }
            sim = SearchMath.propagateSessions(sim, sessions: sessions)
        }
        for i in sim.indices where !allowed[i] { sim[i] = -.infinity }
        if let ref = q.similarTo, let r = row[ref] { sim[r] = -.infinity }

        let order = SearchMath.topK(sim, k: max(q.limit * 4, 200))
        var rankings: [(ids: [Int64], weight: Float)] = [(order.map { snap.ids[$0] }, 1)]
        if let text {  // literal matches (text in the photo, place names) get a vote too
            let hits = try await store.fulltext(text).filter { row[$0].map { sim[$0].isFinite } ?? false }
            rankings.append((hits, 0.8))
        }
        let fused = SearchMath.rrf(rankings, limit: q.limit)
        var response = SearchResponse(hits: fused.compactMap { f in row[f.id].map { hit($0, sim[$0]) } })
        if let v = q.imageVector { response.queryTags = tags(for: v) }
        response.elapsed = clock.now - start
        return response
    }

    private func hit(_ r: Int, _ score: Float?) -> SearchHit {
        SearchHit(id: snap.ids[r], localID: snap.localIDs[r], score: score)
    }

    private func mask(_ f: SearchFilters) async throws -> [Bool] {
        guard f.isActive else { return Array(repeating: true, count: snap.ids.count) }
        let after = f.after?.timeIntervalSince1970, before = f.before?.timeIntervalSince1970
        let place = f.place?.lowercased()
        let withText: Set<Int64> = f.hasText ? Set(try await store.idsWithText()) : []
        return snap.ids.indices.map { i in
            if let after, (snap.taken[i] ?? -.infinity) < after { return false }
            if let before, (snap.taken[i] ?? .infinity) >= before { return false }
            if let place, !(snap.place[i]?.lowercased().contains(place) ?? false) { return false }
            if f.hasText, !withText.contains(snap.ids[i]) { return false }
            return true
        }
    }

    // MARK: - Explanations

    func explain(_ id: Int64, context: MatchContext) async throws -> Explanation {
        guard let v = vector(for: id), let explainer else { throw Failure.notIndexed }
        let record = try await store.records([id]).first
        var e: Explanation
        switch context {
        case .similar(let ref):
            e = Explanation(mode: .similar, shared: vector(for: ref).map { explainer.shared(withImage: $0, v) } ?? [])
        case .photo(let tags):
            e = Explanation(mode: .photo, shared: explainer.shared(withTags: tags, v))
        case .text(let q):
            e = Explanation(mode: .text, shared: explainer.matched(query: try embed(text: q), v))
            e.textHits = ConceptExplainer.wordHits(query: q, in: record?.ocr)
            e.placeHits = ConceptExplainer.wordHits(query: q, in: record?.place)
        case .browse:
            e = Explanation(mode: .browse, shared: explainer.tags(v, k: 4))
        }
        if e.shared.isEmpty && e.mode != .browse {
            e.shared = explainer.tags(v, k: 3)
            e.overall = true
        }
        e.place = record?.place
        e.date = record?.taken
        return e
    }
}

extension String {
    var nilIfEmpty: String? { isEmpty ? nil : self }
}
