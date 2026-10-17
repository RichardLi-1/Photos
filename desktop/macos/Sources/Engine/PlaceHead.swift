import Accelerate
import Foundation

/// Our own trained "same place" head (search/photosearch/heads.py, weights from search/data/models/hotel_head.pt),
/// exported by ml/export_place_head.py as plain Float32 and applied with vDSP:
///     z = normalize(x·Pᵀ + GELU(x·W1ᵀ + b1)·W2ᵀ + b2)        768 → 512 (hidden 1024)
/// It maps SigLIP2 vectors into a space where photos of the same room/place cluster (hotels R@1 0.196 → 0.359).
struct PlaceHead: Sendable {
    let inDim: Int, hidden: Int, dim: Int
    private let projT: [Float], w1T: [Float], b1: [Float], w2T: [Float], b2: [Float]
    let fixture: (x: [[Float]], z: [[Float]])

    private struct Meta: Decodable {
        struct Tensor: Decodable { let offset: Int; let shape: [Int] }
        struct Fixture: Decodable { let x: [[Float]]; let z: [[Float]] }
        let in_dim: Int, hidden: Int, dim: Int, floats: Int
        let tensors: [String: Tensor]
        let fixture: Fixture
    }

    init(bundle: Bundle = .main) throws {
        guard let w = bundle.url(forResource: "place_head", withExtension: "f32"),
              let m = bundle.url(forResource: "place_head", withExtension: "json")
        else { throw SigLIPEncoder.Failure.missingResource("place_head") }
        try self.init(weights: w, meta: m)
    }

    init(weights: URL, meta metaURL: URL) throws {
        let meta = try JSONDecoder().decode(Meta.self, from: Data(contentsOf: metaURL))
        let data = try Data(contentsOf: weights, options: .alwaysMapped)
        guard data.count == meta.floats * 4 else { throw SigLIPEncoder.Failure.badOutput }
        let all: [Float] = data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }  // little-endian, as written
        func tensor(_ name: String, _ shape: [Int]) throws -> [Float] {
            guard let t = meta.tensors[name], t.shape == shape else { throw SigLIPEncoder.Failure.badOutput }
            let n = shape.reduce(1, *)
            return Array(all[t.offset..<t.offset + n])
        }
        inDim = meta.in_dim; hidden = meta.hidden; dim = meta.dim
        projT = try tensor("proj_t", [inDim, dim])
        w1T = try tensor("w1_t", [inDim, hidden])
        b1 = try tensor("b1", [hidden])
        w2T = try tensor("w2_t", [hidden, dim])
        b2 = try tensor("b2", [dim])
        fixture = (meta.fixture.x, meta.fixture.z)
    }

    /// `rows × inDim` row-major unit vectors → `rows × dim` row-major unit vectors (chunked to bound memory).
    func embed(_ x: [Float], rows: Int) -> [Float] {
        precondition(x.count == rows * inDim)
        var out = [Float](repeating: 0, count: rows * dim)
        let chunk = 2048
        var h = [Float](repeating: 0, count: chunk * hidden)
        var z = [Float](repeating: 0, count: chunk * dim)
        for start in stride(from: 0, to: rows, by: chunk) {
            let n = min(chunk, rows - start)
            x.withUnsafeBufferPointer { xb in
                let xp = xb.baseAddress! + start * inDim
                vDSP_mmul(xp, 1, projT, 1, &z, 1, vDSP_Length(n), vDSP_Length(dim), vDSP_Length(inDim))   // x·Pᵀ
                vDSP_mmul(xp, 1, w1T, 1, &h, 1, vDSP_Length(n), vDSP_Length(hidden), vDSP_Length(inDim))  // x·W1ᵀ
            }
            for r in 0..<n {  // + b1, exact GELU (torch nn.GELU default)
                for j in 0..<hidden {
                    let v = h[r * hidden + j] + b1[j]
                    h[r * hidden + j] = 0.5 * v * (1 + erf(v * 0.70710678118654752))
                }
            }
            var m = [Float](repeating: 0, count: n * dim)
            vDSP_mmul(h, 1, w2T, 1, &m, 1, vDSP_Length(n), vDSP_Length(dim), vDSP_Length(hidden))       // ·W2ᵀ
            for r in 0..<n {
                var norm: Float = 0
                for j in 0..<dim {
                    let v = z[r * dim + j] + m[r * dim + j] + b2[j]
                    z[r * dim + j] = v
                    norm += v * v
                }
                let inv = norm > 0 ? 1 / norm.squareRoot() : 0
                for j in 0..<dim { out[(start + r) * dim + j] = z[r * dim + j] * inv }
            }
        }
        return out
    }

    /// Min cosine between this implementation and PyTorch on the exported fixture (≥ 0.9999 expected).
    func parity() -> Float {
        let rows = fixture.x.count
        let z = embed(fixture.x.flatMap { $0 }, rows: rows)
        return (0..<rows).map { r in SearchMath.cosine(Array(z[r * dim..<(r + 1) * dim]), fixture.z[r]) }.min() ?? 0
    }
}

/// "Same place": cosine in the head's space plus shooting-session propagation (search.py `_same_place`).
/// Keeps its own copy of the vectors, projected through the head, and rebuilds lazily after indexing adds photos.
actor PlaceIndex {
    private let head: PlaceHead
    private let store: LibraryStore
    private var ids: [Int64] = []
    private var localIDs: [String] = []
    private var row: [Int64: Int] = [:]
    private var z: [Float] = []
    private var sessions: [Int] = []
    private var dirty = true

    init(head: PlaceHead, store: LibraryStore) { self.head = head; self.store = store }

    func invalidate() { dirty = true }

    private func refresh() async throws {
        guard dirty else { return }
        let s = try await store.snapshot(dim: SigLIPEncoder.dim)
        ids = s.ids; localIDs = s.localIDs
        row = Dictionary(uniqueKeysWithValues: ids.enumerated().map { ($1, $0) })
        z = head.embed(s.matrix, rows: ids.count)
        sessions = SearchMath.sessions(taken: s.taken, lat: s.lat, lon: s.lon)
        dirty = false
    }

    func samePlace(as ref: Int64, limit: Int = 120) async throws -> [SearchHit] {
        try await refresh()
        guard let r = row[ref] else { throw SearchEngine.Failure.notIndexed }
        let d = head.dim
        let q = Array(z[r * d..<(r + 1) * d])
        var sim = SearchMath.propagateSessions(SearchMath.scores(matrix: z, rows: ids.count, dim: d, query: q), sessions: sessions)
        sim[r] = -.infinity
        return SearchMath.topK(sim, k: limit).map { SearchHit(id: ids[$0], localID: localIDs[$0], score: sim[$0]) }
    }
}
