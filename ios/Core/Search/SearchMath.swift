import Accelerate
import Foundation

/// Pure vector / ranking math shared by search, explanations and tests (mirrors search/photosearch/search.py).
enum SearchMath {
    /// Reciprocal-rank-fusion constant, as in the prototype (`Searcher.RRF_K`).
    static let rrfK: Float = 60

    static func normalized(_ v: [Float]) -> [Float] {
        var norm: Float = 0
        vDSP_svesq(v, 1, &norm, vDSP_Length(v.count))
        norm = norm.squareRoot()
        guard norm > 0 else { return v }
        return vDSP.divide(v, norm)
    }

    static func dot(_ a: [Float], _ b: [Float]) -> Float {
        precondition(a.count == b.count)
        return vDSP.dot(a, b)
    }

    static func cosine(_ a: [Float], _ b: [Float]) -> Float {
        dot(normalized(a), normalized(b))
    }

    /// `matrix` is row-major `rows × dim` of unit vectors; returns `matrix · query` (cosine for unit vectors).
    static func scores(matrix: [Float], rows: Int, dim: Int, query: [Float]) -> [Float] {
        precondition(query.count == dim && matrix.count == rows * dim)
        guard rows > 0 else { return [] }
        var out = [Float](repeating: 0, count: rows)
        vDSP_mmul(matrix, 1, query, 1, &out, 1, vDSP_Length(rows), 1, vDSP_Length(dim))
        return out
    }

    /// Indices of the `k` highest finite scores, best first (ties → lower index first).
    static func topK(_ scores: [Float], k: Int) -> [Int] {
        guard k > 0 else { return [] }
        let finite = scores.indices.filter { scores[$0].isFinite }
        if finite.count <= k {
            return finite.sorted { scores[$0] > scores[$1] || (scores[$0] == scores[$1] && $0 < $1) }
        }
        // Partial selection: keep a small sorted buffer; O(n log k)-ish and allocation-free per element.
        var best: [Int] = []
        best.reserveCapacity(k + 1)
        for i in finite {
            let s = scores[i]
            if best.count == k, let last = best.last, s <= scores[last] { continue }
            let pos = best.partitioningIndex { scores[$0] < s }
            best.insert(i, at: pos)
            if best.count > k { best.removeLast() }
        }
        return best
    }

    /// Weighted reciprocal-rank fusion: each ranking adds `weight / (rrfK + rank)` (rank from 0) per id.
    /// The prototype uses weight 1 for the embedding ranking and 0.8 for full-text hits.
    static func rrf<ID: Hashable>(_ rankings: [(ids: [ID], weight: Float)], limit: Int) -> [(id: ID, score: Float)] {
        var fused: [ID: Float] = [:]
        var firstSeen: [ID: Int] = [:]
        var order = 0
        for ranking in rankings {
            for (rank, id) in ranking.ids.enumerated() {
                fused[id, default: 0] += ranking.weight / (rrfK + Float(rank))
                if firstSeen[id] == nil { firstSeen[id] = order; order += 1 }
            }
        }
        return fused.sorted { a, b in a.value > b.value || (a.value == b.value && firstSeen[a.key]! < firstSeen[b.key]!) }
            .prefix(limit).map { ($0.key, $0.value) }
    }

    /// SigLIP's calibrated sigmoid head: P(match) = σ(scale · cos + bias).
    static func sigmoid(_ x: [Float], scale: Float, bias: Float) -> [Float] {
        x.map { 1 / (1 + exp(-(scale * $0 + bias))) }
    }

    static func haversineKm(_ lat1: Double, _ lon1: Double, _ lat2: Double, _ lon2: Double) -> Double {
        let p = Double.pi / 180
        let a = pow(sin((lat2 - lat1) * p / 2), 2) + cos(lat1 * p) * cos(lat2 * p) * pow(sin((lon2 - lon1) * p / 2), 2)
        return 12742 * asin(a.squareRoot())
    }

    /// Shooting sessions: consecutive in time (< gap) and, when both have GPS, within `km`. Returns a session per item
    /// (-1 when undated), like `Library.assign_sessions`.
    static func sessions(taken: [Double?], lat: [Double?], lon: [Double?], gap: Double = 45 * 60, km: Double = 1) -> [Int] {
        var out = [Int](repeating: -1, count: taken.count)
        let order = taken.indices.filter { taken[$0] != nil }.sorted { taken[$0]! < taken[$1]! }
        var session = -1
        var prev: Int?
        for i in order {
            var new = prev == nil || taken[i]! - taken[prev!]! > gap
            if !new, let p = prev, let a = lat[i], let b = lon[i], let c = lat[p], let d = lon[p] {
                new = haversineKm(a, b, c, d) > km
            }
            if new { session += 1 }
            out[i] = session
            prev = i
        }
        return out
    }

    /// Photos from the same session as a strong visual match inherit most of its score (bed vs bathroom of one room).
    static func propagateSessions(_ sim: [Float], sessions: [Int], strong: Float = 0.6, gain: Float = 0.9) -> [Float] {
        var best: [Int: Float] = [:]
        for (i, s) in sim.enumerated() where s >= strong && sessions[i] >= 0 {
            best[sessions[i]] = max(best[sessions[i]] ?? -1, s)
        }
        guard !best.isEmpty else { return sim }
        var out = sim
        for i in out.indices {
            if let b = best[sessions[i]], out[i].isFinite { out[i] = max(out[i], gain * b) }
        }
        return out
    }

    /// Float16 bytes → Float32 (vImage; works on every architecture).
    static func floats(fromHalf data: Data) -> [Float] {
        let n = data.count / 2
        var out = [Float](repeating: 0, count: n)
        data.withUnsafeBytes { raw in
            var src = vImage_Buffer(data: UnsafeMutableRawPointer(mutating: raw.baseAddress), height: 1, width: vImagePixelCount(n), rowBytes: n * 2)
            out.withUnsafeMutableBytes { dst in
                var d = vImage_Buffer(data: dst.baseAddress, height: 1, width: vImagePixelCount(n), rowBytes: n * 4)
                vImageConvert_Planar16FtoPlanarF(&src, &d, 0)
            }
        }
        return out
    }

    static func halfData(from v: [Float]) -> Data {
        var data = Data(count: v.count * 2)
        v.withUnsafeBytes { raw in
            var src = vImage_Buffer(data: UnsafeMutableRawPointer(mutating: raw.baseAddress), height: 1, width: vImagePixelCount(v.count), rowBytes: v.count * 4)
            data.withUnsafeMutableBytes { dst in
                var d = vImage_Buffer(data: dst.baseAddress, height: 1, width: vImagePixelCount(v.count), rowBytes: v.count * 2)
                vImageConvert_PlanarFtoPlanar16F(&src, &d, 0)
            }
        }
        return data
    }
}

private extension Array {
    /// First index where `belongsInSecondPartition` is true, for an array already partitioned by it.
    func partitioningIndex(where belongsInSecondPartition: (Element) -> Bool) -> Int {
        var lo = 0, hi = count
        while lo < hi {
            let mid = (lo + hi) / 2
            if belongsInSecondPartition(self[mid]) { hi = mid } else { lo = mid + 1 }
        }
        return lo
    }
}
