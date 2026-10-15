import Foundation

/// "Why it matched": zero-shot concept chips from SigLIP's calibrated sigmoid head (mirrors search/photosearch/explain.py).
/// The concept list and its text embeddings ("a photo of {concept}") ship as resources made by ml/convert_siglip2.py.
struct ConceptExplainer: Sendable {
    let concepts: [String]
    private let matrix: [Float]  // concepts.count × dim
    private let dim: Int
    private let scale: Float, bias: Float

    private struct Meta: Decodable { let concepts: [String]; let dim: Int }

    init(concepts: [String], matrix: [Float], dim: Int, scale: Float, bias: Float) {
        self.concepts = concepts; self.matrix = matrix; self.dim = dim; self.scale = scale; self.bias = bias
    }

    init(bundle: Bundle = .main, calibration: SigLIPEncoder.Calibration) throws {
        guard let metaURL = bundle.url(forResource: "siglip2_concepts", withExtension: "json"),
              let vecURL = bundle.url(forResource: "siglip2_concepts", withExtension: "f16")
        else { throw SigLIPEncoder.Failure.missingResource("siglip2_concepts") }
        let meta = try JSONDecoder().decode(Meta.self, from: Data(contentsOf: metaURL))
        let m = SearchMath.floats(fromHalf: try Data(contentsOf: vecURL))
        guard m.count == meta.concepts.count * meta.dim else { throw SigLIPEncoder.Failure.badOutput }
        self.init(concepts: meta.concepts, matrix: m, dim: meta.dim, scale: calibration.logit_scale, bias: calibration.logit_bias)
    }

    /// P(photo shows concept) for every concept.
    func probabilities(_ v: [Float]) -> [Float] {
        SearchMath.sigmoid(SearchMath.scores(matrix: matrix, rows: concepts.count, dim: dim, query: v), scale: scale, bias: bias)
    }

    /// Top-k above an absolute floor and within 15× of the best (SigLIP probabilities are small but well ranked).
    private func top(_ score: [Float], k: Int, floor: Float) -> [String] {
        let order = SearchMath.topK(score, k: k)
        let best = order.first.map { score[$0] } ?? 0
        return order.filter { score[$0] >= floor && score[$0] >= best / 15 }.map { concepts[$0] }
    }

    func tags(_ v: [Float], k: Int = 4, floor: Float = 0.002) -> [String] {
        top(probabilities(v), k: k, floor: floor)
    }

    /// Concepts present in both photos ("More like this").
    func shared(withImage ref: [Float], _ v: [Float], k: Int = 4) -> [String] {
        top(zip(probabilities(ref), probabilities(v)).map(min), k: k, floor: 0.001)
    }

    /// Of the query photo's own tags, which this result shows too (search by a picked photo).
    func shared(withTags tags: [String], _ v: [Float], k: Int = 4) -> [String] {
        let p = probabilities(v)
        let wanted = Set(tags)
        return top(p.indices.map { wanted.contains(concepts[$0]) ? p[$0] : -1 }, k: k, floor: 0.001)
    }

    /// Concepts close in meaning to the text query (z-score > 2 against all concepts) that the photo clearly shows.
    func matched(query q: [Float], _ v: [Float], k: Int = 3) -> [String] {
        let rel = SearchMath.scores(matrix: matrix, rows: concepts.count, dim: dim, query: q)
        let mean = rel.reduce(0, +) / Float(rel.count)
        let std = (rel.map { ($0 - mean) * ($0 - mean) }.reduce(0, +) / Float(rel.count)).squareRoot()
        let p = probabilities(v)
        return top(rel.indices.map { (rel[$0] - mean) / (std + 1e-6) > 2 ? p[$0] : -1 }, k: k, floor: 0.001)
    }

    static let stopWords: Set<String> = Set("""
        the and for with from that this are was were you your our photo photos picture pictures image images show find \
        some any all into onto over under near
        """.split(separator: " ").map(String.init))

    /// Query words (≥ 3 letters, not stop words) that literally appear in `text` (OCR or place name).
    static func wordHits(query: String, in text: String?) -> [String] {
        guard let text, !text.isEmpty else { return [] }
        let haystack = text.lowercased()
        var seen = Set<String>()
        return query.lowercased().split { !($0.isLetter || $0.isNumber || $0 == "_") }.map(String.init)
            .filter { $0.count >= 3 && !stopWords.contains($0) && seen.insert($0).inserted }
            .filter { word in
                // word-start match, like the prototype's \bword regex
                var range = haystack.startIndex..<haystack.endIndex
                while let r = haystack.range(of: word, range: range) {
                    if r.lowerBound == haystack.startIndex { return true }
                    let c = haystack[haystack.index(before: r.lowerBound)]
                    if !(c.isLetter || c.isNumber || c == "_") { return true }
                    range = r.upperBound..<haystack.endIndex
                }
                return false
            }
    }
}
