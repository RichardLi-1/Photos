//
//  PhotosTests.swift
//  PhotosTests
//
//  Created by Richard Li on 2026-01-28.
//

import Foundation
import Testing
@testable import PhotosApp

private final class TestBundle {}

struct TokenizerTests {
    struct Fixture: Decodable { let text: String; let ids: [Int32] }

    /// siglip2_token_fixtures.json comes from HF `tokenizers` (ml/export_siglip2_tokenizer.py).
    @Test func matchesPythonTokenizer() throws {
        let url = try #require(Bundle(for: TestBundle.self).url(forResource: "siglip2_token_fixtures", withExtension: "json"))
        let fixtures = try JSONDecoder().decode([Fixture].self, from: Data(contentsOf: url))
        #expect(fixtures.count >= 10)
        let tokenizer = try SigLIPTokenizer(bundle: .main)
        for f in fixtures {
            #expect(tokenizer.encode(f.text) == f.ids, "\(f.text)")
        }
    }

    @Test func padsAndTruncatesTo64() throws {
        let tokenizer = try SigLIPTokenizer(bundle: .main)
        let short = tokenizer.encode("dog")
        #expect(short.count == 64 && short.contains(tokenizer.eosID) && short.last == tokenizer.padID)
        let long = tokenizer.encode(String(repeating: "mountain lake ", count: 100))
        #expect(long.count == 64 && long.last == tokenizer.eosID)
    }

    @Test func cleansLikeOpenCLIP() {
        #expect(SigLIPTokenizer.clean("  Sunset over the BEACH!  ") == "sunset over the beach")
        #expect(SigLIPTokenizer.clean("receipt_from_costco") == "receipt from costco")
        #expect(SigLIPTokenizer.clean("Tom &amp; Jerry") == "tom jerry")
    }
}

struct SearchMathTests {
    @Test func cosineAndNormalize() {
        #expect(abs(SearchMath.cosine([1, 0], [1, 0]) - 1) < 1e-6)
        #expect(abs(SearchMath.cosine([1, 0], [0, 2])) < 1e-6)
        #expect(abs(SearchMath.cosine([1, 1], [-1, -1]) + 1) < 1e-6)
        let n = SearchMath.normalized([3, 4])
        #expect(abs(n[0] - 0.6) < 1e-6 && abs(n[1] - 0.8) < 1e-6)
    }

    @Test func matrixScores() {
        let m: [Float] = [1, 0, 0,
                          0, 1, 0,
                          0.6, 0.8, 0]
        let s = SearchMath.scores(matrix: m, rows: 3, dim: 3, query: [0, 1, 0])
        #expect(s == [0, 1, 0.8])
    }

    @Test func topKSkipsNonFiniteAndKeepsOrder() {
        let s: [Float] = [0.1, 0.9, -.infinity, 0.5, 0.9, 0.3]
        #expect(SearchMath.topK(s, k: 3) == [1, 4, 3])
        #expect(SearchMath.topK(s, k: 10) == [1, 4, 3, 5, 0])
        #expect(SearchMath.topK(s, k: 0).isEmpty)
    }

    /// Same fusion as search.py: 1/(60+rank) for the vector ranking, 0.8/(60+rank) for full-text hits.
    @Test func reciprocalRankFusion() {
        let fused = SearchMath.rrf([(ids: [10, 20, 30], weight: 1), (ids: [30, 40], weight: 0.8)], limit: 10)
        #expect(fused.map(\.id) == [30, 10, 20, 40])
        #expect(abs(fused[0].score - (1 / 62 + 0.8 / 60)) < 1e-6)
        #expect(abs(fused[3].score - 0.8 / 61) < 1e-6)
        #expect(SearchMath.rrf([(ids: [1, 2, 3], weight: 1)], limit: 2).map(\.id) == [1, 2])
    }

    @Test func sessionsAndPropagation() {
        let t: [Double?] = [0, 600, 100_000, nil]
        let sessions = SearchMath.sessions(taken: t, lat: [nil, nil, nil, nil], lon: [nil, nil, nil, nil])
        #expect(sessions == [0, 0, 1, -1])
        let sim = SearchMath.propagateSessions([0.8, 0.1, 0.2, 0.3], sessions: sessions)
        #expect(abs(sim[1] - 0.72) < 1e-6 && sim[2] == 0.2)
    }

    @Test func halfRoundTrip() {
        let v: [Float] = [0, 1, -0.5, 0.333, 65504]
        let back = SearchMath.floats(fromHalf: SearchMath.halfData(from: v))
        for (a, b) in zip(v, back) { #expect(abs(a - b) <= abs(a) * 1e-3) }
    }

    @Test func calibratedSigmoid() {
        let p = SearchMath.sigmoid([0, 0.15], scale: 112.8, bias: -16.77)
        #expect(p[0] < 1e-6 && p[1] > 0.5)
    }
}

struct QueryParserTests {
    var cal: Calendar = { var c = Calendar(identifier: .gregorian); c.timeZone = TimeZone(identifier: "UTC")!; return c }()
    var now: Date { cal.date(from: DateComponents(year: 2026, month: 10, day: 15, hour: 12))! }
    func day(_ y: Int, _ m: Int, _ d: Int) -> Date { cal.date(from: DateComponents(year: y, month: m, day: d))! }

    @Test func monthAndYear() {
        let q = QueryParser.parse("beach in july 2023", now: now, calendar: cal)
        #expect(q.text == "beach" && q.after == day(2023, 7, 1) && q.before == day(2023, 8, 1))
    }

    @Test func monthWithoutYearIsMostRecent() {
        let q = QueryParser.parse("snow december", now: now, calendar: cal)
        #expect(q.after == day(2025, 12, 1) && q.text == "snow")
    }

    @Test func bareYearAndRelative() {
        #expect(QueryParser.parse("dogs 2019", now: now, calendar: cal).after == day(2019, 1, 1))
        let lastMonth = QueryParser.parse("receipts last month", now: now, calendar: cal)
        #expect(lastMonth.after == day(2026, 9, 1) && lastMonth.before == day(2026, 10, 1) && lastMonth.text == "receipts")
    }

    @Test func plainQueryHasNoFilters() {
        let q = QueryParser.parse("red vintage car by the sea", now: now, calendar: cal)
        #expect(!q.hasFilters && q.text == "red vintage car by the sea")
    }

    @Test func placeMatching() {
        let q = QueryParser.parse("cafe in lisbon", places: ["Lisbon, Lisboa, Portugal", "Paris, Île-de-France, France"])
        #expect(q.place == "Lisbon")
        #expect(QueryParser.parse("cafe", places: ["Lisbon, Portugal"]).place == nil)
    }
}

struct ExplainTests {
    @Test func wordHits() {
        #expect(ConceptExplainer.wordHits(query: "the wifi password", in: "WiFi: guest\nPassword 1234") == ["wifi", "password"])
        #expect(ConceptExplainer.wordHits(query: "cat", in: "concatenate").isEmpty)
        #expect(ConceptExplainer.wordHits(query: "photos of menus", in: nil).isEmpty)
    }

    @Test func conceptTags() {
        // three orthogonal concepts in 3-d; the photo vector points at "dog"
        let e = ConceptExplainer(concepts: ["dog", "beach", "car"], matrix: [1, 0, 0, 0, 1, 0, 0, 0, 1], dim: 3, scale: 112.8, bias: -16.77)
        #expect(e.tags([1, 0, 0]) == ["dog"])
        #expect(e.shared(withImage: [0.8, 0.6, 0], [0.8, 0.6, 0]).first == "dog")
        #expect(e.shared(withTags: ["beach"], [0, 1, 0]) == ["beach"])
    }
}
