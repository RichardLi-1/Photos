package app.photosearch

import app.photosearch.ml.Half
import app.photosearch.search.DateRange
import app.photosearch.search.Ranking
import app.photosearch.search.SearchRequest
import app.photosearch.search.Searcher
import app.photosearch.search.VectorIndex
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.time.LocalDate
import java.time.ZoneOffset
import kotlin.math.abs
import kotlin.math.sqrt
import kotlin.random.Random

class RankingTest {
    @Test fun topKOrdersAndSkipsInfinite() {
        val s = floatArrayOf(0.1f, 0.9f, Float.NEGATIVE_INFINITY, 0.5f, 0.9f, 0.3f)
        assertArrayEquals(intArrayOf(1, 4, 3), Ranking.topK(s, 3))
        assertArrayEquals(intArrayOf(1, 4, 3, 5, 0), Ranking.topK(s, 10))
        assertEquals(0, Ranking.topK(s, 0).size)
    }

    @Test fun rrfMatchesPrototypeFormula() {
        // search.py: fused[id] = 1/(60+n) for the embedding run, + 0.8/(60+n) for OCR hits
        val out = Ranking.rrf(listOf(Ranking.Run(listOf(10L, 20L, 30L)), Ranking.Run(listOf(30L, 40L), 0.8f))).toMap()
        assertEquals(1f / 60, out[10L]!!, 1e-7f)
        assertEquals(1f / 62 + 0.8f / 60, out[30L]!!, 1e-7f)
        assertEquals(0.8f / 61, out[40L]!!, 1e-7f)
        assertEquals(listOf(30L, 10L, 20L, 40L), Ranking.rrf(listOf(Ranking.Run(listOf(10L, 20L, 30L)), Ranking.Run(listOf(30L, 40L), 0.8f))).map { it.first })
    }

    @Test fun halfRoundTrip() {
        for (x in floatArrayOf(0f, 1f, -1f, 0.1f, 0.0123f, -0.333f, 65504f, 1e-6f)) {
            val y = Half.toFloat(Half.fromFloat(x))
            assertTrue("$x -> $y", abs(x - y) <= maxOf(abs(x) * 1e-3f, 1e-7f))
        }
    }

    private fun unit(v: FloatArray): FloatArray { val n = sqrt(v.sumOf { (it * it).toDouble() }).toFloat(); return FloatArray(v.size) { v[it] / n } }

    @Test fun cosineSearchFusionAndDates() {
        val dim = 16
        val rnd = Random(7)
        val idx = VectorIndex(dim)
        val vecs = (1L..50L).associateWith { unit(FloatArray(dim) { rnd.nextFloat() - 0.5f }) }
        vecs.forEach { (id, v) -> idx.put(id, v) }
        val day = { s: String -> LocalDate.parse(s).atStartOfDay(ZoneOffset.UTC).toInstant().toEpochMilli() }
        val taken = (1L..50L).associateWith { if (it <= 25) day("2019-07-01") else day("2023-02-01") }
        val target = vecs.getValue(7L)
        val s = Searcher(idx, embedText = { target.copyOf() }, takenMs = { taken }, ocrHits = { listOf(42L) },
            zone = ZoneOffset.UTC, today = { LocalDate.of(2026, 10, 15) })

        // fusion as in search.py: the OCR hit (also a visual candidate) gets 1/(60+n) + 0.8/60 and outranks the best pure
        // cosine match (1/60); the photo whose vector equals the query embedding is next, with cosine 1 (fp16 storage)
        val r = s.search(SearchRequest(text = "anything", k = 10))
        assertEquals(listOf(42L, 7L), r.hits.take(2).map { it.id })
        assertEquals(1f, r.hits[1].score!!, 2e-3f)
        val noOcr = Searcher(idx, embedText = { target.copyOf() }, takenMs = { taken }, ocrHits = { emptyList() }, zone = ZoneOffset.UTC)
        assertEquals(7L, noOcr.search(SearchRequest(text = "anything", k = 10)).hits.first().id)

        // brute-force scores equal plain dot products
        val sc = idx.scores(target)
        for (row in 0 until idx.size) assertEquals(Ranking.dot(idx.vector(idx.idAt(row))!!, target), sc[row], 1e-5f)

        // date words become a filter and are removed from the text the model sees
        val r2 = s.search(SearchRequest(text = "anything in 2023", k = 10))
        assertEquals(DateRange(LocalDate.of(2023, 1, 1), LocalDate.of(2024, 1, 1)), r2.query!!.range)
        assertEquals("anything", r2.query!!.text)
        assertTrue(r2.hits.all { it.id > 25 })
        assertTrue(r2.hits.none { it.id == 7L })

        // more like this: never returns the photo itself
        val r3 = s.search(SearchRequest(similarTo = 7L, k = 5))
        assertTrue(r3.hits.none { it.id == 7L })
        assertEquals(5, r3.hits.size)

        // date-only query: everything in range, newest first, no embedding needed
        val r4 = s.search(SearchRequest(text = "summer 2019", k = 100))
        assertEquals(25, r4.hits.size)
    }

    @Test fun indexRemoveKeepsRowsConsistent() {
        val idx = VectorIndex(2)
        for (i in 1L..5L) idx.put(i, floatArrayOf(i.toFloat(), 0f))
        idx.remove { it % 2 == 1L }
        assertEquals(3, idx.size)
        assertEquals(listOf(1L, 3L, 5L), idx.ids().toList())
        assertEquals(3f, idx.vector(3L)!![0], 0f)
        assertEquals(1, idx.rowOf(3L))
    }
}
