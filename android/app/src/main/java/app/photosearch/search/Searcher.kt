package app.photosearch.search

import java.time.LocalDate
import java.time.ZoneId

data class SearchRequest(
    val text: String? = null,
    val imageVector: FloatArray? = null, // a picked photo (photo picker), already embedded
    val similarTo: Long? = null,         // "more like this": a library photo id
    val textWeight: Float = 0.5f,        // when text + image: how much the text steers
    val range: DateRange? = null,        // explicit filter; otherwise parsed from the text
    val k: Int = 200,
)

data class SearchHit(val id: Long, val score: Float?, val fused: Float)

data class SearchResult(val hits: List<SearchHit>, val query: ParsedQuery?, val queryVector: FloatArray?)

/**
 * Hybrid retrieval, ported from search/photosearch/search.py: embedding cosine (text, image or both) fused by
 * reciprocal rank with OCR keyword hits, inside an optional date range parsed from the query.
 * Dependencies are plain functions so the ranking runs in JVM unit tests without models or a database.
 */
class Searcher(
    private val index: VectorIndex,
    private val embedText: (String) -> FloatArray,
    private val takenMs: () -> Map<Long, Long?>,
    private val ocrHits: (String) -> List<Long>,
    private val zone: ZoneId = ZoneId.systemDefault(),
    private val today: () -> LocalDate = { LocalDate.now() },
) {
    fun search(req: SearchRequest): SearchResult {
        val parsed = req.text?.takeIf { it.isNotBlank() }?.let { DateParser.parse(it, today()) }
        val text = parsed?.text?.takeIf { it.isNotBlank() }
        val range = req.range ?: parsed?.range

        val vecs = ArrayList<FloatArray>()
        val imageVec = req.imageVector ?: req.similarTo?.let(index::vector)
        if (imageVec != null) vecs += imageVec
        if (text != null) {
            val t = embedText(text)
            vecs += if (imageVec != null) t.map { it * (req.textWeight / (1 - req.textWeight)) }.toFloatArray() else t
        }
        val allowed = range?.let { allowedIds(it) }
        if (vecs.isEmpty()) { // date-only query ("last summer"): everything in range, newest first
            if (allowed == null) return SearchResult(emptyList(), parsed, null)
            val times = takenMs()
            val ids = allowed.sortedByDescending { times[it] ?: 0L }.take(req.k)
            return SearchResult(ids.map { SearchHit(it, null, 0f) }, parsed, null)
        }
        val q = FloatArray(index.dim).also { s -> vecs.forEach { v -> for (i in s.indices) s[i] += v[i] } }
        normalize(q)

        val sim = index.scores(q)
        for (r in sim.indices) {
            val id = index.idAt(r)
            if ((allowed != null && id !in allowed) || id == req.similarTo) sim[r] = Float.NEGATIVE_INFINITY
        }
        val order = Ranking.topK(sim, maxOf(req.k * 4, 200))
        val runs = mutableListOf(Ranking.Run(order.map { index.idAt(it) }))
        if (text != null) { // literal matches (text in the photo) get a vote too
            val hits = ocrHits(text).filter { (allowed == null || it in allowed) && it != req.similarTo }
            if (hits.isNotEmpty()) runs += Ranking.Run(hits, 0.8f)
        }
        val fused = Ranking.rrf(runs).take(req.k)
        val hits = fused.map { (id, f) -> SearchHit(id, index.rowOf(id)?.let { sim[it] }?.takeIf { it.isFinite() }, f) }
        return SearchResult(hits, parsed, q)
    }

    private fun allowedIds(r: DateRange): Set<Long> {
        val lo = r.start?.atStartOfDay(zone)?.toInstant()?.toEpochMilli() ?: Long.MIN_VALUE
        val hi = r.end?.atStartOfDay(zone)?.toInstant()?.toEpochMilli() ?: Long.MAX_VALUE
        return takenMs().filter { (_, t) -> t != null && t >= lo && t < hi }.keys
    }

    private fun normalize(v: FloatArray) {
        var s = 0.0
        for (x in v) s += x * x
        val n = kotlin.math.sqrt(s).toFloat()
        if (n > 0) for (i in v.indices) v[i] /= n
    }
}
