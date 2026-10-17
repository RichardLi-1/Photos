package app.photosearch.search

import app.photosearch.ml.Half
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.exp
import kotlin.math.sqrt

/**
 * "Why it matched", ported from search/photosearch/explain.py. Uses SigLIP's sigmoid head, so each concept score is a
 * calibrated P(photo shows concept) = sigmoid(scale * cos + bias). Concept text embeddings are precomputed (asset).
 */
class Explainer(val concepts: List<String>, private val t: FloatArray, val dim: Int, private val scale: Float, private val bias: Float) {
    init { require(t.size == concepts.size * dim) }

    fun p(v: FloatArray): FloatArray = FloatArray(concepts.size) { i -> sigmoid(scale * dot(i, v) + bias) }

    private fun dot(i: Int, v: FloatArray): Float {
        var s = 0f
        val o = i * dim
        for (j in 0 until dim) s += t[o + j] * v[j]
        return s
    }

    /** Top-k above an absolute floor and within 15x of the best (SigLIP probabilities are small but well ranked). */
    private fun top(score: FloatArray, k: Int, floor: Float): List<String> {
        val order = score.indices.sortedByDescending { score[it] }.take(k)
        val best = order.firstOrNull()?.let { score[it] } ?: 0f
        return order.filter { score[it] >= floor && score[it] >= best / 15 }.map { concepts[it] }
    }

    fun tags(v: FloatArray, k: Int = 4, floor: Float = 0.002f) = top(p(v), k, floor)

    fun sharedWithImage(ref: FloatArray, v: FloatArray, k: Int = 4): List<String> {
        val a = p(ref); val b = p(v)
        return top(FloatArray(a.size) { minOf(a[it], b[it]) }, k, 0.001f)
    }

    /** Concepts close in meaning to the query (z-score > 2 over the vocabulary) that the photo clearly shows. */
    fun matchedText(query: FloatArray, v: FloatArray, k: Int = 3): List<String> {
        val rel = FloatArray(concepts.size) { dot(it, query) }
        val mean = rel.average().toFloat()
        val std = sqrt(rel.fold(0.0) { a, x -> a + (x - mean) * (x - mean) } / rel.size).toFloat()
        val pv = p(v)
        return top(FloatArray(rel.size) { if ((rel[it] - mean) / (std + 1e-6f) > 2f) pv[it] else -1f }, k, 0.001f)
    }

    companion object {
        fun sigmoid(x: Float) = (1.0 / (1.0 + exp(-x.toDouble()))).toFloat()

        fun load(conceptsJson: String, vectors: ByteBuffer, calibrationJson: String): Explainer {
            val cj = org.json.JSONObject(conceptsJson)
            val arr = cj.getJSONArray("concepts")
            val names = List(arr.length()) { arr.getString(it) }
            val dim = cj.getInt("dim")
            val b = vectors.duplicate().order(ByteOrder.LITTLE_ENDIAN)
            val t = FloatArray(names.size * dim) { Half.toFloat(b.getShort(it * 2)) }
            val cal = org.json.JSONObject(calibrationJson)
            return Explainer(names, t, dim, cal.getDouble("logit_scale").toFloat(), cal.getDouble("logit_bias").toFloat())
        }

        private val STOP = ("the and for with from that this are was were you your our photo photos picture pictures image " +
            "images show find some any all into onto over under near").split(" ").toSet()

        /** Query words (3+ chars, not stop words) that literally appear in [text] (OCR), at a word start. */
        fun wordHits(q: String, text: String?): List<String> {
            if (text.isNullOrBlank()) return emptyList()
            val low = text.lowercase()
            return Regex("\\w{3,}").findAll(q.lowercase()).map { it.value }.distinct()
                .filter { it !in STOP && Regex("\\b" + Regex.escape(it)).containsMatchIn(low) }.toList()
        }
    }
}
