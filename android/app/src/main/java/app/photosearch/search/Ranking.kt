package app.photosearch.search

import java.util.PriorityQueue

object Ranking {
    const val RRF_K = 60

    /** Indices of the [k] largest finite scores, best first (ties: lower index first). */
    fun topK(scores: FloatArray, k: Int): IntArray {
        if (k <= 0) return IntArray(0)
        val heap = PriorityQueue<Int>(k + 1, compareBy<Int> { scores[it] }.thenByDescending { it })
        for (i in scores.indices) {
            val s = scores[i]
            if (!s.isFinite()) continue
            if (heap.size < k) heap.add(i)
            else {
                val worst = heap.peek()!!
                if (s > scores[worst] || (s == scores[worst] && i < worst)) { heap.poll(); heap.add(i) }
            }
        }
        val out = IntArray(heap.size)
        for (n in out.indices.reversed()) out[n] = heap.poll()!!
        return out
    }

    /** One ranked run with a vote weight. */
    class Run(val ids: List<Long>, val weight: Float = 1f)

    /** Reciprocal-rank fusion: score(id) = sum over runs of weight / (RRF_K + rank). Best first, stable on ties. */
    fun rrf(runs: List<Run>, k: Int = RRF_K): List<Pair<Long, Float>> {
        val score = LinkedHashMap<Long, Float>()
        for (run in runs) for ((n, id) in run.ids.withIndex()) score[id] = (score[id] ?: 0f) + run.weight / (k + n)
        return score.entries.sortedByDescending { it.value }.map { it.key to it.value }
    }

    fun dot(a: FloatArray, b: FloatArray): Float {
        var s = 0f
        for (i in a.indices) s += a[i] * b[i]
        return s
    }
}
