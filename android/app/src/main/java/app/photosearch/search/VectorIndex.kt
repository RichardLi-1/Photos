package app.photosearch.search

import app.photosearch.ml.Half

/**
 * All photo embeddings in memory as fp16 (1.5 KB per photo: 10k photos = 15 MB) for brute-force cosine search.
 * Vectors are unit length, so cosine = dot. Thread-safe: the indexer appends while the UI searches.
 */
class VectorIndex(val dim: Int) {
    private var ids = LongArray(1024)
    private var data = ShortArray(1024 * dim)
    private val rowOf = HashMap<Long, Int>()
    @Volatile var size = 0
        private set

    @Synchronized fun put(id: Long, v: FloatArray) {
        require(v.size == dim)
        val row = rowOf[id] ?: run {
            if (size == ids.size) { ids = ids.copyOf(size * 2); data = data.copyOf(size * 2 * dim) }
            size.also { rowOf[id] = it; ids[it] = id; size++ }
        }
        val o = row * dim
        for (j in 0 until dim) data[o + j] = Half.fromFloat(v[j])
    }

    @Synchronized fun remove(keep: (Long) -> Boolean) {
        var w = 0
        for (r in 0 until size) {
            if (!keep(ids[r])) continue
            if (w != r) { ids[w] = ids[r]; System.arraycopy(data, r * dim, data, w * dim, dim) }
            w++
        }
        size = w
        rowOf.clear()
        for (r in 0 until size) rowOf[ids[r]] = r
    }

    @Synchronized fun contains(id: Long) = rowOf.containsKey(id)
    @Synchronized fun idAt(row: Int) = ids[row]
    @Synchronized fun rowOf(id: Long): Int? = rowOf[id]
    @Synchronized fun ids(): LongArray = ids.copyOf(size)

    @Synchronized fun vector(id: Long): FloatArray? {
        val r = rowOf[id] ?: return null
        return FloatArray(dim) { Half.toFloat(data[r * dim + it]) }
    }

    /** Dot product of [q] with every row, in row order. */
    @Synchronized fun scores(q: FloatArray): FloatArray {
        require(q.size == dim)
        val out = FloatArray(size)
        for (r in 0 until size) {
            var s = 0f
            val o = r * dim
            for (j in 0 until dim) s += Half.toFloat(data[o + j]) * q[j]
            out[r] = s
        }
        return out
    }
}
