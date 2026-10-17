package app.photosearch.data

import app.photosearch.ml.Half
import app.photosearch.search.VectorIndex
import java.io.DataOutputStream
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel

/**
 * Embeddings on disk: 'PSV1' u32 dim, then append-only records [i64 photoId | dim x fp16], little endian.
 * Appends are fsync'd per batch; a torn last record (crash mid-write) is ignored on load, and a photo appended twice
 * keeps its latest vector, so indexing can resume anywhere. [compact] rewrites the file without deleted photos.
 */
class VectorFile(private val file: File, val dim: Int) {
    private val recordBytes = 8 + 2 * dim

    @Synchronized fun load(into: VectorIndex) {
        if (!file.exists()) return
        FileChannel.open(file.toPath()).use { ch ->
            val buf = ch.map(FileChannel.MapMode.READ_ONLY, 0, ch.size()).order(ByteOrder.LITTLE_ENDIAN)
            if (buf.limit() < 8 || buf.getInt(0) != MAGIC || buf.getInt(4) != dim) return
            val n = (buf.limit() - 8) / recordBytes
            val v = FloatArray(dim)
            for (r in 0 until n) {
                val o = 8 + r * recordBytes
                val id = buf.getLong(o)
                for (j in 0 until dim) v[j] = Half.toFloat(buf.getShort(o + 8 + 2 * j))
                into.put(id, v)
            }
        }
    }

    @Synchronized fun append(batch: List<Pair<Long, FloatArray>>) {
        if (batch.isEmpty()) return
        val fresh = !file.exists() || file.length() < 8
        if (!fresh) truncateTornTail()
        FileOutputStream(file, !fresh).use { fos ->
            val bb = ByteBuffer.allocate((if (fresh) 8 else 0) + batch.size * recordBytes).order(ByteOrder.LITTLE_ENDIAN)
            if (fresh) bb.putInt(MAGIC).putInt(dim)
            for ((id, v) in batch) {
                require(v.size == dim)
                bb.putLong(id)
                for (x in v) bb.putShort(Half.fromFloat(x))
            }
            fos.write(bb.array())
            fos.fd.sync()
        }
    }

    /** Rewrite with only the rows in [index] (after deletions). Atomic via rename. */
    @Synchronized fun compact(index: VectorIndex) {
        val tmp = File(file.parentFile, file.name + ".tmp")
        DataOutputStream(FileOutputStream(tmp).buffered()).use { out ->
            val head = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN).putInt(MAGIC).putInt(dim)
            out.write(head.array())
            val rec = ByteBuffer.allocate(recordBytes).order(ByteOrder.LITTLE_ENDIAN)
            for (id in index.ids()) {
                val v = index.vector(id) ?: continue
                rec.clear(); rec.putLong(id); for (x in v) rec.putShort(Half.fromFloat(x))
                out.write(rec.array())
            }
        }
        tmp.renameTo(file)
    }

    fun recordCount(): Long = if (file.exists()) maxOf(0, (file.length() - 8) / recordBytes) else 0

    private fun truncateTornTail() {
        val good = 8 + recordCount() * recordBytes
        if (file.length() != good) FileOutputStream(file, true).channel.use { it.truncate(good) }
    }

    companion object { const val MAGIC = 0x31565350 } // "PSV1"
}
