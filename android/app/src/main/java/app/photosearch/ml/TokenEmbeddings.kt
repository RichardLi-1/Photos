package app.photosearch.ml

import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * The 256k x 768 token-embedding table as int8 rows + fp32 row scales ('SGE8', see ml/android/convert_siglip2_onnx.py).
 * Kept out of the ONNX graph so ONNX Runtime never dequantizes the whole ~800 MB table; a query reads only 64 rows
 * from the memory-mapped file.
 */
class TokenEmbeddings(buf: ByteBuffer) {
    private val b = buf.duplicate().order(ByteOrder.LITTLE_ENDIAN)
    val rows: Int
    val dim: Int
    private val scalesAt = 12
    private val dataAt: Int

    init {
        val magic = ByteArray(4) { b.get(it) } // absolute reads only (bulk absolute get is API 34+)
        require(String(magic, Charsets.US_ASCII) == "SGE8") { "not a token-embedding table" }
        rows = b.getInt(4); dim = b.getInt(8)
        dataAt = scalesAt + 4 * rows
    }

    /** [ids.size * dim] floats, row-major: what the text model's `token_embeds` input expects. */
    fun lookup(ids: IntArray): FloatArray {
        val out = FloatArray(ids.size * dim)
        for ((r, id) in ids.withIndex()) {
            require(id in 0 until rows) { "token id $id out of range" }
            val s = b.getFloat(scalesAt + 4 * id)
            val base = dataAt + id * dim
            val o = r * dim
            for (j in 0 until dim) out[o + j] = b.get(base + j) * s
        }
        return out
    }
}
