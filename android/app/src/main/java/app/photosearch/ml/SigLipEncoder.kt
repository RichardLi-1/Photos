package app.photosearch.ml

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.graphics.Bitmap
import java.nio.FloatBuffer
import kotlin.math.sqrt

/**
 * SigLIP2-B/16-384 image + text towers on ONNX Runtime (CPU, int8 weights). Both return unit-length 768-d vectors in
 * the same space, so cosine = dot product. Sessions load lazily: search needs only the text tower, indexing only the
 * image tower.
 */
class SigLipEncoder(private val files: ModelFiles) {
    private val env = OrtEnvironment.getEnvironment()
    private val lock = Any()
    private var imageSession: OrtSession? = null
    private var textSession: OrtSession? = null
    private var tokenizer: GemmaTokenizer? = null
    private var embeds: TokenEmbeddings? = null

    val modelsAvailable: Boolean
        get() = listOf(ModelFiles.IMAGE, ModelFiles.TEXT, ModelFiles.TOKEN_EMBED, ModelFiles.TOKENIZER).all(files::available)

    private fun options(threads: Int) = OrtSession.SessionOptions().apply {
        setIntraOpNumThreads(threads)
        setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
    }

    private fun image(): OrtSession = synchronized(lock) {
        imageSession ?: env.createSession(files.map(ModelFiles.IMAGE), options(2)).also { imageSession = it }
    }

    private fun text(): OrtSession = synchronized(lock) {
        textSession ?: env.createSession(files.map(ModelFiles.TEXT), options(4)).also { textSession = it }
    }

    fun tokenizer(): GemmaTokenizer = synchronized(lock) {
        tokenizer ?: GemmaTokenizer(files.map(ModelFiles.TOKENIZER)).also { tokenizer = it }
    }

    private fun tokenEmbeds(): TokenEmbeddings = synchronized(lock) {
        embeds ?: TokenEmbeddings(files.map(ModelFiles.TOKEN_EMBED)).also { embeds = it }
    }

    fun embedText(query: String): FloatArray {
        val tok = tokenizer()
        val ids = tok.encode(query)
        val x = tokenEmbeds().lookup(ids)
        val shape = longArrayOf(1, tok.contextLength.toLong(), (x.size / tok.contextLength).toLong())
        return run(text(), "token_embeds", x, shape)
    }

    /** Any size; squashed to 384x384 with bicubic (see ImagePreprocess). Decode with ImagePreprocess.decodeSampled. */
    fun embedImage(bitmap: Bitmap): FloatArray {
        val x = ImagePreprocess.toTensor(bitmap)
        return run(image(), "pixels", x, longArrayOf(1, 3, ImagePreprocess.SIZE.toLong(), ImagePreprocess.SIZE.toLong()))
    }

    private fun run(s: OrtSession, input: String, x: FloatArray, shape: LongArray): FloatArray =
        OnnxTensor.createTensor(env, FloatBuffer.wrap(x), shape).use { t ->
            s.run(mapOf(input to t)).use { r ->
                @Suppress("UNCHECKED_CAST")
                val out = (r[0].value as Array<FloatArray>)[0]
                normalize(out)
            }
        }

    /** Free the image tower after an indexing run (~100 MB native memory). */
    fun releaseImage() = synchronized(lock) { imageSession?.close(); imageSession = null }

    companion object {
        fun normalize(v: FloatArray): FloatArray {
            var s = 0.0
            for (x in v) s += x * x
            val n = sqrt(s).toFloat().takeIf { it > 0 } ?: return v
            for (i in v.indices) v[i] /= n
            return v
        }
    }
}
