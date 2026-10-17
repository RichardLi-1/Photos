package app.photosearch.ml

import android.content.Context
import java.io.File
import java.io.FileInputStream
import java.io.FileNotFoundException
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

/**
 * Model files are memory-mapped, never copied: straight out of the APK (stored uncompressed, see noCompress in
 * build.gradle.kts), or, for builds made with -PbundleModels=false, from the app-specific external dir where a
 * developer can `adb push` them (Android/data/app.photosearch/files/models/). Both are private to this app.
 */
class ModelFiles(private val context: Context) {
    fun map(name: String): MappedByteBuffer {
        val side = sideloaded(name)
        if (side != null) return FileInputStream(side).use { it.channel.map(FileChannel.MapMode.READ_ONLY, 0, side.length()) }
        val fd = try {
            context.assets.openFd("models/$name")
        } catch (e: FileNotFoundException) {
            throw MissingModelException(name)
        }
        return fd.use { afd ->
            FileInputStream(afd.fileDescriptor).use { it.channel.map(FileChannel.MapMode.READ_ONLY, afd.startOffset, afd.length) }
        }
    }

    fun readText(name: String): String = context.assets.open("models/$name").bufferedReader().use { it.readText() }

    fun available(name: String): Boolean =
        sideloaded(name) != null || runCatching { context.assets.openFd("models/$name").close() }.isSuccess

    private fun sideloaded(name: String): File? =
        context.getExternalFilesDir("models")?.let { File(it, name) }?.takeIf { it.isFile }

    companion object {
        const val IMAGE = "siglip2_image_q8.onnx"
        const val TEXT = "siglip2_text_q8.onnx"
        const val TOKEN_EMBED = "siglip2_token_embed_q8.bin"
        const val TOKENIZER = "siglip2_tokenizer.bin"
        const val CALIBRATION = "calibration.json"
        const val CONCEPTS = "concepts.json"
        const val CONCEPT_VECTORS = "concepts_f16.bin"
    }
}

class MissingModelException(name: String) : IllegalStateException("Model file missing: $name (see android/README.md)")
