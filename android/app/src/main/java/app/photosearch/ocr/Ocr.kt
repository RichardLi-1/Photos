package app.photosearch.ocr

import android.graphics.Bitmap
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.TextRecognizer
import com.google.mlkit.vision.text.latin.TextRecognizerOptions
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException

interface OcrEngine : AutoCloseable {
    suspend fun read(bitmap: Bitmap): String
}

/**
 * ML Kit text recognition, *bundled* model (runs on device, no model download, no Play Services needed).
 *
 * Privacy: ML Kit logs usage/performance metrics to Google (developers.google.com/ml-kit/android-data-disclosure)
 * with no documented opt-out. So OCR is OFF by default and the user opts in (Settings). Even then nothing can
 * leave the device: this app has no INTERNET permission (stripped from the merged manifest, checked at build time
 * by verifyNoNetwork*), so the OS blocks the attempt. Recognized text never leaves the app either way.
 */
class MlKitOcr : OcrEngine {
    private val recognizer: TextRecognizer = TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS)

    override suspend fun read(bitmap: Bitmap): String = suspendCancellableCoroutine { cont ->
        recognizer.process(InputImage.fromBitmap(bitmap, 0))
            .addOnSuccessListener { r ->
                cont.resume(r.textBlocks.joinToString("\n") { b -> b.lines.joinToString("\n") { it.text } })
            }
            .addOnFailureListener { cont.resumeWithException(it) }
    }

    override fun close() = recognizer.close()
}
