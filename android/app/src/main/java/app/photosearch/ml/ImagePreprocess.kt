package app.photosearch.ml

import android.content.ContentResolver
import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.net.Uri

/**
 * SigLIP2 preprocessing: squash-resize to 384x384, RGB, x/127.5 - 1, CHW.
 * Python resizes with bicubic from full resolution; here the decoder first subsamples by a power of two (short side
 * stays >= 768, which keeps decode memory small), then a PIL-compatible bicubic finishes (Resample.kt). On 12 MP-sized
 * inputs this stays at cosine >= 0.995 to open_clip's preprocessing (bilinear: 0.967); see ml/android/check_preprocess.py.
 */
object ImagePreprocess {
    const val SIZE = 384

    /** Aspect-preserving decode, subsampled by a power of two while the short side stays >= [minSide]. */
    fun decodeSampled(resolver: ContentResolver, uri: Uri, minSide: Int = 2 * SIZE): Bitmap {
        val src = ImageDecoder.createSource(resolver, uri)
        return ImageDecoder.decodeBitmap(src) { decoder, info, _ ->
            decoder.allocator = ImageDecoder.ALLOCATOR_SOFTWARE // pixels are read back on the CPU
            decoder.isMutableRequired = false
            var sample = 1
            val short = minOf(info.size.width, info.size.height)
            while (short / (sample * 2) >= minSide) sample *= 2
            decoder.setTargetSampleSize(sample)
        } // ImageDecoder applies EXIF orientation itself
    }

    /** Any-size bitmap -> model input: bicubic squash to 384x384, then CHW floats in [-1, 1]. */
    fun toTensor(bitmap: Bitmap): FloatArray {
        val argb = if (bitmap.config == Bitmap.Config.ARGB_8888) bitmap else bitmap.copy(Bitmap.Config.ARGB_8888, false)
        val w = argb.width; val h = argb.height
        val px = IntArray(w * h)
        argb.getPixels(px, 0, w, 0, 0, w, h)
        if (argb !== bitmap) argb.recycle()
        return pixelsToChw(if (w == SIZE && h == SIZE) px else Resample.bicubic(px, w, h, SIZE, SIZE))
    }

    /** ARGB ints -> float CHW in [-1, 1]. Pure function (unit-tested). */
    fun pixelsToChw(px: IntArray): FloatArray {
        val n = px.size
        val out = FloatArray(3 * n)
        for (i in 0 until n) {
            val c = px[i]
            out[i] = ((c shr 16) and 0xff) / 127.5f - 1f
            out[n + i] = ((c shr 8) and 0xff) / 127.5f - 1f
            out[2 * n + i] = (c and 0xff) / 127.5f - 1f
        }
        return out
    }
}
