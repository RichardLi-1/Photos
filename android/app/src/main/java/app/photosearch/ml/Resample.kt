package app.photosearch.ml

import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

/**
 * PIL-compatible antialiased bicubic resize (Image.resize(..., BICUBIC)), the filter open_clip's preprocessing uses.
 * Separable: horizontal pass, round/clip to 8 bits (as PIL does), then vertical pass. Pure Kotlin, unit-tested.
 */
object Resample {
    private fun cubic(x: Double): Double {
        val a = -0.5
        val t = abs(x)
        return when {
            t < 1.0 -> ((a + 2.0) * t - (a + 3.0)) * t * t + 1.0
            t < 2.0 -> (((t - 5.0) * t + 8.0) * t - 4.0) * a
            else -> 0.0
        }
    }

    private class Coeffs(val start: IntArray, val weights: Array<DoubleArray>)

    private fun coeffs(inSize: Int, outSize: Int): Coeffs {
        val scale = inSize.toDouble() / outSize
        val filterScale = max(scale, 1.0)
        val support = 2.0 * filterScale
        val start = IntArray(outSize)
        val weights = Array(outSize) { i ->
            val center = (i + 0.5) * scale
            val lo = max(0, (center - support + 0.5).toInt())
            val hi = min(inSize, (center + support + 0.5).toInt())
            start[i] = lo
            val w = DoubleArray(hi - lo) { j -> cubic((j + lo - center + 0.5) / filterScale) }
            val sum = w.sum()
            if (sum != 0.0) for (j in w.indices) w[j] /= sum
            w
        }
        return Coeffs(start, weights)
    }

    private fun clip8(v: Double): Int = kotlin.math.floor(v + 0.5).toInt().coerceIn(0, 255)

    /** ARGB pixels [w*h] -> RGB-packed ARGB pixels [ow*oh] (alpha dropped). */
    fun bicubic(px: IntArray, w: Int, h: Int, ow: Int, oh: Int): IntArray {
        require(px.size == w * h)
        val cx = coeffs(w, ow)
        val tmp = IntArray(ow * h) // horizontal pass
        for (y in 0 until h) {
            val row = y * w
            for (x in 0 until ow) {
                var r = 0.0; var g = 0.0; var b = 0.0
                val s = cx.start[x]; val k = cx.weights[x]
                for (j in k.indices) {
                    val c = px[row + s + j]; val f = k[j]
                    r += ((c shr 16) and 0xff) * f; g += ((c shr 8) and 0xff) * f; b += (c and 0xff) * f
                }
                tmp[y * ow + x] = (clip8(r) shl 16) or (clip8(g) shl 8) or clip8(b)
            }
        }
        val cy = coeffs(h, oh)
        val out = IntArray(ow * oh) // vertical pass
        for (y in 0 until oh) {
            val s = cy.start[y]; val k = cy.weights[y]
            for (x in 0 until ow) {
                var r = 0.0; var g = 0.0; var b = 0.0
                for (j in k.indices) {
                    val c = tmp[(s + j) * ow + x]; val f = k[j]
                    r += ((c shr 16) and 0xff) * f; g += ((c shr 8) and 0xff) * f; b += (c and 0xff) * f
                }
                out[y * ow + x] = (0xff shl 24) or (clip8(r) shl 16) or (clip8(g) shl 8) or clip8(b)
            }
        }
        return out
    }
}
