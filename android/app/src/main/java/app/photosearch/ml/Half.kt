package app.photosearch.ml

/** IEEE fp16 <-> fp32 without android.util.Half, so it also runs in plain-JVM unit tests. */
object Half {
    private val TABLE = FloatArray(65536) { decode(it.toShort()) }

    fun toFloat(h: Short): Float = TABLE[h.toInt() and 0xffff]

    fun fromFloat(f: Float): Short {
        val bits = java.lang.Float.floatToRawIntBits(f)
        val sign = (bits ushr 16) and 0x8000
        val exp = (bits ushr 23) and 0xff
        var mant = bits and 0x7fffff
        if (exp == 0xff) return (sign or 0x7c00 or (if (mant != 0) 0x200 else 0)).toShort()
        val e = exp - 127 + 15
        if (e >= 0x1f) return (sign or 0x7c00).toShort()
        if (e <= 0) {
            if (e < -10) return sign.toShort()
            mant = mant or 0x800000
            val shift = 14 - e
            var h = mant ushr shift
            if ((mant ushr (shift - 1)) and 1 != 0 && ((mant and ((1 shl (shift - 1)) - 1)) != 0 || h and 1 != 0)) h++
            return (sign or h).toShort()
        }
        var h = (e shl 10) or (mant ushr 13)
        val rest = mant and 0x1fff
        if (rest > 0x1000 || (rest == 0x1000 && h and 1 != 0)) h++ // round half to even; may carry into exponent (ok)
        return (sign or h).toShort()
    }

    private fun decode(h: Short): Float {
        val x = h.toInt() and 0xffff
        val sign = (x and 0x8000) shl 16
        val exp = (x ushr 10) and 0x1f
        val mant = x and 0x3ff
        val bits = when {
            exp == 0 && mant == 0 -> sign
            exp == 0 -> { // subnormal
                var m = mant; var e = -1
                do { e++; m = m shl 1 } while (m and 0x400 == 0)
                sign or ((127 - 15 - e) shl 23) or ((m and 0x3ff) shl 13)
            }
            exp == 0x1f -> sign or 0x7f800000 or (mant shl 13)
            else -> sign or ((exp - 15 + 127) shl 23) or (mant shl 13)
        }
        return java.lang.Float.intBitsToFloat(bits)
    }
}
