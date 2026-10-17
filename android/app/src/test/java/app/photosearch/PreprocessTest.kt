package app.photosearch

import app.photosearch.ml.ImagePreprocess
import app.photosearch.ml.Resample
import org.json.JSONArray
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.abs

class PreprocessTest {
    /** Resample.bicubic vs PIL Image.resize(BICUBIC) (fixture from ml/android/export_resample_fixture.py). */
    @Test fun bicubicMatchesPil() {
        val cases = JSONArray(javaClass.getResource("/resample_fixture.json")!!.readText())
        for (i in 0 until cases.length()) {
            val c = cases.getJSONObject(i)
            val inp = c.getJSONArray("in").let { a -> IntArray(a.length()) { a.getInt(it) } }
            val want = c.getJSONArray("out").let { a -> IntArray(a.length()) { a.getInt(it) } }
            val got = Resample.bicubic(inp, c.getInt("w"), c.getInt("h"), c.getInt("ow"), c.getInt("oh"))
            var worst = 0
            for (k in want.indices) for (sh in intArrayOf(0, 8, 16)) {
                worst = maxOf(worst, abs(((want[k] shr sh) and 0xff) - ((got[k] shr sh) and 0xff)))
            }
            assertTrue("case $i: max channel diff $worst", worst <= 1) // PIL uses fixed-point weights; we use doubles
        }
    }

    @Test fun chwNormalization() {
        val x = ImagePreprocess.pixelsToChw(intArrayOf(0xff000000.toInt(), 0xffffffff.toInt(), 0xff7f8081.toInt()))
        assertEquals(-1f, x[0], 1e-6f); assertEquals(1f, x[1], 1e-6f)          // R plane
        assertEquals(127 / 127.5f - 1f, x[2], 1e-6f)
        assertEquals(128 / 127.5f - 1f, x[3 + 2], 1e-6f)                       // G plane
        assertEquals(129 / 127.5f - 1f, x[6 + 2], 1e-6f)                       // B plane
    }
}
