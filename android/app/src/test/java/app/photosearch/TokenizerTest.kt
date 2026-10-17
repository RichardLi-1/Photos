package app.photosearch

import app.photosearch.ml.GemmaTokenizer
import app.photosearch.ml.TextClean
import org.json.JSONArray
import org.junit.Assert.assertEquals
import org.junit.Assume.assumeTrue
import org.junit.BeforeClass
import org.junit.Test
import java.io.File
import java.nio.ByteBuffer

/** Kotlin tokenizer vs open_clip's own SigLIP2 tokenizer (fixtures from ml/android/export_tokenizer.py). */
class TokenizerTest {
    companion object {
        private val bin = File("src/main/assets/models/siglip2_tokenizer.bin")
        private lateinit var tok: GemmaTokenizer
        private lateinit var fixtures: JSONArray

        @BeforeClass @JvmStatic fun load() {
            fixtures = JSONArray(TokenizerTest::class.java.getResource("/tokenizer_fixtures.json")!!.readText())
            if (bin.exists()) tok = GemmaTokenizer(ByteBuffer.wrap(bin.readBytes()))
        }
    }

    @Test fun cleanMatchesOpenClip() {
        for (i in 0 until fixtures.length()) {
            val f = fixtures.getJSONObject(i)
            assertEquals("clean(${f.getString("text")})", f.getString("clean"), TextClean.canonicalize(f.getString("text")))
        }
    }

    @Test fun idsMatchOpenClip() {
        assumeTrue("run ml/android/export_tokenizer.py first", bin.exists())
        assertEquals(64, tok.contextLength)
        var n = 0
        for (i in 0 until fixtures.length()) {
            val f = fixtures.getJSONObject(i)
            val want = f.getJSONArray("ids").let { a -> IntArray(a.length()) { a.getInt(it) } }
            assertEquals("ids(${f.getString("text")})", want.toList(), tok.encode(f.getString("text")).toList())
            n++
        }
        assert(n >= 10)
    }
}
