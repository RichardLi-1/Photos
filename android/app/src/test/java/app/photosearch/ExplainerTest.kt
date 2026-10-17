package app.photosearch

import app.photosearch.search.Explainer
import org.junit.Assert.assertEquals
import org.junit.Test

class ExplainerTest {
    @Test fun wordHits() {
        assertEquals(listOf("wifi", "password"), Explainer.wordHits("wifi password on the whiteboard", "Guest WiFi\nPassword: hunter2"))
        assertEquals(emptyList<String>(), Explainer.wordHits("photos of the menu", null))
    }

    @Test fun sigmoidHead() {
        // two concepts in 2-d; scale/bias like SigLIP's calibrated head
        val e = Explainer(listOf("dog", "cat"), floatArrayOf(1f, 0f, 0f, 1f), 2, 112.8f, -16.8f)
        val p = e.p(floatArrayOf(0.2f, 0.05f))
        assertEquals(Explainer.sigmoid(112.8f * 0.2f - 16.8f), p[0], 1e-6f)
        assertEquals(listOf("dog"), e.tags(floatArrayOf(0.2f, 0.05f)))
    }
}
