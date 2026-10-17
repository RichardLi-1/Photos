package app.photosearch.ui

import android.content.ContentResolver
import android.graphics.Bitmap
import android.util.LruCache
import android.util.Size
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.material3.MaterialTheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.produceState
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import app.photosearch.data.MediaScanner
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.withContext

/** MediaStore thumbnails (ContentResolver.loadThumbnail: the system's cached ones) behind a memory LRU. No image library. */
class Thumbnails(private val resolver: ContentResolver, private val scanner: MediaScanner) {
    private val cache = object : LruCache<Long, Bitmap>((Runtime.getRuntime().maxMemory() / 8).toInt()) {
        override fun sizeOf(key: Long, value: Bitmap) = value.allocationByteCount
    }

    @OptIn(ExperimentalCoroutinesApi::class)
    private val io = Dispatchers.IO.limitedParallelism(4)

    fun cached(id: Long): Bitmap? = cache.get(id)

    suspend fun load(id: Long, px: Int = 256): Bitmap? = cache.get(id) ?: withContext(io) {
        runCatching { resolver.loadThumbnail(scanner.uri(id), Size(px, px), null) }.getOrNull()?.also { cache.put(id, it) }
    }
}

@Composable
fun Thumb(id: Long, thumbs: Thumbnails, modifier: Modifier = Modifier) {
    val bmp: ImageBitmap? by produceState(thumbs.cached(id)?.asImageBitmap(), id) { value = thumbs.load(id)?.asImageBitmap() }
    Box(modifier.background(MaterialTheme.colorScheme.surfaceVariant)) {
        bmp?.let { Image(it, contentDescription = null, contentScale = ContentScale.Crop, modifier = Modifier.matchParentSize()) }
    }
}
