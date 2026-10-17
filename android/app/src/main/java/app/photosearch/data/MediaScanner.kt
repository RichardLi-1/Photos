package app.photosearch.data

import android.content.ContentResolver
import android.content.ContentUris
import android.net.Uri
import android.provider.MediaStore

data class MediaPhoto(
    val id: Long, val takenMs: Long?, val modified: Long, val width: Int, val height: Int, val mime: String?, val bucket: String?,
)

/**
 * Reads the photo library through MediaStore. With partial access (Android 14 "selected photos"), MediaStore simply
 * returns the photos the user picked, so the same queries work for full and limited access.
 */
class MediaScanner(private val resolver: ContentResolver) {
    private val collection: Uri = MediaStore.Images.Media.getContentUri(MediaStore.VOLUME_EXTERNAL)

    fun uri(id: Long): Uri = ContentUris.withAppendedId(collection, id)

    /** Every visible photo, newest first (DATE_TAKEN, falling back to DATE_ADDED). */
    fun all(): List<MediaPhoto> {
        val cols = arrayOf(
            MediaStore.Images.Media._ID, MediaStore.Images.Media.DATE_TAKEN, MediaStore.Images.Media.DATE_ADDED,
            MediaStore.Images.Media.DATE_MODIFIED, MediaStore.Images.Media.WIDTH, MediaStore.Images.Media.HEIGHT,
            MediaStore.Images.Media.MIME_TYPE, MediaStore.Images.Media.BUCKET_DISPLAY_NAME,
        )
        val out = ArrayList<MediaPhoto>()
        resolver.query(collection, cols, null, null, "${MediaStore.Images.Media.DATE_TAKEN} DESC, ${MediaStore.Images.Media.DATE_ADDED} DESC")?.use { c ->
            while (c.moveToNext()) {
                val taken = if (c.isNull(1) || c.getLong(1) <= 0) c.getLong(2) * 1000 else c.getLong(1)
                out += MediaPhoto(c.getLong(0), taken, c.getLong(3), c.getInt(4), c.getInt(5), c.getString(6), c.getString(7))
            }
        }
        return out
    }

    /** Diff MediaStore against the index: insert new/changed photos, drop deleted or no-longer-shared ones. */
    fun sync(db: PhotoDb): SyncResult {
        val now = all()
        val known = db.knownIds()
        val changed = now.filter { known[it.id] != it.modified }
        val present = now.mapTo(HashSet(now.size)) { it.id }
        val gone = known.keys.filter { it !in present }
        if (changed.isNotEmpty()) db.upsert(changed)
        if (gone.isNotEmpty()) db.delete(gone)
        return SyncResult(now.map { it.id }, changed.map { it.id }, gone)
    }
}

data class SyncResult(val ordered: List<Long>, val changed: List<Long>, val removed: List<Long>)
