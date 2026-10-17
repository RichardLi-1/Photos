package app.photosearch.data

import android.content.ContentValues
import android.content.Context
import android.database.sqlite.SQLiteDatabase
import android.database.sqlite.SQLiteOpenHelper

data class PhotoRow(
    val id: Long, val takenMs: Long?, val width: Int, val height: Int, val mime: String?, val bucket: String?,
    val ocr: String?,
)

/**
 * App-private index metadata (no backup: it is derived data). Plain SQLite rather than Room: one small schema, an FTS4
 * table (framework SQLite has FTS4 everywhere; FTS5 is not guaranteed), and no annotation processor in the build.
 *
 * photos.id is the MediaStore _ID. `embedded` / `ocr_done` make indexing resumable photo by photo.
 * Albums (create_album) are app-private collections: Android has no shared user-album API like iOS PhotoKit.
 */
class PhotoDb(context: Context) : SQLiteOpenHelper(context.applicationContext, "index.db", null, 1) {

    override fun onConfigure(db: SQLiteDatabase) {
        db.enableWriteAheadLogging() // the indexer writes while the UI reads
    }

    override fun onCreate(db: SQLiteDatabase) {
        db.execSQL("""CREATE TABLE photos (id INTEGER PRIMARY KEY, taken INTEGER, modified INTEGER, width INTEGER,
            height INTEGER, mime TEXT, bucket TEXT, embedded INTEGER NOT NULL DEFAULT 0, ocr TEXT,
            ocr_done INTEGER NOT NULL DEFAULT 0, failed INTEGER NOT NULL DEFAULT 0)""")
        db.execSQL("CREATE INDEX photos_taken ON photos(taken)")
        db.execSQL("CREATE VIRTUAL TABLE ocr_fts USING fts4(ocr, tokenize=unicode61)") // docid = photos.id
        db.execSQL("CREATE TABLE albums (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT UNIQUE NOT NULL COLLATE NOCASE, created INTEGER)")
        db.execSQL("""CREATE TABLE album_photos (album INTEGER NOT NULL REFERENCES albums(id) ON DELETE CASCADE,
            photo INTEGER NOT NULL, PRIMARY KEY (album, photo))""")
    }

    override fun onUpgrade(db: SQLiteDatabase, oldVersion: Int, newVersion: Int) {}

    // ---------- sync with MediaStore ----------
    fun knownIds(): Map<Long, Long> = readableDatabase.rawQuery("SELECT id, modified FROM photos", null).use { c ->
        HashMap<Long, Long>(c.count).apply { while (c.moveToNext()) put(c.getLong(0), c.getLong(1)) }
    }

    fun upsert(rows: List<MediaPhoto>) = tx {
        for (p in rows) {
            val v = ContentValues().apply {
                put("id", p.id); put("taken", p.takenMs); put("modified", p.modified); put("width", p.width)
                put("height", p.height); put("mime", p.mime); put("bucket", p.bucket)
                put("embedded", 0); put("ocr_done", 0); put("failed", 0); putNull("ocr")
            }
            it.insertWithOnConflict("photos", null, v, SQLiteDatabase.CONFLICT_REPLACE)
            it.delete("ocr_fts", "docid=?", arrayOf(p.id.toString()))
        }
    }

    fun delete(ids: Collection<Long>) = tx {
        for (id in ids) {
            val a = arrayOf(id.toString())
            it.delete("photos", "id=?", a); it.delete("ocr_fts", "docid=?", a); it.delete("album_photos", "photo=?", a)
        }
    }

    // ---------- indexing progress ----------
    fun pendingEmbedding(limit: Int): List<Long> = ids("SELECT id FROM photos WHERE embedded=0 AND failed=0 ORDER BY taken DESC LIMIT $limit")
    fun pendingOcr(limit: Int): List<Long> = ids("SELECT id FROM photos WHERE ocr_done=0 AND failed=0 ORDER BY taken DESC LIMIT $limit")
    fun markEmbedded(ids: Collection<Long>) = tx { db -> ids.forEach { db.execSQL("UPDATE photos SET embedded=1 WHERE id=?", arrayOf(it)) } }
    fun markFailed(id: Long) = writableDatabase.execSQL("UPDATE photos SET failed=1 WHERE id=?", arrayOf(id))
    fun embeddedIds(): Set<Long> = ids("SELECT id FROM photos WHERE embedded=1").toHashSet()
    fun unembed(ids: Collection<Long>) = tx { db -> ids.forEach { db.execSQL("UPDATE photos SET embedded=0 WHERE id=?", arrayOf(it)) } }

    fun setOcr(id: Long, text: String?) = tx {
        it.execSQL("UPDATE photos SET ocr=?, ocr_done=1 WHERE id=?", arrayOf(text, id))
        it.delete("ocr_fts", "docid=?", arrayOf(id.toString()))
        if (!text.isNullOrBlank()) it.execSQL("INSERT INTO ocr_fts(docid, ocr) VALUES (?, ?)", arrayOf(id, text))
    }

    fun counts(): IndexCounts = readableDatabase.rawQuery(
        "SELECT COUNT(*), SUM(embedded), SUM(ocr_done), SUM(failed) FROM photos", null,
    ).use { c -> c.moveToFirst(); IndexCounts(c.getInt(0), c.getInt(1), c.getInt(2), c.getInt(3)) }

    // ---------- reads ----------
    fun get(id: Long): PhotoRow? = getAll(listOf(id)).firstOrNull()

    fun getAll(ids: List<Long>): List<PhotoRow> {
        if (ids.isEmpty()) return emptyList()
        val by = HashMap<Long, PhotoRow>()
        ids.chunked(500).forEach { chunk ->
            readableDatabase.rawQuery(
                "SELECT id, taken, width, height, mime, bucket, ocr FROM photos WHERE id IN (${chunk.joinToString(",")})", null,
            ).use { c ->
                while (c.moveToNext()) by[c.getLong(0)] = PhotoRow(
                    c.getLong(0), if (c.isNull(1)) null else c.getLong(1), c.getInt(2), c.getInt(3), c.getString(4), c.getString(5), c.getString(6),
                )
            }
        }
        return ids.mapNotNull { by[it] }
    }

    /** id -> capture time (ms) for every photo, for date filters. */
    fun takenTimes(): Map<Long, Long?> = readableDatabase.rawQuery("SELECT id, taken FROM photos", null).use { c ->
        HashMap<Long, Long?>(c.count).apply { while (c.moveToNext()) put(c.getLong(0), if (c.isNull(1)) null else c.getLong(1)) }
    }

    /** Photos whose OCR text contains the query's words, most words matched first (FTS4 has no bm25). */
    fun ocrSearch(query: String, limit: Int = 300): List<Long> {
        val words = Regex("\\w+").findAll(query.lowercase()).map { it.value }.filter { it.length > 2 }.distinct().toList()
        if (words.isEmpty()) return emptyList()
        val match = words.joinToString(" OR ") { if (it.length >= 4) "$it*" else it }
        return try {
            readableDatabase.rawQuery("SELECT docid, ocr FROM ocr_fts WHERE ocr_fts MATCH ? LIMIT 2000", arrayOf(match)).use { c ->
                val hits = ArrayList<Triple<Long, Int, Int>>()
                while (c.moveToNext()) {
                    val text = c.getString(1).lowercase()
                    val n = words.count { w -> Regex("\\b" + Regex.escape(w)).containsMatchIn(text) }
                    hits += Triple(c.getLong(0), n, text.length)
                }
                hits.sortedWith(compareByDescending<Triple<Long, Int, Int>> { it.second }.thenBy { it.third }).take(limit).map { it.first }
            }
        } catch (e: Exception) {
            emptyList()
        }
    }

    // ---------- albums (app-private collections) ----------
    fun findAlbum(name: String): Long? =
        readableDatabase.rawQuery("SELECT id FROM albums WHERE name=? COLLATE NOCASE", arrayOf(name)).use { if (it.moveToFirst()) it.getLong(0) else null }

    fun albumName(id: Long): String? =
        readableDatabase.rawQuery("SELECT name FROM albums WHERE id=?", arrayOf(id.toString())).use { if (it.moveToFirst()) it.getString(0) else null }

    fun createAlbum(name: String): Long =
        writableDatabase.insertOrThrow("albums", null, ContentValues().apply { put("name", name); put("created", System.currentTimeMillis()) })

    fun removeFromAlbum(album: Long, photos: Collection<Long>) = tx { db ->
        photos.forEach { db.execSQL("DELETE FROM album_photos WHERE album=? AND photo=?", arrayOf(album, it)) }
    }

    fun albumsOf(photo: Long): List<Pair<Long, String>> = readableDatabase.rawQuery(
        "SELECT a.id, a.name FROM album_photos p JOIN albums a ON a.id=p.album WHERE p.photo=?", arrayOf(photo.toString()),
    ).use { c -> buildList { while (c.moveToNext()) add(c.getLong(0) to c.getString(1)) } }

    fun addToAlbum(album: Long, photos: Collection<Long>) = tx { db ->
        photos.forEach { db.execSQL("INSERT OR IGNORE INTO album_photos(album, photo) VALUES (?, ?)", arrayOf(album, it)) }
    }

    fun albums(): List<Triple<Long, String, Int>> = readableDatabase.rawQuery(
        "SELECT a.id, a.name, COUNT(p.photo) FROM albums a LEFT JOIN album_photos p ON p.album=a.id GROUP BY a.id ORDER BY a.created DESC", null,
    ).use { c -> buildList { while (c.moveToNext()) add(Triple(c.getLong(0), c.getString(1), c.getInt(2))) } }

    fun albumPhotos(album: Long): List<Long> =
        ids("SELECT p.photo FROM album_photos p LEFT JOIN photos x ON x.id=p.photo WHERE p.album=$album ORDER BY x.taken DESC")

    private fun ids(sql: String): List<Long> = readableDatabase.rawQuery(sql, null).use { c ->
        buildList { while (c.moveToNext()) add(c.getLong(0)) }
    }

    private inline fun <T> tx(block: (SQLiteDatabase) -> T): T {
        val db = writableDatabase
        db.beginTransaction()
        try { return block(db).also { db.setTransactionSuccessful() } } finally { db.endTransaction() }
    }
}

data class IndexCounts(val total: Int, val embedded: Int, val ocrDone: Int, val failed: Int)
