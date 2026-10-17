package app.photosearch.tools

import android.content.IntentSender
import android.provider.MediaStore
import app.photosearch.AppGraph
import app.photosearch.search.DateRange
import app.photosearch.search.SearchRequest
import org.json.JSONArray
import org.json.JSONObject
import java.security.SecureRandom
import java.time.Instant
import java.time.LocalDate
import java.time.ZoneId

/**
 * Agent-facing capabilities, mirroring search/photosearch/tools.py by name and argument shape (no agent loop here):
 *   tools.schemas()                 -> JSON-schema tool definitions ({name, description, input_schema})
 *   tools.call("search_photos", args) -> JSONObject, never throws: {"ok": false, "error": ...} on failure
 *
 * Android differences:
 *  - Albums are app-private collections in the index DB. Android has no shared user-album API like PhotoKit;
 *    MediaStore "albums" are just folders (moving files there would need write access to every file).
 *  - trash_photos / restore_photos use MediaStore.createTrashRequest: the *system* shows its own confirmation dialog
 *    and trashed items stay restorable for ~30 days. A dry run (default) previews and returns a confirm_token, like
 *    the Python tool; the real call then needs the token AND the user's tap in the system dialog.
 */
class PhotoTools(private val g: AppGraph) {

    /** Launches a system confirmation (IntentSender from MediaStore) and suspends until the user answers. */
    fun interface SystemConfirmer { suspend fun confirm(sender: IntentSender): Boolean }

    @Volatile var confirmer: SystemConfirmer? = null

    class ToolError(msg: String) : Exception(msg)

    private val rng = SecureRandom()
    private val tokens = HashMap<String, Pair<String, Long>>() // token -> (plan, expiry ms)

    // ---------------------------------------------------------------- tools

    fun searchPhotos(query: String?, similarTo: Long?, after: String?, before: String?, albumId: Long?, hasText: Boolean, limit: Int): JSONObject {
        if (query.isNullOrBlank() && similarTo == null) throw ToolError("give query (text) and/or similar_to (a photo id)")
        g.ensureLoaded()
        if (similarTo != null && !g.index.contains(similarTo)) throw ToolError("photo $similarTo not found / not embedded")
        val range = if (after != null || before != null) DateRange(after?.let(::date), before?.let(::date)) else null
        val lim = limit.coerceIn(1, MAX_LIMIT)
        val members = albumId?.let { album(it); g.db.albumPhotos(it).toHashSet() }
        val res = g.searcher.search(SearchRequest(text = query, similarTo = similarTo, range = range, k = if (members != null || hasText) MAX_LIMIT * 5 else lim))
        val rows = g.db.getAll(res.hits.map { it.id }).associateBy { it.id }
        val hits = res.hits.filter { h ->
            (members == null || h.id in members) && (!hasText || !rows[h.id]?.ocr.isNullOrBlank())
        }.take(lim)
        return JSONObject().put("count", hits.size).put("results", JSONArray(hits.mapNotNull { h -> rows[h.id]?.let { brief(it.id, h.score) } }))
            .apply { res.query?.range?.let { put("date_filter", JSONObject().put("after", it.start?.toString()).put("before", it.end?.toString())) } }
    }

    fun getPhotos(ids: List<Long>): JSONObject {
        if (ids.isEmpty() || ids.size > MAX_GET) throw ToolError("photo_ids: 1..$MAX_GET ids")
        val rows = g.db.getAll(ids).associateBy { it.id }
        return JSONObject().put("photos", JSONArray(ids.map { id ->
            val r = rows[id] ?: return@map JSONObject().put("id", id).put("error", "not found")
            brief(id, null).put("width", r.width).put("height", r.height).put("mime", r.mime).put("text", r.ocr)
                .put("albums", JSONArray(g.db.albumsOf(id).map { (aid, n) -> JSONObject().put("id", aid).put("name", n) }))
                .put("embedded", g.index.contains(id))
        }))
    }

    fun explainPhoto(id: Long): JSONObject {
        g.ensureLoaded()
        val r = g.db.get(id) ?: throw ToolError("photo $id not found")
        val v = g.index.vector(id) ?: throw ToolError("photo $id is not embedded yet")
        val tags = g.explainer.tags(v, k = 6)
        var words = "Shows ${tags.joinToString(", ").ifEmpty { "nothing clearly recognisable" }}"
        iso(r.takenMs)?.let { words += "; taken $it" }
        return JSONObject().put("id", id).put("summary", words).put("concepts", JSONArray(tags)).put("photo", brief(id, null))
    }

    fun listAlbums(): JSONObject = JSONObject().put("albums", JSONArray(g.db.albums().map { (id, name, n) ->
        JSONObject().put("id", id).put("name", name).put("count", n)
    }))

    fun getAlbum(id: Long, limit: Int, offset: Int): JSONObject {
        val name = album(id)
        val all = g.db.albumPhotos(id)
        val page = all.drop(offset.coerceAtLeast(0)).take(limit.coerceIn(1, MAX_LIMIT))
        return JSONObject().put("album", JSONObject().put("id", id).put("name", name).put("count", all.size))
            .put("photos", JSONArray(page.map { brief(it, null) }))
    }

    fun createAlbum(name: String, ids: List<Long>?): JSONObject {
        val n = name.trim().takeIf { it.length in 1..120 } ?: throw ToolError("album name must be 1-120 characters")
        if (g.db.findAlbum(n) != null) throw ToolError("an album named '$n' already exists")
        val id = g.db.createAlbum(n)
        val added = ids?.let { addToAlbum(id, it).getInt("added") } ?: 0
        return JSONObject().put("album", JSONObject().put("id", id).put("name", n)).put("added", added)
    }

    fun addToAlbum(album: Long, ids: List<Long>): JSONObject {
        album(album)
        if (ids.size > MAX_ALBUM_BATCH) throw ToolError("too many photo_ids: ${ids.size} > $MAX_ALBUM_BATCH")
        val known = g.db.getAll(ids).map { it.id }.toSet()
        g.db.addToAlbum(album, known)
        return JSONObject().put("added", known.size).put("not_found", JSONArray(ids.filter { it !in known }))
    }

    fun removeFromAlbum(album: Long, ids: List<Long>): JSONObject {
        album(album)
        g.db.removeFromAlbum(album, ids)
        return JSONObject().put("removed", ids.size).put("note", "photos stay in the library")
    }

    suspend fun trashPhotos(ids: List<Long>, reason: String, dryRun: Boolean, token: String?): JSONObject =
        trashOrRestore("trash_photos", ids, reason, dryRun, token, trash = true)

    suspend fun restorePhotos(ids: List<Long>): JSONObject = trashOrRestore("restore_photos", ids, "restore", false, null, trash = false)

    private suspend fun trashOrRestore(tool: String, ids: List<Long>, reason: String, dryRun: Boolean, token: String?, trash: Boolean): JSONObject {
        if (ids.isEmpty() || ids.size > MAX_TRASH) throw ToolError("photo_ids: 1..$MAX_TRASH ids")
        if (trash && reason.isBlank()) throw ToolError("reason required (shown to the user)")
        val todo = if (trash) g.db.getAll(ids).map { it.id }.sorted() else ids.distinct().sorted()
        val out = JSONObject().put(if (dryRun) "would_trash" else if (trash) "trashed" else "restored", todo.size)
            .put("photo_ids", JSONArray(todo)).put("not_found", JSONArray(ids.filter { it !in todo })).put("reason", reason)
            .put("note", "system trash: the user confirms in a system dialog; restorable for about 30 days")
        if (trash && dryRun) return out.put("dry_run", true).put("confirm_token", if (todo.isEmpty()) null else issue(tool, todo, reason))
        if (todo.isEmpty()) return out.put("dry_run", false)
        if (trash) check(tool, todo, reason, token)
        val c = confirmer ?: throw ToolError("no UI to show the system confirmation; open the app")
        val req = MediaStore.createTrashRequest(g.app.contentResolver, todo.map(g.scanner::uri), trash)
        val ok = c.confirm(req.intentSender)
        if (!ok) return out.put(if (trash) "trashed" else "restored", 0).put("dry_run", false).put("declined_by_user", true)
        return out.put("dry_run", false)
    }

    // ---------------------------------------------------------------- registry

    fun schemas(): JSONArray = JSONArray(SPECS.map { (name, desc, schema) ->
        JSONObject().put("name", name).put("description", desc).put("input_schema", JSONObject(schema))
    })

    suspend fun call(name: String, args: JSONObject = JSONObject()): JSONObject = try {
        val body = when (name) {
            "search_photos" -> searchPhotos(args.str("query"), args.long("similar_to"), args.str("after"), args.str("before"),
                args.long("album_id"), args.optBoolean("has_text", false), args.optInt("limit", 24))
            "get_photos" -> getPhotos(args.ids())
            "explain_photo" -> explainPhoto(args.long("photo_id") ?: throw ToolError("photo_id required"))
            "list_albums" -> listAlbums()
            "get_album" -> getAlbum(args.long("album_id") ?: throw ToolError("album_id required"), args.optInt("limit", 60), args.optInt("offset", 0))
            "create_album" -> createAlbum(args.str("name") ?: throw ToolError("name required"), if (args.has("photo_ids")) args.ids() else null)
            "add_to_album" -> addToAlbum(args.long("album_id") ?: throw ToolError("album_id required"), args.ids())
            "remove_from_album" -> removeFromAlbum(args.long("album_id") ?: throw ToolError("album_id required"), args.ids())
            "trash_photos" -> trashPhotos(args.ids(), args.str("reason") ?: "", args.optBoolean("dry_run", true), args.str("confirm_token"))
            "restore_photos" -> restorePhotos(args.ids())
            else -> throw ToolError("unknown tool '$name'; available: ${SPECS.map { it.first }}")
        }
        JSONObject().put("ok", true).also { o -> body.keys().forEach { o.put(it, body.get(it)) } }
    } catch (e: ToolError) {
        JSONObject().put("ok", false).put("error", e.message)
    } catch (e: Exception) {
        JSONObject().put("ok", false).put("error", "${e.javaClass.simpleName}: ${e.message}")
    }

    // ---------------------------------------------------------------- helpers

    private fun album(id: Long): String = g.db.albumName(id) ?: throw ToolError("no album with id $id; call list_albums")

    private fun brief(id: Long, score: Float?): JSONObject {
        val r = g.db.get(id)
        return JSONObject().put("id", id).put("date", iso(r?.takenMs)).put("folder", r?.bucket).put("uri", g.scanner.uri(id).toString())
            .apply { r?.ocr?.takeIf { it.isNotBlank() }?.let { put("text", it.replace('\n', ' ').take(80)) } }
            .apply { if (score != null) put("score", Math.round(score * 10000) / 10000.0) }
    }

    private fun iso(ms: Long?) = ms?.let { Instant.ofEpochMilli(it).atZone(ZoneId.systemDefault()).toLocalDateTime().withSecond(0).withNano(0).toString() }

    private fun date(s: String): LocalDate = try { LocalDate.parse(s.take(10)) } catch (e: Exception) { throw ToolError("dates must be ISO like 2024-06-01") }

    private fun planKey(tool: String, ids: List<Long>, reason: String) = "$tool|${ids.joinToString(",")}|$reason"

    @Synchronized private fun issue(tool: String, ids: List<Long>, reason: String): String {
        val now = System.currentTimeMillis()
        tokens.entries.removeAll { it.value.second < now }
        val t = ByteArray(10).also(rng::nextBytes).joinToString("") { "%02x".format(it) }
        tokens[t] = planKey(tool, ids, reason) to now + TOKEN_TTL_MS
        return t
    }

    @Synchronized private fun check(tool: String, ids: List<Long>, reason: String, token: String?) {
        val (plan, exp) = tokens[token ?: ""] ?: throw ToolError("confirm_token missing or unknown: run with dry_run=true first")
        if (exp < System.currentTimeMillis()) throw ToolError("confirm_token expired; preview again")
        if (plan != planKey(tool, ids, reason)) throw ToolError("confirm_token was issued for a different request")
        tokens.remove(token)
    }

    private fun JSONObject.str(k: String) = if (has(k) && !isNull(k)) getString(k) else null
    private fun JSONObject.long(k: String) = if (has(k) && !isNull(k)) getLong(k) else null
    private fun JSONObject.ids(): List<Long> {
        val a = optJSONArray("photo_ids") ?: throw ToolError("photo_ids required")
        return List(a.length()) { a.getLong(it) }.distinct()
    }

    companion object {
        const val MAX_LIMIT = 200
        const val MAX_GET = 100
        const val MAX_TRASH = 200
        const val MAX_ALBUM_BATCH = 1000
        const val TOKEN_TTL_MS = 15 * 60 * 1000L

        private const val IDS = """{"type":"array","items":{"type":"integer"},"minItems":1}"""
        private fun obj(props: String, vararg required: String) =
            """{"type":"object","properties":{$props},"required":[${required.joinToString(",") { "\"$it\"" }}],"additionalProperties":false}"""

        val SPECS: List<Triple<String, String, String>> = listOf(
            Triple("search_photos", "Find photos by meaning (\"dog on a beach at sunset\"), by text visible in the photo, and/or by visual " +
                "similarity to a photo id. Dates in the query (\"last summer\", \"in 2019\") become filters. Optional explicit date range, " +
                "user album, has text. Returns ids with date/score.",
                obj(""""query":{"type":"string"},"similar_to":{"type":"integer"},"after":{"type":"string","description":"ISO date, inclusive"},""" +
                    """"before":{"type":"string","description":"ISO date, exclusive"},"album_id":{"type":"integer"},"has_text":{"type":"boolean"},""" +
                    """"limit":{"type":"integer","minimum":1,"maximum":$MAX_LIMIT,"default":24}""")),
            Triple("get_photos", "Metadata for up to $MAX_GET photo ids: date, folder, size, OCR text, albums.", obj(""""photo_ids":$IDS""", "photo_ids")),
            Triple("explain_photo", "Describe one photo in words: what it shows (zero-shot concepts) and when.", obj(""""photo_id":{"type":"integer"}""", "photo_id")),
            Triple("list_albums", "List the user's albums (app collections) with id, name and photo count.", obj("")),
            Triple("get_album", "List the photos in one album (paged).",
                obj(""""album_id":{"type":"integer"},"limit":{"type":"integer","default":60},"offset":{"type":"integer","default":0}""", "album_id")),
            Triple("create_album", "Create a new album (unique name, case-insensitive), optionally with photo ids.",
                obj(""""name":{"type":"string"},"photo_ids":$IDS""", "name")),
            Triple("add_to_album", "Add photos to an album (up to $MAX_ALBUM_BATCH per call).", obj(""""album_id":{"type":"integer"},"photo_ids":$IDS""", "album_id", "photo_ids")),
            Triple("remove_from_album", "Take photos out of an album; the photos stay in the library.",
                obj(""""album_id":{"type":"integer"},"photo_ids":$IDS""", "album_id", "photo_ids")),
            Triple("trash_photos", "Move photos to the system trash (restorable for ~30 days). Max $MAX_TRASH. Two steps: dry_run=true (default) " +
                "previews and returns a confirm_token; then call again with dry_run=false and the token. Android then shows its own " +
                "confirmation dialog to the user.",
                obj(""""photo_ids":$IDS,"reason":{"type":"string"},"dry_run":{"type":"boolean","default":true},"confirm_token":{"type":"string"}""", "photo_ids", "reason")),
            Triple("restore_photos", "Bring photos back from the system trash (the system asks the user).", obj(""""photo_ids":$IDS""", "photo_ids")),
        )
    }
}
