package app.photosearch.ui

import android.Manifest
import android.app.Application
import android.content.Context
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build
import android.os.SystemClock
import androidx.core.content.ContextCompat
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import app.photosearch.AppGraph
import app.photosearch.data.IndexCounts
import app.photosearch.data.MediaPhoto
import app.photosearch.index.IndexScheduler
import app.photosearch.ml.ImagePreprocess
import app.photosearch.ml.MissingModelException
import app.photosearch.search.Explainer
import app.photosearch.search.SearchRequest
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.time.Instant
import java.time.ZoneId
import java.time.format.DateTimeFormatter
import java.time.format.FormatStyle

enum class Access { NONE, PARTIAL, FULL }

sealed interface Query {
    data class Text(val q: String) : Query
    data class Similar(val id: Long) : Query
    data class Image(val uri: Uri, val text: String = "") : Query
}

data class Explanation(
    val id: Long, val label: String, val shared: List<String>, val textHits: List<String>, val date: String?,
)

data class UiState(
    val access: Access = Access.NONE,
    val browse: List<MediaPhoto> = emptyList(),
    val query: Query? = null,
    val results: List<Long>? = null,
    val status: String? = null,
    val searching: Boolean = false,
    val counts: IndexCounts = IndexCounts(0, 0, 0, 0),
    val progress: Pair<Int, Int>? = null,
    val modelsMissing: Boolean = false,
    val explanation: Explanation? = null,
)

class MainViewModel(app: Application) : AndroidViewModel(app) {
    val g = AppGraph.get(app)
    val thumbs = Thumbnails(app.contentResolver, g.scanner)
    private val _state = MutableStateFlow(UiState(modelsMissing = !g.encoder.modelsAvailable))
    val state: StateFlow<UiState> = _state
    private var searchJob: Job? = null
    private var queryVector: FloatArray? = null
    private var taken: Map<Long, Long?> = emptyMap()

    init {
        viewModelScope.launch {
            IndexScheduler.progress(app).collect { p ->
                _state.update { it.copy(progress = p) }
                refreshCounts()
            }
        }
    }

    /** Re-check access (it can change in system settings while we're paused) and reload. */
    fun onResume() {
        val a = access(getApplication())
        val changed = a != _state.value.access
        _state.update { it.copy(access = a) }
        if (a != Access.NONE && (changed || _state.value.browse.isEmpty())) {
            reloadLibrary()
            IndexScheduler.schedule(getApplication(), g.settings.onlyWhileCharging.value)
        }
    }

    fun reloadLibrary() = viewModelScope.launch(Dispatchers.IO) {
        val all = g.scanner.all()
        taken = all.associate { it.id to it.takenMs }
        g.ensureLoaded()
        _state.update { it.copy(browse = all) }
        refreshCounts()
    }

    private suspend fun refreshCounts() = withContext(Dispatchers.IO) {
        val c = g.db.counts()
        _state.update { it.copy(counts = c) }
    }

    fun search(q: Query?) {
        searchJob?.cancel()
        queryVector = null
        _state.update { it.copy(query = q, results = null, status = null, explanation = null, searching = q != null) }
        if (q == null) return
        searchJob = viewModelScope.launch(Dispatchers.Default) {
            val t0 = SystemClock.elapsedRealtime()
            val res = try {
                g.ensureLoaded()
                when (q) {
                    is Query.Text -> g.searcher.search(SearchRequest(text = q.q))
                    is Query.Similar -> g.searcher.search(SearchRequest(similarTo = q.id))
                    is Query.Image -> {
                        val bmp = ImagePreprocess.decodeSampled(getApplication<Application>().contentResolver, q.uri)
                        val v = g.encoder.embedImage(bmp)
                        g.searcher.search(SearchRequest(text = q.text.ifBlank { null }, imageVector = v))
                    }
                }
            } catch (e: MissingModelException) {
                _state.update { it.copy(searching = false, modelsMissing = true, status = "Search model not installed") }
                return@launch
            } catch (e: Exception) {
                _state.update { it.copy(searching = false, status = "Search failed: ${e.message}") }
                return@launch
            }
            queryVector = res.queryVector
            val ms = SystemClock.elapsedRealtime() - t0
            val c = _state.value.counts
            val partial = if (c.embedded < c.total) " · ${c.embedded} of ${c.total} indexed" else ""
            val dates = res.query?.phrase?.let { " · “$it”" } ?: ""
            _state.update {
                it.copy(results = res.hits.map { h -> h.id }, searching = false,
                    status = "${res.hits.size} best matches$dates · $ms ms · on-device$partial")
            }
        }
    }

    fun explain(id: Long) = viewModelScope.launch(Dispatchers.Default) {
        val e = try {
            explanation(id)
        } catch (e: Exception) {
            Explanation(id, "In this photo", emptyList(), emptyList(), date(id))
        }
        _state.update { it.copy(explanation = e) }
    }

    fun dismissExplanation() = _state.update { it.copy(explanation = null) }

    private fun explanation(id: Long): Explanation {
        val ex = g.explainer
        val v = g.index.vector(id) ?: return Explanation(id, "Not indexed yet", emptyList(), emptyList(), date(id))
        val q = _state.value.query
        val qv = queryVector
        var (label, shared) = when {
            q is Query.Text && qv != null -> "Matches “${q.q}”" to ex.matchedText(qv, v)
            q is Query.Similar -> "In common with that photo" to (g.index.vector(q.id)?.let { ex.sharedWithImage(it, v) } ?: emptyList())
            q is Query.Image && qv != null -> "In common with your photo" to ex.sharedWithImage(qv, v)
            else -> "In this photo" to ex.tags(v)
        }
        if (shared.isEmpty() && q != null) { label = "Overall look"; shared = ex.tags(v, k = 3) }
        val textHits = if (q is Query.Text) Explainer.wordHits(q.q, g.db.get(id)?.ocr) else emptyList()
        return Explanation(id, label, shared, textHits, date(id))
    }

    fun date(id: Long): String? = (taken[id] ?: g.db.get(id)?.takenMs)?.let {
        DateTimeFormatter.ofLocalizedDate(FormatStyle.MEDIUM).format(Instant.ofEpochMilli(it).atZone(ZoneId.systemDefault()))
    }

    fun setOcr(on: Boolean) {
        g.settings.setOcr(on)
        if (on) IndexScheduler.schedule(getApplication(), g.settings.onlyWhileCharging.value, replace = true)
    }

    fun setOnlyWhileCharging(on: Boolean) {
        g.settings.setOnlyWhileCharging(on)
        IndexScheduler.schedule(getApplication(), on, replace = true)
    }

    fun indexNow() { IndexScheduler.schedule(getApplication(), g.settings.onlyWhileCharging.value, replace = true) }
    fun pauseIndexing() { IndexScheduler.pause(getApplication()) }

    companion object {
        fun permissionsToRequest(): Array<String> = when {
            Build.VERSION.SDK_INT >= 34 -> arrayOf(Manifest.permission.READ_MEDIA_IMAGES, Manifest.permission.READ_MEDIA_VISUAL_USER_SELECTED)
            Build.VERSION.SDK_INT >= 33 -> arrayOf(Manifest.permission.READ_MEDIA_IMAGES)
            else -> arrayOf(Manifest.permission.READ_EXTERNAL_STORAGE)
        }

        fun access(ctx: Context): Access {
            fun ok(p: String) = ContextCompat.checkSelfPermission(ctx, p) == PackageManager.PERMISSION_GRANTED
            return when {
                Build.VERSION.SDK_INT >= 33 && ok(Manifest.permission.READ_MEDIA_IMAGES) -> Access.FULL
                Build.VERSION.SDK_INT >= 34 && ok(Manifest.permission.READ_MEDIA_VISUAL_USER_SELECTED) -> Access.PARTIAL
                Build.VERSION.SDK_INT < 33 && ok(Manifest.permission.READ_EXTERNAL_STORAGE) -> Access.FULL
                else -> Access.NONE
            }
        }
    }
}
