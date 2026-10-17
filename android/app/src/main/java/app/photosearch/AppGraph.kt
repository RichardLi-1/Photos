package app.photosearch

import android.content.Context
import app.photosearch.data.MediaScanner
import app.photosearch.data.PhotoDb
import app.photosearch.data.VectorFile
import app.photosearch.ml.ModelFiles
import app.photosearch.ml.SigLipEncoder
import app.photosearch.search.Explainer
import app.photosearch.search.Searcher
import app.photosearch.search.VectorIndex
import app.photosearch.tools.PhotoTools
import java.io.File

/** Process-wide singletons (no DI framework: a handful of objects). Everything lives in app-private storage. */
class AppGraph(context: Context) {
    val app: Context = context.applicationContext
    val settings = Settings(app)
    val db = PhotoDb(app)
    val scanner = MediaScanner(app.contentResolver)
    val files = ModelFiles(app)
    val encoder = SigLipEncoder(files)
    val index = VectorIndex(DIM)
    val vectors = VectorFile(File(app.noBackupFilesDir, "vectors_siglip2_b16_384.f16"), DIM)

    val indexLock = kotlinx.coroutines.sync.Mutex()
    @Volatile private var loaded = false

    /** Load embeddings into memory once (fast: a sequential read of ~1.5 KB per photo). Call off the main thread. */
    fun ensureLoaded() {
        if (loaded) return
        synchronized(this) {
            if (loaded) return
            vectors.load(index)
            // a vector without a photo row (photo deleted) is dropped; a row marked embedded without a vector is redone
            val rows = db.embeddedIds()
            index.remove { it in rows }
            db.unembed(rows.filter { !index.contains(it) })
            loaded = true
        }
    }

    val explainer: Explainer by lazy {
        Explainer.load(files.readText(ModelFiles.CONCEPTS), files.map(ModelFiles.CONCEPT_VECTORS), files.readText(ModelFiles.CALIBRATION))
    }

    val searcher = Searcher(
        index = index,
        embedText = { encoder.embedText(it) },
        takenMs = { db.takenTimes() },
        ocrHits = { db.ocrSearch(it) },
    )

    val tools by lazy { PhotoTools(this) }

    companion object {
        const val DIM = 768
        @Volatile private var instance: AppGraph? = null
        fun get(context: Context): AppGraph =
            instance ?: synchronized(this) { instance ?: AppGraph(context).also { instance = it } }
    }
}
