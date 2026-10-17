package app.photosearch.index

import android.app.NotificationChannel
import android.app.NotificationManager
import android.content.Context
import android.content.pm.ServiceInfo
import androidx.core.app.NotificationCompat
import androidx.work.CoroutineWorker
import androidx.work.ExistingWorkPolicy
import androidx.work.ForegroundInfo
import androidx.work.WorkerParameters
import androidx.work.workDataOf
import app.photosearch.AppGraph
import app.photosearch.R
import app.photosearch.ml.ImagePreprocess
import app.photosearch.ml.MissingModelException
import app.photosearch.ocr.MlKitOcr
import app.photosearch.ocr.OcrEngine
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext

/**
 * Incremental, resumable indexing: sync with MediaStore, then embed (and optionally OCR) pending photos in small
 * batches. Each batch is persisted before the next starts, so if the system stops the work (constraints lost, 10-minute
 * limit, reboot) the next run continues where this one stopped. Long runs show a foreground notification.
 */
class IndexWorker(context: Context, params: WorkerParameters) : CoroutineWorker(context, params) {
    private val g = AppGraph.get(context)

    // catch-up, new-photo and nightly runs can overlap: one at a time, the others wait then find little left to do
    override suspend fun doWork(): Result = withContext(Dispatchers.Default) { g.indexLock.withLock { work() } }

    private suspend fun work(): Result {
        if (!g.encoder.modelsAvailable) return Result.failure(workDataOf(KEY_ERROR to "models missing"))
        g.ensureLoaded()
        val sync = g.scanner.sync(g.db)
        if (sync.removed.isNotEmpty()) {
            val keep = g.db.embeddedIds()
            g.index.remove { it in keep }
            if (g.vectors.recordCount() > g.index.size * 5L / 4 + 64) g.vectors.compact(g.index) // >20% dead rows
        }
        var counts = g.db.counts()
        val ocrOn = g.settings.ocrEnabled.value
        val todo = (counts.total - counts.embedded - counts.failed) + if (ocrOn) counts.total - counts.ocrDone - counts.failed else 0
        if (todo >= FOREGROUND_ABOVE) runCatching { setForeground(foregroundInfo(0, todo)) } // may be refused in background: fine
        var done = 0
        val ocr: OcrEngine? = if (ocrOn) MlKitOcr() else null
        try {
            while (!isStopped) {
                val batch = g.db.pendingEmbedding(BATCH)
                if (batch.isEmpty()) break
                val out = ArrayList<Pair<Long, FloatArray>>(batch.size)
                for (id in batch) {
                    if (isStopped) break
                    try {
                        // one decode: short side 768..1535 px, aspect kept (OCR reads it); the model input is a bicubic squash of it
                        val sampled = ImagePreprocess.decodeSampled(g.app.contentResolver, g.scanner.uri(id))
                        out += id to g.encoder.embedImage(sampled)
                        if (ocr != null) g.db.setOcr(id, ocr.read(sampled).ifBlank { null })
                        sampled.recycle()
                    } catch (e: MissingModelException) {
                        throw e
                    } catch (e: Exception) {
                        g.db.markFailed(id) // unreadable / removed meanwhile: skip, don't retry forever
                    }
                }
                g.vectors.append(out)
                out.forEach { (id, v) -> g.index.put(id, v) }
                g.db.markEmbedded(out.map { it.first })
                done += batch.size
                setProgress(workDataOf(KEY_DONE to done, KEY_TODO to todo))
                if (todo >= FOREGROUND_ABOVE) runCatching { setForeground(foregroundInfo(done, todo)) }
            }
            // OCR for photos embedded before OCR was switched on
            while (ocr != null && !isStopped) {
                val batch = g.db.pendingOcr(BATCH)
                if (batch.isEmpty()) break
                for (id in batch) {
                    if (isStopped) break
                    runCatching {
                        val bmp = ImagePreprocess.decodeSampled(g.app.contentResolver, g.scanner.uri(id))
                        g.db.setOcr(id, ocr.read(bmp).ifBlank { null })
                        bmp.recycle()
                    }.onFailure { g.db.setOcr(id, null) }
                }
                done += batch.size
                setProgress(workDataOf(KEY_DONE to done, KEY_TODO to todo))
            }
        } finally {
            ocr?.close()
            g.encoder.releaseImage()
        }
        counts = g.db.counts()
        if (IndexScheduler.NEW_PHOTOS_TAG in tags) {
            IndexScheduler.armNewPhotos(applicationContext, g.settings.onlyWhileCharging.value, ExistingWorkPolicy.APPEND_OR_REPLACE)
        }
        return if (isStopped) Result.retry() else Result.success(workDataOf(KEY_DONE to counts.embedded))
    }

    private fun foregroundInfo(done: Int, total: Int): ForegroundInfo {
        val nm = applicationContext.getSystemService(NotificationManager::class.java)
        nm.createNotificationChannel(
            NotificationChannel(CHANNEL, applicationContext.getString(R.string.channel_indexing), NotificationManager.IMPORTANCE_LOW),
        )
        val n = NotificationCompat.Builder(applicationContext, CHANNEL)
            .setSmallIcon(R.drawable.ic_notification)
            .setContentTitle(applicationContext.getString(R.string.indexing_title))
            .setContentText(applicationContext.getString(R.string.indexing_progress, done, total))
            .setProgress(total, done, false)
            .setOngoing(true)
            .setSilent(true)
            .addAction(0, applicationContext.getString(R.string.pause), androidx.work.WorkManager.getInstance(applicationContext).createCancelPendingIntent(id))
            .build()
        return ForegroundInfo(NOTIFICATION_ID, n, ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC)
    }

    companion object {
        const val KEY_DONE = "done"
        const val KEY_TODO = "todo"
        const val KEY_ERROR = "error"
        private const val BATCH = 16
        private const val FOREGROUND_ABOVE = 200
        private const val CHANNEL = "indexing"
        private const val NOTIFICATION_ID = 1
    }
}
