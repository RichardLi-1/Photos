package app.photosearch.index

import android.content.Context
import android.provider.MediaStore
import androidx.work.BackoffPolicy
import androidx.work.Constraints
import androidx.work.ExistingPeriodicWorkPolicy
import androidx.work.ExistingWorkPolicy
import androidx.work.OneTimeWorkRequestBuilder
import androidx.work.PeriodicWorkRequestBuilder
import androidx.work.WorkInfo
import androidx.work.WorkManager
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.map
import java.util.concurrent.TimeUnit

/**
 * - catch-up: one-time, unique, battery-not-low (+ charging if the user asked). Enqueued when the app opens or access
 *   changes; KEEP so it never restarts a running job. Retries with backoff after being stopped (it resumes).
 * - new photos: re-run when MediaStore images change (content-URI trigger), same constraints.
 * - nightly: periodic, charging + battery-not-low + idle, as a safety net.
 */
object IndexScheduler {
    private const val CATCH_UP = "index-catch-up"
    private const val NEW_PHOTOS = "index-new-photos"
    private const val PERIODIC = "index-periodic"

    fun schedule(context: Context, onlyWhileCharging: Boolean, replace: Boolean = false) {
        val wm = WorkManager.getInstance(context)
        val base = Constraints.Builder().setRequiresBatteryNotLow(true).setRequiresCharging(onlyWhileCharging).build()
        wm.enqueueUniqueWork(
            CATCH_UP, if (replace) ExistingWorkPolicy.REPLACE else ExistingWorkPolicy.KEEP,
            OneTimeWorkRequestBuilder<IndexWorker>().setConstraints(base)
                .setBackoffCriteria(BackoffPolicy.LINEAR, 1, TimeUnit.MINUTES).build(),
        )
        armNewPhotos(context, onlyWhileCharging, ExistingWorkPolicy.KEEP)
        val nightly = Constraints.Builder().setRequiresCharging(true).setRequiresBatteryNotLow(true).setRequiresDeviceIdle(true).build()
        wm.enqueueUniquePeriodicWork(
            PERIODIC, ExistingPeriodicWorkPolicy.KEEP,
            PeriodicWorkRequestBuilder<IndexWorker>(1, TimeUnit.DAYS).setConstraints(nightly).build(),
        )
    }

    /** Content-URI triggers fire once; the triggered worker re-arms itself by chaining a fresh one after it. */
    fun armNewPhotos(context: Context, onlyWhileCharging: Boolean, policy: ExistingWorkPolicy) {
        val trigger = Constraints.Builder().setRequiresBatteryNotLow(true).setRequiresCharging(onlyWhileCharging)
            .addContentUriTrigger(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, true)
            .setTriggerContentUpdateDelay(30, TimeUnit.SECONDS).build()
        WorkManager.getInstance(context).enqueueUniqueWork(
            NEW_PHOTOS, policy, OneTimeWorkRequestBuilder<IndexWorker>().setConstraints(trigger).addTag(NEW_PHOTOS_TAG).build(),
        )
    }

    const val NEW_PHOTOS_TAG = "new-photos"

    fun pause(context: Context) = WorkManager.getInstance(context).cancelUniqueWork(CATCH_UP)

    /** Running progress of the catch-up job (done, todo), or null when idle. */
    fun progress(context: Context): Flow<Pair<Int, Int>?> =
        WorkManager.getInstance(context).getWorkInfosForUniqueWorkFlow(CATCH_UP).map { infos ->
            infos.firstOrNull { it.state == WorkInfo.State.RUNNING }?.progress?.let {
                it.getInt(IndexWorker.KEY_DONE, 0) to it.getInt(IndexWorker.KEY_TODO, 0)
            }
        }
}
