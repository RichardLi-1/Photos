package app.photosearch

import android.Manifest
import android.content.Intent
import android.content.IntentSender
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.Settings as SystemSettings
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.ActivityResult
import androidx.activity.result.IntentSenderRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import app.photosearch.tools.PhotoTools
import app.photosearch.ui.Access
import app.photosearch.ui.MainScreen
import app.photosearch.ui.MainViewModel
import app.photosearch.ui.PhotoSearchTheme
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class MainActivity : ComponentActivity() {
    private val vm: MainViewModel by viewModels()

    private val mediaPermission = registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) {
        vm.onResume()
        if (MainViewModel.access(this) != Access.NONE) {
            vm.reloadLibrary() // partial access: the selection may have changed even though the access level didn't
            vm.indexNow()
            maybeAskNotifications()
        }
    }
    private val notificationPermission = registerForActivityResult(ActivityResultContracts.RequestPermission()) {}

    // System confirmation dialogs (MediaStore trash/restore requests) for PhotoTools
    private var pendingConfirm: CompletableDeferred<Boolean>? = null
    private val confirmLauncher = registerForActivityResult(ActivityResultContracts.StartIntentSenderForResult()) { r: ActivityResult ->
        pendingConfirm?.complete(r.resultCode == RESULT_OK)
        pendingConfirm = null
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        enableEdgeToEdge()
        super.onCreate(savedInstanceState)
        setContent {
            PhotoSearchTheme {
                MainScreen(
                    vm,
                    onRequestAccess = { mediaPermission.launch(MainViewModel.permissionsToRequest()) },
                    onOpenSettings = {
                        startActivity(Intent(SystemSettings.ACTION_APPLICATION_DETAILS_SETTINGS, Uri.fromParts("package", packageName, null)))
                    },
                    onOpenPhoto = { id ->
                        val view = Intent(Intent.ACTION_VIEW).setDataAndType(vm.g.scanner.uri(id), "image/*")
                            .addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
                        runCatching { startActivity(view) }
                    },
                )
            }
        }
    }

    override fun onStart() {
        super.onStart()
        vm.g.tools.confirmer = object : PhotoTools.SystemConfirmer {
            override suspend fun confirm(sender: IntentSender): Boolean {
                val d = CompletableDeferred<Boolean>()
                withContext(Dispatchers.Main) {
                    pendingConfirm?.complete(false)
                    pendingConfirm = d
                    confirmLauncher.launch(IntentSenderRequest.Builder(sender).build())
                }
                val ok = d.await()
                if (ok) vm.reloadLibrary()
                return ok
            }
        }
    }

    override fun onResume() {
        super.onResume()
        vm.onResume()
    }

    override fun onStop() {
        vm.g.tools.confirmer = null
        super.onStop()
    }

    private fun maybeAskNotifications() {
        // Only needed for the progress notification of a long first index.
        if (Build.VERSION.SDK_INT < 33) return
        lifecycleScope.launch {
            val n = withContext(Dispatchers.IO) { vm.g.scanner.all().size }
            if (n >= 200) notificationPermission.launch(Manifest.permission.POST_NOTIFICATIONS)
        }
    }
}
