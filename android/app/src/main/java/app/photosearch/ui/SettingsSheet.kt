package app.photosearch.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.selection.toggleable
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.heading
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import app.photosearch.R

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SettingsSheet(vm: MainViewModel, s: UiState, onDismiss: () -> Unit, onRequestAccess: () -> Unit, onOpenSettings: () -> Unit) {
    val ocr by vm.g.settings.ocrEnabled.collectAsStateWithLifecycle()
    val charging by vm.g.settings.onlyWhileCharging.collectAsStateWithLifecycle()
    ModalBottomSheet(onDismissRequest = onDismiss) {
        Column(
            Modifier.fillMaxWidth().verticalScroll(rememberScrollState()).padding(horizontal = 24.dp).navigationBarsPadding()
                .padding(bottom = 24.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            Section(stringResource(R.string.library))
            Text(
                when (s.access) {
                    Access.FULL -> stringResource(R.string.access_full)
                    Access.PARTIAL -> stringResource(R.string.access_partial)
                    Access.NONE -> stringResource(R.string.access_none)
                },
                style = MaterialTheme.typography.bodyMedium,
            )
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                if (s.access != Access.FULL) OutlinedButton(onClick = onRequestAccess, Modifier.heightIn(min = 48.dp)) {
                    Text(stringResource(if (s.access == Access.PARTIAL) R.string.choose_photos else R.string.allow_access))
                }
                OutlinedButton(onClick = onOpenSettings, Modifier.heightIn(min = 48.dp)) { Text(stringResource(R.string.open_settings)) }
            }

            HorizontalDivider()
            Section(stringResource(R.string.index))
            val c = s.counts
            Text(stringResource(R.string.index_counts, c.embedded, c.total), style = MaterialTheme.typography.bodyMedium)
            if (c.total > 0) LinearProgressIndicator(progress = { c.embedded / c.total.toFloat() }, modifier = Modifier.fillMaxWidth())
            if (ocr) Text(stringResource(R.string.ocr_counts, c.ocrDone, c.total), style = MaterialTheme.typography.bodySmall)
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                OutlinedButton(onClick = vm::indexNow, Modifier.heightIn(min = 48.dp)) { Text(stringResource(R.string.index_now)) }
                if (s.progress != null) OutlinedButton(onClick = vm::pauseIndexing, Modifier.heightIn(min = 48.dp)) { Text(stringResource(R.string.pause)) }
            }
            Toggle(stringResource(R.string.only_charging), stringResource(R.string.only_charging_body), charging, vm::setOnlyWhileCharging)
            Toggle(stringResource(R.string.ocr_title), stringResource(R.string.ocr_body), ocr, vm::setOcr)

            HorizontalDivider()
            Section(stringResource(R.string.privacy))
            Text(stringResource(R.string.privacy_body), style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
    }
}

@Composable
private fun Section(title: String) =
    Text(title, style = MaterialTheme.typography.titleMedium, modifier = Modifier.semantics { heading() })

@Composable
private fun Toggle(title: String, body: String, checked: Boolean, onChange: (Boolean) -> Unit) {
    Row(
        Modifier.fillMaxWidth().heightIn(min = 56.dp).toggleable(checked, role = Role.Switch, onValueChange = onChange),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Column(Modifier.weight(1f)) {
            Text(title, style = MaterialTheme.typography.bodyLarge)
            Text(body, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
        Spacer(Modifier.size(12.dp))
        Switch(checked = checked, onCheckedChange = null) // the whole row toggles: one big target, one TalkBack node
    }
}
