package app.photosearch.ui

import androidx.activity.compose.PredictiveBackHandler
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.ExperimentalFoundationApi
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.asPaddingValues
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawing
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.systemBars
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.lazy.grid.GridCells
import androidx.compose.foundation.lazy.grid.LazyVerticalGrid
import androidx.compose.foundation.lazy.grid.items
import androidx.compose.foundation.lazy.grid.rememberLazyGridState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.Search
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material3.Button
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.drawWithContent
import androidx.compose.ui.geometry.Rect
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.graphics.layer.drawLayer
import androidx.compose.ui.graphics.rememberGraphicsLayer
import androidx.compose.ui.layout.LayoutCoordinates
import androidx.compose.ui.layout.boundsInWindow
import androidx.compose.ui.layout.onGloballyPositioned
import androidx.compose.ui.platform.LocalFocusManager
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.liveRegion
import androidx.compose.ui.semantics.LiveRegionMode
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import app.photosearch.R
import kotlinx.coroutines.CancellationException

@OptIn(ExperimentalFoundationApi::class)
@Composable
fun MainScreen(vm: MainViewModel, onRequestAccess: () -> Unit, onOpenSettings: () -> Unit, onOpenPhoto: (Long) -> Unit) {
    val s by vm.state.collectAsStateWithLifecycle()
    var text by rememberSaveable { mutableStateOf("") }
    var showSettings by rememberSaveable { mutableStateOf(false) }
    var anchor by remember { mutableStateOf<Rect?>(null) } // where the tapped photo is, for the match card
    val gridState = rememberLazyGridState()
    val backdrop = rememberGraphicsLayer()
    val focus = LocalFocusManager.current
    val picker = rememberLauncherForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri ->
        if (uri != null) vm.search(Query.Image(uri, text.trim()))
    }

    // Predictive back: while a search is showing, the back gesture shrinks the results a little, then clears the search.
    var backProgress by remember { mutableFloatStateOf(0f) }
    PredictiveBackHandler(enabled = s.query != null && s.explanation == null) { events ->
        try {
            events.collect { backProgress = it.progress }
            text = ""
            vm.search(null)
        } catch (e: CancellationException) {
            throw e
        } finally {
            backProgress = 0f
        }
    }
    LaunchedEffect(s.query) { gridState.scrollToItem(0) }

    val ids: List<Long> = s.results ?: s.browse.map { it.id }
    val bars = WindowInsets.systemBars.asPaddingValues()

    Box(Modifier.fillMaxSize().background(MaterialTheme.colorScheme.background)) {
        // ---- the photo wall (recorded into `backdrop` so the glass can blur it) ----
        Box(
            Modifier.fillMaxSize()
                .drawWithContent { backdrop.record { this@drawWithContent.drawContent() }; drawLayer(backdrop) }
                .graphicsLayer { val k = 1f - 0.06f * backProgress; scaleX = k; scaleY = k; alpha = 1f - 0.3f * backProgress },
        ) {
            when {
                s.access == Access.NONE -> AccessGate(onRequestAccess, onOpenSettings)
                ids.isEmpty() && !s.searching -> Empty(s)
                else -> LazyVerticalGrid(
                    columns = GridCells.Adaptive(112.dp),
                    state = gridState,
                    modifier = Modifier.fillMaxSize(),
                    contentPadding = PaddingValues(top = bars.calculateTopPadding(), bottom = bars.calculateBottomPadding() + 120.dp),
                    horizontalArrangement = Arrangement.spacedBy(2.dp),
                    verticalArrangement = Arrangement.spacedBy(2.dp),
                ) {
                    items(ids, key = { it }) { id ->
                        val desc = stringResource(R.string.photo_desc, vm.date(id) ?: stringResource(R.string.unknown_date))
                        val why = stringResource(R.string.why_matched)
                        val coords = remember { arrayOfNulls<LayoutCoordinates>(1) } // not state: no recomposition while scrolling
                        Thumb(
                            id, vm.thumbs,
                            Modifier.aspectRatio(1f).animateItem()
                                .onGloballyPositioned { coords[0] = it }
                                .clickable(onClickLabel = why) {
                                    focus.clearFocus()
                                    anchor = coords[0]?.takeIf { it.isAttached }?.boundsInWindow()
                                    vm.explain(id)
                                }
                                .semantics { contentDescription = desc },
                        )
                    }
                }
            }
        }

        // status-bar scrim so system icons stay legible over photos (edge-to-edge)
        Box(
            Modifier.fillMaxWidth().height(bars.calculateTopPadding() + 12.dp)
                .background(Brush.verticalGradient(listOf(Color.Black.copy(alpha = 0.35f), Color.Transparent))),
        )

        // settings, top end
        Glass(backdrop, Modifier.align(Alignment.TopEnd).statusBarsPadding().padding(8.dp), shape = CircleShape, strong = true) {
            IconButton(onClick = { showSettings = true }) {
                Icon(Icons.Filled.Settings, contentDescription = stringResource(R.string.settings))
            }
        }

        // ---- floating search capsule + status pill ----
        Column(
            Modifier.align(Alignment.BottomCenter).fillMaxWidth().navigationBarsPadding().imePadding()
                .padding(horizontal = 16.dp, vertical = 16.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            statusLine(s)?.let { line ->
                Glass(backdrop, shape = RoundedCornerShape(50), strong = true) {
                    Text(
                        line, style = MaterialTheme.typography.labelMedium, color = MaterialTheme.colorScheme.onSurface,
                        maxLines = 2, overflow = TextOverflow.Ellipsis,
                        modifier = Modifier.padding(horizontal = 12.dp, vertical = 6.dp).semantics { liveRegion = LiveRegionMode.Polite },
                    )
                }
            }
            SearchCapsule(
                backdrop = backdrop,
                text = text,
                onText = { text = it },
                image = (s.query as? Query.Image)?.uri,
                enabled = s.access != Access.NONE,
                placeholder = if (s.counts.total > 0) stringResource(R.string.search_hint_n, s.counts.total) else stringResource(R.string.search_hint),
                onSearch = {
                    focus.clearFocus()
                    val img = (s.query as? Query.Image)?.uri
                    when {
                        img != null -> vm.search(Query.Image(img, text.trim()))
                        text.isNotBlank() -> vm.search(Query.Text(text.trim()))
                    }
                },
                onClear = { text = ""; vm.search(null) },
                onPickPhoto = { picker.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)) },
            )
        }

        s.explanation?.let { e ->
            MatchCard(e, anchor, onDismiss = vm::dismissExplanation,
                onMore = { vm.dismissExplanation(); text = ""; vm.search(Query.Similar(e.id)) },
                onOpen = { onOpenPhoto(e.id) })
        }
    }

    if (showSettings) SettingsSheet(vm, s, onDismiss = { showSettings = false }, onRequestAccess = onRequestAccess, onOpenSettings = onOpenSettings)
}

private fun statusLine(s: UiState): String? = when {
    s.modelsMissing -> "Search model not installed — see README"
    s.searching -> "Searching…"
    s.status != null -> s.status
    s.progress != null -> "Indexing on device… ${s.progress.first} of ${s.progress.second}"
    s.counts.total > 0 && s.counts.embedded < s.counts.total - s.counts.failed -> "${s.counts.embedded} of ${s.counts.total} photos indexed"
    else -> null
}

@Composable
private fun SearchCapsule(
    backdrop: androidx.compose.ui.graphics.layer.GraphicsLayer,
    text: String, onText: (String) -> Unit, image: android.net.Uri?, enabled: Boolean, placeholder: String,
    onSearch: () -> Unit, onClear: () -> Unit, onPickPhoto: () -> Unit,
) {
    val onSurface = MaterialTheme.colorScheme.onSurface
    Glass(backdrop, Modifier.widthIn(max = 720.dp).fillMaxWidth(), shape = RoundedCornerShape(32.dp), strong = true) {
        Row(
            Modifier.heightIn(min = 64.dp).padding(start = 20.dp, end = 6.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Icon(Icons.Filled.Search, contentDescription = null, tint = onSurface.copy(alpha = 0.6f))
            Spacer(Modifier.size(12.dp))
            if (image != null) {
                Icon(painterResource(R.drawable.ic_photo_search), contentDescription = stringResource(R.string.searching_by_photo),
                    tint = MaterialTheme.colorScheme.primary, modifier = Modifier.padding(end = 8.dp))
            }
            val hint = if (image != null) stringResource(R.string.refine_hint) else placeholder
            BasicTextField(
                value = text, onValueChange = onText, enabled = enabled, singleLine = true,
                textStyle = MaterialTheme.typography.titleLarge.copy(color = onSurface),
                cursorBrush = SolidColor(MaterialTheme.colorScheme.primary),
                keyboardOptions = KeyboardOptions(imeAction = ImeAction.Search),
                keyboardActions = KeyboardActions(onSearch = { onSearch() }),
                modifier = Modifier.weight(1f).padding(vertical = 14.dp).semantics { contentDescription = hint },
                decorationBox = { inner ->
                    Box {
                        if (text.isEmpty()) Text(hint, style = MaterialTheme.typography.titleLarge, color = onSurface.copy(alpha = 0.55f),
                            maxLines = 1, overflow = TextOverflow.Ellipsis)
                        inner()
                    }
                },
            )
            if (text.isNotEmpty() || image != null) {
                IconButton(onClick = onClear) { Icon(Icons.Filled.Close, contentDescription = stringResource(R.string.clear)) }
            }
            IconButton(onClick = onPickPhoto, enabled = enabled) {
                Icon(painterResource(R.drawable.ic_photo_search), contentDescription = stringResource(R.string.search_with_photo))
            }
        }
    }
}

@Composable
private fun AccessGate(onRequestAccess: () -> Unit, onOpenSettings: () -> Unit) {
    Column(
        Modifier.fillMaxSize().padding(WindowInsets.safeDrawing.asPaddingValues()).padding(32.dp),
        verticalArrangement = Arrangement.Center, horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Text(stringResource(R.string.access_title), style = MaterialTheme.typography.headlineSmall)
        Spacer(Modifier.size(12.dp))
        Text(stringResource(R.string.access_body), style = MaterialTheme.typography.bodyLarge, color = MaterialTheme.colorScheme.onSurfaceVariant)
        Spacer(Modifier.size(24.dp))
        Button(onClick = onRequestAccess, modifier = Modifier.heightIn(min = 48.dp)) { Text(stringResource(R.string.allow_access)) }
        androidx.compose.material3.TextButton(onClick = onOpenSettings, modifier = Modifier.heightIn(min = 48.dp)) {
            Text(stringResource(R.string.open_settings))
        }
    }
}

@Composable
private fun Empty(s: UiState) {
    Box(Modifier.fillMaxSize().padding(32.dp), contentAlignment = Alignment.Center) {
        Text(
            if (s.query != null) stringResource(R.string.no_results) else stringResource(R.string.no_photos),
            style = MaterialTheme.typography.bodyLarge, color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
    }
}
