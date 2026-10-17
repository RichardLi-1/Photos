package app.photosearch.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Rect
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.semantics.heading
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.IntOffset
import androidx.compose.ui.unit.IntRect
import androidx.compose.ui.unit.IntSize
import androidx.compose.ui.unit.LayoutDirection
import androidx.compose.ui.unit.dp
import androidx.compose.ui.window.Popup
import androidx.compose.ui.window.PopupPositionProvider
import androidx.compose.ui.window.PopupProperties
import app.photosearch.R

/** Small card beside a tapped photo: what it has in common with the current search (web: MatchCard.tsx). */
@OptIn(ExperimentalLayoutApi::class)
@Composable
fun MatchCard(e: Explanation, anchor: Rect?, onDismiss: () -> Unit, onMore: () -> Unit, onOpen: () -> Unit) {
    Popup(
        popupPositionProvider = BesideAnchor(anchor),
        onDismissRequest = onDismiss,
        properties = PopupProperties(focusable = true), // back + outside tap dismiss; TalkBack moves into the card
    ) {
        // The popup is its own window, so it can't sample the grid: use the solid variant of the glass.
        Glass(null, Modifier.width(280.dp), shape = RoundedCornerShape(20.dp)) {
            Column(Modifier.padding(horizontal = 16.dp, vertical = 12.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                Text(e.label, style = MaterialTheme.typography.labelMedium, color = MaterialTheme.colorScheme.onSurfaceVariant,
                    maxLines = 2, overflow = TextOverflow.Ellipsis, modifier = Modifier.semantics { heading() })
                if (e.shared.isNotEmpty()) {
                    FlowRow(horizontalArrangement = Arrangement.spacedBy(4.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
                        e.shared.forEach { t ->
                            Text(t, style = MaterialTheme.typography.bodyMedium,
                                modifier = Modifier.clip(RoundedCornerShape(50)).background(MaterialTheme.colorScheme.surfaceVariant)
                                    .padding(horizontal = 10.dp, vertical = 4.dp))
                        }
                    }
                } else {
                    Text(stringResource(R.string.nothing_specific), style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant)
                }
                if (e.textHits.isNotEmpty()) {
                    Text(stringResource(R.string.written_in_photo, e.textHits.joinToString(", ")), style = MaterialTheme.typography.bodyMedium)
                }
                e.date?.let { Text(it, style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant) }
                Row(horizontalArrangement = Arrangement.spacedBy(4.dp)) {
                    TextButton(onClick = onMore, modifier = Modifier.heightIn(min = 48.dp)) { Text(stringResource(R.string.more_like_this)) }
                    TextButton(onClick = onOpen, modifier = Modifier.heightIn(min = 48.dp)) { Text(stringResource(R.string.open)) }
                }
            }
        }
    }
}

/** Below the photo if there's room, else above it; horizontally centred on it and kept 12dp inside the window. */
private class BesideAnchor(private val anchor: Rect?) : PopupPositionProvider {
    override fun calculatePosition(anchorBounds: IntRect, windowSize: IntSize, layoutDirection: LayoutDirection, popupContentSize: IntSize): IntOffset {
        val margin = 36
        val a = anchor ?: return IntOffset((windowSize.width - popupContentSize.width) / 2, (windowSize.height - popupContentSize.height) / 2)
        val x = (a.center.x - popupContentSize.width / 2f).toInt().coerceIn(margin, maxOf(margin, windowSize.width - popupContentSize.width - margin))
        val below = a.bottom.toInt() - 24
        val y = if (below + popupContentSize.height < windowSize.height - 300) below
        else (a.top.toInt() + 24 - popupContentSize.height).coerceAtLeast(margin)
        return IntOffset(x, y)
    }
}
