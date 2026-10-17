package app.photosearch.ui

import android.os.Build
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxScope
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.drawBehind
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.BlurEffect
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Shape
import androidx.compose.ui.graphics.TileMode
import androidx.compose.ui.graphics.drawscope.translate
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.graphics.layer.GraphicsLayer
import androidx.compose.ui.graphics.layer.drawLayer
import androidx.compose.ui.layout.onGloballyPositioned
import androidx.compose.ui.layout.positionInRoot
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp

/**
 * Translucent "glass" like the web demo's floating pill. On Android 12+ the backdrop (the photo grid, recorded into
 * [backdrop] by its parent) is redrawn behind the shape through a RenderEffect blur; older versions get a near-solid
 * surface instead (no RenderEffect, and a cheap fallback is better than a slow one).
 */
@Composable
fun Glass(
    backdrop: GraphicsLayer?,
    modifier: Modifier = Modifier,
    shape: Shape = RoundedCornerShape(28.dp),
    blur: Dp = 24.dp,
    strong: Boolean = false,
    content: @Composable BoxScope.() -> Unit,
) {
    val dark = isSystemInDarkTheme()
    val canBlur = backdrop != null && Build.VERSION.SDK_INT >= Build.VERSION_CODES.S
    val tint = when {
        !canBlur -> if (dark) Color(0xF2202024) else Color(0xF5FFFFFF)
        strong -> if (dark) Color(0xB81E1E22) else Color(0xBDFFFFFF)
        else -> if (dark) Color(0x80141418) else Color(0x6BFFFFFF)
    }
    var origin by remember { mutableStateOf(Offset.Zero) }
    Box(
        modifier
            .shadow(16.dp, shape, ambientColor = Color.Black.copy(alpha = 0.25f), spotColor = Color.Black.copy(alpha = 0.25f))
            .clip(shape)
            .onGloballyPositioned { origin = it.positionInRoot() },
    ) {
        if (canBlur) {
            Box(
                Modifier.matchParentSize()
                    .graphicsLayer {
                        val r = blur.toPx()
                        renderEffect = BlurEffect(r, r, TileMode.Clamp)
                        clip = true
                    }
                    .drawBehind { translate(-origin.x, -origin.y) { drawLayer(backdrop!!) } },
            )
        }
        Box(
            Modifier.matchParentSize().background(tint)
                .border(1.dp, Color.White.copy(alpha = if (dark) 0.12f else 0.35f), shape),
        )
        content()
    }
}
