package app.photosearch.ui

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color

// Same palette as the web demo (web/src/index.css): white/black canvas, ink, iOS-blue accent.
private val Accent = Color(0xFF0A84FF)
private val AccentLight = Color(0xFF0060DF) // darker on white for 4.5:1 text contrast

private val Light = lightColorScheme(
    primary = AccentLight, onPrimary = Color.White, background = Color.White, onBackground = Color(0xFF111114),
    surface = Color.White, onSurface = Color(0xFF111114), surfaceVariant = Color(0xFFF2F2F5),
    onSurfaceVariant = Color(0xFF5C5C63), surfaceContainerHigh = Color(0xFFF2F2F5), outline = Color(0xFF8E8E93),
)
private val Dark = darkColorScheme(
    primary = Accent, onPrimary = Color.White, background = Color.Black, onBackground = Color(0xFFF5F5F7),
    surface = Color(0xFF121214), onSurface = Color(0xFFF5F5F7), surfaceVariant = Color(0xFF242428),
    onSurfaceVariant = Color(0xFFB0B0B8), surfaceContainerHigh = Color(0xFF242428), outline = Color(0xFF8E8E93),
)

@Composable
fun PhotoSearchTheme(content: @Composable () -> Unit) {
    MaterialTheme(colorScheme = if (isSystemInDarkTheme()) Dark else Light, content = content)
}
