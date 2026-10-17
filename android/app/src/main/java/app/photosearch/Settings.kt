package app.photosearch

import android.content.Context
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow

/** User choices, in app-private SharedPreferences. */
class Settings(context: Context) {
    private val prefs = context.getSharedPreferences("settings", Context.MODE_PRIVATE)

    private val _ocr = MutableStateFlow(prefs.getBoolean(OCR, false))
    /** Read text in photos with ML Kit. Off by default: see ocr/Ocr.kt for why. */
    val ocrEnabled: StateFlow<Boolean> = _ocr
    fun setOcr(on: Boolean) { prefs.edit().putBoolean(OCR, on).apply(); _ocr.value = on }

    private val _charging = MutableStateFlow(prefs.getBoolean(CHARGING, false))
    /** Background catch-up indexing only while charging (the periodic job always requires charging). */
    val onlyWhileCharging: StateFlow<Boolean> = _charging
    fun setOnlyWhileCharging(on: Boolean) { prefs.edit().putBoolean(CHARGING, on).apply(); _charging.value = on }

    private companion object {
        const val OCR = "ocr_enabled"
        const val CHARGING = "only_while_charging"
    }
}
