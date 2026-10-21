package io.github.xororz.localdream.ui.theme

import android.os.Build
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.ExperimentalMaterial3ExpressiveApi
import androidx.compose.material3.MaterialExpressiveTheme
import androidx.compose.ui.graphics.Color
import androidx.compose.material3.MotionScheme
import androidx.compose.material3.dynamicDarkColorScheme
import androidx.compose.material3.dynamicLightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.compositionLocalOf
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.platform.LocalContext
import io.github.xororz.localdream.data.DarkModePreference
import io.github.xororz.localdream.data.ThemePreferences
import io.github.xororz.localdream.data.ThemeState

/**
 * Holds the live theme preferences plus a setter that also persists. Provided
 * via [LocalThemeController] so settings UIs can read and mutate without
 * threading state down through every screen.
 */
class ThemeController(initial: ThemeState, private val onPersist: (ThemeState) -> Unit) {
    var state: ThemeState by mutableStateOf(initial)
        private set

    fun update(transform: (ThemeState) -> ThemeState) {
        val next = transform(state)
        if (next != state) {
            state = next
            onPersist(next)
        }
    }
}

val LocalThemeController = compositionLocalOf<ThemeController> {
    error("ThemeController not provided")
}

@Composable
fun rememberThemeController(): ThemeController {
    val context = LocalContext.current
    val prefs = remember { ThemePreferences(context) }
    return remember { ThemeController(prefs.read(), prefs::write) }
}

private inline val ThemeState.systemFollowsDark: Boolean
    @Composable get() = when (darkMode) {
        DarkModePreference.SYSTEM -> isSystemInDarkTheme()
        DarkModePreference.LIGHT -> false
        DarkModePreference.DARK -> true
    }

@OptIn(ExperimentalMaterial3ExpressiveApi::class)
@Composable
fun LocalDreamTheme(themeState: ThemeState, content: @Composable () -> Unit) {
    val darkTheme = themeState.systemFollowsDark
    val colorScheme = when {
        themeState.dynamicColor && Build.VERSION.SDK_INT >= Build.VERSION_CODES.S -> {
            val context = LocalContext.current
            if (darkTheme) dynamicDarkColorScheme(context) else dynamicLightColorScheme(context)
        }

        else -> themeState.preset.scheme(darkTheme)
    }

    // Graphite/ice glass surfaces; the classic theme remains untouched.
    val presentationScheme = if (themeState.liquidGlass) {
        if (darkTheme) {
            colorScheme.copy(
                background = Color(0xFF0B1019),
                surface = Color(0xFF111925),
                surfaceContainer = Color(0xFF1A2433),
                surfaceContainerLow = Color(0xFF15202C),
                surfaceContainerHigh = Color(0xFF253242),
                primary = Color(0xFF9FC6FF),
                onPrimary = Color(0xFF102239),
            )
        } else {
            colorScheme.copy(
                background = Color(0xFFF1F5FB),
                surface = Color(0xFFFAFBFF),
                surfaceContainer = Color(0xFFE7EDF7),
                surfaceContainerLow = Color(0xFFEFF3FA),
                surfaceContainerHigh = Color(0xFFDDE5F2),
                primary = Color(0xFF285EA9),
                onPrimary = Color.White,
            )
        }
    } else colorScheme

    MaterialExpressiveTheme(
        colorScheme = presentationScheme,
        typography = Typography,
        shapes = Shapes,
        motionScheme = MotionScheme.expressive(),
        content = content,
    )
}
