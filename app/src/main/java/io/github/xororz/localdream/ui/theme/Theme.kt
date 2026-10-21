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

    // A completely separate graphite-and-porcelain Liquid theme.
    // Classic mode keeps every existing color and dynamic-color preference.
    val presentationScheme = if (themeState.liquidGlass) {
        if (darkTheme) colorScheme.copy(
            background = Color(0xFF09090B),
            surface = Color(0xFF111113),
            surfaceContainerLowest = Color(0xFF09090B),
            surfaceContainerLow = Color(0xFF151518),
            surfaceContainer = Color(0xFF1A1A1E),
            surfaceContainerHigh = Color(0xFF202024),
            surfaceContainerHighest = Color(0xFF29292E),
            primary = Color(0xFFE8E6E3),
            onPrimary = Color(0xFF101012),
            primaryContainer = Color(0xFF343438),
            onPrimaryContainer = Color(0xFFF5F4F2),
            secondary = Color(0xFFC8C5C2),
            onSecondary = Color(0xFF151517),
            secondaryContainer = Color(0xFF2C2C30),
            onSecondaryContainer = Color(0xFFF1EFED),
            tertiary = Color(0xFFD7D3CD),
            onTertiary = Color(0xFF151517),
            tertiaryContainer = Color(0xFF29282B),
            onTertiaryContainer = Color(0xFFF2F0EC),
            surfaceTint = Color(0xFFE8E6E3),
            inverseSurface = Color(0xFFF2F1EE),
            inverseOnSurface = Color(0xFF1A1A1D),
            inversePrimary = Color(0xFF37373A),
            scrim = Color.Black,
            onBackground = Color(0xFFF4F3F1),
            onSurface = Color(0xFFF4F3F1),
            onSurfaceVariant = Color(0xFFACAAAE),
            outline = Color(0xFF747477),
            outlineVariant = Color(0xFF414145),
        ) else colorScheme.copy(
            background = Color(0xFFF7F6F3),
            surface = Color(0xFFFDFCF9),
            surfaceContainerLowest = Color.White,
            surfaceContainerLow = Color(0xFFF1F0EC),
            surfaceContainer = Color(0xFFE9E8E4),
            surfaceContainerHigh = Color(0xFFDEDDD8),
            surfaceContainerHighest = Color(0xFFD2D1CC),
            primary = Color(0xFF262629),
            onPrimary = Color.White,
            primaryContainer = Color(0xFFE1DFDA),
            onPrimaryContainer = Color(0xFF202023),
            secondary = Color(0xFF55545A),
            onSecondary = Color.White,
            secondaryContainer = Color(0xFFE6E4DF),
            onSecondaryContainer = Color(0xFF1B1B1D),
            tertiary = Color(0xFF66605B),
            onTertiary = Color.White,
            tertiaryContainer = Color(0xFFEAE7E1),
            onTertiaryContainer = Color(0xFF272526),
            surfaceTint = Color(0xFF343437),
            inverseSurface = Color(0xFF232325),
            inverseOnSurface = Color(0xFFF4F2EE),
            inversePrimary = Color(0xFFE0DEDA),
            scrim = Color.Black,
            onBackground = Color(0xFF19191B),
            onSurface = Color(0xFF19191B),
            onSurfaceVariant = Color(0xFF656368),
            outline = Color(0xFF96949A),
            outlineVariant = Color(0xFFCFCDD0),
        )
    } else colorScheme

    MaterialExpressiveTheme(
        colorScheme = presentationScheme,
        typography = Typography,
        shapes = Shapes,
        motionScheme = MotionScheme.expressive(),
        content = content,
    )
}
