package io.github.xororz.localdream.utils

import android.content.Context
import android.os.Build
import android.os.VibrationEffect
import android.os.Vibrator
import android.os.VibratorManager

/**
 * One small haptic policy for Local Dream.
 *
 * Android's predefined effects are intentionally used instead of hand-tuned
 * waveforms: OEMs can map TICK/CLICK/HEAVY_CLICK to the actuator in the phone,
 * which feels substantially cleaner across devices.
 */
object HapticSettings {
    private const val PREFS = "app_prefs"
    private const val MASTER = "haptics_enabled"
    private const val INTERACTIONS = "haptics_interactions"
    private const val STAGES = "haptics_generation_stages"
    private const val PROGRESS = "haptics_progress_pulses"
    private const val FAILURES = "haptics_failures"
    private const val STYLE = "haptics_style"

    enum class Style(val value: String, val label: String) {
        Subtle("subtle", "Subtle"),
        Balanced("balanced", "Balanced"),
        Strong("strong", "Strong");

        companion object {
            fun from(value: String?): Style =
                entries.firstOrNull { it.value == value } ?: Subtle
        }
    }

    data class Snapshot(
        val enabled: Boolean,
        val interactions: Boolean,
        val generationStages: Boolean,
        val progressPulses: Boolean,
        val failures: Boolean,
        val style: Style,
    )

    fun read(context: Context): Snapshot {
        val prefs = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
        return Snapshot(
            enabled = prefs.getBoolean(MASTER, true),
            interactions = prefs.getBoolean(INTERACTIONS, true),
            generationStages = prefs.getBoolean(STAGES, true),
            progressPulses = prefs.getBoolean(PROGRESS, true),
            failures = prefs.getBoolean(FAILURES, true),
            style = Style.from(prefs.getString(STYLE, Style.Subtle.value)),
        )
    }

    fun write(context: Context, value: Snapshot) {
        context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
            .edit()
            .putBoolean(MASTER, value.enabled)
            .putBoolean(INTERACTIONS, value.interactions)
            .putBoolean(STAGES, value.generationStages)
            .putBoolean(PROGRESS, value.progressPulses)
            .putBoolean(FAILURES, value.failures)
            .putString(STYLE, value.style.value)
            .apply()
    }
}

object AppHaptics {
    enum class Kind { Interaction, Stage, Progress, Success, Failure }

    fun perform(context: Context, kind: Kind) {
        val settings = HapticSettings.read(context)
        if (!settings.enabled) return
        when (kind) {
            Kind.Interaction -> if (!settings.interactions) return
            Kind.Stage -> if (!settings.generationStages) return
            Kind.Progress -> if (!settings.progressPulses) return
            Kind.Failure -> if (!settings.failures) return
            Kind.Success -> if (!settings.generationStages) return
        }

        val vibrator = vibrator(context) ?: return
        if (!vibrator.hasVibrator()) return

        val effect = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            when (kind) {
                Kind.Interaction -> predefinedForStyle(settings.style, interaction = true)
                Kind.Stage -> predefinedForStyle(settings.style, interaction = false)
                Kind.Progress -> VibrationEffect.createPredefined(VibrationEffect.EFFECT_TICK)
                Kind.Success -> VibrationEffect.createPredefined(VibrationEffect.EFFECT_CLICK)
                Kind.Failure -> VibrationEffect.createPredefined(VibrationEffect.EFFECT_DOUBLE_CLICK)
            }
        } else {
            // minSdk is 28. Predefined OEM-tuned effects arrived in API 29,
            // so keep the legacy fallback extremely short rather than using a
            // long custom waveform.
            val millis = when (kind) {
                Kind.Progress, Kind.Interaction -> 8L
                Kind.Stage, Kind.Success -> 14L
                Kind.Failure -> 24L
            }
            VibrationEffect.createOneShot(millis, VibrationEffect.DEFAULT_AMPLITUDE)
        }
        runCatching { vibrator.vibrate(effect) }
    }

    @androidx.annotation.RequiresApi(Build.VERSION_CODES.Q)
    private fun predefinedForStyle(
        style: HapticSettings.Style,
        interaction: Boolean,
    ): VibrationEffect = when (style) {
        HapticSettings.Style.Subtle ->
            VibrationEffect.createPredefined(VibrationEffect.EFFECT_TICK)
        HapticSettings.Style.Balanced ->
            VibrationEffect.createPredefined(
                if (interaction) VibrationEffect.EFFECT_TICK else VibrationEffect.EFFECT_CLICK,
            )
        HapticSettings.Style.Strong ->
            VibrationEffect.createPredefined(
                if (interaction) VibrationEffect.EFFECT_CLICK else VibrationEffect.EFFECT_HEAVY_CLICK,
            )
    }

    @Suppress("DEPRECATION")
    private fun vibrator(context: Context): Vibrator? =
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            context.getSystemService(VibratorManager::class.java)?.defaultVibrator
        } else {
            context.getSystemService(Context.VIBRATOR_SERVICE) as? Vibrator
        }
}
