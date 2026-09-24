package io.github.xororz.localdream.utils

import android.app.ActivityManager
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.BatteryManager
import android.os.Build
import android.os.Debug
import android.os.PowerManager
import android.os.Process
import java.util.Locale
import kotlin.math.abs

data class GenerationTelemetrySnapshot(
    val processRamBytes: Long,
    val availableRamBytes: Long,
    val totalRamBytes: Long,
    val batteryPercent: Int,
    val batteryTempC: Float?,
    val batteryCurrentMa: Int?,
    val thermalStatus: String,
)

object GenerationTelemetry {
    fun sample(context: Context): GenerationTelemetrySnapshot {
        val app = context.applicationContext
        val manager = app.getSystemService(ActivityManager::class.java)
        val memory = ActivityManager.MemoryInfo()
        manager?.getMemoryInfo(memory)
        val pssKb = manager?.getProcessMemoryInfo(intArrayOf(Process.myPid()))
            ?.firstOrNull()?.totalPss?.toLong()
            ?: runCatching { Debug.getPss() }.getOrDefault(0L)

        val battery = app.registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
        val level = battery?.getIntExtra(BatteryManager.EXTRA_LEVEL, -1) ?: -1
        val scale = battery?.getIntExtra(BatteryManager.EXTRA_SCALE, 100) ?: 100
        val percent = if (level >= 0 && scale > 0) level * 100 / scale else -1
        val tempRaw = battery?.getIntExtra(
            BatteryManager.EXTRA_TEMPERATURE,
            Int.MIN_VALUE,
        ) ?: Int.MIN_VALUE
        val temperature = tempRaw.takeIf { it != Int.MIN_VALUE }?.div(10f)

        val batteryManager = app.getSystemService(BatteryManager::class.java)
        val currentUa = runCatching {
            batteryManager?.getLongProperty(BatteryManager.BATTERY_PROPERTY_CURRENT_NOW)
        }.getOrNull()
        val currentMa = currentUa
            ?.takeIf { it != Long.MIN_VALUE && abs(it) < 50_000_000L }
            ?.let { (it / 1000L).toInt() }

        val thermal = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            when (app.getSystemService(PowerManager::class.java)?.currentThermalStatus) {
                PowerManager.THERMAL_STATUS_NONE -> "Cool"
                PowerManager.THERMAL_STATUS_LIGHT -> "Light"
                PowerManager.THERMAL_STATUS_MODERATE -> "Moderate"
                PowerManager.THERMAL_STATUS_SEVERE -> "Severe"
                PowerManager.THERMAL_STATUS_CRITICAL -> "Critical"
                PowerManager.THERMAL_STATUS_EMERGENCY -> "Emergency"
                PowerManager.THERMAL_STATUS_SHUTDOWN -> "Shutdown"
                else -> "Unknown"
            }
        } else {
            "Unknown"
        }

        return GenerationTelemetrySnapshot(
            processRamBytes = pssKb * 1024L,
            availableRamBytes = memory.availMem,
            totalRamBytes = memory.totalMem,
            batteryPercent = percent,
            batteryTempC = temperature,
            batteryCurrentMa = currentMa,
            thermalStatus = thermal,
        )
    }

    fun formatBytes(bytes: Long): String {
        if (bytes <= 0L) return "—"
        val gib = bytes / (1024.0 * 1024.0 * 1024.0)
        return if (gib >= 1.0) {
            String.format(Locale.US, "%.1f GB", gib)
        } else {
            String.format(Locale.US, "%.0f MB", bytes / (1024.0 * 1024.0))
        }
    }
}
