package io.github.xororz.localdream.service

import android.app.ActivityManager
import android.content.Context
import android.net.ConnectivityManager
import android.os.Build
import android.os.PowerManager
import kotlin.math.max
import kotlin.math.min

internal data class XetRuntimeTuning(
    val memoryBudgetBytes: Long,
    val thermalStatus: Int,
    val meteredNetwork: Boolean,
    val downstreamKbps: Int,
    val lowMemory: Boolean,
) {
    /**
     * Xet already adapts concurrency to network conditions internally, so a noisy Android
     * bandwidth estimate must never restart an in-flight transfer. Restart only when the
     * device enters a genuinely stricter operating condition.
     */
    fun requiresRestartComparedTo(other: XetRuntimeTuning): Boolean =
        thermalStatus > other.thermalStatus ||
            (lowMemory && !other.lowMemory) ||
            (meteredNetwork && !other.meteredNetwork)

    fun summary(): String =
        "budget=$memoryBudgetBytes concurrency=xet-adaptive " +
            "thermal=$thermalStatus lowMemory=$lowMemory metered=$meteredNetwork " +
            "downstreamKbps=$downstreamKbps"

    companion object {
        fun from(context: Context): XetRuntimeTuning {
            val activityManager =
                context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
            val memoryInfo = ActivityManager.MemoryInfo()
            activityManager.getMemoryInfo(memoryInfo)

            val totalMemory = memoryInfo.totalMem.coerceAtLeast(1L)
            val availableMemory = memoryInfo.availMem.coerceAtLeast(1L)
            val pressureThreshold = memoryInfo.threshold.coerceAtLeast(1L)

            val systemReserve = max(pressureThreshold * 2L, totalMemory / 8L)
            val headroom = (availableMemory - systemReserve).coerceAtLeast(pressureThreshold)
            // Bound Xet by a fraction of real device headroom instead of guessing a fixed
            // phone profile. The native side derives all reconstruction buffers from this.
            var memoryBudget =
                min(totalMemory / 32L, headroom / 6L).coerceAtLeast(pressureThreshold / 2L)

            val powerManager = context.getSystemService(Context.POWER_SERVICE) as PowerManager
            val thermalStatus =
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                    powerManager.currentThermalStatus
                } else {
                    PowerManager.THERMAL_STATUS_NONE
                }

            val thermalNumerator =
                when {
                    thermalStatus >= PowerManager.THERMAL_STATUS_SEVERE -> 1L
                    thermalStatus >= PowerManager.THERMAL_STATUS_MODERATE -> 2L
                    thermalStatus >= PowerManager.THERMAL_STATUS_LIGHT -> 3L
                    else -> 4L
                }
            memoryBudget = (memoryBudget * thermalNumerator / 4L).coerceAtLeast(1L)

            val connectivityManager =
                context.getSystemService(Context.CONNECTIVITY_SERVICE) as ConnectivityManager
            val network = connectivityManager.activeNetwork
            val capabilities = network?.let(connectivityManager::getNetworkCapabilities)
            val downstreamKbps = capabilities?.linkDownstreamBandwidthKbps?.coerceAtLeast(0) ?: 0
            val meteredNetwork = connectivityManager.isActiveNetworkMetered

            return XetRuntimeTuning(
                memoryBudgetBytes = memoryBudget,
                thermalStatus = thermalStatus,
                meteredNetwork = meteredNetwork,
                downstreamKbps = downstreamKbps,
                lowMemory = memoryInfo.lowMemory,
            )
        }
    }
}
