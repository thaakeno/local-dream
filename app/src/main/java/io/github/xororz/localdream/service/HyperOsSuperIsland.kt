package io.github.xororz.localdream.service

import android.app.Notification
import android.content.Context
import android.graphics.drawable.Icon
import android.net.Uri
import android.os.Bundle
import android.provider.Settings
import io.github.xororz.localdream.R
import java.util.Locale
import org.json.JSONObject

/**
 * Xiaomi HyperOS 3 Super Island integration for model downloads.
 *
 * The island augments the same foreground-service notification. No second
 * notification is posted. The expanded focus card follows Xiaomi's documented
 * download shape: IM image/text + app icon + simple progress component.
 */
internal object HyperOsSuperIsland {
    private const val PARAM_KEY = "miui.focus.param"
    private const val PICS_KEY = "miui.focus.pics"
    private const val ICON_KEY = "miui.focus.pic_localdream"

    data class Capability(
        val protocolVersion: Int,
        val hasFocusPermission: Boolean,
    ) {
        val supportsSuperIsland: Boolean
            get() = protocolVersion >= 3 && hasFocusPermission
    }

    fun detect(context: Context): Capability {
        val protocol = runCatching {
            Settings.System.getInt(
                context.contentResolver,
                "notification_focus_protocol",
                0,
            )
        }.getOrDefault(0)

        val permission = if (protocol > 0) {
            runCatching {
                val extras = Bundle().apply {
                    putString("package", context.packageName)
                }
                context.contentResolver.call(
                    Uri.parse("content://miui.statusbar.notification.public"),
                    "canShowFocus",
                    null,
                    extras,
                )?.getBoolean("canShowFocus", false) ?: false
            }.getOrDefault(false)
        } else {
            false
        }

        return Capability(protocol, permission)
    }

    fun decorateDownload(
        context: Context,
        notification: Notification,
        capability: Capability,
        modelName: String,
        progress: Float,
        downloadedBytes: Long,
        totalBytes: Long,
        speedBytesPerSecond: Long,
        etaSeconds: Long?,
        paused: Boolean,
    ) {
        if (!capability.supportsSuperIsland) return

        val percent = (progress.coerceIn(0f, 1f) * 100f).toInt()
        val compactName = modelName.take(40)
        val doneText = formatBytes(downloadedBytes)
        val totalText = totalBytes.takeIf { it > 0L }?.let(::formatBytes)
        val leftBytes = if (totalBytes > downloadedBytes) totalBytes - downloadedBytes else 0L
        val leftText = totalBytes.takeIf { it > 0L }?.let { formatBytes(leftBytes) }
        val speedText = speedBytesPerSecond.takeIf { it > 0L }?.let(::formatSpeed)
        val etaText = etaSeconds?.takeIf { it > 0L }?.let(::formatEta)

        // Island A/B areas are extremely narrow. Keep those values compact and
        // reserve the full "downloaded / total · speed · ETA · left" line for
        // the expanded Focus card below.
        val islandDone = formatBytesCompact(downloadedBytes)
        val islandLeft = totalBytes.takeIf { it > 0L }?.let {
            formatBytesCompact(leftBytes)
        }
        val islandEta = etaSeconds?.takeIf { it > 0L }?.let(::formatEtaCompact)

        val primaryLine = when {
            paused && totalText != null -> "$doneText / $totalText · Paused"
            paused -> "$doneText · Paused"
            totalText != null -> "$doneText / $totalText"
            else -> doneText
        }
        val secondaryLine = listOfNotNull(
            speedText,
            etaText?.let { "$it left" },
            leftText?.let { "$it left" },
        ).joinToString(" · ").take(72)

        val accent = "#9DB7FF"
        val track = "#33415F"

        notification.extras.putBundle(
            PICS_KEY,
            Bundle().apply {
                putParcelable(
                    ICON_KEY,
                    Icon.createWithResource(context, R.mipmap.ic_launcher),
                )
            },
        )

        val iconInfo = JSONObject()
            .put("type", 1)
            .put("pic", ICON_KEY)

        val ringProgress = JSONObject()
            .put("progress", percent)
            .put("colorReach", accent)
            .put("colorUnReach", track)
            .put("isCCW", false)

        val bigArea = JSONObject()
            .put(
                "imageTextInfoLeft",
                JSONObject()
                    .put("type", 1)
                    .put("picInfo", iconInfo)
                    .put(
                        "textInfo",
                        JSONObject()
                            .put("frontTitle", "")
                            .put("title", "$percent%")
                            .put("content", islandDone)
                            .put("showHighlightColor", false)
                            .put("narrowFont", true),
                    ),
            )
            .put(
                "progressTextInfo",
                JSONObject()
                    .put("progressInfo", ringProgress)
                    .put(
                        "textInfo",
                        JSONObject()
                            .put("frontTitle", if (paused) "Paused" else "ETA")
                            .put("title", islandEta ?: "…")
                            .put("content", islandLeft?.let { "$it left" } ?: "")
                            .put("showHighlightColor", false)
                            .put("narrowFont", true),
                    ),
            )

        val island = JSONObject()
            .put("islandProperty", 2)
            .put("islandOrder", false)
            .put("dismissIsland", false)
            .put("needCloseAnimation", true)
            .put("highlightColor", accent)
            .put("bigIslandArea", bigArea)
            .put(
                "smallIslandArea",
                JSONObject().put(
                    "combinePicInfo",
                    JSONObject()
                        .put("picInfo", iconInfo)
                        .put("progressInfo", ringProgress),
                ),
            )

        val params = JSONObject()
            .put("protocol", 1)
            .put("business", "download")
            .put("updatable", true)
            .put("reopen", "reopen")
            .put("enableFloat", false)
            .put("islandFirstFloat", false)
            .put("filterWhenNoPermission", false)
            .put("ticker", "$percent% · $compactName")
            .put("aodTitle", "$percent% · $compactName")
            // Xiaomi template 7 / 20: download-focused IM component + clean
            // horizontal progress, without the unrelated base/hint/action
            // components that made the previous island oversized.
            .put(
                "chatInfo",
                JSONObject()
                    .put("picProfile", ICON_KEY)
                    .put("appiconPkg", context.packageName)
                    .put("title", compactName)
                    .put(
                        "content",
                        listOf(primaryLine, secondaryLine)
                            .filter { it.isNotBlank() }
                            .joinToString(" · ")
                            .take(96),
                    ),
            )
            .put("picInfo", JSONObject().put("type", 1))
            .put(
                "progressInfo",
                JSONObject()
                    .put("progress", percent)
                    .put("colorProgress", accent)
                    .put("colorProgressEnd", accent),
            )
            .put("param_island", island)

        notification.extras.putString(
            PARAM_KEY,
            JSONObject().put("param_v2", params).toString(),
        )
    }

    private fun formatBytesCompact(bytes: Long): String = when {
        bytes < 1024L -> "$bytes B"
        bytes < 1024L * 1024L -> String.format(Locale.US, "%.0fK", bytes / 1024.0)
        bytes < 1024L * 1024L * 1024L ->
            String.format(Locale.US, "%.0fM", bytes / (1024.0 * 1024.0))
        else -> String.format(Locale.US, "%.1fG", bytes / (1024.0 * 1024.0 * 1024.0))
    }

    private fun formatEtaCompact(seconds: Long): String {
        val safe = seconds.coerceAtLeast(0L)
        return when {
            safe >= 3600L -> String.format(Locale.US, "%.1fh", safe / 3600.0)
            safe >= 60L -> "${safe / 60L}m"
            else -> "${safe}s"
        }
    }

    private fun formatSpeed(bytesPerSecond: Long): String =
        String.format(Locale.US, "%.1f MB/s", bytesPerSecond / (1024.0 * 1024.0))

    private fun formatBytes(bytes: Long): String = when {
        bytes < 1024L -> "$bytes B"
        bytes < 1024L * 1024L -> String.format(Locale.US, "%.0f KB", bytes / 1024.0)
        bytes < 1024L * 1024L * 1024L ->
            String.format(Locale.US, "%.0f MB", bytes / (1024.0 * 1024.0))
        else -> String.format(Locale.US, "%.2f GB", bytes / (1024.0 * 1024.0 * 1024.0))
    }

    private fun formatEta(seconds: Long): String {
        val safe = seconds.coerceAtLeast(0L)
        val hours = safe / 3600L
        val minutes = (safe % 3600L) / 60L
        val secs = safe % 60L
        return when {
            hours > 0L -> String.format(Locale.US, "%dh %02dm", hours, minutes)
            minutes > 0L -> String.format(Locale.US, "%dm %02ds", minutes, secs)
            else -> String.format(Locale.US, "%ds", secs)
        }
    }
}
