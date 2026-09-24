package io.github.xororz.localdream.service

import android.app.Notification
import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import android.graphics.drawable.Icon
import android.net.Uri
import android.os.Bundle
import android.provider.Settings
import io.github.xororz.localdream.R
import java.util.Locale
import org.json.JSONArray
import org.json.JSONObject

internal object HyperOsSuperIsland {
    private const val PARAM_KEY = "miui.focus.param"
    private const val PICS_KEY = "miui.focus.pics"
    private const val ACTIONS_KEY = "miui.focus.actions"
    private const val ICON_KEY = "miui.focus.pic_localdream"
    private const val TOGGLE_ACTION_KEY = "miui.focus.action_toggle"
    private const val CANCEL_ACTION_KEY = "miui.focus.action_cancel"

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
                val extras = Bundle().apply { putString("package", context.packageName) }
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
        val compactName = modelName.take(28)
        val done = formatBytesCompact(downloadedBytes)
        val total = totalBytes.takeIf { it > 0L }?.let(::formatBytesCompact)
        val speed = speedBytesPerSecond.takeIf { it > 0L }?.let(::formatSpeedCompact)
        val eta = etaSeconds?.takeIf { it >= 0L }?.let(::formatEtaCompact)

        notification.extras.putBundle(
            PICS_KEY,
            Bundle().apply {
                putParcelable(ICON_KEY, Icon.createWithResource(context, R.mipmap.ic_launcher))
            },
        )

        val toggleIntent = PendingIntent.getService(
            context,
            72,
            Intent(context, ModelDownloadService::class.java).apply {
                action = if (paused) {
                    ModelDownloadService.ACTION_RESUME_DOWNLOAD
                } else {
                    ModelDownloadService.ACTION_PAUSE_DOWNLOAD
                }
            },
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
        )
        val cancelIntent = PendingIntent.getService(
            context,
            73,
            Intent(context, ModelDownloadService::class.java).apply {
                action = ModelDownloadService.ACTION_CANCEL_DOWNLOAD
            },
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
        )

        notification.extras.putBundle(
            ACTIONS_KEY,
            Bundle().apply {
                putParcelable(
                    TOGGLE_ACTION_KEY,
                    Notification.Action.Builder(
                        Icon.createWithResource(
                            context,
                            if (paused) android.R.drawable.ic_media_play
                            else android.R.drawable.ic_media_pause,
                        ),
                        if (paused) "Resume" else "Pause",
                        toggleIntent,
                    ).build(),
                )
                putParcelable(
                    CANCEL_ACTION_KEY,
                    Notification.Action.Builder(
                        Icon.createWithResource(
                            context,
                            android.R.drawable.ic_menu_close_clear_cancel,
                        ),
                        "Cancel",
                        cancelIntent,
                    ).build(),
                )
            },
        )

        val accent = "#9DB7FF"
        val track = "#33415F"
        val iconInfo = JSONObject().put("type", 1).put("pic", ICON_KEY)
        val ring = JSONObject()
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
                            .put("content", if (total != null) "$done/$total" else done)
                            .put("showHighlightColor", false)
                            .put("narrowFont", true),
                    ),
            )
            .put(
                "progressTextInfo",
                JSONObject()
                    .put("progressInfo", ring)
                    .put(
                        "textInfo",
                        JSONObject()
                            .put("frontTitle", if (paused) "Paused" else "")
                            .put("title", eta ?: speed ?: "")
                            .put("content", if (eta != null && speed != null) speed else "")
                            .put("showHighlightColor", false)
                            .put("narrowFont", true),
                    ),
            )

        val island = JSONObject()
            .put("islandProperty", 1)
            .put("islandOrder", false)
            .put("dismissIsland", false)
            .put("needCloseAnimation", true)
            .put("highlightColor", accent)
            .put("bigIslandArea", bigArea)
            .put(
                "smallIslandArea",
                JSONObject().put(
                    "combinePicInfo",
                    JSONObject().put("picInfo", iconInfo).put("progressInfo", ring),
                ),
            )

        val params = JSONObject()
            .put("protocol", 1)
            .put("business", "download")
            .put("updatable", true)
            .put("enableFloat", false)
            .put("islandFirstFloat", false)
            .put("filterWhenNoPermission", false)
            .put("ticker", "$percent% · $compactName")
            .put("aodTitle", "$percent% · $compactName")
            .put(
                "baseInfo",
                JSONObject()
                    .put("type", 2)
                    .put("title", compactName)
                    .put(
                        "content",
                        if (paused) "Paused"
                        else listOfNotNull(speed, eta).joinToString(" · "),
                    ),
            )
            .put(
                "actions",
                JSONArray()
                    .put(JSONObject().put("action", TOGGLE_ACTION_KEY))
                    .put(JSONObject().put("action", CANCEL_ACTION_KEY)),
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

    private fun formatSpeedCompact(bytesPerSecond: Long): String =
        String.format(Locale.US, "%.1fM/s", bytesPerSecond / (1024.0 * 1024.0))
}
