package io.github.xororz.localdream.service

import android.app.Notification
import android.app.PendingIntent
import android.content.Context
import android.graphics.drawable.Icon
import android.net.Uri
import android.os.Bundle
import android.provider.Settings
import io.github.xororz.localdream.R
import org.json.JSONArray
import org.json.JSONObject

/**
 * Xiaomi HyperOS 3 Super Island integration for model downloads.
 *
 * Xiaomi's official client API is an extension of a normal Android notification:
 * the island payload lives in notification.extras["miui.focus.param"]. We therefore
 * decorate Local Dream's existing foreground-service notification instead of posting
 * a second notification.
 */
internal object HyperOsSuperIsland {
    private const val PARAM_KEY = "miui.focus.param"
    private const val PICS_KEY = "miui.focus.pics"
    private const val ACTIONS_KEY = "miui.focus.actions"
    private const val ICON_KEY = "miui.focus.pic_localdream"
    private const val TOGGLE_ACTION_KEY = "miui.focus.action_download_toggle"
    private const val CANCEL_ACTION_KEY = "miui.focus.action_download_cancel"

    data class Capability(
        val protocolVersion: Int,
        val hasFocusPermission: Boolean,
    ) {
        val supportsSuperIsland: Boolean
            get() = protocolVersion >= 3
    }

    fun detect(context: Context): Capability {
        val protocol = runCatching {
            Settings.System.getInt(
                context.contentResolver,
                "notification_focus_protocol",
                0,
            )
        }.getOrDefault(0)

        // Xiaomi documents this ContentProvider call as potentially slow, so callers
        // cache Capability for the lifetime of the download service.
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
        statusText: String?,
        paused: Boolean,
        pauseResumeIntent: PendingIntent,
        cancelIntent: PendingIntent,
    ) {
        if (!capability.supportsSuperIsland) return

        val percent = (progress.coerceIn(0f, 1f) * 100f).toInt()
        val compactName = modelName.take(36)
        val stateText = when {
            paused -> context.getString(R.string.download_paused)
            !statusText.isNullOrBlank() -> statusText.take(64)
            else -> context.getString(R.string.download_starting)
        }
        val accent = "#9DB7FF"
        val track = "#33415F"

        val icon = Icon.createWithResource(context, R.mipmap.ic_launcher)
        notification.extras.putBundle(
            PICS_KEY,
            Bundle().apply { putParcelable(ICON_KEY, icon) },
        )

        val toggleTitle =
            context.getString(if (paused) R.string.resume else R.string.pause)
        val toggleIcon = Icon.createWithResource(
            context,
            if (paused) android.R.drawable.ic_media_play else android.R.drawable.ic_media_pause,
        )
        val cancelIcon = Icon.createWithResource(
            context,
            android.R.drawable.ic_menu_close_clear_cancel,
        )
        notification.extras.putBundle(
            ACTIONS_KEY,
            Bundle().apply {
                putParcelable(
                    TOGGLE_ACTION_KEY,
                    Notification.Action.Builder(toggleIcon, toggleTitle, pauseResumeIntent).build(),
                )
                putParcelable(
                    CANCEL_ACTION_KEY,
                    Notification.Action.Builder(
                        cancelIcon,
                        context.getString(R.string.cancel),
                        cancelIntent,
                    ).build(),
                )
            },
        )

        val iconInfo = JSONObject()
            .put("type", 1)
            .put("pic", ICON_KEY)

        val circularProgress = JSONObject()
            .put("progress", percent)
            .put("colorReach", accent)
            .put("colorUnReach", track)
            .put("isCCW", false)

        val island = JSONObject()
            .put("islandProperty", 2)
            .put("islandOrder", false)
            .put("dismissIsland", false)
            .put("highlightColor", accent)
            .put(
                "bigIslandArea",
                JSONObject()
                    .put(
                        "imageTextInfoLeft",
                        JSONObject()
                            .put("type", 1)
                            .put("picInfo", iconInfo),
                    )
                    .put(
                        "textInfo",
                        JSONObject()
                            .put("frontTitle", context.getString(R.string.app_name))
                            .put("title", "$percent%")
                            .put("content", compactName)
                            .put("showHighlightColor", false)
                            .put("narrowFont", true),
                    )
                    .put(
                        "progressTextInfo",
                        JSONObject().put("progressInfo", circularProgress),
                    ),
            )
            .put(
                "smallIslandArea",
                JSONObject().put(
                    "combinePicInfo",
                    JSONObject()
                        .put("picInfo", iconInfo)
                        .put("progressInfo", circularProgress),
                ),
            )

        val params = JSONObject()
            // Xiaomi's public OS2/OS3 client examples use payload protocol 1;
            // notification_focus_protocol=3 is the device capability check above.
            .put("protocol", 1)
            .put("business", "download")
            .put("updatable", true)
            .put("reopen", "reopen")
            .put("enableFloat", false)
            .put("islandFirstFloat", false)
            .put("filterWhenNoPermission", false)
            .put("ticker", "$percent% · $compactName")
            .put("aodTitle", "$percent% · $compactName")
            .put(
                "chatInfo",
                JSONObject()
                    .put("picProfile", ICON_KEY)
                    .put("title", compactName)
                    .put("content", stateText),
            )
            .put(
                "progressInfo",
                JSONObject()
                    .put("progress", percent)
                    .put("colorProgress", accent)
                    .put("colorProgressEnd", accent),
            )
            .put(
                "baseInfo",
                JSONObject()
                    .put("type", 2)
                    .put("title", compactName)
                    .put("content", "$percent% · $stateText")
                    .put("colorTitle", accent),
            )
            .put(
                "hintInfo",
                JSONObject()
                    .put("type", 1)
                    .put("title", stateText)
                    .put(
                        "actionInfo",
                        JSONObject()
                            .put("action", TOGGLE_ACTION_KEY)
                            .put("actionTitle", toggleTitle)
                            .put("actionTitleColor", "#FFFFFF")
                            .put("actionBgColor", accent)
                            .put("actionIntentType", 1),
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
}
