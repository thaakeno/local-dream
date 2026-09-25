package io.github.xororz.localdream.utils

import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context

data class ClipboardCopyResult(
    val copied: Boolean,
    val truncated: Boolean,
)

object SafeClipboard {
    // Android's Binder transaction buffer is 1 MiB and shared by concurrent
    // transactions. Keep clipboard payloads comfortably below that ceiling.
    private const val MAX_SAFE_CHARS = 96_000
    private const val FALLBACK_CHARS = 32_000

    fun copyText(
        context: Context,
        label: String,
        text: String,
    ): ClipboardCopyResult {
        val clipboard = context.getSystemService(ClipboardManager::class.java)
            ?: return ClipboardCopyResult(copied = false, truncated = false)

        val truncated = text.length > MAX_SAFE_CHARS
        val payload = if (truncated) {
            val tail = text.takeLast(MAX_SAFE_CHARS)
            "… full text omitted because it is too large for Android's clipboard; " +
                "save/share the file for the complete report …\n\n" + tail
        } else {
            text
        }

        if (runCatching {
                clipboard.setPrimaryClip(ClipData.newPlainText(label, payload))
            }.isSuccess
        ) {
            return ClipboardCopyResult(copied = true, truncated = truncated)
        }

        // Leave extra headroom if the process is already using Binder heavily.
        val fallback = text.takeLast(FALLBACK_CHARS)
        val fallbackPayload =
            "… clipboard fallback: only the last $FALLBACK_CHARS characters were copied; " +
                "save/share the file for the complete report …\n\n" + fallback
        val copied = runCatching {
            clipboard.setPrimaryClip(ClipData.newPlainText(label, fallbackPayload))
        }.isSuccess
        return ClipboardCopyResult(copied = copied, truncated = true)
    }
}
