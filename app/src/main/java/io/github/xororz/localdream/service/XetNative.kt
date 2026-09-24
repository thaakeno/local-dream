package io.github.xororz.localdream.service

import android.util.Log

internal object XetNative {
    private const val TAG = "XetNative"

    val available: Boolean

    init {
        available = runCatching {
            System.loadLibrary("localdream_xet")
            true
        }.getOrElse {
            Log.w(TAG, "Xet native library unavailable; normal HTTP fallback will be used", it)
            false
        }
    }

    external fun nativeDownload(
        hash: String,
        size: Long,
        refreshUrl: String,
        destPath: String,
        cacheDir: String,
        offset: Long,
        memoryBudgetBytes: Long,
        minConcurrency: Int,
        initialConcurrency: Int,
        maxConcurrency: Int,
    ): Int

    external fun nativeProgressBytes(): Long

    external fun nativeProgressTotalBytes(): Long

    external fun nativeProgressBytesPerSecond(): Long

    external fun nativeTransferBytes(): Long

    external fun nativeTransferTotalBytes(): Long

    external fun nativeTransferBytesPerSecond(): Long

    external fun nativeCancel()

    external fun nativeLastError(): String?
}
