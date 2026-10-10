package io.github.xororz.localdream.service

import android.content.Context
import java.io.File
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow

/** Per-device experimental transport choice; the stable DSPQueue is default. */
object BreezeHexagonTransport {
    private const val PREFS = "breeze_hexagon_transport"
    private const val KEY = "fastrpc_ion_enabled"
    private const val KEY_FAILURE = "fastrpc_last_error"
    private val failureMutable = MutableStateFlow<String?>(null)
    val failure: StateFlow<String?> = failureMutable

    fun lastFailure(context: Context): String? =
        failureMutable.value ?: context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
            .getString(KEY_FAILURE, null)

    fun clearFailure(context: Context) {
        context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
            .edit().remove(KEY_FAILURE).apply()
        failureMutable.value = null
    }

    fun recordFailure(context: Context, reason: String) {
        val message = reason.trim().take(1500).ifBlank { "Unknown native startup error" }
        // Reflect the actual backend: FastRPC failed and DSPQueue is active.
        context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
            .edit().putBoolean(KEY, false).putString(KEY_FAILURE, message).apply()
        failureMutable.value = message
    }

    fun isEnabled(context: Context): Boolean =
        context.getSharedPreferences(PREFS, Context.MODE_PRIVATE).getBoolean(KEY, false)

    fun setEnabled(context: Context, enabled: Boolean) {
        if (enabled) clearFailure(context)
        context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
            .edit().putBoolean(KEY, enabled).apply()
    }

    fun isPackaged(context: Context): Boolean {
        val native = File(context.applicationInfo.nativeLibraryDir)
        return File(native, "libbreeze_server_fastrpc.so").isFile &&
            File(native, "libbreeze_selftest_fastrpc.so").isFile &&
            runCatching {
                context.assets.open("breezefastrpc/libggml-htp-v81.so")
                    .use { it.available() > 65536 }
            }.getOrDefault(false)
    }
}
