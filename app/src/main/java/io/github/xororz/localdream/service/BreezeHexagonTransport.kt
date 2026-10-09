package io.github.xororz.localdream.service

import android.content.Context
import java.io.File

/** Per-device experimental transport choice; the stable DSPQueue is default. */
object BreezeHexagonTransport {
    private const val PREFS = "breeze_hexagon_transport"
    private const val KEY = "fastrpc_ion_enabled"

    fun isEnabled(context: Context): Boolean =
        context.getSharedPreferences(PREFS, Context.MODE_PRIVATE).getBoolean(KEY, false)

    fun setEnabled(context: Context, enabled: Boolean) {
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
