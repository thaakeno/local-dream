package io.github.xororz.localdream.service

import android.content.Context

/**
 * Separate opt-in experimental generator algorithm.
 *
 * OFF means the unchanged device-proven DepthRunner codepath. The switch is
 * independent of Breeze CFG and QNN model selection. It takes effect when the
 * Breeze native process is restarted, as controlled by the generation panel.
 */
object BreezePerformanceMode {
    private const val PREFS = "breeze_performance_mode"
    private const val KEY_EXPERIMENTAL_RVQ = "experimental_rvq"

    fun isEnabled(context: Context): Boolean =
        context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
            .getBoolean(KEY_EXPERIMENTAL_RVQ, false)

    fun setEnabled(context: Context, enabled: Boolean) {
        context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
            .edit()
            .putBoolean(KEY_EXPERIMENTAL_RVQ, enabled)
            .apply()
    }
}
