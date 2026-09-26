package io.github.xororz.localdream.utils

import android.content.Context
import android.os.Build
import io.github.xororz.localdream.BuildConfig
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * Persistent, session-scoped log for native backend processes.
 *
 * This is intentionally separate from CrashDiagnostics' generation log:
 * backend startup happens before a generation session exists, and rotating a
 * generation log must never erase the FastRPC / Hexagon lines that explain why
 * HTP failed to start.
 */
object BackendDiagnostics {
    private const val FILE_NAME = "native_backend.log"
    private const val MAX_BYTES = 2_000_000L
    private val lock = Any()
    private val formatter = SimpleDateFormat("yyyy-MM-dd HH:mm:ss.SSS", Locale.US)

    private fun file(context: Context): File =
        File(context.applicationContext.filesDir, FILE_NAME)

    fun beginSession(context: Context, label: String) = synchronized(lock) {
        val f = file(context)
        runCatching {
            f.parentFile?.mkdirs()
            f.writeText(
                buildString {
                    appendLine("===== Local Dream native backend =====")
                    appendLine("started=${formatter.format(Date())}")
                    appendLine("app=${BuildConfig.VERSION_NAME} (${BuildConfig.VERSION_CODE})")
                    appendLine("commit=${BuildConfig.GIT_SHA}")
                    appendLine("device=${Build.MANUFACTURER} ${Build.MODEL}")
                    appendLine("android=${Build.VERSION.RELEASE} sdk=${Build.VERSION.SDK_INT}")
                    appendLine("soc=${Build.SOC_MODEL}")
                    appendLine("session=$label")
                    appendLine()
                },
            )
        }
    }

    fun append(context: Context, tag: String, message: String) = synchronized(lock) {
        val clean = message
            .replace("\u0000", "")
            .trimEnd()
        if (clean.isBlank()) return@synchronized

        val f = file(context)
        runCatching {
            f.parentFile?.mkdirs()
            if (!f.exists()) beginSession(context, "unscoped")
            f.appendText(
                "${formatter.format(Date())} [$tag] $clean\n",
            )
            trimIfNeeded(f)
        }
    }

    fun appendThrowable(context: Context, tag: String, throwable: Throwable) {
        append(
            context,
            tag,
            buildString {
                append(throwable.javaClass.simpleName)
                throwable.message?.let { append(": $it") }
                throwable.stackTrace.take(12).forEach { frame ->
                    append("\n  at $frame")
                }
            },
        )
    }

    fun read(context: Context): String = synchronized(lock) {
        val f = file(context)
        if (!f.isFile) {
            "No native backend log yet. Start a model once and the complete " +
                "server / Hexagon output will appear here."
        } else {
            runCatching { f.readText() }
                .getOrElse { "Could not read native backend log: ${it.message}" }
        }
    }

    fun clear(context: Context) = synchronized(lock) {
        runCatching { file(context).delete() }
    }

    private fun trimIfNeeded(f: File) {
        if (f.length() <= MAX_BYTES) return
        val bytes = f.readBytes()
        val keep = bytes.copyOfRange(
            (bytes.size - MAX_BYTES.toInt()).coerceAtLeast(0),
            bytes.size,
        )
        f.writeText(
            "===== older backend lines trimmed =====\n" +
                keep.toString(Charsets.UTF_8),
        )
    }
}
