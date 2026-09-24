package io.github.xororz.localdream.utils

import android.content.Context
import android.util.Log
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * Small persistent diagnostic log for model downloads.
 *
 * This intentionally records transfer decisions and errors only. It never logs
 * auth tokens, signed storage URLs, prompts, or image contents. The file is
 * bounded so enabling diagnostics cannot grow app storage indefinitely.
 */
object DownloadDiagnostics {
    private const val TAG = "DownloadDiagnostics"
    private const val FILE_NAME = "download-diagnostics.log"
    private const val MAX_FILE_BYTES = 512 * 1024L
    private const val KEEP_CHARS = 256 * 1024
    private val lock = Any()
    private val formatter = SimpleDateFormat("yyyy-MM-dd HH:mm:ss.SSS", Locale.US)

    fun info(context: Context, message: String) = append(context, "I", message, null)

    fun warn(context: Context, message: String, throwable: Throwable? = null) =
        append(context, "W", message, throwable)

    fun error(context: Context, message: String, throwable: Throwable? = null) =
        append(context, "E", message, throwable)

    fun read(context: Context): String = synchronized(lock) {
        val file = logFile(context)
        if (!file.isFile) {
            return@synchronized "No download logs yet."
        }
        runCatching { file.readText() }
            .getOrElse { "Could not read download logs: ${it.message}" }
    }

    fun clear(context: Context) {
        synchronized(lock) {
            runCatching { logFile(context).delete() }
                .onFailure { Log.w(TAG, "Could not clear download diagnostics", it) }
        }
    }

    private fun append(context: Context, level: String, message: String, throwable: Throwable?) {
        val safeMessage = message
            .replace(Regex("""(?i)(authorization|token|access[-_ ]?token)=[^&\s]+""")) { match ->
                "${match.groupValues[1]}=<redacted>"
            }
            .take(8_000)
        val line = buildString {
            append(formatter.format(Date()))
            append(" ")
            append(level)
            append(" ")
            append(safeMessage)
            if (throwable != null) {
                append(" | ")
                append(throwable::class.java.simpleName)
                append(": ")
                append(throwable.message.orEmpty().take(2_000))
            }
            append('\n')
        }

        when (level) {
            "E" -> Log.e(TAG, safeMessage, throwable)
            "W" -> Log.w(TAG, safeMessage, throwable)
            else -> Log.i(TAG, safeMessage)
        }

        synchronized(lock) {
            val file = logFile(context)
            runCatching {
                file.parentFile?.mkdirs()
                file.appendText(line)
                if (file.length() > MAX_FILE_BYTES) {
                    val text = file.readText()
                    val tail = text.takeLast(KEEP_CHARS)
                    val firstNewline = tail.indexOf('\n')
                    file.writeText(
                        if (firstNewline >= 0 && firstNewline + 1 < tail.length) {
                            tail.substring(firstNewline + 1)
                        } else {
                            tail
                        },
                    )
                }
            }.onFailure {
                Log.w(TAG, "Could not persist download diagnostics", it)
            }
        }
    }

    private fun logFile(context: Context): File =
        File(context.applicationContext.filesDir, FILE_NAME)
}
