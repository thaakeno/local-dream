package io.github.xororz.localdream.utils

import android.content.Context
import android.os.Build
import io.github.xororz.localdream.BuildConfig
import java.io.File
import java.io.OutputStream
import java.io.RandomAccessFile
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * Unabridged process-session log. Every model startup archives the previous
 * session, and all generations within the current process append to one file.
 * Only the in-app preview and Android clipboard are truncated, never the log.
 */
object BackendDiagnostics {
    private const val FILE_NAME = "native_backend.log"
    private const val ARCHIVE_DIR = "native_backend_sessions"
    private const val MAX_ARCHIVED_SESSIONS = 12
    private val lock = Any()
    private val formatter = SimpleDateFormat("yyyy-MM-dd HH:mm:ss.SSS", Locale.US)

    private fun file(context: Context): File =
        File(context.applicationContext.filesDir, FILE_NAME)

    private fun archivesDir(context: Context): File =
        File(context.applicationContext.filesDir, ARCHIVE_DIR)

    private fun listArchiveFiles(context: Context): List<File> =
        archivesDir(context).listFiles { f ->
            f.isFile && f.name.startsWith("session_") && f.name.endsWith(".log")
        }?.sortedByDescending { it.name } ?: emptyList()

    fun archivedSessions(context: Context): List<String> = synchronized(lock) {
        listArchiveFiles(context).map { it.name }
    }

    private fun sessionFile(context: Context, archivedName: String?): File {
        if (archivedName == null) return file(context)
        return listArchiveFiles(context).firstOrNull { it.name == archivedName }
            ?: throw java.io.FileNotFoundException("Backend log session unavailable")
    }

    private fun newSessionHeader(context: Context, label: String) = buildString {
        appendLine("===== Local Dream native backend =====")
        appendLine("started=${formatter.format(Date())}")
        appendLine("app=${BuildConfig.VERSION_NAME} (${BuildConfig.VERSION_CODE})")
        appendLine("commit=${BuildConfig.GIT_SHA}")
        appendLine("device=${Build.MANUFACTURER} ${Build.MODEL}")
        appendLine("android=${Build.VERSION.RELEASE} sdk=${Build.VERSION.SDK_INT}")
        appendLine("soc=${if (Build.VERSION.SDK_INT >= 31) Build.SOC_MODEL else "unknown"}")
        appendLine("session=$label")
        appendLine()
    }

    fun beginSession(context: Context, label: String) = synchronized(lock) {
        val current = file(context)
        runCatching {
            current.parentFile?.mkdirs()
            var archived = true
            if (current.isFile && current.length() > 0L) {
                val directory = archivesDir(context)
                directory.mkdirs()
                val stamp = SimpleDateFormat("yyyyMMdd_HHmmss_SSS", Locale.US).format(Date())
                var index = 0
                var target: File
                do {
                    target = File(directory, "session_${stamp}_${index++}.log")
                } while (target.exists())
                archived = current.renameTo(target)
                if (!archived) {
                    archived = runCatching {
                        current.copyTo(target, overwrite = false)
                        current.delete()
                    }.getOrDefault(false)
                }
                if (archived) {
                    listArchiveFiles(context).drop(MAX_ARCHIVED_SESSIONS)
                        .forEach { it.delete() }
                }
            }
            val header = newSessionHeader(context, label)
            if (archived) current.writeText(header)
            else current.appendText("\n===== archival failed; preceding session retained =====\n" + header)
        }
    }

    fun append(context: Context, tag: String, message: String) = synchronized(lock) {
        val clean = message.replace("\u0000", "").trimEnd()
        if (clean.isBlank()) return@synchronized
        val f = file(context)
        runCatching {
            f.parentFile?.mkdirs()
            if (!f.exists()) beginSession(context, "unscoped")
            f.appendText("${formatter.format(Date())} [$tag] $clean\n")
        }
    }

    fun appendThrowable(context: Context, tag: String, throwable: Throwable) {
        append(context, tag, buildString {
            append(throwable.javaClass.simpleName)
            throwable.message?.let { append(": $it") }
            throwable.stackTrace.take(12).forEach { frame ->
                append("\n  at $frame")
            }
        })
    }

    fun read(context: Context): String = synchronized(lock) {
        val f = file(context)
        if (!f.isFile) "No native backend log yet. Start a model first."
        else runCatching { f.readText() }
            .getOrElse { "Could not read backend log: ${it.message}" }
    }

    fun readPreview(
        context: Context,
        archivedName: String? = null,
        maxChars: Int = 180_000,
    ): String = synchronized(lock) {
        runCatching {
            val f = sessionFile(context, archivedName)
            if (!f.isFile) return@synchronized "No native backend log yet."
            val maxBytes = maxChars.toLong() * 3L
            RandomAccessFile(f, "r").use { input ->
                val size = input.length()
                val start = (size - maxBytes).coerceAtLeast(0L)
                input.seek(start)
                val bytes = ByteArray((size - start).toInt())
                input.readFully(bytes)
                val tail = bytes.toString(Charsets.UTF_8).takeLast(maxChars)
                if (start > 0) "… preview only; Save exports the full session …\n\n" + tail
                else tail
            }
        }.getOrElse { "Could not read backend log: ${it.message}" }
    }

    fun exportTo(context: Context, destination: OutputStream, archivedName: String? = null) =
        synchronized(lock) {
            sessionFile(context, archivedName).inputStream().use { source ->
                source.copyTo(destination)
            }
        }

    fun clear(context: Context) = synchronized(lock) {
        runCatching { file(context).delete() }
    }
}
