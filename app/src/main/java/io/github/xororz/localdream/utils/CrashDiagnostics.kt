package io.github.xororz.localdream.utils

import android.app.ActivityManager
import android.app.ApplicationExitInfo
import android.content.Context
import android.os.Build
import android.os.Debug
import android.os.Process
import android.util.Log
import io.github.xororz.localdream.BuildConfig
import java.io.BufferedReader
import java.io.File
import java.io.InputStreamReader
import java.io.PrintWriter
import java.io.StringWriter
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

object CrashDiagnostics {
    private const val TAG = "CrashDiagnostics"
    private const val FILE_NAME = "runtime-diagnostics.log"
    private const val TRACE_FILE_NAME = "last-exit-trace.bin"
    private const val MAX_BYTES = 4L * 1024L * 1024L
    private const val KEEP_CHARS = 2 * 1024 * 1024
    private const val PREFS = "diagnostic_state"
    private const val LAST_EXIT_TS = "last_exit_timestamp"

    private val lock = Any()
    private val formatter = SimpleDateFormat("yyyy-MM-dd HH:mm:ss.SSS", Locale.US)
    @Volatile private var installed = false

    fun install(context: Context) {
        if (installed) return
        synchronized(lock) {
            if (installed) return
            installed = true
            val app = context.applicationContext
            val previous = Thread.getDefaultUncaughtExceptionHandler()
            Thread.setDefaultUncaughtExceptionHandler { thread, throwable ->
                runCatching {
                    record(
                        app,
                        "JAVA_CRASH",
                        "Uncaught exception thread=${thread.name} id=${thread.id}",
                        throwable,
                    )
                }
                previous?.uncaughtException(thread, throwable)
            }
            record(
                app,
                "SESSION",
                "start app=${BuildConfig.VERSION_NAME}(${BuildConfig.VERSION_CODE}) " +
                    "commit=${BuildConfig.GIT_SHA.take(12)} pid=${Process.myPid()} " +
                    "device=${Build.MANUFACTURER} ${Build.MODEL} sdk=${Build.VERSION.SDK_INT}",
            )
        }
    }

    fun record(
        context: Context,
        source: String,
        message: String,
        throwable: Throwable? = null,
    ) {
        val entry = buildString {
            append(formatter.format(Date()))
            append(" [")
            append(source)
            append("] ")
            append(message.take(24_000))
            if (throwable != null) {
                append('\n')
                val sw = StringWriter()
                throwable.printStackTrace(PrintWriter(sw))
                append(sw.toString().take(96_000))
            }
            if (!endsWith("\n")) append('\n')
        }
        synchronized(lock) {
            val f = file(context)
            runCatching {
                f.parentFile?.mkdirs()
                f.appendText(entry)
                trimIfNeeded(f)
            }.onFailure { Log.w(TAG, "Could not persist diagnostics", it) }
        }
    }

    fun recordBackendLine(context: Context, line: String) {
        record(context, "BACKEND", line)
    }

    fun recordPreviousExits(context: Context) {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.R) return
        val app = context.applicationContext
        val manager = app.getSystemService(ActivityManager::class.java) ?: return
        val prefs = app.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
        val lastSeen = prefs.getLong(LAST_EXIT_TS, 0L)
        val exits = runCatching {
            manager.getHistoricalProcessExitReasons(app.packageName, 0, 8)
        }.getOrElse {
            record(app, "EXIT", "Failed to query ApplicationExitInfo", it)
            return
        }

        var newest = lastSeen
        exits.sortedBy { it.timestamp }.forEach { info ->
            if (info.timestamp <= lastSeen) return@forEach
            newest = maxOf(newest, info.timestamp)
            record(app, "EXIT", formatExitInfo(info))
            runCatching {
                info.traceInputStream?.use { input ->
                    val trace = File(app.filesDir, TRACE_FILE_NAME)
                    trace.outputStream().use { output -> input.copyTo(output) }
                    record(app, "EXIT_TRACE", "saved ${trace.length()} bytes")
                }
            }.onFailure {
                record(app, "EXIT_TRACE", "trace read failed: ${it.message}")
            }
        }
        if (newest > lastSeen) {
            prefs.edit().putLong(LAST_EXIT_TS, newest).apply()
        }
    }

    fun fullReport(context: Context): String {
        val app = context.applicationContext
        val manager = app.getSystemService(ActivityManager::class.java)
        val mem = ActivityManager.MemoryInfo()
        manager?.getMemoryInfo(mem)
        val pssBytes = runCatching { Debug.getPss() * 1024L }.getOrDefault(0L)

        return buildString {
            appendLine("===== Local Dream diagnostics =====")
            appendLine("app=${BuildConfig.VERSION_NAME} (${BuildConfig.VERSION_CODE})")
            appendLine("commit=${BuildConfig.GIT_SHA}")
            appendLine("repo=${BuildConfig.GIT_REPOSITORY}")
            appendLine("device=${Build.MANUFACTURER} ${Build.MODEL} / ${Build.DEVICE}")
            appendLine(
                "soc=" + if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
                    Build.SOC_MODEL
                } else {
                    "unknown"
                },
            )
            appendLine("android=${Build.VERSION.RELEASE} sdk=${Build.VERSION.SDK_INT}")
            appendLine("pid=${Process.myPid()} processPss=${formatBytes(pssBytes)}")
            if (manager != null) {
                appendLine(
                    "ramAvail=${formatBytes(mem.availMem)} " +
                        "ramTotal=${formatBytes(mem.totalMem)} lowMemory=${mem.lowMemory}",
                )
            }
            appendLine()
            appendLine("===== Persistent runtime / crash log =====")
            appendLine(read(app))
            appendLine()
            appendLine("===== Download diagnostics =====")
            appendLine(DownloadDiagnostics.read(app))
            appendLine()
            appendLine("===== Current process logcat =====")
            appendLine(currentLogcat())
            LogCapture.lastCapturedLogs.value?.takeIf { it.isNotBlank() }?.let {
                appendLine()
                appendLine("===== Explicit generation capture =====")
                appendLine(it)
            }
            val trace = File(app.filesDir, TRACE_FILE_NAME)
            if (trace.isFile) {
                appendLine()
                appendLine("===== Previous system trace =====")
                appendLine("traceBytes=${trace.length()} (preserved in app diagnostics storage)")
            }
        }
    }

    fun read(context: Context): String = synchronized(lock) {
        val f = file(context)
        if (!f.isFile) {
            "No persistent runtime diagnostics yet."
        } else {
            runCatching { f.readText() }
                .getOrElse { "Could not read diagnostics: ${it.message}" }
        }
    }

    fun clear(context: Context) {
        synchronized(lock) {
            runCatching { file(context).delete() }
            runCatching { File(context.applicationContext.filesDir, TRACE_FILE_NAME).delete() }
        }
    }

    private fun formatExitInfo(info: ApplicationExitInfo): String = buildString {
        append("process=${info.processName} ")
        append("reason=${reasonName(info.reason)}(${info.reason}) ")
        append("status=${info.status} importance=${info.importance} ")
        append("pss=${formatBytes(info.pss * 1024L)} ")
        append("rss=${formatBytes(info.rss * 1024L)} ")
        append("timestamp=${formatter.format(Date(info.timestamp))}")
        info.description?.takeIf { it.isNotBlank() }?.let {
            append(" description=")
            append(it)
        }
    }

    private fun reasonName(reason: Int): String = when (reason) {
        ApplicationExitInfo.REASON_EXIT_SELF -> "EXIT_SELF"
        ApplicationExitInfo.REASON_SIGNALED -> "SIGNALED"
        ApplicationExitInfo.REASON_LOW_MEMORY -> "LOW_MEMORY"
        ApplicationExitInfo.REASON_CRASH -> "JAVA_CRASH"
        ApplicationExitInfo.REASON_CRASH_NATIVE -> "NATIVE_CRASH"
        ApplicationExitInfo.REASON_ANR -> "ANR"
        ApplicationExitInfo.REASON_INITIALIZATION_FAILURE -> "INIT_FAILURE"
        ApplicationExitInfo.REASON_PERMISSION_CHANGE -> "PERMISSION_CHANGE"
        ApplicationExitInfo.REASON_EXCESSIVE_RESOURCE_USAGE -> "EXCESSIVE_RESOURCE"
        ApplicationExitInfo.REASON_USER_REQUESTED -> "USER_REQUESTED"
        ApplicationExitInfo.REASON_USER_STOPPED -> "USER_STOPPED"
        ApplicationExitInfo.REASON_DEPENDENCY_DIED -> "DEPENDENCY_DIED"
        ApplicationExitInfo.REASON_OTHER -> "OTHER"
        else -> "UNKNOWN"
    }

    private fun currentLogcat(): String = runCatching {
        val pid = Process.myPid()
        val proc = Runtime.getRuntime().exec(
            arrayOf("logcat", "-d", "--pid=$pid", "-v", "threadtime", "-t", "3500"),
        )
        BufferedReader(InputStreamReader(proc.inputStream)).use { it.readText() }
    }.getOrElse { "logcat snapshot unavailable: ${it.message}" }

    private fun trimIfNeeded(f: File) {
        if (f.length() <= MAX_BYTES) return
        val tail = f.readText().takeLast(KEEP_CHARS)
        val newline = tail.indexOf('\n')
        f.writeText(if (newline >= 0) tail.substring(newline + 1) else tail)
    }

    private fun file(context: Context): File =
        File(context.applicationContext.filesDir, FILE_NAME)

    private fun formatBytes(bytes: Long): String {
        if (bytes <= 0L) return "0 B"
        val mib = bytes / (1024.0 * 1024.0)
        return if (mib >= 1024.0) {
            String.format(Locale.US, "%.2f GB", mib / 1024.0)
        } else {
            String.format(Locale.US, "%.0f MB", mib)
        }
    }
}
