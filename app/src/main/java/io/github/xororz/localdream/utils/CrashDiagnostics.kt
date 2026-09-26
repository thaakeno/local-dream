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
    private const val GENERATION_FILE_NAME = "generation-diagnostics.log"
    private const val TRACE_FILE_NAME = "last-exit-trace.bin"
    private const val MAX_BYTES = 4L * 1024L * 1024L
    private const val GENERATION_MAX_BYTES = 6L * 1024L * 1024L
    private const val KEEP_CHARS = 2 * 1024 * 1024
    private const val GENERATION_KEEP_CHARS = 3 * 1024 * 1024
    private const val GENERATION_FLUSH_CHARS = 64 * 1024
    private const val PREFS = "diagnostic_state"
    private const val LAST_EXIT_TS = "last_exit_timestamp"
    private const val PENDING_CRASH = "pending_crash_report"
    private const val PENDING_CRASH_TS = "pending_crash_timestamp"

    private val lock = Any()
    private val formatter = SimpleDateFormat("yyyy-MM-dd HH:mm:ss.SSS", Locale.US)
    @Volatile private var installed = false
    // Backend stdout can produce thousands of lines while HTP weights are being
    // loaded. Writing every line with File.appendText() competes with model I/O
    // and makes the phone jank, so batch those writes in memory and flush at
    // phase/error boundaries or once the batch reaches a modest size.
    private val generationPending = StringBuilder()

    fun install(context: Context) {
        if (installed) return
        synchronized(lock) {
            if (installed) return
            installed = true
            val app = context.applicationContext
            val previous = Thread.getDefaultUncaughtExceptionHandler()
            Thread.setDefaultUncaughtExceptionHandler { thread, throwable ->
                runCatching {
                    markCrashPending(app, System.currentTimeMillis())
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
        // Backend output is the detailed generation trace. Keep it in its own
        // rolling file so crash diagnostics stay readable while every HTP
        // allocation, graph split and stage transition is still preserved.
        recordGeneration(context, "BACKEND", line)
    }

    fun recordGeneration(
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
            generationPending.append(entry)
            val flushNow = throwable != null ||
                source == "PHASE" ||
                source == "COMPLETE" ||
                source == "ERROR" ||
                source == "CANCELLED" ||
                source == "BACKEND_EXIT" ||
                generationPending.length >= GENERATION_FLUSH_CHARS
            if (flushNow) {
                flushGenerationLocked(context)
            }
        }
    }

    fun flushGeneration(context: Context) {
        synchronized(lock) {
            flushGenerationLocked(context)
        }
    }

    fun recordGenerationTelemetry(context: Context, label: String) {
        runCatching {
            val snapshot = GenerationTelemetry.sample(context)
            recordGeneration(
                context,
                "TELEMETRY",
                buildString {
                    append(label)
                    append(" pss=")
                    append(GenerationTelemetry.formatBytes(snapshot.processRamBytes))
                    append(" ramAvail=")
                    append(GenerationTelemetry.formatBytes(snapshot.availableRamBytes))
                    append("/")
                    append(GenerationTelemetry.formatBytes(snapshot.totalRamBytes))
                    append(" battery=")
                    append(snapshot.batteryPercent)
                    append("% temp=")
                    append(snapshot.batteryTempC?.let { String.format(Locale.US, "%.1fC", it) } ?: "unknown")
                    append(" thermal=")
                    append(snapshot.thermalStatus)
                },
            )
        }
    }

    fun beginGenerationSession(context: Context, label: String) {
        synchronized(lock) {
            generationPending.clear()
            runCatching { generationFile(context).delete() }
        }
        // Each generation/preload gets its own file. This deliberately drops
        // logs from earlier runs so a YuE2 crash report cannot be polluted by
        // some unrelated image generation from hours ago.
        recordGeneration(context, "SESSION", label)
        flushGeneration(context)
    }

    fun readGeneration(context: Context): String = synchronized(lock) {
        flushGenerationLocked(context)
        val f = generationFile(context)
        if (!f.isFile) {
            "No generation diagnostics yet."
        } else {
            runCatching { f.readText() }
                .getOrElse { "Could not read generation diagnostics: ${it.message}" }
        }
    }

    fun hasPendingCrashReport(context: Context): Boolean =
        context.applicationContext.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
            .getBoolean(PENDING_CRASH, false)

    fun recoveryReport(context: Context): String {
        val app = context.applicationContext
        val prefs = app.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
        val crashTs = prefs.getLong(PENDING_CRASH_TS, 0L)
            .takeIf { it > 0L }
            ?: System.currentTimeMillis()
        val windowStart = crashTs - 5L * 60L * 1000L
        val windowEnd = crashTs + 60L * 1000L

        return buildString {
            appendLine("===== Local Dream crash recovery =====")
            appendLine("A previous Local Dream process ended abnormally.")
            appendLine("crashTime=${formatter.format(Date(crashTs))}")
            appendLine("app=${BuildConfig.VERSION_NAME} (${BuildConfig.VERSION_CODE})")
            appendLine("commit=${BuildConfig.GIT_SHA}")
            appendLine("device=${Build.MANUFACTURER} ${Build.MODEL}")
            appendLine("android=${Build.VERSION.RELEASE} sdk=${Build.VERSION.SDK_INT}")
            appendLine()
            appendLine("===== Runtime events around the crash =====")
            appendLine(
                readWindow(
                    file(app),
                    windowStart,
                    windowEnd,
                    "No runtime events recorded near this crash.",
                ),
            )
            appendLine()
            appendLine("===== Current generation / backend session =====")
            // beginGenerationSession() rotates this file, so it contains only
            // the active YuE2/image run rather than the previous six hours.
            appendLine(readGeneration(app))
            val trace = File(app.filesDir, TRACE_FILE_NAME)
            if (trace.isFile) {
                appendLine()
                appendLine("===== Previous system trace =====")
                appendLine("traceBytes=${trace.length()} (preserved in app diagnostics storage)")
            }
        }
    }

    fun acknowledgeCrashReport(context: Context) {
        context.applicationContext.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
            .edit()
            .putBoolean(PENDING_CRASH, false)
            .apply()
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
            if (isCrashLikeExit(info.reason)) {
                markCrashPending(app, info.timestamp)
            }
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
            appendLine("===== Last generation / backend log =====")
            appendLine(readGeneration(app))
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
            generationPending.clear()
            runCatching { file(context).delete() }
            runCatching { generationFile(context).delete() }
            runCatching { File(context.applicationContext.filesDir, TRACE_FILE_NAME).delete() }
        }
    }

    private fun isCrashLikeExit(reason: Int): Boolean = when (reason) {
        ApplicationExitInfo.REASON_SIGNALED,
        ApplicationExitInfo.REASON_LOW_MEMORY,
        ApplicationExitInfo.REASON_CRASH,
        ApplicationExitInfo.REASON_CRASH_NATIVE,
        ApplicationExitInfo.REASON_ANR,
        ApplicationExitInfo.REASON_INITIALIZATION_FAILURE,
        ApplicationExitInfo.REASON_EXCESSIVE_RESOURCE_USAGE,
        ApplicationExitInfo.REASON_DEPENDENCY_DIED,
        -> true
        else -> false
    }

    private fun markCrashPending(context: Context, timestamp: Long) {
        context.applicationContext.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
            .edit()
            .putBoolean(PENDING_CRASH, true)
            .putLong(PENDING_CRASH_TS, timestamp)
            .commit()
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

    private fun readWindow(
        source: File,
        startMs: Long,
        endMs: Long,
        emptyMessage: String,
    ): String {
        if (!source.isFile) return emptyMessage
        return runCatching {
            val result = StringBuilder()
            var keepCurrent = false
            val stampLength = 23 // yyyy-MM-dd HH:mm:ss.SSS
            source.forEachLine { line ->
                val timestamp = if (line.length >= stampLength) {
                    runCatching {
                        formatter.parse(line.substring(0, stampLength))?.time
                    }.getOrNull()
                } else {
                    null
                }
                if (timestamp != null) {
                    keepCurrent = timestamp in startMs..endMs
                }
                if (keepCurrent) {
                    result.appendLine(line)
                }
            }
            result.toString()
                .takeLast(180_000)
                .ifBlank { emptyMessage }
        }.getOrElse {
            "Could not read scoped diagnostics: ${it.message}"
        }
    }

    private fun currentLogcat(): String = runCatching {
        val pid = Process.myPid()
        val proc = Runtime.getRuntime().exec(
            arrayOf("logcat", "-d", "--pid=$pid", "-v", "threadtime", "-t", "3500"),
        )
        BufferedReader(InputStreamReader(proc.inputStream)).use { it.readText() }
    }.getOrElse { "logcat snapshot unavailable: ${it.message}" }

    private fun trimIfNeeded(
        f: File,
        maxBytes: Long = MAX_BYTES,
        keepChars: Int = KEEP_CHARS,
    ) {
        if (f.length() <= maxBytes) return
        val tail = f.readText().takeLast(keepChars)
        val newline = tail.indexOf('\n')
        f.writeText(if (newline >= 0) tail.substring(newline + 1) else tail)
    }

    private fun flushGenerationLocked(context: Context) {
        if (generationPending.isEmpty()) return
        val pending = generationPending.toString()
        generationPending.clear()
        val f = generationFile(context)
        runCatching {
            f.parentFile?.mkdirs()
            f.appendText(pending)
            trimIfNeeded(f, GENERATION_MAX_BYTES, GENERATION_KEEP_CHARS)
        }.onFailure {
            // Preserve the newest diagnostics for the next flush if storage is
            // temporarily unavailable, but never let the RAM buffer grow without bound.
            generationPending.append(pending.takeLast(GENERATION_FLUSH_CHARS * 2))
            Log.w(TAG, "Could not persist generation diagnostics", it)
        }
    }

    private fun file(context: Context): File =
        File(context.applicationContext.filesDir, FILE_NAME)

    private fun generationFile(context: Context): File =
        File(context.applicationContext.filesDir, GENERATION_FILE_NAME)

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
