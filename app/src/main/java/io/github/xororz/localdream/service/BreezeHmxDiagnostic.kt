package io.github.xororz.localdream.service

import android.content.Context
import io.github.xororz.localdream.utils.BackendDiagnostics
import java.io.File
import java.util.concurrent.TimeUnit

/**
 * Isolated SM8850 Hexagon v81 HMX correctness validation.
 *
 * Production decoding always keeps GGML_HEXAGON_NHMX=0. This runner creates
 * an entirely separate subprocess after the caller stops the speech service.
 * A failed parity test never changes a persisted setting or enables HMX.
 */
object BreezeHmxDiagnostic {
    data class Result(val passed: Boolean, val message: String)

    fun run(context: Context): Result {
        val app = context.applicationContext
        val native = File(app.applicationInfo.nativeLibraryDir, "libbreeze_selftest.so")
        val runtime = File(app.filesDir, "runtime_breeze_htp")
        val skel = File(runtime, "libggml-htp-v81.so")
        if (!native.isFile) return Result(false, "Native Breeze self-test is missing.")
        runtime.mkdirs()
        if (!skel.isFile || skel.length() < 100_000L) {
            app.assets.open("breezelibs/libggml-htp-v81.so").use { source ->
                skel.outputStream().use { destination -> source.copyTo(destination) }
            }
        }
        skel.setReadable(true, true)

        val dspPaths = listOf(
            runtime.absolutePath,
            "/vendor/lib/rfsa/adsp",
            "/vendor/dsp/cdsp",
            "/dsp",
        ).joinToString(";")
        val proc = ProcessBuilder(native.absolutePath).apply {
            directory(runtime)
            redirectErrorStream(true)
            environment().putAll(
                mapOf(
                    "LD_LIBRARY_PATH" to listOf(
                        app.applicationInfo.nativeLibraryDir,
                        runtime.absolutePath,
                        "/system/lib64",
                        "/vendor/lib64",
                    ).joinToString(":"),
                    "ADSP_LIBRARY_PATH" to dspPaths,
                    "DSP_LIBRARY_PATH" to dspPaths,
                    "GGML_HEXAGON_DEVICES" to "HTP0:0",
                    "GGML_HEXAGON_NHMX" to "1",
                    "GGML_HEXAGON_NHVX" to "0",
                    "GGML_HEXAGON_MM_SELECT" to "2",
                    "GGML_HEXAGON_FA_SELECT" to "1",
                    "GGML_HEXAGON_GDN_SELECT" to "1",
                    "GGML_HEXAGON_OPFUSION" to "0",
                    "BREEZE_HMX_DEPENDENCY_PROBE" to "1",
                ),
            )
        }.start()

        val output = StringBuffer()
        val reader = Thread({
            runCatching {
                proc.inputStream.bufferedReader().useLines { lines ->
                    lines.forEach { line ->
                        synchronized(output) {
                            if (output.length < 180_000) output.append(line.take(1600)).append('\n')
                        }
                    }
                }
            }
        }, "breeze-hmx-probe-reader").apply {
            isDaemon = true
            start()
        }
        val exited = proc.waitFor(90L, TimeUnit.SECONDS)
        if (!exited) proc.destroyForcibly()
        reader.join(1800)
        val transcript = synchronized(output) { output.toString() }
        transcript.lineSequence()
            .filter { line ->
                line.contains("BREEZE_HMX_CHAIN") ||
                    line.contains("BREEZE_SELFTEST") ||
                    line.contains("hwinfo:") ||
                    line.contains("failed", ignoreCase = true)
            }
            .forEach { BackendDiagnostics.append(app, "BREEZE_HMX_TEST", it.take(1600)) }

        if (!exited) return Result(false, "HMX probe timed out; normal HVX restored.")
        val complete = "[BREEZE_HMX_CHAIN] complete 16 cases" in transcript
        val passed = proc.exitValue() == 0 && complete &&
            "[BREEZE_SELFTEST] all-ok" in transcript
        if (!passed) {
            val detail = transcript.lineSequence()
                .filter { "failed:" in it || "mismatch" in it || "non-finite" in it }
                .lastOrNull()?.take(250)
                ?: "exit=${proc.exitValue()}; see Backend log"
            return Result(false, "HMX failed numerical validation: $detail. HVX remains enabled.")
        }
        return Result(
            true,
            "Dependency-chain numerical checks passed with HMX requested. " +
                "Inspect the Backend log for HMX utilization and timings; " +
                "production generation remains on HVX.",
        )
    }
}
