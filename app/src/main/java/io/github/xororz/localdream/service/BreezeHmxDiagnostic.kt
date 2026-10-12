package io.github.xororz.localdream.service

import android.content.Context
import io.github.xororz.localdream.utils.BackendDiagnostics
import java.io.File
import java.util.concurrent.TimeUnit

/**
 * Diagnostic-only runner for SM8850/v81 HMX producer/consumer correctness.
 *
 * Each backend is tested in a fresh native process. No setting is persisted,
 * and production generation remains DSPQueue/HVX-only with the original CFG.
 */
object BreezeHmxDiagnostic {
    data class Result(val passed: Boolean, val message: String)

    fun logFile(context: Context): File =
        File(context.filesDir, "hmx_diagnostics/latest.txt")

    private fun appendRunLog(context: Context, label: String, lines: List<String>) {
        val file = logFile(context)
        file.parentFile?.mkdirs()
        file.appendText(
            "\n===== " + label + " =====\n" +
                lines.joinToString("\n") + "\n"
        )
    }

    private enum class Engine(val label: String, val nhmx: String, val matmul: String) {
        HVX("HVX reference", "0", "1"),
        HMX("HMX experimental", "1", "2"),
    }

    private data class ProbeRun(
        val engine: Engine,
        val complete: Boolean,
        val errors: String?,
        val hmxUnits: Int?,
        val samples: Map<String, Double>,
    ) {
        val passed: Boolean get() = complete && errors == null && samples.size == 16
    }

    fun run(context: Context): Result {
        val app = context.applicationContext
        val log = logFile(app)
        log.parentFile?.mkdirs()
        log.writeText(
            "Local Dream SM8850/v81 HMX correctness diagnostic\n" +
                "Production backend: DSPQueue/HVX (unchanged)\n" +
                "Build: 28ceab1; tests: HVX reference, unfused HMX, " +
                "HMX with safe fusion\n"
        )
        val runner = File(app.applicationInfo.nativeLibraryDir, "libbreeze_selftest.so")
        if (!runner.isFile) return Result(false, "The Breeze native self-test is missing.")

        val runtime = File(app.filesDir, "runtime_breeze_htp")
        // The button is only available after Breeze reports Ready. Reuse its
        // installed, version-checked DSP runtime instead of duplicating the
        // install/extraction logic in another subsystem.
        val skel = File(runtime, "libggml-htp-v81.so")
        if (!skel.isFile || skel.length() < 100_000L) {
            return Result(false, "Load Breeze first; its Hexagon runtime is not installed.")
        }

        val baseline = execute(app, runner, runtime, Engine.HVX)
        if (!baseline.passed) {
            return Result(false, "HVX reference probe failed: ${baseline.errors ?: "incomplete"}")
        }
        val hmx = execute(app, runner, runtime, Engine.HMX, fused = false)
        if (!hmx.passed) {
            return Result(false, "Unfused HMX failed: ${hmx.errors ?: "incomplete"}; production HVX is unchanged.")
        }
        // Validate the normal optimizer configuration too. The proven-bad
        // v81 HMX MUL_MAT+ADD epilogue is now rejected by the upstream-style
        // backend eligibility check; other permitted graph fusions still run.
        val fusedHmx = execute(app, runner, runtime, Engine.HMX, fused = true)
        if (!fusedHmx.passed) {
            return Result(false, "Safe-fusion HMX failed: ${fusedHmx.errors ?: "incomplete"}; production HVX is unchanged.")
        }
        if (hmx.hmxUnits == null || hmx.hmxUnits <= 0 ||
            fusedHmx.hmxUnits == null || fusedHmx.hmxUnits <= 0) {
            return Result(
                false,
                "Numerical parity passed, but the native backend did not report " +
                    "available HMX hardware. Acceleration is unverified; HVX stays active.",
            )
        }

        // These are warmed *isolated synthetic chains*, never TTS throughput.
        // Only compare identical shapes, never total test-process wall time.
        val ratios = baseline.samples.mapNotNull { (shape, hvxMs) ->
            val hmxMs = hmx.samples[shape] ?: return@mapNotNull null
            if (hvxMs > 0.0 && hmxMs > 0.0) hvxMs / hmxMs else null
        }.sorted()
        val median = ratios.getOrNull(ratios.size / 2)
        val comparison = if (ratios.size == 16 && median != null) {
            "Median isolated chain HMX/HVX speed ratio: " +
                String.format(java.util.Locale.US, "%.2fx", median) + ". "
        } else {
            "Speed comparison unavailable. "
        }

        return Result(
            true,
            "Numerical parity passed in HVX, HMX, and safe-fusion HMX modes (16 shapes each), " +
                "HMX units reported: ${hmx.hmxUnits}. " +
                comparison +
                "Synthetic microbenchmark only; production remains HVX until " +
                "actual HMX operator utilization, full audio quality, and sustained " +
                "throughput are validated.",
        )
    }

    private fun execute(
        context: Context,
        executable: File,
        runtime: File,
        engine: Engine,
        fused: Boolean = false,
    ): ProbeRun {
        val dspPath = listOf(
            runtime.absolutePath,
            "/vendor/lib/rfsa/adsp",
            "/vendor/dsp/cdsp",
            "/dsp",
        ).joinToString(";")
        val process = ProcessBuilder(executable.absolutePath).apply {
            directory(runtime)
            redirectErrorStream(true)
            environment().putAll(
                mapOf(
                    "LD_LIBRARY_PATH" to listOf(
                        context.applicationInfo.nativeLibraryDir,
                        runtime.absolutePath,
                        "/system/lib64",
                        "/vendor/lib64",
                    ).joinToString(":"),
                    "ADSP_LIBRARY_PATH" to dspPath,
                    "DSP_LIBRARY_PATH" to dspPath,
                    "GGML_HEXAGON_DEVICES" to "HTP0:0",
                    "GGML_HEXAGON_NHMX" to engine.nhmx,
                    "GGML_HEXAGON_NHVX" to "0",
                    "GGML_HEXAGON_MM_SELECT" to engine.matmul,
                    "GGML_HEXAGON_FA_SELECT" to "1",
                    "GGML_HEXAGON_GDN_SELECT" to "1",
                    "GGML_HEXAGON_OPFUSION" to (if (fused) "1" else "0"),
                    "GGML_HEXAGON_VERBOSE" to "1",
                    "BREEZE_HMX_DEPENDENCY_PROBE" to "1",
                ),
            )
        }.start()

        val lines = mutableListOf<String>()
        val reader = Thread({
            runCatching {
                process.inputStream.bufferedReader().useLines { stream ->
                    stream.forEach { line ->
                        synchronized(lines) {
                            if (lines.size < 2500 && (
                                line.contains("BREEZE_") ||
                                line.contains("ggml-hex", ignoreCase = true) ||
                                line.contains("hmx", ignoreCase = true) ||
                                line.contains("failed", ignoreCase = true) ||
                                line.contains("error", ignoreCase = true)
                            )) lines.add(line.take(1600))
                        }
                    }
                }
            }
        }, "breeze-hmx-diagnostic-reader").apply {
            isDaemon = true
            start()
        }
        val exited = process.waitFor(90L, TimeUnit.SECONDS)
        if (!exited) {
            process.destroyForcibly()
            process.waitFor(3L, TimeUnit.SECONDS)
        }
        reader.join(2500L)
        val transcript = synchronized(lines) { lines.toList() }
        appendRunLog(context, engine.label + " fusion=" + if (fused) "1" else "0", transcript)
        transcript
            .filter { line ->
                line.contains("BREEZE_HMX_CHAIN") ||
                    line.contains("BREEZE_SELFTEST") ||
                    line.contains("hwinfo:") ||
                    line.contains("failed", ignoreCase = true)
            }
            .forEach {
                BackendDiagnostics.append(
                    context, "BREEZE_HMX_TEST",
                    "${engine.label} fusion=${if (fused) 1 else 0}: " + it
                )
            }

        val failed = transcript.lastOrNull {
            it.contains("[BREEZE_SELFTEST] failed") ||
                it.contains("mismatch") ||
                it.contains("non-finite")
        }
        val error = when {
            !exited -> "timeout"
            process.exitValue() != 0 -> failed?.take(260) ?: "exit=${process.exitValue()}"
            else -> null
        }
        val hwinfo = Regex("""hwinfo:.*\bhmx\s+(\d+)""")
            .find(transcript.joinToString("\n"))?.groupValues?.get(1)?.toIntOrNull()
        val bench = Regex(
            """\[BREEZE_HMX_CHAIN_BENCH\] rows=(\d+) variant=(\d+) replay_ms=([0-9.]+)"""
        )
        val samples = transcript.mapNotNull { line ->
            val match = bench.find(line) ?: return@mapNotNull null
            "${match.groupValues[1]}:${match.groupValues[2]}" to
                (match.groupValues[3].toDoubleOrNull() ?: return@mapNotNull null)
        }.toMap()
        return ProbeRun(
            engine,
            error == null &&
                transcript.any { it.contains("[BREEZE_HMX_CHAIN] complete 16 cases") } &&
                transcript.any { it.contains("[BREEZE_SELFTEST] all-ok") },
            error,
            hwinfo,
            samples,
        )
    }
}
