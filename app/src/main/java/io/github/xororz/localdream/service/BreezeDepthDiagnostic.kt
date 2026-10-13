package io.github.xororz.localdream.service

import android.content.Context
import io.github.xororz.localdream.data.Model
import io.github.xororz.localdream.utils.BackendDiagnostics
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.Locale
import java.util.concurrent.TimeUnit
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.sqrt

/**
 * Full-model Breeze Depth parity/latency diagnostic, isolated from normal
 * speech generation. Uses one native test executable for HVX and HMX with
 * identical preselected residual codebooks, hidden activations and CFG 3.2.
 */
object BreezeDepthDiagnostic {
    data class Result(val passed: Boolean, val message: String)

    private const val PREFS = "breeze_depth_qkv_nx"
    fun isQualified(context: Context): Boolean =
        context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
            .getBoolean("qualified", false)

    fun isEnabled(context: Context): Boolean {
        val prefs = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
        return prefs.getBoolean("qualified", false) && prefs.getBoolean("enabled", false)
    }

    fun setEnabled(context: Context, enabled: Boolean) {
        val prefs = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
        if (enabled && !prefs.getBoolean("qualified", false)) return
        prefs.edit().putBoolean("enabled", enabled).apply()
    }

    private enum class Mode(
        val label: String, val nhmx: String, val fusion: String, val groupedQkv: Boolean,
    ) {
        HVX("HVX reference", "0", "0", false),
        HVX_GROUPED("HVX grouped QKV / MUL_MAT_NX candidate", "0", "0", true),
        HMX("HMX unfused", "1", "0", false),
        HMX_SAFE("HMX safe fusion", "1", "1", false),
    }

    private data class RunData(
        val mode: Mode, val ms: Double, val coverage: Int?, val trace: List<FloatArray>,
        val nxFusions: Int,
    )

    fun logFile(context: Context) = File(context.filesDir, "depth_diagnostics/latest.txt")

    fun run(context: Context, modelId: String): Result {
        val app = context.applicationContext
        val file = logFile(app)
        file.parentFile?.mkdirs()
        file.writeText(
            "Breeze TTS 2 full Depth diagnostic | SM8850/v81\n" +
                "Real GGUF weights, 15 sequential depth steps, CFG 3.2\n" +
                "Synthetic deterministic input activations; fixed residual codebook history\n" +
                "HVX QKV NX experiment: original graph versus explicitly grouped projections\n" +
                "Five warmed repetitions per mode; median timing; real GGUF/CFG 3.2\n" +
                "Production DSPQueue/HVX is unchanged\n"
        )
        val model = File(File(Model.getModelsDir(app), modelId), "model.gguf")
        if (!model.isFile || model.length() < 2_000_000_000L) {
            return Result(false, "Breeze GGUF not installed; cannot test real Depth weights.")
        }
        val runtime = File(app.filesDir, "runtime_breeze_htp")
        if (!File(runtime, "libggml-htp-v81.so").isFile) {
            return Result(false, "Hexagon runtime is not installed. Load Breeze first.")
        }
        val executable = File(app.applicationInfo.nativeLibraryDir, "libbreeze_selftest.so")
        if (!executable.isFile) return Result(false, "Full Depth native diagnostic missing.")

        val baseline = execute(app, runtime, executable, model, Mode.HVX, file)
            ?: return Result(false, "HVX full Depth baseline failed; share the Depth log.")
        val grouped = execute(app, runtime, executable, model, Mode.HVX_GROUPED, file)
            ?: return Result(false, "Grouped QKV Depth test failed; leave production HVX unchanged and share log.")
        val hmx = execute(app, runtime, executable, model, Mode.HMX, file)
            ?: return Result(false, "Unfused HMX full Depth test failed; share the Depth log.")
        val safe = execute(app, runtime, executable, model, Mode.HMX_SAFE, file)
            ?: return Result(false, "Safe-fusion HMX full Depth test failed; share the Depth log.")

        val groupedPass = compare(file, baseline, grouped)
        val unfusedPass = compare(file, baseline, hmx)
        val safePass = compare(file, baseline, safe)
        val groupedRatio = if (grouped.ms > 0) baseline.ms / grouped.ms else 0.0
        val speedRatio = if (hmx.ms > 0) baseline.ms / hmx.ms else 0.0
        val safeRatio = if (safe.ms > 0) baseline.ms / safe.ms else 0.0
        file.appendText(
            "\nDepth model durations (excluding load/warmup): " +
                "HVX=${format(baseline.ms)} ms; grouped-QKV=${format(grouped.ms)} ms; " +
                "HMX=${format(hmx.ms)} ms; safe-fusion=${format(safe.ms)} ms\n" +
                "Speedup: grouped-QKV=${format(groupedRatio)}x, " +
                "unfused-HMX=${format(speedRatio)}x, safe-fusion=${format(safeRatio)}x\n" +
                "Reported MUL_MAT_NX fusions (grouped): ${grouped.nxFusions}\n" +
                "HMX matmul coverage (if emitted): ${hmx.coverage ?: "not reported"}\n"
        )
        val passes = groupedPass && unfusedPass && safePass
        val qualifies = passes && grouped.nxFusions >= 12 && groupedRatio >= 1.08
        // Never activate an optimization solely because it compiled. The
        // user must explicitly turn it on after the real-model on-device
        // parity + repeated-timing gate has been passed.
        val prefs = app.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
        prefs.edit().putBoolean("qualified", qualifies).apply()
        if (!qualifies) prefs.edit().putBoolean("enabled", false).apply()
        file.appendText(
            "QKV production eligibility: ${if (qualifies) "QUALIFIED" else "NOT_QUALIFIED"} " +
                "(requires parity, >=12 Nx fusion events, and >=1.08x median speedup).\n"
        )
        val message = if (!passes) {
            "Full Depth logits differ beyond diagnostic tolerance. " +
                "No production changes. Share the Depth log."
        } else {
            "All 15 Depth steps passed the numerical checks at CFG 3.2. " +
                "Experimental grouped-QKV HVX: ${format(groupedRatio)}x " +
                "(${grouped.nxFusions} logged MUL_MAT_NX fusions; " +
                "optional fast path ${if (qualifies) "qualified" else "not qualified"}). " +
                "HMX ${format(speedRatio)}x, safe fusion ${format(safeRatio)}x " +
                "(medians of five warmed frames, not sustained FPS). " +
                if ((hmx.coverage ?: 0) > 0) {
                    "HMX matmul execution reported."
                } else {
                    "Actual HMX matmul usage not yet verified."
                }
        }
        // Qualification alone never changes production. The user must enable
        // the optional grouped path and can revert to reference HVX instantly.
        return Result(qualifies, message)
    }

    private fun execute(
        context: Context,
        runtime: File,
        executable: File,
        model: File,
        mode: Mode,
        report: File,
    ): RunData? {
        val traceFile = File(report.parentFile, "depth_${mode.name.lowercase(Locale.US)}.bin")
        traceFile.delete()
        val dsp = listOf(
            runtime.absolutePath, "/vendor/lib/rfsa/adsp", "/vendor/dsp/cdsp", "/dsp"
        ).joinToString(";")
        val process = ProcessBuilder(
            executable.absolutePath, "--depth-parity",
            model.absolutePath, traceFile.absolutePath,
        ).apply {
            directory(runtime)
            redirectErrorStream(true)
            environment().putAll(
                mapOf(
                    "LD_LIBRARY_PATH" to listOf(
                        context.applicationInfo.nativeLibraryDir, runtime.absolutePath,
                        "/system/lib64", "/vendor/lib64"
                    ).joinToString(":"),
                    "ADSP_LIBRARY_PATH" to dsp,
                    "DSP_LIBRARY_PATH" to dsp,
                    "GGML_HEXAGON_DEVICES" to "HTP0:0",
                    "GGML_HEXAGON_NHMX" to mode.nhmx,
                    "GGML_HEXAGON_NHVX" to "0",
                    "GGML_HEXAGON_MM_SELECT" to (if (mode == Mode.HVX) "1" else "2"),
                    "GGML_HEXAGON_OPFUSION" to mode.fusion,
                    "GGML_HEXAGON_PROFILE" to "1",
                    "GGML_HEXAGON_VERBOSE" to (if (mode.groupedQkv) "1" else "0"),
                    "BREEZE_DEPTH_QKV_GROUP" to (if (mode.groupedQkv) "1" else "0"),
                    "BREEZE_DEPTH_PROFILE" to "1",
                )
            )
        }.start()

        val lines = mutableListOf<String>()
        val reader = Thread({
            runCatching {
                process.inputStream.bufferedReader().useLines { stream ->
                    stream.forEach { line ->
                        if (
                            line.contains("BREEZE_DEPTH") ||
                            line.contains("hwinfo:") ||
                            line.contains("mul_mat coverage:") ||
                            line.contains("fused MUL_MAT_NX") ||
                            line.contains("failed", ignoreCase = true) ||
                            line.contains("error:", ignoreCase = true)
                        ) synchronized(lines) {
                            if (lines.size < 1000) lines.add(line.take(1500))
                        }
                    }
                }
            }
        }, "breeze-depth-model-reader").apply { isDaemon = true; start() }

        val done = process.waitFor(180L, TimeUnit.SECONDS)
        if (!done) {
            process.destroyForcibly()
            process.waitFor(3L, TimeUnit.SECONDS)
        }
        reader.join(2000)
        val output = synchronized(lines) { lines.toList() }
        report.appendText("\n===== ${mode.label} =====\n" +
            output.joinToString("\n") + "\n")
        val success = done && process.exitValue() == 0 &&
            output.any { "[BREEZE_DEPTH_MODEL_TEST]" in it && "all-ok" in it }
        if (!success) return null

        val timing = Regex("""duration_ms=([0-9.]+)""")
            .find(output.joinToString("\n"))?.groupValues?.get(1)?.toDoubleOrNull()
            ?: return null
        val coverage = Regex("""mul_mat coverage:.*?hmx=(\d+)""")
            .find(output.joinToString("\n"))?.groupValues?.get(1)?.toIntOrNull()
        val nxFusions = output.count { it.contains("fused MUL_MAT_NX") }
        report.appendText("${mode.label}: NX fusion log lines=${nxFusions}\n")
        return try {
            RunData(mode, timing, coverage, loadTrace(traceFile), nxFusions)
        } catch (err: Exception) {
            report.appendText("Trace read failed: ${err.message}\n")
            null
        }
    }

    private fun loadTrace(file: File): List<FloatArray> {
        require(file.isFile && file.length() in 100L..16_000_000L) {
            "Invalid Depth trace file size"
        }
        val b = ByteBuffer.wrap(file.readBytes()).order(ByteOrder.LITTLE_ENDIAN)
        require(b.int == 0x42445054 && b.int == 1 && b.int == 15) {
            "Unexpected Depth trace format or step count"
        }
        val data = (1..15).map {
            require(b.remaining() >= 4) { "Truncated Depth trace" }
            val count = b.int
            require(count in 1..32768 && b.remaining() >= count * 4) {
                "Invalid logits length"
            }
            FloatArray(count) { b.float }
        }
        require(!b.hasRemaining()) { "Unexpected trace trailer" }
        return data
    }

    private fun compare(report: File, baseline: RunData, test: RunData): Boolean {
        var passed = true
        for (j in 0 until 15) {
            val a = baseline.trace[j]
            val b = test.trace[j]
            if (a.size != b.size) {
                report.appendText("Step ${j + 1}: logits dimension mismatch\n")
                passed = false
                continue
            }
            var worst = 0.0
            var peak = 0.0
            var sqDiff = 0.0
            var sqBase = 0.0
            var finite = true
            for (i in a.indices) {
                if (!a[i].isFinite() || !b[i].isFinite()) {
                    finite = false
                    break
                }
                val delta = abs(a[i] - b[i]).toDouble()
                worst = max(worst, delta)
                peak = max(peak, abs(a[i].toDouble()))
                sqDiff += delta * delta
                sqBase += a[i].toDouble() * a[i].toDouble()
            }
            val relL2 = sqrt(sqDiff / max(sqBase, 1e-12))
            val threshold = max(0.10, peak * 0.03)
            val ok = finite && worst <= threshold && relL2 <= 0.03
            if (!ok) passed = false
            report.appendText(
                "${test.mode.label} codebook=${j + 1} max_abs=${format(worst)} " +
                    "rel_l2=${format(relL2)} threshold=${format(threshold)} " +
                    "status=${if (ok) "PASS" else "FAIL"}\n"
            )
        }
        return passed
    }

    private fun format(x: Double) = String.format(Locale.US, "%.4f", x)
}
