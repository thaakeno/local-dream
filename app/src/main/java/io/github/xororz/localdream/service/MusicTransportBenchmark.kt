package io.github.xororz.localdream.service

import android.content.Context
import android.os.Build
import java.io.File
import java.util.concurrent.TimeUnit
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONObject

data class MusicTransportSample(
    val transport: String,
    val medianMs: Double,
    val meanMs: Double,
    val runs: Int,
    val nodes: Int,
    val checksum: Double,
)

data class MusicTransportBenchmarkResult(
    val dspQueue: MusicTransportSample,
    val fastRpc: MusicTransportSample,
) {
    val faster: String
        get() = if (fastRpc.medianMs < dspQueue.medianMs) "FastRPC" else "DSPQueue"
    val speedup: Double
        get() {
            val slow = maxOf(dspQueue.medianMs, fastRpc.medianMs)
            val fast = minOf(dspQueue.medianMs, fastRpc.medianMs)
            return if (fast > 0.0) slow / fast else 1.0
        }
}

object MusicTransportBenchmark {
    private const val TIMEOUT_SECONDS = 45L
    private const val RESULT_PREFIX = "BENCH_JSON "

    private fun htpArch(): String? {
        val soc = Build.SOC_MODEL.orEmpty().uppercase()
        return when {
            "SM8850" in soc -> "v81"
            "SM8750" in soc -> "v79"
            else -> null
        }
    }

    suspend fun run(context: Context): MusicTransportBenchmarkResult = withContext(Dispatchers.IO) {
        val arch = htpArch()
            ?: error("Transport benchmark supports SM8750/v79 and SM8850/v81")
        val dsp = runOne(context, "dspqueue", arch)
        val fast = runOne(context, "fastrpc", arch)

        // Both binaries execute the exact same deterministic graph. A checksum
        // mismatch means one transport produced a different result and the A/B
        // must not be trusted.
        val tolerance = 1e-4
        if (kotlin.math.abs(dsp.checksum - fast.checksum) > tolerance) {
            error(
                "Transport result mismatch: DSPQueue=${dsp.checksum}, " +
                    "FastRPC=${fast.checksum}",
            )
        }
        MusicTransportBenchmarkResult(dsp, fast)
    }

    private fun runOne(context: Context, transport: String, arch: String): MusicTransportSample {
        val runtime = File(context.filesDir, "runtime_yue2_bench_$transport")
        runtime.deleteRecursively()
        check(runtime.mkdirs()) { "Could not create $transport benchmark runtime" }

        val skelAsset = if (transport == "fastrpc") {
            "yue2libs/libggml-htp-$arch-fastrpc.so"
        } else {
            "yue2libs/libggml-htp-$arch.so"
        }
        val skel = File(runtime, "libggml-htp-$arch.so")
        context.assets.open(skelAsset).use { input ->
            skel.outputStream().use { output -> input.copyTo(output, 1024 * 1024) }
        }
        skel.setReadable(true, true)
        skel.setExecutable(true, true)

        val nativeDir = File(context.applicationInfo.nativeLibraryDir)
        val executable = File(
            nativeDir,
            if (transport == "fastrpc") {
                "libyue2_bench_fastrpc.so"
            } else {
                "libyue2_bench_dspqueue.so"
            },
        )
        check(executable.isFile) { "Missing $transport benchmark binary" }

        val process = ProcessBuilder(executable.absolutePath)
            .directory(runtime)
            .redirectErrorStream(true)
            .apply {
                environment().apply {
                    put("GGML_BACKEND", "HTP0")
                    put("GGML_HEXAGON_NDEV", "1")
                    put("LD_LIBRARY_PATH", "${nativeDir.absolutePath}:${runtime.absolutePath}:/vendor/lib64")
                    put(
                        "ADSP_LIBRARY_PATH",
                        "${runtime.absolutePath};/vendor/dsp/cdsp;" +
                            "/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;" +
                            "/vendor/dsp/dsp;/vendor/dsp/images;/dsp",
                    )
                    if (transport == "dspqueue") {
                        put("GGML_HEXAGON_OPPOLL", "0")
                        put("GGML_HEXAGON_OPBATCH", "64")
                        put("GGML_HEXAGON_OPQUEUE", "8")
                    }
                }
            }
            .start()

        val outputThread = StringBuilder()
        val reader = Thread {
            process.inputStream.bufferedReader().useLines { lines ->
                lines.forEach { line ->
                    synchronized(outputThread) {
                        outputThread.append(line).append('\n')
                    }
                }
            }
        }.apply {
            name = "yue2-$transport-bench-log"
            start()
        }

        val finished = process.waitFor(TIMEOUT_SECONDS, TimeUnit.SECONDS)
        if (!finished) {
            process.destroyForcibly()
            reader.join(1000)
            error("$transport benchmark timed out")
        }
        reader.join(1000)

        val output = synchronized(outputThread) { outputThread.toString() }
        if (process.exitValue() != 0) {
            error(
                "$transport benchmark failed (${process.exitValue()}): " +
                    output.lineSequence().takeLast(8).joinToString(" | "),
            )
        }

        val line = output.lineSequence().lastOrNull { it.startsWith(RESULT_PREFIX) }
            ?: error("$transport benchmark produced no result")
        val json = JSONObject(line.removePrefix(RESULT_PREFIX))
        return MusicTransportSample(
            transport = json.getString("transport"),
            medianMs = json.getDouble("median_ms"),
            meanMs = json.getDouble("mean_ms"),
            runs = json.getInt("runs"),
            nodes = json.getInt("nodes"),
            checksum = json.getDouble("checksum"),
        )
    }
}
