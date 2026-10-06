package io.github.xororz.localdream.service

import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Intent
import android.os.IBinder
import androidx.core.app.NotificationCompat
import io.github.xororz.localdream.R
import io.github.xororz.localdream.data.Model
import io.github.xororz.localdream.utils.BackendDiagnostics
import io.github.xororz.localdream.utils.Http
import java.io.File
import java.io.IOException
import java.util.Locale
import java.util.concurrent.TimeUnit
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import okhttp3.Call
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject

class SpeechGenerationService : Service() {
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    private var workJob: Job? = null
    private var activeCall: Call? = null
    private var process: Process? = null
    private var monitorThread: Thread? = null
    private var servingModelId: String? = null
    private lateinit var runtimeDir: File

    companion object {
        private const val CHANNEL_ID = "speech_generation_channel"
        private const val NOTIFICATION_ID = 8
        private const val BACKEND = "http://127.0.0.1:8082"
        private const val EXECUTABLE = "libbreeze_server.so"
        private const val SELFTEST_EXECUTABLE = "libbreeze_selftest.so"
        private const val RUNTIME_DIR = "runtime_breeze_htp"
        private const val RUNTIME_VERSION =
            "breeze-a0e177-hexagon-ab9acc-v179-reference-vocoder-audio-validated"

        const val ACTION_PRELOAD = "io.github.xororz.localdream.PRELOAD_BREEZE"
        const val ACTION_GENERATE = "io.github.xororz.localdream.GENERATE_BREEZE"
        const val ACTION_STOP = "io.github.xororz.localdream.STOP_BREEZE"

        private val _state = MutableStateFlow<SpeechState>(SpeechState.Idle)
        val state: StateFlow<SpeechState> = _state

        fun resetForModel(modelId: String) {
            when (val current = _state.value) {
                is SpeechState.Ready -> if (current.modelId != modelId) {
                    _state.value = SpeechState.Idle
                }
                is SpeechState.Loading -> if (current.modelId != modelId) {
                    _state.value = SpeechState.Idle
                }
                is SpeechState.Error -> _state.value = SpeechState.Idle
                else -> Unit
            }
        }
    }

    sealed class SpeechState {
        object Idle : SpeechState()

        data class Loading(
            val modelId: String,
            val detail: String,
            val startedAtMillis: Long,
        ) : SpeechState()

        data class Ready(
            val modelId: String,
            val loadMillis: Long,
        ) : SpeechState()

        data class Generating(
            val modelId: String,
            val detail: String,
            val generatedSeconds: Float,
            val startedAtMillis: Long,
            val progress: Float? = null,
            val estimatedSeconds: Float? = null,
            val elapsedSeconds: Float? = null,
            val etaSeconds: Float? = null,
            val fps: Float? = null,
            val realtimeFactor: Float? = null,
        ) : SpeechState()

        data class Complete(
            val file: File,
            val text: String,
            val instruction: String,
            val modelId: String,
            val seed: Long,
            val elapsedMillis: Long,
            val historyId: String,
        ) : SpeechState()

        data class Error(
            val message: String,
            val modelId: String? = null,
        ) : SpeechState()
    }

    override fun onCreate() {
        super.onCreate()
        val manager = getSystemService(NOTIFICATION_SERVICE) as NotificationManager
        manager.createNotificationChannel(
            NotificationChannel(
                CHANNEL_ID,
                "Speech generation",
                NotificationManager.IMPORTANCE_LOW,
            ),
        )
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        startForeground(
            NOTIFICATION_ID,
            NotificationCompat.Builder(this, CHANNEL_ID)
                .setContentTitle("Local Dream · Breeze TTS 2")
                .setContentText("Speech engine")
                .setSmallIcon(R.drawable.ic_launcher_monochrome)
                .setOngoing(true)
                .build(),
        )

        when (intent?.action) {
            ACTION_PRELOAD -> {
                val modelId = intent.getStringExtra("modelId").orEmpty()
                if (modelId.isBlank()) fail("Missing Breeze model id", null)
                else preload(modelId)
            }

            ACTION_GENERATE -> generate(intent)
            ACTION_STOP -> stopEverything()
            else -> stopEverything()
        }
        return START_NOT_STICKY
    }

    override fun onDestroy() {
        activeCall?.cancel()
        workJob?.cancel()
        destroyProcess()
        scope.coroutineContext[Job]?.cancel()
        super.onDestroy()
    }

    private fun preload(modelId: String) {
        if (servingModelId == modelId && process?.isAlive == true) {
            workJob?.cancel()
            workJob = scope.launch {
                if (healthReady()) {
                    _state.value = SpeechState.Ready(modelId, 0L)
                } else {
                    startServer(modelId)
                }
            }
            return
        }
        workJob?.cancel()
        workJob = scope.launch { startServer(modelId) }
    }

    private suspend fun startServer(modelId: String) {
        val started = System.currentTimeMillis()
        _state.value = SpeechState.Loading(
            modelId,
            "Preparing speech engine",
            started,
        )
        try {
            destroyProcess()
            val modelDir = File(Model.getModelsDir(this), modelId)
            val modelFile = File(modelDir, "model.gguf")
            if (!modelFile.isFile || modelFile.length() < 2_000_000_000L) {
                throw IllegalStateException("Breeze model.gguf is missing or incomplete")
            }

            prepareRuntime()
            val executable = File(applicationInfo.nativeLibraryDir, EXECUTABLE)
            if (!executable.isFile) {
                throw IllegalStateException("Breeze native server is missing from this APK")
            }

            _state.value = SpeechState.Loading(
                modelId,
                "Starting voice model",
                started,
            )

            val command = listOf(
                executable.absolutePath,
                modelFile.absolutePath,
                "--host", "127.0.0.1",
                "--port", "8082",
                "--ws-port", "-1",
                // Generation now uses the upstream reference vocoder graph.
                // These server knobs are kept conservative for API parity.
                "--chunk-first", "40",
                "--chunk-max", "40",
                "--split-chars", "600",
                "--verbose",
            )

            val nativeDir = applicationInfo.nativeLibraryDir
            val dspPath = listOf(
                runtimeDir.absolutePath,
                "/vendor/lib/rfsa/adsp",
                "/vendor/dsp/cdsp",
                "/dsp",
            ).joinToString(";")

            val env = mutableMapOf(
                // Match the device-proven v153 process environment exactly.
                // The backend defaults are already opbatch=1280, opqueue=32,
                // oppoll=0 and opfusion=1.
                "LD_LIBRARY_PATH" to listOf(
                    nativeDir,
                    runtimeDir.absolutePath,
                    "/system/lib64",
                    "/vendor/lib64",
                ).joinToString(":"),
                "ADSP_LIBRARY_PATH" to dspPath,
                "DSP_LIBRARY_PATH" to dspPath,
                "GGML_HEXAGON_DEVICES" to "HTP0:0",
                "GGML_HEXAGON_NHMX" to "1",
                "GGML_HEXAGON_NHVX" to "0",
                "GGML_HEXAGON_MM_SELECT" to "2",
                "GGML_HEXAGON_OPFUSION" to "1",
            )

            BackendDiagnostics.beginSession(
                this,
                "Breeze strict HTP model=$modelId",
            )
            BackendDiagnostics.append(
                this,
                "BREEZE_ENV",
                "backend=HTP0:0 transport=DSPQueue fallback=disabled " +
                    "queue=v153-default-1280x32 opfusion=1 hmx=1 " +
                    "getrows=exact-v153 dcache=exact-v153-128b modelmap=exact-v153 " +
                    "codebooks=ordinary-htp-mirror vocoder=upstream-reference-window40 " +
                    "signal_validation=native+pcm16 runtime=${runtimeDir.absolutePath}",
            )
            runBackendSelfTest(env, modelId, started)
            BackendDiagnostics.append(this, "BREEZE_CMD", command.joinToString(" "))

            val proc = ProcessBuilder(command).apply {
                directory(runtimeDir)
                redirectErrorStream(true)
                environment().putAll(env)
            }.start()
            process = proc
            servingModelId = modelId
            monitorProcess(proc, modelId)

            repeat(480) { attempt ->
                if (healthReady()) {
                    val elapsed = System.currentTimeMillis() - started
                    _state.value = SpeechState.Ready(modelId, elapsed)
                    return
                }
                if (!proc.isAlive) {
                    throw IllegalStateException(
                        "Speech engine stopped during startup.",
                    )
                }
                if (attempt % 8 == 0) {
                    _state.value = SpeechState.Loading(
                        modelId,
                        "Loading voice model · ${attempt / 2}s",
                        started,
                    )
                }
                delay(500)
            }
            throw IllegalStateException("Speech engine did not become ready")
        } catch (e: CancellationException) {
            throw e
        } catch (e: Exception) {
            fail(e.message ?: "Breeze startup failed", modelId)
            destroyProcess()
        }
    }

    private fun generate(intent: Intent) {
        val modelId = intent.getStringExtra("modelId").orEmpty()
        val text = intent.getStringExtra("text")?.trim().orEmpty()
        val instruction = intent.getStringExtra("instruction")?.trim().orEmpty()
        if (modelId.isBlank() || text.isBlank()) {
            fail("Enter text to speak.", modelId.ifBlank { null })
            return
        }

        val seed = intent.getLongExtra("seed", 42L)
        val cfg = intent.getFloatExtra("cfg", 1f).coerceIn(1f, 4f)
        val temperature = intent.getFloatExtra("temperature", 0.9f).coerceIn(0.1f, 2f)
        val topK = intent.getIntExtra("topK", 50).coerceIn(1, 200)
        val topP = intent.getFloatExtra("topP", 1f).coerceIn(0.1f, 1f)
        val repetition = intent.getFloatExtra("repetition", 1.1f).coerceIn(1f, 2f)
        val splitChars = intent.getIntExtra("splitChars", 600).coerceIn(100, 2000)
        val maxNewTokens = intent.getIntExtra("maxNewTokens", 750).coerceIn(64, 3000)

        workJob?.cancel()
        workJob = scope.launch {
            try {
                if (servingModelId != modelId || process?.isAlive != true || !healthReady()) {
                    startServer(modelId)
                    if (_state.value !is SpeechState.Ready) return@launch
                }

                val started = System.currentTimeMillis()
                _state.value = SpeechState.Generating(
                    modelId = modelId,
                    detail = "Starting speech generation",
                    generatedSeconds = 0f,
                    startedAtMillis = started,
                )

                val payload = JSONObject().apply {
                    put("model", "breeze-tts-2")
                    put("input", text)
                    put(
                        "instructions",
                        instruction.ifBlank { "Speak clearly and naturally." },
                    )
                    put("response_format", "pcm")
                    put("cfg_scale", cfg.toDouble())
                    put("seed", seed)
                    put("temperature", temperature.toDouble())
                    put("top_k", topK)
                    put("top_p", topP.toDouble())
                    put("repetition_penalty", repetition.toDouble())
                    put("split_chars", splitChars)
                    put("max_new_tokens", maxNewTokens)
                }
                val request = Request.Builder()
                    .url("$BACKEND/v1/audio/speech")
                    .post(
                        payload.toString()
                            .toRequestBody("application/json; charset=utf-8".toMediaType()),
                    )
                    .build()

                val call = client.newCall(request)
                activeCall = call
                val historyDir = SpeechHistoryStore.directory(this@SpeechGenerationService)
                val output = File(
                    historyDir,
                    "breeze_${System.currentTimeMillis()}_${seed}.wav",
                )
                val temp = File(output.parentFile, "${output.name}.part")

                var sampleRate = 24000
                var pcmBytes = 0L
                call.execute().use { response ->
                    if (!response.isSuccessful) {
                        val message = response.body?.string()?.take(800)
                            ?: "HTTP ${response.code}"
                        throw IOException("Breeze generation failed: $message")
                    }
                    sampleRate = response.header("X-Sample-Rate")?.toIntOrNull()
                        ?.takeIf { it in 8000..192000 } ?: 24000
                    val body = response.body ?: throw IOException("Breeze returned no audio")
                    var lastUi = 0L
                    temp.outputStream().buffered().use { out ->
                        // Reserve a canonical PCM WAV header. Breeze streams raw s16le PCM;
                        // we finalize the exact RIFF/data lengths once generation ends.
                        out.write(ByteArray(44))
                        body.byteStream().use { input ->
                            val buffer = ByteArray(64 * 1024)
                            while (true) {
                                val n = input.read(buffer)
                                if (n < 0) break
                                out.write(buffer, 0, n)
                                pcmBytes += n
                                val now = System.currentTimeMillis()
                                if (now - lastUi >= 250L) {
                                    val seconds = pcmBytes / 2f / sampleRate.toFloat()
                                    val current = _state.value as? SpeechState.Generating
                                    if (current?.modelId == modelId) {
                                        _state.value = current.copy(
                                            detail = if (current.progress == null) {
                                                "Streaming first audio"
                                            } else {
                                                current.detail
                                            },
                                            generatedSeconds = maxOf(
                                                current.generatedSeconds,
                                                seconds,
                                            ),
                                        )
                                    }
                                    lastUi = now
                                }
                            }
                        }
                    }
                }
                activeCall = null

                val nativeFailure = _state.value as? SpeechState.Error
                if (nativeFailure?.modelId == modelId) {
                    temp.delete()
                    throw IOException(nativeFailure.message)
                }

                if (pcmBytes < 2L || pcmBytes % 2L != 0L) {
                    temp.delete()
                    throw IOException("Breeze returned empty or truncated PCM audio")
                }
                validatePcm16Payload(temp, pcmBytes)
                finalizePcmWav(temp, pcmBytes, sampleRate)
                if (!temp.renameTo(output)) {
                    temp.copyTo(output, overwrite = true)
                    temp.delete()
                }

                val item = SpeechHistoryStore.add(
                    context = this@SpeechGenerationService,
                    file = output,
                    text = text,
                    instruction = instruction.ifBlank { "Speak clearly and naturally." },
                    modelId = modelId,
                    seed = seed,
                )
                val elapsed = System.currentTimeMillis() - started
                _state.value = SpeechState.Complete(
                    output,
                    text,
                    instruction,
                    modelId,
                    seed,
                    elapsed,
                    item.id,
                )
            } catch (e: CancellationException) {
                _state.value = SpeechState.Idle
            } catch (e: Exception) {
                if (activeCall?.isCanceled() == true) {
                    _state.value = SpeechState.Idle
                } else {
                    fail(e.message ?: "Breeze generation failed", modelId)
                }
            } finally {
                activeCall = null
            }
        }
    }

    private fun validatePcm16Payload(file: File, pcmBytes: Long) {
        var samples = 0L
        var nonZero = 0L
        var peak = 0
        var sumSquares = 0.0
        java.io.RandomAccessFile(file, "r").use { raf ->
            raf.seek(44L)
            val buffer = ByteArray(64 * 1024)
            var remaining = pcmBytes
            var carry = -1
            while (remaining > 0L) {
                val want = minOf(buffer.size.toLong(), remaining).toInt()
                val n = raf.read(buffer, 0, want)
                if (n <= 0) break
                remaining -= n.toLong()
                var i = 0
                if (carry >= 0 && n > 0) {
                    var v = carry or ((buffer[0].toInt() and 0xff) shl 8)
                    if (v >= 0x8000) v -= 0x10000
                    val a = kotlin.math.abs(v)
                    if (v != 0) nonZero++
                    if (a > peak) peak = a
                    sumSquares += v.toDouble() * v.toDouble()
                    samples++
                    carry = -1
                    i = 1
                }
                while (i + 1 < n) {
                    var v = (buffer[i].toInt() and 0xff) or
                        ((buffer[i + 1].toInt() and 0xff) shl 8)
                    if (v >= 0x8000) v -= 0x10000
                    val a = kotlin.math.abs(v)
                    if (v != 0) nonZero++
                    if (a > peak) peak = a
                    sumSquares += v.toDouble() * v.toDouble()
                    samples++
                    i += 2
                }
                if (i < n) carry = buffer[i].toInt() and 0xff
            }
            if (remaining != 0L || carry >= 0) {
                throw IOException("Breeze returned truncated PCM audio")
            }
        }

        val rms = if (samples > 0L) kotlin.math.sqrt(sumSquares / samples.toDouble()) else 0.0
        BackendDiagnostics.append(
            this,
            "BREEZE_PCM",
            "samples=$samples nonzero=$nonZero peak_s16=$peak rms_s16=" +
                String.format(Locale.US, "%.3f", rms),
        )
        val minNonZero = maxOf(32L, samples / 1000L)
        if (peak < 33 || rms < 2.0 || nonZero < minNonZero) {
            throw IOException(
                "Breeze produced effectively silent PCM " +
                    "(peak_s16=$peak, rms_s16=${String.format(Locale.US, "%.3f", rms)}, " +
                    "nonzero=$nonZero)",
            )
        }
    }

    private fun finalizePcmWav(file: File, pcmBytes: Long, sampleRate: Int) {
        if (pcmBytes > 0xffffffffL - 36L) {
            throw IOException("Generated audio is too large for a WAV file")
        }
        java.io.RandomAccessFile(file, "rw").use { raf ->
            fun le16(value: Int) {
                raf.write(value and 0xff)
                raf.write((value ushr 8) and 0xff)
            }
            fun le32(value: Long) {
                raf.write((value and 0xff).toInt())
                raf.write(((value ushr 8) and 0xff).toInt())
                raf.write(((value ushr 16) and 0xff).toInt())
                raf.write(((value ushr 24) and 0xff).toInt())
            }
            raf.seek(0)
            raf.writeBytes("RIFF")
            le32(36L + pcmBytes)
            raf.writeBytes("WAVE")
            raf.writeBytes("fmt ")
            le32(16)
            le16(1) // PCM
            le16(1) // mono
            le32(sampleRate.toLong())
            le32(sampleRate.toLong() * 2L)
            le16(2)
            le16(16)
            raf.writeBytes("data")
            le32(pcmBytes)
        }
    }

    private val client: OkHttpClient by lazy {
        Http.client.newBuilder()
            .connectTimeout(30, TimeUnit.SECONDS)
            .readTimeout(0, TimeUnit.MILLISECONDS)
            .writeTimeout(60, TimeUnit.SECONDS)
            .callTimeout(0, TimeUnit.MILLISECONDS)
            .retryOnConnectionFailure(false)
            .build()
    }

    private val healthClient: OkHttpClient by lazy {
        Http.client.newBuilder()
            .connectTimeout(2, TimeUnit.SECONDS)
            .readTimeout(2, TimeUnit.SECONDS)
            .writeTimeout(2, TimeUnit.SECONDS)
            .callTimeout(3, TimeUnit.SECONDS)
            .retryOnConnectionFailure(false)
            .build()
    }

    private suspend fun healthReady(): Boolean = runCatching {
        healthClient.newCall(
            Request.Builder().url("$BACKEND/health").get().build(),
        ).execute().use { it.isSuccessful }
    }.getOrDefault(false)

    private fun runBackendSelfTest(
        env: Map<String, String>,
        modelId: String,
        startedAtMillis: Long,
    ) {
        val marker = File(runtimeDir, ".selftest_ok")
        if (runCatching { marker.readText() }.getOrNull() == RUNTIME_VERSION) return

        _state.value = SpeechState.Loading(
            modelId,
            "Validating Hexagon speech kernels",
            startedAtMillis,
        )
        val executable = File(applicationInfo.nativeLibraryDir, SELFTEST_EXECUTABLE)
        if (!executable.isFile) {
            throw IllegalStateException("Breeze HTP self-test is missing from this APK")
        }

        val proc = ProcessBuilder(executable.absolutePath).apply {
            directory(runtimeDir)
            redirectErrorStream(true)
            environment().putAll(env)
        }.start()

        val output = proc.inputStream.bufferedReader().use { it.readText() }
        val exited = proc.waitFor(15, TimeUnit.SECONDS)
        if (!exited) {
            proc.destroyForcibly()
            throw IllegalStateException("Breeze HTP self-test timed out")
        }
        output.lineSequence()
            .filter { it.isNotBlank() }
            .forEach { BackendDiagnostics.append(this, "BREEZE_SELFTEST", it.take(2000)) }

        if (proc.exitValue() != 0 || !output.contains("[BREEZE_SELFTEST] all-ok")) {
            throw IllegalStateException(
                "Hexagon speech kernel self-test failed before model loading",
            )
        }
        marker.writeText(RUNTIME_VERSION)
    }

    private fun prepareRuntime() {
        runtimeDir = File(filesDir, RUNTIME_DIR)
        val stamp = File(runtimeDir, ".runtime_version")
        if (
            runtimeDir.exists() &&
            runCatching { stamp.readText() }.getOrNull() != RUNTIME_VERSION
        ) {
            runtimeDir.deleteRecursively()
        }
        runtimeDir.mkdirs()

        assets.list("breezelibs").orEmpty().forEach { name ->
            val target = File(runtimeDir, name)
            val assetSize = assets.open("breezelibs/$name").use { it.available().toLong() }
            if (!target.isFile || target.length() != assetSize) {
                assets.open("breezelibs/$name").use { input ->
                    target.outputStream().use { output -> input.copyTo(output) }
                }
            }
            target.setReadable(true, true)
            target.setExecutable(true, true)
        }
        stamp.writeText(RUNTIME_VERSION)
        stageDeviceFastRpcLibraries()
    }

    private fun stageDeviceFastRpcLibraries() {
        val roots = listOf(File("/vendor/lib64"), File("/system/vendor/lib64"))
        listOf(
            "libcdsprpc.so",
            "libvmmem.so",
            "vendor.qti.hardware.dsp-V1-ndk.so",
            "vendor.qti.hardware.dsp@1.0.so",
        ).forEach { name ->
            val target = File(runtimeDir, name)
            runCatching { target.delete() }
            val source = roots.asSequence()
                .map { File(it, name) }
                .firstOrNull { it.isFile }
                ?: return@forEach
            runCatching {
                source.inputStream().use { input ->
                    target.outputStream().use { output -> input.copyTo(output) }
                }
                target.setReadable(true, true)
                target.setExecutable(true, true)
            }.onFailure {
                runCatching { target.delete() }
            }
        }
    }

    private val nativeProgressRegex = Regex(
        """(\d{1,3})%\|.*?\|\s*([0-9.]+)/([0-9.]+)s\s*""" +
            """\[([0-9:]+)<([0-9:]+),\s*([0-9.]+)\s*fps,\s*([0-9.]+)x\]""",
    )

    private fun parseClockSeconds(value: String): Float? {
        val parts = value.split(':').mapNotNull { it.toFloatOrNull() }
        if (parts.isEmpty()) return null
        return when (parts.size) {
            1 -> parts[0]
            2 -> parts[0] * 60f + parts[1]
            else -> parts.takeLast(3).let { it[0] * 3600f + it[1] * 60f + it[2] }
        }
    }

    private fun handleNativeOutput(modelId: String, raw: String) {
        val line = raw.trim()
        if (line.isEmpty()) return

        BackendDiagnostics.append(
            this,
            "BREEZE_NATIVE",
            line.take(3000),
        )

        if (line.startsWith("generation error:", ignoreCase = true)) {
            val nativeMessage = line.substringAfter(':').trim()
            fail(
                if (nativeMessage.isBlank()) {
                    "Speech generation stopped inside the native engine."
                } else {
                    "Speech generation stopped: " + nativeMessage.take(280)
                },
                modelId,
            )
            return
        }

        val match = nativeProgressRegex.find(line) ?: return
        val current = _state.value as? SpeechState.Generating ?: return
        if (current.modelId != modelId) return

        val percent = match.groupValues[1].toFloatOrNull()?.coerceIn(0f, 100f)
        val generated = match.groupValues[2].toFloatOrNull()
        val estimated = match.groupValues[3].toFloatOrNull()
        val elapsed = parseClockSeconds(match.groupValues[4])
        val eta = parseClockSeconds(match.groupValues[5])
        val fps = match.groupValues[6].toFloatOrNull()
        val realtime = match.groupValues[7].toFloatOrNull()

        val estimateReached = percent != null && percent >= 100f
        _state.value = current.copy(
            detail = if (estimateReached) {
                "Waiting for end-of-speech"
            } else {
                "Synthesizing speech"
            },
            generatedSeconds = maxOf(current.generatedSeconds, generated ?: 0f),
            // Breeze reports progress against an estimated spoken duration.
            // 100% does not mean EOS has fired, so never show a fake completed bar.
            progress = if (estimateReached) null else percent?.div(100f),
            estimatedSeconds = if (estimateReached) null else estimated,
            elapsedSeconds = elapsed,
            etaSeconds = if (estimateReached) null else eta,
            fps = fps,
            realtimeFactor = realtime,
        )
    }

    private fun monitorProcess(proc: Process, modelId: String) {
        monitorThread?.interrupt()
        monitorThread = Thread({
            runCatching {
                proc.inputStream.bufferedReader().use { reader ->
                    val buffer = CharArray(2048)
                    val record = StringBuilder()
                    while (true) {
                        val count = reader.read(buffer)
                        if (count < 0) break
                        for (i in 0 until count) {
                            val ch = buffer[i]
                            if (ch == '\r' || ch == '\n') {
                                if (record.isNotEmpty()) {
                                    handleNativeOutput(modelId, record.toString())
                                    record.setLength(0)
                                }
                            } else {
                                record.append(ch)
                                if (record.length >= 8192) {
                                    handleNativeOutput(modelId, record.toString())
                                    record.setLength(0)
                                }
                            }
                        }
                    }
                    if (record.isNotEmpty()) {
                        handleNativeOutput(modelId, record.toString())
                    }
                }
            }
            val code = runCatching { proc.waitFor() }.getOrDefault(-1)
            if (process === proc && code != 0 && _state.value !is SpeechState.Idle) {
                fail(
                    "Speech engine stopped unexpectedly (code $code).",
                    modelId,
                )
            }
        }, "breeze-native-log").apply {
            isDaemon = true
            start()
        }
    }

    private fun destroyProcess() {
        activeCall?.cancel()
        activeCall = null
        process?.let { proc ->
            runCatching { proc.destroy() }
            runCatching {
                if (!proc.waitFor(800, TimeUnit.MILLISECONDS)) proc.destroyForcibly()
            }
        }
        process = null
        servingModelId = null
        monitorThread?.interrupt()
        monitorThread = null
    }

    private fun stopEverything() {
        activeCall?.cancel()
        workJob?.cancel()
        destroyProcess()
        _state.value = SpeechState.Idle
        stopForeground(STOP_FOREGROUND_REMOVE)
        stopSelf()
    }

    private fun fail(message: String, modelId: String?) {
        BackendDiagnostics.append(this, "BREEZE_ERROR", message)
        _state.value = SpeechState.Error(message, modelId)
    }
}
