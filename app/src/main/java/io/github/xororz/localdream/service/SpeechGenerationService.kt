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
        private const val RUNTIME_DIR = "runtime_breeze_htp"
        private const val RUNTIME_VERSION =
            "breeze-a0e177-hexagon-ab9acc-strict-htp-v1"

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
                .setContentText("Qualcomm Hexagon HTP")
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
            "Preparing strict Hexagon HTP runtime",
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
                "Opening one physical HTP session",
                started,
            )

            val command = listOf(
                executable.absolutePath,
                modelFile.absolutePath,
                "--host", "127.0.0.1",
                "--port", "8082",
                "--ws-port", "-1",
                "--chunk-first", "2",
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
                "GGML_HEXAGON_OPPOLL" to "0",
                "GGML_HEXAGON_OPBATCH" to "64",
                "GGML_HEXAGON_OPQUEUE" to "8",
            )

            BackendDiagnostics.beginSession(
                this,
                "Breeze strict HTP model=$modelId",
            )
            BackendDiagnostics.append(
                this,
                "BREEZE_ENV",
                "backend=HTP0:0 fallback=disabled queue=64/8 hmx=1 " +
                    "runtime=${runtimeDir.absolutePath}",
            )
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
                        "Breeze HTP runtime exited during startup. Check native backend logs.",
                    )
                }
                if (attempt % 8 == 0) {
                    _state.value = SpeechState.Loading(
                        modelId,
                        "Loading ${modelFile.name} directly on HTP · ${attempt / 2}s",
                        started,
                    )
                }
                delay(500)
            }
            throw IllegalStateException("Breeze HTP runtime did not become ready")
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
                    modelId,
                    "Generating first audio frames on Hexagon HTP",
                    0f,
                    started,
                )

                val payload = JSONObject().apply {
                    put("model", "breeze-tts-2")
                    put("input", text)
                    put(
                        "instructions",
                        instruction.ifBlank { "Speak clearly and naturally." },
                    )
                    put("response_format", "wav")
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

                call.execute().use { response ->
                    if (!response.isSuccessful) {
                        val message = response.body?.string()?.take(800)
                            ?: "HTTP ${response.code}"
                        throw IOException("Breeze generation failed: $message")
                    }
                    val body = response.body ?: throw IOException("Breeze returned no audio")
                    var bytes = 0L
                    var lastUi = 0L
                    body.byteStream().use { input ->
                        temp.outputStream().buffered().use { out ->
                            val buffer = ByteArray(64 * 1024)
                            while (true) {
                                val n = input.read(buffer)
                                if (n < 0) break
                                out.write(buffer, 0, n)
                                bytes += n
                                val now = System.currentTimeMillis()
                                if (now - lastUi >= 250L) {
                                    val seconds =
                                        ((bytes - 44L).coerceAtLeast(0L) / 2f / 24000f)
                                    _state.value = SpeechState.Generating(
                                        modelId,
                                        "Streaming ${"%.1f".format(Locale.US, seconds)} s of 24 kHz audio",
                                        seconds,
                                        started,
                                    )
                                    lastUi = now
                                }
                            }
                        }
                    }
                }
                activeCall = null

                if (temp.length() < 64L) {
                    temp.delete()
                    throw IOException("Breeze returned an empty/truncated WAV stream")
                }
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

    private fun monitorProcess(proc: Process, modelId: String) {
        monitorThread?.interrupt()
        monitorThread = Thread({
            runCatching {
                proc.inputStream.bufferedReader().useLines { lines ->
                    lines.forEach { line ->
                        BackendDiagnostics.append(
                            this,
                            "BREEZE_NATIVE",
                            line.take(3000),
                        )
                    }
                }
            }
            val code = runCatching { proc.waitFor() }.getOrDefault(-1)
            if (process === proc && code != 0 && _state.value !is SpeechState.Idle) {
                fail(
                    "Breeze HTP runtime exited with code $code. Check native backend logs.",
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
