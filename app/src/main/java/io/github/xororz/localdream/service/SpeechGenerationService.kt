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
    @Volatile private var nativeEffectiveFrames: Int = 0
    @Volatile private var nativeDecodedFrames: Int = 0
    @Volatile private var nativeGeneratedFrames: Int = 0
    @Volatile private var nativeVocoderMsPerFrame: Float = 0f
    @Volatile private var usingQnnVocoder: Boolean = false
    @Volatile private var nativeGenerationDone: Boolean = false
    @Volatile private var nativeSegmentCount: Int = 1
    @Volatile private var nativeSegmentIndex: Int = 1
    @Volatile private var nativeCompletedFrames: Int = 0
    @Volatile private var nativeOverallProgress: Float = 0f
    @Volatile private var nativeCodecOverallProgress: Float = 0f
    @Volatile private var nativeVocoderOverallProgress: Float = 0f
    private var activeQnnSelftestMarker: File? = null
    private var activeQnnSelftestKey: String? = null

    companion object {
        private const val CHANNEL_ID = "speech_generation_channel"
        private const val NOTIFICATION_ID = 8
        private const val BACKEND = "http://127.0.0.1:8082"
        private const val EXECUTABLE = "libbreeze_server.so"
        private const val SELFTEST_EXECUTABLE = "libbreeze_selftest.so"
        private const val RUNTIME_DIR = "runtime_breeze_htp"
        private const val RUNTIME_VERSION =
            "breeze-a0e177-hexagon-ab9acc-v212-embedded-backbone"

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
            val progress: Float = 0f,
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
            val codecProgress: Float? = null,
            val vocoderProgress: Float? = null,
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
            // A recreated/startForegroundService call without an action must
            // never tear down a warm 2+ GB Breeze process.
            else -> Unit
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
                repeat(8) {
                    if (healthReady()) {
                        _state.value = SpeechState.Ready(modelId, 0L)
                        return@launch
                    }
                    delay(75)
                }
                startServer(modelId)
            }
            return
        }
        workJob?.cancel()
        workJob = scope.launch { startServer(modelId) }
    }

    private fun qnnSelftestKey(install: BreezeQnnVocoderArtifact.Install): String =
        listOf(
            RUNTIME_VERSION,
            install.version,
            install.contextFile.name,
            install.contextFile.length(),
            install.contextFile.lastModified(),
            install.lutFile.name,
            install.lutFile.length(),
            install.selftestFeaturesFile.length(),
            install.selftestAudioFile.length(),
        ).joinToString("|")

    private fun qnnSelftestMarker(install: BreezeQnnVocoderArtifact.Install): File =
        File(install.contextFile.parentFile, ".runtime_selftest_ok")

    private fun qnnSelftestValidated(install: BreezeQnnVocoderArtifact.Install): Boolean {
        val marker = qnnSelftestMarker(install)
        val key = qnnSelftestKey(install)
        return marker.isFile && runCatching { marker.readText() }.getOrNull() == key
    }

    private suspend fun startServer(modelId: String) {
        val started = System.currentTimeMillis()
        _state.value = SpeechState.Loading(
            modelId,
            "Preparing speech engine",
            started,
            0.06f,
        )
        try {
            destroyProcess()
            val modelDir = File(Model.getModelsDir(this), modelId)
            val modelFile = File(modelDir, "model.gguf")
            if (!modelFile.isFile || modelFile.length() < 2_000_000_000L) {
                throw IllegalStateException("Breeze model.gguf is missing or incomplete")
            }

            _state.value = SpeechState.Loading(
                modelId, "Preparing accelerator runtime", started, 0.12f,
            )
            val fullQnnEnabled = BreezeQnnGeneratorArtifact.isEnabled(this)
            val qnnGeneratorInstall = if (fullQnnEnabled) {
                BreezeQnnGeneratorArtifact.localInstall(this)
            } else {
                null
            }

            // Do not combine the resident v4 multigraph vocoder context with
            // the legacy ggml-Hexagon generator on SM8850. The v4 context
            // self-test succeeds, but the first subsequent ggml-Hexagon graph
            // can abort in dspqueue_read with 0x2e / code 134. Use the proven
            // v3 single-graph QNN vocoder for the hybrid path. If v3 is not
            // installed, fall back to Breeze's built-in vocoder rather than
            // crashing the native process. Full-QNN may still opt into v4.
            val qnnInstall = if (qnnGeneratorInstall != null) {
                BreezeQnnVocoderArtifact.localInstall(this)
            } else {
                BreezeQnnVocoderArtifact.localLegacyHexagonSafeInstall(this)
            }
            val qnnVocoderFile = qnnInstall?.contextFile
            usingQnnVocoder = qnnInstall != null
            val usingQnnRuntime = qnnInstall != null || qnnGeneratorInstall != null
            val qnnSelftestCached = qnnInstall?.let { qnnSelftestValidated(it) } == true
            activeQnnSelftestMarker = qnnInstall?.let { qnnSelftestMarker(it) }
            activeQnnSelftestKey = qnnInstall?.let { qnnSelftestKey(it) }
            prepareRuntime(usingQnnRuntime)
            val executable = File(applicationInfo.nativeLibraryDir, EXECUTABLE)
            if (!executable.isFile) {
                throw IllegalStateException("Breeze native server is missing from this APK")
            }

            _state.value = SpeechState.Loading(
                modelId,
                "Loading model weights",
                started,
                0.32f,
            )

            val command = listOf(
                executable.absolutePath,
                modelFile.absolutePath,
                "--host", "127.0.0.1",
                "--port", "8082",
                "--ws-port", "-1",
                // QNN flush size is selected from the installed artifact at runtime.
                // v4 can emit at 8 frames; v3 waits for its fixed 64-frame graph.
                "--chunk-first", "8",
                "--chunk-max", "32",
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
                // SM8850 / HTP v81 has a reproducible dependent-op visibility
                // failure when an HMX matmul is chained into unary/GLU work.
                // Keep the proven large v153 DSPQueue packet, but run every
                // accelerator kernel on HVX. This is still strict HTP: there is
                // no CPU fallback and all model/vocoder compute stays on Hexagon.
                "GGML_HEXAGON_NHMX" to "0",
                "GGML_HEXAGON_NHVX" to "0",
                "GGML_HEXAGON_MM_SELECT" to "1",
                "GGML_HEXAGON_FA_SELECT" to "1",
                "GGML_HEXAGON_GDN_SELECT" to "1",
                "GGML_HEXAGON_OPFUSION" to "1",
                // Busy-poll the DSPQueue completion ring in latency mode.
                // This removes the response sleep/wake path used by the
                // autoregressive one-frame generator.
                "GGML_HEXAGON_OPPOLL" to "0",
            )
            if (usingQnnRuntime) {
                env["BREEZE_QNN_LIB_DIR"] = runtimeDir.absolutePath
                env["LOCALDREAM_QNN_POWER_MODE"] = "high_performance"
            }
            qnnGeneratorInstall?.let { generator ->
                env["BREEZE_QNN_DEPTH_PATH"] =
                    generator.depthContextFile.absolutePath
                env["BREEZE_QNN_BACKBONE_PREFILL_PATH"] =
                    generator.backbonePrefillFile.absolutePath
                env["BREEZE_QNN_BACKBONE_STEP_PATH"] =
                    generator.backboneStepFile.absolutePath
                env["BREEZE_QNN_GENERATOR_ENGINE"] = generator.engine
                env["BREEZE_QNN_BACKBONE_MAX_SEQ"] = generator.backboneMaxSeq.toString()
            }
            // Isolated per-codebook timings are intentionally opt-in so logging
            // does not affect actual throughput. Fused experimental graph is
            // never activated in production after the SM8850 DSPQueue failure.
            env["BREEZE_DEPTH_FUSION_TRIAL"] = "0"
            env["BREEZE_DEPTH_PROFILE"] = "1"
            val gpuDepthRequested = getSharedPreferences(
                "breeze_runtime_tuning", MODE_PRIVATE,
            ).getString("depth_backend", "npu") == "gpu"
            env["BREEZE_DEPTH_BACKEND"] = if (gpuDepthRequested) "vulkan" else "htp"
            if (gpuDepthRequested) env["BREEZE_GPU_GGUF_PATH"] = modelFile.absolutePath
            // Fuse the 16-codebook embedding gather into the autoregressive
            // backbone graph, eliminating one HTP roundtrip per audio frame.
            // This is independent of the unsafe whole-frame depth fusion.
            env["BREEZE_BACKBONE_EMBED_FUSE"] =
                if (getSharedPreferences("breeze_runtime_tuning", MODE_PRIVATE)
                        .getBoolean("fuse_backbone_embed", true)) "1" else "0"
            if (qnnVocoderFile != null) {
                env["BREEZE_QNN_VOCODER_PATH"] = qnnVocoderFile.absolutePath
                env["BREEZE_QNN_VOCODER_LUT_PATH"] = qnnInstall!!.lutFile.absolutePath
                env["BREEZE_QNN_SELFTEST_FEATURES_PATH"] =
                    qnnInstall.selftestFeaturesFile.absolutePath
                env["BREEZE_QNN_SELFTEST_AUDIO_PATH"] =
                    qnnInstall.selftestAudioFile.absolutePath
                env["BREEZE_QNN_FIRST_NEW"] = if (qnnInstall.version >= 4) "8" else "64"
                env["BREEZE_QNN_STEADY_NEW"] = "39"
                if (qnnSelftestCached) {
                    env["BREEZE_QNN_SKIP_SELFTEST"] = "1"
                }
            }

            BackendDiagnostics.beginSession(
                this,
                "Breeze strict HTP model=$modelId",
            )
            val installedPreferredQnn = BreezeQnnVocoderArtifact.localInstall(this)
            if (
                qnnGeneratorInstall == null &&
                installedPreferredQnn?.version == 4
            ) {
                BackendDiagnostics.append(
                    this,
                    "BREEZE_QNN_COMPAT",
                    if (qnnInstall?.version == 3) {
                        "v4 installed but legacy ggml-Hexagon generator selected proven v3 vocoder coexistence path"
                    } else {
                        "v4 installed but disabled for legacy ggml-Hexagon coexistence; using built-in vocoder because v3 is unavailable"
                    },
                )
            }
            BackendDiagnostics.append(
                this,
                "BREEZE_ENV",
                "backend=HTP0:0 transport=DSPQueue fallback=disabled " +
                    "queue=v202-opbatch1280x32-oppoll0 opfusion=1 hmx=0 execution=hvx-only-v81 gelu_erf=dsp-libm-reference-v81 " +
                    "getrows=exact-v153 dcache=upstream-pr29977-64b modelmap=ordinary-delayed+quant-repack " +
                    "codebooks=ordinary-htp-mirror quantweights=repack-upload-any-map visibility=none-v153-scheduler " +
                    "generator_mode=" + (if (fullQnnEnabled) "full-qnn" else "legacy-hexagon") + " " +
                    "generator=" + (if (qnnGeneratorInstall != null) qnnGeneratorInstall.engine else "ggml-hexagon") + " " +
                    "vocoder=" + (
                        if (usingQnnVocoder) {
                            if ((qnnInstall?.version ?: 0) >= 4) "qnn-htp-multigraph-v4"
                            else "qnn-htp-feature64-v3"
                        } else "ggml-stateful-fallback"
                    ) + " " +
                    "qnn_target=sm8850-v81 qnn_selftest=" +
                    (if (qnnSelftestCached) "cached" else "reference-pcm") + " " +
                    "qnn_scheduler=" + (
                        if ((qnnInstall?.version ?: 0) >= 4) "first8-steady39-tail32"
                        else "first64-steady39-fixed64"
                    ) + " qnn_left_context=25 qnn_host_lut=fp32 eos=eos-first " +
                    "snake=precomputed+fused diag=projection-preflight-v195 signal_validation=stream+pcm16 " +
                    "runtime=${runtimeDir.absolutePath}",
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
                if (attempt % 12 == 0) {
                    val current = _state.value as? SpeechState.Loading
                    if (current?.modelId == modelId && current.progress < 0.98f) {
                        _state.value = current.copy(
                            detail = if (usingQnnVocoder) {
                                "Finalizing QNN + Breeze runtime"
                            } else {
                                "Finalizing Breeze runtime"
                            },
                            progress = maxOf(current.progress, 0.92f),
                        )
                    }
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
                if (servingModelId == modelId && process?.isAlive == true) {
                    var ready = false
                    for (attempt in 0 until 8) {
                        if (healthReady()) {
                            ready = true
                            break
                        }
                        if (attempt < 7) delay(75)
                    }
                    if (!ready) {
                        startServer(modelId)
                        if (_state.value !is SpeechState.Ready) return@launch
                    }
                } else {
                    startServer(modelId)
                    if (_state.value !is SpeechState.Ready) return@launch
                }

                val started = System.currentTimeMillis()
                nativeEffectiveFrames = 0
                nativeDecodedFrames = 0
                nativeGeneratedFrames = 0
                nativeGenerationDone = false
                nativeSegmentCount = 1
                nativeSegmentIndex = 1
                nativeCompletedFrames = 0
                nativeOverallProgress = 0f
                nativeCodecOverallProgress = 0f
                nativeVocoderOverallProgress = 0f
                nativeVocoderMsPerFrame = if (usingQnnVocoder) 50f else 0f
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
                val historyDir = SpeechHistoryStore.directory(this@SpeechGenerationService, modelId)
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

                val elapsed = System.currentTimeMillis() - started
                val audioDurationMillis =
                    (pcmBytes / 2L * 1000L / sampleRate.toLong()).coerceAtLeast(0L)
                val item = SpeechHistoryStore.add(
                    context = this@SpeechGenerationService,
                    file = output,
                    text = text,
                    instruction = instruction.ifBlank { "Speak clearly and naturally." },
                    modelId = modelId,
                    seed = seed,
                    generationMillis = elapsed,
                    audioDurationMillis = audioDurationMillis,
                    cfg = cfg,
                    temperature = temperature,
                    topK = topK,
                    topP = topP,
                    repetition = repetition,
                    splitChars = splitChars,
                    maxNewTokens = maxNewTokens,
                    accelerated = usingQnnVocoder,
                )
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
            0.22f,
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

    private fun prepareRuntime(includeQnn: Boolean) {
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
        if (includeQnn) {
            assets.list("qnnlibs").orEmpty().forEach { name ->
                if (!name.endsWith(".so")) return@forEach
                val target = File(runtimeDir, name)
                val assetSize = assets.open("qnnlibs/" + name).use { it.available().toLong() }
                if (!target.isFile || target.length() != assetSize) {
                    assets.open("qnnlibs/" + name).use { input ->
                        target.outputStream().use { output -> input.copyTo(output) }
                    }
                }
                target.setReadable(true, true)
                target.setExecutable(true, true)
            }
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
    private val nativeSegmentsRegex = Regex(
        """\[BREEZE_SEGMENTS\] count=(\d+) split_chars=(\d+) anchor_chars=(\d+)""",
    )
    private val nativeSegmentRegex = Regex(
        """\[BREEZE_SEGMENT\] index=(\d+) total=(\d+) chars=(\d+)""",
    )
    private val nativeLimitRegex = Regex(
        """\[BREEZE_LIMIT\] estimate=([0-9.]+)s estimated_frames=(\d+) configured=(\d+) soft=(\d+) hard=(\d+) eos_first=1""",
    )
    private val nativeGenerationDoneRegex = Regex(
        """\[BREEZE_GENERATION_DONE\] frames=(\d+) eos=(\d+) hard_limit=(\d+)""",
    )
    private val nativeStageRegex = Regex(
        """\[BREEZE_STAGE\] frames=(\d+) total_frames=(\d+) depth_ms_per_frame=([0-9.]+) backbone_ms_per_frame=([0-9.]+)""",
    )
    private val nativeVocoderStreamRegex = Regex(
        """\[BREEZE_VOCODER_STREAM\] flush=(\d+) new_frames=(\d+) samples=(\d+) ms=([0-9.]+)""",
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

    private fun globalSegmentProgress(local: Float?): Float? {
        if (local == null) return null
        val count = nativeSegmentCount.coerceAtLeast(1)
        val index = nativeSegmentIndex.coerceIn(1, count)
        return (((index - 1).toFloat() + local.coerceIn(0f, 1f)) / count.toFloat())
            .coerceIn(0f, 1f)
    }

    private fun estimatedTotalFrames(localEffective: Int): Int {
        if (localEffective <= 0) return 0
        val count = nativeSegmentCount.coerceAtLeast(1)
        val index = nativeSegmentIndex.coerceIn(1, count)
        val remainingIncludingCurrent = count - index + 1
        return nativeCompletedFrames + localEffective * remainingIncludingCurrent
    }

    private fun monotonicOverall(value: Float?): Float? {
        if (value == null) return nativeOverallProgress.takeIf { it > 0f }
        nativeOverallProgress = maxOf(
            nativeOverallProgress,
            value.coerceIn(0f, 1f),
        )
        return nativeOverallProgress
    }

    private fun monotonicCodec(value: Float?): Float? {
        if (value == null) return nativeCodecOverallProgress.takeIf { it > 0f }
        nativeCodecOverallProgress = maxOf(
            nativeCodecOverallProgress,
            value.coerceIn(0f, 1f),
        )
        return nativeCodecOverallProgress
    }

    private fun monotonicVocoder(value: Float?): Float? {
        if (value == null) return nativeVocoderOverallProgress.takeIf { it > 0f }
        nativeVocoderOverallProgress = maxOf(
            nativeVocoderOverallProgress,
            value.coerceIn(0f, 1f),
        )
        return nativeVocoderOverallProgress
    }

    private fun handleNativeOutput(modelId: String, raw: String) {
        val line = raw.trim()
        if (line.isEmpty()) return

        BackendDiagnostics.append(
            this,
            "BREEZE_NATIVE",
            line.take(3000),
        )

        val loading = _state.value as? SpeechState.Loading
        if (loading?.modelId == modelId) {
            val milestone = when {
                line.contains("new session", ignoreCase = true) ->
                    "Connecting to Hexagon HTP" to 0.38f
                line.contains("quantized GGUF weights repacked", ignoreCase = true) ->
                    "Preparing quantized model weights" to 0.56f
                line.contains("codebook mirrors verified", ignoreCase = true) ->
                    "Verifying audio codebooks" to 0.64f
                line.contains("first decoder projection verified", ignoreCase = true) ->
                    "Model validation complete" to 0.72f
                line.contains("[BREEZE_QNN_SELFTEST] max_abs=", ignoreCase = true) ->
                    "Validating QNN waveform numerics" to 0.80f
                line.contains("[BREEZE_QNN_SELFTEST] skipped cached=1", ignoreCase = true) ->
                    "Using validated QNN cache" to 0.82f
                line.contains("[BREEZE_QNN] ready", ignoreCase = true) ->
                    "QNN vocoder validated" to 0.86f
                line.startsWith("loading ") ->
                    "Starting Breeze server" to 0.90f
                line.contains("listening on", ignoreCase = true) ->
                    "Speech engine is almost ready" to 0.98f
                else -> null
            }
            if (milestone != null && milestone.second >= loading.progress) {
                _state.value = loading.copy(
                    detail = milestone.first,
                    progress = milestone.second,
                )
            }
        }

        if (
            line.contains("[BREEZE_QNN] ready", ignoreCase = true) &&
            line.contains("selftest=passed", ignoreCase = true)
        ) {
            val marker = activeQnnSelftestMarker
            val key = activeQnnSelftestKey
            if (marker != null && key != null) {
                runCatching {
                    marker.parentFile?.mkdirs()
                    marker.writeText(key)
                }
            }
        }

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

        val current = _state.value as? SpeechState.Generating ?: return
        if (current.modelId != modelId) return

        nativeSegmentsRegex.find(line)?.let { match ->
            nativeSegmentCount = (match.groupValues[1].toIntOrNull() ?: 1).coerceAtLeast(1)
            nativeSegmentIndex = 1
            nativeCompletedFrames = 0
            nativeEffectiveFrames = 0
            nativeDecodedFrames = 0
            nativeGeneratedFrames = 0
            nativeGenerationDone = false
            return
        }

        nativeSegmentRegex.find(line)?.let { match ->
            val nextIndex = (match.groupValues[1].toIntOrNull() ?: 1).coerceAtLeast(1)
            val total = (match.groupValues[2].toIntOrNull() ?: nativeSegmentCount).coerceAtLeast(1)
            if (nextIndex > nativeSegmentIndex) {
                nativeCompletedFrames += nativeGeneratedFrames.coerceAtLeast(0)
            }
            nativeSegmentCount = total
            nativeSegmentIndex = nextIndex.coerceAtMost(total)
            nativeEffectiveFrames = 0
            nativeDecodedFrames = 0
            nativeGeneratedFrames = 0
            nativeGenerationDone = false
            val latest = _state.value as? SpeechState.Generating ?: return
            val base = globalSegmentProgress(0f)
            _state.value = latest.copy(
                detail = if (total > 1) {
                    "Generating speech $nativeSegmentIndex/$total"
                } else {
                    "Generating voice tokens"
                },
                progress = monotonicOverall(base),
                codecProgress = monotonicCodec(base),
                vocoderProgress = monotonicVocoder(base),
            )
            return
        }

        nativeLimitRegex.find(line)?.let { match ->
            nativeEffectiveFrames = match.groupValues[4].toIntOrNull() ?: 0
            if (nativeEffectiveFrames > 0) {
                val totalFrames = estimatedTotalFrames(nativeEffectiveFrames)
                val base = globalSegmentProgress(0f)
                _state.value = current.copy(
                    detail = if (nativeSegmentCount > 1) {
                        "Generating speech $nativeSegmentIndex/$nativeSegmentCount"
                    } else {
                        "Generating voice tokens"
                    },
                    estimatedSeconds = totalFrames.takeIf { it > 0 }?.times(0.08f),
                    progress = monotonicOverall(base),
                    codecProgress = monotonicCodec(base),
                    vocoderProgress = monotonicVocoder(base),
                )
            }
            return
        }

        nativeGenerationDoneRegex.find(line)?.let { match ->
            val frames = match.groupValues[1].toIntOrNull() ?: 0
            if (frames > 0) {
                nativeGenerationDone = true
                nativeGeneratedFrames = frames
                nativeEffectiveFrames = frames
                val latest = _state.value as? SpeechState.Generating ?: return
                val vocoderP = (nativeDecodedFrames.toFloat() / frames.toFloat()).coerceIn(0f, 1f)
                val localOverall = if (usingQnnVocoder) {
                    0.72f + 0.28f * vocoderP
                } else {
                    0.58f + 0.42f * vocoderP
                }
                val totalFrames = estimatedTotalFrames(frames)
                _state.value = latest.copy(
                    detail = if (nativeDecodedFrames < frames) {
                        "Decoding waveform on QNN HTP"
                    } else if (nativeSegmentIndex < nativeSegmentCount) {
                        "Preparing next speech segment"
                    } else {
                        "Finalizing audio"
                    },
                    generatedSeconds = (nativeCompletedFrames + frames) * 0.08f,
                    codecProgress = monotonicCodec(globalSegmentProgress(1f)),
                    vocoderProgress = monotonicVocoder(globalSegmentProgress(vocoderP)),
                    progress = monotonicOverall(globalSegmentProgress(localOverall)),
                    estimatedSeconds = totalFrames.takeIf { it > 0 }?.times(0.08f),
                )
            }
            return
        }

        nativeStageRegex.find(line)?.let { match ->
            val frames = match.groupValues[1].toIntOrNull() ?: return
            val nativeTotalFrames = match.groupValues[2].toIntOrNull()
                ?: (nativeCompletedFrames + frames)
            nativeGeneratedFrames = frames
            val elapsed = (
                System.currentTimeMillis() - current.startedAtMillis
            ).coerceAtLeast(1L) / 1000f
            val totalGeneratedFrames = maxOf(nativeCompletedFrames + frames, nativeTotalFrames)
            val fps = totalGeneratedFrames / elapsed
            val generated = totalGeneratedFrames * 0.08f
            val effective = nativeEffectiveFrames
            val codecP = if (effective > 0) {
                (frames.toFloat() / effective).coerceIn(0f, 1f)
            } else null
            val vocoderP = if (effective > 0) {
                (nativeDecodedFrames.toFloat() / effective).coerceIn(0f, 1f)
            } else null
            val localProgress = if (codecP != null && vocoderP != null) {
                if (usingQnnVocoder) {
                    0.72f * codecP + 0.28f * vocoderP
                } else {
                    0.58f * codecP + 0.42f * vocoderP
                }
            } else null
            val progress = globalSegmentProgress(localProgress)
            val generationEta = if (effective > frames && fps > 0f) {
                (effective - frames) / fps
            } else 0f
            val vocoderEta = if (
                usingQnnVocoder && effective > nativeDecodedFrames &&
                nativeVocoderMsPerFrame > 0f
            ) {
                (effective - nativeDecodedFrames) * nativeVocoderMsPerFrame / 1000f
            } else 0f
            _state.value = current.copy(
                detail = when {
                    !nativeGenerationDone && effective > 0 && frames >= effective ->
                        "Extending to natural end-of-speech"
                    nativeDecodedFrames > 0 && frames < effective ->
                        "Alternating voice tokens and QNN waveform"
                    nativeGenerationDone && nativeDecodedFrames < effective ->
                        "Finishing QNN waveform"
                    else -> "Generating voice tokens"
                },
                generatedSeconds = maxOf(current.generatedSeconds, generated),
                progress = monotonicOverall(progress),
                codecProgress = monotonicCodec(globalSegmentProgress(codecP)),
                vocoderProgress = monotonicVocoder(globalSegmentProgress(vocoderP)),
                estimatedSeconds = if (!nativeGenerationDone && effective > 0 && frames >= effective) {
                    null
                } else {
                    estimatedTotalFrames(effective).takeIf { it > 0 }?.times(0.08f)
                },
                elapsedSeconds = elapsed,
                etaSeconds = if (!nativeGenerationDone && effective > 0 && frames >= effective) {
                    null
                } else {
                    generationEta + vocoderEta
                },
                fps = fps,
                realtimeFactor = generated / elapsed,
            )
            return
        }

        nativeVocoderStreamRegex.find(line)?.let { match ->
            val decoded = match.groupValues[2].toIntOrNull() ?: 0
            val flushMs = match.groupValues[4].toFloatOrNull() ?: 0f
            nativeDecodedFrames += decoded
            if (decoded > 0 && flushMs > 0f) {
                val sample = flushMs / decoded
                nativeVocoderMsPerFrame = if (nativeVocoderMsPerFrame <= 0f) {
                    sample
                } else {
                    nativeVocoderMsPerFrame * 0.7f + sample * 0.3f
                }
            }
            val latest = _state.value as? SpeechState.Generating ?: return
            val effective = nativeEffectiveFrames
            val codecP = if (effective > 0) {
                (nativeGeneratedFrames.toFloat() / effective).coerceIn(0f, 1f)
            } else null
            val vocoderP = if (effective > 0) {
                (nativeDecodedFrames.toFloat() / effective).coerceIn(0f, 1f)
            } else null
            val localOverall = if (codecP != null && vocoderP != null) {
                if (usingQnnVocoder) 0.72f * codecP + 0.28f * vocoderP
                else 0.58f * codecP + 0.42f * vocoderP
            } else null
            val overall = globalSegmentProgress(localOverall)
            val elapsed = (
                System.currentTimeMillis() - latest.startedAtMillis
            ).coerceAtLeast(1L) / 1000f
            val vocoderEta = if (
                effective > nativeDecodedFrames && nativeVocoderMsPerFrame > 0f
            ) {
                (effective - nativeDecodedFrames) * nativeVocoderMsPerFrame / 1000f
            } else 0f
            _state.value = latest.copy(
                detail = when {
                    nativeGeneratedFrames < effective ->
                        "Alternating voice tokens and QNN waveform"
                    nativeDecodedFrames < effective ->
                        "Finishing QNN waveform"
                    else -> "Finalizing audio"
                },
                generatedSeconds = maxOf(
                    latest.generatedSeconds,
                    (nativeCompletedFrames + maxOf(nativeGeneratedFrames, nativeDecodedFrames)) * 0.08f,
                ),
                progress = monotonicOverall(overall),
                codecProgress = monotonicCodec(globalSegmentProgress(codecP)),
                vocoderProgress = monotonicVocoder(globalSegmentProgress(vocoderP)),
                elapsedSeconds = elapsed,
                etaSeconds = vocoderEta,
            )
            return
        }

        if (line.startsWith("[BREEZE_VOCODER] reference begin")) {
            val latest = _state.value as? SpeechState.Generating ?: return
            _state.value = latest.copy(
                detail = "Decoding waveform on Hexagon",
                progress = null,
                etaSeconds = null,
                fps = null,
                realtimeFactor = null,
            )
            return
        }

        // Once the native engine gave us the exact frame ceiling, its old
        // tqdm percentage is only a spoken-duration estimate and becomes
        // misleading while a vocoder flush is running.
        if (nativeEffectiveFrames > 0) return
        val match = nativeProgressRegex.find(line) ?: return

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
            progress = monotonicOverall(if (estimateReached) null else percent?.div(100f)),
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
