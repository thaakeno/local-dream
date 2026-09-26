package io.github.xororz.localdream.service

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Context
import android.content.Intent
import android.os.IBinder
import androidx.core.app.NotificationCompat
import io.github.xororz.localdream.R
import io.github.xororz.localdream.utils.CrashDiagnostics
import io.github.xororz.localdream.utils.Http
import java.io.File
import java.util.concurrent.TimeUnit
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import okhttp3.Call
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject

/**
 * Foreground text-to-music client for the native yue2.cpp daemon.
 *
 * yue2.cpp already exposes an async job API plus an SSE log stream. We consume
 * those supported interfaces directly and translate its native AR/NAR/VAE logs
 * into typed Material UI phases instead of patching the upstream runtime.
 */
class MusicGenerationService : Service() {
    private val scope = CoroutineScope(Dispatchers.IO + Job())
    private var workJob: Job? = null
    private var activeJobId: String? = null
    private var synthCall: Call? = null
    private var logCall: Call? = null
    private var cancelRequested = false

    companion object {
        private const val CHANNEL_ID = "music_generation_channel"
        private const val NOTIFICATION_ID = 7
        private const val BACKEND = "http://127.0.0.1:8081"

        const val ACTION_PRELOAD = "io.github.xororz.localdream.PRELOAD_MUSIC"
        const val ACTION_GENERATE = "io.github.xororz.localdream.GENERATE_MUSIC"
        const val ACTION_STOP = "io.github.xororz.localdream.STOP_MUSIC"

        private val client: OkHttpClient by lazy {
            Http.client.newBuilder()
                .connectTimeout(60, TimeUnit.SECONDS)
                .readTimeout(3600, TimeUnit.SECONDS)
                .writeTimeout(120, TimeUnit.SECONDS)
                .callTimeout(3600, TimeUnit.SECONDS)
                .retryOnConnectionFailure(true)
                .build()
        }

        private val quickClient: OkHttpClient by lazy {
            Http.client.newBuilder()
                .connectTimeout(3, TimeUnit.SECONDS)
                .readTimeout(3, TimeUnit.SECONDS)
                .writeTimeout(3, TimeUnit.SECONDS)
                .callTimeout(5, TimeUnit.SECONDS)
                .retryOnConnectionFailure(false)
                .build()
        }

        private val _state = MutableStateFlow<MusicState>(MusicState.Idle)
        val state: StateFlow<MusicState> = _state

        private val _residentModelId = MutableStateFlow<String?>(null)
        val residentModelId: StateFlow<String?> = _residentModelId

        fun reset() {
            if (_state.value !is MusicState.Generating &&
                _state.value !is MusicState.Preloading
            ) {
                _state.value = MusicState.Idle
            }
        }

        fun resetForModel(modelId: String) {
            val currentResident = _residentModelId.value
            if (currentResident != null && currentResident != modelId) {
                _residentModelId.value = null
            }
            when (val current = _state.value) {
                is MusicState.Preloading -> if (current.modelId != modelId) {
                    _state.value = MusicState.Idle
                }
                is MusicState.Ready -> if (current.modelId != modelId) {
                    _state.value = MusicState.Idle
                }
                else -> Unit
            }
        }

        fun preload(context: Context, modelId: String) {
            context.startForegroundService(
                Intent(context, MusicGenerationService::class.java)
                    .setAction(ACTION_PRELOAD)
                    .putExtra("modelId", modelId),
            )
        }

        fun clearResident() {
            _residentModelId.value = null
            val current = _state.value
            if (current is MusicState.Ready || current is MusicState.Preloading) {
                _state.value = MusicState.Idle
            }
        }

        fun stop(context: Context) {
            context.startService(
                Intent(context, MusicGenerationService::class.java).setAction(ACTION_STOP),
            )
        }
    }

    sealed class MusicState {
        object Idle : MusicState()

        data class Preloading(
            val modelId: String,
            val phase: String,
            val detail: String,
            val progress: Float? = null,
            val step: Int = 0,
            val total: Int = 0,
            val startedAtMillis: Long,
        ) : MusicState()

        data class Ready(
            val modelId: String,
            val preloadMillis: Long,
        ) : MusicState()

        data class Generating(
            val phase: String,
            val detail: String,
            val progress: Float? = null,
            val step: Int = 0,
            val total: Int = 0,
            val startedAtMillis: Long,
            val targetSeconds: Int,
        ) : MusicState()

        data class Complete(
            val file: File,
            val score: String,
            val lmSeed: Long,
            val acousticSeed: Long,
            val targetSeconds: Int,
            val elapsedMillis: Long,
        ) : MusicState()

        data class Error(val message: String, val modelId: String? = null) : MusicState()
    }

    override fun onCreate() {
        super.onCreate()
        val manager = getSystemService(NOTIFICATION_SERVICE) as NotificationManager
        manager.createNotificationChannel(
            NotificationChannel(CHANNEL_ID, "Music generation", NotificationManager.IMPORTANCE_LOW),
        )
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        startForeground(NOTIFICATION_ID, notification("Preparing YuE2"))
        when (intent?.action) {
            ACTION_STOP -> cancelWork(startId)
            ACTION_PRELOAD -> {
                val modelId = intent.getStringExtra("modelId").orEmpty()
                if (modelId.isBlank()) {
                    _state.value = MusicState.Error("Missing YuE2 model id")
                    finishService()
                } else {
                    startPreload(modelId)
                }
            }
            ACTION_GENERATE -> startGeneration(intent)
            else -> finishService()
        }
        return START_NOT_STICKY
    }

    private fun startPreload(modelId: String) {
        if (_residentModelId.value == modelId) {
            _state.value = MusicState.Ready(modelId, 0L)
            finishService()
            return
        }
        when (val current = _state.value) {
            is MusicState.Preloading -> if (current.modelId == modelId) return
            is MusicState.Generating -> return
            else -> Unit
        }

        workJob?.cancel()
        cancelRequested = false
        val started = System.currentTimeMillis()
        CrashDiagnostics.beginGenerationSession(
            this,
            "YuE2 preload model=$modelId",
        )
        _state.value = MusicState.Preloading(
            modelId = modelId,
            phase = "server",
            detail = "Connecting to native YuE2 runtime",
            progress = 0.02f,
            startedAtMillis = started,
        )
        notifyPhase("Loading YuE2 model")

        workJob = scope.launch {
            try {
                waitForBackend(modelId, started)
                if (cancelRequested) return@launch

                val payload = JSONObject().apply {
                    put("style", "instrumental warmup")
                    put("lyrics", "")
                    put("cot", "off")
                    // One 25 Hz semantic frame. This is a real end-to-end
                    // LM -> NAR -> VAE pass, not a fake loading animation.
                    put("duration", 0.04)
                    put("lm_seed", 0)
                    put("seed", 0)
                    put("steps", 1)
                    put("lm_batch_size", 1)
                    put("synth_batch_size", 1)
                    put("cfg_scale", 1.0)
                    put("output_format", "wav")
                    put(
                        "semantic_sampling",
                        JSONObject().apply {
                            put("temperature", 1.0)
                            put("top_p", 0.95)
                        },
                    )
                }

                _state.value = MusicState.Preloading(
                    modelId = modelId,
                    phase = "submit",
                    detail = "Starting one-frame model warmup",
                    progress = 0.06f,
                    startedAtMillis = started,
                )
                val id = submit(payload)
                activeJobId = id
                CrashDiagnostics.recordGeneration(
                    this@MusicGenerationService,
                    "PHASE",
                    "warmup job=$id submitted",
                )

                val logJob = launch { followPreloadLogs(id, modelId, started) }
                try {
                    pollPreload(id, modelId, started)
                } finally {
                    logCall?.cancel()
                    logJob.cancel()
                }
            } catch (e: CancellationException) {
                if (cancelRequested) _state.value = MusicState.Idle
            } catch (e: Exception) {
                if (!cancelRequested) {
                    val message = e.message ?: "YuE2 preload failed"
                    CrashDiagnostics.recordGeneration(
                        this@MusicGenerationService,
                        "ERROR",
                        message,
                        e,
                    )
                    _state.value = MusicState.Error(message, modelId)
                    notifyPhase("YuE2 model load failed")
                }
            } finally {
                synthCall = null
                logCall = null
                activeJobId = null
                if (_state.value !is MusicState.Preloading) finishService()
            }
        }
    }

    private suspend fun waitForBackend(modelId: String, started: Long) {
        repeat(120) { attempt ->
            if (cancelRequested) throw CancellationException()
            val ready = runCatching {
                quickClient.newCall(
                    Request.Builder().url("$BACKEND/health").get().build(),
                ).execute().use { it.isSuccessful }
            }.getOrDefault(false)
            if (ready) return

            _state.value = MusicState.Preloading(
                modelId = modelId,
                phase = "server",
                detail = "Starting native server · attempt ${attempt + 1}",
                progress = 0.02f,
                startedAtMillis = started,
            )
            delay(250)
        }
        throw IllegalStateException("YuE2 server did not become ready")
    }

    private suspend fun pollPreload(id: String, modelId: String, started: Long) {
        while (!cancelRequested) {
            when (jobStatus(id)) {
                "done" -> {
                    val elapsed = System.currentTimeMillis() - started
                    _residentModelId.value = modelId
                    CrashDiagnostics.recordGeneration(
                        this,
                        "COMPLETE",
                        "YuE2 preload ready model=$modelId elapsed=${elapsed}ms",
                    )
                    _state.value = MusicState.Ready(modelId, elapsed)
                    notifyPhase("YuE2 ready")
                    return
                }
                "failed" -> throw IllegalStateException(
                    "YuE2 warmup failed. Check the native-stage details.",
                )
                "cancelled" -> {
                    _state.value = MusicState.Idle
                    return
                }
            }
            delay(250)
        }
    }

    private fun startGeneration(intent: Intent) {
        val style = intent.getStringExtra("style")?.trim().orEmpty()
        val lyrics = intent.getStringExtra("lyrics").orEmpty()
        val modelId = intent.getStringExtra("modelId")
        if (style.isBlank() && lyrics.isBlank()) {
            _state.value = MusicState.Error("Describe the music or enter lyrics.", modelId)
            finishService()
            return
        }

        workJob?.cancel()
        val cot = intent.getStringExtra("cot")?.takeIf {
            it in setOf("full", "melody", "off")
        } ?: "full"
        val duration = intent.getIntExtra("duration", 20).coerceIn(5, 20)
        val steps = intent.getIntExtra("steps", 32).coerceIn(1, 64)
        val seed = intent.getLongExtra("seed", -1L)
        val temperature = intent.getFloatExtra("semantic_temperature", 1f)
            .coerceIn(0.5f, 1.5f)
        val topP = intent.getFloatExtra("semantic_top_p", 0.95f)
            .coerceIn(0.5f, 1f)
        val cfg = intent.getFloatExtra("cfg_scale", -1f).let { value ->
            if (value < 0f) -1f else value.coerceIn(0.5f, 2f)
        }
        val started = System.currentTimeMillis()

        cancelRequested = false
        CrashDiagnostics.beginGenerationSession(
            this,
            "YuE2 generate model=${modelId ?: "unknown"} duration=${duration}s steps=$steps cot=$cot",
        )
        _state.value = MusicState.Generating(
            phase = "queued",
            detail = if (_residentModelId.value == modelId) {
                "Using resident YuE2 pipeline"
            } else {
                "Submitting to YuE2"
            },
            progress = 0.01f,
            startedAtMillis = started,
            targetSeconds = duration,
        )
        notifyPhase("Starting YuE2")

        workJob = scope.launch {
            try {
                val payload = JSONObject().apply {
                    put("style", style)
                    put("lyrics", lyrics)
                    put("cot", cot)
                    put("duration", duration.toDouble())
                    put("lm_seed", seed)
                    put("seed", seed)
                    put("steps", steps)
                    put("lm_batch_size", 1)
                    put("synth_batch_size", 1)
                    put("cfg_scale", cfg.toDouble())
                    put("output_format", "mp3")
                    put("mp3_bitrate", 320)
                    put(
                        "semantic_sampling",
                        JSONObject().apply {
                            // Preserve YuE2's checkpoint-native top-k,
                            // repetition penalty, window and token bounds.
                            put("temperature", temperature.toDouble())
                            put("top_p", topP.toDouble())
                        },
                    )
                }

                val id = submit(payload)
                activeJobId = id
                CrashDiagnostics.recordGeneration(
                    this@MusicGenerationService,
                    "PHASE",
                    "job=$id submitted",
                )

                val logJob = launch {
                    followGenerationLogs(id, started, duration, steps)
                }
                try {
                    pollUntilComplete(id, started, duration)
                } finally {
                    logCall?.cancel()
                    logJob.cancel()
                }
            } catch (e: CancellationException) {
                if (cancelRequested) {
                    _state.value = MusicState.Idle
                }
            } catch (e: Exception) {
                if (!cancelRequested) {
                    val message = e.message ?: "Music generation failed"
                    CrashDiagnostics.recordGeneration(
                        this@MusicGenerationService,
                        "ERROR",
                        message,
                        e,
                    )
                    _state.value = MusicState.Error(message, modelId)
                    notifyPhase("Music generation failed")
                }
            } finally {
                synthCall = null
                logCall = null
                activeJobId = null
                finishService()
            }
        }
    }

    private fun submit(payload: JSONObject): String {
        val request = Request.Builder()
            .url("$BACKEND/synth")
            .post(payload.toString().toRequestBody("application/json".toMediaType()))
            .build()
        synthCall = client.newCall(request)
        synthCall!!.execute().use { response ->
            val body = response.body?.string().orEmpty()
            if (!response.isSuccessful) {
                throw IllegalStateException(
                    runCatching { JSONObject(body).optString("error") }.getOrNull()
                        ?.takeIf { it.isNotBlank() }
                        ?: "YuE2 request failed (${response.code})",
                )
            }
            return JSONObject(body).getString("id")
        }
    }

    private fun jobStatus(id: String): String {
        val request = Request.Builder().url("$BACKEND/job?id=$id").get().build()
        client.newCall(request).execute().use { response ->
            if (!response.isSuccessful) {
                throw IllegalStateException("YuE2 job disappeared")
            }
            return JSONObject(response.body?.string().orEmpty()).optString("status")
        }
    }

    private suspend fun pollUntilComplete(
        id: String,
        started: Long,
        targetSeconds: Int,
    ) {
        while (!cancelRequested) {
            when (jobStatus(id)) {
                "done" -> {
                    fetchResult(id, started, targetSeconds)
                    return
                }
                "failed" -> throw IllegalStateException("YuE2 native pipeline failed")
                "cancelled" -> {
                    _state.value = MusicState.Idle
                    return
                }
            }
            delay(300)
        }
    }

    private fun fetchResult(id: String, started: Long, targetSeconds: Int) {
        _state.value = MusicState.Generating(
            phase = "result",
            detail = "Collecting encoded track",
            progress = 0.99f,
            startedAtMillis = started,
            targetSeconds = targetSeconds,
        )
        val request = Request.Builder().url("$BACKEND/job?id=$id&result=1").get().build()
        client.newCall(request).execute().use { response ->
            if (!response.isSuccessful) throw IllegalStateException("YuE2 result unavailable")
            val bytes = response.body?.bytes() ?: throw IllegalStateException("Empty YuE2 result")
            val boundary = response.header("Content-Type")
                ?.substringAfter("boundary=", "yue2-batch-boundary")
                ?.trim()
                ?.trim('"')
                ?: "yue2-batch-boundary"
            val parsed = parseSingleTrackMultipart(bytes, boundary)
            val outDir = File(filesDir, "music").apply { mkdirs() }
            val out = File(outDir, "yue2-${System.currentTimeMillis()}.mp3")
            out.writeBytes(parsed.audio)

            val replay = runCatching { JSONObject(parsed.replayJson) }.getOrNull()
            val elapsed = System.currentTimeMillis() - started
            CrashDiagnostics.recordGeneration(
                this,
                "COMPLETE",
                "YuE2 generation complete elapsed=${elapsed}ms file=${out.name}",
            )
            _state.value = MusicState.Complete(
                file = out,
                score = replay?.optString("abc").orEmpty(),
                lmSeed = replay?.optLong("lm_seed", -1L) ?: -1L,
                acousticSeed = replay?.optLong("seed", -1L) ?: -1L,
                targetSeconds = targetSeconds,
                elapsedMillis = elapsed,
            )
            notifyPhase("Music ready")
        }
    }

    private suspend fun followLogs(id: String, started: Long, targetSeconds: Int, renderSteps: Int) {
        val request = Request.Builder().url("$BACKEND/logs").get().build()
        logCall = client.newCall(request)
        logCall!!.execute().use { response ->
            if (!response.isSuccessful) return
            val source = response.body?.source() ?: return
            var active = false
            while (!source.exhausted() && !cancelRequested) {
                val raw = source.readUtf8Line() ?: break
                if (!raw.startsWith("data: ")) continue
                val line = raw.removePrefix("data: ")
                if (!active) {
                    if (line.contains("[Server] Job $id:")) active = true else continue
                }
                phaseFromLog(line, started, targetSeconds, renderSteps)?.let {
                    _state.value = it
                    notifyPhase(it.detail)
                }
            }
        }
    }

    private fun phaseFromLog(
        line: String,
        started: Long,
        targetSeconds: Int,
        renderSteps: Int,
    ): MusicState.Generating? {
        fun state(
            phase: String,
            detail: String,
            progress: Float? = null,
            step: Int = 0,
            total: Int = 0,
        ) = MusicState.Generating(phase, detail, progress, step, total, started, targetSeconds)

        val ar = Regex("""\[AR] (ABC|semantic) (\d+)/(\d+)""").find(line)
        if (ar != null) {
            val kind = ar.groupValues[1]
            val step = ar.groupValues[2].toInt()
            val total = ar.groupValues[3].toInt()
            return state(
                if (kind == "ABC") "planning" else "semantic",
                if (kind == "ABC") "Composing symbolic score" else "Writing semantic audio tokens",
                (step.toFloat() / total.coerceAtLeast(1)).coerceIn(0f, 1f),
                step,
                total,
            )
        }

        val nar = Regex("""\[NAR] Step (\d+)/(\d+)""").find(line)
        if (nar != null) {
            val step = nar.groupValues[1].toInt()
            val total = nar.groupValues[2].toInt()
            return state(
                "flow",
                "Rendering acoustic latents on HTP",
                (step.toFloat() / total.coerceAtLeast(1)).coerceIn(0f, 1f),
                step,
                total,
            )
        }

        return when {
            line.contains("[Load] LM") || line.contains("Load LM") ->
                state("loading_ar", "Loading AR composer on HTP")
            line.contains("[AR] ABC") ->
                state("planning", "Composing melody and harmony")
            line.contains("[AR] semantic") ->
                state("semantic", "Writing up to ${targetSeconds * 25} semantic frames")
            line.contains("[Store] Load NAR") || line.contains("[NAR] Loaded") ->
                state("loading_nar", "Swapping to the NAR renderer")
            line.contains("[NAR] Graph") ->
                state("flow", "Preparing $renderSteps-step acoustic flow")
            line.contains("[Store] Load VAE") || line.contains("[VAE] Loaded") ->
                state("loading_vae", "Loading Oobleck decoder")
            line.contains("[VAE] Tiled decode") ->
                state("decoding", "Decoding 48 kHz stereo audio")
            line.contains("[VAE] Decoded") || line.contains("[VAE] Tiled decode done") ->
                state("finalizing", "Encoding 320 kbps MP3")
            else -> null
        }
    }

    private data class ParsedTrack(val replayJson: String, val audio: ByteArray)

    private fun parseSingleTrackMultipart(data: ByteArray, boundary: String): ParsedTrack {
        val marker = "--$boundary".toByteArray()
        val headerEnd = "\r\n\r\n".toByteArray()
        var cursor = indexOf(data, marker, 0)
        var replay = ""
        var audio: ByteArray? = null
        while (cursor >= 0) {
            val headersStart = cursor + marker.size
            val hEnd = indexOf(data, headerEnd, headersStart)
            if (hEnd < 0) break
            val headers = data.copyOfRange(headersStart, hEnd).toString(Charsets.UTF_8)
            val bodyStart = hEnd + headerEnd.size
            val next = indexOf(data, marker, bodyStart)
            if (next < 0) break
            var bodyEnd = next
            while (
                bodyEnd > bodyStart &&
                (data[bodyEnd - 1] == '\n'.code.toByte() || data[bodyEnd - 1] == '\r'.code.toByte())
            ) {
                bodyEnd--
            }
            when {
                headers.contains("application/json", ignoreCase = true) ->
                    replay = data.copyOfRange(bodyStart, bodyEnd).toString(Charsets.UTF_8)
                headers.contains("audio/mpeg", ignoreCase = true) ->
                    audio = data.copyOfRange(bodyStart, bodyEnd)
            }
            cursor = next
        }
        return ParsedTrack(
            replayJson = replay,
            audio = audio ?: throw IllegalStateException("YuE2 result had no MP3 track"),
        )
    }

    private fun indexOf(haystack: ByteArray, needle: ByteArray, start: Int): Int {
        if (needle.isEmpty()) return start.coerceAtMost(haystack.size)
        val last = haystack.size - needle.size
        if (last < start) return -1
        outer@ for (i in start.coerceAtLeast(0)..last) {
            for (j in needle.indices) if (haystack[i + j] != needle[j]) continue@outer
            return i
        }
        return -1
    }

    private fun notification(text: String): Notification =
        NotificationCompat.Builder(this, CHANNEL_ID)
            .setSmallIcon(R.drawable.ic_launcher_foreground)
            .setContentTitle("Local Dream · YuE2")
            .setContentText(text)
            .setOngoing(true)
            .setOnlyAlertOnce(true)
            .build()

    private fun notifyPhase(text: String) {
        (getSystemService(NOTIFICATION_SERVICE) as NotificationManager)
            .notify(NOTIFICATION_ID, notification(text))
    }
}
