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
    private var activeJobId: String? = null
    private var synthCall: Call? = null
    private var logCall: Call? = null
    private var cancelRequested = false

    companion object {
        private const val CHANNEL_ID = "music_generation_channel"
        private const val NOTIFICATION_ID = 7
        private const val BACKEND = "http://127.0.0.1:8081"

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

        private val _state = MutableStateFlow<MusicState>(MusicState.Idle)
        val state: StateFlow<MusicState> = _state

        fun reset() {
            if (_state.value !is MusicState.Generating) _state.value = MusicState.Idle
        }

        fun stop(context: Context) {
            context.startService(
                Intent(context, MusicGenerationService::class.java).setAction(ACTION_STOP),
            )
        }
    }

    sealed class MusicState {
        object Idle : MusicState()

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

        data class Error(val message: String) : MusicState()
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
            ACTION_STOP -> {
                cancelRequested = true
                activeJobId?.let { id ->
                    scope.launch {
                        runCatching {
                            client.newCall(
                                Request.Builder()
                                    .url("$BACKEND/job?id=$id&cancel=1")
                                    .post(ByteArray(0).toRequestBody(null))
                                    .build(),
                            ).execute().close()
                        }
                    }
                }
                synthCall?.cancel()
                logCall?.cancel()
                _state.value = MusicState.Idle
                stopSelf()
            }
            ACTION_GENERATE -> startGeneration(intent)
            else -> stopSelf()
        }
        return START_NOT_STICKY
    }

    private fun startGeneration(intent: Intent) {
        val style = intent.getStringExtra("style")?.trim().orEmpty()
        val lyrics = intent.getStringExtra("lyrics").orEmpty()
        if (style.isBlank() && lyrics.isBlank()) {
            _state.value = MusicState.Error("Describe the music or enter lyrics.")
            stopSelf()
            return
        }

        val cot = intent.getStringExtra("cot")?.takeIf { it in setOf("full", "melody", "off") } ?: "full"
        val duration = intent.getIntExtra("duration", 20).coerceIn(5, 20)
        val steps = intent.getIntExtra("steps", 16).coerceIn(4, 32)
        val seed = intent.getLongExtra("seed", -1L)
        val temperature = intent.getFloatExtra("semantic_temperature", 1f).coerceIn(0.5f, 1.5f)
        val topP = intent.getFloatExtra("semantic_top_p", 0.95f).coerceIn(0.5f, 1f)
        val cfg = intent.getFloatExtra("cfg_scale", -1f)
        val started = System.currentTimeMillis()

        cancelRequested = false
        _state.value = MusicState.Generating(
            phase = "queued",
            detail = "Starting native YuE2 runtime",
            startedAtMillis = started,
            targetSeconds = duration,
        )
        notifyPhase("Starting YuE2")

        scope.launch {
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
                            put("temperature", temperature.toDouble())
                            put("top_p", topP.toDouble())
                            put("top_k", 100)
                            put("repetition_penalty", 1.2)
                            put("penalty_window", 50)
                            put("min_tokens", minOf(200, duration * 25))
                            put("max_tokens", duration * 25)
                        },
                    )
                }

                val request = Request.Builder()
                    .url("$BACKEND/synth")
                    .post(payload.toString().toRequestBody("application/json".toMediaType()))
                    .build()
                synthCall = client.newCall(request)
                val synthResponse = synthCall!!.execute()
                val synthBody = synthResponse.body?.string().orEmpty()
                if (!synthResponse.isSuccessful) {
                    throw IllegalStateException(
                        runCatching { JSONObject(synthBody).optString("error") }.getOrNull()
                            ?.takeIf { it.isNotBlank() }
                            ?: "YuE2 request failed (${synthResponse.code})",
                    )
                }
                val id = JSONObject(synthBody).getString("id")
                activeJobId = id

                val logJob = launch { followLogs(id, started, duration, steps) }
                try {
                    pollUntilComplete(id, started, duration)
                } finally {
                    logCall?.cancel()
                    logJob.cancel()
                }
            } catch (e: CancellationException) {
                _state.value = MusicState.Idle
            } catch (e: Exception) {
                if (!cancelRequested) {
                    _state.value = MusicState.Error(e.message ?: "Music generation failed")
                    notifyPhase("Music generation failed")
                }
            } finally {
                synthCall = null
                logCall = null
                activeJobId = null
                stopSelf()
            }
        }
    }

    private suspend fun pollUntilComplete(id: String, started: Long, targetSeconds: Int) {
        while (!cancelRequested) {
            val request = Request.Builder().url("$BACKEND/job?id=$id").get().build()
            client.newCall(request).execute().use { response ->
                if (!response.isSuccessful) throw IllegalStateException("YuE2 job disappeared")
                when (JSONObject(response.body?.string().orEmpty()).optString("status")) {
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
            }
            delay(350)
        }
    }

    private fun fetchResult(id: String, started: Long, targetSeconds: Int) {
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
            _state.value = MusicState.Complete(
                file = out,
                score = replay?.optString("abc").orEmpty(),
                lmSeed = replay?.optLong("lm_seed", -1L) ?: -1L,
                acousticSeed = replay?.optLong("seed", -1L) ?: -1L,
                targetSeconds = targetSeconds,
                elapsedMillis = System.currentTimeMillis() - started,
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
