package io.github.xororz.localdream.service

import android.content.Context
import android.net.Uri
import io.github.xororz.localdream.utils.BackendDiagnostics
import java.io.File
import java.io.IOException
import java.util.UUID
import java.util.concurrent.TimeUnit
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.MultipartBody
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.asRequestBody
import org.json.JSONArray
import org.json.JSONObject

data class BreezeVoiceProfile(
    val id: String, val name: String, val description: String,
    val transcript: String, val instruction: String, val createdAt: Long,
    val favorite: Boolean = false,
)

/** Private, on-device voice profiles; native .breeze codecs survive server restarts. */
object BreezeVoiceLibrary {
    private const val URL = "http://127.0.0.1:8082"
    private val client = OkHttpClient.Builder()
        .connectTimeout(8, TimeUnit.SECONDS)
        .readTimeout(120, TimeUnit.SECONDS)
        .writeTimeout(120, TimeUnit.SECONDS).build()

    fun referenceFile(context: Context, profile: BreezeVoiceProfile): File? =
        File(directory(context), "${profile.id}.wav").takeIf { it.isFile }

    fun directory(context: Context): File =
        File(context.filesDir, "breeze_voice_profiles").apply { mkdirs() }

    fun load(context: Context): List<BreezeVoiceProfile> = runCatching {
        val file = File(directory(context), "voices.json")
        if (!file.isFile) return emptyList()
        val arr = JSONArray(file.readText())
        (0 until arr.length()).map { arr.getJSONObject(it) }.map { x ->
            BreezeVoiceProfile(
                x.getString("id"), x.optString("name"), x.optString("description"),
                x.optString("transcript"), x.optString("instruction"),
                x.optLong("createdAt"), x.optBoolean("favorite", false),
            )
        }.filter { File(directory(context), "${it.id}.breeze").isFile }
    }.getOrElse { emptyList() }

    private fun save(context: Context, profiles: List<BreezeVoiceProfile>) {
        val arr = JSONArray()
        profiles.forEach { p ->
            arr.put(JSONObject().apply {
                put("id", p.id); put("name", p.name)
                put("description", p.description); put("transcript", p.transcript)
                put("instruction", p.instruction); put("createdAt", p.createdAt)
                put("favorite", p.favorite)
            })
        }
        val directory = directory(context)
        val tmp = File(directory, "voices.json.tmp")
        tmp.writeText(arr.toString())
        val dest = File(directory, "voices.json")
        if (!tmp.renameTo(dest)) {
            tmp.copyTo(dest, overwrite = true)
            tmp.delete()
        }
    }

    suspend fun register(
        context: Context, source: Uri, name: String, description: String,
        transcript: String, instruction: String,
    ): BreezeVoiceProfile = withContext(Dispatchers.IO) {
        val display = name.trim().take(64)
        val spoken = transcript.trim().take(6000)
        require(display.isNotEmpty()) { "Enter a voice name" }
        require(spoken.isNotEmpty()) { "Exact reference transcript is required" }
        val id = "voice_" + UUID.randomUUID().toString().replace("-", "")
        val tmp = File(context.cacheDir, "$id.wav")
        try {
            val stream = context.contentResolver.openInputStream(source)
                ?: throw IOException("Cannot read reference audio")
            stream.use { input ->
                tmp.outputStream().use { output ->
                    val bytes = input.copyTo(output)
                    require(bytes in 44..(20L * 1024L * 1024L)) {
                        "Reference WAV must be smaller than 20 MB"
                    }
                }
            }
            tmp.inputStream().use {
                val header = ByteArray(12)
                require(it.read(header) == 12 &&
                    String(header, 0, 4) == "RIFF" &&
                    String(header, 8, 4) == "WAVE") {
                    "Reference must be WAV; use one with clear, clean speech"
                }
            }
            val payload = MultipartBody.Builder().setType(MultipartBody.FORM)
                .addFormDataPart("name", id)
                .addFormDataPart("ref_text", spoken)
                .addFormDataPart("ref_audio", tmp.name,
                    tmp.asRequestBody("audio/wav".toMediaType()))
                .build()
            client.newCall(Request.Builder().url("$URL/v1/voices")
                .post(payload).build()).execute().use { response ->
                val result = response.body?.string().orEmpty()
                if (!response.isSuccessful)
                    throw IOException("Native Breeze voice encoder: HTTP ${response.code} ${result.take(180)}")
                if (JSONObject(result).optString("id") != id)
                    throw IOException("Native voice ID verification failed")
            }
            require(File(directory(context), "$id.breeze").isFile) {
                "Native encoder did not save the voice"
            }
            // Keep the source recording for instant previews, independently from history.
            // The native .breeze reference stays untouched for conditioning.
            val preview = File(directory(context), "$id.wav")
            tmp.copyTo(preview, overwrite = true)
            BackendDiagnostics.append(context, "BREEZE_VOICE",
                "register id=$id bytes=${preview.length()} transcript_chars=${spoken.length} preview=local")
            val profile = BreezeVoiceProfile(id, display,
                description.trim().take(800), spoken, instruction.trim().take(1200),
                System.currentTimeMillis())
            synchronized(this@BreezeVoiceLibrary) {
                save(context, listOf(profile) + load(context))
            }
            profile
        } finally {
            tmp.delete()
        }
    }

    suspend fun setFavorite(context: Context, id: String, value: Boolean) =
        withContext(Dispatchers.IO) {
            synchronized(this@BreezeVoiceLibrary) {
                save(context, load(context).map {
                    if (it.id == id) it.copy(favorite = value) else it
                })
            }
        }

    suspend fun update(context: Context, profile: BreezeVoiceProfile) =
        withContext(Dispatchers.IO) {
            synchronized(this@BreezeVoiceLibrary) {
                save(context, load(context).map { if (it.id == profile.id) profile else it })
            }
        }

    suspend fun delete(context: Context, id: String) = withContext(Dispatchers.IO) {
        require(Regex("[A-Za-z0-9_-]+").matches(id)) { "Invalid voice ID" }
        synchronized(this@BreezeVoiceLibrary) {
            val file = File(directory(context), "$id.breeze")
            if (file.isFile && !file.delete()) throw IOException("Could not delete voice")
            File(directory(context), "$id.wav").delete()
            save(context, load(context).filterNot { it.id == id })
            BackendDiagnostics.append(context, "BREEZE_VOICE", "delete id=$id")
        }
        runCatching {
            client.newCall(Request.Builder().url("$URL/v1/voices/$id")
                .delete().build()).execute().close()
        }
    }
}
