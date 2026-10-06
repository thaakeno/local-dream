package io.github.xororz.localdream.service

import android.content.Context
import android.os.Build
import io.github.xororz.localdream.utils.Http
import java.io.File
import java.io.FileOutputStream
import java.security.MessageDigest
import java.util.Locale
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.withContext
import okhttp3.Request
import org.json.JSONObject

/**
 * Shared native Breeze vocoder accelerator.
 *
 * Every Breeze GGUF precision emits the same 16-codebook audio-token format,
 * so Q4/Q6/Q8/F16/DD2/DD4 all reuse one QAIRT context for the phone's SoC.
 */
object BreezeQnnVocoderArtifact {
    private const val RELEASE_TAG = "breeze-qnn-vocoder-v1"
    private const val BASE_URL =
        "https://github.com/thaakeno/local-dream/releases/download/" + RELEASE_TAG
    private const val DIR = "breeze_qnn_vocoder/v1"

    sealed class Status {
        object Checking : Status()
        data class Unsupported(val detected: String) : Status()
        data class Missing(val soc: String) : Status()
        data class Downloading(
            val soc: String,
            val received: Long,
            val total: Long,
        ) : Status() {
            val progress: Float?
                get() = if (total > 0L) {
                    (received.toFloat() / total.toFloat()).coerceIn(0f, 1f)
                } else null
        }
        data class Ready(val soc: String, val file: File) : Status()
        data class Error(val soc: String?, val message: String) : Status()
    }

    private val _status = MutableStateFlow<Status>(Status.Checking)
    val status: StateFlow<Status> = _status

    private fun directory(context: Context) = File(context.filesDir, DIR)

    private fun deviceFingerprint(): String {
        val parts = ArrayList<String>()
        if (Build.VERSION.SDK_INT >= 31) parts += Build.SOC_MODEL.orEmpty()
        parts += Build.HARDWARE.orEmpty()
        parts += Build.BOARD.orEmpty()
        parts += Build.DEVICE.orEmpty()
        parts += Build.PRODUCT.orEmpty()
        return parts.joinToString(" ").uppercase(Locale.US)
    }

    fun detectedSoc(): String = deviceFingerprint().trim().ifBlank { "unknown" }

    fun supportedSoc(): String? {
        val detected = deviceFingerprint()
        return listOf("SM8850", "SM8750", "SM8650", "SM8550", "SM8475", "SM8450", "SM8350")
            .firstOrNull { detected.contains(it) }
    }

    fun localFile(context: Context): File? {
        val soc = supportedSoc() ?: return null
        val marker = File(directory(context), "installed.json")
        if (!marker.isFile) return null
        return runCatching {
            val json = JSONObject(marker.readText())
            if (json.optString("soc") != soc) return@runCatching null
            val file = File(directory(context), json.getString("file"))
            val expected = json.optLong("bytes", -1L)
            if (!file.isFile || (expected > 0L && file.length() != expected)) null else file
        }.getOrNull()
    }

    fun refresh(context: Context) {
        val soc = supportedSoc()
        val installed = localFile(context)
        _status.value = when {
            soc == null -> Status.Unsupported(detectedSoc())
            installed != null -> Status.Ready(soc, installed)
            else -> Status.Missing(soc)
        }
    }

    suspend fun download(context: Context) = withContext(Dispatchers.IO) {
        val soc = supportedSoc()
        if (soc == null) {
            _status.value = Status.Unsupported(detectedSoc())
            return@withContext
        }

        runCatching {
            val manifestRequest = Request.Builder()
                .url(BASE_URL + "/manifest.json")
                .get()
                .build()
            val spec = Http.client.newCall(manifestRequest).execute().use { response ->
                if (!response.isSuccessful) {
                    error("Accelerator manifest is not available yet (HTTP " + response.code + ")")
                }
                val body = response.body?.string()
                    ?: error("Empty accelerator manifest")
                JSONObject(body).getJSONObject("files").getJSONObject(soc)
            }

            val fileName = spec.getString("file")
            val expectedBytes = spec.getLong("bytes")
            val expectedSha = spec.getString("sha256").lowercase(Locale.US)
            val dir = directory(context)
            dir.mkdirs()
            val target = File(dir, fileName)
            val part = File(dir, fileName + ".part")
            part.delete()

            val request = Request.Builder()
                .url(BASE_URL + "/" + fileName)
                .get()
                .build()
            Http.client.newCall(request).execute().use { response ->
                if (!response.isSuccessful) {
                    error("Accelerator download failed (HTTP " + response.code + ")")
                }
                val body = response.body ?: error("Empty accelerator download")
                val total = body.contentLength().takeIf { it > 0L } ?: expectedBytes
                val digest = MessageDigest.getInstance("SHA-256")
                var received = 0L
                body.byteStream().use { input ->
                    FileOutputStream(part).use { output ->
                        val buffer = ByteArray(1024 * 1024)
                        while (true) {
                            val count = input.read(buffer)
                            if (count < 0) break
                            output.write(buffer, 0, count)
                            digest.update(buffer, 0, count)
                            received += count
                            _status.value = Status.Downloading(soc, received, total)
                        }
                    }
                }
                if (received != expectedBytes) {
                    error(
                        "Accelerator download was incomplete (" +
                            received + "/" + expectedBytes + " bytes)",
                    )
                }
                val actualSha = digest.digest().joinToString("") { "%02x".format(it.toInt() and 0xff) }
                if (actualSha != expectedSha) error("Accelerator checksum mismatch")
            }

            if (target.exists()) target.delete()
            if (!part.renameTo(target)) {
                part.copyTo(target, overwrite = true)
                part.delete()
            }

            // The compiled context is a derivative model artifact. Keep the
            // upstream license and the required derivative NOTICE beside it so
            // every in-app recipient receives the distribution terms too.
            listOf(
                "LICENSE-Breeze-TTS-2.txt",
                "NOTICE-Breeze-QNN-Vocoder.txt",
            ).forEach { legalName ->
                val legalRequest = Request.Builder()
                    .url(BASE_URL + "/" + legalName)
                    .get()
                    .build()
                Http.client.newCall(legalRequest).execute().use { response ->
                    if (!response.isSuccessful) {
                        error(
                            "Required accelerator license file failed to download (HTTP " +
                                response.code + ")",
                        )
                    }
                    val body = response.body ?: error("Empty accelerator license file")
                    File(dir, legalName).outputStream().use { output ->
                        body.byteStream().use { input -> input.copyTo(output) }
                    }
                }
                if (!File(dir, legalName).isFile || File(dir, legalName).length() == 0L) {
                    error("Required accelerator license file is empty")
                }
            }

            File(dir, "installed.json").writeText(
                JSONObject()
                    .put("soc", soc)
                    .put("file", fileName)
                    .put("bytes", expectedBytes)
                    .put("sha256", expectedSha)
                    .toString(),
            )
            _status.value = Status.Ready(soc, target)
        }.onFailure { error ->
            _status.value = Status.Error(soc, error.message ?: "Accelerator download failed")
        }
    }
}
