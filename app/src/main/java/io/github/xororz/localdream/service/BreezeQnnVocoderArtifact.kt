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

object BreezeQnnVocoderArtifact {
    private const val RELEASE_TAG = "breeze-qnn-vocoder-sm8850-v2"
    private const val BASE_URL =
        "https://github.com/thaakeno/local-dream/releases/download/" + RELEASE_TAG
    private const val DIR = "breeze_qnn_vocoder/v2-sm8850"

    data class Install(
        val soc: String,
        val contextFile: File,
        val lutFile: File,
    )

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

    private fun dir(context: Context) = File(context.filesDir, DIR)

    private fun fingerprint(): String {
        val parts = ArrayList<String>()
        if (Build.VERSION.SDK_INT >= 31) parts += Build.SOC_MODEL.orEmpty()
        parts += Build.HARDWARE.orEmpty()
        parts += Build.BOARD.orEmpty()
        parts += Build.DEVICE.orEmpty()
        parts += Build.PRODUCT.orEmpty()
        return parts.joinToString(" ").uppercase(Locale.US)
    }

    fun detectedSoc(): String = fingerprint().trim().ifBlank { "unknown" }

    fun supportedSoc(): String? =
        if (fingerprint().contains("SM8850")) "SM8850" else null

    fun localInstall(context: Context): Install? {
        val soc = supportedSoc() ?: return null
        val marker = File(dir(context), "installed.json")
        if (!marker.isFile) return null

        return runCatching {
            val json = JSONObject(marker.readText())
            if (json.optInt("version") != 2 || json.optString("soc") != soc) {
                return@runCatching null
            }

            val contextFile = File(dir(context), json.getString("contextFile"))
            val lutFile = File(dir(context), json.getString("lutFile"))
            val contextBytes = json.getLong("contextBytes")
            val lutBytes = json.getLong("lutBytes")

            if (
                !contextFile.isFile ||
                !lutFile.isFile ||
                contextFile.length() != contextBytes ||
                lutFile.length() != lutBytes
            ) {
                null
            } else {
                Install(soc, contextFile, lutFile)
            }
        }.getOrNull()
    }

    fun localFile(context: Context): File? = localInstall(context)?.contextFile

    fun refresh(context: Context) {
        val soc = supportedSoc()
        val install = localInstall(context)
        _status.value = when {
            soc == null -> Status.Unsupported(detectedSoc())
            install != null -> Status.Ready(soc, install.contextFile)
            else -> Status.Missing(soc)
        }
    }

    private suspend fun downloadFile(
        name: String,
        expectedBytes: Long,
        expectedSha: String,
        target: File,
        soc: String,
        base: Long,
        total: Long,
    ) {
        val part = File(target.parentFile, target.name + ".part")
        part.delete()

        val request = Request.Builder().url(BASE_URL + "/" + name).get().build()
        Http.client.newCall(request).execute().use { response ->
            if (!response.isSuccessful) {
                error("Accelerator download failed (HTTP " + response.code + ")")
            }

            val body = response.body ?: error("Empty accelerator download")
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
                        _status.value = Status.Downloading(soc, base + received, total)
                    }
                }
            }

            if (received != expectedBytes) {
                error("Incomplete accelerator download for " + name)
            }

            val actualSha = digest.digest().joinToString("") {
                "%02x".format(it.toInt() and 0xff)
            }
            if (actualSha != expectedSha.lowercase(Locale.US)) {
                error("Checksum mismatch for " + name)
            }
        }

        if (target.exists()) target.delete()
        if (!part.renameTo(target)) {
            part.copyTo(target, overwrite = true)
            part.delete()
        }
    }

    suspend fun download(context: Context) = withContext(Dispatchers.IO) {
        val soc = supportedSoc()
        if (soc == null) {
            _status.value = Status.Unsupported(detectedSoc())
            return@withContext
        }

        runCatching {
            val manifestRequest =
                Request.Builder().url(BASE_URL + "/manifest.json").get().build()
            val spec = Http.client.newCall(manifestRequest).execute().use { response ->
                if (!response.isSuccessful) error("Accelerator manifest unavailable")
                val body = response.body?.string() ?: error("Empty manifest")
                JSONObject(body).getJSONObject("files").getJSONObject(soc)
            }

            val contextSpec = spec.getJSONObject("context")
            val lutSpec = spec.getJSONObject("lut")

            val contextName = contextSpec.getString("file")
            val contextBytes = contextSpec.getLong("bytes")
            val contextSha = contextSpec.getString("sha256")

            val lutName = lutSpec.getString("file")
            val lutBytes = lutSpec.getLong("bytes")
            val lutSha = lutSpec.getString("sha256")
            val total = contextBytes + lutBytes

            val destination = dir(context)
            destination.mkdirs()
            val contextFile = File(destination, contextName)
            val lutFile = File(destination, lutName)

            downloadFile(
                contextName, contextBytes, contextSha,
                contextFile, soc, 0L, total,
            )
            downloadFile(
                lutName, lutBytes, lutSha,
                lutFile, soc, contextBytes, total,
            )

            listOf(
                "LICENSE-Breeze-TTS-2.txt",
                "NOTICE-Breeze-QNN-Vocoder.txt",
            ).forEach { name ->
                val request = Request.Builder().url(BASE_URL + "/" + name).get().build()
                Http.client.newCall(request).execute().use { response ->
                    if (!response.isSuccessful) error("License download failed")
                    val body = response.body ?: error("Empty license response")
                    File(destination, name).outputStream().use { output ->
                        body.byteStream().use { input -> input.copyTo(output) }
                    }
                }
            }

            File(destination, "installed.json").writeText(
                JSONObject()
                    .put("version", 2)
                    .put("soc", soc)
                    .put("contextFile", contextName)
                    .put("contextBytes", contextBytes)
                    .put("contextSha256", contextSha)
                    .put("lutFile", lutName)
                    .put("lutBytes", lutBytes)
                    .put("lutSha256", lutSha)
                    .toString(),
            )

            // Remove the old silent-output v1 artifact only after v2 has been
            // downloaded and verified completely.
            File(context.filesDir, "breeze_qnn_vocoder/v1").deleteRecursively()
            _status.value = Status.Ready(soc, contextFile)
        }.onFailure { error ->
            _status.value = Status.Error(
                soc,
                error.message ?: "Accelerator download failed",
            )
        }
    }
}
