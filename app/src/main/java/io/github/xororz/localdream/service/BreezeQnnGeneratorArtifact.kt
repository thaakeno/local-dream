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
 * Optional SM8850/V81 QNN generator accelerator.
 *
 * The release asset may be split into sub-2GiB pieces. The installer streams
 * and verifies each part, then reconstructs one native QNN context binary.
 */
object BreezeQnnGeneratorArtifact {
    private const val RELEASE_TAG = "breeze-qnn-generator-sm8850-v1"
    private const val BASE_URL =
        "https://github.com/thaakeno/local-dream/releases/download/" + RELEASE_TAG
    private const val DIR = "breeze_qnn_generator/v1-sm8850"
    private const val CONTEXT_NAME = "breeze-generator-sm8850-v81.bin"

    data class Install(
        val soc: String,
        val contextFile: File,
        val engine: String,
        val graphNames: List<String>,
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
        data class Ready(val soc: String, val file: File, val engine: String) : Status()
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
            if (json.optInt("version") != 1 || json.optString("soc") != soc) {
                return@runCatching null
            }
            val contextFile = File(dir(context), json.getString("contextFile"))
            val contextBytes = json.getLong("contextBytes")
            if (!contextFile.isFile || contextFile.length() != contextBytes) {
                return@runCatching null
            }
            val namesJson = json.getJSONArray("graphNames")
            val graphNames = List(namesJson.length()) { namesJson.getString(it) }
            Install(
                soc = soc,
                contextFile = contextFile,
                engine = json.optString("engine", "qnn-generator-v1"),
                graphNames = graphNames,
            )
        }.getOrNull()
    }

    fun localFile(context: Context): File? = localInstall(context)?.contextFile

    fun refresh(context: Context) {
        val soc = supportedSoc()
        val install = localInstall(context)
        _status.value = when {
            soc == null -> Status.Unsupported(detectedSoc())
            install != null -> Status.Ready(soc, install.contextFile, install.engine)
            else -> Status.Missing(soc)
        }
    }

    private fun sha256Hex(bytes: ByteArray): String =
        bytes.joinToString("") { "%02x".format(it.toInt() and 0xff) }

    suspend fun download(context: Context) = withContext(Dispatchers.IO) {
        val soc = supportedSoc()
        if (soc == null) {
            _status.value = Status.Unsupported(detectedSoc())
            return@withContext
        }

        runCatching {
            val root = Http.client.newCall(
                Request.Builder().url("$BASE_URL/manifest.json").get().build(),
            ).execute().use { response ->
                if (!response.isSuccessful) error("QNN generator release is not ready yet")
                JSONObject(response.body?.string() ?: error("Empty generator manifest"))
            }
            if (
                root.optInt("version") != 1 ||
                root.optInt("soc_model") != 87 ||
                root.optString("htp_arch") != "V81"
            ) {
                error("Generator manifest is not native SM8850/V81")
            }

            val spec = root.getJSONObject("files").getJSONObject(soc)
            val contextSpec = spec.getJSONObject("context")
            val expectedBytes = contextSpec.getLong("bytes")
            val expectedSha = contextSpec.getString("sha256").lowercase(Locale.US)
            val parts = contextSpec.getJSONArray("parts")
            val engine = root.optString("engine", "qnn-generator-v1")
            val graphJson = root.getJSONArray("graph_names")
            val graphNames = List(graphJson.length()) { graphJson.getString(it) }
            if (parts.length() == 0 || graphNames.isEmpty()) error("Incomplete generator manifest")

            var totalPartsBytes = 0L
            for (i in 0 until parts.length()) {
                totalPartsBytes += parts.getJSONObject(i).getLong("bytes")
            }
            if (totalPartsBytes != expectedBytes) error("Generator part sizes do not match")

            val destination = dir(context)
            destination.mkdirs()
            val target = File(destination, CONTEXT_NAME)
            val temp = File(destination, "$CONTEXT_NAME.part")
            temp.delete()

            val wholeDigest = MessageDigest.getInstance("SHA-256")
            var completed = 0L
            FileOutputStream(temp).use { output ->
                for (i in 0 until parts.length()) {
                    val part = parts.getJSONObject(i)
                    val name = part.getString("file")
                    val partBytes = part.getLong("bytes")
                    val partSha = part.getString("sha256").lowercase(Locale.US)
                    val partDigest = MessageDigest.getInstance("SHA-256")
                    var received = 0L
                    Http.client.newCall(
                        Request.Builder().url("$BASE_URL/$name").get().build(),
                    ).execute().use { response ->
                        if (!response.isSuccessful) {
                            error("Generator part download failed (HTTP ${response.code})")
                        }
                        val body = response.body ?: error("Empty generator part")
                        body.byteStream().use { input ->
                            val buffer = ByteArray(1024 * 1024)
                            while (true) {
                                val count = input.read(buffer)
                                if (count < 0) break
                                output.write(buffer, 0, count)
                                partDigest.update(buffer, 0, count)
                                wholeDigest.update(buffer, 0, count)
                                received += count
                                _status.value = Status.Downloading(
                                    soc,
                                    completed + received,
                                    expectedBytes,
                                )
                            }
                        }
                    }
                    if (received != partBytes) error("Incomplete generator part $name")
                    if (sha256Hex(partDigest.digest()) != partSha) {
                        error("Generator checksum mismatch for $name")
                    }
                    completed += received
                }
            }

            if (completed != expectedBytes) error("Incomplete QNN generator context")
            if (sha256Hex(wholeDigest.digest()) != expectedSha) {
                error("QNN generator context checksum mismatch")
            }
            if (target.exists()) target.delete()
            if (!temp.renameTo(target)) {
                temp.copyTo(target, overwrite = true)
                temp.delete()
            }

            File(destination, "installed.json").writeText(
                JSONObject()
                    .put("version", 1)
                    .put("soc", soc)
                    .put("engine", engine)
                    .put("contextFile", CONTEXT_NAME)
                    .put("contextBytes", expectedBytes)
                    .put("graphNames", graphJson)
                    .toString(),
            )

            _status.value = Status.Ready(soc, target, engine)
        }.onFailure { error ->
            File(dir(context), "$CONTEXT_NAME.part").delete()
            _status.value = Status.Error(
                soc,
                error.message ?: "QNN generator download failed",
            )
        }
    }
}
