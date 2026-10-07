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
import org.json.JSONArray
import org.json.JSONObject

/**
 * Optional SM8850/V81 full QNN generator accelerator.
 *
 * V2 installs two context binaries:
 *  - Qwen3 Breeze backbone (prefill buckets + batch-1/batch-2 decode)
 *  - residual depth decoder (batch-1/batch-2)
 */
object BreezeQnnGeneratorArtifact {
    private const val RELEASE_TAG = "breeze-qnn-generator-sm8850-v2"
    private const val BASE_URL =
        "https://github.com/thaakeno/local-dream/releases/download/" + RELEASE_TAG
    private const val DIR = "breeze_qnn_generator/v2-sm8850"
    private const val DEPTH_CONTEXT_NAME = "breeze-depth-sm8850-v81.bin"
    private const val BACKBONE_CONTEXT_NAME = "breeze-backbone-sm8850-v81.bin"

    data class Install(
        val soc: String,
        val depthContextFile: File,
        val backboneContextFile: File,
        val engine: String,
        val depthGraphNames: List<String>,
        val backboneGraphNames: List<String>,
        val backboneMaxSeq: Int,
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
        data class Ready(
            val soc: String,
            val depthFile: File,
            val backboneFile: File,
            val engine: String,
        ) : Status()
        data class Error(val soc: String?, val message: String) : Status()
    }

    private data class ContextSpec(
        val bytes: Long,
        val sha256: String,
        val parts: JSONArray,
    )

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

    private fun jsonStrings(array: JSONArray): List<String> =
        List(array.length()) { array.getString(it) }

    fun localInstall(context: Context): Install? {
        val soc = supportedSoc() ?: return null
        val marker = File(dir(context), "installed.json")
        if (!marker.isFile) return null
        return runCatching {
            val json = JSONObject(marker.readText())
            if (json.optInt("version") != 2 || json.optString("soc") != soc) {
                return@runCatching null
            }
            val depth = File(dir(context), json.getString("depthContextFile"))
            val backbone = File(dir(context), json.getString("backboneContextFile"))
            if (
                !depth.isFile ||
                !backbone.isFile ||
                depth.length() != json.getLong("depthContextBytes") ||
                backbone.length() != json.getLong("backboneContextBytes")
            ) {
                return@runCatching null
            }
            Install(
                soc = soc,
                depthContextFile = depth,
                backboneContextFile = backbone,
                engine = json.optString("engine", "qnn-full-generator-kv-v2"),
                depthGraphNames = jsonStrings(json.getJSONArray("depthGraphNames")),
                backboneGraphNames = jsonStrings(json.getJSONArray("backboneGraphNames")),
                backboneMaxSeq = json.optInt("backboneMaxSeq", 512),
            )
        }.getOrNull()
    }

    fun refresh(context: Context) {
        val soc = supportedSoc()
        val install = localInstall(context)
        _status.value = when {
            soc == null -> Status.Unsupported(detectedSoc())
            install != null -> Status.Ready(
                soc,
                install.depthContextFile,
                install.backboneContextFile,
                install.engine,
            )
            else -> Status.Missing(soc)
        }
    }

    private fun sha256Hex(bytes: ByteArray): String =
        bytes.joinToString("") { "%02x".format(it.toInt() and 0xff) }

    private fun parseContextSpec(json: JSONObject): ContextSpec =
        ContextSpec(
            bytes = json.getLong("bytes"),
            sha256 = json.getString("sha256").lowercase(Locale.US),
            parts = json.getJSONArray("parts"),
        )

    private suspend fun downloadContext(
        soc: String,
        destination: File,
        fileName: String,
        spec: ContextSpec,
        completedBefore: Long,
        totalAll: Long,
    ): File {
        var listedBytes = 0L
        for (i in 0 until spec.parts.length()) {
            listedBytes += spec.parts.getJSONObject(i).getLong("bytes")
        }
        if (listedBytes != spec.bytes) error("QNN context part sizes do not match")

        val target = File(destination, fileName)
        val temp = File(destination, "$fileName.part")
        temp.delete()
        val wholeDigest = MessageDigest.getInstance("SHA-256")
        var completed = 0L

        FileOutputStream(temp).use { output ->
            for (i in 0 until spec.parts.length()) {
                val part = spec.parts.getJSONObject(i)
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
                                completedBefore + completed + received,
                                totalAll,
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

        if (completed != spec.bytes) error("Incomplete QNN generator context")
        if (sha256Hex(wholeDigest.digest()) != spec.sha256) {
            error("QNN generator context checksum mismatch")
        }
        if (target.exists()) target.delete()
        if (!temp.renameTo(target)) {
            temp.copyTo(target, overwrite = true)
            temp.delete()
        }
        return target
    }

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
                if (!response.isSuccessful) error("Full QNN generator release is not ready yet")
                JSONObject(response.body?.string() ?: error("Empty generator manifest"))
            }
            if (
                root.optInt("version") != 2 ||
                root.optInt("soc_model") != 87 ||
                root.optString("htp_arch") != "V81"
            ) {
                error("Generator manifest is not native SM8850/V81 v2")
            }

            val spec = root.getJSONObject("files").getJSONObject(soc)
            val depthSpec = parseContextSpec(spec.getJSONObject("depth_context"))
            val backboneSpec = parseContextSpec(spec.getJSONObject("backbone_context"))
            val totalAll = depthSpec.bytes + backboneSpec.bytes
            val destination = dir(context).apply { mkdirs() }

            val depth = downloadContext(
                soc,
                destination,
                DEPTH_CONTEXT_NAME,
                depthSpec,
                0L,
                totalAll,
            )
            val backbone = downloadContext(
                soc,
                destination,
                BACKBONE_CONTEXT_NAME,
                backboneSpec,
                depthSpec.bytes,
                totalAll,
            )

            val engine = root.optString("engine", "qnn-full-generator-kv-v2")
            val depthGraphs = root.getJSONArray("depth_graph_names")
            val backboneGraphs = root.getJSONArray("backbone_graph_names")
            File(destination, "installed.json").writeText(
                JSONObject()
                    .put("version", 2)
                    .put("soc", soc)
                    .put("engine", engine)
                    .put("depthContextFile", DEPTH_CONTEXT_NAME)
                    .put("depthContextBytes", depth.length())
                    .put("backboneContextFile", BACKBONE_CONTEXT_NAME)
                    .put("backboneContextBytes", backbone.length())
                    .put("depthGraphNames", depthGraphs)
                    .put("backboneGraphNames", backboneGraphs)
                    .put("backboneMaxSeq", root.optInt("backbone_max_seq", 512))
                    .toString(),
            )

            _status.value = Status.Ready(soc, depth, backbone, engine)
        }.onFailure { error ->
            val destination = dir(context)
            File(destination, "$DEPTH_CONTEXT_NAME.part").delete()
            File(destination, "$BACKBONE_CONTEXT_NAME.part").delete()
            _status.value = Status.Error(
                soc,
                error.message ?: "Full QNN generator download failed",
            )
        }
    }
}
