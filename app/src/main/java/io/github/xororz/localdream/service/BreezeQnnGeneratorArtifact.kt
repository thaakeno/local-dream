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
 * V3 uses four independently linked single-graph QNN context binaries:
 * backbone prefill, backbone AR1 step, depth prefill, depth AR1 step.
 */
object BreezeQnnGeneratorArtifact {
    private const val RELEASE_TAG = "breeze-qnn-generator-sm8850-v3"
    private const val BASE_URL =
        "https://github.com/thaakeno/local-dream/releases/download/" + RELEASE_TAG
    private const val DIR = "breeze_qnn_generator/v3-sm8850"

    private const val DEPTH_PREFILL = "breeze-depth-prefill-sm8850-v81.bin"
    private const val DEPTH_STEP = "breeze-depth-step-sm8850-v81.bin"
    private const val BACKBONE_PREFILL = "breeze-backbone-prefill-sm8850-v81.bin"
    private const val BACKBONE_STEP = "breeze-backbone-step-sm8850-v81.bin"

    data class Install(
        val soc: String,
        val depthPrefillFile: File,
        val depthStepFile: File,
        val backbonePrefillFile: File,
        val backboneStepFile: File,
        val engine: String,
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
            val directory: File,
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

    fun localInstall(context: Context): Install? {
        val soc = supportedSoc() ?: return null
        val base = dir(context)
        val marker = File(base, "installed.json")
        if (!marker.isFile) return null
        return runCatching {
            val json = JSONObject(marker.readText())
            if (json.optInt("version") != 3 || json.optString("soc") != soc) {
                return@runCatching null
            }

            fun checked(nameKey: String, bytesKey: String): File {
                val file = File(base, json.getString(nameKey))
                val expected = json.getLong(bytesKey)
                if (!file.isFile || file.length() != expected) {
                    error("Incomplete QNN generator file: ${file.name}")
                }
                return file
            }

            Install(
                soc = soc,
                depthPrefillFile = checked("depthPrefillFile", "depthPrefillBytes"),
                depthStepFile = checked("depthStepFile", "depthStepBytes"),
                backbonePrefillFile = checked(
                    "backbonePrefillFile",
                    "backbonePrefillBytes",
                ),
                backboneStepFile = checked("backboneStepFile", "backboneStepBytes"),
                engine = json.optString("engine", "qnn-full-generator-separate-v3"),
                backboneMaxSeq = json.optInt("backboneMaxSeq", 512),
            )
        }.getOrNull()
    }

    fun refresh(context: Context) {
        val soc = supportedSoc()
        val install = localInstall(context)
        _status.value = when {
            soc == null -> Status.Unsupported(detectedSoc())
            install != null -> Status.Ready(soc, dir(context), install.engine)
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
                root.optInt("version") != 3 ||
                root.optInt("soc_model") != 87 ||
                root.optString("htp_arch") != "V81"
            ) {
                error("Generator manifest is not native SM8850/V81 v3")
            }

            val spec = root.getJSONObject("files").getJSONObject(soc)
            val depthPrefillSpec = parseContextSpec(spec.getJSONObject("depth_prefill"))
            val depthStepSpec = parseContextSpec(spec.getJSONObject("depth_step"))
            val backbonePrefillSpec =
                parseContextSpec(spec.getJSONObject("backbone_prefill"))
            val backboneStepSpec = parseContextSpec(spec.getJSONObject("backbone_step"))
            val all = listOf(
                Triple(DEPTH_PREFILL, depthPrefillSpec, 0),
                Triple(DEPTH_STEP, depthStepSpec, 1),
                Triple(BACKBONE_PREFILL, backbonePrefillSpec, 2),
                Triple(BACKBONE_STEP, backboneStepSpec, 3),
            )
            val totalAll = all.sumOf { it.second.bytes }
            val destination = dir(context).apply { mkdirs() }
            var completedBefore = 0L
            val downloaded = ArrayList<File>(4)
            for ((name, item, _) in all) {
                downloaded += downloadContext(
                    soc,
                    destination,
                    name,
                    item,
                    completedBefore,
                    totalAll,
                )
                completedBefore += item.bytes
            }

            val engine = root.optString("engine", "qnn-full-generator-separate-v3")
            File(destination, "installed.json").writeText(
                JSONObject()
                    .put("version", 3)
                    .put("soc", soc)
                    .put("engine", engine)
                    .put("depthPrefillFile", DEPTH_PREFILL)
                    .put("depthPrefillBytes", downloaded[0].length())
                    .put("depthStepFile", DEPTH_STEP)
                    .put("depthStepBytes", downloaded[1].length())
                    .put("backbonePrefillFile", BACKBONE_PREFILL)
                    .put("backbonePrefillBytes", downloaded[2].length())
                    .put("backboneStepFile", BACKBONE_STEP)
                    .put("backboneStepBytes", downloaded[3].length())
                    .put("backboneMaxSeq", root.optInt("backbone_max_seq", 512))
                    .toString(),
            )

            _status.value = Status.Ready(soc, destination, engine)
        }.onFailure { error ->
            val destination = dir(context)
            listOf(DEPTH_PREFILL, DEPTH_STEP, BACKBONE_PREFILL, BACKBONE_STEP)
                .forEach { File(destination, "$it.part").delete() }
            _status.value = Status.Error(
                soc,
                error.message ?: "Full QNN generator download failed",
            )
        }
    }
}
