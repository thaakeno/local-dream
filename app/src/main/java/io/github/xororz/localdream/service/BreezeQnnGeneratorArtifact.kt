package io.github.xororz.localdream.service

import android.content.Context
import android.os.Build
import io.github.xororz.localdream.utils.Http
import java.io.File
import java.io.FileOutputStream
import java.security.MessageDigest
import java.util.Locale
import kotlin.coroutines.cancellation.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.withContext
import okhttp3.Request
import org.json.JSONArray
import org.json.JSONObject

/**
 * Optional SM8850/V81 full QNN generator accelerator.
 *
 * V4 reuses the proven batch-1 depth context and keeps backbone prompt/AR1
 * as separate contexts. Downloads are resumable and can be driven by the
 * shared ModelDownloadService so Breeze gets the same progress UI/notification
 * path as the other large models.
 */
object BreezeQnnGeneratorArtifact {
    const val DOWNLOAD_MODEL_ID = "breeze-qnn-generator-sm8850-v4"
    const val DOWNLOAD_MODEL_NAME = "Breeze Full QNN Generator"

    private const val RELEASE_TAG = "breeze-qnn-generator-sm8850-v4"
    private const val BASE_URL =
        "https://github.com/thaakeno/local-dream/releases/download/" + RELEASE_TAG
    private const val DIR = "breeze_qnn_generator/v4-sm8850"

    private const val DEPTH = "breeze-depth-sm8850-v81.bin"
    private const val BACKBONE_PREFILL = "breeze-backbone-prefill-sm8850-v81.bin"
    private const val BACKBONE_STEP = "breeze-backbone-step-sm8850-v81.bin"

    data class Install(
        val soc: String,
        val depthContextFile: File,
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

    data class TransferProgress(
        val received: Long,
        val total: Long,
        val currentFileName: String,
    )

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
            if (json.optInt("version") != 4 || json.optString("soc") != soc) {
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
                depthContextFile = checked("depthContextFile", "depthContextBytes"),
                backbonePrefillFile =
                    checked("backbonePrefillFile", "backbonePrefillBytes"),
                backboneStepFile = checked("backboneStepFile", "backboneStepBytes"),
                engine = json.optString(
                    "engine",
                    "qnn-full-generator-depth-b1-backbone-b2-v4",
                ),
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

    private fun hashExisting(file: File, digest: MessageDigest) {
        if (!file.isFile || file.length() == 0L) return
        file.inputStream().buffered(1024 * 1024).use { input ->
            val buffer = ByteArray(1024 * 1024)
            while (true) {
                val count = input.read(buffer)
                if (count < 0) break
                digest.update(buffer, 0, count)
            }
        }
    }

    private suspend fun downloadContext(
        soc: String,
        destination: File,
        fileName: String,
        spec: ContextSpec,
        completedBefore: Long,
        totalAll: Long,
        onProgress: ((TransferProgress) -> Unit)?,
    ): File {
        var listedBytes = 0L
        for (i in 0 until spec.parts.length()) {
            listedBytes += spec.parts.getJSONObject(i).getLong("bytes")
        }
        if (listedBytes != spec.bytes) error("QNN context part sizes do not match")

        val target = File(destination, fileName)
        if (target.isFile && target.length() == spec.bytes) {
            val received = completedBefore + spec.bytes
            _status.value = Status.Downloading(soc, received, totalAll)
            onProgress?.invoke(TransferProgress(received, totalAll, fileName))
            return target
        }

        val temp = File(destination, "$fileName.part")
        if (temp.exists() && temp.length() > spec.bytes) temp.delete()

        var existing = if (temp.isFile) temp.length() else 0L
        val wholeDigest = MessageDigest.getInstance("SHA-256")
        hashExisting(temp, wholeDigest)

        var partStart = 0L
        for (i in 0 until spec.parts.length()) {
            currentCoroutineContext().ensureActive()
            val part = spec.parts.getJSONObject(i)
            val remoteName = part.getString("file")
            val partBytes = part.getLong("bytes")
            val partEnd = partStart + partBytes

            if (existing >= partEnd) {
                partStart = partEnd
                continue
            }

            val resumeOffset = (existing - partStart).coerceAtLeast(0L)
            val request = Request.Builder()
                .url("$BASE_URL/$remoteName")
                .apply {
                    if (resumeOffset > 0L) header("Range", "bytes=$resumeOffset-")
                }
                .get()
                .build()

            Http.client.newCall(request).execute().use { response ->
                if (resumeOffset > 0L && response.code == 200) {
                    temp.delete()
                    return downloadContext(
                        soc,
                        destination,
                        fileName,
                        spec,
                        completedBefore,
                        totalAll,
                        onProgress,
                    )
                }
                if (!response.isSuccessful || (resumeOffset > 0L && response.code != 206)) {
                    error("Generator part download failed (HTTP ${response.code})")
                }

                val body = response.body ?: error("Empty generator part")
                FileOutputStream(temp, true).use { output ->
                    body.byteStream().use { input ->
                        val buffer = ByteArray(1024 * 1024)
                        while (true) {
                            currentCoroutineContext().ensureActive()
                            val count = input.read(buffer)
                            if (count < 0) break
                            output.write(buffer, 0, count)
                            wholeDigest.update(buffer, 0, count)
                            existing += count
                            if (existing > spec.bytes) {
                                error("Generator context exceeded expected size")
                            }
                            val received = completedBefore + existing
                            _status.value = Status.Downloading(soc, received, totalAll)
                            onProgress?.invoke(
                                TransferProgress(received, totalAll, remoteName),
                            )
                        }
                    }
                }
            }

            if (existing != partEnd) {
                error("Incomplete generator part $remoteName")
            }
            partStart = partEnd
        }

        if (existing != spec.bytes) error("Incomplete QNN generator context")
        if (sha256Hex(wholeDigest.digest()) != spec.sha256) {
            temp.delete()
            error("QNN generator context checksum mismatch")
        }

        if (target.exists()) target.delete()
        if (!temp.renameTo(target)) {
            temp.copyTo(target, overwrite = true)
            temp.delete()
        }
        return target
    }

    suspend fun download(
        context: Context,
        onProgress: ((TransferProgress) -> Unit)? = null,
    ): Boolean = withContext(Dispatchers.IO) {
        val soc = supportedSoc()
        if (soc == null) {
            _status.value = Status.Unsupported(detectedSoc())
            return@withContext false
        }

        try {
            val root = Http.client.newCall(
                Request.Builder().url("$BASE_URL/manifest.json").get().build(),
            ).execute().use { response ->
                if (!response.isSuccessful) error("Full QNN generator release is not ready yet")
                JSONObject(response.body?.string() ?: error("Empty generator manifest"))
            }

            if (
                root.optInt("version") != 4 ||
                root.optInt("soc_model") != 87 ||
                root.optString("htp_arch") != "V81"
            ) {
                error("Generator manifest is not native SM8850/V81 v4")
            }

            val spec = root.getJSONObject("files").getJSONObject(soc)
            val depthSpec = parseContextSpec(spec.getJSONObject("depth"))
            val backbonePrefillSpec =
                parseContextSpec(spec.getJSONObject("backbone_prefill"))
            val backboneStepSpec =
                parseContextSpec(spec.getJSONObject("backbone_step"))

            val all = listOf(
                DEPTH to depthSpec,
                BACKBONE_PREFILL to backbonePrefillSpec,
                BACKBONE_STEP to backboneStepSpec,
            )
            val totalAll = all.sumOf { it.second.bytes }
            val destination = dir(context).apply { mkdirs() }
            var completedBefore = 0L
            val downloaded = ArrayList<File>(3)

            for ((name, item) in all) {
                downloaded += downloadContext(
                    soc,
                    destination,
                    name,
                    item,
                    completedBefore,
                    totalAll,
                    onProgress,
                )
                completedBefore += item.bytes
            }

            val engine = root.optString(
                "engine",
                "qnn-full-generator-depth-b1-backbone-b2-v4",
            )
            File(destination, "installed.json").writeText(
                JSONObject()
                    .put("version", 4)
                    .put("soc", soc)
                    .put("engine", engine)
                    .put("depthContextFile", DEPTH)
                    .put("depthContextBytes", downloaded[0].length())
                    .put("backbonePrefillFile", BACKBONE_PREFILL)
                    .put("backbonePrefillBytes", downloaded[1].length())
                    .put("backboneStepFile", BACKBONE_STEP)
                    .put("backboneStepBytes", downloaded[2].length())
                    .put("backboneMaxSeq", root.optInt("backbone_max_seq", 512))
                    .toString(),
            )

            _status.value = Status.Ready(soc, destination, engine)
            true
        } catch (cancelled: CancellationException) {
            throw cancelled
        } catch (error: Throwable) {
            _status.value = Status.Error(
                soc,
                error.message ?: "Full QNN generator download failed",
            )
            false
        }
    }
}
