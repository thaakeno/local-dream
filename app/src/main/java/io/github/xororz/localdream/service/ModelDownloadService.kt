package io.github.xororz.localdream.service

import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Intent
import android.net.TrafficStats
import android.os.Build
import android.os.IBinder
import android.os.SystemClock
import android.util.Log
import androidx.core.app.NotificationCompat
import io.github.xororz.localdream.BuildConfig
import io.github.xororz.localdream.R
import io.github.xororz.localdream.data.GenerationPreferences
import io.github.xororz.localdream.data.Model
import io.github.xororz.localdream.utils.DownloadDiagnostics
import io.github.xororz.localdream.utils.Http
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.io.RandomAccessFile
import java.util.Locale
import java.util.concurrent.TimeUnit
import java.util.zip.ZipInputStream
import kotlin.coroutines.cancellation.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.async
import kotlinx.coroutines.cancel
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import okhttp3.Call
import okhttp3.Request

class ModelDownloadService : Service() {
    private val serviceScope = CoroutineScope(Dispatchers.IO + Job())
    private var downloadJob: Job? = null
    @Volatile private var activeCall: Call? = null
    @Volatile private var pauseRequested = false
    @Volatile private var cancelRequested = false
    private var activeRequest: DownloadRequest? = null

    private val hyperOsFocusCapability by lazy { HyperOsSuperIsland.detect(this) }
    private var lastSystemNotificationUpdateMs = 0L

    private data class DownloadRequest(
        val modelId: String,
        val modelName: String,
        val fileUrl: String,
        val isZip: Boolean,
        val modelType: String,
        val fileNames: List<String>,
        val markerFile: String?,
        val inferenceProfile: String?,
        val targetFileName: String?,
    )

    private data class RemoteInfo(
        val size: Long,
        val xetHash: String? = null,
        val xetRefreshUrl: String? = null,
    )

    private class XetDownloadException(message: String) : IOException(message)

    private data class ThroughputSample(
        val timeMs: Long,
        val bytes: Long,
    )

    private class RollingThroughputEstimator(
        private val windowMs: Long,
    ) {
        private val samples = ArrayDeque<ThroughputSample>()

        fun sample(timeMs: Long, bytes: Long): Long {
            samples.addLast(ThroughputSample(timeMs, bytes))
            while (samples.size > 2 && timeMs - samples.first().timeMs > windowMs) {
                samples.removeFirst()
            }
            if (samples.size < 2) return 0L

            val first = samples.first()
            val last = samples.last()
            val elapsedMs = (last.timeMs - first.timeMs).coerceAtLeast(1L)
            val deltaBytes = (last.bytes - first.bytes).coerceAtLeast(0L)
            return (deltaBytes * 1000.0 / elapsedMs).toLong().coerceAtLeast(0L)
        }
    }


    private val notificationManager by lazy {
        getSystemService(NOTIFICATION_SERVICE) as NotificationManager
    }

    private val client = Http.client.newBuilder()
        .connectTimeout(30, TimeUnit.SECONDS)
        .readTimeout(5, TimeUnit.MINUTES)
        .build()

    private val metadataClient = client.newBuilder()
        .followRedirects(false)
        .followSslRedirects(false)
        .build()

    companion object {
        private const val TAG = "ModelDownloadService"
        private const val NOTIFICATION_CHANNEL_ID = "model_download_channel"
        private const val NOTIFICATION_ID = 2001

        private val _downloadState = MutableStateFlow<DownloadState>(DownloadState.Idle)
        val downloadState: StateFlow<DownloadState> = _downloadState

        const val ACTION_START_DOWNLOAD = "action_start_download"
        const val ACTION_CANCEL_DOWNLOAD = "action_cancel_download"
        const val ACTION_PAUSE_DOWNLOAD = "action_pause_download"
        const val ACTION_RESUME_DOWNLOAD = "action_resume_download"

        const val EXTRA_MODEL_ID = "model_id"
        const val EXTRA_MODEL_NAME = "model_name"
        const val EXTRA_FILE_URL = "file_url"
        const val EXTRA_IS_ZIP = "is_zip"
        const val EXTRA_MODEL_TYPE = "model_type"
        const val TYPE_SD = "sd"
        const val TYPE_UPSCALER = "upscaler"
        const val TYPE_MULTI_FILE = "multi_file"
        const val EXTRA_FILE_NAMES = "file_names"
        const val EXTRA_MARKER_FILE = "marker_file"
        const val EXTRA_INFERENCE_PROFILE = "inference_profile"
        const val EXTRA_TARGET_FILE_NAME = "target_file_name"
    }

    sealed class DownloadState {
        object Idle : DownloadState()

        data class Downloading(
            val modelId: String,
            val progress: Float,
            val downloadedBytes: Long,
            val totalBytes: Long,
            val bytesPerSecond: Long,
            val networkBytesPerSecond: Long = 0L,
            val etaSeconds: Long?,
            val currentFileName: String? = null,
            val usingXet: Boolean = false,
            val xetTransferBytes: Long = 0L,
            val xetTransferTotalBytes: Long = 0L,
        ) : DownloadState()

        data class Paused(
            val modelId: String,
            val progress: Float,
            val downloadedBytes: Long,
            val totalBytes: Long,
            val currentFileName: String? = null,
            val usingXet: Boolean = false,
        ) : DownloadState()

        data class Extracting(val modelId: String) : DownloadState()
        data class Success(val modelId: String) : DownloadState()
        data class Error(val modelId: String, val message: String) : DownloadState()
    }

    override fun onCreate() {
        super.onCreate()
        createNotificationChannel()
        val island = hyperOsFocusCapability
        DownloadDiagnostics.info(
            this,
            "Download service started • app=${BuildConfig.VERSION_NAME} (${BuildConfig.VERSION_CODE}) • " +
                "commit=${BuildConfig.GIT_SHA.take(7)} • repo=${BuildConfig.GIT_REPOSITORY} • " +
                "xetNative=${XetNative.available} • hyperOsFocusProtocol=${island.protocolVersion} • " +
                "focusPermission=${island.hasFocusPermission}",
        )
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        when (intent?.action) {
            ACTION_START_DOWNLOAD -> {
                val modelId = intent.getStringExtra(EXTRA_MODEL_ID) ?: return START_NOT_STICKY
                val request = DownloadRequest(
                    modelId = modelId,
                    modelName = intent.getStringExtra(EXTRA_MODEL_NAME) ?: modelId,
                    fileUrl = intent.getStringExtra(EXTRA_FILE_URL) ?: return START_NOT_STICKY,
                    isZip = intent.getBooleanExtra(EXTRA_IS_ZIP, false),
                    modelType = intent.getStringExtra(EXTRA_MODEL_TYPE) ?: TYPE_SD,
                    fileNames = intent.getStringArrayListExtra(EXTRA_FILE_NAMES).orEmpty(),
                    markerFile = intent.getStringExtra(EXTRA_MARKER_FILE),
                    inferenceProfile = intent.getStringExtra(EXTRA_INFERENCE_PROFILE),
                    targetFileName = intent.getStringExtra(EXTRA_TARGET_FILE_NAME),
                )
                activeRequest = request
                pauseRequested = false
                cancelRequested = false
                startForeground(NOTIFICATION_ID, createNotification(request.modelName, 0f))
                startDownload(request)
            }

            ACTION_PAUSE_DOWNLOAD -> pauseDownload()
            ACTION_RESUME_DOWNLOAD -> resumeDownload()
            ACTION_CANCEL_DOWNLOAD -> cancelDownload()
        }
        return START_NOT_STICKY
    }

    private fun startDownload(request: DownloadRequest) {
        activeCall?.cancel()
        if (XetNative.available) runCatching { XetNative.nativeCancel() }
        downloadJob?.cancel()

        pauseRequested = false
        cancelRequested = false

        downloadJob = serviceScope.launch {
            var tempFile: File? = null
            var extractTempDir: File? = null

            try {
                val existingState = _downloadState.value
                if (existingState !is DownloadState.Paused) {
                    _downloadState.value = DownloadState.Downloading(
                        modelId = request.modelId,
                        progress = 0f,
                        downloadedBytes = 0,
                        totalBytes = 0,
                        bytesPerSecond = 0,
                        etaSeconds = null,
                    )
                }

                val tempDir = File(filesDir, "temp_downloads").apply { mkdirs() }

                if (request.modelType == TYPE_MULTI_FILE) {
                    val xetSetting = GenerationPreferences(this@ModelDownloadService)
                        .getXetAcceleratedDownloads()
                    val isOfficialHub =
                        request.fileUrl.startsWith("https://huggingface.co", ignoreCase = true)
                    val xetEnabled = xetSetting && isOfficialHub && XetNative.available

                    DownloadDiagnostics.info(
                        this@ModelDownloadService,
                        "Package start model=${request.modelId} files=${request.fileNames.size} " +
                            "xetSetting=$xetSetting officialHub=$isOfficialHub " +
                            "xetNative=${XetNative.available} xetEnabled=$xetEnabled",
                    )

                    downloadPackageFiles(
                        modelId = request.modelId,
                        modelName = request.modelName,
                        baseUrl = request.fileUrl,
                        fileNames = request.fileNames,
                        markerFile = request.markerFile,
                        inferenceProfile = request.inferenceProfile,
                        preferXet = xetEnabled,
                    )
                    completeDownload(request.modelId, request.modelName)
                    return@launch
                }

                // Single files use the same Hub metadata/Xet path as package
                // files, so pasted HF resolve URLs get native Xet acceleration,
                // correct byte progress and resumable HTTP fallback.
                tempFile = File(tempDir, "${request.modelId}.part")
                val xetSetting = GenerationPreferences(this@ModelDownloadService)
                    .getXetAcceleratedDownloads()
                val preferXet =
                    xetSetting && isHuggingFaceHubUrl(request.fileUrl) && XetNative.available
                val remote = probeRemote(request.fileUrl, preferXet)
                val canXet =
                    preferXet &&
                        remote.xetHash != null &&
                        remote.xetRefreshUrl != null &&
                        remote.size > 0L

                if (canXet) {
                    try {
                        downloadFileXet(
                            hash = remote.xetHash!!,
                            refreshUrl = remote.xetRefreshUrl!!,
                            destFile = tempFile,
                            modelId = request.modelId,
                            modelName = request.modelName,
                            currentFileName = request.targetFileName,
                            expectedSize = remote.size,
                            packageOffset = 0L,
                            packageTotal = remote.size,
                        )
                    } catch (e: XetDownloadException) {
                        DownloadDiagnostics.warn(
                            this@ModelDownloadService,
                            "Direct Xet failed; falling back to resumable HTTP: ${e.message}",
                            e,
                        )
                        downloadFileHttp(
                            url = request.fileUrl,
                            destFile = tempFile,
                            modelId = request.modelId,
                            modelName = request.modelName,
                            currentFileName = request.targetFileName,
                            expectedSize = remote.size,
                        )
                    }
                } else {
                    downloadFileHttp(
                        url = request.fileUrl,
                        destFile = tempFile,
                        modelId = request.modelId,
                        modelName = request.modelName,
                        currentFileName = request.targetFileName,
                        expectedSize = remote.size,
                    )
                }

                when (request.modelType) {
                    TYPE_SD -> {
                        val modelDir = File(getModelsDir(), request.modelId)
                        if (request.isZip) {
                            if (modelDir.exists()) modelDir.deleteRecursively()
                            modelDir.mkdirs()
                            extractTempDir = File(tempDir, "${request.modelId}_extract").apply {
                                deleteRecursively()
                                mkdirs()
                            }
                            _downloadState.value = DownloadState.Extracting(request.modelId)
                            updateNotification(request.modelName, 0f, isExtracting = true)
                            unzipFile(tempFile, extractTempDir)
                            extractTempDir.listFiles()?.forEach { file ->
                                file.renameTo(File(modelDir, file.name))
                            }
                            extractTempDir.delete()
                            extractTempDir = null
                        } else {
                            modelDir.mkdirs()
                            val sourceName = request.targetFileName
                                ?.substringAfterLast('/')
                                ?.takeIf { it.isNotBlank() }
                                ?: request.fileUrl.substringBefore('?').substringAfterLast('/')
                                    .takeIf { it.isNotBlank() }
                                ?: "model.bin"
                            val safeName = sourceName.replace(
                                Regex("""[^A-Za-z0-9._-]+"""),
                                "_",
                            )
                            val target = File(modelDir, safeName)
                            if (target.exists()) target.delete()
                            if (!tempFile.renameTo(target)) {
                                tempFile.copyTo(target, overwrite = true)
                                tempFile.delete()
                            }
                            tempFile = null
                        }
                    }

                    TYPE_UPSCALER -> {
                        val upscalerDir = File(getModelsDir(), request.modelId).apply {
                            if (!exists()) mkdirs()
                        }
                        val targetFile = File(upscalerDir, Model.UPSCALER_FILE_NAME)
                        if (targetFile.exists()) targetFile.delete()
                        if (!tempFile.renameTo(targetFile)) {
                            tempFile.copyTo(targetFile, overwrite = true)
                        }
                    }
                }

                tempFile?.delete()
                tempFile = null
                completeDownload(request.modelId, request.modelName)
            } catch (e: CancellationException) {
                handleInterrupted(request)
            } catch (e: Exception) {
                if (pauseRequested || cancelRequested) {
                    handleInterrupted(request)
                } else {
                    Log.e(TAG, "Download failed", e)
                    DownloadDiagnostics.error(
                        this@ModelDownloadService,
                        "Download failed model=${request.modelId}: ${e.message}",
                        e,
                    )
                    tempFile?.takeIf { request.modelType != TYPE_MULTI_FILE }?.delete()
                    extractTempDir?.deleteRecursively()

                    _downloadState.value =
                        DownloadState.Error(request.modelId, e.message ?: getString(R.string.unknown_error))
                    updateNotification(request.modelName, 0f, error = e.message)

                    delay(3000)
                    _downloadState.value = DownloadState.Idle
                    activeRequest = null
                    stopForeground(STOP_FOREGROUND_REMOVE)
                    stopSelf()
                }
            } finally {
                activeCall = null
            }
        }
    }

    private suspend fun completeDownload(modelId: String, modelName: String) {
        _downloadState.value = DownloadState.Success(modelId)
        updateNotification(modelName, 1f, success = true)
        delay(2000)
        _downloadState.value = DownloadState.Idle
        activeRequest = null
        stopForeground(STOP_FOREGROUND_REMOVE)
        stopSelf()
    }

    private fun handleInterrupted(request: DownloadRequest) {
        activeCall = null
        if (pauseRequested && !cancelRequested) {
            val state = _downloadState.value
            val paused = when (state) {
                is DownloadState.Downloading -> DownloadState.Paused(
                    modelId = state.modelId,
                    progress = state.progress,
                    downloadedBytes = state.downloadedBytes,
                    totalBytes = state.totalBytes,
                    currentFileName = state.currentFileName,
                    usingXet = state.usingXet,
                )

                is DownloadState.Paused -> state
                else -> DownloadState.Paused(
                    modelId = request.modelId,
                    progress = 0f,
                    downloadedBytes = 0L,
                    totalBytes = 0L,
                )
            }
            _downloadState.value = paused
            notificationManager.notify(
                NOTIFICATION_ID,
                createPausedNotification(
                    request.modelName,
                    paused.progress,
                    paused.downloadedBytes,
                    paused.totalBytes,
                ),
            )
        } else {
            discardPartialFiles(request.modelType, request.modelId)
            if (request.modelType != TYPE_MULTI_FILE) {
                File(filesDir, "temp_downloads/${request.modelId}.part").delete()
            }
            _downloadState.value = DownloadState.Idle
            activeRequest = null
            stopForeground(STOP_FOREGROUND_REMOVE)
            stopSelf()
        }
    }

    private suspend fun downloadPackageFiles(
        modelId: String,
        modelName: String,
        baseUrl: String,
        fileNames: List<String>,
        markerFile: String?,
        inferenceProfile: String?,
        preferXet: Boolean,
    ) = withContext(Dispatchers.IO) {
        require(fileNames.isNotEmpty()) { "empty package file list" }
        val modelDir = File(getModelsDir(), modelId).apply { mkdirs() }
        val base = baseUrl.removeSuffix("/")

        val parts = fileNames.map { entry ->
            val remote = entry.substringBefore('|')
            val local = entry.substringAfter('|', remote.substringAfterLast('/'))
            remote to local
        }

        val remoteInfo = parts.associate { (remote, _) ->
            val url = "$base/$remote"
            remote to probeRemote(url, preferXet)
        }
        val allSizesKnown = remoteInfo.values.all { it.size > 0L }
        val totalBytes = if (allSizesKnown) remoteInfo.values.sumOf { it.size } else 0L
        var completedBytes = 0L

        DownloadDiagnostics.info(
            this@ModelDownloadService,
            "Metadata complete model=$modelId total=" +
                if (totalBytes > 0) totalBytes.toString() else "unknown",
        )

        for ((remote, local) in parts) {
            val dest = File(modelDir, local)
            val part = File(modelDir, "$local.part")
            val info = remoteInfo[remote] ?: RemoteInfo(-1L)
            val expected = info.size
            val url = "$base/$remote"

            recoverInterruptedXetPart(part, expected)

            if (dest.exists() && dest.length() > 0L &&
                (expected <= 0L || dest.length() == expected)
            ) {
                completedBytes += dest.length()
                DownloadDiagnostics.info(
                    this@ModelDownloadService,
                    "Keeping completed file $local bytes=${dest.length()}",
                )
                continue
            }

            if (expected > 0 && part.exists() && part.length() > expected) {
                DownloadDiagnostics.warn(
                    this@ModelDownloadService,
                    "Partial file $local is larger than trusted Hub size " +
                        "(${part.length()} > $expected); restarting that file",
                )
                part.delete()
            }

            val canXet = preferXet &&
                expected > 0 &&
                info.xetHash != null &&
                info.xetRefreshUrl != null

            DownloadDiagnostics.info(
                this@ModelDownloadService,
                "File $local expected=" +
                    if (expected > 0) expected.toString() else "unknown" +
                    " partial=${if (part.exists()) part.length() else 0L} " +
                    "xetEligible=$canXet",
            )

            if (canXet) {
                try {
                    downloadFileXet(
                        hash = info.xetHash!!,
                        refreshUrl = info.xetRefreshUrl!!,
                        destFile = part,
                        modelId = modelId,
                        modelName = modelName,
                        currentFileName = local,
                        expectedSize = expected,
                        packageOffset = completedBytes,
                        packageTotal = totalBytes,
                    )
                } catch (e: XetDownloadException) {
                    Log.w(TAG, "Xet failed for $local, falling back to resumable HTTP: ${e.message}")
                    DownloadDiagnostics.warn(
                        this@ModelDownloadService,
                        "Xet failed file=$local; preserving partial and falling back to HTTP: ${e.message}",
                        e,
                    )
                    downloadFileHttp(
                        url = url,
                        destFile = part,
                        modelId = modelId,
                        modelName = modelName,
                        currentFileName = local,
                        expectedSize = expected,
                        packageOffset = completedBytes,
                        packageTotal = totalBytes,
                    )
                }
            } else {
                downloadFileHttp(
                    url = url,
                    destFile = part,
                    modelId = modelId,
                    modelName = modelName,
                    currentFileName = local,
                    expectedSize = expected,
                    packageOffset = completedBytes,
                    packageTotal = totalBytes,
                )
            }

            if (expected > 0 && part.length() != expected) {
                throw IOException("Incomplete $local: ${part.length()}/$expected")
            }
            if (dest.exists()) dest.delete()
            if (!part.renameTo(dest)) throw IOException("Failed to install $local")
            completedBytes += dest.length()
            DownloadDiagnostics.info(
                this@ModelDownloadService,
                "File complete $local bytes=${dest.length()}",
            )
        }

        if (!inferenceProfile.isNullOrBlank()) {
            val profileFile = File(modelDir, "inference_profile.conf")
            profileFile.writeText(inferenceProfile.trim() + "\n")
            DownloadDiagnostics.info(
                this@ModelDownloadService,
                "Installed inference profile model=$modelId bytes=${profileFile.length()}",
            )
        }
        if (!markerFile.isNullOrEmpty()) File(modelDir, markerFile).createNewFile()
    }

    private fun recoverInterruptedXetPart(part: File, expectedSize: Long) {
        val sidecar = File(part.absolutePath + ".xetresume")
        if (!sidecar.exists()) return

        val frontier = runCatching { sidecar.readText().trim().toLong() }.getOrNull()
        if (frontier == null || frontier < 0L || (expectedSize > 0L && frontier > expectedSize)) {
            DownloadDiagnostics.warn(
                this,
                "Invalid Xet resume marker for ${part.name}; restarting partial file",
            )
            part.delete()
            sidecar.delete()
            return
        }

        if (part.exists() && part.length() != frontier) {
            DownloadDiagnostics.info(
                this,
                "Recovering interrupted Xet partial ${part.name}: " +
                    "sparseLength=${part.length()} contiguous=$frontier",
            )
            RandomAccessFile(part, "rw").use { it.setLength(frontier) }
        }
        sidecar.delete()
    }

    private fun isHuggingFaceHubUrl(url: String): Boolean =
        runCatching {
            val parsed = Request.Builder().url(url).build().url
            isHuggingFaceHubHost(parsed.host)
        }.getOrDefault(false)

    private fun probeRemote(url: String, wantXet: Boolean): RemoteInfo {
        val parsed = runCatching { Request.Builder().url(url).build().url }.getOrNull()
            ?: return RemoteInfo(-1L)

        if (!isHuggingFaceHubHost(parsed.host)) {
            return RemoteInfo(remoteSize(url))
        }

        return runCatching {
            // Match huggingface_hub's metadata behavior:
            // HEAD /resolve, Accept-Encoding: identity, follow redirects only while
            // they stay on the Hub, and stop before CDN/object storage. On a
            // redirect, Content-Length describes the redirect response itself and
            // MUST NOT be treated as the model file size.
            var currentUrl = url

            for (hop in 0 until 8) {
                var resolved: RemoteInfo? = null
                var nextHubUrl: String? = null

                val request = Request.Builder()
                    .url(currentUrl)
                    .head()
                    .header("Accept-Encoding", "identity")
                    .header("X-HF-Download-Counter", "1")
                    .build()

                metadataClient.newCall(request).execute().use { response ->
                    val location = response.header("Location")
                    val isRedirect = response.code in 300..399 && !location.isNullOrBlank()

                    if (!response.isSuccessful && !isRedirect) {
                        throw IOException("Hub metadata HTTP ${response.code}")
                    }

                    val trustedSize = response.header("X-Linked-Size")?.toLongOrNull()
                        ?: if (!isRedirect) {
                            response.header("Content-Length")?.toLongOrNull()
                        } else {
                            null
                        }

                    val hash = response.header("X-Xet-Hash")
                    val refresh = extractXetRefreshUrl(response)

                    DownloadDiagnostics.info(
                        this@ModelDownloadService,
                        "HF metadata hop=$hop code=${response.code} host=${response.request.url.host} " +
                            "size=${trustedSize ?: "unknown"} xetHash=${!hash.isNullOrBlank()} " +
                            "xetAuth=${!refresh.isNullOrBlank()} redirect=$isRedirect",
                    )

                    if (wantXet && !hash.isNullOrBlank() && !refresh.isNullOrBlank()) {
                        resolved = RemoteInfo(
                            size = trustedSize ?: -1L,
                            xetHash = hash,
                            xetRefreshUrl = refresh,
                        )
                    } else if (isRedirect) {
                        val next = response.request.url.resolve(location!!)
                        if (next != null && isSameOrHubHost(response.request.url.host, next.host)) {
                            nextHubUrl = next.toString()
                        } else {
                            // This is the final CDN/storage redirect. Its metadata
                            // belongs to the real file only through X-Linked-Size.
                            resolved = RemoteInfo(trustedSize ?: -1L)
                        }
                    } else {
                        resolved = RemoteInfo(trustedSize ?: -1L)
                    }
                }

                if (resolved != null) {
                    if (wantXet && (resolved!!.xetHash == null || resolved!!.xetRefreshUrl == null)) {
                        DownloadDiagnostics.warn(
                            this@ModelDownloadService,
                            "Xet metadata unavailable for Hub file; using resumable HTTP",
                        )
                    }
                    return@runCatching resolved!!
                }

                if (nextHubUrl == null) break
                currentUrl = nextHubUrl!!
            }

            DownloadDiagnostics.warn(
                this@ModelDownloadService,
                "Hub metadata exceeded redirect resolution without usable metadata",
            )
            RemoteInfo(-1L)
        }.getOrElse {
            DownloadDiagnostics.warn(
                this@ModelDownloadService,
                "Hub metadata probe failed; HTTP fallback will discover size from GET: ${it.message}",
                it,
            )
            RemoteInfo(-1L)
        }
    }

    private fun extractXetRefreshUrl(response: okhttp3.Response): String? {
        val linkValue = response.header("Link")
        val linkRoute = linkValue
            ?.split(',')
            ?.firstOrNull { part ->
                part.contains("rel=\"xet-auth\"", ignoreCase = true) ||
                    part.contains("rel=xet-auth", ignoreCase = true)
            }
            ?.substringBefore(';')
            ?.trim()
            ?.removePrefix("<")
            ?.removeSuffix(">")

        val raw = linkRoute ?: response.header("X-Xet-Refresh-Route") ?: return null
        return response.request.url.resolve(raw)?.toString()
            ?: raw.takeIf { it.startsWith("https://") || it.startsWith("http://") }
    }

    private fun isSameOrHubHost(sourceHost: String, targetHost: String): Boolean =
        sourceHost.equals(targetHost, ignoreCase = true) || isHuggingFaceHubHost(targetHost)

    private fun isHuggingFaceHubHost(host: String): Boolean {
        val normalized = host.lowercase(Locale.US)
        return normalized == "huggingface.co" ||
            normalized == "hf.co" ||
            normalized == "hub-ci.huggingface.co"
    }

    private fun remoteSize(url: String): Long {
        val parsed = runCatching { Request.Builder().url(url).build().url }.getOrNull()
            ?: return -1L

        if (isHuggingFaceHubHost(parsed.host)) {
            return probeRemote(url, wantXet = false).size
        }

        val request = Request.Builder()
            .url(url)
            .head()
            .header("Accept-Encoding", "identity")
            .build()
        return runCatching {
            client.newCall(request).execute().use { response ->
                if (!response.isSuccessful) {
                    -1L
                } else {
                    parseContentRangeTotal(response.header("Content-Range"))
                        ?: response.header("Content-Length")?.toLongOrNull()
                        ?: -1L
                }
            }
        }.getOrDefault(-1L)
    }

    private fun parseContentRangeTotal(value: String?): Long? {
        if (value.isNullOrBlank()) return null
        val total = value.substringAfterLast('/', missingDelimiterValue = "").trim()
        return total.takeIf { it != "*" }?.toLongOrNull()
    }

    private fun parseContentRangeStart(value: String?): Long? {
        if (value.isNullOrBlank()) return null
        val range = value.substringAfter("bytes ", missingDelimiterValue = "")
            .substringBefore('/')
        return range.substringBefore('-').toLongOrNull()
    }

    private suspend fun downloadFileHttp(
        url: String,
        destFile: File,
        modelId: String,
        modelName: String,
        currentFileName: String? = null,
        expectedSize: Long = -1L,
        packageOffset: Long = 0L,
        packageTotal: Long = 0L,
    ) = withContext(Dispatchers.IO) {
        var existing = if (destFile.exists()) destFile.length() else 0L
        val trustedMetadataSize = expectedSize.takeIf { it > 0L }

        if (trustedMetadataSize != null && existing == trustedMetadataSize) {
            val total = if (packageTotal > 0) packageTotal else packageOffset + trustedMetadataSize
            emitProgress(
                modelId, modelName, packageOffset + existing, total,
                0L, 0L, currentFileName, false,
            )
            return@withContext
        }

        if (trustedMetadataSize != null && existing > trustedMetadataSize) {
            DownloadDiagnostics.warn(
                this@ModelDownloadService,
                "HTTP partial exceeds trusted metadata for ${currentFileName ?: modelId}: " +
                    "$existing > $trustedMetadataSize; restarting file",
            )
            destFile.delete()
            existing = 0L
        }

        val builder = Request.Builder()
            .url(url)
            .header("Accept-Encoding", "identity")
        if (existing > 0L) builder.header("Range", "bytes=$existing-")

        val call = client.newCall(builder.build())
        activeCall = call

        call.execute().use { response ->
            if (response.code == 416) {
                val serverTotal = parseContentRangeTotal(response.header("Content-Range"))
                if (serverTotal != null && existing == serverTotal) {
                    DownloadDiagnostics.info(
                        this@ModelDownloadService,
                        "HTTP resume already complete file=${currentFileName ?: modelId} bytes=$existing",
                    )
                    return@use
                }
                throw IOException(
                    "Server rejected resume range for ${currentFileName ?: modelId} " +
                        "(local=$existing server=${serverTotal ?: "unknown"})",
                )
            }

            if (!response.isSuccessful) {
                throw IOException(getString(R.string.error_download_failed, response.code.toString()))
            }

            var append = existing > 0L && response.code == 206
            if (append) {
                val rangeStart = parseContentRangeStart(response.header("Content-Range"))
                if (rangeStart != null && rangeStart != existing) {
                    throw IOException(
                        "Resume range mismatch for ${currentFileName ?: modelId}: " +
                            "requested=$existing received=$rangeStart",
                    )
                }
            }

            if (existing > 0L && response.code == 200) {
                // Server ignored Range. Restart this file cleanly rather than
                // appending duplicate bytes.
                DownloadDiagnostics.warn(
                    this@ModelDownloadService,
                    "Server ignored Range for ${currentFileName ?: modelId}; restarting file from byte 0",
                )
                existing = 0L
                append = false
            }

            val body = response.body ?: throw IOException("Response body is null")
            val responseBytes = body.contentLength().takeIf { it > 0L }
            val rangeTotal = parseContentRangeTotal(response.header("Content-Range"))

            // Prefer totals learned from the actual GET response. Metadata size
            // is only a fallback when the body is chunked and carries no total.
            val responseDerivedTotal = rangeTotal ?: when {
                !append && responseBytes != null -> responseBytes
                append && responseBytes != null -> existing + responseBytes
                else -> null
            }
            val totalFileBytes = responseDerivedTotal ?: trustedMetadataSize ?: -1L
            val effectivePackageTotal = if (packageTotal > 0L) {
                packageTotal
            } else if (totalFileBytes > 0L) {
                packageOffset + totalFileBytes
            } else {
                0L
            }

            DownloadDiagnostics.info(
                this@ModelDownloadService,
                "HTTP start file=${currentFileName ?: modelId} code=${response.code} " +
                    "resume=$existing total=" +
                    if (totalFileBytes > 0) totalFileBytes.toString() else "unknown",
            )

            var downloadedThisRequest = 0L
            var lastUpdateTime = System.currentTimeMillis()
            var lastSampleBytes = 0L
            var smoothedBytesPerSecond = 0.0

            java.io.BufferedOutputStream(FileOutputStream(destFile, append), 1024 * 1024).use { output ->
                body.byteStream().buffered(1024 * 1024).use { input ->
                    val buffer = ByteArray(1024 * 1024)
                    while (true) {
                        val bytes = input.read(buffer)
                        if (bytes < 0) break
                        output.write(buffer, 0, bytes)
                        downloadedThisRequest += bytes

                        val now = System.currentTimeMillis()
                        if (now - lastUpdateTime >= 500) {
                            val elapsedMs = (now - lastUpdateTime).coerceAtLeast(1L)
                            val delta = downloadedThisRequest - lastSampleBytes
                            val instant = delta * 1000.0 / elapsedMs
                            smoothedBytesPerSecond = if (smoothedBytesPerSecond <= 0) {
                                instant
                            } else {
                                smoothedBytesPerSecond * 0.75 + instant * 0.25
                            }
                            lastUpdateTime = now
                            lastSampleBytes = downloadedThisRequest

                            val fileDone = existing + downloadedThisRequest
                            val reportedDone = packageOffset + fileDone
                            val speed = smoothedBytesPerSecond.toLong().coerceAtLeast(0L)
                            val eta = if (effectivePackageTotal > reportedDone && speed > 0) {
                                (effectivePackageTotal - reportedDone) / speed
                            } else {
                                null
                            }
                            emitProgress(
                                modelId, modelName, reportedDone, effectivePackageTotal,
                                speed, eta, currentFileName, false,
                            )
                        }
                    }
                }
            }

            val finalSize = destFile.length()
            val authoritativeTotal = responseDerivedTotal
            if (authoritativeTotal != null && finalSize != authoritativeTotal) {
                throw IOException(
                    "Incomplete HTTP download for ${currentFileName ?: modelId}: " +
                        "$finalSize/$authoritativeTotal",
                )
            }

            // If the GET was chunked, a trusted Hub X-Linked-Size remains useful
            // as an integrity check. Never use redirect Content-Length here.
            if (authoritativeTotal == null &&
                trustedMetadataSize != null &&
                finalSize != trustedMetadataSize
            ) {
                throw IOException(
                    "Incomplete HTTP download for ${currentFileName ?: modelId}: " +
                        "$finalSize/$trustedMetadataSize",
                )
            }

            DownloadDiagnostics.info(
                this@ModelDownloadService,
                "HTTP complete file=${currentFileName ?: modelId} bytes=$finalSize",
            )

            val finalTotal = if (packageTotal > 0) {
                packageTotal
            } else {
                packageOffset + finalSize
            }
            emitProgress(
                modelId, modelName, packageOffset + finalSize, finalTotal,
                smoothedBytesPerSecond.toLong(), 0L, currentFileName, false,
            )
        }
        activeCall = null
    }

    private suspend fun downloadFileXet(
        hash: String,
        refreshUrl: String,
        destFile: File,
        modelId: String,
        modelName: String,
        currentFileName: String,
        expectedSize: Long,
        packageOffset: Long,
        packageTotal: Long,
    ) = coroutineScope {
        val tuning = XetRuntimeTuning.from(this@ModelDownloadService)
        val effectivePackageTotal = if (packageTotal > 0L) {
            packageTotal
        } else {
            packageOffset + expectedSize
        }

        fun currentUidRxBytes(): Long? {
            val value = TrafficStats.getUidRxBytes(applicationInfo.uid)
            return value.takeIf { it != TrafficStats.UNSUPPORTED.toLong() && it >= 0L }
        }

        val xetRuntimeDir = File(cacheDir, "hf_xet_runtime").apply { mkdirs() }
        DownloadDiagnostics.info(
            this@ModelDownloadService,
            "Xet start file=$currentFileName size=$expectedSize " +
                "resume=${destFile.length()} stream=unordered progress=exact-logical " +
                "speed=logical+uid-rx cacheWritable=${xetRuntimeDir.canWrite()} " +
                "cacheFree=${xetRuntimeDir.usableSpace} " + tuning.summary(),
        )

        while (destFile.length() < expectedSize) {
            val offset = destFile.length()
            val logicalSpeedEstimator = RollingThroughputEstimator(windowMs = 20_000L)
            val etaSpeedEstimator = RollingThroughputEstimator(windowMs = 30_000L)
            val networkSpeedEstimator = RollingThroughputEstimator(windowMs = 4_000L)
            val attemptStartedAt = SystemClock.elapsedRealtime()
            var lastDiagnosticAt = attemptStartedAt

            val nativeJob = async(Dispatchers.IO) {
                XetNative.nativeDownload(
                    hash = hash,
                    size = expectedSize,
                    refreshUrl = refreshUrl,
                    destPath = destFile.absolutePath,
                    cacheDir = xetRuntimeDir.absolutePath,
                    offset = offset,
                    memoryBudgetBytes = tuning.memoryBudgetBytes,
                )
            }

            while (!nativeJob.isCompleted) {
                // 200 ms at ordinary phone download rates gives roughly MB-level UI
                // updates without flooding Compose or SystemUI.
                delay(200L)

                val reconstructedBytes =
                    runCatching { XetNative.nativeProgressBytes() }
                        .getOrDefault(offset)
                        .coerceIn(offset, expectedSize)
                val now = SystemClock.elapsedRealtime()
                val rxBytes = currentUidRxBytes()

                val logicalSpeed =
                    logicalSpeedEstimator.sample(now, reconstructedBytes).coerceAtLeast(0L)
                val etaSpeed =
                    etaSpeedEstimator.sample(now, reconstructedBytes).coerceAtLeast(0L)
                val networkSpeed =
                    if (rxBytes != null) {
                        networkSpeedEstimator.sample(now, rxBytes).coerceAtLeast(0L)
                    } else {
                        0L
                    }

                val reportedDone = packageOffset + reconstructedBytes
                val elapsedMs = now - attemptStartedAt
                val logicalDelta = reconstructedBytes - offset
                val etaReady = elapsedMs >= 15_000L &&
                    logicalDelta >= 8L * 1024L * 1024L
                val eta = if (
                    etaReady &&
                    effectivePackageTotal > reportedDone &&
                    etaSpeed > 0L
                ) {
                    (effectivePackageTotal - reportedDone) / etaSpeed
                } else {
                    null
                }

                emitProgress(
                    modelId = modelId,
                    modelName = modelName,
                    done = reportedDone,
                    total = effectivePackageTotal,
                    speed = logicalSpeed,
                    eta = eta,
                    currentFileName = currentFileName,
                    usingXet = true,
                    networkSpeed = networkSpeed,
                )

                if (now - lastDiagnosticAt >= 10_000L) {
                    DownloadDiagnostics.info(
                        this@ModelDownloadService,
                        "Xet progress file=$currentFileName " +
                            "logical=$reconstructedBytes/$expectedSize " +
                            "logicalBps=$logicalSpeed networkBps=$networkSpeed " +
                            "eta=${eta ?: "calculating"}",
                    )
                    lastDiagnosticAt = now
                }

                if (pauseRequested || cancelRequested) {
                    XetNative.nativeCancel()
                    break
                }
            }

            val result = nativeJob.await()

            if (pauseRequested || cancelRequested) {
                throw CancellationException("download interrupted")
            }
            if (result == 0) break
            if (result == 1) throw CancellationException("Xet cancelled")

            val nativeError = XetNative.nativeLastError().orEmpty()
            val message = nativeError.ifBlank { "unknown Xet error" }
            DownloadDiagnostics.warn(
                this@ModelDownloadService,
                "Xet native error file=$currentFileName result=$result error=$message",
            )
            throw XetDownloadException(message)
        }

        if (destFile.length() != expectedSize) {
            throw XetDownloadException(
                "Incomplete Xet download: ${destFile.length()}/$expectedSize",
            )
        }

        DownloadDiagnostics.info(
            this@ModelDownloadService,
            "Xet complete file=$currentFileName bytes=$expectedSize",
        )
        emitProgress(
            modelId = modelId,
            modelName = modelName,
            done = packageOffset + expectedSize,
            total = effectivePackageTotal,
            speed = 0L,
            eta = 0L,
            currentFileName = currentFileName,
            usingXet = true,
            networkSpeed = 0L,
        )
    }

    private fun emitProgress(
        modelId: String,
        modelName: String,
        done: Long,
        total: Long,
        speed: Long,
        eta: Long?,
        currentFileName: String?,
        usingXet: Boolean,
        xetTransferBytes: Long = 0L,
        xetTransferTotalBytes: Long = 0L,
        networkSpeed: Long = 0L,
    ) {
        val progress = if (total > 0) {
            (done.toDouble() / total.toDouble()).coerceIn(0.0, 1.0).toFloat()
        } else {
            0f
        }

        _downloadState.value = DownloadState.Downloading(
            modelId = modelId,
            progress = progress,
            downloadedBytes = done,
            totalBytes = total,
            bytesPerSecond = speed,
            networkBytesPerSecond = networkSpeed,
            etaSeconds = eta,
            currentFileName = currentFileName,
            usingXet = usingXet,
            xetTransferBytes = xetTransferBytes,
            xetTransferTotalBytes = xetTransferTotalBytes,
        )

        val status = buildDownloadNotificationStatus(
            done = done,
            total = total,
            bytesPerSecond = speed,
            etaSeconds = eta,
        )

        // App UI updates at 200 ms. SystemUI / Super Island only needs a 1 Hz
        // refresh; it is still the same foreground notification ID, never a duplicate.
        val now = SystemClock.elapsedRealtime()
        if (lastSystemNotificationUpdateMs == 0L || now - lastSystemNotificationUpdateMs >= 1_000L) {
            lastSystemNotificationUpdateMs = now
            updateNotification(
                modelName = modelName,
                progress = progress,
                statusText = status,
                downloadedBytes = done,
                totalBytes = total,
                speedBytesPerSecond = speed,
                etaSeconds = eta,
            )
        }
    }

    private suspend fun unzipFile(zipFile: File, destDir: File) = withContext(Dispatchers.IO) {
        ZipInputStream(zipFile.inputStream().buffered()).use { zis ->
            var entry = zis.nextEntry
            while (entry != null) {
                if (!entry.isDirectory) {
                    val fileName = entry.name.substringAfterLast('/')
                    if (fileName.isNotEmpty() && !fileName.startsWith(".") && !fileName.startsWith("__MACOSX")) {
                        val file = File(destDir, fileName)
                        java.io.BufferedOutputStream(FileOutputStream(file)).use { output ->
                            zis.copyTo(output)
                        }
                    }
                }
                zis.closeEntry()
                entry = zis.nextEntry
            }
        }
    }

    private fun buildDownloadNotificationStatus(
        done: Long,
        total: Long,
        bytesPerSecond: Long,
        etaSeconds: Long?,
    ): String {
        val amount = if (total > 0L) {
            "${formatBytesShort(done)} / ${formatBytesShort(total)}"
        } else {
            formatBytesShort(done)
        }
        val speed = if (bytesPerSecond > 0L) {
            String.format(Locale.US, "%.1f MB/s", bytesPerSecond / (1024.0 * 1024.0))
        } else {
            null
        }
        val eta = etaSeconds?.takeIf { it > 0L }?.let { "${formatEta(it)} left" }
        return listOfNotNull(amount, speed, eta).joinToString(" • ")
    }

    private fun formatBytesShort(bytes: Long): String = when {
        bytes < 1024L -> "$bytes B"
        bytes < 1024L * 1024L -> String.format(Locale.US, "%.0f KB", bytes / 1024.0)
        bytes < 1024L * 1024L * 1024L ->
            String.format(Locale.US, "%.0f MB", bytes / (1024.0 * 1024.0))
        else -> String.format(Locale.US, "%.2f GB", bytes / (1024.0 * 1024.0 * 1024.0))
    }

    private fun buildDownloadStatus(bytesPerSecond: Long, etaSeconds: Long?): String? {
        if (bytesPerSecond <= 0) return null
        val speed = String.format(Locale.US, "%.1f MB/s", bytesPerSecond / (1024.0 * 1024.0))
        val eta = etaSeconds?.let { formatEta(it) }
        return if (eta != null) "$speed • $eta remaining" else speed
    }

    private fun formatEta(seconds: Long): String {
        val safe = seconds.coerceAtLeast(0L)
        val hours = safe / 3600
        val minutes = (safe % 3600) / 60
        val secs = safe % 60
        return when {
            hours > 0 -> String.format(Locale.US, "%dh %02dm", hours, minutes)
            minutes > 0 -> String.format(Locale.US, "%dm %02ds", minutes, secs)
            else -> String.format(Locale.US, "%ds", secs)
        }
    }

    private fun pauseDownload() {
        val state = _downloadState.value
        if (state !is DownloadState.Downloading) return

        pauseRequested = true
        cancelRequested = false
        DownloadDiagnostics.info(
            this,
            "Pause requested model=${state.modelId} bytes=${state.downloadedBytes}",
        )
        _downloadState.value = DownloadState.Paused(
            modelId = state.modelId,
            progress = state.progress,
            downloadedBytes = state.downloadedBytes,
            totalBytes = state.totalBytes,
            currentFileName = state.currentFileName,
            usingXet = state.usingXet,
        )

        activeCall?.cancel()
        if (XetNative.available) runCatching { XetNative.nativeCancel() }
        downloadJob?.cancel()

        activeRequest?.let {
            notificationManager.notify(
                NOTIFICATION_ID,
                createPausedNotification(
                    it.modelName,
                    state.progress,
                    state.downloadedBytes,
                    state.totalBytes,
                ),
            )
        }
    }

    private fun resumeDownload() {
        val request = activeRequest ?: return
        if (_downloadState.value !is DownloadState.Paused) return

        pauseRequested = false
        cancelRequested = false
        DownloadDiagnostics.info(this, "Resume requested model=${request.modelId}")
        notificationManager.notify(
            NOTIFICATION_ID,
            createNotification(request.modelName, (_downloadState.value as DownloadState.Paused).progress),
        )
        startDownload(request)
    }

    private fun cancelDownload() {
        val request = activeRequest
        pauseRequested = false
        cancelRequested = true
        DownloadDiagnostics.info(this, "Cancel requested model=${request?.modelId ?: "unknown"}")
        activeCall?.cancel()
        if (XetNative.available) runCatching { XetNative.nativeCancel() }

        val job = downloadJob
        if (job?.isActive == true) {
            job.cancel()
        } else if (request != null) {
            discardPartialFiles(request.modelType, request.modelId)
            if (request.modelType != TYPE_MULTI_FILE) {
                File(filesDir, "temp_downloads/${request.modelId}.part").delete()
            }
            _downloadState.value = DownloadState.Idle
            activeRequest = null
            stopForeground(STOP_FOREGROUND_REMOVE)
            stopSelf()
        }
    }

    private fun discardPartialFiles(modelType: String, modelId: String) {
        if (modelType != TYPE_MULTI_FILE) return
        val modelDir = File(getModelsDir(), modelId)
        modelDir.listFiles { file -> file.isFile && file.name.endsWith(".part") }
            ?.forEach { it.delete() }
    }

    private fun getModelsDir(): File = File(filesDir, "models").apply {
        if (!exists()) mkdirs()
    }

    private fun createNotificationChannel() {
        val channel = NotificationChannel(
            NOTIFICATION_CHANNEL_ID,
            getString(R.string.model_download_channel),
            NotificationManager.IMPORTANCE_LOW,
        ).apply {
            description = getString(R.string.model_download_channel_desc)
        }
        notificationManager.createNotificationChannel(channel)
    }

    private fun createNotification(
        modelName: String,
        progress: Float,
        isExtracting: Boolean = false,
        statusText: String? = null,
        downloadedBytes: Long = 0L,
        totalBytes: Long = 0L,
        speedBytesPerSecond: Long = 0L,
        etaSeconds: Long? = null,
    ): android.app.Notification {
        val title = if (isExtracting) getString(R.string.extracting) else getString(R.string.downloading_model, modelName)
        val openAppIntent = packageManager.getLaunchIntentForPackage(packageName)?.apply {
            flags = Intent.FLAG_ACTIVITY_SINGLE_TOP or Intent.FLAG_ACTIVITY_NEW_TASK
        }
        val appPendingIntent = PendingIntent.getActivity(
            this, 0, openAppIntent, PendingIntent.FLAG_IMMUTABLE,
        )

        val pausePendingIntent = PendingIntent.getService(
            this,
            2,
            Intent(this, ModelDownloadService::class.java).apply { action = ACTION_PAUSE_DOWNLOAD },
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
        )
        val cancelPendingIntent = PendingIntent.getService(
            this,
            1,
            Intent(this, ModelDownloadService::class.java).apply { action = ACTION_CANCEL_DOWNLOAD },
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
        )

        val builder = NotificationCompat.Builder(this, NOTIFICATION_CHANNEL_ID)
            .setContentTitle(title)
            .setContentText(statusText)
            .setSmallIcon(android.R.drawable.stat_sys_download)
            .setProgress(100, (progress * 100).toInt(), isExtracting)
            .setOngoing(true)
            .setContentIntent(appPendingIntent)

        if (!isExtracting) {
            builder.addAction(
                android.R.drawable.ic_media_pause,
                getString(R.string.pause),
                pausePendingIntent,
            )
        }
        builder.addAction(
            android.R.drawable.ic_menu_close_clear_cancel,
            getString(R.string.cancel),
            cancelPendingIntent,
        )
        val notification = builder
            .setOnlyAlertOnce(true)
            .build()
        HyperOsSuperIsland.decorateDownload(
            context = this,
            notification = notification,
            capability = hyperOsFocusCapability,
            modelName = modelName,
            progress = progress,
            downloadedBytes = downloadedBytes,
            totalBytes = totalBytes,
            speedBytesPerSecond = speedBytesPerSecond,
            etaSeconds = etaSeconds,
            paused = false,
        )
        return notification
    }

    private fun createPausedNotification(
        modelName: String,
        progress: Float,
        downloadedBytes: Long,
        totalBytes: Long,
    ): android.app.Notification {
        val openAppIntent = packageManager.getLaunchIntentForPackage(packageName)?.apply {
            flags = Intent.FLAG_ACTIVITY_SINGLE_TOP or Intent.FLAG_ACTIVITY_NEW_TASK
        }
        val appPendingIntent = PendingIntent.getActivity(
            this,
            0,
            openAppIntent,
            PendingIntent.FLAG_IMMUTABLE,
        )
        val resumePendingIntent = PendingIntent.getService(
            this,
            3,
            Intent(this, ModelDownloadService::class.java).apply { action = ACTION_RESUME_DOWNLOAD },
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
        )
        val cancelPendingIntent = PendingIntent.getService(
            this,
            1,
            Intent(this, ModelDownloadService::class.java).apply { action = ACTION_CANCEL_DOWNLOAD },
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
        )

        val notification = NotificationCompat.Builder(this, NOTIFICATION_CHANNEL_ID)
            .setContentTitle(getString(R.string.download_paused))
            .setContentText(modelName)
            .setSmallIcon(android.R.drawable.stat_sys_download)
            .setProgress(100, (progress * 100).toInt(), false)
            .setOngoing(true)
            .setOnlyAlertOnce(true)
            .setContentIntent(appPendingIntent)
            .addAction(android.R.drawable.ic_media_play, getString(R.string.resume), resumePendingIntent)
            .addAction(
                android.R.drawable.ic_menu_close_clear_cancel,
                getString(R.string.cancel),
                cancelPendingIntent,
            )
            .build()

        HyperOsSuperIsland.decorateDownload(
            context = this,
            notification = notification,
            capability = hyperOsFocusCapability,
            modelName = modelName,
            progress = progress,
            downloadedBytes = downloadedBytes,
            totalBytes = totalBytes,
            speedBytesPerSecond = 0L,
            etaSeconds = null,
            paused = true,
        )
        return notification
    }

    private fun updateNotification(
        modelName: String,
        progress: Float,
        success: Boolean = false,
        error: String? = null,
        isExtracting: Boolean = false,
        statusText: String? = null,
        downloadedBytes: Long = 0L,
        totalBytes: Long = 0L,
        speedBytesPerSecond: Long = 0L,
        etaSeconds: Long? = null,
    ) {
        val notification = when {
            success -> NotificationCompat.Builder(this, NOTIFICATION_CHANNEL_ID)
                .setContentTitle(getString(R.string.download_complete))
                .setContentText(modelName)
                .setSmallIcon(android.R.drawable.stat_sys_download_done)
                .setOngoing(false)
                .build()

            error != null -> NotificationCompat.Builder(this, NOTIFICATION_CHANNEL_ID)
                .setContentTitle(getString(R.string.download_failed))
                .setContentText(error)
                .setSmallIcon(android.R.drawable.stat_notify_error)
                .setOngoing(false)
                .build()

            else -> createNotification(
                modelName = modelName,
                progress = progress,
                isExtracting = isExtracting,
                statusText = statusText,
                downloadedBytes = downloadedBytes,
                totalBytes = totalBytes,
                speedBytesPerSecond = speedBytesPerSecond,
                etaSeconds = etaSeconds,
            )
        }
        notificationManager.notify(NOTIFICATION_ID, notification)
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onTimeout(startId: Int) {
        super.onTimeout(startId)
        handleTimeout(0)
    }

    override fun onTimeout(startId: Int, fgsType: Int) {
        super.onTimeout(startId, fgsType)
        handleTimeout(fgsType)
    }

    private fun handleTimeout(fgsType: Int) {
        Log.e(TAG, "Foreground service timeout (fgsType=$fgsType)")
        DownloadDiagnostics.error(this, "Foreground service timeout fgsType=$fgsType")
        activeCall?.cancel()
        if (XetNative.available) runCatching { XetNative.nativeCancel() }
        downloadJob?.cancel()
        _downloadState.value = DownloadState.Error("timeout", "Foreground service timeout")
        stopForeground(STOP_FOREGROUND_REMOVE)
        stopSelf()
    }

    override fun onDestroy() {
        activeCall?.cancel()
        if (XetNative.available) runCatching { XetNative.nativeCancel() }
        serviceScope.cancel()
        super.onDestroy()
    }
}
