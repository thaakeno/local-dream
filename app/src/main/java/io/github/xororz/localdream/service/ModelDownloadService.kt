package io.github.xororz.localdream.service

import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Intent
import android.os.Build
import android.os.IBinder
import android.os.PowerManager
import android.util.Log
import androidx.core.app.NotificationCompat
import io.github.xororz.localdream.R
import io.github.xororz.localdream.data.GenerationPreferences
import io.github.xororz.localdream.data.Model
import io.github.xororz.localdream.utils.Http
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
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

    private data class DownloadRequest(
        val modelId: String,
        val modelName: String,
        val fileUrl: String,
        val isZip: Boolean,
        val modelType: String,
        val fileNames: List<String>,
        val markerFile: String?,
    )

    private data class RemoteInfo(
        val size: Long,
        val xetHash: String? = null,
        val xetRefreshUrl: String? = null,
    )

    private class XetDownloadException(message: String) : IOException(message)

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
    }

    sealed class DownloadState {
        object Idle : DownloadState()

        data class Downloading(
            val modelId: String,
            val progress: Float,
            val downloadedBytes: Long,
            val totalBytes: Long,
            val bytesPerSecond: Long,
            val etaSeconds: Long?,
            val currentFileName: String? = null,
            val usingXet: Boolean = false,
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
                    val xetEnabled = GenerationPreferences(this@ModelDownloadService)
                        .getXetAcceleratedDownloads() &&
                        request.fileUrl.startsWith("https://huggingface.co", ignoreCase = true) &&
                        XetNative.available

                    downloadPackageFiles(
                        modelId = request.modelId,
                        modelName = request.modelName,
                        baseUrl = request.fileUrl,
                        fileNames = request.fileNames,
                        markerFile = request.markerFile,
                        preferXet = xetEnabled,
                    )
                    completeDownload(request.modelId, request.modelName)
                    return@launch
                }

                // Stable name makes ordinary single-file downloads resumable too.
                tempFile = File(tempDir, "${request.modelId}.part")
                downloadFileHttp(
                    url = request.fileUrl,
                    destFile = tempFile,
                    modelId = request.modelId,
                    modelName = request.modelName,
                    expectedSize = remoteSize(request.fileUrl),
                )

                when (request.modelType) {
                    TYPE_SD -> {
                        if (request.isZip) {
                            val modelDir = File(getModelsDir(), request.modelId)
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

                tempFile.delete()
                tempFile = null
                completeDownload(request.modelId, request.modelName)
            } catch (e: CancellationException) {
                handleInterrupted(request)
            } catch (e: Exception) {
                if (pauseRequested || cancelRequested) {
                    handleInterrupted(request)
                } else {
                    Log.e(TAG, "Download failed", e)
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
                createPausedNotification(request.modelName, paused.progress),
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
        val totalBytes = remoteInfo.values.sumOf { it.size.coerceAtLeast(0L) }
        var completedBytes = 0L

        for ((remote, local) in parts) {
            val dest = File(modelDir, local)
            val part = File(modelDir, "$local.part")
            val info = remoteInfo[remote] ?: RemoteInfo(-1L)
            val expected = info.size
            val url = "$base/$remote"

            if (dest.exists() && expected > 0 && dest.length() == expected) {
                completedBytes += expected
                continue
            }

            if (expected > 0 && part.exists() && part.length() > expected) {
                part.delete()
            }

            val canXet = preferXet &&
                expected > 0 &&
                info.xetHash != null &&
                info.xetRefreshUrl != null

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
            completedBytes += if (expected > 0) expected else dest.length()
        }

        if (!markerFile.isNullOrEmpty()) File(modelDir, markerFile).createNewFile()
    }

    private fun probeRemote(url: String, wantXet: Boolean): RemoteInfo {
        if (!wantXet) return RemoteInfo(remoteSize(url))

        return runCatching {
            // Hugging Face's Xet protocol exposes X-Xet-* metadata on the
            // resolve response. A /resolve/main/... URL can first redirect to
            // another Hub URL (/api/resolve-cache/...) before the Xet/LFS
            // redirect. Do NOT follow the final storage/CDN redirect, otherwise
            // the request drops onto the legacy HTTP download path.
            //
            // Use GET here, matching the Xet file-id protocol. Response bodies
            // are never consumed; redirects are handled manually.
            var currentUrl = url
            var bestSize = -1L

            for (hop in 0 until 8) {
                var nextHubUrl: String? = null
                val request = Request.Builder()
                    .url(currentUrl)
                    .get()
                    .header("X-HF-Download-Counter", "1")
                    .build()

                val resolved = metadataClient.newCall(request).execute().use { response ->
                    val size = response.header("X-Linked-Size")?.toLongOrNull()
                        ?: response.header("Content-Range")
                            ?.substringAfterLast('/')
                            ?.toLongOrNull()
                        ?: response.header("Content-Length")?.toLongOrNull()
                        ?: -1L
                    if (size > 0) bestSize = size

                    val hash = response.header("X-Xet-Hash")
                    if (!hash.isNullOrBlank()) {
                        val refresh = extractXetRefreshUrl(response, url)
                            ?: deriveXetRefreshUrl(url)

                        if (!refresh.isNullOrBlank()) {
                            Log.i(TAG, "Xet metadata resolved for $url (hop=$hop, size=$bestSize)")
                            return@use RemoteInfo(
                                size = if (bestSize > 0) bestSize else remoteSize(url),
                                xetHash = hash,
                                xetRefreshUrl = refresh,
                            )
                        }

                        Log.w(TAG, "Xet hash found but no refresh route could be resolved for $url")
                    }

                    val location = response.header("Location")
                    if (response.code in 300..399 && !location.isNullOrBlank()) {
                        val next = response.request.url.resolve(location)
                        if (next != null && isHuggingFaceHubHost(next.host)) {
                            nextHubUrl = next.toString()
                        }
                    }
                    null
                }

                if (resolved != null) return@runCatching resolved
                if (nextHubUrl == null) break
                currentUrl = nextHubUrl!!
            }

            Log.w(TAG, "Xet metadata not found for $url; falling back to HTTP")
            RemoteInfo(if (bestSize > 0) bestSize else remoteSize(url))
        }.getOrElse {
            Log.w(TAG, "Xet metadata probe failed; using HTTP", it)
            RemoteInfo(remoteSize(url))
        }
    }

    private fun extractXetRefreshUrl(response: okhttp3.Response, originalUrl: String): String? {
        val refreshHeader = response.header("X-Xet-Refresh-Route")
        val refreshLink = response.header("Link")
            ?.split(',')
            ?.firstOrNull { part ->
                part.contains("rel=\"xet-auth\"", ignoreCase = true) ||
                    part.contains("rel=xet-auth", ignoreCase = true)
            }
            ?.substringBefore(';')
            ?.trim()
            ?.removePrefix("<")
            ?.removeSuffix(">")

        val raw = refreshLink ?: refreshHeader ?: return null
        return when {
            raw.startsWith("https://") || raw.startsWith("http://") -> raw
            raw.startsWith("/") -> {
                val base = Request.Builder().url(originalUrl).build().url
                "${base.scheme}://${base.host}$raw"
            }
            else -> "https://huggingface.co/$raw"
        }
    }

    private fun deriveXetRefreshUrl(url: String): String? {
        val parsed = runCatching { Request.Builder().url(url).build().url }.getOrNull()
            ?: return null
        if (!isHuggingFaceHubHost(parsed.host)) return null

        val segments = parsed.pathSegments
        val resolveIndex = segments.indexOf("resolve")
        if (resolveIndex < 2 || resolveIndex + 1 >= segments.size) return null

        val repoId = segments.take(resolveIndex).joinToString("/")
        val revision = segments[resolveIndex + 1]
        return "${parsed.scheme}://${parsed.host}/api/models/$repoId/xet-read-token/$revision"
    }

    private fun isHuggingFaceHubHost(host: String): Boolean {
        val normalized = host.lowercase(Locale.US)
        return normalized == "huggingface.co" || normalized.endsWith(".huggingface.co")
    }

    private fun remoteSize(url: String): Long {
        val request = Request.Builder().url(url).head().build()
        return runCatching {
            client.newCall(request).execute().use { response ->
                if (response.isSuccessful) {
                    response.header("X-Linked-Size")?.toLongOrNull()
                        ?: response.header("Content-Length")?.toLongOrNull()
                        ?: -1L
                } else {
                    -1L
                }
            }
        }.getOrDefault(-1L)
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
        if (expectedSize > 0 && existing > expectedSize) {
            destFile.delete()
            existing = 0L
        }
        if (expectedSize > 0 && existing == expectedSize) {
            emitProgress(
                modelId, modelName, packageOffset + existing,
                if (packageTotal > 0) packageTotal else expectedSize,
                0L, null, currentFileName, false,
            )
            return@withContext
        }

        val builder = Request.Builder().url(url)
        if (existing > 0L) builder.header("Range", "bytes=$existing-")

        val call = client.newCall(builder.build())
        activeCall = call

        call.execute().use { response ->
            if (existing > 0L && response.code == 416 && expectedSize == existing) return@use
            if (!response.isSuccessful) {
                throw IOException(getString(R.string.error_download_failed, response.code.toString()))
            }

            var append = existing > 0L && response.code == 206
            if (existing > 0L && response.code == 200) {
                existing = 0L
                append = false
            }

            val body = response.body ?: throw IOException("Response body is null")
            val responseBytes = body.contentLength()
            val totalFileBytes = when {
                expectedSize > 0 -> expectedSize
                append && responseBytes > 0 -> existing + responseBytes
                else -> responseBytes
            }

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
                            val reportedTotal = if (packageTotal > 0) packageTotal else totalFileBytes
                            val speed = smoothedBytesPerSecond.toLong().coerceAtLeast(0L)
                            val eta = if (reportedTotal > reportedDone && speed > 0) {
                                (reportedTotal - reportedDone) / speed
                            } else {
                                null
                            }
                            emitProgress(
                                modelId, modelName, reportedDone, reportedTotal,
                                speed, eta, currentFileName, false,
                            )
                        }
                    }
                }
            }

            val finalSize = destFile.length()
            if (totalFileBytes > 0 && finalSize != totalFileBytes) {
                throw IOException("Incomplete HTTP download: $finalSize/$totalFileBytes")
            }

            val total = if (packageTotal > 0) packageTotal else totalFileBytes
            emitProgress(
                modelId, modelName, packageOffset + finalSize, total,
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
        var profile = currentXetProfile()

        while (destFile.length() < expectedSize) {
            val offset = destFile.length()
            var requestedThermalRestart = false
            var lastBytes = offset
            var lastTime = System.currentTimeMillis()
            var smoothedSpeed = 0.0

            val nativeJob = async(Dispatchers.IO) {
                XetNative.nativeDownload(
                    hash = hash,
                    size = expectedSize,
                    refreshUrl = refreshUrl,
                    destPath = destFile.absolutePath,
                    offset = offset,
                    profile = profile,
                )
            }

            while (!nativeJob.isCompleted) {
                delay(500)
                val nowBytes = destFile.length()
                val now = System.currentTimeMillis()
                val elapsed = (now - lastTime).coerceAtLeast(1L)
                val instant = (nowBytes - lastBytes).coerceAtLeast(0L) * 1000.0 / elapsed
                if (instant > 0) {
                    smoothedSpeed = if (smoothedSpeed <= 0) instant else smoothedSpeed * 0.75 + instant * 0.25
                }
                lastBytes = nowBytes
                lastTime = now

                val reportedDone = packageOffset + nowBytes
                val speed = smoothedSpeed.toLong().coerceAtLeast(0L)
                val eta = if (packageTotal > reportedDone && speed > 0) {
                    (packageTotal - reportedDone) / speed
                } else {
                    null
                }
                emitProgress(
                    modelId, modelName, reportedDone, packageTotal,
                    speed, eta, currentFileName, true,
                )

                if (pauseRequested || cancelRequested) {
                    XetNative.nativeCancel()
                    break
                }

                val saferProfile = currentXetProfile()
                if (saferProfile < profile) {
                    profile = saferProfile
                    requestedThermalRestart = true
                    Log.i(TAG, "Thermal pressure: restarting Xet at mobile profile $profile")
                    XetNative.nativeCancel()
                    break
                }
            }

            val result = nativeJob.await()

            if (pauseRequested || cancelRequested) {
                throw CancellationException("download interrupted")
            }
            if (result == 0) break
            if (result == 1 && requestedThermalRestart) continue
            if (result == 1) throw CancellationException("Xet cancelled")

            throw XetDownloadException(XetNative.nativeLastError() ?: "unknown Xet error")
        }

        if (destFile.length() != expectedSize) {
            throw XetDownloadException("Incomplete Xet download: ${destFile.length()}/$expectedSize")
        }

        emitProgress(
            modelId, modelName, packageOffset + expectedSize, packageTotal,
            0L, 0L, currentFileName, true,
        )
    }

    private fun currentXetProfile(): Int {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.Q) return 2
        val powerManager = getSystemService(POWER_SERVICE) as PowerManager
        return when (powerManager.currentThermalStatus) {
            PowerManager.THERMAL_STATUS_SEVERE,
            PowerManager.THERMAL_STATUS_CRITICAL,
            PowerManager.THERMAL_STATUS_EMERGENCY,
            PowerManager.THERMAL_STATUS_SHUTDOWN -> 1

            PowerManager.THERMAL_STATUS_MODERATE -> 2
            else -> 3
        }
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
            etaSeconds = eta,
            currentFileName = currentFileName,
            usingXet = usingXet,
        )

        val mode = if (usingXet) getString(R.string.download_mode_xet) else getString(R.string.download_mode_http)
        val file = currentFileName?.let { " • $it" }.orEmpty()
        val status = listOfNotNull(
            if (speed > 0) buildDownloadStatus(speed, eta) else null,
            "$mode$file",
        ).joinToString(" • ")

        updateNotification(modelName, progress, statusText = status)
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
                createPausedNotification(it.modelName, state.progress),
            )
        }
    }

    private fun resumeDownload() {
        val request = activeRequest ?: return
        if (_downloadState.value !is DownloadState.Paused) return

        pauseRequested = false
        cancelRequested = false
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
        return builder.build()
    }

    private fun createPausedNotification(modelName: String, progress: Float): android.app.Notification {
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

        return NotificationCompat.Builder(this, NOTIFICATION_CHANNEL_ID)
            .setContentTitle(getString(R.string.download_paused))
            .setContentText(modelName)
            .setSmallIcon(android.R.drawable.stat_sys_download)
            .setProgress(100, (progress * 100).toInt(), false)
            .setOngoing(true)
            .addAction(android.R.drawable.ic_media_play, getString(R.string.resume), resumePendingIntent)
            .addAction(
                android.R.drawable.ic_menu_close_clear_cancel,
                getString(R.string.cancel),
                cancelPendingIntent,
            )
            .build()
    }

    private fun updateNotification(
        modelName: String,
        progress: Float,
        success: Boolean = false,
        error: String? = null,
        isExtracting: Boolean = false,
        statusText: String? = null,
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

            else -> createNotification(modelName, progress, isExtracting, statusText)
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
