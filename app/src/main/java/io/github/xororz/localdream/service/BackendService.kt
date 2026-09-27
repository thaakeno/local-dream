package io.github.xororz.localdream.service

import io.github.xororz.localdream.utils.BackendDiagnostics
import io.github.xororz.localdream.utils.CrashDiagnostics

import android.app.*
import android.content.Intent
import android.os.IBinder
import android.util.Log
import androidx.core.app.NotificationCompat
import io.github.xororz.localdream.BuildConfig
import io.github.xororz.localdream.R
import io.github.xororz.localdream.data.DitEngine
import io.github.xororz.localdream.data.DitResolution
import io.github.xororz.localdream.data.Model
import io.github.xororz.localdream.data.QwenFamilyStorage
import java.io.File
import java.io.IOException
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch

class BackendService : Service() {
    @Volatile
    private var process: Process? = null

    // Set true around an intentional teardown (and reset just before a new
    // start) so the monitor thread doesn't surface the resulting process exit
    // as a backend Error. backendState is a process-wide StateFlow shared
    // across Service instances, so a stale Error would otherwise be read by
    // the next model's health check and shown as "backend start failed".
    @Volatile
    private var stopping = false

    // Desired-state reconciliation. `desired` is the config the screen wants
    // running (null = nothing); `serving` is what the live process was actually
    // started for. Both are touched only on the single backend thread via
    // reconcile(), so no extra locking is needed. idleStopJob is the pending
    // grace-period teardown scheduled by a stop request.
    private var desired: BackendConfig? = null
    private var serving: BackendConfig? = null
    private var idleStopJob: Job? = null
    private lateinit var runtimeDir: File
    private lateinit var musicRuntimeDir: File

    @Volatile
    private var runtimeDirReady = false

    // All backend process management (asset copies, exec, destroy/waitFor)
    // runs on this single thread: jobs stay ordered relative to each other
    // and the main thread never blocks on waitFor() or large file copies.
    private val backendDispatcher =
        Executors.newSingleThreadExecutor { r -> Thread(r, "backend-control") }
            .asCoroutineDispatcher()
    private val serviceScope = CoroutineScope(SupervisorJob() + backendDispatcher)

    companion object {
        private const val TAG = "BackendService"
        private const val EXECUTABLE_NAME = "libstable_diffusion_core.so"
        private const val MUSIC_EXECUTABLE_NAME = "libyue2_server.so"
        const val RUNTIME_DIR = "runtime_libs"
        private const val MUSIC_RUNTIME_DIR = "runtime_yue2_htp"
        private const val MUSIC_RUNTIME_VERSION = "jz_fastrpc_883df324_v049_clean"
        private const val RUNTIME_VERSION = "qnn_2_50_0_260828"
        private const val RUNTIME_VERSION_FILE = ".runtime_version"
        private const val NOTIFICATION_ID = 2
        private const val CHANNEL_ID = "backend_service_channel"

        // Grace window before a stop request actually tears the backend down.
        // A re-entry within this window (same or different model) cancels the
        // teardown and reconciles in-place, so quick back-then-reopen reuses
        // the live process and model switches stay on one Service instance
        // (single-threaded, no cross-instance start/stop race). Affects only
        // reuse/latency, never correctness: a slower re-entry just starts fresh.
        private const val IDLE_GRACE_MS = 1500L
        private const val MAX_BACKEND_ERROR_CHARS = 700

        // Local Dream caps a mobile score plan to <=1024 tokens and audio to
        // <=20 s (500 semantic frames). 2560 keeps a 20 s planned song in one
        // NAR chunk while cutting KV residency drastically versus 6144. That
        // leaves the Q8 weights, KV and graph scratch inside SM8850's practical
        // FastRPC address-space budget instead of running into ENORPCMEMORY.
        private const val MUSIC_MAX_SEQ = 2560

        const val ACTION_STOP = "io.github.xororz.localdream.STOP_GENERATION"
        const val ACTION_RESTART = "io.github.xororz.localdream.RESTART_BACKEND"

        // Brings the service up as a foreground service without touching any
        // backend process. Sent when host mode starts (while the app is still
        // in the foreground) so later remote /select commands are plain
        // startService() deliveries to an already-foreground service instead
        // of new-FGS starts, which Android 12+ blocks from the background.
        const val ACTION_STANDBY = "io.github.xororz.localdream.STANDBY_BACKEND"

        // Pseudo backend type: native process in --upscaler_mode (no model
        // dir). Used by host mode so a controller's standalone upscale page
        // can run on this device's NPU.
        const val BACKEND_TYPE_UPSCALER = "upscaler"

        // --type values served by the downloadable DiT engine.
        fun isDitBackend(backendType: String): Boolean = backendType == "zimage" ||
            backendType == "klein" || backendType == "qwen21"

        fun isMusicBackend(backendType: String): Boolean = backendType == "yue2"

        // One reused dir, stamped with the SDK it holds. Per-file copying only
        // refreshes libs whose size changed, so an SDK bump would otherwise
        // leave same-size stale libs and libs the new SDK dropped behind;
        // wiping on a stamp mismatch keeps the dir exactly one version's worth.
        fun prepareRuntimeDirRoot(filesDir: File): File {
            val runtimeDir = File(filesDir, RUNTIME_DIR)
            val stamp = File(runtimeDir, RUNTIME_VERSION_FILE)
            if (runtimeDir.exists() &&
                runCatching { stamp.readText() }.getOrNull() != RUNTIME_VERSION
            ) {
                Log.i(TAG, "Runtime dir holds another SDK version, wiping")
                runtimeDir.deleteRecursively()
            }
            if (!runtimeDir.exists()) runtimeDir.mkdirs()
            runCatching { stamp.writeText(RUNTIME_VERSION) }
                .onFailure { Log.w(TAG, "Write runtime version stamp failed", it) }
            return runtimeDir
        }

        private object StateHolder {
            val _backendState = MutableStateFlow<BackendState>(BackendState.Idle)

            // Live milestone stream for native startup. MusicRunScreen renders
            // this directly instead of showing an opaque disabled button while
            // FastRPC / Hexagon is being initialized.
            val _startupStatus = MutableStateFlow<BackendStartupStatus?>(null)

            // modelId the live process is serving (null when none). Process-wide
            // so a screen can tell whether 8081 is already serving *its* model
            // vs. a previous model still alive in the stop grace window.
            val _servingModelId = MutableStateFlow<String?>(null)

            // Resolution the live process was started for. Reported through
            // host mode's /status so a controller only declares Ready once the
            // backend matches its full config, not just the model id.
            val _servingResolution = MutableStateFlow<Pair<Int, Int>?>(null)
        }

        val backendState: StateFlow<BackendState> = StateHolder._backendState

        val startupStatus: StateFlow<BackendStartupStatus?> = StateHolder._startupStatus

        val servingModelId: StateFlow<String?> = StateHolder._servingModelId

        val servingResolution: StateFlow<Pair<Int, Int>?> = StateHolder._servingResolution

        private fun extractBackendError(line: String): String? {
            val trimmed = line.trim()
            val lower = trimmed.lowercase()
            val errorMarker = "[ ERROR ]"
            val message = when {
                // Qwen's text encoder reports this capability line during
                // startup even for a pure text-to-image request (Img2Img=0,
                // no references). It is not a backend failure. A real
                // reference-image failure will still be returned by /generate.
                "no vision weights detected, vision disabled" in lower -> return null

                // FastRPC/DSPQueue transport failures abort the process and are
                // the useful root cause; surface these ahead of the resulting
                // HTTP EOF seen by the Android client.
                "dspqueue_read failed" in lower ||
                    "dspqueue_write failed" in lower ||
                    "dspqueue" in lower && "failed" in lower ->
                    trimmed

                errorMarker in trimmed -> {
                    val payload = trimmed.substringAfter(errorMarker).trim()
                    val detailMarker = " - "
                    if (detailMarker in payload) {
                        payload.substringAfter(detailMarker).trim()
                    } else {
                        payload
                    }
                }

                trimmed.startsWith("ERROR:", ignoreCase = true) ->
                    trimmed.substringAfter(':').trim()

                "dsp_register_rpcmem failed" in lower ||
                    "failed to init rpc mempool" in lower ||
                    "ggml_htp_execute_batch failed" in lower ||
                    "weight_inval reset failed" in lower ||
                    "aee_enorpcmemory" in lower ||
                    "[load] fatal:" in lower ||
                    "[pipeline] fatal:" in lower ||
                    "failed to load libcdsprpc.so" in lower ||
                    "failed to open session" in lower ||
                    "failed to create device/session" in lower ||
                    "failed to allocate weight buffer" in lower ||
                    "ggml_assert" in lower ||
                    "aee_eunabletoload" in lower ->
                    trimmed

                else -> return null
            }
            if (message.isBlank()) return null
            return if (message.length <= MAX_BACKEND_ERROR_CHARS) {
                message
            } else {
                message.take(MAX_BACKEND_ERROR_CHARS - 1) + "…"
            }
        }

        private fun updateState(state: BackendState) {
            StateHolder._backendState.value = state
        }

        private fun updateStartup(status: BackendStartupStatus?) {
            StateHolder._startupStatus.value = status
        }

        private fun updateServing(config: BackendConfig?) {
            StateHolder._servingModelId.value = config?.modelId
            StateHolder._servingResolution.value = config?.let { Pair(it.width, it.height) }
        }
    }

    data class BackendStartupStatus(
        val modelId: String,
        val phase: String,
        val detail: String,
        val progress: Float,
        val startedAtMillis: Long,
    )

    sealed class BackendState {
        object Idle : BackendState()
        object Starting : BackendState()
        object Running : BackendState()

        // modelId is the model this failure pertains to, or null for a failure
        // that affects any model (e.g. runtime preparation). Lets a screen
        // ignore an error left over from a *different* model's process (a crash
        // in the stop grace window) instead of mistaking it for its own.
        data class Error(val message: String, val modelId: String? = null) : BackendState()
    }

    // What a backend process is (or should be) running for. Equality drives
    // reconcile()'s "already serving this exact config" decision. listenOnAll
    // is part of the config on purpose: entering/exiting host mode changes the
    // required bind address, and reusing a live process across that boundary
    // would either leave the port unreachable for the controller or leave it
    // exposed after host mode ends.
    private data class BackendConfig(
        val modelId: String,
        val backendType: String,
        val width: Int,
        val height: Int,
        val listenOnAll: Boolean,
        val htpMode: String,
    )

    override fun onCreate() {
        super.onCreate()
        createNotificationChannel()
        serviceScope.launch { prepareRuntimeDir() }
    }

    override fun onBind(intent: Intent): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        Log.i(TAG, "service command: ${intent?.action}")
        try {
            startForeground(
                NOTIFICATION_ID,
                createNotification(this.getString(R.string.backend_notify)),
            )
        } catch (e: Exception) {
            // Android 12+ can reject foreground promotion when the command
            // arrived with the app in the background (e.g. the service was
            // reclaimed and re-created by a remote host-mode command).
            // Continue as a background service rather than crash: the backend
            // process still works, the system may just reclaim it sooner.
            Log.w(TAG, "startForeground rejected: ${e.message}")
        }

        // Commands only declare intent; the single backend thread converges the
        // actual process to it via reconcile(). This keeps every start/stop
        // ordered and race-free regardless of how fast the screen comes and goes.
        when (intent?.action) {
            ACTION_STOP -> serviceScope.launch { requestStop(startId) }

            // Foreground promotion only; no backend change.
            ACTION_STANDBY -> {}

            else -> {
                val forceRestart = intent?.action == ACTION_RESTART
                val config = parseConfig(intent)
                serviceScope.launch { requestStart(config, forceRestart) }
            }
        }

        return START_NOT_STICKY
    }

    private fun parseConfig(intent: Intent?): BackendConfig? {
        val modelId = intent?.getStringExtra("modelId") ?: return null
        // Backend type is decided by the caller (it already has the Model);
        // re-deriving it here would require a full model-directory scan.
        val backendType = intent.getStringExtra("backendType") ?: return null
        val requestedWidth = intent.getIntExtra("width", 512)
        val requestedHeight = intent.getIntExtra("height", 512)
        val width = if (isDitBackend(backendType)) DitResolution.snap(requestedWidth) else requestedWidth
        val height = if (isDitBackend(backendType)) DitResolution.snap(requestedHeight) else requestedHeight
        if (width != requestedWidth || height != requestedHeight) {
            Log.w(
                TAG,
                "unsupported DiT resolution ${requestedWidth}x$requestedHeight; using ${width}x$height",
            )
        }
        // Host mode is read from RemoteHostService's in-process state, not a
        // persisted flag: a crash can never leave a stale "expose the port"
        // bit behind, and a config-equality check below forces a restart when
        // the bind address requirement changes.
        val listenOnAll = getSharedPreferences("app_prefs", MODE_PRIVATE)
            .getBoolean("listen_on_all_addresses", false) ||
            RemoteHostService.isRunning.value
        val requestedHtpMode = intent.getStringExtra("htp_mode")?.lowercase() ?: "auto"
        val htpMode = requestedHtpMode.takeIf { it == "auto" || it == "single" || it == "dual" }
            ?: "auto"
        return BackendConfig(modelId, backendType, width, height, listenOnAll, htpMode)
    }

    // Declares the desired backend and converges to it. Cancels any pending
    // idle teardown first so a quick re-entry keeps the live process.
    private fun requestStart(config: BackendConfig?, forceRestart: Boolean) {
        if (config == null) {
            updateState(BackendState.Error("Model not found"))
            return
        }
        idleStopJob?.cancel()
        idleStopJob = null
        desired = config
        reconcile(forceRestart)
    }

    // Declares that nothing should run, but only tears down after a grace
    // window. A re-entry within the window cancels this job and reconciles in
    // place; otherwise the process is stopped and the Service stops itself.
    private fun requestStop(startId: Int) {
        desired = null
        idleStopJob?.cancel()
        idleStopJob = serviceScope.launch {
            delay(IDLE_GRACE_MS)
            // A re-entry during the delay normally cancels us. The startId guard
            // closes the remaining edge where a new command's onStartCommand
            // raced in just as the grace fired: stopSelfResult() refuses to stop
            // when a newer start exists, leaving the (re)started service alive
            // with its foreground notification intact.
            if (desired == null) {
                stopBackend()
                // While host mode is active the service must survive with its
                // foreground status: remote /select commands arrive over the
                // network with no visible activity, and Android 12+ blocks
                // promoting a freshly started service to the foreground from
                // that state. Keeping this one alive makes those commands
                // plain startService() deliveries to a live FGS.
                if (!RemoteHostService.isRunning.value && stopSelfResult(startId)) {
                    stopForeground(STOP_FOREGROUND_REMOVE)
                }
            }
        }
    }

    // Converges the actual process to `desired`. Runs only on the single
    // backend thread, so reading current state and any start/stop are atomic
    // with respect to other commands.
    private fun reconcile(forceRestart: Boolean) {
        val want = desired ?: return
        // prepareRuntimeDir runs earlier on this same thread and publishes its
        // own error on failure; if it didn't finish ready, leave that state.
        if (!runtimeDirReady) {
            return
        }
        val alreadyServing = process?.isAlive == true && serving == want
        if (alreadyServing && !forceRestart) {
            Log.i(TAG, "backend already serving ${want.modelId} ${want.width}x${want.height}")
            updateServing(want)
            if (!isMusicBackend(want.backendType) || backendState.value is BackendState.Running) {
                updateState(BackendState.Running)
            }
            return
        }
        stopBackend()
        if (startBackend(want)) {
            serving = want
            updateServing(want)
            // yue-server is only ready after its "[Server] Listening" line.
            // Other backends keep their existing spawn == running behavior.
            if (!isMusicBackend(want.backendType)) {
                updateState(BackendState.Running)
            }
        } else {
            serving = null
            updateServing(null)
            // startBackend() publishes a specific Error when it knows the
            // cause. Preserve that instead of replacing it with a generic
            // startup failure.
            if (backendState.value !is BackendState.Error) {
                updateState(BackendState.Error("Backend start failed", want.modelId))
            }
        }
    }

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
        updateState(BackendState.Error("Service timeout", servingModelId.value))
        stopForeground(STOP_FOREGROUND_REMOVE)
        stopSelf()
        serviceScope.launch {
            desired = null
            idleStopJob?.cancel()
            try {
                stopBackend()
            } catch (e: Exception) {
                Log.e(TAG, "stopBackend on timeout failed", e)
            }
        }
    }

    private fun createNotificationChannel() {
        val name = "Backend Service"
        val descriptionText = "Backend service for image generation"
        val importance = NotificationManager.IMPORTANCE_LOW
        val channel = NotificationChannel(CHANNEL_ID, name, importance).apply {
            description = descriptionText
        }
        val notificationManager = getSystemService(NOTIFICATION_SERVICE) as NotificationManager
        notificationManager.createNotificationChannel(channel)
    }

    private fun createNotification(contentText: String): Notification {
        val openAppIntent = packageManager.getLaunchIntentForPackage(packageName)?.apply {
            flags = Intent.FLAG_ACTIVITY_SINGLE_TOP or Intent.FLAG_ACTIVITY_NEW_TASK
        }
        val pendingIntent = PendingIntent.getActivity(
            this,
            0,
            openAppIntent,
            PendingIntent.FLAG_IMMUTABLE,
        )

        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle(this.getString(R.string.backend_notify_title))
            .setContentText(contentText)
            .setSmallIcon(R.drawable.ic_launcher_monochrome)
            .setContentIntent(pendingIntent)
            .setOngoing(true)
            .build()
    }

    private fun stageDeviceFastRpcCompatibilityLibraries(destination: File) {
        val names = listOf(
            "libcdsprpc.so",
            "libvmmem.so",
            "vendor.qti.hardware.dsp-V1-ndk.so",
            "vendor.qti.hardware.dsp@1.0.so",
        )
        val roots = listOf(
            File("/vendor/lib64"),
            File("/system/vendor/lib64"),
        )

        names.forEach { name ->
            val target = File(destination, name)

            // Never keep an OEM transport copied from an older firmware build.
            // If the current firmware cannot be read we prefer the namespace
            // fallback in ggml-hexagon rather than shadowing it with stale bytes.
            if (target.exists()) {
                runCatching { target.delete() }
            }

            val source = roots.asSequence()
                .map { File(it, name) }
                .firstOrNull { it.isFile }

            if (source == null) {
                BackendDiagnostics.append(
                    this,
                    "FASTRPC",
                    "$name not present under vendor lib64",
                )
                return@forEach
            }

            try {
                source.inputStream().use { input ->
                    target.outputStream().use { output -> input.copyTo(output) }
                }
                target.setReadable(true, true)
                target.setExecutable(true, true)
                BackendDiagnostics.append(
                    this,
                    "FASTRPC",
                    "staged device $name (${target.length()}B) from ${source.absolutePath}",
                )
            } catch (e: Exception) {
                runCatching { target.delete() }
                BackendDiagnostics.append(
                    this,
                    "FASTRPC",
                    "device copy unavailable for $name: ${e.javaClass.simpleName}: ${e.message}",
                )
            }
        }
    }

    private fun prepareRuntimeDir() {
        try {
            runtimeDir = prepareRuntimeDirRoot(filesDir)

            try {
                val qnnlibsAssets = assets.list("qnnlibs")
                qnnlibsAssets?.forEach { fileName ->
                    val targetLib = File(runtimeDir, fileName)

                    val needsCopy = !targetLib.exists() ||
                        run {
                            val assetInputStream = assets.open("qnnlibs/$fileName")
                            val assetSize = assetInputStream.use { it.available().toLong() }
                            targetLib.length() != assetSize
                        }

                    if (needsCopy) {
                        val assetInputStream = assets.open("qnnlibs/$fileName")
                        assetInputStream.use { input ->
                            targetLib.outputStream().use { output ->
                                input.copyTo(output)
                            }
                        }
                        Log.d(TAG, "Copied $fileName from assets to runtime directory")
                    }

                    targetLib.setReadable(true, true)
                    targetLib.setExecutable(true, true)
                }
                Log.i(TAG, "QNN libraries prepared in runtime directory")

                // HyperOS 3 / Android 16 on some SM8850 devices does not expose
                // libcdsprpc.so to the ordinary app namespace even with
                // <uses-native-library>. As a second line of defense, stage the
                // device's *own* FastRPC userspace chain into the app-private
                // runtime directory when SELinux permits reading it. This is
                // deliberately device-local: we never ship a mismatched OEM
                // FastRPC binary in the APK.
                stageDeviceFastRpcCompatibilityLibraries(runtimeDir)

                // The DiT engine's Hexagon skels share this directory: it is
                // already on the DSP search path, and they are only useful on
                // the devices whose HTP version the engine covers.
                if (DitEngine.isSupportedDevice()) {
                    // Skel rebuilds often keep the exact (page-aligned) size,
                    // so also refresh any copy older than the installed APK.
                    val apkUpdateTime =
                        packageManager.getPackageInfo(packageName, 0).lastUpdateTime
                    assets.list("ditlibs")?.forEach { fileName ->
                        val target = File(runtimeDir, fileName)
                        val assetSize =
                            assets.open("ditlibs/$fileName").use { it.available().toLong() }
                        if (!target.exists() || target.length() != assetSize ||
                            target.lastModified() < apkUpdateTime
                        ) {
                            assets.open("ditlibs/$fileName").use { input ->
                                target.outputStream().use { output -> input.copyTo(output) }
                            }
                            Log.d(TAG, "Copied $fileName from assets to runtime directory")
                        }
                        target.setReadable(true, true)
                        target.setExecutable(true, true)
                    }
                }

                // YuE2 uses a dedicated matched AP/DSP Hexagon runtime. Never
                // mix its FastRPC/mempool skel with image-generation skels.
                musicRuntimeDir = File(filesDir, MUSIC_RUNTIME_DIR)
                val musicStamp = File(musicRuntimeDir, ".runtime_version")
                if (
                    musicRuntimeDir.exists() &&
                    runCatching { musicStamp.readText() }.getOrNull() != MUSIC_RUNTIME_VERSION
                ) {
                    musicRuntimeDir.deleteRecursively()
                }
                if (!musicRuntimeDir.exists()) musicRuntimeDir.mkdirs()

                val musicApkUpdateTime =
                    packageManager.getPackageInfo(packageName, 0).lastUpdateTime
                assets.list("yue2libs")?.forEach { fileName ->
                    val target = File(musicRuntimeDir, fileName)
                    val assetSize =
                        assets.open("yue2libs/$fileName").use { it.available().toLong() }
                    if (
                        !target.exists() ||
                        target.length() != assetSize ||
                        target.lastModified() < musicApkUpdateTime
                    ) {
                        assets.open("yue2libs/$fileName").use { input ->
                            target.outputStream().use { output -> input.copyTo(output) }
                        }
                    }
                    target.setReadable(true, true)
                    target.setExecutable(true, true)
                }
                musicStamp.writeText(MUSIC_RUNTIME_VERSION)
                stageDeviceFastRpcCompatibilityLibraries(musicRuntimeDir)

                BackendDiagnostics.append(
                    this,
                    "YUE2_RUNTIME",
                    "dir=${musicRuntimeDir.absolutePath} " +
                        "files=${musicRuntimeDir.list()?.sorted()?.joinToString()}",
                )
            } catch (e: IOException) {
                Log.e(TAG, "Failed to prepare QNN/YuE2 runtime assets", e)
                throw RuntimeException("Failed to prepare QNN libraries from assets", e)
            }

            if (BuildConfig.FLAVOR == "filter") {
                try {
                    val safetyCheckerTarget = File(filesDir, "safety_checker.mnn")
                    val assetSize = assets.open("safety_checker.mnn")
                        .use { it.available().toLong() }

                    if (!safetyCheckerTarget.exists() ||
                        safetyCheckerTarget.length() != assetSize
                    ) {
                        assets.open("safety_checker.mnn").use { input ->
                            safetyCheckerTarget.outputStream().use { output ->
                                input.copyTo(output)
                            }
                        }
                        Log.i(
                            TAG,
                            "Safety checker model copied to: ${safetyCheckerTarget.absolutePath}",
                        )
                    }

                    safetyCheckerTarget.setReadable(true, true)
                } catch (e: IOException) {
                    Log.e(TAG, "copy safety_checker.mnn failed", e)
                    throw RuntimeException("Failed to copy safety checker model", e)
                }
            }

            runtimeDir.setReadable(true, true)
            runtimeDir.setExecutable(true, true)
            runtimeDirReady = true

            Log.i(TAG, "Runtime directory prepared: ${runtimeDir.absolutePath}")
            Log.i(TAG, "Runtime files: ${runtimeDir.list()?.joinToString()}")
        } catch (e: Exception) {
            Log.e(TAG, "Prepare runtime dir failed", e)
            updateState(BackendState.Error("Prepare runtime dir failed: ${e.message}"))
        }
    }

    private data class HexagonQueueProfile(
        val label: String,
        val transformerBytes: Long,
        val opBatch: Int,
        val opQueue: Int,
    )

    /**
     * Pick a conservative Hexagon DSP queue shape from the transformer
     * footprint rather than from a model id / SoC allow-list.
     *
     * The current ggml-hexagon defaults (1280 ops, queue depth 32) are very
     * aggressive for multi-gigabyte Qwen graphs and can end in FastRPC
     * dspqueue error 0x2e under sustained denoising. Smaller in-flight queues
     * trade a little host-side overlap for dramatically lower DSP queue
     * pressure without changing a single model weight, scheduler step, or
     * sampling parameter.
     */
    private fun qwenHexagonQueueProfile(modelsDir: File): HexagonQueueProfile {
        val transformer = sequenceOf(
            File(modelsDir, "dit.gguf"),
            File(modelsDir, "dit.safetensors"),
        ).firstOrNull { it.isFile }

        val bytes = transformer?.length() ?: 0L
        val gib = 1024L * 1024L * 1024L
        return when {
            bytes >= 6L * gib -> HexagonQueueProfile(
                label = "large-safe",
                transformerBytes = bytes,
                opBatch = 512,
                opQueue = 8,
            )
            bytes >= 4L * gib -> HexagonQueueProfile(
                label = "balanced",
                transformerBytes = bytes,
                opBatch = 768,
                opQueue = 12,
            )
            else -> HexagonQueueProfile(
                label = "standard",
                transformerBytes = bytes,
                opBatch = 1024,
                opQueue = 16,
            )
        }
    }

    private fun startBackend(config: BackendConfig): Boolean {
        val modelId = config.modelId
        val backendType = config.backendType
        val width = config.width
        val height = config.height
        Log.i(
            TAG,
            "backend start, model: $modelId, resolution: $width×$height, htpMode=${config.htpMode}",
        )

        // reconcile() has already stopped any previous process; just re-arm
        // crash reporting for the process we are about to start.
        stopping = false
        updateState(BackendState.Starting)
        if (isMusicBackend(backendType)) {
            updateStartup(
                BackendStartupStatus(
                    modelId = modelId,
                    phase = "launch",
                    detail = "Launching native YuE2 process",
                    progress = 0.04f,
                    startedAtMillis = System.currentTimeMillis(),
                ),
            )
        } else {
            updateStartup(null)
        }

        try {
            val nativeDir = applicationInfo.nativeLibraryDir
            val modelsDir = if (backendType == "qwen21") {
                QwenFamilyStorage.prepareRuntimeDir(this, modelId)
                    ?: File(Model.getModelsDir(this), modelId)
            } else {
                File(Model.getModelsDir(this), modelId)
            }

            BackendDiagnostics.beginSession(
                this,
                "backendType=$backendType model=$modelId",
            )
            BackendDiagnostics.append(
                this,
                "START",
                "modelsDir=${modelsDir.absolutePath}",
            )

            if (isMusicBackend(backendType)) {
                if (modelId !in setOf(
                        "yue2_3b_q5km",
                        "yue2_3b_q6k",
                        "yue2_3b_q8",
                        "yue2_3b_bf16",
                    )
                ) {
                    val message =
                        "YuE2 NPU runtime only accepts HTP-supported GGUF precisions. " +
                            "Refusing CPU/compatibility fallback for $modelId."
                    BackendDiagnostics.append(this, "ERROR", message)
                    updateState(BackendState.Error(message, config.modelId))
                    return false
                }

                val backbone = File(modelsDir, "backbone.gguf")
                val vae = File(modelsDir, "vae.gguf")
                val invalid = when {
                    !backbone.isFile -> "YuE2 backbone.gguf is missing"
                    backbone.length() < 2_400_000_000L ->
                        "YuE2 backbone looks incomplete (${backbone.length()} bytes)"
                    !vae.isFile -> "YuE2 vae.gguf is missing"
                    vae.length() < 450_000_000L ->
                        "YuE2 VAE looks incomplete (${vae.length()} bytes)"
                    else -> null
                }
                BackendDiagnostics.append(
                    this,
                    "MODEL",
                    "backbone=${backbone.length()}B vae=${vae.length()}B",
                )
                if (invalid != null) {
                    BackendDiagnostics.append(this, "ERROR", invalid)
                    updateState(BackendState.Error(invalid, config.modelId))
                    return false
                }
            }

            val executableFile = File(
                nativeDir,
                if (isMusicBackend(backendType)) MUSIC_EXECUTABLE_NAME else EXECUTABLE_NAME,
            )

            if (!executableFile.exists()) {
                Log.e(TAG, "error: executable does not exist: ${executableFile.absolutePath}")
                updateState(BackendState.Error("Backend executable is missing", config.modelId))
                return false
            }

            val preferences = this.getSharedPreferences("app_prefs", MODE_PRIVATE)
            val useImg2img = preferences.getBoolean("use_img2img", true)
            // Part of the config (captured at command time in parseConfig) so
            // reconcile() restarts the process when the bind address
            // requirement changes instead of reusing a mismatched one.
            val listenOnAll = config.listenOnAll

            val command = when {
                backendType == BACKEND_TYPE_UPSCALER -> {
                    // Same invocation as the standalone upscale screen's private
                    // process; run through this service so host mode gets the
                    // usual reconcile/stop-grace lifecycle.
                    mutableListOf(
                        executableFile.absolutePath,
                        "--upscaler_mode",
                        "--lib_dir",
                        runtimeDir.absolutePath,
                        "--port",
                        "8081",
                    )
                }

                isMusicBackend(backendType) -> {
                    // YuE2 is deliberately stage-swapped on mobile. Upstream's
                    // --keep-loaded mode is meant for devices with a generous
                    // accelerator memory budget; it keeps AR + NAR + VAE
                    // resident together and is a bad fit for a phone.
                    //
                    // The ABC planner's native budget is 4096 tokens. The KV
                    // cache must include the prompt prefix and EOS too, so a
                    // 4096 context is structurally impossible. Use yue2.cpp's
                    // documented reduced-memory 8192-token profile; our <=20 s
                    // semantic stage is then clamped to <=500 frames by
                    // upstream itself, while the full 4096 ABC budget remains
                    // available for quality.
                    //
                    // FLASH_ATTN_EXT currently has a documented correctness bug
                    // on Snapdragon Hexagon v75, so use the plain attention path
                    // and still offload supported matmuls to HTP.
                    mutableListOf(
                        executableFile.absolutePath,
                        "--model",
                        File(modelsDir, "backbone.gguf").absolutePath,
                        "--vae",
                        File(modelsDir, "vae.gguf").absolutePath,
                        "--host",
                        if (listenOnAll) "0.0.0.0" else "127.0.0.1",
                        "--port",
                        "8081",
                        "--max-batch",
                        "1",
                        "--max-seq",
                        MUSIC_MAX_SEQ.toString(),
                        "--vae-core",
                        "512",
                        "--vae-halo",
                        "16",
                    )
                }

                else -> mutableListOf(
                    executableFile.absolutePath,
                    "--type",
                    backendType,
                    "--model_dir",
                    modelsDir.absolutePath,
                    "--port",
                    "8081",
                )
            }
            // DiT types load libdit_engine.so and its FastRPC skel from the
            // native library directory they ship in, not the QNN runtime dir.
            val ditEngineDir = if (isDitBackend(backendType)) DitEngine.dir(this) else null
            if (ditEngineDir != null) {
                if (!DitEngine.isInstalled(this)) {
                    Log.e(TAG, "DiT engine missing at $ditEngineDir")
                    updateState(
                        BackendState.Error(
                            getString(R.string.dit_engine_missing),
                            config.modelId,
                        ),
                    )
                    return false
                }
                command += listOf("--lib_dir", ditEngineDir.absolutePath)
                // The DiT engine lives in nativeLibraryDir, while the shared
                // /upscale endpoint needs the extracted QNN runtime. Keep the
                // two paths explicit so generation and upscaling can coexist
                // in the same backend process.
                command += listOf("--qnn_lib_dir", runtimeDir.absolutePath)
            } else if (backendType != "sd15cpu" && backendType != "sdxlmnn" &&
                backendType != BACKEND_TYPE_UPSCALER && !isMusicBackend(backendType)
            ) {
                command += listOf("--lib_dir", runtimeDir.absolutePath)
            }
            if (!useImg2img && backendType != BACKEND_TYPE_UPSCALER &&
                !isMusicBackend(backendType)
            ) {
                command += "--no_img2img"
            }
            if (backendType == "sd15npu" && (width != 512 || height != 512)) {
                val patchFile = if (width == height) {
                    val squarePatch = File(modelsDir, "$width.patch")
                    if (squarePatch.exists()) {
                        squarePatch
                    } else {
                        File(modelsDir, "${width}x$height.patch")
                    }
                } else {
                    File(modelsDir, "${width}x$height.patch")
                }

                if (patchFile.exists()) {
                    command += listOf("--patch", patchFile.absolutePath)
                    Log.i(TAG, "Using patch file: ${patchFile.name}")
                } else {
                    Log.w(
                        TAG,
                        "Patch file not found: ${patchFile.absolutePath}, falling back to 512×512",
                    )
                }
            }
            if (File(modelsDir, "V_PRED").exists()) {
                command += "--use_v_pred"
            }
            // The upscaler-mode process takes no safety-checker flag (same as
            // the standalone upscale screen's own invocation).
            if (BuildConfig.FLAVOR == "filter" && backendType != BACKEND_TYPE_UPSCALER &&
                !isMusicBackend(backendType)
            ) {
                command += listOf(
                    "--safety_checker",
                    File(filesDir, "safety_checker.mnn").absolutePath,
                )
            }
            // SDXL and Anima are the large NPU formats that benefit from
            // per-stage load/release. They share the same backend --lowram flag
            // but keep separate UI toggles so each can opt in independently.
            if ((backendType == "sdxl" || backendType == "sdxlmnn") && preferences.getBoolean("sdxl_lowram", false)) {
                command += "--lowram"
            }
            if (backendType == "anima" && preferences.getBoolean("anima_lowram", true)) {
                command += "--lowram"
                // Aggressive variant: never hold both DiT halves resident at
                // once, so 12GB devices can run Anima low-RAM. Slower per step.
                if (preferences.getBoolean("anima_seq_dit", false)) {
                    command += "--anima_seq_dit"
                }
            }
            if (listenOnAll && !isMusicBackend(backendType)) {
                command += "--listen_all"
            }
            val env = mutableMapOf<String, String>()

            val systemLibPaths = mutableListOf(
                runtimeDir.absolutePath,
                "/system/lib64",
                "/vendor/lib64",
                "/vendor/lib64/egl",
            )
            try {
                val maliSymlink = File("/system/vendor/lib64/egl/libGLES_mali.so")
                if (maliSymlink.exists()) {
                    val realPath = maliSymlink.canonicalPath
                    val soc = realPath.split("/").getOrNull(realPath.split("/").size - 2)

                    if (soc != null) {
                        val socPaths = listOf(
                            "/vendor/lib64/$soc",
                            "/vendor/lib64/egl/$soc",
                        )

                        socPaths.forEach { path ->
                            if (!systemLibPaths.contains(path)) {
                                systemLibPaths.add(path)
                                Log.d("LibPath", "Added SoC path: $path")
                            }
                        }
                    }
                }
            } catch (e: Exception) {
                Log.w("LibPath", "Failed to resolve Mali paths: ${e.message}")
            }
            val systemLibPathsStr = systemLibPaths.joinToString(":")
            env["LD_LIBRARY_PATH"] = systemLibPathsStr
            env["DSP_LIBRARY_PATH"] = runtimeDir.absolutePath

            if (isMusicBackend(backendType)) {
                // Session selection and backend selection are separate: the
                // FastRPC session is HTP0:0 while the GGML device name is HTP0.
                // Force that exact accelerator so YuE2 never silently selects
                // another backend at module load time.
                val musicHtpSessionSpec = "HTP0:0"

                env["GGML_HEXAGON_DEVICES"] = musicHtpSessionSpec
                env["GGML_BACKEND"] = "HTP0"
                env["LOCAL_DREAM_YUE2_BACKEND"] = "fastrpc-mempool-0.4.9"

                env["LD_LIBRARY_PATH"] = listOf(
                    nativeDir,
                    musicRuntimeDir.absolutePath,
                    "/system/lib64",
                ).joinToString(":")

                val dspPath = listOf(
                    musicRuntimeDir.absolutePath,
                    "/vendor/lib/rfsa/adsp",
                    "/vendor/dsp/cdsp",
                    "/dsp",
                ).joinToString(";")
                env["ADSP_LIBRARY_PATH"] = dspPath
                env["DSP_LIBRARY_PATH"] = dspPath

                val localFastRpc = File(musicRuntimeDir, "libcdsprpc.so")
                val mempoolSkel = File(musicRuntimeDir, "libggml-htp-v81.so")
                BackendDiagnostics.append(
                    this,
                    "FASTRPC",
                    "localCopy=${localFastRpc.isFile} size=${localFastRpc.length()} " +
                        "mempoolSkel=${mempoolSkel.isFile} skelSize=${mempoolSkel.length()}",
                )

                val message =
                    "YuE2 HTP: backend=FastRPC-mempool-0.4.9 selector=HTP0 " +
                        "session=$musicHtpSessionSpec model=$modelId max_seq=$MUSIC_MAX_SEQ " +
                        "flashAttention=auto-v81 runtime=${musicRuntimeDir.absolutePath}"
                Log.i(TAG, message)
                BackendDiagnostics.append(this, "ENV", message)
            }

            if (ditEngineDir != null) {
                if (backendType == "qwen21") {
                    // Keep the full platform/vendor host-library search path.
                    // libcdsprpc.so is provided by Qualcomm under /vendor/lib64
                    // on the target devices, so replacing LD_LIBRARY_PATH with
                    // runtimeDir:/system/lib64 makes the Hexagon backend unable
                    // to create HTP0.
                    //
                    // Multi-session Hexagon execution has an upstream silent
                    // corruption failure mode on large split graphs. "Auto"
                    // therefore resolves to one virtual session for correctness.
                    // "Dual" remains available for testing our strict scheduler
                    // split barriers without hiding the choice from the user.
                    val resolvedMode = if (config.htpMode == "dual") "dual" else "single"
                    env["GGML_HEXAGON_DEVICES"] =
                        if (resolvedMode == "dual") "HTP0:0,HTP0:1" else "HTP0:0"
                    env["LOCAL_DREAM_HTP_MODE"] = resolvedMode

                    // FastRPC's DSPQueue can fail with 0x2e when too many ops
                    // remain in flight on a large graph. Keep polling enabled
                    // and derive queue pressure from the actual transformer
                    // footprint instead of hardcoding precision/model names.
                    val queueProfile = qwenHexagonQueueProfile(modelsDir)
                    env["GGML_HEXAGON_OPPOLL"] = "1"
                    env["GGML_HEXAGON_OPBATCH"] = queueProfile.opBatch.toString()
                    env["GGML_HEXAGON_OPQUEUE"] = queueProfile.opQueue.toString()
                    val transformerGiB =
                        queueProfile.transformerBytes.toDouble() / (1024.0 * 1024.0 * 1024.0)
                    val queueMessage =
                        "Qwen HTP sessions: requested=${config.htpMode}, resolved=$resolvedMode, " +
                            "devices=${env["GGML_HEXAGON_DEVICES"]}, queue=${queueProfile.label} " +
                            "transformer=${String.format(java.util.Locale.US, "%.2f", transformerGiB)}GiB " +
                            "opbatch=${queueProfile.opBatch} opqueue=${queueProfile.opQueue} oppoll=1"
                    Log.i(TAG, queueMessage)
                    BackendDiagnostics.append(this, "QWEN_HTP", queueMessage)
                }
                // ggml-hexagon asks FastRPC for its skel by bare name, so both
                // the runtime directory holding the skels and the platform
                // defaults have to be on the DSP search path; dropping the
                // defaults would leave the skel unable to resolve the libraries
                // it links against.
                val dspPath = listOf(
                    runtimeDir.absolutePath,
                    "/vendor/lib/rfsa/adsp",
                    "/vendor/dsp/cdsp",
                    "/dsp",
                ).joinToString(";")
                env["ADSP_LIBRARY_PATH"] = dspPath
                env["DSP_LIBRARY_PATH"] = dspPath
                // The engine detects the HTP version over FastRPC itself, so
                // GGML_HEXAGON_ARCH stays unset: pinning it here would have to
                // be revisited for every new part, and an override that
                // disagrees with the hardware loads the wrong skel.
                Log.i(TAG, "DiT engine: ADSP_LIBRARY_PATH=$dspPath")
            }

            Log.d(TAG, "COMMAND: ${command.joinToString(" ")}")
            Log.d(TAG, "DIR: $runtimeDir")
            Log.d(TAG, "LD_LIBRARY_PATH=${env["LD_LIBRARY_PATH"]}")
            Log.d(TAG, "DSP_LIBRARY_PATH=${env["DSP_LIBRARY_PATH"]}")
            BackendDiagnostics.append(this, "COMMAND", command.joinToString(" "))
            BackendDiagnostics.append(
                this,
                "PATHS",
                "runtime=${if (isMusicBackend(backendType)) musicRuntimeDir.absolutePath else runtimeDir.absolutePath} native=$nativeDir",
            )

            val processBuilder = ProcessBuilder(command).apply {
                directory(if (isMusicBackend(backendType)) musicRuntimeDir else File(nativeDir))
                redirectErrorStream(true)
                environment().putAll(env)
            }

            val proc = processBuilder.start()
            process = proc

            startMonitorThread(proc, config)

            return true
        } catch (e: Exception) {
            Log.e(TAG, "backend start failed", e)
            BackendDiagnostics.appendThrowable(this, "START_ERROR", e)
            updateState(BackendState.Error("backend start failed: ${e.message}", config.modelId))
            return false
        }
    }

    private fun musicBackendNameFromLoadLine(line: String): String? =
        Regex("""\[Load]\s+\S+\s+backend:\s+([^\s(]+)""", RegexOption.IGNORE_CASE)
            .find(line)
            ?.groupValues
            ?.getOrNull(1)
            ?.trim()

    private fun updateMusicStartupFromLine(modelId: String, line: String) {
        val current = StateHolder._startupStatus.value
        if (current?.modelId != modelId) return

        val lower = line.lowercase()
        val next = when {
            "[store] created" in lower ->
                current.copy(
                    phase = "process",
                    detail = "Native process started",
                    progress = maxOf(current.progress, 0.10f),
                )
            "[bpe] loaded" in lower ->
                current.copy(
                    phase = "tokenizer",
                    detail = "Tokenizer ready",
                    progress = maxOf(current.progress, 0.22f),
                )
            "[gguf]" in lower ->
                current.copy(
                    phase = "model",
                    detail = "Validated YuE2 model metadata",
                    progress = maxOf(current.progress, 0.34f),
                )
            "loading driver libcdsprpc.so" in lower ->
                current.copy(
                    phase = "fastrpc",
                    detail = "Opening Qualcomm FastRPC transport",
                    progress = maxOf(current.progress, 0.46f),
                )
            "hexagon arch version" in lower ->
                current.copy(
                    phase = "htp",
                    detail = line.substringAfter("ggml-hex: ").trim(),
                    progress = maxOf(current.progress, 0.62f),
                )
            " new session " in lower && "ggml-hex:" in lower ->
                current.copy(
                    phase = "session",
                    detail = "Hexagon HTP session opened",
                    progress = maxOf(current.progress, 0.78f),
                )
            "[load]" in lower && "backend:" in lower -> {
                val backendName = musicBackendNameFromLoadLine(line)
                current.copy(
                    phase = if (backendName?.startsWith("HTP", ignoreCase = true) == true) {
                        "backend"
                    } else {
                        "backend_check"
                    },
                    detail = line.trim(),
                    progress = maxOf(current.progress, 0.86f),
                )
            }
            "[server] yue-server" in lower ->
                current.copy(
                    phase = "server",
                    detail = "YuE2 server initialized",
                    progress = maxOf(current.progress, 0.94f),
                )
            "[server] listening on" in lower ->
                current.copy(
                    phase = "ready",
                    detail = line.substringAfter("[Server] ").trim(),
                    progress = 1.0f,
                )
            else -> null
        }

        if (next != null) {
            updateStartup(next)
            if (next.phase == "ready") {
                updateState(BackendState.Running)
            }
        }
    }

    private fun startMonitorThread(proc: Process, config: BackendConfig) {
        Thread {
            var firstBackendError: String? = null
            var backendErrorCount = 0
            var musicHtpConfirmed = false
            val exitCode = try {
                proc.inputStream.bufferedReader().use { reader ->
                    var line: String?
                    while (reader.readLine().also { line = it } != null) {
                        val backendLine = line ?: continue
                        if (isMusicBackend(config.backendType)) {
                            musicBackendNameFromLoadLine(backendLine)?.let { backendName ->
                                if (backendName.startsWith("HTP", ignoreCase = true)) {
                                    musicHtpConfirmed = true
                                    BackendDiagnostics.append(
                                        this@BackendService,
                                        "HTP_SELECTED",
                                        "YuE2 native backend=$backendName",
                                    )
                                } else {
                                    val detail =
                                        "YuE2 requires Hexagon HTP, but native runtime selected $backendName"
                                    firstBackendError = firstBackendError ?: detail
                                    backendErrorCount++
                                    updateState(BackendState.Error(detail, config.modelId))
                                    StateHolder._startupStatus.value?.let { current ->
                                        if (current.modelId == config.modelId) {
                                            updateStartup(
                                                current.copy(
                                                    phase = "error",
                                                    detail = detail,
                                                ),
                                            )
                                        }
                                    }
                                    BackendDiagnostics.append(
                                        this@BackendService,
                                        "ERROR",
                                        detail,
                                    )
                                    // NPU-only policy: never leave yue-server
                                    // alive on a CPU-selected backend.
                                    proc.destroy()
                                }
                            }

                            if (
                                backendLine.contains("[Server] Listening on", ignoreCase = true) &&
                                !musicHtpConfirmed
                            ) {
                                val detail =
                                    "YuE2 server started without a confirmed Hexagon HTP backend"
                                firstBackendError = firstBackendError ?: detail
                                backendErrorCount++
                                updateState(BackendState.Error(detail, config.modelId))
                                BackendDiagnostics.append(
                                    this@BackendService,
                                    "ERROR",
                                    detail,
                                )
                                proc.destroy()
                            } else {
                                updateMusicStartupFromLine(config.modelId, backendLine)
                            }
                        }
                        extractBackendError(backendLine)?.let { detail ->
                            backendErrorCount++
                            if (firstBackendError == null) {
                                firstBackendError = detail
                            }
                            if (isMusicBackend(config.backendType)) {
                                val current = StateHolder._startupStatus.value
                                if (current?.modelId == config.modelId) {
                                    updateStartup(
                                        current.copy(
                                            phase = "error",
                                            detail = detail,
                                        ),
                                    )
                                }
                                updateState(BackendState.Error(detail, config.modelId))

                                val lower = backendLine.lowercase()
                                val fatalHtpExecution =
                                    "ggml_htp_execute_batch failed" in lower ||
                                        "weight_inval reset failed" in lower ||
                                        "aee_enorpcmemory" in lower
                                if (fatalHtpExecution && proc.isAlive) {
                                    BackendDiagnostics.append(
                                        this@BackendService,
                                        "FATAL_HTP",
                                        "Stopping YuE2 after unrecoverable HTP execution failure",
                                    )
                                    proc.destroy()
                                }
                            }
                        }
                        Log.i(TAG, "Backend: $backendLine")
                        BackendDiagnostics.append(
                            this@BackendService,
                            "NATIVE",
                            backendLine,
                        )
                        CrashDiagnostics.recordBackendLine(this@BackendService, backendLine)
                    }
                }
                val code = proc.waitFor()
                CrashDiagnostics.flushGeneration(this@BackendService)
                code
            } catch (e: Exception) {
                if (!isLiveCrash(proc)) {
                    Log.i(TAG, "backend monitor closed during intentional teardown: ${e.message}")
                    return@Thread
                }
                Log.e(TAG, "monitor error", e)
                BackendDiagnostics.appendThrowable(this@BackendService, "MONITOR_ERROR", e)
                if (isLiveCrash(proc)) {
                    updateState(
                        BackendState.Error(
                            "Backend monitor failed: ${e.message ?: e.javaClass.simpleName}",
                            config.modelId,
                        ),
                    )
                }
                return@Thread
            }
            Log.i(TAG, "Backend process exited with code: $exitCode")
            BackendDiagnostics.append(
                this@BackendService,
                "EXIT",
                "process exited code=$exitCode model=${config.modelId}",
            )
            CrashDiagnostics.record(
                this@BackendService,
                "BACKEND_EXIT",
                "process exited code=$exitCode model=${config.modelId}",
            )
            // Only surface as an error when this is still the active process and
            // we didn't intentionally stop it; a torn-down or superseded process
            // exiting is expected and must not poison the shared backendState.
            if (isLiveCrash(proc)) {
                val detail = firstBackendError?.let { first ->
                    if (backendErrorCount > 1) {
                        "$first (+${backendErrorCount - 1} more backend errors)"
                    } else {
                        first
                    }
                }
                updateState(
                    BackendState.Error(
                        detail?.let { "Backend failed (code $exitCode): $it" }
                            ?: "Backend process exited with code: $exitCode",
                        config.modelId,
                    ),
                )
            } else {
                Log.i(TAG, "backend exit ($exitCode) was intentional/stale, not reporting")
            }
        }.apply {
            isDaemon = true
            start()
        }
    }

    // True when proc is still the tracked process and no intentional stop is in
    // progress, i.e. its exit really is an unexpected crash worth reporting.
    private fun isLiveCrash(proc: Process): Boolean = !stopping && process === proc

    override fun onDestroy() {
        super.onDestroy()
        // The scope is never cancelled, so this job still runs after
        // onDestroy returns; closing the dispatcher afterwards lets its
        // thread wind down once the backend process has exited.
        serviceScope.launch {
            idleStopJob?.cancel()
            stopBackend()
            backendDispatcher.close()
        }
    }

    private fun stopBackend() {
        Log.i(TAG, "to stop backend")
        // Mark the upcoming exit as intentional before destroy() so the monitor
        // thread (which wakes the instant the process dies) won't race ahead and
        // report it as a crash.
        stopping = true
        process?.let { proc ->
            try {
                proc.destroy()

                val gracefulSeconds = if (serving?.backendType == "yue2") 15L else 5L
                if (!proc.waitFor(gracefulSeconds, TimeUnit.SECONDS)) {
                    Log.w(TAG, "backend did not exit in ${gracefulSeconds}s; forcing stop")
                    proc.destroyForcibly()
                }

                Log.i(TAG, "process end, code: ${proc.exitValue()}")
                updateState(BackendState.Idle)
            } catch (e: Exception) {
                Log.e(TAG, "error", e)
                updateState(BackendState.Error("error: ${e.message}"))
            } finally {
                process = null
            }
        }
        serving = null
        updateServing(null)
        if (stopping) {
            updateStartup(null)
        }
    }
}
