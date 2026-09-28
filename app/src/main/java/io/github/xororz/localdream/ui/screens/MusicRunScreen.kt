package io.github.xororz.localdream.ui.screens

import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.BatteryManager
import android.os.Build
import android.os.PowerManager
import androidx.compose.animation.AnimatedContent
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.togetherWith
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.tween
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.AutoAwesome
import androidx.compose.material.icons.filled.Bolt
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.History
import androidx.compose.material.icons.filled.Memory
import androidx.compose.material.icons.filled.MusicNote
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.Speed
import androidx.compose.material.icons.filled.Tune
import io.github.xororz.localdream.service.MusicTransportBenchmarkResult
import androidx.compose.material3.Button
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ElevatedCard
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilterChip
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.MediumTopAppBar
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Slider
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableLongStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.produceState
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.navigation.NavController
import io.github.xororz.localdream.data.ModelRepository
import io.github.xororz.localdream.navigation.popBackStackIfResumed
import io.github.xororz.localdream.service.BackendService
import io.github.xororz.localdream.service.MusicGenerationService
import io.github.xororz.localdream.service.MusicGenerationService.MusicState
import io.github.xororz.localdream.service.MusicHistoryItem
import io.github.xororz.localdream.service.MusicHistoryStore
import io.github.xororz.localdream.service.MusicTransportBenchmark
import io.github.xororz.localdream.ui.components.MusicPlayerCard
import io.github.xororz.localdream.ui.components.SmoothIndeterminateLinearWavyProgressIndicator
import io.github.xororz.localdream.ui.components.SmoothLinearWavyProgressIndicator
import io.github.xororz.localdream.utils.AppHaptics
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlin.math.abs
import android.widget.Toast

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun MusicRunScreen(
    modelId: String,
    navController: NavController,
) {
    val context = androidx.compose.ui.platform.LocalContext.current
    val repository = remember { ModelRepository.getInstance(context) }
    LaunchedEffect(Unit) { repository.ensureLoaded() }
    val model = repository.models.firstOrNull { it.id == modelId }

    val backendState by BackendService.backendState.collectAsState()
    val startupStatus by BackendService.startupStatus.collectAsState()
    val servingModelId by BackendService.servingModelId.collectAsState()
    val musicState by MusicGenerationService.state.collectAsState()

    var style by rememberSaveable {
        mutableStateOf("solo acoustic piano, instrumental, gentle expressive melody, warm intimate tone, no vocals")
    }
    var lyrics by rememberSaveable {
        mutableStateOf("")
    }
    var duration by rememberSaveable { mutableIntStateOf(60) }
    var planning by rememberSaveable { mutableStateOf("full") }
    var steps by rememberSaveable { mutableIntStateOf(32) }
    var odeMethod by rememberSaveable { mutableStateOf("midpoint") }
    var outputFormat by rememberSaveable { mutableStateOf("wav16") }
    var history by remember { mutableStateOf(MusicHistoryStore.load(context)) }
    var seed by rememberSaveable { mutableLongStateOf(-1L) }
    var semanticTemperature by rememberSaveable { mutableFloatStateOf(1f) }
    var semanticTopP by rememberSaveable { mutableFloatStateOf(0.95f) }
    var cfgScale by rememberSaveable { mutableFloatStateOf(-1f) }
    var showConfig by remember { mutableStateOf(false) }
    var showHistorySheet by remember { mutableStateOf(false) }
    var showScore by remember { mutableStateOf(false) }
    var lastHapticPhase by remember { mutableStateOf<String?>(null) }
    var lastProgressBucket by remember { mutableIntStateOf(-1) }
    var lastTerminalHaptic by remember { mutableStateOf<String?>(null) }

    val backendReady = backendState is BackendService.BackendState.Running &&
        servingModelId == modelId
    val precision = model?.variantPrecision.orEmpty()
    val htpAccelerated = precision in setOf("Q5_K_M", "Q6_K", "Q8_0", "BF16")
    val q8FastPath = precision in setOf("Q5_K_M", "Q6_K", "Q8_0", "BF16")
    val startupForModel = startupStatus?.takeIf { it.modelId == modelId }
    val backendError = (backendState as? BackendService.BackendState.Error)
        ?.takeIf { it.modelId == null || it.modelId == modelId }

    LaunchedEffect(musicState) {
        when (val state = musicState) {
            is MusicState.Generating -> {
                if (state.phase != lastHapticPhase) {
                    if (lastHapticPhase != null) {
                        AppHaptics.perform(context, AppHaptics.Kind.Stage)
                    }
                    lastHapticPhase = state.phase
                }

                state.progress?.let { progress ->
                    val bucket = (progress.coerceIn(0f, 1f) * 10f).toInt()
                    if (bucket in 1..9 && bucket > lastProgressBucket) {
                        AppHaptics.perform(context, AppHaptics.Kind.Progress)
                        lastProgressBucket = bucket
                    }
                }
                lastTerminalHaptic = null
            }

            is MusicState.Complete -> {
                history = MusicHistoryStore.load(context)
                if (lastTerminalHaptic != "complete") {
                    AppHaptics.perform(context, AppHaptics.Kind.Success)
                    lastTerminalHaptic = "complete"
                }
            }

            is MusicState.Error -> {
                if (lastTerminalHaptic != "error") {
                    AppHaptics.perform(context, AppHaptics.Kind.Failure)
                    lastTerminalHaptic = "error"
                }
            }

            MusicState.Idle -> {
                lastHapticPhase = null
                lastProgressBucket = -1
            }

            else -> Unit
        }
    }

    LaunchedEffect(backendError?.message) {
        if (backendError != null && lastTerminalHaptic != "backend_error") {
            AppHaptics.perform(context, AppHaptics.Kind.Failure)
            lastTerminalHaptic = "backend_error"
        }
    }

    LaunchedEffect(model?.id) {
        if (model == null || !model.isDownloaded || !model.isMusic) return@LaunchedEffect
        MusicGenerationService.resetForModel(model.id)
        context.startForegroundService(
            Intent(context, BackendService::class.java).apply {
                putExtra("modelId", model.id)
                putExtra("backendType", model.backendType)
                putExtra("width", 512)
                putExtra("height", 512)
                putExtra("htp_mode", "single")
            },
        )
    }

    DisposableEffect(Unit) {
        onDispose {
            // Cancel a live warmup/generation before the native process is
            // asked to exit. yue-server gets time to unwind its active job,
            // which avoids the cancel/back crash race.
            MusicGenerationService.stop(context)
            MusicGenerationService.clearResident()
            context.startService(
                Intent(context, BackendService::class.java).setAction(BackendService.ACTION_STOP),
            )
        }
    }

    Scaffold(
        topBar = {
            MediumTopAppBar(
                title = {
                    Column {
                        Text("YuE2 · Text to music")
                        Text(
                            when {
                                backendError != null -> "Native runtime failed"
                                !backendReady && startupForModel != null -> startupForModel.detail
                                !backendReady -> "Starting native YuE2 runtime…"
                                !htpAccelerated ->
                                    "Compatibility runtime · $precision"
                                else ->
                                    "$precision · HTP runtime ready"
                            },
                            style = MaterialTheme.typography.bodySmall,
                            color = if (backendReady) {
                                MaterialTheme.colorScheme.primary
                            } else {
                                MaterialTheme.colorScheme.onSurfaceVariant
                            },
                        )
                    }
                },
                navigationIcon = {
                    IconButton(onClick = { navController.popBackStackIfResumed() }) {
                        Icon(Icons.AutoMirrored.Filled.ArrowBack, contentDescription = "Back")
                    }
                },
                actions = {
                    IconButton(
                        onClick = {
                            AppHaptics.perform(context, AppHaptics.Kind.Interaction)
                            showHistorySheet = true
                        },
                    ) {
                        Icon(Icons.Default.History, contentDescription = "Music history")
                    }
                    IconButton(
                        onClick = {
                            AppHaptics.perform(context, AppHaptics.Kind.Interaction)
                            showConfig = true
                        },
                    ) {
                        Icon(Icons.Default.Tune, contentDescription = "Generation settings")
                    }
                },
            )
        },
    ) { padding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
                .verticalScroll(rememberScrollState()),
            verticalArrangement = Arrangement.spacedBy(14.dp),
        ) {
            if (!backendReady && backendError == null) {
                NativeRuntimeStartupCard(
                    status = startupForModel,
                    modelId = modelId,
                )
            }

            if (backendError != null) {
                NativeRuntimeErrorCard(
                    message = backendError.message,
                    onRestart = {
                        context.startForegroundService(
                            Intent(context, BackendService::class.java).apply {
                                action = BackendService.ACTION_RESTART
                                putExtra("modelId", modelId)
                                putExtra("backendType", "yue2")
                                putExtra("width", 512)
                                putExtra("height", 512)
                                putExtra("htp_mode", "single")
                            },
                        )
                    },
                )
            }

            ElevatedCard(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 16.dp),
                shape = MaterialTheme.shapes.extraLarge,
                colors = CardDefaults.elevatedCardColors(
                    containerColor = MaterialTheme.colorScheme.surfaceContainer,
                ),
            ) {
                Column(
                    modifier = Modifier.padding(18.dp),
                    verticalArrangement = Arrangement.spacedBy(14.dp),
                ) {
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(12.dp),
                    ) {
                        Surface(
                            shape = MaterialTheme.shapes.extraLarge,
                            color = MaterialTheme.colorScheme.tertiaryContainer,
                        ) {
                            Icon(
                                Icons.Default.MusicNote,
                                contentDescription = null,
                                modifier = Modifier.padding(12.dp).size(24.dp),
                                tint = MaterialTheme.colorScheme.onTertiaryContainer,
                            )
                        }
                        Column(modifier = Modifier.weight(1f)) {
                            Text(
                                "Describe the track",
                                style = MaterialTheme.typography.titleLarge,
                                fontWeight = FontWeight.SemiBold,
                            )
                            Text(
                                "Style + optional structured lyrics. Audio conditioning is intentionally off for now.",
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant,
                            )
                        }
                    }

                    OutlinedTextField(
                        value = style,
                        onValueChange = { style = it.take(1000) },
                        modifier = Modifier.fillMaxWidth(),
                        label = { Text("Style prompt") },
                        placeholder = { Text("genre, mood, vocals, instruments, production…") },
                        minLines = 3,
                    )
                    OutlinedTextField(
                        value = lyrics,
                        onValueChange = { lyrics = it.take(6000) },
                        modifier = Modifier.fillMaxWidth(),
                        label = { Text("Lyrics · optional") },
                        supportingText = {
                            Text("Use [Verse] / [Chorus] labels, or leave empty for instrumental generation.")
                        },
                        minLines = 5,
                    )

                    Surface(
                        shape = MaterialTheme.shapes.large,
                        color = MaterialTheme.colorScheme.surfaceContainerHigh,
                    ) {
                        Row(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(horizontal = 14.dp, vertical = 12.dp),
                            verticalAlignment = Alignment.CenterVertically,
                        ) {
                            Column(modifier = Modifier.weight(1f)) {
                                Text(
                                    "$duration s · ${qualityLabel(steps, odeMethod)} · ${planningLabel(planning)}",
                                    style = MaterialTheme.typography.titleSmall,
                                    fontWeight = FontWeight.SemiBold,
                                )
                                Text(
                                    buildString {
                                        append(model?.variantPrecision ?: "GGUF")
                                        append(" · 48 kHz stereo · ")
                                        append(outputFormatLabel(outputFormat))
                                        append(" · ")
                                        append(if (htpAccelerated) "Hexagon HTP" else "NPU unavailable")
                                    },
                                    style = MaterialTheme.typography.labelMedium,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                                )
                            }
                            TextButton(
                                onClick = {
                                    AppHaptics.perform(context, AppHaptics.Kind.Interaction)
                                    showConfig = true
                                },
                            ) {
                                Text("Tune")
                            }
                        }
                    }

                    Button(
                        onClick = {
                            AppHaptics.perform(context, AppHaptics.Kind.Interaction)
                            MusicGenerationService.reset()
                            context.startForegroundService(
                                Intent(context, MusicGenerationService::class.java)
                                    .setAction(MusicGenerationService.ACTION_GENERATE)
                                    .apply {
                                        putExtra("modelId", modelId)
                                        putExtra("style", style)
                                        putExtra("lyrics", lyrics)
                                        putExtra("cot", planning)
                                        putExtra("duration", duration)
                                        putExtra("steps", steps)
                                        putExtra("ode_method", odeMethod)
                                        putExtra("seed", seed)
                                        putExtra("semantic_temperature", semanticTemperature)
                                        putExtra("semantic_top_p", semanticTopP)
                                        putExtra("cfg_scale", cfgScale)
                                        putExtra("output_format", outputFormat)
                                    },
                            )
                        },
                        enabled = backendReady &&
                            htpAccelerated &&
                            style.isNotBlank() &&
                            musicState !is MusicState.Generating,
                        modifier = Modifier.fillMaxWidth(),
                        shape = MaterialTheme.shapes.large,
                    ) {
                        Icon(Icons.Default.PlayArrow, contentDescription = null)
                        Spacer(Modifier.width(8.dp))
                        Text(
                            when {
                                backendError != null -> "Runtime unavailable"
                                !backendReady && startupForModel != null ->
                                    startupButtonLabel(startupForModel.phase)
                                !backendReady -> "Starting native runtime…"
                                !htpAccelerated -> "Generate · compatibility"
                                else -> "Generate music"
                            },
                        )
                    }
                }
            }

            if (backendReady && !htpAccelerated) {
                Surface(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(horizontal = 16.dp),
                    shape = MaterialTheme.shapes.large,
                    color = MaterialTheme.colorScheme.errorContainer.copy(alpha = 0.72f),
                ) {
                    Text(
                        "This YuE2 precision is not supported by the NPU-only music runtime.",
                        modifier = Modifier.padding(14.dp),
                        color = MaterialTheme.colorScheme.onErrorContainer,
                    )
                }
            }

            AnimatedContent(
                targetState = musicState,
                transitionSpec = { fadeIn() togetherWith fadeOut() },
                label = "musicGenerationState",
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 16.dp),
            ) { state ->
                when (state) {
                    is MusicState.Preloading -> MusicPreloadCard(
                        state = state,
                        precision = precision.ifBlank { "GGUF" },
                        onCancel = { MusicGenerationService.stop(context) },
                    )
                    is MusicState.Ready -> MusicReadyCard(
                        precision = precision.ifBlank { "GGUF" },
                        preloadMillis = state.preloadMillis,
                    )
                    is MusicState.Generating -> MusicProgressCard(
                        state = state,
                        precision = model?.variantPrecision ?: "GGUF",
                        runtimeLabel = "HTP0 · Native-only DSPQueue",
                        onCancel = {
                            AppHaptics.perform(context, AppHaptics.Kind.Interaction)
                            MusicGenerationService.stop(context)
                        },
                    )
                    is MusicState.Complete -> {
                        Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
                            MusicPlayerCard(
                                file = state.file,
                                title = "YuE2 generation",
                                subtitle = "48 kHz stereo · ${state.targetSeconds}.0 s audio · generated in " +
                                    String.format(java.util.Locale.US, "%.1f s", state.elapsedMillis / 1000.0),
                            )
                            val currentItem = history.firstOrNull { it.id == state.historyId }
                                ?: MusicHistoryItem(
                                    id = state.historyId.ifBlank { System.currentTimeMillis().toString() },
                                    filePath = state.file.absolutePath,
                                    style = state.style.ifBlank { style },
                                    lyrics = state.lyrics.ifBlank { lyrics },
                                    score = state.score,
                                    lmSeed = state.lmSeed,
                                    acousticSeed = state.acousticSeed,
                                    targetSeconds = state.targetSeconds,
                                    elapsedMillis = state.elapsedMillis,
                                    createdAtMillis = System.currentTimeMillis(),
                                    format = state.format.ifBlank { state.file.extension },
                                    solver = state.solver,
                                    steps = state.steps,
                                    planning = state.planning,
                                    modelId = state.modelId,
                                )
                            MusicTrackActions(
                                item = currentItem,
                                onHistoryChanged = { history = MusicHistoryStore.load(context) },
                            )
                            if (state.score.isNotBlank()) {
                                ElevatedCard(
                                    modifier = Modifier.fillMaxWidth(),
                                    colors = CardDefaults.elevatedCardColors(
                                        containerColor = MaterialTheme.colorScheme.surfaceContainer,
                                    ),
                                ) {
                                    Column(Modifier.padding(16.dp)) {
                                        Row(verticalAlignment = Alignment.CenterVertically) {
                                            Column(Modifier.weight(1f)) {
                                                Text(
                                                    "Symbolic plan",
                                                    style = MaterialTheme.typography.titleSmall,
                                                    fontWeight = FontWeight.SemiBold,
                                                )
                                                Text(
                                                    "YuE2 composed this ABC score before rendering the audio.",
                                                    style = MaterialTheme.typography.bodySmall,
                                                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                                                )
                                            }
                                            TextButton(onClick = { showScore = !showScore }) {
                                                Text(if (showScore) "Hide" else "Show")
                                            }
                                        }
                                        AnimatedVisibility(visible = showScore) {
                                            Text(
                                                state.score,
                                                modifier = Modifier
                                                    .fillMaxWidth()
                                                    .heightIn(max = 260.dp)
                                                    .verticalScroll(rememberScrollState())
                                                    .padding(top = 10.dp),
                                                style = MaterialTheme.typography.bodySmall.copy(
                                                    fontFamily = FontFamily.Monospace,
                                                ),
                                            )
                                        }
                                    }
                                }
                            }
                        }
                    }
                    is MusicState.Error -> {
                        ElevatedCard(
                            colors = CardDefaults.elevatedCardColors(
                                containerColor = MaterialTheme.colorScheme.errorContainer,
                            ),
                        ) {
                            Column(Modifier.padding(16.dp)) {
                                Text(
                                    "Generation failed",
                                    fontWeight = FontWeight.SemiBold,
                                    color = MaterialTheme.colorScheme.onErrorContainer,
                                )
                                Text(
                                    state.message,
                                    color = MaterialTheme.colorScheme.onErrorContainer,
                                )
                                if (q8FastPath) {
                                    Spacer(Modifier.size(8.dp))
                                    TextButton(
                                        onClick = {
                                            context.startForegroundService(
                                                Intent(context, BackendService::class.java).apply {
                                                    action = BackendService.ACTION_RESTART
                                                    putExtra("modelId", modelId)
                                                    putExtra("backendType", "yue2")
                                                },
                                            )
                                        },
                                    ) {
                                        Text("Restart native runtime")
                                    }
                                }
                            }
                        }
                    }
                    MusicState.Idle -> {
                        if (backendReady) Surface(
                            shape = MaterialTheme.shapes.extraLarge,
                            color = MaterialTheme.colorScheme.surfaceContainerLow,
                        ) {
                            Row(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .padding(16.dp),
                                verticalAlignment = Alignment.CenterVertically,
                                horizontalArrangement = Arrangement.spacedBy(10.dp),
                            ) {
                                Icon(
                                    Icons.Default.Bolt,
                                    contentDescription = null,
                                    tint = MaterialTheme.colorScheme.primary,
                                )
                                Text(
                                    if (q8FastPath) {
                                        "Mobile-safe staged loading: AR → NAR → Oobleck. Only one heavy module stays on HTP at a time."
                                    } else {
                                        "Pipeline: AR score → semantic codes → NAR flow → Oobleck decode"
                                    },
                                    style = MaterialTheme.typography.bodySmall,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                                )
                            }
                        }
                    }
                }
            }
            if (history.isNotEmpty()) {
                MusicHistorySection(
                    items = history.filterNot {
                        val current = musicState as? MusicState.Complete
                        current != null && it.id == current.historyId
                    },
                    onHistoryChanged = { history = MusicHistoryStore.load(context) },
                )
            }
            Spacer(Modifier.size(22.dp))
        }
    }

    if (showConfig) {
        MusicGenerationConfigSheet(
            duration = duration,
            onDuration = { duration = it },
            planning = planning,
            onPlanning = { planning = it },
            steps = steps,
            onSteps = { steps = it },
            odeMethod = odeMethod,
            onOdeMethod = { odeMethod = it },
            seed = seed,
            onSeed = { seed = it },
            temperature = semanticTemperature,
            onTemperature = { semanticTemperature = it },
            topP = semanticTopP,
            onTopP = { semanticTopP = it },
            cfgScale = cfgScale,
            onCfgScale = { cfgScale = it },
            outputFormat = outputFormat,
            onOutputFormat = { outputFormat = it },
            onDismiss = { showConfig = false },
        )
    }

    if (showHistorySheet) {
        MusicHistorySheet(
            items = history,
            onHistoryChanged = { history = MusicHistoryStore.load(context) },
            onDismiss = { showHistorySheet = false },
        )
    }
}

@Composable
private fun NativeRuntimeStartupCard(
    status: BackendService.BackendStartupStatus?,
    modelId: String,
) {
    val startedAt = status?.startedAtMillis ?: remember(modelId) { System.currentTimeMillis() }
    val elapsed by produceState(initialValue = 0L, startedAt) {
        while (true) {
            value = (System.currentTimeMillis() - startedAt).coerceAtLeast(0L) / 1000L
            delay(250)
        }
    }

    ElevatedCard(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 16.dp),
        shape = MaterialTheme.shapes.extraLarge,
        colors = CardDefaults.elevatedCardColors(
            containerColor = MaterialTheme.colorScheme.surfaceContainer,
        ),
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(11.dp),
            ) {
                Surface(
                    shape = MaterialTheme.shapes.large,
                    color = MaterialTheme.colorScheme.primaryContainer,
                ) {
                    Icon(
                        Icons.Default.Memory,
                        contentDescription = null,
                        modifier = Modifier.padding(10.dp).size(21.dp),
                        tint = MaterialTheme.colorScheme.onPrimaryContainer,
                    )
                }
                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        "Starting YuE2 on Hexagon",
                        style = MaterialTheme.typography.titleMedium,
                        fontWeight = FontWeight.SemiBold,
                    )
                    Text(
                        status?.detail ?: "Launching native process",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
                Text(
                    "${elapsed}s",
                    style = MaterialTheme.typography.labelMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }

            NativeStartupStageStrip(status?.phase ?: "launch")

            if (status != null) {
                SmoothLinearWavyProgressIndicator(
                    progress = status.progress.coerceIn(0f, 1f),
                    modifier = Modifier.fillMaxWidth(),
                )
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                ) {
                    Text(
                        startupPhaseLabel(status.phase),
                        style = MaterialTheme.typography.labelMedium,
                        color = MaterialTheme.colorScheme.primary,
                    )
                    Text(
                        "${(status.progress * 100).toInt()}% startup",
                        style = MaterialTheme.typography.labelMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
            } else {
                SmoothIndeterminateLinearWavyProgressIndicator(
                    modifier = Modifier.fillMaxWidth(),
                )
            }

            Text(
                "Live native milestones · FastRPC → HTP v81 → yue-server. " +
                    "This progress updates from the actual backend log, not a timer.",
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
    }
}

@Composable
private fun NativeRuntimeErrorCard(
    message: String,
    onRestart: () -> Unit,
) {
    ElevatedCard(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 16.dp),
        shape = MaterialTheme.shapes.extraLarge,
        colors = CardDefaults.elevatedCardColors(
            containerColor = MaterialTheme.colorScheme.errorContainer,
        ),
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp),
        ) {
            Text(
                "Native runtime failed",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.SemiBold,
                color = MaterialTheme.colorScheme.onErrorContainer,
            )
            Text(
                message,
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onErrorContainer,
            )
            HorizontalDivider(
                color = MaterialTheme.colorScheme.onErrorContainer.copy(alpha = 0.20f),
            )
            TextButton(onClick = onRestart) {
                Text("Restart YuE2 runtime")
            }
        }
    }
}

@Composable
private fun NativeStartupStageStrip(activePhase: String) {
    val stages = listOf(
        "process" to "Process",
        "model" to "Model",
        "fastrpc" to "FastRPC",
        "htp" to "HTP",
        "server" to "Server",
    )
    val index = when (activePhase) {
        "launch", "process" -> 0
        "tokenizer", "model" -> 1
        "fastrpc" -> 2
        "htp", "session", "backend" -> 3
        "server", "ready" -> 4
        else -> 0
    }

    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.spacedBy(6.dp),
    ) {
        stages.forEachIndexed { i, (_, label) ->
            Surface(
                modifier = Modifier.weight(1f),
                shape = MaterialTheme.shapes.small,
                color = if (i <= index) {
                    MaterialTheme.colorScheme.primaryContainer
                } else {
                    MaterialTheme.colorScheme.surfaceContainerHighest
                },
            ) {
                Text(
                    label,
                    modifier = Modifier.padding(vertical = 6.dp),
                    style = MaterialTheme.typography.labelSmall,
                    color = if (i <= index) {
                        MaterialTheme.colorScheme.onPrimaryContainer
                    } else {
                        MaterialTheme.colorScheme.onSurfaceVariant
                    },
                    textAlign = androidx.compose.ui.text.style.TextAlign.Center,
                )
            }
        }
    }
}

private fun startupPhaseLabel(phase: String): String = when (phase) {
    "launch" -> "Launching process"
    "process" -> "Process started"
    "tokenizer" -> "Tokenizer"
    "model" -> "Model metadata"
    "fastrpc" -> "FastRPC transport"
    "htp" -> "Detecting Hexagon"
    "session" -> "Opening HTP session"
    "backend" -> "Selecting HTP backend"
    "server" -> "Starting yue-server"
    "ready" -> "Ready"
    else -> phase.replace('_', ' ').replaceFirstChar { it.uppercase() }
}

private fun startupButtonLabel(phase: String): String = when (phase) {
    "launch", "process" -> "Launching YuE2…"
    "tokenizer", "model" -> "Reading model…"
    "fastrpc" -> "Opening FastRPC…"
    "htp", "session", "backend" -> "Initializing HTP…"
    "server" -> "Starting server…"
    else -> "Starting native runtime…"
}

@Composable
private fun MusicPreloadCard(
    state: MusicState.Preloading,
    precision: String,
    onCancel: () -> Unit,
) {
    val elapsed by produceState(initialValue = 0L, state.startedAtMillis) {
        while (true) {
            value = (System.currentTimeMillis() - state.startedAtMillis)
                .coerceAtLeast(0L) / 1000L
            delay(500)
        }
    }

    ElevatedCard(
        shape = MaterialTheme.shapes.extraLarge,
        colors = CardDefaults.elevatedCardColors(
            containerColor = MaterialTheme.colorScheme.surfaceContainer,
        ),
    ) {
        Column(
            modifier = Modifier.padding(18.dp),
            verticalArrangement = Arrangement.spacedBy(14.dp),
        ) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        "Loading $precision",
                        style = MaterialTheme.typography.titleMedium,
                        fontWeight = FontWeight.SemiBold,
                    )
                    Text(
                        state.detail,
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
                OutlinedButton(onClick = onCancel) {
                    Icon(Icons.Default.Close, contentDescription = null)
                    Spacer(Modifier.width(5.dp))
                    Text("Cancel")
                }
            }

            MusicStageStrip(activePhase = state.phase, preload = true)

            if (state.progress != null) {
                SmoothLinearWavyProgressIndicator(
                    progress = state.progress,
                    modifier = Modifier.fillMaxWidth(),
                )
            } else {
                SmoothIndeterminateLinearWavyProgressIndicator(
                    modifier = Modifier.fillMaxWidth(),
                )
            }

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
            ) {
                Text(
                    when {
                        state.total > 0 -> "${state.step}/${state.total}"
                        state.progress != null -> "${(state.progress * 100).toInt()}%"
                        else -> "Native load"
                    },
                    style = MaterialTheme.typography.labelMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
                Text(
                    "Elapsed ${elapsed}s",
                    style = MaterialTheme.typography.labelMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }

            Text(
                "YuE2 native warmup progress.",
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
    }
}

@Composable
private fun MusicReadyCard(
    precision: String,
    preloadMillis: Long,
) {
    Surface(
        shape = MaterialTheme.shapes.extraLarge,
        color = MaterialTheme.colorScheme.primaryContainer.copy(alpha = 0.55f),
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(10.dp),
        ) {
            Icon(
                Icons.Default.Bolt,
                contentDescription = null,
                tint = MaterialTheme.colorScheme.primary,
            )
            Column {
                Text(
                    "$precision · ready",
                    style = MaterialTheme.typography.titleSmall,
                    fontWeight = FontWeight.SemiBold,
                )
                Text(
                    if (preloadMillis > 0L) {
                        "Native pipeline prepared in " +
                            String.format(java.util.Locale.US, "%.1fs", preloadMillis / 1000f)
                    } else {
                        "Native YuE2 server is ready."
                    },
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        }
    }
}

@Composable
private fun MusicStageStrip(
    activePhase: String,
    preload: Boolean,
) {
    val stages = if (preload) {
        listOf(
            "server" to "Server",
            "composer" to "AR",
            "renderer" to "NAR",
            "decoder" to "VAE",
            "finalizing" to "Ready",
        )
    } else {
        listOf(
            "planning" to "Plan",
            "semantic" to "Tokens",
            "flow" to "Render",
            "decoding" to "Decode",
            "finalizing" to "Finish",
        )
    }
    val aliases = when (activePhase) {
        "submit", "composer_ready" -> if (preload) "composer" else activePhase
        "loading_ar", "starting", "queued" -> if (preload) "composer" else "planning"
        "loading_nar" -> if (preload) "renderer" else "flow"
        "loading_vae", "result" -> if (preload) "decoder" else "decoding"
        else -> activePhase
    }
    val activeIndex = stages.indexOfFirst { it.first == aliases }.coerceAtLeast(0)

    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.spacedBy(6.dp),
    ) {
        stages.forEachIndexed { index, (_, label) ->
            Surface(
                modifier = Modifier.weight(1f),
                shape = MaterialTheme.shapes.small,
                color = if (index <= activeIndex) {
                    MaterialTheme.colorScheme.primaryContainer
                } else {
                    MaterialTheme.colorScheme.surfaceContainerHighest
                },
            ) {
                Text(
                    label,
                    modifier = Modifier.padding(vertical = 6.dp),
                    style = MaterialTheme.typography.labelSmall,
                    color = if (index <= activeIndex) {
                        MaterialTheme.colorScheme.onPrimaryContainer
                    } else {
                        MaterialTheme.colorScheme.onSurfaceVariant
                    },
                    textAlign = androidx.compose.ui.text.style.TextAlign.Center,
                )
            }
        }
    }
}

private data class MusicDeviceTelemetry(
    val batteryPercent: Int? = null,
    val batteryTempC: Float? = null,
    val voltageV: Float? = null,
    val currentMa: Float? = null,
    val powerW: Float? = null,
    val thermal: String = "Unknown",
    val thermalHeadroom: Float? = null,
)

@Composable
private fun rememberMusicDeviceTelemetry(context: Context): MusicDeviceTelemetry {
    val telemetry by produceState(initialValue = MusicDeviceTelemetry(), context) {
        val batteryManager = context.getSystemService(Context.BATTERY_SERVICE) as BatteryManager
        val powerManager = context.getSystemService(Context.POWER_SERVICE) as PowerManager
        while (true) {
            val sticky = context.registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
            val level = sticky?.getIntExtra(BatteryManager.EXTRA_LEVEL, -1) ?: -1
            val scale = sticky?.getIntExtra(BatteryManager.EXTRA_SCALE, 100) ?: 100
            val percent = if (level >= 0 && scale > 0) level * 100 / scale else null
            val tempRaw = sticky?.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, Int.MIN_VALUE)
            val temp = tempRaw?.takeIf { it != Int.MIN_VALUE }?.div(10f)
            val voltageMv = sticky?.getIntExtra(BatteryManager.EXTRA_VOLTAGE, Int.MIN_VALUE)
            val voltage = voltageMv?.takeIf { it != Int.MIN_VALUE }?.div(1000f)
            val currentUa = batteryManager.getIntProperty(BatteryManager.BATTERY_PROPERTY_CURRENT_NOW)
            val currentMa = currentUa.takeIf { it != Int.MIN_VALUE }?.div(1000f)
            val watts = if (currentMa != null && voltage != null) {
                abs(currentMa / 1000f * voltage)
            } else {
                null
            }

            val thermalStatus = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                when (powerManager.currentThermalStatus) {
                    PowerManager.THERMAL_STATUS_NONE -> "Cool"
                    PowerManager.THERMAL_STATUS_LIGHT -> "Light"
                    PowerManager.THERMAL_STATUS_MODERATE -> "Moderate"
                    PowerManager.THERMAL_STATUS_SEVERE -> "Severe"
                    PowerManager.THERMAL_STATUS_CRITICAL -> "Critical"
                    PowerManager.THERMAL_STATUS_EMERGENCY -> "Emergency"
                    PowerManager.THERMAL_STATUS_SHUTDOWN -> "Shutdown"
                    else -> "Unknown"
                }
            } else {
                "Unknown"
            }
            val headroom = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
                runCatching { powerManager.getThermalHeadroom(0) }
                    .getOrNull()
                    ?.takeIf { it.isFinite() }
            } else {
                null
            }

            value = MusicDeviceTelemetry(
                batteryPercent = percent,
                batteryTempC = temp,
                voltageV = voltage,
                currentMa = currentMa?.let(::abs),
                powerW = watts,
                thermal = thermalStatus,
                thermalHeadroom = headroom,
            )
            delay(1000)
        }
    }
    return telemetry
}

@Composable
private fun MusicDeviceTelemetryCard(telemetry: MusicDeviceTelemetry) {
    val primary = buildString {
        telemetry.batteryPercent?.let { append("Battery $it%") }
        telemetry.batteryTempC?.let {
            if (isNotEmpty()) append(" · ")
            append(String.format(java.util.Locale.US, "%.1f°C", it))
        }
        telemetry.powerW?.let {
            if (isNotEmpty()) append(" · ")
            append(String.format(java.util.Locale.US, "%.1f W", it))
        }
        telemetry.currentMa?.let {
            if (isNotEmpty()) append(" · ")
            append(String.format(java.util.Locale.US, "%.0f mA", it))
        }
    }.ifBlank { "Battery telemetry unavailable" }

    val secondary = buildString {
        append("Thermal ")
        append(telemetry.thermal)
        telemetry.thermalHeadroom?.let {
            append(" · headroom ")
            append(String.format(java.util.Locale.US, "%.2f", it))
        }
        telemetry.voltageV?.let {
            append(" · ")
            append(String.format(java.util.Locale.US, "%.2f V", it))
        }
    }

    Surface(
        modifier = Modifier.fillMaxWidth(),
        shape = MaterialTheme.shapes.medium,
        color = MaterialTheme.colorScheme.surfaceContainerHighest,
    ) {
        Column(
            modifier = Modifier.padding(horizontal = 12.dp, vertical = 9.dp),
            verticalArrangement = Arrangement.spacedBy(2.dp),
        ) {
            Text(
                primary,
                style = MaterialTheme.typography.labelMedium,
                fontWeight = FontWeight.Medium,
            )
            Text(
                secondary,
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
    }
}

@Composable
private fun MusicProgressCard(
    state: MusicState.Generating,
    precision: String,
    runtimeLabel: String,
    onCancel: () -> Unit,
) {
    val elapsed by produceState(initialValue = 0L, state.startedAtMillis) {
        while (true) {
            value = (System.currentTimeMillis() - state.startedAtMillis).coerceAtLeast(0L) / 1000L
            delay(500)
        }
    }
    val context = androidx.compose.ui.platform.LocalContext.current
    val telemetry = rememberMusicDeviceTelemetry(context)

    ElevatedCard(
        shape = MaterialTheme.shapes.extraLarge,
        colors = CardDefaults.elevatedCardColors(
            containerColor = MaterialTheme.colorScheme.surfaceContainer,
        ),
    ) {
        Column(
            modifier = Modifier.padding(18.dp),
            verticalArrangement = Arrangement.spacedBy(14.dp),
        ) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        phaseLabel(state.phase),
                        style = MaterialTheme.typography.titleMedium,
                        fontWeight = FontWeight.SemiBold,
                    )
                    Text(
                        state.detail,
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
                OutlinedButton(onClick = onCancel) {
                    Icon(Icons.Default.Close, contentDescription = null)
                    Spacer(Modifier.width(5.dp))
                    Text("Cancel")
                }
            }

            MusicStageStrip(activePhase = state.phase, preload = false)

            MusicDeviceTelemetryCard(telemetry)

            if (state.phase == "decoding") {
                MusicDecodeProgress(state)
            } else if (state.progress != null) {
                SmoothLinearWavyProgressIndicator(
                    progress = state.progress,
                    modifier = Modifier.fillMaxWidth(),
                )
            } else {
                SmoothIndeterminateLinearWavyProgressIndicator(
                    modifier = Modifier.fillMaxWidth(),
                )
            }

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
            ) {
                Text(
                    if (state.total > 0) "${state.step}/${state.total}" else "Native stage",
                    style = MaterialTheme.typography.labelMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
                Text(
                    "Elapsed ${elapsed}s",
                    style = MaterialTheme.typography.labelMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }

            Text(
                "$runtimeLabel · CPU offload disabled · $precision · ${state.targetSeconds * 25} frames · 48 kHz stereo",
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
    }
}

@Composable
private fun MusicDecodeProgress(state: MusicState.Generating) {
    val completed = when {
        state.total > 0 -> (state.step.toFloat() / state.total).coerceIn(0f, 1f)
        state.progress != null -> ((state.progress - 0.90f) / 0.07f).coerceIn(0f, 1f)
        else -> 0f
    }
    val animated by animateFloatAsState(
        targetValue = completed,
        animationSpec = tween(durationMillis = 320),
        label = "vaeDecodeProgress",
    )
    val hasActiveTile = state.total > 0 && state.step < state.total
    val runningNativeGraph = state.total <= 0
    val activeTile = if (state.total > 0) (state.step + 1).coerceAtMost(state.total) else 0

    Column(verticalArrangement = Arrangement.spacedBy(7.dp)) {
        // A VAE tile is one native HTP graph invocation, so there is no honest
        // percentage inside a tile. Keep the line visibly moving while HTP is
        // executing it, then animate the real completed-tile progress at each
        // native milestone instead of inventing timer-based progress.
        if (state.total > 0) {
            SmoothLinearWavyProgressIndicator(
                progress = animated.coerceAtLeast(0.05f),
                modifier = Modifier.fillMaxWidth(),
            )
        } else {
            SmoothIndeterminateLinearWavyProgressIndicator(
                modifier = Modifier.fillMaxWidth(),
            )
        }
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
        ) {
            Text(
                when {
                    state.total > 0 && hasActiveTile ->
                        "HTP decoder · tile $activeTile/${state.total} running"
                    state.total > 0 ->
                        "HTP decoder · ${state.step}/${state.total} tiles complete"
                    else -> "VAE · HTP decode"
                },
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.primary,
            )
            Text(
                if (state.total > 0) {
                    "${(completed * 100f).toInt()}% complete"
                } else {
                    "Native graph"
                },
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun MusicGenerationConfigSheet(
    duration: Int,
    onDuration: (Int) -> Unit,
    planning: String,
    onPlanning: (String) -> Unit,
    steps: Int,
    onSteps: (Int) -> Unit,
    odeMethod: String,
    onOdeMethod: (String) -> Unit,
    seed: Long,
    onSeed: (Long) -> Unit,
    temperature: Float,
    onTemperature: (Float) -> Unit,
    topP: Float,
    onTopP: (Float) -> Unit,
    cfgScale: Float,
    onCfgScale: (Float) -> Unit,
    outputFormat: String,
    onOutputFormat: (String) -> Unit,
    onDismiss: () -> Unit,
) {
    var seedText by remember(seed) { mutableStateOf(if (seed < 0) "" else seed.toString()) }
    val context = LocalContext.current
    var currentTransport by remember {
        mutableStateOf(MusicTransportBenchmark.getPreferredTransport(context))
    }
    var benchmarkRunning by remember { mutableStateOf(false) }
    var benchmarkResult by remember { mutableStateOf<MusicTransportBenchmarkResult?>(null) }
    var benchmarkError by remember { mutableStateOf<String?>(null) }
    val coroutineScope = rememberCoroutineScope()

    ModalBottomSheet(onDismissRequest = onDismiss) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .verticalScroll(rememberScrollState()),
        ) {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 20.dp)
                    .padding(bottom = 30.dp),
                verticalArrangement = Arrangement.spacedBy(20.dp),
            ) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Surface(
                        shape = MaterialTheme.shapes.extraLarge,
                        color = MaterialTheme.colorScheme.primaryContainer,
                    ) {
                        Icon(
                            Icons.Default.Tune,
                            contentDescription = null,
                            modifier = Modifier.padding(12.dp),
                            tint = MaterialTheme.colorScheme.onPrimaryContainer,
                        )
                    }
                    Spacer(Modifier.width(12.dp))
                    Column {
                        Text(
                            "Music generation",
                            style = MaterialTheme.typography.headlineSmall,
                            fontWeight = FontWeight.SemiBold,
                        )
                        Text(
                            "YuE2 reference controls · 1 to 4 minute generation",
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                    }
                }

                SettingSlider(
                    title = "Duration",
                    valueText = "${duration}s (${duration / 60}m ${duration % 60}s)",
                    value = duration.toFloat(),
                    range = 10f..240f,
                    steps = 22,
                    onValue = { onDuration(it.toInt()) },
                    supporting = "60s (1 min) default, up to 240s (4 min). Full symbolic planning and exact NAR attention tiling structure the full song.",
                )

                Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    Text("Symbolic planning", style = MaterialTheme.typography.titleSmall)
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        listOf("full", "melody", "off").forEach { mode ->
                            FilterChip(
                                selected = planning == mode,
                                onClick = { onPlanning(mode) },
                                label = { Text(planningLabel(mode)) },
                            )
                        }
                    }
                    Text(
                        "Full plan is the YuE2 default for new songs and is recommended for strict instrumental prompts. Direct is kept as an expert score-free mode.",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }

                Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    Text("Acoustic solver", style = MaterialTheme.typography.titleSmall)
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        listOf(
                            "midpoint" to "Midpoint",
                            "dpmpp_2m" to "DPM++ 2M · SGM",
                        ).forEach { (method, label) ->
                            FilterChip(
                                selected = odeMethod == method,
                                onClick = { onOdeMethod(method) },
                                label = { Text(label) },
                            )
                        }
                    }
                    Text(
                        if (odeMethod == "midpoint") {
                            "YuE2 reference solver. Midpoint performs two full NAR model evaluations per step, matching official release quality."
                        } else {
                            "Exact 2nd-order Rectified Flow Multistep (Adams-Bashforth 2). Integrates the continuous velocity field in one HTP pass per step."
                        },
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }

                Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    Text("Render steps", style = MaterialTheme.typography.titleSmall)
                    val standardStepOptions = listOf(
                        6 to "6 · Speed",
                        8 to "8 · Fast",
                        16 to "16 · Fast Quality",
                        32 to "32 · Reference",
                    )
                    val stepOptions = if (standardStepOptions.none { it.first == steps }) {
                        standardStepOptions + (steps to "$steps · Active")
                    } else {
                        standardStepOptions
                    }
                    Row(
                        modifier = Modifier.horizontalScroll(rememberScrollState()),
                        horizontalArrangement = Arrangement.spacedBy(8.dp),
                    ) {
                        stepOptions.forEach { (option, label) ->
                            FilterChip(
                                selected = steps == option,
                                onClick = { onSteps(option) },
                                label = { Text(label) },
                            )
                        }
                    }
                    Text(
                        if (odeMethod == "midpoint") {
                            "32 is the YuE2 reference midpoint quality default (64 HTP evaluations). Fast presets remain available."
                        } else {
                            "16 is Fast Quality (Space preset); 32 is reference quality using exact 2nd-order Rectified Flow Multistep."
                        },
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }

                Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    Text("Audio output", style = MaterialTheme.typography.titleSmall)
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        listOf(
                            "wav16" to "WAV · lossless",
                            "wav32" to "WAV32 · debug",
                            "mp3" to "MP3 · 320k",
                        ).forEach { (format, label) ->
                            FilterChip(
                                selected = outputFormat == format,
                                onClick = { onOutputFormat(format) },
                                label = { Text(label) },
                            )
                        }
                    }
                    Text(
                        "WAV16 lossless is the default for optimal Android playback compatibility and zero loss. WAV32 preserves float output for parity checks.",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }

                Surface(
                    shape = MaterialTheme.shapes.large,
                    color = MaterialTheme.colorScheme.surfaceContainerHighest,
                ) {
                    Column(
                        modifier = Modifier.padding(14.dp),
                        verticalArrangement = Arrangement.spacedBy(8.dp),
                    ) {
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically,
                        ) {
                            Text(
                                "HTP Transport",
                                style = MaterialTheme.typography.titleSmall,
                                fontWeight = FontWeight.SemiBold,
                            )
                            Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                                FilterChip(
                                    selected = currentTransport == "dspqueue",
                                    onClick = {
                                        currentTransport = "dspqueue"
                                        MusicTransportBenchmark.setPreferredTransport(context, "dspqueue")
                                    },
                                    label = { Text("DSPQueue") },
                                )
                                FilterChip(
                                    selected = currentTransport == "fastrpc",
                                    onClick = {
                                        currentTransport = "fastrpc"
                                        MusicTransportBenchmark.setPreferredTransport(context, "fastrpc")
                                    },
                                    label = { Text("FastRPC") },
                                )
                            }
                        }
                        Text(
                            "Preferred: ${currentTransport.uppercase()}. FastRPC uses zero-copy mempool; DSPQueue queues command packets directly on HTP.",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )

                        OutlinedButton(
                            onClick = {
                                if (benchmarkRunning) return@OutlinedButton
                                benchmarkRunning = true
                                benchmarkError = null
                                coroutineScope.launch {
                                    runCatching { MusicTransportBenchmark.run(context) }
                                        .onSuccess { res ->
                                            benchmarkResult = res
                                            currentTransport = res.faster.lowercase()
                                            MusicTransportBenchmark.setPreferredTransport(context, currentTransport)
                                            benchmarkRunning = false
                                        }
                                        .onFailure { err ->
                                            benchmarkError = err.message ?: "Benchmark failed"
                                            benchmarkRunning = false
                                        }
                                }
                            },
                            enabled = !benchmarkRunning,
                            modifier = Modifier.fillMaxWidth(),
                        ) {
                            if (benchmarkRunning) {
                                CircularProgressIndicator(
                                    modifier = Modifier.size(16.dp),
                                    strokeWidth = 2.dp,
                                )
                                Spacer(Modifier.width(8.dp))
                                Text("Benchmarking DSPQueue vs FastRPC…")
                            } else {
                                Icon(Icons.Default.Speed, contentDescription = null)
                                Spacer(Modifier.width(8.dp))
                                Text("Run on-device A/B benchmark")
                            }
                        }

                        benchmarkResult?.let { res ->
                            Text(
                                "Winner: ${res.faster} (${String.format(java.util.Locale.US, "%.2f", res.speedup)}x faster) · DSPQueue ${String.format(java.util.Locale.US, "%.1f", res.dspQueue.medianMs)}ms vs FastRPC ${String.format(java.util.Locale.US, "%.1f", res.fastRpc.medianMs)}ms",
                                style = MaterialTheme.typography.labelMedium,
                                color = MaterialTheme.colorScheme.primary,
                            )
                        }
                        benchmarkError?.let { err ->
                            Text(
                                err,
                                style = MaterialTheme.typography.labelSmall,
                                color = MaterialTheme.colorScheme.error,
                            )
                        }
                    }
                }

                SettingSlider(
                    title = "Semantic creativity",
                    valueText = String.format(java.util.Locale.US, "%.2f", temperature),
                    value = temperature,
                    range = 0.5f..1.5f,
                    steps = 19,
                    onValue = onTemperature,
                    supporting = "YuE2 reference default 1.00. Lower is more conservative; higher explores more token choices.",
                )
                SettingSlider(
                    title = "Top-p",
                    valueText = String.format(java.util.Locale.US, "%.2f", topP),
                    value = topP,
                    range = 0.5f..1f,
                    steps = 9,
                    onValue = onTopP,
                    supporting = "Reference semantic default 0.95. Top-k stays at the official 100.",
                )
                Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    Text("Semantic CFG", style = MaterialTheme.typography.titleSmall)
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        listOf(
                            -1f to "Auto",
                            1.0f to "1.00",
                            1.01f to "1.01",
                        ).forEach { (value, label) ->
                            FilterChip(
                                selected = cfgScale == value,
                                onClick = { onCfgScale(value) },
                                label = { Text(label) },
                            )
                        }
                    }
                    Text(
                        "Auto uses 1.00 on mobile Direct for the fast single-branch path and 1.00 for planned modes. Pick 1.01 explicitly to reproduce reference Direct CFG.",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }

                OutlinedTextField(
                    value = seedText,
                    onValueChange = {
                        seedText = it.filter { ch -> ch.isDigit() }.take(18)
                        onSeed(seedText.toLongOrNull() ?: -1L)
                    },
                    label = { Text("Seed") },
                    placeholder = { Text("Random") },
                    supportingText = {
                        Text("Blank = random. One seed drives both the score/token draw and acoustic noise.")
                    },
                    modifier = Modifier.fillMaxWidth(),
                )

                Button(
                    onClick = onDismiss,
                    modifier = Modifier.fillMaxWidth(),
                    shape = MaterialTheme.shapes.large,
                ) {
                    Text("Done")
                }
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun MusicHistorySheet(
    items: List<MusicHistoryItem>,
    onHistoryChanged: () -> Unit,
    onDismiss: () -> Unit,
) {
    ModalBottomSheet(onDismissRequest = onDismiss) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .verticalScroll(rememberScrollState()),
        ) {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 20.dp)
                    .padding(bottom = 32.dp),
                verticalArrangement = Arrangement.spacedBy(16.dp),
            ) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Column {
                        Text(
                            "Music History",
                            style = MaterialTheme.typography.headlineSmall,
                            fontWeight = FontWeight.SemiBold,
                        )
                        Text(
                            "${items.size} track${if (items.size == 1) "" else "s"} saved on device",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                    }
                    IconButton(onClick = onDismiss) {
                        Icon(Icons.Default.Close, contentDescription = "Close")
                    }
                }

                if (items.isEmpty()) {
                    Box(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(vertical = 40.dp),
                        contentAlignment = Alignment.Center,
                    ) {
                        Text(
                            "No generated tracks yet",
                            style = MaterialTheme.typography.bodyMedium,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                    }
                } else {
                    items.forEach { item ->
                        ElevatedCard(
                            modifier = Modifier.fillMaxWidth(),
                            shape = MaterialTheme.shapes.large,
                        ) {
                            Column(
                                modifier = Modifier.padding(16.dp),
                                verticalArrangement = Arrangement.spacedBy(10.dp),
                            ) {
                                MusicPlayerCard(
                                    file = item.file,
                                    title = item.style.take(56).ifBlank { "YuE2 generation" },
                                    subtitle = "${item.targetSeconds}.0 s · ${outputFormatLabel(item.format)} · " +
                                        String.format(java.util.Locale.US, "%.1f s render", item.elapsedMillis / 1000.0),
                                )
                                MusicTrackActions(item, onHistoryChanged)
                            }
                        }
                    }
                }
            }
        }
    }
}


@Composable
private fun MusicTrackActions(
    item: MusicHistoryItem,
    onHistoryChanged: () -> Unit,
) {
    val context = androidx.compose.ui.platform.LocalContext.current
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        OutlinedButton(
            onClick = {
                runCatching { MusicHistoryStore.exportToMusic(context, item) }
                    .onSuccess {
                        Toast.makeText(context, "Saved to Music/Local Dream", Toast.LENGTH_SHORT).show()
                    }
                    .onFailure {
                        Toast.makeText(context, "Save failed: ${it.message}", Toast.LENGTH_LONG).show()
                    }
            },
            modifier = Modifier.weight(1f),
        ) { Text("Save") }
        OutlinedButton(
            onClick = {
                runCatching {
                    val uri = MusicHistoryStore.exportToMusic(context, item)
                    context.startActivity(
                        Intent.createChooser(
                            Intent(Intent.ACTION_SEND).apply {
                                type = MusicHistoryStore.mimeType(item.format)
                                putExtra(Intent.EXTRA_STREAM, uri)
                                addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
                            },
                            "Share track",
                        ),
                    )
                }.onFailure {
                    Toast.makeText(context, "Share failed: ${it.message}", Toast.LENGTH_LONG).show()
                }
            },
            modifier = Modifier.weight(1f),
        ) { Text("Share") }
        TextButton(
            onClick = {
                MusicHistoryStore.delete(context, item.id)
                onHistoryChanged()
            },
        ) { Text("Delete") }
    }
}

@Composable
private fun MusicHistorySection(
    items: List<MusicHistoryItem>,
    onHistoryChanged: () -> Unit,
) {
    if (items.isEmpty()) return
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Text(
            "Music history",
            style = MaterialTheme.typography.titleLarge,
            fontWeight = FontWeight.SemiBold,
        )
        Text(
            "Generated tracks stay on-device with their prompt, seeds and render settings.",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
        items.take(12).forEach { item ->
            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                MusicPlayerCard(
                    file = item.file,
                    title = item.style.take(56).ifBlank { "YuE2 generation" },
                    subtitle = "${item.targetSeconds}.0 s · ${outputFormatLabel(item.format)} · " +
                        String.format(java.util.Locale.US, "%.1f s render", item.elapsedMillis / 1000.0),
                )
                MusicTrackActions(item, onHistoryChanged)
            }
        }
    }
}

private fun outputFormatLabel(format: String): String = when (format) {
    "wav32" -> "WAV32 float"
    "wav24" -> "WAV24"
    "wav16" -> "WAV16 lossless"
    else -> "320 kbps MP3"
}

@Composable
private fun SettingSlider(
    title: String,
    valueText: String,
    value: Float,
    range: ClosedFloatingPointRange<Float>,
    steps: Int,
    onValue: (Float) -> Unit,
    supporting: String,
) {
    Column(verticalArrangement = Arrangement.spacedBy(5.dp)) {
        Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
            Text(title, style = MaterialTheme.typography.titleSmall)
            Text(
                valueText,
                style = MaterialTheme.typography.labelLarge,
                color = MaterialTheme.colorScheme.primary,
            )
        }
        Slider(
            value = value.coerceIn(range.start, range.endInclusive),
            onValueChange = onValue,
            valueRange = range,
            steps = steps,
        )
        Text(
            supporting,
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
    }
}

private fun qualityLabel(steps: Int, odeMethod: String): String = when {
    odeMethod == "midpoint" && steps >= 32 -> "Reference midpoint"
    odeMethod == "midpoint" -> "Midpoint · $steps"
    steps <= 6 -> "Max speed · DPM++"
    steps <= 8 -> "Fast · DPM++"
    steps <= 16 -> "Balanced · DPM++"
    else -> "Quality · DPM++"
}

private fun planningLabel(mode: String): String = when (mode) {
    "full" -> "Full plan"
    "melody" -> "Melody"
    else -> "Direct semantic"
}

private fun phaseLabel(phase: String): String = when (phase) {
    "queued" -> "Starting YuE2"
    "loading_ar" -> "Loading composer"
    "planning" -> "Planning the song"
    "semantic" -> "Writing semantic audio"
    "loading_nar" -> "Switching to renderer"
    "flow" -> "Rendering acoustics"
    "loading_vae" -> "Loading audio decoder"
    "decoding" -> "Decoding waveform"
    "finalizing" -> "Finishing track"
    else -> phase.replace('_', ' ').replaceFirstChar { it.uppercase() }
}
