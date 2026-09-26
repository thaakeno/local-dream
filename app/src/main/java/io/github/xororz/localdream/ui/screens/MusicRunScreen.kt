package io.github.xororz.localdream.ui.screens

import android.content.Context
import android.content.Intent
import androidx.compose.animation.AnimatedContent
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.togetherWith
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
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.AutoAwesome
import androidx.compose.material.icons.filled.Bolt
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.Tune
import androidx.compose.material3.Button
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.ElevatedCard
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilterChip
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
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.navigation.NavController
import io.github.xororz.localdream.data.ModelRepository
import io.github.xororz.localdream.navigation.popBackStackIfResumed
import io.github.xororz.localdream.service.BackendService
import io.github.xororz.localdream.service.MusicGenerationService
import io.github.xororz.localdream.service.MusicGenerationService.MusicState
import io.github.xororz.localdream.ui.components.MusicPlayerCard
import io.github.xororz.localdream.ui.components.SmoothIndeterminateLinearWavyProgressIndicator
import io.github.xororz.localdream.ui.components.SmoothLinearWavyProgressIndicator
import kotlinx.coroutines.delay

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
    val servingModelId by BackendService.servingModelId.collectAsState()
    val musicState by MusicGenerationService.state.collectAsState()

    var style by rememberSaveable {
        mutableStateOf("cinematic electronic pop, emotional female vocals, punchy drums, wide synths")
    }
    var lyrics by rememberSaveable {
        mutableStateOf("[Verse]\nNeon rain across the glass\nwe make the moment last\n\n[Chorus]\nTurn the signal into light")
    }
    var duration by rememberSaveable { mutableIntStateOf(20) }
    var planning by rememberSaveable { mutableStateOf("full") }
    var steps by rememberSaveable { mutableIntStateOf(32) }
    var seed by rememberSaveable { mutableLongStateOf(-1L) }
    var semanticTemperature by rememberSaveable { mutableFloatStateOf(1f) }
    var semanticTopP by rememberSaveable { mutableFloatStateOf(0.95f) }
    var cfgScale by rememberSaveable { mutableFloatStateOf(-1f) }
    var showConfig by remember { mutableStateOf(false) }
    var showScore by remember { mutableStateOf(false) }

    val backendReady = backendState is BackendService.BackendState.Running &&
        servingModelId == modelId

    LaunchedEffect(model?.id) {
        if (model == null || !model.isDownloaded || !model.isMusic) return@LaunchedEffect
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
                            if (backendReady) {
                                "HTP ready · ${model?.variantPrecision ?: "GGUF"}"
                            } else {
                                "Starting native HTP runtime…"
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
                    IconButton(onClick = { showConfig = true }) {
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
                                Icons.Default.AutoAwesome,
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
                                    "$duration s · ${qualityLabel(steps)} · ${planningLabel(planning)}",
                                    style = MaterialTheme.typography.titleSmall,
                                    fontWeight = FontWeight.SemiBold,
                                )
                                Text(
                                    "${model?.variantPrecision ?: "GGUF"} · 48 kHz stereo · 320 kbps MP3 · HTP primary",
                                    style = MaterialTheme.typography.labelMedium,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                                )
                            }
                            TextButton(onClick = { showConfig = true }) {
                                Text("Tune")
                            }
                        }
                    }

                    Button(
                        onClick = {
                            MusicGenerationService.reset()
                            context.startForegroundService(
                                Intent(context, MusicGenerationService::class.java)
                                    .setAction(MusicGenerationService.ACTION_GENERATE)
                                    .apply {
                                        putExtra("style", style)
                                        putExtra("lyrics", lyrics)
                                        putExtra("cot", planning)
                                        putExtra("duration", duration)
                                        putExtra("steps", steps)
                                        putExtra("seed", seed)
                                        putExtra("semantic_temperature", semanticTemperature)
                                        putExtra("semantic_top_p", semanticTopP)
                                        putExtra("cfg_scale", cfgScale)
                                    },
                            )
                        },
                        enabled = backendReady &&
                            style.isNotBlank() &&
                            musicState !is MusicState.Generating,
                        modifier = Modifier.fillMaxWidth(),
                        shape = MaterialTheme.shapes.large,
                    ) {
                        Icon(Icons.Default.PlayArrow, contentDescription = null)
                        Spacer(Modifier.width(8.dp))
                        Text(if (backendReady) "Generate music" else "Loading HTP runtime…")
                    }
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
                    is MusicState.Generating -> MusicProgressCard(
                        state = state,
                        precision = model?.variantPrecision ?: "GGUF",
                        onCancel = { MusicGenerationService.stop(context) },
                    )
                    is MusicState.Complete -> {
                        Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
                            MusicPlayerCard(
                                file = state.file,
                                title = "YuE2 generation",
                                subtitle = "48 kHz stereo · ${state.targetSeconds}s target · ${state.elapsedMillis / 1000f}s generated",
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
                            }
                        }
                    }
                    MusicState.Idle -> {
                        Surface(
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
                                    "Pipeline: AR score → semantic codes → NAR flow → Oobleck decode",
                                    style = MaterialTheme.typography.bodySmall,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                                )
                            }
                        }
                    }
                }
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
            seed = seed,
            onSeed = { seed = it },
            temperature = semanticTemperature,
            onTemperature = { semanticTemperature = it },
            topP = semanticTopP,
            onTopP = { semanticTopP = it },
            cfgScale = cfgScale,
            onCfgScale = { cfgScale = it },
            onDismiss = { showConfig = false },
        )
    }
}

@Composable
private fun MusicProgressCard(
    state: MusicState.Generating,
    precision: String,
    onCancel: () -> Unit,
) {
    val elapsed by produceState(initialValue = 0L, state.startedAtMillis) {
        while (true) {
            value = (System.currentTimeMillis() - state.startedAtMillis).coerceAtLeast(0L) / 1000L
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
                "HTP0 · $precision backbone · ${state.targetSeconds * 25} frame budget · 48 kHz stereo",
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
    seed: Long,
    onSeed: (Long) -> Unit,
    temperature: Float,
    onTemperature: (Float) -> Unit,
    topP: Float,
    onTopP: (Float) -> Unit,
    cfgScale: Float,
    onCfgScale: (Float) -> Unit,
    onDismiss: () -> Unit,
) {
    var seedText by remember(seed) { mutableStateOf(if (seed < 0) "" else seed.toString()) }

    ModalBottomSheet(onDismissRequest = onDismiss) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .verticalScroll(rememberScrollState())
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
                        "Reference YuE2 controls, capped to 20 seconds for mobile.",
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
            }

            SettingSlider(
                title = "Duration",
                valueText = "$duration seconds",
                value = duration.toFloat(),
                range = 5f..20f,
                steps = 14,
                onValue = { onDuration(it.toInt()) },
                supporting = "YuE2 emits 25 semantic frames per second. Maximum is intentionally 20s.",
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
                    "Full plans melody + chords before audio tokens. Melody keeps a lighter plan; Direct skips the score.",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }

            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Text("Render quality", style = MaterialTheme.typography.titleSmall)
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    listOf(8, 16, 32).forEach { option ->
                        FilterChip(
                            selected = steps == option,
                            onClick = { onSteps(option) },
                            label = {
                                Text(
                                    when (option) {
                                        8 -> "Fast · 8"
                                        16 -> "Balanced · 16"
                                        else -> "Reference · 32"
                                    },
                                )
                            },
                        )
                    }
                }
                Text(
                    "These are midpoint flow-matching steps. 32 is the yue2.cpp reference default; 16 is the mobile balance.",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
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
                    "Auto keeps YuE2's protocol default: 1.00 for full/melody planning and 1.01 for direct mode.",
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

private fun qualityLabel(steps: Int): String = when {
    steps <= 8 -> "Fast"
    steps >= 32 -> "Best"
    else -> "Balanced"
}

private fun planningLabel(mode: String): String = when (mode) {
    "full" -> "Full plan"
    "melody" -> "Melody"
    else -> "Direct"
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
