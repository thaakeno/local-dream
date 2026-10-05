package io.github.xororz.localdream.ui.screens

import android.content.Intent
import android.widget.Toast
import androidx.core.content.FileProvider
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.AutoAwesome
import androidx.compose.material.icons.filled.GraphicEq
import androidx.compose.material.icons.filled.History
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.Tune
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.drawBehind
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.text.TextLayoutResult
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.navigation.NavController
import io.github.xororz.localdream.data.ModelRepository
import io.github.xororz.localdream.navigation.popBackStackIfResumed
import io.github.xororz.localdream.service.SpeechGenerationService
import io.github.xororz.localdream.service.SpeechGenerationService.SpeechState
import io.github.xororz.localdream.service.SpeechHistoryItem
import io.github.xororz.localdream.service.SpeechHistoryStore
import io.github.xororz.localdream.ui.components.MusicPlayerCard
import io.github.xororz.localdream.ui.components.SmoothIndeterminateLinearWavyProgressIndicator
import io.github.xororz.localdream.ui.components.SmoothLinearWavyProgressIndicator
import io.github.xororz.localdream.utils.AppHaptics
import java.io.File
import java.util.Locale
import kotlin.math.roundToInt
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

private data class SpeechTemplate(
    val name: String,
    val text: String,
    val instruction: String,
)

private const val DEFAULT_SEED = 42L
private const val DEFAULT_CFG = 1f
private const val DEFAULT_TEMPERATURE = 0.9f
private const val DEFAULT_TOP_K = 50
private const val DEFAULT_TOP_P = 1f
private const val DEFAULT_REPETITION = 1.1f
private const val DEFAULT_SPLIT_CHARS = 600
private const val DEFAULT_MAX_NEW_TOKENS = 750

private fun shareSpeechFile(context: android.content.Context, file: File) {
    if (!file.isFile) return
    val uri = FileProvider.getUriForFile(
        context,
        context.packageName + ".fileprovider",
        file,
    )
    val send = Intent(Intent.ACTION_SEND).apply {
        type = "audio/wav"
        putExtra(Intent.EXTRA_STREAM, uri)
        addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        clipData = android.content.ClipData.newRawUri("Breeze TTS audio", uri)
    }
    context.startActivity(Intent.createChooser(send, "Share generated speech"))
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SpeechRunScreen(
    modelId: String,
    navController: NavController,
) {
    val context = androidx.compose.ui.platform.LocalContext.current
    val scope = rememberCoroutineScope()
    val repository = remember { ModelRepository.getInstance(context) }
    LaunchedEffect(Unit) { repository.ensureLoaded() }
    val model = repository.models.firstOrNull { it.id == modelId }
    val speechState by SpeechGenerationService.state.collectAsState()

    var text by rememberSaveable {
        mutableStateOf("Welcome aboard. Your journey begins now.")
    }
    var instruction by rememberSaveable {
        mutableStateOf("A warm, thoughtful young woman with a clear, calm delivery.")
    }
    var seed by rememberSaveable { mutableLongStateOf(DEFAULT_SEED) }
    var cfg by rememberSaveable { mutableFloatStateOf(DEFAULT_CFG) }
    var temperature by rememberSaveable { mutableFloatStateOf(DEFAULT_TEMPERATURE) }
    var topK by rememberSaveable { mutableIntStateOf(DEFAULT_TOP_K) }
    var topP by rememberSaveable { mutableFloatStateOf(DEFAULT_TOP_P) }
    var repetition by rememberSaveable { mutableFloatStateOf(DEFAULT_REPETITION) }
    var splitChars by rememberSaveable { mutableIntStateOf(DEFAULT_SPLIT_CHARS) }
    var maxNewTokens by rememberSaveable { mutableIntStateOf(DEFAULT_MAX_NEW_TOKENS) }
    var showTune by remember { mutableStateOf(false) }
    var showHistory by remember { mutableStateOf(false) }
    var history by remember { mutableStateOf<List<SpeechHistoryItem>>(emptyList()) }

    val templates = remember {
        listOf(
            SpeechTemplate(
                "Warm narrator",
                "The city was quiet before sunrise, and for a moment the whole world felt still.",
                "A warm, thoughtful young woman with a clear, calm delivery. Natural pacing, intimate studio sound.",
            ),
            SpeechTemplate(
                "Slow whisper",
                "(sigh) I knew you would come back. I just did not think it would take this long.",
                "A soft adult female voice, close-mic whisper, slow pacing, restrained emotion, breathy but intelligible.",
            ),
            SpeechTemplate(
                "Dramatic",
                "You had one chance to walk away. Now we finish what you started.",
                "A confident adult woman with cinematic intensity, controlled anger, deliberate pauses and strong emphasis.",
            ),
            SpeechTemplate(
                "Mandarin",
                "今天的风很轻，我们慢慢走，不用着急。",
                "自然的北京普通话女声，年轻成年，温柔清晰，语速稍慢，像真实对话。",
            ),
        )
    }
    val events = listOf(
        "(laugh)", "(sigh)", "(cough)", "(clears throat)",
        "[笑]", "[叹气]", "[咳嗽]", "[清嗓子]",
    )

    LaunchedEffect(modelId, model?.isDownloaded) {
        if (model?.isDownloaded == true && model.isVoice) {
            SpeechGenerationService.resetForModel(modelId)
            context.startForegroundService(
                Intent(context, SpeechGenerationService::class.java)
                    .setAction(SpeechGenerationService.ACTION_PRELOAD)
                    .putExtra("modelId", modelId),
            )
        }
        history = withContext(Dispatchers.IO) { SpeechHistoryStore.load(context) }
    }

    var lastSpeechProgress by remember { mutableFloatStateOf(-1f) }
    var lastTerminalHaptic by remember { mutableStateOf<String?>(null) }

    LaunchedEffect(speechState) {
        when (val state = speechState) {
            is SpeechState.Generating -> {
                state.progress?.let { progress ->
                    if (progress > lastSpeechProgress + 0.001f) {
                        AppHaptics.perform(context, AppHaptics.Kind.Progress)
                        lastSpeechProgress = progress
                    }
                }
                lastTerminalHaptic = null
            }
            is SpeechState.Complete -> {
                history = withContext(Dispatchers.IO) { SpeechHistoryStore.load(context) }
                if (lastTerminalHaptic != "complete") {
                    AppHaptics.perform(context, AppHaptics.Kind.Success)
                    lastTerminalHaptic = "complete"
                }
                lastSpeechProgress = -1f
            }
            is SpeechState.Error -> {
                if (lastTerminalHaptic != "error") {
                    AppHaptics.perform(context, AppHaptics.Kind.Failure)
                    lastTerminalHaptic = "error"
                }
                lastSpeechProgress = -1f
            }
            SpeechState.Idle, is SpeechState.Ready, is SpeechState.Loading -> {
                lastSpeechProgress = -1f
            }
        }
    }

    DisposableEffect(Unit) {
        onDispose {
            context.startService(
                Intent(context, SpeechGenerationService::class.java)
                    .setAction(SpeechGenerationService.ACTION_STOP),
            )
        }
    }

    val busy = speechState is SpeechState.Loading || speechState is SpeechState.Generating
    val precision = model?.variantPrecision.orEmpty()
    val statusText = when (val state = speechState) {
        is SpeechState.Loading -> state.detail
        is SpeechState.Ready -> "Ready"
        is SpeechState.Generating -> state.progress?.let {
            "Generating ${(it.coerceIn(0f, 1f) * 100f).roundToInt()}%"
        } ?: "Starting generation"
        is SpeechState.Complete -> "Ready"
        is SpeechState.Error -> "Generation failed"
        SpeechState.Idle -> if (model?.isDownloaded == true) "Ready" else "Model required"
    }

    Scaffold(
        topBar = {
            MediumTopAppBar(
                title = {
                    Column {
                        Text("Breeze TTS 2 · Text to speech")
                        Text(
                            statusText,
                            style = MaterialTheme.typography.bodySmall,
                            color = if (speechState is SpeechState.Error) {
                                MaterialTheme.colorScheme.error
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
                            showHistory = true
                            scope.launch {
                                history = withContext(Dispatchers.IO) {
                                    SpeechHistoryStore.load(context)
                                }
                            }
                        },
                    ) {
                        Icon(Icons.Default.History, contentDescription = "Speech history")
                    }
                    IconButton(onClick = { showTune = true }) {
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
                            color = MaterialTheme.colorScheme.primaryContainer,
                        ) {
                            Icon(
                                Icons.Default.GraphicEq,
                                contentDescription = null,
                                modifier = Modifier.padding(12.dp),
                                tint = MaterialTheme.colorScheme.onPrimaryContainer,
                            )
                        }
                        Column(modifier = Modifier.weight(1f)) {
                            Text(
                                "Design a voice",
                                style = MaterialTheme.typography.titleLarge,
                                fontWeight = FontWeight.SemiBold,
                            )
                            Text(
                                "Text plus voice direction. Voice cloning is intentionally disabled for now.",
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant,
                            )
                        }
                    }

                    LazyRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        items(templates, key = { it.name }) { template ->
                            AssistChip(
                                onClick = {
                                    text = template.text
                                    instruction = template.instruction
                                },
                                label = { Text(template.name) },
                                leadingIcon = {
                                    Icon(Icons.Default.AutoAwesome, contentDescription = null)
                                },
                            )
                        }
                    }

                    InlineEventEditor(
                        value = text,
                        onValueChange = { text = it.take(12000) },
                        events = events,
                        modifier = Modifier.fillMaxWidth(),
                    )

                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        Text(
                            "Inline events",
                            style = MaterialTheme.typography.labelLarge,
                            modifier = Modifier.weight(1f),
                        )
                        Text(
                            "Tap to insert",
                            style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                    }
                    LazyRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        items(events) { event ->
                            val chinese = event.startsWith("[")
                            AssistChip(
                                onClick = {
                                    text = if (text.isBlank()) event else text + " " + event
                                    AppHaptics.perform(context, AppHaptics.Kind.Interaction)
                                },
                                label = { Text(event) },
                                colors = AssistChipDefaults.assistChipColors(
                                    containerColor = if (chinese) {
                                        MaterialTheme.colorScheme.secondaryContainer
                                    } else {
                                        MaterialTheme.colorScheme.tertiaryContainer
                                    },
                                    labelColor = if (chinese) {
                                        MaterialTheme.colorScheme.onSecondaryContainer
                                    } else {
                                        MaterialTheme.colorScheme.onTertiaryContainer
                                    },
                                ),
                            )
                        }
                    }

                    OutlinedTextField(
                        value = instruction,
                        onValueChange = { instruction = it.take(1200) },
                        modifier = Modifier.fillMaxWidth(),
                        label = { Text("Voice direction") },
                        placeholder = {
                            Text("Voice, age, tone, pacing, emotion, recording style…")
                        },
                        minLines = 3,
                    )

                    when (val state = speechState) {
                        is SpeechState.Loading -> SpeechLoadingCard(state.detail)
                        is SpeechState.Generating -> SpeechProgressCard(state)
                        is SpeechState.Error -> {
                            Surface(
                                shape = MaterialTheme.shapes.extraLarge,
                                color = MaterialTheme.colorScheme.errorContainer,
                            ) {
                                Column(
                                    modifier = Modifier.padding(16.dp),
                                    verticalArrangement = Arrangement.spacedBy(10.dp),
                                ) {
                                    Text(
                                        "Generation failed",
                                        style = MaterialTheme.typography.titleMedium,
                                        fontWeight = FontWeight.SemiBold,
                                    )
                                    Text(state.message, style = MaterialTheme.typography.bodySmall)
                                    OutlinedButton(
                                        onClick = {
                                            context.startForegroundService(
                                                Intent(
                                                    context,
                                                    SpeechGenerationService::class.java,
                                                )
                                                    .setAction(
                                                        SpeechGenerationService.ACTION_PRELOAD,
                                                    )
                                                    .putExtra("modelId", modelId),
                                            )
                                        },
                                    ) {
                                        Icon(Icons.Default.Refresh, contentDescription = null)
                                        Spacer(Modifier.width(6.dp))
                                        Text("Restart")
                                    }
                                }
                            }
                        }
                        else -> Unit
                    }

                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(10.dp),
                    ) {
                        OutlinedButton(
                            onClick = {
                                if (speechState is SpeechState.Generating) {
                                    context.startService(
                                        Intent(context, SpeechGenerationService::class.java)
                                            .setAction(SpeechGenerationService.ACTION_STOP),
                                    )
                                } else {
                                    showTune = true
                                }
                            },
                            modifier = Modifier.weight(1f),
                        ) {
                            Icon(Icons.Default.Tune, contentDescription = null)
                            Spacer(Modifier.width(6.dp))
                            Text(if (speechState is SpeechState.Generating) "Stop" else "Tune")
                        }

                        Button(
                            onClick = {
                                AppHaptics.perform(context, AppHaptics.Kind.Interaction)
                                context.startForegroundService(
                                    Intent(context, SpeechGenerationService::class.java)
                                        .setAction(SpeechGenerationService.ACTION_GENERATE)
                                        .putExtra("modelId", modelId)
                                        .putExtra("text", text)
                                        .putExtra("instruction", instruction)
                                        .putExtra("seed", seed)
                                        .putExtra("cfg", cfg)
                                        .putExtra("temperature", temperature)
                                        .putExtra("topK", topK)
                                        .putExtra("topP", topP)
                                        .putExtra("repetition", repetition)
                                        .putExtra("splitChars", splitChars)
                                        .putExtra("maxNewTokens", maxNewTokens),
                                )
                            },
                            enabled = !busy && text.isNotBlank() && model?.isDownloaded == true,
                            modifier = Modifier.weight(1f),
                        ) {
                            Icon(Icons.Default.GraphicEq, contentDescription = null)
                            Spacer(Modifier.width(6.dp))
                            Text("Generate")
                        }
                    }
                }
            }

            val complete = speechState as? SpeechState.Complete
            if (complete != null && complete.file.isFile) {
                MusicPlayerCard(
                    file = complete.file,
                    title = "Breeze TTS 2 · " + precision,
                    subtitle = "Seed " + complete.seed + " · " +
                        String.format(Locale.US, "%.1f s generation", complete.elapsedMillis / 1000f),
                    modifier = Modifier.padding(horizontal = 16.dp),
                    onSave = {
                        scope.launch {
                            val saved = runCatching {
                                SpeechHistoryStore.exportToMusic(context, complete.file)
                            }.getOrNull()
                            Toast.makeText(
                                context,
                                if (saved != null) {
                                    "Saved to Music/LocalDream"
                                } else {
                                    "Could not save audio"
                                },
                                Toast.LENGTH_SHORT,
                            ).show()
                        }
                    },
                    onShare = { shareSpeechFile(context, complete.file) },
                )
            }

            Spacer(Modifier.height(4.dp))
        }
    }

    if (showTune) {
        ModalBottomSheet(onDismissRequest = { showTune = false }) {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .verticalScroll(rememberScrollState())
                    .padding(horizontal = 18.dp)
                    .padding(bottom = 28.dp),
                verticalArrangement = Arrangement.spacedBy(16.dp),
            ) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(12.dp),
                ) {
                    Column(modifier = Modifier.weight(1f)) {
                        Text(
                            "Breeze generation",
                            style = MaterialTheme.typography.headlineSmall,
                            fontWeight = FontWeight.Bold,
                        )
                        Text(
                            "Voice sampling, consistency and long-text controls.",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                    }
                    OutlinedButton(
                        onClick = {
                            cfg = DEFAULT_CFG
                            temperature = DEFAULT_TEMPERATURE
                            topK = DEFAULT_TOP_K
                            topP = DEFAULT_TOP_P
                            repetition = DEFAULT_REPETITION
                            splitChars = DEFAULT_SPLIT_CHARS
                            maxNewTokens = DEFAULT_MAX_NEW_TOKENS
                            seed = DEFAULT_SEED
                            AppHaptics.perform(context, AppHaptics.Kind.Interaction)
                        },
                    ) {
                        Icon(Icons.Default.Refresh, contentDescription = null)
                        Spacer(Modifier.width(6.dp))
                        Text("Reset defaults")
                    }
                }

                Surface(
                    shape = MaterialTheme.shapes.large,
                    color = MaterialTheme.colorScheme.surfaceContainerHigh,
                ) {
                    Text(
                        "The defaults are the safe baseline. Change one thing at a time when tuning a voice so you can hear what actually helped.",
                        modifier = Modifier.padding(14.dp),
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }

                BreezeSettingSlider(
                    label = "CFG scale",
                    value = cfg,
                    range = 1f..4f,
                    description = "How strongly Breeze follows the voice direction. 1.0 is the natural default; higher values push the requested style harder but can make speech sound forced.",
                    onValueChange = { cfg = it },
                ) { String.format(Locale.US, "%.2f", it) }

                BreezeSettingSlider(
                    label = "Temperature",
                    value = temperature,
                    range = 0.3f..1.5f,
                    description = "Controls randomness and expressiveness. Lower is steadier and more repeatable; higher gives more variation and emotion but can increase odd pronunciations.",
                    onValueChange = { temperature = it },
                ) { String.format(Locale.US, "%.2f", it) }

                BreezeSettingSlider(
                    label = "Top P",
                    value = topP,
                    range = 0.5f..1f,
                    description = "Nucleus sampling. Lower values keep only the most likely choices and sound safer; 1.0 keeps the full candidate distribution.",
                    onValueChange = { topP = it },
                ) { String.format(Locale.US, "%.2f", it) }

                BreezeSettingSlider(
                    label = "Repetition penalty",
                    value = repetition,
                    range = 1f..1.5f,
                    description = "Discourages repeated sounds, syllables and phrases. Increase it only if Breeze gets stuck repeating; too high can damage fluency.",
                    onValueChange = { repetition = it },
                ) { String.format(Locale.US, "%.2f", it) }

                BreezeSettingSlider(
                    label = "Top K",
                    value = topK.toFloat(),
                    range = 10f..100f,
                    description = "Maximum token choices considered at each step. Lower is more predictable; higher allows more varied delivery. 50 is a good general default.",
                    onValueChange = { topK = it.roundToInt() },
                ) { it.roundToInt().toString() }

                BreezeSettingSlider(
                    label = "Long-text split",
                    value = splitChars.toFloat(),
                    range = 200f..1200f,
                    description = "Approximate characters per speech segment. Smaller chunks start sooner and use less memory; larger chunks preserve continuity but take longer before audio arrives.",
                    onValueChange = { splitChars = it.roundToInt() },
                ) { it.roundToInt().toString() + " chars" }

                BreezeSettingSlider(
                    label = "Frame cap / piece",
                    value = maxNewTokens.toFloat(),
                    range = 250f..1500f,
                    description = "Maximum acoustic frames Breeze may generate for each segment. Raise it for long slow passages; setting it too low can cut a segment off.",
                    onValueChange = { maxNewTokens = it.roundToInt() },
                ) { it.roundToInt().toString() }

                HorizontalDivider()

                Row(
                    modifier = Modifier.fillMaxWidth(),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                ) {
                    Column(modifier = Modifier.weight(1f)) {
                        Text("Seed", fontWeight = FontWeight.SemiBold)
                        Text(
                            "Controls repeatability. Reuse the same seed with the same text/settings to get a similar result.",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                        Text(
                            seed.toString(),
                            style = MaterialTheme.typography.labelLarge,
                            color = MaterialTheme.colorScheme.primary,
                        )
                    }
                    TextButton(
                        onClick = {
                            seed = (System.nanoTime() and 0x7fffffff).coerceAtLeast(1L)
                        },
                    ) {
                        Text("Randomize")
                    }
                    TextButton(onClick = { seed = DEFAULT_SEED }) {
                        Text("42")
                    }
                }

                Button(
                    onClick = { showTune = false },
                    modifier = Modifier.fillMaxWidth(),
                ) {
                    Text("Done")
                }
            }
        }
    }

    if (showHistory) {
        ModalBottomSheet(onDismissRequest = { showHistory = false }) {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 16.dp)
                    .padding(bottom = 24.dp),
                verticalArrangement = Arrangement.spacedBy(12.dp),
            ) {
                Text(
                    "Speech history",
                    style = MaterialTheme.typography.headlineSmall,
                    fontWeight = FontWeight.Bold,
                )
                Text(
                    "Generated WAV files stay local on this device.",
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
                LazyColumn(
                    modifier = Modifier
                        .fillMaxWidth()
                        .heightIn(max = 560.dp),
                    verticalArrangement = Arrangement.spacedBy(10.dp),
                ) {
                    if (history.isEmpty()) {
                        item {
                            Text(
                                "No speech generations yet.",
                                modifier = Modifier.padding(vertical = 24.dp),
                            )
                        }
                    }
                    items(history, key = { it.id }) { item ->
                        val file = File(item.filePath)
                        if (file.isFile) {
                            Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                                MusicPlayerCard(
                                    file = file,
                                    title = item.text.take(64),
                                    subtitle = item.instruction.take(90),
                                    onSave = {
                                        scope.launch {
                                            val saved = runCatching {
                                                SpeechHistoryStore.exportToMusic(context, file)
                                            }.getOrNull()
                                            Toast.makeText(
                                                context,
                                                if (saved != null) {
                                                    "Saved to Music/LocalDream"
                                                } else {
                                                    "Could not save audio"
                                                },
                                                Toast.LENGTH_SHORT,
                                            ).show()
                                        }
                                    },
                                    onShare = { shareSpeechFile(context, file) },
                                )
                                TextButton(
                                    onClick = {
                                        scope.launch {
                                            SpeechHistoryStore.delete(context, item.id)
                                            history = withContext(Dispatchers.IO) {
                                                SpeechHistoryStore.load(context)
                                            }
                                        }
                                    },
                                ) {
                                    Text("Delete")
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}


@Composable
private fun InlineEventEditor(
    value: String,
    onValueChange: (String) -> Unit,
    events: List<String>,
    modifier: Modifier = Modifier,
) {
    var textLayout by remember(value) { mutableStateOf<TextLayoutResult?>(null) }
    val textColor = MaterialTheme.colorScheme.onSurface
    val placeholderColor = MaterialTheme.colorScheme.onSurfaceVariant
    val borderColor = MaterialTheme.colorScheme.outline
    val latinEventColor = MaterialTheme.colorScheme.tertiaryContainer
    val chineseEventColor = MaterialTheme.colorScheme.secondaryContainer
    val radius = 7.dp
    val horizontalPad = 4.dp
    val verticalPad = 2.dp

    Column(
        modifier = modifier,
        verticalArrangement = Arrangement.spacedBy(7.dp),
    ) {
        Text(
            "Text",
            style = MaterialTheme.typography.labelLarge,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
        Surface(
            shape = RoundedCornerShape(14.dp),
            color = MaterialTheme.colorScheme.surfaceContainerLow,
            border = androidx.compose.foundation.BorderStroke(1.dp, borderColor),
        ) {
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .heightIn(min = 170.dp)
                    .padding(16.dp),
            ) {
                if (value.isEmpty()) {
                    Text(
                        "What should Breeze say?",
                        color = placeholderColor,
                        style = MaterialTheme.typography.bodyLarge,
                    )
                }
                BasicTextField(
                    value = value,
                    onValueChange = onValueChange,
                    modifier = Modifier
                        .fillMaxWidth()
                        .heightIn(min = 138.dp)
                        .drawBehind {
                            val layout = textLayout ?: return@drawBehind
                            events.forEach { event ->
                                var searchFrom = 0
                                while (searchFrom < value.length) {
                                    val start = value.indexOf(event, searchFrom)
                                    if (start < 0) break
                                    val end = (start + event.length).coerceAtMost(value.length)
                                    var activeLine = -1
                                    var left = Float.POSITIVE_INFINITY
                                    var top = Float.POSITIVE_INFINITY
                                    var right = Float.NEGATIVE_INFINITY
                                    var bottom = Float.NEGATIVE_INFINITY

                                    fun drawSegment() {
                                        if (activeLine < 0 || !left.isFinite()) return
                                        val hp = horizontalPad.toPx()
                                        val vp = verticalPad.toPx()
                                        drawRoundRect(
                                            color = if (event.startsWith("[")) {
                                                chineseEventColor
                                            } else {
                                                latinEventColor
                                            },
                                            topLeft = Offset(left - hp, top - vp),
                                            size = Size(
                                                (right - left) + hp * 2f,
                                                (bottom - top) + vp * 2f,
                                            ),
                                            cornerRadius = CornerRadius(
                                                radius.toPx(),
                                                radius.toPx(),
                                            ),
                                        )
                                    }

                                    for (offset in start until end) {
                                        val line = layout.getLineForOffset(offset)
                                        val box = layout.getBoundingBox(offset)
                                        if (activeLine != -1 && line != activeLine) {
                                            drawSegment()
                                            left = Float.POSITIVE_INFINITY
                                            top = Float.POSITIVE_INFINITY
                                            right = Float.NEGATIVE_INFINITY
                                            bottom = Float.NEGATIVE_INFINITY
                                        }
                                        activeLine = line
                                        left = minOf(left, box.left)
                                        top = minOf(top, box.top)
                                        right = maxOf(right, box.right)
                                        bottom = maxOf(bottom, box.bottom)
                                    }
                                    drawSegment()
                                    searchFrom = end.coerceAtLeast(start + 1)
                                }
                            }
                        },
                    textStyle = MaterialTheme.typography.bodyLarge.copy(color = textColor),
                    cursorBrush = SolidColor(MaterialTheme.colorScheme.primary),
                    onTextLayout = { textLayout = it },
                )
            }
        }
        Text(
            value.length.toString() + " characters · English + Mandarin",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
    }
}

@Composable
private fun SpeechLoadingCard(detail: String) {
    Surface(
        shape = MaterialTheme.shapes.extraLarge,
        color = MaterialTheme.colorScheme.surfaceContainerHigh,
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp),
        ) {
            Text(
                detail,
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.SemiBold,
            )
            Text(
                "The model stays warm after loading, so the next generation starts faster.",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            SmoothIndeterminateLinearWavyProgressIndicator(
                modifier = Modifier.fillMaxWidth(),
            )
        }
    }
}

@Composable
private fun SpeechProgressCard(state: SpeechState.Generating) {
    val progress = state.progress?.coerceIn(0f, 1f)
    val percent = progress?.let { (it * 100f).roundToInt() }
    val elapsedLive by produceState(
        initialValue = 0f,
        key1 = state.startedAtMillis,
        key2 = state.elapsedSeconds,
    ) {
        while (true) {
            val wallElapsed =
                (System.currentTimeMillis() - state.startedAtMillis).coerceAtLeast(0L) / 1000f
            value = maxOf(state.elapsedSeconds ?: 0f, wallElapsed)
            kotlinx.coroutines.delay(500L)
        }
    }
    val waitingForEos = state.detail == "Waiting for end-of-speech"
    val adjustedEta = state.etaSeconds?.let { nativeEta ->
        val nativeElapsed = state.elapsedSeconds ?: elapsedLive
        maxOf(0f, nativeEta - maxOf(0f, elapsedLive - nativeElapsed))
    }

    Surface(
        shape = MaterialTheme.shapes.extraLarge,
        color = MaterialTheme.colorScheme.secondaryContainer.copy(alpha = 0.72f),
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(13.dp),
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        "Generating speech",
                        style = MaterialTheme.typography.titleMedium,
                        fontWeight = FontWeight.SemiBold,
                    )
                    Text(
                        state.detail,
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSecondaryContainer.copy(alpha = 0.72f),
                    )
                }
                if (percent != null) {
                    Text(
                        "$percent%",
                        style = MaterialTheme.typography.headlineSmall,
                        fontWeight = FontWeight.Bold,
                    )
                }
            }

            if (progress != null) {
                SmoothLinearWavyProgressIndicator(
                    progress = progress,
                    modifier = Modifier.fillMaxWidth(),
                )
            } else {
                SmoothIndeterminateLinearWavyProgressIndicator(
                    modifier = Modifier.fillMaxWidth(),
                )
            }

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                SpeechMetric(
                    label = "Audio",
                    value = buildString {
                        append(String.format(Locale.US, "%.1f s", state.generatedSeconds))
                        state.estimatedSeconds?.let {
                            append(" / ")
                            append(String.format(Locale.US, "%.1f s", it))
                        }
                    },
                    modifier = Modifier.weight(1f),
                )
                SpeechMetric(
                    label = "Elapsed",
                    value = formatSpeechTime(elapsedLive),
                    modifier = Modifier.weight(1f),
                )
            }
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                SpeechMetric(
                    label = "ETA",
                    value = when {
                        waitingForEos -> "Waiting for EOS"
                        adjustedEta != null -> formatSpeechTime(adjustedEta)
                        else -> "Calculating"
                    },
                    modifier = Modifier.weight(1f),
                )
                SpeechMetric(
                    label = "Speed",
                    value = when {
                        state.realtimeFactor != null && state.fps != null ->
                            String.format(
                                Locale.US,
                                "%.2f× · %.1f fps",
                                state.realtimeFactor,
                                state.fps,
                            )
                        state.realtimeFactor != null ->
                            String.format(Locale.US, "%.2f× realtime", state.realtimeFactor)
                        else -> "Warming up"
                    },
                    modifier = Modifier.weight(1f),
                )
            }

            Text(
                if (waitingForEos) {
                    "The spoken-length estimate has been reached. Breeze is still decoding until the model emits end-of-speech."
                } else {
                    "Progress and ETA are native spoken-length estimates; the model itself decides the exact end."
                },
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSecondaryContainer.copy(alpha = 0.65f),
            )
        }
    }
}

@Composable
private fun SpeechMetric(
    label: String,
    value: String,
    modifier: Modifier = Modifier,
) {
    Surface(
        modifier = modifier,
        shape = RoundedCornerShape(12.dp),
        color = MaterialTheme.colorScheme.surface.copy(alpha = 0.38f),
    ) {
        Column(
            modifier = Modifier.padding(horizontal = 12.dp, vertical = 9.dp),
            verticalArrangement = Arrangement.spacedBy(2.dp),
        ) {
            Text(
                label,
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            Text(
                value,
                style = MaterialTheme.typography.labelLarge,
                fontWeight = FontWeight.SemiBold,
            )
        }
    }
}

private fun formatSpeechTime(seconds: Float): String {
    val total = seconds.coerceAtLeast(0f).roundToInt()
    val minutes = total / 60
    val remain = total % 60
    return if (minutes > 0) {
        String.format(Locale.US, "%d:%02d", minutes, remain)
    } else {
        "${remain}s"
    }
}

@Composable
private fun BreezeSettingSlider(
    label: String,
    value: Float,
    range: ClosedFloatingPointRange<Float>,
    description: String,
    onValueChange: (Float) -> Unit,
    valueLabel: (Float) -> String,
) {
    Column(verticalArrangement = Arrangement.spacedBy(5.dp)) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
        ) {
            Text(label, fontWeight = FontWeight.SemiBold)
            Text(
                valueLabel(value),
                color = MaterialTheme.colorScheme.primary,
                style = MaterialTheme.typography.labelLarge,
            )
        }
        Text(
            description,
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
        Slider(
            value = value.coerceIn(range.start, range.endInclusive),
            onValueChange = onValueChange,
            valueRange = range,
        )
    }
}
