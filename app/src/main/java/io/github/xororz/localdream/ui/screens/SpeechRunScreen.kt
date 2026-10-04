package io.github.xororz.localdream.ui.screens

import android.content.Intent
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.AutoAwesome
import androidx.compose.material.icons.filled.GraphicEq
import androidx.compose.material.icons.filled.History
import androidx.compose.material.icons.filled.Memory
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.Tune
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
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
    var seed by rememberSaveable { mutableLongStateOf(42L) }
    var cfg by rememberSaveable { mutableFloatStateOf(1f) }
    var temperature by rememberSaveable { mutableFloatStateOf(0.9f) }
    var topK by rememberSaveable { mutableIntStateOf(50) }
    var topP by rememberSaveable { mutableFloatStateOf(1f) }
    var repetition by rememberSaveable { mutableFloatStateOf(1.1f) }
    var splitChars by rememberSaveable { mutableIntStateOf(600) }
    var maxNewTokens by rememberSaveable { mutableIntStateOf(750) }
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

    LaunchedEffect(speechState) {
        when (speechState) {
            is SpeechState.Complete -> {
                history = withContext(Dispatchers.IO) { SpeechHistoryStore.load(context) }
                AppHaptics.perform(context, AppHaptics.Kind.Success)
            }
            is SpeechState.Error -> AppHaptics.perform(context, AppHaptics.Kind.Failure)
            else -> Unit
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
    val profile = model?.variantProfile.orEmpty()
    val statusText = when (val state = speechState) {
        is SpeechState.Loading -> state.detail
        is SpeechState.Ready -> precision + " · strict Hexagon HTP · ready"
        is SpeechState.Generating -> state.detail
        is SpeechState.Complete -> precision + " · strict Hexagon HTP · ready"
        is SpeechState.Error -> "HTP runtime failed · no fallback"
        SpeechState.Idle -> precision + " · strict Hexagon HTP"
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

                    OutlinedTextField(
                        value = text,
                        onValueChange = { text = it.take(12000) },
                        modifier = Modifier.fillMaxWidth(),
                        label = { Text("Text") },
                        placeholder = { Text("What should Breeze say?") },
                        minLines = 5,
                        supportingText = {
                            Text(text.length.toString() + " characters · English + Mandarin")
                        },
                    )

                    Text(
                        "Inline events",
                        style = MaterialTheme.typography.labelLarge,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                    LazyRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        items(events) { event ->
                            AssistChip(
                                onClick = {
                                    text = if (text.isBlank()) event else text + " " + event
                                },
                                label = { Text(event) },
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

                    Surface(
                        shape = MaterialTheme.shapes.large,
                        color = MaterialTheme.colorScheme.surfaceContainerHighest,
                    ) {
                        Row(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(14.dp),
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.spacedBy(12.dp),
                        ) {
                            Icon(Icons.Default.Memory, contentDescription = null)
                            Column(modifier = Modifier.weight(1f)) {
                                Text("Qualcomm HTP only", fontWeight = FontWeight.SemiBold)
                                Text(
                                    precision + " · " + profile + " · fallback disabled",
                                    style = MaterialTheme.typography.bodySmall,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                                )
                            }
                            Text(
                                "24 kHz",
                                style = MaterialTheme.typography.labelLarge,
                                color = MaterialTheme.colorScheme.primary,
                            )
                        }
                    }

                    if (speechState is SpeechState.Loading) {
                        LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
                    }

                    when (val state = speechState) {
                        is SpeechState.Generating -> {
                            Surface(
                                shape = MaterialTheme.shapes.large,
                                color = MaterialTheme.colorScheme.secondaryContainer,
                            ) {
                                Column(
                                    modifier = Modifier.padding(14.dp),
                                    verticalArrangement = Arrangement.spacedBy(6.dp),
                                ) {
                                    Text("Generating on HTP", fontWeight = FontWeight.SemiBold)
                                    Text(state.detail, style = MaterialTheme.typography.bodySmall)
                                    if (state.generatedSeconds > 0f) {
                                        Text(
                                            String.format(
                                                Locale.US,
                                                "%.1f s synthesized",
                                                state.generatedSeconds,
                                            ),
                                            style = MaterialTheme.typography.labelMedium,
                                        )
                                    }
                                }
                            }
                        }
                        is SpeechState.Error -> {
                            Surface(
                                shape = MaterialTheme.shapes.large,
                                color = MaterialTheme.colorScheme.errorContainer,
                            ) {
                                Column(
                                    modifier = Modifier.padding(14.dp),
                                    verticalArrangement = Arrangement.spacedBy(8.dp),
                                ) {
                                    Text(
                                        "Strict HTP runtime failed",
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
                                        Text("Retry HTP")
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
                Text(
                    "Breeze generation",
                    style = MaterialTheme.typography.headlineSmall,
                    fontWeight = FontWeight.Bold,
                )
                Text(
                    "Official GGUF defaults stay visible and editable. Backend execution remains strict HTP.",
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
                BreezeSettingSlider("CFG scale", cfg, 1f..4f, { cfg = it }) {
                    String.format(Locale.US, "%.2f", it)
                }
                BreezeSettingSlider(
                    "Temperature",
                    temperature,
                    0.3f..1.5f,
                    { temperature = it },
                ) { String.format(Locale.US, "%.2f", it) }
                BreezeSettingSlider("Top P", topP, 0.5f..1f, { topP = it }) {
                    String.format(Locale.US, "%.2f", it)
                }
                BreezeSettingSlider(
                    "Repetition penalty",
                    repetition,
                    1f..1.5f,
                    { repetition = it },
                ) { String.format(Locale.US, "%.2f", it) }
                BreezeSettingSlider("Top K", topK.toFloat(), 10f..100f, {
                    topK = it.roundToInt()
                }) { it.roundToInt().toString() }
                BreezeSettingSlider(
                    "Long-text split",
                    splitChars.toFloat(),
                    200f..1200f,
                    { splitChars = it.roundToInt() },
                ) { it.roundToInt().toString() + " chars" }
                BreezeSettingSlider(
                    "Frame cap / piece",
                    maxNewTokens.toFloat(),
                    250f..1500f,
                    { maxNewTokens = it.roundToInt() },
                ) { it.roundToInt().toString() }
                HorizontalDivider()
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Column(modifier = Modifier.weight(1f)) {
                        Text("Seed", fontWeight = FontWeight.SemiBold)
                        Text(
                            seed.toString(),
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                    }
                    TextButton(
                        onClick = {
                            seed = (System.nanoTime() and 0x7fffffff).coerceAtLeast(1L)
                        },
                    ) {
                        Text("Randomize")
                    }
                    TextButton(onClick = { seed = 42L }) { Text("42") }
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
private fun BreezeSettingSlider(
    label: String,
    value: Float,
    range: ClosedFloatingPointRange<Float>,
    onValueChange: (Float) -> Unit,
    valueLabel: (Float) -> String,
) {
    Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
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
        Slider(
            value = value.coerceIn(range.start, range.endInclusive),
            onValueChange = onValueChange,
            valueRange = range,
        )
    }
}
