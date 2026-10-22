package io.github.xororz.localdream.ui.components


import android.content.Intent
import androidx.compose.animation.AnimatedContent
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.animateContentSize
import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.togetherWith
import androidx.compose.foundation.Image
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.ExperimentalFoundationApi
import androidx.compose.foundation.background
import androidx.compose.foundation.combinedClickable
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.AutoAwesome
import androidx.compose.material.icons.filled.Bolt
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.CloudDownload
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.GraphicEq
import androidx.compose.material.icons.filled.Memory
import androidx.compose.material.icons.filled.MusicNote
import androidx.compose.material.icons.filled.Pause
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.Search
import androidx.compose.material.icons.filled.SdStorage
import androidx.compose.material.icons.filled.Sort
import androidx.compose.material.icons.filled.Speed
import androidx.compose.material.icons.filled.Tune
import androidx.compose.material3.AssistChip
import androidx.compose.material3.Badge
import androidx.compose.material3.Button
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ElevatedCard
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilterChip
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.ListItem
import androidx.compose.material3.ListItemDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Surface
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.VerticalDivider
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.res.painterResource
import io.github.xororz.localdream.R
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import io.github.xororz.localdream.data.Model
import java.io.File
import io.github.xororz.localdream.utils.AppHaptics
import java.util.Locale
import kotlinx.coroutines.launch



/** Locally bundled, actual project marks. No network or glass shaders. */
@Composable
private fun StudioBrandMark(kind: String, modifier: Modifier = Modifier) {
    val logo = when (kind) {
        "voice" -> R.drawable.ic_model_breeze
        "music" -> R.drawable.ic_model_yue2
        else -> R.drawable.ic_model_qwen_transparent
    }
    val capsule = RoundedCornerShape(16.dp)
    Box(
        modifier = modifier
            .then(if (kind == "image") Modifier.size(width = 90.dp, height = 60.dp)
                else Modifier.size(60.dp))
            .background(Color.White.copy(alpha = 0.09f), capsule),
        contentAlignment = Alignment.Center,
    ) {
        Image(
            painter = painterResource(logo),
            contentDescription = when (kind) {
                "voice" -> "BreezeBlue logo"
                "music" -> "YuE logo"
                else -> "Qwen Image official logo"
            },
            contentScale = androidx.compose.ui.layout.ContentScale.Fit,
            modifier = Modifier.fillMaxSize().padding(if (kind == "image") 3.dp else 9.dp),
        )
    }
}

private data class StudioColors(val start: Color, val middle: Color, val end: Color)

private fun studioColors(kind: String): StudioColors = when (kind) {
    "voice" -> StudioColors(Color(0xFF083C49), Color(0xFF126C7A), Color(0xFF176E87))
    "music" -> StudioColors(Color(0xFF321243), Color(0xFF6B2E78), Color(0xFF86477E))
    else -> StudioColors(Color(0xFF111D54), Color(0xFF1D4799), Color(0xFF593B9C))
}

@Composable
private fun StudioGradientHeader(
    kind: String,
    title: String,
    subtitle: String,
    modifier: Modifier = Modifier,
) {
    val colors = studioColors(kind)
    val transition = rememberInfiniteTransition(label = "family-gradient-$kind")
    val flow by transition.animateFloat(
        initialValue = 0f,
        targetValue = 1f,
        animationSpec = infiniteRepeatable(
            animation = tween(12000, easing = LinearEasing),
            repeatMode = RepeatMode.Reverse,
        ),
        label = "family-glow-$kind",
    )
    // Slow, low-amplitude color movement: no blur, RenderEffect or offscreen layers.
    val brush = Brush.linearGradient(
        listOf(
            androidx.compose.ui.graphics.lerp(colors.start, colors.middle, flow * 0.13f),
            androidx.compose.ui.graphics.lerp(colors.middle, colors.end, flow * 0.20f),
            androidx.compose.ui.graphics.lerp(colors.end, colors.middle, flow * 0.11f),
        ),
    )
    Box(modifier = modifier.fillMaxWidth().background(brush).padding(18.dp)) {
        Row(verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(13.dp)) {
            StudioBrandMark(kind)
            Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(5.dp)) {
                Text(title, style = MaterialTheme.typography.titleLarge,
                    fontWeight = FontWeight.Bold, color = Color.White, maxLines = 2)
                Text(subtitle, style = MaterialTheme.typography.bodySmall,
                    color = Color.White.copy(alpha = .82f), maxLines = 2)
            }
        }
    }
}

@Composable
private fun StudioFamilyCard(
    title: String,
    subtitle: String,
    kind: String,
    variants: List<Model>,
    featured: Model,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
) {
    val installed = variants.count { it.isDownloaded }
    val officialQwenTurbo = kind == "image" &&
        variants.any { it.id == "qwen_image_2_1_turbo" }
    Surface(
        onClick = onClick,
        modifier = modifier.fillMaxWidth(),
        shape = RoundedCornerShape(25.dp),
        color = MaterialTheme.colorScheme.surfaceContainerLow,
        border = BorderStroke(1.dp, MaterialTheme.colorScheme.outlineVariant.copy(alpha = .55f)),
    ) {
        Column {
            StudioGradientHeader(kind, title, subtitle)
            Row(
                modifier = Modifier.fillMaxWidth()
                    .padding(horizontal = 18.dp, vertical = 15.dp),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                Surface(
                    color = MaterialTheme.colorScheme.primaryContainer,
                    shape = RoundedCornerShape(9.dp),
                ) {
                    Text(
                        if (officialQwenTurbo) "Standard + Turbo"
                        else featured.variantPrecision.ifBlank { "Models" },
                        modifier = Modifier.padding(horizontal = 10.dp, vertical = 7.dp),
                        color = MaterialTheme.colorScheme.onPrimaryContainer,
                        fontWeight = FontWeight.SemiBold,
                        style = MaterialTheme.typography.labelMedium,
                    )
                }
                Text(
                    if (installed > 0) "$installed installed" else "${variants.size} options",
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
                Spacer(Modifier.weight(1f))
                Text("Explore", fontWeight = FontWeight.SemiBold,
                    style = MaterialTheme.typography.labelMedium)
                Icon(Icons.Default.Tune, contentDescription = "Configure",
                    modifier = Modifier.size(19.dp),
                    tint = MaterialTheme.colorScheme.primary)
            }
        }
    }
}

enum class CatalogSortMode(val label: String) {
    Smart("Smart"),
    Installed("Installed first"),
    Size("Size"),
    Name("Name"),
}

enum class CatalogFilterMode(val label: String) {
    All("All"),
    Installed("Installed"),
    Dit("DiT"),
    Music("Music"),
    Voice("Voice"),
    Sdxl("SDXL"),
    Custom("Custom"),
}

fun filterAndSortCatalog(
    models: List<Model>,
    query: String,
    filter: CatalogFilterMode,
    sort: CatalogSortMode,
): List<Model> {
    val needle = query.trim().lowercase(Locale.US)
    val filtered = models.filter { model ->
        val matchesSearch = needle.isBlank() ||
            model.name.lowercase(Locale.US).contains(needle) ||
            model.description.lowercase(Locale.US).contains(needle)
        val matchesFilter = when (filter) {
            CatalogFilterMode.All -> true
            CatalogFilterMode.Installed -> model.isDownloaded
            CatalogFilterMode.Dit -> model.isDit
            CatalogFilterMode.Music -> model.isMusic
            CatalogFilterMode.Voice -> model.isVoice
            CatalogFilterMode.Sdxl -> model.isSdxl
            CatalogFilterMode.Custom -> model.isCustom
        }
        matchesSearch && matchesFilter
    }
    return when (sort) {
        CatalogSortMode.Smart -> filtered
        CatalogSortMode.Installed ->
            filtered.sortedWith(compareByDescending<Model> { it.isDownloaded }.thenBy { it.name.lowercase() })
        CatalogSortMode.Size ->
            filtered.sortedByDescending { model ->
                if (model.downloadBytesEstimate > 0L) {
                    model.downloadBytesEstimate
                } else {
                    val raw = model.approximateSize.trim().uppercase(Locale.US)
                    val number = raw
                        .removeSuffix("GB")
                        .removeSuffix("MB")
                        .trim()
                        .toDoubleOrNull() ?: 0.0
                    when {
                        raw.endsWith("GB") -> (number * 1_000_000_000L).toLong()
                        raw.endsWith("MB") -> (number * 1_000_000L).toLong()
                        else -> 0L
                    }
                }
            }
        CatalogSortMode.Name ->
            filtered.sortedBy { it.name.lowercase() }
    }
}

@Composable
fun ModelCatalogControls(
    query: String,
    onQueryChange: (String) -> Unit,
    filter: CatalogFilterMode,
    onFilterChange: (CatalogFilterMode) -> Unit,
    sort: CatalogSortMode,
    onSortChange: (CatalogSortMode) -> Unit,
    modifier: Modifier = Modifier,
) {
    
    var sortMenu by remember { mutableStateOf(false) }
    Column(
        modifier = modifier.fillMaxWidth(),
        verticalArrangement = Arrangement.spacedBy(10.dp),
    ) {
        OutlinedTextField(
            value = query,
            onValueChange = onQueryChange,
            modifier = Modifier.fillMaxWidth(),
            singleLine = true,
            leadingIcon = { Icon(Icons.Default.Search, contentDescription = null) },
            trailingIcon = {
                if (query.isNotEmpty()) {
                    IconButton(onClick = { onQueryChange("") }) {
                        Icon(Icons.Default.Close, contentDescription = "Clear search")
                    }
                }
            },
            placeholder = { Text("Search models") },
            shape = MaterialTheme.shapes.large,
        )
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .horizontalScroll(rememberScrollState()),
            horizontalArrangement = Arrangement.spacedBy(8.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            CatalogFilterMode.entries.forEach { option ->
                FilterChip(
                    selected = filter == option,
                    onClick = { onFilterChange(option) },
                    label = { Text(option.label) },
                )
            }
            Box {
                AssistChip(
                    onClick = { sortMenu = true },
                    label = { Text(sort.label) },
                    leadingIcon = { Icon(Icons.Default.Sort, contentDescription = null) },
                )
                DropdownMenu(
                    expanded = sortMenu,
                    onDismissRequest = { sortMenu = false },
                ) {
                    CatalogSortMode.entries.forEach { option ->
                        DropdownMenuItem(
                            text = { Text(option.label) },
                            onClick = {
                                onSortChange(option)
                                sortMenu = false
                            },
                        )
                    }
                }
            }
        }
    }
}

private fun gb(bytes: Long): String =
    if (bytes <= 0L) "—" else String.format(Locale.US, "%.2f GB", bytes / 1_000_000_000.0)

@Composable
private fun AnimatedFamilyHero(
    title: String, subtitle: String, music: Boolean,
    voice: Boolean = false, modifier: Modifier = Modifier,
) {
    val kind = if (voice) "voice" else if (music) "music" else "image"
    Surface(modifier = modifier.fillMaxWidth(),
        shape = RoundedCornerShape(22.dp),
        color = MaterialTheme.colorScheme.surfaceContainer) {
        StudioGradientHeader(kind, title, subtitle)
    }
}

@Composable
private fun AnimatedMusicMark(modifier: Modifier = Modifier) {
    val transition = rememberInfiniteTransition(label = "musicMark")
    val levels = List(4) { index ->
        val value by transition.animateFloat(
            initialValue = 0.30f + index * 0.08f,
            targetValue = 0.95f - index * 0.07f,
            animationSpec = infiniteRepeatable(
                animation = tween(420 + index * 100),
                repeatMode = RepeatMode.Reverse,
            ),
            label = "wave$index",
        )
        value
    }
    Box(modifier = modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
        Icon(
            Icons.Default.MusicNote,
            contentDescription = null,
            tint = Color.White,
            modifier = Modifier
                .align(Alignment.CenterStart)
                .size(22.dp),
        )
        Row(
            modifier = Modifier.align(Alignment.CenterEnd),
            horizontalArrangement = Arrangement.spacedBy(3.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            levels.forEach { level ->
                Box(
                    Modifier
                        .width(3.dp)
                        .height((25f * level).dp)
                        .background(Color.White.copy(alpha = 0.92f), CircleShape),
                )
            }
        }
    }
}

@OptIn(ExperimentalFoundationApi::class)
@Composable
private fun FamilyVariantTile(
    selected: Boolean,
    installed: Boolean,
    title: String,
    subtitle: String,
    icon: androidx.compose.ui.graphics.vector.ImageVector,
    onClick: () -> Unit,
    onLongClick: (() -> Unit)? = null,
    modifier: Modifier = Modifier,
) {
    val context = LocalContext.current
    val accent = MaterialTheme.colorScheme.primary
    val shape = RoundedCornerShape(18.dp)
    Surface(
        modifier = modifier.fillMaxWidth().combinedClickable(
            onClick = {
                AppHaptics.perform(context, AppHaptics.Kind.Interaction)
                onClick()
            },
            onLongClick = {
                if (installed) {
                    AppHaptics.perform(context, AppHaptics.Kind.Stage)
                    onLongClick?.invoke()
                }
            },
        ),
        shape = shape,
        color = if (selected) accent.copy(alpha = .11f)
            else MaterialTheme.colorScheme.surfaceContainerHigh,
        border = BorderStroke(if (selected) 1.5.dp else 1.dp,
            if (selected) accent else MaterialTheme.colorScheme.outlineVariant.copy(alpha = .65f)),
    ) {
        Row(Modifier.fillMaxWidth().padding(horizontal = 15.dp, vertical = 14.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            Box(Modifier.size(43.dp)
                .background(if (selected) accent.copy(alpha = .17f)
                    else MaterialTheme.colorScheme.surfaceContainerHighest,
                    RoundedCornerShape(13.dp)),
                contentAlignment = Alignment.Center) {
                Icon(icon, null, Modifier.size(21.dp),
                    tint = if (selected) accent else MaterialTheme.colorScheme.onSurfaceVariant)
            }
            Column(Modifier.weight(1f),
                verticalArrangement = Arrangement.spacedBy(5.dp)) {
                Text(title, style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.SemiBold)
                Text(subtitle, style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    maxLines = 2)
            }
            if (selected) {
                Icon(Icons.Default.CheckCircle, "Selected",
                    Modifier.size(23.dp), tint = accent)
            } else if (installed) {
                Text("READY", style = MaterialTheme.typography.labelSmall,
                    fontWeight = FontWeight.Bold,
                    color = MaterialTheme.colorScheme.onSurfaceVariant)
            } else {
                Icon(Icons.Default.CloudDownload, "Available to download",
                    Modifier.size(19.dp),
                    tint = MaterialTheme.colorScheme.onSurfaceVariant)
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class, ExperimentalFoundationApi::class)
@Composable
fun QwenFamilyCard(
    variants: List<Model>,
    onOpen: (Model) -> Unit,
    onDownload: (Model) -> Unit,
    onDeletePrecision: (String) -> Unit,
    onDeleteAdapter: (String) -> Unit,
    modifier: Modifier = Modifier,
) {
    if (variants.isEmpty()) return

    val context = LocalContext.current
    var showSheet by remember { mutableStateOf(false) }
    val installedPrecisions = listOf("Q4_0", "Q8_0", "FP8").filter { precision ->
        variants.any {
            it.variantPrecision == precision &&
                it.variantAdapter.isEmpty() &&
                it.isDownloaded
        }
    }
    val recommended = variants.firstOrNull { it.recommendedVariant }
        ?: variants.firstOrNull { it.isDownloaded }
        ?: variants.first()

    
    StudioFamilyCard(
        title = "Qwen Image 2.1",
        subtitle = "Image generation · selectable precision",
        kind = "image",
        variants = variants,
        featured = recommended,
        onClick = {
            AppHaptics.perform(context, AppHaptics.Kind.Interaction)
            showSheet = true
        },
        modifier = modifier,
    )

    if (showSheet) {
        QwenVariantSheet(
            variants = variants,
            onDeletePrecision = onDeletePrecision,
            onDeleteAdapter = onDeleteAdapter,
            onDismiss = { showSheet = false },
            onOpen = {
                showSheet = false
                onOpen(it)
            },
            onDownload = {
                showSheet = false
                onDownload(it)
            },
        )
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun QwenVariantSheet(
    variants: List<Model>,
    onDeletePrecision: (String) -> Unit,
    onDeleteAdapter: (String) -> Unit,
    onDismiss: () -> Unit,
    onOpen: (Model) -> Unit,
    onDownload: (Model) -> Unit,
) {
    val official = variants.firstOrNull { it.id == "qwen_image_2_1_turbo" }
    val baseVariants = variants.filterNot { it.id == "qwen_image_2_1_turbo" }
    val initial = baseVariants.firstOrNull { it.recommendedVariant }
        ?: baseVariants.firstOrNull { it.isDownloaded } ?: baseVariants.firstOrNull()
    if (initial == null && official == null) return
    var useOfficial by remember { mutableStateOf(initial == null) }
    var precision by remember { mutableStateOf(initial?.variantPrecision ?: "Q8_0") }
    var adapter by remember { mutableStateOf(initial?.variantAdapter ?: "") }
    val base = baseVariants.firstOrNull {
        it.variantPrecision == precision && it.variantAdapter == adapter
    } ?: initial
    val current = (if (useOfficial) official else base) ?: initial ?: official ?: return
    val isReady = current.isDownloaded
    fun precisionReady(p: String): Boolean = baseVariants.any {
        it.variantPrecision == p && it.variantAdapter.isBlank() && it.isDownloaded
    }
    fun adapterReady(a: String): Boolean = a.isBlank() || baseVariants.any {
        it.variantAdapter == a && it.isDownloaded
    }

    ModalBottomSheet(
        onDismissRequest = onDismiss,
        shape = RoundedCornerShape(topStart = 28.dp, topEnd = 28.dp),
        containerColor = MaterialTheme.colorScheme.surfaceContainerLow,
    ) {
        Column(Modifier.fillMaxWidth().fillMaxHeight(.92f)) {
            Column(
                Modifier.weight(1f).verticalScroll(rememberScrollState())
                    .padding(horizontal = 16.dp),
                verticalArrangement = Arrangement.spacedBy(19.dp),
            ) {
                AnimatedFamilyHero(
                    "Qwen Image 2.1",
                    "Standard and Official Turbo are different transformers",
                    music = false,
                )
                Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    Text("Model architecture", style = MaterialTheme.typography.titleMedium,
                        fontWeight = FontWeight.Bold)
                    if (initial != null) {
                        FamilyVariantTile(
                            selected = !useOfficial,
                            installed = baseVariants.any { it.isDownloaded },
                            title = "Qwen Image 2.1 · Standard",
                            subtitle = "Choose Q4 / Q8 / FP8; optionally add a Viggle LoRA",
                            icon = Icons.Default.AutoAwesome,
                            onClick = { useOfficial = false },
                        )
                    }
                    if (official != null) {
                        FamilyVariantTile(
                            selected = useOfficial,
                            installed = official.isDownloaded,
                            title = "Qwen Image 2.1 Turbo · FP8",
                            subtitle = "Official distilled model · 8 steps · 13.7 GB · no LoRA",
                            icon = Icons.Default.Bolt,
                            onClick = { useOfficial = true },
                        )
                    }
                }
                if (!useOfficial && initial != null) {
                    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                        Text("Standard precision", style = MaterialTheme.typography.titleMedium,
                            fontWeight = FontWeight.Bold)
                        listOf("Q4_0", "Q8_0", "FP8").forEach { p ->
                            FamilyVariantTile(
                                selected = precision == p,
                                installed = precisionReady(p),
                                title = p,
                                subtitle = when (p) {
                                    "Q4_0" -> "Smaller GGUF · 4.20 GB"
                                    "Q8_0" -> "High quality GGUF · 7.69 GB"
                                    else -> "FP8 transformer · 7.12 GB"
                                },
                                icon = if (p == "Q4_0") Icons.Default.SdStorage else Icons.Default.Memory,
                                onClick = { precision = p },
                                onLongClick = if (precisionReady(p)) {
                                    { onDeletePrecision(p) }
                                } else null,
                            )
                        }
                    }
                    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                        Text("Optional Viggle LoRA", style = MaterialTheme.typography.titleMedium,
                            fontWeight = FontWeight.Bold)
                        Text("For Standard only. This is NOT Qwen Image 2.1 Turbo.",
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                            style = MaterialTheme.typography.bodySmall)
                        listOf("", "r128", "r256").forEach { choice ->
                            FamilyVariantTile(
                                selected = adapter == choice,
                                installed = choice.isNotBlank() && adapterReady(choice),
                                title = if (choice.isBlank()) "None · standard model"
                                    else "Viggle $choice",
                                subtitle = when (choice) {
                                    "r128" -> "6-step LoRA · 680 MB"
                                    "r256" -> "6-step LoRA · 1.36 GB"
                                    else -> "No extra adapter · 20 steps"
                                },
                                icon = if (choice.isBlank()) Icons.Default.AutoAwesome
                                    else Icons.Default.Speed,
                                onClick = { adapter = choice },
                                onLongClick = if (choice.isNotBlank() && adapterReady(choice)) {
                                    { onDeleteAdapter(choice) }
                                } else null,
                            )
                        }
                    }
                } else {
                    Surface(
                        shape = RoundedCornerShape(17.dp),
                        color = MaterialTheme.colorScheme.primaryContainer.copy(alpha = .42f),
                    ) {
                        Column(Modifier.fillMaxWidth().padding(16.dp),
                            verticalArrangement = Arrangement.spacedBy(6.dp)) {
                            Text("Actual Turbo model", style = MaterialTheme.typography.titleMedium,
                                fontWeight = FontWeight.Bold)
                            Text("Separate distilled FP8 DiT. Default 8-step inference, CFG 1. Installs independently and does not load Viggle adapters.",
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant)
                        }
                    }
                }
                Surface(
                    color = MaterialTheme.colorScheme.surfaceContainerHigh,
                    shape = RoundedCornerShape(18.dp),
                ) {
                    Row(Modifier.fillMaxWidth().padding(15.dp),
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                        Icon(if (isReady) Icons.Default.CheckCircle else Icons.Default.CloudDownload,
                            null, tint = MaterialTheme.colorScheme.primary)
                        Column(Modifier.weight(1f),
                            verticalArrangement = Arrangement.spacedBy(4.dp)) {
                            Text(if (useOfficial) "Official Turbo FP8"
                                else "$precision" + if (adapter.isEmpty()) " · Standard"
                                else " + Viggle $adapter",
                                style = MaterialTheme.typography.titleSmall,
                                fontWeight = FontWeight.SemiBold)
                            Text(if (isReady) "Installed" else "Download required",
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant)
                        }
                        Text(current.approximateSize,
                            style = MaterialTheme.typography.labelLarge,
                            fontWeight = FontWeight.SemiBold)
                    }
                }
                Spacer(Modifier.height(4.dp))
            }
            Surface(color = MaterialTheme.colorScheme.surfaceContainerLow, tonalElevation = 4.dp) {
                Button(
                    onClick = { if (isReady) onOpen(current) else onDownload(current) },
                    modifier = Modifier.fillMaxWidth()
                        .padding(horizontal = 16.dp, vertical = 12.dp)
                        .heightIn(min = 54.dp),
                    shape = RoundedCornerShape(16.dp),
                ) {
                    Icon(if (isReady) Icons.Default.PlayArrow else Icons.Default.CloudDownload,
                        contentDescription = null)
                    Spacer(Modifier.width(8.dp))
                    Text(if (isReady) "Use selected model"
                        else if (useOfficial) "Download Official Turbo"
                        else "Download selected files",
                        fontWeight = FontWeight.SemiBold)
                }
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun Yue2FamilyCard(
    variants: List<Model>,
    onOpen: (Model) -> Unit,
    onDownload: (Model) -> Unit,
    onDelete: (Model) -> Unit,
    modifier: Modifier = Modifier,
) {
    if (variants.isEmpty()) return

    val context = LocalContext.current
    val supported = variants.filter {
        it.variantPrecision == "Q8_0" || it.variantPrecision == "BF16"
    }
    if (supported.isEmpty()) return

    val legacyInstalled = variants.filter {
        (it.variantPrecision == "Q5_K_M" || it.variantPrecision == "Q6_K") &&
            it.isDownloaded
    }
    val initial = supported.firstOrNull { it.recommendedVariant }
        ?: supported.firstOrNull { it.isDownloaded }
        ?: supported.first()
    var selectedId by remember(supported) { mutableStateOf(initial.id) }
    var showSheet by remember { mutableStateOf(false) }
    val selected = supported.firstOrNull { it.id == selectedId } ?: initial
    val installed = supported.count { it.isDownloaded }

    
    StudioFamilyCard(
        title = "YuE2 3B",
        subtitle = "Full-length music and vocals",
        kind = "music",
        variants = supported,
        featured = selected,
        onClick = {
            AppHaptics.perform(context, AppHaptics.Kind.Interaction)
            showSheet = true
        },
        modifier = modifier,
    )

    if (showSheet) {
        Yue2VariantSheet(
            variants = supported,
            legacyInstalled = legacyInstalled,
            selectedId = selected.id,
            onDelete = onDelete,
            onSelect = { selectedId = it.id },
            onDismiss = { showSheet = false },
            onOpen = {
                showSheet = false
                onOpen(it)
            },
            onDownload = {
                showSheet = false
                onDownload(it)
            },
        )
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun Yue2VariantSheet(
    variants: List<Model>,
    legacyInstalled: List<Model>,
    selectedId: String,
    onDelete: (Model) -> Unit,
    onSelect: (Model) -> Unit,
    onDismiss: () -> Unit,
    onOpen: (Model) -> Unit,
    onDownload: (Model) -> Unit,
) {
    val context = LocalContext.current
    val sorted = variants.sortedBy {
        when (it.variantPrecision) {
            "Q8_0" -> 0
            else -> 1
        }
    }
    val selected = variants.firstOrNull { it.id == selectedId }
        ?: variants.firstOrNull { it.recommendedVariant }
        ?: variants.first()

    ModalBottomSheet(
        onDismissRequest = onDismiss,
        containerColor = MaterialTheme.colorScheme.surfaceContainerLow,
        shape = RoundedCornerShape(topStart = 36.dp, topEnd = 36.dp),
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .verticalScroll(rememberScrollState())
                .padding(horizontal = 18.dp)
                .padding(bottom = 28.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp),
        ) {
            AnimatedFamilyHero(
                title = "YuE2 3B",
                subtitle = "Choose model quality",
                music = true,
            )

            sorted.chunked(1).forEach { row ->
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                ) {
                    row.forEach { variant ->
                        FamilyVariantTile(
                            selected = selected.id == variant.id,
                            installed = variant.isDownloaded,
                            title = variant.variantPrecision,
                            subtitle = when (variant.variantPrecision) {
                                "Q8_0" -> "Near-lossless · 4.34 GB"
                                else -> "Reference quality · 7.70 GB"
                            },
                            icon = when (variant.variantPrecision) {
                                "Q8_0" -> Icons.Default.Memory
                                else -> Icons.Default.AutoAwesome
                            },
                            onClick = { onSelect(variant) },
                            onLongClick = if (variant.isDownloaded) {
                                { onDelete(variant) }
                            } else {
                                null
                            },
                            modifier = Modifier.weight(1f),
                        )
                    }
                }
            }

            AnimatedContent(
                targetState = selected.id,
                transitionSpec = { fadeIn(tween(180)) togetherWith fadeOut(tween(120)) },
                label = "yue2VariantDetails",
            ) {
                Surface(
                    shape = MaterialTheme.shapes.extraLarge,
                    color = MaterialTheme.colorScheme.surfaceContainerHigh,
                ) {
                    Column(
                        modifier = Modifier.padding(16.dp),
                        verticalArrangement = Arrangement.spacedBy(12.dp),
                    ) {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Column(modifier = Modifier.weight(1f)) {
                                Text(
                                    "YuE2 3B · ${selected.variantPrecision}",
                                    style = MaterialTheme.typography.titleMedium,
                                    fontWeight = FontWeight.SemiBold,
                                )
                                Text(
                                    when (selected.variantPrecision) {
                                        "BF16" ->
                                            "Highest-memory reference option."
                                        else ->
                                            "Recommended mobile quality and memory balance."
                                    },
                                    style = MaterialTheme.typography.bodySmall,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                                )
                            }
                            if (selected.isDownloaded) {
                                Icon(
                                    Icons.Default.CheckCircle,
                                    contentDescription = null,
                                    tint = MaterialTheme.colorScheme.primary,
                                )
                            }
                        }

                        Text(
                            when (selected.variantPrecision) {
                                "Q8_0" ->
                                    "Near-lossless upstream default and the recommended mobile option."
                                else ->
                                    "Full BF16 reference option for devices with enough memory."
                            },
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )

                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween,
                        ) {
                            Text(
                                "Full package",
                                color = MaterialTheme.colorScheme.onSurfaceVariant,
                            )
                            Text(
                                selected.approximateSize,
                                fontWeight = FontWeight.SemiBold,
                            )
                        }
                    }
                }
            }

            val instrumentalAdapter = File(
                File(Model.getModelsDir(context), selected.id),
                "instrumental_lora.gguf",
            )
            val adapterInstalled = instrumentalAdapter.isFile && instrumentalAdapter.length() > 1_000_000L

            Surface(
                shape = MaterialTheme.shapes.extraLarge,
                color = MaterialTheme.colorScheme.surfaceContainerHigh,
                border = BorderStroke(
                    1.dp,
                    if (adapterInstalled) {
                        MaterialTheme.colorScheme.primary.copy(alpha = 0.28f)
                    } else {
                        MaterialTheme.colorScheme.outlineVariant
                    },
                ),
            ) {
                Column(
                    modifier = Modifier.padding(16.dp),
                    verticalArrangement = Arrangement.spacedBy(10.dp),
                ) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Surface(
                            shape = CircleShape,
                            color = MaterialTheme.colorScheme.secondaryContainer,
                        ) {
                            Icon(
                                Icons.Default.MusicNote,
                                contentDescription = null,
                                modifier = Modifier.padding(10.dp),
                                tint = MaterialTheme.colorScheme.onSecondaryContainer,
                            )
                        }
                        Spacer(Modifier.width(12.dp))
                        Column(modifier = Modifier.weight(1f)) {
                            Text(
                                "Instrumental CoT adapter",
                                style = MaterialTheme.typography.titleMedium,
                                fontWeight = FontWeight.SemiBold,
                            )
                            Text(
                                "Rank-64 YuE2 AR LoRA · score-first instrumentals · F16 GGUF",
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant,
                            )
                        }
                        if (adapterInstalled) {
                            Icon(
                                Icons.Default.CheckCircle,
                                contentDescription = "Installed",
                                tint = MaterialTheme.colorScheme.primary,
                            )
                        }
                    }

                    Text(
                        "Recommended for piano and other no-vocal tracks. Local Dream combines it with Full planning and the official Vocal→Ins melody transfer.",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )

                    Button(
                        enabled = selected.isDownloaded && !adapterInstalled,
                        onClick = {
                            Model(
                                id = selected.id,
                                name = "YuE2 Instrumental CoT adapter",
                                description = "Instrumental AR LoRA",
                                baseUrl = "https://huggingface.co",
                                packageFiles = listOf(
                                    "thepatch/YuE2-3B-GGUF/resolve/main/" +
                                        "yue2-instrumental-cot-full-v1.0-F16-LoRA.gguf|" +
                                        "instrumental_lora.gguf",
                                ),
                                packageMarker = "YUE2_INSTRUMENTAL_LORA",
                            ).startDownload(context)
                        },
                        modifier = Modifier.fillMaxWidth(),
                        shape = MaterialTheme.shapes.large,
                    ) {
                        Icon(
                            if (adapterInstalled) Icons.Default.CheckCircle else Icons.Default.CloudDownload,
                            contentDescription = null,
                        )
                        Spacer(Modifier.width(8.dp))
                        Text(
                            when {
                                adapterInstalled -> "Instrumental adapter installed"
                                !selected.isDownloaded -> "Download ${selected.variantPrecision} first"
                                else -> "Download instrumental adapter"
                            },
                        )
                    }
                }
            }

            if (legacyInstalled.isNotEmpty()) {
                Surface(
                    shape = MaterialTheme.shapes.large,
                    color = MaterialTheme.colorScheme.errorContainer.copy(alpha = 0.58f),
                    border = BorderStroke(
                        1.dp,
                        MaterialTheme.colorScheme.error.copy(alpha = 0.18f),
                    ),
                ) {
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(horizontal = 14.dp, vertical = 11.dp),
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        Column(modifier = Modifier.weight(1f)) {
                            Text(
                                "Legacy download",
                                style = MaterialTheme.typography.labelMedium,
                                color = MaterialTheme.colorScheme.onErrorContainer,
                            )
                            Text(
                                legacyInstalled.joinToString(" · ") {
                                    "${it.variantPrecision} ${it.approximateSize}"
                                },
                                style = MaterialTheme.typography.bodyMedium,
                                fontWeight = FontWeight.SemiBold,
                                color = MaterialTheme.colorScheme.onErrorContainer,
                            )
                        }
                        TextButton(onClick = { onDelete(legacyInstalled.first()) }) {
                            Text("Remove")
                        }
                    }
                }
            }

            Button(
                onClick = {
                    if (selected.isDownloaded) onOpen(selected) else onDownload(selected)
                },
                modifier = Modifier.fillMaxWidth(),
                shape = MaterialTheme.shapes.large,
            ) {
                Icon(
                    if (selected.isDownloaded) Icons.Default.MusicNote else Icons.Default.CloudDownload,
                    contentDescription = null,
                )
                Spacer(Modifier.width(8.dp))
                Text(
                    if (selected.isDownloaded) {
                        "Create music with ${selected.variantPrecision}"
                    } else {
                        "Download ${selected.approximateSize}"
                    },
                )
            }
        }
    }
}


@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun BreezeFamilyCard(
    variants: List<Model>,
    onOpen: (Model) -> Unit,
    onDownload: (Model) -> Unit,
    onDelete: (Model) -> Unit,
    modifier: Modifier = Modifier,
) {
    if (variants.isEmpty()) return

    val context = LocalContext.current
    val order = listOf(
        "Q8_0",
        "Q6_K",
        "Q4_K",
        "F16",
        "Q8_0 · DD4",
        "Q8_0 · DD2",
        "Q4_K · DD2",
    )
    val sorted = variants.sortedBy {
        order.indexOf(it.variantPrecision).let { index ->
            if (index < 0) Int.MAX_VALUE else index
        }
    }
    val initial = sorted.firstOrNull { it.recommendedVariant }
        ?: sorted.firstOrNull { it.isDownloaded }
        ?: sorted.first()
    var selectedId by remember(sorted) { mutableStateOf(initial.id) }
    var showSheet by remember { mutableStateOf(false) }
    val selected = sorted.firstOrNull { it.id == selectedId } ?: initial
    val installed = sorted.count { it.isDownloaded }

    
    StudioFamilyCard(
        title = "Breeze TTS 2",
        subtitle = "Expressive speech and voices",
        kind = "voice",
        variants = sorted,
        featured = selected,
        onClick = {
            AppHaptics.perform(context, AppHaptics.Kind.Interaction)
            showSheet = true
        },
        modifier = modifier,
    )

    if (showSheet) {
        ModalBottomSheet(
        onDismissRequest = { showSheet = false },
        containerColor = MaterialTheme.colorScheme.surfaceContainerLow,
        shape = RoundedCornerShape(topStart = 36.dp, topEnd = 36.dp),
    ) {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .verticalScroll(rememberScrollState())
                    .padding(horizontal = 18.dp)
                    .padding(bottom = 28.dp),
                verticalArrangement = Arrangement.spacedBy(16.dp),
            ) {
                AnimatedFamilyHero(
                    title = "Breeze TTS 2",
                    subtitle = "English + Mandarin text to speech",
                    music = false,
                    voice = true,
                )

                Text(
                    "Quality models",
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.SemiBold,
                )
                sorted.filterNot { "DD" in it.variantPrecision }
                    .chunked(1)
                    .forEach { row ->
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.spacedBy(8.dp),
                        ) {
                            row.forEach { variant ->
                                FamilyVariantTile(
                                    selected = selected.id == variant.id,
                                    installed = variant.isDownloaded,
                                    title = variant.variantPrecision,
                                    subtitle = variant.variantProfile + " · " +
                                        variant.approximateSize,
                                    icon = if (variant.recommendedVariant) {
                                        Icons.Default.GraphicEq
                                    } else {
                                        Icons.Default.Memory
                                    },
                                    onClick = { selectedId = variant.id },
                                    onLongClick = if (variant.isDownloaded) {
                                        { onDelete(variant) }
                                    } else {
                                        null
                                    },
                                    modifier = Modifier.weight(1f),
                                )
                            }

                        }
                    }

                Text(
                    "Experimental depth quants",
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.SemiBold,
                )
                Text(
                    "Faster/smaller for short dialogue. The quantized depth decoder can lose high-frequency detail over long continuous narration.",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
                sorted.filter { "DD" in it.variantPrecision }
                    .chunked(1)
                    .forEach { row ->
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.spacedBy(8.dp),
                        ) {
                            row.forEach { variant ->
                                FamilyVariantTile(
                                    selected = selected.id == variant.id,
                                    installed = variant.isDownloaded,
                                    title = variant.variantPrecision,
                                    subtitle = variant.variantProfile + " · " +
                                        variant.approximateSize,
                                    icon = Icons.Default.Speed,
                                    onClick = { selectedId = variant.id },
                                    onLongClick = if (variant.isDownloaded) {
                                        { onDelete(variant) }
                                    } else {
                                        null
                                    },
                                    modifier = Modifier.weight(1f),
                                )
                            }

                        }
                    }

                val current = sorted.firstOrNull { it.id == selectedId } ?: selected
                Surface(
                    modifier = Modifier.fillMaxWidth(),
                    shape = RoundedCornerShape(18.dp),
                    color = MaterialTheme.colorScheme.surfaceContainerHigh,
                ) {
                    Row(
                        Modifier.fillMaxWidth().padding(15.dp),
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(12.dp),
                    ) {
                        Icon(Icons.Default.CheckCircle, null,
                            modifier = Modifier.size(23.dp),
                            tint = MaterialTheme.colorScheme.primary)
                        Column(Modifier.weight(1f),
                            verticalArrangement = Arrangement.spacedBy(3.dp)) {
                            Text("Selected · ${current.variantPrecision}",
                                style = MaterialTheme.typography.titleSmall,
                                fontWeight = FontWeight.SemiBold)
                            Text(current.variantProfile,
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant)
                        }
                        Text(current.approximateSize,
                            style = MaterialTheme.typography.labelLarge,
                            fontWeight = FontWeight.SemiBold)
                    }
                }

                Button(
                    onClick = {
                        showSheet = false
                        if (current.isDownloaded) onOpen(current) else onDownload(current)
                    },
                    modifier = Modifier.fillMaxWidth().heightIn(min = 52.dp),
                    shape = RoundedCornerShape(16.dp),
                    colors = androidx.compose.material3.ButtonDefaults.buttonColors(
                        containerColor = MaterialTheme.colorScheme.primary,
                        contentColor = MaterialTheme.colorScheme.onPrimary,
                    ),
                ) {
                    Icon(
                        if (current.isDownloaded) Icons.Default.PlayArrow
                        else Icons.Default.CloudDownload,
                        null,
                    )
                    Spacer(Modifier.width(9.dp))
                    Text(if (current.isDownloaded) "Open selected voice model"
                        else "Download selected model",
                        fontWeight = FontWeight.SemiBold)
                }
            }
        }
    }
}
