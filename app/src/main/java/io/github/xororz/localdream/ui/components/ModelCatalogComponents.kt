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
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.ExperimentalFoundationApi
import androidx.compose.foundation.background
import androidx.compose.foundation.combinedClickable
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.foundation.shape.CircleShape
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
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import io.github.xororz.localdream.data.Model
import io.github.xororz.localdream.service.BreezeQnnGeneratorArtifact
import io.github.xororz.localdream.service.ModelDownloadService
import java.io.File
import io.github.xororz.localdream.utils.AppHaptics
import java.util.Locale
import kotlinx.coroutines.launch

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
    title: String,
    subtitle: String,
    music: Boolean,
    voice: Boolean = false,
    modifier: Modifier = Modifier,
) {
    val gradient = when {
        voice -> Brush.linearGradient(
            listOf(
                Color(0xFF092D33),
                Color(0xFF126A73),
                Color(0xFF2AA9A5),
            ),
        )
        music -> Brush.linearGradient(
            listOf(
                Color(0xFF351344),
                Color(0xFF713069),
                Color(0xFFB84D81),
            ),
        )
        else -> Brush.linearGradient(
            listOf(
                Color(0xFF071D45),
                Color(0xFF0A4A9F),
                Color(0xFF2878E8),
            ),
        )
    }

    Box(
        modifier = modifier
            .fillMaxWidth()
            .background(gradient, MaterialTheme.shapes.extraLarge)
            .padding(horizontal = 18.dp, vertical = 18.dp),
    ) {
        Column(verticalArrangement = Arrangement.spacedBy(14.dp)) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(13.dp),
            ) {
                Surface(
                    modifier = Modifier.size(54.dp),
                    shape = MaterialTheme.shapes.large,
                    color = Color.White.copy(alpha = 0.12f),
                    border = BorderStroke(1.dp, Color.White.copy(alpha = 0.08f)),
                ) {
                    if (voice) {
                        Icon(
                            Icons.Default.GraphicEq,
                            contentDescription = null,
                            tint = Color.White,
                            modifier = Modifier.padding(13.dp),
                        )
                    } else if (music) {
                        AnimatedMusicMark(Modifier.padding(11.dp))
                    } else {
                        Icon(
                            Icons.Default.AutoAwesome,
                            contentDescription = null,
                            tint = Color.White,
                            modifier = Modifier.padding(13.dp),
                        )
                    }
                }

                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        title,
                        style = MaterialTheme.typography.headlineSmall,
                        fontWeight = FontWeight.Bold,
                        color = Color.White,
                    )
                    Text(
                        subtitle,
                        style = MaterialTheme.typography.bodyMedium,
                        color = Color.White.copy(alpha = 0.78f),
                    )
                }

            }

        }
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
    Surface(
        modifier = modifier
            .heightIn(min = 106.dp)
            .combinedClickable(
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
        shape = MaterialTheme.shapes.extraLarge,
        color = if (selected) {
            MaterialTheme.colorScheme.primaryContainer.copy(alpha = 0.72f)
        } else {
            MaterialTheme.colorScheme.surfaceContainerHigh
        },
        border = BorderStroke(
            1.dp,
            if (selected) {
                MaterialTheme.colorScheme.primary.copy(alpha = 0.72f)
            } else {
                MaterialTheme.colorScheme.outlineVariant.copy(alpha = 0.55f)
            },
        ),
        tonalElevation = if (selected) 3.dp else 0.dp,
    ) {
        Column(
            modifier = Modifier.padding(14.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Icon(
                    icon,
                    contentDescription = null,
                    modifier = Modifier.size(20.dp),
                    tint = if (selected) {
                        MaterialTheme.colorScheme.primary
                    } else {
                        MaterialTheme.colorScheme.onSurfaceVariant
                    },
                )
                Spacer(Modifier.width(8.dp))
                Text(
                    title,
                    style = MaterialTheme.typography.titleSmall,
                    fontWeight = FontWeight.SemiBold,
                    modifier = Modifier.weight(1f),
                )
                if (installed) {
                    Icon(
                        Icons.Default.CheckCircle,
                        contentDescription = "Installed",
                        modifier = Modifier.size(19.dp),
                        tint = MaterialTheme.colorScheme.primary,
                    )
                }
            }
            Text(
                subtitle,
                style = MaterialTheme.typography.labelMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                maxLines = 2,
            )
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

    ElevatedCard(
        onClick = {
            AppHaptics.perform(context, AppHaptics.Kind.Interaction)
            showSheet = true
        },
        modifier = modifier.fillMaxWidth(),
        shape = MaterialTheme.shapes.extraLarge,
        colors = CardDefaults.elevatedCardColors(
            containerColor = MaterialTheme.colorScheme.surfaceContainer,
        ),
        elevation = CardDefaults.elevatedCardElevation(defaultElevation = 1.dp),
    ) {
        Column {
            AnimatedFamilyHero(
                title = "Qwen Image 2.1",
                subtitle = "Image generation",
                music = false,
            )

            HorizontalDivider(
                modifier = Modifier.padding(horizontal = 18.dp),
                color = MaterialTheme.colorScheme.outlineVariant.copy(alpha = 0.55f),
            )

            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .height(IntrinsicSize.Min)
                    .padding(horizontal = 18.dp, vertical = 14.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        "Recommended",
                        style = MaterialTheme.typography.labelMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                    Text(
                        "${recommended.variantPrecision}" +
                            if (recommended.variantAdapter.isNotEmpty()) {
                                " + Viggle ${recommended.variantAdapter}"
                            } else {
                                " · Base"
                            },
                        style = MaterialTheme.typography.titleMedium,
                        fontWeight = FontWeight.SemiBold,
                    )
                }

                VerticalDivider(
                    modifier = Modifier
                        .fillMaxHeight()
                        .padding(vertical = 2.dp),
                    color = MaterialTheme.colorScheme.outlineVariant.copy(alpha = 0.55f),
                )

                TextButton(
                    onClick = {
                        AppHaptics.perform(context, AppHaptics.Kind.Interaction)
                        showSheet = true
                    },
                    modifier = Modifier.padding(start = 8.dp),
                ) {
                    Icon(Icons.Default.Tune, contentDescription = null)
                    Spacer(Modifier.width(6.dp))
                    Text("Configure")
                }
            }
        }
    }

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
    val initial = variants.firstOrNull { it.recommendedVariant }
        ?: variants.firstOrNull { it.isDownloaded }
        ?: variants.first()
    var precision by remember { mutableStateOf(initial.variantPrecision) }
    var adapter by remember { mutableStateOf(initial.variantAdapter) }

    val precisions = listOf("Q4_0", "Q8_0", "FP8")
    val adapters = listOf("", "r128", "r256")
    val selected = variants.firstOrNull {
        it.variantPrecision == precision && it.variantAdapter == adapter
    } ?: variants.first()

    fun precisionReady(p: String): Boolean =
        variants.any {
            it.variantPrecision == p &&
                it.variantAdapter.isEmpty() &&
                it.isDownloaded
        }

    fun adapterReady(a: String): Boolean =
        a.isEmpty() || variants.any { it.variantAdapter == a && it.isDownloaded }

    val baseReady = precisionReady(precision)
    val turboReady = adapterReady(adapter)

    ModalBottomSheet(onDismissRequest = onDismiss) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .verticalScroll(rememberScrollState())
                .padding(horizontal = 18.dp)
                .padding(bottom = 28.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp),
        ) {
            AnimatedFamilyHero(
                title = "Qwen Image 2.1",
                subtitle = "Choose quality and optional Viggle LoRA",
                music = false,
            )

            Column(verticalArrangement = Arrangement.spacedBy(9.dp)) {
                Text(
                    "Transformer",
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.SemiBold,
                )
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                ) {
                    precisions.forEach { p ->
                        FamilyVariantTile(
                            selected = precision == p,
                            installed = precisionReady(p),
                            title = p,
                            subtitle = when (p) {
                                "Q4_0" -> "4.20 GB · compact"
                                "Q8_0" -> "7.69 GB · quality"
                                else -> "7.12 GB · FP8"
                            },
                            icon = when (p) {
                                "Q4_0" -> Icons.Default.SdStorage
                                "Q8_0" -> Icons.Default.Memory
                                else -> Icons.Default.Bolt
                            },
                            onClick = { precision = p },
                            onLongClick = if (precisionReady(p)) {
                                { onDeletePrecision(p) }
                            } else {
                                null
                            },
                            modifier = Modifier.weight(1f),
                        )
                    }
                }
            }

            Column(verticalArrangement = Arrangement.spacedBy(9.dp)) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text(
                        "Turbo adapter",
                        style = MaterialTheme.typography.titleMedium,
                        fontWeight = FontWeight.SemiBold,
                        modifier = Modifier.weight(1f),
                    )
                    Text(
                        "shared across all precisions",
                        style = MaterialTheme.typography.labelMedium,
                        color = MaterialTheme.colorScheme.primary,
                    )
                }
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                ) {
                    adapters.forEach { a ->
                        FamilyVariantTile(
                            selected = adapter == a,
                            installed = a.isNotEmpty() && adapterReady(a),
                            title = when (a) {
                                "r128" -> "r128"
                                "r256" -> "r256"
                                else -> "Base"
                            },
                            subtitle = when (a) {
                                "r128" -> "680 MB · 6-pass"
                                "r256" -> "1.36 GB · 6-pass"
                                else -> "20 steps · no extra file"
                            },
                            icon = if (a.isEmpty()) Icons.Default.PlayArrow else Icons.Default.Speed,
                            onClick = { adapter = a },
                            onLongClick = if (a.isNotEmpty() && adapterReady(a)) {
                                { onDeleteAdapter(a) }
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
                label = "qwenVariantDetails",
            ) {
                Surface(
                    shape = MaterialTheme.shapes.extraLarge,
                    color = MaterialTheme.colorScheme.surfaceContainerHigh,
                ) {
                    Column(
                        modifier = Modifier.padding(16.dp),
                        verticalArrangement = Arrangement.spacedBy(11.dp),
                    ) {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Column(modifier = Modifier.weight(1f)) {
                                Text(
                                    "$precision" +
                                        if (adapter.isBlank()) " · Base" else " + Viggle $adapter",
                                    style = MaterialTheme.typography.titleMedium,
                                    fontWeight = FontWeight.SemiBold,
                                )
                                Text(
                                    if (adapter.isBlank()) {
                                        "Standard Qwen Image 2.1 runtime"
                                    } else {
                                        "Viggle LoRA enabled"
                                    },
                                    style = MaterialTheme.typography.bodySmall,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                                )
                            }
                            if (baseReady && turboReady) {
                                Icon(
                                    Icons.Default.CheckCircle,
                                    contentDescription = null,
                                    tint = MaterialTheme.colorScheme.primary,
                                )
                            }
                        }

                        Text(
                            "Common encoder/VAE files and Viggle LoRAs are stored once. Switching Q4/Q8/FP8 reuses them automatically.",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                        Text(
                            "Full stack from zero: ${selected.approximateSize}",
                            style = MaterialTheme.typography.labelMedium,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                    }
                }
            }

            Button(
                onClick = {
                    if (baseReady && turboReady) onOpen(selected) else onDownload(selected)
                },
                modifier = Modifier.fillMaxWidth(),
                shape = MaterialTheme.shapes.large,
            ) {
                Icon(
                    if (baseReady && turboReady) Icons.Default.PlayArrow else Icons.Default.CloudDownload,
                    contentDescription = null,
                )
                Spacer(Modifier.width(8.dp))
                Text(
                    when {
                        baseReady && turboReady ->
                            "Use $precision" + if (adapter.isBlank()) "" else " + $adapter"
                        !baseReady && turboReady ->
                            "Download $precision transformer"
                        baseReady && !turboReady ->
                            "Download Viggle $adapter only"
                        else -> "Download missing assets"
                    },
                )
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

    ElevatedCard(
        onClick = {
            AppHaptics.perform(context, AppHaptics.Kind.Interaction)
            showSheet = true
        },
        modifier = modifier.fillMaxWidth(),
        shape = MaterialTheme.shapes.extraLarge,
        colors = CardDefaults.elevatedCardColors(
            containerColor = MaterialTheme.colorScheme.surfaceContainer,
        ),
        elevation = CardDefaults.elevatedCardElevation(defaultElevation = 1.dp),
    ) {
        Column {
            AnimatedFamilyHero(
                title = "YuE2 3B",
                subtitle = "Text to music",
                music = true,
            )

            HorizontalDivider(
                modifier = Modifier.padding(horizontal = 18.dp),
                color = MaterialTheme.colorScheme.outlineVariant.copy(alpha = 0.55f),
            )

            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .height(IntrinsicSize.Min)
                    .padding(horizontal = 18.dp, vertical = 14.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        if (selected.isDownloaded) "Ready to create" else "Recommended",
                        style = MaterialTheme.typography.labelMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                    Text(
                        "${selected.variantPrecision} · ${selected.variantProfile}",
                        style = MaterialTheme.typography.titleMedium,
                        fontWeight = FontWeight.SemiBold,
                    )
                }

                VerticalDivider(
                    modifier = Modifier
                        .fillMaxHeight()
                        .padding(vertical = 2.dp),
                    color = MaterialTheme.colorScheme.outlineVariant.copy(alpha = 0.55f),
                )

                TextButton(
                    onClick = {
                        AppHaptics.perform(context, AppHaptics.Kind.Interaction)
                        showSheet = true
                    },
                    modifier = Modifier.padding(start = 8.dp),
                ) {
                    Icon(Icons.Default.Tune, contentDescription = null)
                    Spacer(Modifier.width(6.dp))
                    Text("Configure")
                }
            }
        }
    }

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

    ModalBottomSheet(onDismissRequest = onDismiss) {
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

            sorted.chunked(2).forEach { row ->
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
    val scope = rememberCoroutineScope()
    val qnnGeneratorState by BreezeQnnGeneratorArtifact.status.collectAsState()
    var fullQnnEnabled by remember {
        mutableStateOf(BreezeQnnGeneratorArtifact.isEnabled(context))
    }
    val sharedDownloadState by ModelDownloadService.downloadState.collectAsState()
    val qnnTransferState = sharedDownloadState.takeIf { state ->
        when (state) {
            is ModelDownloadService.DownloadState.Downloading ->
                state.modelId == BreezeQnnGeneratorArtifact.DOWNLOAD_MODEL_ID
            is ModelDownloadService.DownloadState.Paused ->
                state.modelId == BreezeQnnGeneratorArtifact.DOWNLOAD_MODEL_ID
            is ModelDownloadService.DownloadState.Success ->
                state.modelId == BreezeQnnGeneratorArtifact.DOWNLOAD_MODEL_ID
            is ModelDownloadService.DownloadState.Error ->
                state.modelId == BreezeQnnGeneratorArtifact.DOWNLOAD_MODEL_ID
            else -> false
        }
    }
    LaunchedEffect(Unit) {
        BreezeQnnGeneratorArtifact.refresh(context)
    }
    LaunchedEffect(qnnTransferState) {
        if (qnnTransferState is ModelDownloadService.DownloadState.Success) {
            BreezeQnnGeneratorArtifact.refresh(context)
        }
    }
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

    ElevatedCard(
        onClick = {
            AppHaptics.perform(context, AppHaptics.Kind.Interaction)
            showSheet = true
        },
        modifier = modifier.fillMaxWidth(),
        shape = MaterialTheme.shapes.extraLarge,
        colors = CardDefaults.elevatedCardColors(
            containerColor = MaterialTheme.colorScheme.surfaceContainer,
        ),
        elevation = CardDefaults.elevatedCardElevation(defaultElevation = 1.dp),
    ) {
        Column {
            AnimatedFamilyHero(
                title = "Breeze TTS 2",
                subtitle = "Text to speech",
                music = false,
                voice = true,
            )
            HorizontalDivider(
                modifier = Modifier.padding(horizontal = 18.dp),
                color = MaterialTheme.colorScheme.outlineVariant.copy(alpha = 0.55f),
            )
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .height(IntrinsicSize.Min)
                    .padding(horizontal = 18.dp, vertical = 14.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        if (selected.isDownloaded) "Ready to speak" else "Recommended",
                        style = MaterialTheme.typography.labelMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                    Text(
                        selected.variantPrecision + " · " + selected.variantProfile,
                        style = MaterialTheme.typography.titleMedium,
                        fontWeight = FontWeight.SemiBold,
                    )
                }
                VerticalDivider(
                    modifier = Modifier
                        .fillMaxHeight()
                        .padding(vertical = 2.dp),
                    color = MaterialTheme.colorScheme.outlineVariant.copy(alpha = 0.55f),
                )
                TextButton(
                    onClick = { showSheet = true },
                    modifier = Modifier.padding(start = 8.dp),
                ) {
                    Icon(Icons.Default.Tune, contentDescription = null)
                    Spacer(Modifier.width(6.dp))
                    Text("Configure")
                }
            }
        }
    }

    if (showSheet) {
        ModalBottomSheet(onDismissRequest = { showSheet = false }) {
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
                    .chunked(2)
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
                            if (row.size == 1) Spacer(Modifier.weight(1f))
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
                    .chunked(2)
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
                            if (row.size == 1) Spacer(Modifier.weight(1f))
                        }
                    }

                Text(
                    "Snapdragon acceleration",
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.SemiBold,
                )
                Surface(
                    shape = MaterialTheme.shapes.extraLarge,
                    color = MaterialTheme.colorScheme.surfaceContainerHigh,
                ) {
                    Column(
                        modifier = Modifier.padding(16.dp),
                        verticalArrangement = Arrangement.spacedBy(12.dp),
                    ) {
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.spacedBy(12.dp),
                        ) {
                            Surface(
                                modifier = Modifier.size(46.dp),
                                shape = MaterialTheme.shapes.large,
                                color = MaterialTheme.colorScheme.tertiaryContainer,
                            ) {
                                Icon(
                                    Icons.Default.Bolt,
                                    contentDescription = null,
                                    modifier = Modifier.padding(11.dp),
                                    tint = MaterialTheme.colorScheme.onTertiaryContainer,
                                )
                            }
                            Column(modifier = Modifier.weight(1f)) {
                                Text(
                                    "Full QNN Generator",
                                    style = MaterialTheme.typography.titleSmall,
                                    fontWeight = FontWeight.SemiBold,
                                )
                                Text(
                                    when (val transfer = qnnTransferState) {
                                        is ModelDownloadService.DownloadState.Downloading ->
                                            "Downloading accelerator"
                                        is ModelDownloadService.DownloadState.Paused ->
                                            "Download paused"
                                        is ModelDownloadService.DownloadState.Error ->
                                            transfer.message
                                        else -> when (val status = qnnGeneratorState) {
                                            is BreezeQnnGeneratorArtifact.Status.Ready ->
                                                if (fullQnnEnabled) {
                                                    "Installed · Full QNN enabled"
                                                } else {
                                                    "Installed · Legacy generator enabled"
                                                }
                                            is BreezeQnnGeneratorArtifact.Status.Unsupported ->
                                                "Not available on this Snapdragon"
                                            is BreezeQnnGeneratorArtifact.Status.Error ->
                                                status.message
                                            BreezeQnnGeneratorArtifact.Status.Checking ->
                                                "Checking device"
                                            is BreezeQnnGeneratorArtifact.Status.Missing ->
                                                "Optional · QNN backbone + depth accelerator"
                                            is BreezeQnnGeneratorArtifact.Status.Downloading ->
                                                "Preparing shared download service"
                                        }
                                    },
                                    style = MaterialTheme.typography.bodySmall,
                                    color = if (
                                        qnnTransferState is ModelDownloadService.DownloadState.Error ||
                                        qnnGeneratorState is BreezeQnnGeneratorArtifact.Status.Error
                                    ) {
                                        MaterialTheme.colorScheme.error
                                    } else {
                                        MaterialTheme.colorScheme.onSurfaceVariant
                                    },
                                    maxLines = 2,
                                )
                            }

                            when {
                                qnnGeneratorState is BreezeQnnGeneratorArtifact.Status.Ready -> {
                                    Switch(
                                        checked = fullQnnEnabled,
                                        onCheckedChange = { enabled ->
                                            fullQnnEnabled = enabled
                                            BreezeQnnGeneratorArtifact.setEnabled(context, enabled)
                                            AppHaptics.perform(
                                                context,
                                                AppHaptics.Kind.Interaction,
                                            )
                                        },
                                    )
                                }
                                qnnTransferState is ModelDownloadService.DownloadState.Downloading ||
                                    qnnTransferState is ModelDownloadService.DownloadState.Paused -> Unit
                                qnnGeneratorState is BreezeQnnGeneratorArtifact.Status.Unsupported ||
                                    qnnGeneratorState is BreezeQnnGeneratorArtifact.Status.Checking -> Unit
                                else -> {
                                    Button(
                                        onClick = {
                                            context.startForegroundService(
                                                Intent(
                                                    context,
                                                    ModelDownloadService::class.java,
                                                ).setAction(
                                                    ModelDownloadService
                                                        .ACTION_START_BREEZE_QNN_GENERATOR,
                                                ),
                                            )
                                        },
                                    ) {
                                        Icon(Icons.Default.CloudDownload, contentDescription = null)
                                        Spacer(Modifier.width(6.dp))
                                        Text(
                                            if (
                                                qnnGeneratorState is BreezeQnnGeneratorArtifact.Status.Error ||
                                                qnnTransferState is ModelDownloadService.DownloadState.Error
                                            ) {
                                                "Retry"
                                            } else {
                                                "Download"
                                            },
                                        )
                                    }
                                }
                            }
                        }

                        val transfer = qnnTransferState
                        if (
                            transfer is ModelDownloadService.DownloadState.Downloading ||
                            transfer is ModelDownloadService.DownloadState.Paused
                        ) {
                            val progress = when (transfer) {
                                is ModelDownloadService.DownloadState.Downloading -> transfer.progress
                                is ModelDownloadService.DownloadState.Paused -> transfer.progress
                                else -> 0f
                            }
                            val downloaded = when (transfer) {
                                is ModelDownloadService.DownloadState.Downloading ->
                                    transfer.downloadedBytes
                                is ModelDownloadService.DownloadState.Paused ->
                                    transfer.downloadedBytes
                                else -> 0L
                            }
                            val total = when (transfer) {
                                is ModelDownloadService.DownloadState.Downloading ->
                                    transfer.totalBytes
                                is ModelDownloadService.DownloadState.Paused ->
                                    transfer.totalBytes
                                else -> 0L
                            }
                            val speed = (
                                transfer as? ModelDownloadService.DownloadState.Downloading
                                )?.bytesPerSecond ?: 0L
                            val eta = (
                                transfer as? ModelDownloadService.DownloadState.Downloading
                                )?.etaSeconds
                            val currentFile = when (transfer) {
                                is ModelDownloadService.DownloadState.Downloading ->
                                    transfer.currentFileName
                                is ModelDownloadService.DownloadState.Paused ->
                                    transfer.currentFileName
                                else -> null
                            }
                            val paused = transfer is ModelDownloadService.DownloadState.Paused

                            SmoothLinearWavyProgressIndicator(
                                progress = progress,
                                modifier = Modifier.fillMaxWidth(),
                            )

                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                horizontalArrangement = Arrangement.SpaceBetween,
                            ) {
                                Text(
                                    String.format(
                                        Locale.US,
                                        "%d%% · %.0f / %.0f MB",
                                        (progress * 100f).toInt(),
                                        downloaded / 1048576.0,
                                        total / 1048576.0,
                                    ),
                                    style = MaterialTheme.typography.labelSmall,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                                )
                                if (!paused && speed > 0L) {
                                    Text(
                                        buildString {
                                            append(
                                                String.format(
                                                    Locale.US,
                                                    "%.1f MB/s",
                                                    speed / 1048576.0,
                                                ),
                                            )
                                            eta?.takeIf { it > 0L }?.let {
                                                append(" · ")
                                                if (it >= 60L) {
                                                    append("${it / 60L}m")
                                                } else {
                                                    append("${it}s")
                                                }
                                                append(" left")
                                            }
                                        },
                                        style = MaterialTheme.typography.labelSmall,
                                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                                    )
                                }
                            }

                            currentFile?.let {
                                Text(
                                    it,
                                    style = MaterialTheme.typography.labelSmall,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                                    maxLines = 1,
                                )
                            }

                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                horizontalArrangement = Arrangement.spacedBy(8.dp),
                            ) {
                                TextButton(
                                    onClick = {
                                        context.startService(
                                            Intent(
                                                context,
                                                ModelDownloadService::class.java,
                                            ).setAction(
                                                if (paused) {
                                                    ModelDownloadService.ACTION_RESUME_DOWNLOAD
                                                } else {
                                                    ModelDownloadService.ACTION_PAUSE_DOWNLOAD
                                                },
                                            ),
                                        )
                                    },
                                ) {
                                    Icon(
                                        if (paused) Icons.Default.PlayArrow else Icons.Default.Pause,
                                        contentDescription = null,
                                    )
                                    Spacer(Modifier.width(6.dp))
                                    Text(if (paused) "Resume" else "Pause")
                                }
                                TextButton(
                                    onClick = {
                                        context.startService(
                                            Intent(
                                                context,
                                                ModelDownloadService::class.java,
                                            ).setAction(
                                                ModelDownloadService.ACTION_CANCEL_DOWNLOAD,
                                            ),
                                        )
                                    },
                                ) {
                                    Icon(Icons.Default.Close, contentDescription = null)
                                    Spacer(Modifier.width(6.dp))
                                    Text("Cancel")
                                }
                            }
                        }

                        Text(
                            "Separate from the Breeze GGUF. Uses the proven SM8850 batch-1 QNN depth context with serial CFG branches plus QNN backbone prompt/AR1 contexts. Hexagon remains the fallback.",
                            style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                    }
                }

                val current = sorted.firstOrNull { it.id == selectedId } ?: selected
                Surface(
                    shape = MaterialTheme.shapes.extraLarge,
                    color = MaterialTheme.colorScheme.surfaceContainerHigh,
                ) {
                    Column(
                        modifier = Modifier.padding(16.dp),
                        verticalArrangement = Arrangement.spacedBy(10.dp),
                    ) {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Column(modifier = Modifier.weight(1f)) {
                                Text(
                                    "Breeze TTS 2 · " + current.variantPrecision,
                                    style = MaterialTheme.typography.titleMedium,
                                    fontWeight = FontWeight.SemiBold,
                                )
                                Text(
                                    current.variantProfile,
                                    style = MaterialTheme.typography.bodySmall,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                                )
                            }
                            if (current.isDownloaded) {
                                Icon(
                                    Icons.Default.CheckCircle,
                                    contentDescription = "Installed",
                                    tint = MaterialTheme.colorScheme.primary,
                                )
                            }
                        }
                        Text(
                            when (current.variantPrecision) {
                                "Q8_0" ->
                                    "Recommended. Near-lossless against F16 while keeping the full depth decoder at Q8."
                                "Q6_K" ->
                                    "Balanced memory and quality with the full depth decoder preserved."
                                "Q4_K" ->
                                    "Smallest normal model. Good mobile footprint without the experimental depth quant."
                                "F16" ->
                                    "Reference-quality model with the highest memory requirement."
                                else ->
                                    "Experimental depth-quantized build intended for faster short utterances."
                            },
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween,
                        ) {
                            Text(
                                "Download",
                                color = MaterialTheme.colorScheme.onSurfaceVariant,
                            )
                            Text(
                                current.approximateSize,
                                fontWeight = FontWeight.SemiBold,
                            )
                        }
                    }
                }

                Button(
                    onClick = {
                        showSheet = false
                        if (current.isDownloaded) onOpen(current) else onDownload(current)
                    },
                    modifier = Modifier.fillMaxWidth(),
                    shape = MaterialTheme.shapes.large,
                ) {
                    Icon(
                        if (current.isDownloaded) {
                            Icons.Default.GraphicEq
                        } else {
                            Icons.Default.CloudDownload
                        },
                        contentDescription = null,
                    )
                    Spacer(Modifier.width(8.dp))
                    Text(
                        if (current.isDownloaded) {
                            "Open text to speech"
                        } else {
                            "Download " + current.variantPrecision + " · " +
                                current.approximateSize
                        },
                    )
                }
            }
        }
    }
}
