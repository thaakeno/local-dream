package io.github.xororz.localdream.ui.components

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
import androidx.compose.foundation.background
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.AutoAwesome
import androidx.compose.material.icons.filled.Bolt
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.CloudDownload
import androidx.compose.material.icons.filled.GraphicEq
import androidx.compose.material.icons.filled.Memory
import androidx.compose.material.icons.filled.MusicNote
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
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.ListItem
import androidx.compose.material3.ListItemDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import io.github.xororz.localdream.data.Model
import java.util.Locale

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
private fun FamilyStatusPill(
    text: String,
    emphasized: Boolean = false,
) {
    Surface(
        shape = MaterialTheme.shapes.extraLarge,
        color = if (emphasized) {
            MaterialTheme.colorScheme.primaryContainer
        } else {
            MaterialTheme.colorScheme.surfaceContainerHighest
        },
    ) {
        Text(
            text = text,
            modifier = Modifier.padding(horizontal = 10.dp, vertical = 6.dp),
            style = MaterialTheme.typography.labelMedium,
            color = if (emphasized) {
                MaterialTheme.colorScheme.onPrimaryContainer
            } else {
                MaterialTheme.colorScheme.onSurfaceVariant
            },
        )
    }
}

@Composable
private fun AnimatedFamilyHero(
    title: String,
    subtitle: String,
    badge: String,
    music: Boolean,
    modifier: Modifier = Modifier,
) {
    val transition = rememberInfiniteTransition(label = if (music) "yueHero" else "qwenHero")
    val phase by transition.animateFloat(
        initialValue = 0f,
        targetValue = 1f,
        animationSpec = infiniteRepeatable(
            animation = tween(5200, easing = LinearEasing),
            repeatMode = RepeatMode.Reverse,
        ),
        label = "gradientPhase",
    )
    val primary = MaterialTheme.colorScheme.primary
    val secondary = MaterialTheme.colorScheme.secondary
    val tertiary = MaterialTheme.colorScheme.tertiary
    val gradient = if (music) {
        Brush.linearGradient(
            colors = listOf(
                tertiary.copy(alpha = 0.88f),
                secondary.copy(alpha = 0.72f),
                primary.copy(alpha = 0.58f),
            ),
            start = Offset(phase * 420f, 0f),
            end = Offset(920f - phase * 260f, 520f),
        )
    } else {
        Brush.linearGradient(
            colors = listOf(
                primary.copy(alpha = 0.90f),
                secondary.copy(alpha = 0.72f),
                tertiary.copy(alpha = 0.55f),
            ),
            start = Offset(0f, phase * 260f),
            end = Offset(900f, 420f - phase * 170f),
        )
    }

    Box(
        modifier = modifier
            .fillMaxWidth()
            .background(gradient, MaterialTheme.shapes.extraLarge)
            .padding(horizontal = 18.dp, vertical = 18.dp),
    ) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(14.dp),
        ) {
            Surface(
                modifier = Modifier.size(58.dp),
                shape = MaterialTheme.shapes.extraLarge,
                color = Color.White.copy(alpha = 0.16f),
            ) {
                if (music) {
                    AnimatedMusicMark(Modifier.padding(10.dp))
                } else {
                    Icon(
                        Icons.Default.AutoAwesome,
                        contentDescription = null,
                        tint = Color.White,
                        modifier = Modifier.padding(15.dp),
                    )
                }
            }
            Column(modifier = Modifier.weight(1f)) {
                Text(
                    title,
                    style = MaterialTheme.typography.titleLarge,
                    fontWeight = FontWeight.Bold,
                    color = Color.White,
                )
                Spacer(Modifier.height(2.dp))
                Text(
                    subtitle,
                    style = MaterialTheme.typography.bodyMedium,
                    color = Color.White.copy(alpha = 0.82f),
                )
            }
            Surface(
                shape = MaterialTheme.shapes.extraLarge,
                color = Color.White.copy(alpha = 0.16f),
            ) {
                Text(
                    badge,
                    modifier = Modifier.padding(horizontal = 11.dp, vertical = 7.dp),
                    style = MaterialTheme.typography.labelLarge,
                    color = Color.White,
                )
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

@Composable
private fun FamilyVariantTile(
    selected: Boolean,
    installed: Boolean,
    title: String,
    subtitle: String,
    icon: androidx.compose.ui.graphics.vector.ImageVector,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
) {
    ElevatedCard(
        onClick = onClick,
        modifier = modifier,
        shape = MaterialTheme.shapes.extraLarge,
        colors = CardDefaults.elevatedCardColors(
            containerColor = if (selected) {
                MaterialTheme.colorScheme.primaryContainer
            } else {
                MaterialTheme.colorScheme.surfaceContainerHigh
            },
        ),
        elevation = CardDefaults.elevatedCardElevation(
            defaultElevation = if (selected) 4.dp else 0.dp,
        ),
    ) {
        Column(
            modifier = Modifier.padding(13.dp),
            verticalArrangement = Arrangement.spacedBy(6.dp),
        ) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Icon(
                    icon,
                    contentDescription = null,
                    modifier = Modifier.size(19.dp),
                    tint = if (selected) {
                        MaterialTheme.colorScheme.primary
                    } else {
                        MaterialTheme.colorScheme.onSurfaceVariant
                    },
                )
                Spacer(Modifier.width(7.dp))
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
                        modifier = Modifier.size(18.dp),
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
            if (installed) {
                Text(
                    "On device",
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.primary,
                    fontWeight = FontWeight.SemiBold,
                )
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun QwenFamilyCard(
    variants: List<Model>,
    onOpen: (Model) -> Unit,
    onDownload: (Model) -> Unit,
    modifier: Modifier = Modifier,
) {
    if (variants.isEmpty()) return

    var showSheet by remember { mutableStateOf(false) }
    val installedPrecisions = listOf("Q4_0", "Q8_0", "FP8").filter { precision ->
        variants.any {
            it.variantPrecision == precision &&
                it.variantAdapter.isEmpty() &&
                it.isDownloaded
        }
    }
    val r128Installed = variants.any { it.variantAdapter == "r128" && it.isDownloaded }
    val r256Installed = variants.any { it.variantAdapter == "r256" && it.isDownloaded }
    val recommended = variants.firstOrNull { it.recommendedVariant }
        ?: variants.firstOrNull { it.isDownloaded }
        ?: variants.first()

    ElevatedCard(
        onClick = { showSheet = true },
        modifier = modifier.fillMaxWidth(),
        shape = MaterialTheme.shapes.extraLarge,
        colors = CardDefaults.elevatedCardColors(
            containerColor = MaterialTheme.colorScheme.surfaceContainer,
        ),
    ) {
        Column(verticalArrangement = Arrangement.spacedBy(0.dp)) {
            AnimatedFamilyHero(
                title = "Qwen Image 2.1",
                subtitle = "Image generation · Hexagon HTP",
                badge = if (installedPrecisions.isEmpty()) {
                    "NPU"
                } else {
                    "${installedPrecisions.size}/3 ready"
                },
                music = false,
            )

            Column(
                modifier = Modifier.padding(18.dp),
                verticalArrangement = Arrangement.spacedBy(14.dp),
            ) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                ) {
                    listOf(
                        Triple("Q4_0", "Q4", "Compact"),
                        Triple("Q8_0", "Q8", "Quality"),
                        Triple("FP8", "FP8", "Fast"),
                    ).forEach { (precision, label, note) ->
                        val installed = precision in installedPrecisions
                        Surface(
                            shape = MaterialTheme.shapes.large,
                            color = if (installed) {
                                MaterialTheme.colorScheme.primaryContainer.copy(alpha = 0.58f)
                            } else {
                                MaterialTheme.colorScheme.surfaceContainerHigh
                            },
                            modifier = Modifier.weight(1f),
                        ) {
                            Column(Modifier.padding(11.dp)) {
                                Row(verticalAlignment = Alignment.CenterVertically) {
                                    Text(
                                        label,
                                        style = MaterialTheme.typography.titleSmall,
                                        fontWeight = FontWeight.SemiBold,
                                    )
                                    if (installed) {
                                        Spacer(Modifier.weight(1f))
                                        Icon(
                                            Icons.Default.CheckCircle,
                                            null,
                                            modifier = Modifier.size(16.dp),
                                            tint = MaterialTheme.colorScheme.primary,
                                        )
                                    }
                                }
                                Text(
                                    note,
                                    style = MaterialTheme.typography.labelSmall,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                                )
                            }
                        }
                    }
                }

                Surface(
                    shape = MaterialTheme.shapes.large,
                    color = MaterialTheme.colorScheme.surfaceContainerHigh,
                ) {
                    Column(
                        modifier = Modifier.padding(14.dp),
                        verticalArrangement = Arrangement.spacedBy(9.dp),
                    ) {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Icon(
                                Icons.Default.Speed,
                                contentDescription = null,
                                tint = MaterialTheme.colorScheme.primary,
                            )
                            Spacer(Modifier.width(8.dp))
                            Text(
                                "Shared Turbo adapters",
                                style = MaterialTheme.typography.titleSmall,
                                fontWeight = FontWeight.SemiBold,
                                modifier = Modifier.weight(1f),
                            )
                            Text(
                                "download once",
                                style = MaterialTheme.typography.labelMedium,
                                color = MaterialTheme.colorScheme.primary,
                            )
                        }
                        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                            FamilyStatusPill(
                                "r128 · " + if (r128Installed) "installed" else "680 MB",
                                emphasized = r128Installed,
                            )
                            FamilyStatusPill(
                                "r256 · " + if (r256Installed) "installed" else "1.36 GB",
                                emphasized = r256Installed,
                            )
                        }
                        Text(
                            "A Viggle LoRA is now shared across Q4, Q8 and FP8 instead of being downloaded again for every transformer.",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                    }
                }

                Row(
                    modifier = Modifier.fillMaxWidth(),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Column(modifier = Modifier.weight(1f)) {
                        Text(
                            "Recommended",
                            style = MaterialTheme.typography.labelMedium,
                            color = MaterialTheme.colorScheme.primary,
                        )
                        Text(
                            "${recommended.variantPrecision}" +
                                if (recommended.variantAdapter.isNotEmpty()) {
                                    " + Viggle ${recommended.variantAdapter}"
                                } else {
                                    " base"
                                },
                            style = MaterialTheme.typography.titleMedium,
                            fontWeight = FontWeight.SemiBold,
                        )
                    }
                    TextButton(onClick = { showSheet = true }) {
                        Text("Configure")
                        Spacer(Modifier.width(5.dp))
                        Icon(Icons.Default.Tune, null)
                    }
                }
            }
        }
    }

    if (showSheet) {
        QwenVariantSheet(
            variants = variants,
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
                .padding(horizontal = 18.dp)
                .padding(bottom = 28.dp),
            verticalArrangement = Arrangement.spacedBy(18.dp),
        ) {
            AnimatedFamilyHero(
                title = "Qwen Image 2.1",
                subtitle = "Build one runtime from shared assets",
                badge = "HTP",
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
                            installed = adapterReady(a),
                            title = when (a) {
                                "r128" -> "r128"
                                "r256" -> "r256"
                                else -> "Base"
                            },
                            subtitle = when (a) {
                                "r128" -> "680 MB · 6-pass"
                                "r256" -> "1.36 GB · 6-pass"
                                else -> "20 steps · no LoRA"
                            },
                            icon = if (a.isEmpty()) Icons.Default.PlayArrow else Icons.Default.Speed,
                            onClick = { adapter = a },
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
                                        "Exact Viggle v0.2.1 six-pass runtime"
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

                        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                            FamilyStatusPill(
                                if (baseReady) "Transformer installed" else "Transformer missing",
                                emphasized = baseReady,
                            )
                            if (adapter.isNotBlank()) {
                                FamilyStatusPill(
                                    if (turboReady) "LoRA already installed" else "LoRA missing",
                                    emphasized = turboReady,
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
    modifier: Modifier = Modifier,
) {
    if (variants.isEmpty()) return
    val initial = variants.firstOrNull { it.isDownloaded && it.recommendedVariant }
        ?: variants.firstOrNull { it.isDownloaded }
        ?: variants.firstOrNull { it.recommendedVariant }
        ?: variants.first()
    var selectedId by remember(variants) { mutableStateOf(initial.id) }
    var showSheet by remember { mutableStateOf(false) }
    val selected = variants.firstOrNull { it.id == selectedId } ?: initial
    val installed = variants.count { it.isDownloaded }

    ElevatedCard(
        onClick = { showSheet = true },
        modifier = modifier.fillMaxWidth(),
        shape = MaterialTheme.shapes.extraLarge,
        colors = CardDefaults.elevatedCardColors(
            containerColor = MaterialTheme.colorScheme.surfaceContainer,
        ),
    ) {
        Column(verticalArrangement = Arrangement.spacedBy(0.dp)) {
            AnimatedFamilyHero(
                title = "YuE2 3B",
                subtitle = "Text to music · 48 kHz stereo",
                badge = if (installed > 0) "$installed ready" else "NPU",
                music = true,
            )

            Column(
                modifier = Modifier.padding(18.dp),
                verticalArrangement = Arrangement.spacedBy(14.dp),
            ) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Icon(
                        Icons.Default.GraphicEq,
                        contentDescription = null,
                        tint = MaterialTheme.colorScheme.tertiary,
                    )
                    Spacer(Modifier.width(8.dp))
                    Column(modifier = Modifier.weight(1f)) {
                        Text(
                            "Choose backbone precision",
                            style = MaterialTheme.typography.titleMedium,
                            fontWeight = FontWeight.SemiBold,
                        )
                        Text(
                            "The F32 Oobleck decoder is shared by every option.",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                    }
                }

                variants.sortedBy {
                    when (it.variantPrecision) {
                        "Q5_K_M" -> 0
                        "Q6_K" -> 1
                        "Q8_0" -> 2
                        else -> 3
                    }
                }.chunked(2).forEach { row ->
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
                                    "Q5_K_M" -> "Compact · 3.15 GB"
                                    "Q6_K" -> "Balanced · 3.47 GB"
                                    "Q8_0" -> "Near-lossless · 4.34 GB"
                                    else -> "BF16 weights · 7.70 GB"
                                },
                                icon = when (variant.variantPrecision) {
                                    "Q5_K_M" -> Icons.Default.SdStorage
                                    "Q6_K" -> Icons.Default.Bolt
                                    "Q8_0" -> Icons.Default.Memory
                                    else -> Icons.Default.AutoAwesome
                                },
                                onClick = {
                                    selectedId = variant.id
                                    showSheet = true
                                },
                                modifier = Modifier.weight(1f),
                            )
                        }
                    }
                }

                Surface(
                    shape = MaterialTheme.shapes.large,
                    color = MaterialTheme.colorScheme.surfaceContainerHigh,
                    onClick = { showSheet = true },
                ) {
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(horizontal = 14.dp, vertical = 12.dp),
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        Column(modifier = Modifier.weight(1f)) {
                            Text(
                                "${selected.variantPrecision} · ${selected.variantProfile}",
                                style = MaterialTheme.typography.titleSmall,
                                fontWeight = FontWeight.SemiBold,
                            )
                            Text(
                                "AR plan → semantic tokens → NAR flow → Oobleck · ≤20s",
                                style = MaterialTheme.typography.labelMedium,
                                color = MaterialTheme.colorScheme.onSurfaceVariant,
                            )
                        }
                        Text(
                            if (selected.isDownloaded) "Create" else "Details",
                            style = MaterialTheme.typography.labelLarge,
                            color = MaterialTheme.colorScheme.primary,
                        )
                        Spacer(Modifier.width(5.dp))
                        Icon(Icons.Default.Tune, null)
                    }
                }
            }
        }
    }

    if (showSheet) {
        Yue2VariantSheet(
            variants = variants,
            selectedId = selected.id,
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
    selectedId: String,
    onSelect: (Model) -> Unit,
    onDismiss: () -> Unit,
    onOpen: (Model) -> Unit,
    onDownload: (Model) -> Unit,
) {
    val sorted = variants.sortedBy {
        when (it.variantPrecision) {
            "Q5_K_M" -> 0
            "Q6_K" -> 1
            "Q8_0" -> 2
            else -> 3
        }
    }
    val selected = variants.firstOrNull { it.id == selectedId }
        ?: variants.firstOrNull { it.recommendedVariant }
        ?: variants.first()

    ModalBottomSheet(onDismissRequest = onDismiss) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 18.dp)
                .padding(bottom = 28.dp),
            verticalArrangement = Arrangement.spacedBy(18.dp),
        ) {
            AnimatedFamilyHero(
                title = "YuE2 3B",
                subtitle = "Local text-to-music on Snapdragon",
                badge = "48 kHz",
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
                                "Q5_K_M" -> "Compact · 3.15 GB"
                                "Q6_K" -> "Balanced · 3.47 GB"
                                "Q8_0" -> "Near-lossless · 4.34 GB"
                                else -> "BF16 weights · 7.70 GB"
                            },
                            icon = when (variant.variantPrecision) {
                                "Q5_K_M" -> Icons.Default.SdStorage
                                "Q6_K" -> Icons.Default.Bolt
                                "Q8_0" -> Icons.Default.Memory
                                else -> Icons.Default.AutoAwesome
                            },
                            onClick = { onSelect(variant) },
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
                                            "BF16 is the tensor data type; GGUF is only the model file/container format."
                                        else ->
                                            "GGUF container · quantized backbone · F32 Oobleck waveform decoder"
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

                        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                            FamilyStatusPill(
                                if (selected.variantPrecision == "BF16") {
                                    "BF16 weights"
                                } else {
                                    selected.variantPrecision
                                },
                                true,
                            )
                            FamilyStatusPill("GGUF container")
                            FamilyStatusPill("F32 VAE")
                        }

                        Text(
                            when (selected.variantPrecision) {
                                "Q5_K_M" ->
                                    "Smallest supported upstream quant. YuE2's audio-code LM is not offered below Q5."
                                "Q6_K" ->
                                    "Balanced mobile option with lower memory/bandwidth than Q8."
                                "Q8_0" ->
                                    "Near-lossless quality-first option and the safest default."
                                else ->
                                    "Full BF16 backbone stored inside a GGUF container. This is not a 'BF16 quant'; it is the least-compressed reference option."
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
