package io.github.xororz.localdream.ui.components

import androidx.compose.animation.AnimatedContent
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.animateContentSize
import androidx.compose.animation.core.tween
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.togetherWith
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.AutoAwesome
import androidx.compose.material.icons.filled.Bolt
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.CloudDownload
import androidx.compose.material.icons.filled.Memory
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
import androidx.compose.ui.Modifier
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
    val installed = variants.count { it.isDownloaded }

    ElevatedCard(
        onClick = { showSheet = true },
        modifier = modifier
            .fillMaxWidth()
            .animateContentSize(),
        shape = MaterialTheme.shapes.extraLarge,
        colors = CardDefaults.elevatedCardColors(
            containerColor = MaterialTheme.colorScheme.primaryContainer.copy(alpha = 0.55f),
        ),
    ) {
        Column(
            modifier = Modifier.padding(18.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(12.dp),
            ) {
                Surface(
                    shape = MaterialTheme.shapes.large,
                    color = MaterialTheme.colorScheme.primary,
                ) {
                    Icon(
                        Icons.Default.AutoAwesome,
                        contentDescription = null,
                        tint = MaterialTheme.colorScheme.onPrimary,
                        modifier = Modifier.padding(12.dp).size(24.dp),
                    )
                }
                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        "Qwen Image 2.1",
                        style = MaterialTheme.typography.titleLarge,
                        fontWeight = FontWeight.SemiBold,
                    )
                    Text(
                        "One model family • choose precision + Turbo adapter",
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
                Badge {
                    Text(if (installed > 0) "$installed installed" else "NPU")
                }
            }

            Row(
                modifier = Modifier.horizontalScroll(rememberScrollState()),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                AssistChip(
                    onClick = { showSheet = true },
                    label = { Text("Q4_0") },
                    leadingIcon = { Icon(Icons.Default.SdStorage, null) },
                )
                AssistChip(
                    onClick = { showSheet = true },
                    label = { Text("Q8_0") },
                    leadingIcon = { Icon(Icons.Default.Memory, null) },
                )
                AssistChip(
                    onClick = { showSheet = true },
                    label = { Text("FP8") },
                    leadingIcon = { Icon(Icons.Default.Bolt, null) },
                )
                AssistChip(
                    onClick = { showSheet = true },
                    label = { Text("Viggle 6-pass") },
                    leadingIcon = { Icon(Icons.Default.Speed, null) },
                )
            }

            Text(
                "Native 1024 • no upscaling • Q8 quality / FP8 speed / Q4 compact",
                style = MaterialTheme.typography.labelMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
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

    val precisions = listOf("Q4_0", "Q8_0", "FP8").filter { p ->
        variants.any { it.variantPrecision == p }
    }
    val adapters = listOf("", "r128", "r256").filter { a ->
        variants.any { it.variantAdapter == a }
    }
    val selected = variants.firstOrNull {
        it.variantPrecision == precision && it.variantAdapter == adapter
    } ?: variants.first()

    ModalBottomSheet(onDismissRequest = onDismiss) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 22.dp)
                .padding(bottom = 30.dp),
            verticalArrangement = Arrangement.spacedBy(18.dp),
        ) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Surface(
                    shape = MaterialTheme.shapes.large,
                    color = MaterialTheme.colorScheme.primaryContainer,
                ) {
                    Icon(
                        Icons.Default.Tune,
                        null,
                        modifier = Modifier.padding(12.dp).size(24.dp),
                        tint = MaterialTheme.colorScheme.onPrimaryContainer,
                    )
                }
                Spacer(Modifier.width(12.dp))
                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        "Configure Qwen Image 2.1",
                        style = MaterialTheme.typography.headlineSmall,
                        fontWeight = FontWeight.SemiBold,
                    )
                    Text(
                        "Pick the visual transformer and optional Turbo LoRA.",
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
            }

            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Text("Transformer precision", style = MaterialTheme.typography.titleSmall)
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .horizontalScroll(rememberScrollState()),
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                ) {
                    precisions.forEach { p ->
                        val subtitle = when (p) {
                            "Q4_0" -> "4.20 GB • compact"
                            "Q8_0" -> "7.69 GB • quality"
                            else -> "7.12 GB • fast"
                        }
                        FilterChip(
                            selected = precision == p,
                            onClick = { precision = p },
                            label = { Text("$p  $subtitle") },
                            leadingIcon = {
                                Icon(
                                    when (p) {
                                        "Q4_0" -> Icons.Default.SdStorage
                                        "Q8_0" -> Icons.Default.Memory
                                        else -> Icons.Default.Bolt
                                    },
                                    null,
                                )
                            },
                        )
                    }
                }
            }

            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Text("Turbo adapter", style = MaterialTheme.typography.titleSmall)
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .horizontalScroll(rememberScrollState()),
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                ) {
                    adapters.forEach { a ->
                        val label = when (a) {
                            "r128" -> "Viggle r128 • 680 MB"
                            "r256" -> "Viggle r256 • 1.36 GB"
                            else -> "Off • base model"
                        }
                        FilterChip(
                            selected = adapter == a,
                            onClick = { adapter = a },
                            label = { Text(label) },
                            leadingIcon = {
                                Icon(
                                    if (a.isEmpty()) Icons.Default.PlayArrow else Icons.Default.AutoAwesome,
                                    null,
                                )
                            },
                        )
                    }
                }
            }

            AnimatedContent(
                targetState = selected.id,
                transitionSpec = {
                    fadeIn(tween(180)) togetherWith fadeOut(tween(120))
                },
                label = "qwenVariantDetails",
            ) {
                Surface(
                    shape = MaterialTheme.shapes.extraLarge,
                    color = MaterialTheme.colorScheme.surfaceContainerHigh,
                ) {
                    Column(
                        modifier = Modifier.padding(16.dp),
                        verticalArrangement = Arrangement.spacedBy(10.dp),
                    ) {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Text(
                                selected.name,
                                style = MaterialTheme.typography.titleMedium,
                                fontWeight = FontWeight.SemiBold,
                                modifier = Modifier.weight(1f),
                            )
                            if (selected.isDownloaded) {
                                Icon(
                                    Icons.Default.CheckCircle,
                                    contentDescription = null,
                                    tint = MaterialTheme.colorScheme.primary,
                                )
                            }
                        }
                        Text(
                            selected.description,
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )

                        ListItem(
                            colors = ListItemDefaults.colors(containerColor = Color.Transparent),
                            leadingContent = { Icon(Icons.Default.Memory, null) },
                            headlineContent = { Text("Visual transformer") },
                            supportingContent = {
                                Text("${selected.variantPrecision} ${selected.variantFormat}")
                            },
                            trailingContent = { Text(gb(selected.modelBytes)) },
                        )
                        ListItem(
                            colors = ListItemDefaults.colors(containerColor = Color.Transparent),
                            leadingContent = { Icon(Icons.Default.AutoAwesome, null) },
                            headlineContent = { Text("Turbo LoRA") },
                            supportingContent = {
                                Text(
                                    if (selected.variantAdapter.isEmpty()) {
                                        "Disabled • 20-step base path"
                                    } else {
                                        "Viggle v0.2.1 ${selected.variantAdapter} • exact 6-pass"
                                    },
                                )
                            },
                            trailingContent = {
                                Text(if (selected.adapterBytes > 0) gb(selected.adapterBytes) else "—")
                            },
                        )
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween,
                        ) {
                            Text(
                                "Full package",
                                style = MaterialTheme.typography.labelLarge,
                                color = MaterialTheme.colorScheme.onSurfaceVariant,
                            )
                            Text(
                                selected.approximateSize,
                                style = MaterialTheme.typography.titleMedium,
                                fontWeight = FontWeight.SemiBold,
                            )
                        }

                        AnimatedVisibility(visible = selected.recommendedVariant) {
                            Text(
                                "Recommended quality/mobile balance",
                                style = MaterialTheme.typography.labelMedium,
                                color = MaterialTheme.colorScheme.primary,
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
            ) {
                Icon(
                    if (selected.isDownloaded) Icons.Default.PlayArrow else Icons.Default.CloudDownload,
                    contentDescription = null,
                )
                Spacer(Modifier.width(8.dp))
                Text(
                    if (selected.isDownloaded) {
                        "Open ${selected.variantPrecision}"
                    } else {
                        "Download ${selected.approximateSize}"
                    },
                )
            }

            TextButton(onClick = onDismiss, modifier = Modifier.fillMaxWidth()) {
                Text("Close")
            }
        }
    }
}
