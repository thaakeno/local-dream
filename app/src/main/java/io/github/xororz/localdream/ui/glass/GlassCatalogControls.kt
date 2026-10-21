package io.github.xororz.localdream.ui.glass

import androidx.compose.foundation.clickable
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.outlined.Close
import androidx.compose.material.icons.outlined.Search
import androidx.compose.material.icons.outlined.Sort
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import io.github.xororz.localdream.ui.components.CatalogFilterMode
import io.github.xororz.localdream.ui.components.CatalogSortMode

/** Scrollable, accessible model discovery instead of stacked outlined fields. */
@Composable
fun GlassCatalogControls(
    query: String,
    onQueryChange: (String) -> Unit,
    filter: CatalogFilterMode,
    onFilterChange: (CatalogFilterMode) -> Unit,
    sort: CatalogSortMode,
    onSortChange: (CatalogSortMode) -> Unit,
    modifier: Modifier = Modifier,
) {
    var sortOpen by remember { mutableStateOf(false) }
    val foreground = MaterialTheme.colorScheme.onSurface
    val dim = MaterialTheme.colorScheme.onSurfaceVariant
    Column(modifier.fillMaxWidth(), verticalArrangement = Arrangement.spacedBy(12.dp)) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text("EXPLORE MODELS", style = MaterialTheme.typography.labelMedium,
                fontWeight = FontWeight.SemiBold, color = dim)
            Spacer(Modifier.weight(1f))
            Box {
                Row(
                    modifier = Modifier.clickable { sortOpen = true }
                        .padding(horizontal = 9.dp, vertical = 6.dp),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(6.dp),
                ) {
                    Icon(Icons.Outlined.Sort, null, Modifier.size(17.dp), tint = dim)
                    Text(sort.label, style = MaterialTheme.typography.labelMedium, color = dim)
                }
                DropdownMenu(expanded = sortOpen, onDismissRequest = { sortOpen = false }) {
                    CatalogSortMode.entries.forEach { option ->
                        DropdownMenuItem(
                            text = { Text(option.label) },
                            onClick = { sortOpen = false; onSortChange(option) },
                        )
                    }
                }
            }
        }
        DreamGlassSurface(
            modifier = Modifier.fillMaxWidth().height(60.dp),
            shape = RoundedCornerShape(28.dp),
            role = DreamGlassRole.CONTROL,
        ) {
            Row(
                modifier = Modifier.fillMaxSize().padding(horizontal = 18.dp),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(12.dp),
            ) {
                Icon(Icons.Outlined.Search, null, Modifier.size(23.dp), tint = dim)
                BasicTextField(
                    value = query,
                    onValueChange = onQueryChange,
                    modifier = Modifier.weight(1f),
                    singleLine = true,
                    textStyle = MaterialTheme.typography.bodyLarge.copy(color = foreground),
                    cursorBrush = SolidColor(foreground),
                    decorationBox = { inner ->
                        Box {
                            if (query.isEmpty()) {
                                Text("Find a model", color = dim,
                                    style = MaterialTheme.typography.bodyLarge)
                            }
                            inner()
                        }
                    },
                )
                if (query.isNotEmpty()) {
                    IconButton(onClick = { onQueryChange("") }) {
                        Icon(Icons.Outlined.Close, "Clear search",
                            Modifier.size(20.dp), tint = dim)
                    }
                }
            }
        }
        DreamGlassSurface(
            modifier = Modifier.fillMaxWidth().height(48.dp),
            shape = RoundedCornerShape(24.dp),
            role = DreamGlassRole.NAVIGATION,
        ) {
            Row(
                Modifier.fillMaxSize().horizontalScroll(rememberScrollState())
                    .padding(horizontal = 5.dp, vertical = 4.dp),
                horizontalArrangement = Arrangement.spacedBy(4.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                CatalogFilterMode.entries.forEach { option ->
                    val selected = filter == option
                    Surface(
                        modifier = Modifier.height(40.dp)
                            .clickable { onFilterChange(option) },
                        shape = RoundedCornerShape(20.dp),
                        color = if (selected) foreground.copy(alpha = .17f)
                            else MaterialTheme.colorScheme.surface.copy(alpha = .02f),
                    ) {
                        Box(Modifier.padding(horizontal = 16.dp), contentAlignment = Alignment.Center) {
                            Text(option.label, maxLines = 1, overflow = TextOverflow.Clip,
                                style = MaterialTheme.typography.labelLarge,
                                fontWeight = if (selected) FontWeight.SemiBold else FontWeight.Normal,
                                color = if (selected) foreground else dim)
                        }
                    }
                }
            }
        }
    }
}
