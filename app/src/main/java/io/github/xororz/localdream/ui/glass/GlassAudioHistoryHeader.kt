package io.github.xororz.localdream.ui.glass

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.outlined.Search
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp

/** Unboxed, editorial History header. No giant Material panel over the list. */
@Composable
fun GlassAudioHistoryHeader(
    title: String,
    description: String,
    count: Int,
    query: String,
    onQueryChange: (String) -> Unit,
    favorites: Boolean,
    onFavoritesChange: () -> Unit,
    newest: Boolean,
    onSortChange: () -> Unit,
    onRefresh: () -> Unit,
) {
    val foreground = MaterialTheme.colorScheme.onSurface
    val muted = MaterialTheme.colorScheme.onSurfaceVariant
    Column(
        modifier = Modifier.fillMaxWidth().padding(horizontal = 20.dp, vertical = 14.dp),
        verticalArrangement = Arrangement.spacedBy(13.dp),
    ) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(3.dp)) {
                Text("YOUR COLLECTION", style = MaterialTheme.typography.labelSmall,
                    color = muted, fontWeight = FontWeight.SemiBold)
                Text(title, style = MaterialTheme.typography.headlineMedium,
                    color = foreground, fontWeight = FontWeight.SemiBold)
                Text(description, style = MaterialTheme.typography.bodySmall, color = muted)
            }
            Text(count.toString(), style = MaterialTheme.typography.headlineLarge,
                color = foreground)
        }
        DreamGlassSurface(
            modifier = Modifier.fillMaxWidth().height(56.dp),
            shape = RoundedCornerShape(28.dp),
            role = DreamGlassRole.CONTROL,
        ) {
            Row(
                Modifier.fillMaxSize().padding(horizontal = 18.dp),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(12.dp),
            ) {
                Icon(Icons.Outlined.Search, null, Modifier.size(21.dp), tint = muted)
                BasicTextField(
                    value = query,
                    onValueChange = onQueryChange,
                    modifier = Modifier.weight(1f),
                    singleLine = true,
                    textStyle = MaterialTheme.typography.bodyMedium.copy(color = foreground),
                    cursorBrush = SolidColor(foreground),
                    decorationBox = { inner ->
                        Box {
                            if (query.isBlank()) Text("Search your history",
                                color = muted, style = MaterialTheme.typography.bodyMedium)
                            inner()
                        }
                    },
                )
            }
        }
        Row(
            Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(7.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            GlassHistoryAction("Favorites", favorites, onFavoritesChange)
            GlassHistoryAction(if (newest) "Newest" else "Oldest", false, onSortChange)
            Spacer(Modifier.weight(1f))
            TextButton(onClick = onRefresh) {
                Text("Refresh", style = MaterialTheme.typography.labelMedium)
            }
        }
    }
}

@Composable
private fun GlassHistoryAction(label: String, selected: Boolean, onClick: () -> Unit) {
    DreamGlassSurface(
        modifier = Modifier.clickable(onClick = onClick).height(40.dp),
        shape = RoundedCornerShape(20.dp),
        role = if (selected) DreamGlassRole.SELECTION else DreamGlassRole.CONTROL,
    ) {
        Box(Modifier.padding(horizontal = 14.dp), contentAlignment = Alignment.Center) {
            Text(label, style = MaterialTheme.typography.labelMedium,
                color = MaterialTheme.colorScheme.onSurface,
                fontWeight = if (selected) FontWeight.SemiBold else FontWeight.Normal)
        }
    }
}
