package io.github.xororz.localdream.ui.glass

import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.spring
import androidx.compose.foundation.ExperimentalFoundationApi
import androidx.compose.foundation.combinedClickable
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.interaction.collectIsPressedAsState
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.outlined.ArrowForward
import androidx.compose.material.icons.outlined.CheckCircle
import androidx.compose.material.icons.outlined.CloudDownload
import androidx.compose.material.icons.outlined.PushPin
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import io.github.xororz.localdream.data.Model

/** Dense, interactive model catalog entry. Classic continues to use the old code. */
@OptIn(ExperimentalFoundationApi::class)
@Composable
fun CompactModelCard(
    model: Model,
    isSelected: Boolean,
    isSelectionMode: Boolean,
    isPinned: Boolean,
    onClick: () -> Unit,
    onLongClick: () -> Unit,
    modifier: Modifier = Modifier,
) {
    val interactions = remember { MutableInteractionSource() }
    val pressed by interactions.collectIsPressedAsState()
    val pressure by animateFloatAsState(
        if (pressed) .975f else 1f,
        spring(dampingRatio = .72f, stiffness = 550f),
        label = "model-card-pressure",
    )
    val disabled = isSelectionMode && !model.isDownloaded
    val foreground = if (disabled) MaterialTheme.colorScheme.onSurfaceVariant else MaterialTheme.colorScheme.onSurface
    val secondary = MaterialTheme.colorScheme.onSurfaceVariant
    DreamGlassSurface(
        modifier = modifier.fillMaxWidth().graphicsLayer {
            scaleX = pressure
            scaleY = pressure
            alpha = if (disabled) .5f else 1f
        }.clip(RoundedCornerShape(25.dp)).combinedClickable(
            interactionSource = interactions,
            indication = null,
            enabled = !disabled,
            onClick = onClick,
            onLongClick = { if (model.isDownloaded && !isSelectionMode) onLongClick() },
        ),
        shape = RoundedCornerShape(25.dp),
        role = if (isSelected) DreamGlassRole.SELECTION else DreamGlassRole.CARD,
    ) {
        Row(
            modifier = Modifier.fillMaxWidth().padding(horizontal = 15.dp, vertical = 12.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            Surface(
                shape = RoundedCornerShape(17.dp),
                color = MaterialTheme.colorScheme.primary.copy(alpha = if (model.isDownloaded) .14f else .08f),
                modifier = Modifier.size(50.dp),
            ) {
                Box(contentAlignment = Alignment.Center) {
                    Icon(
                        if (model.isDownloaded) Icons.Outlined.CheckCircle else Icons.Outlined.CloudDownload,
                        contentDescription = null,
                        tint = MaterialTheme.colorScheme.primary,
                        modifier = Modifier.size(23.dp),
                    )
                }
            }
            Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(3.dp)) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text(
                        model.name,
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis,
                        style = MaterialTheme.typography.titleMedium,
                        color = foreground,
                        fontWeight = FontWeight.SemiBold,
                        modifier = Modifier.weight(1f, fill = false),
                    )
                    if (isPinned) {
                        Spacer(Modifier.width(6.dp))
                        Icon(Icons.Outlined.PushPin, "Pinned", Modifier.size(12.dp), tint = secondary)
                    }
                }
                Text(
                    model.description,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis,
                    style = MaterialTheme.typography.bodySmall,
                    color = secondary,
                )
                Row(horizontalArrangement = Arrangement.spacedBy(9.dp)) {
                    Text(if (model.runOnCpu) "CPU" else "NPU", style = MaterialTheme.typography.labelSmall, color = secondary)
                    Text(model.approximateSize, style = MaterialTheme.typography.labelSmall, color = secondary)
                    Text(
                        if (model.isDownloaded) "Ready" else "Download",
                        style = MaterialTheme.typography.labelSmall,
                        color = if (model.isDownloaded) MaterialTheme.colorScheme.primary else secondary,
                    )
                }
            }
            Icon(
                if (isSelected) Icons.Outlined.CheckCircle else Icons.Outlined.ArrowForward,
                contentDescription = if (isSelected) "Selected" else "Open model",
                tint = MaterialTheme.colorScheme.primary,
                modifier = Modifier.size(19.dp),
            )
        }
    }
}
