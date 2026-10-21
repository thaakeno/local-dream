package io.github.xororz.localdream.ui.glass

import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.spring
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.ExperimentalFoundationApi
import androidx.compose.foundation.combinedClickable
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.interaction.collectIsPressedAsState
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.outlined.ArrowForward
import androidx.compose.material.icons.outlined.CheckCircle
import androidx.compose.material.icons.outlined.PushPin
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import io.github.xororz.localdream.data.Model
import io.github.xororz.localdream.ui.glass.backdrop.backdrops.layerBackdrop
import io.github.xororz.localdream.ui.glass.backdrop.backdrops.rememberLayerBackdrop

/** One model, one individually captured material scene, never a painted blue tile. */
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
        targetValue = if (pressed) .982f else 1f,
        animationSpec = spring(dampingRatio = .76f, stiffness = 520f),
        label = "model-press",
    )
    val disabled = isSelectionMode && !model.isDownloaded
    val shape = RoundedCornerShape(30.dp)
    val backdrop = rememberLayerBackdrop()
    val foreground = MaterialTheme.colorScheme.onSurface
    val muted = MaterialTheme.colorScheme.onSurfaceVariant
    val dark = MaterialTheme.colorScheme.background.red < .5f

    CompositionLocalProvider(LocalDreamBackdrop provides backdrop) {
        Box(
            modifier.fillMaxWidth().height(155.dp)
                .graphicsLayer {
                    scaleX = pressure
                    scaleY = pressure
                    alpha = if (disabled) .52f else 1f
                }
                .clip(shape),
        ) {
            Canvas(Modifier.matchParentSize().layerBackdrop(backdrop)) {
                drawRect(
                    brush = Brush.linearGradient(
                        if (dark) listOf(
                            Color(0xFF161618), Color(0xFF262627), Color(0xFF111112)
                        ) else listOf(
                            Color(0xFFE3E1DD), Color(0xFFF7F6F3), Color(0xFFE6E4E0)
                        ),
                    ),
                )
                val point = Offset(size.width * .91f, size.height * .08f)
                val ink = if (dark) Color.White else Color.Black
                for (i in 1..9) {
                    drawLine(
                        color = ink.copy(alpha = .035f + i * .003f),
                        start = Offset(point.x - i * size.width * .035f, 0f),
                        end = Offset(size.width, size.height * (.13f + .08f * i)),
                        strokeWidth = 1.dp.toPx(),
                    )
                }
            }
            DreamGlassSurface(
                modifier = Modifier.matchParentSize().combinedClickable(
                    interactionSource = interactions,
                    indication = null,
                    enabled = !disabled,
                    onClick = onClick,
                    onLongClick = {
                        if (model.isDownloaded && !isSelectionMode) onLongClick()
                    },
                ),
                shape = shape,
                role = if (isSelected) DreamGlassRole.SELECTION else DreamGlassRole.CARD,
            ) {
                Column(
                    Modifier.fillMaxWidth().padding(horizontal = 21.dp, vertical = 16.dp),
                    verticalArrangement = Arrangement.spacedBy(6.dp),
                ) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Text(
                            if (model.runOnCpu) "CPU  /  MODEL" else "NPU  /  MODEL",
                            style = MaterialTheme.typography.labelSmall,
                            color = muted,
                        )
                        if (isPinned) {
                            Spacer(Modifier.width(8.dp))
                            Icon(Icons.Outlined.PushPin, "Pinned", Modifier.size(13.dp), tint = muted)
                        }
                        Spacer(Modifier.weight(1f))
                        Text(
                            if (model.isDownloaded) "INSTALLED" else "AVAILABLE",
                            style = MaterialTheme.typography.labelSmall,
                            color = foreground.copy(alpha = .88f),
                        )
                    }
                    Spacer(Modifier.height(3.dp))
                    Text(
                        model.name,
                        style = MaterialTheme.typography.headlineSmall,
                        fontWeight = FontWeight.SemiBold,
                        color = foreground,
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis,
                    )
                    Text(
                        model.description,
                        style = MaterialTheme.typography.bodySmall,
                        color = muted,
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis,
                    )
                    Spacer(Modifier.weight(1f))
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Text(model.approximateSize, style = MaterialTheme.typography.labelMedium,
                            color = muted)
                        Spacer(Modifier.weight(1f))
                        Text(
                            if (model.isDownloaded) "Open" else "Get model",
                            style = MaterialTheme.typography.labelLarge,
                            color = foreground,
                        )
                        Spacer(Modifier.width(7.dp))
                        Icon(
                            if (isSelected) Icons.Outlined.CheckCircle else Icons.Outlined.ArrowForward,
                            if (isSelected) "Selected" else "Open model",
                            Modifier.size(18.dp),
                            tint = foreground,
                        )
                    }
                }
            }
        }
    }
}
