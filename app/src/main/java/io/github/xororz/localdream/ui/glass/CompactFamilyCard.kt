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
import androidx.compose.material.icons.outlined.AutoAwesome
import androidx.compose.material.icons.outlined.ChevronRight
import androidx.compose.material.icons.outlined.GraphicEq
import androidx.compose.material.icons.outlined.MusicNote
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import io.github.xororz.localdream.data.Model
import io.github.xororz.localdream.ui.glass.backdrop.backdrops.layerBackdrop
import io.github.xororz.localdream.ui.glass.backdrop.backdrops.rememberLayerBackdrop

/**
 * Refractive family entry. Its metallic artwork is deliberately recorded into
 * a separate LayerBackdrop BEHIND the text-bearing lens. The lens therefore
 * samples real content without recursive feedback or a fake blue card fill.
 * It is static while inference runs: no perpetual shaders or recompositions.
 */
@OptIn(ExperimentalFoundationApi::class)
@Composable
fun CompactFamilyCard(
    title: String,
    subtitle: String,
    variants: List<Model>,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
) {
    val interactions = remember { MutableInteractionSource() }
    val pressed by interactions.collectIsPressedAsState()
    val scale by animateFloatAsState(
        targetValue = if (pressed) .982f else 1f,
        animationSpec = spring(dampingRatio = .72f, stiffness = 510f),
        label = "family-glass-pressure",
    )
    val source = rememberLayerBackdrop()
    val shape = RoundedCornerShape(32.dp)
    val installed = variants.count { it.isDownloaded }
    val kind = when {
        subtitle.contains("music", ignoreCase = true) -> "MUSIC"
        subtitle.contains("speech", ignoreCase = true) -> "VOICE"
        else -> "IMAGE"
    }
    val mark = when (kind) {
        "MUSIC" -> Icons.Outlined.MusicNote
        "VOICE" -> Icons.Outlined.GraphicEq
        else -> Icons.Outlined.AutoAwesome
    }
    val isDark = MaterialTheme.colorScheme.background.red < .5f
    val foreground = MaterialTheme.colorScheme.onSurface

    CompositionLocalProvider(LocalDreamBackdrop provides source) {
        Box(
            modifier.fillMaxWidth().height(168.dp)
                .graphicsLayer { scaleX = scale; scaleY = scale }
                .clip(shape),
        ) {
            Canvas(Modifier.matchParentSize().layerBackdrop(source)) {
                val light = if (isDark) Color(0xFFFAF6ED) else Color(0xFF595654)
                drawRect(
                    brush = Brush.linearGradient(
                        if (isDark) listOf(
                            Color(0xFF171719), Color(0xFF302F2E), Color(0xFF121213)
                        ) else listOf(
                            Color(0xFFDFDEDA), Color(0xFFF9F8F5), Color(0xFFE5E3DE)
                        ),
                    ),
                )
                val center = Offset(size.width * .83f, size.height * .47f)
                val unit = size.height * .13f
                for (index in 1..8) {
                    drawCircle(
                        color = light.copy(alpha = .13f),
                        radius = unit * index * (if (kind == "MUSIC") .82f else .65f),
                        center = center,
                        style = Stroke(width = 1.4.dp.toPx()),
                    )
                }
                drawCircle(
                    color = light.copy(alpha = .07f),
                    radius = size.height * .41f,
                    center = center,
                )
            }
            DreamGlassSurface(
                modifier = Modifier.matchParentSize().combinedClickable(
                    interactionSource = interactions,
                    indication = null,
                    onClick = onClick,
                ),
                shape = shape,
                role = DreamGlassRole.CARD,
            ) {
                Column(
                    modifier = Modifier.fillMaxWidth()
                        .padding(horizontal = 22.dp, vertical = 19.dp),
                    verticalArrangement = Arrangement.spacedBy(12.dp),
                ) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Icon(mark, null, Modifier.size(18.dp),
                            tint = foreground.copy(alpha = .88f))
                        Spacer(Modifier.width(10.dp))
                        Text(
                            kind + "  /  LOCAL ENGINE",
                            style = MaterialTheme.typography.labelSmall,
                            color = foreground.copy(alpha = .72f),
                        )
                        Spacer(Modifier.weight(1f))
                        Text(
                            if (installed > 0) installed.toString() + " READY" else "EXPLORE",
                            style = MaterialTheme.typography.labelSmall,
                            color = foreground.copy(alpha = .72f),
                        )
                    }
                    Spacer(Modifier.height(3.dp))
                    Text(
                        title,
                        style = MaterialTheme.typography.headlineMedium,
                        fontWeight = FontWeight.SemiBold,
                        color = foreground,
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis,
                    )
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Text(
                            variants.size.toString() + " variants  ·  " + subtitle,
                            style = MaterialTheme.typography.bodySmall,
                            color = foreground.copy(alpha = .78f),
                            maxLines = 1,
                            overflow = TextOverflow.Ellipsis,
                            modifier = Modifier.weight(1f),
                        )
                        Icon(Icons.Outlined.ChevronRight, "Open model options",
                            Modifier.size(24.dp), tint = foreground)
                    }
                }
            }
        }
    }
}
