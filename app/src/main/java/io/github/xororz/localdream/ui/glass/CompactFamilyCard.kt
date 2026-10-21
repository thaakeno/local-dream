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
import androidx.compose.material.icons.outlined.AutoAwesome
import androidx.compose.material.icons.outlined.ChevronRight
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import io.github.xororz.localdream.data.Model

/** Dense family entry with original variant chooser preserved. */
@OptIn(ExperimentalFoundationApi::class)
@Composable
fun CompactFamilyCard(
    title: String,
    subtitle: String,
    variants: List<Model>,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
) {
    val input = remember { MutableInteractionSource() }
    val pressed by input.collectIsPressedAsState()
    val scale by animateFloatAsState(
        if (pressed) .974f else 1f,
        animationSpec = spring(dampingRatio = .72f, stiffness = 530f),
        label = "family-pressure",
    )
    val installed = variants.count { it.isDownloaded }
    val familyShape = RoundedCornerShape(26.dp)
    DreamGlassSurface(
        modifier = modifier.fillMaxWidth()
            .graphicsLayer { scaleX = scale; scaleY = scale }
            .clip(familyShape)
            .combinedClickable(interactionSource = input, indication = null, onClick = onClick),
        shape = familyShape,
        role = DreamGlassRole.CARD,
    ) {
        Row(
            Modifier.fillMaxWidth().padding(horizontal = 15.dp, vertical = 12.dp),
            horizontalArrangement = Arrangement.spacedBy(12.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Surface(
                Modifier.size(50.dp),
                shape = RoundedCornerShape(17.dp),
                color = MaterialTheme.colorScheme.primary.copy(alpha = .13f),
            ) {
                Box(contentAlignment = Alignment.Center) {
                    Icon(Icons.Outlined.AutoAwesome, contentDescription = null,
                        modifier = Modifier.size(24.dp), tint = MaterialTheme.colorScheme.primary)
                }
            }
            Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(4.dp)) {
                Text(title, style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
                Text(subtitle, style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant)
                Text(
                    "${variants.size} variants · ${installed} installed",
                    style = MaterialTheme.typography.labelSmall,
                    color = if (installed > 0) MaterialTheme.colorScheme.primary
                        else MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
            Icon(Icons.Outlined.ChevronRight, contentDescription = "Choose variant",
                tint = MaterialTheme.colorScheme.primary)
        }
    }
}
