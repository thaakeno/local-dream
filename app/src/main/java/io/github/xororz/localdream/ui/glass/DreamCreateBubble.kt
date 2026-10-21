package io.github.xororz.localdream.ui.glass

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.animateContentSize
import androidx.compose.animation.core.spring
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.animation.expandHorizontally
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.shrinkHorizontally
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.outlined.Add
import androidx.compose.material.icons.outlined.AutoAwesome
import androidx.compose.material.icons.outlined.History
import androidx.compose.material.icons.outlined.Settings
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.getValue
import androidx.compose.runtime.setValue
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.layout.onSizeChanged
import androidx.compose.ui.unit.IntSize
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.foundation.gestures.detectDragGestures
import androidx.compose.ui.platform.LocalContext
import android.os.Build
import android.provider.Settings
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp

/** One live glass boundary morphs from a small bubble into inline quick tools. */
@Composable
fun DreamCreateBubble(onAddModel: () -> Unit, onHistory: () -> Unit, onSettings: () -> Unit) {
    var expanded by rememberSaveable { mutableStateOf(false) }
    var touch by remember { mutableStateOf(Offset.Zero) }
    var bubbleSize by remember { mutableStateOf(IntSize.Zero) }
    val context = LocalContext.current
    val reducedMotion = remember(context) {
        runCatching {
            Settings.Global.getFloat(
                context.contentResolver,
                Settings.Global.ANIMATOR_DURATION_SCALE,
                1f,
            ) == 0f
        }.getOrDefault(false)
    }
    val liquidEnabled = LocalDreamGlassEnabled.current && !expanded &&
        Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU
    val shader = remember(liquidEnabled) {
        if (liquidEnabled) BubbleGlassShader() else null
    }
    val phaseTransition = rememberInfiniteTransition(label = "creation-bubble-surface")
    val phase by phaseTransition.animateFloat(
        initialValue = 0f,
        targetValue = 6.28318f,
        animationSpec = infiniteRepeatable(
            tween(durationMillis = 12_000, easing = LinearEasing),
        ),
        label = "creation-bubble-meniscus",
    )
    DreamGlassSurface(
        modifier = Modifier
            .onSizeChanged { bubbleSize = it }
            .pointerInput(liquidEnabled, bubbleSize) {
                if (liquidEnabled) detectDragGestures(
                    onDrag = { change, _ ->
                        change.consume()
                        touch = Offset(
                            change.position.x / bubbleSize.width.coerceAtLeast(1) * 2f - 1f,
                            change.position.y / bubbleSize.height.coerceAtLeast(1) * 2f - 1f,
                        )
                    },
                    onDragEnd = { touch = Offset.Zero },
                    onDragCancel = { touch = Offset.Zero },
                )
            }
            .graphicsLayer {
                renderEffect = if (liquidEnabled && shader != null && bubbleSize.width > 0) {
                    shader.effect(
                        bubbleSize.width.toFloat(),
                        bubbleSize.height.toFloat(),
                        if (reducedMotion) 0f else phase,
                        touch.x,
                        touch.y,
                    )
                } else null
            }
            .animateContentSize(spring(dampingRatio = .74f, stiffness = 400f)),
        shape = RoundedCornerShape(28.dp),
        role = DreamGlassRole.BUBBLE,
    ) {
        Row(
            Modifier.height(56.dp).padding(horizontal = 7.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            AnimatedVisibility(
                visible = expanded,
                enter = expandHorizontally(spring()) + fadeIn(),
                exit = shrinkHorizontally(spring()) + fadeOut(),
            ) {
                Row {
                    IconButton(onClick = { expanded = false; onAddModel() }) {
                        Icon(Icons.Outlined.Add, "Add model")
                    }
                    IconButton(onClick = { expanded = false; onHistory() }) {
                        Icon(Icons.Outlined.History, "History")
                    }
                    IconButton(onClick = { expanded = false; onSettings() }) {
                        Icon(Icons.Outlined.Settings, "Settings")
                    }
                }
            }
            IconButton(onClick = { expanded = !expanded }, modifier = Modifier.size(44.dp)) {
                Icon(Icons.Outlined.AutoAwesome, if (expanded) "Collapse actions" else "Open quick actions")
            }
        }
    }
}
