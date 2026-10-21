package io.github.xororz.localdream.ui.glass

import android.os.Build
import androidx.compose.animation.core.Spring
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CornerBasedShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.CardColors
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CardElevation
import androidx.compose.material3.ElevatedCard
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.remember
import androidx.compose.runtime.staticCompositionLocalOf
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Shape
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.foundation.gestures.detectHorizontalDragGestures
import androidx.compose.foundation.selection.selectable
import androidx.compose.foundation.selection.selectableGroup
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.semantics.Role
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.ui.platform.LocalContext
import android.provider.Settings
import kotlin.math.roundToInt
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.foundation.layout.ColumnScope
import io.github.xororz.localdream.ui.glass.backdrop.Backdrop
import io.github.xororz.localdream.ui.glass.backdrop.backdrops.layerBackdrop
import io.github.xororz.localdream.ui.glass.backdrop.backdrops.rememberLayerBackdrop
import io.github.xororz.localdream.ui.glass.backdrop.drawBackdrop
import io.github.xororz.localdream.ui.glass.backdrop.effects.blur
import io.github.xororz.localdream.ui.glass.backdrop.effects.colorControls
import io.github.xororz.localdream.ui.glass.backdrop.effects.lens
import io.github.xororz.localdream.ui.glass.backdrop.highlight.Highlight
import io.github.xororz.localdream.ui.glass.backdrop.shadow.Shadow

/**
 * One shared optical engine for the new UI. The preference only affects presentation;
 * native inference, navigation and generation sessions are never recreated.
 */
val LocalDreamGlassEnabled = staticCompositionLocalOf { false }
val LocalDreamBackdrop = staticCompositionLocalOf<Backdrop?> { null }

enum class DreamGlassRole { CARD, CONTROL, NAVIGATION, SELECTION, OVERLAY, BUBBLE }

private data class GlassOptics(
    val blur: Dp, val depth: Dp, val refraction: Dp,
    val tint: Float, val resolution: Float, val dispersion: Boolean,
)

private fun optics(role: DreamGlassRole): GlassOptics = when (role) {
    DreamGlassRole.CARD -> GlassOptics(5.dp, 18.dp, 12.dp, .38f, .5f, false)
    DreamGlassRole.CONTROL -> GlassOptics(3.dp, 11.dp, 8.dp, .38f, .5f, false)
    DreamGlassRole.NAVIGATION -> GlassOptics(8.dp, 20.dp, 17.dp, .48f, .5f, true)
    DreamGlassRole.SELECTION -> GlassOptics(1.dp, 16.dp, 19.dp, .16f, 1f, true)
    DreamGlassRole.OVERLAY -> GlassOptics(10.dp, 24.dp, 19.dp, .63f, .5f, true)
    DreamGlassRole.BUBBLE -> GlassOptics(2.dp, 22.dp, 23.dp, .14f, .5f, true)
}

/**
 * Render the scene only once into a separate backdrop layer. Glass surfaces are
 * drawn above that layer and never sample themselves (no recursive feedback).
 */
@Composable
fun DreamGlassBackdropLayer(
    enabled: Boolean = LocalDreamGlassEnabled.current,
    content: @Composable BoxScope.() -> Unit,
) {
    val base = MaterialTheme.colorScheme.background
    val surface = MaterialTheme.colorScheme.surface
    val backdrop = rememberLayerBackdrop(onDraw = remember(base) {
        { drawRect(base); drawContent() }
    })
    CompositionLocalProvider(LocalDreamBackdrop provides if (enabled) backdrop else null) {
        Box(Modifier.fillMaxSize()) {
            Box(
                Modifier.matchParentSize()
                    .then(if (enabled) Modifier.layerBackdrop(backdrop) else Modifier)
                    .background(Brush.verticalGradient(listOf(base, surface.copy(alpha = .42f), base))),
            )
            content()
        }
    }
}

/** The text and icons are intentionally painted *after* the refracted surface. */
@Composable
fun DreamGlassSurface(
    modifier: Modifier = Modifier,
    shape: CornerBasedShape = RoundedCornerShape(26.dp),
    role: DreamGlassRole = DreamGlassRole.CONTROL,
    content: @Composable BoxScope.() -> Unit,
) {
    val backdrop = LocalDreamBackdrop.current
    val enabled = LocalDreamGlassEnabled.current &&
        Build.VERSION.SDK_INT >= Build.VERSION_CODES.S && backdrop != null
    val density = LocalDensity.current
    val o = remember(role) { optics(role) }
    val tint = if (MaterialTheme.colorScheme.background.red < .5f) {
        Color(0xFF121214).copy(alpha = o.tint)
    } else {
        Color(0xFFFAFAF7).copy(alpha = o.tint)
    }
    val shapeModifier = if (enabled && backdrop != null) {
        Modifier.drawBackdrop(
            backdrop = backdrop,
            shape = { shape },
            effects = {
                colorControls(saturation = 1.20f)
                blur(with(density) { o.blur.toPx() } * o.resolution)
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                    lens(
                        refractionHeight = with(density) { o.depth.toPx() } * o.resolution,
                        refractionAmount = with(density) { o.refraction.toPx() } * o.resolution,
                        depthEffect = true,
                        chromaticAberration = o.dispersion,
                    )
                }
            },
            highlight = { Highlight(width = .7.dp, alpha = .55f) },
            shadow = { Shadow(radius = 12.dp, alpha = .14f) },
            onDrawSurface = { drawRect(tint) },
            backdropScale = o.resolution,
        )
    } else {
        Modifier.background(MaterialTheme.colorScheme.surfaceContainerLow, shape)
            .border(.5.dp, MaterialTheme.colorScheme.outlineVariant, shape)
    }
    Box(modifier.then(shapeModifier).clip(shape), content = content)
}

/** Drop-in replacement: leaves the classic Material card completely intact when disabled. */
@Composable
fun DreamElevatedCard(
    modifier: Modifier = Modifier,
    colors: CardColors = CardDefaults.elevatedCardColors(),
    elevation: CardElevation = CardDefaults.elevatedCardElevation(),
    shape: Shape = MaterialTheme.shapes.large,
    onClick: (() -> Unit)? = null,
    content: @Composable ColumnScope.() -> Unit,
) {
    if (!LocalDreamGlassEnabled.current) {
        if (onClick != null) {
            ElevatedCard(onClick = onClick, modifier = modifier, colors = colors, elevation = elevation, shape = shape, content = content)
        } else {
            ElevatedCard(modifier = modifier, colors = colors, elevation = elevation, shape = shape, content = content)
        }
    } else {
        val glassShape = shape as? CornerBasedShape ?: RoundedCornerShape(26.dp)
        DreamGlassSurface(
            modifier = if (onClick != null) modifier.clickable(onClick = onClick) else modifier,
            shape = glassShape,
            role = DreamGlassRole.CARD,
        ) {
            Column(Modifier.fillMaxWidth(), content = content)
        }
    }
}

/**
 * Recall's velocity-preserving, lifted liquid selection motion, adapted for
 * Local Dream text tabs. Drag gestures scrub the lens; the spring retargets
 * from its visible position rather than resetting between selections.
 */
@Composable
fun DreamGlassTabRow(
    titles: List<String>,
    selectedIndex: Int,
    onSelected: (Int) -> Unit,
    modifier: Modifier = Modifier,
) {
    if (titles.isEmpty()) return
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
    val scope = rememberCoroutineScope()
    val lens = remember(titles.size) { LiquidLensMotion(scope, selectedIndex) }
    LaunchedEffect(selectedIndex, reducedMotion) {
        lens.moveTo(selectedIndex.coerceIn(titles.indices), reducedMotion)
    }
    BoxWithConstraints(
        modifier = modifier.fillMaxWidth()
            .padding(horizontal = 12.dp, vertical = 8.dp)
            .height(56.dp),
    ) {
        val cell = maxWidth / titles.size
        val cellPx = with(LocalDensity.current) { cell.toPx() }
        DreamGlassSurface(
            Modifier.fillMaxSize(),
            RoundedCornerShape(28.dp),
            DreamGlassRole.NAVIGATION,
        ) {}
        Box(
            Modifier.fillMaxSize()
                .padding(4.dp)
                .selectableGroup()
                .pointerInput(cellPx, titles.size, reducedMotion) {
                    detectHorizontalDragGestures(
                        onDragStart = { lens.beginDrag(reducedMotion) },
                        onHorizontalDrag = { change, delta ->
                            change.consume()
                            lens.dragBy(
                                delta / cellPx.coerceAtLeast(1f),
                                titles.lastIndex,
                                reducedMotion,
                            )
                        },
                        onDragEnd = {
                            val target = lens.position.roundToInt().coerceIn(titles.indices)
                            lens.moveTo(target, reducedMotion, fromDrag = true)
                            if (target != selectedIndex) onSelected(target)
                        },
                        onDragCancel = { lens.cancel(selectedIndex, reducedMotion) },
                    )
                },
        ) {
            val pillHeight = 48.dp
            DreamGlassSurface(
                modifier = Modifier
                    .offset(x = (cell - 8.dp) * lens.position)
                    .width(cell - 8.dp)
                    .height(pillHeight)
                    .graphicsLayer {
                        translationY = -8.dp.toPx() * lens.lift.value
                        scaleX = 1f + lens.deformation
                        scaleY = 1f - lens.deformation * .5f
                        shadowElevation = 9.dp.toPx() * lens.lift.value
                    },
                shape = RoundedCornerShape(24.dp),
                role = DreamGlassRole.SELECTION,
            ) {}
            Row(Modifier.fillMaxSize()) {
                titles.forEachIndexed { index, label ->
                    Box(
                        modifier = Modifier.width(cell - 8.dp).fillMaxHeight().selectable(
                            selected = index == selectedIndex,
                            role = Role.Tab,
                            onClick = {
                                if (index != selectedIndex) {
                                    lens.moveTo(index, reducedMotion)
                                    onSelected(index)
                                }
                            },
                        ),
                        contentAlignment = Alignment.Center,
                    ) {
                        Text(
                            label,
                            style = MaterialTheme.typography.labelLarge,
                            fontWeight = if (index == selectedIndex)
                                FontWeight.SemiBold else FontWeight.Normal,
                            color = if (index == selectedIndex)
                                MaterialTheme.colorScheme.onSurface
                            else MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                    }
                }
            }
        }
    }
}
