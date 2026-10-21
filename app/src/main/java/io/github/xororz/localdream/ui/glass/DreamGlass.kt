package io.github.xororz.localdream.ui.glass

import android.os.Build
import androidx.compose.animation.core.Spring
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.spring
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
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.runtime.staticCompositionLocalOf
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Shape
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.foundation.layout.ColumnScope
import io.github.xororz.localdream.ui.glass.backdrop.Backdrop
import io.github.xororz.localdream.ui.glass.backdrop.backdrops.LayerBackdrop
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
    DreamGlassRole.CARD -> GlassOptics(8.dp, 14.dp, 9.dp, .49f, .33f, false)
    DreamGlassRole.CONTROL -> GlassOptics(3.dp, 10.dp, 7.dp, .53f, .33f, false)
    DreamGlassRole.NAVIGATION -> GlassOptics(9.dp, 21.dp, 16.dp, .56f, .5f, true)
    DreamGlassRole.SELECTION -> GlassOptics(1.dp, 14.dp, 17.dp, .25f, .5f, true)
    DreamGlassRole.OVERLAY -> GlassOptics(12.dp, 20.dp, 16.dp, .79f, .33f, true)
    DreamGlassRole.BUBBLE -> GlassOptics(2.dp, 22.dp, 23.dp, .19f, .5f, true)
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
                    .background(Brush.verticalGradient(listOf(base, surface, base))),
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
        Color(0xFF121923).copy(alpha = o.tint)
    } else {
        Color(0xFFFAFAFF).copy(alpha = o.tint)
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
            highlight = { Highlight.Default },
            shadow = { Shadow.Default },
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
    content: @Composable ColumnScope.() -> Unit,
) {
    if (!LocalDreamGlassEnabled.current) {
        ElevatedCard(modifier = modifier, colors = colors, elevation = elevation, shape = shape, content = content)
    } else {
        val glassShape = shape as? CornerBasedShape ?: RoundedCornerShape(26.dp)
        DreamGlassSurface(
            modifier = modifier,
            shape = glassShape,
            role = DreamGlassRole.CARD,
        ) {
            Column(Modifier.fillMaxWidth(), content = content)
        }
    }
}

/**
 * BitChord-like lifted selection travel, built on the Recall backdrop engine.
 * Uses a damped spring; gesture/reselection retargeting keeps its live velocity.
 */
@Composable
fun DreamGlassTabRow(
    titles: List<String>,
    selectedIndex: Int,
    onSelected: (Int) -> Unit,
    modifier: Modifier = Modifier,
) {
    if (titles.isEmpty()) return
    val current by animateFloatAsState(
        targetValue = selectedIndex.coerceIn(0, titles.lastIndex).toFloat(),
        animationSpec = spring(
            dampingRatio = .68f,
            stiffness = Spring.StiffnessMediumLow,
        ),
        label = "glass-selection-travel",
    )
    BoxWithConstraints(
        modifier = modifier.fillMaxWidth().padding(horizontal = 12.dp, vertical = 8.dp)
            .height(54.dp),
    ) {
        val cell = maxWidth / titles.size
        DreamGlassSurface(Modifier.fillMaxSize(), RoundedCornerShape(27.dp), DreamGlassRole.NAVIGATION) {}
        DreamGlassSurface(
            Modifier.offset(x = cell * current).width(cell).fillMaxHeight().padding(4.dp),
            RoundedCornerShape(23.dp),
            DreamGlassRole.SELECTION,
        ) {}
        Row(Modifier.fillMaxSize()) {
            titles.forEachIndexed { i, label ->
                Box(
                    Modifier.width(cell).fillMaxHeight().clickable { onSelected(i) },
                    contentAlignment = Alignment.Center,
                ) {
                    Text(
                        label,
                        style = MaterialTheme.typography.labelLarge,
                        fontWeight = if (i == selectedIndex) FontWeight.SemiBold else FontWeight.Normal,
                        color = if (i == selectedIndex) MaterialTheme.colorScheme.onSurface
                            else MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
            }
        }
    }
}
