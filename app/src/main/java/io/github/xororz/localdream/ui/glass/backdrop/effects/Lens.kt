/*
 * Vendored from Kyant0/backdrop v2.0.0 (io.github.kyant0:backdrop)
 * https://github.com/Kyant0/backdrop — Copyright 2025 Kyant0, Apache License 2.0
 *
 * Vendored so the library ships as source with this app (binary AARs compiled
 * against older Compose broke at runtime) and to add a backdrop resolution
 * scale for cheaper effect rendering. KMP expect/actual declarations were
 * merged into this single Android source set. Package renamed accordingly.
 */
package io.github.xororz.localdream.ui.glass.backdrop.effects

import androidx.annotation.FloatRange
import androidx.compose.foundation.shape.AbsoluteRoundedCornerShape
import androidx.compose.foundation.shape.CornerBasedShape
import androidx.compose.ui.unit.Density
import androidx.compose.ui.unit.LayoutDirection
import androidx.compose.ui.util.fastCoerceAtLeast
import androidx.compose.ui.util.fastCoerceAtMost
import io.github.xororz.localdream.ui.glass.backdrop.BackdropEffectScope
import io.github.xororz.localdream.ui.glass.backdrop.internal.RoundedRectRefractionShaderString
import io.github.xororz.localdream.ui.glass.backdrop.internal.RoundedRectRefractionWithDispersionShaderString
import io.github.xororz.localdream.ui.glass.backdrop.internal.RuntimeShaderEffect
import io.github.xororz.localdream.ui.glass.backdrop.isRuntimeShaderSupported

fun BackdropEffectScope.lens(
    @FloatRange(from = 0.0) refractionHeight: Float,
    @FloatRange(from = 0.0) refractionAmount: Float,
    depthEffect: Boolean = false,
    chromaticAberration: Boolean = false,
) {
    if (!isRuntimeShaderSupported()) return
    if (refractionHeight <= 0f || refractionAmount <= 0f) return

    if (padding > 0f) {
        padding = (padding - refractionHeight).fastCoerceAtLeast(0f)
    }

    val cornerRadii = cornerRadii
    val effect =
        if (cornerRadii != null) {
            val shader =
                if (!chromaticAberration) {
                    obtainRuntimeShader(
                        "Refraction",
                        RoundedRectRefractionShaderString,
                    )
                } else {
                    obtainRuntimeShader(
                        "RefractionWithDispersion",
                        RoundedRectRefractionWithDispersionShaderString,
                    )
                }
            shader.apply {
                setFloatUniform("size", size.width, size.height)
                setFloatUniform("offset", -padding, -padding)
                setFloatUniform("cornerRadii", cornerRadii)
                setFloatUniform("refractionHeight", refractionHeight)
                setFloatUniform("refractionAmount", -refractionAmount)
                setFloatUniform("depthEffect", if (depthEffect) 1f else 0f)
                if (chromaticAberration) {
                    setFloatUniform("chromaticAberration", 1f)
                }
            }
            RuntimeShaderEffect(shader, "content")
        } else {
            throwUnsupportedSDFException()
        }
    effect(effect)
}

// Vendored change: support for io.github.kyant0:shapes' RoundedRectangularShape was
// dropped so the vendored sources have no external dependency; this app only passes
// CornerBasedShape here.
private val BackdropEffectScope.cornerRadii: FloatArray?
    get() =
        when (val shape = shape) {
            is AbsoluteRoundedCornerShape -> {
                val size = size
                val scaledDensity = Density(density * contentScale, fontScale)
                val maxRadius = size.minDimension / 2f
                val topLeft = shape.topStart.toPx(size, scaledDensity)
                val topRight = shape.topEnd.toPx(size, scaledDensity)
                val bottomRight = shape.bottomEnd.toPx(size, scaledDensity)
                val bottomLeft = shape.bottomStart.toPx(size, scaledDensity)
                floatArrayOf(
                    topLeft.fastCoerceAtMost(maxRadius),
                    topRight.fastCoerceAtMost(maxRadius),
                    bottomRight.fastCoerceAtMost(maxRadius),
                    bottomLeft.fastCoerceAtMost(maxRadius),
                )
            }

            is CornerBasedShape -> {
                val size = size
                val scaledDensity = Density(density * contentScale, fontScale)
                val maxRadius = size.minDimension / 2f
                val isLtr = layoutDirection == LayoutDirection.Ltr
                val topLeft =
                    if (isLtr) {
                        shape.topStart.toPx(size, scaledDensity)
                    } else {
                        shape.topEnd.toPx(size, scaledDensity)
                    }
                val topRight =
                    if (isLtr) {
                        shape.topEnd.toPx(size, scaledDensity)
                    } else {
                        shape.topStart.toPx(size, scaledDensity)
                    }
                val bottomRight =
                    if (isLtr) {
                        shape.bottomEnd.toPx(size, scaledDensity)
                    } else {
                        shape.bottomStart.toPx(size, scaledDensity)
                    }
                val bottomLeft =
                    if (isLtr) {
                        shape.bottomStart.toPx(size, scaledDensity)
                    } else {
                        shape.bottomEnd.toPx(size, scaledDensity)
                    }
                floatArrayOf(
                    topLeft.fastCoerceAtMost(maxRadius),
                    topRight.fastCoerceAtMost(maxRadius),
                    bottomRight.fastCoerceAtMost(maxRadius),
                    bottomLeft.fastCoerceAtMost(maxRadius),
                )
            }

            else -> {
                null
            }
        }

private fun throwUnsupportedSDFException(): Nothing =
    throw UnsupportedOperationException(
        "Only RoundedRectangularShape or CornerBasedShape is supported in lens effects.",
    )
