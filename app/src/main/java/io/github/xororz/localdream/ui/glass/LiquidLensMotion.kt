package io.github.xororz.localdream.ui.glass

import androidx.compose.animation.core.*
import androidx.compose.runtime.*
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Job
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.launch
import kotlin.math.abs

/** Travel, held lift and deformation have distinct springs, like BitChord's lifted selection lens. */
@Stable
class LiquidLensMotion(
    private val scope: CoroutineScope,
    initial: Int,
) {
    private val travel = Animatable(initial.toFloat())
    val lift = Animatable(0f)
    private var job: Job? = null
    private var dragPosition by mutableFloatStateOf(initial.toFloat())
    var dragging by mutableStateOf(false)
        private set
    var deformation by mutableFloatStateOf(0f)
        private set
    private var target = initial
    val position: Float get() = if (dragging) dragPosition else travel.value

    fun moveTo(
        next: Int,
        reduced: Boolean,
        fromDrag: Boolean = false,
    ) {
        if (!fromDrag && next == target && !dragging && !reduced) return
        val start = position
        val velocity = travel.velocity
        target = next
        job?.cancel()
        dragging = false
        job =
            scope.launch {
                if (fromDrag) travel.snapTo(start)
                if (reduced) {
                    travel.snapTo(next.toFloat())
                    lift.snapTo(0f)
                    deformation = 0f
                    return@launch
                }
                coroutineScope {
                    val lifting = launch { lift.animateTo(1f, spring(.62f, 620f)) }
                    var previousVelocity = velocity
                    var previousTime = System.nanoTime()
                    travel.animateTo(next.toFloat(), spring(.78f, 430f), initialVelocity = velocity) {
                        // A bounded launch/braking response, not a perpetually running physics clock.
                        val now = System.nanoTime()
                        val elapsed = ((now - previousTime) / 1_000_000_000f).coerceIn(.001f, .04f)
                        val acceleration = (abs(this.velocity) - abs(previousVelocity)) / elapsed
                        val targetBend = (acceleration * .0006f).coerceIn(-.09f, .09f)
                        deformation += (targetBend - deformation) * .28f
                        previousVelocity = this.velocity
                        previousTime = now
                    }
                    lifting.join()
                }
                deformation = 0f
                lift.animateTo(0f, spring(.64f, 480f))
            }
    }

    fun beginDrag(reduced: Boolean) {
        val start = position
        job?.cancel()
        dragPosition = start
        dragging = true
        deformation = 0f
        job = scope.launch { lift.animateTo(if (reduced) 0f else 1f, if (reduced) snap() else spring(.64f, 620f)) }
    }

    fun dragBy(
        delta: Float,
        last: Int,
        reduced: Boolean,
    ) {
        if (!dragging) return
        dragPosition = (dragPosition + delta).coerceIn(0f, last.toFloat())
        deformation = if (reduced) 0f else (abs(delta) * 1.8f).coerceIn(0f, .09f)
    }

    fun cancel(
        selected: Int,
        reduced: Boolean,
    ) = moveTo(selected, reduced, fromDrag = true)
}
