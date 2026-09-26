package io.github.xororz.localdream.ui.screens

import android.graphics.Bitmap
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.expandVertically
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.shrinkVertically
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.ElevatedCard
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.produceState
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import io.github.xororz.localdream.ui.components.SmoothIndeterminateLinearWavyProgressIndicator
import io.github.xororz.localdream.ui.components.SmoothLinearWavyProgressIndicator
import io.github.xororz.localdream.utils.GenerationTelemetry
import io.github.xororz.localdream.utils.GenerationTelemetrySnapshot
import kotlinx.coroutines.delay

@Composable
internal fun GenerationProgressCard(
    visible: Boolean,
    progress: Float,
    step: Int,
    totalSteps: Int,
    phase: String,
    batchIndex: Int,
    batchCount: Int,
    startedAtMillis: Long?,
    showStats: Boolean,
    intermediateBitmap: Bitmap?,
    acceleratorLabel: String = "NPU · HTP",
    onCancel: () -> Unit,
) {
    AnimatedVisibility(
        visible = visible,
        enter = expandVertically() + fadeIn(),
        exit = shrinkVertically() + fadeOut(),
    ) {
        ElevatedCard(
            modifier = Modifier.fillMaxWidth(),
            shape = MaterialTheme.shapes.large,
        ) {
            Column(
                modifier = Modifier.padding(16.dp),
                verticalArrangement = Arrangement.spacedBy(10.dp),
            ) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(12.dp),
                ) {
                    Column(modifier = Modifier.weight(1f)) {
                        Text(
                            text = if (batchIndex > 0 && batchCount > 1) {
                                "Generating ($batchIndex/$batchCount)"
                            } else {
                                "Generating"
                            },
                            style = MaterialTheme.typography.titleMedium,
                            fontWeight = FontWeight.SemiBold,
                        )
                        val phaseLabel = generationPhaseLabel(phase)
                        val numericStage =
                            totalSteps > 0 &&
                                (phase == "denoising" || phase == "loading_model")
                        Text(
                            text = if (numericStage) {
                                val prefix = if (phase == "denoising") "Step " else ""
                                "$phaseLabel · $prefix${step.coerceIn(0, totalSteps)}/$totalSteps · " +
                                    "${(progress * 100).toInt()}%"
                            } else {
                                phaseLabel
                            },
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                        GenerationElapsedLabel(startedAtMillis)
                    }
                    OutlinedButton(onClick = onCancel) { Text("Cancel") }
                }

                val determinate =
                    totalSteps > 0 &&
                        (phase == "denoising" || phase == "loading_model")
                if (determinate) {
                    SmoothLinearWavyProgressIndicator(
                        progress = progress,
                        modifier = Modifier.fillMaxWidth(),
                    )
                } else {
                    SmoothIndeterminateLinearWavyProgressIndicator(
                        modifier = Modifier.fillMaxWidth(),
                    )
                }

                if (showStats) {
                    GenerationTelemetryPanel(startedAtMillis, acceleratorLabel)
                }

                intermediateBitmap?.let { bitmap ->
                    Card(
                        modifier = Modifier.fillMaxWidth(),
                        shape = RoundedCornerShape(14.dp),
                    ) {
                        Image(
                            bitmap = bitmap.asImageBitmap(),
                            contentDescription = "Generation preview",
                            modifier = Modifier.fillMaxWidth(),
                            contentScale = ContentScale.FillWidth,
                        )
                    }
                }
            }
        }
    }
}

private fun generationPhaseLabel(phase: String): String = when (phase) {
    "queued" -> "Starting generation"
    "preparing" -> "Preparing model runtime"
    "preparing_latents" -> "Preparing latents"
    "loading_model" -> "Loading model"
    "loading_denoiser" -> "Loading denoiser"
    "encoding_input" -> "Encoding input image"
    "encoding_prompt" -> "Encoding prompt"
    "inverting" -> "Preparing latent inversion"
    "denoising" -> "Sampling"
    "decoding" -> "Decoding image"
    "finalizing" -> "Finalizing result"
    else -> phase.replace('_', ' ').replaceFirstChar { it.uppercase() }
}

@Composable
private fun GenerationElapsedLabel(startedAtMillis: Long?) {
    val elapsedSeconds by produceState(initialValue = 0L, key1 = startedAtMillis) {
        while (true) {
            value = startedAtMillis?.let {
                (System.currentTimeMillis() - it).coerceAtLeast(0L) / 1000L
            } ?: 0L
            delay(500L)
        }
    }
    Text(
        text = "Elapsed $elapsedSeconds" + "s",
        style = MaterialTheme.typography.labelSmall,
        color = MaterialTheme.colorScheme.onSurfaceVariant,
    )
}

@Composable
private fun GenerationTelemetryPanel(
    @Suppress("UNUSED_PARAMETER") startedAtMillis: Long?,
    acceleratorLabel: String,
) {
    val context = LocalContext.current
    val telemetry by produceState<GenerationTelemetrySnapshot?>(
        initialValue = null,
        key1 = startedAtMillis,
    ) {
        while (true) {
            value = runCatching { GenerationTelemetry.sample(context) }.getOrNull()
            delay(1000L)
        }
    }
    val t = telemetry ?: return
    val temp = t.batteryTempC?.let {
        String.format(java.util.Locale.US, "%.1f°C", it)
    } ?: "—"
    val current = t.batteryCurrentMa?.let { "${kotlin.math.abs(it)} mA" } ?: "—"

    Card(
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surfaceContainerHigh,
        ),
        shape = RoundedCornerShape(14.dp),
    ) {
        Column(
            modifier = Modifier.padding(horizontal = 12.dp, vertical = 10.dp),
            verticalArrangement = Arrangement.spacedBy(4.dp),
        ) {
            Text(
                acceleratorLabel,
                style = MaterialTheme.typography.labelLarge,
                color = MaterialTheme.colorScheme.primary,
            )
            Text(
                "RAM ${GenerationTelemetry.formatBytes(t.processRamBytes)} app · " +
                    "${GenerationTelemetry.formatBytes(t.availableRamBytes)} free",
                style = MaterialTheme.typography.bodySmall,
            )
            Text(
                "Battery " +
                    (if (t.batteryPercent >= 0) "${t.batteryPercent}%" else "—") +
                    " · $temp · $current · Thermal ${t.thermalStatus}",
                style = MaterialTheme.typography.bodySmall,
            )
            Text(
                "Elapsed ${elapsed}s",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
    }
}
