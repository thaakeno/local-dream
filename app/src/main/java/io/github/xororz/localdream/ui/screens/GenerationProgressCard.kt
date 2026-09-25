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
                        val phaseLabel = when (phase) {
                            "encoding_input" -> "Encoding input"
                            "encoding_prompt" -> "Encoding prompt"
                            "denoising" -> "Denoising"
                            "decoding" -> "Decoding"
                            "finalizing" -> "Finalizing"
                            else -> "Preparing runtime"
                        }
                        Text(
                            text = if (phase == "denoising" && totalSteps > 0) {
                                "$phaseLabel · Step ${step.coerceAtLeast(0)}/$totalSteps · " +
                                    "${(progress * 100).toInt()}%"
                            } else {
                                phaseLabel
                            },
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                    }
                    OutlinedButton(onClick = onCancel) { Text("Cancel") }
                }

                SmoothLinearWavyProgressIndicator(
                    progress = progress,
                    modifier = Modifier.fillMaxWidth(),
                )

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

@Composable
private fun GenerationTelemetryPanel(startedAtMillis: Long?, acceleratorLabel: String) {
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
    val elapsed = startedAtMillis?.let {
        (System.currentTimeMillis() - it).coerceAtLeast(0L) / 1000L
    } ?: 0L
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
