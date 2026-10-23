package io.github.xororz.localdream.ui.screens

import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Clear
import androidx.compose.material.icons.filled.ContentPaste
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.Share
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.ButtonGroupDefaults
import androidx.compose.material3.ExperimentalMaterial3ExpressiveApi
import androidx.compose.material3.FilledTonalButton
import androidx.compose.material3.FilterChip
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Surface
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LocalMinimumInteractiveComponentSize
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Slider
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.ToggleButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.draw.scale
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import io.github.xororz.localdream.R
import io.github.xororz.localdream.data.Resolution
import kotlin.math.roundToInt

/**
 * The generation-parameter dialog opened from the prompt page. Pure UI: every
 * state mutation is routed back through callbacks so the screen keeps ownership
 * of parameter state and persistence.
 */
@OptIn(ExperimentalMaterial3ExpressiveApi::class)
@Composable
internal fun AdvancedSettingsDialog(
    isSdxl: Boolean,
    isDit: Boolean = false,
    runOnCpu: Boolean,
    useImg2img: Boolean,
    isRunning: Boolean,
    aspectRatio: String,
    availableResolutions: List<Resolution>,
    currentWidth: Int,
    currentHeight: Int,
    scheduler: String,
    steps: Float,
    cfg: Float,
    useOpenCL: Boolean,
    isQwen21: Boolean = false,
    ditKind: String = "",
    isAnima: Boolean = false,
    mossIsTurbo: Boolean = false,
    htpMode: String = "auto",
    batchCounts: Int,
    denoiseStrength: Float,
    seed: String,
    returnedSeed: Long?,
    onAspectRatioSelected: (String) -> Unit,
    onCustomAspectRatioClick: () -> Unit,
    onResolutionSelected: (Resolution) -> Unit,
    onDitWidthChange: (Float) -> Unit = {},
    onDitHeightChange: (Float) -> Unit = {},
    onSchedulerChange: (String) -> Unit,
    onStepsChange: (Float) -> Unit,
    onCfgChange: (Float) -> Unit,
    onSizeChange: (Float) -> Unit,
    onCpuSelected: () -> Unit,
    onGpuSelected: () -> Unit,
    onHtpModeChange: (String) -> Unit = {},
    onBatchCountsChange: (Float) -> Unit,
    onDenoiseStrengthChange: (Float) -> Unit,
    onSeedChange: (String) -> Unit,
    onUseLastSeed: () -> Unit,
    onImportFromClipboard: () -> Unit,
    onShare: () -> Unit,
    onReset: () -> Unit,
    onDismiss: () -> Unit,
) {
    AlertDialog(
        onDismissRequest = onDismiss,
        title = {
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Text(
                    stringResource(R.string.advanced_settings_title),
                    modifier = Modifier.weight(1f),
                )
                IconButton(onClick = onImportFromClipboard) {
                    Icon(
                        imageVector = Icons.Default.ContentPaste,
                        contentDescription = stringResource(R.string.import_from_clipboard),
                    )
                }
                IconButton(onClick = onShare) {
                    Icon(
                        imageVector = Icons.Default.Share,
                        contentDescription = stringResource(R.string.share),
                    )
                }
            }
        },
        text = {
            Column(
                verticalArrangement = Arrangement.spacedBy(16.dp),
                modifier = Modifier
                    .fillMaxWidth()
                    .verticalScroll(rememberScrollState())
                    .padding(vertical = 4.dp),
            ) {
                // One sampler is not one scheduler. Explain the active model
                // implementation rather than exposing incompatible knobs.
                Text(
                    text = when {
                        ditKind == "moss" && mossIsTurbo ->
                            "MOSS Turbo · 8-step distilled image generation"
                        ditKind == "moss" -> "MOSS-Image · full-step flow matching"
                        ditKind == "qwen21" -> "Qwen Image 2.1 · native FlowMatch"
                        isDit -> "Diffusion Transformer · model-native sampling"
                        isAnima -> "Anima · native flow matching"
                        isSdxl -> "SDXL · diffusion sampling"
                        else -> "Stable Diffusion · diffusion sampling"
                    },
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
                Text(
                    "Canvas",
                    style = MaterialTheme.typography.titleSmall,
                )
                // On SDXL the aspect ratio is inpaint-based padding, so it
                // needs the VAE encoder that --no_img2img skips. DiT models get
                // width and height directly instead.
                if (isSdxl && useImg2img) {
                    Column(modifier = Modifier.fillMaxWidth()) {
                        Text(
                            stringResource(R.string.aspect_ratio),
                            style = MaterialTheme.typography.bodyMedium,
                        )
                        val presets = listOf("1:1", "3:4", "4:3")
                        val isCustom = aspectRatio !in presets
                        Row(
                            modifier = Modifier
                                .fillMaxWidth()
                                .horizontalScroll(rememberScrollState()),
                            horizontalArrangement = Arrangement.spacedBy(
                                ButtonGroupDefaults.ConnectedSpaceBetween,
                            ),
                        ) {
                            presets.forEachIndexed { index, ratio ->
                                val shapes = if (index == 0) {
                                    ButtonGroupDefaults.connectedLeadingButtonShapes()
                                } else {
                                    ButtonGroupDefaults.connectedMiddleButtonShapes()
                                }
                                ToggleButton(
                                    checked = aspectRatio == ratio,
                                    onCheckedChange = { checked ->
                                        if (checked) onAspectRatioSelected(ratio)
                                    },
                                    shapes = shapes,
                                    enabled = !isRunning,
                                ) {
                                    Text(ratio)
                                }
                            }
                            ToggleButton(
                                checked = isCustom,
                                onCheckedChange = { onCustomAspectRatioClick() },
                                shapes = ButtonGroupDefaults.connectedTrailingButtonShapes(),
                                enabled = !isRunning,
                            ) {
                                Text(
                                    if (isCustom) {
                                        aspectRatio
                                    } else {
                                        stringResource(R.string.aspect_ratio_custom)
                                    },
                                )
                            }
                        }
                    }
                }
                if (isDit) {
                    Column(modifier = Modifier.fillMaxWidth()) {
                        Text(
                            stringResource(R.string.image_size, currentWidth, currentHeight),
                            style = MaterialTheme.typography.bodyMedium,
                        )
                        Text(
                            stringResource(R.string.width),
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                        Slider(
                            value = currentWidth.toFloat(),
                            onValueChange = onDitWidthChange,
                            valueRange = DIT_MIN_SIZE.toFloat()..DIT_MAX_SIZE.toFloat(),
                            steps = DIT_SIZE_STEPS,
                            enabled = !isRunning,
                            modifier = Modifier.fillMaxWidth(),
                        )
                        Text(
                            stringResource(R.string.height),
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                        Slider(
                            value = currentHeight.toFloat(),
                            onValueChange = onDitHeightChange,
                            valueRange = DIT_MIN_SIZE.toFloat()..DIT_MAX_SIZE.toFloat(),
                            steps = DIT_SIZE_STEPS,
                            enabled = !isRunning,
                            modifier = Modifier.fillMaxWidth(),
                        )
                    }
                }
                if (!runOnCpu && !isSdxl && !isDit && availableResolutions.isNotEmpty()) {
                    Column(modifier = Modifier.fillMaxWidth()) {
                        Text(
                            stringResource(R.string.resolution),
                            style = MaterialTheme.typography.bodyMedium,
                        )
                        Row(
                            modifier = Modifier
                                .fillMaxWidth()
                                .horizontalScroll(rememberScrollState()),
                            horizontalArrangement = Arrangement.spacedBy(
                                ButtonGroupDefaults.ConnectedSpaceBetween,
                            ),
                        ) {
                            availableResolutions.forEachIndexed { index, resolution ->
                                val shapes = when (index) {
                                    0 -> ButtonGroupDefaults.connectedLeadingButtonShapes()

                                    availableResolutions.lastIndex ->
                                        ButtonGroupDefaults.connectedTrailingButtonShapes()

                                    else -> ButtonGroupDefaults.connectedMiddleButtonShapes()
                                }
                                ToggleButton(
                                    checked = currentWidth == resolution.width &&
                                        currentHeight == resolution.height,
                                    onCheckedChange = { checked ->
                                        if (checked) onResolutionSelected(resolution)
                                    },
                                    shapes = shapes,
                                    enabled = !isRunning,
                                ) {
                                    Text(resolution.toString())
                                }
                            }
                        }
                    }
                }

                HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant)
                Text("Sampling", style = MaterialTheme.typography.titleSmall)
                Surface(
                    modifier = Modifier.fillMaxWidth(),
                    shape = RoundedCornerShape(20.dp),
                    tonalElevation = 2.dp,
                ) {
                    Column(
                        modifier = Modifier.padding(14.dp),
                        verticalArrangement = Arrangement.spacedBy(12.dp),
                    ) {
                        when {
                            ditKind == "moss" -> {
                                val activeSchedule = scheduler.substringAfter('|', "simple")
                                val options = listOf(
                                    "simple" to "Simple",
                                    "sgm_uniform" to "SGM Uniform",
                                    "discrete" to "Normal",
                                    "karras" to "Karras",
                                )
                                Text("Sampler", style = MaterialTheme.typography.labelLarge)
                                Text(
                                    "Euler · Flow matching",
                                    style = MaterialTheme.typography.bodyMedium,
                                )
                                Text("Noise schedule", style = MaterialTheme.typography.labelLarge)
                                FlowRow(
                                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                                    verticalArrangement = Arrangement.spacedBy(0.dp),
                                ) {
                                    options.forEach { (id, label) ->
                                        FilterChip(
                                            selected = activeSchedule == id,
                                            onClick = { onSchedulerChange("euler|$id") },
                                            enabled = !isRunning,
                                            label = { Text(label) },
                                        )
                                    }
                                }
                                Text(
                                    if (mossIsTurbo) {
                                        "Simple is recommended for 8-step Turbo. Shift 5 · CFG 1. The other schedules are for A/B testing."
                                    } else {
                                        "SGM Uniform is recommended for full-step MOSS SFT/Base. Euler · shift 5."
                                    },
                                    style = MaterialTheme.typography.bodySmall,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                                )
                            }
                            isDit -> {
                                Text("Sampler", style = MaterialTheme.typography.labelLarge)
                                Text("Euler", style = MaterialTheme.typography.bodyMedium)
                                Text("Noise schedule", style = MaterialTheme.typography.labelLarge)
                                Text(
                                    if (ditKind == "qwen21") {
                                        "Native FlowMatch · model-controlled sigma schedule"
                                    } else {
                                        "Model-native · automatically managed by the engine"
                                    },
                                    style = MaterialTheme.typography.bodyMedium,
                                )
                                Text(
                                    if (ditKind == "qwen21") {
                                        "Qwen Turbo variants can use a published fixed sigma sequence. Karras or SDXL samplers would not be applied by this engine."
                                    } else {
                                        "This model controls its sampling schedule. Unused SDXL sampler choices are intentionally hidden."
                                    },
                                    style = MaterialTheme.typography.bodySmall,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                                )
                            }
                            isAnima -> {
                                Text("Sampler", style = MaterialTheme.typography.labelLarge)
                                FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                                    listOf("euler" to "Euler", "euler_a" to "Euler A").forEach { (id, label) ->
                                        FilterChip(
                                            selected = scheduler == id,
                                            onClick = { onSchedulerChange(id) },
                                            enabled = !isRunning,
                                            label = { Text(label) },
                                        )
                                    }
                                }
                                Text(
                                    "Native FlowMatch · shift 3. Euler A adds ancestral noise.",
                                    style = MaterialTheme.typography.bodySmall,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                                )
                            }
                            else -> {
                                val baseId = scheduler.removeSuffix("_karras")
                                val karras = scheduler.endsWith("_karras")
                                val canKarras = baseId != "lcm"
                                val samplers = listOf(
                                    "dpm" to "DPM++ 2M",
                                    "dpm_sde" to "DPM++ 2M SDE",
                                    "euler" to "Euler",
                                    "euler_a" to "Euler A",
                                    "lcm" to "LCM",
                                )
                                Text("Sampler", style = MaterialTheme.typography.labelLarge)
                                FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                                    samplers.forEach { (id, label) ->
                                        FilterChip(
                                            selected = baseId == id,
                                            onClick = {
                                                onSchedulerChange(
                                                    if (karras && id != "lcm") "${id}_karras" else id,
                                                )
                                            },
                                            enabled = !isRunning,
                                            label = { Text(label) },
                                        )
                                    }
                                }
                                Text("Noise schedule", style = MaterialTheme.typography.labelLarge)
                                FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                                    FilterChip(
                                        selected = !karras || !canKarras,
                                        onClick = { onSchedulerChange(baseId) },
                                        enabled = !isRunning,
                                        label = { Text("Normal") },
                                    )
                                    FilterChip(
                                        selected = karras && canKarras,
                                        onClick = { onSchedulerChange("${baseId}_karras") },
                                        enabled = !isRunning && canKarras,
                                        label = { Text("Karras") },
                                    )
                                }
                                Text(
                                    if (baseId == "lcm") {
                                        "LCM handles its own noise schedule; Karras is not supported."
                                    } else {
                                        "These samplers and Normal/Karras schedules are implemented in the SD1.5/SDXL engine."
                                    },
                                    style = MaterialTheme.typography.bodySmall,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                                )
                            }
                        }
                    }
                }
                HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant)
                Text("Guidance & steps", style = MaterialTheme.typography.titleSmall)

                Column {
                    Text(
                        stringResource(R.string.steps, steps.roundToInt()),
                        style = MaterialTheme.typography.bodyMedium,
                    )
                    Slider(
                        value = steps,
                        onValueChange = { onStepsChange(it.roundToInt().toFloat()) },
                        valueRange = 1f..50f,
                        modifier = Modifier.fillMaxWidth(),
                    )
                }

                Column {
                    Text(
                        "CFG Scale: %.1f".format(cfg),
                        style = MaterialTheme.typography.bodyMedium,
                    )
                    Slider(
                        value = cfg,
                        onValueChange = { onCfgChange((it * 10f).roundToInt() / 10f) },
                        valueRange = 1f..30f,
                        modifier = Modifier.fillMaxWidth(),
                    )
                }
                HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant)
                Text("Output & runtime", style = MaterialTheme.typography.titleSmall)
                // The free 128-512 slider belongs to SD1.5 CPU. SDXL MNN also runs
                // on CPU/GPU but renders on the fixed 1024 canvas like every other
                // SDXL package, so it must not be offered a range it cannot use.
                // The runtime picker below stays for both.
                if (runOnCpu && !isSdxl) {
                    Column {
                        Text(
                            stringResource(
                                R.string.image_size,
                                currentWidth,
                                currentHeight,
                            ),
                            style = MaterialTheme.typography.bodyMedium,
                        )
                        Slider(
                            value = currentWidth.toFloat(),
                            onValueChange = onSizeChange,
                            valueRange = 128f..512f,
                            steps = 5,
                            modifier = Modifier.fillMaxWidth(),
                        )
                    }
                }
                if (runOnCpu) {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(
                            ButtonGroupDefaults.ConnectedSpaceBetween,
                        ),
                    ) {
                        Text(
                            "Runtime",
                            style = MaterialTheme.typography.bodyMedium,
                            modifier = Modifier.padding(end = 4.dp),
                        )
                        ToggleButton(
                            checked = !useOpenCL,
                            onCheckedChange = { checked ->
                                if (checked) onCpuSelected()
                            },
                            shapes = ButtonGroupDefaults.connectedLeadingButtonShapes(),
                            modifier = Modifier.weight(1f),
                        ) {
                            Text("CPU")
                        }
                        ToggleButton(
                            checked = useOpenCL,
                            onCheckedChange = { checked ->
                                if (checked) onGpuSelected()
                            },
                            shapes = ButtonGroupDefaults.connectedTrailingButtonShapes(),
                            modifier = Modifier.weight(1f),
                        ) {
                            Text("GPU")
                        }
                    }
                }
                if (isQwen21) {
                    Column(
                        modifier = Modifier.fillMaxWidth(),
                        verticalArrangement = Arrangement.spacedBy(6.dp),
                    ) {
                        Text(
                            "HTP sessions",
                            style = MaterialTheme.typography.bodyMedium,
                        )
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.spacedBy(
                                ButtonGroupDefaults.ConnectedSpaceBetween,
                            ),
                        ) {
                            listOf(
                                "auto" to "Auto",
                                "single" to "Single",
                                "dual" to "Dual",
                            ).forEachIndexed { index, (mode, label) ->
                                ToggleButton(
                                    checked = htpMode == mode,
                                    onCheckedChange = { checked ->
                                        if (checked && !isRunning) onHtpModeChange(mode)
                                    },
                                    enabled = !isRunning,
                                    shapes = when (index) {
                                        0 -> ButtonGroupDefaults.connectedLeadingButtonShapes()
                                        2 -> ButtonGroupDefaults.connectedTrailingButtonShapes()
                                        else -> ButtonGroupDefaults.connectedMiddleButtonShapes()
                                    },
                                    modifier = Modifier.weight(1f),
                                ) {
                                    Text(label)
                                }
                            }
                        }
                        Text(
                            if (htpMode == "dual") {
                                "Dual is experimental. Strict split barriers are enabled, but output can still differ from Single."
                            } else {
                                "Auto uses one HTP session for correctness. Use Dual only for A/B performance testing."
                            },
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                    }
                }

                Column {
                    Text(
                        stringResource(R.string.batch_count, batchCounts),
                        style = MaterialTheme.typography.bodyMedium,
                    )
                    Slider(
                        value = batchCounts.toFloat(),
                        onValueChange = onBatchCountsChange,
                        valueRange = 1f..10f,
                        steps = 8,
                        modifier = Modifier.fillMaxWidth(),
                    )
                }
                if (useImg2img) {
                    Column {
                        Text(
                            "[img2img]Denoise Strength: %.2f".format(denoiseStrength),
                            style = MaterialTheme.typography.bodyMedium,
                        )
                        Slider(
                            value = denoiseStrength,
                            onValueChange = onDenoiseStrengthChange,
                            valueRange = 0f..1f,
                            steps = 99,
                            modifier = Modifier.fillMaxWidth(),
                        )
                    }
                }
                HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant)
                Text("Seed", style = MaterialTheme.typography.titleSmall)
                Column(
                    modifier = Modifier.fillMaxWidth(),
                    verticalArrangement = Arrangement.spacedBy(8.dp),
                ) {
                    OutlinedTextField(
                        value = seed,
                        onValueChange = onSeedChange,
                        label = { Text(stringResource(R.string.random_seed)) },
                        keyboardOptions = KeyboardOptions(
                            keyboardType = KeyboardType.Number,
                        ),
                        modifier = Modifier.fillMaxWidth(),
                        shape = MaterialTheme.shapes.medium,
                        trailingIcon = {
                            if (seed.isNotEmpty()) {
                                IconButton(onClick = { onSeedChange("") }) {
                                    Icon(
                                        Icons.Default.Clear,
                                        contentDescription = "clear",
                                    )
                                }
                            }
                        },
                    )

                    if (returnedSeed != null) {
                        FilledTonalButton(
                            onClick = onUseLastSeed,
                            modifier = Modifier.fillMaxWidth(),
                        ) {
                            Icon(
                                Icons.Default.Refresh,
                                contentDescription = stringResource(R.string.use_last_seed),
                                modifier = Modifier
                                    .size(20.dp)
                                    .padding(end = 4.dp),
                            )
                            Text(
                                stringResource(
                                    R.string.use_last_seed,
                                    returnedSeed.toString(),
                                ),
                            )
                        }
                    }
                }
            }
        },
        confirmButton = {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
            ) {
                TextButton(
                    onClick = onReset,
                    colors = ButtonDefaults.textButtonColors(
                        contentColor = MaterialTheme.colorScheme.error,
                    ),
                ) {
                    Icon(
                        Icons.Default.Refresh,
                        contentDescription = stringResource(R.string.reset),
                        modifier = Modifier
                            .size(20.dp)
                            .padding(end = 4.dp),
                    )
                    Text(stringResource(R.string.reset))
                }

                TextButton(onClick = onDismiss) {
                    Text(stringResource(R.string.confirm))
                }
            }
        },
    )
}
