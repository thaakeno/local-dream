package io.github.xororz.localdream.ui.components

import android.content.Context
import android.media.MediaPlayer
import android.os.Build
import android.os.SystemClock
import androidx.compose.animation.AnimatedContent
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.togetherWith
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.gestures.detectDragGestures
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.Download
import androidx.compose.material.icons.filled.Favorite
import androidx.compose.material.icons.filled.FavoriteBorder
import androidx.compose.material.icons.filled.GraphicEq
import androidx.compose.material.icons.filled.Pause
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.Replay
import androidx.compose.material.icons.filled.Share
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import java.io.BufferedInputStream
import java.io.BufferedOutputStream
import java.io.DataInputStream
import java.io.DataOutputStream
import java.io.File
import java.io.RandomAccessFile
import java.security.MessageDigest
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.withContext
import kotlin.math.abs
import kotlin.math.max

@Composable
fun MusicPlayerCard(
    file: File,
    title: String,
    subtitle: String,
    modifier: Modifier = Modifier,
    metadataLine: String? = null,
    onSave: (() -> Unit)? = null,
    onShare: (() -> Unit)? = null,
    onDelete: (() -> Unit)? = null,
    onReproduce: (() -> Unit)? = null,
    onUse: (() -> Unit)? = null,
    favorite: Boolean = false,
    onFavoriteToggle: (() -> Unit)? = null,
    badges: List<String> = emptyList(),
    compact: Boolean = false,
) {
    var player by remember(file.absolutePath) { mutableStateOf<MediaPlayer?>(null) }
    var preparing by remember(file.absolutePath) { mutableStateOf(false) }
    var prepared by remember(file.absolutePath) { mutableStateOf(false) }
    var playing by remember(file.absolutePath) { mutableStateOf(false) }
    var position by remember(file.absolutePath) { mutableIntStateOf(0) }
    var duration by remember(file.absolutePath) { mutableIntStateOf(1) }
    var scrubFraction by remember(file.absolutePath) { mutableStateOf<Float?>(null) }
    var resumeAfterScrub by remember(file.absolutePath) { mutableStateOf(false) }
    var suppressPositionPollingUntil by remember(file.absolutePath) { mutableLongStateOf(0L) }

    val context = LocalContext.current
    val waveformKey = remember(file.absolutePath, file.lastModified(), file.length()) {
        SpeechWaveformCache.key(file, 112)
    }
    val waveform by produceState<List<Float>>(
        initialValue = SpeechWaveformCache.peek(waveformKey) ?: emptyList(),
        key1 = waveformKey,
    ) {
        if (value.isEmpty()) {
            value = withContext(Dispatchers.IO) {
                SpeechWaveformCache.load(context, file, 112, waveformKey)
            }
        }
    }

    fun releasePlayer() {
        player?.let { current ->
            runCatching { current.stop() }
            runCatching { current.reset() }
            current.release()
        }
        player = null
        preparing = false
        prepared = false
        playing = false
        position = 0
        duration = 1
    }

    fun ensurePreparedAndPlay() {
        val current = player
        if (prepared && current != null) {
            if (position >= duration - 200) {
                current.seekTo(0)
                position = 0
            }
            current.start()
            playing = true
            return
        }
        if (preparing) return

        preparing = true
        val created = MediaPlayer()
        player = created
        created.setOnPreparedListener { ready ->
            duration = ready.duration.coerceAtLeast(1)
            prepared = true
            preparing = false
            ready.start()
            playing = true
        }
        created.setOnCompletionListener {
            playing = false
            position = duration
        }
        created.setOnErrorListener { _, _, _ ->
            preparing = false
            prepared = false
            playing = false
            true
        }
        runCatching {
            created.setDataSource(file.absolutePath)
            created.prepareAsync()
        }.onFailure { releasePlayer() }
    }

    DisposableEffect(file.absolutePath) {
        onDispose { releasePlayer() }
    }

    LaunchedEffect(playing, player, scrubFraction) {
        while (playing) {
            if (scrubFraction == null && SystemClock.uptimeMillis() >= suppressPositionPollingUntil) {
                position = runCatching { player?.currentPosition ?: position }.getOrDefault(position)
            }
            delay(33)
        }
    }

    val cardPadding = if (compact) 10.dp else 18.dp
    val cardSpacing = if (compact) 6.dp else 14.dp
    val leadingSize = if (compact) 38.dp else 64.dp
    val leadingPadding = if (compact) 9.dp else 18.dp
    val waveformHeight = if (compact) 40.dp else 76.dp

    ElevatedCard(
        modifier = modifier.fillMaxWidth(),
        shape = if (compact) MaterialTheme.shapes.large else MaterialTheme.shapes.extraLarge,
        colors = CardDefaults.elevatedCardColors(
            containerColor = MaterialTheme.colorScheme.surfaceContainer,
        ),
    ) {
        Column(
            modifier = Modifier.padding(cardPadding),
            verticalArrangement = Arrangement.spacedBy(cardSpacing),
        ) {
            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(if (compact) 9.dp else 14.dp),
            ) {
                Surface(
                    modifier = Modifier.size(leadingSize),
                    shape = if (compact) MaterialTheme.shapes.large else MaterialTheme.shapes.extraLarge,
                    color = MaterialTheme.colorScheme.tertiaryContainer,
                ) {
                    Icon(
                        Icons.Default.GraphicEq,
                        contentDescription = null,
                        modifier = Modifier.padding(leadingPadding),
                        tint = MaterialTheme.colorScheme.onTertiaryContainer,
                    )
                }
                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        title,
                        style = if (compact) {
                            MaterialTheme.typography.titleSmall
                        } else {
                            MaterialTheme.typography.titleMedium
                        },
                        fontWeight = FontWeight.SemiBold,
                        maxLines = if (compact) 2 else 2,
                    )
                    Text(
                        subtitle,
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        maxLines = if (compact) 1 else 2,
                    )
                    metadataLine?.takeIf { it.isNotBlank() }?.let {
                        Text(
                            it,
                            style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.primary,
                            maxLines = if (compact) 1 else 2,
                        )
                    }
                    if (badges.isNotEmpty()) {
                        Row(
                            horizontalArrangement = Arrangement.spacedBy(6.dp),
                            verticalAlignment = Alignment.CenterVertically,
                        ) {
                            badges.take(if (compact) 2 else 3).forEachIndexed { index, badge ->
                                Surface(
                                    shape = CircleShape,
                                    color = when (index) {
                                        0 -> MaterialTheme.colorScheme.surfaceContainerHighest
                                        1 -> MaterialTheme.colorScheme.secondaryContainer
                                        else -> MaterialTheme.colorScheme.tertiaryContainer
                                    },
                                ) {
                                    Text(
                                        badge,
                                        modifier = Modifier.padding(horizontal = 8.dp, vertical = 3.dp),
                                        style = MaterialTheme.typography.labelSmall,
                                        color = when (index) {
                                            1 -> MaterialTheme.colorScheme.onSecondaryContainer
                                            2 -> MaterialTheme.colorScheme.onTertiaryContainer
                                            else -> MaterialTheme.colorScheme.onSurfaceVariant
                                        },
                                    )
                                }
                            }
                        }
                    }
                    if (preparing) {
                        Text(
                            "Preparing playback…",
                            style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.primary,
                        )
                    }
                }
                if (onFavoriteToggle != null) {
                    IconButton(onClick = onFavoriteToggle) {
                        Icon(
                            if (favorite) Icons.Default.Favorite else Icons.Default.FavoriteBorder,
                            contentDescription = if (favorite) "Remove favorite" else "Add favorite",
                            tint = if (favorite) MaterialTheme.colorScheme.primary
                            else MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                    }
                }
                Surface(shape = CircleShape, color = MaterialTheme.colorScheme.primary) {
                    IconButton(
                        enabled = !preparing,
                        onClick = {
                            val current = player
                            if (playing && current != null) {
                                current.pause()
                                playing = false
                            } else {
                                ensurePreparedAndPlay()
                            }
                        },
                    ) {
                        AnimatedContent(
                            targetState = playing,
                            transitionSpec = {
                                fadeIn(tween(120)) togetherWith fadeOut(tween(90))
                            },
                            label = "speechPlayPause",
                        ) { active ->
                            Icon(
                                if (active) Icons.Default.Pause else Icons.Default.PlayArrow,
                                contentDescription = if (active) "Pause" else "Play",
                                tint = MaterialTheme.colorScheme.onPrimary,
                            )
                        }
                    }
                }
            }

            if (waveform.isNotEmpty()) {
                RealAudioWaveform(
                    peaks = waveform,
                    progress = scrubFraction
                        ?: if (duration > 0) position.toFloat() / duration else 0f,
                    playing = playing,
                    enabled = prepared,
                    onScrubStart = { fraction ->
                        if (prepared) {
                            resumeAfterScrub = playing
                            if (playing) {
                                player?.pause()
                                playing = false
                            }
                            scrubFraction = fraction
                            position = (duration * fraction).toInt().coerceIn(0, duration)
                        }
                    },
                    onScrub = { fraction ->
                        if (prepared) {
                            scrubFraction = fraction
                            position = (duration * fraction).toInt().coerceIn(0, duration)
                        }
                    },
                    onScrubEnd = { fraction ->
                        if (prepared) {
                            val target = (duration * fraction).toInt().coerceIn(0, duration)
                            position = target
                            suppressPositionPollingUntil = SystemClock.uptimeMillis() + 250L
                            val current = player
                            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                                current?.seekTo(target.toLong(), MediaPlayer.SEEK_CLOSEST)
                            } else {
                                @Suppress("DEPRECATION")
                                current?.seekTo(target)
                            }
                            scrubFraction = null
                            if (resumeAfterScrub && current != null) {
                                current.start()
                                playing = true
                            }
                            resumeAfterScrub = false
                        }
                    },
                    onTap = { fraction ->
                        if (prepared) {
                            val target = (duration * fraction).toInt().coerceIn(0, duration)
                            position = target
                            suppressPositionPollingUntil = SystemClock.uptimeMillis() + 200L
                            val current = player
                            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                                current?.seekTo(target.toLong(), MediaPlayer.SEEK_CLOSEST)
                            } else {
                                @Suppress("DEPRECATION")
                                current?.seekTo(target)
                            }
                        }
                    },
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(waveformHeight),
                )
            } else {
                LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
            }

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Text(
                    formatTime(position),
                    style = MaterialTheme.typography.labelMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
                Row(verticalAlignment = Alignment.CenterVertically) {
                    IconButton(
                        enabled = prepared,
                        modifier = if (compact) Modifier.size(36.dp) else Modifier,
                        onClick = {
                            player?.seekTo(0)
                            position = 0
                            if (!playing) ensurePreparedAndPlay()
                        },
                    ) {
                        Icon(Icons.Default.Replay, contentDescription = "Replay")
                    }
                    Text(
                        if (prepared) formatTime(duration) else "--:--",
                        style = MaterialTheme.typography.labelMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
            }

            if (compact) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    if (onUse != null) {
                        OutlinedButton(
                            onClick = onUse,
                            modifier = Modifier.weight(1f),
                            contentPadding = PaddingValues(horizontal = 9.dp, vertical = 6.dp),
                        ) {
                            Text("Use")
                        }
                    }
                    if (onReproduce != null) {
                        Button(
                            onClick = onReproduce,
                            modifier = Modifier.weight(1f),
                            contentPadding = PaddingValues(horizontal = 9.dp, vertical = 6.dp),
                        ) {
                            Icon(Icons.Default.Replay, contentDescription = null)
                            Spacer(Modifier.width(4.dp))
                            Text("Reproduce")
                        }
                    }
                    if (onSave != null) {
                        IconButton(onClick = onSave, modifier = Modifier.size(40.dp)) {
                            Icon(Icons.Default.Download, contentDescription = "Save")
                        }
                    }
                    if (onShare != null) {
                        IconButton(onClick = onShare, modifier = Modifier.size(40.dp)) {
                            Icon(Icons.Default.Share, contentDescription = "Share")
                        }
                    }
                    if (onDelete != null) {
                        IconButton(onClick = onDelete, modifier = Modifier.size(40.dp)) {
                            Icon(
                                Icons.Default.Delete,
                                contentDescription = "Delete",
                                tint = MaterialTheme.colorScheme.error,
                            )
                        }
                    }
                }
            } else {
                if (onReproduce != null || onUse != null) {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(10.dp),
                    ) {
                        if (onUse != null) {
                            OutlinedButton(onClick = onUse, modifier = Modifier.weight(1f)) {
                                Text("Use settings")
                            }
                        }
                        if (onReproduce != null) {
                            Button(onClick = onReproduce, modifier = Modifier.weight(1f)) {
                                Icon(Icons.Default.Replay, contentDescription = null)
                                Spacer(Modifier.width(6.dp))
                                Text("Reproduce")
                            }
                        }
                    }
                }

                if (onSave != null || onShare != null) {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(10.dp),
                    ) {
                        if (onSave != null) {
                            OutlinedButton(onClick = onSave, modifier = Modifier.weight(1f)) {
                                Icon(Icons.Default.Download, contentDescription = null)
                                Spacer(Modifier.width(6.dp))
                                Text("Save")
                            }
                        }
                        if (onShare != null) {
                            Button(onClick = onShare, modifier = Modifier.weight(1f)) {
                                Icon(Icons.Default.Share, contentDescription = null)
                                Spacer(Modifier.width(6.dp))
                                Text("Share")
                            }
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun RealAudioWaveform(
    peaks: List<Float>,
    progress: Float,
    playing: Boolean,
    enabled: Boolean,
    onScrubStart: (Float) -> Unit,
    onScrub: (Float) -> Unit,
    onScrubEnd: (Float) -> Unit,
    onTap: (Float) -> Unit,
    modifier: Modifier = Modifier,
) {
    val playedColor = MaterialTheme.colorScheme.primary
    val idleColor = MaterialTheme.colorScheme.outlineVariant
    val playheadColor = MaterialTheme.colorScheme.onSurface
    val transition = rememberInfiniteTransition(label = "waveformPulse")
    val pulse by transition.animateFloat(
        initialValue = 0.94f,
        targetValue = 1.06f,
        animationSpec = infiniteRepeatable(
            animation = tween(420),
            repeatMode = RepeatMode.Reverse,
        ),
        label = "waveformPulseValue",
    )
    val clamped = progress.coerceIn(0f, 1f)
    var dragFraction by remember { mutableStateOf<Float?>(null) }
    val displayed = dragFraction ?: clamped

    Canvas(
        modifier = modifier
            .clip(MaterialTheme.shapes.large)
            .background(MaterialTheme.colorScheme.surfaceContainerLow)
            .pointerInput(enabled, peaks) {
                if (enabled) {
                    detectDragGestures(
                        onDragStart = { offset ->
                            val fraction = (offset.x / size.width.toFloat()).coerceIn(0f, 1f)
                            dragFraction = fraction
                            onScrubStart(fraction)
                        },
                        onDrag = { change, _ ->
                            change.consume()
                            val fraction =
                                (change.position.x / size.width.toFloat()).coerceIn(0f, 1f)
                            dragFraction = fraction
                            onScrub(fraction)
                        },
                        onDragEnd = {
                            val fraction = dragFraction ?: displayed
                            dragFraction = null
                            onScrubEnd(fraction)
                        },
                        onDragCancel = {
                            val fraction = dragFraction ?: displayed
                            dragFraction = null
                            onScrubEnd(fraction)
                        },
                    )
                }
            }
            .pointerInput(enabled, peaks) {
                if (enabled) {
                    detectTapGestures { offset ->
                        val fraction = (offset.x / size.width.toFloat()).coerceIn(0f, 1f)
                        onTap(fraction)
                    }
                }
            }
            .padding(horizontal = 8.dp, vertical = 10.dp),
    ) {
        if (peaks.isEmpty()) return@Canvas
        val step = size.width / peaks.size.toFloat()
        val stroke = max(1.4f, step * 0.48f)
        val center = size.height / 2f
        val maxHalf = size.height * 0.44f
        val playX = size.width * displayed

        peaks.forEachIndexed { index, raw ->
            val x = step * (index + 0.5f)
            var level = raw.coerceIn(0.04f, 1f)
            if (playing && dragFraction == null && abs(x - playX) < step * 3f) {
                level *= pulse
            }
            val half = maxHalf * level.coerceAtMost(1f)
            drawLine(
                color = if (x <= playX) playedColor else idleColor,
                start = androidx.compose.ui.geometry.Offset(x, center - half),
                end = androidx.compose.ui.geometry.Offset(x, center + half),
                strokeWidth = stroke,
                cap = StrokeCap.Round,
            )
        }
        if (enabled) {
            drawLine(
                color = playheadColor,
                start = androidx.compose.ui.geometry.Offset(playX, 4f),
                end = androidx.compose.ui.geometry.Offset(playX, size.height - 4f),
                strokeWidth = 1.6.dp.toPx(),
                cap = StrokeCap.Round,
            )
            drawCircle(
                color = playheadColor,
                radius = 3.8.dp.toPx(),
                center = androidx.compose.ui.geometry.Offset(playX, center),
            )
        }
    }
}

private object SpeechWaveformCache {
    private const val MAX_MEMORY_ITEMS = 96
    private const val CACHE_VERSION = 1

    private val memory = object : LinkedHashMap<String, List<Float>>(
        MAX_MEMORY_ITEMS,
        0.75f,
        true,
    ) {
        override fun removeEldestEntry(
            eldest: MutableMap.MutableEntry<String, List<Float>>?,
        ): Boolean = size > MAX_MEMORY_ITEMS
    }

    fun key(file: File, buckets: Int): String {
        val raw = buildString {
            append(file.absolutePath)
            append('|')
            append(file.length())
            append('|')
            append(file.lastModified())
            append('|')
            append(buckets)
        }
        val digest = MessageDigest.getInstance("SHA-256")
            .digest(raw.toByteArray(Charsets.UTF_8))
        return digest.take(12).joinToString("") { "%02x".format(it) }
    }

    @Synchronized
    fun peek(key: String): List<Float>? = memory[key]

    private fun cacheFile(context: Context, key: String): File {
        val root = File(context.cacheDir, "speech_waveforms")
        if (!root.exists()) root.mkdirs()
        return File(root, "v${CACHE_VERSION}_${key}.bin")
    }

    fun load(
        context: Context,
        file: File,
        buckets: Int,
        key: String = key(file, buckets),
    ): List<Float> {
        synchronized(this) {
            memory[key]?.let { return it }
        }

        val disk = cacheFile(context, key)
        val cached = runCatching {
            if (!disk.isFile) return@runCatching null
            DataInputStream(BufferedInputStream(disk.inputStream())).use { input ->
                val version = input.readInt()
                val count = input.readInt()
                if (version != CACHE_VERSION || count != buckets) return@use null
                List(count) { input.readFloat() }
            }
        }.getOrNull()
        if (cached != null) {
            synchronized(this) { memory[key] = cached }
            return cached
        }

        val peaks = readPcmWaveform(file, buckets)
        if (peaks.isNotEmpty()) {
            val temp = File(disk.parentFile, disk.name + ".tmp")
            runCatching {
                DataOutputStream(BufferedOutputStream(temp.outputStream())).use { output ->
                    output.writeInt(CACHE_VERSION)
                    output.writeInt(peaks.size)
                    peaks.forEach(output::writeFloat)
                }
                if (disk.exists()) disk.delete()
                if (!temp.renameTo(disk)) {
                    temp.copyTo(disk, overwrite = true)
                    temp.delete()
                }
            }.onFailure { temp.delete() }
            synchronized(this) { memory[key] = peaks }
        }
        return peaks
    }
}

private fun readPcmWaveform(file: File, buckets: Int): List<Float> {
    if (!file.isFile || file.length() <= 44L || buckets <= 0) return emptyList()
    return runCatching {
        RandomAccessFile(file, "r").use { raf ->
            val dataBytes = (raf.length() - 44L).coerceAtLeast(0L)
            val samples = dataBytes / 2L
            if (samples <= 0L) return@use emptyList<Float>()
            val bucketSamples = max(1L, samples / buckets.toLong())
            val peaks = FloatArray(buckets)
            raf.seek(44L)
            var sampleIndex = 0L
            while (sampleIndex < samples) {
                val lo = raf.read()
                val hi = raf.read()
                if (lo < 0 || hi < 0) break
                var value = lo or (hi shl 8)
                if (value >= 0x8000) value -= 0x10000
                val bucket = (sampleIndex / bucketSamples)
                    .toInt()
                    .coerceIn(0, buckets - 1)
                val amplitude = abs(value) / 32768f
                if (amplitude > peaks[bucket]) peaks[bucket] = amplitude
                sampleIndex++
            }
            val maxPeak = peaks.maxOrNull()?.coerceAtLeast(0.001f) ?: 1f
            peaks.map { (it / maxPeak).coerceIn(0.04f, 1f) }
        }
    }.getOrElse { emptyList() }
}

private fun formatTime(ms: Int): String {
    val total = ms.coerceAtLeast(0) / 1000
    return "${total / 60}:${(total % 60).toString().padStart(2, '0')}"
}
