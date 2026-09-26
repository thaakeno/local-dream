package io.github.xororz.localdream.ui.components

import android.media.MediaPlayer
import androidx.compose.animation.AnimatedContent
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.togetherWith
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Pause
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.Replay
import androidx.compose.material3.ElevatedCard
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Slider
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import java.io.File
import kotlinx.coroutines.delay

@Composable
fun MusicPlayerCard(
    file: File,
    title: String,
    subtitle: String,
    modifier: Modifier = Modifier,
) {
    val player = remember(file.absolutePath) {
        MediaPlayer().apply {
            setDataSource(file.absolutePath)
            prepare()
        }
    }
    var playing by remember(player) { mutableStateOf(false) }
    var position by remember(player) { mutableIntStateOf(0) }
    val duration = remember(player) { player.duration.coerceAtLeast(1) }

    DisposableEffect(player) {
        player.setOnCompletionListener {
            playing = false
            position = 0
        }
        onDispose {
            runCatching { player.stop() }
            player.release()
        }
    }

    LaunchedEffect(playing, player) {
        while (playing) {
            position = runCatching { player.currentPosition }.getOrDefault(position)
            delay(120)
        }
    }

    ElevatedCard(
        modifier = modifier.fillMaxWidth(),
        shape = MaterialTheme.shapes.extraLarge,
        colors = CardDefaults.elevatedCardColors(
            containerColor = MaterialTheme.colorScheme.surfaceContainer,
        ),
    ) {
        Column(
            modifier = Modifier.padding(18.dp),
            verticalArrangement = Arrangement.spacedBy(14.dp),
        ) {
            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(14.dp),
            ) {
                Surface(
                    modifier = Modifier.size(72.dp),
                    shape = MaterialTheme.shapes.extraLarge,
                    color = MaterialTheme.colorScheme.tertiaryContainer,
                ) {
                    EqualizerGlyph(
                        active = playing,
                        modifier = Modifier.padding(18.dp),
                    )
                }
                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        title,
                        style = MaterialTheme.typography.titleLarge,
                        fontWeight = FontWeight.SemiBold,
                    )
                    Text(
                        subtitle,
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
                Surface(
                    shape = CircleShape,
                    color = MaterialTheme.colorScheme.primary,
                ) {
                    IconButton(
                        onClick = {
                            if (playing) {
                                player.pause()
                                playing = false
                            } else {
                                if (position >= duration - 200) {
                                    player.seekTo(0)
                                    position = 0
                                }
                                player.start()
                                playing = true
                            }
                        },
                    ) {
                        AnimatedContent(
                            targetState = playing,
                            transitionSpec = { fadeIn(tween(140)) togetherWith fadeOut(tween(100)) },
                            label = "musicPlayPause",
                        ) { isPlaying ->
                            Icon(
                                if (isPlaying) Icons.Default.Pause else Icons.Default.PlayArrow,
                                contentDescription = if (isPlaying) "Pause" else "Play",
                                tint = MaterialTheme.colorScheme.onPrimary,
                            )
                        }
                    }
                }
            }

            Slider(
                value = position.coerceIn(0, duration).toFloat(),
                onValueChange = {
                    position = it.toInt()
                    player.seekTo(position)
                },
                valueRange = 0f..duration.toFloat(),
                modifier = Modifier.fillMaxWidth(),
            )

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
                        onClick = {
                            player.seekTo(0)
                            position = 0
                            if (!playing) {
                                player.start()
                                playing = true
                            }
                        },
                    ) {
                        Icon(Icons.Default.Replay, contentDescription = "Replay")
                    }
                    Text(
                        formatTime(duration),
                        style = MaterialTheme.typography.labelMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
            }
        }
    }
}

@Composable
private fun EqualizerGlyph(active: Boolean, modifier: Modifier = Modifier) {
    val transition = rememberInfiniteTransition(label = "equalizer")
    val levels = List(5) { index ->
        val animated by transition.animateFloat(
            initialValue = 0.28f + index * 0.04f,
            targetValue = 0.95f - index * 0.06f,
            animationSpec = infiniteRepeatable(
                animation = tween(360 + index * 95),
                repeatMode = RepeatMode.Reverse,
            ),
            label = "bar$index",
        )
        if (active) animated else 0.42f + (index % 3) * 0.12f
    }

    Row(
        modifier = modifier
            .fillMaxWidth()
            .heightIn(min = 32.dp),
        horizontalArrangement = Arrangement.spacedBy(4.dp, Alignment.CenterHorizontally),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        levels.forEach { level ->
            Box(
                modifier = Modifier
                    .width(4.dp)
                    .height((34f * level).dp)
                    .clip(CircleShape)
                    .background(MaterialTheme.colorScheme.onTertiaryContainer),
            )
        }
    }
}

private fun formatTime(ms: Int): String {
    val total = (ms.coerceAtLeast(0) / 1000)
    return "${total / 60}:${(total % 60).toString().padStart(2, '0')}"
}
