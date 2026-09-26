package io.github.xororz.localdream.ui.components

import android.graphics.Bitmap
import androidx.activity.compose.BackHandler
import androidx.compose.animation.core.Animatable
import androidx.compose.animation.core.tween
import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.Orientation
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.gestures.draggable
import androidx.compose.foundation.gestures.rememberDraggableState
import androidx.compose.foundation.gestures.rememberTransformableState
import androidx.compose.foundation.gestures.transformable
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.pager.HorizontalPager
import androidx.compose.foundation.pager.rememberPagerState
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Close
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.layout.onSizeChanged
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import coil.compose.AsyncImage
import coil.request.ImageRequest
import io.github.xororz.localdream.data.HistoryItem
import kotlin.math.abs

private data class ResultGalleryEntry(
    val stableKey: String,
    val bitmap: Bitmap?,
    val historyItem: HistoryItem?,
)

@Composable
fun ResultGalleryOverlay(
    currentBitmap: Bitmap,
    currentHistoryId: Long?,
    recentHistory: List<HistoryItem>,
    onDismiss: () -> Unit,
    onHistoryItemChanged: (HistoryItem) -> Unit,
) {
    val entries = remember {
        val history = recentHistory.map {
            ResultGalleryEntry(
                stableKey = "history:${it.id}",
                bitmap = null,
                historyItem = it,
            )
        }
        val existing = currentHistoryId?.let { id ->
            history.indexOfFirst { it.historyItem?.id == id }
        } ?: -1
        if (existing >= 0) {
            history
        } else {
            listOf(
                ResultGalleryEntry(
                    stableKey = "current",
                    bitmap = currentBitmap,
                    historyItem = null,
                ),
            ) + history
        }
    }
    if (entries.isEmpty()) return

    val initialPage = remember {
        currentHistoryId?.let { id ->
            entries.indexOfFirst { it.historyItem?.id == id }.takeIf { it >= 0 }
        } ?: 0
    }
    val pagerState = rememberPagerState(
        initialPage = initialPage,
        pageCount = { entries.size },
    )
    var zoomedPage by remember { mutableStateOf<Int?>(null) }

    BackHandler(onBack = onDismiss)

    LaunchedEffect(pagerState.currentPage, entries) {
        zoomedPage = null
        entries.getOrNull(pagerState.currentPage)?.historyItem?.let(onHistoryItemChanged)
    }

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(Color.Black),
    ) {
        HorizontalPager(
            state = pagerState,
            userScrollEnabled = zoomedPage == null,
            modifier = Modifier.fillMaxSize(),
        ) { page ->
            ResultGalleryPhoto(
                entry = entries[page],
                onDismiss = onDismiss,
                onZoomChanged = { zoomed ->
                    if (page == pagerState.currentPage) {
                        zoomedPage = if (zoomed) page else null
                    }
                },
            )
        }

        Surface(
            modifier = Modifier
                .align(Alignment.TopStart)
                .padding(top = 48.dp, start = 14.dp),
            color = Color.Black.copy(alpha = 0.48f),
            shape = MaterialTheme.shapes.extraLarge,
        ) {
            IconButton(onClick = onDismiss) {
                Icon(
                    Icons.Default.Close,
                    contentDescription = "Close",
                    tint = Color.White,
                )
            }
        }

        if (entries.size > 1) {
            Surface(
                modifier = Modifier
                    .align(Alignment.BottomCenter)
                    .padding(bottom = 24.dp),
                color = Color.Black.copy(alpha = 0.5f),
                shape = MaterialTheme.shapes.extraLarge,
            ) {
                Text(
                    text = "${pagerState.currentPage + 1} / ${entries.size}",
                    modifier = Modifier.padding(horizontal = 14.dp, vertical = 7.dp),
                    color = Color.White,
                    style = MaterialTheme.typography.labelLarge,
                )
            }
        }
    }
}

@Composable
private fun ResultGalleryPhoto(
    entry: ResultGalleryEntry,
    onDismiss: () -> Unit,
    onZoomChanged: (Boolean) -> Unit,
) {
    val context = LocalContext.current
    var scale by remember(entry.stableKey) { mutableFloatStateOf(1f) }
    var panX by remember(entry.stableKey) { mutableFloatStateOf(0f) }
    var panY by remember(entry.stableKey) { mutableFloatStateOf(0f) }
    var dismissDrag by remember(entry.stableKey) { mutableFloatStateOf(0f) }
    var viewportWidth by remember(entry.stableKey) { mutableFloatStateOf(1f) }
    var viewportHeight by remember(entry.stableKey) { mutableFloatStateOf(1f) }
    val dragAlpha = (1f - abs(dismissDrag) / 900f).coerceIn(0.35f, 1f)

    val transformState = rememberTransformableState { zoomChange, panChange, _ ->
        val oldScale = scale
        val newScale = (oldScale * zoomChange).coerceIn(1f, 5f)
        scale = newScale
        onZoomChanged(newScale > 1.01f)
        if (newScale > 1.01f) {
            val maxPanX = viewportWidth * (newScale - 1f) / 2f
            val maxPanY = viewportHeight * (newScale - 1f) / 2f
            panX = (panX + panChange.x).coerceIn(-maxPanX, maxPanX)
            panY = (panY + panChange.y).coerceIn(-maxPanY, maxPanY)
        } else {
            panX = 0f
            panY = 0f
        }
    }
    val verticalDragState = rememberDraggableState { delta ->
        if (scale <= 1.01f) {
            dismissDrag = (dismissDrag + delta).coerceAtLeast(0f)
        }
    }

    Box(
        modifier = Modifier
            .fillMaxSize()
            .onSizeChanged {
                viewportWidth = it.width.toFloat().coerceAtLeast(1f)
                viewportHeight = it.height.toFloat().coerceAtLeast(1f)
            }
            .graphicsLayer { alpha = dragAlpha }
            .draggable(
                state = verticalDragState,
                orientation = Orientation.Vertical,
                enabled = scale <= 1.01f,
                onDragStopped = {
                    if (dismissDrag > 180f) {
                        onDismiss()
                    } else if (dismissDrag != 0f) {
                        val settle = Animatable(dismissDrag)
                        settle.animateTo(
                            targetValue = 0f,
                            animationSpec = tween(180),
                        ) {
                            dismissDrag = value
                        }
                    }
                },
            ),
        contentAlignment = Alignment.Center,
    ) {
        AsyncImage(
            model = ImageRequest.Builder(context)
                .data(entry.bitmap ?: entry.historyItem?.imageFile)
                .size(coil.size.Size.ORIGINAL)
                .crossfade(false)
                .build(),
            contentDescription = "Generated image",
            modifier = Modifier
                .fillMaxSize()
                .graphicsLayer {
                    scaleX = scale
                    scaleY = scale
                    translationX = panX
                    translationY = panY + dismissDrag
                }
                .pointerInput(entry.stableKey) {
                    detectTapGestures(
                        onDoubleTap = { tap ->
                            if (scale > 1.05f) {
                                scale = 1f
                                panX = 0f
                                panY = 0f
                                onZoomChanged(false)
                            } else {
                                scale = 2.5f
                                val dx = tap.x - size.width / 2f
                                val dy = tap.y - size.height / 2f
                                val maxPanX = viewportWidth * (scale - 1f) / 2f
                                val maxPanY = viewportHeight * (scale - 1f) / 2f
                                panX = (-dx * 0.75f).coerceIn(-maxPanX, maxPanX)
                                panY = (-dy * 0.75f).coerceIn(-maxPanY, maxPanY)
                                onZoomChanged(true)
                            }
                        },
                    )
                }
                .transformable(
                    state = transformState,
                    canPan = { _: Offset -> scale > 1.01f },
                ),
            contentScale = ContentScale.Fit,
        )
    }
}
