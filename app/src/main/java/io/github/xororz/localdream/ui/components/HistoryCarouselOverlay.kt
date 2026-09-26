package io.github.xororz.localdream.ui.components

import androidx.activity.compose.BackHandler
import androidx.compose.animation.AnimatedContent
import androidx.compose.animation.animateContentSize
import androidx.compose.animation.core.FastOutSlowInEasing
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.tween
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.togetherWith
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.RowScope
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.itemsIndexed
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.pager.HorizontalPager
import androidx.compose.foundation.pager.rememberPagerState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
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
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.blur
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import coil.compose.AsyncImage
import coil.request.ImageRequest
import io.github.xororz.localdream.data.HistoryItem
import kotlin.math.absoluteValue
import kotlinx.coroutines.launch

private const val VAPORIZE_MS = 520

@Composable
fun HistoryCarouselOverlay(
    items: List<HistoryItem>,
    initialItemId: Long,
    onDismiss: () -> Unit,
    onCurrentItemChanged: (HistoryItem) -> Unit,
    deletingItemId: Long? = null,
    topEndContent: @Composable RowScope.() -> Unit = {},
) {
    if (items.isEmpty()) return

    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    val itemIdsKey = remember(items) { items.joinToString(",") { it.id.toString() } }
    val initialPage = remember(itemIdsKey, initialItemId) {
        items.indexOfFirst { it.id == initialItemId }.coerceAtLeast(0)
    }
    val pagerState = rememberPagerState(
        initialPage = initialPage,
        pageCount = { items.size },
    )
    val thumbnailState = rememberLazyListState()
    val safePage = pagerState.currentPage.coerceIn(0, items.lastIndex)
    val currentItem = items[safePage]
    var promptExpanded by remember(currentItem.id) { mutableStateOf(false) }

    BackHandler(enabled = deletingItemId == null, onBack = onDismiss)

    LaunchedEffect(initialItemId, itemIdsKey) {
        val target = items.indexOfFirst { it.id == initialItemId }
        if (target >= 0 && target != pagerState.currentPage) {
            pagerState.scrollToPage(target)
        }
    }

    LaunchedEffect(pagerState.currentPage, itemIdsKey) {
        val page = pagerState.currentPage.coerceIn(0, items.lastIndex)
        items.getOrNull(page)?.let(onCurrentItemChanged)
        if (items.isNotEmpty()) {
            thumbnailState.animateScrollToItem(page)
        }
    }

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(Color.Black),
    ) {
        AnimatedContent(
            targetState = currentItem,
            transitionSpec = { fadeIn(tween(180)) togetherWith fadeOut(tween(180)) },
            label = "historyBackdrop",
            modifier = Modifier.fillMaxSize(),
        ) { item ->
            AsyncImage(
                model = ImageRequest.Builder(context)
                    .data(item.imageFile)
                    .crossfade(false)
                    .build(),
                contentDescription = null,
                modifier = Modifier
                    .fillMaxSize()
                    .graphicsLayer {
                        scaleX = 1.2f
                        scaleY = 1.2f
                        alpha = 0.34f
                    }
                    .blur(44.dp),
                contentScale = ContentScale.Crop,
            )
        }

        Box(
            modifier = Modifier
                .fillMaxSize()
                .background(Color.Black.copy(alpha = 0.58f)),
        )

        HorizontalPager(
            state = pagerState,
            userScrollEnabled = deletingItemId == null,
            contentPadding = PaddingValues(horizontal = 18.dp),
            pageSpacing = 12.dp,
            modifier = Modifier.fillMaxSize(),
        ) { page ->
            val signedOffset =
                (pagerState.currentPage - page) + pagerState.currentPageOffsetFraction
            val distance = signedOffset.absoluteValue.coerceIn(0f, 1f)
            val item = items[page]
            val deleting = item.id == deletingItemId
            val vaporProgress by animateFloatAsState(
                targetValue = if (deleting) 1f else 0f,
                animationSpec = tween(VAPORIZE_MS, easing = FastOutSlowInEasing),
                label = "historyVaporize",
            )
            val ratio = (
                item.params.width.toFloat() /
                    item.params.height.toFloat().coerceAtLeast(1f)
                ).coerceIn(0.18f, 5.5f)
            val dustTint = MaterialTheme.colorScheme.primary

            BoxWithConstraints(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(top = 68.dp, bottom = 206.dp),
                contentAlignment = Alignment.Center,
            ) {
                val viewportRatio =
                    (maxWidth.value / maxHeight.value.coerceAtLeast(1f))
                        .coerceAtLeast(0.01f)
                val imageWidth =
                    if (ratio >= viewportRatio) maxWidth else maxHeight * ratio
                val imageHeight =
                    if (ratio >= viewportRatio) maxWidth / ratio else maxHeight

                Box(
                    modifier = Modifier
                        .size(imageWidth, imageHeight)
                        .graphicsLayer {
                            val carouselScale = 1f - distance * 0.045f
                            val deleteScale = 1f - vaporProgress * 0.12f
                            scaleX = carouselScale * deleteScale
                            scaleY = carouselScale * deleteScale
                            alpha = (1f - distance * 0.22f) * (1f - vaporProgress)
                            translationX = -signedOffset * 12f + vaporProgress * 24f
                            translationY = -vaporProgress * 28f
                            rotationZ =
                                signedOffset.coerceIn(-1f, 1f) * 0.8f +
                                    vaporProgress * 3f
                        },
                ) {
                    Surface(
                        modifier = Modifier.fillMaxSize(),
                        shape = RoundedCornerShape(22.dp),
                        color = Color.Transparent,
                        shadowElevation = 16.dp,
                    ) {
                        AsyncImage(
                            model = ImageRequest.Builder(context)
                                .data(item.imageFile)
                                .crossfade(true)
                                .build(),
                            contentDescription = "History image ${page + 1}",
                            modifier = Modifier
                                .fillMaxSize()
                                .blur((vaporProgress * 7f).dp),
                            contentScale = ContentScale.Crop,
                        )
                    }

                    if (vaporProgress > 0f) {
                        Canvas(
                            modifier = Modifier
                                .fillMaxSize()
                                .graphicsLayer { alpha = 1f - vaporProgress * 0.3f },
                        ) {
                            repeat(42) { i ->
                                val fx = ((i * 37) % 101) / 100f
                                val fy = ((i * 61 + 17) % 101) / 100f
                                val driftX = (16f + (i % 7) * 7f) * vaporProgress
                                val driftY = (8f + (i % 5) * 9f) * vaporProgress
                                val localStart = ((i % 9) * 0.035f).coerceAtMost(0.28f)
                                val local =
                                    ((vaporProgress - localStart) / (1f - localStart))
                                        .coerceIn(0f, 1f)
                                val alpha = (local * (1f - local) * 3.2f).coerceIn(0f, 0.85f)
                                val radius = 1.8f + (i % 4) * 1.35f + local * 2.2f
                                drawCircle(
                                    color = if (i % 3 == 0) {
                                        dustTint.copy(alpha = alpha)
                                    } else {
                                        Color.White.copy(alpha = alpha * 0.82f)
                                    },
                                    radius = radius,
                                    center = Offset(
                                        x = size.width * fx + driftX,
                                        y = size.height * fy - driftY,
                                    ),
                                )
                            }
                        }
                    }
                }
            }
        }

        Surface(
            modifier = Modifier
                .align(Alignment.TopStart)
                .padding(top = 48.dp, start = 14.dp),
            shape = CircleShape,
            color = Color.Black.copy(alpha = 0.5f),
        ) {
            IconButton(
                onClick = onDismiss,
                enabled = deletingItemId == null,
            ) {
                Icon(
                    imageVector = Icons.Default.Close,
                    contentDescription = "Close history preview",
                    tint = Color.White,
                )
            }
        }

        Surface(
            modifier = Modifier
                .align(Alignment.TopEnd)
                .padding(top = 48.dp, end = 12.dp),
            shape = RoundedCornerShape(50),
            color = Color.Black.copy(alpha = 0.48f),
        ) {
            Row(
                modifier = Modifier.padding(horizontal = 2.dp),
                horizontalArrangement = Arrangement.spacedBy(2.dp),
                content = topEndContent,
            )
        }

        Column(
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .fillMaxWidth()
                .padding(horizontal = 12.dp, vertical = 10.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Surface(
                    shape = RoundedCornerShape(50),
                    color = Color.Black.copy(alpha = 0.48f),
                ) {
                    Text(
                        text = "${safePage + 1} / ${items.size}",
                        modifier = Modifier.padding(horizontal = 11.dp, vertical = 6.dp),
                        style = MaterialTheme.typography.labelMedium,
                        color = Color.White,
                    )
                }
                Spacer(Modifier.size(8.dp))
                LazyRow(
                    state = thumbnailState,
                    modifier = Modifier.weight(1f),
                    contentPadding = PaddingValues(horizontal = 3.dp),
                    horizontalArrangement = Arrangement.spacedBy(7.dp),
                ) {
                    itemsIndexed(
                        items = items,
                        key = { _, item -> item.id },
                    ) { index, item ->
                        val selected = index == safePage
                        Surface(
                            modifier = Modifier
                                .size(if (selected) 58.dp else 50.dp)
                                .clickable(enabled = deletingItemId == null) {
                                    scope.launch {
                                        pagerState.animateScrollToPage(index)
                                    }
                                },
                            shape = RoundedCornerShape(if (selected) 15.dp else 12.dp),
                            color = Color.Black.copy(alpha = 0.28f),
                            border = BorderStroke(
                                width = if (selected) 2.dp else 1.dp,
                                color = if (selected) {
                                    Color.White
                                } else {
                                    Color.White.copy(alpha = 0.2f)
                                },
                            ),
                            shadowElevation = if (selected) 8.dp else 0.dp,
                        ) {
                            AsyncImage(
                                model = ImageRequest.Builder(context)
                                    .data(item.imageFile)
                                    .crossfade(true)
                                    .build(),
                                contentDescription = "Open image ${index + 1}",
                                modifier = Modifier.fillMaxSize(),
                                contentScale = ContentScale.Crop,
                            )
                        }
                    }
                }
            }

            Spacer(Modifier.height(8.dp))

            Surface(
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(22.dp),
                color = Color.Black.copy(alpha = 0.58f),
                tonalElevation = 0.dp,
                shadowElevation = 10.dp,
            ) {
                AnimatedContent(
                    targetState = currentItem,
                    transitionSpec = { fadeIn(tween(150)) togetherWith fadeOut(tween(120)) },
                    label = "historyMetadata",
                ) { item ->
                    Column(
                        modifier = Modifier
                            .padding(horizontal = 15.dp, vertical = 11.dp)
                            .animateContentSize(),
                        verticalArrangement = Arrangement.spacedBy(4.dp),
                    ) {
                        Text(
                            text = "${item.modelId} · ${item.params.width}×${item.params.height}" +
                                (item.params.generationTime?.let { " · $it" } ?: ""),
                            style = MaterialTheme.typography.bodyMedium,
                            color = Color.White,
                            maxLines = 1,
                            overflow = TextOverflow.Ellipsis,
                        )
                        if (item.params.prompt.isNotBlank()) {
                            Text(
                                text = item.params.prompt,
                                modifier = Modifier.clickable {
                                    promptExpanded = !promptExpanded
                                },
                                style = MaterialTheme.typography.bodySmall,
                                color = Color.White.copy(alpha = 0.78f),
                                maxLines = if (promptExpanded) Int.MAX_VALUE else 2,
                                overflow = if (promptExpanded) {
                                    TextOverflow.Visible
                                } else {
                                    TextOverflow.Ellipsis
                                },
                            )
                            if (item.params.prompt.length > 90 ||
                                item.params.prompt.count { it == '\n' } > 1
                            ) {
                                Text(
                                    text = if (promptExpanded) "Show less" else "Show full prompt",
                                    modifier = Modifier.clickable {
                                        promptExpanded = !promptExpanded
                                    },
                                    style = MaterialTheme.typography.labelMedium,
                                    color = MaterialTheme.colorScheme.primary,
                                )
                            }
                        }
                    }
                }
            }
        }
    }
}
