package io.github.xororz.localdream.ui.components

import androidx.activity.compose.BackHandler
import androidx.compose.animation.AnimatedContent
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.togetherWith
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
import androidx.compose.foundation.layout.width
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
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.blur
import androidx.compose.ui.draw.clip
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
import kotlin.math.roundToInt
import kotlinx.coroutines.launch

@Composable
fun HistoryCarouselOverlay(
    items: List<HistoryItem>,
    initialItemId: Long,
    onDismiss: () -> Unit,
    onCurrentItemChanged: (HistoryItem) -> Unit,
    topEndContent: @Composable RowScope.() -> Unit = {},
) {
    if (items.isEmpty()) return

    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    val initialPage = remember(items, initialItemId) {
        items.indexOfFirst { it.id == initialItemId }.coerceAtLeast(0)
    }
    val pagerState = rememberPagerState(
        initialPage = initialPage,
        pageCount = { items.size },
    )
    val currentItem = items.getOrNull(pagerState.currentPage) ?: items.first()

    BackHandler(onBack = onDismiss)

    LaunchedEffect(pagerState.currentPage, items) {
        items.getOrNull(pagerState.currentPage)?.let(onCurrentItemChanged)
    }

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(Color.Black),
    ) {
        // Edge-to-edge background follows the current image. It is deliberately
        // cropped and blurred, so any space around a portrait/landscape image
        // feels intentional instead of becoming a grey letterbox.
        AnimatedContent(
            targetState = currentItem,
            transitionSpec = { fadeIn() togetherWith fadeOut() },
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
                        scaleX = 1.18f
                        scaleY = 1.18f
                        alpha = 0.38f
                    }
                    .blur(42.dp),
                contentScale = ContentScale.Crop,
            )
        }

        Box(
            modifier = Modifier
                .fillMaxSize()
                .background(Color.Black.copy(alpha = 0.54f)),
        )

        HorizontalPager(
            state = pagerState,
            contentPadding = PaddingValues(horizontal = 22.dp),
            pageSpacing = 14.dp,
            modifier = Modifier.fillMaxSize(),
        ) { page ->
            val signedOffset =
                (pagerState.currentPage - page) + pagerState.currentPageOffsetFraction
            val distance = signedOffset.absoluteValue.coerceIn(0f, 1f)
            val item = items[page]
            val ratio = (
                item.params.width.toFloat() /
                    item.params.height.toFloat().coerceAtLeast(1f)
                ).coerceIn(0.18f, 5.5f)

            BoxWithConstraints(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(top = 76.dp, bottom = 116.dp),
                contentAlignment = Alignment.Center,
            ) {
                val viewportRatio =
                    (maxWidth.value / maxHeight.value.coerceAtLeast(1f))
                        .coerceAtLeast(0.01f)
                val imageWidth =
                    if (ratio >= viewportRatio) maxWidth else maxHeight * ratio
                val imageHeight =
                    if (ratio >= viewportRatio) maxWidth / ratio else maxHeight

                // The card itself now has the exact generated aspect ratio.
                // No giant generic rectangle means no grey bars above/below.
                Surface(
                    modifier = Modifier
                        .size(imageWidth, imageHeight)
                        .graphicsLayer {
                            val scale = 1f - distance * 0.055f
                            scaleX = scale
                            scaleY = scale
                            alpha = 1f - distance * 0.28f
                            translationX = -signedOffset * 18f
                            rotationZ = signedOffset.coerceIn(-1f, 1f) * 1.1f
                            shadowElevation = 26f * (1f - distance)
                        },
                    shape = RoundedCornerShape(24.dp),
                    color = Color.Transparent,
                    shadowElevation = 14.dp,
                ) {
                    AsyncImage(
                        model = ImageRequest.Builder(context)
                            .data(item.imageFile)
                            .crossfade(true)
                            .build(),
                        contentDescription = "History image ${page + 1}",
                        modifier = Modifier.fillMaxSize(),
                        contentScale = ContentScale.Crop,
                    )
                }
            }
        }

        Surface(
            modifier = Modifier
                .align(Alignment.TopStart)
                .padding(top = 48.dp, start = 14.dp),
            shape = CircleShape,
            color = Color.Black.copy(alpha = 0.48f),
        ) {
            IconButton(onClick = onDismiss) {
                Icon(
                    imageVector = Icons.Default.Close,
                    contentDescription = "Close history preview",
                    tint = Color.White,
                )
            }
        }

        Surface(
            modifier = Modifier
                .align(Alignment.TopCenter)
                .padding(top = 56.dp),
            shape = RoundedCornerShape(50),
            color = Color.Black.copy(alpha = 0.44f),
        ) {
            Text(
                text = "${pagerState.currentPage + 1} / ${items.size}",
                modifier = Modifier.padding(horizontal = 13.dp, vertical = 7.dp),
                style = MaterialTheme.typography.labelLarge,
                color = Color.White,
            )
        }

        Surface(
            modifier = Modifier
                .align(Alignment.TopEnd)
                .padding(top = 48.dp, end = 12.dp),
            shape = RoundedCornerShape(50),
            color = Color.Black.copy(alpha = 0.44f),
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
                .padding(horizontal = 14.dp, vertical = 12.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            InteractiveHistoryDots(
                count = items.size,
                currentPage = pagerState.currentPage,
                currentPageOffsetFraction = pagerState.currentPageOffsetFraction,
                onPageSelected = { page ->
                    scope.launch { pagerState.animateScrollToPage(page) }
                },
            )

            Spacer(Modifier.height(9.dp))

            Surface(
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(24.dp),
                color = Color.Black.copy(alpha = 0.56f),
                tonalElevation = 0.dp,
                shadowElevation = 12.dp,
            ) {
                AnimatedContent(
                    targetState = currentItem,
                    transitionSpec = { fadeIn() togetherWith fadeOut() },
                    label = "historyMetadata",
                ) { item ->
                    Column(
                        modifier = Modifier.padding(horizontal = 16.dp, vertical = 12.dp),
                        verticalArrangement = Arrangement.spacedBy(3.dp),
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
                                style = MaterialTheme.typography.bodySmall,
                                color = Color.White.copy(alpha = 0.72f),
                                maxLines = 2,
                                overflow = TextOverflow.Ellipsis,
                            )
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun InteractiveHistoryDots(
    count: Int,
    currentPage: Int,
    currentPageOffsetFraction: Float,
    onPageSelected: (Int) -> Unit,
) {
    if (count <= 1) return

    // Keep the control compact for large histories, but make it move with the
    // swipe instead of only changing after the pager settles.
    val center = (currentPage + currentPageOffsetFraction).coerceIn(0f, (count - 1).toFloat())
    val centerPage = center.roundToInt()
    val visibleCount = count.coerceAtMost(7)
    val start = (centerPage - visibleCount / 2)
        .coerceIn(0, (count - visibleCount).coerceAtLeast(0))

    Surface(
        shape = RoundedCornerShape(50),
        color = Color.Black.copy(alpha = 0.38f),
    ) {
        Row(
            modifier = Modifier.padding(horizontal = 9.dp, vertical = 7.dp),
            horizontalArrangement = Arrangement.spacedBy(5.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            repeat(visibleCount) { slot ->
                val page = start + slot
                val distance = (page - center).absoluteValue.coerceIn(0f, 1f)
                val active = 1f - distance
                val width = 7f + 15f * active
                val alpha = 0.42f + 0.58f * active

                Box(
                    modifier = Modifier
                        .width(width.dp)
                        .height(7.dp)
                        .clip(RoundedCornerShape(50))
                        .background(Color.White.copy(alpha = alpha))
                        .clickable { onPageSelected(page) },
                )
            }
        }
    }
}
