package io.github.xororz.localdream.ui.components

import androidx.activity.compose.BackHandler
import androidx.compose.animation.AnimatedContent
import androidx.compose.animation.SizeTransform
import androidx.compose.animation.core.Spring
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.spring
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.togetherWith
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.RowScope
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxHeight
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
import kotlin.math.min

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
        // Full-bleed moving backdrop: no flat grey letterbox strips around
        // portrait/landscape generations anymore.
        AnimatedContent(
            targetState = currentItem,
            transitionSpec = {
                (fadeIn(animationSpec = spring(stiffness = Spring.StiffnessMediumLow)) togetherWith
                    fadeOut(animationSpec = spring(stiffness = Spring.StiffnessMediumLow)))
                    .using(SizeTransform(clip = false))
            },
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
                        scaleX = 1.14f
                        scaleY = 1.14f
                        alpha = 0.42f
                    }
                    .blur(34.dp),
                contentScale = ContentScale.Crop,
            )
        }

        Box(
            modifier = Modifier
                .fillMaxSize()
                .background(Color.Black.copy(alpha = 0.56f)),
        )

        HorizontalPager(
            state = pagerState,
            contentPadding = PaddingValues(horizontal = 24.dp),
            pageSpacing = 14.dp,
            modifier = Modifier.fillMaxSize(),
        ) { page ->
            val signedOffset =
                (pagerState.currentPage - page) + pagerState.currentPageOffsetFraction
            val distance = signedOffset.absoluteValue.coerceIn(0f, 1f)
            val scale = 1f - (0.065f * distance)

            Box(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(top = 94.dp, bottom = 126.dp)
                    .graphicsLayer {
                        scaleX = scale
                        scaleY = scale
                        alpha = 1f - (distance * 0.34f)
                        rotationZ = signedOffset.coerceIn(-1f, 1f) * 1.8f
                        translationY = 18f * distance
                        shadowElevation = 24f * (1f - distance)
                    }
                    .clip(RoundedCornerShape(28.dp))
                    .background(Color.Black.copy(alpha = 0.18f)),
                contentAlignment = Alignment.Center,
            ) {
                AsyncImage(
                    model = ImageRequest.Builder(context)
                        .data(items[page].imageFile)
                        .crossfade(true)
                        .build(),
                    contentDescription = "History image ${page + 1}",
                    modifier = Modifier.fillMaxSize(),
                    contentScale = ContentScale.Fit,
                )
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
            color = Color.Black.copy(alpha = 0.46f),
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
            color = Color.Black.copy(alpha = 0.46f),
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
                .padding(horizontal = 16.dp, vertical = 14.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            HistoryPageDots(
                count = items.size,
                currentPage = pagerState.currentPage,
            )

            Spacer(Modifier.height(10.dp))

            Surface(
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(26.dp),
                color = Color.Black.copy(alpha = 0.58f),
                tonalElevation = 0.dp,
                shadowElevation = 14.dp,
            ) {
                AnimatedContent(
                    targetState = currentItem,
                    transitionSpec = {
                        fadeIn() togetherWith fadeOut()
                    },
                    label = "historyMetadata",
                ) { item ->
                    Column(
                        modifier = Modifier.padding(horizontal = 17.dp, vertical = 13.dp),
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
private fun HistoryPageDots(
    count: Int,
    currentPage: Int,
) {
    if (count <= 1) return

    val visibleCount = min(count, 7)
    val half = visibleCount / 2
    val start = when {
        count <= visibleCount -> 0
        currentPage < half -> 0
        currentPage > count - half - 1 -> count - visibleCount
        else -> currentPage - half
    }

    Surface(
        shape = RoundedCornerShape(50),
        color = Color.Black.copy(alpha = 0.36f),
    ) {
        Row(
            modifier = Modifier.padding(horizontal = 10.dp, vertical = 7.dp),
            horizontalArrangement = Arrangement.spacedBy(6.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            repeat(visibleCount) { slot ->
                val page = start + slot
                val selected = page == currentPage
                val width by animateFloatAsState(
                    targetValue = if (selected) 18f else 6f,
                    animationSpec = spring(
                        dampingRatio = Spring.DampingRatioMediumBouncy,
                        stiffness = Spring.StiffnessMedium,
                    ),
                    label = "historyDotWidth",
                )
                val alpha by animateFloatAsState(
                    targetValue = if (selected) 1f else 0.46f,
                    animationSpec = spring(stiffness = Spring.StiffnessMedium),
                    label = "historyDotAlpha",
                )

                Box(
                    modifier = Modifier
                        .width(width.dp)
                        .height(6.dp)
                        .clip(RoundedCornerShape(50))
                        .background(Color.White.copy(alpha = alpha)),
                )
            }
        }
    }
}
