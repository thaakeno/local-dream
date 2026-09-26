package io.github.xororz.localdream.ui.screens

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.widget.Toast
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.Favorite
import androidx.compose.material.icons.filled.FavoriteBorder
import androidx.compose.material.icons.filled.FilterList
import androidx.compose.material.icons.filled.Info
import androidx.compose.material.icons.filled.Save
import androidx.compose.material.icons.filled.Shuffle
import androidx.compose.material.icons.filled.CreateNewFolder
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LocalContentColor
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.produceState
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.input.nestedscroll.nestedScroll
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalResources
import androidx.compose.ui.res.pluralStringResource
import androidx.compose.ui.res.stringResource
import androidx.navigation.NavController
import androidx.paging.compose.collectAsLazyPagingItems
import io.github.xororz.localdream.R
import io.github.xororz.localdream.data.GenerationPreferences
import io.github.xororz.localdream.data.HistoryFilter
import io.github.xororz.localdream.data.HistoryItem
import io.github.xororz.localdream.data.HistoryManager
import io.github.xororz.localdream.navigation.HISTORY_REPRODUCE_ID_KEY
import io.github.xororz.localdream.navigation.HISTORY_VARIATION_COUNT_KEY
import io.github.xororz.localdream.navigation.HISTORY_VARIATION_ID_KEY
import io.github.xororz.localdream.navigation.Screen
import io.github.xororz.localdream.navigation.popBackStackIfResumed
import io.github.xororz.localdream.ui.components.AddToCollectionDialog
import io.github.xororz.localdream.ui.components.GenerationParamsDialog
import io.github.xororz.localdream.ui.components.HistoryCarouselOverlay
import io.github.xororz.localdream.ui.components.ManageCollectionsDialog
import io.github.xororz.localdream.ui.components.OverlayIconButton
import io.github.xororz.localdream.ui.components.ShareParamsFlow
import io.github.xororz.localdream.ui.components.ZoomableImageOverlay
import io.github.xororz.localdream.utils.saveImage
import io.github.xororz.localdream.utils.saveImageFromFile
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

// Global history browser reachable from the model list. Shares the grid,
// filter bar, and filter sheet with the model run screen but offers only
// model-independent actions (view, params, favorite, save, delete) - the
// generation-coupled flows (img2img, reproduce, upscale, ultrafix) need a
// running model and stay on the run screen.
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun HistoryScreen(navController: NavController) {
    val context = LocalContext.current
    val resources = LocalResources.current
    val scope = rememberCoroutineScope()
    val historyManager = remember { HistoryManager(context) }
    val generationPreferences = remember { GenerationPreferences(context) }
    val shareUseBase64 by remember { generationPreferences.observeShareUseBase64() }
        .collectAsState(initial = false)

    val msgImageSaved = stringResource(R.string.image_saved)
    val msgDeleted = stringResource(R.string.delete_success)
    val msgDeleteFailed = stringResource(R.string.delete_failed)
    val msgSavedCountWithFailed = stringResource(R.string.saved_count_with_failed)
    val msgDeletedCountWithFailed = stringResource(R.string.deleted_count_with_failed)

    var historyFilter by remember { mutableStateOf(HistoryFilter()) }
    val pagedItems = remember(historyFilter) { historyManager.pager(historyFilter) }
        .collectAsLazyPagingItems()
    val totalCount by remember(historyFilter) { historyManager.observeCount(historyFilter) }
        .collectAsState(initial = 0)
    val knownModelIds by remember { historyManager.observeKnownModelIds() }
        .collectAsState(initial = emptyList())
    val knownSchedulers by remember { historyManager.observeKnownSchedulers() }
        .collectAsState(initial = emptyList())
    val knownSizes by remember { historyManager.observeKnownSizes() }
        .collectAsState(initial = emptyList())
    val collections by remember { historyManager.observeCollections() }
        .collectAsState(initial = emptyList())

    var showFilterSheet by remember { mutableStateOf(false) }

    var isSelectionMode by remember { mutableStateOf(false) }
    val selectedIds = remember { mutableStateListOf<Long>() }
    var showBatchSaveDialog by remember { mutableStateOf(false) }
    var showBatchDeleteDialog by remember { mutableStateOf(false) }
    var isBatchSaving by remember { mutableStateOf(false) }
    var batchSaveCurrent by remember { mutableStateOf(0) }
    var batchSaveTotal by remember { mutableStateOf(0) }
    var batchSaveFailed by remember { mutableStateOf(0) }

    var previewItem by remember { mutableStateOf<HistoryItem?>(null) }
    var carouselItems by remember { mutableStateOf<List<HistoryItem>>(emptyList()) }
    var showParamsDialog by remember { mutableStateOf(false) }
    var showShareDialog by remember { mutableStateOf(false) }
    var showDeleteDialog by remember { mutableStateOf(false) }
    var deletingHistoryItemId by remember { mutableStateOf<Long?>(null) }
    var showAddToCollectionDialog by remember { mutableStateOf(false) }
    var collectionTargetIds by remember { mutableStateOf<List<Long>>(emptyList()) }
    var showManageCollectionsDialog by remember { mutableStateOf(false) }
    var variationTarget by remember { mutableStateOf<HistoryItem?>(null) }

    val scrollBehavior = TopAppBarDefaults.pinnedScrollBehavior()

    Scaffold(
        modifier = Modifier.nestedScroll(scrollBehavior.nestedScrollConnection),
        topBar = {
            TopAppBar(
                title = { Text(stringResource(R.string.history_tab)) },
                navigationIcon = {
                    IconButton(onClick = { navController.popBackStackIfResumed() }) {
                        Icon(
                            Icons.AutoMirrored.Filled.ArrowBack,
                            stringResource(R.string.back),
                        )
                    }
                },
                actions = {
                    val advanced = historyFilter.hasAdvancedFilters()
                    IconButton(onClick = { showFilterSheet = true }) {
                        Icon(
                            imageVector = Icons.Default.FilterList,
                            contentDescription = stringResource(R.string.history_view_filter),
                            tint = if (advanced) {
                                MaterialTheme.colorScheme.primary
                            } else {
                                LocalContentColor.current
                            },
                        )
                    }
                },
                scrollBehavior = scrollBehavior,
            )
        },
    ) { paddingValues ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(paddingValues),
        ) {
            ModelRunHistoryPage(
                historyFilter = historyFilter,
                currentModelId = null,
                pagedItems = pagedItems,
                totalCount = totalCount,
                isSelectionMode = isSelectionMode,
                selectedIds = selectedIds.toSet(),
                isBatchSaving = isBatchSaving,
                onFilterChange = { historyFilter = it },
                onShowFilterSheet = { showFilterSheet = true },
                onItemClick = { item ->
                    if (isSelectionMode) {
                        if (item.id in selectedIds) {
                            selectedIds.remove(item.id)
                            if (selectedIds.isEmpty()) {
                                isSelectionMode = false
                            }
                        } else {
                            selectedIds.add(item.id)
                        }
                    } else {
                        val loaded = pagedItems.itemSnapshotList.items
                        carouselItems = if (loaded.any { it.id == item.id }) loaded else listOf(item)
                        previewItem = item
                    }
                },
                onItemLongClick = { item ->
                    if (!isSelectionMode) {
                        isSelectionMode = true
                        selectedIds.clear()
                        selectedIds.add(item.id)
                    }
                },
                onExitSelection = {
                    isSelectionMode = false
                    selectedIds.clear()
                },
                onToggleSelectAll = {
                    // Select-all spans every match, not just loaded pages, so it
                    // queries the full id set rather than reading the grid.
                    if (totalCount > 0 && selectedIds.size >= totalCount) {
                        selectedIds.clear()
                        isSelectionMode = false
                    } else {
                        scope.launch {
                            val allIds = historyManager.queryIds(historyFilter)
                            selectedIds.clear()
                            selectedIds.addAll(allIds)
                        }
                    }
                },
                collections = collections,
                onCollectionSelected = { collectionId ->
                    historyFilter = historyFilter.copy(
                        collectionIds = collectionId?.let { setOf(it) },
                    )
                },
                onCreateCollection = {
                    collectionTargetIds = emptyList()
                    showAddToCollectionDialog = true
                },
                onManageCollections = { showManageCollectionsDialog = true },
                onBatchSave = { showBatchSaveDialog = true },
                onBatchAddToCollection = {
                    collectionTargetIds = selectedIds.toList()
                    showAddToCollectionDialog = true
                },
                onBatchRemoveFromCollection = { collectionId ->
                    val ids = selectedIds.toList()
                    scope.launch {
                        val ok = historyManager.removeFromCollection(collectionId, ids)
                        if (ok) {
                            selectedIds.clear()
                            isSelectionMode = false
                            Toast.makeText(
                                context,
                                context.getString(R.string.collection_updated),
                                Toast.LENGTH_SHORT,
                            ).show()
                        } else {
                            Toast.makeText(
                                context,
                                context.getString(R.string.collection_failed),
                                Toast.LENGTH_SHORT,
                            ).show()
                        }
                    }
                },
                onBatchDelete = { showBatchDeleteDialog = true },
            )
        }
    }

    if (showFilterSheet) {
        HistoryFilterSheet(
            initialFilter = historyFilter,
            knownModelIds = knownModelIds,
            knownSchedulers = knownSchedulers,
            knownSizes = knownSizes,
            onApply = {
                historyFilter = it
                showFilterSheet = false
            },
            onDismiss = { showFilterSheet = false },
        )
    }

    previewItem?.let { item ->
        val imagePath = item.imageFile.absolutePath
        val previewBitmap by produceState<Bitmap?>(null, imagePath) {
            value = withContext(Dispatchers.IO) {
                BitmapFactory.decodeFile(imagePath)
            }
        }
        HistoryCarouselOverlay(
            items = carouselItems.ifEmpty { listOf(item) },
            initialItemId = item.id,
            onDismiss = { previewItem = null },
            onCurrentItemChanged = { current ->
                if (previewItem?.id != current.id) previewItem = current
            },
            deletingItemId = deletingHistoryItemId,
            topEndContent = {
                OverlayIconButton(
                    icon = Icons.Default.Info,
                    contentDescription = "View parameters",
                    onClick = { showParamsDialog = true },
                )
                OverlayIconButton(
                    icon = if (item.favorite) {
                        Icons.Default.Favorite
                    } else {
                        Icons.Default.FavoriteBorder
                    },
                    contentDescription = "toggle favorite",
                    onClick = {
                        // Keep the overlay's own copy in sync; the grid
                        // refreshes through the observed flow.
                        val updated = item.copy(favorite = !item.favorite)
                        previewItem = updated
                        carouselItems = carouselItems.map {
                            if (it.id == updated.id) updated else it
                        }
                        scope.launch(Dispatchers.IO) {
                            historyManager.setFavorite(item.id, !item.favorite)
                        }
                    },
                )
                OverlayIconButton(
                    icon = Icons.Default.Shuffle,
                    contentDescription = "Generate variations",
                    onClick = { variationTarget = previewItem ?: item },
                )
                OverlayIconButton(
                    icon = Icons.Default.CreateNewFolder,
                    contentDescription = "Add to collection",
                    onClick = {
                        collectionTargetIds = listOf((previewItem ?: item).id)
                        showAddToCollectionDialog = true
                    },
                )
                OverlayIconButton(
                    icon = Icons.Default.Save,
                    contentDescription = "Save to gallery",
                    onClick = {
                        val bitmapToSave = previewBitmap
                        if (bitmapToSave != null) {
                            scope.launch {
                                saveImage(
                                    context = context,
                                    bitmap = bitmapToSave,
                                    onSuccess = {
                                        Toast.makeText(
                                            context,
                                            msgImageSaved,
                                            Toast.LENGTH_SHORT,
                                        ).show()
                                    },
                                    onError = { errorMsg ->
                                        Toast.makeText(
                                            context,
                                            errorMsg,
                                            Toast.LENGTH_SHORT,
                                        ).show()
                                    },
                                )
                            }
                        }
                    },
                )
                OverlayIconButton(
                    icon = Icons.Default.Delete,
                    contentDescription = "Delete image",
                    onClick = { showDeleteDialog = true },
                )
            },
        )

        if (showParamsDialog) {
            GenerationParamsDialog(
                title = stringResource(R.string.generation_params_title),
                params = item.params,
                modelId = item.modelId,
                displayMode = item.mode,
                showImg2imgButton = false,
                showReproduceButton = true,
                onShare = {
                    showParamsDialog = false
                    showShareDialog = true
                },
                onSendToImg2img = {},
                onReproduce = {
                    val target = previewItem ?: item
                    showParamsDialog = false
                    navController.currentBackStackEntry
                        ?.savedStateHandle
                        ?.set(HISTORY_REPRODUCE_ID_KEY, target.id)
                    previewItem = null
                    navController.navigate(Screen.ModelRun.createRoute(target.modelId))
                },
                onDismiss = { showParamsDialog = false },
            )
        }

        if (showShareDialog) {
            ShareParamsFlow(
                source = item.params,
                modelId = item.modelId,
                useBase64Initial = shareUseBase64,
                onUseBase64Changed = { value ->
                    scope.launch { generationPreferences.setShareUseBase64(value) }
                },
                onDismiss = { showShareDialog = false },
            )
        }

        if (showDeleteDialog) {
            ModelRunConfirmDialog(
                title = stringResource(R.string.delete_image),
                text = stringResource(R.string.delete_image_confirm),
                confirmText = stringResource(R.string.delete),
                destructiveConfirm = true,
                onConfirm = {
                    val deleting = previewItem ?: item
                    showDeleteDialog = false
                    deletingHistoryItemId = deleting.id
                    scope.launch {
                        delay(560)
                        val success = historyManager.deleteHistoryItem(deleting)
                        if (success) {
                            val before = carouselItems.ifEmpty { listOf(deleting) }
                            val oldIndex = before.indexOfFirst { it.id == deleting.id }
                                .coerceAtLeast(0)
                            val remaining = before.filterNot { it.id == deleting.id }
                            carouselItems = remaining
                            deletingHistoryItemId = null
                            if (remaining.isEmpty()) {
                                previewItem = null
                            } else {
                                previewItem = remaining[
                                    oldIndex.coerceAtMost(remaining.lastIndex)
                                ]
                            }
                            Toast.makeText(context, msgDeleted, Toast.LENGTH_SHORT).show()
                        } else {
                            deletingHistoryItemId = null
                            Toast.makeText(context, msgDeleteFailed, Toast.LENGTH_SHORT).show()
                        }
                    }
                },
                onDismiss = { showDeleteDialog = false },
            )
        }
    }

    if (showAddToCollectionDialog) {
        AddToCollectionDialog(
            collections = collections,
            itemCount = collectionTargetIds.size,
            onAddToExisting = { collection ->
                scope.launch {
                    val ok = historyManager.addToCollection(collection.id, collectionTargetIds)
                    showAddToCollectionDialog = false
                    if (ok) {
                        selectedIds.clear()
                        isSelectionMode = false
                        Toast.makeText(
                            context,
                            context.getString(R.string.collection_updated),
                            Toast.LENGTH_SHORT,
                        ).show()
                    } else {
                        Toast.makeText(
                            context,
                            context.getString(R.string.collection_failed),
                            Toast.LENGTH_SHORT,
                        ).show()
                    }
                }
            },
            onCreateAndAdd = { name ->
                scope.launch {
                    val collection = historyManager.createCollection(name)
                    val ok = collection != null &&
                        historyManager.addToCollection(collection.id, collectionTargetIds)
                    showAddToCollectionDialog = false
                    if (ok) {
                        selectedIds.clear()
                        isSelectionMode = false
                        Toast.makeText(
                            context,
                            context.getString(R.string.collection_created),
                            Toast.LENGTH_SHORT,
                        ).show()
                    } else {
                        Toast.makeText(
                            context,
                            context.getString(R.string.collection_failed),
                            Toast.LENGTH_SHORT,
                        ).show()
                    }
                }
            },
            onDismiss = { showAddToCollectionDialog = false },
        )
    }

    if (showManageCollectionsDialog) {
        ManageCollectionsDialog(
            collections = collections,
            onRename = { collection, name ->
                scope.launch { historyManager.renameCollection(collection.id, name) }
            },
            onDelete = { collection ->
                scope.launch {
                    historyManager.deleteCollection(collection.id)
                    if (collection.id in historyFilter.collectionIds.orEmpty()) {
                        historyFilter = historyFilter.copy(collectionIds = null)
                    }
                }
            },
            onDismiss = { showManageCollectionsDialog = false },
        )
    }

    variationTarget?.let { target ->
        VariationCountDialog(
            onGenerate = { count ->
                variationTarget = null
                navController.currentBackStackEntry?.savedStateHandle?.apply {
                    set(HISTORY_VARIATION_ID_KEY, target.id)
                    set(HISTORY_VARIATION_COUNT_KEY, count)
                }
                previewItem = null
                navController.navigate(Screen.ModelRun.createRoute(target.modelId))
            },
            onDismiss = { variationTarget = null },
        )
    }

    if (showBatchSaveDialog && selectedIds.isNotEmpty()) {
        ModelRunConfirmDialog(
            title = stringResource(R.string.batch_save),
            text = pluralStringResource(
                R.plurals.batch_save_confirm,
                selectedIds.size,
                selectedIds.size,
            ),
            confirmText = stringResource(R.string.yes),
            onConfirm = {
                val ids = selectedIds.toList()
                showBatchSaveDialog = false
                if (ids.isNotEmpty()) {
                    batchSaveTotal = ids.size
                    batchSaveCurrent = 0
                    batchSaveFailed = 0
                    isBatchSaving = true
                    scope.launch(Dispatchers.IO) {
                        // Resolve the selection to items; an id missing here
                        // (deleted meanwhile) counts as a failure.
                        val items = historyManager.getItems(ids)
                        val missing = ids.size - items.size
                        if (missing > 0) {
                            withContext(Dispatchers.Main) {
                                batchSaveFailed += missing
                                batchSaveCurrent += missing
                            }
                        }
                        items.forEach { item ->
                            var success = false
                            if (item.imageFile.exists()) {
                                saveImageFromFile(
                                    context = context,
                                    sourceFile = item.imageFile,
                                    onSuccess = { success = true },
                                    onError = { },
                                )
                            }
                            withContext(Dispatchers.Main) {
                                batchSaveCurrent += 1
                                if (!success) batchSaveFailed += 1
                            }
                        }
                        withContext(Dispatchers.Main) {
                            val failed = batchSaveFailed
                            val saved = batchSaveTotal - failed
                            val message = if (failed == 0) {
                                resources.getQuantityString(
                                    R.plurals.saved_count,
                                    saved,
                                    saved,
                                )
                            } else {
                                msgSavedCountWithFailed.format(saved, failed)
                            }
                            Toast.makeText(context, message, Toast.LENGTH_SHORT).show()
                            isBatchSaving = false
                            selectedIds.clear()
                            isSelectionMode = false
                        }
                    }
                }
            },
            onDismiss = { showBatchSaveDialog = false },
        )
    }

    if (isBatchSaving) {
        BatchSaveProgressDialog(current = batchSaveCurrent, total = batchSaveTotal)
    }

    if (showBatchDeleteDialog && selectedIds.isNotEmpty()) {
        ModelRunConfirmDialog(
            title = stringResource(R.string.batch_delete),
            text = pluralStringResource(
                R.plurals.batch_delete_confirm,
                selectedIds.size,
                selectedIds.size,
            ),
            confirmText = stringResource(R.string.delete),
            destructiveConfirm = true,
            onConfirm = {
                val ids = selectedIds.toList()
                showBatchDeleteDialog = false
                scope.launch {
                    val itemsToDelete = historyManager.getItems(ids)
                    val successCount = historyManager.deleteHistoryItems(itemsToDelete)
                    val failCount = ids.size - successCount
                    selectedIds.clear()
                    isSelectionMode = false

                    val message = if (failCount == 0) {
                        resources.getQuantityString(
                            R.plurals.deleted_count,
                            successCount,
                            successCount,
                        )
                    } else {
                        msgDeletedCountWithFailed.format(successCount, failCount)
                    }
                    Toast.makeText(context, message, Toast.LENGTH_SHORT).show()
                }
            },
            onDismiss = { showBatchDeleteDialog = false },
        )
    }
}
