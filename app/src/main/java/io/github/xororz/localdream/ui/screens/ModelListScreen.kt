package io.github.xororz.localdream.ui.screens

import android.content.ClipData
import android.content.ClipboardManager
import android.content.ContentValues
import android.content.Context
import android.content.Intent
import android.net.Uri
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import android.provider.OpenableColumns
import android.util.Log
import android.widget.Toast
import androidx.activity.compose.BackHandler
import androidx.activity.compose.PredictiveBackHandler
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.animateColorAsState
import androidx.compose.animation.core.*
import androidx.compose.animation.core.Animatable
import androidx.compose.animation.expandVertically
import androidx.compose.animation.fadeIn
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.ExperimentalFoundationApi
import androidx.compose.foundation.background
import androidx.compose.foundation.combinedClickable
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.pager.HorizontalPager
import androidx.compose.foundation.pager.rememberPagerState
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.selection.toggleable
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.automirrored.filled.Help
import androidx.compose.material.icons.automirrored.filled.InsertDriveFile
import androidx.compose.material.icons.automirrored.filled.KeyboardArrowRight
import androidx.compose.material.icons.filled.*
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.Folder
import androidx.compose.material.icons.outlined.PushPin
import androidx.compose.material3.*
import androidx.compose.material3.ButtonGroupDefaults
import androidx.compose.material3.ExposedDropdownMenuBox
import androidx.compose.material3.ExposedDropdownMenuDefaults
import androidx.compose.material3.MaterialShapes
import androidx.compose.material3.ToggleButton
import androidx.compose.material3.toShape
import androidx.compose.runtime.*
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.focus.FocusRequester
import androidx.compose.ui.focus.focusRequester
import androidx.compose.ui.focus.onFocusChanged
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.input.nestedscroll.nestedScroll
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalResources
import androidx.compose.ui.res.pluralStringResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.LinkAnnotation
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.TextLinkStyles
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextDecoration
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.text.withLink
import androidx.compose.ui.unit.dp
import androidx.core.content.edit
import androidx.documentfile.provider.DocumentFile
import androidx.navigation.NavController
import io.github.xororz.localdream.R
import io.github.xororz.localdream.data.*
import io.github.xororz.localdream.data.DarkModePreference
import io.github.xororz.localdream.navigation.Screen
import io.github.xororz.localdream.service.ModelDownloadService
import io.github.xororz.localdream.ui.components.AboutSection
import io.github.xororz.localdream.ui.components.BlockingProgressOverlay
import io.github.xororz.localdream.ui.components.CatalogFilterMode
import io.github.xororz.localdream.ui.components.CatalogSortMode
import io.github.xororz.localdream.ui.components.ModelCatalogControls
import io.github.xororz.localdream.ui.components.QwenFamilyCard
import io.github.xororz.localdream.ui.components.Yue2FamilyCard
import io.github.xororz.localdream.ui.components.filterAndSortCatalog
import io.github.xororz.localdream.ui.components.SmoothCircularWavyProgressIndicator
import io.github.xororz.localdream.ui.components.SmoothLinearWavyProgressIndicator
import io.github.xororz.localdream.ui.theme.LocalThemeController
import io.github.xororz.localdream.ui.theme.Motion
import io.github.xororz.localdream.ui.theme.ThemePreset
import io.github.xororz.localdream.ui.theme.scheme
import io.github.xororz.localdream.utils.CrashDiagnostics
import io.github.xororz.localdream.utils.DownloadDiagnostics
import io.github.xororz.localdream.utils.LogCapture
import io.github.xororz.localdream.utils.SafeClipboard
import io.github.xororz.localdream.utils.TempCleaner
import java.io.BufferedOutputStream
import java.io.File
import java.io.FileOutputStream
import java.io.OutputStream
import java.text.DecimalFormat
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.atomic.AtomicLong
import java.util.zip.ZipInputStream
import kotlin.coroutines.cancellation.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

data class LoRAFile(val uri: Uri, val weight: Float = 1.0f)

private fun getCleanFileName(uri: Uri): String {
    val fileName = uri.lastPathSegment ?: "Unknown file"
    return if (fileName.startsWith("primary:")) {
        fileName.removePrefix("primary:")
    } else {
        fileName
    }
}

@Composable
private fun DownloadLogsDialog(onDismiss: () -> Unit) {
    val context = LocalContext.current
    var refreshKey by remember { mutableIntStateOf(0) }
    val logs = remember(refreshKey) { DownloadDiagnostics.read(context) }

    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text(stringResource(R.string.download_logs)) },
        text = {
            Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
                Text(
                    text = stringResource(R.string.download_logs_hint),
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
                Surface(
                    modifier = Modifier
                        .fillMaxWidth()
                        .heightIn(min = 180.dp, max = 420.dp),
                    shape = MaterialTheme.shapes.medium,
                    color = MaterialTheme.colorScheme.surfaceContainerHighest,
                ) {
                    Text(
                        text = logs,
                        modifier = Modifier
                            .verticalScroll(rememberScrollState())
                            .padding(12.dp),
                        style = MaterialTheme.typography.bodySmall.copy(
                            fontFamily = FontFamily.Monospace,
                        ),
                    )
                }
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                ) {
                    TextButton(
                        onClick = {
                            DownloadDiagnostics.clear(context)
                            refreshKey++
                        },
                    ) {
                        Text(stringResource(R.string.clear_logs))
                    }
                    TextButton(onClick = { refreshKey++ }) {
                        Text(stringResource(R.string.refresh))
                    }
                }
            }
        },
        confirmButton = {
            Button(
                onClick = {
                    val clipboard =
                        context.getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
                    clipboard.setPrimaryClip(
                        ClipData.newPlainText("Local Dream download logs", logs),
                    )
                    Toast.makeText(
                        context,
                        context.getString(R.string.logs_copied),
                        Toast.LENGTH_SHORT,
                    ).show()
                },
            ) {
                Icon(Icons.Default.ContentCopy, contentDescription = null)
                Spacer(Modifier.width(8.dp))
                Text(stringResource(R.string.copy_logs))
            }
        },
        dismissButton = {
            TextButton(onClick = onDismiss) {
                Text(stringResource(R.string.close))
            }
        },
    )
}

@Composable
private fun DeleteConfirmDialog(
    selectedCount: Int,
    onConfirm: (keepHistory: Boolean) -> Unit,
    onDismiss: () -> Unit,
) {
    var keepHistory by remember { mutableStateOf(true) }
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text(stringResource(R.string.delete_model)) },
        text = {
            Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
                Text(stringResource(R.string.delete_confirm, selectedCount))
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    modifier = Modifier
                        .fillMaxWidth()
                        .toggleable(
                            value = keepHistory,
                            role = Role.Checkbox,
                            onValueChange = { keepHistory = it },
                        ),
                ) {
                    Checkbox(
                        checked = keepHistory,
                        onCheckedChange = null,
                    )
                    Spacer(Modifier.width(8.dp))
                    Column {
                        Text(
                            text = stringResource(R.string.delete_keep_history),
                            style = MaterialTheme.typography.bodyMedium,
                        )
                        Text(
                            text = stringResource(R.string.delete_keep_history_hint),
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                    }
                }
            }
        },
        confirmButton = {
            TextButton(
                onClick = { onConfirm(keepHistory) },
                colors = ButtonDefaults.textButtonColors(
                    contentColor = MaterialTheme.colorScheme.error,
                ),
            ) {
                Text(stringResource(R.string.delete))
            }
        },
        dismissButton = {
            TextButton(onClick = onDismiss) {
                Text(stringResource(R.string.cancel))
            }
        },
    )
}

@Composable
private fun RenameModelDialog(
    currentName: String,
    existingIds: Set<String>,
    onConfirm: (String) -> Unit,
    onDismiss: () -> Unit,
) {
    var name by remember { mutableStateOf(currentName) }
    // The id is the directory name: spaces are stripped, matching how custom
    // models are created. Validate against that derived id.
    val newId = name.replace(" ", "")
    val isBlank = newId.isEmpty()
    val isReserved = ModelRepository.isReservedModelId(newId)
    val isTaken = newId in existingIds
    val errorText = when {
        isReserved -> stringResource(R.string.custom_model_id_reserved)
        isTaken -> stringResource(R.string.rename_name_exists)
        else -> null
    }
    val canConfirm = !isBlank && !isReserved && !isTaken

    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text(stringResource(R.string.rename_model)) },
        text = {
            OutlinedTextField(
                value = name,
                onValueChange = { name = it },
                label = { Text(stringResource(R.string.rename_model_label)) },
                modifier = Modifier.fillMaxWidth(),
                singleLine = true,
                isError = errorText != null,
                supportingText = errorText?.let { { Text(it) } },
            )
        },
        confirmButton = {
            TextButton(
                onClick = { if (canConfirm) onConfirm(name) },
                enabled = canConfirm,
            ) {
                Text(stringResource(R.string.confirm))
            }
        },
        dismissButton = {
            TextButton(onClick = onDismiss) {
                Text(stringResource(R.string.cancel))
            }
        },
    )
}

@OptIn(
    ExperimentalMaterial3Api::class,
    ExperimentalMaterial3ExpressiveApi::class,
    ExperimentalFoundationApi::class,
)
@Composable
fun ModelListScreen(navController: NavController, modifier: Modifier = Modifier) {
    val context = LocalContext.current
    val resources = LocalResources.current
    val scope = rememberCoroutineScope()

    // String resources hoisted to composable scope (lint: LocalContextGetResourceValueCall).
    val msgDownloadDone = stringResource(R.string.download_done)
    val msgFileDeleted = stringResource(R.string.file_deleted)
    val msgEmbeddingDeleted = stringResource(R.string.embedding_deleted)
    val msgEmbeddingImported = stringResource(R.string.embedding_imported)
    val msgLogSaved = stringResource(R.string.log_saved)
    val msgLogSaveFailed = stringResource(R.string.log_save_failed)
    val msgModelConversionSuccess = stringResource(R.string.model_conversion_success)
    val msgModelConversionFailed = stringResource(R.string.model_conversion_failed)
    val msgNpuModelAddedSuccess = stringResource(R.string.npu_model_added_success)
    val msgNpuModelAddFailed = stringResource(R.string.npu_model_add_failed)
    val msgDeleteSuccess = stringResource(R.string.delete_success)
    val msgDeleteFailed = stringResource(R.string.delete_failed)
    val msgUnsupportNpu = stringResource(R.string.unsupport_npu)
    val msgTagImportFailed = stringResource(R.string.tag_import_failed)
    val msgCleanTempNone = stringResource(R.string.clean_temp_none)
    val msgCleanTempDone = stringResource(R.string.clean_temp_done)
    val msgRenameSuccess = stringResource(R.string.rename_success)
    val msgRenameFailed = stringResource(R.string.rename_failed)
    val msgRemoteOffline = stringResource(R.string.remote_banner_offline)

    var downloadingModel by remember { mutableStateOf<Model?>(null) }
    var currentProgress by remember { mutableStateOf<DownloadProgress?>(null) }
    var currentSpeedBytesPerSecond by remember { mutableLongStateOf(0L) }
    var currentNetworkBytesPerSecond by remember { mutableLongStateOf(0L) }
    var currentEtaSeconds by remember { mutableStateOf<Long?>(null) }
    var currentDownloadFile by remember { mutableStateOf<String?>(null) }
    var currentDownloadUsesXet by remember { mutableStateOf(false) }
    var downloadPaused by remember { mutableStateOf(false) }
    var downloadError by remember { mutableStateOf<String?>(null) }
    var showDownloadConfirm by remember { mutableStateOf<Model?>(null) }
    var showDownloadDetails by remember { mutableStateOf(false) }
    var showDeleteConfirm by remember { mutableStateOf(false) }

    var isSelectionMode by remember { mutableStateOf(false) }
    var selectedModels by remember { mutableStateOf(setOf<Model>()) }

    // Ordered pinned ids; drives the pinned-first sort within each tab. Loaded
    // once and kept in sync as the user pins/unpins/renames.
    var pinnedIds by remember { mutableStateOf(PinnedModels.get(context)) }
    var renameTarget by remember { mutableStateOf<Model?>(null) }
    var catalogQuery by remember { mutableStateOf("") }
    var catalogSort by remember { mutableStateOf(CatalogSortMode.Smart) }
    var catalogFilter by remember { mutableStateOf(CatalogFilterMode.All) }

    val snackbarHostState = remember { SnackbarHostState() }
    val scrollBehavior =
        TopAppBarDefaults.exitUntilCollapsedScrollBehavior(rememberTopAppBarState())

    var showSettingsDialog by remember { mutableStateOf(false) }
    var showDirectDownloadDialog by remember { mutableStateOf(false) }
    var showFileManagerDialog by remember { mutableStateOf(false) }
    var showBackupDialog by remember { mutableStateOf(false) }
    var showCleanTempDialog by remember { mutableStateOf(false) }
    var tempScanBytes by remember { mutableLongStateOf(0L) }
    var showEmbeddingManagerDialog by remember { mutableStateOf(false) }
    var showCustomModelDialog by remember { mutableStateOf(false) }
    var showCustomNpuModelDialog by remember { mutableStateOf(false) }
    var isConverting by remember { mutableStateOf(false) }
    var conversionProgress by remember { mutableStateOf("") }
    var extractByteProgress by remember { mutableStateOf<ExtractByteProgress?>(null) }
    var tempBaseUrl by remember { mutableStateOf("") }
    var selectedSource by remember { mutableStateOf("huggingface") }
    val generationPreferences = remember { GenerationPreferences(context) }
    var currentBaseUrl by remember { mutableStateOf("https://huggingface.co/") }
    val xetAcceleratedDownloads by generationPreferences.observeXetAcceleratedDownloads()
        .collectAsState(initial = false)

    val modelRepository = remember { ModelRepository.getInstance(context) }
    val upscalerRepository = remember { UpscalerRepository.getInstance(context) }
    val remoteRepository = remember { RemoteRepository.getInstance(context) }
    // Connected-device mode: the list shows the host device's installed
    // models; all local management actions (download/import/delete/rename)
    // are hidden because they would act on this device's storage.
    val remoteActive = remoteRepository.isActive

    var showHelpDialog by remember { mutableStateOf(false) }

    val isFirstLaunch = remember {
        context.getSharedPreferences("app_prefs", Context.MODE_PRIVATE)
            .getBoolean("is_first_launch", true)
    }

    // Collected (not keyed on the state) so a state transition cannot cancel
    // an in-flight handler: with LaunchedEffect(state) the Success snackbar
    // was dismissed early when the service reset to Idle two seconds later.
    LaunchedEffect(downloadingModel) {
        if (downloadingModel == null) showDownloadDetails = false
    }

    LaunchedEffect(Unit) {
        ModelDownloadService.downloadState.collect { state ->
            when (state) {
                is ModelDownloadService.DownloadState.Downloading -> {
                    val model = modelRepository.models.find { it.id == state.modelId } ?: Model(
                        id = state.modelId,
                        name = state.modelId,
                        description = "Direct download",
                        baseUrl = "",
                        isCustom = true,
                    )
                    if (model != null) {
                        // If the Activity attaches to a download that was already running,
                        // show the full sheet once. After the user minimizes it, ordinary
                        // progress updates never force it open again.
                        if (downloadingModel == null) {
                            showDownloadDetails = true
                        }
                        downloadingModel = model
                        currentProgress = DownloadProgress(
                            progress = state.progress,
                            downloadedBytes = state.downloadedBytes,
                            totalBytes = state.totalBytes,
                        )
                        currentSpeedBytesPerSecond = state.bytesPerSecond
                        currentNetworkBytesPerSecond = state.networkBytesPerSecond
                        currentEtaSeconds = state.etaSeconds
                        currentDownloadFile = state.currentFileName
                        currentDownloadUsesXet = state.usingXet
                        downloadPaused = false
                    }
                }

                is ModelDownloadService.DownloadState.Paused -> {
                    val model = modelRepository.models.find { it.id == state.modelId } ?: Model(
                        id = state.modelId,
                        name = state.modelId,
                        description = "Direct download",
                        baseUrl = "",
                        isCustom = true,
                    )
                    if (model != null) {
                        downloadingModel = model
                        currentProgress = DownloadProgress(
                            progress = state.progress,
                            downloadedBytes = state.downloadedBytes,
                            totalBytes = state.totalBytes,
                        )
                        currentSpeedBytesPerSecond = 0L
                        currentNetworkBytesPerSecond = 0L
                        currentEtaSeconds = null
                        currentDownloadFile = state.currentFileName
                        currentDownloadUsesXet = state.usingXet
                        downloadPaused = true
                    }
                }

                is ModelDownloadService.DownloadState.Extracting -> {
                    val model = modelRepository.models.find { it.id == state.modelId } ?: Model(
                        id = state.modelId,
                        name = state.modelId,
                        description = "Direct download",
                        baseUrl = "",
                        isCustom = true,
                    )
                    if (model != null) {
                        downloadingModel = model
                        currentProgress = null
                        currentDownloadFile = null
                        currentDownloadUsesXet = false
                        downloadPaused = false
                    }
                }

                is ModelDownloadService.DownloadState.Success -> {
                    modelRepository.refreshModelState(state.modelId)
                    downloadingModel = null
                    currentProgress = null
                    currentSpeedBytesPerSecond = 0L
                    currentNetworkBytesPerSecond = 0L
                    currentEtaSeconds = null
                    currentDownloadFile = null
                    currentDownloadUsesXet = false
                    downloadPaused = false
                    // Fire-and-forget so the snackbar's display time does not
                    // block this collector from seeing further states.
                    scope.launch { snackbarHostState.showSnackbar(msgDownloadDone) }
                }

                is ModelDownloadService.DownloadState.Error -> {
                    downloadingModel = null
                    currentProgress = null
                    currentSpeedBytesPerSecond = 0L
                    currentNetworkBytesPerSecond = 0L
                    currentEtaSeconds = null
                    currentDownloadFile = null
                    currentDownloadUsesXet = false
                    downloadPaused = false
                    downloadError = state.message
                }

                is ModelDownloadService.DownloadState.Idle -> {
                    if (downloadingModel != null) {
                        downloadingModel = null
                        currentProgress = null
                        currentSpeedBytesPerSecond = 0L
                        currentNetworkBytesPerSecond = 0L
                        currentEtaSeconds = null
                        currentDownloadFile = null
                        currentDownloadUsesXet = false
                        downloadPaused = false
                    }
                }
            }
        }
    }

    LaunchedEffect(Unit) {
        if (isFirstLaunch) {
            showHelpDialog = true
            // Written here instead of inside remember: composition may be
            // discarded, effects only run once it is committed.
            context.getSharedPreferences("app_prefs", Context.MODE_PRIVATE)
                .edit { putBoolean("is_first_launch", false) }
        }
        scope.launch {
            currentBaseUrl = generationPreferences.getBaseUrl()
            selectedSource = generationPreferences.getSelectedSource()
        }
        modelRepository.ensureLoaded()
        // Re-establish a saved device link and pull a fresh catalog; on
        // failure the offline banner offers a retry.
        remoteRepository.restore()
        if (remoteRepository.isActive) {
            remoteRepository.refresh()
        }
    }

    val listedModels = if (remoteActive) remoteRepository.models else modelRepository.models
    val cpuModels = remember(listedModels, pinnedIds) {
        PinnedModels.sort(listedModels.filter { it.runOnCpu }, pinnedIds)
    }
    val npuModels = remember(listedModels, pinnedIds) {
        PinnedModels.sort(listedModels.filter { !it.runOnCpu }, pinnedIds)
    }

    val lastViewedPage = remember {
        val preferences = context.getSharedPreferences("app_prefs", Context.MODE_PRIVATE)
        preferences.getInt("last_viewed_page", 0)
    }

    val pagerState = rememberPagerState(
        initialPage = lastViewedPage,
        pageCount = { 2 },
    )

    LaunchedEffect(pagerState.currentPage) {
        val preferences = context.getSharedPreferences("app_prefs", Context.MODE_PRIVATE)
        preferences.edit { putInt("last_viewed_page", pagerState.currentPage) }
    }

    val tabTitles = listOf(
        stringResource(R.string.cpu_models),
        stringResource(R.string.npu_models),
    )

    if (isSelectionMode) {
        BackHandler {
            isSelectionMode = false
            selectedModels = emptySet()
        }
    }
    LaunchedEffect(downloadError) {
        downloadError?.let {
            scope.launch {
                snackbarHostState.showSnackbar(
                    message = it,
                    duration = SnackbarDuration.Short,
                )
                downloadError = null
            }
        }
    }
    if (showDirectDownloadDialog) {
        DirectDownloadDialog(onDismiss = { showDirectDownloadDialog = false })
    }

    if (showHelpDialog) {
        AlertDialog(
            onDismissRequest = { },
            title = { Text(stringResource(R.string.about_app)) },
            text = {
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .verticalScroll(rememberScrollState())
                        .padding(vertical = 8.dp),
                ) {
                    val mustReadText = stringResource(R.string.must_read)
                    val linkColor = MaterialTheme.colorScheme.primary

                    val annotatedString = buildAnnotatedString {
                        var position = 0
                        for (match in Regex("""https://\S+""").findAll(mustReadText)) {
                            append(mustReadText.substring(position, match.range.first))
                            withLink(
                                LinkAnnotation.Url(
                                    url = match.value,
                                    styles = TextLinkStyles(
                                        style = SpanStyle(
                                            color = linkColor,
                                            textDecoration = TextDecoration.Underline,
                                        ),
                                    ),
                                ),
                            ) {
                                append(match.value)
                            }
                            position = match.range.last + 1
                        }
                        append(mustReadText.substring(position))
                    }

                    Text(
                        text = annotatedString,
                        style = MaterialTheme.typography.bodyMedium,
                        modifier = Modifier.padding(bottom = 12.dp),
                    )
                }
            },
            confirmButton = {
                TextButton(onClick = { showHelpDialog = false }) {
                    Text(stringResource(R.string.got_it))
                }
            },
        )
    }

    LaunchedEffect(showSettingsDialog) {
        if (showSettingsDialog) {
            tempBaseUrl = currentBaseUrl
        }
    }

    if (showBackupDialog) {
        DataBackupDialog(
            installedModelIds = modelRepository.models
                .filter { it.isDownloaded }
                .map { it.id }
                .toSet(),
            onDismiss = { showBackupDialog = false },
        )
    }

    if (showCleanTempDialog) {
        AlertDialog(
            onDismissRequest = { showCleanTempDialog = false },
            title = { Text(stringResource(R.string.clean_temp_files)) },
            text = { Text(stringResource(R.string.clean_temp_confirm, formatBytes(tempScanBytes))) },
            confirmButton = {
                TextButton(onClick = {
                    showCleanTempDialog = false
                    scope.launch {
                        val freed = TempCleaner.clean(context)
                        Toast.makeText(
                            context,
                            msgCleanTempDone.format(formatBytes(freed)),
                            Toast.LENGTH_SHORT,
                        ).show()
                    }
                }) { Text(stringResource(R.string.confirm)) }
            },
            dismissButton = {
                TextButton(onClick = { showCleanTempDialog = false }) {
                    Text(stringResource(R.string.cancel))
                }
            },
        )
    }

    if (showFileManagerDialog) {
        FileManagerDialog(
            context = context,
            onDismiss = { showFileManagerDialog = false },
            onFileDeleted = {
                scope.launch {
                    modelRepository.refreshAllModels()
                    snackbarHostState.showSnackbar(msgFileDeleted)
                }
            },
        )
    }

    if (showEmbeddingManagerDialog) {
        EmbeddingManagerDialog(
            context = context,
            onDismiss = { showEmbeddingManagerDialog = false },
            onEmbeddingDeleted = {
                scope.launch {
                    snackbarHostState.showSnackbar(msgEmbeddingDeleted)
                }
            },
            onEmbeddingImported = {
                scope.launch {
                    snackbarHostState.showSnackbar(msgEmbeddingImported)
                }
            },
        )
    }

    val capturedLogs = LogCapture.lastCapturedLogs.value
    if (capturedLogs != null) {
        val capturedLogsPreview = if (capturedLogs.length > 160_000) {
            "… preview trimmed; Save contains the full log …\n\n" +
                capturedLogs.takeLast(160_000)
        } else {
            capturedLogs
        }
        AlertDialog(
            onDismissRequest = { LogCapture.consume() },
            title = { Text(stringResource(R.string.captured_logs_title)) },
            text = {
                if (capturedLogs.isBlank()) {
                    Text(stringResource(R.string.no_logs_captured))
                } else {
                    Box(
                        modifier = Modifier
                            .fillMaxWidth()
                            .heightIn(max = 400.dp)
                            .background(
                                MaterialTheme.colorScheme.surfaceContainerHighest,
                                MaterialTheme.shapes.extraSmall,
                            )
                            .padding(8.dp),
                    ) {
                        Text(
                            text = capturedLogsPreview,
                            style = MaterialTheme.typography.bodySmall,
                            modifier = Modifier
                                .fillMaxWidth()
                                .verticalScroll(rememberScrollState()),
                        )
                    }
                }
            },
            confirmButton = {
                Row {
                    TextButton(
                        enabled = capturedLogs.isNotBlank(),
                        onClick = {
                            val result = SafeClipboard.copyText(
                                context,
                                "Local Dream captured logs",
                                capturedLogs,
                            )
                            val message = when {
                                !result.copied -> "Could not copy logs; use Save"
                                result.truncated -> "Log is huge; copied a safe tail. Save for the full log"
                                else -> context.getString(R.string.logs_copied)
                            }
                            Toast.makeText(context, message, Toast.LENGTH_SHORT).show()
                        },
                    ) {
                        Icon(Icons.Default.ContentCopy, contentDescription = null)
                        Spacer(Modifier.width(6.dp))
                        Text(stringResource(R.string.copy_logs))
                    }
                    TextButton(onClick = {
                    val timestamp = SimpleDateFormat("yyyy-MM-dd_HH-mm-ss", Locale.US)
                        .format(Date())
                    val filename = "local_dream_log_$timestamp.log"
                    scope.launch(Dispatchers.IO) {
                        val savedPath = try {
                            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                                val values = ContentValues().apply {
                                    put(MediaStore.Downloads.DISPLAY_NAME, filename)
                                    put(MediaStore.Downloads.MIME_TYPE, "text/plain")
                                    put(
                                        MediaStore.Downloads.RELATIVE_PATH,
                                        Environment.DIRECTORY_DOWNLOADS + "/LocalDream",
                                    )
                                }
                                val resolver = context.contentResolver
                                val uri = resolver.insert(
                                    MediaStore.Downloads.EXTERNAL_CONTENT_URI,
                                    values,
                                ) ?: throw java.io.IOException("MediaStore insert failed")
                                resolver.openOutputStream(uri)?.use { out ->
                                    out.write(capturedLogs.toByteArray(Charsets.UTF_8))
                                } ?: throw java.io.IOException("openOutputStream failed")
                                "Downloads/LocalDream/$filename"
                            } else {
                                val dir = File(
                                    Environment.getExternalStoragePublicDirectory(
                                        Environment.DIRECTORY_DOWNLOADS,
                                    ),
                                    "LocalDream",
                                )
                                if (!dir.exists()) dir.mkdirs()
                                val file = File(dir, filename)
                                FileOutputStream(file).use { out ->
                                    out.write(capturedLogs.toByteArray(Charsets.UTF_8))
                                }
                                file.absolutePath
                            }
                        } catch (e: Exception) {
                            Log.e("LogCapture", "save failed", e)
                            null
                        }
                        withContext(Dispatchers.Main) {
                            val msg = if (savedPath != null) {
                                msgLogSaved.format(savedPath)
                            } else {
                                msgLogSaveFailed
                            }
                            snackbarHostState.showSnackbar(msg)
                            LogCapture.consume()
                        }
                    }
                    }) {
                        Text(stringResource(R.string.save))
                    }
                }
            },
            dismissButton = {
                TextButton(onClick = { LogCapture.consume() }) {
                    Text(stringResource(R.string.close))
                }
            },
        )
    }

    if (showCustomModelDialog) {
        CustomModelDialog(
            context,
            onDismiss = { showCustomModelDialog = false },
            onModelAdded = { modelName, fileUri, clipSkip, loraFiles ->
                showCustomModelDialog = false
                scope.launch {
                    convertCustomModel(
                        context = context,
                        modelName = modelName,
                        fileUri = fileUri,
                        clipSkip = clipSkip,
                        loraFiles = loraFiles,
                        onProgress = { progress ->
                            conversionProgress = progress
                        },
                        onStart = {
                            isConverting = true
                        },
                        onSuccess = {
                            isConverting = false
                            scope.launch {
                                modelRepository.refreshAllModels()
                                snackbarHostState.showSnackbar(msgModelConversionSuccess)
                            }
                        },
                        onError = { error ->
                            isConverting = false
                            scope.launch {
                                snackbarHostState.showSnackbar(
                                    msgModelConversionFailed.format(error),
                                )
                            }
                        },
                    )
                }
            },
        )
    }

    if (showCustomNpuModelDialog) {
        CustomNpuModelDialog(
            context,
            onDismiss = { showCustomNpuModelDialog = false },
            onModelAdded = { modelName, zipUri ->
                showCustomNpuModelDialog = false
                scope.launch {
                    extractNpuModel(
                        context = context,
                        modelName = modelName,
                        zipUri = zipUri,
                        onProgress = { progress ->
                            conversionProgress = progress
                        },
                        onByteProgress = { extracted, total, fraction ->
                            extractByteProgress = ExtractByteProgress(extracted, total, fraction)
                        },
                        onStart = {
                            extractByteProgress = null
                            isConverting = true
                        },
                        onSuccess = {
                            isConverting = false
                            extractByteProgress = null
                            scope.launch {
                                modelRepository.refreshAllModels()
                                snackbarHostState.showSnackbar(msgNpuModelAddedSuccess)
                            }
                        },
                        onError = { error ->
                            isConverting = false
                            extractByteProgress = null
                            scope.launch {
                                snackbarHostState.showSnackbar(
                                    msgNpuModelAddFailed.format(error),
                                )
                            }
                        },
                    )
                }
            },
        )
    }

    if (showDeleteConfirm && selectedModels.isNotEmpty()) {
        DeleteConfirmDialog(
            selectedCount = selectedModels.size,
            onConfirm = { keepHistory ->
                showDeleteConfirm = false
                isSelectionMode = false

                scope.launch {
                    var successCount = 0
                    selectedModels.forEach { model ->
                        if (model.deleteModel(context, keepHistory)) {
                            successCount++
                        }
                    }

                    modelRepository.refreshAllModels()

                    snackbarHostState.showSnackbar(
                        if (successCount == selectedModels.size) {
                            msgDeleteSuccess
                        } else {
                            msgDeleteFailed
                        },
                    )

                    selectedModels = emptySet()
                }
            },
            onDismiss = {
                showDeleteConfirm = false
            },
        )
    }

    renameTarget?.let { model ->
        RenameModelDialog(
            currentName = model.name,
            existingIds = modelRepository.models.map { it.id }.toSet() - model.id,
            onConfirm = { newName ->
                renameTarget = null
                isSelectionMode = false
                selectedModels = emptySet()
                scope.launch {
                    val result = model.rename(context, newName)
                    if (result is RenameResult.Success) {
                        modelRepository.refreshAllModels()
                        pinnedIds = PinnedModels.get(context)
                        snackbarHostState.showSnackbar(msgRenameSuccess)
                    } else {
                        snackbarHostState.showSnackbar(msgRenameFailed)
                    }
                }
            },
            onDismiss = { renameTarget = null },
        )
    }

    showDownloadConfirm?.let { model ->
        if (downloadingModel != null) {
            AlertDialog(
                onDismissRequest = { showDownloadConfirm = null },
                title = { Text(stringResource(R.string.cannot_download)) },
                text = { Text(stringResource(R.string.cannot_download_hint)) },
                confirmButton = {
                    TextButton(onClick = { showDownloadConfirm = null }) {
                        Text(stringResource(R.string.confirm))
                    }
                },
            )
        } else {
            AlertDialog(
                onDismissRequest = { showDownloadConfirm = null },
                title = { Text(stringResource(R.string.download_model)) },
                text = {
                    Text(stringResource(R.string.download_model_hint, model.name))
                },
                confirmButton = {
                    TextButton(
                        onClick = {
                            showDownloadConfirm = null
                            downloadingModel = model
                            currentProgress = null
                            showDownloadDetails = true
                            model.startDownload(context)
                        },
                    ) {
                        Text(stringResource(R.string.confirm))
                    }
                },
                dismissButton = {
                    TextButton(onClick = { showDownloadConfirm = null }) {
                        Text(stringResource(R.string.cancel))
                    }
                },
            )
        }
    }

    Scaffold(
        topBar = {
            MediumTopAppBar(
                title = {
                    Column {
                        Row(
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.spacedBy(8.dp),
                        ) {
                            Text(
                                text = "Local Dream",
                                maxLines = 1,
                                overflow = TextOverflow.Ellipsis,
                            )
                            Icon(
                                Icons.Default.AutoAwesome,
                                contentDescription = null,
                                tint = MaterialTheme.colorScheme.primary,
                                modifier = Modifier.size(22.dp),
                            )
                        }
                        Text(
                            text = if (isSelectionMode) {
                                pluralStringResource(
                                    R.plurals.selected_items,
                                    selectedModels.size,
                                    selectedModels.size,
                                )
                            } else {
                                stringResource(R.string.available_models)
                            },
                            style = MaterialTheme.typography.bodyMedium,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                            maxLines = 1,
                            overflow = TextOverflow.Ellipsis,
                        )
                    }
                },
                navigationIcon = {
                    if (isSelectionMode) {
                        IconButton(onClick = {
                            isSelectionMode = false
                            selectedModels = emptySet()
                        }) {
                            Icon(Icons.Default.Close, stringResource(R.string.cancel))
                        }
                    }
                },
                actions = {
                    if (isSelectionMode) {
                        if (selectedModels.isNotEmpty()) {
                            // Left to right: pin, rename (single custom only), delete.
                            val allPinned = selectedModels.all { it.id in pinnedIds }
                            IconButton(onClick = {
                                val ids = selectedModels.map { it.id }
                                if (allPinned) {
                                    PinnedModels.unpin(context, ids)
                                } else {
                                    PinnedModels.pin(context, ids)
                                }
                                pinnedIds = PinnedModels.get(context)
                                isSelectionMode = false
                                selectedModels = emptySet()
                            }) {
                                Icon(
                                    imageVector = if (allPinned) {
                                        Icons.Default.PushPin
                                    } else {
                                        Icons.Outlined.PushPin
                                    },
                                    contentDescription = stringResource(
                                        if (allPinned) {
                                            R.string.unpin_from_top
                                        } else {
                                            R.string.pin_to_top
                                        },
                                    ),
                                )
                            }

                            val singleSelected = selectedModels.singleOrNull()
                            if (singleSelected != null && singleSelected.isCustom) {
                                IconButton(onClick = { renameTarget = singleSelected }) {
                                    Icon(Icons.Default.Edit, stringResource(R.string.rename))
                                }
                            }

                            IconButton(onClick = { showDeleteConfirm = true }) {
                                Icon(Icons.Default.Delete, stringResource(R.string.delete))
                            }
                        }
                    } else {
                        // A single overflow menu keeps the collapsed large top
                        // bar's title from being squeezed by multiple action
                        // icons competing for width.
                        var menuExpanded by remember { mutableStateOf(false) }
                        IconButton(onClick = { menuExpanded = true }) {
                            Icon(
                                Icons.Default.MoreVert,
                                contentDescription = stringResource(R.string.more_options),
                            )
                        }
                        DropdownMenu(
                            expanded = menuExpanded,
                            onDismissRequest = { menuExpanded = false },
                        ) {
                            DropdownMenuItem(
                                text = { Text(stringResource(R.string.help)) },
                                leadingIcon = {
                                    Icon(Icons.AutoMirrored.Filled.Help, contentDescription = null)
                                },
                                onClick = {
                                    menuExpanded = false
                                    showHelpDialog = true
                                },
                            )
                            DropdownMenuItem(
                                text = { Text(stringResource(R.string.history_tab)) },
                                leadingIcon = {
                                    Icon(Icons.Default.History, contentDescription = null)
                                },
                                onClick = {
                                    menuExpanded = false
                                    navController.navigate(Screen.History.route)
                                },
                            )
                            // In connected-device mode the standalone upscale
                            // page runs on the host's NPU, so it is offered
                            // when the host has an upscaler installed; locally
                            // it needs this device's Qualcomm NPU.
                            val showUpscaleEntry = if (remoteActive) {
                                remoteRepository.upscalerPaths.isNotEmpty()
                            } else {
                                Model.isQualcommDevice()
                            }
                            if (showUpscaleEntry) {
                                DropdownMenuItem(
                                    text = { Text(stringResource(R.string.image_upscale)) },
                                    leadingIcon = {
                                        Icon(Icons.Default.AutoFixHigh, contentDescription = null)
                                    },
                                    onClick = {
                                        menuExpanded = false
                                        navController.navigate(Screen.Upscale.route)
                                    },
                                )
                            }
                            DropdownMenuItem(
                                text = { Text(stringResource(R.string.remote_link)) },
                                leadingIcon = {
                                    Icon(Icons.Default.Devices, contentDescription = null)
                                },
                                onClick = {
                                    menuExpanded = false
                                    navController.navigate(Screen.RemoteLink.route)
                                },
                            )
                            DropdownMenuItem(
                                text = { Text(stringResource(R.string.settings)) },
                                leadingIcon = {
                                    Icon(Icons.Default.Settings, contentDescription = null)
                                },
                                onClick = {
                                    menuExpanded = false
                                    showSettingsDialog = true
                                },
                            )
                        }
                    }
                },
                scrollBehavior = scrollBehavior,
            )
        },
        snackbarHost = { SnackbarHost(snackbarHostState) },
    ) { paddingValues ->
        Column(
            modifier = modifier
                .fillMaxSize()
                .padding(paddingValues)
                .nestedScroll(scrollBehavior.nestedScrollConnection),
        ) {
            if (remoteActive) {
                RemoteModeBanner(
                    deviceName = remoteRepository.connection?.deviceName ?: "",
                    online = remoteRepository.online,
                    refreshing = remoteRepository.refreshing,
                    onRefresh = { scope.launch { remoteRepository.refresh() } },
                    onDisconnect = { remoteRepository.disconnect() },
                    modifier = Modifier.padding(horizontal = 12.dp, vertical = 8.dp),
                )
            }
            AnimatedVisibility(
                visible = downloadingModel != null && !showDownloadDetails,
            ) {
            downloadingModel?.let { model ->
                val progress = currentProgress
                DownloadMiniCard(
                    modelName = model.name,
                    progress = progress?.progress ?: 0f,
                    downloadedBytes = progress?.downloadedBytes ?: 0L,
                    totalBytes = progress?.totalBytes ?: 0L,
                    bytesPerSecond = currentSpeedBytesPerSecond,
                    networkBytesPerSecond = currentNetworkBytesPerSecond,
                    etaSeconds = currentEtaSeconds,
                    paused = downloadPaused,
                    usingXet = currentDownloadUsesXet,
                    modifier = Modifier.padding(horizontal = 12.dp, vertical = 8.dp),
                    onClick = { showDownloadDetails = true },
                )
            }
            }

            PrimaryTabRow(
                selectedTabIndex = pagerState.currentPage,
                modifier = Modifier.fillMaxWidth(),
            ) {
                tabTitles.forEachIndexed { index, title ->
                    Tab(
                        selected = pagerState.currentPage == index,
                        onClick = {
                            scope.launch {
                                pagerState.animateScrollToPage(index)
                            }
                        },
                        text = {
                            Text(
                                text = title,
                                style = MaterialTheme.typography.titleSmall,
                            )
                        },
                    )
                }
            }

            HorizontalPager(
                state = pagerState,
                modifier = Modifier.weight(1f),
            ) { page ->
                val rawModels = if (page == 0) cpuModels else npuModels
                val qwenVariants = if (page == 1 && !remoteActive) {
                    rawModels.filter { it.catalogFamily == "qwen21" }
                } else {
                    emptyList()
                }
                val yue2Variants = if (page == 1 && !remoteActive) {
                    rawModels.filter { it.catalogFamily == "yue2" }
                } else {
                    emptyList()
                }
                val standalone = if (qwenVariants.isNotEmpty() || yue2Variants.isNotEmpty()) {
                    rawModels.filter {
                        it.catalogFamily != "qwen21" && it.catalogFamily != "yue2"
                    }
                } else {
                    rawModels
                }
                val models = filterAndSortCatalog(
                    models = standalone,
                    query = catalogQuery,
                    filter = catalogFilter,
                    sort = catalogSort,
                )
                val qwenNeedle = catalogQuery.trim().lowercase(Locale.US)
                val qwenVisible = qwenVariants.isNotEmpty() &&
                    (qwenNeedle.isBlank() ||
                        "qwen image 2.1 q4 q8 fp8 gguf turbo viggle".contains(qwenNeedle)) &&
                    when (catalogFilter) {
                        CatalogFilterMode.All, CatalogFilterMode.Dit -> true
                        CatalogFilterMode.Installed -> qwenVariants.any { it.isDownloaded }
                        CatalogFilterMode.Music, CatalogFilterMode.Sdxl, CatalogFilterMode.Custom -> false
                    }
                val yue2Needle = catalogQuery.trim().lowercase(Locale.US)
                val yue2Visible = yue2Variants.isNotEmpty() &&
                    (yue2Needle.isBlank() ||
                        "yue2 yue 2 music audio song text to music q8 gguf".contains(yue2Needle)) &&
                    when (catalogFilter) {
                        CatalogFilterMode.All, CatalogFilterMode.Music -> true
                        CatalogFilterMode.Installed -> yue2Variants.any { it.isDownloaded }
                        CatalogFilterMode.Dit, CatalogFilterMode.Sdxl, CatalogFilterMode.Custom -> false
                    }

                LazyColumn(
                    modifier = Modifier.fillMaxSize(),
                    contentPadding = PaddingValues(12.dp),
                    verticalArrangement = Arrangement.spacedBy(12.dp),
                ) {
                    item(key = "catalog-controls") {
                        ModelCatalogControls(
                            query = catalogQuery,
                            onQueryChange = { catalogQuery = it },
                            filter = catalogFilter,
                            onFilterChange = { catalogFilter = it },
                            sort = catalogSort,
                            onSortChange = { catalogSort = it },
                        )
                    }

                    if (page == 0 && !remoteActive) {
                        item {
                            AddCustomModelButton(
                                onClick = { showCustomModelDialog = true },
                                modifier = Modifier.fillMaxWidth(),
                            )
                        }
                    }

                    if (page == 1 && !remoteActive) {
                        item {
                            AddModelOutlinedCard(
                                label = stringResource(R.string.add_custom_npu_model),
                                accent = true,
                                onClick = { showCustomNpuModelDialog = true },
                                modifier = Modifier.fillMaxWidth(),
                            )
                        }
                    }

                    if (yue2Visible) {
                        item(key = "yue2-family") {
                            Yue2FamilyCard(
                                variants = yue2Variants,
                                onOpen = { model ->
                                    navController.navigate(Screen.MusicRun.createRoute(model.id))
                                },
                                onDownload = { model ->
                                    showDownloadConfirm = model
                                },
                                modifier = Modifier.animateItem(
                                    fadeInSpec = tween(Motion.DurationMedium),
                                    fadeOutSpec = tween(Motion.DurationMedium),
                                    placementSpec = Motion.springExpressiveSpatial(),
                                ),
                            )
                        }
                    }

                    if (qwenVisible) {
                        item(key = "qwen-family") {
                            QwenFamilyCard(
                                variants = qwenVariants,
                                onOpen = { model ->
                                    navController.navigate(Screen.ModelRun.createRoute(model.id))
                                },
                                onDownload = { model ->
                                    showDownloadConfirm = model
                                },
                                modifier = Modifier.animateItem(
                                    fadeInSpec = tween(Motion.DurationMedium),
                                    fadeOutSpec = tween(Motion.DurationMedium),
                                    placementSpec = Motion.springExpressiveSpatial(),
                                ),
                            )
                        }
                    }

                    items(
                        items = models,
                        key = { model -> model.id },
                    ) { model ->
                        ModelCard(
                            model = model,
                            modifier = Modifier.animateItem(
                                fadeInSpec = tween(Motion.DurationMedium),
                                fadeOutSpec = tween(Motion.DurationMedium),
                                placementSpec = Motion.springExpressiveSpatial(),
                            ),
                            isSelected = selectedModels.contains(model),
                            isSelectionMode = isSelectionMode,
                            isPinned = model.id in pinnedIds,
                            onClick = {
                                if (remoteActive) {
                                    // The model runs on the host device, so this
                                    // device's SoC support is irrelevant; just
                                    // require the host to be reachable.
                                    if (remoteRepository.online) {
                                        navController.navigate(
                                            Screen.ModelRun.createRoute(model.id, remote = true),
                                        )
                                    } else {
                                        scope.launch {
                                            snackbarHostState.showSnackbar(msgRemoteOffline)
                                        }
                                    }
                                    return@ModelCard
                                }
                                if (!Model.isDeviceSupported() && !model.runOnCpu && !model.isCustom) {
                                    scope.launch {
                                        snackbarHostState.showSnackbar(msgUnsupportNpu)
                                    }
                                    return@ModelCard
                                }
                                if (isSelectionMode) {
                                    if (model.isDownloaded) {
                                        selectedModels = if (selectedModels.contains(model)) {
                                            selectedModels - model
                                        } else {
                                            selectedModels + model
                                        }

                                        if (selectedModels.isEmpty()) {
                                            isSelectionMode = false
                                        }
                                    }
                                } else {
                                    if (!model.isDownloaded) {
                                        showDownloadConfirm = model
                                    } else {
                                        navController.navigate(Screen.ModelRun.createRoute(model.id))
                                    }
                                }
                            },
                            onLongClick = {
                                // Selection mode drives local file management
                                // (pin/rename/delete); none of it applies to
                                // the host's models.
                                if (!remoteActive && model.isDownloaded && !isSelectionMode) {
                                    isSelectionMode = true
                                    selectedModels = setOf(model)
                                }
                            },
                        )
                    }

                    if (models.isEmpty() && !qwenVisible && !yue2Visible && modelRepository.isLoaded) {
                        item {
                            var visible by remember { mutableStateOf(false) }
                            LaunchedEffect(Unit) { visible = true }
                            AnimatedVisibility(
                                visible = visible,
                                enter = fadeIn(animationSpec = Motion.Fade) + expandVertically(),
                            ) {
                                Column(
                                    modifier = Modifier
                                        .fillMaxWidth()
                                        .padding(vertical = 32.dp),
                                    horizontalAlignment = Alignment.CenterHorizontally,
                                    verticalArrangement = Arrangement.spacedBy(12.dp),
                                ) {
                                    Icon(
                                        imageVector = Icons.Default.SearchOff,
                                        contentDescription = null,
                                        modifier = Modifier.size(48.dp),
                                        tint = MaterialTheme.colorScheme.onSurfaceVariant,
                                    )
                                    Text(
                                        text = if (page == 0) {
                                            stringResource(R.string.no_cpu_models)
                                        } else {
                                            stringResource(R.string.no_npu_models)
                                        },
                                        style = MaterialTheme.typography.bodyLarge,
                                        textAlign = TextAlign.Center,
                                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                                    )
                                }
                            }
                        }
                    }
                }
            }

        }
    }

    // Settings overlay with predictive back support.
    // drawerOffset: 0f = fully open, 1f = fully off-screen to the right.
    val drawerOffset = remember { Animatable(1f) }
    val drawerAnimSpec = tween<Float>(Motion.DurationLong, easing = Motion.Emphasized)
    LaunchedEffect(showSettingsDialog) {
        drawerOffset.animateTo(
            targetValue = if (showSettingsDialog) 0f else 1f,
            animationSpec = drawerAnimSpec,
        )
    }
    if (showSettingsDialog) {
        PredictiveBackHandler { progressFlow ->
            try {
                progressFlow.collect { event ->
                    drawerOffset.snapTo(event.progress)
                }
                // Committed: close the drawer; LaunchedEffect finishes the animation.
                showSettingsDialog = false
            } catch (_: CancellationException) {
                // Cancelled: slide back to open.
                drawerOffset.animateTo(0f, animationSpec = drawerAnimSpec)
            }
        }
    }
    if (drawerOffset.value < 1f) {
        val settingsScrollBehavior = TopAppBarDefaults.pinnedScrollBehavior()
        Box(
            modifier = Modifier
                .fillMaxSize()
                .graphicsLayer { translationX = size.width * drawerOffset.value }
                .background(MaterialTheme.colorScheme.surface),
        ) {
            Scaffold(
                modifier = Modifier.nestedScroll(settingsScrollBehavior.nestedScrollConnection),
                topBar = {
                    TopAppBar(
                        title = { Text(stringResource(R.string.settings)) },
                        navigationIcon = {
                            IconButton(onClick = { showSettingsDialog = false }) {
                                Icon(
                                    Icons.AutoMirrored.Filled.ArrowBack,
                                    stringResource(R.string.back),
                                )
                            }
                        },
                        scrollBehavior = settingsScrollBehavior,
                    )
                },
            ) { paddingValues ->
                LazyColumn(
                    modifier = Modifier
                        .fillMaxSize()
                        .padding(paddingValues)
                        .padding(horizontal = 16.dp),
                    verticalArrangement = Arrangement.spacedBy(24.dp),
                    contentPadding = PaddingValues(vertical = 16.dp),
                ) {
                    // Download source settings section
                    item {
                        Column {
                            Row(
                                verticalAlignment = Alignment.CenterVertically,
                                horizontalArrangement = Arrangement.spacedBy(8.dp),
                                modifier = Modifier.padding(bottom = 12.dp),
                            ) {
                                Icon(
                                    imageVector = Icons.Default.CloudDownload,
                                    contentDescription = null,
                                    tint = MaterialTheme.colorScheme.primary,
                                    modifier = Modifier.size(20.dp),
                                )
                                Text(
                                    stringResource(R.string.download_source),
                                    style = MaterialTheme.typography.titleMedium,
                                )
                            }
                            Text(
                                stringResource(R.string.download_settings_hint),
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant,
                                modifier = Modifier.padding(bottom = 12.dp),
                            )

                            var expanded by remember { mutableStateOf(false) }
                            val focusRequester = remember { FocusRequester() }

                            ExposedDropdownMenuBox(
                                expanded = expanded,
                                onExpandedChange = { expanded = !expanded },
                            ) {
                                OutlinedTextField(
                                    value = when (selectedSource) {
                                        "huggingface" -> "https://huggingface.co/"
                                        "hf-mirror" -> "https://hf-mirror.com/"
                                        else -> tempBaseUrl
                                    },
                                    onValueChange = {
                                        if (selectedSource == "custom") tempBaseUrl = it
                                    },
                                    label = { Text(stringResource(R.string.download_from)) },
                                    readOnly = selectedSource != "custom",
                                    modifier = Modifier
                                        .fillMaxWidth()
                                        .menuAnchor(
                                            ExposedDropdownMenuAnchorType.PrimaryEditable,
                                            enabled = true,
                                        )
                                        .focusRequester(focusRequester)
                                        .onFocusChanged { focusState ->
                                            if (!focusState.isFocused && selectedSource == "custom") {
                                                scope.launch {
                                                    if (tempBaseUrl.isNotEmpty() && tempBaseUrl != currentBaseUrl) {
                                                        generationPreferences.saveBaseUrl(
                                                            tempBaseUrl,
                                                        )
                                                        currentBaseUrl = tempBaseUrl
                                                        modelRepository.refreshAllModels()
                                                        upscalerRepository.refreshBaseUrl()
                                                    }
                                                }
                                            }
                                        },
                                    trailingIcon = {
                                        IconButton(onClick = {}) {
                                            ExposedDropdownMenuDefaults.TrailingIcon(
                                                expanded = expanded,
                                            )
                                        }
                                    },
                                    singleLine = true,
                                )

                                LaunchedEffect(selectedSource) {
                                    if (selectedSource == "custom") {
                                        focusRequester.requestFocus()
                                    }
                                }
                                ExposedDropdownMenu(
                                    expanded = expanded,
                                    onDismissRequest = { expanded = false },
                                ) {
                                    DropdownMenuItem(
                                        text = { Text(stringResource(R.string.source_huggingface)) },
                                        onClick = {
                                            selectedSource = "huggingface"
                                            val newUrl = "https://huggingface.co/"
                                            tempBaseUrl = newUrl
                                            expanded = false
                                            scope.launch {
                                                generationPreferences.saveSelectedSource("huggingface")
                                                generationPreferences.saveBaseUrl(newUrl)
                                                if (currentBaseUrl != newUrl) {
                                                    currentBaseUrl = newUrl
                                                    modelRepository.refreshAllModels()
                                                    upscalerRepository.refreshBaseUrl()
                                                }
                                            }
                                        },
                                    )
                                    DropdownMenuItem(
                                        text = { Text(stringResource(R.string.source_hf_mirror)) },
                                        onClick = {
                                            selectedSource = "hf-mirror"
                                            val newUrl = "https://hf-mirror.com/"
                                            tempBaseUrl = newUrl
                                            expanded = false
                                            scope.launch {
                                                generationPreferences.saveSelectedSource("hf-mirror")
                                                generationPreferences.saveBaseUrl(newUrl)
                                                if (currentBaseUrl != newUrl) {
                                                    currentBaseUrl = newUrl
                                                    modelRepository.refreshAllModels()
                                                    upscalerRepository.refreshBaseUrl()
                                                }
                                            }
                                        },
                                    )
                                    DropdownMenuItem(
                                        text = { Text(stringResource(R.string.source_custom)) },
                                        onClick = {
                                            selectedSource = "custom"
                                            tempBaseUrl = "https://"
                                            expanded = false
                                            scope.launch {
                                                generationPreferences.saveSelectedSource("custom")
                                            }
                                        },
                                    )
                                }
                            }

                            Spacer(Modifier.height(12.dp))
                            Card(
                                modifier = Modifier.fillMaxWidth(),
                                colors = CardDefaults.cardColors(
                                    containerColor = MaterialTheme.colorScheme.surfaceContainerLow,
                                ),
                            ) {
                                Row(
                                    modifier = Modifier
                                        .fillMaxWidth()
                                        .padding(horizontal = 16.dp, vertical = 12.dp),
                                    verticalAlignment = Alignment.CenterVertically,
                                    horizontalArrangement = Arrangement.spacedBy(12.dp),
                                ) {
                                    Column(modifier = Modifier.weight(1f)) {
                                        Text(
                                            text = stringResource(R.string.xet_accelerated_downloads),
                                            style = MaterialTheme.typography.bodyLarge,
                                        )
                                        Text(
                                            text = stringResource(R.string.xet_accelerated_downloads_hint),
                                            style = MaterialTheme.typography.bodySmall,
                                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                                        )
                                    }
                                    Switch(
                                        checked = xetAcceleratedDownloads,
                                        onCheckedChange = { enabled ->
                                            scope.launch {
                                                generationPreferences.setXetAcceleratedDownloads(enabled)
                                            }
                                        },
                                        enabled = selectedSource == "huggingface",
                                    )
                                }
                            }
                            Spacer(Modifier.height(12.dp))
                            var showDownloadLogs by remember { mutableStateOf(false) }
                            if (showDownloadLogs) {
                                DownloadLogsDialog(onDismiss = { showDownloadLogs = false })
                            }

                            Card(
                                modifier = Modifier.fillMaxWidth(),
                                colors = CardDefaults.cardColors(
                                    containerColor = MaterialTheme.colorScheme.surfaceContainerLow,
                                ),
                            ) {
                                Row(
                                    modifier = Modifier
                                        .fillMaxWidth()
                                        .padding(horizontal = 16.dp, vertical = 12.dp),
                                    verticalAlignment = Alignment.CenterVertically,
                                    horizontalArrangement = Arrangement.spacedBy(12.dp),
                                ) {
                                    Icon(
                                        imageVector = Icons.Default.BugReport,
                                        contentDescription = null,
                                        tint = MaterialTheme.colorScheme.primary,
                                    )
                                    Column(modifier = Modifier.weight(1f)) {
                                        Text(
                                            text = stringResource(R.string.download_logs),
                                            style = MaterialTheme.typography.bodyLarge,
                                        )
                                        Text(
                                            text = stringResource(R.string.download_logs_settings_hint),
                                            style = MaterialTheme.typography.bodySmall,
                                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                                        )
                                    }
                                    Column(
                                        verticalArrangement = Arrangement.spacedBy(6.dp),
                                    ) {
                                        OutlinedButton(onClick = { showDownloadLogs = true }) {
                                            Text(stringResource(R.string.view_logs))
                                        }
                                        OutlinedButton(
                                            onClick = {
                                                val report = CrashDiagnostics.fullReport(context)
                                                val result = SafeClipboard.copyText(
                                                    context,
                                                    "Local Dream diagnostics",
                                                    report,
                                                )
                                                val message = when {
                                                    !result.copied -> "Could not copy diagnostics; save the log instead"
                                                    result.truncated -> "Diagnostics are huge; copied a safe tail"
                                                    else -> "Full diagnostics copied"
                                                }
                                                Toast.makeText(
                                                    context,
                                                    message,
                                                    Toast.LENGTH_SHORT,
                                                ).show()
                                            },
                                        ) {
                                            Text("Copy full")
                                        }
                                    }
                                }
                            }
                        }
                    }
                    item {
                        Card(
                            modifier = Modifier.fillMaxWidth(),
                            colors = CardDefaults.cardColors(
                                containerColor = MaterialTheme.colorScheme.surfaceContainerLow,
                            ),
                        ) {
                            Row(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .padding(horizontal = 16.dp, vertical = 12.dp),
                                verticalAlignment = Alignment.CenterVertically,
                                horizontalArrangement = Arrangement.spacedBy(12.dp),
                            ) {
                                Icon(
                                    imageVector = Icons.Default.Download,
                                    contentDescription = null,
                                    tint = MaterialTheme.colorScheme.primary,
                                )
                                Column(modifier = Modifier.weight(1f)) {
                                    Text(
                                        "Download from URL",
                                        style = MaterialTheme.typography.bodyLarge,
                                    )
                                    Text(
                                        "Paste a Hugging Face file URL. Xet is used automatically when available.",
                                        style = MaterialTheme.typography.bodySmall,
                                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                                    )
                                }
                                OutlinedButton(onClick = { showDirectDownloadDialog = true }) {
                                    Text("Open")
                                }
                            }
                        }
                    }
                    // Appearance (theme) section
                    item { AppearanceSection() }
                    // Feature settings section
                    item {
                        Column {
                            Row(
                                verticalAlignment = Alignment.CenterVertically,
                                horizontalArrangement = Arrangement.spacedBy(8.dp),
                                modifier = Modifier.padding(bottom = 12.dp),
                            ) {
                                Icon(
                                    imageVector = Icons.Default.Tune,
                                    contentDescription = null,
                                    tint = MaterialTheme.colorScheme.primary,
                                    modifier = Modifier.size(20.dp),
                                )
                                Text(
                                    stringResource(R.string.feature_settings),
                                    style = MaterialTheme.typography.titleMedium,
                                )
                            }

                            Card(
                                modifier = Modifier.fillMaxWidth(),
                                colors = CardDefaults.cardColors(
                                    containerColor = MaterialTheme.colorScheme.surfaceContainerLow,
                                ),
                            ) {
                                val preferences = LocalContext.current.getSharedPreferences(
                                    "app_prefs",
                                    Context.MODE_PRIVATE,
                                )
                                var useImg2img by remember {
                                    mutableStateOf(
                                        preferences.getBoolean("use_img2img", true).also {
                                            if (!preferences.contains("use_img2img")) {
                                                preferences.edit {
                                                    putBoolean(
                                                        "use_img2img",
                                                        true,
                                                    )
                                                }
                                            }
                                        },
                                    )
                                }
                                var showProcess by remember {
                                    mutableStateOf(
                                        preferences.getBoolean("show_diffusion_process", false),
                                    )
                                }
                                var captureLogs by remember {
                                    mutableStateOf(
                                        preferences.getBoolean("enable_log_capture", false),
                                    )
                                }
                                var showGenerationStats by remember {
                                    mutableStateOf(
                                        preferences.getBoolean("show_generation_stats", false),
                                    )
                                }
                                var listenOnAllAddresses by remember {
                                    mutableStateOf(
                                        preferences.getBoolean("listen_on_all_addresses", false),
                                    )
                                }
                                var enableTagAutocomplete by remember {
                                    mutableStateOf(
                                        preferences.getBoolean("enable_tag_autocomplete", true)
                                            .also {
                                                if (!preferences.contains("enable_tag_autocomplete")) {
                                                    preferences.edit {
                                                        putBoolean("enable_tag_autocomplete", true)
                                                    }
                                                }
                                            },
                                    )
                                }
                                val tagRepository =
                                    remember { TagAutocompleteRepository.getInstance(context) }
                                val tagDictState by tagRepository.state.collectAsState()
                                var tagImportInProgress by remember { mutableStateOf(false) }
                                val mainCsvPickerLauncher = rememberLauncherForActivityResult(
                                    contract = ActivityResultContracts.GetContent(),
                                ) { uri ->
                                    if (uri == null) return@rememberLauncherForActivityResult
                                    val displayName = getFileNameFromUri(context, uri)
                                    tagImportInProgress = true
                                    scope.launch {
                                        val result = tagRepository.importMainCsv(uri, displayName)
                                        tagImportInProgress = false
                                        val message = when (result) {
                                            is ImportResult.Success ->
                                                resources.getQuantityString(
                                                    R.plurals.tag_import_success,
                                                    result.lineCount,
                                                    result.lineCount,
                                                )

                                            is ImportResult.Error -> msgTagImportFailed
                                        }
                                        Toast.makeText(context, message, Toast.LENGTH_SHORT).show()
                                    }
                                }
                                val translationCsvPickerLauncher =
                                    rememberLauncherForActivityResult(
                                        contract = ActivityResultContracts.GetContent(),
                                    ) { uri ->
                                        if (uri == null) return@rememberLauncherForActivityResult
                                        val displayName = getFileNameFromUri(context, uri)
                                        tagImportInProgress = true
                                        scope.launch {
                                            val result =
                                                tagRepository.importTranslationCsv(uri, displayName)
                                            tagImportInProgress = false
                                            val message = when (result) {
                                                is ImportResult.Success ->
                                                    resources.getQuantityString(
                                                        R.plurals.tag_import_success,
                                                        result.lineCount,
                                                        result.lineCount,
                                                    )

                                                is ImportResult.Error -> msgTagImportFailed
                                            }
                                            Toast.makeText(context, message, Toast.LENGTH_SHORT)
                                                .show()
                                        }
                                    }
                                var sdxlLowRam by remember {
                                    mutableStateOf(
                                        preferences.getBoolean("sdxl_lowram", true).also {
                                            if (!preferences.contains("sdxl_lowram")) {
                                                preferences.edit {
                                                    putBoolean("sdxl_lowram", true)
                                                }
                                            }
                                        },
                                    )
                                }
                                var animaLowRam by remember {
                                    mutableStateOf(
                                        preferences.getBoolean("anima_lowram", true).also {
                                            if (!preferences.contains("anima_lowram")) {
                                                preferences.edit {
                                                    putBoolean("anima_lowram", true)
                                                }
                                            }
                                        },
                                    )
                                }
                                var animaSeqDit by remember {
                                    mutableStateOf(
                                        preferences.getBoolean("anima_seq_dit", false),
                                    )
                                }

                                SwitchSettingRow(
                                    title = "img2img",
                                    description = stringResource(R.string.img2img_hint),
                                    checked = useImg2img,
                                    onCheckedChange = {
                                        useImg2img = it
                                        preferences.edit { putBoolean("use_img2img", it) }
                                    },
                                )
                                HorizontalDivider(
                                    modifier = Modifier.padding(horizontal = 16.dp),
                                )
                                SwitchSettingRow(
                                    title = stringResource(R.string.show_process),
                                    description = stringResource(R.string.show_process_hint),
                                    checked = showProcess,
                                    onCheckedChange = {
                                        showProcess = it
                                        preferences.edit {
                                            putBoolean("show_diffusion_process", it)
                                        }
                                    },
                                )
                                AnimatedVisibility(visible = showProcess) {
                                    Column {
                                        HorizontalDivider(
                                            modifier = Modifier.padding(horizontal = 16.dp),
                                        )
                                        Column(
                                            modifier = Modifier
                                                .fillMaxWidth()
                                                .padding(16.dp),
                                        ) {
                                            var stride by remember {
                                                mutableFloatStateOf(
                                                    preferences.getInt("show_diffusion_stride", 1)
                                                        .toFloat(),
                                                )
                                            }
                                            Text(
                                                text = stringResource(R.string.preview_stride),
                                                style = MaterialTheme.typography.titleSmall,
                                            )
                                            Text(
                                                pluralStringResource(
                                                    R.plurals.preview_stride_hint,
                                                    stride.toInt(),
                                                    stride.toInt(),
                                                ),
                                                style = MaterialTheme.typography.bodySmall,
                                                color = MaterialTheme.colorScheme.onSurfaceVariant,
                                            )
                                            Slider(
                                                value = stride,
                                                onValueChange = {
                                                    stride = it
                                                    preferences.edit {
                                                        putInt("show_diffusion_stride", it.toInt())
                                                    }
                                                },
                                                valueRange = 1f..10f,
                                                steps = 8,
                                                modifier = Modifier.fillMaxWidth(),
                                            )
                                        }
                                    }
                                }
                                HorizontalDivider(
                                    modifier = Modifier.padding(horizontal = 16.dp),
                                )
                                SwitchSettingRow(
                                    title = stringResource(R.string.capture_logs),
                                    description = stringResource(R.string.capture_logs_hint),
                                    checked = captureLogs,
                                    onCheckedChange = {
                                        captureLogs = it
                                        preferences.edit {
                                            putBoolean("enable_log_capture", it)
                                        }
                                    },
                                )
                                HorizontalDivider(
                                    modifier = Modifier.padding(horizontal = 16.dp),
                                )
                                SwitchSettingRow(
                                    title = "Generation stats",
                                    description = "Show live step, app RAM, free RAM, battery temperature/current and thermal state.",
                                    checked = showGenerationStats,
                                    onCheckedChange = {
                                        showGenerationStats = it
                                        preferences.edit {
                                            putBoolean("show_generation_stats", it)
                                        }
                                    },
                                )
                                HorizontalDivider(
                                    modifier = Modifier.padding(horizontal = 16.dp),
                                )
                                SwitchSettingRow(
                                    title = stringResource(R.string.tag_autocomplete),
                                    description = stringResource(R.string.tag_autocomplete_hint),
                                    checked = enableTagAutocomplete,
                                    onCheckedChange = {
                                        enableTagAutocomplete = it
                                        preferences.edit {
                                            putBoolean("enable_tag_autocomplete", it)
                                        }
                                    },
                                )
                                AnimatedVisibility(visible = enableTagAutocomplete) {
                                    Column {
                                        HorizontalDivider(
                                            modifier = Modifier.padding(horizontal = 16.dp),
                                        )
                                        Column(
                                            modifier = Modifier
                                                .fillMaxWidth()
                                                .padding(16.dp),
                                        ) {
                                            Text(
                                                text = stringResource(R.string.tag_main_dictionary),
                                                style = MaterialTheme.typography.titleSmall,
                                            )
                                            Text(
                                                text = if (tagDictState.mainImported) {
                                                    pluralStringResource(
                                                        R.plurals.tag_imported_status,
                                                        tagDictState.mainEntryCount,
                                                        tagDictState.mainFileName ?: "",
                                                        tagDictState.mainEntryCount,
                                                    )
                                                } else {
                                                    stringResource(R.string.tag_main_dictionary_hint)
                                                },
                                                style = MaterialTheme.typography.bodySmall,
                                                color = MaterialTheme.colorScheme.onSurfaceVariant,
                                            )
                                            Row(
                                                modifier = Modifier
                                                    .fillMaxWidth()
                                                    .padding(top = 8.dp),
                                                horizontalArrangement = Arrangement.spacedBy(8.dp),
                                            ) {
                                                Button(
                                                    onClick = { mainCsvPickerLauncher.launch("*/*") },
                                                    enabled = !tagImportInProgress,
                                                    modifier = Modifier.weight(1f),
                                                ) {
                                                    Text(
                                                        if (tagDictState.mainImported) {
                                                            stringResource(R.string.tag_reimport)
                                                        } else {
                                                            stringResource(R.string.tag_import)
                                                        },
                                                    )
                                                }
                                                if (tagDictState.mainImported) {
                                                    OutlinedButton(
                                                        onClick = { tagRepository.clearMainCsv() },
                                                        enabled = !tagImportInProgress,
                                                    ) {
                                                        Text(stringResource(R.string.tag_clear))
                                                    }
                                                }
                                            }
                                        }
                                        HorizontalDivider(
                                            modifier = Modifier.padding(horizontal = 16.dp),
                                        )
                                        Column(
                                            modifier = Modifier
                                                .fillMaxWidth()
                                                .padding(16.dp),
                                        ) {
                                            Text(
                                                text = stringResource(R.string.tag_translation_dictionary),
                                                style = MaterialTheme.typography.titleSmall,
                                            )
                                            Text(
                                                text = if (tagDictState.translationImported) {
                                                    pluralStringResource(
                                                        R.plurals.tag_imported_status,
                                                        tagDictState.translationEntryCount,
                                                        tagDictState.translationFileName ?: "",
                                                        tagDictState.translationEntryCount,
                                                    )
                                                } else {
                                                    stringResource(R.string.tag_translation_dictionary_hint)
                                                },
                                                style = MaterialTheme.typography.bodySmall,
                                                color = MaterialTheme.colorScheme.onSurfaceVariant,
                                            )
                                            Row(
                                                modifier = Modifier
                                                    .fillMaxWidth()
                                                    .padding(top = 8.dp),
                                                horizontalArrangement = Arrangement.spacedBy(8.dp),
                                            ) {
                                                Button(
                                                    onClick = {
                                                        translationCsvPickerLauncher.launch(
                                                            "*/*",
                                                        )
                                                    },
                                                    enabled = !tagImportInProgress,
                                                    modifier = Modifier.weight(1f),
                                                ) {
                                                    Text(
                                                        if (tagDictState.translationImported) {
                                                            stringResource(R.string.tag_reimport)
                                                        } else {
                                                            stringResource(R.string.tag_import)
                                                        },
                                                    )
                                                }
                                                if (tagDictState.translationImported) {
                                                    OutlinedButton(
                                                        onClick = { tagRepository.clearTranslationCsv() },
                                                        enabled = !tagImportInProgress,
                                                    ) {
                                                        Text(stringResource(R.string.tag_clear))
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                                HorizontalDivider(
                                    modifier = Modifier.padding(horizontal = 16.dp),
                                )
                                SwitchSettingRow(
                                    title = stringResource(R.string.sdxl_lowram),
                                    description = stringResource(R.string.sdxl_lowram_hint),
                                    checked = sdxlLowRam,
                                    onCheckedChange = {
                                        sdxlLowRam = it
                                        preferences.edit { putBoolean("sdxl_lowram", it) }
                                    },
                                )
                                HorizontalDivider(
                                    modifier = Modifier.padding(horizontal = 16.dp),
                                )
                                SwitchSettingRow(
                                    title = stringResource(R.string.anima_lowram),
                                    description = stringResource(R.string.anima_lowram_hint),
                                    checked = animaLowRam,
                                    onCheckedChange = {
                                        animaLowRam = it
                                        preferences.edit { putBoolean("anima_lowram", it) }
                                    },
                                )
                                AnimatedVisibility(visible = animaLowRam) {
                                    Column {
                                        HorizontalDivider(
                                            modifier = Modifier.padding(horizontal = 16.dp),
                                        )
                                        SwitchSettingRow(
                                            title = stringResource(R.string.anima_seq_dit),
                                            description = stringResource(R.string.anima_seq_dit_hint),
                                            checked = animaSeqDit,
                                            onCheckedChange = {
                                                animaSeqDit = it
                                                preferences.edit {
                                                    putBoolean("anima_seq_dit", it)
                                                }
                                            },
                                        )
                                    }
                                }
                                HorizontalDivider(
                                    modifier = Modifier.padding(horizontal = 16.dp),
                                )
                                SwitchSettingRow(
                                    title = stringResource(R.string.listen_on_all_addresses),
                                    description = stringResource(R.string.listen_on_all_addresses_hint),
                                    checked = listenOnAllAddresses,
                                    onCheckedChange = {
                                        listenOnAllAddresses = it
                                        preferences.edit {
                                            putBoolean("listen_on_all_addresses", it)
                                        }
                                    },
                                )
                            }
                        }
                    }
                    // Embedding management
                    item {
                        SettingNavCard(
                            icon = Icons.Default.Description,
                            label = stringResource(R.string.embedding_manager),
                            onClick = { showEmbeddingManagerDialog = true },
                            modifier = Modifier.fillMaxWidth(),
                        )
                    }

                    // File management
                    item {
                        SettingNavCard(
                            icon = Icons.Default.FolderOpen,
                            label = stringResource(R.string.file_manager),
                            onClick = { showFileManagerDialog = true },
                            modifier = Modifier.fillMaxWidth(),
                        )
                    }

                    // History backup and restore
                    item {
                        SettingNavCard(
                            icon = Icons.Default.SettingsBackupRestore,
                            label = stringResource(R.string.backup_restore),
                            onClick = { showBackupDialog = true },
                            modifier = Modifier.fillMaxWidth(),
                        )
                    }

                    // Clean up app scratch / orphaned temp files
                    item {
                        SettingNavCard(
                            icon = Icons.Default.CleaningServices,
                            label = stringResource(R.string.clean_temp_files),
                            onClick = {
                                scope.launch {
                                    val bytes = TempCleaner.scan(context)
                                    if (bytes <= 0L) {
                                        Toast.makeText(context, msgCleanTempNone, Toast.LENGTH_SHORT)
                                            .show()
                                    } else {
                                        tempScanBytes = bytes
                                        showCleanTempDialog = true
                                    }
                                }
                            },
                            modifier = Modifier.fillMaxWidth(),
                        )
                    }

                    // Build/source information belongs at the very bottom of Settings.
                    item { AboutSection() }
                }
            }
        }
    }

    BlockingProgressOverlay(visible = isConverting) {
        val byteProgress = extractByteProgress
        if (byteProgress != null) {
            SmoothCircularWavyProgressIndicator(
                progress = byteProgress.fraction,
                modifier = Modifier.size(72.dp),
            )
            Text(
                text = "${(byteProgress.fraction * 100).toInt()}%  ${formatBytes(byteProgress.extractedBytes)}",
                style = MaterialTheme.typography.bodyLarge.copy(
                    fontFeatureSettings = "tnum",
                ),
                color = MaterialTheme.colorScheme.onSurface,
            )
        } else {
            ContainedLoadingIndicator()
            Text(
                text = if (conversionProgress.isNotEmpty()) {
                    conversionProgress
                } else {
                    stringResource(R.string.converting)
                },
                style = MaterialTheme.typography.bodyLarge,
                color = MaterialTheme.colorScheme.onSurface,
            )
        }
    }

    if (showDownloadDetails && downloadingModel != null) {
        ModalBottomSheet(
            onDismissRequest = { showDownloadDetails = false },
            dragHandle = { BottomSheetDefaults.DragHandle() },
        ) {
            val progress = currentProgress
            DownloadDetailsSheet(
                modelName = downloadingModel?.name.orEmpty(),
                progress = progress?.progress ?: 0f,
                downloadedBytes = progress?.downloadedBytes ?: 0L,
                totalBytes = progress?.totalBytes ?: 0L,
                bytesPerSecond = currentSpeedBytesPerSecond,
                networkBytesPerSecond = currentNetworkBytesPerSecond,
                etaSeconds = currentEtaSeconds,
                currentFileName = currentDownloadFile,
                paused = downloadPaused,
                usingXet = currentDownloadUsesXet,
                onPauseResume = {
                    context.startService(
                        Intent(context, ModelDownloadService::class.java).apply {
                            action = if (downloadPaused) {
                                ModelDownloadService.ACTION_RESUME_DOWNLOAD
                            } else {
                                ModelDownloadService.ACTION_PAUSE_DOWNLOAD
                            }
                        },
                    )
                },
                onCancel = {
                    showDownloadDetails = false
                    context.startService(
                        Intent(context, ModelDownloadService::class.java).apply {
                            action = ModelDownloadService.ACTION_CANCEL_DOWNLOAD
                        },
                    )
                },
            )
        }
    }

}

@Composable
private fun DownloadMiniCard(
    modelName: String,
    progress: Float,
    downloadedBytes: Long,
    totalBytes: Long,
    bytesPerSecond: Long,
    networkBytesPerSecond: Long,
    etaSeconds: Long?,
    paused: Boolean,
    usingXet: Boolean,
    modifier: Modifier = Modifier,
    onClick: () -> Unit,
) {
    val animatedProgress by animateFloatAsState(
        targetValue = progress.coerceIn(0f, 1f),
        animationSpec = tween(durationMillis = 160, easing = LinearEasing),
        label = "downloadMiniProgress",
    )
    val confirmedProgress = progress.coerceIn(0f, 1f)
    val shownBytes = downloadedBytes.coerceAtLeast(0L)

    ElevatedCard(
        onClick = onClick,
        modifier = modifier.fillMaxWidth(),
        colors = CardDefaults.elevatedCardColors(
            containerColor = MaterialTheme.colorScheme.surfaceContainerHigh,
        ),
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 16.dp, vertical = 12.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            Box(
                modifier = Modifier.size(42.dp),
                contentAlignment = Alignment.Center,
            ) {
                CircularProgressIndicator(
                    progress = { animatedProgress },
                    modifier = Modifier.fillMaxSize(),
                    strokeWidth = 3.dp,
                )
                Icon(
                    imageVector = if (paused) Icons.Default.Pause else Icons.Default.CloudDownload,
                    contentDescription = null,
                    modifier = Modifier.size(20.dp),
                    tint = MaterialTheme.colorScheme.primary,
                )
            }

            Column(
                modifier = Modifier.weight(1f),
                verticalArrangement = Arrangement.spacedBy(4.dp),
            ) {
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                ) {
                    Text(
                        text = modelName,
                        style = MaterialTheme.typography.titleSmall,
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis,
                        modifier = Modifier.weight(1f),
                    )
                    if (usingXet) {
                        Surface(
                            shape = MaterialTheme.shapes.small,
                            color = MaterialTheme.colorScheme.secondaryContainer,
                        ) {
                            Text(
                                text = stringResource(R.string.download_mode_xet_short),
                                modifier = Modifier.padding(horizontal = 7.dp, vertical = 3.dp),
                                style = MaterialTheme.typography.labelSmall,
                                color = MaterialTheme.colorScheme.onSecondaryContainer,
                            )
                        }
                    }
                }

                Text(
                    text = when {
                        paused -> stringResource(R.string.download_paused)
                        (if (usingXet) networkBytesPerSecond else bytesPerSecond) > 0L -> buildString {
                            val displaySpeed =
                                if (usingXet && networkBytesPerSecond > 0L) {
                                    networkBytesPerSecond
                                } else {
                                    bytesPerSecond
                                }
                            append(formatDownloadSpeed(displaySpeed))
                            etaSeconds?.let {
                                append("  •  ")
                                append(formatDownloadEta(it))
                                append(" left")
                            }
                        }
                        else -> stringResource(R.string.download_starting)
                    },
                    style = MaterialTheme.typography.bodySmall.copy(fontFeatureSettings = "tnum"),
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    maxLines = 1,
                )

                LinearProgressIndicator(
                    progress = { animatedProgress },
                    modifier = Modifier.fillMaxWidth(),
                )

                if (totalBytes > 0L) {
                    Text(
                        text = "${(confirmedProgress * 100).toInt()}%  •  " +
                            "${formatBytes(shownBytes)} / ${formatBytes(totalBytes)}",
                        style = MaterialTheme.typography.labelSmall.copy(
                            fontFeatureSettings = "tnum",
                        ),
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
            }

            Icon(
                imageVector = Icons.AutoMirrored.Filled.KeyboardArrowRight,
                contentDescription = stringResource(R.string.download_open_details),
                tint = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
    }
}

@Composable
private fun DownloadDetailsSheet(
    modelName: String,
    progress: Float,
    downloadedBytes: Long,
    totalBytes: Long,
    bytesPerSecond: Long,
    networkBytesPerSecond: Long,
    etaSeconds: Long?,
    currentFileName: String?,
    paused: Boolean,
    usingXet: Boolean,
    onPauseResume: () -> Unit,
    onCancel: () -> Unit,
) {
    val animatedProgress by animateFloatAsState(
        targetValue = progress.coerceIn(0f, 1f),
        animationSpec = tween(durationMillis = 160, easing = LinearEasing),
        label = "downloadSheetProgress",
    )
    val confirmedProgress = progress.coerceIn(0f, 1f)
    val shownBytes = downloadedBytes.coerceAtLeast(0L)

    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 24.dp)
            .padding(bottom = 32.dp),
        verticalArrangement = Arrangement.spacedBy(18.dp),
    ) {
        Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Text(
                text = stringResource(R.string.downloading_model, modelName),
                style = MaterialTheme.typography.titleLarge,
            )
            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                if (usingXet) {
                    Surface(
                        shape = MaterialTheme.shapes.small,
                        color = MaterialTheme.colorScheme.secondaryContainer,
                    ) {
                        Text(
                            text = stringResource(R.string.download_mode_xet_short),
                            modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp),
                            style = MaterialTheme.typography.labelMedium,
                            color = MaterialTheme.colorScheme.onSecondaryContainer,
                        )
                    }
                }
                if (paused) {
                    Text(
                        text = stringResource(R.string.download_paused),
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
            }
        }

        SmoothLinearWavyProgressIndicator(
            progress = animatedProgress,
            modifier = Modifier.fillMaxWidth(),
        )

        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Text(
                text = "${(confirmedProgress * 100).toInt()}%",
                style = MaterialTheme.typography.titleMedium.copy(fontFeatureSettings = "tnum"),
            )
            if (totalBytes > 0L) {
                Column(horizontalAlignment = Alignment.End) {
                    Text(
                        text = "${formatBytes(shownBytes)} / ${formatBytes(totalBytes)}",
                        style = MaterialTheme.typography.bodyMedium.copy(fontFeatureSettings = "tnum"),
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                    Text(
                        text = "${formatBytes((totalBytes - shownBytes).coerceAtLeast(0L))} left",
                        style = MaterialTheme.typography.labelSmall.copy(fontFeatureSettings = "tnum"),
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
            }
        }

        if (!paused && (bytesPerSecond > 0L || networkBytesPerSecond > 0L)) {
            ElevatedCard(
                colors = CardDefaults.elevatedCardColors(
                    containerColor = MaterialTheme.colorScheme.primaryContainer,
                ),
            ) {
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(horizontal = 16.dp, vertical = 14.dp),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(12.dp),
                ) {
                    if (bytesPerSecond > 0L) {
                        Column(modifier = Modifier.weight(1f)) {
                            Text(
                                text = formatDownloadSpeed(bytesPerSecond),
                                style = MaterialTheme.typography.titleMedium,
                                color = MaterialTheme.colorScheme.onPrimaryContainer,
                            )
                            Text(
                                text = stringResource(
                                    if (usingXet) {
                                        R.string.download_effective_speed
                                    } else {
                                        R.string.download_speed
                                    },
                                ),
                                style = MaterialTheme.typography.labelMedium,
                                color = MaterialTheme.colorScheme.onPrimaryContainer,
                            )
                        }
                    }

                    if (usingXet && networkBytesPerSecond > 0L) {
                        Column(modifier = Modifier.weight(1f)) {
                            Text(
                                text = formatDownloadSpeed(networkBytesPerSecond),
                                style = MaterialTheme.typography.titleMedium,
                                color = MaterialTheme.colorScheme.onPrimaryContainer,
                            )
                            Text(
                                text = stringResource(R.string.download_network_speed),
                                style = MaterialTheme.typography.labelMedium,
                                color = MaterialTheme.colorScheme.onPrimaryContainer,
                            )
                        }
                    }

                    etaSeconds?.let {
                        Column(
                            modifier = Modifier.weight(1f),
                            horizontalAlignment = Alignment.End,
                        ) {
                            Text(
                                text = formatDownloadEta(it),
                                style = MaterialTheme.typography.titleMedium,
                                color = MaterialTheme.colorScheme.onPrimaryContainer,
                            )
                            Text(
                                text = stringResource(R.string.download_remaining),
                                style = MaterialTheme.typography.labelMedium,
                                color = MaterialTheme.colorScheme.onPrimaryContainer,
                            )
                        }
                    }
                }
            }
        }

        currentFileName?.let {
            ListItem(
                colors = ListItemDefaults.colors(containerColor = Color.Transparent),
                leadingContent = {
                    Icon(Icons.AutoMirrored.Filled.InsertDriveFile, contentDescription = null)
                },
                headlineContent = { Text(it) },
                supportingContent = {
                    Text(stringResource(R.string.download_current_file))
                },
            )
        }

        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            FilledTonalButton(
                onClick = onPauseResume,
                modifier = Modifier.weight(1f),
            ) {
                Icon(
                    imageVector = if (paused) Icons.Default.PlayArrow else Icons.Default.Pause,
                    contentDescription = null,
                )
                Spacer(Modifier.width(8.dp))
                Text(stringResource(if (paused) R.string.resume else R.string.pause))
            }
            OutlinedButton(
                onClick = onCancel,
                modifier = Modifier.weight(1f),
                colors = ButtonDefaults.outlinedButtonColors(
                    contentColor = MaterialTheme.colorScheme.error,
                ),
            ) {
                Icon(Icons.Default.Close, contentDescription = null)
                Spacer(Modifier.width(8.dp))
                Text(stringResource(R.string.cancel))
            }
        }
    }
}

internal fun formatDownloadSpeed(bytesPerSecond: Long): String =
    String.format(Locale.US, "%.1f MB/s", bytesPerSecond / (1024.0 * 1024.0))

internal fun formatDownloadEta(seconds: Long): String {
    val safe = seconds.coerceAtLeast(0L)
    val hours = safe / 3600
    val minutes = (safe % 3600) / 60
    val secs = safe % 60
    return when {
        hours > 0 -> String.format(Locale.US, "%dh %02dm", hours, minutes)
        minutes > 0 -> String.format(Locale.US, "%dm %02ds", minutes, secs)
        else -> String.format(Locale.US, "%ds", secs)
    }
}

internal fun formatBytes(bytes: Long): String = when {
    bytes < 1024 -> "$bytes B"
    bytes < 1024 * 1024 -> "${bytes / 1024} KB"
    bytes < 1024 * 1024 * 1024 -> "${bytes / (1024 * 1024)} MB"
    else -> String.format(Locale.US, "%.2f GB", bytes / (1024.0 * 1024.0 * 1024.0))
}

/**
 * Status strip shown above the tabs while connected-device mode is active:
 * which host the listed models come from, whether it is reachable, and the
 * refresh/disconnect actions.
 */
@Composable
private fun RemoteModeBanner(
    deviceName: String,
    online: Boolean,
    refreshing: Boolean,
    onRefresh: () -> Unit,
    onDisconnect: () -> Unit,
    modifier: Modifier = Modifier,
) {
    Card(
        modifier = modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = if (online) {
                MaterialTheme.colorScheme.secondaryContainer
            } else {
                MaterialTheme.colorScheme.errorContainer
            },
        ),
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 16.dp, vertical = 8.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            Icon(
                imageVector = Icons.Default.Devices,
                contentDescription = null,
                modifier = Modifier.size(20.dp),
                tint = if (online) {
                    MaterialTheme.colorScheme.onSecondaryContainer
                } else {
                    MaterialTheme.colorScheme.onErrorContainer
                },
            )
            Column(modifier = Modifier.weight(1f)) {
                Text(
                    text = stringResource(R.string.remote_banner_connected, deviceName),
                    style = MaterialTheme.typography.labelLarge,
                    color = if (online) {
                        MaterialTheme.colorScheme.onSecondaryContainer
                    } else {
                        MaterialTheme.colorScheme.onErrorContainer
                    },
                )
                if (!online) {
                    Text(
                        text = stringResource(R.string.remote_banner_offline),
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onErrorContainer,
                    )
                }
            }
            TextButton(onClick = onRefresh, enabled = !refreshing) {
                Text(stringResource(R.string.remote_refresh_models_short))
            }
            TextButton(onClick = onDisconnect) {
                Text(stringResource(R.string.remote_disconnect))
            }
        }
    }
}

@Composable
fun TabPageIndicator(pageCount: Int, currentPage: Int, modifier: Modifier = Modifier) {
    Row(
        horizontalArrangement = Arrangement.spacedBy(8.dp),
        modifier = modifier,
    ) {
        repeat(pageCount) { index ->
            val isSelected = currentPage == index
            val sizeFloat by animateFloatAsState(
                targetValue = if (isSelected) 10f else 8f,
                animationSpec = Motion.springExpressiveSpatial(),
                label = "IndicatorSize",
            )
            val color by animateColorAsState(
                targetValue = if (isSelected) {
                    MaterialTheme.colorScheme.primary
                } else {
                    MaterialTheme.colorScheme.outlineVariant
                },
                animationSpec = tween(Motion.DurationMedium),
                label = "IndicatorColor",
            )
            Box(
                modifier = Modifier
                    .size(sizeFloat.dp)
                    .background(
                        color = color,
                        shape = CircleShape,
                    ),
            )
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class, ExperimentalFoundationApi::class)
@Composable
fun ModelCard(
    model: Model,
    isSelected: Boolean,
    isSelectionMode: Boolean,
    onClick: () -> Unit,
    onLongClick: () -> Unit,
    modifier: Modifier = Modifier,
    isPinned: Boolean = false,
) {
    val isDisabledInSelection = !model.isDownloaded && isSelectionMode

    val elevation by animateFloatAsState(
        targetValue = if (isSelected) 4f else 1f,
        animationSpec = Motion.springExpressiveSpatial(),
        label = "CardElevationAnimation",
    )

    val targetContainer = when {
        isSelected -> MaterialTheme.colorScheme.secondaryContainer
        isDisabledInSelection -> MaterialTheme.colorScheme.surfaceContainerLow
        else -> MaterialTheme.colorScheme.surfaceContainer
    }
    val backgroundColor by animateColorAsState(
        targetValue = targetContainer,
        animationSpec = tween(Motion.DurationMedium, easing = Motion.Standard),
        label = "CardBackgroundColorAnimation",
    )

    val primaryContent = when {
        isSelected -> MaterialTheme.colorScheme.onSecondaryContainer
        isDisabledInSelection -> MaterialTheme.colorScheme.onSurface.copy(alpha = 0.38f)
        else -> MaterialTheme.colorScheme.onSurface
    }
    val secondaryContent = when {
        isSelected -> MaterialTheme.colorScheme.onSecondaryContainer
        isDisabledInSelection -> MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.38f)
        else -> MaterialTheme.colorScheme.onSurfaceVariant
    }

    ElevatedCard(
        modifier = modifier.fillMaxWidth(),
        colors = CardDefaults.elevatedCardColors(
            containerColor = backgroundColor,
            contentColor = primaryContent,
        ),
        elevation = CardDefaults.elevatedCardElevation(
            defaultElevation = elevation.dp,
        ),
        shape = MaterialTheme.shapes.large,
    ) {
        // The clickable lives inside the card so its press/ripple indication is
        // clipped to the card's rounded shape. On the outer modifier the ripple
        // would render as a rectangle and show square corners on long-press.
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .combinedClickable(
                    onClick = {
                        if (!isSelectionMode || model.isDownloaded) onClick()
                    },
                    onLongClick = {
                        if (model.isDownloaded && !isSelectionMode) onLongClick()
                    },
                ),
        ) {
            Badge(
                modifier = Modifier
                    .align(Alignment.TopEnd)
                    .padding(8.dp),
                containerColor = if (model.runOnCpu) {
                    MaterialTheme.colorScheme.tertiaryContainer
                } else {
                    MaterialTheme.colorScheme.primaryContainer
                },
                contentColor = if (model.runOnCpu) {
                    MaterialTheme.colorScheme.onTertiaryContainer
                } else {
                    MaterialTheme.colorScheme.onPrimaryContainer
                },
            ) {
                Text(
                    text = if (model.runOnCpu) "CPU" else "NPU",
                    style = MaterialTheme.typography.labelSmall,
                )
            }

            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(16.dp),
            ) {
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(6.dp),
                ) {
                    if (isPinned) {
                        Icon(
                            imageVector = Icons.Default.PushPin,
                            contentDescription = stringResource(R.string.pin_to_top),
                            tint = primaryContent,
                            modifier = Modifier.size(18.dp),
                        )
                    }
                    Text(
                        text = model.name,
                        style = MaterialTheme.typography.titleLarge,
                        fontWeight = FontWeight.Normal,
                        color = primaryContent,
                        modifier = Modifier.weight(1f, fill = false),
                    )
                }
                Spacer(modifier = Modifier.height(4.dp))
                Text(
                    text = model.description,
                    style = MaterialTheme.typography.bodyMedium,
                    maxLines = 2,
                    overflow = TextOverflow.Ellipsis,
                    color = secondaryContent,
                )
                Spacer(modifier = Modifier.height(8.dp))

                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Row(
                        horizontalArrangement = Arrangement.spacedBy(12.dp),
                        verticalAlignment = Alignment.CenterVertically,
                        modifier = Modifier.weight(1f),
                    ) {
                        InfoChip(
                            icon = Icons.Default.SdStorage,
                            label = model.approximateSize,
                            color = secondaryContent,
                        )
                        InfoChip(
                            icon = Icons.Default.AspectRatio,
                            label = if (model.runOnCpu) {
                                "128~512"
                            } else {
                                "${model.generationSize}×${model.generationSize}"
                            },
                            color = secondaryContent,
                        )
                        if (model.id == "qwen_image_2_1_viggle_turbo") {
                            InfoChip(
                                icon = Icons.Default.Speed,
                                label = "${model.defaults.steps.toInt()} steps",
                                color = secondaryContent,
                            )
                        }
                    }

                    when {
                        model.isDownloaded -> {
                            Row(
                                verticalAlignment = Alignment.CenterVertically,
                                horizontalArrangement = Arrangement.spacedBy(8.dp),
                            ) {
                                val statusColor =
                                    if (isSelected) {
                                        MaterialTheme.colorScheme.onSecondaryContainer
                                    } else {
                                        MaterialTheme.colorScheme.primary
                                    }
                                Row(
                                    verticalAlignment = Alignment.CenterVertically,
                                    horizontalArrangement = Arrangement.spacedBy(4.dp),
                                ) {
                                    Icon(
                                        imageVector = Icons.Default.CheckCircle,
                                        contentDescription = "downloaded",
                                        tint = statusColor,
                                        modifier = Modifier.size(16.dp),
                                    )
                                    Text(
                                        text = stringResource(R.string.downloaded),
                                        style = MaterialTheme.typography.labelMedium,
                                        color = statusColor,
                                    )
                                }
                            }
                        }

                        else -> {
                            InfoChip(
                                icon = Icons.Default.CloudDownload,
                                label = stringResource(R.string.download),
                                color = secondaryContent,
                            )
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun InfoChip(icon: ImageVector, label: String, color: Color) {
    Row(
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(4.dp),
    ) {
        Icon(
            imageVector = icon,
            contentDescription = null,
            tint = color,
            modifier = Modifier.size(16.dp),
        )
        Text(
            text = label,
            style = MaterialTheme.typography.labelMedium,
            color = color,
        )
    }
}

private fun formatFileSize(size: Long): String {
    val df = DecimalFormat("#.##")
    return when {
        size < 1024 -> "${size}B"
        size < 1024 * 1024 -> "${df.format(size / 1024.0)}KB"
        size < 1024 * 1024 * 1024 -> "${df.format(size / (1024.0 * 1024.0))}MB"
        else -> "${df.format(size / (1024.0 * 1024.0 * 1024.0))}GB"
    }
}

@OptIn(ExperimentalMaterial3ExpressiveApi::class)
@Composable
private fun FileManagerDialog(context: Context, onDismiss: () -> Unit, onFileDeleted: () -> Unit) {
    var modelFolders by remember { mutableStateOf<List<Pair<String, Int>>>(emptyList()) }
    var selectedFolder by remember { mutableStateOf<String?>(null) }
    var folderFiles by remember { mutableStateOf<List<File>>(emptyList()) }
    var showDeleteConfirm by remember { mutableStateOf<File?>(null) }
    var showClearCacheConfirm by remember { mutableStateOf(false) }
    var isLoading by remember { mutableStateOf(true) }
    // Tracked separately so the "Clear Cache" button can light up without
    // exposing the cache directory as a fake "file" entry in the list.
    var cacheDir by remember { mutableStateOf<File?>(null) }
    var cacheSize by remember { mutableLongStateOf(0L) }
    val scope = rememberCoroutineScope()

    val msgCacheCleared = stringResource(R.string.cache_cleared)

    suspend fun loadFolders() {
        val folders = withContext(Dispatchers.IO) {
            val modelsDir = Model.getModelsDir(context)
            val result = mutableListOf<Pair<String, Int>>()

            if (modelsDir.exists() && modelsDir.isDirectory) {
                modelsDir.listFiles()?.forEach { modelDir ->
                    if (modelDir.isDirectory) {
                        val fileCount = modelDir.listFiles()?.size ?: 0
                        if (fileCount > 0) {
                            result.add(Pair(modelDir.name, fileCount))
                        }
                    }
                }
            }
            result
        }
        modelFolders = folders
        isLoading = false
    }

    suspend fun loadFilesForFolder(folderName: String) {
        val (cd, size, files) = withContext(Dispatchers.IO) {
            val folderDir = File(Model.getModelsDir(context), folderName)
            val all = folderDir.listFiles()?.toList() ?: emptyList()
            val cache = all.firstOrNull { it.isDirectory && it.name == "cache" }
            val cacheBytes =
                cache?.walkTopDown()?.filter { it.isFile }?.sumOf { it.length() } ?: 0L
            Triple(cache, cacheBytes, all.filter { it.isFile })
        }
        cacheDir = cd
        cacheSize = size
        folderFiles = files
    }

    LaunchedEffect(Unit) {
        loadFolders()
    }

    if (showDeleteConfirm != null) {
        AlertDialog(
            onDismissRequest = { showDeleteConfirm = null },
            title = { Text(stringResource(R.string.delete_file)) },
            text = { Text(stringResource(R.string.delete_file_confirm, showDeleteConfirm!!.name)) },
            confirmButton = {
                TextButton(
                    onClick = {
                        val fileToDelete = showDeleteConfirm!!
                        showDeleteConfirm = null
                        scope.launch {
                            val deleted = withContext(Dispatchers.IO) { fileToDelete.delete() }
                            if (deleted) {
                                onFileDeleted()
                                selectedFolder?.let { loadFilesForFolder(it) }
                                loadFolders()
                            }
                        }
                    },
                    colors = ButtonDefaults.textButtonColors(
                        contentColor = MaterialTheme.colorScheme.error,
                    ),
                ) {
                    Text(stringResource(R.string.delete))
                }
            },
            dismissButton = {
                TextButton(onClick = { showDeleteConfirm = null }) {
                    Text(stringResource(R.string.cancel))
                }
            },
        )
    }

    if (showClearCacheConfirm) {
        AlertDialog(
            onDismissRequest = { showClearCacheConfirm = false },
            title = { Text(stringResource(R.string.clear_cache)) },
            text = { Text(stringResource(R.string.clear_cache_confirm)) },
            confirmButton = {
                TextButton(
                    onClick = {
                        val dirToClear = cacheDir
                        showClearCacheConfirm = false
                        scope.launch {
                            withContext(Dispatchers.IO) { dirToClear?.deleteRecursively() }
                            Toast.makeText(
                                context,
                                msgCacheCleared,
                                Toast.LENGTH_SHORT,
                            ).show()
                            onFileDeleted()
                            selectedFolder?.let { loadFilesForFolder(it) }
                        }
                    },
                    colors = ButtonDefaults.textButtonColors(
                        contentColor = MaterialTheme.colorScheme.error,
                    ),
                ) {
                    Text(stringResource(R.string.clear_cache))
                }
            },
            dismissButton = {
                TextButton(onClick = { showClearCacheConfirm = false }) {
                    Text(stringResource(R.string.cancel))
                }
            },
        )
    }

    AlertDialog(
        onDismissRequest = onDismiss,
        title = {
            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                if (selectedFolder != null) {
                    IconButton(
                        onClick = { selectedFolder = null },
                        modifier = Modifier.size(24.dp),
                    ) {
                        Icon(
                            imageVector = Icons.AutoMirrored.Filled.ArrowBack,
                            contentDescription = stringResource(R.string.back_to_folders),
                            modifier = Modifier.size(20.dp),
                        )
                    }
                }
                Text(
                    text = selectedFolder?.let {
                        stringResource(R.string.model_folder, it)
                    } ?: stringResource(R.string.file_manager),
                )
            }
        },
        text = {
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .height(400.dp),
            ) {
                if (isLoading) {
                    Box(
                        modifier = Modifier.fillMaxSize(),
                        contentAlignment = Alignment.Center,
                    ) {
                        ContainedLoadingIndicator()
                        Text(
                            stringResource(R.string.loading_files),
                            modifier = Modifier.padding(top = 48.dp),
                            style = MaterialTheme.typography.bodyMedium,
                        )
                    }
                } else if (selectedFolder == null) {
                    if (modelFolders.isEmpty()) {
                        Box(
                            modifier = Modifier.fillMaxSize(),
                            contentAlignment = Alignment.Center,
                        ) {
                            Column(
                                horizontalAlignment = Alignment.CenterHorizontally,
                            ) {
                                Icon(
                                    imageVector = Icons.Default.FolderOpen,
                                    contentDescription = null,
                                    modifier = Modifier.size(48.dp),
                                    tint = MaterialTheme.colorScheme.onSurfaceVariant,
                                )
                                Spacer(modifier = Modifier.height(16.dp))
                                Text(
                                    stringResource(R.string.no_model_files),
                                    style = MaterialTheme.typography.bodyMedium,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                                )
                            }
                        }
                    } else {
                        LazyColumn(
                            modifier = Modifier.fillMaxSize(),
                            verticalArrangement = Arrangement.spacedBy(8.dp),
                        ) {
                            items(modelFolders) { (folderName, fileCount) ->
                                Card(
                                    onClick = {
                                        selectedFolder = folderName
                                        folderFiles = emptyList()
                                        cacheDir = null
                                        cacheSize = 0L
                                        scope.launch { loadFilesForFolder(folderName) }
                                    },
                                    modifier = Modifier.fillMaxWidth(),
                                    colors = CardDefaults.cardColors(
                                        containerColor = MaterialTheme.colorScheme.surfaceContainerLow,
                                    ),
                                ) {
                                    Row(
                                        modifier = Modifier
                                            .fillMaxWidth()
                                            .padding(16.dp),
                                        horizontalArrangement = Arrangement.SpaceBetween,
                                        verticalAlignment = Alignment.CenterVertically,
                                    ) {
                                        Row(
                                            verticalAlignment = Alignment.CenterVertically,
                                            horizontalArrangement = Arrangement.spacedBy(12.dp),
                                        ) {
                                            Icon(
                                                imageVector = Icons.Default.Folder,
                                                contentDescription = null,
                                                tint = MaterialTheme.colorScheme.primary,
                                            )
                                            Column {
                                                Text(
                                                    text = folderName,
                                                    style = MaterialTheme.typography.titleSmall,
                                                )
                                                Text(
                                                    text = pluralStringResource(
                                                        R.plurals.file_count,
                                                        fileCount,
                                                        fileCount,
                                                    ),
                                                    style = MaterialTheme.typography.bodySmall,
                                                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                                                )
                                            }
                                        }
                                        Icon(
                                            imageVector = Icons.Default.ChevronRight,
                                            contentDescription = null,
                                            tint = MaterialTheme.colorScheme.onSurfaceVariant,
                                        )
                                    }
                                }
                            }
                        }
                    }
                } else {
                    if (folderFiles.isEmpty()) {
                        Box(
                            modifier = Modifier.fillMaxSize(),
                            contentAlignment = Alignment.Center,
                        ) {
                            Text(
                                stringResource(R.string.no_model_files),
                                style = MaterialTheme.typography.bodyMedium,
                                color = MaterialTheme.colorScheme.onSurfaceVariant,
                            )
                        }
                    } else {
                        LazyColumn(
                            modifier = Modifier.fillMaxSize(),
                            verticalArrangement = Arrangement.spacedBy(8.dp),
                        ) {
                            items(folderFiles) { file ->
                                Card(
                                    modifier = Modifier.fillMaxWidth(),
                                    colors = CardDefaults.cardColors(
                                        containerColor = MaterialTheme.colorScheme.surfaceContainerLow,
                                    ),
                                ) {
                                    Row(
                                        modifier = Modifier
                                            .fillMaxWidth()
                                            .padding(12.dp),
                                        horizontalArrangement = Arrangement.SpaceBetween,
                                        verticalAlignment = Alignment.CenterVertically,
                                    ) {
                                        Row(
                                            verticalAlignment = Alignment.CenterVertically,
                                            horizontalArrangement = Arrangement.spacedBy(12.dp),
                                            modifier = Modifier.weight(1f),
                                        ) {
                                            Icon(
                                                imageVector = Icons.AutoMirrored.Filled.InsertDriveFile,
                                                contentDescription = null,
                                                tint = MaterialTheme.colorScheme.secondary,
                                            )
                                            Column {
                                                Text(
                                                    text = file.name,
                                                    style = MaterialTheme.typography.titleSmall,
                                                )
                                                Text(
                                                    text = formatFileSize(file.length()),
                                                    style = MaterialTheme.typography.bodySmall,
                                                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                                                )
                                            }
                                        }

                                        IconButton(
                                            onClick = { showDeleteConfirm = file },
                                            colors = IconButtonDefaults.iconButtonColors(
                                                contentColor = MaterialTheme.colorScheme.error,
                                            ),
                                        ) {
                                            Icon(
                                                imageVector = Icons.Default.Delete,
                                                contentDescription = stringResource(R.string.delete_file),
                                            )
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        confirmButton = {
            TextButton(onClick = onDismiss) {
                Text(stringResource(R.string.close))
            }
        },
        dismissButton = {
            if (selectedFolder != null && cacheDir != null) {
                TextButton(
                    onClick = { showClearCacheConfirm = true },
                    colors = ButtonDefaults.textButtonColors(
                        contentColor = MaterialTheme.colorScheme.error,
                    ),
                ) {
                    Icon(
                        imageVector = Icons.Default.CleaningServices,
                        contentDescription = null,
                        modifier = Modifier.size(18.dp),
                    )
                    Spacer(modifier = Modifier.size(6.dp))
                    Text(
                        stringResource(
                            R.string.clear_cache_with_size,
                            formatFileSize(cacheSize),
                        ),
                    )
                }
            }
        },
    )
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun AddCustomModelButton(onClick: () -> Unit, modifier: Modifier = Modifier) {
    AddModelOutlinedCard(
        label = stringResource(R.string.add_custom_model),
        onClick = onClick,
        modifier = modifier,
        accent = false,
    )
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun AddModelOutlinedCard(label: String, onClick: () -> Unit, accent: Boolean, modifier: Modifier = Modifier) {
    val accentColor = if (accent) {
        MaterialTheme.colorScheme.primary
    } else {
        MaterialTheme.colorScheme.onSurfaceVariant
    }
    OutlinedCard(
        onClick = onClick,
        modifier = modifier,
        shape = MaterialTheme.shapes.large,
        colors = CardDefaults.outlinedCardColors(
            containerColor = MaterialTheme.colorScheme.surface,
            contentColor = accentColor,
        ),
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            horizontalArrangement = Arrangement.Center,
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Icon(
                imageVector = Icons.Default.Add,
                contentDescription = null,
                modifier = Modifier.size(20.dp),
            )
            Spacer(modifier = Modifier.width(8.dp))
            Text(
                text = label,
                style = MaterialTheme.typography.titleSmall,
            )
        }
    }
}

@Composable
fun CustomNpuModelDialog(context: Context, onDismiss: () -> Unit, onModelAdded: (String, Uri) -> Unit) {
    var modelName by remember { mutableStateOf("") }
    var selectedZipUri by remember { mutableStateOf<Uri?>(null) }
    val isIdReserved = modelName.isNotBlank() &&
        ModelRepository.isReservedModelId(modelName.replace(" ", ""))

    val zipPickerLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent(),
    ) { uri ->
        uri?.let {
            selectedZipUri = it
            if (modelName.isBlank()) {
                getFileNameFromUri(context, it)?.let { fileName ->
                    modelName = fileName.substringBeforeLast(".").substringBefore("_qnn")
                }
            }
        }
    }

    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text(stringResource(R.string.add_custom_npu_model)) },
        text = {
            Column(
                modifier = Modifier.fillMaxWidth(),
                verticalArrangement = Arrangement.spacedBy(16.dp),
            ) {
                val hintText = stringResource(R.string.custom_npu_model_hint)
                val linkColor = MaterialTheme.colorScheme.primary
                val hintAnnotated = buildAnnotatedString {
                    val urlRegex = Regex("https://[^\\s,，、。]+")
                    var lastIndex = 0
                    for (match in urlRegex.findAll(hintText)) {
                        append(hintText.substring(lastIndex, match.range.first))
                        withLink(
                            LinkAnnotation.Url(
                                url = match.value,
                                styles = TextLinkStyles(
                                    style = SpanStyle(
                                        color = linkColor,
                                        textDecoration = TextDecoration.Underline,
                                    ),
                                ),
                            ),
                        ) {
                            append(match.value)
                        }
                        lastIndex = match.range.last + 1
                    }
                    append(hintText.substring(lastIndex))
                }
                Text(
                    text = hintAnnotated,
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )

                OutlinedTextField(
                    value = modelName,
                    onValueChange = { modelName = it },
                    label = { Text(stringResource(R.string.custom_model_name)) },
                    modifier = Modifier.fillMaxWidth(),
                    singleLine = true,
                    placeholder = { Text(stringResource(R.string.custom_model_name_hint)) },
                    isError = isIdReserved,
                    supportingText = if (isIdReserved) {
                        { Text(stringResource(R.string.custom_model_id_reserved)) }
                    } else {
                        null
                    },
                )

                FilledTonalButton(
                    onClick = {
                        zipPickerLauncher.launch("application/zip")
                    },
                    modifier = Modifier.fillMaxWidth(),
                ) {
                    Icon(
                        imageVector = Icons.Default.Folder,
                        contentDescription = null,
                        modifier = Modifier.size(18.dp),
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    Text(
                        text = selectedZipUri?.let { stringResource(R.string.zip_file_selected) }
                            ?: stringResource(R.string.select_zip_file),
                    )
                }

                selectedZipUri?.let { uri ->
                    Text(
                        text = "Selected: ${getCleanFileName(uri)}",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
            }
        },
        confirmButton = {
            TextButton(
                onClick = {
                    if (modelName.isNotBlank() && selectedZipUri != null && !isIdReserved) {
                        onModelAdded(modelName, selectedZipUri!!)
                    }
                },
                enabled = modelName.isNotBlank() && selectedZipUri != null && !isIdReserved,
            ) {
                Text(stringResource(R.string.add_model))
            }
        },
        dismissButton = {
            TextButton(onClick = onDismiss) {
                Text(stringResource(R.string.cancel))
            }
        },
    )
}

@OptIn(ExperimentalMaterial3ExpressiveApi::class)
@Composable
fun CustomModelDialog(
    context: Context,
    onDismiss: () -> Unit,
    onModelAdded: (String, Uri, Int, List<LoRAFile>) -> Unit,
) {
    var modelName by remember { mutableStateOf("") }
    var selectedFileUri by remember { mutableStateOf<Uri?>(null) }
    var clipSkip by remember { mutableIntStateOf(1) }
    var selectedLoraFiles by remember { mutableStateOf<List<LoRAFile>>(emptyList()) }
    val isIdReserved = modelName.isNotBlank() &&
        ModelRepository.isReservedModelId(modelName.replace(" ", ""))

    val filePickerLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent(),
    ) { uri ->
        uri?.let {
            selectedFileUri = it
            if (modelName.isBlank()) {
                getFileNameFromUri(context, it)?.let { fileName ->
                    modelName = fileName.substringBeforeLast(".")
                }
            }
        }
    }

    val loraPickerLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent(),
    ) { uri ->
        uri?.let {
            selectedLoraFiles = selectedLoraFiles + LoRAFile(it)
        }
    }

    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text(stringResource(R.string.add_custom_model)) },
        text = {
            Column(
                modifier = Modifier.fillMaxWidth(),
                verticalArrangement = Arrangement.spacedBy(16.dp),
            ) {
                Text(
                    text = stringResource(R.string.custom_model_hint),
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )

                OutlinedTextField(
                    value = modelName,
                    onValueChange = { modelName = it },
                    label = { Text(stringResource(R.string.custom_model_name)) },
                    modifier = Modifier.fillMaxWidth(),
                    singleLine = true,
                    placeholder = { Text(stringResource(R.string.custom_model_name_hint)) },
                    isError = isIdReserved,
                    supportingText = if (isIdReserved) {
                        { Text(stringResource(R.string.custom_model_id_reserved)) }
                    } else {
                        null
                    },
                )

                Column(
                    modifier = Modifier.fillMaxWidth(),
                ) {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(
                            ButtonGroupDefaults.ConnectedSpaceBetween,
                        ),
                    ) {
                        ToggleButton(
                            checked = clipSkip == 1,
                            onCheckedChange = { checked -> if (checked) clipSkip = 1 },
                            shapes = ButtonGroupDefaults.connectedLeadingButtonShapes(),
                            modifier = Modifier.weight(1f),
                        ) {
                            Text("Clip Skip 1")
                        }
                        ToggleButton(
                            checked = clipSkip == 2,
                            onCheckedChange = { checked -> if (checked) clipSkip = 2 },
                            shapes = ButtonGroupDefaults.connectedTrailingButtonShapes(),
                            modifier = Modifier.weight(1f),
                        ) {
                            Text("Clip Skip 2")
                        }
                    }
                    Text(
                        text = stringResource(R.string.clip_skip_hint),
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        modifier = Modifier.padding(top = 4.dp),
                    )
                }

                FilledTonalButton(
                    onClick = {
                        filePickerLauncher.launch("application/octet-stream")
                    },
                    modifier = Modifier.fillMaxWidth(),
                ) {
                    Icon(
                        imageVector = Icons.Default.Folder,
                        contentDescription = null,
                        modifier = Modifier.size(18.dp),
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    Text(
                        text = selectedFileUri?.let { stringResource(R.string.file_selected) }
                            ?: stringResource(R.string.select_model_file),
                    )
                }

                selectedFileUri?.let { uri ->
                    Text(
                        text = "Selected: ${getCleanFileName(uri)}",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }

                Column(
                    modifier = Modifier.fillMaxWidth(),
                ) {
                    Text(
                        text = stringResource(R.string.lora_files_optional),
                        style = MaterialTheme.typography.titleSmall,
                        modifier = Modifier.padding(bottom = 8.dp),
                    )

                    FilledTonalButton(
                        onClick = {
                            loraPickerLauncher.launch("application/octet-stream")
                        },
                        modifier = Modifier.fillMaxWidth(),
                    ) {
                        Icon(
                            imageVector = Icons.Default.Add,
                            contentDescription = null,
                            modifier = Modifier.size(18.dp),
                        )
                        Spacer(modifier = Modifier.width(8.dp))
                        Text(stringResource(R.string.add_lora_file))
                    }

                    if (selectedLoraFiles.isNotEmpty()) {
                        Spacer(modifier = Modifier.height(8.dp))
                        Text(
                            text = stringResource(R.string.selected_lora_files),
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                        Spacer(modifier = Modifier.height(4.dp))

                        selectedLoraFiles.forEachIndexed { index, loraFile ->
                            key(loraFile.uri.toString()) {
                                Column(
                                    modifier = Modifier
                                        .fillMaxWidth()
                                        .padding(vertical = 2.dp),
                                ) {
                                    Row(
                                        modifier = Modifier.fillMaxWidth(),
                                        horizontalArrangement = Arrangement.SpaceBetween,
                                        verticalAlignment = Alignment.CenterVertically,
                                    ) {
                                        Text(
                                            text = "${index + 1}. ${getCleanFileName(loraFile.uri)}",
                                            style = MaterialTheme.typography.bodySmall,
                                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                                            modifier = Modifier.weight(1f),
                                        )

                                        IconButton(
                                            onClick = {
                                                selectedLoraFiles =
                                                    selectedLoraFiles.filterIndexed { i, _ -> i != index }
                                            },
                                            modifier = Modifier.size(24.dp),
                                        ) {
                                            Icon(
                                                imageVector = Icons.Default.Close,
                                                contentDescription = "delete",
                                                modifier = Modifier.size(16.dp),
                                                tint = MaterialTheme.colorScheme.error,
                                            )
                                        }
                                    }

                                    Row(
                                        modifier = Modifier.fillMaxWidth(),
                                        verticalAlignment = Alignment.CenterVertically,
                                    ) {
                                        Text(
                                            text = stringResource(R.string.lora_weight),
                                            style = MaterialTheme.typography.labelSmall,
                                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                                        )
                                        Spacer(modifier = Modifier.width(8.dp))

                                        Slider(
                                            value = loraFile.weight,
                                            onValueChange = { newWeight ->
                                                selectedLoraFiles =
                                                    selectedLoraFiles.mapIndexed { i, file ->
                                                        if (i == index) file.copy(weight = newWeight) else file
                                                    }
                                            },
                                            valueRange = 0f..2f,
                                            steps = 39,
                                            modifier = Modifier
                                                .weight(1f)
                                                .height(24.dp),
                                        )

                                        Text(
                                            text = "%.2f".format(loraFile.weight),
                                            style = MaterialTheme.typography.labelSmall,
                                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                                            modifier = Modifier.width(35.dp),
                                        )
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        confirmButton = {
            TextButton(
                onClick = {
                    if (modelName.isNotBlank() && selectedFileUri != null && !isIdReserved) {
                        onModelAdded(modelName, selectedFileUri!!, clipSkip, selectedLoraFiles)
                    }
                },
                enabled = modelName.isNotBlank() && selectedFileUri != null && !isIdReserved,
            ) {
                Text(stringResource(R.string.add_model))
            }
        },
        dismissButton = {
            TextButton(onClick = onDismiss) {
                Text(stringResource(R.string.cancel))
            }
        },
    )
}

@Immutable
data class ExtractByteProgress(val extractedBytes: Long, val totalCompressedBytes: Long, val fraction: Float)

private class CountingInputStream(delegate: java.io.InputStream) : java.io.FilterInputStream(delegate) {
    @Volatile
    var bytesRead: Long = 0L
        private set

    override fun read(): Int {
        val b = `in`.read()
        if (b != -1) bytesRead++
        return b
    }

    override fun read(b: ByteArray, off: Int, len: Int): Int {
        val n = `in`.read(b, off, len)
        if (n > 0) bytesRead += n
        return n
    }
}

suspend fun extractNpuModel(
    context: Context,
    modelName: String,
    zipUri: Uri,
    onProgress: (String) -> Unit,
    onByteProgress: (extractedBytes: Long, totalCompressedBytes: Long, fraction: Float) -> Unit,
    onStart: () -> Unit,
    onSuccess: () -> Unit,
    onError: (String) -> Unit,
) = withContext(Dispatchers.IO) {
    try {
        withContext(Dispatchers.Main) {
            onStart()
            onProgress(context.getString(R.string.preparing_npu_model))
        }

        val modelId = modelName.replace(" ", "")

        val modelsDir = File(context.filesDir, "models")
        if (!modelsDir.exists()) {
            modelsDir.mkdirs()
        }

        val modelDir = File(modelsDir, modelId)
        if (modelDir.exists()) {
            modelDir.deleteRecursively()
        }
        modelDir.mkdirs()

        val totalCompressedBytes: Long = try {
            context.contentResolver.openAssetFileDescriptor(zipUri, "r")?.use { it.length }
                ?: -1L
        } catch (_: Exception) {
            -1L
        }

        withContext(Dispatchers.Main) {
            onProgress(context.getString(R.string.extracting_zip_file))
        }

        val rawInputStream = context.contentResolver.openInputStream(zipUri)
            ?: throw Exception(context.getString(R.string.cannot_open_file))

        val countingStream = CountingInputStream(rawInputStream)
        val extractedBytesAtomic = AtomicLong(0L)

        coroutineScope {
            val progressJob = launch {
                while (isActive) {
                    delay(120L)
                    val fraction = if (totalCompressedBytes > 0) {
                        (countingStream.bytesRead.toFloat() / totalCompressedBytes)
                            .coerceIn(0f, 1f)
                    } else {
                        0f
                    }
                    onByteProgress(extractedBytesAtomic.get(), totalCompressedBytes, fraction)
                }
            }

            try {
                ZipInputStream(countingStream.buffered()).use { zipInputStream ->
                    var zipEntry = zipInputStream.nextEntry

                    while (zipEntry != null) {
                        if (!zipEntry.isDirectory) {
                            val fileName = zipEntry.name.substringAfterLast('/')

                            if (fileName.isNotEmpty() &&
                                !fileName.startsWith(".") &&
                                !fileName.startsWith("__MACOSX")
                            ) {
                                val outputFile = File(modelDir, fileName)

                                BufferedOutputStream(outputFile.outputStream()).use { outputStream ->
                                    val tracking = object : OutputStream() {
                                        override fun write(b: Int) {
                                            outputStream.write(b)
                                            extractedBytesAtomic.incrementAndGet()
                                        }
                                        override fun write(b: ByteArray, off: Int, len: Int) {
                                            outputStream.write(b, off, len)
                                            extractedBytesAtomic.addAndGet(len.toLong())
                                        }
                                    }
                                    zipInputStream.copyTo(tracking)
                                }
                            }
                        }
                        zipEntry = zipInputStream.nextEntry
                    }
                }
            } finally {
                progressJob.cancel()
            }
        }

        onByteProgress(extractedBytesAtomic.get(), totalCompressedBytes, 1f)

        if (modelId != "upscaler_anime" && modelId != "upscaler_realistic") {
            val npuCustomFile = File(modelDir, "npucustom")
            npuCustomFile.createNewFile()
        }

        withContext(Dispatchers.Main) {
            onSuccess()
        }
    } catch (e: Exception) {
        Log.e("NpuModelExtract", "Extraction failed", e)

        val modelId = modelName.replace(" ", "")
        val modelDir = File(File(context.filesDir, "models"), modelId)
        if (modelDir.exists()) {
            modelDir.deleteRecursively()
        }

        withContext(Dispatchers.Main) {
            onError(e.message ?: context.getString(R.string.unknown_error))
        }
    }
}

@OptIn(ExperimentalMaterial3ExpressiveApi::class)
@Composable
fun EmbeddingManagerDialog(
    context: Context,
    onDismiss: () -> Unit,
    onEmbeddingDeleted: () -> Unit,
    onEmbeddingImported: () -> Unit,
) {
    var embeddingFiles by remember { mutableStateOf<List<File>>(emptyList()) }
    var showDeleteConfirm by remember { mutableStateOf<File?>(null) }
    var isLoading by remember { mutableStateOf(true) }
    val scope = rememberCoroutineScope()

    fun loadEmbeddings() {
        val embeddingsDir = File(context.filesDir, "embeddings")
        if (!embeddingsDir.exists()) {
            embeddingsDir.mkdirs()
        }
        embeddingFiles = embeddingsDir.listFiles()?.filter {
            it.isFile && it.extension == "safetensors"
        }?.sortedBy { it.name } ?: emptyList()
        isLoading = false
    }

    var errorMessage by remember { mutableStateOf<String?>(null) }

    val embeddingPickerLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent(),
    ) { uri ->
        uri?.let {
            scope.launch {
                importEmbedding(context, it, {
                    loadEmbeddings()
                    onEmbeddingImported()
                }) { error ->
                    errorMessage = error
                }
            }
        }
    }

    if (errorMessage != null) {
        AlertDialog(
            onDismissRequest = { errorMessage = null },
            title = { Text(stringResource(R.string.embedding_import_failed, "")) },
            text = { Text(errorMessage ?: "") },
            confirmButton = {
                TextButton(onClick = { errorMessage = null }) {
                    Text(stringResource(R.string.confirm))
                }
            },
        )
    }

    LaunchedEffect(Unit) {
        loadEmbeddings()
    }

    if (showDeleteConfirm != null) {
        AlertDialog(
            onDismissRequest = { showDeleteConfirm = null },
            title = { Text(stringResource(R.string.delete_embedding)) },
            text = {
                Text(
                    stringResource(
                        R.string.delete_embedding_confirm,
                        showDeleteConfirm!!.name,
                    ),
                )
            },
            confirmButton = {
                TextButton(
                    onClick = {
                        val fileToDelete = showDeleteConfirm!!
                        if (fileToDelete.delete()) {
                            onEmbeddingDeleted()
                            loadEmbeddings()
                        }
                        showDeleteConfirm = null
                    },
                    colors = ButtonDefaults.textButtonColors(
                        contentColor = MaterialTheme.colorScheme.error,
                    ),
                ) {
                    Text(stringResource(R.string.delete))
                }
            },
            dismissButton = {
                TextButton(onClick = { showDeleteConfirm = null }) {
                    Text(stringResource(R.string.cancel))
                }
            },
        )
    }

    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text(stringResource(R.string.embedding_manager)) },
        text = {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .height(400.dp),
            ) {
                if (isLoading) {
                    Box(
                        modifier = Modifier.fillMaxSize(),
                        contentAlignment = Alignment.Center,
                    ) {
                        ContainedLoadingIndicator()
                    }
                } else if (embeddingFiles.isEmpty()) {
                    Box(
                        modifier = Modifier
                            .weight(1f)
                            .fillMaxWidth(),
                        contentAlignment = Alignment.Center,
                    ) {
                        Column(
                            horizontalAlignment = Alignment.CenterHorizontally,
                        ) {
                            Icon(
                                imageVector = Icons.Default.Description,
                                contentDescription = null,
                                modifier = Modifier.size(48.dp),
                                tint = MaterialTheme.colorScheme.onSurfaceVariant,
                            )
                            Spacer(modifier = Modifier.height(16.dp))
                            Text(
                                stringResource(R.string.no_embeddings),
                                style = MaterialTheme.typography.bodyMedium,
                                color = MaterialTheme.colorScheme.onSurfaceVariant,
                            )
                        }
                    }
                } else {
                    LazyColumn(
                        modifier = Modifier
                            .weight(1f)
                            .fillMaxWidth(),
                        verticalArrangement = Arrangement.spacedBy(8.dp),
                    ) {
                        items(embeddingFiles) { file ->
                            Card(
                                modifier = Modifier.fillMaxWidth(),
                                colors = CardDefaults.cardColors(
                                    containerColor = MaterialTheme.colorScheme.surfaceContainerLow,
                                ),
                            ) {
                                Row(
                                    modifier = Modifier
                                        .fillMaxWidth()
                                        .padding(12.dp),
                                    horizontalArrangement = Arrangement.SpaceBetween,
                                    verticalAlignment = Alignment.CenterVertically,
                                ) {
                                    Row(
                                        verticalAlignment = Alignment.CenterVertically,
                                        horizontalArrangement = Arrangement.spacedBy(12.dp),
                                        modifier = Modifier.weight(1f),
                                    ) {
                                        Icon(
                                            imageVector = Icons.Default.Description,
                                            contentDescription = null,
                                            tint = MaterialTheme.colorScheme.primary,
                                        )
                                        Column {
                                            Text(
                                                text = file.nameWithoutExtension,
                                                style = MaterialTheme.typography.titleSmall,
                                            )
                                            Text(
                                                text = formatFileSize(file.length()),
                                                style = MaterialTheme.typography.bodySmall,
                                                color = MaterialTheme.colorScheme.onSurfaceVariant,
                                            )
                                        }
                                    }

                                    IconButton(
                                        onClick = { showDeleteConfirm = file },
                                        colors = IconButtonDefaults.iconButtonColors(
                                            contentColor = MaterialTheme.colorScheme.error,
                                        ),
                                    ) {
                                        Icon(
                                            imageVector = Icons.Default.Delete,
                                            contentDescription = stringResource(R.string.delete_embedding),
                                        )
                                    }
                                }
                            }
                        }
                    }
                }

                Spacer(modifier = Modifier.height(16.dp))

                FilledTonalButton(
                    onClick = {
                        embeddingPickerLauncher.launch("application/octet-stream")
                    },
                    modifier = Modifier.fillMaxWidth(),
                ) {
                    Icon(
                        imageVector = Icons.Default.Add,
                        contentDescription = null,
                        modifier = Modifier.size(18.dp),
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    Text(stringResource(R.string.import_embedding))
                }
            }
        },
        confirmButton = {
            TextButton(onClick = onDismiss) {
                Text(stringResource(R.string.close))
            }
        },
    )
}

suspend fun importEmbedding(context: Context, fileUri: Uri, onSuccess: () -> Unit, onError: (String) -> Unit) = withContext(Dispatchers.IO) {
    try {
        val embeddingsDir = File(context.filesDir, "embeddings")
        if (!embeddingsDir.exists()) {
            embeddingsDir.mkdirs()
        }

        val fileName =
            context.contentResolver.query(fileUri, null, null, null, null)?.use { cursor ->
                val nameIndex = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME)
                cursor.moveToFirst()
                cursor.getString(nameIndex)
            } ?: "embedding_${System.currentTimeMillis()}.safetensors"

        // Validate file extension
        if (!fileName.endsWith(".safetensors", ignoreCase = true)) {
            withContext(Dispatchers.Main) {
                onError(context.getString(R.string.only_safetensors_supported))
            }
            return@withContext
        }

        val targetFile = File(embeddingsDir, fileName)

        context.contentResolver.openInputStream(fileUri)?.use { input ->
            targetFile.outputStream().use { output ->
                input.copyTo(output)
            }
        }

        withContext(Dispatchers.Main) {
            onSuccess()
        }
    } catch (e: Exception) {
        withContext(Dispatchers.Main) {
            onError(e.message ?: context.getString(R.string.unknown_error))
        }
    }
}

suspend fun convertCustomModel(
    context: Context,
    modelName: String,
    fileUri: Uri,
    clipSkip: Int,
    loraFiles: List<LoRAFile>,
    onProgress: (String) -> Unit,
    onStart: () -> Unit,
    onSuccess: () -> Unit,
    onError: (String) -> Unit,
) = withContext(Dispatchers.IO) {
    try {
        withContext(Dispatchers.Main) {
            onStart()
            onProgress(context.getString(R.string.preparing_model))
        }

        val modelId = modelName.replace(" ", "")

        val modelsDir = File(context.filesDir, "models")
        if (!modelsDir.exists()) {
            modelsDir.mkdirs()
        }

        val modelDir = File(modelsDir, modelId)
        if (modelDir.exists()) {
            modelDir.deleteRecursively()
        }
        modelDir.mkdirs()

        withContext(Dispatchers.Main) {
            onProgress(context.getString(R.string.copying_model_file))
        }

        val inputStream = context.contentResolver.openInputStream(fileUri)
            ?: throw Exception(context.getString(R.string.cannot_open_file))
        val modelFile = File(modelDir, "model.safetensors")

        inputStream.use { input ->
            modelFile.outputStream().use { output ->
                input.copyTo(output)
            }
        }

        withContext(Dispatchers.Main) {
            onProgress(context.getString(R.string.copying_lora_files))
        }

        loraFiles.forEachIndexed { index, loraFile ->
            val loraInputStream = context.contentResolver.openInputStream(loraFile.uri)
                ?: throw Exception("Cannot open LoRA file ${index + 1}")
            val loraFileTarget = File(modelDir, "lora.${index + 1}.safetensors")
            val loraWeightFile = File(modelDir, "lora.${index + 1}.weight")

            loraInputStream.use { input ->
                loraFileTarget.outputStream().use { output ->
                    input.copyTo(output)
                }
            }

            loraWeightFile.writeText(loraFile.weight.toString())
        }

        withContext(Dispatchers.Main) {
            onProgress(context.getString(R.string.copying_base_files))
        }

        fun copyAssetsRecursively(assetPath: String, targetDir: File) {
            val assetManager = context.assets
            val assets = assetManager.list(assetPath) ?: emptyArray()

            if (assets.isEmpty()) {
                try {
                    val assetInputStream = assetManager.open(assetPath)
                    val fileName = assetPath.substringAfterLast("/")
                    val targetFile = File(targetDir, fileName)

                    assetInputStream.use { input ->
                        targetFile.outputStream().use { output ->
                            input.copyTo(output)
                        }
                    }
                } catch (e: Exception) {
                    Log.w("ModelConvert", "Could not copy asset: $assetPath", e)
                }
            } else {
                for (asset in assets) {
                    val subAssetPath = "$assetPath/$asset"
                    val subAssets = assetManager.list(subAssetPath) ?: emptyArray()

                    if (subAssets.isEmpty()) {
                        try {
                            val assetInputStream = assetManager.open(subAssetPath)
                            val targetFile = File(targetDir, asset)

                            assetInputStream.use { input ->
                                targetFile.outputStream().use { output ->
                                    input.copyTo(output)
                                }
                            }
                        } catch (e: Exception) {
                            Log.w(
                                "ModelConvert",
                                "Could not copy file: $subAssetPath",
                                e,
                            )
                        }
                    } else {
                        val subTargetDir = File(targetDir, asset)
                        subTargetDir.mkdirs()
                        copyAssetsRecursively(subAssetPath, subTargetDir)
                    }
                }
            }
        }

        copyAssetsRecursively("cvtbase", modelDir)

        withContext(Dispatchers.Main) {
            onProgress(context.getString(R.string.converting_model))
        }

        val nativeDir = context.applicationInfo.nativeLibraryDir
        val executableFile = File(nativeDir, "libstable_diffusion_core.so")

        if (!executableFile.exists()) {
            throw Exception("Executable not found: ${executableFile.absolutePath}")
        }

        var command = listOf(
            executableFile.absolutePath,
            "--convert",
            modelDir.absolutePath,
        )
        val clipSourceFile =
            File(modelDir, if (clipSkip == 2) "clip_skip_2.mnn" else "clip_skip_1.mnn")
        val clipTargetFile = File(modelDir, "clip_v2.mnn")
        clipSourceFile.copyTo(clipTargetFile, overwrite = true)
        if (clipSkip == 2) {
            command += listOf("--clip_skip_2")
        }
        val env = mutableMapOf<String, String>()
        val systemLibPaths = listOf(
            nativeDir,
            "/system/lib64",
            "/vendor/lib64",
            "/vendor/lib64/egl",
        ).joinToString(":")

        env["LD_LIBRARY_PATH"] = systemLibPaths
        env["DSP_LIBRARY_PATH"] = nativeDir

        val processBuilder = ProcessBuilder(command).apply {
            directory(File(nativeDir))
            redirectErrorStream(true)
            environment().putAll(env)
        }

        val process = processBuilder.start()

        process.inputStream.bufferedReader().use { reader ->
            var line: String?
            while (reader.readLine().also { line = it } != null) {
                Log.i("ModelConvert", "Convert: $line")
                withContext(Dispatchers.Main) {
                    onProgress(context.getString(R.string.converting_with_line, line.orEmpty()))
                }
            }
        }

        val exitCode = process.waitFor()
        Log.i("ModelConvert", "Conversion process exited with code: $exitCode")

        val finishedFile = File(modelDir, "finished")
        if (finishedFile.exists()) {
            modelFile.delete()
            val clipSkip1File = File(modelDir, "clip_skip_1.mnn")
            if (clipSkip1File.exists()) {
                clipSkip1File.delete()
            }
            val clipSkip2File = File(modelDir, "clip_skip_2.mnn")
            if (clipSkip2File.exists()) {
                clipSkip2File.delete()
            }

            loraFiles.forEachIndexed { index, _ ->
                val loraFile = File(modelDir, "lora.${index + 1}.safetensors")
                val loraWeightFile = File(modelDir, "lora.${index + 1}.weight")
                if (loraFile.exists()) {
                    loraFile.delete()
                }
                if (loraWeightFile.exists()) {
                    loraWeightFile.delete()
                }
            }

            withContext(Dispatchers.Main) {
                onSuccess()
            }
        } else {
            modelDir.deleteRecursively()
            withContext(Dispatchers.Main) {
                onError(context.getString(R.string.conversion_need_sd15))
            }
        }
    } catch (e: Exception) {
        Log.e("ModelConvert", "Conversion failed", e)

        val modelId = modelName.replace(" ", "")
        val modelDir = File(File(context.filesDir, "models"), modelId)
        if (modelDir.exists()) {
            modelDir.deleteRecursively()
        }

        withContext(Dispatchers.Main) {
            onError(e.message ?: context.getString(R.string.unknown_error))
        }
    }
}

private fun getFileNameFromUri(context: Context, uri: Uri): String? = try {
    when (uri.scheme) {
        "content" -> {
            context.contentResolver.query(uri, null, null, null, null)?.use { cursor ->
                val nameIndex = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME)
                if (cursor.moveToFirst() && nameIndex != -1) {
                    cursor.getString(nameIndex)
                } else {
                    null
                }
            }
        }

        "file" -> {
            uri.lastPathSegment
        }

        else -> {
            DocumentFile.fromSingleUri(context, uri)?.name
        }
    }
} catch (e: Exception) {
    Log.e("GetFileName", "Get file name from uri failed", e)
    null
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun SettingNavCard(icon: ImageVector, label: String, onClick: () -> Unit, modifier: Modifier = Modifier) {
    Card(
        onClick = onClick,
        modifier = modifier,
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surfaceContainerLow,
        ),
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 16.dp, vertical = 14.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Icon(
                imageVector = icon,
                contentDescription = null,
                tint = MaterialTheme.colorScheme.primary,
            )
            Spacer(Modifier.width(16.dp))
            Text(
                text = label,
                style = MaterialTheme.typography.bodyLarge,
                modifier = Modifier.weight(1f),
            )
            Icon(
                imageVector = Icons.AutoMirrored.Filled.KeyboardArrowRight,
                contentDescription = null,
                tint = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
    }
}

@Composable
private fun SwitchSettingRow(
    title: String,
    description: String,
    checked: Boolean,
    onCheckedChange: (Boolean) -> Unit,
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(16.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.SpaceBetween,
    ) {
        Column(modifier = Modifier.weight(1f)) {
            Text(
                text = title,
                style = MaterialTheme.typography.titleSmall,
            )
            Text(
                text = description,
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
        Switch(
            checked = checked,
            onCheckedChange = onCheckedChange,
        )
    }
}

@OptIn(ExperimentalMaterial3Api::class, ExperimentalMaterial3ExpressiveApi::class)
@Composable
private fun AppearanceSection() {
    val themeController = LocalThemeController.current
    val state = themeController.state
    val dynamicColorSupported = Build.VERSION.SDK_INT >= Build.VERSION_CODES.S
    val isDark = when (state.darkMode) {
        DarkModePreference.SYSTEM -> isSystemInDarkTheme()
        DarkModePreference.LIGHT -> false
        DarkModePreference.DARK -> true
    }

    Column {
        Row(
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(8.dp),
            modifier = Modifier.padding(bottom = 12.dp),
        ) {
            Icon(
                imageVector = Icons.Default.Palette,
                contentDescription = null,
                tint = MaterialTheme.colorScheme.primary,
                modifier = Modifier.size(20.dp),
            )
            Text(
                stringResource(R.string.appearance),
                style = MaterialTheme.typography.titleMedium,
            )
        }
        Card(
            modifier = Modifier.fillMaxWidth(),
            colors = CardDefaults.cardColors(
                containerColor = MaterialTheme.colorScheme.surfaceContainerLow,
            ),
        ) {
            if (dynamicColorSupported) {
                SwitchSettingRow(
                    title = stringResource(R.string.dynamic_color),
                    description = stringResource(R.string.dynamic_color_hint),
                    checked = state.dynamicColor,
                    onCheckedChange = { value ->
                        themeController.update { it.copy(dynamicColor = value) }
                    },
                )
                HorizontalDivider(modifier = Modifier.padding(horizontal = 16.dp))
            }

            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(16.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                Text(
                    stringResource(R.string.theme_preset),
                    style = MaterialTheme.typography.titleSmall,
                )
                Text(
                    stringResource(R.string.theme_preset_hint),
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(top = 4.dp),
                    horizontalArrangement = Arrangement.spacedBy(12.dp),
                ) {
                    ThemePreset.entries.forEach { preset ->
                        ThemeSwatch(
                            preset = preset,
                            isDark = isDark,
                            selected = preset == state.preset && !state.dynamicColor,
                            enabled = !state.dynamicColor,
                            onClick = {
                                themeController.update { it.copy(preset = preset) }
                            },
                            modifier = Modifier.weight(1f),
                        )
                    }
                }
            }
            HorizontalDivider(modifier = Modifier.padding(horizontal = 16.dp))
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(16.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                Text(
                    stringResource(R.string.dark_mode),
                    style = MaterialTheme.typography.titleSmall,
                )
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(
                        ButtonGroupDefaults.ConnectedSpaceBetween,
                    ),
                ) {
                    val modes = DarkModePreference.entries
                    modes.forEachIndexed { index, mode ->
                        val shapes = when (index) {
                            0 -> ButtonGroupDefaults.connectedLeadingButtonShapes()
                            modes.lastIndex -> ButtonGroupDefaults.connectedTrailingButtonShapes()
                            else -> ButtonGroupDefaults.connectedMiddleButtonShapes()
                        }
                        ToggleButton(
                            checked = mode == state.darkMode,
                            onCheckedChange = { checked ->
                                if (checked) themeController.update { it.copy(darkMode = mode) }
                            },
                            shapes = shapes,
                            modifier = Modifier.weight(1f),
                        ) {
                            Text(
                                text = stringResource(
                                    when (mode) {
                                        DarkModePreference.SYSTEM -> R.string.dark_mode_system
                                        DarkModePreference.LIGHT -> R.string.dark_mode_light
                                        DarkModePreference.DARK -> R.string.dark_mode_dark
                                    },
                                ),
                            )
                        }
                    }
                }
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class, ExperimentalMaterial3ExpressiveApi::class)
@Composable
private fun ThemeSwatch(
    preset: ThemePreset,
    isDark: Boolean,
    selected: Boolean,
    enabled: Boolean,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
) {
    val scheme = preset.scheme(isDark)
    val alpha = if (enabled) 1f else 0.45f
    val description = stringResource(preset.nameRes)
    val polygon = when (preset) {
        ThemePreset.TANGERINE -> MaterialShapes.Cookie9Sided
        ThemePreset.FOREST -> MaterialShapes.Clover4Leaf
        ThemePreset.OCEAN -> MaterialShapes.Sunny
        ThemePreset.AMBER -> MaterialShapes.Cookie6Sided
    }
    val shape = polygon.toShape()
    Box(modifier = modifier, contentAlignment = Alignment.Center) {
        Surface(
            onClick = onClick,
            enabled = enabled,
            shape = shape,
            color = scheme.primary.copy(alpha = alpha),
            border = if (selected) {
                BorderStroke(3.dp, MaterialTheme.colorScheme.primary)
            } else {
                null
            },
            modifier = Modifier
                .size(48.dp)
                .semantics { contentDescription = description },
        ) {
            Box(contentAlignment = Alignment.Center) {
                if (selected) {
                    Icon(
                        imageVector = Icons.Default.Check,
                        contentDescription = null,
                        tint = scheme.onPrimary,
                        modifier = Modifier.size(22.dp),
                    )
                }
            }
        }
    }
}
