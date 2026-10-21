package io.github.xororz.localdream.ui.screens

import android.content.ClipData
import android.content.Intent
import android.widget.Toast
import androidx.compose.animation.AnimatedContent
import androidx.compose.animation.animateContentSize
import androidx.compose.animation.core.spring
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.outlined.AudioFile
import androidx.compose.material.icons.outlined.Favorite
import androidx.compose.material.icons.outlined.MusicNote
import androidx.compose.material.icons.outlined.Search
import androidx.compose.material.icons.outlined.SwapVert
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.core.content.FileProvider
import io.github.xororz.localdream.service.MusicHistoryItem
import io.github.xororz.localdream.service.MusicHistoryStore
import io.github.xororz.localdream.service.SpeechHistoryItem
import io.github.xororz.localdream.service.SpeechHistoryStore
import io.github.xororz.localdream.ui.components.MusicPlayerCard
import io.github.xororz.localdream.ui.glass.DreamGlassRole
import io.github.xororz.localdream.ui.glass.DreamGlassSurface
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.text.DateFormat
import java.util.Date

/** History is one destination, but each medium retains its own persistent store. */
internal enum class MediaHistoryTab { IMAGES, MUSIC, SPEECH }

@Composable
internal fun UnifiedAudioHistoryPage(tab: MediaHistoryTab) {
    require(tab != MediaHistoryTab.IMAGES)
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    var tracks by remember { mutableStateOf<List<MusicHistoryItem>>(emptyList()) }
    var voices by remember { mutableStateOf<List<SpeechHistoryItem>>(emptyList()) }
    var refreshing by remember { mutableStateOf(true) }
    var revision by remember { mutableIntStateOf(0) }
    var query by remember { mutableStateOf("") }
    var favoritesOnly by remember { mutableStateOf(false) }
    var newestFirst by remember { mutableStateOf(true) }
    var deleteAction by remember { mutableStateOf<(() -> Unit)?>(null) }

    LaunchedEffect(tab, revision) {
        refreshing = true
        if (tab == MediaHistoryTab.MUSIC) {
            tracks = withContext(Dispatchers.IO) { MusicHistoryStore.load(context) }
        } else {
            voices = SpeechHistoryStore.load(context)
        }
        refreshing = false
    }

    val isMusic = tab == MediaHistoryTab.MUSIC
    val filteredMusic = remember(tracks, query, favoritesOnly, newestFirst) {
        tracks.asSequence()
            .filter { !favoritesOnly || it.favorite }
            .filter {
                query.isBlank() || it.style.contains(query, true) ||
                    it.lyrics.contains(query, true) || it.modelId.contains(query, true)
            }
            .let { if (newestFirst) it.sortedByDescending { m -> m.createdAtMillis }
                   else it.sortedBy { m -> m.createdAtMillis } }
            .toList()
    }
    val filteredSpeech = remember(voices, query, favoritesOnly, newestFirst) {
        voices.asSequence()
            .filter { !favoritesOnly || it.favorite }
            .filter {
                query.isBlank() || it.text.contains(query, true) ||
                    it.instruction.contains(query, true) || it.modelId.contains(query, true)
            }
            .let { if (newestFirst) it.sortedByDescending { v -> v.createdAt }
                   else it.sortedBy { v -> v.createdAt } }
            .toList()
    }
    val allCount = if (isMusic) tracks.size else voices.size
    val filteredCount = if (isMusic) filteredMusic.size else filteredSpeech.size
    val header = if (isMusic) "Music library" else "Voice library"
    val description = if (isMusic)
        "Your YuE2 tracks, prompts and generation settings"
    else "Breeze TTS recordings, voices and generations"

    Column(Modifier.fillMaxSize()) {
        DreamGlassSurface(
            modifier = Modifier.fillMaxWidth().padding(horizontal = 14.dp, vertical = 8.dp),
            shape = RoundedCornerShape(28.dp),
            role = DreamGlassRole.OVERLAY,
        ) {
            Column(
                Modifier.fillMaxWidth().padding(16.dp).animateContentSize(spring()),
                verticalArrangement = Arrangement.spacedBy(9.dp),
            ) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Icon(
                        if (isMusic) Icons.Outlined.MusicNote else Icons.Outlined.AudioFile,
                        contentDescription = null, tint = MaterialTheme.colorScheme.primary,
                    )
                    Spacer(Modifier.width(10.dp))
                    Column(Modifier.weight(1f)) {
                        Text(header, style = MaterialTheme.typography.titleLarge,
                            fontWeight = FontWeight.SemiBold)
                        Text(description, style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                            maxLines = 2)
                    }
                    Text("$allCount", style = MaterialTheme.typography.headlineSmall,
                        color = MaterialTheme.colorScheme.primary)
                }
                OutlinedTextField(
                    value = query,
                    onValueChange = { query = it },
                    modifier = Modifier.fillMaxWidth(),
                    singleLine = true,
                    shape = RoundedCornerShape(19.dp),
                    textStyle = MaterialTheme.typography.bodyMedium,
                    leadingIcon = { Icon(Icons.Outlined.Search, null) },
                    placeholder = { Text("Search prompts or models") },
                )
                Row(
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    FilterChip(
                        selected = favoritesOnly,
                        onClick = { favoritesOnly = !favoritesOnly },
                        label = { Text("Favorites") },
                        leadingIcon = { Icon(Icons.Outlined.Favorite, null, Modifier.size(16.dp)) },
                    )
                    FilterChip(
                        selected = !newestFirst,
                        onClick = { newestFirst = !newestFirst },
                        label = { Text(if (newestFirst) "Newest" else "Oldest") },
                        leadingIcon = { Icon(Icons.Outlined.SwapVert, null, Modifier.size(16.dp)) },
                    )
                    Spacer(Modifier.weight(1f))
                    TextButton(onClick = { revision++ }) { Text("Refresh") }
                }
            }
        }
        if (refreshing) {
            LinearProgressIndicator(Modifier.fillMaxWidth().padding(horizontal = 22.dp))
        }
        AnimatedContent(
            targetState = Triple(isMusic, filteredCount == 0, refreshing),
            label = "history-collection-transition",
            modifier = Modifier.fillMaxSize(),
        ) { state ->
            if (!state.third && state.second) {
                Box(Modifier.fillMaxSize().padding(24.dp), contentAlignment = Alignment.Center) {
                    Column(horizontalAlignment = Alignment.CenterHorizontally,
                        verticalArrangement = Arrangement.spacedBy(10.dp)) {
                        Icon(if (isMusic) Icons.Outlined.MusicNote else Icons.Outlined.AudioFile,
                            contentDescription = null, modifier = Modifier.size(42.dp),
                            tint = MaterialTheme.colorScheme.primary)
                        Text(if (allCount == 0) "Nothing generated yet" else "No matches",
                            style = MaterialTheme.typography.titleMedium)
                        Text(if (isMusic) "Finished YuE2 tracks appear here."
                             else "Finished Breeze voice recordings appear here.",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant)
                    }
                }
            } else {
                LazyColumn(
                    modifier = Modifier.fillMaxSize(),
                    contentPadding = PaddingValues(start = 14.dp, end = 14.dp, bottom = 32.dp, top = 4.dp),
                    verticalArrangement = Arrangement.spacedBy(10.dp),
                ) {
                    if (state.first) {
                        items(filteredMusic, key = { "yue2:" + it.id }) { item ->
                            MusicPlayerCard(
                                file = item.file,
                                title = item.style.take(96).ifBlank { "YuE2 track" },
                                subtitle = mediaDate(item.createdAtMillis) +
                                    " · ${item.targetSeconds}s · ${item.format.uppercase()}",
                                metadataLine = "${item.steps} steps · seed ${item.lmSeed} · ${item.modelId}",
                                compact = true,
                                favorite = item.favorite,
                                onFavoriteToggle = {
                                    tracks = tracks.map {
                                        if (it.id == item.id) it.copy(favorite = !item.favorite) else it
                                    }
                                    scope.launch(Dispatchers.IO) {
                                        MusicHistoryStore.setFavorite(context, item.id, !item.favorite)
                                        val updated = MusicHistoryStore.load(context)
                                        withContext(Dispatchers.Main) { tracks = updated }
                                    }
                                },
                                onSave = {
                                    scope.launch {
                                        val result = withContext(Dispatchers.IO) {
                                            runCatching { MusicHistoryStore.exportToMusic(context, item) }
                                        }
                                        toast(context, if (result.isSuccess) "Saved to Music/Local Dream"
                                            else "Could not export track")
                                    }
                                },
                                onShare = {
                                    scope.launch {
                                        val uri = withContext(Dispatchers.IO) {
                                            runCatching { MusicHistoryStore.exportToMusic(context, item) }.getOrNull()
                                        }
                                        if (uri != null) shareAudio(context, uri,
                                            MusicHistoryStore.mimeType(item.format), "Share YuE2 track")
                                        else toast(context, "Could not share track")
                                    }
                                },
                                onDelete = {
                                    deleteAction = {
                                        scope.launch {
                                            withContext(Dispatchers.IO) {
                                                MusicHistoryStore.delete(context, item.id)
                                            }
                                            tracks = withContext(Dispatchers.IO) { MusicHistoryStore.load(context) }
                                        }
                                    }
                                },
                            )
                        }
                    } else {
                        items(filteredSpeech, key = { "breeze:" + it.modelId + ":" + it.id }) { item ->
                            val audioFile = remember(item.filePath) { File(item.filePath) }
                            MusicPlayerCard(
                                file = audioFile,
                                title = item.text.take(90).ifBlank { "Breeze recording" },
                                subtitle = mediaDate(item.createdAt) +
                                    " · ${item.audioDurationMillis / 1000}s · ${item.modelId}",
                                metadataLine = "Seed ${item.seed} · CFG ${item.cfg}",
                                compact = true,
                                favorite = item.favorite,
                                onFavoriteToggle = {
                                    voices = voices.map {
                                        if (it.id == item.id && it.modelId == item.modelId)
                                            it.copy(favorite = !item.favorite) else it
                                    }
                                    scope.launch {
                                        SpeechHistoryStore.setFavorite(
                                            context, item.modelId, item.id, !item.favorite)
                                        voices = SpeechHistoryStore.load(context)
                                    }
                                },
                                onSave = {
                                    scope.launch {
                                        val saved = runCatching {
                                            SpeechHistoryStore.exportToMusic(context, audioFile)
                                        }.getOrNull()
                                        toast(context, if (saved != null) "Saved to Music/LocalDream"
                                            else "Could not export recording")
                                    }
                                },
                                onShare = {
                                    val uri = runCatching {
                                        FileProvider.getUriForFile(context,
                                            context.packageName + ".fileprovider", audioFile)
                                    }.getOrNull()
                                    if (uri != null) shareAudio(context, uri, "audio/wav",
                                        "Share Breeze recording")
                                    else toast(context, "Could not share recording")
                                },
                                onDelete = {
                                    deleteAction = {
                                        scope.launch {
                                            SpeechHistoryStore.delete(context, item.modelId, item.id)
                                            voices = SpeechHistoryStore.load(context)
                                        }
                                    }
                                },
                            )
                        }
                    }
                }
            }
        }
    }
    if (deleteAction != null) {
        AlertDialog(
            onDismissRequest = { deleteAction = null },
            title = { Text("Delete recording?") },
            text = { Text("This removes the local audio file and its history entry.") },
            confirmButton = {
                TextButton(onClick = {
                    val action = deleteAction
                    deleteAction = null
                    action?.invoke()
                }) { Text("Delete", color = MaterialTheme.colorScheme.error) }
            },
            dismissButton = { TextButton(onClick = { deleteAction = null }) { Text("Cancel") } },
        )
    }
}

private fun mediaDate(time: Long): String =
    DateFormat.getDateTimeInstance(DateFormat.MEDIUM, DateFormat.SHORT).format(Date(time))

private fun toast(context: android.content.Context, message: String) {
    Toast.makeText(context, message, Toast.LENGTH_SHORT).show()
}

private fun shareAudio(context: android.content.Context, uri: android.net.Uri,
                       mimeType: String, title: String) {
    val send = Intent(Intent.ACTION_SEND).apply {
        type = mimeType
        putExtra(Intent.EXTRA_STREAM, uri)
        addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        clipData = ClipData.newRawUri("Local Dream audio", uri)
    }
    context.startActivity(Intent.createChooser(send, title))
}
