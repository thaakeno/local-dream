package io.github.xororz.localdream.ui.screens

import android.content.Intent
import android.net.Uri
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import io.github.xororz.localdream.service.ModelDownloadService

@Composable
internal fun DirectDownloadDialog(onDismiss: () -> Unit) {
    val context = LocalContext.current
    var url by remember { mutableStateOf("") }
    var name by remember { mutableStateOf("") }

    val normalized = normalizeHubUrl(url.trim())
    val parsed = runCatching { Uri.parse(normalized) }.getOrNull()
    val validHttp = parsed?.scheme.equals("https", true) || parsed?.scheme.equals("http", true)
    val fileName = parsed?.lastPathSegment?.takeIf { it.isNotBlank() } ?: "model.bin"
    val suggestedName = fileName.substringBeforeLast('.', fileName)
    val displayName = name.ifBlank { suggestedName }
    val modelId = displayName
        .replace(Regex("""[^A-Za-z0-9._-]+"""), "_")
        .trim('_', '.')
        .ifBlank { "DirectDownload" }

    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("Download model URL") },
        text = {
            Column {
                Text(
                    "Paste a Hugging Face file URL. /blob/ links are converted to /resolve/. Xet is used automatically when the Hub exposes Xet metadata.",
                    modifier = Modifier.padding(bottom = 12.dp),
                )
                OutlinedTextField(
                    value = url,
                    onValueChange = { url = it },
                    label = { Text("Hugging Face download URL") },
                    modifier = Modifier.fillMaxWidth(),
                    isError = url.isNotBlank() && !validHttp,
                )
                OutlinedTextField(
                    value = name,
                    onValueChange = { name = it },
                    label = { Text("Model name (optional)") },
                    placeholder = { Text(suggestedName) },
                    modifier = Modifier.fillMaxWidth().padding(top = 10.dp),
                    singleLine = true,
                )
            }
        },
        confirmButton = {
            TextButton(
                enabled = validHttp && normalized.isNotBlank(),
                onClick = {
                    val intent = Intent(context, ModelDownloadService::class.java).apply {
                        action = ModelDownloadService.ACTION_START_DOWNLOAD
                        putExtra(ModelDownloadService.EXTRA_MODEL_ID, modelId)
                        putExtra(ModelDownloadService.EXTRA_MODEL_NAME, displayName)
                        putExtra(ModelDownloadService.EXTRA_FILE_URL, normalized)
                        putExtra(ModelDownloadService.EXTRA_TARGET_FILE_NAME, fileName)
                        putExtra(
                            ModelDownloadService.EXTRA_IS_ZIP,
                            fileName.endsWith(".zip", ignoreCase = true),
                        )
                        putExtra(ModelDownloadService.EXTRA_MODEL_TYPE, ModelDownloadService.TYPE_SD)
                    }
                    ContextCompat.startForegroundService(context, intent)
                    onDismiss()
                },
            ) { Text("Download") }
        },
        dismissButton = {
            TextButton(onClick = onDismiss) { Text("Cancel") }
        },
    )
}

private fun normalizeHubUrl(url: String): String =
    if (url.isBlank()) url else url.replace("/blob/", "/resolve/")
