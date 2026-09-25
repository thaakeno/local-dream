package io.github.xororz.localdream.ui.components

import android.widget.Toast
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ContentCopy
import androidx.compose.material.icons.filled.Save
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.dp
import io.github.xororz.localdream.utils.SafeClipboard

@Composable
fun CrashRecoveryDialog(
    report: String,
    onDismiss: () -> Unit,
) {
    val context = LocalContext.current
    val saveLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.CreateDocument("text/plain"),
    ) { uri ->
        if (uri == null) return@rememberLauncherForActivityResult
        runCatching {
            context.contentResolver.openOutputStream(uri)?.bufferedWriter()?.use {
                it.write(report)
            } ?: error("Could not open output file")
        }.onSuccess {
            Toast.makeText(context, "Crash report saved", Toast.LENGTH_SHORT).show()
        }.onFailure {
            Toast.makeText(
                context,
                "Could not save crash report: ${it.message}",
                Toast.LENGTH_LONG,
            ).show()
        }
    }

    // Rendering several MB of log text itself can make a recovering app jank.
    // Show a useful tail here. Save always keeps the complete report; Copy uses
    // a Binder-safe payload and falls back to a bounded tail for huge reports.
    val preview = if (report.length > 160_000) {
        "… preview trimmed; Copy/Save contains the full report …\n\n" +
            report.takeLast(160_000)
    } else {
        report
    }

    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("Local Dream recovered from a crash") },
        text = {
            Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
                Text(
                    "The full runtime/crash report and last generation/backend log were recovered. " +
                        "Save keeps the complete report. Copy keeps it whole when safe and uses a bounded tail when it is too large for Android's clipboard.",
                    style = MaterialTheme.typography.bodyMedium,
                )
                Surface(
                    modifier = Modifier
                        .fillMaxWidth()
                        .heightIn(min = 220.dp, max = 520.dp),
                    color = MaterialTheme.colorScheme.surfaceContainerHighest,
                    shape = MaterialTheme.shapes.medium,
                ) {
                    Text(
                        text = preview,
                        modifier = Modifier
                            .verticalScroll(rememberScrollState())
                            .padding(12.dp),
                        style = MaterialTheme.typography.bodySmall.copy(
                            fontFamily = FontFamily.Monospace,
                        ),
                    )
                }
            }
        },
        confirmButton = {
            Row(horizontalArrangement = Arrangement.spacedBy(4.dp)) {
                TextButton(
                    onClick = {
                        val result = SafeClipboard.copyText(
                            context,
                            "Local Dream crash report",
                            report,
                        )
                        val message = when {
                            !result.copied -> "Could not copy crash report; use Save"
                            result.truncated -> "Report is huge; copied a safe tail. Save for the full report"
                            else -> "Crash + generation report copied"
                        }
                        Toast.makeText(context, message, Toast.LENGTH_SHORT).show()
                    },
                ) {
                    Icon(Icons.Default.ContentCopy, contentDescription = null)
                    Text("Copy")
                }
                TextButton(
                    onClick = {
                        saveLauncher.launch(
                            "LocalDream_crash_${System.currentTimeMillis()}.txt",
                        )
                    },
                ) {
                    Icon(Icons.Default.Save, contentDescription = null)
                    Text("Save")
                }
            }
        },
        dismissButton = {
            TextButton(onClick = onDismiss) {
                Text("Close")
            }
        },
    )
}
