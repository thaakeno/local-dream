package io.github.xororz.localdream.ui.components

import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.CollectionsBookmark
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.Edit
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.AssistChip
import androidx.compose.material3.AssistChipDefaults
import androidx.compose.material3.Button
import androidx.compose.material3.FilledTonalIconButton
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import io.github.xororz.localdream.R
import io.github.xororz.localdream.data.HistoryCollection

@Composable
fun HistoryCollectionBar(
    collections: List<HistoryCollection>,
    selectedCollectionIds: Set<Long>?,
    onSelectCollection: (Long?) -> Unit,
    onCreateCollection: () -> Unit,
    onManageCollections: () -> Unit,
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .horizontalScroll(rememberScrollState())
            .padding(horizontal = 12.dp, vertical = 6.dp),
        horizontalArrangement = Arrangement.spacedBy(8.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        val allSelected = selectedCollectionIds.isNullOrEmpty()
        AssistChip(
            onClick = { onSelectCollection(null) },
            label = { Text(stringResource(R.string.collection_all)) },
            leadingIcon = { Icon(Icons.Default.CollectionsBookmark, contentDescription = null) },
            colors = AssistChipDefaults.assistChipColors(
                containerColor = if (allSelected) {
                    MaterialTheme.colorScheme.primaryContainer
                } else {
                    MaterialTheme.colorScheme.surfaceContainer
                },
            ),
            border = null,
            shape = CircleShape,
            modifier = Modifier.height(38.dp),
        )
        collections.forEach { collection ->
            val selected = collection.id in selectedCollectionIds.orEmpty()
            AssistChip(
                onClick = { onSelectCollection(collection.id) },
                label = {
                    Text(
                        "${collection.name} · ${collection.itemCount}",
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis,
                    )
                },
                colors = AssistChipDefaults.assistChipColors(
                    containerColor = if (selected) {
                        MaterialTheme.colorScheme.primaryContainer
                    } else {
                        MaterialTheme.colorScheme.surfaceContainer
                    },
                ),
                border = null,
                shape = CircleShape,
                modifier = Modifier.height(38.dp),
            )
        }
        FilledTonalIconButton(onClick = onCreateCollection) {
            Icon(Icons.Default.Add, contentDescription = stringResource(R.string.collection_new))
        }
        if (collections.isNotEmpty()) {
            FilledTonalIconButton(onClick = onManageCollections) {
                Icon(Icons.Default.Settings, contentDescription = stringResource(R.string.collection_manage))
            }
        }
    }
}

@Composable
fun AddToCollectionDialog(
    collections: List<HistoryCollection>,
    itemCount: Int,
    onAddToExisting: (HistoryCollection) -> Unit,
    onCreateAndAdd: (String) -> Unit,
    onDismiss: () -> Unit,
) {
    var newName by remember { mutableStateOf("") }
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text(stringResource(R.string.collection_add_to)) },
        text = {
            Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                Text(
                    stringResource(R.string.collection_add_count, itemCount),
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
                collections.forEach { collection ->
                    AssistChip(
                        onClick = { onAddToExisting(collection) },
                        label = { Text("${collection.name} · ${collection.itemCount}", maxLines = 1) },
                        leadingIcon = {
                            Icon(Icons.Default.CollectionsBookmark, contentDescription = null)
                        },
                        modifier = Modifier.fillMaxWidth(),
                    )
                }
                OutlinedTextField(
                    value = newName,
                    onValueChange = { newName = it.take(60) },
                    label = { Text(stringResource(R.string.collection_new_name)) },
                    singleLine = true,
                    modifier = Modifier.fillMaxWidth(),
                )
                Button(
                    onClick = { onCreateAndAdd(newName.trim()) },
                    enabled = newName.isNotBlank(),
                    modifier = Modifier.fillMaxWidth(),
                ) {
                    Icon(Icons.Default.Add, contentDescription = null)
                    Text(
                        stringResource(R.string.collection_create_and_add),
                        modifier = Modifier.padding(start = 6.dp),
                    )
                }
            }
        },
        confirmButton = {},
        dismissButton = {
            TextButton(onClick = onDismiss) {
                Text(stringResource(R.string.close))
            }
        },
    )
}

@Composable
fun ManageCollectionsDialog(
    collections: List<HistoryCollection>,
    onRename: (HistoryCollection, String) -> Unit,
    onDelete: (HistoryCollection) -> Unit,
    onDismiss: () -> Unit,
) {
    var editing by remember { mutableStateOf<HistoryCollection?>(null) }
    var editName by remember { mutableStateOf("") }

    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text(stringResource(R.string.collection_manage)) },
        text = {
            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                collections.forEach { collection ->
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        Column(modifier = Modifier.weight(1f)) {
                            Text(
                                collection.name,
                                style = MaterialTheme.typography.titleSmall,
                                maxLines = 1,
                                overflow = TextOverflow.Ellipsis,
                            )
                            Text(
                                stringResource(R.string.collection_items_count, collection.itemCount),
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant,
                            )
                        }
                        FilledTonalIconButton(
                            onClick = {
                                editing = collection
                                editName = collection.name
                            },
                        ) {
                            Icon(Icons.Default.Edit, contentDescription = stringResource(R.string.rename))
                        }
                        FilledTonalIconButton(onClick = { onDelete(collection) }) {
                            Icon(
                                Icons.Default.Delete,
                                contentDescription = stringResource(R.string.delete),
                                tint = MaterialTheme.colorScheme.error,
                            )
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
    )

    editing?.let { collection ->
        AlertDialog(
            onDismissRequest = { editing = null },
            title = { Text(stringResource(R.string.collection_rename)) },
            text = {
                OutlinedTextField(
                    value = editName,
                    onValueChange = { editName = it.take(60) },
                    singleLine = true,
                    modifier = Modifier.fillMaxWidth(),
                )
            },
            confirmButton = {
                TextButton(
                    onClick = {
                        onRename(collection, editName.trim())
                        editing = null
                    },
                    enabled = editName.isNotBlank() && editName.trim() != collection.name,
                ) {
                    Text(stringResource(R.string.rename))
                }
            },
            dismissButton = {
                TextButton(onClick = { editing = null }) {
                    Text(stringResource(R.string.cancel))
                }
            },
        )
    }
}
