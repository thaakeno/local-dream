package io.github.xororz.localdream.ui.components

import android.content.Context
import android.content.Intent
import android.net.Uri
import androidx.compose.foundation.Image
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Info
import androidx.compose.material.icons.filled.OpenInNew
import androidx.compose.material3.AssistChip
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.ListItem
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import io.github.xororz.localdream.BuildConfig
import io.github.xororz.localdream.R

@Composable
fun AboutSection() {
    val context = LocalContext.current
    val repo = BuildConfig.GIT_REPOSITORY.takeIf { it.isNotBlank() && it != "unknown" }
    val upstream = BuildConfig.UPSTREAM_REPOSITORY.takeIf { it.isNotBlank() }
    val fullSha = BuildConfig.GIT_SHA.takeIf { it.isNotBlank() && it != "unknown" }
    val shortSha = fullSha?.take(8) ?: stringResource(R.string.about_unknown)
    val branch = BuildConfig.GIT_BRANCH.takeIf { it.isNotBlank() && it != "unknown" }
        ?: stringResource(R.string.about_unknown)

    Column {
        Row(
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(8.dp),
            modifier = Modifier.padding(bottom = 12.dp),
        ) {
            Icon(
                imageVector = Icons.Default.Info,
                contentDescription = null,
                tint = MaterialTheme.colorScheme.primary,
                modifier = Modifier.size(20.dp),
            )
            Text(
                stringResource(R.string.about_section),
                style = MaterialTheme.typography.titleMedium,
            )
        }

        Card(
            modifier = Modifier.fillMaxWidth(),
            colors = CardDefaults.cardColors(
                containerColor = MaterialTheme.colorScheme.surfaceContainerLow,
            ),
        ) {
            Column {
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(16.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Image(
                        painter = painterResource(R.drawable.ic_launcher_foreground),
                        contentDescription = null,
                        modifier = Modifier.size(54.dp),
                    )
                    Spacer(Modifier.width(14.dp))
                    Column(modifier = Modifier.weight(1f)) {
                        Text(
                            text = "Local Dream",
                            style = MaterialTheme.typography.titleLarge,
                            fontWeight = FontWeight.SemiBold,
                        )
                        Text(
                            text = "${BuildConfig.VERSION_NAME} • " +
                                stringResource(R.string.about_build_code, BuildConfig.VERSION_CODE),
                            style = MaterialTheme.typography.bodyMedium,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                    }
                    if (BuildConfig.IS_FORK) {
                        AssistChip(
                            onClick = {
                                repo?.let { openGithub(context, "https://github.com/$it") }
                            },
                            label = { Text(stringResource(R.string.about_fork_badge)) },
                            leadingIcon = {
                                Icon(
                                    painter = painterResource(R.drawable.ic_github),
                                    contentDescription = null,
                                    modifier = Modifier.size(18.dp),
                                )
                            },
                        )
                    }
                }

                HorizontalDivider()

                AboutRow(
                    title = stringResource(R.string.about_source),
                    value = repo ?: stringResource(R.string.about_unknown),
                    github = true,
                    onClick = repo?.let { value ->
                        { openGithub(context, "https://github.com/$value") }
                    },
                )
                AboutRow(
                    title = stringResource(R.string.about_commit),
                    value = shortSha,
                    monospace = true,
                    onClick = if (repo != null && fullSha != null) {
                        { openGithub(context, "https://github.com/$repo/commit/$fullSha") }
                    } else {
                        null
                    },
                )
                AboutRow(
                    title = stringResource(R.string.about_branch),
                    value = branch,
                    monospace = true,
                    onClick = repo?.let { value ->
                        { openGithub(context, "https://github.com/$value/tree/${Uri.encode(branch)}") }
                    },
                )
                if (BuildConfig.IS_FORK && upstream != null) {
                    AboutRow(
                        title = stringResource(R.string.about_upstream),
                        value = upstream,
                        github = true,
                        onClick = { openGithub(context, "https://github.com/$upstream") },
                    )
                }
            }
        }
    }
}

@Composable
private fun AboutRow(
    title: String,
    value: String,
    github: Boolean = false,
    monospace: Boolean = false,
    onClick: (() -> Unit)? = null,
) {
    ListItem(
        modifier = if (onClick != null) Modifier.clickable(onClick = onClick) else Modifier,
        colors = androidx.compose.material3.ListItemDefaults.colors(
            containerColor = androidx.compose.ui.graphics.Color.Transparent,
        ),
        leadingContent = if (github) {
            {
                Icon(
                    painter = painterResource(R.drawable.ic_github),
                    contentDescription = null,
                    modifier = Modifier.size(22.dp),
                    tint = MaterialTheme.colorScheme.onSurface,
                )
            }
        } else {
            null
        },
        headlineContent = {
            Text(
                text = title,
                style = MaterialTheme.typography.labelLarge,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        },
        supportingContent = {
            Text(
                text = value,
                style = MaterialTheme.typography.bodyMedium.copy(
                    fontFamily = if (monospace) FontFamily.Monospace else FontFamily.Default,
                ),
            )
        },
        trailingContent = if (onClick != null) {
            {
                Icon(
                    imageVector = Icons.Default.OpenInNew,
                    contentDescription = stringResource(R.string.about_open_github),
                    modifier = Modifier.size(18.dp),
                )
            }
        } else {
            null
        },
    )
}

private fun openGithub(context: Context, url: String) {
    runCatching {
        context.startActivity(
            Intent(Intent.ACTION_VIEW, Uri.parse(url)).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK),
        )
    }
}
