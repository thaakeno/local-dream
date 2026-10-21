package io.github.xororz.localdream.ui.glass

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.animateContentSize
import androidx.compose.animation.core.spring
import androidx.compose.animation.expandHorizontally
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.shrinkHorizontally
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.outlined.Add
import androidx.compose.material.icons.outlined.AutoAwesome
import androidx.compose.material.icons.outlined.History
import androidx.compose.material.icons.outlined.Settings
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp

/** One live glass boundary morphs from a small bubble into inline quick tools. */
@Composable
fun DreamCreateBubble(onAddModel: () -> Unit, onHistory: () -> Unit, onSettings: () -> Unit) {
    var expanded by rememberSaveable { mutableStateOf(false) }
    DreamGlassSurface(
        modifier = Modifier.animateContentSize(spring(dampingRatio = .74f, stiffness = 400f)),
        shape = RoundedCornerShape(28.dp),
        role = DreamGlassRole.BUBBLE,
    ) {
        Row(
            Modifier.height(56.dp).padding(horizontal = 7.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            AnimatedVisibility(
                visible = expanded,
                enter = expandHorizontally(spring()) + fadeIn(),
                exit = shrinkHorizontally(spring()) + fadeOut(),
            ) {
                Row {
                    IconButton(onClick = { expanded = false; onAddModel() }) {
                        Icon(Icons.Outlined.Add, "Add model")
                    }
                    IconButton(onClick = { expanded = false; onHistory() }) {
                        Icon(Icons.Outlined.History, "History")
                    }
                    IconButton(onClick = { expanded = false; onSettings() }) {
                        Icon(Icons.Outlined.Settings, "Settings")
                    }
                }
            }
            IconButton(onClick = { expanded = !expanded }, modifier = Modifier.size(44.dp)) {
                Icon(Icons.Outlined.AutoAwesome, if (expanded) "Collapse actions" else "Open quick actions")
            }
        }
    }
}
