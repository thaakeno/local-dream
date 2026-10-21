package io.github.xororz.localdream.ui.glass

import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ColumnScope
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Card
import androidx.compose.material3.CardColors
import androidx.compose.material3.CardDefaults
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp

/** Preserve Classic exactly; use real refractive surfaces for settings in Liquid. */
@Composable
fun GlassSettingsCard(
    modifier: Modifier = Modifier,
    colors: CardColors = CardDefaults.cardColors(),
    content: @Composable ColumnScope.() -> Unit,
) {
    if (LocalDreamGlassEnabled.current) {
        DreamGlassSurface(
            modifier = modifier,
            shape = RoundedCornerShape(28.dp),
            role = DreamGlassRole.CARD,
        ) {
            Column(Modifier.fillMaxWidth(), content = content)
        }
    } else {
        Card(modifier = modifier, colors = colors, content = content)
    }
}
