package io.github.xororz.localdream.ui.screens

import android.content.Intent
import android.widget.Toast
import androidx.core.content.FileProvider
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.AutoAwesome
import androidx.compose.material.icons.filled.GraphicEq
import androidx.compose.material.icons.filled.History
import androidx.compose.material.icons.filled.Favorite
import androidx.compose.material.icons.filled.FilterList
import androidx.compose.material.icons.filled.Lock
import androidx.compose.material.icons.filled.LockOpen
import androidx.compose.material.icons.filled.Search
import androidx.compose.material.icons.filled.Sort
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.Tune
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.drawBehind
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.text.TextLayoutResult
import androidx.compose.ui.text.TextRange
import androidx.compose.ui.text.input.TextFieldValue
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.window.Dialog
import androidx.compose.ui.window.DialogProperties
import androidx.navigation.NavController
import io.github.xororz.localdream.data.ModelRepository
import io.github.xororz.localdream.navigation.popBackStackIfResumed
import io.github.xororz.localdream.service.BreezeQnnGeneratorArtifact
import io.github.xororz.localdream.service.BreezeQnnVocoderArtifact
import io.github.xororz.localdream.service.SpeechGenerationService
import io.github.xororz.localdream.service.SpeechGenerationService.SpeechState
import io.github.xororz.localdream.service.SpeechHistoryItem
import io.github.xororz.localdream.service.SpeechHistoryStore
import io.github.xororz.localdream.ui.components.MusicPlayerCard
import io.github.xororz.localdream.ui.components.SmoothIndeterminateLinearWavyProgressIndicator
import io.github.xororz.localdream.ui.components.SmoothLinearWavyProgressIndicator
import io.github.xororz.localdream.utils.AppHaptics
import java.io.File
import java.text.DateFormat
import java.util.Date
import java.util.Locale
import kotlin.math.roundToInt
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

private data class SpeechTemplate(
    val name: String,
    val category: String,
    val text: String,
    val instruction: String,
)

private data class InlineEventCue(
    val tag: String,
    val label: String,
    val official: Boolean,
    val language: String,
)

private enum class SpeechHistorySort(val label: String) {
    Newest("Newest"),
    Oldest("Oldest"),
    Fastest("Fastest"),
    Longest("Longest"),
}

private val BREEZE_INLINE_EVENTS = listOf(
    InlineEventCue("(laugh)", "Laugh", true, "English"),
    InlineEventCue("(sigh)", "Sigh", true, "English"),
    InlineEventCue("(cough)", "Cough", true, "English"),
    InlineEventCue("(clears throat)", "Clear throat", true, "English"),
    InlineEventCue("[笑]", "笑", true, "中文"),
    InlineEventCue("[叹气]", "叹气", true, "中文"),
    InlineEventCue("[咳嗽]", "咳嗽", true, "中文"),
    InlineEventCue("[清嗓子]", "清嗓子", true, "中文"),
    // Breeze's event vocabulary is open-ended in the cpp runtime. These are
    // useful extended cues, but deliberately kept visually separate from the
    // official BreezeBlue examples above.
    InlineEventCue("(laughs)", "Laughs", false, "Extended"),
    InlineEventCue("(chuckles)", "Chuckles", false, "Extended"),
    InlineEventCue("(giggles)", "Giggles", false, "Extended"),
    InlineEventCue("(crying)", "Crying", false, "Extended"),
    InlineEventCue("(sobs)", "Sobs", false, "Extended"),
    InlineEventCue("(whimpers)", "Whimpers", false, "Extended"),
    InlineEventCue("(groans)", "Groans", false, "Extended"),
    InlineEventCue("(moans)", "Moans", false, "Extended"),
    InlineEventCue("(sighs)", "Sighs", false, "Extended"),
    InlineEventCue("(gasps)", "Gasps", false, "Extended"),
    InlineEventCue("(inhales)", "Inhales", false, "Extended"),
    InlineEventCue("(exhales)", "Exhales", false, "Extended"),
    InlineEventCue("(breathing heavily)", "Heavy breath", false, "Extended"),
    InlineEventCue("(whispers)", "Whispers", false, "Extended"),
    InlineEventCue("(shouts)", "Shouts", false, "Extended"),
    InlineEventCue("(screams)", "Screams", false, "Extended"),
    InlineEventCue("(singing)", "Singing", false, "Extended"),
    InlineEventCue("(humming)", "Humming", false, "Extended"),
    InlineEventCue("(stutters)", "Stutters", false, "Extended"),
    InlineEventCue("(pause)", "Pause", false, "Extended"),
    InlineEventCue("(coughs)", "Coughs", false, "Extended"),
    InlineEventCue("(sniffs)", "Sniffs", false, "Extended"),
    InlineEventCue("(smacks lips)", "Lip smack", false, "Extended"),
    InlineEventCue("(clicks tongue)", "Tongue click", false, "Extended"),
    InlineEventCue("(yawns)", "Yawns", false, "Extended"),
    InlineEventCue("(sneezes)", "Sneezes", false, "Extended"),
    InlineEventCue("(hiccups)", "Hiccups", false, "Extended"),
    InlineEventCue("(burps)", "Burps", false, "Extended"),
    InlineEventCue("(gulps)", "Gulps", false, "Extended"),
    InlineEventCue("(gags)", "Gags", false, "Extended"),
    InlineEventCue("(grunts)", "Grunts", false, "Extended"),
    InlineEventCue("(scoffs)", "Scoffs", false, "Extended"),
    InlineEventCue("(snorts)", "Snorts", false, "Extended"),
)

private val BREEZE_TEMPLATES = listOf(
    SpeechTemplate(
        "Late-night voice memo",
        "Realistic",
        "(sigh) Okay... I probably should have called earlier. The train stalled outside the station, my phone was nearly dead, and by the time I got home I just wanted five minutes of silence.",
        "Young adult male recorded as a casual phone voice memo. Close mic, natural room tone, imperfect pacing, small hesitations and breaths. Conversational and believable, never performed.",
    ),
    SpeechTemplate(
        "Quiet confession",
        "Realistic",
        "I kept rewriting this in my head because every version sounded rehearsed. So... I'll just say it. I miss you, and I don't know what I'm supposed to do with that.",
        "Adult woman speaking privately to someone she trusts. Soft natural voice, hesitant but sincere, small pauses, restrained emotion, close-mic phone recording.",
    ),
    SpeechTemplate(
        "Vanguard One",
        "Cinematic",
        "(gasps) Control, this is Vanguard One. The star is gone. Not dimmed—gone. (shouts) TURN THE SHIP. NOW. (breathing heavily) There's something moving where it used to be.",
        "Battle-worn spacecraft commander with a deep urgent voice. Controlled military delivery breaking under impossible pressure, clipped phrases, fast breathing, sudden commands, radio-like tension.",
    ),
    SpeechTemplate(
        "Interrogation room",
        "Cinematic",
        "You walked in here expecting me to raise my voice. I'm not going to. I'm going to ask you once, very clearly... who opened that door?",
        "Mature woman with cold authority. Low controlled volume, deliberate pauses, precise diction, contained anger and a dangerous calm. Intimate cinematic close mic.",
    ),
    SpeechTemplate(
        "Close whisper",
        "Whisper",
        "(sigh) Keep your voice down. The walls are thinner than they look. Come closer... I'll tell you what actually happened.",
        "Soft adult feminine voice, extremely close and intimate. Slow silky breathy whisper, very low volume, delicate breaths, relaxed pacing, clear articulation, natural ASMR-like proximity.",
    ),
    SpeechTemplate(
        "Sleep story",
        "Whisper",
        "The rain had been falling for hours, soft enough that the city seemed farther away than usual. By midnight, even the traffic had disappeared.",
        "Warm adult narrator in a calm near-whisper. Slow even breathing, gentle pacing, soft consonants, sleepy late-night tone, clean close-mic recording.",
    ),
    SpeechTemplate(
        "Anime heroine",
        "Anime",
        "(giggles) You really came all this way just to prove me wrong? Fine. One round. If I win, you're buying dinner.",
        "Adult anime-inspired feminine voice with bright natural energy, playful confidence and expressive timing. Light, polished and charming without becoming squeaky or exaggerated.",
    ),
    SpeechTemplate(
        "Anime rival",
        "Anime",
        "I don't care how impossible it looks. We trained for this exact moment. So stop staring at the sky and move.",
        "Young adult anime-inspired male voice. Focused, athletic and intense, quick confident delivery, controlled urgency, grounded performance rather than cartoon shouting.",
    ),
    SpeechTemplate(
        "Documentary",
        "Narration",
        "At the edge of the desert, the temperature can fall more than thirty degrees after sunset. For the animals that live here, surviving the night is a second battle.",
        "Mature documentary narrator with a resonant, measured voice. Clear diction, calm authority, natural pauses and a polished broadcast recording.",
    ),
    SpeechTemplate(
        "Presidential address",
        "Narration",
        "My fellow citizens... we are entering difficult days. But this country has faced fear before, and we did not survive by surrendering to it.",
        "Mature presidential voice: calm, authoritative and measured. Clear diction, deliberate pauses, controlled emotion, reassuring warmth and practiced public-speaking confidence.",
    ),
    SpeechTemplate(
        "Betrayed",
        "Intense",
        "(shouts) Don't stand there and lie to me. I gave you every chance to tell me the truth. (sigh) Just... get out. Before I say something I can't take back.",
        "Adult woman who feels personally betrayed. Sharp powerful voice, heavy breathing, sudden volume surges, brief voice cracks, then a cold low finish. Raw but believable.",
    ),
    SpeechTemplate(
        "Natural Mandarin",
        "Mandarin",
        "今天路上有点堵，不过没关系。我们慢慢走，到了以后先找个安静的地方坐一会儿。",
        "自然的普通话成年女声，像朋友之间真实聊天。语速舒适，语气温和，轻微停顿和自然呼吸，不要播音腔。",
    ),
    SpeechTemplate(
        "Missed last train",
        "Realistic",
        "(sigh) It says the next train is in forty-six minutes. Yeah, I'm fine. Just standing here with a coffee that's gone cold. Tell me about your day?",
        "Adult man speaking on a late-night call outdoors. Tired but interested, conversational warmth, slight breathiness and irregular pauses, no artificial drama.",
    ),
    SpeechTemplate(
        "Unexpected good news",
        "Realistic",
        "Wait. Read that again. (laugh) No, seriously, are you sure? I didn't even think they'd reply. Give me a second... I'm actually shaking.",
        "Young adult woman getting wonderful news in private. Starts disbelieving, breaks into an involuntary laugh, then quiet overwhelming relief, unpolished authentic delivery.",
    ),
    SpeechTemplate(
        "The last customer",
        "Realistic",
        "Hey, sorry, we're about to close. (sigh) You know what? Take your time. It looks like you've had a worse day than me.",
        "Middle-aged café worker with a low slightly rough voice. Starts drained and brisk, then unexpectedly softens. Realistic background-quiet conversation.",
    ),
    SpeechTemplate(
        "Hospital hallway",
        "Realistic",
        "She's awake. The doctor just came out and said she's awake. (breathing heavily) I don't know what to do with my hands right now. Can you come?",
        "Adult man making a phone call after hours of worry. Trembling relief, a little breathless and exhausted, sentences arriving in uneven bursts rather than an actor's monologue.",
    ),
    SpeechTemplate(
        "The lighthouse signal",
        "Cinematic",
        "(static) Coast Guard, do you copy? We have a light on the water. It's been following the ship for forty minutes. There is no boat underneath it.",
        "Older coastal radio operator, restrained low register with growing uncertainty. Concrete, precise delivery and tense silences; cinematic but believable.",
    ),
    SpeechTemplate(
        "The king's surrender",
        "Cinematic",
        "Take the crown. Keep the city. But you will not have another soldier from me. I have buried enough sons for one lifetime.",
        "Exhausted aging monarch with a deep weathered voice. Long measured phrases, grief held behind authority, a final line spoken quietly rather than shouted.",
    ),
    SpeechTemplate(
        "Emergency broadcast",
        "Cinematic",
        "This is not a test. Stay away from the windows. Do not respond if someone outside calls you by name. (breathing heavily) Please... just trust me.",
        "Female emergency announcer, professionally controlled at first and privately frightened by the final sentence. Crisp broadcast enunciation, clipped urgency.",
    ),
    SpeechTemplate(
        "Last message from orbit",
        "Cinematic",
        "The Earth is coming over the horizon now. It's beautiful. (sigh) My oxygen alarm's been ringing for six minutes. I think I'll watch the sunrise.",
        "Adult astronaut speaking a final transmission with quiet acceptance. Close mic, warm and slightly strained breath, intimate pauses, no melodramatic sobbing.",
    ),
    SpeechTemplate(
        "Library after midnight",
        "Whisper",
        "(whispers) That book wasn't on the shelf yesterday. Look at the date inside. It's tomorrow. Don't turn the next page until I'm back.",
        "Soft adult woman's whisper in a deserted library. Very close, steady and clear, little restrained breaths, gentle conspiratorial urgency.",
    ),
    SpeechTemplate(
        "Rain on the windowsill",
        "Whisper",
        "Listen... the rain's getting softer. (inhales) You don't have to solve anything tonight. Just stay here a little longer.",
        "Low soothing adult voice with true quiet breathy whisper, intimate natural pacing and soft consonants. Comforting rather than exaggerated ASMR.",
    ),
    SpeechTemplate(
        "The secret recording",
        "Whisper",
        "(whispers) Okay. If you're hearing this, the door finally opened. I left the key where we used to hide our letters. You remember.",
        "Older feminine voice recorded into a small handheld recorder. Gentle whisper, fragile nostalgia and quiet urgency, no theatrical sound effects.",
    ),
    SpeechTemplate(
        "One last bedtime tale",
        "Whisper",
        "A fox came to the edge of the garden every evening. He never asked for food. He just sat with the old man until the lights went out.",
        "Deep, cozy storyteller speaking in a near-whisper. Soft gravelly texture, slow reassuring rhythm, natural pauses and gentle articulation.",
    ),
    SpeechTemplate(
        "The retired swordsman",
        "Anime",
        "(sigh) I spent twenty years trying to perfect that technique. You figured it out in a week. (laugh) Well... I suppose that's why I trained you.",
        "Older anime mentor with dry humor, soft gravelly authority and affectionate pride. Small understated laugh, natural timing instead of overacting.",
    ),
    SpeechTemplate(
        "Last round of the tournament",
        "Anime",
        "You can keep the trophy. I didn't come here to win one. I came here to see if I could finally beat you.",
        "Young adult female rival, cool controlled intensity and effortless confidence. Smooth, deliberate delivery with a small smile in the final line.",
    ),
    SpeechTemplate(
        "The quiet villain",
        "Anime",
        "How strange. You've won every battle, and you still look terrified. Tell me... what do you think happens when there's nothing left to fight?",
        "Androgynous mature antagonist, unusually serene and intelligent. Silky low resonance, precise teasing pauses, unsettling calm without shouting.",
    ),
    SpeechTemplate(
        "Captain's final order",
        "Anime",
        "(shouts) Everyone below deck! I'll hold the line. (sigh) And if anyone asks... tell them I was never good at following orders.",
        "Young adult anime captain, urgent projection giving way to an almost amused private farewell. Clear expressive arc, earned emotion, not squeaky.",
    ),
    SpeechTemplate(
        "The black ocean",
        "Narration",
        "More than six thousand meters below the surface, sunlight has never reached the seafloor. Yet here, life has built a city around a crack in the Earth.",
        "Mature nature documentary narrator with warm low timbre. Deep curiosity, measured scientific clarity and spacious pauses, no blockbuster trailer voice.",
    ),
    SpeechTemplate(
        "A forgotten invention",
        "Narration",
        "In 1903, one small workshop produced an engine nobody expected to work. Its inventor kept the first successful test a secret for eleven days.",
        "Thoughtful historical documentarian with crisp detailed articulation, curious restrained excitement, intimate studio microphone.",
    ),
    SpeechTemplate(
        "The museum at dawn",
        "Narration",
        "Every scratch on this bronze mask has a story. Some were made by the artist. Others were left by people who held it centuries after the artist died.",
        "Gentle cultured museum guide speaking to a small group. Clear natural warmth, accessible pacing, subtle reverence without pomp.",
    ),
    SpeechTemplate(
        "The unsolved radio signal",
        "Narration",
        "At 2:17 in the morning, every receiver in the observatory picked up the same three notes. The transmission lasted exactly nine seconds. It was never repeated.",
        "Investigative podcast narrator with low calm authority and deliberate suspense. Conversational precision, subtle curiosity, not a caricature of true crime.",
    ),
    SpeechTemplate(
        "The rescue call",
        "Intense",
        "(breathing heavily) I can see the car. The water's rising. I've got the door open, but the seatbelt won't move. (shouts) I NEED A KNIFE!",
        "Adult first responder in a real emergency. Controlled professional commands collapse briefly into panic, audible exertion, urgent natural breaths.",
    ),
    SpeechTemplate(
        "The broken promise",
        "Intense",
        "Don't tell me you tried. You promised you'd be there when she woke up. (sigh) She waited for you all morning.",
        "Middle-aged woman struggling to keep anger below the surface. Low clipped beginning, emotional crack, final sentence nearly whispered in disappointment.",
    ),
    SpeechTemplate(
        "No one gets left behind",
        "Intense",
        "(shouts) Turn the vehicle around! I don't care what command said. There are still people in that building, and we are not leaving them.",
        "Weathered emergency team leader with forceful grounded authority. Urgent shouting, tight controlled breaths and quick concrete phrasing.",
    ),
    SpeechTemplate(
        "After the verdict",
        "Intense",
        "(laugh) So that's it? All those years, and you get to walk out? (sigh) Look at me. At least have the courage to look at me.",
        "Adult man absorbing an unjust verdict. Shock becomes a brief bitter laugh, then shaking quiet anger. Raw vulnerable delivery, not theatrical rage.",
    ),
    SpeechTemplate(
        "The final warning",
        "Intense",
        "I'm only going to say this once. Put it down. (breathing heavily) I don't want to hurt you. Don't make me choose.",
        "Adult woman speaking under extreme pressure. Low firm voice that almost breaks, careful pauses, breath held between words; serious and human.",
    ),
    SpeechTemplate(
        "深夜来电",
        "Mandarin",
        "(叹气) 你还没睡啊？我刚到家。今天真的有点累，不过听见你的声音，突然就觉得好多了。",
        "自然的成年男性普通话，深夜打电话的轻声语气。略显疲惫，但听到对方回应后明显放松，呼吸和停顿自然，不要播音腔。",
    ),
    SpeechTemplate(
        "地铁里的好消息",
        "Mandarin",
        "等等，你是说真的？(笑) 我拿到录取通知了？天哪，我还以为自己一点机会都没有！",
        "年轻成年女性普通话，收到录取消息的真实反应。从不敢相信到忍不住笑出来，惊喜自然，不要刻意夸张。",
    ),
    SpeechTemplate(
        "雨夜的故事",
        "Mandarin",
        "雨下了一整晚。街上的灯一个接一个熄灭，只有那家小书店，还透着一盏暖黄色的光。",
        "温暖成熟的中文叙述女声，轻柔清晰，像在夜晚给人读故事。慢速但不拖沓，留有自然的句间停顿。",
    ),
    SpeechTemplate(
        "最后的讯号",
        "Mandarin",
        "(喘息) 指挥中心，听得到吗？我们的电量只剩百分之三。外面那个东西……它又回来了。",
        "成年男性中文科幻通讯语气，压低声音的紧张感逐步升级。短促呼吸，最后一句近乎耳语，不要夸张吼叫。",
    ),
    SpeechTemplate(
        "老朋友重逢",
        "Mandarin",
        "你还是老样子啊。嘴上说着不来，结果比谁都到得早。(笑) 坐吧，我有好多事想跟你聊。",
        "中年女性自然普通话，老友重逢时带点打趣和怀念，轻松微笑的语气，亲切自然，有生活感。",
    ),
)

private const val DEFAULT_SEED = 42L
private const val DEFAULT_CFG = 1f
private const val DEFAULT_TEMPERATURE = 0.9f
private const val DEFAULT_TOP_K = 50
private const val DEFAULT_TOP_P = 1f
private const val DEFAULT_REPETITION = 1.1f
private const val DEFAULT_SPLIT_CHARS = 600
private const val DEFAULT_MAX_NEW_TOKENS = 750

private fun freshSpeechSeed(): Long =
    ((System.nanoTime() xor System.currentTimeMillis()) and 0x7fffffffL).coerceAtLeast(1L)

private fun shareSpeechFile(context: android.content.Context, file: File) {
    if (!file.isFile) return
    val uri = FileProvider.getUriForFile(
        context,
        context.packageName + ".fileprovider",
        file,
    )
    val send = Intent(Intent.ACTION_SEND).apply {
        type = "audio/wav"
        putExtra(Intent.EXTRA_STREAM, uri)
        addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        clipData = android.content.ClipData.newRawUri("Breeze TTS audio", uri)
    }
    context.startActivity(Intent.createChooser(send, "Share generated speech"))
}

private fun startSpeechGeneration(
    context: android.content.Context,
    modelId: String,
    text: String,
    instruction: String,
    seed: Long,
    cfg: Float,
    temperature: Float,
    topK: Int,
    topP: Float,
    repetition: Float,
    splitChars: Int,
    maxNewTokens: Int,
) {
    context.startForegroundService(
        Intent(context, SpeechGenerationService::class.java)
            .setAction(SpeechGenerationService.ACTION_GENERATE)
            .putExtra("modelId", modelId)
            .putExtra("text", text)
            .putExtra("instruction", instruction)
            .putExtra("seed", seed)
            .putExtra("cfg", cfg)
            .putExtra("temperature", temperature)
            .putExtra("topK", topK)
            .putExtra("topP", topP)
            .putExtra("repetition", repetition)
            .putExtra("splitChars", splitChars)
            .putExtra("maxNewTokens", maxNewTokens),
    )
}

private fun breezeHistoryModelLabel(modelId: String): String = when (modelId) {
    "breeze_tts2_q8" -> "Q8"
    "breeze_tts2_q6" -> "Q6"
    "breeze_tts2_q4" -> "Q4"
    "breeze_tts2_f16" -> "F16"
    "breeze_tts2_q8_dd4" -> "Q8 · DD4"
    "breeze_tts2_q8_dd2" -> "Q8 · DD2"
    "breeze_tts2_q4_dd2" -> "Q4 · DD2"
    else -> modelId.removePrefix("breeze_tts2_").replace('_', ' ').uppercase(Locale.US)
}

private fun breezeHistoryModelOrder(modelId: String): Int = when (modelId) {
    "breeze_tts2_q8" -> 0
    "breeze_tts2_q8_dd4" -> 1
    "breeze_tts2_q8_dd2" -> 2
    "breeze_tts2_q6" -> 3
    "breeze_tts2_q4" -> 4
    "breeze_tts2_q4_dd2" -> 5
    "breeze_tts2_f16" -> 6
    else -> 99
}

private fun formatHistoryDate(timestamp: Long): String =
    if (timestamp <= 0L) "Earlier generation"
    else DateFormat.getDateTimeInstance(
        DateFormat.MEDIUM,
        DateFormat.SHORT,
    ).format(Date(timestamp))

private fun formatMillisCompact(ms: Long): String {
    if (ms <= 0L) return "--"
    return if (ms < 10_000L) {
        String.format(Locale.US, "%.1fs", ms / 1000.0)
    } else {
        val total = (ms / 1000L).toInt()
        String.format(Locale.US, "%d:%02d", total / 60, total % 60)
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SpeechRunScreen(
    modelId: String,
    navController: NavController,
) {
    val context = androidx.compose.ui.platform.LocalContext.current
    val scope = rememberCoroutineScope()
    val repository = remember { ModelRepository.getInstance(context) }
    LaunchedEffect(Unit) { repository.ensureLoaded() }
    val model = repository.models.firstOrNull { it.id == modelId }
    val speechState by SpeechGenerationService.state.collectAsState()
    val acceleratorState by BreezeQnnVocoderArtifact.status.collectAsState()
    val generatorState by BreezeQnnGeneratorArtifact.status.collectAsState()
    var fullQnnGeneratorEnabled by remember {
        mutableStateOf(BreezeQnnGeneratorArtifact.isEnabled(context))
    }

    // This screen owns the warm Breeze process. Leaving the generation panel
    // must release the GGUF, QNN contexts and HTP power votes immediately.
    DisposableEffect(modelId) {
        onDispose {
            runCatching {
                context.startService(
                    Intent(context, SpeechGenerationService::class.java)
                        .setAction(SpeechGenerationService.ACTION_STOP),
                )
            }
        }
    }
    var acceleratorPromptDismissed by remember { mutableStateOf(false) }
    var acceleratorDownloadRequested by remember { mutableStateOf(false) }

    var text by rememberSaveable {
        mutableStateOf("Welcome aboard. Your journey begins now.")
    }
    var instruction by rememberSaveable {
        mutableStateOf("A warm, thoughtful young woman with a clear, calm delivery.")
    }
    var seed by rememberSaveable { mutableLongStateOf(DEFAULT_SEED) }
    var seedLocked by rememberSaveable { mutableStateOf(true) }
    var cfg by rememberSaveable { mutableFloatStateOf(DEFAULT_CFG) }
    var temperature by rememberSaveable { mutableFloatStateOf(DEFAULT_TEMPERATURE) }
    var topK by rememberSaveable { mutableIntStateOf(DEFAULT_TOP_K) }
    var topP by rememberSaveable { mutableFloatStateOf(DEFAULT_TOP_P) }
    var repetition by rememberSaveable { mutableFloatStateOf(DEFAULT_REPETITION) }
    var splitChars by rememberSaveable { mutableIntStateOf(DEFAULT_SPLIT_CHARS) }
    var maxNewTokens by rememberSaveable { mutableIntStateOf(DEFAULT_MAX_NEW_TOKENS) }
    var showTune by remember { mutableStateOf(false) }
    var showAdvancedTune by rememberSaveable { mutableStateOf(false) }
    var showHistory by remember { mutableStateOf(false) }
    var history by remember { mutableStateOf<List<SpeechHistoryItem>>(emptyList()) }
    var templateCategory by rememberSaveable { mutableStateOf("Realistic") }
    var selectedVoiceScene by rememberSaveable { mutableStateOf<String?>(null) }
    var historySearch by rememberSaveable { mutableStateOf("") }
    var historySort by rememberSaveable { mutableStateOf(SpeechHistorySort.Newest.name) }
    var historyFavoritesOnly by rememberSaveable { mutableStateOf(false) }
    var historyAcceleratedOnly by rememberSaveable { mutableStateOf(false) }
    var historyModelIds by remember { mutableStateOf<Set<String>>(emptySet()) }
    var showHistoryFilters by remember { mutableStateOf(false) }

    LaunchedEffect(modelId, model?.isDownloaded) {
        BreezeQnnVocoderArtifact.refresh(context)
        BreezeQnnGeneratorArtifact.refresh(context)
        if (model?.isDownloaded == true && model.isVoice) {
            val supported = BreezeQnnVocoderArtifact.supportedSoc() != null
            val acceleratorReady = BreezeQnnVocoderArtifact.localFile(context) != null
            if (!supported || acceleratorReady) {
                SpeechGenerationService.resetForModel(modelId)
                context.startForegroundService(
                    Intent(context, SpeechGenerationService::class.java)
                        .setAction(SpeechGenerationService.ACTION_PRELOAD)
                        .putExtra("modelId", modelId),
                )
            }
        }
        history = withContext(Dispatchers.IO) { SpeechHistoryStore.load(context) }
    }

    LaunchedEffect(acceleratorState) {
        if (
            acceleratorDownloadRequested &&
            acceleratorState is BreezeQnnVocoderArtifact.Status.Ready
        ) {
            acceleratorDownloadRequested = false
            context.startService(
                Intent(context, SpeechGenerationService::class.java)
                    .setAction(SpeechGenerationService.ACTION_STOP),
            )
            delay(180)
            SpeechGenerationService.resetForModel(modelId)
            context.startForegroundService(
                Intent(context, SpeechGenerationService::class.java)
                    .setAction(SpeechGenerationService.ACTION_PRELOAD)
                    .putExtra("modelId", modelId),
            )
        }
    }

    var lastSpeechProgress by remember { mutableFloatStateOf(-1f) }
    var lastTerminalHaptic by remember { mutableStateOf<String?>(null) }

    LaunchedEffect(speechState) {
        when (val state = speechState) {
            is SpeechState.Generating -> {
                state.progress?.let { progress ->
                    if (progress > lastSpeechProgress + 0.001f) {
                        AppHaptics.perform(context, AppHaptics.Kind.Progress)
                        lastSpeechProgress = progress
                    }
                }
                lastTerminalHaptic = null
            }
            is SpeechState.Complete -> {
                history = withContext(Dispatchers.IO) { SpeechHistoryStore.load(context) }
                if (lastTerminalHaptic != "complete") {
                    AppHaptics.perform(context, AppHaptics.Kind.Success)
                    lastTerminalHaptic = "complete"
                }
                lastSpeechProgress = -1f
            }
            is SpeechState.Error -> {
                if (lastTerminalHaptic != "error") {
                    AppHaptics.perform(context, AppHaptics.Kind.Failure)
                    lastTerminalHaptic = "error"
                }
                lastSpeechProgress = -1f
            }
            SpeechState.Idle, is SpeechState.Ready, is SpeechState.Loading -> {
                lastSpeechProgress = -1f
            }
        }
    }

    val busy = speechState is SpeechState.Loading || speechState is SpeechState.Generating
    val precision = model?.variantPrecision.orEmpty()
    val statusText = when (val state = speechState) {
        is SpeechState.Loading ->
            state.detail + " · " + (state.progress.coerceIn(0f, 1f) * 100f).roundToInt() + "%"
        is SpeechState.Ready -> "Ready"
        is SpeechState.Generating -> state.progress?.let {
            "Generating ${(it.coerceIn(0f, 1f) * 100f).roundToInt()}%"
        } ?: "Starting generation"
        is SpeechState.Complete -> "Ready"
        is SpeechState.Error -> "Generation failed"
        SpeechState.Idle -> if (model?.isDownloaded == true) "Ready" else "Model required"
    }

    Scaffold(
        topBar = {
            MediumTopAppBar(
                title = {
                    Column {
                        Text("Breeze TTS 2 · Text to speech")
                        Text(
                            statusText,
                            style = MaterialTheme.typography.bodySmall,
                            color = if (speechState is SpeechState.Error) {
                                MaterialTheme.colorScheme.error
                            } else {
                                MaterialTheme.colorScheme.onSurfaceVariant
                            },
                        )
                    }
                },
                navigationIcon = {
                    IconButton(onClick = { navController.popBackStackIfResumed() }) {
                        Icon(Icons.AutoMirrored.Filled.ArrowBack, contentDescription = "Back")
                    }
                },
                actions = {
                    IconButton(
                        onClick = {
                            AppHaptics.perform(context, AppHaptics.Kind.Interaction)
                            showHistory = true
                            scope.launch {
                                history = withContext(Dispatchers.IO) {
                                    SpeechHistoryStore.load(context)
                                }
                            }
                        },
                    ) {
                        Icon(Icons.Default.History, contentDescription = "Speech history")
                    }
                    IconButton(onClick = { showTune = true }) {
                        Icon(Icons.Default.Tune, contentDescription = "Generation settings")
                    }
                },
            )
        },
    ) { padding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
                .verticalScroll(rememberScrollState()),
            verticalArrangement = Arrangement.spacedBy(14.dp),
        ) {
            ElevatedCard(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 16.dp),
                shape = MaterialTheme.shapes.extraLarge,
                colors = CardDefaults.elevatedCardColors(
                    containerColor = MaterialTheme.colorScheme.surfaceContainer,
                ),
            ) {
                Column(
                    modifier = Modifier.padding(18.dp),
                    verticalArrangement = Arrangement.spacedBy(14.dp),
                ) {
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(12.dp),
                    ) {
                        Surface(
                            shape = MaterialTheme.shapes.extraLarge,
                            color = MaterialTheme.colorScheme.primaryContainer,
                        ) {
                            Icon(
                                Icons.Default.GraphicEq,
                                contentDescription = null,
                                modifier = Modifier.padding(12.dp),
                                tint = MaterialTheme.colorScheme.onPrimaryContainer,
                            )
                        }
                        Column(modifier = Modifier.weight(1f)) {
                            Text(
                                "Design a voice",
                                style = MaterialTheme.typography.titleLarge,
                                fontWeight = FontWeight.SemiBold,
                            )
                            Text(
                                "Text plus voice direction. Voice cloning is intentionally disabled for now.",
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant,
                            )
                        }
                    }

                    // A lightweight, curated scene library. No imagery, heavy
                    // animations or extra inference settings on the compose path.
                    val visibleScenes = remember(templateCategory) {
                        BREEZE_TEMPLATES.filter { it.category == templateCategory }
                    }
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.SpaceBetween,
                    ) {
                        Column(verticalArrangement = Arrangement.spacedBy(2.dp)) {
                            Text(
                                "Explore voices",
                                style = MaterialTheme.typography.titleMedium,
                                fontWeight = FontWeight.SemiBold,
                            )
                            Text(
                                "Pick a scene, then make it yours.",
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant,
                            )
                        }
                        Text(
                            "${BREEZE_TEMPLATES.size} ideas",
                            style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                    }
                    LazyRow(
                        horizontalArrangement = Arrangement.spacedBy(8.dp),
                        contentPadding = PaddingValues(end = 8.dp),
                    ) {
                        items(BREEZE_TEMPLATES.map { it.category }.distinct()) { category ->
                            FilterChip(
                                selected = templateCategory == category,
                                onClick = {
                                    templateCategory = category
                                    AppHaptics.perform(context, AppHaptics.Kind.Interaction)
                                },
                                label = { Text(category) },
                                shape = RoundedCornerShape(16.dp),
                            )
                        }
                    }
                    LazyRow(
                        horizontalArrangement = Arrangement.spacedBy(10.dp),
                        contentPadding = PaddingValues(end = 8.dp),
                    ) {
                        items(visibleScenes, key = { it.name }) { scene ->
                            val selected = selectedVoiceScene == scene.name
                            Surface(
                                onClick = {
                                    text = scene.text
                                    instruction = scene.instruction
                                    selectedVoiceScene = scene.name
                                    AppHaptics.perform(context, AppHaptics.Kind.Interaction)
                                },
                                modifier = Modifier
                                    .width(224.dp)
                                    .height(150.dp),
                                shape = RoundedCornerShape(22.dp),
                                color = if (selected) {
                                    MaterialTheme.colorScheme.primaryContainer.copy(alpha = 0.58f)
                                } else {
                                    MaterialTheme.colorScheme.surfaceContainerHigh
                                },
                                border = androidx.compose.foundation.BorderStroke(
                                    1.dp,
                                    if (selected) MaterialTheme.colorScheme.primary
                                    else MaterialTheme.colorScheme.outlineVariant.copy(alpha = 0.42f),
                                ),
                            ) {
                                Column(
                                    modifier = Modifier.padding(14.dp),
                                    verticalArrangement = Arrangement.spacedBy(7.dp),
                                ) {
                                    Row(
                                        verticalAlignment = Alignment.CenterVertically,
                                        horizontalArrangement = Arrangement.SpaceBetween,
                                        modifier = Modifier.fillMaxWidth(),
                                    ) {
                                        Text(
                                            if (selected) "LOADED" else "VOICE SCENE",
                                            style = MaterialTheme.typography.labelSmall,
                                            color = if (selected) MaterialTheme.colorScheme.primary
                                            else MaterialTheme.colorScheme.onSurfaceVariant,
                                            fontWeight = FontWeight.Medium,
                                        )
                                        Icon(
                                            if (selected) Icons.Default.CheckCircle
                                            else Icons.Default.ArrowOutward,
                                            contentDescription = null,
                                            modifier = Modifier.size(17.dp),
                                            tint = if (selected) MaterialTheme.colorScheme.primary
                                            else MaterialTheme.colorScheme.onSurfaceVariant,
                                        )
                                    }
                                    Text(
                                        scene.name,
                                        style = MaterialTheme.typography.titleSmall,
                                        fontWeight = FontWeight.SemiBold,
                                        maxLines = 1,
                                        overflow = androidx.compose.ui.text.style.TextOverflow.Ellipsis,
                                    )
                                    Text(
                                        scene.text.replace(Regex("^\\([^)]*\\)\\s*"), "")
                                            .trim(),
                                        style = MaterialTheme.typography.bodySmall,
                                        maxLines = 3,
                                        overflow = androidx.compose.ui.text.style.TextOverflow.Ellipsis,
                                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                                    )
                                }
                            }
                        }
                    }

                    InlineEventEditor(
                        value = text,
                        onValueChange = { text = it.take(12000); selectedVoiceScene = null },
                        events = BREEZE_INLINE_EVENTS,
                        modifier = Modifier.fillMaxWidth(),
                    )

                    OutlinedTextField(
                        value = instruction,
                        onValueChange = { instruction = it.take(1200); selectedVoiceScene = null },
                        modifier = Modifier.fillMaxWidth(),
                        label = { Text("Voice direction") },
                        placeholder = {
                            Text("Voice, age, tone, pacing, emotion, recording style…")
                        },
                        minLines = 3,
                    )

                    when (val state = speechState) {
                        is SpeechState.Loading -> SpeechLoadingCard(state)
                        is SpeechState.Generating -> SpeechProgressCard(state)
                        is SpeechState.Error -> {
                            Surface(
                                shape = MaterialTheme.shapes.extraLarge,
                                color = MaterialTheme.colorScheme.errorContainer,
                            ) {
                                Column(
                                    modifier = Modifier.padding(16.dp),
                                    verticalArrangement = Arrangement.spacedBy(10.dp),
                                ) {
                                    Text(
                                        "Generation failed",
                                        style = MaterialTheme.typography.titleMedium,
                                        fontWeight = FontWeight.SemiBold,
                                    )
                                    Text(state.message, style = MaterialTheme.typography.bodySmall)
                                    OutlinedButton(
                                        onClick = {
                                            context.startForegroundService(
                                                Intent(
                                                    context,
                                                    SpeechGenerationService::class.java,
                                                )
                                                    .setAction(
                                                        SpeechGenerationService.ACTION_PRELOAD,
                                                    )
                                                    .putExtra("modelId", modelId),
                                            )
                                        },
                                    ) {
                                        Icon(Icons.Default.Refresh, contentDescription = null)
                                        Spacer(Modifier.width(6.dp))
                                        Text("Restart")
                                    }
                                }
                            }
                        }
                        else -> Unit
                    }

                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(10.dp),
                    ) {
                        OutlinedButton(
                            onClick = {
                                if (speechState is SpeechState.Generating) {
                                    context.startService(
                                        Intent(context, SpeechGenerationService::class.java)
                                            .setAction(SpeechGenerationService.ACTION_STOP),
                                    )
                                } else {
                                    showTune = true
                                }
                            },
                            modifier = Modifier.weight(1f),
                        ) {
                            Icon(Icons.Default.Tune, contentDescription = null)
                            Spacer(Modifier.width(6.dp))
                            Text(if (speechState is SpeechState.Generating) "Stop" else "Tune")
                        }

                        Button(
                            onClick = {
                                AppHaptics.perform(context, AppHaptics.Kind.Interaction)
                                val generationSeed = if (seedLocked) {
                                    seed
                                } else {
                                    freshSpeechSeed().also { seed = it }
                                }
                                startSpeechGeneration(
                                    context, modelId, text, instruction, generationSeed,
                                    cfg, temperature, topK, topP, repetition,
                                    splitChars, maxNewTokens,
                                )
                            },
                            enabled = !busy && text.isNotBlank() && model?.isDownloaded == true,
                            modifier = Modifier.weight(1f),
                        ) {
                            Icon(Icons.Default.GraphicEq, contentDescription = null)
                            Spacer(Modifier.width(6.dp))
                            Text("Generate")
                        }
                    }
                }
            }

            val complete = speechState as? SpeechState.Complete
            if (complete != null && complete.file.isFile) {
                MusicPlayerCard(
                    file = complete.file,
                    title = "Breeze TTS 2 · " + precision,
                    subtitle = "Seed " + complete.seed + " · " +
                        String.format(Locale.US, "%.1f s generation", complete.elapsedMillis / 1000f),
                    metadataLine = if (
                        acceleratorState is BreezeQnnVocoderArtifact.Status.Ready ||
                        acceleratorState is BreezeQnnVocoderArtifact.Status.UpgradeAvailable
                    ) {
                        "On-device · Accelerated waveform"
                    } else {
                        "On-device · Local synthesis"
                    },
                    modifier = Modifier.padding(horizontal = 16.dp),
                    onReproduce = {
                        text = complete.text
                        instruction = complete.instruction
                        seed = complete.seed
                        seedLocked = true
                        startSpeechGeneration(
                            context, modelId, complete.text, complete.instruction,
                            complete.seed, cfg, temperature, topK, topP,
                            repetition, splitChars, maxNewTokens,
                        )
                    },
                    onSave = {
                        scope.launch {
                            val saved = runCatching {
                                SpeechHistoryStore.exportToMusic(context, complete.file)
                            }.getOrNull()
                            Toast.makeText(
                                context,
                                if (saved != null) {
                                    "Saved to Music/LocalDream"
                                } else {
                                    "Could not save audio"
                                },
                                Toast.LENGTH_SHORT,
                            ).show()
                        }
                    },
                    onShare = { shareSpeechFile(context, complete.file) },
                )
            }

            Spacer(Modifier.height(4.dp))
        }
    }

    val accelerator = acceleratorState
    val showAcceleratorDialog =
        !acceleratorPromptDismissed &&
        model?.isDownloaded == true &&
        accelerator !is BreezeQnnVocoderArtifact.Status.Ready &&
        accelerator !is BreezeQnnVocoderArtifact.Status.Unsupported &&
        accelerator !is BreezeQnnVocoderArtifact.Status.Checking

    if (showAcceleratorDialog) {
        AlertDialog(
            onDismissRequest = {
                if (accelerator !is BreezeQnnVocoderArtifact.Status.Downloading) {
                    acceleratorPromptDismissed = true
                }
            },
            title = {
                Text(
                    if (accelerator is BreezeQnnVocoderArtifact.Status.UpgradeAvailable) {
                        "Faster waveform decoder"
                    } else {
                        "Waveform decoder"
                    },
                )
            },
            text = {
                Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                    val soc = BreezeQnnVocoderArtifact.supportedSoc().orEmpty()
                    Text(
                        if (accelerator is BreezeQnnVocoderArtifact.Status.UpgradeAvailable) {
                            "Your current v3 accelerator still works. Update to v4 to add native " +
                                "8/32/64-frame QNN graphs, so short first/tail chunks no longer " +
                                "pay for a full 64-frame graph."
                        } else {
                            "Download the optimized waveform decoder. It is shared by every " +
                                "Breeze Q4/Q6/Q8/F16/DD model on this phone and keeps waveform " +
                                "synthesis on the NPU."
                        },
                    )
                    when (accelerator) {
                        is BreezeQnnVocoderArtifact.Status.Downloading -> {
                            accelerator.progress?.let { progress ->
                                LinearProgressIndicator(
                                    progress = { progress },
                                    modifier = Modifier.fillMaxWidth(),
                                )
                            } ?: LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
                            Text(
                                String.format(
                                    Locale.US,
                                    "%.0f / %.0f MB",
                                    accelerator.received / 1048576.0,
                                    accelerator.total / 1048576.0,
                                ),
                                style = MaterialTheme.typography.bodySmall,
                            )
                        }
                        is BreezeQnnVocoderArtifact.Status.Error -> {
                            Text(
                                accelerator.message,
                                color = MaterialTheme.colorScheme.error,
                                style = MaterialTheme.typography.bodySmall,
                            )
                        }
                        else -> Unit
                    }
                }
            },
            confirmButton = {
                if (accelerator !is BreezeQnnVocoderArtifact.Status.Downloading) {
                    TextButton(
                        onClick = {
                            acceleratorDownloadRequested = true
                            scope.launch { BreezeQnnVocoderArtifact.download(context) }
                        },
                    ) {
                        Text(
                            when (accelerator) {
                                is BreezeQnnVocoderArtifact.Status.Error -> "Retry"
                                is BreezeQnnVocoderArtifact.Status.UpgradeAvailable -> "Update"
                                else -> "Download"
                            },
                        )
                    }
                }
            },
            dismissButton = {
                if (accelerator !is BreezeQnnVocoderArtifact.Status.Downloading) {
                    TextButton(
                        onClick = {
                            acceleratorPromptDismissed = true
                            SpeechGenerationService.resetForModel(modelId)
                            context.startForegroundService(
                                Intent(context, SpeechGenerationService::class.java)
                                    .setAction(SpeechGenerationService.ACTION_PRELOAD)
                                    .putExtra("modelId", modelId),
                            )
                        },
                    ) {
                        Text(
                            if (accelerator is BreezeQnnVocoderArtifact.Status.UpgradeAvailable) {
                                "Keep v3"
                            } else {
                                "Use slow fallback"
                            },
                        )
                    }
                }
            },
        )
    }

    if (showTune) {
        ModalBottomSheet(onDismissRequest = { showTune = false }) {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .verticalScroll(rememberScrollState())
                    .padding(horizontal = 18.dp)
                    .padding(bottom = 28.dp),
                verticalArrangement = Arrangement.spacedBy(16.dp),
            ) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(12.dp),
                ) {
                    Column(modifier = Modifier.weight(1f)) {
                        Text(
                            "Generation controls",
                            style = MaterialTheme.typography.headlineSmall,
                            fontWeight = FontWeight.Bold,
                        )
                        Text(
                            "Expression first. Sampling and engine controls stay out of the way until you need them.",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                    }
                    OutlinedButton(
                        onClick = {
                            cfg = DEFAULT_CFG
                            temperature = DEFAULT_TEMPERATURE
                            topK = DEFAULT_TOP_K
                            topP = DEFAULT_TOP_P
                            repetition = DEFAULT_REPETITION
                            splitChars = DEFAULT_SPLIT_CHARS
                            maxNewTokens = DEFAULT_MAX_NEW_TOKENS
                            seed = DEFAULT_SEED
                            seedLocked = true
                            AppHaptics.perform(context, AppHaptics.Kind.Interaction)
                        },
                    ) {
                        Icon(Icons.Default.Refresh, contentDescription = null)
                        Spacer(Modifier.width(6.dp))
                        Text("Reset")
                    }
                }

                Text(
                    "Voice & expression",
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.SemiBold,
                )
                BreezeSettingSlider(
                    label = "Direction strength",
                    value = cfg,
                    range = 1f..4f,
                    description = if (cfg > 1.05f) {
                        "Stronger voice-direction and event adherence. CFG above 1 runs an extra guidance branch, so it costs noticeably more compute."
                    } else {
                        "Natural single-branch generation. Raise it when the voice direction or inline events are being ignored."
                    },
                    onValueChange = { cfg = it },
                ) { String.format(Locale.US, "%.2f", it) }

                BreezeSettingSlider(
                    label = "Expressiveness",
                    value = temperature,
                    range = 0.3f..1.5f,
                    description = "Lower is steadier and more repeatable. Higher allows more variation and emotion, but can also increase odd pronunciations.",
                    onValueChange = { temperature = it },
                ) { String.format(Locale.US, "%.2f", it) }

                Surface(
                    shape = MaterialTheme.shapes.extraLarge,
                    color = MaterialTheme.colorScheme.surfaceContainerHigh,
                ) {
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(16.dp),
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(12.dp),
                    ) {
                        Surface(
                            shape = MaterialTheme.shapes.large,
                            color = MaterialTheme.colorScheme.secondaryContainer,
                        ) {
                            Icon(
                                if (seedLocked) Icons.Default.Lock else Icons.Default.LockOpen,
                                contentDescription = null,
                                modifier = Modifier.padding(10.dp),
                                tint = MaterialTheme.colorScheme.onSecondaryContainer,
                            )
                        }
                        Column(modifier = Modifier.weight(1f)) {
                            Text(
                                if (seedLocked) "Seed locked" else "Seed randomized",
                                fontWeight = FontWeight.SemiBold,
                            )
                            Text(
                                if (seedLocked) {
                                    "Seed $seed will be reused. Useful for exact A/B tests."
                                } else {
                                    "A fresh seed is generated every time you press Generate."
                                },
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant,
                            )
                        }
                        Switch(
                            checked = seedLocked,
                            onCheckedChange = {
                                seedLocked = it
                                if (!it) seed = freshSpeechSeed()
                            },
                        )
                    }
                }
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(10.dp),
                ) {
                    OutlinedButton(
                        onClick = {
                            seed = freshSpeechSeed()
                            seedLocked = true
                        },
                        modifier = Modifier.weight(1f),
                    ) {
                        Text("New locked seed")
                    }
                    OutlinedButton(
                        onClick = {
                            seed = DEFAULT_SEED
                            seedLocked = true
                        },
                        modifier = Modifier.weight(1f),
                    ) {
                        Text("Use 42")
                    }
                }

                HorizontalDivider()

                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(vertical = 2.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Column(modifier = Modifier.weight(1f)) {
                        Text(
                            "Advanced",
                            style = MaterialTheme.typography.titleMedium,
                            fontWeight = FontWeight.SemiBold,
                        )
                        Text(
                            "Sampling, long-text and experimental engine controls.",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                    }
                    Switch(
                        checked = showAdvancedTune,
                        onCheckedChange = { showAdvancedTune = it },
                    )
                }

                if (showAdvancedTune) {
                    BreezeSettingSlider(
                        label = "Top P",
                        value = topP,
                        range = 0.5f..1f,
                        description = "Nucleus sampling. Lower values restrict token choices; 1.0 keeps the full candidate distribution.",
                        onValueChange = { topP = it },
                    ) { String.format(Locale.US, "%.2f", it) }

                    BreezeSettingSlider(
                        label = "Top K",
                        value = topK.toFloat(),
                        range = 10f..100f,
                        description = "Maximum token candidates considered at each sampling step.",
                        onValueChange = { topK = it.roundToInt() },
                    ) { it.roundToInt().toString() }

                    BreezeSettingSlider(
                        label = "Repetition penalty",
                        value = repetition,
                        range = 1f..1.5f,
                        description = "Discourages repeated sounds and phrases. Too high can damage fluency.",
                        onValueChange = { repetition = it },
                    ) { String.format(Locale.US, "%.2f", it) }

                    BreezeSettingSlider(
                        label = "Long-text split",
                        value = splitChars.toFloat(),
                        range = 200f..1200f,
                        description = "Approximate characters per passage. 240 keeps the autoregressive context bounded so long speech does not progressively slow down. Larger passages trade speed for continuity.",
                        onValueChange = { splitChars = it.roundToInt() },
                    ) { it.roundToInt().toString() + " chars" }

                    BreezeSettingSlider(
                        label = "Frame safety cap",
                        value = maxNewTokens.toFloat(),
                        range = 250f..1500f,
                        description = "Maximum acoustic frames Breeze may generate for each segment. This is a safety ceiling, not a quality knob.",
                        onValueChange = { maxNewTokens = it.roundToInt() },
                    ) { it.roundToInt().toString() }

                    Surface(
                        shape = MaterialTheme.shapes.extraLarge,
                        color = MaterialTheme.colorScheme.surfaceContainerHigh,
                    ) {
                        Row(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(16.dp),
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.spacedBy(12.dp),
                        ) {
                            Column(modifier = Modifier.weight(1f)) {
                                Text(
                                    "Experimental Full QNN generator",
                                    fontWeight = FontWeight.SemiBold,
                                )
                                Text(
                                    if (fullQnnGeneratorEnabled) {
                                        "QNN backbone + depth enabled for A/B testing. The QNN waveform decoder is independent."
                                    } else {
                                        "Off. The proven GGUF/ggml-Hexagon generator is being used."
                                    },
                                    style = MaterialTheme.typography.bodySmall,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                                )
                            }
                            Switch(
                                checked = fullQnnGeneratorEnabled,
                                enabled = generatorState is BreezeQnnGeneratorArtifact.Status.Ready,
                                onCheckedChange = { enabled ->
                                    fullQnnGeneratorEnabled = enabled
                                    BreezeQnnGeneratorArtifact.setEnabled(context, enabled)
                                    AppHaptics.perform(context, AppHaptics.Kind.Interaction)
                                    scope.launch {
                                        context.startService(
                                            Intent(context, SpeechGenerationService::class.java)
                                                .setAction(SpeechGenerationService.ACTION_STOP),
                                        )
                                        delay(180)
                                        SpeechGenerationService.resetForModel(modelId)
                                        if (model?.isDownloaded == true) {
                                            context.startForegroundService(
                                                Intent(
                                                    context,
                                                    SpeechGenerationService::class.java,
                                                )
                                                    .setAction(
                                                        SpeechGenerationService.ACTION_PRELOAD,
                                                    )
                                                    .putExtra("modelId", modelId),
                                            )
                                        }
                                    }
                                },
                            )
                        }
                    }
                }

                Button(
                    onClick = { showTune = false },
                    modifier = Modifier.fillMaxWidth(),
                ) {
                    Text("Done")
                }
            }
        }
    }

    if (showHistory) {
        val sortMode = runCatching { SpeechHistorySort.valueOf(historySort) }
            .getOrDefault(SpeechHistorySort.Newest)
        val query = historySearch.trim()
        val modelOptions = history
            .map { it.modelId }
            .distinct()
            .sortedWith(
                compareBy<String> { breezeHistoryModelOrder(it) }
                    .thenBy { breezeHistoryModelLabel(it) },
            )
        val visibleHistory = history
            .asSequence()
            .filter { historyModelIds.isEmpty() || it.modelId in historyModelIds }
            .filter {
                query.isBlank() ||
                    it.text.contains(query, ignoreCase = true) ||
                    it.instruction.contains(query, ignoreCase = true) ||
                    breezeHistoryModelLabel(it.modelId).contains(query, ignoreCase = true)
            }
            .filter { !historyFavoritesOnly || it.favorite }
            .filter { !historyAcceleratedOnly || it.accelerated }
            .toList()
            .let { items ->
                when (sortMode) {
                    SpeechHistorySort.Newest -> items.sortedByDescending { it.createdAt }
                    SpeechHistorySort.Oldest -> items.sortedBy { it.createdAt }
                    SpeechHistorySort.Fastest -> items.sortedBy {
                        if (it.generationMillis > 0L) it.generationMillis else Long.MAX_VALUE
                    }
                    SpeechHistorySort.Longest -> items.sortedByDescending { it.audioDurationMillis }
                }
            }
        val historyListState = rememberLazyListState()
        var sortMenuOpen by remember { mutableStateOf(false) }
        val activeFilterCount =
            historyModelIds.size +
                (if (historyFavoritesOnly) 1 else 0) +
                (if (historyAcceleratedOnly) 1 else 0)

        Dialog(
            onDismissRequest = { showHistory = false },
            properties = DialogProperties(
                usePlatformDefaultWidth = false,
                decorFitsSystemWindows = false,
            ),
        ) {
            Surface(
                modifier = Modifier.fillMaxSize(),
                color = MaterialTheme.colorScheme.background,
            ) {
                Column(
                    modifier = Modifier
                        .fillMaxSize()
                        .systemBarsPadding(),
                ) {
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(horizontal = 18.dp, vertical = 14.dp),
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(8.dp),
                    ) {
                        Column(modifier = Modifier.weight(1f)) {
                            Text(
                                "Speech library",
                                style = MaterialTheme.typography.headlineSmall,
                                fontWeight = FontWeight.Bold,
                            )
                            Text(
                                visibleHistory.size.toString() +
                                    if (visibleHistory.size == 1) " result · all Breeze models"
                                    else " results · all Breeze models",
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant,
                            )
                        }
                        IconButton(onClick = { showHistoryFilters = true }) {
                            BadgedBox(
                                badge = {
                                    if (activeFilterCount > 0) {
                                        Badge { Text(activeFilterCount.toString()) }
                                    }
                                },
                            ) {
                                Icon(
                                    Icons.Default.FilterList,
                                    contentDescription = "Filter speech history",
                                    tint = if (activeFilterCount > 0) {
                                        MaterialTheme.colorScheme.primary
                                    } else {
                                        MaterialTheme.colorScheme.onSurfaceVariant
                                    },
                                )
                            }
                        }
                        TextButton(onClick = { showHistory = false }) {
                            Text("Close")
                        }
                    }

                    OutlinedTextField(
                        value = historySearch,
                        onValueChange = { historySearch = it.take(200) },
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(horizontal = 16.dp),
                        singleLine = true,
                        leadingIcon = {
                            Icon(Icons.Default.Search, contentDescription = null)
                        },
                        label = { Text("Search script, direction or model") },
                    )

                    LazyRow(
                        modifier = Modifier.padding(horizontal = 16.dp, vertical = 10.dp),
                        horizontalArrangement = Arrangement.spacedBy(8.dp),
                    ) {
                        item {
                            FilterChip(
                                selected = historyModelIds.isEmpty(),
                                onClick = { historyModelIds = emptySet() },
                                label = { Text("All models") },
                            )
                        }
                        items(modelOptions, key = { it }) { option ->
                            FilterChip(
                                selected = option in historyModelIds,
                                onClick = {
                                    historyModelIds =
                                        if (option in historyModelIds) {
                                            historyModelIds - option
                                        } else {
                                            historyModelIds + option
                                        }
                                },
                                label = { Text(breezeHistoryModelLabel(option)) },
                            )
                        }
                        item {
                            Box {
                                AssistChip(
                                    onClick = { sortMenuOpen = true },
                                    label = { Text(sortMode.label) },
                                    leadingIcon = {
                                        Icon(
                                            Icons.Default.Sort,
                                            contentDescription = null,
                                            modifier = Modifier.size(18.dp),
                                        )
                                    },
                                )
                                DropdownMenu(
                                    expanded = sortMenuOpen,
                                    onDismissRequest = { sortMenuOpen = false },
                                ) {
                                    SpeechHistorySort.values().forEach { option ->
                                        DropdownMenuItem(
                                            text = { Text(option.label) },
                                            onClick = {
                                                historySort = option.name
                                                sortMenuOpen = false
                                            },
                                        )
                                    }
                                }
                            }
                        }
                    }

                    HorizontalDivider()

                    if (visibleHistory.isEmpty()) {
                        Box(
                            modifier = Modifier
                                .fillMaxWidth()
                                .weight(1f),
                            contentAlignment = Alignment.Center,
                        ) {
                            Column(
                                horizontalAlignment = Alignment.CenterHorizontally,
                                verticalArrangement = Arrangement.spacedBy(8.dp),
                            ) {
                                Text(
                                    if (history.isEmpty()) {
                                        "No Breeze generations yet."
                                    } else {
                                        "Nothing matches these filters."
                                    },
                                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                                )
                                if (
                                    historySearch.isNotBlank() ||
                                    historyFavoritesOnly ||
                                    historyAcceleratedOnly ||
                                    historyModelIds.isNotEmpty()
                                ) {
                                    TextButton(
                                        onClick = {
                                            historySearch = ""
                                            historyFavoritesOnly = false
                                            historyAcceleratedOnly = false
                                            historyModelIds = emptySet()
                                        },
                                    ) {
                                        Text("Clear filters")
                                    }
                                }
                            }
                        }
                    } else {
                        Box(
                            modifier = Modifier
                                .fillMaxWidth()
                                .weight(1f),
                        ) {
                            LazyColumn(
                                state = historyListState,
                                modifier = Modifier.fillMaxSize(),
                                contentPadding = PaddingValues(
                                    start = 16.dp,
                                    end = 22.dp,
                                    top = 16.dp,
                                    bottom = 28.dp,
                                ),
                                verticalArrangement = Arrangement.spacedBy(10.dp),
                            ) {
                                items(visibleHistory, key = { it.id }) { item ->
                                    val file = File(item.filePath)
                                    if (file.isFile) {
                                        Column(verticalArrangement = Arrangement.spacedBy(7.dp)) {
                                            MusicPlayerCard(
                                                file = file,
                                                title = item.text.take(96),
                                                subtitle = formatHistoryDate(item.createdAt) + " · " +
                                                    formatMillisCompact(item.audioDurationMillis) + " audio · " +
                                                    formatMillisCompact(item.generationMillis) + " generated",
                                                metadataLine = buildString {
                                                    append("Seed ")
                                                    append(item.seed)
                                                    append(" · CFG ")
                                                    append(String.format(Locale.US, "%.2f", item.cfg))
                                                },
                                                badges = listOfNotNull(
                                                    breezeHistoryModelLabel(item.modelId),
                                                    if (item.accelerated) "QNN" else null,
                                                ),
                                                favorite = item.favorite,
                                                onFavoriteToggle = {
                                                    val nextFavorite = !item.favorite
                                                    // Update immediately, then commit. A slow
                                                    // history.json write must not look like a dead button.
                                                    history = history.map { old ->
                                                        if (old.id == item.id &&
                                                            old.modelId == item.modelId
                                                        ) old.copy(favorite = nextFavorite) else old
                                                    }
                                                    scope.launch {
                                                        val saved = runCatching {
                                                            SpeechHistoryStore.setFavorite(
                                                                context,
                                                                item.modelId,
                                                                item.id,
                                                                nextFavorite,
                                                            )
                                                        }.getOrDefault(false)
                                                        history = withContext(Dispatchers.IO) {
                                                            SpeechHistoryStore.load(context)
                                                        }
                                                        if (!saved) {
                                                            Toast.makeText(
                                                                context,
                                                                "Could not update favorite",
                                                                Toast.LENGTH_SHORT,
                                                            ).show()
                                                        } else {
                                                            AppHaptics.perform(
                                                                context,
                                                                AppHaptics.Kind.Interaction,
                                                            )
                                                        }
                                                    }
                                                },
                                                onUse = {
                                                    text = item.text
                                                    instruction = item.instruction
                                                    seed = item.seed
                                                    seedLocked = true
                                                    cfg = item.cfg
                                                    temperature = item.temperature
                                                    topK = item.topK
                                                    topP = item.topP
                                                    repetition = item.repetition
                                                    splitChars = item.splitChars
                                                    maxNewTokens = item.maxNewTokens
                                                    showHistory = false
                                                    AppHaptics.perform(
                                                        context,
                                                        AppHaptics.Kind.Interaction,
                                                    )
                                                },
                                                onReproduce = {
                                                    text = item.text
                                                    instruction = item.instruction
                                                    seed = item.seed
                                                    seedLocked = true
                                                    cfg = item.cfg
                                                    temperature = item.temperature
                                                    topK = item.topK
                                                    topP = item.topP
                                                    repetition = item.repetition
                                                    splitChars = item.splitChars
                                                    maxNewTokens = item.maxNewTokens
                                                    showHistory = false
                                                    AppHaptics.perform(
                                                        context,
                                                        AppHaptics.Kind.Interaction,
                                                    )
                                                    startSpeechGeneration(
                                                        context = context,
                                                        modelId = item.modelId,
                                                        text = item.text,
                                                        instruction = item.instruction,
                                                        seed = item.seed,
                                                        cfg = item.cfg,
                                                        temperature = item.temperature,
                                                        topK = item.topK,
                                                        topP = item.topP,
                                                        repetition = item.repetition,
                                                        splitChars = item.splitChars,
                                                        maxNewTokens = item.maxNewTokens,
                                                    )
                                                },
                                                onSave = {
                                                    scope.launch {
                                                        val saved = runCatching {
                                                            SpeechHistoryStore.exportToMusic(
                                                                context,
                                                                file,
                                                            )
                                                        }.getOrNull()
                                                        Toast.makeText(
                                                            context,
                                                            if (saved != null) {
                                                                "Saved to Music/LocalDream"
                                                            } else {
                                                                "Could not save audio"
                                                            },
                                                            Toast.LENGTH_SHORT,
                                                        ).show()
                                                    }
                                                },
                                                onShare = { shareSpeechFile(context, file) },
                                                onDelete = {
                                                    scope.launch {
                                                        SpeechHistoryStore.delete(
                                                            context,
                                                            item.modelId,
                                                            item.id,
                                                        )
                                                        history = withContext(Dispatchers.IO) {
                                                            SpeechHistoryStore.load(context)
                                                        }
                                                    }
                                                },
                                                compact = true,
                                            )
                                        }
                                    }
                                }
                            }
                            SpeechHistoryScrollbar(
                                state = historyListState,
                                modifier = Modifier
                                    .align(Alignment.CenterEnd)
                                    .fillMaxHeight()
                                    .padding(vertical = 10.dp, horizontal = 5.dp),
                            )
                        }
                    }
                }
            }
        }

        if (showHistoryFilters) {
            ModalBottomSheet(
                onDismissRequest = { showHistoryFilters = false },
            ) {
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .verticalScroll(rememberScrollState())
                        .padding(horizontal = 20.dp)
                        .padding(bottom = 28.dp),
                    verticalArrangement = Arrangement.spacedBy(14.dp),
                ) {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        Column(modifier = Modifier.weight(1f)) {
                            Text(
                                "Filter speech",
                                style = MaterialTheme.typography.headlineSmall,
                                fontWeight = FontWeight.Bold,
                            )
                            Text(
                                "One library across Q4, Q8, DD variants and F16.",
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant,
                            )
                        }
                        TextButton(
                            onClick = {
                                historyModelIds = emptySet()
                                historyFavoritesOnly = false
                                historyAcceleratedOnly = false
                                historySort = SpeechHistorySort.Newest.name
                            },
                        ) {
                            Text("Reset")
                        }
                    }

                    Text(
                        "Models",
                        style = MaterialTheme.typography.titleMedium,
                        fontWeight = FontWeight.SemiBold,
                    )
                    modelOptions.forEach { option ->
                        val selected = option in historyModelIds
                        Surface(
                            onClick = {
                                historyModelIds =
                                    if (selected) historyModelIds - option
                                    else historyModelIds + option
                            },
                            shape = MaterialTheme.shapes.large,
                            color = if (selected) {
                                MaterialTheme.colorScheme.secondaryContainer
                            } else {
                                MaterialTheme.colorScheme.surfaceContainerHigh
                            },
                        ) {
                            Row(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .padding(horizontal = 16.dp, vertical = 13.dp),
                                verticalAlignment = Alignment.CenterVertically,
                            ) {
                                Text(
                                    breezeHistoryModelLabel(option),
                                    modifier = Modifier.weight(1f),
                                    fontWeight = if (selected) FontWeight.SemiBold else FontWeight.Normal,
                                )
                                Switch(
                                    checked = selected,
                                    onCheckedChange = {
                                        historyModelIds =
                                            if (it) historyModelIds + option
                                            else historyModelIds - option
                                    },
                                )
                            }
                        }
                    }

                    HorizontalDivider()

                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        Column(modifier = Modifier.weight(1f)) {
                            Text("Favorites only", fontWeight = FontWeight.SemiBold)
                            Text(
                                "Show only generations you starred.",
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant,
                            )
                        }
                        Switch(
                            checked = historyFavoritesOnly,
                            onCheckedChange = { historyFavoritesOnly = it },
                        )
                    }

                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        Column(modifier = Modifier.weight(1f)) {
                            Text("QNN waveform only", fontWeight = FontWeight.SemiBold)
                            Text(
                                "Only runs saved with the accelerated waveform decoder.",
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant,
                            )
                        }
                        Switch(
                            checked = historyAcceleratedOnly,
                            onCheckedChange = { historyAcceleratedOnly = it },
                        )
                    }

                    Button(
                        onClick = { showHistoryFilters = false },
                        modifier = Modifier.fillMaxWidth(),
                    ) {
                        Text("Done")
                    }
                }
            }
        }
    }

}

@Composable
private fun SpeechHistoryScrollbar(
    state: androidx.compose.foundation.lazy.LazyListState,
    modifier: Modifier = Modifier,
) {
    val layout = state.layoutInfo
    val total = layout.totalItemsCount
    val visible = layout.visibleItemsInfo
    if (total <= 1 || visible.isEmpty() || visible.size >= total) return

    val first = visible.first().index
    val last = visible.last().index
    val visibleCount = (last - first + 1).coerceAtLeast(1)
    val thumbFraction = (visibleCount.toFloat() / total.toFloat()).coerceIn(0.08f, 1f)
    val maxFirst = (total - visibleCount).coerceAtLeast(1)
    val positionFraction = (first.toFloat() / maxFirst.toFloat()).coerceIn(0f, 1f)

    BoxWithConstraints(
        modifier = modifier.width(5.dp),
    ) {
        val thumbHeight = (maxHeight * thumbFraction).coerceAtLeast(28.dp)
        val travel = (maxHeight - thumbHeight).coerceAtLeast(0.dp)
        Box(
            modifier = Modifier
                .fillMaxSize()
                .clip(CircleShape)
                .background(MaterialTheme.colorScheme.surfaceContainerHighest),
        )
        Box(
            modifier = Modifier
                .offset(y = travel * positionFraction)
                .fillMaxWidth()
                .height(thumbHeight)
                .clip(CircleShape)
                .background(MaterialTheme.colorScheme.primary.copy(alpha = 0.72f)),
        )
    }
}

@Composable
private fun InlineEventEditor(
    value: String,
    onValueChange: (String) -> Unit,
    events: List<InlineEventCue>,
    modifier: Modifier = Modifier,
) {
    var textLayout by remember(value) { mutableStateOf<TextLayoutResult?>(null) }
    var showExtended by rememberSaveable { mutableStateOf(false) }
    var fieldValue by rememberSaveable(stateSaver = TextFieldValue.Saver) {
        mutableStateOf(TextFieldValue(value, selection = TextRange(value.length)))
    }
    LaunchedEffect(value) {
        if (value != fieldValue.text) {
            fieldValue = TextFieldValue(value, selection = TextRange(value.length))
        }
    }

    val textColor = MaterialTheme.colorScheme.onSurface
    val placeholderColor = MaterialTheme.colorScheme.onSurfaceVariant
    val borderColor = MaterialTheme.colorScheme.outline
    val officialEnglishColor = MaterialTheme.colorScheme.tertiaryContainer
    val officialChineseColor = MaterialTheme.colorScheme.secondaryContainer
    val extendedColor = MaterialTheme.colorScheme.primaryContainer.copy(alpha = 0.55f)
    val radius = 7.dp
    val horizontalPad = 4.dp
    val verticalPad = 2.dp

    fun insertCue(cue: InlineEventCue) {
        val selectionStart = minOf(fieldValue.selection.start, fieldValue.selection.end)
            .coerceIn(0, fieldValue.text.length)
        val selectionEnd = maxOf(fieldValue.selection.start, fieldValue.selection.end)
            .coerceIn(selectionStart, fieldValue.text.length)
        val before = fieldValue.text.substring(0, selectionStart)
        val after = fieldValue.text.substring(selectionEnd)
        val prefix = if (before.isNotEmpty() && !before.last().isWhitespace()) " " else ""
        val suffix = if (after.isNotEmpty() && !after.first().isWhitespace()) " " else ""
        val insertion = prefix + cue.tag + suffix
        val updated = before + insertion + after
        val cursor = (before.length + insertion.length).coerceIn(0, updated.length)
        fieldValue = TextFieldValue(updated, selection = TextRange(cursor))
        onValueChange(updated)
    }

    Column(
        modifier = modifier,
        verticalArrangement = Arrangement.spacedBy(9.dp),
    ) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Text(
                "Script",
                style = MaterialTheme.typography.labelLarge,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier.weight(1f),
            )
            Text(
                value.length.toString() + " / 12000",
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
        Surface(
            shape = RoundedCornerShape(16.dp),
            color = MaterialTheme.colorScheme.surfaceContainerLow,
            border = androidx.compose.foundation.BorderStroke(1.dp, borderColor),
        ) {
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .heightIn(min = 180.dp)
                    .padding(16.dp),
            ) {
                if (fieldValue.text.isEmpty()) {
                    Text(
                        "What should Breeze say?",
                        color = placeholderColor,
                        style = MaterialTheme.typography.bodyLarge,
                    )
                }
                BasicTextField(
                    value = fieldValue,
                    onValueChange = { next ->
                        val clippedText = next.text.take(12000)
                        val clippedSelection = TextRange(
                            next.selection.start.coerceAtMost(clippedText.length),
                            next.selection.end.coerceAtMost(clippedText.length),
                        )
                        fieldValue = TextFieldValue(
                            text = clippedText,
                            selection = clippedSelection,
                            composition = next.composition?.let {
                                TextRange(
                                    it.start.coerceAtMost(clippedText.length),
                                    it.end.coerceAtMost(clippedText.length),
                                )
                            },
                        )
                        onValueChange(clippedText)
                    },
                    modifier = Modifier
                        .fillMaxWidth()
                        .heightIn(min = 148.dp)
                        .drawBehind {
                            val layout = textLayout ?: return@drawBehind
                            events.forEach { cue ->
                                val event = cue.tag
                                var searchFrom = 0
                                while (searchFrom < fieldValue.text.length) {
                                    val eventStart = fieldValue.text.indexOf(event, searchFrom)
                                    if (eventStart < 0) break
                                    val eventEnd =
                                        (eventStart + event.length).coerceAtMost(fieldValue.text.length)
                                    var activeLine = -1
                                    var left = Float.POSITIVE_INFINITY
                                    var top = Float.POSITIVE_INFINITY
                                    var right = Float.NEGATIVE_INFINITY
                                    var bottom = Float.NEGATIVE_INFINITY

                                    fun drawSegment() {
                                        if (activeLine < 0 || !left.isFinite()) return
                                        val hp = horizontalPad.toPx()
                                        val vp = verticalPad.toPx()
                                        drawRoundRect(
                                            color = when {
                                                !cue.official -> extendedColor
                                                cue.language == "中文" -> officialChineseColor
                                                else -> officialEnglishColor
                                            },
                                            topLeft = Offset(left - hp, top - vp),
                                            size = Size(
                                                (right - left) + hp * 2f,
                                                (bottom - top) + vp * 2f,
                                            ),
                                            cornerRadius = CornerRadius(
                                                radius.toPx(),
                                                radius.toPx(),
                                            ),
                                        )
                                    }

                                    for (offset in eventStart until eventEnd) {
                                        val line = layout.getLineForOffset(offset)
                                        val box = layout.getBoundingBox(offset)
                                        if (activeLine != -1 && line != activeLine) {
                                            drawSegment()
                                            left = Float.POSITIVE_INFINITY
                                            top = Float.POSITIVE_INFINITY
                                            right = Float.NEGATIVE_INFINITY
                                            bottom = Float.NEGATIVE_INFINITY
                                        }
                                        activeLine = line
                                        left = minOf(left, box.left)
                                        top = minOf(top, box.top)
                                        right = maxOf(right, box.right)
                                        bottom = maxOf(bottom, box.bottom)
                                    }
                                    drawSegment()
                                    searchFrom = eventEnd.coerceAtLeast(eventStart + 1)
                                }
                            }
                        },
                    textStyle = MaterialTheme.typography.bodyLarge.copy(color = textColor),
                    cursorBrush = SolidColor(MaterialTheme.colorScheme.primary),
                    onTextLayout = { textLayout = it },
                )
            }
        }

        Row(
            modifier = Modifier.fillMaxWidth(),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Column(modifier = Modifier.weight(1f)) {
                Text(
                    "Inline vocal events",
                    style = MaterialTheme.typography.labelLarge,
                )
                Text(
                    "Inserted at the cursor. Official tags are highlighted separately.",
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
            FilterChip(
                selected = showExtended,
                onClick = { showExtended = !showExtended },
                label = { Text(if (showExtended) "Extended" else "Official") },
            )
        }

        LazyRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            items(events.filter { it.official }, key = { it.tag }) { cue ->
                AssistChip(
                    onClick = { insertCue(cue) },
                    label = { Text(cue.tag) },
                    colors = AssistChipDefaults.assistChipColors(
                        containerColor = if (cue.language == "中文") {
                            MaterialTheme.colorScheme.secondaryContainer
                        } else {
                            MaterialTheme.colorScheme.tertiaryContainer
                        },
                    ),
                )
            }
        }
        if (showExtended) {
            LazyRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                items(events.filterNot { it.official }, key = { it.tag }) { cue ->
                    AssistChip(
                        onClick = { insertCue(cue) },
                        label = { Text(cue.tag) },
                        colors = AssistChipDefaults.assistChipColors(
                            containerColor = MaterialTheme.colorScheme.primaryContainer.copy(alpha = 0.72f),
                        ),
                    )
                }
            }
            Text(
                "Extended cues use Breeze's open-ended descriptive event behavior and may be less consistent than the official examples.",
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
    }
}

@Composable
private fun SpeechLoadingCard(state: SpeechState.Loading) {
    val elapsed by produceState(
        initialValue = 0f,
        key1 = state.startedAtMillis,
    ) {
        while (true) {
            value = (
                System.currentTimeMillis() - state.startedAtMillis
            ).coerceAtLeast(0L) / 1000f
            kotlinx.coroutines.delay(250L)
        }
    }
    val progress = state.progress.coerceIn(0f, 0.99f)

    Surface(
        shape = MaterialTheme.shapes.extraLarge,
        color = MaterialTheme.colorScheme.surfaceContainerHigh,
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        "Loading Breeze",
                        style = MaterialTheme.typography.titleMedium,
                        fontWeight = FontWeight.SemiBold,
                    )
                    Text(
                        state.detail,
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
                Text(
                    (progress * 100f).roundToInt().toString() + "%",
                    style = MaterialTheme.typography.titleLarge,
                    fontWeight = FontWeight.Bold,
                )
            }
            SmoothLinearWavyProgressIndicator(
                progress = progress,
                modifier = Modifier
                    .fillMaxWidth()
                    .height(8.dp),
            )
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(6.dp),
            ) {
                LoadingStage("Runtime", progress >= 0.12f, Modifier.weight(1f))
                LoadingStage("Model", progress >= 0.56f, Modifier.weight(1f))
                LoadingStage("Vocoder", progress >= 0.84f, Modifier.weight(1f))
                LoadingStage("Ready", progress >= 0.98f, Modifier.weight(1f))
            }
            Text(
                "Elapsed " + formatSpeechTime(elapsed) +
                    " · first load does the heavy setup; the model stays warm afterwards.",
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
    }
}

@Composable
private fun LoadingStage(
    label: String,
    complete: Boolean,
    modifier: Modifier = Modifier,
) {
    Surface(
        modifier = modifier,
        shape = RoundedCornerShape(10.dp),
        color = if (complete) {
            MaterialTheme.colorScheme.primaryContainer
        } else {
            MaterialTheme.colorScheme.surfaceContainerLow
        },
    ) {
        Text(
            label,
            modifier = Modifier.padding(horizontal = 7.dp, vertical = 6.dp),
            style = MaterialTheme.typography.labelSmall,
            color = if (complete) {
                MaterialTheme.colorScheme.onPrimaryContainer
            } else {
                MaterialTheme.colorScheme.onSurfaceVariant
            },
            maxLines = 1,
        )
    }
}

@Composable
private fun SpeechProgressCard(state: SpeechState.Generating) {
    val progress = state.progress?.coerceIn(0f, 1f)
    val percent = progress?.let { (it * 100f).roundToInt() }
    val elapsedLive by produceState(
        initialValue = 0f,
        key1 = state.startedAtMillis,
        key2 = state.elapsedSeconds,
    ) {
        while (true) {
            val wallElapsed =
                (System.currentTimeMillis() - state.startedAtMillis)
                    .coerceAtLeast(0L) / 1000f
            value = maxOf(state.elapsedSeconds ?: 0f, wallElapsed)
            kotlinx.coroutines.delay(250L)
        }
    }

    ElevatedCard(
        shape = MaterialTheme.shapes.extraLarge,
        colors = CardDefaults.elevatedCardColors(
            containerColor = MaterialTheme.colorScheme.surfaceContainerHigh,
        ),
        elevation = CardDefaults.elevatedCardElevation(defaultElevation = 2.dp),
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(13.dp),
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Column(modifier = Modifier.weight(1f)) {
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(8.dp),
                    ) {
                        Text(
                            "Generating speech",
                            style = MaterialTheme.typography.titleMedium,
                            fontWeight = FontWeight.SemiBold,
                        )
                    }
                    Text(
                        state.detail,
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
                if (percent != null) {
                    Text(
                        percent.toString() + "%",
                        style = MaterialTheme.typography.headlineSmall,
                        fontWeight = FontWeight.Bold,
                    )
                }
            }

            if (progress != null) {
                SmoothLinearWavyProgressIndicator(
                    progress = progress,
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(10.dp),
                )
            } else {
                SmoothIndeterminateLinearWavyProgressIndicator(
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(10.dp),
                )
            }

            SpeechStageProgress(
                label = "Voice tokens",
                progress = state.codecProgress,
                activeText = "Generating voice tokens",
            )
            SpeechStageProgress(
                label = "Waveform",
                progress = state.vocoderProgress,
                activeText = "Decoding waveform",
            )

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                SpeechMetric(
                    label = "Audio",
                    value = buildString {
                        append(String.format(Locale.US, "%.1fs", state.generatedSeconds))
                        state.estimatedSeconds?.let {
                            append(" / ")
                            append(String.format(Locale.US, "%.1fs", it))
                        }
                    },
                    modifier = Modifier.weight(1f),
                )
                SpeechMetric(
                    label = "Elapsed",
                    value = formatSpeechTime(elapsedLive),
                    modifier = Modifier.weight(1f),
                )
            }
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                SpeechMetric(
                    label = "ETA",
                    value = state.etaSeconds?.let { formatSpeechTime(it) } ?: "Estimating",
                    modifier = Modifier.weight(1f),
                )
                SpeechMetric(
                    label = "Speed",
                    value = when {
                        state.realtimeFactor != null && state.fps != null ->
                            String.format(
                                Locale.US,
                                "%.2f× · %.1f fps",
                                state.realtimeFactor,
                                state.fps,
                            )
                        state.realtimeFactor != null ->
                            String.format(Locale.US, "%.2f× realtime", state.realtimeFactor)
                        else -> "Warming up"
                    },
                    modifier = Modifier.weight(1f),
                )
            }

        }
    }
}

@Composable
private fun SpeechStageProgress(
    label: String,
    progress: Float?,
    activeText: String,
) {
    val value = progress?.coerceIn(0f, 1f)
    Column(verticalArrangement = Arrangement.spacedBy(5.dp)) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
        ) {
            Text(label, style = MaterialTheme.typography.labelMedium)
            Text(
                if (value != null) {
                    (value * 100f).roundToInt().toString() + "%"
                } else {
                    activeText
                },
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
        if (value != null) {
            SmoothLinearWavyProgressIndicator(
                progress = value,
                modifier = Modifier
                    .fillMaxWidth()
                    .height(6.dp),
            )
        } else {
            SmoothIndeterminateLinearWavyProgressIndicator(
                modifier = Modifier
                    .fillMaxWidth()
                    .height(6.dp),
            )
        }
    }
}

@Composable
private fun SpeechMetric(
    label: String,
    value: String,
    modifier: Modifier = Modifier,
) {
    Surface(
        modifier = modifier,
        shape = RoundedCornerShape(12.dp),
        color = MaterialTheme.colorScheme.surface.copy(alpha = 0.38f),
    ) {
        Column(
            modifier = Modifier.padding(horizontal = 12.dp, vertical = 9.dp),
            verticalArrangement = Arrangement.spacedBy(2.dp),
        ) {
            Text(
                label,
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            Text(
                value,
                style = MaterialTheme.typography.labelLarge,
                fontWeight = FontWeight.SemiBold,
            )
        }
    }
}

private fun formatSpeechTime(seconds: Float): String {
    val total = seconds.coerceAtLeast(0f).roundToInt()
    val minutes = total / 60
    val remain = total % 60
    return if (minutes > 0) {
        String.format(Locale.US, "%d:%02d", minutes, remain)
    } else {
        "${remain}s"
    }
}

@Composable
private fun BreezeSettingSlider(
    label: String,
    value: Float,
    range: ClosedFloatingPointRange<Float>,
    description: String,
    onValueChange: (Float) -> Unit,
    valueLabel: (Float) -> String,
) {
    Column(verticalArrangement = Arrangement.spacedBy(5.dp)) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
        ) {
            Text(label, fontWeight = FontWeight.SemiBold)
            Text(
                valueLabel(value),
                color = MaterialTheme.colorScheme.primary,
                style = MaterialTheme.typography.labelLarge,
            )
        }
        Text(
            description,
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
        Slider(
            value = value.coerceIn(range.start, range.endInclusive),
            onValueChange = onValueChange,
            valueRange = range,
        )
    }
}
