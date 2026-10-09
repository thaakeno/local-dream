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
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.ArrowOutward
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
import io.github.xororz.localdream.service.BreezePerformanceMode
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
        "3 AM voice memo",
        "Realistic",
        "(sigh) Hey. It's almost three. I know I said I'd call, but I just got home. Anyway... remind me tomorrow to tell you what happened on the train.",
        "Sleepy young adult man leaving a genuine phone voice memo. Close mic, slightly dry throat, absent-minded stops and starts, conversational uneven rhythm. Never sound scripted.",
    ),
    SpeechTemplate(
        "Missed last train",
        "Realistic",
        "The board says forty-six minutes. (laugh) Forty-six. I bought coffee to stay awake and somehow that's made everything worse. Are you still up?",
        "Tired but amused young woman on a late-night station call. Informal clipped phrases, small self-conscious laugh, realistic pacing and cool nighttime quiet.",
    ),
    SpeechTemplate(
        "Voice message to Mom",
        "Realistic",
        "Hey, Mom. (sigh) I'm home, okay? I ate something too. You don't have to call back. I just wanted you to know I got here.",
        "Young adult man recording a short reassuring voice message after a difficult day. Warm, understated affection, slightly weary low voice, natural pauses and breathing.",
    ),
    SpeechTemplate(
        "Unexpected good news",
        "Realistic",
        "Wait, hang on. (laugh) They actually accepted it? I must've read the email six times. Give me a minute, I don't even know what to say.",
        "Young adult woman on an excited phone call. Genuine disbelieving laugh, irregular quick speech, breath catching as it sinks in, believable unpolished recording.",
    ),
    SpeechTemplate(
        "Kitchen confession",
        "Realistic",
        "Okay, so... (sigh) I broke the mug. The blue one you brought back from Italy. I tried to glue it before you woke up, which somehow made it worse.",
        "Apologetic adult male talking from his kitchen. Light embarrassed humor, a few hesitations, low relaxed register and a realistic intimate room recording.",
    ),
    SpeechTemplate(
        "Hospital corridor",
        "Realistic",
        "She's awake. (breathing heavily) The nurse just told me. I haven't slept and I can't stop smiling. Can you come? I really need to see someone.",
        "Adult man calling family after a hospital vigil. Exhaustion dissolves into relief, quiet shaking breath, voice catches without artificial crying.",
    ),
    SpeechTemplate(
        "The difficult apology",
        "Realistic",
        "(sigh) I listened to your message again. You were right about what I said. I wish I'd understood it then. Can we talk when you have time?",
        "Adult woman leaving a sincere private apology. Thoughtful, slightly strained voice, careful improvised phrasing and real pauses, no theatrical emphasis.",
    ),
    SpeechTemplate(
        "First day at work",
        "Realistic",
        "(laugh) Well, nobody told me the office door locks at six. So I'm outside, holding a sandwich and my brand-new badge. Pretty solid first day.",
        "Young adult man recording a lighthearted voice note, amused by an awkward mishap. Casual dry humor, natural breaths, authentic smartphone mic.",
    ),
    SpeechTemplate(
        "Old friends",
        "Realistic",
        "I recognized your laugh before I saw you. (chuckles) Twenty years, and you still do that thing where you look away when you know you're wrong.",
        "Middle-aged woman greeting an old friend in person. Warm spontaneous laughter, affectionate teasing, familiar conversational cadence.",
    ),
    SpeechTemplate(
        "Driving home",
        "Realistic",
        "I'm parked outside the house. (sigh) I don't know why I'm telling you this, but I can't make myself go in yet. Just talk to me for a minute.",
        "Quiet adult man speaking hands-free after a hard day. Slightly rough tired timbre, private hesitation, controlled unease, natural breath and pauses.",
    ),
    SpeechTemplate(
        "The unread message",
        "Realistic",
        "I typed a whole paragraph and deleted it. (laugh) Then I typed hello. Deleted that too. So... apparently this is the part where I actually call.",
        "Soft young adult woman in a candid call. Nervous embarrassed laughter, believable hesitation, gentle intimacy, never polished or performative.",
    ),
    SpeechTemplate(
        "Morning voice note",
        "Realistic",
        "(yawns) Morning. The alarm didn't go off, the cat's sitting on my keys, and I've somehow made coffee without water. I'll be there... eventually.",
        "Sleepy adult man leaving a playful rushed voice note. Natural low register and light morning rasp, small genuine yawn, unhurried delivery.",
    ),
    SpeechTemplate(
        "The last voicemail",
        "Emotional",
        "(sigh) I found your old message today. You were asking if I needed anything from the shop. I kept it because I liked hearing you say my name.",
        "Grieving adult woman speaking alone, low intimate voice, fragile composure and a restrained little catch on the last sentence. No forced sobbing.",
    ),
    SpeechTemplate(
        "Dad, I did it",
        "Emotional",
        "(laugh) Dad, I passed. I know you said not to worry, but I did. I really did. (sigh) I wish you could've been here for this.",
        "Young adult man addressing an absent father through a private recording. Starts with relieved excitement, ends quietly exposed, natural trembling breath.",
    ),
    SpeechTemplate(
        "The empty chair",
        "Emotional",
        "I still put out two cups in the morning. (sigh) It's ridiculous, I know. It's just... for a second, before I turn around, I forget.",
        "Older woman with a weathered, gentle voice. Grief is in routine rather than melodrama. Slow natural cadence, soft breath and small pauses.",
    ),
    SpeechTemplate(
        "Found alive",
        "Emotional",
        "(gasps) You're okay. You're actually okay. (laugh) Don't move, just stay right there. I'm coming. I'm coming right now.",
        "Adult woman receiving news that a missing loved one is safe. Shock becomes a breathless relieved laugh, authentic urgency and delicate voice cracks.",
    ),
    SpeechTemplate(
        "Letters in a box",
        "Emotional",
        "I found the letters we wrote when we were seventeen. (chuckles) You kept the awful drawing I made. I can't believe you kept it all this time.",
        "Adult man touched by an unexpected memory. Warm and wistful, slight smile breaks through vulnerability, gentle faltering rhythm.",
    ),
    SpeechTemplate(
        "The apology years late",
        "Emotional",
        "You were right to leave. (sigh) I spent years telling myself a different story. I just wanted you to hear the truth from me, once.",
        "Middle-aged man admitting fault after years. Quiet low chest resonance, remorse held back, precise sincere speech and natural silences.",
    ),
    SpeechTemplate(
        "Goodbye at the station",
        "Emotional",
        "(sigh) Your train's here. Right. Okay. I won't do the whole speech. Just... get there safe, and call me when you see the sea.",
        "Soft adult woman trying to keep a goodbye ordinary. Brief breath-catching pauses, affectionate smile under sadness, believable platform farewell.",
    ),
    SpeechTemplate(
        "After the rescue",
        "Emotional",
        "(breathing heavily) They got everyone out. All of them. I thought we'd lost the little one. (sigh) I'm going to sit down now.",
        "Exhausted adult male rescuer, relief after sustained panic. Breathless opening easing into a nearly inaudible finish, grounded physical fatigue.",
    ),
    SpeechTemplate(
        "The old dog",
        "Emotional",
        "He still waits by the door at six. (sigh) That's when you used to get home. I know dogs don't count days. But he keeps waiting.",
        "Older male voice with gentle warmth and buried grief. Quiet domestic sincerity, long emotional pauses, no forced quivering.",
    ),
    SpeechTemplate(
        "What I never said",
        "Emotional",
        "I kept saying we'd have more time. (sigh) And every time you asked me to stay for dinner, I said next week. I'm sorry I didn't stay.",
        "Adult daughter processing regret, softly spoken confession with restrained emotion and an honest broken rhythm; not a dramatic monologue.",
    ),
    SpeechTemplate(
        "Last signal from orbit",
        "Cinematic",
        "Earth is coming over the horizon now. It's beautiful. (sigh) The oxygen alarm's been ringing for six minutes. I think I'll watch the sunrise.",
        "Adult male astronaut's final transmission. Low warm voice, measured fading breath, intimacy and quiet acceptance instead of grand tragedy.",
    ),
    SpeechTemplate(
        "The lighthouse",
        "Cinematic",
        "Coast Guard, this is Grey Point. (clears throat) There is a light out beyond the reef. It's been following the boat. There is no boat.",
        "Older lighthouse keeper on coastal radio. Practical calm shifting toward disbelief, steady low voice, compressed urgency without artificial sound effects.",
    ),
    SpeechTemplate(
        "The king's decision",
        "Cinematic",
        "Take the crown. Keep the city. (sigh) But you'll have no more soldiers from me. I've buried enough sons for one lifetime.",
        "Aging male monarch with deep weathered resonance, slow solemn authority and grief held just below the surface. Final line low and decisive.",
    ),
    SpeechTemplate(
        "The sealed bunker",
        "Cinematic",
        "(whispers) We heard knocking again. Three knocks, every hour. (breathing heavily) The problem is... the last person outside died yesterday.",
        "Adult female bunker survivor, tight nearly whispered narration, rising unease, restrained breath and unsettling matter-of-fact realism.",
    ),
    SpeechTemplate(
        "Emergency broadcast",
        "Cinematic",
        "This is not a test. Stay away from the windows. (inhales) If someone outside calls you by name, do not answer. Please listen carefully.",
        "Mature male emergency announcer with controlled public-service diction cracking into private fear, steady baritone and clipped phrases.",
    ),
    SpeechTemplate(
        "The time loop",
        "Cinematic",
        "(laugh) You said that yesterday. No, don't look at me like that. You said those exact words yesterday... and the day before.",
        "Adult woman confronting an impossible repetition. A brief disbelieving laugh gives way to a low frightened insistence, close cinematic microphone.",
    ),
    SpeechTemplate(
        "Captain's last order",
        "Cinematic",
        "(shouts) Get everyone below deck! The bridge is mine. (sigh) And tell them I didn't forget a single name.",
        "Weathered male ship captain, command shouted over imagined chaos then a smaller, intensely personal farewell. Clear emotional contrast.",
    ),
    SpeechTemplate(
        "The empty throne",
        "Cinematic",
        "I waited outside that room for eleven years. (sigh) Now the door's open, and all I can think about is the boy I was when they shut it.",
        "Mature woman with low velvety authority, guarded grief and a measured personal confession. Minimal theatricality, strong consonants.",
    ),
    SpeechTemplate(
        "The final question",
        "Cinematic",
        "You can arrest me. (chuckles) You can erase every recording. But answer one thing first: why did it know my daughter's name?",
        "Middle-aged male investigator, exhausted resolve and one dry humorless chuckle. Steady menacing understatement rather than shouting.",
    ),
    SpeechTemplate(
        "The hidden page",
        "Whisper",
        "(whispers) That book wasn't here yesterday. The note inside has tomorrow's date. (inhales) Don't turn the next page until I get back.",
        "Adult woman whispering just inches from a microphone. Delicate breaths, soft clear consonants, tense but authentic secrecy.",
    ),
    SpeechTemplate(
        "Rain on glass",
        "Whisper",
        "(sigh) Listen. The rain's getting softer. You don't need to fix anything tonight. Just sit here and let the room be quiet.",
        "Warm adult feminine close whisper, slow gentle breath, clear wording and natural softness. Bedtime calm, no contrived ASMR theatrics.",
    ),
    SpeechTemplate(
        "A midnight secret",
        "Whisper",
        "(whispers) I left the key under the third stair. You remember the old house, don't you? (sigh) I never meant to stay away this long.",
        "Older feminine whisper recorded into a small handheld mic. Fragile wistfulness, intimate spacing and quiet urgency.",
    ),
    SpeechTemplate(
        "By the fireplace",
        "Whisper",
        "The fox came back again tonight. (chuckles) Sat by the gate like he owned the place. I think he's waiting for the snow.",
        "Gentle older male storyteller speaking in a near whisper, warm grainy texture, dry little chuckle and calm nighttime pacing.",
    ),
    SpeechTemplate(
        "Beside the crib",
        "Whisper",
        "(whispers) There you are. I thought you'd never fall asleep. (sigh) I'll stay until the light comes through the curtains.",
        "Soft maternal adult whisper, affectionate exhaustion, barely audible breathing, tender naturally uneven rhythm.",
    ),
    SpeechTemplate(
        "Inside the archive",
        "Whisper",
        "(inhales) Shh. Listen to that. Someone's walking upstairs. (whispers) We're the only people with keys to this floor.",
        "Young adult male in a hushed close-mic thriller scene. Crisp restrained whisper, short breaths and genuine cautious stillness.",
    ),
    SpeechTemplate(
        "You made it home",
        "Whisper",
        "(sigh) You're home. Good. I left the hallway light on. There's tea in the kitchen, if you want it. We can talk tomorrow.",
        "Mature male comforting near-whisper, soft gravelly warmth and natural exhale. Safe domestic intimacy, not romanticized or breathy caricature.",
    ),
    SpeechTemplate(
        "The sleeper train",
        "Whisper",
        "(whispers) Don't wake her. She's finally sleeping. The mountains will be outside the window when she opens her eyes.",
        "Quiet adult female storyteller on a night train. Gentle deliberate near-whisper, warm breaths and tiny affectionate smile.",
    ),
    SpeechTemplate(
        "The retired swordsman",
        "Anime",
        "(sigh) I spent twenty years perfecting that move. You learned it in a week. (laugh) Well... I suppose that's why I chose you.",
        "Older male anime mentor with dry humor, low gravelly pride, warmth underneath a stern voice. Controlled expressiveness, not caricature.",
    ),
    SpeechTemplate(
        "Last round",
        "Anime",
        "You can keep the trophy. (chuckles) I didn't come for it. I came here to see if I could finally beat you.",
        "Confident adult female anime rival, lightly teasing beginning and sincere controlled intensity. Musical expressive phrasing, clear natural register.",
    ),
    SpeechTemplate(
        "The quiet villain",
        "Anime",
        "How curious. You've won every battle, yet you're still frightened. (sigh) Tell me... what will you do when there's no one left to fight?",
        "Mature male anime antagonist, unusually serene, silky low resonance, intelligent precise pauses and unsettling quiet compassion.",
    ),
    SpeechTemplate(
        "Captain's promise",
        "Anime",
        "(shouts) Everyone below deck! I'll hold the line. (sigh) If they ask about me... tell them I was terrible at following orders.",
        "Young adult male anime captain. Sudden confident commands soften into an almost smiling farewell. High emotional contrast, no melodramatic screaming.",
    ),
    SpeechTemplate(
        "The stubborn healer",
        "Anime",
        "(sigh) Stop apologizing. You were hurt, so I helped you. It's really that simple. (chuckles) Next time, try not to fight a mountain.",
        "Adult female healer character, tender exasperation with teasing warmth. Soft clear anime-inspired voice, natural affectionate timing.",
    ),
    SpeechTemplate(
        "At the final gate",
        "Anime",
        "(breathing heavily) That's it? That's the guardian? (laugh) After everything they told us, I expected someone taller.",
        "Young male adventurer, exhausted nervous laughter hiding fear, energetic animation-style timing while staying vocally believable.",
    ),
    SpeechTemplate(
        "The last spell",
        "Anime",
        "I can open the door once. Only once. (inhales) So whatever you need to say to him, say it before the light goes out.",
        "Older female sorcerer, calm deliberate magical authority and protective urgency. Clear grounded resonance, one decisive breath before the warning.",
    ),
    SpeechTemplate(
        "Rival's return",
        "Anime",
        "(chuckles) You got stronger. Good. I've been waiting a long time for you to say my name without looking away.",
        "Adult male anime rival, relaxed confident baritone, amused tension and a sharp final emphasis; polished but not shouty.",
    ),
    SpeechTemplate(
        "After sunset",
        "Documentary",
        "At the edge of the desert, the temperature can fall more than thirty degrees after sunset. (inhales) For the animals that live here, surviving the night is a second battle.",
        "Mature male nature documentary narrator. Rich but restrained baritone, patient measured diction, quiet awe and a controlled breath between ideas.",
    ),
    SpeechTemplate(
        "The black ocean",
        "Documentary",
        "Six thousand meters below the surface, sunlight has never reached the seafloor. Yet around a crack in the Earth, life gathers in astonishing numbers.",
        "Deeply curious mature male ocean documentary voice, warm bass resonance, measured scientific clarity and spacious, natural pauses.",
    ),
    SpeechTemplate(
        "The forest beneath",
        "Documentary",
        "Under every step in this forest lies another world. Threads of fungus move through the soil, connecting roots we can see to lives we cannot.",
        "Warm reflective female nature narrator. Intimate curiosity, gentle precise enunciation, low restrained wonder; not a movie-trailer performance.",
    ),
    SpeechTemplate(
        "The ice remembers",
        "Documentary",
        "A glacier moves so slowly that the change is almost invisible. (sigh) But leave a camera in one place long enough, and the mountain appears to breathe.",
        "Older thoughtful male narrator, contemplative pacing and soft amazement, clean documentary delivery, natural pause before the image.",
    ),
    SpeechTemplate(
        "The last lighthouse",
        "Documentary",
        "For generations, this light warned ships away from the rocks. Today the lamp is automated. (sigh) But the keeper still comes to check the glass.",
        "Mature male human-interest documentary voice, warm low register and a hint of nostalgia. Calm factual tone with understated emotion.",
    ),
    SpeechTemplate(
        "The night hunter",
        "Documentary",
        "An owl can cross a clearing so quietly that prey never hears its approach. (inhales) For a mouse below, the darkness offers almost no warning.",
        "Clear adult female wildlife narrator, measured tension, crisp articulation and gentle breath, scientific awe without exaggerated danger.",
    ),
    SpeechTemplate(
        "City after rain",
        "Documentary",
        "When the streets empty, another city emerges. Foxes follow the railway lines. Bats circle the old bridges. And the night shift begins.",
        "Mature male urban wildlife documentarian, inviting conversational authority, slight rhythmic acceleration through the examples.",
    ),
    SpeechTemplate(
        "The great migration",
        "Documentary",
        "Every year, animals cross rivers, mountains and open plains to find food. (sigh) For the youngest travelers, even the first mile can be dangerous.",
        "Deep male nature-documentary delivery, compassionate measured tone with restrained urgency. Slow clear syllables, natural exhale.",
    ),
    SpeechTemplate(
        "Signals from space",
        "Documentary",
        "A radio telescope listens to a sky that appears silent. (inhales) Some of the signals it receives began their journey long before humans existed.",
        "Mature adult woman narrating an astronomy documentary, precise informative pacing, quiet cosmic wonder and soft breath.",
    ),
    SpeechTemplate(
        "Beneath the old city",
        "Documentary",
        "The streets above have changed a hundred times. Down here, behind a sealed doorway, workers found a room nobody had entered in centuries.",
        "Male archaeology documentary narrator with measured suspense, natural low resonance and genuine historical curiosity; no invented discovery presented as fact beyond the scene.",
    ),
    SpeechTemplate(
        "The coral nursery",
        "Documentary",
        "A young coral begins life no larger than a grain of sand. If it finds the right place to settle, it may help build a home for thousands of other creatures.",
        "Warm female marine-life narrator, bright scientific curiosity, smooth accessible cadence and relaxed confident tone.",
    ),
    SpeechTemplate(
        "Inside a volcano",
        "Documentary",
        "Deep below a volcano, rock can melt and collect in reservoirs. (inhales) At the surface, the landscape may look perfectly still.",
        "Male science documentary narrator, low calm authority and patient emphasis, understated tension in the contrast between stillness and heat.",
    ),
    SpeechTemplate(
        "A nation after the storm",
        "Presidential",
        "My fellow Americans, the storm took our roofs. It did not take our resolve. (sigh) Tonight, we begin the harder work: rebuilding what we share.",
        "Original mature male presidential speaker, warm confident baritone, thoughtful rhythmic pauses and conversational emphasis inspired by Barack Obama's measured oratorical style. Natural conviction, precise diction, emotionally controlled; not a celebrity voice clone.",
    ),
    SpeechTemplate(
        "The work ahead",
        "Presidential",
        "Progress is rarely loud. It happens in classrooms before sunrise and factories after midnight. (inhales) That's where our future is being built.",
        "Original mature male presidential speaker, warm confident baritone, thoughtful rhythmic pauses and conversational emphasis inspired by Barack Obama's measured oratorical style. Natural conviction, precise diction, emotionally controlled; not a celebrity voice clone.",
    ),
    SpeechTemplate(
        "An address at midnight",
        "Presidential",
        "Tonight, our families deserve the truth. We do not have every answer yet. (sigh) But we will face the uncertainty together, and we will not look away.",
        "Original mature male presidential speaker, warm confident baritone, thoughtful rhythmic pauses and conversational emphasis inspired by Barack Obama's measured oratorical style. Natural conviction, precise diction, emotionally controlled; not a celebrity voice clone.",
    ),
    SpeechTemplate(
        "A message to graduates",
        "Presidential",
        "You will make mistakes. So did everyone standing behind me. (chuckles) What matters is what you do the morning after you realize you were wrong.",
        "Original mature male presidential speaker, warm confident baritone, thoughtful rhythmic pauses and conversational emphasis inspired by Barack Obama's measured oratorical style. Natural conviction, precise diction, emotionally controlled; not a celebrity voice clone.",
    ),
    SpeechTemplate(
        "Farewell from the podium",
        "Presidential",
        "When I first stood here, I thought leadership meant having the answers. (sigh) I leave knowing it means listening when the answers are difficult.",
        "Original mature male presidential speaker, warm confident baritone, thoughtful rhythmic pauses and conversational emphasis inspired by Barack Obama's measured oratorical style. Natural conviction, precise diction, emotionally controlled; not a celebrity voice clone.",
    ),
    SpeechTemplate(
        "The courage to disagree",
        "Presidential",
        "We will not agree on everything. We never have. (inhales) But disagreement need not become contempt. We can argue fiercely and still build together.",
        "Original mature male presidential speaker, warm confident baritone, thoughtful rhythmic pauses and conversational emphasis inspired by Barack Obama's measured oratorical style. Natural conviction, precise diction, emotionally controlled; not a celebrity voice clone.",
    ),
    SpeechTemplate(
        "The quiet heroes",
        "Presidential",
        "There are people who will never stand behind this microphone. (sigh) They teach, they heal, they stay when everyone else leaves. This nation rests on them.",
        "Original mature male presidential speaker, warm confident baritone, thoughtful rhythmic pauses and conversational emphasis inspired by Barack Obama's measured oratorical style. Natural conviction, precise diction, emotionally controlled; not a celebrity voice clone.",
    ),
    SpeechTemplate(
        "The next generation",
        "Presidential",
        "Someday, the children listening tonight will ask what we did when the moment came. (inhales) I want us to be able to answer without lowering our eyes.",
        "Original mature male presidential speaker, warm confident baritone, thoughtful rhythmic pauses and conversational emphasis inspired by Barack Obama's measured oratorical style. Natural conviction, precise diction, emotionally controlled; not a celebrity voice clone.",
    ),
    SpeechTemplate(
        "The emergency briefing",
        "Presidential",
        "I will not tell you this will be easy. (sigh) I will tell you that help is moving, every hour matters, and no community will be forgotten.",
        "Original mature male presidential speaker, warm confident baritone, thoughtful rhythmic pauses and conversational emphasis inspired by Barack Obama's measured oratorical style. Natural conviction, precise diction, emotionally controlled; not a celebrity voice clone.",
    ),
    SpeechTemplate(
        "A speech on hope",
        "Presidential",
        "Hope is not the belief that the road is short. It is the decision to take another step when the road disappears into darkness. (inhales) We take that step together.",
        "Original mature male presidential speaker, warm confident baritone, thoughtful rhythmic pauses and conversational emphasis inspired by Barack Obama's measured oratorical style. Natural conviction, precise diction, emotionally controlled; not a celebrity voice clone.",
    ),
    SpeechTemplate(
        "The rescue call",
        "Intense",
        "(breathing heavily) I can see the car. The water's rising. I've got the door open, but the belt won't move. (shouts) I NEED A KNIFE!",
        "Male first responder in real danger, clear commands driven by physical exertion, breathless intensity and credible panic. Sudden volume contrast.",
    ),
    SpeechTemplate(
        "The broken promise",
        "Intense",
        "You said you'd be there when she woke up. (sigh) She waited all morning. And somehow you still think this is about me being angry.",
        "Mature woman whose anger barely contains hurt. Low clipped accusation with one tired exhale, last line quiet and cutting.",
    ),
    SpeechTemplate(
        "No one left behind",
        "Intense",
        "(shouts) Turn this vehicle around! There are still people inside that building. (breathing heavily) I'm not signing their names onto a list.",
        "Seasoned male rescue leader with rough strained voice, urgent projection, breaths broken by effort and resolve held through exhaustion.",
    ),
    SpeechTemplate(
        "The courtroom",
        "Intense",
        "(laugh) So that's it? All those years, and he just walks out? (sigh) Look at me. At least have the courage to look at me.",
        "Adult male betrayed by the verdict, sharp disbelief becoming restrained heartbreak. Bitter spontaneous laugh, realistic vocal tension.",
    ),
    SpeechTemplate(
        "The last warning",
        "Intense",
        "Put it down. (breathing heavily) I'm asking you one more time. I don't want anyone hurt, but I need to see your hands.",
        "Female crisis negotiator, low steady professional commands under extreme strain, careful clipped breaths, quiet determination.",
    ),
    SpeechTemplate(
        "After the fire",
        "Intense",
        "(cough) Everyone's outside. I counted twice. (breathing heavily) No, wait... where's the little boy who was holding the red balloon?",
        "Exhausted adult firefighter, smoky rough voice with sudden rising dread, believable cough and physical fatigue.",
    ),
    SpeechTemplate(
        "The silent threat",
        "Intense",
        "(whispers) Don't turn around. Keep walking. (inhales) The man behind us has followed you through four streets. I'm right here.",
        "Adult male protector speaking at close range, taut controlled whisper and urgent precision, no cartoon menace.",
    ),
    SpeechTemplate(
        "The locked door",
        "Intense",
        "(shouts) Open the door! I know you can hear me! (sigh) Please. Just let me know she's alive. That's all I'm asking.",
        "Adult woman shifting from furious shouting to exhausted pleading, subtle voice fracture and deep exhausted exhale.",
    ),
    SpeechTemplate(
        "The broken radio",
        "Intense",
        "(breathing heavily) Command, this is Echo Three. We're out of time. (shouts) SEND THE SIGNAL NOW! I'll keep the channel open.",
        "Male field operator in crisis, tight short phrases, explosive command and audible exertion, grounded militaristic urgency.",
    ),
    SpeechTemplate(
        "The old clockmaker",
        "Character",
        "(chuckles) This one hasn't worked since the winter of '83. I keep it anyway. It reminds me that not everything broken needs replacing.",
        "Wise elderly male craftsperson, rich aged gravel, warm little chuckle, thoughtful pacing and subtle personal nostalgia.",
    ),
    SpeechTemplate(
        "The tired detective",
        "Character",
        "(sigh) The witness remembers the rain. The driver remembers the clock. Nobody remembers the man who got into the back seat.",
        "Middle-aged male noir detective, dry rasp and observant low register, understated cynicism and quick flashes of curiosity.",
    ),
    SpeechTemplate(
        "The android's doubt",
        "Character",
        "I was designed to remember everything. (inhales) Yet there is a moment from yesterday I cannot find. I think... I chose to forget it.",
        "Calm synthetic-leaning male character, near-human warm clarity and careful measured rhythm, subtle unease rather than robotic filtering.",
    ),
    SpeechTemplate(
        "The innkeeper",
        "Character",
        "(laugh) Another traveler asking for the road north. Sit down first. There's stew on the fire, and the snow won't care how brave you are.",
        "Middle-aged female fantasy innkeeper, earthy warmth and easy laugh, grounded village dialect without a forced accent.",
    ),
    SpeechTemplate(
        "The exhausted boxer",
        "Character",
        "(breathing heavily) Kid, you can't win every round. Sometimes you just stay on your feet. (sigh) That's a kind of victory too.",
        "Older male trainer, chesty rough low voice, physical breath and sincere experience, modest spoken encouragement.",
    ),
    SpeechTemplate(
        "The archivist",
        "Character",
        "(clears throat) Ah, the silver box. Nobody's asked for that in fifty years. Tell me, young man... who gave you my brother's name?",
        "Elderly male archivist, dry cautious timbre, formal careful diction giving way to personal surprise and apprehension.",
    ),
    SpeechTemplate(
        "The sea captain",
        "Character",
        "(chuckles) Maps are suggestions, not promises. Out here, the wind gets the final say. Now tie that knot before it teaches you itself.",
        "Weathered mature female sea captain, resonant relaxed authority, amused low chuckle and confident quick instructions.",
    ),
    SpeechTemplate(
        "The last chess move",
        "Character",
        "(sigh) I saw the trap ten moves ago. I just wanted to know whether you'd find it. (chuckles) Well played.",
        "Older male chess master, soft amused voice, patient knowing rhythm and gentle pride with no villainous performance.",
    ),
    SpeechTemplate(
        "深夜来电",
        "Mandarin",
        "(叹气) 你还没睡啊？我刚到家。今天真的有点累，不过听见你的声音，突然就觉得好多了。",
        "自然的成年男性普通话，深夜电话留言般的轻声交流。疲惫逐渐变成放松，轻微呼吸，语气亲近而自然。",
    ),
    SpeechTemplate(
        "地铁里的好消息",
        "Mandarin",
        "等等，你是说真的？(笑) 我拿到录取通知了？天哪，我还以为自己一点机会都没有！",
        "年轻成年女性普通话，惊喜时忍不住笑，语速一时加快又停下来确认。口语化、有生活感，不要播音腔。",
    ),
    SpeechTemplate(
        "雨夜的故事",
        "Mandarin",
        "雨下了一整晚。(叹气) 街上的灯一盏一盏熄灭，只有那间小书店，还透着温暖的光。",
        "成年女性中文故事叙述声，低而柔和，句间有舒服的停顿和细微叹息，像真实的夜间读书。",
    ),
    SpeechTemplate(
        "最后的讯号",
        "Mandarin",
        "(喘息) 指挥中心，能听到吗？我们的电量只剩百分之三。外面那个东西……它又回来了。",
        "成年男性普通话科幻无线通讯，紧张但不浮夸。短促喘息，最后一句压低声音，像害怕被听见。",
    ),
    SpeechTemplate(
        "老朋友重逢",
        "Mandarin",
        "你还是老样子啊。(笑) 说着不来，结果比谁都先到。坐吧，我有好多事想跟你聊。",
        "中年女性普通话，老友重逢时放松的笑意、轻微打趣和久别后的温暖，语气自然、不做作。",
    ),
    SpeechTemplate(
        "没说出口的话",
        "Mandarin",
        "(叹气) 我写了很多次，却一直没有发给你。其实也没什么特别的……就是想问问，你过得好不好。",
        "成年男性普通话，慢速真诚的私人语音留言，稍微哽住但不过度哭泣，呼吸细腻而可信。",
    ),
    SpeechTemplate(
        "森林深处",
        "Mandarin",
        "夜幕落下后，森林并没有沉睡。(吸气) 落叶下面、树梢之间，无数细小的生命才刚刚开始活动。",
        "成熟男性自然纪录片中文旁白，低沉清晰，富有观察力和克制的惊叹，句间留白，自然呼吸。",
    ),
    SpeechTemplate(
        "回家的路",
        "Mandarin",
        "(笑) 你猜怎么着，我今天走错路了。结果绕到小时候那家面馆，老板居然还记得我。",
        "年轻成年女性普通话，像给朋友发语音，带惊喜的小笑声和自然停顿，轻快亲切。",
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
    // Persisted independently of CFG and the existing Full-QNN A/B switch.
    var fastRvqEnabled by remember {
        mutableStateOf(BreezePerformanceMode.isEnabled(context))
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
                                    "Experimental RVQ fast decode",
                                    fontWeight = FontWeight.SemiBold,
                                )
                                Text(
                                    if (fastRvqEnabled) {
                                        "ON · Speculative codebook verification. Keeps CFG and the original sampling distribution. Switch off for the verified legacy decoder."
                                    } else {
                                        "OFF · Verified original decoder. Switch on to A/B test the new algorithm. 20 FPS is a target, not a measured result."
                                    },
                                    style = MaterialTheme.typography.bodySmall,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                                )
                            }
                            Switch(
                                checked = fastRvqEnabled,
                                enabled = !busy,
                                onCheckedChange = { enabled ->
                                    fastRvqEnabled = enabled
                                    BreezePerformanceMode.setEnabled(context, enabled)
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
                                                Intent(context, SpeechGenerationService::class.java)
                                                    .setAction(SpeechGenerationService.ACTION_PRELOAD)
                                                    .putExtra("modelId", modelId),
                                            )
                                        }
                                    }
                                },
                            )
                        }
                    }

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
