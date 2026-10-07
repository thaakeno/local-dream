package io.github.xororz.localdream.service

import android.content.ContentValues
import android.content.Context
import android.net.Uri
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import java.io.File
import java.util.UUID
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject

data class SpeechHistoryItem(
    val id: String,
    val filePath: String,
    val text: String,
    val instruction: String,
    val modelId: String,
    val seed: Long,
    val createdAt: Long,
    val generationMillis: Long = 0L,
    val audioDurationMillis: Long = 0L,
    val cfg: Float = 1f,
    val temperature: Float = 0.9f,
    val topK: Int = 50,
    val topP: Float = 1f,
    val repetition: Float = 1.1f,
    val splitChars: Int = 240,
    val maxNewTokens: Int = 750,
    val accelerated: Boolean = false,
    val favorite: Boolean = false,
)

object SpeechHistoryStore {
    private const val HISTORY_FILE = "history.json"
    private const val MAX_ITEMS = 100
    private val mutex = Mutex()

    fun directory(context: Context): File =
        File(context.filesDir, "speech_history").apply { mkdirs() }

    fun directory(context: Context, modelId: String): File =
        File(directory(context), safeModelId(modelId)).apply { mkdirs() }

    private fun safeModelId(modelId: String): String =
        modelId.replace(Regex("[^A-Za-z0-9._-]"), "_").ifBlank { "unknown-model" }

    suspend fun load(context: Context): List<SpeechHistoryItem> = withContext(Dispatchers.IO) {
        mutex.withLock {
            val root = directory(context)
            val legacy = loadFile(File(root, HISTORY_FILE))
            val partitioned = root.listFiles()
                ?.filter { it.isDirectory }
                ?.flatMap { loadFile(File(it, HISTORY_FILE)) }
                .orEmpty()
            (legacy + partitioned)
                .distinctBy { it.id }
                .filter { File(it.filePath).isFile }
                .sortedByDescending { it.createdAt }
        }
    }

    suspend fun loadForModel(
        context: Context,
        modelId: String,
    ): List<SpeechHistoryItem> = withContext(Dispatchers.IO) {
        mutex.withLock { loadForModelUnlocked(context, modelId) }
    }

    suspend fun add(
        context: Context,
        file: File,
        text: String,
        instruction: String,
        modelId: String,
        seed: Long,
        generationMillis: Long = 0L,
        audioDurationMillis: Long = 0L,
        cfg: Float = 1f,
        temperature: Float = 0.9f,
        topK: Int = 50,
        topP: Float = 1f,
        repetition: Float = 1.1f,
        splitChars: Int = 600,
        maxNewTokens: Int = 750,
        accelerated: Boolean = false,
    ): SpeechHistoryItem = withContext(Dispatchers.IO) {
        mutex.withLock {
            val item = SpeechHistoryItem(
                id = UUID.randomUUID().toString(),
                filePath = file.absolutePath,
                text = text,
                instruction = instruction,
                modelId = modelId,
                seed = seed,
                createdAt = System.currentTimeMillis(),
                generationMillis = generationMillis,
                audioDurationMillis = audioDurationMillis,
                cfg = cfg,
                temperature = temperature,
                topK = topK,
                topP = topP,
                repetition = repetition,
                splitChars = splitChars,
                maxNewTokens = maxNewTokens,
                accelerated = accelerated,
            )
            val next = (listOf(item) + loadForModelUnlocked(context, modelId))
                .filter { it.modelId == modelId && File(it.filePath).isFile }
                .distinctBy { it.id }
                .take(MAX_ITEMS)
            saveForModelUnlocked(context, modelId, next)
            item
        }
    }

    suspend fun exportToMusic(context: Context, file: File): Uri? =
        withContext(Dispatchers.IO) {
            if (!file.isFile) return@withContext null
            val displayName = "LocalDream_Breeze_" + System.currentTimeMillis() + ".wav"
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                val values = ContentValues().apply {
                    put(MediaStore.Audio.Media.DISPLAY_NAME, displayName)
                    put(MediaStore.Audio.Media.MIME_TYPE, "audio/wav")
                    put(
                        MediaStore.Audio.Media.RELATIVE_PATH,
                        Environment.DIRECTORY_MUSIC + "/LocalDream",
                    )
                    put(MediaStore.Audio.Media.IS_PENDING, 1)
                }
                val resolver = context.contentResolver
                val uri = resolver.insert(MediaStore.Audio.Media.EXTERNAL_CONTENT_URI, values)
                    ?: return@withContext null
                try {
                    resolver.openOutputStream(uri, "w")?.use { out ->
                        file.inputStream().use { it.copyTo(out) }
                    } ?: error("Could not open exported audio")
                    values.clear()
                    values.put(MediaStore.Audio.Media.IS_PENDING, 0)
                    resolver.update(uri, values, null, null)
                    uri
                } catch (t: Throwable) {
                    runCatching { resolver.delete(uri, null, null) }
                    throw t
                }
            } else {
                @Suppress("DEPRECATION")
                val dir = File(
                    Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_MUSIC),
                    "LocalDream",
                ).apply { mkdirs() }
                val target = File(dir, displayName)
                file.copyTo(target, overwrite = true)
                Uri.fromFile(target)
            }
        }

    suspend fun setFavorite(
        context: Context,
        modelId: String,
        id: String,
        favorite: Boolean,
    ): Boolean = withContext(Dispatchers.IO) {
        mutex.withLock {
            val items = loadForModelUnlocked(context, modelId)
            var changed = false
            val next = items.map { item ->
                if (item.id == id) {
                    changed = true
                    item.copy(favorite = favorite)
                } else item
            }
            if (changed) saveForModelUnlocked(context, modelId, next)
            changed
        }
    }

    suspend fun delete(
        context: Context,
        modelId: String,
        id: String,
    ): Boolean = withContext(Dispatchers.IO) {
        mutex.withLock {
            val items = loadForModelUnlocked(context, modelId)
            val target = items.firstOrNull { it.id == id } ?: return@withLock false
            runCatching { File(target.filePath).delete() }
            saveForModelUnlocked(context, modelId, items.filterNot { it.id == id })
            true
        }
    }

    private fun loadForModelUnlocked(
        context: Context,
        modelId: String,
    ): List<SpeechHistoryItem> {
        val modelFile = File(directory(context, modelId), HISTORY_FILE)
        val partitioned = loadFile(modelFile)
            .filter { it.modelId == modelId && File(it.filePath).isFile }
        if (partitioned.isNotEmpty() || modelFile.isFile) return partitioned

        // One-time lazy migration from the pre-v60 global history. This keeps
        // existing generations, but from now on every Breeze variant owns an
        // independent index and directory.
        val legacyFile = File(directory(context), HISTORY_FILE)
        val legacy = loadFile(legacyFile)
            .filter { it.modelId == modelId && File(it.filePath).isFile }
            .take(MAX_ITEMS)
        if (legacy.isNotEmpty()) saveForModelUnlocked(context, modelId, legacy)
        return legacy
    }

    private fun loadFile(file: File): List<SpeechHistoryItem> {
        if (!file.isFile) return emptyList()
        return runCatching {
            val array = JSONArray(file.readText())
            buildList {
                for (i in 0 until array.length()) {
                    val obj = array.getJSONObject(i)
                    val path = obj.optString("filePath")
                    if (path.isBlank()) continue
                    add(
                        SpeechHistoryItem(
                            id = obj.optString("id"),
                            filePath = path,
                            text = obj.optString("text"),
                            instruction = obj.optString("instruction"),
                            modelId = obj.optString("modelId"),
                            seed = obj.optLong("seed", 42L),
                            createdAt = obj.optLong("createdAt", 0L),
                            generationMillis = obj.optLong("generationMillis", 0L),
                            audioDurationMillis = obj.optLong("audioDurationMillis", 0L),
                            cfg = obj.optDouble("cfg", 1.0).toFloat(),
                            temperature = obj.optDouble("temperature", 0.9).toFloat(),
                            topK = obj.optInt("topK", 50),
                            topP = obj.optDouble("topP", 1.0).toFloat(),
                            repetition = obj.optDouble("repetition", 1.1).toFloat(),
                            splitChars = obj.optInt("splitChars", 240),
                            maxNewTokens = obj.optInt("maxNewTokens", 750),
                            accelerated = obj.optBoolean("accelerated", false),
                            favorite = obj.optBoolean("favorite", false),
                        ),
                    )
                }
            }
        }.getOrElse { emptyList() }
    }

    private fun saveForModelUnlocked(
        context: Context,
        modelId: String,
        items: List<SpeechHistoryItem>,
    ) {
        val file = File(directory(context, modelId), HISTORY_FILE)
        saveFile(file, items.filter { it.modelId == modelId }.take(MAX_ITEMS))
    }

    private fun saveFile(file: File, items: List<SpeechHistoryItem>) {
        val array = JSONArray()
        items.forEach { item ->
            array.put(
                JSONObject().apply {
                    put("id", item.id)
                    put("filePath", item.filePath)
                    put("text", item.text)
                    put("instruction", item.instruction)
                    put("modelId", item.modelId)
                    put("seed", item.seed)
                    put("createdAt", item.createdAt)
                    put("generationMillis", item.generationMillis)
                    put("audioDurationMillis", item.audioDurationMillis)
                    put("cfg", item.cfg.toDouble())
                    put("temperature", item.temperature.toDouble())
                    put("topK", item.topK)
                    put("topP", item.topP.toDouble())
                    put("repetition", item.repetition.toDouble())
                    put("splitChars", item.splitChars)
                    put("maxNewTokens", item.maxNewTokens)
                    put("accelerated", item.accelerated)
                    put("favorite", item.favorite)
                },
            )
        }
        file.parentFile?.mkdirs()
        val temp = File(file.parentFile, file.name + ".tmp")
        temp.writeText(array.toString())
        if (file.exists()) file.delete()
        if (!temp.renameTo(file)) {
            temp.copyTo(file, overwrite = true)
            temp.delete()
        }
    }
}
