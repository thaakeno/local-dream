package io.github.xororz.localdream.service

import android.content.Context
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
)

object SpeechHistoryStore {
    private const val HISTORY_FILE = "history.json"
    private const val MAX_ITEMS = 100
    private val mutex = Mutex()

    fun directory(context: Context): File =
        File(context.filesDir, "speech_history").apply { mkdirs() }

    suspend fun load(context: Context): List<SpeechHistoryItem> = withContext(Dispatchers.IO) {
        mutex.withLock { loadUnlocked(context) }
    }

    suspend fun add(
        context: Context,
        file: File,
        text: String,
        instruction: String,
        modelId: String,
        seed: Long,
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
            )
            val next = (listOf(item) + loadUnlocked(context))
                .filter { File(it.filePath).isFile }
                .take(MAX_ITEMS)
            saveUnlocked(context, next)
            item
        }
    }

    suspend fun delete(context: Context, id: String): Boolean = withContext(Dispatchers.IO) {
        mutex.withLock {
            val items = loadUnlocked(context)
            val target = items.firstOrNull { it.id == id } ?: return@withLock false
            runCatching { File(target.filePath).delete() }
            saveUnlocked(context, items.filterNot { it.id == id })
            true
        }
    }

    private fun loadUnlocked(context: Context): List<SpeechHistoryItem> {
        val file = File(directory(context), HISTORY_FILE)
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
                        ),
                    )
                }
            }
        }.getOrElse { emptyList() }
    }

    private fun saveUnlocked(context: Context, items: List<SpeechHistoryItem>) {
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
                },
            )
        }
        val file = File(directory(context), HISTORY_FILE)
        val temp = File(file.parentFile, "$HISTORY_FILE.tmp")
        temp.writeText(array.toString())
        if (file.exists()) file.delete()
        if (!temp.renameTo(file)) {
            temp.copyTo(file, overwrite = true)
            temp.delete()
        }
    }
}
