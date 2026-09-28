package io.github.xororz.localdream.service

import android.content.ContentValues
import android.content.Context
import android.net.Uri
import android.os.Environment
import android.provider.MediaStore
import java.io.File
import org.json.JSONArray
import org.json.JSONObject

data class MusicHistoryItem(
    val id: String,
    val filePath: String,
    val style: String,
    val lyrics: String,
    val score: String,
    val lmSeed: Long,
    val acousticSeed: Long,
    val targetSeconds: Int,
    val elapsedMillis: Long,
    val createdAtMillis: Long,
    val format: String,
    val solver: String,
    val steps: Int,
    val planning: String,
    val modelId: String,
) {
    val file: File get() = File(filePath)
}

object MusicHistoryStore {
    private const val DIR = "music"
    private const val INDEX = "history.json"
    private const val MAX_ITEMS = 100

    private fun dir(context: Context) = File(context.filesDir, DIR).apply { mkdirs() }
    private fun index(context: Context) = File(dir(context), INDEX)

    @Synchronized
    fun load(context: Context): List<MusicHistoryItem> {
        val file = index(context)
        if (!file.isFile) return emptyList()
        return runCatching {
            val arr = JSONArray(file.readText())
            buildList {
                for (i in 0 until arr.length()) {
                    val o = arr.optJSONObject(i) ?: continue
                    val item = MusicHistoryItem(
                        id = o.optString("id"),
                        filePath = o.optString("filePath"),
                        style = o.optString("style"),
                        lyrics = o.optString("lyrics"),
                        score = o.optString("score"),
                        lmSeed = o.optLong("lmSeed", -1L),
                        acousticSeed = o.optLong("acousticSeed", -1L),
                        targetSeconds = o.optInt("targetSeconds", 0),
                        elapsedMillis = o.optLong("elapsedMillis", 0L),
                        createdAtMillis = o.optLong("createdAtMillis", 0L),
                        format = o.optString("format", "wav32"),
                        solver = o.optString("solver", "dpmpp_2m"),
                        steps = o.optInt("steps", 32),
                        planning = o.optString("planning", "full"),
                        modelId = o.optString("modelId"),
                    )
                    if (item.id.isNotBlank() && item.file.isFile) add(item)
                }
            }.sortedByDescending { it.createdAtMillis }
        }.getOrElse { emptyList() }
    }

    @Synchronized
    fun add(context: Context, item: MusicHistoryItem) {
        val next = (listOf(item) + load(context).filterNot { it.id == item.id })
            .take(MAX_ITEMS)
        write(context, next)
    }

    @Synchronized
    fun delete(context: Context, id: String, deleteFile: Boolean = true) {
        val current = load(context)
        val victim = current.firstOrNull { it.id == id }
        if (deleteFile) runCatching { victim?.file?.delete() }
        write(context, current.filterNot { it.id == id })
    }

    private fun write(context: Context, items: List<MusicHistoryItem>) {
        val arr = JSONArray()
        items.forEach { item ->
            arr.put(JSONObject().apply {
                put("id", item.id)
                put("filePath", item.filePath)
                put("style", item.style)
                put("lyrics", item.lyrics)
                put("score", item.score)
                put("lmSeed", item.lmSeed)
                put("acousticSeed", item.acousticSeed)
                put("targetSeconds", item.targetSeconds)
                put("elapsedMillis", item.elapsedMillis)
                put("createdAtMillis", item.createdAtMillis)
                put("format", item.format)
                put("solver", item.solver)
                put("steps", item.steps)
                put("planning", item.planning)
                put("modelId", item.modelId)
            })
        }
        val target = index(context)
        val tmp = File(target.parentFile, target.name + ".tmp")
        tmp.writeText(arr.toString())
        if (!tmp.renameTo(target)) {
            target.writeText(arr.toString())
            tmp.delete()
        }
    }

    fun mimeType(format: String): String =
        if (format == "mp3") "audio/mpeg" else "audio/wav"

    fun extension(format: String): String =
        if (format == "mp3") "mp3" else "wav"

    fun exportToMusic(context: Context, item: MusicHistoryItem): Uri {
        require(item.file.isFile) { "Generated audio file is missing" }
        val ext = extension(item.format)
        val title = item.style
            .replace(Regex("""[^A-Za-z0-9 _-]"""), "")
            .trim()
            .take(48)
            .ifBlank { "YuE2" }
        val displayName = "$title-${item.createdAtMillis}.$ext"
        val values = ContentValues().apply {
            put(MediaStore.Audio.Media.DISPLAY_NAME, displayName)
            put(MediaStore.Audio.Media.MIME_TYPE, mimeType(item.format))
            put(MediaStore.Audio.Media.RELATIVE_PATH, Environment.DIRECTORY_MUSIC + "/Local Dream")
            put(MediaStore.Audio.Media.IS_PENDING, 1)
        }
        val resolver = context.contentResolver
        val uri = resolver.insert(MediaStore.Audio.Media.EXTERNAL_CONTENT_URI, values)
            ?: error("Could not create Music entry")
        try {
            resolver.openOutputStream(uri, "w")!!.use { out ->
                item.file.inputStream().use { input -> input.copyTo(out, 1024 * 1024) }
            }
            values.clear()
            values.put(MediaStore.Audio.Media.IS_PENDING, 0)
            resolver.update(uri, values, null, null)
            return uri
        } catch (t: Throwable) {
            resolver.delete(uri, null, null)
            throw t
        }
    }
}
