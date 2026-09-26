package io.github.xororz.localdream.data

import android.content.Context
import android.util.Log
import java.io.File
import java.nio.file.Files

/**
 * Shared on-disk asset pool for the Qwen Image 2.1 family.
 *
 * Transformer precision and Viggle LoRA are orthogonal choices, so storing
 * every precision+adapter combination as a full package wastes many gigabytes.
 * This pool stores common encoder/VAE files, each transformer once, and each
 * LoRA once. Runtime model directories are assembled with hard links (symlink
 * fallback) so the native backend still sees its legacy filenames unchanged.
 */
object QwenFamilyStorage {
    const val SHARED_DIR_NAME = "_qwen21_shared"

    const val COMMON_LLM = "llm.gguf"
    const val COMMON_VISION = "llm_vision.gguf"
    const val COMMON_TOKENIZER = "tokenizer.json"
    const val COMMON_VAE = "vae.safetensors"

    const val DIT_Q4 = "dit_q4.gguf"
    const val DIT_Q8 = "dit_q8.gguf"
    const val DIT_FP8 = "dit_fp8.safetensors"

    const val LORA_R128 = "viggle_r128.safetensors"
    const val LORA_R256 = "viggle_r256.safetensors"

    private const val TAG = "QwenFamilyStorage"
    private val lock = Any()

    data class VariantSpec(
        val precision: String,
        val adapter: String,
        val transformerFile: String,
        val runtimeTransformerName: String,
    )

    private val specs = mapOf(
        "qwen_image_2_1" to VariantSpec("Q4_0", "", DIT_Q4, "dit.gguf"),
        "qwen_image_2_1_viggle_turbo_gguf" to
            VariantSpec("Q4_0", "r128", DIT_Q4, "dit.gguf"),
        "qwen_image_2_1_q4_viggle_r256" to
            VariantSpec("Q4_0", "r256", DIT_Q4, "dit.gguf"),
        "qwen_image_2_1_q8" to VariantSpec("Q8_0", "", DIT_Q8, "dit.gguf"),
        "qwen_image_2_1_q8_viggle_r128" to
            VariantSpec("Q8_0", "r128", DIT_Q8, "dit.gguf"),
        "qwen_image_2_1_q8_viggle_r256" to
            VariantSpec("Q8_0", "r256", DIT_Q8, "dit.gguf"),
        "qwen_image_2_1_fp8" to
            VariantSpec("FP8", "", DIT_FP8, "dit.safetensors"),
        "qwen_image_2_1_fp8_viggle_r128" to
            VariantSpec("FP8", "r128", DIT_FP8, "dit.safetensors"),
        "qwen_image_2_1_viggle_turbo" to
            VariantSpec("FP8", "r256", DIT_FP8, "dit.safetensors"),
    )

    fun sharedDir(context: Context): File =
        File(Model.getModelsDir(context), SHARED_DIR_NAME).apply { mkdirs() }

    fun variantSpec(modelId: String): VariantSpec? = specs[modelId]

    fun localName(entry: String): String {
        val remote = entry.substringBefore('|')
        return entry.substringAfter('|', remote.substringAfterLast('/'))
    }

    fun isPackageReady(context: Context, packageFiles: List<String>): Boolean {
        migrateLegacyAssets(context)
        val dir = sharedDir(context)
        return packageFiles.all { entry ->
            File(dir, localName(entry)).let { it.isFile && it.length() > 0L }
        }
    }

    fun isPrecisionReady(context: Context, precision: String): Boolean {
        migrateLegacyAssets(context)
        val file = when (precision) {
            "Q4_0" -> DIT_Q4
            "Q8_0" -> DIT_Q8
            "FP8" -> DIT_FP8
            else -> return false
        }
        val dir = sharedDir(context)
        return commonFiles().all { File(dir, it).isFile } &&
            File(dir, file).isFile
    }

    fun isAdapterReady(context: Context, adapter: String): Boolean {
        if (adapter.isBlank()) return true
        migrateLegacyAssets(context)
        val file = when (adapter) {
            "r128" -> LORA_R128
            "r256" -> LORA_R256
            else -> return false
        }
        return File(sharedDir(context), file).isFile
    }

    /**
     * Return a native-compatible model directory for a built-in Qwen variant.
     * Custom qwen21 imports are intentionally not handled here.
     */
    fun prepareRuntimeDir(context: Context, modelId: String): File? {
        val spec = specs[modelId] ?: return null
        synchronized(lock) {
            migrateLegacyAssetsLocked(context)
            val shared = sharedDir(context)
            val required = commonFiles() + spec.transformerFile +
                listOfNotNull(adapterFile(spec.adapter))
            val missing = required.filterNot {
                File(shared, it).let { file -> file.isFile && file.length() > 0L }
            }
            require(missing.isEmpty()) {
                "Qwen family assets missing for $modelId: ${missing.joinToString()}"
            }

            val runtime = File(context.filesDir, "runtime_models/qwen21/$modelId")
            if (runtime.exists()) runtime.deleteRecursively()
            runtime.mkdirs()

            link(shared, runtime, COMMON_LLM, COMMON_LLM)
            link(shared, runtime, COMMON_VISION, COMMON_VISION)
            link(shared, runtime, COMMON_TOKENIZER, COMMON_TOKENIZER)
            link(shared, runtime, COMMON_VAE, COMMON_VAE)
            link(shared, runtime, spec.transformerFile, spec.runtimeTransformerName)

            adapterFile(spec.adapter)?.let { adapter ->
                link(shared, runtime, adapter, "turbo_lora.safetensors")
                File(runtime, "inference_profile.conf").writeText(
                    Model.QWEN_IMAGE_2_1_VIGGLE_TURBO_PROFILE.trim() + "\n",
                )
            }
            File(runtime, "QWEN_IMAGE_2_1").createNewFile()
            return runtime
        }
    }

    /**
     * Adopt files from alpha.34 and earlier without copying multi-gigabyte
     * weights. Existing downloads become shared family assets immediately.
     */
    fun migrateLegacyAssets(context: Context) {
        synchronized(lock) {
            migrateLegacyAssetsLocked(context)
        }
    }

    private fun migrateLegacyAssetsLocked(context: Context) {
        val shared = sharedDir(context)

        val allVariantIds = specs.keys.toList()
        adoptFirst(context, shared, COMMON_LLM, allVariantIds, COMMON_LLM)
        adoptFirst(context, shared, COMMON_VISION, allVariantIds, COMMON_VISION)
        adoptFirst(context, shared, COMMON_TOKENIZER, allVariantIds, COMMON_TOKENIZER)
        adoptFirst(context, shared, COMMON_VAE, allVariantIds, COMMON_VAE)

        adoptFirst(
            context,
            shared,
            DIT_Q4,
            listOf(
                "qwen_image_2_1",
                "qwen_image_2_1_viggle_turbo_gguf",
                "qwen_image_2_1_q4_viggle_r256",
            ),
            "dit.gguf",
        )
        adoptFirst(
            context,
            shared,
            DIT_Q8,
            listOf(
                "qwen_image_2_1_q8",
                "qwen_image_2_1_q8_viggle_r128",
                "qwen_image_2_1_q8_viggle_r256",
            ),
            "dit.gguf",
        )
        adoptFirst(
            context,
            shared,
            DIT_FP8,
            listOf(
                "qwen_image_2_1_fp8",
                "qwen_image_2_1_fp8_viggle_r128",
                "qwen_image_2_1_viggle_turbo",
            ),
            "dit.safetensors",
        )

        adoptFirst(
            context,
            shared,
            LORA_R128,
            listOf(
                "qwen_image_2_1_viggle_turbo_gguf",
                "qwen_image_2_1_q8_viggle_r128",
                "qwen_image_2_1_fp8_viggle_r128",
            ),
            "turbo_lora.safetensors",
        )
        adoptFirst(
            context,
            shared,
            LORA_R256,
            listOf(
                "qwen_image_2_1_q4_viggle_r256",
                "qwen_image_2_1_q8_viggle_r256",
                "qwen_image_2_1_viggle_turbo",
            ),
            "turbo_lora.safetensors",
        )
    }

    private fun commonFiles() = listOf(
        COMMON_LLM,
        COMMON_VISION,
        COMMON_TOKENIZER,
        COMMON_VAE,
    )

    private fun adapterFile(adapter: String): String? = when (adapter) {
        "r128" -> LORA_R128
        "r256" -> LORA_R256
        else -> null
    }

    private fun adoptFirst(
        context: Context,
        shared: File,
        targetName: String,
        legacyIds: List<String>,
        legacyName: String,
    ) {
        val target = File(shared, targetName)
        if (target.isFile && target.length() > 0L) return

        val source = legacyIds.asSequence()
            .map { File(File(Model.getModelsDir(context), it), legacyName) }
            .firstOrNull { it.isFile && it.length() > 0L }
            ?: return

        runCatching {
            createLink(source, target)
            Log.i(TAG, "Adopted ${source.parentFile?.name}/$legacyName as $targetName")
        }.onFailure {
            Log.w(TAG, "Could not adopt $legacyName as $targetName: ${it.message}")
        }
    }

    private fun link(shared: File, runtime: File, sourceName: String, targetName: String) {
        createLink(File(shared, sourceName), File(runtime, targetName))
    }

    private fun createLink(source: File, target: File) {
        require(source.isFile) { "Missing shared asset ${source.absolutePath}" }
        target.parentFile?.mkdirs()
        if (target.exists() || Files.isSymbolicLink(target.toPath())) target.delete()

        runCatching {
            Files.createLink(target.toPath(), source.toPath())
        }.recoverCatching {
            Files.createSymbolicLink(target.toPath(), source.toPath())
        }.getOrThrow()
    }
}
