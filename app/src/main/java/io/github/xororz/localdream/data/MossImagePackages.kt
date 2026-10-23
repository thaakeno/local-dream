package io.github.xororz.localdream.data

import android.content.Context
import java.io.File

/**
 * File contract for the separate MOSS-Image engine. The September 2026
 * stable-diffusion.cpp fork in libdit_engine.so does NOT support MOSS.
 *
 * Keep source urls, filenames and tokenizer independent of Qwen Image 2.1:
 * Qwen3.5 has a different vocabulary from the embedded Qwen tokenizer.
 */
object MossImagePackages {
    const val DIT_KIND = "moss"
    const val ENGINE_LIB = "libmoss_engine.so"
    const val MODEL_MARKER = "MOSS_IMAGE"

    private val common = listOf(
        "unsloth/Qwen3.5-9B-GGUF/resolve/main/Qwen3.5-9B-Q4_K_M.gguf|llm.gguf",
        "Comfy-Org/MOSS-Image/resolve/main/vae/flux2-vae.safetensors|vae.safetensors",
        "OpenMOSS-Team/MOSS-Image-Base/resolve/main/tokenizer/tokenizer.json|tokenizer.json",
    )

    // The 8-step DMD adapter is already merged into these molbal GGUFs.
    // Download one checkpoint per model. Common encoder/VAE/tokenizer files
    // are shared safely by the downloader when already installed.
    const val MERGED_TURBO_FAST_ID = "moss_image_sft_turbo_merged_q4_0"
    const val MERGED_TURBO_QUALITY_ID = "moss_image_sft_turbo_merged_q4_k_m"

    val MERGED_TURBO_FAST_Q4_0 = listOf(
        "molbal/MOSS-Image-GGUF/resolve/main/moss_image_sft_turbo-Q4_0.gguf|dit.gguf",
    ) + common

    val MERGED_TURBO_QUALITY_Q4_K_M = listOf(
        "molbal/MOSS-Image-GGUF/resolve/main/moss_image_sft_turbo-Q4_K_M.gguf|dit.gguf",
    ) + common

    val SHARED_RUNTIME_FILES = setOf("llm.gguf", "vae.safetensors", "tokenizer.json")
    val KNOWN_MODEL_IDS = listOf(
        "moss_image_sft_q4", "moss_image_base_q4",
        "moss_image_sft_turbo_q4", "moss_image_base_turbo_q4",
        MERGED_TURBO_FAST_ID, MERGED_TURBO_QUALITY_ID,
    )

    val SFT_Q4 = listOf(
        "leejet/MOSS-Image-SFT-GGUF/resolve/main/MOSS-Image-SFT-Q4_K.gguf|dit.gguf",
    ) + common

    val BASE_Q4 = listOf(
        "leejet/MOSS-Image-Base-GGUF/resolve/main/MOSS-Image-Base-Q4_K.gguf|dit.gguf",
    ) + common

    // Community 8-step adapters; separate packages, never silently enabled.
    // The 4-step data-space DMD sampler is NOT supported by the stock engine.
    val SFT_TURBO_Q4 = SFT_Q4 + listOf(
        "lodestones/MOSS-Image-SFT-8-steps-dmd/resolve/main/lora.safetensors|moss_turbo_lora.safetensors",
    )
    val BASE_TURBO_Q4 = BASE_Q4 + listOf(
        "lodestones/MOSS-Image-Base-8-steps-dmd/resolve/main/lora.safetensors|moss_turbo_lora.safetensors",
    )

    // MOSS's reference quality recipe. Accelerated community LoRAs need
    // separate fidelity verification and must not silently replace Base/SFT.
    const val STEPS = 50
    const val CFG = 3f
    const val FLOW_SHIFT = 5f

    /** A missing native engine never pretends the model is ready to run. */
    fun engineAvailable(context: Context): Boolean =
        File(context.applicationInfo.nativeLibraryDir, ENGINE_LIB).isFile
}
