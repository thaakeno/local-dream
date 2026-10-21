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

    val SFT_Q4 = listOf(
        "leejet/MOSS-Image-SFT-GGUF/resolve/main/MOSS-Image-SFT-Q4_K.gguf|dit.gguf",
    ) + common

    val BASE_Q4 = listOf(
        "leejet/MOSS-Image-Base-GGUF/resolve/main/MOSS-Image-Base-Q4_K.gguf|dit.gguf",
    ) + common

    // MOSS's reference quality recipe. Accelerated community LoRAs need
    // separate fidelity verification and must not silently replace Base/SFT.
    const val STEPS = 50
    const val CFG = 3f
    const val FLOW_SHIFT = 5f

    /** A missing native engine never pretends the model is ready to run. */
    fun engineAvailable(context: Context): Boolean =
        File(context.applicationInfo.nativeLibraryDir, ENGINE_LIB).isFile
}
