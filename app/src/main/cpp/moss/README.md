# MOSS-Image isolated runtime

MOSS must **not** be compiled into or replace the September 26 Qualcomm
stable-diffusion.cpp fork. That engine lacks MOSS and has a customized C API
(typed progress, phases, tokenizer count and conditioning caches).
The official leejet engine supporting MOSS must be built as a separate
`libmoss_engine.so` against a pinned version of stable-diffusion.cpp and
communicate through the existing `DitEngine.h` ABI with a bridge.

Pinned official source for the first integration:
`leejet/stable-diffusion.cpp@95b3090664d019fcf8fa032a44d1d4793ba244b8`
(October 18, 2026, includes MOSS and split QKV weights).

Assets for SFT and Base are in `MossImagePackages.kt`, with their own
tokenizer.json, Qwen3.5-9B, FLUX.2 VAE and MOSS Q4_K transformer.
Do not use Qwen Image 2.1 tokenizer/encoder for this model.

Reference: 1024x1024, 50 steps, CFG 3, Euler, scheduler flow shift 5.
Dimensions divisible by 16; the negative CFG branch skips layer 22 inside
official stable-diffusion.cpp. Optional 8-step DMD LoRA must be a clearly
separate user-selectable profile, never silently applied to the original model.

The Android catalog is gated by presence of the compiled, packaged engine;
metadata alone does **not** make MOSS executable. Full release also requires
an ABI smoke test, package download tests, runtime memory measurements and
output comparison with stable-diffusion.cpp desktop references.

Sources:
https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/moss_image.md
https://huggingface.co/leejet/MOSS-Image-SFT-GGUF
https://huggingface.co/leejet/MOSS-Image-Base-GGUF
https://huggingface.co/Comfy-Org/MOSS-Image
