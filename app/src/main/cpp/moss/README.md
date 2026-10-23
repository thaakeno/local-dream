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

## SM8850 HTP memory planning (alpha.72 follow-up)
The upstream pinned partitioner at 95b3090 packs **weights** right up to the
per-device `max_vram` cap without reserving scheduler/workspace memory. On the
2026-10-23 SM8850 log Qwen3.5 assigned 2580 MiB of weights to HTP0:0 and its
runtime required 2855.86 MiB, exceeding the 2662.40 MiB budget. Raising the
runtime limit is unsafe against the ~3.35GB v81 address-mapping window.

The MOSS-only staged upstream patch reserves 320 MiB per virtual HTP session
*when planning weight placement*. Runtime `max_vram` was 2.60 GiB in alpha.72,
with the original capacity checks active and no CPU inference fallback.
The original pinned git submodule remains unchanged. The same source patch is
applied from scratch on every native build and compiled by CI; see
`patches/0001-hexagon-layer-split-workspace-headroom.patch` and
`tests/test_htp_partition.py`.

This is a targeted correction to the **observed** deterministic budget failure,
not a claim of successful on-device MOSS image output. A 512x512 Turbo run must
still confirm text encoding, each of the eight diffusion steps, VAE decoding,
image artifact creation, and absence of DSPQueue mapping faults. Keep Breeze,
SDXL and permanent signing untouched.

## Alpha.73 — MOSS Turbo graph placement (2026-10-23)
The on-device alpha.72 retest **verified Qwen3.5 completion in 19.59 seconds**
and successful release of its per-device buffers. The next failure occurred
immediately as the Turbo-LoRA diffusion graph began graph-cut layer partitioning.
The old 2.60 GiB per-session budget combined with a 320 MiB planner reserve
allowed only `2 × (2662.4 − 320) = 4684.8 MiB` of graph weights, below the
~4.59 GiB LoRA-augmented diffusion model (~4700 MiB). That precludes a
two-way layer split even though the text encoder now fits.

MOSS-only managed `max_vram` is now **2.75 GiB per virtual HTP session**.
The *existing* 320 MiB packing reserve is preserved, leaving 4992 MiB total
planned capacity. This is still 367 MiB below the exact device-reported
3,337,998,336-byte FastRPC op mapping window per HTP session, and all upstream
runtime memory checks are preserved. No CPU/GPU inference fallback, no changes
to Qwen/SDXL/Breeze, and no changes to the original permanent signer/workflows.
New version: `3.0.0-alpha.73 (228)`.

The synthetic regression protects the two model-planning scenarios but does
not establish success of full 8-step inference. Confirm on-device separately:
Qwen conditioning, full DiT partition, each diffusion step, VAE decoding,
output image, and absence of `AEE_` / FastRPC errors. If allocation still
fails, collect the **native** MOSS graph-cut failure line and per-HTP memory
details; do not blindly increase beyond the FastRPC mapping window.

## Alpha.74 (229): SIGSEGV triage, not an unverified memory increase
The alpha.73 on-device log confirms Qwen conditioning succeeds in ~16.7 s,
then MOSS DiT raises a native SIGSEGV (code 139), fault address
`0x3eb3e60d3fb6d580`, after printing HTP split capacity information.
This is a **memory access violation**, not proof of insufficient NPU memory;
we do not increase the 2.75 GiB cap again. Upstream's graph-cut partitioner
has multiple distinct stages (param segmentation, placement, weight manager)
between the budget messages and execution. It is unsafe to guess which failed.

The staged pinned source now emits `[MOSS_SPLIT_TRACE]` breadcrumbs for these
stages and rejects a total partition weight requirement above the two NPU
weight capacities as a normal error, rather than running an impossible
placement. The MOSS server records `[MOSS_NATIVE_MAP] base=...` after the
engine loads, so a subsequent fatal PC can be expressed as a module offset
(`pc - base`) for symbolication. Original repo and all other engines remain
untouched; original CI workflows and signing key remain unchanged. There is
**no claim that this alone resolves the SIGSEGV** without the next device log.
