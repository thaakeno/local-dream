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

## Alpha.75 (230): avoid broken Hexagon virtual-session buffer cloning
Alpha.74 successfully partitioned the 43-segment MOSS graph (4308.0 MiB of
weights), loaded all 378 diffusion tensors, and allocated 2,569.67/2,021.80
MB on the two HTP sessions. Then `ggml_backend_sched_split_inputs_grow`
reported extra cross-backend inputs and the process SIGSEGV'd.

Disassembling **the actual alpha.74 signed APK** at the reported
`pc=0x7852d690a8` minus `[MOSS_NATIVE_MAP] base=0x7850a36000`
gives ELF `0x23330a8`: ARM64 `ldadd` on the reference count of the
`ggml_hexagon_shared_buffer::mem` shared_ptr in
`ggml_hexagon_session::clone_buffer()` (pinned leejet/ggml file
`src/ggml-hexagon/ggml-hexagon.cpp`, lines 4710-4716). The fault address
is an invalid refcount pointer, *not* a budget-check or scheduler-growth error.

MOSS's separate staged pinned ggml is therefore modified **only** at the
virtual HTP-to-HTP async-copy entry point: for two *different virtual*
sessions on the same physical Hexagon device, reject the unsafe async
shortcut by returning `false`, letting the **existing upstream scheduler**
synchronize the source and destination and copy tensor data through
ggml's normal `ggml_backend_tensor_copy` host staging. Only tensor
**transfers** use the host; neither Qwen nor the DiT performs CPU
inference. No change for same-session fast copies, distinct physical
NPUs, QNN/SDXL/Breeze/other native engines, HTP limits, model weights,
app signing, or GitHub workflows.

The staged source test verifies the exact pinned source receives the
guard before its unsafe `clone_buffer` call. There is no on-device
8-step success claim until alpha.75 is tested on SM8850.

## Alpha.76 (231) — upstream single-HTP segmented architecture, October 23

Research: Nightmare Mobile, a public Local Dream-based app, reports actual
MOSS-Image SFT generation on Snapdragon 8 Elite in ~50s at 512×512, and a
merged Turbo GGUF variant in ~40s. The reference launches its default DiT
backend on ONE HTP session and stages weights from disk when a managed
memory limit is applied. Source: github.com/AbrahamPaulJ/nightmare-mobile.
Its fused positional-encoding fast path is a further optimisation, not used
in our baseline verification.

The exact pinned stable-diffusion.cpp GGMLRunner enables segmented execution
only if `!is_multi_device()`. Previous MOSS forced dual virtual sessions,
which *disabled* this path and caused crashes in Hexagon buffer lifetime
handling during cross-session tensor transfers. It also used
`te=disk,vae=disk`, leaving diffusion weight residency unbounded.
Increasing the budget or bypassing a single async copy did not fix it.

The MOSS runtime now chooses ONLY HTP0:0 for text encoding, diffusion and
VAE, with `all=disk` parameter streaming and a 2.60 GiB per-runner managed
budget to segment large graphs. Device *registration* still enumerates
HTP0:0 and HTP0:1 because restricting it to one previously failed
initialization; unused HTP0:1 is not selected as a compute backend.
Inference remains NPU-only. No changes to Breeze, SDXL or Qwen Image.

Legacy dual-device partition/cross-copy patches are **no longer applied**.
A pristine staged copy of the exact pinned upstream and its ggml submodule
builds with a NEW native CMake directory, preventing stale dual-device build
objects from getting repackaged. Regression assertions verify single-device
runtime/backend, all=disk, a managed budget, upstream segmentation support
and absence of old custom ggml changes.

The original GitHub branch, CI, package identity and permanent signer are
unchanged. Full 8-step generation and speed remain UNVERIFIED on SM8850
until on-device confirmation.

## Alpha.77 (232) — restore native Hexagon acceleration (October 23)

Comparing our launch with Nightmare Mobile and the pinned ggml driver
identified two accidental performance inhibitors still present after the
alpha.76 single-backend fix. Android's MOSS-only launcher explicitly set
`GGML_HEXAGON_NHMX=0` (HVX only, no hardware matrix engine) and
`GGML_HEXAGON_OPFUSION=0` (all operator fusions disabled).
The exact pinned ggml's defaults are **HMX enabled**, **fusion enabled**,
1280 operations per batch, 8 queued batches, and automatic thread counts.
The old launcher additionally forced 1024 operations and 16 queued batches.

Those experimental overrides have been **removed**, not replaced with
new arbitrary tuning values. Hexagon now measures the device capabilities
and automatically chooses the eligible HMX or HVX execution path. A default
power profile is used for benchmark comparability. No special fallback to
CPU is requested; strict MOSS compute is still on HTP0:0, using `all=disk`
segmented streaming and its 2.60 GiB budget. The runtime library is still
the standalone, unmodified pinned leejet/ggml in alpha.76.

Nightmare's separate QK/RoPE fused operation (`ggml_qknorm_rope`) does NOT
exist in our pinned leejet/ggml snapshot. Applying its C++ patch directly
is incompatible, so we did not silently graft untested Qualcomm DSP code
or replace our GGML with the older Nightmare stack. A safe port needs
a compatible ggml implementation, matched DSP skels and output parity
tests after single-HTP end-to-end generation succeeds.

Run-time speedup is **not yet measured on the user's SM8850**. The exact
Nightmare benchmark uses Snapdragon 8 Elite (S25 Ultra), not Elite Gen 5.

## Source-tree cleanup

The abandoned dual-device patch and its ad hoc instrumentation scripts have
been removed from the active source tree. The historical commits remain
available through Git. The only MOSS-specific regression script kept in the
build path is `tests/test_single_htp_segmentation.py`, which asserts that
the staged source is unmodified upstream and that memory segmentation and
driver-default accelerators are selected. The standalone MOSS engine still
uses the independent `moss-stable-diffusion.cpp` submodule with no changes
to the regular Qwen/SDXL DiT dependencies.

## Alpha.78 (233): Nightmare-style 8-step merged Turbo checkpoints

The stock MOSS catalog now offers the two upstream molbal
`molbal/MOSS-Image-GGUF` Apache-2.0 checkpoints:

* **MOSS-Image Turbo Q4_0 (Fast)** uses
  `moss_image_sft_turbo-Q4_0.gguf` (~3.33 GB) with DMD merged.
  Nightmare Mobile's similar Q4_0 checkpoint reports ~40s/512² on 8 Elite,
  not a measured speed on SM8850.
* **MOSS-Image Turbo Q4_K_M (Quality)** uses
  `moss_image_sft_turbo-Q4_K_M.gguf` (~3.60 GB) with DMD merged.
  Finer quantization may preserve more detail, but faster/quality comparison
  and SM8850 NPU compatibility are not yet established.

Each has its own independent model ID and directory; both default to
8 Euler steps and CFG 1. The native bridge refuses to apply an extra
runtime DMD LoRA to these merged models even if a stale adapter file
exists. The previous SFT/Base options and explicit runtime 8-step LoRAs
remain intact.

The downloader reuses the exact Qwen3.5 GGUF, FLUX VAE, and tokenizer
across any completed MOSS package with matching remote content size:
Android's same-filesystem hard link means no extra storage, no duplicated
download, and independent deletion. Missing donors/failed link fall back
to normal resumable Hub downloads. Model marker is checked before linking.

Nightmare's separate fused `GGML_OP_QKNORM_ROPE` requires changes
through ggml header, CPU reference, host backend, Hexagon ABI and
a MATCHED recompiled DSP skel; it is not compatible with pinned leejet/ggml,
so is not silently claimed or exposed. We retain upstream HMX/fusion
and segmented single-HTP architecture as a correctness baseline.

No image was successfully generated yet on SM8850 with this release.

## Alpha.79 (234) — phase-one MOSS fused Q/K RMSNorm + RoPE

**Source provenance:** applies the exact Nightmare Mobile patches
`backend-patches/dit/028-moss-image.sdcpp.patch` and
`backend-patches/dit/035-moss-fused-rope.sdcpp.patch` to an ISOLATED copy
of `happyyzy/stable-diffusion.cpp@a63954dc...`. Its
`ggml@f8758628...` already includes the `GGML_OP_QKNORM_ROPE` factory,
CPU/backend implementation and Hexagon support. This is the EXISTING
stable-diffusion source pinned by Local Dream's classic DiT, not an
unverified or newly added public dependency. Its existing checkout remains
pristine. The obsolete standalone leejet MOSS gitlink remains checked out
for history but is no longer part of the compiled MOSS engine.

Why two patches: the happyyzy source alone lacks MOSS. 028 adds MOSS and
Qwen3.5 tokenizer integration; 035 fuses norm/position in its 42 diffusion
blocks and two text blocks. Both patches apply in order with fuzz disabled;
a clean native build directory prevents stale objects. The MOSS bridge
provides Qwen3.5's tokenizer via `SD_QWEN35_TOKENIZER`, the upstream
patch's actual supported API (not the newer leejet-only `p.tokenizer`).

**No UI toggle.** Fused RoPE is requested by default on MOSS; the patch
automatically falls back to composed operations when the backend does not
support the fused op, tensor shape is incompatible or precision differs.
Developers can create `files/moss_composed_rope.flag` inside the app's
private data directory on a debuggable build to force the baseline
(`NM_MOSS_COMPOSED_ROPE=1`). Restart the MOSS backend after changing it.
The flag affects MOSS only and does not touch Qwen, Z-Image, SDXL or YuE2.
A kernel SIGSEGV cannot safely be caught within native inference; use a
separate-process rollout and preserve old builds for rollback.

**Measuring:** log `[MOSS_BENCH] rope=... resolution=... steps=...
seed=... total_s=...` without disclosing prompts. For A/B compare
identical GGUF, prompt, fixed seed, resolution, steps and device state,
alternate baseline/fused, repeat multiple runs and use medians. Upstream
graph timings separately show text encoder, diffusion steps and VAE.
First use 512x512, then 1024x1024 only after 512 completes.

The 3–4× reports are Nightmare's developer benchmarks on a different
chip, NOT yet achieved or confirmed on SM8850. No user-facing 5× claim.

Other engines, shared runtime, model downloads, signing, package ID and
existing GitHub workflows remain unchanged.

## Alpha.80 (235): real MOSS progress (2026-10-23)

The native untyped sd progress callback was reporting 8/8 before the actual
8 diffusion passes had finished; both runtime-LoRA and merged Q4_0 exhibited
this. The user's merged Turbo log shows 43 graph segments, repeated passes,
and an explicit UI-initiated cancellation, NOT a completed decode.

MOSS alone now observes confirmed `qwen3* executing segment N/T` and
`MOSS-Image executing segment N/T` host graph-cut log events. Entering
segment N means N-1 finished. Each repeat from 43 to 1 increments the
actual diffusion pass, and the UI shows current pass/total and current
completed layers/pass without inventing a time remaining estimate. Qwen
encoding gets its own layer progress. Native 'sampling completed' marks
transition to the indeterminate VAE decode stage. Native success still
requires a valid image; SSE complete is emitted only after RGB output is
returned and encoded. The previous callback now deliberately ignores
untyped step counters. No user toggle or changes to other model engines.

This prevents incorrect 8/8 UI; it does not assert that MOSS is faster or
guarantee completion on the SM8850. Host logs are currently the only
available supported source of per-segment truth without introducing a new
ABI callback into third-party native code.

## Alpha.81 (236) — MOSS v81 weight residency and per-step haptics

Alpha.80 on POCO F8 Ultra / SM8850, merged Turbo Q4_0, 8 Euler steps:
1024² native generation 162.164 s, of which 143.47 s sampling;
512² 72.789 s, of which 56.42 s sampling. At 1024² the first
20/43 transformer segments used already-resident parameters (~5.7 s),
but later segments repeatedly loaded 8 GGUF tensors (~0.20 s per
upload), costing nearly half of each ~17 s denoising pass. MOSS's
native compute buffer peak was 677.56 MB at 1024² and 183.97 MB at 512².

For SM8850 only, the MOSS backend can now use a headroom-constrained
budget of HTP0:0=2.84 GiB at 1024² (2.88 GiB at 512²) instead of
always capping to 2.60 GiB; more transformer weights can remain
resident between sampler passes. Values are exact allowlisted per
MOSS bridge and NEVER affect other backends. Only one physical/virtual
HTP is used for compute; no dual-session split, no CPU fallback.
These are planned/estimated gains, NOT yet confirmed on hardware.
The per-device default for other chips remains 2.60 GiB.

Settings → MOSS NPU performance → Faster weight residency toggles
the SM8850 profile. Turn it off for the previously tested 2.60 GiB
profile and restart MOSS. If a fast-budget context fails to initialize,
the native bridge retries with compatibility budget; unsafe inference
SIGSEGVs cannot be recovered automatically. No quantization changes,
resolution reduction, step-skipping, output interpolation, or other
quality-altering shortcuts.

For MOSS, the Android haptic pulse now fires when the number of
COMPLETED sampler passes increments, not on individual native graph
segments, Qwen encoder segments, or VAE decoding. The stage-specific
generation card additionally renders a small 8-part step meter.
The master haptics and progress-pulse settings still work normally.

A committed source change and successful CI/native build cannot prove
a real-device speedup. The device log reports the exact MOSS budget;
validate same-seed identical GGUF 1024² runs in both modes,
alternating test order with thermals controlled, then compare medians.

## Alpha.82 (237) — eliminate idle tensor-loader sleeps

The SM8850 1024² log exposed recurrent `loading 8/378 tensors` operations
taking exactly ~0.20 s, despite reporting only ~0.01–0.02 s to copy to
HTP. The pinned happyyzy source `src/model_loader.cpp` explicitly sleeps
**200 milliseconds** in its worker-supervisor loop when the batch contains
more than four tensors. During 1024², layers 21–42 reload repeatedly across
all eight sampler passes; the sleep is overhead unrelated to model math.

The isolated MOSS staged-engine source receives
`036-moss-fast-tensor-loader-poll.patch`: 200 ms supervisory polling becomes
5 ms. Worker completion is still joined before any tensor can be used,
and all tensor reads/copies, weights, algorithms, scheduler, resolution,
denoising steps, seed and NPU architecture are unchanged. The patch never
touches other engines or the pristine shared upstream gitlink.

This is a *testable speed optimization*, not a measured speed guarantee:
the theoretical avoidable waiting can be tens of seconds over a high-res
eight-step run, but actual end-to-end improvement and DSP stability still
require same-seed comparison on SM8850.

The final MOSS sampling-pass vibration is also delivered at the transition
to VAE decode when the last graph-completion state was conflated by Compose.
A cancelled run resets its sampling-haptic counter. The model title and
description now use text ellipsis rather than an off-screen scroll viewport.
