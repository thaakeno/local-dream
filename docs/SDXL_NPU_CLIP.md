# SDXL CLIP on Snapdragon NPU — exact graph contract

The legacy Local Dream SDXL bundles ship MNN CPU CLIP encoders, not QNN
CLIP context binaries. Changing an execution flag **does not** convert them.

The native SDXL pipeline now recognizes two additional per-model files:
- `clip_qnn.bin` — CLIP-L, a QNN HTP context
- `clip2_qnn.bin` — CLIP-G, a QNN HTP context

Both must exist in the SDXL model folder. With both present, the pipeline
executes these QNN graphs **instead of** running the MNN sessions, and
closes the text graph handles after prompt encoding.

## Required compatibility

These must be compiled from the **same CLIP weights**, hidden layer and
pre/post-processing as the MNN models already present. Graphs must expose:

| Context | Input (floating tensor named `input_embedding`) | Output |
|---|---|---|
| CLIP-L | `[1,77,768]` token + positional *embeddings*, **not token IDs** | single `[1,77,768]` hidden tensor |
| CLIP-G | `[1,77,1280]` token + positional embeddings | `[1,77,1280]` hidden tensor, followed by `[1,77,1280]` pooled tensor (or `[1,1280]` pooled vector) |

The second output must match the reference pooled projection/hidden
semantics expected by the existing SDXL pipeline. Output tensor order
matters. Encoder variants using `[1,77]` INT32 token IDs do not qualify.

The runtime **rejects incompatible shapes** rather than feeding garbage
conditioning into the diffusion model.

## Validation gate

Before shipping an NPU CLIP binary, compare its hidden states and pooled
conditioning numerically against the existing MNN reference using fixed
tokens (including EOS position, long prompts, prompt weights, and negative
prompts), then perform image-level A/B tests. Confirm that the model loads
and executes with QAIRT 2.51 on SM8850/V81 using the **efficient** power
profile.

`LOCALDREAM_SDXL_CLIP_BACKEND=npu` makes missing context binaries a hard
error. By default the engine prefers the QNN contexts when available and
logs an explicit warning if the older CPU MNN fallback is used.

No QNN CLIP contexts have yet been compiled and validated for this
repository. **Do not claim NPU text encoding until both matching binaries
pass the validation gate.**
