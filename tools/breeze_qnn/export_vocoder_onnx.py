#!/usr/bin/env python3
"""
Export the official Breeze TTS 2 / Qwen3-TTS 12 Hz decoder as one static
causal vocoder graph for QAIRT/QNN HTP compilation.

Why a static wrapper exists instead of tracing decoder.forward directly:
Transformers 4.57 builds its causal/sliding mask through torch.vmap. That path
is correct in eager mode but is not JIT/ONNX traceable and fails with
"_Map_base::at". We therefore precompute the exact eager sliding-window mask
and rotary embedding once for the fixed 64-frame graph, then call the original
decoder layers directly. A parity check against the untouched eager decoder is
mandatory before export.

The graph input is float32 for the QNN SampleApp runtime. Values are exact
integer code ids represented as floats and are cast to int32 inside the graph
before the codebook gathers.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--frames", type=int, default=64)
    args = ap.parse_args()

    from transformers.masking_utils import create_sliding_window_causal_mask
    from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
        Qwen3TTSTokenizerV2Model,
    )

    model_dir = Path(args.model_dir)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    model = Qwen3TTSTokenizerV2Model.from_pretrained(
        model_dir,
        local_files_only=True,
        torch_dtype=torch.float32,
    )
    decoder = model.decoder.eval().float()

    # Force the mathematically explicit attention implementation both for the
    # eager reference and the static export graph.
    decoder.config._attn_implementation = "eager"
    decoder.pre_transformer.config._attn_implementation = "eager"

    pt = decoder.pre_transformer
    frames = int(args.frames)
    position_ids = torch.arange(frames, dtype=torch.long).unsqueeze(0)
    cache_position = torch.arange(frames, dtype=torch.long)
    dummy_hidden = torch.zeros(
        (1, frames, int(pt.config.hidden_size)),
        dtype=torch.float32,
    )

    # Build the exact mask using the upstream helper outside tracing. This
    # preserves Qwen's sliding-window convention without exporting the vmap
    # mask factory itself.
    static_mask = create_sliding_window_causal_mask(
        config=pt.config,
        input_embeds=dummy_hidden,
        attention_mask=None,
        cache_position=cache_position,
        past_key_values=None,
        position_ids=position_ids,
    )
    if static_mask is None:
        # For a sequence shorter than the configured window, Transformers can
        # legally skip the explicit mask for SDPA, but eager attention cannot.
        neg = torch.finfo(torch.float32).min
        q = torch.arange(frames)[:, None]
        k = torch.arange(frames)[None, :]
        window = int(getattr(pt.config, "sliding_window", frames))
        allowed = (k <= q) & (k > q - window)
        static_mask = torch.where(
            allowed,
            torch.zeros((), dtype=torch.float32),
            torch.full((), neg, dtype=torch.float32),
        )[None, None, :, :]
    if static_mask.ndim != 4:
        raise RuntimeError(f"unexpected static mask shape: {tuple(static_mask.shape)}")

    # RoPE positions are also fixed for this graph. Precomputing them removes
    # the dynamic_rope_update decorator from the exported execution path.
    with torch.inference_mode():
        static_cos, static_sin = pt.rotary_emb(dummy_hidden, position_ids)

    class StaticVocoder(torch.nn.Module):
        def __init__(
            self,
            d: torch.nn.Module,
            mask: torch.Tensor,
            cos: torch.Tensor,
            sin: torch.Tensor,
            pos: torch.Tensor,
        ):
            super().__init__()
            self.decoder = d
            self.register_buffer("attention_mask", mask.contiguous())
            self.register_buffer("rope_cos", cos.contiguous())
            self.register_buffer("rope_sin", sin.contiguous())
            self.register_buffer("position_ids", pos.contiguous())

        def forward(self, codes_f32: torch.Tensor) -> torch.Tensor:
            codes = codes_f32.to(torch.int32)
            d = self.decoder
            p = d.pre_transformer

            hidden = d.quantizer.decode(codes)
            hidden = d.pre_conv(hidden).transpose(1, 2)
            hidden = p.input_proj(hidden)

            for layer in p.layers:
                hidden = layer(
                    hidden,
                    attention_mask=self.attention_mask,
                    position_ids=self.position_ids,
                    past_key_values=None,
                    use_cache=False,
                    cache_position=None,
                    position_embeddings=(self.rope_cos, self.rope_sin),
                )

            hidden = p.norm(hidden)
            hidden = p.output_proj(hidden)
            hidden = hidden.permute(0, 2, 1)

            for blocks in d.upsample:
                for block in blocks:
                    hidden = block(hidden)
            wav = hidden
            for block in d.decoder:
                wav = block(wav)
            return wav.clamp(min=-1, max=1).contiguous()

    wrapper = StaticVocoder(
        decoder,
        static_mask,
        static_cos,
        static_sin,
        position_ids,
    ).eval()

    g = torch.Generator(device="cpu").manual_seed(194)
    codes_i32 = torch.randint(
        0,
        int(decoder.config.codebook_size),
        (1, int(decoder.config.num_quantizers), frames),
        generator=g,
        dtype=torch.int32,
    )
    codes_f32 = codes_i32.float()

    with torch.inference_mode():
        eager_ref = decoder(codes_i32)
        ref = wrapper(codes_f32)

    if eager_ref.shape != ref.shape:
        raise RuntimeError(
            f"static wrapper shape {tuple(ref.shape)} != eager {tuple(eager_ref.shape)}"
        )
    parity = float(torch.max(torch.abs(eager_ref - ref)))
    if parity > 2e-5:
        raise RuntimeError(f"static wrapper parity failed: max_abs={parity}")

    expected = (1, 1, frames * int(model.decode_upsample_rate))
    if tuple(ref.shape) != expected:
        raise RuntimeError(f"unexpected decoder shape {tuple(ref.shape)} != {expected}")
    if not torch.isfinite(ref).all():
        raise RuntimeError("reference decoder produced non-finite PCM")

    torch.onnx.export(
        wrapper,
        (codes_f32,),
        str(output),
        input_names=["codes"],
        output_names=["audio"],
        opset_version=17,
        do_constant_folding=True,
        export_params=True,
        dynamic_axes=None,
        dynamo=False,
    )

    meta = {
        "frames": frames,
        "num_quantizers": int(decoder.config.num_quantizers),
        "codebook_size": int(decoder.config.codebook_size),
        "samples_per_frame": int(model.decode_upsample_rate),
        "sample_rate": int(model.output_sample_rate),
        "input_dtype": "float32 (exact integer code ids)",
        "output_dtype": "float32",
        "eager_static_max_abs": parity,
        "reference_peak": float(ref.abs().max()),
        "reference_rms": float(torch.sqrt(torch.mean(ref.float().square()))),
        "reference_checksum": float(ref.float().sum()),
    }
    output.with_suffix(".export.json").write_text(json.dumps(meta, indent=2) + "\n")
    torch.save({"codes": codes_f32, "audio": ref}, output.with_suffix(".reference.pt"))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
