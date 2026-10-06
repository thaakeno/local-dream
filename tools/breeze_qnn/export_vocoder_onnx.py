#!/usr/bin/env python3
"""
Export the official Breeze TTS 2 / Qwen3-TTS 12 Hz decoder as one static
causal vocoder graph for QAIRT/QNN HTP compilation.

The graph intentionally uses a fixed 64-frame input. At runtime Local Dream
pads future codec frames and crops the PCM to the valid prefix. The decoder is
causal, so padded future frames cannot change already-valid samples. For later
chunks Local Dream supplies the official 25-frame left context and crops it,
matching Qwen3TTSTokenizerV2Decoder.chunked_decode semantics.
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

    class Vocoder(torch.nn.Module):
        def __init__(self, d: torch.nn.Module):
            super().__init__()
            self.decoder = d

        def forward(self, codes: torch.Tensor) -> torch.Tensor:
            # QNN handles int32 graph inputs cleanly; PyTorch embedding accepts
            # int32 indices as well, avoiding an unnecessary int64 Android IO.
            return self.decoder(codes).contiguous()

    wrapper = Vocoder(decoder).eval()
    # Deterministic in-range codes ensure all 16 codebooks are exercised.
    g = torch.Generator(device="cpu").manual_seed(194)
    codes = torch.randint(
        0, int(decoder.config.codebook_size),
        (1, int(decoder.config.num_quantizers), args.frames),
        generator=g,
        dtype=torch.int32,
    )

    with torch.inference_mode():
        ref = wrapper(codes)
    expected = (1, 1, args.frames * int(model.decode_upsample_rate))
    if tuple(ref.shape) != expected:
        raise RuntimeError(f"unexpected decoder shape {tuple(ref.shape)} != {expected}")
    if not torch.isfinite(ref).all():
        raise RuntimeError("reference decoder produced non-finite PCM")

    torch.onnx.export(
        wrapper,
        (codes,),
        str(output),
        input_names=["codes"],
        output_names=["audio"],
        opset_version=17,
        do_constant_folding=True,
        export_params=True,
        dynamic_axes=None,
    )

    meta = {
        "frames": args.frames,
        "num_quantizers": int(decoder.config.num_quantizers),
        "codebook_size": int(decoder.config.codebook_size),
        "samples_per_frame": int(model.decode_upsample_rate),
        "sample_rate": int(model.output_sample_rate),
        "input_dtype": "int32",
        "output_dtype": "float32",
        "reference_peak": float(ref.abs().max()),
        "reference_rms": float(torch.sqrt(torch.mean(ref.float().square()))),
        "reference_checksum": float(ref.float().sum()),
    }
    output.with_suffix(".export.json").write_text(json.dumps(meta, indent=2) + "\n")
    torch.save({"codes": codes, "audio": ref}, output.with_suffix(".reference.pt"))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
