#!/usr/bin/env python3
"""Export Breeze TTS 2 depth decoder as two QNN-friendly fixed-shape graphs.

Graph contract:
  depth_prefill(backbone_hidden, first_codebook) -> logits, K, V
  depth_step(token, position, K, V) -> logits, K, V

The cache stays a fixed 16-slot tensor. Sampling remains native in Local Dream,
so temperature/top-k/top-p behavior is not baked into the accelerator.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors import safe_open

MAX_DEPTH_SEQ = 16
REPO_ID = "BreezeBlue/Breeze-TTS-2"


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def repeat_kv(x: torch.Tensor, groups: int) -> torch.Tensor:
    if groups == 1:
        return x
    b, kvh, s, d = x.shape
    return (
        x[:, :, None, :, :]
        .expand(b, kvh, groups, s, d)
        .reshape(b, kvh * groups, s, d)
    )


class FunctionalDepthBase(torch.nn.Module):
    def __init__(self, depth):
        super().__init__()
        self.depth = depth.eval()
        c = depth.config
        self.layers = int(c.num_hidden_layers)
        self.hidden = int(c.hidden_size)
        self.backbone_hidden = int(c.backbone_hidden_size)
        self.vocab = int(c.vocab_size)
        self.num_codebooks = int(c.num_codebooks)
        self.num_heads = int(c.num_attention_heads)
        self.num_kv_heads = int(c.num_key_value_heads)
        self.head_dim = int(getattr(c, "head_dim", self.hidden // self.num_heads))
        self.kv_groups = self.num_heads // self.num_kv_heads
        self.max_seq = MAX_DEPTH_SEQ
        if self.num_codebooks != self.max_seq:
            raise RuntimeError(
                f"Expected {self.max_seq} Breeze codebooks, got {self.num_codebooks}"
            )

        dtype = next(depth.parameters()).dtype
        dummy = torch.zeros((1, self.max_seq, self.hidden), dtype=dtype)
        positions = torch.arange(self.max_seq, dtype=torch.long).unsqueeze(0)
        with torch.no_grad():
            cos, sin = depth.model.rotary_emb(dummy, positions)
        self.register_buffer("cos_table", cos[0].contiguous())
        self.register_buffer("sin_table", sin[0].contiguous())
        self.register_buffer(
            "slot_ids",
            torch.arange(self.max_seq, dtype=torch.int32),
            persistent=False,
        )

    def _rope(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        idx = positions.to(torch.long).reshape(-1)
        cos = self.cos_table.index_select(0, idx).unsqueeze(0).unsqueeze(1)
        sin = self.sin_table.index_select(0, idx).unsqueeze(0).unsqueeze(1)
        return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)

    def _mlp(self, layer, x: torch.Tensor) -> torch.Tensor:
        return layer.mlp.down_proj(
            layer.mlp.act_fn(layer.mlp.gate_proj(x)) * layer.mlp.up_proj(x)
        )

    def _run_layer(
        self,
        layer,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        old_k: torch.Tensor | None,
        old_v: torch.Tensor | None,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = hidden_states
        x = layer.input_layernorm(hidden_states)
        attn = layer.self_attn
        b, s, _ = x.shape
        q = attn.q_proj(x).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        k = attn.k_proj(x).view(b, s, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = attn.v_proj(x).view(b, s, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q, k = self._rope(q, k, positions)

        if old_k is None:
            pad = self.max_seq - s
            k_full = F.pad(k, (0, 0, 0, pad))
            v_full = F.pad(v, (0, 0, 0, pad))
        else:
            # depth_step is single-token. Keep one fixed 16-slot cache and
            # overwrite the runtime-selected position without dynamic shapes.
            p = positions.reshape(1, 1, 1, 1)
            slots = self.slot_ids.reshape(1, 1, self.max_seq, 1)
            write = (slots == p).to(k.dtype)
            k_full = old_k * (1.0 - write) + k * write
            v_full = old_v * (1.0 - write) + v * write

        kr = repeat_kv(k_full, self.kv_groups)
        vr = repeat_kv(v_full, self.kv_groups)
        scores = torch.matmul(q, kr.transpose(2, 3)) * float(attn.scaling)
        scores = scores + valid_mask.to(scores.dtype)
        probs = torch.softmax(scores.float(), dim=-1).to(q.dtype)
        out = torch.matmul(probs, vr)
        out = out.transpose(1, 2).reshape(b, s, self.num_heads * self.head_dim)
        hidden_states = residual + attn.o_proj(out)

        residual = hidden_states
        hidden_states = residual + self._mlp(
            layer,
            layer.post_attention_layernorm(hidden_states),
        )
        return hidden_states, k_full, v_full


class DepthPrefill(FunctionalDepthBase):
    def __init__(self, depth):
        super().__init__(depth)
        # q0 can see k0; q1 can see k0,k1. All other fixed-cache slots masked.
        mask = torch.full((1, 1, 2, self.max_seq), -10000.0, dtype=next(depth.parameters()).dtype)
        mask[:, :, 0, 0] = 0
        mask[:, :, 1, :2] = 0
        self.register_buffer("attention_mask", mask)

    def forward(
        self,
        backbone_hidden: torch.Tensor,
        first_codebook: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        d = self.depth.model
        token = first_codebook.to(torch.long).reshape(1)
        token_embed = d.embed_tokens(token)
        backbone = backbone_hidden
        if d.backbone_hidden_state_projector is not None:
            backbone = d.backbone_hidden_state_projector(backbone)
        pair = torch.stack((backbone, token_embed), dim=1)
        hidden_states = d.inputs_embeds_projector(pair)
        positions = torch.tensor([0, 1], dtype=torch.int32, device=hidden_states.device)

        keys = []
        values = []
        for layer in d.layers:
            hidden_states, k, v = self._run_layer(
                layer,
                hidden_states,
                positions,
                None,
                None,
                self.attention_mask,
            )
            keys.append(k)
            values.append(v)

        hidden_states = d.norm(hidden_states)
        weight = self.depth.codebooks_head.weight[0]
        logits = torch.matmul(hidden_states[:, 1, :].float(), weight.float())
        key_cache = torch.cat(keys, dim=1)
        value_cache = torch.cat(values, dim=1)
        return logits, key_cache, value_cache


class DepthStep(FunctionalDepthBase):
    def forward(
        self,
        token: torch.Tensor,
        position: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        d = self.depth.model
        p = position.to(torch.int32).reshape(1)
        # At position 2 we consume the token predicted by depth head 0, which
        # lives in codebook embedding block 1. Therefore offset = position - 1.
        index = token.to(torch.long).reshape(1) + (
            p.to(torch.long) - 1
        ) * self.vocab
        hidden_states = d.inputs_embeds_projector(d.embed_tokens(index).unsqueeze(1))

        # Single query may attend to cache slots <= runtime position.
        allowed = self.slot_ids.reshape(1, 1, 1, self.max_seq) <= p.reshape(1, 1, 1, 1)
        mask = torch.where(
            allowed,
            torch.zeros((), dtype=hidden_states.dtype, device=hidden_states.device),
            torch.full((), -10000.0, dtype=hidden_states.dtype, device=hidden_states.device),
        )

        keys = []
        values = []
        for i, layer in enumerate(d.layers):
            lo = i * self.num_kv_heads
            hi = lo + self.num_kv_heads
            hidden_states, k, v = self._run_layer(
                layer,
                hidden_states,
                p,
                key_cache[:, lo:hi],
                value_cache[:, lo:hi],
                mask,
            )
            keys.append(k)
            values.append(v)

        hidden_states = d.norm(hidden_states)
        head_index = (p.to(torch.long) - 1).clamp(0, self.num_codebooks - 2)
        weight = self.depth.codebooks_head.weight.index_select(0, head_index)[0]
        logits = torch.matmul(hidden_states[:, 0, :].float(), weight.float())
        return logits, torch.cat(keys, dim=1), torch.cat(values, dim=1)


def load_depth(source_dir: Path, cache_dir: Path):
    sys.path.insert(0, str(source_dir))
    from models.breeze import BreezeDepthDecoderForCausalLM
    from models.breeze_config import BreezeConfig

    cache_dir.mkdir(parents=True, exist_ok=True)
    config_path = hf_hub_download(REPO_ID, "config.json", cache_dir=cache_dir)
    index_path = hf_hub_download(
        REPO_ID,
        "model.safetensors.index.json",
        cache_dir=cache_dir,
    )
    config = BreezeConfig.from_json_file(config_path)
    dc = config.depth_decoder_config
    dc._attn_implementation = "eager"

    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float16)
    try:
        depth = BreezeDepthDecoderForCausalLM(dc).eval()
    finally:
        torch.set_default_dtype(old_dtype)
    depth = depth.half()

    index = json.loads(Path(index_path).read_text())
    weight_map = index["weight_map"]
    wanted = {
        key: shard
        for key, shard in weight_map.items()
        if key.startswith("depth_decoder.")
    }

    depth_embed_name = "depth_decoder.model.embed_tokens.weight"
    if depth_embed_name not in wanted:
        tied = "backbone_model.embed_tokens.embed_audio_tokens.weight"
        if tied not in weight_map:
            raise RuntimeError("Depth embedding weight is missing and tied source was not found")
        wanted[tied] = weight_map[tied]

    targets = dict(depth.named_parameters())
    loaded = set()
    for shard in sorted(set(wanted.values())):
        shard_path = hf_hub_download(REPO_ID, shard, cache_dir=cache_dir)
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            for full_name, mapped_shard in wanted.items():
                if mapped_shard != shard:
                    continue
                if full_name == "backbone_model.embed_tokens.embed_audio_tokens.weight":
                    local_name = "model.embed_tokens.weight"
                else:
                    local_name = full_name[len("depth_decoder.") :]
                if local_name not in targets:
                    continue
                tensor = handle.get_tensor(full_name)
                target = targets[local_name]
                if tuple(tensor.shape) != tuple(target.shape):
                    raise RuntimeError(
                        f"Shape mismatch {full_name}: {tuple(tensor.shape)} != {tuple(target.shape)}"
                    )
                with torch.no_grad():
                    target.copy_(tensor.to(dtype=target.dtype))
                loaded.add(local_name)
                del tensor

    missing = sorted(set(targets) - loaded)
    if missing:
        raise RuntimeError("Missing depth weights: " + ", ".join(missing[:20]))
    return depth


@torch.inference_mode()
def parity(depth, prefill: DepthPrefill, step: DepthStep):
    dtype = next(depth.parameters()).dtype
    first = torch.tensor([37], dtype=torch.int32)
    backbone = torch.linspace(-0.25, 0.25, depth.config.backbone_hidden_size, dtype=dtype).reshape(1, -1)

    p_logits, k, v = prefill(backbone, first)

    d = depth.model
    token_embed = d.embed_tokens(first.long())
    b = backbone
    if d.backbone_hidden_state_projector is not None:
        b = d.backbone_hidden_state_projector(b)
    pair = torch.stack((b, token_embed), dim=1)
    ref = d(
        inputs_embeds=pair,
        use_cache=False,
        cache_position=torch.tensor([0, 1], dtype=torch.long),
    ).last_hidden_state
    ref_logits = torch.matmul(
        ref[:, 1, :].float(),
        depth.codebooks_head.weight[0].float(),
    )
    prefill_err = float((p_logits - ref_logits).abs().max())

    sampled = torch.argmax(ref_logits, dim=-1).to(torch.int32)
    s_logits, _, _ = step(sampled, torch.tensor([2], dtype=torch.int32), k, v)

    offset_idx = sampled.long() + depth.config.vocab_size
    third = d.embed_tokens(offset_idx)
    full_emb = torch.cat((pair, third.unsqueeze(1)), dim=1)
    ref3 = d(
        inputs_embeds=full_emb,
        use_cache=False,
        cache_position=torch.tensor([0, 1, 2], dtype=torch.long),
    ).last_hidden_state
    ref_step = torch.matmul(
        ref3[:, 2, :].float(),
        depth.codebooks_head.weight[1].float(),
    )
    step_err = float((s_logits - ref_step).abs().max())
    print("depth parity prefill_max_abs=", prefill_err, "step_max_abs=", step_err)
    if prefill_err > 0.08 or step_err > 0.10:
        raise RuntimeError("Functional depth graph parity failed")
    return prefill_err, step_err


def export_one(module, args, path: Path, input_names, output_names):
    module.eval()
    torch.onnx.export(
        module,
        args,
        str(path),
        input_names=input_names,
        output_names=output_names,
        opset_version=17,
        do_constant_folding=True,
        export_params=True,
        dynamic_axes=None,
        dynamo=False,
        external_data=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--cache-dir", required=True)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    depth = load_depth(Path(args.source_dir), Path(args.cache_dir))
    prefill = DepthPrefill(depth)
    step = DepthStep(depth)
    prefill_err, step_err = parity(depth, prefill, step)

    dtype = next(depth.parameters()).dtype
    backbone = torch.zeros((1, depth.config.backbone_hidden_size), dtype=dtype)
    first = torch.zeros((1,), dtype=torch.int32)
    with torch.no_grad():
        _, k, v = prefill(backbone, first)

    export_one(
        prefill,
        (backbone, first),
        out / "breeze_depth_prefill.onnx",
        ["backbone_hidden", "first_codebook"],
        ["logits", "key_cache", "value_cache"],
    )
    export_one(
        step,
        (
            torch.zeros((1,), dtype=torch.int32),
            torch.tensor([2], dtype=torch.int32),
            k,
            v,
        ),
        out / "breeze_depth_step.onnx",
        ["token", "position", "key_cache", "value_cache"],
        ["logits", "key_cache_out", "value_cache_out"],
    )

    meta = {
        "version": 1,
        "engine": "qnn-depth-kv-v1",
        "graph_names": ["depth_prefill", "depth_step"],
        "max_depth_seq": MAX_DEPTH_SEQ,
        "num_layers": prefill.layers,
        "num_kv_heads": prefill.num_kv_heads,
        "head_dim": prefill.head_dim,
        "cache_shape": [1, prefill.layers * prefill.num_kv_heads, MAX_DEPTH_SEQ, prefill.head_dim],
        "backbone_hidden_size": prefill.backbone_hidden,
        "hidden_size": prefill.hidden,
        "vocab_size": prefill.vocab,
        "prefill_parity_max_abs": prefill_err,
        "step_parity_max_abs": step_err,
        "weights_dtype": str(dtype).replace("torch.", ""),
    }
    (out / "generator-export.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
