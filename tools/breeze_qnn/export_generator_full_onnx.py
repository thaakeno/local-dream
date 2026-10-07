#!/usr/bin/env python3
"""Export the Breeze TTS 2 hot generator to QNN-friendly ONNX graphs.

V2 covers both autoregressive transformers:
  * backbone_prefill(inputs_embeds, attention_mask, positions)
      -> hidden, logits, key_cache, value_cache
  * backbone_step(input_embed, positions, key_cache, value_cache, attention_mask)
      -> hidden, logits, new_key, new_value
  * depth_prefill(backbone_hidden, first_codebook)
      -> logits, key_cache, value_cache
  * depth_step(token, position, key_cache, value_cache)
      -> logits, key_cache, value_cache

Backbone prefill has a dynamic sequence axis and is compiled by AI Hub into
64/128/256/512-token buckets. Backbone step keeps a 512-slot KV cache and emits
only the new K/V rows, avoiding a full-cache output/copy every acoustic frame.
Depth graphs have a dynamic batch axis so the linked context contains batch-1
and CFG batch-2 variants. Sampling remains native in Local Dream.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors import safe_open

REPO_ID = "BreezeBlue/Breeze-TTS-2"
MAX_DEPTH_SEQ = 16
MAX_BACKBONE_SEQ = 512
BACKBONE_PREFILL_EXAMPLE = 64


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


def load_config(source_dir: Path, cache_dir: Path):
    sys.path.insert(0, str(source_dir))
    from models.breeze_config import BreezeConfig

    cache_dir.mkdir(parents=True, exist_ok=True)
    config_path = hf_hub_download(REPO_ID, "config.json", cache_dir=cache_dir)
    index_path = hf_hub_download(
        REPO_ID,
        "model.safetensors.index.json",
        cache_dir=cache_dir,
    )
    return BreezeConfig.from_json_file(config_path), Path(index_path)


def copy_selected_weights(
    index_path: Path,
    cache_dir: Path,
    targets: dict[str, torch.nn.Parameter],
    map_name,
) -> None:
    index = json.loads(index_path.read_text())
    weight_map = index["weight_map"]
    wanted: dict[str, str] = {}
    mapped: dict[str, str] = {}
    for full_name, shard in weight_map.items():
        local_name = map_name(full_name)
        if local_name is not None and local_name in targets:
            wanted[full_name] = shard
            mapped[full_name] = local_name

    loaded: set[str] = set()
    for shard in sorted(set(wanted.values())):
        shard_path = hf_hub_download(REPO_ID, shard, cache_dir=cache_dir)
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            for full_name, mapped_shard in wanted.items():
                if mapped_shard != shard:
                    continue
                local_name = mapped[full_name]
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
        raise RuntimeError("Missing weights: " + ", ".join(missing[:30]))


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
        if cos.dim() == 3:
            cos = cos[0]
            sin = sin[0]
        self.register_buffer("cos_table", cos.contiguous())
        self.register_buffer("sin_table", sin.contiguous())
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
        mask = torch.full(
            (1, 1, 2, self.max_seq),
            -10000.0,
            dtype=next(depth.parameters()).dtype,
        )
        mask[:, :, 0, 0] = 0
        mask[:, :, 1, :2] = 0
        self.register_buffer("attention_mask", mask)

    def forward(
        self,
        backbone_hidden: torch.Tensor,
        first_codebook: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        d = self.depth.model
        token = first_codebook.to(torch.long).reshape(-1)
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
                layer, hidden_states, positions, None, None, self.attention_mask
            )
            keys.append(k)
            values.append(v)

        hidden_states = d.norm(hidden_states)
        weight = self.depth.codebooks_head.weight[0]
        logits = torch.matmul(hidden_states[:, 1, :].float(), weight.float())
        return logits, torch.cat(keys, dim=1), torch.cat(values, dim=1)


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
        index = token.to(torch.long).reshape(-1) + (
            p.to(torch.long) - 1
        ) * self.vocab
        hidden_states = d.inputs_embeds_projector(d.embed_tokens(index).unsqueeze(1))

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


def load_depth(source_dir: Path, cache_dir: Path, config, index_path: Path):
    from models.breeze import BreezeDepthDecoderForCausalLM

    dc = config.depth_decoder_config
    dc._attn_implementation = "eager"
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float16)
    try:
        depth = BreezeDepthDecoderForCausalLM(dc).eval().half()
    finally:
        torch.set_default_dtype(old_dtype)

    targets = dict(depth.named_parameters())

    def mapper(full_name: str):
        if full_name.startswith("depth_decoder."):
            return full_name[len("depth_decoder.") :]
        if (
            full_name == "backbone_model.embed_tokens.embed_audio_tokens.weight"
            and "model.embed_tokens.weight" in targets
        ):
            return "model.embed_tokens.weight"
        return None

    copy_selected_weights(index_path, cache_dir, targets, mapper)
    return depth


class BackboneCore(torch.nn.Module):
    def __init__(self, layers, norm, rotary_emb, lm_head, qwen_config):
        super().__init__()
        self.layers = layers
        self.norm = norm
        self.lm_head = lm_head
        self.num_layers = int(qwen_config.num_hidden_layers)
        self.hidden = int(qwen_config.hidden_size)
        self.num_heads = int(qwen_config.num_attention_heads)
        self.num_kv_heads = int(qwen_config.num_key_value_heads)
        self.head_dim = int(getattr(qwen_config, "head_dim", self.hidden // self.num_heads))
        self.kv_groups = self.num_heads // self.num_kv_heads
        self.max_seq = MAX_BACKBONE_SEQ
        dtype = next(self.layers.parameters()).dtype
        dummy = torch.zeros((1, self.max_seq, self.hidden), dtype=dtype)
        positions = torch.arange(self.max_seq, dtype=torch.long).unsqueeze(0)
        with torch.no_grad():
            cos, sin = rotary_emb(dummy, positions)
        if cos.dim() == 3:
            cos = cos[0]
            sin = sin[0]
        self.register_buffer("cos_table", cos.contiguous())
        self.register_buffer("sin_table", sin.contiguous())

    def _rope(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        flat = positions.to(torch.long).reshape(-1)
        b = positions.shape[0]
        s = positions.shape[-1]
        cos = self.cos_table.index_select(0, flat).reshape(b, s, self.head_dim).unsqueeze(1)
        sin = self.sin_table.index_select(0, flat).reshape(b, s, self.head_dim).unsqueeze(1)
        return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)

    def layer_prefill(
        self,
        layer,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = hidden_states
        x = layer.input_layernorm(hidden_states)
        attn = layer.self_attn
        b, s, _ = x.shape
        q = attn.q_proj(x).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        k = attn.k_proj(x).view(b, s, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = attn.v_proj(x).view(b, s, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = attn.q_norm(q)
        k = attn.k_norm(k)
        q, k = self._rope(q, k, positions)

        kr = repeat_kv(k, self.kv_groups)
        vr = repeat_kv(v, self.kv_groups)
        scores = torch.matmul(q * float(attn.scaling), kr.transpose(2, 3))
        scores = scores + attention_mask.to(scores.dtype)
        probs = torch.softmax(scores.float(), dim=-1).to(q.dtype)
        out = torch.matmul(probs, vr)
        out = out.transpose(1, 2).reshape(b, s, self.num_heads * self.head_dim)
        hidden_states = residual + attn.o_proj(out)
        residual = hidden_states
        x = layer.post_attention_layernorm(hidden_states)
        hidden_states = residual + layer.mlp.down_proj(
            layer.mlp.act_fn(layer.mlp.gate_proj(x)) * layer.mlp.up_proj(x)
        )
        return hidden_states, k, v

    def layer_step(
        self,
        layer,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        old_k: torch.Tensor,
        old_v: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = hidden_states
        x = layer.input_layernorm(hidden_states)
        attn = layer.self_attn
        b = x.shape[0]
        q = attn.q_proj(x).view(b, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k = attn.k_proj(x).view(b, 1, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = attn.v_proj(x).view(b, 1, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = attn.q_norm(q)
        k = attn.k_norm(k)
        q, k = self._rope(q, k, positions.reshape(b, 1))

        k_attn = torch.cat((old_k, k), dim=2)
        v_attn = torch.cat((old_v, v), dim=2)
        kr = repeat_kv(k_attn, self.kv_groups)
        vr = repeat_kv(v_attn, self.kv_groups)
        scores = torch.matmul(q * float(attn.scaling), kr.transpose(2, 3))
        scores = scores + attention_mask.to(scores.dtype)
        probs = torch.softmax(scores.float(), dim=-1).to(q.dtype)
        out = torch.matmul(probs, vr)
        out = out.transpose(1, 2).reshape(b, 1, self.num_heads * self.head_dim)
        hidden_states = residual + attn.o_proj(out)
        residual = hidden_states
        x = layer.post_attention_layernorm(hidden_states)
        hidden_states = residual + layer.mlp.down_proj(
            layer.mlp.act_fn(layer.mlp.gate_proj(x)) * layer.mlp.up_proj(x)
        )
        return hidden_states, k, v


class BackbonePrefill(torch.nn.Module):
    def __init__(self, core: BackboneCore):
        super().__init__()
        self.core = core

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = inputs_embeds
        keys = []
        values = []
        for layer in self.core.layers:
            x, k, v = self.core.layer_prefill(layer, x, positions, attention_mask)
            keys.append(k)
            values.append(v)
        x = self.core.norm(x)
        last = x[:, -1, :]
        logits = torch.matmul(last.float(), self.core.lm_head.weight.float().transpose(0, 1))
        return last, logits, torch.cat(keys, dim=1), torch.cat(values, dim=1)


class BackboneStep(torch.nn.Module):
    def __init__(self, core: BackboneCore):
        super().__init__()
        self.core = core

    def forward(
        self,
        input_embed: torch.Tensor,
        positions: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = input_embed
        new_keys = []
        new_values = []
        for i, layer in enumerate(self.core.layers):
            lo = i * self.core.num_kv_heads
            hi = lo + self.core.num_kv_heads
            x, k, v = self.core.layer_step(
                layer,
                x,
                positions,
                key_cache[:, lo:hi],
                value_cache[:, lo:hi],
                attention_mask,
            )
            new_keys.append(k)
            new_values.append(v)
        x = self.core.norm(x)
        hidden = x[:, 0, :]
        logits = torch.matmul(hidden.float(), self.core.lm_head.weight.float().transpose(0, 1))
        return hidden, logits, torch.cat(new_keys, dim=1), torch.cat(new_values, dim=1)


def load_backbone(source_dir: Path, cache_dir: Path, config, index_path: Path):
    from models.breeze_backbone_factory import BreezeBackboneFactory
    from transformers import AutoConfig

    config._attn_implementation = "eager"
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float16)
    try:
        backbone = BreezeBackboneFactory.create_backbone(config).eval().half()
        lm_head = torch.nn.Linear(
            int(config.hidden_size),
            int(config.vocab_size) + 1,
            bias=False,
            dtype=torch.float16,
        ).eval()
    finally:
        torch.set_default_dtype(old_dtype)

    holder = torch.nn.Module()
    holder.layers = backbone.layers
    holder.norm = backbone.norm
    holder.lm_head = lm_head
    targets = dict(holder.named_parameters())

    def mapper(full_name: str):
        if full_name.startswith("backbone_model.layers."):
            return "layers." + full_name[len("backbone_model.layers.") :]
        if full_name.startswith("backbone_model.norm."):
            return "norm." + full_name[len("backbone_model.norm.") :]
        if full_name == "lm_head.weight":
            return "lm_head.weight"
        return None

    copy_selected_weights(index_path, cache_dir, targets, mapper)

    bc = config.backbone_config
    if isinstance(bc, dict):
        qwen_config = AutoConfig.for_model(**bc)
    else:
        qwen_config = bc
    qwen_config._attn_implementation = "eager"
    core = BackboneCore(
        holder.layers,
        holder.norm,
        backbone.rotary_emb,
        holder.lm_head,
        qwen_config,
    ).eval()
    return backbone, core


def make_prefill_inputs(seq: int, hidden: int, dtype: torch.dtype):
    embeds = torch.linspace(
        -0.05, 0.05, seq * hidden, dtype=dtype
    ).reshape(1, seq, hidden)
    mask = torch.full((1, 1, seq, seq), -10000.0, dtype=dtype)
    causal = torch.tril(torch.ones((seq, seq), dtype=torch.bool))
    mask[0, 0].masked_fill_(causal, 0.0)
    positions = torch.arange(seq, dtype=torch.int32).reshape(1, seq)
    return embeds, mask, positions


@torch.inference_mode()
def depth_parity(depth, prefill: DepthPrefill, step: DepthStep):
    dtype = next(depth.parameters()).dtype
    first = torch.tensor([37], dtype=torch.int32)
    backbone = torch.linspace(
        -0.25, 0.25, depth.config.backbone_hidden_size, dtype=dtype
    ).reshape(1, -1)
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
        ref[:, 1, :].float(), depth.codebooks_head.weight[0].float()
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
        ref3[:, 2, :].float(), depth.codebooks_head.weight[1].float()
    )
    step_err = float((s_logits - ref_step).abs().max())
    print("depth parity prefill_max_abs=", prefill_err, "step_max_abs=", step_err)
    if prefill_err > 0.08 or step_err > 0.10:
        raise RuntimeError("Functional depth graph parity failed")
    return prefill_err, step_err


@torch.inference_mode()
def backbone_parity(backbone, core: BackboneCore, prefill: BackbonePrefill, step: BackboneStep):
    seq = 4
    dtype = next(core.parameters()).dtype
    embeds, mask, pos = make_prefill_inputs(seq, core.hidden, dtype)
    h, logits, k, v = prefill(embeds, mask, pos)

    ref = backbone(
        inputs_embeds=embeds,
        attention_mask=torch.ones((1, seq), dtype=torch.long),
        position_ids=pos.long(),
        use_cache=False,
    ).last_hidden_state
    ref_h = ref[:, -1, :]
    ref_logits = torch.matmul(ref_h.float(), core.lm_head.weight.float().transpose(0, 1))
    prefill_err = max(
        float((h - ref_h).abs().max()),
        float((logits - ref_logits).abs().max()),
    )

    next_embed = torch.linspace(
        0.04, -0.04, core.hidden, dtype=dtype
    ).reshape(1, 1, core.hidden)
    step_pos = torch.tensor([seq], dtype=torch.int32)
    full_k = torch.zeros(
        (1, core.num_layers * core.num_kv_heads, core.max_seq, core.head_dim),
        dtype=dtype,
    )
    full_v = torch.zeros_like(full_k)
    full_k[:, :, :seq].copy_(k)
    full_v[:, :, :seq].copy_(v)
    step_mask = torch.full((1, 1, 1, core.max_seq + 1), -10000.0, dtype=dtype)
    step_mask[:, :, :, :seq] = 0
    step_mask[:, :, :, -1] = 0
    sh, slogits, _, _ = step(next_embed, step_pos, full_k, full_v, step_mask)

    all_embeds = torch.cat((embeds, next_embed), dim=1)
    all_pos = torch.arange(seq + 1, dtype=torch.long).reshape(1, seq + 1)
    ref2 = backbone(
        inputs_embeds=all_embeds,
        attention_mask=torch.ones((1, seq + 1), dtype=torch.long),
        position_ids=all_pos,
        use_cache=False,
    ).last_hidden_state[:, -1, :]
    ref2_logits = torch.matmul(ref2.float(), core.lm_head.weight.float().transpose(0, 1))
    step_err = max(
        float((sh - ref2).abs().max()),
        float((slogits - ref2_logits).abs().max()),
    )
    print("backbone parity prefill_max_abs=", prefill_err, "step_max_abs=", step_err)
    if prefill_err > 0.15 or step_err > 0.20:
        raise RuntimeError("Functional backbone graph parity failed")
    return prefill_err, step_err


def export_one(
    module,
    args,
    path: Path,
    input_names,
    output_names,
    dynamic_axes=None,
):
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
        dynamic_axes=dynamic_axes,
        dynamo=False,
        external_data=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--cache-dir", required=True)
    args = ap.parse_args()

    source = Path(args.source_dir)
    cache = Path(args.cache_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    config, index_path = load_config(source, cache)

    depth = load_depth(source, cache, config, index_path)
    dp = DepthPrefill(depth).eval()
    ds = DepthStep(depth).eval()
    depth_prefill_err, depth_step_err = depth_parity(depth, dp, ds)
    dtype = next(depth.parameters()).dtype
    b_hidden = torch.zeros((1, depth.config.backbone_hidden_size), dtype=dtype)
    first = torch.zeros((1,), dtype=torch.int32)
    with torch.no_grad():
        _, dk, dv = dp(b_hidden, first)

    export_one(
        dp,
        (b_hidden, first),
        out / "breeze_depth_prefill.onnx",
        ["backbone_hidden", "first_codebook"],
        ["logits", "key_cache", "value_cache"],
        dynamic_axes={
            "backbone_hidden": {0: "batch"},
            "first_codebook": {0: "batch"},
            "logits": {0: "batch"},
            "key_cache": {0: "batch"},
            "value_cache": {0: "batch"},
        },
    )
    export_one(
        ds,
        (
            torch.zeros((1,), dtype=torch.int32),
            torch.tensor([2], dtype=torch.int32),
            dk,
            dv,
        ),
        out / "breeze_depth_step.onnx",
        ["token", "position", "key_cache", "value_cache"],
        ["logits", "key_cache_out", "value_cache_out"],
        dynamic_axes={
            "token": {0: "batch"},
            "key_cache": {0: "batch"},
            "value_cache": {0: "batch"},
            "logits": {0: "batch"},
            "key_cache_out": {0: "batch"},
            "value_cache_out": {0: "batch"},
        },
    )

    del dp, ds, depth, dk, dv
    gc.collect()

    backbone, core = load_backbone(source, cache, config, index_path)
    bp = BackbonePrefill(core).eval()
    bs = BackboneStep(core).eval()
    backbone_prefill_err, backbone_step_err = backbone_parity(backbone, core, bp, bs)
    dtype = next(core.parameters()).dtype
    embeds, mask, positions = make_prefill_inputs(
        BACKBONE_PREFILL_EXAMPLE, core.hidden, dtype
    )
    export_one(
        bp,
        (embeds, mask, positions),
        out / "breeze_backbone_prefill.onnx",
        ["inputs_embeds", "attention_mask", "positions"],
        ["hidden", "logits", "key_cache", "value_cache"],
        dynamic_axes={
            "inputs_embeds": {1: "seq"},
            "attention_mask": {2: "seq", 3: "seq"},
            "positions": {1: "seq"},
            "key_cache": {2: "seq"},
            "value_cache": {2: "seq"},
        },
    )
    step_cache_shape = (
        1,
        core.num_layers * core.num_kv_heads,
        core.max_seq,
        core.head_dim,
    )
    bk = torch.zeros(step_cache_shape, dtype=dtype)
    bv = torch.zeros_like(bk)
    step_mask = torch.full(
        (1, 1, 1, core.max_seq + 1),
        -10000.0,
        dtype=dtype,
    )
    step_mask[:, :, :, -1] = 0
    export_one(
        bs,
        (
            torch.zeros((1, 1, core.hidden), dtype=dtype),
            torch.zeros((1,), dtype=torch.int32),
            bk,
            bv,
            step_mask,
        ),
        out / "breeze_backbone_step.onnx",
        ["input_embed", "positions", "key_cache", "value_cache", "attention_mask"],
        ["hidden", "logits", "new_key", "new_value"],
        dynamic_axes={
            "input_embed": {0: "batch"},
            "positions": {0: "batch"},
            "key_cache": {0: "batch"},
            "value_cache": {0: "batch"},
            "attention_mask": {0: "batch"},
            "hidden": {0: "batch"},
            "logits": {0: "batch"},
            "new_key": {0: "batch"},
            "new_value": {0: "batch"},
        },
    )

    meta = {
        "version": 2,
        "engine": "qnn-full-generator-kv-v2",
        "depth": {
            "graph_names": [
                "depth_prefill_b1", "depth_prefill_b2",
                "depth_step_b1", "depth_step_b2",
            ],
            "cache_shape_b1": [1, dp.layers * dp.num_kv_heads if False else 24, 16, 128],
            "prefill_parity_max_abs": depth_prefill_err,
            "step_parity_max_abs": depth_step_err,
        },
        "backbone": {
            "graph_names": [
                "backbone_prefill_64",
                "backbone_prefill_128",
                "backbone_prefill_256",
                "backbone_prefill_512",
                "backbone_step_b1",
                "backbone_step_b2",
            ],
            "prefill_buckets": [64, 128, 256, 512],
            "max_seq": core.max_seq,
            "num_layers": core.num_layers,
            "num_kv_heads": core.num_kv_heads,
            "head_dim": core.head_dim,
            "cache_shape_b1": list(step_cache_shape),
            "prefill_parity_max_abs": backbone_prefill_err,
            "step_parity_max_abs": backbone_step_err,
        },
        "weights_dtype": str(dtype).replace("torch.", ""),
    }
    (out / "generator-export.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
