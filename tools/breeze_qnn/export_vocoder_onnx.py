#!/usr/bin/env python3
"""Export Breeze TTS 2 vocoder after the residual vector quantizer.

The old INT32->Gather QNN graph compiled and ran fast on SM8850 but returned
constant near-zero PCM. This version preprojects every released codebook row
into a small FP16 LUT on the host. QNN receives plain FLOAT32 quantizer features
and runs only the expensive transformer + convolutional vocoder on HTP.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch

def project_group_table(group):
    rows=[]
    for layer in group.vq.layers:
        emb=layer._codebook.embedding_sum / layer._codebook.cluster_usage.clamp(
            min=layer._codebook.epsilon)[:,None]
        if isinstance(layer.project_out, torch.nn.Identity):
            p=emb
        elif isinstance(layer.project_out, torch.nn.Linear):
            p=torch.nn.functional.linear(emb,layer.project_out.weight,layer.project_out.bias)
        else:
            raise RuntimeError("unsupported project_out "+type(layer.project_out).__name__)
        if isinstance(group.output_proj, torch.nn.Identity):
            f=p
        elif isinstance(group.output_proj, torch.nn.Conv1d):
            f=torch.nn.functional.linear(p,group.output_proj.weight.squeeze(-1),group.output_proj.bias)
        else:
            raise RuntimeError("unsupported output_proj "+type(group.output_proj).__name__)
        rows.append(f)
    return torch.stack(rows,0)

def decode_lut(lut,codes):
    b,k,t=codes.shape
    out=torch.zeros((b,lut.shape[-1],t),dtype=torch.float32)
    for cb in range(k):
        out += lut[cb][codes[:,cb].long()].float().permute(0,2,1)
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model-dir",required=True)
    ap.add_argument("--output",required=True)
    ap.add_argument("--lut-output",required=True)
    ap.add_argument("--frames",type=int,default=64)
    a=ap.parse_args()
    from transformers.masking_utils import create_sliding_window_causal_mask
    from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2Model

    out=Path(a.output); lut_out=Path(a.lut_output)
    out.parent.mkdir(parents=True,exist_ok=True); lut_out.parent.mkdir(parents=True,exist_ok=True)
    model=Qwen3TTSTokenizerV2Model.from_pretrained(a.model_dir,local_files_only=True,torch_dtype=torch.float32)
    d=model.decoder.eval().float()
    d.config._attn_implementation="eager"; d.pre_transformer.config._attn_implementation="eager"

    lut32=torch.cat([project_group_table(d.quantizer.rvq_first),
                     project_group_table(d.quantizer.rvq_rest)],0).contiguous()
    lut=lut32.half().contiguous()
    nq=int(d.config.num_quantizers); bins=int(d.config.codebook_size)
    if lut.shape[0]!=nq or lut.shape[1]!=bins: raise RuntimeError(f"bad LUT {tuple(lut.shape)}")
    lut.cpu().numpy().tofile(lut_out)

    p=d.pre_transformer; frames=int(a.frames)
    pos=torch.arange(frames,dtype=torch.long).unsqueeze(0)
    cache=torch.arange(frames,dtype=torch.long)
    dummy=torch.zeros((1,frames,int(p.config.hidden_size)),dtype=torch.float32)
    mask=create_sliding_window_causal_mask(config=p.config,input_embeds=dummy,
        attention_mask=None,cache_position=cache,past_key_values=None,position_ids=pos)
    if mask is None:
        neg=torch.finfo(torch.float32).min
        q=torch.arange(frames)[:,None]; k=torch.arange(frames)[None,:]
        window=int(getattr(p.config,"sliding_window",frames))
        allowed=(k<=q)&(k>q-window)
        mask=torch.where(allowed,torch.zeros((),dtype=torch.float32),
                         torch.full((),neg,dtype=torch.float32))[None,None,:,:]
    with torch.no_grad(): cos,sin=p.rotary_emb(dummy,pos)
    cos=cos.detach().clone(); sin=sin.detach().clone()

    class V(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.d=d
            self.register_buffer("mask",mask.contiguous())
            self.register_buffer("cos",cos.contiguous())
            self.register_buffer("sin",sin.contiguous())
            self.register_buffer("pos",pos.contiguous())
        def forward(self,qfeatures):
            d=self.d; p=d.pre_transformer
            h=d.pre_conv(qfeatures).transpose(1,2); h=p.input_proj(h)
            for layer in p.layers:
                h=layer(h,attention_mask=self.mask,position_ids=self.pos,
                    past_key_values=None,use_cache=False,cache_position=None,
                    position_embeddings=(self.cos,self.sin))
            h=p.output_proj(p.norm(h)).permute(0,2,1)
            for blocks in d.upsample:
                for block in blocks: h=block(h)
            for block in d.decoder: h=block(h)
            return h.clamp(-1,1).float().contiguous()
    v=V().eval()
    g=torch.Generator().manual_seed(196)
    codes=torch.randint(0,bins,(1,nq,frames),generator=g,dtype=torch.int32)
    with torch.inference_mode():
        official_q=d.quantizer.decode(codes).float()
        qfeatures=decode_lut(lut,codes)
        qmax=float((official_q-qfeatures).abs().max()); qmean=float((official_q-qfeatures).abs().mean())
        eager=d(codes).float(); ref=v(qfeatures).float()
    amax=float((eager-ref).abs().max()); amean=float((eager-ref).abs().mean())
    if qmax>0.01: raise RuntimeError(f"LUT parity max_abs={qmax}")
    if amax>0.08 or amean>0.008: raise RuntimeError(f"audio parity max={amax} mean={amean}")
    expected=(1,1,frames*int(model.decode_upsample_rate))
    if tuple(ref.shape)!=expected or not torch.isfinite(ref).all(): raise RuntimeError("bad reference PCM")

    torch.onnx.export(v,(qfeatures,),str(out),input_names=["quantized"],output_names=["audio"],
        opset_version=17,do_constant_folding=True,export_params=True,dynamic_axes=None,dynamo=False)
    meta={"frames":frames,"num_quantizers":nq,"codebook_size":bins,
      "feature_channels":int(lut.shape[-1]),"samples_per_frame":int(model.decode_upsample_rate),
      "sample_rate":int(model.output_sample_rate),"input_dtype":"float32","output_dtype":"float32",
      "lut_dtype":"float16","lut_shape":list(lut.shape),"lut_bytes":int(lut.numel()*2),
      "lut_quantizer_max_abs":qmax,"lut_quantizer_mean_abs":qmean,
      "eager_feature_audio_max_abs":amax,"eager_feature_audio_mean_abs":amean,
      "reference_peak":float(ref.abs().max()),"reference_rms":float(torch.sqrt(torch.mean(ref.square()))),
      "reference_checksum":float(ref.sum())}
    out.with_suffix(".export.json").write_text(json.dumps(meta,indent=2)+"\n")
    qfeatures.cpu().numpy().astype(np.float32).tofile(out.with_suffix(".features.raw"))
    torch.save({"features":qfeatures,"audio":ref,"codes":codes},out.with_suffix(".reference.pt"))
    print(json.dumps(meta,indent=2))
if __name__=="__main__": main()
