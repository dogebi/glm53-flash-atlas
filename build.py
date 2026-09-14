#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build edge0-35b-atlas/index.html — skill: tensor-atlas-v4-retarget.

src.html is the nisten "Tensor Atlas v4" engine (LLMViz-DeepSeek-V4.1-Flash, MIT (c) 2026
netsin) kept whole for the shell. Only the model half is replaced.

Subject: Edge0/Edge0-35B-A3B-preview — a 4-bit MLX build of Qwen/Qwen3.6-35B-A3B made for the
edge0 streaming runtime: hybrid attention (30 gated-linear-attention layers + 10 full-attention
layers), 256-expert MoE with top-8 routing plus a shared expert, a 27-block vision tower in the
base model (absent from the edge build), a Recover-LoRA adapter set and a trained prerouter head.

Three measured modes, per-tensor bytes read from the safetensors headers over HTTP Range:
  bf16   Qwen/Qwen3.6-35B-A3B            26 shards  1,045 tensors  71.9036 GB
  fp8    Qwen/Qwen3.6-35B-A3B-FP8        42 shards 64,196 tensors  37.4548 GB
  edge0  Edge0/Edge0-35B-A3B-preview      6 shards  2,476 tensors  19.6895 GB
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

DIR = Path(__file__).resolve().parent
SRC, OUT = DIR / "src.html", DIR / "index.html"
BF_JSON, FP8_JSON, E0_JSON = DIR / "hf-bf16.json", DIR / "hf-fp8.json", DIR / "hf-nvfp4.json"
CFG_BASE, CFG_FP8, CFG_E0 = DIR / "hf-config-bf16.json", DIR / "hf-config-fp8.json", DIR / "hf-config-nvfp4.json"

# ---- name folding -----------------------------------------------------------------------------
SCALE_SUFFIX = re.compile(r"\.(weight_scale_inv|weight_scale_2|weight_scale|scales|biases|input_scale|output_scale)$")
PREFIX = [(r"^language_model\.model\.", "model.language_model."),
          (r"^language_model\.lm_head\.", "lm_head.")]
# FP8 splits experts; MLX stacks them in switch_mlp
EXPERT = re.compile(r"^(?:model\.language_model\.layers\.(\d+)|mtp\.layers\.(\d+))\.mlp\.(?:experts\.\d+|switch_mlp)\.(gate_proj|up_proj|down_proj|gate_up_proj)(\.weight)?$")
# GLM-5.3-Flash ships no adapters: nothing to reuse from the edge0 fold table.


def load(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def make_folder(bfset: set[str]):
    def fold_expert(n: str) -> str | None:
        m = EXPERT.match(n)
        if not m:
            return None
        tail = "gate_up_proj" if m.group(3) in ("gate_proj", "up_proj") else "down_proj"
        base = f"model.language_model.layers.{m.group(1)}" if m.group(1) is not None else "mtp.layers.0"
        return f"{base}.mlp.experts.{tail}"

    def synthetic(n: str) -> str | None:
        """GLM-5.3-Flash ships no adapter files: every tensor folds by naming rule alone."""
        return None

    def folder(name: str) -> str | None:
        n = name
        for pat, rep in PREFIX:
            n = re.sub(pat, rep, n)
        syn = synthetic(n)
        if syn:
            return syn
        # family folds come first: a per-expert tensor is one bank entry in every mode, even in the
        # mode whose own name set we are folding against (otherwise the raw name passes through and
        # the bank ends up with zero bytes in that mode -> NaN geometry downstream).
        f = fold_expert(n) or fold_expert(SCALE_SUFFIX.sub("", n))
        if f:
            return f
        stripped = SCALE_SUFFIX.sub("", n)
        for c in (n, stripped, stripped + ".weight"):
            if c in bfset:
                return c
        return None
    return folder


def collect(path: Path, folder) -> tuple[dict, list[str]]:
    out: dict[str, int] = {}
    unmapped: list[str] = []
    for n, t in load(path)["tensors"].items():
        lg = folder(n)
        if lg is None:
            unmapped.append(n)
            continue
        out[lg] = out.get(lg, 0) + t["bytes"]
    return out, unmapped


def templates(allow_unmapped: bool = False) -> dict:
    bfset = set(load(BF_JSON)["tensors"])
    folder = make_folder(bfset)
    bf, un_bf = collect(BF_JSON, folder)
    fp8, un_fp8 = collect(FP8_JSON, folder)
    e0, un_e0 = collect(E0_JSON, folder)
    for label, un in (("bf16", un_bf), ("fp8", un_fp8), ("edge0", un_e0)):
        if un and not allow_unmapped:
            raise SystemExit(f"{label}: {len(un)} tensors map to nothing, e.g. {un[:4]}")
        if un:
            print(f"  unplaced {label} tensors: {len(un)} (e.g. {un[0]})")
    names = sorted(set(bf) | set(fp8) | set(e0))
    dims = {n: t["shape"] for src in (BF_JSON, FP8_JSON, E0_JSON)
            for n, t in load(src)["tensors"].items()}  # raw; logical dims come from the first shard seen

    N_EXP = load(CFG_BASE).get("text_config", load(CFG_BASE)).get("n_routed_experts", 288)

    # one pass per file: logical name -> the shape a mode stores it with (bank dims synthesized)
    shapes: dict[str, list[int]] = {}
    for src in (BF_JSON, FP8_JSON, E0_JSON):
        for raw, t in load(src)["tensors"].items():
            lg = folder(raw)
            if lg is None or lg in shapes:
                continue
            sh = t["shape"]
            if lg.endswith("experts.gate_up_proj") or lg.endswith("experts.down_proj"):
                # the bank's shape comes from the weight tensor; scale/input_scale tensors are empty
                # or block-shaped and would collapse the bank's parameter count
                if not raw.endswith(".weight"):
                    continue
                shapes[lg] = [N_EXP, *sh] if len(sh) == 2 else sh
            elif len(sh) <= 2:
                shapes[lg] = sh
    print(f"  logical tensors: {len(shapes):,}")

    def logical_dims(name: str) -> list[int]:
        return shapes.get(name, [1])

    both = {n: {"dims": logical_dims(n), "b16": bf.get(n, 0), "b8": fp8.get(n, 0), "b4": e0.get(n, 0)}
            for n in names}
    del dims

    def layer_of(i: int) -> dict:
        pre = f"model.language_model.layers.{i}."
        return {n[len(pre):]: v for n, v in both.items() if n.startswith(pre)}

    sigs: dict[str, dict] = {}
    for i in range(45):
        got = layer_of(i)
        if not got:
            raise SystemExit(f"layer {i} has no tensors")
        key = json.dumps({k: v["dims"] for k, v in sorted(got.items())})
        if key in sigs:
            sigs[key]["indices"].append(i)
        else:
            sigs[key] = {"indices": [i], "items": got}
    arch = {}
    for s in sigs.values():
        items = s["items"]
        if "self_attn.kv_a_proj_with_mqa.weight" in items:
            dom = "sparse"                       # MLA + deepseek-style sparse indexer
        elif "experts.gate_up_proj" in items or "mlp.experts.gate_up_proj" in items:
            dom = "linearmoe"
        else:
            dom = "lineardense"                  # first_k_dense_replace = 3
        arch[dom] = s
    mtp = {k: v for k, v in both.items()
           if k.startswith("model.language_model.layers.45.")}
    if not mtp:
        raise SystemExit("no MTP layer (expected index 45)")

    vis_layers: dict[str, list[int]] = {}
    for n in both:
        m = re.match(r"^model\.visual\.blocks\.(\d+)\.(.+)$", n)
        if m:
            vis_layers.setdefault(m.group(2), []).append(int(m.group(1)))
    counts = {k: len(v) for k, v in vis_layers.items()}
    if counts and len(set(counts.values())) != 1:
        raise SystemExit(f"visual blocks are not uniform: {counts}")
    n_vis = next(iter(counts.values())) if counts else 0
    print(f"  vision blocks: {n_vis} rows x {len(counts)} tensors")
    vis = {}
    for suffix in vis_layers:
        v = both[f"model.visual.blocks.*.{suffix}"] if f"model.visual.blocks.*.{suffix}" in both else None
        full = f"model.visual.blocks.{vis_layers[suffix][0]}.{suffix}"
        v = both[full]
        vis[f"model.visual.blocks.*.{suffix}"] = {"dims": v["dims"], "count": n_vis,
                                                 "b16": v["b16"] * n_vis, "b8": v["b8"] * n_vis,
                                                 "b4": v["b4"] * n_vis}
    io = {n: dict(v, count=1) for n, v in both.items()
          if not n.startswith("model.language_model.layers.")
          and not re.match(r"^model\.visual\.blocks\.\d+\.", n)}
    return {"arch": arch, "vis": vis, "io": io, "mtp": mtp}


# names arrive either layer-relative ("mlp.gate.weight") or fully qualified
# ("model.language_model.layers.3.mlp.gate.weight"), so anchors must not require a leading dot;
# shared_expert must be tested before the routed-expert pattern.
CAT = [(r"shared_expert", "shared"),
       (r"(?:^|\.)mlp\.experts|switch_mlp", "expert"),
       (r"(?:^|\.)mlp\.gate", "router"),
       (r"indexer", "index"), (r"self_attn|linear_attn", "attn"),
       (r"hc_(attn|ffn)", "mhc"),
       (r"norm|_log$|dt_bias|_ape$", "norm")]


def cat_of(name: str) -> str:
    if "visual" in name:
        return "vision"
    if "embed_tokens" in name:
        return "vocab"
    if name.startswith("lm_head"):
        return "head"
    for pat, c in CAT:
        if re.search(pat, name):
            return c
    return "norm"


NOTE = {
    "mlp.experts.gate_up_proj": "288 experts x (4096 -> 2048 gate and up), folded from per-expert tensors",
    "mlp.experts.down_proj": "288 experts x (2048 -> 4096), folded from per-expert tensors",
    "mlp.gate.weight": "router: 288 logits per token (sigmoid, noaux_tc)",
    "mlp.gate.e_score_correction_bias": "per-expert bias correction (288 values)",
    "mlp.shared_experts.gate_proj.weight": "shared expert, gate",
    "mlp.shared_experts.up_proj.weight": "shared expert, up",
    "mlp.shared_experts.down_proj.weight": "shared expert, down",
    "hc_attn_base": "mHC: attention-lane hyper-connection base",
    "hc_attn_fn": "mHC: attention-lane mixing function",
    "hc_attn_scale": "mHC: attention-lane scale",
    "hc_ffn_base": "mHC: FFN-lane hyper-connection base",
    "hc_ffn_fn": "mHC: FFN-lane mixing function",
    "hc_ffn_scale": "mHC: FFN-lane scale",
    "self_attn.indexer.wq_b.weight": "indexer query projection (32 heads x 128)",
    "self_attn.indexer.wk.weight": "indexer key projection",
    "self_attn.indexer.weights_proj.weight": "indexer head weights",
    "self_attn.indexer.k_norm.weight": "indexer key norm",
    "self_attn.indexer.index_kpool_compress_gate": "index-pool compression gate",
    "self_attn.indexer.index_kpool_compress_ape": "index-pool compression embedding",
    "self_attn.q_a_proj.weight": "MLA query compression (4096 -> 1536)",
    "self_attn.q_a_layernorm.weight": "norm inside the query compression path",
    "self_attn.q_b_proj.weight": "MLA query expansion (1536 -> heads)",
    "self_attn.kv_a_proj_with_mqa.weight": "MLA compressed KV (4096 -> 512 + rope)",
    "self_attn.kv_a_layernorm.weight": "norm on the compressed KV path",
    "self_attn.kv_b_proj.weight": "MLA KV expansion (512 -> per-head K/V)",
    "self_attn.q_proj.weight": "gated-linear: fused q projection",
    "self_attn.k_proj.weight": "gated-linear: k projection",
    "self_attn.v_proj.weight": "gated-linear: v projection",
    "self_attn.q_conv1d.weight": "short depthwise conv on q",
    "self_attn.k_conv1d.weight": "short depthwise conv on k",
    "self_attn.v_conv1d.weight": "short depthwise conv on v",
    "self_attn.f_a_proj.weight": "gated-linear: gate f_a",
    "self_attn.f_b_proj.weight": "gated-linear: gate f_b",
    "self_attn.g_a_proj.weight": "gated-linear: decay g_a",
    "self_attn.g_b_proj.weight": "gated-linear: decay g_b",
    "self_attn.b_proj.weight": "gated-linear: decay b",
    "self_attn.A_log": "recurrent state decay (log space)",
    "self_attn.dt_bias": "recurrent step bias",
    "self_attn.o_norm.weight": "per-head norm on the linear output",
    "self_attn.o_proj.weight": "attention output projection",
    "mlp.gate_proj.weight": "dense FFN gate (layers 0-2)",
    "mlp.up_proj.weight": "dense FFN up (layers 0-2)",
    "mlp.down_proj.weight": "dense FFN down (layers 0-2)",
    "eh_proj.weight": "MTP: splice draft state into the stream",
    "enorm.weight": "MTP: norm on the embedding side",
    "hnorm.weight": "MTP: norm on the hidden side",
}
def js_array(var: str, items, comment: str, vision: bool = False) -> str:
    lines = [f"const {var}=[", f"  /* {comment} */"]
    for name, v in items:
        display = name
        tail = name.split(".", 3)[-1] if name.startswith("model.language_model.layers.") else (
            name.split(".", 2)[-1] if name.startswith("model.visual.blocks.*.") else name)
        note = NOTE.get(tail, "")
        count = v.get("count", 1)
        extra = f",{count}" if count > 1 else ""
        label = display + (f"  ({note})" if note else "")
        lines.append(f'  W({json.dumps(label)},{json.dumps(v["dims"])},{json.dumps(cat_of(name))},'
                     f'"",{{bf16:{v["b16"]},fp8:{v["b8"]},nvfp4:{v["b4"]}}}{extra}),')
    lines.append("];")
    return "\n".join(lines)


DATA_HEAD = r"""// ---------- data.js ----------
/* GLM-5.3-Flash — Tensor Atlas. Architecture read from the published config of
   zai-org/GLM-5.3-Flash (Glm5NextForConditionalGeneration); every byte measured from the
   safetensors headers of the three checkpoints named in Sources, summed per logical tensor,
   with .weight_scale_inv folded into its owner and the 288 per-expert tensors folded into the
   fused gate/up and down banks a bank-wise reader expects. */
const CFG = @@CFG@@;
const CT=CFG.text_config, VC=CFG.vision_config;
const MODE_DEFAULT='fp8';
"""

DATA_TAIL = r"""const COL={blue:'#638bff',enc:'#5ca7ff',dec:'#55d7c1',auxa:'#b99bff',expert:'#76b9ff',shared:'#d6e99c',attn:'#55ddd0',router:'#f4b765',auxb:'#e3a8dc',vision:'#80ceea',head:'#c4d0ff',full:'#efbc71',index:'#b798ff',mhc:'#709bbd',norm:'#a7b5cc',muted:'#8390a6',lin:'#9bb6d8',linear:'#9bb6d8',engram:'#b99bff',mtp:'#e9a6dd'};
const clamp=(x,a,b)=>Math.max(a,Math.min(b,x));
const lerp=(a,b,t)=>a+(b-a)*t;
const smooth=x=>x*x*(3-2*x);
const num=x=>Math.round(x).toLocaleString('en-US');
function fmtP(p){return p>=1e12?(p/1e12).toFixed(3)+'T':p>=1e9?(p/1e9).toFixed(2)+'B':p>=1e6?(p/1e6).toFixed(2)+'M':p>=1e3?(p/1e3).toFixed(1)+'K':num(p);}
function bytes(n,binary=false){let b=binary?1024:1000,u=binary?['B','KiB','MiB','GiB','TiB']:['B','KB','MB','GB','TB'],k=0;while(n>=b&&k<4){n/=b;k++;}return (k===0?num(n):n.toFixed(n>=100?1:2))+' '+u[k];}
const SOURCE_BASE='https://huggingface.co/zai-org/GLM-5.3-Flash';
const BF16_MODEL='https://huggingface.co/zai-org/GLM-5.3-Flash-BF16';
const NVFP4_MODEL='https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4';
const SOURCES=[
 ['Model card',SOURCE_BASE,'GLM-5.3-Flash: a 320B-parameter hybrid MoE (18B active) with a 1M-token context, multimodal input, manifold-constrained hyper-connections and one MTP layer. This atlas is about where its bytes live, not what it scores.'],
 ['config.json (FP8)',SOURCE_BASE+'/blob/main/config.json','Embedded field for field: 45 stack layers plus 1 MTP layer, 288 routed experts with top-8 (sigmoid, noaux_tc, n_group 1), hidden 4096, moe_intermediate 2048, dense intermediate 12288 for the first 3 layers, q_lora 1536 / kv_lora 512, a 128-dim indexer with top-2048 selection, and mHC with hc_mult 4.'],
 ['config.json (BF16)',BF16_MODEL+'/blob/main/config.json','The unquantized reference config: the same architecture with no quantization block, which is why every per-tensor byte figure in the bf16 mode is exactly two bytes per parameter.'],
 ['config.json (NVFP4)',NVFP4_MODEL+'/blob/main/config.json','NVIDIA\u2019s NVFP4 build: 4-bit float with per-block scales, applied to the linear weights while the routers, norms, embed_tokens and the indexer scalars keep a higher-precision dtype.'],
 ['FP8 checkpoint',SOURCE_BASE+'/tree/main','The published checkpoint, 62 shards: e4m3 weights with per-block weight_scale_inv, experts written per expert rather than as fused banks. This is the mode the page opens in.'],
 ['BF16 checkpoint',BF16_MODEL+'/tree/main','120 shards, the largest mode on this page, and the only one that shows the architecture without a quantization recipe on top of it.'],
 ['NVFP4 checkpoint',NVFP4_MODEL+'/tree/main','NVIDIA\u2019s 4-bit float build: the smallest mode here, and the one where the gap between weight bytes and total bytes is widest, because the parts that stay high precision no longer round away.'],
 ['Method',SOURCE_BASE,'Every figure on this page is the sum of tensor byte ranges read from those shard headers over HTTP Range. No weights were downloaded, and no number here is a shape \u00d7 bytes estimate.']
];
const MODE_INFO={
 bf16:{label:'BF16 reference',short:'BF16',color:COL.enc,note:'zai-org/GLM-5.3-Flash-BF16 \u00b7 120 shards \u00b7 2 bytes per parameter'},
 fp8:{label:'FP8 (published)',short:'FP8',color:COL.dec,note:'zai-org/GLM-5.3-Flash \u00b7 62 shards \u00b7 e4m3 + per-block scales'},
 nvfp4:{label:'NVFP4 (NVIDIA)',short:'NVFP4',color:COL.expert,note:'nvidia/GLM-5.3-Flash-NVFP4 \u00b7 33 shards \u00b7 4-bit float + block scales'}
};
const MODE_KEYS=['bf16','fp8','nvfp4'];
/* Lanes: layers 3, 7, 11 \u2026 43 run sparse MLA attention with an indexer and hold the KV cache;
   the other 34 run gated linear attention and keep a constant-size recurrent state instead. */
function modeFor(i){return CT.layer_types[i]==='linear_attention'?'linear':'full';}
function ownerFor(i){if(modeFor(i)==='full')return i;for(let j=i;j>=0;j--)if(modeFor(j)==='full')return j;return null;}
function indexOwnerFor(i){return modeFor(i)==='full'?i:ownerFor(i);}
function W(name,shape,cat,note='',ex=null,count=1){
 return {name,shape,cat,note,count,
   p:shape.reduce((a,b)=>a*b,1)*count,
   format:ex&&ex.nvfp4&&ex.nvfp4<ex.bf16*0.4?'nvfp4':(ex&&ex.fp8&&ex.fp8<ex.bf16*0.9?'fp8':'bf16'),
   ex};
}
function wBytes(w,mode='bf16'){return w.ex?w.ex[mode]:2*w.p;}
function wFormat(w,mode){return {bf16:'BF16',fp8:'FP8 / block',nvfp4:'NVFP4'}[mode];}
const sumP=ws=>ws.reduce((a,w)=>a+w.p,0);
const sumB=(ws,m)=>ws.reduce((a,w)=>a+wBytes(w,m),0);
@@TABLES@@
const FP8_CONFIG=@@FP8CFG@@;
const NV_CONFIG=@@NVCFG@@;
function layerWeights(i){return modeFor(i)==='full'?SPARSE_W:(i<CT.first_k_dense_replace?DENSE_W:LINEAR_W);}
const LAYERS=Array.from({length:45},(_,i)=>({id:'L'+i,index:i,label:'Layer '+String(i).padStart(2,'0'),
 part:modeFor(i)==='full'?'sparse':'linear',mode:modeFor(i),owner:ownerFor(i),indexOwner:indexOwnerFor(i),
 ratio:1,ws:layerWeights(i)}));
const AUX_TABLES=[];
const DRAFT=[{id:'D0',index:45,label:'MTP layer 45',ws:DRAFT_W}];
const EMBED={id:'embed',label:'Token embedding',ws:EMBED_W};
const HEAD={id:'head',label:'Output norm + untied head',ws:HEAD_W};
const VISION={id:'vision',label:'Vision tower \u00b7 24 blocks',ws:VISION_W};
const MODULES=[EMBED,...LAYERS,HEAD,...DRAFT,VISION];
const ALL_W=MODULES.flatMap(m=>m.ws);
const TOTALS=Object.fromEntries(MODE_KEYS.map(m=>[m,sumB(ALL_W,m)]));
const TOTAL_P=sumP(ALL_W);
const MOE_EXPERT_P=(2*2048*4096+2048*4096);
const MOE_TOTAL_P=MOE_EXPERT_P*288*45;
const FP8_DELTA=TOTALS.bf16-TOTALS.fp8;
const NV_DELTA=TOTALS.bf16-TOTALS.nvfp4;
const KV_FULL_PER_TOKEN=2*(512+0)*2*11;
const KV_LINEAR_STATE=0;
const CATEGORIES=[
 ['expert','Routed experts \u00b7 288 per layer',COL.expert,LAYERS.flatMap(l=>l.ws.filter(w=>w.cat==='expert'))],
 ['attn','Attention (linear + sparse MLA)',COL.attn,LAYERS.flatMap(l=>l.ws.filter(w=>w.cat==='attn'))],
 ['index','Sparse indexer',COL.index,LAYERS.flatMap(l=>l.ws.filter(w=>w.cat==='index'))],
 ['shared','Shared expert',COL.shared,LAYERS.flatMap(l=>l.ws.filter(w=>w.cat==='shared'))],
 ['router','Routers and gates',COL.router,LAYERS.flatMap(l=>l.ws.filter(w=>w.cat==='router'))],
 ['mhc','Hyper-connections (mHC)',COL.mhc,LAYERS.flatMap(l=>l.ws.filter(w=>w.cat==='mhc'))],
 ['vision','Vision tower + merger',COL.vision,VISION.ws],
 ['mtp','MTP layer',COL.mtp,DRAFT.flatMap(m=>m.ws)],
 ['vocab','Embedding + head',COL.head,[...EMBED.ws,...HEAD.ws]],
 ['norm','Norms and scalars',COL.norm,ALL_W.filter(w=>w.cat==='norm'&&!DRAFT.some(d=>d.ws.includes(w)))]
];
const EXP={
 overview:{title:'320B parameters, 18B awake.',body:'GLM-5.3-Flash is a sparse MoE with a 1M-token window: 288 routed experts per layer, 8 chosen per token by a sigmoid router with a learned bias, a shared expert that always runs, and an MTP layer that drafts the next token. The hybrid stack alternates gated linear attention with sparse MLA attention, so most layers carry a fixed-size recurrent state instead of a growing cache.'},
 expert:{title:'288 experts, 8 per token.',body:'Every sparse-FFN layer stores 288 experts of 2048 intermediate width, each as three matrices (gate, up, down) over a 4096-wide residual stream. In the FP8 build that is written per expert, with a scale tensor beside every weight; this atlas folds expert tensors into gate/up and down banks so the ledger stays readable, and every folded byte still comes from a measured header.'},
 full:{title:'Eleven sparse-attention layers.',body:'Layers 3, 7, 11 \u2026 43 run MLA attention: a 1536-rank query path, a 512-rank compressed KV path, and a separate 128-dim indexer that selects the top 2048 key positions before attention is computed. These eleven layers are the only ones whose cache grows with the context.'},
 swa:{title:'Thirty-four gated-linear layers.',body:'The rest of the stack is gated linear attention: q/k/v projections with short depthwise convolutions, per-channel decay, output gating and a per-head norm. Its state is constant-size, which is how a 1M-token window stays affordable \u2014 only the eleven sparse layers pay for a growing cache.'},
 selfcond:{title:'mHC: four residual lanes, kept manifold-constrained.',body:'Every layer carries hc_attn and hc_ffn triples (base, fn, scale) \u2014 6 scalars per layer, 270 across the stack. They are the hyper-connection weights that mix four parallel residual streams, normalised with a Sinkhorn iteration so the mixing matrix stays on its manifold.'},
 vision:{title:'A 24-block vision tower.',body:'The multimodal half: 24 transformer blocks at hidden 1024 with attention bias and QK norms, a patch-embedding downsample, and a merger that projects into the language model\u2019s 4096 width. It is a small slice of the checkpoint in every mode \u2014 and it is the part NVFP4 leaves almost untouched.'},
 mtp:{title:'One extra layer that runs ahead.',body:'Layer 45 is the multi-token-prediction head: its own full MoE layer plus eh_proj, enorm and hnorm to splice the draft state back into the main stream. It is stored, not a runtime trick \u2014 which is why it shows up in the ledger of all three modes.'},
 bf16:{title:'The reference: 2 bytes per parameter.',body:'The BF16 repository is the architecture without a quantization recipe: 120 shards, every explicit tensor at two bytes per parameter, and the largest total on this page. Nothing is folded away here \u2014 the expert banks are measured the same way, just written per expert.'},
 fp8:{title:'The published checkpoint.',body:'FP8 ships e4m3 weights with a per-block weight_scale_inv beside each one \u2014 which is why its tensor count is many times the BF16 count while its payload is roughly half. This is the mode the page opens in, because it is the checkpoint most deployments actually load.'},
 nvfp4:{title:'4-bit float, biggest gap between weights and file.',body:'NVFP4 converts the linear weights to 4-bit float with per-block scales, but leaves the routers, norms, embeddings, the indexer scalars and other sensitive tensors at a higher precision. The result is a much smaller file than the weight ratio alone would predict \u2014 and the reason the saving is not a clean 4\u00d7.'},
 storage:{title:'Where 320B parameters go.',body:'Almost all of it is expert mass: 288 experts \u00d7 45 layers of gate/up/down. Switch mode and watch the same architecture re-price itself \u2014 BF16 to FP8 is roughly a halving, NVFP4 goes further, and the parts that resist quantization are exactly the parts the configs exclude.'},
 cache:{title:'Eleven growing caches, thirty-four fixed states.',body:'Sparse-MLA layers keep a compressed KV cache at lora rank 512 plus a 128-dim index; the gated-linear layers keep a constant-size recurrent state whose footprint does not depend on the context. That asymmetry is the whole reason a 1M-token window is affordable.'}
};
const BENCH=[
 {name:'MMLU Pro',capability:'Knowledge',values:[0,0]},
 {name:'GPQA Diamond',capability:'Science',values:[0,0]},
 {name:'AIME 2025',capability:'Math',values:[0,0]},
 {name:'LiveCodeBench',capability:'Coding',values:[0,0]},
 {name:'MMMU',capability:'Vision',values:[0,0]}
];
const BENCH_MODELS=['measured bytes: bf16 / fp8 / nvfp4'];
function pickExperts(seed,n=288,k=8){let x=(seed+1)*2654435761>>>0;const s=new Set;while(s.size<k){x=(Math.imul(x,1664525)+1013904223)>>>0;s.add(x%n);}return [...s].sort((a,b)=>a-b);}
const PHASES=[
 {name:'Route',label:'Route',from:0,to:6,active:'8 of 288',color:COL.router,caption:'Sigmoid router with a learned bias.',desc:'Each token scores 288 experts through a sigmoid router (noaux_tc, one group) and the top 8 are mixed by their normalised weights. A shared expert runs for every token beside them.'},
 {name:'Fetch',label:'Fetch',from:6,to:11,active:'per expert',color:COL.expert,caption:'Expert banks are the mass.',desc:'288 experts \u00d7 45 layers of gate/up/down is where nearly all of the checkpoint lives; in the FP8 and NVFP4 builds each expert also carries its own scale tensors.'},
 {name:'Compute',label:'Compute',from:11,to:17,active:'18B active',color:COL.enc,caption:'Hybrid attention plus the MoE FFN.',desc:'Sparse-MLA layers read a top-2048 selection from a 1M-token window; gated-linear layers carry a fixed recurrent state. Both mix the routed experts with the always-on shared expert, then the mHC hyper-connection folds four residual lanes back into one.'},
 {name:'Predict',label:'Predict',from:17,to:20,active:'indexer',color:COL.index,caption:'Choose the keys before attention.',desc:'The 128-dim indexer scores key positions and keeps the top 2048 before the MLA attention is computed \u2014 a separate small network with its own weights in the ledger.'},
 {name:'Draft',label:'Draft',from:20,to:22,active:'MTP',color:COL.mtp,caption:'One layer runs ahead.',desc:'Layer 45 is stored like any other layer and drafts the next token so the following forward pass has a head start.'}
];
const TRAIN_PHASES=[
 {name:'Pretrain',from:0,to:8,active:'MoE + MLA',color:COL.enc,caption:'Build the base model.',desc:'Conceptual schematic of the published recipe: a sparse MoE with hybrid attention, trained with the router bias that keeps expert load balanced.'},
 {name:'Quantize',from:8,to:11,active:'FP8 / NVFP4',color:COL.auxa,caption:'Choose what not to convert.',desc:'The quantization config lists the modules that stay high precision \u2014 routers, norms, embeddings, the indexer scalars and the attention k/v paths. Those exceptions are visible in the byte ledger.'},
 {name:'Distil',from:11,to:18,active:'Recover',color:COL.auxb,caption:'Close the quantization gap.',desc:'Schematic: quantization-aware recovery against the BF16 reference. This atlas measures the three published checkpoints; it does not measure training runs.'}
];
function phasesFor(mode){return mode==='training'?TRAIN_PHASES:PHASES;}
function phaseAt(t,mode='inference'){const ps=phasesFor(mode);return ps.find(p=>t>=p.from&&t<p.to)||ps[ps.length-1];}
const TC={engram:'#b99bff',auxa:'#c9a3fa',auxb:'#e3a8dc',q:'#8caaff',local:'#ffc477',kv:'#61dccb',index:'#b798ff',out:'#b2bfff',expert:'#66deb0',shared:'#eee081',router:'#f494be',mhc:'#c9a3fa',norm:'#90a9b5',mtp:'#e9a6dd',vision:'#62d5d0',head:'#a5b9eb',attn:'#55ddd0'};
function tensorKind(w){return TC[w.cat]?w.cat:'head';}
function tensorColor(w){return TC[tensorKind(w)]||COL.attn;}
function tensorShort(w){return w.name.replace(/^model\.language_model\.layers\.\d+\./,'').replace(/^model\.visual\.blocks\.\*\./,'ViT.').replace(/^model\.visual\./,'visual.').replace(/^model\.language_model\./,'').replace(/\.weight_scale_inv$/,'').replace(/\.weight$/,'');}
function weightParts(w,mode){const total=wBytes(w,mode);return {data:total,scales:0,aux:0,total};}
function displayWeights(m){return m.ws;}
function findWeight(m,name){return displayWeights(m).find(w=>w.name===name)||m.ws.find(w=>w.name===name);}
function orderWeights(m){const ws=displayWeights(m);const rank=w=>{let n=w.name;
 if(n.includes('experts.gate_up'))return 0; if(n.includes('experts.down'))return 1;
 if(n.endsWith('mlp.gate.weight'))return 2; if(n.includes('e_score_correction_bias'))return 3;
 if(n.includes('shared_experts.gate'))return 4; if(n.includes('shared_experts.up'))return 5; if(n.includes('shared_experts.down'))return 6;
 if(n.includes('hc_attn'))return 7; if(n.includes('hc_ffn'))return 8;
 if(n.includes('indexer.wq_b')||n.includes('indexer.wk'))return 9; if(n.includes('indexer.weights_proj'))return 10;
 if(n.includes('indexer.k_norm'))return 11; if(n.includes('index_kpool'))return 12;
 if(n.includes('q_a_proj'))return 13; if(n.includes('q_a_layernorm'))return 14; if(n.includes('q_b_proj'))return 15;
 if(n.includes('kv_a_proj_with_mqa'))return 16; if(n.includes('kv_a_layernorm'))return 17; if(n.includes('kv_b_proj'))return 18;
 if(n.includes('self_attn.q_proj'))return 19; if(n.includes('self_attn.k_proj'))return 20; if(n.includes('self_attn.v_proj'))return 21;
 if(n.includes('q_conv1d'))return 22; if(n.includes('k_conv1d'))return 23; if(n.includes('v_conv1d'))return 24;
 if(n.includes('f_a_proj')||n.includes('f_b_proj'))return 25; if(n.includes('g_a_proj')||n.includes('g_b_proj'))return 26;
 if(n.includes('b_proj'))return 27; if(n.includes('A_log')||n.includes('dt_bias'))return 28; if(n.includes('o_norm'))return 29;
 if(n.includes('o_proj'))return 30;
 if(n.includes('mlp.gate_proj'))return 31; if(n.includes('mlp.up_proj'))return 32; if(n.includes('mlp.down_proj'))return 33;
 if(n.includes('input_layernorm'))return 34; if(n.includes('post_attention'))return 35;
 if(n.includes('eh_proj')||n.includes('enorm')||n.includes('hnorm'))return 36;
 if(n.includes('embed_tokens'))return 0; if(n.includes('lm_head'))return 1;
 if(n.includes('visual'))return 37; return 38;};
 return [...ws].sort((a,b)=>rank(a)-rank(b));}
function isAttentionTensor(w){return w.cat==='attn'||w.cat==='index';}
function tensorInfo(w,m){
 const p=fmtP(w.p),size=bytes(wBytes(w,'bf16'));
 let t='Stored tensor.',b='Two bytes per parameter in the BF16 reference. Switch the precision to see what FP8 and NVFP4 do with it.';
 if(w.name.includes('experts.gate_up')){t='Fused expert gate and up banks.';b='288 experts \u00d7 (4096 \u2192 2048) for gate and up, folded into one bank so the ledger stays readable. Only 8 of the 288 are active for a token; the rest are stored mass.';}
 else if(w.name.includes('experts.down')){t='Fused expert down banks.';b='288 experts \u00d7 (2048 \u2192 4096): each expert projects its 2048-wide activation back into the 4096-wide residual stream.';}
 else if(w.name.endsWith('mlp.gate.weight')){t='Router.';b='4096 \u2192 288 expert logits, scored with a sigmoid and combined with a learned per-expert bias (noaux_tc). The router keeps a high-precision dtype in both quantized builds.';}
 else if(w.name.includes('e_score_correction_bias')){t='Router bias correction.';b='288 values that correct each expert\u2019s score so load stays balanced without an auxiliary loss. It is one of the tensors the quantization configs exclude.';}
 else if(w.name.includes('shared_experts')){t='Shared expert.';b='The always-on path beside the routed experts: 4096 \u2192 2048 \u2192 4096. Every token passes through it, so it is dense by construction.';}
 else if(w.name.includes('hc_attn')||w.name.includes('hc_ffn')){t='Hyper-connection weight.';b='Part of the mHC triple (base, fn, scale) that mixes four parallel residual lanes per layer, normalised with a Sinkhorn iteration so the mixing stays on its manifold.';}
 else if(w.name.includes('indexer.wq_b')||w.name.includes('indexer.wk')){t='Indexer projection.';b='A 32-head, 128-dim scorer that runs before the MLA attention and decides which key positions (top 2048) attention is allowed to see.';}
 else if(w.name.includes('indexer.weights_proj')){t='Indexer head weights.';b='Projects the indexer\u2019s hidden state into its per-head scores.';}
 else if(w.name.includes('indexer.k_norm')){t='Indexer key norm.';b='Weight and bias applied to indexer keys \u2014 a small tensor with an outsized effect on which 2048 keys are selected.';}
 else if(w.name.includes('index_kpool')){t='Index pool compression.';b='A learned gate and a learned embedding that compress the key pool the indexer scores over.';}
 else if(w.name.includes('q_a_proj')){t='MLA query compression.';b='4096 \u2192 1536: the query path is compressed before it is expanded head-wise, which is what keeps a 1M-token attention cheap.';}
 else if(w.name.includes('q_b_proj')){t='MLA query expansion.';b='1536 \u2192 64 heads \u00d7 192.';}
 else if(w.name.includes('kv_a_proj_with_mqa')){t='MLA compressed KV.';b='4096 \u2192 512 (+ the rope dim): the KV cache is stored compressed at rank 512 and expanded per head at read time.';}
 else if(w.name.includes('kv_b_proj')){t='MLA KV expansion.';b='512 \u2192 64 heads \u00d7 (192 + 128). This is the matrix that turns the compressed cache back into per-head keys and values.';}
 else if(w.name.includes('A_log')||w.name.includes('dt_bias')){t='Linear-attention state parameter.';b='Per-channel decay stored in log space, and the step bias beside it. Dozens of values per layer \u2014 the cheapest part of the block, and the reason its state is constant-size.';}
 else if(w.name.includes('_conv1d')){t='Short convolution.';b='Depthwise causal convolution over the q/k/v stream before the gated-linear recurrence.';}
 else if(w.name.includes('f_a_proj')||w.name.includes('f_b_proj')||w.name.includes('g_a_proj')||w.name.includes('g_b_proj')||w.name.includes('b_proj')){t='Gated-linear projection.';b='Input/output gates and the per-channel decay terms of the gated linear attention block.';}
 else if(w.name.includes('o_norm')){t='Output norm of the linear block.';b='Per-head normalisation applied to the gated-linear output before its projection.';}
 else if(w.name.includes('o_proj')){t='Attention output projection.';b='The dense look-back of both lanes: 8192 \u2192 4096 in the sparse layers, 4096 \u2192 4096 in the linear ones.';}
 else if(w.name.includes('mlp.gate_proj')||w.name.includes('mlp.up_proj')||w.name.includes('mlp.down_proj')){t='Dense FFN.';b='Layers 0\u20132 are dense: 4096 \u2192 12288 \u2192 4096 with SwiGLU, before the stack switches to routed experts.';}
 else if(w.name.includes('eh_proj')||w.name.includes('enorm')||w.name.includes('hnorm')){t='MTP splice.';b='Part of the multi-token-prediction layer: it combines the hidden state with the embedding of the drafted token and renormalises before the extra MoE layer.';}
 else if(w.name.includes('embed_tokens')){t='Token embedding.';b='A 154,880-row table at hidden 4096, untied from the output head, so input and output are separate tensors in every mode.';}
 else if(w.name.includes('lm_head')){t='Output head.';b='4096 \u2192 154,880 logits, untied from the embedding; it keeps a high-precision dtype in the quantized builds.';}
 else if(w.name.includes('visual')){t='Vision tower tensor.';b='Part of the 24-block tower (hidden 1024, bias in attention, QK norms) plus the patch downsample and the merger into the language width.';}
 else if(w.name.includes('norm')){t='Norm or scalar.';b='Normalisation weights stay high precision in every mode here. Quantizing them would save almost nothing and cost stability.';}
 return {title:t,body:b,size,p};}
function layerStory(m){
 if(!m.mode)return null;let t,body,detail;
 if(m.mode==='linear'){t='Gated-linear layer';body='A gated linear attention block \u2014 q/k/v projections with short convolutions, per-channel decay, output gate and per-head norm \u2014 then 8 of 288 experts plus the shared expert, with an mHC hyper-connection around both halves.';detail='This is the default layer of the stack: its recurrent state is fixed-size, so a long prompt does not grow its memory.';}
 else {t='Sparse-attention layer';body='MLA attention: compressed 1536-rank queries, a 512-rank compressed KV path, and a 128-dim indexer that keeps only the top 2048 key positions before attention runs.';detail='Eleven layers \u2014 3, 7, 11 \u2026 43 \u2014 and the only places in the stack where a cache grows with the context.';}
 return {title:t,body,detail};}
"""

def build_data_js(t: dict) -> str:
    arch, io = t["arch"], t["io"]
    sparse = sorted(arch["sparse"]["items"].items())
    linearmoe = sorted(arch["linearmoe"]["items"].items())
    dense = sorted(arch["lineardense"]["items"].items())
    vis = sorted(t["vis"].items())
    mtp = sorted(t["mtp"].items())

    embed = sorted((n, v) for n, v in io.items() if "embed_tokens" in n)
    head = sorted((n, v) for n, v in io.items() if n.startswith("lm_head") or n.endswith("language_model.norm.weight"))
    visio = sorted((n, v) for n, v in io.items() if n.startswith("model.visual."))
    placed = {n for g in (embed, head, visio) for n, _ in g} | set(t["vis"])
    missing = sorted(set(io) - placed)
    if missing:
        raise SystemExit(f"top-level tensors placed in no module: {missing[:6]}")

    tables = "\n".join([
        js_array("DENSE_W", dense, "layers 0-2: dense FFN gate/up/down + gated linear attention + mHC"),
        js_array("LINEAR_W", linearmoe, "one gated-linear MoE layer: attention block + 8-of-288 experts + mHC"),
        js_array("SPARSE_W", sparse, "one sparse-MLA layer: compressed q/kv + indexer + the same expert bank"),
        js_array("EMBED_W", embed, "the untied input embedding"),
        js_array("HEAD_W", head, "the untied output head"),
        js_array("VISION_W", vis, "one vision block x24 (bytes already summed)"),
        js_array("VISIO_W", visio, "vision downsample and merger tensors", True),
        js_array("DRAFT_W", mtp, "the MTP layer 45 (enorm, hnorm, eh_proj + its own MoE block)", True),
    ])

    cfg = dict(load(CFG_BASE))
    tc = dict(cfg.get("text_config", {}))
    sparse_layers = [i for i, t_ in enumerate(tc.get("layer_types", [])) if t_ != "linear_attention"]
    tc["kv_source_layer_ids"] = sparse_layers
    tc["index_source_layer_ids"] = sparse_layers
    cfg["text_config"] = tc
    cfg.setdefault("quantization", {"note": "none in the BF16 reference"})

    js = DATA_HEAD + DATA_TAIL
    js = js.replace("const VISION={id:'vision',label:'Vision tower \u00b7 24 blocks',ws:VISION_W};",
                    "const VISION={id:'vision',label:'Vision tower \u00b7 24 blocks',ws:[...VISION_W,...VISIO_W]};")
    js = js.replace("@@TABLES@@", tables)
    js = js.replace("@@CFG@@", json.dumps(cfg, ensure_ascii=False))
    js = js.replace("@@FP8CFG@@", json.dumps(load(CFG_FP8), ensure_ascii=False))
    js = js.replace("@@NVCFG@@", json.dumps(load(CFG_E0), ensure_ascii=False))
    return js


REGEX_SUBS = [
    # mode keys: the engine called the reference checkpoint "native"; nvfp4 stays nvfp4
    (r"'native'", "'bf16'"),
    (r"\[\s*'native',\s*'nvfp4',\s*'bf16'\s*\]", "['bf16','fp8','nvfp4']"),
    (r"ex\['nvfp4'\]&&ex\['nvfp4'\]<ex\['bf16'\]\*0\.5", "ex['nvfp4']&&ex['nvfp4']<ex['bf16']*0.5"),
    # deck split: 45 stack layers, first reading group / second reading group
    (r"m\.index<20\?", "m.index<23?"),
    (r"m\.index%20", "m.index%23"),
    (r"m\.index<20", "m.index<23"),
    (r"CT\.compress_ratios\[i\]", "1"),
    (r"CT\.kv_lora_rank\|\|512", "512"),
    # scene anchors: this model has 45 layers and one MTP module (D0), not D2
    (r"pos\('D2'\)", "pos('D0')"),
    (r"pos\('L37'\)", "pos('L44')"),
    (r"pos\('L36'\)", "pos('L43')"),
    (r"'L19 -> L20'", "'L22 -> L23'"),
    (r"'ENCODER / L00-L19'", "'LAYERS 00-22'"),
    (r"'First 20 layers / runs in decode'", "'two lanes, four layers apart'"),
    (r"'Next 20 layers / runs in decode'", "'same expert bank in both lanes'"),
    (r"'Ordered layers, not physical shard offsets\.'", "'One stack: gated linear with a sparse layer every fourth block.'"),
    (r"'Optimized prefill: prepare decoder KV'", "'Sparse lane: select the top-2048 keys'"),
    (r"'Forward pass continues\. No model swap\.'", "'Forward pass continues over the same stack.'"),
    (r"'Prefill'", "'Route'"),
    (r"'SWA replay'", "'Fetch'"),
    (r"'Decode'", "'Compute'"),
    (r"'DSpark'", "'Draft'"),
    (r"\bENGRAM\b", "AUX_TABLES"),
    (r'aria-label="DeepSeek V4\\.1 Flash architecture explorer home"', 'aria-label="GLM-5.3-Flash architecture explorer home"'),
    (r'Schema-derived tensor payload estimate', 'Measured tensor payload, audited from shard headers'),
    (r'aria-label=\\"DeepSeek V4.1 Flash architecture explorer home\\"', 'aria-label=\\"GLM-5.3-Flash architecture explorer home\\"'),
    (r'<h1>DeepSeek <em>V4.1 Flash</em>', '<h1>GLM <em>5.3 Flash</em>'),
    (r'SCHEMA-DERIVED PAYLOAD', 'MEASURED PAYLOAD'),
    (r'Schema-derived tensor payload estimate', 'Measured tensor payload, audited from shard headers'),
    (r'DeepSeek-ViT', 'GLM-ViT'),
    (r"'deepseek-exact-selected-shard-audit\\.json'", "'glm53-flash-exact-selected-shard-audit.json'"),
    (r'The scene remains the labeled derived model\\.', 'The scene is a labelled logical model: 1 cubic unit = 1 GB.'),
    (r"'GRADIENTS','Backward path, not reverse inference\.',COL\.mhc", "'GRADIENTS','Quantization-aware recovery, not reverse inference.',COL.auxa"),
    # engine copy that narrated the original model: retarget to this model's parts and numbers
    (r"startsWith\('D'\)\?128:384", "startsWith('D')?288:288"),
    (r"\?'128':'384'", "?'288':'288'"),
    (r"'FOUR SHARED GLOBAL BANKS','890 B / original token across the model, not per layer\.'",
     "'288 ROUTED EXPERTS PER LAYER','the whole bank is stored; 8 of 288 are chosen per token.'"),
    (r"'TOP-512 POSITIONS PER QUERY','512 illuminated sample marks illustrate sparse selection\.'",
     "'TOP-2,048 POSITIONS PER QUERY','a 128-dim indexer keeps the top 2,048 of the visible keys.'"),
    (r"'2,048 CANDIDATE BLOCKS x 8','A schematic pool; the true cap is 16,384 candidate positions\.'",
     "'INDEXER x 32 HEADS','schematic pool; the indexer scores keys at 128 dims before attention.'"),
    (r"The ledger remains schema-derived\. A complete set of checkpoint headers is required before calling its totals exact on-",
     "The ledger is measured from the published shard headers: every mode on this page was read over HTTP Range, so the totals are audited, not estimated on-"),
    (r"Every logical value is charged two bytes\. This baseline is hypothetical, not an available BF16 repository or a device-memory prediction\.",
     "Every explicit tensor in this repository is stored at two bytes per parameter. It is a real published checkpoint, not a projected baseline."),
    (r"\['attn\.wq_a\.weight','attn\.q_norm\.weight'\],\['attn\.q_norm\.weight','attn\.wq_b\.weight'\],\['attn\.wo_a\.weight','attn\.wo_b\.weight'\]",
     "['self_attn.q_a_proj.weight','self_attn.q_a_layernorm.weight'],['self_attn.q_a_layernorm.weight','self_attn.q_b_proj.weight'],['self_attn.o_proj.weight','mlp.gate.weight']"),
    (r"m\.id\[0\]==='E'\?TC\.engram:m\.id\[0\]==='D'\?TC\.mhc:",
     "m.part==='sparse'?COL.full:m.part==='linear'?COL.linear:m.id[0]==='D'?TC.mhc:"),
]
LITERAL_SUBS: list[tuple[str, str]] = [
    ('Model-card comparisons', 'Payload comparison'),
    ('384 experts', '288 experts'),
    ('coords=[[-2.6-rSide/2,0,0],[-1.4+eSide/2,0,0],[-1.4+eSide+2.4,0,-2.1],[-1.4+eSide+5.4,0,-2.1],[-1.4+eSide+5.4,0,2],[-1.4+eSide+2.4,0,2],[-4,0,Math.max(rSide,eSide)/2+2.4],[.6,0,Math.max(rSide,eSide)/2+2.4]]', 'sizes=CATEGORIES.map(c=>Math.cbrt(sumB(c[3],s.precision)/VOLUME_UNIT)),cols=3,gap=Math.max(...sizes,1)+2.6,rows=Math.ceil(CATEGORIES.length/cols),coords=CATEGORIES.map((c,i)=>[(i%cols-(cols-1)/2)*gap,0,(Math.floor(i/cols)-(rows-1)/2)*gap])'),
    ("this.curve(pos('vision'),pos('aligner'),TC.vision,.5,.4,.4);this.curve(pos('aligner'),pos('embed'),TC.vision,.5,.2,.6);", "this.curve(pos('vision'),pos('D0'),TC.vision,.5,.4,.4);this.curve(pos('D0'),pos('embed'),TC.mtp,.5,.2,.6);"),
    ('L00-L39', 'L00-L44'),
    ('L20-L39', 'L23-L44'),
    ('Split <small>20 + 20</small>', 'Split <small>22 + 23</small>'),
    ('40 layers, in order.', '45 layers, in order.'),
    ('40 layers.', '45 layers.'),
    ('of 40.', 'of 45.'),
    ("  if(this.state.view==='storage')this.storage();else if(this.state.view==='cache')this.cache();else this.architecture();", "  try{if(this.state.view==='storage')this.storage();else if(this.state.view==='cache')this.cache();else this.architecture();}catch(err){if(!this.viewError){this.viewError=1;console.error('atlas: '+this.state.view+' view failed:',err);}}"),
    ('aria-label="DeepSeek V4.1 Flash architecture explorer home"', 'aria-label="GLM-5.3-Flash architecture explorer home"'),
    ("'deepseek-exact-selected-shard-audit.json'", "'glm53-flash-exact-selected-shard-audit.json'"),
    ('The scene remains the labeled derived model.', 'The scene is a labelled logical model: 1 cubic unit = 1 GB.'),
    ('DEEPSEEK MODEL ATLAS', 'GLM-5.3-FLASH MODEL ATLAS'),
    ('DeepSeek Model Atlas requires JavaScript', 'GLM-5.3-Flash Model Atlas requires JavaScript'),
    ('DeepSeek Harness Minimal', 'the publisher harness'),
    ("DeepSeek-AI's card", "the publisher's card"),
    ('the complete DeepSeek training recipe', 'the complete training recipe'),
    ('not the DeepSeek training implementation', "not the publisher's training implementation"),
    ('DSPARK', 'SPARSE INDEXER'),
    ('DSpark', 'Sparse indexer'),
]
def main() -> int:
    import objectcode
    import panels
    t = templates()
    totals = {"bf16": load(BF_JSON)["payload_bytes"], "fp8": load(FP8_JSON)["payload_bytes"],
              "nvfp4": load(E0_JSON)["payload_bytes"]}
    arch, io = t["arch"], t["io"]
    key = {"bf16": "b16", "fp8": "b8", "nvfp4": "b4"}

    def page_total(mode: str) -> int:
        k = key[mode]
        total = 0
        for name in ("sparse", "linearmoe", "lineardense"):
            a = arch[name]
            total += sum(v[k] for v in a["items"].values()) * len(a["indices"])
        total += sum(v[k] for v in t["mtp"].values())
        total += sum(v[k] for v in io.values())
        total += sum(v[k] for v in t["vis"].values())
        return total

    print("data · " + " / ".join(
        f"{len(arch[n]['indices'])} {n}" for n in ("lineardense", "linearmoe", "sparse"))
        + f" + MTP({len(t['mtp'])}) + {len(t['vis'])} vision banks x24 + {len(io)} top-level")
    for m in totals:
        if page_total(m) != totals[m]:
            print(f"GATE FAIL page {m} {page_total(m):,} vs measured {totals[m]:,} "
                  f"(delta {page_total(m)-totals[m]:,})")
            return 3
    print("GATE payload == measured ✔ " + " · ".join(f"{m} {page_total(m)/1e9:.4f} GB" for m in totals))

    if "--data-only" in sys.argv:
        (DIR / "data.js").write_text(build_data_js(t), encoding="utf-8")
        print("wrote data.js")
        return 0

    data_js = build_data_js(t)
    lines = SRC.read_text(encoding="utf-8").split("\n")
    i0 = next(i for i, l in enumerate(lines) if l.startswith("const CFG = {"))
    i1 = next(i for i, l in enumerate(lines) if l.startswith("class CanvasRenderer{"))
    text = "\n".join(lines[:i0] + data_js.split("\n") + [''] + lines[i1:])
    print(f"splice · lines {i0+1}..{i1} -> data.js ({len(data_js.splitlines())} lines)")

    text, rep = objectcode.apply_rewrites(text, panels.REWRITES)
    print(f"panels · {len(rep)} whole-line rewrites applied (structure checked)")

    n_re = 0
    for pat, sub in REGEX_SUBS:
        text, k = re.subn(pat, lambda _m, sub=sub: sub, text)
        n_re += k
    lit = [(a, b) for a, b in LITERAL_SUBS if a in text]
    print(f"code   · {n_re} regex substitutions, {len(lit)} literal subs"
          f" ({len(LITERAL_SUBS) - len(lit)} superseded upstream)")
    text, rep2 = objectcode.apply_literals(text, lit)
    for r in rep2:
        print(f"         · {r[:78]}")

    must_have = ["GLM-5.3-Flash", "288 experts", "gated linear", "sparse", "indexer", "mHC",
                 "NVFP4", "FP8", "top-8", "1M-token", "multi-token-prediction", "vision tower"]
    must_not = ["DeepSeek", "DSpark", "Engram table", "CSA2", "129,280", "5,120", "384 routed",
                "890", "Top-512", "schema-derived", "hypothetical", "wo_a", "Qwen", "Edge0"]
    body = text.split("</head>", 1)[-1].replace("deepseek_sparse_attention", "<layer-type>")
    miss = [tok for tok in must_have if tok.lower() not in body.lower()]
    bad = [tok for tok in must_not if tok.lower() in body.lower()]
    if miss:
        print(f"RESIDUE FAIL missing: {miss}")
        return 5
    if bad:
        for tok in bad:
            k = body.lower().find(tok.lower())
            print(f"RESIDUE {tok!r} at {k}: ...{body[max(0,k-120):k+100]!r}...")
        return 6
    # mandatory preflight (skill: tensor-atlas-v4-retarget) — runs on the candidate page BEFORE it
    # lands on disk, and proves the checker itself can still fail (--selftest)
    import subprocess
    import tempfile
    pre = DIR / "preflight.py"
    if pre.exists():
        st = subprocess.run([sys.executable, str(pre), "--selftest"], capture_output=True, text=True)
        print((st.stdout or st.stderr).strip().splitlines()[-1] if (st.stdout or st.stderr).strip() else "self-test: no output")
        if st.returncode != 0:
            print("PREFLIGHT SELF-TEST FAILED — the guard is broken, refusing to ship", file=sys.stderr)
            return 8
        with tempfile.TemporaryDirectory() as td:
            cand = Path(td) / "candidate"
            cand.mkdir()
            (cand / "index.html").write_text(text, encoding="utf-8")
            r = subprocess.run([sys.executable, str(pre), str(cand)], capture_output=True, text=True)
            print("\n".join(r.stdout.strip().splitlines()[-12:]))
            if r.returncode != 0:
                print("PREFLIGHT FAILED — index.html NOT written", file=sys.stderr)
                return 7
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT.name} · {len(text):,} B")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
