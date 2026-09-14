#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 1 of the GLM-5.3-Flash retarget: folding + module model in build.py.

Applies the model-specific half of the adaptation to the copied edge0 builder:
paths, fold rules (per-expert -> fused banks, unconditional), three layer archetypes,
the MTP layer, the 24-block vision tower, GLM categories and the GLM head of the data half.
"""
import pathlib
import re
import sys

P = pathlib.Path(__file__).resolve().parent / "build.py"
s = P.read_text(encoding="utf-8")
n = 0


def sub(old: str, new: str, count: int = 1, must: bool = True) -> None:
    global s, n
    if s.count(old) < count:
        if must:
            raise SystemExit(f"anchor missing ({s.count(old)}x): {old[:90]!r}")
        return
    s = s.replace(old, new, count)
    n += 1


# ---- 1. paths ------------------------------------------------------------------------------
sub('BF_JSON, FP8_JSON, E0_JSON = DIR / "hf-bf16.json", DIR / "hf-fp8.json", DIR / "hf-edge0.json"',
    'BF_JSON, FP8_JSON, E0_JSON = DIR / "hf-bf16.json", DIR / "hf-fp8.json", DIR / "hf-nvfp4.json"')
sub('CFG_BASE, CFG_FP8, CFG_E0 = DIR / "hf-config-base.json", DIR / "hf-config-fp8.json", DIR / "hf-config-edge0.json"',
    'CFG_BASE, CFG_FP8, CFG_E0 = DIR / "hf-config-bf16.json", DIR / "hf-config-fp8.json", DIR / "hf-config-nvfp4.json"')

# ---- 2. fold: experts always fold into fused banks; GLM has no LoRA / prerouter ------------
sub('''            f = fold_expert(c)
            if f and f in bfset:
                return f''',
    '''            f = fold_expert(c)
            if f:
                return f''')
sub('''LORA = re.compile(r"^model\\.language_model\\.layers\\.\\d+\\.(.+)\\.lora_([AB])$")
PREROUTER = re.compile(r"^layers\\.(\\d+)\\.(fc1|fc2|linear_init)\\.weight$")''',
    '''# GLM-5.3-Flash ships no adapters: nothing to reuse from the edge0 fold table.''')

# ---- 3. templates(): 45 stack layers + MTP layer 45, three archetypes, 24-row vision -------
sub('''    def logical_dims(name: str) -> list[int]:
        """Shape of the logical tensor, taken from a mode that stores it directly."""
        for src in (BF_JSON, FP8_JSON, E0_JSON):
            for raw, t in load(src)["tensors"].items():
                if folder(raw) == name:
                    sh = t["shape"]
                    if name.endswith("experts.gate_up_proj") or name.endswith("experts.down_proj"):
                        if len(sh) == 3:      # fused bank [E, out, in]
                            return sh
                    elif name.endswith("switch_mlp"):
                        return sh
                    elif len(sh) <= 2:
                        return sh
        return [1]''',
    '''    N_EXP = load(CFG_BASE).get("text_config", load(CFG_BASE)).get("n_routed_experts", 288)

    def logical_dims(name: str) -> list[int]:
        """Shape of the logical tensor; expert banks are synthesized as [E, out, in]."""
        for src in (BF_JSON, FP8_JSON, E0_JSON):
            for raw, t in load(src)["tensors"].items():
                if folder(raw) == name:
                    sh = t["shape"]
                    if name.endswith("experts.gate_up_proj") or name.endswith("experts.down_proj"):
                        return [N_EXP, *sh] if len(sh) == 2 else sh   # per-expert [out, in]
                    if len(sh) <= 2:
                        return sh
        return [1]''')

sub('''    sigs: dict[str, dict] = {}
    for i in range(40):
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
        dom = "linear" if "linear_attn.in_proj_qkv.weight" in s["items"] else "full"
        arch[f"lm_{dom}"] = s
    if len(arch) != 2:
        raise SystemExit(f"expected two layer archetypes, got {list(arch)}")''',
    '''    sigs: dict[str, dict] = {}
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
    mtp = {f"model.language_model.layers.45.{k}": v
           for k, v in both.items() if k.startswith("model.language_model.layers.45.")}
    if not mtp:
        raise SystemExit("no MTP layer (expected index 45)")''')

sub('''    counts = {k: len(v) for k, v in vis_layers.items()}
    if counts and set(counts.values()) != {27}:
        raise SystemExit(f"visual blocks are not uniform: {counts}")''',
    '''    counts = {k: len(v) for k, v in vis_layers.items()}
    if counts and len(set(counts.values())) != 1:
        raise SystemExit(f"visual blocks are not uniform: {counts}")
    n_vis = next(iter(counts.values())) if counts else 0
    print(f"  vision blocks: {n_vis} rows x {len(counts)} tensors")''')

sub('''        vis[f"model.visual.blocks.*.{suffix}"] = {"dims": v["dims"], "count": 27,
                                                 "b16": v["b16"] * 27, "b8": v["b8"] * 27, "b4": v["b4"] * 27}''',
    '''        vis[f"model.visual.blocks.*.{suffix}"] = {"dims": v["dims"], "count": n_vis,
                                                 "b16": v["b16"] * n_vis, "b8": v["b8"] * n_vis,
                                                 "b4": v["b4"] * n_vis}''')

sub('''    return {"arch": arch, "vis": vis, "io": io}''',
    '''    return {"arch": arch, "vis": vis, "io": io, "mtp": mtp}''')

# ---- 4. categories ------------------------------------------------------------------------
sub('''CAT = [(r"experts|switch_mlp", "expert"), (r"\\.mlp\\.gate", "router"), (r"self_attn|linear_attn", "attn"),
       (r"shared_expert", "shared"), (r"norm|_log$|dt_bias|\\.bias", "norm"), (r"lora|prerouter", "adapter")]''',
    '''CAT = [(r"experts", "expert"), (r"\\.mlp\\.gate", "router"),
       (r"indexer", "index"), (r"self_attn|linear_attn", "attn"),
       (r"shared_expert", "shared"), (r"hc_(attn|ffn)", "mhc"),
       (r"norm|_log$|dt_bias|_ape$", "norm")]''')

sub('''def cat_of(name: str) -> str:
    if "lora_" in name:
        return "lora"
    if name.startswith("prerouter"):
        return "prerouter"
    if "visual" in name:''',
    '''def cat_of(name: str) -> str:
    if "visual" in name:''')

print(f"step1 patches applied: {n}")
P.write_text(s, encoding="utf-8")
