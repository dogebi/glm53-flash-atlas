#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sanity: category buckets vs the measured payloads (router 0 B? MTP nvfp4 == bf16?)."""
import importlib.util
import re
import sys

HERE = "/mnt/d/dev/glm53-flash-atlas"
spec = importlib.util.spec_from_file_location("b", f"{HERE}/build.py")
b = importlib.util.module_from_spec(spec)
sys.modules["b"] = b
spec.loader.exec_module(b)

t = b.templates()
io, vis = t["io"], t["vis"]

rows = []
for name in ("sparse", "linearmoe", "lineardense"):
    a = t["arch"][name]
    for n, v in a["items"].items():
        rows.append((n, v, len(a["indices"])))
mtp = [(n, v, 1) for n, v in t["mtp"].items()]
io_rows = [(n, v, 1) for n, v in io.items()]
vis_rows = [(n, v, len(t["vis"])) for n, v in vis.items()]

def buckets(item):
    n = item[0]
    if "experts" in n:
        return "expert"
    if "mlp.gate" in n:
        return "router"
    if "indexer" in n:
        return "index"
    if "self_attn" in n or "linear_attn" in n:
        return "attn"
    if "shared_expert" in n:
        return "shared"
    if "hc_" in n:
        return "mhc"
    if "visual" in n:
        return "vision"
    return "norm"

agg = {}
for group, label in ((rows, "layers"), (mtp, "mtp"), (io_rows, "io"), (vis_rows, "vision_banks")):
    for n, v, count in group:
        k = buckets((n, v))
        a = agg.setdefault(k, {"count": 0, "bf16": 0, "fp8": 0, "nvfp4": 0})
        a["count"] += 1
        a["bf16"] += v["b16"] * (count if n.startswith("model.visual.blocks.*") else 1)
        a["fp8"] += v["b8"] * (count if n.startswith("model.visual.blocks.*") else 1)
        a["nvfp4"] += v["b4"] * (count if n.startswith("model.visual.blocks.*") else 1)

print(f"{'bucket':<10} {'rows':>5} {'bf16 GB':>9} {'fp8 GB':>8} {'nvfp4 GB':>9}")
for k, a in sorted(agg.items()):
    print(f"{k:<10} {a['count']:>5} {a['bf16']/1e9:>9.3f} {a['fp8']/1e9:>8.3f} {a['nvfp4']/1e9:>9.3f}")

router = [n for n, _, _ in rows + mtp + io_rows if "mlp.gate" in n]
print("\nrouter rows:", len(router), router[:3])
mtp_n = [n for n, _, _ in mtp]
print("mtp rows:", len(mtp_n), "| nvfp4 == bf16 ?",
      all(abs(t['mtp'][n]['b4'] - t['mtp'][n]['b16']) < 1 for n in mtp_n))
# is the MTP block excluded from the NVFP4 recipe?
import json
cfg = json.load(open(f"{HERE}/hf-config-nvfp4.json"))
q = cfg.get("quantization_config") or {}
print("nvfp4 ignore:", str(q.get("ignore"))[:200])
print("nvfp4 targets:", str((q.get("config_groups") or {}).get("group_0", {}).get("targets"))[:200])
