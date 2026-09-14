#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Why do layer-3 expert banks carry ~0 bytes in the bf16 mode?"""
import importlib.util
import json
import pathlib
import re
import sys

HERE = pathlib.Path("/mnt/d/dev/cscAI/twin-company/glm53-flash-atlas")
spec = importlib.util.spec_from_file_location("b", HERE / "build.py")
b = importlib.util.module_from_spec(spec)
sys.modules["b"] = b
spec.loader.exec_module(b)

bf = b.load(b.BF_JSON)["tensors"]
fp8 = b.load(b.FP8_JSON)["tensors"]
nv = b.load(b.E0_JSON)["tensors"]

for label, tens in (("bf16", bf), ("fp8", fp8), ("nvfp4", nv)):
    ex = [(n, t) for n, t in tens.items() if re.match(r"^model\.language_model\.layers\.3\.mlp\.experts\.", n)]
    tot = sum(t["bytes"] for _, t in ex)
    print(f"--- {label}: {len(ex)} raw expert tensors in layer 3, {tot/1e9:.4f} GB")
    for n, t in ex[:3]:
        print(f"      {n}  {t['shape']}  {t['bytes']:,}")

folder = b.make_folder(set(bf))
print("\nfold of layer-3 expert tensors:")
for label, tens in (("bf16", bf), ("fp8", fp8), ("nvfp4", nv)):
    for n, t in list(tens.items()):
        if re.match(r"^model\.language_model\.layers\.3\.mlp\.experts\.0\.", n):
            print(f"  {label:<6} {n:<70} -> {folder(n)}")
