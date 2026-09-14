#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 12: category buckets and MTP labels.

1. `cat_of()` received *layer-relative* names (`mlp.gate.weight`, `mlp.shared_experts...`) because
   `templates()` strips the layer prefix, so the full-name patterns matched nothing: every router
   fell through to the "norm" fallback and the shared expert was bucketed as a routed expert
   (`experts` matched before `shared_expert`). The page therefore showed an empty
   "Routers and gates" and an inflated expert bucket.
2. The MTP dict comprehension prefixed an already-full key, emitting names like
   `model.language_model.layers.45.model.language_model.layers.45.mlp.gate.weight`.
"""
import ast
import pathlib

P = pathlib.Path("/mnt/d/dev/glm53-flash-atlas/build.py")
s = P.read_text(encoding="utf-8")

old_cat = '''CAT = [(r"experts", "expert"), (r"\\.mlp\\.gate", "router"),
       (r"indexer", "index"), (r"self_attn|linear_attn", "attn"),
       (r"shared_expert", "shared"), (r"hc_(attn|ffn)", "mhc"),
       (r"norm|_log$|dt_bias|_ape$", "norm")]'''
new_cat = '''# names arrive either layer-relative ("mlp.gate.weight") or fully qualified
# ("model.language_model.layers.3.mlp.gate.weight"), so anchors must not require a leading dot;
# shared_expert must be tested before the routed-expert pattern.
CAT = [(r"shared_expert", "shared"),
       (r"(?:^|\\.)mlp\\.experts|switch_mlp", "expert"),
       (r"(?:^|\\.)mlp\\.gate", "router"),
       (r"indexer", "index"), (r"self_attn|linear_attn", "attn"),
       (r"hc_(attn|ffn)", "mhc"),
       (r"norm|_log$|dt_bias|_ape$", "norm")]'''
assert old_cat in s, "CAT not found"
s = s.replace(old_cat, new_cat)

old_mtp = '''    mtp = {f"model.language_model.layers.45.{k}": v
           for k, v in both.items() if k.startswith("model.language_model.layers.45.")}'''
new_mtp = '''    mtp = {k: v for k, v in both.items()
           if k.startswith("model.language_model.layers.45.")}'''
assert old_mtp in s, "mtp comprehension not found"
s = s.replace(old_mtp, new_mtp)

ast.parse(s)
P.write_text(s, encoding="utf-8")
print("cat_of buckets fixed; MTP keys no longer double-prefixed")
