#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 10: make the expert fold win over the raw-name passthrough (NaN layer detail bug).

`folder()` tried `c in bfset` *before* the expert family fold, so in the BF16 mode (the name set the
folder is built from) every per-expert tensor passed through unchanged: 864 raw names per layer
instead of two banks. Only the FP8/NVFP4 scale tensors folded, which left `experts.gate_up_proj`
with `bf16: 0` — the layer detail then divided by that zero and filled 926 of 941 solids with NaN
coordinates, so clicking a layer rendered nothing at all (no exception, no console error).

Also: the bank's *shape* must come from the `.weight` tensor, not from a scale tensor, otherwise the
bank's parameter count collapses to something like [288, 512].
"""
import ast
import pathlib

P = pathlib.Path(__file__).resolve().parent / "build.py"
s = P.read_text(encoding="utf-8")

old = '''        stripped = SCALE_SUFFIX.sub("", n)
        for c in (n, stripped, stripped + ".weight"):
            if c in bfset:
                return c
            f = fold_expert(c)
            if f:
                return f
        return None'''
new = '''        # family folds come first: a per-expert tensor is one bank entry in every mode, even in the
        # mode whose own name set we are folding against (otherwise the raw name passes through and
        # the bank ends up with zero bytes in that mode -> NaN geometry downstream).
        f = fold_expert(n) or fold_expert(SCALE_SUFFIX.sub("", n))
        if f:
            return f
        stripped = SCALE_SUFFIX.sub("", n)
        for c in (n, stripped, stripped + ".weight"):
            if c in bfset:
                return c
        return None'''
assert old in s, "folder body not found"
s = s.replace(old, new)

old_shapes = '''            if lg.endswith("experts.gate_up_proj") or lg.endswith("experts.down_proj"):
                shapes[lg] = [N_EXP, *sh] if len(sh) == 2 else sh
            elif len(sh) <= 2:
                shapes[lg] = sh'''
new_shapes = '''            if lg.endswith("experts.gate_up_proj") or lg.endswith("experts.down_proj"):
                # the bank's shape comes from the weight tensor; scale/input_scale tensors are empty
                # or block-shaped and would collapse the bank's parameter count
                if not raw.endswith(".weight"):
                    continue
                shapes[lg] = [N_EXP, *sh] if len(sh) == 2 else sh
            elif len(sh) <= 2:
                shapes[lg] = sh'''
assert old_shapes in s, "shapes body not found"
s = s.replace(old_shapes, new_shapes)

ast.parse(s)
P.write_text(s, encoding="utf-8")
print("expert fold now precedes the raw-name passthrough; bank dims come from .weight")
