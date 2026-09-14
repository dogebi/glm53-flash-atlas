#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 8: teach the preflight that a missing palette *key* is a fatal data slip.

Shipped incident: `COL.linear` was read by the engine while the data half defined `COL.lin`, so
`emit()` did `undefined.map(...)`, the view-dispatch try/catch swallowed it, and the Architecture
view rendered empty with no visible error. The interpolation check catches a missing constant; this
adds the object-key sibling of that check, to the skill script and to every atlas copy.
"""
import ast
import pathlib
import re

CHECK = '''
    # 11: every palette key the engine reads must be defined in the data half
    palette_problems: list[str] = []
    for name in ("COL", "TC"):
        mm = re.search(rf"const {name}=\\{{([^}}]*)\\}}", body)
        if not mm:
            palette_problems.append(f"{name} object not found")
            continue
        defined = set(re.findall(r"([A-Za-z_$][\\w$]*)\\s*:", mm.group(1)))
        used = set(re.findall(rf"\\b{name}\\.([A-Za-z_$][\\w$]*)", body))
        miss_keys = sorted(used - defined)
        if miss_keys:
            palette_problems.append(f"{name} read but undefined: {miss_keys}")
    report(not palette_problems, "palette keys the engine reads are defined",
           "; ".join(palette_problems))
'''

ANCHOR = '    report("sumB(ALL_W" in body, "totals are derived from the data half")'
CANDIDATES = sorted(pathlib.Path("/mnt/d/dev/cscAI/twin-company").glob("*/preflight.py")) + \
    [pathlib.Path("/home/nttcom/.hermes/skills/documentation/tensor-atlas-v4-retarget/scripts/atlas_preflight.py")]

for p in CANDIDATES:
    if not p.exists():
        continue
    s = p.read_text(encoding="utf-8")
    if "palette keys the engine reads are defined" in s:
        print(f"already present: {p}")
        continue
    if ANCHOR not in s:
        print(f"anchor missing (skipped): {p}")
        continue
    s = s.replace(ANCHOR, ANCHOR + "\n" + CHECK.rstrip("\n"), 1)
    ast.parse(s)
    p.write_text(s, encoding="utf-8")
    print(f"updated: {p}")
