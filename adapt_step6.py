#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 6: retarget the 40-layer leftovers to this model's 45 stack layers.

The engine builds its layout toggle from two quoted fragments, so the anchors must include the
markup that surrounds them (a bare "Tower L00-L39" never appears in src.html).
"""
import ast
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
src = (HERE / "src.html").read_text(encoding="utf-8")
P = HERE / "build.py"
s = P.read_text(encoding="utf-8")

CANDIDATES = [
    ("Tower <small>L00-L39</small>", "Tower <small>L00-L44</small>"),
    ("L00-L39 / one inventory", "L00-L44 / one inventory"),
    ("L00-L39</small>", "L00-L44</small>"),
    ("L20-L39 then head. No layers skipped.", "L23-L44 then head. No layers skipped."),
    ("L20-L39 then the output head.", "L23-L44 then the output head."),
    ("L20-L39", "L23-L44"),
    ("L00-L39", "L00-L44"),
    ("40 layers, in order.", "45 layers, in order."),
    ("40 layers.", "45 layers."),
    ("of 40.", "of 45."),
    ('Split <small>', 'Split <small>'),  # placeholder, resolved below
]

# the split button's numbers: find what the engine actually printed there
import re
m = re.search(r"Split <small>([^<]{0,20})</small>", src)
split_label = m.group(1) if m else None
print("split button label in src.html:", repr(split_label))
PAIRS = [p for p in CANDIDATES if p[0] != "Split <small>"]
if split_label:
    PAIRS.insert(0, (f"Split <small>{split_label}</small>", "Split <small>22 + 23</small>"))

present = [(a, b) for a, b in PAIRS if src.count(a) >= 1]
absent = [a for a, _ in PAIRS if src.count(a) == 0]
print("applying:", [a[:44] for a, _ in present])
print("absent (covered by a longer anchor):", [a[:44] for a in absent])
if not present:
    raise SystemExit("no label anchors matched — nothing to do")

entry = "".join(f"    ({a!r}, {b!r}),\n" for a, b in present)
head = "LITERAL_SUBS: list[tuple[str, str]] = [\n"
i = s.index(head) + len(head)
s = s[:i] + entry + s[i:]
ast.parse(s)
P.write_text(s, encoding="utf-8")
print(f"added {len(present)} layer-count literals")
