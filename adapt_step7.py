#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 7: replace the overlapping layer-count literals with a non-overlapping set.

apply_literals applies entries in order and expects each to match, so anchors must not contain one
another: keep the generic forms only ("L00-L39" covers "Tower <small>L00-L39</small>").
"""
import ast
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
src = (HERE / "src.html").read_text(encoding="utf-8")
P = HERE / "build.py"
lines = P.read_text(encoding="utf-8").split("\n")

PAIRS = [
    ("L00-L39", "L00-L44"),
    ("L20-L39", "L23-L44"),
    ("Split <small>20 + 20</small>", "Split <small>22 + 23</small>"),
    ("40 layers, in order.", "45 layers, in order."),
    ("40 layers.", "45 layers."),
    ("of 40.", "of 45."),
]
for a, b in PAIRS:
    print(f"  src hits={src.count(a):<2} {a!r}")

# drop every earlier layer-count literal, then insert this set
drop_tokens = ["L00-L39", "L20-L39", "20 + 20", "40 layers", "of 40."]
kept = [l for l in lines if not (l.strip().startswith("(") and any(t in l for t in drop_tokens))]
print("literal lines dropped:", len(lines) - len(kept))

head = "LITERAL_SUBS: list[tuple[str, str]] = [\n"
i = next(k for k, l in enumerate(kept) if l.startswith(head.rstrip()))
entry = "".join(f"    ({a!r}, {b!r}),\n" for a, b in PAIRS)
kept.insert(i + 1, entry.rstrip("\n"))
out = "\n".join(kept)
ast.parse(out)
P.write_text(out, encoding="utf-8")
print("layer-count literals rewritten:", len(PAIRS))
