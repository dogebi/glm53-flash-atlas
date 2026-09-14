#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 3: move engine-copy fixes that contain backslash-quote pairs into LITERAL_SUBS.

Regex escaping around `\\"` (a backslash-quote as it appears inside the engine's JSX template
literals) is easy to get wrong; plain-string replacement with a count check is not. Every pair is
asserted to exist in src.html before it is written into the builder.
"""
import ast
import pathlib
import re

HERE = pathlib.Path(__file__).resolve().parent
B = HERE / "build.py"
BS = chr(92)
Q = '"'

src = (HERE / "src.html").read_text(encoding="utf-8")

# (literal text in src.html, replacement) — built with explicit backslash counts
PAIRS = [
    ('aria-label="DeepSeek V4.1 Flash architecture explorer home"',
     'aria-label="GLM-5.3-Flash architecture explorer home"'),
    ("'deepseek-exact-selected-shard-audit.json'", "'glm53-flash-exact-selected-shard-audit.json'"),
    ("The scene remains the labeled derived model.", "The scene is a labelled logical model: 1 cubic unit = 1 GB."),
    ("Schema-derived tensor payload estimate", "Measured tensor payload, audited from shard headers"),
    ("SCHEMA-DERIVED PAYLOAD", "MEASURED PAYLOAD"),
    ("<h1>DeepSeek <em>V4.1 Flash</em>", "<h1>GLM <em>5.3 Flash</em>"),
    ("DeepSeek-ViT", "GLM-ViT"),
    ("DEEPSEEK MODEL ATLAS", "GLM-5.3-FLASH MODEL ATLAS"),
    ("DeepSeek Model Atlas requires JavaScript", "GLM-5.3-Flash Model Atlas requires JavaScript"),
    ("DeepSeek Harness Minimal", "the publisher harness"),
    ("DeepSeek-AI's card", "the publisher's card"),
    ("the complete DeepSeek training recipe", "the complete training recipe"),
    ("not the DeepSeek training implementation", "not the publisher's training implementation"),
    ("DSPARK", "SPARSE INDEXER"),
    ("DSpark", "Sparse indexer"),
]
missing = [a for a, _ in PAIRS if src.count(a) == 0]
if missing:
    raise SystemExit(f"these literals are absent from src.html: {missing}")

text = B.read_text(encoding="utf-8")
# drop the fragile regex entries for exactly these strings
dropped = 0
for a, _ in PAIRS:
    for idx_re in (a, a.replace("'", chr(92) + "'")):
        pat = f"    (r{repr(idx_re)}, {repr('__drop__')}),"
        if pat in text:
            text = text.replace(pat, "")
            dropped += 1
# rebuild LITERAL_SUBS
lit_lines = [f"    ({a!r}, {b!r})," for a, b in PAIRS]
i = text.index("LITERAL_SUBS")
j = text.index("\n", text.index("= []", i))
text = text[:i] + "LITERAL_SUBS: list[tuple[str, str]] = [\n" + "\n".join(lit_lines) + "\n]\n" + text[j + 1:]
ast.parse(text)
B.write_text(text, encoding="utf-8")
print(f"literal subs: {len(PAIRS)} (regex entries dropped by exact match: {dropped})")
for a, b in PAIRS:
    print(f"  {a[:58]!r} -> {b[:52]!r}")
