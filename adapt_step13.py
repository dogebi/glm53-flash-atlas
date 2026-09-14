#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 13: bring back the original benchmark panel (nisten's options set) on the GLM atlas.

The request: the Benchmarks dropdown must offer what the original page offers — the published
benchmark comparison, not a component picker. So this restores the original BENCH table and
BENCH_MODELS row set (7 models, one of which is the GLM-5.3 line) and the original chart, with
attribution that makes clear these are publisher/leaderboard figures, not this atlas's measurements.
"""
import ast
import pathlib
import re

HERE = pathlib.Path("/mnt/d/dev/glm53-flash-atlas")
SRC = (HERE / "src.html").read_text(encoding="utf-8")
BUILD = HERE / "build.py"
PANELS = HERE / "panels.py"

# ---- 1. the original tables ---------------------------------------------------------------
m_bench = re.search(r"const BENCH=\[.*?\]\];", SRC, re.S)
m_models = re.search(r"const BENCH_MODELS=\[.*?\];", SRC, re.S)
assert m_bench and m_models, "original BENCH tables not found in src.html"
orig_bench = m_bench.group(0)
orig_models = m_models.group(0)
print("original benchmarks:", re.findall(r"\['([^']+)'", orig_bench))
print("original models    :", m_models.group(0)[:120], "...")

b = BUILD.read_text(encoding="utf-8")
i = b.index("const BENCH=[")
j = b.index("function pickExperts(")
new_tables = (orig_bench + "\n" + orig_models + "\n"
              "/* This atlas measures bytes; the rows above are the comparison set published by the\n"
              "   original atlas page (each model's own card), reproduced so the selector keeps the\n"
              "   same options. GLM-5.3-Flash publishes no scores in the card this build reads. */\n")
b = b[:i] + new_tables + b[j:]

# ---- 2. the panel: original chart, GLM row featured, honest attribution --------------------
m_panel = re.search(r" renderBench\(\)\{.*?\n", SRC)
assert m_panel, "original renderBench line not found"
panel = m_panel.group(0).rstrip("\n")
panel = panel.replace("REPORTED EVALUATIONS / NATIVE MODEL",
                      "PUBLISHER-REPORTED COMPARISON / ORIGINAL SET")
panel = panel.replace("Capability,<br/><em>with context.</em>", "Capability,<br/><em>as published.</em>")
panel = panel.replace("The supplied model-card results at maximum reasoning effort. No synthetic benchmark projections.",
                      "Scores collected from each model\\u2019s own card by the original atlas page. This page measures bytes; it claims no scores of its own, and GLM-5.3-Flash is shown by its GLM-5.3 row.")
# feature the GLM row (index 3 in the published set) instead of the original subject (index 6)
panel = panel.replace("i===6?' featured':''", "i===3?' featured':''")
panel = panel.replace("i===6?COL.dec:COL.enc", "i===3?COL.dec:COL.enc")
panel = panel.replace("i===6?1:.32", "i===3?1:.32")
assert "i===3?' featured':''" in panel and "GLM-5.3 row" in panel
print("panel retargeted, length", len(panel))

lines = PANELS.read_text(encoding="utf-8").split("\n")
k = next(i for i, l in enumerate(lines) if "REPORTED EVALUATIONS / NATIVE MODEL" in l)
lines[k + 1] = "  " + repr(panel) + "),"
out = "\n".join(lines)
ast.parse(out)
PANELS.write_text(out, encoding="utf-8")

# write the data-half edit (the earlier code path re-read the file and lost it)
ast.parse(b)
BUILD.write_text(b, encoding="utf-8")
print("build.py: original BENCH table + BENCH_MODELS restored")
print("panels.py renderBench restored to the original chart")
