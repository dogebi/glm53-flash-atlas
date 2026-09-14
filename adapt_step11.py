#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 11: make the Benchmarks tab's selector functional.

The panel kept the engine's `<select>` bound to `state.bench` but rendered the same three bars
whatever was selected, so the control looked dead. The select now picks a *component*
(CATEGORIES) and the bars show what each published precision stores for it — measured payloads,
which is all this atlas can honestly show.
"""
import ast
import pathlib

HERE = pathlib.Path("/mnt/d/dev/glm53-flash-atlas")
PANELS = HERE / "panels.py"
BUILD = HERE / "build.py"

NEW_JS = (' renderBench(){const s=this.state,cat=CATEGORIES[s.bench]||CATEGORIES[0],ws=cat[3],'
          "base=Math.max(sumB(ws,'bf16'),1);return html`<div><div class=\"inspect-path\">"
          'COMPONENT PAYLOAD / MEASURED FROM SHARD HEADERS</div><h2 class="editorial">Bytes,<br/>'
          '<em>not scores.</em></h2><p class="lede">Pick a component and compare what each published '
          'precision stores for it. This page reads shard headers; it claims no capability numbers.</p>'
          '<label class="field-label">Compare a component<select value=${s.bench} '
          'onChange=${e=>this.patch({bench:Number(e.target.value)})}>'
          '${CATEGORIES.map((c,i)=>html`<option value=${i}>${c[1]}</option>`)}</select></label>'
          '<div class="benchmark-chart">${MODE_KEYS.map(mode=>{const b=sumB(ws,mode);return html`'
          '<div class="benchmark-row"><div><span>${MODE_INFO[mode].label}</span><b>${bytes(b,s.binary)}'
          '</b></div><div class="bench-track"><i style=${{width:Math.max(0,Math.min(100,b/base*100))'
          "+'%',background:MODE_INFO[mode].color,opacity:mode===s.precision?1:.4}}/></div></div>`;})}"
          '</div><p class="fine">${cat[1]} \\u2014 ${bytes(sumB(ws,s.precision),s.binary)} in '
          '${MODE_INFO[s.precision].label}, '
          "${(sumB(ws,s.precision)/Math.max(TOTALS[s.precision],1)*100).toFixed(1)}% of that mode's "
          '${bytes(TOTALS[s.precision],s.binary)}. Bars are scaled to the same component at BF16 so '
          'the three modes stay comparable.</p>')

lines = PANELS.read_text(encoding="utf-8").split("\n")
i = next(k for k, l in enumerate(lines) if "REPORTED EVALUATIONS / NATIVE MODEL" in l)
print("entry anchor line:", i + 1)
print("replacing literal on line", i + 2, "->", lines[i + 1][:60], "...")
lines[i + 1] = "  " + repr(NEW_JS) + "),"
out = "\n".join(lines)
ast.parse(out)
PANELS.write_text(out, encoding="utf-8")
print("panels.py: renderBench is selector-driven")

# the tab label should say what the tab now does
src = (HERE / "src.html").read_text(encoding="utf-8")
assert src.count("Model-card comparisons") == 1
b = BUILD.read_text(encoding="utf-8")
head = "LITERAL_SUBS: list[tuple[str, str]] = [\n"
k = b.index(head) + len(head)
b = b[:k] + "    ('Model-card comparisons', 'Payload comparison'),\n" + b[k:]
ast.parse(b)
BUILD.write_text(b, encoding="utf-8")
print("build.py: tab eyebrow renamed to 'Payload comparison'")
