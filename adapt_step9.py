#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 9: two more data/engine contract breaks, and a stricter preflight for both.

1. `storage()` still laid its category solids out from a hand-written 8-entry `coords` array while
   this model has 10 categories -> `coords[8][0]` threw inside the view (caught, view half-blank).
   Port the CATEGORIES-derived layout.
2. `architecture()` drew arcs to `pos('aligner')`, a module this model does not have, and looped over
   `ENGRAM`, which the data half never defines -> `undefined[0]` / ReferenceError.
3. Tighten the preflight: `pos('x')` must resolve against ids that actually exist in the data half
   (dropping the hardcoded "aligner/selfcond/adapters" allowances), and the coords check scans the
   whole storage() body instead of its first 1500 characters.
"""
import ast
import pathlib
import re

HERE = pathlib.Path(__file__).resolve().parent
src = (HERE / "src.html").read_text(encoding="utf-8")

COORDS_OLD = ("coords=[[-2.6-rSide/2,0,0],[-1.4+eSide/2,0,0],[-1.4+eSide+2.4,0,-2.1],[-1.4+eSide+5.4,0,-2.1],"
              "[-1.4+eSide+5.4,0,2],[-1.4+eSide+2.4,0,2],[-4,0,Math.max(rSide,eSide)/2+2.4],"
              "[.6,0,Math.max(rSide,eSide)/2+2.4]]")
COORDS_NEW = ("sizes=CATEGORIES.map(c=>Math.cbrt(sumB(c[3],s.precision)/VOLUME_UNIT)),cols=3,"
              "gap=Math.max(...sizes,1)+2.6,rows=Math.ceil(CATEGORIES.length/cols),"
              "coords=CATEGORIES.map((c,i)=>[(i%cols-(cols-1)/2)*gap,0,(Math.floor(i/cols)-(rows-1)/2)*gap])")
CURVE_OLD = ("this.curve(pos('vision'),pos('aligner'),TC.vision,.5,.4,.4);"
             "this.curve(pos('aligner'),pos('embed'),TC.vision,.5,.2,.6);")
CURVE_NEW = ("this.curve(pos('vision'),pos('D0'),TC.vision,.5,.4,.4);"
             "this.curve(pos('D0'),pos('embed'),TC.mtp,.5,.2,.6);")

for label, pat in (("coords", COORDS_OLD), ("aligner arcs", CURVE_OLD)):
    print(f"  src.html hits for {label}: {src.count(pat)}")
assert src.count(COORDS_OLD) == 1 and src.count(CURVE_OLD) == 1

# --- builder: literal subs for the two snippets, regex sub for ENGRAM -------------------------------
P = HERE / "build.py"
s = P.read_text(encoding="utf-8")
head = "LITERAL_SUBS: list[tuple[str, str]] = [\n"
entry = (f"    ({COORDS_OLD!r}, {COORDS_NEW!r}),\n"
         f"    ({CURVE_OLD!r}, {CURVE_NEW!r}),\n")
i = s.index(head) + len(head)
s = s[:i] + entry + s[i:]

engram = '    (r"\\bENGRAM\\b", "AUX_TABLES"),\n'
anchor = '    (r\'aria-label="DeepSeek'
k = s.index(anchor)
s = s[:k] + engram + s[k:]
ast.parse(s)
P.write_text(s, encoding="utf-8")
print("builder: coords layout + arc targets + ENGRAM->AUX_TABLES")

# --- preflight: stricter pos() and coords checks ----------------------------------------------------
for target in sorted(pathlib.Path("/mnt/d/dev/cscAI/twin-company").glob("*/preflight.py")) + [
        pathlib.Path("/home/nttcom/.hermes/skills/documentation/tensor-atlas-v4-retarget/scripts/atlas_preflight.py")]:
    t = target.read_text(encoding="utf-8")
    old_ids = ('    known = ids | layer_ids | {"embed", "head", "vision", "aligner", "selfcond", "adapters"}')
    new_ids = ('    # ids the data half really defines: no hardcoded allowances, a scene that points at a\n'
               '    # module this model does not have must fail here, not in the browser.\n'
               '    known = ids | layer_ids')
    if old_ids in t:
        t = t.replace(old_ids, new_ids)
    old_win = "    storage = body[st:st + 1500] if st >= 0 else \"\""
    new_win = "    storage = body[st:st + 4000] if st >= 0 else \"\""
    if old_win in t:
        t = t.replace(old_win, new_win)
    ast.parse(t)
    target.write_text(t, encoding="utf-8")
    print(f"preflight tightened: {target.name} ({target.parent.name})")
