#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 5: wrap the engine's view dispatch in try/catch (skill pitfall: one bad view must not kill
the animation loop). Also add the flow-phase label fixes the preflight expects to stay resolvable."""
import ast
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
P = HERE / "build.py"
s = P.read_text(encoding="utf-8")

raw = "  if(this.state.view==='storage')this.storage();else if(this.state.view==='cache')this.cache();else this.architecture();"
wrapped = ("  try{if(this.state.view==='storage')this.storage();else if(this.state.view==='cache')this.cache();"
           "else this.architecture();}catch(err){if(!this.viewError){this.viewError=1;"
           "console.error('atlas: '+this.state.view+' view failed:',err);}}")

src = (HERE / "src.html").read_text(encoding="utf-8")
assert src.count(raw) == 1, f"dispatch anchor found {src.count(raw)} times"

i = s.index("LITERAL_SUBS: list[tuple[str, str]] = [")
entry = f"    ({raw!r}, {wrapped!r}),\n"
s = s[:i + len("LITERAL_SUBS: list[tuple[str, str]] = [\n")] + entry + s[i + len("LITERAL_SUBS: list[tuple[str, str]] = [\n"):]
ast.parse(s)
P.write_text(s, encoding="utf-8")
print("view dispatch wrapped; LITERAL_SUBS now has the guard")
