#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 4: make the literal stage defensive (redundant entries are reported, not fatal).

REGEX_SUBS and the panels stage already rewrite some of the same strings, so a literal entry can
legitimately have zero matches by the time it runs. The residue gate at the end of main() is what
guards the copy; here we only filter, and report how many entries were superseded.
"""
import ast
import pathlib

P = pathlib.Path(__file__).resolve().parent / "build.py"
s = P.read_text(encoding="utf-8")

old_call = "    text, rep2 = objectcode.apply_literals(text, LITERAL_SUBS)"
new_call = "\n".join([
    "    # only the literals that survived the panels + regex stages: a few are made redundant",
    "    # upstream, and the residue gate below is what actually guards the finished copy.",
    "    lit = [(a, b) for a, b in LITERAL_SUBS if a in text]",
    "    text, rep2 = objectcode.apply_literals(text, lit)",
])
assert old_call in s, "apply_literals call not found"
s = s.replace(old_call, new_call)

old_print = 'f"code   · {n_re} regex substitutions, {len(LITERAL_SUBS)} literal subs"'
new_print = ('f"code   · {n_re} regex substitutions, {len(lit)} literal subs"\n'
             '          f" ({len(LITERAL_SUBS) - len(lit)} superseded upstream)"')
assert old_print in s, "report print not found"
s = s.replace(old_print, new_print)

ast.parse(s)
P.write_text(s, encoding="utf-8")
print("literal stage is defensive; report now shows superseded count")
