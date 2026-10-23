#!/usr/bin/env python
"""w10_apply_fixes_v1.py (busi W10): applies the OLD/NEW blocks of W10_fix_blocks_v1.md to the target files and writes the versioned
outputs (engine_collapse_v36d.py -> engine_collapse_v36e.py in the engine directory; page_ops_v6.py -> page_ops_v7.py in the page
directory), copying w10_lib_v1.py beside the engine output. Every OLD must be found exactly once: exactly, or, failing that, with runs
of 2 or more inner spaces matched flexibly (the alignment of trailing comments), reported per block; any other count stops the script.
Blocks with "mode: manual" are listed, not applied (the W1 builder's page_core import line). --check counts, applies in memory and
compiles the result without writing.
Usage: qi2 python w10_apply_fixes_v1.py [--check] [--engine-dir DIR] [--page-dir DIR]   (the defaults: $W10_ENGINE_DIR, $W10_PAGE_DIR)"""
import argparse
import ast
import os
import re
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def parse_blocks(text):
    lines = text.split("\n")
    blocks = []
    i = 0
    while i < len(lines):
        m = re.match(r"^=== BLOCK (\S+) ===\s*$", lines[i])
        if not m:
            i += 1
            continue
        bid = m.group(1); i += 1
        meta = {}
        while i < len(lines) and lines[i] != "--- OLD ---":
            mm = re.match(r"^(\w+):\s*(.*)$", lines[i])
            if mm:
                meta[mm.group(1)] = mm.group(2).strip()
            i += 1
        if i >= len(lines):
            raise SystemExit("block %s: no '--- OLD ---' marker" % bid)
        i += 1
        old = []
        while i < len(lines) and lines[i] != "--- NEW ---":
            old.append(lines[i]); i += 1
        if i >= len(lines):
            raise SystemExit("block %s: no '--- NEW ---' marker" % bid)
        i += 1
        new = []
        while i < len(lines) and lines[i] != "=== END BLOCK ===":
            new.append(lines[i]); i += 1
        if i >= len(lines):
            raise SystemExit("block %s: no '=== END BLOCK ===' marker" % bid)
        i += 1
        if "target" not in meta or "output" not in meta:
            raise SystemExit("block %s: target or output missing" % bid)
        blocks.append({"id": bid, "meta": meta, "old": "\n".join(old), "new": "\n".join(new)})
    return blocks


def flexible_regex(old):
    """the OLD text as a regex: exact, except that a run of 2 or more spaces not at the start of a line matches any run of spaces."""
    parts = []
    for ln in old.split("\n"):
        m = re.match(r"^( *)(.*)$", ln)
        indent, rest = m.group(1), m.group(2)
        segs = re.split(r"( {2,})", rest)
        rx = "".join(" +" if (s != "" and s.strip() == "" and len(s) >= 2) else re.escape(s) for s in segs)
        parts.append(re.escape(indent) + rx)
    return "\n".join(parts)


def apply_block(text, b):
    n = text.count(b["old"])
    if n == 1:
        return text.replace(b["old"], b["new"]), "exact"
    hits = list(re.finditer(flexible_regex(b["old"]), text))
    if len(hits) == 1:
        h = hits[0]
        return text[:h.start()] + b["new"] + text[h.end():], "flexible inner spaces"
    raise SystemExit("block %s: OLD found %d times exactly and %d times with flexible spaces (exactly 1 needed)" % (b["id"], n, len(hits)))


def find_dir(name, dirs):
    for d in dirs:
        if d and os.path.isfile(os.path.join(d, name)):
            return d
    raise SystemExit("target %s not found in %s (set --engine-dir / --page-dir or W10_ENGINE_DIR / W10_PAGE_DIR)" % (name, [d for d in dirs if d]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", default=os.path.join(HERE, "W10_fix_blocks_v1.md"))
    ap.add_argument("--engine-dir", default=os.environ.get("W10_ENGINE_DIR"))
    ap.add_argument("--page-dir", default=os.environ.get("W10_PAGE_DIR"))
    ap.add_argument("--check", action="store_true", help="count, apply in memory and compile; write nothing")
    a = ap.parse_args()
    blocks = parse_blocks(open(a.blocks).read())
    print("W10 apply: %d blocks parsed from %s" % (len(blocks), a.blocks))
    groups = {}
    for b in blocks:
        key = (b["meta"]["target"], b["meta"]["output"], b["meta"].get("mode", "auto"))
        groups.setdefault(key, []).append(b)
    for (target, output, mode), bl in groups.items():
        if mode == "manual":
            for b in bl:
                print("block %s: MANUAL (%s -> %s): the builder applies it in that version; OLD: %r -> NEW: %r" % (b["id"], target, output, b["old"], b["new"]))
            continue
        d = find_dir(target, [a.engine_dir, a.page_dir, HERE])
        src = os.path.join(d, target)
        text = open(src).read()
        for b in bl:
            text, how = apply_block(text, b)
            print("block %s: OLD found once in %s (%s)" % (b["id"], target, how))
        try:
            ast.parse(text)
        except SyntaxError as ex:
            raise SystemExit("the result for %s does not compile: %s" % (output, ex))
        print("%s -> %s: %d blocks, compiles" % (target, output, len(bl)))
        if not a.check:
            outp = os.path.join(d, output)
            with open(outp, "w") as fh:
                fh.write(text)
            print("wrote %s" % outp)
            if target.startswith("engine_collapse"):
                lib = os.path.join(d, "w10_lib_v1.py")
                shutil.copy(os.path.join(HERE, "w10_lib_v1.py"), lib)
                print("copied w10_lib_v1.py to %s" % lib)
    print("W10 apply: done (%s)" % ("check only" if a.check else "written"))


if __name__ == "__main__":
    main()
