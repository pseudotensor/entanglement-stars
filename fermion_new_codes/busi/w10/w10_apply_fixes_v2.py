#!/usr/bin/env python
"""w10_apply_fixes_v2.py (busi W10): applies the OLD/NEW blocks of W10_fix_blocks_v2.md to the target files and writes the versioned
outputs (engine_collapse_v36d.py -> engine_collapse_v36e.py in the engine directory; page_ops_v6.py -> page_ops_v7.py in the page
directory), copying w10_lib_v1.py beside the engine output. Every OLD must be found exactly once: exactly, or, failing that, with runs
of 2 or more inner spaces matched flexibly (the alignment of trailing comments), reported per block; any other count stops the script
and nothing is written. Blocks with "mode: manual" are listed, not applied (the W1 builder's page_core import line). --check counts,
applies in memory and compiles the result without writing.

Version 2 against w10_apply_fixes_v1.py. W10_reading_v1.md section 3 announced "v1 with the blocks file taken from --blocks (default
W10_fix_blocks_v2.md); nothing else changes". Two additions were made after that reading, for the reconciliation with W2's --step dg
build (engine_collapse_v36f.py over v36d; the head of RUN_W10_v2.md); both are inactive unless asked for, and without them the script is v1:
  --engine-target NAME --engine-output NAME   the engine blocks applied over another base than the blocks file's (engine_collapse_v36f.py
      -> engine_collapse_v36g.py). The code blocks must be found exactly once in that base like in any other (a 0 or 2 count stops the
      script: the dg edit touched a W10 site; a v3 of the blocks file with that base's OLD text follows). The docstring block (D3.1, whose
      OLD is the base's first docstring line) is, when not found, rebased: its W10 paragraph (NEW without its last line) is inserted after
      the module docstring's opening quotes with the version names substituted, and the base's own first line follows. In rebase mode the
      NEW texts also get the phrase substitutions of rebase_phrases (the output's name in the records' "w10" entry and in the print
      prefixes; "no number of v36d" -> the base's version); nothing else: a variable or a key named v36d is untouched.
  --force   an existing output file (or a differing w10_lib_v1.py beside it) is not overwritten unless --force: the script stops and names
      the file (the stale-v36e hazard of the reading's section 1: a file under the output's name with the base's content); --check reports
      it as a WARNING.
Usage: qi2 python w10_apply_fixes_v2.py [--check] [--engine-dir DIR] [--page-dir DIR] [--blocks FILE]
                                         [--engine-target engine_collapse_v36f.py --engine-output engine_collapse_v36g.py] [--force]
(the directory defaults: $W10_ENGINE_DIR, $W10_PAGE_DIR)"""
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


def stem_version(name):
    m = re.match(r"^engine_collapse_v(\w+)\.py$", name)
    return m.group(1) if m else None


def rebase_phrases(blocks_target, blocks_output, target, output):
    """the phrase substitutions of rebase mode: explicit and few, so that a variable or a key named after the base's version is never
    renamed (the engine's v36d variable and "v36d" record entry stay)."""
    bt, bo, tv, ov = stem_version(blocks_target), stem_version(blocks_output), stem_version(target), stem_version(output)
    if None in (bt, bo, tv, ov):
        raise SystemExit("rebase: every name must be engine_collapse_v<version>.py (the blocks' %s -> %s; asked %s -> %s)" % (blocks_target, blocks_output, target, output))
    return [(blocks_output[:-3], output[:-3]),                                                 # engine_collapse_v36e -> engine_collapse_v36g
            ("version %s = version %s" % (bo, bt), "version %s = version %s" % (ov, tv)),      # the header
            ("no number of version %s" % bt, "no number of version %s" % tv),                  # the header
            ("no number of v%s" % bt, "no number of v%s" % tv),                                # the records' "w10" entry
            ("Version %s's docstring follows." % bt, "Version %s's docstring follows." % tv),  # the header
            ("  v%s %%s" % bo, "  v%s %%s" % ov)]                                               # the print prefixes of the resolve rule


def subst(text, phrases):
    for o, n in phrases:
        text = text.replace(o, n)
    return text


def header_rebase(text, b):
    """a header block (OLD the base's first docstring line, NEW ending with it) applied to a base whose first line differs: NEW's
    paragraph, substituted, goes after the opening quotes and the base's own first line follows."""
    old_body = b["old"][3:]
    tail = "\n" + old_body
    if not (b["old"].startswith('"""') and b["new"].startswith('"""') and b["new"].endswith(tail)):
        raise SystemExit("block %s: OLD not found and the block is not a header block (OLD a docstring's first line, NEW ending with it)" % b["id"])
    if not text.startswith('"""'):
        raise SystemExit("block %s: the base does not open with a module docstring; the header cannot be rebased" % b["id"])
    para = subst(b["new"][: -len(tail)], b.get("phrases", []))
    return para + "\n" + text[3:]


def apply_block(text, b, header_fallback=False):
    new = subst(b["new"], b.get("phrases", []))
    n = text.count(b["old"])
    if n == 1:
        return text.replace(b["old"], new), "exact"
    hits = list(re.finditer(flexible_regex(b["old"]), text))
    if len(hits) == 1:
        h = hits[0]
        return text[:h.start()] + new + text[h.end():], "flexible inner spaces"
    if n == 0 and not hits and header_fallback and b["old"].startswith('"""'):
        return header_rebase(text, b), "header"
    raise SystemExit("block %s: OLD found %d times exactly and %d times with flexible spaces (exactly 1 needed)" % (b["id"], n, len(hits)))


def find_dir(name, dirs):
    for d in dirs:
        if d and os.path.isfile(os.path.join(d, name)):
            return d
    raise SystemExit("target %s not found in %s (set --engine-dir / --page-dir or W10_ENGINE_DIR / W10_PAGE_DIR; a rebase base must be copied into the engine directory first)" % (name, [d for d in dirs if d]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", default=os.path.join(HERE, "W10_fix_blocks_v2.md"))
    ap.add_argument("--engine-dir", default=os.environ.get("W10_ENGINE_DIR"))
    ap.add_argument("--page-dir", default=os.environ.get("W10_PAGE_DIR"))
    ap.add_argument("--engine-target", default=None, help="rebase: apply the engine blocks over this base instead of the blocks file's target (e.g. engine_collapse_v36f.py)")
    ap.add_argument("--engine-output", default=None, help="rebase: the output name (e.g. engine_collapse_v36g.py)")
    ap.add_argument("--check", action="store_true", help="count, apply in memory and compile; write nothing")
    ap.add_argument("--force", action="store_true", help="overwrite an existing output (and a differing w10_lib_v1.py beside it)")
    a = ap.parse_args()
    blocks = parse_blocks(open(a.blocks).read())
    print("W10 apply: %d blocks parsed from %s" % (len(blocks), a.blocks))
    rebase = a.engine_target is not None or a.engine_output is not None
    if rebase:
        if not (a.engine_target and a.engine_output):
            raise SystemExit("--engine-target and --engine-output go together")
        eng = [b for b in blocks if b["meta"]["target"].startswith("engine_collapse") and b["meta"].get("mode", "auto") != "manual"]
        if not eng:
            raise SystemExit("rebase: no engine block in %s" % a.blocks)
        phrases = rebase_phrases(eng[0]["meta"]["target"], eng[0]["meta"]["output"], a.engine_target, a.engine_output)
        for b in eng:
            b["meta"] = dict(b["meta"], target=a.engine_target, output=a.engine_output)
            b["phrases"] = phrases
        print("W10 apply: rebase mode: %d engine blocks over %s -> %s; the phrase substitutions in their NEW texts: %s" % (
            len(eng), a.engine_target, a.engine_output, "; ".join("%r -> %r" % p for p in phrases)))
    groups = {}
    for b in blocks:
        key = (b["meta"]["target"], b["meta"]["output"], b["meta"].get("mode", "auto"))
        groups.setdefault(key, []).append(b)
    results = []
    for (target, output, mode), bl in groups.items():
        if mode == "manual":
            for b in bl:
                print("block %s: MANUAL (%s -> %s): the builder applies it in that version; OLD: %r -> NEW: %r" % (b["id"], target, output, b["old"], b["new"]))
            continue
        d = find_dir(target, [a.engine_dir, a.page_dir, HERE])
        text = open(os.path.join(d, target)).read()
        for b in bl:
            text, how = apply_block(text, b, header_fallback=rebase and target.startswith("engine_collapse"))
            if how == "header":
                print("block %s: OLD not found in %s (another base's first docstring line): header rebased, the W10 paragraph inserted after the opening quotes and the base's own first line kept" % (b["id"], target))
            else:
                print("block %s: OLD found once in %s (%s)" % (b["id"], target, how))
        try:
            ast.parse(text)
        except SyntaxError as ex:
            raise SystemExit("the result for %s does not compile: %s" % (output, ex))
        print("%s -> %s: %d blocks, compiles" % (target, output, len(bl)))
        results.append((d, target, output, text))
    here_lib = os.path.join(HERE, "w10_lib_v1.py")
    stops = []
    for d, target, output, text in results:
        outp = os.path.join(d, output)
        if os.path.exists(outp):
            stale = "import w10_lib_v1 as W10" not in open(outp).read()
            msg = "%s exists already (%s); --force overwrites" % (outp, "a stale copy without the W10 import: rm it, RUN_W10_v2 step 0" if stale else "an earlier apply")
            if a.check:
                print("WARNING " + msg)
            else:
                stops.append(msg)
        if target.startswith("engine_collapse"):
            lib = os.path.join(d, "w10_lib_v1.py")
            if os.path.exists(lib) and open(lib).read() != open(here_lib).read():
                msg = "%s exists and differs from %s; --force overwrites" % (lib, here_lib)
                if a.check:
                    print("WARNING " + msg)
                else:
                    stops.append(msg)
    if a.check:
        print("W10 apply: done (check only)")
        return
    if stops and not a.force:
        raise SystemExit("stop (nothing written): " + " | ".join(stops))
    for d, target, output, text in results:
        outp = os.path.join(d, output)
        with open(outp, "w") as fh:
            fh.write(text)
        print("wrote %s" % outp)
        if target.startswith("engine_collapse"):
            lib = os.path.join(d, "w10_lib_v1.py")
            if os.path.exists(lib) and open(lib).read() == open(here_lib).read():
                print("w10_lib_v1.py already beside %s, identical" % output)
            else:
                shutil.copy(here_lib, lib)
                print("copied w10_lib_v1.py to %s" % lib)
    print("W10 apply: done (written)")


if __name__ == "__main__":
    main()
