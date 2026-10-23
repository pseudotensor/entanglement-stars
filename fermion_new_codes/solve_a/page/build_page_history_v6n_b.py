#!/usr/bin/env python
"""build_page_history_v6n_b.py: the smallest change so that page_history_v6n imports page_d13_readout_v1b (v1 with the cross-cut block
trace norms of EVOLVEDCUT EC1) instead of page_d13_readout_v1: one exact replacement of the import line, applied in place to the built
page_history_v6n.py and/or to build_page_history_v6n.py (whose replacement list writes that line), whichever exist in --dir
(--target auto|built|builder|both). The "old" must match exactly once (abort otherwise); idempotent (a file already importing v1b is
reported and left alone); the new text is syntax-checked before the file is replaced; the md5s are printed. Usage:
    python build_page_history_v6n_b.py [--dir DIR] [--target auto]
Order: build_page_d13_readout_v1b.py first (page_d13_readout_v1b.py beside page_d13_readout_v1.py and page_ops_v6.py); then this script; a
v6n built afterwards by the patched build_page_history_v6n.py carries the import already, a v6n built before is patched here. Nothing else
in v6n changes: D13.format_line, D13.ksum_table, D13.memory_figure and ChannelReadout.checkpoint carry the new readouts through the same calls."""
import argparse
import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

OLD = "import page_d13_readout_v1 as D13   # noqa: E402"
NEW = "import page_d13_readout_v1b as D13  # noqa: E402  (v6n_b: v1b adds the cross-cut block trace norms, EVOLVEDCUT EC1)"
APPLIED = "import page_d13_readout_v1b as D13"
FILES = {"built": "page_history_v6n.py", "builder": "build_page_history_v6n.py"}


def patch(path):
    text = open(path, encoding="utf-8").read()
    md5 = hashlib.md5(text.encode("utf-8")).hexdigest()
    if APPLIED in text and OLD not in text:
        print("%s: already imports page_d13_readout_v1b (md5 %s): left alone" % (path, md5))
        return
    n = text.count(OLD)
    if n != 1:
        sys.exit("ABORT: %s: the import line matches %d times (must be exactly 1): %r" % (path, n, OLD))
    new = text.replace(OLD, NEW)
    compile(new, path, "exec")                                                  # the syntax check before anything is written
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(new)
    os.replace(tmp, path)
    print("%s: the import patched (md5 %s -> %s); syntax checked" % (path, md5, hashlib.md5(new.encode("utf-8")).hexdigest()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=HERE, help="where page_history_v6n.py and/or build_page_history_v6n.py live")
    ap.add_argument("--target", default="auto", choices=["auto", "built", "builder", "both"],
                    help="auto: every target present (abort if none); built: page_history_v6n.py; builder: build_page_history_v6n.py; both: both required")
    a = ap.parse_args()
    paths = {k: os.path.join(a.dir, v) for k, v in FILES.items()}
    if a.target == "auto":
        todo = [p for p in paths.values() if os.path.isfile(p)]
        if not todo:
            sys.exit("ABORT: neither %s nor %s is in %s" % (FILES["built"], FILES["builder"], a.dir))
    elif a.target == "both":
        todo = [paths["built"], paths["builder"]]
    else:
        todo = [paths[a.target]]
    for p in todo:
        if not os.path.isfile(p):
            sys.exit("ABORT: %s is missing" % p)
        patch(p)
    v1b = [os.path.join(a.dir, "page_d13_readout_v1b.py")] + [os.path.join(d, "page_d13_readout_v1b.py") for d in os.environ.get("PAGE_CODE_V6", "").split(os.pathsep) if d]
    if not any(os.path.isfile(p) for p in v1b):
        print("WARNING: page_d13_readout_v1b.py is not in %s (nor $PAGE_CODE_V6): run build_page_d13_readout_v1b.py before launching v6n" % a.dir)
    print("done: %s" % ", ".join(todo))


if __name__ == "__main__":
    main()
