#!/usr/bin/env python
"""gen_hold_args_v1.py (the lane's analyst builder; engine spec 43.3 (2)): the held-field controls rerun as they were. Reads a stored hold's
summary settings (page_history_v6e's vars(a)), writes the page_history_v6k argument list that reproduces them (every stored setting, with
--out, --tag, --init-state and --device replaced by the rerun's, and --collar-split added), parses it with page_history_v6k's own parser and
asserts that every stored setting other than those four equals the parsed one; prints the arguments one per line (for a launcher's array).
Usage: gen_hold_args_v1.py SUMMARY_JSON OUT TAG INIT_STATE DEVICE"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def main():
    p, out, tag, init, dev = sys.argv[1:6]
    S = json.load(open(p))["settings"]
    over = {"out": out, "tag": tag, "init_state": init, "device": dev}
    argv = []
    for k, v in S.items():
        v = over.get(k, v)
        if v is None or v is False:
            continue
        opt = "--" + k.replace("_", "-")
        if v is True:
            argv.append(opt)
        else:
            argv += [opt, str(v)]
    argv.append("--collar-split")
    import page_history_v6k as PK
    a = vars(PK.parse(argv))
    bad = []
    for k, v in S.items():
        if k in over:
            continue
        w = a.get(k, "<absent>")
        if not (v == w or (isinstance(v, float) and isinstance(w, (int, float)) and float(v) == float(w))):
            bad.append("%s stored %r parsed %r" % (k, v, w))
    if bad:
        sys.exit("settings differ: " + "; ".join(bad))
    if not a.get("collar_split"):
        sys.exit("--collar-split not set")
    print("\n".join(argv))


if __name__ == "__main__":
    main()
