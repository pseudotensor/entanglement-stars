#!/usr/bin/env python
"""cmp_v6r_gate.py (busi lane PERF PG-A1..A3): the bitwise comparison of two Page run directories written with the same --tag (the v6r gate's
arms). Every file of A must exist in B. json: every number compared by its exact value (floats by their IEEE bits, signed zero and NaN included),
every string with each arm's run directory replaced by a placeholder; npz: dtype, shape and payload bytes per array; pt (state, columns):
torch.load on the CPU, every tensor by dtype, shape and contiguous payload bytes, every other entry as json's. md: the lines compared. Exempt
(reported, never gated): keys naming wall time or device memory (seconds, wall, elapsed, time_s, device_peak..., started/finished stamps),
the summary's args keys that only one arm has (v6r's three options), the code-name string fields when --code-names-differ is given (v6o against
v6q/v6r). Exit 0 = bitwise on every non-exempt entry.
Usage: python cmp_v6r_gate.py RUN_A RUN_B [--code-names-differ]"""
import json
import math
import os
import re
import struct
import sys

import numpy as np

EXEMPT_KEY = re.compile(r"(seconds|wall|elapsed|time_s|_s$|device_peak|started|finished|date|utc|host|pid|t_wall|runtime|duration)", re.I)
NEW_ARGS = {"prop_chunk", "v6r_share_traces", "v6r_full_cache"}
CODE_KEY = re.compile(r"(^code$|engine|^label_code)", re.I)


def fbits(x):
    return struct.pack("<d", float(x))


class Cmp:
    def __init__(self, ra, rb, code_names):
        self.ra = os.path.abspath(ra); self.rb = os.path.abspath(rb); self.code_names = code_names
        self.diff = []; self.exempt = []; self.n = 0

    def norm(self, s, root):
        return s.replace(root, "<RUN>").replace(os.path.relpath(root, os.getcwd()), "<RUN>").replace(os.path.basename(root), "<ARM>")

    def walk(self, a, b, path):
        key = path.rsplit("/", 1)[-1]
        if EXEMPT_KEY.search(key):
            if a != b:
                self.exempt.append(path)
            return
        if isinstance(a, dict) and isinstance(b, dict):
            for k in sorted(set(a) | set(b), key=str):
                p = "%s/%s" % (path, k)
                if k not in a or k not in b:
                    if k in NEW_ARGS:
                        self.exempt.append(p + " (one arm only)")
                    else:
                        self.diff.append(p + " (one arm only)")
                    continue
                self.walk(a[k], b[k], p)
            return
        if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
            if len(a) != len(b):
                self.diff.append("%s (length %d against %d)" % (path, len(a), len(b))); return
            for i, (x, y) in enumerate(zip(a, b)):
                self.walk(x, y, "%s[%d]" % (path, i))
            return
        self.n += 1
        if isinstance(a, bool) or isinstance(b, bool) or a is None or b is None:
            if a != b or type(a) != type(b):
                self.diff.append("%s: %r against %r" % (path, a, b))
            return
        if isinstance(a, (int, float, np.floating, np.integer)) and isinstance(b, (int, float, np.floating, np.integer)):
            if isinstance(a, (int, np.integer)) and isinstance(b, (int, np.integer)):
                if int(a) != int(b):
                    self.diff.append("%s: %r against %r" % (path, a, b))
            elif fbits(a) != fbits(b):
                self.diff.append("%s: %r against %r" % (path, a, b))
            return
        if isinstance(a, str) and isinstance(b, str):
            sa, sb = self.norm(a, self.ra), self.norm(b, self.rb)
            if sa != sb:
                if self.code_names and (CODE_KEY.search(key) or "page_history_v6" in sa):
                    if re.sub(r"page_history_v6[a-z]", "page_history_v6X", sa) == re.sub(r"page_history_v6[a-z]", "page_history_v6X", sb):
                        self.exempt.append(path + " (code name)"); return
                self.diff.append("%s: %r against %r" % (path, sa[:160], sb[:160]))
            return
        try:
            import torch
            if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
                if a.dtype != b.dtype or tuple(a.shape) != tuple(b.shape):
                    self.diff.append("%s: tensor %s %s against %s %s" % (path, a.dtype, tuple(a.shape), b.dtype, tuple(b.shape))); return
                ab = a.detach().cpu().contiguous().numpy().tobytes(); bb = b.detach().cpu().contiguous().numpy().tobytes()
                if ab != bb:
                    d = (a.detach().cpu() - b.detach().cpu()).abs().max().item() if a.numel() else 0.0
                    self.diff.append("%s: tensor payload differs (max |a - b| %.3e)" % (path, d))
                return
        except ImportError:
            pass
        if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
            self.arr(a, b, path); return
        if type(a) != type(b) or a != b:
            self.diff.append("%s: %r against %r (types %s, %s)" % (path, str(a)[:80], str(b)[:80], type(a).__name__, type(b).__name__))

    def arr(self, a, b, path):
        self.n += 1
        if a.dtype != b.dtype or a.shape != b.shape:
            self.diff.append("%s: array %s %s against %s %s" % (path, a.dtype, a.shape, b.dtype, b.shape)); return
        if a.dtype == object:
            self.walk(a.tolist(), b.tolist(), path); return
        if np.ascontiguousarray(a).tobytes() != np.ascontiguousarray(b).tobytes():
            try:
                d = float(np.nanmax(np.abs(a.astype(complex) - b.astype(complex)))) if a.size else 0.0
            except Exception:
                d = float("nan")
            self.diff.append("%s: array payload differs (max |a - b| %.3e)" % (path, d))

    def file(self, name):
        fa = os.path.join(self.ra, name); fb = os.path.join(self.rb, name)
        if not os.path.exists(fb):
            self.diff.append("%s: missing in B" % name); return
        if name.endswith(".json"):
            self.walk(json.load(open(fa)), json.load(open(fb)), name)
        elif name.endswith(".npz"):
            za = np.load(fa, allow_pickle=True); zb = np.load(fb, allow_pickle=True)
            for k in sorted(set(za.files) | set(zb.files)):
                if k not in za.files or k not in zb.files:
                    self.diff.append("%s/%s (one arm only)" % (name, k)); continue
                if EXEMPT_KEY.search(k):
                    continue
                self.arr(za[k], zb[k], "%s/%s" % (name, k))
        elif name.endswith(".pt"):
            import torch
            self.walk(torch.load(fa, map_location="cpu", weights_only=False), torch.load(fb, map_location="cpu", weights_only=False), name)
        elif name.endswith(".md") or name.endswith(".txt"):
            la = [self.norm(x, self.ra) for x in open(fa).read().splitlines()]
            lb = [self.norm(x, self.rb) for x in open(fb).read().splitlines()]
            if self.code_names:
                la = [re.sub(r"page_history_v6[a-z]", "page_history_v6X", x) for x in la]; lb = [re.sub(r"page_history_v6[a-z]", "page_history_v6X", x) for x in lb]
            bad = [i for i, (x, y) in enumerate(zip(la, lb)) if x != y]
            if len(la) != len(lb) or bad:
                for i in bad[:5]:
                    tgt = self.exempt if re.search(r"seconds|wall| s\b|GiB", la[i] + lb[i]) else self.diff
                    tgt.append("%s line %d: %r against %r" % (name, i + 1, la[i][:160], lb[i][:160]))
                if len(la) != len(lb):
                    self.diff.append("%s: %d lines against %d" % (name, len(la), len(lb)))
        else:
            if open(fa, "rb").read() != open(fb, "rb").read():
                self.diff.append("%s: bytes differ" % name)


def main():
    args = [x for x in sys.argv[1:] if not x.startswith("--")]
    c = Cmp(args[0], args[1], "--code-names-differ" in sys.argv)
    names = sorted(os.listdir(c.ra)); nb = sorted(os.listdir(c.rb))
    extra = sorted(set(nb) - set(names))
    for nm in names:
        c.file(nm)
    for nm in extra:
        c.diff.append("%s: in B only" % nm)
    print("cmp_v6r_gate: A %s, B %s: %d files, %d leaf entries compared" % (c.ra, c.rb, len(names), c.n))
    print("exempt differences (%d, not gated):" % len(c.exempt))
    for x in c.exempt[:40]:
        print("   " + x)
    print("NON-EXEMPT differences (%d):" % len(c.diff))
    for x in c.diff[:60]:
        print("   " + x)
    print("VERDICT: %s" % ("BITWISE" if not c.diff else "DIFFERS"))
    sys.exit(0 if not c.diff else 1)


if __name__ == "__main__":
    main()
