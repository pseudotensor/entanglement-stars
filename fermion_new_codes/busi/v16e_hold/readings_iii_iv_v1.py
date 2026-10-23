#!/usr/bin/env python
"""readings_iii_iv_v1.py (busi; ReviewA_v16e_v1.md erratum v3 sec. 2 (iii), (iv)). No engine run.
(iii) byte for byte but walls: json records leaf by leaf (walls skipped: seconds, timing/*_s, keys ending _s, start; path strings reported
     apart after the run dirs are masked) and every npz row (fields, spectra, handoff) by tobytes() at the common records, for the pairs given;
(iii') rotated_blocks (and partly_occupied_blocks, tol_block, gap_between_blocks, mode, nsub, n_intermediate) per channel per record in every arm;
(iv)  the comparator cross-check: for each arm pair and common record, every differing non-float leaf as tolcmp_v2 (d9b9ee32) classifies it
     (its own cmp_record: decision / counter / Newton trace / resume label / indicator bit) and as cmp_rd2_ruling_v2 (700ecf68, over
     cmp_rd2_replay_v1 17cd4868 and cmp_rd2_ruling_v1 ec05a1f0: identical class / resume label / other / indicator), and the keys where one
     comparator gates and the other does not.
Usage: readings_iii_iv_v1.py CMPCODE_DIR OUT_DIR"""
import importlib.util
import json
import os
import re
import sys

import numpy as np

CC, O = sys.argv[1], sys.argv[2]
os.makedirs(O, exist_ok=True)
sys.path.insert(0, CC)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


TV2 = load("tolcmp_v2", CC + "/tolcmp_v2.py")
R2 = load("cmp_rd2_ruling_v2", CC + "/cmp_rd2_ruling_v2.py")
B = "<internal>/perf_redesign"; V = B + "/v16e_hold"
TAGP = "lad_ISO_128_s4_nkqh2_m0p2_rs128_n4096_R04_k10_tau8_ell4_prep4_A_R2_avg_newton_v36nq"
ARMS = {"P": (V + "/card_merge/run_h2", "collapse_" + TAGP), "D": (B + "/RD1_build/entry_iso128/D", "collapse_rd13_D"),
        "B": (B + "/RD1_build/entry_iso128/B", "collapse_rd13_B")}
for x in ("D0", "A", "A2", "AB", "Fd", "Fd5"):
    ARMS[x] = (V + "/card_merge/iso/" + x, "collapse_mf_" + x)
for x in sys.argv[3:]:                       # extra arms NAME=DIR:STEM (the (v) arms)
    n, p = x.split("="); d, s = p.split(":"); ARMS[n] = (d, s)
TS = [5.0625 + 0.0625 * k for k in range(8)]
J = {}; Z = {}
for n, (d, s) in ARMS.items():
    if not os.path.exists(os.path.join(d, s + ".json")):
        continue
    js = json.load(open(os.path.join(d, s + ".json")))
    J[n] = {round(r["T_over_rs"], 6): r for r in js["records"] if r.get("T_over_rs") is not None and r["T_over_rs"] > 5.0 + 1e-9}
    Z[n] = {}
    for k in ("fields", "spectra", "handoff"):
        p = os.path.join(d, "%s_%s.npz" % (s, k))
        if os.path.exists(p):
            Z[n][k] = np.load(p, allow_pickle=True)
WALL = re.compile(r"(^|/)(seconds|start|slice_saved)$|_s$")
PATHY = re.compile(r"(^|/)(engine_v34|label|record_label|engine)$")


def leaves(x, p=""):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from leaves(v, p + "/" + str(k))
    elif isinstance(x, list):
        yield p + "#len", len(x)
        for i, v in enumerate(x):
            yield from leaves(v, p + "[%d]" % i)
    else:
        yield p, x


def mask_paths(s):
    return re.sub(r"/(workspace|data|opt)/[^ ;,)]*", "<path>", s) if isinstance(s, str) else s


def rows_T(Zk):
    T = np.asarray(Zk["T"], float) if "T" in Zk.files else None
    return T


def bytes_pair(a, b, t, rs=128.0):
    out = {"json_diff": [], "json_path_only": [], "npz_rows_compared": 0, "npz_rows_differ": []}
    La, Lb = dict(leaves(J[a][t])), dict(leaves(J[b][t]))
    for k in sorted(set(La) | set(Lb)):
        if WALL.search(k):
            continue
        x, y = La.get(k, "<absent>"), Lb.get(k, "<absent>")
        if x == y or (isinstance(x, float) and isinstance(y, float) and np.isnan(x) and np.isnan(y)):
            continue
        if PATHY.search(k) and mask_paths(x) == mask_paths(y):
            out["json_path_only"].append(k); continue
        out["json_diff"].append((k, str(x)[:80], str(y)[:80]))
    for kind in ("fields", "spectra", "handoff"):
        if kind not in Z[a] or kind not in Z[b]:
            continue
        A_, B_ = Z[a][kind], Z[b][kind]
        Ta, Tb = rows_T(A_), rows_T(B_)
        if Ta is None or Tb is None:
            continue
        ia = int(np.argmin(np.abs(Ta - t * rs))); ib = int(np.argmin(np.abs(Tb - t * rs)))
        if abs(Ta[ia] - t * rs) > 1e-6 or abs(Tb[ib] - t * rs) > 1e-6:
            continue
        for nm in sorted(set(A_.files) | set(B_.files)):
            if nm == "label" or nm not in A_.files or nm not in B_.files:
                if nm != "label":
                    out["npz_rows_differ"].append((kind, nm, "absent on one side")); continue
                continue
            xa, xb = A_[nm], B_[nm]
            if xa.ndim == 0 or xa.shape[0] != len(Ta):
                continue
            ra, rb = xa[ia], xb[ib]
            out["npz_rows_compared"] += 1
            if xa.dtype == object:
                same = np.asarray(ra).tobytes() == np.asarray(rb).tobytes() if not isinstance(ra, np.ndarray) or ra.dtype != object else all(np.asarray(u).tobytes() == np.asarray(v).tobytes() for u, v in zip(ra, rb))
            else:
                same = np.asarray(ra).tobytes() == np.asarray(rb).tobytes()
            if not same:
                out["npz_rows_differ"].append((kind, nm, float(np.max(np.abs(np.asarray(ra, float) - np.asarray(rb, float)))) if xa.dtype != object else "object"))
    return out


res = {"iii_bytes": {}, "iii_rotated": {}, "iv_cross": {}}
print("== (iii) byte for byte but walls")
for a, b in [("D0", "D"), ("Fd5", "Fd"), ("A", "AB")] + [(x, y) for x, y in (("dON", "Fd"), ("bON", "AB"), ("dOFF", "D"), ("dON_pw2", "dON"), ("bON_pw2", "bON")) if x in J and y in J]:
    if a not in J or b not in J:
        continue
    for t in TS:
        if t not in J[a] or t not in J[b]:
            continue
        o = bytes_pair(a, b, t)
        res["iii_bytes"]["%s-%s@%.4f" % (a, b, t)] = o
        print("  %-4s vs %-4s T %.4f: json leaves differing (walls skipped, paths masked) %d %s; path-only %d; npz rows compared %d, differing %d %s" % (
            a, b, t, len(o["json_diff"]), o["json_diff"][:4], len(o["json_path_only"]), o["npz_rows_compared"], len(o["npz_rows_differ"]), o["npz_rows_differ"][:4]))
print("== (iii') rotated_blocks per channel (only channels where any arm is nonzero or arms differ), with partly_occupied_blocks, tol_block")
KEYS = ("rotated_blocks", "partly_occupied_blocks", "tol_block", "n_intermediate", "mode", "nsub", "gap_between_blocks")
for t in TS:
    chs = set()
    for n in J:
        if t in J[n]:
            chs |= set((J[n][t].get("continuation") or {}).keys())
    for ch in sorted(chs, key=lambda c: int(c)):
        vals = {n: {k: ((J[n][t].get("continuation") or {}).get(ch) or {}).get(k) for k in KEYS} for n in J if t in J[n]}
        rb = {n: v["rotated_blocks"] for n, v in vals.items()}
        if any(rb.values()) or len(set(rb.values())) > 1:
            res["iii_rotated"]["%.4f/%s" % (t, ch)] = vals
            print("  T %.4f ch %s rotated_blocks %s | partly_occupied %s | tol_block %s | n_intermediate %s | gap_between_blocks %s" % (
                t, ch, rb, {n: v["partly_occupied_blocks"] for n, v in vals.items()}, {n: v["tol_block"] for n, v in vals.items()},
                {n: v["n_intermediate"] for n, v in vals.items()}, {n: ("%.3e" % v["gap_between_blocks"]) if isinstance(v["gap_between_blocks"], float) else v["gap_between_blocks"] for n, v in vals.items()}))
print("== (iv) comparator cross-check")
PAIRS = [("A", "D"), ("AB", "D"), ("Fd", "D"), ("A", "P"), ("AB", "P"), ("Fd", "P"), ("D", "P"), ("B", "D"), ("B", "P"), ("D0", "D"), ("D0", "P"), ("A2", "A")]
PAIRS += [(x, y) for x in ("dON", "dOFF", "bON", "bOFF") for y in ("D", "P") if x in J]
for a, b in PAIRS:
    if a not in J or b not in J:
        continue
    tot = {"tol_decision_not_rd2_ident": {}, "rd2_ident_not_tol_decision": {}, "both": {}}
    for t in TS:
        if t not in J[a] or t not in J[b]:
            continue
        acc = TV2.Acc()
        TV2.cmp_record(J[a][t], J[b][t], acc, t)
        tdec = {p[len("%.5f" % t):]: v for p, kind, v in acc.decisions}
        tcnt = {p[len("%.5f" % t):]: v for p, kind, v in acc.counters}
        ttr = {p[len("%.5f" % t):] for p, _, _ in acc.ntraces}
        tres = {r[0][len("%.5f" % t):] for r in acc.resume}
        fl, disc = R2.cmp_record(J[a][t], J[b][t])
        rid = {}
        for k, x, y in disc:
            if R2.RESUME_RE.match(k):
                if not (x in (None, "<absent>") or y in (None, "<absent>")):
                    rid[k] = (x, y)
            elif R2.identical_key_v2(k):
                rid[k] = (x, y)
        for k in set(tdec) | set(rid):
            if k in tdec and k in rid:
                tot["both"].setdefault(k, []).append(t)
            elif k in tdec:
                why = "rd2: not identical class" + ("" if any(k == q for q, _, _ in disc) else " (not compared: skipped key or float)")
                tot["tol_decision_not_rd2_ident"].setdefault(k, []).append((t, str(tdec[k])[:60], why))
            else:
                why = "tolcmp_v2: " + ("counter" if k in tcnt else "Newton trace" if k in ttr else "resume label" if any(k.startswith(q) for q in tres) else "not a decision (skipped key)")
                tot["rd2_ident_not_tol_decision"].setdefault(k, []).append((t, str(rid[k])[:60], why))
    res["iv_cross"]["%s-%s" % (a, b)] = tot
    print("  %-4s vs %-4s: gated by both %d keys; tolcmp_v2 decision but not cmp_rd2 identical %d keys; cmp_rd2 identical but not tolcmp_v2 decision %d keys" % (
        a, b, len(tot["both"]), len(tot["tol_decision_not_rd2_ident"]), len(tot["rd2_ident_not_tol_decision"])))
    for k, L in sorted(tot["both"].items()):
        print("      both: %s at %s" % (k, ["%.4f" % q for q in L]))
    for k, L in sorted(tot["tol_decision_not_rd2_ident"].items()):
        print("      tolcmp_v2 only: %s %s" % (k, L[:3]))
    for k, L in sorted(tot["rd2_ident_not_tol_decision"].items()):
        print("      cmp_rd2_v2 only: %s %s" % (k, L[:3]))
json.dump(res, open(O + "/readings_iii_iv.json", "w"), indent=1, default=str)
