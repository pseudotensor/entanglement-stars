#!/usr/bin/env python
"""ladders_catalog_v1.py (composition builder, the solve lane; MASS_ladder_v1 and SCALING_ladder_v1): each ladder member's own dust
catalog at T = 0 (the radial modes per channel), from which its own onset radius 4 a n_rad,max is read.

Two sources, each labelled in the output:
  (1) --status NAME=PATH: the launcher's status file of the member (the t0 line "dust per channel at T = 0" and the "preparation:" line),
      as written on the card and mirrored to the box; the last occurrence of each line is read.
  (2) --prepare NAME=JSON: the engine's own preparation, engine_collapse_v36d.prepare_dust on the CPU backend with the arguments stored in
      the member's run json (m, g, r_s, R_0/r_s, reading, sites, kappa_sea, prep_ell, half_offset, shell, site_fields, collar_policy,
      support_rel; the engine's defaults passes 200, tol 1e-10, the same call as the engine's main). No record and no propagation. The
      preparation's N_B, E0 and f_kin are compared with the run json's head (the stored preparation); the differences are printed and a
      difference above 1e-8 relative is flagged.
Output: one json (--out) with, per member: the columns per channel, the top channel, the largest column count n_rad,max, 4 a n_rad,max,
the particles per channel, the lowest occupation per channel, the source and (for --prepare) the head comparison and the seconds.
Usage (box CPU, conda env qi2; nice -n 10; OMP_NUM_THREADS etc. set by the caller):
  python ladders_catalog_v1.py --code-dir DIR [--status NAME=PATH ...] [--prepare NAME=JSON ...] --out OUT.json [--threads 8]"""
import argparse
import json
import math
import os
import re
import sys
import time

T0_ROW = re.compile(r"kappa (\d+) \(w ([0-9.eE+-]+)\): (\d+) columns, ([0-9.]+) particles, lowest occupation ([0-9.]+|-)")
PREP_LINE = re.compile(r"preparation: N_B (\S+) against r_s/\(2 g m\) = (\S+) \(difference (\S+)\); E0 (\S+); E0/\(N_B m\) (\S+); E0/m (\S+); "
                       r"passes (\S+); last change (\S+); f_kin (\S+);")


def summarize(rows):
    cols = {k: r["columns"] for k, r in rows.items()}
    held = [k for k, c in cols.items() if c > 0]
    nmax = max(cols.values()) if cols else 0
    return {"columns": {str(k): cols[k] for k in sorted(cols)}, "top": (max(held) if held else None), "n_rad_max": nmax,
            "onset_4a_nradmax": 4.0 * nmax,
            "particles": {str(k): rows[k]["particles"] for k in sorted(rows)},
            "lowest_occupation": {str(k): rows[k]["lowest_occupation"] for k in sorted(rows)},
            "weighted_total": float(sum(r["particles"] for r in rows.values()))}


def from_status(path):
    rows, prep = None, None
    for ln in open(path, errors="replace"):
        if "dust per channel at T = 0" in ln:
            rows = {}
            for mm in T0_ROW.finditer(ln):
                rows[int(mm.group(1))] = {"w": float(mm.group(2)), "columns": int(mm.group(3)), "particles": float(mm.group(4)),
                                          "lowest_occupation": (float(mm.group(5)) if mm.group(5) != "-" else None)}
        mp = PREP_LINE.search(ln)
        if mp:
            prep = {"N_B": float(mp.group(1)), "E0": float(mp.group(4)), "passes": mp.group(7), "f_kin": float(mp.group(9).rstrip(";"))}
    if not rows:
        return None
    out = summarize(rows)
    out.update({"source": "status file t0 line", "path": path, "preparation_line": prep})
    return out


def from_prepare(json_path, code_dir, threads):
    import numpy as np
    sys.path.insert(0, code_dir)
    import torch
    torch.set_num_threads(int(threads))
    import engine_collapse_v5 as C5
    import engine_collapse_v36d as E36
    C5.backend(None)
    J = json.load(open(json_path))
    A, H = J["args"], J.get("head") or {}
    E36.OPT["collar_policy"] = A.get("collar_policy", "flag")
    E36.OPT["support_rel"] = A.get("support_rel", 1e-14)
    E36.OPT["site_fields"] = A.get("site_fields", "dense")
    m, g, rs = float(A["m"]), float(A["g"]), float(A["rs"])
    prep = A.get("prep_ell") if A.get("prep_ell") is not None else A["ell"]
    shell = [float(x) * rs for x in A["shell"].split(",")] if A.get("shell") else None
    chs = [C5.Chan(k, m, int(A["nsites"]), bool(A.get("half_offset"))) for k in range(1, int(A["kappa_sea"]) + 1)]
    R = chs[0].R
    nb = rs / (2.0 * g * m)
    t0 = time.time()
    N, v, dust, info, head = E36.prepare_dust(chs, g, R, int(prep), nb, float(A["R0_over_rs"]) * rs, m, A.get("reading", "A"), shell)
    sec = time.time() - t0
    rows = {}
    for ch in chs:
        if ch.kappa in dust:
            Phi, fk = dust[ch.kappa]
            rows[ch.kappa] = {"w": float(ch.w), "columns": int(Phi.shape[1]), "particles": float(ch.w * float(np.sum(fk))),
                              "lowest_occupation": float(np.min(fk)) if len(fk) else None}
        else:
            rows[ch.kappa] = {"w": float(ch.w), "columns": 0, "particles": 0.0, "lowest_occupation": None}
    out = summarize(rows)
    cmp_ = {}
    for k in ("N_B", "E0", "f_kin"):
        a_, b_ = head.get(k), H.get(k)
        if a_ is not None and b_ is not None:
            d = abs(float(a_) - float(b_)) / max(abs(float(b_)), 1e-300)
            cmp_[k] = {"prepared": float(a_), "stored_head": float(b_), "rel_diff": d, "flag": d > 1e-8}
        else:
            cmp_[k] = {"prepared": a_, "stored_head": b_, "rel_diff": None, "flag": True}
    out.update({"source": "engine_collapse_v36d.prepare_dust on the box CPU with the run json's arguments", "path": json_path,
                "prep_ell": prep, "passes": head.get("passes"), "last_change": head.get("last_change"), "seconds": sec,
                "head_comparison": cmp_, "script_of_run": J.get("script")})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code-dir", default="<internal>/engine")
    ap.add_argument("--status", action="append", default=[])
    ap.add_argument("--prepare", action="append", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    res = json.load(open(a.out)) if os.path.exists(a.out) else {}
    for item in a.status:
        name, path = item.split("=", 1)
        r = from_status(path)
        res[name] = r if r else {"source": "status file without a t0 line", "path": path}
        print(name, json.dumps({k: res[name].get(k) for k in ("columns", "top", "n_rad_max", "onset_4a_nradmax", "source")}), flush=True)
        json.dump(res, open(a.out, "w"), indent=1)
    for item in a.prepare:
        name, path = item.split("=", 1)
        r = from_prepare(path, a.code_dir, a.threads)
        res[name] = r
        print(name, json.dumps({k: r.get(k) for k in ("columns", "top", "n_rad_max", "onset_4a_nradmax", "head_comparison", "seconds")}), flush=True)
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
