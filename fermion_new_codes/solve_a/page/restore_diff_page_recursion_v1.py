#!/usr/bin/env python
"""restore_diff_page_recursion_v1.py (builder analyst, 2026-10-16; engine spec 38.1 (3)): a record-only save-and-restore diff of the
recursion (page_recursion_v6b.py) on the self-test's small case T4 (r_s/a 4, 192 sites, kappa 1) in v5 mode, on the box CPU. It names the
arrays and fields that differ after a restore, for the two restore paths the regression used: (a) --resume from a block checkpoint (v2's
block-10 resume), (b) --start-ckpt from the block-0 checkpoint (the regression's R0s, R1, R3). For each path it takes one uninterrupted run
and one run checkpointed and restored at the same block, and compares, right after the restore and after further blocks: every Run attribute
(numbers, lists and dicts by their JSON form) and every channel attribute, the tensors by max |difference| in the site basis where the code
keeps them in the stage's eigenbasis (the memory D = V c, the FIFO modes V c_q), the stage's eigenvectors themselves by the gauge-invariant
projector difference ||V V^dagger - V' V'^dagger|| over the occupied and the memory's columns, and the curve's S_esc per block. A field the
restored run lacks or holds new is named as missing or added. It fixes nothing (a fix, only if a field is missing, is a new recursion file).
Usage: python restore_diff_page_recursion_v1.py OUT_DIR [--blocks-before 3 --blocks-after 4 --engine-path DIR]"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import page_recursion_v6b as PR      # noqa: E402

T4_ARGV = ["--rs", "4", "--g", "0.25", "--nsites", "192", "--channels", "1", "--ell", "2", "--Rb", "0.5", "--Rb0", "3", "--R-cut", "12", "--R-collar", "6",
           "--tau-s", "2", "--m-cap", "1000", "--eps-coll", "1e-5", "--eps-rank", "1e-9", "--eps-form", "1e-7", "--eps-orth", "1e-12", "--fifo-depth", "100",
           "--fifo-prune-weight", "0", "--store-cap", "100000", "--tau-tr", "8", "--dt-tr", "0.25"]           # page_selftest_v6b.py's T4_ARGV
V5MODE = ["--hull", "v5", "--pair-ledger", "v5", "--collector", "v5", "--direct-every", "5", "--end-rule", "v5"]
SKIP_RUN = {"a", "dev", "chans", "timings", "gpu_peak_run", "prefix"}        # the settings object, the device, the channels (compared apart), wall times, the tag
TIMING_KEYS = ("wall", "seconds", "_s", "timing", "gpu", "device_peak")       # wall-clock and device-memory entries inside records (not state)


def strip_timing(x):
    if isinstance(x, dict):
        return {k: strip_timing(v) for k, v in x.items() if not any(t in str(k) for t in TIMING_KEYS)}
    if isinstance(x, list):
        return [strip_timing(v) for v in x]
    return x


def tensor_like(x):
    return isinstance(x, torch.Tensor)


def jsonable(x):
    try:
        return json.loads(json.dumps(x, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    except (TypeError, ValueError):
        return str(x)


def diff_values(a, b):
    """max |a - b| for numbers and nested lists/dicts of numbers (None when the structures differ), 0.0 for equal strings/bools."""
    if isinstance(a, bool) or isinstance(b, bool) or isinstance(a, str) or isinstance(b, str) or a is None or b is None:
        return 0.0 if a == b else float("inf")
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) if not (isinstance(a, float) and math.isnan(a) and isinstance(b, float) and math.isnan(b)) else 0.0
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return float("inf")
        return max([diff_values(x, y) for x, y in zip(a, b)] or [0.0])
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return float("inf")
        return max([diff_values(a[k], b[k]) for k in a] or [0.0])
    return 0.0 if a == b else float("inf")


def snap(run):
    """the run's state in comparable form: Run attributes by their JSON form; per channel the site-basis memory and FIFO, the stage record's
    arrays, the scalars, and the stage eigenbasis's projectors."""
    out = {"run": {}, "channels": {}}
    for k, v in vars(run).items():
        if k in SKIP_RUN or callable(v) or tensor_like(v):
            continue
        out["run"][k] = strip_timing(jsonable(v))
    for ch in run.chans:
        st = ch.stage
        V = st["V"]
        c = {}
        c["D_site (V c)"] = (V @ ch.c).detach().clone()
        c["MD"] = ch.MD.detach().clone()
        c["half"] = ch.half.detach().clone()
        c["fifo modes (V c_q)"] = torch.cat([V @ cq for cq, _, _ in ch.fifo], dim=1) if ch.fifo else torch.zeros((V.shape[0], 0), dtype=V.dtype)
        c["fifo hq"] = torch.cat([hq.reshape(-1).to(torch.float64) for _, hq, _ in ch.fifo]) if ch.fifo else torch.zeros(0, dtype=torch.float64)
        c["fifo tq"] = torch.tensor([float(tq) for _, _, tq in ch.fifo], dtype=torch.float64)
        c["stage e"] = st["e"].detach().clone()
        c["stage neg"] = st["neg"].to(torch.float64)
        neg = st["neg"]
        c["P_neg (V_- V_-^dagger)"] = (V[:, neg] @ V[:, neg].mH).detach().clone()
        rec = ch.rec or {}
        for key in ("c_init", "MD_init"):
            if key in rec and tensor_like(rec[key]):
                c["rec " + key + (" (V .)" if key == "c_init" else "")] = (V @ rec[key]).detach().clone() if key == "c_init" else rec[key].detach().clone()
        for key in ("a", "halfQ", "gQ"):
            if key in rec and isinstance(rec[key], list) and rec[key]:
                c["rec " + key] = torch.cat([x.reshape(-1).to(torch.complex128) for x in rec[key]])
        if "cQ" in rec and rec["cQ"]:
            c["rec cQ (V .)"] = torch.cat([V @ x for x in rec["cQ"]], dim=1)
        scal = {}
        for k, v in vars(ch).items():
            if tensor_like(v) or k in ("stage", "rec", "fifo", "c", "MD", "half", "ops", "averager", "device") or callable(v):
                continue
            jv = jsonable(v)
            if isinstance(jv, str) and " object at 0x" in jv:
                continue                                                       # an operator object (its address), not state
            scal[k] = strip_timing(jv)
        for k in ("tQ", "n_coll", "E_rad", "E_delta_init", "t_init", "n_coll_all", "E_trunc", "gram_energy_bound", "cQ_saved", "cQ_from"):
            if k in rec:
                scal["rec " + k] = jsonable(rec[k])
        c["_scalars"] = scal
        out["channels"][str(ch.kappa)] = c
    return out


def compare(A, B):
    """the fields that differ between two snapshots: (name, max |diff|, shapes)"""
    rows = []
    ka, kb = set(A["run"]), set(B["run"])
    for k in sorted(ka - kb):
        rows.append(("run." + k, "missing after the restore", None))
    for k in sorted(kb - ka):
        rows.append(("run." + k, "added by the restore", None))
    for k in sorted(ka & kb):
        d = diff_values(A["run"][k], B["run"][k])
        if d != 0.0:
            rows.append(("run." + k, d, None))
    for kap in sorted(set(A["channels"]) | set(B["channels"])):
        ca, cb = A["channels"].get(kap, {}), B["channels"].get(kap, {})
        for k in sorted(set(ca) | set(cb)):
            if k == "_scalars":
                sa, sb = ca.get(k, {}), cb.get(k, {})
                for s in sorted(set(sa) - set(sb)):
                    rows.append(("kappa %s.%s" % (kap, s), "missing after the restore", None))
                for s in sorted(set(sb) - set(sa)):
                    rows.append(("kappa %s.%s" % (kap, s), "added by the restore", None))
                for s in sorted(set(sa) & set(sb)):
                    d = diff_values(sa[s], sb[s])
                    if d != 0.0:
                        rows.append(("kappa %s.%s" % (kap, s), d, None))
                continue
            if k not in ca or k not in cb:
                rows.append(("kappa %s.%s" % (kap, k), "missing after the restore" if k in ca else "added by the restore", None))
                continue
            x, y = ca[k], cb[k]
            if tuple(x.shape) != tuple(y.shape):
                rows.append(("kappa %s.%s" % (kap, k), "shape %s against %s" % (tuple(x.shape), tuple(y.shape)), None))
                continue
            d = float((x - y).abs().max()) if x.numel() else 0.0
            if d != 0.0:
                rows.append(("kappa %s.%s" % (kap, k), d, tuple(x.shape)))
    return rows


def curve_S(run):
    return [(r["t_over_rs1"], r["S_esc_total"]) for r in run.curve]


def path(name, base, out, nb, na):
    """main()'s two orders. (a) resume: A uninterrupted (start, the block-0 checkpoint, nb blocks); B the same with a checkpoint after its nb
    blocks, restored as B2 by resume(). (b) start: A as main() starts a fresh run (start, the block-0 checkpoint kept); B2 as main() starts
    from a stored block-0 checkpoint (the file copied, resume(), the pair ledger recomputed at block 0)."""
    import shutil
    rep = {"path": name}
    s_B1 = None
    if name == "resume":
        A = PR.Run(PR.parse(base + ["--tag", "rdA_resume"])); A.start(); A.checkpoint()
        for _ in range(nb):
            A.block()
        B = PR.Run(PR.parse(base + ["--tag", "rdB_resume"])); B.start(); B.checkpoint()
        for _ in range(nb):
            B.block()
        B.checkpoint()
        s_B1 = snap(B)
        del B
        B2 = PR.Run(PR.parse(base + ["--tag", "rdB_resume", "--resume"])); B2.resume()
    else:
        A = PR.Run(PR.parse(base + ["--tag", "rdA_start", "--keep-start-ckpt"])); A.start(); A.checkpoint()
        shutil.copyfile(A.prefix + "_ckpt.pt", A.prefix + "_ckpt_start.pt")
        B2 = PR.Run(PR.parse(base + ["--tag", "rdB_start", "--start-ckpt", A.prefix + "_ckpt_start.pt"]))
        shutil.copyfile(A.prefix + "_ckpt_start.pt", B2.prefix + "_ckpt.pt")
        B2.resume()
        if B2.n_block == 0:
            B2.recompute_pair_ledger()
    s_A1 = snap(A)
    s_B2 = snap(B2)
    rep["after_restore"] = [list(r) for r in compare(s_A1, s_B2)]
    if s_B1 is not None:
        rep["checkpointed_run_against_its_restore"] = [list(r) for r in compare(s_B1, s_B2)]
    nmore = na if name == "resume" else nb + na
    for _ in range(nmore):
        A.block(); B2.block()
    rep["after_more_blocks"] = [list(r) for r in compare(snap(A), snap(B2))]
    cA, cB = curve_S(A), curve_S(B2)
    rep["S_esc_per_block"] = {"uninterrupted": cA, "restored": cB,
                              "max_abs_diff_common": max([abs(x[1] - y[1]) for x, y in zip(cA[-len(cB):], cB)] or [0.0]) if cB else None}
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("--blocks-before", type=int, default=3); ap.add_argument("--blocks-after", type=int, default=4)
    ap.add_argument("--engine-path", default=None); ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    base = T4_ARGV + V5MODE + ["--tau-tr-doublings", "0", "--ckpt-every", "1000", "--device", "cpu", "--out", a.out] + (["--engine-path", a.engine_path] if a.engine_path else [])
    res = {"code": "restore_diff_page_recursion_v1.py", "recursion": "page_recursion_v6b.py", "case": "T4 (page_selftest_v6b.py) in v5 mode",
           "blocks_before": a.blocks_before, "blocks_after": a.blocks_after, "paths": []}
    for name in ("resume", "start"):
        r = path(name, base, a.out, a.blocks_before, a.blocks_after)
        res["paths"].append(r)
        print("== %s: after the restore %d differing fields; after %d more blocks %d; S_esc max |diff| %.3e" % (
            name, len(r["after_restore"]), a.blocks_after, len(r["after_more_blocks"]), r["S_esc_per_block"]["max_abs_diff_common"]), flush=True)
        for row in r["after_restore"][:40]:
            print("   after the restore: %s: %s%s" % (row[0], row[1] if isinstance(row[1], str) else "%.3e" % row[1], (" (shape %s)" % (row[2],)) if row[2] else ""), flush=True)
        for row in r["after_more_blocks"][:40]:
            print("   after %d more blocks: %s: %s%s" % (a.blocks_after, row[0], row[1] if isinstance(row[1], str) else "%.3e" % row[1], (" (shape %s)" % (row[2],)) if row[2] else ""), flush=True)
        if r.get("checkpointed_run_against_its_restore") is not None:
            print("   the checkpointed run against its own restore: %d differing fields" % len(r["checkpointed_run_against_its_restore"]), flush=True)
            for row in r["checkpointed_run_against_its_restore"][:40]:
                print("      %s: %s" % (row[0], row[1] if isinstance(row[1], str) else "%.3e" % row[1]), flush=True)
    res["seconds"] = time.time() - t0
    with open(os.path.join(a.out, "restore_diff_results.json"), "w") as f:
        json.dump(res, f, indent=1, default=str)
    print("RESTORE DIFF done in %.0f s (results in %s)" % (time.time() - t0, os.path.join(a.out, "restore_diff_results.json")), flush=True)


if __name__ == "__main__":
    main()
