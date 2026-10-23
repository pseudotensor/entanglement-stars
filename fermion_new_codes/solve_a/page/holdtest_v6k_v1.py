#!/usr/bin/env python
"""holdtest_v6k_v1.py (the lane's analyst builder; engine spec 43.3 (1)): the held-field (C) path of page_history_v6k on the history
oracle's synthetic fixture (page_history_oracle_v6f.synthetic_history: r_s/a 12, 256 sites, channels 1 and 2, held at T/r_s 6 for 6 r_s).
 1. off: v6k with option (v) off against v6j, every row and record value equal (first_diff of the oracle, wall-clock keys skipped).
 2. on: v6k with --collar-split against v6k off, every row and record value equal once the added keys (collar_split, collar_split_weighted,
    collar_split_label) are removed: the option reads and does not change the run.
 3. route: at every checkpoint of run 2 and per channel, the split recomputed independently from the saved collar block (C and P_- on the
    largest window) by engine_massive_channel_D_v5b.signed_energy itself (its eigencolumn route on the region starting at the window's
    first site; its window starting there is the same sine window) and compared with the code's numbers for that window size; criterion
    1e-10 relative to the largest flux part.
 4. printed, no criterion: per checkpoint the windowed net energy flux at 16, 32, 64 sites against the exact collar current L_at_t of the
    row, and the incoming/outgoing ratio (the local reading's check, not an identity).
Results in holdtest_v6k_results.json. Usage: --out DIR --device cpu --threads 8 --engine-dir DIR (where engine_massive_channel_D_v5b lives)."""
import argparse
import glob
import json
import math
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--device", default="cpu"); ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--engine-dir", dest="engine_dir", required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    import page_history_v6j as PJ
    import page_history_v6k as PK
    import page_history_oracle_v6f as O6F
    import page_history_oracle_v6j as OJ
    sys.path.insert(0, a.engine_dir)
    import engine_massive_channel_D_v5b as DV
    t0 = time.time()

    def argv(tag, extra=None):
        return (["--nsites", "256", "--channels", "1,2", "--R-cut-rs", "2.5", "--R-collar-rs", "1.5", "--collect-every-rs", "1", "--ext-every-rs", "4",
                 "--eps-loc", "1e-2", "--join-rs", "8,13", "--trace-chunk", "64", "--device", a.device, "--threads", str(a.threads),
                 "--out", a.out, "--tag", tag, "--chunk", "512", "--freeze-at-rs", "6", "--freeze-duration-rs", "6"] + (extra or []))

    RJ = PJ.HistRun(PJ.parse(argv("hold_v6j")), history=PJ.History(None, None, 256, synthetic=O6F.synthetic_history())); RJ.run()
    RK0 = PK.HistRun(PK.parse(argv("hold_v6k_off")), history=PK.History(None, None, 256, synthetic=O6F.synthetic_history())); RK0.run()
    RK1 = PK.HistRun(PK.parse(argv("hold_v6k_on", ["--collar-split"])), history=PK.History(None, None, 256, synthetic=O6F.synthetic_history())); RK1.run()
    res = {"code": "holdtest_v6k_v1.py", "rows": len(RK1.rows), "frozen_rows": sum(1 for r in RK1.rows if r.get("frozen"))}
    d1 = OJ.rows_equal(RJ, RK0)
    res["1_off_equal_v6j"] = {"pass": d1 is None, "first_difference": d1}
    ADDED = ("collar_split", "collar_split_weighted", "collar_split_label")

    def strip(x):
        if isinstance(x, dict):
            return {k: strip(v) for k, v in x.items() if k not in ADDED}
        if isinstance(x, list):
            return [strip(v) for v in x]
        return x

    class _R:
        pass
    S1 = _R(); S1.rows = strip(RK1.rows); S1.records = strip(RK1.records)
    d2 = OJ.rows_equal(RK0, S1)
    res["2_on_equal_off"] = {"pass": d2 is None, "first_difference": d2}
    # 3. the route check from the saved blocks
    worst = 0.0; n_cmp = 0; checks = []
    files = sorted(glob.glob(os.path.join(a.out, "hold_v6k_on_collar_blocks", "ck_T*.npz")))
    by_t = {round(float(np.load(f)["t"]), 9): f for f in files}
    for row in RK1.rows:
        f = by_t.get(round(float(row["t"]), 9))
        if f is None:
            checks.append({"t_over_rs": row["t_over_rs"], "missing_block": True}); worst = float("inf"); continue
        z = np.load(f)
        for c in RK1.chs:
            kap = int(c.ops.kappa)
            cs = row["per_channel"][str(kap)]["collar_split"]
            sites = z["k%d_sites" % kap]; slo = int(sites[0]) - 1; shi = int(sites[-1])
            R = np.asarray(c.ops.R, float)
            N, v = RK1.H.fields(int(row["k"]))                                  # the row's own record (the held rows: the frozen record)
            Dfull = z["k%d_C" % kap] - z["k%d_Pm" % kap]; Dfull = 0.5 * (Dfull + Dfull.conj().T)
            ic = int(cs["centre_site"]) - 1
            for Ls in sorted(cs["windows"], key=int):
                L = int(Ls); lo = ic - L // 2
                if lo < slo or lo + L > shi:
                    checks.append({"t_over_rs": row["t_over_rs"], "kappa": kap, "L": L, "skipped": "clipped at the chain's start"})
                    continue
                o = 2 * (lo - slo)
                Dl = Dfull[o:o + 2 * L, o:o + 2 * L]
                lam, xk = np.linalg.eigh(Dl)
                keep = np.abs(lam) > 1e-15
                X = np.zeros((2 * (len(R) - lo), int(keep.sum())), np.complex128)
                X[:2 * L] = xk[:, keep] * np.sqrt(np.abs(lam[keep]))[None, :]
                acc = DV.signed_energy((X, np.sign(lam[keep])), R, N, v, kap, 0.0, lo, L)
                arr = acc["_arrays"]
                iw = int(np.where(arr["centre_site"] == lo + L // 2)[0][0])
                q = arr["q"]; Nc = arr["N"][iw]; vc = arr["v"][iw]; Rc = arr["R"][iw]
                s_ = np.sin(q); c_ = np.cos(q); E = np.sqrt(s_ ** 2 + (kap / Rc) ** 2)
                om_p = Nc * E + 0.5 * vc * np.sin(2 * q); om_m = -Nc * E + 0.5 * vc * np.sin(2 * q)
                vp = Nc * s_ * c_ / E + vc * np.cos(2 * q); vm = -Nc * s_ * c_ / E + vc * np.cos(2 * q)
                npp = arr["n_plus"][iw]; nmm = arr["n_minus"][iw]
                W2 = float(np.sum(np.sin(math.pi * (np.arange(L) + 0.5) / L) ** 2))
                ref = {"E_flux_plus_out": float(np.sum((om_p * vp * npp)[vp > 0])) / W2, "E_flux_plus_in": float(np.sum((om_p * vp * npp)[vp <= 0])) / W2,
                       "E_flux_minus_out": float(np.sum((om_m * vm * nmm)[vm > 0])) / W2, "E_flux_minus_in": float(np.sum((om_m * vm * nmm)[vm <= 0])) / W2,
                       "E_density": (float(np.sum(om_p * npp)) + float(np.sum(om_m * nmm))) / W2}
                w_code = cs["windows"][Ls]
                scale = max(abs(ref[k]) for k in ref) or 1.0
                dev = max(abs(ref[k] - w_code[k]) for k in ref) / scale
                worst = max(worst, dev); n_cmp += 1
                checks.append({"t_over_rs": row["t_over_rs"], "kappa": kap, "L": L, "rel_dev": dev, "code": {k: w_code[k] for k in ref}, "D_v5b_route": ref})
    res["3_route"] = {"pass": bool(n_cmp > 0 and worst <= 1e-10), "worst_rel_dev": worst, "compared": n_cmp, "criterion": 1e-10, "checks": checks}
    # 4. printed
    pr = []
    for row in RK1.rows:
        w = row["collar_split_weighted"]
        pr.append({"t_over_rs": row["t_over_rs"], "frozen": bool(row.get("frozen")), "L_at_t": row.get("L_at_t"),
                   "net": {L: w[L]["E_flux_net"] for L in w}, "out": {L: w[L]["E_flux_out"] for L in w}, "in": {L: w[L]["E_flux_in"] for L in w},
                   "in_over_out": {L: w[L]["in_over_out"] for L in w}})
    res["4_printed"] = pr
    res["pass"] = bool(res["1_off_equal_v6j"]["pass"] and res["2_on_equal_off"]["pass"] and res["3_route"]["pass"])
    res["seconds"] = time.time() - t0
    with open(os.path.join(a.out, "holdtest_v6k_results.json"), "w") as f:
        json.dump(res, f, indent=1, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
    for k in ("1_off_equal_v6j", "2_on_equal_off"):
        print("%s  %s %s" % ("PASS" if res[k]["pass"] else "FAIL", k, res[k]["first_difference"] or ""), flush=True)
    print("%s  3_route worst %.3e over %d comparisons" % ("PASS" if res["3_route"]["pass"] else "FAIL", worst, n_cmp), flush=True)
    for p in pr:
        print("T/r_s %.3f frozen %s L_at_t %s net %s in/out %s" % (p["t_over_rs"], p["frozen"], p["L_at_t"],
              {k: "%.3e" % x for k, x in p["net"].items()}, {k: (("%.3f" % x) if x is not None else "-") for k, x in p["in_over_out"].items()}), flush=True)
    print("HOLDTEST %s in %.0f s" % ("PASS" if res["pass"] else "FAIL", res["seconds"]), flush=True)


if __name__ == "__main__":
    main()
