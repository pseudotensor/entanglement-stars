#!/usr/bin/env python3
"""engine_vacflow_v4.py (builder analyst; J-VF, SPEC_final4b_engine.md sec. 15.2's E_cut flag: version 4 of engine_vacflow_v3.py, the flag's
measurement only). Version 3's --benchmark printed the E_cut x 1.5 change as the maximum over every site below R_read, which sits at the
channel's truncated inner edge (R = kappa/E_cut, where the change of E_cut moves the chain's first site), not at the probe radii the lead's
rule names ("a channel whose change exceeds 10 percent of its own |raw| at a probe radius is flagged"). Version 4's --ecut-probes K1,K2,..
runs each channel at E_cut and 1.5 E_cut (and dt/2 as the control) for the ramps of --tr-list and prints, per channel, ramp and key (raw, NO;
the hold reads): at each probe radius R_p in {1.5, 2, 4, 8, 16} r_s the channel's own cumulative contribution E_k(R_p) = sum_{R <= R_p} rho_k
and its change under E_cut x 1.5 (the flag: |dE_k(R_p)| > 0.1 |E_k(R_p)| for raw), the per-site density and its change at R_p, and the
maximum per-site change with its radius beside the inner edges kappa/E_cut and kappa/(1.5 E_cut). Every production path is version 3's
(imported unchanged). Args: --rs --profile --m --Ecut --tr-list --dt --gpu --ecut-probes --out-json --md."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
import argparse
import json
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine_vacflow_v3 as V3  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rs", type=float, default=16.0)
    ap.add_argument("--profile", default="exact", choices=["exact", "notrap"])
    ap.add_argument("--m", type=float, default=0.0)
    ap.add_argument("--Ecut", type=float, default=6.0)
    ap.add_argument("--tr-list", default="2,32")
    ap.add_argument("--dt", type=float, default=0.5)
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--ecut-probes", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--md", required=True)
    a = ap.parse_args()
    import torch
    dev = torch.device("cuda:%d" % a.gpu) if a.gpu is not None else torch.device("cpu")
    rs = a.rs
    nsites = int(64 * rs)
    Rfull = np.arange(nsites, dtype=float) + 1.0
    trs = [float(x) for x in a.tr_list.split(",")]
    probes = [p for p in (1.5 * rs, 2 * rs, 4 * rs, 8 * rs, 16 * rs) if p < nsites]
    out = []
    L = ["## J-VF: the E_cut x 1.5 flag at the probe radii (engine_vacflow_v4.py; r_s %g, profile %s, m %g, E_cut %g against %g; ramps %s r_s; the hold reads)" % (
        rs, a.profile, a.m, a.Ecut, 1.5 * a.Ecut, a.tr_list), "",
         "| kappa | t_r/r_s | key | R_p/r_s | E_k(R_p) | dE_k under E_cut x 1.5 | |dE|/|E| | flag (raw, > 0.1) | rho_k(R_p) | d rho_k | dt/2 change of E_k(R_p) |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for K in [int(x) for x in a.ecut_probes.split(",")]:
        t0 = time.time()
        c1 = V3.run_channel(K, Rfull, rs, a.profile, a.m, a.Ecut, trs, a.dt, dev)
        c2 = V3.run_channel(K, Rfull, rs, a.profile, a.m, a.Ecut, trs, a.dt / 2, dev)
        c3 = V3.run_channel(K, Rfull, rs, a.profile, a.m, a.Ecut * 1.5, trs, a.dt, dev)
        rec = {"kappa": K, "edge": float(K / a.Ecut), "edge_control": float(K / (1.5 * a.Ecut)), "seconds": time.time() - t0, "ramps": {}}
        for tr in trs:
            k = V3.trkey(tr)
            rec["ramps"][k] = {}
            for key in ("raw_hold", "NO_hold"):
                x1, x2, x3 = (np.asarray(c["reads"][k][key], float) for c in (c1, c2, c3))
                cs1, cs2, cs3 = np.cumsum(x1), np.cumsum(x2), np.cumsum(x3)
                d13 = np.abs(x3 - x1)
                imax = int(np.argmax(d13[Rfull < 16 * rs]))
                rows = []
                for p in probes:
                    i = int(p) - 1
                    E, dE, dEt = float(cs1[i]), float(cs3[i] - cs1[i]), float(cs2[i] - cs1[i])
                    rel = abs(dE) / abs(E) if E != 0 else float("inf")
                    flag = bool(key == "raw_hold" and rel > 0.1)
                    rows.append({"R_p_over_rs": p / rs, "E": E, "dE_ecut": dE, "rel": rel, "flag": flag, "rho": float(x1[i]), "drho_ecut": float(x3[i] - x1[i]), "dE_dt": dEt})
                    L.append("| %d | %g | %s | %g | %.4e | %.3e | %.3e | %s | %.3e | %.3e | %.2e |" % (K, tr, key.split("_")[0], p / rs, E, dE, rel, "FLAG" if flag else "-",
                             float(x1[i]), float(x3[i] - x1[i]), dEt))
                rec["ramps"][k][key] = {"probes": rows, "max_site_change": float(d13[imax]), "R_of_max": float(Rfull[imax])}
                L.append("| %d | %g | %s | max per-site change %.3e at R = %g (edges %.2f, %.2f) | | | | | | | |" % (K, tr, key.split("_")[0], float(d13[imax]), float(Rfull[imax]),
                         K / a.Ecut, K / (1.5 * a.Ecut)))
        out.append(rec)
        print("kappa %d done (%.0f s)" % (K, time.time() - t0), flush=True)
        json.dump({"script": "engine_vacflow_v4.py", "args": vars(a), "channels": out}, open(a.out_json, "w"), indent=1)
    L += ["", "- every number above is printed by the script; E_k(R_p) is the channel's own unweighted contribution to the shell sum up to R_p (version 3's totals",
          "  carry the weight 16 kappa); the lead's rule (engine 15.2) flags a channel whose E_cut change exceeds 10 percent of its own |raw| at a probe radius."]
    open(a.md, "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
