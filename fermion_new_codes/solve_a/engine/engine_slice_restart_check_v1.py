#!/usr/bin/env python
"""engine_slice_restart_check_v1.py (builder analyst; inbox #36 B: "continue the sigma/a 4 member from its stored T/r_s 40 state ...; if
the T/r_s 40 state is not restartable, the production's own last state with its engine"). Record-only, CPU. Whether a stored orbital slice
can restart a run: the slice's sea, dust (with its occupations) and R2 columns per channel give the record's sources exactly as the engine's
replay gate forms them (engine_collapse_v19.resume_state: eps = sum_kappa w_kappa [dens(h, sea) - dens(h, R2) + dens(h, dust, f)], J the same
with Q; engine_collapse_v5.dens and Chan, the channels rebuilt on the host), compared with the record's stored shell totals eps_tot and J_tot
by the gate's own measure (the largest absolute difference over the largest stored value; the replay's acceptance is 1e-8); beside it per
channel the columns' orthonormality defects max |S - I| (S = X^dag X for the sea and R2) and the slice's stored precision. Nothing is
changed. Args: --slice (repeated, label=path=fields_npz) --m --nsites --kappa-sea --md."""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine_collapse_v5 as C5  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice", action="append", required=True, help="label=slice_npz=fields_npz")
    ap.add_argument("--m", type=float, default=0.2); ap.add_argument("--nsites", type=int, default=2048); ap.add_argument("--kappa-sea", type=int, default=10)
    ap.add_argument("--md", required=True)
    a = ap.parse_args()
    C5.backend(None)
    torch = C5.BK["torch"]
    chs = [C5.Chan(k, a.m, a.nsites, False) for k in range(1, a.kappa_sea + 1)]
    L = ["## Whether a stored orbital slice restarts the run (engine_slice_restart_check_v1.py; inbox #36 B; record-only)", "",
         "- the sources from the slice's columns as the engine's replay gate forms them, against the record's stored shell totals (the largest |difference| over the largest |stored|; the replay acceptance is 1e-8); the columns' orthonormality defects; the slice's stored dtype.", "",
         "| slice | T/rs | dtype | eps: rel. difference | J: rel. difference | max over channels of max abs(S - I), sea | R2 | E_pairs from the slice | stored eps_pairs sum |", "|---|---|---|---|---|---|---|---|---|"]
    for spec in a.slice:
        lab, sp_, fp = spec.split("=", 2)
        Z = np.load(sp_, allow_pickle=True); F = np.load(fp, allow_pickle=True)
        T = float(Z["T"]); rs = float(Z["rs"])
        Tz = np.asarray(F["T"], float); k = int(np.argmin(np.abs(Tz - T)))
        if abs(Tz[k] - T) > 1e-9:
            L.append("| %s | %.3f | - | the fields npz holds no record at that T | | | | | |" % (lab, T / rs)); continue
        w = {int(kk): float(ww) for kk, ww in zip(Z["kappas"], Z["weights"])}
        eps = np.zeros(a.nsites); J = np.zeros(a.nsites); ep = np.zeros(a.nsites)
        dS = dR = 0.0
        for ch in chs:
            sea = torch.as_tensor(np.asarray(Z["sea_%d" % ch.kappa]).astype(np.complex128))
            R2 = torch.as_tensor(np.asarray(Z["R2_%d" % ch.kappa]).astype(np.complex128))
            du = torch.as_tensor(np.asarray(Z["dust_%d" % ch.kappa]).astype(np.complex128)); f = np.asarray(Z["f_%d" % ch.kappa], float)
            ds = C5.dens(ch.h, sea) - C5.dens(ch.h, R2)
            eps += w[ch.kappa] * (ds + C5.dens(ch.h, du, f)); ep += w[ch.kappa] * ds
            J += w[ch.kappa] * (C5.dens(ch.Q, sea) - C5.dens(ch.Q, R2) + C5.dens(ch.Q, du, f))
            Ss = (sea.conj().T @ sea); Sr = (R2.conj().T @ R2)
            dS = max(dS, float((Ss - torch.eye(Ss.shape[0], dtype=Ss.dtype)).abs().max()))
            dR = max(dR, float((Sr - torch.eye(Sr.shape[0], dtype=Sr.dtype)).abs().max()))
            del sea, R2, du, Ss, Sr
        es = np.asarray(F["eps_tot"][k], float); Js = np.asarray(F["J_tot"][k], float)
        de = float(np.max(np.abs(eps - es))) / max(float(np.max(np.abs(es))), 1e-300)
        dj = float(np.max(np.abs(J - Js))) / max(float(np.max(np.abs(Js))), 1e-300)
        eps_p = float(np.asarray(F["eps_pairs"][k], float).sum()) if "eps_pairs" in F.files else float("nan")
        L.append("| %s | %.3f | %s | %.3e | %.3e | %.3e | %.3e | %.6e | %.6e |" % (lab, T / rs, str(Z["dtype"]), de, dj, dS, dR, float(ep.sum()), eps_p))
        print(L[-1], flush=True)
    L += ["", "- every number above is printed by the script from the slices and the runs' fields npz."]
    open(a.md, "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
