#!/usr/bin/env python
"""engine_c53p_v1.py (builder analyst; C5.3' of SPEC_final4b_engine.md sec. 17.3 (6)): the pair block of an engine_collapse_v10 run replayed
independently on its stored applied-Hamiltonian history and compared with the run's export.

The replay: the initial sea = the negative eigenvectors of h_m[N_cinit, v_cinit] per channel (the run's start-up fields); per step k the
applied fields (N_applied[k], v_applied[k]: the h_m that continued the reference and propagated the sea in the run); the reference R2 rerun
by engine_collapse_v10.continue_resolved from the previous step's (the same rule; a rerun, not an independent continuation); the pair block
eps_pairs = sum_kappa 16 kappa [dens(h, sea) - dens(h, R2)] and J_pairs likewise with Q (the run's definitions); the sea then propagated over
tau by a Chebyshev series of h_m in two half steps (the run uses the exact eigen-propagator: the propagation is the independent part), the
series' terms until the Bessel coefficient falls below 1e-15. Compared per step: the largest shell difference of eps_pairs and of J_pairs
from the export (absolute, and over the export's largest |value|), the totals (the pair energy and the summed current), and the sea's
orthonormality max |S^dag S - 1| over the channels (rank and unitarity). Args: --fields --m --kappa-sea --half-offset --gpu --every --md --out-json."""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
from scipy.special import jv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine_collapse_v5 as C5  # noqa: E402
import engine_collapse_v10 as C10  # noqa: E402


def cheb_prop(hT, X, dt, bound):
    torch = C5.BK["torch"]
    z = bound * dt
    K = int(z + 10)
    while abs(jv(K, z)) > 1e-15:
        K += 1
    Ht = lambda Y: (hT @ Y) / bound
    T0 = X; T1 = Ht(X)
    out = jv(0, z) * T0 + 2 * (-1j) * jv(1, z) * T1
    for k in range(2, K + 1):
        T2 = 2 * Ht(T1) - T0
        out = out + (2 * ((-1j) ** k) * jv(k, z)) * T2
        T0, T1 = T1, T2
    return out, K


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fields", required=True)
    ap.add_argument("--m", type=float, default=0.2)
    ap.add_argument("--kappa-sea", type=int, default=10)
    ap.add_argument("--half-offset", type=int, default=0)
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--md", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--tag", default=None, help="accepted for the GPU runner, unused")
    a = ap.parse_args()
    C5.backend(a.gpu)
    torch = C5.BK["torch"]
    Z = np.load(a.fields, allow_pickle=True)
    for k in ("N_applied", "v_applied", "N_cinit", "v_cinit", "eps_pairs", "J_pairs"):
        if k not in Z.files:
            raise SystemExit("the export lacks %s (engine_collapse_v10 with the applied fields is needed)" % k)
    R = np.asarray(Z["R"], float); nsites = len(R); tau = float(Z["tau"]); T = np.asarray(Z["T"], float)
    Na, va = np.asarray(Z["N_applied"], float), np.asarray(Z["v_applied"], float)
    epx, Jpx = np.asarray(Z["eps_pairs"], float), np.asarray(Z["J_pairs"], float)
    chs = [C5.Chan(k, a.m, nsites, bool(a.half_offset)) for k in range(1, a.kappa_sea + 1)]
    st = {}
    for ch in chs:
        hm = ch.clocked_dense(np.asarray(Z["N_cinit"], float), np.asarray(Z["v_cinit"], float))
        e, U = C5.eigT(hm)
        neg = e < 0
        st[ch.kappa] = {"sea": U[:, neg], "ref": U[:, neg], "hm_ref": hm, "n_sea": int(neg.sum())}
    rows = []
    t0 = time.time()
    for k in range(len(T)):
        ep = np.zeros(nsites); Jp = np.zeros(nsites)
        orth = 0.0; Ks = []
        cache = {}
        for ch in chs:
            s_ = st[ch.kappa]
            hm = ch.clocked_dense(Na[k], va[k])
            e, U = C5.eigT(hm)
            ref, cinfo = C10.continue_resolved(s_["ref"], s_["hm_ref"], hm, e, U, s_["n_sea"])
            if ref is None:
                raise SystemExit("step %d kappa %d: the continuation rerun was rejected (%s)" % (k, ch.kappa, cinfo))
            ep += ch.w * (C5.dens(ch.h, s_["sea"]) - C5.dens(ch.h, ref))
            Jp += ch.w * (C5.dens(ch.Q, s_["sea"]) - C5.dens(ch.Q, ref))
            G = s_["sea"].conj().T @ s_["sea"]
            orth = max(orth, float((G - torch.eye(G.shape[0], dtype=G.dtype, device=G.device)).abs().max()))
            cache[ch.kappa] = (hm, ref)
        sc_e = max(float(np.max(np.abs(epx[k]))), 1e-300); sc_J = max(float(np.max(np.abs(Jpx[k]))), 1e-300)
        row = {"step": k, "T": float(T[k]), "max_d_eps": float(np.max(np.abs(ep - epx[k]))), "max_d_eps_rel": float(np.max(np.abs(ep - epx[k]))) / sc_e,
               "max_d_J": float(np.max(np.abs(Jp - Jpx[k]))), "max_d_J_rel": float(np.max(np.abs(Jp - Jpx[k]))) / sc_J,
               "E_pairs_replay": float(ep.sum()), "E_pairs_export": float(epx[k].sum()), "J_sum_replay": float(Jp.sum()), "J_sum_export": float(Jpx[k].sum()),
               "orthonormality": orth}
        # propagate under this step's applied h_m (two Chebyshev half steps) and keep the reference
        for ch in chs:
            hm, ref = cache[ch.kappa]
            bound = float(hm.abs().sum(dim=1).max())
            X = st[ch.kappa]["sea"]
            for _ in range(2):
                X, K = cheb_prop(hm, X, tau / 2.0, bound)
            Ks.append(K)
            st[ch.kappa]["sea"] = X; st[ch.kappa]["ref"] = ref; st[ch.kappa]["hm_ref"] = hm
        row["cheb_terms_max"] = max(Ks)
        rows.append(row)
        if k % a.every == 0 or k == len(T) - 1:
            print("step %d T %.1f: max |d eps_pairs| %.2e (rel %.2e), max |d J_pairs| %.2e (rel %.2e); pair energy replay %.9e export %.9e; orthonormality %.1e; terms %d (%.0f s)" % (
                k, T[k], row["max_d_eps"], row["max_d_eps_rel"], row["max_d_J"], row["max_d_J_rel"], row["E_pairs_replay"], row["E_pairs_export"], orth, max(Ks),
                time.time() - t0), flush=True)
            json.dump({"script": "engine_c53p_v1.py", "args": vars(a), "rows": rows}, open(a.out_json, "w"), indent=1)
    L = ["## C5.3': the pair block replayed on the applied history (engine_c53p_v1.py on %s; %d steps, tau %g)" % (os.path.basename(a.fields), len(T), tau), "",
         "| step | T | max d eps_pairs (rel) | max d J_pairs (rel) | pair energy: replay, export | summed current: replay, export | orthonormality |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        if r["step"] % a.every == 0 or r["step"] == len(T) - 1:
            L.append("| %d | %.1f | %.2e (%.2e) | %.2e (%.2e) | %.9e, %.9e | %.6e, %.6e | %.1e |" % (r["step"], r["T"], r["max_d_eps"], r["max_d_eps_rel"], r["max_d_J"],
                     r["max_d_J_rel"], r["E_pairs_replay"], r["E_pairs_export"], r["J_sum_replay"], r["J_sum_export"], r["orthonormality"]))
    L += ["", "- the largest over all steps: d eps_pairs %.2e (rel %.2e), d J_pairs %.2e (rel %.2e), orthonormality %.1e; every number above is printed by the script." % (
        max(r["max_d_eps"] for r in rows), max(r["max_d_eps_rel"] for r in rows), max(r["max_d_J"] for r in rows), max(r["max_d_J_rel"] for r in rows),
        max(r["orthonormality"] for r in rows))]
    open(a.md, "w").write("\n".join(L) + "\n")
    json.dump({"script": "engine_c53p_v1.py", "args": vars(a), "rows": rows}, open(a.out_json, "w"), indent=1)


if __name__ == "__main__":
    main()
