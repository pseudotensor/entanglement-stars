#!/usr/bin/env python
"""engine_step_diag_v1.py (builder analyst; engine spec 28.5, S9 (a) of REVIEWA_REVIEW_web40e: "on the stored histories by post-processing (CPU
where the stored eigenvalues and masks suffice, else by replay): per step and channel ||Delta h||/g_mix and ||h_dot||/g_mix^2 over one
engine sub-step, g_mix the smallest occupied-to-unoccupied gap of the record's eigenvalues, the operator norm by a few Lanczos iterations
(the Frobenius norm the labelled bound); steps with either above 1/8 marked unresolved; per run the unresolved fraction over all
(channel, step) pairs and over the pocket-phase steps"). Per native record k with a stored spectrum (the run's spectra npz: eigenvalues e and
the packed occupation mask per channel) and per channel: Delta h = h_m[N_k, v_k] - h_m[N_k-1, v_k-1] with the records' applied site fields
(the engine's clocked_host_sparse: (1/2){N, h} + (1/2){v, Q}); ||Delta h|| by Lanczos (scipy eigsh, the largest magnitude, tol 1e-8), its
Frobenius norm and its Gershgorin bound beside; the engine's sub-step count recomputed by its own rule (tau_LZ = 1/(c_LZ sqrt(G/tau)), G the
Gershgorin bound; nsub = max(1, round(c_LZ sqrt(tau G))) when tau_LZ <= tau, else 1); per sub-step ||Delta h||/(nsub g_mix) and the rate
||Delta h||/(tau g_mix^2); g_mix the smallest |e_o - e_e| between the occupied and the empty eigenvalues of record k (the mask). Unresolved
when either ratio exceeds --threshold (1/8). Per run: the unresolved fraction over all (channel, step) pairs and over the pocket-phase
steps (records whose stored n_pocket sum is positive), the largest ratios and where. Every number printed by the script. CPU.
Args: --run dir/tag --engine (the run's engine module for the channels: engine_collapse_v19) --c-LZ --threshold --md --out-json."""
import argparse
import importlib
import json
import math
import os
import sys
import time

import numpy as np
import scipy.sparse.linalg as spla

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def gmix(e, occ):
    """the smallest |e_o - e_e| between the occupied and the empty eigenvalues."""
    order = np.argsort(e); es = e[order]; os_ = occ[order]
    d = np.diff(es); diff = os_[1:] != os_[:-1]
    return float(np.min(d[diff])) if diff.any() else float("inf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--engine", default="engine_collapse_v19"); ap.add_argument("--c-LZ", dest="c_LZ", type=float, default=None)
    ap.add_argument("--threshold", type=float, default=0.125); ap.add_argument("--md", required=True); ap.add_argument("--out-json", required=True)
    a = ap.parse_args()
    E = importlib.import_module(a.engine)
    E.C5.backend(None)
    d, tag = os.path.split(a.run)
    J = json.load(open(os.path.join(d, "collapse_%s.json" % tag)))
    Z = np.load(os.path.join(d, "collapse_%s_fields.npz" % tag), allow_pickle=True)
    S = np.load(os.path.join(d, "collapse_%s_spectra.npz" % tag), allow_pickle=True)
    ra = J["args"]; tau = float(ra["tau"]); rs = float(ra["rs"]); cLZ = a.c_LZ if a.c_LZ is not None else float(ra.get("c_LZ", 1.0))
    chs = [E.Chan(k, ra["m"], ra["nsites"], bool(ra.get("half_offset", 0))) for k in range(1, ra["kappa_sea"] + 1)]
    Tz = np.asarray(Z["T"], float); TS = np.asarray(S["T"], float)
    recs = {round(float(r["T"]), 6): r for r in J["records"] if r.get("E") is not None}
    rows = []; t0 = time.time()
    for i, T in enumerate(TS):
        kz = int(np.argmin(np.abs(Tz - T)))
        if kz == 0 or abs(Tz[kz] - T) > 1e-9:
            continue
        Nk, vk = np.asarray(Z["N_applied"][kz], float), np.asarray(Z["v_applied"][kz], float)
        Np, vp = np.asarray(Z["N_applied"][kz - 1], float), np.asarray(Z["v_applied"][kz - 1], float)
        rec = recs.get(round(T, 6), {})
        npk = sum((rec.get("n_pocket") or {}).values()) if isinstance(rec.get("n_pocket"), dict) else int(rec.get("n_pocket") or 0)
        for ch in chs:
            dH = (E.clocked_host_sparse(ch, Nk, vk) - E.clocked_host_sparse(ch, Np, vp)).tocsr()
            nG = float(np.max(np.asarray(abs(dH).sum(axis=1)).ravel())) if dH.nnz else 0.0
            nF = float(np.sqrt((abs(dH.data) ** 2).sum())) if dH.nnz else 0.0
            if dH.nnz:
                try:
                    n2 = float(abs(spla.eigsh(dH, k=1, which="LM", tol=1e-8, return_eigenvectors=False)[0]))
                except Exception:
                    n2 = nF
            else:
                n2 = 0.0
            e = np.asarray(S["e_%d" % ch.kappa][i], float)
            occ = np.unpackbits(np.asarray(S["mask_%d" % ch.kappa][i], np.uint8))[:len(e)].astype(bool)
            g = gmix(e, occ)
            sT = nG / tau
            tLZ = 1.0 / (cLZ * math.sqrt(sT)) if sT > 0 else math.inf
            nsub = max(1, int(round(cLZ * math.sqrt(tau * nG)))) if tLZ <= tau else 1
            r1 = n2 / (nsub * g) if g > 0 else math.inf; r2 = n2 / (tau * g * g) if g > 0 else math.inf
            rows.append({"T_over_rs": T / rs, "kappa": ch.kappa, "norm2": n2, "normF": nF, "gersh": nG, "g_mix": g, "nsub": nsub, "ratio_step": r1, "ratio_rate": r2,
                         "unresolved": bool(r1 > a.threshold or r2 > a.threshold), "pocket_record": bool(npk > 0)})
        if i % 20 == 0:
            print("record %d (T/rs %.3f), %.0f s" % (i, T / rs, time.time() - t0), flush=True)
    n = len(rows); un = sum(r["unresolved"] for r in rows)
    pk = [r for r in rows if r["pocket_record"]]; unp = sum(r["unresolved"] for r in pk)
    w1 = max(rows, key=lambda r: r["ratio_step"]) if rows else None; w2 = max(rows, key=lambda r: r["ratio_rate"]) if rows else None
    L = ["## Per-step resolution diagnostics (engine_step_diag_v1.py; engine spec 28.5, S9 (a)): %s (tau %g, c_LZ %g, threshold %g)" % (tag, tau, cLZ, a.threshold), "",
         "- (channel, step) pairs: %d; unresolved (either ratio above the threshold): %d (%.4f); over the pocket-phase records: %d pairs, %d unresolved (%s)." % (
             n, un, (un / n) if n else float("nan"), len(pk), unp, ("%.4f" % (unp / len(pk))) if pk else "-"),
         "- the largest ||dh||/(nsub g_mix): %.4e at T/rs %.3f, kappa %d (g_mix %.3e, ||dh|| %.3e, nsub %d); the largest ||dh||/(tau g_mix^2): %.4e at T/rs %.3f, kappa %d (g_mix %.3e)." % (
             w1["ratio_step"], w1["T_over_rs"], w1["kappa"], w1["g_mix"], w1["norm2"], w1["nsub"], w2["ratio_rate"], w2["T_over_rs"], w2["kappa"], w2["g_mix"]) if rows else "- no rows",
         "", "| T/rs | unresolved channels (of %d) | the largest ratio per sub-step (kappa) | the largest rate ratio (kappa) | the smallest g_mix (kappa) | pocket record |" % len(set(r["kappa"] for r in rows)),
         "|---|---|---|---|---|---|"]
    byT = {}
    for r in rows:
        byT.setdefault(r["T_over_rs"], []).append(r)
    for T in sorted(byT):
        rr = byT[T]; u = sum(x["unresolved"] for x in rr)
        a1 = max(rr, key=lambda x: x["ratio_step"]); a2 = max(rr, key=lambda x: x["ratio_rate"]); a3 = min(rr, key=lambda x: x["g_mix"])
        if u or rr[0]["pocket_record"] or abs(T * 8 - round(T * 8)) < 1e-9 and abs(T - round(T)) < 1e-9:
            L.append("| %.3f | %d | %.3e (%d) | %.3e (%d) | %.3e (%d) | %s |" % (T, u, a1["ratio_step"], a1["kappa"], a2["ratio_rate"], a2["kappa"], a3["g_mix"], a3["kappa"], "yes" if rr[0]["pocket_record"] else "-"))
    L += ["", "- rows printed: every record with an unresolved channel or a pocket, and every whole T/rs; the full per-(channel, step) list is in the json; every number above is printed by the script; the norm is Lanczos's (the Frobenius norm and the Gershgorin bound in the json)."]
    open(a.md, "w").write("\n".join(L) + "\n")
    json.dump({"script": "engine_step_diag_v1.py", "run": a.run, "c_LZ": cLZ, "threshold": a.threshold, "rows": rows}, open(a.out_json, "w"), default=float)
    print("\n".join(L[:4]))


if __name__ == "__main__":
    main()
