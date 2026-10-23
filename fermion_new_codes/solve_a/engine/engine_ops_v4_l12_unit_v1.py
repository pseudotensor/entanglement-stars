#!/usr/bin/env python
"""engine_ops_v4_l12_unit_v1.py (builder analyst; the brief BRIEF_v26_builder_S4_S9, the unit tests: "S4 (d) on the L = 12 chain at m a 0
(reuse engine_l12_oriented_v2's setup) through ops v4 itself ...: ops v4 against ops v3 and against the propagated state").

The chain of engine_l12_unit_v1 at m a = 0 with the ramp amplitude --amp (1.3, engine_l12_oriented_v2's setting: each D-sector's lowest pair
of levels passes through zero near s 0.65); h(s) = H1 + r(s) DH, r(s) = s^3 (10 - 15 s + 6 s^2); per D-sector (12 levels, real skew,
+- paired: the engine's channel analogue) the occupation bits of the flat sea in H1's eigenbasis. Per sector and per (n_rec, tau): the
propagated state R(s) (DOP853, rtol 1e-12, atol 1e-14, its dense output at the records) against four continuations along n_rec record steps (the engine's step tau/n_rec,
h_of = Delta r DH, eig_at for the Landau-Zener sub-steps): engine_ops_v3.continue_sea_v3 and engine_ops_v4.continue_sea_v4 (the
production routines, unchanged, c_LZ 1, the window kept as the engine keeps it: the last accepted entry with ref, f, gap, weyl_gap, mask
and, for v4, lines) with v4's isolation "weyl" (the default) and "angle" (the labelled variant), and the fixed-bit spectral reference
(re-fills at the crossing). With --insert-crossing the record grid also holds s*, where the lowest pair's eps(s) is smallest (bounded
minimization: the protected zero, eps(s*) printed), so one record sits on the exact degeneracy (ops v3 rotates the block there).
Printed: d(R, P) = ||R - P||_F^2/2 at s = 1 and the largest over the records, summed over the two sectors; d(P_v4, P_v3) at s = 1 and
the largest; per run the transported pair-steps, the largest transport residual and neutrality defect, the certificate failures
(2 eta_n >= g_n) and the v3 runs' neutrality defect of its occupied line in the lowest pair's plane at the record after s*. Every number is
printed by the script. CPU. Args: --amp --taus --nrecs --insert-crossing --md --json."""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import scipy.sparse as sp
from scipy.integrate import solve_ivp
from scipy.linalg import eigh
from scipy.optimize import minimize_scalar

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine_l12_unit_v1 as U1  # noqa: E402
import engine_ops_v3 as OPS3  # noqa: E402
import engine_ops_v4 as OPS4  # noqa: E402


def d2(A, B):
    return 0.5 * float(np.linalg.norm(A - B, "fro") ** 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--amp", type=float, default=1.3); ap.add_argument("--taus", default="1024,4096,16384,65536"); ap.add_argument("--nrecs", default="8,32,128,512")
    ap.add_argument("--insert-crossing", default="32,128", help="n_rec values also run with s* inserted")
    ap.add_argument("--md", required=True); ap.add_argument("--json", required=True)
    a = ap.parse_args()
    import torch
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    U1.M_A = 0.0; U1.AMP = a.amp
    H0, Q, H1, DH, sectors = U1.build()
    assert np.max(np.abs(H1 + H1.conj())) < 1e-14 and np.max(np.abs(DH + DH.conj())) < 1e-14, "not real skew at m 0"
    e0, z0 = eigh(H0); P0 = z0[:, e0 < 0] @ z0[:, e0 < 0].conj().T
    hS = [H1[np.ix_(ids, ids)] for ids in sectors]; dS = [DH[np.ix_(ids, ids)] for ids in sectors]
    bits = []
    for k, ids in enumerate(sectors):
        e, z = eigh(hS[k]); w = np.real(np.diag(z.conj().T @ P0[np.ix_(ids, ids)] @ z)); bits.append(w > 0.5)
    hs = lambda k, s: hS[k] + U1.ramp(s) * dS[k]
    tt = lambda X: torch.as_tensor(X, dtype=torch.complex128)

    def eig_t(Hm):
        e, Uv = torch.linalg.eigh(tt(Hm)); return e.real, Uv

    def eps_low(s, k=0):
        e = eigh(hs(k, s), eigvals_only=True)
        n = len(e)
        return 0.5 * (e[n // 2] - e[n // 2 - 1])
    xs = np.linspace(0.0, 1.0, 4001)
    ev = np.array([eps_low(x) for x in xs]); k0 = int(np.argmin(ev))
    res = minimize_scalar(eps_low, bounds=(xs[max(k0 - 1, 0)], xs[min(k0 + 1, len(xs) - 1)]), method="bounded", options={"xatol": 1e-15})
    s_star, eps_star = float(res.x), float(res.fun)
    eps_star2 = eps_low(s_star, 1)

    def spectral(k, s):
        e, z = eigh(hs(k, s)); return z[:, bits[k]] @ z[:, bits[k]].conj().T

    prop_cache = {}

    def propagate(k, tau, s_list):
        """the propagated state at the record points: one DOP853 run per (sector, tau) with dense output (cached)."""
        e, z = eigh(hS[k]); Y0 = z[:, bits[k]]
        key = (k, tau)
        if key not in prop_cache:
            def rhs(s, y):
                return (-1j * tau * (hs(k, s) @ y.reshape(Y0.shape))).ravel()
            sol = solve_ivp(rhs, (0.0, 1.0), Y0.ravel().astype(complex), method="DOP853", rtol=1e-12, atol=1e-14, dense_output=True)
            assert sol.success, sol.message
            prop_cache[key] = sol
        sol = prop_cache[key]
        return [Y0 if s_ == 0.0 else sol.sol(s_).reshape(Y0.shape) for s_ in s_list]

    def engine_run(k, s_list, tau_step, rule, iso="weyl"):
        """rule: v3 or v4 (iso weyl | angle); the window as the engine keeps it."""
        opts = dict(OPS4.OPTS); opts["isolation"] = iso
        e_i, U_i = eig_t(hS[k])
        occ0 = bits[k]
        ref0 = U_i[:, torch.as_tensor(np.where(occ0)[0])]
        n_sea = int(occ0.sum())
        e_np = e_i.numpy()
        _r, inf0, _o, blocks0, _rot, blk0 = OPS4._assign_v3_full(ref0, e_i, U_i, n_sea)
        win = {"ref": ref0, "f": (np.array([0.0]), np.array([0.0])), "gap": inf0.get("gap_between_blocks"), "weyl_gap": inf0.get("weyl_gap"), "mask": occ0.copy()}
        if rule == "v4":
            win["lines"] = OPS4.lines_from(torch, e_np, U_i, occ0, np.ones(len(e_np), dtype=np.int64), [], opts["n_track"])
        window = [win]
        Ps = [(ref0 @ ref0.conj().T).numpy()]
        st = {"unverified": 0, "transported": 0, "max_transport_residual": 0.0, "max_neutrality_defect": 0.0, "pairs": [], "ties": 0, "partly_occupied": 0}
        eig_at = lambda ch, f0, f1: eig_t(hS[k] + float(np.asarray(f0).ravel()[0]) * dS[k])
        h_of = lambda ch, d0, d1: sp.csr_matrix(float(np.asarray(d0).ravel()[0]) * dS[k])
        lines_after = []
        for j in range(1, len(s_list)):
            s = s_list[j]; f_k = (np.array([U1.ramp(s)]), np.array([0.0]))
            e_k, U_k = eig_t(hs(k, s))
            if rule == "v3":
                ref, ci, mask = OPS3.continue_sea_v3(None, window, f_k, e_k, U_k, n_sea, tau_step[j], 1.0, eig_at, h_of)
                window = [{"ref": ref, "f": f_k, "gap": ci.get("gap_between_blocks"), "weyl_gap": ci.get("weyl_gap", ci.get("gap_between_blocks")), "mask": mask}]
            else:
                ref, ci, mask = OPS4.continue_sea_v4(None, window, f_k, e_k, U_k, n_sea, tau_step[j], 1.0, eig_at, h_of, opts=opts)
                window = [{"ref": ref, "f": f_k, "gap": ci.get("gap_between_blocks"), "weyl_gap": ci.get("weyl_gap", ci.get("gap_between_blocks")), "mask": mask,
                           "lines": ci.get("lines")}]
                for q in (ci.get("oriented") or {}).get("pairs_all_substeps") or []:
                    st["transported"] += 1
                    st["max_transport_residual"] = max(st["max_transport_residual"], q["transport_residual"])
                    st["max_neutrality_defect"] = max(st["max_neutrality_defect"], q["neutrality_defect"])
                    if len(st["pairs"]) < 400:
                        st["pairs"].append({"s": s, "substep": q["substep"], "eps": q["eps"], "sep": q["sep"], "eta": q["eta"], "cos_min": q["cos_min"], "det_O": q["det_O"],
                                            "signed_energy": q["signed_energy"], "occupied": q["occupied"], "transport_residual": q["transport_residual"]})
            st["unverified"] += int(not ci.get("verified", False)); st["ties"] += int(bool(ci.get("ambiguous_tie"))); st["partly_occupied"] += int((ci.get("partly_occupied_blocks") or 0) > 0)
            # the occupied line's neutrality in the lowest pair's plane (the line the rule holds there)
            e_np = e_k.numpy(); n = len(e_np)
            Z = U_k[:, torch.as_tensor([n // 2 - 1, n // 2])]
            W = Z.conj().T @ ref
            G = (W @ W.conj().T).numpy()
            lam, vec = np.linalg.eigh(G)
            phi = (Z @ tt(vec[:, -1])).numpy()
            lines_after.append({"s": s, "weight_in_plane": float(np.trace(G).real), "neutrality_defect": float(abs(phi @ phi))})
            Ps.append((ref @ ref.conj().T).numpy())
        return Ps, st, lines_after

    rows = []
    t0 = time.time()
    grids = [(int(x), False) for x in a.nrecs.split(",") if x.strip()] + [(int(x), True) for x in a.insert_crossing.split(",") if x.strip()]
    for nrec, ins in grids:
        s_list = [kk / nrec for kk in range(nrec + 1)]
        if ins:
            s_list = sorted(set(s_list + [s_star]))
        for tau in [float(x) for x in a.taus.split(",")]:
            tau_step = [0.0] + [tau * (s_list[j] - s_list[j - 1]) for j in range(1, len(s_list))]
            dist = {lab: np.zeros(len(s_list)) for lab in ("v3", "v4_weyl", "v4_angle", "spectral")}
            cross = {"v4w_v3": np.zeros(len(s_list)), "v4a_v3": np.zeros(len(s_list))}
            stats = {}; neut = {}
            for k in range(len(sectors)):
                Ys = propagate(k, tau, s_list)
                Rs = [Y @ Y.conj().T for Y in Ys]
                P3, s3, l3 = engine_run(k, s_list, tau_step, "v3")
                P4w, s4w, l4w = engine_run(k, s_list, tau_step, "v4", "weyl")
                P4a, s4a, l4a = engine_run(k, s_list, tau_step, "v4", "angle")
                Psp = [spectral(k, s) for s in s_list]
                for lab, PP in (("v3", P3), ("v4_weyl", P4w), ("v4_angle", P4a), ("spectral", Psp)):
                    dist[lab] += np.array([d2(R_, P_) for R_, P_ in zip(Rs, PP)])
                cross["v4w_v3"] += np.array([d2(x_, y_) for x_, y_ in zip(P4w, P3)]); cross["v4a_v3"] += np.array([d2(x_, y_) for x_, y_ in zip(P4a, P3)])
                stats["sector%d" % (k + 1)] = {"v3": s3, "v4_weyl": s4w, "v4_angle": s4a}
                if ins:
                    jx = s_list.index(s_star)
                    neut["sector%d" % (k + 1)] = {"at_s_star": {"v3": l3[jx - 1], "v4_weyl": l4w[jx - 1], "v4_angle": l4a[jx - 1]},
                                                  "next_record": {"v3": l3[jx] if jx < len(l3) else None, "v4_weyl": l4w[jx] if jx < len(l4w) else None,
                                                                  "v4_angle": l4a[jx] if jx < len(l4a) else None}}
            row = {"n_rec": nrec, "inserted_s_star": ins, "records": len(s_list) - 1, "tau": tau, "final": {k_: float(v_[-1]) for k_, v_ in dist.items()},
                   "largest": {k_: float(v_.max()) for k_, v_ in dist.items()}, "v4_vs_v3_final": {k_: float(v_[-1]) for k_, v_ in cross.items()},
                   "v4_vs_v3_largest": {k_: float(v_.max()) for k_, v_ in cross.items()}, "stats": stats, "neutrality_at_crossing": neut}
            rows.append(row)
            print("n_rec %d%s tau %g done (%.0f s)" % (nrec, " (s* inserted)" if ins else "", tau, time.time() - t0), flush=True)
    out = {"script": "engine_ops_v4_l12_unit_v1.py", "amp": a.amp, "bits": [[int(b) for b in x] for x in bits], "s_star": s_star, "eps_at_s_star": [eps_star, eps_star2], "rows": rows}
    json.dump(out, open(a.json, "w"), indent=1, default=float)
    L = ["## S4 (d) on the L = 12 chain at m a 0 through engine_ops_v4 itself (engine_ops_v4_l12_unit_v1.py)", "",
         "- the chain of engine_l12_unit_v1 at m a 0 (h = i A, A real skew, asserted), ramp amplitude %g; per D-sector (12 levels, +- paired) the flat sea's bits %s; each sector's lowest pair has its smallest eps at s* = %.12f (eps(s*) = %.3e, %.3e in the two sectors: the protected zero)." % (
             a.amp, "; ".join("(" + ",".join(str(int(b)) for b in x) + ")" for x in bits), s_star, eps_star, eps_star2),
         "- d(R, P) = ||R - P||_F^2/2 summed over the two sectors, R the propagated state; v3: engine_ops_v3.continue_sea_v3; v4 weyl / v4 angle: engine_ops_v4.continue_sea_v4 with the Weyl isolation (default) / the angle variant; spectral: the fixed bits in eigenvalue order (re-fills at the crossing); all at c_LZ 1 with the engine's window.", "",
         "| n_rec | s* inserted | tau | d(R, v3) s = 1; largest | d(R, v4 weyl) s = 1; largest | d(R, v4 angle) s = 1; largest | d(R, spectral) s = 1; largest | d(v4 weyl, v3) s = 1; largest | d(v4 angle, v3) s = 1; largest | v4 weyl transported pair-steps (sector 1, 2) | v4 angle transported | largest transport residual (weyl; angle) | unverified steps v3 (sector 1, 2) | unverified steps v4 weyl |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        f, g = r["final"], r["largest"]; c, cl = r["v4_vs_v3_final"], r["v4_vs_v3_largest"]
        S = r["stats"]
        L.append("| %d | %s | %g | %.3e; %.3e | %.3e; %.3e | %.3e; %.3e | %.3e; %.3e | %.3e; %.3e | %.3e; %.3e | %d, %d | %d, %d | %.2e; %.2e | %d, %d | %d, %d |" % (
            r["n_rec"], "yes" if r["inserted_s_star"] else "no", r["tau"], f["v3"], g["v3"], f["v4_weyl"], g["v4_weyl"], f["v4_angle"], g["v4_angle"], f["spectral"], g["spectral"],
            c["v4w_v3"], cl["v4w_v3"], c["v4a_v3"], cl["v4a_v3"], S["sector1"]["v4_weyl"]["transported"], S["sector2"]["v4_weyl"]["transported"],
            S["sector1"]["v4_angle"]["transported"], S["sector2"]["v4_angle"]["transported"],
            max(S["sector1"]["v4_weyl"]["max_transport_residual"], S["sector2"]["v4_weyl"]["max_transport_residual"]),
            max(S["sector1"]["v4_angle"]["max_transport_residual"], S["sector2"]["v4_angle"]["max_transport_residual"]),
            S["sector1"]["v3"]["unverified"], S["sector2"]["v3"]["unverified"], S["sector1"]["v4_weyl"]["unverified"], S["sector2"]["v4_weyl"]["unverified"]))
    L += ["", "### At the exact zero (the record at s*) and the next record: the occupied line in the lowest pair's plane", "",
          "| n_rec | tau | sector | rule | weight in the plane at s* | neutrality defect at s* | weight at the next record | neutrality defect at the next record |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if not r["inserted_s_star"]:
            continue
        for sec, dd in sorted(r["neutrality_at_crossing"].items()):
            for rule in ("v3", "v4_weyl", "v4_angle"):
                x = dd["at_s_star"][rule]; y = dd["next_record"][rule] or {}
                L.append("| %d | %g | %s | %s | %.12f | %.3e | %s | %s |" % (r["n_rec"], r["tau"], sec, rule, x["weight_in_plane"], x["neutrality_defect"],
                         ("%.12f" % y["weight_in_plane"]) if y else "-", ("%.3e" % y["neutrality_defect"]) if y else "-"))
    tr = [q for r in rows for sec in ("sector1",) for q in r["stats"][sec]["v4_weyl"]["pairs"]][:12]
    L += ["", "### The first transported pair-steps (v4 weyl, sector 1, over the runs in order)", "", "| s | sub-step | eps | sep | eta | smallest principal cosine | det O | signed energy Tr(hP) | occupied | transport residual |", "|---|---|---|---|---|---|---|---|---|---|"]
    for q in tr:
        L.append("| %.6f | %d | %.3e | %.3e | %.3e | %.12f | %+.0f | %+.6e | %s | %.2e |" % (q["s"], q["substep"], q["eps"], q["sep"], q["eta"], q["cos_min"], q["det_O"], q["signed_energy"], q["occupied"], q["transport_residual"]))
    L += ["", "- every number above is printed by the script (%.0f s)." % (time.time() - t0)]
    open(a.md, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
