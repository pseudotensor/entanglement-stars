#!/usr/bin/env python
"""Strand 6 (j), table J4: the constraint operator's variance in the cube's (2,0) states (SPEC_strand6_validation.md item (j) (4) with the lead's
amendment of JUDGMENT_strand6_validation.md sec. 11; builder analyst).  On the (2,0) block of the light packet on the open cube (strand2b_c8_lib's
BlockModel: |I> = d_a^dag d_b^dag |Omega>, 120 states, the sources e_x = rho^P_x as M_x = e^S H0pp(X_x) e^S): the dressed sources preserve the
dressed particle and hole numbers and S acts within the two-particle sector, so the block is invariant under every e_x and Var(e[f]) =
Tr(rho E_f^2) - Tr(rho E_f)^2 with E_f = sum_x f_x M_x is exact on the block (rho = C C^dag/d0, C the lowest d0 levels of H = sum N_y M_y at the
case's psi and N).  N_f = Tr(rho H0pp(Y_f)) with Y_x = R[w_x]^dag R[w_x] (the particle site densities in the A2 frame: d_w = sum_r R_wr d_r, R the
eigenvectors of B_0 at N = 1, w over the site's two components; no holes in the block), and beside it the site frame's normal-ordered c-charge
Tr(rho H0pp(Z_f)), Z_x = vp[j_x]^dag vp[j_x] over the site's four c-modes (the hole part of :c^dag c: vanishes on the block).  Per window: sigma(e[f])/<e[f]>,
N_f, eps_f = <e[f]>/N_f, F_E,f = Var(e[f])/(eps_f^2 N_f), sqrt(F_E,f/N_f); windows: f = all 8 sites, every single site (a corner), every edge (the
12 nearest-neighbour pairs), every face (the 6 four-site faces): per family the smallest and largest value over unmasked windows.  Cases: the uniform continuation
solution at k = 1 and k = 1e4 (solve/results/strand2b/c8encl_v7) and the two non-uniform classes at k = 1 (strand2b/sec15/item5.json part_a,
the k = 1 entries), at m = 0.1, (g_M, g_B, Lambda) = (0.2, 1, 1); the free state (g_M = g_B = 0) of the same block at the same psi, N beside each.
Control: <e_x> against the case record's source where it has one; sum_x Y_x = 2 on the block.  Writes --md and --out-json.
Version 2 (JUDGMENT_strand6_validation.md sec. 12's reprint): a window entry whose mean source lies at the rounding floor (|<e[f]>| below 1e-12 of
the largest site mean) is printed as "-" (its sigma/<e> is not a number); the window means and variances are printed for f = all.
Version 3: the floor defined before its first use (version 2 stopped there).
Version 4 (JUDGMENT_strand6_validation.md sec. 13, the hand-back's item 3): the caption reads 'per family the smallest and largest value over
unmasked windows'; nothing else changes."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import itertools
import json
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "strand2"))
sys.path.insert(0, os.path.join(HERE, "..", "strand2", "frozen"))
import strand2a_lib as S2  # noqa: E402
import strand2b_c8_lib_v6 as C8  # noqa: E402


def ev(rho, O):
    return complex(np.trace(rho @ O)).real


def windows(lat):
    pos = np.asarray(lat["positions"], float)
    n = len(pos)
    out = {"all": [np.ones(n)]}
    out["corner"] = [np.eye(n)[x] for x in range(n)]
    out["edge"] = []
    for x, y in itertools.combinations(range(n), 2):
        if abs(np.linalg.norm(pos[x] - pos[y]) - 1.0) < 1e-9:
            f = np.zeros(n)
            f[[x, y]] = 1
            out["edge"].append(f)
    out["face"] = []
    for ax in range(3):
        for v in sorted(set(pos[:, ax].tolist())):
            sel = np.nonzero(np.abs(pos[:, ax] - v) < 1e-9)[0]
            if len(sel) == 4:
                f = np.zeros(n)
                f[sel] = 1
                out["face"].append(f)
    return out


def analyse(bm, psi, N, d0_fixed=None):
    sy = C8.SystemC8v6(bm, np.eye(bm.n), 1.0)
    sy.d0_fixed = d0_fixed
    st = sy.state(psi, N)
    b = st["b"]
    M = b["M"]
    C = st["C"]
    rho = C @ C.conj().T / st["d0"]
    light = b["light"]
    R = light.R
    ncp = bm.lat["ncomp"]
    Y, Z = [], []
    for x in range(bm.n):
        w = list(range(x * ncp, (x + 1) * ncp))
        Y.append(bm.H0pp(R[w, :].conj().T @ R[w, :]))
        js = [2 * ww + s for ww in w for s in (0, 1)]
        vp = light.vp
        Z.append(bm.H0pp(vp[js, :].conj().T @ vp[js, :]))
    Y, Z = np.array(Y), np.array(Z)
    ctrl = float(np.max(np.abs(Y.sum(0) - 2 * np.eye(Y.shape[1]))))
    means = np.array([ev(rho, M[x]) for x in range(bm.n)])
    return {"st": st, "M": M, "rho": rho, "Y": Y, "Z": Z, "control_sumY_minus_2": ctrl, "site_means": means}


def window_stats(A, wins):
    out = {}
    floor = 1e-12 * float(np.max(np.abs(A["site_means"])))
    for fam, fl in wins.items():
        vals = []
        for f in fl:
            E = np.tensordot(f, A["M"], axes=(0, 0))
            m = ev(A["rho"], E)
            v = ev(A["rho"], E @ E) - m * m
            Nf = ev(A["rho"], np.tensordot(f, A["Y"], axes=(0, 0)))
            Qf = ev(A["rho"], np.tensordot(f, A["Z"], axes=(0, 0)))
            eps = m / Nf if (Nf and abs(m) > floor) else None
            F = v / (eps * eps * Nf) if (eps and Nf) else None
            vals.append({"mean": m, "var": v, "sigma_over_mean": float(np.sqrt(max(v, 0)) / m) if abs(m) > floor else None, "N_f": Nf, "bare_charge": Qf, "F_E": F,
                         "sqrt_F_over_N": float(np.sqrt(F / Nf)) if (F is not None and F >= 0 and Nf > 0) else None})
        out[fam] = vals
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encl", required=True, help="comma list of the uniform cases (c8encl_v7 case JSONs)")
    ap.add_argument("--classes", required=True, help="strand2b/sec15/item5.json")
    ap.add_argument("--md", required=True)
    ap.add_argument("--out-json", required=True)
    a = ap.parse_args()
    lat = S2.lattice("C8")
    wins = windows(lat)
    cases = []
    for p in a.encl.split(","):
        R = json.load(open(p))
        cases.append({"label": "uniform k = %g" % R["k"], "m": R["m"], "gM": R["gM"], "gB": R["gB"], "Lambda": R["Lambda"], "psi": R["psi"], "N": R["N"],
                      "d0": R["d0"], "source": R.get("source")})
    I5 = json.load(open(a.classes))["part_a"]
    for cl in ("class 1", "class 2"):
        e = [x for x in I5[cl] if abs(x["k"] - 1.0) < 1e-12][0]
        cases.append({"label": "%s k = 1" % cl, "m": 0.1, "gM": 0.2, "gB": 1.0, "Lambda": 1.0, "psi": e["psi"], "N": e["N"], "d0": e.get("d0_at_tolerance", 1),
                      "source": e.get("source")})
    res = []
    for c in cases:
        Lam = np.inf if c["Lambda"] == "inf" else float(c["Lambda"])
        bm = C8.BlockModel(lat, c["m"], c["gM"], c["gB"], Lam, (2, 0))
        A = analyse(bm, np.array(c["psi"]), np.array(c["N"]), int(c["d0"]))
        bm0 = C8.BlockModel(lat, c["m"], 0.0, 0.0, Lam, (2, 0))
        A0 = analyse(bm0, np.array(c["psi"]), np.array(c["N"]), None)
        src_ctrl = float(np.max(np.abs(A["site_means"] / np.array(c["source"]) - 1))) if c.get("source") else None
        r = {"label": c["label"], "d0": int(A["st"]["d0"]), "d0_free": int(A0["st"]["d0"]), "control_sumY": A["control_sumY_minus_2"], "control_source": src_ctrl,
             "state": window_stats(A, wins), "free": window_stats(A0, wins), "site_means": A["site_means"].tolist()}
        res.append(r)
        print("%s: d0 %d; F_E(all) %s; source control %s" % (c["label"], r["d0"], r["state"]["all"][0]["F_E"], src_ctrl), flush=True)
    json.dump({"meta": {"script": "strand6j_cube_var_v4.py", "builder": "analyst", "windows": {k: len(v) for k, v in wins.items()}}, "cases": res},
              open(a.out_json, "w"), indent=1, default=float)

    def f(x, d=4):
        if x is None:
            return "-"
        x = float(x)
        return ("%." + str(d) + "g") % x if (x == 0 or 1e-3 <= abs(x) < 1e5) else "%.3e" % x
    L = ["# Strand 6 (j), table J4: the cube's (2,0) states (strand6j_cube_var_v4.py)", "",
         "Windows: all 8 sites; %d corners; %d edges; %d faces (per family the smallest and largest value over unmasked windows)." % (len(wins["corner"]), len(wins["edge"]), len(wins["face"])), "",
         "| case | state | d0 | f = all: mean, var, sigma/<e>, N_f, Q, F_E, sqrt(F/N) | corner F_E (min, max) | edge F_E | face F_E | corner sigma/<e> | face sqrt(F/N) |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in res:
        for lab in ("state", "free"):
            w = r[lab]
            al = w["all"][0]

            def rng(fam, key):
                v = [x[key] for x in w[fam] if x[key] is not None]
                return "%s, %s" % (f(min(v)), f(max(v))) if v else "-"
            L.append("| %s | %s | %d | %s, %s, %s, %s, %s, %s, %s | %s | %s | %s | %s | %s |" % (
                r["label"], lab, r["d0"] if lab == "state" else r["d0_free"], f(al["mean"]), f(al["var"]), f(al["sigma_over_mean"]), f(al["N_f"]), f(al["bare_charge"]), f(al["F_E"]),
                f(al["sqrt_F_over_N"]), rng("corner", "F_E"), rng("edge", "F_E"), rng("face", "F_E"), rng("corner", "sigma_over_mean"), rng("face", "sqrt_F_over_N")))
    L += ["", "Controls: " + "; ".join("%s: sum_x Y_x - 2 = %s, <e_x>/source - 1 = %s" % (r["label"], f(r["control_sumY"], 2), f(r["control_source"], 2)) for r in res), ""]
    open(a.md, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
