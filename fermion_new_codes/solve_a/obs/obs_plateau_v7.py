#!/usr/bin/env python3
"""obs_plateau_v7.py (builder analyst; PL-ext per SPEC_final4b_observables.md sec. 13.1; version 7 of obs_plateau_v6.py).

As version 6 (the slice's dense functions re-solved from the stored shell totals, the blended profile rho(x) = R_cut + w log(1 + e^((x - R_cut)/w)),
its surface R_h and kappa_T by Richardson-combined differences of the blended N + v, the full scattering and the Hawking reading) plus 13.1:
(a) the lapse mapping: the reading uses v/N's jets in z = kappa' (R - R_h) with kappa' = (v/N)' at R_h (= kappa_T/N_t), the measured
    c = (1 - theta)/kappa'^2 with theta = T_conv/(kappa_T/(2 pi)) (Killing units), P25's c at v/N's jets with r = omega/kappa_T and
    mu' = N_t mu; v's jets in kappa_T printed beside, labelled;
(b) the lapse conditions: |N_t - 1| < 1e-3 and max |N(R) - N_t|/N_t < 1e-3 over |R - R_h| <= 2 k_s^2/kappa', k_s = (2 omega/N_t)^(1/3)
    (16 lattice units at r = 1, kappa' a = 1/128); a row failing them gets no P25 comparison ("lapse affected");
(c) the gate per row: (1) the mixing on the slice (obs_plateau_mix_v5.region_mixing at the row's geometry, omega and m; the outer junction at
    --R-b r_s with width --w-b), the contamination |t|/((kappa' a)^2 |c| pi r |beta| (1 - p)), p = beta^2/Gamma, acceptance 0.02;
    (2) the import: the slice re-solved at rtol 1e-10 (against the run's 1e-12) and with the collar moved by 2 ell (the chain padded with
    zero sources), the coefficient's largest relative change, acceptance 0.005; the slice's kappa_T by B13's two routes (--sj-json, the
    fits' and the constraints' kappa) agreeing to 1e-7 relative; (3) the jets: two Richardson estimates of A, B, C of v/N (h = 1/4, 1/8 and
    1/8, 1/16), their differences propagated through P25 (the sum of the absolute partial changes), acceptance 0.005 of |c|; (4) the lapse
    conditions; (5) max |N - 1| and the sources' largest |rho_ell|, |J_ell| over [R_cut, R_h] relative to their maxima printed;
(d) the budget per row: mixing + import + jets + the envelope (--envelope, the exact ladder's observed discrepancy, labelled a calibration),
    printed as the complete sum or "incomplete" when a term is missing, never a partial sum; the gate's pass per term.
Args: --fields --records --slice --g --Rcut-sites --wcut --R-lead --r --mu --R-b --w-b --envelope --collar-policy --sj-json --no-import
--out-json --md."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
import argparse
import json
import math
import sys

import numpy as np
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "engine", "code"))
sys.path.insert(0, os.path.join(HERE, "..", "engine"))
import obs_plateau_jets_v1 as JT  # noqa: E402
import obs_plateau_mix_v5 as MX  # noqa: E402
import obs_plateau_v4 as P4  # noqa: E402


def d14(F, x, h):
    f = {q: F(x + q * h) for q in (-2, -1, 0, 1, 2)}
    return np.array([(f[1] - f[-1]) / (2 * h), (f[1] - 2 * f[0] + f[-1]) / h ** 2, (f[2] - 2 * f[1] + 2 * f[-1] - f[-2]) / (2 * h ** 3),
                     (f[2] - 4 * f[1] + 6 * f[0] - 4 * f[-1] + f[-2]) / h ** 4])


def rich(F, x, h):
    a, b = d14(F, x, h), d14(F, x, h / 2)
    return (4.0 * b - a) / 3.0


def dense_fn(F8, eps, J, g, R, ell, N0, collar_policy, rtol=1e-12):
    Nr, vr, Wr, info = F8.solve_constraints(eps, J, g, R, ell, N0=N0, collar_policy=collar_policy, rtol=rtol)
    if info.get("failed"):
        raise SystemExit("the re-solve failed: %s" % info["failed"])
    return (lambda rho: (info["N_dense"](np.asarray(rho, float)), info["v_dense"](np.asarray(rho, float)))), info


def geometry(fn, Rc, w, rs):
    rho_of = lambda x: Rc + w * np.logaddexp(0.0, (np.asarray(x, float) - Rc) / w)
    Nf = lambda x: float(fn(rho_of(np.array([x])))[0][0])
    vf = lambda x: float(fn(rho_of(np.array([x])))[1][0])
    f_bl = lambda x: Nf(x) + vf(x)
    q_bl = lambda x: vf(x) / Nf(x)
    xs = np.arange(Rc + 1.0, 3 * rs, 0.25)
    fv = np.array([f_bl(x) for x in xs])
    zz = np.nonzero(fv[:-1] * fv[1:] <= 0)[0]
    Rh = brentq(f_bl, xs[zz[-1]], xs[zz[-1] + 1], xtol=1e-13)
    kT = float((4 * (f_bl(Rh + 5e-3) - f_bl(Rh - 5e-3)) / 1e-2 - (f_bl(Rh + 1e-2) - f_bl(Rh - 1e-2)) / 2e-2) / 3.0)
    dq1, dq2 = rich(q_bl, Rh, 0.25), rich(q_bl, Rh, 0.125)
    dv1 = rich(vf, Rh, 0.25)
    kp1, kp2 = float(dq1[0]), float(dq2[0])
    jets = lambda d, k: [d[1] / (2 * k ** 2), d[2] / (6 * k ** 3), d[3] / (24 * k ** 4)]
    return {"R_h": Rh, "kappa_T": kT, "kappa_prime": kp1, "N_t": Nf(Rh), "jq": jets(dq1, kp1), "jq2": jets(dq2, kp2), "jv": jets(dv1, kT),
            "Nf": Nf, "rho_of": rho_of}


def reading(fn, geo, Rc, w, RL, rs, r, mu):
    Rch, vch, Nch, gi = P4.blended_chain(fn, rs, Rc, w, RL * rs, None, geo["kappa_T"])
    om, m = r * geo["kappa_T"], mu * geo["kappa_T"]
    sc = P4.scatter(vch, Nch, m, 0.0, om)
    rd = P4.hawking_reading(sc, om, geo["kappa_T"] / (2 * math.pi))
    S = sc["S"]
    full = None
    if S.shape[0] == S.shape[1]:
        I = np.eye(S.shape[0])
        full = (float(np.max(np.abs(S.conj().T @ S - I))), float(np.max(np.abs(S @ S.conj().T - I))))
    th = rd.get("theta")
    c = (1.0 - th) / geo["kappa_prime"] ** 2 if th is not None else None
    return {"theta": th, "c": c, "beta2": rd.get("beta2"), "Gamma": rd.get("Gamma"), "unitarity_full": full, "omega": om, "m": m}


def p25(j, r, mu_p):
    c0, c2, cm = JT.p25_coeffs(*j)
    return c0 + c2 * r * r + cm * mu_p * mu_p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fields", required=True)
    ap.add_argument("--records", required=True)
    ap.add_argument("--slice", type=int, default=-1)
    ap.add_argument("--g", type=float, default=0.25)
    ap.add_argument("--Rcut-sites", type=float, nargs="+", default=[12.0, 16.0])
    ap.add_argument("--wcut", type=float, nargs="+", default=[4.0, 6.0])
    ap.add_argument("--R-lead", type=float, nargs="+", default=[32.0], help="units of the solved 2 g E")
    ap.add_argument("--r", type=float, nargs="+", default=[0.25, 0.5, 1.0])
    ap.add_argument("--mu", type=float, nargs="+", default=[0.0, 0.2])
    ap.add_argument("--R-b", type=float, default=0.5, help="the mixing region's outer junction, units of 2 g E (inside the surface)")
    ap.add_argument("--w-b", type=float, default=8.0)
    ap.add_argument("--envelope", type=float, default=0.007)
    ap.add_argument("--collar-policy", default="flag")
    ap.add_argument("--sj-json", default=None, help="obs_slice_jets_v3's JSON for this slice (B13's two kappa routes)")
    ap.add_argument("--no-import", action="store_true")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--md", required=True)
    a = ap.parse_args()
    import engine_fields_v8 as F8
    Z = np.load(a.fields, allow_pickle=True)
    idx = a.slice if a.slice >= 0 else len(Z["T"]) + a.slice
    T = float(Z["T"][idx])
    Rs = np.asarray(Z["R"], float); ell = int(Z["ell"])
    eps = np.asarray(Z["eps_tot"][idx], float); J = np.asarray(Z["J_tot"][idx], float); N0 = np.asarray(Z["N"][idx], float)
    fn, info = dense_fn(F8, eps, J, a.g, Rs, ell, N0, a.collar_policy)
    E = float(info["E_collar"]); rs = 2 * a.g * E
    variants = {}
    if not a.no_import:
        variants["rtol 1e-10"] = dense_fn(F8, eps, J, a.g, Rs, ell, N0, a.collar_policy, rtol=1e-10)[0]
        npad = 2 * ell
        Rp = np.concatenate([Rs, Rs[-1] + np.arange(1, npad + 1, dtype=float)])
        variants["collar + 2 ell"] = dense_fn(F8, np.concatenate([eps, np.zeros(npad)]), np.concatenate([J, np.zeros(npad)]), a.g, Rp, ell,
                                              np.concatenate([N0, np.ones(npad)]), a.collar_policy)[0]
    kappa_routes = None
    if a.sj_json:
        S_ = json.load(open(a.sj_json))
        row = [q for q in S_["rows"] if q.get("slice") in (idx, a.slice)] or S_["rows"]
        kf = row[0].get("kappa_fits"); kc = (row[0].get("constraint_route") or {}).get("kappa")
        kappa_routes = {"fits": kf, "constraints": kc, "relative": (abs(kf - kc) / abs(kc)) if (kf and kc) else None}
    rows = []
    for RL in a.R_lead:
        for Rc in a.Rcut_sites:
            for w in a.wcut:
                geo = geometry(fn, Rc, w, rs)
                geo_v = {k: geometry(fv, Rc, w, rs) for k, fv in variants.items()}
                # (5) the window's lapse and sources over [R_cut, R_h]
                rr = np.linspace(Rc, geo["R_h"], 400)
                Nw = info["N_dense"](rr)
                rho_w = np.abs(info["rho_ell"](rr)); J_w = np.abs(info["J_ell"](rr))
                rmax = max(float(np.max(np.abs(info["rho_ell"](np.linspace(0, info["r_collar"], 4001))))), 1e-300)
                jmax = max(float(np.max(np.abs(info["J_ell"](np.linspace(0, info["r_collar"], 4001))))), 1e-300)
                win = {"max_N_minus_1": float(np.max(np.abs(Nw - 1.0))), "rho_rel_max": float(rho_w.max()) / rmax, "J_rel_max": float(J_w.max()) / jmax}
                for mu in a.mu:
                    for r in a.r:
                        if r <= mu:
                            continue
                        rd = reading(fn, geo, Rc, w, RL, rs, r, mu)
                        Nt = geo["N_t"]; kp = geo["kappa_prime"]; om = rd["omega"]
                        ks = (2.0 * om / Nt) ** (1.0 / 3.0)
                        X = 2.0 * ks * ks / kp
                        xs = np.linspace(geo["R_h"] - X, geo["R_h"] + X, 201)
                        Nx = np.array([geo["Nf"](x) for x in xs])
                        lapse = {"N_t_minus_1": Nt - 1.0, "max_rel_N_var": float(np.max(np.abs(Nx - Nt))) / Nt, "half_width": X}
                        lapse_ok = abs(Nt - 1.0) < 1e-3 and lapse["max_rel_N_var"] < 1e-3
                        mu_p = Nt * mu
                        cP = p25(geo["jq"], r, mu_p)
                        cPv = p25(geo["jv"], r, mu)
                        c = rd["c"]
                        read = ((c - cP) / abs(cP)) if (c is not None and lapse_ok) else None
                        # (1) the mixing on the slice
                        mxr = MX.region_mixing(fn, Rc, a.R_b * rs, w, a.w_b, om, rd["m"])
                        p_ = (rd["beta2"] / rd["Gamma"]) if (rd.get("beta2") and rd.get("Gamma")) else None
                        mix_term = MX.contamination(mxr["branch_to_longwave_max"], kp, abs(c) if c else 0.0, r, p_) if (c and p_) else None
                        # (2) the import: the variants' coefficient change
                        imp = None
                        if variants and c:
                            ch = []
                            for k_, gv in geo_v.items():
                                rv = reading(variants[k_], gv, Rc, w, RL, rs, r, mu)
                                if rv["c"] is not None:
                                    ch.append(abs(rv["c"] - c) / abs(c))
                            imp = max(ch) if ch else None
                        # (3) the jets: the two Richardson estimates' differences propagated through P25
                        dj = [abs(x - y) for x, y in zip(geo["jq"], geo["jq2"])]
                        jet_term = 0.0
                        for i_ in range(3):
                            jp = list(geo["jq"]); jp[i_] += dj[i_]
                            jet_term += abs(p25(jp, r, mu_p) - cP)
                        jet_term = jet_term / abs(c) if c else None
                        terms = {"mixing": mix_term, "import": imp, "jets": jet_term, "envelope": a.envelope}
                        complete = all(v_ is not None for v_ in terms.values())
                        budget = sum(terms.values()) if complete else "incomplete"
                        gate = {"mixing": (mix_term is not None and mix_term < 0.02), "import": (imp is not None and imp < 0.005),
                                "kappa_routes": (kappa_routes is not None and kappa_routes["relative"] is not None and kappa_routes["relative"] < 1e-7),
                                "jets": (jet_term is not None and jet_term < 0.005), "lapse": lapse_ok}
                        q = {"R_lead_over_rs": RL, "R_cut": Rc, "w": w, "r": r, "mu": mu, "mu_prime": mu_p, "R_h": geo["R_h"], "kappa_T": geo["kappa_T"],
                             "kappa_prime": kp, "N_t": Nt, "theta": rd["theta"], "c": c, "c_P25_vN": cP, "c_P25_v_labelled": cPv, "reading": read,
                             "label": None if lapse_ok else "lapse affected", "jets_v_over_N": geo["jq"], "jets_v_over_N_second": geo["jq2"], "jets_v": geo["jv"],
                             "lapse": lapse, "window": win, "mixing": mxr, "p": p_, "terms": terms, "budget": budget, "gate": gate,
                             "kappa_routes": kappa_routes, "unitarity_full": rd["unitarity_full"], "beta2": rd["beta2"], "Gamma": rd["Gamma"]}
                        rows.append(q)
                        print("R_cut %g w %g r %g mu %g: theta %s c %s P25(v/N) %.5f reading %s budget %s gate %s" % (
                            Rc, w, r, mu, ("%.9f" % rd["theta"]) if rd["theta"] else None, ("%.5f" % c) if c is not None else None, cP,
                            ("%+.4f" % read) if read is not None else q["label"], ("%.4f" % budget) if complete else budget, {k: v_ for k, v_ in gate.items()}), flush=True)
                        json.dump({"script": "obs_plateau_v7.py", "args": vars(a), "slice": idx, "T": T, "E": E, "rows": rows}, open(a.out_json, "w"), indent=1, default=float)
    L = ["## PL-ext on the solved exterior (obs_plateau_v7.py, observables 13.1): slice %d (T/r_s %.3f, E %.6f, 2gE %.4f); the engine's dense functions re-solved" % (idx, T / rs, E, rs), "",
         "B13's two kappa routes: %s." % (("fits %.12e, constraints %.12e, relative %.1e" % (kappa_routes["fits"], kappa_routes["constraints"], kappa_routes["relative"])) if (kappa_routes and kappa_routes["relative"] is not None) else "not supplied"), "",
         "| R_cut | w | r | mu (mu') | R_h | N_t | kappa' 4gE | A, B, C of v/N (in kappa') | theta | c | P25 c at v/N's jets | reading | lapse (N_t - 1; max rel var; half width) | mixing; import; jets; envelope | budget | gate (mixing, import, kappa routes, jets, lapse) |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    f_ = lambda x, fmt="%.2e": (fmt % x) if isinstance(x, (int, float)) and x is not None else "-"
    for q in rows:
        g_ = q["gate"]
        L.append("| %g | %g | %g | %g (%.4f) | %.4f | %.6f | %.6f | %s | %s | %s | %.5f | %s | %.1e; %.1e; %.1f | %s; %s; %s; %s | %s | %s |" % (
            q["R_cut"], q["w"], q["r"], q["mu"], q["mu_prime"], q["R_h"], q["N_t"], q["kappa_prime"] * 4 * a.g * E, ", ".join("%.5f" % x for x in q["jets_v_over_N"]),
            ("%.9f" % q["theta"]) if q["theta"] else "-", ("%.5f" % q["c"]) if q["c"] is not None else "-", q["c_P25_vN"],
            ("%+.4f" % q["reading"]) if q["reading"] is not None else (q["label"] or "-"), q["lapse"]["N_t_minus_1"], q["lapse"]["max_rel_N_var"], q["lapse"]["half_width"],
            f_(q["terms"]["mixing"]), f_(q["terms"]["import"]), f_(q["terms"]["jets"]), f_(q["terms"]["envelope"], "%.3f"),
            ("%.4f" % q["budget"]) if not isinstance(q["budget"], str) else q["budget"], ", ".join({True: "pass", False: "FAIL"}[g_[k]] for k in ("mixing", "import", "kappa_routes", "jets", "lapse"))))
    L += ["", "- every number above is printed by the script; the reading is (c measured - P25 at v/N's jets in kappa')/|P25| with c = (1 - theta)/kappa'^2; the budget is",
          "  the complete additive sum of the four terms or \"incomplete\"; the envelope is the exact ladder's observed discrepancy (a calibration)."]
    open(a.md, "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
