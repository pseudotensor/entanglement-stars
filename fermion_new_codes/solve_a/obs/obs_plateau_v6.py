#!/usr/bin/env python3
"""obs_plateau_v6.py (builder analyst; B11 of SPEC_final4b_observables.md sec. 12.6: PL-ext per sec. 12.2; version 6 of obs_plateau_v5.py).

Version 6: (1) the slice's fields are the engine's dense functions re-solved from the stored shell totals (engine_collapse_v6's export;
engine_fields_v5.resolve_dense with the run's collar policy): no spline anywhere, no off-site interpolation (the import's budget term is 0);
(2) per geometry (R_cut, w, R_lead) the blended profile rho(x) = R_cut + w log(1 + e^((x - R_cut)/w)) composed with the dense functions, its
surface R_h (N + v = 0) refined by brentq and its kappa by Richardson-combined central differences of the blended N + v (h = 1e-2, 5e-3; never
the outer saturation's unstable form: no outer saturation is used); (3) the blended-solved profile's own jets at R_h (Richardson central
differences to the fourth order, h = 1/4 and 1/8) for v and for v/N, and P25's coefficients at them; (4) per row (r, mu): theta = T_conv/T_H
with T_H = kappa/(2 pi), c = (1 - theta)/kappa^2, P25's c at the blended-solved jets, the reading (c - c_P25)/|c_P25|, the full unitarity
max |S^dagger S - 1| and max |S S^dagger - 1|, and the budget of sec. 12.2: the junction's mixing (from the gate run's JSON, --mix-json:
obs_plateau_mix_v4 on the same slice, the branch-to-long-wave |t| at the row's R_cut, w and r turned into the coefficient contamination
2 |t|/(2 pi r |beta|)/((kappa a)^2 |c_P25|) with |beta|^2 from the row's own scattering) + the import (0) + the exact ladder's residual
envelope (--envelope, 0.007 at kappa a = 1/128 per observables 12.1); the label "solved exterior (quasi-static, tau, ell), control's interior
below R_cut; the engine's dense functions (re-solved)". Args: --fields --records --slice --g --Rcut-sites --wcut --R-lead --r --mu
--mix-json --envelope --collar-policy --out-json --md."""
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
import obs_plateau_v4 as P4  # noqa: E402


def d14(F, x, h):
    f = {q: F(x + q * h) for q in (-2, -1, 0, 1, 2)}
    return np.array([(f[1] - f[-1]) / (2 * h), (f[1] - 2 * f[0] + f[-1]) / h ** 2, (f[2] - 2 * f[1] + 2 * f[-1] - f[-2]) / (2 * h ** 3),
                     (f[2] - 4 * f[1] + 6 * f[0] - 4 * f[-1] + f[-2]) / h ** 4])


def rich(F, x, h=0.25):
    a, b = d14(F, x, h), d14(F, x, h / 2)
    return (4.0 * b - a) / 3.0


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
    ap.add_argument("--mix-json", default=None)
    ap.add_argument("--envelope", type=float, default=0.007)
    ap.add_argument("--collar-policy", default="flag")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--md", required=True)
    a = ap.parse_args()
    import engine_fields_v5 as F5
    Z = np.load(a.fields, allow_pickle=True)
    recs = [r for r in json.load(open(a.records))["records"] if "E" in r]
    idx = a.slice if a.slice >= 0 else len(Z["T"]) + a.slice
    T = float(Z["T"][idx])
    rec = min(recs, key=lambda q: abs(q["T"] - T))
    Rs = np.asarray(Z["R"], float)
    ell = int(Z["ell"])
    N0 = np.asarray(Z["N"][idx], float)
    Nr, vr, Wr, info = F5.resolve_dense(np.asarray(Z["eps_tot"][idx], float), np.asarray(Z["J_tot"][idx], float), a.g, Rs, ell, N0=N0, collar_policy=a.collar_policy)
    if info.get("failed"):
        raise SystemExit("the re-solve failed: %s" % info["failed"])
    E = float(info["E_collar"]); rs = 2 * a.g * E
    fn = lambda rho: (info["N_dense"](np.asarray(rho, float)), info["v_dense"](np.asarray(rho, float)))
    resolve_dv = float(np.max(np.abs(vr - np.asarray(Z["v"][idx], float))))
    mix = json.load(open(a.mix_json))["rows"] if a.mix_json else []
    label = "solved exterior (quasi-static, tau = %g, ell = %d), control's interior below R_cut; the engine's dense functions (re-solved)" % (float(Z["tau"]), ell)
    rows = []
    for RL in a.R_lead:
        for Rc in a.Rcut_sites:
            for w in a.wcut:
                rho_of = lambda x: Rc + w * np.logaddexp(0.0, (np.asarray(x, float) - Rc) / w)
                f_bl = lambda x: float(np.sum(fn(rho_of(np.array([x])))[0] + fn(rho_of(np.array([x])))[1]))
                v_bl = lambda x: float(fn(rho_of(np.array([x])))[1][0])
                q_bl = lambda x: float(fn(rho_of(np.array([x])))[1][0] / fn(rho_of(np.array([x])))[0][0])
                xs = np.arange(Rc + 1.0, 3 * rs, 0.25)
                fv = np.array([f_bl(x) for x in xs])
                zz = np.nonzero(fv[:-1] * fv[1:] <= 0)[0]
                Rh = brentq(f_bl, xs[zz[-1]], xs[zz[-1] + 1], xtol=1e-13)
                kap = float((4 * (f_bl(Rh + 5e-3) - f_bl(Rh - 5e-3)) / 1e-2 - (f_bl(Rh + 1e-2) - f_bl(Rh - 1e-2)) / 2e-2) / 3.0)
                dv = rich(v_bl, Rh); dq = rich(q_bl, Rh)
                jv = [dv[1] / (2 * kap ** 2), dv[2] / (6 * kap ** 3), dv[3] / (24 * kap ** 4)]
                jq = [dq[1] / (2 * kap ** 2), dq[2] / (6 * kap ** 3), dq[3] / (24 * kap ** 4)]
                cv = JT.p25_coeffs(*jv); cq = JT.p25_coeffs(*jq)
                Rch, vch, Nch, gi = P4.blended_chain(fn, rs, Rc, w, RL * rs, None, kap)
                for mu in a.mu:
                    for r in a.r:
                        if r <= mu:
                            continue
                        om, m = r * kap, mu * kap
                        sc = P4.scatter(vch, Nch, m, 0.0, om)
                        S = sc["S"]
                        full = None
                        if S.shape[0] == S.shape[1]:
                            I = np.eye(S.shape[0])
                            full = (float(np.max(np.abs(S.conj().T @ S - I))), float(np.max(np.abs(S @ S.conj().T - I))))
                        rd = P4.hawking_reading(sc, om, kap / (2 * math.pi))
                        th = rd.get("theta")
                        cm = (1.0 - th) / kap ** 2 if th is not None else None
                        cp = cv[0] + cv[1] * r * r + cv[2] * mu * mu
                        cpq = cq[0] + cq[1] * r * r + cq[2] * mu * mu
                        mixrow = [q for q in mix if abs(q.get("R_a", -1) - Rc) < 1e-9 and abs(q.get("w_a", -1) - w) < 1e-9 and abs(q["r"] - r) < 1e-9]
                        cont = None
                        if mixrow and rd.get("beta2"):
                            t = mixrow[0]["branch_to_longwave_max"]
                            cont = 2 * t / (2 * math.pi * r * math.sqrt(rd["beta2"])) / (kap ** 2 * abs(cp))
                        budget = (cont or 0.0) + 0.0 + a.envelope
                        rows.append({"R_lead_over_rs": RL, "R_cut": Rc, "w": w, "R_h": Rh, "kappa": kap, "kappa_4gE": kap * 4 * a.g * E, "r": r, "mu": mu,
                                     "theta": th, "c": cm, "jets_v": jv, "jets_v_over_N": jq, "c_P25_v": cp, "c_P25_vN": cpq,
                                     "reading": ((cm - cp) / abs(cp)) if cm is not None else None, "mixing_contamination": cont, "budget": budget,
                                     "unitarity_columns": sc["unitarity"], "unitarity_full": full, "Gamma": rd.get("Gamma"), "beta2": rd.get("beta2")})
                        print("R_cut %g w %g r %g mu %g: theta %s c %s P25(blended jets) %.5f reading %s budget %.4f unitarity %s" % (
                            Rc, w, r, mu, ("%.9f" % th) if th else None, ("%.5f" % cm) if cm is not None else None, cp,
                            ("%+.4f" % rows[-1]["reading"]) if rows[-1]["reading"] is not None else None, budget, full), flush=True)
                        json.dump({"script": "obs_plateau_v6.py", "args": vars(a), "label": label, "slice": idx, "T": T, "E": E, "rows": rows},
                                  open(a.out_json, "w"), indent=1, default=float)
    L = ["## PL-ext on the solved exterior (obs_plateau_v6.py): slice %d (T/r_s %.3f, E %.6f, 2gE %.4f); %s" % (idx, T / rs, E, rs, label), "",
         "The re-solve's largest |v| difference from the stored site values: %.2e; the solver's residuals %s, %s; the collar margin %s." % (
             resolve_dv, info.get("residual_W"), info.get("residual_invN"), info.get("collar_margin")), "",
         "| R_lead/r_s | R_cut | w | R_h | kappa 4 g E | A, B, C of v (blended-solved) | r | mu | theta | c | P25 c at the jets (v; v/N) | (c - P25)/|P25| | mixing contamination | budget | unitarity (full) |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for q in rows:
        L.append("| %g | %g | %g | %.4f | %.6f | %.5f, %.5f, %.5f | %g | %g | %s | %s | %.5f; %.5f | %s | %s | %.4f | %s |" % (
            q["R_lead_over_rs"], q["R_cut"], q["w"], q["R_h"], q["kappa_4gE"], *q["jets_v"], q["r"], q["mu"], ("%.9f" % q["theta"]) if q["theta"] else "-",
            ("%.5f" % q["c"]) if q["c"] is not None else "-", q["c_P25_v"], q["c_P25_vN"], ("%+.4f" % q["reading"]) if q["reading"] is not None else "-",
            ("%.4f" % q["mixing_contamination"]) if q["mixing_contamination"] is not None else "-", q["budget"],
            ("%.1e, %.1e" % tuple(q["unitarity_full"])) if q["unitarity_full"] else "-"))
    L += ["", "- every number above is printed by the script; the reading is (c measured - P25 at the blended-solved profile's own jets)/|P25|, its budget",
          "  the junction's mixing (from the gate run, when given) + the import (0: no interpolation) + the exact ladder's envelope (--envelope)."]
    open(a.md, "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
