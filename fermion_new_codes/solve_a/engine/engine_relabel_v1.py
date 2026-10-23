#!/usr/bin/env python
"""engine_relabel_v1.py (builder analyst; SPEC_final4b_engine.md 18.1 and 18.7, post-processing of every stored collapse record; no stored
field changes and no run restarts): per run (its json and fields npz) and per record,
(1) the indicator labels: version 10 to 15's certificate entries renamed as what they are, q_c -> q_observed, eps_map -> delta_rtol (with
    its source record), B -> B_estimate, "certified | uncertified" -> "indicator pass | fail (B_estimate < 1e-9)", sticky from the first
    failure as before; engine_collapse_v17 records carry their own indicator, copied;
(2) the collar: the padded cost (engine_fields_v12.padded_collar_cost on the dense pair re-solved from the stored shell totals by
    solve_constraints_ivp: delta E between the old collar a and b = a + 8 ell, ln N_b(a), acceptance 1e-9 each, evaluated at the old collar a,
    stated in the header) with the status "lifted (padded cost)" where both pass and the collar was flagged, "flagged" where not, "clean"
    where the collar lay inside the chain; the record's own tail-sum screen beside it;
(3) the surface jets (records with a trapping radius R_t): the constraint-differentiated route of B13 (obs_slice_jets_v3.constraint_jets on
    the re-solved dense pair: no h^-k amplification) as the primary jet, the record's finite-difference jets as the comparator; per jet the
    error |constraint - FD| + the constraint route's source spread times |jet| (its propagated source-fit error), raised to the fixture's
    miss at the same order where smaller (R25a's manufactured profile: A_q 9.1e-8, C_q 4.169e-3, C_v 2.917e-4, absolute); v's jets in
    z = kappa_T (R - R_t), v/N's in z = kappa' (R - R_t) with kappa' = (v/N)' (the corrected normalization) and, labelled, in kappa_T beside;
    P25's c_0, c_2, c_m at each set of jets (obs_plateau_jets_v1.p25_coeffs);
(4) once: the lower fixture table of R25a's section 5, whose constraint-route v/N coefficients -2.116707, 4.190774, -8.711791 were
    kappa_T-normalized, multiplied by N_t^2, N_t^3, N_t^4 (N_t = 1 - 0.3 e^-0.64) and the c's recomputed, against the exact series of
    q = -1/sqrt(1 + 2z) (-3/2, 5/2, -35/8).
Writes <out>/relabel_v1_<tag>.json and .md per run and <out>/relabel_v1_fixture.md. Args: --runs dir/tag ... --out --skip-padded --records."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
import argparse
import json
import math
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
for _p in (os.path.join(HERE, "..", "..", "obs", "code"), os.path.join(HERE, "..", "obs")):
    sys.path.insert(0, _p)
import engine_fields_v12 as FLD  # noqa: E402
import obs_slice_jets_v3 as SJ3  # noqa: E402
import obs_plateau_jets_v1 as JT  # noqa: E402

FIXTURE_MISS = {"v_over_N": [9.1e-8, None, 4.169e-3], "v": [None, None, 2.917e-4]}


def constraint_route(info, g, Rt, ell):
    info = dict(info)
    info["src_direct"] = info["src"].direct
    cj = SJ3.constraint_jets(info, g, Rt, ell)
    kT = float(cj["f"][1]); kq = float(cj["v_over_N"][1])
    v, q = cj["v"], cj["v_over_N"]
    jets_v = [float(v[2] / kT ** 2), float(v[3] / kT ** 3), float(v[4] / kT ** 4)]
    jets_q = [float(q[2] / kq ** 2), float(q[3] / kq ** 3), float(q[4] / kq ** 4)]
    jets_q_kT = [float(q[2] / kT ** 2), float(q[3] / kT ** 3), float(q[4] / kT ** 4)]
    return {"kappa_T": kT, "kappa_prime": kq, "jets_v": jets_v, "jets_v_over_N": jets_q, "jets_v_over_N_in_kappa_T": jets_q_kT, "source_spread": cj["source_spread"]}


def jet_errors(cr, fd):
    out = {}
    for key, fdkey in (("v", "jets_v"), ("v_over_N", "jets_v_over_N")):
        c = cr[fdkey]
        f = fd.get(fdkey)
        if f is None:
            continue
        errs = []
        for i in range(3):
            e = abs(c[i] - f[i]) + cr["source_spread"] * abs(c[i])
            floor = FIXTURE_MISS[key][i]
            if floor is not None and e < floor:
                e = floor
            errs.append(e)
        out[key] = {"constraint": c, "fd": f, "error": errs, "c_constraint": list(JT.p25_coeffs(*c)), "c_fd": list(JT.p25_coeffs(*f))}
    return out


def fixture_table():
    Nt = 1.0 - 0.3 * math.exp(-0.64)
    old = [-2.116707, 4.190774, -8.711791]
    new = [old[0] * Nt ** 2, old[1] * Nt ** 3, old[2] * Nt ** 4]
    exact = [-1.5, 2.5, -35.0 / 8.0]
    L = ["## R25a's lower fixture table, constraint route, v/N's coefficients rescaled (engine_relabel_v1.py)", "",
         "N_t = 1 - 0.3 e^-0.64 = %.10f; the stored coefficients were kappa_T-normalized; times N_t^2, N_t^3, N_t^4 they are in z = kappa' (R - R_t)." % Nt, "",
         "| order | stored (kappa_T) | rescaled (kappa') | exact (q = -1/sqrt(1 + 2z)) | rescaled - exact | relative |", "|---|---|---|---|---|---|"]
    for i, (o, n, e) in enumerate(zip(old, new, exact)):
        L.append("| z^%d | %.6f | %.6f | %.6f | %.2e | %.2e |" % (i + 2, o, n, e, n - e, (n - e) / abs(e)))
    c_new = JT.p25_coeffs(*new); c_ex = JT.p25_coeffs(*exact)
    L += ["", "P25's c_0, c_2, c_m at the rescaled jets: %.6f, %.6f, %.6f; at the exact jets: %.6f, %.6f, %.6f." % (tuple(c_new) + tuple(c_ex)),
          "", "- every number above is printed by the script."]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="dir/tag of each run (collapse_<tag>.json and collapse_<tag>_fields.npz in dir)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-padded", action="store_true")
    ap.add_argument("--records", type=int, default=None, help="only the first K records (a test)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    open(os.path.join(a.out, "relabel_v1_fixture.md"), "w").write(fixture_table())
    for run in a.runs:
        d, tag = os.path.split(run)
        J = json.load(open(os.path.join(d, "collapse_%s.json" % tag)))
        Z = np.load(os.path.join(d, "collapse_%s_fields.npz" % tag), allow_pickle=True)
        R = np.asarray(Z["R"], float); g = float(Z["g"]); ell = float(Z["ell"]); Tz = np.asarray(Z["T"], float)
        rs = float(J["args"]["rs"])
        recs = [r for r in J["records"] if r.get("E") is not None]
        if a.records:
            recs = recs[:a.records]
        out = {"script": "engine_relabel_v1.py", "run": tag, "source": J.get("script"), "records": []}
        first_fail = None
        t0 = time.time()
        L = ["## %s relabelled (engine_relabel_v1.py; %s)" % (tag, J.get("script")), "",
             "The collar's padded cost is evaluated at the old collar a (the record's r_collar) against b = a + 8 ell; acceptance |delta E| < 1e-9 E and |ln N_b(a)| < 1e-9.", "",
             "| T/rs | q_observed | delta_rtol (source) | B_estimate | indicator | run label | collar screen dE/E | padded dE/E | ln N_b(a) | collar status | R_t | jets: v (constraint; FD; error) | v/N in kappa' (constraint; FD; error) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in recs:
            k = int(np.argmin(np.abs(Tz - r["T"])))
            row = {"T": r["T"], "T_over_rs": r["T"] / rs}
            ind = r.get("indicator")
            if ind is not None:                                         # engine_collapse_v17
                row.update({"q_observed": None, "delta_rtol": ind.get("delta_rtol"), "delta_rtol_source": ind.get("delta_rtol_source"),
                            "B_estimate": ind.get("B_estimate"), "indicator_pass": bool(ind.get("indicator_pass"))})
                screen = (ind.get("collar_screen") or {}).get("delta_E_rel_abs")
            else:
                c = r.get("certificate") or {}
                if c:
                    row.update({"q_observed": c.get("q_c"), "q_observed_prefloor": c.get("q_c_prefloor"), "delta_rtol": c.get("eps_map"),
                                "delta_rtol_source": c.get("eps_map_source", "this record"), "B_estimate": c.get("B"), "B_estimate_prefloor": c.get("B_prefloor"),
                                "indicator_pass": bool(c.get("B") is not None and c.get("B") < 1e-9)})
                else:                                                   # engines before version 10 computed no certificate
                    row.update({"q_observed": None, "delta_rtol": None, "delta_rtol_source": "not computed by this engine", "B_estimate": None, "indicator_pass": None})
                screen = (c.get("collar") or {}).get("delta_E_rel_abs")
            if row["indicator_pass"] is False and first_fail is None:
                first_fail = row["T_over_rs"]
            if row["indicator_pass"] is None and first_fail is None:
                row["run_label"] = "indicator not computed"
            else:
                row["run_label"] = "indicator pass" if first_fail is None else "indicator fail from T/rs %.2f" % first_fail
            row["collar_screen_delta_E_rel_abs"] = screen
            info = None
            if not a.skip_padded or r.get("R_t"):
                eps = np.asarray(Z["eps_tot"][k], float); Jt = np.asarray(Z["J_tot"][k], float)
                _, _, _, info = FLD.solve_constraints_ivp(eps, Jt, g, R, ell, collar_policy="flag", support_rel=1e-14)
                if info.get("failed"):
                    row["resolve_failed"] = info["failed"]; info = None
            if info is not None and not a.skip_padded:
                flagged = bool(info.get("collar_margin", {}).get("deficit_ell", 0) > 0)
                if flagged:
                    pc = FLD.padded_collar_cost(eps, Jt, g, R, ell, info)
                    row["collar_padded"] = pc
                    row["collar_status"] = "lifted (padded cost)" if pc["passes"] else "flagged"
                else:
                    row["collar_status"] = "clean"
            Rt = r.get("R_t")
            if Rt and info is not None:
                try:
                    cr = constraint_route(info, g, Rt, ell)
                    row["jets"] = {"constraint": cr, "comparison": jet_errors(cr, r.get("surface_jets") or {})}
                except Exception as exc:                             # noqa: BLE001
                    row["jets"] = {"failed": repr(exc)}
            out["records"].append(row)
            pc = row.get("collar_padded") or {}
            jv = (row.get("jets") or {}).get("comparison", {}).get("v")
            jq = (row.get("jets") or {}).get("comparison", {}).get("v_over_N")
            f3 = lambda x: ", ".join("%.6f" % y for y in x)
            L.append("| %.3f | %s | %s (%s) | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                row["T_over_rs"], ("%.3f" % row["q_observed"]) if row.get("q_observed") is not None else "-",
                ("%.1e" % row["delta_rtol"]) if row.get("delta_rtol") is not None else "-", row.get("delta_rtol_source") or "-",
                ("%.1e" % row["B_estimate"]) if row.get("B_estimate") is not None else "none", {True: "pass", False: "fail", None: "not computed"}[row["indicator_pass"]], row["run_label"],
                ("%.1e" % screen) if screen is not None else "-", ("%.1e" % pc["delta_E_rel"]) if pc else "-", ("%.1e" % pc["lnN_b_at_a"]) if pc else "-",
                row.get("collar_status", "-"), ("%.3f" % Rt) if Rt else "-",
                ("%s; %s; %s" % (f3(jv["constraint"]), f3(jv["fd"]), ", ".join("%.1e" % e for e in jv["error"]))) if jv else "-",
                ("%s; %s; %s" % (f3(jq["constraint"]), f3(jq["fd"]), ", ".join("%.1e" % e for e in jq["error"]))) if jq else "-"))
        out["seconds"] = time.time() - t0
        json.dump(out, open(os.path.join(a.out, "relabel_v1_%s.json" % tag), "w"), indent=1, default=float)
        L += ["", "- every number above is printed by the script from the run's json and fields npz (%d records, %.0f s); the stored records are unchanged." % (len(recs), out["seconds"])]
        open(os.path.join(a.out, "relabel_v1_%s.md" % tag), "w").write("\n".join(L) + "\n")
        print("%s: %d records relabelled in %.0f s; run label %s" % (tag, len(recs), out["seconds"], recs and out["records"][-1]["run_label"]), flush=True)


if __name__ == "__main__":
    main()
