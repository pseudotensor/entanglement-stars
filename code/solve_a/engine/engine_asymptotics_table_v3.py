#!/usr/bin/env python
"""engine_asymptotics_table_v3.py (builder analyst; engine spec 30.5 and 30.6 (7): "version 1.1 adds the tau 2 and tau 1 rows and row (6)
when C-capture version 6 lands"): version 2 with the table's title given by --title (the delivery's version). Nothing else changes;
version 2's docstring follows.
engine_asymptotics_table_v2.py (builder analyst): version 1 with the goal's two-route sentence found (version 1's pattern could
not cross the decimal points of its numbers and printed "not found"). Nothing else changes; version 1's docstring follows.
engine_asymptotics_table_v1.py (builder analyst; the goal's task 7 and engine spec 29.10: "The table (engine_asymptotics_table_v1; from
existing records, no new run; delivered as ASYMPTOTICS_checks_v1.md), each row labelled by r_s/a, sigma/a, tau and the engine version"):
the numerical checks of the asymptotic expansion that this lane's stored records reach, printed from the records (nothing retyped):
(1) the first trapping radius against 2 g E at every record of the first episode on the production, the tau 4 control and the ell 8 record
(R_t, 2 g E(T) and their ratio; the first episode from its first record with a trapping zero to the last before the first record without);
(2) the surface trajectory against the late-time law R - r_s proportional to e^(-T/r_s): per history and episode, R_t - 2 g E per record
and, on every run of at least 3 consecutive records with R_t above 2 g E, the fitted rate of ln(R_t - 2 g E) in T against -1/(2 g E)
(T the records' flat-slice time, the clock at infinity in the exterior where N = 1); "not applicable" on the records with R_t below 2 g E;
(3) the back-averaged trapping root and slope against 1 - (9/4) sigma^2/r_s^2 at the records' own sigma (obs_j3p_scope_v2's jsons:
sigma^2 = ell^2 + a^2/12 for this lane's Gaussian): per history the prediction and the ranges of r_H,m/r_s and 2 r_s kappa_m;
(4) the exact-profile ladder's coefficients: the J-PL report's per-size fit table at R_lead/r_s 32, verbatim, and the goal's two-route
sentence, verbatim, against -5/8, -1, 3/4; (5) c_E = E g^1.5 (m a)^2 and c_R = (R_surf/a) g^0.5 (m a)^2 at the largest class-1 member of each
static family (class 1: N_B up to the dense zone table's continuum N_B,max, engine_ovpg_v3's control (iii)), the continuum's values at its
maximum on the same table beside, and the null-test triples at N_B up to N_B,max with their ell spread (E_16 - E_4)/E_8 and D/(E_16 - E_4)
(engine_star_nulltest_v1.null_entries); (6) the capture run's non-return fraction when it exists. Every number printed by the script.
Args: --histories (label=json[,json]|r_s|ell|tau|engine;...) --scope (label=json;...) --jpl --goal --families (label=json[,json];...)
--ovpg3 --capture (md path or none) --md."""
import argparse
import glob
import json
import math
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine_star_nulltest_v1 as NT  # noqa: E402


def records(paths):
    out = {}
    for p in paths:
        J = json.load(open(p))
        g = float(J["args"]["g"])
        for r in J["records"]:
            if r.get("E") is None:
                continue
            out[round(float(r["T"]), 6)] = (r, g)
    return [out[k] for k in sorted(out)]


def episodes(recs):
    eps, cur = [], []
    for r, g in recs:
        if r.get("R_t") is not None:
            cur.append((r, g))
        elif cur:
            eps.append(cur); cur = []
    if cur:
        eps.append(cur)
    return eps


def main():
    ap = argparse.ArgumentParser()
    for k in ("histories", "scope", "jpl", "goal", "families", "ovpg3", "capture", "md"):
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--title", default="ASYMPTOTICS_checks_v1")
    a = ap.parse_args()
    L = ["## %s: the numerical checks of the asymptotic expansion from this lane's stored records (engine_asymptotics_table_v3.py;" % a.title + " the goal's task 7, engine spec 29.10; no new run)", "",
         "- every row is labelled by r_s/a (nominal; the solved 2 g E beside), the kernel's sigma/a (this lane's Gaussian of standard deviation ell: sigma^2 = ell^2 + a^2/12, a = 1), tau and the engine version; every number is read from the records by the script.", ""]
    hist = []
    for item in a.histories.split(";"):
        lab, rest = item.split("=", 1)
        paths, rs, ell, tau, eng = rest.split("|")
        hist.append((lab, paths.split(","), float(rs), float(ell), float(tau), eng))
    L += ["### (1) The first trapping radius against 2 g E (every record of the first episode)", "",
          "| history | r_s/a | sigma/a | tau | engine | T/rs | R_t | 2 g E | R_t/(2 g E) |", "|---|---|---|---|---|---|---|---|---|"]
    traj = []
    for lab, paths, rs, ell, tau, eng in hist:
        recs = records(paths)
        eps = episodes(recs)
        sig = math.sqrt(ell * ell + 1.0 / 12.0)
        if not eps:
            L.append("| %s | %g | %.3f | %g | %s | no trapping zero in any record | - | - | - |" % (lab, rs, sig, tau, eng)); continue
        for r, g in eps[0]:
            twogE = 2.0 * g * float(r["E"])
            L.append("| %s | %g | %.3f | %g | %s | %.3f | %.4f | %.4f | %.5f |" % (lab, rs, sig, tau, eng, float(r["T"]) / rs, float(r["R_t"]), twogE, float(r["R_t"]) / twogE))
        traj.append((lab, rs, sig, tau, eng, eps))
    L += ["", "### (2) The surface trajectory against R - r_s proportional to e^(-T/r_s) (T: the records' flat-slice time, the clock at infinity in the exterior)", "",
          "| history | r_s/a | sigma/a | tau | engine | episode (T/rs from, to) | records | R_t above 2 g E | fitted rate of ln(R_t - 2gE) in T (records) | -1/(2 g E) (mean over the fit) | rate x 2 g E |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for lab, rs, sig, tau, eng, eps in traj:
        for ie, ep in enumerate(eps):
            T = np.array([float(r["T"]) for r, g in ep]); Rt = np.array([float(r["R_t"]) for r, g in ep]); tg = np.array([2.0 * g * float(r["E"]) for r, g in ep])
            above = Rt > tg
            runs, cur = [], []
            for i, ok in enumerate(above):
                if ok:
                    cur.append(i)
                else:
                    if len(cur) >= 3:
                        runs.append(cur)
                    cur = []
            if len(cur) >= 3:
                runs.append(cur)
            if not runs:
                L.append("| %s | %g | %.3f | %g | %s | %d (%.3f, %.3f) | %d | %d | not applicable (R_t below 2 g E on %d of %d records) | - | - |" % (
                    lab, rs, sig, tau, eng, ie + 1, T[0] / rs, T[-1] / rs, len(T), int(above.sum()), int((~above).sum()), len(T)))
                continue
            for run in runs:
                tt = T[run]; yy = np.log(Rt[run] - tg[run])
                rate = float(np.polyfit(tt, yy, 1)[0]); ref = -1.0 / float(np.mean(tg[run]))
                L.append("| %s | %g | %.3f | %g | %s | %d (%.3f, %.3f) | %d | %d | %+.5e (%d, T/rs %.3f to %.3f) | %+.5e | %+.4f |" % (
                    lab, rs, sig, tau, eng, ie + 1, T[0] / rs, T[-1] / rs, len(T), int(above.sum()), rate, len(run), tt[0] / rs, tt[-1] / rs, ref, rate * float(np.mean(tg[run]))))
    L += ["", "### (3) The back-averaged trapping root and slope against 1 - (9/4) sigma^2/r_s^2 at the records' own sigma (obs_j3p_scope_v2)", "",
          "| history | sigma^2 | records | r_s = 2gE (range) | prediction (range) | r_H,m/r_s (range) | 2 r_s kappa_m (range) | 4 g E d_R(N+v)|_H - 1 (range, dense) |", "|---|---|---|---|---|---|---|---|"]
    for item in a.scope.split(";"):
        lab, p = item.split("=", 1)
        S = json.load(open(p)); rows = [r for r in S["rows"] if not r.get("missing") and r.get("r_Hm") is not None]
        if not rows:
            L.append("| %s | - | 0 | - | - | - | - | - |" % lab); continue
        f = lambda key, fn=lambda r, k: r[k]: [fn(r, key) for r in rows]
        rr = [r["r_Hm"] / r["r_s"] for r in rows]; kk = [2.0 * r["r_s"] * r["kappa_m"] for r in rows if r.get("kappa_m") is not None]
        tst = [r.get("test_4gE_dN+v_minus_1") for r in rows]
        tst = [x for x in tst if x is not None]
        rng = lambda v, fm: ("%s to %s" % (fm % min(v), fm % max(v))) if v else "-"
        L.append("| %s | %.6f | %d | %s | %s | %s | %s | %s |" % (lab, rows[0]["sigma2"], len(rows), rng(f("r_s"), "%.3f"), rng(f("prediction"), "%.6f"), rng(rr, "%.4f"), rng(kk, "%.4f"), rng(tst, "%+.3f")))
    L += ["", "### (4) The exact-profile ladder's coefficients against -5/8, -1, 3/4", ""]
    jl = open(a.jpl).read().split("\n")
    hi = [i for i, l in enumerate(jl) if l.startswith("| continuation R_1/r_s, w | R_lead/r_s |")]
    if hi:
        i = hi[0]
        L += ["- the J-PL report's per-size fit (%s; the rows at R_lead/r_s 32, verbatim):" % os.path.basename(a.jpl), "", jl[i], jl[i + 1]]
        for l in jl[i + 2:]:
            if not l.startswith("|"):
                break
            if l.split("|")[2].strip() == "32":
                L.append(l)
    gm = re.search(r"the fitted coefficients c_0, c_2, c_m of .*?against the exact -5/8, -1 and 3/4", open(a.goal).read())
    L += ["", "- the goal's statement of the two lane routes (verbatim): \"%s\"." % (gm.group(0) if gm else "not found")]
    L += ["", "### (5) c_E = E g^1.5 (m a)^2 and c_R = (R_surf/a) g^0.5 (m a)^2 at the largest class-1 member of each family (the families' point-sampled site fields), the dense-table continuum maximum beside", "",
          "| family (m a, g) | N_B,max (dense table) | largest class-1 member: N_B | ell | E | R_surf | c_E | c_R | continuum at its maximum: M, R_surf | c_E | c_R | null-test triples at N_B up to N_B,max: N_B, (E_16 - E_4)/E_8, D/(E_16 - E_4) |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    O = json.load(open(a.ovpg3))
    dense = {(round(p["m"], 6), round(p["g"], 8)): p["closures"]["dense_thermo"]["max_M"] for p in O["pairs"]}
    for item in a.families.split(";"):
        lab, ps = item.split("=", 1)
        recs = [(os.path.basename(p), json.load(open(p))) for p in ps.split(",")]
        mem = {}
        for nm, R in recs:
            for m_ in R["members"]:
                if m_.get("failed") or m_.get("E") is None or m_.get("N_B") is None:
                    continue
                mem.setdefault(round(float(m_["N_B"]), 3), m_)
        anym = next(iter(mem.values()))
        ma, g = float(anym["m"]), float(anym["g"])
        dm = dense.get((round(ma, 6), round(g, 8)))
        if dm is None:
            L.append("| %s | no dense-table closure | - | - | - | - | - | - | - | - | - | - |" % lab); continue
        c1 = [m_ for k, m_ in mem.items() if k <= dm["N_B"] + 1e-9 and (m_.get("min_N_minus_absv") or 0) > 0]
        if not c1:
            L.append("| %s | %.4f | none | - | - | - | - | - | - | - | - | - |" % (lab, dm["N_B"])); continue
        top = max(c1, key=lambda m_: float(m_["N_B"]))
        cE = float(top["E"]) * g ** 1.5 * ma ** 2; cR = float(top["R_surf"]) * g ** 0.5 * ma ** 2
        cEc = float(dm["M"]) * g ** 1.5 * ma ** 2; cRc = float(dm["R_surf"]) * g ** 0.5 * ma ** 2
        nts = []
        for x in NT.null_entries(recs):
            if x["NB_target"] <= dm["N_B"] + 1e-9 and "D" in x:
                nts.append("%.3f, %.4f, %+.3f" % (x["NB_target"], x["S"] / x["E"]["8"], x["ratio"]))
        L.append("| %s | %.4f | %.3f | %s | %.6f | %.3f | %.5f | %.4f | %.4f, %.3f | %.5f | %.4f | %s |" % (
            lab, dm["N_B"], float(top["N_B"]), top.get("ell", "-"), float(top["E"]), float(top["R_surf"]), cE, cR, float(dm["M"]), float(dm["R_surf"]), cEc, cRc, "; ".join(nts) if nts else "none"))
    L += ["", "### (6) The capture run's non-return fraction", ""]
    if a.capture != "none" and os.path.exists(a.capture):
        cl = [l for l in open(a.capture).read().split("\n") if "non-return" in l]
        L += ["- from %s (its lines naming the non-return fraction, verbatim):" % os.path.basename(a.capture)] + ["  " + l for l in cl[:12]] if cl else ["- %s holds no line naming a non-return fraction." % os.path.basename(a.capture)]
    else:
        L += ["- pending C-capture version 5 (its output not yet on this host)."]
    L += ["", "- every number above is printed by the script from the stored records, obs_j3p_scope_v2's jsons, the J-PL report, the static families' records and engine_ovpg_v3's json; the reading against the expansion is the lead's and the analytic lanes'."]
    open(a.md, "w").write("\n".join(L) + "\n")
    print("\n".join(L[:12]))


if __name__ == "__main__":
    main()
