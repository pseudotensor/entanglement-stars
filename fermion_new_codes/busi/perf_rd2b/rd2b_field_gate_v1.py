#!/usr/bin/env python
"""rd2b_field_gate_v1.py (busi, analyst; PERF RD-2b self-test parts (a) and (b) at field level, busi/notes/PERF_RD2b_build_v1.md sec. 1):
engine_fields_v16e against engine_fields_v16d and v16c on the stored sources of a run's fields npz, one process, one thread count.
Per selected record, in order, each solve followed by site_fields_kernel_average (the engines' averaged site fields) and trapping_radius:
  P12 = v16c at rtol 1e-12 / atol 1e-16 (the production route), P14 = v16c at rtol 1e-14 / atol 1e-18 (the reference);
  Dd = v16d with the route unset, De = v16e with the route unset (the defaults);
  Cd = v16d --fields collocation (RD-2), Ce = v16e --fields collocation (RD-2b).
(a) De against Dd and against P12 byte for byte (N_sites, v_sites, W_nodes, N_nodes, v_nodes, E_collar, the averaged fields, the dense N, v, W,
    U on rfine, every non-callable info entry, the trapping-radius result); and Ce against Cd byte for byte wherever Cd is on the collocation
    route (the intact path of v16e is v16d's).
(b) B1 = Ce against P12, B2 = Ce against P14 (the ruling's field-level lines: E rel 3.0e-8, roots 3.4e-7 with the same count, min(N - |v|)
    4.6e-9, site fields 1e-7 dense and averaged, the gauge and regime labels identical, Ce not failing where P12 succeeds); B3 = Ce and P12
    against the stored N, v rows within 1e-5; B4, the accuracy class: Ce's distance from P14 at most twice P12's distance from P14 plus 1e-12,
    for E rel and the site fields. Ce's route per record (collocation or a fallback, with the reason) is reported; a fallback is no budget line.
usage: python rd2b_field_gate_v1.py OUT_DIR CODE16C V16D V16E FIELDS_NPZ --tag T (--last N | --recs i,j,.. | --negcentre [--with-next])
  CODE16C: a code dir whose engine_fields_v16.py is v16c (fa84dad3), with engine_fields_v4/v12/v14 and engine_fields_kcache_v1."""
import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time

import numpy as np

TH = {"E_rel": 3.0e-8, "root": 3.4e-7, "minNv": 4.6e-9, "field": 1e-7, "floor_field": 1e-5, "class_factor": 2.0, "class_abs": 1e-12}


def sanitize(info):
    out = {}
    for k, v in info.items():
        if callable(v) or k == "src":
            continue
        if k == "fields_route" and isinstance(v, dict):
            v = {k2: v2 for k2, v2 in v.items() if k2 not in ("seconds", "timing")}   # the walls of the route, not its numbers
        if isinstance(v, np.ndarray):
            out[k] = hashlib.md5(np.ascontiguousarray(v).tobytes()).hexdigest() + str(v.shape) + str(v.dtype)
        else:
            out[k] = v
    return json.dumps(out, sort_keys=True, default=lambda x: repr(x))


def mx(a, b):
    return float(np.max(np.abs(np.asarray(a, float) - np.asarray(b, float))))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir"); ap.add_argument("code_dir"); ap.add_argument("v16d"); ap.add_argument("v16e"); ap.add_argument("fields")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--last", type=int, default=None); ap.add_argument("--recs", default=None); ap.add_argument("--negcentre", action="store_true")
    a = ap.parse_args()
    for k_ in ("QI2_FIELDS_ROUTE", "QI2_FIELDS_WARM", "QI2_FIELDS_FLAT"):
        assert k_ not in os.environ, "%s set in the environment" % k_
    sys.path.insert(0, a.code_dir)
    import engine_fields_v16 as F16C
    import engine_fields_v4 as F4
    import engine_fields_v12 as F12
    import engine_fields_v14 as F14
    import engine_fields_kcache_v1 as KC
    md5c = hashlib.md5(open(F16C.__file__, "rb").read()).hexdigest()
    assert md5c.startswith("fa84dad3"), "the code dir's engine_fields_v16.py is not v16c: %s" % md5c
    F16D = load("engine_fields_v16d", a.v16d); F16E = load("engine_fields_v16e", a.v16e)
    md5d = hashlib.md5(open(a.v16d, "rb").read()).hexdigest(); md5e = hashlib.md5(open(a.v16e, "rb").read()).hexdigest()
    KC.install([F4, F12, F14, F16C, F16D, F16E])
    assert F16D._RD2["route"] == "ivp" and F16E._RD2["route"] == "ivp" and F16E._RD2["flat"]
    os.makedirs(a.out_dir, exist_ok=True)
    z = np.load(a.fields, allow_pickle=True)
    R = np.asarray(z["R"], float); g = float(z["g"]); ell = float(z["ell"])
    solver = json.loads(str(z["solver"]))
    cp = solver.get("collar_policy", "flag"); sr = float(solver.get("support_rel", 1e-14))
    nT = len(z["T"])
    if a.recs:
        idx = [int(x) for x in a.recs.split(",")]
    elif a.last:
        idx = list(range(nT - a.last, nT))
    else:
        idx = list(range(nT))

    def centre_regular(i):
        src = F12.SmoothedSources5(np.asarray(z["eps_tot"][i], float), np.asarray(z["J_tot"][i], float), R, ell, F16C.coarse_grid(R, ell)[-1], sr)
        return F16C.centre_coefficients(src, g, ell)["regular"]
    if a.negcentre:
        idx = [i for i in idx if not centre_regular(i)]
    head = ("rd2b_field_gate %s: %s, %d sites, ell %g, g %g, collar_policy %s, support_rel %g, %d records %s; v16c %s, v16d %s, v16e %s; threads %s; python %s numpy %s"
            % (a.tag, os.path.basename(a.fields), len(R), ell, g, cp, sr, len(idx), idx, md5c, md5d, md5e, os.environ.get("OMP_NUM_THREADS"),
               sys.version.split()[0], np.__version__))
    print(head, flush=True)
    lines = [head]

    def solve(mod, eps, J, rtol, atol):
        t0 = time.perf_counter()
        Nn, vn, Wn, info = mod.solve_constraints_ivp(eps, J, g, R, ell, N0=None, rtol=rtol, atol=atol, collar_policy=cp, support_rel=sr)
        t1 = time.perf_counter()
        if info.get("failed"):
            return {"failed": info["failed"], "solve_s": t1 - t0, "info": info}
        Na, va, _ = mod.site_fields_kernel_average(info, R, ell)
        tz = mod.trapping_radius(Na, va, R, info)
        return {"failed": None, "N": Nn, "v": vn, "W": Wn, "Na": Na, "va": va, "info": info, "tz": tz, "solve_s": t1 - t0,
                "E": float(info["E_collar"]), "minNv_sites": float(np.min(Na - np.abs(va)))}

    def bitwise(X, Y):
        if X["failed"] or Y["failed"]:
            return X["failed"] == Y["failed"]
        ok = all(np.asarray(X[k]).tobytes() == np.asarray(Y[k]).tobytes() for k in ("N", "v", "W", "Na", "va"))
        ok &= all(np.asarray(X["info"][k]).tobytes() == np.asarray(Y["info"][k]).tobytes() for k in ("N_nodes", "v_nodes"))
        ok &= np.float64(X["E"]).tobytes() == np.float64(Y["E"]).tobytes()
        rf = Y["info"]["rfine"]
        ok &= all(np.asarray(X["info"][k](rf)).tobytes() == np.asarray(Y["info"][k](rf)).tobytes() for k in ("N_dense", "v_dense", "W_field", "U_dense"))
        ok &= sanitize(X["info"]) == sanitize(Y["info"])
        ok &= json.dumps(X["tz"], sort_keys=True) == json.dumps(Y["tz"], sort_keys=True)
        return bool(ok)

    def compare(X, Y):
        d = {}
        d["E_rel"] = abs(X["E"] - Y["E"]) / abs(Y["E"])
        zx, zy = X["tz"]["zeros"], Y["tz"]["zeros"]
        d["roots_n"] = [len(zx), len(zy)]
        d["root"] = max([abs(p - q) for p, q in zip(zx, zy)], default=0.0) if len(zx) == len(zy) else float("inf")
        d["minNv"] = max(abs(X["tz"]["min"] - Y["tz"]["min"]), abs(X["minNv_sites"] - Y["minNv_sites"]))
        d["N_sites"] = mx(X["N"], Y["N"]); d["v_sites"] = mx(X["v"], Y["v"]); d["N_avg"] = mx(X["Na"], Y["Na"]); d["v_avg"] = mx(X["va"], Y["va"])
        d["field"] = max(d["N_sites"], d["v_sites"], d["N_avg"], d["v_avg"])
        d["gauge_equal"] = X["info"]["gauge"]["label"] == Y["info"]["gauge"]["label"]
        d["regime_equal"] = X["info"]["regime"]["label"] == Y["info"]["regime"]["label"]
        d["r_star"] = [(X["info"]["gauge"]["zeros"] or [{}])[0].get("r"), (Y["info"]["gauge"]["zeros"] or [{}])[0].get("r")]
        d["pass"] = bool(d["E_rel"] <= TH["E_rel"] and d["root"] <= TH["root"] and d["minNv"] <= TH["minNv"] and d["field"] <= TH["field"]
                         and d["gauge_equal"] and d["regime_equal"])
        return d

    # warm-up (untimed): the kernel cache and the averaging matrices
    e0, j0 = np.asarray(z["eps_tot"][idx[0]], float), np.asarray(z["J_tot"][idx[0]], float)
    solve(F16C, e0, j0, 1e-12, 1e-16)
    F16E._RD2["route"] = "collocation"; solve(F16E, e0, j0, 1e-12, 1e-16); F16E._RD2["route"] = "ivp"
    for M in (F16D, F16E):
        for k_ in M._RD2["stats"]:
            if isinstance(M._RD2["stats"][k_], (int, float)):
                M._RD2["stats"][k_] = type(M._RD2["stats"][k_])(0)
        M._RD2["stats"]["fallback_reasons"] = {}
    recs = []; a_fail = []; b_fail = []
    for i in idx:
        eps = np.asarray(z["eps_tot"][i], float); J = np.asarray(z["J_tot"][i], float)
        rec = {"i": i, "T": float(z["T"][i])}
        P12 = solve(F16C, eps, J, 1e-12, 1e-16)
        Dd = solve(F16D, eps, J, 1e-12, 1e-16)
        De = solve(F16E, eps, J, 1e-12, 1e-16)
        F16D._RD2["route"] = "collocation"; F16E._RD2["route"] = "collocation"
        sd0 = F16D.rd2_stats(); se0 = F16E.rd2_stats()
        Cd = solve(F16D, eps, J, 1e-12, 1e-16)
        Ce = solve(F16E, eps, J, 1e-12, 1e-16)
        sd1 = F16D.rd2_stats(); se1 = F16E.rd2_stats()
        F16D._RD2["route"] = "ivp"; F16E._RD2["route"] = "ivp"
        P14 = solve(F16C, eps, J, 1e-14, 1e-18)
        rfd = (Cd["info"].get("fields_route") or {}); rfe = (Ce["info"].get("fields_route") or {})
        rec["route_Cd"] = rfd.get("route"); rec["route_Ce"] = rfe.get("route"); rec["branch_Ce"] = rfe.get("branch", "intact" if rfe.get("route") == "collocation" else None)
        rec["fallback_Cd"] = sd1["fallback"] - sd0["fallback"]; rec["fallback_Ce"] = se1["fallback"] - se0["fallback"]
        rec["reason_Cd"] = rfd.get("reason"); rec["reason_Ce"] = rfe.get("reason")
        rec["newton_it_Ce"] = rfe.get("newton_iterations"); rec["timing_Ce"] = rfe.get("timing"); rec["r_star_Ce"] = rfe.get("r_star")
        rec["s"] = {k: X["solve_s"] for k, X in (("P12", P12), ("Dd", Dd), ("De", De), ("Cd", Cd), ("Ce", Ce), ("P14", P14))}
        rec["nfev_P12"] = (P12["info"].get("ivp") or {}).get("nfev")
        rec["a_De_Dd"] = bitwise(De, Dd); rec["a_De_P12"] = bitwise(De, P12)
        rec["a_Ce_Cd"] = bitwise(Ce, Cd) if rec["route_Cd"] == "collocation" else None
        ok_a = rec["a_De_Dd"] and rec["a_De_P12"] and (rec["a_Ce_Cd"] is not False)
        rec["a_pass"] = bool(ok_a)
        if not ok_a:
            a_fail.append(i)
        if P12["failed"]:
            rec["B"] = "P12 failed: %s" % P12["failed"]
        elif Ce["failed"]:
            rec["B"] = "Ce failed where P12 succeeds: %s" % Ce["failed"]; b_fail.append(i)
        else:
            rec["B1"] = compare(Ce, P12)
            rec["B2"] = compare(Ce, P14) if not P14["failed"] else None
            rec["P12_vs_P14"] = compare(P12, P14) if not P14["failed"] else None
            zN = np.asarray(z["N"][i], float); zv = np.asarray(z["v"][i], float)
            rec["B3"] = {"Ce": max(mx(Ce["Na"], zN), mx(Ce["va"], zv)), "P12": max(mx(P12["Na"], zN), mx(P12["va"], zv))}
            rec["B3"]["pass"] = bool(rec["B3"]["Ce"] <= TH["floor_field"] and rec["B3"]["P12"] <= TH["floor_field"])
            if rec["B2"] is not None:
                b4 = {}
                for sub in ("E_rel", "field"):
                    b4[sub] = [rec["B2"][sub], rec["P12_vs_P14"][sub]]
                    b4[sub + "_pass"] = bool(rec["B2"][sub] <= TH["class_factor"] * rec["P12_vs_P14"][sub] + TH["class_abs"])
                b4["pass"] = b4["E_rel_pass"] and b4["field_pass"]
                rec["B4"] = b4
            okb = rec["B1"]["pass"] and (rec["B2"] is not None and rec["B2"]["pass"]) and rec["B3"]["pass"] and rec.get("B4", {}).get("pass", False)
            rec["B_pass"] = bool(okb)
            if not okb:
                b_fail.append(i)
        B1 = rec.get("B1") or {}; B2 = rec.get("B2") or {}; PP = rec.get("P12_vs_P14") or {}
        line = ("rec %d T %.6g: (a) De=Dd %s De=P12 %s Ce=Cd %s | Cd %s (fallback %d) Ce %s [%s] (fallback %d%s), r_* %s, Newton it %s | P12 %.3f s (nfev %s) Cd %.3f s Ce %.3f s P14 %.3f s"
                % (i, rec["T"], rec["a_De_Dd"], rec["a_De_P12"], rec["a_Ce_Cd"], rec["route_Cd"], rec["fallback_Cd"], rec["route_Ce"],
                   (rec["branch_Ce"] or "")[:14], rec["fallback_Ce"], (": " + str(rec["reason_Ce"])) if rec["reason_Ce"] else "",
                   ("%.10f" % rec["r_star_Ce"]) if rec["r_star_Ce"] else None, rec["newton_it_Ce"], P12["solve_s"], rec["nfev_P12"], Cd["solve_s"], Ce["solve_s"], P14["solve_s"]))
        if B1:
            line += (" | B1 Ce-P12: E %.1e root %.1e (n %s) minNv %.1e field %.1e (Navg %.1e vavg %.1e) labels %s/%s %s | B2 Ce-P14: E %.1e root %.1e minNv %.1e field %.1e %s"
                     " | P12-P14: E %.1e field %.1e | B4 class %s | B3 vs stored: Ce %.1e P12 %.1e %s | r_* Ce-P12 %.1e"
                     % (B1["E_rel"], B1["root"], B1["roots_n"], B1["minNv"], B1["field"], B1["N_avg"], B1["v_avg"], B1["gauge_equal"], B1["regime_equal"],
                        "PASS" if B1["pass"] else "FAIL", B2.get("E_rel", float("nan")), B2.get("root", float("nan")), B2.get("minNv", float("nan")),
                        B2.get("field", float("nan")), "PASS" if B2.get("pass") else "FAIL", PP.get("E_rel", float("nan")), PP.get("field", float("nan")),
                        "PASS" if rec.get("B4", {}).get("pass") else "FAIL", rec["B3"]["Ce"], rec["B3"]["P12"], "PASS" if rec["B3"]["pass"] else "FAIL",
                        abs(B1["r_star"][0] - B1["r_star"][1]) if (B1["r_star"][0] is not None and B1["r_star"][1] is not None) else float("nan")))
        else:
            line += " | " + str(rec.get("B"))
        line += " | Ce timing %s" % " ".join("%s %.3f" % (k_, v_) for k_, v_ in (rec["timing_Ce"] or {}).items())
        print(line, flush=True); lines.append(line)
        recs.append(rec)
    ok = [r for r in recs if "B1" in r]

    def agg(key, sub):
        vals = [r[key][sub] for r in ok if r.get(key)]
        return float(np.max(vals)) if vals else None
    summ = {"records": len(recs), "a_pass": len(recs) - len(a_fail), "a_fail": a_fail, "b_fail": b_fail,
            "Ce_collocation": sum(1 for r in recs if r["route_Ce"] == "collocation"), "Cd_collocation": sum(1 for r in recs if r["route_Cd"] == "collocation"),
            "fallbacks_Cd": int(sum(r["fallback_Cd"] for r in recs)), "fallbacks_Ce": int(sum(r["fallback_Ce"] for r in recs)),
            "fallback_reasons_Ce": F16E.rd2_stats()["fallback_reasons"], "fallback_reasons_Cd": F16D.rd2_stats()["fallback_reasons"],
            "negative_centre_Ce": F16E.rd2_stats()["negative_centre"],
            "B1_fail_records": [r["i"] for r in ok if not r["B1"]["pass"]], "B2_fail_records": [r["i"] for r in ok if not (r["B2"] or {}).get("pass")],
            "B3_fail_records": [r["i"] for r in ok if not r["B3"]["pass"]], "B4_fail_records": [r["i"] for r in ok if not r.get("B4", {}).get("pass")]}
    for key in ("B1", "B2", "P12_vs_P14"):
        for sub in ("E_rel", "root", "minNv", "field", "N_avg", "v_avg"):
            summ["%s_%s_max" % (key, sub)] = agg(key, sub)
    summ["B3_Ce_max"] = float(np.max([r["B3"]["Ce"] for r in ok])) if ok else None
    summ["B3_P12_max"] = float(np.max([r["B3"]["P12"] for r in ok])) if ok else None
    for k in ("P12", "Cd", "Ce", "P14"):
        summ["%s_solve_mean_s" % k] = float(np.mean([r["s"][k] for r in recs]))
    summ["factor_Ce"] = summ["P12_solve_mean_s"] / summ["Ce_solve_mean_s"]
    summ["factor_Cd"] = summ["P12_solve_mean_s"] / summ["Cd_solve_mean_s"]
    its = [r["newton_it_Ce"] for r in recs if r["newton_it_Ce"] is not None]
    summ["newton_it_Ce_mean"] = float(np.mean(its)) if its else None
    summ["verdict_a"] = "PASS" if not a_fail else "FAIL"
    summ["verdict_b"] = "PASS" if not b_fail else "FAIL"
    summ["verdict_b_lines"] = {k: ("PASS" if not summ["%s_fail_records" % k] else "FAIL") for k in ("B1", "B2", "B3", "B4")}
    lines.append("SUMMARY %s" % json.dumps(summ))
    print(lines[-1], flush=True)
    with open(os.path.join(a.out_dir, "rd2b_field_gate_%s.txt" % a.tag), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    with open(os.path.join(a.out_dir, "rd2b_field_gate_%s.json" % a.tag), "w") as fh:
        json.dump({"summary": summ, "records": recs, "args": vars(a), "thresholds": TH, "stats_v16e": F16E.rd2_stats(), "stats_v16d": F16D.rd2_stats()},
                  fh, indent=1, default=float)


if __name__ == "__main__":
    main()
