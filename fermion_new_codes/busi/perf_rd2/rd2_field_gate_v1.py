#!/usr/bin/env python
"""rd2_field_gate_v1.py (busi, analyst; PERF RD-2 / RD-5 self-test parts (a) and (b), busi/notes/PERF_RD2_build_v1.md sec. 1): the field
module engine_fields_v16d against engine_fields_v16c on the stored sources of a run's fields npz, one process, one thread count.
Per record (the last NREC of the npz, in order), each solve followed by site_fields_kernel_average (the engines' averaged site fields):
  P12 = v16c at rtol 1e-12 / atol 1e-16 (the production route), D = v16d with the route unset (the default),
  C = v16d --fields collocation (cold Newton start), W = v16d collocation with the RD-5 warm start (the cache carried from record to record),
  P14 = v16c at rtol 1e-14 / atol 1e-18 (the reference).
(a) D against P12 byte for byte: N_sites, v_sites, W_nodes, N_nodes, v_nodes, E_collar, the averaged fields, the dense N, v, W, U on rfine and
    every non-callable info entry (json of the sanitized dict).
(b) B1 = C against P12, B2 = C against P14, B3 = C and P12 against the stored N, v rows; W against C at B1's thresholds (RD-5), and a nearby
    solve (the sources moved by 1e-3 of the change from the previous record) warm-started from C's solution (Wp) against its cold solve (Cp).
Thresholds (written before the test, the notes' sec. 1): E rel 1e-9, roots 3.4e-7 (same count), min(N - |v|) 4.6e-9, site fields 1e-7
(dense and averaged), gauge and regime labels equal, C not failing where P12 succeeds; B3 fields 1e-5.
usage: python rd2_field_gate_v1.py OUT_DIR CODE_DIR V16D_PATH FIELDS_NPZ NREC --tag T
  CODE_DIR: a code dir whose engine_fields_v16.py is v16c (fa84dad3), with engine_fields_v4/v12/v14 and engine_fields_kcache_v1."""
import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time

import numpy as np

TH = {"E_rel": 1e-9, "root": 3.4e-7, "minNv": 4.6e-9, "field": 1e-7, "floor_field": 1e-5}


def sanitize(info):
    out = {}
    for k, v in info.items():
        if callable(v) or k == "src":
            continue
        if isinstance(v, np.ndarray):
            out[k] = hashlib.md5(np.ascontiguousarray(v).tobytes()).hexdigest() + str(v.shape) + str(v.dtype)
        else:
            out[k] = v
    return json.dumps(out, sort_keys=True, default=lambda x: repr(x))


def mx(a, b):
    return float(np.max(np.abs(np.asarray(a, float) - np.asarray(b, float))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir"); ap.add_argument("code_dir"); ap.add_argument("v16d"); ap.add_argument("fields"); ap.add_argument("nrec", type=int)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    sys.path.insert(0, a.code_dir)
    import engine_fields_v16 as F16C
    import engine_fields_v4 as F4
    import engine_fields_v12 as F12
    import engine_fields_v14 as F14
    import engine_fields_kcache_v1 as KC
    md5c = hashlib.md5(open(F16C.__file__, "rb").read()).hexdigest()
    assert md5c.startswith("fa84dad3"), "the code dir's engine_fields_v16.py is not v16c: %s" % md5c
    spec = importlib.util.spec_from_file_location("engine_fields_v16d", a.v16d)
    F16D = importlib.util.module_from_spec(spec); spec.loader.exec_module(F16D)
    md5d = hashlib.md5(open(a.v16d, "rb").read()).hexdigest()
    KC.install([F4, F12, F14, F16C, F16D])
    assert F16D._RD2["route"] == "ivp" and not F16D._RD2["warm"], "QI2_FIELDS_* set in the environment"
    os.makedirs(a.out_dir, exist_ok=True)
    z = np.load(a.fields, allow_pickle=True)
    R = np.asarray(z["R"], float); g = float(z["g"]); ell = float(z["ell"])
    solver = json.loads(str(z["solver"]))
    cp = solver.get("collar_policy", "flag"); sr = float(solver.get("support_rel", 1e-14))
    nT = len(z["T"]); idx = list(range(nT - a.nrec, nT))
    head = ("rd2_field_gate %s: %s, %d sites, ell %g, g %g, collar_policy %s, support_rel %g, records %d..%d (T %.6g..%.6g); v16c %s, v16d %s; threads %s; python %s numpy %s"
            % (a.tag, os.path.basename(a.fields), len(R), ell, g, cp, sr, idx[0], idx[-1], float(z["T"][idx[0]]), float(z["T"][idx[-1]]), md5c, md5d,
               os.environ.get("OMP_NUM_THREADS"), sys.version.split()[0], np.__version__))
    print(head, flush=True)
    lines = [head]

    def solve(mod, eps, J, rtol, atol):
        t0 = time.perf_counter()
        Nn, vn, Wn, info = mod.solve_constraints_ivp(eps, J, g, R, ell, N0=None, rtol=rtol, atol=atol, collar_policy=cp, support_rel=sr)
        t1 = time.perf_counter()
        if info.get("failed"):
            return {"failed": info["failed"], "solve_s": t1 - t0, "info": info}
        Na, va, _ = mod.site_fields_kernel_average(info, R, ell)
        t2 = time.perf_counter()
        tz = mod.trapping_radius(Na, va, R, info)
        return {"failed": None, "N": Nn, "v": vn, "W": Wn, "Na": Na, "va": va, "info": info, "tz": tz, "solve_s": t1 - t0, "avg_s": t2 - t1,
                "E": float(info["E_collar"]), "minNv_sites": float(np.min(Na - np.abs(va)))}

    def compare(X, Y):
        """X against Y on the budget's lines (Y the reference)."""
        d = {}
        d["E_rel"] = abs(X["E"] - Y["E"]) / abs(Y["E"])
        zx, zy = X["tz"]["zeros"], Y["tz"]["zeros"]
        d["roots_n"] = [len(zx), len(zy)]
        d["root"] = max([abs(p - q) for p, q in zip(zx, zy)], default=0.0) if len(zx) == len(zy) else float("inf")
        d["minNv_dense"] = abs(X["tz"]["min"] - Y["tz"]["min"])
        d["minNv_sites"] = abs(X["minNv_sites"] - Y["minNv_sites"])
        d["minNv"] = max(d["minNv_dense"], d["minNv_sites"])
        d["N_sites"] = mx(X["N"], Y["N"]); d["v_sites"] = mx(X["v"], Y["v"]); d["N_avg"] = mx(X["Na"], Y["Na"]); d["v_avg"] = mx(X["va"], Y["va"])
        d["field"] = max(d["N_sites"], d["v_sites"], d["N_avg"], d["v_avg"])
        d["gauge_equal"] = X["info"]["gauge"]["label"] == Y["info"]["gauge"]["label"]
        d["regime_equal"] = X["info"]["regime"]["label"] == Y["info"]["regime"]["label"]
        d["pass"] = bool(d["E_rel"] <= TH["E_rel"] and d["root"] <= TH["root"] and d["minNv"] <= TH["minNv"] and d["field"] <= TH["field"]
                         and d["gauge_equal"] and d["regime_equal"])
        return d

    F16D.rd2_configure("collocation", "on")
    F16D._RD2["route"] = "ivp"; F16D._RD2["warm"] = False
    # warm-up (untimed, the prototype's): the kernel cache and the averaging matrices; the warm cache emptied after it
    e0, j0 = np.asarray(z["eps_tot"][idx[0]], float), np.asarray(z["J_tot"][idx[0]], float)
    solve(F16C, e0, j0, 1e-12, 1e-16)
    F16D._RD2["route"] = "collocation"; solve(F16D, e0, j0, 1e-12, 1e-16)
    F16D._RD2["route"] = "ivp"; F16D._RD2["last"] = None
    s_w = F16D.rd2_stats()
    for k_ in ("collocation", "collocation_s", "newton_iterations"):
        F16D._RD2["stats"][k_] -= s_w[k_]
    recs = []
    a_fail = []; b_fail = []; w_fail = []
    for i in idx:
        eps = np.asarray(z["eps_tot"][i], float); J = np.asarray(z["J_tot"][i], float)
        rec = {"i": i, "T": float(z["T"][i])}
        P12 = solve(F16C, eps, J, 1e-12, 1e-16)
        F16D._RD2["route"] = "ivp"; F16D._RD2["warm"] = False
        D = solve(F16D, eps, J, 1e-12, 1e-16)
        F16D._RD2["route"] = "collocation"; F16D._RD2["warm"] = False
        s0 = F16D.rd2_stats()
        C = solve(F16D, eps, J, 1e-12, 1e-16)
        F16D._RD2["warm"] = True
        Wm = solve(F16D, eps, J, 1e-12, 1e-16)
        s1 = F16D.rd2_stats()
        # RD-5 inside a record: a nearby solve (the sources moved by 1e-3 of the change from the previous record, the size of a Newton or J.v
        # step), cold (Cp) and warm-started from C's solution of this record (Wp)
        ip = max(i - 1, 0)
        dE_ = eps - np.asarray(z["eps_tot"][ip], float); dJ_ = J - np.asarray(z["J_tot"][ip], float)
        F16D._RD2["warm"] = False
        Cp = solve(F16D, eps + 1e-3 * dE_, J + 1e-3 * dJ_, 1e-12, 1e-16)
        F16D._RD2["last"] = None; F16D._RD2["warm"] = True
        solve(F16D, eps, J, 1e-12, 1e-16)
        Wp = solve(F16D, eps + 1e-3 * dE_, J + 1e-3 * dJ_, 1e-12, 1e-16)
        F16D._RD2["route"] = "ivp"; F16D._RD2["warm"] = False
        P14 = solve(F16C, eps, J, 1e-14, 1e-18)
        rec["timing_C"] = (C["info"].get("fields_route") or {}).get("timing")
        rec["pair"] = {"Cp_it": (Cp["info"].get("fields_route") or {}).get("newton_iterations"), "Wp_it": (Wp["info"].get("fields_route") or {}).get("newton_iterations"),
                       "Wp_warm": (Wp["info"].get("fields_route") or {}).get("warm_start"), "Cp_s": Cp["solve_s"], "Wp_s": Wp["solve_s"]}
        if not (Cp["failed"] or Wp["failed"]):
            rec["pair"]["Wp_vs_Cp"] = compare(Wp, Cp)
            if not rec["pair"]["Wp_vs_Cp"]["pass"]:
                w_fail.append(i)
        else:
            rec["pair"]["failed"] = [Cp["failed"], Wp["failed"]]; w_fail.append(i)
        rec["fallbacks"] = s1["fallback"] - s0["fallback"]
        rec["routes"] = {"C": (C["info"].get("fields_route") or {}).get("route"), "W": (Wm["info"].get("fields_route") or {}).get("route")}
        rec["newton_it"] = {"C": (C["info"].get("fields_route") or {}).get("newton_iterations"), "W": (Wm["info"].get("fields_route") or {}).get("newton_iterations"),
                            "W_warm": (Wm["info"].get("fields_route") or {}).get("warm_start")}
        rec["s"] = {k: (X["solve_s"], X.get("avg_s")) for k, X in (("P12", P12), ("D", D), ("C", C), ("W", Wm), ("P14", P14))}
        rec["nfev_P12"] = (P12["info"].get("ivp") or {}).get("nfev")
        # (a) bitwise
        if P12["failed"] or D["failed"]:
            ok_a = (P12["failed"] == D["failed"])
        else:
            ok_a = all(np.asarray(D[k]).tobytes() == np.asarray(P12[k]).tobytes() for k in ("N", "v", "W", "Na", "va"))
            ok_a &= all(np.asarray(D["info"][k]).tobytes() == np.asarray(P12["info"][k]).tobytes() for k in ("N_nodes", "v_nodes"))
            ok_a &= np.float64(D["E"]).tobytes() == np.float64(P12["E"]).tobytes()
            rf = P12["info"]["rfine"]
            ok_a &= all(np.asarray(D["info"][k](rf)).tobytes() == np.asarray(P12["info"][k](rf)).tobytes() for k in ("N_dense", "v_dense", "W_field", "U_dense"))
            ok_a &= sanitize(D["info"]) == sanitize(P12["info"])
            ok_a &= json.dumps(D["tz"], sort_keys=True) == json.dumps(P12["tz"], sort_keys=True)
        rec["a_bitwise"] = bool(ok_a)
        if not ok_a:
            a_fail.append(i)
        # (b)
        if P12["failed"]:
            rec["B"] = "P12 failed: %s" % P12["failed"]
        elif C["failed"]:
            rec["B"] = "C failed where P12 succeeds: %s" % C["failed"]; b_fail.append(i)
        else:
            rec["B1"] = compare(C, P12)
            rec["B2"] = compare(C, P14) if not P14["failed"] else None
            rec["P12_vs_P14"] = compare(P12, P14) if not P14["failed"] else None
            zN = np.asarray(z["N"][i], float); zv = np.asarray(z["v"][i], float)
            rec["B3"] = {"C": max(mx(C["Na"], zN), mx(C["va"], zv)), "P12": max(mx(P12["Na"], zN), mx(P12["va"], zv))}
            rec["B3"]["pass"] = bool(rec["B3"]["C"] <= TH["floor_field"] and rec["B3"]["P12"] <= TH["floor_field"])
            okb = rec["B1"]["pass"] and (rec["B2"] is not None and rec["B2"]["pass"]) and rec["B3"]["pass"]
            rec["B_pass"] = bool(okb)
            if not okb:
                b_fail.append(i)
            if Wm["failed"]:
                rec["W"] = "W failed: %s" % Wm["failed"]; w_fail.append(i)
            else:
                rec["W_vs_C"] = compare(Wm, C)
                if not rec["W_vs_C"]["pass"]:
                    w_fail.append(i)
        B1 = rec.get("B1") or {}; B2 = rec.get("B2") or {}; PP = rec.get("P12_vs_P14") or {}
        line = ("rec %d T %.6g: (a) %s | routes C %s W %s, Newton it C %s W %s (warm %s), fallbacks %d | P12 %.3f s (nfev %s) C %.3f s W %.3f s P14 %.3f s; averaging %.3f s"
                % (i, rec["T"], "BITWISE" if ok_a else "DIFFER", rec["routes"]["C"], rec["routes"]["W"], rec["newton_it"]["C"], rec["newton_it"]["W"], rec["newton_it"]["W_warm"],
                   rec["fallbacks"], P12["solve_s"], rec["nfev_P12"], C["solve_s"], Wm["solve_s"], P14["solve_s"], C.get("avg_s") or 0.0))
        if B1:
            line += (" | B1 C-P12: E %.1e root %.1e (n %s) minNv %.1e field %.1e (Navg %.1e vavg %.1e) %s | B2 C-P14: E %.1e root %.1e minNv %.1e field %.1e %s | P12-P14: E %.1e field %.1e | B3 vs stored: C %.1e P12 %.1e %s | W-C: E %.1e field %.1e %s"
                     % (B1["E_rel"], B1["root"], B1["roots_n"], B1["minNv"], B1["field"], B1["N_avg"], B1["v_avg"], "PASS" if B1["pass"] else "FAIL",
                        B2.get("E_rel", float("nan")), B2.get("root", float("nan")), B2.get("minNv", float("nan")), B2.get("field", float("nan")), "PASS" if B2.get("pass") else "FAIL",
                        PP.get("E_rel", float("nan")), PP.get("field", float("nan")), rec["B3"]["C"], rec["B3"]["P12"], "PASS" if rec["B3"]["pass"] else "FAIL",
                        (rec.get("W_vs_C") or {}).get("E_rel", float("nan")), (rec.get("W_vs_C") or {}).get("field", float("nan")), "PASS" if (rec.get("W_vs_C") or {}).get("pass") else "FAIL"))
        else:
            line += " | " + str(rec.get("B"))
        pr = rec["pair"]
        line += " | pair: Newton it cold %s warm %s (warm %s), %.3f -> %.3f s, Wp-Cp E %.1e field %.1e %s | C timing %s" % (
            pr["Cp_it"], pr["Wp_it"], pr["Wp_warm"], pr["Cp_s"], pr["Wp_s"], (pr.get("Wp_vs_Cp") or {}).get("E_rel", float("nan")),
            (pr.get("Wp_vs_Cp") or {}).get("field", float("nan")), "PASS" if (pr.get("Wp_vs_Cp") or {}).get("pass") else "FAIL",
            " ".join("%s %.3f" % (k_, v_) for k_, v_ in (rec["timing_C"] or {}).items()))
        print(line, flush=True); lines.append(line)
        recs.append(rec)
    ok = [r for r in recs if "B1" in r]

    def agg(key, sub):
        vals = [r[key][sub] for r in ok if r.get(key)]
        return float(np.max(vals)) if vals else None
    summ = {"records": len(recs), "a_bitwise": len(recs) - len(a_fail), "a_fail": a_fail, "b_fail": b_fail, "w_fail": w_fail,
            "fallbacks": int(sum(r["fallbacks"] for r in recs)),
            "B1_fail_records": [r["i"] for r in ok if not r["B1"]["pass"]], "B2_fail_records": [r["i"] for r in ok if not (r["B2"] or {}).get("pass")],
            "B3_fail_records": [r["i"] for r in ok if not r["B3"]["pass"]]}
    for key in ("B1", "B2", "P12_vs_P14", "W_vs_C"):
        for sub in ("E_rel", "root", "minNv", "field", "N_avg", "v_avg"):
            summ["%s_%s_max" % (key, sub)] = agg(key, sub)
    summ["B3_C_max"] = float(np.max([r["B3"]["C"] for r in ok])) if ok else None
    summ["B3_P12_max"] = float(np.max([r["B3"]["P12"] for r in ok])) if ok else None
    for k in ("P12", "C", "W", "P14"):
        summ["%s_solve_mean_s" % k] = float(np.mean([r["s"][k][0] for r in recs]))
    summ["factor_C"] = summ["P12_solve_mean_s"] / summ["C_solve_mean_s"]
    summ["factor_W"] = summ["P12_solve_mean_s"] / summ["W_solve_mean_s"]
    summ["newton_it_C_mean"] = float(np.mean([r["newton_it"]["C"] for r in recs if r["newton_it"]["C"] is not None])) if recs else None
    summ["newton_it_W_mean"] = float(np.mean([r["newton_it"]["W"] for r in recs if r["newton_it"]["W"] is not None])) if recs else None
    summ["pair_it_cold_mean"] = float(np.mean([r["pair"]["Cp_it"] for r in recs if r["pair"]["Cp_it"] is not None]))
    summ["pair_it_warm_mean"] = float(np.mean([r["pair"]["Wp_it"] for r in recs if r["pair"]["Wp_it"] is not None]))
    summ["pair_cold_mean_s"] = float(np.mean([r["pair"]["Cp_s"] for r in recs])); summ["pair_warm_mean_s"] = float(np.mean([r["pair"]["Wp_s"] for r in recs]))
    for sub in ("E_rel", "field", "minNv", "root"):
        vals = [r["pair"]["Wp_vs_Cp"][sub] for r in recs if r["pair"].get("Wp_vs_Cp")]
        summ["pair_Wp_vs_Cp_%s_max" % sub] = float(np.max(vals)) if vals else None
    summ["verdict_a"] = "PASS" if not a_fail else "FAIL"
    summ["verdict_b"] = "PASS" if not b_fail else "FAIL"
    summ["verdict_b_lines"] = {"B1": "PASS" if not summ["B1_fail_records"] else "FAIL", "B2": "PASS" if not summ["B2_fail_records"] else "FAIL",
                               "B3": "PASS" if not summ["B3_fail_records"] else "FAIL"}
    summ["verdict_rd5_field"] = "PASS" if not w_fail else "FAIL"
    lines.append("SUMMARY %s" % json.dumps(summ))
    print(lines[-1], flush=True)
    with open(os.path.join(a.out_dir, "rd2_field_gate_%s.txt" % a.tag), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    with open(os.path.join(a.out_dir, "rd2_field_gate_%s.json" % a.tag), "w") as fh:
        json.dump({"summary": summ, "records": recs, "args": vars(a), "thresholds": TH, "stats_v16d": F16D.rd2_stats()}, fh, indent=1, default=float)


if __name__ == "__main__":
    main()
