#!/usr/bin/env python
"""Task 3 report, parts (a) and (d) (SPEC_task3_prescribed_flow.md secs. 4 and 7; builder analyst): the tables from flow_chart_v2/v3's
records in one folder, every number emitted here.  A1 (the symbol: C1, the largest group velocity at the surface per channel mass, the
turning table), A2 (per carrier and launch: the measured initial centre velocity against the packet-averaged symbol; the measured turning
radius from the sign change against the law and the spec's budget), A3 (the measured velocity against 9/8, the packet-averaged symbol and
max_k v_g), D1 (the dust centre against the marginally bound law, the width and the carrier, until the crossing of r_s; the budget
sigma_x |v'| t + t/(2 sigma_x m)), D2 (the released packet against the symbol's ODE), D3 (per frequency: the norm inside r_s and beyond
R_ref = 2 r_s along the run, the largest inside, the final returned fraction, and from v3 the short-wave share inside r_s), D4 (the
band-top packet's speed against w along the path).  Args: --folder --prefix-v3 (the v3 tags) --json --md.
Version 2: the velocity table lists every A2 record of the folder (version 1 listed only the v3_-prefixed ones); the checks file name is
--checks (default checks_v4)."""
import argparse
import glob
import json
import math
import os

import numpy as np


def f(v, d=5):
    if v is None:
        return "-"
    v = float(v)
    return "%.*g" % (d, v) if (v == 0 or 1e-3 <= abs(v) < 1e5) else "%.3e" % v


def load(folder, tag):
    p = os.path.join(folder, tag + ".json")
    return json.load(open(p)) if os.path.exists(p) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True)
    ap.add_argument("--json", required=True)
    ap.add_argument("--checks", default="checks_v4")
    ap.add_argument("--md", required=True)
    a = ap.parse_args()
    F = a.folder
    out = {}
    L = ["# Task 3 parts (a) and (d): the turning law, the bound, the dust and the ingoing packets (flow_task3_report_v1.py; prescribed flow "
         "(simpler version); PG radial; v declared)", ""]
    ck = load(F, a.checks)
    if ck:
        out["checks"] = ck
        L += ["## Unit checks (flow_chart_v4.py checks)", "",
              "- Hermiticity %s; C0 (the paper's chain_generator, every eigenvalue twice) %s; C1 (the ring against the symbol, largest) %s; C2 "
              "(the ramp product's unitarity) %s; the sea's idempotency %s; the flat cut at nx = 128, nk = 24: %s against the record's verify.py "
              "SA %s (difference %s)." % (f(ck["hermiticity"], 3), f(ck["C0_chain_generator_twice"], 3), f(max(ck["C1_ring"].values()), 3),
                                         f(ck["C2_unitarity"], 3), f(ck["sea_idempotency"], 3), f(ck["flat_cut_nx128_nk24"], 11),
                                         f(ck["record_verify_SA"], 11), f(ck["flat_cut_minus_record"], 3)), ""]
    a1 = load(F, "A1")
    if a1:
        out["A1"] = {"vg_max_surface": a1["vg_max_surface"], "bound_check": a1["bound_check_9_over_8"]}
        L += ["## A1, the symbol (the periodic uniform-flow ring; the tables)", "",
              "- C1 largest |eigenvalue - symbol| %s; max_k v_g at w = 1: %s (9/8 - max at m_c = 0: %s)." % (
                  f(max(a1["C1_ring_max_error"].values()), 3), "; ".join("%s: %s at k = %s" % (k, f(v[0], 10), f(v[1], 6)) for k, v in a1["vg_max_surface"].items()),
                  f(-a1["bound_check_9_over_8"], 3)),
              "- R_turn/r_s by carrier: " + "; ".join("%s: %s" % (mc, ", ".join("k %s -> %s" % (f(k, 4), f(r, 5)) for k, r in zip(t["k"], t["R_turn_over_rs"])))
                                                    for mc, t in a1["turning_table"].items()) + ".", ""]
    # A2
    L += ["## A2, the turning radius (the sign change of the measured initial velocity along the launch scan)", "",
          "| record | r_s | m | kappa_j | carrier k0 | sigma_x | R_turn law | R_turn measured | difference | spec budget |", "|---|---|---|---|---|---|---|---|---|---|"]
    out["A2"] = {}
    for p in sorted(glob.glob(os.path.join(F, "*A2*.json"))):
        d = json.load(open(p))
        tag = os.path.basename(p)[:-5]
        out["A2"][tag] = {"turning": d["turning"], "rows": [{k: v for k, v in r.items() if k != "centres"} for r in d["rows"]]}
        for k0, t in d["turning"].items():
            L.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (tag, f(d["rs"]), f(d["mass"]), f(d["kappa_j"]), f(float(k0), 6),
                                                                      f([r["sigma_x"] for r in d["rows"] if abs(r["k0"] - float(k0)) < 1e-6][0]),
                                                                      f(t["law"], 6), f(t["measured"], 6), f(t["measured"] - t["law"], 4), f(t.get("budget"), 4)))
    L += ["", "| record | carrier k0 | R_launch | v measured (0 to 4) | v symbol (packet-averaged at R_launch) | difference |", "|---|---|---|---|---|---|"]
    for tag, d in out["A2"].items():
        for r in d["rows"]:
            if "v_measured" in r:
                L.append("| %s | %s | %s | %s | %s | %s |" % (tag, f(r["k0"], 5), f(r["R_launch"], 5), f(r["v_measured"], 5), f(r["v_symbol_packet"], 5),
                                                           f(r["v_measured"] - r["v_symbol_packet"], 3)))
    # A3
    L += ["", "## A3, the bound (the packet at k0 = arccos(1/4), sigma_x = 16, launched at R = r_s)", ""]
    for tag in ("A3_m0", "A3_m0.1"):
        d = load(F, tag)
        if d:
            out[tag] = {k: d[k] for k in ("v_measured", "v_symbol_packet", "max_k_vg", "bound_9_8", "mass", "rs")}
            L.append("- %s (r_s %s): measured %s; the packet-averaged symbol at R_launch %s; max_k v_g at R_launch %s; 9/8 = %s." % (
                tag, f(d["rs"]), f(d["v_measured"], 7), f(d["v_symbol_packet"], 7), f(d["max_k_vg"], 7), f(9 / 8, 7)))
    # D1, D2
    L += ["", "## D1, the dust at p = 0 (r_s = 32, R_m = 8, released at 4 r_s), and D2, the packet at rest in the lattice frame", ""]
    for tag in ("D1_m0.5", "D1_m0.1", "D2_m0.5", "D2_m0.1"):
        d = load(F, tag)
        if not d:
            continue
        tr = d["track"]
        T = np.array([x["t"] for x in tr])
        C = np.array([x["centre"] for x in tr])
        if tag.startswith("D1"):
            ref = np.array(d["dust_law"])
            rs = d["rs"]
            sel = ref >= rs
            m = d["mass"]
            dv = np.array([abs(math.sqrt(rs) / (2 * r ** 1.5)) for r in ref])
            budget = d["sigma_x"] * dv * T + T / (2 * d["sigma_x"] * m)
            dev = np.abs(C - ref)
            rel = dev / ref
            out[tag] = {"max_rel_dev_outside_rs": float(rel[sel].max()), "max_dev_over_budget_outside_rs": float((dev[sel] / np.maximum(budget[sel], 1e-12)).max()),
                        "carrier_max_outside_rs": float(max(abs(x["carrier"]) for x, s in zip(tr, sel) if s)), "sigma_k": 1 / (2 * d["sigma_x"]),
                        "crossing_time_law": d["crossing_time_law"]}
            L.append("- %s: until the law reaches r_s (t <= %s): the largest |centre - law|/law %s, the largest |centre - law|/budget %s, the largest |carrier| %s "
                     "(sigma_k/4 = %s); every 20 time units: %s." % (
                         tag, f(d["crossing_time_law"], 5), f(out[tag]["max_rel_dev_outside_rs"], 3), f(out[tag]["max_dev_over_budget_outside_rs"], 3),
                         f(out[tag]["carrier_max_outside_rs"], 3), f(1 / (8 * d["sigma_x"]), 3),
                         "; ".join("t %s: %s (law %s, width %s, carrier %s)" % (f(T[i], 4), f(C[i], 5), f(ref[i], 5), f(tr[i]["width"], 4), f(tr[i]["carrier"], 3))
                                   for i in range(0, len(T), max(1, int(round(20 / (T[1] - T[0]))))))))
        else:
            ode = np.array(d["symbol_ode_R"])
            rel = np.abs(C - ode) / ode
            out[tag] = {"p0": d["p0"], "max_rel_dev": float(rel.max())}
            L.append("- %s: p0 = %s; the largest |centre - ODE|/ODE %s; every 30 time units: %s." % (
                tag, f(d["p0"], 6), f(rel.max(), 3), "; ".join("t %s: %s (ODE %s, width %s)" % (f(T[i], 4), f(C[i], 5), f(ode[i], 5), f(tr[i]["width"], 4))
                                                               for i in range(0, len(T), max(1, int(round(30 / (T[1] - T[0]))))))))
    # D3
    L += ["", "## D3, the ingoing packets (built on the ingoing branch in the exterior; r_s = 32, R_m = 8, 1024 sites)", "",
          "| record | m | omega r_s | sigma_x | R_0 | largest norm inside r_s | its time | final norm beyond 2 r_s | final inside r_s | short-wave share inside r_s at the largest inside | at the end |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    out["D3"] = {}
    for tag in ("D3_m0", "D3_m0.1"):
        d = load(F, tag)
        if not d:
            continue
        for r in d["rows"]:
            if "track" not in r:
                L.append("| %s | %s | %s | - | - | %s | | | | | |" % (tag, f(d["mass"]), f(r["omega_rs"]), r.get("skipped")))
                continue
            tr = r["track"]
            ins = [x["inside_rs"] for x in tr]
            i = int(np.argmax(ins))
            sw = tr[i].get("inside_short_wave")
            swe = tr[-1].get("inside_short_wave")
            out["D3"]["%s_w%s" % (tag, r["omega_rs"])] = {"max_inside": ins[i], "t_max_inside": tr[i]["t"], "final_beyond_Rref": tr[-1]["beyond_Rref"],
                                                          "final_inside": tr[-1]["inside_rs"], "short_wave_at_max": sw, "short_wave_end": swe}
            L.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                tag, f(d["mass"]), f(r["omega_rs"]), f(r["sigma_x"], 4), f(r["R0"], 5), f(ins[i], 4), f(tr[i]["t"], 5), f(tr[-1]["beyond_Rref"], 4),
                f(tr[-1]["inside_rs"], 3), "%s of %s" % (f(sw, 3), f(ins[i], 3)) if sw is not None else "-",
                "%s of %s" % (f(swe, 3), f(tr[-1]["inside_rs"], 3)) if swe is not None else "-"))
    # D4
    for tag in ("D4_m0",):
        d = load(F, tag)
        if d:
            tr = d["track"]
            T = np.array([x["t"] for x in tr])
            C = np.array([x["centre"] for x in tr])
            sp = np.gradient(C, T)
            w = np.array(d["w_along"])
            sel = (C < 2 * d["rs"]) & (T > 0)
            out[tag] = {"speed_over_w": [float(x) for x in (sp[sel] / w[sel])]}
            L.append("- %s (the band-top packet k0 = pi/2 released at r_s/2): the centre's speed over w(centre) along the path inside 2 r_s: %s." % (
                tag, ", ".join(f(x, 3) for x in sp[sel] / w[sel])))
    json.dump(out, open(a.json, "w"), indent=1, default=float)
    open(a.md, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
