"""C5_v1d_compose_v1.py: reads the solve lane's dust_comparator_v1d runs (C5_v1d/s{2,4,8}/s*_<mode>_continuum_rest.json) and the checking
lane's runs (lead_api/check40j_c5/out_p8*/run_<mode>.json, outputs only), prints the two-route fold table, the delays against each
scale's own control, the shares of the engine's lags, and the per-channel fold census. Read-only on every input; writes compose_v1.json."""
import glob
import json
import os
import sys

B = sys.argv[1] if len(sys.argv) > 1 else "<internal>/C5_v1d"
F = "<internal>/check40j_c5"
REVIEWA = {  # (sigma, mode) -> the checking lane's run file (the runs its section 8 and addendum quote)
    (4, "control"): "out_p8/run_control.json", (4, "columns"): "out_p8f/run_columns.json", (4, "census"): "out_p8g/run_census.json",
    (4, "constant"): "out_p8g/run_constant_kk-1.json",
    (2, "control"): "out_p8h2/run_control.json", (2, "columns"): "out_p8i2/run_columns.json", (2, "census"): "out_p8h2/run_census.json",
    (2, "constant"): "out_p8h2/run_constant_kk-1.json",
    (8, "control"): "out_p8h8/run_control.json", (8, "columns"): "out_p8i8/run_columns.json", (8, "census"): "out_p8h8/run_census.json",
    (8, "constant"): "out_p8h8/run_constant_kk-1.json"}
DELIVERED = {2: (10.72795, 45.011, 10.7500, 47.900), 4: (10.81569, 43.794, 10.8750, 49.814), 8: (10.98791, 34.200, 11.0000, 40.165)}
ENGINE = {2: (12.000, 0.655), 4: (12.125, 0.661), 8: (12.750, 0.762)}
LAG = {2: 1.272, 4: 1.309, 8: 1.762}
MODES = ("control", "columns", "census", "constant")


def load(p):
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


def row(d):
    if d is None or d.get("fold") is None:
        return None
    fo, fr = d["fold"], d.get("first_record") or {}
    return {"T": fo["T_over_rs"], "R": fo["R"], "R2g": fo["R_over_2gE0"], "fr_T": fr.get("T_over_rs"), "fr_R": fr.get("outer"), "fr_R2g": fr.get("outer_over_2gE0"),
            "E0": d["E0"], "drift": d.get("E_drift_net_max"), "end": d.get("end_label"), "pins": (d.get("centre") or {}).get("pins", d.get("pins"))}


out = {"mine": {}, "reviewa": {}}
for s in (2, 4, 8):
    for md in MODES:
        mine = load(os.path.join(B, "s%d" % s, "s%d_%s_continuum_rest.json" % (s, md))) or load(os.path.join(B, "s%d_%s" % (s, md), "s%d_%s_continuum_rest.json" % (s, md)))
        fab = load(os.path.join(F, REVIEWA[(s, md)]))
        out["mine"]["%d_%s" % (s, md)] = mine and {"row": row(mine), "rest_sums": mine.get("rest_sums"), "per_channel": (mine.get("fold") or {}).get("per_channel"),
                                                   "count_inside": (mine.get("fold") or {}).get("count_fraction_inside")}
        out["reviewa"]["%d_%s" % (s, md)] = fab and {"row": row(fab), "sum_n_eps": fab.get("sum_n_eps_T0"),
                                                   "per_channel_inside": (fab.get("fold") or {}).get("count_inside_per_channel"),
                                                   "R_min_per_channel": (fab.get("fold") or {}).get("R_min_per_channel")}

print("TWO-ROUTE FOLD TABLE: mode | mine: T_f, R_f, R_f/2gE0, first record T, its R/2gE0, E0 | checking lane: the same | |dT_f|, |dR_f/2gE0|")
for s in (2, 4, 8):
    for md in MODES:
        m = (out["mine"].get("%d_%s" % (s, md)) or {}).get("row")
        f = (out["reviewa"].get("%d_%s" % (s, md)) or {}).get("row")
        fm = lambda r: ("%.5f %.3f %.4f %.3f %.4f %.6f" % (r["T"], r["R"], r["R2g"], r["fr_T"], r["fr_R2g"], r["E0"])) if r else "-"  # noqa: E731
        dd = ("%.1e %.1e" % (abs(m["T"] - f["T"]), abs(m["R2g"] - f["R2g"]))) if (m and f) else "-"
        print("s%d %-8s | %s | %s | %s" % (s, md, fm(m), fm(f), dd))

print("\nCONTROL REGRESSION against the delivered folds (T_f, R_f, first record T, its R):")
for s in (2, 4, 8):
    m = (out["mine"].get("%d_control" % s) or {}).get("row")
    if m:
        print("s%d: mine %.5f %.3f %.4f %.3f; delivered %.5f %.3f %.4f %.3f" % ((s, m["T"], m["R"], m["fr_T"], m["fr_R"]) + DELIVERED[s]))

print("\nDELAYS against each scale's own control (fold; first record) and shares of the engine's lag (fold convention; first-record convention):")
out["delays"] = {}
for s in (2, 4, 8):
    c = (out["mine"].get("%d_control" % s) or {}).get("row")
    for md in ("columns", "census", "constant"):
        m = (out["mine"].get("%d_%s" % (s, md)) or {}).get("row")
        if not (m and c):
            continue
        d = m["T"] - c["T"]
        dr = m["fr_T"] - c["fr_T"]
        lag_fr = ENGINE[s][0] - c["fr_T"]
        out["delays"]["%d_%s" % (s, md)] = {"fold": d, "first_record": dr, "share_fold": d / LAG[s], "share_first_record": dr / lag_fr if lag_fr else None,
                                            "remainder_fold": LAG[s] - d}
        print("s%d %-8s: delay %.4f r_s (share %.1f percent of %.3f; remainder %.3f); first record %.3f of %.3f (%.1f percent); engine root %.3f of its 2GE against %.4f (fold) %.4f (first record)" % (
            s, md, d, 100 * d / LAG[s], LAG[s], LAG[s] - d, dr, lag_fr, 100 * dr / lag_fr if lag_fr else float("nan"), ENGINE[s][1], m["R2g"], m["fr_R2g"]))

print("\nPER-CHANNEL at the fold (mine: count inside R_f / total, innermost R; checking lane: count inside, innermost R):")
for s in (2, 4, 8):
    for md in MODES:
        mi = out["mine"].get("%d_%s" % (s, md)) or {}
        fb = out["reviewa"].get("%d_%s" % (s, md)) or {}
        pc = mi.get("per_channel") or {}
        print("s%d %-8s mine: %s | sum n eps at rest %s | lane: inside %s, R_min %s" % (
            s, md, "; ".join("k%s %.2f/%.0f %.2f" % (k, v["count_inside"], v["count_total"], v["innermost_R"]) for k, v in pc.items()),
            ("%.5f" % mi["rest_sums"]["sum_n_eps"]) if mi.get("rest_sums") else "-",
            {k: round(v, 2) for k, v in (fb.get("per_channel_inside") or {}).items()}, {k: round(v, 2) for k, v in (fb.get("R_min_per_channel") or {}).items()}))
    for md in MODES:
        r = (out["mine"].get("%d_%s" % (s, md)) or {}).get("row")
        if r:
            print("   s%d %s: E drift net %.1e, pins %s, end: %s" % (s, md, r["drift"] or 0.0, r["pins"], r["end"]))

with open(os.path.join(B, "compose_v1.json"), "w") as f:
    json.dump(out, f, indent=1)
