#!/usr/bin/env python
"""ISO128_formation_readout_v1.py (solve lane, inbox #251 item 2; gate (ii) of the ISO_64 reconciler).
Reads stored files only (box CPU, light): the ISO_128_s4 H2 restart's run json (records 0 to 11.25), its log (records to 11.5), its
T0011.50 slice (N_applied, v_applied, R only), the T0005.00 mirror json, the restart and stop logs; applies the v1.3 producer's
production rule (ladders_readout_v1_3.load_member, unchanged) to ISO_128 and to the comparison members; prints matched-clock rows.
Usage: python ISO128_formation_readout_v1.py OUT_JSON"""
import hashlib
import json
import re
import sys
import zipfile

import numpy as np

sys.path.insert(0, "<internal>/qi2_ladders_code")
import ladders_readout_v1_3 as P  # noqa: E402
import engine_ladder_compare_v5 as LC  # noqa: E402

B = "<internal>/solve_a_bulk"
TAG = "lad_ISO_128_s4_nkqh2_m0p2_rs128_n4096_R04_k10_tau8_ell4_prep4_A_R2_avg_newton_v36nq"
H2 = B + "/iso128_h2/run_h2/collapse_%s.json" % TAG
H2LOG = B + "/iso128_h2/logs/%s.log" % TAG
SL115 = B + "/iso128_h2/states_h2/orbitals_%s_T0011.50.npz" % TAG
M5 = B + "/resume_mirror/pod6/%s/T0005.00/collapse_%s.json" % (TAG, TAG)
RESTART = B + "/iso128_h2<internal>/restart.log"
STOP = B + "/iso128_h2/STOP_AT_FORMATION_h2.txt"
L = B + "/ladders"
MEM = {
    "ISO_128_s4_plain": L + "/p6/runs/lad_ISO_128_s4_m0p2_rs128_n4096_R04_k10_tau8_ell4_prep4_A_R2_avg_plain_v36n/collapse_lad_ISO_128_s4_m0p2_rs128_n4096_R04_k10_tau8_ell4_prep4_A_R2_avg_plain_v36n.json",
    "S4_128_s4": L + "/r2/runs/lad_S4_128_s4_m0p2_rs128_n4096_R04_k10_tau8_ell4_prep4_A_R2_avg_plain_v36d/collapse_lad_S4_128_s4_m0p2_rs128_n4096_R04_k10_tau8_ell4_prep4_A_R2_avg_plain_v36d.json",
    "S4_128_s4_nk_v36m": L + "/p8/runs/lad_S4_128_s4_nk_m0p2_rs128_n4096_R04_k10_tau8_ell4_prep4_A_R2_avg_newton_v36m/collapse_lad_S4_128_s4_nk_m0p2_rs128_n4096_R04_k10_tau8_ell4_prep4_A_R2_avg_newton_v36m.json",
    "S4_128_s4_nk_v36q": L + "/r3/runs/lad_S4_128_s4_nk_m0p2_rs128_n4096_R04_k10_tau8_ell4_prep4_A_R2_avg_newton_v36q/collapse_lad_S4_128_s4_nk_m0p2_rs128_n4096_R04_k10_tau8_ell4_prep4_A_R2_avg_newton_v36q.json",
    "S4_128_s8": L + "/r1/runs/lad_S4_128_s8_m0p2_rs128_n4096_R04_k10_tau8_ell8_prep4_A_R2_avg_plain_v36d/collapse_lad_S4_128_s8_m0p2_rs128_n4096_R04_k10_tau8_ell8_prep4_A_R2_avg_plain_v36d.json",
    "S4_128_s16": L + "/p8/runs/lad_S4_128_s16_m0p2_rs128_n4096_R04_k10_tau8_ell16_prep4_A_R2_avg_plain_v36d/collapse_lad_S4_128_s16_m0p2_rs128_n4096_R04_k10_tau8_ell16_prep4_A_R2_avg_plain_v36d.json",
    "ISO_64_s4": B + "/isoB/collapse_isoB.json",
}
CLOCKS = [2.75, 2.875, 3.0, 3.5625, 11.0, 11.0625, 11.1875, 11.25, 11.5, 12.375]


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def canon(recs):
    return hashlib.md5(json.dumps(recs, sort_keys=True).encode()).hexdigest()


def member(name, path):
    meta, cols, ints, zeros = P.load_member(name, path, name, "", None)
    T = cols["T"]
    nc = np.where(ints["counts"] == 0)[0]
    ce = len(T) - 1 if len(nc) == 0 else nc[0] - 1
    first0 = next((i for i in range(len(T)) if zeros[i]), None)
    J = json.load(open(path))
    rows = {}
    for t in CLOCKS:
        k = np.where(np.abs(T - t) < 1e-9)[0]
        if len(k) == 0:
            continue
        i = int(k[0])
        z = zeros[i]
        rows[t] = {"EE0": cols["EE0"][i], "Rt_rs": cols["Rt_rs"][i], "Rt_2gE": cols["Rt_2gE"][i],
                   "rin_rs": (min(z) / meta["args"]["rs"] if z else None), "minNv": cols["minNv"][i], "counts": bool(ints["counts"][i]),
                   "certified": bool(i <= ce)}
    return {"md5": meta["md5"], "records": len(T), "T_last": T[-1], "cert_end": (T[ce] if ce >= 0 else None),
            "first_noncounting": (T[nc[0]] if len(nc) else None), "first_zero": (T[first0] if first0 is not None else None),
            "first_zero_certified": (first0 is not None and first0 <= ce),
            "first_zero_row": ({"Rt_rs": cols["Rt_rs"][first0], "Rt_2gE": cols["Rt_2gE"][first0], "EE0": cols["EE0"][first0],
                                "rin_rs": min(zeros[first0]) / meta["args"]["rs"], "minNv": cols["minNv"][first0]} if first0 is not None else None),
            "E0": J["head"]["E0"], "banded": [float(T[i]) for i in np.where(ints["banded"] == 1)[0]], "rows": rows,
            "res_max_cert": float(np.nanmax(cols["res"][:ce + 1])) if ce >= 0 else None,
            "conf_max_cert": float(np.nanmax(cols["conf"][:ce + 1])) if ce >= 0 and np.any(np.isfinite(cols["conf"][:ce + 1])) else None}


def main():
    out = {"inputs": {p: md5(p) for p in (H2, H2LOG, M5, RESTART, STOP, SL115)}}
    # (2) the certified history from the stored json, by the producer's rule
    out["ISO_128_s4"] = member("ISO_128_s4", H2)
    J = json.load(open(H2))
    R = J["records"]
    out["h2_records"] = [len(R), R[0]["T_over_rs"], R[-1]["T_over_rs"]]
    src = [("orig" if "plain_v36n_T0000.00" in (r.get("engine_v34") or "") else "h2" if "nkq_" in (r.get("engine_v34") or "") and "T0003.00" in (r.get("engine_v34") or "") else "?") for r in R]
    out["provenance"] = P.rle([(r["T_over_rs"], s) for r, s in zip(R, src)])
    out["head_resume_slice"] = {k: J["head"]["resume_slice_v34"].get(k) for k in ("slice", "run", "T_over_rs", "record_index", "slice_fields_vs_stored_max")}
    i3 = max(i for i, r in enumerate(R) if r["T_over_rs"] <= 3.0 + 1e-12)
    M = json.load(open(M5))["records"]
    out["records_0_to_3"] = {"n": i3 + 1, "canon_md5_final_json": canon(R[:i3 + 1]),
                             "canon_md5_T0005_mirror_json": canon(M[:i3 + 1]), "mirror_records": [len(M), M[-1]["T_over_rs"]],
                             "mirror_equal_to_5": canon(M) == canon(R[:len(M)])}
    per = []
    for r in R:
        sv = LC.v32_solve(r)
        per.append([r["T_over_rs"], LC.rec_residual(r, True), (sv or {}).get("confirming_residual") if sv else LC.confirm_residual(r),
                    r.get("converged"), LC.pockets(r), int(r.get("inversions_total") or 0)])
    out["none_residual_records"] = [[p[0], p[1], p[2], R[i].get("iteration_history"), R[i].get("converged"), R[i].get("unconverged")] for i, p in enumerate(per) if p[1] is None or p[2] is None]
    per = [[p[0], (p[1] if p[1] is not None else -1.0), (p[2] if p[2] is not None else -1.0)] + p[3:] for p in per]
    out["per_record_max"] = {"kept": max(p[1] for p in per), "kept_at": max(per, key=lambda p: p[1])[0],
                             "conf": max(p[2] for p in per), "conf_at": max(per, key=lambda p: p[2])[0],
                             "kept_0_3": max(p[1] for p in per if p[0] <= 3.0), "conf_0_3": max(p[2] for p in per if p[0] <= 3.0),
                             "kept_3_1125": max(p[1] for p in per if p[0] > 3.0), "conf_3_1125": max(p[2] for p in per if p[0] > 3.0),
                             "unconverged": [p[0] for p in per if p[3] is not True], "pockets": sum(p[4] for p in per),
                             "inversions": sum(p[5] for p in per), "record0": per[0]}
    # log cross-check of the json's records 3.0625 to 11.25, and the log's 11.3125 to 11.5
    txt = open(H2LOG).read()
    nl = {round(float(a), 3): (int(e), float(k), float(c)) for a, e, k, c in re.findall(
        r"^T/rs ([0-9.]+) newton .*?(\d+) evaluations in .*?kept residual ([0-9.e+-]+) \(evaluation \d+\), the confirming plain step's residual ([0-9.e+-]+)", txt, re.M)}
    el = {round(float(a), 3): float(b) for a, b in re.findall(r"^T/rs ([0-9.]+) E/E0 ([0-9.]+) ", txt, re.M)}
    ok = bad = 0
    for r in R:
        t = round(r["T_over_rs"], 3)
        if t in nl and r["T_over_rs"] > 3.0:
            sv = r["solve_v32"]
            same = ("%.2e" % nl[t][1] == "%.2e" % sv["kept_residual"] and "%.2e" % nl[t][2] == "%.2e" % sv["confirming_residual"]
                    and "%.7f" % el[t] == "%.7f" % r["E_over_E0"])
            ok += same
            bad += (not same)
    out["log_vs_json_3_to_11p25"] = {"equal": ok, "differ": bad}
    out["log_after_json"] = {t: {"newton": nl.get(t), "EE0": el.get(t)} for t in sorted(nl) if t > R[-1]["T_over_rs"] + 1e-9}
    m = re.search(r"^T/rs 11\.500 E/E0 .*$", txt, re.M)
    out["log_11p5_line"] = m.group(0)[:420]
    # (1) the 11.5 record recomputed from the T0011.50 slice's applied fields
    z = np.load(SL115)
    Rg, N, v = z["R"], z["N_applied"], z["v_applied"]
    f = N - np.abs(v)
    roots = []
    for i in range(len(f) - 1):
        if f[i] == 0 or f[i] * f[i + 1] < 0:
            roots.append(float(Rg[i] - f[i] * (Rg[i + 1] - Rg[i]) / (f[i + 1] - f[i])))
    E0, g = J["head"]["E0"], J["args"]["g"]
    out["slice_11p5"] = {"T": float(z["T"]), "rs": float(z["rs"]), "n": int(len(Rg)), "min_N_minus_absv": float(f.min()),
                         "r_at_min": float(Rg[int(np.argmin(f))]), "roots_linear": roots,
                         "two_g_E_from_log_EE0": 2 * g * E0 * 0.9940693, "E0": E0, "g": g}
    out["restart_log"] = open(RESTART).read()
    out["stop_file"] = open(STOP).read()[:1500]
    # (3) comparison members
    out["members"] = {}
    for k, p in MEM.items():
        try:
            out["members"][k] = member(k, p)
        except Exception as e:  # noqa: BLE001
            out["members"][k] = {"error": repr(e)}
    json.dump(out, open(sys.argv[1], "w"), indent=1, default=float)
    print(json.dumps(out, indent=1, default=float)[:60000])


if __name__ == "__main__":
    main()
