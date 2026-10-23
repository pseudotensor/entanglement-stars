#!/usr/bin/env python
"""base_rate_v1.py (busi; ReviewA_v16e_v1.md erratum v3 sec. 2 (ii)): the number of UNCONVERGED records per route at 4096 and 2048
sites over every stored record of the listed runs. A record is the run's own when its clock is after the run's restart clock
(head.resume_slice_v34.T_over_rs; a production counts from its first Newton record, its own earlier restarts included). Per record:
converged flag, kept residual (the last solve's kept_residual), Newton iterations, evaluations, eighs, negative centre (gauge label
'flat-slice gauge continued'), the solve status. Writes OUT/base_rate.json and OUT/tables/<label>.txt; prints the summary.
Usage: base_rate_v1.py OUTDIR"""
import json
import os
import sys

B = "<internal>/perf_redesign"; C = "<internal>/solve_a_bulk"
TAG = "lad_ISO_128_s4_nkqh2_m0p2_rs128_n4096_R04_k10_tau8_ell4_prep4_A_R2_avg_newton_v36nq"
RUNS = [  # label, size, route, json, own-records rule ("all" = every Newton record; "after" = after the restart clock)
    ("ISO_128 production (pod6; nkq to 3.0, nkqh2 from 3.0625)", 4096, "parent (v36nq dense + ivp)", C + "/iso128_h2/run_h2/collapse_%s.json" % TAG, "all"),
    ("job (b) line (reopen2/iso128, v36nq_rd2 on v16d, pw 2)", 4096, "v16d (dense + collocation v16d)", C + "/reopen2/iso128/run/collapse_lad_ISO_128_s4_rd_m0p2_rs128_n4096_R04_k10_tau8_ell4_prep4_A_R2_avg_newton_v36nq_rd2.json", "after"),
    ("RD-1 entry D (rd13 defaults, h100-f)", 4096, "parent (v36nq dense + ivp)", B + "/RD1_build/entry_iso128/D/collapse_rd13_D.json", "after"),
    ("RD-1 entry B (rd13 --eigh banded, h100-f)", 4096, "parent + banded eigh (ivp)", B + "/RD1_build/entry_iso128/B/collapse_rd13_B.json", "after"),
    ("MERGE-1 D0 (merged defaults, h100-f)", 4096, "parent (v36nq dense + ivp)", B + "/v16e_hold/card_merge/iso/D0/collapse_mf_D0.json", "after"),
    ("MERGE-1 Fd (dense + collocation v16e)", 4096, "v16e (dense + collocation v16e)", B + "/v16e_hold/card_merge/iso/Fd/collapse_mf_Fd.json", "after"),
    ("MERGE-1 Fd5 (Fd to 5.125)", 4096, "v16e (dense + collocation v16e)", B + "/v16e_hold/card_merge/iso/Fd5/collapse_mf_Fd5.json", "after"),
    ("MERGE-1 A (banded + cache + collocation v16e)", 4096, "v16e + banded", B + "/v16e_hold/card_merge/iso/A/collapse_mf_A.json", "after"),
    ("MERGE-1 A2 (A + pw 2)", 4096, "v16e + banded", B + "/v16e_hold/card_merge/iso/A2/collapse_mf_A2.json", "after"),
    ("MERGE-1 AB (banded + collocation v16e)", 4096, "v16e + banded", B + "/v16e_hold/card_merge/iso/AB/collapse_mf_AB.json", "after"),
    ("RUNG2c40_v36q production (H100, v36q)", 2048, "parent (v36q dense + ivp)", C + "/ladders/r1/runs/RUNG2c40_v36q/collapse_RUNG2c40_v36q.json", "all"),
    ("RUNG2c40b_srf production", 2048, "parent (srf line)", C + "/ladders/r1/runs/RUNG2c40b_srf/collapse_RUNG2c40b_srf.json", "all"),
    ("P64_s4 production (qi2_h100_1, v36nq + H2, pw 2)", 2048, "parent (v36nq dense + ivp)", C + "/p64_newton_h100_1/final/run/collapse_lad_P64_s4_newton_v36nq_h100_1.json", "all"),
    ("RD-2 S1 I (v37a_rd2 defaults, h100-f)", 2048, "parent", B + "/RD2_build/h100f/v2/S1/I/collapse_RUNG2c40b_srf_rd2I.json", "after"),
    ("RD-2 S1 C (collocation v16d)", 2048, "v16d", B + "/RD2_build/h100f/v2/S1/C/collapse_RUNG2c40b_srf_rd2C.json", "after"),
    ("RD-2 S1 W (collocation v16d + warm start)", 2048, "v16d", B + "/RD2_build/h100f/v2/S1/W/collapse_RUNG2c40b_srf_rd2W.json", "after"),
    ("RD-2 S2 I (v36nq_rd2 defaults, h100-f)", 2048, "parent", B + "/RD2_build/h100f/v2/S2/I/collapse_RUNG2c40_v36q_rd2I.json", "after"),
    ("RD-2 S2 C (collocation v16d)", 2048, "v16d", B + "/RD2_build/h100f/v2/S2/C/collapse_RUNG2c40_v36q_rd2C.json", "after"),
    ("RD-2 S2 W (collocation v16d + warm start)", 2048, "v16d", B + "/RD2_build/h100f/v2/S2/W/collapse_RUNG2c40_v36q_rd2W.json", "after"),
    ("RD-2 box replay I (S1 segment)", 2048, "parent", B + "/RD2_build/replay/I/collapse_RUNG2c40b_srf_rd2I.json", "after"),
    ("RD-2 box replay C (S1 segment)", 2048, "v16d", B + "/RD2_build/replay/C/collapse_RUNG2c40b_srf_rd2C.json", "after"),
    ("RD-2 box replay W (S1 segment)", 2048, "v16d", B + "/RD2_build/replay/W/collapse_RUNG2c40b_srf_rd2W.json", "after"),
    ("RD-2b S2 I (defaults on v16e = v16d's, h100-f)", 2048, "parent", B + "/RD2b_build/h100f/S2/I/collapse_RUNG2c40_v36q_rd2bI.json", "after"),
    ("RD-2b S2 C (collocation v16e)", 2048, "v16e", B + "/RD2b_build/h100f/S2/C/collapse_RUNG2c40_v36q_rd2bC.json", "after"),
    ("RD-1 P64 replay D (rd13 defaults, box)", 2048, "parent", B + "/RD1_build/replay64/D/collapse_rd13_D.json", "after"),
    ("RD-1 P64 replay B (banded + sternheimer, box)", 2048, "parent + banded + sternheimer", B + "/RD1_build/replay64/B/collapse_rd13_B.json", "after"),
    ("MERGE-1 P64 G0 (merged defaults, box)", 2048, "parent", B + "/MERGE_build/replay64/G0/collapse_mf_G0.json", "after"),
    ("MERGE-1 P64 GA (banded + cache + collocation v16e, box)", 2048, "v16e + banded", B + "/MERGE_build/replay64/GA/collapse_mf_GA.json", "after"),
]


def scan(js, rule):
    d = json.load(open(js))
    rc = ((d.get("head") or {}).get("resume_slice_v34") or {}).get("T_over_rs")
    out = []
    for r in d["records"]:
        t = r.get("T_over_rs")
        if t is None:
            continue
        sv = (r.get("solve_v32") or {}).get("solves") or []
        if rule == "after" and rc is not None and t <= rc + 1e-9:
            continue
        if rule == "all" and not sv and r.get("converged") is None:
            continue
        s = sv[-1] if sv else {}
        out.append({"T": t, "converged": r.get("converged"), "kept": s.get("kept_residual"), "it": r.get("iterations"),
                    "ev": sum(int(x.get("evaluations") or 0) for x in sv) if sv else None, "eigh": r.get("eigendecompositions"),
                    "negc": ((r.get("gauge") or {}).get("label") or "").startswith("flat-slice gauge continued"),
                    "status": s.get("status"), "newton": bool(sv), "mix": (d.get("args") or {}).get("mix")})
    return out, rc, d.get("args") or {}


def main():
    O = sys.argv[1]; os.makedirs(O + "/tables", exist_ok=True)
    res = []
    for lab, n, route, js, rule in RUNS:
        if not os.path.exists(js):
            print("MISSING", lab, js); continue
        rows, rc, args = scan(js, rule)
        newton = [x for x in rows if x["newton"]]
        unc = [x for x in rows if x["converged"] is False]
        negc = [x for x in rows if x["negc"]]
        kept = [x["kept"] for x in rows if isinstance(x["kept"], float)]
        conv_kept = sorted(x["kept"] for x in rows if x["converged"] is True and isinstance(x["kept"], float))
        e = {"label": lab, "size": n, "route": route, "json": js, "restart_clock": rc, "records": len(rows), "newton_records": len(newton),
             "T_first": rows[0]["T"] if rows else None, "T_last": rows[-1]["T"] if rows else None, "unconverged": len(unc),
             "negative_centre_records": len(negc), "unconverged_negative_centre": sum(1 for x in unc if x["negc"]),
             "unconverged_records": [(x["T"], x["kept"], x["it"], x["ev"], x["negc"], x["status"]) for x in unc],
             "kept_max_converged": conv_kept[-1] if conv_kept else None, "kept_median_converged": conv_kept[len(conv_kept) // 2] if conv_kept else None,
             "converged_kept_above_1e-8": sum(1 for k in conv_kept if k > 1e-8), "mix": args.get("mix"), "fields": args.get("fields"),
             "eigh": args.get("eigh"), "pair_workers": args.get("pair_workers"), "nsites": args.get("nsites")}
        res.append(e)
        with open(O + "/tables/%s.txt" % lab.split(" (")[0].replace(" ", "_").replace("/", "_"), "w") as f:
            f.write("%s\n%s\n" % (lab, js))
            for x in rows:
                f.write("%8.4f conv %-5s kept %-10s it %-3s ev %-3s eigh %-4s negc %-5s %s\n" % (x["T"], x["converged"], ("%.3e" % x["kept"]) if isinstance(x["kept"], float) else x["kept"],
                        x["it"], x["ev"], x["eigh"], x["negc"], (x["status"] or "")[:120]))
        print("%-62s %4d %-34s records %3d (T %s..%s) UNCONVERGED %d; negative-centre records %d (of them unconverged %d); converged kept median %s max %s, >1e-8: %d" % (
            lab, n, route, len(rows), e["T_first"], e["T_last"], len(unc), len(negc), e["unconverged_negative_centre"],
            "%.1e" % e["kept_median_converged"] if e["kept_median_converged"] else None, "%.1e" % e["kept_max_converged"] if e["kept_max_converged"] else None, e["converged_kept_above_1e-8"]))
        for x in unc:
            print("      UNCONVERGED T %.4f kept %.3e it %s ev %s negative centre %s: %s" % (x["T"], x["kept"] or -1, x["it"], x["ev"], x["negc"], (x["status"] or "")[:150]))
    json.dump(res, open(O + "/base_rate.json", "w"), indent=1)


main()
