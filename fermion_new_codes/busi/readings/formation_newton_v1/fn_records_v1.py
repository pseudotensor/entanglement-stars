#!/usr/bin/env python3
"""busi reading formation_newton_v1 (reviewer A): the record-level reading (json and logs only; no linear algebra).
Sections: 1 inputs (md5, size, mtime, record counts, clocks, heads, args); 2 the production base (to40 json against the from-0 member's
own json); 3 record by record, Newton against production, at the matched clocks 0 to 12.5; 4 the first root, the zero crossing and
R_t at the matched clocks (the paper's criteria); 5 certification of the 101 Newton records under the production rule (and the
production's own residuals over 0 to 12.5); 6 the logs against the jsons (both runs) and the two log-only records; 7 timing and the
provenance files (LAUNCH, STOPPER, mirror and pull logs). Output: records.json and records_tables.md in OUT."""
import datetime, json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fn_common import *

OUTJ = {}
L = []  # markdown tables


def say(*a):
    print(*a, flush=True)


def utc(ts):
    return datetime.datetime.fromtimestamp(ts, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------- 1. inputs
say("== 1. inputs")
paths = {"newton_json_final": NJ, "newton_json_readout": NJ_READOUT, "newton_json_early_mirror": NJ_EARLY, "newton_fields": NF, "newton_log_final": NLOG,
         "newton_log_early_mirror": NLOG_EARLY, "prod_to40_json": PJ, "prod_to40_fields": PF, "prod_to40_log": PLOG, "prod_from0_json": P0J,
         "prod_from0_fields": P0F, "prod_from0_log": P0LOG, "LAUNCH": ND + "/LAUNCH.txt", "STOPPER": ND + "/STOPPER.txt", "args_v1": ND + "/args_v1.txt",
         "dryparse_args": ND + "/dryparse_args.json", "readout_v1_py": ND + "/ops/readout_v1.py", "readout_v1_md": ND + "/readout/readout_v1.md",
         "rc_audit_v1_md": ND + "/readout/rc_audit_v1.md", "rc_audit_v1_json": ND + "/readout/rc_audit_v1.json",
         "MD5_card": ND + "/final/MD5_card.txt", "MD5_box": ND + "/final/MD5_box.txt", "pull_final_log": ND + "/final/pull_final_v1.log",
         "pull_missing_log": ND + "/final/pull_missing_v2.log", "mirror_log": ND + "/ops/mirror.log",
         "rc_audit_py": "<internal>/rc_audit_v1.py", "compare_v5": "<internal>/engine_ladder_compare_v5.py",
         "engine_v36nq": BOX + "/compositions/ISO128_formation_v1.1/code/engine_collapse_v36nq.py",
         "engine_fields_v16b": BOX + "/compositions/ISO128_formation_v1.1/code/engine_fields_v16b.py"}
inputs = {}
for k, p in paths.items():
    if os.path.exists(p):
        st = os.stat(p)
        inputs[k] = {"path": p, "bytes": st.st_size, "mtime_utc": utc(st.st_mtime), "md5": md5(p)}
    else:
        inputs[k] = {"path": p, "missing": True}
    say("  %-26s %s %12s %s %s" % (k, inputs[k].get("md5", "MISSING")[:8], inputs[k].get("bytes", "-"), inputs[k].get("mtime_utc", "-"), p))
OUTJ["inputs"] = inputs
# the final pull's md5 lists against my own pass (fn_md5_v1.sh wrote md5_final_mine.txt)
mine = {}
if os.path.exists(OUT + "/md5_final_mine.txt"):
    for line in open(OUT + "/md5_final_mine.txt"):
        h, f = line.split(None, 1)
        mine[f.strip()] = h
card = {}
for line in open(paths["MD5_card"]):
    h, f = line.split(None, 1)
    card[f.strip()] = h
box = {}
for line in open(paths["MD5_box"]):
    h, f = line.split(None, 1)
    box[f.strip()] = h
OUTJ["pull"] = {"card_list": len(card), "box_list_first_pull": len(box), "mine": len(mine),
                "mine_equals_card": (set(mine) == set(card) and all(mine[f] == card[f] for f in card)),
                "box_first_pull_differs_from_card": sorted(f for f in box if card.get(f) != box[f]),
                "box_first_pull_missing": sorted(set(card) - set(box)),
                "slices_on_card": sorted(f.split("_T")[-1][:-4] for f in card if f.startswith("states/")),
                "pull_final_log": open(paths["pull_final_log"]).read().strip().split("\n"),
                "pull_missing_log": open(paths["pull_missing_log"]).read().strip().split("\n")}
say("  pull: card list %d files, first pull's box list %d, mine %d, mine == card: %s; first pull differing: %s; first pull missing %d" % (
    len(card), len(box), len(mine), OUTJ["pull"]["mine_equals_card"], OUTJ["pull"]["box_first_pull_differs_from_card"], len(OUTJ["pull"]["box_first_pull_missing"])))

JN, JP, J0 = load(NJ), load(PJ), load(P0J)
JR = load(NJ_READOUT)
OUTJ["readout_copy_equals_final"] = (json.dumps(JR, sort_keys=True) == json.dumps(JN, sort_keys=True))
RN, RP, R0 = records(JN), records(JP), records(J0)
say("  records: newton %d (%.4g to %.4g), prod to40 %d (%.4g to %.4g), prod from0 %d (%.4g to %.4g); readout copy equals final json: %s" % (
    len(RN), RN[0]["T_over_rs"], RN[-1]["T_over_rs"], len(RP), RP[0]["T_over_rs"], RP[-1]["T_over_rs"], len(R0), R0[0]["T_over_rs"], R0[-1]["T_over_rs"], OUTJ["readout_copy_equals_final"]))
# clocks
TN = [ckey(r["T_over_rs"]) for r in RN]
TP = {ckey(r["T_over_rs"]): r for r in RP}
T0 = {ckey(r["T_over_rs"]): r for r in R0}
steps = sorted(set(round(b - a, 9) for a, b in zip(TN, TN[1:])))
clock = {"newton_n": len(TN), "newton_first": TN[0], "newton_last": TN[-1], "newton_steps": steps,
         "newton_T_abs_equals_64T": all(abs(fl(r["T"]) - RS * float(r["T_over_rs"])) < 1e-9 for r in RN),
         "prod_T_abs_equals_64T": all(abs(fl(r["T"]) - RS * float(r["T_over_rs"])) < 1e-9 for r in RP),
         "newton_clocks_all_in_prod": all(t in TP for t in TN), "newton_clocks_all_in_from0": all(t in T0 for t in TN),
         "expected_101": (len(TN) == 101 and TN == [i / 8 for i in range(101)]),
         "json_head_script": {"newton": JN.get("script"), "prod": JP.get("script"), "from0": J0.get("script")},
         "record_engine_strings": {"newton": sorted(set(str(r.get("engine")) for r in RN)), "prod": sorted(set(str(r.get("engine")) for r in RP[:101]))},
         "newton_engine_v35": sorted(set(str(r.get("engine_v35")) for r in RN)),
         "newton_v36b_pair_workers": sorted(set(str((r.get("v36b") or {}).get("pair_workers")) for r in RN)),
         "newton_resumed_flag_in_labels": [k for k in ("label", "label_v34") if k in JN["head"] and "restarted" in str(JN["head"][k])],
         "prod_resumed_flag_in_labels": [k for k in ("label", "label_v34") if k in JP["head"] and "restarted" in str(JP["head"][k])]}
OUTJ["clocks"] = clock
say("  clocks:", json.dumps(clock)[:900])
# heads
hn, hp = JN["head"], JP["head"]
head_cmp = {}
for k in sorted(set(hn) | set(hp)):
    a, b = hn.get(k), hp.get(k)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        head_cmp[k] = {"newton": a, "prod": b, "rel_diff": ((a - b) / abs(b) if b else (a - b))}
    elif k in ("label", "dictionary", "kernel", "label_v34"):
        head_cmp[k] = {"equal": (a == b), "newton_len": len(str(a)), "prod_len": len(str(b))}
    else:
        head_cmp[k] = {"equal": (json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)), "newton": str(a)[:200], "prod": str(b)[:200]}
OUTJ["head"] = head_cmp
say("  head E0: newton %r prod %r rel %.3g; last_change %r / %r; N_B %r / %r; passes %r / %r" % (hn["E0"], hp["E0"], head_cmp["E0"]["rel_diff"], hn["last_change"], hp["last_change"], hn["N_B"], hp["N_B"], hn["passes"], hp["passes"]))
# args
an, ap = JN["args"], JP["args"]
args_diff = {"only_newton": {k: an[k] for k in sorted(set(an) - set(ap))}, "only_prod": {k: ap[k] for k in sorted(set(ap) - set(an))},
             "differ": {k: {"newton": an[k], "prod": ap[k]} for k in sorted(set(an) & set(ap)) if json.dumps(an[k], sort_keys=True) != json.dumps(ap[k], sort_keys=True)},
             "equal": sorted(k for k in set(an) & set(ap) if json.dumps(an[k], sort_keys=True) == json.dumps(ap[k], sort_keys=True))}
delivery_rows = ["mix", "nk_maxiter", "nk_krylov", "nk_jvp", "nk_line_search", "nk_halvings", "nk_restarts", "nk_plain_fallback", "nk_host_release", "nk_set",
                 "iter_cache", "prep_memory", "end_above", "pair_workers", "dust_set", "dust_kappa_max", "dust_multiplicity", "T_over_rs", "resume_run",
                 "resume_slice", "slices_c128", "slice_dir", "out", "tag"]
listed = set(delivery_rows)
differing = set(args_diff["only_newton"]) | set(args_diff["only_prod"]) | set(args_diff["differ"])
args_diff["delivery_rows"] = delivery_rows
args_diff["differing_not_listed_by_delivery"] = sorted(differing - listed)
args_diff["listed_by_delivery_but_equal_or_absent"] = sorted(listed - differing)
args_diff["from0_args_equal_to_to40_except"] = sorted(k for k in set(J0["args"]) | set(ap) if json.dumps(J0["args"].get(k), sort_keys=True) != json.dumps(ap.get(k), sort_keys=True))
OUTJ["args"] = args_diff
say("  args: only newton %s; only prod %s; differ %s; equal %d keys; not listed by the delivery: %s; listed but not differing: %s" % (
    sorted(args_diff["only_newton"]), sorted(args_diff["only_prod"]), sorted(args_diff["differ"]), len(args_diff["equal"]),
    args_diff["differing_not_listed_by_delivery"], args_diff["listed_by_delivery_but_equal_or_absent"]))
say("  from0 args differ from to40 in:", args_diff["from0_args_equal_to_to40_except"])

# ---------------------------------------------------------------- 2. the production base
say("== 2. production to40 json against the from-0 member's json")
common = sorted(set(TP) & set(T0))
neq = [t for t in common if json.dumps(TP[t], sort_keys=True) != json.dumps(T0[t], sort_keys=True)]
base = {"common_clocks": len(common), "first": common[0], "last": common[-1], "differing": neq,
        "matched_0_to_12p5_identical": all(json.dumps(TP[t], sort_keys=True) == json.dumps(T0[t], sort_keys=True) for t in TN),
        "to40_head_label_v34": str(hp.get("label_v34"))[-400:], "to40_resume_slice": ap.get("resume_slice"), "to40_resume_run": ap.get("resume_run"),
        "from0_T_over_rs_arg": J0["args"].get("T_over_rs"), "from0_tag": J0["args"].get("tag"), "to40_tag": ap.get("tag")}
if neq:
    t = neq[0]
    a, b = TP[t], T0[t]
    base["first_differing_keys"] = sorted(k for k in set(a) | set(b) if json.dumps(a.get(k), sort_keys=True) != json.dumps(b.get(k), sort_keys=True))[:40]
    base["first_differing_EE0"] = {"to40": a.get("E_over_E0"), "from0": b.get("E_over_E0")}
OUTJ["base"] = base
say("  ", json.dumps(base)[:1200])

# ---------------------------------------------------------------- 3. record by record
say("== 3. record by record at the 101 matched clocks")
rows = []
for r in RN:
    t = ckey(r["T_over_rs"])
    p = TP[t]
    qn, qp = quantities(r), quantities(p)
    d = {"T": t, "EE0_n": qn["EE0"], "EE0_p": qp["EE0"], "dEE0": qn["EE0"] - qp["EE0"], "dE_abs": qn["E"] - qp["E"],
         "min_n": qn["minNv"], "min_p": qp["minNv"], "dmin": qn["minNv"] - qp["minNv"],
         "rmin_n": qn["r_minNv"], "rmin_p": qp["r_minNv"], "drmin": (qn["r_minNv"] - qp["r_minNv"]) if (qn["r_minNv"] is not None and qp["r_minNv"] is not None) else None,
         "Rt_n": qn["Rt"], "Rt_p": qp["Rt"], "dRt": (qn["Rt"] - qp["Rt"]) if (qn["Rt"] is not None and qp["Rt"] is not None) else None,
         "dRt_rel": ((qn["Rt"] - qp["Rt"]) / qp["Rt"]) if (qn["Rt"] is not None and qp["Rt"]) else None,
         "rin_n": qn["rin"], "rin_p": qp["rin"], "drin": (qn["rin"] - qp["rin"]) if (qn["rin"] is not None and qp["rin"] is not None) else None,
         "Rc_n": qn["Rc"], "Rc_p": qp["Rc"], "dRc": qn["Rc"] - qp["Rc"], "Rsh_n": qn["Rsh"], "Rsh_p": qp["Rsh"],
         "Nmin_n": qn["Nmin"], "Nmin_p": qp["Nmin"], "dNmin": qn["Nmin"] - qp["Nmin"],
         "nzeros_n": qn["nzeros"], "nzeros_p": qp["nzeros"], "pockets_n": qn["pockets"], "pockets_p": qp["pockets"], "inv_n": qn["inversions"], "inv_p": qp["inversions"],
         "Rt_2gE_n": qn["Rt_2gE_json"], "Rt_2gE_p": qp["Rt_2gE_json"], "label_equal": (qn["label"] == qp["label"]), "label_n": qn["label"], "label_p": qp["label"],
         "converged_n": qn["converged"], "converged_p": qp["converged"], "it_n": qn["iterations"], "it_p": qp["iterations"], "eigh_n": qn["eigh"], "eigh_p": qp["eigh"],
         "sec_n": qn["seconds"], "sec_p": qp["seconds"], "mask_n": qn["mask_neg"], "mask_p": qp["mask_neg"]}
    rows.append(d)
OUTJ["rows"] = rows


def amax(key, rows=rows):
    best = max((x for x in rows if x[key] is not None), key=lambda x: abs(x[key]), default=None)
    return (best[key], best["T"]) if best else (None, None)


summ = {}
for key in ("dEE0", "dE_abs", "dmin", "drmin", "dRt", "dRt_rel", "drin", "dRc", "dNmin"):
    v, t = amax(key)
    summ[key] = {"max_abs": (abs(v) if v is not None else None), "signed": v, "at_T": t}
summ["nzeros_equal_all"] = all(x["nzeros_n"] == x["nzeros_p"] for x in rows)
summ["pockets_equal_all"] = all(x["pockets_n"] == x["pockets_p"] for x in rows)
summ["inversions_equal_all"] = all(x["inv_n"] == x["inv_p"] for x in rows)
summ["labels_equal_all"] = all(x["label_equal"] for x in rows)
summ["labels_differ_at"] = [x["T"] for x in rows if not x["label_equal"]]
summ["Rsh_equal_all"] = all(x["Rsh_n"] == x["Rsh_p"] for x in rows)
summ["Rsh_differ_at"] = [(x["T"], x["Rsh_n"], x["Rsh_p"]) for x in rows if x["Rsh_n"] != x["Rsh_p"]]
summ["mask_differ_at"] = [x["T"] for x in rows if x["mask_n"] != x["mask_p"]]
summ["pockets_total"] = {"n": sum(x["pockets_n"] or 0 for x in rows), "p": sum(x["pockets_p"] or 0 for x in rows)}
summ["inversions_total"] = {"n": sum(x["inv_n"] or 0 for x in rows), "p": sum(x["inv_p"] or 0 for x in rows)}
summ["iterations"] = {"n": sorted(set(x["it_n"] for x in rows)), "p": sorted(set(x["it_p"] for x in rows))}
summ["converged_all"] = {"n": all(x["converged_n"] for x in rows), "p": all(x["converged_p"] for x in rows)}
# the trend: the running maximum, the count of records where |d| falls against the previous record, the values at selected clocks
for key in ("dEE0", "dmin"):
    seq = [abs(x[key]) for x in rows]
    runmax = []
    m = 0.0
    for v in seq:
        m = max(m, v)
        runmax.append(m)
    falls = sum(1 for a, b in zip(seq, seq[1:]) if b < a)
    first_above = {}
    for thr in (1e-11, 1e-10, 1e-9, 1e-8, 1e-7):
        k = next((x["T"] for x, v in zip(rows, seq) if v > thr), None)
        first_above["%.0e" % thr] = k
    summ[key + "_trend"] = {"falls_against_previous": falls, "of_pairs": len(seq) - 1, "max_is_at_last_record": (seq[-1] == max(seq)),
                            "first_T_above": first_above, "at_T0": seq[0], "at_0p125": seq[1],
                            "samples": {("%.3f" % x["T"]): x[key] for x in rows if abs(x["T"] * 1 - round(x["T"])) < 1e-9 or x["T"] in (11.5, 12.125, 12.25, 12.375, 12.5)}}
OUTJ["summary"] = summ
say("  max |dE/E0| %.3e at %s; max |dmin| %.3e at %s; max |dRc| %.3e at %s; max |dRt| %.3e (rel %.3e) at %s; max |drin| %.3e at %s; max |dE| %.3e" % (
    summ["dEE0"]["max_abs"], summ["dEE0"]["at_T"], summ["dmin"]["max_abs"], summ["dmin"]["at_T"], summ["dRc"]["max_abs"], summ["dRc"]["at_T"],
    summ["dRt"]["max_abs"], summ["dRt_rel"]["max_abs"], summ["dRt"]["at_T"], summ["drin"]["max_abs"], summ["drin"]["at_T"], summ["dE_abs"]["max_abs"]))
say("  trend dEE0:", json.dumps(summ["dEE0_trend"])[:700])
say("  trend dmin:", json.dumps(summ["dmin_trend"])[:700])
say("  zeros/pockets/inversions/labels/Rsh equal on all:", summ["nzeros_equal_all"], summ["pockets_equal_all"], summ["inversions_equal_all"], summ["labels_equal_all"], summ["Rsh_equal_all"], "labels differ at", summ["labels_differ_at"][:10], "Rsh differ at", summ["Rsh_differ_at"][:10])
for x in rows:
    if x["T"] in (11.5, 12.0, 12.125, 12.25, 12.375, 12.5):
        say("   T %.3f dEE0 %+.3e dmin %+.3e dRt %s dRt_rel %s drin %s dRc %+.3e" % (x["T"], x["dEE0"], x["dmin"], fmt(x["dRt"], "%+.3e"), fmt(x["dRt_rel"], "%+.3e"), fmt(x["drin"], "%+.3e"), x["dRc"]))

# ---------------------------------------------------------------- 4. the first root, the zero crossing, matched R_t
say("== 4. first root, zero crossing, matched R_t")


def first_root(R):
    """the delivery's and the stopper's definition: the first record with a numeric R_t and min(N - |v|) < 0."""
    for r in R:
        q = quantities(r)
        if q["Rt"] is not None and q["minNv"] is not None and q["minNv"] < 0:
            return q
    return None


def first_with(R, pred):
    for r in R:
        q = quantities(r)
        if pred(q):
            return q
    return None


def zero_cross(R):
    Q = [quantities(r) for r in R]
    for a, b in zip(Q, Q[1:]):
        if a["minNv"] >= 0 > b["minNv"]:
            return a["T"] + (b["T"] - a["T"]) * a["minNv"] / (a["minNv"] - b["minNv"]), a, b
    return None, None, None


RP101 = [TP[t] for t in TN]
form = {}
for name, R in (("newton", RN), ("prod", RP101), ("prod_full_to40", RP), ("from0", [T0[t] for t in TN])):
    fr = first_root(R)
    fz = first_with(R, lambda q: q["nzeros"] > 0)
    fm = first_with(R, lambda q: q["minNv"] is not None and q["minNv"] < 0)
    zc, a, b = zero_cross(R)
    form[name] = {"first_root": fr, "first_with_zero": (fz["T"] if fz else None), "first_negative_min": (fm["T"] if fm else None), "zero_crossing_T": zc,
                  "zero_crossing_bracket": ((a["T"], a["minNv"], b["T"], b["minNv"]) if a else None)}
    if fr:
        say("  %s first root T %.4f R_t %.6f a = %.6f r_s = %.6f of 2gE (json %.6f; 2gE %.4f a) r_in %.6f a (%.5f r_s) min %.6f E/E0 %.8f E %.6f zeros %s; first zero %s; first negative %s; crossing %.5f" % (
            name, fr["T"], fr["Rt"], fr["Rt_rs"], fr["Rt_2gE"], fr["Rt_2gE_json"], fr["twogE"], fr["rin"], fr["rin_rs"], fr["minNv"], fr["EE0"], fr["E"], fr["zeros"], form[name]["first_with_zero"], form[name]["first_negative_min"], zc))
# matched clocks with a root in both
matched = []
for x in rows:
    if x["Rt_n"] is not None or x["Rt_p"] is not None:
        matched.append({"T": x["T"], "Rt_n": x["Rt_n"], "Rt_p": x["Rt_p"], "dRt_rel": x["dRt_rel"], "rin_n": x["rin_n"], "rin_p": x["rin_p"], "drin_rel": ((x["rin_n"] - x["rin_p"]) / x["rin_p"]) if (x["rin_n"] is not None and x["rin_p"]) else None,
                        "within_10pct": (abs(x["dRt_rel"]) <= 0.10) if x["dRt_rel"] is not None else None, "Rt_2gE_n": x["Rt_2gE_n"], "Rt_2gE_p": x["Rt_2gE_p"]})
form["matched_clocks"] = matched
fn, fp = form["newton"]["first_root"], form["prod"]["first_root"]
form["criteria"] = {"clock_dT": (fn["T"] - fp["T"]) if (fn and fp) else None, "clock_agree_0p5": (abs(fn["T"] - fp["T"]) <= 0.5) if (fn and fp) else None,
                    "Rt_all_within_10pct": all(m["within_10pct"] for m in matched if m["within_10pct"] is not None) and bool(matched),
                    "max_abs_dRt_rel_matched": max((abs(m["dRt_rel"]) for m in matched if m["dRt_rel"] is not None), default=None),
                    "all_matched_rel_below_1e-5": all(abs(m["dRt_rel"]) < 1e-5 for m in matched if m["dRt_rel"] is not None),
                    "all_matched_rel_below_1e-6": all(abs(m["dRt_rel"]) < 1e-6 for m in matched if m["dRt_rel"] is not None),
                    "zero_crossing_dT": (form["newton"]["zero_crossing_T"] - form["prod"]["zero_crossing_T"]) if (form["newton"]["zero_crossing_T"] and form["prod"]["zero_crossing_T"]) else None}
OUTJ["formation"] = form
for m in matched:
    say("   matched T %.3f R_t %s / %s rel %s; r_in %s / %s rel %s; R_t/2gE %s / %s" % (m["T"], fmt(m["Rt_n"], "%.6f"), fmt(m["Rt_p"], "%.6f"), fmt(m["dRt_rel"], "%+.3e"), fmt(m["rin_n"], "%.6f"), fmt(m["rin_p"], "%.6f"), fmt(m["drin_rel"], "%+.3e"), fmt(m["Rt_2gE_n"], "%.7f"), fmt(m["Rt_2gE_p"], "%.7f")))
say("  criteria:", json.dumps(form["criteria"]))

# ---------------------------------------------------------------- 5. certification
say("== 5. certification under the production rule")
crows = []
for r in RN:
    q = newton_residuals(r)
    crows.append({"T": ckey(r["T_over_rs"]), "kept": q["kept"], "conf": q["conf"], "kept_evaluation": q["kept_evaluation"], "evaluations": q["evaluations"],
                  "meets_stop_rule": q["meets_stop_rule"], "failed_solve": q["failed_solve"], "status": q["status"], "n_solves": q["n_solves"], "dE": q["dE"],
                  "kept_hist": q["kept_hist"], "conf_hist": q["conf_hist"], "dE_hist": q["dE_hist"], "how": q["how"],
                  "hist_len": len(r.get("iteration_history") or []), "converged": r.get("converged"), "unconverged": r.get("unconverged")})
ext = certify(crows)
ident = {"kept_equals_history_entry": all(x["kept"] is not None and x["kept_hist"] is not None and x["kept"] == x["kept_hist"] for x in crows),
         "conf_equals_following_entry": all(x["conf"] is not None and x["conf_hist"] is not None and x["conf"] == x["conf_hist"] for x in crows),
         "dE_equals_following_entry": all(x["dE"] is not None and x["dE_hist"] is not None and x["dE"] == x["dE_hist"] for x in crows),
         "kept_not_equal_at": [x["T"] for x in crows if not (x["kept"] is not None and x["kept"] == x["kept_hist"])],
         "conf_not_equal_at": [x["T"] for x in crows if not (x["conf"] is not None and x["conf"] == x["conf_hist"])],
         "n_solves_set": sorted(set(x["n_solves"] for x in crows)), "status_set": sorted(set(str(x["status"]) for x in crows)),
         "meets_stop_rule_all": all(x["meets_stop_rule"] for x in crows), "failed_solve_any": [x["T"] for x in crows if x["failed_solve"]],
         "evaluations_set": sorted(set(x["evaluations"] for x in crows)), "evaluations_total": sum(x["evaluations"] for x in crows),
         "converged_all": all(x["converged"] for x in crows), "unconverged_any": [x["T"] for x in crows if x["unconverged"]],
         "hist_len_set": sorted(set(x["hist_len"] for x in crows)), "kept_evaluation_set": sorted(set(x["kept_evaluation"] for x in crows)),
         "kept_gt_conf_count": sum(1 for x in crows if x["kept"] > x["conf"]), "conf_gt_kept_at": [x["T"] for x in crows if x["conf"] > x["kept"]]}
cert = {"extent": ext, "identities": ident, "rows": crows,
        "named": {("%.3f" % x["T"]): {"kept": x["kept"], "conf": x["conf"], "evaluations": x["evaluations"], "counts": x["counts"], "band": x["band"]} for x in crows if x["T"] in (0.0, 11.5, 12.0, 12.125, 12.25, 12.375, 12.5)},
        "n_kept_above_1e-4": sum(1 for x in crows if x["kept"] > BAND), "n_kept_above_1e-3": sum(1 for x in crows if x["kept"] > CAP),
        "n_conf_above_1e-4": sum(1 for x in crows if x["conf"] > BAND), "n_conf_above_1e-3": sum(1 for x in crows if x["conf"] > CAP),
        "median_conf": sorted(x["conf"] for x in crows)[len(crows) // 2], "median_kept": sorted(x["kept"] for x in crows)[len(crows) // 2]}
# the certification of the production's own records over 0 to 12.5 (plain)
prows = []
for t in TN:
    r = TP[t]
    q = plain_residuals(r)
    prows.append({"T": t, "kept": q["kept"], "conf": q["conf"], "how": q["how"], "converged": q["converged"], "iterations": q["iterations"], "hist_len": len(r.get("iteration_history") or [])})
pext = certify(prows)
cert["production_0_to_12p5"] = {"extent": pext, "rows": prows, "unconverged_at": [x["T"] for x in prows if not x["converged"]],
                                "n_kept_above_floor": sum(1 for x in prows if x["kept"] is not None and x["kept"] > FLOOR),
                                "kept_at_named": {("%.3f" % x["T"]): x["kept"] for x in prows if x["T"] in (0.0, 11.5, 12.0, 12.125, 12.25, 12.375, 12.5)},
                                "iterations_set": sorted(set(x["iterations"] for x in prows))}
OUTJ["certification"] = cert
say("  newton extent:", json.dumps(ext))
say("  identities:", json.dumps(ident)[:900])
say("  named:", json.dumps(cert["named"]))
say("  production 0..12.5:", json.dumps(cert["production_0_to_12p5"]["extent"]), "unconverged at", cert["production_0_to_12p5"]["unconverged_at"], "kept above 1e-6:", cert["production_0_to_12p5"]["n_kept_above_floor"], "iterations", cert["production_0_to_12p5"]["iterations_set"])
say("  production kept at named:", json.dumps(cert["production_0_to_12p5"]["kept_at_named"]))

# ---------------------------------------------------------------- 6. the logs
say("== 6. logs against the jsons")
LN = parse_log(NLOG)
LNE = parse_log(NLOG_EARLY)
L0 = parse_log(P0LOG)
LP = parse_log(PLOG)
logs = {"newton_final": {"lines": LN["n_lines"], "ee0_records": len(LN["ee0"]), "newton_lines": len(LN["newton"]), "unparsed_T_lines": LN["unparsed_T_lines"][:10],
                         "first": min(LN["ee0"]), "last": max(LN["ee0"]), "log_only_clocks": sorted(set(LN["ee0"]) - set(TN))},
        "newton_early_mirror": {"lines": LNE["n_lines"], "ee0_records": len(LNE["ee0"]), "last": max(LNE["ee0"]) if LNE["ee0"] else None},
        "prod_from0": {"lines": L0["n_lines"], "ee0_records": len(L0["ee0"]), "newton_lines": len(L0["newton"]), "first": min(L0["ee0"]), "last": max(L0["ee0"]), "unparsed_T_lines": L0["unparsed_T_lines"][:10]},
        "prod_to40": {"lines": LP["n_lines"], "ee0_records": len(LP["ee0"]), "first": (min(LP["ee0"]) if LP["ee0"] else None), "last": (max(LP["ee0"]) if LP["ee0"] else None)}}


def cmp_log(R, LOG, newton):
    out = {"n": 0, "EE0_7dec_equal": 0, "Rt_3dec_equal": 0, "rin_3dec_equal": 0, "min_4dec_equal": 0, "Rc_2dec_equal": 0, "pockets_equal": 0, "inversions_equal": 0,
           "it_equal": 0, "eigh_equal": 0, "mask_equal": 0, "kept_3sig_equal": 0, "conf_3sig_equal": 0, "evals_equal": 0, "dE_2sig_equal": 0, "missing_in_log": [], "mismatch": []}
    for r in R:
        t = ckey(r["T_over_rs"])
        e = LOG["ee0"].get(t)
        if e is None:
            out["missing_in_log"].append(t)
            continue
        q = quantities(r)
        out["n"] += 1
        ok = {"EE0_7dec_equal": ("%.7f" % q["EE0"]) == ("%.7f" % e["EE0"]),
              "Rt_3dec_equal": (q["Rt"] is None and e["Rt"] is None) or (q["Rt"] is not None and e["Rt"] is not None and ("%.3f" % q["Rt"]) == ("%.3f" % e["Rt"])),
              "rin_3dec_equal": (q["rin"] is None and e["rin"] is None) or (q["rin"] is not None and e["rin"] is not None and ("%.3f" % q["rin"]) == ("%.3f" % e["rin"])),
              "min_4dec_equal": ("%.4f" % q["minNv"]) == ("%.4f" % e["minNv"]), "Rc_2dec_equal": ("%.2f" % q["Rc"]) == ("%.2f" % e["Rc"]),
              "pockets_equal": q["pockets"] == e["pockets"], "inversions_equal": q["inversions"] == e["inversions"], "it_equal": q["iterations"] == e["it"],
              "eigh_equal": q["eigh"] == e["eigh"], "mask_equal": (str(q["mask_neg"]) == e["mask"])}
        if newton:
            n = LOG["newton"].get(t)
            qq = newton_residuals(r)
            if n is not None:
                ok.update({"kept_3sig_equal": ("%.2e" % qq["kept"]) == ("%.2e" % n["kept"]), "conf_3sig_equal": ("%.2e" % qq["conf"]) == ("%.2e" % n["conf"]),
                           "evals_equal": qq["evaluations"] == n["evaluations"], "dE_2sig_equal": ("%.1e" % qq["dE"]) == ("%.1e" % n["dE"])})
        for k, v in ok.items():
            if v:
                out[k] += 1
            else:
                out["mismatch"].append((t, k, str(q.get(k.split("_")[0]))[:30], str(e)[:0]))
    return out


logs["newton_json_vs_log"] = cmp_log(RN, LN, True)
logs["prod_json_vs_from0_log"] = cmp_log(RP101, L0, False)
logs["log_only_records"] = {("%.3f" % t): {k: LN["ee0"][t][k] for k in ("EE0", "Rt", "rin", "minNv", "Rc", "Rsh", "pockets", "inversions", "mask", "it", "eigh", "twogE")} for t in logs["newton_final"]["log_only_clocks"]}
logs["log_only_newton_lines"] = {("%.3f" % t): LN["newton"].get(t) for t in logs["newton_final"]["log_only_clocks"]}
# the log's residual summary over the 103 records (the delivery reads only the 101)
logs["newton_log_kept_max"] = max((v["kept"], t) for t, v in LN["newton"].items())
logs["newton_log_conf_max"] = max((v["conf"], t) for t, v in LN["newton"].items())
logs["newton_log_evaluations"] = sorted(set(v["evaluations"] for v in LN["newton"].values()))
logs["newton_log_kinds"] = sorted(set(v["kind"] + " | " + v["status"] for v in LN["newton"].values()))
logs["newton_log_seconds_sum_103"] = sum(v["seconds"] for v in LN["newton"].values())
# the projection lines (the engine's own clock) and the pair-worker line
proj = []
for line in open(NLOG, errors="replace"):
    if line.startswith("projection at") or line.startswith("version 36b"):
        proj.append(line.strip()[:200])
logs["projection_lines"] = proj
OUTJ["logs"] = logs
say("  ", json.dumps({k: v for k, v in logs.items() if k in ("newton_final", "newton_early_mirror", "prod_from0", "prod_to40")}))
say("  newton json vs log:", json.dumps(logs["newton_json_vs_log"])[:800])
say("  prod json vs from0 log:", json.dumps(logs["prod_json_vs_from0_log"])[:800])
say("  log-only records:", json.dumps(logs["log_only_records"]))
say("  log kept max %s conf max %s evaluations %s kinds %s" % (logs["newton_log_kept_max"], logs["newton_log_conf_max"], logs["newton_log_evaluations"], logs["newton_log_kinds"]))

# ---------------------------------------------------------------- 7. timing and provenance
say("== 7. timing and provenance")
launch = open(ND + "/LAUNCH.txt").read()
stopper = open(ND + "/STOPPER.txt").read()
mirror = open(ND + "/ops/mirror.log", errors="replace").read()
m_start = re.search(r"START (\S+) pid (\d+)", launch)
m_stopper = re.search(r"stopper (\d+)", launch)
rc_lines = [l for l in mirror.split("\n") if " rsync rc " in l]
first_fail = next((l for l in rc_lines if not l.endswith("rc 0")), None)
sec_sum = sum(fl(r.get("seconds")) or 0.0 for r in RN)
prov = {"launch_start": (m_start.group(1) if m_start else None), "engine_pid": (m_start.group(2) if m_start else None), "stopper_pid": (m_stopper.group(1) if m_stopper else None),
        "launch_env": next((l for l in launch.split("\n") if l.startswith("env ")), None), "launch_md5_lines": [l for l in launch.split("\n") if re.match(r"^[0-9a-f]{32} ", l)],
        "launch_has_nice": ("nice" in launch), "stopper_lines": stopper.strip().split("\n")[:2] + ["..."],
        "stopper_formation_line": next((l for l in stopper.split("\n") if "FIRST FORMATION" in l), None),
        "stopper_has_stop_line": any("stopping engine" in l or "engine 6428 stopped" in l for l in stopper.split("\n")),
        "mirror_rc_lines": len(rc_lines), "mirror_first_nonzero_rc": first_fail, "mirror_last_ok": next((l for l in reversed(rc_lines) if l.endswith("rc 0")), None),
        "mirror_tail": mirror.strip().split("\n")[-2:],
        "json_record_seconds_sum": sec_sum, "head_seconds_prep": fl(hn.get("seconds")), "mean_record_seconds": sec_sum / len(RN),
        "engine_wall_from_notes_s": (datetime.datetime(2026, 10, 21, 5, 24, 59) - datetime.datetime(2026, 10, 21, 4, 54, 9)).total_seconds(),
        "fields_final_mtime_utc": inputs["newton_fields"]["mtime_utc"], "json_final_mtime_utc": inputs["newton_json_final"]["mtime_utc"],
        "slice_mtimes_utc": {}}
for f in sorted(os.listdir(ND + "/final/states")):
    prov["slice_mtimes_utc"][f.split("_T")[-1][:-4]] = utc(os.stat(ND + "/final/states/" + f).st_mtime)
prov["engine_wall_card_hours"] = prov["engine_wall_from_notes_s"] / 3600.0
prov["wall_per_record_s"] = prov["engine_wall_from_notes_s"] / len(RN)
OUTJ["provenance"] = prov
say("  ", json.dumps({k: v for k, v in prov.items() if k != "slice_mtimes_utc"})[:1800])
say("  slice mtimes:", json.dumps(prov["slice_mtimes_utc"]))

# ---------------------------------------------------------------- tables
L.append("## record by record (Newton - production), selected clocks")
L.append("| T/r_s | E/E0 N | E/E0 P | dE/E0 | min N | min P | d min | R_t N | R_t P | dR_t rel | r_in N | r_in P | kept | confirming | counts |")
L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
cmap = {x["T"]: x for x in crows}
for x in rows:
    if abs(x["T"] - round(x["T"])) < 1e-9 or x["T"] >= 11.5:
        c = cmap[x["T"]]
        L.append("| %.3f | %.7f | %.7f | %+.2e | %.6f | %.6f | %+.2e | %s | %s | %s | %s | %s | %.2e | %.2e | %s |" % (
            x["T"], x["EE0_n"], x["EE0_p"], x["dEE0"], x["min_n"], x["min_p"], x["dmin"], fmt(x["Rt_n"], "%.5f"), fmt(x["Rt_p"], "%.5f"), fmt(x["dRt_rel"], "%+.2e"),
            fmt(x["rin_n"], "%.5f"), fmt(x["rin_p"], "%.5f"), c["kept"], c["conf"], "yes" if c["counts"] else "NO"))
L.append("")
L.append("## matched clocks with a root")
L.append("| T/r_s | R_t Newton | R_t production | rel | r_in Newton | r_in production | rel | R_t/2gE N | R_t/2gE P |")
L.append("|---|---|---|---|---|---|---|---|---|")
for m in matched:
    L.append("| %.3f | %s | %s | %s | %s | %s | %s | %s | %s |" % (m["T"], fmt(m["Rt_n"], "%.6f"), fmt(m["Rt_p"], "%.6f"), fmt(m["dRt_rel"], "%+.2e"), fmt(m["rin_n"], "%.6f"), fmt(m["rin_p"], "%.6f"), fmt(m["drin_rel"], "%+.2e"), fmt(m["Rt_2gE_n"], "%.7f"), fmt(m["Rt_2gE_p"], "%.7f")))
L.append("")
L.append("## Newton residuals (all 101)")
L.append("| T/r_s | kept | confirming | evaluations | kept eval | stop rule | counts | band |")
L.append("|---|---|---|---|---|---|---|---|")
for c in crows:
    L.append("| %.3f | %.3e | %.3e | %d | %d | %s | %s | %s |" % (c["T"], c["kept"], c["conf"], c["evaluations"], c["kept_evaluation"], c["meets_stop_rule"], c["counts"], c["band"]))
with open(OUT + "/records_tables.md", "w") as f:
    f.write("\n".join(L) + "\n")
h = dump(OUTJ, OUT + "/records.json")
say("wrote", OUT + "/records.json", h, "and records_tables.md", md5(OUT + "/records_tables.md"))
