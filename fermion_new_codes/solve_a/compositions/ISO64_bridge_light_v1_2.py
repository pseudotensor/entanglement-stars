#!/usr/bin/env python
"""ISO64_bridge_light_v1_2.py (composition 4, v1.2, the solve lane): the J5 light-channel readout on the isotropic bridge history isoB
(ISO_64_s4 composed), extended from kappa 1 to every isoB channel final on the box (kappa 1, 2, 3, 4, 9, 12, 16). Conventions of
PAGE_solved_history_v4.1 T1 and of ISO64_bridge_light_v1.py (9156cd47): x_kappa the d13 'excess' S_X(C) - S_X(P_-(h)), weight 16 kappa,
nats; F_meas the post-pass f_meas (w1_postpass_v1f), weighted; F_apriori the 'excess_floor_high', weighted. B2 = the v6o_B2 production
channels (<internal>/prod), the same kappa at matched clocks. Runs no propagation; reads stored files only.
Also: the settings of every channel against kappa 1's, the partial resume-mirror states (listed, not used), and the resume-mirror
snapshots of kappa 2 and 9 against their finals at the common clocks.
Writes OUT/ISO64_light_v1_2.json and OUT/ISO64_light_tables_v1_2.md.
Usage: python ISO64_bridge_light_v1_2.py OUT_DIR"""
import glob
import hashlib
import json
import os
import sys

CODE_D13 = "<internal>/code_d13"
PP_DIR = "<internal>/w1"
os.environ.setdefault("W1_CODE_D13", CODE_D13)
for _p in (CODE_D13, "<internal>/code_v6", PP_DIR):
    sys.path.insert(0, _p)
import w1_postpass_v1f as PP  # noqa: E402

CUTS = ["2", "16", "24", "30", "48"]
TF = "<internal>/tourn2_r"
RM = "<internal>/resume_mirror"
ISO = {1: "<internal>/isoB_k1"}
for _k in (2, 3, 4, 9, 12, 16):
    ISO[_k] = "%s/isoB_k%d" % (TF, _k)
B2 = ("<internal>/prod", "v6o_B2")
MISSING = [5, 6, 7, 8, 10, 11, 13, 14, 15]
MAN = []
GAP_CLOCKS = [26.0, 28.0, 30.0, 32.0, 34.0]
# agreement rule (stated in the notes): agree if |dS| <= 0.10 max(|S_iso|, |S_B2|), or |dS| <= F_meas(iso) + F_meas(B2) (floor level)
REL_TOL = 0.10


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load(d, tag, k, label):
    ch = PP.load_channel(d, tag, k)
    if ch is None:
        raise SystemExit("missing %s %s %d" % (d, tag, k))
    for suf in ("summary.json", "d13.json", "d13.npz"):
        p = os.path.realpath(os.path.join(d, "%s_k%d_%s" % (tag, k, suf)))
        if os.path.exists(p):
            MAN.append({"label": "%s kappa %d %s" % (label, k, suf), "path": p, "md5": md5(p)})
    return ch


def cell(ch, t, lab):
    c = (((ch["clocks"].get(t) or {}).get("cuts") or {}).get(lab)) or {}
    if c.get("excess") is None:
        return None
    return {"x": c["excess"], "fmeas": PP.f_meas(ch, t, lab), "fhigh": c.get("excess_floor_high")}


def settings_view(sj):
    S = json.load(open(sj))
    st = dict(S.get("settings") or {})
    w1 = S.get("w1") or {}
    d13 = S.get("d13") or {}
    hist = dict(S.get("history") or {})
    v = {"code": S.get("code"), "final": S.get("final"), "end_reason": S.get("end_reason"), "n_records": len(S.get("records") or []),
         "last_record_T": None}
    recs = S.get("records") or []
    if recs and isinstance(recs[-1], dict):
        for key in ("t_over_rs", "T_over_rs", "T_rs", "t"):
            if key in recs[-1]:
                v["last_record_T"] = recs[-1][key]
                break
    for key, val in st.items():
        v["settings." + key] = val
    for key in ("propagation", "interp", "interp_sub", "sector", "sector_used", "sector_propagate", "cheb_tol", "T_init",
                "sector_check_clocks_over_rs", "fast_banded_probes"):
        v["w1." + key] = w1.get(key)
    for key in ("on", "inline", "cfg", "TH_ref", "formation", "columns_clocks_over_rs", "readout_clocks_over_rs", "vacuum_clocks_over_rs", "store"):
        v["d13." + key] = d13.get(key)
    hist.pop("records", None)
    for key, val in hist.items():
        v["history." + key] = val
    return v


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    iso = {k: load(d, "isoB_k%d" % k, k, "isoB final") for k, d in ISO.items()}
    b2 = {k: load(B2[0], B2[1], k, "B2 production") for k in range(1, 19)}
    for p in (CODE_D13 + "/page_history_v6o.py", CODE_D13 + "/page_w1_sector_v1d.py", PP_DIR + "/w1_postpass_v1f.py", os.path.abspath(__file__)):
        MAN.append({"label": "code", "path": p, "md5": md5(p)})
    res = {"channels": {}, "b2": {}, "inventory": {}, "settings_check": {}, "partials": {}, "snapshot_vs_final": {}, "partial_sum": {},
           "gap_2rs": {}, "missing": MISSING, "rule": {"rel_tol": REL_TOL, "floor": "F_meas(iso) + F_meas(B2), weighted and summed"}, "manifest": MAN}
    # inventory and settings check against kappa 1
    views = {k: settings_view(os.path.join(ISO[k], "isoB_k%d_k%d_summary.json" % (k, k))) for k in ISO}
    ref = views[1]
    for k, v in views.items():
        diffs = {}
        for key in sorted(set(ref) | set(v)):
            if key in ("final", "end_reason", "n_records", "last_record_T"):
                continue
            if json.dumps(ref.get(key), sort_keys=True, default=str) != json.dumps(v.get(key), sort_keys=True, default=str):
                diffs[key] = {"kappa1": ref.get(key), "this": v.get(key)}
        res["settings_check"][k] = {"final": v["final"], "end_reason": v["end_reason"], "n_records": v["n_records"], "last_record_T": v["last_record_T"],
                                    "code": v["code"], "nsites": v.get("settings.nsites"), "interp": v.get("w1.interp"), "interp_sub": v.get("w1.interp_sub"),
                                    "R_cut_rs": v.get("settings.R_cut_rs"), "T_end_rs": v.get("settings.T_end_rs"), "cheb_tol": v.get("w1.cheb_tol"),
                                    "sector": v.get("w1.sector"), "fields": v.get("settings.fields"), "differs_from_kappa1": diffs}
    for k, ch in iso.items():
        cl = sorted(t for t in ch["clocks"] if (ch["clocks"][t].get("cuts")))
        res["inventory"][k] = {"final_flag": ch["final"], "clocks_first": cl[0] if cl else None, "clocks_last": cl[-1] if cl else None, "n_clocks": len(cl),
                               "clocks": cl, "audit": (ch["w1"] or {}).get("audit")}
        res["channels"][k] = {"%g" % t: {lab: cell(ch, t, lab) for lab in CUTS} for t in cl}
    for k, ch in b2.items():
        cl = sorted(t for t in ch["clocks"] if (ch["clocks"][t].get("cuts")) and t <= 40.0 + 1e-9)
        res["b2"][k] = {"%g" % t: {lab: cell(ch, t, lab) for lab in CUTS} for t in cl}
    # partial resume-mirror states: every isoB_k* under resume_mirror
    for d in sorted(glob.glob(RM + "/*/isoB_k*/*")):
        for sj in glob.glob(d + "/isoB_k*_summary.json"):
            k = int(os.path.basename(sj).split("_k")[1].split("_")[0])
            S = json.load(open(sj))
            ch = PP.load_channel(d, "isoB_k%d" % k, k)
            cl = sorted(t for t in ch["clocks"] if (ch["clocks"][t].get("cuts")))
            res["partials"][d] = {"kappa": k, "final": S.get("final"), "end_reason": S.get("end_reason"), "n_records": len(S.get("records") or []),
                                  "d13_clock_last": cl[-1] if cl else None, "summary_md5": md5(sj), "audit": ((S.get("w1") or {}).get("audit") or {}).get("records_failed")}
            if k in iso:   # snapshot against its final at the common clocks
                nmax = 0.0; ncell = 0
                for t in cl:
                    for lab in CUTS:
                        a = cell(ch, t, lab); b = cell(iso[k], t, lab)
                        if a and b:
                            ncell += 1
                            nmax = max(nmax, abs(a["x"] - b["x"]))
                res["snapshot_vs_final"][d] = {"kappa": k, "cells": ncell, "max_abs_dx": nmax}
    # partial sum over the 7 final channels, isoB and B2 over the same set at matched clocks
    ks = sorted(iso)
    clocks = sorted(set.intersection(*[set(float(t) for t in res["channels"][k]) for k in ks]))
    for t in clocks:
        tk = "%g" % t
        for lab in CUTS:
            a = [res["channels"][k][tk][lab] for k in ks]
            b = [(res["b2"][k].get(tk) or {}).get(lab) for k in ks]
            if not all(a) or not all(b):
                continue
            Si = sum(16 * k * c["x"] for k, c in zip(ks, a)); Sb = sum(16 * k * c["x"] for k, c in zip(ks, b))
            Fi = sum(16 * k * c["fmeas"] for k, c in zip(ks, a)) if all(c["fmeas"] is not None for c in a) else None
            Fb = sum(16 * k * c["fmeas"] for k, c in zip(ks, b)) if all(c["fmeas"] is not None for c in b) else None
            d = Si - Sb
            floor = (Fi or 0.0) + (Fb or 0.0)
            agree = abs(d) <= REL_TOL * max(abs(Si), abs(Sb)) or abs(d) <= floor
            res["partial_sum"]["%s|%s" % (tk, lab)] = {"T": t, "cut": lab, "S_iso": Si, "F_iso": Fi, "S_B2": Sb, "F_B2": Fb, "d": d,
                                                       "ratio_iso_over_B2": (Si / Sb) if Sb != 0 else None, "agree": bool(agree),
                                                       "floor_level": bool(abs(d) <= floor)}
    # the v1.1 2 r_s gap, channel by channel
    for k in ks:
        g = {}
        for t in GAP_CLOCKS + [24.0, 36.0, 38.0, 40.0]:
            tk = "%g" % t
            a = (res["channels"][k].get(tk) or {}).get("2"); b = (res["b2"][k].get(tk) or {}).get("2")
            if a and b:
                g[tk] = {"iso": 16 * k * a["x"], "B2": 16 * k * b["x"], "B2_over_iso": (b["x"] / a["x"]) if a["x"] != 0 else None,
                         "F_iso": 16 * k * a["fmeas"] if a["fmeas"] is not None else None}
        res["gap_2rs"][k] = g
    json.dump(res, open(os.path.join(out, "ISO64_light_v1_2.json"), "w"), indent=0, default=float)

    def fmt(v, f="%.4e"):
        return "-" if v is None else (f % v)
    L = ["# ISO64 bridge light readout v1.2 (producer's on-box tables)\n", "## inventory and settings against kappa 1\n"]
    for k in ks:
        v = res["inventory"][k]; s = res["settings_check"][k]
        au = v["audit"] or {}
        L.append("- kappa %d: final %s; end_reason %s; records %d; d13 clocks %s to %s (%d); code %s; nsites %s; interp %s/%s; cuts %s; T_end %s; cheb_tol %s; sector %s; "
                 "audit records %s failed %s banded mismatches %s floor premise failures %s; settings keys differing from kappa 1: %s"
                 % (k, s["final"], s["end_reason"], s["n_records"], v["clocks_first"], v["clocks_last"], v["n_clocks"], s["code"], s["nsites"], s["interp"], s["interp_sub"],
                    s["R_cut_rs"], s["T_end_rs"], s["cheb_tol"], s["sector"], au.get("records_audited"), au.get("records_failed"), au.get("fast_banded_mismatches"),
                    au.get("floor_premise_failures"), json.dumps(s["differs_from_kappa1"], default=str)))
    L.append("\n## partial resume-mirror states (not used)\n")
    for d, v in res["partials"].items():
        L.append("- %s: kappa %d, final %s, end_reason %s, records %d, last d13 clock %s, summary md5 %s, records failed %s" % (d, v["kappa"], v["final"], v["end_reason"], v["n_records"], v["d13_clock_last"], v["summary_md5"], v["audit"]))
    for d, v in res["snapshot_vs_final"].items():
        L.append("- snapshot %s against kappa %d's final: %d common cells, max |dx| %.3e" % (d, v["kappa"], v["cells"], v["max_abs_dx"]))
    L.append("\n## A. per channel weighted term 16 kappa x_kappa (nats) per cut and d13 clock, F_meas weighted (2 r_s cut), beside B2's same channel\n")
    L.append("| kappa | T/r_s | " + " | ".join("isoB %s" % c for c in CUTS) + " | isoB F_meas (2) | " + " | ".join("B2 %s" % c for c in CUTS) + " | B2 F_meas (2) |")
    L.append("|---|---|" + "---|" * (2 * len(CUTS) + 2))
    for k in ks:
        for tk in sorted(res["channels"][k], key=float):
            a = res["channels"][k][tk]; b = res["b2"][k].get(tk) or {}
            L.append("| %d | %s | %s | %s | %s | %s |" % (k, tk, " | ".join(fmt(16 * k * a[c]["x"]) if a[c] else "-" for c in CUTS),
                     fmt(16 * k * a["2"]["fmeas"] if a["2"] and a["2"]["fmeas"] is not None else None, "%.2e"),
                     " | ".join(fmt(16 * k * b[c]["x"]) if b.get(c) else "-" for c in CUTS),
                     fmt(16 * k * b["2"]["fmeas"] if b.get("2") and b["2"]["fmeas"] is not None else None, "%.2e")))
    L.append("\n## B. partial 16-kappa-weighted sum over kappa %s (7 of 16), isoB against B2 over the same set at matched clocks\n" % ",".join(map(str, ks)))
    L.append("| T/r_s | cut | S isoB (7) | F_meas isoB | S B2 (same 7) | F_meas B2 | isoB / B2 | agree |")
    L.append("|---|---|---|---|---|---|---|---|")
    for key, r in sorted(res["partial_sum"].items(), key=lambda kv: (kv[1]["T"], float(kv[1]["cut"]))):
        L.append("| %g | %s | %s | %s | %s | %s | %s | %s |" % (r["T"], r["cut"], fmt(r["S_iso"]), fmt(r["F_iso"], "%.2e"), fmt(r["S_B2"]), fmt(r["F_B2"], "%.2e"),
                 fmt(r["ratio_iso_over_B2"], "%.3f"), ("agree (floor)" if r["floor_level"] else "agree") if r["agree"] else "DISAGREE"))
    L.append("\n## C. the v1.1 2 r_s gap channel by channel: B2 / isoB at the 2 r_s cut\n")
    gt = sorted({tk for k in ks for tk in res["gap_2rs"][k]}, key=float)
    L.append("| kappa | " + " | ".join("%s: isoB / B2 (B2/isoB)" % t for t in gt) + " |")
    L.append("|---|" + "---|" * len(gt))
    for k in ks:
        L.append("| %d | %s |" % (k, " | ".join(("%.3e / %.3e (%s)" % (res["gap_2rs"][k][t]["iso"], res["gap_2rs"][k][t]["B2"], fmt(res["gap_2rs"][k][t]["B2_over_iso"], "%.2f")))
                                                if t in res["gap_2rs"][k] else "-" for t in gt)))
    L.append("\n## missing channels (listed, not filled)\n\n- kappa %s\n" % ", ".join(map(str, MISSING)))
    L.append("\n## manifest\n")
    L += ["- %s  %s  [%s]" % (m["md5"], m["path"], m["label"]) for m in MAN]
    open(os.path.join(out, "ISO64_light_tables_v1_2.md"), "w").write("\n".join(L) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
