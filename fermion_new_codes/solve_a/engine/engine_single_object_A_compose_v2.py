#!/usr/bin/env python
"""engine_single_object_A_compose_v2.py (builder analyst; version 2 = version 1 with two header sentences corrected (the slices' precision is
printed per record, not asserted per family; the appended-exterior control named where it ran); inbox #41's revised S4 part A, its compose, with inbox #43's directions: icloud's
analytic R1 side, <internal>/strand_R_single_object_correlations_scope.tex, cited as the independent
analytic route; the identity checks on complete splits only (the outer split; the trapped annulus against its full complement); the slabs
printed as correlations; the weight at |ak| > pi/4 printed as the occupation of that band, never as escape).
Reads the family files written by engine_single_object_A_v1.py and engine_single_object_A_v2.py (never edits them) and writes one md and one
summary json. The device check compares two runs of the same record leaf by leaf (records matched by T/r_s, channels by kappa, every numeric
readout under the same key path; timings, clocks and script names excluded). No mutual information anywhere (D15). Every number in the md is
printed by this script from the files named on its command line.
Args: --families a.json,b.json,... --device-check first.json:second.json:label[;...] --strand-r tex --strand-r-scope tex --md --out-json."""
import argparse
import json
import math

LABEL = ("the single object's own cross-surface 2-point block and transfer on the solved history of the massive collapse, per radial channel; "
         "the interior beyond the hydrodynamic gate H sigma > 0.1 from T/r_s 11.75, the record's own limit; the massless channel, where present, "
         "a test channel under owner decision D14")
ENT_LABEL = "per radial channel; not the entropy coefficient"
DS = (2, 4, 8)
SKIP = ("seconds", "utc", "script", "gpu", "name", "label", "window", "convention", "run", "fields_module", "family", "note", "entropy_label",
        "no_mutual_information")


def fmt(x, f="%.3e"):
    return (f % x) if isinstance(x, (int, float)) and not isinstance(x, bool) and x is not None and x == x else "-"


def chans(F):
    for r in F["records"]:
        for ch in (r.get("channels") or []):
            yield r, ch


def splits_of(ch):
    for kind in ("dense", "applied"):
        s_ = ch["splits"].get(kind) or {}
        if s_.get("root", "x") is None or s_.get("same_split_as") or not s_.get("K"):
            continue
        yield kind, s_


def strand_r_checks(F):
    """the checks where A meets strand R, over every record, channel and complete split of the family."""
    out = {"outer_SO1_residual_max": 0.0, "outer_SO2_agreement_max": 0.0, "outer_sigma_max_max": 0.0, "annulus_SO1_residual_max": None,
           "annulus_SO2_agreement_max": None, "annulus_sigma_max_max": None, "stored_purity_max": 0.0, "loewdin_orthonormality_max": 0.0,
           "FD_minus_FC_max": 0.0, "FC_abs_max": 0.0, "complete_splits": 0, "annulus_splits": 0, "localized_lambda_min": None}
    for r, ch in chans(F):
        out["stored_purity_max"] = max(out["stored_purity_max"], ch["stored_idempotency"])
        out["loewdin_orthonormality_max"] = max(out["loewdin_orthonormality_max"], ch["loewdin_orthonormality"])
        for kind, s_ in splits_of(ch):
            out["complete_splits"] += 1
            idt = s_["identity"]
            out["outer_SO1_residual_max"] = max(out["outer_SO1_residual_max"], idt["CX_minus_CX2_minus_KKdag_op"])
            out["outer_SO2_agreement_max"] = max(out["outer_SO2_agreement_max"], idt["agreement"])
            out["outer_sigma_max_max"] = max(out["outer_sigma_max_max"], s_["K"]["sigma_max"])
            an = s_.get("annulus")
            if an:
                out["annulus_splits"] += 1
                for k_, v_ in (("annulus_SO1_residual_max", an["identity"]["op"]), ("annulus_SO2_agreement_max", an["identity"]["agreement"]),
                               ("annulus_sigma_max_max", an["K_complement"]["sigma_max"])):
                    out[k_] = v_ if out[k_] is None else max(out[k_], v_)
            cc = s_["cut_current"]
            out["FD_minus_FC_max"] = max(out["FD_minus_FC_max"], abs(cc["F_D"] - cc["F_C"]))
            out["FC_abs_max"] = max(out["FC_abs_max"], abs(cc["F_C"]))
            ob = s_.get("outward_branch") or {}
            if ob.get("localized_lambda_min") is not None:
                out["localized_lambda_min"] = ob["localized_lambda_min"] if out["localized_lambda_min"] is None else min(out["localized_lambda_min"], ob["localized_lambda_min"])
    out["SO1_bound_half_holds"] = bool(out["outer_sigma_max_max"] <= 0.5 and (out["annulus_sigma_max_max"] is None or out["annulus_sigma_max_max"] <= 0.5))
    return out


def leaves(obj, path=()):
    """the numeric leaves of a json object as {path: value}; keys naming timings, clocks, scripts or text skipped."""
    out = {}
    if isinstance(obj, dict):
        for k_, v_ in obj.items():
            if any(s_ in k_ for s_ in SKIP):
                continue
            out.update(leaves(v_, path + (k_,)))
    elif isinstance(obj, list):
        for i, v_ in enumerate(obj):
            out.update(leaves(v_, path + (str(i),)))
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool) and obj == obj:
        out[path] = float(obj)
    return out


def device_check(pa, pb, label):
    A = json.load(open(pa)); B = json.load(open(pb))
    def keyed(F):
        d = {}
        for r in F["records"]:
            Tk = "%.6f" % r["T_over_rs"]
            for ch in (r.get("channels") or []):
                for p, v in leaves(ch).items():
                    d[(Tk, "kappa %d" % ch["kappa"]) + p] = v
            for p, v in leaves({k_: r[k_] for k_ in ("roots_dense", "roots_applied", "outer_inner", "E", "two_gE") if k_ in r}).items():
                d[(Tk, "record") + p] = v
        return d
    a = keyed(A); b = keyed(B)
    common = sorted(set(a) & set(b))
    groups = {}
    worst = (0.0, None); worst_rel = (0.0, None)
    for k_ in common:
        dv = abs(a[k_] - b[k_])
        g = k_[2] if k_[1] != "record" else "record"
        if g == "splits" and len(k_) > 4:
            g = "splits." + k_[4]
        groups[g] = max(groups.get(g, 0.0), dv)
        if dv > worst[0]:
            worst = (dv, k_)
        sc = max(abs(a[k_]), abs(b[k_]))
        if sc > 1e-8 and dv / sc > worst_rel[0]:
            worst_rel = (dv / sc, k_)
    return {"label": label, "first": pa, "second": pb, "first_script": A.get("script"), "second_script": B.get("script"), "first_gpu": A.get("gpu"),
            "second_gpu": B.get("gpu"), "records_first": [r["T_over_rs"] for r in A["records"]], "records_second": [r["T_over_rs"] for r in B["records"]],
            "leaves_compared": len(common), "only_first": len(set(a) - set(b)), "only_second": len(set(b) - set(a)), "max_abs_diff": worst[0],
            "max_abs_diff_at": "/".join(worst[1]) if worst[1] else None, "max_rel_diff_above_1e-8": worst_rel[0],
            "max_rel_diff_at": "/".join(worst_rel[1]) if worst_rel[1] else None, "per_group_max_abs_diff": groups}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", required=True)
    ap.add_argument("--device-check", default=None)
    ap.add_argument("--strand-r-scope", default="<internal>/strand_R_single_object_correlations_scope.tex")
    ap.add_argument("--strand-r", default="<internal>/strand_R_single_object_correlations.tex")
    ap.add_argument("--hreduce", default="<internal>/web40f_single_object_identity.tex")
    ap.add_argument("--md", required=True); ap.add_argument("--out-json", required=True)
    a = ap.parse_args()
    paths = a.families.split(",")
    fams = [json.load(open(p)) for p in paths]
    checks = [strand_r_checks(F) for F in fams]
    dchk = []
    if a.device_check:
        for item in a.device_check.split(";"):
            pa, pb, lab = item.split(":", 2)
            dchk.append(device_check(pa, pb, lab))
    L = ["## S4 (revised) part A: the single object's correlations across the trapping root, per record and radial channel", "",
         "Composed by engine_single_object_A_compose_v2.py from the family files %s (engine_single_object_A_v1.py and _v2.py; inbox #41; post-processing "
         "of stored states, no propagation)." % ", ".join(paths), "",
         "- %s." % LABEL,
         "- No mutual information is computed or printed (D15). S_X appears only as the internal check of item 2, labelled \"%s\"." % ENT_LABEL,
         "- No 16 kappa weight on any number below (per radial channel, the g_n guard).",
         "- The stored columns: ||C^2 - C||_op of the stored state is printed as stored, per record and channel (the slices' own precision; some "
         "slices are complex64); every other readout of C uses the columns re-orthonormalized in float64 (Loewdin), and the continued reference P as stored.",
         "- The Killing vacuum P_-: the negative-energy projector of h_m at the record's applied fields on the stored chain (to its end, where the "
         "engine's exterior fields already sit). The exterior appended to 128 r_s ran as the labelled control where a record line below lists it.", "",
         "### The independent analytic route (inbox #43)", "",
         "icloud's analytic R1 side is the independent route for these readouts: the scope %s and its source %s (SO1 to SO15). The source's section 7 "
         "compares hreduce's %s in full (shared formulas agree; no numerical or method credit). Where the two meet:" % (a.strand_r_scope, a.strand_r, a.hreduce),
         "- SO1, for a pure number-conserving Gaussian state on a complete split X + Y with K = Pi_X C Pi_Y: C_X(1 - C_X) = K K^dagger, so 0 <= sigma_j(K) <= 1/2. "
         "Here the identity is checked only on complete splits: the outer split (X inside the outer root, Y the rest) and the trapped annulus against its full "
         "complement. On a complete split the SO1 residual of the stored columns is exactly Pi_X (C - C^2) Pi_X (strand R), so the stored "
         "||C^2 - C||_op bounds it. The residual printed below is the Loewdin projector's.",
         "- SO2: S_X = Tr s(C_X) = sum_j s((1 - sqrt(1 - 4 sigma_j^2))/2), each nonzero singular value counted once. Both routes are printed. "
         "The sigma route sums every singular value of K (no threshold), so no omitted-tail bound is needed; the count above 1e-12 is a readout only.",
         "- SO3: a slab pair is not a complete split. The omitted complement term survives, and slab singular values and norms are correlation "
         "readouts. The slab tables print sigma_max and the Frobenius norm of the cross blocks as correlations, with no identity. C - P and C - P_- are "
         "block differences, not covariances (strand R: the signed C - R_sp is not a covariance).",
         "- Section 2: the cut current F = -2 Im Tr(h_XY C_YX) may use D = C - R_sp in place of C when [h, R_sp] = 0 at the instant; "
         "|F_D - F_C| is printed as that check.",
         "- Section 3: the native massive symbol has positive outward velocity for pi/4 < q < pi/2 at every inward speed, \"without implying occupation "
         "or escape through the asymptotic mass gap\". Item 5's second quarter [pi/4, pi/2) is that band. Its weight, and the weight at |ak| >= pi/4, "
         "are printed as the occupation of the band by the state's own orbitals (localized, all occupied, and the dust's columns). They are never "
         "escape or transfer; transfer is part B's readout.", "",
         "| family | complete splits (outer; annulus) | max SO1 residual, outer | max SO1 residual, annulus | max S_X agreement, outer | max S_X agreement, annulus | max sigma_max(K), outer | max sigma_max, annulus | SO1 bound 1/2 holds | max stored ||C^2 - C||_op | max Loewdin orthonormality | max abs(F_D - F_C) | max abs(F_C) | smallest localized eigenvalue |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for F, c in zip(fams, checks):
        L.append("| %s | %d; %d | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            F["family"], c["complete_splits"], c["annulus_splits"], fmt(c["outer_SO1_residual_max"], "%.1e"), fmt(c["annulus_SO1_residual_max"], "%.1e"),
            fmt(c["outer_SO2_agreement_max"], "%.1e"), fmt(c["annulus_SO2_agreement_max"], "%.1e"), fmt(c["outer_sigma_max_max"], "%.6f"),
            fmt(c["annulus_sigma_max_max"], "%.6f"), c["SO1_bound_half_holds"], fmt(c["stored_purity_max"], "%.1e"), fmt(c["loewdin_orthonormality_max"], "%.1e"),
            fmt(c["FD_minus_FC_max"], "%.1e"), fmt(c["FC_abs_max"], "%.4e"), fmt(c["localized_lambda_min"], "%.6f")))
    L.append("")
    if dchk:
        L += ["### The device check (D)", "",
              "| check | first (script; gpu) | second (script; gpu) | records first; second | leaves compared (only first; only second) | max abs difference (at) | max relative difference above 1e-8 (at) | per group max abs difference |",
              "|---|---|---|---|---|---|---|---|"]
        for d in dchk:
            L.append("| %s | %s (%s; %s) | %s (%s; %s) | %s; %s | %d (%d; %d) | %.1e (%s) | %.1e (%s) | %s |" % (
                d["label"], d["first"], d["first_script"], d["first_gpu"], d["second"], d["second_script"], d["second_gpu"],
                ", ".join("%.3f" % x for x in d["records_first"]), ", ".join("%.3f" % x for x in d["records_second"]), d["leaves_compared"], d["only_first"],
                d["only_second"], d["max_abs_diff"], d["max_abs_diff_at"], d["max_rel_diff_above_1e-8"], d["max_rel_diff_at"],
                "; ".join("%s %.1e" % (k_, v_) for k_, v_ in sorted(d["per_group_max_abs_diff"].items()))))
        L.append("")
    for F in fams:
        L += ["### Family %s (%s; %s; field module %s; %s to %s)" % (F["family"], F["run"], F.get("script"), F["fields_module"], F.get("utc_start"), F.get("utc_end")), ""]
        for r in F["records"]:
            L.append("- %s: T/rs %.3f; E %s, 2GE %s; the dense roots of N + v (slope): %s; the applied-field roots: %s; the stored zeros %s; outer, inner: dense %s, applied %s%s." % (
                r["name"], r["T_over_rs"], fmt(r["E"], "%.6f"), fmt(r["two_gE"], "%.4f"), ", ".join("%.4f (%+.4f)" % (q["R"], q["slope"]) for q in r["roots_dense"]) or "none",
                ", ".join("%.4f (%+.4f)" % (q["R"], q["slope"]) for q in r["roots_applied"]) or "none", r["stored_trapping_zeros"],
                r["outer_inner"]["dense"], r["outer_inner"]["applied"], ("; " + r["note"]) if r.get("note") else ""))
            if r.get("vacuum_control"):
                vc = r["vacuum_control"]
                L.append("  The vacuum control (the exterior appended to %d sites, R %.0f; %.0f s): the slab blocks of P_- against the stored chain's, max abs. difference %s." % (
                    vc["n_ext_sites"], vc["R_end"], vc["eig_seconds"], "; ".join("%s %.1e" % (q["slab"], q["max_abs_diff"]) for q in vc["rows"])))
        L.append("")
        for kind in ("dense", "applied"):
            L += ["#### %s, the %s root: items 1, 2 and 4 per record and channel (the identity on complete splits only)" % (F["family"], kind), "",
                  "| record | kappa | R_outer (X sites) | n sigma > 1e-12 | sigma_max(K) | norm_F(K) | the top singular values | stored C^2 - C (op) | SO1 residual, outer: C_X - C_X^2 - K K^dagger (op) | S_X: trace; sigma route; agreement (%s) | annulus against its full complement (complete): sites; n sigma; sigma_max; norm_F; SO1 residual; S_X agreement | annulus to the outside Y (not a complete split; correlation): sigma_max; norm_F | F (D = C - P) | F (C) | abs(F_D - F_C) |" % ENT_LABEL,
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
            for r, ch in chans(F):
                s_ = ch["splits"].get(kind) or {}
                if s_.get("root", "x") is None:
                    L.append("| %.3f | %d | no root | - | - | - | - | %.1e | - | - | - | - | - | - | - |" % (r["T_over_rs"], ch["kappa"], ch["stored_idempotency"])); continue
                if s_.get("same_split_as"):
                    L.append("| %.3f | %d | %.4f (%d): the same site split as the %s root | | | | | %.1e | | | | | | | |" % (r["T_over_rs"], ch["kappa"], s_["R_outer"], s_["iX"], s_["same_split_as"], ch["stored_idempotency"])); continue
                K = s_["K"]; idt = s_["identity"]; an = s_.get("annulus"); cc = s_["cut_current"]
                ann = ("%d to %d; %d; %.3e; %.3e; %.1e; %.1e" % (an["sites"][0], an["sites"][1], an["K_complement"]["n_above_1e-12"], an["K_complement"]["sigma_max"],
                                                              an["K_complement"]["fro"], an["identity"]["op"], an["identity"]["agreement"])) if an else "-"
                ano = ("%.3e; %.3e" % (an["K_to_outside"]["sigma_max"], an["K_to_outside"]["fro"])) if an else "-"
                L.append("| %.3f | %d | %.4f (%d) | %d | %.6e | %.6e | %s | %.1e | %.1e | %.6f; %.6f; %.1e | %s | %s | %+.6e | %+.6e | %.1e |" % (
                    r["T_over_rs"], ch["kappa"], s_["R_outer"], s_["iX"], K["n_above_1e-12"], K["sigma_max"], K["fro"], " ".join("%.3e" % x for x in K["sigma_above_1e-12"][:8]),
                    ch["stored_idempotency"], idt["CX_minus_CX2_minus_KKdag_op"], idt["S_X_trace"], idt["S_X_sigma"], idt["agreement"], ann, ano, cc["F_D"], cc["F_C"],
                    abs(cc["F_D"] - cc["F_C"])))
            L.append("")
            for geo in ("P07", "centred"):
                L += ["#### %s, the %s root: item 3, the 4-site slabs (%s): correlations (strand R SO3; no identity on a slab pair), sigma_max; norm_F of the cross block" % (
                    F["family"], kind, "P07 sec. 10: the nearest sites at least d from the root" if geo == "P07" else "the task's wording: centred d sites from the root"), "",
                    "| record | kappa | d | inside sites R | outside sites R | C | P_- (Killing vacuum) | P (continued reference) | C - P_- (block difference) | C - P (block difference) |",
                    "|---|---|---|---|---|---|---|---|---|---|"]
                for r, ch in chans(F):
                    s_ = ch["splits"].get(kind) or {}
                    if not s_.get("slabs"):
                        continue
                    for d in DS:
                        q = s_["slabs"].get("%s_%d" % (geo, d))
                        if not q:
                            continue
                        bs = lambda b: "%.4e; %.4e" % (b["sigma_max"], b["fro"]) if b else "-"
                        L.append("| %.3f | %d | %d | %s | %s | %s | %s | %s | %s | %s |" % (
                            r["T_over_rs"], ch["kappa"], d, ",".join("%.0f" % x for x in q["inside_sites_R"]), ",".join("%.0f" % x for x in q["outside_sites_R"]),
                            bs(q["C"]), bs(q.get("P_minus")), bs(q["P"]), bs(q.get("C_minus_P_minus")), bs(q["C_minus_P"])))
                L.append("")
            L += ["#### %s, the %s root: item 5, the occupation of the momentum bands by the state's own orbitals (an occupation readout; never escape or transfer)" % (F["family"], kind), "",
                  "The weight of the interior components (sites 1 to iX; rectangular window; orthonormal DFT per spinor component) at |ak| >= pi/4, and in the "
                  "second quarter [pi/4, pi/2), strand R's outward band; the quarter shares are [0, pi/4), [pi/4, pi/2), [pi/2, 3pi/4], (3pi/4, pi].", "",
                  "| record | kappa | localized orbitals (eigenvalue >= 0.99; smallest) | localized: occupation at abs(ak) >= pi/4 | localized: occupation of [pi/4, pi/2) | quarters, localized | all occupied (comparator): occupation at abs(ak) >= pi/4 | all occupied: [pi/4, pi/2) | quarters, all | the dust's columns: occupation at abs(ak) >= pi/4 (not escape) | the dust: [pi/4, pi/2) | quarters, dust |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|"]
            for r, ch in chans(F):
                s_ = ch["splits"].get(kind) or {}
                ob = s_.get("outward_branch")
                if not ob:
                    continue
                qs_ = lambda q: ", ".join("%.4f" % x for x in q["quarters"]) if q else "-"
                q2 = lambda q: fmt(q["quarters"][1], "%.4f") if q else "-"
                L.append("| %.3f | %d | %d (%s) | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                    r["T_over_rs"], ch["kappa"], ob["localized_count"], fmt(ob["localized_lambda_min"], "%.6f"), fmt((ob["localized"] or {}).get("f_gt"), "%.4f"),
                    q2(ob["localized"]), qs_(ob["localized"]), fmt((ob["occupied_all"] or {}).get("f_gt"), "%.4f"), q2(ob["occupied_all"]), qs_(ob["occupied_all"]),
                    fmt((ob.get("dust") or {}).get("f_gt"), "%.4f"), q2(ob.get("dust")), qs_(ob.get("dust"))))
            L.append("")
    worst_id = max(max(c["outer_SO1_residual_max"], c["annulus_SO1_residual_max"] or 0.0) for c in checks)
    worst_ag = max(max(c["outer_SO2_agreement_max"], c["annulus_SO2_agreement_max"] or 0.0) for c in checks)
    L += ["- Item 2 over every record, channel and complete split of the %d families: the largest SO1 residual %.1e and the largest S_X agreement %.1e (the "
          "target 1e-10). Every number above is printed by this script from the family files." % (len(fams), worst_id, worst_ag)]
    open(a.md, "w").write("\n".join(L) + "\n")
    summary = {"script": "engine_single_object_A_compose_v2.py", "families": paths, "family_names": [F["family"] for F in fams], "label": LABEL,
               "entropy_label": ENT_LABEL, "no_mutual_information": "D15", "strand_R": {"scope": a.strand_r_scope, "source": a.strand_r, "hreduce_compared_there": a.hreduce},
               "strand_R_checks": dict(zip([F["family"] for F in fams], checks)), "device_checks": dchk, "worst_SO1_residual": worst_id, "worst_S_X_agreement": worst_ag}
    rows = []
    for F in fams:
        for r, ch in chans(F):
            for kind, s_ in splits_of(ch):
                ob = s_.get("outward_branch") or {}
                an = s_.get("annulus")
                rows.append({"family": F["family"], "T_over_rs": r["T_over_rs"], "kappa": ch["kappa"], "root": kind, "R_outer": s_["R_outer"], "iX": s_["iX"],
                             "sigma_max": s_["K"]["sigma_max"], "fro": s_["K"]["fro"], "n_above_1e-12": s_["K"]["n_above_1e-12"], "SO1_residual": s_["identity"]["CX_minus_CX2_minus_KKdag_op"],
                             "S_X_trace": s_["identity"]["S_X_trace"], "S_X_sigma": s_["identity"]["S_X_sigma"], "S_X_agreement": s_["identity"]["agreement"],
                             "annulus": ({"sites": an["sites"], "sigma_max": an["K_complement"]["sigma_max"], "fro": an["K_complement"]["fro"], "SO1_residual": an["identity"]["op"],
                                          "S_X_agreement": an["identity"]["agreement"], "to_outside_sigma_max": an["K_to_outside"]["sigma_max"]} if an else None),
                             "F_D": s_["cut_current"]["F_D"], "F_C": s_["cut_current"]["F_C"], "stored_purity": ch["stored_idempotency"],
                             "band_occupation": {"localized_ge_pi4": (ob.get("localized") or {}).get("f_gt"), "localized_pi4_pi2": ((ob.get("localized") or {}).get("quarters") or [None, None])[1],
                                                 "all_ge_pi4": (ob.get("occupied_all") or {}).get("f_gt"), "dust_ge_pi4": (ob.get("dust") or {}).get("f_gt"),
                                                 "dust_pi4_pi2": ((ob.get("dust") or {}).get("quarters") or [None, None])[1], "localized_count": ob.get("localized_count"),
                                                 "label": "occupation of the band by the state's own orbitals; never escape or transfer"}})
    summary["rows"] = rows
    json.dump(summary, open(a.out_json, "w"), indent=1, default=float)
    print("composed %s: %d families, %d rows; the largest SO1 residual %.1e, S_X agreement %.1e; device checks: %s" % (
        a.md, len(fams), len(rows), worst_id, worst_ag, "; ".join("%s max abs %.1e over %d leaves" % (d["label"], d["max_abs_diff"], d["leaves_compared"]) for d in dchk) or "none"), flush=True)


if __name__ == "__main__":
    main()
