#!/usr/bin/env python
"""ladders_compose_v1_2.py (composition builder, the solve lane): writes MASS_ladder_v1.2.md and SCALING_ladder_v1.2.md from the
version 1.1 producer's ladders_readout_v1_2.json only (no run file is opened; every number is the producer's). Version 1's composer
with the correctness reading's labels: (n.c.) and (unc.) on every readout, the certified and 30.3 extents, the confirming-residual audit,
the formation bracket, the fits as 2-coefficient diagnostics, the chain-column onsets as upper bounds, the centroid named as such.
Usage: python ladders_compose_v1_2.py --readout JSON --sources JSON --out-dir DIR"""
import argparse
import json
import math
import os

PREP_RULE = ("Every member and control shares one preparation rule: the dust is the levitating well's zero-temperature filling within the "
             "sea set kappa = 1 to kappa_sea (engine prepare_dust with fill_lowest: the well of depth set by the release radius R_0, the "
             "lowest levels of all channels filled to the count N_B = r_s/(2 g m a), g = 1/4, the projection onto the clocked positive band), "
             "released from coordinate rest at R_0 = 4 r_s (2 r_s for the 2 preparation members), prepared at ell 4 and evolved at the member's "
             "ell. No member runs the isotropic catalog; each member's own onset prints as 4 a times the largest chain column count of its "
             "catalog at T = 0 (20 a for the production's set), an upper bound on the per-valley onset 4 a n_rad,max that the T = 0 valley "
             "census fixes (12 a to 20 a for the production's set by the flat-well control's split), beside the isotropic form "
             "1.222 N_B^(1/3) a; the isotropic onset and the exclusion floor 0.49 N_B^(1/3) a are comparators, not demonstrated bounds on "
             "the chain's dust.")


def f(x, fmt="%.4f"):
    if x is None:
        return "-"
    try:
        if isinstance(x, float) and not math.isfinite(x):
            return "-"
        return fmt % x
    except TypeError:
        return str(x)


def e(x):
    return f(x, "%.2e")


def table(h, rows):
    out = ["| " + " | ".join(h) + " |", "|" + "---|" * len(h)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def clocks_text(v, n=12):
    if not v:
        return "none"
    s = ", ".join("%.4f" % t for t in v[:n])
    return s + (" (and %d more)" % (len(v) - n) if len(v) > n else "")


def flag(r):
    if r is None:
        return ""
    if not r.get("counts"):
        return " (n.c.)"
    if not r.get("certified", True):
        return " (unc.)"
    return ""


class Doc:
    def __init__(self, D, src):
        self.D, self.M, self.src = D, D["members"], src

    def short(self, nm):
        m = self.M[nm]["meta"]
        return "(%g, %g) s%g t%g" % (m["m"], m["rs"], m["sigma"], m["tau"])

    def member_table(self, names, labels):
        h = ["member", "role", "m a", "r_s/a", "sigma/a", "tau/a (tau/r_s)", "sites", "N_B", "N_B (m a)^3/1.087 (nominal count ratio)",
             "E0 (head)", "2 g E0 (a)", "E0/(N_B m)", "E0/E_max (charge ratio, c_E 0.135858)", "f_kin (head)", "p_F/m isotropic release form",
             "(p_F/m)^2/5 isotropic (a release comparator)", "own onset, chain columns (4 a times columns; upper bound) (/r_s)",
             "isotropic 1.222 N_B^(1/3) a (/r_s)", "exclusion floor 0.49 N_B^(1/3) a", "records (first to last T/r_s; dropped without E)",
             "last converged record (certified prefix)", "30.3 prefix's last counting record", "first non-counting record (its gate)", "end state"]
        rows = []
        for nm in names:
            M = self.M[nm]; m = M["meta"]; c = M["counts"]; v = M["convergence"]; cat = M["catalog"]
            own = ("%s a (%s)" % (f(cat.get("onset_own_a"), "%.0f"), f(cat.get("onset_own_rs"), "%.3f"))) if cat.get("onset_own_a") else "not available"
            rows.append([nm, labels.get(nm, m["role"]), f(m["m"], "%g"), f(m["rs"], "%g"), f(m["sigma"], "%g"),
                         "%g (%s)" % (m["tau"], f(c["tau_over_rs"], "%.5f")), m["nsites"], f(c["N_B"], "%.0f"), f(c["E_over_Emax"], "%.2f"),
                         f(c["E0"], "%.4f"), f(c["two_gE0"], "%.3f"), f(c["E0_over_NBm"], "%.4f"), f(c["E0_over_Emax"], "%.2f"), f(c["f_kin"], "%.4f"),
                         f(c["p_F_over_m_iso"], "%.3f"), f(c["pressure_corr_iso"], "%.6f"), own,
                         "%s a (%s)" % (f(c["onset_iso_a"], "%.1f"), f(c["onset_iso_rs"], "%.3f")), f(c["excl_floor_a"], "%.1f") + " a",
                         "%d (%s to %s; %s)" % (v["records"], f(v["first"], "%.3f"), f(v["last"], "%.4f"), v["records_dropped_without_E"]),
                         f(v["last_converged"]) if v["last_converged"] is not None else "none (its first record does not count)",
                         f(v["r303_last_counting"]),
                         ("%s (%s)" % (f(v["first_noncounting"]), v["first_noncounting_gate"])) if v["first_noncounting"] is not None else "none",
                         self.end_state(nm)])
        return table(h, rows)

    def end_state(self, nm):
        M = self.M[nm]; m = M["meta"]; v = M["convergence"]; fo = M["formation"]
        T_end = float(m["args"].get("T_over_rs") or 0)
        bits = []
        if m.get("snapshot_of"):
            bits.append("running on its card; read from version 1's snapshot of its mirrored records, to %.4f" % v["last"])
        elif v["last"] is not None and v["last"] + 1e-9 < T_end:
            bits.append("stopped at %.4f of its requested %g" % (v["last"], T_end))
        else:
            bits.append("records to %g" % v["last"])
        if v["first_noncounting"] is not None:
            bits.append("certified to %s" % (f(v["last_converged"]) if v["last_converged"] is not None else "no record"))
        if not fo:
            bits.append("no trapping zero in any record")
        return "; ".join(bits)

    def formation_table(self, names):
        h = ["member", "first saved record with a zero T_f/r_s", "bracket (previous record T/r_s, min(N - |v|) there; width r_s)",
             "zero of the stored min(N - |v|) series in the bracket: linear; quadratic (a diagnostic, not an event location)",
             "R_t/a (first record)", "R_t/r_s", "R_t/(2 g E) (root fraction)", "E/E0", "zeros (a)", "no tangency; slope at R_t",
             "root gate sigma/R_t", "residual (kept); confirming", "certification", "first episode closes at", "min(N - |v|) in the first episode"]
        rows = []
        for nm in names:
            M = self.M[nm]; fo = M["formation"]; ep = (M["episodes"] or [None])[0]
            if not fo:
                rows.append([nm, "none to %s" % f(M["convergence"]["last"], "%.4f")] + ["-"] * 13)
                continue
            br = fo["bracket"]
            cl = "-"
            if ep:
                if ep.get("closes_at") is not None:
                    cl = f(ep["closes_at"]) + ("" if ep.get("closes_certified", True) else " (unc.)")
                else:
                    cl = "open to %s" % f(ep["last"])
            dep = "-"
            if ep:
                dep = "%s (%s%s)" % (f(ep["min_minNv"]), f(ep["min_at"]), "" if (ep["min_cert_at"] == ep["min_at"]) else ", unc.")
                if ep["min_cert_at"] != ep["min_at"]:
                    dep += "; over certified records %s (%s)" % (f(ep["min_cert"]), f(ep["min_cert_at"]))
                if ep["noncounting_inside"] or ep["uncertified_inside"]:
                    dep += "; %d of %d records do not count, %d uncertified" % (ep["noncounting_inside"], ep["records"], ep["uncertified_inside"])
            rows.append([nm, f(fo["T"]), "%s, %s; %s" % (f(br["prev_T"]), f(br["prev_minNv"]), f(br["width"], "%.5f")),
                         "%s; %s" % (f(br["T_lin"]), f(br["T_quad"])), f(fo["R_t"], "%.3f"), f(fo["R_t_over_rs"]), f(fo["R_t_over_2gE"]),
                         f(fo["E_over_E0"], "%.5f"), ", ".join("%.3f" % z for z in fo["zeros"]), "%s; %s" % ("yes" if fo["simple"] else "no", e(fo["slope"])),
                         f(fo["gate_sigma_over_Rt"], "%.3f"), "%s; %s" % (e(fo["residual"]), e(fo["confirming"])), fo["label"], cl, dep])
        return table(h, rows)

    def episodes_text(self, names):
        lines = []
        for nm in names:
            eps = self.M[nm]["episodes"]
            if not eps:
                continue
            parts = []
            for i, ep in enumerate(eps[:6]):
                lab = []
                if ep["noncounting_inside"]:
                    lab.append("%d of %d records do not count" % (ep["noncounting_inside"], ep["records"]))
                if ep["uncertified_inside"]:
                    lab.append("%d uncertified" % ep["uncertified_inside"])
                parts.append("%d: %s to %s (%d records; R_t %s to %s a; min(N - |v|) %s at %s; %s%s)" % (
                    i + 1, f(ep["first"]), f(ep["last"]), ep["records"], f(ep["R_t_first"], "%.3f"), f(ep["R_t_last"], "%.3f"),
                    f(ep["min_minNv"]), f(ep["min_at"]) + ("" if ep["min_cert_at"] == ep["min_at"] else " (unc.)"),
                    ("closes at %s%s" % (f(ep["closes_at"]), "" if ep.get("closes_certified", True) else " (unc.)")) if ep["closes_at"] is not None else "open at the last record",
                    ("; " + ", ".join(lab)) if lab else ""))
            more = (" (and %d more)" % (len(eps) - 6)) if len(eps) > 6 else ""
            lines.append("- %s: %s%s" % (nm, "; ".join(parts), more))
        return "\n".join(lines)

    def compression_table(self, names):
        h = ["member", "min R_c (dust-energy centroid) over the certified records (a)", "at T/r_s", "min R_c/r_s",
             "own onset, chain columns (upper bound)", "isotropic 1.222 N_B^(1/3) a", "exclusion floor", "minimum at the last certified record"]
        rows = []
        for nm in names:
            M = self.M[nm]; rc = M["Rc_min"]; c = M["counts"]; cat = M["catalog"]
            rows.append([nm, f(rc.get("min_Rc"), "%.2f"), f(rc.get("at")), f(rc.get("min_Rc_over_rs"), "%.3f"),
                         (f(cat.get("onset_own_a"), "%.0f") + " a") if cat.get("onset_own_a") else "not available", f(c["onset_iso_a"], "%.1f") + " a",
                         f(c["excl_floor_a"], "%.1f") + " a", "yes (still falling at the end of the certified records)" if rc.get("over_end") else "no"])
        return table(h, rows)

    def matched_tables(self, names, key, title, fmt="%.4f", step=2):
        h = ["T/r_s"] + names
        rows = []
        for t in range(0, 41, step):
            row = ["%d" % t]
            for nm in names:
                r = self.M[nm]["matched"].get("%g" % t)
                if r is None:
                    row.append("-")
                else:
                    v = r.get(key)
                    row.append((f(v, fmt) if v is not None else ("none" if key.startswith("R_t") else "-")) + flag(r))
            rows.append(row)
        return "%s\n\n%s" % (title, table(h, rows))

    def from_formation_table(self, names):
        h = ["member", "T_f + 0", "T_f + 0.5", "T_f + 1", "T_f + 2", "T_f + 4"]
        rows = []
        for nm in names:
            M = self.M[nm]
            if not M["formation"]:
                continue
            row = [nm]
            for d in ("0", "0.5", "1", "2", "4"):
                r = M["matched_from_formation"].get(d)
                if r is None:
                    row.append("-")
                else:
                    row.append("R_t/r_s %s, R_t/(2gE) %s, E/E0 %s, min(N-|v|) %s, gate %s%s" % (
                        f(r["R_t_over_rs"]) if r["R_t"] is not None else "none", f(r["R_t_over_2gE"]) if r["R_t"] is not None else "-",
                        f(r["EE0"], "%.5f"), f(r["minNv"]), f(r["gate"], "%.3f"), flag(r)))
            rows.append(row)
        return table(h, rows)

    def convergence_table(self, names):
        h = ["member", "script; solver", "engine entries in the records", "records (dropped without E)", "converged True as stored",
             "counting records", "capped (converged False)", "above 1e-3", "confirming residual above stop_fields (Newton): count; clocks; max (at)",
             "residual max (at)", "max over the records up to the last converged record", "below 1e-4 / 1e-4 to 1e-3",
             "30.3 prefix's last counting record", "last converged record (certified prefix)"]
        rows = []
        for nm in names:
            M = self.M[nm]; m = M["meta"]; v = M["convergence"]
            eng = "; ".join("%s (%s to %s, %d)" % (u["value"], f(u["first"], "%.3f"), f(u["last"], "%.3f"), u["records"]) for u in m["engine_entries"][:3])
            if m["newton"]:
                sol = "newton (at most %s steps, %s Krylov vectors, set %s, ends above %s)" % (m["args"].get("nk_maxiter"), m["args"].get("nk_krylov"),
                                                                                           m["args"].get("nk_set"), m["args"].get("end_above"))
                cft = "%d; %s; %s (%s); stop_fields %s" % (v["conf_fail_count"], clocks_text(v["conf_fail_clocks"], 8), e(v["conf_max"]), f(v["conf_max_at"]),
                                                          e(v["stop_fields"]))
            else:
                sol = "plain (at most %s iterations)" % m["args"].get("maxit")
                cft = "- (plain)"
            rows.append([nm, "%s; %s" % (m["script"], sol), eng, "%d (%s)" % (v["records"], v["records_dropped_without_E"]), v["converged_true_stored"],
                         v["counting"], v["capped"], v["above_1e-3_count"], cft, "%s (%s)" % (e(v["residual_max"]), f(v["residual_max_at"])),
                         e(v["residual_max_certified"]), "%d / %d" % (v["below_1e-4"], v["band_1e-4_1e-3"]), f(v["r303_last_counting"]),
                         f(v["last_converged"]) if v["last_converged"] is not None else "none"])
        return table(h, rows)

    def excluded_text(self, names):
        lines = []
        for nm in names:
            ex = self.M[nm]["convergence"]["excluded"]
            if ex:
                s = "; ".join("%s (%s; residual %s%s)" % (f(x["T"]), x["gate"], e(x["residual"]),
                                                          (", confirming %s" % e(x["confirming"])) if x.get("confirming") is not None else "") for x in ex)
                lines.append("- %s: %d records do not count: %s" % (nm, len(ex), s))
        return "\n".join(lines) if lines else "- none"

    def spreads_table(self, pairs):
        h = ["member", "control", "kind", "formation classes (member / control)", "dT_f (control - member)", "dR_t (a)", "dR_t/R_t", "d(R_t/r_s)",
             "d(R_t/(2gE))", "clocks counting and certified in both (from to)", "max |d E/E0|", "max |d min(N - |v|)|", "max |d R_c|/r_s",
             "max |d R_t| (a); max |d R_t|/R_t (at) where both hold a zero (count)",
             "counting records regardless of certification (labelled diagnostic): common clocks; max |d R_t| (a); max |d R_t|/R_t (at); at 16 (count)"]
        rows = []
        for s in self.D["spreads"]:
            if (s["member"], s["control"]) not in pairs:
                continue
            rows.append([s["member"], s["control"], s["kind"], s.get("classes", "-"), f(s.get("dT_f")), f(s.get("dR_t"), "%.3f"), f(s.get("dR_t_rel"), "%.4f"),
                         f(s.get("dR_t_over_rs")), f(s.get("dR_t_over_2gE")),
                         "%s (%s to %s)" % (s.get("common_clocks", "-"), f(s.get("common_from")), f(s.get("common_to"))),
                         e(s.get("max_abs_dEE0")), e(s.get("max_abs_dminNv")), e(s.get("max_abs_dRc_over_rs")),
                         "%s; %s (%s) (%s)" % (f(s.get("max_abs_dRt"), "%.3f"), f(s.get("max_rel_dRt"), "%.4f"), f(s.get("max_rel_dRt_at")),
                                               s.get("Rt_common_count", "-")),
                         "%s; %s; %s (%s); %s (%s)" % (s.get("raw_common_clocks", "-"), f(s.get("raw_max_abs_dRt"), "%.3f"), f(s.get("raw_max_rel_dRt"), "%.4f"),
                                                       f(s.get("raw_max_rel_dRt_at")), f(s.get("raw_rel_dRt_at_16"), "%.4f"), s.get("raw_Rt_common_count", "-"))])
        return table(h, rows)

    def fit_text(self, key, ylabels=("T_f", "R_t_over_rs", "R_t_over_2gE")):
        F = self.D["fits"].get(key)
        if not F or isinstance(F, (str, list)):
            return "- not evaluable (%s)" % (F if F else "too few members")
        lines = ["- model: %s" % F["model"], "- members: %s (%d members, %d coefficients, residual degrees of freedom %d)" % (
            ", ".join(F["members"]), F["n"], F["coefficients"], F["dof"])]
        if "x" in F:
            lines.append("- x per member: %s" % ", ".join("%s %s" % (nm, f(x, "%.5f")) for nm, x in zip(F["members"], F["x"])))
        if "z" in F:
            lines.append("- (z, h) per member: %s" % ", ".join("%s (%s, %s)" % (nm, f(z), f(h)) for nm, z, h in zip(F["members"], F["z"], F["h"])))
        if F.get("uncertified"):
            lines.append("- formation records inside this fit that are not certified (labelled; no law): %s" % ", ".join(
                "%s (%s)" % (nm, self.M[nm]["formation"]["class"]) for nm in F["uncertified"]))
        h = ["Y", "coefficients", "member: Y, fit, residual", "rms residual over members; per residual degree of freedom", "max |residual|"]
        rows = []
        for y in ylabels:
            if y not in F:
                continue
            r = F[y]
            per = "; ".join("%s: %s, %s, %s" % (nm, f(a_), f(b_), "%+.4f" % c_) for nm, a_, b_, c_ in zip(F["members"], r["y"], r["pred"], r["residual"]))
            if F["dof"] > 0:
                coef = ", ".join("c%d %s" % (i, "%.6g" % c) for i, c in enumerate(r["coef"]))
                coef += (" (c0 from first-record clocks; cadence error up to the record step per member; not an intercept)" if y == "T_f" else
                         " (first-record radii, at the sampling scatter of the first record; not a radius law)")
                rms = "%s; %s" % (e(r["rms"]), e(r["rms"] * math.sqrt(F["n"] / F["dof"])))
            else:
                coef = "not printed (dof 0: a dependence)"
                rms = "0 (dof 0)"
            rows.append([y, coef, per, rms, e(r["max_abs"]) if F["dof"] > 0 else "0 (dof 0)"])
        return "\n".join(lines) + "\n\n" + table(h, rows)

    def power3_text(self, key, var):
        P = self.D["fits"].get(key)
        if not P:
            return "- not evaluable"
        lines = ["- members: %s; the exact form Y = Y_0 + A %s^p through the 3 points (no residual; not a law):" % (", ".join(P["members"]), var)]
        for y in ("T_f", "R_t_over_rs", "R_t_over_2gE"):
            rts = P.get(y) or []
            if rts:
                lines.append("  - %s: %s" % (y, "; ".join("p %s, Y_0 %s, A %s" % (f(r["p"], "%.3f"), f(r["Y0"], "%.4f"), f(r["A"], "%.4g")) for r in rts)))
            else:
                lines.append("  - %s: no real exponent in [-12, 12] (no finite-intercept power through the 3 points)" % y)
        return "\n".join(lines)

    def catalog_text(self, names):
        h = ["member", "catalog read from", "chain columns per channel (kappa: columns)", "top channel", "largest column count", "4 a times it (upper bound)",
             "head comparison of the rerun preparation (N_B, E0, f_kin relative differences)"]
        rows = []
        for nm in names:
            cat = self.M[nm]["catalog"]
            if cat.get("onset_own_a") is None:
                rows.append([nm, "not available (%s)" % cat.get("source"), "-", "-", "-", "-", "-"])
                continue
            key = cat["from"]
            colstr = " ".join("%s:%s" % (k, v) for k, v in sorted(cat["columns"].items(), key=lambda kv: int(kv[0])))
            hc = cat.get("head_comparison")
            hcs = ", ".join("%s %s" % (k, e(v.get("rel_diff"))) for k, v in hc.items()) if hc else "- (status file)"
            src = "%s of %s%s" % (cat["source"], key, "" if key == nm else " (same preparation arguments)")
            rows.append([nm, src, colstr, cat.get("top"), cat.get("n_rad_max"), "%s a" % f(cat.get("onset_own_a"), "%.0f"), hcs])
        return table(h, rows)

    def sources(self, names, extra):
        h = ["member", "run json (as read)", "bytes", "md5", "script field"]
        rows = [[nm, self.M[nm]["meta"]["read_from"] + ((" (version 1's snapshot of %s)" % self.M[nm]["meta"]["snapshot_of"]) if self.M[nm]["meta"].get("snapshot_of") else ""),
                 self.M[nm]["meta"]["bytes"], self.M[nm]["meta"]["md5"], self.M[nm]["meta"]["script"]] for nm in names]
        return table(h, rows) + "\n\n" + "\n".join("- %s" % x for x in extra)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--readout", required=True)
    ap.add_argument("--sources", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--diff", required=True, help="v1.2: ladders_diff_v1_2.json (the values that changed from version 1.1)")
    a = ap.parse_args()
    D = json.load(open(a.readout))
    D["_diff"] = json.load(open(a.diff))
    SRC = json.load(open(a.sources))
    d = Doc(D, SRC)
    from compose_text_v1_2 import mass_doc, scaling_doc
    open(os.path.join(a.out_dir, "MASS_ladder_v1.2.md"), "w").write(mass_doc(d, D, SRC, PREP_RULE))
    open(os.path.join(a.out_dir, "SCALING_ladder_v1.2.md"), "w").write(scaling_doc(d, D, SRC, PREP_RULE))
    print("wrote v1.2 documents in", a.out_dir)


if __name__ == "__main__":
    main()
