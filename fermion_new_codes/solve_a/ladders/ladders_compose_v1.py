#!/usr/bin/env python
"""ladders_compose_v1.py (composition builder, the solve lane): writes MASS_ladder_v1.md and SCALING_ladder_v1.md from the producer's
ladders_readout_v1.json only (no run file is opened here; every number is the producer's). Usage:
  python ladders_compose_v1.py --readout ladders_readout_v1.json --catalog-md5 FILE --sources SOURCES.json --out-dir DIR"""
import argparse
import json
import math
import os

PREP_RULE = ("Every member and control shares one preparation rule: the dust is the levitating well's zero-temperature filling within the "
             "sea set kappa = 1 to kappa_sea (engine prepare_dust with fill_lowest: the well of depth set by the release radius R_0, the "
             "lowest levels of all channels filled to the count N_B = r_s/(2 g m a), g = 1/4, the projection onto the clocked positive band), "
             "released from coordinate rest at R_0 = 4 r_s (2 r_s for the 2 preparation members), prepared at ell 4 and evolved at the member's "
             "ell; no member runs the isotropic catalog, so each member's onset radius is its own 4 a n_rad,max, printed beside the isotropic "
             "form 1.222 N_B^(1/3) a.")


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


class Doc:
    def __init__(self, D, src):
        self.D, self.M, self.src = D, D["members"], src

    def short(self, nm):
        m = self.M[nm]["meta"]
        return "(%g, %g) s%g t%g" % (m["m"], m["rs"], m["sigma"], m["tau"])

    def member_table(self, names, labels):
        h = ["member", "role", "m a", "r_s/a", "sigma/a", "tau/a (tau/r_s)", "sites", "N_B", "E/E_max", "p_F/m (isotropic form)",
             "own onset 4 a n_rad,max (/r_s)", "isotropic 1.222 N_B^(1/3) a (/r_s)", "exclusion floor 0.49 N_B^(1/3) a", "records (first to last T/r_s)",
             "last converged record", "first record above 1e-3", "end state"]
        rows = []
        for nm in names:
            M = self.M[nm]; m = M["meta"]; c = M["counts"]; v = M["convergence"]; cat = M["catalog"]
            own = ("%s a (%s)" % (f(cat.get("onset_own_a"), "%.0f"), f(cat.get("onset_own_rs"), "%.3f"))) if cat.get("onset_own_a") else "not available"
            rows.append([nm, labels.get(nm, m["role"]), f(m["m"], "%g"), f(m["rs"], "%g"), f(m["sigma"], "%g"),
                         "%g (%s)" % (m["tau"], f(c["tau_over_rs"], "%.5f")), m["nsites"], f(c["N_B"], "%.0f"), f(c["E_over_Emax"], "%.2f"),
                         f(c["p_F_over_m_iso"], "%.3f"), own, "%s a (%s)" % (f(c["onset_iso_a"], "%.1f"), f(c["onset_iso_rs"], "%.3f")),
                         f(c["excl_floor_a"], "%.1f") + " a", "%d (%s to %s)" % (v["records"], f(v["first"], "%.3f"), f(v["last"], "%.4f")),
                         f(v["last_converged"], "%.4f"), f(v["first_above_1e-3"], "%.4f"), self.end_state(nm)])
        return table(h, rows)

    def end_state(self, nm):
        M = self.M[nm]; m = M["meta"]; v = M["convergence"]; fo = M["formation"]
        T_end = float(m["args"].get("T_over_rs") or 0)
        bits = []
        if m.get("running"):
            bits.append("running on its card; the mirrored records read to %.4f" % v["last"])
        elif v["last"] is not None and v["last"] + 1e-9 < T_end:
            bits.append("stopped at %.4f of its requested %g" % (v["last"], T_end))
        else:
            bits.append("finished at %g" % v["last"])
        if v["first_above_1e-3"] is not None:
            bits.append("uncertified from %.4f (residual above 1e-3)" % v["first_above_1e-3"])
        if not fo:
            bits.append("no trapping zero in any record")
        return "; ".join(bits)

    def formation_table(self, names):
        h = ["member", "formation T/r_s", "R_t/a", "R_t/r_s", "R_t/(2 g E)", "E/E0", "zeros (a)", "simple (no tangency; slope)",
             "root gate sigma/R_t", "residual (kept)", "certification", "first episode closes at", "min(N - |v|) in the first episode (at)",
             "previous record T/r_s, min(N - |v|)"]
        rows = []
        for nm in names:
            M = self.M[nm]; fo = M["formation"]; ep = (M["episodes"] or [None])[0]
            if not fo:
                rows.append([nm, "none to %s" % f(M["convergence"]["last"], "%.4f")] + ["-"] * 12)
                continue
            rows.append([nm, f(fo["T"]), f(fo["R_t"], "%.3f"), f(fo["R_t_over_rs"]), f(fo["R_t_over_2gE"]), f(fo["E_over_E0"], "%.5f"),
                         ", ".join("%.3f" % z for z in fo["zeros"]), "%s; %s" % ("yes" if fo["simple"] else "no", e(fo["slope"])),
                         f(fo["gate_sigma_over_Rt"], "%.3f"), e(fo["residual"]), fo["label"],
                         (f(ep["closes_at"]) if ep and ep.get("closes_at") is not None else ("open to %s" % f(ep["last"]) if ep else "-")),
                         ("%s (%s)" % (f(ep["min_minNv"]), f(ep["min_at"]))) if ep else "-",
                         "%s, %s" % (f(fo["prev_record_T"]), f(fo["prev_minNv"]))])
        return table(h, rows)

    def episodes_text(self, names):
        lines = []
        for nm in names:
            eps = self.M[nm]["episodes"]
            if not eps:
                continue
            parts = []
            for i, ep in enumerate(eps[:6]):
                parts.append("%d: %s to %s (%d records; R_t %s to %s a; min(N - |v|) %s at %s; %s)" % (
                    i + 1, f(ep["first"]), f(ep["last"]), ep["records"], f(ep["R_t_first"], "%.3f"), f(ep["R_t_last"], "%.3f"),
                    f(ep["min_minNv"]), f(ep["min_at"]), ("closes at %s" % f(ep["closes_at"])) if ep["closes_at"] is not None else "open at the last record"))
            more = (" (and %d more)" % (len(eps) - 6)) if len(eps) > 6 else ""
            lines.append("- %s: %s%s" % (nm, "; ".join(parts), more))
        return "\n".join(lines)

    def compression_table(self, names):
        h = ["member", "min R_c over the counting records (a)", "at T/r_s", "min R_c/r_s", "own onset 4 a n_rad,max", "isotropic 1.222 N_B^(1/3) a",
             "exclusion floor", "min R_c at the last counting record"]
        rows = []
        for nm in names:
            M = self.M[nm]; rc = M["Rc_min"]; c = M["counts"]; cat = M["catalog"]
            rows.append([nm, f(rc.get("min_Rc"), "%.2f"), f(rc.get("at")), f(rc.get("min_Rc_over_rs"), "%.3f"),
                         (f(cat.get("onset_own_a"), "%.0f") + " a") if cat.get("onset_own_a") else "not available", f(c["onset_iso_a"], "%.1f") + " a",
                         f(c["excl_floor_a"], "%.1f") + " a", "yes (still falling at the end of the counting records)" if rc.get("over_end") else "no"])
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
                elif not r["counts"]:
                    row.append((f(r.get(key), fmt) if r.get(key) is not None else "none") + " (n.c.)")
                else:
                    v = r.get(key)
                    row.append(f(v, fmt) if v is not None else ("none" if key.startswith("R_t") else "-"))
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
                        f(r["EE0"], "%.5f"), f(r["minNv"]), f(r["gate"], "%.3f"), "" if r["counts"] else " (n.c.)"))
            rows.append(row)
        return table(h, rows)

    def convergence_table(self, names):
        h = ["member", "script; solver", "engine entries in the records", "records", "converged (counting)", "capped (converged False)",
             "residual max (at)", "max in the counting prefix", "below 1e-4 / 1e-4 to 1e-3", "non-counting records inside the 30.3 prefix",
             "first above 1e-3", "last converged record"]
        rows = []
        for nm in names:
            M = self.M[nm]; m = M["meta"]; v = M["convergence"]
            eng = "; ".join("%s (%s to %s, %d)" % (u["value"], f(u["first"], "%.3f"), f(u["last"], "%.3f"), u["records"]) for u in m["engine_entries"][:3])
            sol = m["args"].get("mix")
            if m["newton"]:
                sol = "newton (at most %s steps, %s Krylov vectors, set %s, ends above %s)" % (m["args"].get("nk_maxiter"), m["args"].get("nk_krylov"),
                                                                                           m["args"].get("nk_set"), m["args"].get("end_above"))
            else:
                sol = "plain (at most %s iterations)" % m["args"].get("maxit")
            rows.append([nm, "%s; %s" % (m["script"], sol), eng, v["records"], "%d (%d)" % (v["converged_true"], v["counting"]), v["capped"],
                         "%s (%s)" % (e(v["residual_max"]), f(v["residual_max_at"])), e(v["residual_max_in_strict_prefix"]),
                         "%d / %d" % (v["below_1e-4"], v["band_1e-4_1e-3"]), clocks_text(v["noncounting_inside_r303"]),
                         f(v["first_above_1e-3"]), f(v["last_converged"])])
        return table(h, rows)

    def excluded_text(self, names):
        lines = []
        for nm in names:
            ex = self.M[nm]["convergence"]["excluded"]
            if ex:
                s = "; ".join("%s (%s, %s)" % (f(x["T"]), ("capped" if not x["converged"] else "kept") + (", above 1e-3" if (x["residual"] or 0) > 1e-3 else ""), e(x["residual"])) for x in ex[:40])
                lines.append("- %s: %d records excluded: %s%s" % (nm, len(ex), s, (" (and %d more)" % (len(ex) - 40)) if len(ex) > 40 else ""))
        return "\n".join(lines) if lines else "- none"

    def spreads_table(self, pairs):
        h = ["member", "control", "kind", "dT_f (control - member)", "dR_t (a)", "d(R_t/r_s)", "d(R_t/(2gE))", "both formations certified",
             "common counting clocks (to)", "max |d E/E0|", "max |d min(N - |v|)|", "max |d R_c|/r_s", "max |d R_t| (a) where both hold a zero (count)"]
        rows = []
        for s in self.D["spreads"]:
            if (s["member"], s["control"]) not in pairs:
                continue
            rows.append([s["member"], s["control"], s["kind"], f(s.get("dT_f")), f(s.get("dR_t"), "%.3f"), f(s.get("dR_t_over_rs")),
                         f(s.get("dR_t_over_2gE")), {True: "yes", False: "no"}.get(s.get("both_certified"), "-"),
                         "%s (%s)" % (s.get("common_clocks", "-"), f(s.get("common_to"))), e(s.get("max_abs_dEE0")), e(s.get("max_abs_dminNv")),
                         e(s.get("max_abs_dRc_over_rs")), "%s (%s)" % (f(s.get("max_abs_dRt"), "%.3f"), s.get("Rt_common_count", "-"))])
        return table(h, rows)

    def fit_text(self, key, ylabels=("T_f", "R_t_over_rs", "R_t_over_2gE")):
        F = self.D["fits"].get(key)
        if not F or isinstance(F, (str, list)):
            return "- not evaluable (%s)" % (F if F else "too few members")
        lines = ["- model: %s" % F["model"], "- members: %s (%d members, %d coefficients, dof %d)" % (", ".join(F["members"]), F["n"], F["coefficients"], F["dof"])]
        if "x" in F:
            lines.append("- x per member: %s" % ", ".join("%s %s" % (nm, f(x, "%.5f")) for nm, x in zip(F["members"], F["x"])))
        if "z" in F:
            lines.append("- (z, h) per member: %s" % ", ".join("%s (%s, %s)" % (nm, f(z), f(h)) for nm, z, h in zip(F["members"], F["z"], F["h"])))
        if F.get("uncertified"):
            lines.append("- uncertified formation records inside this fit (labelled; excluded from every law): %s" % ", ".join(F["uncertified"]))
        h = ["Y", "coefficients", "member: Y, fit, residual", "rms residual", "max |residual|"]
        rows = []
        for y in ylabels:
            if y not in F:
                continue
            r = F[y]
            per = "; ".join("%s: %s, %s, %s" % (nm, f(a_), f(b_), "%+.4f" % c_) for nm, a_, b_, c_ in zip(F["members"], r["y"], r["pred"], r["residual"]))
            coef = ", ".join("c%d %s" % (i, "%.6g" % c) for i, c in enumerate(r["coef"])) if F["dof"] > 0 else "not printed (dof 0: a dependence)"
            rows.append([y, coef, per, e(r["rms"]) if F["dof"] > 0 else "0 (dof 0)", e(r["max_abs"]) if F["dof"] > 0 else "0 (dof 0)"])
        return "\n".join(lines) + "\n\n" + table(h, rows)

    def catalog_text(self, names):
        h = ["member", "catalog read from", "columns per channel (kappa: radial modes)", "top channel", "n_rad,max", "4 a n_rad,max",
             "head comparison of the rerun preparation (N_B, E0, f_kin relative differences)"]
        rows = []
        seen = set()
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
            seen.add(key)
        return table(h, rows)

    def sources(self, names, extra):
        h = ["member", "run json (as read)", "bytes", "md5", "script field"]
        rows = [[nm, self.M[nm]["meta"]["read_from"] + ("" if self.M[nm]["meta"]["read_from"] == self.M[nm]["meta"]["path"] else
                                                       " (snapshot of %s)" % self.M[nm]["meta"]["path"]),
                 self.M[nm]["meta"]["bytes"], self.M[nm]["meta"]["md5"], self.M[nm]["meta"]["script"]] for nm in names]
        return table(h, rows) + "\n\n" + "\n".join("- %s" % x for x in extra)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--readout", required=True)
    ap.add_argument("--sources", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    D = json.load(open(a.readout))
    SRC = json.load(open(a.sources))
    d = Doc(D, SRC)
    M = D["members"]
    from compose_text_v1 import mass_doc, scaling_doc  # the fixed wording of the two documents (no numbers of its own)
    open(os.path.join(a.out_dir, "MASS_ladder_v1.md"), "w").write(mass_doc(d, D, SRC, PREP_RULE))
    open(os.path.join(a.out_dir, "SCALING_ladder_v1.md"), "w").write(scaling_doc(d, D, SRC, PREP_RULE))
    print("wrote", os.path.join(a.out_dir, "MASS_ladder_v1.md"), os.path.join(a.out_dir, "SCALING_ladder_v1.md"))


if __name__ == "__main__":
    main()
