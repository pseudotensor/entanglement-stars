#!/usr/bin/env python
"""engine_ladder_compare_v6.py (busi builder analyst; rc_ruling_v1 (busi/notes/REVIEWA_rc_floor_v1.md) sections 5.2 to 5.4 and 7, the
two-residual stop rule for Newton records): version 5 with (1) a Newton record read by both of its residuals: the kept (as version 5) and the
confirming residual, read from solve_v32.confirming_residual (the kept round's for a multi-round record: later entries of the iteration
history read nowhere; a record without solve_v32 reads the history's last entry, as version 5 printed it); (2) the 30.3 extent: a run is
certified until the record before its first record at which the larger of the 2 residuals exceeds 1e-3 (a Newton record missing either
residual is not read as certified: it ends the extent too, as rc_audit_v1 read it), and not from that record on; (3) the band: a Newton record
prints in a band when either residual is above 1e-4 (and neither above 1e-3); (4) the ratio (confirming over kept, the plain step's
amplification, a diagnostic and no test) printed in the Newton table beside the 2 residuals, with each record's reading and which residual
put it there; every table that prints a Newton record's residual prints both and the ratio, and the residual over the root slope uses the
larger of the 2; (5) the certification line printed for every member (version 5 printed the 30.3 line for the smallest sigma only; the
criterion still reads that member's extent, as before) and --json-out writes each member's per-record reading (the acceptance of 5.2 is read
from it). A plain record carries 1 residual and is read by it exactly as version 5 read it (band 1e-4 <= r <= 1e-3, above 1e-3 ends the
extent). The engine tolerance --stop-fields reads nowhere here. Version 5: engine_ladder_compare_v5.py (builder analyst; engine spec 33.2 (1) and 33.5 (4): "COLLAPSE_ladder_v1.3 when the v32 records
change, its v31 solver rows relabelled"): version 4 with (1) an engine_collapse_v32 record's residuals read from its own solve record
(solve_v32: the kept residual and the confirming step's of the solve the record keeps, the continuation's last stage when it ran, and its
evaluation count summed over the solves), since the iteration history of a continued record also holds the failed direct solve's and the
midpoint stage's evaluations, a different problem's; (2) the solver column printed from the run: a v32 member its options (Newton steps,
Krylov vectors per cycle, restarts, the line search, the continuation, the end rule), a v31 member "at most N Newton steps; LGMRES,
scipy's default 30 Krylov vectors per Newton step (--nk-inner inert)" (33.2 (1)); (3) the Newton table gains the solve the record kept and
the stop rule as the v32 record states it (meets_stop_rule). Nothing else changes. Version 4: engine_ladder_compare_v4.py (builder analyst; the hreduce lane's review of COLLAPSE_ladder_v1, deliveries/hreduce/
dm7_ladder_review.tex items 2 to 5 with its red team, inbox #11): version 3 with (1) a coverage table first (each member's script, solver,
records and first and last T/rs: the members need not be one completed run); (2) the kernel stated as the three-dimensional Gaussian of
standard deviation ell with per-axis variance sigma^2 = ell^2 (no cell convolution in this back-map, so no a^2/12 term); (3) for a
Newton-Krylov member two residuals per record, the kept one (Newton's acceptance: the retained evaluation's max(dN, dv), the record's
unconverged residual where it is flagged) and the confirming plain step's (the last entry of its iteration history), never identified;
for a plain member the last iterate's (version 3's); (4) the column "R_t bound" renamed "residual over the root slope", a sensitivity
diagnostic and not a root-error bound (no inverse-response bound is supplied); (5) the definitions printed: R_t the outermost zero of the
dense N + v (a geometric root, not the applied-field matter root, not a tracked material surface), R_c the dust-energy centroid, E/E0
against the common preparation's charge E0 (the members' initial E/E0 differ). Nothing else changes. Version 3: engine_ladder_compare_v3.py (builder analyst): version 2 with each criterion's matched times taken over its own members (version 2 took
the times common to all members, so the factor-2 reading stopped at the shortest member's last record) and the matched-time table printed
over the times common to the members other than the smallest sigma when there are three or more, a member without that record printed
"-"; nothing else changes. Version 2: engine_ladder_compare_v2.py (builder analyst): version 1 with the R_t bound printed in scientific notation (version 1 printed it to
three decimals, so every bound read 0.000) and each member's self-consistent solver (the run's --mix, with its step or Newton's
settings) in the members' table; nothing else changes. Version 1's docstring: engine_ladder_compare_v1.py (builder analyst; engine spec 30.3: "engine_ladder_compare_v1 (a new file) prints the matched-time table of
the three members (formation, R_t, min(N - |v|), R_c, E/E0, the pockets, the residual column) and the criterion's verdict, read in turn
39"). Reads the fixed-preparation ladder's member runs (engine_collapse_v27 jsons, one per sigma/a) and prints:
(1) per member: the run's last record, the first trapping surface's formation (the first record holding a trapping zero) with its R_t,
    R_t/(2 g E) and the record's residual, and the first record after it holding none (the first episode's end);
(2) the matched-time table at every record time present in all members that is a multiple of --every r_s, plus every member's formation
    time: per member R_t (or none), min(N - |v|), R_c, E/E0, the pockets (the sum over channels of n_pocket), the self-consistent
    iteration's residual (the returned iterate's max(max |N_out - N_in|, max |v_out - v_in|), the engine's own rule) and the residual's
    bound on R_t (the residual over the slope d(N + v)/dr at R_t, surface_jets' first entry; 30.3: "that residual entering the member's
    R_t as the bound");
(3) the reading rule of 30.3 applied as written: sigma 2's records with residual below 1e-4 are read with the bound; from the first
    record above 1e-3 on, sigma 2 is "not certified" and the ladder reads sigma 4 and 8 as a factor-2 control; records between 1e-4 and
    1e-3 are counted and listed (the rule names no reading for them; not ruled here); then 29.1's criterion, "the formation within
    0.5 r_s and the radius at matched times within 10 percent across sigma 2, 4, 8": the formation spread (max - min of the members'
    formation times), and over the matched times at which every compared member holds a zero the largest R_t spread (max/min - 1); the
    matched times at which some but not all compared members hold a zero are counted and listed separately (the criterion's text does
    not say how they count; printed, not ruled). The verdict line states each part with its numbers; the lead reads it.
Every number is printed by the script. Args: --member sigma=json (repeated) --every (0.5) --md --label."""
import argparse
import json

CAP = 1e-3      # version 6 (rule 30.3 amended): certified until the record before the first record with a residual above 1e-3 (either, for Newton)
BAND = 1e-4     # version 6: a record prints in a band when a residual is above 1e-4 (either, for Newton)


NEWTON = set()  # the members whose runs used --mix newton (version 4)


def v32_solve(r):
    """version 5: an engine_collapse_v32 record's own solve record, or None."""
    sv = r.get("solve_v32")
    return sv if isinstance(sv, dict) and sv.get("kept_residual") is not None else None


def rec_residual(r, newton=False):
    sv = v32_solve(r)
    if sv is not None:
        return sv["kept_residual"]
    """version 4: the kept residual (a Newton member: the retained evaluation's, the minimum over its evaluations, the confirming
    step excluded; a plain member: the last iterate's when converged); the unconverged record's stored residual otherwise."""
    ih = r.get("iteration_history") or []
    if not ih:
        return None
    if not r.get("converged"):
        u = r.get("unconverged") or {}
        return u.get("residual")
    if newton:
        ev = [max(h[1], h[2]) for h in ih[1:-1]]
        return min(ev) if ev else None
    h = ih[-1]
    return max(h[1], h[2])


def confirm_residual(r):
    """version 4: a Newton member's confirming plain step's residual (the last entry of its iteration history)."""
    sv = v32_solve(r)
    if sv is not None:
        return sv.get("confirming_residual")
    ih = r.get("iteration_history") or []
    return max(ih[-1][1], ih[-1][2]) if ih else None


def confirm_residual_v6(r):
    """version 6 (rc_ruling_v1 5.3): the confirming residual the stop rule reads, solve_v32.confirming_residual (the kept round's; rc_audit_v1's
    choice); where the record has no solve_v32 entry, version 5's reader (the iteration history's last entry)."""
    sv = r.get("solve_v32") if isinstance(r.get("solve_v32"), dict) else {}
    rc = sv.get("confirming_residual")
    return rc if rc is not None else confirm_residual(r)


def read_v6(r, newton):
    """version 6: (kept, confirming, larger, ratio) of a record; a plain record (kept, None, kept, None); for a Newton record "larger" is None
    when either residual is missing."""
    k_ = rec_residual(r, newton)
    if not newton:
        return k_, None, k_, None
    c_ = confirm_residual_v6(r)
    big = max(k_, c_) if (k_ is not None and c_ is not None) else None
    ratio = (c_ / k_) if (k_ not in (None, 0) and c_ is not None) else None
    return k_, c_, big, ratio


def which_v6(k_, c_, thr):
    """version 6: which residual is above thr: "kept", "confirming", "both", or "" (neither); a missing one is named "missing"."""
    if k_ is None or c_ is None:
        return "missing %s" % ("kept" if k_ is None else "confirming")
    a_, b_ = k_ > thr, c_ > thr
    return "both" if (a_ and b_) else ("kept" if a_ else ("confirming" if b_ else ""))


def certify_v6(R, newton):
    """version 6 (30.3 amended): the member's extent and bands. Newton: the first record at which the larger of the 2 residuals exceeds 1e-3
    (or a residual is missing) ends the extent; a band record has either residual above 1e-4 and neither above 1e-3. Plain: version 5's rule
    on the one residual (a record without a residual is skipped; band 1e-4 <= r <= 1e-3)."""
    rows = []
    for i, r in enumerate(R):
        k_, c_, big, ratio = read_v6(r, newton)
        if newton:
            over = big is None or big > CAP
            band = (not over) and big > BAND
            by = which_v6(k_, c_, CAP if over else BAND) if (over or band) else ""
        else:
            over = k_ is not None and k_ > CAP
            band = k_ is not None and BAND <= k_ <= CAP
            by = "kept" if (over or band) else ""
        rows.append({"i": i, "T_over_rs": r["T_over_rs"], "kept_residual": k_, "confirming_residual": c_, "larger": big, "ratio": ratio,
                     "above_1e-3": bool(over), "band": bool(band), "by": by})
    k = next((x["i"] for x in rows if x["above_1e-3"]), None)
    for x in rows:
        x["certified"] = (k is None or x["i"] < k)
    ext = {"first_fail_T": rows[k]["T_over_rs"] if k is not None else None, "first_fail_i": k, "first_fail_by": rows[k]["by"] if k is not None else None,
           "certified_through_T": (rows[k - 1]["T_over_rs"] if (k is not None and k > 0) else (None if k == 0 else (rows[-1]["T_over_rs"] if rows else None))),
           "n_certified": (k if k is not None else len(rows)), "records": len(rows),
           "band_T": [x["T_over_rs"] for x in rows if x["band"]], "band_by": [x["by"] for x in rows if x["band"]],
           "band_T_certified": [x["T_over_rs"] for x in rows if x["band"] and x["certified"]],
           "below_1e-4": sum(1 for x in rows if x["larger"] is not None and not x["band"] and not x["above_1e-3"]),
           "missing": sum(1 for x in rows if x["larger"] is None)}
    return ext, rows


def fres_v6(k_, c_, ratio, newton):
    """version 6: a residual cell: the kept alone for a plain record; "kept; confirming (x ratio)" for a Newton record."""
    if not newton:
        return "%.2e" % k_ if k_ is not None else ""
    return "%s; %s (x %s)" % ("%.2e" % k_ if k_ is not None else "-", "%.2e" % c_ if c_ is not None else "-", "%.3g" % ratio if ratio is not None else "-")


def slope_at_Rt(r):
    sj = r.get("surface_jets") or {}
    d = sj.get("d_N_plus_v")
    return abs(d[0]) if d else None


def pockets(r):
    p = r.get("n_pocket") or {}
    return int(sum(p.values())) if p else 0


def has_zero(r):
    return bool(r.get("trapping_zeros"))


def solver_label(J):
    """version 5: the solver as the run states it."""
    A_ = J.get("args", {}) or {}
    mx = A_.get("mix", "plain")
    if mx == "damped":
        return "damped step %s" % A_.get("mix_alpha")
    if mx != "newton":
        return mx
    if "engine_collapse_v32" in str(J.get("script")) and A_.get("nk_krylov") is not None:
        return ("newton (v32: at most %s Newton steps; LGMRES %s Krylov vectors per cycle, %s restarts; line search %s, at most %s halvings; "
                "source continuation %s; the member ends above %s)" % (A_.get("nk_maxiter"), A_.get("nk_krylov"), A_.get("nk_restarts"), A_.get("nk_line_search"),
                                                                     A_.get("nk_halvings"), A_.get("source_continuation"), A_.get("end_above")))
    return "newton (at most %s Newton steps; LGMRES, scipy's default 30 Krylov vectors per Newton step (--nk-inner inert))" % A_.get("nk_maxiter")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--member", action="append", required=True, help="sigma=path to the member's run json")
    ap.add_argument("--every", type=float, default=0.5)
    ap.add_argument("--md", default=None)
    ap.add_argument("--label", default="the fixed-preparation ladder at r_s/a 64")
    ap.add_argument("--json-out", default=None, help="(version 6) write each member's certification (extent, bands, per-record reading) as json")
    a = ap.parse_args()
    M = {}
    for s in a.member:
        k, p = s.split("=", 1)
        J = json.load(open(p))
        R = [r for r in J["records"] if "E" in r and "T_over_rs" in r]
        M[float(k)] = {"path": p, "J": J, "R": R, "byT": {round(r["T_over_rs"], 6): r for r in R}}
        if (J.get("args") or {}).get("mix") == "newton":
            NEWTON.add(float(k))
    sig = sorted(M)
    out = ["## %s: the members compared (engine_ladder_compare_v6.py)" % a.label, "",
           "- the kernel: the three-dimensional Gaussian of standard deviation ell, per-axis variance sigma^2 = ell^2 (sigma/a = ell/a; this back-map has no cell convolution, so no a^2/12 term; the paper's compact kernel, ell^2/15 + a^2/12, is a different object).",
           "- the definitions: R_t the outermost zero of the dense N + v (a geometric root, not the applied-field matter root and not a tracked material surface); R_c the dust-energy centroid; E/E0 against the common preparation's charge E0 (the members' initial E/E0 differ); the residual the self-consistent iteration's max(max |N_out - N_in|, max |v_out - v_in|): for a Newton-Krylov member 2 residuals of the same map 1 plain step apart, the kept (at the kept iterate) and the confirming (at its output, the state the record stores; solve_v32's, the kept round's), printed \"kept; confirming (x ratio)\", the ratio the plain step's amplification (a diagnostic, no test); the stop rule (30.3 amended, rc_ruling_v1 7) reads both; \"residual over the root slope\" is the residual (for a Newton member the larger of the 2) over |d(N + v)/dr| at R_t, a sensitivity diagnostic, not a root-error bound.",
           "", "| sigma/a | script | solver | records | first T/rs | last T/rs |", "|---|---|---|---|---|---|"]
    for s_ in sorted(M):
        A_ = M[s_]["J"].get("args", {})
        out.append("| %g | %s | %s | %d | %.3f | %.3f |" % (s_, M[s_]["J"].get("script"), A_.get("mix", "plain"), len(M[s_]["R"]), M[s_]["R"][0]["T_over_rs"] if M[s_]["R"] else float("nan"), M[s_]["R"][-1]["T_over_rs"] if M[s_]["R"] else float("nan")))
    out.append("")
    out.append("| sigma/a | run | script | solver | records | last T/rs | formation T/rs | its R_t | its R_t/(2 g E) | its residual | first episode ends T/rs |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|")
    form = {}
    for s in sig:
        R = M[s]["R"]
        f = next((r for r in R if has_zero(r)), None)
        form[s] = f["T_over_rs"] if f else None
        end = None
        if f:
            end = next((r["T_over_rs"] for r in R if r["T_over_rs"] > f["T_over_rs"] and not has_zero(r)), None)
        fk_, fc_, fb_, fr_ = read_v6(f, s in NEWTON) if f else (None, None, None, None)
        A_ = M[s]["J"].get("args", {})
        mx = A_.get("mix", "plain")
        solver = solver_label(M[s]["J"])
        out.append("| %g | %s | %s | %s | %d | %.3f | %s | %s | %s | %s | %s |" % (
            s, M[s]["path"].split("/")[-1], M[s]["J"].get("script"), solver, len(R), R[-1]["T_over_rs"] if R else float("nan"),
            "%.3f" % f["T_over_rs"] if f else "none to %.3f" % R[-1]["T_over_rs"], "%.3f" % f["R_t"] if f else "", "%.4f" % f["R_t_over_2gE"] if f and f.get("R_t_over_2gE") is not None else "",
            fres_v6(fk_, fc_, fr_, s in NEWTON) if f else "", "%.3f" % end if end is not None else ("" if not f else "none to %.3f" % R[-1]["T_over_rs"])))
    def times_for(members):
        common = set(M[members[0]]["byT"])
        for s_ in members[1:]:
            common &= set(M[s_]["byT"])
        tt = set(t for t in common if abs(t / a.every - round(t / a.every)) < 1e-9)
        tt |= set(round(form[s_], 6) for s_ in members if form[s_] is not None and round(form[s_], 6) in common)
        return sorted(tt)
    times = times_for(sig[1:] if len(sig) >= 3 else sig)
    hdr = "| T/rs |" + "".join(" R_t (s %g) | min(N-abs v) | R_c | E/E0 | pockets | residual (%s) | residual over the root slope |" % (s, "kept; confirming (x ratio)" if s in NEWTON else "kept") for s in sig)
    out += ["", "## The matched-time table (record times present in %s, every %g r_s, plus their formation times; \"-\": no record yet)" % ("the members other than sigma/a %g" % sig[0] if len(sig) >= 3 else "all members", a.every), "", hdr,
            "|---|" + "---|" * (7 * len(sig))]
    for t in times:
        row = "| %.3f |" % t
        for s in sig:
            r = M[s]["byT"].get(round(t, 6))
            if r is None:
                row += " - | - | - | - | - | - | - |"
                continue
            rk_, rc_, res, rr_ = read_v6(r, s in NEWTON); sl = slope_at_Rt(r)
            row += " %s | %.4f | %.2f | %.7f | %d | %s | %s |" % ("%.3f" % r["R_t"] if has_zero(r) and r.get("R_t") is not None else "none", r["min_N_minus_absv"], r["R_c"], r["E_over_E0"],
                                                               pockets(r), fres_v6(rk_, rc_, rr_, s in NEWTON), "%.2e" % (res / sl) if (res is not None and sl and has_zero(r)) else "")
        out.append(row)
    s2 = min(sig)
    R2 = M[s2]["R"]
    CERT = {s: certify_v6(M[s]["R"], s in NEWTON) for s in sig}   # version 6: every member's reading (30.3 amended)
    over = CERT[s2][0]["first_fail_T"]
    out += ["", "## The reading rule (30.3 amended, rc_ruling_v1 7: a Newton record by both residuals) and the criterion (29.1)", ""]
    for s in sig:
        ex, rows_ = CERT[s]
        nw = s in NEWTON
        what = "the larger of the kept and the confirming residual" if nw else "the residual (a plain member: 1 residual)"
        bands = ["%.3f%s" % (t_, (" (%s)" % b_) if nw else "") for t_, b_ in zip(ex["band_T"], ex["band_by"])]
        out.append("- sigma/a %g%s: %d records, read by %s; at most 1e-4 on %d, in the band (%s) on %d%s%s; above 1e-3 first at %s; certified %s%s" % (
            s, " (the criterion's member)" if s == s2 else "", ex["records"], what, ex["below_1e-4"],
            "either residual above 1e-4" if nw else "1e-4 to 1e-3", len(ex["band_T"]),
            (" (T/rs " + ", ".join(bands[:40]) + (", ..." if len(bands) > 40 else "") + ")") if bands else "",
            ("; a residual missing on %d (not read as certified)" % ex["missing"]) if ex["missing"] else "",
            ("T/rs %.3f (%s; not certified from that record on)" % (ex["first_fail_T"], ex["first_fail_by"])) if ex["first_fail_T"] is not None else "no record (to T/rs %.3f)" % M[s]["R"][-1]["T_over_rs"],
            ("through T/rs %.3f (%d records)" % (ex["certified_through_T"], ex["n_certified"])) if ex["certified_through_T"] is not None else "none (0 records)",
            "" if nw else ("; %d with the stop rule met" % sum(1 for r in M[s]["R"] if r.get("converged")))))

    def criterion(members, t_end):
        fs = [form[s] for s in members]
        fpart = ("formation spread %.3f r_s (%s)" % (max(fs) - min(fs), ", ".join("sigma %g at %.3f" % (s, form[s]) for s in members))) if all(f is not None for f in fs) else \
            "formation: %s" % ", ".join("sigma %g %s" % (s, ("at %.3f" % form[s]) if form[s] is not None else "none to %.3f" % M[s]["R"][-1]["T_over_rs"]) for s in members)
        allz, somez = [], []
        for t in times_for(members):
            if t_end is not None and t >= t_end:
                continue
            rs_ = [M[s]["byT"][round(t, 6)] for s in members]
            z = [has_zero(r) for r in rs_]
            if all(z):
                Rt = [r["R_t"] for r in rs_]
                allz.append((t, max(Rt) / min(Rt) - 1))
            elif any(z):
                somez.append((t, sum(z)))
        rpart = ("largest R_t spread %.4f (max/min - 1) over %d matched times with a zero in every member (at T/rs %.3f)" % (max(x[1] for x in allz), len(allz), max(allz, key=lambda x: x[1])[0])) if allz else "no matched time with a zero in every member"
        spart = ("%d matched times with a zero in some but not all members (T/rs %s)" % (len(somez), ", ".join("%.3f (%d of %d)" % (t, k, len(members)) for t, k in somez[:30]) + (", ..." if len(somez) > 30 else ""))) if somez else "no matched time with a zero in some but not all members"
        if t_end is not None and any(form[s] is not None and form[s] >= t_end for s in members):
            fpart += " (a formation at or after the first record above 1e-3 is not certified)"
        okf = None if any(f is None for f in fs) else (max(fs) - min(fs) <= 0.5 and not (t_end is not None and any(f >= t_end for f in fs)))
        okr = bool(allz) and max(x[1] for x in allz) <= 0.10
        return fpart, rpart, spart, okf, okr
    full = sig
    fpart, rpart, spart, okf, okr = criterion(full, over)
    out.append("- 29.1 across sigma/a %s%s: %s; %s; %s. Formation within 0.5 r_s: %s; radius within 10 percent at the matched times with a zero in every member: %s." % (
        ", ".join("%g" % s for s in full), (" (matched times before sigma %g's first record above 1e-3, T/rs %.3f)" % (s2, over)) if over is not None else "", fpart, rpart, spart,
        "not evaluable (a member holds no zero in its records)" if okf is None else ("yes" if okf else "no"), "yes" if okr else ("no" if rpart.startswith("largest") else "not evaluable")))
    if len(sig) >= 3:
        f2 = [s for s in sig if s != s2]
        fpart2, rpart2, spart2, okf2, okr2 = criterion(f2, None)
        out.append("- the factor-2 reading (sigma/a %s, all records): %s; %s; %s. Formation within 0.5 r_s: %s; radius within 10 percent: %s." % (
            ", ".join("%g" % s for s in f2), fpart2, rpart2, spart2, "not evaluable (a member holds no zero in its records)" if okf2 is None else ("yes" if okf2 else "no"), "yes" if okr2 else ("no" if rpart2.startswith("largest") else "not evaluable")))
    for s in sorted(NEWTON):
        out += ["", "### The Newton-Krylov member sigma/a %g: its two residuals per record (the kept, Newton's acceptance; the confirming plain step's)" % s, "",
                "| T/rs | stop rule (engine flag) | evaluations | residual (kept) | residual (confirming step) | ratio (confirming / kept) | reading (30.3 amended) | the solve kept |", "|---|---|---|---|---|---|---|---|"]
        rowsT = {round(x["T_over_rs"], 6): x for x in CERT[s][1]}
        for r in M[s]["R"]:
            if not r.get("iteration_history"):
                continue
            k_, c_, big_, ratio_ = read_v6(r, True)
            x_ = rowsT.get(round(r["T_over_rs"], 6)) or {}
            rd_ = ("above 1e-3 (%s)" % x_["by"] if x_.get("above_1e-3") else ("band (%s)" % x_["by"] if x_.get("band") else "at most 1e-4")) if x_ else "-"
            if x_ and not x_.get("certified"):
                rd_ += "; not certified"
            sv = v32_solve(r)
            stop = ("met" if sv.get("meets_stop_rule") else "not met") if sv is not None else ("met" if r.get("converged") else "not met")
            nev = sv.get("evaluations") if sv is not None else r.get("iterations")
            kept = (sv.get("kept_from") or "-") if sv is not None else "-"
            out.append("| %.3f | %s | %s | %s | %s | %s | %s | %s |" % (r["T_over_rs"], stop, nev, "%.2e" % k_ if k_ is not None else "-", "%.2e" % c_ if c_ is not None else "-",
                                                             "%.3g" % ratio_ if ratio_ is not None else "-", rd_, kept))
    out.append("- runs read: " + "; ".join("sigma/a %g %s" % (s, M[s]["path"]) for s in sig))
    out.append("- every number above is printed by the script; the members are read as they stand (a member still running is read to its last record).")
    txt = "\n".join(out)
    print(txt)
    if a.md:
        open(a.md, "w").write(txt + "\n")
    if a.json_out:   # version 6: the certification per member, every per-record reading
        json.dump({"script": "engine_ladder_compare_v6.py", "cap": CAP, "band": BAND, "label": a.label,
                   "members": [{"sigma": s, "path": M[s]["path"], "newton": s in NEWTON, "extent": CERT[s][0], "rows": CERT[s][1]} for s in sig]},
                  open(a.json_out, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
