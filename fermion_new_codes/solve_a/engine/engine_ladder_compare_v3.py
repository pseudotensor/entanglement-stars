#!/usr/bin/env python
"""engine_ladder_compare_v3.py (builder analyst): version 2 with each criterion's matched times taken over its own members (version 2 took
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


def rec_residual(r):
    ih = r.get("iteration_history") or []
    if not ih:
        return None
    if r.get("converged"):
        h = ih[-1]
        return max(h[1], h[2])
    u = r.get("unconverged") or {}
    return u.get("residual")


def slope_at_Rt(r):
    sj = r.get("surface_jets") or {}
    d = sj.get("d_N_plus_v")
    return abs(d[0]) if d else None


def pockets(r):
    p = r.get("n_pocket") or {}
    return int(sum(p.values())) if p else 0


def has_zero(r):
    return bool(r.get("trapping_zeros"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--member", action="append", required=True, help="sigma=path to the member's run json")
    ap.add_argument("--every", type=float, default=0.5)
    ap.add_argument("--md", default=None)
    ap.add_argument("--label", default="the fixed-preparation ladder at r_s/a 64")
    a = ap.parse_args()
    M = {}
    for s in a.member:
        k, p = s.split("=", 1)
        J = json.load(open(p))
        R = [r for r in J["records"] if "E" in r and "T_over_rs" in r]
        M[float(k)] = {"path": p, "J": J, "R": R, "byT": {round(r["T_over_rs"], 6): r for r in R}}
    sig = sorted(M)
    out = ["## %s: the members compared (engine_ladder_compare_v3.py)" % a.label, ""]
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
        res = rec_residual(f) if f else None
        A_ = M[s]["J"].get("args", {})
        mx = A_.get("mix", "plain")
        solver = mx + ((" step %s" % A_.get("mix_alpha")) if mx == "damped" else (" (at most %s Newton steps, %s Krylov vectors)" % (A_.get("nk_maxiter"), A_.get("nk_inner")) if mx == "newton" else ""))
        out.append("| %g | %s | %s | %s | %d | %.3f | %s | %s | %s | %s | %s |" % (
            s, M[s]["path"].split("/")[-1], M[s]["J"].get("script"), solver, len(R), R[-1]["T_over_rs"] if R else float("nan"),
            "%.3f" % f["T_over_rs"] if f else "none to %.3f" % R[-1]["T_over_rs"], "%.3f" % f["R_t"] if f else "", "%.4f" % f["R_t_over_2gE"] if f and f.get("R_t_over_2gE") is not None else "",
            "%.2e" % res if res is not None else "", "%.3f" % end if end is not None else ("" if not f else "none to %.3f" % R[-1]["T_over_rs"])))
    def times_for(members):
        common = set(M[members[0]]["byT"])
        for s_ in members[1:]:
            common &= set(M[s_]["byT"])
        tt = set(t for t in common if abs(t / a.every - round(t / a.every)) < 1e-9)
        tt |= set(round(form[s_], 6) for s_ in members if form[s_] is not None and round(form[s_], 6) in common)
        return sorted(tt)
    times = times_for(sig[1:] if len(sig) >= 3 else sig)
    hdr = "| T/rs |" + "".join(" R_t (s %g) | min(N-abs v) | R_c | E/E0 | pockets | residual | R_t bound |" % s for s in sig)
    out += ["", "## The matched-time table (record times present in %s, every %g r_s, plus their formation times; \"-\": no record yet)" % ("the members other than sigma/a %g" % sig[0] if len(sig) >= 3 else "all members", a.every), "", hdr,
            "|---|" + "---|" * (7 * len(sig))]
    for t in times:
        row = "| %.3f |" % t
        for s in sig:
            r = M[s]["byT"].get(round(t, 6))
            if r is None:
                row += " - | - | - | - | - | - | - |"
                continue
            res = rec_residual(r); sl = slope_at_Rt(r)
            row += " %s | %.4f | %.2f | %.7f | %d | %s | %s |" % ("%.3f" % r["R_t"] if has_zero(r) and r.get("R_t") is not None else "none", r["min_N_minus_absv"], r["R_c"], r["E_over_E0"],
                                                               pockets(r), "%.2e" % res if res is not None else "", "%.2e" % (res / sl) if (res is not None and sl and has_zero(r)) else "")
        out.append(row)
    s2 = min(sig)
    R2 = M[s2]["R"]
    band = [r["T_over_rs"] for r in R2 if rec_residual(r) is not None and 1e-4 <= rec_residual(r) <= 1e-3]
    over = next((r["T_over_rs"] for r in R2 if rec_residual(r) is not None and rec_residual(r) > 1e-3), None)
    out += ["", "## The reading rule (30.3) and the criterion (29.1)", ""]
    out.append("- sigma/a %g: %d records; residual below 1e-4 on %d, between 1e-4 and 1e-3 on %d%s, above 1e-3 first at %s" % (
        s2, len(R2), sum(1 for r in R2 if rec_residual(r) is not None and rec_residual(r) < 1e-4), len(band),
        " (T/rs " + ", ".join("%.3f" % x for x in band[:40]) + (", ..." if len(band) > 40 else "") + ")" if band else "",
        "T/rs %.3f (not certified from that record on)" % over if over is not None else "no record (to T/rs %.3f)" % R2[-1]["T_over_rs"]))
    for s in sig:
        if s == s2:
            continue
        rr = [rec_residual(r) for r in M[s]["R"] if rec_residual(r) is not None]
        out.append("- sigma/a %g: residual max %.2e over %d records (%d with the stop rule met)" % (s, max(rr) if rr else float("nan"), len(rr), sum(1 for r in M[s]["R"] if r.get("converged"))))

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
    out.append("- runs read: " + "; ".join("sigma/a %g %s" % (s, M[s]["path"]) for s in sig))
    out.append("- every number above is printed by the script; the members are read as they stand (a member still running is read to its last record).")
    txt = "\n".join(out)
    print(txt)
    if a.md:
        open(a.md, "w").write(txt + "\n")


if __name__ == "__main__":
    main()
