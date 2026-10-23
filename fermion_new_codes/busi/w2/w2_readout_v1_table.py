#!/usr/bin/env python3
"""W2 readout v1 (busi): the table script. Prints the readout from w2_readout_v1.py's json (no run files read).

Usage: python -I w2_readout_v1_table.py w2_readout_v1.json [--unresolved-only] [--from 11.5]
"""
import json, sys


def f(x, fm="%+.3e"):
    return "-" if x is None else fm % x


def main():
    a = sys.argv[1:]
    J = json.load(open(a[0]))
    allc = "--unresolved-only" not in a
    t_from = float(a[a.index("--from") + 1]) if "--from" in a else 11.5
    R = J["records"]
    print("# W2 readout v1 (w2_readout_v1.py md5 %s; written %s)" % (J["md5_self"], J["written_utc"]))
    print("E0 %.12g; records %d, T/r_s 0 .. %.4f; complete to %.1f: %s; floor %.0e, S2 remainder %.0e" % (
        J["E0"], len(R), J["T_last"], J["t_end"], "YES" if J["complete"] else "NO (test on the records that exist)", J["floor"], J["S2_remainder"]))
    print("record-solve floor %.0e (a record-solve move above it is counted as an unresolved reassignment)" % J.get("floor_record_solve", 0))
    if J.get("prim_log"):
        L = J["prim_log"]
        print("prim log: last T/rs %s, STOP %d, Traceback %d" % (L["last_T_over_rs_in_log"], L["n_STOP"], L["n_Traceback"]))
    print("\n## inputs")
    for k, v in J["inputs"].items():
        print("  %-8s %s  %s" % (k, v["md5"], v["path"]))
    print("\n## composition checks")
    for k, v in J["checks"].items():
        if isinstance(v, dict) and "n" in v:
            print("  %-26s n %3d  max |dE|/E0 %s  missing %s" % (k, v["n"], f(v["max_abs_dE_over_E0"], "%.2e"), v["missing"][:6]))
        else:
            print("  %-26s %s" % (k, v))
    print("\n## the composed history from T/r_s %.3f (moves of E0; class by the Reorientation-5 criterion)" % t_from)
    print("T/rs | source | printable | E/E0 | dE | start | resolved (n) | unresolved: prim would-be (n) / overlap sw (n) / rec. reassign | remainder | pk | lvl | class | cause")
    for x in R:
        if x["T_over_rs"] < t_from - 1e-9:
            continue
        at, b = x["attribution"], x["bins"]
        src = {"certified": "v36f", "anchor": "anc", "prim": "prim"}[x["segment"]]
        print("%.3f | %s | %s | %.9f | %s | %s | %s (%d) | %s (%d) / %s (%d) / %s | %s%s | %d | %s/%s | %s | %s" % (
            x["T_over_rs"], src, x["printable"], x["E_over_E0"], f(x["dE_over_E0"]), f(b.get("start")),
            f(at["resolved_jump_over_E0"]), at["n_resolved_diabatic"], f(at["unresolved_would_be_over_E0"]), at["n_unresolved"],
            f(at["overlap_switch_over_E0"]), at["n_overlap_switch"], f(at["record_reassign_over_E0"]),
            f(at["remainder_over_E0"] if at["remainder_over_E0"] is not None else at["unattributed_over_E0"]),
            "*" if x["remainder_above_S2"] else ("u" if at["unattributed_over_E0"] is not None else ""),
            x["pockets"], x["level"] if x["level"] is not None else "-", x["nsub"] if x["nsub"] is not None else "-",
            x["class"], x["cause"]))
    print("  (* remainder above S2's threshold; u: v36f move with no bins, printed in the remainder column)")
    print("\n## sums by segment (of E0)")
    for k, s in J["segments"].items():
        print("  %-18s n %3d  dE %s  resolved %s (%d diabatic of %d)  unresolved prim %s (%d)  overlap sw %s (%d)  rec. reassign %s  remainder %s  start %s  unattributed %s" % (
            k, s["n"], f(s["dE_over_E0"]), f(s["resolved_jump_over_E0"]), s["n_resolved_diabatic"], s["n_resolved"],
            f(s["unresolved_would_be_over_E0"]), s["n_unresolved"], f(s["overlap_switch_over_E0"]), s["n_overlap_switch"],
            f(s["record_reassign_over_E0"]), f(s["remainder_over_E0"]), f(s["start_over_E0"]), f(s["unattributed_over_E0"])))
        print("  %-18s printable yes %d no %d; classes %s; max |remainder| %s; max |closure| %s; max E_rec/prev mismatch %s; unconverged %s; E/E0 %s" % (
            "", s["printable_yes"], s["printable_no"], s["classes"], f(s["max_abs_remainder_over_E0"], "%.2e"), f(s["max_abs_closure"], "%.1e"),
            f(s["max_E_rec_prev_mismatch"], "%.1e"), s["unconverged"], s["E_over_E0_first_last"] and ["%.9f" % v for v in s["E_over_E0_first_last"]]))
    print("\n## crossings of the composed history's prim records (%s)" % ("every one" if allc else "unresolved ones only"))
    for x in R:
        cr = x.get("crossings") or []
        if not cr:
            continue
        sel = cr if allc else [c for c in cr if c["class"] == "unresolved"]
        if sel or allc:
            for c in sel:
                print("  %.3f sub %s/%s ch %s %s->%s %s %s dec %s res %s falls %s gap %s jf %s jg %s alt %s %s" % (
                    x["T_over_rs"], c["sub"], c["nsub"], c["channel"], c["i"], c["n"], c["kind"], c["decision"], c["decided"], c["resolved"],
                    c["falls"], f(c["gap_min"], "%.2e"), f(c["jump_fit_over_E0"]), f(c["jump_grid_over_E0"]), f(c["jump_alt_over_E0"]), c["class"]))
    print("\n## the rule dependence in the epoch: anchor (maximum overlap) against prim (R2' _r2b to 13.5, v36v at 13.625)")
    print("T/rs | anchor E/E0 | conv | pk | prim E/E0 | src | prim - anchor | prim printable | lvl | conv | pk")
    for r in J["rule_dependence"]["rows"]:
        print("%.3f | %.9f | %s | %d | %s | %s | %s | %s | %s | %s | %s" % (
            r["T_over_rs"], r["anchor_E_over_E0"], r["anchor_converged"], r["anchor_pockets"], f(r.get("prim_E_over_E0"), "%.9f"),
            r.get("prim_source", "-"), f(r.get("prim_minus_anchor")), r.get("prim_printable", "-"), r.get("prim_level", "-"),
            r.get("prim_converged", "-"), r.get("prim_pockets", "-")))
    s = J["rule_dependence"]["at_shared"]
    if s:
        print("  rule dependence at %.3f: prim %.7f - anchor %.7f = %+.2e of E0 (prim record printable %s, level %s; the anchor solve converged %s)" % (
            s["T_over_rs"], s["prim_E_over_E0"], s["anchor_E_over_E0"], s["prim_minus_anchor"], s["prim_printable"], s["prim_level"], s["anchor_converged"]))
    print("\n## the maximum-overlap comparator (v36l), labelled as that convention's")
    v = J["v36l_comparator"]
    print("  %s; %d records to %.3f" % (v["convention"], v["records"], v["T_last"]))
    print("  formation (first record with a trapping zero): %.3f (R_t/r_s %.4f, E/E0 %.8f)" % (v["formation_clock"], v["formation_R_t_over_rs"], v["formation_E_over_E0"]))
    print("  first episode %.3f .. %.3f; its close (first record without a zero) %s (min(N - |v|) there %s)" % (
        v["first_episode"][0], v["first_episode"][1], v["first_episode_close"], f(v["close_min_N_minus_absv"], "%+.4f")))
    print("  intervals with zeros: %s" % ["%.3f-%.3f" % (i[0], i[1]) for i in v["intervals_with_zeros"]])
    print("  min(N - |v|): %s" % {k: f(x, "%+.4f") for k, x in v["min_N_minus_absv_at"].items()})
    c = J["composed_trap"]
    print("  for comparison, %s: formation %s, first episode %s, intervals %s" % (
        c["convention"], c["formation_clock"], c["first_episode"], ["%.3f-%.3f" % (i[0], i[1]) for i in c["intervals_with_zeros"]]))

    pv = J.get("prim_vs_v36l")
    if pv:
        print("\n## the prim continuation against the v36l comparator from the shared 14.5 state (E/E0 only)")
        print("  third consecutive clock within %.0e: %s; first clock off the floor: %s; max |prim - v36l| %s; last %s" % (
            pv["floor"], pv["third_consecutive_within_floor"], pv["first_clock_off_floor"], f(pv["max_abs"], "%.3e"),
            pv["last"] and "%.3f %+.3e" % (pv["last"]["T_over_rs"], pv["last"]["prim_minus_v36l"])))
        prev = 0.0
        print("  (clocks where the difference moves by more than the floor, or either run has pockets)")
        for x in pv["rows"]:
            moved = abs(x["prim_minus_v36l"] - prev) > pv["floor"]
            prev = x["prim_minus_v36l"]
            if moved or x["prim_pockets"] or x["v36l_pockets"]:
                print("  %.3f prim - v36l %+.3e  pockets prim %d v36l %d  v36l switches %s" % (
                    x["T_over_rs"], x["prim_minus_v36l"], x["prim_pockets"], x["v36l_pockets"], x["v36l_switches"]))


if __name__ == "__main__":
    main()
