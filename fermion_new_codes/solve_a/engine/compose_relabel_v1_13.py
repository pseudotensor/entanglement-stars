#!/usr/bin/env python
"""compose_relabel_v1_13.py (builder analyst; inbox #16, the icloud lane's delta check of TEMPERATURE_solved_exterior_v1.1,
deliveries/solve_b/review_TEMPERATURE_solved_exterior_v1_1.tex: "Apply anything it still requires"): version 1.12 with the title's
source list and the sources line extended and rows added before the sources line, one umbrella row and one per clause of
TEMPERATURE_solved_exterior_v1.1's quoted sentence that the check corrects (its items 1 to 4), withdrawn or relabelled until the lead's
version 1.2 (turn 42). Every number is read from the J3' records by this script (nothing retyped): the floor ratios n_0/n_w of the
withheld fits (table (c)), the frozen-stage labels and within counts (the budget table), the late-phase table's kappa_T, the quasi-frozen
check's relative change. Args: --v112 --j3dir --out."""
import argparse
import os
import re


def cells(line):
    return [c.strip() for c in line.strip().strip("|").split("|")]


def table_after(lines, start_prefix, header_prefix):
    """Rows (dicts) of the first table whose header starts with header_prefix after the line starting with start_prefix."""
    si = [i for i, l in enumerate(lines) if l.startswith(start_prefix)]
    assert si, start_prefix
    for i in range(si[0], len(lines)):
        if lines[i].startswith(header_prefix):
            head = cells(lines[i])
            rows = []
            for l in lines[i + 2:]:
                if not l.startswith("|"):
                    break
                c = cells(l)
                if c[0] == "":
                    continue
                rows.append(dict(zip(head, c)))
            return rows
    raise AssertionError(header_prefix)


def tab_c(md):
    L = open(md).read().split("\n")
    return table_after(L, "### (c) T against", "| omega r_s | m/omega | hold/r_s |")


def budget(md):
    L = open(md).read().split("\n")
    return table_after(L, "### The bias budget", "| omega r_s | m/omega | hold/r_s | frozen stage | T/T_H |")


def floor_of(cell):
    m = re.search(r"withheld \(floor ([0-9.]+) of n_w\)", cell)
    return m.group(1) if m else None


def pick(rows, w, mw, hold):
    r = [x for x in rows if x["omega r_s"] == w and x["m/omega"] == mw and x["hold/r_s"] == hold]
    assert len(r) == 1, (w, mw, hold, len(r))
    return r[0]


def is_num(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def main():
    ap = argparse.ArgumentParser()
    for k in ("v112", "j3dir", "out"):
        ap.add_argument("--" + k, required=True)
    a = ap.parse_args()
    J = lambda n: os.path.join(a.j3dir, n)
    PMAX, PMIN, SW, T39_75, T39_5 = (J("J3P_raymap_v9_prod19_max.md"), J("J3P_raymap_v8_prod19_min.md"), J("J3P_raymap_v9_tau4_swingmax.md"),
                                     J("J3P_raymap_v9_tau4_max.md"), J("J3P_raymap_v8_tau4_min.md"))
    TH = "T/T_H (A; B)"
    # item 1: the floor ratios
    def fl(md, w, hold):
        c = tab_c(md)
        return ", ".join("m/omega %s %s" % (mw, floor_of(pick(c, w, mw, hold)[TH]) or "read (%s)" % pick(c, w, mw, hold)[TH]) for mw in ("0", "0.25"))
    f_pmax32 = fl(PMAX, "0.25", "32"); f_pmin32 = fl(PMIN, "0.5", "32")
    f_sw32 = "omega r_s 0.25: %s; omega r_s 0.5: %s" % (fl(SW, "0.25", "32"), fl(SW, "0.5", "32"))
    f_sw64 = "omega r_s 0.25: %s; omega r_s 0.5: %s" % (fl(SW, "0.25", "64"), fl(SW, "0.5", "64"))
    print("floors", f_pmax32, "|", f_pmin32, "|", f_sw32, "|", f_sw64)
    # item 2: omega r_s 1, m/omega 0, hold 16 at the two production phases
    def h16(md):
        c = pick(tab_c(md), "1", "0", "16")[TH].split(";")[0].strip()
        st = pick(budget(md), "1", "0", "16")["frozen stage"]
        return c, st
    x1, s1 = h16(PMAX); x2, s2 = h16(PMIN)
    print("h16", x1, s1, x2, s2)
    # item 3: the late-phase table's kappa_T
    LP = open(J("obs_late_phase_v1.md")).read().split("\n")
    prod = {r["T/rs"]: r for r in table_after(LP, "### prod19", "| T/rs | R_t | kappa_T |")}
    tau4 = {r["T/rs"]: r for r in table_after(LP, "### tau4", "| T/rs | R_t | kappa_T |")}
    kx, kn = float(tau4["36.2500"]["kappa_T"]), float(tau4["39.5000"]["kappa_T"])
    below, above = (kx - kn) / kx, (kx - kn) / kn
    k3475, k35125, k35375 = prod["34.7500"]["kappa_T"], prod["35.1250"]["kappa_T"], prod["35.3750"]["kappa_T"]
    ends = abs(float(k3475) - float(k35375)) / float(k3475)
    print("kappa", kx, kn, below, above, k3475, k35125, k35375, ends)
    # the quasi-frozen check's relative change (the 36.25 run's version 7 md)
    SW7 = open(J("J3P_raymap_v7_tau4_swingmax.md")).read().split("\n")
    qf = table_after(SW7, "### The quasi-frozen check", "| omega r_s | m/omega | hold/r_s | background frozen at T/rs |")
    rc = ["omega r_s %s m/omega %s: %s" % (r["omega r_s"], r["m/omega"], r["relative change"]) for r in qf if r["omega r_s"] in ("0.25", "0.5")]
    nb = sorted(set(r["background frozen at T/rs"] for r in qf))
    print("rc", rc, nb)
    # item 4: the two trough runs kept separate
    def within(md):
        b = [r for r in budget(md) if is_num(r["T/T_H"])]
        return sum(1 for r in b if r["within"] == "yes"), len(b)
    w75, n75 = within(T39_75); w5, n5 = within(T39_5)
    def rng(md, mw):
        b = [float(r["T/T_H"]) for r in budget(md) if r["omega r_s"] == "0.5" and r["m/omega"] == mw and r["hold/r_s"] in ("48", "64") and is_num(r["T/T_H"])]
        return (min(b), max(b)) if b else None
    r75_0, r75_25, r5_0, r5_25 = rng(T39_75, "0"), rng(T39_75, "0.25"), rng(T39_5, "0"), rng(T39_5, "0.25")
    print("trough", w75, n75, w5, n5, r75_0, r75_25, r5_0, r5_25)
    fr = lambda r: ("%.5f to %.5f" % r) if r else "none"

    R = "review_TEMPERATURE_solved_exterior_v1_1.tex"
    rows = [
        "| TEMPERATURE_solved_exterior_v1.1.md (the lead's sentence of turn 41) | under the icloud lane's delta check (%s, inbox #16), which leaves the supported conclusion unchanged (\"the selected finite-probe readings are order one relative to T_H, with no common thermal temperature established on the continuing history\"); the clauses in the rows below are withdrawn or relabelled until the lead's version 1.2 (turn 42) |" % R,
        "| its \"the 0.25 packet decays to the no-formation floor by a hold of 32 r_s\", \"the 0.5 packet decays to the floor\" and \"both decay to the floor by a hold of 32 r_s\" | reads: the fit is withheld by the declared floor-contamination criterion n_0/n_w > 0.03 (floor_max 0.03), not because n_w reaches n_0: at hold 32 the production maximum's omega r_s 0.25 packets have n_0/n_w %s; the production minimum's omega r_s 0.5 packets %s; the tau 4 swing maximum's %s; the latter at hold 64 %s (J3P_raymap_v9_prod19_max, v8_prod19_min, v9_tau4_swingmax, table (c); %s item 1) |" % (f_pmax32, f_pmin32, f_sw32, f_sw64, R),
        "| its \"no packet reads at both phases, so no same-packet temperature spread across phases exists\" | reads restricted to the selected long-wavelength frozen-stage readings for which the fallback is used: the omega r_s 1, m/omega 0, hold-16 packet reads T/T_H %s at the maximum (frozen stage %s) and %s at the minimum (frozen stage %s), its table using a temperature-level oscillation term (%s item 2) |" % (x1, s1, x2, s2, R),
        "| its \"the slope 37 percent above its trough at 39.5\" | reads \"its trough at 39.5 %.0f percent below the maximum\": |kappa_T,max - kappa_T,min|/kappa_T,max = %.4f (kappa_T %s at 36.25, %s at 39.5; obs_late_phase_v1's tau 4 table); the maximum lies %.0f percent above the trough (%s item 3) |" % (100 * below, below, tau4["36.2500"]["kappa_T"], tau4["39.5000"]["kappa_T"], 100 * above, R),
        "| its \"frozen one r_s earlier the same occupations are 49 to 178 times larger\" | reads with the ratio named: the relative change |Delta n_w|/n_w against the history frozen at T/rs %s (%s; J3P_raymap_v7_tau4_swingmax's quasi-frozen check at hold 32) (%s item 3) |" % (", ".join(nb), "; ".join(rc), R),
        "| its \"on a broad maximum 34.75 to 35.375, the slope within 0.2 percent\" | reads: the 0.2 percent compares the two selected maximum endpoints (kappa_T %s at 34.75 and %s at 35.375, |difference|/kappa_T(34.75) = %.4f), not a uniform bound on 34.75 to 35.375: the intervening record 35.125 has kappa_T %s (obs_late_phase_v1's production table; %s item 3) |" % (k3475, k35375, ends, k35125, R),
        "| its \"at its trough (39.5 and 39.75, 0.4 percent apart) it reads 1.22 to 1.30 at omega r_s 0.5, falling over the holds without a plateau, 3 of 9 readings inside their budgets\" | reads with the two records kept separate: the 39.75 run (J3P_raymap_v9_tau4_max) %d of %d readings inside its budgets, omega r_s 0.5 at holds 48 and 64 reading %s (m/omega 0) and %s (m/omega 0.25); the 39.5 run (J3P_raymap_v8_tau4_min, a different step partner and fallback) %d of %d, reading %s (m/omega 0) and %s (m/omega 0.25); \"These are diagnostic counts, not coverage probabilities\" (%s item 4) |" % (w75, n75, fr(r75_0), fr(r75_25), w5, n5, fr(r5_0), fr(r5_25), R),
    ]
    t = open(a.v112).read()
    tm = re.search(r"^# RELABEL_table_v1\.12: (.*?)\(engine spec (.*?); the icloud lane's review of the solved-exterior temperature, inbox #14; 33\.2, 33\.5\)$", t, re.M)
    assert tm
    t = t.replace(tm.group(0), "# RELABEL_table_v1.13: %s(engine spec %s; the icloud lane's review of the solved-exterior temperature, inbox #14; 33.2, 33.5; its delta check of version 1.1, inbox #16)" % (tm.group(1), tm.group(2)), 1)
    pm = re.search(r"Version 1\.12 = version 1\.11 with ", t)
    assert pm
    t = t.replace(pm.group(0), "Version 1.13 = version 1.12 with %d rows added (inbox #16: the icloud lane's delta check of TEMPERATURE_solved_exterior_v1.1, its four items, the clauses withdrawn or relabelled until the lead's version 1.2; the records' numbers printed by compose_relabel_v1_13). Version 1.12 = version 1.11 with " % len(rows), 1)
    si = t.index("\n- Sources: ")
    t = t[:si].rstrip("\n") + "\n" + "\n".join(rows) + "\n" + t[si:]
    sm = re.search(r"\n- Sources: ([^\n]*)", t)
    assert sm
    t = t.replace(sm.group(0), "\n- Sources: " + sm.group(1).rstrip(".") + "; deliveries/solve_b/review_TEMPERATURE_solved_exterior_v1_1.tex; deliveries/solve_a/TEMPERATURE_solved_exterior_v1.1.md; solve/results/obs/J3v7/J3P_raymap_v9_prod19_max, _v8_prod19_min, _v9_tau4_swingmax, _v7_tau4_swingmax, _v9_tau4_max, _v8_tau4_min (.md) and obs_late_phase_v1.md.", 1)
    open(a.out, "w").write(t)
    print("wrote %s (%d rows added)" % (a.out, len(rows)))


if __name__ == "__main__":
    main()
