#!/usr/bin/env python
"""compose_relabel_v1_15.py (builder analyst; inbox #19 and #20: the icloud lane's checks of TEMPERATURE_solved_exterior_v1.2,
review_TEMPERATURE_solved_exterior_v1_2.tex, and of COLLAPSE_certificate_statement_v1.2, review_COLLAPSE_certificate_statement_v1_2.tex;
inbox #18, the check of COLLAPSE_certificate_v1.2, requires no change): version 1.14 with the title's source list and the sources line
extended and rows added before the sources line: per text an umbrella row and one row per clause the checks correct, withdrawn or relabelled
until the lead's versions 1.3 (turn 43). Every number is computed by this script from the records (nothing taken from the reviews): the
certificate v3 tables' share, median, agreement and mass-defect columns (inclusive and strict share counts per table, the interval counts,
the late mass ranges, the all-node count, the frozen records' rows), the homogeneous validation's largest relative difference, the J3'
records' log-slope lines and hold-16 readings. Args: --v114 --j3dir --certdir --out."""
import argparse
import os
import re

TABLES = [("the production", "certificate_v3_c19_rs64.md"), ("the sigma/a 8 member", "certificate_v3_c27_rs64_A_R2_tau8_ell8_prep4_avg.md"),
          ("the sigma/a 4 member", "certificate_v3_c27_rs64_A_R2_tau8_ell4_prep4_avg.md"), ("r_s 32 A tau 16", "certificate_v3_set17_rs32_A_tau16.md"),
          ("r_s 32 A tau 8", "certificate_v3_set17_rs32_A_tau8.md"), ("r_s 32 A tau 4", "certificate_v3_set17_rs32_A_tau4.md"), ("r_s 32 B tau 8", "certificate_v3_set17_rs32_B_tau8.md")]


def cells(line):
    return [c.strip() for c in line.strip().strip("|").split("|")]


def cert_rows(md):
    L = open(md).read().split("\n")
    hi = [i for i, l in enumerate(L) if l.startswith("| T/rs | d_t | tt direct |")]
    assert len(hi) == 1, md
    head = cells(L[hi[0]])
    out = []
    for l in L[hi[0] + 2:]:
        if not l.startswith("|"):
            break
        c = dict(zip(head, cells(l)))
        r = {"T": float(c["T/rs"]), "tts": float(c["tt share within 0.1"]), "rrs": float(c["rr share within 0.1"]),
             "ttm": float(c["tt median"]), "rrm": float(c["rr median"]),
             "tta": float(c["tt agreement max (r)"].split()[0]), "rra": float(c["rr agreement max (r)"].split()[0])}
        q, d = [float(x) for x in c["mass defect (quad, direct)"].split(",")]
        r["mq"], r["md"] = q, d
        out.append(r)
    return out


def tab_c(md):
    L = open(md).read().split("\n")
    si = [i for i, l in enumerate(L) if l.startswith("### (c) T against")][0]
    rows, summ, key = [], [], None
    for l in L[si + 1:]:
        if l.startswith("### "):
            break
        if not l.startswith("|") or l.startswith("|---") or l.startswith("| omega r_s"):
            continue
        c = cells(l)
        if c[0]:
            key = (c[0], c[1]); rows.append(c)
        else:
            m = re.search(r"log slope of n_w per r_s ([-+0-9.e]+)", l)
            if m:
                summ.append((key, m.group(1)))
    return rows, summ


def frozen_stage(md, w, mw, hold):
    L = open(md).read().split("\n")
    hi = [i for i, l in enumerate(L) if l.startswith("| omega r_s | m/omega | hold/r_s | frozen stage | T/T_H |")][0]
    for l in L[hi + 2:]:
        if not l.startswith("|"):
            break
        c = cells(l)
        if c[0] == w and c[1] == mw and c[2] == hold:
            return c[3]
    return None


def main():
    ap = argparse.ArgumentParser()
    for k in ("v114", "j3dir", "certdir", "out"):
        ap.add_argument("--" + k, required=True)
    a = ap.parse_args()
    # --- the certificate statement (inbox #20)
    T = {nm: cert_rows(os.path.join(a.certdir, f)) for nm, f in TABLES}
    cnt = []
    for nm, _ in TABLES:
        R = T[nm]
        inc = sum(1 for r in R if r["tts"] >= 0.8 and r["rrs"] >= 0.8)
        st = sum(1 for r in R if r["tts"] > 0.8 and r["rrs"] > 0.8)
        cnt.append("%s %d (strict %d) of %d" % (nm, inc, st, len(R)))
    print("counts:", "; ".join(cnt))
    P = T["the production"]
    passes = lambda r: r["tts"] >= 0.8 and r["rrs"] >= 0.8  # noqa: E731
    i1 = [r for r in P if 0.0 <= r["T"] <= 6.125 + 1e-9]
    i2 = [r for r in P if 6.25 - 1e-9 <= r["T"] <= 10.875 + 1e-9]
    p1 = [r for r in i1 if passes(r)]
    lo = min(i1, key=lambda r: min(r["tts"], r["rrs"]))
    med1 = [x for r in i1 for x in (r["ttm"], r["rrm"])]
    print("interval 0-6.125:", len(i1), len(p1), lo["T"], min(lo["tts"], lo["rrs"]), min(med1), max(med1))
    p2 = [r for r in i2 if passes(r)]
    print("interval 6.25-10.875:", len(i2), len(p2))
    late = [r for r in P if 20.875 - 1e-9 <= r["T"] <= 39.875 + 1e-9]
    last = P[-1]
    dmin, dmax = min(r["md"] for r in late), max(r["md"] for r in late)
    qmin, qmax = min(r["mq"] for r in late), max(r["mq"] for r in late)
    print("late mass:", dmin, dmax, qmin, qmax, "last", last["T"], last["md"], last["mq"])
    alln = [r for r in P if r["tta"] < 0.1 and r["rra"] < 0.1]
    alln13 = [r for r in alln if r["T"] >= 13.0 - 1e-9]
    fr = {t: next(r for r in P if abs(r["T"] - t) < 1e-9) for t in (35.375, 37.75)}
    print("all-node:", len(alln), len(alln13), {t: (fr[t]["tta"], fr[t]["rra"], fr[t]["tts"], fr[t]["rrs"]) for t in fr})
    V = open(os.path.join(a.certdir, "certificate_v3_validate.md")).read()
    vals = [float(x) for x in re.findall(r"over the nodes: tt ([0-9.e+-]+), rr", V)] + [float(x) for x in re.findall(r", rr ([0-9.e+-]+)\.\s*$", V, re.M)]
    vmax = max(vals)
    print("validation max", vmax)
    # --- the temperature (inbox #19)
    J = lambda n: os.path.join(a.j3dir, n)
    PMAX, PMIN, SW = J("J3P_raymap_v9_prod19_max.md"), J("J3P_raymap_v8_prod19_min.md"), J("J3P_raymap_v9_tau4_swingmax.md")
    def h16(md, mw):
        rows, _ = tab_c(md)
        r = [c for c in rows if c[0] == "1" and c[1] == mw and c[2] == "16"][0]
        return r[12].split(";")[0].strip()
    x = {(ph, mw): h16(md, mw) for ph, md in (("max", PMAX), ("min", PMIN)) for mw in ("0", "0.25")}
    fs = {(ph, mw): frozen_stage(md, "1", mw, "16") for ph, md in (("max", PMAX), ("min", PMIN)) for mw in ("0", "0.25")}
    lw = {(ph, w, mw): frozen_stage(md, w, mw, "16") for ph, md in (("max", PMAX), ("min", PMIN)) for w in ("0.25", "0.5") for mw in ("0", "0.25")}
    print("h16", x, fs, lw)
    _, sm = tab_c(PMAX)
    s_pmax = ["m/omega %s %s" % (k[1], v) for k, v in sm if k[0] == "0.25"]
    _, ss = tab_c(SW)
    s_sw = ["omega r_s %s m/omega %s %s" % (k[0], k[1], v) for k, v in ss if k[0] in ("0.25", "0.5")]
    print("slopes", s_pmax, s_sw)

    R19 = "review_TEMPERATURE_solved_exterior_v1_2.tex"
    R20 = "review_COLLAPSE_certificate_statement_v1_2.tex"
    lwtxt = "; ".join("%s phase omega r_s %s m/omega %s: %s" % (ph, w, mw, lw[(ph, w, mw)]) for ph in ("max", "min") for w in ("0.25", "0.5") for mw in ("0", "0.25"))
    rows = [
        "| TEMPERATURE_solved_exterior_v1.2.md (the lead's sentence of turn 42) | under the icloud lane's delta check (%s, inbox #19): \"Two small literal precisions remain, without changing the order-one finite-probe conclusion\"; the two clauses below relabelled until the lead's version 1.3 (turn 43) |" % R19,
        "| its \"their hold-16 fits, 1.87 at the maximum and 1.37 at the minimum, are the one packet read at both phases\" and its condition's \"Not printed: ... the hold-16 values as frozen-stage readings\" | reads: omega r_s 1 at hold 16 is the common frozen-stage frequency and hold for both probe masses (frozen stage: %s at the maximum, %s at the minimum for m/omega 0; %s and %s for m/omega 0.25), reading T/T_H %s and %s (m/omega 0) and %s and %s (m/omega 0.25) at the maximum and minimum; the long-wavelength hold-16 fits also exist at both phases with the frozen-stage test failed (%s), so the exclusion reads \"the longwave hold-16 values as frozen-stage readings\" (%s item 1) |" % (
            fs[("max", "0")], fs[("min", "0")], fs[("max", "0.25")], fs[("min", "0.25")], x[("max", "0")], x[("min", "0")], x[("max", "0.25")], x[("min", "0.25")], lwtxt, R19),
        "| its \"the occupation falling at 6.6 percent per r_s\" and \"the occupations falling at 7.7 to 9.8 percent per r_s\" | reads as the stored log-linear fit of log n_w across holds 16, 32, 48 and 64 (the producer's log slope of n_w per r_s over the entire stored series), not a local decay rate or a fit over holds 32 to 64: the production maximum's omega r_s 0.25 packets %s; the tau 4 swing maximum's %s (J3P_raymap_v9_prod19_max and v9_tau4_swingmax, table (c); %s item 2) |" % (", ".join(s_pmax), ", ".join(s_sw), R19),
        "| COLLAPSE_certificate_statement_v1.2.md (the lead's text of turn 42) | under the icloud lane's bounded check (%s, inbox #20): \"Both normalization repairs remain closed\" and the aggregate counts hold under the inclusive, rowwise threshold; the clauses below relabelled until the lead's version 1.3 (turn 43); the check of COLLAPSE_certificate_v1.2 itself (review_COLLAPSE_certificate_v1_2.tex, inbox #18) requires no change |" % R20,
        "| its \"agree within 1/10 at more than 0.8 of the support nodes in both rows\" | reads \"at least 0.8 of the retained support nodes, separately in each row\" (34.3's inclusive rule, at the printed three-decimal shares): %s; the row fractions do not establish an 80 percent common intersection of passing nodes (%s item 1) |" % ("; ".join(cnt), R20),
        "| its \"on every record from T/r_s 0 to 6.125 (shares 0.925 to 1.000, medians 6e-5 to 1.6e-2 of the residual), on 27 of the 39 records from 6.25 to 10.875\" | reads: %d of the %d records from 0 to 6.125 pass, the smallest row share %.3f at T/rs %.3f, the medians %.2e to %.2e; %d of the %d records from 6.25 to 10.875 pass (the production table of COLLAPSE_certificate_v1.2; %s item 2) |" % (
            len(p1), len(i1), min(lo["tts"], lo["rrs"]), lo["T"], min(med1), max(med1), len(p2), len(i2), R20),
        "| its \"The direct route's mass-balance defect over its largest term is 7e-5 to 2.5e-3 from 20.875 on ... the quadrature's 4e-4 to 0.30 late\" and \"its homogeneous validation agreeing to 2.3e-06 or better\" | reads with the window 20.875 <= T/r_s <= 39.875: the direct defect %.2e to %.2e and the quadrature's %.2e to %.2e there; the last record %.3f has %.2e (direct) and %.3g (quadrature), outside those ranges; \"its largest term\" is the denominator max(|M-dot|, |L_G|), the row term not included; the homogeneous validation's largest relative difference is %.2e (certificate_v3_validate.md) (%s item 3) |" % (
            dmin, dmax, qmin, qmax, last["T"], last["md"], last["mq"], vmax, R20),
        "| its \"On 75 records from 13.0 on the largest disagreement over the whole support is below 0.1 in both rows, so every node, the innermost included, agrees there\" | reads \"every retained support node\" (the summaries retain the nodes ell <= r <= r_collar - 2 ell with |rho| above 1e-6 of its largest value; whether r = ell lies in that support on every such record is not shown): %d records with both agreement maxima below 0.1 (%d from T/rs 13.0 on); the frozen records 35.375 and 37.75 pass the rowwise criterion (shares %.3f and %.3f; %.3f and %.3f) with agreement maxima %.3g and %.3g (tt) and %.3g and %.3g (rr), so they are not all-node records (%s item 4) |" % (
            len(alln), len(alln13), fr[35.375]["tts"], fr[35.375]["rrs"], fr[37.75]["tts"], fr[37.75]["rrs"], fr[35.375]["tta"], fr[37.75]["tta"], fr[35.375]["rra"], fr[37.75]["rra"], R20),
    ]
    t = open(a.v114).read()
    tm = re.search(r"^# RELABEL_table_v1\.14: (.*?)\(engine spec (.*?); its delta check of version 1\.1, inbox #16; 34\.2, 34\.3\)$", t, re.M)
    assert tm
    t = t.replace(tm.group(0), "# RELABEL_table_v1.15: %s(engine spec %s; its delta check of version 1.1, inbox #16; 34.2, 34.3; the icloud lane's checks of the temperature v1.2 and the certificate statement v1.2, inbox #19, #20)" % (tm.group(1), tm.group(2)), 1)
    pm = re.search(r"Version 1\.14 = version 1\.13 with ", t)
    assert pm
    t = t.replace(pm.group(0), "Version 1.15 = version 1.14 with %d rows added (inbox #19 and #20: the icloud lane's checks of TEMPERATURE_solved_exterior_v1.2 and COLLAPSE_certificate_statement_v1.2, their clauses relabelled until the lead's versions 1.3; the records' numbers printed by compose_relabel_v1_15; inbox #18 requires no change). Version 1.14 = version 1.13 with " % len(rows), 1)
    si = t.index("\n- Sources: ")
    t = t[:si].rstrip("\n") + "\n" + "\n".join(rows) + "\n" + t[si:]
    smm = re.search(r"\n- Sources: ([^\n]*)", t)
    assert smm
    t = t.replace(smm.group(0), "\n- Sources: " + smm.group(1).rstrip(".") + "; deliveries/solve_b/review_TEMPERATURE_solved_exterior_v1_2.tex, review_COLLAPSE_certificate_statement_v1_2.tex, review_COLLAPSE_certificate_v1_2.tex; deliveries/solve_a/TEMPERATURE_solved_exterior_v1.2.md, COLLAPSE_certificate_statement_v1.2.md, COLLAPSE_certificate_v1.2.md (its seven tables, solve/results/engine/CERT3/).", 1)
    open(a.out, "w").write(t)
    print("wrote %s (%d rows added)" % (a.out, len(rows)))


if __name__ == "__main__":
    main()
