#!/usr/bin/env python
"""compose_relabel_v1_11.py (builder analyst; inbox #14, the icloud lane's review of TEMPERATURE_solved_exterior_v1,
deliveries/solve_b/review_TEMPERATURE_solved_exterior_v1.tex: "apply its required changes (solve: to the result and its labels)"):
version 1.10 with the title's source list and the sources line extended and rows added before the sources line, one per clause of
TEMPERATURE_solved_exterior_v1.md that the review corrects. The rows withdraw or relabel those clauses until the lead's version 1.1
(turn 41); they quote the review and print the J3' version 8 records' own numbers, read here from the four mds and two jsons
(nothing retyped). Args: --v110 --j3dir --out."""
import argparse
import json
import os
import re


def cells(line):
    return [c.strip() for c in line.strip().strip("|").split("|")]


def budget_rows(md):
    """The budget table: header '| omega r_s | m/omega | hold/r_s | frozen stage | T/T_H | ...| within |'."""
    lines = open(md).read().splitlines()
    hi = [i for i, l in enumerate(lines) if l.startswith("| omega r_s | m/omega | hold/r_s | frozen stage | T/T_H |")]
    assert len(hi) == 1, (md, len(hi))
    head = cells(lines[hi[0]])
    out = []
    for l in lines[hi[0] + 2:]:
        if not l.startswith("|"):
            break
        c = cells(l)
        out.append(dict(zip(head, c)))
    return out


def plateau_lines(md):
    """Table (c): per packet (omega r_s, m/omega) the summary line 'frozen-stage holds above the floor N, spread/mean X, plateau P'."""
    lines = open(md).read().splitlines()
    si = [i for i, l in enumerate(lines) if l.startswith("### (c) T against")]
    assert len(si) == 1, md
    out, key = [], None
    for l in lines[si[0] + 1:]:
        if l.startswith("### "):
            break
        if not l.startswith("|") or l.startswith("|---") or l.startswith("| omega r_s"):
            continue
        c = cells(l)
        if c[0] != "":
            key = (c[0], c[1])
        else:
            m = re.search(r"frozen-stage holds above the floor (\d+), spread/mean ([^,]*), plateau (yes|no)", l)
            assert m and key, (md, l)
            out.append((key, m.group(1), m.group(2), m.group(3)))
    return out


def is_num(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v110", required=True); ap.add_argument("--j3dir", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    names = ["prod19_max", "prod19_min", "tau4_max", "tau4_min"]
    md = {n: os.path.join(a.j3dir, "J3P_raymap_v8_%s.md" % n) for n in names}
    bud = {n: budget_rows(md[n]) for n in names}
    pla = {n: plateau_lines(md[n]) for n in names}

    # the within counts per record (the denominator: the budget table's rows with a T/T_H reading)
    counts = []
    for n in names:
        rd = [r for r in bud[n] if is_num(r["T/T_H"])]
        ny = sum(1 for r in rd if r["within"] == "yes")
        nn = sum(1 for r in rd if r["within"] == "no")
        counts.append("%s: %d of %d rows with a reading within, %d not" % ("J3P_raymap_v8_%s.md" % n, ny, len(rd), nn))
        print(counts[-1])

    # the two rows the review quotes (tau 4 maximum, omega r_s 0.5, hold 48, m/omega 0 and 0.25)
    quoted = []
    for mw in ("0", "0.25"):
        r = [x for x in bud["tau4_max"] if x["omega r_s"] == "0.5" and x["m/omega"] == mw and x["hold/r_s"] == "48"]
        assert len(r) == 1
        r = r[0]
        quoted.append("m/omega %s: T/T_H %s, total (linear; quadrature) %s, within %s" % (mw, r["T/T_H"], r["total (linear; quadrature)"], r["within"]))
        print(quoted[-1])

    # the plateau lines at omega r_s 0.5 (both maxima) and 0.25 (the production minimum)
    def pl(n, w):
        out = []
        for (key, nh, sm, p) in pla[n]:
            if key[0] == w:
                out.append("m/omega %s: holds above the floor %s, spread/mean %s, plateau %s" % (key[1], nh, sm, p))
        return "; ".join(out)
    p_max = pl("prod19_max", "0.5"); p_min = pl("prod19_min", "0.25"); p_t4x = pl("tau4_max", "0.5"); p_t4n = pl("tau4_min", "0.5")
    for s in (p_max, p_min, p_t4x, p_t4n):
        print(s)

    # the jsons: m/T_H at the record's charge and the root-deficit proxy 1 - r_Hm/(2GE)
    js = {n: json.load(open(os.path.join(a.j3dir, "J3P_raymap_v8_%s.json" % n))) for n in ("prod19_max", "prod19_min")}
    mth = [js[n]["frozen"]["m_engine_over_T_H"] for n in ("prod19_max", "prod19_min")]
    prox = [1.0 - js[n]["frozen"]["rHm_over_rs"] for n in ("prod19_max", "prod19_min")]
    print("m/T_H %.4f %.4f; proxy %.4f %.4f" % (mth[0], mth[1], prox[0], prox[1]))

    R = "review_TEMPERATURE_solved_exterior_v1.tex"
    rows = [
        "| TEMPERATURE_solved_exterior_v1.md (the lead's sentence of turn 40) | under the icloud lane's review (%s, inbox #14); the clauses in the rows below are withdrawn or relabelled until the lead's version 1.1 (turn 41); the review's own \"Usable paper wording\" is the reviewer's text, read there |" % R,
        "| its \"fixed packets of the fermion propagated back to the early sea read a conversion temperature T/T_H\" | reads as the review's specification of the measured state: \"a light j=1/2, kappa_j=+1 two-component radial endpoint-current probe\" on \"the stored applied fields before the selected record\", frozen afterward, with \"the declared vacuum exterior beyond the engine chain\" appended, its initial covariance \"the smoothed negative spectral occupation\" of the initial probe Hamiltonian, and per receiver separately a transmission-weighted single-packet fit; \"not a fit of one common temperature to a spectrum, the engine's massive physical covariance, or a full three-dimensional angular-channel calculation\" (%s item 1) |" % R,
        "| its \"over holds of 48 and 64 r_s\" and \"at the stationary phases of its late oscillation\" | read with the plateau tests of the records' table (c): production maximum at omega r_s 0.5, %s; production minimum at omega r_s 0.25, %s; tau 4 maximum at omega r_s 0.5, %s; tau 4 minimum at omega r_s 0.5, %s; \"selecting holds 48/64 does not establish stationary emission\"; the phase finder \"selects sign changes of stored kappa_T differences, not stationarity of the fields or state\" (%s item 2) |" % (p_max, p_min, p_t4x, p_t4n, R),
        "| its \"both inside the certified late window 28.6 to 39.9\" | reads \"the producer's late window\": \"'Certified late window' must remain a producer label\"; the certificate inspected \"compares lapse-divided quadrature rows with raw direct rows\" and \"does not certify this temperature transfer\" (%s item 2; the two normalization corrections are in engine_certificate_v3, whose tables are pending) |" % R,
        "| its \"every reading lies within 13 to 30 percent of 1/(8 pi G E) and inside its printed bias budget of 0.38 to 0.84 of T_H\" | withdrawn: the range \"concerns selected production rows, not every quoted tau/a=4 reading\"; at the tau 4 maximum, omega r_s 0.5, hold 48 (J3P_raymap_v8_tau4_max.md) %s; the counts per record: %s; the sums \"are diagnostic budgets, not certified errors\" and the complete flag \"only tests that all terms exist\" (%s item 3) |" % ("; ".join(quoted), "; ".join(counts), R),
        "| its \"and the two phases' spread 24 to 31 percent, the exterior's non-stationarity\" | reads: where the partner occupation is withheld the selected production oscillation terms \"fall back to stored-kappa_T spreads\", so \"the quoted 24--31 percent must not be presented as a measured same-packet temperature spread where the fallback is used\"; the continuation-label term imports a convention spread over T/r_s 11 to 21.5, \"explicitly 'not a bound,'\" into the later reading (%s item 3) |" % R,
        "| its \"the averaging shift (9/4) sigma^2/r_s^2 1.3 percent\" | reads: the shift is AC's \"conditional vacuum comparison with its extension/tail errors\"; \"its approximately 1.3 percent value is not an error bound for the matter-bearing applied profile\"; sigma^2 = ell^2 stands (the review agrees with the pure-Gaussian producer; the Gaussian-plus-cell translation ell^2 + a^2/12 and the compact ell^2/15 + a^2/12 are other kernels) (%s item 4) |" % R,
        "| its \"the dust outside the trapping root carries 0.32 and 0.37 of E at the two phases\" | reads \"the root-deficit proxy 1 - r_Hm/(2GE)\", r_Hm the applied-field root: %.4f at the maximum and %.4f at the minimum (J3P_raymap_v8_prod19_max.json and _min.json, 1 - frozen/rHm_over_rs); not an outside Gauss mass (that needs \"the corresponding geometric fields/enclosed mass\") and not dust (that needs \"its source split\") (%s item 4) |" % (prox[0], prox[1], R),
        "| its \"the collapse's own fermion has m/T_H 160 and its flux at infinity carries e^(-4 pi m r_s)\" | reads \"m/T_H %.4f and %.4f at the two records\" (J3P_raymap_v8_prod19_max.json and _min.json, frozen/m_engine_over_T_H; \"160.85 uses the nominal initial charge\"), and the factor e^(-4 pi m r_s) is \"a conditional thermal threshold suppression, not a measured primary-field flux\" (%s item 4) |" % (mth[0], mth[1], R),
        "| its \"the lattice endpoint operator against its continuum proxy on the identical back-averaged profile differs by at most 2.0 percent of T at the maximum ... and has no reading at the minimum\" | reads as the endpoint comparator, kept separate: a stationary conversion-temperature comparison at a and a/4 \"on the same interpolated, inner/outer-blended profile with specified incoming lead occupations\", \"not an error estimate for the preceding packet fit, an exact continuum limit or the coupled three-dimensional coefficient\"; at the minimum the zero native beta^2 and positive proxy values \"characterize those selected stationary ports; they do not erase the finite-history probe occupation\" (%s item 5) |" % R,
    ]

    t = open(a.v110).read()
    tm = re.search(r"^# RELABEL_table_v1\.10: (.*?)\(engine spec (.*?); the hreduce lane's DM7 review, inbox #11; 32\.6\)$", t, re.M)
    assert tm
    t = t.replace(tm.group(0), "# RELABEL_table_v1.11: %s(engine spec %s; the hreduce lane's DM7 review, inbox #11; 32.6; the icloud lane's review of the solved-exterior temperature, inbox #14)" % (tm.group(1), tm.group(2)), 1)
    pm = re.search(r"Version 1\.10 = version 1\.9 with ", t)
    assert pm
    t = t.replace(pm.group(0), "Version 1.11 = version 1.10 with %d rows added (inbox #14: the icloud lane's review of TEMPERATURE_solved_exterior_v1, its clauses withdrawn or relabelled until the lead's version 1.1, the records' numbers printed by compose_relabel_v1_11). Version 1.10 = version 1.9 with " % len(rows), 1)
    si = t.index("\n- Sources: ")
    t = t[:si].rstrip("\n") + "\n" + "\n".join(rows) + "\n" + t[si:]
    sm = re.search(r"\n- Sources: ([^\n]*)", t)
    assert sm
    t = t.replace(sm.group(0), "\n- Sources: " + sm.group(1).rstrip(".") + "; deliveries/solve_b/review_TEMPERATURE_solved_exterior_v1.tex; deliveries/solve_a/TEMPERATURE_solved_exterior_v1.md; solve/results/obs/J3v7/J3P_raymap_v8_prod19_max, _prod19_min, _tau4_max, _tau4_min (.md, .json).", 1)
    open(a.out, "w").write(t)
    print("wrote %s (%d rows added)" % (a.out, len(rows)))


if __name__ == "__main__":
    main()
