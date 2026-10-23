#!/usr/bin/env python
"""engine_star_family_summary_v1.py (builder analyst; task 1 of the goal, the static star family: "the family against central density;
the mass against radius, the maximum mass (its scaling m_P^3/m^2 as G/a^2 and m a vary), the maximum compactness, the minimum of N - |v|
over the family (no static member reaches zero)"): reads engine_star_v6 family records (json) and prints, per family (m a, g = G/a^2), its
members ordered by the central lapse (N_B, E, R_surf, the compactness 2 g E/R_surf, N_centre, min(N - |v|), E_F/m, passes, flags), the
largest-E member and whether E still rises at the family's densest member (the maximum mass reached or not), the ell control (E at ell 4, 8,
16 on the controlled members and the fitted ell -> 0 value), the continuum Oppenheimer-Volkoff maximum N_B the family was seeded from, and
across the families the invariant E_max g^(3/2) (m a)^2 (constant if the maximum mass scales as m_P^3/m^2), the largest compactness and the
smallest min(N - |v|). Every number is read from the records. Args: --families (json paths) --md."""
import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", nargs="+", required=True)
    ap.add_argument("--md", required=True)
    a = ap.parse_args()
    L = ["## The static star families (engine_star_family_summary_v1.py over %s)" % ", ".join(os.path.basename(f) for f in a.families), ""]
    cross = []
    for f in a.families:
        d = json.load(open(f))
        m = float(d["args"]["m"]); g = float(d["args"]["g"])
        mem = [q for q in d["members"] if not q.get("failed")]
        mem.sort(key=lambda q: -q["N_centre"])
        s = d.get("summary", {})
        L += ["### m a = %g, g = G/a^2 = %g (%s; ell %s, %d sites; %d of %d members converged; the continuum OV maximum N_B %.1f)" % (
            m, g, os.path.basename(f), d["args"].get("ell"), d["args"].get("nsites"), s.get("n_ok", len(mem)), s.get("n", len(d["members"])), s.get("NB_max_continuum", float("nan"))), "",
              "| N_B | E | R_surf | 2gE/R_surf | N_centre | min(N - abs v) | E_F/m | passes | flags |", "|---|---|---|---|---|---|---|---|---|"]
        fx = lambda x, f_: (f_ % x) if x is not None else "-"
        for q in mem:
            L.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (fx(q["N_B"], "%.2f"), fx(q["E"], "%.6f"), fx(q.get("R_surf"), "%.2f"), fx(q.get("compactness_surf"), "%.4f"),
                                                                 fx(q["N_centre"], "%.5f"), fx(q.get("min_N_minus_absv"), "%.5f"), fx(q["E_F"] / m if q.get("E_F") is not None else None, "%.5f"),
                                                                 q.get("passes"), ",".join(q.get("flags") or []) or "-"))
        iE = max(range(len(mem)), key=lambda i: mem[i]["E"])
        top = mem[iE]
        rising = (iE == len(mem) - 1)
        L += ["", "Largest E: %.6f at N_B %.2f (N_centre %.5f); %s." % (top["E"], top["N_B"], top["N_centre"],
              "E still rises at the densest member: the maximum mass is not bracketed" if rising else "denser members have smaller E: the maximum mass is bracketed")]
        for c in d.get("ell_control") or []:
            rows = c.get("rows", {})
            fit = c.get("fit_E_a_plus_b_ell2")
            L.append("ell control at N_B %.0f: E %s; the spread %.3e; the fit E(ell) = E_0 + b ell^2 gives E_0 %s." % (
                c.get("NB_target", float("nan")), ", ".join("%.6f (ell %s)" % (rows[k]["E"], k) for k in sorted(rows, key=float) if not rows[k].get("failed")),
                c.get("spread_E", float("nan")), ("%.6f" % fit[1]) if fit else "-"))
        L.append("")
        cross.append({"m": m, "g": g, "E_max": top["E"], "N_B_at_E_max": top["N_B"], "bracketed": not rising, "C_max": max(q["compactness_surf"] for q in mem if q.get("compactness_surf") is not None),
                      "min_N_minus_absv": min(q["min_N_minus_absv"] for q in mem if q.get("min_N_minus_absv") is not None), "inv": top["E"] * g ** 1.5 * m * m, "n": len(mem)})
    L += ["### Across the families", "", "| m a | g | members | largest E | its N_B | maximum bracketed | E_max g^1.5 (m a)^2 | largest 2gE/R_surf | smallest min(N - abs v) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for c in sorted(cross, key=lambda c: (c["g"], c["m"])):
        L.append("| %g | %g | %d | %.6f | %.2f | %s | %.5f | %.4f | %.5f |" % (c["m"], c["g"], c["n"], c["E_max"], c["N_B_at_E_max"], "yes" if c["bracketed"] else "no",
                                                                     c["inv"], c["C_max"], c["min_N_minus_absv"]))
    L += ["", "- every number above is read from the family records by the script; the compactness is the record's 2 g E/R_surf at the surface radius."]
    open(a.md, "w").write("\n".join(L) + "\n")
    print("\n".join(L[-len(cross) - 4:]))


if __name__ == "__main__":
    main()
