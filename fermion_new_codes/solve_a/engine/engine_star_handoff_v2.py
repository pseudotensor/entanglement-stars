#!/usr/bin/env python
"""engine_star_handoff_v2.py (builder analyst; engine spec 22.3 (1), STAR_family_handoff_v1.1: version 1 with the continuum comparator
recomputed on a thermodynamically consistent equation of state (engine_ovpg_v2: P the table's spline, n = P', e = mu n - P): per family
the consistent continuum maximum (a z_c scan around version 1's maximum, the parabola) beside version 1's, per member both E_cont(N_B) and
both E/E_cont - 1 with the shift E_cont(v2)/E_cont(v1) - 1, the invariant against both continuum values. Version 1's docstring follows.
engine_star_handoff_v1.py: engine spec 20.5, delivery (2): "STAR_family_handoff_v1.md now (the five tables, the ell
controls with E_0 and the spread as the ell error, the invariant, E against J-OVPG's E_cont(N_B) per member for the (a/R_*)^2 reading,
'maximum not bracketed')"): per family (m a, g) the converged members of its ladder and of its extensions (engine_star_v6 records; ordered by
the central lapse), each with N_B, E, the continuum Oppenheimer-Volkoff-PG member's mass at the same N_B (engine_star_v6.continuum_member_at_NB
on the stable branch; "above the continuum maximum" where none exists), E/E_cont - 1, R_surf, a/R_surf, 2gE/R_surf, N_centre,
min(N - |v|), E_F/m; whether the maximum is bracketed (a rejected member above the largest converged one, with the bisection's interval);
the ell controls (E at ell 4, 8, 16, the fit E_0 + b ell^2, the spread as the ell error); the invariant E g^1.5 (m a)^2 at the largest-E
member against the continuum's M_max g^1.5 (m a)^2. Every number is read from the records or printed by the continuum solver.
Args: --families "label=json[,json...]" ... --eos --md."""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine_star_v6 as S6  # noqa: E402
import engine_ovpg_v2 as OV2  # noqa: E402
import numpy as np  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", nargs="+", required=True, help="label=path1.json[,path2.json]: a family's ladder and its extensions")
    ap.add_argument("--eos", default="EOS/eos_v4_tables.npz")
    ap.add_argument("--ovpg", default="OVPG/ovpg_families.json")
    ap.add_argument("--md", required=True)
    a = ap.parse_args()
    ov = json.load(open(a.ovpg))
    L = ["## The static star families (engine_star_handoff_v2.py: the continuum comparator on both equations of state)", ""]
    cross = []
    for spec in a.families:
        label, paths = spec.split("=", 1)
        recs = [json.load(open(p)) for p in paths.split(",") if os.path.exists(p)]
        if not recs:
            continue
        m = float(recs[0]["args"]["m"]); g = float(recs[0]["args"]["g"])
        fam = [f for f in ov["families"] if abs(f["m"] - m) < 1e-12 and abs(f["g"] - g) < 1e-12]
        Mmax = fam[0]["maximum"]["M_max"] if fam else None; zmax = fam[0]["maximum"]["z_c"] if fam else None
        Mscaled = fam[0]["M_max_scaled"] if fam else None
        mem, rej, ellc = [], [], []
        for R in recs:
            for q in R["members"]:
                (rej if q.get("failed") else mem).append(q)
            ellc += R.get("ell_control") or []
        seen = set(); uniq = []
        for q in sorted(mem, key=lambda q: q["N_B"]):
            key = round(q["N_B"], 3)
            if key in seen:
                continue
            seen.add(key); uniq.append(q)
        uniq.sort(key=lambda q: -q["N_centre"])
        F2 = OV2.family(a.eos, m, g, np.linspace(0.6 * zmax, 1.15 * zmax, 23)) if zmax else None
        Mmax2 = F2["maximum"]["M_max"] if (F2 and F2["maximum"]) else None; zmax2 = F2["maximum"]["z_c"] if (F2 and F2["maximum"]) else None
        L += ["### m a = %g, g = %g (%s; %d converged members%s; the continuum maximum, version 1's table equation of state: M_max %s at z_c %s, M_max g^1.5 (m a)^2 = %s; the consistent equation of state: M_max %s at z_c %s, M_max g^1.5 (m a)^2 = %s)" % (
            m, g, label, len(uniq), ("; %d rejected" % len(rej)) if rej else "", ("%.5f" % Mmax) if Mmax else "-", ("%.5f" % zmax) if zmax else "-",
            ("%.6f" % Mscaled) if Mscaled else "-", ("%.5f" % Mmax2) if Mmax2 else "-", ("%.5f" % zmax2) if zmax2 else "-", ("%.6f" % (Mmax2 * g ** 1.5 * m * m)) if Mmax2 else "-"), "",
              "| N_B | E | E_cont v1 | E/E_cont v1 - 1 | E_cont (consistent) | E/E_cont - 1 (consistent) | shift E_cont(consistent)/E_cont(v1) - 1 | R_surf | 2gE/R_surf | N_centre | min(N - abs v) |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for q in uniq:
            seed, cinfo = S6.continuum_member_at_NB(a.eos, m, g, q["N_B"], z_hi=zmax)
            Ec = cinfo.get("M") if (seed is not None and isinstance(cinfo, dict)) else None
            rel = (q["E"] / Ec - 1.0) if Ec else None
            z2, c2 = OV2.continuum_member_at_NB(a.eos, m, g, q["N_B"], z_hi=zmax2) if zmax2 else (None, None)
            Ec2 = c2.get("M") if (z2 is not None and isinstance(c2, dict)) else None
            rel2 = (q["E"] / Ec2 - 1.0) if Ec2 else None
            fx = lambda x, f_: (f_ % x) if x is not None else "-"
            L.append("| %.2f | %.6f | %s | %s | %s | %s | %s | %s | %s | %.5f | %.5f |" % (
                q["N_B"], q["E"], ("%.6f" % Ec) if Ec else "above the continuum maximum", ("%+.2e" % rel) if rel is not None else "-",
                ("%.6f" % Ec2) if Ec2 else "above the continuum maximum", ("%+.2e" % rel2) if rel2 is not None else "-",
                ("%+.2e" % (Ec2 / Ec - 1.0)) if (Ec and Ec2) else "-", fx(q.get("R_surf"), "%.2f"), fx(q.get("compactness_surf"), "%.4f"), q["N_centre"], q["min_N_minus_absv"]))
        top = max(uniq, key=lambda q: q["E"])
        densest = uniq[-1]
        above = [r for r in rej if r["NB_target"] > top["N_B"]]
        bracket = ("bracketed: the largest converged N_B %.3f, the smallest rejected %.3f (%s)" % (top["N_B"], min(r["NB_target"] for r in above), "; ".join(sorted(set(f for r in above for f in r.get("flags", []))))[:200])
                   if above else "maximum not bracketed: E still rises at the densest member (N_B %.2f)" % densest["N_B"])
        L += ["", "- Largest E %.6f at N_B %.2f; %s." % (top["E"], top["N_B"], bracket)]
        for c in ellc:
            rows = c.get("rows", {}); fit = c.get("fit_E_a_plus_b_ell2")
            Es = {k: rows[k]["E"] for k in sorted(rows, key=float) if not rows[k].get("failed")}
            L.append("- ell control at N_B %.0f: E %s; the fit E_0 + b ell^2: E_0 %s (b %s); the spread %.3e (the ell error)." % (
                c.get("NB_target", float("nan")), ", ".join("%.6f (ell %s)" % (v, k) for k, v in Es.items()), ("%.6f" % fit[1]) if fit else "-",
                ("%.3e" % fit[0]) if fit else "-", c.get("spread_E", float("nan"))))
        inv = top["E"] * g ** 1.5 * m * m
        L += ["- the invariant E g^1.5 (m a)^2 at the largest-E member %.5f against the continuum's %s (a lower bound while the maximum is not bracketed)." % (
            inv, ("%.6f" % Mscaled) if Mscaled else "-"), ""]
        cross.append((m, g, len(uniq), top["E"], top["N_B"], bool(above), inv, Mscaled, (Mmax2 * g ** 1.5 * m * m) if Mmax2 else None))
    L += ["### Across the families", "", "| m a | g | members | largest E | its N_B | bracketed | E g^1.5 (m a)^2 | continuum M_max g^1.5 (m a)^2, v1 table | the same, consistent |", "|---|---|---|---|---|---|---|---|---|"]
    for c in cross:
        L.append("| %g | %g | %d | %.6f | %.2f | %s | %.5f | %s | %s |" % (c[0], c[1], c[2], c[3], c[4], "yes" if c[5] else "no", c[6], ("%.6f" % c[7]) if c[7] else "-", ("%.6f" % c[8]) if c[8] else "-"))
    L += ["", "- every number above is read from the family records or printed by the continuum solvers (engine_star_v6.continuum_member_at_NB with version 1's table equation of state; engine_ovpg_v2 with the consistent one)."]
    open(a.md, "w").write("\n".join(L) + "\n")
    print("\n".join(L[-(len(cross) + 4):]))


if __name__ == "__main__":
    main()
