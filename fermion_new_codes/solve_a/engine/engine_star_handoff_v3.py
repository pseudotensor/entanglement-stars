#!/usr/bin/env python
"""engine_star_handoff_v3.py (builder analyst; engine spec 26.4 and 26.8: "STAR_family_handoff_v2.md from engine_star_handoff_v3 (the class
per member, the convergence-bracket column, the ell spread per member where controls exist, the consistent spline's node density) with
FILE 3 as its statement"): version 2 with (1) the lead's statement (--statement, verbatim) at the head; (2) per member its class by
engine 26.4: 1 hydrodynamic (N_B up to the continuum N_B,max, the records' summary NB_max_continuum), 2 ell dependent (beyond it with
min(N - |v|) > 0), 3 beyond N = |v| (min(N - |v|) <= 0); (3) the self-consistency failures above the largest converged member read as
the convergence bracket (no mass bracket); (4) per member with ell controls the spread (largest minus smallest converged E over ell, over
E at the member's own ell) and any unconverged control named; (5) before the tables, per m a, the consistent spline's node density near
mu = m (the table's nodes in [m, 1.01 m], [m, 1.1 m] and their smallest and largest spacing); (6) the invariant at the largest-E
class-1 member beside the largest-E member of any class. Version 2's docstring: engine spec 22.3 (1), STAR_family_handoff_v1.1: version 1 with the continuum comparator
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
    ap.add_argument("--statement", default=None, help="the lead's statement file, printed verbatim at the head")
    a = ap.parse_args()
    ov = json.load(open(a.ovpg))
    L = []
    if a.statement and os.path.exists(a.statement):
        L += [open(a.statement).read().strip(), "", "---", ""]
    L += ["## The static star families (engine_star_handoff_v3.py: the continuum comparator on both equations of state; the lead's classes per member)", ""]
    tab = np.load(a.eos)
    ms_done = set()
    for spec in a.families:
        for p in spec.split("=", 1)[1].split(","):
            if os.path.exists(p):
                mm = float(json.load(open(p))["args"]["m"])
                if mm in ms_done:
                    continue
                ms_done.add(mm)
                sel = np.abs(tab["m_hat"] - mm) < 1e-12
                mu = np.sort(np.asarray(tab["mu_hat"][sel], float))
                for hi in (1.01, 1.1):
                    w = mu[(mu >= mm) & (mu <= hi * mm)]
                    dw = np.diff(w) if len(w) > 1 else np.array([np.nan])
                    L.append("- the consistent spline's nodes near mu = m at m a %g: %d nodes in [m, %g m], spacing %.3e to %.3e (the table %s)." % (
                        mm, len(w), hi, float(np.nanmin(dw)), float(np.nanmax(dw)), os.path.basename(a.eos)))
                break
    L.append("")
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
        NBmax = next((R.get("summary", {}).get("NB_max_continuum") for R in recs if (R.get("summary") or {}).get("NB_max_continuum")), None)
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
              "| N_B | class | E | E_cont v1 | E/E_cont v1 - 1 | E_cont (consistent) | E/E_cont - 1 (consistent) | shift E_cont(consistent)/E_cont(v1) - 1 | R_surf | 2gE/R_surf | N_centre | min(N - abs v) | ell spread |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        spread_at = {}
        for c in ellc:
            rows = c.get("rows", {})
            Es = {k: rows[k]["E"] for k in rows if not rows[k].get("failed")}
            bad = [k for k in rows if rows[k].get("failed")]
            own = rows.get("%g" % float(recs[0]["args"]["ell"]), {}).get("E") or (list(Es.values())[0] if Es else None)
            if len(Es) >= 2 and own:
                spread_at[round(c.get("NB_target", -1), 2)] = "%.2f percent over ell %s%s" % (100.0 * (max(Es.values()) - min(Es.values())) / abs(own), ", ".join(sorted(Es, key=float)), ("; ell %s unconverged" % ", ".join(bad)) if bad else "")
            elif bad:
                spread_at[round(c.get("NB_target", -1), 2)] = "ell %s unconverged" % ", ".join(bad)
        klass = lambda q: (1 if (NBmax and q["N_B"] <= NBmax + 1e-9) else (2 if q["min_N_minus_absv"] > 0.0 else 3))
        for q in uniq:
            seed, cinfo = S6.continuum_member_at_NB(a.eos, m, g, q["N_B"], z_hi=zmax)
            Ec = cinfo.get("M") if (seed is not None and isinstance(cinfo, dict)) else None
            rel = (q["E"] / Ec - 1.0) if Ec else None
            z2, c2 = OV2.continuum_member_at_NB(a.eos, m, g, q["N_B"], z_hi=zmax2) if zmax2 else (None, None)
            Ec2 = c2.get("M") if (z2 is not None and isinstance(c2, dict)) else None
            rel2 = (q["E"] / Ec2 - 1.0) if Ec2 else None
            fx = lambda x, f_: (f_ % x) if x is not None else "-"
            L.append("| %.2f | %d | %.6f | %s | %s | %s | %s | %s | %s | %s | %.5f | %.5f | %s |" % (
                q["N_B"], klass(q), q["E"], ("%.6f" % Ec) if Ec else "above the continuum maximum", ("%+.2e" % rel) if rel is not None else "-",
                ("%.6f" % Ec2) if Ec2 else "above the continuum maximum", ("%+.2e" % rel2) if rel2 is not None else "-",
                ("%+.2e" % (Ec2 / Ec - 1.0)) if (Ec and Ec2) else "-", fx(q.get("R_surf"), "%.2f"), fx(q.get("compactness_surf"), "%.4f"), q["N_centre"], q["min_N_minus_absv"],
                spread_at.get(round(q["N_B"], 2), "-")))
        top = max(uniq, key=lambda q: q["E"])
        densest = uniq[-1]
        above = [r for r in rej if r["NB_target"] > top["N_B"]]
        bracket = ("the convergence bracket (the self-consistency iteration's failures; no mass bracket, engine 26.4): the largest converged N_B %.3f, the smallest failed %.3f (%s)" % (top["N_B"], min(r["NB_target"] for r in above), "; ".join(sorted(set(f for r in above for f in r.get("flags", []))))[:200])
                   if above else "no failure above the densest converged member (N_B %.2f)" % densest["N_B"])
        c1 = [q for q in uniq if klass(q) == 1]
        top1 = max(c1, key=lambda q: q["E"]) if c1 else None
        L += ["", "- The continuum N_B,max %s; the classes: %d hydrodynamic, %d ell dependent, %d beyond N = |v|." % (
            ("%.3f" % NBmax) if NBmax else "-", sum(1 for q in uniq if klass(q) == 1), sum(1 for q in uniq if klass(q) == 2), sum(1 for q in uniq if klass(q) == 3)),
              "- Largest E %.6f at N_B %.2f (class %d); %s." % (top["E"], top["N_B"], klass(top), bracket)]
        if top1:
            L.append("- the largest-E class-1 member: E %.6f at N_B %.2f, its invariant E g^1.5 (m a)^2 %.5f." % (top1["E"], top1["N_B"], top1["E"] * g ** 1.5 * m * m))
        for c in ellc:
            rows = c.get("rows", {}); fit = c.get("fit_E_a_plus_b_ell2")
            Es = {k: rows[k]["E"] for k in sorted(rows, key=float) if not rows[k].get("failed")}
            L.append("- ell control at N_B %.0f: E %s; the fit E_0 + b ell^2: E_0 %s (b %s); the spread %.3e (the ell error)." % (
                c.get("NB_target", float("nan")), ", ".join("%.6f (ell %s)" % (v, k) for k, v in Es.items()), ("%.6f" % fit[1]) if fit else "-",
                ("%.3e" % fit[0]) if fit else "-", c.get("spread_E", float("nan"))))
        inv = top["E"] * g ** 1.5 * m * m
        L += ["- the invariant E g^1.5 (m a)^2 at the largest-E member of any class %.5f; the continuum's %s." % (
            inv, ("%.6f" % Mscaled) if Mscaled else "-"), ""]
        cross.append((m, g, len(uniq), top["E"], top["N_B"], bool(above), inv, Mscaled, (Mmax2 * g ** 1.5 * m * m) if Mmax2 else None))
    L += ["### Across the families", "", "| m a | g | members | largest E | its N_B | a convergence failure above it | E g^1.5 (m a)^2 | continuum M_max g^1.5 (m a)^2, v1 table | the same, consistent |", "|---|---|---|---|---|---|---|---|---|"]
    for c in cross:
        L.append("| %g | %g | %d | %.6f | %.2f | %s | %.5f | %s | %s |" % (c[0], c[1], c[2], c[3], c[4], "yes" if c[5] else "no", c[6], ("%.6f" % c[7]) if c[7] else "-", ("%.6f" % c[8]) if c[8] else "-"))
    L += ["", "- every number above is read from the family records or printed by the continuum solvers (engine_star_v6.continuum_member_at_NB with version 1's table equation of state; engine_ovpg_v2 with the consistent one)."]
    open(a.md, "w").write("\n".join(L) + "\n")
    print("\n".join(L[-(len(cross) + 4):]))


if __name__ == "__main__":
    main()
