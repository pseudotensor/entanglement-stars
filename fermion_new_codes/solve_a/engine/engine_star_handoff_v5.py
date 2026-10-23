#!/usr/bin/env python
"""engine_star_handoff_v5.py (builder analyst; engine spec 29.6: "handoff v3's comparator column is the dense-zone-table thermodynamic
closure at every family (control (iii)), the version-2 spline a labelled column ... FILE 5 is appended to STAR_family_statement_v3 as
version 3.1 and handoff v3's final print carries it at the head"): version 4 with (1) the head carrying statement version 3.1 after
version 3's paragraphs (--statement-v31: its header line and its paragraphs, verbatim); (2) per member the continuum mass on the dense zone
table (engine_ovpg_v3.mass_at_NB with the thermodynamic closure on the tables stored in engine_ovpg_v3's json, z between the closure's
first converged centre and its N_B maximum) as the comparator column, the version-2 spline's column kept and labelled; (3) the class
boundary at the dense table's N_B,max (class 1 up to it), the records' version-1 N_B,max printed beside. Nothing else changes;
version 4's docstring follows.
engine_star_handoff_v4.py (builder analyst; engine spec 28.4 and 28.10: "Handoff v3 (engine_star_handoff_v4) when the (0.1, 1/64) ell
control lands: the member, a fields-definition line per family, the null-test column, FILE 2 at its head"): engine_star_handoff_v3 with
(1) the head composed mechanically, verbatim: the first line of STAR_family_statement_v3.md (its header, which says "version 2.1's first four
paragraphs stand verbatim before the paragraphs below"), then the first four paragraphs of STAR_family_statement_v2.1.md (the paragraphs
after its header line), then the paragraphs of STAR_family_statement_v3.md after its header (--statement-v21, --statement-v3; the structure
is asserted, nothing is retyped); (2) the (0.1, 1/64) family (Sfam8/star_star8box_m0.1_g0.015625_max.json) as a sixth family (by the
launcher's --families); (3) per family a fields-definition line read from each record's script and args: the field module the script
imports (the line "import engine_fields_vN as FLD" of code/<script>) and --site-fields when the script has it (engine_star_v7: averaged =
engine_fields_v14.site_fields_kernel_average, the declared definition; dense = point sampled, the control); engine_fields_v3 without the
option = point-sampled site fields (its quintic splines of the constraint solution sampled at the sites), the goal's control definition
(engine spec 28.4); (4) the null test of engine_star_nulltest_v1 (D = E_16 - 5 E_8 + 4 E_4, D/(E_16 - E_4), the 0.2 rule) as a column of
each family's table and in each ell-control line; (5) per family the continuum turning points of engine_ovpg_v3 (--ovpg3, its json) beside
the records' continuum N_B,max: version 2's thermodynamic N_B,max and M_max and the native kinetic stress's (q_i = sin k_i, version 2's e
and n; the spec), with the dense-zone-table controls; (6) ell-control entries repeated in several records of a family printed once (version 3
printed the (0.2, 1/4) entry at N_B 138.844 twice), and members repeated in several records (the same N_B to 1e-3; version 3 keeps the first)
named in one line with their E difference. Everything else (the classes, the convergence bracket, the comparators E_cont v1 and v2 per
member, the spline node density, the invariants, the cross-family table) is version 3's code. Version 3's docstring follows.
engine_star_handoff_v3.py (builder analyst; engine spec 26.4 and 26.8: "STAR_family_handoff_v2.md from engine_star_handoff_v3 (the class
per member, the convergence-bracket column, the ell spread per member where controls exist, the consistent spline's node density) with
FILE 3 as its statement"): version 2 with (1) the lead's statement (--statement, verbatim) at the head; (2) per member its class by
engine 26.4: 1 hydrodynamic (N_B up to the continuum N_B,max, the records' summary NB_max_continuum), 2 ell dependent (beyond it with
min(N - |v|) > 0), 3 beyond N = |v| (min(N - |v|) <= 0); (3) the self-consistency failures above the largest converged member read as
the convergence bracket (no mass bracket); (4) per member with ell controls the spread (largest minus smallest converged E over ell, over
E at the member's own ell) and any unconverged control named; (5) before the tables, per m a, the consistent spline's node density near
mu = m (the table's nodes in [m, 1.01 m], [m, 1.1 m] and their smallest and largest spacing); (6) the invariant at the largest-E
class-1 member beside the largest-E member of any class. Args: --families "label=json[,json...]" ... --eos --ovpg --ovpg3 --md
--statement-v21 --statement-v3 (or --statement for one verbatim file)."""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine_star_v6 as S6  # noqa: E402
import engine_ovpg_v2 as OV2  # noqa: E402
import engine_star_nulltest_v1 as NT  # noqa: E402
import engine_ovpg_v3 as OV3  # noqa: E402
import numpy as np  # noqa: E402

CODE = os.path.dirname(os.path.abspath(__file__))


def compose_head(v21_path, v3_path):
    """v3's header line, v2.1's first four paragraphs (after its header), v3's paragraphs (after its header); blocks split on blank lines."""
    b21 = open(v21_path).read().strip("\n").split("\n\n")
    b3 = open(v3_path).read().strip("\n").split("\n\n")
    assert b21[0].startswith("# The static star families: the lead's statement for handoff version 2.1") and "\n" not in b21[0], b21[0][:120]
    assert b3[0].startswith("# The static star families: the lead's statement for handoff version 3") and "\n" not in b3[0], b3[0][:120]
    assert "version 2.1's first four paragraphs stand verbatim before the paragraphs below" in b3[0]
    assert len(b21) >= 5 and all(p.strip() and not p.startswith("#") and "\n" not in p for p in b21[1:5]), [p[:40] for p in b21]
    assert len(b3) >= 2 and all(p.strip() and not p.startswith("#") and "\n" not in p for p in b3[1:]), [p[:40] for p in b3]
    blocks = [b3[0]] + b21[1:5] + b3[1:]
    info = "head: v3 header + v2.1 paragraphs 1-4 (%s) + v3 paragraphs (%s)" % (
        "; ".join(p[:30] for p in b21[1:5]), "; ".join(p[:30] for p in b3[1:]))
    return "\n\n".join(blocks), info


def compose_head_v31(v21_path, v3_path, v31_path):
    """version 4's head, then statement version 3.1's header line and its paragraphs, verbatim."""
    head, info = compose_head(v21_path, v3_path)
    b31 = open(v31_path).read().strip("\n").split("\n\n")
    assert b31[0].startswith("# The static star families: version 3.1") and "\n" not in b31[0], b31[0][:120]
    assert len(b31) >= 2 and all(p.strip() and not p.startswith("#") for p in b31[1:]), [p[:40] for p in b31]
    return head + "\n\n" + "\n\n".join(b31), info + " + v3.1 (%s)" % "; ".join(p[:30] for p in b31[1:])


def fields_definition(path, R):
    """the site-field definition of one record, read from its script's source and its args."""
    script = R.get("script") or "?"
    args = R.get("args") or {}
    src = os.path.join(CODE, script)
    mod = None
    if os.path.exists(src):
        mt = re.search(r"^import (engine_fields_v\d+) as FLD\s*$", open(src).read(), re.M)
        mod = mt.group(1) if mt else None
    sf = args.get("site_fields")
    if sf == "averaged":
        what = "averaged-back site fields (%s.site_fields_kernel_average, the adjoint kernels): the declared definition" % (mod or "?")
    elif sf == "dense":
        what = "point-sampled site fields (%s, --site-fields dense): the control definition" % (mod or "?")
    elif mod == "engine_fields_v3":
        what = "point-sampled site fields (engine_fields_v3: the constraint solution's quintic splines sampled at the sites): the goal's control definition"
    else:
        what = "field module %s, no --site-fields option in the record: definition not identified by this script" % (mod or "not found")
    return "%s by %s (imports %s; --site-fields %s): %s" % (os.path.basename(path), script, mod or "?", sf if sf else "not an option of this script", what)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", nargs="+", required=True, help="label=path1.json[,path2.json]: a family's ladder and its extensions")
    ap.add_argument("--eos", default="EOS/eos_v4_tables.npz")
    ap.add_argument("--ovpg", default="OVPG/ovpg_families.json")
    ap.add_argument("--ovpg3", default=None, help="engine_ovpg_v3's json (the native-stress turning points)")
    ap.add_argument("--md", required=True)
    ap.add_argument("--statement", default=None, help="one statement file printed verbatim at the head (version 3's option)")
    ap.add_argument("--statement-v21", default=None)
    ap.add_argument("--statement-v3", default=None)
    ap.add_argument("--statement-v31", default=None)
    a = ap.parse_args()
    ov = json.load(open(a.ovpg))
    ov3 = json.load(open(a.ovpg3)) if (a.ovpg3 and os.path.exists(a.ovpg3)) else None
    L = []
    if a.statement_v21 and a.statement_v3 and a.statement_v31:
        head, info = compose_head_v31(a.statement_v21, a.statement_v3, a.statement_v31)
        print(info)
        L += [head, "", "---", ""]
    elif a.statement_v21 and a.statement_v3:
        head, info = compose_head(a.statement_v21, a.statement_v3)
        print(info)
        L += [head, "", "---", ""]
    elif a.statement and os.path.exists(a.statement):
        L += [open(a.statement).read().strip(), "", "---", ""]
    L += ["## The static star families (engine_star_handoff_v5.py: version 3's tables with the fields definition per record, the ell null test "
          "per triple, the continuum turning points of engine_ovpg_v3, and the dense zone table's continuum mass per member as the comparator; the class boundary at the dense table's N_B,max)", ""]
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
                    dw = np.diff(w) if len(w) > 1 else None
                    L.append("- the consistent spline's nodes near mu = m at m a %g: %d nodes in [m, %g m]%s (the table %s)." % (
                        mm, len(w), hi, (", spacing %.3e to %.3e" % (float(dw.min()), float(dw.max()))) if dw is not None else ", no spacing (fewer than two nodes)",
                        os.path.basename(a.eos)))
                break
    L.append("")
    cross = []
    for spec in a.families:
        label, paths = spec.split("=", 1)
        plist = [p for p in paths.split(",") if os.path.exists(p)]
        missing = [p for p in paths.split(",") if not os.path.exists(p)]
        recs = [json.load(open(p)) for p in plist]
        if not recs:
            L += ["### %s: no record found (%s)" % (label, ", ".join(missing)), ""]
            continue
        m = float(recs[0]["args"]["m"]); g = float(recs[0]["args"]["g"])
        fam = [f for f in ov["families"] if abs(f["m"] - m) < 1e-12 and abs(f["g"] - g) < 1e-12]
        Mmax = fam[0]["maximum"]["M_max"] if fam else None; zmax = fam[0]["maximum"]["z_c"] if fam else None
        Mscaled = fam[0]["M_max_scaled"] if fam else None
        NBmax_v1 = next((R.get("summary", {}).get("NB_max_continuum") for R in recs if (R.get("summary") or {}).get("NB_max_continuum")), None)
        eos3d = None; zlo_d = ztop_d = None; NBmax = NBmax_v1
        if ov3:
            P3d = [R3 for R3 in ov3["pairs"] if abs(R3["m"] - m) < 1e-12 and abs(R3["g"] - g) < 1e-12]
            T3d = [T for T in ov3.get("tables", []) if abs(T["m"] - m) < 1e-12]
            if P3d and T3d:
                Fd, _ = OV3.eos_functions(m, {k: np.asarray(v, float) for k, v in T3d[0]["table"].items()})
                eos3d = (Fd["e"], Fd["P"], Fd["n"])
                cd = P3d[0]["closures"]["dense_thermo"]
                zlo_d, ztop_d = cd["z_first_ok"], cd["max_N_B"]["z_c"]
                NBmax = cd["max_N_B"]["N_B"]
        mem, rej = [], []
        for p, R in zip(plist, recs):
            for q in R["members"]:
                (rej if q.get("failed") else mem).append(dict(q, _record=os.path.basename(p)))
        nulls = NT.null_entries([(os.path.basename(p), R) for p, R in zip(plist, recs)])
        seen = {}; uniq = []; dups = []
        for q in sorted(mem, key=lambda q: q["N_B"]):
            key = round(q["N_B"], 3)
            if key in seen:
                dups.append((q, seen[key]))
                continue
            seen[key] = q; uniq.append(q)
        uniq.sort(key=lambda q: -q["N_centre"])
        F2 = OV2.family(a.eos, m, g, np.linspace(0.6 * zmax, 1.15 * zmax, 23)) if zmax else None
        Mmax2 = F2["maximum"]["M_max"] if (F2 and F2["maximum"]) else None; zmax2 = F2["maximum"]["z_c"] if (F2 and F2["maximum"]) else None
        L += ["### m a = %g, g = %g (%s; %d converged members%s; the continuum maximum, version 1's table equation of state: M_max %s at z_c %s, M_max g^1.5 (m a)^2 = %s; the consistent equation of state: M_max %s at z_c %s, M_max g^1.5 (m a)^2 = %s)" % (
            m, g, label, len(uniq), ("; %d rejected" % len(rej)) if rej else "", ("%.5f" % Mmax) if Mmax else "-", ("%.5f" % zmax) if zmax else "-",
            ("%.6f" % Mscaled) if Mscaled else "-", ("%.5f" % Mmax2) if Mmax2 else "-", ("%.5f" % zmax2) if zmax2 else "-", ("%.6f" % (Mmax2 * g ** 1.5 * m * m)) if Mmax2 else "-"), ""]
        L += ["- fields definition: " + "; ".join(fields_definition(p, R) for p, R in zip(plist, recs)) + "."]
        if missing:
            L.append("- records named by the launcher and not found: %s." % ", ".join(missing))
        if ov3:
            P3 = [R3 for R3 in ov3["pairs"] if abs(R3["m"] - m) < 1e-12 and abs(R3["g"] - g) < 1e-12]
            if P3:
                C = P3[0]["closures"]
                cl = lambda k: "N_B,max %.4f, M_max %.6f%s" % (C[k]["max_N_B"]["N_B"], C[k]["max_M"]["M"], "" if (C[k]["max_N_B"]["bracketed"] and C[k]["max_M"]["bracketed"]) else " (NOT bracketed)")
                L.append("- the continuum turning points (engine_ovpg_v3, %s) beside the records' N_B,max %s: thermodynamic (version 2) %s; native kinetic stress "
                         "q_i = sin k_i in place of P (version 2's e and n; the spec) %s; controls on the dense zone table: thermodynamic %s; native q_i = sin k_i %s; "
                         "native q_i = Q_i = sin(2k_i)/2 (S^(Q)) %s; p = chi - e (the b -> 0 limit of the two-row closure's J/b - rho) %s." % (
                             os.path.basename(a.ovpg3), ("%.3f" % NBmax) if NBmax else "-", cl("v2_thermo"), cl("v3_native_sin"), cl("dense_thermo"),
                             cl("dense_native_sin"), cl("dense_native_SQ"), cl("dense_chi_diag")))
            else:
                L.append("- engine_ovpg_v3 (%s) has no entry at this (m a, g)." % os.path.basename(a.ovpg3))
        if dups:
            L.append("- members repeated across records (version 3 keeps the first by N_B): %s." % "; ".join(
                "N_B %.3f in %s (E %.6f) and %s (E %.6f), relative difference %.1e" % (q["N_B"], s["_record"], s["E"], q["_record"], q["E"], abs(q["E"] / s["E"] - 1)) for q, s in dups))
        L += ["- the class boundary: the dense zone table's N_B,max %s (the records' version-1 N_B,max %s beside); the comparator column: the dense table's continuum mass at the member's N_B (engine 29.6); the version-2 spline's column is labelled." % (
            ("%.4f" % NBmax) if NBmax else "-", ("%.4f" % NBmax_v1) if NBmax_v1 else "-")]
        L += ["", "| N_B | class | E | E_cont (dense table; the comparator) | E/E_cont - 1 (dense) | E_cont v1 | E/E_cont v1 - 1 | E_cont (version-2 spline; labelled) | E/E_cont - 1 (version-2 spline) | shift E_cont(consistent)/E_cont(v1) - 1 | R_surf | 2gE/R_surf | N_centre | min(N - abs v) | ell spread | ell null test D = E_16 - 5E_8 + 4E_4 (D/(E_16 - E_4) of E_16 - E_4; the 0.2 rule) |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        spread_at, null_at = {}, {}
        for x in nulls:
            Es = x["E"]; bad = x["unconverged"]
            own = Es.get("%g" % float(recs[0]["args"]["ell"])) or (list(Es.values())[0] if Es else None)
            k2 = round(x["NB_target"], 2)
            if len(Es) >= 2 and own:
                spread_at[k2] = "%.2f percent over ell %s%s" % (100.0 * (max(Es.values()) - min(Es.values())) / abs(own), ", ".join(sorted(Es, key=float)), ("; ell %s unconverged" % ", ".join(bad)) if bad else "")
            elif bad:
                spread_at[k2] = "ell %s unconverged" % ", ".join(bad)
            null_at[k2] = NT.fmt_entry(x)
        klass = lambda q: (1 if (NBmax and q["N_B"] <= NBmax + 1e-9) else (2 if q["min_N_minus_absv"] > 0.0 else 3))
        for q in uniq:
            seed, cinfo = S6.continuum_member_at_NB(a.eos, m, g, q["N_B"], z_hi=zmax)
            Ec = cinfo.get("M") if (seed is not None and isinstance(cinfo, dict)) else None
            rel = (q["E"] / Ec - 1.0) if Ec else None
            z2, c2 = OV2.continuum_member_at_NB(a.eos, m, g, q["N_B"], z_hi=zmax2) if zmax2 else (None, None)
            Ec2 = c2.get("M") if (z2 is not None and isinstance(c2, dict)) else None
            rel2 = (q["E"] / Ec2 - 1.0) if Ec2 else None
            fx = lambda x, f_: (f_ % x) if x is not None else "-"
            Ecd = None; note_d = "-"
            if eos3d is not None:
                md_, note_d = OV3.mass_at_NB(m, g, eos3d, q["N_B"], zlo_d, ztop_d)
                Ecd = md_["M"] if md_ else None
            reld = (q["E"] / Ecd - 1.0) if Ecd else None
            L.append("| %.2f | %d | %.6f | %s | %s | %s | %s | %s | %s | %s | %s | %s | %.5f | %.5f | %s | %s |" % (
                q["N_B"], klass(q), q["E"], ("%.6f" % Ecd) if Ecd else (note_d or "-"), ("%+.2e" % reld) if reld is not None else "-",
                ("%.6f" % Ec) if Ec else "above the continuum maximum", ("%+.2e" % rel) if rel is not None else "-",
                ("%.6f" % Ec2) if Ec2 else "above the continuum maximum", ("%+.2e" % rel2) if rel2 is not None else "-",
                ("%+.2e" % (Ec2 / Ec - 1.0)) if (Ec and Ec2) else "-", fx(q.get("R_surf"), "%.2f"), fx(q.get("compactness_surf"), "%.4f"), q["N_centre"], q["min_N_minus_absv"],
                spread_at.get(round(q["N_B"], 2), "-"), null_at.get(round(q["N_B"], 2), "-")))
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
        if not nulls:
            L.append("- no ell control in the records.")
        for x in nulls:
            fit = x["fit"]
            Es = {k: x["E"][k] for k in sorted(x["E"], key=float)}
            L.append("- ell control at N_B %.0f%s: E %s%s; the fit E_0 + b ell^2: E_0 %s (b %s); the spread %.3e (the ell error); the null test %s." % (
                x["NB_target"], (" (%s)" % ", ".join(x["records"])) if len(x["records"]) > 1 else "", ", ".join("%.6f (ell %s)" % (v, k) for k, v in Es.items()),
                ("; ell %s unconverged" % ", ".join(x["unconverged"])) if x["unconverged"] else "", ("%.6f" % fit[1]) if fit else "-",
                ("%.3e" % fit[0]) if fit else "-", x["spread_E_record"] if x["spread_E_record"] is not None else float("nan"),
                ("D %.6f, E_16 - E_4 %.6f, D/(E_16 - E_4) %.4f: %s" % (x["D"], x["S"], x["ratio"], "above 0.2, the fit is not an error bar (the spread is the error)" if x["fails"] else "within 0.2"))
                if "D" in x else "not applicable (no converged triple)"))
        inv = top["E"] * g ** 1.5 * m * m
        L += ["- the invariant E g^1.5 (m a)^2 at the largest-E member of any class %.5f; the continuum's %s." % (
            inv, ("%.6f" % Mscaled) if Mscaled else "-"), ""]
        nat = None
        if ov3:
            P3 = [R3 for R3 in ov3["pairs"] if abs(R3["m"] - m) < 1e-12 and abs(R3["g"] - g) < 1e-12]
            nat = P3[0]["closures"]["v3_native_sin"]["max_M"]["M"] * g ** 1.5 * m * m if P3 else None
        cross.append((m, g, len(uniq), top["E"], top["N_B"], bool(above), inv, Mscaled, (Mmax2 * g ** 1.5 * m * m) if Mmax2 else None, nat))
    L += ["### Across the families", "", "| m a | g | members | largest E | its N_B | a convergence failure above it | E g^1.5 (m a)^2 | continuum M_max g^1.5 (m a)^2, v1 table | the same, consistent | the same, native kinetic stress q = sin k (engine_ovpg_v3) |", "|---|---|---|---|---|---|---|---|---|---|"]
    for c in cross:
        L.append("| %g | %g | %d | %.6f | %.2f | %s | %.5f | %s | %s | %s |" % (c[0], c[1], c[2], c[3], c[4], "yes" if c[5] else "no", c[6], ("%.6f" % c[7]) if c[7] else "-", ("%.6f" % c[8]) if c[8] else "-", ("%.6f" % c[9]) if c[9] else "-"))
    L += ["", "- every number above is read from the family records or printed by the continuum solvers (engine_star_v6.continuum_member_at_NB with version 1's table equation of state; engine_ovpg_v2 with the consistent one; engine_ovpg_v3's json for the native-stress turning points); the null test by engine_star_nulltest_v1.null_entries."]
    open(a.md, "w").write("\n".join(L) + "\n")
    print("\n".join(L[-(len(cross) + 4):]))


if __name__ == "__main__":
    main()
