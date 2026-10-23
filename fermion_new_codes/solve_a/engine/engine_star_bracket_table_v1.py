#!/usr/bin/env python
"""engine_star_bracket_table_v1.py (builder analyst; engine spec 29.6: "The Sfam9 brackets (averaged-back; on the box) are read in turn 38
from a table the builder prints (engine_star_bracket_table_v1: per family and ell the members' N_B, E, R_surf, min(N - |v|), the dense-table
E at the same N_B, the ell spread and the null test where three ells exist), under a rule fixed now: a maximum of E(N_B) inside the bracket
at ell 4 and at ell 8 with the next member lower at both prints 'the averaged-back family's turning point' with its ell spread; none
prints 'the averaged-back family rises through the continuum maximum as the point-sampled one did; members above N_B,max are ell
dependent'"). Per family: the bracket records of engine_star_v7 (--site-fields averaged) at each ell; the continuum on the dense zone table
(engine_ovpg_v3's control (iii), its scan of (z_c, M, N_B) on the stable branch up to the maximum mass): the continuum M at a member's N_B
by linear interpolation in N_B on that branch, "above N_B,max" beyond it; per N_B present at ell 4, 8 and 16 the spread (E_16 - E_4)/E_8 and
the null test D = E_16 - 5 E_8 + 4 E_4 with D/(E_16 - E_4) (the 0.2 rule); the rule applied mechanically and its sentence printed per
family. Every number printed by the script. Args: --families "label=ell:json,ell:json,...;..." --ovpg3 --md --out-json."""
import argparse
import json

import numpy as np


def dense_branch(pairs, m, g):
    for p in pairs:
        if abs(p["m"] - m) < 1e-9 and abs(p["g"] - g) < 1e-12:
            c = p["closures"]["dense_thermo"]
            pts = [(s["N_B"], s["M"]) for s in c["scan"] if s.get("M") is not None and s.get("N_B") is not None]
            mx = c["max_M"]
            # the stable branch: the scan's points in z_c order up to the maximum mass
            br = []
            for nb, M in pts:
                if br and M < br[-1][1]:
                    break
                br.append((nb, M))
            br.append((mx["N_B"], mx["M"]))
            br = sorted(set(br))
            return np.array([b[0] for b in br]), np.array([b[1] for b in br]), mx
    return None, None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", required=True); ap.add_argument("--ovpg3", required=True); ap.add_argument("--md", required=True); ap.add_argument("--out-json", required=True)
    a = ap.parse_args()
    O = json.load(open(a.ovpg3))
    L = ["## The turning-point brackets on averaged-back fields (engine_star_bracket_table_v1.py; engine spec 29.6; engine_star_v7, --site-fields averaged)", "",
         "- the continuum: the dense zone table's thermodynamic closure (engine_ovpg_v3, control (iii)); its M at a member's N_B interpolated on the stable branch, 'above N_B,max' beyond its maximum; the null test D = E_16 - 5 E_8 + 4 E_4 (D/(E_16 - E_4) above 0.2 in magnitude: the ell^2 fit is not an error bar).", ""]
    out = {"script": "engine_star_bracket_table_v1.py", "families": []}
    for item in a.families.split(";"):
        lab, rest = item.split("=", 1)
        recs = {}
        for part in rest.split(","):
            ell, path = part.split(":", 1)
            recs[int(ell)] = json.load(open(path))
        any_m = next(iter(recs.values()))["members"][0]
        m, g = float(any_m["m"]), float(any_m["g"])
        NBb, Mb, mx = dense_branch(O["pairs"], m, g)
        L += ["### %s: m a %g, G/a^2 %g; the dense table's N_B,max %.4f, M_max %.6f" % (lab, m, g, mx["N_B"], mx["M"]) if mx else "### %s: no dense-table closure" % lab, "",
              "| ell | N_B | E | R_surf | min(N - abs v) | the continuum M at N_B (dense) | E/M_cont - 1 |", "|---|---|---|---|---|---|---|"]
        fam = {"label": lab, "m": m, "g": g, "NB_max_dense": mx["N_B"] if mx else None, "rows": [], "rule": None}
        E_by = {}
        for ell in sorted(recs):
            mem = sorted([x for x in recs[ell]["members"] if not x.get("failed") and x.get("E") is not None], key=lambda x: float(x["N_B"]))
            for x in mem:
                nb = float(x["N_B"])
                if mx and nb <= mx["N_B"] + 1e-9:
                    Mc = float(np.interp(nb, NBb, Mb)); dev = "%+.5f" % (float(x["E"]) / Mc - 1.0); Mcs = "%.6f" % Mc
                else:
                    Mc = None; dev = "-"; Mcs = "above N_B,max"
                L.append("| %d | %.3f | %.6f | %.3f | %.5f | %s | %s |" % (ell, nb, float(x["E"]), float(x["R_surf"]), float(x["min_N_minus_absv"]), Mcs, dev))
                fam["rows"].append({"ell": ell, "N_B": nb, "E": float(x["E"]), "R_surf": float(x["R_surf"]), "min_N_minus_absv": float(x["min_N_minus_absv"]), "M_cont": Mc})
                E_by.setdefault(round(nb, 3), {})[ell] = float(x["E"])
            failed = [float(x.get("NB_target", float("nan"))) for x in recs[ell]["members"] if x.get("failed") or x.get("E") is None]
            if failed:
                L.append("| %d | not converged at N_B %s | - | - | - | - | - |" % (ell, ", ".join("%g" % f for f in failed)))
        trip = [(nb, d) for nb, d in sorted(E_by.items()) if all(k in d for k in (4, 8, 16))]
        if trip:
            L += ["", "| N_B | E_4, E_8, E_16 | (E_16 - E_4)/E_8 | D | D/(E_16 - E_4) | the 0.2 rule |", "|---|---|---|---|---|---|"]
            for nb, d in trip:
                D = d[16] - 5.0 * d[8] + 4.0 * d[4]; S = d[16] - d[4]
                L.append("| %.3f | %.6f, %.6f, %.6f | %.5f | %+.6f | %+.4f | %s |" % (nb, d[4], d[8], d[16], S / d[8], D, D / S if S else float("nan"), ("fails" if S == 0 or abs(D / S) > 0.2 else "within")))
        maxima = {}
        for ell in (4, 8):
            rows = sorted([r for r in fam["rows"] if r["ell"] == ell], key=lambda r: r["N_B"])
            E = [r["E"] for r in rows]
            mi = [i for i in range(1, len(E) - 1) if E[i] > E[i - 1] and E[i] > E[i + 1]]
            maxima[ell] = (rows[mi[0]]["N_B"], rows[mi[0]]["E"]) if mi else None
        if maxima.get(4) and maxima.get(8):
            sp = abs(maxima[4][1] - maxima[8][1]) / maxima[8][1]
            fam["rule"] = "turning point"
            L += ["", "- the rule (engine 29.6): a maximum of E(N_B) inside the bracket at ell 4 (N_B %.3f, E %.6f) and at ell 8 (N_B %.3f, E %.6f), the next member lower at both: the averaged-back family's turning point, its ell spread %.5f (|E_4 - E_8|/E_8 at the maxima)." % (maxima[4][0], maxima[4][1], maxima[8][0], maxima[8][1], sp)]
        else:
            fam["rule"] = "rises through"
            L += ["", "- the rule (engine 29.6): no maximum of E(N_B) inside the bracket at both ell 4 and ell 8 (ell 4: %s; ell 8: %s): the averaged-back family rises through the continuum maximum as the point-sampled one did; members above N_B,max are ell dependent." % (
                ("N_B %.3f" % maxima[4][0]) if maxima.get(4) else "none", ("N_B %.3f" % maxima[8][0]) if maxima.get(8) else "none")]
        L.append("")
        out["families"].append(fam)
    L += ["- every number above is printed by the script from the bracket records and engine_ovpg_v3's json; the reading is the lead's (turn 38)."]
    open(a.md, "w").write("\n".join(L) + "\n")
    json.dump(out, open(a.out_json, "w"), indent=1, default=float)
    print("\n".join(L[:6]))


if __name__ == "__main__":
    main()
