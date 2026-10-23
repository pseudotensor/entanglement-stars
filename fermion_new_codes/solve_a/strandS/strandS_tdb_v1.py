#!/usr/bin/env python
"""Strand S diagnostic (b), the detailed-balance temperature (SPEC sec. 9.2; the lead's t_db draft completed by the builder analyst): from
D-a1's report JSON (strandS_dos_report_v1.py over strandS_dos_v3.py's three sizes), for each count S of the static chart (the whole sea, its
near-surface part R < 2 A_0, the interior r < s, the near exterior s <= r < 8 A_0, the far box r >= 8 A_0; all with the edge pairs excluded) at
each marker and packet mass: dS/dr_h by the two-point differences between neighbouring sizes (at the same marker: matched N_s and matched
T r_h = 1/(4 pi)), T_db/T_H = 12 j (aM)^2 r_h/(dS/dr_h) at the midpoint r_h of each pair, at aM in --aMs with j = log 2 (the light rectangle;
r1223's j = 0.174416 for its edge), the local power p = d log S/d log r_h; beside them T_ray/T_H = theta_r from task 3's B0 (the planar chain's
plateau, --b-json, the r = 1/4 packet at each spacing) and later B2 and task 6's kappa_p; the plain cut's T_db/T_H = 1/R_plain and the graded
cut's 9/2 as the reference rows.  Args: --dos-report --b-json (optional) --json --md."""
import argparse
import json
import math


def f(v, d=4):
    if v is None:
        return "-"
    v = float(v)
    return "%.*g" % (d, v) if (v == 0 or 1e-3 <= abs(v) < 1e6) else "%.3e" % v


JS = {"light": math.log(2), "r1223": (1 / 3) * math.log(1 / 3) - (4 / 3) * math.log(2 / 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dos-report", required=True)
    ap.add_argument("--b-json", default=None)
    ap.add_argument("--aMs", default="0.1,0.01,0.0051,0.0038")
    ap.add_argument("--json", required=True)
    ap.add_argument("--md", required=True)
    a = ap.parse_args()
    D = json.load(open(a.dos_report))
    rows = D["rows"]
    counts = {"whole": lambda r: r["S_noedge"], "near (R < 2A0)": lambda r: r["S_ns_2A0_noedge"], "interior (r < s)": lambda r: r["S_ns_s_noedge"]}
    if rows and rows[0].get("S_ns_8A0_noedge") is not None:
        counts["near exterior (s to 8A0)"] = lambda r: r["S_ns_8A0_noedge"] - r["S_ns_s_noedge"]
        counts["far box (> 8A0)"] = lambda r: r["S_noedge"] - r["S_ns_8A0_noedge"]
    aMs = [float(x) for x in a.aMs.split(",")]
    out = {"tdb": []}
    L = ["# Strand S diagnostic (b): the detailed-balance temperature against T_H and the ray map (strandS_tdb_v1.py)", "",
         "T_db/T_H = 12 j (aM)^2 r_h/(dS/dr_h) (SPEC sec. 9.2; the one match G/a^2 = pi/(6 j (aM)^2) as the conversion, never a calibration), dS/dr_h "
         "from neighbouring sizes at the same marker; j = log 2 (light) unless stated.", "",
         "| marker | m | count | r_h pair | p = dlogS/dlogr_h | dS/dr_h | " + " | ".join("T_db/T_H at aM = %g" % x for x in aMs) + " |",
         "|---|---|---|---|---|---|" + "---|" * len(aMs)]
    keys = sorted({(r["marker"], r["mass"]) for r in rows})
    for mk, mass in keys:
        rr = sorted([r for r in rows if r["marker"] == mk and r["mass"] == mass], key=lambda r: r["r_h"])
        for name, fn in counts.items():
            for r1, r2 in zip(rr[:-1], rr[1:]):
                s1, s2 = fn(r1), fn(r2)
                dS = (s2 - s1) / (r2["r_h"] - r1["r_h"])
                rm = 0.5 * (r1["r_h"] + r2["r_h"])
                try:
                    p = math.log(s2 / s1) / math.log(r2["r_h"] / r1["r_h"])
                except (ValueError, ZeroDivisionError):
                    p = None
                tdb = [12 * JS["light"] * x * x * rm / dS if dS != 0 else None for x in aMs]
                out["tdb"].append({"marker": mk, "mass": mass, "count": name, "r_h": [r1["r_h"], r2["r_h"]], "p": p, "dS_drh": dS, "T_db_over_T_H": tdb})
                L.append("| %s | %s | %s | %s to %s | %s | %s | %s |" % (mk, mass.replace("m=", ""), name, f(r1["r_h"]), f(r2["r_h"]), f(p, 3), f(dS, 4),
                                                                   " | ".join(f(t, 3) for t in tdb)))
    if a.b_json:
        B = json.load(open(a.b_json))
        ray = [r for r in B["rows"] if "plateau" in r and r["branch"] == 0 and r["m_over_kappa"] == 0]
        out["T_ray"] = [{"kappa_a": r["kappa_a"], "ratio": r["ratio"], "theta_r": r["theta_r"]} for r in ray]
        L += ["", "- T_ray/T_H on the planar chain (task 3 B0, the record's package, prescribed flow): " + "; ".join(
            "kappa a %s, omega/kappa %s: theta_r %s" % (f(r["kappa_a"]), f(r["ratio"]), f(r["theta_r"], 7)) for r in ray) + "."]
    L += ["", "- Reference rows (the graded and plain cuts are not counts): the graded cut's first-law comparison is R_gr = 2/9, i.e. T_db/T_H = 9/2 if it "
          "were a count; the plain cut's T_db/T_H = 1/R_plain with R_plain the unsigned overshoot of RESULT_strandS's tables at the same spacing."]
    json.dump(out, open(a.json, "w"), indent=1, default=float)
    open(a.md, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
