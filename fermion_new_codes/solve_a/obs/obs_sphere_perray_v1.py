#!/usr/bin/env python
"""obs_sphere_perray_v1.py (lead's first draft, SPEC_final4b_observables.md sec. 1; builder completes, every BUILDER mark is a decision left open).

The one 4-component fermion's cut coefficients as functions of m a: [100] from strand S's chain table (the per-packet column of the S2 table
is the ungraded coefficient; the graded sum is retired), the 5 tilted mesh-2 normals from the version-6 per-ray re-assemblies (per-packet
columns and eta_massless), the mesh-2 sphere average, the anisotropy ratios and the fraction 4 eta (l_P/a)^2.  Writes JSON and markdown.
Inputs: --table (strandS_table_v1.json, S1), --v6 (comma list of the 6 version-6 JSONs, one per normal; BUILDER: the paths on the box),
--out-json, --md.  The mesh-2 weights are the record's (strandS_sphere_report_v2.py); the mesh-3 list is here for J7 with the weights to be
emitted by the paper's sphere mesh routine (BUILDER)."""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
import argparse, json, math, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "strandS"))
import strandS_flat_v3 as V3  # noqa: E402  (the zone quadrature)
import strandS_flat_v4 as V4  # noqa: E402  (TableVar: the chain table with the fixed-alpha fit below mu_min)

MESH2 = {(1, 0, 0): 0.12818843, (1, 1, 0): 0.070881891, (1, 1, 1): 0.051856238, (2, 1, 0): 0.2814771, (2, 1, 1): 0.26245144, (2, 2, 1): 0.2051449}
MESH3_NEW = [(3, 1, 0), (3, 1, 1), (3, 2, 0), (3, 2, 1), (3, 2, 2), (3, 3, 1), (3, 3, 2)]  # BUILDER: weights from the paper's mesh routine
RECT = {"light": np.array([0.0, 0.5, 0.5, 1.0]), "r1223": np.array([1.0, 2.0, 2.0, 3.0]) / 3.0}
A_OVER_LP = [1.0, 1.5, 1.9, 2.0, 3.0, 4.0, 6.0, 10.0, 19.0]
READ_MISMATCH = [0.0]  # builder: the largest disagreement among repeated masses over all the version-6 files read (printed in the markdown)


def eta100_of_mass(table, masses, rho_min=1e-12, npan=49, ngl=8, nphi=16):
    """eta_100(m) = sum_points W S_ch(sqrt(m^2 + gap^2)): the chain table on strandS_flat_v3's zone quadrature (the S2 per-packet column)."""
    S = V4.TableVar(table)
    d2, W, rho = V3.zone_points(rho_min, npan, ngl, nphi)
    return {float(m): float(np.sum(W * S(np.sqrt(m * m + d2)))) for m in masses}


def read_v6(path):
    """per normal: eta_p(0) and eta_p(m) at the record's 16 masses from the version-6 rows (the per-packet columns are ungraded)."""
    D = json.load(open(path))
    p = tuple(int(v) for v in D["normal"])
    eta = {0.0: float(D["eta_massless"])}
    for r in D["rows"]:
        x = RECT[r["rect"]]
        for xv, e in zip(x, r["eta_per_packet"]):
            m = round(r["aM"] * math.sqrt(xv), 12) if xv > 0 else 0.0
            if m > 0:
                if m in eta:
                    READ_MISMATCH[0] = max(READ_MISMATCH[0], abs(eta[m] - float(e)))
                eta.setdefault(m, float(e))
    # builder: the repeated masses (the light rectangle's two middle packets, and masses shared between rows) must agree to 1e-12
    assert READ_MISMATCH[0] < 1e-12, "repeated masses disagree by %.3e in %s" % (READ_MISMATCH[0], path)
    return p, eta, D.get("ray_fits"), D.get("checks")


def sphere_average(per_normal, weights):
    masses = sorted(set.intersection(*[set(e) for e in per_normal.values()]))
    out = {}
    for m in masses:
        out[m] = sum(weights[p] * per_normal[p][m] for p in weights)
    return out, float(sum(weights.values()))


def fraction_table(eta_sphere):
    """f = 4 eta_sphere(ma) (l_P/a)^2 at the spacings of the spec; the massless row is the physical one (ma < 1e-20 for any known fermion)."""
    rows = []
    for m in sorted(eta_sphere):
        rows.append({"ma": m, "eta_sphere": eta_sphere[m], "f": {str(x): 4 * eta_sphere[m] / (x * x) for x in A_OVER_LP},
                     "a_for_f_1": 2 * math.sqrt(eta_sphere[m]), "a_for_f_0.1": 2 * math.sqrt(10 * eta_sphere[m]),
                     "a_for_f_0.01": 2 * math.sqrt(100 * eta_sphere[m])})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True)
    ap.add_argument("--v6", required=True, help="comma list of the version-6 JSONs of the 6 mesh-2 normals")
    ap.add_argument("--masses100", default="0,0.01,0.02,0.035,0.05,0.07,0.1,0.14,0.2,0.28,0.4,0.56,0.8,1.13,1.6")
    ap.add_argument("--s2", default=None, help="builder: strand S's S2 record (strandS_flat_v3.json); its 44 masses join the [100] function's list")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--md", required=True)
    a = ap.parse_args()
    T = json.load(open(a.table))
    m100 = [float(v) for v in a.masses100.split(",")]
    per = {}
    meta = {}
    for path in a.v6.split(","):
        p, eta, fits, checks = read_v6(path)
        per[p] = eta
        meta[str(p)] = {"path": os.path.basename(path), "checks": checks, "gamma_by_ray": [f["coef"][1] for f in fits] if fits else None}
    assert set(per) == set(MESH2), "the 6 mesh-2 normals are needed; got %s" % sorted(per)
    # builder: the table route at every mass printed (the record's 16 and, with --s2, S2's 44), so the markdown's table column is never a
    # fallback to the version-6 value; S2's own per-packet values beside it (the same route, so they must agree to rounding)
    s2rec = {}
    if a.s2:
        S2 = json.load(open(a.s2))
        for r in S2["rows"]:
            for xv, e in zip(RECT[r["rect"]], r["eta_per_packet"]):
                if xv > 0:
                    s2rec.setdefault(round(r["aM"] * math.sqrt(xv), 12), float(e))
    m100 = sorted(set(round(v, 12) for v in m100) | set(per[(1, 0, 0)]) | set(s2rec))
    eta100 = eta100_of_mass(T, m100)
    s2diff = max((abs(eta100[m] - s2rec[m]) for m in s2rec), default=None)
    sph, wsum = sphere_average(per, MESH2)
    out = {"label": "the one 4-component sine-lattice fermion, flat vacuum; the chain decomposition (strand S instrument), per-ray inner form",
           "eta_100_of_ma_table_route": eta100, "eta_100_S2_record": s2rec, "table_route_minus_S2_max": s2diff,
           "table_route_minus_v6_100_max": max(abs(eta100[m] - per[(1, 0, 0)][m]) for m in per[(1, 0, 0)]), "per_normal": {str(p): per[p] for p in per}, "meta": meta,
           "eta_sphere_mesh2": sph, "weights_sum": wsum, "repeated_mass_mismatch": READ_MISMATCH[0], "fraction": fraction_table(sph),
           "anisotropy": {str(p): {m: per[p][m] / per[(1, 0, 0)][m] for m in sph} for p in per},
           "mesh3_todo": [list(p) for p in MESH3_NEW]}
    json.dump(out, open(a.out_json, "w"), indent=1, default=float)
    L = ["## Task 4: the single fermion's cut coefficients (obs_sphere_perray_v1.py)", "",
         "| ma | eta_100 (table) | " + " | ".join("eta_%s" % "".join(map(str, p)) for p in MESH2) + " | eta_sphere (mesh 2) | 4 eta (l_P/a)^2 at a = 2, 6, 19 l_P |",
         "|---|---|" + "---|" * len(MESH2) + "---|---|"]
    for m in sorted(sph):
        e100 = eta100[m]
        L.append("| %g | %.10f | %s | %.8f | %.4f, %.4f, %.5f |" % (m, e100, " | ".join("%.8f" % per[p][m] for p in MESH2), sph[m],
                                                                  4 * sph[m] / 4, 4 * sph[m] / 36, 4 * sph[m] / 361))
    L.append("")
    L.append("- [100]: the table route minus the version-6 re-assembly, largest over the 17 masses: %.2e; the table route minus S2's record over its %d masses: %s." % (
        out["table_route_minus_v6_100_max"], len(s2rec), "%.2e" % s2diff if s2diff is not None else "-"))
    L.append("")
    L.append("| ma | eta_100(ma), table route (S1's chain table on the zone quadrature) |")
    L.append("|---|---|")
    for m in sorted(eta100):
        L.append("| %.6g | %.10f |" % (m, eta100[m]))
    L.append("")
    L.append("- repeated masses in the version-6 rows agree to %.2e (largest mismatch over the 6 files)." % READ_MISMATCH[0])
    L.append("- weights sum %.8f; eta_sphere(0) = %.10f against version 3's 0.9010508449 and strand 3's mesh-2 0.9001144; the paper's finer-mesh 0.9116 is not reproduced here (J7, mesh 3)." % (wsum, sph[0.0]))
    open(a.md, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
