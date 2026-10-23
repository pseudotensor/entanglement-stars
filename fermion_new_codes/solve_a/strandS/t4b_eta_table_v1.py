#!/usr/bin/env python
"""Task 4 of the rewritten goal (builder analyst; a table for the lead's specification, labelled): the single fermion's per-normal cut
coefficient eta_p(m a) from strandS_sphere_v6.py's per-ray re-assemblies of the six mesh-2 normals (S3v3's saved chain values: the massless
channel and the 16 masses aM sqrt(x) of the record's ladders, aM in {0.4, 0.2, 0.1, 0.05}, x in {1/2, 1} and {1/3, 2/3}), read from the rows'
eta_per_packet; per mass the six normals, the mesh-2 solid-angle average (weights from sphere.triangles(2), the paper's sphere.py), the
anisotropy (max/min - 1 over the normals), and the same with the single-ray (phi = 0) inner form beside it.  eta_p is the record's
normalization (per two-component channel, per unit area of the cut in lattice units, (2 pi)^-2 Int d^2q S/|p| over the zone).
Args: --jsons (comma list of v6 JSONs) --orient --json --md."""
import argparse
import json
import math
import sys


def f(v, d=9):
    return "-" if v is None else "%.*g" % (d, v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsons", required=True)
    ap.add_argument("--orient", required=True)
    ap.add_argument("--json", required=True)
    ap.add_argument("--md", required=True)
    a = ap.parse_args()
    sys.path.insert(0, a.orient)
    import sphere
    from lattice_entropy import canonical
    w = {}
    for T in sphere.triangles(2):
        area, _ = sphere.geometry(T)
        for p in T:
            c = tuple(int(x) for x in canonical(p))
            w[c] = w.get(c, 0.0) + 12 * area / math.pi / 3
    X = {"light": [0.0, 0.5, 0.5, 1.0], "r1223": [1 / 3, 2 / 3, 2 / 3, 1.0]}
    per = {}
    for p in a.jsons.split(","):
        d = json.load(open(p))
        nrm = tuple(d["normal"])
        tab, tab0 = {0.0: d["eta_massless"]}, {0.0: d["runs"][0]["eta_massless_phi0"]}
        for rows, T in ((d["rows"], tab), (d["rows_phi0"], tab0)):
            for r in rows:
                for x, e in zip(X[r["rect"]], r["eta_per_packet"]):
                    m = round(r["aM"] * math.sqrt(x), 12)
                    T[m] = e
        per[nrm] = (tab, tab0)
    masses = sorted(set.intersection(*[set(t.keys()) for t, _ in per.values()]))
    out = {"weights": {str(k): v for k, v in w.items()}, "normals": [list(k) for k in sorted(per)], "rows": []}
    L = ["# Task 4: the single fermion's per-normal cut coefficient eta_p(m a) (t4b_eta_table_v1.py over strandS_sphere_v6.py's per-ray "
         "re-assemblies of the record's six mesh-2 normals)", "",
         "| m a | " + " | ".join("eta %s" % (n,) for n in sorted(per)) + " | sphere (mesh-2 average) | anisotropy max/min - 1 | sphere with the phi = 0 form |",
         "|---|" + "---|" * (len(per) + 3)]
    full = set(per) == set(w)
    for m in masses:
        vals = {n: per[n][0][m] for n in per}
        vals0 = {n: per[n][1][m] for n in per}
        sph = sum(w[n] * vals[n] for n in w) if full else None
        sph0 = sum(w[n] * vals0[n] for n in w) if full else None
        an = max(vals.values()) / min(vals.values()) - 1
        out["rows"].append({"m_a": m, "eta_p": {str(n): v for n, v in vals.items()}, "eta_p_phi0": {str(n): v for n, v in vals0.items()},
                            "sphere": sph, "sphere_phi0": sph0, "anisotropy": an})
        L.append("| %s | " % f(m, 6) + " | ".join(f(vals[n]) for n in sorted(per)) + " | %s | %s | %s |" % (f(sph), f(an, 4), f(sph0)))
    L += ["", "- Every number above is read from the records by the script; the normals' weights: " + "; ".join("%s %s" % (k, f(v, 8)) for k, v in sorted(w.items())) + "."]
    json.dump(out, open(a.json, "w"), indent=1)
    open(a.md, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
