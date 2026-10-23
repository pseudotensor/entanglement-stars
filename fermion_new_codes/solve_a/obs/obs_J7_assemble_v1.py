#!/usr/bin/env python3
"""obs_J7_assemble_v1.py (builder analyst; J7 of SPEC_final4b_observables.md): eta_sphere from the per-normal massless coefficients of
strandS_sphere_v7.py (t4b/J7/sphere_<pqr>_m3.json, "eta_massless") with the paper's sphere quadrature (sphere.py's triangles(M) and geometry: the
weight 12 area/pi per triangle of the octant face's mesh M, each triangle the mean of its three vertex normals, canonical normals), at M = 2 and
M = 3; the weights and their sum emitted (the spec's BUILDER item); printed against J1's mesh-2 value 0.9010503 (obs_sphere_perray_v1, another
instrument version) and the paper's 0.9116. Args: --dir --orient --md --out-json."""
import argparse
import glob
import json
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="<internal>/J7")
    ap.add_argument("--orient", default="<internal>/orient")
    ap.add_argument("--md", required=True)
    ap.add_argument("--out-json", required=True)
    a = ap.parse_args()
    sys.path.insert(0, a.orient)
    import numpy as np
    from lattice_entropy import canonical as _canonical
    canonical = lambda p: tuple(int(x) for x in _canonical(p))
    from sphere import geometry, triangles
    eta = {}
    for f in sorted(glob.glob(os.path.join(a.dir, "sphere_*_m3.json"))):
        D = json.load(open(f))
        p = canonical(tuple(int(x) for x in D["meta"]["normal"].split(",")))
        eta[p] = float(D["eta_massless"])
    L = ["## J7: eta_sphere at meshes 2 and 3 from strandS_sphere_v7's massless per-normal coefficients and the paper's sphere quadrature (obs_J7_assemble_v1.py)", "",
         "### The per-normal coefficients", "", "| canonical normal | eta_massless |", "|---|---|"]
    for p in sorted(eta):
        L.append("| %s | %.8f |" % (p, eta[p]))
    out = {"script": "obs_J7_assemble_v1.py", "eta": {str(p): v for p, v in eta.items()}, "meshes": {}}
    for M in (2, 3):
        Ts = triangles(M)
        need = sorted({canonical(p) for T in Ts for p in T})
        missing = [p for p in need if p not in eta]
        rows = []
        tot = wsum = 0.0
        for T in Ts:
            area, diam = geometry(T)
            w = 12 * area / np.pi
            if not missing:
                tot += w * np.mean([eta[canonical(p)] for p in T])
            wsum += w
            rows.append({"triangle": [list(map(int, p)) for p in T], "weight": w})
        out["meshes"][str(M)] = {"normals": [[int(x) for x in p] for p in need], "missing": [[int(x) for x in p] for p in missing], "weights": rows, "weight_sum": wsum,
                                 "eta_sphere": (tot if not missing else None)}
        L += ["", "### Mesh %d: %d triangles, %d normals%s" % (M, len(Ts), len(need), (", missing %s" % missing) if missing else ""), "",
              "| triangle (vertices) | weight 12 area/pi |", "|---|---|"]
        for r in rows:
            L.append("| %s | %.10f |" % (r["triangle"], r["weight"]))
        L.append("")
        L.append("weight sum %.15f; eta_sphere(mesh %d) = %s" % (wsum, M, ("%.8f" % tot) if not missing else "-"))
    L += ["", "Against: J1's mesh-2 eta_sphere(0) 0.9010503 (obs_sphere_perray_v1, the per-ray inner form on version 6's chain values) and the paper's 0.9116.",
          "", "- every number above is printed by the script."]
    json.dump(out, open(a.out_json, "w"), indent=1, default=float)
    open(a.md, "w").write("\n".join(L) + "\n")
    print("\n".join(L[:20]))


if __name__ == "__main__":
    main()
