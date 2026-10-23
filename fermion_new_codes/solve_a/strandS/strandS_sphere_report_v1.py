#!/usr/bin/env python
"""Strand S job S3's report (SPEC sec. 4 and 9.5; builder analyst): the spherical cut as the paper's mesh-2 solid-angle average of the
per-normal graded coefficients (strandS_sphere_v3.py's JSONs).  The weights w_p = sum over sphere.triangles(2) of (12 area/pi)/3 per vertex,
canonical(p) (the paper's sphere.py and lattice_entropy.canonical, unmodified); per normal and (rectangle, aM): eta_p(q_s), hat S_gr,p, c_S,p,
the inner-disc share, the checks (corners, inversion, the two gaps), the ladder fits; the sphere averages c_S,sphere and R_gr,sphere = hat S_gr,
sphere/((pi/2) hat k_0) with hat k_0 from S2's rows at the same rectangle and spacing (hat k_0 is a bulk quantity, the same for every cut); the
massless average eta_sphere(0) against strand 3's 0.9001144 (the uniform-grid value, expected below by about 4e-4); the [100] cross-check
against S2; per normal the limits of c_S,p by the fits X_0 + c (aM)^2 log(1/aM) + c' (aM)^2 on aM <= 0.4 and <= 0.2, and X_0 + c (aM)^2 log(1/aM)
on aM <= 0.1 (two points; BUILDER: the four spacings allow no three ranges of the three-coefficient form), the spread as the error.
Args: --jsons (comma list) --s2-json --orient (the paper's module folder) --json --md."""
import argparse
import json
import math
import sys

import numpy as np


def f(v, d=6):
    if v is None:
        return "-"
    v = float(v)
    return "%.*g" % (d, v) if (v == 0 or 1e-3 <= abs(v) < 1e5) else "%.4e" % v


def fit(a, v, fmax, three=True):
    a = np.array(a)
    v = np.array(v)
    sel = a <= fmax * (1 + 1e-12)
    cols = [np.ones(sel.sum()), a[sel] ** 2 * np.log(1 / a[sel])] + ([a[sel] ** 2] if three else [])
    X = np.stack(cols, 1)
    if X.shape[0] < X.shape[1]:
        return None
    c, *_ = np.linalg.lstsq(X, v[sel], rcond=None)
    return float(c[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsons", required=True)
    ap.add_argument("--s2-json", required=True)
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
    S2 = json.load(open(a.s2_json))
    k0 = {(r["rect"], r["aM"]): r["k0_route1"] for r in S2["rows"]}
    s2r = {(r["rect"], r["aM"]): r for r in S2["rows"]}
    runs = {}
    for p in a.jsons.split(","):
        d = json.load(open(p))
        runs[tuple(d["normal"])] = d
    out = {"weights": {str(k): v for k, v in w.items()}, "normals_present": [list(k) for k in runs], "per_normal": {}, "sphere": {}}
    L = ["# Strand S job S3: the spherical cut (strandS_sphere_report_v1.py; the mesh-2 average of strandS_sphere_v3.py's per-normal records)", "",
         "- Mesh-2 weights (sphere.py's triangles(2), canonical normals): " + "; ".join("%s %s" % (k, f(v, 8)) for k, v in sorted(w.items())) + ".",
         "- Normals present: %d of %d." % (len(runs), len(w)), "", "## Per normal", "",
         "| normal | weight | corners spread | inversion | two gaps (rel.) | ladder fit residual (phi 0, pi/4) | eta_p(0) | rectangle | aM | hat S_gr,p | c_S,p | inner share |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for nrm, d in sorted(runs.items()):
        ck = d["checks"]
        fits = d["inner_fits"]
        per = {"eta0": d["eta_massless"], "rows": d["rows"], "checks": ck}
        for r in d["rows"]:
            L.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                nrm, f(w.get(nrm), 6), f(ck["corner_spread_max"], 2), f(ck["inversion_max"], 2), f(ck.get("two_gap_rel_max"), 2),
                ", ".join(f(v["max_resid"], 2) for v in fits.values()), f(d["eta_massless"], 10), r["rect"], f(r["aM"]), f(r["S_gr"], 8), f(r["c_S"], 8),
                f(r["inner_share"], 3)))
        lim = {}
        for rect in ("light", "r1223"):
            rr = sorted([r for r in d["rows"] if r["rect"] == rect], key=lambda r: r["aM"])
            aa = [r["aM"] for r in rr]
            cc = [r["c_S"] for r in rr]
            ls = [fit(aa, cc, 0.4), fit(aa, cc, 0.2), fit(aa, cc, 0.1, three=False)]
            ls = [x for x in ls if x is not None]
            lim[rect] = {"limits": ls, "mean": float(np.mean(ls)) if ls else None, "spread": float(max(ls) - min(ls)) if ls else None}
        per["extrapolation"] = lim
        out["per_normal"][str(nrm)] = per
    L += ["", "## Per-normal extrapolations of c_S,p (three fits; 1/(3 pi) = %s)" % f(1 / (3 * math.pi), 10), "",
          "| normal | rectangle | limits | mean | spread | mean/(1/(3 pi)) - 1 |", "|---|---|---|---|---|---|"]
    for nrm, per in sorted(out["per_normal"].items()):
        for rect, e in per["extrapolation"].items():
            L.append("| %s | %s | %s | %s | %s | %s |" % (nrm, rect, ", ".join(f(x, 9) for x in e["limits"]), f(e["mean"], 9), f(e["spread"], 2),
                                                         f(e["mean"] * 3 * math.pi - 1, 3) if e["mean"] else "-"))
    # the sphere average (only if all normals are present)
    if set(runs) == set(w):
        L += ["", "## The sphere (mesh-2 average)", "", "| rectangle | aM | hat S_gr,sphere | c_S,sphere | R_gr,sphere | the [100] value (S2) |", "|---|---|---|---|---|---|"]
        eta0 = sum(w[n] * runs[n]["eta_massless"] for n in w)
        out["sphere"]["eta0"] = eta0
        rows = []
        for rect in ("light", "r1223"):
            aMs = sorted({r["aM"] for r in runs[(1, 0, 0)]["rows"] if r["rect"] == rect})
            for aM in aMs:
                Sg = sum(w[n] * [r for r in runs[n]["rows"] if r["rect"] == rect and r["aM"] == aM][0]["S_gr"] for n in w)
                j = [r for r in runs[(1, 0, 0)]["rows"] if r["rect"] == rect][0]["j"]
                cS = Sg / (aM * aM * j)
                Rg = Sg / (0.5 * math.pi * k0[(rect, aM)])
                rows.append({"rect": rect, "aM": aM, "S_gr": Sg, "c_S": cS, "R_gr": Rg})
                L.append("| %s | %s | %s | %s | %s | c_S %s, R_gr %s |" % (rect, f(aM), f(Sg, 8), f(cS, 8), f(Rg, 8), f(s2r[(rect, aM)]["c_S"], 8), f(s2r[(rect, aM)]["R_gr"], 8)))
        out["sphere"]["rows"] = rows
        for rect in ("light", "r1223"):
            rr = sorted([r for r in rows if r["rect"] == rect], key=lambda r: r["aM"])
            aa = [r["aM"] for r in rr]
            for q in ("c_S", "R_gr"):
                ls = [fit(aa, [r[q] for r in rr], 0.4), fit(aa, [r[q] for r in rr], 0.2), fit(aa, [r[q] for r in rr], 0.1, three=False)]
                ls = [x for x in ls if x is not None]
                out["sphere"]["%s_%s" % (rect, q)] = {"limits": ls, "mean": float(np.mean(ls)), "spread": float(max(ls) - min(ls))}
        L += ["", "- The massless average eta_sphere(0) = %s against strand 3's mesh-2 value 0.9001144 (difference %s)." % (f(eta0, 10), f(eta0 - 0.9001144, 3)),
              "- The sphere's extrapolated limits (three fits): " + "; ".join("%s %s: %s (spread %s)" % (k.split("_")[0], k.split("_", 1)[1], f(v["mean"], 9), f(v["spread"], 2))
                                                                          for k, v in out["sphere"].items() if isinstance(v, dict) and "mean" in v) +
              " (1/(3 pi) = %s, 2/9 = %s)." % (f(1 / (3 * math.pi), 9), f(2 / 9, 9))]
    if (1, 0, 0) in runs:
        dif = max(abs(r["S_gr"] / s2r[(r["rect"], r["aM"])]["S_gr"] - 1) for r in runs[(1, 0, 0)]["rows"])
        L += ["", "- The [100] cross-check against S2: the largest |hat S_gr (sphere code)/hat S_gr (S2) - 1| over the rows %s; eta_100(0) %s against S2's %s." % (
            f(dif, 3), f(runs[(1, 0, 0)]["eta_massless"], 10), f(S2["C1"]["eta100_0"], 10))]
        out["cross_check_100"] = dif
    json.dump(out, open(a.json, "w"), indent=1, default=float)
    open(a.md, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
