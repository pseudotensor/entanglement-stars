#!/usr/bin/env python
"""Strand S job S3's report, version 2 (the lead's judgment sec. 8.5(6); builder analyst): the spherical cut as the paper's mesh-2 solid-angle
average of the per-normal graded coefficients, with the spacings 0.025 and 0.0125 of strandS_sphere_v5.py merged into strandS_sphere_v3.py's
base rows (aM 0.4, 0.2, 0.1, 0.05) per normal.  Per normal and rectangle: c_S,p(aM) at every spacing; the three-range limits of X_0 + c (aM)^2
log(1/aM) + c' (aM)^2 on aM <= 0.2, <= 0.1 and <= 0.05 with their spread (where the merged spacings allow: at least 3 points per range), and the
labelled 2-term form X_0 + c (aM)^2 log(1/aM) on aM <= 0.2 beside them; the sphere average at every spacing present for all normals (weights
w_p from sphere.triangles(2), the paper's sphere.py) with the same fits for c_S and R_gr (hat k_0 from S2's rows: a bulk quantity); the
controls: the matching radius g_lim (strandS_sphere_v5.py --Lcap; 0.03 and 0.08 against the base 0.05), the |p| finite-interval rule, the
direct massive-channel check at fixed transverse momentum, the cap-1024 control (strandS_sphere_v4.py at (2,1,0)), and the [100] cross-check
against S2.  Args: --base (comma list of S3v3 JSONs) --ext (comma list of S3v5 *_ext JSONs; optional) --variants (comma list; optional)
--cap-ctl (optional) --s2-json --orient --json --md."""
import argparse
import json
import math
import sys

import numpy as np

THIRD_PI = 1 / (3 * math.pi)


def f(v, d=6):
    if v is None:
        return "-"
    v = float(v)
    return "%.*g" % (d, v) if (v == 0 or 1e-3 <= abs(v) < 1e5) else "%.4e" % v


def fit(a, v, fmax, three=True):
    a, v = np.array(a, float), np.array(v, float)
    sel = a <= fmax * (1 + 1e-12)
    cols = [np.ones(sel.sum()), a[sel] ** 2 * np.log(1 / a[sel])] + ([a[sel] ** 2] if three else [])
    X = np.stack(cols, 1)
    if X.shape[0] < X.shape[1] or X.shape[0] < 2:
        return None
    if three and X.shape[0] < 3:
        return None
    c, *_ = np.linalg.lstsq(X, v[sel], rcond=None)
    return float(c[0])


def limits(aa, cc):
    three = {str(fm): fit(aa, cc, fm) for fm in (0.2, 0.1, 0.05)}
    ok = [x for x in three.values() if x is not None]
    two = fit(aa, cc, 0.2, three=False)
    return {"three_range": three, "three_mean": float(np.mean(ok)) if ok else None, "three_spread": float(max(ok) - min(ok)) if len(ok) > 1 else None,
            "n_ranges": len(ok), "two_term_le_0.2": two}


def merged_rows(base, ext):
    rows = {(r["rect"], r["aM"]): r for r in base["rows"]}
    if ext:
        for r in ext["rows"]:
            rows.setdefault((r["rect"], r["aM"]), r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--ext", default="")
    ap.add_argument("--variants", default="")
    ap.add_argument("--cap-ctl", default="")
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
    base = {}
    for p in [x for x in a.base.split(",") if x]:
        d = json.load(open(p))
        base[tuple(d["normal"])] = d
    ext = {}
    for p in [x for x in a.ext.split(",") if x]:
        d = json.load(open(p))
        ext[tuple(d["normal"])] = d
    out = {"weights": {str(k): v for k, v in w.items()}, "per_normal": {}, "sphere": {}}
    L = ["# Strand S job S3: the spherical cut, report version 2 (strandS_sphere_report_v2.py; base spacings from strandS_sphere_v3.py, the "
         "spacings 0.025 and 0.0125 from strandS_sphere_v5.py)", "",
         "- Mesh-2 weights: " + "; ".join("%s %s" % (k, f(v, 8)) for k, v in sorted(w.items())) + ".",
         "- Normals with base rows: %d of %d; with the extra spacings: %d." % (len(base), len(w), len(ext)), "",
         "## Per normal: c_S,p(aM) (the checks per run in the records)", "", "| normal | weight | rectangle | aM | hat S_gr,p | c_S,p | inner share | source |",
         "|---|---|---|---|---|---|---|---|"]
    merged = {}
    for nrm in sorted(base):
        rows = merged_rows(base[nrm], ext.get(nrm))
        merged[nrm] = rows
        for (rect, aM), r in sorted(rows.items()):
            src = "v5" if nrm in ext and any(x["rect"] == rect and x["aM"] == aM for x in ext[nrm]["rows"]) and not any(
                x["rect"] == rect and x["aM"] == aM for x in base[nrm]["rows"]) else "v3"
            L.append("| %s | %s | %s | %s | %s | %s | %s | %s |" % (nrm, f(w.get(nrm), 6), rect, f(aM), f(r["S_gr"], 8), f(r["c_S"], 9), f(r["inner_share"], 3), src))
        per = {"eta0": base[nrm]["eta_massless"], "checks": base[nrm]["checks"], "ext_checks": ext[nrm]["checks"] if nrm in ext else None, "limits": {}}
        for rect in ("light", "r1223"):
            rr = sorted([r for (rc, _), r in rows.items() if rc == rect], key=lambda r: r["aM"])
            per["limits"][rect] = limits([r["aM"] for r in rr], [r["c_S"] for r in rr])
            per["limits"][rect]["spacings"] = [r["aM"] for r in rr]
        out["per_normal"][str(nrm)] = per
    L += ["", "## Per-normal limits of c_S,p (1/(3 pi) = %s)" % f(THIRD_PI, 10), "",
          "| normal | rectangle | spacings | three-range limits (aM <= 0.2, 0.1, 0.05) | mean | spread | mean/(1/(3 pi)) - 1 | 2-term limit on aM <= 0.2 (labelled) | "
          "2-term/(1/(3 pi)) - 1 |", "|---|---|---|---|---|---|---|---|---|"]
    for nrm, per in sorted(out["per_normal"].items()):
        for rect, e in per["limits"].items():
            L.append("| %s | %s | %d | %s | %s | %s | %s | %s | %s |" % (
                nrm, rect, len(e["spacings"]), ", ".join(f(x, 9) for x in e["three_range"].values()), f(e["three_mean"], 9), f(e["three_spread"], 2),
                f(e["three_mean"] / THIRD_PI - 1, 3) if e["three_mean"] else "-", f(e["two_term_le_0.2"], 9),
                f(e["two_term_le_0.2"] / THIRD_PI - 1, 3) if e["two_term_le_0.2"] else "-"))
    if set(base) == set(w):
        common = None
        for nrm in w:
            ks = set(merged[nrm].keys())
            common = ks if common is None else common & ks
        rows = []
        for rect in ("light", "r1223"):
            for aM in sorted({k[1] for k in common if k[0] == rect}):
                Sg = sum(w[n] * merged[n][(rect, aM)]["S_gr"] for n in w)
                j = merged[(1, 0, 0)][(rect, aM)]["j"]
                cS = Sg / (aM * aM * j)
                Rg = Sg / (0.5 * math.pi * k0[(rect, aM)]) if (rect, aM) in k0 else None
                rows.append({"rect": rect, "aM": aM, "S_gr": Sg, "c_S": cS, "R_gr": Rg})
        out["sphere"]["rows"] = rows
        L += ["", "## The sphere (mesh-2 average over the six normals)", "", "| rectangle | aM | hat S_gr,sphere | c_S,sphere | R_gr,sphere | [100] (S2) c_S, R_gr |",
              "|---|---|---|---|---|---|"]
        for r in rows:
            s2 = s2r.get((r["rect"], r["aM"]), {})
            L.append("| %s | %s | %s | %s | %s | %s, %s |" % (r["rect"], f(r["aM"]), f(r["S_gr"], 8), f(r["c_S"], 9), f(r["R_gr"], 9), f(s2.get("c_S"), 9), f(s2.get("R_gr"), 9)))
        for rect in ("light", "r1223"):
            rr = sorted([r for r in rows if r["rect"] == rect], key=lambda r: r["aM"])
            for q in ("c_S", "R_gr"):
                if any(r[q] is None for r in rr):
                    continue
                out["sphere"]["%s_%s" % (rect, q)] = limits([r["aM"] for r in rr], [r[q] for r in rr])
        eta0 = sum(w[n] * base[n]["eta_massless"] for n in w)
        out["sphere"]["eta0"] = eta0
        L += ["", "- The sphere's limits: " + "; ".join("%s: three-range %s (mean %s, spread %s), 2-term %s" % (
            k, ", ".join(f(x, 9) for x in v["three_range"].values()), f(v["three_mean"], 9), f(v["three_spread"], 2), f(v["two_term_le_0.2"], 9))
            for k, v in out["sphere"].items() if isinstance(v, dict) and "three_range" in v) + " (1/(3 pi) = %s, 2/9 = %s)." % (f(THIRD_PI, 9), f(2 / 9, 9)),
              "- The massless average eta_sphere(0) = %s against strand 3's mesh-2 value 0.9001144 (difference %s)." % (f(eta0, 10), f(eta0 - 0.9001144, 3))]
    if (1, 0, 0) in base:
        dif = max(abs(r["S_gr"] / s2r[(r["rect"], r["aM"])]["S_gr"] - 1) for r in base[(1, 0, 0)]["rows"] if (r["rect"], r["aM"]) in s2r)
        out["cross_check_100"] = dif
        L += ["- The [100] cross-check against S2: the largest |hat S_gr (sphere code)/hat S_gr (S2) - 1| %s." % f(dif, 3)]
    ctl = []
    for p in [x for x in a.variants.split(",") if x]:
        d = json.load(open(p))
        nrm = tuple(d["normal"])
        m = d["meta"]
        lab = "g_lim %s%s" % (f(d["g_lim"], 4), ", |p| rule" if m.get("pnorm_rule") else "")
        lim = {}
        for rect in ("light", "r1223"):
            rr = sorted([r for r in d["rows"] if r["rect"] == rect], key=lambda r: r["aM"])
            lim[rect] = limits([r["aM"] for r in rr], [r["c_S"] for r in rr])
        rec = {"normal": list(nrm), "label": lab, "limits": lim, "rows": d["rows"]}
        if "massive_check" in d:
            mc = d["massive_check"]
            rec["massive_check_max"] = max(abs(x["difference"]) for x in mc)
            rec["massive_check_n"] = len(mc)
        ctl.append(rec)
    for nrm, d in ext.items():
        if "massive_check" in d:
            mc = d["massive_check"]
            ctl.append({"normal": list(nrm), "label": "massive check (base rule)", "massive_check_max": max(abs(x["difference"]) for x in mc),
                        "massive_check_n": len(mc), "massive_check": mc})
    if a.cap_ctl:
        d = json.load(open(a.cap_ctl))
        nrm = tuple(d["normal"])
        diffs = []
        for r in d["rows"]:
            b = merged.get(nrm, {}).get((r["rect"], r["aM"]))
            if b:
                diffs.append({"rect": r["rect"], "aM": r["aM"], "c_S_cap1024": r["c_S"], "c_S_base": b["c_S"], "relative": r["c_S"] / b["c_S"] - 1})
        ctl.append({"normal": list(nrm), "label": "cap 1024 (strandS_sphere_v4.py)", "rows_vs_base": diffs})
    out["controls"] = ctl
    if ctl:
        L += ["", "## Controls", ""]
        for c in ctl:
            s = "- %s at %s:" % (c["label"], tuple(c["normal"]))
            if "limits" in c:
                s += " " + "; ".join("%s three-range mean %s (spread %s), 2-term %s" % (rect, f(e["three_mean"], 9), f(e["three_spread"], 2), f(e["two_term_le_0.2"], 9))
                                     for rect, e in c["limits"].items())
            if "massive_check_max" in c:
                s += " the direct massive-channel values against the inner form: largest |difference| %s over %d points" % (f(c["massive_check_max"], 3), c["massive_check_n"])
            if "rows_vs_base" in c:
                s += " " + "; ".join("%s aM %s: %s" % (x["rect"], f(x["aM"]), f(x["relative"], 3)) for x in c["rows_vs_base"])
            L.append(s + ".")
    L += ["", "- Every number above is emitted by the script from the records."]
    json.dump(out, open(a.json, "w"), indent=1, default=float)
    open(a.md, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
