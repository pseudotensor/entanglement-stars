#!/usr/bin/env python
"""obs_innerform_control_v1.py (builder analyst; SPEC_final4b_observables.md sec. 1, J1's inner-form control C4).

The per-ray inner form of strandS_sphere_v6.py evaluates the inner disc's massive packets (gap below g_lim = 0.05) as
S(G) = -(2/3) log G + b + gamma G^2 log G + d G^2 + e G^4 log G + z G^4 at G = sqrt(m^2 + g0(q)^2), with each ray's coefficients fitted on
that ray's massless ladder.  That assumes the chain entropy depends on (q, m) only through G along the ray.  This control computes the
chain entropy directly at 8 disc points per normal (drawn with a fixed seed from the run's own inner quadrature nodes, the "saved ray
grids") at m = 0.05 sqrt(1/2) and m = 0.05, and compares with the ray's inner form (and with the single phi = 0 form beside it).

Chain length (builder decision, labelled): the ladder that fixes the inner form keeps L G = mu_L (L = ceil(mu_L'/G), even, uncapped), so
the like-for-like direct value is at that L ("L_rule"); the spec's fixed L = 1600 equals L_rule at G = g_lim and is computed beside it, so the
finite-size part of any difference is visible separately.  The implied bound on the coefficient: max |S_direct - S_form| times the inner
disc's total quadrature weight at that mass, over eta_p(m) from the version-6 rows (the expected size is at or below 1e-5).

Args: --v6dir (the version-6 JSONs sphere_<p>_v6base.json) --rundir (the base runs: their JSON and _points.npz, as named in the v6
"sources") --normals (comma list, default all six) --npts 8 --seed 20261015 --workers --out-json --md."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse
import json
import math
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "strandS"))
import strandS_sphere_v5 as SV  # noqa: E402
import strandS_sphere_v6 as V6  # noqa: E402

MASSES = (round(0.05 * math.sqrt(0.5), 15), 0.05)
NORMALS = ("100", "110", "111", "210", "211", "221")


def L_rule(G):
    L = max(64, int(math.ceil(SV.G["muLp"] / G)))
    return L + (L % 2)


def job(args):
    key, q, m, L = args
    t = time.time()
    return key, float(SV.S_channel(q, L, m)), time.time() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v6dir", required=True)
    ap.add_argument("--rundir", required=True)
    ap.add_argument("--normals", default=",".join(NORMALS))
    ap.add_argument("--npts", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20261015)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--md", required=True)
    a = ap.parse_args()
    t0 = time.time()
    out = {"script": "obs_innerform_control_v1.py", "masses": MASSES, "seed": a.seed, "npts": a.npts, "normals": {}}
    for ino, ps in enumerate(a.normals.split(",")):
        V = json.load(open(os.path.join(a.v6dir, "sphere_%s_v6base.json" % ps)))
        src = V["sources"][0]
        jpath = os.path.join(a.rundir, src)
        npath = jpath.replace(".json", "_points.npz")
        D, m0, aMs, massive, allm, outer, disc = V6.load_run(jpath, npath)
        p = tuple(int(v) for v in m0["normal"].split(","))
        prule = bool(m0.get("pnorm_rule", False))
        SV.init(p, m0["muL"], m0["Lcap"], prule)
        g_lim = SV.G["muLp"] / m0["Lcap"]
        assert abs(g_lim - V["g_lim"]) < 1e-12
        rays = V6.build_rays(m0, g_lim)
        pref = 8.0 / (2 * math.pi) ** 2 / SV.G["norm"]
        fits = V["ray_fits"]
        c0 = D["inner_fits"]["%.6f" % 0.0]["coef"]
        inner = []
        wdisc = 0.0
        for ir, ray in enumerate(rays):
            fr = min(fits, key=lambda f_: abs(f_["phi"] - ray["phi"]))
            assert abs(fr["phi"] - ray["phi"]) < 1e-9, "the version-6 rays differ from the rebuilt ones"
            for ip, (rr, wr, g0, rg) in enumerate(zip(ray["r"], ray["wr"], ray["g0"], ray["region"])):
                if rg == 0:
                    w = pref * ray["wphi"] * wr * rr
                    wdisc += w
                    inner.append({"ray": ir, "node": ip, "phi": ray["phi"], "rho": float(rr), "g0": float(g0), "w": float(w),
                                  "q": (float(rr * math.cos(ray["phi"])), float(rr * math.sin(ray["phi"]))), "coef": fr["coef"]})
        rng = np.random.default_rng(a.seed + ino)
        pick = [inner[i] for i in sorted(rng.choice(len(inner), size=min(a.npts, len(inner)), replace=False))]
        tasks = []
        for k, pt in enumerate(pick):
            for m in MASSES:
                G = math.sqrt(m * m + pt["g0"] ** 2)
                tasks.append(((k, m, "rule"), pt["q"], m, L_rule(G)))
                tasks.append(((k, m, "1600"), pt["q"], m, 1600))
        pool = Pool(a.workers, initializer=SV.init, initargs=(p, m0["muL"], m0["Lcap"], prule))
        res = {key: (S, dt) for key, S, dt in pool.map(job, tasks, chunksize=1)}
        pool.close()
        eta_m = {}
        for r in V["rows"]:
            for xv, e in zip(V6.SV.RECTS[r["rect"]], r["eta_per_packet"]):
                if xv > 0:
                    eta_m.setdefault(round(r["aM"] * math.sqrt(xv), 12), float(e))
        rows = []
        for k, pt in enumerate(pick):
            for m in MASSES:
                G = math.sqrt(m * m + pt["g0"] ** 2)
                Sf = float(SV.inner_S(G, pt["coef"]))
                Sf0 = float(SV.inner_S(G, c0))
                Sr, dtr = res[(k, m, "rule")]
                S16, dt16 = res[(k, m, "1600")]
                rows.append({"point": k, "ray": pt["ray"], "node": pt["node"], "phi": pt["phi"], "rho": pt["rho"], "g0": pt["g0"], "m": m, "G": G,
                             "L_rule": L_rule(G), "S_direct_rule": Sr, "S_direct_1600": S16, "S_form_perray": Sf, "S_form_phi0": Sf0,
                             "d_perray_rule": Sr - Sf, "d_phi0_rule": Sr - Sf0, "d_perray_1600": S16 - Sf, "seconds": dtr + dt16})
        summ = {}
        for m in MASSES:
            rr_ = [r for r in rows if r["m"] == m]
            em = eta_m.get(round(m, 12))
            dmax = max(abs(r["d_perray_rule"]) for r in rr_)
            summ[str(m)] = {"max_abs_d_perray_rule": dmax, "max_abs_d_phi0_rule": max(abs(r["d_phi0_rule"]) for r in rr_),
                            "max_abs_d_perray_1600": max(abs(r["d_perray_1600"]) for r in rr_), "eta_p": em,
                            "disc_weight": wdisc, "implied_rel_bound_perray": (dmax * wdisc / em) if em else None,
                            "implied_rel_bound_phi0": (max(abs(r["d_phi0_rule"]) for r in rr_) * wdisc / em) if em else None}
        out["normals"][ps] = {"normal": list(p), "g_lim": g_lim, "inner_nodes": len(inner), "rows": rows, "summary": summ}
        print("normal %s: %s (%.0f s)" % (ps, {m: "%.2e/%.2e" % (s["max_abs_d_perray_rule"], s["implied_rel_bound_perray"] or float("nan"))
                                                for m, s in summ.items()}, time.time() - t0), flush=True)
        json.dump(out, open(a.out_json, "w"), indent=1, default=float)
    out["seconds"] = time.time() - t0
    json.dump(out, open(a.out_json, "w"), indent=1, default=float)
    L = ["## J1's inner-form control (obs_innerform_control_v1.py): direct chain entropy at %d disc points per normal against the ray's inner form" % a.npts, "",
         "| normal | m | max abs(S_direct - S_form), per-ray form, L by the ladder's rule | same, the phi = 0 form | per-ray form, L = 1600 | eta_p(m) | disc weight | implied relative bound on eta_p (per-ray) | (phi = 0 form) |",
         "|---|---|---|---|---|---|---|---|---|"]
    for ps, N in out["normals"].items():
        for m, s in N["summary"].items():
            L.append("| %s | %.6g | %.3e | %.3e | %.3e | %.9g | %.4e | %.2e | %.2e |" % (
                ps, float(m), s["max_abs_d_perray_rule"], s["max_abs_d_phi0_rule"], s["max_abs_d_perray_1600"], s["eta_p"] or float("nan"),
                s["disc_weight"], s["implied_rel_bound_perray"] or float("nan"), s["implied_rel_bound_phi0"] or float("nan")))
    L.append("")
    L.append("- Every number above is read from the records by the script; the points are the run's own inner quadrature nodes drawn with seed %d; the expected size of the implied bound is at or below 1e-5 (the spec)." % a.seed)
    open(a.md, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
