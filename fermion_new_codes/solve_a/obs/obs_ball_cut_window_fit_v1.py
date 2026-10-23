#!/usr/bin/env python3
"""obs_ball_cut_window_fit_v1.py (builder analyst; S6 = J4 registry diagnostic): fits of eta on obs_ball_cut_window_v1.py's exact
unit-window averages S_win(R) (one box L = cR; its finite-box shift is R-independent at leading order, c^-4 b4 with b4 about -33.5 from the
three-box ladder, and is absorbed in c_0), with the exact window means of the fit functions: 4 pi (R^2 + 1/12), R, mean ln R', 1;
c_log free and fixed at -44/45; all radii and without the smallest; and the same fits on the eight-midpoint means S_mid8 of the same
records (regressors averaged over the eight radii) for comparison.  Nominal standard errors from the residuals where dof > 0.
Args: --window (JSONL) --out-json --md."""
import argparse
import json
import math

import numpy as np

CLOG = -44.0 / 45.0
OFF = np.array([-7, -5, -3, -1, 1, 3, 5, 7], float) / 16.0


def fit(X, y, names, fixlog):
    X = np.array(X, float)
    y = np.array(y, float)
    names = list(names)
    if fixlog:
        y = y - CLOG * X[:, 2]
        X = np.delete(X, 2, axis=1)
        names.remove("clog")
    if len(y) < X.shape[1]:
        return None
    sc = np.abs(X).max(0)
    cs, _, _, sv = np.linalg.lstsq(X / sc, y, rcond=None)
    coef = cs / sc
    resid = y - X @ coef
    dof = len(y) - X.shape[1]
    out = dict(coef=dict(zip(names, map(float, coef))), dof=int(dof), resid=[float(v) for v in resid], max_abs_resid=float(np.abs(resid).max()))
    if fixlog:
        out["coef"]["clog"] = CLOG
    if dof > 0:
        cov = float(resid @ resid) / dof * np.linalg.inv(X.T @ X)
        out["nominal_se"] = dict(zip(names, map(float, np.sqrt(np.abs(np.diag(cov))))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", nargs="+", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--md", required=True)
    a = ap.parse_args()
    recs = {}
    for f in a.window:
        for line in open(f):
            if line.strip():
                r = json.loads(line)
                recs[float(r["R"])] = r
    Rs = sorted(recs)
    names = ["eta", "c1", "clog", "c0"]
    Xw = [[4 * math.pi * recs[R]["mean_R2"], recs[R]["mean_R"], recs[R]["mean_lnR"], 1.0] for R in Rs]
    yw = [recs[R]["S_win"] for R in Rs]
    X8 = [[4 * math.pi * float(np.mean((R + OFF) ** 2)), R, float(np.mean(np.log(R + OFF))), 1.0] for R in Rs]
    y8 = [recs[R]["S_mid8"] for R in Rs]
    md = ["## S6 (J4) fits on the exact unit-window averages (obs_ball_cut_window_fit_v1.py)", "",
          "Radii %s, box %s (one box; its finite-box shift is absorbed in c_0)." % (", ".join("%g" % R for R in Rs),
                                                                                   ", ".join("%dR" % round(recs[R]["box_ratio"]) for R in Rs[:1])), "",
          "| R | balls | S_win | S_win / 4 pi (R^2 + 1/12) | S_mid8 - S_win |", "|---|---|---|---|---|"]
    for R in Rs:
        r = recs[R]
        md.append("| %g | %d | %.10f | %.6f | %.3e |" % (R, len(r["segments"]), r["S_win"], r["S_win"] / (4 * math.pi * r["mean_R2"]), r["S_mid8"] - r["S_win"]))
    md += ["", "| data | radii | c_log | dof | eta | nominal se(eta) | c_1 | c_log | c_0 | max abs residual |", "|---|---|---|---|---|---|---|---|---|---|"]
    fits = []
    for lab, X, y in (("S_win (exact window)", Xw, yw), ("S_mid8 (eight midpoints)", X8, y8)):
        for start in (0, 1):
            for fixlog in (False, True):
                f = fit(X[start:], y[start:], names, fixlog)
                if f is None:
                    continue
                f.update(data=lab, radii=Rs[start:], fixlog=fixlog)
                fits.append(f)
                co = f["coef"]
                md.append("| %s | %g to %g | %s | %d | %.6f | %s | %.4f | %.4f | %.4f | %.2e |" % (
                    lab, Rs[start], Rs[-1], "fixed" if fixlog else "free", f["dof"], co["eta"],
                    ("%.1e" % f["nominal_se"]["eta"]) if "nominal_se" in f else "-", co["c1"], co["clog"], co["c0"], f["max_abs_resid"]))
    with open(a.out_json, "w") as fh:
        json.dump(dict(script="obs_ball_cut_window_fit_v1.py", radii=Rs, fits=fits), fh, indent=1)
    with open(a.md, "w") as fh:
        fh.write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
