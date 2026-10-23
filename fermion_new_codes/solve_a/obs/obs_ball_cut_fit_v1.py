#!/usr/bin/env python3
"""obs_ball_cut_fit_v1.py (builder analyst; S6 = J4 of REVIEWA_REVIEW_web40e): the fits of the ball-cut ladder from obs_ball_cut_v1.py's
run records (JSONL), CPU only, seconds.

Data point per (R, box ratio c): S_bar(R, c) = the mean of the eight registry entropies S(R + j/16, L = cR), j in {-7,-5,-3,-1,1,3,5,7}; the
fit functions are averaged over the same eight radii: 4 pi mean(R_j^2), mean(R_j), mean(ln R_j), 1 (mean(1/R_j) in one variant).
Box: per R, S_bar(c) = S_inf + b4 c^-4 over c = 4, 6, 8 (least squares, residual printed), and the exact three-point
S_inf + b4 c^-4 + b6 c^-6 as a variant; the diagnostic ratio (S(4) - S(8))/(S(6) - S(8)) against the pure c^-4 value.
Fits of eta (model 4 pi eta R^2 + c_1 R + c_log ln R + c_0 [+ c_inv/R]): on S_inf (both box variants) and on each box ratio alone;
c_log free and fixed at -44/45; all radii and without the smallest; nominal standard errors from the residuals where dof > 0 (model
residuals, not statistical errors).  The review's 0.9190 to 0.9193 is printed as a comparison only, never used.
Args: --runs (JSONL files) --check (JSONL of SVD spot checks to compare) --out-json --md."""
import argparse
import json
import math

import numpy as np

OFF = (-7, -5, -3, -1, 1, 3, 5, 7)
CLOG = -44.0 / 45.0


def load(files):
    recs = []
    for f in files:
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    recs.append(json.loads(line))
    return recs


def regressors(R, inv=False):
    Rj = R + np.array(OFF, float) / 16.0
    row = [4 * math.pi * float(np.mean(Rj ** 2)), float(np.mean(Rj)), float(np.mean(np.log(Rj))), 1.0]
    if inv:
        row.append(float(np.mean(1.0 / Rj)))
    return row


def fit(Rs, y, fixlog, inv=False):
    X = np.array([regressors(R, inv) for R in Rs])
    y = np.array(y, float)
    names = ["eta", "c1", "clog", "c0"] + (["cinv"] if inv else [])
    if fixlog:
        y = y - CLOG * X[:, 2]
        X = np.delete(X, 2, axis=1)
        names.remove("clog")
    if len(y) < X.shape[1]:
        return None
    sc = np.abs(X).max(0)
    cs, _, rank, sv = np.linalg.lstsq(X / sc, y, rcond=None)
    coef = cs / sc
    resid = y - X @ coef
    dof = len(y) - X.shape[1]
    out = dict(coef=dict(zip(names, map(float, coef))), dof=int(dof), max_abs_resid=float(np.abs(resid).max()),
               resid=[float(v) for v in resid], cond=float(sv[0] / sv[-1]), radii=[float(R) for R in Rs])
    if fixlog:
        out["coef"]["clog"] = CLOG
    if dof > 0:
        s2 = float(resid @ resid) / dof
        cov = s2 * np.linalg.inv(X.T @ X)
        out["nominal_se"] = dict(zip(names, map(float, np.sqrt(np.abs(np.diag(cov))))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--check", nargs="*", default=[])
    ap.add_argument("--review-lo", type=float, default=0.9190)
    ap.add_argument("--review-hi", type=float, default=0.9193)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--md", required=True)
    a = ap.parse_args()
    recs = load(a.runs)
    D = {}
    for r in recs:
        D.setdefault((float(r["R"]), float(r["box_ratio"])), {})[int(r["delta16"])] = r
    Rs = sorted({k[0] for k in D})
    Cs = sorted({k[1] for k in D})
    out = dict(script="obs_ball_cut_fit_v1.py", runs=a.runs, offsets16=list(OFF), clog_fixed=CLOG, review_comparison=[a.review_lo, a.review_hi])
    md = ["## S6 (J4): the genuine 3-dimensional ball cut, flat massless sea (obs_ball_cut_fit_v1.py on obs_ball_cut_v1.py's records)", "",
          "Site-centred balls |x| <= R + j/16, j in {%s} (eight registry offsets), antiperiodic boxes L = 4R, 6R, 8R, m 0; "
          "S = 4 sum_lambda d_lambda [2 sum_j s((1 + s_lambda_j)/2) + |m_lambda_e - m_lambda_o| ln 2]." % ", ".join("%+d" % j for j in OFF), ""]
    # the data table
    md += ["### Registry-averaged entropies", "",
           "| R | box | L | S_bar (mean of 8) | registry spread (max - min of S/4 pi R_j^2) | S_bar / 4 pi mean R_j^2 | solver |", "|---|---|---|---|---|---|---|"]
    Sbar = {}
    data_rows = []
    for R in Rs:
        for c in Cs:
            d = D.get((R, c), {})
            if sorted(d) != sorted(OFF):
                md.append("| %g | %g | - | incomplete (%d of 8) | | | |" % (R, c, len(d)))
                continue
            S = np.array([d[j]["S"] for j in OFF])
            Rj = R + np.array(OFF) / 16.0
            ratio = S / (4 * math.pi * Rj ** 2)
            Sb = float(S.mean())
            Sbar[(R, c)] = Sb
            L = d[OFF[0]]["L"]
            solver = sorted({d[j].get("solver", "?") for j in OFF})
            row = dict(R=R, box_ratio=c, L=L, S=[float(v) for v in S], S_bar=Sb, S_bar_over_4piR2=Sb / (4 * math.pi * float(np.mean(Rj ** 2))),
                       registry_spread=float(ratio.max() - ratio.min()), solver=solver,
                       frob_rel_max=float(max(abs(d[j]["frob"]["rel_diff"]) for j in OFF)),
                       n=[int(d[j]["n"]) for j in OFF])
            data_rows.append(row)
            md.append("| %g | %g | %d | %.10f | %.2e | %.6f | %s |" % (R, c, L, Sb, row["registry_spread"], row["S_bar_over_4piR2"], "/".join(solver)))
    out["data"] = data_rows
    md += ["", "Every record's sum_lambda d_lambda ||K_lambda||_F^2 equals the mask-correlation ||K||_F^2 to %.1e (largest relative difference)." % (
        max(r["frob_rel_max"] for r in data_rows) if data_rows else float("nan")), ""]
    # the box extrapolation
    md += ["### Box extrapolation per R", "",
           "| R | S_inf (c^-4, 3 boxes) | b4 | residual max | S_inf (c^-4 + c^-6, exact) | b4 | b6 | (S4 - S8)/(S6 - S8) (pure c^-4: %.4f) |" % (
               (4.0 ** -4 - 8.0 ** -4) / (6.0 ** -4 - 8.0 ** -4)), "|---|---|---|---|---|---|---|---|"]
    Sinf4, Sinf46, box_rows = {}, {}, []
    for R in Rs:
        if not all((R, c) in Sbar for c in (4.0, 6.0, 8.0)):
            continue
        cc = np.array([4.0, 6.0, 8.0])
        y = np.array([Sbar[(R, c)] for c in cc])
        X = np.column_stack([np.ones(3), cc ** -4])
        co, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        res = y - X @ co
        X3 = np.column_stack([np.ones(3), cc ** -4, cc ** -6])
        co3 = np.linalg.solve(X3, y)
        diag = (y[0] - y[2]) / (y[1] - y[2]) if y[1] != y[2] else float("nan")
        Sinf4[R], Sinf46[R] = float(co[0]), float(co3[0])
        box_rows.append(dict(R=R, S_inf_c4=float(co[0]), b4=float(co[1]), resid_max=float(np.abs(res).max()), S_inf_c46=float(co3[0]),
                             b4_c46=float(co3[1]), b6_c46=float(co3[2]), diag_ratio=float(diag), S_bar=[float(v) for v in y]))
        md.append("| %g | %.10f | %.4f | %.2e | %.10f | %.4f | %.4f | %.4f |" % (R, co[0], co[1], np.abs(res).max(), co3[0], co3[1], co3[2], diag))
    out["box"] = box_rows
    # fits
    variants = []

    def add(label, Rsel, ys, fixlog, inv=False):
        f = fit(Rsel, ys, fixlog, inv)
        if f is None:
            return
        f["label"] = label
        variants.append(f)

    for src, SS in (("S_inf (c^-4)", Sinf4), ("S_inf (c^-4 + c^-6)", Sinf46)):
        R_all = sorted(SS)
        if len(R_all) >= 3:
            for fixlog in (False, True):
                add("%s, all R, c_log %s" % (src, "fixed" if fixlog else "free"), R_all, [SS[R] for R in R_all], fixlog)
                add("%s, R >= %g, c_log %s" % (src, R_all[1], "fixed" if fixlog else "free"), R_all[1:], [SS[R] for R in R_all[1:]], fixlog)
            add("%s, all R, c_log fixed, + c_inv/R" % src, R_all, [SS[R] for R in R_all], True, True)
    for c in Cs:
        R_c = [R for R in Rs if (R, c) in Sbar]
        if len(R_c) >= 3:
            for fixlog in (False, True):
                add("box %gR only, all R, c_log %s" % (c, "fixed" if fixlog else "free"), R_c, [Sbar[(R, c)] for R in R_c], fixlog)
    out["fits"] = variants
    md += ["", "### Fits of eta (model 4 pi eta R^2 + c_1 R + c_log ln R + c_0, the regressors averaged over the eight registry radii)", "",
           "| data and variant | dof | eta | nominal se(eta) | c_1 | c_log | c_0 | c_inv | max abs residual |", "|---|---|---|---|---|---|---|---|---|"]
    for f in variants:
        co = f["coef"]
        md.append("| %s | %d | %.6f | %s | %.4f | %.4f | %.4f | %s | %.2e |" % (
            f["label"], f["dof"], co["eta"], ("%.1e" % f["nominal_se"]["eta"]) if "nominal_se" in f else "-", co["c1"], co["clog"], co["c0"],
            ("%.3f" % co["cinv"]) if "cinv" in co else "-", f["max_abs_resid"]))
    ex = [f["coef"]["eta"] for f in variants if f["label"].startswith("S_inf")]
    if ex:
        out["eta_envelope_S_inf_variants"] = [min(ex), max(ex)]
        md += ["", "Envelope of eta over the S_inf variants: %.6f to %.6f (half-width %.1e; target half-width 5e-4).  Review's interval "
               "%.4f to %.4f, a comparison only." % (min(ex), max(ex), (max(ex) - min(ex)) / 2, a.review_lo, a.review_hi)]
    # the solver spot checks
    if a.check:
        chk = load(a.check)
        md += ["", "### Solver spot checks (the SVD route against the production record at the same (R + j/16, L))", ""]
        out["spot_checks"] = []
        for r in chk:
            k = (float(r["R"]), float(r["box_ratio"]))
            p = D.get(k, {}).get(int(r["delta16"]))
            if p is None:
                continue
            diff = r["S"] - p["S"]
            out["spot_checks"].append(dict(R=r["R"], delta16=r["delta16"], L=r["L"], S_check=r["S"], solver_check=r["solver"], S_prod=p["S"],
                                           solver_prod=p["solver"], diff=diff))
            md.append("R %g%+d/16, L %d: %s %.10f, %s %.10f, difference %.2e" % (r["R"], r["delta16"], r["L"], r["solver"], r["S"], p["solver"], p["S"], diff))
    with open(a.out_json, "w") as fh:
        json.dump(out, fh, indent=1)
    with open(a.md, "w") as fh:
        fh.write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
