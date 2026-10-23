#!/usr/bin/env python
"""ls_sizectl_read_v1.py (busi, inbox #468; ReviewA_found8_v1.md sec. 3, ReviewB_found8_v1.md sec. 3): the smallest-q size
control. Reads exact/ls_exact_v1_rate2_L<L>_j<j>.json for (L, j) = (32, 1), (64, 1), (96, 1), (128, 1) (the four-q set of v1.2),
(256, 2) (q = 2 pi/128 on twice the box) and (64, 2) (q = 2 pi/32 on twice the box, v1.1's size control). For each of the six
nonlinear rate coefficients it prints:
  (1) the size shifts c(2L, j = 2)/c(L, j = 1) - 1 at q = 2 pi/128 and at q = 2 pi/32;
  (2) the propagation through the free-intercept fit c0 + c2 q^2 + c4 q^4 (least squares on four q): c0 on the v1.2 set, with the
      q_min point replaced by its doubled-box value, and with both the q_min and q_max points replaced; the linear sensitivity
      d c0 / d c(q_min) times the shift; c0/c(q_min) in each case; whether c0 falls with the box;
  (3) separately (reviewer B), the higher-order fit sensitivity on each data set: c0 from [1, q^2, q^4] on four q, [1, q^2, q^4, q^6]
      exact on four q, [1, q^2, q^4] exact on the three smallest q, and [1, q^2] on the two smallest q; the spread over models.
Writes <internal>/ls_sizectl_read_v1.json. Post-processing, one thread."""
import os
os.environ["OMP_NUM_THREADS"] = "1"
import json   # noqa: E402

import numpy as np   # noqa: E402

D = "<internal>/localsea"
keys = [("n_udot", "coef_nprime_udot", "n' d_t u"), ("u_ndot", "coef_ndot_uprime", "d_t n u'"), ("u_ndot", "coef_u_ndotprime", "u d_t n'"),
        ("J_n_ndot", "coef_ndot_nprime", "J: d_t n n'"), ("J_u_udot", "coef_udot_uprime", "J: d_t u u'"), ("J_u_udot", "coef_u_udotprime", "J: u d_t u'")]


def load(L, j):
    r = json.load(open(os.path.join(D, "exact", "ls_exact_v1_rate2_L%d_j%d.json" % (L, j))))["rate2"][str(j)]
    return r["q"], np.array([r[a][b] for a, b, _ in keys])


base = [load(L, 1) for L in (32, 64, 96, 128)]
q = np.array([b[0] for b in base]); C = np.array([b[1] for b in base])
q256, c256 = load(256, 2); q64j2, c64j2 = load(64, 2)
assert abs(q256 - q[3]) < 1e-12 and abs(q64j2 - q[0]) < 1e-12
sets = {"v12": C.copy()}
s = C.copy(); s[3] = c256; sets["qmin_doubled"] = s
s = s.copy(); s[0] = c64j2; sets["qmin_and_qmax_doubled"] = s


def fit(cols, qq, y):
    A = np.stack([qq ** p for p in cols], 1)
    x, *_ = np.linalg.lstsq(A, y, rcond=None)
    return x, float(np.max(np.abs((y - A @ x) / y)))


models = {"free_1_q2_q4_4pt": ((0, 2, 4), slice(0, 4)), "exact_1_q2_q4_q6_4pt": ((0, 2, 4, 6), slice(0, 4)),
          "exact_1_q2_q4_3smallest": ((0, 2, 4), slice(1, 4)), "exact_1_q2_2smallest": ((0, 2), slice(2, 4))}
A4 = np.stack([q ** p for p in (0, 2, 4)], 1)
sens = np.linalg.pinv(A4)[0]          # d c0 / d c(q_i) for the free four-point fit
out = {"q": q.tolist(), "dc0_dc_i_free_fit": sens.tolist(), "coef": {}}
print("q:", " ".join("%.5f" % x for x in q), "| d c0/d c(q_i) of the free fit:", " ".join("%+.4f" % x for x in sens))
for i, (_, _, name) in enumerate(keys):
    o = {"c_L128_j1": float(C[3, i]), "c_L256_j2": float(c256[i]), "shift_qmin": float(c256[i] / C[3, i] - 1),
         "c_L32_j1": float(C[0, i]), "c_L64_j2": float(c64j2[i]), "shift_qmax": float(c64j2[i] / C[0, i] - 1), "sets": {}}
    for sl, Y in sets.items():
        y = Y[:, i]; d = {}
        for m, (cols, sel) in models.items():
            x, r = fit(cols, q[sel], y[sel])
            d[m] = {"c0": float(x[0]), "c0_over_c_qmin": float(x[0] / y[3]), "resid_rel": r}
        c0s = [d[m]["c0"] for m in models]
        d["model_spread_c0"] = float(max(c0s) - min(c0s)); d["model_maxabs_c0"] = float(max(abs(v) for v in c0s))
        o["sets"][sl] = d
    o["c0_change_linear_qmin"] = float(sens[3] * (c256[i] - C[3, i]))
    o["c0_change_linear_qmax"] = float(sens[0] * (c64j2[i] - C[0, i]))
    f = {sl: o["sets"][sl]["free_1_q2_q4_4pt"]["c0"] for sl in sets}
    o["free_c0_falls_with_box"] = bool(abs(f["qmin_doubled"]) < abs(f["v12"]))
    o["free_c0_falls_with_both"] = bool(abs(f["qmin_and_qmax_doubled"]) < abs(f["v12"]))
    out["coef"][name] = o
    print("%-12s shift q_min %+.4f%% q_max %+.4f%% | free c0: v12 %+.3e (%+.4f) -> q_min doubled %+.3e (%+.4f) -> both %+.3e (%+.4f) | linear dc0 %+.2e %+.2e" % (
        name, 100 * o["shift_qmin"], 100 * o["shift_qmax"], f["v12"], o["sets"]["v12"]["free_1_q2_q4_4pt"]["c0_over_c_qmin"], f["qmin_doubled"],
        o["sets"]["qmin_doubled"]["free_1_q2_q4_4pt"]["c0_over_c_qmin"], f["qmin_and_qmax_doubled"],
        o["sets"]["qmin_and_qmax_doubled"]["free_1_q2_q4_4pt"]["c0_over_c_qmin"], o["c0_change_linear_qmin"], o["c0_change_linear_qmax"]))
    for sl in sets:
        d = o["sets"][sl]
        print("      %-22s " % sl + " | ".join("%s c0 %+.3e (%+.4f) r %.1e" % (m, d[m]["c0"], d[m]["c0_over_c_qmin"], d[m]["resid_rel"]) for m in models)
              + " | spread %.2e max|c0| %.2e" % (d["model_spread_c0"], d["model_maxabs_c0"]))
with open(os.path.join(D, "ls_sizectl_read_v1.json"), "w") as fh:
    json.dump(out, fh, indent=1)
