#!/usr/bin/env python
"""ls_rate3q_read_v2.py (busi, inbox #455; v1 with the L = 128 point and its own output name; v1.1 sec. 4 (4) and sec. 8 item 3; ReviewA_found7_v1.md sec. 1.1, ReviewB_found7_v1.md
sec. 1 D): the nonlinear rate coefficients at three (or four) wavevectors, q = 2 pi/L with L = 32, 64, 96 (and 128 if present),
all j = 1, 40 transverse points per axis (exact/ls_exact_v1_rate2_L<L>_j1.json). For each of the six coefficients:
  - c(q)/q^2 at every q (an intercept-free reading predicts it rising toward the smallest q; an intercept predicts it growing like 1/q^2);
  - the unrestricted fit c0 + c2 q^2 + c4 q^4 (exact on three points; least squares with residual on four);
  - the restricted two-point fits on the two smallest q: c0 + c2 q^2, and c2 q^2 + c4 q^4;
  - the restricted three-point fits c0 + c2 q^2 (least squares, residual) and c2 q^2 + c4 q^4 (residual): the reading that fits is
    the one whose residual is at the size-control level (0.5 to 2 percent, v1.1).
Writes <out>/ls_rate3q_read_v2.json. Post-processing, one thread."""
import os
os.environ["OMP_NUM_THREADS"] = "1"
import json   # noqa: E402

import numpy as np   # noqa: E402

D = "<internal>/localsea"
keys = [("n_udot", "coef_nprime_udot", "n' d_t u"), ("u_ndot", "coef_ndot_uprime", "d_t n u'"), ("u_ndot", "coef_u_ndotprime", "u d_t n'"),
        ("J_n_ndot", "coef_ndot_nprime", "J: d_t n n'"), ("J_u_udot", "coef_udot_uprime", "J: d_t u u'"), ("J_u_udot", "coef_u_udotprime", "J: u d_t u'")]
Ls = [L for L in (32, 64, 96, 128) if os.path.exists(os.path.join(D, "exact", "ls_exact_v1_rate2_L%d_j1.json" % L))]
q = []; C = []
for L in Ls:
    r = json.load(open(os.path.join(D, "exact", "ls_exact_v1_rate2_L%d_j1.json" % L)))["rate2"]["1"]
    q.append(r["q"]); C.append([r[a][b] for a, b, _ in keys])
q = np.array(q); C = np.array(C)


def lsq(cols, qq, y):
    A = np.stack([c(qq) for c in cols], 1)
    x, *_ = np.linalg.lstsq(A, y, rcond=None)
    res = y - A @ x
    return x, float(np.max(np.abs(res / y)))


one = lambda x: np.ones_like(x)   # noqa: E731
q2 = lambda x: x ** 2   # noqa: E731
q4 = lambda x: x ** 4   # noqa: E731
out = {"L": Ls, "q": q.tolist(), "coef": {}}
print("L:", Ls, "q:", " ".join("%.5f" % x for x in q))
for i, (_, _, name) in enumerate(keys):
    y = C[:, i]
    o = {"c": y.tolist(), "c_over_q2": (y / q ** 2).tolist(), "halving_like_ratios": (y[:-1] / y[1:]).tolist()}
    if len(q) >= 3:
        x, r = lsq([one, q2, q4], q, y); o["full_c0_c2_c4"] = x.tolist(); o["full_resid_rel"] = r
        o["full_c0_over_c_smallest_q"] = float(x[0] / y[-1])
        x, r = lsq([one, q2], q, y); o["c0_c2_3pt"] = x.tolist(); o["c0_c2_3pt_resid_rel"] = r
        x, r = lsq([q2, q4], q, y); o["c2_c4_3pt"] = x.tolist(); o["c2_c4_3pt_resid_rel"] = r
    s = slice(len(q) - 2, len(q))
    x, _ = lsq([one, q2], q[s], y[s]); o["c0_c2_two_smallest"] = x.tolist(); o["c0_over_c_smallest_q_two_smallest"] = float(x[0] / y[-1])
    x, _ = lsq([q2, q4], q[s], y[s]); o["c2_c4_two_smallest"] = x.tolist()
    out["coef"][name] = o
    print("%-12s c %s | c/q^2 %s | full c0 %.2e (%.3f of c at smallest q) resid %.1e | c0+c2q2 resid %.3f | c2q2+c4q4 resid %.3f | two-smallest c0 %.2e (%.3f)" % (
        name, " ".join("%.4e" % v for v in y), " ".join("%.5f" % v for v in o["c_over_q2"]),
        o.get("full_c0_c2_c4", [np.nan])[0], o.get("full_c0_over_c_smallest_q", np.nan), o.get("full_resid_rel", np.nan),
        o.get("c0_c2_3pt_resid_rel", np.nan), o.get("c2_c4_3pt_resid_rel", np.nan), o["c0_c2_two_smallest"][0],
        o["c0_over_c_smallest_q_two_smallest"]))
with open(os.path.join(D, "ls_rate3q_read_v2.json"), "w") as f:
    json.dump(out, f, indent=1)
