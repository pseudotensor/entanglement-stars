#!/usr/bin/env python
"""ls_zone_v1.py (busi, inbox #447; ReviewA_found4_v1.md sec. 3 items 1-2 and sec. 1.5, 2.3; ReviewB_found4_v1.md sec. 5.1).

DATA script, box CPU only (torch float64, at most 8 threads). The cubic lattice Dirac sea of the paper of record
(fermion_new/03_model.tex Eqs. dm-fermion, dm-current, dm-hamiltonian; a = 1): h0(p) = m beta + sum_i alpha_i sin p_i,
Q_i(p) = sin(2 p_i)/2 (spin scalar), e_x = {Pi_x, h0}/2, p_x = {Pi_x, Q}/2, h[N, v] = {N, h0}/2 + sum_i {v_i, Q_i}/2.
The sea is the lower-band projector P[u] of h[u]; A_sea[u] = Tr(P[u] h[u]). reviewer A's basis (found3 sec. 2.3 item 7):
    A_sea^(2) = (N/a^2) [ s1 (tr K)^2 + s2 tr K^2 + s3 sum_i K_ii^2 + sb tr(Omega^T Omega) + sc |grad N / N|^2 ],
K = sym(grad v)/N, Omega = antisym(grad v)/N.

Band projectors P_s(p) = (1 + s n(p).Gamma)/2 with the unit 4-vector n = (m, sin p)/eps, so
  Tr P_+(p') P_-(p) = 1 - n.n' = |n - n'|^2 / 2 =: O(p, p'),  Tr P_a P_b P_c = (1 + a.b + b.c + c.a)/2  (a = s_a n(p_a), ...).

Quantities (all per unit volume, int_p = int_BZ d^3p/(2 pi)^3):
 (A) two-point, analytic q -> 0 (route A): E''_v(q, vhat) = -int_p mv^2 O/(eps + eps') with mv = vhat.(Q(p) + Q(p'))/2;
     its q^2 coefficient is C(qhat, vhat) = sum v_j v_k qhat_i qhat_l T_jkil, T_jkil = -(1/4) int_p Q_j Q_k g_il / eps,
     g_il = d_i n . d_l n = delta_il c_i^2/eps^2 - d_i c_i d_l c_l / eps^4 (d = sin p, c = cos p). Cubic symmetry leaves
     T1111, TA = T_{11 22}, TB = T_{12 12};  X_L = 2 TB, X_T = TA, s3 = T1111 - TA - 2 TB, where
     C = X_L (qhat.vhat)^2 + X_T + s3 sum_i qhat_i^2 vhat_i^2,  X_L = s1 + (s2 - sb)/2,  X_T = (s2 + sb)/2.
     Clock: E''_N = -int_p (eps' - eps)^2 O / (4 (eps + eps')) = O(q^4): sc = 0 exactly (the interband energy-density
     vertex is (eps' - eps)/2 times an overlap).
 (B) two-point at finite q (route B): the same E'' evaluated at q = t qhat, fitted in t^2 (q^2 and q^4 terms).
 (C) three-point (Daleckii-Krein): the second-order sea density rho2(K) at K = q1 + q2 for flow waves v1 e^{i q1 x},
     v2 e^{i q2 x} at N = 1:  rho2 = sum_orderings int_p sum_{n,k,m} f[x_n, x_k, x_m] m_second m_first
     (s_m eps(p) + s_n eps(p+K))/2 * Tr P_n(p+K) P_k(p+q_first) P_m(p), f the divided difference of the step theta(-x).
     Fitted at O(q^2) on the seven cubic invariants B1..B7 (B1 = (q1.v1)(q2.v2), B2 = (q2.v1)(q1.v2),
     B3 = (q1.v1)(q1.v2) + (q2.v1)(q2.v2), B4 = (q1.q2)(v1.v2), B5 = (q1^2 + q2^2)(v1.v2), B6 = sum q1_i q2_i v1_i v2_i,
     B7 = sum (q1_i^2 + q2_i^2) v1_i v2_i). reviewer A's basis gives rho = -A/N at N = 1, i.e.
     beta1 = 2 s1, beta2 = s2 - sb, beta4 = s2 + sb, beta6 = 2 s3, beta3 = beta5 = beta7 = 0 (no undifferentiated flow).
 (D) rate, linear order (adiabatic C1 = i Pdot/(eps_a - eps_b)): rho1(K) = i int_p (eps - eps') O mv / (eps + eps')^2 per unit
     d_t v (and the same integrand for the current's response to d_t N). O(K^3): the a^-2 d_t div v coefficient c3 is zero.

Outputs: <out>/ls_zone_v1_<tag>.json. Never edits its inputs."""
import os
import sys


def _threads():
    t = "8"
    if "--threads" in sys.argv:
        t = sys.argv[sys.argv.index("--threads") + 1]
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[k] = t
    return int(t)


NT = _threads()
import argparse
import json
import math
import time
import numpy as np
import torch

torch.set_num_threads(NT)
torch.set_default_dtype(torch.float64)
PI = math.pi


# ------------------------------------------------------------------ quadrature on the Brillouin zone
def axis_nodes(n, kind):
    """nodes and weights on [-pi, pi] for one axis; 'uniform' = midpoint rule; 'sidi' = midpoint in u with p = u - sin(2u)/2
    (p' = 2 sin^2 u vanishes at the valleys p = 0, +-pi: the massless integrands' 1/k points are smoothed)."""
    u = -PI + 2 * PI * (torch.arange(n, dtype=torch.float64) + 0.5) / n
    w = torch.full((n,), 2 * PI / n)
    if kind == "uniform":
        return u, w
    p = u - torch.sin(2 * u) / 2
    return p, w * (1 - torch.cos(2 * u))


def zone_iter(n, kind, chunk=1 << 21):
    """yield (p (M,3), weight (M,)) over the n^3 product grid, chunked; weights include 1/(2 pi)^3."""
    x, w = axis_nodes(n, kind)
    X, Y = torch.meshgrid(x, x, indexing="ij")
    WX, WY = torch.meshgrid(w, w, indexing="ij")
    X = X.reshape(-1); Y = Y.reshape(-1); WXY = (WX * WY).reshape(-1)
    per = max(1, chunk // X.numel())
    for k0 in range(0, n, per):
        zs = x[k0:k0 + per]; wz = w[k0:k0 + per]
        nz = zs.numel()
        P = torch.stack([X.repeat(nz), Y.repeat(nz), zs.repeat_interleave(X.numel())], dim=1)
        W = WXY.repeat(nz) * wz.repeat_interleave(X.numel())
        yield P, W / (2 * PI) ** 3


# ------------------------------------------------------------------ symbols
def nvec(p, m):
    """unit 4-vector n = (m, sin p)/eps and eps (p (M,3))."""
    d = torch.sin(p)
    eps = torch.sqrt(m * m + (d * d).sum(1))
    nv = torch.cat([torch.full_like(eps, m)[:, None], d], dim=1) / eps[:, None]
    return nv, eps


def Qs(p):
    return torch.sin(2 * p) / 2


# ------------------------------------------------------------------ (A) two-point tensors, analytic q -> 0
def two_point_tensors(m, n, kind):
    t1111 = ta = tb = 0.0
    for p, W in zone_iter(n, kind):
        d = torch.sin(p); c = torch.cos(p)
        e2 = m * m + (d * d).sum(1); e = torch.sqrt(e2); e5 = e2 * e2 * e
        d1, d2 = d[:, 0], d[:, 1]; c1, c2 = c[:, 0], c[:, 1]
        t1111 += float((W * (-0.25) * d1 ** 2 * c1 ** 4 * (e2 - d1 ** 2) / e5).sum())
        ta += float((W * (-0.25) * d1 ** 2 * c1 ** 2 * c2 ** 2 * (e2 - d2 ** 2) / e5).sum())
        tb += float((W * 0.25 * d1 ** 2 * c1 ** 2 * d2 ** 2 * c2 ** 2 / e5).sum())
    XL, XT = 2 * tb, ta
    return {"T1111": t1111, "TA": ta, "TB": tb, "X_L": XL, "X_T": XT, "s3": t1111 - ta - 2 * tb}


def C_pred(XL, XT, s3, qh, vh):
    qh = np.asarray(qh, float); vh = np.asarray(vh, float)
    return XL * float(qh @ vh) ** 2 + XT * float(qh @ qh) * float(vh @ vh) + s3 * float(np.sum(qh ** 2 * vh ** 2))


# ------------------------------------------------------------------ (B) two-point at finite q
def two_point_finite(m, n, kind, q, vh=None, cN=0.0):
    """E''/Vol for the real wave A cos(q.x) of (cN on the clock, vh on the flow): -int (cN (eps'-eps)/2 + mv)^2 O/(eps+eps')."""
    q = torch.as_tensor(q, dtype=torch.float64)
    vh = torch.zeros(3) if vh is None else torch.as_tensor(vh, dtype=torch.float64)
    tot = 0.0
    for p, W in zone_iter(n, kind):
        n0, e0 = nvec(p, m)
        for qs in (q, -q):          # the real wave's +q and -q transfers (their clock-flow cross terms cancel: time reversal)
            n1, e1 = nvec(p + qs, m)
            O = 0.5 * ((n1 - n0) ** 2).sum(1)
            mv = 0.5 * ((Qs(p) + Qs(p + qs)) * vh).sum(1)
            M = cN * 0.5 * (e1 - e0) + mv
            tot += 0.5 * float((-W * M * M * O / (e0 + e1)).sum())
    return tot


# ------------------------------------------------------------------ (C) three-point density (Daleckii-Krein)
def _dd_step(xa, xb, xc, sa, sb, sc):
    """second divided difference of theta(-x) at three points with band signs s (-1 occupied); coincident same-band points safe."""
    nocc = (sa < 0) + (sb < 0) + (sc < 0)
    if nocc in (0, 3):
        return None
    pts = [(xa, sa), (xb, sb), (xc, sc)]
    occ = [x for x, s in pts if s < 0]
    emp = [x for x, s in pts if s > 0]
    if nocc == 2:
        return -1.0 / ((occ[0] - emp[0]) * (occ[1] - emp[0]))
    return 1.0 / ((occ[0] - emp[0]) * (occ[0] - emp[1]))


def _three_point_chunk(p, W, m, q1, q2, v1, v2):
    K = q1 + q2
    nP, eP = nvec(p, m); nK, eK = nvec(p + K, m)
    QP, QK = Qs(p), Qs(p + K)
    out = torch.zeros_like(eP)
    for qf, vf, vs in ((q2, v2, v1), (q1, v1, v2)):
        nF, eF = nvec(p + qf, m); QF = Qs(p + qf)
        mf = 0.5 * ((QP + QF) * vf).sum(1)
        ms = 0.5 * ((QF + QK) * vs).sum(1)
        band = torch.zeros_like(eP)
        for sn in (-1, 1):
            for sk in (-1, 1):
                for sm in (-1, 1):
                    f = _dd_step(sn * eK, sk * eF, sm * eP, sn, sk, sm)
                    if f is None:
                        continue
                    a = sn * nK; b = sk * nF; c = sm * nP
                    tr = 0.5 * (1 + (a * b).sum(1) + (b * c).sum(1) + (c * a).sum(1))
                    band = band + f * 0.5 * (sm * eP + sn * eK) * tr
        out = out + mf * ms * band
    return float((W * out).sum())


def three_point_clean(m, n, kind, q1, q2, v1, v2):
    q1 = torch.as_tensor(q1, dtype=torch.float64); q2 = torch.as_tensor(q2, dtype=torch.float64)
    v1 = torch.as_tensor(v1, dtype=torch.float64); v2 = torch.as_tensor(v2, dtype=torch.float64)
    return sum(_three_point_chunk(p, W, m, q1, q2, v1, v2) for p, W in zone_iter(n, kind))


def basis7(q1, q2, v1, v2):
    q1, q2, v1, v2 = (np.asarray(x, float) for x in (q1, q2, v1, v2))
    return np.array([
        (q1 @ v1) * (q2 @ v2),
        (q2 @ v1) * (q1 @ v2),
        (q1 @ v1) * (q1 @ v2) + (q2 @ v1) * (q2 @ v2),
        (q1 @ q2) * (v1 @ v2),
        (q1 @ q1 + q2 @ q2) * (v1 @ v2),
        np.sum(q1 * q2 * v1 * v2),
        np.sum((q1 ** 2 + q2 ** 2) * v1 * v2)])


# ------------------------------------------------------------------ (D) linear rate response
def rate_linear(m, n, kind, K, vh):
    """rho1(K)/(i d_t v) = int (eps - eps') O mv/(eps + eps')^2 (equal to the current's response to d_t N along vh)."""
    K = torch.as_tensor(K, dtype=torch.float64); vh = torch.as_tensor(vh, dtype=torch.float64)
    tot = 0.0
    for p, W in zone_iter(n, kind):
        n0, e0 = nvec(p, m); n1, e1 = nvec(p + K, m)
        O = 0.5 * ((n1 - n0) ** 2).sum(1)
        mv = 0.5 * ((Qs(p) + Qs(p + K)) * vh).sum(1)
        tot += float((W * (e0 - e1) * O * mv / (e0 + e1) ** 2).sum())
    return tot


def rate_linear_abs(m, n, kind, K, vh):
    """int |integrand| (to show the zero is not a cancellation of large terms)."""
    K = torch.as_tensor(K, dtype=torch.float64); vh = torch.as_tensor(vh, dtype=torch.float64)
    tot = 0.0
    for p, W in zone_iter(n, kind):
        n0, e0 = nvec(p, m); n1, e1 = nvec(p + K, m)
        O = 0.5 * ((n1 - n0) ** 2).sum(1)
        mv = 0.5 * ((Qs(p) + Qs(p + K)) * vh).sum(1)
        tot += float((W * ((e0 - e1) * O * mv / (e0 + e1) ** 2).abs()).sum())
    return tot


# ------------------------------------------------------------------ fits
def fit_even(ts, ys, powers):
    A = np.array([[t ** k for k in powers] for t in ts])
    c, *_ = np.linalg.lstsq(A, np.asarray(ys), rcond=None)
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=float, required=True)
    ap.add_argument("--n-list", default="48,64,96")
    ap.add_argument("--kind", default="sidi")
    ap.add_argument("--t-list", default="0.02,0.04,0.08")
    ap.add_argument("--t3-list", default="0.01,0.02,0.04")
    ap.add_argument("--n3", type=int, default=64)
    ap.add_argument("--nconf", type=int, default=24)
    ap.add_argument("--parts", default="A,B,C,D")
    ap.add_argument("--seed", type=int, default=447)
    ap.add_argument("--k-list", default="0.02,0.04,0.08,0.16")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--threads", default="8")
    a = ap.parse_args()
    m = a.m
    parts = set(a.parts.split(","))
    res = {"m": m, "kind": a.kind, "threads": NT, "torch": torch.__version__, "script": "ls_zone_v1.py"}
    t0 = time.time()
    nl = [int(x) for x in a.n_list.split(",")]
    if "A" in parts:
        res["A"] = {}
        for n in nl:
            r = two_point_tensors(m, n, a.kind)
            res["A"][str(n)] = r
            print("A n=%d" % n, json.dumps(r), "%.1fs" % (time.time() - t0), flush=True)
    XL, XT, s3 = (res["A"][str(nl[-1])][k] for k in ("X_L", "X_T", "s3")) if "A" in parts else (None, None, None)
    if "B" in parts:
        ts = [float(x) for x in a.t_list.split(",")]
        r2 = 1 / math.sqrt(2); r3 = 1 / math.sqrt(3)
        dirs = {"100L": ([1, 0, 0], [1, 0, 0]), "100T": ([1, 0, 0], [0, 1, 0]), "110L": ([r2, r2, 0], [r2, r2, 0]),
                "110T": ([r2, r2, 0], [r2, -r2, 0]), "110Tz": ([r2, r2, 0], [0, 0, 1]), "111L": ([r3, r3, r3], [r3, r3, r3]),
                "111T": ([r3, r3, r3], [r2, -r2, 0])}
        res["B"] = {}
        nB = nl[-1]
        for lab, (qh, vh) in dirs.items():
            ys = [two_point_finite(m, nB, a.kind, [t * x for x in qh], vh) / t ** 2 for t in ts]
            c = fit_even(ts, ys, [0, 2, 4])
            pred = C_pred(XL, XT, s3, qh, vh) if XL is not None else None
            res["B"][lab] = {"t": ts, "E2_over_t2": ys, "fit_c2_c4_c6": c.tolist(), "pred_A": pred}
            print("B", lab, c.tolist(), "pred", pred, "%.1fs" % (time.time() - t0), flush=True)
        ys = [two_point_finite(m, nB, a.kind, [t, 0, 0], None, cN=1.0) / t ** 2 for t in ts]
        res["B"]["clock100"] = {"t": ts, "E2_over_t2": ys, "fit_c2_c4_c6": fit_even(ts, ys, [0, 2, 4]).tolist()}
        print("B clock", ys, flush=True)
    if "C" in parts:
        rng = np.random.default_rng(a.seed)
        ts = [float(x) for x in a.t3_list.split(",")]
        confs = []
        for k in range(a.nconf):
            q1 = rng.normal(size=3); q2 = rng.normal(size=3)
            q1 /= np.linalg.norm(q1); q2 = q2 / np.linalg.norm(q2) * (0.5 + rng.random())
            v1 = rng.normal(size=3); v2 = rng.normal(size=3)
            confs.append((q1, q2, v1, v2))
        rows, ys_by_t = [], {t: [] for t in ts}
        for (q1, q2, v1, v2) in confs:
            rows.append(basis7(q1, q2, v1, v2))
            for t in ts:
                y = 0.5 * (three_point_clean(m, a.n3, a.kind, t * q1, t * q2, v1, v2)
                           + three_point_clean(m, a.n3, a.kind, -t * q1, -t * q2, v1, v2)) / t ** 2
                ys_by_t[t].append(y)
        A7 = np.array(rows)
        fits = {}
        for t in ts:
            beta, resid, rk, sv = np.linalg.lstsq(A7, np.array(ys_by_t[t]), rcond=None)
            yfit = A7 @ beta
            fits[str(t)] = {"beta": beta.tolist(), "rel_resid": float(np.linalg.norm(yfit - ys_by_t[t]) / np.linalg.norm(ys_by_t[t])),
                            "rank": int(rk)}
            print("C t=%g beta" % t, np.array2string(beta, precision=6), "rel resid %.2e" % fits[str(t)]["rel_resid"],
                  "%.1fs" % (time.time() - t0), flush=True)
        # Richardson in t^2 (error O(t^2 log t) at m = 0)
        if len(ts) >= 2:
            b1 = np.array(fits[str(ts[0])]["beta"]); b2 = np.array(fits[str(ts[1])]["beta"])
            r = (ts[1] / ts[0]) ** 2
            fits["richardson_t01"] = ((r * b1 - b2) / (r - 1)).tolist()
        # K = 0 check against the two-point (rho2(q,-q,v,v) = -2 E''): one configuration per t
        kc = []
        for t in ts[:1]:
            qh = np.array([0.6, 0.48, 0.64]); vh = np.array([0.3, -0.5, 0.81])
            r3p = three_point_clean(m, a.n3, a.kind, t * qh, -t * qh, vh, vh)
            r2p = two_point_finite(m, a.n3, a.kind, (t * qh).tolist(), vh.tolist())
            kc.append({"t": t, "rho2_K0": r3p, "minus2_E2": -2 * r2p, "rel": abs(r3p + 2 * r2p) / abs(r2p)})
            print("C K=0 check", kc[-1], flush=True)
        res["C"] = {"t": ts, "n3": a.n3, "nconf": a.nconf, "fits": fits, "K0_check": kc,
                    "confs": [[x.tolist() for x in c] for c in confs], "y": {str(t): ys_by_t[t] for t in ts}}
    if "D" in parts:
        ks = [float(x) for x in a.k_list.split(",")]
        out = {}
        for lab, (kh, vh) in {"100L": ([1, 0, 0], [1, 0, 0]), "111L": ([1 / math.sqrt(3)] * 3, [1 / math.sqrt(3)] * 3),
                              "100T": ([1, 0, 0], [0, 1, 0])}.items():
            ys = [rate_linear(m, nl[-1], a.kind, [k * x for x in kh], vh) for k in ks]
            yabs = [rate_linear_abs(m, nl[-1], a.kind, [k * x for x in kh], vh) for k in ks]
            # fit y = c1 k + c3 k^3 + c5 k^5 (odd in k)
            c = fit_even(ks, ys, [1, 3, 5])
            out[lab] = {"k": ks, "rho1": ys, "abs_integrand": yabs, "fit_c1_c3_c5": c.tolist()}
            print("D", lab, ys, "fit", c.tolist(), flush=True)
        res["D"] = out
    res["seconds"] = time.time() - t0
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "ls_zone_v1_%s.json" % a.tag), "w") as f:
        json.dump(res, f, indent=1)
    print("DONE %.1fs" % res["seconds"])


if __name__ == "__main__":
    main()
