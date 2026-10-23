#!/usr/bin/env python
"""dust_readouts_selftest_v1.py (busi lane, reviewer A; WEB40G icloud item 7): the self-test of dust_readouts_v1.py on a manufactured chain
(no engine, no stored record): a 2-component sine Dirac chain with mass and a two-step momentum under smooth clock and flow fields with a
trapped annulus that closes along a manufactured history. Checks (PASS/FAIL printed, exit 0 only if all pass):
 1 the manufactured h is Hermitian;
 2 the Chebyshev propagator is unitary to 1e-12 and agrees with the dense exponential to 1e-11 (2 step sizes);
 3 scipy's Taylor control agrees with the Chebyshev propagator to 1e-12 and is unitary to 1e-12;
 4 the symbol: the velocity matrix at q = 0 equals N sigma_x + v I (eigenvalues 0 and -2N at a root, N/2 and -3N/2 at v = -N/2), and the lattice
   matrix at the root has the positive eigenvalue N(cos q - cos 2q) for q = 0.4 (the finite-spacing branch; spec section 1);
 5 the split identity all = in + out + cross with the explicit cross term 2 Re <U X_in|Pi|U X_out>, to 1e-12, on packets that make the cross term nonzero;
 6 the current: J_c Hermitian, rank at most 16 (8 expected for one- and two-step hopping, printed), the time derivative of the weight beyond the cut equal
   to the current to 1e-9 (Richardson-extrapolated central differences on the dense exponential), the site continuity d rho_x/dt = F(x) - F(x+1) to 1e-9,
   the telescoping sum zero, and zero current through cuts outside the chain;
 7 the Killing bins partition the norm to 1e-12, the jump is zero when the fields do not change, and Delta p_I by the bins equals Tr[(P_I^+ - P_I^-) C] by dense
   projectors to 1e-12;
 8 the root finder: N = 1, v = -sqrt(rs/r) gives the outer root rs exactly; a Gaussian flow annulus gives an inner and an outer root;
 9 the coordinate-rest arithmetic of the spec (ma 0.2, 2GE 59.1 a, R_0 256 a): q 0.10733 and omega/m 0.87854 (the notes' digits) within 1e-5, the continuum
   comparator 0.877007;
10 the resume: run_key interrupted after a checkpoint and resumed gives the same arrays (1e-12) and jump rows as an uninterrupted run, and a finished key is skipped.
Usage: python dust_readouts_selftest_v1.py [--threads 4]"""
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dust_readouts_v1 as M  # noqa: E402  (sets the thread caps on import)

import json  # noqa: E402
import math  # noqa: E402
import shutil  # noqa: E402
import tempfile  # noqa: E402

import numpy as np  # noqa: E402
import scipy.linalg as sla  # noqa: E402
import scipy.sparse as sp  # noqa: E402

SX = np.array([[0.0, 1.0], [1.0, 0.0]])
SY = np.array([[0.0, -1j], [1j, 0.0]])
SZ = np.array([[1.0, 0.0], [0.0, -1.0]])


class ToyOps:
    """the manufactured chain: h_0 = (T - T^dag)/(2i) (x) sigma_x + m sigma_z + (kappa/R) sigma_y, Q = (T^2 - T^dag^2)/(4i) (x) I, site-major rows;
    h = (1/2){N, h_0} + (1/2){v, Q} (the declared dictionary: the clock on the energy density, the flow on the momentum density)."""

    def chain(self, R, m, kappa):
        n = len(R)
        T = sp.diags([np.ones(n - 1)], [1], shape=(n, n), format="csr")
        D1 = (T - T.T) / 2j
        T2 = T @ T
        D2 = (T2 - T2.T) / 4j
        h0 = sp.kron(D1, sp.csr_matrix(SX)) + m * sp.kron(sp.identity(n), sp.csr_matrix(SZ)) + sp.kron(sp.diags(kappa / np.asarray(R, float)), sp.csr_matrix(SY))
        Q = sp.kron(D2, sp.identity(2))
        return h0.tocsr(), Q.tocsr()

    def clocked(self, h0, Q, N, v):
        Nd = sp.diags(np.repeat(np.asarray(N, float), 2))
        vd = sp.diags(np.repeat(np.asarray(v, float), 2))
        return (0.5 * (Nd @ h0 + h0 @ Nd) + 0.5 * (vd @ Q + Q @ vd)).tocsr()


class ToyHistory:
    """a manufactured history: n sites, records every tau, a lapse dip and a Gaussian flow annulus whose amplitude falls record by record, so the
    trapped annulus (2 zeros of N + v) closes at some record; the records carry this script's zeros as their trapping zeros."""

    def __init__(self, n=96, nrec=26, tau=0.5, rs=3.5):
        self.R = np.arange(1, n + 1, dtype=float)
        self.T = tau * np.arange(nrec)
        self.rs = rs
        self.tau = tau
        self.ne = n
        self.files = None
        self.rec = []
        for k in range(nrec):
            N, v = self.fields(k)
            outer, inner = M.zeros_of_sum(N, v, self.R, self.ne)
            self.rec.append({"trapping_zeros": [z for z in (inner, outer) if z is not None]})

    def fields(self, k):
        N = 1.0 - 0.3 * np.exp(-((self.R - 30.0) / 12.0) ** 2)
        amp = 1.35 - 0.05 * k
        v = -amp * np.exp(-((self.R - 36.0) / 14.0) ** 2)
        return N, v


def packets(R, centres, widths, qs):
    cols = []
    for c, w, q in zip(centres, widths, qs):
        g = np.exp(-((R - c) / w) ** 2 + 1j * q * R)
        x = np.zeros((2 * len(R),), complex)
        x[0::2] = g
        x[1::2] = 0.5 * g
        cols.append(x / np.linalg.norm(x))
    return np.array(cols).T


RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append(bool(ok))
    print("%s  %s%s" % ("PASS" if ok else "FAIL", name, (": " + detail) if detail else ""), flush=True)
    return bool(ok)


def selftest():
    rng = np.random.default_rng(7)
    ops = ToyOps()
    hist = ToyHistory()
    R, n = hist.R, len(hist.R)
    m = 0.2
    b = math.sqrt(m * m + 1.0)
    h0, Q = ops.chain(R, m, 1.0)
    N0, v0 = hist.fields(0)
    h = ops.clocked(h0, Q, N0, v0)
    hd = h.toarray()
    # 1 hermiticity
    check("1 the manufactured h is Hermitian", np.max(np.abs(hd - hd.conj().T)) < 1e-15, "max |h - h^dag| %.1e" % np.max(np.abs(hd - hd.conj().T)))
    # 2 Chebyshev unitarity and accuracy
    X = rng.standard_normal((2 * n, 3)) + 1j * rng.standard_normal((2 * n, 3))
    X /= np.linalg.norm(X, axis=0)
    ok2, det = True, []
    for tau in (0.5, 3.0):
        Y, K = M.cheb_expm_apply(h, X, tau, 1e-15)
        Ye = sla.expm(-1j * tau * hd) @ X
        un = np.max(np.abs(np.linalg.norm(Y, axis=0) ** 2 - 1.0))
        G = Y.conj().T @ Y
        ug = np.max(np.abs(G - X.conj().T @ X))   # busi fix: the columns are normalized, not orthogonalized, so unitarity keeps X^dag X (the identity only for orthonormal X)
        acc = np.max(np.abs(Y - Ye))
        det.append("tau %g: K %d, |norm^2 - 1| %.1e, |Y^dag Y - 1| %.1e, |Y - exact| %.1e" % (tau, K, un, ug, acc))
        ok2 = ok2 and un < 1e-12 and ug < 1e-12 and acc < 1e-11
    check("2 Chebyshev propagator unitary to 1e-12 and exact to 1e-11", ok2, "; ".join(det))
    # 3 Taylor control
    Yc, _ = M.cheb_expm_apply(h, X, 3.0, 1e-15)
    Yt = M.taylor_expm_apply(h, X, 3.0)
    unt = np.max(np.abs(np.linalg.norm(Yt, axis=0) ** 2 - 1.0))
    check("3 Taylor control agrees with Chebyshev to 1e-12 and is unitary", np.max(np.abs(Yc - Yt)) < 1e-12 and unt < 1e-12,
          "|cheb - taylor| %.1e, |norm^2 - 1| %.1e" % (np.max(np.abs(Yc - Yt)), unt))
    # 4 the symbol at a root
    Nc = 0.8
    ok4, det = True, []
    hc0, Qc = ops.chain(np.arange(1, 41, dtype=float), m, 0.0)
    for vc, expect in ((-Nc, (-2 * Nc, 0.0)), (-0.5 * Nc, (-1.5 * Nc, 0.5 * Nc))):
        hc = ops.clocked(hc0, Qc, np.full(40, Nc), np.full(40, vc)).toarray()
        s0 = 20
        H = {d: hc[2 * s0: 2 * s0 + 2, 2 * (s0 + d): 2 * (s0 + d) + 2] for d in range(-2, 3)}
        V0 = sum(1j * d * H[d] for d in range(-2, 3))
        ev = np.sort(np.linalg.eigvalsh(0.5 * (V0 + V0.conj().T)))
        herm = np.max(np.abs(V0 - V0.conj().T))
        tgt = np.sort(np.linalg.eigvalsh(Nc * SX + vc * np.eye(2)))
        det.append("v = %.2f: velocity-matrix eigenvalues %s (expected %s; N sigma_x + v %s; Hermitian to %.1e)" % (vc, np.round(ev, 12).tolist(), expect, np.round(tgt, 12).tolist(), herm))
        ok4 = ok4 and np.max(np.abs(ev - np.array(expect))) < 1e-13 and herm < 1e-14
    q = 0.4
    hc = ops.clocked(hc0, Qc, np.full(40, Nc), np.full(40, -Nc)).toarray()
    H = {d: hc[40: 42, 40 + 2 * d: 42 + 2 * d] for d in range(-2, 3)}
    Vq = sum(1j * d * H[d] * np.exp(1j * q * d) for d in range(-2, 3))
    evq = np.sort(np.linalg.eigvalsh(0.5 * (Vq + Vq.conj().T)))
    lam = Nc * (math.cos(q) - math.cos(2 * q))
    det.append("q = 0.4 at the root: larger eigenvalue %.9f, N(cos q - cos 2q) = %.9f (> 0: the finite-spacing branch)" % (evq[-1], lam))
    ok4 = ok4 and abs(evq[-1] - lam) < 1e-13 and lam > 0
    check("4 the symbol's velocity matrix: N sigma_x + v (0 and -2N at a root), lattice branch at q = 0.4", ok4, "; ".join(det))
    # 5 the split identity
    Xp = packets(R, [34.0, 40.0, 50.0], [6.0, 8.0, 10.0], [0.3, 0.1, -0.2])
    Ro = 40.0
    mask = np.repeat(R < Ro, 2)
    Xin = Xp * mask[:, None]
    Xout = Xp - Xin
    Yall, _ = M.cheb_expm_apply(h, Xp, 2.0, 1e-15)
    Yin, _ = M.cheb_expm_apply(h, Xin, 2.0, 1e-15)
    Yout = Yall - Yin
    beyond = R > 44.0
    Wall = M.site_weights(Yall, n)[beyond].sum(axis=0)
    Win = M.site_weights(Yin, n)[beyond].sum(axis=0)
    Wout = M.site_weights(Yout, n)[beyond].sum(axis=0)
    cross = (2.0 * np.real(Yin.conj() * Yout)).reshape(n, 2, -1).sum(axis=1)[beyond].sum(axis=0)
    resid = np.max(np.abs(Wall - Win - Wout - cross))
    check("5 the split identity all = in + out + cross (explicit) to 1e-12, cross nontrivial", resid < 1e-12 and np.max(np.abs(cross)) > 1e-4,
          "residual %.1e, |cross| up to %.3e, all %s" % (resid, np.max(np.abs(cross)), np.round(Wall, 6).tolist()))
    # 6 the current
    f = np.ones(3)
    c = 40.0
    J = M.current_operator(h, R > c)
    Jd = J.toarray()
    hermJ = np.max(np.abs(Jd - Jd.conj().T))
    rank = int(np.linalg.matrix_rank(Jd, tol=1e-12))
    F, per = M.current_value(J, Xp, f)

    def W_beyond(t, Xs, msk):
        Y = sla.expm(-1j * t * hd) @ Xs
        return float(np.dot(f, M.site_weights(Y, n)[msk].sum(axis=0)))

    def richardson(fun, dt=1e-3):
        d1 = (fun(dt) - fun(-dt)) / (2 * dt)
        d2 = (fun(dt / 2) - fun(-dt / 2)) / dt
        return (4 * d2 - d1) / 3

    Ffd = richardson(lambda t: W_beyond(t, Xp, R > c))
    rho_t = richardson(lambda t: np.dot(M.site_weights(sla.expm(-1j * t * hd) @ Xp, n), f)) if False else None
    Ys = [sla.expm(-1j * t * hd) @ Xp for t in (1e-3, -1e-3, 5e-4, -5e-4)]
    rho = [np.dot(M.site_weights(Y, n), f) for Y in Ys]
    drho = (4 * ((rho[2] - rho[3]) / 1e-3) - (rho[0] - rho[1]) / 2e-3) / 3
    Fx = np.array([M.current_value(M.current_operator(h, R >= R[x]), Xp, f)[0] for x in range(n)] + [0.0])
    cont = np.max(np.abs(drho - (Fx[:-1] - Fx[1:])))
    tele = abs(np.sum(Fx[:-1] - Fx[1:]) - (Fx[0] - Fx[-1]))
    Fout0 = M.current_value(M.current_operator(h, R > 0.0), Xp, f)[0]
    FoutN = M.current_value(M.current_operator(h, R > R[-1] + 1.0), Xp, f)[0]
    ok6 = hermJ < 1e-14 and rank <= 16 and abs(Ffd - F) < 1e-9 and cont < 1e-9 and tele < 1e-14 and abs(Fout0) < 1e-15 and abs(FoutN) < 1e-15 and abs(Fx[0]) < 1e-15
    check("6 the current: Hermitian, rank <= 16, d/dt(weight beyond) = current to 1e-9, site continuity to 1e-9, telescoping, zero outside", ok6,
          "|J - J^dag| %.1e, rank %d, |FD - F| %.1e (F %+.4e), continuity %.1e, telescoping %.1e, outside %.1e / %.1e" % (hermJ, rank, abs(Ffd - F), F, cont, tele, Fout0, FoutN))
    # 7 the Killing bins and the jump
    bins, mean, e, _ = M.killing_bins(h, Xp, m, b)
    part = np.max(np.abs(bins.sum(axis=0) - np.linalg.norm(Xp, axis=0) ** 2))
    bins_same, _, _, _ = M.killing_bins(h, Xp, m, b)
    N1, v1 = hist.fields(5)
    h1 = ops.clocked(h0, Q, N1, v1)
    bins1, _, _, _ = M.killing_bins(h1, Xp, m, b)
    e0, U0 = np.linalg.eigh(hd)
    e1, U1 = np.linalg.eigh(h1.toarray())
    Pm0 = U0[:, e0 > m] @ U0[:, e0 > m].conj().T
    Pm1 = U1[:, e1 > m] @ U1[:, e1 > m].conj().T
    Cd = Xp @ np.diag(f) @ Xp.conj().T
    dp_dense = float(np.real(np.trace((Pm1 - Pm0) @ Cd)))
    dp_bins = float(np.dot(f, (bins1[3] + bins1[4]) - (bins[3] + bins[4])))
    check("7 Killing bins partition the norm, zero jump at equal fields, jump by bins = by dense projectors (1e-12)",
          part < 1e-12 and np.max(np.abs(bins_same - bins)) < 1e-12 and abs(dp_dense - dp_bins) < 1e-12 and abs(dp_bins) > 1e-6,
          "partition %.1e, equal-field jump %.1e, dense - bins %.1e (Delta p above m %+.4e)" % (part, np.max(np.abs(bins_same - bins)), abs(dp_dense - dp_bins), dp_bins))
    # 8 the root finder
    Ra = np.arange(1, 97, dtype=float)
    o, i = M.zeros_of_sum(np.ones(96), -np.sqrt(40.0 / Ra), Ra)
    o2, i2 = M.zeros_of_sum(N0, v0, R, n)
    check("8 the root finder: N = 1, v = -sqrt(40/r) gives the outer root 40 exactly; the Gaussian annulus has 2 roots with inner < outer",
          o is not None and abs(o - 40.0) < 1e-12 and i is None and o2 is not None and i2 is not None and i2 < o2,
          "root %s (inner %s); annulus inner %s outer %s" % (M.fmt(o, "%.12f"), i, M.fmt(i2, "%.6f"), M.fmt(o2, "%.6f")))
    # 9 coordinate rest
    d = M.coordinate_rest_killing(0.2, math.sqrt(59.1 / 256.0))
    check("9 coordinate rest (ma 0.2, 2GE 59.1 a, R_0 256 a): q 0.10733, omega/m 0.87854 within 1e-5; continuum 0.877007",
          abs(d["q"] - 0.10733) < 1e-5 and abs(d["omega_over_m"] - 0.87854) < 1e-5 and abs(d["continuum_omega_over_m"] - 0.877007) < 1e-6,
          "q %.7f, omega/m %.7f, continuum %.7f, residual %.1e" % (d["q"], d["omega_over_m"], d["continuum_omega_over_m"], d["group_velocity_residual"]))
    # 10 the resume
    tmp = tempfile.mkdtemp(prefix="dust_readouts_selftest_")
    try:
        a = M.default_args(T_end_rs=float(hist.T[-1] / hist.rs), jump_every_rs=8 * hist.tau / hist.rs, checkpoint_every_rs=4 * hist.tau / hist.rs, no_jumps=False,
                           propagator="chebyshev", cheb_tol=1e-15, eig_driver="evr", redo=False)
        key = "TOY_k1"
        TL = float(hist.T[2] / hist.rs)
        logs = []
        quiet = logs.append
        d_a = os.path.join(tmp, "a")
        d_b = os.path.join(tmp, "b")
        os.makedirs(d_a)
        os.makedirs(d_b)
        res_a, arr_a = M.run_key(hist, ops, key, "toy", 1, Xp, f, TL, m, a, d_a, log=quiet)
        try:
            M.run_key(hist, ops, key, "toy", 1, Xp, f, TL, m, a, d_b, log=quiet, interrupt_at=2 + 8)
            interrupted = False
        except M.Interrupted:
            interrupted = True
        ck_present = os.path.exists(os.path.join(d_b, "ckpt_toy_%s.npz" % key))
        res_b, arr_b = M.run_key(hist, ops, key, "toy", 1, Xp, f, TL, m, a, d_b, log=quiet)
        resumed = any("resuming" in s for s in logs)
        worst = 0.0
        for nm in arr_a:
            if nm.startswith("X_end") or nm in ("f",) or "seconds" in nm:   # busi fix: wall-clock timings (step_seconds) differ between any 2 runs
                continue
            xa, xb = np.asarray(arr_a[nm], float), np.asarray(arr_b[nm], float)
            fa, fb = np.isfinite(xa), np.isfinite(xb)
            if xa.shape != xb.shape or np.any(fa != fb):
                worst = np.inf
                break
            if fa.any():
                worst = max(worst, float(np.max(np.abs(xa[fa] - xb[fb]))))
        jumps_eq = len(res_a["jump_ledger_sampled"]["rows"]) == len(res_b["jump_ledger_sampled"]["rows"]) == 2 and all(
            abs(ra["above_m_jump_weighted"] - rb["above_m_jump_weighted"]) < 1e-12 for ra, rb in zip(res_a["jump_ledger_sampled"]["rows"], res_b["jump_ledger_sampled"]["rows"]))
        logs.clear()
        M.run_key(hist, ops, key, "toy", 1, Xp, f, TL, m, a, d_b, log=quiet)
        skipped = any("skipped" in s for s in logs)
        end_eq = abs(res_a["end"]["all"]["b16"] - res_b["end"]["all"]["b16"]) < 1e-12 and res_a["outward_crossing_clock_per_column"] == res_b["outward_crossing_clock_per_column"] \
            and res_a["first_closing_after_launch_over_rs"] == res_b["first_closing_after_launch_over_rs"]
        check("10 resume: interrupted after a checkpoint and resumed = uninterrupted (1e-12), the jump rows equal, a finished key skipped",
              interrupted and ck_present and resumed and worst < 1e-12 and jumps_eq and skipped and end_eq,
              "interrupted %s, checkpoint %s, resumed %s, max |a - b| %.1e, jump rows equal %s, skipped %s; first closing %s, crossing clocks %s, beyond 16 r_s at the end %.6e, "
              "above m launch %.6f end %.6f (sampled sum %+.6f of total %+.6f)" % (
                  interrupted, ck_present, resumed, worst, jumps_eq, skipped, res_a["first_closing_after_launch_over_rs"], res_a["outward_crossing_clock_per_column"],
                  res_a["end"]["all"]["b16"], res_a["above_m_launch_weighted"], res_a["above_m_end_weighted"], res_a["above_m_change_sampled_sum_weighted"],
                  res_a["above_m_change_total_weighted"]))
        # the per-key json is valid json with the compared keys present
        fj = os.path.join(d_a, "toy_%s.json" % key)
        J = json.load(open(fj))
        check("10b the per-key json carries the compared keys", all(k_ in J for k_ in ("end", "cross_term_b16_end", "inside_root_at_launch", "killing_launch_per_column",
                                                                                        "killing_end_per_column", "outward_crossing_clock_per_column", "jump_ledger_sampled", "checks")),
              "keys %d; norm drift %.1e; cross remainder - explicit %.1e" % (len(J), J["checks"]["norm2_max_drift"], J["checks"]["cross_remainder_minus_explicit_max"]))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    ok = all(RESULTS)
    print("%d checks, %d passed: %s" % (len(RESULTS), sum(RESULTS), "ALL PASS" if ok else "FAILURES"), flush=True)
    return ok


if __name__ == "__main__":
    sys.exit(0 if selftest() else 1)
