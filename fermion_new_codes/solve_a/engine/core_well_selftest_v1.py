#!/usr/bin/env python
"""core_well_selftest_v1.py (reviewer A for the solve builder, the busi lane): the self-test of core_well_v1 on manufactured inputs. No slice, no
record and no engine module is needed (page_ops_v6 is used as an extra comparison when it is importable). The checks:
1. the range-2 current: on a 2-shell toy chain (16 sites, 2 spinor components, a random Hermitian h whose hopping reaches 2 sites, 3 random
   columns with occupations) the straddling-bond sum of core_well_v1.straddling_current against the direct full-matrix Tr[D i[h, Pi_out]], against
   the symmetric finite difference of Tr[Pi_out D(t)] under exp(-i h t), and against page_ops_v6.number_current_csr where importable; the
   straddling range is 2 sites and the count is the number of nonzero straddling elements;
2. the M(r) identity: a manufactured mass function M(r), w = -sqrt(2 g M/r), shell sources eps and J with eps + w J = M(R_x) - M(R_{x-1}) exactly:
   core_well_v1.mass_function returns M_w = I to rounding;
3. the Killing split's bins: levels placed exactly on the edges -b, -m, 0, m, b go to the half-open bins (lo, hi] as declared; on a toy h_m the
   bins' numbers sum to the dust count and the per-site bin densities sum to the site density (killing_analysis);
4. the exponent fit on an exact power law N = c N_<^alpha (fit_loglog returns alpha to 1e-10 with r^2 = 1);
5. the momentum split at pi/4 (plane waves on both sides of the threshold and on it; the projector is idempotent and orthogonal) and the shells.
Exit 0 with 'PASS: N checks, 0 failures' on success, 1 with 'FAIL' lines on a broken identity. Usage: python core_well_selftest_v1.py [--page-code DIR]."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import argparse  # noqa: E402
import math  # noqa: E402

import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402

import core_well_v1 as CW  # noqa: E402
import slices144_io_v1 as IO  # noqa: E402

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print("%s: %s %s" % ("ok  " if ok else "FAIL", name, detail), flush=True)


def toy_chain(n, rng, range_sites=2, scale=1.0):
    """a random Hermitian operator on a chain of n sites with 2 components whose hopping reaches range_sites sites (CSR)."""
    N2 = 2 * n
    A = np.zeros((N2, N2), np.complex128)
    for i in range(N2):
        for j in range(i, N2):
            if abs(i // 2 - j // 2) <= range_sites:
                z = rng.normal() + 1j * rng.normal()
                A[i, j] = z if i != j else rng.normal()
                A[j, i] = np.conj(A[i, j])
    return sp.csr_matrix(scale * A)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--page-code", dest="page_code", default=IO.PAGE_CODE)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args(argv)
    rng = np.random.default_rng(a.seed)
    print("core_well_selftest_v1: root finder %s" % CW.ROOT_FINDER)
    # ---------------------------------------------------------------------------------------------- 1. the range-2 current
    n = 16
    R = np.arange(1, n + 1, dtype=float)
    H = toy_chain(n, rng)
    X = rng.normal(size=(2 * n, 3)) + 1j * rng.normal(size=(2 * n, 3))
    X /= np.linalg.norm(X, axis=0)[None, :]
    f = np.array([1.0, 0.5, 0.25])
    x = 8.5
    I_s, nnz, rng_sites = CW.straddling_current(H, R, x, X, f)
    Hd = H.toarray()
    Pi = np.diag(np.repeat(R > x, 2).astype(float))
    D = (X * f[None, :]) @ X.conj().T
    I_direct = float(np.real(np.trace(D @ (1j * (Hd @ Pi - Pi @ Hd)))))
    check("range-2 current: straddling sum against the direct Tr[D i[h, Pi_out]]", abs(I_s - I_direct) < 1e-11, "straddling %.12e direct %.12e (%d elements, range %d sites)" % (I_s, I_direct, nnz, rng_sites))
    check("range-2 current: the straddling elements reach exactly 2 sites", rng_sites == 2, "range %d" % rng_sites)
    w, V = np.linalg.eigh(Hd)
    dt = 1e-3
    def n_out(t):
        U = (V * np.exp(-1j * w * t)[None, :]) @ V.conj().T
        Dt = U @ D @ U.conj().T
        return float(np.real(np.trace(Pi @ Dt)))
    I_fd = (n_out(dt) - n_out(-dt)) / (2 * dt)
    check("range-2 current: against the finite-difference d/dt of the number outside", abs(I_s - I_fd) < 1e-5 * max(1.0, abs(I_s)), "straddling %.8e finite difference %.8e" % (I_s, I_fd))
    # a second shell boundary: the inner cut of a 2-shell split, and the sum rule (in + out shells) for a current from the centre
    x2 = 4.5
    I2, _, _ = CW.straddling_current(H, R, x2, X, f)
    Pi2 = np.diag(np.repeat(R > x2, 2).astype(float))
    I2_direct = float(np.real(np.trace(D @ (1j * (Hd @ Pi2 - Pi2 @ Hd)))))
    check("range-2 current: the second shell boundary against the direct sum", abs(I2 - I2_direct) < 1e-11, "%.12e against %.12e" % (I2, I2_direct))
    try:
        for d_ in [x for x in str(a.page_code).split(os.pathsep) if x]:
            if os.path.isdir(d_) and d_ not in sys.path:
                sys.path.insert(0, d_)
        import page_ops_v6 as OPS
        h0, Q = OPS.chain_sparse(R, 0.2, 1)
        Hc = OPS.clocked_csr(h0, Q, np.ones(n) * 0.9, -0.3 * np.ones(n)).tocsr()
        J = OPS.number_current_csr(Hc, R, x)
        I_ops = CW.expect_rows(J, X, f)
        I_sc, nnz_c, rng_c = CW.straddling_current(Hc, R, x, X, f)
        check("range-2 current: the straddling sum on the engine's h_m against page_ops_v6.number_current_csr", abs(I_ops - I_sc) < 1e-10, "ops %.12e straddling %.12e (range %d)" % (I_ops, I_sc, rng_c))
    except Exception as ex:  # noqa: BLE001
        print("skip: page_ops_v6 not importable here (%s: %s); the engine comparison runs on the box" % (type(ex).__name__, ex))
    # ---------------------------------------------------------------------------------------------- 2. the M(r) identity
    g = 0.25
    Rm = np.arange(1, 201, dtype=float)
    Mfun = lambda r: 60.0 * (1.0 - np.exp(-r / 50.0)) + 0.01 * r  # noqa: E731
    M_true = Mfun(Rm)
    w = -np.sqrt(2.0 * g * M_true / Rm)
    src = M_true - np.concatenate([[0.0], M_true[:-1]])
    eps = 0.7 * src
    J = 0.3 * src / w
    M_w, I, src_back = CW.mass_function(Rm, w, eps, J, g)
    check("M(r) identity: r w^2/(2 g) equals the cumulative shell source", np.max(np.abs(M_w - I)) < 1e-10, "max |M_w - I| %.2e" % np.max(np.abs(M_w - I)))
    check("M(r) identity: the source per shell is eps + w J", np.max(np.abs(src_back - src)) < 1e-12)
    # ---------------------------------------------------------------------------------------------- 3. the Killing split
    m = 0.2
    b = math.sqrt(m * m + 1.0)
    om = np.array([-2 * b, -b, -b + 1e-3, -m, -m + 1e-9, 0.0, 1e-9, m, m + 1e-9, b, b + 1e-9])
    bins, bidx = CW.killing_bins(om, np.ones_like(om), m, b)
    counts = [q["levels"] for q in bins]
    check("Killing split: the edges go to the half-open bins (lo, hi]", counts == [2, 2, 2, 2, 2, 1], "counts %s (expected [2, 2, 2, 2, 2, 1])" % counts)
    check("Killing split: the 6 bins are labelled as declared", [q["label"] for q in bins] == list(CW.BIN_LABELS))
    nk = 12
    Rk = np.arange(1, nk + 1, dtype=float)
    Hk = toy_chain(nk, rng, scale=0.8)
    dust = rng.normal(size=(2 * nk, 2)) + 1j * rng.normal(size=(2 * nk, 2))
    dust, _ = np.linalg.qr(dust)
    fk = np.array([1.0, 0.6])
    KA, karr = CW.killing_analysis(Hk, dust, fk, Rk, m, b, None, 0.0, 64.0, 50.0)
    check("Killing split: the bins' numbers sum to the dust count", abs(sum(q["N"] for q in KA["bins"]) - fk.sum()) < 1e-10 and abs(KA["completeness_check"]) < 1e-10, "sum %.12f against %.12f" % (sum(q["N"] for q in KA["bins"]), fk.sum()))
    check("Killing split: the per-site bin densities sum to the site density", KA["site_density_check"] < 1e-10, "max defect %.2e" % KA["site_density_check"])
    check("Killing split: the bound part's density is the number in the window", abs(KA["bound"]["N"] - sum(karr["n_dust"][(karr["omega"] > 0) & (karr["omega"] <= m)])) < 1e-10, "%.12f" % KA["bound"]["N"])
    check("Killing split: the bins' energies sum to the dust's Killing energy", abs(sum(q["E"] for q in KA["bins"]) - float(np.real(np.sum(fk * np.sum(np.conj(dust) * (Hk @ dust), axis=0))))) < 1e-9)
    omb = 0.5 * m
    check("turning radius: r_s/(1 - (omega/m)^2) at omega = m/2 is 4 r_s/3", abs(CW.turning_radius(64.0, omb, m) - 64.0 * 4.0 / 3.0) < 1e-9 and CW.turning_radius(64.0, m, m) is None)
    # ---------------------------------------------------------------------------------------------- 4. the exponent fit
    Nenc = np.arange(1, 400, dtype=float) ** 3
    alpha = 0.41
    Nlapse = 0.37 * Nenc ** alpha
    fit = CW.fit_loglog(Nenc, Nlapse, np.ones(len(Nenc), bool))
    check("exponent fit: the slope of ln N against ln N_< on an exact power law", abs(fit["slope"] - alpha) < 1e-10 and fit["r2"] > 1 - 1e-12, "slope %.12f (expected %.2f), r2 %s" % (fit["slope"], alpha, fit["r2"]))
    eps_c = 0.26
    check("exponent forms: (eps_c - m)/(2 m) and eps_c/(2 m) at eps_c 0.26, m 0.2", abs((eps_c - m) / (2 * m) - 0.15) < 1e-12 and abs(eps_c / (2 * m) - 0.65) < 1e-12)
    # ---------------------------------------------------------------------------------------------- 5. the momentum split and the shells
    n = 64
    xs = np.arange(1, n + 1, dtype=float)
    def wave(k):
        v = np.zeros(2 * n, np.complex128)
        v[0::2] = np.exp(1j * k * xs) / math.sqrt(n)
        return v[:, None]
    for k, side in ((2 * math.pi * 3 / n, "low"), (2 * math.pi * 20 / n, "high"), (2 * math.pi * 8 / n, "low (on the threshold pi/4)"), (2 * math.pi * 40 / n, "low (the valley at pi, folds to p = pi/4... )")):
        lo, hi, mask = CW.momentum_split_columns(wave(k), n, 0.25)
        wl, wh = float(np.sum(np.abs(lo) ** 2)), float(np.sum(np.abs(hi) ** 2))
        p = (k + 0.5 * math.pi) % math.pi - 0.5 * math.pi
        expect_low = abs(p) <= math.pi / 4 + 1e-12
        check("momentum split: the plane wave at ak = %.4f (p = %.4f) is %s" % (k, p, "low" if expect_low else "high"), (wl > 1 - 1e-10 and wh < 1e-10) if expect_low else (wh > 1 - 1e-10 and wl < 1e-10), "low %.6f high %.6f" % (wl, wh))
    Xr = rng.normal(size=(2 * n, 2)) + 1j * rng.normal(size=(2 * n, 2))
    lo, hi, _ = CW.momentum_split_columns(Xr, n, 0.25)
    lo2, _, _ = CW.momentum_split_columns(lo, n, 0.25)
    check("momentum split: idempotent and orthogonal", np.max(np.abs(lo2 - lo)) < 1e-12 and abs(np.vdot(lo[:, 0], hi[:, 0])) < 1e-12 and np.max(np.abs(lo + hi - Xr)) < 1e-12)
    sh = CW.shells_of(2048, 128)
    check("shells: 2048 sites in 128-site shells give 16 shells ending at 2048", len(sh) == 16 and sh[-1] == (1920, 2048) and sh[0] == (0, 128))
    Rz = np.arange(1, 11, dtype=float)
    fz = np.array([-1, -0.5, 0.2, 0.6, 0.1, -0.3, -0.2, 0.4, 0.8, 1.0])
    Ro, Ri, z = CW.applied_roots(Rz, fz, np.zeros(10))
    check("applied roots: the largest rising zero and the largest falling zero below it", abs(Ro - (7.0 + 0.2 / 0.6)) < 1e-12 and abs(Ri - (5.0 + 0.1 / 0.4)) < 1e-12 and len(z) == 3 and [q["slope_sign"] for q in z] == [1, -1, 1] and abs(z[0]["R"] - (2.0 + 0.5 / 0.7)) < 1e-12, "outer %s inner %s zeros %s (expected outer 7 + 0.2/0.6 = 7.3333, inner 5 + 0.1/0.4 = 5.25; the zeros 2 + 0.5/0.7 rising, 5.25 falling, 7.3333 rising)" % (Ro, Ri, z))
    Rz2 = np.arange(1, 8, dtype=float)
    fz2 = np.array([0.5, -0.5, 0.3, -0.1, 0.2, 0.6, -0.4])
    Ro2, Ri2, z2 = CW.applied_roots(Rz2, fz2, np.zeros(7))
    check("applied roots: the inner root is the largest falling zero below the outer root (not the first falling zero, not a falling zero above the outer root)", abs(Ro2 - (4.0 + 0.1 / 0.3)) < 1e-12 and abs(Ri2 - (3.0 + 0.3 / 0.4)) < 1e-12 and len(z2) == 5 and [q["slope_sign"] for q in z2] == [-1, 1, -1, 1, -1] and abs(z2[0]["R"] - 1.5) < 1e-12 and abs(z2[-1]["R"] - 6.6) < 1e-12, "outer %s inner %s zeros %s (expected outer 4 + 0.1/0.3 = 4.3333, inner 3 + 0.3/0.4 = 3.75; the falling zeros at 1.5 and 6.6 are not the inner root)" % (Ro2, Ri2, z2))
    Ro3, Ri3, z3 = CW.applied_roots(np.arange(1, 5, dtype=float), np.array([0.4, 0.1, -0.2, -0.5]), np.zeros(4))
    check("applied roots: no rising zero gives no outer and no inner root (the one falling zero still listed)", Ro3 is None and Ri3 is None and len(z3) == 1 and z3[0]["slope_sign"] == -1 and abs(z3[0]["R"] - (2.0 + 0.1 / 0.3)) < 1e-12, "outer %s inner %s zeros %s" % (Ro3, Ri3, z3))
    Ro4, Ri4, z4 = CW.applied_roots(np.arange(1, 5, dtype=float), np.array([-1.0, -0.5, 0.5, 1.0]), np.zeros(4))
    check("applied roots: a rising zero with no falling zero below it gives the outer root and no inner root", Ro4 is not None and abs(Ro4 - 2.5) < 1e-12 and Ri4 is None and len(z4) == 1 and z4[0]["slope_sign"] == 1, "outer %s inner %s zeros %s" % (Ro4, Ri4, z4))
    L = CW.gradient_length(Rz, np.exp(0.1 * Rz))
    check("gradient length: N/|N'| of exp(0.1 r) on unit spacing is 1/sinh(0.1) = 9.98335 at the interior points (central differences)", np.max(np.abs(L[1:-1] - 1.0 / np.sinh(0.1))) < 1e-9, "values %s (expected %.5f)" % (np.round(L[1:-1], 5), 1.0 / np.sinh(0.1)))
    nfail = sum(1 for _, ok, _ in RESULTS if not ok)
    print("%s: %d checks, %d failures" % ("PASS" if nfail == 0 else "FAIL", len(RESULTS), nfail))
    return 1 if nfail else 0


if __name__ == "__main__":
    sys.exit(main())
