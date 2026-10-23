#!/usr/bin/env python3
"""obs_bw_modular_v1.py (lead's first draft; SPEC_final4b_observables.md sec. 2; J2, runs now on the CPU).

The Bisognano-Wichmann check with the flow term on the 2-component chain (the exact transverse decomposition of the 3-dimensional planar
half-space at transverse mass m_c).  Chain: sites n = 0..L-1, on-site m_c SZ, sine hop -(i/2) SX, flow hop -(i/4) (v_n + v_{n+2})/2;
the ground state by eigh; A = the right half, the cut at x_c = L/2 - 1/2; the one-particle modular Hamiltonian k_A = log((1 - C_A)/C_A).
Predictions (the lead's derivation, sec. 2.1): uniform v (|v| < 1): the state is the flat sea and k_A -> 2 pi (1/2){X - x_c, h_0}|_A at long
wavelengths; slowly varying v: k_A -> 2 pi [(1/2){b^t(X), h_0} + (1/2){g(X), p}]|_A with b^t = (x* - x*_c) - (f - f_c) v, g = (x* - x*_c) v -
(f - f_c), x*' = 1/(1 - v^2), f' = v/(1 - v^2) (massless; the massive chain's deviation is measured).  Comparators: the uniform boost and the
naive flowing boost 2 pi (1/2){X - x_c, h_0 + (1/2){v, p}}.  Observables: the lowest 24 |eigenvalues|, the Fermi-Dirac test on the boost's
eigenvectors (occupation against 1/(e^{2 pi beta} + 1)), the Rayleigh quotients, the low-subspace residual.  BUILDER: the profile's integrals
x*, f on the continuous profile (midpoint rule on the lattice is the draft), the L ladder, the markdown writer."""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse, json, math, time
from multiprocessing import Pool
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.linalg import eigh

SX = np.array([[0, 1], [1, 0]], complex)
SZ = np.diag([1.0, -1.0]).astype(complex)
I2 = np.eye(2, dtype=complex)


def sym_weight(f, M):
    """(1/2){f(X), M}: elementwise (f_i + f_j)/2 M_ij with the site index mode//2 (the midpoint rule for every hop)."""
    F = np.repeat(np.asarray(f, float), 2)
    return 0.5 * (F[:, None] + F[None, :]) * M


def chain_ops(L, mc):
    """h_0 (sine Dirac chain, N = 1, v = 0) and p (the two-step current at unit flow)."""
    h0 = np.zeros((2 * L, 2 * L), complex)
    p = np.zeros_like(h0)
    for n in range(L):
        h0[2 * n:2 * n + 2, 2 * n:2 * n + 2] = mc * SZ
    for n in range(L - 1):
        h0[2 * n:2 * n + 2, 2 * n + 2:2 * n + 4] = -0.5j * SX
    for n in range(L - 2):
        p[2 * n:2 * n + 2, 2 * n + 4:2 * n + 6] = -0.25j * I2
    h0 = np.triu(h0) + np.triu(h0, 1).conj().T
    p = np.triu(p) + np.triu(p, 1).conj().T
    return h0, p


def ground_projector(h):
    e, Q = eigh(h, driver="evr", check_finite=False)
    V = Q[:, e < 0]
    return V @ V.conj().T, e


def modular_one_particle(CA, clip=1e-15):
    lam, U = eigh(CA, driver="evr", check_finite=False)
    lam = np.clip(lam, clip, 1 - clip)
    eps = np.log((1 - lam) / lam)
    return (U * eps) @ U.conj().T, eps


def chart_functions(x, v, xc):
    """x*(x) - x*_c and f(x) - f_c by the trapezoid rule on the lattice samples (BUILDER: the continuous profile instead)."""
    x = np.asarray(x, float)
    v = np.asarray(v, float)
    ic = np.searchsorted(x, xc)
    def cum(y):
        out = np.zeros_like(x)
        dx = np.diff(x)
        out[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * dx)
        # reference at xc by linear interpolation between ic - 1 and ic
        ref = out[ic - 1] + (out[ic] - out[ic - 1]) * (xc - x[ic - 1]) / (x[ic] - x[ic - 1])
        return out - ref
    xs = cum(1.0 / (1.0 - v * v))
    f = cum(v / (1.0 - v * v))
    return xs, f


def chart_functions_cont(vfun, x, xc, ngl=8):
    """builder: x*(x) - x*_c and f(x) - f_c on the continuous profile (x*' = 1/(1 - v^2), f' = v/(1 - v^2)): Gauss-Legendre with ngl nodes on
    every unit interval, cumulated from x[0]; the reference at xc by the same rule from the site below xc (exact for uniform v)."""
    tg, wg = leggauss(ngl)

    def seg(lo, hi):
        xm = 0.5 * (hi - lo) * tg + 0.5 * (hi + lo)
        vv = np.asarray(vfun(xm), float)
        jac = 0.5 * (hi - lo)
        return jac * float(np.sum(wg / (1 - vv * vv))), jac * float(np.sum(wg * vv / (1 - vv * vv)))
    x = np.asarray(x, float)
    XS = np.zeros_like(x)
    F = np.zeros_like(x)
    for i in range(len(x) - 1):
        s1, s2 = seg(x[i], x[i + 1])
        XS[i + 1] = XS[i] + s1
        F[i + 1] = F[i] + s2
    ic = int(np.searchsorted(x, xc))
    r1, r2 = seg(x[ic - 1], xc)
    return XS - (XS[ic - 1] + r1), F - (F[ic - 1] + r2)


def boost_operators(L, mc, v, xc, vfun=None):
    """2 pi times the one-particle boost operators: the apex boost (prediction), the uniform boost, the naive flowing boost."""
    h0, p = chain_ops(L, mc)
    x = np.arange(L, dtype=float)
    hv = sym_weight(v, p)
    h = h0 + hv
    if vfun is not None:
        xs, f = chart_functions_cont(vfun, x, xc)          # builder: the continuous profile
    else:
        xs, f = chart_functions(x, v, xc)
    bt = xs - f * v
    g = xs * v - f
    b_pred = 2 * math.pi * (sym_weight(bt, h0) + sym_weight(g, p))
    b_unif = 2 * math.pi * sym_weight(x - xc, h0)
    b_naive = 2 * math.pi * sym_weight(x - xc, h)
    return h, h0, b_pred, b_unif, b_naive


def compare(kA, CA, bA, beta_max=3.0, nlow=24):
    beta, phi = eigh(bA, driver="evr", check_finite=False)
    eps = eigh(kA, eigvals_only=True, driver="evr", check_finite=False)
    low = np.argsort(np.abs(beta))[:nlow]
    sel = np.abs(beta) < beta_max
    P = phi[:, sel]
    occ = np.real(np.einsum("in,ij,jn->n", P.conj(), CA, P))
    ray = np.real(np.einsum("in,ij,jn->n", P.conj(), kA, P))
    fd = 1.0 / (np.exp(np.clip(beta[sel], -700, 700)) + 1.0)
    D = P.conj().T @ (kA - bA) @ P
    B = P.conj().T @ bA @ P
    return {"beta_low": np.sort(beta[low]).tolist(), "eps_low": np.sort(eps[np.argsort(np.abs(eps))[:nlow]]).tolist(),
            "fd_test": {"beta": beta[sel].tolist(), "occupation": occ.tolist(), "fermi_dirac": fd.tolist(),
                        "max_abs_diff": float(np.max(np.abs(occ - fd))) if sel.any() else None},
            "rayleigh": {"beta": beta[sel].tolist(), "quotient_over_2pi": (ray / (2 * math.pi)).tolist(),
                         "max_rel_diff": float(np.max(np.abs(ray - beta[sel]) / np.maximum(np.abs(beta[sel]), 1e-3))) if sel.any() else None},
            "low_subspace_residual": float(np.linalg.norm(D) / max(np.linalg.norm(B), 1e-300)), "n_low": int(sel.sum())}


def make_vfun(spec):
    """builder: the profiles as picklable specs: ("uniform", vu) or ("tanh", v0, v1, W, xp)."""
    if spec[0] == "uniform":
        return lambda x, vu=spec[1]: np.full_like(np.asarray(x, float), vu)
    _, v0, v1, W, xp = spec
    return lambda x: v0 + v1 * np.tanh((np.asarray(x, float) - xp) / W)


def run_case(L, mc, vfun, label):
    t0 = time.time()
    x = np.arange(L, dtype=float)
    xc = L / 2 - 0.5
    v = np.asarray(vfun(x), float)
    h, h0, b_pred, b_unif, b_naive = boost_operators(L, mc, v, xc, vfun=vfun)
    C, e = ground_projector(h)
    a = L  # mode index of the first site of A (site L/2)
    CA = C[a:, a:]
    kA, eps = modular_one_particle(CA)
    out = {"label": label, "L": L, "mc": mc, "xc": xc, "gap_min_abs_e": float(np.min(np.abs(e))), "seconds": None}
    for name, b in (("pred", b_pred), ("uniform", b_unif), ("naive", b_naive)):
        out[name] = compare(kA, CA, b[a:, a:])
    # control: the uniform-v sea against the v = 0 sea in the bulk (middle half of the chain)
    if np.ptp(v) < 1e-12:
        C0, _ = ground_projector(h0)
        mid = slice(L // 2, 3 * L // 2)
        out["sea_bulk_diff_max"] = float(np.max(np.abs(C[mid, mid] - C0[mid, mid])))
        out["sea_full_diff_max"] = float(np.max(np.abs(C - C0)))
    out["seconds"] = time.time() - t0
    return out


def job(args):
    L, mc, spec, label = args
    return run_case(L, mc, make_vfun(spec), label)


def write_md(res, path):
    L_ = ["## J2: the Bisognano-Wichmann check with the flow term (obs_bw_modular_v1.py)", "",
          "Low-subspace residual ||P(k_A - b_A)P||/||P b_A P|| on the boost's eigenvectors with |beta| < 3; the Fermi-Dirac test max |occ - 1/(e^beta + 1)| and the Rayleigh max relative difference for the prediction (the apex boost).", "",
          "| case | L | m_c | residual: prediction | uniform boost | naive flowing boost | FD test (prediction) | Rayleigh (prediction) | n_low |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in res:
        L_.append("| %s | %d | %g | %.3e | %.3e | %.3e | %s | %s | %d |" % (
            r["label"], r["L"], r["mc"], r["pred"]["low_subspace_residual"], r["uniform"]["low_subspace_residual"], r["naive"]["low_subspace_residual"],
            "%.2e" % r["pred"]["fd_test"]["max_abs_diff"] if r["pred"]["fd_test"]["max_abs_diff"] is not None else "-",
            "%.2e" % r["pred"]["rayleigh"]["max_rel_diff"] if r["pred"]["rayleigh"]["max_rel_diff"] is not None else "-", r["pred"]["n_low"]))
    sea = [r for r in res if "sea_bulk_diff_max" in r]
    if sea:
        L_ += ["", "- the uniform-v sea against the v = 0 sea: bulk max |C - C_0| %.2e, whole chain %.2e (largest over the uniform cases)." % (
            max(r["sea_bulk_diff_max"] for r in sea), max(r["sea_full_diff_max"] for r in sea))]
    L_ += ["- every number above is read from the cases by the script."]
    open(path, "w").write("\n".join(L_) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, nargs="+", default=[512, 1024, 2048])      # builder: the L ladder
    ap.add_argument("--mc", type=float, nargs="+", default=[0.0, 0.05, 0.2])
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--md", required=True)
    a = ap.parse_args()
    tasks = []
    for L in a.L:
        for mc in a.mc:
            for vu in (0.0, 0.3, 0.6, 0.9):
                tasks.append((L, mc, ("uniform", vu), "uniform v = %g" % vu))
            for v0, v1, W in ((-0.5, 0.3, 32.0), (-0.5, 0.3, 64.0), (-0.5, 0.3, 128.0), (0.0, 0.6, 64.0)):
                for shift in (0.0, W):
                    xp = L / 2 - 0.5 - shift
                    tasks.append((L, mc, ("tanh", v0, v1, W, xp), "tanh v0 %g v1 %g W %g, cut at centre + %g" % (v0, v1, W, shift)))
    res = []
    with Pool(a.workers) as pool:
        for r in pool.imap(job, tasks):
            res.append(r)
            print(r["label"], r["L"], r["mc"], "%.0f s" % r["seconds"], {n: "%.3e" % r[n]["low_subspace_residual"] for n in ("pred", "uniform", "naive")}, flush=True)
            json.dump({"script": "obs_bw_modular_v1.py", "cases": res}, open(a.out_json, "w"), indent=1, default=float)
    write_md(res, a.md)


if __name__ == "__main__":
    main()
