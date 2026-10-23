#!/usr/bin/env python
"""packet_v1_common.py (busi lane, reviewer A slot, inbox #354): the angular-channel chain and its clocked Hamiltonian, rebuilt from scratch for the
packet completeness reading and checked against the engine's module (engine_vacflow_v1.chain_sparse, engine_collapse_v5.Chan.clocked_dense,
engine_collapse_v19.clocked_gpu) where that module is importable.

Conventions (fermion_new/05_star.tex "Method and controls"; engine_vacflow_v1.chain_sparse):
  sites r_n = n a, n = 1 .. n_ch (a = 1), two components per site, no hop across the origin, the last site open;
  (T u)_n = u_{n+1};  p_r = (T - T^dag)/(2i);  h0_kappa = m s3 + s1 p_r + s2 kappa/r_n;  Q = (T^2 - T^dag^2)/(4i) (the two-step current);
  h_m[N, v] = (1/2){N, h0} + (1/2){v, Q}, N and v the back-averaged (applied) site fields repeated on both components.
Matrix elements (basis index 2 n + a, a the component):
  on-site  <n,0|h0|n,0> = m, <n,1|h0|n,1> = -m, <n,0|h0|n,1> = -i kappa/r_n, <n,1|h0|n,0> = +i kappa/r_n;
  bond     <n,0|h0|n+1,1> = <n,1|h0|n+1,0> = -i/2 and the conjugates +i/2 (s1 p_r);
  two-step <n,a|Q|n+2,a> = -i/4, <n+2,a|Q|n,a> = +i/4.
"""
import hashlib
import os

import numpy as np
import scipy.sparse as sp


def md5(path, chunk=1 << 24):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def chain(R, m, kappa):
    """the channel's flat chain h0_kappa and its two-step current Q, sparse 2n x 2n (own construction; see the module docstring)."""
    R = np.asarray(R, float)
    n = len(R)
    b = 2 * np.arange(n)
    rows = [b, b + 1, b, b + 1]
    cols = [b, b + 1, b + 1, b]
    vals = [np.full(n, m, complex), np.full(n, -m, complex), -1j * kappa / R, 1j * kappa / R]
    bi = 2 * np.arange(n - 1)
    r2 = np.concatenate([bi, bi + 1]); c2 = np.concatenate([bi + 3, bi + 2]); v2 = np.full(2 * (n - 1), -0.5j)
    rows += [r2, c2]; cols += [c2, r2]; vals += [v2, np.conj(v2)]
    h0 = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(2 * n, 2 * n))
    i2 = np.arange(n - 2)
    rq = np.concatenate([2 * i2, 2 * i2 + 1]); cq = rq + 4
    Q = sp.csr_matrix((np.concatenate([np.full(2 * (n - 2), -0.25j), np.full(2 * (n - 2), 0.25j)]), (np.concatenate([rq, cq]), np.concatenate([cq, rq]))),
                      shape=(2 * n, 2 * n))
    h0.sum_duplicates(); Q.sum_duplicates()
    return h0, Q


def clocked(h0, Q, N, v):
    """h_m[N, v] = (1/2){N, h0} + (1/2){v, Q} with the site fields repeated on both components (engine_collapse_v5.Chan.clocked_dense)."""
    Nd = sp.diags(np.repeat(np.asarray(N, float), 2)); Vd = sp.diags(np.repeat(np.asarray(v, float), 2))
    return (0.5 * (Nd @ h0 + h0 @ Nd) + 0.5 * (Vd @ Q + Q @ Vd)).tocsr()


def check_against_engine(engine_code_dir, R, m, kappas):
    """max |h0 - engine h0|, max |Q - engine Q| per channel, from engine_vacflow_v1.chain_sparse (read-only import); None if not importable."""
    if not engine_code_dir or not os.path.isdir(engine_code_dir):
        return None
    import sys
    sys.path.insert(0, engine_code_dir)
    try:
        import engine_vacflow_v1 as VF
    except Exception as ex:  # noqa: BLE001
        return {"error": repr(ex)}
    out = {"module": os.path.join(engine_code_dir, "engine_vacflow_v1.py"), "md5": md5(os.path.join(engine_code_dir, "engine_vacflow_v1.py"))}
    for k in kappas:
        h0, Q = chain(R, m, k)
        he, Qe = VF.chain_sparse(np.asarray(R, float), m, float(k))
        out[str(k)] = {"dh0": float(abs(h0 - he).max()), "dQ": float(abs(Q - Qe).max()), "nnz_h0": int(h0.nnz), "nnz_h0_engine": int(he.nnz)}
    return out


def site_density(psi, n):
    """|psi|^2 per site (both components summed) for a vector or a block of columns (2n, ncol) -> (n,) or (n, ncol)."""
    if psi.ndim == 1:
        return (np.abs(psi) ** 2).reshape(n, 2).sum(axis=1)
    return (np.abs(psi) ** 2).reshape(n, 2, -1).sum(axis=1)


def expectation(A, psi):
    """Re <psi|A|psi> per column and <psi|A^2|psi> per column (A Hermitian sparse)."""
    Ap = A @ psi
    e1 = np.real(np.sum(np.conj(psi) * Ap, axis=0))
    e2 = np.real(np.sum(np.abs(Ap) ** 2, axis=0))
    return e1, e2
