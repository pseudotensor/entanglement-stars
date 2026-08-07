#!/usr/bin/env python3
"""Klein-faithful TK star: does a localized fixed point exist WITHOUT the sign flip?

WHY THIS FILE EXISTS
--------------------
The audit established that eq:tov-veff's binding potential

    V_n = m0 (1 - N_n) - mu ,   truncated by max(., 0)

is exactly MINUS the coordinate-time grand-canonical on-site coefficient that
sec:ff-tov's own rule (line 22, "any on-site proper energy E contributes to the
coordinate-time Hamiltonian as N_n E") prescribes.  Identifying mu_inf = m0 - mu,
which is forced if mu is to be a chemical potential at all,

    W_n := N_n m0 - mu_inf = N_n m0 - m0 + mu = -[m0(1 - N_n) - mu] = -V_n

exactly, on every shell, with no approximation.  So the paper uses the MAGNITUDE
of W on the shells where W is negative -- the operation sec:sc-maxent line 33
forbids in as many words ("Replacing either by its magnitude would leave rho(n)
the expectation of no operator at all").

The measurable consequence, confirmed on the committed solutions: the core
EMPTIES.  sum_n g_n (Gd_tau - Gd_sigma) = -1.45, -9.50, -23.92, -30.19 at
mu = 0.05, 0.30, 0.60, 0.70, with the innermost shell 94.4% depleted against a
vacuum at exactly half filling.  The shells Klein says should FILL are the shells
the code EMPTIES.  The published object is a density deficit that gravitates
attractively because the sign of the source was set by hand.

WHAT THIS FILE DOES DIFFERENTLY
-------------------------------
Two roles that the paper's V_defect conflates are separated.  That conflation is
harmless in the defect model -- there V0 really IS both the Hamiltonian
coefficient and the on-site energy -- and it is what forces the sign flip once
the on-site term becomes a free energy H - mu N rather than an energy.

  (1) HAMILTONIAN DIAGONAL, which sets the occupations:
        sigma:  mu             (uniform -- the same matter with gravity switched
                                off, i.e. the N -> 1 value of the tau diagonal)
        tau:    N_n m0 - mu_inf = N_n m0 - m0 + mu
      Their difference is m0 (N_n - 1) = -m0 (1 - N_n) <= 0, so the tau level
      sits BELOW the sigma level inside the well and the occupation RISES there.
      Matter condenses.  It vanishes identically as N -> 1, so the two states
      coincide at infinity and the source is localized by Gibbs clustering --
      exponentially, with no max(., 0) truncation anywhere.

  (2) ON-SITE ENERGY in the source, which is what gravitates:
        the Tolman-redshifted rest energy N_n m0, the SAME coefficient in both
      states, so the mismatch comes entirely from the occupations:
        rho_sigma - rho_tau  =  [bond mismatch]  +  N_n m0 g_n (Gd_s - Gd_t) .
      With Gd_t > Gd_s inside, that term is NEGATIVE -- attractive -- without
      any sign being chosen by hand.

The surface is unchanged and still Klein's: matter binds where N_n m0 < mu_inf,
i.e. N_n < 1 - mu/m0 = N_*, which is eq:tov-surface exactly.  That part of the
paper was always faithful; it is the sign and the truncation that were not.

WHAT IT IS FOR
--------------
This does NOT replace the paper's model.  It answers the one open question the
audit could not: whether a Klein-faithful TK star exists at all, and if so
whether it survives the closure prefactor P = c_*^2 t0^2/(lambda eps_b) = 36.825
that the paper's own equation requires -- the coupling at which the published
construction's branch terminates at 4-10%.

Run:  python3 tov_tk_klein_faithful.py [N] [P_override]
"""

# Post-compaction re-read gate.  This script regenerates PUBLISHED data, so the
# protocol in .taskq/fermion_code_ledger.md requires the whole paper to have been
# re-read first.  That ordering was violated on 2026-08-06 and again on
# 2026-08-07; a rule that depends on being remembered has now failed twice, so it
# is enforced here instead.  No-ops on the compute host, which carries no .taskq.
import sys as _sys, os as _os
_gd = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), ".taskq")
_gd2 = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".taskq")
for _d in (_gd2, _gd):
    if _os.path.isfile(_os.path.join(_d, "gate_guard.py")):
        _sys.path.insert(0, _d)
        from gate_guard import require_reread as _rr
        _rr(__file__)
        break
import os
import sys
import time

import numpy as np

os.environ.setdefault("JAX_PLATFORM_NAME", "gpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit, jacfwd

print(f"JAX devices: {jax.devices()}", flush=True)

N = int(sys.argv[1]) if len(sys.argv) > 1 else 120
beta0, m0, csq, t0 = 5.0, 3.0, 5.0, 1.0
N_floor = 0.005
n_arr = jnp.arange(1, N + 1, dtype=jnp.float64)
g = 4.0 * jnp.pi * n_arr ** 2
diag0 = jnp.zeros(N, dtype=jnp.float64)
off_bg = -t0 * jnp.ones(N - 1, dtype=jnp.float64)


def _bond_correlations(diag_vals, off_vals):
    H = jnp.diag(diag_vals) + jnp.diag(off_vals, 1) + jnp.diag(off_vals, -1)
    evals, evecs = jnp.linalg.eigh(H)
    f = 1.0 / (jnp.exp(jnp.clip(beta0 * evals, -500.0, 500.0)) + 1.0)
    vf = evecs * f[None, :]
    return jnp.sum(evecs * vf, axis=1), jnp.sum(evecs[:-1] * vf[1:], axis=1)


def _bkm_cov(diag_vals, off_vals):
    H = jnp.diag(diag_vals) + jnp.diag(off_vals, 1) + jnp.diag(off_vals, -1)
    evals, evecs = jnp.linalg.eigh(H)
    f = 1.0 / (jnp.exp(jnp.clip(beta0 * evals, -500.0, 500.0)) + 1.0)
    diff = beta0 * (evals[:, None] - evals[None, :])
    phi = jnp.where(jnp.abs(diff) > 1e-12,
                    jnp.expm1(diff) / jnp.where(jnp.abs(diff) > 1e-12, diff, 1.0), 1.0)
    K = f[:, None] * (1.0 - f[None, :]) * phi
    U_n, U_n1 = evecs[:-1, :], evecs[1:, :]
    return (jnp.sum((U_n1 ** 2) @ K * (U_n ** 2), axis=1) +
            jnp.sum((U_n ** 2) @ K * (U_n1 ** 2), axis=1) -
            2.0 * jnp.sum((U_n * U_n1) @ K * (U_n * U_n1), axis=1))


cov_bg = jnp.maximum(_bkm_cov(diag0, off_bg), 1e-30)

# Closure prefactor P = c_*^2 t0^2/(lambda eps_b) with the PER-CHANNEL
# lambda = alpha_S^bond = 0.217 (see the D-count in the sibling scripts).
LAM_ALPHA_S = 0.217
_, _Gs_flat = _bond_correlations(diag0, off_bg)
_eps_b = float(2.0 * t0 * _Gs_flat[(N - 1) // 2])
P_FULL = csq * t0 ** 2 / (LAM_ALPHA_S * _eps_b)
PREF = float(sys.argv[2]) if len(sys.argv) > 2 else P_FULL
print(f"N = {N}   P = {PREF:.4f}  (paper's own P = {P_FULL:.4f}; "
      f"solver shorthand beta0/c*^2 = {beta0/csq:.4f}; eps_b = {_eps_b:.5f})", flush=True)


def _energy(Phi):
    lapse = jnp.maximum(1.0 + Phi / csq, N_floor)
    Nbar = 0.5 * (lapse[:-1] + lapse[1:])
    kappa = g[:-1] * t0 ** 2 * Nbar ** 2 * (_bkm_cov(diag0, -t0 * Nbar) / cov_bg)
    dPhi = Phi[:-1] - Phi[1:]
    return 0.5 * jnp.sum(kappa * dPhi ** 2)


_grad_energy = jax.grad(_energy)


def _rho(Gd, Gs, onsite_energy):
    """eq:sc-rho: bond term NEGATIVE and split half onto each endpoint, plus the
    on-site ENERGY density.  Same functional form as the paper's; only what is
    fed in as the on-site energy differs."""
    rho = (jnp.zeros(N, dtype=jnp.float64)
           .at[:-1].add(-1.0 * t0 * Gs * g[:-1])
           .at[1:].add(-1.0 * t0 * Gs * g[:-1]))
    return rho + onsite_energy * Gd * g


def _residual(Phi, mu):
    lhs = _grad_energy(Phi)
    lapse = jnp.maximum(1.0 + Phi / csq, N_floor)
    Nbar = 0.5 * (lapse[:-1] + lapse[1:])
    off = -t0 * Nbar
    mu_inf = m0 - mu

    # (1) Hamiltonian diagonals.  No max(.,0): localization is by clustering.
    diag_s = jnp.full(N, mu, dtype=jnp.float64)        # gravity switched off
    diag_t = lapse * m0 - mu_inf                        # Klein, in coordinate time

    # (2) On-site ENERGY -- the redshifted rest energy, identical in both states,
    #     so the mismatch is carried entirely by the occupations.
    e_on = lapse * m0

    Gd_s, Gs_s = _bond_correlations(diag_s, off)
    Gd_t, Gs_t = _bond_correlations(diag_t, off)
    F = lhs - PREF * (_rho(Gd_s, Gs_s, e_on) - _rho(Gd_t, Gs_t, e_on))
    return F.at[N - 1].set(Phi[N - 1])


resid = jit(_residual)
jacP = jit(jacfwd(_residual, argnums=0))
print("JIT compiling...", flush=True)
_t = time.time()
jax.block_until_ready(resid(jnp.zeros(N), jnp.float64(0.5)))
jax.block_until_ready(jacP(jnp.zeros(N), jnp.float64(0.5)))
print(f"  done in {time.time()-_t:.1f}s", flush=True)


def newton(Phi0, mu, tol=1e-9, it_max=80):
    Phi = jnp.array(Phi0, dtype=jnp.float64)
    mj = jnp.float64(mu)
    for _ in range(it_max):
        F = np.array(resid(Phi, mj))
        nrm = float(np.max(np.abs(F)))
        if nrm < tol:
            return np.array(Phi), nrm
        J = np.array(jacP(Phi, mj))
        try:
            d = np.linalg.solve(J, -F)
        except np.linalg.LinAlgError:
            return np.array(Phi), nrm
        al = 1.0
        for _ in range(10):
            Pt = Phi + al * jnp.array(d)
            if float(jnp.min(1.0 + Pt / csq)) < N_floor:
                al *= 0.5
                continue
            if np.max(np.abs(np.array(resid(Pt, mj)))) < nrm:
                Phi = Pt
                break
            al *= 0.5
        else:
            Phi = Phi + 0.05 * al * jnp.array(d)
    F = np.array(resid(Phi, mj))
    return np.array(Phi), float(np.max(np.abs(F)))


def props(Phi, mu):
    """Occupation excess is the physical core here -- there is no V to count."""
    lapse = np.maximum(1.0 + np.asarray(Phi) / csq, N_floor)
    off = jnp.array(-t0 * 0.5 * (lapse[:-1] + lapse[1:]))
    mu_inf = m0 - mu
    Gd_s, _ = _bond_correlations(jnp.full(N, mu), off)
    Gd_t, _ = _bond_correlations(jnp.array(lapse * m0 - mu_inf), off)
    dn = np.array(Gd_t) - np.array(Gd_s)
    gn = np.array(g)
    return dict(lapse=lapse, dn=dn, N_B=float(np.sum(gn * dn)),
                n_bound=int(np.sum(dn > 1e-6)),
                n_star=int(np.sum(lapse < 1.0 - mu / m0)),
                minN=float(lapse.min()), minPhi=float(np.min(Phi)))


if __name__ == "__main__":
    print(f"\n{'mu/m0':>7} {'|R|':>10} {'minN':>8} {'minPhi':>10} {'N_B':>10} "
          f"{'n_bound':>8} {'innermost dn':>13}   verdict", flush=True)
    prev = None
    for r in [0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]:
        mu = r * m0
        best = None
        seeds = ([prev] if prev is not None else []) + \
                [-a * np.exp(-0.5 * (np.asarray(n_arr) / w) ** 2)
                 for a in (0.05, 0.2, 0.8, 2.0, 5.0) for w in (2.0, 4.0, 8.0, 15.0)]
        for sd in seeds:
            sd = np.asarray(sd, float).copy()
            sd[-1] = 0.0
            if 1 + sd.min() / csq < 1.5 * N_floor:
                continue
            Phi, rn = newton(sd, mu)
            if rn < 1e-9 and abs(np.min(Phi)) > 1e-4:
                if best is None or rn < best[1]:
                    best = (Phi, rn)
            if best is not None and best[1] < 1e-10:
                break
        if best is None:
            print(f"{r:7.3f} {'--':>10} {'-':>8} {'-':>10} {'-':>10} {'-':>8} "
                  f"{'-':>13}   only the trivial branch", flush=True)
            continue
        Phi, rn = best
        p = props(Phi, mu)
        verdict = ("MATTER CONDENSES" if p["N_B"] > 0 else
                   "still a deficit" if p["N_B"] < 0 else "flat")
        print(f"{r:7.3f} {rn:10.2e} {p['minN']:8.5f} {p['minPhi']:10.4f} "
              f"{p['N_B']:10.3f} {p['n_bound']:8d} {p['dn'][0]:13.5f}   {verdict}",
              flush=True)
        prev = Phi.copy()
