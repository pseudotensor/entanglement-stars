#!/usr/bin/env python3
"""Saddle-node test at the TK fold, plus the surface-offset bound.

`det J = 0` alone does not establish a saddle-node.  For F(Phi, mu) = 0 the
Crandall-Rabinowitz conditions at (Phi*, mu*) are

  (i)   dim ker F_Phi = 1                       (simple zero eigenvalue),
  (ii)  <psi, F_mu> != 0                        (transversality in the parameter),
  (iii) <psi, F_PhiPhi[v, v]> != 0              (quadratic non-degeneracy),

with v the right and psi the left null vector.  Then the solution set near
(Phi*, mu*) is a smooth curve with a quadratic turning point, and mu attains a
local maximum there.  This script measures all three, then runs pseudo-arclength
continuation through the point to check that the branch turns instead of ending,
and finally deflates against the two known solutions at several mu > mu* to look
for disconnected branches.

It also measures the surface offset p := n_* - r_s/a of eq:tov-delta-scaling and
the one-shell interior lapse ratio c := N_{n_*-1}/N_*, which bound it:
positivity of the surface lapse gives p - 1/2 = (r_s/a) delta^2 > 0, and
monotonicity of the w-flux gives w_{n_*-1} >= w_{n_*} - a/r_s, i.e.
p - 1/2 <= 1/(1 - c^2).  Both bounds are r_s-independent.

Data -> fold_transversality.npz
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

# m0/t0 = 0.03, not 3.0: the published mass was chosen against a closure
# prefactor 37x too small and admits no star at the correct P = 36.825.
# See the parameter block of tov_tk_chemical_el.py.  The npz mu keys are
# formatted %.6f for the same reason -- at m0 = 0.03 every physical mu
# rounds to 0.00 under the old %.2f.

os.environ.setdefault("JAX_PLATFORM_NAME", "gpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit, jacfwd

print(f"JAX devices: {jax.devices()}", flush=True)

beta0, m0, csq, N = 5.0, 0.03, 5.0, 200
t0, N_floor = 1.0, 0.005
n_arr = jnp.arange(1, N + 1, dtype=jnp.float64)
g = 4.0 * jnp.pi * n_arr ** 2
diag0 = jnp.zeros(N, dtype=jnp.float64)


def _bkm_cov(diag_vals, off_vals):
    H = jnp.diag(diag_vals) + jnp.diag(off_vals, 1) + jnp.diag(off_vals, -1)
    evals, evecs = jnp.linalg.eigh(H)
    be = jnp.clip(beta0 * evals, -500.0, 500.0)
    f = 1.0 / (jnp.exp(be) + 1.0)
    diff = beta0 * (evals[:, None] - evals[None, :])
    phi = jnp.where(jnp.abs(diff) > 1e-12,
                    jnp.expm1(diff) / jnp.where(jnp.abs(diff) > 1e-12, diff, 1.0), 1.0)
    K = f[:, None] * (1.0 - f[None, :]) * phi
    U_n, U_n1 = evecs[:-1, :], evecs[1:, :]
    return (jnp.sum((U_n1 ** 2) @ K * (U_n ** 2), axis=1) +
            jnp.sum((U_n ** 2) @ K * (U_n1 ** 2), axis=1) -
            2.0 * jnp.sum((U_n * U_n1) @ K * (U_n * U_n1), axis=1))


def _bond_correlations(diag_vals, off_vals):
    H = jnp.diag(diag_vals) + jnp.diag(off_vals, 1) + jnp.diag(off_vals, -1)
    evals, evecs = jnp.linalg.eigh(H)
    be = jnp.clip(beta0 * evals, -500.0, 500.0)
    f = 1.0 / (jnp.exp(be) + 1.0)
    vf = evecs * f[None, :]
    return jnp.sum(evecs * vf, axis=1), jnp.sum(evecs[:-1] * vf[1:], axis=1)

# --- closure prefactor P of eq:sc-closure --------------------------------
# P = c_*^2 t0^2 / (lambda eps_b) with lambda the PER-CHANNEL alpha_S^bond =
# 0.217, and eps_b = 2 t0 Re[i tr(alpha^r G_{n,n+1})] per channel on the FLAT
# reference chain (eq:sc-sumrule), taken at the bulk bond to avoid the open ends
# (bond 1 gives 0.808 and bond 2 gives 0.539 against the bulk 0.6257).
# The scripts previously used the solver shorthand beta0/c_*^2.  For the defect
# model that is a relabelling of V0; here mu enters V = m0(1-N) - mu without the
# prefactor, so it is NOT a relabelling and the fold moves.
#
# lambda is the PER-CHANNEL alpha_S^bond = 0.217, NOT alpha_S = 0.217 D.
# eq:sc-alphaS-required displays lambda = alpha_S, and sec:tov-entropy gives
# alpha_S = D alpha_S^bond, but that combination is not D-consistent here.
# Count D through the paper's own linearized closure (sec:sc-newton): the
# Laplacian carries kappa_n = D g_n t0^2 and the source carries -D V0 g_n/2, so
# D cancels and Phi goes as 1/lambda.  eps_b is defined PER CHANNEL by
# eq:sc-sumrule (the 2 t0/D prefactor), so P = c_*^2 t0^2/(lambda eps_b) is
# D-free only if lambda is.  sec:num-method states outright that D "is carried
# by both sides and so is omitted throughout", and sec:shell-dirichlet that D
# "cancels from the closure" -- both require the per-channel value.  Using
# 0.217 D makes P too small by exactly D = 4.
LAM_ALPHA_S = 0.217


def _closure_prefactor():
    _, _Gs_flat = _bond_correlations(diag0, -t0 * jnp.ones(N - 1, dtype=jnp.float64))
    _eps_b = 2.0 * t0 * _Gs_flat[(N - 1) // 2]
    return csq * t0 ** 2 / (LAM_ALPHA_S * _eps_b)


off_bg = -t0 * jnp.ones(N - 1, dtype=jnp.float64)
cov_bg = jnp.maximum(_bkm_cov(diag0, off_bg), 1e-30)


def _energy(Phi, mu):
    lapse = jnp.maximum(1.0 + Phi / csq, N_floor)
    Nbar = 0.5 * (lapse[:-1] + lapse[1:])
    off = -t0 * jnp.abs(Nbar)
    kappa = g[:-1] * t0 ** 2 * Nbar ** 2 * (_bkm_cov(diag0, off) / cov_bg)
    dPhi = Phi[:-1] - Phi[1:]
    return 0.5 * jnp.sum(kappa * dPhi ** 2)


_grad_energy = jax.grad(_energy, argnums=0)


def _residual(Phi, mu):
    lhs = _grad_energy(Phi, mu)
    lapse = jnp.maximum(1.0 + Phi / csq, N_floor)
    Nbar_abs = jnp.abs(0.5 * (lapse[:-1] + lapse[1:]))
    off = -t0 * Nbar_abs
    # Source convention (eq. sc-rho, sec:ex-observable, compute_exact_data_gpu.py):
    # the bond term is NEGATIVE against a positive V_defect term, carries ONE power
    # of the lapse (the bare shell observable -- the smeared correlator Gs already
    # supplies it, and a second factor would count the lapse twice), and is split
    # half onto each endpoint of the bond.
    V_defect = jnp.maximum(m0 * (1.0 - lapse) - mu, 0.0)
    _, Gs_s = _bond_correlations(diag0, off)
    rho_s = jnp.zeros(N).at[:-1].add(-1.0 * t0 * Gs_s * g[:-1]).at[1:].add(-1.0 * t0 * Gs_s * g[:-1])
    Gd_t, Gs_t = _bond_correlations(V_defect, off)
    rho_t = (jnp.zeros(N).at[:-1].add(-1.0 * t0 * Gs_t * g[:-1]).at[1:].add(-1.0 * t0 * Gs_t * g[:-1])
             + V_defect * Gd_t * g)
    F = lhs - _closure_prefactor() * (rho_s - rho_t)
    return F.at[N - 1].set(Phi[N - 1])


print("JIT compiling...", flush=True)
t_jit = time.time()
resid = jit(_residual)
jacP = jit(jacfwd(_residual, argnums=0))
jacM = jit(jacfwd(_residual, argnums=1))
_d, _m = jnp.zeros(N), jnp.float64(0.1)
for f in (resid, jacP, jacM):
    jax.block_until_ready(f(_d, _m))
print(f"JIT done in {time.time()-t_jit:.1f}s", flush=True)


def newton(Phi0, mu, tol=1e-9, it_max=60):
    Phi = jnp.array(Phi0, dtype=jnp.float64)
    mj = jnp.float64(mu)
    for _ in range(it_max):
        F = np.array(resid(Phi, mj))
        nrm = float(np.max(np.abs(F)))
        if nrm < tol:
            return np.array(Phi), nrm, True
        J = np.array(jacP(Phi, mj))
        try:
            d = np.linalg.solve(J, -F)
        except np.linalg.LinAlgError:
            return np.array(Phi), nrm, False
        al = 1.0
        for _ in range(10):
            Pt = Phi + al * jnp.array(d)
            if np.max(np.abs(np.array(resid(Pt, mj)))) < nrm:
                Phi = Pt
                break
            al *= 0.5
        else:
            Phi = Phi + al * jnp.array(d)
    F = np.array(resid(Phi, mj))
    nrm = float(np.max(np.abs(F)))
    return np.array(Phi), nrm, nrm < tol


def nontrivial(Phi):
    return float(np.max(np.abs(Phi))) > 0.01


def physical(Phi):
    """Nontrivial AND on the positive-lapse branch, with the N_floor clamp inactive."""
    return nontrivial(Phi) and float(np.min(1.0 + np.asarray(Phi) / csq)) > 1.5 * N_floor


# ---------------------------------------------------------------- fold search
print("\n=== locating the fold by bisection in mu ===", flush=True)
d = np.load("tov_tk_chemical_el.npz", allow_pickle=True)
# STAR branch only: multi-shell and clear of the lapse floor by 20x.  physical()
# alone admits the floor-pinned n_eff=1 family, and bisecting from there walks up
# to mu/m0 ~ 0.97 and stops at the clamp, not at a fold.
def _star(mu, Phi):
    lap = 1.0 + np.asarray(Phi) / csq
    ne = int(np.sum(np.maximum(m0 * (1.0 - lap) - mu, 0.0) > 1e-10))
    return nontrivial(Phi) and ne >= 2 and float(np.min(lap)) > 0.1

mus = sorted(float(m) for m in d["mu_values"]
             if f"mu{float(m):.6f}_Phi" in d
             and _star(float(m), np.array(d[f"mu{float(m):.6f}_Phi"])))
if not mus:
    sys.exit("no star-branch solution to seed from")
Phi_lo = np.array(d[f"mu{mus[-1]:.6f}_Phi"])
# Step RELATIVE to m0.  This was the absolute 0.01, written when m0/t0 was 3.0.
# At m0 = 0.03 that is a mu/m0 step of 0.333 -- a third of the whole branch --
# and the sibling fold_continuation_floor.py hit exactly this, reporting "0
# steps" and a NaN eigenvalue instead of a fold.  A continuation step belongs in
# the physical variable mu/m0, not in absolute mu.
step = 0.0033 * m0
mu = mus[-1]
while step > 1e-7 * m0:
    Phi_try, nrm, ok = newton(Phi_lo, mu + step)
    # _star, NOT physical.  The seeds above were filtered to the star branch but
    # the bisection was not, so it accepted any solution clear of 1.5*N_floor --
    # i.e. the floor-pinned n_eff=1 family -- and walked to mu/m0 = 0.96684,
    # stopping at min lapse = 0.0075000, exactly its own acceptance cutoff.  That
    # endpoint is the guard, not a bifurcation, and it is what the committed
    # fold_transversality.npz contains.
    if ok and _star(mu + step, Phi_try):
        mu, Phi_lo = mu + step, Phi_try
        print(f"    mu={mu:.6f}  min N={float(np.min(1+Phi_lo/csq)):.5f}", flush=True)
    else:
        step *= 0.5
mu_star, Phi_star = mu, Phi_lo
print(f"  last positive-lapse solution: mu = {mu_star:.6f}  "
      f"(mu/m0 = {mu_star/m0:.5f}), min N = {float(np.min(1+Phi_star/csq)):.5f}",
      flush=True)

# ------------------------------------------------- Crandall-Rabinowitz tests
J = np.array(jacP(jnp.array(Phi_star), jnp.float64(mu_star)))
# Reduce out the Dirichlet row.  F[N-1] = Phi[N-1] makes row N-1 of J the unit
# vector e_{N-1}, contributing an eigenvalue of exactly 1.  Diagonalizing the
# full J therefore reported that boundary artifact as lam_2 -- the committed
# npz has eigs[1] = 1.00000000 -- so the simplicity ratio was measured against
# a mode that is not physical.  The free variables are Phi[0..N-2].
Jr = J[:-1, :-1]
w, V = np.linalg.eig(Jr)
idx = np.argsort(np.abs(w))
print(f"\n  four smallest |eigenvalues| of F_Phi (Dirichlet row removed): "
      + ", ".join(f"{x:.5g}" for x in np.abs(w[idx[:4]])))
lam1 = abs(w[idx[0]])
print(f"  (i)  simplicity: |lam_2/lam_1| = {abs(w[idx[1]])/lam1:.1f}")
# A simple eigenvalue is only half of condition (i): it must also be ZERO.
# Printing the ratio alone let a run stop at the lapse clamp with |lam_1| = 0.22
# and still report "quadratic fold".
_minN_end = float(np.min(1 + Phi_star / csq))
_at_clamp = _minN_end < 3.0 * N_floor
print(f"  (i)  vanishing: |lam_1| = {lam1:.4e}  -> "
      f"{'OK, degenerate' if lam1 < 0.05 else 'NOT A FOLD: eigenvalue does not vanish'}")
print(f"       min N at endpoint = {_minN_end:.5f}"
      f"{'   <-- AT THE LAPSE CLAMP, this is a continuation limit not a bifurcation'
         if _at_clamp else ''}")
if lam1 >= 0.05 or _at_clamp:
    print("\n  *** The endpoint is NOT a saddle-node.  The Crandall-Rabinowitz")
    print("      conditions below are reported for completeness but do not apply:")
    print("      (ii) and (iii) are meaningful only at a simple ZERO eigenvalue.")

# Null vectors live in the reduced space; pad v with the pinned Phi[N-1] = 0 so
# perturbing along it keeps the boundary condition satisfied, and contract psi
# against the physical rows F[:-1] only.
v_r = np.real(V[:, idx[0]])
v_r /= np.linalg.norm(v_r)
v = np.concatenate([v_r, [0.0]])
wl, Vl = np.linalg.eig(Jr.T)
il = np.argsort(np.abs(wl))
psi = np.real(Vl[:, il[0]])
psi /= np.linalg.norm(psi)
# np.linalg.eig fixes eigenvectors only up to sign, and <psi,F_mu> and
# <psi,F_PhiPhi[v,v]> are each LINEAR in psi, so both flip with that arbitrary
# phase (their ratio, the fold coefficient below, does not).  Orient psi against
# v -- the standard Crandall-Rabinowitz convention -- so reruns reproduce.
if psi @ v_r < 0:
    psi = -psi

Fmu = np.array(jacM(jnp.array(Phi_star), jnp.float64(mu_star)))
tr = float(psi @ Fmu[:-1])
print(f"  (ii) transversality <psi, F_mu>       = {tr:+.6g}"
      f"   (|<psi,v>| = {abs(psi@v_r):.4f})")

# second directional derivative by central differences of the residual, at
# several step sizes (the value must be h-independent to be real)
F0 = np.array(resid(jnp.array(Phi_star), jnp.float64(mu_star)))
quad = np.nan
_qs = []
for h in [1e-2, 3e-3, 1e-3, 3e-4, 1e-4]:
    Fp = np.array(resid(jnp.array(Phi_star + h * v), jnp.float64(mu_star)))
    Fm = np.array(resid(jnp.array(Phi_star - h * v), jnp.float64(mu_star)))
    q = float(psi @ ((Fp - 2 * F0 + Fm)[:-1] / h ** 2))
    print(f"  (iii) h={h:8.0e}  <psi, F_PhiPhi[v,v]> = {q:+.6g}")
    _qs.append(q)
    if h == 1e-3:
        quad = q
# The old gate was `abs(tr) > 0 and abs(quad) > 0`, which any two nonzero floats
# pass -- it printed "quadratic fold" at an endpoint sitting on the lapse clamp
# with |lam_1| = 0.22.  A second derivative extracted by finite differences is
# only real if it is h-INDEPENDENT, so require a stable plateau, and require the
# eigenvalue to actually vanish off the clamp before calling anything a fold.
_mid = [q for q in _qs[1:4] if np.isfinite(q)]
_stable = (len(_mid) == 3 and abs(max(_mid) - min(_mid)) <=
           0.05 * max(abs(q) for q in _mid))
print(f"  (iii) h-independence over h = 3e-3..3e-4: "
      f"{'stable' if _stable else 'NOT STABLE -- the second derivative is not resolved'}"
      f"  (spread {abs(max(_mid)-min(_mid)):.3g} on {np.mean(np.abs(_mid)):.3g})"
      if _mid else "  (iii) no finite values")
_is_fold = (lam1 < 0.05) and (not _at_clamp) and _stable and abs(tr) > 1e-12
if _is_fold:
    print(f"  => quadratic fold, mu - mu* = -(1/2) (<psi,F_vv>/<psi,F_mu>) s^2 "
          f"with coefficient {-0.5*quad/tr:+.4g}")
else:
    print("  => NOT established as a quadratic fold: "
          + ", ".join(filter(None, [
              None if lam1 < 0.05 else "eigenvalue does not vanish",
              "endpoint on the lapse clamp" if _at_clamp else None,
              None if _stable else "second derivative not h-independent",
              None if abs(tr) > 1e-12 else "transversality vanishes"])))

# ------------------------------------------------ pseudo-arclength turnaround
print("\n=== pseudo-arclength continuation through the point ===", flush=True)
x = np.concatenate([Phi_star, [mu_star]])
# at a fold dmu/ds = 0 and the tangent is the null vector; orient it so the
# lapse minimum decreases, i.e. into the more compact (upper) branch
tvec = np.concatenate([v, [0.0]])
if np.min(1.0 + (Phi_star + 0.05 * v) / csq) > np.min(1.0 + Phi_star / csq):
    tvec = -tvec
tvec /= np.linalg.norm(tvec)
ds = 0.05
arc = []
for k in range(120):
    xp = x + ds * tvec
    y = xp.copy()
    for _ in range(40):
        Fy = np.array(resid(jnp.array(y[:-1]), jnp.float64(y[-1])))
        g_arc = float(tvec @ (y - x) - ds)
        res = np.concatenate([Fy, [g_arc]])
        if np.max(np.abs(res)) < 1e-9:
            break
        A = np.zeros((N + 1, N + 1))
        A[:N, :N] = np.array(jacP(jnp.array(y[:-1]), jnp.float64(y[-1])))
        A[:N, N] = np.array(jacM(jnp.array(y[:-1]), jnp.float64(y[-1])))
        A[N, :] = tvec
        try:
            y = y - np.linalg.solve(A, res)
        except np.linalg.LinAlgError:
            break
    else:
        break
    if np.max(np.abs(np.array(resid(jnp.array(y[:-1]), jnp.float64(y[-1]))))) > 1e-8:
        break
    tnew = y - x
    tnew /= np.linalg.norm(tnew)
    x, tvec = y, tnew
    arc.append((x[-1], float(np.min(1.0 + x[:-1] / csq))))
arc = np.array(arc) if arc else np.zeros((0, 2))
if arc.size:
    k = int(np.argmax(arc[:, 0]))
    print(f"  {len(arc)} continuation points; mu max = {arc[k,0]:.6f} "
          f"(mu/m0 = {arc[k,0]/m0:.5f}) at step {k}")
    print(f"  mu at the end of the run = {arc[-1,0]:.6f}; "
          f"turned = {k < len(arc)-1}")
    print(f"  min lapse across the arc: {arc[:,1].min():.4f} -> {arc[:,1].max():.4f}")
else:
    # This silently produced nothing in the committed run: arc_mu was saved with
    # size 0 and the `if arc.size:` guard printed no line at all, so a test that
    # generated no data reported no failure.  Say so.
    print("  *** PSEUDO-ARCLENGTH PRODUCED ZERO POINTS -- the corrector failed on "
          "the very first step.\n"
          "      This is not evidence either way about turning; it is a failed "
          "test.  Do not\n"
          "      quote a turnaround from this run.")

# --------------------------------------------------- disconnected-branch scan
print("\n=== random restarts at mu > mu* (disconnected-branch search) ===",
      flush=True)
rng = np.random.default_rng(0)
found = []
for mu_t in [mu_star * 1.01, mu_star * 1.05, mu_star * 1.20]:
    hits = 0
    for _ in range(24):
        amp = 10.0 ** rng.uniform(-1.5, 1.2)
        wdt = rng.uniform(2.0, 60.0)
        seed = -amp * np.exp(-(np.arange(1, N + 1) / wdt) ** 2)
        seed[-1] = 0.0
        Ph, nrm, ok = newton(seed, mu_t, it_max=30)
        if ok and physical(Ph):
            hits += 1
    found.append((mu_t, hits))
    print(f"  mu/m0 = {mu_t/m0:.4f}: {hits}/24 random starts reached a "
          f"positive-lapse solution", flush=True)

# --------------------------------------------- surface offset p and its bound
print("\n=== surface offset p and the bound p - 1/2 <= 1/(1-c^2) ===", flush=True)
rows = []
# Pair each mu with its OWN Phi.  The lookup used to be
# `d[f"mu{mu_t:.6f}_Phi"] if key in d else Phi_star`, and mu_star rounds to a key
# that already exists ("mu2.90_Phi" for mu_star = 2.9005), so the fold row silently
# used a different solution -- ||Phi* - Phi(2.90)||_inf = 0.285 -- and rows 13 and
# 14 of the committed p_table are identical in every column but the mu label.
_cases = [(mt, np.array(d[f"mu{mt:.6f}_Phi"])) for mt in mus[::4]] + \
         [(mu_star, Phi_star)]
for mu_t, Phi_t in _cases:
    lapse = np.maximum(1.0 + Phi_t / csq, N_floor)
    Vd = np.maximum(m0 * (1.0 - lapse) - mu_t, 0.0)
    bound = np.where(Vd > 1e-10)[0]
    if bound.size == 0 or bound[-1] + 1 >= N:
        continue
    ns = bound[-1] + 1                      # first V = 0 shell (0-based)
    Nstar, Nin = lapse[ns], lapse[ns - 1]
    # r_s from the exterior w-fit on a clean window
    sl = slice(ns + 3, min(N // 2, ns + 43))
    nn = np.arange(1, N + 1, dtype=float)
    wv = lapse ** 2
    A = np.vstack([1.0 / nn[sl] - 1.0 / N, np.ones(sl.stop - sl.start)]).T
    coef, *_ = np.linalg.lstsq(A, wv[sl] - 1.0, rcond=None)
    rs = -coef[0]
    if rs <= 0:
        continue
    # p is DEFINED by eq:tov-delta-scaling as n_* = r_s/a + p, so p - 1/2 is
    # n_* - r_s/a - 1/2.  The identity p - 1/2 = (r_s/a) delta^2 holds only in the
    # near-horizon limit n_* ~ r_s/a, which is nowhere near true on this branch
    # (n_* = 5..11 against r_s/a = 0.16..10), so the proxy alone must not be
    # reported as p.  Both are recorded; the direct one is the definition.
    p_half_proxy = rs * Nstar ** 2          # (r_s/a) delta^2, near-horizon only
    p_half = (ns + 1) - rs - 0.5            # n_* - r_s/a - 1/2, the definition
    c = Nin / Nstar
    rows.append((mu_t / m0, rs, ns + 1, Nstar, c, p_half,
                 np.inf if c >= 1 else 1.0 / (1.0 - c ** 2), p_half_proxy))
    print(f"  mu/m0={mu_t/m0:.4f}  r_s/a={rs:7.3f}  n_*={ns+1:3d}  "
          f"N_*={Nstar:.4f}  c={c:.4f}  p-1/2={p_half:+.4f}  "
          f"(proxy {p_half_proxy:.4f})  bound 1/(1-c^2)={rows[-1][6]:.3f}"
          f"{'   <-- VIOLATES 0 < p-1/2' if p_half <= 0 else ''}"
          f"{'   <-- VIOLATES p-1/2 <= 1/(1-c^2)' if p_half > rows[-1][6] else ''}",
          flush=True)

R = np.array(rows, dtype=float) if rows else np.zeros((0, 8))
if R.size:
    _pv = R[:, 5]
    print(f"\n  p-1/2 across the table: {_pv.min():+.4f} .. {_pv.max():+.4f}"
          f"  (spread factor {abs(_pv.max()/_pv.min()) if _pv.min() != 0 else np.inf:.1f})")
    print(f"  positivity 0 < p-1/2 holds at {int((_pv > 0).sum())}/{len(_pv)} points; "
          f"upper bound holds at {int((_pv <= R[:, 6]).sum())}/{len(_pv)}")
_here = os.path.dirname(os.path.abspath(__file__))
_out = os.path.normpath(os.path.join(_here, "..", "fermion", "numerical", "data", "fold_transversality.npz"))
np.savez(_out, mu_star=mu_star, Phi_star=Phi_star,
         eigs=np.abs(w[idx[:8]]), transversality=tr, quadratic=quad,
         arc_mu=arc[:, 0] if arc.size else np.zeros(0),
         arc_minN=arc[:, 1] if arc.size else np.zeros(0),
         restart_mu=np.array([f[0] for f in found]),
         restart_hits=np.array([f[1] for f in found]),
         p_table=R)
print("\nwrote fold_transversality.npz")
