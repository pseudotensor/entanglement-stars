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
import os
import time

import numpy as np

os.environ.setdefault("JAX_PLATFORM_NAME", "gpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit, jacfwd

print(f"JAX devices: {jax.devices()}", flush=True)

beta0, m0, csq, N = 5.0, 3.0, 5.0, 200
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
    V_defect = jnp.maximum(m0 * (1.0 - lapse) - mu, 0.0)
    _, Gs_s = _bond_correlations(diag0, off)
    rho_s = jnp.zeros(N).at[:-1].add(2.0 * t0 * Nbar_abs * Gs_s * g[:-1])
    Gd_t, Gs_t = _bond_correlations(V_defect, off)
    rho_t = (jnp.zeros(N).at[:-1].add(2.0 * t0 * Nbar_abs * Gs_t * g[:-1])
             + V_defect * Gd_t * g)
    F = lhs - (beta0 / csq) * (rho_s - rho_t)
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
mus = sorted(float(m) for m in d["mu_values"]
             if f"mu{float(m):.2f}_Phi" in d and physical(np.array(d[f"mu{float(m):.2f}_Phi"])))
Phi_lo = np.array(d[f"mu{mus[-1]:.2f}_Phi"])
step = 0.01
mu = mus[-1]
while step > 2e-7:
    Phi_try, nrm, ok = newton(Phi_lo, mu + step)
    if ok and physical(Phi_try):
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
w, V = np.linalg.eig(J)
idx = np.argsort(np.abs(w))
lam = w[idx[:4]].real
print(f"\n  four smallest |eigenvalues| of F_Phi: "
      + ", ".join(f"{x:.5g}" for x in np.abs(w[idx[:4]])))
print(f"  (i)  simplicity: |lam_2/lam_1| = {abs(w[idx[1]])/abs(w[idx[0]]):.1f}")

v = np.real(V[:, idx[0]])
v /= np.linalg.norm(v)
wl, Vl = np.linalg.eig(J.T)
il = np.argsort(np.abs(wl))
psi = np.real(Vl[:, il[0]])
psi /= np.linalg.norm(psi)
# np.linalg.eig fixes eigenvectors only up to sign, and <psi,F_mu> and
# <psi,F_PhiPhi[v,v]> are each LINEAR in psi, so both flip with that arbitrary
# phase (their ratio, the fold coefficient below, does not).  Orient psi against
# v -- the standard Crandall-Rabinowitz convention -- so reruns reproduce.
if psi @ v < 0:
    psi = -psi

Fmu = np.array(jacM(jnp.array(Phi_star), jnp.float64(mu_star)))
tr = float(psi @ Fmu)
print(f"  (ii) transversality <psi, F_mu>       = {tr:+.6g}"
      f"   (|<psi,v>| = {abs(psi@v):.4f})")

# second directional derivative by central differences of the residual, at
# several step sizes (the value must be h-independent to be real)
F0 = np.array(resid(jnp.array(Phi_star), jnp.float64(mu_star)))
quad = np.nan
for h in [1e-2, 3e-3, 1e-3, 3e-4, 1e-4]:
    Fp = np.array(resid(jnp.array(Phi_star + h * v), jnp.float64(mu_star)))
    Fm = np.array(resid(jnp.array(Phi_star - h * v), jnp.float64(mu_star)))
    q = float(psi @ ((Fp - 2 * F0 + Fm) / h ** 2))
    print(f"  (iii) h={h:8.0e}  <psi, F_PhiPhi[v,v]> = {q:+.6g}")
    if h == 1e-3:
        quad = q
if abs(tr) > 0 and abs(quad) > 0:
    print(f"  => quadratic fold, mu - mu* = -(1/2) (<psi,F_vv>/<psi,F_mu>) s^2 "
          f"with coefficient {-0.5*quad/tr:+.4g}")

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
for mu_t in mus[::4] + [mu_star]:
    Phi_t = (np.array(d[f"mu{mu_t:.2f}_Phi"]) if f"mu{mu_t:.2f}_Phi" in d
             else Phi_star)
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
    p_half = rs * Nstar ** 2          # (r_s/a) delta^2 = p - 1/2
    c = Nin / Nstar
    rows.append((mu_t / m0, rs, ns + 1, Nstar, c, p_half,
                 np.inf if c >= 1 else 1.0 / (1.0 - c ** 2)))
    print(f"  mu/m0={mu_t/m0:.4f}  r_s/a={rs:7.3f}  n_*={ns+1:3d}  "
          f"N_*={Nstar:.4f}  c={c:.4f}  p-1/2={p_half:.4f}  "
          f"bound 1/(1-c^2)={rows[-1][6]:.3f}", flush=True)

R = np.array(rows, dtype=float) if rows else np.zeros((0, 7))
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
