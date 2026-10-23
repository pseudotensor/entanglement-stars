"""Active-set continuation: follow the TK branch across the free boundary.

WHY
---
The TK residual is only piecewise C^1, because V_n = max(m0(1-N_n) - mu, 0) has a
kink wherever a shell crosses the binding threshold.  Newton assumes smoothness,
so at a crossing it chatters: at mu/m0 = 0.1925 it stalls at |R| = 5.6e-2 with a
Jacobian whose smallest eigenvalue is 0.52 -- well-conditioned, i.e. a solution
exists and Newton simply cannot reach it.  That stall was being read as the end
of the branch, and then as a fold.

FIX (standard for free-boundary / complementarity problems)
-----------------------------------------------------------
Freeze the ACTIVE SET A = {n : shell n is bound}.  With A fixed,
    V_n = (m0(1-N_n) - mu) on A,   V_n = 0 off A
is smooth in Phi -- no max(), no kink -- so Newton converges quadratically.
Then recompute the set the solution implies and repeat until it stops changing.
The fixed point is a genuine solution of the original problem: at convergence
V > 0 everywhere on A and V <= 0 everywhere off it, so the frozen V equals
max(V_raw, 0) exactly.
"""
import os, sys
import numpy as np
os.environ.setdefault("JAX_PLATFORM_NAME", "gpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ["TK_SKIP_REREAD_GATE"] = "1"
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit, jacfwd

beta0, m0, csq, t0, N = 5.0, 0.03, 5.0, 1.0, 200
N_floor = 0.005
n_arr = jnp.arange(1, N + 1, dtype=jnp.float64)
g = 4.0 * jnp.pi * n_arr ** 2
diag0 = jnp.zeros(N, dtype=jnp.float64)
off_bg = -t0 * jnp.ones(N - 1, dtype=jnp.float64)

def _corr(diag, off):
    H = jnp.diag(diag) + jnp.diag(off, 1) + jnp.diag(off, -1)
    ev, U = jnp.linalg.eigh(H)
    f = 1.0 / (jnp.exp(jnp.clip(beta0 * ev, -500., 500.)) + 1.0)
    vf = U * f[None, :]
    return jnp.sum(U * vf, axis=1), jnp.sum(U[:-1] * vf[1:], axis=1)

def _bkm(diag, off):
    H = jnp.diag(diag) + jnp.diag(off, 1) + jnp.diag(off, -1)
    ev, U = jnp.linalg.eigh(H)
    f = 1.0 / (jnp.exp(jnp.clip(beta0 * ev, -500., 500.)) + 1.0)
    dd = beta0 * (ev[:, None] - ev[None, :])
    phi = jnp.where(jnp.abs(dd) > 1e-12, jnp.expm1(dd) / jnp.where(jnp.abs(dd) > 1e-12, dd, 1.0), 1.0)
    K = f[:, None] * (1 - f[None, :]) * phi
    Un, Un1 = U[:-1, :], U[1:, :]
    return (jnp.sum((Un1**2) @ K * (Un**2), 1) + jnp.sum((Un**2) @ K * (Un1**2), 1)
            - 2.0 * jnp.sum((Un * Un1) @ K * (Un * Un1), 1))

cov_bg = jnp.maximum(_bkm(diag0, off_bg), 1e-30)
_, Gs_flat = _corr(diag0, off_bg)
PREF = float(csq * t0**2 / (0.217 * (2.0 * t0 * Gs_flat[(N - 1) // 2])))

def _energy(Phi):
    lap = jnp.maximum(1.0 + Phi / csq, N_floor)
    Nb = 0.5 * (lap[:-1] + lap[1:])
    kap = g[:-1] * t0**2 * Nb**2 * (_bkm(diag0, -t0 * Nb) / cov_bg)
    return 0.5 * jnp.sum(kap * (Phi[:-1] - Phi[1:])**2)
_gradE = jax.grad(_energy)

def _residual(Phi, mu, mask):
    """mask = 1.0 on the ACTIVE (bound) shells, 0.0 elsewhere.  With the mask
    fixed there is no max(), so this is smooth in Phi."""
    lhs = _gradE(Phi)
    lap = jnp.maximum(1.0 + Phi / csq, N_floor)
    off = -t0 * 0.5 * (lap[:-1] + lap[1:])
    V = (m0 * (1.0 - lap) - mu) * mask
    _, Gs_s = _corr(diag0, off)
    rho_s = jnp.zeros(N).at[:-1].add(-t0 * Gs_s * g[:-1]).at[1:].add(-t0 * Gs_s * g[:-1])
    Gd_t, Gs_t = _corr(V, off)
    rho_t = (jnp.zeros(N).at[:-1].add(-t0 * Gs_t * g[:-1]).at[1:].add(-t0 * Gs_t * g[:-1])
             + V * Gd_t * g)
    return (lhs - PREF * (rho_s - rho_t)).at[N - 1].set(Phi[N - 1])

res = jit(_residual); jacP = jit(jacfwd(_residual, argnums=0))

def newton_fixed_set(Phi0, mu, mask, tol=1e-10, it=60):
    P = jnp.array(Phi0); mj = jnp.float64(mu); mk = jnp.array(mask)
    for _ in range(it):
        F = np.array(res(P, mj, mk)); nr = float(np.max(np.abs(F)))
        if nr < tol: return np.array(P), nr
        J = np.array(jacP(P, mj, mk))
        try: d = np.linalg.solve(J, -F)
        except np.linalg.LinAlgError: return np.array(P), nr
        al = 1.0
        for _ in range(14):
            Pt = P + al * jnp.array(d)
            if float(np.max(np.abs(np.array(res(Pt, mj, mk))))) < nr:
                P = Pt; break
            al *= 0.5
        else:
            return np.array(P), nr
    F = np.array(res(P, mj, mk))
    return np.array(P), float(np.max(np.abs(F)))

def active_set_solve(Phi0, mu, max_outer=25):
    """Outer loop: solve with A frozen, recompute A, repeat to a fixed point."""
    P = np.array(Phi0)
    lap = np.maximum(1.0 + P / csq, N_floor)
    mask = (m0 * (1.0 - lap) - mu > 0).astype(float)
    seen = []
    for k in range(max_outer):
        P, rn = newton_fixed_set(P, mu, mask)
        lap = np.maximum(1.0 + P / csq, N_floor)
        new = (m0 * (1.0 - lap) - mu > 0).astype(float)
        if np.array_equal(new, mask):
            return P, rn, int(mask.sum()), k + 1, True
        key = tuple(new.astype(int))
        if key in seen:                      # 2-cycle: accept the better of the two
            return P, rn, int(new.sum()), k + 1, False
        seen.append(key); mask = new
    return P, rn, int(mask.sum()), max_outer, False


# ---------------------------------------------------------------- fold tests
d = np.load("tov_tk_chemical_el.npz", allow_pickle=True)
mus = sorted(float(m) for m in d["mu_values"] if f"mu{float(m):.6f}_Phi" in d.files)
P = np.array(d[f"mu{mus[-1]:.6f}_Phi"]); mu = mus[-1]
print(f"seed mu/m0 = {mu/m0:.5f}", flush=True)

# 1. march to the fold, then bisect
step = 0.0025 * m0
while step > 1e-9 * m0:
    Pt, rn, ne, k, fx = active_set_solve(P, mu + step)
    lap = np.maximum(1.0 + Pt / csq, N_floor)
    if rn < 1e-8 and ne >= 2 and lap.min() > 0.1:
        mu += step; P = Pt; step = min(step * 1.3, 0.0025 * m0)
    else:
        step *= 0.5
lap = np.maximum(1.0 + P / csq, N_floor)
mask = jnp.array((m0 * (1.0 - lap) - mu > 0).astype(float))
print(f"\nFOLD: mu*/m0 = {mu/m0:.6f}   n_eff = {int(mask.sum())}   "
      f"min N = {lap.min():.6f}   min Phi = {P.min():.5f}", flush=True)

# 2. Crandall-Rabinowitz with the active set FROZEN (residual smooth there)
J = np.array(jacP(jnp.array(P), jnp.float64(mu), mask))[:-1, :-1]
w, V = np.linalg.eig(J); idx = np.argsort(np.abs(w))
lam1, lam2 = abs(w[idx[0]]), abs(w[idx[1]])
print(f"\n(i) four smallest |eig|: " + ", ".join(f"{abs(w[i]):.5g}" for i in idx[:4]))
print(f"(i) vanishing:  |lam_1| = {lam1:.4e}   -> "
      f"{'OK, degenerate' if lam1 < 0.05 else 'does NOT vanish'}")
print(f"(i) simplicity: |lam_2/lam_1| = {lam2/lam1:.1f}")
print(f"    min N = {lap.min():.5f} vs floor {N_floor}  -> "
      f"{'ON THE CLAMP' if lap.min() < 3*N_floor else 'clear of the clamp'}")
v_r = np.real(V[:, idx[0]]); v_r /= np.linalg.norm(v_r)
v = np.concatenate([v_r, [0.0]])
wl, Vl = np.linalg.eig(J.T); il = np.argsort(np.abs(wl))
psi = np.real(Vl[:, il[0]]); psi /= np.linalg.norm(psi)
if psi @ v_r < 0: psi = -psi
jacM = jax.jacfwd(_residual, argnums=1)
Fmu = np.array(jacM(jnp.array(P), jnp.float64(mu), mask))
tr = float(psi @ Fmu[:-1])
print(f"(ii) transversality <psi,F_mu> = {tr:+.6g}   (|<psi,v>| = {abs(psi@v_r):.4f})")
F0 = np.array(res(jnp.array(P), jnp.float64(mu), mask))
qs = []
for h in [1e-3, 3e-4, 1e-4, 3e-5]:
    Fp = np.array(res(jnp.array(P + h*v), jnp.float64(mu), mask))
    Fm = np.array(res(jnp.array(P - h*v), jnp.float64(mu), mask))
    q = float(psi @ ((Fp - 2*F0 + Fm)[:-1] / h**2)); qs.append(q)
    print(f"(iii) h={h:8.0e}  <psi,F_PhiPhi[v,v]> = {q:+.6g}")
mid = qs[:3]
stable = abs(max(mid)-min(mid)) <= 0.05*max(abs(x) for x in mid)
print(f"(iii) h-independent: {stable}  (spread {abs(max(mid)-min(mid)):.3g} on "
      f"{np.mean(np.abs(mid)):.3g})")
ok = lam1 < 0.05 and lap.min() > 3*N_floor and stable and abs(tr) > 1e-12
print(f"\n=> {'QUADRATIC FOLD ESTABLISHED' if ok else 'NOT established'}: "
      f"mu - mu* = -(1/2)(<psi,F_vv>/<psi,F_mu>) s^2, coefficient "
      f"{-0.5*qs[1]/tr:+.4g}" if ok else "")
np.savez("fold_active_set.npz", mu_star=mu, mu_star_over_m0=mu/m0, Phi_star=P,
         eigs=np.abs(w[idx[:8]]), transversality=tr, quadratic=qs[1],
         n_eff=int(mask.sum()), minN=lap.min())
print("wrote fold_active_set.npz", flush=True)
