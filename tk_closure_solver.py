#!/usr/bin/env python3
"""The TK closure solver, shared so that every study uses the SAME one.

This was extracted verbatim from tov_tk_chemical_el.py.  It exists because
tov_convergence_el.py had its own independent solver -- scipy's tridiagonal
eigensolver with a finite-difference Jacobian -- and a convergence study that
uses a different solver from the one whose convergence is being studied is
testing the wrong thing.  The two did agree to six digits where they overlap
(min Phi = -1.254655 at N = 200, mu/m0 = 0.10), so the discretisation result
stood, but the arrangement was indefensible and is now gone.

The reason the split existed was cost: the main solver differentiates a dense
N x N eigendecomposition, and forward mode with N tangents is infeasible at the
N_s = 3000 the convergence study reaches.  The colored Jacobian removes that
obstacle -- its bandwidth is set by the correlation length, which is physical
and does NOT grow with N_s, so the cost is ~2b+1 forward passes at every
lattice size rather than N of them.

Everything here is unchanged from the original; see that file's comments for
the closure prefactor, the active-set method and the colored Jacobian.
"""
import os, time
import numpy as np
os.environ.setdefault("JAX_PLATFORM_NAME", "gpu")
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit, jacfwd

t0 = 1.0
N_floor = 0.005


def make_solver(N, beta0, m0, csq, band=None):
    n_arr = jnp.arange(1, N+1, dtype=jnp.float64)
    g = 4.0 * jnp.pi * n_arr**2
    diag0 = jnp.zeros(N, dtype=jnp.float64)
    nunk = N - 1

    def _bond_correlations(diag_vals, off_vals):
        """Diagonal and off-diagonal correlations from eigh."""
        H = jnp.diag(diag_vals) + jnp.diag(off_vals, 1) + jnp.diag(off_vals, -1)
        evals, evecs = jnp.linalg.eigh(H)
        be = jnp.clip(beta0 * evals, -500.0, 500.0)
        f = 1.0 / (jnp.exp(be) + 1.0)
        vf = evecs * f[None, :]
        Gd = jnp.sum(evecs * vf, axis=1)
        Gs = jnp.sum(evecs[:-1] * vf[1:], axis=1)
        return Gd, Gs

    def _bkm_cov(diag_vals, off_vals):
        """BKM bond-current covariance for each bond."""
        H = jnp.diag(diag_vals) + jnp.diag(off_vals, 1) + jnp.diag(off_vals, -1)
        evals, evecs = jnp.linalg.eigh(H)
        be = jnp.clip(beta0 * evals, -500.0, 500.0)
        f = 1.0 / (jnp.exp(be) + 1.0)
        diff = beta0 * (evals[:, None] - evals[None, :])
        phi = jnp.where(jnp.abs(diff) > 1e-12,
                         jnp.expm1(diff) / jnp.where(jnp.abs(diff) > 1e-12, diff, 1.0),
                         1.0)
        K = f[:, None] * (1.0 - f[None, :]) * phi
        U_n = evecs[:-1, :]; U_n1 = evecs[1:, :]
        t1 = jnp.sum((U_n1**2) @ K * (U_n**2), axis=1)
        t2 = jnp.sum((U_n**2) @ K * (U_n1**2), axis=1)
        t3 = jnp.sum((U_n * U_n1) @ K * (U_n * U_n1), axis=1)
        return t1 + t2 - 2.0 * t3

    # Background BKM covariance (computed once)
    off_bg = -t0 * jnp.ones(N-1, dtype=jnp.float64)
    cov_bg = _bkm_cov(diag0, off_bg)
    cov_bg = jnp.maximum(cov_bg, 1e-30)

    # Closure prefactor of eq:sc-closure: P = c_*^2 t0^2/(lambda eps_b) = 36.825
    # at the TK parameters, with eps_b = 2 t0 Re[...] per channel on the flat
    # reference chain (eq:sc-sumrule), read at the BULK bond (the open-chain ends
    # give 0.808 and 0.539 against the bulk 0.6257, and the bulk is flat to 3e-6
    # over bonds 20..180).
    #
    # lambda here is the PER-CHANNEL alpha_S^bond = 0.217, NOT alpha_S = 0.217 D.
    # The paper uses one symbol for two conventions, and this is the defect that
    # makes eq:sc-closure read 9.206 instead of 36.825:
    #   - In the CLOSURE, lambda multiplies a per-channel Dirichlet form against a
    #     per-channel relative entropy.  kappa^KMS ~ D g_n and S_rel ~ D (the
    #     alpha^r eigenbasis splits into D decoupled copies), so D factors out of
    #     the stationarity condition entirely and lambda cannot itself carry D.
    #     Hence lambda = alpha_S^bond = 0.217 here.
    #   - In the GAUSS-LAW MATCH of eq:sc-GM, the same symbol absorbs one factor
    #     D, because sec:sc-newton writes W_flat/g_n with D dropped while eq:sc-M
    #     carries D explicitly.  There lambda = alpha_S = 0.217 D is right, and it
    #     is what gives the physically correct G ~ 1/D (Phi is D-independent while
    #     the gravitating mass is proportional to D).
    # Both usages are internally consistent; only the shared symbol is wrong.
    #
    # The scripts previously used the solver shorthand beta0/c_*^2, which at
    # (beta0=5, c*^2=5) is exactly 1.000.  For the defect model that is a
    # relabelling of V0; for TK it is NOT, because mu enters V = m0(1-N) - mu
    # without the prefactor.  Note m0/t0 is a FREE dimensionless input, so the
    # published m0 = 3 does not survive the corrected P -- stars reappear at
    # m0 ~ 0.03-0.06.
    LAM_ALPHA_S = 0.217
    _, _Gs_flat = _bond_correlations(diag0, off_bg)
    _eps_b = 2.0 * t0 * _Gs_flat[(N - 1) // 2]
    _pref = float(csq * t0 ** 2 / (LAM_ALPHA_S * _eps_b))
    print(f"  closure prefactor P = {_pref:.4f}  (solver shorthand beta0/c*^2 = "
          f"{beta0/csq:.4f}; eps_b = {float(_eps_b):.5f})", flush=True)

    def _conductances(Phi, mu):
        """BKM conductances kappa(Phi) = g t0^2 Nbar^2 cov/cov_bg."""
        lapse = jnp.maximum(1.0 + Phi / csq, N_floor)
        Nbar = 0.5 * (lapse[:-1] + lapse[1:])
        off = -t0 * jnp.abs(Nbar)
        cov = _bkm_cov(diag0, off)
        cov_ratio = cov / cov_bg
        return g[:-1] * t0**2 * Nbar**2 * cov_ratio

    def _energy(Phi, mu):
        """Conductance energy: (1/2) sum_b kappa_b(Phi) * (dPhi_b)^2."""
        kappa = _conductances(Phi, mu)
        dPhi = Phi[:-1] - Phi[1:]
        return 0.5 * jnp.sum(kappa * dPhi**2)

    _grad_energy = jax.grad(_energy, argnums=0)

    def _residual(Phi, mu, mask=None):
        """JAX-traceable EL residual.  mu is a traced scalar argument.

        mask, when given, FREEZES the active set (the bound shells): V is taken
        as (m0(1-N) - mu) on the mask and 0 off it, with no max().  That removes
        the kink at the free boundary and makes the residual smooth in Phi, so
        Newton converges quadratically instead of chattering.  With mask=None it
        is the original truncated form.

        Why this matters: with the truncated form Newton STALLS wherever a shell
        is crossing threshold -- |R| = 5.6e-2 at mu/m0 = 0.1925 against a
        Jacobian whose smallest eigenvalue is 0.52, i.e. well-conditioned, so a
        solution exists and cannot be reached.  Those stalls were being recorded
        as branch ends and then read as a fold.  The real fold is at
        mu*/m0 = 0.191087 and is only reachable this way.
        """
        # LHS = grad of energy functional (Laplacian + EL correction, exact)
        lhs = _grad_energy(Phi, mu)

        # Self-consistent source
        lapse = jnp.maximum(1.0 + Phi / csq, N_floor)
        Nbar = 0.5 * (lapse[:-1] + lapse[1:])
        Nbar_abs = jnp.abs(Nbar)
        off = -t0 * Nbar_abs

        # Chemical potential threshold: V = max(m0*(1-N) - mu, 0)
        V_raw = m0 * (1.0 - lapse) - mu
        # Source convention (eq. sc-rho, sec:ex-observable, compute_exact_data_gpu.py):
        # the bond term is NEGATIVE against a positive V_defect term, carries ONE
        # power of the lapse (the bare shell observable -- the smeared correlator
        # Gs already supplies it, and a second factor would count the lapse twice),
        # and is split half onto each endpoint of the bond.
        V_defect = jnp.maximum(V_raw, 0.0) if mask is None else V_raw * mask

        # Sigma state (background Hamiltonian, same off-diag)
        Gd_s, Gs_s = _bond_correlations(diag0, off)
        rho_s = jnp.zeros(N, dtype=jnp.float64)
        rho_s = rho_s.at[:-1].add(-1.0 * t0 * Gs_s * g[:-1]).at[1:].add(-1.0 * t0 * Gs_s * g[:-1])

        # Target state (with defect potential)
        Gd_t, Gs_t = _bond_correlations(V_defect, off)
        rho_t = jnp.zeros(N, dtype=jnp.float64)
        rho_t = rho_t.at[:-1].add(-1.0 * t0 * Gs_t * g[:-1]).at[1:].add(-1.0 * t0 * Gs_t * g[:-1])
        rho_t = rho_t + V_defect * Gd_t * g

        # eq:sc-closure prefactor P = c_*^2 t0^2/(lambda eps_b), not the solver
        # shorthand beta0/c_*^2.  See _closure_prefactor below.
        F = lhs - _pref * (rho_s - rho_t)
        # BC: Phi_N = 0 (last entry of full Phi vector)
        F = F.at[N - 1].set(Phi[N - 1])
        return F

    # JIT compile residual and Jacobian (mu is traced → compiles once)
    resid_jit = jit(_residual)
    jac_jit = jit(jacfwd(_residual, argnums=0))
    resid_m = jit(lambda P, mu, mk: _residual(P, mu, mk))
    jac_m = jit(jacfwd(lambda P, mu, mk: _residual(P, mu, mk), argnums=0))

    # Warm up JIT (one-time compilation cost)
    print("  JIT compiling residual...", flush=True)
    t_jit = time.time()
    _dummy = jax.block_until_ready(resid_jit(jnp.zeros(N), jnp.float64(0.1)))
    print(f"  Residual compiled in {time.time() - t_jit:.1f}s", flush=True)

    # Warm up the EXACT Jacobian only if it will actually be used.  It needs N
    # tangents through a dense N x N eigendecomposition -- 26 GiB at N = 600 --
    # so at large N this warm-up alone kills the run even though the colored
    # Jacobian that replaces it a few lines below needs only 2b+1 passes.
    if band is None:
        t_jit = time.time()
        _dummy = jax.block_until_ready(jac_jit(jnp.zeros(N), jnp.float64(0.1)))
        print(f"  Jacobian (jacfwd) compiled in {time.time() - t_jit:.1f}s",
              flush=True)

    # ---------------- Colored (banded) Jacobian -----------------------------
    #
    # The Jacobian is 51% of a Newton iteration (472 ms of 925 ms at N=200), and
    # it is numerically BANDED: the correlations that build it decay
    # exponentially in shell separation, so at beta0 t0 = 5 everything outside
    # bandwidth 30 is 9.3e-14 of the peak entry.
    #
    # A banded Jacobian does not need N forward passes.  Columns more than 2b
    # apart have disjoint row supports, so the columns split into p = 2b+1
    # groups by j mod p and p passes recover every entry.  The unscrambling is
    # EXACT, not a fit: within one group consecutive columns differ by p > 2b,
    # so at most one of them reaches any given row.
    #
    # Measured at N=200 on an A6000 against the 200-tangent jacfwd:
    #     b= 8 ( 17 passes)  6.71x   max rel err 5.9e-08
    #     b=12 ( 25 passes)  5.48x               5.9e-09
    #     b=20 ( 41 passes)  4.19x               4.5e-11
    #     b=30 ( 61 passes)  4.61x               9.3e-14
    #     b=40 ( 81 passes)  3.53x               7.1e-15
    # b=30 beating b=20 with MORE passes is a GPU tiling effect, not noise, and
    # is why the bandwidth is picked by measurement below rather than by taste.
    #
    # Why this cannot change an answer: truncating at ~1e-13 perturbs the Newton
    # DIRECTION only.  The fixed point is defined by the fp64 residual test,
    # which is untouched, so a slightly wrong direction can cost an iteration
    # but cannot move where the iteration lands.
    #
    # The bandwidth is CALIBRATED at run time, never hard-coded, because it
    # tracks the correlation length -- a colder beta0 t0 spreads the Jacobian,
    # and a hard-coded 30 would silently truncate the temperature chain.
    def _calibrate_bandwidth(tol=1e-13, margin=1.6):
        """Smallest b with all |J| outside the band below tol, times a margin."""
        probe = -3.0 * jnp.exp(-0.5 * (n_arr / 8.0) ** 2)
        probe = probe.at[-1].set(0.0)
        lp = jnp.maximum(1.0 + probe / csq, N_floor)
        mk = (m0 * (1.0 - lp) - 0.19 * m0 > 0).astype(jnp.float64)
        J = np.array(jac_m(probe, jnp.float64(0.19 * m0), mk))
        scale = float(np.abs(J).max())
        if not np.isfinite(scale) or scale <= 0.0:
            return None
        off = np.abs(np.subtract.outer(np.arange(N), np.arange(N)))
        for cand in range(2, N):
            sel = J[off > cand]
            if sel.size == 0:            # band covers the whole matrix
                return None
            if float(np.abs(sel).max()) / scale < tol:
                b = int(np.ceil(cand * margin))
                # p = 2b+1 must beat N, or the colouring costs more than the
                # exact Jacobian it replaces.
                return None if 2 * b + 1 >= N else b
        return None                      # not banded: keep the full Jacobian

    def _make_colored_jac(b, masked):
        """p = 2b+1 forward passes.  masked=False is the truncated residual."""
        p = 2 * b + 1
        cols = jnp.arange(N)
        V = (cols[None, :] % p == jnp.arange(p)[:, None]).astype(jnp.float64)
        rows = jnp.arange(N)
        band = jnp.abs(rows[:, None] - cols[None, :]) <= b
        pick = jnp.broadcast_to((cols % p)[None, :], (N, N))
        rowi = jnp.broadcast_to(rows[:, None], (N, N))

        # The eigh JVP holds ~12 N x N intermediates per tangent, so vmapping
        # all p = 97 at once is 22 GiB at N = 1500.  A Python-level chunk loop
        # does NOT fix it: the chunks are independent, so XLA schedules them
        # concurrently and fuses them straight back into one allocation (it
        # still asked for 18 GiB).  lax.scan makes the dependence explicit, so
        # the chunks genuinely run one after another and the peak is one
        # chunk's worth.  Same arithmetic, same Jacobian to the bit.
        _chunk = max(1, min(p, int(3.0e9 / max(N * N * 96.0, 1.0))))
        _nchunk = -(-p // _chunk)
        _Vpad = jnp.concatenate(
            [V, jnp.zeros((_nchunk * _chunk - p, N), dtype=jnp.float64)], axis=0
        ).reshape(_nchunk, _chunk, N)

        def _assemble(f, P):
            def step(carry, Vc):
                return carry, jax.vmap(lambda v: jax.jvp(f, (P,), (v,))[1])(Vc)
            _, JVc = jax.lax.scan(step, 0, _Vpad)      # (nchunk, chunk, N)
            JV = JVc.reshape(_nchunk * _chunk, N)[:p]
            return jnp.where(band, JV[pick, rowi], 0.0)

        if masked:
            @jit
            def jac(P, mu, mk):
                return _assemble(lambda x: _residual(x, mu, mk), P)
        else:
            @jit
            def jac(P, mu):
                return _assemble(lambda x: _residual(x, mu), P)
        return jac

    # band may be supplied to SKIP calibration.  Calibration measures the band
    # by forming the exact N-tangent Jacobian once, which is precisely what is
    # unaffordable at large N -- it asked for 26 GiB at N = 600 and died.  The
    # premise of the whole construction is that the bandwidth tracks the
    # CORRELATION LENGTH, which is physical and N-independent, and the measured
    # values bear that out: 48 at N = 200, 45 at N = 400.  So calibrate once on
    # a small lattice and pass the answer up.
    _band = band if band is not None else _calibrate_bandwidth()
    if _band is None:
        # Not banded at this parameter point -- keep the exact Jacobian.  This
        # is the branch a much colder beta0 t0 would take, and it costs speed
        # rather than accuracy, which is the right way round.
        print("  Jacobian is NOT banded at 1e-13; keeping the full jacfwd",
              flush=True)
    else:
        t_jit = time.time()
        # Both paths matter: active_set_newton uses the masked Jacobian, and
        # newton() -- the fallback the seed loop runs on EVERY failed seed, so
        # the hot path at a rejected mu -- uses the maskless one.
        jac_m = _make_colored_jac(_band, masked=True)
        jac_jit = _make_colored_jac(_band, masked=False)
        jax.block_until_ready(jac_m(jnp.zeros(N), jnp.float64(0.1), jnp.zeros(N)))
        jax.block_until_ready(jac_jit(jnp.zeros(N), jnp.float64(0.1)))
        # (both are the COLORED versions by this point -- rebound just above)
        print(f"  Colored Jacobian: bandwidth {_band}, {2*_band+1} forward "
              f"passes instead of {N} (compiled in {time.time()-t_jit:.1f}s)",
              flush=True)

    def diagnostics(Phi_np, mu_val):
        """Compute all diagnostic quantities from a solution (numpy)."""
        Phi = np.array(Phi_np)
        lapse = np.maximum(1.0 + Phi / csq, N_floor)
        Nbar = 0.5 * (lapse[:-1] + lapse[1:])
        Nbar_abs = np.abs(Nbar)
        off = jnp.array(-t0 * Nbar_abs)

        V_raw = m0 * (1.0 - lapse) - mu_val
        V_defect = np.maximum(V_raw, 0.0)
        n_eff = int(np.sum(V_defect > 1e-10))

        Gd_s, Gs_s = _bond_correlations(diag0, off)
        Gd_s = np.array(Gd_s); Gs_s = np.array(Gs_s)
        Gd_t, Gs_t = _bond_correlations(jnp.array(V_defect), off)
        Gd_t = np.array(Gd_t); Gs_t = np.array(Gs_t)

        n_np = np.array(n_arr)
        g_np = np.array(g)
        # Same convention as _residual above -- this array is SAVED as
        # mu*_source, so if it disagreed with the residual the stored source
        # would not be the one that produced the stored solution.
        bond_s = -2.0 * t0 * Gs_s * g_np[:-1]
        rho_s = np.zeros(N)
        rho_s[:-1] += 0.5 * bond_s
        rho_s[1:] += 0.5 * bond_s
        bond_t = -2.0 * t0 * Gs_t * g_np[:-1]
        rho_t = np.zeros(N)
        rho_t[:-1] += 0.5 * bond_t
        rho_t[1:] += 0.5 * bond_t
        rho_t += V_defect * Gd_t * g_np
        source = _pref * (rho_s - rho_t)

        cov = np.array(_bkm_cov(diag0, off))
        cov_bg_np = np.array(cov_bg)
        kappa_kms = g_np[:-1] * t0**2 * Nbar_abs**2 * (cov / cov_bg_np)

        return {"Phi": Phi, "lapse": lapse, "source": source,
                "V_defect": V_defect, "kappa_kms": kappa_kms,
                "n_eff": n_eff, "Nbar": Nbar_abs,
                "Gd_s": Gd_s, "Gd_t": Gd_t, "Gs_s": Gs_s, "Gs_t": Gs_t,
                "rho_s": rho_s, "rho_t": rho_t}

    def _newton_frozen(Phi, mu_val, mask, tol=1e-10, it=60):
        """Newton with the active set FROZEN -- the residual is smooth here."""
        P = jnp.array(Phi, dtype=jnp.float64)
        mj = jnp.float64(mu_val); mk = jnp.array(mask)
        for _ in range(it):
            F = np.array(resid_m(P, mj, mk)); nr = float(np.max(np.abs(F)))
            if nr < tol:
                return np.array(P), nr
            J = np.array(jac_m(P, mj, mk))
            try:
                dP = np.linalg.solve(J, -F)
            except np.linalg.LinAlgError:
                return np.array(P), nr
            al = 1.0
            for _ in range(14):
                Pt = P + al * jnp.array(dP)
                if float(np.max(np.abs(np.array(resid_m(Pt, mj, mk))))) < nr:
                    P = Pt; break
                al *= 0.5
            else:
                return np.array(P), nr
        return np.array(P), float(np.max(np.abs(np.array(resid_m(P, mj, mk)))))

    def active_set_newton(mu_val, seed, max_outer=25):
        """Freeze the bound set, solve, recompute it, repeat to a fixed point.

        At the fixed point V > 0 on the set and V <= 0 off it, so the frozen V
        equals max(V_raw, 0) exactly and this solves the ORIGINAL problem -- the
        freezing is a solution technique, not a change of model.
        """
        P = np.asarray(seed, dtype=float).copy()
        if P.shape[0] == nunk:
            P = np.append(P, 0.0)
        lap = np.maximum(1.0 + P / csq, N_floor)
        mask = (m0 * (1.0 - lap) - mu_val > 0).astype(float)
        seen = []
        for k in range(max_outer):
            P, rn = _newton_frozen(P, mu_val, mask)
            lap = np.maximum(1.0 + P / csq, N_floor)
            Vr = m0 * (1.0 - lap) - mu_val
            new = (Vr > 0).astype(float)
            if np.array_equal(new, mask):
                # Complementarity: the frozen problem equals the ORIGINAL one only
                # if V > 0 everywhere on the set and V <= 0 everywhere off it.
                # Verify it rather than assume it -- the frozen residual can be
                # driven to 1e-9 by a set that is not self-consistent, which looks
                # converged and is not a solution.
                on_ok = bool(np.all(Vr[mask > 0.5] > -1e-12)) if mask.any() else False
                off_ok = bool(np.all(Vr[mask < 0.5] < 1e-12))
                if on_ok and off_ok:
                    return P, rn
                return P, 1e30       # frozen-residual solution, not a real one
            key = tuple(new.astype(int))
            if key in seen:
                # 2-cycle between adjacent active sets: no fixed point here, so
                # there is no solution to return.  Returning the current iterate
                # (as this did) produced "converged" points at mu/m0 = 0.26-0.36
                # with cores of 8, 34, 9 and 70 shells whose V was NEGATIVE on
                # shells they treated as bound -- non-solutions with 5e-9
                # residuals.  Reject instead.
                return P, 1e30
            seen.append(key); mask = new
        return P, 1e30               # no fixed point within max_outer

    def newton(mu_val, seed, max_iter=80, verbose=False):
        """Newton solve with jacfwd Jacobian. Full N-vector (Phi_N=0 via BC)."""
        Phi = jnp.array(seed, dtype=jnp.float64)
        if Phi.shape[0] == nunk:
            Phi = jnp.concatenate([Phi, jnp.array([0.0])])
        mu_j = jnp.float64(mu_val)

        best_Phi, best_rn = Phi, 1e30
        stagnant = 0

        for it in range(max_iter):
            F = jax.block_until_ready(resid_jit(Phi, mu_j))
            rn = float(jnp.max(jnp.abs(F)))
            if rn < best_rn:
                best_Phi = Phi; best_rn = rn
            if verbose and (it % 5 == 0 or rn < 1e-8):
                lapse_min = float(jnp.min(jnp.maximum(1.0 + Phi / csq, N_floor)))
                print(f"    it={it:3d} |R|={rn:.3e} minN={lapse_min:.4f}", flush=True)
            if rn < 1e-8:
                if verbose: print(f"    CONVERGED at it={it}")
                break

            J = jax.block_until_ready(jac_jit(Phi, mu_j))
            reg = 1e-10 * jnp.maximum(jnp.abs(jnp.diag(J)), 1e-10)
            J_reg = J + jnp.diag(reg)
            dPhi = jnp.linalg.solve(J_reg, -F)

            # Line search
            alpha = 1.0
            improved = False
            for _ in range(8):
                Phi_trial = Phi + alpha * dPhi
                lt = 1.0 + Phi_trial / csq
                if float(jnp.min(lt)) < N_floor:
                    alpha *= 0.5; continue
                F_trial = jax.block_until_ready(resid_jit(Phi_trial, mu_j))
                rn_trial = float(jnp.max(jnp.abs(F_trial)))
                if rn_trial < rn:
                    Phi = Phi_trial
                    improved = True
                    break
                alpha *= 0.5

            if not improved:
                Phi = Phi + 0.01 * dPhi
                stagnant += 1
                if stagnant >= 10:
                    break
            else:
                stagnant = 0

            if float(jnp.any(jnp.isnan(Phi))):
                break

        F_final = jax.block_until_ready(resid_jit(best_Phi, mu_j))
        final_rn = float(jnp.max(jnp.abs(F_final)))
        return np.array(best_Phi), final_rn

    return {"newton": newton, "active_set_newton": active_set_newton,
            "diagnostics": diagnostics, "band": _band,
            # exposed for tests/profile_tk_solver.py only
            "_resid_m": resid_m, "_jac_m": jac_m,
            "n_arr": np.array(n_arr), "g": np.array(g), "N": N, "nunk": nunk}



def exterior_fit(Phi, lapse, n_eff, N, n_arr, g, csq, t0_=1.0):
    """Schwarzschild w-fit, 1/r Phi-fit and w-flux constancy on the exterior.

    Extracted from tov_tk_chemical_el.py's driver so that the convergence study
    measures r_s the SAME way the main scan does.  It did not: it used a window
    starting at n_eff+3 against the main scan's n_eff+4, which is why the two
    disagreed by 0.17% (r_s = 1.5082 vs 1.5057) at N = 200 while agreeing on
    min Phi to all six digits.

    Window r in [r_* + 3a, min(R/2, r_* + 40a)] with r_* = (n_eff+1) a, as
    sec:tov-exterior states.
    """
    w = lapse ** 2
    out = {"r_s_w": 0.0, "w_rel_err": 0.0, "r_s_phi": 0.0,
           "phi_rel_err": 0.0, "J_w_var": 0.0}
    n_start = max(n_eff + 4, 5)
    n_end = min(N // 2, n_eff + 41)
    if n_start < n_end:
        idx = np.arange(n_start - 1, n_end)
        r_fit = n_arr[idx]
        basis_1r = 1.0 / r_fit - 1.0 / N
        target_w = -(w[idx] - 1.0)
        r_s_w = float(np.dot(basis_1r, target_w) / np.dot(basis_1r, basis_1r))
        w_pred = 1.0 - r_s_w * basis_1r
        rms_w = float(np.sqrt(np.mean((w[idx] - w_pred) ** 2)))
        rms_n = float(np.sqrt(np.mean((w[idx] - 1.0) ** 2)))
        out["r_s_w"] = r_s_w
        out["w_rel_err"] = rms_w / rms_n if rms_n > 1e-30 else 0.0
        basis_phi = 1.0 / N - 1.0 / r_fit
        A = float(np.dot(basis_phi, Phi[idx]) / np.dot(basis_phi, basis_phi))
        out["r_s_phi"] = 2.0 * abs(A) / csq
        out["phi_rel_err"] = (float(np.sqrt(np.mean((Phi[idx] - A * basis_phi) ** 2))
                                    / np.sqrt(np.mean(Phi[idx] ** 2)))
                              if np.any(Phi[idx] != 0) else 0.0)
    dw = w[1:] - w[:-1]
    J_w = g[:-1] * t0_ ** 2 * dw
    f_start = min(n_eff + 5, N - 2)
    f_end = min(N // 2, n_eff + 40)
    if f_start < f_end:
        J = J_w[f_start:f_end]
        m = float(np.mean(J))
        out["J_w_var"] = (float(np.max(np.abs(J - m)) / abs(m))
                          if abs(m) > 1e-30 and len(J) > 1 else 0.0)
    return out
