"""Unit tests for the Tolman-Klein closure -- the paper's headline calculation.

Every test here is written to FAIL if a specific defect found in the 2026-08-06
audit were reintroduced.  A test that cannot fail is worse than no test, so each
one states what would break it.

The TK scripts are top-level programs (they run a scan on import), so the tests
extract their residual machinery with `ast` and exec only the definitions.  That
is deliberate: it tests the code the scripts actually run, not a copy.

Run:  python -m pytest tests/test_tk_closure.py -v
"""
from __future__ import annotations

import ast
import os

# Must precede any jax import: the extracted definitions build float64 arrays,
# and the scripts' own `jax.config.update("jax_enable_x64", True)` is a bare
# expression statement, which the definition extractor deliberately drops.
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The six TK scripts that solve the closure.  fold_continuation_floor and
# fold_transversality live under full_numerics/.
TK_SCRIPTS = [
    "tov_tk_chemical_el.py",
    "tov_tk_bt5_eigenvalue.py",
    "tov_tk_bt10_mu_scan.py",
    "tov_tk_beta_chain.py",
    "full_numerics/fold_transversality.py",
    "full_numerics/fold_continuation_floor.py",
]

# Reference parameters (tov_tk_chemical_el.py's own).
BETA0, M0, CSQ, T0, N_FLOOR = 5.0, 3.0, 5.0, 1.0, 0.005


# --------------------------------------------------------------------------
# Reference implementation, written directly from the paper's equations.
# eq:sc-rho     rho(n) = -2 t0 g_n Re[i tr(alpha^r G_{n,n+1})] + V g_n tr G_nn
# sec:ex-observable  the observable is BARE: one power of the lapse, carried by
#                    the smeared correlator, never an explicit second factor.
# The bond energy is split half onto each endpoint so that sum_n rho(n) equals
# the total bond energy.
# --------------------------------------------------------------------------
def _corr(diag, off, beta0):
    from scipy.linalg import eigh_tridiagonal
    ev, U = eigh_tridiagonal(np.asarray(diag, float), np.asarray(off, float))
    f = 1.0 / (np.exp(np.clip(beta0 * ev, -500, 500)) + 1.0)
    vf = U * f[None, :]
    return np.sum(U * vf, axis=1), np.sum(U[:-1] * vf[1:], axis=1)


def reference_source(Phi, N, beta0=BETA0, m0=M0, csq=CSQ, t0=T0):
    """rho_sigma - rho_tgt for the TK closure, from eq:sc-rho."""
    g = 4.0 * np.pi * np.arange(1, N + 1, dtype=float) ** 2
    lapse = np.maximum(1.0 + Phi / csq, N_FLOOR)
    Nbar = np.abs(0.5 * (lapse[:-1] + lapse[1:]))
    off = -t0 * Nbar
    V = np.maximum(m0 * (1.0 - lapse) - 0.0, 0.0)  # mu supplied by caller below
    return lapse, Nbar, off, g, V


def reference_rho(diag, off, g, N, beta0, t0, V=None):
    Gd, Gs = _corr(diag, off, beta0)
    bond = -2.0 * t0 * Gs * g[:-1]          # NEGATIVE, bare, per eq:sc-rho
    rho = np.zeros(N)
    rho[:-1] += 0.5 * bond                   # split half onto each endpoint
    rho[1:] += 0.5 * bond
    if V is not None:
        rho = rho + V * Gd * g               # defect term POSITIVE
    return rho


# --------------------------------------------------------------------------
# Extract each script's residual machinery without running its driver.
# --------------------------------------------------------------------------
def _load_defs(relpath, names):
    """Exec only the module-level constants and the named function defs."""
    src = open(os.path.join(ROOT, relpath)).read()
    tree = ast.parse(src)
    keep = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            keep.append(node)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            keep.append(node)
        elif isinstance(node, ast.Assign):
            # Handle both `a = ...` and `a, b, c = ...`; fold_transversality
            # declares its constants as a tuple assignment, and skipping those
            # left beta0 undefined inside the extracted functions.
            tgts = []
            for t in node.targets:
                if isinstance(t, ast.Name):
                    tgts.append(t.id)
                elif isinstance(t, ast.Tuple):
                    tgts += [e.id for e in t.elts if isinstance(e, ast.Name)]
            # module constants only; skip anything that calls into the driver
            if tgts and not isinstance(node.value, ast.Call):
                keep.append(node)
            elif tgts and isinstance(node.value, ast.Call):
                fn = node.value.func
                nm = getattr(fn, "id", None) or getattr(fn, "attr", None)
                # "grad" and "maximum" are needed to make the residual CALLABLE:
                # _grad_energy = jax.grad(_energy, ...) and
                # cov_bg = jnp.maximum(_bkm_cov(...), 1e-30) are both module-level
                # Call assignments the residual closes over.  jit/jacfwd are
                # deliberately NOT here -- they would trigger compilation.
                if nm in ("zeros", "arange", "ones", "float64", "grad", "maximum"):
                    keep.append(node)
    # Exec node by node and skip anything that depends on the driver (loaded
    # npz files, scan results).  A single module-level exec would abort on the
    # first such statement and we would lose the function definitions after it.
    ns = {"__name__": "_probe"}
    for node in keep:
        mod = ast.Module(body=[node], type_ignores=[])
        try:
            exec(compile(ast.fix_missing_locations(mod), relpath, "exec"), ns)
        except Exception:
            continue
    return ns, src


def _residual_source(relpath):
    """The source text of the script's residual function."""
    src = open(os.path.join(ROOT, relpath)).read()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in ("_residual", "residual"):
            return ast.get_source_segment(src, node)
    raise AssertionError(f"no residual function found in {relpath}")


# ==========================================================================
# 1. Source convention -- the defect that changed sec:ff-tov's physics
# ==========================================================================
@pytest.mark.parametrize("relpath", TK_SCRIPTS)
def test_bond_term_is_negative_against_positive_defect(relpath):
    """FAILS IF: the bond source is written +2 t0 ... Gs (the pre-audit form).

    eq:sc-rho gives rho(n) = -2 t0 g_n Re[...] + V g_n tr G_nn: the hopping part
    is negative at half filling, the repulsive defect positive.  An overall sign
    would be harmless; the RELATIVE sign is the physics, and getting it wrong
    made the bond term cancel the defect instead of reinforcing it, shrinking
    the core source by ~1.9x.
    """
    body = _residual_source(relpath)
    assert "2.0 * t0 * Nbar_abs * Gs" not in body, "old positive+smeared bond term"
    assert "2.0 * t0 * Nb * Gs" not in body, "old positive+smeared bond term (Nb alias)"
    assert "2.0 * t0 * Nbar * Gs" not in body, "old positive+smeared bond term"
    # The defect term must remain POSITIVE and additive.  Match on structure,
    # not on the variable name: fold_continuation_floor calls it V, the others
    # V_defect, and a name-specific pattern would pass vacuously on the wrong one.
    import re as _re
    assert _re.search(r"(?<!-)\b\w*[Vv]\w*\s*\*\s*Gd_t\s*\*\s*g\b", body), \
        "defect term missing or not additive"
    assert not _re.search(r"-\s*\w*[Vv]\w*\s*\*\s*Gd_t\s*\*\s*g\b", body), \
        "defect term is subtracted; eq:sc-rho makes it positive"


@pytest.mark.parametrize("relpath", TK_SCRIPTS)
def test_observable_carries_one_lapse_power(relpath):
    """FAILS IF: an explicit Nbar multiplies the correlator in the source.

    sec:ex-observable: the lapse is the ADM multiplier conjugate to the shell
    energy, so its expectation in the smeared state already carries one power of
    Nbar; "weighting it a second time would count the lapse twice".  Nbar may
    still appear in `off = -t0*Nbar` (that is the smearing) and in kappa ~ Nbar^2
    (a covariance of a current linear in the amplitude) -- just not on the
    observable.
    """
    body = _residual_source(relpath)
    for pat in ("Nbar_abs * Gs", "Nb * Gs", "Nbar * Gs",
                "Gs_s * Nbar", "Gs_t * Nbar"):
        assert pat not in body, f"observable weighted by the lapse twice: {pat!r}"


# ==========================================================================
# 2. Invariants of the reference source
# ==========================================================================
def test_two_state_source_vanishes_without_a_defect():
    """FAILS IF: the two states stop being measured with the same observable.

    With V = 0 the two Hamiltonians coincide, so rho_sigma - rho_tgt must be
    EXACTLY zero -- that cancellation is what removes Yukawa screening and
    leaves a source localised to the core.
    """
    N = 80
    Phi = -0.4 * np.exp(-np.arange(N) / 12.0)
    Phi[-1] = 0.0
    lapse, Nbar, off, g, _ = reference_source(Phi, N)
    rho_s = reference_rho(np.zeros(N), off, g, N, BETA0, T0)
    rho_t = reference_rho(np.zeros(N), off, g, N, BETA0, T0, V=np.zeros(N))
    assert np.max(np.abs(rho_s - rho_t)) < 1e-14


def test_source_is_localised_to_the_core():
    """FAILS IF: the exterior source stops being exponentially small.

    Outside the core V = 0, so the local Hamiltonians agree and the difference
    survives only through the global state -- eq:sc-twostate-localized.
    """
    N, n_core = 120, 8
    Phi = -0.5 * np.exp(-np.arange(N) / 10.0)
    Phi[-1] = 0.0
    lapse, Nbar, off, g, _ = reference_source(Phi, N)
    V = np.zeros(N)
    V[:n_core] = 0.3
    rho_s = reference_rho(np.zeros(N), off, g, N, BETA0, T0)
    rho_t = reference_rho(V, off, g, N, BETA0, T0, V=V)
    src = rho_s - rho_t
    core = np.max(np.abs(src[:n_core]))
    far = np.max(np.abs(src[n_core + 20:]))
    assert far < 1e-6 * core, f"exterior source not localised: {far:.3e} vs {core:.3e}"


def test_bond_energy_is_conserved_by_the_half_split():
    """FAILS IF: the half-onto-each-endpoint assembly is replaced by
    left-endpoint-only (which shifts the source by half a shell).

    Either assembly conserves the total, so this pins the SPLIT, not the total:
    the interior shells must each receive two half-bonds.
    """
    N = 40
    g = 4.0 * np.pi * np.arange(1, N + 1, dtype=float) ** 2
    off = -T0 * np.ones(N - 1)
    rho = reference_rho(np.zeros(N), off, g, N, BETA0, T0)
    _, Gs = _corr(np.zeros(N), off, BETA0)
    bond = -2.0 * T0 * Gs * g[:-1]
    assert abs(rho.sum() - bond.sum()) < 1e-10 * abs(bond.sum())
    # interior shell n gets half of bond n-1 and half of bond n
    n = N // 2
    assert abs(rho[n] - 0.5 * (bond[n - 1] + bond[n])) < 1e-12


# ==========================================================================
# 3. r_s extraction -- the bug that biased every XXZ radius low by 45%
# ==========================================================================
def test_boundary_corrected_estimator_inverts_the_walled_profile():
    """FAILS IF: r_s is extracted by fitting Phi*r to A + B/r.

    The closure is solved with a Dirichlet wall, so
    Phi = -(c*^2 rs/2)(1/r - 1/R) and Phi*r is LINEAR in r.  Fitting a line with
    a hyperbola biases rs low by ~45% and leaves a residual set by the misfit.
    """
    N, R, rs_true, csq = 200, 200.0, 2.21, 0.5
    r = np.arange(1, N + 1, dtype=float)
    Phi = -(csq * rs_true / 2.0) * (1.0 / r - 1.0 / R)
    sl = slice(20, 80)
    GM = -Phi[sl] * r[sl] * R / (R - r[sl])
    rs_corrected = 2.0 * np.median(GM) / csq
    assert abs(rs_corrected - rs_true) < 1e-9, "estimator must invert exactly"
    X = np.column_stack([np.ones(60), 1.0 / r[sl]])
    A = np.linalg.lstsq(X, (Phi * r)[sl], rcond=None)[0][0]
    rs_bad = -2.0 * A / csq
    assert abs(rs_bad / rs_true - 1.0) > 0.3, (
        "the wrong fit form should be badly biased -- if this fails the test "
        "itself has stopped discriminating")


# ==========================================================================
# 4. Fold determination -- must require a VANISHING eigenvalue
# ==========================================================================
def test_fold_test_requires_the_eigenvalue_to_vanish():
    """FAILS IF: fold_transversality checks only simplicity, not vanishing.

    Crandall-Rabinowitz condition (i) is a simple ZERO eigenvalue.  Checking
    |lam_2/lam_1| alone let a run stop at the lapse clamp with |lam_1| = 0.22 and
    still report a quadratic fold.
    """
    src = open(os.path.join(ROOT, "full_numerics/fold_transversality.py")).read()
    assert "vanishing" in src, "no vanishing-eigenvalue check present"
    assert "NOT A FOLD" in src, "no explicit rejection when the eigenvalue is finite"
    assert "N_floor" in src and "clamp" in src.lower(), (
        "no check that the endpoint is away from the lapse clamp")


def test_star_branch_seeding_excludes_the_floor_pinned_family():
    """FAILS IF: a TK script selects seeds by min N > 0.01 alone.

    Under the corrected source the floor-pinned n_eff=1 family reaches down to
    mu/m0 ~ 0.35 with min N = 0.005-0.03, so that threshold no longer separates
    it from the star branch, and continuations walk onto it and report its end
    as a fold.
    """
    for rel in ("tov_tk_bt10_mu_scan.py", "tov_tk_beta_chain.py",
                "full_numerics/fold_continuation_floor.py",
                "full_numerics/fold_transversality.py"):
        src = open(os.path.join(ROOT, rel)).read()
        assert "minN > 0.1" in src or "min N > 0.1" in src or "> 0.1" in src, (
            f"{rel}: no star-branch lapse criterion")


# ==========================================================================
# 5. Cross-script consistency -- all six must agree on the same Phi
# ==========================================================================
def test_all_tk_scripts_share_one_source_convention():
    """FAILS IF: any TK script drifts from the others.

    This is the test that would have caught fold_continuation_floor.py, which
    survived a regex sweep because it names the lapse `Nb` instead of
    `Nbar_abs`, and stayed on the old convention through several reruns.
    """
    import re
    sigs = {}
    for rel in TK_SCRIPTS:
        body = _residual_source(rel)
        # normalise: collapse whitespace, strip the lapse-variable alias
        norm = re.sub(r"\s+", " ", body)
        neg = len(re.findall(r"-1\.0 \* t0 \* Gs_[st] \* g|-2\.0 \* t0 \* Gs_[st] \* g", norm))
        pos = len(re.findall(r"[^-]2\.0 \* t0 \* \w+ \* Gs_[st]", norm))
        sigs[rel] = (neg > 0, pos > 0)
    bad = [k for k, (has_neg, has_pos) in sigs.items() if not has_neg or has_pos]
    assert not bad, f"scripts not on the corrected convention: {bad}"


# ==========================================================================
# 6. NUMERICAL cross-check -- run each script's real residual on one Phi
# ==========================================================================
def _script_source_terms(relpath, Phi, mu):
    """Evaluate rho_sigma - rho_tgt using the script's OWN extracted functions."""
    ns, _ = _load_defs(relpath, ["_bkm_cov", "_bond_correlations"])
    bc = ns.get("_bond_correlations")
    if bc is None:
        return None
    import inspect
    nargs = len(inspect.signature(bc).parameters)
    N = len(Phi)
    g = 4.0 * np.pi * np.arange(1, N + 1, dtype=float) ** 2
    csq = ns.get("csq", CSQ); m0 = ns.get("m0", M0); t0v = ns.get("t0", T0)
    beta0 = ns.get("beta0", BETA0); nfl = ns.get("N_floor", N_FLOOR)
    lapse = np.maximum(1.0 + Phi / csq, nfl)
    Nbar = np.abs(0.5 * (lapse[:-1] + lapse[1:]))
    off = -t0v * Nbar
    V = np.maximum(m0 * (1.0 - lapse) - mu, 0.0)
    zero = np.zeros(N)
    args_s = (zero, off, beta0) if nargs == 3 else (zero, off)
    args_t = (V, off, beta0) if nargs == 3 else (V, off)
    Gd_s, Gs_s = [np.asarray(x, dtype=float) for x in bc(*args_s)]
    Gd_t, Gs_t = [np.asarray(x, dtype=float) for x in bc(*args_t)]
    rho_s = np.zeros(N); b = -2.0 * t0v * Gs_s * g[:-1]
    rho_s[:-1] += 0.5 * b; rho_s[1:] += 0.5 * b
    rho_t = np.zeros(N); b = -2.0 * t0v * Gs_t * g[:-1]
    rho_t[:-1] += 0.5 * b; rho_t[1:] += 0.5 * b
    rho_t = rho_t + V * Gd_t * g
    return rho_s - rho_t


def test_scripts_agree_numerically_with_the_reference_source():
    """FAILS IF: any script's correlator machinery drifts from the others.

    Stronger than the text checks above: this runs each script's OWN
    _bond_correlations on one Phi and compares the resulting two-state source
    against the reference built from eq:sc-rho.  Naming, aliasing and
    refactoring are all invisible to it -- only the numbers matter.
    """
    # Phi must be deep enough to BIND matter: V = max(m0(1-N) - mu, 0) is zero
    # unless 1 - N > mu/m0, and with a shallow well the two states coincide, the
    # source is identically zero and the comparison is 0 == 0.  The first version
    # of this test used Phi = -0.6 exp(-n/9) with mu = 1.2, which gives V = 0
    # everywhere -- it passed even with a 1% perturbation injected into a
    # script's correlator, because there was nothing to compare.
    N = 60
    Phi = -3.5 * np.exp(-np.arange(N) / 9.0); Phi[-1] = 0.0
    g = 4.0 * np.pi * np.arange(1, N + 1, dtype=float) ** 2
    lapse = np.maximum(1.0 + Phi / CSQ, N_FLOOR)
    off = -T0 * np.abs(0.5 * (lapse[:-1] + lapse[1:]))
    checked = 0
    for rel in TK_SCRIPTS:
        # Read m0 from the SCRIPT, never from a constant in this file.  The two
        # drifted apart when the corrected closure prefactor forced m0/t0 from
        # 3.0 to 0.03: this test kept building its reference at 3.0, so it
        # compared a bound reference against a script that binds nothing and
        # failed for a reason that had nothing to do with the correlators.  A
        # test that hard-codes a physical parameter tests the parameter, not the
        # code.
        ns, _ = _load_defs(rel, ["_bond_correlations", "_bkm_cov"])
        m0 = float(ns.get("m0", M0))
        # mu must bind matter on at least a few shells at THIS m0.
        mu = 0.4 * m0 * (1.0 - float(np.min(lapse)))
        V = np.maximum(m0 * (1.0 - lapse) - mu, 0.0)
        # Guard the precondition, so it can never again pass vacuously.
        assert np.count_nonzero(V) >= 3, (
            f"{rel}: setup binds no matter at m0={m0} (V nonzero on "
            f"{np.count_nonzero(V)} shells); the comparison would be trivial")
        ref = (reference_rho(np.zeros(N), off, g, N, BETA0, T0)
               - reference_rho(V, off, g, N, BETA0, T0, V=V))
        assert np.max(np.abs(ref)) > 1e-6, f"{rel}: reference source is ~zero"
        got = _script_source_terms(rel, Phi, mu)
        if got is None:
            continue   # only tov_tk_chemical_el, whose residual is nested in a closure
        scale = max(np.max(np.abs(ref)), 1e-30)
        assert np.max(np.abs(got - ref)) / scale < 1e-10, (
            f"{rel}: two-state source differs from eq:sc-rho reference")
        checked += 1
    assert checked >= 4, f"only {checked} scripts exercised; extraction is failing"


def test_w_flux_is_conserved_in_the_source_free_exterior():
    """FAILS IF: the exterior stops satisfying the discrete vacuum equation.

    eq:tov-Jw -- with kappa = kappa_flat Nbar^2 the vacuum exterior conserves
    J_w = kappa_flat * Delta w exactly on the trigamma profile eq:tov-w-trigamma.
    """
    from scipy.special import polygamma
    N, rs = 400, 30.0
    n0 = 1
    while 1.0 - rs * polygamma(1, n0) <= 0.0:
        n0 += 1
    n = np.arange(n0, n0 + N, dtype=float)
    w = 1.0 - rs * polygamma(1, n)
    kflat = 4.0 * np.pi * n[:-1] ** 2 * T0 ** 2
    Jw = kflat * (w[1:] - w[:-1])
    rel = (Jw.max() - Jw.min()) / abs(Jw.mean())
    assert rel < 1e-9, f"w-flux not conserved on the exact vacuum profile: {rel:.2e}"


def test_closure_prefactor_is_P_not_the_solver_shorthand():
    """FAILS IF: a TK script reverts to pref = beta0/c_*^2.

    eq:sc-closure gives P = c_*^2 t0^2/(lambda eps_b).  At the TK parameters
    (beta0 = 5, c*^2 = 5) the solver shorthand beta0/c_*^2 equals exactly 1.000
    while P = 9.206, and unlike the defect model -- where the shorthand merely
    relabels V0 -- mu enters V = m0(1-N) - mu WITHOUT the prefactor, so the two
    are not related by any rescaling and the fold location differs.
    """
    for rel in TK_SCRIPTS + ["tov_convergence_el.py"]:
        src = open(os.path.join(ROOT, rel)).read()
        assert "(beta0 / csq) * (rho_s - rho_t)" not in src, (
            f"{rel}: still applies the solver shorthand as the closure prefactor")
        assert ("_closure_prefactor" in src or "_pref" in src), (
            f"{rel}: no closure prefactor helper found")


def test_P_and_the_shorthand_actually_differ_at_TK_parameters():
    """Guards the test above: if these ever coincided it would pass vacuously."""
    k = np.linspace(-np.pi, np.pi, 200001)
    eps = -2.0 * np.cos(k)
    f = 0.5 * (1.0 - np.tanh(BETA0 * eps / 2.0))
    eps_b = 2.0 * T0 * np.trapezoid(np.cos(k) * f, k) / (2.0 * np.pi)
    P = CSQ * T0 ** 2 / ((0.217 * 4.0) * eps_b)
    shorthand = BETA0 / CSQ
    assert abs(P / shorthand - 1.0) > 1.0, (
        f"P = {P:.4f} and beta0/c*^2 = {shorthand:.4f} are too close for the "
        "distinction to be testable")


# --------------------------------------------------------------------------
# The 2026-08-06 red-team pass: the fold evidence at beta0t0 = 5 was the lapse
# clamp.  Each test below fails if that class of defect returns.
# --------------------------------------------------------------------------

def test_every_continuation_accept_tests_the_branch_not_just_nontriviality():
    """`max|Phi| > 0.01` alone admits the floor-pinned family.

    Would have failed on the committed tov_tk_bt5_eigenvalue.py, whose four
    acceptance sites all used is_nontrivial() alone.  81 of the 85 solutions it
    stored are the pinned n_eff=1 family, and its continuation terminated at
    min(N) = 0.005000 -- exactly N_floor -- which it reported as a fold.
    """
    src = open(os.path.join(ROOT, "tov_tk_bt5_eigenvalue.py")).read()
    tree = ast.parse(src)
    fn = next(n for n in tree.body
              if isinstance(n, ast.FunctionDef) and n.name == "on_star_branch")
    # on_star_branch calls is_nontrivial itself as its first clause; that is the
    # definition, not an acceptance site, so exclude its body from the walk.
    tree.body = [n for n in tree.body if n is not fn]
    bad = []
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            calls = [n.func.id for n in ast.walk(node.test)
                     if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
            if "is_nontrivial" in calls and "on_star_branch" not in calls:
                bad.append(node.lineno)
    assert not bad, (
        f"is_nontrivial() used as an acceptance test without a branch check at "
        f"line(s) {bad}; the floor-pinned family will be accepted again")
    # and the branch test must actually constrain both shell count and lapse
    body = ast.unparse(fn)
    assert "ne >= 2" in body, "on_star_branch must require a multi-shell core"
    assert "N_floor" in body and "minN" in body, \
        "on_star_branch must hold the lapse off the floor"


def test_fold_transversality_removes_the_dirichlet_row_before_diagonalising():
    """F[N-1] = Phi[N-1] contributes an eigenvalue of exactly 1.

    The committed fold_transversality.npz has eigs[1] = 1.00000000 -- the
    simplicity ratio |lam_2/lam_1| was measured against the boundary condition.
    """
    src = open(os.path.join(ROOT, "full_numerics", "fold_transversality.py")).read()
    assert "Jr = J[:-1, :-1]" in src, "the Dirichlet row must be reduced out"
    assert "np.linalg.eig(Jr)" in src and "np.linalg.eig(Jr.T)" in src, \
        "both the right and left null vectors must come from the reduced Jacobian"
    assert "np.linalg.eig(J)\n" not in src, "the full Jacobian is still diagonalised"


def test_fold_transversality_bisection_uses_the_star_filter():
    """The seeds were filtered to the star branch; the bisection was not.

    It accepted on physical() = minN > 1.5*N_floor, walked onto the pinned
    family, and stopped at min lapse = 0.0075000, exactly that cutoff.
    """
    src = open(os.path.join(ROOT, "full_numerics", "fold_transversality.py")).read()
    assert "if ok and _star(mu + step, Phi_try):" in src, \
        "the bisection must accept on the star-branch filter, not physical()"


def test_quadratic_fold_gate_is_not_vacuous():
    """`abs(tr) > 0 and abs(quad) > 0` passes for any two nonzero floats."""
    src = open(os.path.join(ROOT, "full_numerics", "fold_transversality.py")).read()
    assert "if abs(tr) > 0 and abs(quad) > 0:" not in src, \
        "the vacuous fold gate is back"
    assert "_is_fold" in src and "_stable" in src, (
        "the fold verdict must require a vanishing eigenvalue and an "
        "h-independent second derivative")


def test_surface_radius_is_the_first_empty_shell():
    """r_* is the first shell where V vanishes, i.e. n_eff + 1, not n_eff."""
    for rel in ["tov_tk_chemical_el.py", "tov_tk_el_flux_fig.py"]:
        p = os.path.join(ROOT, rel)
        if not os.path.isfile(p):
            continue
        # ast.unparse drops comments, so this matches CODE only -- the fix's own
        # explanatory comment quotes the old expression.
        code = ast.unparse(ast.parse(open(p).read()))
        assert "float(n_eff)" not in code, (
            f"{rel} still uses r_* = n_eff; the surface is the first shell where "
            f"V vanishes (n_eff + 1), and the printed compactness is ~25% low")
        assert "float(n_eff + 1)" in code, f"{rel} must set r_* = n_eff + 1"


def test_p_offset_uses_its_definition_not_the_near_horizon_proxy():
    """p is defined by n_* = r_s/a + p.  r_s*N_*^2 equals p-1/2 only as n_* -> r_s/a.

    On this branch n_* = 5..11 against r_s/a = 0.16..10, where the proxy is wrong
    by up to 29x and hides that the true offset goes negative at the endpoint.
    """
    src = open(os.path.join(ROOT, "full_numerics", "fold_transversality.py")).read()
    assert "p_half = (ns + 1) - rs - 0.5" in src, \
        "the p-table must compute n_* - r_s/a - 1/2, the definition"
    assert "p_half_proxy" in src, "keep the proxy too, but labelled as a proxy"


def test_the_proxy_and_the_definition_actually_disagree_on_this_branch():
    """Guard for the test above: if they agreed it would pass vacuously.

    Uses the committed p_table, which stores r_s and n_* per row.
    """
    p = os.path.join(ROOT, "fermion", "numerical", "data", "fold_transversality.npz")
    if not os.path.isfile(p):
        pytest.skip("fold_transversality.npz not present")
    d = np.load(p)
    if "p_table" not in d or d["p_table"].size == 0:
        pytest.skip("no p_table")
    R = d["p_table"]
    rs, ns, Nstar = R[:, 1], R[:, 2], R[:, 3]
    direct = ns - rs - 0.5
    proxy = rs * Nstar ** 2
    assert np.max(np.abs(direct - proxy)) > 1.0, (
        "the proxy and the definition agree here, so the test above proves "
        "nothing -- re-derive it on a branch where n_* != r_s/a")


@pytest.mark.parametrize("relpath", TK_SCRIPTS)
def test_residual_is_actually_callable(relpath):
    """Smoke test: build each script's residual and CALL it once.

    Added because a global cleanup of the closure-prefactor helper stripped the
    argument from its call site in the two beta0-sweeping scripts.  Every other
    test in this file inspects source text or rebuilds the source term from
    extracted pieces, so all 30 stayed green while the residual those two
    scripts actually run raised TypeError on the first Newton step.  Source
    inspection cannot catch a broken call; only calling it can.
    """
    import inspect
    ns, _ = _load_defs(relpath, ["_bond_correlations", "_bkm_cov", "_energy",
                                 "_residual", "_closure_prefactor", "residual",
                                 "make_solver"])
    fn = ns.get("_residual") or ns.get("residual")
    if fn is None:
        # tov_tk_chemical_el defines its residual inside make_solver
        mk = ns.get("make_solver")
        if mk is None:
            pytest.skip(f"{relpath}: no module-level residual to call")
        S = mk(24, BETA0, M0, CSQ)
        out = np.asarray(S["newton"](0.05, -0.2 * np.ones(24))[0])
        assert out.shape == (24,) and np.all(np.isfinite(out))
        return
    N = ns.get("N")
    if N is None:
        pytest.skip(f"{relpath}: N not extractable")
    Phi = -0.2 * np.exp(-np.arange(1, N + 1) / 8.0)
    Phi[-1] = 0.0
    nargs = len(inspect.signature(fn).parameters)
    args = [Phi, 0.05] + ([BETA0] if nargs >= 3 else [])
    out = np.asarray(fn(*args))
    assert out.shape == (N,), f"{relpath}: residual returned shape {out.shape}"
    assert np.all(np.isfinite(out)), f"{relpath}: residual returned non-finite values"


def test_closure_prefactor_lambda_is_per_channel_everywhere():
    """FAILS IF: any closure prefactor uses lambda = alpha_S = 0.217 D.

    Added because the fix that moved lambda to the per-channel 0.217 matched on
    the module-level constant `LAM_ALPHA_S = 0.217 * 4.0`, and tov_convergence_el
    writes it INLINE as `(0.217 * 4.0) * eps_b`.  It kept the D-carrying value
    through a commit that claimed to have corrected every script -- a regex sweep
    is not a fix, and only a test that looks at the arithmetic catches it.

    Scoped to the CLOSURE prefactor.  0.217*D is correct elsewhere: it is
    alpha_S in a = 2 sqrt(alpha_S) l_P (eq:sc-planck-relation) and in the
    D-independence of C/alpha_S, both of which full_numerics/verify_analytic_claims.py
    uses legitimately.
    """
    import re as _re
    bad = []
    for rel in TK_SCRIPTS + ["tov_convergence_el.py", "tov_tk_klein_faithful.py"]:
        p = os.path.join(ROOT, rel)
        if not os.path.isfile(p):
            continue
        code = ast.unparse(ast.parse(open(p).read()))   # drops comments
        # any lambda-like factor of 0.217*4 (or 0.868) sitting in a denominator
        # next to eps_b is a closure prefactor using the D-carrying value
        for m in _re.finditer(r"(0\.217\s*\*\s*4(\.0)?|4(\.0)?\s*\*\s*0\.217|0\.868)", code):
            window = code[m.start(): m.start() + 140]
            if "eps_b" in window or "_eps_b" in window:
                bad.append(f"{rel}: {window.splitlines()[0][:90]}")
    assert not bad, (
        "closure prefactor still uses lambda = alpha_S = 0.217 D; it must be the "
        "per-channel alpha_S^bond = 0.217:\n  " + "\n  ".join(bad))


def test_the_two_lambdas_actually_differ():
    """Guard: if 0.217 and 0.217*D coincided, the test above would prove nothing."""
    assert abs(0.217 * 4 - 0.217) > 0.5, "the two lambda conventions must differ"


def test_no_script_reads_npz_mu_keys_at_two_decimals():
    """FAILS IF: any script formats an npz mu key as %.2f.

    tov_tk_chemical_el.py writes keys at %.6f, because at m0/t0 = 0.03 every
    physical mu rounds to "0.00" and the whole scan would collapse to one key.
    Two sweeps missed sites here, both for the same reason: they enumerated the
    variable names they expected (mu, mu_val, mu_t, ...) instead of matching any
    name.  fold_continuation_floor.py uses `mu0` and fold_transversality.py uses
    `mus[-1]`, so both survived, and the floor-control test -- the one that
    decides whether the fold is real -- died with KeyError on the first read of
    the regenerated data file.

    Match ANY identifier.  An enumeration of expected names is what failed.
    """
    import re as _re
    pat = _re.compile(r"""f["']mu\{[A-Za-z_][A-Za-z0-9_]*(?:\[[^\]]*\])?:\.2f\}_""")
    bad = []
    for rel in TK_SCRIPTS + ["tov_convergence_el.py", "tov_tk_chemical_el_fig.py",
                             "tov_tk_el_flux_fig.py"]:
        p = os.path.join(ROOT, rel)
        if not os.path.isfile(p):
            continue
        for i, line in enumerate(open(p), 1):
            if pat.search(line):
                bad.append(f"{rel}:{i}: {line.strip()[:80]}")
    assert not bad, (
        "npz mu keys formatted at 2 decimals; they must be %.6f or every mu at "
        "m0 = 0.03 collapses to one key:\n  " + "\n  ".join(bad))


def test_the_key_format_actually_matters_at_this_m0():
    """Guard: at m0 = 3 the old format was harmless, so the test above would have
    proved nothing.  It only bites because m0 is now 0.03."""
    ns, _ = _load_defs("tov_tk_chemical_el.py", ["_bond_correlations"])
    m0 = float(ns.get("m0", M0))
    # The real grid, so the collision count is the one the solver would hit.
    ratios = ([0.005, 0.010, 0.015, 0.020, 0.030, 0.040, 0.050, 0.060, 0.070,
               0.080, 0.090, 0.100] + [round(0.100 + 0.005 * i, 4) for i in range(1, 19)])
    two = {f"{r * m0:.2f}" for r in ratios}
    six = {f"{r * m0:.6f}" for r in ratios}
    assert len(six) == len(ratios), (
        f"%.6f must separate every mu at m0={m0}; it gives {len(six)} keys for "
        f"{len(ratios)} values")
    assert len(two) < len(ratios) / 2, (
        f"at m0={m0} the %.2f format must collide badly, or this test proves "
        f"nothing: it gives {len(two)} keys for {len(ratios)} values -> {sorted(two)}")


def test_continuation_steps_scale_with_m0():
    """FAILS IF: a mu-continuation step is written as an absolute constant.

    m0/t0 is a free input, so a step in absolute mu means a different physical
    step at every parameter point.  fold_continuation_floor.py used `step =
    0.005` and fold_transversality.py `step = 0.01`, both sized for m0 = 3.0.
    At m0 = 0.03 those are mu/m0 steps of 0.167 and 0.333 against a branch that
    spans 0.005 to 0.190 -- so the first trial step overshoots the entire branch,
    every halving stays coarse, and the floor-control test reported "0 steps"
    with a NaN eigenvalue on the run that was supposed to decide whether the fold
    is real.  The three mu-scan scripts already wrote `dmu = 0.005 * m0` and were
    unaffected, which is what the fix makes uniform.
    """
    import re as _re
    bad = []
    for rel in TK_SCRIPTS:
        p = os.path.join(ROOT, rel)
        if not os.path.isfile(p):
            continue
        for i, line in enumerate(open(p), 1):
            code = line.split("#")[0]
            m = _re.match(r"\s*(step|dmu|d_mu)\s*=\s*([0-9.eE+-]+)\s*$", code)
            if m:
                bad.append(f"{rel}:{i}: {m.group(1)} = {m.group(2)} (absolute)")
    assert not bad, (
        "continuation step written in absolute mu; it must scale with m0:\n  "
        + "\n  ".join(bad))


def test_active_set_requires_complementarity():
    """FAILS IF: the active-set loop accepts a set it has not verified.

    Freezing the bound set makes the residual smooth, but the frozen problem
    equals the ORIGINAL one only where V > 0 on the set and V <= 0 off it.
    Without that check the frozen residual can be driven to 1e-9 by an
    inconsistent set: the run at m0 = 0.03 produced "converged" points at
    mu/m0 = 0.26-0.36 with cores of 8, 34, 9 and 70 shells whose V was negative
    on shells they treated as bound.  They are not solutions.
    """
    src = open(os.path.join(ROOT, "tov_tk_chemical_el.py")).read()
    assert "on_ok and off_ok" in src, \
        "active_set_newton must verify complementarity before returning"
    assert "keep the best" not in src, \
        "the 2-cycle branch must REJECT, not return the current iterate"
    code = ast.unparse(ast.parse(src))
    fn = [n for n in ast.walk(ast.parse(src))
          if isinstance(n, ast.FunctionDef) and n.name == "active_set_newton"]
    assert fn, "active_set_newton not found"
    body = ast.unparse(fn[0])
    assert body.count("1e+30") >= 3 or body.count("1e30") >= 3, (
        "each non-convergent exit (inconsistent set, 2-cycle, outer-loop "
        "exhaustion) must return a rejecting residual")
