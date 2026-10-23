#!/usr/bin/env python
"""engine_adjoint_test_v1.py (reviewer A for the busi lane; WEB40I review, code defect 2, the gate of W2): the adjoint test of the
averaging on a small radial fixture built with the engine's own constructors; PASS or FAIL per check, with the numbers.

Where the three objects live (the line numbers are found in the installed modules' sources at run time and printed first):
  S, the source averaging: the shell totals eps_n = sum_kappa 16 kappa Re Tr(P_n h C) and J_n (with Q) of engine_collapse_v5.dens (input 06)
    are averaged into the fields' sources rho_ell(r) = sum_n K_0(r, R_n) eps_n and J_ell(r) = sum_n K_1(r, R_n) J_n by engine_fields_v12's
    SmoothedSources5 and kernels, imported unchanged by engine_fields_v16 (input 07, the module header "kernels = F12.kernels",
    "SmoothedSources5 = F12.SmoothedSources5"), built in solve_constraints_ivp (input 07: "src = SmoothedSources5(eps_tot, J_tot, R, ell,
    r_nodes[-1], support_rel)") and read by the constraints' right-hand side (integrate_UL.rhs_pos: "rho = float(rho_(r)); J = float(J_(r))");
    the same kernels appear as matrices in padded_collar_cost ("K0, K1 = kernels(rr, R[sel], ell); rho = K0 @ eps_tot[sel]; Jl = K1 @ J_tot[sel]").
    The kernel is the 3-dimensional Gaussian of standard deviation ell, angular averaged over the shell: K_l(r, R) = (2 pi ell^2)^(-3/2)
    exp(-(r^2 + R^2)/(2 ell^2)) i_l(r R/ell^2), i_0(x) = sinh x/x for the energy density (l = 0, a scalar) and i_1(x) = (x cosh x - sinh x)/x^2
    for the radial current (l = 1, a vector); centre_coefficients (input 07) carries their Taylor terms: c_n = exp(-R_n^2/(2 ell^2))/((2 pi)^1.5
    ell^3) = K_0(0, R_n), rho_2 = sum c_n e_n (R_n^2/(6 ell^4) - 1/(2 ell^2)), J_1 = sum c_n J_n R_n/(3 ell^2). Check (a0) below confirms the form.
  W, the radial weights: the spherical reduction's volume measure 4 pi r^2 dr, in which the constraints are integrated (input 07,
    integrate_UL.rhs_pos: U' = 2 fpi r^2 (rho + q J) with fpi = 4 pi g, so that the collar charge is E = U(r_collar)/(2 g) = 4 pi int r^2
    (rho_ell + q J_ell) dr, q = v/N); the shell totals carry unit site weight (eps_n is the shell's whole energy), equivalently the site
    densities eps_n/(4 pi R_n^2 a) carry W_site = 4 pi R_n^2 a: both conventions give the same adjoint.
  S*, the back-averaging: engine_fields_v12.site_fields_kernel_average, imported by engine_fields_v16 (input 07) and called by
    engine_collapse_v36d.solve (input 05: "Na, va, norms = FLD.site_fields_kernel_average(info, R, ell)" under --site-fields averaged, the
    production's setting); the engine states it as N_n = int 4 pi r^2 K_0(r, R_n) N(r) dr, v_n = int 4 pi r^2 K_1(r, R_n) v(r) dr (input 05,
    kernel_identities and the "kernel identities (site fields averaged)" line); the (N_n, v_n) so obtained are the fields of h_m =
    (1/2){N, h} + (1/2){v, Q} (input 05, clocked_gpu) that continues the reference and propagates the state.
The identity under test: with S_qn = K_l(r_q, R_n) on a radial quadrature grid r_q of weights W_q = 4 pi r_q^2 dr_q and unit site weights,
<N, S eps>_W = <S* N, eps> for every N and eps holds exactly when S* = W_site^(-1) S^T W, i.e. N_n = sum_q W_q K_l(r_q, R_n) N(r_q). Then the
first variation of the collar charge at fixed averaging is dE = int 4 pi r^2 (N d rho_ell + v d J_ell) dr (from (N dU)' = N 8 pi g r^2
(d rho + q dJ) with N(r_collar) = 1, the lapse and flow being the exact multipliers of the constraints) = sum_n (N_n d eps_n + v_n d J_n) =
Tr(h_m dC): the step's generator is the energy's gradient, which every conserving step needs (code defects 1 and 2).
The checks (PASS or FAIL per check):
  (a) the operator identity S* = W^(-1) S^T W to roundoff for both kernels: (a0) the engine's kernel matrices against the analytic form;
      (a1) their symmetry K_l(R_i, R_j) = K_l(R_j, R_i) (the volume-weighted adjoint of a symmetric kernel is the same kernel with the volume
      weight); (a2) the engine's back-averaging on polynomial fields against the exact Gaussian moments (N = 1 -> 1; r^2 -> R^2 + 3 ell^2;
      r^4 -> R^4 + 10 R^2 ell^2 + 15 ell^4; v = c r -> c R; v = c r^3 -> c (R^3 + 5 R ell^2)); (a3) the engine's back-averaging of the fixture's own
      dense fields against an independent converged quadrature of S^T W (composite Gauss-Legendre, panels of ell/4, 8 nodes, to R_max + 8 ell);
  (b) the weighted-charge defect: int 4 pi r^2 (N rho_ell + v J_ell) dr against sum_n (N_n eps_n + v_n J_n) with the engine's back-averaged
      site fields (0 expected), beside the point-sampled fields (the (ell/L)^2 control) and a violated back-averaging (the radial measure with
      the volume factor taken at the site: the reviewers' 18.9 percent on their fixture is of this class);
  (c) dE = Tr(h dC) for a random small Hermitian dC at fixed fields: the engine's own field solve at eps + t d eps, J + t dJ (d eps_n =
      Re Tr(P_n h dC) by the engine's dens) for t = +-1, +-1/2, +-1/4, +-1/8, the derivative by central differences with 3-level Richardson
      extrapolation in t^2 (the red team's form: with 3 sizes the un-extrapolated t^6 term, c6 t^6/64, was 6.6e-9 on the fixture, which the
      data could not separate from a defect of that size), against sum_n (N_n d eps_n + v_n dJ_n) with the solve's back-averaged fields; the
      check decides only when the last extrapolation changes the estimate by less than the admitted difference, which is the larger of --tol-c
      and three times the central difference's own floor (the charge's change under rtol 1e-10 against 1e-12 over the smallest perturbation's
      energy change); the base perturbation is halved up to twice until it decides; (c2) the same with a perturbation straddling the collar
      (informational: the collar charge does not see sources beyond the collar).
The fixture (the engine's own constructors): Chan(kappa, m, nsites, half=False) for kappa = 1 .. kappa_sea (engine_collapse_v5), the initial
data by engine_collapse_v36d.prepare_dust(chs, g, R, ell, N_B = r_s/(2 g m), R_0, m, reading "A", shell None) with OPT site_fields averaged
(the levitating well, the zero-temperature filling, the projection, every solve the initial-value problem), the sources the dust's shell
totals (the sea equals its reference at T = 0); defaults m 0.2, g 0.25, r_s/a 8, R_0 4 r_s, nsites 256, kappa_sea 3, ell 4 (every argument
printed). --repair: (b) and (c) repeated with the back-averaging replaced by the exact weighted adjoint (the quadrature of (a3)), so that a
failure is attributed to the back-averaging or not; the repair's replacement block for engine_fields_v16.py prints with the verdict.
Usage: python engine_adjoint_test_v1.py --code <dir holding engine_collapse_v36d.py and its modules> [--gpu N] [--repair] [--rs 8 --nsites 256
--kappa-sea 3 --ell 4 --m 0.2 --g 0.25 --R0-over-rs 4 --seed 1 --threads 16]"""
import argparse
import importlib
import inspect
import math
import os
import sys
import time

import numpy as np

REPAIR_BLOCK = r'''--- the averaging's repair, as an exact replacement in engine_fields_v16.py (input 07); applied only when a check above fails and the
--- repaired run below passes. OLD (exactly once in input 07):
site_fields_kernel_average = F12.site_fields_kernel_average
--- NEW:
def site_fields_kernel_average(info, R, ell, nodes_per_panel=8, panel_frac=0.25, tail=8.0):   # defect 2 repair
    """the back-averaging as the exact weighted adjoint of the source averaging (WEB40I code defect 2): N_n = int 4 pi r^2 K_0(r, R_n) N(r) dr,
    v_n = int 4 pi r^2 K_1(r, R_n) v(r) dr, by composite Gauss-Legendre (panels of panel_frac ell, nodes_per_panel nodes each) on
    [0, max R + tail ell] with the solve's dense fields (the vacuum exterior beyond the collar); norms = the quadrature of int 4 pi r^2 K_0 dr."""
    R = np.asarray(R, float); panel = float(ell) * panel_frac
    npan = int(np.ceil((float(R.max()) + tail * float(ell)) / panel))
    edges = np.linspace(0.0, npan * panel, npan + 1)
    xg, wg = np.polynomial.legendre.leggauss(nodes_per_panel)
    r_q = (0.5 * (edges[1:] + edges[:-1])[:, None] + 0.5 * panel * xg[None, :]).ravel()
    w_q = np.tile(0.5 * panel * wg, npan) * 4.0 * np.pi * r_q ** 2
    K0, K1 = kernels(r_q, R, ell)
    Nq = np.asarray(info["N_dense"](r_q), float).ravel(); vq = np.asarray(info["v_dense"](r_q), float).ravel()
    return (w_q * Nq) @ np.asarray(K0, float), (w_q * vq) @ np.asarray(K1, float), w_q @ np.asarray(K0, float)
--- Records it changes: every record of every history run with --site-fields averaged (every production history), from T = 0, at the relative
--- level of (a3)'s largest difference between the engine's back-averaging and the exact adjoint over the sites inside the collar (printed in
--- the verdict); below 1e-10 no printed digit of the stored records changes and the histories stand, above it they are re-run.'''


def locate(mod, pats):
    """the line numbers of the quoted statements in the installed module's source."""
    try:
        lines = inspect.getsource(mod).splitlines()
        path = inspect.getsourcefile(mod)
    except (OSError, TypeError):
        return [(getattr(mod, "__name__", "?"), p_, []) for p_ in pats]
    return [(os.path.basename(path or "?"), p_, [i + 1 for i, l_ in enumerate(lines) if p_ in l_]) for p_ in pats]


def show_lines(title, rows):
    print(title)
    for fn, pat, hits in rows:
        print("  %s line %s: %s" % (fn, (", ".join(str(h) for h in hits) if hits else "not found"), pat))


def gauss_grid(r_end, ell, panel_frac=0.25, nodes=8):
    """composite Gauss-Legendre on [0, r_end] (panels of panel_frac ell with nodes per panel): r_q and the volume weights 4 pi r_q^2 dr_q."""
    panel = float(ell) * panel_frac
    npan = int(math.ceil(r_end / panel))
    edges = np.linspace(0.0, npan * panel, npan + 1)
    xg, wg = np.polynomial.legendre.leggauss(nodes)
    r_q = (0.5 * (edges[1:] + edges[:-1])[:, None] + 0.5 * panel * xg[None, :]).ravel()
    w_q = np.tile(0.5 * panel * wg, npan) * 4.0 * np.pi * r_q ** 2
    return r_q, w_q


def kmat(FLD, r, Rs, ell):
    """the engine's kernel matrices shaped (len(r), len(Rs)): padded_collar_cost's convention rho = K0 @ eps[sel]."""
    K0, K1 = FLD.kernels(np.asarray(r, float), np.asarray(Rs, float), ell)
    K0 = np.asarray(K0, float); K1 = np.asarray(K1, float)
    if K0.shape == (len(Rs), len(r)) and len(r) != len(Rs):
        K0, K1 = K0.T, K1.T
    return K0, K1


def analytic_kernels(r, Rs, ell):
    """K_l(r, R) = (2 pi ell^2)^(-3/2) exp(-(r^2 + R^2)/(2 ell^2)) i_l(r R/ell^2) in the overflow-free form exp(-(r - R)^2/(2 ell^2)) times
    (1 - e^(-2x))/(2x) for l = 0 and (x (1 + e^(-2x)) - (1 - e^(-2x)))/(2 x^2) for l = 1, x = r R/ell^2."""
    r = np.asarray(r, float)[:, None]; Rs = np.asarray(Rs, float)[None, :]
    x = r * Rs / ell ** 2
    pre = (2.0 * np.pi * ell ** 2) ** (-1.5) * np.exp(-(r - Rs) ** 2 / (2.0 * ell ** 2))
    em = np.exp(-2.0 * x)
    K0 = pre * (1.0 - em) / (2.0 * x)
    K1 = pre * (x * (1.0 + em) - (1.0 - em)) / (2.0 * x ** 2)
    return K0, K1


def exact_adjoint(FLD):
    """the exact weighted adjoint S^T W of the engine's kernels (the repair's function)."""
    def f(info, R, ell, nodes_per_panel=8, panel_frac=0.25, tail=8.0):
        R = np.asarray(R, float)
        r_q, w_q = gauss_grid(float(R.max()) + tail * float(ell), ell, panel_frac, nodes_per_panel)
        K0, K1 = kmat(FLD, r_q, R, ell)
        Nq = np.asarray(info["N_dense"](r_q), float).ravel(); vq = np.asarray(info["v_dense"](r_q), float).ravel()
        return (w_q * Nq) @ K0, (w_q * vq) @ K1, w_q @ K0
    return f


def verdict(name, ok, text):
    print("%s %s: %s" % ("PASS" if ok else "FAIL", name, text), flush=True)
    return bool(ok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", default=None, help="the directory holding engine_collapse_v36d.py and its modules (added to sys.path)")
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--m", type=float, default=0.2)
    ap.add_argument("--g", type=float, default=0.25)
    ap.add_argument("--rs", type=float, default=8.0)
    ap.add_argument("--R0-over-rs", dest="R0_over_rs", type=float, default=4.0)
    ap.add_argument("--nsites", type=int, default=256)
    ap.add_argument("--kappa-sea", dest="kappa_sea", type=int, default=3)
    ap.add_argument("--ell", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--repair", action="store_true", help="repeat (b) and (c) with the exact weighted adjoint in place of the engine's back-averaging")
    ap.add_argument("--tol-a", dest="tol_a", type=float, default=1e-11, help="(a2), (a3): the largest absolute difference admitted (N of order 1)")
    ap.add_argument("--tol-b", dest="tol_b", type=float, default=1e-11, help="(b): the relative weighted-charge defect admitted")
    ap.add_argument("--tol-c", dest="tol_c", type=float, default=3e-9, help="(c): the relative difference admitted (the field solve's floor; printed)")
    a = ap.parse_args()
    if a.code:
        sys.path.insert(0, os.path.abspath(a.code))
    import engine_collapse_v5 as C5
    import engine_collapse_v36d as E
    import engine_fields_v16 as FLD
    try:
        F12 = importlib.import_module("engine_fields_v12")
    except Exception as ex:  # noqa: BLE001
        F12 = None
        print("note: engine_fields_v12 is not importable (%s): the kernel bodies' lines are not printed" % type(ex).__name__)
    print("engine_adjoint_test_v1: the averaging's adjoint test (WEB40I code defect 2; the gate of W2)")
    show_lines("S and W, the source averaging and the radial weights (input 07, engine_fields_v16.py):", locate(FLD, [
        "kernels = F12.kernels", "SmoothedSources5 = F12.SmoothedSources5", "site_fields_kernel_average = F12.site_fields_kernel_average",
        "src = SmoothedSources5(eps_tot, J_tot, R, ell, r_nodes[-1], support_rel)", "rho = float(rho_(r)); J = float(J_(r))",
        "return [2.0 * fpi * r * r * (rho + q * J), -fpi * r * J / q]", "K0, K1 = kernels(rr, R[sel], ell)",
        "c = np.exp(-Rs ** 2 / (2.0 * ell ** 2)) / ((2.0 * np.pi) ** 1.5 * ell ** 3)", "E = float(solW.sol(r_collar)[0] / (2.0 * g))"]))
    show_lines("S*, the back-averaging, and its use in h_m (input 05, engine_collapse_v36d.py):", locate(E, [
        "def solve(eps, J, g, R, ell, N0, rtol=None, atol=None):", "Na, va, norms = FLD.site_fields_kernel_average(info, R, ell)",
        "def kernel_identities(R, ell, c=0.37):", "def clocked_gpu(ch, N, v):", "M.index_put_((hr, hcl), hv * (0.5 * (Nd[hr] + Nd[hcl])).to(torch.complex128), accumulate=True)"]))
    show_lines("the site densities (input 06, engine_collapse_v5.py):", locate(C5, ["def dens(A, Phi, f=None):"]))
    if F12 is not None:
        show_lines("the kernel bodies (engine_fields_v12.py, not among the inputs; called as the engine calls them):", locate(F12, [
            "def kernels", "class SmoothedSources5", "def site_fields_kernel_average", "def charge"]))
    print("the identity: S_qn = K_l(r_q, R_n); W_q = 4 pi r_q^2 dr_q on the radial grid, unit weight on the shell totals (W_site = 4 pi R_n^2 a on site "
          "densities, the same adjoint); S* = W_site^(-1) S^T W <=> N_n = int 4 pi r^2 K_0(r, R_n) N(r) dr, v_n likewise with K_1; then dE = Tr(h_m dC).")
    C5.backend(a.gpu)
    torch = C5.BK["torch"]
    if a.gpu is None:
        torch.set_num_threads(max(1, int(a.threads)))
    E.OPT.update({"site_fields": "averaged", "collar_policy": "flag", "support_rel": 1e-14})
    E.SOLVE_TOL.update({"rtol": 1e-12, "atol": 1e-16})
    rs, g, m, ell = a.rs, a.g, a.m, a.ell
    NB = rs / (2.0 * g * m); R0 = a.R0_over_rs * rs
    chs = [C5.Chan(k, m, a.nsites, False) for k in range(1, a.kappa_sea + 1)]
    R = np.asarray(chs[0].R, float); n = len(R)
    print("the fixture: Chan(kappa, m %g, nsites %d, half False) for kappa 1..%d (weights 16 kappa); prepare_dust(chs, g %g, R, ell %d, N_B %.4f, R_0 %.2f a (%g r_s), m %g, "
          "reading A, shell None); r_s/a %g; OPT site_fields averaged, collar policy flag, support_rel 1e-14; solve rtol 1e-12 atol 1e-16; device %s" % (
              m, a.nsites, a.kappa_sea, g, ell, NB, R0, a.R0_over_rs, m, rs, "cpu" if a.gpu is None else "cuda:%d" % a.gpu), flush=True)
    t0 = time.time()
    N, v, dust, info0, head = E.prepare_dust(chs, g, R, ell, NB, R0, m, "A", None)
    print("the initial data: N_B %.6f, E0 %.10f, passes %d, last change %.2e, f_kin %.5f, discarded %.2e, Gram cond %.2e, collar margin %s (%.0f s)" % (
        head["N_B"], head["E0"], head["passes"], head["last_change"], head["f_kin"], head["projection"]["discarded_norm_max"], head["projection"]["gram_cond_max"],
        head.get("collar_margin"), time.time() - t0), flush=True)
    kid = E.kernel_identities(R, ell)
    print("the engine's own kernel identities: K_0 on 1: max |. - 1| %.2e; K_1 on c r: max abs %.2e (relative %.2e); K_0 weight sums %.15f to %.15f" % (
        kid["K0_constant_max_abs"], kid["K1_linear_max_abs"], kid["K1_linear_max_rel"], kid["K0_weight_sums"][0], kid["K0_weight_sums"][1]))
    eps = np.zeros(n); J = np.zeros(n)
    for ch in chs:
        if ch.kappa in dust:
            Phi, fk = dust[ch.kappa]
            eps += ch.w * C5.dens(ch.h, Phi, fk); J += ch.w * C5.dens(ch.Q, Phi, fk)
    Na, va, Wn, info = E.solve(eps, J, g, R, ell, None)
    if info.get("failed"):
        raise SystemExit("the fixture's solve failed: %s" % info["failed"])
    Na = np.asarray(Na, float); va = np.asarray(va, float)
    Np = np.asarray(info["N_point"], float); vp = np.asarray(info["v_point"], float)
    E0 = float(FLD.charge(info, g)); r_c = float(info["r_collar"])
    cm = info.get("collar_margin") or {}
    print("the fixture's solve at the dust's sources: E0 %.12f (info E_collar %.12f); r_collar %.1f (R_last %.2f, deficit %s ell); residuals W %.1e 1/N %.1e; source spline error %s; "
          "sum eps %.6f, sum J %.6f; sites with |source| > 0: %d" % (E0, float(info["E_collar"]), r_c, float(info["R_last"]), cm.get("deficit_ell"), info.get("residual_W") or 0.0,
                                                                     info.get("residual_invN") or 0.0, info.get("source_spline_error"), eps.sum(), J.sum(), int(np.sum((eps != 0) | (J != 0)))), flush=True)
    results = {}
    # ---------------------------------------------------------------------------------------------------- (a0) the kernels' analytic form
    rr = np.linspace(0.37, R[-1] + 2.0 * ell, 1801)
    K0e, K1e = kmat(FLD, rr, R, ell)
    K0a, K1a = analytic_kernels(rr, R, ell)
    d0 = float(np.max(np.abs(K0e - K0a))) / float(np.max(np.abs(K0a))); d1 = float(np.max(np.abs(K1e - K1a))) / float(np.max(np.abs(K1a)))
    results["a0"] = verdict("(a0) the engine's kernels are the angular-averaged 3D Gaussian of width ell", d0 < 1e-12 and d1 < 1e-12,
                            "max relative difference K_0 %.2e, K_1 %.2e on %d points x %d sites (the docstring's identification holds where these are at roundoff; "
                            "otherwise the engine's kernel differs from the stated form and (a1) to (a3) still test the adjoint of whatever it is)" % (d0, d1, len(rr), n))
    # ---------------------------------------------------------------------------------------------------- (a1) the symmetry of the kernel matrices
    K0s, K1s = kmat(FLD, R, R, ell)
    s0 = float(np.max(np.abs(K0s - K0s.T))) / float(np.max(np.abs(K0s))); s1 = float(np.max(np.abs(K1s - K1s.T))) / float(np.max(np.abs(K1s)))
    results["a1"] = verdict("(a1) K_l(R_i, R_j) = K_l(R_j, R_i)", s0 < 1e-13 and s1 < 1e-13, "max |K - K^T|/max |K| = %.2e (l = 0), %.2e (l = 1) on the %d sites" % (s0, s1, n))
    # ---------------------------------------------------------------------------------------------------- (a2) the moments
    one = lambda r: np.ones_like(np.atleast_1d(np.asarray(r, float)))
    zero = lambda r: np.zeros_like(np.atleast_1d(np.asarray(r, float)))
    c = 0.37
    moments = [("K_0 on N = 1", 0, one, np.ones(n)),
               ("K_0 on N = r^2", 0, (lambda r: np.atleast_1d(np.asarray(r, float)) ** 2), R ** 2 + 3.0 * ell ** 2),
               ("K_0 on N = r^4", 0, (lambda r: np.atleast_1d(np.asarray(r, float)) ** 4), R ** 4 + 10.0 * R ** 2 * ell ** 2 + 15.0 * ell ** 4),
               ("K_1 on v = c r", 1, (lambda r: c * np.atleast_1d(np.asarray(r, float))), c * R),
               ("K_1 on v = c r^3", 1, (lambda r: c * np.atleast_1d(np.asarray(r, float)) ** 3), c * (R ** 3 + 5.0 * R * ell ** 2))]
    inner = R <= R[-1] - 8.0 * ell
    worst_a2 = 0.0; txt = []
    for name, l_, f_, exact in moments:
        Na_, va_, nr_ = FLD.site_fields_kernel_average({"N_dense": (f_ if l_ == 0 else one), "v_dense": (f_ if l_ == 1 else zero)}, R, ell)
        got = np.asarray(Na_ if l_ == 0 else va_, float)
        rel = float(np.max(np.abs(got - exact)[inner])) / max(float(np.max(np.abs(exact[inner]))), 1e-300)
        worst_a2 = max(worst_a2, rel); txt.append("%s %.2e" % (name, rel))
    results["a2"] = verdict("(a2) the engine's back-averaging reproduces the exact Gaussian moments", worst_a2 < 1e-12,
                            "relative errors (normalized by the moment's largest value over the sites at least 8 ell inside the chain's end): %s" % "; ".join(txt))
    # ---------------------------------------------------------------------------------------------------- (a3) against the independent quadrature of S^T W
    adj = exact_adjoint(FLD)
    Na_x, va_x, nr_x = adj(info, R, ell)
    Na_e, va_e, nr_e = FLD.site_fields_kernel_average(info, R, ell)
    Na_e = np.asarray(Na_e, float); va_e = np.asarray(va_e, float); nr_e = np.asarray(nr_e, float)
    same = bool(np.array_equal(Na_e, Na) and np.array_equal(va_e, va))
    dN_in = float(np.max(np.abs(Na_e - Na_x)[inner])); dv_in = float(np.max(np.abs(va_e - va_x)[inner]))
    dN_all = float(np.max(np.abs(Na_e - Na_x))); dv_all = float(np.max(np.abs(va_e - va_x)))
    ext = R >= r_c
    results["a3"] = verdict("(a3) the engine's back-averaging equals the converged quadrature of S^T W on the fixture's dense fields", dN_in < a.tol_a and dv_in < a.tol_a,
                            "max |N_n^engine - N_n^adjoint| %.2e, |v_n| %.2e over the sites inside (8 ell from the chain's end); over all sites %.2e, %.2e; the solve's returned "
                            "site fields %s the direct back-averaging; the engine's weight sums %.15f to %.15f, the quadrature's %.15f to %.15f; beyond the collar (%d sites) the "
                            "engine's N_n lies in [%.12f, %.12f] (1 expected: the vacuum exterior)" % (
                                dN_in, dv_in, dN_all, dv_all, "equal bitwise" if same else "DIFFER from", nr_e.min(), nr_e.max(), nr_x.min(), nr_x.max(), int(ext.sum()),
                                float(Na_e[ext].min()) if ext.any() else float("nan"), float(Na_e[ext].max()) if ext.any() else float("nan")))
    # ---------------------------------------------------------------------------------------------------- (b) the weighted-charge defect
    r_q, w_q = gauss_grid(R[-1] + 8.0 * ell, ell)
    S0, S1 = kmat(FLD, r_q, R, ell)
    Nq = np.asarray(info["N_dense"](r_q), float).ravel(); vq = np.asarray(info["v_dense"](r_q), float).ravel()
    src = info["src"]
    rho_d, J_d = src.direct(r_q); rho_d = np.asarray(rho_d, float).ravel(); J_d = np.asarray(J_d, float).ravel()
    dk0 = float(np.max(np.abs(rho_d - S0 @ eps))); dk1 = float(np.max(np.abs(J_d - S1 @ J)))
    ins = r_q <= r_c
    rho_s = np.asarray(src.rho(r_q[ins]), float).ravel(); J_s = np.asarray(src.J(r_q[ins]), float).ravel()
    LHS = float(np.sum(w_q * (Nq * rho_d + vq * J_d)))
    LHS_in = float(np.sum((w_q * (Nq * rho_d + vq * J_d))[ins]))
    LHS_spl = float(np.sum(w_q[ins] * (Nq[ins] * rho_s + vq[ins] * J_s)))
    RHS_avg = float(np.sum(Na * eps + va * J)); RHS_pt = float(np.sum(Np * eps + vp * J))
    w_r = w_q / (4.0 * np.pi * r_q ** 2)
    Nv = 4.0 * np.pi * R ** 2 * ((w_r * Nq) @ S0); vv = 4.0 * np.pi * R ** 2 * ((w_r * vq) @ S1)
    RHS_v = float(np.sum(Nv * eps + vv * J))
    def_avg = abs(LHS - RHS_avg) / abs(LHS); def_pt = abs(LHS - RHS_pt) / abs(LHS); def_v = abs(LHS - RHS_v) / abs(LHS)
    results["b"] = verdict("(b) the weighted-charge defect", def_avg < a.tol_b,
                           "int 4 pi r^2 (N rho_ell + v J_ell) dr = %.12f (inside the collar %.12f; with the solver's source splines %.12f, the spline level %.1e); "
                           "sum_n (N_n eps_n + v_n J_n) with the engine's back-averaged fields %.12f: defect %.2e; point-sampled fields (the control) %.2e; the violated "
                           "back-averaging (dr with 4 pi R_n^2 at the site) %.2e; the engine's direct smoothed sources against K0 @ eps, K1 @ J: %.1e, %.1e" % (
                               LHS, LHS_in, LHS_spl, abs(LHS_spl - LHS_in) / abs(LHS), RHS_avg, def_avg, def_pt, def_v, dk0, dk1))
    # ---------------------------------------------------------------------------------------------------- (c) dE = Tr(h dC)
    rng = np.random.default_rng(a.seed)

    def perturbation(supp, kcols=8):
        deps = np.zeros(n); dJ = np.zeros(n); pert = {}
        for ch in chs:
            Y = rng.standard_normal((2 * n, kcols)) + 1j * rng.standard_normal((2 * n, kcols))
            Y[np.repeat(~supp, 2), :] = 0.0
            Y, _ = np.linalg.qr(Y)
            D = rng.uniform(-1.0, 1.0, kcols)
            Yt = C5.T_(Y)
            deps += ch.w * C5.dens(ch.h, Yt, D); dJ += ch.w * C5.dens(ch.Q, Yt, D)
            pert[ch.kappa] = (Yt, D)
        return deps, dJ, pert

    def derivative(deps, dJ, tbase, levels=4):
        """the derivative of the collar charge along (deps, dJ) by central differences at tbase/2^k, k = 0 .. levels - 1, extrapolated in powers
        of t^2 (the Richardson table: column k holds the k-fold extrapolations, its first entry from the largest sizes); returns the last column's
        entry, the central differences by size, the table and the solves' info by size. The red team's form: three sizes and two levels leave the
        remainder c6 t^6/64, which on the fixture was the whole 6.6e-9 against the admitted 3e-9; four sizes leave c8 t^8/4096 and the last
        level's change measures the estimate's own uncertainty."""
        def E_at(t):
            _N, _v, _W, inf_ = E.solve(eps + t * deps, J + t * dJ, g, R, ell, None)
            if inf_.get("failed"):
                raise RuntimeError("the solve failed at t = %g: %s" % (t, inf_["failed"]))
            return float(FLD.charge(inf_, g)), inf_
        ts = [tbase / 2.0 ** k for k in range(levels)]
        D1 = {}; info_t = {}
        for t in ts:
            Ep, ip = E_at(t); Em, im = E_at(-t)
            D1[t] = (Ep - Em) / (2.0 * t); info_t[t] = (Ep, Em, ip, im)
        tab = [[D1[t] for t in ts]]
        for k in range(1, levels):
            f = 4.0 ** k; prev = tab[-1]
            tab.append([(f * prev[j + 1] - prev[j]) / (f - 1.0) for j in range(len(prev) - 1)])
        return tab[-1][0], D1, tab, info_t

    supp = R <= R0 + 4.0 * ell
    deps, dJ, pert = perturbation(supp)
    G0 = float(np.sum(Na * deps + va * dJ))
    scale = 0.01 * E0 / abs(G0)
    deps *= scale; dJ *= scale
    pert = {k_: (q[0], q[1] * scale) for k_, q in pert.items()}
    G_avg = float(np.sum(Na * deps + va * dJ)); G_pt = float(np.sum(Np * deps + vp * dJ)); G_v = float(np.sum(Nv * deps + vv * dJ))
    G_direct = 0.0
    for ch in chs:
        Yt, D = pert[ch.kappa]
        hm = E.clocked_gpu(ch, Na, va)
        G_direct += ch.w * float(np.dot(D, (Yt.conj() * (hm @ Yt)).sum(dim=0).real.cpu().numpy()))
        del hm
    print("(c) dC: per channel 8 random orthonormal columns supported on the sites R <= %.1f (the ball), weights uniform in [-1, 1], scaled so that Tr(h_m dC) = 1e-2 E0 at t = 1; "
          "Tr(h_m dC) by the site pairing %.12e, by the dense operator %.12e (difference %.1e): the pairing is the trace" % (R0 + 4.0 * ell, G_avg, G_direct, abs(G_avg - G_direct) / abs(G_avg)))
    _n1, _v1, _w1, i10 = FLD.solve_constraints_ivp(eps, J, g, R, ell, rtol=1e-10, atol=1e-14, collar_policy=E.OPT["collar_policy"], support_rel=E.OPT["support_rel"])
    dEr = abs(float(FLD.charge(i10, g)) - E0) / abs(E0) if not i10.get("failed") else float("nan")
    dEdt = None; conv = float("nan"); adm = a.tol_c; floor = float("nan"); D1 = {}; tab = []
    for tbase in (1.0, 0.5, 0.25):
        try:
            dEdt, D1, tab, info_t = derivative(deps, dJ, tbase)
        except RuntimeError as ex:
            print("  (c) %s; the perturbation reduced" % ex); dEdt = None
            continue
        for t in sorted(D1, reverse=True):
            Ep, Em, ip, im = info_t[t]
            print("  (c) t %g: E(+t)/E0 - 1 = %+.6e, E(-t)/E0 - 1 = %+.6e, central difference %.12e; collar %s / %s" % (
                t, Ep / E0 - 1.0, Em / E0 - 1.0, D1[t], (ip.get("collar_margin") or {}).get("deficit_ell"), (im.get("collar_margin") or {}).get("deficit_ell")))
        print("  (c) the Richardson table at tbase %g, each entry over Tr(h_m dC) minus 1: %s" % (
            tbase, " | ".join("level %d: %s" % (k, " ".join("%+.2e" % (q / G_avg - 1.0) for q in col)) for k, col in enumerate(tab))))
        floor = 2.0 * dEr / (0.01 * min(D1)) if np.isfinite(dEr) else float("nan")      # the central difference's noise floor at the smallest size: the charge's integration error over its energy change there
        adm = max(a.tol_c, 3.0 * floor) if np.isfinite(floor) else a.tol_c
        conv = abs(tab[-1][0] - tab[-2][-1]) / abs(G_avg)          # the last extrapolation's change: the estimate's own uncertainty
        if conv < adm:
            break
        print("  (c) the extrapolation's last change %.1e relative exceeds the admitted %.1e (the option %.1e, three times the floor %.1e): the base perturbation halved" % (conv, adm, a.tol_c, floor))
    if dEdt is None:
        results["c"] = verdict("(c) dE = Tr(h dC)", False, "the field solve failed at the perturbation sizes tried")
    else:
        rel = abs(dEdt - G_avg) / abs(G_avg); rel_pt = abs(dEdt - G_pt) / abs(G_avg); rel_v = abs(dEdt - G_v) / abs(G_avg)
        decides = bool(conv < adm)
        results["c"] = verdict("(c) dE = Tr(h dC)", decides and rel < adm,
                               "dE/dt (Richardson over t = %s, %d levels; the last extrapolation changed the estimate by %.1e relative: the estimate %s) = %.12e against "
                               "Tr(h_m dC) = %.12e with the back-averaged fields: relative difference %.2e (admitted %.1e: the larger of --tol-c %.1e and three times the "
                               "central difference's floor %.1e, the charge changing by %.1e under rtol 1e-10 over the smallest perturbation's energy change); with the "
                               "point-sampled fields %.2e; with the violated back-averaging %.2e" % (
                                   ", ".join("%g" % q for q in sorted(D1, reverse=True)), len(tab), conv,
                                   "decides" if decides else "does NOT decide (the FAIL is the finite difference's, not the averaging's: (b) and --repair settle the averaging)",
                                   dEdt, G_avg, rel, adm, a.tol_c, floor, dEr, rel_pt, rel_v))
    # (c2) informational: a perturbation straddling the collar
    if r_c + 12.0 * ell < R[-1]:
        supp2 = np.abs(R - r_c) <= 4.0 * ell
        deps2, dJ2, pert2 = perturbation(supp2)
        G2 = float(np.sum(Na * deps2 + va * dJ2))
        sc2 = 0.01 * E0 / abs(G2) if G2 != 0.0 else 1.0
        deps2 *= sc2; dJ2 *= sc2
        try:
            d2, D12, R12, inf2 = derivative(deps2, dJ2, 1.0)
            G2 = float(np.sum(Na * deps2 + va * dJ2)); G2_in = float(np.sum((Na * deps2 + va * dJ2)[R < r_c]))
            print("(c2) informational, a perturbation on |R - r_collar| <= 4 ell: dE/dt = %.6e against Tr(h_m dC) = %.6e (its part inside the collar %.6e): the collar charge "
                  "does not see sources beyond the collar while h_m carries the vacuum fields there; the defect per unit source change at the collar is %.2e of Tr(h_m dC); in a "
                  "history it multiplies the sources' share within 8 ell of the collar (the record's collar_margin.eps_share_last_8ell)" % (d2, G2, G2_in, abs(d2 - G2) / abs(G2)))
        except RuntimeError as ex:
            print("(c2) informational: %s" % ex)
    # ---------------------------------------------------------------------------------------------------- the verdict and the repair
    keys = ["a0", "a1", "a2", "a3", "b", "c"]
    ok_all = all(results.get(k_, False) for k_ in keys)
    print("SUMMARY: %s -> %s" % (", ".join("(%s) %s" % (k_, "PASS" if results.get(k_) else "FAIL") for k_ in keys), "PASS: the back-averaging is the weighted adjoint of the source "
          "averaging and dE = Tr(h dC) to the field solve's floor; the gate of W2 is open" if ok_all else "FAIL: see the repair below"))
    print("the averaging's mismatch to the exact adjoint inside the collar is %.1e in N and %.1e in v (a3); per record of a step it multiplies the first-order change of the "
          "sources, Tr(dh dC) of order tau times the energy flux, so a mismatch below 1e-10 gives a drift per record below 1e-13 of E, a mismatch of order 1e-2 (point sampling) "
          "of order 1e-7 and of order 0.2 (the violated form) of order 1e-5 of E" % (dN_in, dv_in))
    if not ok_all or a.repair:
        print(REPAIR_BLOCK)
        if a.repair:
            print("--repair: the back-averaging replaced by the exact weighted adjoint; (b) and (c) repeated")
            FLD.site_fields_kernel_average = adj
            Na_r, va_r, _Wr, info_r = E.solve(eps, J, g, R, ell, None)
            Na_r = np.asarray(Na_r, float); va_r = np.asarray(va_r, float)
            RHS_r = float(np.sum(Na_r * eps + va_r * J))
            print("  repaired (b): defect %.2e (against the same int 4 pi r^2 (N rho_ell + v J_ell) dr)" % (abs(LHS - RHS_r) / abs(LHS)))
            if dEdt is not None:
                G_r = float(np.sum(Na_r * deps + va_r * dJ))
                print("  repaired (c): relative difference %.2e (dE/dt unchanged: the charge does not depend on the back-averaging; the pairing does)" % (abs(dEdt - G_r) / abs(G_avg)))
            print("  records changed by the repair: every record of every --site-fields averaged history from T = 0, at the relative level %.1e (a3); below 1e-10 no printed "
                  "digit changes and the stored histories stand" % max(dN_in, dv_in))
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
