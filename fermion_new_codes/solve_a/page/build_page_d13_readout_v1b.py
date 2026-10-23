#!/usr/bin/env python
"""build_page_d13_readout_v1b.py: page_d13_readout_v1b.py built from page_d13_readout_v1.py by an exact replacement list (each "old"
required to match exactly once in the evolving text; abort otherwise); v1 is not touched. Usage:
    python build_page_d13_readout_v1b.py [--src page_d13_readout_v1.py] [--dst <same directory>/page_d13_readout_v1b.py]
The built file is compiled (py_compile) and the replacements are listed; page_history_v6n imports it after build_page_history_v6n_b.py.

v1b adds (a'), hreduce's EVOLVEDCUT EC1 (|S_X(C) - S_X(P)| <= 2 ||Pi_X (C - P) Pi_Xc||_1 <= ||C - P||_1 for a pure Gaussian state C and a
projector P), at every clock where the held Killing vacuum P_- is formed and at every cut X = {R > x}:
  t_X = 2 ||Pi_X (C - P_-) Pi_Xc||_1 from the singular values of the X-by-Xc block of C - P_-, formed from the columns as
  Phi_X Phi_Xc^dagger - V_X V_Xc^dagger (n_X x n_Xc, two GEMMs on the row slices; never an n2 x n2 or N_occ x N_occ object) and resolved by
  torch.linalg.svdvals (--cut-block-route svd; gram: the smaller Gram's eigvalsh); the check |S_X(C) - S_X(P_-)| <= t_X printed per cut, a
  FAIL line naming the premises (C pure, P_- a projector) with their Gram defects; the full nu = ||C - P_-||_1 by the principal angles between
  ran C and ran P_- where affordable (--cut-norm-full auto|gram|svd), else its labelled bounds (rank x largest singular value, Cauchy-Schwarz);
  the weighted values (16 kappa) in the CLI's clock summary and in the ksum table; the singular values of every block (sv_block_cut<x>) and
  of (1 - C) V (sv_full) in the npz; the memory figure's cross-cut transients. New arguments: --cut-norm, --cut-block-route, --cut-norm-full,
  --cut-norm-full-GB, --cut-bound-tol, --cut-sigma-tol (defaults on, svd, auto, 24, 1e-6, 1e-10)."""
import argparse
import hashlib
import os
import py_compile
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# ------------------------------------------------------------------------------------------------------------ 1. the docstring head
DOC_OLD = r'''"""page_d13_readout_v1.py (reviewer A for the solve builder, the busi lane; reorientation 4, the solve entry's item 1; D13's protocol'''

DOC_NEW = r'''"""page_d13_readout_v1b.py: page_d13_readout_v1.py with (a'), the cross-cut block trace norms of hreduce's EVOLVEDCUT_theorem_v1 EC1
(built from v1 by build_page_d13_readout_v1b.py's exact replacement list; v1 is not touched). At every clock where the held Killing vacuum
P_-(h(t)) is formed (the baseline clocks, T_0 included) and at every cut X = {R > x}:
    t_X = 2 ||Pi_X (C - P_-) Pi_Xc||_1, the exact bound |S_X(C) - S_X(P_-)| <= t_X <= ||C - P_-||_1 for the pure Gaussian state C = Phi Phi^dagger
    and the projector P_- = V V^dagger, from the singular values of the X-by-Xc block of C - P_- formed from the columns as
    Phi_X Phi_Xc^dagger - V_X V_Xc^dagger (n_X x n_Xc; two GEMMs on the row slices, views when the rows are ordered; never an n2 x n2 or
    N_occ x N_occ object), resolved by torch.linalg.svdvals (cross_cut_norms states the route and its cost at n = 16384 on an H100: the
    block at 48 r_s is 26624 x 6144, 2.6 GiB, its GEMMs about a second, its singular values seconds to tens of seconds; the four cuts under a
    minute and under 8 GiB transient per baseline clock); the check |S_X(C) - S_X(P_-)| <= t_X printed per cut, a violation a FAIL line
    carrying the premises' defects ||Phi^dagger Phi - 1||_F, ||V^dagger V - 1||_F and the purity gap (the theorem's premises are the pure
    Gaussian state and the projector P_-); the consistency of the block with (a)'s spectra (the Frobenius identity and the triangle bounds);
    the full nu = ||C - P_-||_1 by the principal angles between ran C and ran P_- where affordable (full_trace_norm: gram, the N_neg x N_neg
    eigvalsh of 1 - M^dagger M with M = Phi^dagger V, about 16 GiB and tens of seconds at n = 16384; svd, exact to rounding, minutes; auto picks
    gram within --cut-norm-full-GB and the free device memory), else its labelled bounds by the rank and the largest singular value and by
    Cauchy-Schwarz with ||C - P_-||_2 exact; the per-channel values in the cut dicts (t_X, abs_dS, slack, bound_ok, the singular values'
    count and largest) and in sc["trace_norm"], the weighted values (16 kappa) in the CLI's clock summary (with the weighted check) and in
    the ksum table (t_X_cut<x>, nu_trace_weighted, bound_ok_cut<x>); the singular values of every block (sv_block_cut<x>) and of (1 - C) V
    (sv_full) in the npz. Thresholds: --cut-norm, --cut-block-route, --cut-norm-full, --cut-norm-full-GB, --cut-bound-tol, --cut-sigma-tol.
    cross_cut_norms(A, B, R2, cuts) serves any two orthonormal column sets (the reference's vacuum against the current one, for instance).
v1's docstring follows.

page_d13_readout_v1.py (reviewer A for the solve builder, the busi lane; reorientation 4, the solve entry's item 1; D13's protocol'''

# ------------------------------------------------------------------------------------------------------------ 3. the arguments
ARGS_OLD = '    g.add_argument("--d13-vacuum", dest="d13_vacuum", default="on", choices=["on", "off"], help="off: no dense vacuum (the vacuum columns absent)")\n'
ARGS_NEW = ARGS_OLD + r'''    g.add_argument("--cut-norm", dest="cut_norm", default="on", choices=["on", "off"],
                   help="(a') v1b: the cross-cut block trace norms t_X = 2||Pi_X (C - P_-) Pi_Xc||_1 and the check |S_X(C) - S_X(P_-)| <= t_X at the clocks with the vacuum")
    g.add_argument("--cut-block-route", dest="cut_block_route", default="svd", choices=["svd", "gram"],
                   help="(a') the block's singular values: svdvals of the n_X x n_Xc block (exact to rounding) or the smaller Gram's eigvalsh (the small ones to sqrt accuracy)")
    g.add_argument("--cut-norm-full", dest="cut_norm_full", default="auto", choices=["auto", "gram", "svd", "bound", "off"],
                   help="(a') the full ||C - P_-||_1: gram (eigvalsh of 1 - M^dagger M, M = Phi^dagger V), svd (svdvals of (1 - C) V, exact), bound (rank x sigma_max and Cauchy-Schwarz, no N x N object), auto (gram where its memory fits --cut-norm-full-GB and the free device memory, else bound)")
    g.add_argument("--cut-norm-full-GB", dest="cut_norm_full_GB", type=float, default=24.0, help="(a') the device memory allowed to the full trace norm (GiB)")
    g.add_argument("--cut-bound-tol", dest="cut_bound_tol", type=float, default=1e-6, help="(a') the rounding allowance of the check (nats): |S_X(C) - S_X(P_-)| <= t_X + tol")
    g.add_argument("--cut-sigma-tol", dest="cut_sigma_tol", type=float, default=1e-10, help="(a') singular values of (1 - C) V above this count in rank(C - P_-)")
'''

# ------------------------------------------------------------------------------------------------------------ 4. the configuration
CFG_OLD = '            "columns_kappas": [int(x) for x in str(g("columns_kappas", "")).split(",") if str(x).strip()]}'
CFG_NEW = r'''            "columns_kappas": [int(x) for x in str(g("columns_kappas", "")).split(",") if str(x).strip()],
            "cut_norm": g("cut_norm", "on"), "cut_block_route": g("cut_block_route", "svd"), "cut_norm_full": g("cut_norm_full", "auto"),       # v1b (a')
            "cut_norm_full_GB": float(g("cut_norm_full_GB", 24.0)), "cut_bound_tol": float(g("cut_bound_tol", 1e-6)), "cut_sigma_tol": float(g("cut_sigma_tol", 1e-10))}'''

# ------------------------------------------------------------------------------------------------------------ 5. the new section
EXT_ANCHOR = "def exterior_entropies(Phi, R2, cuts, Vneg=None):\n"

NEW_SECTION = r'''# ----------------------------------------------------------- (a') the cross-cut block trace norms (hreduce's EVOLVEDCUT EC1; v1b)
EPS64 = 2.220446049250313e-16


def _h2(x):
    """the binary entropy in nats, elementwise; 0 at 0 and at 1."""
    x = torch.as_tensor(x, dtype=RD).clamp(0.0, 1.0)
    y = torch.zeros_like(x)
    m = (x > 0.0) & (x < 1.0)
    xm = x[m]
    y[m] = -(xm * torch.log(xm) + (1.0 - xm) * torch.log1p(-xm))
    return y


def pure_cut_F(s):
    """F(s) = h_2((1 - sqrt(1 - 4 s^2))/2) on 0 <= s <= 1/2: a projector Z has S_X(Z) = sum_j F(sigma_j(K_Z)) with K_Z = Pi_X Z Pi_Xc, since
    Z_X (1 - Z_X) = K_Z K_Z^dagger (the paired single-object identity behind EC1; 0 <= F' <= 2 is the bound's constant)."""
    s = torch.as_tensor(s, dtype=RD)
    return _h2(0.5 * (1.0 - torch.sqrt((1.0 - 4.0 * s * s).clamp_min(0.0))))


def cut_index(R2, x):
    """the modes beyond the cut (R2 > x): the slice start i0 when the rows are ordered by R (R2 non-decreasing, the chain's flattened index
    2 n + a), else None; the index tensors (exterior, interior); the counts (n_X, n_Xc)."""
    mask = R2 > x
    n2 = int(R2.numel()); nX = int(mask.sum()); i0 = n2 - nX
    ordered = bool(mask[i0:].all()) and not bool(mask[:i0].any())
    return (i0 if ordered else None), torch.nonzero(mask).flatten(), torch.nonzero(~mask).flatten(), nX, n2 - nX


def _rows(M, i0, idx, exterior):
    """the exterior (or interior) rows of the columns M: a view by slicing when the rows are ordered, else index_select (a copy)."""
    if i0 is not None:
        return M[i0:] if exterior else M[:i0]
    return M.index_select(0, idx)


def cross_block(Phi, Vneg, i0, idx_X, idx_C):
    """K = Pi_X (C - P_-) Pi_Xc = Phi_X Phi_Xc^dagger - V_X V_Xc^dagger as the n_X x n_Xc matrix from the columns: two GEMMs on the row
    slices (views when the rows are ordered), the second accumulated in place (addmm_); no n2 x n2 or N x N object. Returns (K,
    ||Phi_X Phi_Xc^dagger||_F^2), the second for the check against (a)'s spectrum: ||Pi_X C Pi_Xc||_F^2 = sum_j lambda_j (1 - lambda_j) over
    the eigenvalues of C_Xc."""
    K = _rows(Phi, i0, idx_X, True) @ _rows(Phi, i0, idx_C, False).mH
    fro_C = float(torch.linalg.norm(K)) ** 2
    if Vneg is not None and Vneg.shape[1] and min(K.shape):
        K.addmm_(_rows(Vneg, i0, idx_X, True), _rows(Vneg, i0, idx_C, False).mH, alpha=-1.0)
    return K, fro_C


def block_singular_values(K, route="svd"):
    """the block's singular values, descending: svd (torch.linalg.svdvals, exact to rounding: a QR of the block and the bidiagonal SVD of its
    small side) or gram (eigvalsh of the smaller Gram and a square root: the small singular values to sqrt accuracy, about 1e-8). Returns
    (s, the route's label, an indicative rounding bound on 2 sum_j s_j)."""
    m = int(min(K.shape))
    if m == 0:
        return torch.zeros(0, dtype=RD, device=K.device), "empty block", 0.0
    if route == "gram":
        G = K.mH @ K if K.shape[0] >= K.shape[1] else K @ K.mH
        lam = torch.linalg.eigvalsh(G); del G
        s = torch.sqrt(lam.clamp_min(0.0)).flip(0)
        return s, "gram %d x %d (eigvalsh of the smaller Gram; the small singular values to sqrt accuracy)" % (m, m), 2.0 * m * math.sqrt(2.0 * EPS64 * m * max(float(lam[-1]), 0.0))
    s = torch.linalg.svdvals(K)
    return s, "svd (torch.linalg.svdvals of the %d x %d block)" % (K.shape[0], K.shape[1]), 2.0 * m * 8.0 * EPS64 * float(s[0])


def cross_cut_norms(Phi, Vneg, R2, cuts, route="svd"):
    """(a') per cut (label, R_abs): the singular values sigma_j of the X-by-Xc block of C - P_- (X the modes beyond the cut) and t_X =
    2 sum_j sigma_j = 2 ||Pi_X (C - P_-) Pi_Xc||_1, hreduce's exact bound |S_X(C) - S_X(P_-)| <= t_X <= ||C - P_-||_1 (EVOLVEDCUT EC1), by
    the block route from the columns: the n_X x n_Xc block by two GEMMs on the row slices (cross_block), its singular values by svdvals (or
    the Gram). Any two orthonormal column sets serve as (Phi, Vneg): the same call gives 2 ||Pi_X (A A^dagger - B B^dagger) Pi_Xc||_1.
    Cost at n = 16384 (n2 = 32768, N_occ = N_neg = 16384, r_s = 64) on an H100: the block at the 48 r_s cut is 26624 x 6144 (2.6 GiB); its
    two GEMMs 2 x 26624 x 6144 x 16384 complex multiply-adds (4e13 flops, about a second at the fp64 tensor-core rate); its singular values
    a QR of the block (2e12 complex multiply-adds) and the 6144 x 6144 bidiagonal SVD (6e11), seconds to tens of seconds in cusolver's gesvd
    (the bidiagonal QR iteration is sequential); the four cuts (n_Xc 2048, 3072, 3840, 6144; n_X 30720 down to 26624) together about 1e14
    flops of GEMMs, under a minute, and under 8 GiB transient (the block and svdvals' workspace) per baseline clock. No n2 x n2 (16 GiB) or
    N_occ x N_occ (4 GiB) object is formed; the measured seconds are stored per cut (cross_seconds)."""
    out = {}
    for lab, x in cuts:
        t0 = time.time()
        i0, idx_X, idx_C, nX, nC = cut_index(R2, x)
        K, fro_C = cross_block(Phi, Vneg, i0, idx_X, idx_C)
        fro_D = float(torch.linalg.norm(K)) ** 2
        s, rlab, rb = block_singular_values(K, route)
        del K
        sv = s.cpu().numpy().astype(float); del s
        out[lab] = {"R_abs": float(x), "n_X": nX, "n_Xc": nC, "block_route": rlab, "rows_ordered": i0 is not None, "sv_block": sv,
                    "t_X": 2.0 * float(sv.sum()), "n_sv": int(sv.size), "sigma_max_block": float(sv[0]) if sv.size else 0.0,
                    "n_sv_above_1e-8": int((sv > 1e-8).sum()), "frob2_block": fro_D, "frob2_sv_minus_block": float((sv * sv).sum()) - fro_D,
                    "frob2_block_C": fro_C, "rounding_bound_t_X": rb, "cross_seconds": time.time() - t0,
                    "label": "t_X = 2 ||Pi_X (C - P_-) Pi_Xc||_1 = 2 x the sum of the singular values of the X-by-Xc block of C - P_- (EVOLVEDCUT EC1: |S_X(C) - S_X(P_-)| <= t_X <= ||C - P_-||_1 for a pure Gaussian C and a projector P_-)"}
    if Phi.device.type == "cuda":
        torch.cuda.empty_cache()
    return out


def cross_cut_check(ent, cc, tol=1e-6):
    """the check |S_X(C) - S_X(P_-)| <= t_X per cut (tol in nats for the entropies' rounding), the consistency of the block with (a)'s spectra
    (||Pi_X C Pi_Xc||_F^2 = sum lambda (1 - lambda) over the eigenvalues of C_Xc; the triangle 2 | ||K_C||_1 - ||K_P||_1 | <= t_X <=
    2 (||K_C||_1 + ||K_P||_1) with ||K_Z||_1 = sum sqrt(lambda (1 - lambda)) from the spectra); the results into ent's cut dicts (t_X, abs_dS,
    slack, bound_ok, ...); returns the labels that FAIL."""
    fails = []
    for lab, c in cc.items():
        d = ent.get(lab)
        if d is None:
            continue
        d.update({k_: v_ for k_, v_ in c.items() if k_ != "sv_block"})
        d["bound_tol"] = float(tol)
        for key, nm in (("eig_Xc", "C"), ("eig_Xc_vac", "P")):
            ev = d.get(key)
            if isinstance(ev, np.ndarray) and ev.size:
                lam = np.clip(ev.astype(float), 0.0, 1.0)
                d["K_%s_trace_norm_from_spectrum" % nm] = float(np.sum(np.sqrt(lam * (1.0 - lam))))
                d["K_%s_frob2_from_spectrum" % nm] = float(np.sum(lam * (1.0 - lam)))
        if d.get("K_C_frob2_from_spectrum") is not None:
            d["frob2_C_block_minus_spectrum"] = c["frob2_block_C"] - d["K_C_frob2_from_spectrum"]
        a_, b_ = d.get("K_C_trace_norm_from_spectrum"), d.get("K_P_trace_norm_from_spectrum")
        if a_ is not None and b_ is not None:
            d["t_X_triangle_lower"] = 2.0 * abs(a_ - b_); d["t_X_triangle_upper"] = 2.0 * (a_ + b_)
            d["t_X_triangle_ok"] = bool(d["t_X_triangle_lower"] - 1e-6 <= c["t_X"] <= d["t_X_triangle_upper"] + 1e-6)
        SX, SXv = d.get("S_X"), d.get("S_X_vac")
        if SX is None or SXv is None:
            d.update({"abs_dS": None, "slack": None, "bound_ok": None})
            continue
        d["abs_dS"] = abs(SX - SXv); d["slack"] = c["t_X"] - d["abs_dS"]; d["bound_ok"] = bool(d["slack"] >= -float(tol))
        if not d["bound_ok"]:
            fails.append(lab)
    return fails


def premise_defects(Phi, Vneg, chunk=1024):
    """||Phi^dagger Phi - 1||_F and ||V^dagger V - 1||_F by column blocks (no N x N object): EC1's premises are C = Phi Phi^dagger a projector
    (the pure Gaussian state) and P_- = V V^dagger a projector."""
    out = {}
    for name, X in (("Phi", Phi), ("Vneg", Vneg)):
        if X is None:
            out[name] = None
            continue
        acc = 0.0; n = int(X.shape[1]); ch = max(1, int(chunk))
        for j0 in range(0, n, ch):
            Gb = X.mH @ X[:, j0: j0 + ch]
            jj = torch.arange(j0, min(n, j0 + ch), device=X.device)
            Gb[jj, jj - j0] -= 1.0
            acc += float(torch.linalg.norm(Gb)) ** 2; del Gb
        out[name] = math.sqrt(acc)
    return out


def sigma_max_estimate(Phi, Vneg, iters=60, block=8, seed=0):
    """the largest singular value of C - P_- by a subspace iteration on (C - P_-)^2 from the columns (block vectors, Rayleigh-Ritz; the Ritz
    value is a lower estimate; the certified cap for a projector difference is 1); returns (estimate, iterations)."""
    n2 = int(Phi.shape[0]); dev = Phi.device
    gen = torch.Generator().manual_seed(int(seed))
    X = torch.complex(torch.randn(n2, block, generator=gen, dtype=torch.float64), torch.randn(n2, block, generator=gen, dtype=torch.float64)).to(device=dev, dtype=CD)
    X, _ = torch.linalg.qr(X)

    def D_apply(Y):
        Z = Phi @ (Phi.mH @ Y)
        if Vneg is not None and Vneg.shape[1]:
            Z = Z - Vneg @ (Vneg.mH @ Y)
        return Z
    val = 0.0; it = 0
    for it in range(1, int(iters) + 1):
        Y = D_apply(D_apply(X))
        T = X.mH @ Y
        new = math.sqrt(max(float(torch.linalg.eigvalsh(0.5 * (T + T.mH)).max()), 0.0))
        X, _ = torch.linalg.qr(Y)
        if abs(new - val) <= 1e-12 * max(1.0, new):
            val = new
            break
        val = new
    return val, it


def full_trace_norm(Phi, Vneg, mode="auto", budget_GB=24.0, sigma_tol=1e-10, chunk=1024, power_iters=60):
    """nu = ||C - P_-||_1 for the projectors C = Phi Phi^dagger (N_occ columns) and P_- = V V^dagger (N_neg): with the principal angles theta_j
    between ran C and ran P_- (cos theta_j the singular values of M = Phi^dagger V) the eigenvalues of C - P_- are +-sin theta_j and
    |N_occ - N_neg| of modulus 1, so nu = 2 sum_j sigma_j + (N_occ - N_neg) over the N_neg singular values sigma_j of (1 - C) V (sin theta_j
    and, for N_neg > N_occ, ones); likewise rank(C - P_-) = 2 #{sigma_j > tol} + (N_occ - N_neg) and ||C - P_-||_2^2 = 2 sum sigma_j^2 +
    (N_occ - N_neg) = N_occ + N_neg - 2 Tr[C P_-]; n_pair = Tr[(1 - P_-) C] is EC2's comparator, not a bound. Routes: gram (eigvalsh of
    1 - M^dagger M, N_neg x N_neg: sigma_j^2 to rounding, the small sigma_j to sqrt accuracy, the rounding bound beside), svd
    (torch.linalg.svdvals of W = V - Phi M, n2 x N_neg: exact to rounding), bound (no N x N object: Tr[C P_-] by column blocks, hence
    ||C - P_-||_2 exactly, and the certified bounds nu <= rank(C - P_-) sigma_max <= (N_occ + N_neg) x 1 and nu <= sqrt(N_occ + N_neg) ||C - P_-||_2,
    with sigma_max's subspace-iteration value beside as a lower estimate), auto (gram where its memory estimate fits the budget and the
    device's free memory, else bound). Cost at n = 16384 on an H100: gram forms M (16384 x 16384, 4 GiB, 9e12 complex multiply-adds) and
    M^dagger M (4e12), one eigvalsh of 16384 x 16384 (tens of seconds), about 16 GiB peak; svd forms W (32768 x 16384, 8 GiB) and runs
    cusolver's gesvd on it (minutes: the bidiagonal QR iteration of a 16384-column matrix) with its workspace beside; bound is one pass of
    column-block GEMMs (9e12) and a few GEMV sweeps. The singular values are returned under "sv" (the caller stores them)."""
    t0 = time.time()
    p = int(Phi.shape[1]); q = int(Vneg.shape[1]) if Vneg is not None else 0; n2 = int(Phi.shape[0]); dev = Phi.device
    Gi = 2.0 ** 30
    est = {"gram_GiB": 16.0 * (p * q + 3.0 * q * q) / Gi, "svd_GiB": 16.0 * (p * q + 3.0 * n2 * q) / Gi, "bound_GiB": 16.0 * p * min(q, int(chunk)) / Gi}
    free = None
    if dev.type == "cuda":
        try:
            free = torch.cuda.mem_get_info(dev)[0] / Gi
        except Exception:                                                       # noqa: BLE001
            free = None
    route = mode
    if mode == "auto":
        route = "gram" if (est["gram_GiB"] <= float(budget_GB) and (free is None or est["gram_GiB"] <= 0.85 * free)) else "bound"
    out = {"N_occ": p, "N_neg": q, "rank_difference": p - q, "mode_requested": mode, "route": route, "estimate_GiB": est, "device_free_GiB_before": free,
           "budget_GiB": float(budget_GB), "sigma_tol": float(sigma_tol), "nu": None, "rank": None, "sigma_max": None}
    if route == "off":
        out["label"] = "the full ||C - P_-||_1 not computed (--cut-norm-full off)"; out["seconds"] = time.time() - t0
        return out
    if route in ("gram", "svd") and q == 0:
        route = "bound"; out["route"] = route; out["note"] = "no vacuum columns: the bound route"
    if route in ("gram", "svd"):
        M = Phi.mH @ Vneg
        trCP = float(torch.linalg.norm(M)) ** 2
        if route == "gram":
            G = M.mH @ M; del M
            G.neg_(); G.diagonal().add_(1.0)
            lam = torch.linalg.eigvalsh(G); del G
            s = torch.sqrt(lam.clamp(0.0, 1.0)).flip(0); del lam
        else:
            W = Vneg.clone(); W.addmm_(Phi, M, alpha=-1.0); del M
            s = torch.linalg.svdvals(W); del W
        sv = s.cpu().numpy().astype(float); del s
        out["sv"] = sv
        out.update({"nu": 2.0 * float(sv.sum()) + (p - q), "rank": int(2 * int((sv > float(sigma_tol)).sum()) + (p - q)),
                    "sigma_max": max(float(sv[0]) if sv.size else 0.0, 1.0 if p != q else 0.0), "frob2_from_sv": 2.0 * float((sv * sv).sum()) + (p - q),
                    "n_sigma_below_tol": int((sv <= float(sigma_tol)).sum()), "sigma_min": float(sv[-1]) if sv.size else None,
                    "rounding_bound_nu": (2.0 * q * math.sqrt(2.0 * EPS64 * q)) if route == "gram" else 2.0 * q * 8.0 * EPS64})
    else:
        trCP = 0.0; ch = max(1, int(chunk))
        for j0 in range(0, q, ch):
            Mb = Phi.mH @ Vneg[:, j0: j0 + ch]
            trCP += float(torch.linalg.norm(Mb)) ** 2; del Mb
        out["sigma_max_estimate"], out["power_iterations"] = sigma_max_estimate(Phi, Vneg, power_iters)
    frob2 = p + q - 2.0 * trCP
    out.update({"tr_CP": trCP, "frob2": frob2, "frob": math.sqrt(max(frob2, 0.0)), "n_pair": p - trCP, "n_hole": q - trCP, "rank_bound": p + q, "sigma_max_cap": 1.0,
                "bound_rank_sigma": float(p + q), "bound_cauchy_schwarz": math.sqrt(p + q) * math.sqrt(max(frob2, 0.0))})
    if out["nu"] is not None:
        out["frob2_sv_minus_direct"] = out["frob2_from_sv"] - frob2
        out["bound_rank_sigma_exact"] = out["rank"] * out["sigma_max"]
        out["bound_cauchy_schwarz_exact_rank"] = math.sqrt(max(out["rank"], 0)) * out["frob"]
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    out["seconds"] = time.time() - t0
    out["label"] = ("nu = ||C - P_-||_1 = 2 sum_j sigma_j((1 - C) V) + (N_occ - N_neg) (the principal angles between ran C and ran P_-; the eigenvalues of C - P_- are +-sin theta_j "
                    "and |N_occ - N_neg| of modulus 1); rank(C - P_-) = 2 #{sigma_j > tol} + (N_occ - N_neg); ||C - P_-||_2^2 = N_occ + N_neg - 2 Tr[C P_-]; n_pair = Tr[(1 - P_-) C] is EC2's "
                    "comparator, not a bound; route %s (gram: the small sigma_j to sqrt accuracy, rounding bound beside; svd: exact to rounding; bound: rank <= N_occ + N_neg and "
                    "sigma_max <= 1 certify nu <= bound_rank_sigma, Cauchy-Schwarz certifies nu <= bound_cauchy_schwarz, sigma_max_estimate is a lower estimate)" % route)
    return out


def cross_cut_readout(Phi, Vneg, R2, cuts, ent, cfg, kappa, trs, sc, arrays):
    """(a') at a clock with the vacuum: the block trace norms and the check per cut (printed; a FAIL line carries the premises' defects), the
    full ||C - P_-||_1 or its bounds; the scalars into ent's cut dicts and sc["trace_norm"], the singular values into arrays
    (sv_block_cut<label>, sv_full)."""
    cc = cross_cut_norms(Phi, Vneg, R2, cuts, cfg.get("cut_block_route", "svd"))
    fails = cross_cut_check(ent, cc, float(cfg.get("cut_bound_tol", 1e-6)))
    for lab, c in cc.items():
        arrays["sv_block_cut%s" % lab] = c.pop("sv_block")
    prem = premise_defects(Phi, Vneg, int(cfg.get("trace_chunk", 1024))) if fails else None
    for lab, _ in cuts:
        d = ent.get(lab) or {}
        if d.get("t_X") is None:
            continue
        head = ("D13 kappa %d T/r_s %.3f cross-cut %s r_s: |S_X(C) - S_X(P_-)| %s <= t_X = 2||Pi_X (C - P_-) Pi_Xc||_1 %.6e (slack %s; %d singular values, %d above 1e-8, "
                "largest %.4e; block %d x %d, %s, rows %s; %.1f s)" % (kappa, trs, lab, _f(d.get("abs_dS"), "%.6e"), d["t_X"], _f(d.get("slack"), "%.3e"), d["n_sv"], d["n_sv_above_1e-8"],
                                                                     d["sigma_max_block"], d["n_X"], d["n_Xc"], d["block_route"], "ordered" if d["rows_ordered"] else "indexed", d["cross_seconds"]))
        if d.get("bound_ok") is None:
            log(head + ": no vacuum entropy at this cut")
        elif d["bound_ok"]:
            log(head + ": ok" + ("" if d.get("t_X_triangle_ok", True) else "; WARNING: outside the triangle bounds from (a)'s spectra [%.6e, %.6e]" % (d["t_X_triangle_lower"], d["t_X_triangle_upper"])))
        else:
            log(head + ": FAIL (|S_X(C) - S_X(P_-)| <= 2||Pi_X (C - P_-) Pi_Xc||_1 holds for a pure Gaussian state C and a projector P_-: the premises' defects ||Phi^dagger Phi - 1||_F %s, "
                       "||V^dagger V - 1||_F %s; the purity gap S_X - S_Xc %.2e; the block's Frobenius check against (a)'s spectrum %s)" % (
                           _f(prem.get("Phi"), "%.2e"), _f(prem.get("Vneg"), "%.2e"), d.get("purity_gap", 0.0), _f(d.get("frob2_C_block_minus_spectrum"), "%.2e")))
    tn = full_trace_norm(Phi, Vneg, cfg.get("cut_norm_full", "auto"), float(cfg.get("cut_norm_full_GB", 24.0)), float(cfg.get("cut_sigma_tol", 1e-10)), int(cfg.get("trace_chunk", 1024)))
    if "sv" in tn:
        arrays["sv_full"] = tn.pop("sv")
    tmax = max([c["t_X"] for c in cc.values()], default=None)
    tn["max_t_X"] = tmax
    tn["nu_ge_max_t_X"] = bool(tn["nu"] + 1e-8 >= tmax) if (tn.get("nu") is not None and tmax is not None) else None
    sc["trace_norm"] = tn; sc["cross_cut_fails"] = fails; sc["premise_defects"] = prem
    if tn.get("route") == "off":
        log("D13 kappa %d T/r_s %.3f: the full ||C - P_-||_1 not computed (--cut-norm-full off); max t_X over the cuts %s" % (kappa, trs, _f(tmax, "%.6e")))
    elif tn.get("nu") is not None:
        log("D13 kappa %d T/r_s %.3f: ||C - P_-||_1 = %.6e (%s route%s; N_occ %d, N_neg %d; rank(C - P_-) %d, largest singular value %.6e, ||C - P_-||_2 %.6e, n_pair = Tr[(1 - P_-) C] %.6e; "
            "max t_X over the cuts %s: %s; the Frobenius check %.2e; %.1f s)" % (kappa, trs, tn["nu"], tn["route"], (", rounding bound %.1e" % tn["rounding_bound_nu"]) if tn.get("rounding_bound_nu") else "",
                                                                                   tn["N_occ"], tn["N_neg"], tn["rank"], tn["sigma_max"], tn["frob"], tn["n_pair"], _f(tmax, "%.6e"),
                                                                                   "consistent (2||Pi_X D Pi_Xc||_1 <= ||D||_1)" if tn["nu_ge_max_t_X"] is not False else "WARNING: below a t_X",
                                                                                   tn.get("frob2_sv_minus_direct", 0.0), tn["seconds"]))
    else:
        log("D13 kappa %d T/r_s %.3f: ||C - P_-||_1 by its bounds (%s route, requested %s): <= rank x sigma_max <= %d (rank(C - P_-) <= N_occ + N_neg = %d, sigma_max <= 1 for a projector difference; the subspace "
            "iteration's sigma_max %s after %s steps, a lower estimate), <= sqrt(N_occ + N_neg) ||C - P_-||_2 = %.6e (Cauchy-Schwarz); ||C - P_-||_2 %.6e, n_pair = Tr[(1 - P_-) C] %.6e (EC2's comparator, not a bound); "
            "max t_X over the cuts %s; the gram route's estimate %.1f GiB against the budget %.1f GiB and the free device memory %s GiB; %.1f s" % (
                kappa, trs, tn["route"], tn["mode_requested"], tn["rank_bound"], tn["rank_bound"], _f(tn.get("sigma_max_estimate"), "%.6e"), _f(tn.get("power_iterations"), "%d"), tn["bound_cauchy_schwarz"], tn["frob"], tn["n_pair"],
                _f(tmax, "%.6e"), tn["estimate_GiB"]["gram_GiB"], tn["budget_GiB"], _f(tn.get("device_free_GiB_before"), "%.1f"), tn["seconds"]))
    return cc, tn
'''

# ------------------------------------------------------------------------------------------------------------ 6. the checkpoint
CKPT_OLD = '        ent = exterior_entropies(Phi, self.R2, self.cuts, Vneg)\n        sc["cuts"] = {}\n'
CKPT_NEW = ('        ent = exterior_entropies(Phi, self.R2, self.cuts, Vneg)\n'
            '        if Vneg is not None and self.cfg.get("cut_norm", "on") == "on":                                 # v1b: the cross-cut block trace norms (EVOLVEDCUT EC1)\n'
            '            cross_cut_readout(Phi, Vneg, self.R2, self.cuts, ent, self.cfg, self.kappa, trs, sc, arrays)\n'
            '        sc["cuts"] = {}\n')

# ------------------------------------------------------------------------------------------------------------ 7. format_line
FMT_SX_OLD = '    sx = ", ".join("%s r_s %.6f%s" % (lab, d["S_X"], (" (vacuum %.6f, excess %.6f)" % (d["S_X_vac"], d["excess"])) if d.get("S_X_vac") is not None else "") for lab, d in cuts.items())\n'
FMT_SX_NEW = r'''    sx = ", ".join("%s r_s %.6f%s%s" % (lab, d["S_X"], (" (vacuum %.6f, excess %.6f)" % (d["S_X_vac"], d["excess"])) if d.get("S_X_vac") is not None else "",
                                        (", t_X %.6f %s" % (d["t_X"], "ok" if d.get("bound_ok") else ("FAIL" if d.get("bound_ok") is False else "-"))) if d.get("t_X") is not None else "") for lab, d in cuts.items())
    tn = sc.get("trace_norm") or {}                                                                             # v1b (a')
    if not tn:
        tns = "-"
    elif tn.get("nu") is not None:
        tns = "%.6e (%s route; rank %d, largest singular value %.4e, ||.||_2 %.4e)" % (tn["nu"], tn.get("route"), int(tn.get("rank") or 0), float(tn.get("sigma_max") or 0.0), float(tn.get("frob") or 0.0))
    else:
        tns = "<= %s (rank x sigma_max; rank <= N_occ + N_neg, sigma_max <= 1), <= %s (Cauchy-Schwarz); ||.||_2 %s, sigma_max estimate %s" % (
            _f(tn.get("bound_rank_sigma"), "%.4e"), _f(tn.get("bound_cauchy_schwarz"), "%.4e"), _f(tn.get("frob"), "%.4e"), _f(tn.get("sigma_max_estimate"), "%.4e"))
'''
FMT_RET_OLD = r'''    return ("D13 kappa %d T/r_s %.3f: E_inv %s; S_X [%s] nats (complement gap max %.1e); energy beyond the cut [%s]; shells %s; collar window %s; source modes %s; "
            "first return (r_s) [%s]; enclosed source %s; %.0f s" % (kappa, sc["t_over_rs"], _f(E_inv, "%.6e"), sx, gap, iv, shs, css, srcs, frs, ens, sc.get("seconds", 0.0)))
'''
FMT_RET_NEW = r'''    return ("D13 kappa %d T/r_s %.3f: E_inv %s; S_X [%s] nats (complement gap max %.1e); ||C - P_-||_1 %s; energy beyond the cut [%s]; shells %s; collar window %s; source modes %s; "
            "first return (r_s) [%s]; enclosed source %s; %.0f s" % (kappa, sc["t_over_rs"], _f(E_inv, "%.6e"), sx, gap, tns, iv, shs, css, srcs, frs, ens, sc.get("seconds", 0.0)))
'''

# ------------------------------------------------------------------------------------------------------------ 9-11. the CLI's weighted sums
W_KEYS_OLD = '                for key in ("S_X", "excess"):\n'
W_KEYS_NEW = '                for key in ("S_X", "excess", "t_X", "abs_dS"):\n'
W_EINV_OLD = '            weighted["E_inv"] = weighted.get("E_inv", 0.0) + 16.0 * kappa * E_inv\n'
W_EINV_NEW = W_EINV_OLD + r'''            tn_ = sc.get("trace_norm") or {}                                                                      # v1b (a'): the weighted trace norm and its bounds
            for key in ("nu", "bound_rank_sigma", "bound_cauchy_schwarz", "frob"):
                if tn_.get(key) is not None:
                    weighted.setdefault("%s_trace" % key, 0.0); weighted["%s_trace" % key] += 16.0 * kappa * tn_[key]
'''
W_LOG_OLD = r'''            log("T/r_s %.4f weighted over the state's channels %s (16 kappa): %s" % (trs, chans, ", ".join("%s %.6e" % (k_, v_) for k_, v_ in sorted(weighted.items()))))
'''
W_LOG_NEW = W_LOG_OLD + r'''            for lab in sorted({k_[len("t_X_cut"):] for k_ in weighted if k_.startswith("t_X_cut")}, key=float):             # v1b (a'): the weighted bound check
                ex_, ab_, tx_ = weighted.get("excess_cut" + lab), weighted.get("abs_dS_cut" + lab), weighted["t_X_cut" + lab]
                if ex_ is not None:
                    tol_w = cfg["cut_bound_tol"] * 16.0 * sum(int(c_) for c_ in chans)
                    log("T/r_s %.4f weighted cross-cut bound at %s r_s: |sum_kappa w excess| %.6e <= sum_kappa w |S_X(C) - S_X(P_-)| %s <= sum_kappa w t_X %.6e: %s" % (
                        trs, lab, abs(ex_), _f(ab_, "%.6e"), tx_, "ok" if abs(ex_) <= tx_ + tol_w else "FAIL"))
'''

# ------------------------------------------------------------------------------------------------------------ 12-14. the ksum table
K_KEYS_OLD = '                for key in ("S_X", "excess", "S_X_vac"):\n'
K_KEYS_NEW = '                for key in ("S_X", "excess", "S_X_vac", "t_X", "abs_dS"):\n'
K_IV_OLD = '            I_vals = [((d13_of(k_, t).get("enclosed") or {}).get("I_L")) for k_ in sub]\n'
K_IV_NEW = r'''            for key in ("nu", "bound_rank_sigma", "bound_cauchy_schwarz", "frob"):                                              # v1b (a')
                vals = [((d13_of(k_, t).get("trace_norm") or {}).get(key)) for k_ in sub]
                e["%s_trace_weighted" % key] = sum(per[k_]["w"] * v_ for k_, v_ in zip(sub, vals)) if all(v_ is not None for v_ in vals) else None
            for lab in cut_labels:
                ex_, tx_ = e.get("excess_cut%s" % lab), e.get("t_X_cut%s" % lab)
                e["bound_ok_cut%s" % lab] = bool(abs(ex_) <= tx_ + 1e-6 * sum(per[k_]["w"] for k_ in sub)) if (ex_ is not None and tx_ is not None) else None
''' + K_IV_OLD
K_COLS_OLD = '        cols = ["E_inv_weighted", "L_interval", "L_over_LH_kappaT", "collar_E_flux_out_soft", "collar_E_flux_out_hard", "shells_hard_E_out", "shells_above_N_total"] + ["excess_cut%s" % lab for lab in cut_labels] + ["S_X_cut%s" % lab for lab in cut_labels]\n'
K_COLS_NEW = '        cols = ["E_inv_weighted", "L_interval", "L_over_LH_kappaT", "collar_E_flux_out_soft", "collar_E_flux_out_hard", "shells_hard_E_out", "shells_above_N_total"] + ["excess_cut%s" % lab for lab in cut_labels] + ["S_X_cut%s" % lab for lab in cut_labels] + ["t_X_cut%s" % lab for lab in cut_labels] + ["nu_trace_weighted"]\n'

# ------------------------------------------------------------------------------------------------------------ 15. compare_runs
CMP_OLD = '                    e["S_X_cut%s_diff" % lab] = ca[lab]["S_X"] - cb[lab]["S_X"]\n'
CMP_NEW = CMP_OLD + ('                    if ca[lab].get("t_X") is not None and cb[lab].get("t_X") is not None:                          # v1b\n'
                     '                        e["t_X_cut%s_diff" % lab] = ca[lab]["t_X"] - cb[lab]["t_X"]\n')

# ------------------------------------------------------------------------------------------------------------ 16-18. the memory figure
MEM_SIG_OLD = 'def memory_figure(nsites=8192, n_occ=None, chunk=2048, R_cut_rs=24.0, rs=64.0, source_max=128):\n'
MEM_SIG_NEW = 'def memory_figure(nsites=8192, n_occ=None, chunk=2048, R_cut_rs=24.0, rs=64.0, source_max=128, cut_max_rs=48.0):\n'
MEM_DICT_OLD = '         "exterior_projection_transient_GiB": G(16 * nX * n0), "source_modes_GiB": G(16 * n2 * 3 * source_max), "references_transient_GiB": 2 * G(cols)}\n'
MEM_DICT_NEW = MEM_DICT_OLD + r'''    nXc = 2 * min(nsites, int(cut_max_rs * rs)); blk = 16 * (n2 - nXc) * nXc                                                  # v1b (a'): the largest cross-cut block
    d.update({"cross_cut_block_GiB": G(blk), "cross_cut_svd_peak_GiB": 3 * G(blk), "trace_norm_gram_peak_GiB": 4 * G(gram),
              "trace_norm_svd_peak_GiB": G(cols) + G(gram) + 2 * G(cols)})
    d["cross_cut_checkpoint_peak_GiB"] = d["columns_GiB"] + d["Vneg_GiB"] + 2 * d["shells_U_GiB"] + max(d["cross_cut_svd_peak_GiB"], d["trace_norm_gram_peak_GiB"])
'''
MEM_KEYS_OLD = '            "shells_rebuild_peak_GiB", "probe_clock_peak_GiB", "pair_route_vacuum_peak_GiB")\n'
MEM_KEYS_NEW = '            "shells_rebuild_peak_GiB", "probe_clock_peak_GiB", "pair_route_vacuum_peak_GiB", "cross_cut_block_GiB", "cross_cut_svd_peak_GiB", "trace_norm_gram_peak_GiB", "cross_cut_checkpoint_peak_GiB")\n'

# ------------------------------------------------------------------------------------------------------------ 2, 19. the names
CODE_OLD = 'CODE = "page_d13_readout_v1.py"'
CODE_NEW = 'CODE = "page_d13_readout_v1b.py"'
LABEL_OLD = 'label="D13 readouts (page_d13_readout_v1) on saved states of %s; no collector, no selection, no archive" % (H.label or a.record))'
LABEL_NEW = 'label="D13 readouts (page_d13_readout_v1b: with the cross-cut block trace norms of EVOLVEDCUT EC1) on saved states of %s; no collector, no selection, no archive" % (H.label or a.record))'

PAIRS = [
    (DOC_OLD, DOC_NEW),                                                        # 1. the docstring head
    (CODE_OLD, CODE_NEW),                                                      # 2. the code name
    (ARGS_OLD, ARGS_NEW),                                                      # 3. the arguments
    (CFG_OLD, CFG_NEW),                                                        # 4. the configuration
    (EXT_ANCHOR, NEW_SECTION.rstrip("\n") + "\n\n\n" + EXT_ANCHOR),            # 5. the new section before (a)'s exterior_entropies
    (CKPT_OLD, CKPT_NEW),                                                      # 6. the checkpoint's call
    (FMT_SX_OLD, FMT_SX_NEW),                                                  # 7. format_line: t_X per cut, the trace norm
    (FMT_RET_OLD, FMT_RET_NEW),                                                # 8. format_line: the line
    (W_KEYS_OLD, W_KEYS_NEW),                                                  # 9. the CLI's weighted cut keys
    (W_EINV_OLD, W_EINV_NEW),                                                  # 10. the CLI's weighted trace norm
    (W_LOG_OLD, W_LOG_NEW),                                                    # 11. the CLI's weighted bound check
    (K_KEYS_OLD, K_KEYS_NEW),                                                  # 12. ksum: the cut keys
    (K_IV_OLD, K_IV_NEW),                                                      # 13. ksum: the trace norm and the bound per cut
    (K_COLS_OLD, K_COLS_NEW),                                                  # 14. ksum: the md columns
    (CMP_OLD, CMP_NEW),                                                        # 15. compare_runs
    (MEM_SIG_OLD, MEM_SIG_NEW),                                                # 16. memory_figure's signature
    (MEM_DICT_OLD, MEM_DICT_NEW),                                              # 17. memory_figure's cross-cut transients
    (MEM_KEYS_OLD, MEM_KEYS_NEW),                                              # 18. format_memory_figure's keys
    (LABEL_OLD, LABEL_NEW),                                                    # 19. the store's label
]


def build(src, dst):
    text = open(src, encoding="utf-8").read()
    print("source %s: %d bytes, md5 %s" % (src, len(text.encode("utf-8")), hashlib.md5(text.encode("utf-8")).hexdigest()))
    for i, (old, new) in enumerate(PAIRS, 1):
        n = text.count(old)
        if n != 1:
            sys.exit("ABORT: replacement %d matches %d times (must be exactly 1): %r" % (i, n, old[:120]))
        text = text.replace(old, new)
        print("  %2d. applied (%d -> %d chars): %s" % (i, len(old), len(new), old.strip().splitlines()[0][:100]))
    with open(dst, "w", encoding="utf-8") as f:
        f.write(text)
    py_compile.compile(dst, doraise=True)
    print("built %s: %d replacements, %d bytes, md5 %s; compiled" % (dst, len(PAIRS), len(text.encode("utf-8")), hashlib.md5(text.encode("utf-8")).hexdigest()))
    print("next: python page_d13_readout_selftest_v1b.py --device cpu; then python build_page_history_v6n_b.py (v6n imports v1b)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    cand = [os.path.join(HERE, "page_d13_readout_v1.py"), os.path.join(HERE, "code_v6", "page_d13_readout_v1.py"), os.path.join(HERE, "..", "code_v6", "page_d13_readout_v1.py")]
    default_src = next((c for c in cand if os.path.isfile(c)), cand[0])
    ap.add_argument("--src", default=default_src)
    ap.add_argument("--dst", default=None, help="default: page_d13_readout_v1b.py beside --src (where page_ops_v6 and page_history_v6n find it)")
    a = ap.parse_args()
    dst = a.dst or os.path.join(os.path.dirname(os.path.abspath(a.src)), "page_d13_readout_v1b.py")
    if os.path.abspath(a.src) == os.path.abspath(dst):
        sys.exit("ABORT: --dst must differ from --src (v1 is not touched)")
    if not os.path.isfile(a.src):
        sys.exit("ABORT: the source %s is missing" % a.src)
    build(a.src, dst)


if __name__ == "__main__":
    main()
