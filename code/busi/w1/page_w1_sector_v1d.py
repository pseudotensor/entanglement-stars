#!/usr/bin/env python
"""page_w1_sector_v1d.py (reviewer A, the busi lane; W1 of REVIEWA_REVIEW_web40i.md section 4 and code defects 10 to 13 of its section 5): page_w1_sector_v1c.py
with the held gate's construction allowance and the flat gate's 5-decimal values of W1_fix_v1d.md (every changed line carries "v1d"; the module is imported by
the built page_history_v6o.py and page_d13_readout_v1d.py as W1S). The changes against v1c:
(viii) construction_allowance(n_+) = c_init n_+^(3/2) eps (c_init 4, indicative) and propagation_floor's a priori bound T_cheb = construction + 2 n_+ sum of
       the Chebyshev tails + the rounding allowance: v1c's bound was 0 before any sub-step, so the premise T_meas <= T_cheb failed at every first checkpoint
       against the state's own construction impurity (the dense eigensolve's orthonormality defect, 1.44e-13 on the 96-column fixture, read through the
       sector transform), the held gate's "FAIL (premise)" and the FLOOR WARNING at T/r_s 0 of every v1c run; the impurity at the first checkpoint, T_init,
       is carried and printed with the growth T_meas - T_init (the propagation's own share) and checked against the construction allowance.
(ix)   impurity_trace_norm(X) = ||X^dagger X - 1||_1 from the Gram's eigenvalues (the self-test's flat stage measures both routes' vacuum columns with it).
(x)    relabel_text prints the flat held vacuum's values to 5 decimals (icloud X3's gate values 2.05436, 2.74643, 2.88147 and 3.11088 nats at 1e-5).
Nothing else changes against v1c: the convergence rule, the flat control rule and every readout are v1c's. v1c's docstring follows.

page_w1_sector_v1c.py (reviewer A, the busi lane; W1 of REVIEWA_REVIEW_web40i.md section 4 and code defects 10 to 13 of its section 5): page_w1_sector_v1b.py
with the held gate's floor of W1_fix_v1c.md (every changed line carries "v1c"; the module is imported by the built page_history_v6o.py and
page_d13_readout_v1c.py as W1S). The changes against v1b:
(v)   entropy_modulus(m, T) = m h2(min(T/m, 1/2)): the entropy's modulus of continuity for m paired eigenvalues whose differences sum to at most T
      (Lidskii-Mirsky's pairing, the modulus of a concave function vanishing at 0 and 1, the concavity of h2); the held gate's tolerance is this
      modulus at the state's measured impurity T = ||Phi_+^dagger Phi_+ - 1||_1 plus the entropies' rounding term, never a fixed number: the
      excess on a held record is the impurity's entropy (the pairing defect |S_Xc(C_+) - S_Xc(P_C)| equals half the excess at every checkpoint of
      the v1b log) amplified by the log of h2 at the spectrum's edges, which E_inv does not carry (|Tr[h Delta]| <= ||h|| ||Delta||_1).
(vi)  propagation_floor: the state's trace-norm error a priori (2 n_+ sum of the Chebyshev tails, a theorem, plus an indicative rounding allowance
      16 K_max eps per sub-step per column) and measured (the impurity, the whole error on a held record, a lower bound elsewhere), with the
      premise T_meas <= T_cheb.
(vii) excess_floors: every cut's excess annotated with its low floor (the measured impurity's), its high floor (the a priori bound), the rounding
      term and the resolution (resolved, marginal, unresolved); exterior_entropies_sector takes floor=...; format_w1_line prints the impurity
      against the a priori bound; convergence_rule and convergence_markdown carry the floors' weighted sums beside the excess partial sums.
Nothing of the rule changes (convergence_rule's verdicts, flat_control_rule, relabel_text, geometric_tail, tail_majorant are v1's), so
w1_postpass_v1 may keep importing page_w1_sector_v1; w1_postpass_v1c imports this module for the floors. v1b's docstring follows.

page_w1_sector_v1b.py: page_w1_sector_v1.py with the launch gates' fixes of W1_fix_v1b.md (every changed line carries "v1b"). The changes against v1:
(i)   full_trace_norm_sector takes its route from cfg["cut_norm_full"]: "svd" the exact route (torch.linalg.svdvals of W_+ = V_{+,neg} - Phi_+ (Phi_+^dagger
      V_{+,neg}), n x q, exact to rounding), anything else the Gram route (eigvalsh of 1 - M^dagger M, the small singular values to square-root accuracy);
      it returns rounding_bound_nu, the route's indicative bound on nu (the Gram route's 2 q sqrt(2 eps q) per sector summed over the two sectors), and the
      route prints with the number.
(ii)  exterior_entropies_sector reads S_X on one route at every clock, S_X = 2 S_Xc(C_+) and S_X^vac = 2 S_Xc(P_{-,+}); the check clocks compute the X block
      and the - sector beside and report their defects.
(iii) the sector coordinates of a full state are compacted to their nonzero columns (_compact) before the Grams, the cross-cut blocks and the
      formed-energy occupations.
(iv)  identity_warning: the E_kappa identity audit's threshold with a floor that follows Tr|h|.
v1's docstring follows.

page_w1_sector_v1.py (reviewer A, the busi lane). A new file, imported by page_history_v6o.py and page_d13_readout_v1c.py; nothing of v6n or v1b is
changed by it. Lattice units a = 1, torch complex128, page_ops_v6's chain conventions: sites j = 0 ... n - 1 with R_j = j + 1, 2 components per site,
flattened index 2 j + a, h[N, v] = (1/2){N, h_0} + (1/2){v, Q} at m = 0.

1. The massless sector symmetry. S_stag = diag_j((-1)^j sigma_y) commutes with h_0, Q and every clocked h[N, v] at m = 0: the on-site (kappa/R) sigma_y
   commutes with sigma_y; the bond -(i/2) sigma_x from j to j + 1 picks the sign (-1)^j (-1)^(j+1) = -1 and sigma_y sigma_x sigma_y = -sigma_x, a product
   +1; the two-step Q is spin-diagonal with (-1)^j (-1)^(j+2) = +1; the mass term anticommutes, so the symmetry is the massless channel's. The +1
   eigenvectors w_j^+ = e_{2j} (x) (1, i (-1)^j)/sqrt 2 span the + sector (rank n), the -1 eigenvectors w_j^- = conj(w_j^+) the - sector. Every entry
   of h_0 and Q is purely imaginary and N, v are real, so h = i A with A real antisymmetric: complex conjugation K satisfies K h K = -h and maps the +
   sector onto the -: h_- = -conj(h_+) in the conjugate basis, spec(h) = spec(h_+) U -spec(h_+), the - sector's eigenvectors are the conjugates of the +
   sector's, and Tr min(h, 0) = -Tr|h|/2 (the pairing audit of defect 13). The held Killing vacuum P_-(h) = 1(h < 0) is 1(h_+ < 0) on the + sector
   and conj(1(h_+ > 0)) on the - sector: one dense eigendecomposition of h_+ (n x n) in place of h's (2n x 2n), 1/8 of the flops.
2. The state. A pure Gaussian state C = Phi Phi^dagger prepared as 1(h(T_0) < 0) and propagated by exp(-i h(t) dt) with [S_stag, h(t)] = 0 keeps
   [C, S_stag] = 0 (C = C_+ (+) C_-) and the pairing C_- = conj(1_+ - C_+), since K intertwines the propagators. Every exterior cut Pi_X (R > x) is
   site-diagonal, so S_X(C) = S_X(C_+) + S_X(C_-), and the pairing with s(1 - lambda) = s(lambda) gives S_X(C) = 2 S_X(C_+) (the complement block of a
   pure state has the same entropy, so S_X = 2 S_Xc(C_+) from the smaller block). The + sector state Phi_+ (n x n_+) can itself be propagated by the
   banded h_+ = W_+^dagger h W_+ (half-bandwidth 2), the full columns being assembled at the checkpoints as [W_+ Phi_+, W_- conj(Q_perp)] with Q_perp an
   orthonormal basis of ran(Phi_+)^perp. Used only when the state respects it (sector_report).
3. The formed-energy majorant (defect 10). E_kappa = Tr[|h| (C - P)^2] with P = P_-(h): |h| = h (1 - 2P) and [h, P] = 0 give Tr[h (1 - 2P)(C - P)^2]
   = Tr[h C^2] - 2 Tr[h P C^2] + 2 Tr[h P C] - Tr[h P], which for the pure state C^2 = C is Tr[h (C - P)] = E_inv; spectrally E_kappa =
   sum_{e > 0} e n_j + sum_{e < 0} |e| (1 - n_j), equal to E_inv for any C. convergence_rule: the weighted terms w_kappa E_kappa must fall geometrically
   over the last n_fit channels; the excess's tail by the entropy per formed energy of the last channels times the energy tail; the collar flow's tail
   by its own geometric majorant; never the last channel's share. The proved form is the icloud lane's (X3); tail_majorant is the one function to
   replace when it lands.
4. The sub-stepped propagation (defect 11). record_substeps: "none" the record's fields over [T_k, T_k + tau) (the control), "linear" n_sub sub-steps with
   the fields interpolated linearly between record k and k + 1 at each sub-step's midpoint (the production), "cubic" v6m's Catmull-Rom window.
5. The audits (defect 13): operator_audit on every record's sparse h, spectral_audit on every banded or dense eigensolve, formed_energy on every dense
   vacuum, fast_banded_matches on the field-built banded operator the probes use.
"""
import math
import time

import numpy as np
import scipy.sparse as sp

import torch

import page_ops_v6 as OPS

CD = OPS.CD
RD = OPS.RD
T_START = time.time()
SQ2 = math.sqrt(2.0)
EPS64 = 2.220446049250313e-16                                                   # v1b: the double's unit roundoff (the Gram route's indicative bound)
_WPLUS = {}


def log(msg):
    print("[%8.1f s] %s" % (time.time() - T_START, msg), flush=True)


def _f(x, fmt):
    if x is None:
        return "-"
    try:
        return fmt % x
    except (TypeError, ValueError):
        return str(x)


# ---------------------------------------------------------------------------------------------------------- the sector transform
def stag_signs(nsites):
    """(-1)^j over the sites j = 0 ... n - 1."""
    return np.where(np.arange(int(nsites)) % 2 == 0, 1.0, -1.0)


def stag_sparse(nsites):
    """S_stag = diag_j((-1)^j sigma_y) as a 2n x 2n CSR (sigma_y = [[0, -i], [i, 0]])."""
    n = int(nsites); j = np.arange(n); s = stag_signs(n)
    rows = np.concatenate([2 * j, 2 * j + 1]); cols = np.concatenate([2 * j + 1, 2 * j])
    vals = np.concatenate([-1j * s, 1j * s])
    return sp.csr_matrix((vals, (rows, cols)), shape=(2 * n, 2 * n))


def W_plus_sparse(nsites):
    """W_+ (2n x n): column j = e_{2j} (x) (1, i (-1)^j)/sqrt 2, the + sector's orthonormal basis; W_- = conj(W_+). Cached per n."""
    n = int(nsites)
    if n not in _WPLUS:
        j = np.arange(n); s = stag_signs(n)
        rows = np.concatenate([2 * j, 2 * j + 1]); cols = np.concatenate([j, j])
        vals = np.concatenate([np.full(n, 1.0 / SQ2, complex), 1j * s / SQ2])
        _WPLUS[n] = sp.csr_matrix((vals, (rows, cols)), shape=(2 * n, n))
    return _WPLUS[n]


def to_sector(X, sign=1):
    """the sector coordinates of the columns X (2n x m, device): W_+^dagger X (sign +1) or W_-^dagger X (sign -1), n x m."""
    n = X.shape[0] // 2
    s = torch.as_tensor(stag_signs(n), dtype=RD, device=X.device).to(CD)
    A = X[0::2]; B = X[1::2]
    if sign > 0:
        return (A - (1j * s)[:, None] * B) / SQ2
    return (A + (1j * s)[:, None] * B) / SQ2


def from_sector(Y, sign=1):
    """W_+ Y (sign +1) or W_- Y = conj(W_+) Y (sign -1): 2n x m from the sector coordinates n x m."""
    n = Y.shape[0]
    s = torch.as_tensor(stag_signs(n), dtype=RD, device=Y.device).to(CD)
    Z = torch.empty((2 * n,) + tuple(Y.shape[1:]), dtype=CD, device=Y.device)
    Z[0::2] = Y / SQ2
    Z[1::2] = ((1j if sign > 0 else -1j) * s)[:, None] * Y / SQ2
    return Z


def _compact(Y):
    """v1b: the columns of Y (n x m) with a nonzero norm. The sector coordinates of a full state hold exact zeros for the other sector's columns:
    to_sector(from_sector(Y, s), -s) = (A + i s (i s A))/sqrt 2 = 0 exactly, since (i s)(i s) = -1 and s^2 = 1 in floating point; dropping them changes
    no product Y Y^dagger (the column space is unchanged) and halves every Gram, block and occupation product on the full assembled columns."""
    if Y.shape[1] == 0:
        return Y
    nz = Y.abs().pow(2).sum(dim=0) > 0.0
    return Y if bool(nz.all()) else Y[:, nz]


def sector_plus_csr(h_csr):
    """h_+ = W_+^dagger h W_+ (n x n CSR, half-bandwidth 2): real on the diagonal ((-1)^j kappa N_j/R_j) and the nearest-neighbour entries
    (-(-1)^j (N_j + N_{j+1})/4), imaginary on the next-nearest (-(i/8)(v_j + v_{j+2}))."""
    n = h_csr.shape[0] // 2
    W = W_plus_sparse(n)
    return (W.conj().T @ (h_csr @ W)).tocsr()


def sector_blocks_csr(h_csr):
    """(h_+, h_-, max|W_-^dagger h W_+|): the two sector blocks and the off-block's largest entry (0 at m = 0)."""
    n = h_csr.shape[0] // 2
    W = W_plus_sparse(n)
    hW = h_csr @ W
    hp = (W.conj().T @ hW).tocsr()
    off = W.T @ hW
    hm = (W.T @ (h_csr @ W.conj())).tocsr()
    omax = float(np.max(np.abs(off.data))) if off.nnz else 0.0
    return hp, hm, omax


def trace_diag(A, w=None):
    """Re sum_j w_j A_jj for a sparse A and site weights w (numpy or a device tensor column; None = all ones)."""
    d = np.asarray(A.diagonal())
    if w is None:
        return float(np.real(d.sum()))
    if isinstance(w, torch.Tensor):
        ww = w.detach().cpu().numpy().ravel().real
    else:
        ww = np.asarray(w, float).ravel()
    return float(np.real((d * ww).sum()))


def current_out_plus_csr(hp, R, R_c):
    """J_+ = (i/2)[h_+^2, Pi_{R > R_c}] on the sector basis (one entry per site; page_ops_v6.current_out_csr without the component repeat)."""
    pi = (np.asarray(R, float) > R_c).astype(float)
    h2 = (hp @ hp).tocoo()
    data = 0.5j * h2.data * (pi[h2.col] - pi[h2.row])
    keep = np.abs(data) > 0
    return sp.csr_matrix((data[keep], (h2.row[keep], h2.col[keep])), shape=h2.shape)


# ----------------------------------------------------------------------------------------------------------------- the audits
def operator_audit(h_csr, tol=1e-13):
    """defect 13, per record, exact sparse arithmetic: the pairing premise max|h + conj h| (h = i A with A real antisymmetric, so K h K = -h and the
    spectrum is +-e) and the sector commutator max|[S_stag, h]|; a diagonal (mass) term breaks both. ok when both are below tol x max|h|."""
    n = h_csr.shape[0] // 2
    hc = h_csr.tocsr()
    prem = hc + hc.conj()
    m_prem = float(np.max(np.abs(prem.data))) if prem.nnz else 0.0
    S = stag_sparse(n)
    comm = S @ hc - hc @ S
    m_comm = float(np.max(np.abs(comm.data))) if comm.nnz else 0.0
    hmax = float(np.max(np.abs(hc.data))) if hc.nnz else 1.0
    return {"max_h_plus_conj_h": m_prem, "max_commutator_S_stag": m_comm, "h_max": hmax, "tol_rel": float(tol),
            "ok": bool(m_prem <= tol * hmax and m_comm <= tol * hmax)}


def spectral_audit(e, tol=1e-10):
    """defect 13, per eigensolve: Tr min(h, 0) = -Tr|h|/2 as the residual sum_{e < 0} e + sum|e|/2 relative to sum|e|; the +-e asymmetry sum e; the
    gap min|e|; the zero-mode count."""
    e = np.asarray(e, float)
    if e.size == 0:
        return {"ok": True, "tr_min_h0": 0.0, "tr_abs": 0.0, "pairing_residual": 0.0, "pairing_residual_rel": 0.0, "sum_e": 0.0, "abs_min": None, "n_zero_1e-12": 0, "tol_rel": float(tol)}
    tr_abs = float(np.abs(e).sum()); neg = e < 0
    res = float(e[neg].sum() + 0.5 * tr_abs)
    return {"tr_min_h0": float(e[neg].sum()), "tr_abs": tr_abs, "pairing_residual": res, "pairing_residual_rel": res / max(tr_abs, 1e-300), "sum_e": float(e.sum()),
            "abs_min": float(np.abs(e).min()), "n_zero_1e-12": int((np.abs(e) < 1e-12).sum()), "tol_rel": float(tol), "ok": bool(abs(res) <= tol * max(tr_abs, 1e-300)),
            "label": "Tr min(h, 0) = -Tr|h|/2 by the antiunitary pairing K h K = -h of the massless chain (h = i A, A real antisymmetric)"}


def identity_warning(E_inv, fe, aud, rel=1e-8, floor_rel=1e-11):
    """v1b: the E_kappa identity audit's threshold. |E_kappa(spectral) - E_inv| is compared with max(rel |E_inv|, floor_rel max(Tr|h|, 1)): the spectral
    form sums 2n terms of size |e| with a rounding of order eps Tr|h|, and the banded E_ref against the sector's differs by the same order, so the floor
    follows Tr|h| (the banded audit's tr_abs). Returns (warn, tol_abs, residual); a failed banded pairing audit also warns."""
    aud = aud or {}
    tr_abs = float(aud.get("tr_abs") or 0.0)
    tol = max((rel * abs(float(E_inv))) if E_inv is not None else 0.0, floor_rel * max(tr_abs, 1.0))
    res = (float(fe["E_kappa"]) - float(E_inv)) if (fe and fe.get("E_kappa") is not None and E_inv is not None) else None
    warn = bool((aud and not aud.get("ok", True)) or (res is not None and abs(res) > tol))
    return warn, tol, res


# ------------------------------------------------------------------------------------------- v1c: the propagation's floor (W1_fix_v1c)
def entropy_modulus(m, T):
    """v1c: the entropy's modulus of continuity for m paired eigenvalues whose differences sum to at most T: m h2(min(T/m, 1/2)) in nats
    (page_ops_v6.rank_sensitive_bound). Theorem: (1) for Hermitian blocks A, B with eigenvalues in [0, 1] the sorted eigenvalues satisfy
    sum_k |lambda_k(A) - lambda_k(B)| <= ||A - B||_1 (Lidskii-Mirsky); (2) the binary entropy s is concave with s(0) = s(1) = 0, so its modulus of
    continuity is attained at the endpoints, |s(x) - s(y)| <= h2(|x - y|) for |x - y| <= 1/2 (the increment of a concave function over a fixed length
    is largest at the ends); (3) h2 is concave, so the sum over the m pairs that involve a nonzero eigenvalue (zeros pair with zeros in the sorted
    sequences) is at most m h2(T/m). The log in h2(delta) = delta (ln(1/delta) + 1 + ...) is the amplification of a small change of the state at the
    spectrum's edges (a factor 31 at delta = 1e-13); E_inv has no such factor (|Tr[h Delta]| <= ||h|| ||Delta||_1), which is why the held gate's
    excess exceeds its E_inv drift by two orders (W1_fix_v1c.md section 2)."""
    m = int(m)
    if m <= 0 or T is None or float(T) <= 0.0:
        return 0.0
    return float(OPS.rank_sensitive_bound(m, float(T)))


def construction_allowance(n_plus, c_init=4.0):
    """v1d: the indicative allowance for the state's construction impurity ||Phi^dagger Phi - 1||_1 before any sub-step: c_init n_+^(3/2) eps. A dense
    Hermitian eigensolve returns columns orthonormal to a Gram defect whose entries are of order eps, and the trace norm of an n_+ x n_+ matrix of such
    entries is of order n_+^(3/2) eps (its eigenvalues spread over about 2 sqrt(n_+) eps): measured 1.44e-13 at n_+ 96 (0.69 n_+^(3/2) eps, the sector
    route) and 3.74e-13 at n_+ 192 (0.63, the dense route) on the W1 fixture, so c_init 4 is a factor about 6 above the measured values. Indicative,
    not a theorem, like the rounding allowance; 0 for n_+ 0. v1c's a priori bound had no such term, so it was 0 at the first checkpoint and the
    premise failed there against the state's own construction (W1_fix_v1d.md section 1)."""
    n_plus = int(n_plus)
    return float(c_init) * (float(n_plus) ** 1.5) * EPS64 if n_plus > 0 else 0.0


def impurity_trace_norm(X):
    """v1d: ||X^dagger X - 1||_1, the trace-norm distance of X X^dagger from the nearest projector, from the Gram's eigenvalues (the Gram formed whole:
    N x N for N columns, 1 GiB complex128 at N 8192); 0 for no columns."""
    if X is None or X.shape[1] == 0:
        return 0.0
    G = X.mH @ X
    G.diagonal().sub_(1.0)
    ev = torch.linalg.eigvalsh(0.5 * (G + G.mH)); del G
    return float(ev.abs().sum())


def propagation_floor(n_plus, cheb_tail_acc, n_substeps, K_max, T_meas=None, c_round=16.0, T_init=None, c_init=4.0):                                  # v1d: T_init, c_init
    """v1c: the state's trace-norm error from the propagation, a priori and measured. A priori, the theorem part: each sub-step's Chebyshev
    polynomial p_s(h_s) differs from exp(-i h_s dt) by at most tail_s in operator norm on the Gershgorin interval (page_ops_v6.cheb_order's tail), so
    after the sub-steps every column differs from the exact one by at most sum_s tail_s = cheb_tail_acc (telescoping, ||p_s|| <= 1 + tail_s),
    ||Phi - Phi_exact||_F <= sqrt(n_+) cheb_tail_acc and ||C - C_exact||_1 <= 2 ||Phi - Phi_exact||_F ||Phi||_F <= 2 n_+ cheb_tail_acc. On a held
    record the polynomial keeps the eigencolumns, so the error is the norm drift d_j = |p(e_j)|^(2 n) - 1 of each column, coherent over the sub-steps
    (2 Re(e^(i e dt) r(e/w)) per step, r the truncation); the leading coefficient c_(K+1) = 2 (-i)^(K+1) J_(K+1)(z) is real for even K + 1 (a norm
    error 2 J_(K+1) |T_(K+1)(y)| per step) and imaginary for odd K + 1 (a phase error; the norm drift suppressed by sin(e dt) <= sin z), so the drift's
    size depends on the parity of the Chebyshev order (the v1b log's factor 10 between kappa 1 at K 10 and kappa 2 at K 11). Rounding, indicative and
    not a theorem: c_round K_max eps per sub-step per column (the K applies of a 5-diagonal operator and the three-term recursion, whose growth is at
    most linear on the Gershgorin interval). Measured: T_meas = ||Phi^dagger Phi - 1||_1, the impurity, the distance of C from its range projector
    (the nearest projector), equal to the whole error on a held record and a lower bound of it elsewhere (a rotation of the range is not seen).
    v1d: the a priori bound T_cheb = T_cheb_construction + T_cheb_truncation + T_cheb_rounding_allowance, the construction term c_init n_+^(3/2) eps
    (construction_allowance, indicative) covering the state's own impurity before any sub-step, where v1c's bound was 0 and the premise failed at
    every first checkpoint against the eigensolve's rounding (1.44e-13 on the fixture); T_init, the impurity measured at the first checkpoint, is
    carried and printed with the growth T_meas - T_init (the propagation's own share) and checked against the construction allowance
    (construction_ok). The premise T_meas <= T_cheb is the construction check itself at the first checkpoint."""
    n_plus = int(n_plus); n_sub = int(n_substeps); K = int(K_max)
    trunc = 2.0 * n_plus * float(cheb_tail_acc or 0.0)
    rnd = n_plus * n_sub * float(c_round) * max(K, 1) * EPS64
    cons = construction_allowance(n_plus, c_init)                                                                                                   # v1d
    Tm = float(T_meas) if T_meas is not None else None
    Ti = float(T_init) if T_init is not None else None                                                                                              # v1d
    Tc = cons + trunc + rnd                                                                                                                         # v1d: the construction term in the a priori bound
    return {"n_plus": n_plus, "cheb_tail_acc": float(cheb_tail_acc or 0.0), "n_substeps": n_sub, "cheb_K_max": K, "c_round": float(c_round), "c_init": float(c_init),   # v1d
            "T_cheb_construction": cons, "T_cheb_truncation": trunc, "T_cheb_rounding_allowance": rnd, "T_cheb": Tc, "T_meas": Tm, "T_init": Ti,            # v1d
            "T_growth": (Tm - Ti) if (Tm is not None and Ti is not None) else None,                                                                    # v1d: the propagation's own share of the impurity
            "construction_ok": (bool(Ti <= cons) if Ti is not None else None),                                                                          # v1d
            "premise_ok": (bool(Tm <= Tc) if Tm is not None else None),
            "label": ("T_cheb = %g n_+^(3/2) eps (the construction allowance, indicative: the eigensolve's orthonormality defect before any sub-step; v1d) + 2 n_+ sum_s tail_s (the "
                      "Chebyshev truncation, a theorem: ||C - C_exact||_1 at most this) + n_+ n_sub %g K_max eps (the rounding allowance, indicative); T_meas = ||Phi^dagger Phi - 1||_1 "
                      "the state's impurity (the whole error on a held record, a lower bound elsewhere), T_init its value at the first checkpoint (the construction), T_growth the "
                      "propagation's share; the premise T_meas <= T_cheb says the drift is the construction's and the propagation's" % (float(c_init), float(c_round)))}   # v1d


def excess_floors(ent, R_sites, cuts, floor, sector=True, c_eig=8.0):
    """v1c: per cut (label, R_abs) the floor of the excess from the propagation, written into ent[label]: m_pairs = min(n_Xc, max(n_+, q)) the paired
    eigenvalues (sites for the sector route, modes for the full route); excess_floor_low = f m h2(T_meas/m) + the rounding term (the measured
    impurity's entropy; on a held record the whole propagation error), excess_floor_high = f m h2(T_cheb/m) + the rounding term (the a priori bound
    with the rounding allowance), excess_floor_round = f m h2(c_eig eps) (the two Gram eigensolves' rounding, indicative: c_eig eps per eigenvalue),
    f = 2 for the sector route (S_X = 2 S_Xc(C_+)) and 1 for the full route; excess_resolution: 'resolved' where |excess| exceeds the high floor,
    'marginal' between the floors, 'unresolved' below the low floor. floor: propagation_floor's dict with 'q' (the vacuum's rank) added."""
    if not floor:
        return ent
    Rs = np.asarray(R_sites, float)
    n_plus = int(floor.get("n_plus") or 0)
    q = int(floor["q"]) if floor.get("q") is not None else n_plus
    T_meas = floor.get("T_meas"); T_cheb = floor.get("T_cheb")
    fac = 2.0 if sector else 1.0
    for lab, x in cuts:
        d = ent.get(lab)
        if not isinstance(d, dict):
            continue
        nC = int(np.sum(Rs <= x)) * (1 if sector else 2)
        m = int(min(nC, max(n_plus, q)))
        rnd = fac * entropy_modulus(m, m * float(c_eig) * EPS64)
        low = (fac * entropy_modulus(m, float(T_meas)) + rnd) if T_meas is not None else None
        high = (fac * entropy_modulus(m, float(T_cheb)) + rnd) if T_cheb is not None else None
        d.update({"m_pairs": m, "excess_floor_round": rnd, "excess_floor_low": low, "excess_floor_high": high, "floor_T_meas": T_meas, "floor_T_cheb": T_cheb,
                  "floor_label": "the excess's floor from the propagation: low = %g m h2(T_meas/m) + round (the measured impurity's entropy), high = %g m h2(T_cheb/m) + round "
                                 "(the a priori Chebyshev bound with the construction and rounding allowances, v1d), round = %g m h2(%g eps); m the paired eigenvalues (v1c, W1_fix_v1c)" % (fac, fac, fac, c_eig)})   # v1d
        exc = d.get("excess")
        if exc is None or low is None or high is None:
            d["excess_resolution"] = None
        elif abs(exc) > high:
            d["excess_resolution"] = "resolved"
        elif abs(exc) > low:
            d["excess_resolution"] = "marginal"
        else:
            d["excess_resolution"] = "unresolved"
    return ent


# ------------------------------------------------------------------------------------------------------ the vacuum and the state
def killing_vacuum_sector(h_csr, device):
    """P_-(h) by the + sector: the dense eigendecomposition of h_+ (n x n); the - sector's occupied vectors are the conjugates of the + sector's
    empty ones at the energies -e_+. Returns the full Vneg (2n x n_neg) in the chain's basis and the sector data e_plus, V_plus (kept for the checks)."""
    t0 = time.time()
    hp, hm, off = sector_blocks_csr(h_csr)
    e_p, V_p = OPS.eigh_csr(hp, device)
    neg = e_p < 0; pos = e_p > 0
    Vneg = torch.cat([from_sector(V_p[:, neg], 1), from_sector(V_p[:, pos].conj(), -1)], dim=1)
    e_np = e_p.cpu().numpy().astype(float)
    e_full = np.sort(np.concatenate([e_np, -e_np]))
    E_ref = float(e_np[e_np < 0].sum() - e_np[e_np > 0].sum())
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {"e": e_full, "Vneg": Vneg, "n_neg": int(Vneg.shape[1]), "n_zero_1e-12": int((np.abs(e_full) < 1e-12).sum()), "E_ref": E_ref,
            "route": "sector (eigh of h_+ = W_+^dagger h W_+, n x n; the - sector's occupied vectors conj(V_+[:, e_+ > 0]) at -e_+)", "seconds": time.time() - t0,
            "e_plus": e_np, "V_plus": V_p, "n_neg_plus": int(neg.sum()), "n_pos_plus": int(pos.sum()), "n_zero_plus": int((~neg & ~pos).sum()), "sector_offblock_max": off,
            "audit_plus": spectral_audit(e_full)}


def sector_report(h_csr, Phi, tol_state=1e-8, tol_op=1e-13):
    """the operator's defects (operator_audit) and the state's: Tr C_+, Tr C_-, the off-sector weight ||P_- C P_+||_F = ||Phi_- Phi_+^dagger||_F."""
    op = operator_audit(h_csr, tol_op)
    rep = {"operator": op, "state": None, "ok": bool(op["ok"])}
    if Phi is not None and Phi.shape[1]:
        Pp = to_sector(Phi, 1); Pm = to_sector(Phi, -1)
        trp = float(Pp.abs().pow(2).sum()); trm = float(Pm.abs().pow(2).sum())
        off = float(torch.linalg.norm(Pm @ Pp.mH))
        del Pp, Pm
        st = {"tr_C_plus": trp, "tr_C_minus": trm, "off_sector_F": off, "tol": float(tol_state), "ok": bool(off <= tol_state)}
        rep["state"] = st; rep["ok"] = bool(rep["ok"] and st["ok"])
    return rep


def assemble_full_state(Pp):
    """the full occupied columns from the + sector's: [W_+ Phi_+, W_- conj(Q_perp)], Q_perp an orthonormal basis of ran(Phi_+)^perp in C^n (the - sector's
    state is conj(1_+ - C_+)); 2n x n, orthonormal to rounding."""
    n, m = Pp.shape
    Q, _ = torch.linalg.qr(Pp, mode="complete")
    Qperp = Q[:, m:]
    return torch.cat([from_sector(Pp, 1), from_sector(Qperp.conj(), -1)], dim=1)


def gram_eigs(X):
    """the nonzero spectrum of X X^dagger from the smaller of X^dagger X and X X^dagger; (eigenvalues ascending, the route's label)."""
    if X.shape[0] == 0 or X.shape[1] == 0:
        return torch.zeros(0, dtype=RD, device=X.device), "empty"
    if X.shape[1] <= X.shape[0]:
        G = X.mH @ X; route = "gram %d x %d" % (X.shape[1], X.shape[1])
    else:
        G = X @ X.mH; route = "block %d x %d" % (X.shape[0], X.shape[0])
    G = 0.5 * (G + G.mH)
    try:
        ev = torch.linalg.eigvalsh(G)
    except torch._C._LinAlgError:                                                                                                  # v1d-pod7: cusolver syevd failed to converge on pod 7's H100 (error 4073, kappa 1 at T 0); the same matrix on the CPU's LAPACK
        ev = torch.linalg.eigvalsh(G.cpu()).to(G.device); route += " (cpu eigvalsh fallback)"
    del G
    return ev, route


def entropy_of_eigs(ev):
    return float(OPS.binary_entropy_t(ev).sum()) if ev.numel() else 0.0


def exterior_entropies_sector(Phi, R_sites, cuts, vac=None, check_minus=False, floor=None):
    """per cut (label, R_abs): S_X(C) = 2 S_Xc(C_+) from the + sector coordinates Phi_+ = W_+^dagger Phi restricted to the sites within the cut (the
    smaller Gram's spectrum: the complement block of the pure + sector state, whose entropy equals the X block's to rounding, doubled for the - sector by
    the pairing), the held vacuum's S_X^vac = 2 S_Xc(P_{-,+}) from V_+[:, e_+ < 0], and the excess S_X - S_X^vac. v1b: one route at every clock. With
    check_minus the X blocks and the - sector halves are computed beside and their defects return under '_sector' and in the cut dicts: the purity gaps
    S_X(C_+) - S_Xc(C_+) and S_X(C_-) - S_Xc(C_-), the pairing defects |S_Xc(C_+) - S_Xc(C_-)| and |S_Xc^vac_+ - S_Xc^vac_-|, S_X_both_Xblocks. v1c: with
    floor (propagation_floor's dict with 'q') every cut's excess is annotated with its floors and resolution (excess_floors). Keys per cut: S_X, S_Xc,
    purity_gap, S_X_vac, S_Xc_vac, excess, excess_c, the spectra eig_*_plus (and _minus), the floors."""
    dev = Phi.device
    Rs = np.asarray(R_sites, float)
    Pp = _compact(to_sector(Phi, 1))                                                                                              # v1b: the + sector's columns, the - sector's exact zeros dropped
    Pm = _compact(to_sector(Phi, -1)) if check_minus else None                                                                     # v1b
    Vp = Vm = None
    if vac is not None and vac.get("V_plus") is not None:
        ep = torch.as_tensor(np.asarray(vac["e_plus"], float), dtype=RD, device=dev)
        Vp = vac["V_plus"][:, ep < 0]
        if check_minus:
            Vm = vac["V_plus"][:, ep > 0].conj()
    out = {}
    sec = {"check_minus": bool(check_minus), "cuts": {},
           "route": "the + sector (W_+^dagger Phi, V_+[:, e_+ < 0]): S_X = 2 S_Xc(C_+) and S_X^vac = 2 S_Xc(P_{-,+}) at every clock (v1b: one route; the complement "
                    "block of the pure + sector state, equal to the X block's entropy to rounding, doubled for the - sector by the pairing)"
                    + ("; the X blocks and the - sector computed beside at this clock (the check: the purity gaps and the pairing defects)" if check_minus else "")}
    for lab, x in cuts:
        mask = torch.as_tensor(Rs > x, dtype=torch.bool, device=dev)
        nX = int(mask.sum()); nC = int((~mask).sum())
        evC, rC = gram_eigs(Pp[~mask]); S_c_p = entropy_of_eigs(evC)
        SX = 2.0 * S_c_p                                                                                                           # v1b: the readout's route at every clock
        d = {"R_abs": float(x), "n_modes_X": 2 * nX, "n_modes_Xc": 2 * nC, "n_sites_X": nX, "n_sites_Xc": nC, "route_Xc": "sector +: " + rC,
             "S_Xc_plus": S_c_p, "eig_Xc_plus": evC.cpu().numpy(), "S_X": SX, "S_Xc": SX, "purity_gap": 0.0, "purity_gap_formed": bool(check_minus),       # v1b
             "route_X": "S_X = 2 S_Xc(C_+): the + sector's complement block at every clock (v1b); the X blocks' value at the check clocks under S_X_both_Xblocks",   # v1b
             "S_X_vac": None, "S_Xc_vac": None, "excess": None, "excess_c": None}
        chk = {}
        if check_minus:
            evX, rX = gram_eigs(Pp[mask]); S_x_p = entropy_of_eigs(evX)
            evCm, _ = gram_eigs(Pm[~mask]); S_c_m = entropy_of_eigs(evCm)
            evXm, _ = gram_eigs(Pm[mask]); S_x_m = entropy_of_eigs(evXm)
            d.update({"route_X_check": "sector +,-: " + rX, "S_X_plus": S_x_p, "S_Xc_minus": S_c_m, "S_X_minus": S_x_m, "eig_X_plus": evX.cpu().numpy(),
                      "eig_Xc_minus": evCm.cpu().numpy(), "eig_X_minus": evXm.cpu().numpy(),
                      "purity_gap_plus": S_x_p - S_c_p, "purity_gap_minus": S_x_m - S_c_m, "pairing_defect_state": abs(S_c_p - S_c_m),
                      "S_X_both_Xblocks": S_x_p + S_x_m, "S_Xc_both_sectors": S_c_p + S_c_m,                                                               # v1b
                      "purity_gap": (S_x_p + S_x_m) - (S_c_p + S_c_m)})                                                                                 # v1b: the purity gap of the full state (X blocks against Xc blocks, both sectors)
            chk.update({"pairing_defect_state": d["pairing_defect_state"], "purity_gap": d["purity_gap"]})
        if Vp is not None:
            evCv, _ = gram_eigs(Vp[~mask]); Svc = entropy_of_eigs(evCv)
            d["eig_Xc_vac_plus"] = evCv.cpu().numpy()
            SXv = 2.0 * Svc                                                                                                        # v1b: the vacuum's route at every clock
            if check_minus:
                evCvm, _ = gram_eigs(Vm[~mask]); Svcm = entropy_of_eigs(evCvm)
                d["eig_Xc_vac_minus"] = evCvm.cpu().numpy(); d["pairing_defect_vacuum"] = abs(Svc - Svcm); d["S_Xc_vac_both_sectors"] = Svc + Svcm       # v1b
                chk["pairing_defect_vacuum"] = d["pairing_defect_vacuum"]
            d.update({"S_X_vac": SXv, "S_Xc_vac": SXv, "excess": SX - SXv, "excess_c": SX - SXv})                                   # v1b: the excess on the one route
        sec["cuts"][lab] = chk
        out[lab] = d
    if floor is not None:                                                                                                          # v1c: the floors beside every excess
        excess_floors(out, Rs, cuts, floor, sector=True)                                                                           # v1c
        sec["floor"] = {k_: floor.get(k_) for k_ in ("T_meas", "T_cheb", "T_cheb_construction", "T_cheb_truncation", "T_cheb_rounding_allowance", "T_init", "T_growth",
                                                      "construction_ok", "n_plus", "q", "n_substeps", "cheb_K_max", "premise_ok")}   # v1d: the construction term and T_init beside the floor's inputs
    out["_sector"] = sec
    return out


def full_trace_norm_sector(Pp, Vp, cfg=None, sigma_tol=1e-10):
    """nu = ||C - P_-||_1 = 2 ||C_+ - P_{-,+}||_1 (the - sector's block is -conj of the + sector's) by the principal angles between ran C_+ and ran P_{-,+}:
    the q = #neg(h_+) singular values sigma_j = sin theta_j of (1 - C_+) V_{+,neg}, nu_+ = 2 sum sigma_j + (Tr C_+ - q). v1b: the route from
    cfg["cut_norm_full"]: "svd" the exact route (torch.linalg.svdvals of W_+ = V_{+,neg} - Phi_+ (Phi_+^dagger V_{+,neg}), n x q, exact to rounding),
    anything else the Gram route (eigvalsh of 1 - M^dagger M, M = Phi_+^dagger V_{+,neg}, q x q: the small sigma_j to square-root accuracy);
    rounding_bound_nu the route's indicative bound on nu. On a held record ran C_+ = ran V_{+,neg} and sigma_j = |d_j|, the eigenvalues of
    Phi_+^dagger Phi_+ - 1, so sum sigma_j = (nu_+ - (Tr C_+ - q))/2 is the state's impurity ||Phi_+^dagger Phi_+ - 1||_1 (v1c: the held gate's
    consistency check; N_occ_plus = Tr C_+ and N_neg_plus = q are returned for it)."""
    t0 = time.time()
    mode = str((cfg or {}).get("cut_norm_full", "auto"))                                                                             # v1b
    route = "svd" if mode == "svd" else "gram"                                                                                       # v1b
    Pp = _compact(Pp)                                                                                                                # v1b
    q = int(Vp.shape[1]); n_plus = float(Pp.abs().pow(2).sum())
    M = Pp.mH @ Vp
    trCP = float(torch.linalg.norm(M)) ** 2
    if route == "gram":                                                                                                              # v1b
        G = M.mH @ M; del M
        G.neg_(); G.diagonal().add_(1.0)
        lam = torch.linalg.eigvalsh(0.5 * (G + G.mH)); del G
        s = torch.sqrt(lam.clamp(0.0, 1.0)).flip(0); del lam
        rb = 2.0 * (2.0 * q * math.sqrt(2.0 * EPS64 * q))                                                                           # v1b: both sectors
        rlab = "sector gram (eigvalsh of 1 - M^dagger M, M = Phi_+^dagger V_{+,neg}, %d x %d; the small sigma_j to square-root accuracy; the - sector's trace norm equals the + sector's)" % (q, q)
    else:                                                                                                                            # v1b: the exact route
        W = Vp.clone(); W.addmm_(Pp, M, alpha=-1.0); del M
        s = torch.linalg.svdvals(W); del W
        rb = 2.0 * (2.0 * q * 8.0 * EPS64)                                                                                           # v1b
        rlab = "sector svd (torch.linalg.svdvals of (1 - C_+) V_{+,neg}, %d x %d, exact to rounding; the - sector's trace norm equals the + sector's)" % (int(Pp.shape[0]), q)
    sv = s.cpu().numpy().astype(float); del s
    dn = n_plus - q
    nu_plus = 2.0 * float(sv.sum()) + dn
    frob2_plus = n_plus + q - 2.0 * trCP
    out = {"route": rlab, "route_short": route, "mode_requested": mode, "rounding_bound_nu": rb,                                                      # v1b
           "nu_plus": nu_plus, "nu": 2.0 * nu_plus, "N_occ_plus": n_plus, "N_neg_plus": q, "rank": int(2 * (2 * int((sv > sigma_tol).sum()) + int(round(dn)))),
           "sigma_sum_plus": float(sv.sum()),                                                                                        # v1c: sum sigma_j (= the impurity on a held record)
           "sigma_max": max(float(sv[0]) if sv.size else 0.0, 1.0 if int(round(dn)) != 0 else 0.0), "frob2": 2.0 * frob2_plus, "frob": math.sqrt(max(2.0 * frob2_plus, 0.0)),
           "tr_CP_plus": trCP, "n_pair_plus": n_plus - trCP, "n_pair": 2.0 * (n_plus - trCP), "sv": sv, "sigma_tol": float(sigma_tol),
           "bound_rank_sigma": None, "bound_cauchy_schwarz": None, "seconds": time.time() - t0,
           "label": "nu = ||C - P_-||_1 by the principal angles between ran C_+ and ran P_{-,+} (page_d13_readout_v1b.full_trace_norm's %s route on the + sector), doubled for the - sector; "
                    "rounding_bound_nu is the route's indicative bound (v1b)" % route}
    return out


def cross_cut_readout_sector(Phi, vac, R_sites, cuts, ent, cfg, kappa, trs, sc, arrays):
    """(a') on the + sector: per cut the singular values of the X-by-Xc block of C_+ - P_{-,+} (n_X/2 x n_Xc/2), t_X = 2 (||K_+||_1 + ||K_-||_1) =
    4 sum sv_+ (the - sector's block is -conj of the + sector's), the check |S_X(C) - S_X(P_-)| <= t_X (EVOLVEDCUT EC1), the - sector's block beside
    at the check clocks; the full ||C - P_-||_1 by full_trace_norm_sector on cfg["cut_norm_full"]'s route (v1b). The results go into ent's cut dicts and
    sc['trace_norm']."""
    dev = Phi.device
    Rs = np.asarray(R_sites, float)
    ep = torch.as_tensor(np.asarray(vac["e_plus"], float), dtype=RD, device=dev)
    Vp = vac["V_plus"][:, ep < 0]
    Pp = _compact(to_sector(Phi, 1))                                                                                              # v1b
    check = bool((sc.get("sector") or {}).get("check_minus"))
    Pm = _compact(to_sector(Phi, -1)) if check else None                                                                           # v1b
    Vm = vac["V_plus"][:, ep > 0].conj() if check else None
    tol = float(cfg.get("cut_bound_tol", 1e-6)); fails = []
    for lab, x in cuts:
        t0 = time.time()
        mask = torch.as_tensor(Rs > x, dtype=torch.bool, device=dev)
        K = Pp[mask] @ Pp[~mask].mH
        fro_C = float(torch.linalg.norm(K)) ** 2
        K.addmm_(Vp[mask], Vp[~mask].mH, alpha=-1.0)
        fro_D = float(torch.linalg.norm(K)) ** 2
        s = torch.linalg.svdvals(K); del K
        sv = s.cpu().numpy().astype(float); del s
        tplus = 2.0 * float(sv.sum())
        d = ent[lab]
        d.update({"t_X_plus": tplus, "t_X": 2.0 * tplus, "n_sv": int(sv.size), "sigma_max_block": float(sv[0]) if sv.size else 0.0, "n_sv_above_1e-8": int((sv > 1e-8).sum()),
                  "frob2_block_plus": fro_D, "frob2_block_C_plus": fro_C, "rows_ordered": True, "n_X": 2 * int(mask.sum()), "n_Xc": 2 * int((~mask).sum()),
                  "block_route": "sector +: svdvals of the %d x %d block of C_+ - P_{-,+}, doubled for the - sector (its block is -conj of the + sector's)" % (int(mask.sum()), int((~mask).sum())),
                  "cross_seconds": time.time() - t0, "bound_tol": tol,
                  "label": "t_X = 2 ||Pi_X (C - P_-) Pi_Xc||_1 (EVOLVEDCUT EC1: |S_X(C) - S_X(P_-)| <= t_X <= ||C - P_-||_1 for a pure Gaussian C and a projector P_-) from the + sector's block"})
        arrays["sv_block_plus_cut%s" % lab] = sv
        if check:
            Km = Pm[mask] @ Pm[~mask].mH
            Km.addmm_(Vm[mask], Vm[~mask].mH, alpha=-1.0)
            svm = torch.linalg.svdvals(Km).cpu().numpy().astype(float); del Km
            d["t_X_minus"] = 2.0 * float(svm.sum()); d["pairing_defect_t_X"] = abs(d["t_X_minus"] - tplus)
            arrays["sv_block_minus_cut%s" % lab] = svm
        SX, SXv = d.get("S_X"), d.get("S_X_vac")
        if SX is not None and SXv is not None:
            d["abs_dS"] = abs(SX - SXv); d["slack"] = d["t_X"] - d["abs_dS"]; d["bound_ok"] = bool(d["slack"] >= -tol)
            if not d["bound_ok"]:
                fails.append(lab)
        else:
            d.update({"abs_dS": None, "slack": None, "bound_ok": None})
        log("D13 kappa %d T/r_s %.3f cross-cut %s r_s (sector): |S_X(C) - S_X(P_-)| %s <= t_X %.6e (slack %s; %d singular values of the + block, largest %.4e%s; %.1f s): %s%s" % (
            kappa, trs, lab, _f(d.get("abs_dS"), "%.6e"), d["t_X"], _f(d.get("slack"), "%.3e"), d["n_sv"], d["sigma_max_block"],
            (", - sector pairing defect %.1e" % d["pairing_defect_t_X"]) if check else "", d["cross_seconds"],
            "ok" if d.get("bound_ok") else ("FAIL" if d.get("bound_ok") is False else "no vacuum entropy"),
            ("; a negative slack within the tolerance is the state's impurity (the theorem's premise is a projector): the excess's floor %s" % _f(d.get("excess_floor_low"), "%.1e"))
            if (d.get("slack") is not None and d["slack"] < 0 and d.get("excess_floor_low") is not None) else ""))                   # v1c: the impurity named where the pure-state bound fails by less than the tolerance
    if cfg.get("cut_norm_full", "auto") != "off":
        tn = full_trace_norm_sector(Pp, Vp, cfg, float(cfg.get("cut_sigma_tol", 1e-10)))                                            # v1b: the route from cfg
        arrays["sv_full_plus"] = tn.pop("sv")
    else:
        tn = {"route": "off", "nu": None, "rank": None, "sigma_max": None, "frob": None, "label": "the full ||C - P_-||_1 not computed (--cut-norm-full off)"}
    tmax = max([ent[lab]["t_X"] for lab, _ in cuts if ent[lab].get("t_X") is not None], default=None)
    tn["max_t_X"] = tmax
    tn["nu_ge_max_t_X"] = bool(tn["nu"] + 1e-8 >= tmax) if (tn.get("nu") is not None and tmax is not None) else None
    sc["trace_norm"] = tn; sc["cross_cut_fails"] = fails; sc["premise_defects"] = None
    if tn.get("nu") is not None:
        log("D13 kappa %d T/r_s %.3f: ||C - P_-||_1 = %.6e (%s; indicative rounding bound %.1e; rank %d, largest singular value %.4e, ||.||_2 %.4e, n_pair %.6e; max t_X %s: %s; %.1f s)" % (   # v1b: the route's bound printed
            kappa, trs, tn["nu"], tn["route"], tn.get("rounding_bound_nu") or 0.0, tn["rank"], tn["sigma_max"], tn["frob"], tn["n_pair"], _f(tmax, "%.6e"),
            "consistent" if tn["nu_ge_max_t_X"] is not False else "WARNING: below a t_X", tn["seconds"]))
    del Pp, Pm, Vp, Vm
    if dev.type == "cuda":
        torch.cuda.empty_cache()


def sector_line(sec, ent):
    """one clause for the checkpoint line."""
    fl = sec.get("floor") or {}                                                                                                     # v1c
    fls = (" (impurity ||Phi^dagger Phi - 1||_1 %s against the a priori bound %s: %s; at the first checkpoint %s against the construction allowance %s)" % (
        _f(fl.get("T_meas"), "%.1e"), _f(fl.get("T_cheb"), "%.1e"), "within" if fl.get("premise_ok") else ("OUTSIDE" if fl.get("premise_ok") is False else "-"),
        _f(fl.get("T_init"), "%.1e"), _f(fl.get("T_cheb_construction"), "%.1e"))) if fl else ""                                   # v1d: the construction beside the bound
    if not sec.get("check_minus"):
        return "halved (+ sector; S_X = 2 S_Xc(C_+))" + fls                                                                           # v1c
    cs = sec.get("cuts", {})
    ds = max([c.get("pairing_defect_state", 0.0) for c in cs.values()], default=0.0)
    dv = max([c.get("pairing_defect_vacuum", 0.0) for c in cs.values()], default=0.0)
    pg = max([abs(c.get("purity_gap", 0.0)) for c in cs.values()], default=0.0)
    dt = max([d.get("pairing_defect_t_X", 0.0) for d in ent.values() if isinstance(d, dict)], default=0.0)
    return "halved, both sectors at this clock: pairing defect state %.1e (= |S_Xc(C_+) - S_Xc(P_C)|, the impurity's entropy), vacuum %.1e, purity gap %.1e, t_X %.1e%s" % (ds, dv, pg, dt, fls)   # v1c


def formed_energy(vac, Phi):
    """E_kappa = sum_{e > 0} e n_j + sum_{e < 0} |e| (1 - n_j) on the dense vacuum's levels: the + sector's e_+, n^+ = <v|C|v> and the - sector's -e_+,
    n^- = <conj v|C|conj v>; equal to Tr[h (C - P_-)] = E_inv for any C and to Tr[|h| (C - P_-)^2] for the pure state. Beside it the occupation
    pairing defect max|n^+ + n^- - 1| (C_- = conj(1 - C_+)) and the off-sector weight ||P_- C P_+||_F. None without the sector data."""
    Vp = vac.get("V_plus") if vac is not None else None
    if Vp is None:
        return None
    t0 = time.time()
    dev = Phi.device
    ep = torch.as_tensor(np.asarray(vac["e_plus"], float), dtype=RD, device=dev)
    Pp = _compact(to_sector(Phi, 1)); Pm = _compact(to_sector(Phi, -1))                                                              # v1b
    n_plus = (Vp.mH @ Pp).abs().pow(2).sum(dim=1)
    n_minus = (Vp.conj().mH @ Pm).abs().pow(2).sum(dim=1)
    pos = ep > 0; neg = ep < 0
    E_plus = float((ep[pos] * n_plus[pos]).sum() + (ep[neg].abs() * (1.0 - n_plus[neg])).sum())
    E_minus = float((ep[neg].abs() * n_minus[neg]).sum() + (ep[pos] * (1.0 - n_minus[pos])).sum())
    pair = (n_plus + n_minus - 1.0).abs()
    off = float(torch.linalg.norm(Pm @ Pp.mH))
    out = {"E_kappa": E_plus + E_minus, "E_kappa_plus": E_plus, "E_kappa_minus": E_minus, "occupation_pairing_defect_max": float(pair.max()),
           "occupation_pairing_defect_mean": float(pair.mean()), "off_sector_weight_F": off, "tr_C_plus": float(Pp.abs().pow(2).sum()), "tr_C_minus": float(Pm.abs().pow(2).sum()),
           "n_particles": float(n_plus[pos].sum() + n_minus[neg].sum()), "n_holes": float((1.0 - n_plus[neg]).sum() + (1.0 - n_minus[pos]).sum()), "seconds": time.time() - t0,
           "label": "E_kappa = Tr[|h| (C - P_-)^2] = Tr[h (C - P_-)] = E_inv (the pure state); spectral form on the sector eigendecomposition"}
    del Pp, Pm, n_plus, n_minus
    return out


def format_w1_line(kappa, trs, E_inv, fe, aud, fl=None):
    """v1c: fl (propagation_floor's dict) adds the impurity against the a priori bound; v1d: the construction term and the first checkpoint's impurity."""
    aud = aud or {}
    line = ("v6o W1 kappa %d T/r_s %.3f: E_kappa (spectral) %s against E_inv %s (residual %s); the banded spectrum's pairing residual Tr min(h,0) + Tr|h|/2 = %s "
            "(relative %s; gap min|e| %s); occupation pairing max|n^+ + n^- - 1| %s; off-sector weight %s; particles %s, holes %s" % (
                kappa, trs, _f((fe or {}).get("E_kappa"), "%.6e"), _f(E_inv, "%.6e"), _f(((fe or {}).get("E_kappa") - E_inv) if (fe and E_inv is not None) else None, "%.1e"),
                _f(aud.get("pairing_residual"), "%.2e"), _f(aud.get("pairing_residual_rel"), "%.1e"), _f(aud.get("abs_min"), "%.3e"),
                _f((fe or {}).get("occupation_pairing_defect_max"), "%.1e"), _f((fe or {}).get("off_sector_weight_F"), "%.1e"),
                _f((fe or {}).get("n_particles"), "%.4e"), _f((fe or {}).get("n_holes"), "%.4e")))
    if fl:                                                                                                                          # v1c
        line += ("; impurity ||Phi^dagger Phi - 1||_1 %s against the propagation's a priori bound %s (construction %s, truncation %s, rounding allowance %s; %s sub-steps, K max %s; "
                 "at the first checkpoint %s, the growth since %s): %s" % (                                                        # v1d: the construction term, T_init and the growth on the line
            _f(fl.get("T_meas"), "%.2e"), _f(fl.get("T_cheb"), "%.2e"), _f(fl.get("T_cheb_construction"), "%.2e"), _f(fl.get("T_cheb_truncation"), "%.2e"), _f(fl.get("T_cheb_rounding_allowance"), "%.2e"),
            _f(fl.get("n_substeps"), "%d"), _f(fl.get("cheb_K_max"), "%d"), _f(fl.get("T_init"), "%.2e"), _f(fl.get("T_growth"), "%.2e"),
            "within" if fl.get("premise_ok") else ("OUTSIDE (not the construction's and the propagation's drift alone)" if fl.get("premise_ok") is False else "-")))   # v1d
    return line


# ---------------------------------------------------------------------------------------------------- the field-built banded operator
class FastBanded(OPS.Banded):
    """page_ops_v6.Banded's object (the same apply) built from the fields without the CSR: the clocked massless chain's nonzero diagonals at the offsets
    +-1 (the on-site kappa N_j/R_j sigma_y and the sine bonds with the clock), +-3 (the bonds' other component) and +-4 (the two-step Q with the flow);
    the same products and sums as clocked_csr's, so the diagonals are bitwise OPS.Banded(clocked_csr(...))'s (fast_banded_matches verifies it before
    use); the Gershgorin bound summed over the diagonals (equal to the CSR's up to the summation order's rounding)."""

    def __init__(self, N, v, kappa, R, device):
        N = np.asarray(N, float); v = np.asarray(v, float); R = np.asarray(R, float)
        n = len(R); n2 = 2 * n; k = float(kappa)
        self.n2 = n2
        d1 = np.zeros(n2 - 1, complex)
        d1[0::2] = (-1j * k / R) * N
        d1[1::2] = -0.5j * (0.5 * (N[:-1] + N[1:]))
        d3 = np.zeros(n2 - 3, complex)
        d3[0::2] = -0.5j * (0.5 * (N[:-1] + N[1:]))
        v2 = np.repeat(v, 2)
        d4 = -0.25j * (0.5 * (v2[: n2 - 4] + v2[4:]))
        diags = [(-4, d4.conj()), (-3, d3.conj()), (-1, d1.conj()), (1, d1), (3, d3), (4, d4)]
        self.diags = [(d, torch.as_tensor(D, dtype=CD, device=device)) for d, D in diags if np.any(D != 0)]
        rs = np.zeros(n2)
        for d, D in diags:
            if d >= 0:
                rs[: n2 - d] += np.abs(D)
            else:
                rs[-d:] += np.abs(D)
        self.bound = float(rs.max())


def fast_banded_matches(ops, N, v, device):
    """the field-built operator against OPS.Banded(ops.h_csr(N, v)): every diagonal bitwise, the bound to 1e-12 relative."""
    ref = OPS.Banded(ops.h_csr(N, v), device)
    fb = FastBanded(N, v, ops.kappa, ops.R, device)
    rd = {d: D for d, D in ref.diags}; fd = {d: D for d, D in fb.diags}
    ok = True; worst = 0.0
    for d in sorted(set(rd) | set(fd)):
        if d not in rd or d not in fd:
            D = rd.get(d, fd.get(d))
            m = float(D.abs().max()) if D.numel() else 0.0
            if m > 0:
                ok = False; worst = max(worst, m)
            continue
        diff = float((rd[d] - fd[d]).abs().max()) if rd[d].numel() else 0.0
        worst = max(worst, diff)
        if diff > 0:
            ok = False
    bnd = abs(ref.bound - fb.bound) / max(ref.bound, 1e-300)
    return {"ok": bool(ok and bnd <= 1e-12), "max_diag_diff": worst, "bound_rel_diff": bnd, "offsets_ref": sorted(rd), "offsets_fast": sorted(fd), "bound_ref": ref.bound, "bound_fast": fb.bound}


# ----------------------------------------------------------------------------------------------------------- the sub-steps
def record_substeps(H, k, t0, t1, held, cubic_window, mode, n_sub):
    """[(t_a, t_b, label, (N, v))] for record k over [t0, t1): label None the record's fields (v6n's piecewise-constant propagation, the control; also
    a held record and the history's last record), 'cubic' v6m's Catmull-Rom window with 1-unit sub-steps (and mode 'cubic' everywhere), 'linear' the
    W1 production: n_sub sub-steps with the fields interpolated linearly between record k and k + 1, taken at the sub-step's midpoint."""
    if held or mode == "none" or k + 1 >= len(H):
        if (cubic_window is not None and cubic_window[0] <= t0 < cubic_window[1] and not held) or (mode == "cubic" and not held and k + 1 < len(H)):
            nsub = max(1, int(round(t1 - t0)))
            return [(t0 + i * (t1 - t0) / nsub, t0 + (i + 1) * (t1 - t0) / nsub, "cubic", H.cubic_fields(t0 + (i + 0.5) * (t1 - t0) / nsub)) for i in range(nsub)]
        return [(t0, t1, None, H.fields(k))]
    if mode == "cubic":
        nsub = max(1, int(round(t1 - t0)))
        return [(t0 + i * (t1 - t0) / nsub, t0 + (i + 1) * (t1 - t0) / nsub, "cubic", H.cubic_fields(t0 + (i + 0.5) * (t1 - t0) / nsub)) for i in range(nsub)]
    n = max(1, int(n_sub))
    N0, v0 = H.fields(k); N1, v1 = H.fields(k + 1)
    out = []
    for i in range(n):
        u = (i + 0.5) / n
        out.append((t0 + i * (t1 - t0) / n, t0 + (i + 1) * (t1 - t0) / n, "linear", (N0 + u * (N1 - N0), v0 + u * (v1 - v0))))
    return out


def propagation_label(mode, n_sub, cubic_window=None):
    if mode == "linear":
        return "the fields interpolated linearly between records in %d sub-steps of tau/%d, each at its midpoint (the W1 production: the records' jump replaced by a ramp)" % (int(n_sub), int(n_sub))
    if mode == "none":
        return "v6n's piecewise-constant fields over each record (the W1 control)" + ((", v6m's cubic window %s" % str(cubic_window)) if cubic_window else "")
    return "v6m's cubic (Catmull-Rom) interpolation everywhere in 1-unit sub-steps"


# ------------------------------------------------------------------------------------------------- the majorant and the rule
RULE_TEXT = ("W1 rule (defect 10): the weighted terms w_kappa E_kappa (w = 16 kappa the copies, E_kappa = Tr[|h| (C - P_-)^2] = E_inv) must fall over the last %d "
             "channels with largest successive ratio q < 1; the energy tail beyond K is bounded by w_K E_K q/(1 - q); the excess's tail by sum_j beta(K + j) w_K E_K q^j "
             "with beta the entropy per formed energy of the last channels (its power-law trend in kappa extrapolated where it rises); the accumulated collar "
             "flow's tail by its own geometric majorant; a sum is converged at K when its tail bound is at most %g of its partial sum (floor 1e-9) at every readout "
             "clock from %g r_s; the premises print with every verdict; never the last channel's share; the proved form of the majorant is the icloud lane's X3 "
             "and replaces page_w1_sector_v1.tail_majorant when it lands")


def geometric_tail(terms, n_fit=3, floor=0.0):
    """the geometric majorant of a nonnegative sequence's tail from its last n_fit terms: q the largest successive ratio (premise q < 1), the tail
    t_K q/(1 - q); a last term at or below the floor is converged with the tail 0."""
    t = [abs(float(x)) for x in terms]
    if len(t) < 2:
        return {"ok": False, "q": None, "tail": None, "reason": "fewer than 2 channels", "terms_used": t, "ratios": []}
    t = t[-int(n_fit):] if len(t) >= int(n_fit) else t
    if t[-1] <= floor:
        return {"ok": True, "q": 0.0, "tail": 0.0, "reason": "the last term %.3e is at or below the floor %.3e" % (t[-1], floor), "terms_used": t, "ratios": []}
    ratios = [(t[i + 1] / t[i]) if t[i] > 0 else float("inf") for i in range(len(t) - 1)]
    q = max(ratios)
    if not q < 1.0:
        return {"ok": False, "q": q, "tail": None, "reason": "the terms do not fall over the last %d channels (largest successive ratio %.4f)" % (len(t), q), "terms_used": t, "ratios": ratios}
    return {"ok": True, "q": q, "tail": t[-1] * q / (1.0 - q), "reason": "geometric majorant t_K q/(1 - q), q the largest successive ratio over the last %d channels" % len(t),
            "terms_used": t, "ratios": ratios}


def tail_majorant(kappas, energy_terms, excess_terms, n_fit=3, floor=0.0):
    """the excess's tail bound from the formed-energy majorant (the function to replace when X3's proved form lands)."""
    ge = geometric_tail(energy_terms, n_fit, floor)
    out = {"energy": ge, "ok": False, "tail_bound": None, "beta": None, "p": None, "q": ge.get("q")}
    if not ge["ok"]:
        out["reason"] = "no majorant: " + ge["reason"]
        return out
    if ge["tail"] == 0.0:
        out.update({"ok": True, "tail_bound": 0.0, "reason": "the formed energy of the last channels is at the floor: no excitation to carry entropy, the tail 0"})
        return out
    m = min(len(kappas), int(n_fit))
    kk = list(kappas)[-m:]; Ek = [abs(float(x)) for x in list(energy_terms)[-m:]]; Xk = [abs(float(x)) for x in list(excess_terms)[-m:]]
    betas = [(x / e) if e > floor else None for x, e in zip(Xk, Ek)]
    good = [(k, b) for k, b in zip(kk, betas) if b is not None and b > 0.0]
    p = 0.0
    if len(good) >= 2:
        lk = np.log([float(k) for k, _ in good]); lb = np.log([b for _, b in good])
        p = float(np.polyfit(lk, lb, 1)[0])
    beta_K = max([b for _, b in good], default=0.0)
    q = ge["q"]; K = float(kappas[-1]); E_K = Ek[-1]; p_eff = max(p, 0.0)
    tail = 0.0; j = 1
    while j <= 200000:
        term = beta_K * ((K + j) / K) ** p_eff * E_K * q ** j
        tail += term
        if term <= 1e-9 * max(tail, 1e-300):
            break
        j += 1
    out.update({"ok": True, "tail_bound": tail, "beta": betas, "beta_K": beta_K, "p": p, "n_terms": j,
                "reason": "tail <= sum_{j >= 1} beta(K + j) w_K E_K q^j, beta(kappa) = |excess_kappa|/E_kappa (entropy per formed energy; power-law trend p = %.2f over the last %d channels, extrapolated where it rises), q = %.4f" % (p, m, q)})
    return out


def convergence_rule(per, kappas, clocks, cut_labels, tol, d13_of, from_rs, n_fit=3, floor_rel=1e-12, floor_abs=1e-9):
    """per clock: the per-channel weighted terms (the rows' E_inv_weighted, the cuts' excess and raw S_X, the accumulated collar flow), the majorants
    and the verdicts. per[kappa] = {'rows': {t: row}, 'w': 16 kappa}; d13_of(kappa, t) -> the channel's D13 dict at the clock (with 'cuts'). v1c: the
    excess's propagation floors (excess_floor_high, excess_floor_low) are summed with the weights where the cut dicts hold them and print beside the
    excess partial sums; they decide nothing."""
    kappas = list(kappas); K = kappas[-1]
    per_clock = {}; not_converged = []
    for t in clocks:
        rows = [per[k]["rows"].get(t) or {} for k in kappas]
        E_terms = [r.get("E_inv_weighted") for r in rows]
        F_terms = [r.get("collar_flow_acc_weighted") for r in rows]
        entry = {"t_over_rs": t, "energy": None, "cuts": {}, "collar": None}
        Eabs = None
        if all(x is not None for x in E_terms):
            Eabs = [abs(float(x)) for x in E_terms]
            flo = floor_rel * max(sum(Eabs), 1e-300)
            ge = geometric_tail(Eabs, n_fit, flo)
            entry["energy"] = {"terms": [float(x) for x in E_terms], "partial": float(sum(float(x) for x in E_terms)), "premise": ge, "tail_bound": ge.get("tail"), "floor": flo}
        for lab in cut_labels:
            cuts = [((d13_of(k, t) or {}).get("cuts") or {}).get(lab) or {} for k in kappas]
            raw = [c.get("S_X") for c in cuts]; exc = [c.get("excess") for c in cuts]
            cc = {"excess_terms": None, "raw_terms": None, "converged": None}
            if all(x is not None for x in raw):
                rw = [per[k]["w"] * float(x) for k, x in zip(kappas, raw)]
                gr = geometric_tail(rw, n_fit, 0.0)
                cc.update({"raw_terms": rw, "raw_partial": float(sum(rw)), "raw_premise": gr,
                           "raw_verdict": ("no majorant (%s): the raw sum prints as a regulated table only" % gr["reason"]) if not gr["ok"]
                           else ("the raw terms fall (q %.3f) but the raw sum still prints as a regulated table only; the readout is the excess" % gr["q"])})
            if Eabs is not None and all(x is not None for x in exc):
                xw = [per[k]["w"] * float(x) for k, x in zip(kappas, exc)]
                tm = tail_majorant(kappas, Eabs, xw, n_fit, entry["energy"]["floor"])
                part = float(sum(xw)); tol_abs = tol * max(abs(part), floor_abs)
                conv = bool(tm["ok"] and tm["tail_bound"] is not None and tm["tail_bound"] <= tol_abs)
                cc.update({"excess_terms": xw, "excess_partial": part, "majorant": tm, "tail_bound": tm.get("tail_bound"), "converged": conv, "tol_abs": tol_abs})
                if t >= from_rs - 1e-9 and not conv:
                    not_converged.append({"t_over_rs": t, "quantity": "the excess at %s r_s" % lab, "tail_bound": tm.get("tail_bound"), "partial": part, "reason": tm.get("reason")})
            flh = [c.get("excess_floor_high") for c in cuts]; fll = [c.get("excess_floor_low") for c in cuts]                                     # v1c
            if all(x is not None for x in flh):                                                                                                  # v1c
                cc["floor_high_terms"] = [per[k]["w"] * float(x) for k, x in zip(kappas, flh)]; cc["floor_high_partial"] = float(sum(cc["floor_high_terms"]))   # v1c
            if all(x is not None for x in fll):                                                                                                  # v1c
                cc["floor_low_terms"] = [per[k]["w"] * float(x) for k, x in zip(kappas, fll)]; cc["floor_low_partial"] = float(sum(cc["floor_low_terms"]))     # v1c
            entry["cuts"][lab] = cc
        if all(x is not None for x in F_terms):
            fw = [abs(float(x)) for x in F_terms]; part = float(sum(float(x) for x in F_terms))
            gf = geometric_tail(fw, n_fit, floor_rel * max(sum(fw), 1e-300))
            tol_abs = tol * max(abs(part), floor_abs)
            conv = bool(gf["ok"] and gf["tail"] is not None and gf["tail"] <= tol_abs)
            entry["collar"] = {"terms": [float(x) for x in F_terms], "partial": part, "premise": gf, "tail_bound": gf.get("tail"), "converged": conv, "tol_abs": tol_abs,
                               "energy_tail_bound": entry["energy"]["tail_bound"] if entry["energy"] else None,
                               "note": "the accumulated collar flow's tail by its own geometric majorant; the energy tail is printed beside (a held channel emits at most its formed energy)"}
            if t >= from_rs - 1e-9 and not conv:
                not_converged.append({"t_over_rs": t, "quantity": "the accumulated collar flow", "tail_bound": gf.get("tail"), "partial": part, "reason": gf.get("reason")})
        per_clock[t] = entry
    extend = bool(not_converged)
    if not extend:
        line = "W1 rule at K = %d: the excess at the cuts %s r_s and the accumulated collar flow converge at every readout clock from %g r_s (every tail majorant at most %g of its partial sum)" % (
            K, ", ".join(str(x) for x in cut_labels), from_rs, tol)
    else:
        worst = sorted(not_converged, key=lambda z: -((z["tail_bound"] if z["tail_bound"] is not None else float("inf"))))[:3]
        line = "W1 rule at K = %d: NOT converged for %d (clock, quantity) pairs from %g r_s, e.g. %s" % (
            K, len(not_converged), from_rs, "; ".join("%s at T/r_s %.1f: tail bound %s against the partial sum %.4e (%s)" % (
                z["quantity"], z["t_over_rs"], _f(z["tail_bound"], "%.3e"), z["partial"], z["reason"]) for z in worst))
    return {"rule": RULE_TEXT % (n_fit, tol, from_rs), "tol": tol, "n_fit": n_fit, "K": K, "kappas": kappas, "clocks": list(clocks), "cut_labels": list(cut_labels), "from_rs": from_rs,
            "per_clock": per_clock, "not_converged": not_converged, "extend": extend, "verdict_line": line,
            "floor_label": "the excess's propagation floors (page_w1_sector_v1c.excess_floors: high = the a priori Chebyshev bound with the rounding allowance, low = the measured impurity's entropy) summed with the weights where the cuts hold them; they decide nothing (v1c)"}   # v1c


def convergence_markdown(w1):
    """the markdown of convergence_rule's result: the verdict table at the headline and readout clocks, the per-channel terms. v1c: the excess's
    propagation floor (high, weighted) beside every excess partial sum and per channel."""
    L = ["", "## W1: the convergence of the excess and the collar flow in kappa with the formed-energy majorant (defect 10)", "", w1["rule"], "", w1["verdict_line"], "", w1.get("floor_label", ""), ""]   # v1c
    clocks = w1["clocks"]; from_rs = w1["from_rs"]
    show = [t for t in clocks if t >= from_rs - 1e-9 or abs(t - round(t / 10.0) * 10.0) < 1e-6 or any(abs(t - z) < 1e-6 for z in (12.0, 14.0, 20.0, 30.0))]
    L += ["| T/r_s | quantity | partial sum (K = %d) | last term | energy terms' largest ratio q | energy tail bound | beta_K (entropy per formed energy) | p | tail bound | tolerance | propagation floor (high, weighted) | converged |" % w1["K"],   # v1c
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]                                                                     # v1c
    for t in show:
        e = w1["per_clock"].get(t) or {}
        en = e.get("energy") or {}
        if en:
            L.append("| %.3f | 16 kappa E_kappa (the majorant's terms) | %s | %s | %s | %s | - | - | %s | - | - | %s |" % (             # v1c
                t, _f(en.get("partial"), "%.6e"), _f(en["terms"][-1], "%.3e"), _f(en["premise"].get("q"), "%.4f"), _f(en.get("tail_bound"), "%.3e"), _f(en.get("tail_bound"), "%.3e"),
                "falls" if en["premise"]["ok"] else "does not fall"))
        for lab, c in (e.get("cuts") or {}).items():
            if c.get("excess_terms") is not None:
                tm = c["majorant"]
                L.append("| %.3f | excess at %s r_s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (                            # v1c
                    t, lab, _f(c.get("excess_partial"), "%.6e"), _f(c["excess_terms"][-1], "%.3e"), _f(tm.get("q"), "%.4f"), _f((tm.get("energy") or {}).get("tail"), "%.3e"),
                    _f(tm.get("beta_K"), "%.3e"), _f(tm.get("p"), "%.2f"), _f(c.get("tail_bound"), "%.3e"), _f(c.get("tol_abs"), "%.3e"), _f(c.get("floor_high_partial"), "%.2e"), "yes" if c["converged"] else "NO"))   # v1c
            if c.get("raw_terms") is not None:
                L.append("| %.3f | raw S_X at %s r_s (regulated, not the readout) | %s | %s | %s | - | - | - | - | - | - | %s |" % (     # v1c
                    t, lab, _f(c.get("raw_partial"), "%.6e"), _f(c["raw_terms"][-1], "%.3e"), _f(c["raw_premise"].get("q"), "%.4f"), c.get("raw_verdict")))
        co = e.get("collar")
        if co:
            L.append("| %.3f | the accumulated collar flow | %s | %s | %s | %s | - | - | %s | %s | - | %s |" % (                        # v1c
                t, _f(co.get("partial"), "%.6e"), _f(co["terms"][-1], "%.3e"), _f(co["premise"].get("q"), "%.4f"), _f(co.get("energy_tail_bound"), "%.3e"),
                _f(co.get("tail_bound"), "%.3e"), _f(co.get("tol_abs"), "%.3e"), "yes" if co["converged"] else "NO"))
    L += ["", "### the per-channel terms at the headline clocks: 16 kappa E_kappa, 16 kappa excess_kappa per cut, 16 kappa S_X,kappa (raw) per cut, the excess's propagation floor (high) per cut, the collar flow", ""]   # v1c
    heads = [t for t in show if abs(t - round(t / 20.0) * 20.0) < 1e-6 or abs(t - clocks[-1]) < 1e-6 or any(abs(t - z) < 1e-6 for z in (12.0, 30.0))]
    labs = w1["cut_labels"]
    for t in heads:
        e = w1["per_clock"].get(t) or {}
        en = e.get("energy") or {}
        if not en:
            continue
        L += ["T/r_s %.3f:" % t, "", "| kappa | 16 kappa E_kappa | " + " | ".join("excess %s" % lab for lab in labs) + " | " + " | ".join("raw %s" % lab for lab in labs) + " | " + " | ".join("floor %s" % lab for lab in labs) + " | collar flow acc |",   # v1c
              "|---|---|" + "---|" * (3 * len(labs) + 1)]                                                                           # v1c
        for i, k in enumerate(w1["kappas"]):
            ex = [(e["cuts"].get(lab) or {}).get("excess_terms") for lab in labs]
            rw = [(e["cuts"].get(lab) or {}).get("raw_terms") for lab in labs]
            fh = [(e["cuts"].get(lab) or {}).get("floor_high_terms") for lab in labs]                                               # v1c
            co = (e.get("collar") or {}).get("terms")
            L.append("| %d | %s | %s | %s | %s | %s |" % (k, _f(en["terms"][i], "%.6e"), " | ".join(_f(x[i], "%.6e") if x else "-" for x in ex), " | ".join(_f(x[i], "%.6e") if x else "-" for x in rw),   # v1c
                                                         " | ".join(_f(x[i], "%.2e") if x else "-" for x in fh), _f(co[i], "%.6e") if co else "-"))   # v1c
        L.append("")
    return L


def flat_control_rule(raw_terms, excess_terms, energy_terms, hard_terms, tol=0.01, n_fit=3):
    """defect 10's gate on the flat held vacuum at K = len(raw_terms): the retired rule (the last channel's hard collar share) against the new rule on the
    raw entropy (no majorant: fails) and on the excess (0, converged with the majorant 0)."""
    K = len(raw_terms)
    tot_raw = float(sum(raw_terms)); tot_hard = float(sum(hard_terms))
    last_hard = float(hard_terms[-1])
    old_hard_share = (abs(last_hard) / abs(tot_hard)) if tot_hard != 0 else (0.0 if last_hard == 0 else float("inf"))
    old_raw_share = abs(float(raw_terms[-1])) / abs(tot_raw) if tot_raw else float("inf")
    gr = geometric_tail(raw_terms, n_fit, 0.0)
    kap = list(range(1, K + 1))
    tm = tail_majorant(kap, energy_terms, excess_terms, n_fit, 1e-12 * max(sum(abs(float(x)) for x in energy_terms), 1e-300))
    part_exc = float(sum(excess_terms))
    return {"K": K, "retired_rule_hard_share": old_hard_share, "retired_rule_passes": bool(old_hard_share <= tol), "retired_rule_raw_share": old_raw_share,
            "retired_rule_raw_share_passes_1pct": bool(old_raw_share <= tol),
            "new_rule_raw": {"premise": gr, "converged": bool(gr["ok"] and gr["tail"] is not None and gr["tail"] <= tol * abs(tot_raw)), "partial": tot_raw},
            "new_rule_excess": {"majorant": tm, "partial": part_exc, "converged": bool(tm["ok"] and tm["tail_bound"] is not None and tm["tail_bound"] <= tol * max(abs(part_exc), 1e-9))},
            "label": "the flat held vacuum: E_kappa = 0 and the excess 0 in every channel; the raw sum's terms grow with kappa"}


def relabel_text(vals):
    """defect 12 (paper prohibition Y13): the tracker's smoke-test line, with the flat held vacuum's kappa 1 values at 16, 24, 30 and 48 r_s (v1d: 5 decimals,
    icloud X3's gate values)."""
    v = ", ".join("%.5f" % vals[x] for x in (16.0, 24.0, 30.0, 48.0) if x in vals)                                                  # v1d
    return ("RELABEL (defect 12, Y13): the smoke test's S_X values %s nats at 16, 24, 30 and 48 r_s are the flat held vacuum's raw kappa 1 exterior-cut entropy on the "
            "8192-site chain (the run had --d13-vacuum off and computed no excess); they are not an escaping entropy and print nowhere as one; the readout is the excess "
            "S_X(C) - S_X(P_-(h)) of page_history_v6o's tables, the raw entropy and the held vacuum's beside it as regulated columns" % v)
