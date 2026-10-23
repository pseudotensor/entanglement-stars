#!/usr/bin/env python
"""engine_s7_copyslot_v1.py (builder analyst; engine spec 37.4, S7 part 4, the lead's turn 45; the brief
lead/BRIEF_s7_copyslot_replay9_builder.md; inbox #31 S4): the exactly equal-source copy-slot code on the stored r_s/a 32 history.

37.4 (1), binding: "engine_s7_copyslot_v1 (new; part 4; v2 untouched): on the stored r_s/a 32 tau 8 history at part 3's T0 18.000 (r_in
11.3037, R_t 26.2874), Phi the occupied orbitals of the replayed state, Pi_X the projector onto the sites inside R_t(T0) (the core's sites
where H ell exceeds 0.1 excluded if the record's H is stored, the gap printed), the eigenvectors of Phi^dagger Pi_X Phi with eigenvalue at
least 1 - epsilon at epsilon 1e-8 and 1e-10 (the measured eigenvalues stored; a negative-Killing-energy orbital chosen where that band
exists; the energy histogram printed), f = Phi u; C_z = (P - F) tensor I_2 + F tensor |z><z| on one exact G_3 doublet of the copies (S_i^2 =
I, S_i S_j = i epsilon_ijk S_k, commuting with the source vertex and the stage operator: printed), the spectator compensator optional;
printed: C_z^2 - C_z, Tr_copy C_z - (2P - F) for z = 0, 1 and two intermediate orientations (roundoff), equal particle numbers, the zero
Slater overlap, the first-variation charge shift -<f|h|f>/E; the single column f propagated on the common history by v2's propagator and
stepping; per record p_cap(t) = ||Pi_{>R_t(t)} T(t) f||^2 and at the detectors r 32, 64, 128, 256, 512 (part 3's), the half-zone split of
the column beyond R_t(t) (s7_halfzone_check_v1's projector) as the branch readout, D_det of the copy-resolved detector printed with the
identity D_det = p_cap checked; P08's two-scalar decoder (p, n) with n by its adjoint pass and the coherent information (P08 secs. 7 and
9.2); the step control (tau 4 at the same T0, as part 3); no ell control exists at this size (labelled, as part 3); the archive per block as
RUN_PROTOCOL.md sec. 7. (2) The common history: step 1 on the stored (unencoded) history's fields, labelled so, the first-variation shift
printed; step 2, if engine_collapse_v17 can start from a stored state at T0, the encoded common history re-solved once from C_z on box GPU 2
(released for it) and the readout repeated; else step 1 stands labelled." Inbox #31 S4 adds: "Store C, P (continued) and P_bar = U P_in
U^dagger separately."

Notation. P08's P (the formed Slater projector) is this lane's C, the state; this lane's P is the continued reference R2 and P_bar the
propagated initial reference. So the code here is C_z = (C - F) (x) I_2 + F (x) |z><z| and the copy-trace check is Tr_copy C_z - (2C - F).
The two copies are two of the channel's 16 kappa radial copies (the engine's weight w = 16 kappa per channel): in the radial representation
every source vertex and the stage operator act as A (x) I_2 on them, so the doublet's algebra and commutators are exact zeros by
construction; the intertwiner to the microscopic G_3 of P08 sec. 1.1 is not part of the radial engine (P08 1.1's channel identification,
labelled CONDITIONAL there and here).

What it computes (every number in the outputs is printed by the script):
(1) The replay: engine_s7_twocopy_v2's, copied (the C-init re-prepared and compared with the stored N_cinit, v_cinit; every channel's
reference continued per record by engine_collapse_v17's continue_v17 at the record's applied fields; the state's columns (the sea and the
dust) and P_bar propagated by v2's step U exp(-i e tau) U^dagger of the record's eigendecomposition; the audit: the replayed C - P against
the stored eps_tot, J_tot at every record, the C - P_bar contrast; v2's source kinds).
(2) T0: --T0-over-rs (18.0, part 3's) or --match-json (the primary's T0, the step control). The trapping episode about T0 and the closing
clock as v2 prints them.
(3) The selection at T0, per kappa in --kappas: Phi = the state's columns [sea, dust] (their occupations printed; a column whose occupation
is not 1 would be excluded from Phi and printed); X = the sites with R < R_t(T0); the core exclusion: the record's per-site H is looked up
(the stored engine_collapse_v17 records carry no per-site H: their H_in is the inside Killing energy, a scalar; then no exclusion applies,
printed); L_X = Phi^dagger Pi_X Phi by the SVD of Phi's X rows (its nonzero eigenvalues stored); per epsilon in --eps-list (1e-8, 1e-10)
the retained subspace (lambda >= 1 - epsilon); the stage operator h_m(T0) (the record's applied fields; the operator that steps the state
from T0) compressed to it, its eigen-orbitals and energies (the band; the "negative-Killing-energy band" is the negative part); the chosen
orbital: the retained eigen-orbital with the largest negative compressed energy (the builder's rule, labelled: the cheapest hole of the
negative band, so the smallest common charge shift; if no compressed energy is negative, the lowest, labelled); f = Phi u, u = V_keep c;
its measured epsilon 1 - ||Pi_X f||^2, its weight below r_in and in [r_in, R_t(T0)], its spectral histogram over h_m(T0)'s eigenvalues
(bins of --hist-bin, the weights at e < 0 and e > 0); the retained subspace kept as a contrast (the maximum capture, (6)).
(4) The code at T0 (P08 3.1; dense, the two copies copy-major), per configuration: C_z for z = |0>, |1>, (|0> + |1>)/sqrt 2 and
cos(pi/8)|0> + e^{i pi/3} sin(pi/8)|1>: max |C_z^2 - C_z| beside the baseline's max |C^2 - C| (and their difference: C_z^2 - C_z =
(C^2 - C) (x) I_2 exactly when C F = F); max |Tr_copy C_z - (2C - F)|; each z's explicit Slater columns Y_z = [Phi_rest (x) |0>,
Phi_rest (x) |1>, f (x) |z>] (Phi_rest = Phi V_rest, V_rest the Householder complement of u) against C_z (max |Y_z Y_z^dagger - C_z|),
their copy-summed site sources (eps, J by the engine's densities) against each other and against 2 dens(C) - dens(F); Tr C_z against
2 Tr C - 1 (equal particle numbers between the codewords; the spectator compensator is optional and not used); |det(Y_a^dagger Y_b)|
(|0> against |1>: the zero Slater overlap; the others against |<a|b>|) with the smallest singular value; the doublet S_i = sigma_i (x) 1:
S_i^2 - I, S_i S_j - i eps_ijk S_k, [S_i, 1 (x) a_x(B)] for every site x and B = h, Q (the source vertices), [S_i, 1 (x) h_m(T0)] (the
stage operator), [S_i, (C - F) (x) 1] (the common background), and [S_i, C_z] (the label, nonzero) beside them; the first-variation charge
shift -<f|h_m(T0)|f>/E (P08 3.1's comparator with h the stage operator) and -sum_x eps_x(F)/E (the hole's unclocked energy), beside the
solved shift: the run's own solver (engine_collapse_v17.solve) on the member's total sources with the hole removed from one copy (the
codeword's copy-summed source 2C - F in place of 2C), (E_hole - E)/E with E the unreplaced total's solve.
(5) The step control (--match-npz the primary's npz, --match-json its json): the same T0; X by the same rule on its own record; the orbital
the primary's f projected onto the control's retained subspace at the same epsilon and normalized (the overlap with the primary's f, the
measured epsilon printed; the control's own-rule orbital printed beside it as a contrast).
(6) Per record from T0 (v2's stepping: f(t_{k+1}) = U_k exp(-i e_k tau) U_k^dagger f(t_k), the step that propagates the state), per
configuration: p(t) = ||Pi_D f(t)||^2 on the regions D: beyond R_t(t) (R > R_t(t), while the surface persists: p_cap), beyond R_t(T0) (the
site fixed at T0, every record), R >= r for r in --ladder-radii (part 3's detectors), the collar (R beyond the M1 cut); on v2's collar packet
detector (J3''s probes, both copies) ||F_D^dagger f(t)||^2; the fronts (the outermost site whose tail weight exceeds 1e-12, 1e-9, 1e-6,
1e-3).
The explicit route (the identity D_det = p_cap): per region, ||Delta C_D||_op with Delta C_D = Pi_D (C_0(t) - C_1(t)) Pi_D from the two
codewords' own copy blocks, C(t) = Phi(t) Phi(t)^dagger (the replay's propagated state) and C(t) - F(t) = Phi_rest(t) Phi_rest(t)^dagger
(Phi_rest(t) = Phi(t) V_rest; neither uses the separately propagated column): the copy-1 block (the copy-2 block is its negative), its
largest-magnitude eigenvalues by randomized subspace iteration on the operator (no structure assumed; the second eigenvalue printed as the
rank check), beside p and the many-body bracket p <= D <= min(1, 2p) (P08 (17)); on the packet detector the dense small matrix; a dense
eigvalsh of the formed copy-1 block beyond R_t(T0) at T0's next record and at the end (the third route).
The half-zone split of the column (version 3's zone_masks and halfzone, copied verbatim; s7_halfzone_check_v1's projector): beyond R_t(t)
while the surface persists (the branch readout), beyond R_t(T0) and over the whole chain at every record: p_lo, p_hi (|a p| below, and at
or above, pi/2), the cross term sqrt(p_lo p_hi), the quarter shares, E[cos a p], E[cos 2 a p], Parseval; version 3's halfzone() on
G = (i/2) x x^dagger (its dense eigen route) at T0's next record and at the end as the check.
The decoder (P08 7.1, 7.2, 9.2, re-derived in decoder()'s docstring): per region with p above --decoder-floor, the matched mode
b = Pi_D f(t)/sqrt(p); n by the adjoint pass b_0 = W(t)^dagger b, W(t) = U(t, T0) the accumulated product of the same steps (no extraction:
the recursion (36) reduces to b <- U_k^dagger b), n = ||c - u (u^dagger c)||^2 with c = Phi(T0)^dagger b_0 (9.2's stable form for the hole
code, P_bg = C - F); the forward route n = <b|C(t)|b> - p as its check; q, lambda, chi_dec, I_c, the entanglement-breaking flag, the binary
information lower bound 1 - h_2((1 - p)/2), the tail-certified interior-transfer lower bound ((sqrt p - sqrt eps)_+)^2/(1 - eps) (P08 (8),
eps the measured one); the worst-background threshold p* (I_c at n = (1 - p)/2 crossing zero) solved by bisection and printed.
The maximum capture over the retained subspace (P08 8.1, a contrast): the largest eigenvalue of (Pi_D W_r(t))^dagger (Pi_D W_r(t)).
The common source (the hole): the hole's site sources dens(h, f(t)), dens(Q, f(t)) against the total source, their sums; the solved common
charge shift every --solve-every-rs r_s, at T0 and at the end; the propagation checks ||f(t)|| - 1 and ||f(t) - W(t) f(T0)||.
The correlation comparator (RUN_PROTOCOL 7), per kappa: K = Pi_X C(t) Pi_Y with X the sites inside R_t(t) (R_t(T0) after the closing) and Y
the rest, ||K||_op^2 = max lambda (1 - lambda) over the eigenvalues lambda of Pi_X C Pi_X (C is a projector: checked at T0), the Gaussian
I(X:Y) = 2 S(X) (nats), the lower bound 2 ||K||_op^2 (P08 (21)); the same for the record's Killing vacuum (the negative-energy projector of
h_m(t)); the direct svd of K at T0 and the end as the check.
(7) The archive (RUN_PROTOCOL 7; no extraction, so a record's step is the block): the json carries every per-record value; the npz at T0 and
at the end the state's columns (C), the continued reference P and P_bar per configured kappa, separately (inbox #31 S4), each
configuration's f, u, the reflector, the retained subspace's coefficients and eigenvalues, and every record's f(t).
(8) Compose: the md and json from both members' jsons (the primary; the step control at matched clocks).
Labels: step 1, the stored unencoded history's fields (37.4 (2)); step 2 not run: engine_collapse_v17's run() starts only from
prepare_dust at T = 0 (no resume or stored-state argument, no np.load), so 37.4 (2)'s condition fails and step 1 stands labelled; the ell
control: none at this size (as part 3); not printed until owner decision D11 (ITEM7_s7_part3_sentence_v2 (d), FILE 1's label meanwhile);
36.3 (2)'s half-zone reading withdrawn as a result (37.0 (vii)), its split reused here.

Args: --member dir/tag --T0-over-rs --match-json --match-npz --kappas --eps-list --ladder-radii --omega-over-kappa --mp-over-kappa --accept
--solve-every-rs --decoder-floor --hist-bin --gpu --threads --records --out-json --out-npz; --compose a.json[,b.json] --md --out-json
--md-every-rs."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "16")
import argparse
import hashlib
import json
import math
import sys
import time

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(1, os.path.join(HERE, "..", "flow"))
import engine_collapse_v5 as C5  # noqa: E402
import engine_collapse_v17 as E17  # noqa: E402
import flow_chart_v4 as V4  # noqa: E402

EPS_FRONT = (1e-12, 1e-9, 1e-6, 1e-3)
EPS_CROSS = (1e-3, 1e-6)
K_P = "P (the record's continued reference)"
K_PBAR = "P_bar (the propagated initial reference)"
K_PP = "P (+) P (the record's continued reference in both copies: each codeword's reference, the model's rule)"
SOURCE_KINDS = (K_P, K_PP)
ZS = (("|0>", (1.0, 0.0)), ("|1>", (0.0, 1.0)), ("(|0> + |1>)/sqrt 2", (1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0))),
      ("cos(pi/8)|0> + e^{i pi/3} sin(pi/8)|1>", (math.cos(math.pi / 8.0), complex(math.cos(math.pi / 3.0), math.sin(math.pi / 3.0)) * math.sin(math.pi / 8.0))))
STEP1 = ("step 1 (37.4 (2)): the stored unencoded history's fields (N_applied, v_applied of the member); the codewords' common source "
         "differs from the unencoded one by the hole, whose first-variation and solved charge shifts are printed; not the encoded common history")
STEP2 = ("step 2 not run: engine_collapse_v17's run() starts only from prepare_dust at T = 0 (no resume or stored-state argument, no np.load "
         "in the engine), so 37.4 (2)'s condition (v17 starting from a stored state at T0) fails and step 1 stands labelled")
ELL_LABEL = "the ell control: none exists at this size (no r_s 32 ell 8 member on this lane's v17 set: part 3's finding); none runs"
D11_LABEL = ("not printed until owner decision D11 (ITEM7_s7_part3_sentence_v2 (d): an exactly equal-source statement prints only after D11; "
             "FILE 1's label meanwhile); read in turn 46")
DOUBLET_LABEL = ("the doublet: two of the channel's 16 kappa radial copies; in the radial representation every source vertex and the stage "
                 "operator act as A (x) I_2 on them, so the algebra and commutators below are exact zeros by construction; the intertwiner to "
                 "the microscopic G_3 (P08 1.1) is not part of the radial engine (P08's conditional channel identification)")


class Audit:
    """every source the script computes passes here: the reference's columns are an explicit argument, and the call is recorded with its
    label, its role ("source", or "contrast" which sources nothing) and which registered object the reference was (version 2's class, the
    codewords' kind in place of the preparations')."""

    def __init__(self):
        self.calls = {}

    def subtract(self, ch, Cpart, Pcols, label, role, known, mult=1.0, twocopy=False):
        """eps_x, J_x of C - mult * P_ref: Cpart the densities of C's columns, the reference's (single-copy, or two-copy and copy-summed)
        subtracted here; known = [(object, kind), ...], the first object identical to Pcols names the kind."""
        k_ = next((kind for obj, kind in known if obj is Pcols), "other")
        key = (label, role, k_)
        self.calls[key] = self.calls.get(key, 0) + 1
        pe, pj = cdens2(ch, Pcols, None) if twocopy else (C5.dens(ch.h, Pcols), C5.dens(ch.Q, Pcols))
        return Cpart[0] - mult * pe, Cpart[1] - mult * pj

    def table(self):
        rows, ok = [], True
        for (label, role, k_), c in sorted(self.calls.items()):
            good = (k_ in SOURCE_KINDS) if role == "source" else True
            ok = ok and good
            rows.append({"label": label, "role": role, "reference": k_, "calls": c,
                         "verdict": (("C - P" if k_ == K_P else "C_z - P (+) P") if good else "FAIL") if role == "source" else "contrast (sources nothing)"})
        return rows, ok


def cdens(ch, cols):
    """the densities of C's columns: sum over (columns, f or None) of the engine's site densities of h and Q (version 2's)."""
    e = np.zeros(ch.hs.shape[0] // 2); j = np.zeros_like(e)
    for X, f in cols:
        e = e + C5.dens(ch.h, X, f); j = j + C5.dens(ch.Q, X, f)
    return e, j


def cdens2(ch, Y, f2):
    """the copy-summed densities of the two-copy columns Y (4n rows, copy-major): h and Q acting within each copy (version 2's)."""
    n2 = Y.shape[0] // 2
    e1, j1 = cdens(ch, [(Y[:n2], f2)]); e2, j2 = cdens(ch, [(Y[n2:], f2)])
    return e1 + e2, j1 + j2


def rel(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    return float(np.max(np.abs(x - y))) / max(float(np.max(np.abs(y))), 1e-300)


def projdist(A, B):
    """||A A^dagger - B B^dagger||_F for orthonormal column sets A, B (version 2's)."""
    ov = float((A.conj().T @ B).abs().pow(2).sum())
    return math.sqrt(max(A.shape[1] + B.shape[1] - 2.0 * ov, 0.0))


def site_beyond(R, r):
    """the index of the first site whose radius exceeds r (the region R > r begins there; len(R) if none) (version 2's)."""
    return int(np.searchsorted(np.asarray(R, float), float(r), side="right"))


def build_detector(n, kappa_ch, rs_nom, okap, mpkap, lo):
    """J3''s probe packets (obs_raymap_solved_v5's construction) centred at the collar (site indices lo .. n - 1), restricted to its sites
    and renormalized; the orthonormal span F (2n x nd), the restricted packets W (2n x np) and the record (version 2's, verbatim)."""
    kp = 1.0 / (2.0 * rs_nom)
    xD = 0.5 * ((lo + 1) + n)                      # the collar's centre radius (R = index + 1)
    xi = int(round(xD - 1))
    vext = -math.sqrt(rs_nom / xD)
    a_ = float(kappa_ch) / xD
    mask = np.zeros(2 * n, bool); mask[2 * lo:] = True
    rows, W = [], []
    for mk in mpkap:
        mp = mk * kp
        for r_ in okap:
            omega = r_ * kp; sg = omega / 6.0
            if mp > 0 and omega - 3 * sg < mp * (1 - 1e-9):
                rows.append({"mp_over_kappa": mk, "omega_over_kappa": r_, "skipped": "the packet's 3 sigma window reaches below the threshold m_p"})
                continue
            me = math.sqrt(mp * mp + a_ * a_)
            k, om, a2 = V4.branch_momenta(omega, sg, me, vext, outgoing=True)
            w = V4.build_packet(n, xi, k, a2, me, m=mp, a=a_)
            dk = np.gradient(k)
            kc = float(np.sum(a2 * k * dk) / np.sum(a2 * dk))
            vg = float(np.sum(a2 * V4.group_velocity(k, me, vext) * dk) / np.sum(a2 * dk))
            wr = np.where(mask, w, 0.0)
            ret = float(np.real(np.vdot(wr, wr)))
            wr = wr / math.sqrt(ret)
            rows.append({"mp_over_kappa": mk, "omega_over_kappa": r_, "omega": omega, "m_p": mp, "sigma_omega": sg, "k_c": kc, "v_g": vg,
                         "s_x_density": abs(vg) / (2 * sg), "retained_norm_in_collar": ret})
            W.append(wr)
    Wm = np.array(W).T
    ev = np.linalg.eigvalsh(Wm.conj().T @ Wm)
    keep = ev > 1e-10 * ev.max()
    # the orthonormal basis by column-pivoted Householder QR of the collar's rows alone, embedded with exact zeros elsewhere (a QR of the
    # full column would leave rounding-level weight on the chain's first rows, where Delta C is of order theta)
    Qc, Rc, piv = sla.qr(Wm[2 * lo:], mode="economic", pivoting=True)
    F = np.zeros((2 * n, int(keep.sum())), complex)
    F[2 * lo:] = Qc[:, :int(keep.sum())]
    info = {"kappa_p": kp, "x_D": xD, "x_index": xi, "v_ext": vext, "a": a_, "collar_sites": [lo + 1, n], "packets": rows,
            "gram_eigenvalues": [float(x) for x in ev], "n_detector": int(keep.sum()), "orthonormality": float(np.max(np.abs(F.conj().T @ F - np.eye(int(keep.sum())))))}
    return F, Wm, info


# ------------------------------------------------------------------------------------------------------------------ version 3's split (verbatim)
def zone_masks(M):
    """the signed index q_s of each output of an M-point DFT (numpy's fftfreq order; a p = 2 pi q_s / M in (-pi, pi]) and the zones by integer
    arithmetic on |q_s|: the half-zones |a p| < pi/2 (4 |q_s| < M) and |a p| >= pi/2 (the boundary |a p| = pi/2, 4 |q_s| = M, joins the upper
    half-zone: the builder's convention); the quarter-zones [0, pi/4), [pi/4, pi/2), [pi/2, 3 pi/4], (3 pi/4, pi] (vf10's boundaries pi/4 and
    3 pi/4 joining the middle bands); the count of boundary momenta at |a p| = pi/2."""
    qs = np.rint(np.fft.fftfreq(M) * M).astype(np.int64)
    aq = np.abs(qs)
    lo = 4 * aq < M
    quarters = [8 * aq < M, (8 * aq >= M) & (4 * aq < M), (4 * aq >= M) & (8 * aq <= 3 * M), 8 * aq > 3 * M]
    return qs, lo, ~lo, quarters, int(np.sum(4 * aq == M))


def halfzone(G, i0, cross=True, unsplit=None):
    """36.3 (2)'s split of the copy-off-diagonal difference G on the sites i0 .. n-1 (M sites, two spinor components per site, site-major):
    G_hat = F G_D F^dagger with the chain's DFT on the M sites per spinor component (F[q, j] = exp(-2 pi i q j/M)/sqrt(M): torch's fft along
    the row sites, ifft along the column sites, both orthonormal); the exact bounds 2 ||Pi G_D Pi||_op on the half-zones |a p| < pi/2 and
    >= pi/2 (the largest |eigenvalue| of the Hermitian i G_hat block); the contrasts: the cross bound 2 ||Pi_lo G_D Pi_hi||_op (cross),
    the one-sided momentum weight's shares (half- and quarter-zones), the local moments E[cos a p], E[cos 2 a p] (site overlaps without
    wrap-around), Parseval's identity, and against the unsplit bound (unsplit, if given) the compression check. G is read, never written."""
    torch = C5.BK["torch"]
    n = G.shape[0] // 2
    M = n - int(i0)
    if M <= 0:
        return None
    dev = G.device
    blk = G[2 * i0:, 2 * i0:].reshape(M, 2, M, 2)
    Gk = torch.fft.ifft(torch.fft.fft(blk, dim=0, norm="ortho"), dim=2, norm="ortho")
    qs, lo, hi, quarters, nb = zone_masks(M)
    il = torch.as_tensor(np.nonzero(lo)[0], dtype=torch.int64, device=dev)
    ih = torch.as_tensor(np.nonzero(hi)[0], dtype=torch.int64, device=dev)

    def sub(r, c):
        return Gk.index_select(0, r).index_select(2, c).reshape(2 * len(r), 2 * len(c))

    def herm_bound(idx):
        if len(idx) == 0:
            return 0.0
        return 2.0 * float(torch.linalg.eigvalsh(1j * sub(idx, idx)).abs().max())
    b_lo = herm_bound(il); b_hi = herm_bound(ih)
    b_x = (2.0 * float(torch.linalg.svdvals(sub(il, ih)).max()) if (len(il) and len(ih)) else 0.0) if cross else None
    w = (Gk.abs() ** 2).sum(dim=(1, 2, 3)).cpu().numpy()
    tot_k = float(w.sum()); tot_x = float((blk.abs() ** 2).sum())
    res = {"i0": int(i0), "M": M, "boundary_momenta": nb, "n_lo": int(lo.sum()), "n_hi": int(hi.sum()), "bound_lo": b_lo, "bound_hi": b_hi,
           "bound_cross": b_x, "fro2": tot_x, "parseval": (abs(tot_k - tot_x) / tot_x) if tot_x > 0 else None,
           "share_lo": None, "share_hi": None, "quarter_shares": None, "share_middle": None, "cos_ap": None, "cos_2ap": None}
    if tot_x > 0:
        res["share_lo"] = float(w[lo].sum()) / tot_k; res["share_hi"] = float(w[hi].sum()) / tot_k
        res["quarter_shares"] = [float(w[qm].sum()) / tot_k for qm in quarters]
        res["share_middle"] = res["quarter_shares"][1] + res["quarter_shares"][2]
        res["cos_ap"] = float((blk[1:].conj() * blk[:-1]).real.sum()) / tot_x
        res["cos_2ap"] = float((blk[2:].conj() * blk[:-2]).real.sum()) / tot_x if M > 2 else None
    if unsplit is not None:
        res["unsplit"] = unsplit
        res["check_lower"] = max(b_lo, b_hi, b_x or 0.0) - unsplit
        res["check_upper"] = (unsplit - (max(b_lo, b_hi) + b_x)) if b_x is not None else None
    return res


# ------------------------------------------------------------------------------------------------------------------ part 4's helpers
def fmt(x, f="%.3e"):
    return (f % x) if isinstance(x, (int, float)) and not isinstance(x, bool) and x is not None and x == x else "-"


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def npy(t):
    """a torch tensor as a numpy array (a conjugate view materialized first)."""
    return t.resolve_conj().cpu().numpy()


def utc():
    return time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())


def h2(x):
    """the binary entropy in bits."""
    x = min(max(float(x), 0.0), 1.0)
    if x <= 0.0 or x >= 1.0:
        return 0.0
    return -(x * math.log2(x) + (1.0 - x) * math.log2(1.0 - x))


def H4(ps):
    return -sum(p * math.log2(p) for p in ps if p > 0)


def decoder(p, n):
    """P08 7.1 and 7.2, re-derived: two rails (the matched detector mode d in each copy) of the code C_z, z the logical state; after the
    common transfer T the rails' covariance is n I_2 + p R (R = |z><z|; n = <d|C_bg(t)|d> the common background, p = |<d|T f>|^2); the
    reduced two-mode Gaussian state is a product over its eigenmodes (occupations n + p along z, n across), so the vacuum has probability
    p0 = (1 - n)(1 - n - p), the double occupation p2 = n (n + p), the one-particle block b I + p R with b = n (1 - n - p): linear in R, the
    channel. q = p + 2b (p0 + p2 + q = 1), lambda = p/q (the one-particle flag's depolarizing parameter). Two codewords: decoded trace
    distance p; Holevo (commuting outputs) chi_dec = q [(1 + lambda)/2 log2 (1 + lambda) + (1 - lambda)/2 log2 (1 - lambda)] =
    q [1 - h_2((1 + lambda)/2)]. Bell input: the one-particle Choi block (b/2) I_4 + p Phi+, eigenvalues q r0, q r1 (x3) with
    r0 = (1 + 3 lambda)/4, r1 = (1 - lambda)/4; the flags p0/2 (x2), p2/2 (x2); S(B) from diag(p0, q/2, q/2, p2); so
    I_c = S(B) - S(AB) = 2q - 1 - q H_4(r0, r1, r1, r1) (bits). Entanglement-breaking iff r0 <= 1/2, i.e. lambda <= 1/3 (p <= b). The
    binary-information lower bound from the covariance witness d = p: 1 - h_2((1 - p)/2) (P08 (19))."""
    b = n * (1.0 - n - p); q = p + 2.0 * b
    out = {"p": p, "n": n, "b": b, "q": q, "p0": (1.0 - n) * (1.0 - n - p), "p2": n * (n + p), "n_in_range": bool(-1e-12 <= n <= 1.0 - p + 1e-12),
           "I_acc_lower_bits": 1.0 - h2((1.0 - p) / 2.0)}
    if q <= 0:
        out.update({"lambda": None, "chi_dec_bits": 0.0, "Ic_bits": None, "entanglement_breaking": None})
        return out
    lam = min(p / q, 1.0)
    r0 = (1.0 + 3.0 * lam) / 4.0; r1 = (1.0 - lam) / 4.0
    out.update({"lambda": lam, "chi_dec_bits": q * (1.0 - h2((1.0 + lam) / 2.0)), "Ic_bits": 2.0 * q - 1.0 - q * H4([r0, r1, r1, r1]),
                "entanglement_breaking": bool(lam <= 1.0 / 3.0)})
    return out


def pstar():
    """the worst-background threshold: I_c(p, n = (1 - p)/2) = 0 by bisection on [0.5, 0.7] (P08 7.2's p*; the bracket's signs printed)."""
    f = lambda p: decoder(p, 0.5 * (1.0 - p))["Ic_bits"]
    lo, hi = 0.5, 0.7
    flo, fhi = f(lo), f(hi)
    if not (flo < 0 < fhi):
        return {"p_star": None, "f_lo": flo, "f_hi": fhi}
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(mid) < 0:
            lo = mid
        else:
            hi = mid
    return {"p_star": 0.5 * (lo + hi), "bracket": [lo, hi], "f_at_0.5": flo, "f_at_0.7": fhi, "I_c_at_p_star": f(0.5 * (lo + hi)),
            "nonEB_background_independent": 3.0 - 2.0 * math.sqrt(2.0)}


def tail_bound(p, eps):
    """P08 (8): f = sqrt(1 - eps) f_X + sqrt(eps) f_Y, the transfer a contraction: ||T f_X||^2 >= ((sqrt p - sqrt eps)_+)^2/(1 - eps)."""
    if p is None or eps is None or eps >= 1:
        return None
    return max(math.sqrt(max(p, 0.0)) - math.sqrt(max(eps, 0.0)), 0.0) ** 2 / (1.0 - eps)


def gauss_entropy(lam):
    lam = np.clip(np.asarray(lam, float), 0.0, 1.0)
    s = 0.0
    for x in lam:
        if 1e-300 < x < 1.0 - 1e-16:
            s -= x * math.log(x) + (1.0 - x) * math.log(1.0 - x)
    return s


def reflector(u):
    """the Householder reflector H = I - beta w w^dagger, w = u - alpha e_1, alpha = -e^{i arg u_0}, beta = 2/(w^dagger w) (u a unit vector:
    w^dagger u = 1 + |u_0| is real, so H u = alpha e_1 and H e_1 = u/alpha); H's columns 1 .. n-1 are an orthonormal basis of u's complement,
    V_rest = H[:, 1:], so Phi V_rest = Phi[:, 1:] - beta (Phi w) w[1:]^dagger. Returns (w, beta, alpha)."""
    torch = C5.BK["torch"]
    u0 = complex(u[0])
    ph = u0 / abs(u0) if abs(u0) > 0 else 1.0
    alpha = -ph
    w = u.clone()
    w[0] = w[0] - alpha
    beta = 2.0 / float((w.conj() @ w).real)
    return w, beta, alpha


def rest_cols(Phi, w, beta):
    """Phi V_rest (reflector()): Phi[:, 1:] - beta (Phi w) w[1:]^dagger."""
    return Phi[:, 1:] - beta * (Phi @ w)[:, None] * w[1:].conj()[None, :]


def subspace_top(mv, dim, k=6, iters=4, seed=20261016):
    """the largest-magnitude eigenvalues of a Hermitian operator given by its action mv(X) on columns: randomized subspace iteration (k
    columns, iters applications with re-orthonormalization, then Rayleigh-Ritz); no structure assumed (the explicit route's eigen-solve)."""
    torch = C5.BK["torch"]
    g = torch.Generator().manual_seed(seed)
    X = (torch.randn((dim, k), generator=g, dtype=torch.float64) + 1j * torch.randn((dim, k), generator=g, dtype=torch.float64)).to(torch.complex128)
    X = X.to(C5.BK["dev"])
    X, _ = torch.linalg.qr(X)
    for _ in range(iters):
        X, _ = torch.linalg.qr(mv(X))
    Hs = X.conj().T @ mv(X)
    Hs = 0.5 * (Hs + Hs.conj().T)
    ev = torch.linalg.eigvalsh(Hs).real.cpu().numpy()
    ev = ev[np.argsort(-np.abs(ev))]
    return [float(x) for x in ev]


def halfzone_col(xs):
    """the half-zone split of one column on M sites (xs: M x 2, site-major): version 3's DFT per spinor component (orthonormal, F[q, j] =
    exp(-2 pi i q j/M)/sqrt(M), torch's fft along the sites) and zone_masks; the column's weights on |a p| < pi/2 (p_lo) and >= pi/2 (p_hi)
    (for the rank-one difference x x^dagger (x) sigma_z these are ||Pi_lo Delta C_D Pi_lo||_op and ||Pi_hi Delta C_D Pi_hi||_op), the cross
    term sqrt(p_lo p_hi) (= ||Pi_lo x x^dagger Pi_hi||_op), the shares, the quarter shares, the local moments E[cos a p], E[cos 2 a p] (site
    overlaps without wrap-around, version 3's), Parseval."""
    torch = C5.BK["torch"]
    M = int(xs.shape[0])
    if M <= 0:
        return None
    xk = torch.fft.fft(xs, dim=0, norm="ortho")
    w = (xk.abs() ** 2).sum(dim=1).cpu().numpy()
    qs, lo, hi, quarters, nb = zone_masks(M)
    tot_x = float((xs.abs() ** 2).sum()); tot_k = float(w.sum())
    p_lo = float(w[lo].sum()); p_hi = float(w[hi].sum())
    res = {"M": M, "boundary_momenta": nb, "p": tot_x, "p_lo": p_lo, "p_hi": p_hi, "cross": math.sqrt(max(p_lo * p_hi, 0.0)),
           "parseval": (abs(tot_k - tot_x) / tot_x) if tot_x > 0 else None, "share_lo": None, "share_hi": None, "quarter_shares": None,
           "share_middle": None, "cos_ap": None, "cos_2ap": None}
    if tot_x > 0:
        res["share_lo"] = p_lo / tot_k; res["share_hi"] = p_hi / tot_k
        res["quarter_shares"] = [float(w[qm].sum()) / tot_k for qm in quarters]
        res["share_middle"] = res["quarter_shares"][1] + res["quarter_shares"][2]
        res["cos_ap"] = float((xs[1:].conj() * xs[:-1]).real.sum()) / tot_x
        res["cos_2ap"] = float((xs[2:].conj() * xs[:-2]).real.sum()) / tot_x if M > 2 else None
    return res


def spectral_hist(e, wts, binw):
    """the weights wts over the eigenvalues e in bins of width binw (edges at integer multiples), the bins holding weight above 1e-15; the
    totals at e < 0 and e > 0."""
    e = np.asarray(e, float); wts = np.asarray(wts, float)
    idx = np.floor(e / binw).astype(np.int64)
    bins = {}
    for i_, w_ in zip(idx, wts):
        bins[int(i_)] = bins.get(int(i_), 0.0) + float(w_)
    rows = [[k_ * binw, (k_ + 1) * binw, v_] for k_, v_ in sorted(bins.items()) if v_ > 1e-15]
    return {"bin": binw, "bins": rows, "weight_negative": float(wts[e < 0].sum()), "weight_positive": float(wts[e > 0].sum()),
            "weight_zero": float(wts[e == 0].sum()), "mean_energy": float(np.sum(e * wts) / max(np.sum(wts), 1e-300)),
            "energy_spread": float(math.sqrt(max(np.sum(e * e * wts) / max(np.sum(wts), 1e-300) - (np.sum(e * wts) / max(np.sum(wts), 1e-300)) ** 2, 0.0)))}


# ------------------------------------------------------------------------------------------------------------------ the selection and the code at T0
def select_orbitals(Phi, iX, hm, e0, U0, eps_list, R, rin, Rt0, hist_bin, f_match=None):
    """37.4 (1)'s selection on one channel at T0: L_X = Phi^dagger Pi_X Phi by the SVD of Phi's X rows (sites 0 .. iX-1); per epsilon the
    retained subspace (lambda >= 1 - epsilon), h_m(T0) compressed to it (its eigen-orbitals and energies, the band), the chosen orbital (the
    builder's rule: the largest negative compressed energy; none negative: the lowest; f_match given: the match projected onto the retained
    subspace and normalized), f = Phi u; f's measured epsilon, radial weights and spectral histogram over h_m(T0)'s eigenbasis (e0, U0)."""
    torch = C5.BK["torch"]
    n2 = Phi.shape[0]
    A = Phi[:2 * iX]
    _, s_, Vh = torch.linalg.svd(A, full_matrices=False)
    lam = (s_.real ** 2).cpu().numpy()
    order = np.argsort(-lam)
    lam = lam[order]; Vh = Vh[torch.as_tensor(order.copy(), dtype=torch.int64, device=Vh.device)]
    Rn = np.asarray(R, float)
    site_w = lambda x: (x.abs() ** 2).reshape(n2 // 2, 2).sum(dim=1).cpu().numpy()
    res = {"n_occ": int(Phi.shape[1]), "X_sites": int(iX), "X_last_R": float(Rn[iX - 1]) if iX > 0 else None, "rank_bound": int(2 * iX),
           "lambda": [float(x) for x in lam], "counts": {}, "configs": []}
    for eps in (1e-6, 1e-8, 1e-10, 1e-12):
        res["counts"]["%g" % eps] = int(np.sum(lam >= 1.0 - eps))
    for eps in eps_list:
        keep = np.nonzero(lam >= 1.0 - eps)[0]
        row = {"eps": eps, "retained": int(len(keep)), "lambda_kept": [float(lam[i]) for i in keep]}
        if len(keep) == 0:
            row["chosen"] = None
            row["why"] = "no eigenvalue of Phi^dagger Pi_X Phi at or above 1 - %g (the largest %.15f)" % (eps, float(lam[0]) if len(lam) else float("nan"))
            res["configs"].append(row)
            continue
        Vk = Vh[torch.as_tensor(keep, dtype=torch.int64, device=Vh.device)].conj().T.resolve_conj()          # n_occ x r (materialized)
        Wr = Phi @ Vk
        Hc = Wr.conj().T @ (hm @ Wr)
        Hc = 0.5 * (Hc + Hc.conj().T)
        ec, uc = torch.linalg.eigh(Hc)
        ec_np = ec.real.cpu().numpy()
        neg = np.nonzero(ec_np < 0)[0]
        own = int(neg[-1]) if len(neg) else 0
        row["band_energies"] = [float(x) for x in ec_np]
        row["band_negative"] = int(len(neg))
        row["own_rule"] = ("the retained eigen-orbital with the largest negative compressed energy (the builder's rule: the cheapest hole of the negative band)"
                           if len(neg) else "no negative compressed energy: the lowest (labelled)")
        row["own_index"] = own; row["own_energy"] = float(ec_np[own])
        if f_match is not None:
            cm = Wr.conj().T @ f_match
            nrm = float(torch.linalg.norm(cm))
            cvec = cm / nrm
            row["rule"] = "the primary's f projected onto this member's retained subspace and normalized (the step control's match)"
            row["match_overlap"] = nrm                                          # |<f_primary| P_retained |f_primary>|^(1/2)
            row["own_overlap_with_matched"] = float((uc[:, own].conj() @ cvec).abs())
        else:
            cvec = uc[:, own]
            row["rule"] = row["own_rule"]
        u = Vk @ cvec
        u = u / torch.linalg.norm(u)
        f = Phi @ u
        if f_match is not None:
            row["overlap_with_primary_f"] = float((f.conj() @ f_match).abs())
        wf = site_w(f)
        e0n = e0.cpu().numpy() if hasattr(e0, "cpu") else np.asarray(e0)
        spec_w = ((U0.conj().T @ f).abs() ** 2).cpu().numpy()
        row.update({"chosen": True, "f_norm": float(torch.linalg.norm(f)), "u_norm": float(torch.linalg.norm(u)),
                    "eps_measured": float(1.0 - wf[:iX].sum()), "weight_X": float(wf[:iX].sum()),
                    "weight_below_r_in": float(wf[Rn < float(rin)].sum()) if rin is not None else None,
                    "weight_annulus": float(wf[(Rn >= float(rin)) & (Rn < float(Rt0))].sum()) if rin is not None else None,
                    "weight_beyond_Rt0": float(wf[Rn > float(Rt0)].sum()), "centroid_R": float(np.sum(Rn * wf) / max(wf.sum(), 1e-300)),
                    "killing_energy": float((f.conj() @ (hm @ f)).real), "compressed_energy": float(ec_np[own]) if f_match is None else None,
                    "spectral": spectral_hist(e0n, spec_w, hist_bin), "spectral_sum": float(spec_w.sum()),
                    "radial_profile": [[float(Rn[i]), float(wf[i])] for i in range(min(len(wf), iX + 6))]})
        row["_f"] = f; row["_u"] = u; row["_Vk"] = Vk; row["_Wr"] = Wr
        res["configs"].append(row)
    return res


def comm_blocks(s, B):
    """max |[s (x) 1, C]| for a two-copy operator C given by its copy blocks B[a][b] (n2 x n2 each) and s a 2 x 2 copy matrix."""
    out = 0.0
    for a_ in range(2):
        for b_ in range(2):
            X = sum(s[a_][c_] * B[c_][b_] for c_ in range(2) if s[a_][c_] != 0) - sum(B[a_][c_] * s[c_][b_] for c_ in range(2) if s[c_][b_] != 0)
            if not isinstance(X, (int, float)):
                out = max(out, float(X.abs().max()))
    return out


def doublet_checks(ch, hm_host):
    """the doublet S_i = sigma_i (x) 1 on the two copies (copy-major) against P08 3.1 / RUN_PROTOCOL 3.1's list: S_i^2 = I, S_i S_j =
    i eps_ijk S_k, and the commutators with every source vertex 1 (x) a_x(B), a_x(B) = (Pi_x B + B Pi_x)/2 (B = h, Q; Pi_x the site's two
    components), and with the stage operator 1 (x) h_m(T0) (host sparse)."""
    n2 = ch.hs.shape[0]
    s1 = sp.csr_matrix(np.array([[0, 1], [1, 0]], complex)); s2 = sp.csr_matrix(np.array([[0, -1j], [1j, 0]], complex)); s3 = sp.csr_matrix(np.array([[1, 0], [0, -1]], complex))
    I_ = sp.identity(n2, dtype=complex, format="csr")
    S = [sp.kron(s_, I_, format="csr") for s_ in (s1, s2, s3)]
    I4 = sp.identity(2 * n2, dtype=complex, format="csr")
    mabs = lambda X: float(np.max(np.abs(X.data))) if X.nnz else 0.0
    sq = max(mabs(S[i] @ S[i] - I4) for i in range(3))
    eps3 = {(0, 1): (2, 1.0), (1, 0): (2, -1.0), (1, 2): (0, 1.0), (2, 1): (0, -1.0), (2, 0): (1, 1.0), (0, 2): (1, -1.0)}
    alg = max(mabs(S[i] @ S[j] - 1j * sg * S[k_]) for (i, j), (k_, sg) in eps3.items())
    vert = 0.0
    nsites = n2 // 2
    for B in (ch.hs.tocsr(), ch.Qs.tocsr()):
        for x in range(nsites):
            Pi = sp.csr_matrix((np.ones(2), (np.array([2 * x, 2 * x + 1]), np.array([2 * x, 2 * x + 1]))), shape=(n2, n2))
            ax = 0.5 * (Pi @ B + B @ Pi)
            K2 = sp.kron(sp.identity(2, dtype=complex, format="csr"), ax, format="csr")
            for i in range(3):
                vert = max(vert, mabs(S[i] @ K2 - K2 @ S[i]))
    H2 = sp.kron(sp.identity(2, dtype=complex, format="csr"), hm_host.astype(complex), format="csr")
    stage = max(mabs(S[i] @ H2 - H2 @ S[i]) for i in range(3))
    return {"S_squared_minus_I": sq, "S_i_S_j_minus_i_eps_S_k": alg, "source_vertices_max_commutator": vert, "source_vertices_checked": 2 * nsites,
            "stage_operator_commutator": stage, "label": DOUBLET_LABEL}


def code_checks(ch, Phi, f, u, w, beta):
    """P08 3.1's checks of C_z = (C - F) (x) I_2 + F (x) |z><z| at T0 (dense; the copies copy-major): per z in ZS max |C_z^2 - C_z|, against
    (C^2 - C) (x) I_2, max |Tr_copy C_z - (2C - F)|, Tr C_z - (2 Tr C - 1), the explicit Slater columns Y_z = [Phi_rest (x) |0>, Phi_rest (x)
    |1>, f (x) |z>] against C_z, their copy-summed sources (eps, J) against each other and against 2 dens(C) - dens(F), [S_i, C_z] (the label)
    and [S_i, (C - F) (x) 1] (the common background); |det(Y_a^dagger Y_b)| for the pairs (|0>, |1>) (the zero Slater overlap) and three
    others against |<a|b>|; the reflector's complement V_rest checked (V_rest^dagger u, V_rest^dagger V_rest - I)."""
    torch = C5.BK["torch"]
    dev = Phi.device
    n2 = Phi.shape[0]; no = Phi.shape[1]
    C = Phi @ Phi.conj().T
    F = f[:, None] @ f.conj()[None, :]
    base = C @ C - C
    out = {"baseline_idempotency": float(base.abs().max()), "C_f_minus_f": float((C @ f - f).abs().max()), "trace_C": float(torch.diagonal(C).real.sum()),
           "trace_F": float(torch.diagonal(F).real.sum()), "rows": [], "pairs": []}
    Prest = rest_cols(Phi, w, beta)
    Vr_u = float((Prest.conj().T @ f).abs().max())                              # = V_rest^dagger u (Phi orthonormal)
    G = Prest.conj().T @ Prest
    out["rest_orthonormality"] = float((G - torch.eye(no - 1, dtype=G.dtype, device=dev)).abs().max())
    out["rest_against_f"] = Vr_u
    del G
    e_C, j_C = cdens(ch, [(Phi, None)]); e_F, j_F = cdens(ch, [(f[:, None], None)])
    pred_e = 2.0 * e_C - e_F; pred_j = 2.0 * j_C - j_F
    sc_e = max(float(np.max(np.abs(pred_e))), 1e-300); sc_j = max(float(np.max(np.abs(pred_j))), 1e-300)
    s_ = {"1": [[0, 1], [1, 0]], "2": [[0, -1j], [1j, 0]], "3": [[1, 0], [0, -1]]}
    Ys = {}
    srcs = {}
    for name, (z0, z1) in ZS:
        z0 = complex(z0); z1 = complex(z1)
        B11 = C - F + (abs(z0) ** 2) * F; B22 = C - F + (abs(z1) ** 2) * F
        B12 = (z0 * z1.conjugate()) * F; B21 = (z1 * z0.conjugate()) * F
        Cz = torch.cat([torch.cat([B11, B12], 1), torch.cat([B21, B22], 1)], 0)
        Cz2 = Cz @ Cz - Cz
        idem = float(Cz2.abs().max())
        Cz2[:n2, :n2] -= base; Cz2[n2:, n2:] -= base
        idem_vs = float(Cz2.abs().max())
        del Cz2
        trc = float((B11 + B22 - (2.0 * C - F)).abs().max())
        trz = float(torch.diagonal(Cz).real.sum())
        Y = torch.zeros((2 * n2, 2 * (no - 1) + 1), dtype=Phi.dtype, device=dev)
        Y[:n2, :no - 1] = Prest; Y[n2:, no - 1:2 * (no - 1)] = Prest
        Y[:n2, -1] = z0 * f; Y[n2:, -1] = z1 * f
        yy = float((Y @ Y.conj().T - Cz).abs().max())
        es, js = cdens2(ch, Y, None)
        srcs[name] = (es, js)
        Ys[name] = Y
        lab = {i_: comm_blocks(s_[i_], [[B11, B12], [B21, B22]]) for i_ in s_}
        out["rows"].append({"z": name, "z0": [z0.real, z0.imag], "z1": [z1.real, z1.imag], "idempotency": idem, "idempotency_minus_baseline": idem_vs,
                            "trcopy_minus_2C_minus_F": trc, "trace": trz, "trace_minus_2trC_plus_1": trz - (2.0 * out["trace_C"] - 1.0),
                            "columns_vs_Cz": yy, "src_vs_pred_eps": float(np.max(np.abs(es - pred_e))) / sc_e, "src_vs_pred_J": float(np.max(np.abs(js - pred_j))) / sc_j,
                            "commutator_S_label": lab})
        del Cz, B11, B22, B12, B21
    Zb = torch.zeros_like(C)
    out["commutator_S_background"] = {i_: comm_blocks(s_[i_], [[C - F, Zb], [Zb, C - F]]) for i_ in s_}
    del Zb
    names = [nm for nm, _ in ZS]
    out["src_between_z_eps"] = max(float(np.max(np.abs(srcs[a_][0] - srcs[b_][0]))) for a_ in names for b_ in names) / sc_e
    out["src_between_z_J"] = max(float(np.max(np.abs(srcs[a_][1] - srcs[b_][1]))) for a_ in names for b_ in names) / sc_j
    out["src_scale_eps"] = sc_e; out["src_scale_J"] = sc_j
    zv = {nm: np.array([complex(z[0]), complex(z[1])]) for nm, z in ZS}
    for a_, b_ in ((names[0], names[1]), (names[0], names[2]), (names[1], names[3]), (names[2], names[3])):
        O = Ys[a_].conj().T @ Ys[b_]
        sign, logabs = torch.linalg.slogdet(O)
        smin = float(torch.linalg.svdvals(O).min())
        out["pairs"].append({"a": a_, "b": b_, "log10_abs_det": float(logabs) / math.log(10.0) if math.isfinite(float(logabs)) else None,
                             "abs_det": float(math.exp(float(logabs))) if math.isfinite(float(logabs)) else 0.0,
                             "abs_z_overlap": float(abs(np.vdot(zv[a_], zv[b_]))), "smallest_singular_value": smin})
        del O
    out["hole"] = {"sum_eps_F": float(np.sum(e_F)), "sum_J_F": float(np.sum(j_F)), "max_eps_F": float(np.max(np.abs(e_F))), "max_J_F": float(np.max(np.abs(j_F)))}
    del Ys, C, F, base, Prest
    return out, (e_F, j_F)


# ------------------------------------------------------------------------------------------------------------------ the member run
def member(a):
    C5.backend(a.gpu)
    torch = C5.BK["torch"]; dev = C5.BK["dev"]
    if a.gpu is None:
        torch.set_num_threads(a.threads)
    t_start = time.time()
    d, tag = os.path.split(a.member)
    jpath = os.path.join(d, "collapse_%s.json" % tag); zpath = os.path.join(d, "collapse_%s_fields.npz" % tag)
    J = json.load(open(jpath))
    Z = np.load(zpath, allow_pickle=True)
    if J.get("script") != "engine_collapse_v17.py":
        raise RuntimeError("the member %s was written by %s; this replay carries engine_collapse_v17's rule only" % (tag, J.get("script")))
    ra = J["args"]
    for key in ("collar_policy", "support_rel", "site_fields"):
        E17.OPT[key] = ra.get(key, E17.OPT[key])
    rs = float(ra["rs"]); tau = float(ra["tau"]); n = int(ra["nsites"]); m = float(ra["m"]); g = float(ra["g"]); ell = ra["ell"]
    chs = [C5.Chan(k, m, n, bool(ra.get("half_offset", 0))) for k in range(1, int(ra["kappa_sea"]) + 1)]
    byk = {ch.kappa: ch for ch in chs}
    R = chs[0].R
    shell = [float(q) * rs for q in ra["shell"].split(",")] if ra.get("shell") else None
    recs = {round(float(r["T"]), 6): r for r in J["records"] if r.get("E") is not None}
    Tz = np.asarray(Z["T"], float)
    K = len(Tz) if a.records is None else min(len(Tz), a.records)
    match = json.load(open(a.match_json)) if a.match_json else None
    mnpz = np.load(a.match_npz) if a.match_npz else None
    if match is not None:
        T0req = float(match["head"]["T0_over_rs"]); t0_rule = "--match-json %s (the primary's T0)" % os.path.basename(a.match_json)
    else:
        T0req = a.T0_over_rs; t0_rule = "--T0-over-rs %g (part 3's T0)" % a.T0_over_rs
    hit = [r for r in recs.values() if abs(float(r["T"]) / rs - T0req) < 1e-9]
    if not hit:
        raise RuntimeError("no record at T/rs %g" % T0req)
    rec0 = hit[0]
    T0 = float(rec0["T"])
    k0 = int(np.argmin(np.abs(Tz - T0)))
    if abs(Tz[k0] - T0) > 1e-9 or k0 >= K:
        raise RuntimeError("T0 %.3f is not a stored slice within the records replayed" % (T0 / rs))
    # the surface per slice (R_t held), the episode about T0 and the closing clock (version 2's)
    Rt_of = [recs.get(round(float(t), 6), {}).get("R_t") for t in Tz]
    rin_of = [recs.get(round(float(t), 6), {}).get("r_in") for t in Tz]
    no_record = [float(Tz[k]) / rs for k in range(len(Tz)) if round(float(Tz[k]), 6) not in recs]
    k_first = k0
    while k_first - 1 >= 0 and Rt_of[k_first - 1] is not None and Rt_of[k0] is not None:
        k_first -= 1
    k_close = next((k for k in range(k0, len(Tz)) if Rt_of[k] is None), None)
    k_last = (k_close - 1) if k_close is not None else len(Tz) - 1
    if Rt_of[k0] is None:
        raise RuntimeError("the record at T0 holds no trapping surface R_t")
    reappear = [float(Tz[k]) / rs for k in range(k_close + 1, len(Tz)) if Rt_of[k] is not None] if k_close is not None else []
    episode = {"surface_at_T0": True, "first_T_over_rs": float(Tz[k_first]) / rs, "last_T_over_rs": float(Tz[k_last]) / rs,
               "closing_T_over_rs": float(Tz[k_close]) / rs if k_close is not None else None, "k_close": k_close, "k_last": k_last,
               "closing_minus_T0_rs": (float(Tz[k_close]) - T0) / rs if k_close is not None else None,
               "last_minus_T0_rs": (float(Tz[k_last]) - T0) / rs, "reappearances_T_over_rs": reappear, "slices_without_record": no_record,
               "track": [{"T_over_rs": float(Tz[k]) / rs, "R_t": Rt_of[k], "r_in": rin_of[k]} for k in range(k0, (k_close + 1) if k_close is not None else len(Tz))]}
    Rt0 = float(Rt_of[k0]); rin0 = rec0.get("r_in")
    i_Rt0 = site_beyond(R, Rt0)
    # 37.4 (1): X = the sites inside R_t(T0), the core's sites where H ell > 0.1 excluded if the record stores a per-site H
    h_keys = [k_ for k_ in rec0.keys() if k_ == "H" or k_.startswith("H_") or k_.endswith("_H")]
    h_site = [k_ for k_ in h_keys if isinstance(rec0.get(k_), list) and len(rec0.get(k_)) == n]
    if h_site:
        Hs = np.asarray(rec0[h_site[0]], float)
        core = np.nonzero((np.abs(Hs) * float(ell) > a.h_ell_max) & (R < Rt0))[0]
        x_rule = "the sites with R < R_t(T0) = %.4f, the core's sites where |H| ell > %g excluded (the record's %s)" % (Rt0, a.h_ell_max, h_site[0])
        raise RuntimeError("a per-site H is stored (%s): the annulus selection is not implemented in version 1" % h_site[0])
    iX = i_Rt0
    x_info = {"rule": "the sites with R < R_t(T0) = %.4f (sites 1 .. %d); the record's keys named like H: %s, none a per-site array of length %d (H_in is the inside Killing energy, a scalar): no core exclusion (37.4 (1)'s condition)" % (
        Rt0, iX, h_keys or "none", n), "iX": iX, "R_last_in_X": float(R[iX - 1]), "h_keys": h_keys, "core_excluded": 0, "gap": None}
    kappas = [int(x) for x in a.kappas.split(",")]
    eps_list = [float(x) for x in a.eps_list.split(",")]
    ladder = [int(x) for x in a.ladder_radii.split(",")]
    i_lad = {r: int(np.searchsorted(R, float(r), side="left")) for r in ladder}            # the region R >= r begins at this site (part 3's)
    cuts = sorted({int(r["M1_parts"]["cut"]) for r in recs.values() if r.get("M1_parts") and r["M1_parts"].get("cut") is not None})
    cut = int(rec0["M1_parts"]["cut"])
    det = {}
    for kap in kappas:
        F, Wm, dinfo = build_detector(n, kap, rs, [float(x) for x in a.omega_over_kappa.split(",")], [float(x) for x in a.mp_over_kappa.split(",")], cut)
        det[kap] = {"F": C5.T_(F), "info": dinfo}
    prov = {"utc_start": utc(), "script": os.path.abspath(__file__), "script_md5": md5(os.path.abspath(__file__)),
            "engine_collapse_v17_md5": md5(E17.__file__), "engine_collapse_v5_md5": md5(C5.__file__), "engine_fields_v12_md5": md5(E17.FLD.__file__),
            "member_json_md5": md5(jpath), "member_fields_md5": md5(zpath), "argv": sys.argv, "host": os.uname().nodename,
            "labels": {"ma": m, "G_over_a2": g, "rs_over_a": rs, "four_pi_m_rs": 4.0 * math.pi * m * rs, "ell": ell, "site_fields": ra.get("site_fields"),
                       "reference": ra.get("reference"), "kernel": "the member's source averaging (site_fields %s, ell %s; engine_fields_v12)" % (ra.get("site_fields"), ell),
                       "member_label": J["records"][0].get("label") if J.get("records") else None},
            "step1": STEP1, "step2": STEP2, "ell_control": ELL_LABEL, "print_label": D11_LABEL, "doublet": DOUBLET_LABEL,
            "withdrawn": "36.3 (2)'s half-zone reading is withdrawn as a result (37.0 (vii)); its split (version 3's zone_masks, halfzone) is reused here"}
    head = {"script": "engine_s7_copyslot_v1.py", "member": a.member, "tag": tag, "member_script": J.get("script"), "rs": rs, "tau": tau, "nsites": n, "m": m, "g": g,
            "ell": ell, "kappa_sea": len(chs), "T0": T0, "T0_over_rs": T0 / rs, "T0_rule": t0_rule, "k0": k0, "records": K, "T_end_over_rs": float(Tz[K - 1]) / rs,
            "trapping_zeros_at_T0": {"r_in": rin0, "R_t": Rt0}, "E_stored_at_T0": rec0.get("E"), "episode": episode, "R_t0_site": float(R[i_Rt0]) if i_Rt0 < n else None,
            "X": x_info, "kappas": kappas, "eps_list": eps_list, "ladder_radii": ladder, "ladder_first_sites": {str(r): float(R[i_lad[r]]) for r in ladder},
            "m1_cuts": cuts, "cut": cut, "detector": {str(k): det[k]["info"] for k in kappas}, "accept": a.accept, "threads": a.threads, "gpu": a.gpu,
            "match_json": a.match_json, "match_npz": a.match_npz, "solve_every_rs": a.solve_every_rs, "decoder_floor": a.decoder_floor, "hist_bin": a.hist_bin,
            "p_star": pstar(), "provenance": prov}
    print("member %s (%s): r_s %g tau %g n %d; T0/rs %.3f (%s, k0 %d), %d records to T/rs %.3f; M1 cut %d; %s" % (
        tag, J.get("script"), rs, tau, n, T0 / rs, t0_rule, k0, K, Tz[K - 1] / rs, cut, STEP1), flush=True)
    print("the surface: held from T/rs %.3f to %.3f about T0, closing at %s (%s r_s after T0); reappearances %s; R_t(T0) %.4f (r_in %s), the first site beyond it R %s; X: %s" % (
        episode["first_T_over_rs"], episode["last_T_over_rs"], fmt(episode["closing_T_over_rs"], "%.3f"), fmt(episode["closing_minus_T0_rs"], "%.3f"), reappear,
        Rt0, fmt(rin0, "%.4f"), fmt(head["R_t0_site"], "%.0f"), x_info["rule"]), flush=True)
    ps_ = head["p_star"]
    print("P08 7.2's worst-background threshold by bisection: p* %s (bracket %s; I_c there %s); the background-independent non-entanglement-breaking bound 3 - 2 sqrt 2 = %.15f" % (
        fmt(ps_.get("p_star"), "%.15f"), ps_.get("bracket"), fmt(ps_.get("I_c_at_p_star"), "%.2e"), ps_["nonEB_background_independent"]), flush=True)
    for kap in kappas:
        di = det[kap]["info"]
        print("detector kappa %d: x_D %.2f, collar sites %s, %d packets, %d orthonormal modes (orthonormality %.1e)" % (
            kap, di["x_D"], di["collar_sites"], len(di["packets"]), di["n_detector"], di["orthonormality"]), flush=True)
    AUD = Audit()
    E17.tm_reset()
    t_p = time.time()
    Nc, vc, dust, info0, head0 = E17.prepare_dust(chs, g, R, ell, rs / (2.0 * g * m), float(ra["R0_over_rs"]) * rs, m, ra["reading"], shell)
    Ncs = np.asarray(Z["N_cinit"], float); vcs = np.asarray(Z["v_cinit"], float)
    head["cinit"] = {"dN": float(np.max(np.abs(Nc - Ncs))), "dv": float(np.max(np.abs(vc - vcs))), "passes": head0.get("passes"), "stored_passes": J["head"].get("passes"),
                     "N_B": head0.get("N_B"), "E0": head0.get("E0"), "stored_E0": J["head"].get("E0"), "seconds": time.time() - t_p}
    print("C-init re-prepared in %.0f s: %s passes (stored %s), max |N - N_cinit| %.2e, |v - v_cinit| %.2e, E0 %.10f (stored %.10f)" % (
        head["cinit"]["seconds"], head0.get("passes"), J["head"].get("passes"), head["cinit"]["dN"], head["cinit"]["dv"], head0.get("E0"), J["head"].get("E0")), flush=True)
    st = {}
    for ch in chs:
        e, U = E17.eigT(E17.clocked_gpu(ch, Ncs, vcs))
        neg = e < 0
        Phi_d, f_d = dust.get(ch.kappa, (torch.zeros((2 * n, 0), dtype=torch.complex128, device=dev), np.zeros(0)))
        en = e.real.cpu().numpy(); kk = int(neg.sum())
        gap = float(en[kk] - en[kk - 1]) if 0 < kk < len(en) else None
        st[ch.kappa] = {"sea": U[:, neg].clone(), "dust": Phi_d, "f": np.asarray(f_d, float), "ref": U[:, neg].clone(), "pbar": U[:, neg].clone(), "n_sea": kk,
                        "gap_between_prev": gap, "levels_prev": (np.asarray(en[:kk], float), np.asarray(en[kk:], float), 0)}
        del e, U
    f_ref = (Ncs, vcs)
    replay_rows, s7_rows, cfgs, states = [], [], [], {}
    colhist = {}
    Wacc, Phi0, keepd = {}, {}, {}
    out = {"head": head, "replay": replay_rows, "s7": s7_rows}

    def dump():
        rows_a, ok_a = AUD.table()
        out["audit"] = {"rows": rows_a, "pass": ok_a}
        out["seconds"] = time.time() - t_start
        json.dump(out, open(a.out_json, "w"), indent=1, default=float)

    def solve_total(eps, Jt):
        Na, va, W_, inf_ = E17.solve(eps, Jt, g, R, ell, None)
        return Na, va, (E17.FLD.charge(inf_, g) if not inf_.get("failed") else float("nan")), inf_.get("failed")

    def state_cols(kap):
        s_ = st[kap]
        return torch.cat([s_["sea"], s_["dust"][:, torch.as_tensor(keepd[kap], dtype=torch.int64, device=dev)]], dim=1) if s_["dust"].shape[1] else s_["sea"]

    def snapshot(label, cache_):
        snap = {}
        for kap in kappas:
            s_ = st[kap]
            for key in ("sea", "dust", "pbar"):
                snap["%s_k%d_%s" % (label, kap, key)] = npy(s_[key])
            snap["%s_k%d_ref" % (label, kap)] = npy(cache_[kap][2])
            snap["%s_k%d_f" % (label, kap)] = np.asarray(s_["f"], float)
        return snap

    def regions_of(i_Rt):
        rg = []
        if i_Rt is not None:
            rg.append(("beyond R_t(t)", i_Rt))
        rg.append(("beyond R_t(T0)", i_Rt0))
        rg += [("r %d" % r, i_lad[r]) for r in ladder]
        rg.append(("collar (beyond the M1 cut)", cut))
        return rg

    for k in range(K):
        tr = time.time()
        T = float(Tz[k]); rec = recs.get(round(T, 6), {})
        N_app = np.asarray(Z["N_applied"][k], float); v_app = np.asarray(Z["v_applied"][k], float)
        force = bool(rec.get("verification_differs"))
        t_e = time.time()
        eig = E17.eigh_all(chs, N_app, v_app)
        t_eig = time.time() - t_e
        cache, modes = {}, {}
        smodes = rec.get("continuation_modes") or {}
        mode_differs = []
        for ch in chs:
            e, U = eig.pop(ch.kappa)
            s_ = st[ch.kappa]
            ref, ci = E17.continue_v17(ch, s_["ref"], f_ref, (N_app, v_app), e, U, s_["n_sea"], s_.get("gap_between_prev"), s_.get("levels_prev"), force_subdivide=force)
            if ref is None:
                raise RuntimeError("the replay's continuation failed at T/rs %.3f, kappa %d (%s)" % (T / rs, ch.kappa, ci.get("why")))
            cache[ch.kappa] = (e, U, ref, ci)
            md_ = ci.get("mode", "?")
            modes[md_] = modes.get(md_, 0) + 1
            if str(ch.kappa) in smodes and str(smodes[str(ch.kappa)]).split(";")[0] != md_:
                mode_differs.append(ch.kappa)
        del eig
        # the audit's sources at this record, every channel: C - P (the run's sources) and the contrast C - P_bar (version 2's)
        eP = np.zeros(n); jP = np.zeros(n); eB = np.zeros(n); jB = np.zeros(n)
        Cpart = {}
        for ch in chs:
            e, U, ref, ci = cache[ch.kappa]; s_ = st[ch.kappa]
            known = [(ref, K_P), (s_["pbar"], K_PBAR)]
            Cpart[ch.kappa] = cdens(ch, [(s_["sea"], None), (s_["dust"], s_["f"])])
            ea, ja = AUD.subtract(ch, Cpart[ch.kappa], ref, "replay: the member's total source, every channel (C - P)", "source", known)
            eb, jb = AUD.subtract(ch, Cpart[ch.kappa], s_["pbar"], "replay: the contrast, every channel (C - P_bar)", "contrast", known)
            eP += ch.w * ea; jP += ch.w * ja; eB += ch.w * eb; jB += ch.w * jb
        es = np.asarray(Z["eps_tot"][k], float); Js = np.asarray(Z["J_tot"][k], float)
        esd = es - np.asarray(Z["eps_pairs"][k], float); Jsd = Js - np.asarray(Z["J_pairs"][k], float)
        rr = {"k": k, "T_over_rs": T / rs, "CP_eps": rel(eP, es), "CP_J": rel(jP, Js), "CPbar_eps": rel(eB, es), "CPbar_J": rel(jB, Js),
              "CPbar_vs_dust_eps": rel(eB, esd), "CPbar_vs_dust_J": rel(jB, Jsd), "modes": modes, "modes_differ_from_stored": mode_differs, "force_subdivide": force,
              "pbar_minus_sea": max(float((st[c.kappa]["pbar"] - st[c.kappa]["sea"]).abs().max()) for c in chs), "eig_s": t_eig}
        rr["accepted"] = bool(max(rr["CP_eps"], rr["CP_J"]) <= a.accept)
        replay_rows.append(rr)
        if not rr["accepted"] and k <= k0:
            dump()
            raise RuntimeError("the replayed sources differ from the stored ones at T/rs %.3f (eps %.2e, J %.2e; acceptance %g)" % (T / rs, rr["CP_eps"], rr["CP_J"], a.accept))
        # ---------------------------------------------------------------- T0: the selection, the code, its checks and the solves (37.4 (1))
        if k == k0:
            t_t0 = time.time()
            N_run, v_run, E_run, fail_run = solve_total(eP, jP)
            head["solve_at_T0"] = {"E_total": E_run, "E_stored": rec0.get("E"), "failed": fail_run}
            head["prep"] = {}
            for kap in kappas:
                ch = byk[kap]; s_ = st[kap]; e, U, ref, ci = cache[kap]
                occ = np.concatenate([np.ones(s_["n_sea"]), np.asarray(s_["f"], float)])
                keepd[kap] = [i for i, x in enumerate(np.asarray(s_["f"], float)) if abs(x - 1.0) < 1e-12]
                excl = [float(x) for x in np.asarray(s_["f"], float) if abs(x - 1.0) >= 1e-12]
                Phi = state_cols(kap)
                orth = float((Phi.conj().T @ Phi - torch.eye(Phi.shape[1], dtype=Phi.dtype, device=dev)).abs().max())
                hm = E17.clocked_gpu(ch, N_app, v_app)
                hm_host = E17.clocked_host_sparse(ch, N_app, v_app)
                dbl = doublet_checks(ch, hm_host)
                f_match = {}
                if mnpz is not None:
                    for cf_m in match["head"]["configs"]:
                        if int(cf_m["kappa"]) == kap:
                            f_match[float(cf_m["eps"])] = C5.T_(mnpz["T0_cfg%d_f" % int(cf_m["index"])])
                selk = select_orbitals(Phi, iX, hm, e, U, eps_list, R, rin0, Rt0, a.hist_bin)
                if f_match:
                    selm = {}
                    for eps_ in eps_list:
                        if eps_ in f_match:
                            s1_ = select_orbitals(Phi, iX, hm, e, U, [eps_], R, rin0, Rt0, a.hist_bin, f_match=f_match[eps_])
                            selm[eps_] = s1_["configs"][0]
                    for i_, row in enumerate(selk["configs"]):
                        if row["eps"] in selm and selm[row["eps"]].get("chosen"):
                            own = row
                            row2 = selm[row["eps"]]
                            row2["own_rule_orbital"] = {"energy": own.get("killing_energy"), "eps_measured": own.get("eps_measured"),
                                                        "overlap_with_matched": float((own["_f"].conj() @ row2["_f"]).abs()) if own.get("chosen") else None}
                            selk["configs"][i_] = row2
                hd = {"occupations_sea": int(s_["n_sea"]), "occupations_dust": [float(x) for x in np.asarray(s_["f"], float)], "excluded_occupations": excl,
                      "n_occ": int(Phi.shape[1]), "orthonormality": orth, "doublet": dbl,
                      "lambda": selk["lambda"], "counts": selk["counts"], "X_sites": selk["X_sites"], "configs": []}
                print("T0 kappa %d: Phi = [sea %d, dust %d] (occupations %s; excluded %s), orthonormality %.1e; L_X's eigenvalues (X: %d sites, %d nonzero at most): counts at 1 - eps for eps 1e-6, 1e-8, 1e-10, 1e-12: %s; the top 26: %s" % (
                    kap, s_["n_sea"], len(keepd[kap]), [float(x) for x in np.asarray(s_["f"], float)], excl or "none", orth, iX, 2 * iX,
                    ", ".join(str(selk["counts"][k_]) for k_ in ("1e-06", "1e-08", "1e-10", "1e-12")), " ".join("%.3e" % (1.0 - x) for x in selk["lambda"][:26])), flush=True)
                print("  the doublet: S_i^2 - I %.1e; S_i S_j - i eps_ijk S_k %.1e; [S_i, 1 (x) a_x(B)] over %d source vertices %.1e; [S_i, 1 (x) h_m(T0)] %.1e" % (
                    dbl["S_squared_minus_I"], dbl["S_i_S_j_minus_i_eps_S_k"], dbl["source_vertices_checked"], dbl["source_vertices_max_commutator"], dbl["stage_operator_commutator"]), flush=True)
                for row in selk["configs"]:
                    if not row.get("chosen"):
                        print("  eps %g: %s" % (row["eps"], row.get("why")), flush=True)
                        hd["configs"].append({k_: v_ for k_, v_ in row.items() if not k_.startswith("_")})
                        continue
                    f = row["_f"]; u = row["_u"]
                    w, beta, alpha = reflector(u)
                    chk, (e_F, j_F) = code_checks(ch, Phi, f, u, w, beta)
                    hk = -float((f.conj() @ (hm @ f)).real)
                    Nh, vh, Eh, fh = solve_total(eP - e_F, jP - j_F)
                    row["charge"] = {"first_variation_stage": hk / abs(E_run), "first_variation_unclocked": -chk["hole"]["sum_eps_F"] / abs(E_run),
                                     "solved": (Eh - E_run) / abs(E_run), "E_hole": Eh, "E_total": E_run, "N_vs_total": float(np.max(np.abs(Nh - N_run))),
                                     "v_vs_total": float(np.max(np.abs(vh - v_run))), "failed": [fh, fail_run]}
                    row["checks"] = chk
                    row["reflector"] = {"beta": beta, "alpha": [complex(alpha).real, complex(alpha).imag]}
                    i_cfg = len(cfgs)
                    cfgs.append({"kappa": kap, "eps": row["eps"], "f": f.clone(), "f0": f.clone(), "u": u, "w": w, "beta": beta, "Wr": row["_Wr"].clone(),
                                 "eps_meas": row["eps_measured"], "index": i_cfg, "charge0": row["charge"]})
                    colhist[i_cfg] = []
                    states["T0_cfg%d_f" % i_cfg] = npy(f); states["T0_cfg%d_u" % i_cfg] = npy(u)
                    states["T0_cfg%d_reflector_w" % i_cfg] = npy(w); states["T0_cfg%d_Vkeep" % i_cfg] = npy(row["_Vk"])
                    states["cfg%d_params" % i_cfg] = np.array([kap, row["eps"], row["eps_measured"], beta])
                    hd["configs"].append({k_: v_ for k_, v_ in row.items() if not k_.startswith("_")})
                    sp_ = row["spectral"]
                    print("  eps %g: %d retained (band energies %s); chosen (%s): Killing energy %+.6f, measured eps %.3e, weight below r_in %s, in the annulus %s, centroid R %.3f; spectral weight at e < 0 %.6f, e > 0 %.6f (mean %+.4f, spread %.4f)%s" % (
                        row["eps"], row["retained"], " ".join("%+.4f" % x for x in row["band_energies"]), row["rule"], row["killing_energy"], row["eps_measured"],
                        fmt(row["weight_below_r_in"], "%.4f"), fmt(row["weight_annulus"], "%.4f"), row["centroid_R"], sp_["weight_negative"], sp_["weight_positive"],
                        sp_["mean_energy"], sp_["energy_spread"], ("; overlap with the primary's f %.6f (the projection's norm %.6f)" % (row["overlap_with_primary_f"], row["match_overlap"])) if "match_overlap" in row else ""), flush=True)
                    print("    the code: |C^2 - C| %.1e (the baseline), |C f - f| %.1e; per z: |C_z^2 - C_z| %s; against (C^2 - C) (x) I_2 %s; |Tr_copy C_z - (2C - F)| %s; Tr C_z - (2 Tr C - 1) %s; |Y_z Y_z^dagger - C_z| %s; sources against 2 dens(C) - dens(F) (eps) %s; between z (eps, J) %.1e, %.1e; [S_i, C_z] (the label) %s; [S_i, (C - F) (x) 1] %s" % (
                        chk["baseline_idempotency"], chk["C_f_minus_f"], ", ".join("%.1e" % q["idempotency"] for q in chk["rows"]), ", ".join("%.1e" % q["idempotency_minus_baseline"] for q in chk["rows"]),
                        ", ".join("%.1e" % q["trcopy_minus_2C_minus_F"] for q in chk["rows"]), ", ".join("%+.1e" % q["trace_minus_2trC_plus_1"] for q in chk["rows"]),
                        ", ".join("%.1e" % q["columns_vs_Cz"] for q in chk["rows"]), ", ".join("%.1e" % q["src_vs_pred_eps"] for q in chk["rows"]),
                        chk["src_between_z_eps"], chk["src_between_z_J"], "; ".join(",".join("%.2e" % q["commutator_S_label"][i_] for i_ in ("1", "2", "3")) for q in chk["rows"]),
                        ",".join("%.1e" % chk["commutator_S_background"][i_] for i_ in ("1", "2", "3"))), flush=True)
                    print("    overlaps |det(Y_a^dagger Y_b)|: %s; V_rest: orthonormality %.1e, against f %.1e" % (
                        "; ".join("%s vs %s: %.3e (|<a|b>| %.6f; smallest singular value %.3e)" % (q["a"], q["b"], q["abs_det"], q["abs_z_overlap"], q["smallest_singular_value"]) for q in chk["pairs"]),
                        chk["rest_orthonormality"], chk["rest_against_f"]), flush=True)
                    print("    the charge: first variation -<f|h_m|f>/E %+.6e (the stage operator), -sum eps(F)/E %+.6e (unclocked); solved (E_hole - E)/E %+.6e (E %.10f; max |dN| %.2e, |dv| %.2e)" % (
                        row["charge"]["first_variation_stage"], row["charge"]["first_variation_unclocked"], row["charge"]["solved"], E_run,
                        row["charge"]["N_vs_total"], row["charge"]["v_vs_total"]), flush=True)
                head["prep"][str(kap)] = hd
                Wacc[kap] = torch.eye(2 * n, dtype=torch.complex128, device=dev)
                Phi0[kap] = Phi.clone()
                del hm
            head["configs"] = [{"index": cf["index"], "kappa": cf["kappa"], "eps": cf["eps"], "eps_measured": cf["eps_meas"]} for cf in cfgs]
            head["t0_seconds"] = time.time() - t_t0
            if not cfgs:
                dump()
                raise RuntimeError("no configuration: no retained subspace at any epsilon")
            print("T/rs %.3f: %d configurations (kappa, eps): %s; the T0 work %.1f s" % (T / rs, len(cfgs), ", ".join("(%d, %g)" % (cf["kappa"], cf["eps"]) for cf in cfgs), head["t0_seconds"]), flush=True)
        # ---------------------------------------------------------------- per record from T0: the readouts (37.4 (1))
        if k >= k0:
            Rt_k = Rt_of[k] if (k_close is None or k < k_close) else None        # the surface persists from T0 until the closing slice
            i_Rt = site_beyond(R, Rt_k) if Rt_k is not None else None
            xs_ = (T - T0) / (a.solve_every_rs * rs)
            do_solve = (k == k0) or (k == K - 1) or abs(xs_ - round(xs_)) < 1e-9
            if do_solve and k != k0:
                N_run, v_run, E_run, fail_run = solve_total(eP, jP)
            srow = {"k": k, "T_over_rs": T / rs, "R_t": Rt_k, "R_t_site": (float(R[i_Rt]) if (i_Rt is not None and i_Rt < n) else None), "persists": Rt_k is not None,
                    "R_t_stored": Rt_of[k], "solved_here": bool(do_solve), "E_total": (E_run if do_solve else None), "corr": {}, "configs": []}
            rg = regions_of(i_Rt)
            i_cut = i_Rt if i_Rt is not None else i_Rt0
            for kap in kappas:
                e, U, ref, ci = cache[kap]
                Phi_t = state_cols(kap)
                CX = Phi_t[:2 * i_cut] @ Phi_t[:2 * i_cut].conj().T
                lx = torch.linalg.eigvalsh(0.5 * (CX + CX.conj().T)).real.cpu().numpy()
                V = U[:, e < 0]
                VX = V[:2 * i_cut] @ V[:2 * i_cut].conj().T
                lv = torch.linalg.eigvalsh(0.5 * (VX + VX.conj().T)).real.cpu().numpy()
                k2 = float(np.max(lx * (1.0 - lx))); kv2 = float(np.max(lv * (1.0 - lv)))
                crw = {"X_sites": int(i_cut), "X_rule": "inside R_t(t)" if i_Rt is not None else "inside R_t(T0) (after the closing)",
                       "K_op": math.sqrt(max(k2, 0.0)), "I_XY_nats": 2.0 * gauss_entropy(lx), "pinsker_lower_nats": 2.0 * max(k2, 0.0),
                       "vac_K_op": math.sqrt(max(kv2, 0.0)), "vac_I_XY_nats": 2.0 * gauss_entropy(lv), "vac_pinsker_lower_nats": 2.0 * max(kv2, 0.0), "vac_n": int(V.shape[1])}
                if k == k0 or k == K - 1:
                    Kd = Phi_t[:2 * i_cut] @ Phi_t[2 * i_cut:].conj().T
                    crw["K_op_direct_svd"] = float(torch.linalg.svdvals(Kd).max())
                    Cfull = Phi_t @ Phi_t.conj().T
                    crw["state_idempotency"] = float((Cfull @ Cfull - Cfull).abs().max())
                    del Kd, Cfull
                srow["corr"][str(kap)] = crw
                del CX, VX, V
            for cf in cfgs:
                kap = cf["kappa"]; ch = byk[kap]
                e, U, ref, ci = cache[kap]
                Phi_t = state_cols(kap)
                f_t = cf["f"]
                wsite = (f_t.abs() ** 2).reshape(n, 2).sum(dim=1).cpu().numpy()
                tail = np.cumsum(wsite[::-1])[::-1]
                Prest_t = rest_cols(Phi_t, cf["w"], cf["beta"])
                regs = {}
                for name, i0 in rg:
                    p = float(tail[i0]) if i0 < n else 0.0
                    A = Phi_t[2 * i0:]; B = Prest_t[2 * i0:]
                    if A.shape[0] > 0:
                        mv = lambda X, A=A, B=B: A @ (A.conj().T @ X) - B @ (B.conj().T @ X)
                        top = subspace_top(mv, int(A.shape[0]))
                    else:
                        top = [0.0, 0.0]
                    Dx = abs(top[0]); D2 = abs(top[1]) if len(top) > 1 else None
                    Wd = cf["Wr"][2 * i0:]
                    mc = float(torch.linalg.eigvalsh(Wd.conj().T @ Wd).real.max()) if Wd.shape[0] > 0 else 0.0
                    rgr = {"i0": int(i0), "R_first": float(R[i0]) if i0 < n else None, "p": p, "D_explicit": Dx, "D_explicit_second": D2, "identity_abs": abs(Dx - p),
                           "identity_rel": (abs(Dx - p) / p) if p > 1e-10 else None, "D_lower": p, "D_upper": min(1.0, 2.0 * p), "trace_norm_cov": 2.0 * p,
                           "max_capture": mc, "tail_certified": tail_bound(p, cf["eps_meas"])}
                    if p > a.decoder_floor:
                        b = torch.zeros(2 * n, dtype=torch.complex128, device=dev)
                        b[2 * i0:] = f_t[2 * i0:] / math.sqrt(p)
                        b0 = Wacc[kap].conj().T @ b
                        c = Phi0[kap].conj().T @ b0
                        uc_ = cf["u"].conj() @ c
                        n_adj = float(torch.linalg.norm(c - cf["u"] * uc_) ** 2)
                        cb = Phi_t.conj().T @ b
                        n_fwd = float(torch.linalg.norm(cb) ** 2) - float((f_t.conj() @ b).abs() ** 2)
                        dec = decoder(p, n_adj)
                        dec.update({"n_forward": n_fwd, "n_adjoint_minus_forward": n_adj - n_fwd, "b0_norm_minus_1": float(torch.linalg.norm(b0)) - 1.0})
                        rgr["decoder"] = dec
                    regs[name] = rgr
                Fd = det[kap]["F"]
                zf = Fd.conj().T @ f_t
                ppk = float(torch.linalg.norm(zf) ** 2)
                Az = Fd.conj().T @ Phi_t; Bz = Fd.conj().T @ Prest_t
                Mz = Az @ Az.conj().T - Bz @ Bz.conj().T
                evz = np.sort(np.abs(torch.linalg.eigvalsh(0.5 * (Mz + Mz.conj().T)).real.cpu().numpy()))[::-1]
                hz = {}
                if i_Rt is not None:
                    hz["beyond R_t(t)"] = halfzone_col(f_t[2 * i_Rt:].reshape(-1, 2))
                hz["beyond R_t(T0)"] = halfzone_col(f_t[2 * i_Rt0:].reshape(-1, 2))
                hz["whole chain"] = halfzone_col(f_t.reshape(-1, 2))
                chkd = None
                if k in (k0 + 1, K - 1):
                    Gd = 0.5j * (f_t[:, None] @ f_t.conj()[None, :])
                    hzc = halfzone(Gd, i_Rt0, cross=True, unsplit=None)
                    Ad = Phi_t[2 * i_Rt0:]; Bd = Prest_t[2 * i_Rt0:]
                    Md = Ad @ Ad.conj().T - Bd @ Bd.conj().T
                    evd = np.sort(np.abs(torch.linalg.eigvalsh(0.5 * (Md + Md.conj().T)).real.cpu().numpy()))[::-1]
                    hc = hz["beyond R_t(T0)"]
                    chkd = {"halfzone_dense_lo": hzc["bound_lo"], "halfzone_dense_hi": hzc["bound_hi"], "halfzone_dense_cross": hzc["bound_cross"],
                            "column_lo": hc["p_lo"], "column_hi": hc["p_hi"], "column_cross": hc["cross"],
                            "lo_diff": abs(hzc["bound_lo"] - hc["p_lo"]), "hi_diff": abs(hzc["bound_hi"] - hc["p_hi"]), "cross_diff": abs(hzc["bound_cross"] - hc["cross"]),
                            "dense_explicit_beyond_Rt0": float(evd[0]), "dense_explicit_second": float(evd[1]), "dense_vs_p": abs(float(evd[0]) - regs["beyond R_t(T0)"]["p"])}
                    del Gd, Md
                e_F = C5.dens(ch.h, f_t[:, None]); j_F = C5.dens(ch.Q, f_t[:, None])
                cu = U.conj().T @ f_t
                kil = float(((cu.abs() ** 2) * e.to(torch.float64)).sum())
                hole = {"max_eps_rel": float(np.max(np.abs(e_F))) / max(float(np.max(np.abs(eP))), 1e-300),
                        "max_J_rel": float(np.max(np.abs(j_F))) / max(float(np.max(np.abs(jP))), 1e-300),
                        "sum_eps": float(np.sum(e_F)), "sum_J": float(np.sum(j_F)), "killing_energy": kil}
                if do_solve:
                    if k == k0:
                        hole["charge"] = dict(cf["charge0"]); hole["charge"]["from"] = "T0"
                    else:
                        Nh, vh, Eh, fh = solve_total(eP - e_F, jP - j_F)
                        hole["charge"] = {"solved": (Eh - E_run) / abs(E_run), "first_variation_stage": -kil / abs(E_run), "E_hole": Eh, "E_total": E_run,
                                          "N_vs_total": float(np.max(np.abs(Nh - N_run))), "v_vs_total": float(np.max(np.abs(vh - v_run))), "failed": [fh, fail_run]}
                front = {}
                for ep_ in EPS_FRONT:
                    idx = np.nonzero(tail > ep_)[0]
                    front["%g" % ep_] = float(R[idx[-1]]) if len(idx) else 0.0
                crow = {"index": cf["index"], "kappa": kap, "eps": cf["eps"], "norm_minus_1": float(torch.linalg.norm(f_t)) - 1.0,
                        "f_vs_accumulated": float(torch.linalg.norm(f_t - Wacc[kap] @ cf["f0"])), "p_total": float(tail[0]), "regions": regs,
                        "packet": {"p": ppk, "D_explicit": float(evz[0]), "D_explicit_second": float(evz[1]) if len(evz) > 1 else None, "identity_abs": abs(float(evz[0]) - ppk),
                                   "n_modes": int(Fd.shape[1])}, "front": front, "halfzone": hz, "hole": hole}
                if chkd is not None:
                    crow["dense_checks"] = chkd
                srow["configs"].append(crow)
                colhist[cf["index"]].append(npy(f_t))
                del Prest_t, Az, Bz, Mz
            srow["seconds"] = time.time() - tr
            s7_rows.append(srow)

            def _pr(q, nm, key="p"):
                r_ = q["regions"].get(nm)
                return fmt(r_.get(key)) if r_ else "-"

            def _dec(q):
                nm = "beyond R_t(t)" if "beyond R_t(t)" in q["regions"] else "beyond R_t(T0)"
                dd = q["regions"][nm].get("decoder")
                return ("n %.3e, lambda %s, I_c %s" % (dd["n"], fmt(dd["lambda"], "%.3e"), fmt(dd["Ic_bits"], "%+.3e"))) if dd else "-"
            print("T/rs %.3f: R_t %s; p beyond R_t(t) %s (|explicit - p| %s); beyond R_t(T0) %s; r 32 %s; r 64 %s; front(1e-6) %s; lo share beyond R_t(t) %s; decoder %s; dE/E %s; %.1f s" % (
                T / rs, fmt(Rt_k, "%.4f"), ", ".join(_pr(q, "beyond R_t(t)") for q in srow["configs"]), ", ".join(_pr(q, "beyond R_t(t)", "identity_abs") for q in srow["configs"]),
                ", ".join(_pr(q, "beyond R_t(T0)") for q in srow["configs"]), ", ".join(_pr(q, "r 32") for q in srow["configs"]),
                ", ".join(_pr(q, "r 64") for q in srow["configs"]), ", ".join("%.0f" % q["front"]["1e-06"] for q in srow["configs"]),
                ", ".join(fmt(q["halfzone"].get("beyond R_t(t)", {}).get("share_lo") if q["halfzone"].get("beyond R_t(t)") else None, "%.4f") for q in srow["configs"]),
                "; ".join(_dec(q) for q in srow["configs"]), ", ".join(fmt(q["hole"].get("charge", {}).get("solved"), "%+.4e") for q in srow["configs"]), srow["seconds"]), flush=True)
        else:
            if k % 10 == 0 or k == k0 - 1:
                print("T/rs %.3f: replay eps %.2e J %.2e (C - P_bar against the stored %.2e, against the stored dust part %.2e); modes %s; %.1f s (eigh %.1f)" % (
                    T / rs, rr["CP_eps"], rr["CP_J"], rr["CPbar_eps"], rr["CPbar_vs_dust_eps"], modes, time.time() - tr, t_eig), flush=True)
        # the reference's state after the record (resume_state's order; version 2's)
        for ch in chs:
            e, U, ref, ci = cache[ch.kappa]; s_ = st[ch.kappa]
            s_["ref"] = ref
            if "occ_e" in ci:
                s_["levels_prev"] = (ci.pop("occ_e"), ci.pop("emp_e"), ci.get("partial_pairs", 0))
            if ci.get("gap_between_blocks") is not None or "weyl" in ci:
                s_["gap_between_prev"] = ci.get("gap_between_blocks")
        if a.out_npz and k == k0:
            states.update(snapshot("T0", cache))
        if a.out_npz and k == K - 1:
            states.update(snapshot("end", cache))
            for cf in cfgs:
                states["end_cfg%d_f" % cf["index"]] = npy(cf["f"])
        # the propagation by the record's step (version 2's), the configurations' columns and the accumulated propagator with it
        if k < K - 1:
            for ch in chs:
                e, U, ref, ci = cache[ch.kappa]; s_ = st[ch.kappa]
                ph = torch.exp(-1j * e.to(torch.complex128) * tau)
                prop = lambda X_, U=U, ph=ph: U @ (ph[:, None] * (U.conj().T @ X_))
                s_["sea"] = prop(s_["sea"]); s_["pbar"] = prop(s_["pbar"])
                if s_["dust"].shape[1]:
                    s_["dust"] = prop(s_["dust"])
            if k >= k0:
                for cf in cfgs:
                    e, U, ref, ci = cache[cf["kappa"]]
                    ph = torch.exp(-1j * e.to(torch.complex128) * tau)
                    cf["f"] = U @ (ph * (U.conj().T @ cf["f"]))
                    cf["Wr"] = U @ (ph[:, None] * (U.conj().T @ cf["Wr"]))
                for kap in list(Wacc.keys()):
                    e, U, ref, ci = cache[kap]
                    ph = torch.exp(-1j * e.to(torch.complex128) * tau)
                    Wacc[kap] = U @ (ph[:, None] * (U.conj().T @ Wacc[kap]))
        f_ref = (N_app, v_app)
        del cache
        if k % 5 == 0 or k == K - 1:
            dump()
    out["crossing"] = crossing_clocks(out)
    head["provenance"]["utc_end"] = utc()
    dump()
    for cr in out["crossing"]:
        print("crossing cfg %d (kappa %d, eps %g): closing at T/rs %s (%s r_s after T0); %s" % (
            cr["index"], cr["kappa"], cr["eps"], fmt(cr["closing_T_over_rs"], "%.3f"), fmt(cr["closing_minus_T0_rs"], "%.3f"),
            "; ".join("eps %s: beyond R_t(t) %s, beyond R_t(T0) while held %s, over the run %s" % (
                ek, clock_str(cv["Rt"]), clock_str(cv["Rt0_persisting"]), clock_str(cv["Rt0_run"])) for ek, cv in cr["clocks"].items())), flush=True)
    if a.out_npz:
        for i_, lst in colhist.items():
            if lst:
                states["cfg%d_columns" % i_] = np.array(lst)
        states["column_T_over_rs"] = np.array([r_["T_over_rs"] for r_ in s7_rows])
        np.savez(a.out_npz, **states)
    print("done: %d records in %.0f s" % (K, time.time() - t_start), flush=True)


def clock_str(c):
    if c is None:
        return "none"
    return "T/rs %.3f (%.3f r_s after T0, %s r_s before the closing; p %.3e; R_t %s)" % (
        c["T_over_rs"], c["after_T0_rs"], fmt(c["before_closing_rs"], "%.3f"), c["p"], fmt(c["R_t"], "%.4f"))


def crossing_clocks(Mx):
    """per configuration the first record from T0 at which p beyond R_t(t) exceeds each of EPS_CROSS while the surface persists (part 3's
    clocks for p_cap); the contrasts: p beyond R_t(T0) while the surface persists and over the run; the closing clock beside them."""
    h = Mx["head"]; s7 = Mx["s7"]; ep = h["episode"]; T0r = h["T0_over_rs"]
    res = []
    if not s7:
        return res
    for i_, c0 in enumerate(s7[0]["configs"]):
        row = {"index": c0["index"], "kappa": c0["kappa"], "eps": c0["eps"], "closing_T_over_rs": ep["closing_T_over_rs"], "last_T_over_rs": ep["last_T_over_rs"],
               "closing_minus_T0_rs": ep["closing_minus_T0_rs"], "last_minus_T0_rs": ep["last_minus_T0_rs"], "clocks": {}}

        def first(name, persist_only, eps):
            for r in s7:
                if persist_only and not r["persists"]:
                    continue
                q = r["configs"][i_]["regions"].get(name)
                if q is not None and q["p"] > eps:
                    return {"T_over_rs": r["T_over_rs"], "after_T0_rs": r["T_over_rs"] - T0r, "p": q["p"], "R_t": r["R_t"], "R_t_site": r["R_t_site"],
                            "before_closing_rs": (ep["closing_T_over_rs"] - r["T_over_rs"]) if ep["closing_T_over_rs"] is not None else None}
            return None
        for eps in EPS_CROSS:
            row["clocks"]["%g" % eps] = {"Rt": first("beyond R_t(t)", True, eps), "Rt0_persisting": first("beyond R_t(T0)", True, eps), "Rt0_run": first("beyond R_t(T0)", False, eps)}
        pers = [(r["T_over_rs"], r["configs"][i_]["regions"]["beyond R_t(t)"]["p"]) for r in s7 if r["persists"] and "beyond R_t(t)" in r["configs"][i_]["regions"]]
        row["max_p_Rt_persisting"] = list(max(pers, key=lambda q: q[1])) if pers else None
        row["last_persisting"] = list(pers[-1]) if pers else None
        row["n_persisting_records"] = len(pers)
        res.append(row)
    return res


# ------------------------------------------------------------------------------------------------------------------ compose
def compose(a):
    paths = a.compose.split(",")
    M = [json.load(open(p)) for p in paths]
    P_ = M[0]; Cn = M[1] if len(M) > 1 else None
    hP = P_["head"]
    Ms = [(P_, "primary")] + ([(Cn, "step control")] if Cn is not None else [])
    for Mx, role in Ms:
        if "crossing" not in Mx:
            Mx["crossing"] = crossing_clocks(Mx)
    lad = ["r %d" % r for r in hP["ladder_radii"]]
    L = ["## S7 part 4: the exactly equal-source copy-slot code on the stored r_s/a 32 history; the captured norm beyond the trapping surface, the branch split, the decoder (engine_s7_copyslot_v1.py; engine spec 37.4; inbox #31 S4)", "",
         "- %s." % D11_LABEL, "- %s." % STEP1, "- %s." % STEP2, "- %s." % ELL_LABEL, "- %s." % DOUBLET_LABEL,
         "- %s." % hP["provenance"]["withdrawn"],
         "- notation: P08's P (the formed projector) is this lane's C, the state; the code is C_z = (C - F) (x) I_2 + F (x) |z><z|, F = |f><f|, f an occupied orbital of the channel's state; this lane's P is the continued reference (P(t) (+) P(t) is each codeword's reference), P_bar the propagated initial reference.", ""]
    for Mx, role in Ms:
        h = Mx["head"]; pv = h["provenance"]
        L.append("- %s: %s (%s; r_s %g, tau %g, %d sites, m %g, g %g, ell %s, %d channels; 4 pi m r_s %.4f), T0 T/rs %.3f (%s; r_in %s, R_t %s; the stored E %s, the replayed total's solve %s), %d records to T/rs %.3f; the C-init re-prepared in %s passes (stored %s): max |N - N_cinit| %s, |v - v_cinit| %s, E0 %s (stored %s); %.0f s in all; %s to %s on %s; script md5 %s, member json md5 %s, fields md5 %s." % (
            role, h["tag"], h["member_script"], h["rs"], h["tau"], h["nsites"], h["m"], h["g"], h["ell"], h["kappa_sea"], pv["labels"]["four_pi_m_rs"], h["T0_over_rs"], h["T0_rule"],
            fmt(h["trapping_zeros_at_T0"]["r_in"], "%.4f"), fmt(h["trapping_zeros_at_T0"]["R_t"], "%.4f"), fmt(h["E_stored_at_T0"], "%.10f"), fmt(h["solve_at_T0"]["E_total"], "%.10f"),
            h["records"], h["T_end_over_rs"], h["cinit"]["passes"], h["cinit"]["stored_passes"], fmt(h["cinit"]["dN"], "%.2e"), fmt(h["cinit"]["dv"], "%.2e"),
            fmt(h["cinit"]["E0"], "%.10f"), fmt(h["cinit"]["stored_E0"], "%.10f"), Mx.get("seconds", float("nan")), pv["utc_start"], pv.get("utc_end", "-"), pv["host"],
            pv["script_md5"], pv["member_json_md5"], pv["member_fields_md5"]))
        ep = h["episode"]
        L.append("  The surface: R_t held from T/rs %.3f to %.3f about T0; the closing clock (the first slice after T0 without R_t) T/rs %s, %s r_s after T0 (the last record holding it %s r_s after T0); reappearances %s. X: %s." % (
            ep["first_T_over_rs"], ep["last_T_over_rs"], fmt(ep["closing_T_over_rs"], "%.3f"), fmt(ep["closing_minus_T0_rs"], "%.3f"), fmt(ep["last_minus_T0_rs"], "%.3f"),
            ep["reappearances_T_over_rs"] or "none", h["X"]["rule"]))
    ps_ = hP["p_star"]
    L += ["- P08 7.2's worst-background threshold, solved here by bisection: p* = %s (bracket %s; I_c at the midpoint %s); the background-independent non-entanglement-breaking bound 3 - 2 sqrt 2 = %.15f." % (
        fmt(ps_.get("p_star"), "%.15f"), ps_.get("bracket"), fmt(ps_.get("I_c_at_p_star"), "%.1e"), ps_["nonEB_background_independent"]), ""]
    # the selection
    L += ["### The selection at T0 (37.4 (1)): Phi^dagger Pi_X Phi, the retained subspaces, the band, the chosen orbital", ""]
    for Mx, role in Ms:
        h = Mx["head"]
        for kap, hd in sorted(h["prep"].items(), key=lambda kv: int(kv[0])):
            dbl = hd["doublet"]
            L.append("- %s, kappa %s: Phi = the sea's %d columns and the dust's %d (occupations %s; excluded %s), orthonormality %.1e; X %d sites; the eigenvalues lambda of Phi^dagger Pi_X Phi at or above 1 - eps for eps 1e-6, 1e-8, 1e-10, 1e-12: %s, %s, %s, %s; 1 - lambda for the top 26: %s." % (
                role, kap, hd["occupations_sea"], len(hd["occupations_dust"]), hd["occupations_dust"], hd["excluded_occupations"] or "none", hd["orthonormality"], hd["X_sites"],
                hd["counts"]["1e-06"], hd["counts"]["1e-08"], hd["counts"]["1e-10"], hd["counts"]["1e-12"], " ".join("%.2e" % (1.0 - x) for x in hd["lambda"][:26])))
            L.append("  The doublet (two copies, copy-major): max |S_i^2 - I| %.1e; max |S_i S_j - i eps_ijk S_k| %.1e; max |[S_i, 1 (x) a_x(B)]| over the %d source vertices (B = h, Q, every site) %.1e; max |[S_i, 1 (x) h_m(T0)]| %.1e." % (
                dbl["S_squared_minus_I"], dbl["S_i_S_j_minus_i_eps_S_k"], dbl["source_vertices_checked"], dbl["source_vertices_max_commutator"], dbl["stage_operator_commutator"]))
            for row in hd["configs"]:
                if not row.get("chosen"):
                    L.append("  - eps %g: %s." % (row["eps"], row.get("why")))
                    continue
                sp_ = row["spectral"]
                L.append("  - eps %g: %d retained (1 - lambda: %s); the band (h_m(T0) compressed, ascending): %s (%d negative); the orbital: %s; its Killing energy <f|h_m(T0)|f> %+.6f; measured eps %.3e; weight below r_in %s, in [r_in, R_t(T0)) %s, beyond R_t(T0) %.3e; centroid R %.3f; spectral weight over h_m(T0)'s levels: e < 0 %.6f, e > 0 %.6f (mean %+.4f, spread %.4f; the bins of width %g above 1e-6: %s)%s." % (
                    row["eps"], row["retained"], " ".join("%.2e" % (1.0 - x) for x in row["lambda_kept"]), " ".join("%+.4f" % x for x in row["band_energies"]), row["band_negative"],
                    row["rule"], row["killing_energy"], row["eps_measured"], fmt(row["weight_below_r_in"], "%.4f"), fmt(row["weight_annulus"], "%.4f"), row["weight_beyond_Rt0"], row["centroid_R"],
                    sp_["weight_negative"], sp_["weight_positive"], sp_["mean_energy"], sp_["energy_spread"], sp_["bin"],
                    ", ".join("[%+.2f, %+.2f) %.4f" % (b[0], b[1], b[2]) for b in sp_["bins"] if b[2] > 1e-6),
                    ("; the step control's match: overlap with the primary's f %.6f, the projection's norm %.6f; the own-rule orbital beside it: Killing energy %s, overlap with the matched %s" % (
                        row["overlap_with_primary_f"], row["match_overlap"], fmt((row.get("own_rule_orbital") or {}).get("energy"), "%+.6f"),
                        fmt((row.get("own_rule_orbital") or {}).get("overlap_with_matched"), "%.6f"))) if "match_overlap" in row else ""))
    # the code's checks
    L += ["", "### The code at T0 (P08 3.1): idempotency, the copy trace, the sources, the particle numbers, the Slater overlap, the label", "",
          "| member | kappa | eps | z | max abs(C_z^2 - C_z) | against (C^2 - C) (x) I_2 | max abs(Tr_copy C_z - (2C - F)) | Tr C_z - (2 Tr C - 1) | max abs(Y_z Y_z^dagger - C_z) | sources of Y_z against 2 dens(C) - dens(F) (eps; J, relative) | [S_1, C_z]; [S_2, C_z]; [S_3, C_z] (the label) |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for Mx, role in Ms:
        h = Mx["head"]
        for kap, hd in sorted(h["prep"].items(), key=lambda kv: int(kv[0])):
            for row in hd["configs"]:
                if not row.get("chosen"):
                    continue
                for q in row["checks"]["rows"]:
                    L.append("| %s | %s | %g | %s | %.1e | %.1e | %.1e | %+.1e | %.1e | %.1e; %.1e | %.3e; %.3e; %.3e |" % (
                        h["tag"], kap, row["eps"], q["z"].replace("|", "\\|"), q["idempotency"], q["idempotency_minus_baseline"], q["trcopy_minus_2C_minus_F"], q["trace_minus_2trC_plus_1"],
                        q["columns_vs_Cz"], q["src_vs_pred_eps"], q["src_vs_pred_J"], q["commutator_S_label"]["1"], q["commutator_S_label"]["2"], q["commutator_S_label"]["3"]))
    L.append("")
    for Mx, role in Ms:
        h = Mx["head"]
        for kap, hd in sorted(h["prep"].items(), key=lambda kv: int(kv[0])):
            for row in hd["configs"]:
                if not row.get("chosen"):
                    continue
                ck = row["checks"]; cg = row["charge"]
                L.append("- %s, kappa %s, eps %g: the baseline max |C^2 - C| %.1e, |C f - f| %.1e; Tr C %.6f; the codewords' sources between z: max %.1e (eps), %.1e (J) relative to max |2 dens(C) - dens(F)| (%.6e, %.6e); [S_i, (C - F) (x) 1] %s; V_rest: max |V_rest^dagger V_rest - I| %.1e, max |Phi_rest^dagger f| %.1e; the overlaps: %s." % (
                    h["tag"], kap, row["eps"], ck["baseline_idempotency"], ck["C_f_minus_f"], ck["trace_C"], ck["src_between_z_eps"], ck["src_between_z_J"], ck["src_scale_eps"], ck["src_scale_J"],
                    ", ".join("%.1e" % ck["commutator_S_background"][i_] for i_ in ("1", "2", "3")), ck["rest_orthonormality"], ck["rest_against_f"],
                    "; ".join("|det(Y_%s^dagger Y_%s)| %.3e (|<a|b>| %.6f, smallest singular value %.3e)" % (q["a"], q["b"], q["abs_det"], q["abs_z_overlap"], q["smallest_singular_value"]) for q in ck["pairs"])))
                L.append("  The common charge shift of the hole (the codewords' common source 2C - F against the unencoded 2C): first variation -<f|h_m(T0)|f>/E %+.6e (P08's comparator, h the stage operator), -sum_x eps_x(F)/E %+.6e (the unclocked energy); solved by the run's solver (E_hole - E)/E %+.6e (E %.10f; max |dN| %.2e, |dv| %.2e against the unencoded total's fields). The hole's source: sum eps(F) %+.6e, sum J(F) %+.6e." % (
                    cg["first_variation_stage"], cg["first_variation_unclocked"], cg["solved"], cg["E_total"], cg["N_vs_total"], cg["v_vs_total"], ck["hole"]["sum_eps_F"], ck["hole"]["sum_J_F"]))

    def shown(r, s7, Mx):
        x = (r["T_over_rs"] - s7[0]["T_over_rs"]) / a.md_every_rs
        ep = Mx["head"]["episode"]
        closing = ep["closing_T_over_rs"] is not None and abs(r["T_over_rs"] - ep["closing_T_over_rs"]) < 1e-9
        return r is s7[0] or r is s7[-1] or r["persists"] or closing or abs(x - round(x)) < 1e-9
    # the captured norm per record
    for Mx, role in Ms:
        s7 = Mx["s7"]; h = Mx["head"]
        if not s7:
            continue
        for i_, c0 in enumerate(s7[0]["configs"]):
            L += ["", "### The captured norm p(t) = ||Pi_D f(t)||^2, %s (%s, tau %g), kappa %d, eps %g: beyond R_t(t) while the surface persists (p_cap), its explicit two-codeword route |D_det - p|, beyond R_t(T0) (R >= %s), the ladder, the collar, the packet detector, the fronts, the half-zone shares beyond R_t(t) (beyond R_t(T0) after the closing), the maximum capture over the retained subspace, the solved common charge shift; every record while the surface persists, the closing, every %g r_s, the end" % (
                role, h["tag"], h["tau"], c0["kappa"], c0["eps"], fmt(h["R_t0_site"], "%.0f"), a.md_every_rs), "",
                  "| T/rs | R_t(t) (R >=) | p_cap beyond R_t(t) | abs(D_det - p_cap) (explicit) | beyond R_t(T0) | %s | collar | packet D_det | front 1e-6; 1e-3 | lo; hi share | max capture | dE/E (hole) |" % " | ".join(lad),
                  "|---|---|---|---|---|%s---|---|---|---|---|---|" % ("---|" * len(lad))]
            for r in s7:
                if not shown(r, s7, Mx):
                    continue
                q = r["configs"][i_]; rgq = q["regions"]
                hz_ = q["halfzone"].get("beyond R_t(t)") or q["halfzone"].get("beyond R_t(T0)")
                mc_ = (rgq.get("beyond R_t(t)") or rgq["beyond R_t(T0)"])["max_capture"]
                L.append("| %.3f | %s (%s) | %s | %s | %s | %s | %s | %s | %s; %s | %s; %s | %s | %s |" % (
                    r["T_over_rs"], fmt(r["R_t"], "%.4f"), fmt(r["R_t_site"], "%.0f"), fmt((rgq.get("beyond R_t(t)") or {}).get("p")),
                    fmt((rgq.get("beyond R_t(t)") or {}).get("identity_abs"), "%.1e"), fmt(rgq["beyond R_t(T0)"]["p"]),
                    " | ".join(fmt(rgq[x]["p"]) for x in lad), fmt(rgq["collar (beyond the M1 cut)"]["p"]), fmt(q["packet"]["p"]),
                    "%.0f" % q["front"]["1e-06"], "%.0f" % q["front"]["0.001"], fmt(hz_.get("share_lo") if hz_ else None, "%.4f"), fmt(hz_.get("share_hi") if hz_ else None, "%.4f"),
                    fmt(mc_), fmt(q["hole"].get("charge", {}).get("solved"), "%+.4e")))
    # the decoder
    for Mx, role in Ms:
        s7 = Mx["s7"]; h = Mx["head"]
        if not s7:
            continue
        for i_, c0 in enumerate(s7[0]["configs"]):
            L += ["", "### P08's decoder (p, n), %s, kappa %d, eps %g: the detector beyond R_t(t) while the surface persists, beyond R_t(T0) after the closing; n by the adjoint pass (the forward route its check), q, lambda, chi_dec, I_c (bits), entanglement-breaking (lambda <= 1/3), the binary-information lower bound, the tail-certified interior-transfer bound (P08 (8), eps measured %s)" % (
                role, c0["kappa"], c0["eps"], fmt(next((cf["eps_measured"] for cf in h["configs"] if cf["index"] == c0["index"]), None), "%.3e")), "",
                  "| T/rs | detector | p | n (adjoint) | n adjoint - forward | q | lambda | chi_dec | I_c | EB | 1 - h_2((1 - p)/2) | tail-certified | many-body bracket p <= D <= min(1, 2p) |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
            for r in s7:
                if not shown(r, s7, Mx):
                    continue
                q = r["configs"][i_]
                nm = "beyond R_t(t)" if "beyond R_t(t)" in q["regions"] else "beyond R_t(T0)"
                rg_ = q["regions"][nm]; dd = rg_.get("decoder")
                if dd is None:
                    L.append("| %.3f | %s | %s | - (p below the floor %g) | - | - | - | - | - | - | - | %s | [%s, %s] |" % (r["T_over_rs"], nm, fmt(rg_["p"]), h["decoder_floor"], fmt(rg_["tail_certified"]), fmt(rg_["D_lower"]), fmt(rg_["D_upper"])))
                    continue
                L.append("| %.3f | %s | %s | %.6e | %+.1e | %.6e | %s | %s | %s | %s | %s | %s | [%s, %s] |" % (
                    r["T_over_rs"], nm, fmt(rg_["p"]), dd["n"], dd["n_adjoint_minus_forward"], dd["q"], fmt(dd["lambda"], "%.4e"), fmt(dd["chi_dec_bits"], "%.4e"), fmt(dd["Ic_bits"], "%+.4e"),
                    "yes" if dd["entanglement_breaking"] else "no", fmt(dd["I_acc_lower_bits"], "%.4e"), fmt(rg_["tail_certified"]), fmt(rg_["D_lower"]), fmt(rg_["D_upper"])))
            r = s7[-1]; q = r["configs"][i_]
            L += ["", "- at the end (T/rs %.3f), every region: %s." % (r["T_over_rs"], "; ".join(
                "%s: p %s, n %s, lambda %s, I_c %s, max capture %s" % (nm, fmt(rg_["p"]), fmt((rg_.get("decoder") or {}).get("n"), "%.4e"), fmt((rg_.get("decoder") or {}).get("lambda"), "%.3e"),
                                                                     fmt((rg_.get("decoder") or {}).get("Ic_bits"), "%+.3e"), fmt(rg_["max_capture"])) for nm, rg_ in q["regions"].items()))]
    # the branch readout
    for Mx, role in Ms:
        s7 = Mx["s7"]; h = Mx["head"]
        if not s7:
            continue
        for i_, c0 in enumerate(s7[0]["configs"]):
            L += ["", "### The half-zone split of the column (the branch readout; version 3's projector), %s, kappa %d, eps %g: beyond R_t(t) while the surface persists, beyond R_t(T0) after it; the whole chain beside it" % (role, c0["kappa"], c0["eps"]), "",
                  "| T/rs | region (M sites) | p | p_lo (abs(a p) < pi/2) | p_hi | cross sqrt(p_lo p_hi) | quarters [0, pi/4), [pi/4, pi/2), [pi/2, 3pi/4], (3pi/4, pi] | E[cos a p]; E[cos 2 a p] | Parseval | the whole chain: lo; hi share, E[cos a p] |", "|---|---|---|---|---|---|---|---|---|---|"]
            for r in s7:
                if not shown(r, s7, Mx):
                    continue
                q = r["configs"][i_]
                nm = "beyond R_t(t)" if "beyond R_t(t)" in q["halfzone"] else "beyond R_t(T0)"
                z_ = q["halfzone"][nm]; wz = q["halfzone"]["whole chain"]
                L.append("| %.3f | %s (%d) | %s | %s | %s | %s | %s | %s; %s | %s | %s; %s, %s |" % (
                    r["T_over_rs"], nm, z_["M"], fmt(z_["p"]), fmt(z_["p_lo"]), fmt(z_["p_hi"]), fmt(z_["cross"]),
                    ", ".join("%.4f" % x for x in z_["quarter_shares"]) if z_["quarter_shares"] else "-", fmt(z_["cos_ap"], "%+.4f"), fmt(z_["cos_2ap"], "%+.4f"),
                    fmt(z_["parseval"], "%.1e"), fmt(wz["share_lo"], "%.4f"), fmt(wz["share_hi"], "%.4f"), fmt(wz["cos_ap"], "%+.4f")))
            for r in s7:
                dc = r["configs"][i_].get("dense_checks")
                if dc:
                    L.append("- T/rs %.3f, the dense checks beyond R_t(T0): version 3's halfzone() on G = (i/2) f f^dagger: lo %.6e, hi %.6e, cross %.6e against the column's %.6e, %.6e, %.6e (abs. diff. %.1e, %.1e, %.1e); the formed copy-1 block's dense eigvalsh: largest %.6e, second %.1e, against p %.1e." % (
                        r["T_over_rs"], dc["halfzone_dense_lo"], dc["halfzone_dense_hi"], dc["halfzone_dense_cross"], dc["column_lo"], dc["column_hi"], dc["column_cross"],
                        dc["lo_diff"], dc["hi_diff"], dc["cross_diff"], dc["dense_explicit_beyond_Rt0"], dc["dense_explicit_second"], dc["dense_vs_p"]))
    # the crossing clocks
    L += ["", "### The crossing clocks (part 3's form for p_cap): the first record from T0 at which p beyond R_t(t) exceeds 1e-3 and 1e-6 while the surface persists; the contrasts beyond R_t(T0)", "",
          "| member | kappa | eps | threshold | beyond R_t(t) | beyond R_t(T0) while held (contrast) | beyond R_t(T0) over the run (contrast) | the closing (r_s after T0) |", "|---|---|---|---|---|---|---|---|"]
    for Mx, role in Ms:
        for cr in Mx["crossing"]:
            for ek, cv in cr["clocks"].items():
                L.append("| %s | %d | %g | %s | %s | %s | %s | %s (%s) |" % (Mx["head"]["tag"], cr["kappa"], cr["eps"], ek, clock_str(cv["Rt"]), clock_str(cv["Rt0_persisting"]),
                                                                   clock_str(cv["Rt0_run"]), fmt(cr["closing_T_over_rs"], "%.3f"), fmt(cr["closing_minus_T0_rs"], "%.3f")))
    for Mx, role in Ms:
        for cr in Mx["crossing"]:
            L.append("- %s, kappa %d, eps %g: p_cap over the %d records holding the surface from T0: largest %s at T/rs %s; at the last record holding it %s (T/rs %s)." % (
                Mx["head"]["tag"], cr["kappa"], cr["eps"], cr["n_persisting_records"], fmt(cr["max_p_Rt_persisting"][1] if cr["max_p_Rt_persisting"] else None),
                fmt(cr["max_p_Rt_persisting"][0] if cr["max_p_Rt_persisting"] else None, "%.3f"), fmt(cr["last_persisting"][1] if cr["last_persisting"] else None),
                fmt(cr["last_persisting"][0] if cr["last_persisting"] else None, "%.3f")))
    # the step control
    if Cn is not None and Cn["s7"] and P_["s7"]:
        hC = Cn["head"]
        byT = {round(r["T_over_rs"], 6): r for r in Cn["s7"]}
        L += ["", "### The step control: %s (tau %g) against %s (tau %g) at the times both hold (ratios tau %g / tau %g)" % (hC["tag"], hC["tau"], hP["tag"], hP["tau"], hC["tau"], hP["tau"]), "",
              "| kappa | eps | T/rs | R_t: tau %g, tau %g | p beyond the own R_t(t): tau %g, tau %g (ratio) | beyond R_t(T0): tau %g, tau %g (ratio) | r 32: tau %g, tau %g (ratio) | front 1e-6 |" % (
                  hP["tau"], hC["tau"], hP["tau"], hC["tau"], hP["tau"], hC["tau"], hP["tau"], hC["tau"]), "|---|---|---|---|---|---|---|---|"]
        ratios = {}
        for r in P_["s7"]:
            rc = byT.get(round(r["T_over_rs"], 6))
            if rc is None:
                continue
            for q in r["configs"]:
                qc = next((x for x in rc["configs"] if x["kappa"] == q["kappa"] and x["eps"] == q["eps"]), None)
                if qc is None:
                    continue
                a8 = (q["regions"].get("beyond R_t(t)") or {}).get("p"); a4 = (qc["regions"].get("beyond R_t(t)") or {}).get("p")
                b8 = q["regions"]["beyond R_t(T0)"]["p"]; b4 = qc["regions"]["beyond R_t(T0)"]["p"]
                c8 = q["regions"]["r 32"]["p"]; c4 = qc["regions"]["r 32"]["p"]
                rat = lambda x, y: (y / x) if (x is not None and y is not None and x > 1e-12) else None
                key = (q["kappa"], q["eps"])
                for nm, x, y in (("Rt", a8, a4), ("Rt0", b8, b4), ("r32", c8, c4)):
                    rv = rat(x, y)
                    if rv is not None:
                        ratios.setdefault(key, {}).setdefault(nm, []).append(rv)
                L.append("| %d | %g | %.3f | %s, %s | %s, %s (%s) | %s, %s (%s) | %s, %s (%s) | %.0f, %.0f |" % (
                    q["kappa"], q["eps"], r["T_over_rs"], fmt(r["R_t"], "%.4f"), fmt(rc["R_t"], "%.4f"), fmt(a8), fmt(a4), fmt(rat(a8, a4), "%.3f"), fmt(b8), fmt(b4), fmt(rat(b8, b4), "%.3f"),
                    fmt(c8), fmt(c4), fmt(rat(c8, c4), "%.3f"), q["front"]["1e-06"], qc["front"]["1e-06"]))
        L.append("")
        for key, dct in sorted(ratios.items()):
            L.append("- kappa %d, eps %g: the ratio tau %g / tau %g where the primary's value exceeds 1e-12: %s." % (key[0], key[1], hC["tau"], hP["tau"], "; ".join(
                "%s %.3f to %.3f (%d times)" % (nm, min(v), max(v), len(v)) for nm, v in dct.items())))
        for crP in P_["crossing"]:
            crC = next((x for x in Cn["crossing"] if x["kappa"] == crP["kappa"] and x["eps"] == crP["eps"]), None)
            if crC is None:
                continue
            for ek in crP["clocks"]:
                a8 = crP["clocks"][ek]["Rt"]; a4 = crC["clocks"][ek]["Rt"]
                L.append("- kappa %d, eps %g, threshold %s: the crossing at tau %g %s, at tau %g %s%s." % (crP["kappa"], crP["eps"], ek, hP["tau"], clock_str(a8), hC["tau"], clock_str(a4),
                         ("; the difference %.3f r_s" % abs(a8["T_over_rs"] - a4["T_over_rs"])) if (a8 and a4) else ""))
    # the correlation comparator
    L += ["", "### The correlation comparator (RUN_PROTOCOL 7): K = Pi_X C(t) Pi_Y across the surface (X inside R_t(t), R_t(T0) after the closing), ||K||_op, the Gaussian I(X:Y) = 2 S(X) (nats), the bound 2 ||K||_op^2; the record's Killing vacuum beside it", "",
          "| member | kappa | T/rs | X sites | K_op (the operator norm of K) | I(X:Y) | 2 K_op^2 | vacuum K_op | vacuum I(X:Y) | vacuum 2 K_op^2 | direct svd; idempotency |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for Mx, role in Ms:
        s7 = Mx["s7"]
        for r in s7:
            if not shown(r, s7, Mx):
                continue
            for kap, cr_ in sorted(r["corr"].items(), key=lambda kv: int(kv[0])):
                L.append("| %s | %s | %.3f | %d | %.6e | %.6e | %.6e | %.6e | %.6e | %.6e | %s; %s |" % (
                    Mx["head"]["tag"], kap, r["T_over_rs"], cr_["X_sites"], cr_["K_op"], cr_["I_XY_nats"], cr_["pinsker_lower_nats"], cr_["vac_K_op"], cr_["vac_I_XY_nats"],
                    cr_["vac_pinsker_lower_nats"], fmt(cr_.get("K_op_direct_svd"), "%.6e"), fmt(cr_.get("state_idempotency"), "%.1e")))
    # the archive contract (RUN_PROTOCOL.md sec. 7), per record = per block (no extraction: the record's step is the block)
    L += ["", "### The archive per block (RUN_PROTOCOL.md sec. 7): where each category is held", "",
          "- provenance: the head's provenance (UTC start and end, host, argv, the md5 of this script, of engine_collapse_v17, v5, engine_fields_v12, of the member's json and fields npz; ma, G/a^2, r_s/a, 4 pi m r_s, ell, the site fields and the reference; the step 1 label); the stage is the stored history's (no continuation label of its own).",
          "- source equality: the code's checks at T0 (the codewords' copy-summed sources against each other and against 2 dens(C) - dens(F), Tr_copy C_z - (2C - F)); per record the common source's difference from the unencoded one (the hole: its largest site source against the total's, its sums, its Killing energy) and its solved charge shift; the common history is the stored member's (one history for every codeword by construction).",
          "- preparation: per configuration the orbital (its T0 column, its coefficients u and the retained subspace's in the npz), the localization eigenvalues (all, in the head), eps requested and measured, the spectral histogram, the copy matrices (S_i and C_z's checks), the Slater-projector errors, the overlaps, the charge shifts; no guard (the engine has no branch projector: guard leakage none).",
          "- detector: the regions (beyond R_t(t), beyond R_t(T0), R >= 32 ... 512, the collar) as lossless projector references, version 2's collar packets (J3''s probes, both copies); the half-zone split per record; copy resolution by the explicit two-codeword route; no recorder (Xi_k: no injection; the isometry residual is the propagator's: ||f(t)|| - 1, ||f(t) - W(t) f(T0)||); the boundary-return limit: the fronts against the chain's end.",
          "- exact difference: the column f(t) itself at every record (the npz's cfg*_columns; Delta C(t) = f(t) f(t)^dagger (x) sigma_z), d = ||Delta C_D||_op = p per region by both routes, ||Delta C_D||_1 = 2p.",
          "- information: per region the decoder (p, n by the adjoint pass and its forward check, q, lambda, chi_dec, I_c, the entanglement-breaking flag), the binary-information lower bound, the tail-certified transfer bound; iid use is a qualification of P08 7.2, not a property of one history.",
          "- correlation comparator: per kappa ||K||_op, I(X:Y) = 2 S(X), 2 ||K||^2 for the state and for the record's Killing vacuum.",
          "- errors: propagation (the replay's audit against the stored sources; the column's norm and two-route checks), preparation (eps measured; the code's roundoff checks), tail (P08 (8)), collector (none: projector regions), field history (step 1: the hole's charge shift, first variation and solved; the encoded history not run), reference convention (the codewords' sources subtract P(t) (+) P(t); the difference is reference free), projected branch (none in this engine), truncation (the code lives in one channel of the ten)."]
    # the checks over the run
    L += ["", "### The run's checks", ""]
    for Mx, role in Ms:
        s7 = Mx["s7"]
        if not s7:
            continue
        for i_, c0 in enumerate(s7[0]["configs"]):
            ser = [r["configs"][i_] for r in s7]
            ida = max(q_["identity_abs"] for q in ser for q_ in q["regions"].values())
            idr = [q_["identity_rel"] for q in ser for q_ in q["regions"].values() if q_.get("identity_rel") is not None]
            nadj = [abs(q_["decoder"]["n_adjoint_minus_forward"]) for q in ser for q_ in q["regions"].values() if q_.get("decoder")]
            nin = [q_["decoder"]["n_in_range"] for q in ser for q_ in q["regions"].values() if q_.get("decoder")]
            L.append("- %s, kappa %d, eps %g: over %d records: the identity D_det = p (explicit route) max abs. diff. %.1e, max rel. diff. where p > 1e-10 %s; the packet detector max abs. diff. %.1e; max abs(||f(t)|| - 1) %.1e; max ||f(t) - W(t) f(T0)|| %.1e; n adjoint against forward: max abs. diff. %s over %d decoder evaluations, n within [0, 1 - p] in %d of them; the hole's source at most %.3e (eps) and %.3e (J) of the total's largest; its Killing energy %+.6f at T0, %+.6f at the end." % (
                Mx["head"]["tag"], c0["kappa"], c0["eps"], len(ser), ida, fmt(max(idr) if idr else None, "%.1e"), max(q["packet"]["identity_abs"] for q in ser),
                max(abs(q["norm_minus_1"]) for q in ser), max(q["f_vs_accumulated"] for q in ser), fmt(max(nadj) if nadj else None, "%.1e"), len(nadj), sum(1 for x in nin if x),
                max(q["hole"]["max_eps_rel"] for q in ser), max(q["hole"]["max_J_rel"] for q in ser), ser[0]["hole"]["killing_energy"], ser[-1]["hole"]["killing_energy"]))
    # the audit
    L += ["", "### The audit: every source the script computes, and the reference it subtracted", "",
          "| member | source computed | role | reference received | calls | verdict |", "|---|---|---|---|---|---|"]
    ok_all = True
    for Mx, role in Ms:
        for r in Mx["audit"]["rows"]:
            L.append("| %s | %s | %s | %s | %d | %s |" % (Mx["head"]["tag"], r["label"], r["role"], r["reference"], r["calls"], r["verdict"]))
        ok_all = ok_all and Mx["audit"]["pass"]
    L.append("")
    L.append("- the audit's verdict: %s." % ("PASS" if ok_all else "FAIL"))
    for Mx, role in Ms:
        rp = Mx["replay"]
        nacc = sum(1 for r in rp if r["accepted"])
        mx = lambda key: max(r[key] for r in rp)
        L.append("- %s, the numeric audit over its %d records (T/rs %.3f to %.3f): the replayed C - P against the stored eps_tot, J_tot at most %.2e, %.2e relative (acceptance %g; %d of %d records within it); C - P_bar against the stored dust part at most %.2e, %.2e." % (
            Mx["head"]["tag"], len(rp), rp[0]["T_over_rs"], rp[-1]["T_over_rs"], mx("CP_eps"), mx("CP_J"), Mx["head"]["accept"], nacc, len(rp), mx("CPbar_vs_dust_eps"), mx("CPbar_vs_dust_J")))
        nmd = sum(len(r["modes_differ_from_stored"]) for r in rp)
        allm = {}
        for r in rp:
            for k_, v_ in r["modes"].items():
                allm[k_] = allm.get(k_, 0) + v_
        L.append("  The replay's continuation modes (channel-records): %s; %d channel-records differ from the run's stored mode." % ("; ".join("%s %d" % kv for kv in sorted(allm.items())), nmd))
    open(a.md, "w").write("\n".join(L) + "\n")
    summ = {"script": "engine_s7_copyslot_v1.py (compose)", "members": [m_["head"]["tag"] for m_, _ in Ms], "labels": [D11_LABEL, STEP1, STEP2, ELL_LABEL, DOUBLET_LABEL],
            "crossing": {m_["head"]["tag"]: m_["crossing"] for m_, _ in Ms}, "audit_pass": ok_all, "p_star": hP["p_star"],
            "configs": {m_["head"]["tag"]: m_["head"]["configs"] for m_, _ in Ms}}
    if a.out_json:
        json.dump(summ, open(a.out_json, "w"), indent=1, default=float)
    print("composed %s from %s" % (a.md, ", ".join(paths)), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--member", default=None, help="a stored engine_collapse_v17 run: dir/tag")
    ap.add_argument("--T0-over-rs", dest="T0_over_rs", type=float, default=18.0, help="37.4 (1): part 3's T0")
    ap.add_argument("--match-json", default=None, help="the primary member's json (its T0 and configurations: the step control)")
    ap.add_argument("--match-npz", default=None, help="the primary member's npz (its T0 orbitals: the step control's match)")
    ap.add_argument("--kappas", default="1,2")
    ap.add_argument("--eps-list", default="1e-8,1e-10", help="37.4 (1): the localization thresholds")
    ap.add_argument("--h-ell-max", type=float, default=0.1, help="37.4 (1): the core exclusion's H ell bound (applies only if the record stores a per-site H)")
    ap.add_argument("--ladder-radii", default="32,64,128,256,512")
    ap.add_argument("--omega-over-kappa", default="0.5,1,2")
    ap.add_argument("--mp-over-kappa", default="0,0.25")
    ap.add_argument("--accept", type=float, default=1e-8)
    ap.add_argument("--solve-every-rs", type=float, default=0.25, help="the hole's common charge shift solved every this many r_s from T0 (and at the end)")
    ap.add_argument("--decoder-floor", type=float, default=1e-12, help="the decoder is evaluated where p exceeds this (the matched mode is rounding below it)")
    ap.add_argument("--hist-bin", type=float, default=0.05, help="the spectral histogram's bin width (lattice energy units)")
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--records", type=int, default=None)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-npz", default=None)
    ap.add_argument("--compose", default=None)
    ap.add_argument("--md", default=None)
    ap.add_argument("--md-every-rs", type=float, default=2.0, help="compose: the tables' time stride (T0, the records holding the surface, the closing and the end always shown)")
    a = ap.parse_args()
    if a.compose:
        compose(a)
    else:
        member(a)


if __name__ == "__main__":
    main()
