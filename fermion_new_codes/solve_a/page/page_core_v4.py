#!/usr/bin/env python
"""page_core_v4.py. Changes against page_core_v3.py (builder analyst; SPEC_final4b_engine.md 31.4 "Ruled" (1) to (9) and 31.6 BUILDER
item 1; the GAP03 review review_GAP03_page_curve.tex, findings R3 to R11), each with the item it implements; imports page_ops_v4.
(31.4 (1); R3, R7.1) transition: the complete occupied sea (every column of the stage reference, the n_sea columns of P_k) is propagated
    through the ramp by CF4 in column chunks (--tr-chunk), together with the memory and the FIFO; P^cont_{k+1} is the engine's
    continuation of P_k through the ramp, engine_ops_v3.assign_max_overlap_v3 (tol_block 1e-8, tol_fallback 1e-6, the predecessor's
    mask) at every sub-step s_j = j/nsub of the segment h[1, v_old + s (v_new - v_old)] (one eigendecomposition per sub-step, the last
    the new stage's), with the Weyl certificate per sub-step (verified while 2 dh < the previous record's occupied-to-empty gap between
    blocks, dh = ||dH||_G/nsub; the inter-block isolation beside) and the sub-step count by S9 (a)'s 1/8 rule, nsub = ceil(8 ||dH||_G/g)
    with g the old record's occupied-to-empty gap (capped at --nsub-max and flagged), each sub-step's ||Delta h||/g and ||h_dot||/g^2
    printed and a sub-step above 1/8 marked unresolved; the record line says "certified" (verified, no unresolved sub-step, not capped) or
    "flagged" with the reasons. Rotated (numerically degenerate) blocks of the last assignment are applied to the new stage's eigenbasis so
    that P^cont is the stage's mask (their count and width printed). The new memory is C_exact - P^cont_{k+1}: the sea part
    U P_k U^dagger - P^cont by the principal angles between U Phi_k and P^cont (one SVD of the sine block, Bjorck-Golub, the sines below
    eps_form dropped with their share and entropy bound printed, as initial_memory prints), plus U Delta_k U^dagger (the propagated
    memory), on one split basis of the new reference's halves; then compressed at eps_rank with the dropped energy, share and entropy
    bound printed. The new stage's reference mask is the continued mask (st["neg"] = the continued occupation; e < 0 kept as
    st["neg_sign"]); the flipped modes are no longer inserted (they are in the reference) and 30.1 (4)'s labelled adiabatic coherence is
    withdrawn (it is in the memory). The adiabatic leak is measured over every old occupied mode (the weight of U phi_j outside P^cont),
    ||U P_k U^dagger - P^cont||_F beside it. The R3 reference offset ||P^cont - 1(h < 0)||_F^2 (the pocket counts) and the continuation's
    Weyl margin are printed.
(31.4 (5); R5, R7.3) the FIFO: prune_fifo drops an entry only below the weight threshold (zone plus inside); at the depth cap the depth
    doubles and the doubling is printed; the oldest entry's inward weight is returned (the orthogonality bound to earlier radiation). The
    FIFO is propagated through every transition (no longer cleared): each entry's columns are carried through the ramp, re-expressed in
    the new eigenbasis, kept on their labelled half of the new reference (the minor half, the entry's half defect, dropped and printed;
    a column whose minor half exceeds 1/2 is counted as a half flip) and orthonormalized per half and against the earlier entries, and
    the new memory is projected out of the propagated FIFO (the removed weight and energy printed). The oldest entry's inward weight is
    its largest column weight.
(31.4 (2); R7.2) every within-stage readout comes from the exact stage state C(t) = P_k + U(t) Delta_k U(t)^dagger, the stage-initial
    memory (rec c_init, MD_init) propagated: collar_flux(t) and cross_surface(t) (the cross-surface block and the mutual information across
    the trapping zero), as D7's partner readout already did; never from the retained block after collection.
(31.4 (3)) energy_identity is completed: tr[Delta_k h_k] = E_mem + E_coll + E_x + E_trunc,acc, E_trunc,acc the energy each compression
    drops (E(MD_new on c_new) - E(MD_fin on c_fin)), accumulated over the stage in rec["E_trunc"]; the uncompleted residual is returned
    beside it as the truncation's energy share (not a criterion). The Gram drops of the memory update are counted with a bound on their
    energy (count x largest dropped Gram eigenvalue x ||M_Delta|| x ||h||), printed.
(R5) block_step records the post-split inward weight of the collected modes (the largest weight inside R_cut of a normalized collected
    column), and D8's dimension factor: a compression adds r' s(eps_rank) (r' the compressed space's dimension; v3 n_drop s(eps_rank)) and
    a Gram drop adds dim s(g_max) (ent_bound_gram beside ent_bound); the clipping excursions are returned for the driver's budget.
(R8.3, 31.4 (6)) the stage record keeps the stored modes for the checkpoint (the driver saves them).
(R8.4) the empty memory: every routine handles r = 0 (the no-production control is representable).
(31.4 (9)) cross_surface(t) is the readout of the mutual information across the trapping zero of the stage state.

page_core_v3.py. Changes against page_core_v2.py (builder analyst, BRIEF_page_v3_builder.md), each with the item of
SPEC_final4b_engine.md 30.1 it implements: imports page_ops_v3. (2) initial_memory records the dropped-pair share
sqrt(2 sum s^2)/||Delta_0||_F (||Delta_0||_F = (2 sum s^2 + |n_0 - n_1|)^(1/2) over all sines) and the dropped pairs' entropy bound
sum s(s^2), which joins D8's accumulated bound in place of v2's 2 n_dropped s(largest dropped sine) (printed beside as
D8_lemma_dropped_pairs_v2). (3) energy_identity: tr[Delta_k h_k] = E_mem + E_coll + E_x from the stage record in the Heisenberg frame (E_coll
the collected set's energy with every cross energy between its blocks, E_x its coherence energy with the memory), only while the record
holds every collected mode (rec["n_coll_all"] counts all of them; v2's record stopped at the cap without saying so, and stage_direct now
returns "complete"). (4) transition takes tau_factor (8.3 (8)'s doubling) and records the dropped non-adiabatic coherence on the test
modes, ||U P_k U^dagger - P^cont||_F = (sum_j leak_j)^(1/2) (exact over all modes; the 128 lowest-|e| test modes carry it), relative to
the memory's ||Delta||_F at the transition and to the formation's ||Delta_0||_F, with no tolerance, and its entropy bound sum_i s(leak_i),
which joins D8's accumulated bound. (6) partner_readout labels its rows "within stage 1" or "earlier stages' radiation omitted"
(Channel.n_transitions).

page_core_v2.py. Changes against page_core_v1.py (builder analyst, BRIEF_page_debug_v1.md), each with the failing test it fixes:
(1) initial_memory (spec 4.3, T4b): v1 took sin theta = (1 - sigma^2)^(1/2) from the cosine SVD, which returns 0 or noise for every
    sin theta below about 5e-8 (box CPU: 29 sines above 1e-9 against 34 from the sine block; the kept count differs between the H100
    and the CPU), and normalized a - b, which cancels for small theta. v2 keeps the cosine SVD and the pairs, takes the sines of the
    small angles (sigma >= 1/sqrt 2) from the SVD of the sine block V_+^dagger Phi_0 on their right subspace (Bjorck-Golub), and assembles
    e_+- from the pair's unit split components and (sin theta, cos theta): with a = s u_+ + c b, e_+ = ((1+s)/2)^(1/2) u_+ +
    ((1-s)/2)^(1/2) b and e_- = ((1-s)/2)^(1/2) u_+ - ((1+s)/2)^(1/2) b, the same vectors as spec 4.3's half-sum and half-difference
    without the cancellation; M_Delta,0 = (D_0^dagger E) diag(lambda) (E^dagger D_0) as before. The record adds n_dropped_pairs.
(2) every orthonormalize_halves call passes the stage's negmask (page_ops_v2: stable and exactly split; T4d, T4e, T4g, and the inputs
    of T4f and T4i, which still fail for the reasons in the builder's report: the specification's balance omits cross energies, the
    adiabatic sea's coherences exceed T4i's tolerance), and
    the compression's split hull (block_step (g), transition) is built in the memory's coordinates (page_ops_v2.hull_in_coords: the same
    basis and drop rule, exactly inside the memory's span).
(3) block_step (d): the projection against the FIFO is done twice (classical Gram-Schmidt twice) before the per-half re-orthonormalization
    (T4d, T4g). The candidates largely repeat FIFO modes (fifo_overlap 0.44 to 0.87 in T4 at the production thresholds, 0.45 to 0.93
    in the other T4 runs, not the expected 1e-4: the
    minor spectral half of a collected direction is a normalized tail that lies largely inside R_cut), so one pass leaves residuals whose
    non-orthogonality is amplified by 1/|residual| and compounds (block 3: 3.9e-5, block 4: 0.78 with exact arithmetic elsewhere).
    The record adds fifo_overlap_after (the largest overlap after the projection). The same amplification acts on the memory update
    (f): with exact orthonormality and split, the memory's overlap with the collected modes still grew 1.6e-8, 4.3e-5, 1.3e-4, 4.2e-3,
    0.39 over blocks 2 to 6, so (f) repeats D' = D_t - Q A against Q and the FIFO (zero in exact arithmetic: the memory is orthogonal
    to every earlier collected mode by unitarity, spec 4.2 (d)) before and after the per-half normalization.
(4) transition: t_ref = t_ref_old + tau_tr at the end of the ramp (v1 set 0.0 while the driver's clock continues at t + tau_tr, so every
    block after a transition used the wrong phases exp(-i e (t - t_ref)) and the stage record's t_init was wrong); the icloud lane's
    finding (SPEC_final4b_engine.md 29.9: the channel's time origin stays global across transitions); not a self-test item.
(5) stage_direct with_spectrum after a resume: the stage record restarts its stored modes empty on resume (the checkpoint does not carry
    them) while the collection times and half labels are restored in full; v1 zipped the stored modes with the stage's first collection
    times (wrong phases) and indexed the half labels from 0; v2 records where the stored modes start (rec["cQ_from"], set by
    page_recursion_v2.resume) and reads the matching times, half labels and covariance block (same class as the icloud lane's resume
    finding, whose FIFO part is fixed in page_recursion_v2.checkpoint).

page_core_v1.py (lead reviewer A, GAP03 integration; builder analyst runs it): the retained-channel recursion of the review's S5 in
the lane's radial channel form (GAP03_integrated_v1.md section 4): the stage operator and its eigendecomposition, the formation state,
the initial memory from the principal angles between the formation state and the stage's Killing vacuum, the block step (collection beyond
R_cut with the exact spectral split, the block covariance, the retained propagator D_k, the vacuum contribution Xi_k, the compression, the
entropies S_esc = tr s(M), S(B_k) = tr s(Z_k), the mutual information and the turnover flag), and the transition between stages (the ramp
by CF4, the adiabatic continuation of the sea with its measured defect, the flipped modes). All quantities per channel; the weights 16 kappa
are applied by the driver. torch complex128."""
import importlib.util
import math
import os
import sys
import time

import numpy as np

import torch

import page_ops_v4 as OPS

CD = OPS.CD
RD = OPS.RD
HERE = os.path.dirname(os.path.abspath(__file__))
_ENGINE = {}


def engine_dirs(engine_path=None):
    """v4: the candidate directories of the engine's operator files (engine_ops_v1.py for T1a, engine_ops_v3.py for the continuation),
    in order: --engine-path, $PAGE_ENGINE_PATH, ../engine, ../engine/code, ../../engine/code beside this file, <internal>/code."""
    c = []
    if engine_path:
        c.append(engine_path)
    if os.environ.get("PAGE_ENGINE_PATH"):
        c += os.environ["PAGE_ENGINE_PATH"].split(os.pathsep)
    c += [os.path.join(HERE, "..", "engine"), os.path.join(HERE, "..", "engine", "code"), os.path.join(HERE, "..", "..", "engine", "code"),
          "<internal>/code"]
    return [os.path.abspath(x) for x in c]


def engine_ops_v3(engine_path=None):
    """v4 (31.4 (1)): engine_ops_v3 imported by path (it imports engine_ops_v2's gershgorin, so its directory joins sys.path)."""
    if "v3" in _ENGINE:
        return _ENGINE["v3"]
    dirs = engine_dirs(engine_path)
    found = [d for d in dirs if os.path.isfile(os.path.join(d, "engine_ops_v3.py")) and os.path.isfile(os.path.join(d, "engine_ops_v2.py"))]
    if not found:
        raise ImportError("engine_ops_v3.py with engine_ops_v2.py not found in %s" % dirs)
    if found[0] not in sys.path:
        sys.path.insert(0, found[0])
    spec = importlib.util.spec_from_file_location("engine_ops_v3", os.path.join(found[0], "engine_ops_v3.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.ENGINE_DIR = found[0]
    _ENGINE["v3"] = mod
    return mod


def principal_delta(Phi_c, refmask, eps_form, eps_orth, device, method="cos+sin"):
    """v4: Delta = Phi Phi^dagger - P_ref by the principal angles (spec 4.3), Phi_c the eigenbasis coefficients of an orthonormal set
    (n2 x n0) and refmask the reference's occupied rows (n1 = |refmask|). method "cos+sin" is initial_memory's v2/v3 algorithm unchanged
    (the cosine SVD, the small angles' sines and vectors from the SVD of the sine block on their right subspace); method "sine" (the
    transition, 31.4 (1); n0 = n1) takes every angle from one SVD of the sine block V_+^dagger Phi (Bjorck-Golub: sines and the positive
    halves' principal vectors with full accuracy at small angles; the cosines as the norms of the negative-half parts). The pairs with
    sine at most eps_form are dropped. Returns (c_mem, half_mem, MD, lam, info)."""
    notneg = ~refmask
    Pp, Pm = Phi_c[notneg], Phi_c[refmask]
    n0, n1 = Phi_c.shape[1], int(refmask.sum())
    n2 = Phi_c.shape[0]
    nmin = min(n0, n1)
    s_parts, c_parts, up_parts, bm_parts = [], [], [], []
    n_dropped_pairs = 0; dropped = 0.0; sin_max = 0.0
    dropped_sq = 0.0; bound_pairs = 0.0; total_sq = 0.0
    U = Vr = None
    t0 = time.time()
    if method == "sine" and n0 == n1 and n0 > 0:
        Us, Ss, Whs = torch.linalg.svd(Pp, full_matrices=False)           # sines; the right vectors span every pair with a sine
        Wr = Whs.mH                                                          # n0 x min(npos, n0)
        bS = Pm @ Wr
        cS = torch.linalg.norm(bS, dim=0)
        keepS = Ss > eps_form
        n_dropped_pairs += int(Ss.numel()) - int(keepS.sum())
        total_sq += float((Ss * Ss).sum())
        if bool((~keepS).any()):
            sd = Ss[~keepS]
            dropped = float(sd.max()); dropped_sq = float((sd * sd).sum())
            bound_pairs = OPS.binary_entropy_np((sd * sd).cpu().numpy())
        if Ss.numel():
            sin_max = float(Ss.max())
        if bool(keepS.any()):
            s_parts.append(Ss[keepS]); c_parts.append(cS[keepS]); up_parts.append(Us[:, keepS])
            bm_parts.append(bS[:, keepS] / cS[keepS].clamp_min(1e-300).to(CD)[None, :])
    elif n0 > 0 and n1 > 0:
        U, S, Vh = torch.linalg.svd(Pm, full_matrices=True)                  # the cosine SVD of spec 4.3: W = U Sigma V^dagger
        Vr = Vh.mH
        sig = S.clamp(0.0, 1.0)
        small = sig >= 1.0 / math.sqrt(2.0)
        iS = small.nonzero().ravel(); iL = (~small).nonzero().ravel()
        if iS.numel():
            VS = Vr[:, iS]
            Us, Ss, Whs = torch.linalg.svd(Pp @ VS, full_matrices=False)
            vS = VS @ Whs.mH
            bS = Pm @ vS
            cS = torch.linalg.norm(bS, dim=0)
            keepS = Ss > eps_form
            n_dropped_pairs += int(iS.numel()) - int(keepS.sum())
            total_sq += float((Ss * Ss).sum())
            if bool((~keepS).any()):
                sd = Ss[~keepS]
                dropped = float(sd.max())
                dropped_sq = float((sd * sd).sum())
                bound_pairs = OPS.binary_entropy_np((sd * sd).cpu().numpy())
            if Ss.numel():
                sin_max = max(sin_max, float(Ss.max()))
            if bool(keepS.any()):
                s_parts.append(Ss[keepS]); c_parts.append(cS[keepS]); up_parts.append(Us[:, keepS])
                bm_parts.append(bS[:, keepS] / cS[keepS].to(CD)[None, :])
        if iL.numel():
            vL = Vr[:, iL]
            cL = sig[iL]
            sL = torch.sqrt((1.0 - cL * cL).clamp_min(0.0))
            aP = Pp @ vL
            s_parts.append(sL); c_parts.append(cL); up_parts.append(aP / torch.linalg.norm(aP, dim=0).to(CD)[None, :]); bm_parts.append(U[:, iL])
            sin_max = max(sin_max, float(sL.max()))
            total_sq += float((sL * sL).sum())
    vecs, lams = [], []
    n_kept_pairs = 0
    if s_parts:
        s = torch.cat(s_parts); c = torch.cat(c_parts); up = torch.cat(up_parts, dim=1); bm = torch.cat(bm_parts, dim=1)
        n_kept_pairs = int(s.numel())
        al = torch.sqrt((1.0 + s) / 2.0).to(CD); be = (c / torch.sqrt(2.0 * (1.0 + s))).to(CD)
        Ep = torch.zeros((n2, n_kept_pairs), dtype=CD, device=device); Em = torch.zeros_like(Ep)
        Ep[notneg] = up * al[None, :]; Ep[refmask] = bm * be[None, :]
        Em[notneg] = up * be[None, :]; Em[refmask] = -bm * al[None, :]
        vecs += [Ep, Em]; lams += [s, -s]
    if n0 > n1:                                                               # unpaired (cos+sin only): in ran Phi only, positive half
        Vr_ = Vr if Vr is not None else torch.eye(n0, dtype=CD, device=device)    # R8.4: an empty reference
        Xu = torch.zeros((n2, n0 - nmin), dtype=CD, device=device); Xu[notneg] = Pp @ Vr_[:, nmin:]
        vecs.append(Xu); lams.append(torch.ones(n0 - nmin, dtype=RD, device=device))
    elif n1 > n0:
        U_ = U if U is not None else torch.eye(n1, dtype=CD, device=device)       # R8.4: an empty orthonormal set
        Xu = torch.zeros((n2, n1 - nmin), dtype=CD, device=device); Xu[refmask] = U_[:, nmin:]
        vecs.append(Xu); lams.append(-torch.ones(n1 - nmin, dtype=RD, device=device))
    lemma_v2 = 2 * n_dropped_pairs * OPS.binary_entropy_np([min(max(dropped, 1e-300), 0.5)])
    norm_D = math.sqrt(2.0 * total_sq + abs(n0 - n1))
    share = math.sqrt(2.0 * dropped_sq) / max(norm_D, 1e-300)
    if vecs:
        cE = torch.cat(vecs, dim=1); lam = torch.cat(lams)
        cols, halves = OPS.split_by_half(cE, refmask)
        c_mem, half_mem, _ = OPS.orthonormalize_halves(cols, halves, eps_orth, negmask=refmask)
        X = c_mem.mH @ cE
        MD = (X * lam.to(CD)[None, :]) @ X.mH
        MD = 0.5 * (MD + MD.mH)
    else:                                                                     # R8.4: the empty Delta (no production)
        lam = torch.zeros(0, dtype=RD, device=device)
        c_mem = torch.zeros((n2, 0), dtype=CD, device=device); half_mem = torch.zeros(0, dtype=torch.int64, device=device)
        MD = torch.zeros((0, 0), dtype=CD, device=device)
    info = {"n0": n0, "n1": n1, "rank_kept": n_kept_pairs * 2 + abs(n0 - n1), "sin_theta_max": sin_max, "n_dropped_pairs": n_dropped_pairs,
            "sin_theta_dropped_max": dropped, "dropped_pair_share": share, "norm_Delta_from_sines": norm_D, "entropy_bound_dropped_pairs": bound_pairs,
            "D8_lemma_dropped_pairs_v2": lemma_v2, "dropped_sq": dropped_sq, "total_sq": total_sq, "method": method, "seconds": time.time() - t0}
    return c_mem, half_mem, MD, lam, info


class Channel:
    """one radial channel kappa: the chain, the banded operator, the memory (c in the stage eigenbasis, half labels, M_Delta), the FIFO of
    recently collected modes, the stage data and the stage's direct-covariance records."""

    def __init__(self, kappa, m, nsites, half_offset, device):
        self.kappa = int(kappa)
        self.w = 16.0 * self.kappa
        self.device = device
        self.R = np.arange(nsites, dtype=float) + (0.5 if half_offset else 1.0)
        self.n2 = 2 * nsites
        self.h0, self.Q = OPS.chain_sparse(self.R, m, float(kappa))
        self.B = OPS.ChainBanded(self.h0, self.Q, self.R, device)
        self.R2 = torch.as_tensor(np.repeat(self.R, 2), dtype=RD, device=device)
        # memory
        self.c = None; self.half = None; self.MD = None; self.t_ref = 0.0
        self.fifo = []                       # list of (c_Q (n2 x m) in the stage eigenbasis, t_collect)
        self.stage = None
        self.rec = None                      # stage direct-covariance records
        self.ent_bound = 0.0                 # accumulated entropy bound of every truncation (D8)
        self.n_transitions = 0               # v3 (30.1 (6)): the stage count of this channel's record (D7's label)
        self.norm_Delta0 = None              # v3 (30.1 (4)): ||Delta_0||_F of the formation, from the sines
        # v4
        self.ent_bound_gram = 0.0            # (R5) D8's term of the Gram drops, dim s(g_max) per call with a drop
        self.fifo_depth = None               # (31.4 (5)) the FIFO's depth, doubled at the cap (set from cfg at the first prune)
        self.fifo_doublings = []             # (31.4 (5)) (t, new depth) of every doubling
        self.fifo_oldest_weight = 0.0        # (31.4 (5)) the oldest entry's inward weight at the last prune
        self.gram_energy_bound = 0.0         # (31.4 (3)) the Gram drops' energy bound, accumulated over the stage

    # ------------------------------------------------------------------------------------------------------------ stage
    def build_stage(self, cfg, E, Rb, label=""):
        t0 = time.time()
        rs = 2.0 * cfg.g * E
        vfun = OPS.flow_dense(rs, Rb)
        v_sites = np.asarray(self.averager(vfun), float)
        h = OPS.clocked_csr(self.h0, self.Q, np.ones(len(self.R)), v_sites)
        e, V = OPS.eigh_csr(h, self.device)
        # v4 (R3): the stage-1 reference is declared, P_1 = 1(h_1 < 0); a transition replaces "neg" by the continued mask (31.4 (1))
        st = {"E": float(E), "rs": float(rs), "Rb": float(Rb), "v_sites": v_sites, "h": h, "e": e, "V": V,
              "neg": e < 0, "neg_sign": e < 0, "nneg": int((e < 0).sum()), "nzero": int((e.abs() < 1e-12).sum()), "label": label,
              "reference": "declared: 1(h < 0), the frozen stage's ground projector (R3)", "rot_blocks": []}
        st["r_t"], st["r_in"] = OPS.trapping_zeros(v_sites, self.R)
        st["Jout"] = OPS.Banded(OPS.current_out_csr(h, self.R, cfg.R_collar_abs), self.device)
        st["eigh_s"] = time.time() - t0
        self.stage = st
        return st

    def set_averager(self, ell, mode):
        self.averager = OPS.FlowAverager(self.R, ell, mode)

    # ------------------------------------------------------------------------------------------------------- formation
    def formation(self, cfg):
        """the physical state after the declared formation: Phi_0 = U_form Phi_i with Phi_i the Killing sea of the initial configuration
        (zero modes empty); returns Phi_0 (n2 x n_0) and the record."""
        t0 = time.time()
        rs = cfg.rs
        Rb0, Rbf = cfg.Rb0 * rs, cfg.Rbf * rs
        if cfg.formation == "dust":
            t_form = OPS.dust_time(Rb0, Rbf, rs)
            vfun_t = lambda t: np.asarray(self.averager(OPS.flow_dense(rs, OPS.dust_radius(t, Rb0, rs))), float)
            v_i = vfun_t(0.0)
        else:
            t_form = cfg.t_ramp * rs
            v_fin = np.asarray(self.averager(OPS.flow_dense(rs, Rbf)), float)
            vfun_t = lambda t: (math.sin(math.pi * min(max(t, 0.0), t_form) / (2.0 * t_form)) ** 2) * v_fin
            v_i = 0.0 * v_fin
        h_i = OPS.clocked_csr(self.h0, self.Q, np.ones(len(self.R)), v_i)
        e_i, V_i = OPS.eigh_csr(h_i, self.device)
        occ = e_i < 0
        Phi = V_i[:, occ].clone()
        n0 = int(occ.sum()); nzero = int((e_i.abs() < 1e-12).sum())
        del V_i
        u_max = float(np.max(np.abs(v_i)))
        t1 = time.time()
        chunk = max(1, int(getattr(cfg, "form_chunk", 4096)))
        parts = []
        for j0 in range(0, n0, chunk):                                       # column chunks bound the device memory of the Chebyshev recursion
            parts.append(OPS.cf4_propagate(self.B, Phi[:, j0: j0 + chunk], vfun_t, 0.0, t_form, cfg.dt_form))
        Phi0 = torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]
        del parts
        unit = float(torch.linalg.norm(Phi0.mH @ Phi0 - torch.eye(n0, dtype=CD, device=self.device)))
        rec = {"formation": cfg.formation, "t_form": t_form, "t_form_over_rs": t_form / rs, "n0": n0, "n_zero_modes_initial": nzero,
               "u_max_initial": u_max, "pockets_initial": bool(u_max >= 1.0), "unitarity": unit, "eigh_s": t1 - t0, "cf4_s": time.time() - t1,
               "dt_form": cfg.dt_form}
        return Phi0, rec

    # --------------------------------------------------------------------------------------------------- initial memory
    def initial_memory(self, Phi0, cfg):
        """Delta_0 = Phi_0 Phi_0^dagger - Phi_1 Phi_1^dagger by the principal angles (spec 4.3); the split hull as the memory; the
        inversion readout of spec 3.2. v4: the angles by principal_delta (method "cos+sin", v3's algorithm unchanged); the stage-1
        reference P_1 = 1(h_1 < 0) is declared (R3) and recorded so."""
        st = self.stage
        V, e, neg = st["V"], st["e"], st["neg"]
        Phi = V.mH @ Phi0                                                   # eigenbasis coefficients of the formation state (n2 x n0)
        c_mem, half_mem, MD, lam, pinfo = principal_delta(Phi, neg, cfg.eps_form, cfg.eps_orth, self.device, method="cos+sin")
        n0, n1 = pinfo["n0"], pinfo["n1"]
        n_kept_pairs = (pinfo["rank_kept"] - abs(n0 - n1)) // 2
        sin_max, n_dropped_pairs, dropped = pinfo["sin_theta_max"], pinfo["n_dropped_pairs"], pinfo["sin_theta_dropped_max"]
        bound_pairs, lemma_v2, share, norm_D0 = pinfo["entropy_bound_dropped_pairs"], pinfo["D8_lemma_dropped_pairs_v2"], pinfo["dropped_pair_share"], pinfo["norm_Delta_from_sines"]
        self.ent_bound += bound_pairs                                                            # v3 (30.1 (2)): sum s(s^2) joins D8's bound
        self.norm_Delta0 = norm_D0
        self.c, self.half, self.MD, self.t_ref = c_mem, half_mem, MD, 0.0
        # inversion readout: n_alpha = ||V[:, alpha]^dagger Phi_0||^2
        n_alpha = Phi.abs().pow(2).sum(dim=1)
        pos = e > 0
        Np = float(n_alpha[pos].sum()); Nm = float((1.0 - n_alpha[neg]).sum())
        E_inv = float((e[pos] * n_alpha[pos]).sum() + (e[neg].abs() * (1.0 - n_alpha[neg])).sum())
        TH = 1.0 / (4.0 * math.pi * st["rs"])
        inv = torch.where(pos, n_alpha, 1.0 - n_alpha)
        edges = torch.tensor([0, 1, 2, 3, 4, 6, 8, 12, 25, 1e9], dtype=RD, device=self.device)
        hist = [float(inv[(e.abs() / TH >= edges[i]) & (e.abs() / TH < edges[i + 1])].sum()) for i in range(len(edges) - 1)]
        rec = {"n0": n0, "n1": n1, "rank_kept": n_kept_pairs * 2 + abs(n0 - n1), "sin_theta_max": sin_max, "n_dropped_pairs": n_dropped_pairs,
               "sin_theta_dropped_max": dropped, "memory_rank": int(c_mem.shape[1]), "N_plus": Np, "N_minus": Nm, "E_inv": E_inv,
               "E_inv_over_E": E_inv / st["E"], "inverted_by_bin_TH": hist, "bin_edges_TH": [0, 1, 2, 3, 4, 6, 8, 12, 25, "inf"],
               "trace_Delta": float(lam.sum()), "trace_MD": float(torch.trace(MD).real),
               "dropped_pair_share": share, "norm_Delta0_from_sines": norm_D0, "entropy_bound_dropped_pairs": bound_pairs,
               "D8_lemma_dropped_pairs_v2": lemma_v2, "ent_bound_after_formation": self.ent_bound,
               "reference": st["reference"], "principal_angles_s": pinfo["seconds"]}
        self.init_stage_record()
        return rec

    # ------------------------------------------------------------------------------------------------ stage records
    def init_stage_record(self):
        # v4 (31.4 (3)): E_trunc, the energy the stage's compressions drop, accumulated; the Gram drops' energy bound beside it
        self.rec = {"c_init": self.c.clone(), "MD_init": self.MD.clone(), "t_init": self.t_ref, "a": [], "cQ": [], "halfQ": [], "tQ": [],
                    "n_coll": 0, "E_rad": 0.0, "E_delta_init": self.memory_energy(), "cQ_from": (0, 0), "n_coll_all": 0,
                    "E_trunc": 0.0, "gram_energy_bound": 0.0, "cQ_saved": 0, "cQ_files": []}

    def memory_energy(self, c=None, MD=None):
        """Re tr[M_Delta (c^dagger (e c))]: the memory's excess Killing energy (v4: of any (c, MD) on the stage's eigenbasis)."""
        e = self.stage["e"].to(CD)
        c = self.c if c is None else c
        MD = self.MD if MD is None else MD
        if c.shape[1] == 0:
            return 0.0
        return float(torch.trace(MD @ (c.mH @ (e[:, None] * c))).real)

    def memory_entropy(self):
        M = self.MD + torch.diag((self.half < 0).to(CD))
        return OPS.entropy_cov(M)

    def stage_state(self, t):
        """v4 (31.4 (2)): the exact stage state's excess at time t, Delta(t) = D_init(t) M_init D_init(t)^dagger with D_init(t) =
        V exp(-i e (t - t_init)) c_init (the stage-initial memory propagated; collections do not enter). Returns (D_init(t), M_init)."""
        rec = self.rec
        e, V = self.stage["e"], self.stage["V"]
        ph = torch.exp(-1j * e.to(CD) * (t - rec["t_init"]))
        return V @ (ph[:, None] * rec["c_init"]), rec["MD_init"]

    def collar_flux(self, t=None):
        """Re tr[Delta(t) J_out] at the collar. v4 (31.4 (2), R7.2): from the exact stage state (the stage-initial memory propagated to t,
        default the memory's time), not from the retained block after collection."""
        t = self.t_ref if t is None else t
        if self.rec is None or self.rec["c_init"].shape[1] == 0:
            return 0.0
        D, M0 = self.stage_state(t)
        return float(torch.trace(M0 @ (D.mH @ self.stage["Jout"].apply(D))).real)

    # --------------------------------------------------------------------------------------------------- block step
    def block_step(self, cfg, t_now):
        """spec 4.2 (a) to (j); returns the record of the checkpoint. v4: the collar flux from the exact stage state (31.4 (2)); the
        post-split inward weight of the collected modes (R5); D8's dimension factor for compressions and Gram drops (R5); E_trunc of
        the compression accumulated in the stage record (31.4 (3)); the FIFO pruned by weight only, its depth doubled at the cap
        (31.4 (5))."""
        st = self.stage
        e, V, neg = st["e"], st["V"], st["neg"]
        t0 = time.time()
        r = self.c.shape[1]
        ph = torch.exp(-1j * e.to(CD) * (t_now - self.t_ref))
        c_t = ph[:, None] * self.c
        S_old, exc_old = OPS.entropy_cov(self.MD + torch.diag((self.half < 0).to(CD)))
        bound0 = self.ent_bound + self.ent_bound_gram
        out = {"t": t_now, "r_before": r, "S_old": S_old}
        if r == 0:                                                            # R8.4: the empty memory
            self.c, self.t_ref = c_t, t_now
            self.prune_fifo(cfg, t_now)
            out.update({"m_coll": 0, "S_new": 0.0, "S_B": 0.0, "S_B_Z": 0.0, "I_RB": 0.0, "E_B": 0.0, "r_after": 0, "turnover": False,
                        "E_mem": 0.0, "L_collar": self.collar_flux(t_now), "ent_bound_increment": 0.0, "clip_excursion_max": exc_old,
                        "fifo_oldest_weight": self.fifo_oldest_weight, "fifo_depth": self.fifo_depth, "wall_s": time.time() - t0})
            return out
        D_t = V @ c_t
        mask_out = (self.R2 > cfg.R_cut_abs)
        X = D_t * mask_out.to(CD)[:, None]
        GX = X.mH @ X
        g, u = torch.linalg.eigh(0.5 * (GX + GX.mH))
        sel = g > cfg.eps_coll
        if int(sel.sum()) > cfg.m_cap:
            idx = torch.argsort(g, descending=True)[: cfg.m_cap]
            sel = torch.zeros_like(sel); sel[idx] = True
        out["weight_beyond_cut_total"] = float(g.clamp_min(0.0).sum())
        out["weight_beyond_cut_uncollected"] = float(g[~sel].clamp_min(0.0).sum())
        out["active_norm"] = float(r)                                         # ||D||_F^2 of the orthonormal memory (8.3 (4)'s denominator)
        mQ = int(sel.sum())
        if mQ == 0:
            self.c, self.t_ref = c_t, t_now
            self.prune_fifo(cfg, t_now)
            E_mem = self.memory_energy()
            out.update({"m_coll": 0, "S_new": S_old, "S_B": 0.0, "S_B_Z": 0.0, "I_RB": 0.0, "E_B": 0.0, "r_after": r, "turnover": False,
                        "E_mem": E_mem, "L_collar": self.collar_flux(t_now), "fifo_overlap_max": 0.0, "ent_bound_increment": 0.0,
                        "clip_excursion_max": exc_old, "fifo_oldest_weight": self.fifo_oldest_weight, "fifo_depth": self.fifo_depth,
                        "wall_s": time.time() - t0})
            return out
        gst = {}                                                              # v4 (R5): the Gram drops of this block
        Y = X @ (u[:, sel] / torch.sqrt(g[sel]).to(CD)[None, :])
        cY = V.mH @ Y
        cols, halves = OPS.split_by_half(cY, neg)
        cQ, halfQ, _ = OPS.orthonormalize_halves(cols, halves, cfg.eps_orth, negmask=neg)
        # (d) orthogonalize against the recent collected modes
        ov_max = 0.0; ov_after = 0.0
        F = None
        if self.fifo and cfg.fifo_depth > 0:
            F = torch.cat([torch.exp(-1j * e.to(CD) * (t_now - tq))[:, None] * cq for cq, hq, tq in self.fifo], dim=1)
            ov = F.mH @ cQ
            ov_max = float(ov.abs().max()) if ov.numel() else 0.0
            # v2: Q = Q_cand - F (F^dagger Q_cand) by classical Gram-Schmidt twice, re-orthonormalized per half; the normalization of
            # small residuals (repeated content) amplifies rounding along F by 1/|residual|, so the projection is repeated once after it
            cQ = OPS.project_out(cQ, F)
            cQ, halfQ, _ = OPS.orthonormalize_halves(cQ, halfQ, cfg.eps_orth, negmask=neg)
            cQ = OPS.project_out(cQ, F)
            cQ, halfQ, _ = OPS.orthonormalize_halves(cQ, halfQ, cfg.eps_orth, negmask=neg)
            ov_after = float((F.mH @ cQ).abs().max()) if cQ.shape[1] else 0.0
        mQ = int(cQ.shape[1])
        out["fifo_overlap_max"] = ov_max
        out["fifo_overlap_after"] = ov_after
        if mQ == 0:
            self.c, self.t_ref = c_t, t_now
            self.prune_fifo(cfg, t_now)
            out.update({"m_coll": 0, "S_new": S_old, "S_B": 0.0, "S_B_Z": 0.0, "I_RB": 0.0, "E_B": 0.0, "r_after": r, "turnover": False,
                        "E_mem": self.memory_energy(), "L_collar": self.collar_flux(t_now), "ent_bound_increment": 0.0,
                        "clip_excursion_max": exc_old, "fifo_oldest_weight": self.fifo_oldest_weight, "fifo_depth": self.fifo_depth,
                        "wall_s": time.time() - t0})
            return out
        # v4 (R5): the post-split localization of the collected modes: the largest weight inside R_cut of a normalized collected column
        xQ = V @ cQ
        inw = (xQ.abs().pow(2) * (~mask_out).to(RD)[:, None]).sum(dim=0)
        out["post_split_inward_weight_max"] = float(inw.max())
        out["post_split_inward_weight_mean"] = float(inw.mean())
        del xQ
        # (e) the block
        A = cQ.mH @ c_t                                                     # m x r
        AMA = A @ self.MD @ A.mH
        CBB = AMA + torch.diag((halfQ < 0).to(CD))
        S_B, exc_B = OPS.entropy_cov(CBB)
        OmB = cQ.mH @ (e.to(CD)[:, None] * cQ)
        E_B = float(torch.trace(AMA @ OmB).real)
        # (f) the memory update
        c_p = c_t - cQ @ A
        # v2: the memory is orthogonal to Q and (by unitarity) to every FIFO mode; the projection is repeated against both before and
        # after the per-half normalization, which amplifies the rounding of nearly collected directions by 1/|residual|
        G = cQ if F is None else torch.cat([cQ, F], dim=1)
        c_p = OPS.project_out(c_p, G)
        c_new, half_new, _ = OPS.orthonormalize_halves(c_p, self.half, cfg.eps_orth, negmask=neg, stats=gst)
        c_new = OPS.project_out(c_new, G)
        c_new, half_new, _ = OPS.orthonormalize_halves(c_new, half_new, cfg.eps_orth, negmask=neg, stats=gst)
        Dk = c_new.mH @ c_t                                                 # r' x r
        MD_new = Dk @ self.MD @ Dk.mH
        MD_new = 0.5 * (MD_new + MD_new.mH)
        Pi_old = torch.diag((self.half < 0).to(CD)); Pi_new = torch.diag((half_new < 0).to(CD))
        Xi = Pi_new - Dk @ Pi_old @ Dk.mH
        out["Xi_min_eig"] = float(torch.linalg.eigvalsh(0.5 * (Xi + Xi.mH)).min()) if Xi.shape[0] else 0.0
        Yk = torch.eye(Dk.shape[0], dtype=CD, device=self.device) - Dk @ Dk.mH - Xi
        out["I_minus_DD_minus_Xi_min_eig"] = float(torch.linalg.eigvalsh(0.5 * (Yk + Yk.mH)).min()) if Yk.shape[0] else 0.0
        out["Xi_norm"] = float(torch.linalg.norm(Xi))
        # v4 (R5): D8's term of the Gram drops (dim s(g_max)); (31.4 (3)) their energy bound, count x g_max x ||M_Delta||_op x ||h||
        n_gram = int(gst.get("n_dropped", 0))
        if n_gram:
            self.ent_bound_gram += gst.get("dim", r) * OPS.binary_entropy_scalar(gst.get("g_dropped_max", 0.0))
            mdop = float(torch.linalg.matrix_norm(self.MD, ord=2)) if self.MD.shape[0] else 0.0
            eb = n_gram * gst.get("g_dropped_max", 0.0) * mdop * float(e.abs().max())
            self.gram_energy_bound += eb
            if self.rec is not None:
                self.rec["gram_energy_bound"] = self.rec.get("gram_energy_bound", 0.0) + eb
        out["n_gram_dropped"] = n_gram; out["gram_g_dropped_max"] = gst.get("g_dropped_max", 0.0)
        # (g) compression to the split hull of ran Delta above eps_rank
        lam, uu = torch.linalg.eigh(MD_new) if MD_new.shape[0] else (torch.zeros(0, dtype=RD, device=self.device), MD_new)
        keep = lam.abs() > cfg.eps_rank
        n_drop = int((~keep).sum())
        if n_drop:
            # v4 (R5): D8's factor is the compressed space's dimension r' (v3: the dropped count)
            self.ent_bound += int(c_new.shape[1]) * OPS.binary_entropy_scalar(cfg.eps_rank)
        if bool(keep.any()):
            U_k = uu[:, keep]                                               # r' x k
            # the hull: each eigenvector's components on the two halves, orthonormalized per half (v2: in the memory's coordinates)
            c_fin, half_fin = OPS.hull_in_coords(c_new, half_new, U_k, cfg.eps_orth)
            Tf = c_fin.mH @ c_new                                           # r'' x r'
            MD_fin = Tf @ MD_new @ Tf.mH
            MD_fin = 0.5 * (MD_fin + MD_fin.mH)
        else:
            c_fin = c_new[:, :0]; half_fin = half_new[:0]; MD_fin = MD_new[:0, :0]
        # v4 (31.4 (3)): the energy this compression drops, E(MD_new on c_new) - E(MD_fin on c_fin), accumulated over the stage; beside it
        # the energy of the dropped eigen-directions themselves, sum_d lambda_d <u_d|c^dagger e c|u_d> (printed, not used)
        E_tr = self.memory_energy(c_new, MD_new) - self.memory_energy(c_fin, MD_fin)
        if n_drop:
            hn = c_new.mH @ (e.to(CD)[:, None] * c_new)
            ud = uu[:, ~keep]
            E_tr_eig = float((lam[~keep] * torch.einsum("ij,ik,kj->j", ud.conj(), hn, ud).real).sum())
        else:
            E_tr_eig = 0.0
        if self.rec is not None:
            self.rec["E_trunc"] = self.rec.get("E_trunc", 0.0) + E_tr
            self.rec["E_trunc_eigs"] = self.rec.get("E_trunc_eigs", 0.0) + E_tr_eig
        # (h) entropies and the Z route
        S_new, exc_new = OPS.entropy_cov(MD_fin + torch.diag((half_fin < 0).to(CD)))
        M_old = self.MD + Pi_old
        P_old = OPS.sqrt_m_one_minus_m(M_old)
        Dtot = c_fin.mH @ c_t                                               # r'' x r
        M_fin = MD_fin + torch.diag((half_fin < 0).to(CD))
        Z = torch.cat([torch.cat([torch.eye(r, dtype=CD, device=self.device) - M_old, P_old @ Dtot.mH], dim=1),
                       torch.cat([Dtot @ P_old, M_fin], dim=1)], dim=0)
        S_B_Z, exc_Z = OPS.entropy_cov(Z)
        I_RB = S_old + S_B - S_new
        # (j) stage records (the direct route): a_n = c_init^dagger exp(+i e (t_now - t_init)) c_Q
        rec = self.rec
        if rec is not None:
            if rec["n_coll"] + mQ <= cfg.store_cap:
                ph_i = torch.exp(1j * e.to(CD) * (t_now - rec["t_init"]))
                rec["a"].append(rec["c_init"].mH @ (ph_i[:, None] * cQ)); rec["halfQ"].append(halfQ.clone()); rec["tQ"].append(t_now)
                if cfg.store_modes:
                    rec["cQ"].append(cQ.clone())
                rec["n_coll"] += mQ
            rec["n_coll_all"] = rec.get("n_coll_all", 0) + mQ                # v3: every collected mode, recorded or not (the cap)
            rec["E_rad"] += E_B                                               # the diagonal block energies (v3: E_diag; E_rad is the memory route)
        # commit
        self.c, self.half, self.MD, self.t_ref = c_fin, half_fin, MD_fin, t_now
        self.fifo.append((cQ, halfQ, t_now))
        self.prune_fifo(cfg, t_now)
        E_mem = self.memory_energy()
        out.update({"m_coll": mQ, "S_new": S_new, "S_B": S_B, "S_B_Z": S_B_Z, "I_RB": I_RB, "E_B": E_B, "r_after": int(c_fin.shape[1]),
                    "turnover": bool(I_RB > S_B + 1e-9), "E_mem": E_mem, "L_collar": self.collar_flux(t_now), "n_dropped_rank": n_drop,
                    "E_trunc_block": E_tr, "E_trunc_acc": rec.get("E_trunc", 0.0) if rec is not None else None,
                    "E_trunc_eigs_block": E_tr_eig, "E_trunc_eigs_acc": rec.get("E_trunc_eigs", 0.0) if rec is not None else None,
                    "compressed_dim": int(c_new.shape[1]), "ent_bound_increment": (self.ent_bound + self.ent_bound_gram) - bound0,
                    "clip_excursion_max": max(exc_old, exc_B, exc_new, exc_Z), "clip_excursions": [exc_old, exc_B, exc_new, exc_Z],
                    "fifo_oldest_weight": self.fifo_oldest_weight, "fifo_depth": self.fifo_depth, "fifo_entries": len(self.fifo),
                    "wall_s": time.time() - t0})
        return out

    def prune_fifo(self, cfg, t_now):
        """a collected mode leaves the FIFO when its weight within R_cut + 3 tau_s (propagated to t_now) falls below the threshold
        (--fifo-prune-weight, 1e-10). v4 (31.4 (5), R7.3): no entry is dropped above the threshold; when the entries exceed the depth the
        depth doubles (printed and recorded in fifo_doublings); the oldest entry's inward weight is kept in fifo_oldest_weight (the printed
        orthogonality bound to earlier radiation, 1e-4): the largest inward weight of one of its columns (a unit vector's squared overlap
        with that column is at most its weight where the vector lives)."""
        st = self.stage
        e, V = st["e"], st["V"]
        if self.fifo_depth is None:
            self.fifo_depth = int(cfg.fifo_depth)
        if int(cfg.fifo_depth) <= 0:                                          # no FIFO (block_step does not project against it)
            self.fifo = []; self.fifo_exit_weights = []; self.fifo_oldest_weight = 0.0
            return
        keep = []
        zone = (self.R2 > cfg.R_cut_abs) & (self.R2 < cfg.R_cut_abs + 3.0 * cfg.tau_s_abs)
        inside = (self.R2 <= cfg.R_cut_abs)
        exits = []; weights = []
        zi = (zone | inside).to(RD)[:, None]
        for cq, hq, tq in self.fifo:
            x = V @ (torch.exp(-1j * e.to(CD) * (t_now - tq))[:, None] * cq)
            wcol = (x.abs().pow(2) * zi).sum(dim=0)                           # v4: per column (the entry's weight is their sum, v3's criterion)
            wsum = float(wcol.sum())
            if wsum > getattr(cfg, "fifo_prune_weight", 1e-10):
                keep.append((cq, hq, tq)); weights.append(float(wcol.max()) if wcol.numel() else 0.0)
            else:
                exits.append(wsum)
        while self.fifo_depth > 0 and len(keep) > self.fifo_depth:
            self.fifo_depth *= 2
            self.fifo_doublings.append((float(t_now), int(self.fifo_depth)))
            print("FIFO (kappa %d): %d entries above the weight threshold at t %.2f exceed the depth; the depth doubles to %d (31.4 (5))" % (
                self.kappa, len(keep), t_now, self.fifo_depth), flush=True)
        self.fifo = keep
        self.fifo_exit_weights = exits
        self.fifo_oldest_weight = float(weights[0]) if weights else 0.0

    # ---------------------------------------------------------------------------------------------- stage covariance
    def stage_direct(self, cfg, with_spectrum=False):
        """the stage's collected covariance G_R = Pi_neg + A^dagger M_Delta,init A and its entropy (the direct route), plus the
        energy-resolved occupations when the collected modes are stored and with_spectrum is set."""
        rec = self.rec
        if rec is None or not rec["a"]:
            return None
        A = torch.cat(rec["a"], dim=1)                                       # r0 x m
        halfQ = torch.cat(rec["halfQ"])
        G = A.mH @ rec["MD_init"] @ A
        Gfull = G + torch.diag((halfQ < 0).to(CD))
        S_direct, exc = OPS.entropy_cov(Gfull)
        out = {"S_direct": S_direct, "m": int(A.shape[1]), "excursion": exc,
               "complete": int(rec["n_coll"]) == int(rec.get("n_coll_all", rec["n_coll"]))}     # v3: the record holds every collected mode
        if with_spectrum and cfg.store_modes and rec["cQ"]:
            e = self.stage["e"].to(CD)
            # v2: the stored modes start at block k0 (collected mode m0) after a resume (the checkpoint does not carry them); v1 zipped
            # them with the stage's first collection times and indexed the stage's half labels from 0 (wrong phases after a resume)
            k0, m0 = rec.get("cQ_from", (0, 0))
            cQ = torch.cat([torch.exp(-1j * e * (rec["t_init"] - tq))[:, None] * cq for cq, tq in zip(rec["cQ"], rec["tQ"][k0:])], dim=1)
            sub = torch.arange(m0, m0 + cQ.shape[1], device=self.device)
            halfQ = halfQ[sub]; G = G[sub][:, sub]
            # all collected modes at the common time t_init; Gram check and the energy matrix within each half
            Gram = cQ.mH @ cQ
            out["gram_defect"] = float(torch.linalg.norm(Gram - torch.eye(Gram.shape[0], dtype=CD, device=self.device)))
            Om = cQ.mH @ (e[:, None] * cQ)
            TH = 1.0 / (4.0 * math.pi * self.stage["rs"])
            spec = []
            for hsign in (1, -1):
                idx = (halfQ == hsign).nonzero().ravel()
                if idx.numel() == 0:
                    continue
                Oh = Om[idx][:, idx]
                om, Uo = torch.linalg.eigh(0.5 * (Oh + Oh.mH))
                Gh = Uo.mH @ G[idx][:, idx] @ Uo
                occ = torch.diagonal(Gh).real                                 # excess occupation (particles > 0 half; -deficit on the hole half)
                for k in range(om.shape[0]):
                    spec.append((float(om[k]) / TH, float(occ[k]) * (1 if hsign > 0 else -1), int(hsign)))
            out["spectrum_omega_over_TH_occupation_half"] = spec
        return out

    def energy_identity(self, cfg):
        """v3 (30.1 (3)): the within-stage conservation tr[Delta_k h_k] = E_mem + E_coll + E_x from the stage record (the Heisenberg frame
        of the stage's initial memory), at the memory's current time t_ref: with q_n the collected modes propagated to t_ref,
        A = [a_1 ... a_N] (their coefficients on the initial memory), d the memory and B = c_init^dagger exp(+i e (t_ref - t_init)) d,
        E_coll = tr[(A^dagger M_init A)(q^dagger e q)] (every block pair), E_x = 2 Re tr[(A^dagger M_init B)(d^dagger e q)],
        E_mem = tr[M_Delta (d^dagger e d)]. None unless the record holds every collected mode of the stage and the stored modes."""
        rec = self.rec
        if rec is None or not rec["a"] or not cfg.store_modes or not rec["cQ"]:
            return None
        if int(rec["n_coll"]) != int(rec.get("n_coll_all", rec["n_coll"])) or tuple(rec.get("cQ_from", (0, 0))) != (0, 0):
            return None
        e = self.stage["e"].to(CD)
        A = torch.cat(rec["a"], dim=1)
        q = torch.cat([torch.exp(-1j * e * (self.t_ref - tq))[:, None] * cq for cq, tq in zip(rec["cQ"], rec["tQ"])], dim=1)
        if q.shape[1] != A.shape[1]:
            return None
        MDi = rec["MD_init"]
        E_coll = float(torch.trace((A.mH @ MDi @ A) @ (q.mH @ (e[:, None] * q))).real)
        if self.c.shape[1]:
            B = rec["c_init"].mH @ (torch.exp(1j * e * (self.t_ref - rec["t_init"]))[:, None] * self.c)
            E_x = 2.0 * float(torch.trace((A.mH @ MDi @ B) @ (self.c.mH @ (e[:, None] * q))).real)
            E_mem_exact = float(torch.trace((B.mH @ MDi @ B) @ (self.c.mH @ (e[:, None] * self.c))).real)   # v4: the exact energy in the memory's span
        else:
            E_x = 0.0; E_mem_exact = 0.0
        E_mem = self.memory_energy()
        E_init = rec["E_delta_init"]
        E_tr = rec.get("E_trunc", 0.0)
        # v4 (31.4 (3)): the completed identity, E_init = E_mem + E_coll + E_x + E_trunc,acc (the criterion); the uncompleted residual is
        # the truncation's energy share, printed beside it and not a criterion
        return {"E_init": E_init, "E_mem": E_mem, "E_coll": E_coll, "E_x": E_x, "E_trunc": E_tr, "E_diag": rec["E_rad"], "E_rad_memory_route": E_init - E_mem,
                "residual_rel": (E_init - E_mem - E_coll - E_x - E_tr) / max(abs(E_init), 1e-300),
                "uncompleted_residual_rel": (E_init - E_mem - E_coll - E_x) / max(abs(E_init), 1e-300),
                # v4 (printed, not a criterion): the coverage identity with the exact energy in the memory's span (the stage-initial memory
                # restricted to it) in place of E_mem; E_mem - E_mem,exact is the truncations' energy share inside the memory's span
                "E_mem_exact_span": E_mem_exact, "coverage_residual_rel": (E_init - E_mem_exact - E_coll - E_x) / max(abs(E_init), 1e-300),
                "memory_mismatch_rel": (E_mem - E_mem_exact) / max(abs(E_init), 1e-300), "E_trunc_eigs": rec.get("E_trunc_eigs", 0.0),
                "gram_energy_bound_rel": rec.get("gram_energy_bound", 0.0) / max(abs(E_init), 1e-300), "n_modes": int(A.shape[1])}

    # --------------------------------------------------------------------------------------------------- transition
    def continue_reference(self, cfg, EO, st_old, st_new, v_old, v_new, tau):
        """v4 (31.4 (1)): P^cont_{k+1}, the engine's continuation of the stage reference through the ramp: engine_ops_v3's
        assign_max_overlap_v3 (tol_block 1e-8, tol_fallback 1e-6, the predecessor's mask) at every sub-step s_j = j/nsub of the segment
        h[1, v_old + s (v_new - v_old)] (the ramp's image; the last sub-step is the new stage), continue_sea_v3's Weyl certificate per
        sub-step (verified while 2 dh < the previous record's occupied-to-empty gap between blocks, dh = ||dH||_G/nsub; the inter-block
        isolation beside), the sub-step count by S9 (a)'s 1/8 rule (nsub = ceil(8 ||dH||_G/g), g the old record's gap; capped at
        --nsub-max and flagged), each sub-step's ||Delta h||/g and ||h_dot||/g^2 (||h_dot|| = ||dH||_G max d sigma/dt over the sub-step)
        printed and marked unresolved above 1/8. Rotated blocks of the last assignment are applied to the new stage's eigenbasis.
        Returns the record with the mask (over the new eigenbasis, energy order)."""
        t0 = time.time()
        e_old, V_old, mask_old = st_old["e"], st_old["V"], st_old["neg"]
        n_sea = int(mask_old.sum())
        dH = (st_new["h"] - st_old["h"]).tocsr()
        nG = OPS.gershgorin(dH)
        pm = mask_old.cpu().numpy().astype(bool)
        ref = V_old[:, mask_old]
        _, info0, _ = EO.assign_max_overlap_v3(ref, e_old, V_old, n_sea, prev_mask=pm)
        g0 = info0.get("weyl_gap"); gi = info0.get("gap_between_blocks")
        g_old = g0
        nsub_need = int(math.ceil(8.0 * nG / g0)) if (g0 is not None and g0 > 0) else None
        nsub_max = int(getattr(cfg, "nsub_max", 64))
        capped = (nsub_need is None) or (nsub_need > nsub_max)
        nsub = nsub_max if capped else max(1, nsub_need)
        dh = nG / nsub
        verified = True; iso = True; margin = math.inf
        rows = []; eig_s = 0.0; assign_s = 0.0
        ref_prev_last = ref
        info = info0; mask = pm
        for k in range(1, nsub + 1):
            s_ = k / nsub
            if k == nsub:
                e_s, U_s = st_new["e"], st_new["V"]
            else:
                t1 = time.time()
                h_s = OPS.clocked_csr(self.h0, self.Q, np.ones(len(self.R)), v_old + s_ * (v_new - v_old))
                e_s, U_s = OPS.eigh_csr(h_s, self.device)
                eig_s += time.time() - t1
            if g0 is None or 2.0 * dh >= g0:
                verified = False
            if gi is None or 2.0 * dh >= gi:
                iso = False
            if g0 is not None:
                margin = min(margin, g0 - 2.0 * dh)
            r1 = (dh / g0) if (g0 is not None and g0 > 0) else math.inf
            r2 = (nG * OPS.ramp_rate_max((k - 1) / nsub, k / nsub) / tau / g0 ** 2) if (g0 is not None and g0 > 0) else math.inf
            ref_prev_last = ref
            t1 = time.time()
            ref, info, mask = EO.assign_max_overlap_v3(ref, e_s, U_s, n_sea, prev_mask=pm)
            assign_s += time.time() - t1
            rows.append({"k": k, "s": s_, "gap_before": g0, "dh_over_gap": r1, "hdot_over_gap2": r2, "unresolved": bool(r1 > 0.125 or r2 > 0.125),
                         "weyl_gap_after": info["weyl_gap"], "inversions": info["inversions"], "n_pocket": info["n_pocket"],
                         "ambiguous_tie": bool(info["ambiguous_tie"]), "n_intermediate": info["n_intermediate"], "rotated_blocks": info["rotated_blocks"]})
            g0 = info["weyl_gap"]; gi = info["gap_between_blocks"]; pm = mask
            if k < nsub:
                del U_s
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
        del ref
        # the last assignment's rotated (numerically degenerate) blocks become the new stage's eigenbasis there, so P^cont is its mask
        rot = []
        if info.get("rotated_blocks"):
            e_np = st_new["e"].cpu().numpy().astype(float)
            Vn = st_new["V"]
            A = ref_prev_last.mH @ Vn
            for (i, j) in EO.blocks_of_v3(e_np, info["tol_block"]):
                if j - i > 1:
                    M = A[:, i:j].mH @ A[:, i:j]
                    _, W = torch.linalg.eigh(0.5 * (M + M.mH))
                    Vn[:, i:j] = Vn[:, i:j] @ W
                    rot.append((int(i), int(j), float(e_np[j - 1] - e_np[i])))
            del A
        del ref_prev_last
        n_unres = int(sum(1 for r_ in rows if r_["unresolved"]))
        certified = bool(verified and n_unres == 0 and not capped)
        reasons = []
        if not verified:
            reasons.append("Weyl certificate not verified (2 dh >= the gap at some sub-step)")
        if n_unres:
            reasons.append("%d of %d sub-steps above S9 (a)'s 1/8" % (n_unres, nsub))
        if capped:
            reasons.append("sub-step count capped at %d (the 1/8 rule needs %s)" % (nsub_max, nsub_need if nsub_need is not None else "a positive gap"))
        if rot:
            reasons.append("%d rotated blocks (largest width %.2e)" % (len(rot), max(w_ for _, _, w_ in rot)))
        keys = ("n_occ", "n_intermediate", "min_occupied_weight", "max_empty_weight", "inversions", "n_pocket", "pockets_occupied_positive",
                "holes_empty_negative", "nneg", "gap_between_blocks", "weyl_gap", "partly_occupied_blocks", "rotated_blocks", "straddling_blocks",
                "sign_free_vectors", "tol_block", "ambiguous_tie", "tie_by_predecessor", "empty_weight_sum")
        return {"mask": torch.as_tensor(np.asarray(mask, bool), dtype=torch.bool, device=self.device), "certified": certified,
                "label": "certified" if certified else "flagged: " + "; ".join(reasons), "verified": bool(verified), "isolation_verified": bool(iso),
                "capped": bool(capped), "nsub": int(nsub), "nsub_needed": nsub_need, "norm_dH_G": nG, "gap_old": g_old, "dh": dh,
                "margin_min": (margin if math.isfinite(margin) else None), "unresolved_substeps": n_unres,
                "unresolved_fraction": n_unres / max(nsub, 1), "max_dh_over_gap": max(r_["dh_over_gap"] for r_ in rows),
                "max_hdot_over_gap2": max(r_["hdot_over_gap2"] for r_ in rows), "rows": rows, "rot_blocks": rot,
                "final": {k_: (bool(v_) if isinstance(v_, (bool, np.bool_)) else v_) for k_, v_ in info.items() if k_ in keys},
                "reference_offset_F2": info.get("n_pocket"), "substep_eigh_s": eig_s, "assign_s": assign_s, "seconds": time.time() - t0,
                "engine_dir": getattr(EO, "ENGINE_DIR", None)}

    def transition(self, cfg, E_new, Rb_new, tau_factor=1.0, dt_factor=1.0):
        """v4 (31.4 (1); R3, R7.1): the ramp v_k -> v_{k+1} over tau_tr; the new stage; P^cont_{k+1} by continue_reference; the complete
        occupied sea of the stage reference (n_sea columns, chunks of --tr-chunk), the memory and the FIFO propagated through the ramp by
        CF4; the new memory C_exact - P^cont_{k+1} = (U P_k U^dagger - P^cont) + U Delta_k U^dagger on one split basis of the new halves,
        projected out of the propagated FIFO (31.4 (5)), compressed at eps_rank; the stage reference becomes the continued mask. Every
        dropped piece is printed: the sea part's dropped pairs (share, entropy bound), the memory's weight on the FIFO's split hull, the
        compression (energy, share, entropy bound). dt_factor halves the CF4 step for T4i's DIAG. Returns the record."""
        st_old = self.stage
        t0 = time.time()
        EO = engine_ops_v3(getattr(cfg, "engine_path", None))
        rs_old = st_old["rs"]; rs_new = 2.0 * cfg.g * E_new
        v_old = st_old["v_sites"]
        v_new = np.asarray(self.averager(OPS.flow_dense(rs_new, Rb_new)), float)
        tau = cfg.tau_tr * rs_old * float(tau_factor)                         # v3: tau_factor 2, 4, ... when 8.3 (8) doubles tau_tr
        dt = cfg.dt_tr * float(dt_factor)
        vfun_t = lambda t: v_old + (v_new - v_old) * OPS.ramp_c1(t / tau)
        S_old, _ = self.memory_entropy()
        E_mem_old = self.memory_energy()
        nD_old = float(torch.linalg.norm(self.MD)) if self.MD.numel() else 0.0
        e_old, V_old, ref_old = st_old["e"], st_old["V"], st_old["neg"]
        n_sea = int(ref_old.sum())
        t_start = self.t_ref
        # (b) the new stage and the continuation of the reference (before the sea is propagated: the device holds one of them at a time)
        st_new = self.build_stage(cfg, E_new, Rb_new, label="stage")
        cont = self.continue_reference(cfg, EO, st_old, st_new, v_old, v_new, tau)
        occ = cont["mask"]
        e_new, V_new = st_new["e"], st_new["V"]
        st_new["neg"] = occ; st_new["nneg"] = int(occ.sum()); st_new["rot_blocks"] = cont["rot_blocks"]
        st_new["reference"] = "continued: P^cont through the ramp by engine_ops_v3's rule (%s)" % cont["label"]
        # (a) the columns through the ramp: the memory, the FIFO (each entry taken at t_ref from its own frame) and the complete sea
        D = V_old @ self.c
        Fcols = [V_old @ (torch.exp(-1j * e_old.to(CD) * (t_start - tq))[:, None] * cq) for cq, hq, tq in self.fifo]
        Fhalves = [hq for cq, hq, tq in self.fifo]
        nF = [int(f.shape[1]) for f in Fcols]
        small = torch.cat([D] + Fcols, dim=1)
        cs0 = OPS.cheb_stats()
        t_cf = time.time()
        if small.shape[1]:
            small = OPS.cf4_propagate(self.B, small, vfun_t, 0.0, tau, dt)
        chunk = max(1, int(getattr(cfg, "tr_chunk", 4096)))
        occ_old_idx = ref_old.nonzero().ravel()
        parts = []
        for j0 in range(0, n_sea, chunk):
            parts.append(OPS.cf4_propagate(self.B, V_old[:, occ_old_idx[j0: j0 + chunk]], vfun_t, 0.0, tau, dt))
        Psi = torch.cat(parts, dim=1) if len(parts) > 1 else (parts[0] if parts else V_old[:, :0].clone())
        del parts
        cf4_s = time.time() - t_cf
        cheb = OPS.cheb_delta(cs0)
        D_r = small[:, : D.shape[1]]
        F_r = []; off = D.shape[1]
        for n_ in nF:
            F_r.append(small[:, off: off + n_]); off += n_
        del small, Fcols, D
        # (c) the new memory: C_exact - P^cont = (U P_k U^dagger - P^cont) + U Delta_k U^dagger
        t_pa = time.time()
        Psi_c = V_new.mH @ Psi
        del Psi
        sea_unit = float(torch.linalg.norm(Psi_c.mH @ Psi_c - torch.eye(n_sea, dtype=CD, device=self.device))) if n_sea else 0.0
        leak_occ = Psi_c[~occ].abs().pow(2).sum(dim=0) if n_sea else torch.zeros(0, dtype=RD, device=self.device)
        c_sea, half_sea, MD_sea, lam_sea, pinfo = principal_delta(Psi_c, occ, cfg.eps_form, cfg.eps_orth, self.device, method="sine")
        del Psi_c
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        pa_s = time.time() - t_pa
        self.ent_bound += pinfo["entropy_bound_dropped_pairs"]
        sea_norm = pinfo["norm_Delta_from_sines"]                            # ||U P_k U^dagger - P^cont||_F = (2 sum s^2)^(1/2)
        c_raw = V_new.mH @ D_r
        cols_m, halves_m = OPS.split_by_half(c_raw, occ)
        gst = {}
        c_all, half_all, _ = OPS.orthonormalize_halves(torch.cat([c_sea, cols_m], dim=1), torch.cat([half_sea, halves_m]), cfg.eps_orth,
                                                       negmask=occ, stats=gst)
        Ts = c_all.mH @ c_sea; Tm = c_all.mH @ c_raw
        MD_sea_all = Ts @ MD_sea @ Ts.mH
        MD_all = MD_sea_all + Tm @ self.MD @ Tm.mH
        MD_all = 0.5 * (MD_all + MD_all.mH)
        Xi_tr = float(torch.linalg.norm(MD_sea_all)) if MD_sea_all.numel() else 0.0
        E_all = self.memory_energy(c_all, MD_all)
        E_sea = self.memory_energy(c_all, 0.5 * (MD_sea_all + MD_sea_all.mH)) if MD_sea_all.numel() else 0.0
        nMD_all = float(torch.linalg.norm(MD_all)) if MD_all.numel() else 0.0
        del c_raw, cols_m, D_r
        # (d) the FIFO through the ramp (31.4 (5)): each entry in the new eigenbasis, split by the new halves, orthonormal per half and
        # against the earlier entries; the memory projected out of it
        t_new = t_start + tau
        new_fifo = []; hd_max = 0.0; hd_sum = 0.0; n_before = sum(nF); Fbasis = None
        n_flip = 0
        for Fr, hq in zip(F_r, Fhalves):
            cF = V_new.mH @ Fr
            wocc = (cF.abs().pow(2) * occ.to(RD)[:, None]).sum(dim=0)
            minor = torch.where(hq < 0, 1.0 - wocc, wocc)
            hd_max = max(hd_max, float(minor.max()) if minor.numel() else 0.0); hd_sum += float(minor.sum())
            n_flip += int((minor >= 0.5).sum())
            # each propagated column keeps its labelled half (its dominant half while the half defect is below 1/2); the minor half, the
            # half defect, is dropped and printed (its split hull would promote 1e-8-weight parts to unit directions and the projection
            # below would then remove the memory's partner content: builder's self-test T4j, 0.93 of ||M_Delta||_F removed)
            colsF, halvesF = OPS.split_by_half(cF, occ)
            kF = int(cF.shape[1])
            pick = torch.where(hq > 0, torch.arange(kF, device=self.device), kF + torch.arange(kF, device=self.device))
            cq2, hq2, _ = OPS.orthonormalize_halves(colsF[:, pick], hq.clone(), cfg.eps_orth, negmask=occ)
            if Fbasis is not None and cq2.shape[1]:
                cq2 = OPS.project_out(cq2, Fbasis)
                cq2, hq2, _ = OPS.orthonormalize_halves(cq2, hq2, cfg.eps_orth, negmask=occ)
                cq2 = OPS.project_out(cq2, Fbasis)
                cq2, hq2, _ = OPS.orthonormalize_halves(cq2, hq2, cfg.eps_orth, negmask=occ)
            if cq2.shape[1]:
                new_fifo.append((cq2, hq2, t_new))
                Fbasis = cq2 if Fbasis is None else torch.cat([Fbasis, cq2], dim=1)
        del F_r
        n_after = int(Fbasis.shape[1]) if Fbasis is not None else 0
        removed_F2 = 0.0; removed_E = 0.0
        if Fbasis is not None and c_all.shape[1]:
            c_p = OPS.project_out(c_all, Fbasis)
            c_p, half_p, _ = OPS.orthonormalize_halves(c_p, half_all, cfg.eps_orth, negmask=occ, stats=gst)
            c_p = OPS.project_out(c_p, Fbasis)
            c_p, half_p, _ = OPS.orthonormalize_halves(c_p, half_p, cfg.eps_orth, negmask=occ, stats=gst)
            Tp = c_p.mH @ c_all
            MD_p = Tp @ MD_all @ Tp.mH
            MD_p = 0.5 * (MD_p + MD_p.mH)
            removed_F2 = nMD_all ** 2 - (float(torch.linalg.norm(MD_p)) ** 2 if MD_p.numel() else 0.0)
            removed_E = E_all - self.memory_energy(c_p, MD_p)
        else:
            c_p, half_p, MD_p = c_all, half_all, MD_all
        n_gram = int(gst.get("n_dropped", 0))
        if n_gram:
            self.ent_bound_gram += gst.get("dim", 0) * OPS.binary_entropy_scalar(gst.get("g_dropped_max", 0.0))
        # (e) compression at eps_rank (as in the block step), the dropped energy, share and entropy bound printed
        lam, uu = torch.linalg.eigh(MD_p) if MD_p.shape[0] else (torch.zeros(0, dtype=RD, device=self.device), MD_p)
        keep = lam.abs() > cfg.eps_rank
        n_drop = int((~keep).sum())
        comp_sq = float((lam[~keep] ** 2).sum()) if n_drop else 0.0
        if n_drop:
            self.ent_bound += int(c_p.shape[1]) * OPS.binary_entropy_scalar(cfg.eps_rank)     # R5: D8's factor r' s(eps_rank)
        if bool(keep.any()):
            U_k = uu[:, keep]
            c_fin, half_fin = OPS.hull_in_coords(c_p, half_p, U_k, cfg.eps_orth)
            Tf = c_fin.mH @ c_p
            MD_fin = Tf @ MD_p @ Tf.mH
            MD_fin = 0.5 * (MD_fin + MD_fin.mH)
        else:
            c_fin = c_p[:, :0]; half_fin = half_p[:0]; MD_fin = MD_p[:0, :0]
        E_trunc_tr = self.memory_energy(c_p, MD_p) - self.memory_energy(c_fin, MD_fin)
        # commit: the memory at the end of the ramp, t_ref_old + tau (global timestamps, v2), the propagated FIFO, the continued reference
        self.c, self.half, self.MD, self.t_ref = c_fin, half_fin, MD_fin, t_new
        self.fifo = new_fifo
        S_new, _ = self.memory_entropy()
        self.init_stage_record()
        self.n_transitions += 1
        dropped_F2 = 2.0 * pinfo["dropped_sq"] + comp_sq + max(removed_F2, 0.0)
        return {"tau_factor": float(tau_factor), "dt_factor": float(dt_factor), "tau_tr": tau, "dt_tr": dt, "rs_old": rs_old, "rs_new": rs_new,
                "E_new": float(E_new), "n_sea": n_sea, "n_neg_old": int(st_old["neg_sign"].sum()), "n_neg_new": int((e_new < 0).sum()),
                "continuation": {k_: v_ for k_, v_ in cont.items() if k_ != "mask"}, "continuation_label": cont["label"],
                "reference_offset_F2": cont["reference_offset_F2"], "continuation_margin": cont["margin_min"],
                "released_pushed_up": cont["final"].get("pockets_occupied_positive"), "released_new_pockets": cont["final"].get("holes_empty_negative"),
                "ambiguous_continuations": int(cont["final"].get("n_intermediate") or 0),
                "adiabatic_leak_max": float(leak_occ.max()) if leak_occ.numel() else 0.0, "adiabatic_leak_mean": float(leak_occ.mean()) if leak_occ.numel() else 0.0,
                "adiabatic_leak_sum": float(leak_occ.sum()) if leak_occ.numel() else 0.0, "sea_unitarity": sea_unit,
                "sea_coherence_F": sea_norm, "sea_coherence_over_memory_Delta": sea_norm / max(nD_old, 1e-300),
                "sea_coherence_over_Delta0": (sea_norm / self.norm_Delta0) if self.norm_Delta0 else None,
                "sea_sin_max": pinfo["sin_theta_max"], "sea_pairs_kept": (pinfo["rank_kept"]) // 2, "sea_dropped_pairs": pinfo["n_dropped_pairs"],
                "sea_dropped_pair_share": pinfo["dropped_pair_share"], "sea_entropy_bound_dropped_pairs": pinfo["entropy_bound_dropped_pairs"],
                "sea_energy": E_sea, "memory_energy_before": E_mem_old, "memory_energy_after_assembly": E_all,
                "fifo_entries": len(new_fifo), "fifo_columns_before": n_before, "fifo_columns_after": n_after, "fifo_half_defect_max": hd_max,
                "fifo_half_defect_sum": hd_sum, "fifo_half_flips": n_flip, "memory_removed_on_fifo_F2": removed_F2, "memory_removed_on_fifo_E": removed_E,
                "memory_removed_on_fifo_share": math.sqrt(max(removed_F2, 0.0)) / max(nMD_all, 1e-300),
                "compression_n_dropped": n_drop, "compression_E_trunc": E_trunc_tr, "compression_dropped_share": math.sqrt(comp_sq) / max(nMD_all, 1e-300),
                "transition_dropped_share": math.sqrt(dropped_F2) / max(nMD_all, 1e-300), "Delta_after_F": nMD_all,
                "n_gram_dropped": n_gram, "ent_bound_after": self.ent_bound, "ent_bound_gram_after": self.ent_bound_gram,
                "memory_Delta_norm_before": nD_old, "S_before": S_old, "S_after": S_new, "S_continuity_defect": S_new - S_old,
                "Xi_tr_norm": Xi_tr, "memory_rank": int(c_fin.shape[1]), "n_dropped_rank": n_drop, "r_t_new": st_new["r_t"], "r_in_new": st_new["r_in"],
                "chebyshev": cheb, "timings": {"cf4_full_sea_and_memory_s": cf4_s, "eigh_new_stage_s": st_new["eigh_s"],
                                               "substep_eigh_s": cont["substep_eigh_s"], "substep_eighs": max(cont["nsub"] - 1, 0),
                                               "assign_s": cont["assign_s"], "principal_angles_s": pa_s, "total_s": time.time() - t0},
                "eigh_s": st_new["eigh_s"], "wall_s": time.time() - t0}

    # ------------------------------------------------------------------------------------------------- diagnostics
    def partner_readout(self, cfg, n_modes=4):
        """D7 for the first collected modes of the last checkpoint: w = Delta q - n q, ||w||^2 against n(1 - n), the partner's weights in
        the core (r < r_in), the annulus, outside r_t, beyond R_cut, and its spectral halves."""
        if not self.fifo:
            return []
        st = self.stage
        cq, hq, tq = self.fifo[-1]
        e, V = st["e"], st["V"]
        rows = []
        # the memory at the collection time t_q is the memory before that block's projection; after the commit self.c is the projected
        # memory at t_ref = t_q, so Delta q (q collected) is reconstructed from the stage record's initial memory instead
        rec = self.rec
        if rec is None:
            return []
        ph = torch.exp(-1j * e.to(CD) * (tq - rec["t_init"]))
        c_q = ph[:, None] * rec["c_init"]                                      # the stage's initial memory propagated to t_q
        for j in range(min(n_modes, cq.shape[1])):
            q = cq[:, j: j + 1]
            a = c_q.mH @ q                                                    # r0 x 1
            n_exc = float((a.mH @ rec["MD_init"] @ a).real)
            n = n_exc + (1.0 if int(hq[j]) < 0 else 0.0)
            wv = c_q @ (rec["MD_init"] @ a) - n_exc * q                        # w = C q - n q = Delta q - n_exc q (the vacuum part is exactly 0 or 1)
            x = V @ wv
            rho = x.abs().pow(2).ravel()
            R2 = self.R2
            rin = st["r_in"] or 0.0; rt = st["r_t"] or 0.0
            rows.append({"label": "within stage 1" if self.n_transitions == 0 else "earlier stages' radiation omitted",   # v3 (30.1 (6))
                         "n": n, "norm2_w": float(rho.sum()), "n_1_minus_n": n * (1.0 - n),
                         "w_core": float(rho[R2 < rin].sum()), "w_annulus": float(rho[(R2 >= rin) & (R2 < rt)].sum()),
                         "w_outside_rt": float(rho[(R2 >= rt) & (R2 <= cfg.R_cut_abs)].sum()), "w_beyond_cut": float(rho[R2 > cfg.R_cut_abs].sum()),
                         "w_negative_half": float(wv[st["neg"]].abs().pow(2).sum())})
        return rows

    def cross_surface(self, cfg, ds=(2, 4, 8), width=4, t=None):
        """I(X:Y) between slabs just inside and just outside r_t for the physical state and for the Killing vacuum, with 2||K||_op^2.
        v4 (31.4 (2), (9); R7.2): the physical state is the exact stage state at t (default the memory's time), P_k + Delta(t) with
        Delta(t) from the stage-initial memory propagated (stage_state), not the retained block; the vacuum is the stage reference P_k
        (after a transition the continued reference)."""
        st = self.stage
        if st["r_t"] is None:
            return []
        e, V, neg = st["e"], st["V"], st["neg"]
        t = self.t_ref if t is None else t
        Dst, Mst = self.stage_state(t)
        out = []
        for d in ds:
            X = np.where((self.R >= st["r_t"] - d - width) & (self.R < st["r_t"] - d))[0]
            Y = np.where((self.R >= st["r_t"] + d) & (self.R < st["r_t"] + d + width))[0]
            if len(X) == 0 or len(Y) == 0:
                continue
            ix = np.concatenate([[2 * s, 2 * s + 1] for s in X]); iy = np.concatenate([[2 * s, 2 * s + 1] for s in Y])
            idx = torch.as_tensor(np.concatenate([ix, iy]), dtype=torch.int64, device=self.device)
            rowsV = V[idx, :]                                                   # k x n2
            Cvac = rowsV[:, neg] @ rowsV[:, neg].mH
            Dm = Dst[idx, :]                                                    # k x r0 (the exact stage state's excess on the slabs)
            Cphys = Cvac + Dm @ Mst @ Dm.mH
            nx = len(ix)
            def mi(Cm):
                SX, _ = OPS.entropy_cov(Cm[:nx, :nx]); SY, _ = OPS.entropy_cov(Cm[nx:, nx:]); SXY, _ = OPS.entropy_cov(Cm)
                K = Cm[:nx, nx:]
                kop = float(torch.linalg.matrix_norm(K, ord=2))
                return {"I": SX + SY - SXY, "bound_2K2": 2.0 * kop * kop, "K_op": kop}
            out.append({"d": d, "physical": mi(Cphys), "reference_vacuum": mi(Cvac), "t": t})
        return out
