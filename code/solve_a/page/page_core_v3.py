#!/usr/bin/env python
"""page_core_v3.py. Changes against page_core_v2.py (builder analyst, BRIEF_page_v3_builder.md), each with the item of
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
import math
import time

import numpy as np

import torch

import page_ops_v3 as OPS

CD = OPS.CD
RD = OPS.RD


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

    # ------------------------------------------------------------------------------------------------------------ stage
    def build_stage(self, cfg, E, Rb, label=""):
        t0 = time.time()
        rs = 2.0 * cfg.g * E
        vfun = OPS.flow_dense(rs, Rb)
        v_sites = np.asarray(self.averager(vfun), float)
        h = OPS.clocked_csr(self.h0, self.Q, np.ones(len(self.R)), v_sites)
        e, V = OPS.eigh_csr(h, self.device)
        st = {"E": float(E), "rs": float(rs), "Rb": float(Rb), "v_sites": v_sites, "h": h, "e": e, "V": V,
              "neg": e < 0, "nneg": int((e < 0).sum()), "nzero": int((e.abs() < 1e-12).sum()), "label": label}
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
        inversion readout of spec 3.2."""
        st = self.stage
        V, e, neg = st["V"], st["e"], st["neg"]
        notneg = ~neg
        Phi = V.mH @ Phi0                                                   # eigenbasis coefficients of the formation state (n2 x n0)
        Pp, Pm = Phi[notneg], Phi[neg]                                      # the sine block V_+^dagger Phi_0 and W = Phi_1^dagger Phi_0
        n0, n1 = Phi0.shape[1], int(neg.sum())
        n2 = Phi.shape[0]
        U, S, Vh = torch.linalg.svd(Pm, full_matrices=True)                 # the cosine SVD of spec 4.3: W = U Sigma V^dagger
        Vr = Vh.mH
        nmin = min(n0, n1)
        sig = S.clamp(0.0, 1.0)
        # v2: the small angles (sigma >= 1/sqrt 2) take their sines and their principal vectors from the SVD of the sine block on their
        # right subspace (1 - sigma^2 cancels below sin theta ~ 1e-8); the large angles keep the cosine SVD's pairs and sin = (1 - sigma^2)^(1/2)
        small = sig >= 1.0 / math.sqrt(2.0)
        iS = small.nonzero().ravel(); iL = (~small).nonzero().ravel()
        s_parts, c_parts, up_parts, bm_parts = [], [], [], []               # per kept pair: sin, cos, u_+ (+ half rows), b (- half rows)
        n_dropped_pairs = 0; dropped = 0.0; sin_max = 0.0
        dropped_sq = 0.0; bound_pairs = 0.0; total_sq = 0.0
        if iS.numel():
            VS = Vr[:, iS]
            Us, Ss, Whs = torch.linalg.svd(Pp @ VS, full_matrices=False)    # sines of the small angles; directions beyond min(npos, |S|) have sine 0
            vS = VS @ Whs.mH                                                 # the refined right principal vectors (n0 x m)
            bS = Pm @ vS                                                     # their negative-half parts, of norm cos theta >= 1/sqrt 2
            cS = torch.linalg.norm(bS, dim=0)
            keepS = Ss > cfg.eps_form
            n_dropped_pairs += int(iS.numel()) - int(keepS.sum())
            total_sq += float((Ss * Ss).sum())
            if bool((~keepS).any()):
                sd = Ss[~keepS]
                dropped = float(sd.max())
                dropped_sq = float((sd * sd).sum())
                bound_pairs = OPS.binary_entropy_np((sd * sd).cpu().numpy())    # v3 (30.1 (2)): sum over the dropped pairs of s(s^2)
            if Ss.numel():
                sin_max = max(sin_max, float(Ss.max()))
            if bool(keepS.any()):
                s_parts.append(Ss[keepS]); c_parts.append(cS[keepS]); up_parts.append(Us[:, keepS])
                bm_parts.append(bS[:, keepS] / cS[keepS].to(CD)[None, :])
        if iL.numel():
            vL = Vr[:, iL]
            cL = sig[iL]
            sL = torch.sqrt((1.0 - cL * cL).clamp_min(0.0))                  # >= 1/sqrt 2, no cancellation
            aP = Pp @ vL                                                     # positive-half parts, of norm sin theta
            s_parts.append(sL); c_parts.append(cL); up_parts.append(aP / torch.linalg.norm(aP, dim=0).to(CD)[None, :]); bm_parts.append(U[:, iL])
            sin_max = max(sin_max, float(sL.max()))
            total_sq += float((sL * sL).sum())
        vecs, lams = [], []
        n_kept_pairs = 0
        if s_parts:
            s = torch.cat(s_parts); c = torch.cat(c_parts); up = torch.cat(up_parts, dim=1); bm = torch.cat(bm_parts, dim=1)
            n_kept_pairs = int(s.numel())
            # a = s u_+ + c b, Delta on {u_+, b} = s [[s, c], [c, -s]]: e_+ = sqrt((1+s)/2) u_+ + sqrt((1-s)/2) b (lambda = +s),
            # e_- = sqrt((1-s)/2) u_+ - sqrt((1+s)/2) b (lambda = -s); sqrt((1-s)/2) = c/sqrt(2(1+s)) avoids 1 - s
            al = torch.sqrt((1.0 + s) / 2.0).to(CD); be = (c / torch.sqrt(2.0 * (1.0 + s))).to(CD)
            Ep = torch.zeros((n2, n_kept_pairs), dtype=CD, device=self.device); Em = torch.zeros_like(Ep)
            Ep[notneg] = up * al[None, :]; Ep[neg] = bm * be[None, :]
            Em[notneg] = up * be[None, :]; Em[neg] = -bm * al[None, :]
            vecs += [Ep, Em]; lams += [s, -s]
        if n0 > n1:                                                          # unpaired: in ran Phi_0 only (lambda = +1, positive half)
            Xu = torch.zeros((n2, n0 - nmin), dtype=CD, device=self.device); Xu[notneg] = Pp @ Vr[:, nmin:]
            vecs.append(Xu); lams.append(torch.ones(n0 - nmin, dtype=RD, device=self.device))
        elif n1 > n0:                                                        # unpaired: in ran Phi_1 only (lambda = -1, negative half)
            Xu = torch.zeros((n2, n1 - nmin), dtype=CD, device=self.device); Xu[neg] = U[:, nmin:]
            vecs.append(Xu); lams.append(-torch.ones(n1 - nmin, dtype=RD, device=self.device))
        lemma_v2 = 2 * n_dropped_pairs * OPS.binary_entropy_np([min(max(dropped, 1e-300), 0.5)])   # v2's term: D8's lemma at the largest dropped sine
        self.ent_bound += bound_pairs                                                            # v3 (30.1 (2)): sum s(s^2) joins D8's bound
        norm_D0 = math.sqrt(2.0 * total_sq + abs(n0 - n1))                                       # ||Delta_0||_F from the sines (tr Delta^2)
        share = math.sqrt(2.0 * dropped_sq) / max(norm_D0, 1e-300)                               # the dropped pairs' share of ||Delta_0||_F
        self.norm_Delta0 = norm_D0
        if vecs:
            cE = torch.cat(vecs, dim=1); lam = torch.cat(lams)               # eigenbasis coefficients of the Delta eigenvectors
            cols, halves = OPS.split_by_half(cE, neg)
            c_mem, half_mem, _ = OPS.orthonormalize_halves(cols, halves, cfg.eps_orth, negmask=neg)
            X = c_mem.mH @ cE                                               # r x k
            MD = (X * lam.to(CD)[None, :]) @ X.mH
            MD = 0.5 * (MD + MD.mH)
        else:
            lam = torch.zeros(0, dtype=RD, device=self.device)
            c_mem = torch.zeros((n2, 0), dtype=CD, device=self.device); half_mem = torch.zeros(0, dtype=torch.int64, device=self.device)
            MD = torch.zeros((0, 0), dtype=CD, device=self.device)
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
               "D8_lemma_dropped_pairs_v2": lemma_v2, "ent_bound_after_formation": self.ent_bound}
        self.init_stage_record()
        return rec

    # ------------------------------------------------------------------------------------------------ stage records
    def init_stage_record(self):
        self.rec = {"c_init": self.c.clone(), "MD_init": self.MD.clone(), "t_init": self.t_ref, "a": [], "cQ": [], "halfQ": [], "tQ": [],
                    "n_coll": 0, "E_rad": 0.0, "E_delta_init": self.memory_energy(), "cQ_from": (0, 0), "n_coll_all": 0}

    def memory_energy(self):
        """Re tr[M_Delta (c^dagger (e c))]: the memory's excess Killing energy."""
        e = self.stage["e"].to(CD)
        return float(torch.trace(self.MD @ (self.c.mH @ (e[:, None] * self.c))).real)

    def memory_entropy(self):
        M = self.MD + torch.diag((self.half < 0).to(CD))
        return OPS.entropy_cov(M)

    def collar_flux(self):
        """Re tr[M_Delta (D^dagger J_out D)] at the collar, D = V c."""
        D = self.stage["V"] @ self.c
        return float(torch.trace(self.MD @ (D.mH @ self.stage["Jout"].apply(D))).real)

    # --------------------------------------------------------------------------------------------------- block step
    def block_step(self, cfg, t_now):
        """spec 4.2 (a) to (j); returns the record of the checkpoint."""
        st = self.stage
        e, V, neg = st["e"], st["V"], st["neg"]
        t0 = time.time()
        r = self.c.shape[1]
        ph = torch.exp(-1j * e.to(CD) * (t_now - self.t_ref))
        c_t = ph[:, None] * self.c
        S_old, exc_old = OPS.entropy_cov(self.MD + torch.diag((self.half < 0).to(CD)))
        out = {"t": t_now, "r_before": r, "S_old": S_old}
        if r == 0:
            self.t_ref = t_now
            out.update({"m_coll": 0, "S_new": 0.0, "S_B": 0.0, "S_B_Z": 0.0, "I_RB": 0.0, "E_B": 0.0, "r_after": 0, "turnover": False,
                        "E_mem": 0.0, "L_collar": 0.0, "wall_s": time.time() - t0})
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
        mQ = int(sel.sum())
        if mQ == 0:
            self.c, self.t_ref = c_t, t_now
            E_mem = self.memory_energy()
            out.update({"m_coll": 0, "S_new": S_old, "S_B": 0.0, "S_B_Z": 0.0, "I_RB": 0.0, "E_B": 0.0, "r_after": r, "turnover": False,
                        "E_mem": E_mem, "L_collar": self.collar_flux(), "fifo_overlap_max": 0.0, "wall_s": time.time() - t0})
            return out
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
            out.update({"m_coll": 0, "S_new": S_old, "S_B": 0.0, "S_B_Z": 0.0, "I_RB": 0.0, "E_B": 0.0, "r_after": r, "turnover": False,
                        "E_mem": self.memory_energy(), "L_collar": self.collar_flux(), "wall_s": time.time() - t0})
            return out
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
        c_new, half_new, _ = OPS.orthonormalize_halves(c_p, self.half, cfg.eps_orth, negmask=neg)
        c_new = OPS.project_out(c_new, G)
        c_new, half_new, _ = OPS.orthonormalize_halves(c_new, half_new, cfg.eps_orth, negmask=neg)
        Dk = c_new.mH @ c_t                                                 # r' x r
        MD_new = Dk @ self.MD @ Dk.mH
        MD_new = 0.5 * (MD_new + MD_new.mH)
        Pi_old = torch.diag((self.half < 0).to(CD)); Pi_new = torch.diag((half_new < 0).to(CD))
        Xi = Pi_new - Dk @ Pi_old @ Dk.mH
        out["Xi_min_eig"] = float(torch.linalg.eigvalsh(0.5 * (Xi + Xi.mH)).min()) if Xi.shape[0] else 0.0
        Yk = torch.eye(Dk.shape[0], dtype=CD, device=self.device) - Dk @ Dk.mH - Xi
        out["I_minus_DD_minus_Xi_min_eig"] = float(torch.linalg.eigvalsh(0.5 * (Yk + Yk.mH)).min()) if Yk.shape[0] else 0.0
        out["Xi_norm"] = float(torch.linalg.norm(Xi))
        # (g) compression to the split hull of ran Delta above eps_rank
        lam, uu = torch.linalg.eigh(MD_new)
        keep = lam.abs() > cfg.eps_rank
        n_drop = int((~keep).sum())
        if n_drop:
            self.ent_bound += n_drop * OPS.binary_entropy_np([min(cfg.eps_rank, 0.5)])
        if bool(keep.any()):
            U_k = uu[:, keep]                                               # r' x k
            # the hull: each eigenvector's components on the two halves, orthonormalized per half (v2: in the memory's coordinates)
            c_fin, half_fin = OPS.hull_in_coords(c_new, half_new, U_k, cfg.eps_orth)
            Tf = c_fin.mH @ c_new                                           # r'' x r'
            MD_fin = Tf @ MD_new @ Tf.mH
            MD_fin = 0.5 * (MD_fin + MD_fin.mH)
        else:
            c_fin = c_new[:, :0]; half_fin = half_new[:0]; MD_fin = MD_new[:0, :0]
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
                    "turnover": bool(I_RB > S_B + 1e-9), "E_mem": E_mem, "L_collar": self.collar_flux(), "n_dropped_rank": n_drop,
                    "clip_excursion_max": max(exc_old, exc_B, exc_new, exc_Z), "wall_s": time.time() - t0})
        return out

    def prune_fifo(self, cfg, t_now):
        """a collected mode leaves the FIFO when its weight within R_cut + 3 tau_s (propagated to t_now) falls below 1e-10, or when the depth
        is exceeded; the weight at exit is recorded."""
        st = self.stage
        e, V = st["e"], st["V"]
        keep = []
        zone = (self.R2 > cfg.R_cut_abs) & (self.R2 < cfg.R_cut_abs + 3.0 * cfg.tau_s_abs)
        inside = (self.R2 <= cfg.R_cut_abs)
        exits = []
        for cq, hq, tq in self.fifo:
            x = V @ (torch.exp(-1j * e.to(CD) * (t_now - tq))[:, None] * cq)
            wz = float((x.abs().pow(2) * zone.to(RD)[:, None]).sum()); wi = float((x.abs().pow(2) * inside.to(RD)[:, None]).sum())
            if wz + wi > getattr(cfg, "fifo_prune_weight", 1e-10):
                keep.append((cq, hq, tq))
            else:
                exits.append(wz + wi)
        while len(keep) > cfg.fifo_depth:
            exits.append(float("nan")); keep.pop(0)
        self.fifo = keep
        self.fifo_exit_weights = exits

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
        else:
            E_x = 0.0
        E_mem = self.memory_energy()
        E_init = rec["E_delta_init"]
        return {"E_init": E_init, "E_mem": E_mem, "E_coll": E_coll, "E_x": E_x, "E_diag": rec["E_rad"], "E_rad_memory_route": E_init - E_mem,
                "residual_rel": (E_init - E_mem - E_coll - E_x) / max(abs(E_init), 1e-300), "n_modes": int(A.shape[1])}

    # --------------------------------------------------------------------------------------------------- transition
    def transition(self, cfg, E_new, Rb_new, n_test=128, tau_factor=1.0):
        """spec 4.4: the ramp v_k -> v_{k+1} over tau_tr, the memory and the FIFO through CF4, the sea continued adiabatically, the flipped
        modes, the memory re-expressed in the new eigenbasis; returns the record."""
        st_old = self.stage
        t0 = time.time()
        rs_old = st_old["rs"]; rs_new = 2.0 * cfg.g * E_new
        v_old = st_old["v_sites"]
        v_new = np.asarray(self.averager(OPS.flow_dense(rs_new, Rb_new)), float)
        tau = cfg.tau_tr * rs_old * float(tau_factor)                         # v3: tau_factor 2, 4, ... when 8.3 (8) doubles tau_tr
        vfun_t = lambda t: v_old + (v_new - v_old) * OPS.ramp_c1(t / tau)
        S_old, _ = self.memory_entropy()
        nD_old = float(torch.linalg.norm(self.MD))                           # ||Delta||_F of the memory at the transition (D orthonormal)
        e_old, V_old = st_old["e"], st_old["V"]
        # columns to carry through the ramp: the memory, the test modes, (the FIFO is cleared with its weight printed)
        D = V_old @ self.c
        idx_test = torch.argsort(e_old.abs())[:n_test]
        T_old = V_old[:, idx_test]
        occ_test = (e_old[idx_test] < 0)
        X = torch.cat([D, T_old], dim=1)
        X = OPS.cf4_propagate(self.B, X, vfun_t, 0.0, tau, cfg.dt_tr)
        D_r, T_r = X[:, : D.shape[1]], X[:, D.shape[1]:]
        fifo_w = 0.0
        for cq, hq, tq in self.fifo:
            x = V_old @ (torch.exp(-1j * e_old.to(CD) * (self.t_ref - tq))[:, None] * cq)
            fifo_w += float((x.abs().pow(2) * (self.R2 <= cfg.R_cut_abs + 3.0 * cfg.tau_s_abs).to(RD)[:, None]).sum())
        self.fifo = []
        Phi_old = V_old[:, st_old["neg"]]
        nneg_old = Phi_old.shape[1]
        # the new stage
        st_new = self.build_stage(cfg, E_new, Rb_new, label="stage")
        e_new, V_new, neg_new = st_new["e"], st_new["V"], st_new["neg"]
        # continuation: the nneg_old eigenmodes of h_new with the largest weight in ran P_old
        wts = (Phi_old.mH @ V_new).abs().pow(2).sum(dim=0)                     # (n2,)
        occ_idx = torch.topk(wts, nneg_old).indices
        occ_mask = torch.zeros(self.n2, dtype=torch.bool, device=self.device); occ_mask[occ_idx] = True
        amb = int(((wts > 0.1) & (wts < 0.9)).sum())
        up = (occ_mask & (e_new > 0)).nonzero().ravel()                      # continued-occupied, now positive: Delta += |j><j|
        down = ((~occ_mask) & (e_new < 0)).nonzero().ravel()                 # not continued, now negative: Delta -= |j><j|
        # the adiabatic defect on the test modes
        cT = V_new.mH @ T_r
        w_in_occ = (cT.abs().pow(2) * occ_mask.to(RD)[:, None]).sum(dim=0)
        leak = torch.where(occ_test, 1.0 - w_in_occ, w_in_occ)
        # v3 (30.1 (4)): ||U P_k U^dagger - P^cont||_F^2 = sum over every old mode of its leak (twice the occupied modes' sum, which equals
        # the empty modes'); the test modes carry it (the leak sits at the lowest |e|); printed with no tolerance, and its entropy bound
        # sum_i s(leak_i) joins D8's accumulated bound
        lk = leak.clamp(0.0, 0.5)
        sea_coh = float(torch.sqrt(lk.sum()))
        ent_leak = OPS.binary_entropy_np(lk.cpu().numpy())
        self.ent_bound += ent_leak
        del Phi_old, V_old
        # the memory in the new eigenbasis
        c_raw = V_new.mH @ D_r
        cols, halves = OPS.split_by_half(c_raw, neg_new)
        flipped = torch.cat([up, down])
        if flipped.numel():
            Ef = torch.zeros((self.n2, flipped.numel()), dtype=CD, device=self.device)
            Ef[flipped, torch.arange(flipped.numel(), device=self.device)] = 1.0
            hf = torch.where(e_new[flipped] > 0, torch.ones(flipped.numel(), dtype=torch.int64, device=self.device), -torch.ones(flipped.numel(), dtype=torch.int64, device=self.device))
            cols = torch.cat([cols, Ef], dim=1); halves = torch.cat([halves, hf])
        c_new, half_new, _ = OPS.orthonormalize_halves(cols, halves, cfg.eps_orth, negmask=neg_new)
        Tm = c_new.mH @ c_raw                                                 # r' x r
        MD_new = Tm @ self.MD @ Tm.mH
        if flipped.numel():
            Uf = c_new[flipped, :].conj().T                                   # r' x f: u_j = c_new^dagger e_j
            sgn = torch.cat([torch.ones(up.numel(), dtype=CD, device=self.device), -torch.ones(down.numel(), dtype=CD, device=self.device)])
            MD_new = MD_new + (Uf * sgn[None, :]) @ Uf.mH
        MD_new = 0.5 * (MD_new + MD_new.mH)
        D_tr = Tm
        Pi_old = torch.diag((self.half < 0).to(CD)); Pi_new = torch.diag((half_new < 0).to(CD))
        Xi_tr = (MD_new + Pi_new) - D_tr @ (self.MD + Pi_old) @ D_tr.mH
        # compression (as in the block step)
        lam, uu = torch.linalg.eigh(MD_new)
        keep = lam.abs() > cfg.eps_rank
        n_drop = int((~keep).sum())
        if n_drop:
            self.ent_bound += n_drop * OPS.binary_entropy_np([min(cfg.eps_rank, 0.5)])
        if bool(keep.any()):
            U_k = uu[:, keep]
            c_fin, half_fin = OPS.hull_in_coords(c_new, half_new, U_k, cfg.eps_orth)     # v2: the split hull in the memory's coordinates
            Tf = c_fin.mH @ c_new
            MD_fin = Tf @ MD_new @ Tf.mH
            MD_fin = 0.5 * (MD_fin + MD_fin.mH)
        else:
            c_fin = c_new[:, :0]; half_fin = half_new[:0]; MD_fin = MD_new[:0, :0]
        # v2: the memory now holds the coefficients at the end of the ramp, t_ref_old + tau (v1: 0.0, while the driver's clock continues)
        self.c, self.half, self.MD, self.t_ref = c_fin, half_fin, MD_fin, self.t_ref + tau
        S_new, _ = self.memory_entropy()
        self.init_stage_record()
        self.n_transitions += 1
        return {"tau_factor": float(tau_factor), "sea_coherence_test_modes": sea_coh, "sea_coherence_over_memory_Delta": sea_coh / max(nD_old, 1e-300),
                "sea_coherence_over_Delta0": (sea_coh / self.norm_Delta0) if self.norm_Delta0 else None, "entropy_bound_sea_leak": ent_leak,
                "ent_bound_after": self.ent_bound, "memory_Delta_norm_before": nD_old,
                "tau_tr": tau, "rs_old": rs_old, "rs_new": rs_new, "E_new": float(E_new), "n_neg_old": nneg_old, "n_neg_new": st_new["nneg"],
                "ambiguous_continuations": amb, "released_pushed_up": int(up.numel()), "released_new_pockets": int(down.numel()),
                "adiabatic_leak_max": float(leak.max()), "adiabatic_leak_mean": float(leak.mean()), "fifo_weight_near_cut_dropped": fifo_w,
                "S_before": S_old, "S_after": S_new, "S_continuity_defect": S_new - S_old, "Xi_tr_norm": float(torch.linalg.norm(Xi_tr)),
                "memory_rank": int(c_fin.shape[1]), "n_dropped_rank": n_drop, "r_t_new": st_new["r_t"], "r_in_new": st_new["r_in"],
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

    def cross_surface(self, cfg, ds=(2, 4, 8), width=4):
        """I(X:Y) between slabs just inside and just outside r_t for the physical state and for the Killing vacuum, with 2||K||_op^2."""
        st = self.stage
        if st["r_t"] is None:
            return []
        e, V, neg = st["e"], st["V"], st["neg"]
        out = []
        for d in ds:
            X = np.where((self.R >= st["r_t"] - d - width) & (self.R < st["r_t"] - d))[0]
            Y = np.where((self.R >= st["r_t"] + d) & (self.R < st["r_t"] + d + width))[0]
            ix = np.concatenate([[2 * s, 2 * s + 1] for s in X]); iy = np.concatenate([[2 * s, 2 * s + 1] for s in Y])
            idx = torch.as_tensor(np.concatenate([ix, iy]), dtype=torch.int64, device=self.device)
            rowsV = V[idx, :]                                                   # k x n2
            Cvac = rowsV[:, neg] @ rowsV[:, neg].mH
            Dm = rowsV @ self.c                                                 # k x r (D restricted to the slabs)
            Cphys = Cvac + Dm @ self.MD @ Dm.mH
            nx = len(ix)
            def mi(Cm):
                SX, _ = OPS.entropy_cov(Cm[:nx, :nx]); SY, _ = OPS.entropy_cov(Cm[nx:, nx:]); SXY, _ = OPS.entropy_cov(Cm)
                K = Cm[:nx, nx:]
                kop = float(torch.linalg.matrix_norm(K, ord=2))
                return {"I": SX + SY - SXY, "bound_2K2": 2.0 * kop * kop, "K_op": kop}
            out.append({"d": d, "physical": mi(Cphys), "killing_vacuum": mi(Cvac)})
        return out
