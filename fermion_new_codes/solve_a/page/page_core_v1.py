#!/usr/bin/env python
"""page_core_v1.py (lead reviewer A, GAP03 integration; builder analyst runs it): the retained-channel recursion of the review's S5 in
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

import page_ops_v1 as OPS

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
        Phi1 = V[:, neg]
        n0, n1 = Phi0.shape[1], Phi1.shape[1]
        W = Phi1.mH @ Phi0                                                  # n1 x n0
        U, S, Vh = torch.linalg.svd(W, full_matrices=True)
        nmin = min(n0, n1)
        sig = S.clamp(0.0, 1.0)
        sin_th = torch.sqrt((1.0 - sig * sig).clamp_min(0.0))
        keep = sin_th > cfg.eps_form
        vecs, lams = [], []
        if bool(keep.any()):
            A = Phi0 @ Vh.mH[:, :nmin][:, keep]
            Bm = Phi1 @ U[:, :nmin][:, keep]
            plus = A + Bm; minus = A - Bm
            plus = plus / torch.linalg.norm(plus, dim=0)[None, :]
            minus = minus / torch.linalg.norm(minus, dim=0)[None, :]
            vecs += [(plus + minus) / math.sqrt(2.0), (plus - minus) / math.sqrt(2.0)]
            lams += [sin_th[keep], -sin_th[keep]]
        if n0 > n1:
            vecs.append(Phi0 @ Vh.mH[:, nmin:]); lams.append(torch.ones(n0 - nmin, dtype=RD, device=self.device))
        elif n1 > n0:
            vecs.append(Phi1 @ U[:, nmin:]); lams.append(-torch.ones(n1 - nmin, dtype=RD, device=self.device))
        dropped = float(sin_th[~keep].max()) if bool((~keep).any()) else 0.0
        n_dropped_pairs = int((~keep).sum())
        self.ent_bound += 2 * n_dropped_pairs * OPS.binary_entropy_np([min(max(dropped, 1e-300), 0.5)])   # D8 with the largest dropped |lambda|
        Evecs = torch.cat(vecs, dim=1); lam = torch.cat(lams)
        cE = V.mH @ Evecs                                                   # eigenbasis coefficients of the Delta eigenvectors
        cols, halves = OPS.split_by_half(cE, neg)
        c_mem, half_mem, _ = OPS.orthonormalize_halves(cols, halves, cfg.eps_orth)
        X = c_mem.mH @ cE                                                   # r x k
        MD = (X * lam.to(CD)[None, :]) @ X.mH
        MD = 0.5 * (MD + MD.mH)
        self.c, self.half, self.MD, self.t_ref = c_mem, half_mem, MD, 0.0
        # inversion readout: n_alpha = ||V[:, alpha]^dagger Phi_0||^2
        n_alpha = (V.mH @ Phi0).abs().pow(2).sum(dim=1)
        pos = e > 0
        Np = float(n_alpha[pos].sum()); Nm = float((1.0 - n_alpha[neg]).sum())
        E_inv = float((e[pos] * n_alpha[pos]).sum() + (e[neg].abs() * (1.0 - n_alpha[neg])).sum())
        TH = 1.0 / (4.0 * math.pi * st["rs"])
        inv = torch.where(pos, n_alpha, 1.0 - n_alpha)
        edges = torch.tensor([0, 1, 2, 3, 4, 6, 8, 12, 25, 1e9], dtype=RD, device=self.device)
        hist = [float(inv[(e.abs() / TH >= edges[i]) & (e.abs() / TH < edges[i + 1])].sum()) for i in range(len(edges) - 1)]
        rec = {"n0": n0, "n1": n1, "rank_kept": int(keep.sum()) * 2 + abs(n0 - n1), "sin_theta_max": float(sin_th.max()) if nmin else 0.0,
               "sin_theta_dropped_max": dropped, "memory_rank": int(c_mem.shape[1]), "N_plus": Np, "N_minus": Nm, "E_inv": E_inv,
               "E_inv_over_E": E_inv / st["E"], "inverted_by_bin_TH": hist, "bin_edges_TH": [0, 1, 2, 3, 4, 6, 8, 12, 25, "inf"],
               "trace_Delta": float(lam.sum()), "trace_MD": float(torch.trace(MD).real)}
        self.init_stage_record()
        return rec

    # ------------------------------------------------------------------------------------------------ stage records
    def init_stage_record(self):
        self.rec = {"c_init": self.c.clone(), "MD_init": self.MD.clone(), "t_init": self.t_ref, "a": [], "cQ": [], "halfQ": [], "tQ": [],
                    "n_coll": 0, "E_rad": 0.0, "E_delta_init": self.memory_energy()}

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
        cQ, halfQ, _ = OPS.orthonormalize_halves(cols, halves, cfg.eps_orth)
        # (d) orthogonalize against the recent collected modes
        ov_max = 0.0
        if self.fifo and cfg.fifo_depth > 0:
            F = torch.cat([torch.exp(-1j * e.to(CD) * (t_now - tq))[:, None] * cq for cq, hq, tq in self.fifo], dim=1)
            ov = F.mH @ cQ
            ov_max = float(ov.abs().max()) if ov.numel() else 0.0
            cQ = cQ - F @ ov
            cQ, halfQ, _ = OPS.orthonormalize_halves(cQ, halfQ, cfg.eps_orth)
        mQ = int(cQ.shape[1])
        out["fifo_overlap_max"] = ov_max
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
        c_new, half_new, _ = OPS.orthonormalize_halves(c_p, self.half, cfg.eps_orth)
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
            # the hull: each eigenvector's components on the two halves, orthonormalized per half
            negrows = (half_new < 0)
            U_pos = U_k.clone(); U_pos[negrows] = 0.0
            U_neg = U_k - U_pos
            cand = torch.cat([c_new @ U_pos, c_new @ U_neg], dim=1)
            hcand = torch.cat([torch.ones(U_k.shape[1], dtype=torch.int64, device=self.device), -torch.ones(U_k.shape[1], dtype=torch.int64, device=self.device)])
            c_fin, half_fin, _ = OPS.orthonormalize_halves(cand, hcand, cfg.eps_orth)
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
            rec["E_rad"] += E_B
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
        out = {"S_direct": S_direct, "m": int(A.shape[1]), "excursion": exc}
        if with_spectrum and cfg.store_modes and rec["cQ"]:
            e = self.stage["e"].to(CD)
            cQ = torch.cat([torch.exp(-1j * e * (rec["t_init"] - tq))[:, None] * cq for cq, tq in zip(rec["cQ"], rec["tQ"])], dim=1)
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

    # --------------------------------------------------------------------------------------------------- transition
    def transition(self, cfg, E_new, Rb_new, n_test=128):
        """spec 4.4: the ramp v_k -> v_{k+1} over tau_tr, the memory and the FIFO through CF4, the sea continued adiabatically, the flipped
        modes, the memory re-expressed in the new eigenbasis; returns the record."""
        st_old = self.stage
        t0 = time.time()
        rs_old = st_old["rs"]; rs_new = 2.0 * cfg.g * E_new
        v_old = st_old["v_sites"]
        v_new = np.asarray(self.averager(OPS.flow_dense(rs_new, Rb_new)), float)
        tau = cfg.tau_tr * rs_old
        vfun_t = lambda t: v_old + (v_new - v_old) * OPS.ramp_c1(t / tau)
        S_old, _ = self.memory_entropy()
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
        c_new, half_new, _ = OPS.orthonormalize_halves(cols, halves, cfg.eps_orth)
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
            U_pos = U_k.clone(); U_pos[(half_new < 0)] = 0.0
            U_neg = U_k - U_pos
            cand = torch.cat([c_new @ U_pos, c_new @ U_neg], dim=1)
            hcand = torch.cat([torch.ones(U_k.shape[1], dtype=torch.int64, device=self.device), -torch.ones(U_k.shape[1], dtype=torch.int64, device=self.device)])
            c_fin, half_fin, _ = OPS.orthonormalize_halves(cand, hcand, cfg.eps_orth)
            Tf = c_fin.mH @ c_new
            MD_fin = Tf @ MD_new @ Tf.mH
            MD_fin = 0.5 * (MD_fin + MD_fin.mH)
        else:
            c_fin = c_new[:, :0]; half_fin = half_new[:0]; MD_fin = MD_new[:0, :0]
        self.c, self.half, self.MD, self.t_ref = c_fin, half_fin, MD_fin, 0.0
        S_new, _ = self.memory_entropy()
        self.init_stage_record()
        return {"tau_tr": tau, "rs_old": rs_old, "rs_new": rs_new, "E_new": float(E_new), "n_neg_old": nneg_old, "n_neg_new": st_new["nneg"],
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
            rows.append({"n": n, "norm2_w": float(rho.sum()), "n_1_minus_n": n * (1.0 - n),
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
