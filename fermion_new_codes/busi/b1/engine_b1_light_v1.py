#!/usr/bin/env python
"""engine_b1_light_v1.py (busi lane, B1 build; design busi/notes/B1_design_reviewa_v1.md, review REVIEWA_B1_design_review_v1.md with its
MUST-FIX M1 to M8 and RECOMMENDED R1, R2, R3, R4, R5, R8, R9 applied). The light channel of B1: per kappa the massless chain of
page_history_v6q on a long chain of n_L sites whose first n_e sites are the heavy engine's (no interpolation), its + sector state Phi_+
(n_L x n_+) prepared as the Killing vacuum of h_+(T_0) and propagated by the record's piecewise-constant h_+ (page_interior_v6s.propagate_lean on
BandedLean, R1), its Gauss source normal-ordered against its own continued sea on a truncated chain of n_v = n_e + pad sites
(engine_ops_v2.continue_sea_v2 on an engine_collapse_v5.Chan at m = 0, design D-B) plus the one-time long-chain correction Delta_0 at T_0, fed
back into the heavy engine's sources by engine_b1_v1's hooks. Lattice units a = 1, torch complex128.

Conventions (M8, R9; printed in every head and record label):
- the weights w_kappa = 16 kappa per two-sector chain count the declared field (page_history's ChannelOps.w); no 4 kappa option;
- every entropy is "per chain, two sectors": S_X(C) = 2 S_I(C_+) on the interior block (purity, page_interior_v6s);
- the light reference: the continued sea on the truncated chain, transported once per record and held through the solve (the C64 lagged
  convention, C64_fix_ruling_v4), LZ-matched at crossings (continue_sea_v2); no sector walk;
- the exterior beyond the heavy grid: vacuum PG at the record's collar charge (N = 1, v = -sqrt(2 g E/R)) or sourced (the light's own Gauss
  law marched outward from the collar, R3);
- the light's state density by the sector identity Tr[W A C] = 2 Re Tr[W A_+ C_+] - Tr[W A_+] (A = h_0, Q, the clocked h, the collar current;
  W site-diagonal), the reference's by engine_collapse_v5.dens on the full 2n representation.
"""
import hashlib
import json
import math
import os
import time

import numpy as np
import scipy.sparse as sp
import torch

import page_ops_v6 as OPS
import page_w1_sector_v1d as W1S
import page_interior_v6s as IR
import engine_collapse_v5 as C5
import engine_ops_v2 as OPS2
import engine_stress_v1 as STR

CD = torch.complex128
RD = torch.float64
LN2 = math.log(2.0)
BHK = 0.386                                            # the Gaussian-typical entropy per mode (M5)
T_START = time.time()
LIGHT_LABEL = ("light reference: the continued sea on the truncated chain (n_v = %d), transported once per record and held through the solve "
               "(the C64 lagged convention, C64_fix_ruling_v4), LZ-matched at crossings (engine_ops_v2.continue_sea_v2); no sector walk; "
               "Delta_0 the long-chain correction at T_0; weights 16 kappa per two-sector chain (the declared field); entropies per chain, two sectors; "
               "exterior: %s")


def log(msg):
    print("[B1L %8.1f s] %s" % (time.time() - T_START, msg), flush=True)


def parse_kappas(s):
    return [int(x) for x in str(s or "").split(",") if x.strip() != ""]


def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def entropy_eigs(ev):
    return float(OPS.binary_entropy_t(ev).sum()) if ev.numel() else 0.0


def block_entropy(G):
    """2 S(G) for a Hermitian block G (n x n device tensor) with eigenvalues in [0, 1]: the per-chain two-sector entropy of the + sector block."""
    if G.shape[0] == 0:
        return 0.0, 0.0
    ev = torch.linalg.eigvalsh(0.5 * (G + G.mH))
    exc = max(0.0, float(-ev.min()), float(ev.max()) - 1.0)
    return 2.0 * entropy_eigs(ev), exc


def band_rows(A, m, dev):
    """IR.BandedLean of the leading m x m block of the banded CSR A (half-bandwidth 2): its rows n < m - 2 equal A's rows applied to X[:m]."""
    return IR.BandedLean(A.tocsr()[:m, :m], dev)


def apply_lean(B, X):
    Y = torch.empty_like(X)
    return B.apply_into(X, Y)


def sector_density(B, X, nrows, diag, chunk=1024):
    """per site n < nrows: 2 Re sum_j conj(X[n, j]) (A X)[n, j] - Re A_nn (the sector identity for a site-diagonal weight; B the BandedLean of
    A's leading block, X the + sector state's leading rows)."""
    m = B.n2
    out = torch.zeros(nrows, dtype=RD, device=X.device)
    for j0 in range(0, X.shape[1], chunk):
        Xc = X[:m, j0: j0 + chunk]
        Y = apply_lean(B, Xc)
        out += (Xc[:nrows].conj() * Y[:nrows]).real.sum(dim=1)
        del Y
    return 2.0 * out.cpu().numpy() - np.asarray(diag[:nrows], float)


def row_energy(B, X, chunk=512):
    """r_n = Re sum_j conj(X[n, j]) (A X)[n, j] over all rows (the full-chain BandedLean B); one apply."""
    out = torch.zeros(X.shape[0], dtype=RD, device=X.device)
    m = min(int(chunk), max(1, X.shape[1]))
    buf = torch.empty((X.shape[0], m), dtype=X.dtype, device=X.device)
    for j0 in range(0, X.shape[1], m):
        Xc = X[:, j0: j0 + m]; mm = Xc.shape[1]
        Y = B.apply_into(Xc, buf[:, :mm])
        out += (Xc.conj() * Y).real.sum(dim=1)
    del buf
    return out


def local_trace_sector(Asub, Xsub):
    """Re Tr[X_sub^dagger A_sub X_sub] for a small dense A_sub on the rows of X_sub."""
    return float((Xsub.conj() * (Asub @ Xsub)).real.sum())


def gaussian_smooth(x, ell):
    if ell <= 0:
        return np.asarray(x, float)
    n = int(math.ceil(4 * ell))
    k = np.exp(-0.5 * (np.arange(-n, n + 1) / float(ell)) ** 2); k /= k.sum()
    return np.convolve(np.asarray(x, float), k, mode="same")


class LightChannel:
    """one channel kappa of the light field: the long chain, its + sector state, the truncated reference chain and its window, the offsets."""

    def __init__(self, kappa, sourced, L):
        self.kappa = int(kappa); self.sourced = bool(sourced); self.w = 16.0 * self.kappa
        self.L = L
        n_L, n_e, n_v, dev = L.n_L, L.n_e, L.n_v, L.dev
        self.R = np.arange(n_L, dtype=float) + 1.0
        self.h0, self.Q = OPS.chain_sparse(self.R, 0.0, float(kappa))
        h0p = W1S.sector_plus_csr(self.h0).tocsr(); Qp = W1S.sector_plus_csr(self.Q).tocsr()
        self.h0p, self.Qp = h0p, Qp
        self.h0p_diag = np.real(h0p.diagonal()); self.Qp_diag = np.real(Qp.diagonal())
        self.B_h0_e = band_rows(h0p, n_e + 2, dev); self.B_Q_e = band_rows(Qp, n_e + 2, dev)
        self.B_h0_full = None; self.B_Q_full = None                     # built on demand (sourced exterior, exterior work)
        self.ch_v = C5.Chan(self.kappa, 0.0, n_v)
        self.Phi = None
        self.win = None; self.ref = None; self.n_sea_v = None; self.mask = None; self.e_v = None; self.U_v = None
        self.D0_eps = None; self.D0_J = None
        self.dS0 = None                                                  # per interior size m: S_I(C(T_0)) - S_I(P_-^v(T_0))
        self.stress0 = None
        self.cheb_K_max = 0; self.cheb_tail_acc = 0.0; self.n_substeps = 0; self.T_init = None
        self.flow_acc = {}; self.last_flow = {}; self.L_at_t = None; self.cheb_K_last = 0; self.cheb_tail_last = 0.0
        self.eps_ref_sum_prev = None; self.eps_ref_prev = None
        self.cinfo = None; self.m1_old = None; self.m1_off = None; self.m1_cur = None
        self.J_C0_full = None; self.eps_C0_full = None
        self.n_plus = None

    # ------------------------------------------------------------------------------------------------- fields and operators
    def h_long(self, N_L, v_L):
        return OPS.clocked_csr(self.h0, self.Q, N_L, v_L)

    def hp_long(self, N_L, v_L):
        return W1S.sector_plus_csr(self.h_long(N_L, v_L)).tocsr()

    # ------------------------------------------------------------------------------------------------- densities
    def state_density_heavy(self):
        """eps^C, J^C on the heavy grid (n_e sites) by the sector identity on the unclocked h_0 and Q."""
        n_e = self.L.n_e
        e = sector_density(self.B_h0_e, self.Phi, n_e, self.h0p_diag)
        j = sector_density(self.B_Q_e, self.Phi, n_e, self.Qp_diag)
        return e, j

    def state_density_full(self):
        """eps^C, J^C on every site of the long chain (two full applies)."""
        if self.B_h0_full is None:
            self.B_h0_full = IR.BandedLean(self.h0p, self.L.dev); self.B_Q_full = IR.BandedLean(self.Qp, self.L.dev)
        e = 2.0 * row_energy(self.B_h0_full, self.Phi).cpu().numpy() - self.h0p_diag
        j = 2.0 * row_energy(self.B_Q_full, self.Phi).cpu().numpy() - self.Qp_diag
        return e, j

    def ref_density(self, ref):
        n_e = self.L.n_e
        return C5.dens(self.ch_v.h, ref)[:n_e], C5.dens(self.ch_v.Q, ref)[:n_e]

    def eig_v(self, Nv, vv):
        E19 = self.L.E19
        return E19.eigT(E19.clocked_gpu(self.ch_v, Nv, vv))


class LightSet:
    """the light channels of B1 and everything engine_b1_v1's hooks call (H1 start, H2 sources, H3 trial sources, H4 M1, H5 stresses, H6 record
    and readouts, H7 step, H8 saves)."""

    def __init__(self, a, R_heavy, chs, E19):
        self.a = a; self.E19 = E19
        self.dev = E19.BK["dev"]
        self.R_e = np.asarray(R_heavy, float); self.n_e = len(self.R_e)
        if not np.allclose(self.R_e, np.arange(self.n_e) + 1.0):
            raise ValueError("B1 needs the heavy sites R = 1 ... n_e (--half-offset 0)")
        self.n_L = int(a.light_nsites); self.n_v = int(a.light_nv) if a.light_nv else self.n_e + int(a.light_pad)
        if not (self.n_e + 16 <= self.n_v < self.n_L):
            raise ValueError("--light-nv %d must lie in [n_e + 16, n_L)" % self.n_v)
        self.g = float(a.g); self.rs = float(a.rs)
        for ch in chs:
            if abs(ch.w - 16.0 * ch.kappa) > 0:
                raise ValueError("the heavy channel weight %g at kappa %d is not 16 kappa" % (ch.w, ch.kappa))
        ks = parse_kappas(a.light_kappas); gs = parse_kappas(a.light_guard_kappas)
        if getattr(a, "light_kappa_zero", "off") == "on" and 0 not in gs:
            gs = gs + [0]
        self.chans = [LightChannel(k, True, self) for k in ks] + [LightChannel(k, False, self) for k in gs if k not in ks]
        self.src = [c for c in self.chans if c.sourced]
        self.exterior = a.light_exterior
        self.ext_profile = None                                          # (N_ext, v_ext) beyond n_e from the sourced march (lagged one record)
        self.TM = {}
        self.tm_reset()
        rs = self.rs
        self.cut_rs = [float(x) for x in str(a.light_cuts_rs).split(",") if x.strip() and x.strip() != "root"]
        self.cut_sites = {("%g" % x): int(np.sum(self.R_e <= x * rs)) for x in self.cut_rs}
        self.m_max = int(max(list(self.cut_sites.values()) + [8]))
        self.E_rad_acc = 0.0; self.E_rad_src_acc = 0.0; self.W_rad_acc = 0.0; self.E0 = None; self.E_prev_rec = None
        self.eps_light = np.zeros(self.n_e); self.J_light = np.zeros(self.n_e); self.ldiag = {}
        self.stop_reason = None; self.prev_rec = None
        self.last_readout = None; self.flow_window = {}
        self.eps_ref_cache = {}
        self.head = {}
        self.ext_E_used = None
        self.label = LIGHT_LABEL % (self.n_v, self.exterior)
        self.trial_cache = None

    def tm_reset(self):
        self.TM.clear()
        self.TM.update({"light_prop_s": 0.0, "light_ref_s": 0.0, "light_dens_s": 0.0, "light_flow_s": 0.0, "light_m1_s": 0.0, "light_stress_s": 0.0,
                        "light_readout_s": 0.0, "light_ext_s": 0.0, "light_eigh_n": 0})

    # ------------------------------------------------------------------------------------------------------------ the fields
    def ext_fields(self, E, n):
        """(N, v) on sites n_e ... n - 1 beyond the heavy grid: the vacuum PG exterior at E, or the sourced profile (lagged one record)."""
        R = np.arange(self.n_e, n, dtype=float) + 1.0
        if self.exterior == "sourced" and self.ext_profile is not None:
            N_x, v_x = self.ext_profile
            return N_x[: n - self.n_e].copy(), v_x[: n - self.n_e].copy()
        return np.ones(len(R)), -np.sqrt(2.0 * self.g * max(float(E), 0.0) / R)

    def fields_on(self, f, E, n):
        if n <= self.n_e:
            return np.array(f[0][:n], float), np.array(f[1][:n], float)
        N = np.empty(n); v = np.empty(n)
        N[: self.n_e] = f[0]; v[: self.n_e] = f[1]
        Nx, vx = self.ext_fields(E, n)
        N[self.n_e:] = Nx; v[self.n_e:] = vx
        return N, v

    # ------------------------------------------------------------------------------------------------------------ H1: start
    def start(self, f0, E0, head):
        a = self.a; dev = self.dev; t0 = time.time()
        self.E0 = float(E0); self.E_prev_rec = float(E0)
        conv = {"weights": "16 kappa per two-sector chain (the declared field, M8)", "heavy_w_kappa1": 16.0, "channels": {}}
        eig_dev = torch.device("cpu") if a.light_init == "host" else dev
        for c in self.chans:
            tc = time.time()
            N_L, v_L = self.fields_on(f0, E0, self.n_L)
            h = c.h_long(N_L, v_L)
            aud_h = W1S.operator_audit(h); aud_h0 = W1S.operator_audit(c.h0); aud_Q = W1S.operator_audit(c.Q)
            hp = W1S.sector_plus_csr(h).tocsr()
            cache = None
            if a.light_init_cache:
                os.makedirs(a.light_init_cache, exist_ok=True)
                cache = os.path.join(a.light_init_cache, "b1init_k%d_%s.pt" % (c.kappa, IR._hp_md5(hp)))
            if cache and os.path.exists(cache):
                c.Phi = torch.load(cache, map_location=dev); info = {"route": "cache %s" % cache}
                log("kappa %d: the initial + sector vacuum loaded from %s" % (c.kappa, cache))
            else:
                if dev.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(dev)
                c.Phi, info = IR._dense_init_plus(hp, dev, eig_dev, int(a.light_threads), log)
                if dev.type == "cuda":
                    info["device_peak_GB"] = torch.cuda.max_memory_allocated(dev) / 1e9
                if cache:
                    torch.save(c.Phi, cache)
            c.n_plus = int(c.Phi.shape[1])
            c.T_init = W1S.impurity_trace_norm(c.Phi) if a.light_init_impurity == "on" else None
            # the reference chain at T_0
            Nv, vv = self.fields_on(f0, E0, self.n_v)
            e, U = c.eig_v(Nv, vv)
            en = e.cpu().numpy(); kk = int((en < 0).sum())
            gap = float(en[kk] - en[kk - 1]) if 0 < kk < len(en) else None
            c.ref = U[:, e < 0].clone(); c.n_sea_v = kk; c.e_v = e; c.U_v = U
            c.mask = (en < 0)
            c.win = [{"ref": c.ref, "f": (Nv.copy(), vv.copy()), "gap": gap}]
            # Delta_0: the long chain's vacuum density minus the truncated vacuum's on the heavy grid
            eC, jC = c.state_density_heavy()
            eR, jR = c.ref_density(c.ref)
            c.D0_eps = eC - eR; c.D0_J = jC - jR
            c.eps_ref_prev = eR.copy(); c.eps_ref_sum_prev = float(eR.sum())
            # Delta S_0 per interior size: S_I(C(T_0)) - S_I(P_-^v(T_0)), m = 1 ... m_max + root allowance
            mmax = min(max(self.m_max, 64), self.n_e)
            G = c.Phi[:mmax] @ c.Phi[:mmax].mH
            Yv = W1S.to_sector(c.ref, 1)[:mmax]; Pv = Yv @ Yv.mH
            dS = np.zeros(mmax + 1)
            for m in range(1, mmax + 1):
                dS[m] = block_entropy(G[:m, :m])[0] - block_entropy(Pv[:m, :m])[0]
            c.dS0 = dS
            # conventions (4.2 and M8's sector-entropy known answer)
            hs_c5 = c.ch_v.hs; hs_ops, Q_ops = OPS.chain_sparse(np.arange(self.n_v, dtype=float) + 1.0, 0.0, float(c.kappa))
            d_h = abs(hs_c5 - hs_ops); d_Q = abs(c.ch_v.Qs - Q_ops)
            sec = {}
            Vfull = c.ref
            for lab, m in self.cut_sites.items():
                Sfull = entropy_eigs(torch.linalg.eigvalsh(0.5 * ((Vfull[: 2 * m] @ Vfull[: 2 * m].mH) + (Vfull[: 2 * m] @ Vfull[: 2 * m].mH).mH)))
                Ssec = block_entropy(Pv[:m, :m])[0]
                sec[lab] = {"S_full_block": Sfull, "S_sector_2S_plus": Ssec, "diff": Ssec - Sfull}
            # the two density routes on the same state: the sector identity against C5.dens on the interior factorization
            dens2 = self.interior_factor_dens(c)
            d_routes = float(np.max(np.abs(dens2 - eC[: len(dens2)])))
            cc = {"Chan_vs_chain_sparse_h_max": float(d_h.max()) if d_h.nnz else 0.0, "Chan_vs_chain_sparse_Q_max": float(d_Q.max()) if d_Q.nnz else 0.0,
                  "audit_h0": aud_h0["ok"], "audit_Q": aud_Q["ok"], "audit_h_T0": aud_h["ok"], "audit_h_T0_detail": aud_h,
                  "Delta0_max_abs": float(np.max(np.abs(c.D0_eps))), "Delta0_J_max_abs": float(np.max(np.abs(c.D0_J))),
                  "Delta0_first8": [float(x) for x in c.D0_eps[:8]], "Delta0_last8": [float(x) for x in c.D0_eps[-8:]],
                  "Delta0_sum": float(c.D0_eps.sum()), "density_routes_max_abs": d_routes, "sector_entropy_known_answer": sec,
                  "dS0_at_cuts": {lab: float(dS[m]) for lab, m in self.cut_sites.items() if m <= mmax}, "n_plus": c.n_plus, "n_sea_v": kk,
                  "init": {k_: v_ for k_, v_ in info.items() if isinstance(v_, (int, float, str))}, "T_init_impurity": c.T_init,
                  "seconds": time.time() - tc, "sourced": c.sourced, "w": c.w}
            conv["channels"][str(c.kappa)] = cc
            log("kappa %d%s: T_0 vacuum n_+ %d (n_L %d), reference n_v %d (n_sea %d); Chan vs chain_sparse h %.1e Q %.1e; audits h0 %s Q %s h(T0) %s; "
                "Delta_0 max %.2e (sum %.3e; last site %.2e); density routes %.1e; sector entropy known answer %s; Delta S_0 %s; T_init %s (%.1f s)" % (
                    c.kappa, "" if c.sourced else " (guard)", c.n_plus, self.n_L, self.n_v, kk, cc["Chan_vs_chain_sparse_h_max"], cc["Chan_vs_chain_sparse_Q_max"],
                    aud_h0["ok"], aud_Q["ok"], aud_h["ok"], cc["Delta0_max_abs"], cc["Delta0_sum"], c.D0_eps[-1], d_routes,
                    ", ".join("%s: %.1e" % (k_, v_["diff"]) for k_, v_ in sec.items()), ", ".join("%s: %.3e" % (k_, v_) for k_, v_ in cc["dS0_at_cuts"].items()),
                    ("%.2e" % c.T_init) if c.T_init is not None else "-", cc["seconds"]))
            if self.exterior == "sourced" or a.light_exterior_work == "on":
                c.eps_C0_full, c.J_C0_full = c.state_density_full()
            if dev.type == "cuda":
                torch.cuda.empty_cache()
        self.stress_offsets()
        conv["seconds"] = time.time() - t0
        conv["label"] = self.label
        self.head = {"b1_conventions": conv, "light_label": self.label, "n_L": self.n_L, "n_v": self.n_v, "n_e": self.n_e,
                     "kappas_sourced": [c.kappa for c in self.src], "kappas_guard": [c.kappa for c in self.chans if not c.sourced],
                     "cut_sites": self.cut_sites, "rs_nominal": self.rs, "first_return_rs": self.first_return_rs(E0)}
        head["b1_light"] = self.head
        return conv

    def first_return_rs(self, E):
        R = np.arange(self.n_e, self.n_L, dtype=float) + 1.0
        speed = np.mean(1.0 - np.sqrt(2.0 * self.g * max(E, 0.0) / R)) if len(R) else 1.0
        t_end = (self.n_L - self.n_e) / max(speed, 1e-6)
        return {"front_at_chain_end_rs": t_end / self.rs, "reflection_at_collar_rs": 2.0 * t_end / self.rs}

    # ------------------------------------------------------------------------------------------- the interior factorization (H5, 4.2)
    def interior_columns(self, c, nI):
        """the full-representation columns of C restricted to the first nI sites with weights: [W_+ u_i, W_- conj(u_i)], [lambda_i, 1 - lambda_i]
        (C_+,II = Phi_+[:nI] Phi_+[:nI]^dagger = sum lambda_i u_i u_i^dagger; C_-,II = conj(1 - C_+,II))."""
        X = c.Phi[:nI]
        G = X @ X.mH
        lam, V = torch.linalg.eigh(0.5 * (G + G.mH))
        cols = torch.cat([W1S.from_sector(V, 1), W1S.from_sector(V.conj(), -1)], dim=1)
        lam_np = lam.real.cpu().numpy()
        f = np.concatenate([lam_np, 1.0 - lam_np])
        return cols, f

    def interior_factor_dens(self, c):
        nI = self.n_e + 8 if self.n_e + 8 <= self.n_L else self.n_L
        cols, f = self.interior_columns(c, nI)
        ch = C5.Chan(c.kappa, 0.0, nI)
        return C5.dens(ch.h, cols, f)[: self.n_e]

    def stress_parts_channel(self, c, ref):
        nI = self.n_e + 8
        if not hasattr(c, "_cs"):
            c._cs = STR.ChanStress(C5.Chan(c.kappa, 0.0, nI), torch, self.dev)
        cols, f = self.interior_columns(c, nI)
        rI = ref[: 2 * nI]
        cols = torch.cat([cols, rI], dim=1); f = np.concatenate([f, -np.ones(rI.shape[1])])
        p = STR.stress_parts(c._cs, cols, f)
        return {k_: np.asarray(v_, float)[: self.n_e] for k_, v_ in p.items()}

    def stress_offsets(self):
        for c in self.src:
            c.stress0 = self.stress_parts_channel(c, c.ref)

    def stresses(self):
        """H5: the weighted light parts (sourced channels; the T_0 truncation offset subtracted, labelled) for STR.shell_values."""
        t_ = time.time()
        P = None
        for c in self.src:
            p = self.stress_parts_channel(c, c.ref)
            p = {k_: p[k_] - c.stress0[k_] for k_ in p}
            p["sy_k"] = c.kappa * p["sy"]
            P = {k_: c.w * v_ for k_, v_ in p.items()} if P is None else {k_: P[k_] + c.w * p[k_] for k_ in P}
        sync(self.dev)
        self.TM["light_stress_s"] += time.time() - t_
        return P

    # ------------------------------------------------------------------------------------------------------------ H2: sources
    def _eig_at(self, ch, N, v):
        self.TM["light_eigh_n"] += 1
        return self.E19.eigT(self.E19.clocked_gpu(ch, N, v))

    def continue_ref(self, c, f, E, commit=True):
        a = self.a
        Nv, vv = self.fields_on(f, E, self.n_v)
        e, U = c.eig_v(Nv, vv); self.TM["light_eigh_n"] += 1
        if a.light_ref == "killing":
            ref = U[:, e < 0]; mask = (e < 0).cpu().numpy()
            cinfo = {"mode": "killing", "nsub": 1, "verified": None, "margin_min": None, "n_pocket": 0, "nneg": int(mask.sum()), "gap_between_blocks": None}
        else:
            ref, cinfo, mask = OPS2.continue_sea_v2(c.ch_v, c.win, (Nv, vv), e, U, c.n_sea_v, a.tau, a.c_LZ, self._eig_at, self.E19.clocked_host_sparse, a.window_max)
        if commit:
            c.win.append({"ref": ref, "f": (Nv.copy(), vv.copy()), "gap": cinfo.get("gap_between_blocks")})
            c.win = c.win[-a.window_max:]
            c.ref = ref; c.mask = np.asarray(mask, bool); c.e_v = e; c.U_v = U; c.cinfo = cinfo
        return ref, cinfo, np.asarray(mask, bool), e, U

    def pocket_ledger(self, c, e, U, mask, cap=16):
        en = e.cpu().numpy()
        idx = np.where((mask & (en > 0)) | (~mask & (en < 0)))[0][:cap]
        out = []
        if len(idx) == 0:
            return out
        n_v = self.n_v
        for j in idx:
            u = U[:, int(j): int(j) + 1]
            bare = float((u.conj() * (c.ch_v.h @ u)).real.sum())
            up = W1S.to_sector(u, 1); um = W1S.to_sector(u, -1)
            X = c.Phi[:n_v]
            occ = float((X.mH @ up).abs().pow(2).sum()) + float(um.abs().pow(2).sum()) - float((X.mH @ um.conj()).abs().pow(2).sum())
            out.append({"level": int(j), "killing_e": float(en[j]), "bare_e": bare, "occupied_in_ref": bool(mask[j]), "state_occupation": occ})
        return out

    def sources(self, f_ref, E_prev, T, step):
        """H2: per channel the reference continued to the record's lagged fields (record k - 1's accepted fields and charge), the state's and the
        reference's densities on the heavy grid, eps^L = eps^C - eps^ref - Delta_0; the weighted sums over the sourced channels."""
        a = self.a; rs = self.rs
        self.tm_reset()
        eps_l = np.zeros(self.n_e); J_l = np.zeros(self.n_e)
        diag = {"per_channel": {}}
        lag = 0.0; lag_abs = 0.0
        reuse = (a.light_ref_every > 1 and T > a.light_ref_every_after_rs * rs and step % a.light_ref_every != 0)
        for c in self.chans:
            t_r = time.time()
            if not reuse:
                ref, cinfo, mask, e, U = self.continue_ref(c, f_ref, E_prev, commit=True)
            else:
                ref, cinfo, mask, e, U = c.ref, dict(c.cinfo or {}, mode="reused (--light-ref-every)"), c.mask, c.e_v, c.U_v
            sync(self.dev); self.TM["light_ref_s"] += time.time() - t_r
            t_d = time.time()
            eC, jC = c.state_density_heavy()
            eR, jR = c.ref_density(ref)
            eL = eC - eR - c.D0_eps; jL = jC - jR - c.D0_J
            sync(self.dev); self.TM["light_dens_s"] += time.time() - t_d
            lg = c.w * float((eR - c.eps_ref_prev).sum()); lga = c.w * float(np.abs(eR - c.eps_ref_prev).sum())
            c.eps_ref_prev = eR
            if c.sourced:
                eps_l += c.w * eL; J_l += c.w * jL; lag += lg; lag_abs += lga
            en = e.cpu().numpy()
            pk = self.pocket_ledger(c, e, U, mask) if (cinfo.get("n_pocket") or 0) > 0 else []
            diag["per_channel"][str(c.kappa)] = {"E_hC": float(eC.sum()), "E_shells": c.w * float(eL.sum()), "J_shells": c.w * float(jL.sum()),
                                                "ref_mode": cinfo.get("mode"), "ref_nsub": cinfo.get("nsub"), "ref_verified": cinfo.get("verified"),
                                                "ref_margin_min": cinfo.get("margin_min"), "n_pocket": cinfo.get("n_pocket"), "nneg_ref": int((en < 0).sum()),
                                                "mask_equals_negative_set": bool(np.array_equal(np.asarray(mask, bool), en < 0)), "lag_residual": lg,
                                                "ref_min_occupied_weight": cinfo.get("min_occupied_weight"), "ref_inversions": cinfo.get("inversions"),
                                                "ref_tau_LZ": cinfo.get("tau_LZ"), "pockets": pk, "sourced": c.sourced,
                                                "killing_would_be_abs": c.w * float(np.abs(eC - C5.dens(c.ch_v.h, U[:, e < 0])[: self.n_e] - c.D0_eps).sum()) if (cinfo.get("n_pocket") or 0) > 0 else 0.0}
        self.eps_light, self.J_light = eps_l, J_l
        diag["E_shells"] = float(eps_l.sum()); diag["J_shells"] = float(J_l.sum()); diag["max_abs_eps_light"] = float(np.max(np.abs(eps_l)))
        E0n = self.E0 if (self.E0 and self.E0 > 0) else 1.0
        diag["lag_residual_over_E0"] = lag / E0n; diag["lag_residual_abs_over_E0"] = lag_abs / E0n
        diag["reused_reference"] = bool(reuse)
        self.ldiag = diag
        return eps_l, J_l, diag

    def sources_trial(self, f, E):
        """H3 with --light-ref-iterate 1: the reference continued from the committed window to the trial fields (not committed)."""
        t_r = time.time()
        eps_l = np.zeros(self.n_e); J_l = np.zeros(self.n_e)
        for c in self.src:
            ref, cinfo, mask, e, U = self.continue_ref(c, f, E, commit=False)
            eC, jC = c.state_density_heavy()
            eR, jR = c.ref_density(ref)
            eps_l += c.w * (eC - eR - c.D0_eps); J_l += c.w * (jC - jR - c.D0_J)
        self.TM["light_ref_s"] += time.time() - t_r
        return eps_l, J_l

    # ------------------------------------------------------------------------------------------------------------ H4: M1
    def _inside(self, c, hp, cut):
        """Re Tr[P_in h C] (P_in the first cut sites) by the sector identity on the clocked h_+ rows."""
        B = band_rows(hp, min(cut + 2, self.n_L), self.dev)
        return float(sector_density(B, c.Phi, cut, np.real(hp.diagonal())).sum())

    def _inside_ref(self, c, f, E, cut, ref):
        E19 = self.E19
        Nv, vv = self.fields_on(f, E, self.n_v)
        H = E19.clocked_host_sparse(c.ch_v, Nv, vv)
        return E19.inside_energy_sp(E19.dev_csr(H[: 2 * cut, :]), ref, np.ones(ref.shape[1]), 2 * cut), H

    def m1_old(self, f_old, E_old, cut):
        """the old configuration's terms (record k - 1's fields, the current state and the record's reference): Y1L, Y2L."""
        t_ = time.time()
        for c in self.chans:
            N_L, v_L = self.fields_on(f_old, E_old, min(self.n_L, cut + 8))
            hp = W1S.sector_plus_csr(OPS.clocked_csr(c.h0[: 2 * len(N_L), : 2 * len(N_L)], c.Q[: 2 * len(N_L), : 2 * len(N_L)], N_L, v_L)).tocsr()
            Y1 = self._inside(c, hp, cut)
            Y2, _ = self._inside_ref(c, f_old, E_old, cut, c.ref)
            c.m1_old = (Y1, Y2)
        self.TM["light_m1_s"] += time.time() - t_
        return None

    def m1_new(self, f, E, cut):
        """X1L, X2L at the record's accepted fields; I_E^L at the M1 cut; H_in^L = X1L - X2L relative to record 0."""
        t_ = time.time()
        out = {"per_channel": {}}
        Hin = 0.0; IE = 0.0; Hin_raw = 0.0
        for c in self.chans:
            nloc = min(self.n_L, cut + 8)
            N_L, v_L = self.fields_on(f, E, nloc)
            hp = W1S.sector_plus_csr(OPS.clocked_csr(c.h0[: 2 * nloc, : 2 * nloc], c.Q[: 2 * nloc, : 2 * nloc], N_L, v_L)).tocsr()
            X1 = self._inside(c, hp, cut)
            X2, Hv = self._inside_ref(c, f, E, cut, c.ref)
            lo = max(cut - 8, 0); hi = min(cut + 8, nloc)
            hs_ = hp[lo:hi, lo:hi]
            Js = W1S.current_out_plus_csr(hs_, np.arange(lo, hi, dtype=float) + 1.0, float(cut)).toarray()
            Jt = torch.as_tensor(Js, dtype=CD, device=self.dev)
            ie_state = -2.0 * local_trace_sector(Jt, c.Phi[lo:hi])                    # Tr[C J_in] = -Tr[C J_out] (v19's sign: +d<H_in>/dt)
            ie_ref = self.E19.energy_cut_current_sp(Hv, c.ref, np.ones(c.ref.shape[1]), cut)
            raw = X1 - X2
            if c.m1_off is None:
                c.m1_off = raw
            prev = c.m1_cur
            c.m1_cur = {"X1": X1, "X2": X2, "cut": cut}
            d = {"X1": X1, "X2": X2, "H_in_raw": raw, "H_in": raw - c.m1_off, "I_E": ie_state - ie_ref}
            if prev is not None and hasattr(c, "m1_old") and c.m1_old is not None:
                Y1, Y2 = c.m1_old
                a_exact = Y1 - prev["X1"]; b_work = (X1 - X2) - (Y1 - Y2); c_reset = -(Y2 - prev["X2"]); dH = (X1 - X2) - (prev["X1"] - prev["X2"])
                fl = c.last_flow.get("m1cut")
                d["balance"] = {"dHin": dH, "evolution_exact": a_exact, "field_work": b_work, "reference_reset": c_reset, "closure_algebraic": a_exact + b_work + c_reset - dH,
                                "evolution_plus_flow": (a_exact + fl) if fl is not None else None, "cut_changed": bool(prev["cut"] != cut)}
            if c.sourced:
                Hin += c.w * d["H_in"]; IE += c.w * d["I_E"]; Hin_raw += c.w * raw
            out["per_channel"][str(c.kappa)] = d
        out["H_in"] = Hin; out["I_E"] = IE; out["H_in_raw"] = Hin_raw
        bal = [(c.w, out["per_channel"][str(c.kappa)].get("balance")) for c in self.src]
        if all(b_ is not None for _, b_ in bal) and bal:
            s = lambda k_: sum(w_ * (b_[k_] or 0.0) for w_, b_ in bal)
            out["balance"] = {"dHin_light": s("dHin"), "evolution_light": s("evolution_exact"), "field_work_light": s("field_work"),
                              "reference_reset_light": s("reference_reset"), "closure_light_algebraic": s("closure_algebraic"),
                              "closure_light_evolution_plus_flow": s("evolution_plus_flow") if all(b_.get("evolution_plus_flow") is not None for _, b_ in bal) else None}
        self.TM["light_m1_s"] += time.time() - t_
        self.m1 = out
        return out

    # ------------------------------------------------------------------------------------------------------------ H7: step
    def sourced_profile(self, E, ell):
        """R3: the light's own Gauss law beyond the collar: W' = 2 g N^2 eps^L per site, (ln N)' = -g J^L/(R q), q = -sqrt(U/R), U = W/N^2, from
        W = 2 g E, N = 1 at the collar; eps^L beyond n_v is eps^C(t) - eps^C(T_0) (R3's clarification), on n_e < n <= n_v eps^C - eps^ref - Delta_0
        on the reference chain's grid (here approximated by the T_0 difference too, labelled). Returns (N_ext, v_ext, W_end, N_end)."""
        epsx = np.zeros(self.n_L - self.n_e); Jx = np.zeros(self.n_L - self.n_e)
        for c in self.src:
            eF, jF = c.state_density_full()
            epsx += c.w * (eF - c.eps_C0_full)[self.n_e:]; Jx += c.w * (jF - c.J_C0_full)[self.n_e:]
        epsx = gaussian_smooth(epsx, ell); Jx = gaussian_smooth(Jx, ell)
        R = np.arange(self.n_e, self.n_L, dtype=float) + 1.0
        W = 2.0 * self.g * max(E, 0.0); lnN = 0.0
        Nx = np.empty(len(R)); vx = np.empty(len(R))
        for i in range(len(R)):
            N = math.exp(lnN)
            W += 2.0 * self.g * N * N * epsx[i]
            U = max(W / (N * N), 0.0); q = -math.sqrt(U / R[i])
            if q < 0:
                lnN += -self.g * Jx[i] / (R[i] * q)
            Nx[i] = math.exp(lnN); vx[i] = -math.sqrt(max(W, 0.0) / R[i])
        return Nx, vx, W, math.exp(lnN)

    def step(self, f, E, cut, tau, ell):
        """H7: per channel the record's piecewise-constant h_+ (the accepted fields on the heavy grid, the exterior at E), the flows at the collar
        (R > n_e - 1, R2), the M1 cut and the fixed cuts (before and after the propagation under the same h, one apply each), L_at_t at the
        collar, the propagation by propagate_lean (--light-sub sub-steps under the same h); the exterior work (vacuum exterior) and the sourced
        profile for the next record."""
        a = self.a; dev = self.dev; rs = self.rs
        cuts = {"total": 0, "collar": self.n_e - 1, "m1cut": cut}
        for lab, m in self.cut_sites.items():
            cuts["cut_%s" % lab] = m
        t_x = time.time()
        if self.exterior == "sourced":
            Nx, vx, Wend, Nend = self.sourced_profile(E, ell)
            self.ext_profile = (Nx, vx); self.E_end_src = Wend / (2.0 * self.g); self.N_end = Nend
        self.TM["light_ext_s"] += time.time() - t_x
        work = 0.0
        if self.exterior == "vacuum" and a.light_exterior_work == "on" and self.ext_E_used is not None and self.ext_E_used > 0:
            dE = float(E) - self.ext_E_used
            Rx = np.arange(self.n_e, self.n_L, dtype=float) + 1.0
            vxv = -np.sqrt(2.0 * self.g * max(E, 0.0) / Rx)
            for c in self.src:
                _, jF = c.state_density_full()
                work += c.w * (dE / (2.0 * E)) * float(np.sum(vxv * (jF - c.J_C0_full)[self.n_e:]))
            self.W_rad_acc += work
        self.ext_E_used = float(E)
        for c in self.chans:
            N_L, v_L = self.fields_on(f, E, self.n_L)
            hp = c.hp_long(N_L, v_L)
            t_f = time.time()
            B = IR.BandedLean(hp, dev)
            r0 = row_energy(B, c.Phi)
            nc = min(cut + 8, self.n_L); lo = max(self.n_e - 1 - 8, 0); hi = min(self.n_e - 1 + 8, self.n_L)
            Js = W1S.current_out_plus_csr(hp[lo:hi, lo:hi], np.arange(lo, hi, dtype=float) + 1.0, float(self.n_e - 1)).toarray()
            c.L_at_t = 2.0 * local_trace_sector(torch.as_tensor(Js, dtype=CD, device=dev), c.Phi[lo:hi])
            sync(dev); self.TM["light_flow_s"] += time.time() - t_f
            t_p = time.time()
            nsub = max(1, int(a.light_sub)); Kmax = 0; tail = 0.0
            for _ in range(nsub):
                c.Phi, K_, tl = IR.propagate_lean(B, c.Phi, tau / nsub, a.light_cheb_tol, a.light_chunk)
                Kmax = max(Kmax, int(K_)); tail += float(tl)
            sync(dev); self.TM["light_prop_s"] += time.time() - t_p
            t_f = time.time()
            r1 = row_energy(B, c.Phi)
            d = (2.0 * (r1 - r0)).cpu().numpy()
            suf = np.concatenate([np.cumsum(d[::-1])[::-1], [0.0]])     # suf[i] = sum_{n >= i} d_n: the energy that left the sites < i
            c.last_flow = {lab: float(suf[min(int(i), self.n_L)]) for lab, i in cuts.items()}
            for lab, v_ in c.last_flow.items():
                c.flow_acc[lab] = c.flow_acc.get(lab, 0.0) + v_
            c.cheb_K_max = max(c.cheb_K_max, Kmax); c.cheb_tail_acc += tail; c.n_substeps += nsub; c.cheb_K_last = Kmax; c.cheb_tail_last = tail
            del B
            sync(dev); self.TM["light_flow_s"] += time.time() - t_f
        self.E_rad_acc += sum(c.w * c.last_flow["collar"] for c in self.src)
        self.last_tau = float(tau)
        return None

    # ------------------------------------------------------------------------------------------------------------ H6: record
    def record(self, T, E, Rt, kT, cut, step, eps_light, converged):
        """the per-record light block (3.3 with the M2 ledger) and the stop rule (M2 d)."""
        a = self.a; rs = self.rs; E0 = self.E0
        rec = {"E_shells": float(eps_light.sum()), "J_shells": float(self.J_light.sum()),
               "E_shells_inside_root": (float(eps_light[self.R_e <= Rt].sum()) if Rt else None),
               "E_shells_killing_would_be": sum(d_.get("killing_would_be_abs", 0.0) for d_ in self.ldiag.get("per_channel", {}).values() if d_.get("sourced")),
               "lag_residual_over_E0": self.ldiag.get("lag_residual_over_E0"), "lag_residual_abs_over_E0": self.ldiag.get("lag_residual_abs_over_E0"),
               "E_rad_acc": self.E_rad_acc, "E_end_vac": float(E) + self.E_rad_acc,
               "E_end_src": getattr(self, "E_end_src", None), "N_end": getattr(self, "N_end", None), "W_rad_acc": self.W_rad_acc,
               "first_return_rs": self.first_return_rs(E), "per_channel": {}}
        if "collar" in (self.src[0].last_flow if self.src else {}):
            for lab in self.src[0].last_flow:
                rec["flow_%s" % lab] = sum(c.w * c.last_flow[lab] for c in self.src)
            rec["L_record"] = rec["flow_collar"] / max(getattr(self, "last_tau", a.tau), 1e-300)
            rec["L_at_t"] = sum(c.w * (c.L_at_t or 0.0) for c in self.src)
        rec["H_in_light"] = getattr(self, "m1", {}).get("H_in")
        E0 = E0 if (E0 and E0 > 0) else 1.0
        rec["ledger_Hin_plus_Erad_over_E0"] = ((rec["H_in_light"] + self.E_rad_acc) / E0) if rec["H_in_light"] is not None else None
        for c in self.chans:
            pc = dict(self.ldiag.get("per_channel", {}).get(str(c.kappa), {}))
            pc.update({"flow": dict(c.last_flow), "flow_acc": dict(c.flow_acc), "cheb_K": c.cheb_K_last, "cheb_tail": c.cheb_tail_last, "L_at_t": c.L_at_t})
            rec["per_channel"][str(c.kappa)] = pc
        guard = [c for c in self.chans if not c.sourced and c.kappa != 0]
        if guard and self.src and abs(sum(c.w * c.flow_acc.get("collar", 0.0) for c in self.src)) > 0:
            tot = sum(c.w * c.flow_acc.get("collar", 0.0) for c in self.src)
            rec["guard_share_collar_flow"] = sum(c.w * c.flow_acc.get("collar", 0.0) for c in guard) / tot
        # the stop rule (M2 d)
        if self.prev_rec is not None and a.b1_stop_rule == "on":
            dEe = abs(rec["E_end_vac"] - self.prev_rec["E_end_vac"])
            ins0 = self.prev_rec.get("E_shells_inside_root"); ins1 = rec.get("E_shells_inside_root")
            dins = abs((ins1 or 0.0) - (ins0 or 0.0)) if (ins0 is not None or ins1 is not None) else 0.0
            if dEe > a.b1_stop_dEend * E0:
                self.stop_reason = "stop rule: |Delta E_end| = %.3e E_0 > %.1e E_0 in one record (T/r_s %.3f)" % (dEe / E0, a.b1_stop_dEend, T / rs)
            elif dins > a.b1_stop_inside * E0:
                self.stop_reason = "stop rule: the fed inside-root total changed by %.3e E_0 > %.2f E_0 in one record (T/r_s %.3f)" % (dins / E0, a.b1_stop_inside, T / rs)
        rec["stop_reason"] = self.stop_reason
        rec["timing"] = dict(self.TM)
        self.prev_rec = {"E_end_vac": rec["E_end_vac"], "E_shells_inside_root": rec["E_shells_inside_root"]}
        return rec

    # ------------------------------------------------------------------------------------------------------------ readouts
    def comparator(self, E, T):
        try:
            import page_history_v6q as V6Q
            return V6Q.continuum_comparator([c.kappa for c in self.src], 2.0 * self.g * E, T)
        except Exception as ex:                                                          # noqa: BLE001
            self.cmp_error = "%s: %s" % (type(ex).__name__, str(ex)[:200])
            return None

    def readout(self, T, E, Rt, kT, f, step, out_dir, tag, impurity=False, contour=False, rows=False):
        """every --light-readout-every-rs: per channel the interior blocks at the cuts (2, 4, 8, 16 r_s0 and the root), S_I(C), S_I(P_-^v) at
        the record's fields (the truncated chain's Killing vacuum), the excess with Delta S_0, the excess over R2, the floors, the capacities,
        S_BH at both areas, the purity identity, the fluxes against L_H, beta_eff over the window; the raw blocks into a readout npz."""
        a = self.a; rs = self.rs; t0 = time.time(); dev = self.dev
        cuts = dict(self.cut_sites)
        if Rt:
            cuts["root"] = int(np.sum(self.R_e <= Rt))
        cuts = {k_: v_ for k_, v_ in cuts.items() if v_ >= 1}
        mmax = max(cuts.values())
        res = {"T_over_rs": T / rs, "cuts": {}, "per_channel": {}}
        arrays = {"T": T, "cut_labels": np.array(list(cuts.keys())), "cut_sites": np.array(list(cuts.values()))}
        tot = {lab: {"S_I": 0.0, "S_I_vac": 0.0, "S_I_R2": 0.0, "excess": 0.0, "excess_R2": 0.0, "excess_floor_low": 0.0, "excess_floor_high": 0.0} for lab in cuts}
        for c in self.chans:
            Nv, vv = self.fields_on(f, E, self.n_v)
            e, U = c.eig_v(Nv, vv)
            Yv = W1S.to_sector(U[:, e < 0], 1)[:mmax]; Pv = Yv @ Yv.mH
            Yr = W1S.to_sector(c.ref, 1)[:mmax]; Pr = Yr @ Yr.mH
            G = c.Phi[:mmax] @ c.Phi[:mmax].mH
            pcut = {}
            for lab, m in cuts.items():
                SC, xC = block_entropy(G[:m, :m]); SV, _ = block_entropy(Pv[:m, :m]); SR, _ = block_entropy(Pr[:m, :m])
                dS0 = float(c.dS0[m]) if m < len(c.dS0) else 0.0
                pcut[lab] = {"n_sites": m, "S_I": SC, "S_I_vac": SV, "S_I_R2": SR, "excess": SC - SV - dS0, "excess_R2": SC - SR, "dS0": dS0, "excursion_C": xC}
            imp = W1S.impurity_trace_norm(c.Phi) if impurity else None
            fl = W1S.propagation_floor(c.n_plus, c.cheb_tail_acc, c.n_substeps, c.cheb_K_max, T_meas=imp, T_init=c.T_init)
            fl["q"] = c.n_plus
            W1S.excess_floors(pcut, self.R_e, [(lab, float(self.R_e[m - 1]) + 0.5) for lab, m in cuts.items()], fl, sector=True)
            # purity identity at the smallest fixed cut: ||Phi_I (Phi^dagger Phi - 1) Phi_I^dagger||_F
            m2 = min(cuts.values())
            XI = c.Phi[:m2]
            M = XI @ c.Phi.mH
            pur = float(torch.linalg.norm(M @ M.mH - XI @ XI.mH))
            del M
            cont = None
            if contour:
                hp = c.hp_long(*self.fields_on(f, E, self.n_L))
                Pc, ci = IR.sign_block_contour(hp, mmax, threads=int(a.light_threads))
                PcT = torch.as_tensor(Pc, dtype=CD, device=dev)
                cont = {"info": {k_: v_ for k_, v_ in ci.items() if isinstance(v_, (int, float, str))}, "cuts": {}}
                for lab, m in cuts.items():
                    SPc = block_entropy(PcT[:m, :m])[0]
                    cont["cuts"][lab] = {"S_I_vac_long_contour": SPc, "truncated_plus_dS0_minus_contour": pcut[lab]["S_I_vac"] + pcut[lab]["dS0"] - SPc}
            res["per_channel"][str(c.kappa)] = {"cuts": pcut, "impurity": imp, "floor": {k_: v_ for k_, v_ in fl.items() if k_ != "label"}, "purity_residual": pur,
                                                "contour": cont, "sourced": c.sourced}
            arrays["C_block_%d" % c.kappa] = G.cpu().numpy(); arrays["Pvac_block_%d" % c.kappa] = Pv.cpu().numpy(); arrays["R2_block_%d" % c.kappa] = Pr.cpu().numpy()
            if rows:
                arrays["Phi_rows_%d" % c.kappa] = c.Phi[:mmax].cpu().numpy()
            if c.sourced:
                for lab in cuts:
                    for k_ in tot[lab]:
                        v_ = pcut[lab].get(k_)
                        if v_ is not None:
                            tot[lab][k_] += c.w * v_
            del G, Pv, Pr, Yv, Yr
        for lab, m in cuts.items():
            Nm = sum(c.w * 2 * m for c in self.src)
            tot[lab].update({"n_sites": m, "N_modes": Nm, "capacity_ln2": Nm * LN2, "capacity_BHK": BHK * Nm})
        res["cuts"] = tot
        res["S_BH_root"] = (math.pi * Rt * Rt / self.g) if Rt else None
        res["S_BH_2gE"] = 4.0 * math.pi * self.g * E * E
        res["S_BH_label"] = "S_BH at the root pi R_t^2/g and at 2gE: 4 pi g E^2, both printed, never interchanged (reviewer B)"
        res["root_present"] = bool(Rt)
        T_H = 1.0 / (8.0 * math.pi * self.g * E) if E > 0 else None
        res["T_H"] = T_H
        if a.light_comparator == "on":
            res["L_H_TH"] = self.comparator(E, T_H)
            res["L_H_kappaT"] = self.comparator(E, (kT / (2.0 * math.pi)) if kT else None)
        if self.src and self.src[0].last_flow:
            L = sum(c.w * c.last_flow.get("collar", 0.0) for c in self.src) / max(getattr(self, "last_tau", a.tau), 1e-300)
            res["L_record"] = L
            for k_ in ("L_H_TH", "L_H_kappaT"):
                if res.get(k_):
                    res["L_over_" + k_[2:]] = L / res[k_]
        # the window since the last readout: E_rad, the 4 r_s cut's flow and Delta S, beta_eff
        lab4 = "%g" % 4.0 if ("%g" % 4.0) in tot else (list(self.cut_sites)[0] if self.cut_sites else None)
        flow4 = sum(c.w * c.flow_acc.get("cut_%s" % lab4, 0.0) for c in self.src) if lab4 else None
        if self.last_readout is not None:
            lr = self.last_readout
            res["E_rad_window"] = self.E_rad_acc - lr["E_rad_acc"]
            res["dS_window"] = {lab: tot[lab]["excess"] - lr["excess"].get(lab, 0.0) for lab in tot if lab in lr["excess"]}
            if lab4 and flow4 is not None:
                ef = flow4 - lr["flow4"]
                res["E_flow4_window"] = ef
                res["beta_eff_window"] = (res["dS_window"].get(lab4, 0.0) * T_H / ef) if (T_H and abs(ef) > 0) else None
        self.last_readout = {"E_rad_acc": self.E_rad_acc, "excess": {lab: tot[lab]["excess"] for lab in tot}, "flow4": flow4}
        res["seconds"] = time.time() - t0
        self.TM["light_readout_s"] += res["seconds"]
        if out_dir:
            p = os.path.join(out_dir, "collapse_%s_light_ro_T%08.2f.npz" % (tag, T / rs))
            np.savez(p, **arrays)
            res["npz"] = p
        return res

    # ------------------------------------------------------------------------------------------------------------ state
    def state_dict(self):
        out = {"chans": [], "E_rad_acc": self.E_rad_acc, "W_rad_acc": self.W_rad_acc, "E0": self.E0, "ext_profile": self.ext_profile,
               "ext_E_used": self.ext_E_used, "prev_rec": self.prev_rec, "last_readout": self.last_readout, "head": self.head}
        for c in self.chans:
            out["chans"].append({"kappa": c.kappa, "Phi": c.Phi, "win": c.win, "ref": c.ref, "n_sea_v": c.n_sea_v, "mask": c.mask, "D0_eps": c.D0_eps,
                                 "D0_J": c.D0_J, "dS0": c.dS0, "stress0": c.stress0, "cheb_K_max": c.cheb_K_max, "cheb_tail_acc": c.cheb_tail_acc,
                                 "n_substeps": c.n_substeps, "T_init": c.T_init, "flow_acc": c.flow_acc, "last_flow": c.last_flow,
                                 "eps_ref_prev": c.eps_ref_prev, "m1_off": c.m1_off, "m1_cur": c.m1_cur, "n_plus": c.n_plus,
                                 "eps_C0_full": c.eps_C0_full, "J_C0_full": c.J_C0_full, "e_v": c.e_v, "U_v": c.U_v, "cinfo": c.cinfo})
        return out

    def load_state_dict(self, d):
        for k_ in ("E_rad_acc", "W_rad_acc", "E0", "ext_profile", "ext_E_used", "prev_rec", "last_readout", "head"):
            setattr(self, k_, d[k_])
        self.E_prev_rec = self.E0
        byk = {x["kappa"]: x for x in d["chans"]}
        for c in self.chans:
            x = byk[c.kappa]
            for k_, v_ in x.items():
                if k_ != "kappa":
                    setattr(c, k_, v_)
