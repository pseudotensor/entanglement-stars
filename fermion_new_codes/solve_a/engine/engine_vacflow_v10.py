#!/usr/bin/env python
"""engine_vacflow_v10.py (builder analyst; engine spec 26.6: "vf10: the continuation by engine_ops_v3's convention (c_LZ 1, never rejecting,
labelled the lane's declaration), the sums on a common time grid with incomplete channels listed apart, the erf step of width 6 to 8 sites
with a window of at most 48 sites so that the smoothed cut exists at r_s, the cross-band share printed per cut and the band split read
only where it is below 0.1; vf9's profile, ramp and channels"): version 9 with (1) R2 continued at each readout by
engine_ops_v3.continue_sea_v3 on h(s) = h_0 + s V (the record-to-record change ds V, its Gershgorin rate over the readout interval, the
sub-steps along s where tau_LZ is below that interval; never rejected; per readout the mode, the sub-steps, the Weyl verification,
the pockets and inversions recorded) in place of the degeneracy-aware block rule with halving; (2) the channel sums formed after every
channel over the channels that hold the full readout count (a common time grid), any other channel listed apart; (3) the smoothed cut's
width 6 sites and window 48 sites by default, the step's tail at the window's edge (1/2) erfc(24/6) printed; (4) per cut and phase the
smoothed cut's cross-band share |cross|/|total| printed, the middle bands' share read only where the cross-band share is below 0.1, and
"rides the cos 2p < 0 bands" only where both hold (the share above 0.9). Nothing else changes. Version 9's docstring: engine_vacflow_v9.py
(builder analyst; engine spec 19.4: version 9 = version 8 plus the smoothed cut and the balance's parts). (1) Beside each
sharp cut P_c, the smoothed cut F_c = diag(f_n), f_n = (1/2) erfc((R_n - R_c)/w), w = --smooth-width sites (default 7): its current
I_F = (i/2) Tr[D [h(s)^2, F_c]] for D = C - R0, C, C - R2 and the band projection of the D = C - R0 part on a window of --smooth-window sites
(default 64, the step's tail there below 1e-10) about the cut, with the cross-band remainder; (2) the work parts apart: W_C = ds/dt Tr[C V_c]
and W_R0 = ds/dt Tr[R0(s) V_c] (V_c = (1/2){P_c, V}) beside version 8's W = W_C - W_R0, so each cut's balance closes as
Delta Tr[C H_c] = int (I_C + W_C) dt and the reference's own term Delta Tr[R0 H_c] - int W_R0 dt is named (both evaluated by the report from
the readouts); (3) the record's verdict on the smoothed cut: the two middle bands' share of its total outflow over the ramp, and "rides the
cos 2p < 0 bands" only above 0.9. Nothing else changes. Version 8's docstring follows.
engine_vacflow_v8.py (builder analyst; SPEC_final4b_engine.md 18.2, the decisive rerun of J-VF): the fermion vacuum in the exact
Painleve-Gullstrand flow at r_s 32, one ramp t_r = 32 r_s and the hold of 2 r_s, channels kappa = 1..16 (weights 16 kappa), the full chain
current (--chain-current full: the two-step Q plus the nearest-neighbour term of [X, h_0^2]/(2i), engine_vacflow_v5's variant), with R2
continued along the ramp; version 7's chain, profile, propagation (CF4 with Chebyshev exponentials) and E_cut rule (imported from
engine_vacflow_v7). Read out every 0.5 r_s of the ramp and of the hold (--every), per channel and summed:
(1) the energy inside and outside R = r_s (the horizon crossing) of the declared density hcal_n = (1/2){P_n, h_0}: NO (against R0(s), the
    instantaneous negative projector of h(s) = h_0 + s V), R2 (against the continued reference at s) and raw (against the flat sea);
(2) at the cuts R = r_s, 1.5 r_s, 2 r_s (P_c the sites below the cut): the Killing-energy cut current I_c = (i/2) Tr[D [h(s)^2, P_c]] (the
    rate at which the energy inside the cut grows by transport; outflow is -I_c) for D = C - R0(s) (and for C alone and C - R2 as
    checks; R0 commutes with h(s), so the first two agree to rounding), and its window projection: on the 32 sites about the cut, the
    discrete Fourier basis of each spinor component, the bands p > 0 and p < 0 each split at |p| = pi/4 and 3 pi/4 (cos 2p < 0 exactly
    between: the reversed-drag branch of the two-step flow coupling, 12.4), I_c,b = (i/2) Tr[Pi_b D Pi_b [h^2, P_c]] per band and the
    cross-band remainder; the lead's grouping (|p| below and above pi/4) is the sum of the last two bands of each sign;
(3) the local work inside each cut, W_c = ds/dt Tr[(C - R0(s)) (1/2){P_c, V}] (dh/dt = ds/dt V), and the Killing energies inside each cut
    Tr[C H_c(s)] and Tr[R0(s) H_c(s)], H_c = (1/2){P_c, h(s)}, so the balance d/dt Tr[(C - R0) H_c] = I_c + W_c + (the reference's own
    terms) can be read off the record;
(4) the particle and hole counts, orientation declared once (engine 18.2): particles N_p = Tr[R2 (1 - R0)] = n_sea - ||R0^dagger R2||_F^2,
    holes N_h = Tr[R0 (1 - R2)] = n_neg - ||R0^dagger R2||_F^2 (versions 5 to 7 printed these two under each other's names; corrected here).
The fluences per cut and band (-I integrated by the trapezoid over the readouts, the ramp and the hold apart) are summed over the channels,
and the record states whether the outgoing Killing energy at r_s rides the cos 2p < 0 bands. R2's continuation follows the readouts'
s values (engine_collapse_v6.continue_reference_aware, the degeneracy-aware block rule, an ambiguous step halved up to 6 times).
Args: --rs --profile --m --Ecut --kmax --tr --hold --dt --every --chain-current --gpu --out --tag --benchmark K."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse
import json
import math
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine_vacflow_v7 as V7  # noqa: E402

BANDS = ("p>0 |p|<pi/4", "p>0 pi/4<|p|<3pi/4", "p>0 |p|>3pi/4", "p<0 |p|<pi/4", "p<0 pi/4<|p|<3pi/4", "p<0 |p|>3pi/4")


def band_of(p):
    """the band index of a window momentum p in (-pi, pi]: p >= 0 bands 0 to 2, p < 0 bands 3 to 5, split at |p| = pi/4 and 3 pi/4 (the
    boundary momenta join the middle band; p = 0 joins band 0)."""
    q = abs(p)
    k = 0 if q < math.pi / 4 else (1 if q < 3 * math.pi / 4 else 2)
    return k if p >= 0 else 3 + k


def trapezoid(y, x):
    y = np.asarray(y, float); x = np.asarray(x, float)
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x))) if len(x) > 1 else 0.0


class Readout:
    """the per-readout measurements of one channel (torch on the device)."""

    def __init__(self, torch, dev, h0, V, R, rs, cut_radii, half=8, wband=16, smooth_width=7.0, smooth_window=64):
        self.torch, self.dev = torch, dev
        self.h0, self.V = h0.tocsr(), V.tocsr()
        self.R = R
        n = len(R)
        self.n = n
        self.inside_rs = torch.as_tensor(R < rs, device=dev)
        self.cuts = []
        for rc in cut_radii:
            c = int(np.searchsorted(R, rc))           # the sites 0..c-1 lie below the cut
            if 8 < c < n - 24:
                self.cuts.append((rc, c))
        self.half, self.wband = half, wband
        self.sw, self.swin = float(smooth_width), int(smooth_window) // 2
        from scipy.special import erfc as _erfc
        self._erfc = _erfc
        Ls = 2 * self.swin
        ns_ = np.arange(Ls)
        ps_ = 2 * math.pi * (np.arange(Ls) - Ls // 2) / Ls
        self.PiS = []
        for b in range(len(BANDS)):
            sel = [j for j, p in enumerate(ps_) if band_of(p) == b]
            F = np.zeros((2 * Ls, 2 * len(sel)), complex)
            for col, j in enumerate(sel):
                ph = np.exp(1j * ps_[j] * ns_) / math.sqrt(Ls)
                F[0::2, 2 * col] = ph
                F[1::2, 2 * col + 1] = ph
            self.PiS.append(torch.as_tensor(F @ F.conj().T, device=dev))
        csr = lambda A: torch.sparse_csr_tensor(torch.as_tensor(A.indptr, dtype=torch.int64), torch.as_tensor(A.indices, dtype=torch.int64),
                                                torch.as_tensor(A.data, dtype=torch.complex128), size=A.shape, device=dev)
        self.h0_t, self.V_t = csr(self.h0), csr(self.V)
        # the band projectors on the 2 wband sites about a cut (the same for every cut): Fourier per spinor component
        L = 2 * wband
        nloc = np.arange(L)
        ps = 2 * math.pi * (np.arange(L) - L // 2) / L
        self.Pi = []
        for b in range(len(BANDS)):
            sel = [j for j, p in enumerate(ps) if band_of(p) == b]
            F = np.zeros((2 * L, 2 * len(sel)), complex)
            for col, j in enumerate(sel):
                ph = np.exp(1j * ps[j] * nloc) / math.sqrt(L)
                F[0::2, 2 * col] = ph
                F[1::2, 2 * col + 1] = ph
            self.Pi.append(torch.as_tensor(F @ F.conj().T, device=dev))

    def dens(self, A_t, X):
        AX = A_t @ X
        return (X.conj() * AX).real.sum(dim=1).reshape(-1, 2).sum(dim=1)

    def window_comm(self, H, c, half):
        """[h^2, P_c] on the rows and columns of the 2 half sites about the cut c (host sparse products, then dense on the device)."""
        lo, hi = 2 * max(c - half, 0), 2 * min(c + half, self.n)
        h2W = (H[lo:hi, :] @ H[:, lo:hi]).toarray()
        P = np.zeros(hi - lo); P[: 2 * c - lo] = 1.0
        Cm = h2W * P[None, :] - P[:, None] * h2W
        return lo, hi, self.torch.as_tensor(Cm, device=self.dev)

    def smooth_comm(self, H, c, rc):
        """[h^2, F_c] on the 2 swin sites about the cut, F_c = diag((1/2) erfc((R_n - R_c)/w)) (each site's two components)."""
        lo, hi = 2 * max(c - self.swin, 0), 2 * min(c + self.swin, self.n)
        h2W = (H[lo:hi, :] @ H[:, lo:hi]).toarray()
        Rw = np.repeat(self.R[lo // 2: hi // 2], 2)
        f = 0.5 * self._erfc((Rw - rc) / self.sw)
        Cm = h2W * f[None, :] - f[:, None] * h2W
        return lo, hi, self.torch.as_tensor(Cm, device=self.dev)

    def measure(self, s, sdot, X, R0, R2, d_flat):
        torch = self.torch
        H = (self.h0 + s * self.V).tocsr()
        H_t = self.h0_t if s == 0.0 else None
        out = {"s": s, "sdot": sdot}
        dC = self.dens(self.h0_t, X); d0 = self.dens(self.h0_t, R0)
        dR2 = self.dens(self.h0_t, R2) if R2 is not None else None
        ins = self.inside_rs
        out["E_in_NO"] = float((dC - d0)[ins].sum()); out["E_out_NO"] = float((dC - d0)[~ins].sum())
        out["E_in_raw"] = float((dC - d_flat)[ins].sum()); out["E_out_raw"] = float((dC - d_flat)[~ins].sum())
        if dR2 is not None:
            out["E_in_R2"] = float((dC - dR2)[ins].sum()); out["E_out_R2"] = float((dC - dR2)[~ins].sum())
        # the Killing densities (h(s)) and the work densities (V) for the cuts
        Hs_t = torch.sparse_csr_tensor(torch.as_tensor(H.indptr, dtype=torch.int64), torch.as_tensor(H.indices, dtype=torch.int64),
                                       torch.as_tensor(H.data, dtype=torch.complex128), size=H.shape, device=self.dev) if H_t is None else H_t
        kC = self.dens(Hs_t, X); k0 = self.dens(Hs_t, R0)
        vC = self.dens(self.V_t, X); v0 = self.dens(self.V_t, R0)
        cuts = {}
        for rc, c in self.cuts:
            rec = {"site": c}
            rec["E_K_C"] = float(kC[:c].sum()); rec["E_K_R0"] = float(k0[:c].sum())
            rec["W"] = float(sdot * (vC[:c].sum() - v0[:c].sum()))
            rec["W_C"] = float(sdot * vC[:c].sum()); rec["W_R0"] = float(sdot * v0[:c].sum())
            lo, hi, Cm = self.window_comm(H, c, self.half)
            XW, RW = X[lo:hi, :], R0[lo:hi, :]
            DC = XW @ XW.conj().T
            D0 = RW @ RW.conj().T
            tr = lambda D: float((0.5j * (D * Cm.T).sum()).real)
            rec["I_C"] = tr(DC); rec["I_NO"] = tr(DC - D0)
            if R2 is not None:
                R2W = R2[lo:hi, :]
                rec["I_R2"] = tr(DC - R2W @ R2W.conj().T)
            # the window projection on 2 wband sites about the cut
            lo2, hi2, Cm2 = self.window_comm(H, c, self.wband)
            if hi2 - lo2 == 4 * self.wband:
                X2, R02 = X[lo2:hi2, :], R0[lo2:hi2, :]
                D = X2 @ X2.conj().T - R02 @ R02.conj().T
                trb = lambda D_: float((0.5j * (D_ * Cm2.T).sum()).real)
                Ib = [trb(P @ D @ P) for P in self.Pi]
                rec["I_NO_window"] = trb(D)
                rec["I_bands"] = Ib
                rec["I_cross"] = rec["I_NO_window"] - sum(Ib)
            # the smoothed cut (engine spec 19.4)
            loS, hiS, CmS = self.smooth_comm(H, c, rc)
            if hiS - loS == 4 * self.swin:
                XS, R0S = X[loS:hiS, :], R0[loS:hiS, :]
                DCs = XS @ XS.conj().T
                D0s = R0S @ R0S.conj().T
                trS = lambda D_: float((0.5j * (D_ * CmS.T).sum()).real)
                Ds = DCs - D0s
                rec["IS_C"] = trS(DCs); rec["IS_NO"] = trS(Ds)
                if R2 is not None:
                    R2S = R2[loS:hiS, :]
                    rec["IS_R2"] = trS(DCs - R2S @ R2S.conj().T)
                IbS = [trS(P @ Ds @ P) for P in self.PiS]
                rec["IS_bands"] = IbS
                rec["IS_cross"] = rec["IS_NO"] - sum(IbS)
            cuts["%g" % rc] = rec
        out["cuts"] = cuts
        return out


def run_channel(kappa, Rfull, a, dev, every, benchmark=False):
    import torch
    import engine_collapse_v6 as C6
    C6.BK["torch"] = torch
    rs = a.rs
    n0 = int(np.searchsorted(Rfull, kappa / a.Ecut))
    R = Rfull[n0:]
    h0, Q = V7.chain_sparse(R, a.m, kappa)
    V = V7.flow_term(Q, V7.profile(R, rs, a.profile))
    eig = lambda s: torch.linalg.eigh(torch.as_tensor((h0 + s * V).toarray(), device=dev))
    e0, U0 = eig(0.0)
    nsea = int((e0 < 0).sum())
    X = U0[:, e0 < 0].clone()
    flat = X.clone()
    ro = Readout(torch, dev, h0, V, R, rs, (rs, 1.5 * rs, 2 * rs), smooth_width=a.smooth_width, smooth_window=a.smooth_window)
    d_flat = ro.dens(ro.h0_t, flat)
    P = V7.Prop(h0, V, dev)
    t_r = a.tr * rs
    env = lambda t: math.sin(math.pi * t / (2 * t_r)) ** 2
    denv = lambda t: (math.pi / (2 * t_r)) * math.sin(math.pi * t / t_r)
    nst = max(1, int(math.ceil(t_r / a.dt - 1e-9)))
    h = t_r / nst
    per = max(1, int(round(every / h)))
    import engine_ops_v3 as OPS3
    R2 = X.clone(); s_prev = 0.0
    reads = []
    halv_tot = 0
    t0 = time.time()
    e0n = e0.real.cpu().numpy(); gap0 = float(e0n[nsea] - e0n[nsea - 1]) if 0 < nsea < len(e0n) else None
    window = [{"ref": R2, "f": (0.0, 0.0), "gap": gap0, "weyl_gap": gap0}]
    Vh = V.tocsr()
    h_of = lambda ch_, ds_, dv_: Vh * float(ds_)
    eig_at = lambda ch_, s_, v_: eig(float(s_))
    cstat = {"readouts": 0, "not_verified": 0, "substeps_max": 1, "pockets_max": 0, "inversions_max": 0, "first_unverified_s": None}
    last_ci = {}

    def continue_to(s_new):
        nonlocal R2, s_prev, window, last_ci
        if s_new == s_prev:
            return None
        e, U = eig(s_new)
        ref, ci, mask = OPS3.continue_sea_v3(None, window, (s_new, 0.0), e.real, U, nsea, every, a.c_LZ, eig_at, h_of)
        window = [{"ref": ref, "f": (s_new, 0.0), "gap": ci.get("gap_between_blocks"), "weyl_gap": ci.get("weyl_gap"), "mask": mask}]
        cstat["readouts"] += 1
        if not ci.get("verified"):
            cstat["not_verified"] += 1
            if cstat["first_unverified_s"] is None:
                cstat["first_unverified_s"] = s_new
        cstat["substeps_max"] = max(cstat["substeps_max"], int(ci.get("nsub") or 1))
        cstat["pockets_max"] = max(cstat["pockets_max"], int(ci.get("n_pocket") or 0))
        cstat["inversions_max"] = max(cstat["inversions_max"], int(ci.get("inversions") or 0))
        last_ci = {k_: ci.get(k_) for k_ in ("mode", "nsub", "verified", "n_pocket", "inversions", "tau_LZ", "min_occupied_weight", "ambiguous_tie")}
        R2 = ref; s_prev = s_new
        R0 = U[:, e < 0]
        return (e, R0)

    def readout(t, s, sdot):
        res = continue_to(s) if s != s_prev or not reads else None
        if isinstance(res, str):
            return res
        if res is None:
            e, U = eig(s); R0 = U[:, e < 0]; e_ = e
        else:
            e_, R0 = res
        nneg = int((e_ < 0).sum())
        ov = float((R0.conj().T @ R2).abs().pow(2).sum())
        m = ro.measure(s, sdot, X, R0, R2, d_flat)
        m.update({"t_over_rs": t / rs, "N_p": float(nsea - ov), "N_h": float(nneg - ov), "nneg": nneg, "nsea": nsea, "phase": "ramp" if t <= t_r + 1e-9 else "hold",
                  "continuation": dict(last_ci)})
        reads.append(m)
        return None
    why = readout(0.0, 0.0, 0.0)
    for i in range(nst):
        if why:
            break
        t = i * h
        s1, s2 = env(t + V7.CF_C[0] * h), env(t + V7.CF_C[1] * h)
        X = P.expm(X, h / 2, 2 * (V7.CF_A[0] * s1 + V7.CF_A[1] * s2))
        X = P.expm(X, h / 2, 2 * (V7.CF_A[1] * s1 + V7.CF_A[0] * s2))
        if (i + 1) % per == 0 or i + 1 == nst:
            tt = (i + 1) * h
            why = readout(tt, env(tt), denv(tt))
        if benchmark and i + 1 >= 2 * per:
            break
    nh = 0 if benchmark else max(1, int(round(a.hold * rs / every)))
    for j in range(nh):
        if why:
            break
        X = P.expm(X, a.hold * rs / nh, 1.0)
        why = readout(t_r + (j + 1) * a.hold * rs / nh, 1.0, 0.0)
    return {"kappa": kappa, "n0": n0, "nsea": nsea, "halvings": halv_tot, "reads": reads, "rejected": why, "seconds": time.time() - t0, "bound": P.bound,
            "continuation": dict(cstat)}


def fluences(reads, cut, rs):
    """-I integrated over time (the trapezoid over the readouts, t in lattice time units), the ramp and the hold apart (per band, the cross
    remainder and the total; the work inside the cut likewise)."""
    out = {}
    for phase in ("ramp", "hold"):
        rr = [q for q in reads if q["phase"] == phase]
        if phase == "hold":
            last_ramp = [q for q in reads if q["phase"] == "ramp"][-1:]
            rr = last_ramp + rr
        if len(rr) < 2:
            continue
        t = np.array([q["t_over_rs"] for q in rr]) * rs
        tot = np.array([-q["cuts"][cut]["I_NO"] for q in rr])
        res = {"total": float(trapezoid(tot, t))}
        if all("I_bands" in q["cuts"][cut] for q in rr):
            B = np.array([[-x for x in q["cuts"][cut]["I_bands"]] for q in rr])
            res["bands"] = [float(trapezoid(B[:, b], t)) for b in range(B.shape[1])]
            res["cross"] = float(trapezoid(np.array([-q["cuts"][cut]["I_cross"] for q in rr]), t))
            res["window_total"] = float(trapezoid(np.array([-q["cuts"][cut]["I_NO_window"] for q in rr]), t))
        res["work"] = float(trapezoid(np.array([q["cuts"][cut]["W"] for q in rr]), t))
        cq = lambda key: np.array([q["cuts"][cut].get(key, np.nan) for q in rr], float)
        res["total_C"] = float(trapezoid(-cq("I_C"), t))
        if all("I_R2" in q["cuts"][cut] for q in rr):
            res["total_R2"] = float(trapezoid(-cq("I_R2"), t))
        res["work_C"] = float(trapezoid(cq("W_C"), t)); res["work_R0"] = float(trapezoid(cq("W_R0"), t))
        res["dE_K_C"] = float(rr[-1]["cuts"][cut]["E_K_C"] - rr[0]["cuts"][cut]["E_K_C"]); res["dE_K_R0"] = float(rr[-1]["cuts"][cut]["E_K_R0"] - rr[0]["cuts"][cut]["E_K_R0"])
        res["balance_C"] = res["dE_K_C"] - (-res["total_C"] + res["work_C"])
        res["reference_own"] = res["dE_K_R0"] - res["work_R0"]
        if all("IS_bands" in q["cuts"][cut] for q in rr):
            BS = np.array([[-x for x in q["cuts"][cut]["IS_bands"]] for q in rr])
            res["smooth_total_NO"] = float(trapezoid(-cq("IS_NO"), t))
            res["smooth_total_C"] = float(trapezoid(-cq("IS_C"), t))
            if all("IS_R2" in q["cuts"][cut] for q in rr):
                res["smooth_total_R2"] = float(trapezoid(-cq("IS_R2"), t))
            res["smooth_bands"] = [float(trapezoid(BS[:, b], t)) for b in range(BS.shape[1])]
            res["smooth_cross"] = float(trapezoid(-cq("IS_cross"), t))
        out[phase] = res
    return out


def add_channel(tot, c, nfull):
    """adds a channel to the sums when it holds the full readout count and was not rejected, else lists it apart."""
    w = 16.0 * c["kappa"]
    if c["rejected"] or len(c["reads"]) != nfull:
        tot.setdefault("left_out", []).append({"kappa": c["kappa"], "why": c["rejected"] or "readout count %d of %d" % (len(c["reads"]), nfull)})
        return
    tot["channels_summed"].append(c["kappa"])
    for k_ in ("E_in_NO", "E_out_NO", "E_in_R2", "E_out_R2", "E_in_raw", "E_out_raw"):
        tot[k_] += w * np.array([q.get(k_, np.nan) for q in c["reads"]])
    for k_ in ("N_p", "N_h"):
        tot[k_] += w * np.array([q[k_] for q in c["reads"]])
        tot[k_ + "_unweighted"] = tot.get(k_ + "_unweighted", np.zeros(nfull)) + np.array([q[k_] for q in c["reads"]])
    for ck, f in c["fluences"].items():
        T_ = tot["fluences"].setdefault(ck, {})
        for phase, res in f.items():
            P_ = T_.setdefault(phase, {"total": 0.0, "bands": [0.0] * len(BANDS), "cross": 0.0, "window_total": 0.0, "work": 0.0,
                                       "smooth_bands": [0.0] * len(BANDS), "smooth_cross": 0.0})
            P_["total"] += w * res["total"]; P_["work"] += w * res["work"]
            if "bands" in res:
                P_["bands"] = [x + w * y for x, y in zip(P_["bands"], res["bands"])]
                P_["cross"] += w * res["cross"]; P_["window_total"] += w * res["window_total"]
            for key in ("total_C", "total_R2", "work_C", "work_R0", "dE_K_C", "dE_K_R0", "balance_C", "reference_own", "smooth_total_NO", "smooth_total_C", "smooth_total_R2", "smooth_cross"):
                if key in res:
                    P_[key] = P_.get(key, 0.0) + w * res[key]
            if "smooth_bands" in res:
                P_["smooth_bands"] = [x + w * y for x, y in zip(P_["smooth_bands"], res["smooth_bands"])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rs", type=float, default=32.0)
    ap.add_argument("--profile", default="exact", choices=["exact", "notrap"])
    ap.add_argument("--m", type=float, default=0.0)
    ap.add_argument("--Ecut", type=float, default=6.0)
    ap.add_argument("--kmax", type=int, default=16)
    ap.add_argument("--tr", type=float, default=32.0, help="the ramp time in units of r_s")
    ap.add_argument("--hold", type=float, default=2.0, help="the hold in units of r_s")
    ap.add_argument("--dt", type=float, default=0.5)
    ap.add_argument("--every", type=float, default=0.5, help="the readout spacing in units of r_s")
    ap.add_argument("--chain-current", default="full", choices=["two-step", "full"])
    ap.add_argument("--smooth-width", type=float, default=6.0, help="the erf step's width in sites (engine 26.6: 6 to 8)")
    ap.add_argument("--smooth-window", type=int, default=48, help="the smoothed cut's window in sites (engine 26.6: at most 48)")
    ap.add_argument("--c-LZ", dest="c_LZ", type=float, default=1.0, help="engine_ops_v3's convention constant (the lane's declaration)")
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--benchmark", type=int, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    V7.CC["full"] = (a.chain_current == "full")
    import torch
    dev = torch.device("cuda:%d" % a.gpu) if a.gpu is not None else torch.device("cpu")
    rs = a.rs
    Rfull = np.arange(int(64 * rs), dtype=float) + 1.0
    every = a.every * rs
    os.makedirs(a.out, exist_ok=True)
    if a.benchmark:
        c = run_channel(a.benchmark, Rfull, a, dev, every, benchmark=True)
        q = c["reads"][-1]
        print("benchmark kappa %d: %d readouts in %.1f s (bound %.2f, n0 %d); last readout t/rs %.2f s %.4f: E_in_NO %.4e E_out_NO %.4e; at r_s I_NO %.3e I_C %.3e window %s bands %s cross %s W %.3e; N_p %.3e N_h %.3e; rejected %s" % (
            a.benchmark, len(c["reads"]), c["seconds"], c["bound"], c["n0"], q["t_over_rs"], q["s"], q["E_in_NO"], q["E_out_NO"], q["cuts"]["%g" % rs]["I_NO"], q["cuts"]["%g" % rs]["I_C"],
            q["cuts"]["%g" % rs].get("I_NO_window"), q["cuts"]["%g" % rs].get("I_bands"), q["cuts"]["%g" % rs].get("I_cross"), q["cuts"]["%g" % rs]["W"], q["N_p"], q["N_h"], c["rejected"]), flush=True)
        return
    path = os.path.join(a.out, "vacflow10_%s.json" % a.tag)
    from scipy.special import erfc as _erfc_
    tail = 0.5 * float(_erfc_((a.smooth_window // 2) / a.smooth_width))
    print("the smoothed cut: width %g sites, window %d sites, the step's tail at the window's edge %.3e" % (a.smooth_width, a.smooth_window, tail), flush=True)
    chans = []
    t_all = time.time()
    tot = None
    for kappa in range(1, a.kmax + 1):
        c = run_channel(kappa, Rfull, a, dev, every)
        w = 16.0 * kappa
        c["weight"] = w
        chans.append(c)
        cut_keys = list(c["reads"][0]["cuts"].keys())
        fl = {ck: fluences(c["reads"], ck, rs) for ck in cut_keys}
        c["fluences"] = fl
        tot = None
        nfull = max(len(c_["reads"]) for c_ in chans)
        for c_ in chans:
            if len(c_["reads"]) == nfull and not c_["rejected"]:
                tot = {"t_over_rs": [q["t_over_rs"] for q in c_["reads"]]}
                break
        tot.update({"E_in_NO": np.zeros(nfull), "E_out_NO": np.zeros(nfull), "E_in_R2": np.zeros(nfull), "E_out_R2": np.zeros(nfull), "E_in_raw": np.zeros(nfull),
                    "E_out_raw": np.zeros(nfull), "N_p": np.zeros(nfull), "N_h": np.zeros(nfull), "fluences": {}, "channels_summed": []})
        for c_ in chans:
            add_channel(tot, c_, nfull)
        print("kappa %d (%.0f s; %d readouts; the continuation: %s)" % (kappa, time.time() - t_all, len(c["reads"]), json.dumps(c["continuation"])), flush=True)
        json.dump({"script": "engine_vacflow_v10.py", "args": vars(a), "bands": BANDS, "kappa_done": kappa, "channels": chans, "smooth_tail": tail,
                   "totals": {k_: (v_.tolist() if isinstance(v_, np.ndarray) else v_) for k_, v_ in tot.items()}}, open(path, "w"), default=float)
        continue
        if tot is None:
            tot = {"t_over_rs": [q["t_over_rs"] for q in c["reads"]], "E_in_NO": np.zeros(len(c["reads"])), "E_out_NO": np.zeros(len(c["reads"])),
                   "E_in_R2": np.zeros(len(c["reads"])), "E_out_R2": np.zeros(len(c["reads"])), "E_in_raw": np.zeros(len(c["reads"])), "E_out_raw": np.zeros(len(c["reads"])),
                   "N_p": np.zeros(len(c["reads"])), "N_h": np.zeros(len(c["reads"])), "fluences": {}}
        if c["rejected"] or len(c["reads"]) != len(tot["t_over_rs"]):
            tot.setdefault("left_out", []).append({"kappa": kappa, "why": c["rejected"] or "readout count %d" % len(c["reads"])})
            continue
        for k_ in ("E_in_NO", "E_out_NO", "E_in_R2", "E_out_R2", "E_in_raw", "E_out_raw"):
            tot[k_] += w * np.array([q.get(k_, np.nan) for q in c["reads"]])
        for k_ in ("N_p", "N_h"):
            tot[k_] += w * np.array([q[k_] for q in c["reads"]])
            tot[k_ + "_unweighted"] = tot.get(k_ + "_unweighted", np.zeros(len(c["reads"]))) + np.array([q[k_] for q in c["reads"]])
        for ck, f in fl.items():
            T_ = tot["fluences"].setdefault(ck, {})
            for phase, res in f.items():
                P_ = T_.setdefault(phase, {"total": 0.0, "bands": [0.0] * len(BANDS), "cross": 0.0, "window_total": 0.0, "work": 0.0,
                                           "smooth_bands": [0.0] * len(BANDS), "smooth_cross": 0.0})
                P_["total"] += w * res["total"]; P_["work"] += w * res["work"]
                if "bands" in res:
                    P_["bands"] = [x + w * y for x, y in zip(P_["bands"], res["bands"])]
                    P_["cross"] += w * res["cross"]; P_["window_total"] += w * res["window_total"]
                for key in ("total_C", "total_R2", "work_C", "work_R0", "dE_K_C", "dE_K_R0", "balance_C", "reference_own", "smooth_total_NO", "smooth_total_C", "smooth_total_R2", "smooth_cross"):
                    if key in res:
                        P_[key] = P_.get(key, 0.0) + w * res[key]
                if "smooth_bands" in res:
                    P_["smooth_bands"] = [x + w * y for x, y in zip(P_["smooth_bands"], res["smooth_bands"])]
        fr = fl.get("%g" % rs, {}).get("ramp", {})
        print("kappa %d (%.0f s; %d readouts, halvings %d%s): at the ramp's end E_in_NO %.4e E_out_NO %.4e E_in_R2 %s; outflow fluence through r_s over the ramp %.4e (bands %s; cross %s; work inside %.4e); N_p %.3e N_h %.3e" % (
            kappa, time.time() - t_all, len(c["reads"]), c["halvings"], (", REJECTED: " + c["rejected"]) if c["rejected"] else "",
            [q for q in c["reads"] if q["phase"] == "ramp"][-1]["E_in_NO"], [q for q in c["reads"] if q["phase"] == "ramp"][-1]["E_out_NO"],
            ("%.4e" % [q for q in c["reads"] if q["phase"] == "ramp"][-1].get("E_in_R2", float("nan"))), fr.get("total", float("nan")),
            ("[" + ", ".join("%.2e" % x for x in fr["bands"]) + "]") if "bands" in fr else "-", ("%.2e" % fr["cross"]) if "cross" in fr else "-",
            fr.get("work", float("nan")), c["reads"][-1]["N_p"], c["reads"][-1]["N_h"]), flush=True)
        json.dump({"script": "engine_vacflow_v9.py", "args": vars(a), "bands": BANDS, "kappa_done": kappa, "channels": chans,
                   "totals": {k_: (v_.tolist() if isinstance(v_, np.ndarray) else v_) for k_, v_ in tot.items()}}, open(path, "w"), default=float)
    # the reading per cut (engine 26.6): the cross-band share of the smoothed cut, the middle bands read only below 0.1
    readings = {}
    for ck, f in tot["fluences"].items():
        frc = f.get("ramp")
        if not frc or "smooth_total_NO" not in frc or not frc["smooth_total_NO"]:
            readings[ck] = {"smoothed_cut": "absent"}
            continue
        bs = frc["smooth_bands"]; cross_share = abs(frc["smooth_cross"]) / abs(frc["smooth_total_NO"])
        mid = bs[1] + bs[4]
        rd = {"total_outflow": frc["smooth_total_NO"], "bands": bs, "cross": frc["smooth_cross"], "cross_share": cross_share}
        if cross_share < 0.1:
            rd["middle_share"] = mid / frc["smooth_total_NO"]
            rd["rides_cos2p_negative"] = bool(rd["middle_share"] > 0.9)
        else:
            rd["band_split"] = "not read (the cross-band share %.3f is not below 0.1)" % cross_share
        readings[ck] = rd
        print("cut %s over the ramp: the smoothed outflow %.4e, cross-band share %.4f; %s" % (ck, frc["smooth_total_NO"], cross_share,
              ("middle bands' share %.4f: %s" % (rd["middle_share"], "rides" if rd["rides_cos2p_negative"] else "does not ride")) if "middle_share" in rd else rd["band_split"]), flush=True)
    fr = tot["fluences"].get("%g" % rs, {}).get("ramp")
    verdict = None
    if fr and "bands" in fr:
        b = fr["bands"]
        rev = b[1] + b[4]; nonrev = b[0] + b[2] + b[3] + b[5]
        verdict = {"outflow_total_window": fr["window_total"], "band_diagonal_sum": sum(b), "cross": fr["cross"], "cos2p_negative_bands": rev, "cos2p_positive_bands": nonrev,
                   "lead_grouping": {"|p|<pi/4": b[0] + b[3], "|p|>pi/4": b[1] + b[2] + b[4] + b[5]},
                   "by_sign": {"p>0": b[0] + b[1] + b[2], "p<0": b[3] + b[4] + b[5]},
                   "sharp_middle_share": (rev / fr["window_total"]) if fr["window_total"] else None}
        if "smooth_total_NO" in fr:
            bs = fr["smooth_bands"]; revS = bs[1] + bs[4]
            share = (revS / fr["smooth_total_NO"]) if fr["smooth_total_NO"] else None
            verdict["smoothed"] = {"total_outflow": fr["smooth_total_NO"], "bands": bs, "cross": fr["smooth_cross"], "middle_bands": revS, "middle_share": share,
                                   "lead_grouping": {"|p|<pi/4": bs[0] + bs[3], "|p|>pi/4": bs[1] + bs[2] + bs[4] + bs[5]}}
        else:
            share = None
        verdict["rides_cos2p_negative"] = bool(share is not None and share > 0.9)
        verdict["rule"] = "engine spec 19.4: read on the smoothed cut; 'rides' only when the two middle bands carry more than 0.9 of its total outflow over the ramp"
        print("the outflow through r_s over the ramp (summed over the channels): window total %.4e, band-diagonal %.4e, cross %.4e; cos 2p < 0 bands %.4e, cos 2p > 0 bands %.4e; |p| < pi/4 %.4e, |p| > pi/4 %.4e; p > 0 %.4e, p < 0 %.4e: the smoothed cut: total %s, middle-band share %s: the outgoing energy %s the cos 2p < 0 branches" % (
            fr["window_total"], sum(b), fr["cross"], rev, nonrev, verdict["lead_grouping"]["|p|<pi/4"], verdict["lead_grouping"]["|p|>pi/4"], verdict["by_sign"]["p>0"],
            verdict["by_sign"]["p<0"], ("%.4e" % verdict["smoothed"]["total_outflow"]) if "smoothed" in verdict else "-",
            ("%.4f" % share) if share is not None else "-", "rides" if verdict["rides_cos2p_negative"] else "does not ride (share printed)"), flush=True)
    json.dump({"script": "engine_vacflow_v10.py", "args": vars(a), "bands": BANDS, "kappa_done": a.kmax, "channels": chans, "verdict": verdict, "readings": readings,
               "smooth_tail": tail, "seconds": time.time() - t_all,
               "totals": {k_: (v_.tolist() if isinstance(v_, np.ndarray) else v_) for k_, v_ in tot.items()}}, open(path, "w"), default=float)


if __name__ == "__main__":
    main()
