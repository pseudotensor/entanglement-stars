#!/usr/bin/env python
"""s32b_cutS_v1.py (busi, inbox #420, NOTCOMP item 3.2b: the vacuum coefficient at the trapping cut on a genuine 3D cut; ReviewA_notcomp_v1
sec. 3.2(b), ReviewB_notcomp_v1 sec. 6.3's conditions: the same cubic parent, cut and history; finite-parent, polynomial, stochastic and path
errors certified separately; measured local gaps).

The record's fields (the reference collapse c19 r11, r_s/a 64, sigma/a 4, at T/r_s 15 by default) are lifted to the cubic lattice with
gap10_core_v2's allocation (the K_ell back-averaged radial profiles, cell-averaged; busi GAP10 code, read-only), the flow clipped smoothly
inside the pocket to |v| < s_c N (B.6's comparison configuration; s_c below the frozen-gap threshold of the mass, computed here), and
    S_A(P) = Tr s(C_A),  C_A = Pi_A P Pi_A,  P = (1 - sgn h)/2  (the clipped configuration's ground projector, gapped),
on the digital ball A = {|x| < R_t} at the record's outer trapping root, on an open cube of half-side c = ceil(R_t + shell). The trace is a
stochastic Lanczos quadrature (SLQ): Rademacher vectors supported on the ball's sites within --depth of the cut (deeper sites carry no entropy
up to exp(-depth/xi); checked by a deeper band), Lanczos on C_A from each vector, Gauss quadrature of s. sgn h is the Chebyshev series of
erf(kappa x) on the spectrum scaled by the Schur bound, kappa set by the measured gap. Every vector is paired: the same vector on the flat sea
P_0 (N = 1, v = 0, same m), so the departure S_A(P) - S_A(P_0) has the paired variance. Compare with eta(0.2) A_lat (A_lat = 4 pi R^2).

Modes:
  export   (box) the record's dense fields at T (read_history_v1.dense_record) to a profile npz (the kit's only data input)
  fixture  dense validation on a small box: the profile rescaled to a ball of radius --fix-R; exact S_A by eigh against the polynomial P
           (all unit vectors) and against SLQ (many vectors); prints each error separately
  gap      the clipped operator's smallest |e| by Lanczos on h^2 (the gap that sets the sgn degree), and the timing of one h application
  run      the production estimator (checkpointed per vector batch; resumable)
"""
import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np
import torch

T0W = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))


def log(msg):
    print("[%8.1f s] %s" % (time.time() - T0W, msg), flush=True)


def md5(p, chunk=1 << 24):
    h = hashlib.md5()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                return h.hexdigest()
            h.update(b)


def import_gap10(path):
    sys.path.insert(0, path)
    import gap10_core_v2 as G
    return G


# ------------------------------------------------------------------ the frozen gap and the clip threshold
def frozen_gap(zeta, w, ndir=400, nk=161):
    """min over k and over flow directions n (a Fibonacci sphere) of eps(k) - w |q(k).n|, eps^2 = zeta^2 + sum sin^2 k_i, q_i = sin(2 k_i)/2:
    the frozen symbol's gap per unit N at |v| = w N (radial flow covers every direction on the cut)."""
    k = np.linspace(-math.pi / 2, math.pi / 2, nk)
    KX, KY, KZ = np.meshgrid(k, k, k, indexing="ij")
    eps = np.sqrt(zeta ** 2 + np.sin(KX) ** 2 + np.sin(KY) ** 2 + np.sin(KZ) ** 2)
    q = [0.5 * np.sin(2 * KX), 0.5 * np.sin(2 * KY), 0.5 * np.sin(2 * KZ)]
    i = np.arange(ndir) + 0.5
    ph = np.arccos(1 - 2 * i / ndir); th = math.pi * (1 + 5 ** 0.5) * i
    dirs = np.stack([np.cos(th) * np.sin(ph), np.sin(th) * np.sin(ph), np.cos(ph)], 1)
    dirs = dirs[dirs[:, 2] >= 0]
    g = np.inf
    for n in dirs:
        qn = np.abs(n[0] * q[0] + n[1] * q[1] + n[2] * q[2])
        g = min(g, float(np.min(eps - w * qn)))
    return g


def clip_threshold(zeta):
    lo, hi = 0.5, 2.0
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        if frozen_gap(zeta, mid, ndir=200, nk=81) > 0:
            lo = mid
        else:
            hi = mid
    return lo


# ------------------------------------------------------------------ profile
def export_profile(a):
    sys.path.insert(0, a.reader_dir)
    import read_history_v1 as RH
    H = RH.load_history(a.history, kind="applied", sources=True)
    k = int(np.argmin(np.abs(H["T"] - a.T)))
    D = RH.dense_record(H, k)
    r = np.arange(0.0, a.export_rmax + 1e-9, 1.0 / 16.0)
    N = np.asarray(D["N"](r), float); v = np.asarray(D["v"](r), float)
    zeros = D["zeros"]
    out = dict(r=r, N=N, v=v, T=float(H["T"][k]), rs=float(H["rs"]), R_t_record=float(H["R_t"][k]), zeros=np.array(zeros), m=float(H["args"].get("m", 0.2)),
               ell=float(H["ell"]), label=a.history, npz_md5=H["npz_md5"], json_md5=H["json_md5"])
    np.savez(a.profile, **out)
    log("exported %s T/r_s %.4f (record %d): R_t %.4f (dense zeros %s), min(N - |v|) %s, m %g, ell %g, r to %g -> %s (md5 %s)" % (
        a.history, H["T"][k], k, H["R_t"][k], zeros, D["min"], out["m"], out["ell"], r[-1], a.profile, md5(a.profile)))


def load_profile(path, G, scale=1.0):
    z = np.load(path)
    r = np.asarray(z["r"], float) * scale
    tab = (r[1:], np.asarray(z["N"], float)[1:], np.abs(np.asarray(z["v"], float))[1:])
    prof = G.RadialProfile(float(z["rs"]) * scale, 0.0, "file", "record %s T %.3f" % (str(z["label"]), float(z["T"])), False, tab)
    zeros = [float(x) * scale for x in np.atleast_1d(z["zeros"])]
    return prof, {"T": float(z["T"]), "rs": float(z["rs"]) * scale, "R_t": float(z["R_t_record"]) * scale, "zeros": zeros, "m": float(z["m"]), "ell": float(z["ell"]) * scale,
                  "label": str(z["label"]), "npz_md5": str(z["npz_md5"])}


# ------------------------------------------------------------------ the lattice
def build(G, prof, meta, c, s0, sc, flat, ell, device, core_floor=None, core_r=None):
    L = 2 * c + 1
    if flat:
        v = [np.zeros((L, L, L)) for _ in range(3)]; N = np.ones((L, L, L))
        info = {"flat": True}
    else:
        v, N, _ = G.build_site_fields(L, c, ell, prof, "averaged")
        if N is None:
            N = np.ones((L, L, L))
        idx = (np.arange(L) - c).astype(float)
        X, Y, Z = np.meshgrid(idx, idx, idx, indexing="ij"); rr = np.sqrt(X * X + Y * Y + Z * Z)
        vm = np.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
        w = vm / N
        info = {"flat": False, "w_max_raw": float(w.max()), "N_min_raw": float(N.min())}
        if core_floor is not None:
            f = 0.5 * (1 + np.tanh((core_r - rr) / 2.0))
            N = np.maximum(N, f * core_floor + (1 - f) * N)
            info.update(core_floor=core_floor, core_r=core_r)
        w = vm / N
        big = w > s0
        fw = np.where(big, s0 + (sc - s0) * np.tanh((w - s0) / (sc - s0)), w)
        scl = np.where(vm > 0, fw / np.maximum(w, 1e-300), 1.0)
        v = [vi * scl for vi in v]
        wcmp = np.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2) / N
        wg = np.linspace(0.0, max(sc, float(wcmp.max())) + 1e-9, 24)
        gg = np.array([frozen_gap(meta["m"], x, 120, 61) for x in wg])
        gsite = N * np.interp(wcmp, wg, gg)
        im = np.unravel_index(int(np.argmin(gsite)), gsite.shape)
        info.update(s0=s0, sc=sc, n_clipped=int(big.sum()), w_max_cmp=float(wcmp.max()), N_min=float(N.min()),
                    frozen_gap_min=float(gsite.min()), frozen_gap_min_r=float(rr[im]), frozen_gap_at_cut=float(np.min(np.where(np.abs(rr - meta.get("R_cut", 0.0)) < 1.0, gsite, np.inf))))
    wp, wm = G.bond_weights_np(v)
    lp, lm = G.link_weights_np(N)
    H = G.LatticeH(meta["m"], wp, wm, device, None, N, lp, lm)
    b = G.spectral_bound(meta["m"], wp, wm, N)
    info["bound"] = float(b); info["L"] = L
    return H, info


def ball_masks(L, c, R, depth, device):
    idx = (torch.arange(L, dtype=torch.float64, device=device) - c)
    rr = torch.sqrt(idx[:, None, None] ** 2 + idx[None, :, None] ** 2 + idx[None, None, :] ** 2)
    A = rr < R
    shell = A & (rr >= R - depth)
    deep = A & (rr >= R - 2 * depth) & (rr < R - depth)
    return A, shell, deep, rr


# ------------------------------------------------------------------ sgn by Chebyshev
def sgn_coeffs(delta_s, eps=1e-10, tol=1e-13, nmax=20000):
    from scipy.special import erfcinv, erf
    kap = float(erfcinv(eps)) / delta_s
    n = 64
    while True:
        x = np.cos(math.pi * (np.arange(n) + 0.5) / n)
        f = erf(kap * x)
        cft = np.array([2.0 / n * np.sum(f * np.cos(j * math.pi * (np.arange(n) + 0.5) / n)) for j in range(n)]) if n <= 512 else None
        if cft is None:
            from scipy.fft import dct
            cft = dct(f, type=2) / n
        cft[0] *= 0.5
        tail = np.abs(cft[-max(8, n // 16):]).max()
        if tail < tol or n >= nmax:
            break
        n *= 2
    nz = np.nonzero(np.abs(cft) > tol)[0]
    deg = int(nz.max()) + 1
    return cft[:deg], kap, deg


class Proj:
    """P psi = (psi - sgn(h) psi)/2 with sgn(h) ~ sum_j c_j T_j(h/b) (odd j only)."""

    def __init__(self, H, b, coeffs):
        self.H, self.b, self.c = H, float(b), coeffs
        self.napply = 0

    def apply(self, psi):
        H, b = self.H, self.b
        T0 = psi
        T1 = H.apply(psi) / b
        acc = self.c[1] * T1 if len(self.c) > 1 else torch.zeros_like(psi)
        if abs(self.c[0]) > 0:
            acc = acc + self.c[0] * T0
        for j in range(2, len(self.c)):
            T2 = H.apply(T1).mul_(2.0 / b).sub_(T0)
            if abs(self.c[j]) > 1e-15:
                acc.add_(T2, alpha=float(self.c[j]))
            T0, T1 = T1, T2
        self.napply += 1
        return 0.5 * (psi - acc)


def ent(x):
    x = np.clip(x, 0.0, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where((x > 0) & (x < 1), -x * np.log(x) - (1 - x) * np.log(1 - x), 0.0)


def slq(PA, z, m):
    """sum over the batch of z_b^dag s(C_A) z_b by m-step Lanczos (no reorthogonalization) per batch element; PA(x) applies C_A."""
    B = z.shape[0]
    nz = torch.linalg.vector_norm(z.reshape(B, -1), dim=1)
    q = z / nz.view(B, 1, 1, 1, 1)
    qprev = torch.zeros_like(q)
    alphas, betas = [], []
    beta = torch.zeros(B, dtype=torch.float64, device=z.device)
    for j in range(m):
        w = PA(q)
        al = (q.conj() * w).sum((1, 2, 3, 4)).real
        w = w - al.view(B, 1, 1, 1, 1) * q - beta.view(B, 1, 1, 1, 1) * qprev
        beta = torch.linalg.vector_norm(w.reshape(B, -1), dim=1)
        alphas.append(al.cpu().numpy()); betas.append(beta.cpu().numpy())
        qprev, q = q, w / torch.clamp(beta, min=1e-300).view(B, 1, 1, 1, 1)
    alphas = np.array(alphas); betas = np.array(betas)
    out = []
    for bi in range(B):
        T = np.diag(alphas[:, bi]) + np.diag(betas[:-1, bi], 1) + np.diag(betas[:-1, bi], -1)
        th, U = np.linalg.eigh(T)
        out.append(float(nz[bi].item() ** 2 * np.sum(U[0, :] ** 2 * ent(th))))
    return out


def rademacher(B, L, mask, gen, device):
    z = torch.randint(0, 4, (B, 4, L, L, L), generator=gen, device="cpu").to(device)
    ph = torch.tensor([1, 1j, -1, -1j], dtype=torch.complex128, device=device)
    return ph[z] * mask.view(1, 1, L, L, L)


# ------------------------------------------------------------------ the gap by Lanczos on h^2
def gap_lanczos(H, L, m, device, seed=1):
    g = torch.Generator().manual_seed(seed)
    z = torch.randn((1, 4, L, L, L), generator=g, dtype=torch.float64).to(device).to(torch.complex128)
    q = z / torch.linalg.vector_norm(z)
    qprev = torch.zeros_like(q); beta = 0.0; al_, be_ = [], []
    for j in range(m):
        w = H.apply(H.apply(q))
        al = float((q.conj() * w).sum().real)
        w = w - al * q - beta * qprev
        beta = float(torch.linalg.vector_norm(w))
        al_.append(al); be_.append(beta)
        qprev, q = q, w / beta
    T = np.diag(al_) + np.diag(be_[:-1], 1) + np.diag(be_[:-1], -1)
    th = np.linalg.eigvalsh(T)
    return float(math.sqrt(max(th[0], 0.0))), float(math.sqrt(th[-1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["export", "fixture", "gap", "run"])
    ap.add_argument("--gap10", default=os.path.join(HERE, "gap10"))
    ap.add_argument("--reader-dir", default="<internal>/code")
    ap.add_argument("--history", default="P64_ref_c19r11")
    ap.add_argument("--T", type=float, default=15.0)
    ap.add_argument("--export-rmax", type=float, default=400.0)
    ap.add_argument("--profile", default=os.path.join(HERE, "profile_c19r11_T15.npz"))
    ap.add_argument("--R", type=float, default=None, help="cut radius (a); default the record's R_t")
    ap.add_argument("--shell", type=float, default=40.0)
    ap.add_argument("--depth", type=float, default=16.0)
    ap.add_argument("--s0", type=float, default=1.0)
    ap.add_argument("--sc-frac", type=float, default=0.95, help="s_c = this x the frozen-gap threshold w_c(zeta)")
    ap.add_argument("--core-floor", type=float, default=None)
    ap.add_argument("--core-depth", type=float, default=None, help="core radius = R - this")
    ap.add_argument("--gap", type=float, default=None, help="the gap delta (energy units) used for the sgn degree; default: measured (mode gap) or frozen")
    ap.add_argument("--eps-sgn", type=float, default=1e-10)
    ap.add_argument("--delta-choice", default="half", choices=["lanczos", "half", "frozen"], help="run: which gap sets the sgn degree (the idempotency check at the start tests it)")
    ap.add_argument("--lanczos", type=int, default=40)
    ap.add_argument("--nvec", type=int, default=64)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--fix-R", type=float, default=5.0)
    ap.add_argument("--fix-c", type=int, default=7)
    ap.add_argument("--fix-nvec", type=int, default=256)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default="out_s32b")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    if a.mode == "export":
        export_profile(a); return
    G = import_gap10(a.gap10)
    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    zeta = None
    res = {"mode": a.mode, "args": vars(a), "gap10_md5": md5(os.path.join(a.gap10, "gap10_core_v2.py")), "self_md5": md5(os.path.abspath(__file__)),
           "profile_md5": md5(a.profile)}

    if a.mode == "fixture":
        prof0, meta0 = load_profile(a.profile, G)
        scale = a.fix_R / meta0["R_t"]
        prof, meta = load_profile(a.profile, G, scale=scale)
        zeta = meta["m"]
        wc = clip_threshold(zeta); sc = a.sc_frac * wc
        c = a.fix_c; L = 2 * c + 1
        res.update(scale=scale, R=a.fix_R, c=c, w_c=wc, s_c=sc)
        log("fixture: the profile scaled by %.4f (R_t %.3f -> %.3f), box c %d (L %d, dim %d); zeta %g; w_c %.4f, s_c %.4f" % (scale, meta0["R_t"], a.fix_R, c, L, 4 * L ** 3, zeta, wc, sc))
        for flat in (False, True):
            H, info = build(G, prof, meta, c, a.s0, sc, flat, meta["ell"], dev)
            n = 4 * L ** 3
            # dense h from unit vectors
            Hd = torch.zeros((n, n), dtype=torch.complex128, device=dev)
            for j0 in range(0, n, 512):
                jj = torch.arange(j0, min(n, j0 + 512), device=dev)
                E = torch.zeros((len(jj), n), dtype=torch.complex128, device=dev); E[torch.arange(len(jj)), jj] = 1.0
                Hd[:, j0:j0 + len(jj)] = H.apply(E.view(len(jj), 4, L, L, L)).reshape(len(jj), n).T
            herm = float((Hd - Hd.mH).abs().max())
            e, U = torch.linalg.eigh(0.5 * (Hd + Hd.mH))
            gap = float(e.abs().min()); Pn = U[:, e < 0] @ U[:, e < 0].mH
            log("  %s: dense h built and diagonalized: |h - h^dag| %.1e, gap %.5f, e in [%.4f, %.4f], Schur bound %.4f" % ("flat" if flat else "record", herm, gap, float(e.min()), float(e.max()), info["bound"]))
            Am, sh, dp, rr = ball_masks(L, c, a.fix_R, a.depth * scale if a.depth * scale > 1 else 2.0, dev)
            aidx = torch.nonzero(Am.reshape(-1).repeat(4) if False else torch.stack([Am] * 4, 0).reshape(-1)).squeeze(1)
            CA = Pn[aidx][:, aidx]
            S_exact = float(ent(torch.linalg.eigvalsh(CA).cpu().numpy()).sum())
            log("  S_A exact %.8f (ball sites %d)" % (S_exact, int(Am.sum())))
            # the polynomial projector on the ball's unit vectors
            b = info["bound"]; coeffs, kap, deg = sgn_coeffs(gap / b, a.eps_sgn)
            Pp = Proj(H, b, coeffs)
            cols = []
            for j0 in range(0, len(aidx), 256):
                jj = aidx[j0:j0 + 256]
                E = torch.zeros((len(jj), n), dtype=torch.complex128, device=dev); E[torch.arange(len(jj)), jj] = 1.0
                cols.append(Pp.apply(E.view(len(jj), 4, L, L, L)).reshape(len(jj), n)[:, aidx].T)
            CAp = torch.cat(cols, 1)
            S_poly = float(ent(torch.linalg.eigvalsh(0.5 * (CAp + CAp.mH)).cpu().numpy()).sum())
            perr = float((CAp - CA).abs().max())
            log("  the polynomial P (sgn degree %d): S_A %.8f, max |C_A poly - exact| %.2e" % (deg, S_poly, perr))
            # SLQ with Rademacher vectors on the whole ball
            gen = torch.Generator().manual_seed(a.seed)
            Am4 = Am
            PA = lambda x: Am4.view(1, 1, L, L, L) * Pp.apply(Am4.view(1, 1, L, L, L) * x)
            vals = []
            for i in range(0, a.fix_nvec, a.batch):
                z = rademacher(min(a.batch, a.fix_nvec - i), L, Am4, gen, dev)
                vals += slq(PA, z, a.lanczos)
            vals = np.array(vals)
            S_slq = float(vals.mean()); S_slq_err = float(vals.std(ddof=1) / math.sqrt(len(vals)))
            # SLQ on the shell only (the depth truncation)
            gen = torch.Generator().manual_seed(a.seed + 1)
            vals2 = []
            for i in range(0, a.fix_nvec, a.batch):
                z = rademacher(min(a.batch, a.fix_nvec - i), L, sh, gen, dev)
                vals2 += slq(PA, z, a.lanczos)
            vals2 = np.array(vals2)
            # exact shell-restricted trace of s(C_A): diag of U s U^dag on the shell sites
            ev, V = torch.linalg.eigh(CA)
            dg = ((V.abs() ** 2) @ torch.as_tensor(ent(ev.cpu().numpy()), device=dev, dtype=torch.float64))
            shm = torch.stack([sh] * 4, 0).reshape(-1)[aidx]
            S_shell_exact = float(dg[shm].sum())
            row = dict(flat=flat, info=info, hermiticity=herm, gap=gap, sgn_degree=deg, kappa=kap, S_exact=S_exact, S_poly=S_poly, poly_maxdev=perr,
                       S_slq=S_slq, S_slq_err=S_slq_err, S_shell_exact=S_shell_exact, S_shell_slq=float(vals2.mean()),
                       S_shell_slq_err=float(vals2.std(ddof=1) / math.sqrt(len(vals2))), A_sites=int(Am.sum()), eta_Alat_0p887=0.887 * 4 * math.pi * a.fix_R ** 2)
            res["flat" if flat else "record"] = row
            log("fixture %s: |h - h^dag| %.1e, gap %.4f (bound %.3f, sgn degree %d), S exact %.6f, S poly %.6f (|C_A poly - exact| %.1e), S SLQ %.4f +- %.4f; "
                "shell (depth %.1f) exact %.6f, SLQ %.4f +- %.4f; info %s" % ("flat" if flat else "record", herm, gap, b, deg, S_exact, S_poly, perr, S_slq, S_slq_err,
                a.depth * scale, S_shell_exact, row["S_shell_slq"], row["S_shell_slq_err"], info))
            del Hd, U, Pn, CA, CAp
        json.dump(res, open(os.path.join(a.out, "fixture.json"), "w"), indent=1)
        return

    prof, meta = load_profile(a.profile, G)
    zeta = meta["m"]
    R = a.R or meta["R_t"]
    c = int(math.ceil(R + a.shell)); L = 2 * c + 1
    wc = clip_threshold(zeta); sc = a.sc_frac * wc
    core = None if a.core_floor is None else (a.core_floor, R - (a.core_depth or a.shell))
    res.update(R=R, c=c, L=L, w_c=wc, s_c=sc, meta=meta)
    log("%s: %s T/r_s %.3f, R %.3f (R_t %.3f), box c %d (L %d, dim %.3e), zeta %g, w_c %.4f, s_c %.4f, frozen gap per N at s_c %.4f, at w=1 %.4f" % (
        a.mode, meta["label"], meta["T"], R, meta["R_t"], c, L, 4.0 * L ** 3, zeta, wc, sc, frozen_gap(zeta, sc, 200, 81), frozen_gap(zeta, 1.0, 200, 81)))
    meta["R_cut"] = R
    Hr, info_r = build(G, prof, meta, c, a.s0, sc, False, meta["ell"], dev, *(core or (None, None)))
    log("record operator: %s" % info_r)
    t1 = time.time(); x = torch.randn((a.batch, 4, L, L, L), dtype=torch.float64, device=dev).to(torch.complex128)
    torch.cuda.synchronize() if dev.type == "cuda" else None
    t1 = time.time()
    for _ in range(5):
        y = Hr.apply(x)
    torch.cuda.synchronize() if dev.type == "cuda" else None
    t_apply = (time.time() - t1) / 5
    del x, y
    log("one h application on a batch of %d: %.3f s (%.3f s per vector)" % (a.batch, t_apply, t_apply / a.batch))
    res["t_apply_batch"] = t_apply
    if a.mode == "gap":
        g_lo, g_hi = gap_lanczos(Hr, L, a.lanczos * 5, dev)
        log("Lanczos on h^2 (%d steps): smallest |e| <= %.5f (an upper bound on the gap), largest %.4f (Schur bound %.4f)" % (a.lanczos * 5, g_lo, g_hi, info_r["bound"]))
        res.update(gap_lanczos=g_lo, emax=g_hi, info_record=info_r)
        cands = [("Lanczos", g_lo), ("half Lanczos", 0.5 * g_lo), ("frozen-gap site minimum", info_r["frozen_gap_min"])]
        for lab, dl in cands:
            co, kap, deg = sgn_coeffs(dl / info_r["bound"], a.eps_sgn)
            frac = dl / g_lo
            est = deg * a.lanczos * a.nvec * t_apply / a.batch / 3600.0
            log("  with delta = %.5f (%s): sgn degree %d; the record arm's cost (%d shell vectors x %d Lanczos steps; the flat arm adds about 5 percent) about %.1f device-h at this card's measured h pace" % (dl, lab, deg, a.nvec, a.lanczos, est))
            res["cost_h_delta_frac_%g" % frac] = est; res["sgn_degree_delta_frac_%g" % frac] = deg
        json.dump(res, open(os.path.join(a.out, "gap.json"), "w"), indent=1)
        return
    # ---------------------------------------------------------------- run
    delta = a.gap
    if delta is None:
        gj = os.path.join(a.out, "gap.json")
        gjj = json.load(open(gj))
        delta = {"lanczos": gjj["gap_lanczos"], "half": 0.5 * gjj["gap_lanczos"], "frozen": gjj["info_record"]["frozen_gap_min"]}[a.delta_choice]
        log("delta (%s) from %s: %.5f" % (a.delta_choice, gj, delta))
    H0, info_0 = build(G, prof, meta, c, a.s0, sc, True, meta["ell"], dev)
    cr, kr, dr = sgn_coeffs(delta / info_r["bound"], a.eps_sgn)
    c0, k0, d0 = sgn_coeffs(zeta / info_0["bound"], a.eps_sgn)
    log("sgn degrees: record %d (delta %.5f), flat %d (gap m = %g)" % (dr, delta, d0, zeta))
    Pr = Proj(Hr, info_r["bound"], cr); P0 = Proj(H0, info_0["bound"], c0)
    # the sgn accuracy at this delta: the idempotency defect ||P(Pz) - Pz|| / ||z|| on one random vector (a mode with |e| < delta shows here)
    if not os.path.exists(os.path.join(a.out, "run_ck.json")):
        gq = torch.Generator().manual_seed(7)
        zq = torch.randn((1, 4, L, L, L), generator=gq, dtype=torch.float64).to(dev).to(torch.complex128)
        t1 = time.time(); p1 = Pr.apply(zq); p2 = Pr.apply(p1)
        idem = float(torch.linalg.vector_norm(p2 - p1) / torch.linalg.vector_norm(zq)); herm = float(((zq.conj() * p1).sum().imag).abs() / torch.linalg.vector_norm(zq) ** 2)
        res.update(idempotency_defect=idem, hermiticity_defect=herm)
        log("idempotency defect ||P^2 z - P z||/||z|| %.2e, Im <z|P|z>/|z|^2 %.1e (%.0f s for 2 applications at degree %d)" % (idem, herm, time.time() - t1, dr))
        del zq, p1, p2
    Am, sh, dp, rr = ball_masks(L, c, R, a.depth, dev)
    res.update(A_sites=int(Am.sum()), shell_sites=int(sh.sum()), A_lat=4 * math.pi * R * R, eta_Alat=0.887 * 4 * math.pi * R * R, sgn_deg_record=dr, sgn_deg_flat=d0, delta=delta)
    mk = Am.view(1, 1, L, L, L)
    PAr = lambda x: mk * Pr.apply(mk * x)
    PA0 = lambda x: mk * P0.apply(mk * x)
    ck = os.path.join(a.out, "run_ck.json")
    st = json.load(open(ck)) if os.path.exists(ck) else {"shell": [], "deep": []}
    gen = torch.Generator().manual_seed(a.seed)
    for band, mask, nv in (("shell", sh, a.nvec), ("deep", dp, max(8, a.nvec // 8))):
        done = len(st[band]); i = 0
        while i < nv:
            B = min(a.batch, nv - i)
            z = rademacher(B, L, mask, gen, dev)
            if i < done:
                i += B; continue
            t1 = time.time()
            qr = slq(PAr, z, a.lanczos); q0 = slq(PA0, z, a.lanczos)
            st[band] += [[x, y] for x, y in zip(qr, q0)]
            i += B
            arr = np.array(st[band])
            d = arr[:, 0] - arr[:, 1]
            log("%s vectors %d/%d (%.0f s): S_rec %.2f +- %.2f, S_flat %.2f +- %.2f, departure %.3f +- %.3f" % (band, len(arr), nv, time.time() - t1, arr[:, 0].mean(),
                arr[:, 0].std(ddof=1) / math.sqrt(len(arr)) if len(arr) > 1 else 0, arr[:, 1].mean(), arr[:, 1].std(ddof=1) / math.sqrt(len(arr)) if len(arr) > 1 else 0,
                d.mean(), d.std(ddof=1) / math.sqrt(len(arr)) if len(arr) > 1 else 0))
            json.dump(st, open(ck, "w"))
    for band in ("shell", "deep"):
        arr = np.array(st[band]); d = arr[:, 0] - arr[:, 1]
        res[band] = dict(n=len(arr), S_rec=float(arr[:, 0].mean()), S_rec_err=float(arr[:, 0].std(ddof=1) / math.sqrt(len(arr))), S_flat=float(arr[:, 1].mean()),
                         S_flat_err=float(arr[:, 1].std(ddof=1) / math.sqrt(len(arr))), departure=float(d.mean()), departure_err=float(d.std(ddof=1) / math.sqrt(len(arr))))
    res["info_record"] = info_r
    json.dump(res, open(os.path.join(a.out, "run.json"), "w"), indent=1)
    log("done: %s" % json.dumps({k: res[k] for k in ("shell", "deep", "eta_Alat")}))


if __name__ == "__main__":
    main()
