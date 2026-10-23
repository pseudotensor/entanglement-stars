#!/usr/bin/env python
"""pt_companion_bprime_v1.py (busi, inbox #428, the companion to Test A: reviewer A's B', ReviewA_pc6_v1.md sec. 4 and ReviewA_pc5_v1.md
sec. 4 B): one nominal k = 9 packet in the trapped region of a rooted record on the four-component cubic operator, box of side 2.5
nominal r_s (open faces), r_s/a = 32 and 64; its weight in the channels |kappa| = 1, 2 after one bounce and after a few: f_mix.

As specified the packet is a pocket level of k = 9 at |omega| ~ T_H. pt_pocket_census_v1 finds none on either record (on the r_s/a 64
record the chain has inverted levels only at k = 1, two pocket and two rising at -3.95 T_H; on the r_s/a 32 record none at k = 1..12),
so the packet here is the nearest buildable object, labelled nominal: the chain profile f(r) = exp(-(r - R_c)^2/(4 sigma^2))
exp(i k_0 (r - R_c)) times the chain's upper-band spinor at (k_0, kappa/R_c), R_c the middle of the applied fields' trapped annulus,
sigma a quarter of its width (at least 2 a), k_0 the radial momentum that minimizes the k = 9 upper band's Killing energy
v sin(2k)/2 + N (sin^2 k + (9/R_c)^2)^(1/2) at R_c (the closest the channel comes to a pocket; its energy printed), lifted to the
cube by o5_lift_v1 (spinor harmonics, m_j = 1/2; the convention chosen by the reduction check: the lifted packet's <h> on the cube
against the chain packet's <h> on the chain operator, the four conventions printed). The cubic operator: gap10_core_v2's endpoint-mean
h = (1/2){N, h_0} + (1/2) sum_i {v^i, Q_i} at m = 0 (o5_lift_v1.cubic_apply, ported to torch and checked against it), the record's applied
fields point-sampled on the cube (o5_lift_v1.fields_on_cube). Propagation: Chebyshev (tol 1e-15). Readout at t = 0 (the projection
floor), 1/2, 1, 2, 4 and 8 bounces (one bounce = twice the light crossing of the trapped annulus): the weights in |kappa| = 1, 2 and
in kappa = 9 (o5_lift_v1.channel_projection summed over m_j and shells; both kappa signs), and the norm. On the chains the channel is
conserved, so f_mix = 0 there exactly."""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
from scipy.optimize import minimize_scalar
from scipy.special import jv

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "code_v6"))
sys.path.insert(0, os.path.join(HERE, "o5_copy"))
import page_ops_v6 as OPS  # noqa: E402
import pt_lib_v1 as L  # noqa: E402
import o5_lift_v1 as L1  # noqa: E402

CD = torch.complex128
ALPHA = [torch.as_tensor(A, dtype=CD) for A in L1.ALPHA]


class Cube:
    def __init__(self, N, v, device):
        self.dev = torch.device(device)
        self.N = torch.as_tensor(N, dtype=torch.float64, device=self.dev)
        self.v = [torch.as_tensor(x, dtype=torch.float64, device=self.dev) for x in v]
        self.A = [A.to(self.dev) for A in ALPHA]
        self.np_, self.nm_, self.wp_, self.wm_ = [], [], [], []
        for i in range(3):
            self.np_.append(0.5 * (self.N + self.shift(self.N, i, 1)))
            self.nm_.append(0.5 * (self.N + self.shift(self.N, i, -1)))
            self.wp_.append(0.5 * (self.v[i] + self.shift(self.v[i], i, 2)))
            self.wm_.append(0.5 * (self.v[i] + self.shift(self.v[i], i, -2)))
        self.bound = float(3.0 * self.N.abs().max() + 1.5 * max(float(x.abs().max()) for x in self.v)) * 1.01

    @staticmethod
    def shift(a, axis, s):
        """out[.., x, ..] = a[.., x + s, ..], zero outside (axis counted from the end of the 3 spatial axes)."""
        ax = a.ndim - 3 + axis
        out = torch.zeros_like(a)
        L_ = a.shape[ax]
        src = [slice(None)] * a.ndim; dst = [slice(None)] * a.ndim
        if s > 0:
            src[ax] = slice(s, L_); dst[ax] = slice(0, L_ - s)
        else:
            src[ax] = slice(0, L_ + s); dst[ax] = slice(-s, L_)
        out[tuple(dst)] = a[tuple(src)]
        return out

    def apply(self, psi):
        out = torch.zeros_like(psi)
        for i in range(3):
            D = self.np_[i][None] * self.shift(psi, i, 1) - self.nm_[i][None] * self.shift(psi, i, -1)
            out += (-0.5j) * torch.einsum("ab,bxyz->axyz", self.A[i], D)
            del D
            out += (-0.25j) * (self.wp_[i][None] * self.shift(psi, i, 2) - self.wm_[i][None] * self.shift(psi, i, -2))
        return out

    def expm(self, psi, tau, tol=1e-15):
        w = self.bound
        z = w * tau
        K, tail = OPS.cheb_order(z, tol)
        kk = np.arange(K + 1)
        ck = jv(kk, z) * ((-1j) ** kk) * np.where(kk == 0, 1.0, 2.0)
        T0 = psi; T1 = self.apply(psi) / w
        Y = complex(ck[0]) * T0 + complex(ck[1]) * T1
        for k in range(2, K + 1):
            T2 = 2.0 * self.apply(T1) / w - T0
            Y += complex(ck[k]) * T2
            T0, T1 = T1, T2
        return Y, K, tail


def load_table(which):
    if which == "rs64":
        import pt_fieldsA_v1 as FA
        H = FA.load_record("P64_s4_B2n4096", 100.0)
        k = H["k_rec"]
        return H["R_sites"], H["N_app"][k], H["v_app"][k], 64.0, 2 * H["g"] * H["E_rec"], {"label": H["label"], "npz_md5": H["npz_md5"], "T_rs": 100.0}
    p = "<internal>/record_slice.npz"
    Z = np.load(p); J = json.load(open(p.replace(".npz", ".json")))
    return np.asarray(Z["r"], float), np.asarray(Z["N"], float), np.asarray(Z["v"], float), 32.0, J["slice"]["two_gE_json"], {
        "label": "set17_rs32 T/r_s 18 (record_slice of GAP10 T7)", "npz_md5": L.md5_file(p), "source_md5": J["md5_source_npz"], "T_rs": 18.0}


def channel_weight(coords, vals, kappas, H):
    tot = 0.0
    nch = int(math.ceil(math.sqrt(3) * H)) + 2
    for kh in kappas:
        for assign in (+1, -1):
            jj = kh - 0.5
            for mj in np.arange(-jj, jj + 1e-9, 1.0):
                g = L1.channel_projection(coords, vals, kh, float(mj), nch, 1.0, assign)
                tot += float(np.sum(np.abs(g) ** 2))
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="rs64", choices=("rs64", "rs32"))
    ap.add_argument("--kappa", type=int, default=9)
    ap.add_argument("--box-rs", type=float, default=2.5)
    ap.add_argument("--bounces", default="0,0.5,1,2,4,8")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.device.startswith("cuda") and a.device not in ("cuda:1", "cuda:2"):
        raise SystemExit("only cuda:1 or cuda:2")
    t_start = time.time()
    rt, Nt, vt, rs_nom, rsE, prov = load_table(a.which)
    ut = -vt
    f = Nt - ut
    roots = [float(rt[i] + (rt[i + 1] - rt[i]) * f[i] / (f[i] - f[i + 1])) for i in range(len(f) - 1) if (f[i] > 0) != (f[i + 1] > 0)]
    r_in, r_out = roots[0], roots[1]
    Rc = 0.5 * (r_in + r_out)
    sigma = max(2.0, 0.25 * (r_out - r_in))
    Nc, uc = float(np.interp(Rc, rt, Nt)), float(np.interp(Rc, rt, ut))
    y = a.kappa / Rc
    om_up = lambda k: -uc * 0.5 * math.sin(2 * k) + Nc * math.sqrt(math.sin(k) ** 2 + y * y)
    res = minimize_scalar(om_up, bounds=(1e-6, math.pi / 2), method="bounded", options={"xatol": 1e-12})
    k0 = float(res.x)
    TH = 1.0 / (4 * math.pi * rsE)
    H = int(round(0.5 * a.box_rs * rs_nom))
    L.log("B' %s: record %s, trapped annulus [%.3f, %.3f] a (applied fields), R_c %.3f, sigma %.2f, N %.4f |v| %.4f at R_c; kappa %d: y %.4f, "
          "the upper band's minimum Killing energy %.5e at k_0 %.4f (= %.2f T_H; a pocket needs < 0); box half-width %d (side %.2f nominal r_s)" % (
              a.which, prov["label"], r_in, r_out, Rc, sigma, Nc, uc, a.kappa, y, res.fun, k0, res.fun / TH, H, (2 * H + 1) / rs_nom))
    # the chain packet and its energy on the chain operator (the reduction check's reference)
    n_ch = int(rt[-1])
    Rch = np.arange(1, n_ch + 1, dtype=float)
    h0, Q = OPS.chain_sparse(Rch, 0.0, float(a.kappa))
    hch = OPS.clocked_csr(h0, Q, np.interp(Rch, rt, Nt), -np.interp(Rch, rt, ut))
    s_, e_ = math.sin(k0), math.sqrt(math.sin(k0) ** 2 + y * y)
    u = np.array([s_ - 1j * y, e_], complex) / (math.sqrt(2) * e_)
    fr = np.exp(-(Rch - Rc) ** 2 / (4 * sigma ** 2)) * np.exp(1j * k0 * (Rch - Rc))
    psi_ch = (fr[:, None] * u[None, :]).reshape(-1)
    psi_ch /= np.linalg.norm(psi_ch)
    e_ch = float(np.real(np.vdot(psi_ch, hch @ psi_ch)))
    sd_ch = math.sqrt(max(float(np.real(np.vdot(hch @ psi_ch, hch @ psi_ch))) - e_ch ** 2, 0.0))
    # the lift (four conventions) and the reduction check
    R_lo, R_hi = max(0.5, Rc - 6 * sigma), min(Rc + 6 * sigma, H - 3)
    coords, r = L1.shell_sites(R_lo, R_hi, H)
    fr3 = np.exp(-(r - Rc) ** 2 / (4 * sigma ** 2)) * np.exp(1j * k0 * (r - Rc))
    Lc = 2 * H + 1
    Ncube, vcube = L1.fields_on_cube(H, rt, Nt, ut)
    cube = Cube(Ncube, vcube, a.device)
    best = None
    conv = {}
    for assign in (+1, -1):
        for lower in (+1.0, -1.0):
            Oa, Ob = L1.angular_pair(coords, r, a.kappa, 0.5, assign)
            vals = L1.lift_vector(r, fr3, u, Oa, Ob, lower)
            vals /= math.sqrt(float(np.sum(np.abs(vals) ** 2)))
            psi = np.zeros((4, Lc, Lc, Lc), complex)
            psi[:, coords[:, 0] + H, coords[:, 1] + H, coords[:, 2] + H] = vals
            pt = torch.as_tensor(psi, dtype=CD, device=a.device)
            hp = cube.apply(pt)
            e = float(torch.vdot(pt.reshape(-1), hp.reshape(-1)).real)
            sd = math.sqrt(max(float(torch.vdot(hp.reshape(-1), hp.reshape(-1)).real) - e * e, 0.0))
            conv["assign%+d_lower%+d" % (assign, int(lower))] = {"E": e, "E_sd": sd}
            if best is None or abs(e - e_ch) < abs(best[0] - e_ch):
                best = (e, (assign, lower), pt)
            del hp
    L.log("chain packet <h> %.6e sd %.4e; cube conventions %s; chosen %s" % (e_ch, sd_ch, json.dumps(conv), best[1]))
    # the torch cubic apply against o5_lift_v1.cubic_apply (numpy) on the chosen packet
    hp_t = cube.apply(best[2]).cpu().numpy()
    hp_n = L1.cubic_apply(best[2].cpu().numpy(), Ncube, vcube, 0.0)
    apply_check = float(np.max(np.abs(hp_t - hp_n)))
    L.log("torch cubic apply against o5_lift_v1.cubic_apply: max |diff| %.2e" % apply_check)
    del hp_t, hp_n
    # propagation and channel weights
    Tb = 2.0 * (r_out - r_in)
    bounces = [float(x) for x in a.bounces.split(",")]
    allc, allr = L1.shell_sites(0.5, H * math.sqrt(3) + 1, H)
    psi = best[2]
    t_now = 0.0
    rows = []
    for b in bounces:
        t = b * Tb
        if t > t_now:
            psi, K, tail = cube.expm(psi, t - t_now)
            t_now = t
        arr = psi.cpu().numpy()
        vals = arr[:, allc[:, 0] + H, allc[:, 1] + H, allc[:, 2] + H]
        nrm = float(np.sum(np.abs(arr) ** 2))
        w12 = channel_weight(allc, vals, (1, 2), H)
        w9 = channel_weight(allc, vals, (a.kappa,), H)
        ins = allr <= r_out
        w12_in = channel_weight(allc[ins], vals[:, ins], (1, 2), H)
        w9_in = channel_weight(allc[ins], vals[:, ins], (a.kappa,), H)
        face = float(sum(np.sum(np.abs(arr[:, s1, :, :]) ** 2) + np.sum(np.abs(arr[:, :, s1, :]) ** 2) + np.sum(np.abs(arr[:, :, :, s1]) ** 2)
                         for s1 in (slice(0, 3), slice(Lc - 3, Lc))))
        rin = float(np.sum((np.abs(arr) ** 2)[:, ((np.arange(Lc) - H)[:, None, None] ** 2 + (np.arange(Lc) - H)[None, :, None] ** 2
                                                   + (np.arange(Lc) - H)[None, None, :] ** 2) <= r_out ** 2]))
        rows.append({"bounces": b, "t": t, "norm": nrm, "w_k12": w12, "w_k9": w9, "f_mix": w12 / nrm, "weight_inside_r_out": rin, "face_weight": face,
                     "w_k12_inside_r_out": w12_in, "w_k9_inside_r_out": w9_in, "f_mix_inside": (w12_in / rin) if rin > 0 else None})
        L.log("  %.2f bounces (t %.1f a): norm %.12f, weight |kappa| = 1, 2: %.4e, kappa = %d: %.4f; inside the outer root %.4f of which |kappa| = 1, 2 %.4e, kappa %d %.4f; at the faces %.1e" % (
            b, t, nrm, w12, a.kappa, w9, rin, w12_in, a.kappa, w9_in, face))
    J = {"script": "pt_companion_bprime_v1.py", "args": vars(a), "record": prov, "rs_nom": rs_nom, "rs_E": rsE, "T_H": TH, "trapped_annulus": [r_in, r_out],
         "R_c": Rc, "sigma": sigma, "k0": k0, "omega_min_upper_band": float(res.fun), "omega_min_over_TH": float(res.fun / TH), "box_half_width": H,
         "chain_packet": {"E": e_ch, "E_sd": sd_ch}, "conventions": conv, "chosen": list(best[1]), "apply_check": apply_check, "bounce_time": Tb,
         "rows": rows, "engine_md5s": L.engine_md5s(), "o5_lift_v1_md5": L.md5_file(os.path.join(HERE, "o5_copy", "o5_lift_v1.py")), "seconds": time.time() - t_start}
    os.makedirs(a.out, exist_ok=True)
    p = os.path.join(a.out, "bprime_%s_k%d.json" % (a.which, a.kappa))
    json.dump(J, open(p, "w"), indent=1, default=float)
    L.log("wrote %s" % p)


if __name__ == "__main__":
    main()
