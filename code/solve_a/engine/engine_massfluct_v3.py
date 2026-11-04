#!/usr/bin/env python
"""engine_massfluct_v3.py (builder analyst; version 3 of engine_massfluct_v2.py after the red team R24f, defects 1 to 3: the star's
exchange in the ordering of the cluster's primary, the Hartree pair energy in the same kernel as the denominator, and the variances kept
apart). Everything version 2 prints is printed again (same filling, same loops); added, per channel and summed with the weights 16 kappa:

(1) the Wick-ordered exchange against the star's reference (the sea C_0 of the same h_m[N, v]; D = C - C_0 = P, the matter's occupations
    f_k on the positive modes, the last multiplet fractional as in version 2):
        E_x^W = -(1/2) sum_nn' V_nn' Tr(P hcal_n P hcal_n'),   V_nn' = -g/max(R_n, R_n')
    (engine_hf_cluster_v3's ex_wick with D = P; nonnegative for this kernel; for one occupied mode it cancels the Hartree self-energy);
(2) the matter's Hartree pair energy in the same kernel, E_H = (1/2) sum_nn' V_nn' e_n e_n' with e_n = sum_kappa 16 kappa sum_k f_k <k|hcal_n|k>,
    and y^W = E_x^W/E_H (the cluster's y); beside it, the same pair energy from the record's eps_tot and version 2's binding-energy denominator
    E - N_B m (labelled: not a potential energy);
(3) the one-particle control: in channel 1 the lowest positive mode alone (f = 1, one copy): E_x^W + E_H must vanish to rounding;
(4) the variances at the reading radii, kept apart (red team R24f defect 2): with A = m(R_c) (the unclocked energy-density sum to site c),
        V(C)   = Tr[C A (1 - C) A]               the physical variance of the mass operator (the sea's channels included; cutoff dependent),
        V(C_0) = Tr[C_0 A (1 - C_0) A]           the reference's (no matter; same fields),
        dV     = V(C) - V(C_0) = V(P) - 2 B      the vacuum-subtracted variance,  B = Tr(C_0 A P A),
        V(P)   = Tr[P A (1 - P) A]               the matter's isolated variance,
        W      = -Tr(P A P A)                     the Wick-ordered source product <:A A:_0> - <:A:_0>^2 (not a variance; <= 0);
    version 2's "Var_mat" is Tr[P A (1 - C) A] = V(P) - B and its "Var_sea" is Tr[C_0 A (1 - C) A] = V(C_0) - B.
Version 2's E_x^(0) (the connected correlator with f_k (1 - f_k'), i.e. the symmetric product's self term included) is kept, labelled.
Args: --star --member --gpu --out --tag."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse
import json
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine_ops_v1 as OPS  # noqa: E402


def channel_fluct_v3(h, U, e, fw, R, g, dev=None, light=False):
    """per channel: version 2's F_mat, F_sea, E_x^(0) (matter, sea) unless light, and the new cumulative blocks: Y (matter x matter, weights
    sqrt(f f')), Z (matter x sea, weights sqrt(f')); returns a dict with the profiles F_mat, F_sea, B (= |Z|^2), TrPAPA (= |Y|^2), the matter's
    site energies e_mat, and the sums."""
    import torch
    td = torch.device("cuda:%d" % dev) if dev is not None else torch.device("cpu")
    occ = np.nonzero(fw > 0)[0]
    un = np.nonzero(fw < 1)[0]
    sea = np.nonzero(e < 0)[0]
    mat = np.nonzero((e > 0) & (fw > 0))[0]
    H = torch.as_tensor(h, device=td)
    T = lambda ix: torch.as_tensor(U[:, ix], device=td)
    Us, Um = T(sea), T(mat)
    As, Am = H @ Us, H @ Um
    sm = torch.as_tensor(np.sqrt(fw[mat]), device=td)
    nsites = h.shape[0] // 2
    if not light:
        Uo, Uu = T(occ), T(un)
        Ao, Bu = H @ Uo, H @ Uu
        so = torch.as_tensor(np.sqrt(fw[occ]), device=td)
        su = torch.as_tensor(np.sqrt(1.0 - fw[un]), device=td)
        sea_cols = torch.as_tensor(e[occ] < 0, device=td)
        mat_cols = ~sea_cols
        X = torch.zeros((len(un), len(occ)), dtype=Uo.dtype, device=td)
    Y = torch.zeros((len(mat), len(mat)), dtype=Um.dtype, device=td)
    Z = torch.zeros((len(mat), len(sea)), dtype=Um.dtype, device=td)
    F_mat = np.zeros(nsites); F_sea = np.zeros(nsites); Bp = np.zeros(nsites); Wp = np.zeros(nsites); e_mat = np.zeros(nsites)
    Ex_mat = Ex_sea = 0.0
    ExW_sum = 0.0
    fm = torch.as_tensor(fw[mat], device=td)
    for n in range(nsites):
        sl = slice(2 * n, 2 * n + 2)
        if not light:
            dM = 0.5 * (Uu[sl].conj().T @ Ao[sl] + Bu[sl].conj().T @ Uo[sl])
            dW = (su[:, None] * dM) * so[None, :]
            cross = (dW.conj() * X).real
            sq = (dW.conj() * dW).real
            en = (-g / R[n]) * (sq + 2.0 * cross)
            Ex_mat += float(en[:, mat_cols].sum()); Ex_sea += float(en[:, sea_cols].sum())
            X = X + dW
            a2 = (X.conj() * X).real
            F_mat[n] = float(a2[:, mat_cols].sum()); F_sea[n] = float(a2[:, sea_cols].sum())
        if len(mat):
            dY = 0.5 * (Um[sl].conj().T @ Am[sl] + Am[sl].conj().T @ Um[sl])          # <k'| hcal_n |k>, both matter
            e_mat[n] = float((fm * torch.diagonal(dY).real).sum())
            dYw = (sm[:, None] * dY) * sm[None, :]
            ExW_sum += float(((-g / R[n]) * ((dYw.conj() * dYw).real + 2.0 * (dYw.conj() * Y).real)).sum())
            Y = Y + dYw
            Wp[n] = float((Y.conj() * Y).real.sum())
            dZ = 0.5 * (Um[sl].conj().T @ As[sl] + Am[sl].conj().T @ Us[sl])          # <k'| hcal_n |k>, k' matter, k sea
            Z = Z + sm[:, None] * dZ
            Bp[n] = float((Z.conj() * Z).real.sum())
    return {"F_mat": F_mat, "F_sea": F_sea, "B": Bp, "TrPAPA": Wp, "e_mat": e_mat, "Ex0_mat": 0.5 * Ex_mat, "Ex0_sea": 0.5 * Ex_sea,
            "ExW": -0.5 * ExW_sum, "n_mat": len(mat), "n_sea": len(sea)}


def pair_energy(e_site, R, g):
    """(1/2) sum_nn' V_nn' e_n e_n' with V_nn' = -g/max(R_n, R_n') (sites in increasing R)."""
    c = np.concatenate([[0.0], np.cumsum(e_site)[:-1]])
    return float(0.5 * np.sum((-g / R) * (e_site ** 2 + 2.0 * e_site * c)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--star", required=True)
    ap.add_argument("--member", type=int, default=None)
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    D = json.load(open(a.star))
    mem_list = D["members"]
    if a.member is None:
        ok = [i for i, mm in enumerate(mem_list) if not mm.get("failed")]
        i_mem = max(ok, key=lambda i: mem_list[i]["E"])
    else:
        i_mem = a.member
    M = mem_list[i_mem]
    m, g, nsites = M["m"], M["g"], M["nsites"]
    half = bool(D["args"].get("half_offset", 0))
    R = np.arange(nsites, dtype=float) + (0.5 if half else 1.0)
    N = np.array(M["N"]); v = np.array(M["v"])
    OPS.GPU["dev"] = a.gpu
    t0 = time.time()
    NBt = float(M["N_B"])
    # version 2's filling, unchanged
    ep_all = []
    above = 0
    for kappa in range(1, 2000):
        h, Q, _ = OPS.chain_parts(nsites, m, float(kappa), half)
        e, U = OPS.spectrum(OPS.clocked(h, Q, N, v))
        ep = e[e > 0]
        ep_all.append((kappa, ep))
        lv = np.concatenate([x[1] for x in ep_all]); wv = np.concatenate([np.full(len(x[1]), 16.0 * x[0]) for x in ep_all])
        o = np.argsort(lv); cw = np.cumsum(wv[o])
        E_top_est = float(lv[o][min(int(np.searchsorted(cw, NBt - 1e-9)), len(lv) - 1)]) if cw[-1] >= NBt else np.inf
        above = above + 1 if ep.min() > E_top_est else 0
        if above >= 2:
            break
    lev = sorted([(float(x), 16.0 * k, ic, j) for ic, (k, ep) in enumerate(ep_all) for j, x in enumerate(ep)])
    occ_pos = [np.zeros(len(ep)) for _, ep in ep_all]
    left = NBt; i = 0; E_top = None
    while i < len(lev) and left > 1e-12:
        j = i
        while j < len(lev) and abs(lev[j][0] - lev[i][0]) < 1e-9:
            j += 1
        wsum = sum(x[1] for x in lev[i:j])
        frac = min(1.0, left / wsum)
        for x in lev[i:j]:
            occ_pos[x[2]][x[3]] = frac
        left -= frac * wsum
        E_top = lev[i][0]
        i = j
    print("filling rebuilt: %d channels, E_top %.8f (the record's %s)" % (len(ep_all), E_top, M.get("E_top")), flush=True)
    keys = ("F_mat", "F_sea", "B", "TrPAPA", "e_mat")
    tot = {k: np.zeros(nsites) for k in keys}
    Ex_mat = Ex_sea = ExW = 0.0
    per = []
    NB = 0.0
    control = None
    for (kappa, ep), fpos in zip(ep_all, occ_pos):
        h, Q, _ = OPS.chain_parts(nsites, m, float(kappa), half)
        e, U = OPS.spectrum(OPS.clocked(h, Q, N, v))
        fw = np.where(e < 0, 1.0, 0.0)
        fw[e > 0] = fpos
        r = channel_fluct_v3(h, U, e, fw, R, g, a.gpu)
        w = 16.0 * kappa
        for k in keys:
            tot[k] += w * r[k]
        Ex_mat += w * r["Ex0_mat"]; Ex_sea += w * r["Ex0_sea"]; ExW += w * r["ExW"]
        NB += w * fpos.sum()
        if kappa == 1:
            f1 = np.where(e < 0, 1.0, 0.0)
            f1[np.nonzero(e > 0)[0][0]] = 1.0
            c1 = channel_fluct_v3(h, U, e, f1, R, g, a.gpu, light=True)
            EH1 = pair_energy(c1["e_mat"], R, g)
            control = {"mode_energy": float(e[e > 0].min()), "E_x_W": c1["ExW"], "E_H": EH1, "sum": c1["ExW"] + EH1,
                       "relative": (c1["ExW"] + EH1) / abs(EH1) if EH1 else None}
            print("one-particle control (channel 1, the lowest positive mode alone): E_x^W %.12e E_H %.12e sum %.2e (relative %.1e)" % (
                c1["ExW"], EH1, control["sum"], control["relative"]), flush=True)
        per.append({"kappa": kappa, "n_matter_modes": r["n_mat"], "Var_mat_v2_at_end": float(w * r["F_mat"][-1]), "Var_sea_v2_at_end": float(w * r["F_sea"][-1]),
                    "B_at_end": float(w * r["B"][-1]), "TrPAPA_at_end": float(w * r["TrPAPA"][-1]), "Ex0_mat": float(w * r["Ex0_mat"]), "Ex0_sea": float(w * r["Ex0_sea"]),
                    "E_x_W": float(w * r["ExW"])})
        print("kappa %d: matter modes %d; v2's Var_mat(end) %.4e Var_sea(end) %.4e; B(end) %.4e TrPAPA(end) %.4e; E_x^(0) mat %.4e sea %.4e; E_x^W %.4e (%.0f s)" % (
            kappa, r["n_mat"], w * r["F_mat"][-1], w * r["F_sea"][-1], w * r["B"][-1], w * r["TrPAPA"][-1], w * r["Ex0_mat"], w * r["Ex0_sea"], w * r["ExW"],
            time.time() - t0), flush=True)
    E_H = pair_energy(tot["e_mat"], R, g)
    eps = np.array(M["eps_tot"])
    E_H_rec = pair_energy(eps, R, g)
    VC = tot["F_mat"] + tot["F_sea"]; VC0 = tot["F_sea"] + tot["B"]; dV = tot["F_mat"] - tot["B"]; VP = tot["F_mat"] + tot["B"]; Wk = -tot["TrPAPA"]
    sroot = lambda x: float(np.sign(x) * np.sqrt(abs(x)))
    reads = {}
    for lab in ("R_surf", "R_999"):
        if M.get(lab):
            ic = min(int(np.searchsorted(R, float(M[lab]))), nsites - 1)
            E_ = M["E"]
            reads[lab] = {"r": float(M[lab]), "site": int(ic),
                          "v2_fluct_mat_over_E": float(np.sqrt(max(tot["F_mat"][ic], 0.0)) / E_), "v2_fluct_sea_over_E": float(np.sqrt(max(tot["F_sea"][ic], 0.0)) / E_),
                          "sqrt_V_C_over_E": sroot(VC[ic]) / E_, "sqrt_V_C0_over_E": sroot(VC0[ic]) / E_, "signed_sqrt_dV_over_E": sroot(dV[ic]) / E_,
                          "sqrt_V_P_over_E": sroot(VP[ic]) / E_, "Wick_product_over_E2": float(Wk[ic] / E_ ** 2), "B_over_E2": float(tot["B"][ic] / E_ ** 2)}
    binding = M["E"] - NB * m
    out = {"script": "engine_massfluct_v3.py", "star": os.path.basename(a.star), "member": i_mem, "E_top_rebuilt": E_top, "E_top_record": M.get("E_top"),
           "N_B": NB, "N_B_record": M["N_B"], "E": M["E"], "ell": M.get("ell"), "reads": reads, "one_over_sqrt_NB": float(1.0 / np.sqrt(NB)),
           "Ex0_matter_v2_form": Ex_mat, "Ex0_sea_v2_form": Ex_sea, "E_x_W": ExW, "E_H_same_kernel": E_H, "E_H_from_record_eps_tot": E_H_rec,
           "y_W": ExW / E_H if E_H else None, "binding_E_minus_NBm": binding, "E_x_W_over_binding": ExW / binding if binding else None,
           "one_particle_control": control,
           "profile_r": R[::max(1, nsites // 256)].tolist(),
           "profile_sqrt_V_C_over_m": None, "per_channel": per, "seconds": time.time() - t0}
    mprof = np.cumsum(eps)
    sub = slice(None, None, max(1, nsites // 256))
    with np.errstate(invalid="ignore", divide="ignore"):
        out["profile_sqrt_V_C_over_m"] = (np.sqrt(np.maximum(VC, 0.0)) / np.where(mprof > 0, mprof, np.nan))[sub].tolist()
        out["profile_signed_sqrt_dV_over_m"] = (np.sign(dV) * np.sqrt(np.abs(dV)) / np.where(mprof > 0, mprof, np.nan))[sub].tolist()
        out["profile_sqrt_V_P_over_m"] = (np.sqrt(np.maximum(VP, 0.0)) / np.where(mprof > 0, mprof, np.nan))[sub].tolist()
        out["profile_Wick_product_over_m2"] = (Wk / np.where(mprof > 0, mprof, np.nan) ** 2)[sub].tolist()
    os.makedirs(a.out, exist_ok=True)
    json.dump(out, open(os.path.join(a.out, "massfluct3_%s.json" % a.tag), "w"), indent=1, default=float)
    rs_ = reads.get("R_surf", {})
    print("member %d: N_B %.2f E %.6f; E_x^W %.6e, E_H (same kernel) %.6e (from eps_tot %.6e), y^W %.6e; E_x^W/(E - N_B m) %.6e; E_x^(0) v2 form matter %.6e sea %.6e; "
          "at R_surf: sqrt V(C)/E %s, sqrt V(C_0)/E %s, signed sqrt dV/E %s, sqrt V(P)/E %s, Wick product/E^2 %s (v2's fluct_mat/E %s) (%.0f s)" % (
              i_mem, NB, M["E"], ExW, E_H, E_H_rec, out["y_W"], out["E_x_W_over_binding"], Ex_mat, Ex_sea, rs_.get("sqrt_V_C_over_E"), rs_.get("sqrt_V_C0_over_E"),
              rs_.get("signed_sqrt_dV_over_E"), rs_.get("sqrt_V_P_over_E"), rs_.get("Wick_product_over_E2"), rs_.get("v2_fluct_mat_over_E"), time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
