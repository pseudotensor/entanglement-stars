#!/usr/bin/env python
"""banded_eigh_proto_v1.py (busi lane, reviewer A redesign prototype R1): the O(n^2) banded GPU eigensolver against torch.linalg.eigh on
the production clocked chains, accuracy on stored data and the measured factor on the box GPU.
usage: python banded_eigh_proto_v1.py OUT_DIR CODE_DIR [--w2 SLICE_NPZ] [--iso FIELDS_NPZ] [--synthetic NSITES] [--gpu 0]
  CODE_DIR: a production code tree holding engine_vacflow_v1.py (the chain assembly, read only).
  --w2: the W2 r2b T0013.50 slice (2048 sites; N_applied, v_applied, kappas, e_k, mask_k, R2_k): accuracy against the stored spectrum,
        the stored occupied reference and the dense solver, per channel.
  --iso: the ISO_128 fields npz (4096 sites; the last record's N_applied, v_applied): the 8192 x 8192 case against the dense solver.
  --synthetic: a chain of NSITES sites with the vacuum profile (timing and self-consistency only).
Outputs: OUT_DIR/banded_eigh_proto_v1.json and .txt."""
import argparse
import json
import os
import sys
import time

import numpy as np
import scipy.sparse as sp
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import banded_eigh_gpu_v1 as BE


def clocked_csr(h0, Q, N, v):
    Nd = sp.diags(np.repeat(np.asarray(N, float), 2)); Vd = sp.diags(np.repeat(np.asarray(v, float), 2))
    return (0.5 * (Nd @ h0 + h0 @ Nd) + 0.5 * (Vd @ Q + Q @ Vd)).tocsr()


def dens(A_t, Phi):
    """per site: sum over columns of Re(Phi^dag P_n A Phi) (torch dense A or band apply), as numpy."""
    AP = A_t(Phi)
    d = (Phi.conj() * AP).real.sum(dim=1)
    return d.reshape(-1, 2).sum(dim=1).cpu().numpy()


def unpack_mask(m, n):
    return np.unpackbits(np.asarray(m, np.uint8))[:n].astype(bool)


def run_case(name, h0, Q, N, v, dev, lines, out, e_stored=None, mask_stored=None, R2_stored=None, dense=True, reps=2):
    n = 2 * len(N)
    hm = clocked_csr(h0, Q, N, v)
    band = BE.band_from_csr(hm)
    hb = BE.band_from_csr(h0.tocsr()); qb = BE.band_from_csr(Q.tocsr())
    rec = {"name": name, "n_mat": n}
    # ---- the banded route (warm-up once, then timed)
    e_b, U_b, info = BE.eigh_banded(band, n, dev, half="ph")
    tms = []
    for r in range(reps):
        tm = {}
        e_b, U_b, info = BE.eigh_banded(band, n, dev, half="ph", timing=tm)
        tms.append(tm)
    tm = min(tms, key=lambda q: q["total_s"])
    rec["banded_timing_s"] = tm; rec["banded_info"] = info
    band_t = torch.as_tensor(band, device=dev)
    hb_t = torch.as_tensor(hb, device=dev); qb_t = torch.as_tensor(qb, device=dev)
    # self checks
    HU = BE.band_matvec(band_t, U_b)
    res_b = torch.linalg.norm(HU - U_b * e_b[None, :], dim=0)
    G = torch.linalg.norm(U_b.conj().T @ U_b - torch.eye(n, dtype=U_b.dtype, device=dev))
    orth_max = float((U_b.conj().T @ U_b - torch.eye(n, dtype=U_b.dtype, device=dev)).abs().max())
    rec["banded_residual_max"] = float(res_b.max()); rec["banded_orth_fro"] = float(G); rec["banded_orth_max"] = orth_max
    rec["banded_ph_symmetry_max"] = float((e_b + e_b.flip(0)).abs().max())
    rec["n_neg_banded"] = int((e_b < 0).sum())
    lines.append("%s: n %d banded total %.3f s (setup %.3f bisect %.3f invit %.3f post %.3f); residual max %.2e, orth max %.2e (fro %.2e), brackets max rel %.1e, isolated %.4f, count sweeps mean %.1f max %d, |shift - RQ| max %.1e, neighbour overlap before polish %s, clusters %d %s" % (
        name, n, tm["total_s"], tm["setup_s"], tm["bisect_s"], tm["invit_s"], tm["post_s"], rec["banded_residual_max"], orth_max, float(G), info["bracket_max_rel"], info["isolated_fraction"], info["count_sweeps_mean"], info["count_sweeps_max"], info["shift_vs_rayleigh_max"], info["neighbour_overlap_before_polish"], info["clusters"], info["cluster_sizes"][:8]))
    if e_stored is not None:
        es = np.sort(np.asarray(e_stored, float))
        rec["stored_e_max_abs_diff"] = float(np.max(np.abs(es - e_b.cpu().numpy())))
        lines.append("  against the stored spectrum: max |de| %.2e (G %.3g)" % (rec["stored_e_max_abs_diff"], info["gershgorin"]))
    if dense:
        hd = torch.as_tensor(hm.toarray(), device=dev)
        torch.cuda.synchronize(); t0 = time.time(); e_d, U_d = torch.linalg.eigh(hd); torch.cuda.synchronize(); td = time.time() - t0
        torch.cuda.synchronize(); t0 = time.time(); e_d, U_d = torch.linalg.eigh(hd); torch.cuda.synchronize(); td = min(td, time.time() - t0)
        e_d = e_d.real
        rec["dense_eigh_s"] = td
        rec["factor_box"] = td / tm["total_s"]
        HUd = hd @ U_d
        res_d = torch.linalg.norm(HUd - U_d * e_d[None, :], dim=0)
        rec["dense_residual_max"] = float(res_d.max())
        rec["dense_ph_symmetry_max"] = float((e_d + e_d.flip(0)).abs().max())
        rec["e_max_abs_diff_vs_dense"] = float((e_b - e_d).abs().max())
        rec["e_max_rel_diff_vs_dense"] = float(((e_b - e_d).abs() / e_d.abs().clamp_min(1e-300)).max())
        # projector onto the negative set (the sea) and its densities
        neg_b = e_b < 0; neg_d = e_d < 0
        Pb = U_b[:, neg_b] @ U_b[:, neg_b].conj().T; Pd = U_d[:, neg_d] @ U_d[:, neg_d].conj().T
        dP = Pb - Pd
        rec["sea_projector_fro_diff"] = float(torch.linalg.norm(dP)); rec["sea_projector_max_diff"] = float(dP.abs().max())
        del Pb, Pd, dP
        for lab, A_t in (("h0", lambda X: BE.band_matvec(hb_t, X)), ("Q", lambda X: BE.band_matvec(qb_t, X))):
            db = dens(A_t, U_b[:, neg_b]); dd = dens(A_t, U_d[:, neg_d])
            sc = max(float(np.max(np.abs(dd))), 1e-300)
            rec["sea_dens_%s_max_rel_diff" % lab] = float(np.max(np.abs(db - dd))) / sc
        lines.append("  dense eigh %.3f s -> factor %.1fx on this GPU; e max |diff| %.2e (rel %.2e); dense residual max %.2e; PH symmetry max: banded %.1e dense %.1e; sea projector diff fro %.2e max %.2e; sea densities rel diff h0 %.2e Q %.2e" % (
            td, rec["factor_box"], rec["e_max_abs_diff_vs_dense"], rec["e_max_rel_diff_vs_dense"], rec["dense_residual_max"], rec["banded_ph_symmetry_max"], rec["dense_ph_symmetry_max"],
            rec["sea_projector_fro_diff"], rec["sea_projector_max_diff"], rec["sea_dens_h0_max_rel_diff"], rec["sea_dens_Q_max_rel_diff"]))
        if mask_stored is not None and R2_stored is not None:
            m = unpack_mask(mask_stored, n)
            R2 = torch.as_tensor(R2_stored, device=dev)
            n_occ = R2.shape[1]
            rec["mask_true"] = int(m.sum()); rec["R2_cols"] = n_occ
            # the max-overlap set from each basis against the stored reference (the engine's dgk_track rule)
            wb = ((U_b.conj().T @ R2).abs() ** 2).sum(dim=1).cpu().numpy(); wd = ((U_d.conj().T @ R2).abs() ** 2).sum(dim=1).cpu().numpy()
            sb = np.zeros(n, bool); sb[np.argsort(-wb, kind="stable")[:n_occ]] = True
            sd = np.zeros(n, bool); sd[np.argsort(-wd, kind="stable")[:n_occ]] = True
            rec["tracked_set_equal_banded_dense"] = bool(np.array_equal(sb, sd)); rec["tracked_set_equal_stored_mask"] = bool(np.array_equal(sb, m))
            rec["tracked_min_occupied_weight"] = [float(wb[sb].min()), float(wd[sd].min())]
            rec["tracked_max_empty_weight"] = [float(wb[~sb].max()), float(wd[~sd].max())]
            # occupied-set projector and densities against the stored reference
            Pb = U_b[:, sb] @ U_b[:, sb].conj().T; Ps = R2 @ R2.conj().T
            rec["occ_projector_vs_stored_fro"] = float(torch.linalg.norm(Pb - Ps)); rec["occ_projector_vs_stored_max"] = float((Pb - Ps).abs().max())
            Pd = U_d[:, sd] @ U_d[:, sd].conj().T
            rec["occ_projector_dense_vs_stored_fro"] = float(torch.linalg.norm(Pd - Ps))
            del Pb, Ps, Pd
            for lab, A_t in (("h0", lambda X: BE.band_matvec(hb_t, X)), ("Q", lambda X: BE.band_matvec(qb_t, X))):
                dsb = dens(A_t, U_b[:, sb]); dss = dens(A_t, R2); dsd = dens(A_t, U_d[:, sd])
                sc = max(float(np.max(np.abs(dss))), 1e-300)
                rec["occ_dens_%s_banded_vs_stored_rel" % lab] = float(np.max(np.abs(dsb - dss))) / sc
                rec["occ_dens_%s_dense_vs_stored_rel" % lab] = float(np.max(np.abs(dsd - dss))) / sc
                rec["occ_dens_%s_banded_vs_dense_rel" % lab] = float(np.max(np.abs(dsb - dsd))) / sc
            lines.append("  stored reference (%d columns; mask %d true): tracked set banded == dense %s, == stored mask %s; min occupied weight %.9f/%.9f, max empty %.1e/%.1e; occupied projector vs stored fro %.2e max %.2e (dense vs stored fro %.2e); occupied densities rel diff: banded vs stored h0 %.1e Q %.1e, dense vs stored h0 %.1e Q %.1e, banded vs dense h0 %.1e Q %.1e" % (
                n_occ, int(m.sum()), rec["tracked_set_equal_banded_dense"], rec["tracked_set_equal_stored_mask"], rec["tracked_min_occupied_weight"][0], rec["tracked_min_occupied_weight"][1],
                rec["tracked_max_empty_weight"][0], rec["tracked_max_empty_weight"][1], rec["occ_projector_vs_stored_fro"], rec["occ_projector_vs_stored_max"], rec["occ_projector_dense_vs_stored_fro"],
                rec["occ_dens_h0_banded_vs_stored_rel"], rec["occ_dens_Q_banded_vs_stored_rel"], rec["occ_dens_h0_dense_vs_stored_rel"], rec["occ_dens_Q_dense_vs_stored_rel"],
                rec["occ_dens_h0_banded_vs_dense_rel"], rec["occ_dens_Q_banded_vs_dense_rel"]))
        del hd, U_d, HUd
    out.append(rec)
    del U_b, HU
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir"); ap.add_argument("code_dir")
    ap.add_argument("--w2", default=None); ap.add_argument("--iso", default=None); ap.add_argument("--synthetic", type=int, default=0)
    ap.add_argument("--gpu", type=int, default=0); ap.add_argument("--w2-channels", default="1,2,5,10"); ap.add_argument("--m", type=float, default=0.2)
    ap.add_argument("--no-dense", action="store_true")
    a = ap.parse_args()
    sys.path.insert(0, a.code_dir)
    import engine_vacflow_v1 as VF
    os.makedirs(a.out_dir, exist_ok=True)
    dev = torch.device("cuda:%d" % a.gpu)
    torch.cuda.set_device(dev)
    lines = []; out = []
    lines.append("banded_eigh_proto_v1: GPU %s (%s); torch %s; numba.cuda" % (a.gpu, torch.cuda.get_device_name(dev), torch.__version__))
    if a.w2:
        z = np.load(a.w2, allow_pickle=False)
        R = np.asarray(z["R"], float); N = np.asarray(z["N_applied"], float); v = np.asarray(z["v_applied"], float)
        kappas = [int(k) for k in np.asarray(z["kappas"])]
        chs = [int(c) for c in a.w2_channels.split(",")]
        lines.append("W2 slice %s: %d sites, T %.4f, kappas %s" % (os.path.basename(a.w2), len(R), float(z["T"]), kappas))
        for kap in chs:
            h0, Q = VF.chain_sparse(R, a.m, float(kap))
            run_case("w2_kappa%d" % kap, h0, Q, N, v, dev, lines, out, e_stored=z["e_%d" % kap], mask_stored=z["mask_%d" % kap], R2_stored=z["R2_%d" % kap], dense=not a.no_dense)
            print(lines[-1], flush=True)
    if a.iso:
        z = np.load(a.iso, allow_pickle=False)
        R = np.asarray(z["R"], float); i = len(z["T"]) - 1
        N = np.asarray(z["N_applied"][i], float); v = np.asarray(z["v_applied"][i], float)
        lines.append("ISO fields %s: %d sites, record %d T %.4f" % (os.path.basename(a.iso), len(R), i, float(z["T"][i])))
        for kap in (1, 10):
            h0, Q = VF.chain_sparse(R, a.m, float(kap))
            run_case("iso_kappa%d" % kap, h0, Q, N, v, dev, lines, out, dense=not a.no_dense)
            print(lines[-1], flush=True)
    if a.synthetic:
        n_s = a.synthetic
        R = np.arange(n_s, dtype=float) + 1.0
        rs = n_s / 32.0
        v = VF.profile(R, rs, "exact"); N = 1.0 - 0.3 * np.exp(-(R / (2 * rs)) ** 2)
        h0, Q = VF.chain_sparse(R, a.m, 1.0)
        run_case("synthetic_%d" % n_s, h0, Q, N, v, dev, lines, out, dense=not a.no_dense, reps=1)
        print(lines[-1], flush=True)
    with open(os.path.join(a.out_dir, "banded_eigh_proto_v1.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(a.out_dir, "banded_eigh_proto_v1.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
