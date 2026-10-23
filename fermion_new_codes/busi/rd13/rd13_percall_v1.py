#!/usr/bin/env python
"""rd13_percall_v1.py (busi, analyst; PERF RD-1 + RD-3 self-test part (b)): the production module engine_banded_eigh_v1 per call on the stored
production matrices against the dense route (torch.linalg.eigh, engine_collapse_v5.eigT's call) and the engine's eigenbasis J.v formula.
Cases: the W2 r2b T0013.50 slice (2048 sites, 2n = 4096; its stored e, mask, R2 per channel), the ISO_128 nkqh2 T0005.00 slice (4096 sites,
2n = 8192; its stored e, mask, R2), a synthetic 8192-site chain (2n = 16384; the vacuum profile of the prototype's run_synth16k_v2; no stored
reference). The input of eigT_banded is the dense matrix the engines build (clocked_gpu's operator, here from the host CSR), so the band
extraction (G2) and every guard run as in the engine.
Per channel: (1) eigenvalues: banded vs dense (max abs, max rel), banded and dense vs the stored spectrum; (2) residuals and orthonormality
(max |U^H U - I| over the full matrix) of both; particle-hole symmetry; (3) the sea projector (e < 0) Frobenius and max difference, the sea
densities of h0 and Q; (4) with a stored reference: the max-overlap tracked set (the engines' rule: the n_occ largest column weights of
|U^H R2|^2) from each basis against each other and against the stored mask, the min occupied and max empty weight, the occupied projector
against R2 R2^H and the occupied densities against R2's; (5) the J.v X: on the stored mask's occupied set and the perturbation dh = h_m(dN,
dv) with (dN, dv) the applied fields' change over the record before the slice (the synthetic case: a smooth bump of 1e-3), the density
responses deps, dJ by (i) the engine's eigenbasis formula on the dense basis (the production), (ii) sternheimer_X on the banded basis (the
--eigh banded --jvp sternheimer arm), (iii) sternheimer_X on the dense basis, relative to max |deps|, |dJ| of (i); the unoccupied block
U_u^H Y against M; the seconds of each route.
    python rd13_percall_v1.py OUT_DIR CODE_DIR [--w2 SLICE --w2-fields FIELDS] [--iso SLICE --iso-fields FIELDS] [--synthetic 8192] [--channels-w2 1,2,5,10] [--channels-iso 1,10]
Writes OUT_DIR/rd13_percall_v1.json and .txt."""
import argparse
import json
import os
import sys
import time

import numpy as np
import scipy.sparse as sp
import torch


def clocked_csr(h0, Q, N, v):
    Nd = sp.diags(np.repeat(np.asarray(N, float), 2)); Vd = sp.diags(np.repeat(np.asarray(v, float), 2))
    return (0.5 * (Nd @ h0 + h0 @ Nd) + 0.5 * (Vd @ Q + Q @ Vd)).tocsr()


def unpack_mask(m, n):
    return np.unpackbits(np.asarray(m, np.uint8))[:n].astype(bool)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir"); ap.add_argument("code_dir")
    ap.add_argument("--w2", default=None); ap.add_argument("--w2-fields", default=None)
    ap.add_argument("--iso", default=None); ap.add_argument("--iso-fields", default=None)
    ap.add_argument("--synthetic", type=int, default=0)
    ap.add_argument("--channels-w2", default="1,2,5,10"); ap.add_argument("--channels-iso", default="1,10")
    ap.add_argument("--m", type=float, default=0.2); ap.add_argument("--gpu", type=int, default=0)
    a = ap.parse_args()
    sys.path.insert(0, a.code_dir)
    import engine_vacflow_v1 as VF
    import engine_banded_eigh_v1 as EBE
    os.makedirs(a.out_dir, exist_ok=True)
    dev = torch.device("cuda:%d" % a.gpu); torch.cuda.set_device(dev)
    lines = []; out = []

    def say(s):
        lines.append(s); print(s, flush=True)
    tw = EBE.warmup(dev)
    import hashlib
    md5m = hashlib.md5(open(EBE.__file__, "rb").read()).hexdigest()
    say("rd13_percall_v1: %s (%s, md5 %s); GPU %s; torch %s, numba %s; warmup (JIT) %.1f s" % (EBE.VERSION, EBE.__file__, md5m, torch.cuda.get_device_name(dev), torch.__version__, EBE.numba.__version__, tw))

    def dens(band_t, Phi):
        AP = EBE.band_matvec(band_t, Phi)
        return (Phi.conj() * AP).real.sum(dim=1).reshape(-1, 2).sum(dim=1).cpu().numpy()

    def dens_pair(band_t, Uo, X):
        d_ = (Uo.conj() * EBE.band_matvec(band_t, X)).real.sum(dim=1) + (X.conj() * EBE.band_matvec(band_t, Uo)).real.sum(dim=1)
        return d_.reshape(-1, 2).sum(dim=1).cpu().numpy()

    def rel(x, ref):
        return float(np.max(np.abs(np.asarray(x) - np.asarray(ref)))) / max(float(np.max(np.abs(ref))), 1e-300)

    def case(name, h0, Q, N, v, dN, dv, e_st=None, mask_st=None, R2_st=None):
        n = 2 * len(N)
        hm = clocked_csr(h0, Q, N, v); dh = clocked_csr(h0, Q, dN, dv)
        rec = {"name": name, "n_mat": n}
        hb_t = torch.as_tensor(EBE.band_from_csr(h0.tocsr()), device=dev); qb_t = torch.as_tensor(EBE.band_from_csr(Q.tocsr()), device=dev)
        band_h = EBE.band_from_csr(hm); band_t = torch.as_tensor(band_h, device=dev)
        hmT = torch.as_tensor(hm.toarray(), device=dev)
        # ---- dense (the production call)
        torch.cuda.synchronize(); t0 = time.time(); e_d, U_d = torch.linalg.eigh(hmT); torch.cuda.synchronize(); rec["dense_s"] = time.time() - t0
        e_d = e_d.real
        # ---- banded through the engine entry point (first call timed as is: the kernels are compiled by warmup)
        EBE.stats_snapshot(reset=True)
        torch.cuda.synchronize(); t0 = time.time(); e_b, U_b = EBE.eigT_banded(hmT); torch.cuda.synchronize(); rec["banded_s"] = time.time() - t0
        st = EBE.stats_snapshot(reset=True); rec["banded_stats"] = st
        rec["factor_box"] = rec["dense_s"] / rec["banded_s"]
        rec["route"] = "banded" if st["banded"] == 1 else "dense fallback %s" % st["fallback_why"]
        rec["lifted"] = st["banded_all"] == 0 and st["banded"] == 1
        # (1) eigenvalues
        rec["e_max_abs_diff"] = float((e_b - e_d).abs().max()); rec["e_max_rel_diff"] = float(((e_b - e_d).abs() / e_d.abs().clamp_min(1e-300)).max())
        if e_st is not None:
            es = torch.as_tensor(np.sort(np.asarray(e_st, float)), device=dev)
            rec["e_banded_vs_stored"] = float((e_b - es).abs().max()); rec["e_dense_vs_stored"] = float((e_d - es).abs().max())
        # (2) residuals, orthonormality, PH
        for lab, e_, U_ in (("banded", e_b, U_b), ("dense", e_d, U_d)):
            res = torch.linalg.norm(EBE.band_matvec(band_t, U_) - U_ * e_[None, :], dim=0)
            rec["%s_residual_max" % lab] = float(res.max())
            Gm = U_.conj().T @ U_
            Gm.diagonal().sub_(1.0)
            rec["%s_orth_max" % lab] = float(Gm.abs().max()); del Gm
            rec["%s_ph_max" % lab] = float((e_ + e_.flip(0)).abs().max())
        # (3) the sea projector and densities
        nb_, nd_ = e_b < 0, e_d < 0
        rec["n_neg"] = [int(nb_.sum()), int(nd_.sum())]
        Pb = U_b[:, nb_] @ U_b[:, nb_].conj().T; Pb -= U_d[:, nd_] @ U_d[:, nd_].conj().T
        rec["sea_projector_fro"] = float(torch.linalg.norm(Pb)); rec["sea_projector_max"] = float(Pb.abs().max()); del Pb
        rec["sea_dens_h0_rel"] = rel(dens(hb_t, U_b[:, nb_]), dens(hb_t, U_d[:, nd_])); rec["sea_dens_Q_rel"] = rel(dens(qb_t, U_b[:, nb_]), dens(qb_t, U_d[:, nd_]))
        say("%s: n %d route %s (lift %s); banded %.3f s, dense %.3f s: %.1fx; e max |diff| %.1e (rel %.1e)%s; residual max banded %.1e dense %.1e; orth max banded %.1e dense %.1e; PH banded %.1e dense %.1e; n_neg %s; sea projector fro %.1e max %.1e; sea densities h0 %.1e Q %.1e; module stats: residual %.1e ||h||, overlap %.1e, clusters %d, isolated %.4f, guards %.3f s kernels %.3f s post %.3f s" % (
            name, n, rec["route"], rec["lifted"], rec["banded_s"], rec["dense_s"], rec["factor_box"], rec["e_max_abs_diff"], rec["e_max_rel_diff"],
            ("; vs stored banded %.1e dense %.1e" % (rec["e_banded_vs_stored"], rec["e_dense_vs_stored"])) if e_st is not None else "",
            rec["banded_residual_max"], rec["dense_residual_max"], rec["banded_orth_max"], rec["dense_orth_max"], rec["banded_ph_max"], rec["dense_ph_max"], rec["n_neg"],
            rec["sea_projector_fro"], rec["sea_projector_max"], rec["sea_dens_h0_rel"], rec["sea_dens_Q_rel"], st["max_residual_rel"], st["max_overlap"], st["clusters"], st["isolated_min"],
            st["guard_seconds"], st["kernel_seconds"], st["post_seconds"]))
        # (4) the tracked set against the stored reference
        if mask_st is not None and R2_st is not None:
            m = unpack_mask(mask_st, n)
            R2 = torch.as_tensor(R2_st, device=dev)
            n_occ = R2.shape[1]
            wb = ((U_b.conj().T @ R2).abs() ** 2).sum(dim=1).cpu().numpy(); wd = ((U_d.conj().T @ R2).abs() ** 2).sum(dim=1).cpu().numpy()
            sb = np.zeros(n, bool); sb[np.argsort(-wb, kind="stable")[:n_occ]] = True
            sd = np.zeros(n, bool); sd[np.argsort(-wd, kind="stable")[:n_occ]] = True
            rec["tracked_equal_banded_dense"] = bool(np.array_equal(sb, sd)); rec["tracked_banded_equal_stored_mask"] = bool(np.array_equal(sb, m))
            rec["tracked_dense_equal_stored_mask"] = bool(np.array_equal(sd, m))
            rec["min_occupied_weight"] = [float(wb[sb].min()), float(wd[sd].min())]; rec["max_empty_weight"] = [float(wb[~sb].max()), float(wd[~sd].max())]
            Ps = R2 @ R2.conj().T
            Pb = U_b[:, sb] @ U_b[:, sb].conj().T; Pb -= Ps; rec["occ_projector_banded_vs_stored_fro"] = float(torch.linalg.norm(Pb)); del Pb
            Pd = U_d[:, sd] @ U_d[:, sd].conj().T; Pd -= Ps; rec["occ_projector_dense_vs_stored_fro"] = float(torch.linalg.norm(Pd)); del Pd, Ps
            for lab, bt in (("h0", hb_t), ("Q", qb_t)):
                ds_ = dens(bt, R2); db_ = dens(bt, U_b[:, sb]); dd_ = dens(bt, U_d[:, sd])
                rec["occ_dens_%s_banded_vs_stored" % lab] = rel(db_, ds_); rec["occ_dens_%s_dense_vs_stored" % lab] = rel(dd_, ds_); rec["occ_dens_%s_banded_vs_dense" % lab] = rel(db_, dd_)
            say("  stored reference (%d columns, mask %d true): tracked set banded == dense %s, banded == stored mask %s, dense == stored mask %s; min occupied weight %.12f / %.12f, max empty %.1e / %.1e; occupied projector vs R2 R2^H fro banded %.1e dense %.1e; occupied densities vs stored: banded h0 %.1e Q %.1e, dense h0 %.1e Q %.1e; banded vs dense h0 %.1e Q %.1e" % (
                n_occ, int(m.sum()), rec["tracked_equal_banded_dense"], rec["tracked_banded_equal_stored_mask"], rec["tracked_dense_equal_stored_mask"], rec["min_occupied_weight"][0], rec["min_occupied_weight"][1],
                rec["max_empty_weight"][0], rec["max_empty_weight"][1], rec["occ_projector_banded_vs_stored_fro"], rec["occ_projector_dense_vs_stored_fro"],
                rec["occ_dens_h0_banded_vs_stored"], rec["occ_dens_Q_banded_vs_stored"], rec["occ_dens_h0_dense_vs_stored"], rec["occ_dens_Q_dense_vs_stored"],
                rec["occ_dens_h0_banded_vs_dense"], rec["occ_dens_Q_banded_vs_dense"]))
            occ_b, occ_d = sb, sd
            del R2
        else:
            occ_b, occ_d = nb_.cpu().numpy(), nd_.cpu().numpy()
        # (5) the J.v X
        dhb_t = torch.as_tensor(EBE.band_from_csr(dh), device=dev)
        mo_d = torch.as_tensor(occ_d, device=dev); mo_b = torch.as_tensor(occ_b, device=dev)
        Uo = U_d[:, mo_d]; Uu = U_d[:, ~mo_d]; eo = e_d[mo_d]; eu = e_d[~mo_d]
        torch.cuda.synchronize(); t0 = time.time()
        Hm = Uu.conj().T @ EBE.band_matvec(dhb_t, Uo)
        den = eo[None, :] - eu[:, None]
        small = den.abs() < 1e-12
        M = torch.where(small, torch.zeros_like(Hm), Hm / torch.where(small, torch.ones_like(den), den).to(Hm.dtype))
        X = Uu @ M
        de_ref = dens_pair(hb_t, Uo, X); dJ_ref = dens_pair(qb_t, Uo, X)
        torch.cuda.synchronize(); rec["jv_eigenbasis_s"] = time.time() - t0
        rec["jv_dropped_pairs"] = int(small.sum()); del Hm, den, small, X
        for lab, e_, U_, mo_ in (("banded", e_b, U_b, mo_b), ("dense", e_d, U_d, mo_d)):
            Uo_ = U_[:, mo_]; eo_ = e_[mo_]; eu_ = e_[~mo_]
            EBE.stats_snapshot(reset=True)
            torch.cuda.synchronize(); t0 = time.time()
            Y, inf_ = EBE.sternheimer_X(band_h, eo_, eu_, Uo_, -EBE.band_matvec(dhb_t, Uo_))
            if Y is not None:
                de_ = dens_pair(hb_t, Uo_, Y); dJ_ = dens_pair(qb_t, Uo_, Y)
            torch.cuda.synchronize(); rec["jv_sternheimer_%s_s" % lab] = time.time() - t0
            rec["jv_sternheimer_%s_info" % lab] = inf_
            if Y is None:
                rec["jv_%s_deps_rel" % lab] = None; rec["jv_%s_dJ_rel" % lab] = None
                continue
            rec["jv_%s_deps_rel" % lab] = rel(de_, de_ref); rec["jv_%s_dJ_rel" % lab] = rel(dJ_, dJ_ref)
            del Y
        # the unoccupied block on the dense basis: U_u^H Y against M (recomputed)
        Hm = Uu.conj().T @ EBE.band_matvec(dhb_t, Uo); den = eo[None, :] - eu[:, None]; M = Hm / den.to(Hm.dtype)
        Y, _ = EBE.sternheimer_X(band_h, eo, eu, Uo, -EBE.band_matvec(dhb_t, Uo))
        if Y is not None:
            rec["jv_unocc_block_max_abs_diff"] = float((Uu.conj().T @ Y - M).abs().max()); rec["jv_M_max_abs"] = float(M.abs().max())
        del Hm, den, M, Y
        rec["jv_factor_box"] = rec["jv_eigenbasis_s"] / rec["jv_sternheimer_banded_s"]
        say("  J.v (occupied %d, dropped pairs %d): eigenbasis (dense basis) %.3f s, Sternheimer on the banded basis %.3f s (%.1fx), on the dense basis %.3f s; density response rel diff vs the eigenbasis route: banded basis eps %s J %s, dense basis eps %s J %s; unoccupied block max |U_u^H Y - M| %s (|M| max %s); solve residual %s; min occ-unocc gap %s, min occ-occ gap %s" % (
            int(occ_d.sum()), rec["jv_dropped_pairs"], rec["jv_eigenbasis_s"], rec["jv_sternheimer_banded_s"], rec["jv_factor_box"], rec["jv_sternheimer_dense_s"],
            *["%.1e" % q if q is not None else "FALLBACK" for q in (rec["jv_banded_deps_rel"], rec["jv_banded_dJ_rel"], rec["jv_dense_deps_rel"], rec["jv_dense_dJ_rel"])],
            "%.1e" % rec["jv_unocc_block_max_abs_diff"] if rec.get("jv_unocc_block_max_abs_diff") is not None else "-", "%.2e" % rec["jv_M_max_abs"] if rec.get("jv_M_max_abs") is not None else "-",
            "%.1e" % rec["jv_sternheimer_banded_info"].get("solve_residual_rel", float("nan")), "%.2e" % rec["jv_sternheimer_banded_info"].get("min_occ_unocc_gap", float("nan")),
            "%.2e" % rec["jv_sternheimer_banded_info"].get("min_occ_occ_gap", float("nan"))))
        out.append(rec)
        del hmT, U_d, U_b, Uo, Uu
        torch.cuda.empty_cache()

    if a.w2:
        z = np.load(a.w2, allow_pickle=False); zf = np.load(a.w2_fields, allow_pickle=False)
        R = np.asarray(z["R"], float); N = np.asarray(z["N_applied"], float); v = np.asarray(z["v_applied"], float)
        Ti = np.asarray(zf["T"], float); i1 = int(np.argmin(np.abs(Ti - float(z["T"])))); i0 = max(0, i1 - 1)
        dN = np.asarray(zf["N_applied"][i1], float) - np.asarray(zf["N_applied"][i0], float); dv = np.asarray(zf["v_applied"][i1], float) - np.asarray(zf["v_applied"][i0], float)
        say("W2 slice %s: %d sites, T %.4f; perturbation rows %d - %d (max |dN| %.2e |dv| %.2e)" % (os.path.basename(a.w2), len(R), float(z["T"]), i1, i0, np.max(np.abs(dN)), np.max(np.abs(dv))))
        for kap in [int(c) for c in a.channels_w2.split(",")]:
            h0, Q = VF.chain_sparse(R, a.m, float(kap))
            case("w2_kappa%d" % kap, h0, Q, N, v, dN, dv, e_st=z["e_%d" % kap], mask_st=z["mask_%d" % kap], R2_st=z["R2_%d" % kap])
    if a.iso:
        z = np.load(a.iso, allow_pickle=False); zf = np.load(a.iso_fields, allow_pickle=False)
        R = np.asarray(z["R"], float); N = np.asarray(z["N_applied"], float); v = np.asarray(z["v_applied"], float)
        Ti = np.asarray(zf["T"], float); i1 = int(np.argmin(np.abs(Ti - float(z["T"])))); i0 = max(0, i1 - 1)
        dN = np.asarray(zf["N_applied"][i1], float) - np.asarray(zf["N_applied"][i0], float); dv = np.asarray(zf["v_applied"][i1], float) - np.asarray(zf["v_applied"][i0], float)
        say("ISO_128 slice %s: %d sites, T %.4f (fields row %d, applied fields vs the slice %.1e); perturbation rows %d - %d (max |dN| %.2e |dv| %.2e)" % (
            os.path.basename(a.iso), len(R), float(z["T"]), i1, float(np.max(np.abs(np.asarray(zf["N_applied"][i1]) - N))), i1, i0, np.max(np.abs(dN)), np.max(np.abs(dv))))
        for kap in [int(c) for c in a.channels_iso.split(",")]:
            h0, Q = VF.chain_sparse(R, a.m, float(kap))
            case("iso_kappa%d" % kap, h0, Q, N, v, dN, dv, e_st=z["e_%d" % kap], mask_st=z["mask_%d" % kap], R2_st=z["R2_%d" % kap])
    if a.synthetic:
        n_s = a.synthetic
        R = np.arange(n_s, dtype=float) + 1.0; rs = n_s / 32.0
        v = VF.profile(R, rs, "exact"); N = 1.0 - 0.3 * np.exp(-(R / (2 * rs)) ** 2)
        bump = np.exp(-((R - 2 * rs) / rs) ** 2)
        say("synthetic chain: %d sites (the prototype's run_synth16k_v2 profile), perturbation 1e-3 x a Gaussian bump at 2 r_s in N and v" % n_s)
        h0, Q = VF.chain_sparse(R, a.m, 1.0)
        case("synthetic_%d" % n_s, h0, Q, N, v, 1e-3 * bump, 1e-3 * bump)
    with open(os.path.join(a.out_dir, "rd13_percall_v1.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(a.out_dir, "rd13_percall_v1.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)


if __name__ == "__main__":
    main()
