#!/usr/bin/env python
"""rd1c_percall_v1.py (busi, analyst; PERF RD-1c self-test part (b)): RD-1's module engine_banded_eigh_v1 per call on the twofield HEAVY
chains against the dense route (torch.linalg.eigh, engine_collapse_v5.eigT's code), RD-1's rd13_percall_v1 without the J.v part.
The input of eigT_banded is the dense matrix the engine builds (engine_collapse_v36.clocked_gpu on engine_collapse_v5.Chan, the twofield's
heavy channel), so the band extraction and every guard run as in the engine.
Cases: (1) the c64s8L slice (its R, N_applied, v_applied; per heavy channel its stored e, mask and R2); (2) stored records of a run (the
fields npz row's N_applied, v_applied; that run's spectra row e_k and mask_k).
Per matrix: route (banded with the lift, or the fallback named); eigenvalues banded against dense (max abs, relative to ||h||_G); residuals
and orthonormality of both; the sea projector (e < 0) Frobenius difference; the sea densities of h0 and Q (relative); with R2: the
max-overlap tracked set from each basis against each other and against the stored mask, min occupied / max empty weight, the occupied
projector against R2 R2^H and the occupied densities against R2's; at a record: both routes against the stored spectra row, the e < 0 set of
both routes, the stored mask against each route's negative set; the seconds of both routes.
    python rd1c_percall_v1.py OUT_DIR --slice SLICE [--channels 1,...,10] [--run-dir DIR --tag TAG --records 13.34375,13.625,14.0
        --record-channels 1,5,10 --rs 64] [--gpu 0]
The engine modules come from the directory of this script and BUSI_ENGINE_DIR (engine_collapse_v36 and its imports).
Writes OUT_DIR/rd1c_percall_v1.json and .txt, and the budget lines of busi/notes/PERF_RD1c_build_v1.md (b) with PASS or FAIL each."""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch


def unpack_mask(m, n):
    return np.unpackbits(np.asarray(m, np.uint8))[:n].astype(bool)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir"); ap.add_argument("--slice", required=True); ap.add_argument("--channels", default="1,2,3,4,5,6,7,8,9,10")
    ap.add_argument("--run-dir", default=None); ap.add_argument("--tag", default="C64_s8_ell8_tau2_twofield_v4L")
    ap.add_argument("--records", default="13.34375,13.625,14.0"); ap.add_argument("--record-channels", default="1,5,10"); ap.add_argument("--rs", type=float, default=64.0)
    ap.add_argument("--m", type=float, default=0.2); ap.add_argument("--gpu", type=int, default=0)
    a = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)
    for d in (os.environ.get("BUSI_ENGINE_DIR"), os.environ.get("BUSI_PAGE_DIR")):
        if d and d not in sys.path:
            sys.path.append(d)
    import engine_collapse_v5 as C5
    C5.backend(a.gpu)
    import engine_collapse_v36 as V35
    import engine_banded_eigh_v1 as EBE
    os.makedirs(a.out_dir, exist_ok=True)
    dev = V35.BK["dev"]
    lines = []; out = []

    def say(s):
        lines.append(s); print(s, flush=True)
    tw = EBE.warmup(dev)
    md5m = hashlib.md5(open(EBE.__file__, "rb").read()).hexdigest()
    say("rd1c_percall_v1: %s (%s, md5 %s); GPU %s; torch %s, numba %s; warmup (JIT) %.1f s; driver patch: %s" % (
        EBE.VERSION, EBE.__file__, md5m, torch.cuda.get_device_name(dev), torch.__version__, EBE.numba.__version__, tw, EBE.STATS.get("devnum_patch")))

    def dens_band(band_t, Phi):
        AP = EBE.band_matvec(band_t, Phi)
        return (Phi.conj() * AP).real.sum(dim=1).reshape(-1, 2).sum(dim=1).cpu().numpy()

    def rel(x, ref):
        return float(np.max(np.abs(np.asarray(x) - np.asarray(ref)))) / max(float(np.max(np.abs(ref))), 1e-300)

    chans = {}

    def chan(k):
        if k not in chans:
            chans[k] = V35.Chan(k, a.m, n_sites, False)
        return chans[k]

    def case(name, ch, N, v, e_st=None, mask_st=None, R2_st=None):
        hmT = V35.clocked_gpu(ch, N, v)
        n = hmT.shape[0]
        rec = {"name": name, "kappa": int(ch.kappa), "n_mat": int(n)}
        hb_t = torch.as_tensor(EBE.band_from_csr(ch.hs.tocsr()), device=dev); qb_t = torch.as_tensor(EBE.band_from_csr(ch.Qs.tocsr()), device=dev)
        band_t = torch.as_tensor(EBE.band_from_dense_lower(hmT), device=dev)
        G = EBE.gershgorin_band(band_t.cpu().numpy()); rec["norm_G"] = G
        torch.cuda.synchronize(); t0 = time.time(); e_d, U_d = torch.linalg.eigh(hmT); torch.cuda.synchronize(); rec["dense_s"] = time.time() - t0
        e_d = e_d.real
        EBE.stats_snapshot(reset=True)
        torch.cuda.synchronize(); t0 = time.time(); e_b, U_b = EBE.eigT_banded(hmT); torch.cuda.synchronize(); rec["banded_s"] = time.time() - t0
        st = EBE.stats_snapshot(reset=True); rec["banded_stats"] = {k: v for k, v in st.items() if not isinstance(v, dict)}; rec["fallback_why"] = st["fallback_why"]
        rec["factor"] = rec["dense_s"] / rec["banded_s"]
        rec["route_banded"] = bool(st["banded"] == 1); rec["lifted"] = bool(st["banded"] == 1 and st["banded_all"] == 0)
        rec["e_max_abs_diff"] = float((e_b - e_d).abs().max()); rec["e_rel_G"] = rec["e_max_abs_diff"] / G
        if e_st is not None:
            es = torch.as_tensor(np.sort(np.asarray(e_st, float)), device=dev)
            rec["e_banded_vs_stored"] = float((e_b - es).abs().max()); rec["e_dense_vs_stored"] = float((e_d - es).abs().max())
        for lab, e_, U_ in (("banded", e_b, U_b), ("dense", e_d, U_d)):
            res = torch.linalg.norm(EBE.band_matvec(band_t, U_) - U_ * e_[None, :], dim=0)
            rec["%s_residual_rel_G" % lab] = float(res.max()) / G
            Gm = U_.conj().T @ U_
            Gm.diagonal().sub_(1.0)
            rec["%s_orth_max" % lab] = float(Gm.abs().max()); del Gm
        nb_, nd_ = e_b < 0, e_d < 0
        rec["n_neg"] = [int(nb_.sum()), int(nd_.sum())]; rec["neg_set_equal"] = bool(torch.equal(nb_, nd_))
        Pb = U_b[:, nb_] @ U_b[:, nb_].conj().T; Pb -= U_d[:, nd_] @ U_d[:, nd_].conj().T
        rec["sea_projector_fro"] = float(torch.linalg.norm(Pb)); del Pb
        rec["sea_dens_h0_rel"] = rel(dens_band(hb_t, U_b[:, nb_]), dens_band(hb_t, U_d[:, nd_])); rec["sea_dens_Q_rel"] = rel(dens_band(qb_t, U_b[:, nb_]), dens_band(qb_t, U_d[:, nd_]))
        if mask_st is not None:
            m = unpack_mask(mask_st, n)
            rec["stored_mask_true"] = int(m.sum())
            rec["stored_mask_eq_neg_banded"] = bool(np.array_equal(m, nb_.cpu().numpy())); rec["stored_mask_eq_neg_dense"] = bool(np.array_equal(m, nd_.cpu().numpy()))
        s = ("%s: n %d route %s (lift %s); banded %.3f s, dense %.3f s: %.2fx; e max |d| %.1e (%.1e ||h||_G)%s; residual banded %.1e dense %.1e ||h||_G; orth banded %.1e dense %.1e; "
             "n_neg %s (equal %s); sea projector fro %.1e; sea densities h0 %.1e Q %.1e%s" % (
                 name, n, "banded" if rec["route_banded"] else "DENSE FALLBACK %s" % st["fallback_why"], rec["lifted"], rec["banded_s"], rec["dense_s"], rec["factor"],
                 rec["e_max_abs_diff"], rec["e_rel_G"], ("; vs stored banded %.1e dense %.1e" % (rec["e_banded_vs_stored"], rec["e_dense_vs_stored"])) if e_st is not None else "",
                 rec["banded_residual_rel_G"], rec["dense_residual_rel_G"], rec["banded_orth_max"], rec["dense_orth_max"], rec["n_neg"], rec["neg_set_equal"],
                 rec["sea_projector_fro"], rec["sea_dens_h0_rel"], rec["sea_dens_Q_rel"],
                 ("; stored mask (%d true) == negative set: banded %s dense %s" % (rec["stored_mask_true"], rec["stored_mask_eq_neg_banded"], rec["stored_mask_eq_neg_dense"])) if mask_st is not None else ""))
        say(s)
        if mask_st is not None and R2_st is not None:
            m = unpack_mask(mask_st, n)
            R2 = torch.as_tensor(np.asarray(R2_st), dtype=torch.complex128, device=dev)
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
                ds_ = dens_band(bt, R2); db_ = dens_band(bt, U_b[:, sb]); dd_ = dens_band(bt, U_d[:, sd])
                rec["occ_dens_%s_banded_vs_stored" % lab] = rel(db_, ds_); rec["occ_dens_%s_dense_vs_stored" % lab] = rel(dd_, ds_); rec["occ_dens_%s_banded_vs_dense" % lab] = rel(db_, dd_)
            say("  stored R2 (%d columns, mask %d true): tracked set banded == dense %s, banded == stored mask %s, dense == stored mask %s; min occupied weight %.12f / %.12f, max empty %.1e / %.1e; "
                "occupied projector vs R2 R2^H fro banded %.1e dense %.1e; occupied densities vs stored: banded h0 %.1e Q %.1e, dense h0 %.1e Q %.1e; banded vs dense h0 %.1e Q %.1e" % (
                    n_occ, int(m.sum()), rec["tracked_equal_banded_dense"], rec["tracked_banded_equal_stored_mask"], rec["tracked_dense_equal_stored_mask"], rec["min_occupied_weight"][0],
                    rec["min_occupied_weight"][1], rec["max_empty_weight"][0], rec["max_empty_weight"][1], rec["occ_projector_banded_vs_stored_fro"], rec["occ_projector_dense_vs_stored_fro"],
                    rec["occ_dens_h0_banded_vs_stored"], rec["occ_dens_Q_banded_vs_stored"], rec["occ_dens_h0_dense_vs_stored"], rec["occ_dens_Q_dense_vs_stored"],
                    rec["occ_dens_h0_banded_vs_dense"], rec["occ_dens_Q_banded_vs_dense"]))
            del R2
        out.append(rec)
        del hmT, U_d, U_b
        torch.cuda.empty_cache()

    z = np.load(a.slice, allow_pickle=False)
    R = np.asarray(z["R"], float); n_sites = len(R)
    if abs(R[0] - 1.0) > 1e-12:
        raise SystemExit("the slice's R starts at %.3f; this script builds the half_offset 0 chain" % R[0])
    N = np.asarray(z["N_applied"], float); v = np.asarray(z["v_applied"], float)
    say("slice %s: %d sites, T %.4f" % (os.path.basename(a.slice), n_sites, float(z["T"])))
    for k in [int(c) for c in a.channels.split(",")]:
        case("slice_kappa%d" % k, chan(k), N, v, e_st=z["e_%d" % k], mask_st=z["mask_%d" % k], R2_st=z["R2_%d" % k])
    del z
    if a.run_dir:
        zf = np.load(os.path.join(a.run_dir, "collapse_%s_fields.npz" % a.tag), allow_pickle=True)
        zs = np.load(os.path.join(a.run_dir, "collapse_%s_spectra.npz" % a.tag), allow_pickle=True)
        Tf = np.asarray(zf["T"], float); Ts = np.asarray(zs["T"], float)
        for trs in [float(x) for x in a.records.split(",")]:
            T = trs * a.rs; i = int(np.argmin(np.abs(Tf - T))); j = int(np.argmin(np.abs(Ts - T)))
            if abs(Tf[i] - T) > 1e-9 or abs(Ts[j] - T) > 1e-9:
                say("record %.5f: no fields or spectra row; skipped" % trs); continue
            Nr = np.asarray(zf["N_applied"][i], float); vr = np.asarray(zf["v_applied"][i], float)
            say("record T/rs %.5f of %s (fields row %d, spectra row %d)" % (trs, a.run_dir, i, j))
            for k in [int(c) for c in a.record_channels.split(",")]:
                case("rec%.5f_kappa%d" % (trs, k), chan(k), Nr, vr, e_st=zs["e_%d" % k][j], mask_st=zs["mask_%d" % k][j])

    # the budget lines of the notes, (b)
    def line(txt, ok):
        say("%s %s" % ("PASS" if ok else "FAIL", txt)); return ok
    say("== the declared budget (b)")
    oks = []
    oks.append(line("route banded with the lift, 0 dense fallbacks, on all %d matrices" % len(out), all(r["route_banded"] and r["lifted"] for r in out)))
    oks.append(line("eigenvalues max |d| <= 1e-12 ||h||_G (max %.2e)" % max(r["e_rel_G"] for r in out), all(r["e_rel_G"] <= 1e-12 for r in out)))
    oks.append(line("residuals <= 1e-12 ||h||_G (banded max %.2e; dense max %.2e)" % (max(r["banded_residual_rel_G"] for r in out), max(r["dense_residual_rel_G"] for r in out)),
                    all(r["banded_residual_rel_G"] <= 1e-12 for r in out)))
    oks.append(line("orthonormality <= 1e-11 (banded max %.2e; dense max %.2e)" % (max(r["banded_orth_max"] for r in out), max(r["dense_orth_max"] for r in out)), all(r["banded_orth_max"] <= 1e-11 for r in out)))
    oks.append(line("sea projector Frobenius <= 1e-10 (max %.2e)" % max(r["sea_projector_fro"] for r in out), all(r["sea_projector_fro"] <= 1e-10 for r in out)))
    oks.append(line("sea densities h0, Q relative <= 1e-11 (max %.2e, %.2e)" % (max(r["sea_dens_h0_rel"] for r in out), max(r["sea_dens_Q_rel"] for r in out)),
                    all(r["sea_dens_h0_rel"] <= 1e-11 and r["sea_dens_Q_rel"] <= 1e-11 for r in out)))
    wr = [r for r in out if "tracked_equal_banded_dense" in r]
    oks.append(line("tracked set == dense route's == stored mask on all %d channels with R2" % len(wr), bool(wr) and all(r["tracked_equal_banded_dense"] and r["tracked_banded_equal_stored_mask"] for r in wr)))
    ok_p = all(r["occ_projector_banded_vs_stored_fro"] <= max(10 * r["occ_projector_dense_vs_stored_fro"], 1e-11) for r in wr)
    oks.append(line("occupied projector vs R2 R2^H within 10x the dense route's or below 1e-11 (banded max %.2e, dense max %.2e)" % (
        max(r["occ_projector_banded_vs_stored_fro"] for r in wr), max(r["occ_projector_dense_vs_stored_fro"] for r in wr)), bool(wr) and ok_p))
    ok_d = all(r["occ_dens_%s_banded_vs_stored" % l] <= max(10 * r["occ_dens_%s_dense_vs_stored" % l], 1e-11) for r in wr for l in ("h0", "Q"))
    oks.append(line("occupied densities vs stored within 10x the dense route's or below 1e-11 (banded max h0 %.2e Q %.2e; dense max h0 %.2e Q %.2e)" % (
        max(r["occ_dens_h0_banded_vs_stored"] for r in wr), max(r["occ_dens_Q_banded_vs_stored"] for r in wr), max(r["occ_dens_h0_dense_vs_stored"] for r in wr),
        max(r["occ_dens_Q_dense_vs_stored"] for r in wr)), bool(wr) and ok_d))
    rr = [r for r in out if r["name"].startswith("rec")]
    if a.run_dir:
        oks.append(line("at the %d record matrices the e < 0 set equal on both routes" % len(rr), bool(rr) and all(r["neg_set_equal"] for r in rr)))
    say("reported: factor (dense s / banded s, this GPU) %.2f to %.2f; at the records both routes vs the stored spectra row max %.2e / %.2e; stored mask == negative set banded %s dense %s" % (
        min(r["factor"] for r in out), max(r["factor"] for r in out), max([r.get("e_banded_vs_stored", 0.0) for r in rr] or [0.0]), max([r.get("e_dense_vs_stored", 0.0) for r in rr] or [0.0]),
        [r.get("stored_mask_eq_neg_banded") for r in rr], [r.get("stored_mask_eq_neg_dense") for r in rr]))
    say("rd1c_percall_v1 (b): %s (%d of %d lines)" % ("PASS" if all(oks) else "FAIL", sum(oks), len(oks)))
    with open(os.path.join(a.out_dir, "rd1c_percall_v1.json"), "w") as f:
        json.dump({"module_md5": md5m, "gpu": torch.cuda.get_device_name(dev), "cases": out, "verdict": "PASS" if all(oks) else "FAIL"}, f, indent=1, default=float)
    with open(os.path.join(a.out_dir, "rd1c_percall_v1.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
