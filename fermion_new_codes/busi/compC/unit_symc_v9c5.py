#!/usr/bin/env python
"""unit_symc_v9c5.py (busi lane, PERF SYM-C1/SYM-C2; reviewer A symmetry audit secs. 1(b), 3): the two options of engine_twofield_v9c5 checked and
timed per call on production matrices: the c64s8L line's light channels (kappa 1, 2 and the guard 3, 2048 sites) at the T 13.00 slice's applied
fields and at the fields npz's record 12.96875 (the stored run behind the H3 segment).

A (SYM-C1, on the device given by --gpu, or the CPU): per channel and endpoint, v9c4's route (eigT(clocked_gpu) then light_sectorize_v36's
rotation) against v9c5's (_symc_eig_light then the patched sectorize): eigenvalues position by position (max |de| over max |e|), the sector
signs position by position, the slice's stored e/sigma/mask at 13.00 (the production labels), the masks of the mu S selection at the slice's mu
and the anchor mask, the occupied projectors' largest entry difference max |P_a - P_b|, the occupied energy sum, the light
densities dens(h, ref) and dens(Q, ref), the residual of the sector-built U with its own Rayleigh quotients and with the positional e (both routes: inside a degenerate block the rotation leaves the sorted e against S-ordered columns, so the positional residual is up to tol_block times the scale in v9c4 too), its orthonormality, the off-sector block; then the per-call walls:
the bare eigh at 2n and at n, and the two routes whole (clocked_gpu + eigh + sectorize), 1 warm-up and --reps timed calls.
B (SYM-C2, the host): for each channel the walk's K(s) = (1 - s) K0 + s K1 with K = h - a S, a the slice's mu at both ends (the mu S walk's form),
at s = 0, 1/4, 1/2, 3/4, 1: the verdicts of pd_banded_v36 on band_upper_v36(Ks @ Ks) full and on the + block, at a ladder of 8 d's from the
gap at s = 0 halving (light_walk_v36's ladder) and at 6 d's around the exact threshold min |spec K_+(s)| (the + block's eigvalsh) times
(1 -+ 1e-2, 1e-6, 1e-10); then per-call walls of Ks @ Ks, band_upper_v36 and pd_banded_v36 full against sector (median of --lreps).
Prints PASS/FAIL lines and a json with every number.

Usage. python unit_symc_v9c5.py STAGE OUT.json [--gpu 0] [--reps 5] [--lreps 200]
"""
import argparse
import json
import os
import statistics
import sys
import time

import numpy as np

SLICE = "<internal>/orbitals_C64_s8_ell8_tau2_twofield_v4L_T0013.00.npz"
FIELDS = "<internal>/collapse_C64_s8_ell8_tau2_twofield_v4L_fields.npz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage"); ap.add_argument("out")
    ap.add_argument("--gpu", type=int, default=None); ap.add_argument("--reps", type=int, default=5); ap.add_argument("--lreps", type=int, default=200)
    a = ap.parse_args()
    sys.path.insert(0, a.stage)
    import engine_twofield_v9c5 as E
    E.C5.backend(a.gpu)
    torch = E.BK["torch"]; dev = E.BK["dev"]
    E._SYMC["tol_block"] = 1e-8
    res = {"device": str(dev), "gpu_name": (torch.cuda.get_device_name(a.gpu) if a.gpu is not None else None), "threads": torch.get_num_threads(),
           "omp": os.environ.get("OMP_NUM_THREADS"), "A": {}, "B": {}}
    fails = []

    def chk(name, ok, detail):
        print("%s %s: %s" % ("PASS" if ok else "FAIL", name, detail), flush=True)
        if not ok:
            fails.append(name)

    def sync():
        if a.gpu is not None:
            torch.cuda.synchronize()

    Z = np.load(SLICE); F = np.load(FIELDS, allow_pickle=True)
    rs = float(Z["rs"]); nsites = int(Z["N_applied"].shape[0])
    TF = np.asarray(F["T"], float) / rs; k0 = int(np.argmin(np.abs(TF - 12.96875)))
    ends = {"13.00": (np.asarray(Z["N_applied"], float), np.asarray(Z["v_applied"], float)),
            "12.96875": (np.asarray(F["N_applied"][k0], float), np.asarray(F["v_applied"][k0], float))}
    sgn_dev, S_host = E.light_S_v36(nsites)
    chans = [E.LightChan(k, nsites, False, sourcing=(k < 3)) for k in (1, 2, 3)]

    def full_route(ch, N, v):
        E._SYMC["light"] = False
        e, U = E.eigT(E.clocked_gpu(ch, N, v))
        sig, nrot, imp = E.light_sectorize_v36(e, U, sgn_dev, 1e-8)
        return e, U, sig, nrot, imp

    def sector_route(ch, N, v):
        E._SYMC["light"] = True
        e, U = E._symc_eig_light(ch, N, v)
        sig, nrot, imp = E.light_sectorize_v36(e, U, sgn_dev, 1e-8)
        E._SYMC["light"] = False
        return e, U, sig, nrot, imp

    # ---------------------------------------------------------------- A: SYM-C1
    for ch in chans:
        k = ch.kappa
        for tname, (N, v) in ends.items():
            ef, Uf, sf, nrf, imf = full_route(ch, N, v)
            es, Us, ss, nrs, ims = sector_route(ch, N, v)
            efn = ef.cpu().numpy(); esn = es.cpu().numpy(); sfn = np.sign(sf.cpu().numpy()); ssn = ss.cpu().numpy()
            scale = float(np.max(np.abs(efn)))
            de = float(np.max(np.abs(efn - esn))) / scale
            nsig = int(np.sum(sfn != ssn))
            M = E.clocked_gpu(ch, N, v)
            MU = M @ Us
            resid_pos = float((MU - Us * es[None, :].to(Us.dtype)).abs().max()) / scale
            q_own = (Us.conj() * MU).sum(dim=0).real
            resid = float((MU - Us * q_own[None, :].to(Us.dtype)).abs().max()) / scale
            MUf = M @ Uf
            resid_pos_full = float((MUf - Uf * ef[None, :].to(Uf.dtype)).abs().max()) / scale
            del MU, MUf
            orth = float((Us.conj().T @ Us - torch.eye(Us.shape[1], dtype=Us.dtype, device=Us.device)).abs().max())
            del M
            r = {"de_rel": de, "sigma_sign_mismatch": nsig, "full_rotated_blocks": nrf, "full_impurity": imf, "sector_rotated_blocks": nrs,
                 "sector_impurity": ims, "sector_sigma_exact": bool(np.all(np.abs(ssn) == 1.0)), "resid_rel_own": resid, "resid_rel_positional_sector": resid_pos, "resid_rel_positional_full": resid_pos_full, "orth_max": orth}
            if tname == "13.00":
                eZ = np.asarray(Z["e_L%d" % k], float); sZ = np.sign(np.asarray(Z["sigma_L%d" % k], float))
                mZ = np.unpackbits(np.asarray(Z["mask_L%d" % k], np.uint8))[: 2 * nsites].astype(bool)
                mu = float(Z["mu_L%d" % k])
                r.update({"vs_slice_de_full": float(np.max(np.abs(efn - eZ))) / scale, "vs_slice_de_sector": float(np.max(np.abs(esn - eZ))) / scale,
                          "vs_slice_sigma_mismatch_full": int(np.sum(sfn != sZ)), "vs_slice_sigma_mismatch_sector": int(np.sum(ssn != sZ))})
                masks = {}
                for nm, (e_, s_) in (("full", (efn, sfn)), ("sector", (esn, ssn))):
                    masks[nm] = E.light_select_v36(e_, s_, mu)[0] if np.isfinite(mu) else mZ
                r["mask_mismatch_full_vs_sector"] = int(np.sum(masks["full"] != masks["sector"]))
                r["mask_mismatch_full_vs_slice"] = int(np.sum(masks["full"] != mZ)); r["mask_mismatch_sector_vs_slice"] = int(np.sum(masks["sector"] != mZ))
                occ = masks["full"]
            else:
                occ = efn < 0.0
                occ_s = esn < 0.0
                r["negative_set_mismatch"] = int(np.sum(occ != occ_s))
            oi = torch.as_tensor(occ, device=Uf.device)
            Rf = Uf[:, oi]; Rs = Us[:, oi]
            W = Rf @ Rf.conj().T - Rs @ Rs.conj().T
            r["projector_maxabs"] = float(W.abs().max())
            Ef = float(efn[occ].sum()); Es = float(esn[occ].sum())
            r["E_occ_rel"] = abs(Ef - Es) / max(abs(Ef), 1e-300)
            df = E.dens(ch.h, Rf); ds = E.dens(ch.h, Rs); qf = E.dens(ch.Q, Rf); qs = E.dens(ch.Q, Rs)
            r["dens_h_rel"] = float(np.max(np.abs(df - ds))) / max(float(np.max(np.abs(df))), 1e-300)
            r["dens_Q_rel"] = float(np.max(np.abs(qf - qs))) / max(float(np.max(np.abs(qf))), 1e-300)
            res["A"]["k%d@%s" % (k, tname)] = r
            del Uf, Us, Rf, Rs, W
            ok = (de <= 1e-13 and nsig == 0 and r["sector_sigma_exact"] and resid <= 1e-13 and resid_pos <= 2e-8 and orth <= 1e-12 and r["projector_maxabs"] <= 1e-12
                  and r["E_occ_rel"] <= 1e-13 and r["dens_h_rel"] <= 1e-11 and r.get("mask_mismatch_full_vs_sector", r.get("negative_set_mismatch", 0)) == 0)
            chk("A k%d at %s" % (k, tname), ok, json.dumps(r))
    print("  stats %s" % json.dumps(E._SYMC_STATS), flush=True)
    # per-call walls (kappa 1 at 13.00)
    ch = chans[0]; N, v = ends["13.00"]
    M = E.clocked_gpu(ch, N, v); n = M.shape[0] // 2
    sc = E._symc_sign_dev(n, M.device)
    Hp = 0.5 * (M[0::2, 0::2] + 1j * (M[0::2, 1::2] * sc[None, :]) - 1j * (sc[:, None] * M[1::2, 0::2]) + (sc[:, None] * M[1::2, 1::2]) * sc[None, :])
    tim = {}
    for nm, fn in (("eigh_2n", lambda: torch.linalg.eigh(M)), ("eigh_n", lambda: torch.linalg.eigh(Hp)),
                   ("route_full", lambda: full_route(ch, N, v)), ("route_sector", lambda: sector_route(ch, N, v))):
        fn(); sync()
        ts = []
        for _ in range(a.reps):
            t0 = time.time(); out = fn(); sync(); ts.append(time.time() - t0); del out
        tim[nm] = {"median_s": statistics.median(ts), "all_s": ts}
    del M, Hp
    tim["speedup_eigh"] = tim["eigh_2n"]["median_s"] / tim["eigh_n"]["median_s"]
    tim["speedup_route"] = tim["route_full"]["median_s"] / tim["route_sector"]["median_s"]
    res["A"]["timing"] = tim
    print("TIME A (%s, %d sites): eigh 2n %.3f s, eigh n %.3f s (%.2fx); route full %.3f s, route sector %.3f s (%.2fx)" % (
        res["gpu_name"] or "cpu", nsites, tim["eigh_2n"]["median_s"], tim["eigh_n"]["median_s"], tim["speedup_eigh"], tim["route_full"]["median_s"],
        tim["route_sector"]["median_s"], tim["speedup_route"]), flush=True)

    # ---------------------------------------------------------------- B: SYM-C2
    nver = 0; nbad = 0; walls = {"KK_full": [], "KK_sector": [], "band_full": [], "band_sector": [], "chol_full": [], "chol_sector": []}
    for ch in chans:
        k = ch.kappa
        mu = float(Z["mu_L%d" % k])
        e1 = np.asarray(Z["e_L%d" % k], float); s1 = np.sign(np.asarray(Z["sigma_L%d" % k], float))
        K0 = (E.clocked_host_sparse(ch, *ends["12.96875"]) - mu * S_host).tocsr()
        K1 = (E.clocked_host_sparse(ch, *ends["13.00"]) - mu * S_host).tocsr()
        dK = (K1 - K0).tocsr()
        K0p, o0 = E._symc_plus_host(K0); K1p, o1 = E._symc_plus_host(K1); dKp = (K1p - K0p).tocsr()
        g1 = float(np.min(np.abs(e1 - mu * s1)))
        rows = []
        for s in (0.0, 0.25, 0.5, 0.75, 1.0):
            Ks = (K0 + s * dK).tocsr(); Kp = (K0p + s * dKp).tocsr()
            ab, u = E.band_upper_v36(Ks @ Ks); abp, up = E.band_upper_v36(Kp @ Kp)
            thr = float(torch.linalg.eigvalsh(torch.as_tensor(Kp.toarray(), dtype=torch.complex128, device=dev)).abs().min())
            ds = [g1 * 0.5 ** q for q in range(8)] + [thr * (1.0 + sgn * dl) for dl in (1e-2, 1e-6, 1e-10) for sgn in (-1.0, 1.0)]
            for d in ds:
                vf = E.pd_banded_v36(ab, d); vs = E.pd_banded_v36(abp, d)
                nver += 1; nbad += int(vf != vs)
                rows.append([s, d, bool(vf), bool(vs)])
            if k == 1:
                for _ in range(max(1, a.lreps // 5)):
                    t0 = time.perf_counter(); KK = Ks @ Ks; walls["KK_full"].append(time.perf_counter() - t0)
                    t0 = time.perf_counter(); KKp = Kp @ Kp; walls["KK_sector"].append(time.perf_counter() - t0)
                    t0 = time.perf_counter(); E.band_upper_v36(KK); walls["band_full"].append(time.perf_counter() - t0)
                    t0 = time.perf_counter(); E.band_upper_v36(KKp); walls["band_sector"].append(time.perf_counter() - t0)
                    t0 = time.perf_counter(); E.pd_banded_v36(ab, 0.5 * thr); walls["chol_full"].append(time.perf_counter() - t0)
                    t0 = time.perf_counter(); E.pd_banded_v36(abp, 0.5 * thr); walls["chol_sector"].append(time.perf_counter() - t0)
        res["B"]["k%d" % k] = {"offblock_K0": o0, "offblock_K1": o1, "u_full": int(u), "u_sector": int(up), "verdicts": rows}
        chk("B k%d ladders" % k, (o0 == 0.0 and o1 == 0.0 and sum(1 for r_ in rows if r_[2] != r_[3]) == 0),
            "off-sector blocks %g %g; bands u %d (full) %d (sector); %d verdicts, %d differ; thresholds tested to 1e-10 relative" % (o0, o1, u, up, len(rows), sum(1 for r_ in rows if r_[2] != r_[3])))
    med = {k_: statistics.median(v_) * 1e3 for k_, v_ in walls.items()}
    res["B"]["timing_ms"] = med
    res["B"]["verdicts_total"] = nver; res["B"]["verdicts_differ"] = nbad
    print("TIME B (host, %s threads): Ks@Ks %.3f -> %.3f ms (%.2fx), band_upper %.3f -> %.3f ms (%.2fx), cholesky_banded %.3f -> %.3f ms (%.2fx); ladder stage %.3f -> %.3f ms (%.2fx)" % (
        os.environ.get("OMP_NUM_THREADS"), med["KK_full"], med["KK_sector"], med["KK_full"] / med["KK_sector"], med["band_full"], med["band_sector"], med["band_full"] / med["band_sector"],
        med["chol_full"], med["chol_sector"], med["chol_full"] / med["chol_sector"], med["KK_full"] + med["band_full"] + med["chol_full"],
        med["KK_sector"] + med["band_sector"] + med["chol_sector"], (med["KK_full"] + med["band_full"] + med["chol_full"]) / (med["KK_sector"] + med["band_sector"] + med["chol_sector"])), flush=True)
    chk("B all ladders", nbad == 0, "%d verdicts, %d differ" % (nver, nbad))
    res["fails"] = fails
    json.dump(res, open(a.out, "w"), indent=1, default=float)
    print("unit_symc_v9c5: %d FAIL; json %s" % (len(fails), a.out), flush=True)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
