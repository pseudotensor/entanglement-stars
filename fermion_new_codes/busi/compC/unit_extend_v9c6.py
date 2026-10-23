#!/usr/bin/env python
"""unit_extend_v9c6.py (busi lane, PERF v9c5b item in engine_twofield_v9c6: --perf-sym-extend-sector): the unit part the tolerance ruling
(REVIEWA_perf_tolerance_ruling_v1 sec. 3.2 (3)) asks of a build covering V9cSpectra._extend: per call, the eigenvalues of the extended chord by
the sector route against the full route at 1e-14 relative, and sigma exactly +-1 with 0 sign mismatches. The c64s8L light channels (kappa 1,
2, guard 3; 2048 sites); the chord from the fields npz's record 12.96875 to the T 13.00 slice's applied fields (the stored run behind the H3
segment), extended to s = -0.5, -0.25, 1.25, 1.5 and 2.0 (the primitive extends by one piece; the larger s are a margin). Both routes with
SYM-C1 on (the arms' configuration): the full route is _extend with the option off (eigT(clocked_gpu) then light_sectorize_v36, which keeps
v9c4's rotation on a full-eigh basis), the sector route with it on (_symc_eig_light then the sectorize short cut). Also the occupied projector
of the anchor-free negative set and the light densities dens(h), dens(Q) on it, as in unit_symc_v9c5.
Usage. python unit_extend_v9c6.py STAGE OUT.json [--gpu 0]
"""
import argparse
import json
import sys

import numpy as np

SLICE = "<internal>/orbitals_C64_s8_ell8_tau2_twofield_v4L_T0013.00.npz"
FIELDS = "<internal>/collapse_C64_s8_ell8_tau2_twofield_v4L_fields.npz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage"); ap.add_argument("out"); ap.add_argument("--gpu", type=int, default=None)
    a = ap.parse_args()
    sys.path.insert(0, a.stage)
    import engine_twofield_v9c6 as E
    E.C5.backend(a.gpu)
    torch = E.BK["torch"]
    E._SYMC.update({"light": True, "ladder": True, "tol_block": 1e-8})
    Z = np.load(SLICE); F = np.load(FIELDS, allow_pickle=True)
    rs = float(Z["rs"]); nsites = int(Z["N_applied"].shape[0])
    TF = np.asarray(F["T"], float) / rs; k0 = int(np.argmin(np.abs(TF - 12.96875)))
    f0 = (np.asarray(F["N_applied"][k0], float), np.asarray(F["v_applied"][k0], float)); f1 = (np.asarray(Z["N_applied"], float), np.asarray(Z["v_applied"], float))
    sgn_dev, S_host = E.light_S_v36(nsites)
    res = {"rows": []}; fails = []
    for k in (1, 2, 3):
        ch = E.LightChan(k, nsites, False, sourcing=(k < 3))
        path = E.LightPathV9(ch, f0, f1, sgn_dev, 1e-8, None)
        sp = object.__new__(E.V9cSpectra); sp.path = path; sp.light = True
        for s in (-0.5, -0.25, 1.25, 1.5, 2.0):
            E._SYMC["extend"] = False
            ef, sf, Uf = sp._extend(s)
            E._SYMC["extend"] = True
            es, ss, Us = sp._extend(s)
            E._SYMC["extend"] = False
            scale = float(np.max(np.abs(ef))); de = float(np.max(np.abs(ef - es))) / scale
            nsig = int(np.sum(np.sign(sf) != np.sign(ss))); exact = bool(np.all(np.abs(ss) == 1.0))
            occ = ef < 0.0; occ_s = es < 0.0
            oi = torch.as_tensor(occ, device=Uf.device)
            Rf = Uf[:, oi]; Rs = Us[:, oi]
            P = float((Rf @ Rf.conj().T - Rs @ Rs.conj().T).abs().max())
            dh = float(np.max(np.abs(E.dens(ch.h, Rf) - E.dens(ch.h, Rs)))) / max(float(np.max(np.abs(E.dens(ch.h, Rf)))), 1e-300)
            row = {"kappa": k, "s": s, "de_rel": de, "sigma_sign_mismatch": nsig, "sector_sigma_exact": exact, "negative_set_mismatch": int(np.sum(occ != occ_s)),
                   "projector_maxabs": P, "dens_h_rel": dh}
            ok = de <= 1e-14 and nsig == 0 and exact and row["negative_set_mismatch"] == 0 and P <= 1e-12
            print("%s kappa %d s %+.2f: eigenvalues %.2e relative, sigma mismatches %d, sigma exactly +-1 %s, negative set mismatches %d, projector %.1e, dens(h) %.1e" % (
                "PASS" if ok else "FAIL", k, s, de, nsig, exact, row["negative_set_mismatch"], P, dh), flush=True)
            if not ok:
                fails.append(row)
            res["rows"].append(row)
            del Uf, Us, Rf, Rs
    res["fails"] = fails
    json.dump(res, open(a.out, "w"), indent=1, default=float)
    print("unit_extend_v9c6: %d FAIL of %d" % (len(fails), len(res["rows"])), flush=True)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
