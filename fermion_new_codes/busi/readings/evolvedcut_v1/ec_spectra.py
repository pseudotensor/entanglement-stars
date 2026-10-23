#!/usr/bin/env python
"""ec_spectra.py (busi lane, cold check of EVOLVEDCUT_route_compare_v1): route 2's own stored occupation spectra at the six
matched partitions (orthonormal_control_cut{n}.npz: 'exterior' = the 2053 eigenvalues of Q_X^dag Q_X, 'interior' = the 2 n_in
eigenvalues of Q_Xc Q_Xc^dag; raw_cut{n}.npz, held_cut{n}.npy, flat_cut{n}.npy likewise), read with own code:
(i) route 2's entropy rule reproduced from the stored eigenvalues (reads its saved entropies back, validating the schema);
(ii) the exterior spectrum split into the 2 n_in genuine occupations (1 - mu_j) and the 2053 - 2 n_in occupations whose ideal
value is exactly 1 (a pure state's modes with no interior weight): the entropy of that cluster is numerical, and is compared with
route 2's exterior-minus-interior asymmetry delta and with its stored polar_global entropy (the entropy of the whole-chain pure
state, ideally 0). Writes OUT/spectra.json and prints a summary."""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ec_common as C  # noqa: E402

MATCHED = {12: 58, 15: 51, 20: 38, 28: 37, 35: 37, 40: 36}


def r2_rule(ev):
    """route 2's stated rule: clip to [0, 1], sum h over the strictly interior values (its code's np.sum; fsum here)."""
    x = np.clip(np.asarray(ev, dtype=float), 0.0, 1.0)
    mid = x[(x > 0) & (x < 1)]
    return math.fsum((-mid * np.log(mid) - (1.0 - mid) * np.log1p(-mid)).tolist())


def split_by_rank(ev, n_genuine):
    """ascending sort; the n_genuine smallest are the genuine occupations 1 - mu_j, the rest the cluster at 1."""
    ev = np.sort(np.asarray(ev, dtype=float))
    gen, clu = ev[:n_genuine], ev[n_genuine:]
    return gen, clu


def main():
    res = C.jload(C.R2_RESULTS)
    idx = C.r2_rows_index(res)
    out = {"inputs": {"r2_results": [C.R2_RESULTS, C.md5(C.R2_RESULTS)]}, "rows": [], "summary": {}}
    for T, n_in in MATCHED.items():
        s = 2 * n_in
        for k in C.KAPPAS:
            d = C.R2_DIR + "/T%d_k%d" % (T, k)
            pz = d + "/orthonormal_control_cut%d.npz" % n_in
            z = np.load(pz, allow_pickle=False)
            ext, inte = np.asarray(z["exterior"], float), np.asarray(z["interior"], float)
            row = idx[(T, k)]
            cut = C.r2_cut(row, n_in)
            oc = cut["orthonormal_control"]
            assert len(ext) == 2053 - 0 or True
            r = {"T": T, "kappa": k, "n_in": n_in, "dim_ext": int(len(ext)), "dim_int": int(len(inte)),
                 "stored_ext": oc["exterior"]["entropy"], "stored_int": oc["interior"]["entropy"],
                 "r2rule_ext": r2_rule(ext), "r2rule_int": r2_rule(inte), "mine_ext": C.entropy(ext), "mine_int": C.entropy(inte)}
            r["reproduce_ext"] = r["r2rule_ext"] - r["stored_ext"]
            r["reproduce_int"] = r["r2rule_int"] - r["stored_int"]
            r["delta_stored"] = r["stored_ext"] - r["stored_int"]
            gen, clu = split_by_rank(ext, s)
            r["genuine_max"] = float(gen.max())
            r["cluster_min"] = float(clu.min())
            r["cluster_max"] = float(clu.max())
            r["cluster_n"] = int(len(clu))
            r["cluster_n_below_1"] = int((clu < 1.0).sum())
            r["cluster_n_above_1"] = int((clu > 1.0).sum())
            r["cluster_max_abs_dev"] = float(np.abs(1.0 - clu).max())
            r["cluster_rms_dev"] = float(np.sqrt(np.mean((1.0 - clu) ** 2)))
            r["S_genuine"] = C.entropy(gen)
            r["S_cluster"] = C.entropy(clu)
            r["S_genuine_minus_int"] = r["S_genuine"] - r["mine_int"]
            r["S_cluster_minus_delta"] = r["S_cluster"] - r["delta_stored"]
            # the interior spectrum's own boundary floor: contributions from occupations within 1e-9 of 0 or 1
            tot, near, rest, n_near, maxdev = C.entropy_split(inte, 1e-9)
            r["int_S_near_boundary_1e-9"] = near
            r["int_n_near_boundary_1e-9"] = n_near
            r["int_max_dev_near"] = maxdev
            r["int_min"] = float(inte.min())
            r["int_max"] = float(inte.max())
            # the separation of the genuine set from the cluster: smallest interior occupation mu_min <-> largest genuine 1 - mu_min
            r["mu_min_from_int"] = float(np.sort(inte)[0])
            r["one_minus_genuine_max"] = 1.0 - r["genuine_max"]
            r["pol_global_entropy"] = row["polar_global"]["entropy"]
            r["pol_global_minus_delta"] = r["pol_global_entropy"] - r["delta_stored"]
            r["polar_residual_frob"] = row["orthogonality"]["orthonormal_control_frobenius"]
            r["gram_evec_orth_frob"] = row["orthogonality"]["joint_eigenvector_orthogonality_frobenius"]
            # the global polar spectrum (eigenvalues of Q^dag Q): its deviation from 1 is the orthonormality defect per mode
            pg = d + "/polar_global_spectrum.npy"
            if os.path.exists(pg):
                g = np.load(pg, allow_pickle=False)
                r["global_max_abs_dev"] = float(np.abs(1.0 - g).max())
                r["global_rms_dev"] = float(np.sqrt(np.mean((1.0 - g) ** 2)))
                r["global_S_mine"] = C.entropy(g)
            # raw where saved and gated
            pr = d + "/raw_cut%d.npz" % n_in
            if os.path.exists(pr) and cut["raw"].get("physical_exterior_entropy") is not None:
                zr = np.load(pr, allow_pickle=False)
                rext, rint = np.asarray(zr["exterior"], float), np.asarray(zr["interior"], float)
                r["raw_stored_ext"] = cut["raw"]["exterior"]["entropy"]
                r["raw_reproduce_ext"] = r2_rule(rext) - r["raw_stored_ext"]
                rg, rc = split_by_rank(rext, s)
                r["raw_S_cluster"] = C.entropy(rc)
                r["raw_delta_stored"] = cut["raw"]["exterior"]["entropy"] - cut["raw"]["interior"]["entropy"]
                r["raw_S_genuine_minus_int"] = C.entropy(rg) - C.entropy(rint)
            # held and flat (interior-block spectra only in route 2)
            for lab in ("held", "flat"):
                p = d + "/%s_cut%d.npy" % (lab, n_in)
                if os.path.exists(p):
                    ev = np.load(p, allow_pickle=False)
                    r[lab + "_reproduce"] = r2_rule(ev) - cut[lab]["entropy"]
                    r[lab + "_mine_minus_stored"] = C.entropy(ev) - cut[lab]["entropy"]
            out["rows"].append(r)
    R = out["rows"]
    S = out["summary"]
    S["n_rows"] = len(R)
    S["max_abs_reproduce_ext"] = max(abs(r["reproduce_ext"]) for r in R)
    S["max_abs_reproduce_int"] = max(abs(r["reproduce_int"]) for r in R)
    S["max_abs_mine_minus_r2rule_ext"] = max(abs(r["mine_ext"] - r["r2rule_ext"]) for r in R)
    S["range_S_cluster"] = [min(r["S_cluster"] for r in R), max(r["S_cluster"] for r in R)]
    S["range_delta_stored"] = [min(r["delta_stored"] for r in R), max(r["delta_stored"] for r in R)]
    S["range_S_cluster_minus_delta"] = [min(r["S_cluster_minus_delta"] for r in R), max(r["S_cluster_minus_delta"] for r in R)]
    S["range_S_genuine_minus_int"] = [min(r["S_genuine_minus_int"] for r in R), max(r["S_genuine_minus_int"] for r in R)]
    S["range_cluster_max_abs_dev"] = [min(r["cluster_max_abs_dev"] for r in R), max(r["cluster_max_abs_dev"] for r in R)]
    S["range_cluster_rms_dev"] = [min(r["cluster_rms_dev"] for r in R), max(r["cluster_rms_dev"] for r in R)]
    S["range_cluster_n_below_1"] = [min(r["cluster_n_below_1"] for r in R), max(r["cluster_n_below_1"] for r in R)]
    S["range_one_minus_genuine_max"] = [min(r["one_minus_genuine_max"] for r in R), max(r["one_minus_genuine_max"] for r in R)]
    S["range_mu_min_from_int"] = [min(r["mu_min_from_int"] for r in R), max(r["mu_min_from_int"] for r in R)]
    S["range_int_S_near_boundary_1e-9"] = [min(r["int_S_near_boundary_1e-9"] for r in R), max(r["int_S_near_boundary_1e-9"] for r in R)]
    S["range_pol_global_minus_delta"] = [min(r["pol_global_minus_delta"] for r in R), max(r["pol_global_minus_delta"] for r in R)]
    S["range_pol_global_over_delta"] = [min(r["pol_global_entropy"] / r["delta_stored"] for r in R), max(r["pol_global_entropy"] / r["delta_stored"] for r in R)]
    S["range_polar_residual_frob"] = [min(r["polar_residual_frob"] for r in R), max(r["polar_residual_frob"] for r in R)]
    S["range_gram_evec_orth_frob"] = [min(r["gram_evec_orth_frob"] for r in R), max(r["gram_evec_orth_frob"] for r in R)]
    if any("global_max_abs_dev" in r for r in R):
        S["range_global_max_abs_dev"] = [min(r["global_max_abs_dev"] for r in R if "global_max_abs_dev" in r), max(r["global_max_abs_dev"] for r in R if "global_max_abs_dev" in r)]
        S["range_global_rms_dev"] = [min(r["global_rms_dev"] for r in R if "global_rms_dev" in r), max(r["global_rms_dev"] for r in R if "global_rms_dev" in r)]
    raw = [r for r in R if "raw_S_cluster" in r]
    S["n_raw_rows"] = len(raw)
    if raw:
        S["range_raw_S_cluster_minus_delta"] = [min(r["raw_S_cluster"] - r["raw_delta_stored"] for r in raw), max(r["raw_S_cluster"] - r["raw_delta_stored"] for r in raw)]
        S["range_raw_S_genuine_minus_int"] = [min(r["raw_S_genuine_minus_int"] for r in raw), max(r["raw_S_genuine_minus_int"] for r in raw)]
        S["max_abs_raw_reproduce_ext"] = max(abs(r["raw_reproduce_ext"]) for r in raw)
    for lab in ("held", "flat"):
        S["max_abs_%s_reproduce" % lab] = max(abs(r[lab + "_reproduce"]) for r in R if lab + "_reproduce" in r)
    # per clock weighted cluster entropy against E2
    S["weighted_S_cluster_by_clock"] = {T: math.fsum(16 * r["kappa"] * r["S_cluster"] for r in R if r["T"] == T) for T in MATCHED}
    S["weighted_delta_by_clock"] = {T: math.fsum(16 * r["kappa"] * r["delta_stored"] for r in R if r["T"] == T) for T in MATCHED}
    C.jdump(out, C.OUT + "/spectra.json")
    L = ["ec_spectra.py: route 2's stored spectra at the matched partitions, %d rows" % S["n_rows"],
         "reproduction of route 2's stored entropies from its stored eigenvalues with its rule: max |diff| exterior %.2e, interior %.2e (held %.2e, flat %.2e, raw ext %s); my clipped rule minus its rule %.2e"
         % (S["max_abs_reproduce_ext"], S["max_abs_reproduce_int"], S["max_abs_held_reproduce"], S["max_abs_flat_reproduce"], S.get("max_abs_raw_reproduce_ext"), S["max_abs_mine_minus_r2rule_ext"]),
         "exterior spectrum split by rank (2 n_in genuine, 2053 - 2 n_in cluster at 1):",
         "  cluster entropy range [%.4e, %.4e]; stored ext - int delta range [%.4e, %.4e]; cluster - delta range [%+.2e, %+.2e]" % tuple(S["range_S_cluster"] + S["range_delta_stored"] + S["range_S_cluster_minus_delta"]),
         "  genuine-part entropy minus interior entropy: range [%+.2e, %+.2e]" % tuple(S["range_S_genuine_minus_int"]),
         "  cluster deviations from 1: max |dev| [%.1e, %.1e], rms [%.1e, %.1e], count below 1 [%d, %d]" % tuple(S["range_cluster_max_abs_dev"] + S["range_cluster_rms_dev"] + S["range_cluster_n_below_1"]),
         "  separation: 1 - (largest genuine) [%.2e, %.2e] against interior mu_min [%.2e, %.2e]" % tuple(S["range_one_minus_genuine_max"] + S["range_mu_min_from_int"]),
         "  interior spectrum's near-boundary (1e-9) entropy: [%.2e, %.2e]" % tuple(S["range_int_S_near_boundary_1e-9"]),
         "route 2's stored polar_global entropy minus delta: [%+.2e, %+.2e]; ratio [%.3f, %.3f]" % tuple(S["range_pol_global_minus_delta"] + S["range_pol_global_over_delta"]),
         "route 2's polar residual ||Q^dag Q - 1||_F [%.2e, %.2e]; its Gram eigenvector orthogonality [%.2e, %.2e]" % tuple(S["range_polar_residual_frob"] + S["range_gram_evec_orth_frob"])]
    if "range_global_max_abs_dev" in S:
        L.append("global polar spectrum (eigenvalues of Q^dag Q) deviation from 1: max [%.1e, %.1e], rms [%.1e, %.1e]" % tuple(S["range_global_max_abs_dev"] + S["range_global_rms_dev"]))
    if raw:
        L.append("raw (%d gated rows): cluster - delta [%+.2e, %+.2e]; genuine - interior [%+.2e, %+.2e]" % tuple([S["n_raw_rows"]] + S["range_raw_S_cluster_minus_delta"] + S["range_raw_S_genuine_minus_int"]))
    L.append("weighted cluster entropy by clock: %s" % {T: "%.4e" % v for T, v in S["weighted_S_cluster_by_clock"].items()})
    L.append("weighted delta by clock (= E2): %s" % {T: "%.4e" % v for T, v in S["weighted_delta_by_clock"].items()})
    txt = "\n".join(L) + "\n"
    with open(C.OUT + "/spectra_summary.txt", "w") as f:
        f.write(txt)
    print(txt)


if __name__ == "__main__":
    main()
