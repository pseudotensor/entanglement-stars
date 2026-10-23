#!/usr/bin/env python
"""dissolve_slice_readout_v1.py (busi f52b1152, 2026-10-20; DISSOLVE_redteam_reviewa_v1 ruling 3.1): the orbital-based columns of the
per-record readout list, read at the slice clocks (one slice per call; numpy only, seconds per slice).
Inputs: a complex128 slice orbitals_<tag>_T<clock>.npz (engine_collapse_v36b lines 2320 to 2335: T, rs, N_applied, v_applied, R, kappas,
weights and per channel sea_k, dust_k, f_k, R2_k, e_k, mask_k; the write desk's small extract carries only the k = 1 ball keys) and the
member's run json (the record at the slice's clock gives the dense roots R_in, R_t; sigma = args.ell).
Per seeded channel k (dust_k present with a nonzero column count; the base seeds k = 1 only), with w_k the channel weight and f_j the
ball's occupations:
  number inside r: w_k sum_j f_j sum_{sites R_n < r} (|Phi_j(n, 0)|^2 + |Phi_j(n, 1)|^2) for r in {2 sigma, 4 sigma, r_s/4, r_s/2, r_s,
  R_in, R_t, 8 r_s, 16 r_s, 32 r_s, 64 r_s}, beyond = total minus inside; also as the share of the channel's total;
  momentum histogram: for each column j and spinor component c, the FFT of the column restricted to the sites inside R_t (and, separately,
  over all sites), |c_k|^2 / n summed over j, c with the weights w_k f_j, binned in 64 bins of ap in [-pi, pi) (k = 2 pi fftfreq(n));
  shares (of the restricted content): |ap| >= pi/4; |ap| >= z_c = arccos(N/|v|) at R_t - 4a and at R_t - 8a (z_c = 0 where |v| <= N;
  u = |v|/N interpolated on the applied site fields); k > 0; the same shares over all sites.
A record without a dense zero at the slice's clock: the R_t columns and the restricted histogram are NaN (the all-sites histogram is read).
Outputs: one npz per slice in --out (readout_<slice name>.npz: the radii, the numbers, the histograms, the shares) and one row per slice
appended to --out/dissolve_slice_readout_v1.tsv (header written when the file is new).
Usage: dissolve_slice_readout_v1.py --slice S.npz --run-json J.json --out DIR"""
import argparse, json, os, time
import numpy as np

NB = 64
EDGES = np.linspace(-np.pi, np.pi, NB + 1)


def hist(Phi, wcol, sel):
    """the |c_k|^2 / n of every column and component restricted to sel, weighted, binned on [-pi, pi); returns (hist, k-weights, k)."""
    n = int(sel.sum())
    k = 2 * np.pi * np.fft.fftfreq(n)
    wk = np.zeros(n)
    for j in range(Phi.shape[1]):
        if wcol[j] == 0.0:
            continue
        for c in (0, 1):
            wk += wcol[j] * np.abs(np.fft.fft(Phi[c::2, j][sel])) ** 2 / n
    h, _ = np.histogram(k, bins=EDGES, weights=wk)
    return h, wk, k


def shares(wk, k, zc4, zc8):
    s = wk.sum()
    if s <= 0:
        return [np.nan] * 4
    return [float(wk[np.abs(k) >= np.pi / 4].sum() / s), float(wk[np.abs(k) >= zc4].sum() / s), float(wk[np.abs(k) >= zc8].sum() / s), float(wk[k > 0].sum() / s)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice", required=True); ap.add_argument("--run-json", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    Z = np.load(a.slice, allow_pickle=True)
    R = np.asarray(Z["R"], float); T = float(Z["T"]); rs = float(Z["rs"]); Na = np.asarray(Z["N_applied"], float); va = np.asarray(Z["v_applied"], float)
    kap = [int(x) for x in np.asarray(Z["kappas"]).ravel()]; wts = [float(x) for x in np.asarray(Z["weights"]).ravel()]
    J = json.load(open(a.run_json)); sig = float(J["args"]["ell"])
    recs = [r for r in J["records"] if abs(r["T_over_rs"] - T / rs) < 1e-6]
    if not recs:
        raise SystemExit("no record at T/r_s %.6f in %s" % (T / rs, a.run_json))
    rec = recs[0]; tzs = rec.get("trapping_zeros") or []
    Rt = float(tzs[-1]) if tzs else np.nan; Rin = float(tzs[0]) if len(tzs) >= 2 else np.nan
    u = lambda r: float(np.interp(r, R, np.abs(va) / Na))  # noqa: E731
    zc = lambda r: float(np.arccos(1.0 / u(r))) if (np.isfinite(r) and u(r) > 1.0) else 0.0  # noqa: E731
    zc4, zc8 = (zc(Rt - 4.0), zc(Rt - 8.0)) if np.isfinite(Rt) else (np.nan, np.nan)
    radii = [("2sigma", 2 * sig), ("4sigma", 4 * sig), ("rs/4", rs / 4), ("rs/2", rs / 2), ("rs", rs), ("R_in", Rin), ("R_t", Rt),
             ("8rs", 8 * rs), ("16rs", 16 * rs), ("32rs", 32 * rs), ("64rs", 64 * rs)]
    name = os.path.basename(a.slice).replace(".npz", "")
    out = {"T": T, "rs": rs, "T_over_rs": T / rs, "sigma": sig, "R_in": Rin, "R_t": Rt, "zc_Rt_m4": zc4, "zc_Rt_m8": zc8,
           "u_Rt_m4": u(Rt - 4) if np.isfinite(Rt) else np.nan, "u_Rt_m8": u(Rt - 8) if np.isfinite(Rt) else np.nan,
           "radius_names": np.array([x[0] for x in radii]), "radii": np.array([x[1] for x in radii], float), "bin_edges": EDGES}
    rows = []
    print("dissolve_slice_readout_v1: %s (T/r_s %.4f, sigma %.0f a, r_s %.0f a, sites %d); record roots R_in %.4f R_t %.4f; u at R_t-4a %.4f (z_c %.4f), at R_t-8a %.4f (z_c %.4f)" % (
        name, T / rs, sig, rs, len(R), Rin, Rt, out["u_Rt_m4"], zc4, out["u_Rt_m8"], zc8))
    for kk, w in zip(kap, wts):
        key = "dust_%d" % kk
        if key not in Z.files or Z[key].ndim != 2 or Z[key].shape[1] == 0:
            continue
        Phi = np.asarray(Z[key], complex); f = np.asarray(Z["f_%d" % kk], float).ravel()
        dens = (np.abs(Phi[0::2]) ** 2 + np.abs(Phi[1::2]) ** 2) @ f * w       # per site, weighted
        tot = float(dens.sum())
        nin = np.array([float(dens[R < r].sum()) if np.isfinite(r) else np.nan for _, r in radii])
        wcol = w * f
        h_all, wk_all, k_all = hist(Phi, wcol, np.ones(len(R), bool))
        sh_all = shares(wk_all, k_all, zc4 if np.isfinite(zc4) else 0.0, zc8 if np.isfinite(zc8) else 0.0)
        if np.isfinite(Rt):
            h_in, wk_in, k_in = hist(Phi, wcol, R < Rt); sh_in = shares(wk_in, k_in, zc4, zc8)
        else:
            h_in = np.full(NB, np.nan); sh_in = [np.nan] * 4
        out.update({"k%d_total" % kk: tot, "k%d_inside" % kk: nin, "k%d_beyond" % kk: tot - nin, "k%d_share_inside" % kk: nin / tot,
                    "k%d_hist_inside_Rt" % kk: h_in, "k%d_hist_all" % kk: h_all, "k%d_shares_inside_Rt" % kk: np.array(sh_in),
                    "k%d_shares_all" % kk: np.array(sh_all), "k%d_weight" % kk: w, "k%d_ncols" % kk: Phi.shape[1]})
        print("  channel kappa %d (w %.0f, %d columns, sum f %.3f): total %.4f" % (kk, w, Phi.shape[1], f.sum(), tot))
        print("    inside: " + "; ".join("%s (%.3f) %.4f [%.3f]" % (nm, r, x, x / tot) for (nm, r), x in zip(radii, nin)))
        print("    momentum inside R_t: |ap|>=pi/4 %.3f, >=z_c(R_t-4a) %.3f, >=z_c(R_t-8a) %.3f, k>0 %.3f | all sites: %.3f, %.3f, %.3f, %.3f" % (*sh_in, *sh_all))
        rows.append([name, "%.6f" % (T / rs), str(kk), "%.6g" % tot] + ["%.6g" % x for x in nin] + ["%.6g" % x for x in sh_in] + ["%.6g" % x for x in sh_all]
                    + ["%.6g" % Rin, "%.6g" % Rt, "%.6g" % zc4, "%.6g" % zc8])
    out["channels"] = np.array([int(r[2]) for r in rows], int)
    np.savez_compressed(os.path.join(a.out, "readout_%s.npz" % name), **out)
    tsv = os.path.join(a.out, "dissolve_slice_readout_v1.tsv"); new = not os.path.exists(tsv)
    with open(tsv, "a") as fh:
        if new:
            fh.write("\t".join(["slice", "T_over_rs", "kappa", "total"] + ["in_" + x[0] for x in radii] + ["q_in_pi4", "q_in_zc4", "q_in_zc8", "q_in_kpos"]
                               + ["q_all_pi4", "q_all_zc4", "q_all_zc8", "q_all_kpos", "R_in", "R_t", "zc4", "zc8"]) + "\n")
        for r in rows:
            fh.write("\t".join(r) + "\n")
    print("  wrote %s and a row in %s; %.1f s" % (os.path.join(a.out, "readout_%s.npz" % name), tsv, time.time() - t0))


if __name__ == "__main__":
    main()
