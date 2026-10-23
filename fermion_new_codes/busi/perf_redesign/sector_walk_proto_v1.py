#!/usr/bin/env python
"""sector_walk_proto_v1.py (busi lane, reviewer A redesign prototype R4): the two-field engine's light sector walk (light_sector_walk_v9:
the Weyl-step piece loop with a banded Cholesky ladder on K_j(s)^2 - d^2 at every probe, K_j = h(s) - a_j(s) S) replaced by the gap
functions d_j(s) = min |spec(H_+(s)) - a_j(s)| tabulated on an s grid in one GPU launch (inertia counts on the S = +1 sector block,
bandwidth 2), with the Weyl bound d_j(s) >= d_j(s_k) - ||dK_j||_G |s - s_k| certifying the intervals between grid points. On a stored
record of the C64 R2 segment: the anchors' gap functions, the uncertified windows (the crossings) against the record's own crossing
entries, the engine's piece loop replayed on the tabulated gaps against the record's piece count, and the timing of the tabulation
against the record's walk seconds and the host ladder's per-piece cost measured here.
usage: python sector_walk_proto_v1.py OUT_DIR CODE_DIR SEG_DIR --records 13.34375,13.625 [--channels 1,2] [--ngrid 2048] [--gpu 0]"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import scipy.sparse as sp
import scipy.linalg as sla
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import banded_eigh_gpu_v1 as BE
import banded_path_gpu_v1 as BP


def clocked_csr(h0, Q, N, v):
    Nd = sp.diags(np.repeat(np.asarray(N, float), 2)); Vd = sp.diags(np.repeat(np.asarray(v, float), 2))
    return (0.5 * (Nd @ h0 + h0 @ Nd) + 0.5 * (Vd @ Q + Q @ Vd)).tocsr()


def plus_projector(nsites):
    """P_+ (2n x n): column i = (e_{2i} + (-1)^i i e_{2i+1}) / sqrt 2, the +1 eigenvector of (-1)^i sigma_y on site i."""
    i = np.arange(nsites)
    rows = np.concatenate([2 * i, 2 * i + 1]); cols = np.concatenate([i, i])
    vals = np.concatenate([np.full(nsites, 1.0 / np.sqrt(2.0), complex), (1j * (1.0 - 2.0 * (i % 2))) / np.sqrt(2.0)])
    return sp.csr_matrix((vals, (rows, cols)), shape=(2 * nsites, nsites))


def S_matrix(nsites):
    sgn = (1.0 - 2.0 * (np.arange(nsites) % 2)).astype(float)
    ev = 2 * np.arange(nsites)
    return sp.csr_matrix((np.concatenate([-1j * sgn, 1j * sgn]), (np.concatenate([ev, ev + 1]), np.concatenate([ev + 1, ev]))), shape=(2 * nsites, 2 * nsites))


def gershgorin(A):
    return float(np.max(np.asarray(abs(A).sum(axis=1)).ravel())) if A.nnz else 0.0


def band_upper(A):
    co = A.tocoo(); off = co.col - co.row; u = int(off.max()); n = A.shape[0]
    ab = np.zeros((u + 1, n), dtype=complex); sel = off >= 0
    np.add.at(ab, (u - off[sel], co.col[sel]), co.data[sel]); ab[u, :] = ab[u, :].real
    return ab, u


def pd_banded(ab, d):
    ab2 = ab.copy(); ab2[-1, :] -= d * d
    try:
        sla.cholesky_banded(ab2, lower=False, check_finite=False); return True
    except (np.linalg.LinAlgError, ValueError):
        return False


def replay_pieces(d_funcs, nG, d0, d_eff, F, s_grid, max_pieces=100000):
    """the engine's piece loop (light_sector_walk_v9 without the crossing phase) on tabulated gaps: at every probe the ladder from the
    anchor's current d down by halving until it passes (d_try < d_true(s) - floor), the doubling check, the Weyl step min_j d_rig_j/nG_j.
    Returns (pieces, the s where an anchor first fell below d_eff or None)."""
    m = len(nG); d = list(d0); s = 0.0; pieces = 0; from_eig = [True] * m
    while True:
        d_rig = [(d[j] if from_eig[j] else np.sqrt(max(d[j] * d[j] - F, 0.0))) for j in range(m)]
        step = min(d_rig[j] / nG[j] for j in range(m) if nG[j] > 0.0)
        s1 = min(1.0, s + step); pieces += 1
        if s1 >= 1.0 or pieces >= max_pieces:
            return pieces, None
        for j in range(m):
            dt = float(np.interp(s1, s_grid, d_funcs[j]))
            d_try = d[j]; d_new = None
            while d_try >= d_eff[j]:
                if d_try * d_try + F < dt * dt:
                    d_new = d_try; break
                d_try *= 0.5
            if d_new is None:
                return pieces, s1
            if d_new == d[j] and 2.0 * d_new < 1.0 and (2.0 * d_new) ** 2 + F < dt * dt:
                d_new = 2.0 * d_new
            d[j] = d_new; from_eig[j] = False
        s = s1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir"); ap.add_argument("code_dir"); ap.add_argument("seg_dir")
    ap.add_argument("--records", default="13.34375"); ap.add_argument("--channels", default="1,2"); ap.add_argument("--ngrid", type=int, default=2048)
    ap.add_argument("--gpu", type=int, default=0); ap.add_argument("--ladder-sample", type=int, default=40)
    a = ap.parse_args()
    sys.path.insert(0, a.code_dir)
    import engine_vacflow_v1 as VF
    os.makedirs(a.out_dir, exist_ok=True)
    dev = torch.device("cuda:%d" % a.gpu); torch.cuda.set_device(dev)
    jpath = glob.glob(os.path.join(a.seg_dir, "*.json"))[0]
    j = json.load(open(jpath)); recs = j["records"]
    zf = np.load(glob.glob(os.path.join(a.seg_dir, "*_fields.npz"))[0], allow_pickle=False)
    R = np.asarray(zf["R"], float); nsites = len(R); n = nsites
    Pp = plus_projector(nsites); S = S_matrix(nsites)
    eps = float(np.finfo(float).eps)
    lines = ["sector_walk_proto_v1: %s (%d sites), GPU %s, grid %d" % (os.path.basename(jpath), nsites, torch.cuda.get_device_name(dev), a.ngrid)]
    out = []
    Ts = np.array([r["T_over_rs"] for r in recs])
    for Tq in [float(x) for x in a.records.split(",")]:
        i1 = int(np.argmin(np.abs(Ts - Tq))); i0 = i1 - 1
        r1 = recs[i1]; r0 = recs[i0]
        lines.append("record %d T/rs %.5f (previous %.5f): record seconds %.1f" % (i1, Ts[i1], Ts[i0], r1.get("seconds", float("nan"))))
        N0 = np.asarray(zf["N_applied"][i0], float); v0 = np.asarray(zf["v_applied"][i0], float)
        N1 = np.asarray(zf["N_applied"][i1], float); v1 = np.asarray(zf["v_applied"][i1], float)
        for kap in [int(c) for c in a.channels.split(",")]:
            c1 = r1["light_v36"]["channels"][str(kap)]; c0 = r0["light_v36"]["channels"][str(kap)]
            sec1 = c1.get("sector") or {}; sec0 = c0.get("sector") or {}; cert1 = c1.get("certified") or {}
            anchors0 = list(sec0.get("anchors") or ([c0.get("mu")] if c0.get("mu") is not None else []))
            rec = {"T": float(Ts[i1]), "kappa": kap, "record_pieces": sec1.get("pieces"), "record_cholesky": sec1.get("cholesky"), "record_eigh": sec1.get("eigh"),
                   "record_probes": sec1.get("probes_in_crossing"), "record_crossings": sec1.get("crossings"), "record_walk_seconds": cert1.get("seconds"), "record_mode": c1.get("mode"),
                   "record_norm_dK_G": cert1.get("norm_dK_G"), "record_d_min_certified": cert1.get("d_min_certified"), "record_d_floor": cert1.get("d_floor"), "record_d_min_effective": cert1.get("d_min_effective"),
                   "anchors_start": anchors0, "anchors_end_record": sec1.get("anchors")}
            h0, Q = VF.chain_sparse(R, 0.0, float(kap))
            hA = clocked_csr(h0, Q, N0, v0); hB = clocked_csr(h0, Q, N1, v1); dh = (hB - hA).tocsr()
            HA = (Pp.conj().T @ hA @ Pp).tocsr(); dHp = (Pp.conj().T @ dh @ Pp).tocsr()
            off = (Pp.conj().T @ hA @ (sp.identity(2 * n, format="csr") - Pp @ Pp.conj().T)).tocsr()
            rec["off_sector_max"] = float(abs(off).max()) if off.nnz else 0.0
            band0 = BE.band_from_csr(HA); dband = BE.band_from_csr(dHp)
            G = BE.gershgorin_band(band0) + BE.gershgorin_band(dband); pivmin = 64.0 * eps * G
            # the sector spectra at the ends (torch on the device)
            e0 = torch.linalg.eigvalsh(torch.as_tensor(HA.toarray(), device=dev)).cpu().numpy()
            e1 = torch.linalg.eigvalsh(torch.as_tensor((HA + dHp).toarray(), device=dev)).cpu().numpy()
            bnds = [int(np.searchsorted(e0, a0) - 1) for a0 in anchors0]
            a_end = [0.5 * (e1[b] + e1[b + 1]) for b in bnds]
            rec["boundaries"] = bnds; rec["anchors_end_from_boundaries"] = a_end
            # the engine's Gershgorin quantities on the full K (the sector replay keeps them, as engine_twofield_v9c5 does)
            nG = []; GK = []
            for a0, a1 in zip(anchors0, a_end):
                K0 = (hA - a0 * S).tocsr(); K1 = (hB - a1 * S).tocsr(); dK = (K1 - K0).tocsr()
                nG.append(gershgorin(dK)); GK.append(max(gershgorin(K0), gershgorin(K1)))
            F = [64.0 * eps * g_ * g_ for g_ in GK]; d_eff = [max(1e-7, 2.0 * np.sqrt(f_)) for f_ in F]
            g0 = [float(np.min(np.abs(e0 - a0))) for a0 in anchors0]
            rec.update({"nG": nG, "GK": GK, "d_eff": d_eff, "g0": g0})
            # ---- the gap functions on the grid: counts at (s, a_j(s)), then the two adjacent levels by multisection
            s_grid = np.linspace(0.0, 1.0, a.ngrid)
            m = len(anchors0)
            torch.cuda.synchronize(); t0 = time.time()
            S_all = np.repeat(s_grid, m); A_all = np.concatenate([[a0 + s_ * (a1 - a0) for a0, a1 in zip(anchors0, a_end)] for s_ in s_grid])
            cnt = BP.counts_on_path(band0, dband, n, S_all, A_all, pivmin)
            delta = 0.05
            kidx = np.concatenate([cnt - 1, cnt]); s_two = np.concatenate([S_all, S_all]); a_two = np.concatenate([A_all, A_all])
            lo = a_two - delta; hi = a_two + delta
            blo, bhi = BP.brackets_on_path(band0, dband, n, s_two, kidx, lo, hi, pivmin, nsteps=24, tol_rel=1e-13)
            lam = 0.5 * (blo + bhi)
            torch.cuda.synchronize(); t_tab = time.time() - t0
            M = len(S_all)
            lam_below = lam[:M].reshape(a.ngrid, m); lam_above = lam[M:].reshape(a.ngrid, m); A_g = A_all.reshape(a.ngrid, m)
            d_tab = np.minimum(A_g - lam_below, lam_above - A_g)
            pair_gap = lam_above - lam_below
            rec["tabulation_seconds"] = t_tab; rec["count_threads"] = int(M); rec["bracket_tasks"] = int(2 * M)
            # the certificate between grid points by the Weyl bound; the uncertified windows
            ds = s_grid[1] - s_grid[0]
            cert_ok = np.ones((a.ngrid - 1, m), bool); windows = []
            for jj in range(m):
                lo_b = np.minimum(d_tab[:-1, jj], d_tab[1:, jj]) - nG[jj] * ds / 2.0      # the bound at the midpoint from the nearer end
                cert_ok[:, jj] = lo_b >= d_eff[jj]
                bad = np.where(~cert_ok[:, jj])[0]
                if len(bad):
                    runs = np.split(bad, np.where(np.diff(bad) > 1)[0] + 1)
                    for run in runs:
                        windows.append({"anchor": jj, "s_lo": float(s_grid[run[0]]), "s_hi": float(s_grid[run[-1] + 1]), "d_min_in_window": float(d_tab[run[0]:run[-1] + 2, jj].min()),
                                        "pair_gap_min": float(pair_gap[run[0]:run[-1] + 2, jj].min())})
            rec["windows"] = windows; rec["grid_points"] = int(a.ngrid); rec["d_tab_min_per_anchor"] = [float(d_tab[:, jj].min()) for jj in range(m)]
            rec["d_tab_argmin_s_per_anchor"] = [float(s_grid[int(np.argmin(d_tab[:, jj]))]) for jj in range(m)]
            # the piece loop replayed on the tabulated gaps (non-crossing anchors; stops at the first anchor below the floor)
            d0 = [g_ - 8.0 * eps * gk for g_, gk in zip(g0, GK)]
            pieces, s_fail = replay_pieces([d_tab[:, jj] for jj in range(m)], nG, d0, d_eff, max(F), s_grid)
            rec["replay_pieces_to_first_failure"] = pieces; rec["replay_first_failure_s"] = s_fail
            # the host ladder's cost on this record (a sample of probe points, the full 2n x 2n K as the engine factors it)
            t_lad = []
            for q in np.linspace(0.05, 0.95, a.ladder_sample):
                a_s = anchors0[0] + q * (a_end[0] - anchors0[0])
                Ks = ((hA + q * dh) - a_s * S).tocsr()
                t0 = time.time(); ab, _u = band_upper(Ks @ Ks); ok = pd_banded(ab, 0.5 * d_eff[0]); t_lad.append(time.time() - t0)
            rec["host_ladder_per_probe_s"] = float(np.median(t_lad))
            rec["host_ladder_equiv_for_record_s"] = float(np.median(t_lad)) * float(sec1.get("cholesky") or 0)
            cr = sec1.get("crossings") or []
            line = ("  kappa %d: anchors %d (start %s), off-sector block max %.1e; record: mode '%s', pieces %s, cholesky %s, probe eighs %s, crossings %d at s %s (pair gap at end %s), walk %.2f s on the card, d_min_certified %s, floor %s | "
                    "tabulation: %d grid points x %d anchors = %d counts + %d brackets in %.3f s on this GPU; gap minima per anchor %s at s %s; uncertified windows %s; piece loop replayed on the tabulated gaps: %d pieces to the first floor failure at s %s; host ladder here %.2f ms per probe (x %s calls = %.1f s)" % (
                        kap, m, ["%.3e" % x for x in anchors0], rec["off_sector_max"], c1.get("mode"), sec1.get("pieces"), sec1.get("cholesky"), sec1.get("eigh"), len(cr), ["%.4f" % c_["s"] for c_ in cr], ["%.2e" % c_["pair_gap_end"] for c_ in cr],
                        float(cert1.get("seconds") or 0.0), cert1.get("d_min_certified"), cert1.get("d_min_effective"), a.ngrid, m, M, 2 * M, t_tab, ["%.2e" % x for x in rec["d_tab_min_per_anchor"]], ["%.4f" % x for x in rec["d_tab_argmin_s_per_anchor"]],
                        [(w["anchor"], round(w["s_lo"], 4), round(w["s_hi"], 4), "%.1e" % w["d_min_in_window"]) for w in windows], pieces, s_fail, 1e3 * rec["host_ladder_per_probe_s"], sec1.get("cholesky"), rec["host_ladder_equiv_for_record_s"]))
            lines.append(line); print(line, flush=True)
            out.append(rec)
            np.savez(os.path.join(a.out_dir, "gap_tab_T%.5f_k%d.npz" % (Ts[i1], kap)), s=s_grid, d=d_tab, lam_below=lam_below, lam_above=lam_above, anchors=A_g)
    with open(os.path.join(a.out_dir, "sector_walk_proto_v1.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(a.out_dir, "sector_walk_proto_v1.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)
    print("\n".join(lines[:2]))


if __name__ == "__main__":
    main()
