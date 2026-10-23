#!/usr/bin/env python
"""Item 7, the repair pass on the finished version 4 records (SPEC_strand7_page.md sec. 14.5; builder analyst).  For each record
s7p_<tag>.json (with s7p_<tag>_modes.npz): the background, the cuts and the output list are rebuilt from the record's meta (LockedBackground
with A_0, r_s0, R_0, T_end + 2 tau and static_from; the S7a fields Background; or flat); at every output and sector ch.sectors(bg, T,
parts=True) of strand7_sea_page_v5.py gives U, B_0, sig, X, and the reference is Phi_vac := polar_vacuum(U) (sec. 14.2).  Recomputed: every
"vacad" quantity of cut_analysis on the interior sides (S_int_vacad at every cut, MI_vacad, G_vacad at every d, S_de_int_vacad = 0 by
construction) and dSad := S_int_state - S_int_vacad from the record's stored state values, which are not touched; the exterior sides of the
reference at the check times are dropped (purity holds by construction for a Slater determinant; the v5 runs compute them).  The spectral sums:
when the recomputed smallest sig_k = ||X z_k|| over B_0's eigenvectors is below sigma_th = 1e-10/h, the stored contributions of the sector's
smallest-|omega| pair (the two stored modes of smallest |omega|) are removed from N_ext, N_near, S_diag,rad, S_diag,near and the first bin,
and the pair is listed from its stored entries (label of sec. 14.3, "from the eigh route" added); the modes npz gets the keep row.  The "vac0"
quantities are kept; the T = 0 projector control (the eigh route's lowest n against the polar vacuum at T = 0) is recomputed and printed.
Control C-pg-16 (descriptive: the old route's failure): per channel, sector, output and cut the differences stored minus polar of S_int_vacad,
MI_vacad and G_vacad, and the projector difference ||P_eigh - P_polar||_F at the output.  Writes s7p_<tag>_r2.json and s7p_<tag>_r2_modes.npz
beside the record ("revision": 2, a "repair" block), and --md with the C-pg-16 table per record.  Args: --dir, --tags (comma list),
--device, --md."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import copy
import json
import math
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import strand7_sea_page_v5 as SP  # noqa: E402

C128 = torch.complex128


def rebuild(meta):
    rs0 = float(meta["r_s0"])
    h = float(meta["h"])
    tau = float(meta["tau"])
    T_end = float(meta["T_end"])
    sf = None if meta.get("static_from_over_rs") is None else float(meta["static_from_over_rs"]) * rs0
    S7 = None
    if meta.get("flat") or meta.get("background") == "flat":
        bg = SP.Background.__new__(SP.Background)
        bg.flat, bg.static_from, bg.Tlast, bg.r, bg.T = True, None, 0.0, np.array([0.0, 1.0]), np.array([0.0])
        cuts_obj = SP.Cuts(None, h, flat=True)
    elif meta.get("background") == "locked":
        locked = SP.LockedBackground(float(meta["A0_over_rs0"]) * rs0, rs0, float(meta["R0_over_rs"]) * rs0, T_end + 2 * tau, static_from=sf)
        bg = locked
        cuts_obj = SP.Cuts(None, h, locked=locked, T_end=T_end)
    else:
        S7 = json.load(open(meta["s7a_json"]))
        bg = SP.Background(meta["fields"], flat=False, static_from=sf)
        cuts_obj = SP.Cuts(S7, h)
    return bg, cuts_obj, rs0, h


def repair_record(path, dev):
    R = json.load(open(path))
    meta = R["meta"]
    mp = path[:-5] + "_modes.npz"
    modes = dict(np.load(mp)) if os.path.exists(mp) else {}
    bg, cuts_obj, rs0, h = rebuild(meta)
    n = int(meta["n"])
    sigma_th = SP.SIGMA_TH_H / h
    R2 = copy.deepcopy(R)
    R2["revision"] = 2
    c16 = []
    t0 = time.time()
    T0ctl = {}
    newmodes = {}
    for jkey, rows in R2["channels"].items():
        j = float(jkey)
        ch = SP.ChannelFast(j, n, h, dev)
        Hs0, _, _, p0 = ch.sectors(bg, 0.0, parts=True)
        T0ctl[jkey] = {}
        for name, H in Hs0.items():
            lam, V = SP.eigh_safe(H)
            T0ctl[jkey][name] = SP.projector_diff(SP.polar_vacuum(p0[name][0], n), V[:, :n].to(C128))
        for q in rows:
            T = float(q["T"])
            Hs_T, A, B, prt = ch.sectors(bg, T, parts=True)
            cuts = cuts_obj.at(T, A, rs0)
            cut_dev = max(abs(cuts[k] - q["cuts"][k]) for k in q["cuts"])
            for name, H in Hs_T.items():
                U, B0m, sigG, X = prt[name]
                Phivac = SP.polar_vacuum(U, n)
                lam, V = SP.eigh_safe(H)
                pdiff = SP.projector_diff(Phivac, V[:, :n].to(C128))
                radii = ch.radii(name)
                rdom = ch.r_int if name == "alpha" else ch.r_half
                rrng = ch.r_half if name == "alpha" else ch.r_int
                sec = q["sectors"][name]
                for cname, rho in cuts.items():
                    row = sec["cuts"][cname]
                    rin = np.where(radii < rho)[0]
                    old = {"S_int_vacad": row["S_int_vacad"], "MI_vacad": row["MI_vacad"]}
                    s_new = SP.peschel(Phivac[rin, :])
                    w = 4 * (h if h >= 1 else 1.0)
                    Aw = np.where((radii >= rho - w) & (radii < rho))[0]
                    Bw = np.where((radii >= rho) & (radii < rho + w))[0]
                    AB = np.concatenate([Aw, Bw])
                    mi_new = SP.peschel(Phivac[Aw, :]) + SP.peschel(Phivac[Bw, :]) - SP.peschel(Phivac[AB, :])
                    gd = {}
                    for d_units in SP.CUT_DISTANCES[cname]:
                        d = d_units * (h if h >= 1 else 1.0)
                        a_ = [int(np.argmin(np.abs(rdom - (rho - d)))), n + int(np.argmin(np.abs(rrng - (rho - d))))]
                        b_ = [int(np.argmin(np.abs(rdom - (rho + d)))), n + int(np.argmin(np.abs(rrng - (rho + d))))]
                        gk = "G_vacad_d%g" % d
                        gd[gk] = (row.get(gk), SP.two_point(Phivac, a_, b_))
                    row["S_int_vacad"] = s_new
                    row["MI_vacad"] = mi_new
                    for gk, (o, nv) in gd.items():
                        row[gk] = nv
                    row["dSad"] = row["S_int_state"] - s_new
                    row["S_de_int_vacad"] = 0.0
                    row.pop("S_ext_vacad", None)
                    c16.append({"j": j, "sector": name, "T_over_rs0": q["T_over_rs0"], "cut": cname, "d_S_int_vacad": old["S_int_vacad"] - s_new,
                                "d_MI_vacad": old["MI_vacad"] - mi_new,
                                "d_G_vacad_max": max((abs(o - nv) for o, nv in gd.values() if o is not None), default=0.0), "projector_diff": pdiff})
                # the spectrum
                mu, Z = SP.eigh_safe(B0m)
                sig_k = torch.linalg.norm(X @ Z, dim=0).cpu().numpy()
                smin = float(sig_k.min())
                mk = "j%g_%s_T%g" % (j, name, T)
                sp = sec["spectrum"]
                rep = {"sig_min": smin, "sigma_th": sigma_th, "removed": None}
                if mk in modes:
                    om, na, rr = (modes[mk][i].astype(np.float64) for i in range(3))
                    keep = np.ones(len(om), dtype=bool)
                    if smin < sigma_th:
                        idx = np.argsort(np.abs(om))[:2]
                        keep[idx] = False
                        rho_rad = cuts["rad"]
                        TH = q["T_H"]
                        rem = {"N_ext": 0.0, "N_near": 0.0, "S_diag_rad": 0.0, "S_diag_near": 0.0}
                        for i in idx:
                            ext = rr[i] > rho_rad
                            if om[i] > 0:
                                rem["N_ext" if ext else "N_near"] += na[i]
                            rem["S_diag_rad" if ext else "S_diag_near"] += float(SP.h2(np.array([na[i]]))[0])
                            if ext and abs(om[i]) < TH:
                                b0 = sp["bins"][0]
                                part = b0["particles"] if om[i] > 0 else b0["holes"]
                                val = na[i] if om[i] > 0 else 1 - na[i]
                                part["sum"] -= val
                                part["modes"] -= 1
                                part["mean"] = (part["sum"] / part["modes"]) if part["modes"] > 0 else None
                        for k_, v_ in rem.items():
                            sp[k_] -= v_
                        sp["excluded_modes"] = 2
                        lab = ("edge pair at the wall; the polar column's sign is not determined: n^+ and n^- may be exchanged; n^+ + n^- is the "
                               "pair's weight in the state; from the eigh route")
                        sp["excluded_pairs"] = [{"omega": float(om[i]), "n": float(na[i]), "r": float(rr[i]), "label": lab} for i in idx]
                        rep["removed"] = rem
                    newmodes[mk] = np.stack([om, na, rr, keep.astype(float)]).astype(np.float32)
                sp["sigma_th"] = sigma_th
                sec["controls"]["vac_projector_diff"] = pdiff
                sec["controls"]["sig_min"] = smin
                sec["controls"]["unresolved_pairs"] = int(np.sum(sig_k < sigma_th))
                sec["repair"] = rep
            q["repair_cut_rebuild_max_dev"] = cut_dev
        print("%s j %s done (%.0f s)" % (os.path.basename(path), jkey, time.time() - t0), flush=True)
    R2["repair"] = {"script": "strand7_page_repair_v1.py", "builder": "analyst", "rule": "SPEC_strand7_page.md sec. 14.5", "sigma_th": sigma_th,
                    "vac_projector_diff_T0": T0ctl, "seconds": time.time() - t0}
    out = path[:-5] + "_r2.json"
    with open(out, "w") as fh:
        json.dump(R2, fh, indent=1, default=float)
    if newmodes:
        np.savez_compressed(path[:-5] + "_r2_modes.npz", **newmodes)
    return out, c16, T0ctl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--tags", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--md", required=True)
    a = ap.parse_args()
    dev = torch.device(a.device if torch.cuda.is_available() or a.device == "cpu" else "cpu")
    L = ["# Item 7, the repair pass (strand7_page_repair_v1.py; SPEC_strand7_page.md sec. 14.5): C-pg-16 per record", ""]
    summ = {}
    for tag in [x for x in a.tags.split(",") if x.strip()]:
        p = os.path.join(a.dir, "s7p_%s.json" % tag)
        if not os.path.exists(p):
            L.append("- %s: record not found" % tag)
            continue
        out, c16, T0 = repair_record(p, dev)
        big = [x for x in c16 if max(abs(x["d_S_int_vacad"]), abs(x["d_MI_vacad"]), x["d_G_vacad_max"]) > 1e-6]
        small = [x for x in c16 if x not in big]
        mx = {k: max((abs(x[k]) for x in small), default=0.0) for k in ("d_S_int_vacad", "d_MI_vacad", "d_G_vacad_max", "projector_diff")}
        ln2 = [x for x in big if abs(abs(x["d_S_int_vacad"]) - math.log(2)) < 1e-5]
        summ[tag] = {"r2": out, "entries": len(c16), "above_1e-6": len(big), "ln2_entries": len(ln2), "max_below": mx, "T0": T0}
        L += ["## %s -> %s" % (tag, os.path.basename(out)), "",
              "- %d (channel, sector, output, cut) entries; %d with a difference above 1e-6, of which %d equal ln 2 = 0.693147 in S_int_vacad within 1e-5; "
              "over the others the largest |difference| in S_int_vacad %.3e, MI_vacad %.3e, G_vacad %.3e and the projector difference %.3e; the T = 0 projector "
              "control %s." % (len(c16), len(big), len(ln2), mx["d_S_int_vacad"], mx["d_MI_vacad"], mx["d_G_vacad_max"], mx["projector_diff"],
                               json.dumps({j: {s: "%.2e" % v for s, v in d.items()} for j, d in T0.items()})), ""]
        if big:
            L += ["| j | sector | T/r_s0 | cut | stored - polar S_int_vacad | MI_vacad | G_vacad (largest) | projector difference |", "|---|---|---|---|---|---|---|---|"]
            for x in big:
                L.append("| %g | %s | %g | %s | %.6f | %.3e | %.3e | %.3e |" % (x["j"], x["sector"], x["T_over_rs0"], x["cut"], x["d_S_int_vacad"], x["d_MI_vacad"],
                                                                         x["d_G_vacad_max"], x["projector_diff"]))
            L.append("")
    open(a.md, "w").write("\n".join(L) + "\n")
    json.dump(summ, open(a.md[:-3] + ".json", "w"), indent=1, default=float)
    print("\n".join(L[:40]))


if __name__ == "__main__":
    main()
