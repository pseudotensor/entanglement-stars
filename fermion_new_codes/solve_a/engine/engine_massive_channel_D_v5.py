#!/usr/bin/env python
"""engine_massive_channel_D_v5.py (the lane's analyst builder; engine spec 40.3 (1), (2), the lead's turn 48): D's return marker, a new
signed readout of the pairs' sector beyond 24 r_s on the stored full states (D version 4 was ruled a composer-only addition in 39.4 (6); the
stored D jsons carry no direction split, 40.0 (i)).

Per slice and radial channel kappa (weight w = 16 kappa, the slice's stored weights):
- the state as D builds it (engine_massive_channel_D_v3.channel_D): Phi = [sea, dust], Loewdin-orthonormalized together in float64; the
  Killing vacuum P_- = 1(h_m < 0) of h_m at the slice's applied fields (one dense eigendecomposition); the pairs' deviation is C - P less its
  dust-tagged part, Delta_pairs = Phi_sea Phi_sea^dagger - P_- with Phi_sea the Loewdin sea columns (D's E_out_pairs_Pm = Re Tr[chi h_m Delta_pairs]
  is recomputed here and printed beside D's stored value when given);
- its block beyond R_cut = 24 r_s (the sites R > 24 r_s, the sharp cut), Delta_Y = (Delta_pairs)_YY, diagonalized: Delta_Y = sum_k lambda_k x_k x_k^dagger;
- the signed projection as HORIZON version 2 builds it (engine_single_object_B_v2.signed_branch, here on numpy and on the weighted columns
  x_k): the region split into sine windows of L sites with half overlap (the squared windows add to one inside the region), each window's
  orthonormal DFT zero-padded to 4 L; at the window centre c the spinors u_+-(q) of D(q) = m sigma_z + sin q sigma_x + (kappa/R_c) sigma_y, the
  local band energies omega_+-(q) = +-N_c E(q) + (v_c/2) sin 2q (E(q) = (m^2 + sin^2 q + kappa^2/R_c^2)^(1/2); the flow term is proportional to
  the identity) and the local group velocities v_+-(q) = +-N_c sin q cos q/E(q) + v_c cos 2q; the occupation of (window, q, branch) is
  n = sum_k lambda_k |<u, F x_k>|^2 (a windowed effect of the deviation, not a global projector, #55);
- printed per window size L = 16, 32, 64: on the positive band the energy sum omega_+ n with v_+ > 0 (outgoing) and with v_+ <= 0 (incoming),
  the numbers likewise; the negative band's two parts; the windowed total sum omega n over both bands against the block energy
  Tr[h_YY Delta_Y] (a check of the local reading, not an identity);
- weighted over the channels (w); the ratio incoming/outgoing of the positive-band energy per window size.
The marker (40.3 (2), the lead's threshold): the first slice, in clock order, at which the incoming positive-band energy beyond 24 r_s exceeds
1 percent of the outgoing at every window size (16, 32, 64); every slice from it on is labelled "after the chain end's first return"; the
paper's D numbers are those at or before the slice preceding it. The baseline slices (the production's T/r_s 28 to 40, where nothing of the
formation has reached 32 r_s: s4far's far collector reads 1.7e-08 of E at 40) are printed with the same readout and the same test, so the
threshold's level against the pre-return ratio is visible; if a baseline slice already meets the test, the marker is printed with that fact.
No mutual information is computed (D15). Nothing is fed back; the stored states are only read.
Usage: --slices a.npz,b.npz (any order; sorted by clock) --baseline-upto-rs 40 (slices of the --baseline-tag run at or before it are the
baseline) --d-json a.json,... (D's stored jsons, for E_out_pairs_Pm beside) --threads 8 --out-json --md; or --compose a.json,b.json --md --out-json."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
import argparse
import json
import math
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import engine_single_object_A_v2 as SA  # noqa: E402

LABEL = ("the model's own massive channel; the pairs' deviation (C - P_-, the dust-tagged part removed) beyond 24 r_s split by the local "
         "group-velocity sign on the positive intrinsic band: a windowed reading of the stored states, not a global projector or a flux")
RETURN_LABEL = "after the chain end's first return"
THRESH = 0.01
WINDOWS = (16, 32, 64)


def utc():
    return time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())


def loewdin(X):
    G = SA.mm_d(X.conj().T, X)
    gw, gv = SA.eigh_d(G)
    Xo = SA.mm_d(X, (gv * (1.0 / np.sqrt(gw))[None, :]) @ gv.conj().T)
    return Xo, float(np.max(np.abs(SA.mm_d(Xo.conj().T, Xo) - np.eye(G.shape[0]))))


def herm(A):
    return 0.5 * (A + A.conj().T)


def signed_energy(Xw, R, Nf, vf, kappa, m, i0, L, pad=4):
    """Xw: (2(n - i0)) x K columns on the region's rows (already weighted by sqrt|lambda|) with signs sg (+-1 per column); returns the
    branch/direction sums of n and omega n. Columns are given as (X, sg)."""
    X, sg = Xw
    n = len(R)
    ns = n - i0
    hL = L // 2
    starts = np.arange(-hL, ns, hL)
    idx = starts[:, None] + np.arange(L)[None, :]
    valid = (idx >= 0) & (idx < ns)
    idc = np.clip(idx, 0, ns - 1)
    Xs = X.reshape(ns, 2, -1)
    gw = np.sin(math.pi * (np.arange(L) + 0.5) / L)
    wt = gw[None, :] * valid
    M = pad * L
    q = 2.0 * math.pi * np.fft.fftfreq(M)
    cen = np.clip(starts + hL, 0, ns - 1) + i0
    Nc = np.asarray(Nf, float)[cen]; vc = np.asarray(vf, float)[cen]; Rc = np.asarray(R, float)[cen]
    s_ = np.sin(q)[None, :]; c_ = np.cos(q)[None, :]
    kr = (kappa / Rc)[:, None]
    E = np.sqrt(m * m + s_ ** 2 + kr ** 2)
    nrm = np.sqrt(2.0 * E * (E + m))
    up0 = (m + E) / nrm; up1 = (s_ + 1j * kr) / nrm
    om_p = Nc[:, None] * E + 0.5 * vc[:, None] * np.sin(2.0 * q)[None, :]
    om_m = -Nc[:, None] * E + 0.5 * vc[:, None] * np.sin(2.0 * q)[None, :]
    vp = Nc[:, None] * s_ * c_ / E + vc[:, None] * np.cos(2.0 * q)[None, :]
    vm = -Nc[:, None] * s_ * c_ / E + vc[:, None] * np.cos(2.0 * q)[None, :]
    acc = {k: 0.0 for k in ("n_plus_out", "n_plus_in", "n_minus_out", "n_minus_in", "E_plus_out", "E_plus_in", "E_minus_out", "E_minus_in")}
    B = 256
    for b0 in range(0, X.shape[1], B):
        seg = Xs[idc][:, :, :, b0:b0 + B] * wt[:, :, None, None]           # W x L x 2 x c
        F = np.fft.fft(seg, n=M, axis=1, norm="ortho")                      # W x M x 2 x c
        s = sg[b0:b0 + B][None, None, :]
        a_p = np.conj(up0)[:, :, None] * F[:, :, 0, :] + np.conj(up1)[:, :, None] * F[:, :, 1, :]
        w_tot = np.abs(F[:, :, 0, :]) ** 2 + np.abs(F[:, :, 1, :]) ** 2
        w_p = np.abs(a_p) ** 2
        w_m = w_tot - w_p
        np_ = np.sum(w_p * s, axis=2); nm_ = np.sum(w_m * s, axis=2)        # W x M (signed by lambda)
        po = vp > 0; mo = vm > 0
        acc["n_plus_out"] += float(np.sum(np_[po])); acc["n_plus_in"] += float(np.sum(np_[~po]))
        acc["n_minus_out"] += float(np.sum(nm_[mo])); acc["n_minus_in"] += float(np.sum(nm_[~mo]))
        acc["E_plus_out"] += float(np.sum((om_p * np_)[po])); acc["E_plus_in"] += float(np.sum((om_p * np_)[~po]))
        acc["E_minus_out"] += float(np.sum((om_m * nm_)[mo])); acc["E_minus_in"] += float(np.sum((om_m * nm_)[~mo]))
    acc["E_windowed_total"] = acc["E_plus_out"] + acc["E_plus_in"] + acc["E_minus_out"] + acc["E_minus_in"]
    acc["windows"] = int(len(starts)); acc["N_nonpositive_windows"] = int(np.sum(Nc <= 0)); acc["v_positive_windows"] = int(np.sum(vc > 0))
    return acc


def channel(R, m, kappa, w, Napp, vapp, sea, dust, Rcut, Rcol):
    t0 = time.time()
    Phi = np.concatenate([np.asarray(sea, np.complex128), np.asarray(dust, np.complex128)], axis=1)
    nsea = int(np.asarray(sea).shape[1])
    Phi_o, orth = loewdin(Phi)
    S = Phi_o[:, :nsea]
    H = SA.hm_host(R, m, kappa, Napp, vapp)
    e, V = SA.eigh_d(H.toarray())
    Vn = V[:, e < 0]
    out = {"kappa": int(kappa), "w": float(w), "n_sea": nsea, "n_dust": int(Phi.shape[1] - nsea), "loewdin_orthonormality": orth, "n_negative": int(np.sum(e < 0))}
    hS = H @ S; hV = H @ Vn
    for lab, Rc in (("16", Rcol), ("24", Rcut)):
        i0 = int(np.searchsorted(np.asarray(R, float), Rc, side="right"))
        ry = np.arange(2 * i0, 2 * len(R))
        out["E_out_pairs_Pm_%s" % lab] = float(np.sum((S[ry].conj() * hS[ry]).real) - np.sum((Vn[ry].conj() * hV[ry]).real))
    i0 = int(np.searchsorted(np.asarray(R, float), Rcut, side="right"))
    ry = np.arange(2 * i0, 2 * len(R))
    DY = herm(SA.mm_d(S[ry], S[ry].conj().T) - SA.mm_d(Vn[ry], Vn[ry].conj().T))
    hY = H[ry][:, ry].toarray()
    out["block_energy_Y"] = float(np.real(np.sum(hY * DY.T)))
    lam, xk = np.linalg.eigh(DY)
    keep = np.abs(lam) > 1e-13
    out["lambda_kept"] = int(keep.sum()); out["N_plus"] = float(lam[lam > 0].sum()); out["N_minus"] = float(-lam[lam < 0].sum())
    Xw = (xk[:, keep] * np.sqrt(np.abs(lam[keep]))[None, :], np.sign(lam[keep]))
    out["first_site_beyond_24"] = i0 + 1
    out["windows"] = {}
    for L in WINDOWS:
        out["windows"][str(L)] = signed_energy(Xw, R, Napp, vapp, kappa, m, i0, L)
    out["seconds"] = time.time() - t0
    return out


def run(a):
    import torch
    torch.set_num_threads(a.threads)
    SA.set_device(-1)
    dvals = {}
    for p in [x for x in (a.d_json or "").split(",") if x]:
        Jd = json.load(open(p))
        for sl in Jd.get("slices", []):
            dvals[round(float(sl["T_over_rs"]), 6)] = {str(c["kappa"]): {k_: c["regions"][k_].get("E_out_pairs_Pm") for k_ in c["regions"]} for c in sl["channels"]}
    out = {"script": "engine_massive_channel_D_v5.py", "label": LABEL, "d15": "no mutual information is computed or printed (D15)", "args": vars(a),
           "utc_start": utc(), "threshold": THRESH, "windows": list(WINDOWS), "slices": []}
    for p in [x for x in a.slices.split(",") if x]:
        t_s = time.time()
        Z = np.load(p)
        T = float(Z["T"]); R = np.asarray(Z["R"], float)
        rs = float(a.rs); m = float(a.m)
        Napp = np.asarray(Z["N_applied"], float); vapp = np.asarray(Z["v_applied"], float)
        ks = [int(k) for k in Z["kappas"]] if a.kappas == "all" else [int(k) for k in a.kappas.split(",")]
        wts = {int(k): float(w) for k, w in zip(Z["kappas"], Z["weights"])}
        row = {"slice": p, "T_over_rs": T / rs, "dtype": str(Z["dtype"]), "baseline": bool(a.baseline_tag and a.baseline_tag in os.path.basename(p) and T / rs <= a.baseline_upto_rs + 1e-9),
               "channels": []}
        for kap in ks:
            ch = channel(R, m, kap, wts[kap], Napp, vapp, Z["sea_%d" % kap], Z["dust_%d" % kap], a.R_cut_rs * rs, a.R_collar_rs * rs)
            dv = (dvals.get(round(T / rs, 6)) or {}).get(str(kap))
            ch["D_stored_E_out_pairs_Pm"] = dv
            row["channels"].append(ch)
            w32 = ch["windows"]["32"]
            print("T/rs %.3f kappa %d: E_out pairs beyond 16 %+.6e (D stored %s), beyond 24 %+.6e; block energy %+.6e; L 32: + out %+.4e in %+.4e; - out %+.4e in %+.4e; windowed total %+.4e (%.1f s)" % (
                T / rs, kap, ch["E_out_pairs_Pm_16"], (dv or {}).get("16"), ch["E_out_pairs_Pm_24"], ch["block_energy_Y"], w32["E_plus_out"], w32["E_plus_in"],
                w32["E_minus_out"], w32["E_minus_in"], w32["E_windowed_total"], ch["seconds"]), flush=True)
        Z.close()
        row["weighted"] = weigh(row)
        row["seconds"] = time.time() - t_s
        out["slices"].append(row)
        json.dump(out, open(a.out_json + ".partial", "w"), indent=1, default=float)
    out["slices"].sort(key=lambda r: r["T_over_rs"])
    out["utc_end"] = utc()
    json.dump(out, open(a.out_json, "x"), indent=1, default=float)
    os.remove(a.out_json + ".partial")
    print("wrote %s (%d slices)" % (a.out_json, len(out["slices"])))


def weigh(row):
    W = {}
    for L in WINDOWS:
        s = {}
        for k in ("E_plus_out", "E_plus_in", "E_minus_out", "E_minus_in", "E_windowed_total", "n_plus_out", "n_plus_in"):
            s[k] = float(sum(c["w"] * c["windows"][str(L)][k] for c in row["channels"]))
        s["ratio_in_out_plus"] = s["E_plus_in"] / s["E_plus_out"] if s["E_plus_out"] != 0 else None
        W[str(L)] = s
    W["block_energy_Y"] = float(sum(c["w"] * c["block_energy_Y"] for c in row["channels"]))
    W["E_out_pairs_Pm_16"] = float(sum(c["w"] * c["E_out_pairs_Pm_16"] for c in row["channels"]))
    W["E_out_pairs_Pm_24"] = float(sum(c["w"] * c["E_out_pairs_Pm_24"] for c in row["channels"]))
    W["meets_test"] = all((W[str(L)]["E_plus_out"] > 0 and W[str(L)]["E_plus_in"] > THRESH * W[str(L)]["E_plus_out"]) for L in WINDOWS)
    return W


def compose(a):
    rows = []
    srcs = [x for x in a.compose.split(",") if x]
    for p in srcs:
        rows += json.load(open(p))["slices"]
    rows.sort(key=lambda r: r["T_over_rs"])
    marker = next((r for r in rows if r["weighted"]["meets_test"]), None)
    base_meet = [r["T_over_rs"] for r in rows if r["baseline"] and r["weighted"]["meets_test"]]
    L_ = ["# D_return_marker_v1: the pairs' positive-band energy beyond 24 r_s by the local group-velocity sign (engine spec 40.3 (1), (2))", "",
          "**Label:** %s." % LABEL, "", "Composed %s by engine_massive_channel_D_v5.py from %s; every number printed by the script." % (utc(), ", ".join(srcs)), "",
          "The marker (the lead's threshold, 40.3 (2)): the first slice in clock order at which the incoming positive-band energy beyond 24 r_s exceeds "
          "%g of the outgoing at every window size (%s sites); every slice from it on is labelled \"%s\"; the paper's D numbers are those at or before the "
          "slice preceding it." % (THRESH, ", ".join(str(w) for w in WINDOWS), RETURN_LABEL), ""]
    if marker:
        prev = [r for r in rows if r["T_over_rs"] < marker["T_over_rs"] and not r["baseline"]]
        L_.append("- the marker: T/r_s %.3f%s; the slice preceding it (the continuation's): %s." % (marker["T_over_rs"], " (a baseline slice: the test is met before any return can arrive)" if marker["baseline"] else "",
                  "%.3f" % prev[-1]["T_over_rs"] if prev else "none"))
    else:
        L_.append("- the marker: not met on any composed slice (the last composed slice T/r_s %.3f)." % rows[-1]["T_over_rs"])
    L_.append("- baseline slices meeting the test: %s." % (", ".join("%.3f" % t for t in base_meet) if base_meet else "none"))
    L_ += ["", "Weighted over the channels (16 kappa). Energies in lattice units; ratio = incoming/outgoing on the positive band.", "",
           "| T/r_s | baseline | label | L | + out E | + in E | ratio | - out E | - in E | windowed total | block energy Tr[h_YY Delta_Y] | E_out pairs beyond 24 | E_out pairs beyond 16 |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    after = False
    for r in rows:
        if marker and r is marker:
            after = True
        W = r["weighted"]
        for L in WINDOWS:
            s = W[str(L)]
            L_.append("| %.3f | %s | %s | %d | %+.4e | %+.4e | %s | %+.4e | %+.4e | %+.4e | %+.4e | %+.4e | %+.4e |" % (
                r["T_over_rs"], "yes" if r["baseline"] else "no", RETURN_LABEL if after else "-", L, s["E_plus_out"], s["E_plus_in"],
                ("%.4f" % s["ratio_in_out_plus"]) if s["ratio_in_out_plus"] is not None else "-", s["E_minus_out"], s["E_minus_in"], s["E_windowed_total"],
                W["block_energy_Y"], W["E_out_pairs_Pm_24"], W["E_out_pairs_Pm_16"]))
    L_ += ["", "Checks per slice and channel: D's stored E_out_pairs_Pm beside the recomputed one (beyond 16 and 24 r_s); the windows with N <= 0 or v > 0 at the centre.", "",
           "| T/r_s | kappa | recomputed 16 | D stored 16 | recomputed 24 | D stored 24 | Loewdin | lambda kept | N_plus | N_minus | L 32: windows; N <= 0; v > 0 |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        for c in r["channels"]:
            d = c.get("D_stored_E_out_pairs_Pm") or {}
            w32 = c["windows"]["32"]
            L_.append("| %.3f | %d | %+.8e | %s | %+.8e | %s | %.1e | %d | %.4e | %.4e | %d; %d; %d |" % (
                r["T_over_rs"], c["kappa"], c["E_out_pairs_Pm_16"], ("%+.8e" % d["16"]) if d.get("16") is not None else "-", c["E_out_pairs_Pm_24"],
                ("%+.8e" % d["24"]) if d.get("24") is not None else "-", c["loewdin_orthonormality"], c["lambda_kept"], c["N_plus"], c["N_minus"],
                w32["windows"], w32["N_nonpositive_windows"], w32["v_positive_windows"]))
    L_ += ["", "The windowed total is the local band reading's energy; its difference from the block energy measures the local reading's error "
           "(the window and the local-dispersion approximation), printed, not corrected. No mutual information is computed (D15)."]
    S = {"script": "engine_massive_channel_D_v5.py", "label": LABEL, "threshold": THRESH, "windows": list(WINDOWS), "sources": srcs,
         "marker_T_over_rs": marker["T_over_rs"] if marker else None, "marker_is_baseline": bool(marker and marker["baseline"]), "baseline_meeting_test": base_meet,
         "rows": [{"T_over_rs": r["T_over_rs"], "baseline": r["baseline"], "weighted": r["weighted"]} for r in rows]}
    open(a.md, "x").write("\n".join(L_) + "\n")
    json.dump(S, open(a.out_json, "x"), indent=1, default=float)
    print("composed %s: marker %s" % (a.md, S["marker_T_over_rs"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slices", default=None); ap.add_argument("--d-json", default=None)
    ap.add_argument("--rs", type=float, default=64.0); ap.add_argument("--m", type=float, default=0.2)
    ap.add_argument("--R-collar-rs", type=float, default=16.0); ap.add_argument("--R-cut-rs", type=float, default=24.0)
    ap.add_argument("--kappas", default="all"); ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--baseline-tag", default="_r11_"); ap.add_argument("--baseline-upto-rs", type=float, default=40.0)
    ap.add_argument("--compose", default=None); ap.add_argument("--md", default=None); ap.add_argument("--out-json", required=True)
    a = ap.parse_args()
    if a.compose:
        return compose(a)
    if os.path.exists(a.out_json):
        sys.exit("refusing to overwrite %s" % a.out_json)
    run(a)


if __name__ == "__main__":
    main()
