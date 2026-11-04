#!/usr/bin/env python
"""engine_single_object_slices_v1.py (builder analyst, the HORIZON version 2 builder; engine spec 39.5 (1) (vi), (vii), (ix) and 38.4 (1)
(ii); brief lead/BRIEF_horizon_v2_builder.md): readouts of the stored states themselves (no propagation), per stored slice and radial
channel, at the outer root R_t (the dense root re-solved from the record's shell totals, part A's route) and beyond 2GE and 16 r_s:
(vii) the full detector density of the state's excess over the reference, Tr[Pi_D (C - P)] (C = Phi f Phi^dag over the whole occupied set,
    sea and dust, Loewdin-re-orthonormalized in float64 as part A does; P the stored continued reference R2, likewise re-orthonormalized),
    with its parts: the dust-tagged sector Tr[Pi_D C_dust] and the sea's excess Tr[Pi_D (C_sea - P)]; two routes, the direct sum over the
    sites in D and the complement route (the traces minus the sum over the sites outside D), their difference printed; the selected
    orbitals' readouts of B.5 are compared with these in HORIZON version 2;
(ix) five rates at the slice's clock, separate columns, never summed, per lattice time unit:
    1. the fixed-cut number current through R_t, F = -2 Im Tr(h_XY X_YX) = d/dt Tr[Pi_Y X] at a cut held fixed (outward positive), for
       X = C, P, D = C - P and C_dust (h the engine's h_m at the slice's applied fields);
    2. the moving-cut reassignment, -rho_X(R_t) dR_t/dt: the site density of X at the cut (linearly interpolated between the two sites
       about R_t) times the root's speed (the records' stored R_t, central difference over the neighbouring records), and the step form
       [Tr Pi_{R > R_t(t_{k+1})} X - Tr Pi_{R > R_t(t_k)} X]/tau at the slice's state;
    3. the complete reference term Gamma_P = P_dot + i[h, P] beyond the cut, Tr[Pi_Y Gamma_P]: the reference of the records k - 1 and k + 1
       rebuilt as the engine selected it (the eigenvectors of h_m at the record's stored applied fields picked by the record's stored mask
       in energy order, the run's spectra file), [Tr Pi_Y P(t_{k+1}) - Tr Pi_Y P(t_{k-1})]/(2 tau) at the cut R_t(t_k); the engine propagates
       with record k's h held over its step and its P commutes with that h, so the whole change of Tr[Pi_Y P] is this term; the rebuild
       is checked at the slice itself against the stored R2 (the subspace distance printed);
    4. the Killing-energy flux through R_t, Tr[X (i/2)[h^2, Pi_Y]] = -Im Tr((h^2)_XY X_YX), for X = C, P, D and C_dust;
    5. the Gauss charge loss, -dE/dt of the record's charge E (central difference over the neighbouring records);
(vi) the dust-tagged columns' signed projection on the positive intrinsic Dirac branch with the local group-velocity sign
    (engine_single_object_B_v2.signed_branch, its windows and conventions) inside R_t and beyond it, beside the unsigned windowed share of
    part A (the rectangular-window DFT of the interior components, pi/4 <= |ak| < pi/2), kept and labelled unsigned;
(ii) the dust-tagged sector's weight beyond R_t, 2GE and 16 r_s for every stored channel (the stored columns as stored), and its channel-weighted sum with the slice's
    weights (the 16 kappa of the engine's sources, for comparison with D's dust readout; nothing weighted is a correlation).
The stored columns of a complex64 slice carry a single-precision floor: the complement route is printed beside the direct one. Inputs: the
run's record json, fields npz and spectra npz (one run per call), the slices, the field module. No mutual information (D15); the massive
channel only (D14). Every number is printed by the script. Refuses to overwrite."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
import argparse
import importlib
import json
import math
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import engine_single_object_A_v2 as A2   # noqa: E402  (never edited: hm_host, roots_dense, outer_inner, momentum_content)
import engine_single_object_B_v2 as B2   # noqa: E402  (never edited: signed_branch)

LABEL = B2.LABEL


def loewdin(Phi):
    G = Phi.conj().T @ Phi
    gw, gv = np.linalg.eigh(G)
    return Phi @ ((gv * (1.0 / np.sqrt(gw))[None, :]) @ gv.conj().T), float(np.max(np.abs(G - np.eye(G.shape[0]))))


def site_density(cols, f=None):
    w = np.abs(cols) ** 2
    if f is not None:
        w = w * np.asarray(f, float)[None, :]
    return w.sum(axis=1).reshape(-1, 2).sum(axis=1)                    # per site


def block_current(H, Xyx_fn, iX, reach):
    """-2 Im Tr(H_XY X_YX) over the rows within reach components inside the cut and the columns within reach outside it."""
    lo = max(2 * iX - reach, 0); hi = min(2 * iX + reach, H.shape[0])
    hxy = H[lo:2 * iX, 2 * iX:hi].toarray()
    Xyx = Xyx_fn(2 * iX, hi, lo, 2 * iX)
    return -2.0 * complex(np.sum(hxy * Xyx.T)).imag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", required=True); ap.add_argument("--fields", required=True); ap.add_argument("--spectra", required=True)
    ap.add_argument("--slices", required=True); ap.add_argument("--fields-module", default="engine_fields_v14")
    ap.add_argument("--kappas", default="1,2"); ap.add_argument("--win", type=int, default=32); ap.add_argument("--gpu", type=int, default=-1)
    ap.add_argument("--name", required=True); ap.add_argument("--out-json", required=True)
    a = ap.parse_args()
    if os.path.exists(a.out_json):
        sys.exit("refusing to overwrite %s" % a.out_json)
    dev = torch.device("cuda:%d" % a.gpu) if (a.gpu >= 0 and torch.cuda.is_available()) else torch.device("cpu")
    t_all = time.time()
    J = json.load(open(a.record)); Zf = np.load(a.fields, allow_pickle=True); Sp = np.load(a.spectra)
    ar = J["args"]; rs = float(ar["rs"]); g = float(ar["g"]); m = float(ar["m"]); ell = float(ar["ell"]); tau = float(Zf["tau"])
    FM = importlib.import_module(a.fields_module)
    recs = {round(float(r["T"]), 6): r for r in J["records"] if r.get("E") is not None}
    Tz = np.asarray(Zf["T"], float); Ts = np.asarray(Sp["T"], float)
    R = np.asarray(Zf["R"], float); n = len(R)
    out = {"script": "engine_single_object_slices_v1.py", "name": a.name, "record": a.record, "fields": a.fields, "spectra": a.spectra, "fields_module": a.fields_module,
           "label": LABEL, "no_mutual_information": "D15", "units": "rates per lattice time unit a", "tau": tau, "win": a.win, "slices": [],
           "utc_start": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())}

    def rec_at(T):
        return recs.get(round(float(T), 6), {})

    def eig_ref(k_sp, kap, N_, v_):
        """the reference of record k_sp rebuilt: eigenvectors of h_m at its stored applied fields picked by its stored mask (energy order)."""
        Hm = A2.hm_host(R, m, kap, N_, v_).toarray()
        e, U = torch.linalg.eigh(torch.as_tensor(Hm, dtype=torch.complex128, device=dev))
        bits = np.unpackbits(np.asarray(Sp["mask_%d" % kap][k_sp], np.uint8))[: 2 * n].astype(bool)
        e_st = np.asarray(Sp["e_%d" % kap][k_sp], float)
        en = e.real.cpu().numpy()
        Um = U[:, torch.as_tensor(np.nonzero(bits)[0], device=dev)].cpu().numpy()
        del U
        return Um, {"n_mask": int(bits.sum()), "eig_vs_stored_max": float(np.max(np.abs(en - e_st))), "mask_equals_negative": bool(np.array_equal(bits, en < 0))}

    for sp_ in a.slices.split(","):
        t_s = time.time()
        Z = np.load(sp_)
        T = float(Z["T"]); k = int(np.argmin(np.abs(Tz - T)))
        if abs(Tz[k] - T) > 1e-6:
            sys.exit("the slice's clock %.6f is not a record of %s" % (T, a.fields))
        N_app = np.asarray(Z["N_applied"], float); v_app = np.asarray(Z["v_applied"], float)
        fdiff = float(max(np.max(np.abs(N_app - Zf["N_applied"][k])), np.max(np.abs(v_app - Zf["v_applied"][k]))))
        rec = rec_at(T); recm = rec_at(Tz[k - 1]) if k > 0 else {}; recp = rec_at(Tz[k + 1]) if k + 1 < len(Tz) else {}
        _, _, _, info = FM.solve_constraints_ivp(np.asarray(Zf["eps_tot"][k], float), np.asarray(Zf["J_tot"][k], float), g, R, ell,
                                                 collar_policy=ar.get("collar_policy", "flag"), support_rel=ar.get("support_rel", 1e-14))
        rd = A2.roots_dense(info, ell) if not info.get("failed") else []
        Ro, Ri = A2.outer_inner(rd)
        E = float(rec["E"]); two_gE = 2.0 * g * E
        Em = float(recm["E"]) if recm.get("E") is not None else None; Ep = float(recp["E"]) if recp.get("E") is not None else None
        Rtm = recm.get("R_t"); Rtp = recp.get("R_t"); Rt_st = rec.get("R_t")
        dRdt = ((Rtp - Rtm) / (2.0 * tau)) if (Rtm and Rtp) else (((Rtp - Rt_st) / tau) if (Rtp and Rt_st) else (((Rt_st - Rtm) / tau) if (Rt_st and Rtm) else None))
        ks_m = int(np.argmin(np.abs(Ts - Tz[k - 1]))) if k > 0 else None; ks_p = int(np.argmin(np.abs(Ts - Tz[k + 1]))) if k + 1 < len(Tz) else None
        ks_0 = int(np.argmin(np.abs(Ts - T)))
        have_sp = (ks_m is not None and abs(Ts[ks_m] - Tz[k - 1]) < 1e-6 and ks_p is not None and abs(Ts[ks_p] - Tz[k + 1]) < 1e-6 and abs(Ts[ks_0] - T) < 1e-6)
        row = {"slice": sp_, "dtype": str(Z["sea_1"].dtype), "T_over_rs": T / rs, "k": k, "E": E, "two_gE": two_gE, "roots_dense": rd, "R_outer": Ro, "R_inner": Ri,
               "R_t_stored": Rt_st, "R_t_neighbours": [Rtm, Rtp], "dRt_dt": dRdt, "fields_vs_record_max": fdiff,
               "gauss_charge_loss_rate": (-(Ep - Em) / (2.0 * tau)) if (Em is not None and Ep is not None) else None, "E_neighbours": [Em, Ep],
               "spectra_rows": [ks_m, ks_0, ks_p] if have_sp else None, "channels": [], "dust_all_channels": []}
        print("%s: T/rs %.3f (record %d, %s); E %.6f, 2gE %.4f; dense roots %s; outer %s (stored %s); dR_t/dt %s; -dE/dt %s; spectra rows %s" % (
            os.path.basename(sp_), T / rs, k, row["dtype"], E, two_gE, ", ".join("%.4f (%+.5f)" % (q["R"], q["slope"]) for q in rd) or "none", B2.B1.fmt(Ro, "%.4f"),
            B2.B1.fmt(Rt_st, "%.4f"), B2.B1.fmt(dRdt, "%+.4e"), B2.B1.fmt(row["gauss_charge_loss_rate"], "%+.4e"), row["spectra_rows"]), flush=True)
        # the dust sector in every stored channel (weights only)
        wts = np.asarray(Z["weights"], float) if "weights" in Z.files else None
        for i_k, kap in enumerate([int(x) for x in Z["kappas"]]):
            dst = np.asarray(Z["dust_%d" % kap], np.complex128)
            if dst.shape[1] == 0:
                continue
            fd = np.asarray(Z["f_%d" % kap], float)
            rho = site_density(dst, fd)                                    # the stored columns as stored (no re-orthonormalization; their norms printed)
            row["dust_all_channels"].append({"kappa": kap, "weight": float(wts[i_k]) if wts is not None else None, "n_dust": int(dst.shape[1]), "f_sum": float(fd.sum()),
                                             "stored_norms_minus_1_max": float(np.max(np.abs((np.abs(dst) ** 2).sum(axis=0) - 1.0))),
                                             "beyond_Rt": float(rho[R > Ro].sum()) if Ro else None, "beyond_2gE": float(rho[R > two_gE].sum()), "beyond_16rs": float(rho[R > 16.0 * rs].sum())})
            del dst
        if wts is not None and row["dust_all_channels"]:
            row["dust_weighted_beyond_16rs"] = float(sum(q["weight"] * q["beyond_16rs"] for q in row["dust_all_channels"]))
        if Ro is None:
            row["note"] = "no root of the dense N + v: no cut"
            out["slices"].append(row); json.dump(out, open(a.out_json, "w"), indent=1, default=float)
            continue
        iX = int(np.searchsorted(R, Ro, side="left")); i2 = int(np.searchsorted(R, two_gE, side="left")); i16 = int(np.searchsorted(R, 16.0 * rs, side="left"))
        iXp = int(np.searchsorted(R, Rtp, side="left")) if Rtp else None
        for kap in [int(x) for x in a.kappas.split(",")]:
            t_c = time.time()
            sea = np.asarray(Z["sea_%d" % kap], np.complex128); dst = np.asarray(Z["dust_%d" % kap], np.complex128); fd = np.asarray(Z["f_%d" % kap], float)
            ns, nd = sea.shape[1], dst.shape[1]
            Phi = np.concatenate([sea, dst], axis=1); f = np.concatenate([np.ones(ns), fd])
            G = Phi.conj().T @ Phi; fh = np.sqrt(f)
            mu = np.linalg.eigvalsh((fh[:, None] * G) * fh[None, :]); purity = float(np.max(np.abs(mu * mu - mu)))
            Phi_o, st_orth = loewdin(Phi)
            P_o, ref_orth = loewdin(np.asarray(Z["R2_%d" % kap], np.complex128))
            nref = P_o.shape[1]
            Cf = Phi_o * np.sqrt(f)[None, :]
            rhoC = site_density(Cf); rhoP = site_density(P_o); rhoD_ = site_density(Cf[:, ns:]); rhoS = site_density(Cf[:, :ns])
            trC = float(f.sum()); trCs = float(ns)

            def beyond(i_cut):
                dd = {"D_direct": float(rhoC[i_cut:].sum() - rhoP[i_cut:].sum()),
                      "D_complement": float((trC - rhoC[:i_cut].sum()) - (nref - rhoP[:i_cut].sum())),
                      "dust": float(rhoD_[i_cut:].sum()),
                      "sea_minus_P_direct": float(rhoS[i_cut:].sum() - rhoP[i_cut:].sum()),
                      "sea_minus_P_complement": float((trCs - rhoS[:i_cut].sum()) - (nref - rhoP[:i_cut].sum()))}
                dd["routes_diff_D"] = dd["D_direct"] - dd["D_complement"]
                return dd

            det = {"beyond_Rt": beyond(iX), "beyond_2gE": beyond(i2), "beyond_16rs": beyond(i16)}
            Hm = A2.hm_host(R, m, kap, N_app, v_app)
            H2 = (Hm @ Hm).tocsr()

            def blk(cols, fw=None):
                def fn(r0, r1, c0, c1):
                    A_ = cols[r0:r1]; B_ = cols[c0:c1]
                    return (A_ * (np.asarray(fw, float)[None, :] if fw is not None else 1.0)) @ B_.conj().T
                return fn

            cur = {}
            for nm, (cols, fw) in {"C": (Phi_o, f), "P": (P_o, None), "dust": (Phi_o[:, ns:], fd)}.items():
                cur[nm] = {"number": block_current(Hm, blk(cols, fw), iX, 4), "killing_energy": 0.5 * block_current(H2, blk(cols, fw), iX, 10)}
            cur["D"] = {q: cur["C"][q] - cur["P"][q] for q in ("number", "killing_energy")}
            # the moving-cut reassignment
            def rho_at(rho):
                i1 = min(max(iX, 1), n - 1); i0 = i1 - 1
                return float(rho[i0] + (rho[i1] - rho[i0]) * (Ro - R[i0]) / (R[i1] - R[i0]))
            rhoDx = rhoC - rhoP
            rea = {nm: {"density_at_cut": rho_at(rr), "rate": (-rho_at(rr) * dRdt) if dRdt is not None else None,
                        "step_form": (float(rr[iXp:].sum() - rr[iX:].sum()) / tau) if iXp is not None else None}
                   for nm, rr in (("C", rhoC), ("P", rhoP), ("D", rhoDx), ("dust", rhoD_))}
            # the reference term
            refterm = None
            if have_sp:
                U0, i0_ = eig_ref(ks_0, kap, Zf["N_applied"][k], Zf["v_applied"][k])
                ov = np.linalg.norm(U0.conj().T @ P_o) ** 2
                dist0 = float(math.sqrt(max(2.0 * nref - 2.0 * ov, 0.0)))
                Um, im_ = eig_ref(ks_m, kap, Zf["N_applied"][k - 1], Zf["v_applied"][k - 1])
                Up, ip_ = eig_ref(ks_p, kap, Zf["N_applied"][k + 1], Zf["v_applied"][k + 1])
                yb = lambda U_: float(U_.shape[1] - (np.abs(U_[: 2 * iX]) ** 2).sum())
                tYm = yb(Um); tYp = yb(Up); tY0 = yb(U0)
                refterm = {"rate": (tYp - tYm) / (2.0 * tau), "TrPiY_P_km1": tYm, "TrPiY_P_k": tY0, "TrPiY_P_kp1": tYp,
                           "TrPiY_P_stored": float(nref - rhoP[:iX].sum()), "rebuild_vs_stored_projector_F": dist0,
                           "rebuild_info": {"k-1": im_, "k": i0_, "k+1": ip_}}
                del U0, Um, Up
            # the signed readout of the dust (and part A's unsigned share of its interior)
            Xd = torch.as_tensor(Phi_o[:, ns:] * np.sqrt(fd)[None, :], dtype=torch.complex128, device=dev)
            sg = B2.signed_inside_beyond(Xd, R, N_app, v_app, kap, m, Ro, a.win)
            unsigned_A = A2.momentum_content(Phi_o[: 2 * iX, ns:], iX)
            del Xd
            ch = {"kappa": kap, "n_sea": ns, "n_dust": nd, "n_ref": nref, "dust_occupations": fd.tolist(), "stored_purity_defect": purity,
                  "stored_orthonormality": st_orth, "ref_orthonormality_before_loewdin": ref_orth, "detector": det, "currents": cur, "reassignment": rea,
                  "reference_term": refterm, "signed_dust": sg, "unsigned_dust_interior_A": unsigned_A, "seconds": time.time() - t_c}
            row["channels"].append(ch)
            print("  kappa %d: purity %.1e; beyond R_t: D %s (direct - complement %.1e), dust %.4e, sea - P %s; number currents C %+.4e P %+.4e D %+.4e dust %+.4e; "
                  "reassignment rate D %s dust %s; reference term %s (rebuild vs stored %s); Killing flux D %+.4e dust %+.4e; dust + branch out/in inside R_t %s/%s; %.0f s" % (
                      kap, purity, B2.B1.fmt(det["beyond_Rt"]["D_complement"], "%+.4e"), det["beyond_Rt"]["routes_diff_D"], det["beyond_Rt"]["dust"],
                      B2.B1.fmt(det["beyond_Rt"]["sea_minus_P_complement"], "%+.4e"), cur["C"]["number"], cur["P"]["number"], cur["D"]["number"], cur["dust"]["number"],
                      B2.B1.fmt(rea["D"]["rate"], "%+.4e"), B2.B1.fmt(rea["dust"]["rate"], "%+.4e"), B2.B1.fmt((refterm or {}).get("rate"), "%+.4e"),
                      B2.B1.fmt((refterm or {}).get("rebuild_vs_stored_projector_F"), "%.1e"), cur["D"]["killing_energy"], cur["dust"]["killing_energy"],
                      B2.B1.fmt(((sg.get("inside_Rt") or {}).get("plus_out") or [None])[0] if sg.get("inside_Rt") else None, "%.4f"),
                      B2.B1.fmt(((sg.get("inside_Rt") or {}).get("plus_in") or [None])[0] if sg.get("inside_Rt") else None, "%.4f"), ch["seconds"]), flush=True)
            del sea, dst, Phi, Phi_o, P_o, Cf, Hm, H2
            if dev.type == "cuda":
                torch.cuda.empty_cache()
        row["seconds"] = time.time() - t_s
        out["slices"].append(row)
        json.dump(out, open(a.out_json, "w"), indent=1, default=float)
    out["utc_end"] = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()); out["seconds"] = time.time() - t_all
    json.dump(out, open(a.out_json, "w"), indent=1, default=float)
    print("done: %d slices in %.0f s" % (len(out["slices"]), out["seconds"]), flush=True)


if __name__ == "__main__":
    main()
