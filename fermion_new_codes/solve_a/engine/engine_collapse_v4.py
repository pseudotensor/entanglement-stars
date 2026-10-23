#!/usr/bin/env python
"""engine_collapse_v4.py (builder analyst; version 4 of engine_collapse_v3.py: the two kappa tests of SPEC_final4b_observables.md sec. 9.1
and B1 of sec. 10.2 on every re-solve with a surface: kappa_T = d_R(N + v) at the trapping surface R_t (the outermost zero of N - |v|) and the
second and third derivatives of N + v there, from the parity splines of the node values, against the exact profile's 1/(4 g E),
-3/(16 g^2 E^2) and 15/(64 g^3 E^3) at the solved E (printed as ratios), with 4 g E kappa_T (-> 1 when the surface sits in the analytic
exterior with E the collar charge); the inner zero r_in of N - |v| (the core's inner surface, sec. 10.2 (3)); the flag R_sh < R_t - 5 ell
(the late re-solves the spec reads); kappa_p - kappa_T comes from J3' on the run's history.  Version 3's docstring follows.)

engine_collapse_v3.py (builder analyst; the collapse of task 2 per SPEC_final4b_engine.md secs. 12.1, 12.3, 12.6, 13.1 and 13.3 on
engine_fields_v3, from the builder's engine_collapse_v2).

The model's dust under H_m[N(t), v(t)], quasi-static: the fields frozen over the coupling interval tau (the only time parameter; every record
says "quasi-static (tau = ...)"), the orbitals propagated exactly under the frozen h_m by its eigendecomposition. The reference is R2, the
continued sea (sec. 13.1): the n_sea eigenvectors of the current h_m with the largest weight w_k = ||P_R2(previous) phi_k||^2, never refilled;
the ambiguity monitor counts w in (0.1, 0.9) and records the smallest occupied w (BUILDER item of 13.1's halving rule: an ambiguous re-solve is
flagged; the halving is version 4's if the flag appears). --reference R1 runs the literal propagated sea (the sea is its own reference: zero
sea source by construction; the control at r_s = 32). The same-configuration iteration at every re-solve (sec. 12.6): the fields, the
eigendecomposition, the continued reference and the sources iterated with Anderson mixing to |dE/E| < 1e-9 (the count recorded).
Initial data (sec. 12.1): the levitating well U(r) = U_0 s(r) + m (1 - N_s(r)) (1 - s(r)), s the logistic wall at R_0 of width 4, N_s the
current iterate's static lapse (reading A: h_m[N, v] + U; reading B: h_m[N, 0] + U with N in place of N_s), zero-temperature filling of the
lowest N_B = r_s/(2 g m) levels (the last multiplet fractional), projected on the positive band of the final h_m[N, v], Loewdin; printed: the
discarded norm, the Gram condition number, the stationarity residual after the projection, the dust's momentum, the coldness
f_kin = sum f <h_D - m>/(N_B m) and its bulk-frame value (minus (1/2) sum eps u^2/(N_B m), u = J/eps); --shell R_in,R_out confines the dust between two
logistic walls (S-shell). Monitors per re-solve: E (collar); the dust's and the pairs' energies (the sources split); the trapping zeros of
N - |v| on the smooth fields, R_t against 2 g E and against 2 g M(R_t), E - M(R_t); the threshold radius per channel (|v|/N above
sqrt(1 + m_c^2) + m_c, m_c^2 = m^2 + kappa^2/R^2, massless: 1); n_pocket under R2 (occupied positive plus empty negative), n_neg(R0) - n_sea,
||R2 - R0||_F; the dust's R_sh (0.99) and centroid R_c with dR_c/dt against v(R_c); M1: the Killing-energy cut current of (C - R2) at the
collar, I_E = Tr[(C - R2) i[h_m, H_in]], H_in = (1/2){P_in, h_m}, beside -dE/dt; M9 every tenth re-solve at 0.5 R_sh, R_t and the collar:
the rate of W/(2g) from successive re-solves against the cut current there; the spectra and floors of the solver. Records: JSON per re-solve,
the fields npz with the node values, ell, tau, the reference and the dictionary line. --benchmark runs the initial data and one re-solve.
Usage: python engine_collapse_v3.py --m 0.2 --g 0.25 --rs 64 --R0-over-rs 4 --reading A --nsites 2048 --ell 4 --tau 4 --T-over-rs 40
--kappa-sea 10 --reference R2 --out <dir> --tag <tag> [--gpu 1]"""
import argparse
import json
import math
import os
import time

import numpy as np

import engine_ops_v1 as OPS
import engine_fields_v3 as FLD
import engine_mixing_v1 as MIX


def surface_taylor(info, Rt):
    """version 4: the first three derivatives of N + v at R_t on the smooth fields (the parity splines of the node values; N even, v odd);
    None without a surface."""
    if Rt is None or info.get("N_nodes") is None:
        return None
    r = info["r_nodes"]
    Ns = FLD.parity_spline(r, info["N_nodes"], +1); vs = FLD.parity_spline(r, info["v_nodes"], -1)
    return [float(Ns.derivative(q)(Rt) + vs.derivative(q)(Rt)) for q in (1, 2, 3)]


def logistic(R, R0, w=4.0):
    return 1.0 / (1.0 + np.exp(-(R - R0) / w))


import scipy.sparse as sp
import engine_vacflow_v1 as VF          # chain_sparse: the sparse chain identical to engine_ops_v1.chain_parts (checked)

BK = {"torch": None, "dev": None}


def backend(gpu):
    import torch
    BK["torch"] = torch
    BK["dev"] = torch.device("cuda:%d" % gpu) if gpu is not None else torch.device("cpu")


def T_(x):
    return BK["torch"].as_tensor(np.asarray(x), dtype=BK["torch"].complex128, device=BK["dev"])


class Chan:
    """the channel's unclocked chain h (the energy density's operator) and the two-step Q, sparse on the host and as torch CSR on the device."""

    def __init__(self, kappa, m, nsites, half=False):
        self.kappa = kappa
        self.w = 16.0 * kappa
        self.R = np.arange(nsites, dtype=float) + (0.5 if half else 1.0)
        self.hs, self.Qs = VF.chain_sparse(self.R, m, float(kappa))
        torch = BK["torch"]
        csr = lambda A: torch.sparse_csr_tensor(torch.as_tensor(A.indptr, dtype=torch.int64), torch.as_tensor(A.indices, dtype=torch.int64),
                                                torch.as_tensor(A.data, dtype=torch.complex128), size=A.shape, device=BK["dev"])
        self.h = csr(self.hs); self.Q = csr(self.Qs)

    def clocked_dense(self, N, v):
        Nd = sp.diags(np.repeat(np.asarray(N, float), 2)); Vd = sp.diags(np.repeat(np.asarray(v, float), 2))
        hm = 0.5 * (Nd @ self.hs + self.hs @ Nd) + 0.5 * (Vd @ self.Qs + self.Qs @ Vd)
        return T_(hm.toarray())


def eigT(hmT):
    e, U = BK["torch"].linalg.eigh(hmT)
    return e.real, U


def dens(A, Phi, f=None):
    """per site: sum over the columns of f Re(Phi^dag P_n A Phi) (A a device CSR matrix, Phi device columns), as a host array."""
    if Phi.shape[1] == 0:
        return np.zeros(Phi.shape[0] // 2)
    AP = A @ Phi
    if f is not None:
        AP = AP * T_(f)[None, :].real.to(AP.dtype)
    d = (Phi.conj() * AP).real.sum(dim=1)
    return d.reshape(-1, 2).sum(dim=1).cpu().numpy()


def fill_lowest(levels, NB, tol=1e-9):
    lv = sorted(levels)
    occ = {}
    left = float(NB); i = 0
    while i < len(lv) and left > 1e-12:
        j = i
        while j < len(lv) and abs(lv[j][0] - lv[i][0]) < tol:
            j += 1
        w = sum(x[1] for x in lv[i:j])
        frac = min(1.0, left / w)
        for x in lv[i:j]:
            occ[(x[2], x[3])] = frac
        left -= frac * w
        i = j
    return occ


def prepare_dust(chs, g, R, ell, NB, R0, m, reading, shell, passes=200, tol=1e-10):
    torch = BK["torch"]
    nsites = len(R)
    N = np.ones(nsites); v = np.zeros(nsites)
    mixer = MIX.Anderson()
    U0 = 0.5 * m
    for ip in range(passes):
        Ns = np.sqrt(np.maximum(N * N - v * v, 1e-12)) if reading == "A" else N.copy()
        if shell:
            s_in, s_out = logistic(R, shell[0]), logistic(R, shell[1])
            Uw = U0 * (1 - s_in) + U0 * s_out + m * (1 - Ns) * s_in * (1 - s_out)
        else:
            s = logistic(R, R0)
            Uw = U0 * s + m * (1 - Ns) * (1 - s)
        Ud = T_(np.diag(np.repeat(Uw, 2)))
        spec, levels = {}, []
        for ic, ch in enumerate(chs):
            hw = ch.clocked_dense(N, v if reading == "A" else 0 * v) + Ud
            ew, Uwv = eigT(hw)
            em, Um = eigT(ch.clocked_dense(N, v))
            pos = ew > 0
            ewp = ew[pos].cpu().numpy()
            spec[ch.kappa] = (ewp, Uwv[:, pos], Um[:, em > 0], hw)
            levels += [(float(x), ch.w, ic, j) for j, x in enumerate(ewp[:400])]
        occ = fill_lowest(levels, NB)
        eps = np.zeros(nsites); J = np.zeros(nsites); dust = {}; NBs = 0.0
        diag = {"discarded_norm_max": 0.0, "gram_cond_max": 1.0, "stationarity_max": 0.0}
        for ic, ch in enumerate(chs):
            ewp, Uwv, Pplus, hw = spec[ch.kappa]
            keep = [j for j in range(min(400, len(ewp))) if occ.get((ic, j), 0.0) > 0]
            if not keep:
                continue
            fk = np.array([occ[(ic, j)] for j in keep])
            raw = Pplus @ (Pplus.conj().T @ Uwv[:, keep])
            diag["discarded_norm_max"] = max(diag["discarded_norm_max"], float((1 - (raw.abs() ** 2).sum(dim=0)).max()))
            S = raw.conj().T @ raw
            sv, sU = torch.linalg.eigh(S)
            diag["gram_cond_max"] = max(diag["gram_cond_max"], float(sv.max() / sv.min().clamp_min(1e-300)))
            Phi = raw @ (sU * sv.clamp_min(1e-300).rsqrt()[None, :].to(sU.dtype)) @ sU.conj().T
            Ew = (Phi.conj() * (hw @ Phi)).sum(dim=0).real
            res = (hw @ Phi - Phi * Ew[None, :].to(Phi.dtype)).norm(dim=0)
            diag["stationarity_max"] = max(diag["stationarity_max"], float(res.max()))
            eps += ch.w * dens(ch.h, Phi, fk); J += ch.w * dens(ch.Q, Phi, fk); NBs += ch.w * fk.sum()
            dust[ch.kappa] = (Phi, fk)
        Nn, vn, Wn, info = FLD.solve_constraints(eps, J, g, R, ell, N0=N)
        if info.get("failed"):
            raise RuntimeError("initial data: the solver failed: %s" % info["failed"])
        d = max(float(np.max(np.abs(Nn - N))), float(np.max(np.abs(vn - v))))
        x = mixer.step(np.concatenate([N, v]), np.concatenate([Nn, vn]))
        N, v = x[:nsites], x[nsites:]
        if d < tol:
            break
    N, v, Wn, info = FLD.solve_constraints(eps, J, g, R, ell, N0=N)
    kin = 0.0
    for ch in chs:
        if ch.kappa in dust:
            Phi, fk = dust[ch.kappa]
            hD = (Phi.conj() * (ch.h @ Phi)).sum(dim=0).real.cpu().numpy()
            kin += ch.w * float(fk @ (hD - m))
    u = np.where(np.abs(eps) > 1e-14 * np.abs(eps).max(), J / np.where(np.abs(eps) > 0, eps, 1.0), 0.0)
    bulk = 0.5 * float(np.sum(eps * u * u))
    head = {"N_B": NBs, "passes": ip + 1, "last_change": d, "E0": FLD.charge(info, g), "f_kin": kin / (NBs * m), "f_kin_bulk_frame": (kin - bulk) / (NBs * m),
            "dust_momentum_total": float(J.sum()), "projection": diag, "reading": reading, "shell": shell}
    return N, v, dust, info, head


def continue_reference(ref_prev, U, n_sea):
    """R2: the n_sea current eigenvectors with the largest weight in the previous reference's span; the weights for the ambiguity monitor."""
    w = ((ref_prev.conj().T @ U).abs() ** 2).sum(dim=0)
    occ = BK["torch"].sort(BK["torch"].topk(w, n_sea).indices).values
    amb = int(((w > 0.1) & (w < 0.9)).sum())
    return occ, w, amb


def energy_cut_current(hmT, cols, f, cut, half=8):
    """I_E = Tr[D i[h_m, H_in]] with H_in = (1/2){P_in, h_m}; [h_m, H_in] = (1/2)[h_m^2, P_in] is local at the cut (h_m has range 2 sites),
    so the trace is taken on a window of +-half sites: D_W = c_W diag(f) c_W^dag, (h_m^2)_W = h_m[W, :] h_m[:, W]."""
    torch = BK["torch"]
    lo, hi = 2 * max(cut - half, 0), 2 * min(cut + half, hmT.shape[0] // 2)
    W = torch.arange(lo, hi, device=BK["dev"])
    h2W = hmT[W, :] @ hmT[:, W]
    cW = cols[W, :]
    DW = (cW * T_(f)[None, :]) @ cW.conj().T
    P = torch.zeros(hi - lo, dtype=torch.complex128, device=BK["dev"]); P[: 2 * cut - lo] = 1.0
    Cm = h2W * P[None, :] - P[:, None] * h2W
    return float((0.5j * (DW.T * Cm).sum()).real)      # = d<H_in>/dt (Heisenberg), Tr[D C] = sum(D^T * C)


def threshold_radii(N, v, R, m, kappas):
    out = {}
    ratio = np.abs(v) / np.maximum(N, 1e-300)
    for k in kappas:
        mc = np.sqrt(m * m + (k / R) ** 2)
        thr = np.sqrt(1 + mc * mc) + mc
        over = np.nonzero(ratio > thr)[0]
        out[str(k)] = float(R[over[-1]]) if len(over) else None
    return out


def run(a):
    torch = BK["torch"]
    os.makedirs(a.out, exist_ok=True)
    label = "quasi-static (tau = %g); ell = %d; reference: %s (%s); %s" % (a.tau, a.ell, a.reference, "continued sea" if a.reference == "R2" else "propagated sea", FLD.DICTIONARY)
    rs = a.rs
    NB = rs / (2.0 * a.g * a.m)
    chs = [Chan(k, a.m, a.nsites, bool(a.half_offset)) for k in range(1, a.kappa_sea + 1)]
    R = chs[0].R
    shell = [float(x) * rs for x in a.shell.split(",")] if a.shell else None
    t0 = time.time()
    N, v, dust, info, head = prepare_dust(chs, a.g, R, a.ell, NB, a.R0_over_rs * rs, a.m, a.reading, shell)
    head.update({"label": label, "rs_rest_label": rs, "R0": a.R0_over_rs * rs, "seconds": time.time() - t0})
    print("initial data (%s): N_B %.2f E0 %.6f passes %d f_kin %.4f (bulk frame %.4f) discarded %.2e Gram cond %.2e stationarity %.2e (%.0f s)" % (
        label, head["N_B"], head["E0"], head["passes"], head["f_kin"], head["f_kin_bulk_frame"], head["projection"]["discarded_norm_max"],
        head["projection"]["gram_cond_max"], head["projection"]["stationarity_max"], head["seconds"]), flush=True)
    E0 = head["E0"]
    st = {}
    for ch in chs:
        e, U = eigT(ch.clocked_dense(N, v))
        neg = e < 0
        Phi_d, f_d = dust.get(ch.kappa, (torch.zeros((2 * a.nsites, 0), dtype=torch.complex128, device=BK["dev"]), np.zeros(0)))
        st[ch.kappa] = {"sea": U[:, neg], "dust": Phi_d, "f": f_d, "ref": U[:, neg], "n_sea": int(neg.sum())}
    nsteps = int(round(a.T_over_rs * rs / a.tau))
    recs, fields = [], {"T": [], "N": [], "v": [], "r_nodes": [], "N_nodes": [], "v_nodes": []}
    path = os.path.join(a.out, "collapse_%s.json" % a.tag)
    T = 0.0
    prevW = None
    for step in range(nsteps + 1):
        tr = time.time()
        mixer = MIX.Anderson()
        E_prev = None
        converged = False
        for it in range(30):
            eps = np.zeros(a.nsites); J = np.zeros(a.nsites); eps_pairs = np.zeros(a.nsites); eps_dust = np.zeros(a.nsites)
            cache = {}
            amb_tot, wmin, npocket, nrefill, dR2R0 = 0, 1.0, {}, {}, 0.0
            for ch in chs:
                s_ = st[ch.kappa]
                hm = ch.clocked_dense(N, v)
                e, U = eigT(hm)
                if a.reference == "R2":
                    occ, w, amb = continue_reference(s_["ref"], U, s_["n_sea"])
                    ref = U[:, occ]
                    amb_tot += amb; wmin = min(wmin, float(w[occ].min()))
                    mask = torch.zeros(len(e), dtype=torch.bool, device=BK["dev"]); mask[occ] = True
                    npocket[ch.kappa] = int((e[mask] > 0).sum()) + int((e[~mask] < 0).sum())
                    nneg = int((e < 0).sum())
                    nrefill[ch.kappa] = nneg - s_["n_sea"]
                    ov = float(((U[:, e < 0].conj().T @ ref).abs() ** 2).sum())
                    dR2R0 += math.sqrt(max(nneg + s_["n_sea"] - 2 * ov, 0.0))
                else:
                    ref = s_["sea"]
                cache[ch.kappa] = (hm, e, U, ref)
                ds = dens(ch.h, s_["sea"]) - dens(ch.h, ref); Js = dens(ch.Q, s_["sea"]) - dens(ch.Q, ref)
                dd = dens(ch.h, s_["dust"], s_["f"]); Jd = dens(ch.Q, s_["dust"], s_["f"])
                eps += ch.w * (ds + dd); J += ch.w * (Js + Jd); eps_pairs += ch.w * ds; eps_dust += ch.w * dd
            Nn, vn, Wn, info = FLD.solve_constraints(eps, J, a.g, R, a.ell, N0=N)
            if info.get("failed"):
                break
            E = FLD.charge(info, a.g)
            dE = abs(E - E_prev) / abs(E) if E_prev is not None else np.inf
            E_prev = E
            if dE < 1e-9:
                N, v = Nn, vn
                converged = True
                break
            x = mixer.step(np.concatenate([N, v]), np.concatenate([Nn, vn]))
            N, v = x[:a.nsites], x[a.nsites:]
        if info.get("failed"):
            recs.append({"T": T, "failed": info["failed"], "label": label})
            print("T/rs %.2f: the solver FAILED (%s); the run stops with the record written" % (T / rs, info["failed"]), flush=True)
            break
        if not converged:
            N, v = Nn, vn
        for ch in chs:
            st[ch.kappa]["ref"] = cache[ch.kappa][3]
        tz = FLD.trapping_radius(N, v, R, info)
        Rt = tz["zeros"][-1] if tz["zeros"] else None
        tay = surface_taylor(info, Rt)
        kT = tay[0] if tay else None
        r_in = tz["zeros"][0] if len(tz["zeros"]) >= 2 else None
        Mt = FLD.mass_function(info, a.g, Rt) if Rt else None
        Rc = Rsh = None
        tot = eps_dust.sum()
        if tot > 0:
            cum = np.cumsum(eps_dust); Rsh = float(R[min(int(np.searchsorted(cum, 0.99 * tot)), a.nsites - 1)]); Rc = float(eps_dust @ R / tot)
        cut = int(min(max(round(info["r_collar"]) - 1, 2), a.nsites - 10))
        IE = 0.0
        for ch in chs:
            hm, e, U, ref = cache[ch.kappa]
            s_ = st[ch.kappa]
            cols = torch.cat([s_["sea"], s_["dust"], ref], dim=1)
            fs = np.concatenate([np.ones(s_["sea"].shape[1]), s_["f"], -np.ones(ref.shape[1])])
            IE += ch.w * energy_cut_current(hm, cols, fs, cut)
        rec = {"T": T, "T_over_rs": T / rs, "E": E, "E_over_E0": E / E0, "E_dust_shells": float(eps_dust.sum()), "E_pairs_shells": float(eps_pairs.sum()),
               "pairs_inside_Rt": (float(eps_pairs[R < Rt].sum()) if Rt else None), "pairs_outside_Rt": (float(eps_pairs[R >= Rt].sum()) if Rt else float(eps_pairs.sum())),
               "R_t": Rt, "R_t_over_2gE": (Rt / (2 * a.g * E) if Rt else None), "M_Rt": Mt, "R_t_over_2gM": (Rt / (2 * a.g * Mt) if (Rt and Mt) else None),
               "E_minus_M_Rt": (E - Mt if Mt else None), "kappa_T": kT, "four_gE_kappa_T": (4.0 * a.g * E * kT if kT is not None else None),
               "surface_derivatives": tay, "surface_derivatives_over_exact": ([tay[0] * 4 * a.g * E, tay[1] / (-3.0 / (16 * (a.g * E) ** 2)), tay[2] / (15.0 / (64 * (a.g * E) ** 3))] if tay else None),
               "r_in": r_in, "trapping_zeros": tz["zeros"], "late_surface": bool(Rsh is not None and Rt is not None and Rsh < Rt - 5 * a.ell), "min_N_minus_absv": tz["min"], "r_min_N_minus_absv": tz["r_min"], "N_min": float(N.min()),
               "R_sh": Rsh, "R_c": Rc, "v_at_Rc": (float(np.interp(Rc, R, v)) if Rc else None), "threshold_radius": threshold_radii(N, v, R, a.m, [1, 2, a.kappa_sea]),
               "n_pocket": npocket, "n_refill_R0": nrefill, "R2_minus_R0_F": dR2R0, "ambiguous_weights": amb_tot, "min_occupied_weight": wmin,
               "M1_energy_cut_current": IE, "iterations": it + 1, "converged": converged, "spectrum": info["spectrum"], "r_collar": info["r_collar"],
               "label": label, "seconds": time.time() - tr}
        if recs and Rc is not None and recs[-1].get("R_c") is not None:
            rec["dRc_dT"] = (Rc - recs[-1]["R_c"]) / (T - recs[-1]["T"])
            rec["dE_dT"] = (E - recs[-1]["E"]) / (T - recs[-1]["T"])
        if step % 10 == 0 and prevW is not None and T > prevW[1]:
            rec["M9"] = {}
            for lab, rr in (("0.5 R_sh", 0.5 * Rsh if Rsh else None), ("R_t", Rt), ("collar", float(info["r_collar"]))):
                if rr:
                    rec["M9"][lab] = {"r": rr, "dW_over_2g_dT": float((info["W_dense"](rr)[0] - prevW[0](rr)[0]) / (2 * a.g * (T - prevW[1])))}
        if step % 10 == 0:
            prevW = (info["W_dense"], T)
        recs.append(rec)
        fields["T"].append(T); fields["N"].append(N.copy()); fields["v"].append(v.copy())
        fields["r_nodes"].append(info["r_nodes"]); fields["N_nodes"].append(info["N_nodes"]); fields["v_nodes"].append(info["v_nodes"])
        rat = rec["surface_derivatives_over_exact"]
        print("T/rs %.2f E/E0 %.6f dust %.4f pairs %.3e R_t %s (2gE %.3f, 2gM %s; d1 d2 d3 of N+v over exact %s; r_in %s) min(N-|v|) %.4f R_sh %s R_c %s pockets %s amb %d it %d M1 %.3e (%.0f s)" % (
            T / rs, E / E0, eps_dust.sum(), eps_pairs.sum(), ("%.3f" % Rt) if Rt else None, 2 * a.g * E, ("%.3f" % (2 * a.g * Mt)) if Mt else None,
            ("%.5f, %.4f, %.4f" % tuple(rat)) if rat else None, ("%.3f" % r_in) if r_in else None, tz["min"],
            Rsh, ("%.2f" % Rc) if Rc else None, sum(npocket.values()) if npocket else 0, amb_tot, it + 1, IE, time.time() - tr), flush=True)
        if step % 5 == 0 or step == nsteps:
            with open(path, "w") as fh:
                json.dump({"script": "engine_collapse_v4.py", "args": vars(a), "head": head, "records": recs}, fh, default=float)
        if a.benchmark or step >= nsteps:
            break
        for ch in chs:
            hm, e, U, ref = cache[ch.kappa]
            ph = torch.exp(-1j * e.to(torch.complex128) * a.tau)
            s_ = st[ch.kappa]
            prop = lambda X: U @ (ph[:, None] * (U.conj().T @ X))
            s_["sea"] = prop(s_["sea"])
            if s_["dust"].shape[1]:
                s_["dust"] = prop(s_["dust"])
        T += a.tau
    with open(path, "w") as fh:
        json.dump({"script": "engine_collapse_v4.py", "args": vars(a), "head": head, "records": recs}, fh, default=float)
    np.savez_compressed(os.path.join(a.out, "collapse_%s_fields.npz" % a.tag), R=R, T=np.array(fields["T"]), N=np.array(fields["N"]), v=np.array(fields["v"]),
                        r_nodes=np.array(fields["r_nodes"], dtype=object), N_nodes=np.array(fields["N_nodes"], dtype=object),
                        v_nodes=np.array(fields["v_nodes"], dtype=object), ell=a.ell, tau=a.tau, label=label)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=float, default=0.2)
    ap.add_argument("--g", type=float, default=0.25)
    ap.add_argument("--rs", type=float, default=64.0)
    ap.add_argument("--R0-over-rs", type=float, default=4.0)
    ap.add_argument("--reading", default="A", choices=["A", "B"])
    ap.add_argument("--shell", default=None, help="R_in,R_out in units of r_s (S-shell)")
    ap.add_argument("--nsites", type=int, default=2048)
    ap.add_argument("--ell", type=int, default=4)
    ap.add_argument("--kappa-sea", type=int, default=10)
    ap.add_argument("--tau", type=float, default=4.0)
    ap.add_argument("--T-over-rs", type=float, default=40.0)
    ap.add_argument("--reference", default="R2", choices=["R2", "R1"])
    ap.add_argument("--half-offset", type=int, default=0)
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    backend(a.gpu)
    run(a)


if __name__ == "__main__":
    main()
