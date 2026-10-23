#!/usr/bin/env python
"""build_engine_collapse_v36f.py (reviewer A for the busi lane; WEB40I review, busi W2 and code defect 1): writes engine_collapse_v36f.py
from engine_collapse_v36d.py (input 05) by 17 exact string replacements, the form of build_engine_collapse_v36e.py (input 11). Each OLD is
copied exactly from engine_collapse_v36d.py and must occur exactly once (else the build aborts); each NEW contains inserted text so that
replacing every NEW by its OLD restores the source byte for byte (checked here). Every inserted or changed code line carries the marker
"v36f dg" (a trailing comment; the docstring's lines carry "[v36f dg]"). The option --step {frozen, dg} is "frozen" by default: with it
nothing of the inserted code runs and the json's "args" dict alone gains the keys step, dg_tol and dg_cap, so v36f --step frozen is v36d bitwise
in every record number, printed line and stored array (the gate: engine_collapse_v36f_selftest.py --fixture, check (i)).
Usage: python build_engine_collapse_v36f.py [engine_collapse_v36d.py path] [engine_collapse_v36f.py path]   (defaults: beside this script)"""
import hashlib
import os
import py_compile
import sys

REPLACEMENTS = [
("docstring",
r'''"""engine_collapse_v36d.py (the solve lane's analyst builder 11; engine spec 53.1 (1), (2)): version 36d = version 36c with one change''',
r'''"""engine_collapse_v36f.py (reviewer A for the busi lane; WEB40I review, busi W2 and code defect 1): version 36f = version 36d with one option, off by default, [v36f dg]
so that the defaults are version 36d bitwise in every record number, every printed line and every stored array (the json's "args" dict gains the 3 keys below): [v36f dg]
--step frozen (the default): version 36d's step, the state propagated over tau by the exponential of h_m at the record's accepted fields. [v36f dg]
--step dg [--dg-tol 1e-12] [--dg-cap 30]: the discrete-gradient exponential step. From the record at T (the accepted fields (N_app, v_app), the continued reference P_0, the [v36f dg]
    physical covariance C_0 of the sea's and the dust's columns): C_1 = exp(-i tau H-bar) C_0 exp(i tau H-bar) with H-bar = integral_0^1 h[C_0 + s (C_1 - C_0)] ds by 3-node [v36f dg]
    Gauss-Legendre (s = 1/2 -+ sqrt(15)/10 and 1/2, weights 5/18, 8/18, 5/18). What h depends on: h[C] = h_m[N, v] = (1/2){N, h} + (1/2){v, Q} (clocked_gpu) with (N, v) the [v36f dg]
    field solve's back-averaged site fields at the sources of C, eps = sum_kappa 16 kappa (dens(h, C) - dens(h, P[N, v])) and J likewise with Q, P[N, v] the reference continued from [v36f dg]
    the record's accepted entry to (N, v) (continue_ref_v34): h[C] is the quasi-static fixed point in (N, v) at fixed C, as the record's own solve is. How the field solve enters the [v36f dg]
    3 nodes: the site densities of C_0 + s_k (C_1 - C_0) are (1 - s_k) base_0 + s_k base_1 (linear in the covariance; no node covariance is formed); at each node the trial fields [v36f dg]
    (N_k, v_k) give the eigendecompositions of h_m[N_k, v_k] per channel, the reference continued to them, the node's sources (the node's sea densities less the continued [v36f dg]
    reference's, the dust's added, each channel weighted by 16 kappa) and one constraint solve whose back-averaged site fields are the node's new (N_k, v_k); h_m being linear [v36f dg]
    in the fields, H-bar = h_m[sum_k w_k N_k, sum_k w_k v_k] and its exponential is the engine's own propagator (the eigendecomposition of H-bar, the phases exp(-i tau e)), never [v36f dg]
    a Cayley form. The fixed point is iterated jointly in the node fields (the start: the previous step's node fields extrapolated by one record, else the accepted fields at [v36f dg]
    every node; --mix plain or anderson on them), C_1 re-propagated from C_0 with each H-bar, until the iterate's energy defect Tr((H-bar_k - H-bar_(k-1))(C_1 - C_0)) = [v36f dg]
    sum_n [(N-bar_k - N-bar_(k-1))_n (eps_1 - eps_0)_n + (v-bar_k - v-bar_(k-1))_n (J_1 - J_0)_n], eps and J the physical covariance's weighted site densities (the dust's alone [v36f dg]
    under R1), is below --dg-tol of the record's charge and the node fields change by less than --stop-fields; at most --dg-cap iterations, else the run ends at the record [v36f dg]
    (the stop line; the json's end_v36f). The identity behind the step: E[C_1] - E[C_0] = integral_0^1 Tr(h_s dC/ds) ds = Tr(H-bar (C_1 - C_0)) exactly when the back-averaging [v36f dg]
    is the weighted adjoint of the source averaging (dE = Tr(h dC); engine_adjoint_test_v1) and the reference is spectral along the path (Tr(h P-dot) = 0), and Tr(H-bar C_1) = [v36f dg]
    Tr(H-bar C_0) for the exponential of H-bar itself, so the step conserves the collar charge to the quadrature's order and the fixed point's tolerance; the frozen step's drift [v36f dg]
    (tau^2/2) D^2E[F, F] per record is gone. Per record the json gets "step": "dg", "dg_v36f" (the step that produced the record's state: the iteration count, the final energy [v36f dg]
    defect and the change between iterates, the node fields' change, the node charges, E_pred = E_prev + defect, and this record's E against E_0, the previous record and E_pred), [v36f dg]
    "far_dust_v36f" (the dust beyond 16 and 24 r_s on the record's held operator: its number, its energy by the density route and its Killing split as spectral weights of the [v36f dg]
    exterior block below 0, in [0, m), in [m, b] and above b = sqrt(m^2 + 1/a^2)) and the handoff file the dust number density per site (n_dust, the fixed-cut ledger's input); [v36f dg]
    the slices carry the step's node fields (dg_nodes_N, dg_nodes_v) so that --resume-slice restarts the step from the same start (bitwise on the same device); --resume-from [v36f dg]
    (the v17/v19 replay) is refused under --step dg. Every changed code line of this file is marked "v36f dg". [v36f dg]
Version 36d's docstring follows. [v36f dg]
engine_collapse_v36d.py (the solve lane's analyst builder 11; engine spec 53.1 (1), (2)): version 36d = version 36c with one change'''),
("module state",
r'''NSR = {"on": False, "prev_sig": None, "rec": None, "prev_m": None, "other": None}   # version 36c (--nk-set resolve); 36d: prev_m, other''',
r'''NSR = {"on": False, "prev_sig": None, "rec": None, "prev_m": None, "other": None}   # version 36c (--nk-set resolve); 36d: prev_m, other
GL3_V36F = ((0.5 - math.sqrt(15.0) / 10.0, 5.0 / 18.0), (0.5, 4.0 / 9.0), (0.5 + math.sqrt(15.0) / 10.0, 5.0 / 18.0))   # v36f dg: the 3-node Gauss-Legendre rule on [0, 1] (nodes, weights)
DG = {"on": False, "tol": 1e-12, "cap": 30, "rec": None, "nodes": None, "steps": 0, "collars": (16.0, 24.0)}   # v36f dg: the discrete-gradient step's state (--step dg)'''),
("functions before run",
r'''def run(a):
    torch = BK["torch"]
    os.makedirs(a.out, exist_ok=True)''',
r'''def dg_desc_v36f(a):   # v36f dg
    """version 36f: --step dg in words (the run label, the record's engine_v36f entry)."""   # v36f dg
    return ("the discrete-gradient exponential step: C_1 = exp(-i tau H-bar) C_0 exp(i tau H-bar) with H-bar = h_m[N-bar, v-bar], the 3-node Gauss-Legendre average of the node "   # v36f dg
            "fields over s in [0, 1] (nodes %.6f, 0.5, %.6f; weights 5/18, 8/18, 5/18), each node's fields the quasi-static solve at the sources of C_0 + s (C_1 - C_0) (site densities "   # v36f dg
            "linear in the covariance) with the reference continued from the record's accepted entry to the node's fields; the fixed point in the node fields iterated (%s) until the "   # v36f dg
            "energy defect Tr((H-bar_k - H-bar_(k-1))(C_1 - C_0)) is below %g of the record's charge and the node fields change by less than %g, at most %d iterations (else the run ends "   # v36f dg
            "at the record); the exponential by the engine's own propagator (the eigendecomposition of H-bar), never a Cayley form") % (   # v36f dg
        GL3_V36F[0][0], GL3_V36F[2][0], a.mix, DG["tol"], a.stop_fields, DG["cap"])   # v36f dg


def dg_lagrange_v36f(t):   # v36f dg
    """version 36f: the quadratic Lagrange weights of the 3 Gauss nodes at t (the node fields extrapolated by one record: t = 1 + s_k)."""   # v36f dg
    s = [q[0] for q in GL3_V36F]   # v36f dg
    return [float(np.prod([(t - s[m_]) / (s[j] - s[m_]) for m_ in range(3) if m_ != j])) for j in range(3)]   # v36f dg


def dg_densities_v36f(chs, st, cols):   # v36f dg
    """version 36f: the per-channel site densities (dens(h, sea), dens(Q, sea), dens(h, dust, f), dens(Q, dust, f)) of the columns cols[kappa] = (sea, dust) with st's occupations."""   # v36f dg
    out = {}   # v36f dg
    for ch in chs:   # v36f dg
        sea, dust = cols[ch.kappa]   # v36f dg
        out[ch.kappa] = (dens(ch.h, sea), dens(ch.Q, sea), dens(ch.h, dust, st[ch.kappa]["f"]), dens(ch.Q, dust, st[ch.kappa]["f"]))   # v36f dg
    return out   # v36f dg


def dg_pairing_v36f(chs, base_, dust_only):   # v36f dg
    """version 36f: the physical covariance's weighted site densities for the pairing Tr(h_m dC) = sum_n (N_n d eps_n + v_n d J_n): the sea's and the dust's under R2 (the continued reference's change pairs to zero, Tr(h P-dot) = 0), the dust's alone under R1 (the propagated sea is its own reference)."""   # v36f dg
    eps = None; J = None   # v36f dg
    for ch in chs:   # v36f dg
        hs_, Qs_, dd, Jd = base_[ch.kappa]   # v36f dg
        e_ = ch.w * (dd if dust_only else hs_ + dd); j_ = ch.w * (Jd if dust_only else Qs_ + Jd)   # v36f dg
        eps = e_ if eps is None else eps + e_; J = j_ if J is None else J + j_   # v36f dg
    return np.asarray(eps, float), np.asarray(J, float)   # v36f dg


def dg_node_v36f(a, chs, st, R, base_s, Nk, vk):   # v36f dg
    """version 36f: one evaluation of a Gauss node's map at the trial fields (Nk, vk): the eigendecompositions of h_m[Nk, vk], the reference continued from the record's accepted entry (the window's last) to them, the node's sources (the node's base densities less the continued reference's, the dust's added, each channel weighted by 16 kappa), one field solve (the site fields by OPT, averaged in the production). Returns (N_out, v_out, E_node, info, cont)."""   # v36f dg
    n = len(R)   # v36f dg
    eig = eigh_all(chs, Nk, vk, bool(a.streams))   # v36f dg
    eps = np.zeros(n); J = np.zeros(n); cont = {}   # v36f dg
    for ch in chs:   # v36f dg
        e, U = eig.pop(ch.kappa)   # v36f dg
        hs_, Qs_, dd, Jd = base_s[ch.kappa]   # v36f dg
        if a.reference == "R2":   # v36f dg
            s_ = st[ch.kappa]   # v36f dg
            ref, cinfo, mask = continue_ref_v34(a, ch, s_["window"], (Nk, vk), e, U, s_["n_sea"])   # v36f dg
            ds = hs_ - dens(ch.h, ref); Js = Qs_ - dens(ch.Q, ref)   # v36f dg
            cont[str(ch.kappa)] = {"verified": cinfo.get("verified"), "nsub": cinfo.get("nsub"), "n_pocket": cinfo.get("n_pocket"), "min_occupied_weight": cinfo.get("min_occupied_weight"),   # v36f dg
                                   "mask_equals_negative_set": bool(np.array_equal(np.asarray(mask, bool), (e < 0).cpu().numpy()))}   # v36f dg
            del ref   # v36f dg
        else:   # v36f dg
            ds = 0.0; Js = 0.0; cont[str(ch.kappa)] = {"mode": "R1"}   # v36f dg
        eps += ch.w * (ds + dd); J += ch.w * (Js + Jd)   # v36f dg
        del e, U   # v36f dg
    del eig   # v36f dg
    Nn, vn, Wn, info = solve(eps, J, a.g, R, a.ell, None)   # v36f dg
    if info.get("failed"):   # v36f dg
        return None, None, None, info, cont   # v36f dg
    return np.asarray(Nn, float), np.asarray(vn, float), float(FLD.charge(info, a.g)), info, cont   # v36f dg


def dg_step_v36f(a, chs, st, R, base0, f_acc, E_rec, T, rs):   # v36f dg
    """version 36f: the discrete-gradient exponential step from the record at T to T + tau (the docstring): st holds the record's columns C_0 and the accepted entry of the reference window, base0 their site densities, f_acc the accepted fields (N_app, v_app), E_rec the record's collar charge. The node fields are iterated until the iterate's energy defect Tr((H-bar_k - H-bar_(k-1))(C_1 - C_0)) is below DG["tol"] of E_rec and the node fields change by less than --stop-fields (at most DG["cap"] iterations); on convergence st's columns are replaced by C_1 and DG["nodes"] by the node fields. Returns the step's entry (the next record's dg_v36f)."""   # v36f dg
    torch = BK["torch"]   # v36f dg
    t0 = time.time()   # v36f dg
    if "eigh_n" not in TM:   # v36f dg
        tm_reset()   # v36f dg
    n_eigh0 = TM["eigh_n"]; n = len(R)   # v36f dg
    s_nodes = [q[0] for q in GL3_V36F]; w_nodes = [q[1] for q in GL3_V36F]   # v36f dg
    cols0 = {ch.kappa: (st[ch.kappa]["sea"], st[ch.kappa]["dust"]) for ch in chs}   # v36f dg
    dust_only = (a.reference != "R2")   # v36f dg
    eps0, J0 = dg_pairing_v36f(chs, base0, dust_only)   # v36f dg
    x = [(np.array(f_acc[0], float), np.array(f_acc[1], float)) for _ in range(3)]   # v36f dg
    start = "the record's accepted fields at every node (no previous step's node fields)"   # v36f dg
    if DG["nodes"] is not None:   # v36f dg
        pN, pv = DG["nodes"]; xe = []; ok = True   # v36f dg
        for k in range(3):   # v36f dg
            L = dg_lagrange_v36f(1.0 + s_nodes[k])   # v36f dg
            Ne = L[0] * pN[0] + L[1] * pN[1] + L[2] * pN[2]; ve = L[0] * pv[0] + L[1] * pv[1] + L[2] * pv[2]   # v36f dg
            ok = ok and bool(np.all(np.isfinite(Ne)) and np.all(np.isfinite(ve)) and np.max(np.abs(Ne - f_acc[0])) < 0.5 and np.max(np.abs(ve - f_acc[1])) < 0.5)   # v36f dg
            xe.append((np.asarray(Ne, float), np.asarray(ve, float)))   # v36f dg
        if ok:   # v36f dg
            x = xe; start = "the previous step's node fields extrapolated by one record (quadratic Lagrange through its 3 nodes)"   # v36f dg
    mixer = MIX.Anderson(beta=1.0) if a.mix == "anderson" else None   # v36f dg
    hist = []; conv = False; why = None; n_solves = 0; dE = None; dE_prev = None; dfields = None; cols1 = None; x_new = None; nodeE = None; conts = None; nres = None   # v36f dg
    tol = DG["tol"] * max(abs(E_rec), 1e-300)   # v36f dg
    for it in range(1, DG["cap"] + 1):   # v36f dg
        Nbar = w_nodes[0] * x[0][0] + w_nodes[1] * x[1][0] + w_nodes[2] * x[2][0]; vbar = w_nodes[0] * x[0][1] + w_nodes[1] * x[1][1] + w_nodes[2] * x[2][1]   # v36f dg
        eig = eigh_all(chs, Nbar, vbar, bool(a.streams))   # v36f dg
        cols1 = {}   # v36f dg
        for ch in chs:   # v36f dg
            e_b, U_b = eig.pop(ch.kappa)   # v36f dg
            ph = torch.exp(-1j * e_b.to(torch.complex128) * a.tau)   # v36f dg
            prop = lambda X, U_b=U_b, ph=ph: U_b @ (ph[:, None] * (U_b.conj().T @ X))   # v36f dg
            sea0, dust0 = cols0[ch.kappa]   # v36f dg
            cols1[ch.kappa] = (prop(sea0), prop(dust0) if dust0.shape[1] else dust0)   # v36f dg
            del e_b, U_b   # v36f dg
        del eig   # v36f dg
        base1 = dg_densities_v36f(chs, st, cols1)   # v36f dg
        eps1, J1 = dg_pairing_v36f(chs, base1, dust_only)   # v36f dg
        x_new = []; nodeE = []; conts = []; nres = []   # v36f dg
        for k in range(3):   # v36f dg
            s_k = s_nodes[k]   # v36f dg
            base_s = {ch.kappa: tuple((1.0 - s_k) * b0 + s_k * b1 for b0, b1 in zip(base0[ch.kappa], base1[ch.kappa])) for ch in chs}   # v36f dg
            Nk, vk, Ek, info_k, cont_k = dg_node_v36f(a, chs, st, R, base_s, x[k][0], x[k][1])   # v36f dg
            n_solves += 1   # v36f dg
            if Nk is None:   # v36f dg
                why = "the field solve failed at node %d (s = %.6f) of iteration %d: %s" % (k + 1, s_k, it, info_k.get("failed"))   # v36f dg
                break   # v36f dg
            x_new.append((Nk, vk)); nodeE.append(Ek); conts.append(cont_k)   # v36f dg
            nres.append(max(float(np.max(np.abs(Nk - x[k][0]))), float(np.max(np.abs(vk - x[k][1])))))   # v36f dg
        if why is not None:   # v36f dg
            break   # v36f dg
        Nbar_n = w_nodes[0] * x_new[0][0] + w_nodes[1] * x_new[1][0] + w_nodes[2] * x_new[2][0]; vbar_n = w_nodes[0] * x_new[0][1] + w_nodes[1] * x_new[1][1] + w_nodes[2] * x_new[2][1]   # v36f dg
        dE_prev = dE   # v36f dg
        dE = float(np.sum((Nbar_n - Nbar) * (eps1 - eps0) + (vbar_n - vbar) * (J1 - J0)))   # v36f dg
        dfields = max(nres)   # v36f dg
        chg = (abs(dE - dE_prev) / abs(E_rec)) if (dE_prev is not None and E_rec) else None   # v36f dg
        hist.append([it, dE / E_rec, chg, dfields] + [q / E_rec for q in nodeE])   # v36f dg
        print("  v36f dg T/rs %.4f iteration %d: energy defect %+.3e of E (change between iterates %s), node fields' change %.2e [%s], node E/E_rec - 1 [%s]" % (   # v36f dg
            T / rs, it, dE / E_rec, fmt(chg, "%.3e"), dfields, " ".join("%.1e" % q for q in nres), " ".join("%+.3e" % (q / E_rec - 1.0) for q in nodeE)), flush=True)   # v36f dg
        if abs(dE) < tol and dfields < a.stop_fields:   # v36f dg
            conv = True   # v36f dg
            break   # v36f dg
        if mixer is not None:   # v36f dg
            xm = mixer.step(np.concatenate([np.concatenate([q[0], q[1]]) for q in x]), np.concatenate([np.concatenate([q[0], q[1]]) for q in x_new]))   # v36f dg
            x = [(xm[2 * n * k: 2 * n * k + n].copy(), xm[2 * n * k + n: 2 * n * (k + 1)].copy()) for k in range(3)]   # v36f dg
        else:   # v36f dg
            x = x_new   # v36f dg
    ent = {"from_T": T, "from_T_over_rs": T / rs, "to_T_over_rs": (T + a.tau) / rs, "tau": a.tau, "converged": conv, "iterations": len(hist), "why": why, "start": start,   # v36f dg
           "gauss_nodes": s_nodes, "gauss_weights": w_nodes, "tolerance_energy_relative": DG["tol"], "tolerance_fields": a.stop_fields, "cap": DG["cap"], "mix": a.mix,   # v36f dg
           "energy_defect_final": ((dE / E_rec) if dE is not None else None), "energy_change_between_iterates_final": (hist[-1][2] if hist else None),   # v36f dg
           "fields_change_final": dfields, "node_residuals_final": nres, "node_E": nodeE, "node_E_over_E_rec": ([q / E_rec for q in nodeE] if nodeE else None), "E_rec": E_rec,   # v36f dg
           "E_pred": ((E_rec + dE) if dE is not None else None), "continuation_nodes_final": conts, "history": hist, "eigendecompositions": TM["eigh_n"] - n_eigh0, "solves": n_solves}   # v36f dg
    if conv:   # v36f dg
        for ch in chs:   # v36f dg
            st[ch.kappa]["sea"], st[ch.kappa]["dust"] = cols1[ch.kappa]   # v36f dg
        DG["nodes"] = (np.stack([q[0] for q in x_new]), np.stack([q[1] for q in x_new])); DG["steps"] += 1   # v36f dg
    sync_dev()   # v36f dg
    ent["seconds"] = time.time() - t0   # v36f dg
    print("T/rs %.4f v36f dg step to T/rs %.4f: %s in %d iterations; energy defect %s of E (change between iterates %s), node fields %s; E_pred/E_rec - 1 = %s; eigh %d, solves %d, %.1f s; start: %s%s" % (   # v36f dg
        T / rs, (T + a.tau) / rs, "converged" if conv else "NOT CONVERGED", len(hist), fmt(ent["energy_defect_final"], "%+.3e"), fmt(ent["energy_change_between_iterates_final"], "%.3e"),   # v36f dg
        fmt(dfields, "%.2e"), fmt(ent["energy_defect_final"], "%+.3e"), ent["eigendecompositions"], n_solves, ent["seconds"], start, ("; " + why) if why else ""), flush=True)   # v36f dg
    return ent   # v36f dg


def far_dust_v36f(a, chs, st, cache, base, R, rs, N_app, v_app, collars):   # v36f dg
    """version 36f: the dust beyond each collar at the record on the record's own held operator: its number (the columns' weight beyond the cut), its energy by the density route (N_app eps_dust + v_app J_dust summed beyond the cut) and its Killing split as spectral weights of the exterior block (the dust columns masked to the sites beyond the cut) on the eigenvectors of h_m at the accepted fields: below 0, in [0, m), in [m, b] and above b = sqrt(m^2 + 1/a^2) (the radial chain's band top; a = 1)."""   # v36f dg
    torch = BK["torch"]   # v36f dg
    Rf = np.asarray(R, float); b = math.sqrt(a.m * a.m + 1.0); out = {}   # v36f dg
    for Rc_rs in collars:   # v36f dg
        Rc = float(Rc_rs) * rs   # v36f dg
        if Rc >= Rf[-1]:   # v36f dg
            continue   # v36f dg
        i0 = int(np.searchsorted(Rf, Rc, side="right")); ext = Rf > Rc   # v36f dg
        num = 0.0; ener = 0.0; split = np.zeros(4)   # v36f dg
        for ch in chs:   # v36f dg
            s_ = st[ch.kappa]   # v36f dg
            if not s_["dust"].shape[1]:   # v36f dg
                continue   # v36f dg
            f = np.asarray(s_["f"], float); col = s_["dust"]   # v36f dg
            num += ch.w * float(np.dot(f, (col[2 * i0:, :].abs() ** 2).sum(dim=0).cpu().numpy()))   # v36f dg
            ener += ch.w * float(np.sum(N_app[ext] * base[ch.kappa][2][ext] + v_app[ext] * base[ch.kappa][3][ext]))   # v36f dg
            e_, U_ = cache[ch.kappa][1], cache[ch.kappa][2]   # v36f dg
            if e_ is None or U_ is None:   # v36f dg
                continue   # v36f dg
            ext_cols = torch.zeros_like(col); ext_cols[2 * i0:, :] = col[2 * i0:, :]   # v36f dg
            wi = ((U_.conj().T @ ext_cols).abs() ** 2 * T_(f)[None, :].real).sum(dim=1).cpu().numpy()   # v36f dg
            ev = e_.detach().cpu().numpy()   # v36f dg
            split += ch.w * np.array([wi[ev < 0.0].sum(), wi[(ev >= 0.0) & (ev < a.m)].sum(), wi[(ev >= a.m) & (ev <= b)].sum(), wi[ev > b].sum()])   # v36f dg
            del ext_cols   # v36f dg
        out["%g" % Rc_rs] = {"R_over_rs": float(Rc_rs), "R": Rc, "number": num, "energy_density_route": ener, "b": b,   # v36f dg
                            "killing_split": {"below_0": float(split[0]), "0_to_m": float(split[1]), "m_to_b": float(split[2]), "above_b": float(split[3])}}   # v36f dg
    return out   # v36f dg


def run(a):
    torch = BK["torch"]
    os.makedirs(a.out, exist_ok=True)'''),
("run label",
r'''    if v34_on(a):
        label += "; v34: " + v34_desc(a)
    rs = a.rs''',
r'''    if v34_on(a):
        label += "; v34: " + v34_desc(a)
    if DG["on"]:   # v36f dg
        label += "; v36f: " + dg_desc_v36f(a)   # v36f dg
    rs = a.rs'''),
("resume slice propagation",
r'''        ph = torch.exp(-1j * e.to(torch.complex128) * a.tau)
        prop = lambda X: U @ (ph[:, None] * (U.conj().T @ X))
        sea_n = prop(sea)
        dust_n = prop(dust) if dust.shape[1] else dust''',
r'''        ph = torch.exp(-1j * e.to(torch.complex128) * a.tau)
        prop = lambda X: U @ (ph[:, None] * (U.conj().T @ X))
        sea_n = prop(sea) if not DG["on"] else sea   # v36f dg: under --step dg the slice's columns stay at the slice's clock; the step follows the loop
        dust_n = (prop(dust) if dust.shape[1] else dust) if not DG["on"] else dust   # v36f dg'''),
("resume slice step",
r'''        del e, U
    sync_dev()
    fields = {k_: [] for k_ in ("T", "N", "v", "eps_tot", "J_tot", "eps_pairs", "J_pairs", "N_applied", "v_applied", "r_nodes", "N_nodes", "v_nodes")}''',
r'''        del e, U
    sync_dev()
    if DG["on"]:   # v36f dg: the step from the slice by the discrete-gradient rule, its start from the slice's node fields when stored
        DG["nodes"] = (np.asarray(Z["dg_nodes_N"], float), np.asarray(Z["dg_nodes_v"], float)) if ("dg_nodes_N" in Z.files and "dg_nodes_v" in Z.files) else None   # v36f dg
        dgr = dg_step_v36f(a, chs, st, R, base, (N_s.copy(), v_s.copy()), float(rec_s["E"]), Ts, rs)   # v36f dg
        DG["rec"] = dgr   # v36f dg
        if not dgr["converged"]:   # v36f dg
            raise SystemExit("version 36f dg: the discrete-gradient fixed point did not converge within %d iterations at the restart's step from T/rs %.4f (%s)" % (DG["cap"], Ts / rs, dgr.get("why")))   # v36f dg
    fields = {k_: [] for k_ in ("T", "N", "v", "eps_tot", "J_tot", "eps_pairs", "J_pairs", "N_applied", "v_applied", "r_nodes", "N_nodes", "v_nodes")}'''),
("hand init",
r'''    hand = {"T": [], "N2avg": [], "eps_sea": [], "eps_R2": [], "eps_dust": [], "S_r_ell": [], "S_t_ell": []}''',
r'''    hand = {"T": [], "N2avg": [], "eps_sea": [], "eps_R2": [], "eps_dust": [], "S_r_ell": [], "S_t_ell": []}
    if DG["on"]:   # v36f dg: the dust number density per site per record
        hand["n_dust"] = []   # v36f dg'''),
("resume hand",
r'''        for k_, v_ in pre34["hand"].items():
            hand[k_] = list(v_)
        for k_, v_ in pre34["spectra"].items():''',
r'''        for k_, v_ in pre34["hand"].items():
            hand[k_] = list(v_)
        if DG["on"]:   # v36f dg: the stored records without n_dust (a frozen run's) get NaN rows
            nd_ = list(hand.get("n_dust", []))   # v36f dg
            hand["n_dust"] = [np.full(a.nsites, np.nan)] * (len(hand["T"]) - len(nd_)) + nd_   # v36f dg
        for k_, v_ in pre34["spectra"].items():'''),
("end init",
r'''    s11_store = {}
    end33 = None
    end34 = None''',
r'''    s11_store = {}
    end33 = None
    end34 = None
    end36f = None   # v36f dg'''),
("per record far dust",
r'''        hand["S_r_ell"].append(Sr_l); hand["S_t_ell"].append(St_l)''',
r'''        hand["S_r_ell"].append(Sr_l); hand["S_t_ell"].append(St_l)
        far36f = None   # v36f dg
        if DG["on"]:   # v36f dg: the dust number density per site (the handoff's n_dust) and the far dust on the record's held operator
            nd_ = np.zeros(a.nsites)   # v36f dg
            for ch in chs:   # v36f dg
                if st[ch.kappa]["dust"].shape[1]:   # v36f dg
                    nd_ = nd_ + ch.w * ((st[ch.kappa]["dust"].abs() ** 2) * T_(st[ch.kappa]["f"])[None, :].real).sum(dim=1).reshape(-1, 2).sum(dim=1).cpu().numpy()   # v36f dg
            hand["n_dust"].append(np.asarray(nd_, float))   # v36f dg
            far36f = far_dust_v36f(a, chs, st, cache, base, R, rs, N_app, v_app, DG["collars"])   # v36f dg'''),
("slice arrays",
r'''            saved = os.path.join(a.slice_dir, "orbitals_%s_T%07.2f.npz" % (a.tag, T / rs))
            np.savez(saved, **arrs)''',
r'''            saved = os.path.join(a.slice_dir, "orbitals_%s_T%07.2f.npz" % (a.tag, T / rs))
            if DG["on"] and DG["nodes"] is not None:   # v36f dg: the node fields of the step that produced this record's state (the next step's start at a restart)
                arrs["dg_nodes_N"] = np.asarray(DG["nodes"][0], float); arrs["dg_nodes_v"] = np.asarray(DG["nodes"][1], float)   # v36f dg
            np.savez(saved, **arrs)'''),
("record entry",
r'''        rec["engine"] = "engine_collapse_v25"
        if solve32 is not None:
            rec["solve_v32"] = solve32''',
r'''        rec["engine"] = "engine_collapse_v25"
        if DG["on"]:   # v36f dg
            rec["step"] = "dg"; rec["engine_v36f"] = dg_desc_v36f(a); rec["far_dust_v36f"] = far36f   # v36f dg
            d_ = DG["rec"]; DG["rec"] = None   # v36f dg
            if d_ is not None:   # v36f dg: the step that produced this record's state, with this record's charge against E_0, the previous record and the step's prediction
                d_ = dict(d_); d_["E_record"] = E; d_["E_over_E0"] = E / E0; d_["E_minus_E0_over_E0"] = E / E0 - 1.0   # v36f dg
                d_["E_minus_E_pred_over_E0"] = ((E - d_["E_pred"]) / E0) if d_.get("E_pred") is not None else None   # v36f dg
                d_["E_minus_E_prev_over_E0"] = ((E - recs[-1]["E"]) / E0) if (recs and recs[-1].get("E") is not None) else None   # v36f dg
            rec["dg_v36f"] = d_   # v36f dg
        if solve32 is not None:
            rec["solve_v32"] = solve32'''),
("record line",
r'''        print(_line + (v32_record_segment(solve32) if solve32 is not None else ""), flush=True)''',
r'''        print(_line + (v32_record_segment(solve32) if solve32 is not None else ""), flush=True)
        if DG["on"]:   # v36f dg
            d_ = rec.get("dg_v36f"); f_ = rec.get("far_dust_v36f") or {}   # v36f dg
            print("T/rs %.4f v36f dg: E/E0 - 1 = %+.3e%s; far dust: %s" % (T / rs, E / E0 - 1.0, ("; against the previous record %s, against the step's prediction %s (the step from T/rs %.4f: %d iterations, energy defect %s, node fields %s, %.0f s)" % (   # v36f dg
                fmt(d_.get("E_minus_E_prev_over_E0"), "%+.3e"), fmt(d_.get("E_minus_E_pred_over_E0"), "%+.3e"), d_["from_T_over_rs"], d_["iterations"], fmt(d_.get("energy_defect_final"), "%+.3e"), fmt(d_.get("fields_change_final"), "%.1e"), d_.get("seconds", 0.0))) if d_ is not None else " (the initial state)",   # v36f dg
                "; ".join("beyond %s r_s: number %.5f, energy %.5f, Killing split below 0 %.5f, 0 to m %.5f, m to b %.5f, above b %.5f" % (k_, q["number"], q["energy_density_route"], q["killing_split"]["below_0"], q["killing_split"]["0_to_m"], q["killing_split"]["m_to_b"], q["killing_split"]["above_b"]) for k_, q in f_.items()) or "-"), flush=True)   # v36f dg'''),
("propagation",
r'''        t_p = time.time()
        for ch in chs:
            _hm, e, U, ref, _ci = cache[ch.kappa]''',
r'''        t_p = time.time()
        if DG["on"]:   # v36f dg: the discrete-gradient step in place of the frozen one
            for ch in chs:   # v36f dg
                cache[ch.kappa] = (None, None, None, cache[ch.kappa][3], cache[ch.kappa][4])   # v36f dg: the record's eigendecompositions freed before the step
            dgr = dg_step_v36f(a, chs, st, R, base, f_ref, E, T, rs)   # v36f dg
            DG["rec"] = dgr   # v36f dg
            if not dgr["converged"]:   # v36f dg
                end36f = {"T": T, "T_over_rs": T / rs, "dg": {k_: v_ for k_, v_ in dgr.items() if k_ != "history"}, "rule": "version 36f dg: the discrete-gradient fixed point did not converge within the cap at this record; the record is written and nothing is propagated from it"}   # v36f dg
                print("T/rs %.4f: version 36f dg: STOP: the discrete-gradient fixed point did not converge within %d iterations (%s); the record is written and nothing is propagated from it" % (T / rs, DG["cap"], dgr.get("why") or "the cap"), flush=True)   # v36f dg
                break   # v36f dg
        for ch in (() if DG["on"] else chs):   # v36f dg: the frozen step unless --step dg
            _hm, e, U, ref, _ci = cache[ch.kappa]'''),
("final dump",
r'''        json.dump(dict({"script": "engine_collapse_v34.py", "args": vars(a), "head": head, "records": recs}, **({"end_v32": end32} if end32 is not None else {}),
                       **({"end_s11": end33} if end33 is not None else {}), **({"end_v34": end34} if end34 is not None else {})), fh, default=float)''',
r'''        json.dump(dict({"script": "engine_collapse_v34.py", "args": vars(a), "head": head, "records": recs}, **({"end_v32": end32} if end32 is not None else {}),
                       **({"end_s11": end33} if end33 is not None else {}), **({"end_v34": end34} if end34 is not None else {}),
                       **({"end_v36f": end36f} if end36f is not None else {})), fh, default=float)   # v36f dg'''),
("arguments",
r'''    ap.add_argument("--nk-set", dest="nk_set", default=None, choices=["resolve"], help="(v36c) the set-resolving Newton solve")
    a = ap.parse_args()''',
r'''    ap.add_argument("--nk-set", dest="nk_set", default=None, choices=["resolve"], help="(v36c) the set-resolving Newton solve")
    ap.add_argument("--step", dest="step", default="frozen", choices=["frozen", "dg"], help="(v36f) frozen: the record's fields held over the step (version 36d bitwise); dg: the discrete-gradient exponential step")   # v36f dg
    ap.add_argument("--dg-tol", dest="dg_tol", type=float, default=1e-12, help="(v36f dg) the fixed point's energy tolerance, relative to the record's charge")   # v36f dg
    ap.add_argument("--dg-cap", dest="dg_cap", type=int, default=30, help="(v36f dg) at most this many iterations of the fixed point; the run ends at the record otherwise")   # v36f dg
    a = ap.parse_args()'''),
("validation",
r'''    if a.nk_set == "resolve":
        if a.nk_jvp != "exact" or a.nk_line_search != "halving":
            ap.error("version 36c: --nk-set resolve needs --nk-jvp exact and --nk-line-search halving")
        NSR["on"] = True
    C5.backend(a.gpu)
    run(a)''',
r'''    if a.nk_set == "resolve":
        if a.nk_jvp != "exact" or a.nk_line_search != "halving":
            ap.error("version 36c: --nk-set resolve needs --nk-jvp exact and --nk-line-search halving")
        NSR["on"] = True
    DG["on"] = (a.step == "dg"); DG["tol"] = float(a.dg_tol); DG["cap"] = int(a.dg_cap)   # v36f dg
    if DG["on"] and a.mix not in ("plain", "anderson"):   # v36f dg
        ap.error("version 36f: --step dg iterates its fixed point in the node fields with --mix plain or anderson")   # v36f dg
    if DG["on"] and a.resume_from:   # v36f dg
        ap.error("version 36f: --step dg restarts from a stored complex128 slice (--resume-slice); the --resume-from replay propagates by the frozen step")   # v36f dg
    if DG["on"] and (a.dg_tol <= 0.0 or a.dg_cap < 1):   # v36f dg
        ap.error("version 36f: --dg-tol must be positive and --dg-cap at least 1")   # v36f dg
    C5.backend(a.gpu)
    run(a)'''),
]


def build(text):
    """the forward replacements on the source text; every OLD must occur exactly once. Returns the built text."""
    for name, old, new in REPLACEMENTS:
        n = text.count(old)
        if n != 1:
            raise SystemExit("build_engine_collapse_v36f: the OLD string of '%s' occurs %d times in the source (1 required): the source is not input 05's engine_collapse_v36d.py" % (name, n))
        text = text.replace(old, new)
    return text


def reverse(text):
    """the reverse replacements (NEW -> OLD, in reverse order); every NEW must occur exactly once."""
    for name, old, new in reversed(REPLACEMENTS):
        n = text.count(new)
        if n != 1:
            raise SystemExit("build_engine_collapse_v36f: the NEW text of '%s' occurs %d times in the built file (1 required)" % (name, n))
        text = text.replace(new, old)
    return text


def md5(b):
    return hashlib.md5(b).hexdigest()


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "engine_collapse_v36d.py")
    dst = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(os.path.abspath(src)), "engine_collapse_v36f.py")
    raw = open(src, "rb").read()
    text = raw.decode("utf-8")
    out = build(text)
    back = reverse(out)
    if back != text:
        raise SystemExit("build_engine_collapse_v36f: the reverse replacements do not restore the source (a NEW text overlaps another)")
    marked = sum(1 for l_ in out.splitlines() if ("# v36f dg" in l_ or "[v36f dg]" in l_))
    with open(dst, "wb") as fh:
        fh.write(out.encode("utf-8"))
    py_compile.compile(dst, doraise=True)
    print("built %s from %s: %d replacements, %d marked lines; source md5 %s; built md5 %s; the reverse replacements restore the source byte for byte; compiled" % (
        dst, src, len(REPLACEMENTS), marked, md5(raw), md5(out.encode("utf-8"))))


if __name__ == "__main__":
    main()
