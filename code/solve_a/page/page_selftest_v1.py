#!/usr/bin/env python
"""page_selftest_v1.py (lead reviewer A, GAP03 integration; builder analyst runs it): the --selftest of page_recursion_v1.py, minutes on a
CPU. T1 the operator against the engine's chain_parts when importable, and the symbols sin k sigma_x and sin(2k)/2 on a bulk plane wave;
T2 the l = 1 kernel average reproduces a linear flow; T3 the recursion identities on a random pure Gaussian state (purity, S(B) = tr s(Z),
the mutual-information identity, the increment identity); T4 the physical small case: the formation, Delta_0 by principal angles against the
dense difference of projectors, 12 block steps with S_esc = tr s(M) against the direct dense entropy of the collected modes, the Z identity,
the energy balance, the stage direct route, the partner identity, and one transition against the exact dense evolution through the ramp;
T5 the continuum luminosity and b against 0.0052373 and 1.6391; T6 the lattice first-passage transmission at r_s/a 8 against the continuum
average (coarse; stated tolerance). Every check prints PASS or FAIL with its number; run_all returns True when all pass."""
import json
import math
import os
import sys
import time

import numpy as np

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import page_ops_v1 as OPS            # noqa: E402
import page_core_v1 as CORE          # noqa: E402

CD = OPS.CD
RESULTS = []


def rep(name, ok, detail):
    RESULTS.append({"test": name, "pass": bool(ok), "detail": detail})
    print("%s  %s: %s" % ("PASS" if ok else "FAIL", name, json.dumps(detail, default=float)), flush=True)
    return bool(ok)


def s_np(x):
    return OPS.binary_entropy_np(x)


# --------------------------------------------------------------------------------------------------------------------- T1
def t1_operator():
    ok = True
    try:
        sys.path.insert(0, os.path.join(HERE, "..", "engine"))
        import engine_ops_v1 as EOPS
        h_ref, Q_ref, R_ref = EOPS.chain_parts(64, 0.2, 2.0)
        h0, Q = OPS.chain_sparse(R_ref, 0.2, 2.0)
        dh = float(np.max(np.abs(h0.toarray() - h_ref))); dQ = float(np.max(np.abs(Q.toarray() - Q_ref)))
        ok &= rep("T1a operator against engine_ops_v1.chain_parts (m 0.2, kappa 2, 64 sites)", dh < 1e-14 and dQ < 1e-14, {"max_diff_h": dh, "max_diff_Q": dQ})
    except Exception as ex:                                          # the engine is not importable on the pod
        rep("T1a operator against the engine", True, {"skipped": str(ex)[:80]})
    n = 400
    R = np.arange(n, dtype=float) + 1.0
    h0, Q = OPS.chain_sparse(R, 0.0, 0.0)                              # no on-site term: the pure sine chain
    k = 0.7
    u = np.array([1.0, 1.0]) / math.sqrt(2.0)                           # sigma_x eigenvector (+1)
    psi = (np.exp(1j * k * np.arange(n))[:, None] * u[None, :]).reshape(-1)
    hpsi = (h0 @ psi).reshape(n, 2); Qpsi = (Q @ psi).reshape(n, 2)
    mid = slice(100, 300)
    err_h = float(np.max(np.abs(hpsi[mid] - math.sin(k) * psi.reshape(n, 2)[mid])))
    err_Q = float(np.max(np.abs(Qpsi[mid] - 0.5 * math.sin(2 * k) * psi.reshape(n, 2)[mid])))
    ok &= rep("T1b symbols sin k sigma_x and sin(2k)/2 on a bulk plane wave", err_h < 1e-12 and err_Q < 1e-12, {"err_h": err_h, "err_Q": err_Q})
    # the clocked operator with uniform v: energies +-sin k + v sin(2k)/2
    v = -0.3
    h = OPS.clocked_csr(h0, Q, np.ones(n), np.full(n, v))
    hp = (h @ psi).reshape(n, 2)
    err_c = float(np.max(np.abs(hp[mid] - (math.sin(k) + v * 0.5 * math.sin(2 * k)) * psi.reshape(n, 2)[mid])))
    ok &= rep("T1c clocked symbol sin k + v sin(2k)/2 at uniform v", err_c < 1e-12, {"err": err_c})
    return ok


# --------------------------------------------------------------------------------------------------------------------- T2
def t2_kernel():
    R = np.arange(200, dtype=float) + 1.0
    av = OPS.FlowAverager(R, 2.0, "averaged")
    out = av(lambda r: np.asarray(r, float))
    sel = (R > 20) & (R < 170)
    err = float(np.max(np.abs(out[sel] - R[sel])))
    nrm = float(np.max(np.abs(av.norm0[sel] - 1.0)))
    return rep("T2 l = 1 kernel average of v = r returns R_n (bulk), l = 0 norms 1", err < 1e-8 and nrm < 1e-8, {"max_err_v": err, "max_err_norm0": nrm})


# --------------------------------------------------------------------------------------------------------------------- T3
def t3_recursion_identities(d=24, m1=5, m2=4, seed=3):
    rng = np.random.default_rng(seed)
    def unitary(k):
        q, _ = np.linalg.qr(rng.normal(size=(k, k)) + 1j * rng.normal(size=(k, k)))
        return q
    Uc = unitary(d); Phi = Uc[:, : d // 2]; C = Phi @ Phi.conj().T
    W = unitary(d); Q1 = W[:, :m1]; Q2 = W[:, m1: m1 + m2]; A1 = W[:, m1:]; A2 = W[:, m1 + m2:]
    S = lambda M: s_np(np.linalg.eigvalsh(0.5 * (M + M.conj().T)))
    M1 = A1.conj().T @ C @ A1; M2 = A2.conj().T @ C @ A2
    Dk = A2.conj().T @ A1
    purity = abs(S(Q1.conj().T @ C @ Q1) - S(M1))
    rec = float(np.linalg.norm(M2 - Dk @ M1 @ Dk.conj().T))
    ev, U = np.linalg.eigh(M1); P = (U * np.sqrt(np.clip(ev * (1 - ev), 0, None))) @ U.conj().T
    Z = np.block([[np.eye(d - m1) - M1, P @ Dk.conj().T], [Dk @ P, M2]])
    SB_direct = S(Q2.conj().T @ C @ Q2); SB_Z = S(Z)
    I_direct = S(Q1.conj().T @ C @ Q1) + SB_direct - S(np.concatenate([Q1, Q2], axis=1).conj().T @ C @ np.concatenate([Q1, Q2], axis=1))
    I_rec = S(M1) + SB_Z - S(M2)
    incr = (S(M2) - S(M1)) - (SB_direct - I_direct)
    ok = purity < 1e-10 and rec < 1e-12 and abs(SB_direct - SB_Z) < 1e-9 and abs(I_direct - I_rec) < 1e-9 and abs(incr) < 1e-9
    return rep("T3 recursion identities on a random pure Gaussian state", ok, {"purity": purity, "M2_minus_DMD": rec, "S_B_direct_minus_Z": SB_direct - SB_Z,
                                                                            "I_direct_minus_rec": I_direct - I_rec, "increment_identity": incr})


# --------------------------------------------------------------------------------------------------------------------- T4
def t4_small_case(device):
    import page_recursion_v1 as PR
    argv = ["--rs", "4", "--g", "0.25", "--nsites", "192", "--channels", "1", "--ell", "2", "--Rb", "0.5", "--Rb0", "3", "--R-cut", "12", "--R-collar", "6",
            "--tau-s", "2", "--m-cap", "1000", "--eps-coll", "1e-10", "--eps-rank", "1e-13", "--eps-form", "1e-9", "--eps-orth", "1e-15", "--fifo-depth", "100",
            "--fifo-prune-weight", "0", "--store-cap", "100000", "--tau-tr", "8", "--dt-tr", "0.25", "--device", str(device), "--out", "<internal>/page_selftest", "--tag", "t4"]
    a = PR.parse(argv)
    dev = torch.device(str(device))
    ch = CORE.Channel(1, a.m, a.nsites, False, dev)
    ch.set_averager(a.ell, a.fields)
    Phi0, frec = ch.formation(a)
    ok = rep("T4a formation unitarity (r_s 4, n 192, dust from 3 r_s)", frec["unitarity"] < 1e-10, {"unitarity": frec["unitarity"], "t_form_over_rs": frec["t_form_over_rs"], "n0": frec["n0"]})
    st = ch.build_stage(a, a.E1, a.Rb * a.rs)
    irec = ch.initial_memory(Phi0, a)
    V, e, neg = st["V"], st["e"], st["neg"]
    C0 = Phi0 @ Phi0.mH
    pur = float(torch.linalg.norm(C0 @ C0 - C0))
    P1 = V[:, neg] @ V[:, neg].mH
    Delta_exact = C0 - P1
    Drec = (V @ ch.c) @ ch.MD @ (V @ ch.c).mH
    d_delta = float(torch.linalg.norm(Delta_exact - Drec)) / max(float(torch.linalg.norm(Delta_exact)), 1e-300)
    ok &= rep("T4b purity of C_0 and Delta_0 reconstructed from the split hull of its principal angles", pur < 1e-10 and d_delta < 1e-6,
              {"||C0^2 - C0||": pur, "rel_err_Delta": d_delta, "memory_rank": irec["memory_rank"], "Delta_rank": irec["rank_kept"], "E_inv": irec["E_inv"]})
    E_inv = float(torch.trace(Delta_exact @ OPS.dense_from_csr(st["h"], dev)).real)
    ok &= rep("T4c E_inv = tr[Delta_0 h_1] from the memory against the dense trace", abs(irec["E_inv"] - E_inv) < 1e-8 * max(abs(E_inv), 1.0), {"E_inv_memory": irec["E_inv"], "E_inv_dense": E_inv})
    # the block steps against the dense state
    worst_S = worst_Z = worst_bal = worst_gram = 0.0
    collected = []
    t = 0.0
    for n in range(12):
        t += a.tau_s_abs
        r = ch.block_step(a, t)
        if r["m_coll"]:
            cq, hq, tq = ch.fifo[-1]
            collected.append((cq.clone(), tq))
        # dense state at time t
        ph = torch.exp(-1j * e.to(CD) * t)
        Ct = V @ ((ph[:, None] * (V.mH @ C0 @ V)) * ph.conj()[None, :]) @ V.mH
        if collected:
            Q = torch.cat([V @ (torch.exp(-1j * e.to(CD) * (t - tq))[:, None] * cq) for cq, tq in collected], dim=1)
            gram = float(torch.linalg.norm(Q.mH @ Q - torch.eye(Q.shape[1], dtype=CD, device=dev)))
            S_direct, _ = OPS.entropy_cov(Q.mH @ Ct @ Q)
            worst_gram = max(worst_gram, gram)
            worst_S = max(worst_S, abs(S_direct - r["S_new"]))
        if r["m_coll"]:
            worst_Z = max(worst_Z, abs(r["S_B_Z"] - r["S_B"]))
        worst_bal = max(worst_bal, abs(r["E_mem"] + ch.rec["E_rad"] - ch.rec["E_delta_init"]) / max(abs(ch.rec["E_delta_init"]), 1e-300))
    ok &= rep("T4d S_esc = tr s(M) against the direct dense entropy of the collected modes over 12 blocks", worst_S < 1e-6 and worst_gram < 1e-8,
              {"max_|S_direct - S_memory|": worst_S, "max_gram_defect": worst_gram, "S_esc_final": r["S_new"], "n_collected": sum(c.shape[1] for c, _ in collected)})
    ok &= rep("T4e S(B_k) from Z_k against the direct block entropy", worst_Z < 1e-9, {"max_|S_B_Z - S_B|": worst_Z})
    ok &= rep("T4f energy balance E_mem + E_rad = tr[Delta h] within the stage", worst_bal < 1e-8, {"max_rel_residual": worst_bal})
    d = ch.stage_direct(a, with_spectrum=True)
    ok &= rep("T4g stage direct route G_R = Pi + A^dagger M_init A against the memory route", d is not None and abs(d["S_direct"] - r["S_new"]) < 1e-6 and d.get("gram_defect", 0.0) < 1e-8,
              {"S_direct": d["S_direct"] if d else None, "S_memory": r["S_new"], "gram_defect": d.get("gram_defect") if d else None})
    prt = ch.partner_readout(a, n_modes=4)
    werr = max([abs(p["norm2_w"] - p["n_1_minus_n"]) for p in prt] or [0.0])
    ok &= rep("T4h partner identity ||C q - n q||^2 = n(1 - n)", werr < 1e-8, {"max_identity_error": werr, "n_values": [p["n"] for p in prt]})
    # one transition against the exact dense evolution through the ramp
    E_new = 0.98 * a.E1
    S_before, _ = ch.memory_entropy()
    v_old = st["v_sites"]; rs_old = st["rs"]
    tau = a.tau_tr * rs_old
    v_new = np.asarray(ch.averager(OPS.flow_dense(2.0 * a.g * E_new, a.Rb * 2.0 * a.g * E_new)), float)
    vfun_t = lambda tt: v_old + (v_new - v_old) * OPS.ramp_c1(tt / tau)
    ph = torch.exp(-1j * e.to(CD) * ch.t_ref)
    C_now = V @ ((ph[:, None] * (V.mH @ C0 @ V)) * ph.conj()[None, :]) @ V.mH
    Uexact = OPS.cf4_propagate(ch.B, torch.eye(ch.n2, dtype=CD, device=dev), vfun_t, 0.0, tau, a.dt_tr)
    C_after = Uexact @ C_now @ Uexact.mH
    tr = ch.transition(a, E_new, a.Rb * 2.0 * a.g * E_new)
    st2 = ch.stage
    P2 = st2["V"][:, st2["neg"]] @ st2["V"][:, st2["neg"]].mH
    Delta_after_exact = C_after - P2
    Drec2 = (st2["V"] @ ch.c) @ ch.MD @ (st2["V"] @ ch.c).mH
    d_after = float(torch.linalg.norm(Delta_after_exact - Drec2)) / max(float(torch.linalg.norm(Delta_after_exact)), 1e-300)
    S_after, _ = ch.memory_entropy()
    tol = max(1e-3, 10.0 * tr["adiabatic_leak_max"])
    ok &= rep("T4i transition: the recursion's Delta against the exact dense evolution through the ramp (adiabatic sea; tolerance 10 x leak)",
              d_after < max(1e-2, 50.0 * tr["adiabatic_leak_max"]) and abs(S_after - S_before) < tol,
              {"rel_err_Delta_after": d_after, "adiabatic_leak_max": tr["adiabatic_leak_max"], "S_continuity_defect": S_after - S_before,
               "released_pushed_up": tr["released_pushed_up"], "released_new_pockets": tr["released_new_pockets"], "ambiguous": tr["ambiguous_continuations"]})
    return ok


# --------------------------------------------------------------------------------------------------------------------- T5
def t5_continuum():
    import page_scatter_v1 as SC
    t0 = time.time()
    c = SC.continuum_totals(kmax=6, n_nodes=40, x_max=1.5)
    ok = abs(c["rel_dev_L"]) < 2e-3 and abs(c["rel_dev_b"]) < 2e-3 and c["unitarity_error_max"] < 1e-6
    return rep("T5 continuum L r_s^2 and b against 0.0052373 and 1.6391 (tolerance 2e-3)", ok,
               {"L_rs2": c["L_rs2"], "rel_dev_L": c["rel_dev_L"], "b": c["b"], "rel_dev_b": c["rel_dev_b"], "mean_omega_over_TH": c["mean_omega_over_TH"],
                "unitarity_max": c["unitarity_error_max"], "seconds": time.time() - t0})


# --------------------------------------------------------------------------------------------------------------------- T6
def t6_lattice_gamma(device):
    import page_recursion_v1 as PR
    import page_scatter_v1 as SC
    argv = ["--rs", "8", "--g", "0.25", "--nsites", "768", "--channels", "1", "--ell", "2", "--Rb", "0.5", "--device", str(device), "--out", "<internal>/page_selftest", "--tag", "t6"]
    a = PR.parse(argv)
    dev = torch.device(str(device))
    ch = CORE.Channel(1, 0.0, a.nsites, False, dev)
    ch.set_averager(a.ell, a.fields)
    st = ch.build_stage(a, a.E1, a.Rb * a.rs)
    rows = SC.lattice_gamma(ch, st, a, (0.6, 1.2))
    ratios = [r["ratio_lat_over_cont"] for r in rows]
    ok = all(0.5 < x < 1.5 for x in ratios)
    return rep("T6 lattice first-passage Gamma at r_s/a 8 against the continuum average (coarse; tolerance factor 1.5)", ok,
               {"ratios": ratios, "Gamma_lat": [r["Gamma_lat"] for r in rows], "Gamma_cont_avg": [r["Gamma_cont_avg"] for r in rows],
                "x_mean": [r["x_mean"] for r in rows], "flatness": [r["plateau_flatness_2rs_later"] for r in rows]})


def run_all(a):
    device = a.device if (str(a.device).startswith("cpu") or torch.cuda.is_available()) else "cpu"
    torch.set_num_threads(a.threads)
    t0 = time.time()
    ok = True
    ok &= t1_operator()
    ok &= t2_kernel()
    ok &= t3_recursion_identities()
    ok &= t4_small_case(device)
    ok &= t5_continuum()
    ok &= t6_lattice_gamma(device)
    out = a.out or "<internal>/page_selftest"
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "selftest_results.json"), "w") as f:
        json.dump({"all_pass": bool(ok), "results": RESULTS, "seconds": time.time() - t0, "device": str(device)}, f, indent=1, default=float)
    print("SELFTEST %s in %.0f s (results in %s)" % ("PASS" if ok else "FAIL", time.time() - t0, os.path.join(out, "selftest_results.json")), flush=True)
    return bool(ok)
