#!/usr/bin/env python
"""page_d13_readout_selftest_v1.py: manufactured checks of page_d13_readout_v1 with no data; chains of at most 512 sites on the CPU and on
cuda when present. Checks: (1) the exterior Gram route equals the n x n route; (2) S_X equals the complement's on a pure Gaussian state;
(3) the vacuum's excess is 0 to 1e-10 (the vacuum in another orthonormal basis of its span); (4) E_inv = Tr[hC] - Tr min(h, 0) is 0 on the
vacuum and positive off it, Tr min(h, 0) from the banded eigenvalues equals the dense sum; (5) the ledger identity telescopes on a
manufactured unitary history (per-record flows and jumps at two cuts) and the vacuum carries no current; (6) the back-propagation returns
the probe at the reference clock when the history is static, and on a time-dependent history the back-propagated current equals the direct
one with L_old + L_sea = L; (7) the current's rank is at most 16 and the probes reassemble J_x; (8) the pair route of the vacuum equals the
dense route; (9) the shells' excess against P_- is 0 on the vacuum; (10) the source modes have the stated inside weight and unit
occupation; (11) the Killing density sums to E_inv; (12) the light travel time matches the closed form on the vacuum exterior; (13) a
whole ChannelReadout.checkpoint runs on the manufactured chain. Prints "SELFTEST d13 v1 PASSED" and exits 0, else names each failure."""
import math
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import page_d13_readout_v1 as D13   # noqa: E402

OPS = D13.OPS
CD, RD = D13.CD, D13.RD
FAILS = []


def check(name, ok, detail=""):
    print("  %-72s %s %s" % (name, "ok" if ok else "FAIL", detail), flush=True)
    if not ok:
        FAILS.append("%s %s" % (name, detail))


def manufactured(n, kappa, seed, device):
    """a massless chain of n sites with smooth subcritical fields; returns (R, h_csr, N, v, ops)."""
    rng = np.random.default_rng(seed)
    R = np.arange(n, dtype=float) + 1.0
    N = 1.0 + 0.08 * np.cos(2.0 * math.pi * R / n) + 0.02 * rng.standard_normal()
    v = -0.35 * np.sin(math.pi * R / n) ** 2 - 0.05
    h0, Q = OPS.chain_sparse(R, 0.0, float(kappa))
    return R, OPS.clocked_csr(h0, Q, N, v), N, v, (h0, Q)


def random_isometry(n2, n0, device, seed=1):
    g = torch.Generator(device="cpu").manual_seed(seed)
    A = torch.randn(n2, n0, generator=g, dtype=RD) + 1j * torch.randn(n2, n0, generator=g, dtype=RD)
    Qm, _ = torch.linalg.qr(A.to(CD))
    return Qm.to(device)


def run_device(device):
    print("device %s" % device, flush=True)
    n = 192; kappa = 1
    R, h, N, v, (h0, Q) = manufactured(n, kappa, 7, device)
    R2 = torch.as_tensor(np.repeat(R, 2), dtype=RD, device=device)
    n2 = 2 * n
    Bh = OPS.Banded(h, device)
    # the dense vacuum
    vac = D13.killing_vacuum(h, device, "dense")
    Vneg = vac["Vneg"]; n0 = vac["n_neg"]
    # (1) the Gram route against the n x n route
    Phi = random_isometry(n2, n0, device, 3)
    cut = 96.0
    mask = R2 > cut
    evG = torch.linalg.eigvalsh(Phi[mask].mH @ Phi[mask])
    CX = Phi[mask] @ Phi[mask].mH
    evB = torch.linalg.eigvalsh(0.5 * (CX + CX.mH))
    S_gram = D13.entropy_of_eigs(evG); S_block = D13.entropy_of_eigs(evB)
    check("(1) exterior Gram route equals the n x n route", abs(S_gram - S_block) <= 1e-10, "|dS| %.2e" % abs(S_gram - S_block))
    # (2) purity
    cuts = [("48", 48.0), ("96", 96.0), ("144", 144.0)]
    ent = D13.exterior_entropies(Phi, R2, cuts, None)
    gap = max(abs(d["purity_gap"]) for d in ent.values())
    check("(2) S_X equals the complement's on a pure Gaussian state", gap <= 1e-9, "max |S_X - S_Xc| %.2e" % gap)
    # (3) the vacuum's excess
    U0 = random_isometry(n0, n0, device, 5)
    Phi_v = Vneg @ U0
    ent_v = D13.exterior_entropies(Phi_v, R2, cuts, Vneg)
    exc = max(abs(d["excess"]) for d in ent_v.values())
    check("(3) the vacuum's excess is 0 to 1e-10 (the vacuum in another basis)", exc <= 1e-10, "max |excess| %.2e; S_X %s" % (exc, ["%.4f" % d["S_X"] for d in ent_v.values()]))
    # (4) E_inv
    E_ref_b, info = D13.tr_min_h0(h)
    E_ref_d = float(np.sum(vac["e"][vac["e"] < 0]))
    check("(4a) Tr min(h, 0) from the banded eigenvalues equals the dense sum", abs(E_ref_b - E_ref_d) <= 1e-10, "%.2e (n_neg %d, half-bandwidth %d, sum e %.1e)" % (abs(E_ref_b - E_ref_d), info["n_neg"], info["half_bandwidth"], info["sum_e"]))
    E_inv_vac = D13.tr_blocked(Bh, Phi_v) - E_ref_b
    check("(4b) E_inv is 0 on the vacuum", abs(E_inv_vac) <= 1e-10, "%.2e" % abs(E_inv_vac))
    e_all, V_all = OPS.eigh_csr(h, device)
    pos_idx = int((e_all > 0).nonzero().ravel()[0]); neg_last = int((e_all < 0).nonzero().ravel()[-1])
    Phi_x = Vneg.clone(); Phi_x[:, -1] = V_all[:, pos_idx]
    E_inv_x = D13.tr_blocked(Bh, Phi_x) - E_ref_b
    expect = float(e_all[pos_idx] - e_all[neg_last])
    check("(4c) E_inv is positive off the vacuum and equals the swapped level difference", E_inv_x > 0 and abs(E_inv_x - expect) <= 1e-10, "E_inv %.6e expected %.6e" % (E_inv_x, expect))
    del V_all
    # (5) the ledger telescopes on a manufactured unitary history
    tau = 1.0; nrec = 6
    recs = []
    for k in range(nrec + 1):
        s = 0.6 * k / nrec
        Nk = N * (1.0 + 0.05 * math.sin(2.0 * s)); vk = v * (1.0 + 0.2 * s)
        recs.append(OPS.clocked_csr(h0, Q, Nk, vk))
    cuts_l = [("64", 64.0), ("128", 128.0)]
    chis = {lab: (R2 > x).to(CD)[:, None] for lab, x in cuts_l}
    X = Phi.clone()
    B0 = OPS.Banded(recs[0], device)
    E_start = {lab: D13.tr_blocked(B0, X, chis[lab]) for lab in chis}
    flows = {lab: 0.0 for lab in chis}; jumps = {lab: 0.0 for lab in chis}
    for k in range(nrec):
        Bk = OPS.Banded(recs[k], device)
        e0 = {lab: D13.tr_blocked(Bk, X, chis[lab]) for lab in chis}
        X, _, _ = D13.propagate(Bk, X, tau, 1e-14, 64)
        for lab in chis:
            flows[lab] += D13.tr_blocked(Bk, X, chis[lab]) - e0[lab]
        Bd = OPS.Banded((recs[k + 1] - recs[k]).tocsr(), device)
        for lab in chis:
            jumps[lab] += D13.tr_blocked(Bd, X, chis[lab])
    Bend = OPS.Banded(recs[nrec], device)
    worst = 0.0
    for lab in chis:
        E_end = D13.tr_blocked(Bend, X, chis[lab])
        worst = max(worst, abs(E_end - E_start[lab] - flows[lab] - jumps[lab]))
    check("(5a) the ledger identity telescopes (flows + jumps = the inventory's change) on a unitary history", worst <= 1e-9, "max residual %.2e" % worst)
    Jb = OPS.Banded(OPS.current_out_csr(h, R, 96.0), device)
    Lvac = D13.tr_blocked(Jb, Vneg)
    check("(5b) the vacuum carries no current through the cut", abs(Lvac) <= 1e-10, "%.2e" % abs(Lvac))
    unit = float(torch.linalg.norm(X.mH @ X - torch.eye(n0, dtype=CD, device=device)))
    check("(5c) the propagation keeps the columns orthonormal", unit <= 1e-9, "%.2e" % unit)
    # (6) back-propagation
    W, j, jinfo = D13.probe_vectors(h, R, 96.0, device, 16)
    Wf = W.clone()
    m = 5
    for _ in range(m):
        Wf, _, _ = D13.propagate(Bh, Wf, tau, 1e-14, 64)
    for _ in range(m):
        Wf, _, _ = D13.propagate(Bh, Wf, -tau, 1e-14, 64)
    err = float(torch.linalg.norm(Wf - W))
    check("(6a) the back-propagation returns the probe at the reference clock on a static history", err <= 1e-10, "%.2e" % err)
    # the time-dependent history: C(t_ref) = Phi_x (off the vacuum) at record 0, propagated to the end; probes of J at the end's h
    Xt = Phi_x.clone()
    for k in range(nrec):
        Xt, _, _ = D13.propagate(OPS.Banded(recs[k], device), Xt, tau, 1e-14, 64)
    Wt, jt, _ = D13.probe_vectors(recs[nrec], R, 96.0, device, 16)
    L_direct = D13.tr_blocked(OPS.Banded(OPS.current_out_csr(recs[nrec], R, 96.0), device), Xt)
    for k in range(nrec - 1, -1, -1):
        Wt, _, _ = D13.propagate(OPS.Banded(recs[k], device), Wt, -tau, 1e-14, 64)
    vac0 = D13.killing_vacuum(recs[0], device, "dense")
    spl = D13.probe_split(Wt, jt, Phi_x, vac0["Vneg"])
    check("(6b) the back-propagated current equals the direct one on a time-dependent history", abs(spl["L_total_backpropagated"] - L_direct) <= 1e-9, "|dL| %.2e (L %.4e)" % (abs(spl["L_total_backpropagated"] - L_direct), L_direct))
    check("(6c) L_old + L_sea = L", abs(spl["L_old"] + spl["L_sea"] - spl["L_total_backpropagated"]) <= 1e-12, "%.2e" % abs(spl["L_old"] + spl["L_sea"] - spl["L_total_backpropagated"]))
    check("(6d) |L_old| within the rank-16 bound", abs(spl["L_old"]) <= spl["bound_rank16_trace_norm"] + 1e-12, "|L_old| %.3e, bound %.3e" % (abs(spl["L_old"]), spl["bound_rank16_trace_norm"]))
    # (7) the rank
    Jcsr = OPS.current_out_csr(h, R, 96.0)
    Jd = torch.as_tensor(Jcsr.toarray(), dtype=CD, device=device)
    Jre = (W * j.to(CD)[None, :]) @ W.mH
    check("(7) rank at most 16 and the probes reassemble J_x", jinfo["rank"] <= 16 and float(torch.linalg.norm(Jre - Jd)) <= 1e-12, "rank %d, support %d modes, ||sum j w w^dag - J||_F %.1e" % (jinfo["rank"], jinfo["support_modes"], float(torch.linalg.norm(Jre - Jd))))
    # (8) the pair route
    vp = D13.killing_vacuum(h, device, "pair")
    Pd = Vneg @ Vneg.mH; Pp = vp["Vneg"] @ vp["Vneg"].mH
    dP = float(torch.linalg.norm(Pd - Pp))
    check("(8) the pair route of the vacuum equals the dense route", dP <= 1e-8 and vp["n_neg"] == n0 and abs(vp["E_ref"] - E_ref_d) <= 1e-9,
          "||P_pair - P_dense||_F %.2e, n_neg %d/%d, dE_ref %.1e, pairs %d, big clusters %d (tol %g)" % (dP, vp["n_neg"], n0, vp["E_ref"] - E_ref_d, vp.get("n_pairs", -1), vp.get("n_big_clusters", -1), vp.get("cluster_tol_used", 0)))
    # (9) the shells on the vacuum
    rs = 32.0; TH_ref = 1.0 / (4.0 * math.pi * rs)
    sh = D13.killing_shells(h, R, 64.0, device, TH_ref, 0.5, 40.0, 1.0, 25.0)
    cont = D13.shell_content(sh, Phi_v, Vneg)
    T = cont["totals"]
    worstP = max(abs(T[k_]) for k_ in T if k_.endswith("_P") and T[k_] is not None)
    check("(9) the shells' excess against P_- is 0 on the vacuum (the block-ground reference's edge mixing printed)", worstP <= 1e-10,
          "max |excess_P| %.1e; block-ground hard out %.2e soft out %.2e above-band N %.2e (edge mixing, not a criterion)" % (worstP, T["hard_E_out"], T["soft_E_out"], T["above_N_total"]))
    D13.free_shells(sh)
    # (10) the source modes
    sm = D13.source_modes(Phi_x, R2, 40.0, h, device, Vneg, 0.9, 64)
    if sm["S"].shape[1]:
        ok = bool(np.all(np.abs(sm["w_in"] - sm["lam"]) <= 1e-9)) and bool(np.all(np.abs(sm["occ_C"] - 1.0) <= 1e-9))
        check("(10) source modes: inside weight = eigenvalue, occupation 1", ok, "%d modes, lam min %.4f, max |occ - 1| %.1e, occ_vac extremes %.3f %.3f" % (sm["S"].shape[1], sm["lam"].min(), np.max(np.abs(sm["occ_C"] - 1.0)), sm["occ_vac"].min(), sm["occ_vac"].max()))
        tr = D13.source_transfer(sm["S"], Phi_x, R2, [("collar", 64.0)])
        check("(10b) the inferred occupation w/tau is at most 1", bool(np.all(np.nan_to_num(tr["occ_inferred_collar"], nan=0.0) <= 1.0 + 1e-9)), "max %.4f" % np.nanmax(tr["occ_inferred_collar"]))
    else:
        check("(10) source modes found inside the region", False, "none")
    # (11) the density sums to E_inv
    rho = D13.killing_density_sites(Bh, Phi_x, Vneg)
    check("(11) the Killing density of D sums to E_inv", abs(float(np.sum(rho)) - E_inv_x) <= 1e-9, "%.2e" % abs(float(np.sum(rho)) - E_inv_x))
    # (12) the light time against the closed form on the vacuum exterior
    c = 16.0; Rv = np.arange(1, 1001, dtype=float); Nv = np.ones_like(Rv); vv = -np.sqrt(c / Rv)
    a_, b_ = 100.0, 500.0
    F = lambda x: x + 2.0 * math.sqrt(c * x) + 2.0 * c * math.log(math.sqrt(x) - math.sqrt(c))   # noqa: E731
    exact = F(b_) - F(a_)
    num = D13.light_time(Nv, vv, Rv, a_, b_, True)
    check("(12) the outgoing light time matches the closed form on N = 1, v = -sqrt(c/R)", abs(num - exact) / exact <= 1e-3, "rel %.1e" % (abs(num - exact) / exact))
    # (13) a whole checkpoint
    cfg = D13.cfg_from_args(type("A", (), {})())
    cfg.update({"cuts_rs": [1.5, 3.0, 4.5], "ledger_cuts_rs": [1.5, 3.0], "probe_cuts_rs": [1.5, 3.0], "R_cut_rs": 2.0, "R_collar_rs": 1.5, "collar_window": 16, "g": 0.25})
    cr = D13.ChannelReadout(kappa, R, device, rs, cfg); cr.TH_ref = TH_ref
    cr.set_vacuum(0.0, 0, vac["e"], Vneg, "dense", E_ref_d)
    cr.add_sources(0, 0.0, Phi_x, h, cr.vac, 40.0, "test")
    try:
        sc, arrays = cr.checkpoint(Phi_x, h, Bh, 0.0, 0, N, v, 1.0 / (2.0 * 0.25) * rs, rs, TH_ref, cr.vac, dens={"g": 0.25}, T_start=0.0, join={"max_dN": 0.0, "max_rel_dv": 0.0}, form={"t": 0.0, "r_root": 40.0})
        print("  " + D13.format_line(kappa, sc, E_inv_x))
        D13.split_arrays(sc, arrays, ())
        okc = ("cuts" in sc and "inventory" in sc and "shells" in sc and "collar_split" in sc and "sources" in sc and "first_return" in sc)
        check("(13) a whole ChannelReadout.checkpoint runs and the readouts are present", okc, "keys %s; arrays %d" % (sorted(sc.keys()), len(arrays)))
        check("(13b) the checkpoint's density sum equals E_inv", abs(sc.get("rho_sum_check", 0.0) - E_inv_x) <= 1e-9, "%.2e" % abs(sc.get("rho_sum_check", 0.0) - E_inv_x))
    except Exception as ex:                                                 # noqa: BLE001
        import traceback; traceback.print_exc()
        check("(13) a whole ChannelReadout.checkpoint runs", False, "%s: %s" % (type(ex).__name__, ex))
    # the resumable store
    try:
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            st = D13.Store(os.path.join(td, "t_d13"))
            st.put_clock(1.0, kappa, {"x": 1.0}, {"arr": np.arange(3.0)}); st.flush()
            st2 = D13.Store(os.path.join(td, "t_d13"))
            check("(14) the store is resumable (json and npz read back)", st2.has_clock(1.0, kappa) and "T%s_k%d_arr" % (D13.Store.ckey(1.0), kappa) in st2.Z, "")
    except Exception as ex:                                                 # noqa: BLE001
        check("(14) the store is resumable", False, str(ex))


def main():
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda:0"))
    for dv in devices:
        run_device(dv)
    if FAILS:
        print("SELFTEST d13 v1 FAILED (%d):" % len(FAILS))
        for f_ in FAILS:
            print("  - " + f_)
        sys.exit(1)
    print("SELFTEST d13 v1 PASSED")
    sys.exit(0)


if __name__ == "__main__":
    main()
