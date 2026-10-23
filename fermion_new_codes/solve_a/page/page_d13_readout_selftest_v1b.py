#!/usr/bin/env python
"""page_d13_readout_selftest_v1b.py: the self-test of page_d13_readout_v1b's (a') on a manufactured pure Gaussian state (no run data):
(1) the block route's singular values of Pi_X (C - P_-) Pi_Xc equal the dense route's (C - P_- formed as n2 x n2, the block sliced, svdvals)
    at every cut, with the rows ordered (slices) and permuted (index_select); the Gram route within sqrt rounding;
(2) EC1's bound |S_X(C) - S_X(P_-)| <= t_X at every cut with S_X from exterior_entropies (a), which is checked against the dense entropies;
    the pure-cut identity S_X(Z) = sum_j F(sigma_j(K_Z)) for Z = C and Z = P_-; the consistency and triangle checks of cross_cut_check;
(3) the full ||C - P_-||_1 by the gram and svd routes against sum |eig(C - P_-)| (dense), with N_neg = N_occ, N_occ + 3 and N_occ - 2; the
    rank, the largest singular value, ||C - P_-||_2 and n_pair against the dense values; nu >= t_X at every cut; the bound route's bounds
    above nu and its sigma_max estimate at or below the exact value; the zero case C = P_- (the T_0 vacuum);
(4) the FAIL path: C = 0.9 x a projector (not a pure state) against that projector breaks the bound; the FAIL line names the premise defect;
(5) the orchestration (cross_cut_readout), format_line, the Store round trip (sv_block_cut<x> and sv_full in the npz, the scalars in the
    json), the argument plumbing (add_arguments, cfg_from_args) and the memory figure at n = 16384.
Usage: python page_d13_readout_selftest_v1b.py [--device cpu] [--n 40] [--seed 1]; exits 0 when every check passes. page_ops_v6.py must be
importable (page_d13_readout_v1b imports it): same directory, $PAGE_CODE_V6 or code_v6 beside."""
import argparse
import json
import math
import os
import sys
import tempfile

import numpy as np
import scipy.sparse as sp
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
for _d in [HERE] + [x for x in os.environ.get("PAGE_CODE_V6", "").split(os.pathsep) if x] + [os.path.join(HERE, "code_v6"), os.path.join(HERE, "..", "code_v6")]:
    if os.path.isfile(os.path.join(_d, "page_d13_readout_v1b.py")) and os.path.abspath(_d) not in sys.path:
        sys.path.insert(0, os.path.abspath(_d))
import page_d13_readout_v1b as D13  # noqa: E402

CD, RD = D13.CD, D13.RD
CHECKS = []


def check(name, ok, detail=""):
    ok = bool(ok)
    CHECKS.append((name, ok))
    print("  [%s] %s%s" % ("ok" if ok else "FAIL", name, (": " + detail) if detail else ""), flush=True)
    return ok


def banded_hermitian(n2, bw, rng):
    A = np.zeros((n2, n2), dtype=complex)
    for d in range(0, bw + 1):
        A += np.diag(rng.normal(size=n2 - d) + 1j * rng.normal(size=n2 - d), d)
    return 0.5 * (A + A.conj().T)


def shift_to_count(h, n_neg):
    """h + c with exactly n_neg negative eigenvalues (c mid-gap)."""
    e = np.linalg.eigvalsh(h)
    return h - 0.5 * (e[n_neg - 1] + e[n_neg]) * np.eye(h.shape[0])


def unitary(h1, tau):
    e, Q = np.linalg.eigh(h1)
    return (Q * np.exp(-1j * tau * e)) @ Q.conj().T


def vacuum_columns(h, device):
    """the occupied columns of 1(h < 0): page_d13_readout_v1b.killing_vacuum (dense route) where page_ops_v6 serves, else numpy's eigh."""
    e, V = np.linalg.eigh(h); neg = e < 0
    try:
        vac = D13.killing_vacuum(sp.csr_matrix(h), device, "dense")
        if vac["n_neg"] == int(neg.sum()):
            return vac["Vneg"], "killing_vacuum (dense)"
        print("  note: killing_vacuum's count %d differs from numpy's %d: numpy's columns used" % (vac["n_neg"], int(neg.sum())))
    except Exception as ex:                                                 # noqa: BLE001
        print("  note: killing_vacuum not used (%s): numpy's eigh" % ex)
    return torch.as_tensor(V[:, neg], dtype=CD, device=device), "numpy eigh"


def h2_np(x):
    x = np.clip(np.asarray(x, float), 0.0, 1.0)
    out = np.zeros_like(x); m = (x > 0) & (x < 1)
    out[m] = -(x[m] * np.log(x[m]) + (1.0 - x[m]) * np.log1p(-x[m]))
    return out


def entropy_dense(Z, mask):
    return float(h2_np(np.linalg.eigvalsh(Z[np.ix_(mask, mask)])).sum())


def pad(a, b):
    n = max(len(a), len(b))
    return np.pad(np.asarray(a, float), (0, n - len(a))), np.pad(np.asarray(b, float), (0, n - len(b)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cpu"); ap.add_argument("--n", type=int, default=40, help="sites (n2 = 2 n modes)"); ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args(argv)
    dev = torch.device(a.device if (a.device.startswith("cpu") or torch.cuda.is_available()) else "cpu")
    rng = np.random.default_rng(a.seed)
    n = int(a.n); n2 = 2 * n
    R = np.arange(1, n + 1, dtype=float); R2_np = np.repeat(R, 2); R2 = torch.as_tensor(R2_np, dtype=RD, device=dev)
    cuts = [("%g" % x, float(x)) for x in (round(n / 4), round(n / 2), round(3 * n / 4))]
    cfg = D13.cfg_from_args(argparse.Namespace())
    print("page_d13_readout_v1b self-test on %s: n = %d sites (n2 = %d), cuts %s" % (dev, n, n2, [x for _, x in cuts]))

    h0 = banded_hermitian(n2, 4, rng)
    V0, how = vacuum_columns(h0, dev); p = int(V0.shape[1])
    print("manufactured: the ground state of a random banded Hermitian h_0 (%s): %d occupied columns" % (how, p))
    h_pert = banded_hermitian(n2, 4, rng)
    cases = []
    for tau in (0.0, 0.05, 0.7):
        U = unitary(banded_hermitian(n2, 4, rng), tau) if tau > 0 else np.eye(n2)
        Phi = torch.as_tensor(U, dtype=CD, device=dev) @ V0
        if tau == 0.0:
            cases.append(("tau 0 against P_0 itself (C = P_-: everything zero)", Phi, V0))
            continue
        for dq, nm in ((0, "N_neg = N_occ"), (3, "N_neg = N_occ + 3"), (-2, "N_neg = N_occ - 2")):
            Vn, _ = vacuum_columns(shift_to_count(h0 + 0.3 * h_pert, p + dq), dev)
            cases.append(("tau %g, %s" % (tau, nm), Phi, Vn))

    for name, Phi, Vneg in cases:
        q = int(Vneg.shape[1])
        print("case %s: N_occ %d, N_neg %d" % (name, p, q))
        C = (Phi @ Phi.mH).cpu().numpy(); P = (Vneg @ Vneg.mH).cpu().numpy(); Dd = C - P
        ent = D13.exterior_entropies(Phi, R2, cuts, Vneg)
        cc = D13.cross_cut_norms(Phi, Vneg, R2, cuts, "svd")
        ccg = D13.cross_cut_norms(Phi, Vneg, R2, cuts, "gram")
        perm = torch.as_tensor(rng.permutation(n2), device=dev)
        ccp = D13.cross_cut_norms(Phi[perm], Vneg[perm], R2[perm], cuts, "svd")
        for lab, x in cuts:
            mask = R2_np > x
            sd = np.linalg.svd(Dd[np.ix_(mask, ~mask)], compute_uv=False)
            sb, sd_ = pad(cc[lab]["sv_block"], sd)
            check("cut %s: block route = dense route (singular values)" % lab, np.max(np.abs(sb - sd_)) < 1e-11, "max difference %.2e" % np.max(np.abs(sb - sd_)))
            check("cut %s: t_X (block) = 2 sum sigma (dense)" % lab, abs(cc[lab]["t_X"] - 2.0 * sd.sum()) < 1e-10, "t_X %.6e" % cc[lab]["t_X"])
            sg, sd2 = pad(ccg[lab]["sv_block"], sd)
            check("cut %s: Gram route within sqrt rounding" % lab, np.max(np.abs(sg - sd2)) < 3e-7, "max difference %.2e" % np.max(np.abs(sg - sd2)))
            check("cut %s: permuted rows (index route) = ordered rows (slices)" % lab, abs(ccp[lab]["t_X"] - cc[lab]["t_X"]) < 1e-10 and (not ccp[lab]["rows_ordered"]) and cc[lab]["rows_ordered"])
            SXd = entropy_dense(C, mask); SPd = entropy_dense(P, mask)
            check("cut %s: S_X(C) of (a) = dense" % lab, abs(ent[lab]["S_X"] - SXd) < 1e-8, "%.6e" % SXd)
            check("cut %s: S_X(P_-) of (a) = dense" % lab, abs(ent[lab]["S_X_vac"] - SPd) < 1e-8, "%.6e" % SPd)
            check("cut %s: EC1 |S_X(C) - S_X(P_-)| <= t_X" % lab, abs(SXd - SPd) <= cc[lab]["t_X"] + 1e-9, "|dS| %.6e, t_X %.6e" % (abs(SXd - SPd), cc[lab]["t_X"]))
            for Z, nm, S in ((C, "C", SXd), (P, "P_-", SPd)):
                sZ = np.linalg.svd(Z[np.ix_(mask, ~mask)], compute_uv=False)
                F = float(D13.pure_cut_F(torch.as_tensor(sZ)).sum())
                check("cut %s: S_X(%s) = sum F(sigma(K)) (the pure-cut identity)" % (lab, nm), abs(F - S) < 1e-8, "%.6e against %.6e" % (F, S))
        fails = D13.cross_cut_check(ent, cc, 1e-8)
        check("cross_cut_check: no FAIL, the Frobenius check against (a)'s spectrum and the triangle bounds hold",
              not fails and all(abs(ent[lab].get("frob2_C_block_minus_spectrum", 0.0)) < 1e-9 and ent[lab].get("t_X_triangle_ok", False) for lab, _ in cuts))
        eig = np.linalg.eigvalsh(Dd)
        nu_d = float(np.abs(eig).sum()); rank_d = int((np.abs(eig) > 1e-10).sum()); smax_d = float(np.abs(eig).max()); fro_d = float(np.sqrt((eig ** 2).sum()))
        npair_d = float(p - np.trace(C @ P).real)
        tn_s = D13.full_trace_norm(Phi, Vneg, "svd"); tn_g = D13.full_trace_norm(Phi, Vneg, "gram")
        tn_b = D13.full_trace_norm(Phi, Vneg, "bound"); tn_a = D13.full_trace_norm(Phi, Vneg, "auto")
        check("full nu (svd route) = sum |eig(C - P_-)|", abs(tn_s["nu"] - nu_d) < 1e-9, "%.6e against %.6e" % (tn_s["nu"], nu_d))
        check("full nu (gram route) = sum |eig| within its rounding bound", abs(tn_g["nu"] - nu_d) <= max(1e-6, tn_g["rounding_bound_nu"]), "%.6e against %.6e (bound %.1e)" % (tn_g["nu"], nu_d, tn_g["rounding_bound_nu"]))
        check("rank(C - P_-) (svd route) = the dense count", tn_s["rank"] == rank_d, "%d against %d" % (tn_s["rank"], rank_d))
        check("largest singular value (svd route) = dense", abs(tn_s["sigma_max"] - smax_d) < 1e-9, "%.6e" % smax_d)
        check("||C - P_-||_2 direct = dense, and from the singular values", abs(tn_b["frob"] - fro_d) < 1e-9 and abs(tn_s["frob2_sv_minus_direct"]) < 1e-8, "%.6e" % fro_d)
        check("n_pair = Tr[(1 - P_-) C]", abs(tn_b["n_pair"] - npair_d) < 1e-9, "%.6e" % npair_d)
        check("nu >= max t_X over the cuts", tn_s["nu"] + 1e-9 >= max(cc[lab]["t_X"] for lab, _ in cuts))
        check("bound route: rank x sigma_max and Cauchy-Schwarz above nu", tn_b["bound_rank_sigma"] >= nu_d - 1e-9 and tn_b["bound_cauchy_schwarz"] >= nu_d - 1e-9,
              "%.3e, %.3e against nu %.6e" % (tn_b["bound_rank_sigma"], tn_b["bound_cauchy_schwarz"], nu_d))
        check("bound route: the sigma_max estimate at or below the exact value and within 10 percent",
              tn_b["sigma_max_estimate"] <= smax_d + 1e-8 and tn_b["sigma_max_estimate"] >= 0.9 * smax_d - 1e-12, "%.6e against %.6e in %d steps" % (tn_b["sigma_max_estimate"], smax_d, tn_b["power_iterations"]))
        check("auto route resolves", tn_a["route"] in ("gram", "bound") and (tn_a["nu"] is not None or tn_a["route"] == "bound"), tn_a["route"])

    # (5) the orchestration, format_line, the store
    name, Phi, Vneg = cases[-1]
    print("orchestration on the case %s:" % name)
    ent = D13.exterior_entropies(Phi, R2, cuts, Vneg)
    sc = {"t_over_rs": 0.0}; arrays = {}
    cc, tn = D13.cross_cut_readout(Phi, Vneg, R2, cuts, ent, cfg, 1, 0.0, sc, arrays)
    sc["cuts"] = {}
    for lab, d in ent.items():
        for key in ("eig_X", "eig_Xc", "eig_X_vac", "eig_Xc_vac"):
            if isinstance(d.get(key), np.ndarray):
                arrays["%s_cut%s" % (key, lab)] = d.pop(key)
        sc["cuts"][lab] = d
    print(D13.format_line(1, sc, 0.0))
    check("readout: the trace norm set, no FAIL, the singular values in arrays",
          sc["trace_norm"]["nu"] is not None and not sc["cross_cut_fails"] and all(("sv_block_cut%s" % lab) in arrays for lab, _ in cuts) and "sv_full" in arrays)
    with tempfile.TemporaryDirectory() as tmp:
        st = D13.Store(os.path.join(tmp, "selftest_d13"), settings={"selftest": True}, label="selftest")
        D13.split_arrays(sc, arrays, ())
        st.put_clock(0.0, 1, sc, arrays); st.flush()
        key0 = D13.Store.ckey(0.0)
        with np.load(st.pz, allow_pickle=False) as z:
            files = set(z.files)
            ok = all(("T%s_k1_sv_block_cut%s" % (key0, lab)) in files for lab, _ in cuts) and ("T%s_k1_sv_full" % key0) in files
            if ok:
                cc2 = D13.cross_cut_norms(Phi, Vneg, R2, cuts, "svd")
                ok = all(np.allclose(z["T%s_k1_sv_block_cut%s" % (key0, lab)], cc2[lab]["sv_block"], atol=1e-12) for lab, _ in cuts)
        J = json.load(open(st.pj))
        c1 = J["clocks"][key0]["per_channel"]["1"]
        check("store: the singular values in the npz, t_X and bound_ok in the json",
              ok and all(isinstance(c1["cuts"][lab].get("t_X"), float) and c1["cuts"][lab].get("bound_ok") is True for lab, _ in cuts) and c1["trace_norm"]["nu"] is not None)

    # (4) the FAIL path
    print("the expected FAIL (C = 0.9 x a projector, not a pure state, against that projector as P_-):")
    cuts_f = [("2", 2.0)] + cuts
    Phi_bad = math.sqrt(0.9) * Phi
    ent_b = D13.exterior_entropies(Phi_bad, R2, cuts_f, Phi)
    sc_b = {"t_over_rs": 0.0}; arr_b = {}
    D13.cross_cut_readout(Phi_bad, Phi, R2, cuts_f, ent_b, dict(cfg, cut_norm_full="bound"), 1, 0.0, sc_b, arr_b)
    check("FAIL path: the cut 2 fails the check", "2" in sc_b["cross_cut_fails"], "fails %s" % sc_b["cross_cut_fails"])
    check("FAIL path: the premise defect ||Phi^dagger Phi - 1||_F is reported, P_- clean",
          sc_b["premise_defects"]["Phi"] > 1e-3 and sc_b["premise_defects"]["Vneg"] < 1e-9, "%.3e, %.1e" % (sc_b["premise_defects"]["Phi"], sc_b["premise_defects"]["Vneg"]))

    # (5) the arguments and the memory figure
    ap2 = argparse.ArgumentParser(); D13.add_arguments(ap2, common=False)
    c2 = D13.cfg_from_args(ap2.parse_args([]))
    check("arguments: the (a') thresholds with their defaults",
          c2["cut_norm"] == "on" and c2["cut_block_route"] == "svd" and c2["cut_norm_full"] == "auto" and c2["cut_norm_full_GB"] == 24.0 and c2["cut_bound_tol"] == 1e-6 and c2["cut_sigma_tol"] == 1e-10)
    c3 = D13.cfg_from_args(ap2.parse_args(["--cut-norm-full", "svd", "--cut-bound-tol", "1e-7", "--cut-block-route", "gram", "--cut-norm", "off"]))
    check("arguments: the (a') thresholds set", c3["cut_norm_full"] == "svd" and c3["cut_bound_tol"] == 1e-7 and c3["cut_block_route"] == "gram" and c3["cut_norm"] == "off")
    fig = D13.memory_figure(16384, None, 2048, 24.0, 64.0, 128)
    print(D13.format_memory_figure(fig))
    check("memory figure at n = 16384: the cross-cut keys", all(k in fig for k in ("cross_cut_block_GiB", "cross_cut_svd_peak_GiB", "trace_norm_gram_peak_GiB", "cross_cut_checkpoint_peak_GiB")) and 2.0 < fig["cross_cut_block_GiB"] < 3.0)

    bad = [nm for nm, ok in CHECKS if not ok]
    print("SELFTEST %s: %d checks, %d failed%s" % ("OK" if not bad else "FAILED", len(CHECKS), len(bad), (": " + "; ".join(bad)) if bad else ""))
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
