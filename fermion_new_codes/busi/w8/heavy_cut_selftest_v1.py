#!/usr/bin/env python
"""heavy_cut_selftest_v1.py: the self-tests of heavy_cut_entropy_v1 (W8), run before any slice.

1. product: a manufactured product state in which the sea's exterior block and the dust's are supported on disjoint rows (the sea on the
   even sites, the dust on the odd sites, both straddling the cut), so C_X = Q_X (+) D_X and K_sd = 0 exactly: K_sd below 1e-10,
   S_H = S_Q + S_d, the identity S_H = S_P + (S_Q - S_P) + S_d - K_sd to rounding, the Gram route against the dense exterior block's
   spectrum, the purity gaps of the pure Q and P, every bound of the module.
2. correlated: random orthonormal sea and dust columns across every site with fractional dust occupations: the Gram route against the dense
   entropy of C_X = Y_X Y_X^dagger (1e-10), K_sd strictly positive, 0 <= K_sd <= 2 min(S_Q, S_d) and |S_Q - S_d| <= S_H <= S_Q + S_d, the
   straddling-orbital bound, the complement gaps against the dust's own entropy, the pure variant (f all 1) with the whole state's purity
   gap at rounding, and the derivation's identity Tr s(Gamma) = S_H for the 2-block covariance Gamma = T^dagger T, T = [Q_X^(1/2),
   D_X^(1/2)], with its diagonal blocks Q_X and D_X (formed here on a 96-dimensional toy only; the production code never forms it, D15);
   the per-copy pattern route against the direct sum over the copies.
3. flat vacuum: the heavy chain's held Killing vacuum (m a 0.2, kappa 1, 2048 sites, N = 1 and v = 0) at the cuts 2, 16 and 24 r_s
   (r_s 64): the module's S_P (page_ops_v6.eigh_csr and block_entropy) against page_own_object_v1's route for the same value
   (page_d13_readout_v1b.killing_vacuum and exterior_entropies), equal to 1e-9, the complement's equality, and t_X = 0 for Q = P. Needs
   page_ops_v6 (--page-code); without it the test is SKIPPED, never passed.
   --gate-8192: the massless kappa 1 flat vacuum on the 8192-site chain at 2, 16, 24 and 48 r_s against W1's gate values 2.0544, 2.7464,
   2.8815 and 3.1109 nats (REVIEWA_REVIEW_web40i section 4, W1; assumed at N = 1, v = 0), a comparison reported beside the 3 tests.

Writes <out>/heavy_cut_selftest_v1.json and .md; exit code 1 where a test FAILs. No dates; elapsed seconds only."""
import argparse
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import heavy_cut_entropy_v1 as HC  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

CD = torch.complex128
RD = torch.float64
T0 = time.time()
GATE_8192 = {"2": 2.0544, "16": 2.7464, "24": 2.8815, "48": 3.1109}


def log(msg):
    print("[%8.1f s] %s" % (time.time() - T0, msg), flush=True)


def random_columns(n2, k, rows, gen, dev):
    """k orthonormal complex columns supported on the given rows (a QR of a complex Gaussian block), zero elsewhere."""
    A = torch.complex(torch.randn(len(rows), k, generator=gen, dtype=RD), torch.randn(len(rows), k, generator=gen, dtype=RD))
    Qm, _ = torch.linalg.qr(A)
    X = torch.zeros((n2, k), dtype=CD)
    X[torch.as_tensor(rows, dtype=torch.int64)] = Qm
    return X.to(dev)


def dense_entropy(Y, mask):
    """Tr s(Y_X Y_X^dagger) from the dense exterior block (the oracle)."""
    C = Y[mask] @ Y[mask].mH
    C = 0.5 * (C + C.mH)
    ev = torch.linalg.eigvalsh(C)
    return HC.entropy(ev), HC.excursion(ev)


def sqrtm_psd(M):
    ev, U = torch.linalg.eigh(0.5 * (M + M.mH))
    return (U * torch.sqrt(ev.clamp_min(0.0)).to(CD)[None, :]) @ U.mH


def check(res, name, ok, detail):
    res["checks"].append({"name": name, "pass": bool(ok), "detail": detail})
    log("  %s: %s (%s)" % ("PASS" if ok else "FAIL", name, detail))
    return bool(ok)


def toy_chain(n, cut_site):
    R = np.arange(1, n + 1, dtype=float)
    R2 = np.repeat(R, 2)
    x = cut_site + 0.5
    mask = torch.as_tensor(R2 > x)
    return R, R2, x, mask


def test_product(dev, seed):
    res = {"name": "product (K_sd = 0 exactly)", "checks": []}
    gen = torch.Generator().manual_seed(seed)
    n = 48
    n2 = 2 * n
    R, R2, x, mask = toy_chain(n, 24)
    mask = mask.to(dev)
    even_rows = [r for j in range(0, n, 2) for r in (2 * j, 2 * j + 1)]
    odd_rows = [r for j in range(1, n, 2) for r in (2 * j, 2 * j + 1)]
    S = random_columns(n2, 12, even_rows, gen, dev)
    Dc = random_columns(n2, 5, odd_rows, gen, dev)
    f = np.array([1.0, 0.9, 0.5, 0.3, 1.0])
    V = random_columns(n2, 12, even_rows, gen, dev)
    d, arr = HC.cut_terms(S, Dc, f, V, mask, "test", x)
    tX, _ = HC.local_cross_cut(S, V, mask)
    HC.attach_cross_cut(d, tX, "local")
    Y = torch.cat([S, Dc * torch.sqrt(torch.as_tensor(f, dtype=RD, device=dev)).to(CD)[None, :]], dim=1)
    SH_dense, exc = dense_entropy(Y, mask)
    allok = True
    allok &= check(res, "K_sd = 0 on disjoint supports", abs(d["K_sd"]) < 1e-10 and abs(d["K_sd_c"]) < 1e-10, "K_sd %.2e (complement %.2e)" % (d["K_sd"], d["K_sd_c"]))
    allok &= check(res, "S_H = S_Q + S_d exactly", abs(d["S_H"] - d["S_Q"] - d["S_d"]) < 1e-10, "S_H %.10f, S_Q + S_d %.10f" % (d["S_H"], d["S_Q"] + d["S_d"]))
    allok &= check(res, "the identity's residual", abs(d["decomposition_residual"]) < 1e-12, "%.2e" % d["decomposition_residual"])
    allok &= check(res, "Gram route against the dense exterior block", abs(d["S_H"] - SH_dense) < 1e-10, "Gram %.12f, dense %.12f, excursion %.1e" % (d["S_H"], SH_dense, exc))
    allok &= check(res, "purity gaps of the pure Q and P", abs(d["bounds"]["purity_gap_Q"]) < 1e-10 and abs(d["bounds"]["purity_gap_P"]) < 1e-10, "Q %.2e, P %.2e" % (d["bounds"]["purity_gap_Q"], d["bounds"]["purity_gap_P"]))
    allok &= check(res, "every bound of the module holds", d["bounds"]["all_bounds_hold"], "K_sd upper %.6f, S_H in [%.6f, %.6f], S_d %.6f <= %.6f, t_X %.6f >= |S_Q - S_P| %.6f" % (
        d["bounds"]["K_sd_upper"], d["bounds"]["S_H_lower"], d["bounds"]["S_H_upper"], d["S_d"], d["bounds"]["S_d_straddling_bound"], tX, abs(d["excess_sea"])))
    allok &= check(res, "complement gaps bounded by the dust's own entropy", abs(d["bounds"]["purity_gap_H"]) <= d["bounds"]["dust_mixedness"] + 1e-12 and abs(d["bounds"]["purity_gap_d"]) <= d["bounds"]["dust_mixedness"] + 1e-12,
                   "gap H %.4f, gap d %.4f, sum s(f) %.4f" % (d["bounds"]["purity_gap_H"], d["bounds"]["purity_gap_d"], d["bounds"]["dust_mixedness"]))
    res["values"] = {k: d[k] for k in ("S_H", "S_P", "S_Q", "S_d", "K_sd", "excess_sea", "heavy_excess")}
    res["pass"] = bool(allok)
    return res


def test_correlated(dev, seed):
    res = {"name": "correlated (against the dense entropy)", "checks": []}
    gen = torch.Generator().manual_seed(seed + 1)
    n = 48
    n2 = 2 * n
    R, R2, x, mask = toy_chain(n, 24)
    mask = mask.to(dev)
    all_rows = list(range(n2))
    Phi = random_columns(n2, 17, all_rows, gen, dev)
    S, Dc = Phi[:, :12], Phi[:, 12:]
    f = np.array([1.0, 0.9, 0.5, 0.3, 0.75])
    V = random_columns(n2, 14, all_rows, gen, dev)
    d, arr = HC.cut_terms(S, Dc, f, V, mask, "test", x)
    tX, sv = HC.local_cross_cut(S, V, mask)
    HC.attach_cross_cut(d, tX, "local")
    fsq = torch.sqrt(torch.as_tensor(f, dtype=RD, device=dev)).to(CD)
    Y = torch.cat([S, Dc * fsq[None, :]], dim=1)
    SH_dense, exc = dense_entropy(Y, mask)
    allok = True
    allok &= check(res, "Gram route against the dense exterior block", abs(d["S_H"] - SH_dense) < 1e-10, "Gram %.12f, dense %.12f, excursion %.1e" % (d["S_H"], SH_dense, exc))
    allok &= check(res, "K_sd strictly positive on a correlated cut", d["K_sd"] > 1e-3, "K_sd %.6f" % d["K_sd"])
    allok &= check(res, "0 <= K_sd <= 2 min(S_Q, S_d)", d["bounds"]["K_sd_in_bounds"] and d["bounds"]["K_sd_c_in_bounds"], "K_sd %.6f, upper %.6f" % (d["K_sd"], d["bounds"]["K_sd_upper"]))
    allok &= check(res, "|S_Q - S_d| <= S_H <= S_Q + S_d", d["bounds"]["S_H_in_bounds"], "S_H %.6f in [%.6f, %.6f]" % (d["S_H"], d["bounds"]["S_H_lower"], d["bounds"]["S_H_upper"]))
    allok &= check(res, "straddling-orbital and count bounds of S_d", d["bounds"]["S_d_in_bounds"], "S_d %.6f <= sum s(f w) %.6f, count %.6f" % (d["S_d"], d["bounds"]["S_d_straddling_bound"], d["bounds"]["S_d_count_bound"]))
    allok &= check(res, "EC1 bound of the sea's excess", d["bounds"]["excess_sea_bound_holds"], "|S_Q - S_P| %.6f <= t_X %.6f" % (abs(d["excess_sea"]), tX))
    allok &= check(res, "the heavy excess's bracket", d["bounds"]["heavy_excess_in_bracket"], "|S_H - S_P| %.6f <= t_X + S_d %.6f" % (abs(d["heavy_excess"]), tX + d["S_d"]))
    allok &= check(res, "complement gaps bounded by the dust's own entropy", d["bounds"]["purity_gaps_in_bounds"], "gap H %.4f, gap d %.4f, sum s(f) %.4f" % (d["bounds"]["purity_gap_H"], d["bounds"]["purity_gap_d"], d["bounds"]["dust_mixedness"]))
    # the derivation's identity on the toy: Gamma = T^dagger T with T = [Q_X^(1/2), D_X^(1/2)]
    QX = S[mask] @ S[mask].mH
    DX = (Dc * fsq[None, :])[mask] @ (Dc * fsq[None, :])[mask].mH
    T = torch.cat([sqrtm_psd(QX), sqrtm_psd(DX)], dim=1)
    Gam = T.mH @ T
    Gam = 0.5 * (Gam + Gam.mH)
    evG = torch.linalg.eigvalsh(Gam)
    nX = int(mask.sum())
    blk = max(float((Gam[:nX, :nX] - QX).abs().max()), float((Gam[nX:, nX:] - DX).abs().max()))
    allok &= check(res, "Tr s(Gamma) = S_H with the blocks Q_X and D_X (the bounds' derivation, toy only)", abs(HC.entropy(evG) - d["S_H"]) < 1e-9 and blk < 1e-10 and float(evG.max()) <= 1.0 + 1e-10,
                   "Tr s(Gamma) %.12f, S_H %.12f, block defect %.1e, max eigenvalue %.12f" % (HC.entropy(evG), d["S_H"], blk, float(evG.max())))
    # the pure variant: every f = 1, the whole state's purity gap at rounding
    d1, _ = HC.cut_terms(S, Dc, np.ones(5), V, mask, "pure", x)
    allok &= check(res, "pure variant (f all 1): S_H(X) = S_H(Xc)", abs(d1["bounds"]["purity_gap_H"]) < 1e-10 and d1["bounds"]["dust_mixedness"] == 0.0, "gap %.2e" % d1["bounds"]["purity_gap_H"])
    # the per-copy pattern route against the direct sum
    copies = [[0, 1], [2, 3, 4], [0, 1]]
    p = HC.pattern_terms(S, Dc, f, V, mask, copies, d["S_Q"], d["S_P"], d["S_Q_c"], d["S_P_c"])
    direct_H = 0.0
    direct_d = 0.0
    for c in copies:
        Dsub = Dc[:, c] * torch.sqrt(torch.as_tensor(f[c], dtype=RD, device=dev)).to(CD)[None, :]
        direct_H += dense_entropy(torch.cat([S, Dsub], dim=1), mask)[0]
        direct_d += dense_entropy(Dsub, mask)[0]
    allok &= check(res, "per-copy pattern route against the direct sum over the copies", abs(p["S_H"] - direct_H) < 1e-10 and abs(p["S_d"] - direct_d) < 1e-10 and p["n_copies"] == 3 and p["n_distinct_subsets"] == 2,
                   "S_H %.10f (direct %.10f), S_d %.10f (direct %.10f), K_sd %.6f in [0, %.6f]: %s" % (p["S_H"], direct_H, p["S_d"], direct_d, p["K_sd"], p["K_sd_upper"], p["all_bounds_hold"]))
    res["values"] = {k: d[k] for k in ("S_H", "S_P", "S_Q", "S_d", "K_sd", "excess_sea", "heavy_excess")}
    res["values"]["t_X"] = tX
    res["pass"] = bool(allok)
    return res


def import_page(page_code):
    for d_ in [page_code, os.environ.get("PAGE_CODE_V6"), os.environ.get("SLICES144_PAGE_CODE")]:
        if d_ and os.path.isdir(d_) and d_ not in sys.path:
            sys.path.insert(0, d_)
    try:
        import page_ops_v6 as OPS
    except ImportError as ex:
        return None, None, "page_ops_v6 not importable (%s): pass --page-code" % ex
    D13 = None
    err = ""
    for name in ("page_d13_readout_v1b", "page_d13_readout_v1"):
        try:
            import importlib
            D13 = importlib.import_module(name)
            break
        except Exception as ex:  # noqa: BLE001
            err += "%s: %s; " % (name, ex)
    return OPS, D13, err


def flat_vacuum_columns(OPS, n, m, kappa, dev):
    R = np.arange(1, n + 1, dtype=float)
    h0, Q = OPS.chain_sparse(R, m, kappa)
    H = OPS.clocked_csr(h0, Q, np.ones(n), np.zeros(n)).tocsr()
    info, V, Vall = HC.killing_vacuum(H, dev, None, OPS)
    del Vall
    return R, H, info, V


def test_flat_vacuum(dev, page_code, n=2048, m=0.2, kappa=1, rs=64.0, cuts=(2.0, 16.0, 24.0)):
    res = {"name": "flat vacuum against page_own_object_v1's route", "checks": []}
    OPS, D13, err = import_page(page_code)
    if OPS is None:
        res["pass"] = None
        res["skipped"] = err
        log("  SKIPPED: %s" % err)
        return res
    t1 = time.time()
    R, H, info, V = flat_vacuum_columns(OPS, n, m, kappa, dev)
    R2_t = torch.as_tensor(np.repeat(R, 2), dtype=RD, device=dev)
    res["vacuum"] = {k: v for k, v in info.items() if k != "e"}
    res["vacuum"]["eigh_seconds"] = time.time() - t1
    allok = True
    mine = {}
    for c in cuts:
        lab = "%g" % c
        mask = R2_t > c * rs
        b = HC.block_entropy(V, mask)
        tX, _ = HC.local_cross_cut(V, V, mask)
        mine[lab] = {"S_P": b["S_X"], "S_P_c": b["S_Xc"], "t_X_QequalsP": tX}
        allok &= check(res, "complement equality at %s r_s" % lab, abs(b["S_X"] - b["S_Xc"]) < 1e-9, "S_X %.10f, S_Xc %.10f" % (b["S_X"], b["S_Xc"]))
        allok &= check(res, "t_X = 0 for Q = P at %s r_s" % lab, tX < 1e-9, "t_X %.2e" % tX)
    res["values_module"] = mine
    if D13 is None:
        res["page_route"] = "page_d13_readout_v1b not importable (%s): the comparison with page_own_object_v1's route is not possible here; the module's route is page_ops_v6.eigh_csr and the Gram spectrum, the same arithmetic" % err
        check(res, "page_own_object_v1's route importable", False, res["page_route"])
        allok = False
    else:
        vac = D13.killing_vacuum(H, dev, "dense")
        ent = D13.exterior_entropies(vac["Vneg"], R2_t, [("%g" % c, c * rs) for c in cuts], None)
        res["values_page_route"] = {lab: {"S_X": e_["S_X"], "S_Xc": e_["S_Xc"]} for lab, e_ in ent.items()}
        allok &= check(res, "the vacuum's occupied count and E_ref agree with page_d13_readout_v1b.killing_vacuum", vac["n_neg"] == info["n_neg"] and abs(vac["E_ref"] - info["E_ref"]) < 1e-8 * max(1.0, abs(info["E_ref"])),
                       "n_neg %d against %d, E_ref %.10e against %.10e" % (info["n_neg"], vac["n_neg"], info["E_ref"], vac["E_ref"]))
        for c in cuts:
            lab = "%g" % c
            allok &= check(res, "S_P at %s r_s against page_own_object_v1's route" % lab, abs(mine[lab]["S_P"] - ent[lab]["S_X"]) < 1e-9, "module %.10f, page route %.10f" % (mine[lab]["S_P"], ent[lab]["S_X"]))
        del vac
    res["label"] = "the heavy chain's held Killing vacuum at N = 1, v = 0 (m a %g, kappa %d, %d sites, r_s %g): the module's S_P against page_d13_readout_v1b.killing_vacuum + exterior_entropies (page_own_object_v1's calls)" % (m, kappa, n, rs)
    res["pass"] = bool(allok)
    del V
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return res


def test_gate_8192(dev, page_code, rs=64.0):
    res = {"name": "W1 gate values on the 8192-site massless kappa 1 flat vacuum (a comparison)", "checks": []}
    OPS, D13, err = import_page(page_code)
    if OPS is None:
        res["pass"] = None
        res["skipped"] = err
        log("  SKIPPED: %s" % err)
        return res
    if dev.type != "cuda":
        log("  the 16384-dimensional eigendecomposition on the CPU: minutes with 16 threads")
    t1 = time.time()
    R, H, info, V = flat_vacuum_columns(OPS, 8192, 0.0, 1, dev)
    R2_t = torch.as_tensor(np.repeat(R, 2), dtype=RD, device=dev)
    res["vacuum"] = {k: v for k, v in info.items() if k != "e"}
    res["eigh_seconds"] = time.time() - t1
    allok = True
    vals = {}
    for lab, ref in GATE_8192.items():
        mask = R2_t > float(lab) * rs
        b = HC.block_entropy(V, mask)
        vals[lab] = {"S_X": b["S_X"], "S_Xc": b["S_Xc"], "gate": ref, "difference": b["S_X"] - ref}
        allok &= check(res, "cut %s r_s against the gate value %.4f" % (lab, ref), abs(b["S_X"] - ref) <= 2e-4, "S_X %.6f, difference %+.2e" % (b["S_X"], b["S_X"] - ref))
    res["values"] = vals
    res["label"] = ("the flat (N = 1, v = 0) massless kappa 1 chain of 8192 sites, the held Killing vacuum's exterior-cut entropy at 2, 16, 24 and 48 r_s against the gate values of W1 "
                    "(REVIEWA_REVIEW_web40i section 4); the tolerance 2e-4 is the printed 4 decimals; a failure under this assumption would say that the gate's 'flat' is not N = 1, v = 0 on this chain")
    res["pass"] = bool(allok)
    del V
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--page-code", dest="page_code", default="<internal>/v6", help="the folder of page_ops_v6.py and page_d13_readout_v1b.py (input 05's default)")
    ap.add_argument("--nsites", type=int, default=2048, help="the flat vacuum test's chain")
    ap.add_argument("--gate-8192", dest="gate_8192", action="store_true", help="also the W1 gate comparison on the 8192-site massless chain (a 16384-dimensional eigendecomposition)")
    ap.add_argument("--out", default=os.path.join(HERE, "selftest_out"))
    a = ap.parse_args(argv)
    torch.set_num_threads(int(a.threads))
    dev = HC.pick_device(a.device, log)
    os.makedirs(a.out, exist_ok=True)
    log("heavy_cut_selftest_v1 on %s (%d threads)" % (dev, a.threads))
    results = []
    for fn, args in ((test_product, (dev, a.seed)), (test_correlated, (dev, a.seed)), (test_flat_vacuum, (dev, a.page_code, a.nsites))):
        log("test: %s" % fn.__name__)
        try:
            results.append(fn(*args))
        except Exception as ex:  # noqa: BLE001
            log("  FAIL: %s raised %s: %s" % (fn.__name__, type(ex).__name__, ex))
            results.append({"name": fn.__name__, "pass": False, "error": "%s: %s" % (type(ex).__name__, ex), "checks": []})
    comparison = None
    if a.gate_8192:
        log("comparison: the W1 gate values on the 8192-site chain")
        try:
            comparison = test_gate_8192(dev, a.page_code)
        except Exception as ex:  # noqa: BLE001
            comparison = {"name": "gate 8192", "pass": False, "error": "%s: %s" % (type(ex).__name__, ex), "checks": []}
    n_pass = sum(1 for r in results if r.get("pass") is True)
    n_fail = sum(1 for r in results if r.get("pass") is False)
    n_skip = sum(1 for r in results if r.get("pass") is None)
    verdict = "PASS" if (n_fail == 0 and n_skip == 0) else ("FAIL" if n_fail else "INCOMPLETE (a test was skipped)")
    out = {"code": "heavy_cut_selftest_v1.py", "module": HC.CODE, "device": str(dev), "verdict": verdict, "n_pass": n_pass, "n_fail": n_fail, "n_skipped": n_skip, "tests": results,
           "gate_8192_comparison": comparison, "seconds": time.time() - T0}
    with open(os.path.join(a.out, "heavy_cut_selftest_v1.json"), "w") as f:
        json.dump(out, f, indent=1, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
    L = ["# heavy_cut_selftest_v1: %s (%d PASS, %d FAIL, %d skipped; device %s; %.0f s)" % (verdict, n_pass, n_fail, n_skip, dev, out["seconds"]), ""]
    for r in results + ([comparison] if comparison else []):
        st = "PASS" if r.get("pass") is True else ("FAIL" if r.get("pass") is False else "SKIPPED")
        L.append("## %s: %s" % (r.get("name"), st))
        if r.get("skipped"):
            L.append("- skipped: %s" % r["skipped"])
        if r.get("error"):
            L.append("- error: %s" % r["error"])
        for c in r.get("checks", []):
            L.append("- %s: %s (%s)" % ("PASS" if c["pass"] else "FAIL", c["name"], c["detail"]))
        if r.get("label"):
            L.append("- %s" % r["label"])
        L.append("")
    L.append("- no date in this file; the elapsed seconds are the only time.")
    with open(os.path.join(a.out, "heavy_cut_selftest_v1.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    log("verdict %s: %s and .md" % (verdict, os.path.join(a.out, "heavy_cut_selftest_v1.json")))
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
