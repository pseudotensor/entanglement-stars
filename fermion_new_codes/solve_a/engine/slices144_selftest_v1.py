#!/usr/bin/env python
"""slices144_selftest_v1.py (reviewer A for the solve builder). The self-test of slices144_io_v1, interior_continuation_v1,
core_kinematics_v1 and page_own_object_v1 on manufactured slices in the inventory's layout (np.savez uncompressed, the same keys, 2
components per site interleaved, kappa_sea 3 with weights 16 kappa, dust columns 2, 1, 0): a complex128 slice and its complex64 copy.

Identities checked (PASS/FAIL per line; the exit code is 1 on any failure, 2 where a required module was absent and a check could not run):
reader: every member read through the memory map equals np.load's (both slices); the complex64 flag and its floor; Phi = [sea | dust] with
the right multiplicity; the Loewdin defects (complex128 near 1e-15, complex64 near 1e-7).
interior: the dense roots and slopes of a manufactured N - |v| with analytic roots and slopes (the route named); the applied roots between
sites within the linear interpolation's error; the centroid of a manufactured density; the Killing split's completeness (sum_j n_j = the
dust weight beyond the cut), its block identity (sum omega n = Re Tr[h_YY D_YY]) and its 3 parts' sum; the number current through a cut
against -2 Im Tr(h_AB D_BA) (engine_core_slices_v3's form) and against the finite difference of the number beyond the cut under exp(-i h t);
the pocket statistics of a manufactured mask.
core: the interior orbitals' weights inside the window equal their eigenvalues (page_d13_readout_v1b.source_modes); the momentum
distributions' normalization (1/M) sum_k n(k) = the number; <cos 2ak> on the grid equals its real-space form for an orbital and for a
window set; the fraction at |ak| > pi/4 of a pure plane wave.
page: S_X(C) = S_Xc(C) on a random pure state; the vacuum's excess 0 for a state that is the Killing vacuum under a unitary mixing of its
columns; the cross-cut bounds S_X <= (pi C*/2) t_X and S_X <= the pair ledger; the first return on N = 1, v = 0 equals t_start + 2(R_end - x);
the light-channel lookup on a manufactured D13 store.
Args: --engine-code, --page-code, --threads, --tmp (the folder for the manufactured slices; a temporary one by default), --nsites (40),
--keep (keep the manufactured files)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import slices144_io_v1 as IO  # noqa: E402
import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import shutil  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append({"name": name, "ok": bool(ok), "detail": str(detail)})
    print("%s  %s  %s" % ("PASS" if ok else "FAIL", name, detail), flush=True)
    return bool(ok)


def skip(name, why):
    RESULTS.append({"name": name, "ok": None, "detail": why})
    print("SKIP  %s  %s" % (name, why), flush=True)


def fields(R):
    N = np.ones(len(R))
    v = -1.5 * np.exp(-((R - 20.0) / 5.0) ** 2)
    return N, v


def make_slice(path, n, m, kappas, c64):
    R = np.arange(1, n + 1, dtype=float)
    N, v = fields(R)
    data = {"T": np.array(768.0), "rs": np.array(64.0), "R": R, "N_applied": N, "v_applied": v, "kappas": np.array(kappas, np.int64),
            "weights": np.array([16.0 * k for k in kappas]), "dtype": np.array("complex64" if c64 else "complex128")}
    nd_of = {1: 2, 2: 1, 3: 0}
    cast = np.complex64 if c64 else np.complex128
    for k in kappas:
        h0, Q, H = IO.operators(R, m, k, N, v)
        e, V = IO.eigh(H.toarray())
        e0, V0 = IO.eigh(h0.toarray())
        nd = nd_of.get(k, 0)
        pos = np.nonzero(e > 0)[0][:nd]
        mask = np.zeros(2 * n, bool)
        mask[:n] = True
        data["sea_%d" % k] = V[:, :n].astype(cast)
        data["dust_%d" % k] = V[:, pos].astype(cast)
        data["f_%d" % k] = np.ones(nd)
        data["R2_%d" % k] = V0[:, :n].astype(cast)
        data["e_%d" % k] = e
        data["mask_%d" % k] = np.packbits(mask)
    np.savez(path, **data)
    return path


class FakeCtx:
    def __init__(self, log):
        self.log = log
        self.H = {"label": "manufactured", "fields_module": None}
        self.m, self.g, self.ell, self.rs, self.tau = 0.2, 0.25, 4.0, 64.0, 8.0

    def engine_label(self, entry, rec):
        return "manufactured"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine-code", dest="engine_code", default=IO.ENGINE_CODE)
    ap.add_argument("--page-code", dest="page_code", default=IO.PAGE_CODE)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--tmp", default=None)
    ap.add_argument("--nsites", type=int, default=40)
    ap.add_argument("--keep", action="store_true")
    a = ap.parse_args(argv)
    t_all = time.time()
    log = IO.Logger(None)
    IO.setup(a.engine_code, a.page_code, a.threads, log)
    log("setup: %s" % json.dumps(IO.SETUP, default=str))
    import interior_continuation_v1 as IC
    import core_kinematics_v1 as CK
    import page_own_object_v1 as PG
    tmp = a.tmp or tempfile.mkdtemp(prefix="slices144_selftest_")
    os.makedirs(tmp, exist_ok=True)
    n, m = a.nsites, 0.2
    kappas = [1, 2, 3]
    p128 = make_slice(os.path.join(tmp, "orbitals_manufactured_c128_T0012.00.npz"), n, m, kappas, False)
    p64 = make_slice(os.path.join(tmp, "orbitals_manufactured_c64_T0012.00.npz"), n, m, kappas, True)
    # ---------------------------------------------------------------------------------------------------------- the reader
    for p, c64 in ((p128, False), (p64, True)):
        rd = IO.SliceReader(p, {"history": "test"}, log)
        with np.load(p, allow_pickle=False) as Z:
            worst = 0.0
            allok = True
            for key in Z.files:
                x = rd.array(key)
                ok = np.array_equal(np.asarray(x), Z[key])
                allok &= ok
                if not ok:
                    worst = max(worst, 1.0)
            check("reader: every member through the memory map equals np.load (%s)" % ("c64" if c64 else "c128"), allok, "%d members" % len(Z.files))
        check("reader: complex64 flag (%s)" % ("c64" if c64 else "c128"), rd.complex64 == c64, "dtype %s, floor applies %s, source floor %s" % (rd.dtype_str, rd.floor["applies"], rd.floor["source_rel_range"]))
        check("reader: T/r_s, sites, kappas, weights", abs(rd.T_over_rs - 12.0) < 1e-12 and rd.nsites == n and rd.kappas == kappas and rd.weights[2] == 32.0, "T/r_s %.3f, n %d, weights %s" % (rd.T_over_rs, rd.nsites, rd.weights))
        ch = rd.channel(1, orthonormalize=True)
        check("reader: Phi = [sea | dust] with the multiplicity (%s)" % ("c64" if c64 else "c128"), ch["Phi"].shape == (2 * n, n + 2) and ch["weight"] == 16.0 and ch["n_dust"] == 2 and ch["f"].shape[0] == n + 2,
              "shape %s, w %g, orthonormalization %s" % (ch["Phi"].shape, ch["weight"], ch["orthonormalization"]))
        tol0 = 1e-12 if not c64 else 1e-5
        check("reader: the stored orthonormality defect is at the precision's floor (%s)" % ("c64" if c64 else "c128"), ch["stored_orthonormality"] < tol0 and ch["loewdin_orthonormality"] < 1e-11,
              "stored %.1e, after Loewdin %.1e" % (ch["stored_orthonormality"], ch["loewdin_orthonormality"]))
        rd.close()
    rd = IO.SliceReader(p128, {"history": "test"}, log)
    R, N_app, v_app = rd.R, rd.N_applied, rd.v_applied
    # -------------------------------------------------------------------------------------------------------- interior
    u = 5.0 * math.sqrt(math.log(1.5))
    roots_true = [20.0 - u, 20.0 + u]
    slopes_true = [-2.0 * u / 25.0, 2.0 * u / 25.0]
    info = {"N_dense": lambda r: np.ones_like(np.asarray(r, float)), "v_dense": lambda r: -1.5 * np.exp(-((np.asarray(r, float) - 20.0) / 5.0) ** 2)}
    slopes, route = IC.dense_slope(info, roots_true, 4.0 / 32.0)
    check("interior: the signed slopes at the analytic roots", all(abs(s - st) < 1e-8 for s, st in zip(slopes, slopes_true)), "%s against %s (%s)" % (["%.6f" % s for s in slopes], ["%.6f" % s for s in slopes_true], route))
    az = IO.interp_zeros(R, N_app - np.abs(v_app))
    check("interior: the applied roots between sites", len(az) == 2 and all(abs(q["R"] - rt) < 0.05 for q, rt in zip(az, roots_true)) and az[0]["slope"] < 0 < az[1]["slope"],
          "%s against %s" % (["%.3f (%+.4f)" % (q["R"], q["slope"]) for q in az], ["%.3f" % r for r in roots_true]))
    dens = np.exp(-((R - 15.0) / 3.0) ** 2)
    check("interior: the centroid of a manufactured density", abs(IC.centroid(R, dens) - 15.0) < 1e-6, "%.8f" % IC.centroid(R, dens))
    dust, f = rd.dust(1)
    h0, Q, H = IO.operators(R, m, 1, N_app, v_app)
    sp, arr = IC.dust_killing_split(H, R, dust, f, 25.0, m, 1.0, 0.1)
    check("interior: the Killing split's completeness, block identity and 3-part sum", abs(sp["completeness_check"]) < 1e-10 and abs(sp["E_block"] - sp["E_block_trace"]) < 1e-10 and
          abs(sp["N_above_m"] + sp["N_below_m"] + sp["N_neg_m"] - sp["N_out"]) < 1e-12, "completeness %.1e, block %.1e, N_out %.6f (>= m %.6f)" % (sp["completeness_check"], sp["E_block"] - sp["E_block_trace"], sp["N_out"], sp["N_above_m"]))
    x = 18.0
    cur = IC.cut_current(H, R, x, dust, f, "number")
    A = np.repeat(R <= x, 2)
    Hd = H.toarray()
    F_cs3 = -2.0 * float(np.sum(f * np.imag(np.sum(np.conj(dust[A]) * (Hd[np.ix_(A, ~A)] @ dust[~A]), axis=0))))
    e, V = IO.eigh(Hd)
    dt = 1e-3
    def nout(t):
        ph = V @ (np.exp(-1j * e * t)[:, None] * (V.conj().T @ dust))
        return float(np.sum(f * np.sum(np.abs(ph[~A]) ** 2, axis=0)))
    fd = (nout(dt) - nout(-dt)) / (2.0 * dt)
    check("interior: the number current = -2 Im Tr(h_AB D_BA) = d/dt (number beyond the cut), outward positive", abs(cur - F_cs3) < 1e-12 and abs(cur - fd) < 1e-5, "i[h, Pi] %.8e, -2 Im form %.8e, finite difference %.8e" % (cur, F_cs3, fd))
    ps = IC.pocket_stats(rd.e(1), rd.mask(1), n, 2)
    check("interior: the pocket statistics of the manufactured mask", ps["occupied"] == n and ps["inverted_runs"] == (0 if ps["mask_equals_negative_set"] else ps["inverted_runs"]), "%s" % {k: ps[k] for k in ("occupied", "negative_levels", "inverted_runs", "mask_equals_negative_set")})
    # ------------------------------------------------------------------------------------------------------------ core
    if IO.D13 is None:
        skip("core: interior orbitals' weights equal their eigenvalues", "page_d13_readout_v1b not importable")
    else:
        torch = IO.torch
        ch = rd.channel(1, orthonormalize=True)
        Phi = ch["Phi"]
        R_win = roots_true[1]
        src = IO.D13.source_modes(torch.as_tensor(Phi), torch.as_tensor(np.repeat(R, 2)), R_win, H, torch.device("cpu"), None, 0.5, 64)
        check("core: interior orbitals' weights inside the window equal their eigenvalues", src["S"].shape[1] > 0 and float(np.max(np.abs(src["w_in"] - src["lam"]))) < 1e-10 and float(np.max(np.abs(src["occ_C"] - 1.0))) < 1e-10,
              "%d orbitals (lam >= 0.5), max |w_in - lam| %.1e, max |occ - 1| %.1e" % (src["S"].shape[1], float(np.max(np.abs(src["w_in"] - src["lam"]))) if src["S"].shape[1] else 0.0, float(np.max(np.abs(src["occ_C"] - 1.0))) if src["S"].shape[1] else 0.0))
        M, k = CK.k_grid(n, 256)
        s = src["S"].numpy()[:, 0]
        nk = CK.orbital_momentum(s, M)
        mo = CK.moments(nk, k, 0.25 * math.pi, M)
        rs2 = float(np.real(np.vdot(s[:-4], s[4:])))
        check("core: the orbital momentum distribution's normalization and <cos 2ak> against its real-space form", abs(mo["number"] - 1.0) < 1e-10 and abs(mo["cos2k"] - rs2) < 1e-10,
              "(1/M) sum n(k) %.10f, <cos 2k> grid %.8f real space %.8f, frac(|ak| > pi/4) %.4f" % (mo["number"], mo["cos2k"], rs2, mo["frac_abs_k_above"]))
        Wrows = np.repeat(R < R_win, 2)
        xw = R[R < R_win]
        nkw = CK.window_momentum(Phi[Wrows], ch["f"], xw, k)
        mow = CK.moments(nkw, k, 0.25 * math.pi, M)
        tr = float(np.sum(ch["f"] * np.sum(np.abs(Phi[Wrows]) ** 2, axis=0)))
        rsw = CK.window_cos2k_real(Phi[Wrows], ch["f"], xw)
        check("core: the window set's number and <cos 2ak> against the trace and the real-space form", abs(mow["number"] - tr) < 1e-9 and abs(mow["cos2k"] * tr - rsw) < 1e-9, "N %.8f trace %.8f; cos2k N %.8f real space %.8f" % (mow["number"], tr, mow["cos2k"] * tr, rsw))
        q = 2.0 * math.pi * 100 / M
        pw = np.zeros(2 * n, np.complex128)
        pw[0::2] = np.exp(1j * q * R) / math.sqrt(n)
        mpw = CK.moments(CK.orbital_momentum(pw, M), k, 0.25 * math.pi, M)
        check("core: a pure plane wave at ak = %.4f sits below pi/4 with cos 2ak = %.4f" % (q, math.cos(2 * q)), mpw["frac_abs_k_above"] < 0.05 and abs(mpw["cos2k"] - math.cos(2 * q)) < 0.05, "frac %.4f, <cos 2k> %.4f" % (mpw["frac_abs_k_above"], mpw["cos2k"]))
    # ------------------------------------------------------------------------------------------------------------ page
    if IO.D13 is None:
        skip("page: S_X(C) = S_Xc(C), the vacuum's excess 0, the cross-cut bound, the first return", "page_d13_readout_v1b not importable")
    else:
        torch = IO.torch
        rng = np.random.default_rng(7)
        X = rng.standard_normal((2 * n, n + 3)) + 1j * rng.standard_normal((2 * n, n + 3))
        Qo, _ = np.linalg.qr(X)
        R2_t = torch.as_tensor(np.repeat(R, 2))
        cuts = [("10", 10.0), ("25", 25.0)]
        vac = IO.D13.killing_vacuum(H, torch.device("cpu"), "dense")
        ent = IO.D13.exterior_entropies(torch.as_tensor(Qo), R2_t, cuts, vac["Vneg"])
        gap = max(abs(ent[lab]["purity_gap"]) for lab, _ in cuts)
        check("page: S_X(C) = S_Xc(C) on a random pure state", gap < 1e-9, "S_X %s, complement gap max %.1e" % (["%.6f" % ent[lab]["S_X"] for lab, _ in cuts], gap))
        Vn = vac["Vneg"].numpy()
        Um, _ = np.linalg.qr(rng.standard_normal((Vn.shape[1], Vn.shape[1])) + 1j * rng.standard_normal((Vn.shape[1], Vn.shape[1])))
        ent2 = IO.D13.exterior_entropies(torch.as_tensor(Vn @ Um), R2_t, cuts, vac["Vneg"])
        exc = max(abs(ent2[lab]["excess"]) for lab, _ in cuts)
        check("page: the vacuum's excess is 0 for the Killing vacuum under a unitary mixing of its columns", exc < 1e-9, "excess max %.1e (S_X vac %s)" % (exc, ["%.6f" % ent2[lab]["S_X_vac"] for lab, _ in cuts]))
        cc = PG.cross_cut_bound(ent["10"]["eig_X"])
        ok_b = ent["10"]["S_X"] <= cc["bound_pi_Cstar_over_2_t_X"] + 1e-12 and (cc["bound_pair_ledger_sharp"] is None or ent["10"]["S_X"] <= cc["bound_pair_ledger_sharp"] + 1e-12)
        check("page: the cross-cut bounds hold", ok_b, "S_X %.6f <= (pi C*/2) t_X %.6f (t_X %.6f); pair ledger %s; module t_X: %s" % (ent["10"]["S_X"], cc["bound_pi_Cstar_over_2_t_X"], cc["t_X"], IO.fx(cc["bound_pair_ledger_sharp"], "%.6f"), PG.t_X_from_module(ent["10"]["eig_X"])["route"]))
        fr = IO.D13.first_return(np.ones(n), np.zeros(n), R, 10.0, 768.0, 0.0, 64.0)
        check("page: the first return on N = 1, v = 0 is t_start + 2 (R_end - x)", fr["first_return_from_start_rs"] is not None and abs(fr["first_return_from_start_rs"] * 64.0 - 2.0 * (R[-1] - 10.0)) < 1e-9, "%.6f r_s against %.6f" % (fr["first_return_from_start_rs"] or -1, 2.0 * (R[-1] - 10.0) / 64.0))
        fake = os.path.join(tmp, "light")
        os.makedirs(fake, exist_ok=True)
        with open(os.path.join(fake, "fake_d13.json"), "w") as fjs:
            json.dump({"clocks": {"0012.0000": {"t_over_rs": 12.0, "per_channel": {"1": {"cuts": {"16": {"S_X": 0.5, "S_X_vac": 0.1, "excess": 0.4}}}, "2": {"cuts": {"16": {"S_X": 0.25, "S_X_vac": 0.05, "excess": 0.2}}}}}}}, fjs)
        idx = PG.light_index([fake], log)
        lc = PG.light_at(idx, 12.03, 0.0625, ["16", "24"])
        check("page: the light-channel lookup sums 16 kappa S_X at the matched clock", lc is not None and abs(lc["cuts"]["16"]["S_X"] - (16 * 0.5 + 32 * 0.25)) < 1e-12 and lc["cuts"]["24"] is None, "%s" % (lc["cuts"] if lc else None))
    rd.close()
    # --------------------------------------------------------------------------------------------------------- the summary
    fails = [r for r in RESULTS if r["ok"] is False]
    skips = [r for r in RESULTS if r["ok"] is None]
    with open(os.path.join(tmp, "selftest_results.json"), "w") as f:
        json.dump({"setup": IO.SETUP, "results": RESULTS, "seconds": time.time() - t_all}, f, default=IO.jdefault, indent=1)
    print("%s: %d checks, %d failures, %d skipped; %.1f s; results in %s" % ("PASS" if (not fails and not skips) else ("FAIL" if fails else "INCOMPLETE"), len(RESULTS), len(fails), len(skips), time.time() - t_all, tmp), flush=True)
    if not a.keep and not a.tmp:
        shutil.rmtree(tmp, ignore_errors=True)
    sys.exit(1 if fails else (2 if skips else 0))


if __name__ == "__main__":
    main()
