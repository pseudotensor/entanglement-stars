#!/usr/bin/env python
"""engine_twofield_v8_selftest.py (the busi lane, Computation C; the fix ruling's G0, busi/notes/C64_fix_ruling_v4.md I.3 and inbox busi #193
item 2): the checks of engine_twofield_v8.py, the ruling's engine_twofield_v4, on manufactured small systems on the CPU (no --gpu) or on the
device C5.backend(--gpu) gives, plus 2 tools of the gate. Subcommands:
  all | fixture | marks | endtoend | compare-runs       (all = fixture, and marks when --v3 and --v8 are given)
  fixture (CPU, about a minute): one record of the 2-field map on a manufactured chain (128 sites; 3 heavy channels at m 0.2; the light channels
          1,2 sourcing and 3 the guard; the seas prepared at fields 4e-3 away from the previous accepted fields' references, so that the sources
          are not zero; a manufactured field solve N = N0 + c K eps, v = v0 + c K J with K the Gaussian of width 3, in place of engine_fields_v16,
          whose charge is the integrated source; the production's Newton loop: --mix newton --nk-line-search halving --nk-halvings 6 --nk-krylov 40
          --nk-restarts 0 --source-continuation two-stage --end-above 1e-3 --nk-jvp exact; the v36 fixes on):
          1. the implicit mode bitwise with version 3: engine_twofield_v3's and v8's newton_record_v32 on the same state: the solve's fields,
             sources, iteration history, solve_v32 entry, every cache entry (eigenvalues, eigenvectors, references, continuation entries, masks,
             sectors) and the counters, number by number and array by array;
          2. the lagged mode: (a) the transported reference equal to the continuation's result at the accepted fields (continue_ref_v34 and
             light_continue_v36 called again on a fresh eigendecomposition at the accepted applied fields: the reference columns, the masks, the
             entries and the rotated eigenvectors bitwise; the evaluations' references the previous accepted record's); (b) the solve's
             evaluations at most the implicit's; (c) the lag residual equal to the implicit solve's first Newton residual from the same start (the
             identity the scheme rests on in this engine) and E_cons - E to the implicit's first |dE/E|;
          3. the lag residual zero when the reference does not move: lag_evaluation_v8 with the held references in place of the transported ones
             (exactly 0), and a whole lagged record on a stationary state (the sea its own reference at the previous fields: the transport is the
             identity, the lag residual 0 to roundoff);
          4. the json "lag" block's keys and their nesting; the record's keys and label in the v8 source;
          5. the exact J.v without the reference terms equal to a central difference of the lagged map (both -dx: the held map's Jacobian is -1;
             no field solve entered by the J.v), the implicit J.v differing from -dx (the reference terms present) with its own central difference
             printed;
  marks --v3 V3 --v8 V8: every line difflib finds added or changed in v8 against v3, outside the module docstring, carries the mark
          "v8 lagged (the ruling's v4)"; nothing deleted without a replacement;
  endtoend --v3 V3 --v8 V8 [--gpu G] [--nsites 1024] [--rs 32] [--records 4] [--work DIR]: the stand-in fixture (the solve lane's small-n trapped
          2-field fixture is not among the inputs): 3 engine runs from T = 0 with the r_s/a 32 sigma/a 4 member's argument set at --nsites sites
          for --records records (version 3; version 8 implicit; version 8 lagged), then version 8 implicit against version 3 bitwise
          (compare-runs: every record number but wall clocks, every row of the fields, spectra, handoff and s11 npz), and the lagged run read: per
          record its evaluations against the implicit's, its lag residual, E_cons - E, the "lag" keys and the record label, and at T = 0 (the one
          record both runs start from the same state) the lag residual against the implicit's first Newton residual; a version 3 run that fails at
          this size (the preparation or a record) is reported as the fixture missing, not as a failure of the build;
  compare-runs A/TAG B/TAG [--after T]: 2 runs' records after the clock T (every number but wall clocks, slice paths and the kernel cache's
          counters; the timing counters compared) and the fields, spectra, handoff and s11 arrays at the common clocks: BITWISE or the
          differences, the largest printed.
Each check prints PASS or FAIL; the exit code is the number of failures. The engine modules are found as the engine finds them (BUSI_ENGINE_DIR,
BUSI_PAGE_DIR, default <internal>/engine and .../page); engine_twofield_v8.py and engine_twofield_v3.py beside this file or in
--code-dir."""
import argparse
import difflib
import json
import os
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace

sys.dont_write_bytecode = True
_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)
for _d in (os.environ.get("BUSI_ENGINE_DIR", "<internal>/engine"), os.environ.get("BUSI_PAGE_DIR", "<internal>/page")):
    if _d and os.path.isdir(_d) and _d not in sys.path:
        sys.path.append(_d)

import numpy as np

MARK = "v8 lagged (the ruling's v4)"
E_FIX_OFFSET = 100.0   # red team R6: the manufactured solve's charge is the integrated source plus this offset (a collar charge is never 0 in a run, E0 118 at r_s/a 64); without it a stationary record's E is 0 and version 3's |dE/E| = |E - E_prev| / |E| in nk_stage_v32's F divides by zero (G0's FAIL 3b)
LAG_KEYS = {"residual", "E_cons", "d_eps_total", "d_J_total", "d_E_inv_light", "max_site_abs_d_eps_outside_cell"}
SKIP_KEYS = {"seconds", "kernel_cache", "slice_saved", "light_slice_saved"}


# ------------------------------------------------------------------------------------------------------------------------------ the compare
def _is_tensor(x):
    return hasattr(x, "device") and hasattr(x, "cpu") and hasattr(x, "numel")


def deep_diff(x, y, path, out, skip=SKIP_KEYS):
    """every difference between 2 nested structures: numbers exactly (NaN equals NaN), arrays and tensors bitwise; a dict named timing by its
    counters alone (its seconds are wall clocks); (path, kind, size) appended."""
    if isinstance(x, dict) and isinstance(y, dict):
        if path.endswith("timing") or path.endswith("TM") or path.endswith("TML"):
            x = {k: v for k, v in x.items() if not str(k).endswith("_s")}; y = {k: v for k, v in y.items() if not str(k).endswith("_s")}
        if set(x) != set(y):
            out.append((path, "keys", sorted(str(k) for k in set(x) ^ set(y))))
        for k in x:
            if k in y and k not in skip:
                deep_diff(x[k], y[k], path + "/" + str(k), out, skip)
        return
    if isinstance(x, (list, tuple)) and isinstance(y, (list, tuple)):
        if len(x) != len(y):
            out.append((path, "length", (len(x), len(y))))
            return
        for i, (p, q) in enumerate(zip(x, y)):
            deep_diff(p, q, path + "[%d]" % i, out, skip)
        return
    if _is_tensor(x) and _is_tensor(y):
        if tuple(x.shape) != tuple(y.shape):
            out.append((path, "shape", (tuple(x.shape), tuple(y.shape))))
        elif x.numel() and not bool((x == y).all()):
            out.append((path, "tensor", float((x - y).abs().max())))
        return
    if isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
        xa = np.asarray(x); ya = np.asarray(y)
        if xa.shape != ya.shape:
            out.append((path, "shape", (xa.shape, ya.shape)))
        elif xa.dtype.kind == "O" or ya.dtype.kind == "O":
            for i, (p, q) in enumerate(zip(xa.ravel(), ya.ravel())):
                deep_diff(p, q, path + "[%d]" % i, out, skip)
        elif xa.dtype.kind in "SU" or ya.dtype.kind in "SU":
            if not np.array_equal(xa, ya):
                out.append((path, "text", 1.0))
        elif not (np.array_equal(xa, ya, equal_nan=True) if xa.dtype.kind in "fc" else np.array_equal(xa, ya)):
            out.append((path, "array", float(np.max(np.abs(xa.astype(complex) - ya.astype(complex))))))
        return
    if isinstance(x, (bool, int, float, np.integer, np.floating)) and isinstance(y, (bool, int, float, np.integer, np.floating)):
        fx, fy = float(x), float(y)
        if fx != fy and not (fx != fx and fy != fy):
            out.append((path, "number", abs(fx - fy) / max(abs(fx), abs(fy), 1e-300)))
        return
    if x != y:
        out.append((path, "value", (repr(x)[:60], repr(y)[:60])))


def report(name, diffs, extra=""):
    if diffs:
        worst = max((d_[2] if isinstance(d_[2], float) else 0.0) for d_ in diffs)
        print("FAIL %s: %d differences, the largest %.3e%s; the first 5: %s" % (name, len(diffs), worst, extra, diffs[:5]), flush=True)
        return 1
    print("PASS %s: bitwise%s" % (name, extra), flush=True)
    return 0


# ------------------------------------------------------------------------------------------------------------------------------ the fixture
def fixture_args(n, kappa_sea):
    """the production's Newton and light arguments on a small chain; rs only enters the stage lines' clock."""
    return SimpleNamespace(nsites=n, half_offset=0, m=0.2, g=0.25, rs=float(n) / 4.0, ell=3, tau=2.0, c_LZ=1.0, reference="R2", streams=0, weyl_certified=None,
                           weyl_refine_max=10, kappa_sea=kappa_sea, light_kappas="1,2", light_guard_kappas="3", light_coupling="on", light_transport="banded",
                           light_d_min=1e-7, light_max_pieces=4096, light_tol_block=1e-8, light_sigma_tol=1e-6, mix="newton", stop_E=1e-7, stop_fields=1e-6,
                           nk_maxiter=30, nk_inner=15, nk_krylov=40, nk_restarts=0, nk_line_search="halving", nk_halvings=6, source_continuation="two-stage",
                           end_above=1e-3, nk_jvp="exact", reference_solve="implicit")


def sparse_probe_and_patch(mods, ch):
    """CPU builds of torch without the sparse CSR @ dense complex product: the engines' dens and dev_csr run densified in this test, both alike."""
    torch = mods[0].BK["torch"]
    X = torch.zeros((ch.h.shape[0], 2), dtype=torch.complex128, device=mods[0].BK["dev"]); X[0, 0] = 1.0
    try:
        mods[0].dens(ch.h, X)
        return False
    except Exception as ex:  # noqa: BLE001
        T_ = mods[0].T_

        def dens_dense(A, Phi, f=None):
            if Phi.shape[1] == 0:
                return np.zeros(Phi.shape[0] // 2)
            AP = A.to_dense() @ Phi
            if f is not None:
                AP = AP * T_(f)[None, :].real.to(AP.dtype)
            d = (Phi.conj() * AP).real.sum(dim=1)
            return d.reshape(-1, 2).sum(dim=1).cpu().numpy()

        def dev_csr_dense(A):
            return T_(A.toarray())
        for m in mods:
            m.dens = dens_dense; m.dev_csr = dev_csr_dense
        print("note: sparse CSR @ dense is not available on this device (%s); the engines' dens and dev_csr run densified in this test, both engines alike" % str(ex)[:90], flush=True)
        return True


def make_fixture(V, a, stationary=False, c=1e-3, offset=4e-3):
    """the previous accepted record's state at the fields (N0, v0): the heavy references (the negative levels) and the light references (the mu S
    selection, light_init_v36) at (N0, v0); the seas the ground states at (N0 + offset bump, v0 - offset x) unless stationary (then the seas are
    the references); no dust; the manufactured field solve about (N0, v0), its charge E_FIX_OFFSET plus the integrated source."""
    torch = V.BK["torch"]
    n = a.nsites
    chs = [V.Chan(k, a.m, n, bool(a.half_offset)) for k in range(1, a.kappa_sea + 1)]
    chl = V.light_channels_v36(a)
    sgn_dev, S_host = V.light_S_v36(n)
    R = chs[0].R; x = R / n
    N0 = 1.0 + 0.05 * np.exp(-((x - 0.4) / 0.12) ** 2); v0 = -0.25 * x * np.exp(-((x - 0.5) / 0.3) ** 2)
    if stationary:
        Ns, vs = N0.copy(), v0.copy()
    else:
        Ns = N0 + offset * np.exp(-((x - 0.5) / 0.2) ** 2); vs = v0 - offset * x
    st = {}
    for ch in chs:
        e, U = V.eigT(V.clocked_gpu(ch, N0, v0)); neg = e < 0; en = e.real.cpu().numpy(); k = int(neg.sum())
        gap = float(en[k] - en[k - 1])
        ref0 = U[:, neg].clone()
        if stationary:
            sea = ref0.clone()
        else:
            es, Us = V.eigT(V.clocked_gpu(ch, Ns, vs)); negs = es < 0
            if int(negs.sum()) != k:
                raise SystemExit("fixture: the sea's negative count %d differs from the reference's %d on kappa %d" % (int(negs.sum()), k, ch.kappa))
            sea = Us[:, negs].clone()
        st[ch.kappa] = {"sea": sea, "dust": torch.zeros((2 * n, 0), dtype=torch.complex128, device=V.BK["dev"]), "f": np.zeros(0), "ref": ref0, "n_sea": k,
                        "window": [{"ref": ref0, "f": (N0.copy(), v0.copy()), "gap": gap}]}
    stL, _headL = V.light_init_v36(a, chl, N0, v0, sgn_dev, S_host)
    if not stationary:
        stLs, _h2 = V.light_init_v36(a, chl, Ns, vs, sgn_dev, S_host)
        for chL in chl:
            if stLs[chL.key]["n_sea"] != stL[chL.key]["n_sea"]:
                raise SystemExit("fixture: the light sea's count differs from the reference's on kappa %d" % chL.kappa)
            stL[chL.key]["sea"] = stLs[chL.key]["sea"]
    st.update(stL)
    K = np.exp(-0.5 * ((R[:, None] - R[None, :]) / 3.0) ** 2); K = K / K.sum(axis=1, keepdims=True)
    TM = V.TM

    def do_solve(srcs_):
        t_s = time.time()
        eps = np.asarray(srcs_[0], float); J = np.asarray(srcs_[1], float)
        Nn = N0 + c * (K @ eps); vn = v0 + c * (K @ J)
        E = E_FIX_OFFSET + float(np.sum(eps))                                                            # red team R6: nonzero on a stationary record (the stub's charge reads info["E"])
        TM["solve_s"] += time.time() - t_s; TM["solve_n"] += 1; TM["solve_nfev"] += 1
        return Nn, vn, None, {"failed": None, "E": E, "E_collar": E, "ivp": {"nfev": 1}}
    return SimpleNamespace(chs=chs, chl=chl, st=st, sgn_dev=sgn_dev, S_host=S_host, R=R, f_ref=(N0.copy(), v0.copy()), do_solve=do_solve, light_on=True)


def record_solve(V, a, FX):
    """run()'s record solve on the engine module V: the base densities, the lagged first iterate, newton_record_v32 (the same lines as run())."""
    V.tm_reset(); V.tml_reset_v36()
    for k_ in V.JV:
        V.JV[k_] = None if k_ == "ref_mismatch" else (0.0 if k_ == "seconds" else 0)
    chs, chl, st = FX.chs, FX.chl, FX.st
    base = {}
    for ch in chs:
        s_ = st[ch.kappa]
        base[ch.kappa] = (V.dens(ch.h, s_["sea"]), V.dens(ch.Q, s_["sea"]), V.dens(ch.h, s_["dust"], s_["f"]), V.dens(ch.Q, s_["dust"], s_["f"]))
    for chL in chl:
        base[chL.key] = (V.dens(chL.h, st[chL.key]["sea"]), V.dens(chL.Q, st[chL.key]["sea"]))

    def sources_with(base_, cache_):
        eps = np.zeros(a.nsites); J = np.zeros(a.nsites); eps_pairs = np.zeros(a.nsites); eps_dust = np.zeros(a.nsites); J_pairs = np.zeros(a.nsites)
        for ch in chs:
            ref = cache_[ch.kappa][3]
            hs_, Qs_, dd, Jd = base_[ch.kappa]
            ds = hs_ - V.dens(ch.h, ref); Js = Qs_ - V.dens(ch.Q, ref)
            eps += ch.w * (ds + dd); J += ch.w * (Js + Jd); eps_pairs += ch.w * ds; eps_dust += ch.w * dd; J_pairs += ch.w * Js
        if FX.light_on:
            for chL in chl:
                if chL.sourcing:
                    ref = cache_[chL.key][3]; hs_, Qs_ = base_[chL.key]
                    eps += chL.w * (hs_ - V.dens(chL.h, ref)); J += chL.w * (Qs_ - V.dens(chL.Q, ref))
        return eps, J, eps_pairs, eps_dust, J_pairs
    sources = lambda c_: sources_with(base, c_)
    N_app, v_app = FX.f_ref[0].copy(), FX.f_ref[1].copy()
    cache = {ch.kappa: (None, None, None, st[ch.kappa]["ref"], {"mode": "lagged"}) for ch in chs}
    for chL in chl:
        cache[chL.key] = (None, None, None, st[chL.key]["ref"], {"mode": "lagged"})
    srcs = sources(cache)
    Nn, vn, Wn, info = FX.do_solve(srcs)
    E_prev = V.FLD.charge(info, a.g)
    it_hist = [[None, float(np.max(np.abs(Nn - N_app))), float(np.max(np.abs(vn - v_app)))]]
    k_it, converged, best, cache, srcs, Nn, vn, Wn, info, N_app, v_app, solve32 = V.newton_record_v32(
        a, chs, st, cache, sources_with, base, None, None, FX.do_solve, Nn, vn, E_prev, it_hist, 0.0, chl=chl, light_on=FX.light_on, sgn_dev=FX.sgn_dev, S_host=FX.S_host)
    return {"k_it": k_it, "converged": converged, "cache": cache, "srcs": srcs, "Nn": Nn, "vn": vn, "N_app": N_app, "v_app": v_app, "solve32": solve32,
            "it_hist": it_hist, "sources": sources, "E": V.FLD.charge(info, a.g), "TM": dict(V.TM), "TML": dict(V.TML), "JV": dict(V.JV)}


class LoopSpy:
    """captures the F of each Newton stage (its .jvp is the exact J.v) by wrapping the engine's nk_loop_v32; the numbers are untouched."""

    def __init__(self, V):
        self.V = V; self.F = []

    def __enter__(self):
        self.orig = self.V.nk_loop_v32

        def spy(a_, F_, x0_):
            self.F.append(F_)
            return self.orig(a_, F_, x0_)
        self.V.nk_loop_v32 = spy
        return self

    def __exit__(self, *ex):
        self.V.nk_loop_v32 = self.orig


def routes(cache, chs, chl):
    out = []
    for ch in chs:
        ci = cache[ch.kappa][4]
        out.append("heavy %d %s" % (ch.kappa, "verified" if ci.get("verified") else ("LZ nsub %s" % ci.get("nsub"))))
    for chL in chl:
        ci = cache[chL.key][4]; c_ = ci.get("certified")
        out.append("light %d %s" % (chL.kappa, ("certified, %s pieces" % c_.get("pieces")) if (ci.get("verified") and isinstance(c_, dict)) else ("FALLBACK (%s)" % str(ci.get("why"))[:40])))
    return "; ".join(out)


def test_fixture(V3, V8, gpu, n=128, kappa_sea=3):
    V8.C5.backend(gpu)
    V35 = V8.V35
    V35.V36["fixes"] = True
    a_i = fixture_args(n, kappa_sea)
    FX = make_fixture(V8, a_i)
    sparse_probe_and_patch([V3, V8], FX.chs[0])
    stub = SimpleNamespace(charge=lambda info, g: float(info["E"]), __name__="fixture_fields")
    saved = (V3.FLD, V8.FLD, V35.FLD)
    V3.FLD = V8.FLD = V35.FLD = stub
    fails = 0
    try:
        print("fixture: %d sites, heavy kappa 1..%d at m %.1f, light %s sourcing and %s the guard, device %s" % (n, kappa_sea, a_i.m, a_i.light_kappas, a_i.light_guard_kappas, V8.BK["dev"]), flush=True)
        # 1. implicit bitwise with version 3
        r3 = record_solve(V3, a_i, FX)
        with LoopSpy(V8) as spyI:
            r8 = record_solve(V8, a_i, FX)
        diffs = []
        for k_ in ("Nn", "vn", "N_app", "v_app", "E", "k_it", "converged", "it_hist", "solve32"):
            deep_diff(r3[k_], r8[k_], k_, diffs)
        deep_diff(list(r3["srcs"]), list(r8["srcs"]), "sources", diffs)
        for key in r3["cache"]:
            deep_diff(r3["cache"][key], r8["cache"][key], "cache[%s]" % (key,), diffs)
        deep_diff(r3["TM"], r8["TM"], "TM", diffs); deep_diff(r3["TML"], r8["TML"], "TML", diffs); deep_diff(r3["JV"], r8["JV"], "JV", diffs)
        fails += report("1. implicit mode bitwise with version 3 (newton_record_v32 on the manufactured record)", diffs,
                        " (%d evaluations, kept residual %.2e from %s; the solve's first Newton residual %.3e; %s)" % (
                            r8["solve32"]["evaluations"], r8["solve32"]["kept_residual"], r8["solve32"]["kept_from"], max(r8["it_hist"][1][1], r8["it_hist"][1][2]), routes(r8["cache"], FX.chs, FX.chl)))
        # 2. lagged: the transport, the evaluations, the lag residual
        a_L = fixture_args(n, kappa_sea); a_L.reference_solve = "lagged"
        with LoopSpy(V8) as spyL:
            rL = record_solve(V8, a_L, FX)
        cache_L = rL["cache"]
        held_ok = all(cache_L[ch.kappa][3] is FX.st[ch.kappa]["ref"] for ch in FX.chs) and all(cache_L[chL.key][3] is FX.st[chL.key]["ref"] for chL in FX.chl)
        N_app, v_app = rL["N_app"], rL["v_app"]
        lag = V8.lagged_close_v8(a_L, FX.chs, FX.chl, FX.st, cache_L, rL["srcs"], N_app, v_app, rL["sources"], FX.do_solve, FX.sgn_dev, FX.S_host, FX.R)
        diffs = []
        if not held_ok:
            diffs.append(("held references", "identity", "an evaluation's reference was not the previous accepted record's"))
        for ch in FX.chs:
            e2, U2 = V8.eigT(V8.clocked_gpu(ch, N_app, v_app))
            ref2, ci2, m2 = V8.continue_ref_v34(a_L, ch, FX.st[ch.kappa]["window"], (N_app, v_app), e2, U2, FX.st[ch.kappa]["n_sea"])
            ci2["_mask"] = m2
            deep_diff(cache_L[ch.kappa][3], ref2, "heavy %d ref" % ch.kappa, diffs); deep_diff(cache_L[ch.kappa][4], ci2, "heavy %d cinfo" % ch.kappa, diffs)
        for chL in FX.chl:
            e2, U2 = V8.eigT(V8.clocked_gpu(chL, N_app, v_app))
            ref2, ci2, m2 = V8.light_continue_v36(a_L, chL, FX.st[chL.key]["window"], (N_app, v_app), e2, U2, FX.st[chL.key]["n_sea"], FX.sgn_dev, FX.S_host)
            ci2["_mask"] = m2
            deep_diff(cache_L[chL.key][3], ref2, "light %d ref" % chL.kappa, diffs); deep_diff(cache_L[chL.key][4], ci2, "light %d cinfo" % chL.kappa, diffs)
            deep_diff(cache_L[chL.key][2], U2, "light %d U (rotated)" % chL.kappa, diffs)
        fails += report("2a. lagged: the transported reference equals the continuation's result at the accepted fields", diffs, " (%s)" % routes(cache_L, FX.chs, FX.chl))
        ev_L, ev_I = rL["solve32"]["evaluations"], r8["solve32"]["evaluations"]
        ok = bool(rL["converged"] and ev_L <= ev_I)
        print("%s 2b. lagged: the solve meets the stop rule in %d evaluations (kept residual %.2e) against the implicit's %d (kept %.2e): at most the implicit's" % (
            "PASS" if ok else "FAIL", ev_L, rL["solve32"]["kept_residual"], ev_I, r8["solve32"]["kept_residual"]), flush=True)
        fails += 0 if ok else 1
        r0 = max(r8["it_hist"][1][1], r8["it_hist"][1][2]); dE0 = r8["it_hist"][1][0]
        rel = abs(lag["residual"] - r0) / max(r0, 1e-300)
        dEL = abs(lag["E_cons"] - rL["E"]) / abs(lag["E_cons"]); relE = abs(dEL - dE0) / max(dE0, 1e-300)
        dx_imp = max(float(np.max(np.abs(r8["N_app"] - N_app))), float(np.max(np.abs(r8["v_app"] - v_app))))
        ok = rel <= 1e-12 and relE <= 1e-9
        print("%s 2c. the lag residual %.6e equals the implicit solve's first Newton residual %.6e (relative difference %.1e); E_cons - E over E_cons %.4e against the implicit's first |dE/E| %.4e (relative %.1e); the implicit's accepted fields differ from the lagged by %.2e (the stop rule %g)" % (
            "PASS" if ok else "FAIL", lag["residual"], r0, rel, dEL, dE0, relE, dx_imp, a_L.stop_fields), flush=True)
        fails += 0 if ok else 1
        # 3. zero when the reference does not move
        cache_same = {key: (None, ent[1], ent[2], FX.st[key]["ref"], ent[4]) for key, ent in cache_L.items()}
        lag0 = V8.lag_evaluation_v8(a_L, FX.chl, FX.st, cache_same, rL["srcs"], N_app, v_app, rL["sources"], FX.do_solve, FX.R)
        zeros = (lag0["residual"], lag0["d_eps_total"]["heavy"], lag0["d_eps_total"]["light"], lag0["d_J_total"]["heavy"], lag0["d_J_total"]["light"], lag0["d_E_inv_light"], lag0["max_site_abs_d_eps_outside_cell"])
        ok = all(z == 0.0 for z in zeros) and lag0["E_cons"] == rL["E"]
        print("%s 3a. the lag residual is exactly 0 when the transported reference is the held one: %s, E_cons - E %s" % ("PASS" if ok else "FAIL", zeros, lag0["E_cons"] - rL["E"]), flush=True)
        fails += 0 if ok else 1
        try:   # busi 2026-10-18 (analyst): 3b's exception reported as its FAIL so that checks 4 to marks still run
            FXs = make_fixture(V8, a_L, stationary=True)
            rS = record_solve(V8, a_L, FXs)
            lagS = V8.lagged_close_v8(a_L, FXs.chs, FXs.chl, FXs.st, rS["cache"], rS["srcs"], rS["N_app"], rS["v_app"], rS["sources"], FXs.do_solve, FXs.sgn_dev, FXs.S_host, FXs.R)
            diffs = []; bitwise = True; resid_sub = 0.0; routes_ok = True
            for key in rS["cache"]:
                ref_n = rS["cache"][key][3]; ref_p = FXs.st[key]["ref"]; ci_ = rS["cache"][key][4]
                d_ = []; deep_diff(ref_n, ref_p, "stationary ref[%s]" % (key,), d_)
                if d_:   # red team R6: engine_ops_v3's assignment rotates a numerically degenerate block even at zero field change; the claim is the same subspace, (1 - P_prev) ref_new = 0
                    bitwise = False
                    resid_sub = max(resid_sub, float((ref_n - ref_p @ (ref_p.conj().T @ ref_n)).abs().max()))
                if ci_.get("verified") is not True:
                    routes_ok = False; diffs.append((key, "route not verified at zero field change", ci_.get("mode")))
            fields_same = float(np.max(np.abs(rS["N_app"] - FXs.f_ref[0]))) == 0.0 and float(np.max(np.abs(rS["v_app"] - FXs.f_ref[1]))) == 0.0
            zeros = (lagS["residual"], lagS["d_eps_total"]["heavy"], lagS["d_eps_total"]["light"], lagS["d_J_total"]["heavy"], lagS["d_J_total"]["light"], lagS["d_E_inv_light"], lagS["max_site_abs_d_eps_outside_cell"])
            ok = (lagS["residual"] is not None and lagS["residual"] <= 1e-14 and rS["solve32"]["kept_residual"] == 0.0 and rS["solve32"]["evaluations"] == 2 and fields_same
                  and lagS["E_cons"] is not None and abs(lagS["E_cons"] - rS["E"]) <= 1e-12 and all(abs(z) <= 1e-12 for z in zeros) and resid_sub <= 1e-12 and routes_ok)
            print("%s 3b. a stationary record (the sea its own reference at the previous fields, zero source, E %.1f): the accepted fields equal the previous %s; the solve's kept residual %.1e in %d evaluations; the lag residual %s (expected exactly 0), the lag totals %s, E_cons - E %s; the transported references %s the previous (subspace residual %.1e); %s" % (
                "PASS" if ok else "FAIL", rS["E"], fields_same, rS["solve32"]["kept_residual"], rS["solve32"]["evaluations"], lagS["residual"], zeros[1:], (lagS["E_cons"] - rS["E"]) if lagS["E_cons"] is not None else None,
                "bitwise equal to" if bitwise else "span", resid_sub, routes(rS["cache"], FXs.chs, FXs.chl) + (("; %s" % diffs[:2]) if diffs else "")), flush=True)
        except Exception:
            import traceback
            ok = False
            print("FAIL 3b. a stationary record: exception\n%s" % traceback.format_exc(), flush=True)
        fails += 0 if ok else 1
        # 4. the keys
        src = open(V8.__file__, encoding="utf-8").read()
        ok = (set(lag) == LAG_KEYS and set(lag["d_eps_total"]) == {"heavy", "light"} and set(lag["d_J_total"]) == {"heavy", "light"}
              and 'rec["reference_solve"] = "lagged"; rec["lag"] = lag_v8' in src and "reference transported, held in the solve" in src and '"--reference-solve"' in src
              and "exact in the covariance step and the field solve, the reference held" in src)
        print("%s 4. the lag block's keys %s with d_eps_total and d_J_total over {heavy, light}; the record's reference_solve and lag keys, the record label and the J.v label in the source" % (
            "PASS" if ok else "FAIL", sorted(lag)), flush=True)
        fails += 0 if ok else 1
        # 5. the exact J.v without the reference terms against a central difference of the lagged map
        F = spyL.F[0]
        x = np.concatenate([N_app, v_app]); rng = np.random.default_rng(193)
        dx = 1e-3 * rng.standard_normal(x.size); scale = float(np.max(np.abs(dx)))
        n0 = V8.TM["solve_n"]; jv = np.asarray(F.jvp(x, dx), float); n1 = V8.TM["solve_n"]
        h = 1e-3
        fd = (np.asarray(F(x + h * dx), float) - np.asarray(F(x - h * dx), float)) / (2.0 * h)
        e_jv = float(np.max(np.abs(jv + dx))) / scale; e_fd = float(np.max(np.abs(fd + dx))) / scale
        ok = e_jv == 0.0 and e_fd <= 1e-9 and n1 == n0
        print("%s 5. lagged: the exact J.v without the reference terms: |J.v + dx| / |dx| %.1e (expected 0, the held map's Jacobian -1), the central difference of the lagged map |FD + dx| / |dx| %.1e (expected roundoff), field solves entered by the J.v %d (expected 0)" % (
            "PASS" if ok else "FAIL", e_jv, e_fd, n1 - n0), flush=True)
        fails += 0 if ok else 1
        Fi = spyI.F[0]; xi = np.concatenate([r8["N_app"], r8["v_app"]])
        jvi = np.asarray(Fi.jvp(xi, dx), float)
        fdi = (np.asarray(Fi(xi + h * dx), float) - np.asarray(Fi(xi - h * dx), float)) / (2.0 * h)
        e_ref = float(np.max(np.abs(jvi + dx))) / scale; e_fdi = float(np.max(np.abs(jvi - fdi))) / scale
        ok = e_ref > 0.0
        print("%s 5b. implicit: the exact J.v carries the reference terms, |J.v + dx| / |dx| %.3e (expected above 0); against its central difference %.3e of |dx| (information: the linearization with the assignment held against the continued map)" % (
            "PASS" if ok else "FAIL", e_ref, e_fdi), flush=True)
        fails += 0 if ok else 1
    finally:
        V3.FLD, V8.FLD, V35.FLD = saved
    return fails


# ---------------------------------------------------------------------------------------------------------------------------------- marks
def test_marks(v3_path, v8_path):
    a_ = open(v3_path, encoding="utf-8").read().splitlines(); b_ = open(v8_path, encoding="utf-8").read().splitlines()
    doc_end = next(i for i, l_ in enumerate(b_) if i > 0 and l_.rstrip().endswith('"""'))
    sm = difflib.SequenceMatcher(None, a_, b_, autojunk=False)
    bad = []; hunks = 0; added = 0; changed_lines = 0; blank = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        hunks += 1; added += j2 - j1
        if j2 <= doc_end + 1:
            continue
        if tag == "delete":
            bad.append(("deleted without replacement", a_[i1:i2][:2]))
            continue
        for l_ in b_[j1:j2]:
            if not l_.strip():   # red team R6: a blank line between added functions carries no statement and no mark (G0's FAIL marks: 16 of them, 14 between the 8 new functions and 2 after the block)
                blank += 1
                continue
            changed_lines += 1
            if MARK not in l_:
                bad.append(("changed line without the mark", l_[:100]))
    if bad:
        print("FAIL marks: %d hunks, %d lines added or changed (%d blank lines skipped), %d without the mark or deleted: %s" % (hunks, added, blank, len(bad), bad[:4]), flush=True)
        return 1
    print("PASS marks: %d changed hunks, %d lines added or changed outside the docstring (%d blank lines between the added functions skipped), every one carries %r, nothing deleted without a replacement" % (hunks, changed_lines, blank, MARK), flush=True)
    return 0


# --------------------------------------------------------------------------------------------------------------------------- compare-runs
def load_run(run):
    d_, tag = os.path.split(run)
    J = json.load(open(os.path.join(d_, "collapse_%s.json" % tag)))
    npz = {nm: os.path.join(d_, "collapse_%s_%s.npz" % (tag, nm)) for nm in ("fields", "spectra", "handoff", "s11")}
    return J, {k: p for k, p in npz.items() if os.path.exists(p)}


def compare_runs(A, B, after=None):
    """2 runs' records (every number but SKIP_KEYS and the timing seconds) and every npz row at the common clocks; returns (common clocks, diffs)."""
    JA, ZA = load_run(A); JB, ZB = load_run(B)
    rs = float(JA["args"]["rs"])

    def recs(J):
        return {round(r["T"] / rs, 9): r for r in J["records"] if r.get("E") is not None and (after is None or r["T"] / rs > after + 1e-9)}
    RA, RB = recs(JA), recs(JB)
    common = sorted(set(RA) & set(RB))
    diffs = []
    if set(RA) != set(RB):
        diffs.append(("records", "clocks", (sorted(set(RA) - set(RB))[:3], sorted(set(RB) - set(RA))[:3])))
    for t in common:
        deep_diff(RA[t], RB[t], "record@%.5f" % t, diffs)
    for nm in ZA:
        if nm not in ZB:
            diffs.append((nm, "missing npz", 1.0))
            continue
        za = np.load(ZA[nm], allow_pickle=True); zb = np.load(ZB[nm], allow_pickle=True)
        TA = np.asarray(za["T"], float) / rs; TB = np.asarray(zb["T"], float) / rs
        for k in za.files:
            if k == "T" or k not in zb.files:
                continue
            x = za[k]; y = zb[k]
            if x.dtype.kind in "SU" or (x.dtype.kind == "O" and x.ndim == 0):
                continue
            if x.ndim >= 1 and x.shape[0] == len(TA) and y.ndim >= 1 and y.shape[0] == len(TB):
                for t in common:
                    ia = int(np.argmin(np.abs(TA - t))); ib = int(np.argmin(np.abs(TB - t)))
                    deep_diff(x[ia], y[ib], "%s/%s@%.5f" % (nm, k, t), diffs)
            else:
                deep_diff(x, y, "%s/%s" % (nm, k), diffs)
    return common, diffs


# ------------------------------------------------------------------------------------------------------------------------------ endtoend
def e2e_args(nsites, rs, records, tau, out, tag, gpu, extra):
    """the r_s/a 32 sigma/a 4 member's argument set (pod_launch_C_v9's COMMON) at a small site count, no slices, no Simpson record."""
    T = (records - 1) * float(tau) / float(rs)
    cmd = ["--m", "0.2", "--g", "0.25", "--rs", repr(float(rs)), "--R0-over-rs", "4", "--reading", "A", "--nsites", str(nsites), "--prep-ell", "4", "--ell", "4",
           "--kappa-sea", "10", "--tau", repr(float(tau)), "--T-over-rs", repr(T), "--reference", "R2", "--half-offset", "0", "--collar-policy", "flag",
           "--support-rel", "1e-14", "--site-fields", "averaged", "--stop-E", "1e-7", "--stop-fields", "1e-6", "--maxit", "20", "--mix", "newton", "--nk-maxiter", "30",
           "--nk-krylov", "40", "--nk-restarts", "0", "--nk-line-search", "halving", "--nk-halvings", "6", "--source-continuation", "two-stage", "--end-above", "1e-3",
           "--nk-jvp", "exact", "--delta-rtol-every", "10", "--commutator-every", "1", "--streams", "0", "--m1-simpson-steps", "", "--c-LZ", "1.0", "--window-max", "4",
           "--light-kappas", "1,2", "--light-guard-kappas", "3", "--light-coupling", "on", "--light-transport", "banded", "--light-d-min", "1e-7",
           "--s11-collars", "8,16,24", "--s11-stop", "1", "--field-kernel-cache", "--v36-fixes", "--slices-c128", "", "--slices-c64", "", "--slices-light-c128", "",
           "--out", out, "--tag", tag]
    if gpu is not None:
        cmd += ["--gpu", str(gpu)]
    return cmd + extra


def test_endtoend(v3, v8, gpu, nsites, rs, records, work):
    work = work or tempfile.mkdtemp(prefix="v8_e2e_")
    os.makedirs(work, exist_ok=True)
    env = dict(os.environ)
    for k_ in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env.setdefault(k_, "16")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    runs = [("e2e_v3", v3, []), ("e2e_v8i", v8, ["--reference-solve", "implicit"]), ("e2e_v8L", v8, ["--reference-solve", "lagged"])]
    for tag, script, extra in runs:
        cmd = [sys.executable, os.path.abspath(script)] + e2e_args(nsites, rs, records, 2.0, work, tag, gpu, extra)
        log = os.path.join(work, tag + ".log"); t0 = time.time()
        print("endtoend: running %s (%s)" % (tag, log), flush=True)
        with open(log, "w") as fh:
            rc = subprocess.call(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT, cwd=os.path.dirname(os.path.abspath(script)) or None)
        print("endtoend: %s exit %d in %.0f s" % (tag, rc, time.time() - t0), flush=True)
        if rc != 0:
            if tag == "e2e_v3":
                print("SKIP endtoend: version 3 itself exits %d at %d sites, rs %g (see %s): the stand-in fixture is missing at this size, not a failure of the build; report it" % (rc, nsites, rs, log), flush=True)
                return 0
            print("FAIL endtoend: %s exits %d where version 3 ran (see %s)" % (tag, rc, log), flush=True)
            return 1
    common, diffs = compare_runs(os.path.join(work, "e2e_v3"), os.path.join(work, "e2e_v8i"))
    fails = report("endtoend: version 8 implicit against version 3 over %d records at %d sites (the records, the fields, spectra, handoff and s11 rows)" % (len(common), nsites), diffs)
    JI = json.load(open(os.path.join(work, "collapse_e2e_v8i.json"))); JL = json.load(open(os.path.join(work, "collapse_e2e_v8L.json")))
    RI = [r for r in JI["records"] if r.get("E") is not None]; RL = [r for r in JL["records"] if r.get("E") is not None]
    ok = len(RI) == len(RL) and len(RL) > 0
    for ri, rl in zip(RI, RL):
        evi = (ri.get("solve_v32") or {}).get("evaluations"); evl = (rl.get("solve_v32") or {}).get("evaluations")
        lag = rl.get("lag") or {}
        keys_ok = (set(lag) == LAG_KEYS and rl.get("reference_solve") == "lagged" and set(lag.get("d_eps_total") or {}) == {"heavy", "light"}
                   and set(lag.get("d_J_total") or {}) == {"heavy", "light"} and "reference transported, held in the solve" in str(rl.get("record_label")))
        ok = ok and keys_ok and evi is not None and evl is not None and evl <= evi
        L = rl.get("light_v36") or {}
        print("  T/rs %.4f: evaluations lagged %s against implicit %s; lag residual %s (the stop rule 1e-6); E_cons - E %s; the solve's kept residual lagged %s; keys %s; heavy %s; light certified %s; light fallbacks %s" % (
            rl["T_over_rs"], evl, evi, lag.get("residual"), (lag.get("E_cons") - rl["E"]) if lag.get("E_cons") is not None else None, (rl.get("solve_v32") or {}).get("kept_residual"),
            "ok" if keys_ok else "WRONG", rl.get("continuation_label"), L.get("certified_on_record"), (L.get("timing") or {}).get("walk_fallbacks")), flush=True)
    r0i = max(RI[0]["iteration_history"][1][1], RI[0]["iteration_history"][1][2]) if RI and len(RI[0].get("iteration_history", [])) > 1 else None
    lag0 = (RL[0].get("lag") or {}).get("residual") if RL else None
    rel = abs(lag0 - r0i) / max(r0i, 1e-300) if (r0i is not None and lag0 is not None) else None
    ok = ok and rel is not None and rel <= 1e-12
    print("%s endtoend: lagged: the records' lag keys and label, the evaluations at most the implicit's at every record, and at T = 0 the lag residual %s against the implicit solve's first Newton residual %s (relative %s)" % (
        "PASS" if ok else "FAIL", lag0, r0i, rel), flush=True)
    return fails + (0 if ok else 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", nargs="?", default="all", choices=["all", "fixture", "marks", "endtoend", "compare-runs"])
    ap.add_argument("runs", nargs="*", help="compare-runs: A/TAG B/TAG")
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--v3", default=None); ap.add_argument("--v8", default=None)
    ap.add_argument("--code-dir", default=_here, help="where engine_twofield_v3.py and engine_twofield_v8.py are imported from")
    ap.add_argument("--nsites", type=int, default=1024); ap.add_argument("--rs", type=float, default=32.0); ap.add_argument("--records", type=int, default=4)
    ap.add_argument("--work", default=None)
    ap.add_argument("--after", type=float, default=None, help="compare-runs: compare the records after this T/rs")
    a = ap.parse_args()
    if a.code_dir and a.code_dir not in sys.path:
        sys.path.insert(0, a.code_dir)
    if a.what == "compare-runs":
        if len(a.runs) != 2:
            ap.error("compare-runs needs A/TAG B/TAG")
        common, diffs = compare_runs(a.runs[0], a.runs[1], a.after)
        rc = report("compare-runs %s against %s over %d common records%s" % (a.runs[0], a.runs[1], len(common), (" after T/rs %g" % a.after) if a.after is not None else ""), diffs)
        for d_ in diffs[:40]:
            print("  ", d_)
        sys.exit(rc)
    if a.what == "endtoend":
        if not (a.v3 and a.v8):
            ap.error("endtoend needs --v3 and --v8 (the 2 engine files)")
        sys.exit(test_endtoend(a.v3, a.v8, a.gpu, a.nsites, a.rs, a.records, a.work))
    fails = 0
    if a.what in ("all", "fixture"):
        import engine_twofield_v3 as V3
        import engine_twofield_v8 as V8
        fails += test_fixture(V3, V8, a.gpu)
    if a.what == "marks" or (a.what == "all" and a.v3 and a.v8):
        if not (a.v3 and a.v8):
            ap.error("marks needs --v3 and --v8")
        fails += test_marks(a.v3, a.v8)
    print("selftest: %d failure(s)" % fails)
    sys.exit(fails)


if __name__ == "__main__":
    main()
