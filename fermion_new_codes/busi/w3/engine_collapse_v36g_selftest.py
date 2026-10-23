#!/usr/bin/env python
"""engine_collapse_v36g_selftest.py (reviewer A for the busi lane; W3 (critical) and W4 (b) of WEB40I_TASKS/busi.md): the gates of engine_collapse_v36g,
each printing PASS or FAIL per check and exiting 0 only when every check passes.
  --build-check SRC DST              DST equals build_engine_collapse_v36g's output from SRC (md5), the reverse replacements restore SRC byte for byte,
                                     every added or changed line carries "v36g W3", DST compiles (box CPU, seconds).
  --unit [--engine-dir DIR]          the manufactured-record tests on the built engine's own functions (box CPU, under a minute, no GPU): the shell kernels'
                                     normalization and linear identity; the blend's exactness; the paired-averaging mixing's exact identities and its
                                     adjointness to roundoff (the matrix pairing and the composite pairing through the ell_out kernel on a radial grid), the
                                     composition against the wide kernel; frozen sources unchanged inside the radius, live outside, lambda projected to the
                                     target on a manufactured charge functional; the point-core sampling exact inside and outside; paired-avg through
                                     cpl_solve_v36g; --pair-sources off zeroing exactly the pair terms (bitwise the engine's line when on); the gate profile
                                     (sigma/R at a root, the gate radius against the closed form, a gated fixture); the loop's redo; the projection's
                                     target rule; the relaxed resume check.
  --gate RUN_D/tag RUN_G/tag [--slice-dirs D_STATES G_STATES]
                                     the bitwise comparison of two runs' json (every number but the wall-clock seconds; the args may differ by the v36g keys
                                     and the output paths), fields, spectra and handoff npz, and the same-named slices: v36d against v36g with its options
                                     off (the default-bitwise gate, RUN_W3_W4b_v1.md section 1.3).
  --gate-arrays RUN/tag [--T-from X --T-to Y] [--gate-value 0.1] [--engine-dir DIR]
                                     the hydrodynamic gate per stored record from the fields npz: eps_tot and J_tot re-solved to the dense N, v by
                                     engine_fields_v16.solve_constraints_ivp at the npz's g, ell and solver settings, G(r) = sigma max(|v'/N|, |v/(r N)|)
                                     with sigma = ell, and the node route from r_nodes, N_nodes, v_nodes beside it; prints per record the maximum, its
                                     radius and dominant component, the failing intervals, the gate radius, the record's E, roots and min(N - |v|), then the
                                     last gated record before the first ungated one and that record's gate radius with its rounding to the next multiple
                                     of ell (the completion's R_* when --completion-radius-over-rs is absent). "gated" means G <= the gate value at every
                                     fine-grid point from ell/16 to the collar."""
import argparse
import contextlib
import hashlib
import importlib.util
import json
import math
import os
import sys
from types import SimpleNamespace

import numpy as np

TRAPZ = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


class Out:
    def __init__(self):
        self.n = 0; self.bad = 0

    def check(self, ok, msg):
        self.n += 1
        if not ok:
            self.bad += 1
        print("%s: %s" % ("PASS" if ok else "FAIL", msg), flush=True)
        return bool(ok)

    def done(self, what):
        print("%s: %d checks, %d failed: %s" % (what, self.n, self.bad, "PASS" if self.bad == 0 else "FAIL"), flush=True)
        return 0 if self.bad == 0 else 1


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def import_engine(engine_dir):
    sys.path.insert(0, os.path.abspath(engine_dir))
    import engine_collapse_v36g as E
    return E


def reset_state(E):
    E.CPL.update({"requested": False, "mode": None, "armed": False, "redo": False, "last_gated": None, "gate_rows": [], "arming": None, "arming_own": None, "T_arm": None,
                  "R_star": None, "R_star_auto": None, "beta": None, "mix": None, "frozen_eps": None, "frozen_J": None, "E_gated": None, "T_gated": None, "lambda": 1.0,
                  "charge_mode": "matched", "charge_tol": 1e-9, "E_stored": {}, "store": {"T": [], "eps_solver": [], "J_solver": [], "lambda": [], "E_target": [], "E": []}})
    E.PSRC["on"] = True


@contextlib.contextmanager
def patched(E, fake_solve, fake_charge):
    solve0, fld0 = E.solve, E.FLD
    E.solve = fake_solve
    E.FLD = SimpleNamespace(charge=fake_charge)
    try:
        yield
    finally:
        E.solve, E.FLD = solve0, fld0


# ------------------------------------------------------------------------------------------------------------------------- build check
def build_check(src, dst):
    out = Out()
    here = os.path.dirname(os.path.abspath(__file__))
    B = load_module(os.path.join(here, "build_engine_collapse_v36g.py"), "build_engine_collapse_v36g")
    raw = open(src, "rb").read(); text = raw.decode("utf-8")
    built = B.build(text)
    disk = open(dst, "rb").read().decode("utf-8")
    out.check(hashlib.md5(built.encode("utf-8")).hexdigest() == hashlib.md5(disk.encode("utf-8")).hexdigest(), "%s equals the builder's output from %s (md5 %s)" % (dst, src, hashlib.md5(disk.encode("utf-8")).hexdigest()))
    out.check(B.reverse(built) == text, "the reverse replacements restore the source byte for byte")
    bad = B.unmarked_added_lines(text, built)
    out.check(not bad, "every added or changed line carries '%s' (%d offenders)" % (B.MARK, len(bad)))
    try:
        import py_compile
        py_compile.compile(dst, doraise=True); out.check(True, "%s compiles" % dst)
    except Exception as ex:  # noqa: BLE001
        out.check(False, "%s does not compile: %s" % (dst, ex))
    off = [ln for ln in disk.splitlines() if "accumulate_sources_v36g(ch.w" in ln]
    out.check(len(off) == 2, "both source loops call accumulate_sources_v36g (%d calls)" % len(off))
    out.check(disk.count("cpl_solve_v36g(srcs_[0], srcs_[1], a.g, R, a.ell, T, rs)") == 1, "do_solve calls cpl_solve_v36g once")
    out.check("ph = torch.exp(-1j * e.to(torch.complex128) * a.tau)" in disk and disk.count("ph = torch.exp(-1j * e.to(torch.complex128) * a.tau)") == text.count("ph = torch.exp(-1j * e.to(torch.complex128) * a.tau)"),
              "the propagation block is untouched (the same count of its exponential lines as in v36d)")
    return out.done("build-check")


# ---------------------------------------------------------------------------------------------------------------------------- unit tests
def t_kernels(E, out):
    for ell in (4.0, math.sqrt(48.0), 8.0):
        for Rn in (1.0, 3.0, 20.0, 120.0):
            r = np.linspace(1e-6, Rn + 16.0 * ell, 800001)
            K0, K1 = E.shell_kernels_v36g(r, Rn, ell)
            w = 4.0 * np.pi * r ** 2
            n0 = float(TRAPZ(w * K0, r)); n1 = float(TRAPZ(w * K1 * r, r)) / Rn
            out.check(abs(n0 - 1.0) < 1e-7, "kernel K0 normalization (ell %.3f, R %g): int 4 pi r^2 K0 dr - 1 = %.2e" % (ell, Rn, n0 - 1.0))
            out.check(abs(n1 - 1.0) < 1e-7, "kernel K1 linear identity (ell %.3f, R %g): int 4 pi r^2 K1 r dr / R - 1 = %.2e" % (ell, Rn, n1 - 1.0))
    ell = math.sqrt(48.0)
    a0, a1 = E.shell_kernels_v36g(7.0, 30.0, ell); b0, b1 = E.shell_kernels_v36g(30.0, 7.0, ell)
    out.check(abs(float(a0 - b0)) <= 1e-15 * abs(float(a0)) and abs(float(a1 - b1)) <= 1e-15 * abs(float(a1)), "kernel symmetry K(r, R) = K(R, r)")
    R0 = 2.0; r_lo = (0.1 - 1e-9) * ell ** 2 / R0; r_hi = (0.1 + 1e-9) * ell ** 2 / R0
    lo0, lo1 = E.shell_kernels_v36g(r_lo, R0, ell); hi0, hi1 = E.shell_kernels_v36g(r_hi, R0, ell)
    d0 = abs(float(hi0 - lo0)) / abs(float(lo0)); d1 = abs(float(hi1 - lo1)) / abs(float(lo1))
    out.check(d0 < 1e-7 and d1 < 1e-7, "kernel series and closed form continuous across the switch x = 0.1: dK0/K0 %.1e dK1/K1 %.1e" % (d0, d1))


def t_blend(E, out):
    R = np.arange(1, 257, dtype=float); Rs = 40.0; w = 4.0
    b = E.blend_v36g(R, Rs, w)
    out.check(bool(np.all(b[R <= Rs - w] == 1.0)) and bool(np.all(b[R >= Rs + w] == 0.0)), "blend exactly 1 for R <= R_* - ell and exactly 0 for R >= R_* + ell")
    out.check(bool(np.all(np.diff(b) <= 0.0)), "blend monotone")
    out.check(abs(float(E.blend_v36g(Rs, Rs, w))) - 0.5 < 1e-15, "blend 1/2 at the gate radius")
    bb = E.blend_v36g(np.array([Rs - w + 1e-4, Rs + w - 1e-4]), Rs, w)
    out.check(abs(1.0 - bb[0]) < 1e-11 and abs(bb[1]) < 1e-11, "blend C^2 at both ends (quintic smoothstep: deviations %.1e %.1e at 1e-4 inside the ends)" % (1.0 - bb[0], bb[1]))


def t_mixing(E, out):
    R = np.arange(1, 1025, dtype=float); Rs = 40.0; ell_out = 4.0; ell_in = 8.0
    beta = E.blend_v36g(R, Rs, ell_out)
    mx = E.mixing_v36g(R, beta, ell_in, ell_out)
    out.check(abs(mx["ell_x"] - math.sqrt(48.0)) < 1e-12, "mixing width ell_x = sqrt(8^2 - 4^2) a = %.6f" % mx["ell_x"])
    idn = E.mix_identities_v36g(mx, R)
    out.check(abs(idn["charge_sum_rel"]) < 1e-13, "mixing conserves the charge exactly (relative %.1e)" % idn["charge_sum_rel"])
    out.check(idn["K0_constant_max_abs"] < 1e-13, "B^T on N = 1 gives 1 (K0 on constants, max |dev| %.1e)" % idn["K0_constant_max_abs"])
    out.check(idn["K1_linear_max_rel"] < 1e-13, "B^T on v = c R gives c R (K1 on linear flows, relative %.1e)" % idn["K1_linear_max_rel"])
    out.check(idn["adjoint_K0_rel"] < 1e-13 and idn["adjoint_K1_rel"] < 1e-13, "adjoint pairing <B^T N, e> = <N, B e> to roundoff (%.1e, %.1e)" % (idn["adjoint_K0_rel"], idn["adjoint_K1_rel"]))
    rg = np.linspace(0.05, 200.0, 4001); wq = 4.0 * np.pi * rg ** 2 * (rg[1] - rg[0])
    K0o, K1o = E.shell_kernels_v36g(rg[:, None], R[None, :], ell_out)
    rng = np.random.RandomState(7)
    e = rng.standard_normal(len(R)); Jr = rng.standard_normal(len(R)); Nf = rng.standard_normal(len(rg)); vf = rng.standard_normal(len(rg))
    Be, BJ = E.mix_forward_v36g(mx, e, Jr)
    lhs0 = float((wq * Nf) @ (K0o @ Be)); lhs1 = float((wq * vf) @ (K1o @ BJ))
    BtN, Btv = E.mix_backward_v36g(mx, K0o.T @ (wq * Nf), K1o.T @ (wq * vf))
    rhs0 = float(BtN @ e); rhs1 = float(Btv @ Jr)
    out.check(abs(lhs0 - rhs0) <= 1e-12 * max(abs(lhs0), 1.0) and abs(lhs1 - rhs1) <= 1e-12 * max(abs(lhs1), 1.0),
              "the composite pairing <N, S_out B e>_(4 pi r^2 dr) = <B^T S_out* N, e> through the ell_out kernel on a radial grid (%.1e, %.1e)" % (abs(lhs0 - rhs0), abs(lhs1 - rhs1)))
    cc = E.composition_check_v36g(mx, R, Rs)
    out.check(cc["K0_rel_dev"] < 1e-2 and cc["K1_rel_dev"] < 1e-2, "composition: the mixed unit shell at R = %g under the ell_out kernel against the ell_in kernel: K0 %.2e, K1 %.2e (the unit-shell quadrature)" % (cc["R_site"], cc["K0_rel_dev"], cc["K1_rel_dev"]))
    out.check(cc["R_site"] <= Rs - ell_out, "the composition check's shell lies in the exact interior (beta = 1)")


def fake_charge_factory(wN):
    def charge(info, g):
        return info["E_fake"]

    def E_of(es):
        return float(wN @ es) + 1e-3 * float(np.sum(es)) ** 2 / len(es)
    return charge, E_of


def t_frozen(E, out):
    rng = np.random.RandomState(3)
    R = np.arange(1, 513, dtype=float); Rs = 40.0; ell = 4.0
    beta = E.blend_v36g(R, Rs, ell)
    eps = np.exp(-R / 60.0) * (1.0 + 0.1 * rng.standard_normal(len(R))); J = -0.3 * eps * (1.0 + 0.1 * rng.standard_normal(len(R)))
    eg = eps * (1.3 + 0.2 * rng.standard_normal(len(R))); Jg = J * (0.8 + 0.1 * rng.standard_normal(len(R)))
    wN = 1.0 + 0.1 * np.sin(R / 7.0)
    charge, E_of = fake_charge_factory(wN)

    def fake_solve(es, Js, g, R_, ell_, N0):
        return es.copy(), Js.copy(), None, {"E_fake": E_of(es), "N_point": es.copy(), "v_point": Js.copy()}
    lam_true = 0.9
    target = E_of((1.0 - beta) * eps + beta * (lam_true * eg))
    reset_state(E)
    E.CPL.update({"requested": True, "armed": True, "mode": "frozen-sources", "beta": beta, "frozen_eps": eg, "frozen_J": Jg, "E_gated": target, "T_gated": 0.0,
                  "charge_mode": "gated", "charge_tol": 1e-10, "lambda": 1.0})
    with patched(E, fake_solve, charge):
        Nn, vn, Wn, info = E.cpl_solve_v36g(eps, J, 0.25, R, ell, 752.0, 64.0)
    sv = info["v36g"]; es, Js = info["v36g_sources"]; lam = sv["lambda"]
    inside = R <= Rs - ell; outside = R >= Rs + ell
    out.check(sv["charge_converged"] and abs(sv["E"] - target) <= 1e-10 * abs(target), "frozen-sources: the charge projected to the target to 1e-10 (residual %.1e in %d solves)" % (sv["charge_residual_rel"], sv["charge_solves"]))
    out.check(abs(lam - lam_true) < 1e-6, "frozen-sources: lambda %.8f against the manufactured 0.9" % lam)
    out.check(bool(np.array_equal(es[inside], lam * eg[inside])) and bool(np.array_equal(Js[inside], lam * Jg[inside])), "frozen sources unchanged inside R_* - ell (the gated record's values times lambda, exactly)")
    out.check(bool(np.array_equal(es[outside], eps[outside])) and bool(np.array_equal(Js[outside], J[outside])), "the live sources exact outside R_* + ell")
    mid = (~inside) & (~outside)
    out.check(bool(np.all(np.abs(es[mid] - ((1.0 - beta[mid]) * eps[mid] + beta[mid] * (lam * eg[mid]))) <= 1e-15 * np.abs(es[mid]) + 1e-300)), "the blended band follows (1 - beta) eps + beta lambda eps_g")
    out.check(E.CPL["lambda"] == lam, "lambda carried to the next evaluation")
    E.CPL.update({"charge_mode": "none", "lambda": 1.0})
    with patched(E, fake_solve, charge):
        _, _, _, info2 = E.cpl_solve_v36g(eps, J, 0.25, R, ell, 752.0, 64.0)
    out.check(info2["v36g"]["lambda"] == 1.0 and info2["v36g"]["E_target"] is None and info2["v36g"]["charge_solves"] == 1, "frozen-sources with --completion-charge none: lambda = 1, one solve")
    reset_state(E)


def t_point(E, out):
    rng = np.random.RandomState(5)
    R = np.arange(1, 257, dtype=float); Rs = 40.0; ell = 4.0
    beta = E.blend_v36g(R, Rs, ell)
    Na = 1.0 - 0.3 * np.exp(-R / 30.0); va = -0.5 * np.exp(-R / 50.0)
    Np = Na * (1.0 + 0.05 * rng.standard_normal(len(R))); vp = va * (1.0 + 0.05 * rng.standard_normal(len(R)))

    def fake_solve(es, Js, g, R_, ell_, N0):
        return Na.copy(), va.copy(), None, {"E_fake": 1.0, "N_point": Np.copy(), "v_point": vp.copy(), "site_fields": "averaged"}
    reset_state(E)
    E.CPL.update({"requested": True, "armed": True, "mode": "point-core", "beta": beta})
    eps = np.exp(-R / 60.0); J = -0.3 * eps
    with patched(E, fake_solve, lambda info, g: info["E_fake"]):
        Nb, vb, _, info = E.cpl_solve_v36g(eps, J, 0.25, R, ell, 752.0, 64.0)
    inside = R <= Rs - ell; outside = R >= Rs + ell
    out.check(bool(np.array_equal(Nb[inside], Np[inside])) and bool(np.array_equal(vb[inside], vp[inside])), "point-core: the site fields are the point-sampled dense fields inside R_* - ell, exactly")
    out.check(bool(np.array_equal(Nb[outside], Na[outside])) and bool(np.array_equal(vb[outside], va[outside])), "point-core: the back-averaged fields outside R_* + ell, exactly")
    es, Js = info["v36g_sources"]
    out.check(bool(np.array_equal(es, eps)) and bool(np.array_equal(Js, J)), "point-core: the source averaging unchanged (the solver saw the raw sources)")
    out.check(info["v36g"]["point_sampled_sites"] == int(np.sum(beta >= 1.0)), "point-core: the record entry counts the point-sampled sites")
    reset_state(E)


def t_paired(E, out):
    R = np.arange(1, 513, dtype=float); Rs = 40.0; ell = 4.0
    beta = E.blend_v36g(R, Rs, ell)
    mx = E.mixing_v36g(R, beta, 8.0, ell)
    seen = {}

    def fake_solve(es, Js, g, R_, ell_, N0):
        seen["es"] = es.copy(); seen["Js"] = Js.copy()
        return 1.0 + es.copy(), Js.copy() - 1.0, None, {"E_fake": 1.0, "N_point": es.copy(), "v_point": Js.copy()}
    reset_state(E)
    E.CPL.update({"requested": True, "armed": True, "mode": "paired-avg", "beta": beta, "mix": mx})
    eps = np.exp(-R / 60.0); J = -0.3 * eps
    with patched(E, fake_solve, lambda info, g: info["E_fake"]):
        Nb, vb, _, info = E.cpl_solve_v36g(eps, J, 0.25, R, ell, 752.0, 64.0)
    es_ref, Js_ref = E.mix_forward_v36g(mx, eps, J)
    out.check(bool(np.array_equal(seen["es"], es_ref)) and bool(np.array_equal(seen["Js"], Js_ref)), "paired-avg: the solver saw B eps, B J")
    Nb_ref, vb_ref = E.mix_backward_v36g(mx, 1.0 + es_ref, Js_ref - 1.0)
    out.check(bool(np.array_equal(Nb, Nb_ref)) and bool(np.array_equal(vb, vb_ref)), "paired-avg: the returned site fields are B^T of the back-averaged ones")
    outside = R >= Rs + ell + 8.0 * mx["ell_x"]
    out.check(bool(np.all(np.abs(seen["es"][outside] - eps[outside]) <= 1e-14 * np.abs(eps[outside]) + 1e-300)), "paired-avg: the sources far outside the gate radius unchanged to roundoff")
    out.check(info["v36g"]["ell_in"] == 8.0 and info["v36g"]["ell_out"] == 4.0, "paired-avg: the record entry carries ell_in and ell_out")
    reset_state(E)


def t_pairs(E, out):
    rng = np.random.RandomState(11)
    n = 300; w = 16.0
    ds, Js, dd, Jd = (rng.standard_normal(n) for _ in range(4))
    ref = [np.zeros(n) for _ in range(5)]
    ref[0] += w * (ds + dd); ref[1] += w * (Js + Jd); ref[2] += w * ds; ref[3] += w * dd; ref[4] += w * Js
    E.PSRC["on"] = True
    on = [np.zeros(n) for _ in range(5)]
    E.accumulate_sources_v36g(w, ds, Js, dd, Jd, *on)
    out.check(all(np.array_equal(x, y) for x, y in zip(on, ref)), "pair sources on: accumulate_sources_v36g is the engine's own line bitwise (eps, J, eps_pairs, eps_dust, J_pairs)")
    E.PSRC["on"] = False
    off = [np.zeros(n) for _ in range(5)]
    E.accumulate_sources_v36g(w, ds, Js, dd, Jd, *off)
    E.PSRC["on"] = True
    out.check(np.array_equal(off[0], w * dd) and np.array_equal(off[1], w * Jd), "pair sources off: eps = w dd and J = w Jd exactly (the pair terms w ds, w Js dropped and nothing else)")
    out.check(np.array_equal(off[2], w * ds) and np.array_equal(off[3], w * dd) and np.array_equal(off[4], w * Js), "pair sources off: eps_pairs, eps_dust and J_pairs still accumulate (the reference subtraction kept for the record)")
    tol_d = 4.0 * np.finfo(float).eps * w * (np.abs(ds) + np.abs(dd)); tol_j = 4.0 * np.finfo(float).eps * w * (np.abs(Js) + np.abs(Jd))
    out.check(bool(np.all(np.abs((on[0] - off[0]) - w * ds) <= tol_d)) and bool(np.all(np.abs((on[1] - off[1]) - w * Js) <= tol_j)),
              "pair sources off: the dropped part is w ds and w Js to the rounding of the engine's own sum w (ds + dd) (within 4 ulp of w (|ds| + |dd|); the exact statements are the two checks above: fl(ds + dd) - dd is not ds bitwise, so the red team replaced the equality)")


def t_gate(E, out):
    sig = 4.0; R_root = 20.0
    rf = np.arange(0.0, 400.0 + 1e-9, sig / 16.0)
    infoA = {"r_collar": 400.0, "rfine": rf, "N_dense": lambda r: np.ones_like(np.atleast_1d(np.asarray(r, float))),
             "v_dense": lambda r: -np.sqrt(R_root / np.abs(np.atleast_1d(np.asarray(r, float))))}
    prof = E.gate_profile_v36g(infoA, sig, 0.1)
    expected = (10.0 * sig * math.sqrt(R_root)) ** (2.0 / 3.0)
    out.check(not prof["gated"], "gate fixture A (a root at %g a): not gated" % R_root)
    out.check(abs(prof["gate_radius"] - expected) <= sig / 16.0 + 1e-9, "gate radius %.3f a against the closed form (10 sigma sqrt(R_root))^(2/3) = %.3f a (within the fine grid)" % (prof["gate_radius"], expected))
    k = int(np.argmin(np.abs(prof["r"] - R_root)))
    out.check(abs(prof["G"][k] - sig / R_root) < 1e-9, "gate value at the root is sigma/R = %.4f (K_t dominant: %s)" % (sig / R_root, prof["dominant_at_max"] == "K_t"))
    out.check(len(prof["failing_intervals"]) == 1 and prof["failing_intervals"][0][0] <= sig / 16.0 + 1e-9, "one failing interval from the centre to the gate radius")
    c = 0.0125; R0 = 50.0

    def vB(r):
        r = np.abs(np.atleast_1d(np.asarray(r, float)))
        return np.where(r < R0, -c * r, -c * R0 ** 1.5 / np.sqrt(np.maximum(r, 1e-300)))
    infoB = {"r_collar": 400.0, "rfine": rf, "N_dense": infoA["N_dense"], "v_dense": vB}
    profB = E.gate_profile_v36g(infoB, sig, 0.1)
    out.check(profB["gated"] and profB["gate_radius"] == 0.0 and profB["G_max"] <= sig * c * (1.0 + 1e-6), "gate fixture B (uniform ball, sigma |K| = %.3f): gated, gate radius 0" % (sig * c))
    out.check(E._intervals_v36g(np.array([True, True, False, True]), np.array([1.0, 2.0, 3.0, 4.0])) == [[1.0, 2.0], [4.0, 4.0]], "failing intervals")


def t_steps(E, out):
    reset_state(E)
    seq = []
    for k in E.steps_v36g(3, 7):
        seq.append(k)
        if k == 5 and seq.count(5) == 1:
            E.CPL["redo"] = True
    out.check(seq == [3, 4, 5, 5, 6, 7], "steps_v36g: the engine's indices with one redo of the arming step (%s)" % seq)
    out.check(list(E.steps_v36g(0, 4)) == list(range(0, 5)), "steps_v36g without a redo is range(step0, nsteps + 1)")
    reset_state(E)


def t_target(E, out):
    reset_state(E)
    E.CPL.update({"charge_mode": "matched", "E_stored": {12.0: 100.0}, "E_gated": 99.0})
    t1, l1 = E.frozen_target_v36g(12.0 * 64.0, 64.0); t2, l2 = E.frozen_target_v36g(12.125 * 64.0, 64.0)
    out.check(t1 == 100.0 and l1.startswith("matched"), "matched target: the stored member's E at the same clock")
    out.check(t2 == 99.0 and "FALLBACK" in l2, "matched target falls back to the gated record's E with a FALLBACK label where the stored history has no record")
    E.CPL["charge_mode"] = "gated"
    out.check(E.frozen_target_v36g(800.0, 64.0)[0] == 99.0, "gated target")
    E.CPL["charge_mode"] = "none"
    out.check(E.frozen_target_v36g(800.0, 64.0)[0] is None, "none: no target")
    reset_state(E)


def t_relaxed(E, out):
    a = SimpleNamespace(resume_args_check="strict", massless_ball=False, c_LZ=1.0, mix="plain", fields_v17=False, resume_slice="s.npz")
    out.check(not E.resume_args_relaxed_v36g(a, {}, "massless_ball"), "strict: a missing key is refused")
    a.resume_args_check = "relaxed"
    out.check(E.resume_args_relaxed_v36g(a, {}, "massless_ball") and E.resume_args_relaxed_v36g(a, {}, "c_LZ"), "relaxed: a missing key at its default is accepted")
    a.c_LZ = 2.0
    out.check(not E.resume_args_relaxed_v36g(a, {}, "c_LZ"), "relaxed: a missing key with a non-default value is refused")
    out.check(not E.resume_args_relaxed_v36g(a, {"mix": "newton"}, "mix"), "relaxed: a present differing key is refused")

    class Z:
        files = ["sea_1", "dust_1"]

        def __getitem__(self, k):
            return np.zeros(2, np.complex128)
    out.check(E.slice_dtype_v36g(a, Z()) == "complex128", "relaxed: a slice without a dtype entry read by its arrays' dtype")
    a.resume_args_check = "strict"
    try:
        E.slice_dtype_v36g(a, Z()); out.check(False, "strict: a slice without a dtype entry stops the run")
    except SystemExit:
        out.check(True, "strict: a slice without a dtype entry stops the run")


def t_words(E, out):
    reset_state(E)
    for mode in ("frozen-sources", "paired-avg", "point-core"):
        a = SimpleNamespace(completion=mode, completion_gate_value=0.1, ell=4, completion_from_T_over_rs=11.625, completion_radius_over_rs=None, completion_charge="matched",
                            completion_charge_tol=1e-9, completion_ell_in=8.0, completion_run=None, resume_run="d/t")
        E.CPL.update({"requested": True, "mode": mode})
        d = E.completion_desc_v36g(a); h = E.v36g_head(a)
        out.check(mode in d and h["completion"]["mode"] == mode and h["completion"]["armed"] is False, "head and description name the completion %s" % mode)
    E.PSRC["on"] = False
    E.CPL.update({"requested": False, "mode": None})
    ent = E.record_entry_v36g(SimpleNamespace(completion_gate_value=0.1, ell=4, completion_ell_in=8.0, completion_charge="matched", g=0.25), {}, 0.0, 64.0, np.array([0.5, -0.2]), np.array([0.1, 0.1]))
    out.check(ent["pair_sources"]["mode"] == "off" and abs(ent["pair_sources"]["dropped_eps_pairs_sum"] - 0.3) < 1e-15 and "OFF" in E.record_line_v36g(ent), "record entry and line with the pair sources off")
    reset_state(E)


def unit(engine_dir):
    E = import_engine(engine_dir)
    out = Out()
    for t in (t_kernels, t_blend, t_mixing, t_frozen, t_point, t_paired, t_pairs, t_gate, t_steps, t_target, t_relaxed, t_words):
        try:
            t(E, out)
        except Exception as ex:  # noqa: BLE001
            out.check(False, "%s raised %s: %s" % (t.__name__, type(ex).__name__, str(ex)[:200]))
    return out.done("unit")


# -------------------------------------------------------------------------------------------------------------------------- bitwise gate
ALLOWED_EXTRA_ARGS = {"completion", "completion_from_T_over_rs", "completion_radius_over_rs", "completion_gate_value", "completion_ell_in", "completion_charge",
                      "completion_run", "completion_charge_tol", "pair_sources", "resume_args_check"}
PATH_ARGS = {"out", "slice_dir", "resume_slice", "resume_run", "completion_run", "resume_from", "resume_grandparent"}
TIMING_KEYS = {"seconds", "seconds_preparation", "handoff_s", "oracle34_s", "propagation_prev_s"}


def cmp_json(a, b, path, diffs):
    if isinstance(a, dict) and isinstance(b, dict):
        ka, kb = set(a), set(b)
        if path == "args":
            kb = kb - ALLOWED_EXTRA_ARGS
        if ka != kb:
            diffs.append("%s: keys differ (only in the first: %s; only in the second: %s)" % (path, sorted(ka - kb), sorted(kb - ka)))
        for k in sorted(ka & kb):
            if k in TIMING_KEYS or (k.endswith("_s") and isinstance(a[k], (int, float)) and not isinstance(a[k], bool)):   # red team (W3 review): every wall-clock entry, the kernel cache's compute_s among them (V36["kcache"] is the cache's process statistics; its hits and misses are equal between two fresh runs of the same arguments and stay compared)
                continue
            if path == "args" and k in PATH_ARGS:
                continue
            if k == "timing" and isinstance(a[k], dict) and isinstance(b[k], dict):
                for kk in sorted(set(a[k]) & set(b[k])):
                    if not kk.endswith("_s"):
                        cmp_json(a[k][kk], b[k][kk], "%s.timing.%s" % (path, kk), diffs)
                continue
            if k in ("slice_saved", "fields_file", "audit"):
                x = os.path.basename(str(a[k])) if a[k] is not None else None; y = os.path.basename(str(b[k])) if b[k] is not None else None
                if x != y:
                    diffs.append("%s.%s: %r != %r (basenames)" % (path, k, x, y))
                continue
            cmp_json(a[k], b[k], "%s.%s" % (path, k), diffs)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            diffs.append("%s: lengths %d and %d" % (path, len(a), len(b))); return
        for i, (x, y) in enumerate(zip(a, b)):
            cmp_json(x, y, "%s[%d]" % (path, i), diffs)
    else:
        if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
            return
        if a != b or type(a) != type(b):
            diffs.append("%s: %r != %r" % (path, a, b))


def cmp_npz(pd, pg, label, diffs):
    ed, eg = os.path.exists(pd), os.path.exists(pg)
    if not ed and not eg:
        return
    if ed != eg:
        diffs.append("%s: present in one run only" % label); return
    Zd = np.load(pd, allow_pickle=True); Zg = np.load(pg, allow_pickle=True)
    kd, kg = set(Zd.files), set(Zg.files)
    if kd != kg:
        diffs.append("%s: keys differ: %s" % (label, sorted(kd ^ kg)))
    for k in sorted(kd & kg):
        ad, ag = Zd[k], Zg[k]
        if ad.dtype == object or ag.dtype == object:
            ok = ad.shape == ag.shape and all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(ad.ravel(), ag.ravel()))
        elif ad.dtype.kind in "US" or ag.dtype.kind in "US":
            ok = str(ad) == str(ag)
        else:
            ok = ad.shape == ag.shape and ad.dtype == ag.dtype and bool(np.array_equal(ad, ag, equal_nan=True))
        if not ok:
            diffs.append("%s[%s]: differs (shapes %s %s, dtypes %s %s)" % (label, k, ad.shape, ag.shape, ad.dtype, ag.dtype))


def gate(run_d, run_g, slice_dirs=None):
    out = Out()
    dd, td = os.path.split(run_d); dg, tg = os.path.split(run_g)
    JD = json.load(open(os.path.join(dd, "collapse_%s.json" % td))); JG = json.load(open(os.path.join(dg, "collapse_%s.json" % tg)))
    diffs = []
    cmp_json(JD.get("args", {}), JG.get("args", {}), "args", diffs)
    extra = set(JG.get("args", {})) - set(JD.get("args", {}))
    out.check(extra <= ALLOWED_EXTRA_ARGS, "the args differ only by the v36g keys (extra: %s)" % sorted(extra))
    cmp_json(JD.get("head", {}), JG.get("head", {}), "head", diffs)
    cmp_json(JD.get("records", []), JG.get("records", []), "records", diffs)
    for k in set(JD) | set(JG):
        if k in ("args", "head", "records"):
            continue
        cmp_json(JD.get(k), JG.get(k), k, diffs)
    out.check(not diffs, "json: every number but the wall-clock seconds identical (%d records; %d differences%s)" % (len(JD.get("records", [])), len(diffs), ("; the first: " + "; ".join(diffs[:5])) if diffs else ""))
    for suffix in ("fields", "spectra", "handoff", "s11"):
        d2 = []
        cmp_npz(os.path.join(dd, "collapse_%s_%s.npz" % (td, suffix)), os.path.join(dg, "collapse_%s_%s.npz" % (tg, suffix)), suffix, d2)
        out.check(not d2, "%s npz identical in every array%s" % (suffix, ("; " + "; ".join(d2[:5])) if d2 else ""))
    out.check(not os.path.exists(os.path.join(dg, "collapse_%s_v36g.npz" % tg)), "no v36g npz written with the options off")
    if slice_dirs:
        sd, sg = slice_dirs
        names = sorted(set(f for f in os.listdir(sd) if f.endswith(".npz")) | set(f for f in os.listdir(sg) if f.endswith(".npz")))
        d3 = []
        for f in names:
            cmp_npz(os.path.join(sd, f), os.path.join(sg, f), "slice %s" % f, d3)
        out.check(not d3 and names, "%d same-named slices identical%s" % (len(names), ("; " + "; ".join(d3[:5])) if d3 else ""))
    rc = out.done("gate")
    print("GATE PASS" if rc == 0 else "GATE FAIL", flush=True)
    return rc


# --------------------------------------------------------------------------------------------------------------------------- gate arrays
def gate_arrays(run, T_from, T_to, gate_value, engine_dir):
    E = import_engine(engine_dir)
    import engine_fields_v16 as FLD
    rdir, rtag = os.path.split(run)
    Z = np.load(os.path.join(rdir, "collapse_%s_fields.npz" % rtag), allow_pickle=True)
    J = json.load(open(os.path.join(rdir, "collapse_%s.json" % rtag)))
    rs = float(J["args"]["rs"]); g = float(Z["g"]); ell = float(Z["ell"]); R = np.asarray(Z["R"], float)
    meta = {}
    if "solver" in Z.files:
        try:
            meta = json.loads(str(Z["solver"]))
        except Exception:  # noqa: BLE001
            meta = {}
    cp = meta.get("collar_policy", "flag"); sr = float(meta.get("support_rel", 1e-14)); rtol = float(meta.get("rtol", 1e-12)); atol = float(meta.get("atol", 1e-16))
    recs = {round(float(r_["T"]) / rs, 6): r_ for r_ in J["records"] if r_.get("E") is not None}
    Ts = np.asarray(Z["T"], float)
    print("gate arrays of %s: the stored eps_tot, J_tot of each record re-solved to the dense fields by engine_fields_v16.solve_constraints_ivp (g %g, ell %g, collar policy %s, support_rel %g, rtol %g, atol %g); "
          "the gate G(r) = sigma max(|v'/N|, |v/(r N)|) with sigma = ell = %g a, h = ell/64, from ell/16 to the collar; gated means G <= %g everywhere; the node route uses r_nodes, N_nodes, v_nodes (spacing ell, v' by np.gradient)" % (
              run, g, ell, cp, sr, rtol, atol, ell, gate_value), flush=True)
    print("%9s %10s %9s %4s %9s %9s %12s %9s %28s %10s  %s" % ("T/rs", "G max", "r(G max)", "dom", "gate r(a)", "G node", "E", "min(N-|v|)", "zeros", "gated", "failing intervals"), flush=True)
    rows = []
    for i, T in enumerate(Ts):
        t = T / rs
        if t < T_from - 1e-9 or t > T_to + 1e-9:
            continue
        N_, v_, W_, info = FLD.solve_constraints_ivp(np.asarray(Z["eps_tot"][i], float), np.asarray(Z["J_tot"][i], float), g, R, ell, rtol=rtol, atol=atol, collar_policy=cp, support_rel=sr)
        if info.get("failed"):
            print("%9.4f  the re-solve failed: %s" % (t, info["failed"]), flush=True); continue
        prof = E.gate_profile_v36g(info, ell, gate_value)
        rn = np.asarray(Z["r_nodes"][i], float); Nn = np.asarray(Z["N_nodes"][i], float); vn = np.asarray(Z["v_nodes"][i], float)
        sel = rn > 0.0
        Gn = float(np.max(ell * np.maximum(np.abs(np.gradient(vn, rn))[sel] / np.maximum(np.abs(Nn[sel]), 1e-300), np.abs(vn[sel]) / (rn[sel] * np.maximum(np.abs(Nn[sel]), 1e-300))))) if sel.sum() > 2 else float("nan")
        rec = recs.get(round(t, 6), {})
        rows.append((t, prof))
        print("%9.4f %10.4f %9.2f %4s %9.2f %9.4f %12.6f %9.4f %28s %10s  %s" % (
            t, prof["G_max"], prof["r_at_max"], prof["dominant_at_max"], prof["gate_radius"], Gn, rec.get("E", float("nan")), rec.get("min_N_minus_absv", float("nan")) or float("nan"),
            ", ".join("%.2f" % z for z in (rec.get("trapping_zeros") or [])), "gated" if prof["gated"] else "NOT gated", ["[%.2f, %.2f]" % (x_, y_) for x_, y_ in prof["failing_intervals"]]), flush=True)
    first_un = next(((t, p) for t, p in rows if not p["gated"]), None)
    if first_un is None:
        print("every record in the window is gated: no gate radius; widen --T-to", flush=True); return 0
    before = [t for t, p in rows if p["gated"] and t < first_un[0]]
    later = [t for t, p in rows if p["gated"] and t > first_un[0]]
    Rg = first_un[1]["gate_radius"]; Rstar = math.ceil(Rg / ell - 1e-9) * ell
    print("the last gated record before the first ungated one: T/rs %s; the first ungated record: T/rs %.4f with the gate radius %.2f a (%.4f r_s), %s dominant at the maximum %.4f; "
          "R_* (auto) = the next multiple of ell = %.0f a (%.4f r_s)%s" % (
              ("%.4f" % before[-1]) if before else "none in the window (start --T-from earlier)", first_un[0], Rg, Rg / rs, first_un[1]["dominant_at_max"], first_un[1]["G_max"], Rstar, Rstar / rs,
              ("; records gated again later: %s" % ", ".join("%.4f" % t for t in later)) if later else ""), flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-check", nargs=2, metavar=("SRC", "DST"))
    ap.add_argument("--unit", action="store_true")
    ap.add_argument("--engine-dir", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--gate", nargs=2, metavar=("RUN_D", "RUN_G"))
    ap.add_argument("--slice-dirs", nargs=2, default=None)
    ap.add_argument("--gate-arrays", default=None, metavar="RUN")
    ap.add_argument("--T-from", type=float, default=0.0); ap.add_argument("--T-to", type=float, default=float("inf")); ap.add_argument("--gate-value", type=float, default=0.1)
    a = ap.parse_args()
    rc = 0
    if a.build_check:
        rc |= build_check(a.build_check[0], a.build_check[1])
    if a.unit:
        rc |= unit(a.engine_dir)
    if a.gate:
        rc |= gate(a.gate[0], a.gate[1], a.slice_dirs)
    if a.gate_arrays:
        rc |= gate_arrays(a.gate_arrays, a.T_from, a.T_to, a.gate_value, a.engine_dir)
    if not (a.build_check or a.unit or a.gate or a.gate_arrays):
        ap.print_help(); rc = 1
    sys.exit(rc)


if __name__ == "__main__":
    main()
