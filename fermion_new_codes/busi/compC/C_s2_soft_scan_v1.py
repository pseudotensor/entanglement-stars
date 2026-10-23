#!/usr/bin/env python
"""C_s2_soft_scan_v1.py (the busi lane, Computation C): the soft-direction scan of C_s2_fold_reading_v1.md, section 1, at the sigma/a 2 member's T = 0
floor iterates of s = 0 (the uncoupled map) and s = 1 (the coupled map), with engine_twofield_v6's own functions; none of them is edited.

What it does, per s.
(1) The floor iterate x: the kept applied fields of the engine's own T = 0 solve under gate_v6_s2_v1.py's G3 arguments (the record json's
    records[0].solve_v32.kept_residual and the fields npz's N_applied[0], v_applied[0]). Taken from the gate's runs under --gate-dir when present:
    g2_s2_v6_off_ramp for s = 0 (the coupling off: its direct stage is the ramp's base stage bitwise, both kept 1.88054e-04, N at site 0 0.7696011)
    and g3_s2_v6_on_ramp for s = 1 (kept from the closing stage, 2.64066e-04, N at site 0 0.7753630); else the engine is run again here, in a
    subprocess, with G3's arguments at --T-over-rs 0 (floor/). The iterate is reproduced in process: G_s(x) - x by the engine's own F must give the
    logged kept residual within --floor-rel-tol (1e-6; bitwise is expected), or the scan stops.
(2) v: soft_gain_v6 at x with the stage's exact J.v. Its products are recorded; its Arnoldi arithmetic is replayed on the recorded products (the same
    operations in the same order: its V and H bitwise), which hands back the Ritz vector it does not return. The replay is checked against the recorded
    inputs (they are its basis vectors) and its gain against the returned one. v is unit in the 2-norm with the N(0) component positive; one more
    product gives the Rayleigh quotient and the eigen-residual of v directly.
(3) The line: x + alpha v with alpha = dN0 / v[0], dN0 the uncorrected displacement of N at site 0, on --dN0-grid both sides (14 alphas) and 0,
    marched outward from the corrected floor (each point starts from the previous corrected point shifted along v). At each point --proj-steps (2)
    projected Newton steps correct the complement of v: GMRES on P (G' - I) P dx = -P r with the exact J.v, dx orthogonal to v (a step halved at most
    twice when it does not lower the perpendicular residual); then r = G_s(x) - x is read as soft = (v.r) v[0] (the soft component's value at the N(0)
    entry), perp = |r - (v.r) v|_inf, full = |r|_inf, with central_cell_v5 (N_dense(0), W/2g at r = 1, 2, J_ell/v at r = 1) and E.
(4) A sign change of soft between adjacent points is polished along alpha (Illinois regula falsi, the same correction at every trial) and the
    engine's own Newton stage, nk_stage_v32 with its nk_loop_v32 and confirm_v32 under the stop rule (--stop-fields 1e-6, --stop-E 1e-7), is run from
    the polished point; a bowl's minimum is given to the same stage when its full residual is below --confirm-below.
(5) The verdict per s by the rule stated in RUN_C_s2_soft_scan_v1.md, section 2 (CROSSING / CROSSING, UNCONFIRMED / BOWL / FIXED POINT WITHOUT A
    SIGN CHANGE / UNDECIDED).

The engine objects it reaches, and how (nothing is guessed at run time):
  engine_twofield_v6.main()  with sys.argv set to G3's arguments (gate_v6_s2_v1.engine_args verbatim, --ell 2 --light-coupling on, the gate's ramp
    arguments) and engine_twofield_v6.newton_record_v32 replaced, for that call only, by capture_newton_record_v32 (the same signature): main() parses,
    sets OPT, SOLVE_TOL, the kernel cache, V36["fixes"] and the backend exactly as the engine does; run(a) builds the record setup at T = 0 (prepare_dust,
    the seas and references, light_init_v36, the base densities, its own closures sources_with, sources_with_heavy and do_solve, the lagged cache, the
    first solve) and hands all of it to newton_record_v32, where the capture keeps it and raises CapturedSetup: no Newton step, nothing written.
  engine_twofield_v6.nk_stage_v32(a, chs, st, cache0, sources_fn, do_solve, x0, it_hist, T, name, start, ctr, chl=(), light_on=False, sgn_dev=None,
    S_host=None, v4_report=False, gain_vectors=0)  called once per s with engine_twofield_v6.nk_loop_v32 replaced, for that call only, by LoopCapture,
    which keeps the stage's F (the fixed-point residual G_s(x) - x with the engine's bookkeeping) and F.jvp (the exact J.v: version 35's, the light
    channels weighted by LS["s"]) and returns ("captured ...", []): the stage returns with 0 evaluations. s = 0 uses sources_with_heavy(base, .) with
    chl=() and light_on=False (homotopy_ramp_v6's base stage, kwA); s = 1 uses sources_with(base, .) with the record's chl and light_on (its direct and
    closing stages). Called again, with the engine's own nk_loop_v32, from the polished point: the confirmation.
  engine_twofield_v6.soft_gain_v6(a, jvp, b, K, T, name)  b = {"app": (N, v)} at the floor iterate, K = --gain (24).
  engine_twofield_v6.central_cell_v5(a, info); FLD.charge(info, g); LS (the light weight, set as homotopy_ramp_v6.stage_at does and restored to 1.0);
    mem_v6(); free_device_v6(); release_stage_v6(stage); print_stage_v32(a, T, stage); stage_entry_v32(stage); v32_own_loop(a) (asserted True first).
  The do_solve handed to nk_stage_v32 is run()'s own closure wrapped to record each solve's output, so that G_s(x)'s solve (the one F makes) is read
    right after F returns; the engine's arithmetic is untouched.
Usage (pod 3): cd <internal>/code && python C_s2_soft_scan_v1.py --engine-dir <internal>/code --gate-dir <internal>/gateV6
  --out <internal>/soft_scan_v1 --gpu 0      (pod_C_s2_soft_scan_v1.sh). Resumable: the json and the state npz are written after every
  point; a restart reuses the floors, v and the points done. Outputs: soft_scan_v1.json, soft_scan_v1_table.txt, soft_scan_v1_state.npz (x_floor, v, the
  corrected x per point, the polished x), floor/ (the engine's T = 0 runs when the gate's are absent), setup/ (the capture's --out, empty)."""
import argparse
import contextlib
import json
import math
import os
import subprocess
import sys
import time

import numpy as np

sys.dont_write_bytecode = True

E = None                        # engine_twofield_v6, imported from --engine-dir in main()
SETUP = {}                      # the record setup at T = 0, taken from run() by capture_newton_record_v32
REC = {"last": None, "n": 0}    # the last field solve's output (do_solve_rec): G_s(x)'s solve, read right after the engine's F returns
T_START = time.time()

ARG_KEYS = ("m", "g", "rs", "R0_over_rs", "reading", "shell", "nsites", "ell", "prep_ell", "field_rtol", "field_atol", "kappa_sea", "tau", "reference", "half_offset",
            "collar_policy", "support_rel", "site_fields", "stop_E", "stop_fields", "maxit", "mix", "nk_maxiter", "nk_inner", "mix_alpha", "c_LZ", "window_max",
            "nk_line_search", "nk_halvings", "nk_krylov", "nk_restarts", "source_continuation", "end_above", "nk_jvp", "field_kernel_cache", "v36_fixes", "massless_ball",
            "light_kappas", "light_guard_kappas", "light_transport", "light_d_min", "light_max_pieces", "light_tol_block", "light_sigma_tol", "light_start_continuation",
            "light_coupling_ramp", "light_ramp_bisect", "light_ramp_floor_factor", "light_ramp_gain_vectors", "light_ramp_base_above", "light_start_plain_steps",
            "s11_collars", "s11_stop", "c_readouts", "c_heavy_einv")


def say(msg):
    print("[soft scan %6.0f s] %s" % (time.time() - T_START, msg), flush=True)


def f_(x, fm="%.3e"):
    if x is None:
        return "None"
    if isinstance(x, (bool, np.bool_)):
        return str(bool(x))
    if isinstance(x, (int, float, np.integer, np.floating)):
        try:
            return fm % x
        except (TypeError, ValueError):
            return str(x)
    return str(x)


def _jd(x):
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    return str(x)


# ------------------------------------------------------------------------------------------------------- G3's arguments (gate_v6_s2_v1.py)
def g3_engine_args(o, t_over_rs=None):
    """gate_v6_s2_v1.py's engine_args(a) verbatim (its defaults --records 3 --nsites 2048 --rs 32 --tau 2, the v36 options on); --T-over-rs is 0 for the floor runs."""
    T = (o.records - 1) * o.tau / o.rs if t_over_rs is None else float(t_over_rs)
    args = ["--m", "0.2", "--g", "0.25", "--rs", "%g" % o.rs, "--R0-over-rs", "4", "--reading", "A", "--nsites", str(o.nsites), "--prep-ell", "4", "--kappa-sea", "10",
            "--tau", "%g" % o.tau, "--T-over-rs", "%.12g" % T, "--reference", "R2", "--half-offset", "0", "--collar-policy", "flag", "--support-rel", "1e-14",
            "--site-fields", "averaged", "--stop-E", "1e-7", "--stop-fields", "1e-6", "--maxit", "20", "--mix", "newton", "--nk-maxiter", "30", "--nk-krylov", "40",
            "--nk-restarts", "0", "--nk-line-search", "halving", "--nk-halvings", "6", "--source-continuation", "two-stage", "--end-above", "1e-3", "--nk-jvp", "exact",
            "--delta-rtol-every", "10", "--commutator-every", "1", "--streams", "0", "--m1-simpson-steps", "", "--c-LZ", "1.0", "--window-max", "4",
            "--light-kappas", "1,2", "--light-guard-kappas", "3", "--light-transport", "banded", "--light-d-min", "1e-7", "--s11-collars", "8,16,24", "--s11-stop", "1",
            "--slices-c128", "", "--slices-light-c128", ""]
    if o.v36_opts:
        args += ["--field-kernel-cache", "--v36-fixes"]
    return args


def g3_extra_args(o, coupling):
    """G3's call: --ell 2 --light-coupling on and the gate's ramp_args (--ramp 0.25,0.5,0.75,1 --bisect 2 --gain 24 --floor 2; --base-above when given)."""
    args = ["--ell", "2", "--light-coupling", coupling, "--light-coupling-ramp", o.ramp, "--light-ramp-bisect", str(o.bisect), "--light-ramp-gain-vectors", str(o.gain),
            "--light-ramp-floor-factor", "%g" % o.floor]
    if o.base_above is not None:
        args += ["--light-ramp-base-above", "%g" % o.base_above]
    return args


# ------------------------------------------------------------------------------------------------------------- the engine in process
def import_engine(o):
    global E
    import importlib
    os.environ.setdefault("BUSI_ENGINE_DIR", o.engine_dir)
    os.environ.setdefault("BUSI_PAGE_DIR", o.engine_dir)
    d = os.path.abspath(o.engine_dir)
    if d not in sys.path:
        sys.path.insert(0, d)
    E = importlib.import_module(os.path.splitext(os.path.basename(o.engine))[0])
    for name in ("main", "run", "newton_record_v32", "nk_stage_v32", "nk_loop_v32", "soft_gain_v6", "central_cell_v5", "release_stage_v6", "free_device_v6", "mem_v6",
                 "print_stage_v32", "stage_entry_v32", "v32_own_loop", "LS", "MEM", "FLD", "BK"):
        if not hasattr(E, name):
            raise SystemExit("the engine %s has no %s" % (getattr(E, "__file__", o.engine), name))
    say("engine %s imported" % E.__file__)
    return E


class CapturedSetup(Exception):
    """raised by capture_newton_record_v32 inside run(): the record setup at T = 0 has been taken."""


def capture_newton_record_v32(a, chs, st, cache0, sources_with, base_new, base_prev, f_prev, do_solve, Nn0, vn0, E0, it_hist, T,
                              chl=(), light_on=False, sgn_dev=None, S_host=None, ramp=None, ramp_bisect=2, prev_kept=None,
                              sources_with_heavy=None, start_cont="off", plain_steps=12, **extra):
    """newton_record_v32's signature: keeps what run() passes at the first record (the setup at T = 0) and stops run() there."""
    SETUP.update({"a": a, "chs": chs, "st": st, "cache0": cache0, "sources_with": sources_with, "base": base_new, "base_prev": base_prev, "f_prev": f_prev,
                  "do_solve": do_solve, "Nn0": np.asarray(Nn0, float).copy(), "vn0": np.asarray(vn0, float).copy(), "E0": float(E0), "it_hist": it_hist, "T": T,
                  "chl": chl, "light_on": light_on, "sgn_dev": sgn_dev, "S_host": S_host, "sources_with_heavy": sources_with_heavy, "ramp": ramp, "extra": extra})
    raise CapturedSetup()


def capture_setup(o, engine_path):
    argv = [engine_path] + g3_engine_args(o) + g3_extra_args(o, "on") + ["--out", os.path.join(o.out, "setup"), "--tag", "setup"]
    if o.gpu is not None:
        argv += ["--gpu", str(o.gpu)]
    say("the record setup at T = 0: engine_twofield_v6.main() under G3's arguments with newton_record_v32 replaced by the capture: %s" % " ".join(argv[1:]))
    orig_nr, orig_argv = E.newton_record_v32, sys.argv
    E.newton_record_v32 = capture_newton_record_v32
    sys.argv = argv
    t0 = time.time()
    try:
        E.main()
    except CapturedSetup:
        pass
    finally:
        E.newton_record_v32 = orig_nr
        sys.argv = orig_argv
    if "a" not in SETUP:
        raise SystemExit("the record setup was not captured: run() returned or stopped before newton_record_v32")
    S = SETUP; a = S["a"]
    bad = []
    if float(S["T"]) != 0.0:
        bad.append("T %r" % (S["T"],))
    if S["base_prev"] is not None or S["f_prev"] is not None:
        bad.append("a previous record's base (not the first record)")
    if not S["light_on"]:
        bad.append("the coupling is off")
    if S["sources_with_heavy"] is None:
        bad.append("no sources_with_heavy")
    if int(a.nsites) != int(o.nsites):
        bad.append("nsites %d against %d" % (a.nsites, o.nsites))
    if bad:
        raise SystemExit("the captured setup is not the first record's coupled setup: %s" % "; ".join(bad))
    S["R"] = S["chs"][0].R
    say("captured in %.0f s: %d heavy channels, light channels %s (coupling %s, sourcing %s), E of the first solve %.9f, N at site 0 of version 31's start %.7f, base densities for %d keys" % (
        time.time() - t0, len(S["chs"]), [c.kappa for c in S["chl"]], a.light_coupling, [c.sourcing for c in S["chl"]], S["E0"], S["Nn0"][0], len(S["base"])))


def do_solve_rec(srcs_):
    """run()'s do_solve, its output recorded: F makes exactly 1 solve per evaluation, so REC["last"] right after F is G_s(x)'s solve."""
    out = SETUP["do_solve"](srcs_)
    REC["last"] = out; REC["n"] += 1
    return out


@contextlib.contextmanager
def light_weight(s):
    """homotopy_ramp_v6.stage_at's weight: LS["s"] = s around the stage, 1.0 restored (read by sources_with and the exact J.v; nothing reads it in the heavy-only stage)."""
    E.LS["s"] = float(s)
    try:
        yield
    finally:
        E.LS["s"] = 1.0


class LoopCapture:
    """stands in for nk_loop_v32(a, F, x0v) during one nk_stage_v32 call: keeps F and F.jvp, takes no step, returns (status, steps) as the loop does."""

    def __init__(self):
        self.F = None; self.jvp = None; self.calls = 0

    def __call__(self, a, F, x0v):
        self.calls += 1; self.F = F; self.jvp = getattr(F, "jvp", None)
        return "captured by C_s2_soft_scan_v1 (no Newton step)", []


def make_context(o, S, s, sources_fn, chl_s, light_on_s, x0):
    a = S["a"]; n = a.nsites
    if not E.v32_own_loop(a):
        raise SystemExit("v32_own_loop(a) is False: nk_stage_v32 would run scipy's loop, which the capture cannot intercept")
    if getattr(a, "nk_jvp", None) != "exact":
        raise SystemExit("--nk-jvp exact is required (the exact J.v is the stage's F.jvp)")
    cap = LoopCapture(); orig = E.nk_loop_v32; E.nk_loop_v32 = cap
    ctr = {"n": 0, "E_prev": S["E0"]}; it_hist = []
    try:
        with light_weight(s):
            out = E.nk_stage_v32(a, S["chs"], S["st"], S["cache0"], sources_fn, do_solve_rec, (x0[:n].copy(), x0[n:].copy()), it_hist, 0.0,
                                 "soft scan s = %g: the stage whose F and exact J.v the scan uses (no Newton step)" % s, "the floor iterate's applied fields", ctr,
                                 chl=chl_s, light_on=light_on_s, sgn_dev=S["sgn_dev"], S_host=S["S_host"], v4_report=False, gain_vectors=0)
    finally:
        E.nk_loop_v32 = orig
    if (cap.calls != 1 or cap.F is None or cap.jvp is None or out.get("evaluations") != 0 or out.get("best") is not None
            or not str(out.get("status", "")).startswith("captured")):
        raise SystemExit("the capture of nk_stage_v32's F and exact J.v failed: calls %d, status %r, evaluations %r" % (cap.calls, out.get("status"), out.get("evaluations")))
    say("s = %g: nk_stage_v32's F and exact J.v captured (status %r; %d evaluations; light channels in the stage %s, light_on %s)" % (
        s, out["status"], ctr["n"], [c.kappa for c in chl_s], light_on_s))
    return {"s": s, "F": cap.F, "jvp": cap.jvp, "evalG": out.get("evalG"), "ctr": ctr, "it_hist": it_hist, "stage": out, "products": 0}


def eval_F(ctx, x):
    """the engine's F at x: r = G_s(x) - x (2n), the solve (Nn, vn, Wn, info) and whether the evaluation is usable."""
    REC["last"] = None
    x = np.asarray(x, dtype=float)
    try:
        with light_weight(ctx["s"]):
            r = np.asarray(ctx["F"](x), dtype=float)
    except Exception as ex:
        say("  F raised: %s" % str(ex)[:160])
        return None, REC["last"], False
    sol = REC["last"]
    ok = bool(r.shape == x.shape and np.all(np.isfinite(r)) and sol is not None and not (sol[3] or {}).get("failed"))
    return r, sol, ok


def jvp_at(ctx, x, w):
    """the engine's exact J.v at x: (G' - I) w (the iterate's own evaluation reused from the memo when x is its last F's argument)."""
    ctx["products"] += 1
    with light_weight(ctx["s"]):
        return np.asarray(ctx["jvp"](np.asarray(x, dtype=float), np.asarray(w, dtype=float)), dtype=float)


# ------------------------------------------------------------------------------------------------------- v: soft_gain_v6 and its replay
def arnoldi_vhat(o, S, ctx, x_floor, K, s):
    a = S["a"]; n = a.nsites; m = 2 * n
    ins, outs = [], []

    def jvp_rec(x, dx):
        out = ctx["jvp"](x, dx)
        ctx["products"] += 1
        ins.append(np.array(dx, dtype=float, copy=True)); outs.append(np.array(out, dtype=float, copy=True))
        return out

    t0 = time.time()
    with light_weight(s):
        sg = E.soft_gain_v6(a, jvp_rec, {"app": (x_floor[:n].copy(), x_floor[n:].copy())}, int(K), 0.0, "soft scan s = %g, the floor iterate" % s)
    k_eng = int(sg.get("products") or 0)
    if k_eng == 0 or sg.get("gain") is None:
        return {"error": "soft_gain_v6 made no product (%s)" % sg.get("why"), "sg": sg}
    # the replay: soft_gain_v6's Arnoldi (its loop verbatim) on the recorded products; V and H bitwise its own, the Ritz vector kept
    V = np.zeros((m, K + 1)); H = np.zeros((K + 1, K)); V[0, 0] = 1.0
    k_done = 0
    for k in range(k_eng):
        if k >= len(outs) or not np.array_equal(ins[k], V[:, k]):
            return {"error": "the replay of soft_gain_v6's Arnoldi does not match its recorded inputs at product %d" % (k + 1), "sg": sg}
        w = outs[k] + V[:, k]
        for _pass in range(2):
            for j in range(k + 1):
                c_ = float(V[:, j] @ w)
                H[j, k] += c_
                w = w - c_ * V[:, j]
        H[k + 1, k] = float(np.linalg.norm(w)); k_done = k + 1
        if H[k + 1, k] <= 1e-13:
            break
        V[:, k + 1] = w / H[k + 1, k]
    if k_done != k_eng:
        return {"error": "the replay stopped after %d products, soft_gain_v6 after %d" % (k_done, k_eng), "sg": sg}
    ev, Y = np.linalg.eig(H[:k_done, :k_done])
    i_ = int(np.argmax(ev.real)); g = ev[i_]; y = Y[:, i_]
    vec = V[:, :k_done] @ y
    if abs(float(sg["gain"]) - float(np.real(g))) > 1e-12 * max(1.0, abs(float(np.real(g)))):
        return {"error": "the replayed gain %.15g differs from soft_gain_v6's %.15g" % (float(np.real(g)), float(sg["gain"])), "sg": sg}
    if abs(float(np.imag(g))) > 1e-10:
        return {"error": "the soft Ritz value is complex (imag %.2e): no real soft direction" % float(np.imag(g)), "sg": sg}
    nv = float(np.linalg.norm(vec))
    if nv == 0.0 or float(np.max(np.abs(np.imag(vec)))) > 1e-10 * nv:
        return {"error": "the soft Ritz vector is not real", "sg": sg}
    vec = np.real(vec)
    j_ = int(np.argmax(np.abs(vec))); site = int(j_ if j_ < n else j_ - n); field = "N" if j_ < n else "v"
    if sg.get("site") != site or sg.get("field") != field:
        return {"error": "the replayed dominant component (%s site %d) differs from soft_gain_v6's (%s site %s)" % (field, site, sg.get("field"), sg.get("site")), "sg": sg}
    vhat = vec / nv
    sign_src = 0 if abs(vhat[0]) >= 0.05 else j_
    if vhat[sign_src] < 0.0:
        vhat = -vhat
    if abs(float(vhat[0])) < 0.05:
        return {"error": "the soft vector's N(0) share is %.3f: the N(0) coordinate does not parametrize it" % float(vhat[0]), "sg": sg}
    # the direct check: 1 product
    Av = jvp_at(ctx, x_floor, vhat) + vhat
    g_rq = float(vhat @ Av); eres = float(np.linalg.norm(Av - g_rq * vhat))
    out = {"sg": sg, "vhat": vhat, "gain_replay": float(np.real(g)), "gain_rayleigh": g_rq, "eigen_residual_2": eres, "share_N0": float(vhat[0]),
           "share_N0_sq": float(vhat[0] ** 2), "dominant": {"field": field, "site": site, "component": float(abs(vec[j_]) / nv)},
           "v_component_share_sq": float(np.sum(vhat[n:] ** 2)), "second_largest_abs": float(np.sort(np.abs(vhat))[-2]), "seconds": time.time() - t0}
    say("s = %g: v from soft_gain_v6 (%d products): g %.9f (replay %.9f, Rayleigh %.9f, eigen-residual |Av - gv| %.2e), v's N(0) component %.4f (share of the norm^2 %.3f), "
        "the v field's share %.3f, dominant %s site %d (%s)" % (s, k_eng, float(sg["gain"]), out["gain_replay"], g_rq, eres, out["share_N0"], out["share_N0_sq"],
                                                                 out["v_component_share_sq"], field, site, f_(sg.get("component"), "%.3f")))
    return out


# ---------------------------------------------------------------------------------------------------------- the projected correction
def gmres_v1(matvec, b, m, rtol):
    """GMRES, one cycle of at most m products, 2-pass orthogonalization, the least squares on the Hessenberg at each step; (x, relative residual, products)."""
    nb = float(np.linalg.norm(b))
    if nb == 0.0:
        return np.zeros_like(b), 0.0, 0
    V = [b / nb]
    H = np.zeros((m + 1, m)); e1 = np.zeros(m + 1); e1[0] = nb
    y = np.zeros(0); rr = 1.0; k_done = 0
    for k in range(m):
        w = np.asarray(matvec(V[k]), dtype=float)
        if w.shape != b.shape or not np.all(np.isfinite(w)):
            raise RuntimeError("a non-finite product in the projected Newton solve (product %d)" % (k + 1))
        for _pass in range(2):
            for j in range(k + 1):
                c_ = float(V[j] @ w)
                H[j, k] += c_
                w = w - c_ * V[j]
        H[k + 1, k] = float(np.linalg.norm(w)); k_done = k + 1
        y = np.linalg.lstsq(H[:k + 2, :k + 1], e1[:k + 2], rcond=None)[0]
        rr = float(np.linalg.norm(H[:k + 2, :k + 1] @ y - e1[:k + 2])) / nb
        if rr <= rtol or H[k + 1, k] <= 1e-14 * nb:
            break
        V.append(w / H[k + 1, k])
    x = np.zeros_like(b)
    for j in range(k_done):
        x = x + y[j] * V[j]
    return x, rr, k_done


def correct_point(o, S, ctx, vhat, x_floor, x_start, alpha, tag, pre=None):
    """the line point x_start corrected in the complement of vhat by --proj-steps projected Newton steps; the soft, perpendicular and full residuals read."""
    a = S["a"]; n = a.nsites; v0 = float(vhat[0])
    t0 = time.time()
    p = {"tag": tag, "alpha": float(alpha), "dN0_line": float(alpha * v0), "ok": False, "note": None, "projected_steps": [], "evaluations": 0, "products": 0}
    x = np.asarray(x_start, dtype=float).copy()
    N_, v_ = x[:n], x[n:]
    if not (np.all(np.isfinite(x)) and float(np.min(N_)) > 0.0 and float(np.min(N_ - np.abs(v_))) > 0.0):
        p["note"] = "outside the gauge before any evaluation (min N %.3e, min N - |v| %.3e)" % (float(np.min(N_)), float(np.min(N_ - np.abs(v_))))
        p["seconds"] = time.time() - t0
        return p, x, None, None
    if pre is not None:
        r, sol, ok = pre[0], pre[1], True
    else:
        r, sol, ok = eval_F(ctx, x); p["evaluations"] += 1
    if not ok:
        p["note"] = "the solve failed at the line point before any correction (%s)" % (((sol[3] or {}).get("failed") if sol is not None else "no solve"),)
        p["seconds"] = time.time() - t0
        return p, x, None, None
    c = float(vhat @ r); perp = float(np.max(np.abs(r - c * vhat)))
    p.update({"soft_N0_before": c * v0, "perp_before": perp, "full_before": float(np.max(np.abs(r)))})
    prod0 = ctx["products"]
    for k in range(int(o.proj_steps)):
        if perp <= o.perp_stop:
            break
        rp = r - c * vhat
        xk = x

        def matvec(w, xk=xk):
            pw = w - float(vhat @ w) * vhat
            jw = jvp_at(ctx, xk, pw)
            return jw - float(vhat @ jw) * vhat + float(vhat @ w) * vhat

        try:
            dx, rr, np_ = gmres_v1(matvec, -rp, int(o.krylov_max), float(o.krylov_rtol))
        except Exception as ex:
            p["projected_steps"].append({"step": k + 1, "perp_before": perp, "failed": str(ex)[:160]})
            p["note"] = "the projected Newton solve failed at step %d: %s" % (k + 1, str(ex)[:80])
            break
        dx = dx - float(vhat @ dx) * vhat
        step = {"step": k + 1, "perp_before": perp, "krylov_products": int(np_), "krylov_rel_residual": float(rr), "step_inf": float(np.max(np.abs(dx))),
                "accepted_fraction": None, "trials": []}
        accepted = False
        for frac in (1.0, 0.5, 0.25):
            x_t = x + frac * dx
            r_t, sol_t, ok_t = eval_F(ctx, x_t); p["evaluations"] += 1
            if not ok_t:
                step["trials"].append([frac, None]); continue
            c_t = float(vhat @ r_t); perp_t = float(np.max(np.abs(r_t - c_t * vhat)))
            step["trials"].append([frac, perp_t])
            if perp_t < perp:
                x, r, sol, c, perp = x_t, r_t, sol_t, c_t, perp_t
                step["accepted_fraction"] = frac; accepted = True
                break
        step["perp_after"] = perp
        p["projected_steps"].append(step)
        if not accepted:
            p["note"] = "the projected Newton step %d lowered the perpendicular residual at no fraction (the point before it is kept)" % (k + 1)
            break
    p["products"] = ctx["products"] - prod0
    info = sol[3]
    p.update({"ok": True, "soft_coeff": c, "soft_N0": c * v0, "perp_inf": float(np.max(np.abs(r - c * vhat))), "full_inf": float(np.max(np.abs(r))),
              "N_app_0": float(x[0]), "dN0_corrected": float(x[0] - x_floor[0]), "alpha_check": float(vhat @ (x - x_floor)),
              "E": float(E.FLD.charge(info, a.g)), "central_cell": E.central_cell_v5(a, info), "seconds": time.time() - t0})
    m6 = E.mem_v6()
    if m6 is not None:
        p["memory_gib"] = {"allocated": m6[0], "peak": m6[1], "reserved": m6[2]}
    return p, x, r, sol


def print_point(s, p, prefix="point"):
    if not p.get("ok"):
        say("s = %g %s %s: NOT EVALUATED: %s (%.0f s)" % (s, prefix, p.get("tag"), p.get("note"), p.get("seconds") or 0.0))
        return
    cc = p.get("central_cell") or {}
    steps = ", ".join(("%.1e -> %.1e (%d prod, frac %s)" % (q["perp_before"], q.get("perp_after", float("nan")), q.get("krylov_products", 0), q.get("accepted_fraction")))
                      if "failed" not in q else ("step %d FAILED" % q["step"]) for q in p["projected_steps"])
    say("s = %g %s dN0_line %+.4g (alpha %+.5g): N_app(0) %.7f (dN0 %+.5f); soft_N0 %+.4e (coeff %+.4e) perp %.2e full %.2e%s; N_dense(0) %s W/2g(1) %s W/2g(2) %s J/v(1) %s E %.9f; "
        "%d evaluations %d products (%.0f s); perp before %.2e; steps [%s]%s" % (
            s, prefix, p["dN0_line"], p["alpha"], p["N_app_0"], p["dN0_corrected"], p["soft_N0"], p["soft_coeff"], p["perp_inf"], p["full_inf"],
            "" if p["perp_inf"] <= PERP_OK["v"] else " (perp ABOVE the valid bound %.0e)" % PERP_OK["v"], f_(cc.get("N_dense_0"), "%.7f"), f_(cc.get("W_over_2g_at_1"), "%.4e"),
            f_(cc.get("W_over_2g_at_2"), "%.4e"), f_(cc.get("J_over_v_at_1"), "%.4e"), p["E"], p["evaluations"], p["products"], p["seconds"], p["perp_before"], steps,
            (" [%s]" % p["note"]) if p.get("note") else ""))


PERP_OK = {"v": 1e-5}


# ----------------------------------------------------------------------------------------------------------- the polish and the stage
def polish_bracket(o, S, ctx, vhat, x_floor, b, xa, xb):
    """Illinois regula falsi on soft_N0(alpha) over the bracket b, every trial a corrected line point started from the nearer end's corrected x."""
    a0, f0, a1, f1 = float(b["alpha_a"]), float(b["soft_a"]), float(b["alpha_b"]), float(b["soft_b"])
    xs = {a0: xa, a1: xb}
    hist = []; best = None; side = 0; note = None
    t0 = time.time()
    for it in range(int(o.polish_max)):
        if f1 == f0:
            note = "equal soft values at the bracket's ends"; break
        at = (a0 * f1 - a1 * f0) / (f1 - f0)
        lo, hi = min(a0, a1), max(a0, a1)
        if not (lo < at < hi):
            at = 0.5 * (a0 + a1)
        near = a0 if abs(at - a0) <= abs(at - a1) else a1
        x_start = xs[near] + (at - near) * vhat
        p, xc, r, sol = correct_point(o, S, ctx, vhat, x_floor, x_start, at, "polish %d" % (it + 1))
        p["iteration"] = it + 1; hist.append(p)
        if not p["ok"]:
            note = "the polish's evaluation failed at iteration %d (%s)" % (it + 1, p.get("note")); break
        ft = float(p["soft_N0"]); xs[at] = xc
        print_point(ctx["s"], p, prefix="polish %d" % (it + 1))
        if best is None or abs(ft) < abs(best[0]["soft_N0"]):
            best = (p, xc)
        if abs(ft) <= o.polish_tol:
            note = "|soft| below the polish tolerance %g" % o.polish_tol; break
        if (ft > 0.0) != (f0 > 0.0):
            a1, f1 = at, ft
            if side == -1:
                f0 *= 0.5
            side = -1
        else:
            a0, f0 = at, ft
            if side == 1:
                f1 *= 0.5
            side = 1
        if abs(a1 - a0) <= 1e-10 * max(1.0, abs(a0), abs(a1)):
            note = "the bracket closed to %.1e" % abs(a1 - a0); break
    out = {"done": True, "bracket": b, "iterations": hist, "best": (best[0] if best is not None else None), "note": note, "final_bracket": [min(a0, a1), max(a0, a1)],
           "seconds": time.time() - t0}
    return out, (best[1] if best is not None else None)


def confirm_stage(o, S, s, sources_fn, chl_s, light_on_s, label, p, x):
    """the engine's own Newton stage (nk_stage_v32 with its loop, its confirming step and its soft gain) from the point x; its scalars kept, its arrays released."""
    a = S["a"]; n = a.nsites
    ctr = {"n": 0, "E_prev": S["E0"]}; it_hist = []
    say("s = %g: the engine's loop from %s (alpha %.6g, dN0 %+.5f, soft %+.3e, perp %.2e, full %.3e)" % (s, label, p["alpha"], p["dN0_corrected"], p["soft_N0"], p["perp_inf"], p["full_inf"]))
    with light_weight(s):
        stage = E.nk_stage_v32(a, S["chs"], S["st"], S["cache0"], sources_fn, S["do_solve"], (x[:n].copy(), x[n:].copy()), it_hist, 0.0,
                               "soft scan s = %g: the engine's loop from %s" % (s, label),
                               "the scan's %s (alpha %.6g, dN0 %+.5f, soft %+.3e, full %.3e)" % (label, p["alpha"], p["dN0_corrected"], p["soft_N0"], p["full_inf"]), ctr,
                               chl=chl_s, light_on=light_on_s, sgn_dev=S["sgn_dev"], S_host=S["S_host"], v4_report=True, gain_vectors=int(o.gain))
    E.print_stage_v32(a, 0.0, stage)
    stage.pop("evalG", None)
    entry = E.stage_entry_v32(stage)
    b = stage.get("best")
    xa = (np.concatenate([np.asarray(b["app"][0], float), np.asarray(b["app"][1], float)]) if b is not None else None)
    out = {"done": True, "from": label, "start_point": {k_: p.get(k_) for k_ in ("alpha", "dN0_corrected", "N_app_0", "soft_N0", "perp_inf", "full_inf")},
           "status": stage.get("status"), "meets_stop_rule": bool(stage.get("conv")), "kept_residual": (b["resid"] if b is not None else None),
           "kept_evaluation": (b["k_it"] if b is not None else None), "confirming_residual": stage.get("r_c"), "confirming_dE": stage.get("dE_c"),
           "evaluations": stage.get("evaluations"), "seconds": stage.get("seconds"), "soft_gain": stage.get("soft_gain_v6"), "displacement": stage.get("v4_displacement"),
           "central_cell": (E.central_cell_v5(a, b["sol"][3]) if b is not None else None), "kept_N_app_0": (float(xa[0]) if xa is not None else None),
           "kept_moved_from_start_inf": (float(np.max(np.abs(xa - x))) if xa is not None else None), "entry": entry}
    E.release_stage_v6(stage)
    E.free_device_v6()
    sg = out["soft_gain"] or {}
    say("s = %g: the engine's loop from %s: %s; kept %s (evaluation %s), confirming %s, dE %s; %s; soft gain at the kept iterate g %s; the kept iterate moved %s from the start" % (
        s, label, out["status"], f_(out["kept_residual"], "%.3e"), out["kept_evaluation"], f_(out["confirming_residual"], "%.3e"), f_(out["confirming_dE"], "%.1e"),
        "MEETS the stop rule" if out["meets_stop_rule"] else "does not meet the stop rule", f_(sg.get("gain"), "%.6f"), f_(out["kept_moved_from_start_inf"], "%.2e")))
    return out


# ------------------------------------------------------------------------------------------------------------------- the verdict
def quadratic_fit(valid):
    sel = [p for p in valid if abs(p["dN0_corrected"]) <= 0.1]
    if len(sel) < 3:
        return None
    x = np.array([p["dN0_corrected"] for p in sel]); y = np.array([p["soft_N0"] for p in sel])
    c2, c1, c0 = np.polyfit(x, y, 2)
    fit = np.polyval([c2, c1, c0], x)
    out = {"points": len(sel), "dN0_window": 0.1, "c0": float(c0), "c1_slope": float(c1), "c2_half_curvature": float(c2), "discriminant": float(c1 * c1 - 4.0 * c2 * c0),
           "rms": float(np.sqrt(np.mean((fit - y) ** 2)))}
    if c2 != 0.0:
        out["vertex_dN0"] = float(-c1 / (2.0 * c2)); out["vertex_value"] = float(c0 - c1 * c1 / (4.0 * c2))
        if out["discriminant"] >= 0.0:
            sq = math.sqrt(out["discriminant"])
            out["roots_dN0"] = sorted([float((-c1 - sq) / (2.0 * c2)), float((-c1 + sq) / (2.0 * c2))])
    return out


def verdict_s(o, S, s, P):
    a = S["a"]; stop = float(a.stop_fields)
    pts = sorted([p for p in P.get("points", []) if p.get("ok")], key=lambda q: q["alpha"])
    valid = [p for p in pts if p["perp_inf"] <= o.perp_ok]
    conf = P.get("confirm") or {}; pol = P.get("polish") or {}; best = pol.get("best")
    V = {"stop_rule_fields": stop, "perp_ok": float(o.perp_ok), "points_ok": len(pts), "points_valid": len(valid),
         "dN0_corrected_range_valid": ([valid[0]["dN0_corrected"], valid[-1]["dN0_corrected"]] if valid else None),
         "brackets": len(P.get("brackets") or []), "quadratic_fit": quadratic_fit(valid), "loop_run": bool(conf.get("done")),
         "loop_kept_residual": conf.get("kept_residual"), "loop_confirming_residual": conf.get("confirming_residual"), "loop_meets_stop_rule": bool(conf.get("meets_stop_rule")),
         "soft_gain_at_loop_kept": (conf.get("soft_gain") or {}).get("gain")}
    loop_ok = bool(conf.get("meets_stop_rule"))
    loop_txt = ("; the engine's loop from it: %s, kept %s, confirming %s, soft gain g %s" % (
        "MEETS the stop rule" if loop_ok else "does not meet the stop rule", f_(conf.get("kept_residual"), "%.2e"), f_(conf.get("confirming_residual"), "%.2e"),
        f_((conf.get("soft_gain") or {}).get("gain"), "%.6f"))) if conf.get("done") else "; the engine's loop was not run from it"
    if P.get("brackets"):
        br = P["brackets"][0]
        polished_ok = bool(best is not None and best["full_inf"] < stop and best["perp_inf"] <= o.perp_ok)
        if best is not None:
            V.update({"root_alpha": best["alpha"], "root_dN0_corrected": best["dN0_corrected"], "root_N_app_0": best["N_app_0"], "root_soft_N0": best["soft_N0"],
                      "root_perp_inf": best["perp_inf"], "root_full_inf": best["full_inf"], "root_central_cell": best.get("central_cell")})
        where = ("at alpha %.6g (N at site 0 %.7f, dN0 %+.5f from the floor)" % (best["alpha"], best["N_app_0"], best["dN0_corrected"])) if best is not None else "unpolished"
        if polished_ok or loop_ok:
            V["verdict"] = "CROSSING"
            V["text"] = ("CROSSING: the soft residual changes sign between dN0 %+.4g and %+.4g; the fixed point %s: the polished point's residual %s (soft %s, perp %s)%s" % (
                br["dN0_a"], br["dN0_b"], where, f_(best["full_inf"] if best else None, "%.2e"), f_(best["soft_N0"] if best else None, "%+.2e"),
                f_(best["perp_inf"] if best else None, "%.2e"), loop_txt))
        else:
            V["verdict"] = "CROSSING, UNCONFIRMED"
            V["text"] = ("CROSSING, UNCONFIRMED: the soft residual changes sign between dN0 %+.4g and %+.4g (%s), polished to soft %s, full %s (perp %s) %s, above the stop rule %g%s" % (
                br["dN0_a"], br["dN0_b"], "both ends corrected" if br["both_valid"] else "an end NOT corrected to the valid bound",
                f_(best["soft_N0"] if best else None, "%+.2e"), f_(best["full_inf"] if best else None, "%.2e"), f_(best["perp_inf"] if best else None, "%.2e"), where, stop, loop_txt))
    elif len(valid) >= 3:
        pm = min(valid, key=lambda q: abs(q["soft_N0"])); i_ = valid.index(pm)
        interior = 0 < i_ < len(valid) - 1
        V.update({"min_alpha": pm["alpha"], "min_dN0_corrected": pm["dN0_corrected"], "min_N_app_0": pm["N_app_0"], "min_soft_N0": pm["soft_N0"], "min_perp_inf": pm["perp_inf"],
                  "min_full_inf": pm["full_inf"], "min_interior": interior, "min_central_cell": pm.get("central_cell")})
        at = "at alpha %.5g (N at site 0 %.7f, dN0 %+.5f)" % (pm["alpha"], pm["N_app_0"], pm["dN0_corrected"])
        if pm["full_inf"] < stop or loop_ok:
            V["verdict"] = "FIXED POINT WITHOUT A SIGN CHANGE"
            V["text"] = "FIXED POINT WITHOUT A SIGN CHANGE (a tangency within the stop rule): the full residual %s %s, soft %s%s" % (f_(pm["full_inf"], "%.2e"), at, f_(pm["soft_N0"], "%+.2e"), loop_txt)
        elif interior:
            V["verdict"] = "BOWL"
            V["text"] = ("BOWL: no sign change of the soft residual over dN0 %+.4g to %+.4g (%d corrected points, perp <= %g); its minimum |soft| %.3e %s, the full residual %.3e there, "
                         "above the stop rule %g%s" % (valid[0]["dN0_corrected"], valid[-1]["dN0_corrected"], len(valid), o.perp_ok, abs(pm["soft_N0"]), at, pm["full_inf"], stop, loop_txt))
        else:
            V["verdict"] = "UNDECIDED"
            V["text"] = ("UNDECIDED: the minimum |soft| %.3e lies %s, at the edge of the corrected range dN0 %+.4g to %+.4g; the line was not corrected beyond it%s" % (
                abs(pm["soft_N0"]), at, valid[0]["dN0_corrected"], valid[-1]["dN0_corrected"], loop_txt))
    else:
        V["verdict"] = "UNDECIDED"
        V["text"] = "UNDECIDED: %d corrected points (perp <= %g) of %d evaluated; the complement correction did not hold along the line" % (len(valid), o.perp_ok, len(pts))
    return V


# ------------------------------------------------------------------------------------------------------------------- the floors
def read_floor(pj, pz, want_coupling, a):
    try:
        J = json.load(open(pj)); Z = np.load(pz, allow_pickle=True)
    except Exception as ex:
        return None, "unreadable: %s" % str(ex)[:80]
    recs = [r for r in J.get("records", []) if r.get("E") is not None]
    if not recs or abs(float(recs[0]["T"])) > 1e-12:
        return None, "no T = 0 record"
    Tz = np.asarray(Z["T"], float)
    if len(Tz) == 0 or abs(float(Tz[0])) > 1e-12:
        return None, "the fields npz has no T = 0 row"
    ra = J.get("args") or {}
    if str(ra.get("light_coupling")) != want_coupling:
        return None, "the coupling is %s, not %s" % (ra.get("light_coupling"), want_coupling)
    bad = [k for k in ARG_KEYS if json.loads(json.dumps(ra.get(k), default=_jd)) != json.loads(json.dumps(getattr(a, k, None), default=_jd))]
    if bad:
        return None, "arguments differ from G3's: %s" % bad
    s32 = recs[0].get("solve_v32") or {}
    if s32.get("kept_residual") is None:
        return None, "no solve_v32.kept_residual at T = 0"
    return {"N": np.asarray(Z["N_applied"][0], float).copy(), "v": np.asarray(Z["v_applied"][0], float).copy(), "kept": float(s32["kept_residual"]), "kept_from": s32.get("kept_from"),
            "confirming": s32.get("confirming_residual"), "meets_stop_rule": s32.get("meets_stop_rule"), "E": recs[0].get("E"), "json": pj, "npz": pz,
            "soft_gain_direct": s32.get("soft_gain_direct_v6"), "homotopy_final_kept": (s32.get("homotopy_ramp_v6") or {}).get("final_kept_residual")}, None


def run_floor_engine(o, engine_path, coupling, tag, fdir):
    os.makedirs(fdir, exist_ok=True)
    cmd = [sys.executable, engine_path] + g3_engine_args(o, t_over_rs=0.0) + g3_extra_args(o, coupling) + ["--out", fdir, "--tag", tag]
    if o.gpu is not None:
        cmd += ["--gpu", str(o.gpu)]
    env = dict(os.environ)
    for k_, v_ in (("OMP_NUM_THREADS", "16"), ("MKL_NUM_THREADS", "16"), ("OPENBLAS_NUM_THREADS", "16"), ("BUSI_ENGINE_DIR", o.engine_dir), ("BUSI_PAGE_DIR", o.engine_dir)):
        env.setdefault(k_, v_)
    log = os.path.join(fdir, "%s.log" % tag)
    say("running the engine's T = 0 solve for the floor (%s): %s > %s" % (tag, " ".join(cmd), log))
    t0 = time.time()
    with open(log, "w") as fh:
        rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=o.engine_dir, env=env)
    say("  exit %d in %.0f s" % (rc, time.time() - t0))
    if rc != 0:
        try:
            lines = open(log, errors="replace").read().splitlines()[-25:]
        except OSError:
            lines = []
        for ln in lines:
            print("    | %s" % ln.rstrip(), flush=True)
        raise SystemExit("the floor run %s failed (exit %d)" % (tag, rc))


def get_floor(o, S, engine_path, s):
    want = "off" if s == 0.0 else "on"
    tag = "floor_s%g_%s" % (s, want)
    fdir = os.path.join(o.out, "floor")
    gate_tag = [q.strip() for q in o.gate_tags.split(",")][0 if s == 0.0 else 1]
    cands = []
    if o.gate_dir and not o.rerun_floors:
        cands.append((os.path.join(o.gate_dir, "collapse_%s.json" % gate_tag), os.path.join(o.gate_dir, "collapse_%s_fields.npz" % gate_tag), "the gate's run %s (%s)" % (gate_tag, o.gate_dir)))
    cands.append((os.path.join(fdir, "collapse_%s.json" % tag), os.path.join(fdir, "collapse_%s_fields.npz" % tag), "this scan's engine run %s" % tag))
    for pj, pz, desc in cands:
        if not (os.path.exists(pj) and os.path.exists(pz)):
            continue
        fl, why = read_floor(pj, pz, want, S["a"])
        if fl is not None:
            fl["source"] = desc
            return fl
        say("  %s is not usable as the s = %g floor: %s" % (desc, s, why))
    run_floor_engine(o, engine_path, want, tag, fdir)
    pj, pz, desc = cands[-1]
    fl, why = read_floor(pj, pz, want, S["a"])
    if fl is None:
        raise SystemExit("the floor run %s is not usable: %s" % (tag, why))
    fl["source"] = desc
    return fl


# ------------------------------------------------------------------------------------------------------------------- the outputs
def write_table(o, RES, path):
    L = ["# C_s2_soft_scan_v1: the soft-direction scan at the sigma/a 2 T = 0 floor iterates (C_s2_fold_reading_v1.md, section 1); the verdict rule: RUN_C_s2_soft_scan_v1.md, section 2",
         "# dN0_line: the uncorrected displacement of N at site 0 along v (alpha v[0]); alpha: the coefficient on the unit vector v; N_app(0): the corrected point's N at site 0;",
         "# soft_N0 = (v.r) v[0], the soft residual component at the N(0) entry; soft_coeff = v.r; perp = |r - (v.r) v|_inf; full = |r|_inf; valid: perp <= %g; the cell from central_cell_v5" % o.perp_ok]
    for sk, P in sorted(RES.get("per_s", {}).items(), key=lambda kv: float(kv[0])):
        fl = P.get("floor") or {}; sg = P.get("soft_gain") or {}; sv = P.get("soft_vector") or {}
        L += ["", "## s = %s: %s" % (sk, P.get("map"))]
        L.append("# floor: kept %s from %s (%s); reproduced %s (relative difference %s); N at site 0 %s; uncorrected soft_N0 %s, perp %s" % (
            f_(fl.get("kept"), "%.6e"), fl.get("kept_from"), fl.get("source"), f_(fl.get("residual_reproduced"), "%.6e"), f_(fl.get("relative_difference"), "%.1e"),
            f_(fl.get("N_app_0"), "%.7f"), f_(fl.get("soft_N0_uncorrected"), "%+.4e"), f_(fl.get("perp_inf_uncorrected"), "%.2e")))
        L.append("# soft gain g %s (1 - g %s; Arnoldi residual %s after %s products; Rayleigh %s, eigen-residual %s); v: N(0) component %s, dominant %s site %s" % (
            f_(sg.get("gain"), "%.6f"), f_(sg.get("one_minus_gain"), "%.3e"), f_(sg.get("residual"), "%.1e"), sg.get("products"), f_(sv.get("gain_rayleigh"), "%.6f"),
            f_(sv.get("eigen_residual_2"), "%.1e"), f_(sv.get("share_N0"), "%.4f"), sg.get("field"), sg.get("site")))
        L.append("# %9s %10s %10s %11s %11s %9s %9s %5s | %10s %11s %11s %11s %13s | %3s %4s %5s  %s" % (
            "dN0_line", "alpha", "N_app(0)", "soft_N0", "soft_coeff", "perp", "full", "valid", "N_dense(0)", "W/2g(1)", "W/2g(2)", "J/v(1)", "E", "ev", "prod", "s", "note"))
        rows = sorted(P.get("points", []), key=lambda q: q["alpha"])
        for p in rows + [dict(q, tag="polish %s" % q.get("iteration")) for q in (P.get("polish") or {}).get("iterations", [])]:
            if not p.get("ok"):
                L.append("  %9.4g %10.5g  NOT EVALUATED: %s" % (p["dN0_line"], p["alpha"], p.get("note")))
                continue
            cc = p.get("central_cell") or {}
            L.append("  %+9.4g %+10.5g %10.7f %+11.4e %+11.4e %9.2e %9.2e %5s | %10s %11s %11s %11s %13.9f | %3d %4d %5.0f  %s" % (
                p["dN0_line"], p["alpha"], p["N_app_0"], p["soft_N0"], p["soft_coeff"], p["perp_inf"], p["full_inf"], "yes" if p["perp_inf"] <= o.perp_ok else "NO",
                f_(cc.get("N_dense_0"), "%.7f"), f_(cc.get("W_over_2g_at_1"), "%.4e"), f_(cc.get("W_over_2g_at_2"), "%.4e"), f_(cc.get("J_over_v_at_1"), "%.4e"), p["E"],
                p["evaluations"], p["products"], p["seconds"], (p["tag"] if str(p.get("tag", "")).startswith("polish") else "") + ((" [%s]" % p["note"]) if p.get("note") else "")))
        for side, e_ in sorted((P.get("side_end") or {}).items()):
            L.append("# the %s side ended at dN0_line %+.4g: %s" % (side, e_.get("alpha", 0.0) * float(sv.get("share_N0") or 1.0), e_.get("why")))
        for b in P.get("brackets") or []:
            L.append("# sign change between dN0 %+.4g (soft %+.3e) and %+.4g (soft %+.3e): both ends valid %s, clear of their perp %s" % (b["dN0_a"], b["soft_a"], b["dN0_b"], b["soft_b"], b["both_valid"], b["clear"]))
        c_ = P.get("confirm") or {}
        if c_.get("done"):
            L.append("# the engine's loop from %s: %s; kept %s, confirming %s, dE %s; %s; soft gain g %s" % (
                c_.get("from"), c_.get("status"), f_(c_.get("kept_residual"), "%.3e"), f_(c_.get("confirming_residual"), "%.3e"), f_(c_.get("confirming_dE"), "%.1e"),
                "MEETS the stop rule" if c_.get("meets_stop_rule") else "does not meet the stop rule", f_((c_.get("soft_gain") or {}).get("gain"), "%.6f")))
        V = P.get("verdict") or {}
        if V:
            qf = V.get("quadratic_fit") or {}
            L.append("# quadratic fit of soft_N0 over |dN0| <= 0.1 (%s points): slope %s, half-curvature %s, discriminant %s, vertex at dN0 %s value %s, rms %s" % (
                qf.get("points"), f_(qf.get("c1_slope"), "%.3e"), f_(qf.get("c2_half_curvature"), "%.3e"), f_(qf.get("discriminant"), "%.3e"), f_(qf.get("vertex_dN0"), "%.4e"),
                f_(qf.get("vertex_value"), "%.3e"), f_(qf.get("rms"), "%.1e")))
            L.append("# VERDICT s = %s: %s" % (sk, V.get("text")))
    mem = RES.get("memory") or {}
    L += ["", "# peak device memory %s GiB; %s" % (f_(mem.get("peak_gib"), "%.2f"), RES.get("written"))]
    with open(path, "w") as fh:
        fh.write("\n".join(L) + "\n")


def save_all(o, RES, STATE, paths):
    RES["memory"] = {"peak_gib": (float(E.MEM["peak_gib"]) if E is not None else None)}
    RES["written"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    tmp = paths["json"] + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(RES, fh, default=_jd, indent=1)
    os.replace(tmp, paths["json"])
    np.savez(paths["state"], **STATE)
    write_table(o, RES, paths["table"])


# ------------------------------------------------------------------------------------------------------------------- one s
def scan_one_s(o, S, s, floor, RES, STATE, paths, grid):
    a = S["a"]; n = a.nsites; sk = "%g" % s
    P = RES["per_s"].setdefault(sk, {"s": s})
    heavy_only = (s == 0.0)
    chl_s = () if heavy_only else S["chl"]
    light_on_s = False if heavy_only else bool(S["light_on"])
    sources_fn = (lambda c_: S["sources_with_heavy"](S["base"], c_)) if heavy_only else (lambda c_: S["sources_with"](S["base"], c_))
    P["map"] = ("the uncoupled map G_0: sources_with_heavy(base, .), the light channels not evaluated (homotopy_ramp_v6's base stage)" if heavy_only else
                "the coupled map G_1: sources_with(base, .) with the light weight LS[s] = 1 (the direct and closing stages)")
    x_floor = np.concatenate([floor["N"], floor["v"]])
    STATE["x_floor_s%s" % sk] = x_floor
    P["floor"] = {k_: floor.get(k_) for k_ in ("kept", "kept_from", "confirming", "meets_stop_rule", "E", "json", "npz", "source", "soft_gain_direct", "homotopy_final_kept")}
    P["floor"]["N_app_0"] = float(x_floor[0]); P["floor"]["v_app_0"] = float(x_floor[n])
    say("s = %g floor: %s; kept %.6e from %s; N at site 0 %.7f" % (s, floor["source"], floor["kept"], floor["kept_from"], x_floor[0]))
    ctx = make_context(o, S, s, sources_fn, chl_s, light_on_s, x_floor)
    # the floor reproduced by the engine's F
    t0 = time.time()
    r0, sol0, ok0 = eval_F(ctx, x_floor)
    if not ok0:
        raise SystemExit("s = %g: the engine's F failed at the floor iterate" % s)
    full0 = float(np.max(np.abs(r0))); logged = float(floor["kept"]); rel = abs(full0 - logged) / max(abs(logged), 1e-300)
    P["floor"].update({"residual_reproduced": full0, "relative_difference": rel, "E_reproduced": float(E.FLD.charge(sol0[3], a.g)), "central_cell": E.central_cell_v5(a, sol0[3])})
    say("s = %g: the floor iterate's residual by the engine's F %.10e against the logged kept %.10e (relative difference %.1e; %s) (%.0f s)" % (
        s, full0, logged, rel, "reproduced" if rel <= o.floor_rel_tol else "NOT REPRODUCED", time.time() - t0))
    if rel > o.floor_rel_tol:
        raise SystemExit("s = %g: the floor iterate is not reproduced to the logged residual (relative difference %.2e > %g): the state is not the gate's" % (s, rel, o.floor_rel_tol))
    # v
    key_v = "vhat_s%s" % sk
    if key_v in STATE and P.get("soft_gain") and P.get("soft_vector"):
        vhat = np.asarray(STATE[key_v], float)
        say("s = %g: v restored from the state (g %s, N(0) component %.4f)" % (s, f_(P["soft_gain"].get("gain"), "%.6f"), vhat[0]))
    else:
        arn = arnoldi_vhat(o, S, ctx, x_floor, o.gain, s)
        if arn.get("error"):
            P["soft_gain"] = arn.get("sg"); P["verdict"] = {"verdict": "UNDECIDED", "text": "UNDECIDED: no soft direction: %s" % arn["error"]}
            say("VERDICT s = %g: %s" % (s, P["verdict"]["text"]))
            save_all(o, RES, STATE, paths)
            return
        vhat = arn["vhat"]; STATE[key_v] = vhat
        P["soft_gain"] = arn["sg"]; P["soft_vector"] = {k_: arn[k_] for k_ in ("gain_replay", "gain_rayleigh", "eigen_residual_2", "share_N0", "share_N0_sq", "dominant", "v_component_share_sq", "second_largest_abs", "seconds")}
        save_all(o, RES, STATE, paths)
    v0 = float(vhat[0])
    c0 = float(vhat @ r0)
    P["floor"].update({"soft_N0_uncorrected": c0 * v0, "soft_coeff_uncorrected": c0, "perp_inf_uncorrected": float(np.max(np.abs(r0 - c0 * vhat)))})
    say("s = %g: at the floor the residual's soft component at N(0) is %+.4e of the full %.4e (perpendicular %.2e)" % (s, c0 * v0, full0, P["floor"]["perp_inf_uncorrected"]))
    # the line
    P.setdefault("points", []); P.setdefault("side_end", {})
    done = {int(p["index"]): p for p in P["points"]}

    def keep_point(p, xc):
        P["points"] = sorted([q for q in P["points"] if int(q["index"]) != int(p["index"])] + [p], key=lambda q: int(q["index"]))
        STATE["xc_s%s_%d" % (sk, int(p["index"]))] = np.asarray(xc, float)
        done[int(p["index"])] = p
        save_all(o, RES, STATE, paths)
        E.free_device_v6()

    if 0 in done and ("xc_s%s_0" % sk) in STATE:
        p0 = done[0]; x0c = np.asarray(STATE["xc_s%s_0" % sk], float)
        say("s = %g: the corrected floor (alpha 0) restored" % s)
    else:
        p0, x0c, _r, _sol = correct_point(o, S, ctx, vhat, x_floor, x_floor, 0.0, "dN0 0 (the floor)", pre=(r0, sol0))
        p0.update({"index": 0, "side": "0"})
        if not p0["ok"]:
            x0c = x_floor
        keep_point(p0, x0c); print_point(s, p0)
    for side, sign in (("+", 1.0), ("-", -1.0)):
        if side in P["side_end"]:
            continue
        x_prev, a_prev = x0c, 0.0
        for j, g_ in enumerate(grid):
            i = (1 + j) if sign > 0 else (1 + len(grid) + j)
            alpha = sign * g_ / v0
            key = "xc_s%s_%d" % (sk, i)
            if i in done and key in STATE:
                x_prev, a_prev = np.asarray(STATE[key], float), alpha
                continue
            x_start = x_prev + (alpha - a_prev) * vhat
            p, xc, _r, _sol = correct_point(o, S, ctx, vhat, x_floor, x_start, alpha, "dN0 %+g" % (sign * g_))
            p.update({"index": i, "side": side})
            ended = (not p["ok"]) or p["full_inf"] > o.full_cap
            if ended:
                P["side_end"][side] = {"index": i, "alpha": alpha, "dN0_line": sign * g_, "why": (p["note"] if not p["ok"] else "the full residual %.3e exceeds the cap %g" % (p["full_inf"], o.full_cap))}
            keep_point(p, xc); print_point(s, p)
            if ended:
                say("s = %g: the %s side of the line ends at dN0 %+g: %s" % (s, side, sign * g_, P["side_end"][side]["why"]))
                break
            x_prev, a_prev = xc, alpha
    # the sign changes
    pts = sorted([p for p in P["points"] if p.get("ok")], key=lambda q: q["alpha"])
    brackets = []
    for p, q in zip(pts, pts[1:]):
        fp, fq = float(p["soft_N0"]), float(q["soft_N0"])
        if (fp > 0.0) != (fq > 0.0) or fp == 0.0 or fq == 0.0:
            brackets.append({"alpha_a": p["alpha"], "alpha_b": q["alpha"], "dN0_a": p["dN0_corrected"], "dN0_b": q["dN0_corrected"], "soft_a": fp, "soft_b": fq,
                             "index_a": int(p["index"]), "index_b": int(q["index"]), "both_valid": bool(p["perp_inf"] <= o.perp_ok and q["perp_inf"] <= o.perp_ok),
                             "clear": bool(abs(fp) > 2.0 * p["perp_inf"] and abs(fq) > 2.0 * q["perp_inf"])})
    brackets.sort(key=lambda b: (not (b["clear"] and b["both_valid"]), min(abs(b["alpha_a"]), abs(b["alpha_b"]))))
    P["brackets"] = brackets
    say("s = %g: %d evaluated points, %d sign change(s) of the soft residual%s" % (s, len(pts), len(brackets), (": " + "; ".join(
        "dN0 %+.4g (%+.2e) to %+.4g (%+.2e)%s" % (b["dN0_a"], b["soft_a"], b["dN0_b"], b["soft_b"], "" if (b["clear"] and b["both_valid"]) else " (not clear of the perpendicular residual or not corrected)") for b in brackets)) if brackets else ""))
    # the polish
    if brackets and not o.no_polish and not (P.get("polish") or {}).get("done"):
        b = brackets[0]
        xa = np.asarray(STATE["xc_s%s_%d" % (sk, b["index_a"])], float); xb = np.asarray(STATE["xc_s%s_%d" % (sk, b["index_b"])], float)
        pol, xpol = polish_bracket(o, S, ctx, vhat, x_floor, b, xa, xb)
        P["polish"] = pol
        if xpol is not None:
            STATE["xpol_s%s" % sk] = xpol
        save_all(o, RES, STATE, paths)
        bp = pol.get("best")
        say("s = %g: the polish (%d iterations; %s): best soft %s, perp %s, full %s at alpha %s (N at site 0 %s, dN0 %s)" % (
            s, len(pol["iterations"]), pol.get("note"), f_(bp["soft_N0"] if bp else None, "%+.3e"), f_(bp["perp_inf"] if bp else None, "%.2e"), f_(bp["full_inf"] if bp else None, "%.3e"),
            f_(bp["alpha"] if bp else None, "%.6g"), f_(bp["N_app_0"] if bp else None, "%.7f"), f_(bp["dN0_corrected"] if bp else None, "%+.5f")))
    P["counts"] = {"evaluations": int(ctx["ctr"]["n"]), "products": int(ctx["products"])}
    # the captured closures released before the engine's own stage (its caches beside theirs would pass 30 GiB)
    ctx.clear(); del ctx
    E.free_device_v6()
    # the confirmation by the engine's own loop
    cand = None
    pol = P.get("polish") or {}
    if pol.get("best") is not None and ("xpol_s%s" % sk) in STATE:
        cand = ("the polished point", pol["best"], np.asarray(STATE["xpol_s%s" % sk], float))
    else:
        valid = [p for p in pts if p["perp_inf"] <= o.perp_ok]
        if valid:
            pm = min(valid, key=lambda q: abs(q["soft_N0"]))
            key = "xc_s%s_%d" % (sk, int(pm["index"]))
            if pm["full_inf"] <= o.confirm_below and key in STATE:
                cand = ("the bowl's minimum point", pm, np.asarray(STATE[key], float))
    if cand is not None and not (P.get("confirm") or {}).get("done"):
        P["confirm"] = confirm_stage(o, S, s, sources_fn, chl_s, light_on_s, cand[0], cand[1], cand[2])
        save_all(o, RES, STATE, paths)
    elif cand is None:
        P["confirm"] = {"done": False, "why": "no sign change and no corrected point with the full residual below --confirm-below %g" % o.confirm_below}
    P["verdict"] = verdict_s(o, S, s, P)
    say("VERDICT s = %g: %s" % (s, P["verdict"]["text"]))
    save_all(o, RES, STATE, paths)


# ------------------------------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="the soft-direction scan at the sigma/a 2 T = 0 floor iterates (busi lane, Computation C)")
    ap.add_argument("--engine-dir", required=True); ap.add_argument("--engine", default="engine_twofield_v6.py")
    ap.add_argument("--out", required=True); ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--gate-dir", default="<internal>/gateV6", help="gate v6's outputs: the floors from its runs when present (empty: none)")
    ap.add_argument("--gate-tags", default="g2_s2_v6_off_ramp,g3_s2_v6_on_ramp", help="the gate's run tags for s = 0 (coupling off) and s = 1 (coupling on)")
    ap.add_argument("--rerun-floors", action="store_true", help="run the engine's T = 0 solves for the floors even when the gate's runs are present")
    ap.add_argument("--records", type=int, default=3); ap.add_argument("--nsites", type=int, default=2048); ap.add_argument("--rs", type=float, default=32.0); ap.add_argument("--tau", type=float, default=2.0)
    ap.add_argument("--ramp", default="0.25,0.5,0.75,1"); ap.add_argument("--bisect", type=int, default=2); ap.add_argument("--gain", type=int, default=24)
    ap.add_argument("--floor", type=float, default=2.0); ap.add_argument("--base-above", type=float, default=None)
    ap.add_argument("--no-v36-opts", dest="v36_opts", action="store_false", help="without --field-kernel-cache --v36-fixes (the gate ran with them)")
    ap.add_argument("--s-values", default="0,1", help="0 (the uncoupled map) and/or 1 (the coupled map)")
    ap.add_argument("--dN0-grid", default="0.005,0.01,0.02,0.04,0.08,0.15,0.3", help="the uncorrected displacements of N at site 0 along v, each side of 0 (14 alphas)")
    ap.add_argument("--proj-steps", type=int, default=2, help="projected Newton steps per line point"); ap.add_argument("--krylov-max", type=int, default=40); ap.add_argument("--krylov-rtol", type=float, default=1e-4)
    ap.add_argument("--perp-stop", type=float, default=1e-8, help="no further projected step below this perpendicular residual")
    ap.add_argument("--perp-ok", type=float, default=1e-5, help="a line point counts as corrected (valid) when its perpendicular residual is at most this")
    ap.add_argument("--full-cap", type=float, default=1.0, help="a side of the line ends at a point whose full residual exceeds this")
    ap.add_argument("--polish-max", type=int, default=8); ap.add_argument("--polish-tol", type=float, default=1e-7); ap.add_argument("--confirm-below", type=float, default=1e-5)
    ap.add_argument("--no-polish", action="store_true"); ap.add_argument("--fresh", action="store_true", help="ignore a previous json/state in --out")
    ap.add_argument("--floor-rel-tol", type=float, default=1e-6)
    o = ap.parse_args()
    if o.gain <= 0:
        ap.error("--gain must be positive (the soft gain's Arnoldi products)")
    s_values = []
    for q in o.s_values.split(","):
        if q.strip():
            s_ = float(q)
            if s_ not in (0.0, 1.0):
                ap.error("--s-values: only 0 (the uncoupled map) and 1 (the coupled map) have a floor iterate to scan from")
            s_values.append(s_)
    grid = [float(q) for q in o.dN0_grid.split(",") if q.strip()]
    if not grid or any(g_ <= 0.0 for g_ in grid) or any(q_ >= p_ for q_, p_ in zip(grid, grid[1:])):
        ap.error("--dN0-grid: positive increasing displacements")
    PERP_OK["v"] = float(o.perp_ok)
    os.makedirs(o.out, exist_ok=True)
    engine_path = os.path.join(o.engine_dir, o.engine)
    if not os.path.exists(engine_path):
        raise SystemExit("no engine at %s" % engine_path)
    paths = {"json": os.path.join(o.out, "soft_scan_v1.json"), "table": os.path.join(o.out, "soft_scan_v1_table.txt"), "state": os.path.join(o.out, "soft_scan_v1_state.npz")}
    RES = None; STATE = {}
    if not o.fresh and os.path.exists(paths["json"]):
        RES = json.load(open(paths["json"]))
        if os.path.exists(paths["state"]):
            STATE = dict(np.load(paths["state"]))
        say("resuming from %s (%d point(s) done: %s)" % (paths["json"], sum(len(P.get("points", [])) for P in RES.get("per_s", {}).values()),
                                                        {k_: len(P.get("points", [])) for k_, P in RES.get("per_s", {}).items()}))
    if RES is None:
        RES = {"script": "C_s2_soft_scan_v1.py", "args": vars(o), "engine_args_g3": g3_engine_args(o) + g3_extra_args(o, "on"), "grid_dN0": grid, "per_s": {},
               "verdict_rule": "RUN_C_s2_soft_scan_v1.md, section 2"}
    say("the soft-direction scan: s %s; dN0 grid %s (both sides and 0); %d projected Newton steps per point (GMRES at most %d products, rtol %g); valid below perp %g; "
        "polish at most %d iterations to |soft| %g; the engine's loop from a polished point or from a minimum with the full residual below %g" % (
            s_values, grid, o.proj_steps, o.krylov_max, o.krylov_rtol, o.perp_ok, o.polish_max, o.polish_tol, o.confirm_below))
    import_engine(o)
    capture_setup(o, engine_path)
    S = SETUP; a = S["a"]
    E.mem_v6(reset=True)
    RES["setup"] = {"E_first_solve": S["E0"], "nsites": int(a.nsites), "light_channels": [c.kappa for c in S["chl"]], "light_sourcing": [c.sourcing for c in S["chl"]],
                    "light_on": bool(S["light_on"]), "N_start_0": float(S["Nn0"][0]), "stop_fields": a.stop_fields, "stop_E": a.stop_E, "engine_args": vars(a)}
    floors = {}
    for s in s_values:
        floors[s] = get_floor(o, S, engine_path, s)
    for s in s_values:
        scan_one_s(o, S, s, floors[s], RES, STATE, paths, grid)
    save_all(o, RES, STATE, paths)
    m6 = E.mem_v6()
    say("SOFT SCAN DONE: %s; peak device memory %.2f GiB%s; %.0f s; %s, %s" % (
        "; ".join("s = %s %s" % (k_, (P.get("verdict") or {}).get("verdict")) for k_, P in sorted(RES["per_s"].items(), key=lambda kv: float(kv[0]))), E.MEM["peak_gib"],
        (" (allocated now %.2f, reserved %.2f)" % (m6[0], m6[2])) if m6 is not None else "", time.time() - T_START, paths["json"], paths["table"]))


if __name__ == "__main__":
    main()
