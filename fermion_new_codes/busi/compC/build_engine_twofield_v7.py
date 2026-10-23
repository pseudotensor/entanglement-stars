#!/usr/bin/env python
"""build_engine_twofield_v7.py (the busi lane, Computation C; inbox #158): writes engine_twofield_v7.py from engine_twofield_v3.py by exact
string replacements. Each OLD below is copied from version 3 (input 02); the only tolerance in the match is the run of spaces the file uses to
align a trailing comment (2 or more spaces before "#" match any such run; the code and the comment text are exact). Every replacement must
match exactly its stated number of times or the build stops; the result is compiled (a syntax check) before it is written. Every line the
replacements add or change carries the mark "v7" (engine_twofield_v7_selftest.py marks checks it).
Usage: python build_engine_twofield_v7.py --v3 PATH/engine_twofield_v3.py --out PATH/engine_twofield_v7.py
Why a build script: version 3 is about 1000 lines; the change touches 19 places. The lane's rule that version N's text is carried verbatim with
every change marked is easier to audit as a list of replacements than as a second copy of the file."""
import argparse
import re
import sys


def _s(x):
    """the snippets below start on the line after the opening quotes: that first newline is not part of them."""
    return x[1:] if x.startswith("\n") else x


def pattern_of(old):
    """old as a regex: exact everywhere except that a run of 2 or more spaces before '#' matches any such run (the comment alignment)."""
    parts = re.split(r"( {2,}#)", old)
    out = []
    for p in parts:
        if re.fullmatch(r" {2,}#", p):
            out.append("[ ]{2,}#")
        else:
            out.append(re.escape(p))
    return re.compile("".join(out))


# ------------------------------------------------------------------------------------------------------------------------- the replacements
R = []

# 1. the docstring: version 7's paragraph before version 3's (the lane's form: the previous version's docstring follows)
R.append(("docstring", _s(r'''
"""engine_twofield_v3.py (the busi lane, Computation C): version 2 with gate D's judgment applied (busi/notes/C_gateD_judgment_v1.md); every'''), _s(r'''
"""engine_twofield_v7.py (the busi lane, Computation C; inbox #158 and the builder's profile): version 3 with one performance option, --perf-v7
[--perf-v7-stages solves,light,writer], off by default, so that with it off every number this file writes is version 3's (the json names this
script and its args carry the 2 new arguments; nothing else differs). Every change against version 3 is marked "v7:". The option changes where
and when 3 stages of a record run, never what they compute: each stage is the same code on the same bytes, so the gate's verdict on every
compared quantity is expected BITWISE (RUN_engine_twofield_v7.md). (i) solves: inside the exact J.v (--nk-jvp exact) the 2 central-difference
field solves run side by side, the plus solve in this process and the minus solve in 1 worker process (spawned at start and configured like
main(): the field module, the kernel cache, the fixes flag, the solver options and tolerances, the same BLAS thread counts from the environment,
checked; it never touches the device); the record's solve counters advance as the serial pair advances them; a worker failure takes the serial
path and says so. (ii) light: per evaluation the light channels' eigendecompositions (on the device, version 3's eigT(clocked_gpu(...)), 0.23 s
each like the heavy ones) and their sector rotation run before the heavy channels' eigendecompositions, each channel's host-only part of the
continuation (mu at the midgap, the certified walk with its Cholesky ladder, the mu S selection and its count) runs on a thread while the device
diagonalises the heavy channels, and the cut check, the reference's columns and the fallback run after them on the main thread; the device
operations keep their order within each channel and are independent across channels. (iii) writer: the run files (json, fields, handoff, spectra,
s11) at every 5th record and the orbitals and light slices are written by a forked child from a copy-on-write snapshot, under temporary names in
.perf_v7_tmp beside the final ones and renamed into place when complete, the parent going on with the next record; the parent reaps the child
before the next write and at the end; a child that exits nonzero has its write redone in the foreground, and a second failure turns the stage off
for the rest of the run (printed). The per-record line "perf v7 ..." prints the pairs' seconds, the walks' seconds and the join waits, the writer's
state and the seconds the parent was blocked; the head carries perf_v7 (the configuration); the records carry nothing new.
Version 3's docstring follows.
engine_twofield_v3.py (the busi lane, Computation C): version 2 with gate D's judgment applied (busi/notes/C_gateD_judgment_v1.md); every'''), 1))

# 2. the imports
R.append(("imports", _s(r'''
import sys
import time

sys.dont_write_bytecode = True'''), _s(r'''
import sys
import threading                                      # v7: the light walks' threads
import time
import traceback                                      # v7: the worker's and the writer's error reports
import warnings                                       # v7: the fork's thread warning silenced (the child never touches the other threads' locks)

sys.dont_write_bytecode = True'''), 1))

# 3. the performance machinery, before tml_reset_v36
R.append(("perf block", "\n\ndef tml_reset_v36():\n", _s(r'''


# ------------------------------------------------------------------------------------------------------------ v7: the performance option
PERF = {"on": False, "stages": set(), "pool": None, "pool_start_s": None, "pool_ping": None, "R": None, "env": {}, "writer": None,
        "writer_disabled": False, "writer_failures": 0, "writer_runs": 0, "writer_last": None, "pending": [],
        "pairs": 0, "pair_main_s": 0.0, "pair_worker_s": 0.0, "pair_wall_s": 0.0, "light_walks": 0, "light_host_s": 0.0, "light_wait_s": 0.0,
        "writer_block_s": 0.0, "forks": 0, "fork_s": 0.0}                                             # v7: the option's state and its per-record counters
PERF_STAGES = ("solves", "light", "writer")                                                            # v7: --perf-v7-stages
PERF_RECORD_KEYS = ("pairs", "pair_main_s", "pair_worker_s", "pair_wall_s", "light_walks", "light_host_s", "light_wait_s", "writer_block_s", "forks", "fork_s")   # v7
PERF_THREAD_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")                       # v7: the BLAS thread counts the worker must share


def perf_on():
    """v7: whether --perf-v7 is on; every changed site tests this and runs version 3's statements when it is not."""
    return bool(PERF["on"])


def perf_reset_v7():
    """v7: the per-record counters behind the perf line."""
    for k_ in PERF_RECORD_KEYS:
        PERF[k_] = 0.0 if k_.endswith("_s") else 0


def _perf_cfg_v7(a):
    """v7: what main() sets in this process and the worker must set the same way (no device there)."""
    return {"fields_v17": bool(getattr(a, "fields_v17", False)), "field_kernel_cache": bool(getattr(a, "field_kernel_cache", False)),
            "v36_fixes": bool(getattr(a, "v36_fixes", False)), "collar_policy": a.collar_policy, "support_rel": a.support_rel, "site_fields": a.site_fields,
            "field_rtol": a.field_rtol, "field_atol": a.field_atol, "env": {k_: os.environ.get(k_) for k_ in PERF_THREAD_ENV}}


def _perf_worker_init(cfg):
    """v7: the solve worker's configuration: the same field module, kernel cache, fixes flag, solver options and tolerances as main(); the BLAS
    thread counts checked against the main process's (a different count could change a last bit of a BLAS reduction); C5.backend is not called,
    so the worker never touches the device."""
    global FLD
    for k_, v_ in cfg["env"].items():
        if os.environ.get(k_) != v_:
            raise RuntimeError("v7: the worker's %s is %r, the main process's %r" % (k_, os.environ.get(k_), v_))
    if cfg["fields_v17"]:
        import engine_fields_v17 as F17
        FLD = F17; V35.FLD = F17
    if cfg["field_kernel_cache"]:
        import engine_fields_kcache_v1 as KC
        import engine_fields_v4 as _F4, engine_fields_v12 as _F12, engine_fields_v14 as _F14, engine_fields_v16 as _F16
        KC.install([_F4, _F12, _F14, _F16, FLD]); V35.V36["kcache"] = KC.STATS
    V35.V36["fixes"] = bool(cfg["v36_fixes"])
    OPT["collar_policy"] = cfg["collar_policy"]; OPT["support_rel"] = cfg["support_rel"]; OPT["site_fields"] = cfg["site_fields"]
    SOLVE_TOL["rtol"] = cfg["field_rtol"]; SOLVE_TOL["atol"] = cfg["field_atol"]


def _perf_worker_ping():
    """v7: the worker's identity for the start line and the head."""
    return {"pid": os.getpid(), "env": {k_: os.environ.get(k_) for k_ in PERF_THREAD_ENV}, "field_module": V35.FLD.__name__,
            "kernel_cache": bool(V35.V36["kcache"]), "site_fields": OPT["site_fields"], "rtol": SOLVE_TOL["rtol"], "atol": SOLVE_TOL["atol"]}


def _perf_solve_worker(eps, J, g, R, ell):
    """v7: one field solve in the worker (engine_collapse_v36.solve, the call do_solve makes); the arrays and the counters' inputs come back, not
    the closures of info (the exact J.v reads the fields, the failure flag and the evaluation count only)."""
    t0 = time.time()
    Nn, vn, Wn, info = solve(eps, J, g, R, ell, None)
    nfev = int(info["ivp"]["nfev"]) if info.get("ivp") else None
    return Nn, vn, Wn, info.get("failed"), nfev, time.time() - t0


def perf_start_v7(a):
    """v7: --perf-v7: the stages and, for the solves stage, the worker process (spawned fresh, so that it never inherits the device; configured
    by _perf_worker_init; a start failure turns the stage off and says so)."""
    import concurrent.futures
    import multiprocessing
    stages = set(x_.strip() for x_ in str(a.perf_v7_stages).split(",") if x_.strip())
    bad = stages - set(PERF_STAGES)
    if bad:
        raise SystemExit("v7: unknown --perf-v7-stages %s (known: %s)" % (",".join(sorted(bad)), ",".join(PERF_STAGES)))
    PERF["on"] = True; PERF["stages"] = stages; PERF["env"] = {k_: os.environ.get(k_) for k_ in PERF_THREAD_ENV}
    if "solves" in stages:
        t0 = time.time(); pool = None
        try:
            pool = concurrent.futures.ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn"), initializer=_perf_worker_init,
                                                          initargs=(_perf_cfg_v7(a),))
            PERF["pool_ping"] = pool.submit(_perf_worker_ping).result(timeout=900)
            PERF["pool"] = pool; PERF["pool_start_s"] = time.time() - t0
        except Exception as ex:  # noqa: BLE001
            stages.discard("solves")
            if pool is not None:
                try:
                    pool.shutdown(wait=False)
                except Exception:  # noqa: BLE001
                    pass
            print("v7: the solve worker did not start (%s); the solves stage is off, the 2 central-difference solves run one after the other as in version 3" % str(ex)[:300], flush=True)
    print("v7: performance stages %s%s; BLAS threads %s" % (
        ",".join(sorted(stages)) or "none",
        ("; the solve worker pid %d ready in %.1f s (%s, kernel cache %s)" % (PERF["pool_ping"]["pid"], PERF["pool_start_s"], PERF["pool_ping"]["field_module"],
                                                                             PERF["pool_ping"]["kernel_cache"])) if PERF["pool"] is not None else "",
        json.dumps(PERF["env"])), flush=True)


def perf_pool_lost_v7(ex):
    """v7: a failure of the worker during the run: the serial path from here on, said once."""
    print("v7: the solve worker failed (%s); the solves stage is off from here, the 2 solves run one after the other" % str(ex)[:300], flush=True)
    pool = PERF.get("pool"); PERF["pool"] = None; PERF["stages"].discard("solves")
    if pool is not None:
        try:
            pool.shutdown(wait=False)
        except Exception:  # noqa: BLE001
            pass


def perf_run_setup_v7(a, R):
    """v7: what run() needs: the sites' radii for the worker's solve calls and the temporary directories of the writer."""
    PERF["R"] = np.asarray(R, float)
    if "writer" in PERF["stages"]:
        os.makedirs(os.path.join(a.out, ".perf_v7_tmp"), exist_ok=True)
        if a.slice_dir:
            os.makedirs(os.path.join(a.slice_dir, ".perf_v7_tmp"), exist_ok=True)


def perf_desc_v7(a):
    """v7: the head's perf_v7 entry (the configuration; no physics)."""
    torch = BK.get("torch")
    return {"option": "--perf-v7 --perf-v7-stages %s" % a.perf_v7_stages, "stages": sorted(PERF["stages"]),
            "solves": ("the exact J.v's 2 central-difference field solves side by side: the plus solve in this process, the minus solve in 1 worker process "
                       "(spawned; the same solve, kernel cache, tolerances and BLAS thread counts)") if "solves" in PERF["stages"] else "off",
            "solve_worker": PERF.get("pool_ping"), "solve_worker_startup_seconds": PERF.get("pool_start_s"), "blas_threads_env": PERF["env"],
            "light": ("each light channel's eigendecomposition and sector rotation before the heavy channels'; the host part of its continuation (mu, the "
                      "certified walk, the mu S selection) on a thread during the heavy channels' eigendecompositions; the cut check, the reference's columns "
                      "and the fallback after them on the main thread") if "light" in PERF["stages"] else "off",
            "writer": ("the run files and the slices written by a forked child from a copy-on-write snapshot under temporary names in .perf_v7_tmp, renamed "
                       "into place when complete; the parent reaps the child before the next write; a failed child's write is redone in the foreground, a "
                       "second failure turns the stage off") if "writer" in PERF["stages"] else "off",
            "versions": {"python": sys.version.split()[0], "numpy": np.__version__, "scipy": __import__("scipy").__version__, "torch": getattr(torch, "__version__", None)},
            "note": "where and when 3 stages run, nothing they compute; the written numbers are version 3's"}


def perf_solve_pair_v7(do_solve, srcs_p, srcs_m, a):
    """v7 (solves): the exact J.v's 2 central-difference solves side by side: the plus solve here through do_solve (version 3's accounting), the
    minus solve in the worker on the same bytes; the record's solve counters advance for it as do_solve would (solve_n by 1, solve_nfev by its count,
    solve_s by its own seconds). Without the worker, or on its failure, the 2 serial calls in version 3's order."""
    pool = PERF.get("pool")
    if pool is None or "solves" not in PERF["stages"]:
        return do_solve(srcs_p), do_solve(srcs_m)
    t0 = time.time()
    try:
        fut = pool.submit(_perf_solve_worker, np.asarray(srcs_m[0], float), np.asarray(srcs_m[1], float), a.g, PERF["R"], a.ell)
    except Exception as ex:  # noqa: BLE001
        perf_pool_lost_v7(ex)
        return do_solve(srcs_p), do_solve(srcs_m)
    sp_ = do_solve(srcs_p)
    t_main = time.time() - t0
    try:
        Nn, vn, Wn, failed, nfev, sec = fut.result()
    except Exception as ex:  # noqa: BLE001
        perf_pool_lost_v7(ex)
        return sp_, do_solve(srcs_m)
    TM["solve_s"] += sec; TM["solve_n"] += 1
    if nfev is not None:
        TM["solve_nfev"] += int(nfev)
    PERF["pairs"] += 1; PERF["pair_main_s"] += t_main; PERF["pair_worker_s"] += sec; PERF["pair_wall_s"] += time.time() - t0
    return sp_, (Nn, vn, Wn, ({"failed": failed} if failed else {"failed": None, "ivp": {"nfev": nfev}}))


# v7 (light): light_continue_v36 in 3 phases with the same statements in the same order per channel: A on the main thread right after the
# channel's eigendecomposition (the sector rotation touches the device), B host only (a thread), C on the main thread after the heavy channels.
def light_continue_phaseA_v7(a, ch, window, f_cur, e, U, n_sea, sgn_dev):
    """v7: light_continue_v36's lines before the route: the sector rotation (device, in place on U) and the host copies of the levels and sectors."""
    t0 = time.time()
    prev = window[-1]
    sigma, n_rot, imp = light_sectorize_v36(e, U, sgn_dev, a.light_tol_block)
    e_np = e.cpu().numpy().astype(float); sig = sigma.cpu().numpy().astype(float)
    route = a.light_transport
    info = {"rotated_blocks": n_rot, "sigma_impurity_max": imp, "nneg": int((e_np < 0).sum()), "route": route, "reanchored": False}
    return {"prev": prev, "sigma": sigma, "e_np": e_np, "sig": sig, "route": route, "info": info, "imp": imp, "seconds": time.time() - t0}


def light_continue_phaseB_v7(a, ch, A_, f_cur, n_sea, S_host):
    """v7: the banded route's host-only lines (mu at the midgap, the certified walk, the mu S selection and its count); no torch call, so a thread
    runs it while the main thread sits in the heavy channels' eigendecompositions."""
    t0 = time.time()
    prev = A_["prev"]; e_np = A_["e_np"]; sig = A_["sig"]; imp = A_["imp"]
    n_plus = prev.get("n_plus"); mu0 = prev.get("mu")
    mu1, gap_plus = light_mu_midgap_v36(e_np, sig, n_plus) if n_plus is not None else (None, None)
    walk = None; why = None; occ = None; gap_end = None
    if imp > a.light_sigma_tol:
        why = "sector impurity %.2e" % imp
    elif mu0 is None:
        why = "no anchored mu (the previous assignment left a sector inversion)"
    elif mu1 is None:
        why = "no midgap of the S = +1 sector at the trial fields"
    else:
        walk = light_walk_v36(ch, prev["f"], f_cur, mu0, mu1, prev.get("gap"), S_host, a.light_d_min, a.light_max_pieces)
        if not walk["certified"]:
            why = walk["why"]
    if why is None:
        occ, gap_end = light_select_v36(e_np, sig, mu1)
        if int(occ.sum()) != n_sea:
            why = "the mu S selection counts %d against n_sea %d" % (int(occ.sum()), n_sea)
    return {"n_plus": n_plus, "mu0": mu0, "mu1": mu1, "gap_plus": gap_plus, "walk": walk, "why": why, "occ": occ, "gap_end": gap_end, "seconds": time.time() - t0}


def light_continue_phaseC_v7(a, ch, window, f_cur, e, U, n_sea, A_, B_):
    """v7: light_continue_v36's remaining lines on the main thread: the cut check and the reference's columns (device), the fallback, the record
    entries; the routes v3 and v34-eigvals whole (they call the device)."""
    torch = BK["torch"]
    t0 = time.time()
    prev = A_["prev"]; sigma = A_["sigma"]; e_np = A_["e_np"]; sig = A_["sig"]; route = A_["route"]; info = A_["info"]
    if route == "v3":
        ref, inf3, mask = OPS3.continue_sea_v3(ch, window, f_cur, e, U, n_sea, a.tau, a.c_LZ, eig_at, clocked_host_sparse)
        info.update(inf3); info.update({"mode": "v3: %s" % inf3.get("mode"), "certified": None, "fallback": False, "why": None})
    elif route == "v34-eigvals":
        ref, inf4, mask = CERT.continue_sea_v34(ch, window, f_cur, e, U, n_sea, a.tau, a.c_LZ, eig_at, clocked_host_sparse, "eigvals", a.weyl_refine_max)
        info.update(inf4); info.update({"mode": "v34 eigvals: %s" % inf4.get("mode"), "fallback": False, "why": None})
    else:
        n_plus = B_["n_plus"]; mu1 = B_["mu1"]; gap_plus = B_["gap_plus"]; walk = B_["walk"]; why = B_["why"]; occ = B_["occ"]; gap_end = B_["gap_end"]; cut = None
        if why is None and walk["norm_dK_G"] > 0.0:
            cut = light_cut_check_v3(prev["ref"], e_np, sig, occ, U, mu1, walk["norm_dK_G"], gap_end)
            if cut["crossing_levels"] > 0:
                why = ("within-sector crossing at the cut below the convention's resolution: %d of the %d level(s) within reach of the cut are selected with weight "
                       "below 1/2 on the previous reference or unselected with weight above 1/2 (certified d_min %.2e, ||dK||_G %.2e); the spectral continuation "
                       "refills the sector; the LZ convention is taken" % (cut["crossing_levels"], cut["window_levels"], walk["d_min_certified"], walk["norm_dK_G"]))
        if why is None:
            ref = U[:, torch.as_tensor(occ, device=U.device)]; mask = occ
            info.update({"mode": "mu S selector, certified walk (%d pieces)" % walk["pieces"], "verified": True, "isolation_verified": True, "certified": walk,
                         "nsub": walk["pieces"], "fallback": False, "why": None, "mu": mu1, "gap": gap_end, "weyl_gap": gap_end, "gap_plus_half": gap_plus,
                         "n_plus": n_plus, "margin_min": walk["d_min_certified"], "norm_dH_G": walk["norm_dK_G"], "cut_check": cut, "cut_crossing": False})
        else:
            ref, inff, mask = OPS3.continue_sea_v3(ch, window, f_cur, e, U, n_sea, a.tau, a.c_LZ, eig_at, clocked_host_sparse)
            info.update(inff)
            info.update({"mode": "fallback: the LZ convention, version 3's maximum-overlap continuation (%s) (%s)" % (inff.get("mode"), why), "verified": False,
                         "isolation_verified": False, "certified": walk, "nsub": int(inff.get("nsub") or 1), "fallback": True, "why": why, "margin_min": None,
                         "cut_check": cut, "cut_crossing": bool(cut is not None and cut["crossing_levels"] > 0)})
    mask = np.asarray(mask, bool)
    if route != "banded" or info.get("fallback"):
        ok, n_plus_new = light_sorted_consistent_v36(e_np, sig, mask)
        mu_new, gap_plus = light_mu_midgap_v36(e_np, sig, n_plus_new) if ok else (None, None)
        gap_new = light_select_v36(e_np, sig, mu_new)[1] if mu_new is not None else None
        info.update({"mu": mu_new, "gap": gap_new, "weyl_gap": (gap_new if gap_new is not None else info.get("weyl_gap")), "gap_plus_half": gap_plus,
                     "n_plus": (n_plus_new if ok else None), "reanchored": bool(ok and info.get("fallback"))})
    info.update({"n_occ": int(mask.sum()), "n_sea": n_sea, "pockets_occupied_positive": int(np.sum(mask & (e_np > 0.0))),
                 "holes_empty_negative": int(np.sum(~mask & (e_np < 0.0))), "inversions_sector": light_sector_inversions_v36(e_np, sig, mask)})
    info["n_pocket"] = info["pockets_occupied_positive"] + info["holes_empty_negative"]
    info["_sigma"] = sigma
    info["seconds"] = A_["seconds"] + (B_["seconds"] if B_ is not None else 0.0) + (time.time() - t0)
    return ref, info, mask


def _light_phaseB_thread_v7(a, chL, A_, f_cur, n_sea, S_host, box):
    """v7: phase B on a thread; its result or its exception into box (the exception is raised again on the main thread, as version 3 raises it)."""
    try:
        box["B"] = light_continue_phaseB_v7(a, chL, A_, f_cur, n_sea, S_host)
    except BaseException as ex:  # noqa: BLE001
        box["err"] = (ex, traceback.format_exc())


def light_eval_begin_v7(a, chl, st, N_a, v_a, sgn_dev, S_host):
    """v7 (light): light_eval_v36's first half, taken before the heavy channels' eigendecompositions: per light channel the eigendecomposition (the
    device, the same call), phase A on the main thread, phase B started on its own thread. None when the stage is off (the caller then takes
    version 3's light_eval_v36 after the heavy channels)."""
    if "light" not in PERF["stages"]:
        return None
    f_cur = (N_a, v_a)
    handles = []
    for chL in chl:
        t_ = time.time()
        e, U = eigT(clocked_gpu(chL, N_a, v_a))
        sync_dev()
        TML["eigh_s"] += time.time() - t_; TML["eigh_n"] += 1
        s_ = st[chL.key]
        A_ = light_continue_phaseA_v7(a, chL, s_["window"], f_cur, e, U, s_["n_sea"], sgn_dev)
        box = {"B": None, "err": None}; th = None
        if A_["route"] == "banded":
            th = threading.Thread(target=_light_phaseB_thread_v7, args=(a, chL, A_, f_cur, s_["n_sea"], S_host, box), name="v7-light-walk-%d" % chL.kappa, daemon=True)
            th.start()
        handles.append((chL, e, U, A_, th, box, f_cur))
    return handles


def light_eval_end_v7(a, chl, st, cache_, handles):
    """v7 (light): light_eval_v36's second half, after the heavy channels: each walk joined (its exception raised here), phase C, the same cache
    entries and the same counters as version 3 (continuation_s as the 3 phases' seconds)."""
    for (chL, e, U, A_, th, box, f_cur) in handles:
        B_ = None
        if th is not None:
            t_w = time.time()
            th.join()
            PERF["light_wait_s"] += time.time() - t_w
            if box["err"] is not None:
                raise box["err"][0]
            B_ = box["B"]
            PERF["light_walks"] += 1; PERF["light_host_s"] += B_["seconds"]
        t_c = time.time()
        s_ = st[chL.key]
        ref, cinfo, mask = light_continue_phaseC_v7(a, chL, s_["window"], f_cur, e, U, s_["n_sea"], A_, B_)
        cinfo["_mask"] = mask
        TML["continuation_s"] += A_["seconds"] + (B_["seconds"] if B_ is not None else 0.0) + (time.time() - t_c)
        c_ = cinfo.get("certified")
        if isinstance(c_, dict):
            TML["walk_pieces"] += int(c_.get("pieces") or 0); TML["walk_cholesky"] += int(c_.get("cholesky_calls") or 0)
        if cinfo.get("fallback"):
            TML["walk_fallbacks"] += 1
        if cinfo.get("cut_crossing"):
            TML["cut_crossings"] += 1
        cache_[chL.key] = (None, e, U, ref, cinfo)


# v7 (writer): the run files and the slices written by a forked child, each file under a temporary name and renamed into place when complete.
def writer_active_v7():
    """v7: whether the writer stage acts (on, and not turned off by 2 failures)."""
    return perf_on() and "writer" in PERF["stages"] and not PERF["writer_disabled"]


def _tmp_path_v7(path):
    d_ = os.path.join(os.path.dirname(path), ".perf_v7_tmp")
    os.makedirs(d_, exist_ok=True)
    return os.path.join(d_, os.path.basename(path))


def run_file_v7(a, name):
    """v7: the path a run file is written to: version 3's with the writer stage off; with it on the same name under .perf_v7_tmp, renamed into
    place by commit_run_files_v7() once every file of the group is written."""
    final = os.path.join(a.out, name)
    if not writer_active_v7():
        return final
    tmp = _tmp_path_v7(final)
    PERF["pending"].append((tmp, final))
    return tmp


def commit_run_files_v7():
    """v7: the renames of run_file_v7's group (a no-op with the stage off)."""
    pend = PERF["pending"]; PERF["pending"] = []
    for tmp, final in pend:
        os.replace(tmp, final)


def json_dump_v7(path, obj):
    """v7: version 3's json.dump(obj, fh, default=float) at path; with the writer stage on, written under a temporary name and renamed into place."""
    if not writer_active_v7():
        with open(path, "w") as fh:
            json.dump(obj, fh, default=float)
        return
    tmp = _tmp_path_v7(path)
    with open(tmp, "w") as fh:
        json.dump(obj, fh, default=float)
    os.replace(tmp, path)


def savez_v7(path, arrs):
    """v7: np.savez(path, **arrs) (a slice; the arrays are host copies already); with the writer stage on, written by the forked child under a
    temporary name and renamed into place."""
    if not writer_active_v7():
        np.savez(path, **arrs)
        return

    def write_():
        tmp = _tmp_path_v7(path)
        np.savez(tmp, **arrs)
        os.replace(tmp, path)
    perf_fork_write_v7(os.path.basename(path), write_)


def persist_v7(label, fn):
    """v7: fn() writes the run files: here with the writer stage off, in a forked child with it on."""
    if writer_active_v7():
        perf_fork_write_v7(label, fn)
    else:
        fn()


def perf_fork_write_v7(label, fn):
    """v7: fn() in a forked child (a copy-on-write snapshot of the records and arrays at this instant), the parent going on; the child exits with
    os._exit (no atexit handlers, no device teardown: it never touches the device). The parent reaps the previous child first, so at most one write
    is in flight and 2 writes never interleave on one file."""
    perf_writer_wait_v7(block=True)
    t0 = time.time()
    sys.stdout.flush(); sys.stderr.flush()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pid = os.fork()
    if pid == 0:
        code = 0
        try:
            fn()
        except BaseException:  # noqa: BLE001
            code = 1
            try:
                sys.stderr.write("v7: the background write of %s failed in the child:\n%s" % (label, traceback.format_exc())); sys.stderr.flush()
            except BaseException:  # noqa: BLE001
                pass
        os._exit(code)
    PERF["writer"] = {"pid": pid, "label": label, "fn": fn, "t0": t0}
    PERF["forks"] += 1; PERF["fork_s"] += time.time() - t0


def perf_writer_wait_v7(block=True):
    """v7: the writer child reaped (blocking, or a poll); a nonzero exit has its write redone here, and a second failure turns the stage off."""
    w = PERF.get("writer")
    if w is None:
        return
    t0 = time.time()
    try:
        pid, status = os.waitpid(w["pid"], 0 if block else os.WNOHANG)
    except ChildProcessError:
        pid, status = w["pid"], 0
    if pid == 0:
        return
    if block:
        PERF["writer_block_s"] += time.time() - t0
    PERF["writer"] = None
    took = time.time() - w["t0"]
    PERF["writer_runs"] += 1; PERF["writer_last"] = (w["label"], took)
    ok = os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0
    if not ok:
        PERF["writer_failures"] += 1
        if PERF["writer_failures"] >= 2:
            PERF["writer_disabled"] = True
        print("v7: the background write of %s ended with status %d after %.0f s; it is redone in the foreground now%s" % (
            w["label"], status, took, "; the writer stage is off from here (2 failures)" if PERF["writer_disabled"] else ""), flush=True)
        w["fn"]()


def perf_finish_v7():
    """v7: the end of the run: the writer reaped, the worker stopped."""
    perf_writer_wait_v7(block=True)
    pool = PERF.get("pool")
    if pool is not None:
        try:
            pool.shutdown(wait=True)
        except Exception:  # noqa: BLE001
            pass
        PERF["pool"] = None


def perf_print_v7(trs):
    """v7: the per-record perf line; also polls the writer child (a failed one has its write redone here)."""
    perf_writer_wait_v7(block=False)
    w = PERF.get("writer")
    if w is not None:
        ws = "writing %s, %.0f s so far" % (w["label"], time.time() - w["t0"])
    elif PERF["writer_last"] is not None:
        ws = "the last write (%s) took %.0f s in the child" % PERF["writer_last"]
    else:
        ws = "idle"
    if PERF["writer_disabled"]:
        ws += "; the stage is off after %d failures" % PERF["writer_failures"]
    print("  perf v7 T/rs %.3f [stages %s]: solve pairs %d (plus solves here %.1f s, minus solves in the worker %.1f s, wall %.1f s: %.1f s saved); light walks %d on host threads %.1f s, %.1f s waited at the joins; writer %s, the parent blocked %.1f s this record (%d forks, %.2f s)" % (
        trs, ",".join(sorted(PERF["stages"])) or "none", PERF["pairs"], PERF["pair_main_s"], PERF["pair_worker_s"], PERF["pair_wall_s"],
        PERF["pair_main_s"] + PERF["pair_worker_s"] - PERF["pair_wall_s"], PERF["light_walks"], PERF["light_host_s"], PERF["light_wait_s"], ws,
        PERF["writer_block_s"], PERF["forks"], PERF["fork_s"]), flush=True)


def tml_reset_v36():
'''), 1))

# 4. run(): the radii for the worker and the temporary directories; the head's entry
R.append(("run setup", _s(r'''
    R = chs[0].R
    chl = light_channels_v36(a)'''), _s(r'''
    R = chs[0].R
    if perf_on():                                                                                      # v7: the worker's radii, the writer's directories
        perf_run_setup_v7(a, R)
    chl = light_channels_v36(a)'''), 1))
R.append(("head entry", _s(r'''
    cstr = {ch.kappa: STR.ChanStress(ch, torch, BK["dev"]) for ch in chs}
    cstrL = {chL.key: STR.ChanStress(chL, torch, BK["dev"]) for chL in chl}'''), _s(r'''
    if perf_on():                                                                                      # v7: the configuration in the head (no record changes)
        head["perf_v7"] = perf_desc_v7(a)
    cstr = {ch.kappa: STR.ChanStress(ch, torch, BK["dev"]) for ch in chs}
    cstrL = {chL.key: STR.ChanStress(chL, torch, BK["dev"]) for chL in chl}'''), 1))

# 5. save_fields: the run files' paths through run_file_v7 and the group's renames; dump_run_files_v7 beside it
R.append(("save_fields s11", 'os.path.join(a.out, "collapse_%s_s11.npz" % a.tag)', 'run_file_v7(a, "collapse_%s_s11.npz" % a.tag)', 1))
R.append(("save_fields fields", 'os.path.join(a.out, "collapse_%s_fields.npz" % a.tag)', 'run_file_v7(a, "collapse_%s_fields.npz" % a.tag)', 1))
R.append(("save_fields handoff", 'os.path.join(a.out, "collapse_%s_handoff.npz" % a.tag)', 'run_file_v7(a, "collapse_%s_handoff.npz" % a.tag)', 1))
R.append(("save_fields spectra and the dump closure", _s(r'''
            np.savez(os.path.join(a.out, "collapse_%s_spectra.npz" % a.tag), **{k: np.array(v_) for k, v_ in spectra.items()})
    last_prop_s = 0.0'''), _s(r'''
            np.savez(run_file_v7(a, "collapse_%s_spectra.npz" % a.tag), **{k: np.array(v_) for k, v_ in spectra.items()})
        commit_run_files_v7()                                                                           # v7: the group renamed into place (a no-op with the writer stage off)

    def dump_run_files_v7(extra=None):                                                                  # v7: the json and the npz files in one call (here, or in the writer's child)
        json_dump_v7(path, dict({"script": "engine_twofield_v7.py", "args": vars(a), "head": head, "records": recs}, **(extra or {})))
        save_fields()
    last_prop_s = 0.0'''), 1))

# 6. the plain iteration loop: the light channels before the heavy ones when the stage is on
R.append(("plain loop eigh", _s(r'''
            eig = eigh_all(chs, N_app, v_app, bool(a.streams))
            cache = dict(cache)'''), _s(r'''
            lt_ = light_eval_begin_v7(a, chl, st, N_app, v_app, sgn_dev, S_host) if perf_on() else None    # v7: the light eigendecompositions first, the walks on threads
            eig = eigh_all(chs, N_app, v_app, bool(a.streams))
            cache = dict(cache)'''), 1))
R.append(("plain loop light", _s(r'''
            del eig
            light_eval_v36(a, chl, st, cache, N_app, v_app, sgn_dev, S_host)                            # v36'''), _s(r'''
            del eig
            if lt_ is None:
                light_eval_v36(a, chl, st, cache, N_app, v_app, sgn_dev, S_host)                            # v36
            else:
                light_eval_end_v7(a, chl, st, cache, lt_)                                                   # v7: the walks joined, the references built'''), 1))

# 7. the two Newton evaluations (newton_record's and nk_stage_v32's evalG, identical text)
R.append(("evalG eigh", _s(r'''
        eig = eigh_all(chs, N_a, v_a, bool(a.streams))
        cache_ = dict(cache0)'''), _s(r'''
        lt_ = light_eval_begin_v7(a, chl, st, N_a, v_a, sgn_dev, S_host) if perf_on() else None    # v7: the light eigendecompositions first, the walks on threads
        eig = eigh_all(chs, N_a, v_a, bool(a.streams))
        cache_ = dict(cache0)'''), 2))
R.append(("evalG light", _s(r'''
        del eig
        light_eval_v36(a, chl, st, cache_, N_a, v_a, sgn_dev, S_host)                                 # v36'''), _s(r'''
        del eig
        if lt_ is None:
            light_eval_v36(a, chl, st, cache_, N_a, v_a, sgn_dev, S_host)                                 # v36
        else:
            light_eval_end_v7(a, chl, st, cache_, lt_)                                                    # v7: the walks joined, the references built'''), 2))

# 8. the exact J.v's two central-difference solves
R.append(("jvp pair", _s(r'''
            sp_ = do_solve((eps0 + t * deps, J0 + t * dJ) + tuple(srcs_[2:]))
            sm_ = do_solve((eps0 - t * deps, J0 - t * dJ) + tuple(srcs_[2:]))'''), _s(r'''
            if perf_on():                                                                               # v7: the 2 solves side by side (the minus solve in the worker)
                sp_, sm_ = perf_solve_pair_v7(do_solve, (eps0 + t * deps, J0 + t * dJ) + tuple(srcs_[2:]), (eps0 - t * deps, J0 - t * dJ) + tuple(srcs_[2:]), a)
            else:
                sp_ = do_solve((eps0 + t * deps, J0 + t * dJ) + tuple(srcs_[2:]))
                sm_ = do_solve((eps0 - t * deps, J0 - t * dJ) + tuple(srcs_[2:]))'''), 1))

# 9. the slices
R.append(("orbitals slice", _s(r'''
            saved = os.path.join(a.slice_dir, "orbitals_%s_T%07.2f.npz" % (a.tag, T / rs))
            np.savez(saved, **arrs)'''), _s(r'''
            saved = os.path.join(a.slice_dir, "orbitals_%s_T%07.2f.npz" % (a.tag, T / rs))
            savez_v7(saved, arrs)                                                                        # v7: by the writer's child with the stage on'''), 1))
R.append(("light slice", _s(r'''
            savedL = os.path.join(a.slice_dir, "light_%s_T%07.2f.npz" % (a.tag, T / rs))
            np.savez(savedL, **arrs)'''), _s(r'''
            savedL = os.path.join(a.slice_dir, "light_%s_T%07.2f.npz" % (a.tag, T / rs))
            savez_v7(savedL, arrs)                                                                       # v7: by the writer's child with the stage on'''), 1))

# 10. the three run-file writes
R.append(("periodic dump", _s(r'''
        if step % 5 == 0 or step == nsteps or step == step0:
            with open(path, "w") as fh:
                json.dump({"script": "engine_twofield_v3.py", "args": vars(a), "head": head, "records": recs}, fh, default=float)   # v3
            save_fields()'''), _s(r'''
        if step % 5 == 0 or step == nsteps or step == step0:
            persist_v7("the run files at T/rs %.4f" % (T / rs), dump_run_files_v7)                       # v7: version 3's json and npz writes, in the writer's child with the stage on'''), 1))
R.append(("s11 stop dump", _s(r'''
            with open(path, "w") as fh:
                json.dump({"script": "engine_twofield_v3.py", "args": vars(a), "head": head, "records": recs, "end_s11": end33}, fh, default=float)   # v3
            save_fields()
            break'''), _s(r'''
            persist_v7("the run files at the P2 stop", lambda: dump_run_files_v7({"end_s11": end33}))     # v7
            break'''), 1))
R.append(("final dump", _s(r'''
    with open(path, "w") as fh:
        json.dump(dict({"script": "engine_twofield_v3.py", "args": vars(a), "head": head, "records": recs}, **({"end_v32": end32} if end32 is not None else {}),   # v3
                       **({"end_s11": end33} if end33 is not None else {}), **({"end_v34": end34} if end34 is not None else {})), fh, default=float)
    save_fields()'''), _s(r'''
    persist_v7("the run files at the end", lambda: dump_run_files_v7(dict(**({"end_v32": end32} if end32 is not None else {}),                      # v7
                                                                        **({"end_s11": end33} if end33 is not None else {}), **({"end_v34": end34} if end34 is not None else {}))))
    perf_finish_v7()                                                                                    # v7: the writer reaped, the worker stopped'''), 1))

# 11. the light-follow json names this script
R.append(("light-follow script", '"script": "engine_twofield_v3.py (light-follow)"', '"script": "engine_twofield_v7.py (light-follow)"', 1))

# 12. per record: the counters' reset and the perf line
R.append(("record reset", _s(r'''
        tm_reset()
        tml_reset_v36()                                                                                 # v36
'''), _s(r'''
        tm_reset()
        tml_reset_v36()                                                                                 # v36
        perf_reset_v7()                                                                                 # v7
'''), 1))
R.append(("perf line", _s(r'''
        if s11 is not None:
            print("%s T/rs %.4f: M min outside the cell %s at r %s (negative points %d); U_phys min %s;'''), _s(r'''
        if perf_on():                                                                                   # v7: the perf line (the pairs, the walks, the writer)
            perf_print_v7(T / rs)
        if s11 is not None:
            print("%s T/rs %.4f: M min outside the cell %s at r %s (negative points %d); U_phys min %s;'''), 1))

# 13. main(): the 2 arguments, the start before the device, the finish after the run
R.append(("arguments", _s(r'''
    ap.add_argument("--light-follow", dest="light_follow", default=None, help="(v36 e) DIR/TAG of a stored run: the light channels alone on its applied fields")
    a = ap.parse_args()'''), _s(r'''
    ap.add_argument("--light-follow", dest="light_follow", default=None, help="(v36 e) DIR/TAG of a stored run: the light channels alone on its applied fields")
    ap.add_argument("--perf-v7", dest="perf_v7", action="store_true", help="(v7) the performance option: the exact J.v's 2 field solves side by side, the light walks on host threads during the heavy eigendecompositions, the run files and slices written by a forked child; off: version 3's numbers bitwise")   # v7
    ap.add_argument("--perf-v7-stages", dest="perf_v7_stages", default="solves,light,writer", help="(v7) the stages --perf-v7 turns on, from solves,light,writer")   # v7
    a = ap.parse_args()'''), 1))
R.append(("start and finish", _s(r'''
    C5.backend(a.gpu)
    run(a)'''), _s(r'''
    if a.perf_v7:                                                                                        # v7: the stages and the worker, before the device is initialised here
        perf_start_v7(a)
    C5.backend(a.gpu)
    run(a)
    perf_finish_v7()                                                                                     # v7'''), 1))


def build(src):
    out = src
    log = []
    for name, old, new, count in R:
        pat = pattern_of(old)
        n = len(pat.findall(out))
        if n != count:
            raise SystemExit("build: replacement %r matched %d times, expected %d; the version 3 text differs from input 02 at that place" % (name, n, count))
        out = pat.sub(lambda m: new, out)
        log.append((name, count))
    return out, log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v3", required=True, help="engine_twofield_v3.py (input 02)")
    ap.add_argument("--out", required=True, help="engine_twofield_v7.py to write")
    a = ap.parse_args()
    src = open(a.v3, encoding="utf-8").read()
    if "engine_twofield_v3.py (the busi lane, Computation C): version 2 with gate D's judgment applied" not in src:
        raise SystemExit("build: %s is not engine_twofield_v3.py" % a.v3)
    out, log = build(src)
    compile(out, a.out, "exec")
    with open(a.out, "w", encoding="utf-8") as fh:
        fh.write(out)
    n_added = sum(1 for l_ in out.splitlines() if "v7" in l_) - sum(1 for l_ in src.splitlines() if "v7" in l_)
    print("built %s from %s: %d replacements (%s); %d lines with the mark v7 added; %d -> %d lines" % (
        a.out, a.v3, len(log), ", ".join("%s x%d" % q for q in log), n_added, len(src.splitlines()), len(out.splitlines())))


if __name__ == "__main__":
    main()
