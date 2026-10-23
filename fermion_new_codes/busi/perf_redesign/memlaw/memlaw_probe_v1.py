#!/usr/bin/env python
"""memlaw_probe_v1.py (busi, inbox #384): run a Newton engine of the v36nq lineage (engine_collapse_v36nq.py or its rd13 build) UNMODIFIED
under phase markers and read torch's caching-allocator peaks per phase, with a census of the live CUDA tensors at chosen marks.

How: the engine file is imported as a module (bytecode writing off), then module-level functions are wrapped (the engine resolves its
globals at call time, so its own calls go through the wrappers): every wrapper entry and exit is an event; at each event the device peak
since the previous event (torch.cuda.max_memory_allocated, then reset_peak_memory_stats) is booked to the label stack that was active over
that segment. The exact J.v (a closure inside nk_stage_v32) is reached by wrapping F.jvp inside the rounds_v36c / nk_loop_v32 wrappers.
Nothing the engine computes is changed (the wrappers only read allocator counters, and the census only walks gc's object list).

Usage: memlaw_probe_v1.py --engine PATH --out PATH.json [--census-per-label 2] [--max-wall S] -- <engine args>
"""
import argparse
import atexit
import gc
import importlib.util
import json
import os
import sys
import threading
import time

ap = argparse.ArgumentParser()
ap.add_argument("--engine", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--census-per-label", type=int, default=2)
ap.add_argument("--max-wall", type=float, default=0.0, help="dump and exit after this many seconds (0: no cap)")
argv = sys.argv[1:]
if "--" not in argv:
    sys.exit("memlaw_probe_v1: engine arguments go after --")
k = argv.index("--")
P = ap.parse_args(argv[:k])
ENGINE_ARGV = argv[k + 1:]

sys.dont_write_bytecode = True
eng_dir = os.path.dirname(os.path.abspath(P.engine))
sys.path.insert(0, eng_dir)
stem = os.path.splitext(os.path.basename(P.engine))[0]

import numpy as np  # noqa: E402
import torch  # noqa: E402

spec = importlib.util.spec_from_file_location(stem, P.engine)
E = importlib.util.module_from_spec(spec)
sys.modules[stem] = E
spec.loader.exec_module(E)

T0 = time.time()
ST = {"stack": ["run"], "segments": {}, "timeline": [], "census": [], "census_n": {}, "global_peak": 0, "global_peak_label": None,
      "global_peak_t": None, "events": 0, "host_hwm_kB": 0, "nsites": None, "lock": threading.Lock(), "dumped": False}


def _dev():
    d = E.BK.get("dev")
    if d is None or getattr(d, "type", "cpu") != "cuda" or not torch.cuda.is_initialized():
        return None
    return d


def _host_kB():
    try:
        with open("/proc/self/status") as fh:
            for ln in fh:
                if ln.startswith("VmRSS:"):
                    return int(ln.split()[1])
    except OSError:
        pass
    return -1


def _event(name):
    """book the segment that ends now to the current label stack; then reset the peak."""
    d = _dev()
    ST["events"] += 1
    if d is None:
        return
    pk = torch.cuda.max_memory_allocated(d)
    cur = torch.cuda.memory_allocated(d)
    lab = "/".join(ST["stack"])
    s = ST["segments"].setdefault(lab, {"peak": 0, "n": 0, "max_at_end": 0})
    s["n"] += 1
    if pk > s["peak"]:
        s["peak"] = pk
    if cur > s["max_at_end"]:
        s["max_at_end"] = cur
    if pk > ST["global_peak"]:
        ST["global_peak"] = pk; ST["global_peak_label"] = lab; ST["global_peak_t"] = time.time() - T0
    hk = _host_kB()
    if hk > ST["host_hwm_kB"]:
        ST["host_hwm_kB"] = hk
    ST["timeline"].append([round(time.time() - T0, 3), name, lab, pk, cur, hk])
    torch.cuda.reset_peak_memory_stats(d)


def _classify(shape, n2):
    if len(shape) == 2:
        r, c = shape
        if r == n2 and c == n2:
            return "2n x 2n (U, dense h_m, U copies)"
        if r == n2 and c == n2 // 2:
            return "2n x n (sea, R2, Uo, Uu, ref, A^T)"
        if r == n2 // 2 and c == n2:
            return "n x 2n (overlap A = ref^H U)"
        if r == n2 // 2 and c == n2 // 2:
            return "n x n (Hm, M, den)"
        if r == n2:
            return "2n x k (dust, partial column sets)"
        if c == n2:
            return "k x 2n"
        if r * c * 1.0 >= 1e6:
            return "other 2-d >= 1e6 elements (%dx%d)" % (r, c)
    if len(shape) == 3:
        return "3-d (banded scratch W)"
    return "small or 1-d"


def census(tag):
    """live CUDA tensors grouped by class, unique storages counted once."""
    d = _dev()
    if d is None:
        return
    nm = ST["census_n"].get(tag, 0)
    if nm >= P.census_per_label:
        return
    ST["census_n"][tag] = nm + 1
    t_ = time.time()
    n2 = 2 * int(ST["nsites"] or 0)
    seen = set(); groups = {}; tot = 0
    for o in gc.get_objects():
        try:
            if not torch.is_tensor(o) or o.device.type != "cuda" or o.is_sparse or o.layout != torch.strided:
                continue
            st_ = o.untyped_storage()
            p_ = st_.data_ptr()
            if p_ in seen:
                continue
            seen.add(p_)
            nb = st_.nbytes()
            g_ = _classify(tuple(o.shape), n2)
            q = groups.setdefault(g_, {"n": 0, "bytes": 0})
            q["n"] += 1; q["bytes"] += nb; tot += nb
        except Exception:
            continue
    ST["census"].append({"tag": tag, "t": round(time.time() - T0, 3), "label": "/".join(ST["stack"]), "allocated": torch.cuda.memory_allocated(d),
                         "reserved": torch.cuda.memory_reserved(d), "census_bytes": tot, "groups": groups, "census_s": round(time.time() - t_, 2)})


def wrap(obj, name, label, census_enter=None, census_exit=None):
    f = getattr(obj, name)

    def w(*a, **kw):
        _event("enter " + label)
        if census_enter:
            census(census_enter)
        ST["stack"].append(label)
        try:
            return f(*a, **kw)
        finally:
            _event("exit " + label)
            ST["stack"].pop()
            if census_exit:
                census(census_exit)
    w.__wrapped__ = f
    setattr(obj, name, w)


def mark(obj, name, label):
    f = getattr(obj, name)

    def w(*a, **kw):
        _event(label)
        return f(*a, **kw)
    setattr(obj, name, w)


# the phases (module-level names the engine calls through its globals)
wrap(E, "prepare_dust", "prep", census_exit="prep_exit")
wrap(E, "resume_slice_v34", "resume", census_exit="resume_exit")
wrap(E, "newton_record_v32", "newton_record")
wrap(E, "nk_stage_v32", "nk_stage")
wrap(E, "confirm_v32", "confirm", census_enter="confirm_enter")
wrap(E, "eigh_all", "eigh_all", census_exit="eigh_all_exit")
wrap(E, "eigT_t", "eigT")
wrap(E, "eig_at", "eig_at")
wrap(E, "clocked_gpu", "clocked_gpu")
wrap(E, "continue_ref_v34", "continuation")
wrap(E, "solve", "field_solve")
wrap(E, "pair_solve", "pair_solve")
wrap(E, "release_round_v36m", "v36m_release")
wrap(E, "restore_round_v36m", "v36m_restore", census_exit="v36m_restore_exit")
wrap(E, "iter_cache_restore", "iter_cache_restore")
wrap(E, "commutator_norm", "commutator")
wrap(E, "inside_energy_sp", "M1_inside")
wrap(E, "energy_cut_current_sp", "M1_cut")
wrap(E.STR, "stress_parts", "stress")
if hasattr(E, "rd13_eigT"):
    wrap(E, "rd13_eigT", "eigT_banded")
    wrap(E, "rd13_jvp_X", "sternheimer")
mark(E, "sync_dev", "sync")
mark(E, "dev_csr", "dev_csr")
_np_savez = np.savez


def _savez(*a, **kw):
    _event("enter slice_save"); ST["stack"].append("slice_save")
    try:
        return _np_savez(*a, **kw)
    finally:
        _event("exit slice_save"); ST["stack"].pop()


E.np.savez = _savez
_gc_collect = gc.collect


def _gcc(*a, **kw):
    _event("gc.collect (after the propagation)")
    census("record_end_after_prop")
    return _gc_collect(*a, **kw)


E.gc.collect = _gcc


def _wrap_jvp(F):
    j = getattr(F, "jvp", None)
    if j is None or getattr(j, "_memlaw", False):
        return

    def jw(*a, **kw):
        _event("enter jvp"); census("jvp_enter"); ST["stack"].append("jvp")
        try:
            return j(*a, **kw)
        finally:
            _event("exit jvp"); ST["stack"].pop()
    jw._memlaw = True
    F.jvp = jw


for nm_ in ("rounds_v36c", "nk_loop_v32", "nk_loop_v36c"):
    f0 = getattr(E, nm_)

    def mk(f0, nm_):
        def w(*a, **kw):
            for x in a:
                if callable(x) and hasattr(x, "jvp"):
                    _wrap_jvp(x)
            _event("enter " + nm_); ST["stack"].append(nm_)
            try:
                return f0(*a, **kw)
            finally:
                _event("exit " + nm_); ST["stack"].pop()
        return w
    setattr(E, nm_, mk(f0, nm_))


def dump(why):
    with ST["lock"]:
        d = _dev()
        out = {"probe": "memlaw_probe_v1", "why": why, "engine": P.engine, "argv": ENGINE_ARGV, "wall_s": time.time() - T0,
               "nsites": ST["nsites"], "global_peak_allocated": ST["global_peak"], "global_peak_label": ST["global_peak_label"],
               "global_peak_t": ST["global_peak_t"], "host_rss_hwm_kB": ST["host_hwm_kB"], "events": ST["events"],
               "segments": ST["segments"], "census": ST["census"], "timeline": ST["timeline"],
               "v36m": dict(E.OFF.get("rec") or {}) if hasattr(E, "OFF") else None}
        if d is not None:
            ms = torch.cuda.memory_stats(d)
            out["reserved_peak"] = ms.get("reserved_bytes.all.peak"); out["reserved_now"] = torch.cuda.memory_reserved(d)
            fr, totb = torch.cuda.mem_get_info(d); out["device_total"] = totb; out["device_free_now"] = fr
            out["device_name"] = torch.cuda.get_device_name(d)
        if hasattr(E, "RD13") and E.RD13.get("mod") is not None:
            out["rd13_stats"] = {k_: v_ for k_, v_ in E.RD13["mod"].STATS.items() if isinstance(v_, (int, float, str)) or v_ is None}
        tmp = P.out + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(out, fh, default=float)
        os.replace(tmp, P.out)


def _periodic():
    while True:
        time.sleep(60)
        try:
            dump("periodic")
        except Exception as ex:  # noqa: BLE001
            print("memlaw_probe_v1: periodic dump failed: %s" % ex, flush=True)
        if P.max_wall and time.time() - T0 > P.max_wall:
            try:
                _event("wall cap"); dump("max-wall %.0f s reached" % P.max_wall)
            finally:
                print("memlaw_probe_v1: max-wall reached, exit", flush=True)
                os._exit(3)


threading.Thread(target=_periodic, daemon=True).start()
atexit.register(lambda: dump("exit"))

for i_, a_ in enumerate(ENGINE_ARGV):
    if a_ == "--nsites":
        ST["nsites"] = int(ENGINE_ARGV[i_ + 1])
sys.argv = [P.engine] + ENGINE_ARGV
try:
    E.main()
finally:
    _event("main returned")
    dump("main returned")
