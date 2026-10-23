#!/usr/bin/env python
"""build_engine_collapse_v36u.py (busi lane PERF build, inbox #299/#302; final4/PERF/PERF_TRACKER.md rows W2-P1 and W2-P2; deliveries/lead_gaps/REVIEWA_perf_w2_v1.md
sec. 2 P1 and P2): writes engine_collapse_v36u.py from engine_collapse_v36p_fixv4c.py (md5 b623d3b0, the W2 production engine) by exact string
replacements in build_engine_collapse_v36p.py's form. (The review names the build v36s_perf; v36t is taken on the box by the fix tournament's ptc
engine, so this is v36u.) Each OLD is copied exactly from engine_collapse_v36p_fixv4c.py and must occur exactly once (else the build aborts); each
NEW contains inserted or changed text so that replacing every NEW by its OLD restores the source byte for byte (checked here). Every inserted or
changed line carries the marker "v36u perf" (a trailing comment; the docstring's lines "[v36u perf]"; checked here).

Two options, both off by default, so that the defaults are fixv4c bitwise in every record number, printed line, stored array and entry key (the
json's "args" dict alone gains the three keys):
  W2-P1 --dg-eigh-cache-gib G (0 = off) [--dg-eigh-cache-host 0|1]: the step-scoped eigendecomposition cache. Inside dg_step_v36k only, eigh_all
        keeps the (e, U) it computes keyed by (channel, the exact bytes of N and v) in an LRU capped at G GiB; a request whose key is present returns
        the stored pair and is counted exactly as a computed one (TM["eigh_n"]), so every stored count is unchanged; cleared at the end of every
        sub-step and of the step; off in the record solve.
  W2-P2 --dg-node-workers 3 (0 = off): the three Gauss-node field solves of each dg iterate in three persistent spawned workers (engine_pairsolve_v2.py
        beside the engine), each sent as soon as its node's sources exist so that it runs behind the next node's eigendecompositions, the results read
        back in node order; a worker error solves that node in the parent (counted). dgk_node_v36k is split into dgk_node_sources_v36k (its device
        part, unchanged) and the solve.
Usage: python build_engine_collapse_v36u.py [engine_collapse_v36p_fixv4c.py path] [engine_collapse_v36u.py path]   (defaults: beside this script)."""
import hashlib
import os
import py_compile
import sys

MARK = "v36u perf"
SRC_MD5 = "b623d3b010fa846cffcf914b4fa4900a"

REPLACEMENTS = [
("docstring",
r'''"""engine_collapse_v36p.py (reviewer A for the busi lane; PRIM_ruling_v1): version 36p''',
r'''"""engine_collapse_v36u.py (busi PERF build; REVIEWA_perf_w2_v1 P1 and P2; built by build_engine_collapse_v36u.py over engine_collapse_v36p_fixv4c.py md5 b623d3b0) [v36u perf]
version 36u = version 36p fixv4c with two performance options, off by default, so that the defaults are fixv4c bitwise (the json's "args" gains [v36u perf]
dg_eigh_cache_gib, dg_eigh_cache_host and dg_node_workers): --dg-eigh-cache-gib G, the step-scoped eigendecomposition cache (inside dg_step_v36k [v36u perf]
eigh_all returns the stored (e, U) of a channel whose field bytes (N, v) it decomposed before in the same sub-step or the step's start; requests [v36u perf]
counted as before; cleared per sub-step and step; off in the record solve; --dg-eigh-cache-host 1 parks the pairs in pinned host memory), and [v36u perf]
--dg-node-workers 3, the dg iterate's three Gauss-node field solves in three persistent workers (engine_pairsolve_v2), pipelined behind the next [v36u perf]
node's eigendecompositions and read back in node order. Both are bitwise by construction (identical bytes in, the same cusolver or LAPACK call; the [v36u perf]
same solve() on the same arrays); the per-process counters go to the record's "v36u_perf" entry (present only with an option on). [v36u perf]
[v36u perf: fixv4c's docstring follows] engine_collapse_v36p.py (reviewer A for the busi lane; PRIM_ruling_v1): version 36p'''),
("module state: the cache and the node workers",
r'''PAIR = {"n": 0, "workers": None, "cfg": None, "products": 0, "seconds": 0.0, "fallbacks": 0, "hits": 0, "misses": 0, "errors": []}   # version 36b
''',
r'''PAIR = {"n": 0, "workers": None, "cfg": None, "products": 0, "seconds": 0.0, "fallbacks": 0, "hits": 0, "misses": 0, "errors": []}   # version 36b
EIGC = {"on": False, "cap_bytes": 0, "host": False, "entries": None, "bytes": 0, "hits": 0, "misses": 0, "evictions": 0, "rec_hits": 0, "rec_misses": 0}   # v36u perf (W2-P1): the step-scoped eigendecomposition cache (--dg-eigh-cache-gib): (channel, the exact bytes of N and v) -> (e, U); on inside dg_step_v36k only
NODEW = {"n": 0, "workers": None, "cfg": None, "solves": 0, "fallbacks": 0, "seconds": 0.0, "kc_hits": 0, "kc_misses": 0, "errors": []}   # v36u perf (W2-P2): the Gauss nodes' field solves in persistent worker processes (--dg-node-workers 3; engine_pairsolve_v2)


def eigc_fbytes(N, v):   # v36u perf
    """v36u perf: the field part of the cache key: the exact bytes of the float64 arrays clocked_gpu assembles from (np.asarray(N, float), np.asarray(v, float)); identical bytes give the identical matrix and, the eigensolver being deterministic on one device, the identical (e, U)."""   # v36u perf
    return (np.ascontiguousarray(np.asarray(N, float)).tobytes(), np.ascontiguousarray(np.asarray(v, float)).tobytes())   # v36u perf


def eigc_get(key):   # v36u perf
    """v36u perf: the stored (e, U) of a key (moved to the LRU's end), copied back to the device when parked on the host (an exact copy); None when absent."""   # v36u perf
    ent = EIGC["entries"].get(key) if EIGC["entries"] is not None else None   # v36u perf
    if ent is None:   # v36u perf
        return None   # v36u perf
    EIGC["entries"].move_to_end(key); EIGC["hits"] += 1; EIGC["rec_hits"] += 1   # v36u perf
    e, U, on_host = ent   # v36u perf
    if on_host:   # v36u perf
        e = e.to(BK["dev"]); U = U.to(BK["dev"])   # v36u perf: the exact bytes back on the device
    return e, U   # v36u perf


def eigc_put(key, pair):   # v36u perf
    """v36u perf: store the pair (e, U) under key at the LRU's end (parked in pinned host memory under --dg-eigh-cache-host 1 on a CUDA device); the oldest entries dropped above the byte cap; nothing when the cache is off."""   # v36u perf
    if key is None or not EIGC["on"]:   # v36u perf
        return   # v36u perf
    e, U = pair; EIGC["misses"] += 1; EIGC["rec_misses"] += 1   # v36u perf
    nb = int(U.numel() * U.element_size() + e.numel() * e.element_size())   # v36u perf
    if nb > EIGC["cap_bytes"]:   # v36u perf
        return   # v36u perf
    on_host = bool(EIGC["host"] and U.is_cuda)   # v36u perf
    if on_host:   # v36u perf
        torch = BK["torch"]   # v36u perf
        Uh = torch.empty(U.shape, dtype=U.dtype, device="cpu", pin_memory=True); Uh.copy_(U); eh = e.detach().to("cpu"); e, U = eh, Uh   # v36u perf: an exact copy to pinned host memory
    EIGC["entries"][key] = (e, U, on_host); EIGC["bytes"] += nb   # v36u perf
    while EIGC["bytes"] > EIGC["cap_bytes"] and len(EIGC["entries"]) > 1:   # v36u perf
        _k, (eo, Uo, _h) = EIGC["entries"].popitem(last=False); EIGC["bytes"] -= int(Uo.numel() * Uo.element_size() + eo.numel() * eo.element_size()); EIGC["evictions"] += 1   # v36u perf


def eigc_clear():   # v36u perf
    """v36u perf: every entry dropped (the end of a sub-step and of a step)."""   # v36u perf
    if EIGC["entries"] is not None:   # v36u perf
        EIGC["entries"].clear()   # v36u perf
    EIGC["bytes"] = 0   # v36u perf


def eigc_set(on):   # v36u perf
    """v36u perf: the cache on (inside dg_step_v36k, when a cap is set) or off; cleared either way."""   # v36u perf
    eigc_clear(); EIGC["on"] = bool(on and EIGC["cap_bytes"] > 0 and EIGC["entries"] is not None)   # v36u perf


def nodew_start():   # v36u perf
    """v36u perf: spawn NODEW["n"] workers once (engine_pairsolve_v2.worker_main: the parent's own solve() with the same field module, kernel cache, OPT and SOLVE_TOL; the charge returned beside N, v)."""   # v36u perf
    import multiprocessing as mp   # v36u perf
    import engine_pairsolve_v2 as PW2   # v36u perf
    ctx = mp.get_context("spawn"); ws = []   # v36u perf
    for _ in range(NODEW["n"]):   # v36u perf
        pc, cc = ctx.Pipe(); p = ctx.Process(target=PW2.worker_main, args=(cc, NODEW["cfg"]), daemon=True); p.start(); ws.append((p, pc))   # v36u perf
    for p, pc in ws:   # v36u perf
        pc.recv()   # v36u perf
    NODEW["workers"] = ws   # v36u perf
    print("v36u perf: %d node-solve workers started (pids %s)" % (len(ws), ",".join(str(p.pid) for p, _ in ws)), flush=True)   # v36u perf


def nodew_send(k, eps, J, g, R, ell):   # v36u perf
    """v36u perf: node k's sources to worker k; False when the workers are unavailable (the caller then solves in the parent)."""   # v36u perf
    try:   # v36u perf
        if NODEW["workers"] is None:   # v36u perf
            nodew_start()   # v36u perf
        NODEW["workers"][k][1].send((np.asarray(eps, float), np.asarray(J, float), g, R, ell)); return True   # v36u perf
    except Exception as ex:   # v36u perf
        NODEW["errors"].append("%s: %s" % (type(ex).__name__, str(ex)[:200])); NODEW["workers"] = None; return False   # v36u perf


def nodew_recv(k):   # v36u perf
    """v36u perf: node k's result (N, v, E or None, failed) from worker k; None on a worker error (the caller then solves in the parent)."""   # v36u perf
    try:   # v36u perf
        r_ = NODEW["workers"][k][1].recv()   # v36u perf
        if r_[0] != "ok":   # v36u perf
            NODEW["errors"].append(str(r_[1])[:200]); return None   # v36u perf
        _o, Nn, vn, failed, _nfev, En, dh, dm = r_; NODEW["kc_hits"] += int(dh); NODEW["kc_misses"] += int(dm)   # v36u perf
        return Nn, vn, En, failed   # v36u perf
    except Exception as ex:   # v36u perf
        NODEW["errors"].append("%s: %s" % (type(ex).__name__, str(ex)[:200])); NODEW["workers"] = None; return None   # v36u perf
'''),
("eigh_all: the cache",
r'''def eigh_all(chs, N, v, streams=False):
    """the channels' eigendecompositions at the fields (N, v): one after another, or (streams) from threads, each on its own CUDA stream."""
    torch = BK["torch"]
    out = {}
    if not streams or BK["dev"] is None or BK["dev"].type != "cuda":
        for ch in chs:
            out[ch.kappa] = eigT_t(clocked_gpu(ch, N, v))
        return out
    t_ = time.time()
    hms = {ch.kappa: clocked_gpu(ch, N, v) for ch in chs}
    sync_dev()
    ss = {ch.kappa: torch.cuda.Stream(device=BK["dev"]) for ch in chs}

    def work(k):
        with torch.cuda.stream(ss[k]):
            e, U = torch.linalg.eigh(hms[k])
            out[k] = (e.real, U)
        ss[k].synchronize()
    th = [threading.Thread(target=work, args=(ch.kappa,)) for ch in chs]
    for t in th:
        t.start()
    for t in th:
        t.join()
    sync_dev()
    hms.clear()
    TM["eigh_s"] += time.time() - t_; TM["eigh_n"] += len(chs)
    return out
''',
r'''def eigh_all(chs, N, v, streams=False):
    """the channels' eigendecompositions at the fields (N, v): one after another, or (streams) from threads, each on its own CUDA stream. v36u perf: inside dg_step_v36k under --dg-eigh-cache-gib a channel whose (N, v) bytes are in the step-scoped cache returns its stored (e, U), counted as an eigendecomposition (TM["eigh_n"]) as before; the hits in TM["eigh_cache_hits"] (the timing dict)."""   # v36u perf
    torch = BK["torch"]
    out = {}
    fb = eigc_fbytes(N, v) if EIGC["on"] else None   # v36u perf
    todo = []   # v36u perf: the channels to decompose, in the given order (all of them when the cache is off)
    for ch in chs:   # v36u perf
        key = (ch.kappa,) + fb if fb is not None else None   # v36u perf
        hit = eigc_get(key) if key is not None else None   # v36u perf
        if hit is not None:   # v36u perf
            out[ch.kappa] = hit; TM["eigh_n"] = TM.get("eigh_n", 0) + 1; TM["eigh_cache_hits"] = TM.get("eigh_cache_hits", 0) + 1   # v36u perf: the request counted as an eigendecomposition (every stored count unchanged)
        else:   # v36u perf
            todo.append((ch, key))   # v36u perf
    if EIGC["on"] and not todo:   # v36u perf
        return {ch.kappa: out[ch.kappa] for ch in chs}   # v36u perf
    if not streams or BK["dev"] is None or BK["dev"].type != "cuda":
        for ch, key in todo:   # v36u perf
            out[ch.kappa] = eigT_t(clocked_gpu(ch, N, v))
            eigc_put(key, out[ch.kappa])   # v36u perf
        return {ch.kappa: out[ch.kappa] for ch in chs} if EIGC["on"] else out   # v36u perf: the channels' order kept
    t_ = time.time()
    hms = {ch.kappa: clocked_gpu(ch, N, v) for ch, _k in todo}   # v36u perf
    sync_dev()
    ss = {ch.kappa: torch.cuda.Stream(device=BK["dev"]) for ch, _k in todo}   # v36u perf

    def work(k):
        with torch.cuda.stream(ss[k]):
            e, U = torch.linalg.eigh(hms[k])
            out[k] = (e.real, U)
        ss[k].synchronize()
    th = [threading.Thread(target=work, args=(ch.kappa,)) for ch, _k in todo]   # v36u perf
    for t in th:
        t.start()
    for t in th:
        t.join()
    sync_dev()
    hms.clear()
    TM["eigh_s"] += time.time() - t_; TM["eigh_n"] += len(todo)   # v36u perf: len(chs) when the cache is off
    for ch, key in todo:   # v36u perf
        eigc_put(key, out[ch.kappa])   # v36u perf
    return {ch.kappa: out[ch.kappa] for ch in chs} if EIGC["on"] else out   # v36u perf: the channels' order kept
'''),
("dgk_node_v36k: the device part split off (head)",
r'''    n = len(R)   # v36k chart
    eig = eigh_all(chs, Nk, vk, bool(a.streams))   # v36k chart
''',
r'''    eps, J, tinfo, masks, refs, spec = dgk_node_sources_v36k(a, chs, st, R, base_s, Nk, vk, anchor, keep_ref, check)   # v36u perf: the device part (below, unchanged), then the solve as before
    Nn, vn, Wn, info = solve(eps, J, a.g, R, a.ell, None)   # v36k chart
    if info.get("failed"):   # v36k chart
        return None, None, None, info, tinfo, masks, refs, spec   # v36k chart; v36l attr
    return np.asarray(Nn, float), np.asarray(vn, float), float(FLD.charge(info, a.g)), info, tinfo, masks, refs, spec   # v36k chart; v36l attr


def dgk_node_sources_v36k(a, chs, st, R, base_s, Nk, vk, anchor, keep_ref=False, check=False):   # v36u perf (W2-P2)
    """v36u perf: dgk_node_v36k's device part, its lines up to the field solve unchanged: the eigendecompositions at (Nk, vk), the tracked reference, the node's sources. Returns (eps, J, per-channel track info, masks, refs, spec), so that the solve can run in a worker."""   # v36u perf
    n = len(R)   # v36k chart
    eig = eigh_all(chs, Nk, vk, bool(a.streams))   # v36k chart
'''),
("dgk_node_v36k: the device part split off (tail)",
r'''    del eig   # v36k chart
    Nn, vn, Wn, info = solve(eps, J, a.g, R, a.ell, None)   # v36k chart
    if info.get("failed"):   # v36k chart
        return None, None, None, info, tinfo, masks, refs, spec   # v36k chart; v36l attr
    return np.asarray(Nn, float), np.asarray(vn, float), float(FLD.charge(info, a.g)), info, tinfo, masks, refs, spec   # v36k chart; v36l attr
''',
r'''    del eig   # v36k chart
    return eps, J, tinfo, masks, refs, spec   # v36u perf
'''),
("dgk_substep_v36k: the node loop pipelined",
r'''        x_new = []; nodeE = []; tinfos = []; nres = []; masks = []; specs = []; bases = []   # v36k chart; v36l attr
        for k in range(3):   # v36k chart
''',
r'''        x_new = []; nodeE = []; tinfos = []; nres = []; masks = []; specs = []; bases = []   # v36k chart; v36l attr
        if NODEW["n"] >= 3:   # v36u perf (W2-P2): the three nodes' device parts in node order, each node's solve sent to its worker as soon as its sources exist (it runs behind the next node's eigendecompositions), the results read back in node order
            t_w = time.time(); pend = []   # v36u perf
            for k in range(3):   # v36u perf
                s_k = s_nodes[k]   # v36u perf
                base_s = {ch.kappa: tuple((1.0 - s_k) * b0 + s_k * b1 for b0, b1 in zip(base0[ch.kappa], base1[ch.kappa])) for ch in chs}   # v36u perf: the serial loop's line
                eps_k, J_k, ti_k, mk, _r, sp_k = dgk_node_sources_v36k(a, chs, st, R, base_s, x[k][0], x[k][1], dgp_point_anchor_v36p(anchor, k))   # v36u perf
                pend.append((s_k, base_s, ti_k, mk, sp_k, eps_k, J_k, nodew_send(k, eps_k, J_k, a.g, R, a.ell)))   # v36u perf
            for k in range(3):   # v36u perf
                s_k, base_s, ti_k, mk, sp_k, eps_k, J_k, sent = pend[k]   # v36u perf
                res = nodew_recv(k) if sent else None   # v36u perf
                if res is None:   # v36u perf: a worker error: this node's solve in the parent, the same code on the same arrays; counted
                    NODEW["fallbacks"] += 1; Nn_, vn_, _W, info_f = solve(eps_k, J_k, a.g, R, a.ell, None)   # v36u perf
                    res = (Nn_, vn_, (None if info_f.get("failed") else float(FLD.charge(info_f, a.g))), info_f.get("failed"))   # v36u perf
                Nn_, vn_, Ek, failed_k = res; NODEW["solves"] += 1; n_solves += 1   # v36u perf: counted per node in node order, as the serial loop counts
                if failed_k:   # v36u perf
                    why = "the field solve failed at node %d (s = %.6f) of iteration %d: %s" % (k + 1, s_k, it, failed_k)   # v36u perf: the serial loop's message
                    TM["eigh_n"] -= len(chs) * (2 - k)   # v36u perf: the later nodes' decompositions, which the serial loop never did, taken off the count
                    for k2 in range(k + 1, 3):   # v36u perf: the later nodes' replies drained (the pipes stay in step)
                        if pend[k2][-1]:   # v36u perf
                            nodew_recv(k2)   # v36u perf
                    break   # v36u perf
                Nk, vk = np.asarray(Nn_, float), np.asarray(vn_, float)   # v36u perf: dgk_node_v36k's return conversion
                x_new.append((Nk, vk)); nodeE.append(Ek); tinfos.append(ti_k); masks.append(mk); specs.append(sp_k); bases.append(base_s)   # v36u perf: the serial loop's line
                nres.append(max(float(np.max(np.abs(Nk - x[k][0]))), float(np.max(np.abs(vk - x[k][1])))))   # v36u perf: the serial loop's line
            NODEW["seconds"] += time.time() - t_w   # v36u perf
        for k in (range(3) if NODEW["n"] < 3 else ()):   # v36k chart; v36u perf: the serial loop unless the node workers are on
'''),
("dgk_substep_v36k: the cache cleared at the sub-step's end",
r'''    sync_dev(); ent["seconds"] = time.time() - t0   # v36k chart
''',
r'''    eigc_clear()   # v36u perf (W2-P1): the sub-step's cached decompositions dropped (the next sub-step's chord starts at the endpoint's output fields)
    sync_dev(); ent["seconds"] = time.time() - t0   # v36k chart
'''),
("dg_step_v36k: the cache on",
r'''    n_eigh0 = TM["eigh_n"]   # v36k chart
    anchor0 = dgk_anchor_v36k(a, chs, st, f_acc)   # v36k chart
''',
r'''    eigc_set(True)   # v36u perf (W2-P1): the cache on for this step (the anchor's decompositions at f_acc serve the first E_start pass and the first sub-step's s = 0 and iterate 1)
    n_eigh0 = TM["eigh_n"]   # v36k chart
    anchor0 = dgk_anchor_v36k(a, chs, st, f_acc)   # v36k chart
'''),
("dg_step_v36k: the cache off",
r'''    sync_dev()   # v36k chart
    ent["seconds"] = time.time() - t0   # v36k chart
''',
r'''    sync_dev()   # v36k chart
    eigc_set(False)   # v36u perf (W2-P1): off and cleared before the record solve
    ent["seconds"] = time.time() - t0   # v36k chart
'''),
("run: the record's v36u_perf entry",
r'''            PAIR.update({"products": 0, "seconds": 0.0, "fallbacks": 0, "hits": 0, "misses": 0, "errors": []})   # per record, as JV
''',
r'''            PAIR.update({"products": 0, "seconds": 0.0, "fallbacks": 0, "hits": 0, "misses": 0, "errors": []})   # per record, as JV
        if EIGC["cap_bytes"] > 0 or NODEW["n"]:   # v36u perf: the per-process counters of the perf options (present only with an option on; exempt in the gate as the v36b entry is)
            rec["v36u_perf"] = {"eigh_cache": {"cap_gib": EIGC["cap_bytes"] / 2.0 ** 30, "host": EIGC["host"], "hits": EIGC["rec_hits"], "misses": EIGC["rec_misses"], "evictions_total": EIGC["evictions"]},   # v36u perf
                                "node_workers": {"n": NODEW["n"], "solves": NODEW["solves"], "fallbacks": NODEW["fallbacks"], "seconds": NODEW["seconds"], "worker_kcache_hits": NODEW["kc_hits"], "worker_kcache_misses": NODEW["kc_misses"], "errors": NODEW["errors"][-3:]}}   # v36u perf
            EIGC["rec_hits"] = 0; EIGC["rec_misses"] = 0; NODEW.update({"solves": 0, "fallbacks": 0, "seconds": 0.0, "kc_hits": 0, "kc_misses": 0, "errors": []})   # v36u perf: per record, as the v36b entry
'''),
("main: the options",
r'''    ap.add_argument("--streams", type=int, default=0)
''',
r'''    ap.add_argument("--streams", type=int, default=0)
    ap.add_argument("--dg-eigh-cache-gib", dest="dg_eigh_cache_gib", type=float, default=0.0, help="(v36u perf, W2-P1) the byte cap of the step-scoped eigendecomposition cache in GiB (0: off, fixv4c bitwise); a hit returns the stored (e, U) of the same channel at the same field bytes; one entry is 16 n^2 bytes (n = 2 nsites)")   # v36u perf
    ap.add_argument("--dg-eigh-cache-host", dest="dg_eigh_cache_host", type=int, default=0, choices=[0, 1], help="(v36u perf, W2-P1) 1: the cached pairs parked in pinned host memory and copied back on a hit (exact), for a device without room for the cap")   # v36u perf
    ap.add_argument("--dg-node-workers", dest="dg_node_workers", type=int, default=0, choices=[0, 3], help="(v36u perf, W2-P2) 3: the three Gauss nodes' field solves of each dg iterate in three persistent worker processes (engine_pairsolve_v2.py beside the engine), each sent behind the next node's eigendecompositions; 0: serial (fixv4c)")   # v36u perf
'''),
("main: the options read",
r'''    C5.backend(a.gpu)
    run(a)
''',
r'''    if a.dg_eigh_cache_gib < 0.0:   # v36u perf
        ap.error("v36u perf: --dg-eigh-cache-gib must be nonnegative")   # v36u perf
    if (a.dg_eigh_cache_gib > 0.0 or a.dg_node_workers) and not DGK["on"]:   # v36u perf
        ap.error("v36u perf: --dg-eigh-cache-gib and --dg-node-workers act inside the chart-frozen dg step (--step dg --dg-chart frozen)")   # v36u perf
    if a.dg_eigh_cache_gib > 0.0:   # v36u perf
        import collections as _co; EIGC.update({"cap_bytes": int(a.dg_eigh_cache_gib * 2.0 ** 30), "host": bool(a.dg_eigh_cache_host), "entries": _co.OrderedDict()})   # v36u perf
    if a.dg_node_workers:   # v36u perf
        NODEW["n"] = int(a.dg_node_workers)   # v36u perf
        NODEW["cfg"] = {"engine": os.path.splitext(os.path.basename(__file__))[0], "fld": FLD.__name__, "kcache": bool(a.field_kernel_cache), "opt": dict(OPT), "tol": dict(SOLVE_TOL)}   # v36u perf: this file's own module name (PAIR["cfg"] names engine_collapse_v36c)
    C5.backend(a.gpu)
    run(a)
'''),
]


def build(text):
    """the forward replacements on the source text; every OLD must occur exactly once. Returns the built text."""
    for name, old, new in REPLACEMENTS:
        n = text.count(old)
        if n != 1:
            raise SystemExit("build_engine_collapse_v36u: the OLD string of '%s' occurs %d times in the source (1 required): the source is not engine_collapse_v36p_fixv4c.py" % (name, n))
        text = text.replace(old, new)
    return text


def reverse(text):
    """the reverse replacements (NEW -> OLD, in reverse order); every NEW must occur exactly once."""
    for name, old, new in reversed(REPLACEMENTS):
        n = text.count(new)
        if n != 1:
            raise SystemExit("build_engine_collapse_v36u: the NEW text of '%s' occurs %d times in the built file (1 required)" % (name, n))
        text = text.replace(new, old)
    return text


def md5(b):
    return hashlib.md5(b).hexdigest()


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "engine_collapse_v36p_fixv4c.py")
    dst = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(os.path.abspath(src)), "engine_collapse_v36u.py")
    raw = open(src, "rb").read()
    if md5(raw) != SRC_MD5:
        raise SystemExit("build_engine_collapse_v36u: the source md5 is %s, not fixv4c's %s" % (md5(raw), SRC_MD5))
    text = raw.decode("utf-8")
    out = build(text)
    back = reverse(out)
    if back != text:
        raise SystemExit("build_engine_collapse_v36u: the reverse replacements do not restore the source (a NEW text overlaps another)")
    srcset = set(text.splitlines())
    unmarked = [l_ for l_ in out.splitlines() if l_ not in srcset and MARK not in l_]
    if unmarked:
        raise SystemExit("build_engine_collapse_v36u: %d new lines without the marker, the first: %r" % (len(unmarked), unmarked[0][:200]))
    marked = sum(1 for l_ in out.splitlines() if MARK in l_)
    with open(dst, "wb") as fh:
        fh.write(out.encode("utf-8"))
    py_compile.compile(dst, doraise=True)
    print("built %s from %s: %d replacements, %d marked lines, every new line marked; source md5 %s; built md5 %s; the reverse replacements restore the source byte for byte; compiled; needs engine_pairsolve_v2.py beside it under --dg-node-workers 3" % (
        dst, src, len(REPLACEMENTS), marked, md5(raw), md5(out.encode("utf-8"))))


if __name__ == "__main__":
    main()
