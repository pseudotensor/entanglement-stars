#!/usr/bin/env python
"""build_engine_collapse_v36nq_rd13m.py (busi, analyst; inbox #387, MEMLAW_8192_v1 Option 1): engine_collapse_v36nq_rd13m.py from the RD-1
build engine_collapse_v36nq_rd13.py (md5 7d362033..., busi/notes/PERF_RD1_build_v1.md) by marked replacement blocks, in the form of
build_engine_collapse_v36nq_rd13.py.
Rules: the source's md5 must begin with 7d362033 (--expect-src-md5); every OLD anchor is copied verbatim and occurs exactly once (checked);
every line a NEW block adds or changes carries "rd13m stream" (checked); the reverse pass (NEW -> OLD) restores the source byte for byte
(checked in memory and on the written file; --reverse); the output compiles.
    python build_engine_collapse_v36nq_rd13m.py --src .../engine_collapse_v36nq_rd13.py [--dst .../engine_collapse_v36nq_rd13m.py] [--check-only]
    python build_engine_collapse_v36nq_rd13m.py --reverse --src .../engine_collapse_v36nq_rd13m.py --dst .../restored.py --expect .../engine_collapse_v36nq_rd13.py

The option --nk-cache-host off | on (default off: nothing of the patch runs but dictionary reads and the option check, so every output is
the source engine's bitwise). On (the channel-streamed caches of MEMLAW_8192_v1 Option 1; requires --mix newton, --nk-jvp exact,
--reference R2, --streams 0, a CUDA device, and neither --weyl-oracle nor --massless-ball):
  (1) nk_stage_v32's evaluation runs per channel: the eigendecomposition (eigh_all's sequential call eigT_t(clocked_gpu(...))), the
      continuation, the two R2 densities the sources read (the same dens calls on the same tensors), then U and R2 to pinned host memory by
      exact copy (size and strides kept); e, the mask and cinfo stay. The channels' computations are independent, so the order change
      (channel by channel instead of all eigendecompositions first) changes no value.
  (2) the exact J.v takes each channel's U back by exact copy, with the next channel's copy started one ahead on a side stream (the
      default stream waits on its event before use); R2 back only for the first products' reference check.
  (3) the record's post phase: U and R2 back per channel where they are read (the R2 overlap, the commutator, the M1 sums); before the
      state update the chosen evaluation's R2 come back to the device and are installed as now; after it the previous record's R2 (held by
      the solve's reference cycles until the record's collection) are released by one gc.collect (the collection v36j does after the
      propagation, done earlier as well; nothing read or written changes), with run()'s stale last-channel locals; U back per channel for
      the Simpson half step and the propagation.
  (4) the handoff stress by stress_parts_chunked_v36s: the column-local work (the sparse products, the elementwise products, the
      two-component site sums) in chunks of 1024 columns into full-width site arrays, the column reductions at full width as
      engine_stress_v2.stress_parts (measured bitwise equal at 4096 and 8192 sites, memlaw_post_v1).
  Each record carries an "rd13m" entry (bytes and seconds out and in, prefetch hits and misses, density misses) and one printed line.
Blocks:
  A  the module docstring's lineage lines;
  B  before NSR: SCH (the option's state), HostT_v36s, sch_* helpers, evalG_stream_v36s, stress_parts_chunked_v36s, sch_configure, sch_record;
  C  nk_stage_v32.evalG: the streamed evaluation under the option;
  D  nk_stage_v32.jvp: U (and R2 for the check) back per channel, with the one-ahead prefetch;
  E1 the record's continuation diagnostics loop (U and R2 for the overlap);  E2 the commutator;  E3 the M1 balance's new R2;
  E4 the state update: the R2 installed from the host, then the release;  E5 the Simpson half step's U;  E6 the stress;  E7 the propagation's U;
  E8 the record's "rd13m" entry;  F main(): sch_configure after rd13_configure;  G the option --nk-cache-host (after --jvp).
"""
import argparse
import hashlib
import os
import py_compile

MARK = "rd13m stream"
SRC_MD5_PREFIX = "7d362033"


def blk(s):
    return s[1:] if s.startswith("\n") else s


EDITS = []

EDITS.append(("A lineage", blk(r'''
"""engine_collapse_v36nq_rd13.py (busi, analyst; PERF RD-1 + RD-3; built by build_engine_collapse_v36nq_rd13.py over v36nq, md5 8549c542) [rd13 banded]
'''), blk(r'''
"""engine_collapse_v36nq_rd13m.py (busi, analyst; inbox #387; built by build_engine_collapse_v36nq_rd13m.py over v36nq_rd13, md5 7d362033) [rd13m stream]
version 36nq_rd13m = version 36nq_rd13 with --nk-cache-host off | on (off: the source's code path, bitwise): on, the Newton solve's [rd13m stream]
    evaluation caches (U and R2 per channel) live in pinned host memory and come to the device one channel at a time, the record's post [rd13m stream]
    phase reads them per channel, the previous record's R2 are released at the state update, and the handoff stress runs column-chunked [rd13m stream]
    (MEMLAW_8192_v1 Option 1; bitwise by exact copies and lifetimes). Version 36nq_rd13's docstring follows. [rd13m stream]
engine_collapse_v36nq_rd13.py (busi, analyst; PERF RD-1 + RD-3; built by build_engine_collapse_v36nq_rd13.py over v36nq, md5 8549c542) [rd13 banded]
''')))

EDITS.append(("B helpers", blk(r'''
NSR = {"on": False, "prev_sig": None, "rec": None, "prev_m": None, "other": None}   # version 36c (--nk-set resolve); 36d: prev_m, other
'''), blk(r'''
SCH = {"on": False, "stream": None, "pf": None, "rec": None, "width": 1024, "pinned": None}   # rd13m stream: --nk-cache-host (the channel-streamed caches)
def _sch_rec():   # rd13m stream: the record's counters
    if SCH["rec"] is None:   # rd13m stream
        SCH["rec"] = {"bytes_out": 0, "bytes_in": 0, "seconds_out": 0.0, "seconds_in": 0.0, "parked": 0, "fetched": 0, "prefetch_hits": 0, "prefetch_misses": 0, "dens_misses": 0, "releases": 0}   # rd13m stream
    return SCH["rec"]   # rd13m stream
def _host_v36s(t):   # rd13m stream: _host_v36m's exact host copy (size and strides kept, pinned where granted) with the option's own pinned flag (v36m's record entry untouched)
    torch = BK["torch"]; h = None   # rd13m stream
    if SCH.get("pinned") is not False:   # rd13m stream
        try:   # rd13m stream
            h = torch.empty_strided(tuple(t.size()), tuple(t.stride()), dtype=t.dtype, device="cpu", pin_memory=True); SCH["pinned"] = True   # rd13m stream
        except (RuntimeError, TypeError):   # rd13m stream
            SCH["pinned"] = False   # rd13m stream
    if h is None:   # rd13m stream
        h = torch.empty_strided(tuple(t.size()), tuple(t.stride()), dtype=t.dtype, device="cpu")   # rd13m stream
    h.copy_(t)   # rd13m stream
    return h   # rd13m stream
class HostT_v36s(object):   # rd13m stream: a device tensor parked in pinned host memory (an exact copy, its size and strides kept), with the densities computed from it before it left
    __slots__ = ("h", "dens", "nbytes")   # rd13m stream
    def __init__(self, t, dens=None):   # rd13m stream
        t_ = time.time()   # rd13m stream
        self.h = _host_v36s(t); self.dens = dens or {}; self.nbytes = t.numel() * t.element_size()   # rd13m stream: the copy_ to the host waits for the device
        r_ = _sch_rec(); r_["bytes_out"] += self.nbytes; r_["seconds_out"] += time.time() - t_; r_["parked"] += 1   # rd13m stream
    @property   # rd13m stream
    def shape(self):   # rd13m stream
        return self.h.shape   # rd13m stream
def sch_drop_pf():   # rd13m stream: an unused prefetch: its copy completed, then its device buffer released
    p_ = SCH["pf"]   # rd13m stream
    if p_ is not None:   # rd13m stream
        p_[2].synchronize(); SCH["pf"] = None   # rd13m stream
def sch_get(x, nxt=None):   # rd13m stream: x's device tensor (x itself when it is not parked); with nxt, nxt's copy started on the side stream (one ahead)
    if not isinstance(x, HostT_v36s):   # rd13m stream
        return x   # rd13m stream
    torch = BK["torch"]; dev = BK["dev"]; t_ = time.time(); r_ = _sch_rec()   # rd13m stream
    p_ = SCH["pf"]   # rd13m stream
    if p_ is not None and p_[0] is x:   # rd13m stream: the prefetched copy: the default stream waits for it
        torch.cuda.current_stream(dev).wait_event(p_[2]); d = p_[1]; SCH["pf"] = None; r_["prefetch_hits"] += 1   # rd13m stream
    else:   # rd13m stream
        sch_drop_pf()   # rd13m stream
        d = _dev_v36m(x.h); r_["prefetch_misses"] += 1   # rd13m stream: _dev_v36m's copy_ from pinned memory, in stream order
    r_["bytes_in"] += x.nbytes; r_["fetched"] += 1   # rd13m stream
    if nxt is not None and isinstance(nxt, HostT_v36s) and nxt is not x:   # rd13m stream
        b_ = torch.empty_strided(tuple(nxt.h.size()), tuple(nxt.h.stride()), dtype=nxt.h.dtype, device=dev)   # rd13m stream: allocated on the default stream, used there after the event
        e0 = torch.cuda.Event(); e0.record(torch.cuda.current_stream(dev))   # rd13m stream
        SCH["stream"].wait_event(e0)   # rd13m stream
        with torch.cuda.stream(SCH["stream"]):   # rd13m stream
            b_.copy_(nxt.h, non_blocking=True)   # rd13m stream
            e1 = torch.cuda.Event(); e1.record(SCH["stream"])   # rd13m stream
        SCH["pf"] = (nxt, b_, e1)   # rd13m stream
    r_["seconds_in"] += time.time() - t_   # rd13m stream
    return d   # rd13m stream
def sch_next(cache_, chs, ch, slot=2):   # rd13m stream: the parked tensor of the channel after ch (the first after the last: the next product's first)
    if not SCH["on"]:   # rd13m stream
        return None   # rd13m stream
    k_ = [c_.kappa for c_ in chs].index(ch.kappa)   # rd13m stream
    return cache_[chs[(k_ + 1) % len(chs)].kappa][slot]   # rd13m stream
def dens_v36s(A, Phi, f=None):   # rd13m stream: dens (engine_collapse_v5.dens) with a parked R2: the density computed from it before it left (else fetched and counted)
    if isinstance(Phi, HostT_v36s):   # rd13m stream
        if f is None and id(A) in Phi.dens:   # rd13m stream
            return Phi.dens[id(A)]   # rd13m stream
        _sch_rec()["dens_misses"] += 1   # rd13m stream
        Phi = sch_get(Phi)   # rd13m stream
    return C5.dens(A, Phi, f)   # rd13m stream
def evalG_stream_v36s(a, chs, st, cache0, sources_fn, do_solve, N_a, v_a):   # rd13m stream: nk_stage_v32.evalG with --nk-cache-host on (the docstring's (1))
    sch_drop_pf()   # rd13m stream
    cache_ = dict(cache0)   # rd13m stream
    for ch in chs:   # rd13m stream
        e, U = eigT_t(clocked_gpu(ch, N_a, v_a))   # rd13m stream: eigh_all's sequential path's call
        t_c = time.time()   # rd13m stream
        s_ = st[ch.kappa]   # rd13m stream
        ref, cinfo, mask = continue_ref_v34(a, ch, s_["window"], (N_a, v_a), e, U, s_["n_sea"])   # rd13m stream
        cinfo["_mask"] = mask   # rd13m stream
        TM["continuation_s"] += time.time() - t_c   # rd13m stream
        TM["fast" if cinfo.get("nsub", 1) == 1 else "midpoint"] += 1   # rd13m stream
        if _same_storage_v36m(ref, U):   # rd13m stream
            raise RuntimeError("rd13m: the continued R2 is a view of U (not handled by --nk-cache-host)")   # rd13m stream
        dd_ = {id(ch.h): C5.dens(ch.h, ref), id(ch.Q): C5.dens(ch.Q, ref)}   # rd13m stream: the sources' two calls
        cache_[ch.kappa] = (None, e, HostT_v36s(U), HostT_v36s(ref, dd_), cinfo)   # rd13m stream
        del U, ref   # rd13m stream
    srcs_ = sources_fn(cache_)   # rd13m stream
    return cache_, srcs_, do_solve(srcs_)   # rd13m stream
def stress_parts_chunked_v36s(cs, cols, f):   # rd13m stream: engine_stress_v2.stress_parts, the column-local work in chunks (the docstring's (4))
    torch = cs.torch   # rd13m stream
    if cols.shape[1] == 0:   # rd13m stream
        z = np.zeros(cols.shape[0] // 2)   # rd13m stream
        return {k: z.copy() for k in ("Gr", "Gi", "G0i", "fc", "b", "sy", "eps", "J")}   # rd13m stream
    fw = torch.as_tensor(np.asarray(f, float), device=cs.dev, dtype=torch.float64)   # rd13m stream
    ns_, nc = cols.shape[0] // 2, cols.shape[1]; W_ = SCH["width"]   # rd13m stream
    site = lambda X: X.reshape(-1, 2, X.shape[1]).sum(dim=1)   # rd13m stream
    g = torch.empty((ns_, nc), dtype=torch.complex128, device=cs.dev); g0 = torch.empty_like(g)   # rd13m stream
    fcn = torch.empty((ns_, nc), dtype=torch.float64, device=cs.dev); b = torch.empty_like(fcn); sy = torch.empty_like(fcn); eps = torch.empty_like(fcn); Jd = torch.empty_like(fcn)   # rd13m stream
    for j0 in range(0, nc, W_):   # rd13m stream
        c = cols[:, j0:j0 + W_]   # rd13m stream
        A = cs.hbond @ c; Qc = cs.Q @ c; QA = cs.Q @ A   # rd13m stream
        H = cs.h @ c; QH = cs.Q @ H   # rd13m stream
        g[:, j0:j0 + W_] = 0.5 * site(A.conj() * Qc + QA.conj() * c)   # rd13m stream
        g0[:, j0:j0 + W_] = 0.5 * site(H.conj() * Qc + QH.conj() * c)   # rd13m stream
        Sc = cs.hsite @ c   # rd13m stream
        Xc = 1j * (cs.hsite @ Qc - cs.Q @ Sc)   # rd13m stream
        fcn[:, j0:j0 + W_] = site((c.conj() * Xc).real)   # rd13m stream
        b[:, j0:j0 + W_] = site((c.conj() * c).real * cs.sz[:, None])   # rd13m stream
        c0 = c[0::2, :]; c1 = c[1::2, :]   # rd13m stream
        sy[:, j0:j0 + W_] = 2.0 * (c0.conj() * c1).imag   # rd13m stream
        eps[:, j0:j0 + W_] = site((c.conj() * H).real)   # rd13m stream
        Jd[:, j0:j0 + W_] = site((c.conj() * Qc).real)   # rd13m stream
        del A, Qc, QA, H, QH, Sc, Xc, c0, c1, c   # rd13m stream
    red = lambda X: (X * fw[None, :]).sum(dim=1).cpu().numpy()   # rd13m stream
    gw = (g * fw[None, :].to(g.dtype)).sum(dim=1).cpu().numpy(); g0w = (g0 * fw[None, :].to(g0.dtype)).sum(dim=1).cpu().numpy()   # rd13m stream
    return {"Gr": gw.real, "Gi": gw.imag, "G0i": g0w.imag, "fc": red(fcn), "b": red(b), "sy": red(sy), "eps": red(eps), "J": red(Jd)}   # rd13m stream
def sch_stress(cs, cols, f):   # rd13m stream: the stress parts by the source's routine, or chunked under the option
    return stress_parts_chunked_v36s(cs, cols, f) if SCH["on"] else STR.stress_parts(cs, cols, f)   # rd13m stream
def sch_install(cache, chs):   # rd13m stream: the chosen evaluation's R2 back on the device before the state update (in the cache, so every later reader sees the device tensor)
    if not SCH["on"]:   # rd13m stream
        return   # rd13m stream
    sch_drop_pf()   # rd13m stream
    for ch in chs:   # rd13m stream
        hm_, e_, U_, ref_, ci_ = cache[ch.kappa]   # rd13m stream
        cache[ch.kappa] = (hm_, e_, U_, sch_get(ref_), ci_)   # rd13m stream
def sch_configure(a, ap):   # rd13m stream: main(), after rd13_configure
    if getattr(a, "nk_cache_host", "off") != "on":   # rd13m stream
        return   # rd13m stream
    if a.mix != "newton" or getattr(a, "nk_jvp", None) != "exact" or a.reference != "R2" or a.streams or getattr(a, "weyl_oracle", False) or getattr(a, "massless_ball", False):   # rd13m stream
        ap.error("rd13m: --nk-cache-host on needs --mix newton, --nk-jvp exact, --reference R2, --streams 0, and neither --weyl-oracle nor --massless-ball")   # rd13m stream
    if BK["dev"] is None or BK["dev"].type != "cuda":   # rd13m stream
        ap.error("rd13m: --nk-cache-host on needs a CUDA device (--gpu)")   # rd13m stream
    SCH["on"] = True; SCH["stream"] = BK["torch"].cuda.Stream(device=BK["dev"])   # rd13m stream
    globals()["dens"] = dens_v36s   # rd13m stream: the sources' dens calls read the densities a parked R2 carries
    print("rd13m: --nk-cache-host on (the Newton caches in pinned host memory, one channel on the device; the stress chunked by %d columns)" % SCH["width"], flush=True)   # rd13m stream
def sch_record():   # rd13m stream: the record's "rd13m" entry and its line
    r_ = dict(_sch_rec()); SCH["rec"] = None; r_["pinned"] = SCH["pinned"]   # rd13m stream
    print("  rd13m: %d parked (%.2f GB out in %.1f s), %d fetched (%.2f GB in, %.1f s; prefetch hits %d, misses %d), density misses %d, releases %d" % (   # rd13m stream
        r_["parked"], r_["bytes_out"] / 1e9, r_["seconds_out"], r_["fetched"], r_["bytes_in"] / 1e9, r_["seconds_in"], r_["prefetch_hits"], r_["prefetch_misses"], r_["dens_misses"], r_["releases"]), flush=True)   # rd13m stream
    return r_   # rd13m stream
NSR = {"on": False, "prev_sig": None, "rec": None, "prev_m": None, "other": None}   # version 36c (--nk-set resolve); 36d: prev_m, other
''')))

EDITS.append(("C evalG", blk(r'''
    n_start = ctr["n"]

    def evalG(N_a, v_a):
        eig = eigh_all(chs, N_a, v_a, bool(a.streams))
'''), blk(r'''
    n_start = ctr["n"]

    def evalG(N_a, v_a):
        if SCH["on"]:   # rd13m stream: the streamed evaluation (the docstring's (1))
            return evalG_stream_v36s(a, chs, st, cache0, sources_fn, do_solve, N_a, v_a)   # rd13m stream
        eig = eigh_all(chs, N_a, v_a, bool(a.streams))
''')))

EDITS.append(("D jvp", blk(r'''
            _, e, U, ref, cinfo = cache_[ch.kappa]
            mask = torch.as_tensor(np.asarray(cinfo["_mask"], bool), device=U.device)
'''), blk(r'''
            _, e, U, ref, cinfo = cache_[ch.kappa]
            U = sch_get(U, sch_next(cache_, chs, ch))   # rd13m stream: U back by exact copy (itself when not parked), the next channel's started
            mask = torch.as_tensor(np.asarray(cinfo["_mask"], bool), device=U.device)
''')))

EDITS.append(("D2 jvp check", blk(r'''
            if JV["ref_mismatch"] is None or JV["_chk"] < len(chs):
                dg = (ref.conj() * ref).real.sum(dim=1) - (Uo.conj() * Uo).real.sum(dim=1)
'''), blk(r'''
            if JV["ref_mismatch"] is None or JV["_chk"] < len(chs):
                ref = sch_get(ref)   # rd13m stream
                dg = (ref.conj() * ref).real.sum(dim=1) - (Uo.conj() * Uo).real.sum(dim=1)
''')))

EDITS.append(("E1 diagnostics", blk(r'''
                hm_, e, U, ref, cinfo = cache[ch.kappa]
                s_ = st[ch.kappa]
                masks[ch.kappa] = np.asarray(cinfo.pop("_mask"), bool)
'''), blk(r'''
                hm_, e, U, ref, cinfo = cache[ch.kappa]
                U = sch_get(U); ref = sch_get(ref)   # rd13m stream
                s_ = st[ch.kappa]
                masks[ch.kappa] = np.asarray(cinfo.pop("_mask"), bool)
''')))

EDITS.append(("E2 commutator", blk(r'''
                    comm[str(ch.kappa)] = commutator_norm(ch, N_app, v_app, cache[ch.kappa][3])
'''), blk(r'''
                    comm[str(ch.kappa)] = commutator_norm(ch, N_app, v_app, sch_get(cache[ch.kappa][3]))   # rd13m stream
''')))

EDITS.append(("E3 M1", blk(r'''
                s_ = st[ch.kappa]; H_old = clocked_host_sparse(ch, f_ref[0], f_ref[1]); Hr_old = dev_csr(H_old[:2 * cut, :]); ref_old = s_["ref"]; ref_new = cache[ch.kappa][3]
'''), blk(r'''
                s_ = st[ch.kappa]; H_old = clocked_host_sparse(ch, f_ref[0], f_ref[1]); Hr_old = dev_csr(H_old[:2 * cut, :]); ref_old = s_["ref"]; ref_new = sch_get(cache[ch.kappa][3])   # rd13m stream
''')))

EDITS.append(("E4 install", blk(r'''
        TM["M1_s"] += time.time() - t_m1
        for ch in chs:
            ci_ = cache[ch.kappa][4]
            st[ch.kappa]["ref"] = cache[ch.kappa][3]
            st[ch.kappa]["window"].append({"ref": cache[ch.kappa][3], "f": (N_app.copy(), v_app.copy()), "gap": ci_.get("gap_between_blocks"),
                                           "weyl_gap": ci_.get("weyl_gap", ci_.get("gap_between_blocks")), "mask": masks.get(ch.kappa)})
            st[ch.kappa]["window"] = st[ch.kappa]["window"][-1:]
'''), blk(r'''
        TM["M1_s"] += time.time() - t_m1
        sch_install(cache, chs)   # rd13m stream: the chosen evaluation's R2 back on the device (a no-op when off)
        for ch in chs:
            ci_ = cache[ch.kappa][4]
            st[ch.kappa]["ref"] = cache[ch.kappa][3]
            st[ch.kappa]["window"].append({"ref": cache[ch.kappa][3], "f": (N_app.copy(), v_app.copy()), "gap": ci_.get("gap_between_blocks"),
                                           "weyl_gap": ci_.get("weyl_gap", ci_.get("gap_between_blocks")), "mask": masks.get(ch.kappa)})
            st[ch.kappa]["window"] = st[ch.kappa]["window"][-1:]
        if SCH["on"]:   # rd13m stream: the previous record's R2 (held by the solve's reference cycles) and run()'s stale last-channel locals released
            U = ref = ref_old = ref_new = Ccols = None; gc.collect(); _sch_rec()["releases"] += 1   # rd13m stream
''')))

EDITS.append(("E5 simpson", blk(r'''
                _hm, e, U, ref, _ci = cache[ch.kappa]
                ph2 = torch.exp(-1j * e.to(torch.complex128) * (0.5 * a.tau))
'''), blk(r'''
                _hm, e, U, ref, _ci = cache[ch.kappa]
                U = sch_get(U)   # rd13m stream
                ph2 = torch.exp(-1j * e.to(torch.complex128) * (0.5 * a.tau))
''')))

EDITS.append(("E6 stress", blk(r'''
            pd_ = STR.stress_parts(cs, s_["dust"], s_["f"])
            pp_ = STR.stress_parts(cs, torch.cat([s_["sea"], ref], dim=1), np.concatenate([np.ones(s_["sea"].shape[1]), -np.ones(ref.shape[1])]))
'''), blk(r'''
            pd_ = sch_stress(cs, s_["dust"], s_["f"])   # rd13m stream
            pp_ = sch_stress(cs, torch.cat([s_["sea"], ref], dim=1), np.concatenate([np.ones(s_["sea"].shape[1]), -np.ones(ref.shape[1])]))   # rd13m stream
''')))

EDITS.append(("E7 propagation", blk(r'''
            _hm, e, U, ref, _ci = cache[ch.kappa]
            ph = torch.exp(-1j * e.to(torch.complex128) * a.tau)
'''), blk(r'''
            _hm, e, U, ref, _ci = cache[ch.kappa]
            U = sch_get(U)   # rd13m stream
            ph = torch.exp(-1j * e.to(torch.complex128) * a.tau)
''')))

EDITS.append(("E8 record", blk(r'''
            rec["rd13"] = rd13_record()   # rd13 banded
'''), blk(r'''
            rec["rd13"] = rd13_record()   # rd13 banded
        if SCH["on"]:   # rd13m stream
            rec["rd13m"] = sch_record()   # rd13m stream
''')))

EDITS.append(("F main", blk(r'''
    rd13_configure(a, ap)   # rd13 banded
    run(a)
'''), blk(r'''
    rd13_configure(a, ap)   # rd13 banded
    sch_configure(a, ap)   # rd13m stream
    run(a)
''')))

EDITS.append(("G option", blk(r'''
    ap.add_argument("--pair-workers", dest="pair_workers"'''), blk(r'''
    ap.add_argument("--nk-cache-host", dest="nk_cache_host", default="off", choices=["off", "on"], help="(rd13m) on: the Newton caches in pinned host memory, one channel at a time on the device (MEMLAW_8192_v1 Option 1); bitwise")   # rd13m stream
    ap.add_argument("--pair-workers", dest="pair_workers"''')))


def md5(b):
    return hashlib.md5(b).hexdigest()


def forward(text):
    for name, old, new in EDITS:
        c = text.count(old)
        if c != 1:
            raise SystemExit("anchor %s occurs %d times (must be 1); nothing written" % (name, c))
        text = text.replace(old, new)
    return text


def reverse(text):
    for name, old, new in reversed(EDITS):
        c = text.count(new)
        if c != 1:
            raise SystemExit("reverse %s: the NEW block occurs %d times (must be 1)" % (name, c))
        text = text.replace(new, old)
    return text


def marker_check():
    bad, n_marked = [], 0
    for name, old, new in EDITS:
        olds = set(old.splitlines())
        olds |= {l[3:] for l in old.splitlines() if l.startswith('"""')}
        for ln in new.splitlines():
            if ln in olds or ln.strip() == "":
                continue
            if MARK in ln:
                n_marked += 1
            else:
                bad.append((name, ln))
    return bad, n_marked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="engine_collapse_v36nq_rd13.py")
    ap.add_argument("--dst", default=None)
    ap.add_argument("--reverse", action="store_true")
    ap.add_argument("--expect", default=None)
    ap.add_argument("--expect-src-md5", default=SRC_MD5_PREFIX)
    ap.add_argument("--check-only", action="store_true")
    a = ap.parse_args()
    if a.dst is None:
        a.dst = os.path.join(os.path.dirname(os.path.abspath(a.src)), "engine_collapse_v36nq_rd13m.py" if not a.reverse else "engine_collapse_v36nq_rd13_restored.py")
    src = open(a.src, "rb").read()
    text = src.decode("utf-8")
    if a.reverse:
        out = reverse(text).encode("utf-8")
        if not a.check_only:
            if os.path.exists(a.dst):
                raise SystemExit("%s exists; nothing written (never overwritten)" % a.dst)
            open(a.dst, "wb").write(out)
        print("reverse: %s %s (md5 %s)" % ("would write" if a.check_only else "wrote", a.dst, md5(out)))
        if a.expect:
            ok = out == open(a.expect, "rb").read()
            print("restored equals %s byte for byte: %s" % (a.expect, ok))
            raise SystemExit(0 if ok else 1)
        return
    print("source %s: md5 %s, %d bytes" % (a.src, md5(src), len(src)))
    if a.expect_src_md5 and not md5(src).startswith(a.expect_src_md5):
        raise SystemExit("the source's md5 does not start with %s; nothing written" % a.expect_src_md5)
    if MARK in text:
        raise SystemExit("the source already carries %r; nothing written" % MARK)
    for name, old, new in EDITS:
        print("  anchor %-16s occurs %d time(s) in the input (%d -> %d lines)" % (name, text.count(old), len(old.splitlines()), len(new.splitlines())))
    bad, n_marked = marker_check()
    if bad:
        for name, ln in bad:
            print("  UNMARKED new line in %s: %r" % (name, ln))
        raise SystemExit("new lines without the marker")
    print("every new or changed line carries %r (%d marked lines)" % (MARK, n_marked))
    built = forward(text)
    out = built.encode("utf-8")
    back = reverse(built).encode("utf-8")
    print("reversible (NEW -> OLD restores the input byte for byte): %s" % (back == src))
    if back != src:
        raise SystemExit("the reverse pass does not restore the input; nothing written")
    try:
        compile(built, a.dst, "exec")
    except SyntaxError as e:
        raise SystemExit("the result does not compile (%s); nothing written" % e)
    if a.check_only:
        print("check only: %s would be %d bytes, md5 %s" % (a.dst, len(out), md5(out)))
        return
    if os.path.exists(a.dst):
        if open(a.dst, "rb").read() == out:
            print("%s exists and equals the build byte for byte (md5 %s); not rewritten" % (a.dst, md5(out)))
            return
        raise SystemExit("%s exists and differs from the build; nothing written (never overwritten)" % a.dst)
    open(a.dst, "wb").write(out)
    written = open(a.dst, "rb").read()
    if written != out or reverse(written.decode("utf-8")).encode("utf-8") != src:
        raise SystemExit("the written file does not reverse to the input")
    py_compile.compile(a.dst, doraise=True)
    n_lines = sum(1 for ln in built.splitlines() if MARK in ln)
    print("wrote %s: %d bytes, md5 %s; compiles; the reverse of the written file equals the source byte for byte" % (a.dst, len(out), md5(out)))
    print("check: grep -c '%s' %s  ->  %d" % (MARK, a.dst, n_lines))


if __name__ == "__main__":
    main()
