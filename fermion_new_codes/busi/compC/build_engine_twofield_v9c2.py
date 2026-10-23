#!/usr/bin/env python
"""build_engine_twofield_v9c2.py (busi lane, PERF tracker rows C-A1 to C-A5, owner order #299; reviewer B perf_c ranks 1, 3, 4, 5 mapped onto the
real sector-walk engine): engine_twofield_v9c2.py over engine_twofield_v9c1.py (md5 84938595abb52529abc909f5fdabcddc) in the lane's patch form:
exact replacements, each OLD occurring once in v9c1; every line of v9c2 not in v9c1 carries "[v9c2 perf]"; --reverse restores v9c1 byte for byte;
the source md5 is checked; an existing output is never overwritten.

Four options, each off by default; at the defaults every statement executed is v9c1's (the wrappers call v9c1's own expression first thing) and the
perf arguments are removed from the parsed namespace, so the json's args equal v9c1's.

  --perf-eig-cache-gib G (C-A1, reviewer B rank 1): exactly repeated endpoint eigendecompositions served from a bounded LRU of private copies (at most G
      GiB, device-memory guarded), keyed by the channel set and the exact bytes, dtype and shape of the fields (N, v). Heavy: eigh_all rebound by
      name (run, newton_record's and nk_stage_v32's evalG). Light: the two light evaluation sites (light_eval_held_v8, light_eval_v36). A hit
      returns fresh clones with the strides kept; the copy is taken before any continuation rotates U. The counters stay logical (eigh_n counts
      requests, as v9c1's records); the physical counts go to the per-record "v9c2 perf" line only. Where it hits on the real engine: in
      --reference-solve lagged the references are held through the solve, so G(x0) = x0 bitwise and the confirming step (confirm_v32) evaluates at
      the bytes of evaluation 1 (the S5 log: "kept residual 0.00e+00 ... confirming plain step's residual 0.00e+00"): 1 of the record's 2
      evaluations (10 heavy + 3 light eigh at c64s8) is a repeat. On continuation records, stage 1 starts at the previous record's accepted fields
      and stage 2 at stage 1's kept iterate, both evaluated before.
  --perf-dens-memo on (C-A4, reviewer B rank 4): dens(A, Phi) without weights memoized per record by the identity and version counter of A and Phi
      (weak references; an in-place change bumps the version), cleared at every record start; a copy is returned. The held references' densities
      recur in the lagged first iterate, every evaluation and the lag evaluation; the transported ones in the lag evaluation and the readouts.
  --perf-uhx-cache-gib G (C-A3, reviewer B rank 3): the heavy U^H sea and U^H dust of the readout (heavy E_inv) reused by the Simpson half step and the
      propagation (same U, same columns: identity and version checked, weak references), at most G GiB, cleared at every record start.
  --perf-writer on (C-A5, reviewer B rank 5): the checkpoint npz files of save_fields (s11, fields, handoff, spectra) written by one writer thread
      in v9c1's order, each to <name>.v9c2tmp.npz then os.replace'd (a reader sees the old file or the new one); the arrays are copied at submission;
      the next save_fields, and the end of the run, wait for the previous batch (at most one batch pending; errors re-raised). The json stays
      synchronous, as in v9c1. The orbital slices are untouched.

Not built (tracker rows C-A2, C-A6): see the PERF tracker.

Usage. python build_engine_twofield_v9c2.py engine_twofield_v9c1.py [engine_twofield_v9c2.py]
         --check: the build in memory with the marks check and the reverse round trip (nothing written)
         --reverse engine_twofield_v9c2.py restored.py [--against engine_twofield_v9c1.py]
"""
import argparse
import hashlib
import os
import sys

MARK = "[v9c2 perf]"
SOURCE_MD5 = "84938595abb52529abc909f5fdabcddc"

BLOCK = r'''# >>> v9c2 perf: PERF C-A1, C-A3, C-A4, C-A5 (owner order #299; reviewer B perf_c ranks 1, 3, 4, 5); every option off by default, and then every statement executed is v9c1's
import collections as _v9c2_collections
import copy as _v9c2_copy
import weakref as _v9c2_weakref
from concurrent.futures import ThreadPoolExecutor as _V9c2Pool
V9C2_VERSION = "engine_twofield_v9c2"
V9C2_DEFAULTS = {"perf_eig_cache_gib": 0.0, "perf_dens_memo": "off", "perf_uhx_cache_gib": 0.0, "perf_writer": "off"}
V9C2 = {"eig_cap": 0, "dens": False, "uhx_cap": 0, "writer": None, "on": False}
_V9C2_RAW = {"eigh_all": eigh_all, "dens": dens}
_V9C2_EIG = _v9c2_collections.OrderedDict()
_V9C2_DENS = _v9c2_collections.OrderedDict()
_V9C2_UHX = {}
_V9C2_BYTES = {"eig": 0, "uhx": 0}
_V9C2_PENDING = []
_V9C2_STATS = {}
_V9C2_DENS_MAX = 4096


class _V9c2NoCache(Exception):
    pass


def _v9c2_inc(k, n=1):
    _V9C2_STATS[k] = _V9C2_STATS.get(k, 0) + n


def v9c2_add_arguments(ap):
    ap.add_argument("--perf-eig-cache-gib", type=float, default=0.0, help="v9c2 perf (C-A1): exactly repeated endpoint eigendecompositions from a bounded LRU of private copies, at most this many GiB (0: off, v9c1)")
    ap.add_argument("--perf-dens-memo", default="off", choices=["off", "on"], help="v9c2 perf (C-A4): dens(A, Phi) memoized within a record by identity and version (off: v9c1)")
    ap.add_argument("--perf-uhx-cache-gib", type=float, default=0.0, help="v9c2 perf (C-A3): the heavy readout's U^H sea and U^H dust reused by the Simpson half step and the propagation, at most this many GiB (0: off, v9c1)")
    ap.add_argument("--perf-writer", default="off", choices=["off", "on"], help="v9c2 perf (C-A5): save_fields' npz files written by one bounded writer thread, atomically replaced (off: v9c1)")


def v9c2_set(a):
    vals = {k: getattr(a, k, d) for k, d in V9C2_DEFAULTS.items()}
    if not (vals["perf_eig_cache_gib"] >= 0.0 and vals["perf_uhx_cache_gib"] >= 0.0):
        raise SystemExit("v9c2 perf: --perf-eig-cache-gib and --perf-uhx-cache-gib are nonnegative")
    V9C2["eig_cap"] = int(vals["perf_eig_cache_gib"] * 1024 ** 3)
    V9C2["dens"] = vals["perf_dens_memo"] == "on"
    V9C2["uhx_cap"] = int(vals["perf_uhx_cache_gib"] * 1024 ** 3)
    for k, d in V9C2_DEFAULTS.items():
        if hasattr(a, k) and getattr(a, k) == d:
            delattr(a, k)
    if V9C2["eig_cap"]:
        globals()["eigh_all"] = v9c2_eigh_all
    if V9C2["dens"]:
        globals()["dens"] = v9c2_dens
    if vals["perf_writer"] == "on":
        V9C2["writer"] = _V9c2Pool(max_workers=1)
    V9C2["on"] = bool(V9C2["eig_cap"] or V9C2["dens"] or V9C2["uhx_cap"] or V9C2["writer"] is not None)
    if V9C2["on"]:
        print("v9c2 perf: %s; eig cache %s GiB, dens memo %s, U^H cache %s GiB, writer %s (bitwise options; the counters in the records stay logical, the physical counts on the per-record v9c2 perf line)" % (
            V9C2_VERSION, vals["perf_eig_cache_gib"], vals["perf_dens_memo"], vals["perf_uhx_cache_gib"], vals["perf_writer"]), flush=True)


def _v9c2_akey(x):
    x = np.asarray(x)
    if x.dtype.hasobject:
        raise _V9c2NoCache()
    return (x.dtype.str, x.shape, np.ascontiguousarray(x).tobytes())


def _v9c2_clone(x):
    torch = BK["torch"]
    if isinstance(x, torch.Tensor):
        y = x.clone(memory_format=torch.preserve_format)
        if y.stride() != x.stride() or y.dtype != x.dtype or y.device != x.device:
            raise _V9c2NoCache()
        return y
    if isinstance(x, np.ndarray):
        return x.copy(order="K")
    if isinstance(x, tuple):
        return tuple(_v9c2_clone(q) for q in x)
    if isinstance(x, dict):
        return {k: _v9c2_clone(q) for k, q in x.items()}
    raise _V9c2NoCache()


def _v9c2_nbytes(x):
    torch = BK["torch"]
    if isinstance(x, torch.Tensor):
        return x.numel() * x.element_size()
    if isinstance(x, np.ndarray):
        return int(x.nbytes)
    if isinstance(x, (tuple, list)):
        return sum(_v9c2_nbytes(q) for q in x)
    if isinstance(x, dict):
        return sum(_v9c2_nbytes(q) for q in x.values())
    return 0


def _v9c2_room(n):
    torch = BK["torch"]; dev = BK["dev"]
    if dev is None or getattr(dev, "type", None) != "cuda":
        return True
    free, _tot = torch.cuda.mem_get_info(dev)
    reusable = torch.cuda.memory_reserved(dev) - torch.cuda.memory_allocated(dev)
    return free + reusable > n + 2 * 1024 ** 3


def _v9c2_spectral(key, compute, n_matrices, account=None):
    hit = _V9C2_EIG.get(key) if key is not None else None
    if hit is not None:
        _V9C2_EIG.move_to_end(key)
        t_ = time.time()
        out = _v9c2_clone(hit[0])
        sync_dev()
        dt = time.time() - t_
        if account is not None:
            account(n_matrices, dt)
        _v9c2_inc("eig_hit", n_matrices)
        return out
    out = compute()
    _v9c2_inc("eig_computed", n_matrices)
    if key is None:
        return out
    try:
        size = _v9c2_nbytes(out)
        if size > V9C2["eig_cap"]:
            _v9c2_inc("eig_over_cap")
            return out
        while _V9C2_EIG and _V9C2_BYTES["eig"] + size > V9C2["eig_cap"]:
            _k, (_o, old) = _V9C2_EIG.popitem(last=False)
            _V9C2_BYTES["eig"] -= old
            _v9c2_inc("eig_evicted")
        if not _v9c2_room(size):
            _v9c2_inc("eig_memory_guard")
            return out
        saved = _v9c2_clone(out)
        sync_dev()
        _V9C2_EIG[key] = (saved, size)
        _V9C2_BYTES["eig"] += size
    except _V9c2NoCache:
        _v9c2_inc("eig_uncacheable")
    return out


def v9c2_eigh_all(chs, N, v, streams=False):
    if not V9C2["eig_cap"]:
        return _V9C2_RAW["eigh_all"](chs, N, v, streams)
    try:
        key = ("heavy", tuple(ch.kappa for ch in chs), bool(streams), _v9c2_akey(N), _v9c2_akey(v))
    except _V9c2NoCache:
        key = None

    def account(n_, dt_):
        TM["eigh_n"] = TM.get("eigh_n", 0) + n_
        TM["eigh_s"] = TM.get("eigh_s", 0.0) + dt_
    return _v9c2_spectral(key, lambda: _V9C2_RAW["eigh_all"](chs, N, v, streams), len(chs), account)


def v9c2_eig_light(chL, N, v):
    if not V9C2["eig_cap"]:
        return eigT(clocked_gpu(chL, N, v))
    try:
        key = ("light", chL.key, _v9c2_akey(N), _v9c2_akey(v))
    except _V9c2NoCache:
        key = None
    return _v9c2_spectral(key, lambda: eigT(clocked_gpu(chL, N, v)), 1, None)


def v9c2_dens(A, Phi, f=None):
    raw = _V9C2_RAW["dens"]
    if f is not None or not V9C2["dens"]:
        return raw(A, Phi, f)
    try:
        key = (id(A), id(Phi), Phi._version, Phi.data_ptr(), tuple(Phi.shape), tuple(Phi.stride()), str(Phi.dtype), str(Phi.device))
    except (AttributeError, RuntimeError, TypeError):
        return raw(A, Phi, f)
    q = _V9C2_DENS.get(key)
    if q is not None and q[0]() is A and q[1]() is Phi and q[2] == Phi._version:
        _V9C2_DENS.move_to_end(key)
        _v9c2_inc("dens_hit")
        return q[3].copy()
    d = raw(A, Phi, f)
    _v9c2_inc("dens_computed")
    try:
        _V9C2_DENS[key] = (_v9c2_weakref.ref(A), _v9c2_weakref.ref(Phi), Phi._version, np.array(d, copy=True))
        while len(_V9C2_DENS) > _V9C2_DENS_MAX:
            _V9C2_DENS.popitem(last=False)
    except TypeError:
        pass
    return d


def v9c2_uhx(U, X):
    if not V9C2["uhx_cap"]:
        return U.conj().T @ X
    key = (id(U), id(X))
    q = _V9C2_UHX.get(key)
    if q is not None and q[0]() is U and q[1]() is X and q[2] == U._version and q[3] == X._version:
        _v9c2_inc("uhx_hit")
        return q[4]
    A = U.conj().T @ X
    _v9c2_inc("uhx_computed")
    size = A.numel() * A.element_size()
    if _V9C2_BYTES["uhx"] + size <= V9C2["uhx_cap"] and _v9c2_room(size):
        _V9C2_UHX[key] = (_v9c2_weakref.ref(U), _v9c2_weakref.ref(X), U._version, X._version, A)
        _V9C2_BYTES["uhx"] += size
    else:
        _v9c2_inc("uhx_over_cap")
    return A


def v9c2_record_reset(trs):
    if V9C2["on"] and _V9C2_STATS:
        print("  v9c2 perf (the record before T/rs %.4f): %s; eig cache %.2f GiB in %d entries" % (
            trs, ", ".join("%s %s" % (k, (("%.2f" % v) if isinstance(v, float) else v)) for k, v in sorted(_V9C2_STATS.items())), _V9C2_BYTES["eig"] / 1024.0 ** 3, len(_V9C2_EIG)), flush=True)
    _V9C2_STATS.clear()
    _V9C2_DENS.clear()
    _V9C2_UHX.clear(); _V9C2_BYTES["uhx"] = 0


def _v9c2_snapshot(v):
    if isinstance(v, np.ndarray):
        return _v9c2_copy.deepcopy(v) if v.dtype.hasobject else np.array(v, copy=True)
    return v


def v9c2_save(_v9c2_fn, _v9c2_path, **kw):
    pool = V9C2["writer"]
    if pool is None:
        return _v9c2_fn(_v9c2_path, **kw)
    t_ = time.time()
    kw = {k: _v9c2_snapshot(v) for k, v in kw.items()}
    target = _v9c2_path if str(_v9c2_path).endswith(".npz") else str(_v9c2_path) + ".npz"
    tmp = target[:-4] + ".v9c2tmp.npz"

    def job():
        _v9c2_fn(tmp, **kw)
        os.replace(tmp, target)
    _V9C2_PENDING.append(pool.submit(job))
    _v9c2_inc("writer_submitted")
    _v9c2_inc("writer_snapshot_s", time.time() - t_)
    return None


def v9c2_flush():
    if not _V9C2_PENDING:
        return
    t_ = time.time()
    try:
        while _V9C2_PENDING:
            _V9C2_PENDING.pop(0).result()
    finally:
        _v9c2_inc("writer_wait_s", time.time() - t_)


def v9c2_close():
    try:
        v9c2_flush()
    finally:
        if V9C2["writer"] is not None:
            V9C2["writer"].shutdown(wait=True)
            V9C2["writer"] = None
        if V9C2["on"] and _V9C2_STATS:
            print("  v9c2 perf (the last record): %s" % ", ".join("%s %s" % (k, (("%.2f" % v) if isinstance(v, float) else v)) for k, v in sorted(_V9C2_STATS.items())), flush=True)
# <<< v9c2 perf'''

MAIN_SEP = "# ------------------------------------------------------------------------------------------------------- version 35's main, with v36 args\ndef main():\n"
L1947 = "        e, U = eigT(clocked_gpu(chL, N_a, v_a))" + " " * 58 + "# v8 lagged (the ruling's v4)\n"


class BuildStop(Exception):
    pass


class Rep(object):
    def __init__(self, name, old, new):
        self.name = name; self.old = old; self.new = new


def mark(text):
    out = []
    for line in text.split("\n"):
        out.append(line if (line.strip() == "" or MARK in line) else line + "  # " + MARK)
    return "\n".join(out)


def replacements():
    M = "  # " + MARK
    return [
        Rep("P1 block", "\n\n" + MAIN_SEP, "\n\n" + mark(BLOCK) + "\n\n\n" + MAIN_SEP),
        Rep("P2 args", "    v9c_add_arguments(ap)  # v9c prim\n", "    v9c_add_arguments(ap)  # v9c prim\n    v9c2_add_arguments(ap)" + M + "\n"),
        Rep("P3 main", "    C5.backend(a.gpu)\n    run(a)\n",
            "    C5.backend(a.gpu)\n    v9c2_set(a)" + M + "\n    try:" + M + "\n        run(a)" + M + "\n    finally:" + M + "\n        v9c2_close()" + M + "\n"),
        Rep("P4 record reset", '        TM["propagation_prev_s"] = last_prop_s\n', '        TM["propagation_prev_s"] = last_prop_s\n        v9c2_record_reset(T / rs)' + M + "\n"),
        Rep("A1 light held", L1947, "        e, U = v9c2_eig_light(chL, N_a, v_a)  # v8 lagged (the ruling's v4); " + MARK + "\n"),
        Rep("A1 light eval", "        e, U = eigT(clocked_gpu(chL, N_a, v_a))\n        sync_dev()\n",
            "        e, U = v9c2_eig_light(chL, N_a, v_a)" + M + "\n        sync_dev()\n"),
        Rep("A3 readout sea", '                nocc_s = ((U.conj().T @ s_["sea"]).abs() ** 2).sum(dim=1)\n',
            '                nocc_s = ((v9c2_uhx(U, s_["sea"])).abs() ** 2).sum(dim=1)' + M + "\n"),
        Rep("A3 readout dust", '                    nocc_d = ((U.conj().T @ s_["dust"]).abs() ** 2 * T_(s_["f"])[None, :].real.to(torch.float64)).sum(dim=1)\n',
            '                    nocc_d = ((v9c2_uhx(U, s_["dust"])).abs() ** 2 * T_(s_["f"])[None, :].real.to(torch.float64)).sum(dim=1)' + M + "\n"),
        Rep("A3 Simpson", "                halfp = lambda X: U @ (ph2[:, None] * (U.conj().T @ X))\n",
            "                halfp = lambda X: U @ (ph2[:, None] * v9c2_uhx(U, X))" + M + "\n"),
        Rep("A3 propagation", "            prop = lambda X: U @ (ph[:, None] * (U.conj().T @ X))\n",
            "            prop = lambda X: U @ (ph[:, None] * v9c2_uhx(U, X))" + M + "\n"),
        Rep("A5 flush", "    def save_fields():\n", "    def save_fields():\n        v9c2_flush()" + M + "\n"),
        Rep("A5 s11", '            np.savez_compressed(os.path.join(a.out, "collapse_%s_s11.npz" % a.tag), **{k_: np.array(v_) for k_, v_ in s11_store.items()})\n',
            '            v9c2_save(np.savez_compressed, os.path.join(a.out, "collapse_%s_s11.npz" % a.tag), **{k_: np.array(v_) for k_, v_ in s11_store.items()})' + M + "\n"),
        Rep("A5 fields", '        np.savez_compressed(os.path.join(a.out, "collapse_%s_fields.npz" % a.tag), R=R, T=np.array(fields["T"]), N=np.array(fields["N"]), v=np.array(fields["v"]),\n',
            '        v9c2_save(np.savez_compressed, os.path.join(a.out, "collapse_%s_fields.npz" % a.tag), R=R, T=np.array(fields["T"]), N=np.array(fields["N"]), v=np.array(fields["v"]),' + M + "\n"),
        Rep("A5 handoff", '            np.savez_compressed(os.path.join(a.out, "collapse_%s_handoff.npz" % a.tag), R=R, r_nodes=r_nodes_fixed, g=a.g, ell=a.ell, m=a.m, tau=a.tau,\n',
            '            v9c2_save(np.savez_compressed, os.path.join(a.out, "collapse_%s_handoff.npz" % a.tag), R=R, r_nodes=r_nodes_fixed, g=a.g, ell=a.ell, m=a.m, tau=a.tau,' + M + "\n"),
        Rep("A5 spectra", '            np.savez(os.path.join(a.out, "collapse_%s_spectra.npz" % a.tag), **{k: np.array(v_) for k, v_ in spectra.items()})\n',
            '            v9c2_save(np.savez, os.path.join(a.out, "collapse_%s_spectra.npz" % a.tag), **{k: np.array(v_) for k, v_ in spectra.items()})' + M + "\n"),
    ]


def precheck(text, reps, reverse=False):
    return [(r, text.count(r.new if reverse else r.old)) for r in reps if text.count(r.new if reverse else r.old) != 1]


def apply(text, reps, reverse=False):
    bad = precheck(text, reps, reverse)
    if bad:
        raise BuildStop("; ".join("%s: %s occurs %d time(s), not once" % (r.name, "NEW" if reverse else "OLD", n) for r, n in bad))
    for r in (reversed(reps) if reverse else reps):
        old, new = (r.new, r.old) if reverse else (r.old, r.new)
        text = text.replace(old, new, 1)
    return text


def marks_check(src, out):
    old = set(src.split("\n")); bad = []
    for q, line in enumerate(out.split("\n"), 1):
        if line.strip() == "" or line in old:
            continue
        if MARK not in line:
            bad.append((q, line))
    return bad


def md5(text):
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def read(path):
    with open(path, "rb") as f:
        return f.read().decode("utf-8")


def write_new(path, text):
    if os.path.exists(path):
        raise BuildStop("the output %s exists; it is never overwritten" % path)
    with open(path, "xb") as f:
        f.write(text.encode("utf-8"))


def main(argv=None):
    ap = argparse.ArgumentParser(description="engine_twofield_v9c2 over engine_twofield_v9c1 (exact replacements; fails closed)")
    ap.add_argument("source"); ap.add_argument("out", nargs="?", default=None)
    ap.add_argument("--check", action="store_true"); ap.add_argument("--reverse", action="store_true"); ap.add_argument("--against", default=None)
    a = ap.parse_args(argv)
    src = read(a.source); reps = replacements()
    try:
        if a.reverse:
            back = apply(src, reps, reverse=True)
            out = a.out or (os.path.splitext(a.source)[0] + ".restored.py")
            write_new(out, back)
            print("reverse: %s (md5 %s) -> %s (md5 %s)" % (a.source, md5(src), out, md5(back)))
            if a.against:
                ok = read(a.against) == back
                print("against %s: %s" % (a.against, "BYTE FOR BYTE" if ok else "DIFFERS"))
                return 0 if ok else 2
            return 0
        if md5(src) != SOURCE_MD5:
            print("STOP: the source md5 is %s, not v9c1's %s" % (md5(src), SOURCE_MD5)); return 2
        out_text = apply(src, reps)
    except BuildStop as stop:
        print("STOP: %s; nothing written" % stop); return 2
    marks = marks_check(src, out_text)
    if marks:
        print("STOP: %d new line(s) lack %r:" % (len(marks), MARK))
        for q, line in marks[:20]:
            print("  %6d: %s" % (q, line[:160]))
        return 2
    if apply(out_text, reps, reverse=True) != src:
        print("STOP: the reverse does not restore v9c1 byte for byte"); return 2
    try:
        compile(out_text, "engine_twofield_v9c2.py", "exec")
    except SyntaxError as exc:
        print("STOP: the built text does not compile: %s" % exc); return 2
    n_marked = sum(1 for line in out_text.split("\n") if MARK in line)
    print("build: %s (md5 %s) -> v9c2 (md5 %s); %d replacements applied; %d lines carry %r; marks ok; reverse byte for byte; compiled" % (
        a.source, md5(src), md5(out_text), len(reps), n_marked, MARK))
    if a.check:
        print("CHECK PASS (nothing written)"); return 0
    out = a.out or os.path.join(os.path.dirname(os.path.abspath(a.source)), "engine_twofield_v9c2.py")
    try:
        write_new(out, out_text)
    except BuildStop as stop:
        print("STOP: %s" % stop); return 2
    print("written: %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
