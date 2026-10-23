#!/usr/bin/env python
"""build_engine_twofield_v9c7.py (busi lane, PERF tracker row RD-1c; reviewer A algorithm redesign final4/PERF/REVIEWA_algorithm_redesign_v1.md
sec. 1, RD-1's module engine_banded_eigh_v1 73ee46fa, ADOPTED for the v36nq lines): engine_twofield_v9c7.py over engine_twofield_v9c6.py
(md5 89b852eb0bc4ea78bbe1f368aec58a63) in the lane's patch form: exact replacements, each OLD occurring once in v9c6; every line of v9c7
not in v9c6 carries "[v9c7 perf]"; --reverse restores v9c6 byte for byte; the source md5 is checked; an existing output is never overwritten.

The option (consumed by an argparse action into a module dict; the parsed namespace and the json args stay v9c6's):

  --heavy-eigh dense|banded (default dense: every statement executed is v9c6's, apart from one dict test per heavy call site and one call per
      record that returns at once).
      banded: the HEAVY (massive) channels' eigendecompositions by RD-1's engine_banded_eigh_v1.eigT_banded (Sylvester multisection and
      inverse iteration on the band, the particle-hole lift, the Rayleigh-Ritz cluster pass and the Lowdin polish; its guards G1 to G5
      send a call to the dense torch.linalg.eigh, which is engine_collapse_v5.eigT's code, so a zero mode (G4) or an off-band entry (G1)
      keeps the dense route per call). Only these heavy call sites are routed:
        (1) the heavy eigh_all (run()'s plain iterations, newton_record's and nk_stage_v32's evaluations), through v9c2's cache when it is on
            (_V9C2_RAW["eigh_all"]) or the global name when it is off; the sequential path (as RD-1: --streams' threads would call
            torch.linalg.eigh directly; every C64 line runs --streams 0);
        (2) HeavyPathV9.eig (the heavy sector walk's probes, the primitive's heavy spectra in [0, 1]);
        (3) V9cSpectra._extend's heavy branch (the primitive's chord extension s outside [0, 1]);
        (4) continue_ref_v9's continue_ref_v34 call (the heavy continuation's intermediate eig_at configurations: continue_ref_v34's own two
            statements with the banded eig_at; its source is checked at the option's start, fail closed);
        (5) run()'s fresh-start heavy eigendecomposition at T = 0 (not on the resume path).
      Not routed (unchanged): every light eigh (SYM-C1's sector eigh, the light continuation's eig_at, the light walk's probes, the light
      chord extension), engine_collapse_v36's own heavy eighs inside resume_slice_v34 / resume_state_v25 / prepare_dust /
      ball_fixed_point_v33 (one per channel per process start, or the preparation), --weyl-oracle's oracle_v34 (a diagnostic), and the
      host eigvals of the sector walk (engine_ops_v34_cert.eigvals_host). A light channel reaching a routed site keeps the dense route
      (counted "light_guard"; never expected).
      At the option's start: RD-4's CUDA 13 driver workaround (_rd4_patch_driver; nothing changes on a driver whose base and _v2 symbols
      agree), the module imported by name with its md5 checked against the adopted 73ee46fa (else SystemExit), its JIT warm-up, and a
      kernel self-check on a 64-site massive chain built by the engine's own Chan and clocked_gpu (route banded with the lift, eigenvalues
      within 1e-12 ||h||_G of torch.linalg.eigh, residuals within 1e-11 ||h||_G, the sea projector within 1e-10; else SystemExit).
      Per record one "v9c7 perf" line (the module's counters: calls, banded, no-lift, dense fallbacks with their reasons, seconds), and the
      run's totals at the end. The records and npz files carry no new key.

Usage. python build_engine_twofield_v9c7.py engine_twofield_v9c6.py [engine_twofield_v9c7.py]
         --check: the build in memory with the marks check and the reverse round trip (nothing written)
         --reverse engine_twofield_v9c7.py restored.py [--against engine_twofield_v9c6.py]
"""
import argparse
import hashlib
import os
import sys

MARK = "[v9c7 perf]"
SOURCE_MD5 = "89b852eb0bc4ea78bbe1f368aec58a63"   # engine_twofield_v9c6.py

BLOCK = r'''# >>> v9c7 perf: PERF RD-1c (RD-1's engine_banded_eigh_v1 on the heavy field only): --heavy-eigh dense (the default) runs v9c6's statements
import argparse as _rd1c_argparse
import hashlib as _rd1c_hashlib
import importlib as _rd1c_importlib
import inspect as _rd1c_inspect

V9C7_VERSION = "engine_twofield_v9c7"
_RD1C = {"eigh": "dense", "on": False, "mod": None, "md5": None}
_RD1C_MODULE = "engine_banded_eigh_v1"
_RD1C_MODULE_MD5 = "73ee46facc09b8cb77f095b2803a5c8e"
_RD1C_RUN = {"light_guard": 0}
_RD1C_TOT = {}
_RD1C_CONT_SRC = ('    if getattr(a, "weyl_certified", None):\n'
                  '        return CERT.continue_sea_v34(ch, window, f_cur, e, U, n_sea, a.tau, a.c_LZ, eig_at, clocked_host_sparse, a.weyl_certified, a.weyl_refine_max)\n'
                  '    return OPS3.continue_sea_v3(ch, window, f_cur, e, U, n_sea, a.tau, a.c_LZ, eig_at, clocked_host_sparse)\n')


class _Rd1cAction(_rd1c_argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        _RD1C[self.dest] = values
        # Deliberately do not modify namespace: output args stay v9c6's.


def _rd1c_add_arguments(ap):
    ap.add_argument("--heavy-eigh", dest="eigh", choices=["dense", "banded"], action=_Rd1cAction, default=_rd1c_argparse.SUPPRESS,
                    help="RD-1c: the heavy (massive) channels' eigendecompositions: dense (the default, v9c6's torch.linalg.eigh) or banded (RD-1's engine_banded_eigh_v1 on the GPU, its guards sending a call to the dense route); the light route and the zero-mode cases keep their routes")


def _rd1c_log(msg):
    print("  v9c7 perf: %s" % msg, flush=True)


def _rd1c_eigT_t(hmT, ch=None):
    # engine_collapse_v36.eigT_t's statements (the eigh timing in TM) with the banded entry point; a light channel keeps the dense route
    if ch is not None and isinstance(ch, LightChan):
        _RD1C_RUN["light_guard"] += 1
        return V35.eigT_t(hmT)
    t_ = time.time()
    e, U = _RD1C["mod"].eigT_banded(hmT, log=_rd1c_log)
    sync_dev()
    TM["eigh_s"] = TM.get("eigh_s", 0.0) + time.time() - t_; TM["eigh_n"] = TM.get("eigh_n", 0) + 1
    return e, U


def _rd1c_eig_at(ch, N, v):
    # engine_collapse_v36.eig_at for a heavy channel
    return _rd1c_eigT_t(clocked_gpu(ch, N, v), ch)


def _rd1c_eigh_all(chs, N, v, streams=False):
    # engine_collapse_v36.eigh_all's sequential path for the heavy channels (the streams' threads would call torch.linalg.eigh directly)
    out = {}
    for ch in chs:
        out[ch.kappa] = _rd1c_eigT_t(clocked_gpu(ch, N, v), ch)
    return out


def _rd1c_continue_ref_v34(a, ch, window, f_cur, e, U, n_sea):
    # engine_collapse_v36.continue_ref_v34's two statements (checked in rd1c_set) with the heavy banded eig_at
    if getattr(a, "weyl_certified", None):
        return CERT.continue_sea_v34(ch, window, f_cur, e, U, n_sea, a.tau, a.c_LZ, _rd1c_eig_at, clocked_host_sparse, a.weyl_certified, a.weyl_refine_max)
    return OPS3.continue_sea_v3(ch, window, f_cur, e, U, n_sea, a.tau, a.c_LZ, _rd1c_eig_at, clocked_host_sparse)


def _rd1c_selfcheck(mod, m):
    # the kernels on a 64-site massive chain built by the engine's own Chan and clocked_gpu, against torch.linalg.eigh
    torch = BK["torch"]
    ch = Chan(1, float(m), 64, False)
    r = np.arange(64, dtype=float)
    N = 1.0 - 0.3 * np.exp(-((r - 20.0) / 8.0) ** 2); v = 0.2 * np.exp(-((r - 24.0) / 6.0) ** 2)
    hmT = clocked_gpu(ch, N, v)
    s0 = mod.stats_snapshot(reset=True)
    e_b, U_b = mod.eigT_banded(hmT)
    s1 = mod.stats_snapshot(reset=True)
    e_d, U_d = torch.linalg.eigh(hmT); e_d = e_d.real
    G = float(torch.linalg.norm(hmT, ord=float("inf")))
    if not (s1["banded"] == 1 and s1["banded_all"] == 0 and s1["dense_fallback"] == 0):
        raise RuntimeError("the self-check matrix did not take the lifted banded route (%s)" % s1["fallback_why"])
    de = float((e_b - e_d).abs().max()) / G
    res = float(torch.linalg.norm(hmT @ U_b - U_b * e_b[None, :], dim=0).max()) / G
    nb, nd = e_b < 0, e_d < 0
    P = U_b[:, nb] @ U_b[:, nb].conj().T - U_d[:, nd] @ U_d[:, nd].conj().T
    dp = float(torch.linalg.norm(P))
    if not (de <= 1e-12 and res <= 1e-11 and dp <= 1e-10):
        raise RuntimeError("eigenvalues %.2e, residual %.2e (relative to ||h||_G), sea projector %.2e" % (de, res, dp))
    del s0
    return de, res, dp


def rd1c_set(a):
    if _RD1C["eigh"] != "banded":
        return
    dev = BK.get("dev")
    if dev is None or getattr(dev, "type", "cpu") != "cuda":
        raise SystemExit("v9c7 perf: --heavy-eigh banded runs on the GPU (numba.cuda): a --gpu run is required")
    src = _rd1c_inspect.getsource(V35.continue_ref_v34)
    if not src.endswith(_RD1C_CONT_SRC):
        raise SystemExit("v9c7 perf: engine_collapse_v36.continue_ref_v34 is not the text the banded heavy continuation reproduces; nothing run")
    t_ = time.time()
    _rd4_patch_driver()
    try:
        mod = _rd1c_importlib.import_module(_RD1C_MODULE)
        with open(mod.__file__, "rb") as f_:
            md5 = _rd1c_hashlib.md5(f_.read()).hexdigest()
    except Exception as exc:
        raise SystemExit("v9c7 perf: --heavy-eigh banded needs %s (numba.cuda): %s: %s" % (_RD1C_MODULE, type(exc).__name__, exc))
    if md5 != _RD1C_MODULE_MD5:
        raise SystemExit("v9c7 perf: %s at %s has md5 %s, not the adopted %s; nothing run" % (_RD1C_MODULE, mod.__file__, md5, _RD1C_MODULE_MD5))
    try:
        tw = mod.warmup(dev)
        de, res, dp = _rd1c_selfcheck(mod, 0.2)
    except Exception as exc:
        raise SystemExit("v9c7 perf: the banded eigensolver failed its start check (%s: %s)" % (type(exc).__name__, exc))
    mod.stats_snapshot(reset=True)
    _RD1C.update({"on": True, "mod": mod, "md5": md5})
    _V9C2_RAW["eigh_all"] = _rd1c_eigh_all
    if globals().get("eigh_all") is not v9c2_eigh_all:
        globals()["eigh_all"] = _rd1c_eigh_all
    print("v9c7 perf: %s; --heavy-eigh banded (%s %s, md5 %s, on %s; driver patch: %s); warm-up %.1f s, start check e %.1e, residual %.1e (of ||h||_G), sea projector %.1e, %.1f s in all; the heavy eigh_all by %s; the light route unchanged%s" % (
        V9C7_VERSION, _RD1C_MODULE, mod.__file__, md5, mod.STATS.get("device"), mod.STATS.get("devnum_patch"), tw, de, res, dp, time.time() - t_,
        "v9c2's cache over the banded route" if globals().get("eigh_all") is v9c2_eigh_all else "the banded route",
        "; --streams: the heavy eigh_all takes its sequential path" if getattr(a, "streams", 0) else ""), flush=True)


def _rd1c_line(snap, light_guard):
    why = dict(snap.get("fallback_why") or {})
    return "heavy eigh calls %d, banded %d (no lift %d), dense fallbacks %d%s, light guard %d; %.2f s (guards %.2f, kernels %.2f, post %.2f); max residual %.1e ||h||_G, max neighbour overlap %.1e, clusters %d" % (
        int(snap.get("calls", 0)), int(snap.get("banded", 0)), int(snap.get("banded_all", 0)), int(snap.get("dense_fallback", 0)),
        (" %s" % why) if why else "", int(light_guard), float(snap.get("seconds", 0.0)), float(snap.get("guard_seconds", 0.0)),
        float(snap.get("kernel_seconds", 0.0)), float(snap.get("post_seconds", 0.0)), float(snap.get("max_residual_rel", 0.0)),
        float(snap.get("max_overlap", 0.0)), int(snap.get("clusters", 0)))


def _rd1c_take():
    snap = _RD1C["mod"].stats_snapshot(reset=True)
    lg = int(_RD1C_RUN["light_guard"]); _RD1C_RUN["light_guard"] = 0
    _RD1C_TOT["light_guard"] = _RD1C_TOT.get("light_guard", 0) + lg
    for k, v in snap.items():
        if isinstance(v, bool) or not isinstance(v, (int, float)) or k in ("warmup_s", "isolated_min"):
            continue
        if k.startswith("max_"):
            _RD1C_TOT[k] = max(_RD1C_TOT.get(k, 0.0), v)
        else:
            _RD1C_TOT[k] = _RD1C_TOT.get(k, 0) + v
    for k in ("fallback_why", "st_fallback_why"):
        for w, c in (snap.get(k) or {}).items():
            _RD1C_TOT.setdefault(k, {})[w] = _RD1C_TOT.setdefault(k, {}).get(w, 0) + c
    return snap, lg


def rd1c_record_reset(trs):
    if not _RD1C["on"]:
        return
    snap, lg = _rd1c_take()
    print("  v9c7 perf (the record before T/rs %.4f): %s" % (trs, _rd1c_line(snap, lg)), flush=True)


def rd1c_close():
    if not _RD1C["on"]:
        return
    snap, lg = _rd1c_take()
    print("  v9c7 perf (the last record): %s" % _rd1c_line(snap, lg), flush=True)
    print("  v9c7 perf (the run): %s" % _rd1c_line(_RD1C_TOT, _RD1C_TOT.get("light_guard", 0)), flush=True)
# <<< v9c7 perf'''

OLD_BLOCK_ANCHOR = "# <<< v9c6 perf  # [v9c6 perf]\n"

OLD_HEAVY_PATH = '        e, U = eig_at(self.ch, N, v); self.eigh_n += 1                                                   # v9 sector\n'
HEAVY_PATH_HEAD = '        if _RD1C["on"]:\n            e, U = _rd1c_eig_at(self.ch, N, v); self.eigh_n += 1\n        else:\n'

OLD_EXTEND_HEAVY = '        e, U = eig_at_(ch, N, v); p.eigh_n = int(getattr(p, "eigh_n", 0)) + 1  # v9c prim\n'
EXTEND_HEAVY_HEAD = '        if _RD1C["on"]:\n            e, U = _rd1c_eig_at(ch, N, v); p.eigh_n = int(getattr(p, "eigh_n", 0)) + 1\n        else:\n'

OLD_CONT = "    ref, cinfo, mask = continue_ref_v34(a, ch, window, f_cur, e, U, n_sea)                               # v9 sector: version 8's call\n"
CONT_HEAD = '    if _RD1C["on"]:\n        ref, cinfo, mask = _rd1c_continue_ref_v34(a, ch, window, f_cur, e, U, n_sea)\n    else:\n'

OLD_FRESH = '            e, U = eigT(clocked_gpu(ch, N, v))\n'
FRESH_HEAD = '            if _RD1C["on"]:\n                e, U = _rd1c_eigT_t(clocked_gpu(ch, N, v), ch)\n            else:\n'


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


def indent4(t):
    return "\n".join(("    " + x) if x.strip() else x for x in t.rstrip("\n").split("\n")) + "\n"


def branch(old, head):
    # head's last line is "else:"; the original statement re-indented under it, marked (the reverse restores it verbatim)
    return mark(head.rstrip("\n")) + "\n" + mark(indent4(old).rstrip("\n")) + "\n"


def replacements():
    M = "  # " + MARK
    return [
        Rep("P1 block", OLD_BLOCK_ANCHOR, OLD_BLOCK_ANCHOR + "\n\n" + mark(BLOCK) + "\n"),
        Rep("P2 args", "    _rd4_add_arguments(ap)  # [v9c6 perf]\n", "    _rd4_add_arguments(ap)  # [v9c6 perf]\n    _rd1c_add_arguments(ap)" + M + "\n"),
        Rep("P3 set", "    rd4_set(a)  # [v9c6 perf]\n", "    rd4_set(a)  # [v9c6 perf]\n    rd1c_set(a)" + M + "\n"),
        Rep("P3 close", "        rd4_close()  # [v9c6 perf]\n", "        rd1c_close()" + M + "\n        rd4_close()  # [v9c6 perf]\n"),
        Rep("P4 record", "        v9c3_record_reset(T / rs)  # [v9c3 perf]\n", "        v9c3_record_reset(T / rs)  # [v9c3 perf]\n        rd1c_record_reset(T / rs)" + M + "\n"),
        Rep("H1 heavy path", OLD_HEAVY_PATH, branch(OLD_HEAVY_PATH, HEAVY_PATH_HEAD)),
        Rep("H2 heavy extend", OLD_EXTEND_HEAVY, branch(OLD_EXTEND_HEAVY, EXTEND_HEAVY_HEAD)),
        Rep("H3 heavy continuation", OLD_CONT, branch(OLD_CONT, CONT_HEAD)),
        Rep("H4 fresh start", OLD_FRESH, branch(OLD_FRESH, FRESH_HEAD)),
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
    ap = argparse.ArgumentParser(description="engine_twofield_v9c7 over engine_twofield_v9c6 (exact replacements; fails closed)")
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
            print("STOP: the source md5 is %s, not v9c6's %s" % (md5(src), SOURCE_MD5)); return 2
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
        print("STOP: the reverse does not restore v9c6 byte for byte"); return 2
    try:
        compile(out_text, "engine_twofield_v9c7.py", "exec")
    except SyntaxError as exc:
        print("STOP: the built text does not compile: %s" % exc); return 2
    n_marked = sum(1 for line in out_text.split("\n") if MARK in line)
    print("build: %s (md5 %s) -> v9c7 (md5 %s); %d replacements applied; %d lines carry %r; marks ok; reverse byte for byte; compiled" % (
        a.source, md5(src), md5(out_text), len(reps), n_marked, MARK))
    if a.check:
        print("CHECK PASS (nothing written)"); return 0
    out = a.out or os.path.join(os.path.dirname(os.path.abspath(a.source)), "engine_twofield_v9c7.py")
    try:
        write_new(out, out_text)
    except BuildStop as stop:
        print("STOP: %s" % stop); return 2
    print("written: %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
