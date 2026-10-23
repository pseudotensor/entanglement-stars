#!/usr/bin/env python
"""rd1d_memtrace_v1.py (busi, analyst; PERF RD-1d self-test (c)): run an engine_collapse_v36nq_rd13m* engine unchanged, with the device's
memory read per phase of the Newton record solve. Observation only: the wrapper reads torch.cuda.memory_allocated / max_memory_allocated
and resets the peak counter at each phase boundary (neither touches an allocation or a computation), and wraps by module attribute:
  newton_record_v32 (the record), nk_stage_v32 (each solve: "direct", "stage 1 ...", "stage 2 ..."), confirm_v32 (a confirming step),
  eigh_all (one evaluation's eigendecompositions), and the banded module's _dense (a dense fallback call), rd1d_release / rd1d_restore.
Each interval's peak is charged to the innermost open phase; at each phase's end one line goes to the trace file:
  MEMTRACE <t> <phase path> T/rs <T> entry <GiB> exit <GiB> peak <GiB>   (allocated, the peak since the phase opened, its children included)
Optional --cap-gib C: torch.cuda.set_per_process_memory_fraction(C / total) before the engine starts (the allocator then refuses beyond C
GiB, as a smaller card would); nothing else changes.
    python rd1d_memtrace_v1.py --engine-dir DIR --engine engine_collapse_v36nq_rd13mfd.py --trace FILE [--cap-gib C] -- <engine args>
"""
import importlib
import os
import sys
import time


def main():
    argv = sys.argv[1:]
    if "--" not in argv:
        raise SystemExit("usage: ... -- <engine args>")
    k = argv.index("--")
    own, eng_args = argv[:k], argv[k + 1:]
    opt = {"--engine-dir": None, "--engine": None, "--trace": None, "--cap-gib": None}
    i = 0
    while i < len(own):
        if own[i] not in opt:
            raise SystemExit("unknown option %s" % own[i])
        opt[own[i]] = own[i + 1]; i += 2
    sys.path.insert(0, opt["--engine-dir"])
    import torch
    trace = open(opt["--trace"], "a", buffering=1)
    t00 = time.time()
    stack = []          # open phases: [name, entry GiB, peak GiB]
    state = {"T": None}

    def dev():
        return torch.device("cuda", torch.cuda.current_device())

    def mark():
        """the peak since the last mark charged to every open phase (each phase's peak includes its children's)."""
        if not torch.cuda.is_initialized():
            return 0.0
        p = torch.cuda.max_memory_allocated(dev()) / 2**30
        for f in stack:
            f[2] = max(f[2], p)
        torch.cuda.reset_peak_memory_stats(dev())
        return torch.cuda.memory_allocated(dev()) / 2**30

    def wrap(fn, label):
        def w(*a, **kw):
            nm = label(a, kw)
            now = mark()
            stack.append([nm, now, now])
            try:
                return fn(*a, **kw)
            finally:
                ex = mark()
                f = stack.pop()
                for g in stack:
                    g[2] = max(g[2], f[2])
                trace.write("MEMTRACE %.1f %s T/rs %s entry %.3f exit %.3f peak %.3f\n" % (
                    time.time() - t00, "/".join([g[0] for g in stack] + [f[0]]), state["T"], f[1], ex, f[2]))
        w.__wrapped__ = fn
        return w

    eng_name = os.path.splitext(opt["--engine"])[0]
    E = importlib.import_module(eng_name)
    for mod_name in ("engine_banded_eigh_v1", "engine_banded_eigh_v2"):
        if os.path.exists(os.path.join(opt["--engine-dir"], mod_name + ".py")):
            try:
                M = importlib.import_module(mod_name)
                M._dense = wrap(M._dense, lambda a, kw, m=mod_name: "dense_fallback(%s)" % m[-2:])
            except Exception as ex:      # no CUDA toolkit for numba: that module is not used then
                trace.write("MEMTRACE note: %s not importable (%r)\n" % (mod_name, ex))

    def lab_rec(a, kw):
        state["T"] = "%.4f" % (a[13] / a[0].rs)
        return "record"
    E.newton_record_v32 = wrap(E.newton_record_v32, lab_rec)
    E.nk_stage_v32 = wrap(E.nk_stage_v32, lambda a, kw: "stage[%s]" % str(a[9]).split(",")[0])
    E.confirm_v32 = wrap(E.confirm_v32, lambda a, kw: "confirm")
    E.eigh_all = wrap(E.eigh_all, lambda a, kw: "eigh_all")
    for nm in ("rd1d_release", "rd1d_restore"):
        if hasattr(E, nm):
            setattr(E, nm, wrap(getattr(E, nm), lambda a, kw, n=nm: n))
    if opt["--cap-gib"] is not None:
        torch.cuda.init()
        tot = torch.cuda.get_device_properties(0).total_memory
        frac = float(opt["--cap-gib"]) * 2**30 / tot
        torch.cuda.set_per_process_memory_fraction(frac, 0)
        trace.write("MEMTRACE cap %.3f GiB (fraction %.6f of %.3f GiB)\n" % (float(opt["--cap-gib"]), frac, tot / 2**30))
    sys.argv = [os.path.join(opt["--engine-dir"], opt["--engine"])] + eng_args
    trace.write("MEMTRACE start %s %s\n" % (opt["--engine"], " ".join(eng_args)))
    try:
        E.main()
    finally:
        if torch.cuda.is_initialized():
            trace.write("MEMTRACE end process max_memory_allocated(last interval) %.3f GiB\n" % (torch.cuda.max_memory_allocated(dev()) / 2**30))


if __name__ == "__main__":
    main()
