#!/usr/bin/env python
"""build_engine_collapse_v36nq_rd13.py (busi, analyst; PERF RD-1 + RD-3, REVIEWA_algorithm_redesign_v1 sec. 1 and 3): engine_collapse_v36nq_rd13.py
from engine_collapse_v36nq.py (md5 8549c542..., the ISO_128 and P64_s4 Newton engine) by marked replacement blocks, in the form of
build_engine_collapse_v36nq_aw.py (busi/code/perf_host).
Rules: the source's md5 must begin with 8549c542 (--expect-src-md5); every OLD anchor is copied verbatim from v36nq and occurs exactly once
(checked); every line a NEW block adds or changes carries "rd13 banded" (checked); the reverse pass (NEW -> OLD, each NEW once) restores v36nq
byte for byte (checked in memory before writing and again on the written file; available as --reverse); the output compiles.
    python build_engine_collapse_v36nq_rd13.py [--src .../engine_collapse_v36nq.py] [--dst .../engine_collapse_v36nq_rd13.py] [--check-only]
    python build_engine_collapse_v36nq_rd13.py --reverse --src .../engine_collapse_v36nq_rd13.py --dst .../restored.py --expect .../engine_collapse_v36nq.py

The two options (both default to the parent's code path; with both at their defaults nothing of the patch runs but two dictionary reads and
the import-free option check, so the records, fields, spectra, handoff and slices are v36nq's bitwise; the json's args gain the keys eigh and jvp):
--eigh dense | banded: banded rebinds the module's eigT (every eigendecomposition of the clocked chain: eigT_t, eig_at, eigh_all, the
    continuation's sub-steps, the preparation, the resume) to engine_banded_eigh_v1.eigT_banded (the banded O(n^2) GPU solver with its guards
    and the dense fallback); eigh_all then takes its sequential path (the --streams threads call torch.linalg.eigh directly). The small dense
    eighs of Gram and weight matrices (S, M) stay torch.linalg.eigh. engine_collapse_v17's own eigT (the --resume-from replay) is not rebound.
--jvp eigenbasis | sternheimer (sternheimer needs --nk-jvp exact): inside nk_stage_v32's exact J.v, per channel, X by one shifted banded solve
    per occupied state (engine_banded_eigh_v1.sternheimer_X; the band of h_m at the iterate's applied fields) in place of the two eigenbasis
    products Hm = U_u^H dh U_o, X = U_u M; on a failed guard that channel runs the eigenbasis lines (counted).
Either option on: the kernels are compiled once at the start (engine_banded_eigh_v1.warmup, its seconds printed), the record carries an "rd13"
entry (the module's counters for the record: calls, banded, dense fallbacks and why, seconds, residuals; the Sternheimer channels, fallbacks,
seconds) and one printed line.
Blocks:
  A  the module docstring's lineage lines;
  B  after the JV dictionary: RD13 (the options' state), rd13_configure, rd13_eigT, rd13_jvp_X, rd13_record;
  C  eigh_all: the sequential path under --eigh banded;
  D  nk_stage_v32.jvp: the Sternheimer X (the eigenbasis lines re-indented under "if X is None");
  E  the record: the "rd13" entry;
  F  main(): rd13_configure after C5.backend;
  G  the options --eigh and --jvp (after --nk-jvp).
"""
import argparse
import hashlib
import os
import py_compile

MARK = "rd13 banded"
SRC_MD5_PREFIX = "8549c542"


def blk(s):
    return s[1:] if s.startswith("\n") else s


EDITS = []

EDITS.append(("A lineage", blk(r'''
"""engine_collapse_v36nq.py (busi, reviewer A; ISO_ruling_v1): version 36nq = version 36q with options, each off by default, so that the defaults run'''), blk(r'''
"""engine_collapse_v36nq_rd13.py (busi, analyst; PERF RD-1 + RD-3; built by build_engine_collapse_v36nq_rd13.py over v36nq, md5 8549c542) [rd13 banded]
version 36nq_rd13 = version 36nq with two options, each at its default the parent's code path (v36nq bitwise): [rd13 banded]
--eigh banded: every eigendecomposition of the clocked chain by engine_banded_eigh_v1.eigT_banded (the banded O(n^2) GPU solver: inertia [rd13 banded]
    multisection, inverse iteration with a banded LU, the particle-hole lift, a Rayleigh-Ritz cluster pass and a Lowdin polish; guarded, [rd13 banded]
    with the dense route as the fallback) in place of torch.linalg.eigh; tolerance class (REVIEWA_algorithm_redesign_v1 sec. 1); [rd13 banded]
--jvp sternheimer (needs --nk-jvp exact): the exact J.v's X by one shifted banded solve per occupied state in place of the two eigenbasis [rd13 banded]
    products (sec. 3); the eigenbasis formula on a failed guard. Either option on: an "rd13" entry per record. Version 36nq's docstring follows. [rd13 banded]
engine_collapse_v36nq.py (busi, reviewer A; ISO_ruling_v1): version 36nq = version 36q with options, each off by default, so that the defaults run''')))

EDITS.append(("B helpers", blk(r'''
JV = {"products": 0, "seconds": 0.0, "dropped_pairs": 0, "extra_evaluations": 0, "ref_mismatch": None, "_chk": 0}   # version 35 (--nk-jvp exact)
'''), blk(r'''
JV = {"products": 0, "seconds": 0.0, "dropped_pairs": 0, "extra_evaluations": 0, "ref_mismatch": None, "_chk": 0}   # version 35 (--nk-jvp exact)
RD13 = {"on": False, "eigh": "dense", "jvp": "eigenbasis", "mod": None}   # rd13 banded: the options' state (both defaults: nothing of rd13 runs)
def rd13_configure(a, ap):   # rd13 banded: main(), after C5.backend: the options checked, the module imported and its kernels compiled, eigT rebound
    RD13["eigh"] = getattr(a, "eigh", "dense"); RD13["jvp"] = getattr(a, "jvp", "eigenbasis")   # rd13 banded
    if RD13["eigh"] == "dense" and RD13["jvp"] == "eigenbasis":   # rd13 banded: the defaults: the parent's code path
        return   # rd13 banded
    if RD13["jvp"] == "sternheimer" and getattr(a, "nk_jvp", None) != "exact":   # rd13 banded
        ap.error("rd13: --jvp sternheimer acts inside the exact J.v (--nk-jvp exact)")   # rd13 banded
    if BK["dev"] is None or BK["dev"].type != "cuda":   # rd13 banded
        ap.error("rd13: --eigh banded and --jvp sternheimer run CUDA kernels (numba.cuda): a GPU (--gpu) is required")   # rd13 banded
    import engine_banded_eigh_v1 as EBE   # rd13 banded
    RD13["on"] = True; RD13["mod"] = EBE   # rd13 banded
    w_ = EBE.warmup(BK["dev"])   # rd13 banded: the JIT compile, once
    if RD13["eigh"] == "banded":   # rd13 banded
        globals()["eigT"] = rd13_eigT   # rd13 banded: eigT_t, eig_at, eigh_all and every direct eigT call resolve eigT here at call time
    print("rd13: --eigh %s --jvp %s; %s (%s, numba %s); kernels compiled in %.1f s" % (RD13["eigh"], RD13["jvp"], EBE.VERSION, EBE.STATS["device"], EBE.numba.__version__, w_), flush=True)   # rd13 banded
def rd13_eigT(hmT):   # rd13 banded: the drop-in for engine_collapse_v5.eigT
    return RD13["mod"].eigT_banded(hmT, log=(lambda s_: print("  rd13: " + s_, flush=True)))   # rd13 banded
def rd13_jvp_X(ch, x, eo, eu, Uo, dh):   # rd13 banded: the Sternheimer X of one channel at the iterate x = (N_applied, v_applied), or None (the eigenbasis lines run)
    n_ = len(x) // 2   # rd13 banded
    band_ = RD13["mod"].band_from_csr(clocked_host_sparse(ch, np.asarray(x[:n_], dtype=float), np.asarray(x[n_:], dtype=float)))   # rd13 banded: the upper diagonals of h_m at the iterate
    Y_, inf_ = RD13["mod"].sternheimer_X(band_, eo, eu, Uo, -(dh @ Uo))   # rd13 banded
    if Y_ is None:   # rd13 banded
        print("  rd13: kappa %d J.v by the eigenbasis formula (%s)" % (ch.kappa, inf_.get("fallback")), flush=True)   # rd13 banded
    return Y_   # rd13 banded
def rd13_record():   # rd13 banded: the record's "rd13" entry (the module's counters since the previous record) and its line
    s_ = RD13["mod"].stats_snapshot(reset=True)   # rd13 banded
    s_["eigh"] = RD13["eigh"]; s_["jvp"] = RD13["jvp"]   # rd13 banded
    f_ = ("  rd13: eigh %s: %d calls, %d banded (%d without the lift), %d dense %s, %.1f s (kernels %.1f, post %.1f, guards %.1f), max residual %.1e ||h||, "   # rd13 banded
          "max neighbour overlap %.1e; jvp %s: %d channels, %d eigenbasis %s, %.1f s, max solve residual %.1e")   # rd13 banded
    print(f_ % (s_["eigh"], s_["calls"], s_["banded"], s_["banded_all"], s_["dense_fallback"], s_["fallback_why"] or "", s_["seconds"], s_["kernel_seconds"], s_["post_seconds"], s_["guard_seconds"], s_["max_residual_rel"], s_["max_overlap"], s_["jvp"], s_["st_channels"], s_["st_fallback"], s_["st_fallback_why"] or "", s_["st_seconds"], s_["st_max_residual_rel"]), flush=True)   # rd13 banded
    return s_   # rd13 banded
''')))

EDITS.append(("C eigh_all", blk(r'''
    if not streams or BK["dev"] is None or BK["dev"].type != "cuda":
        for ch in chs:
            out[ch.kappa] = eigT_t(clocked_gpu(ch, N, v))
        return out'''), blk(r'''
    if not streams or BK["dev"] is None or BK["dev"].type != "cuda" or RD13["eigh"] == "banded":   # rd13 banded: --eigh banded takes the sequential path (the stream threads call torch.linalg.eigh)
        for ch in chs:
            out[ch.kappa] = eigT_t(clocked_gpu(ch, N, v))
        return out''')))

EDITS.append(("D jvp", blk(r'''
            dh = dev_csr(clocked_host_sparse(ch, dN, dv))
            Hm = Uu.conj().T @ (dh @ Uo)
            den = eo[None, :] - eu[:, None]
            small = den.abs() < 1e-12
            JV["dropped_pairs"] += int(small.sum())
            M = torch.where(small, torch.zeros_like(Hm), Hm / torch.where(small, torch.ones_like(den), den).to(Hm.dtype))
            X = Uu @ M
'''), blk(r'''
            dh = dev_csr(clocked_host_sparse(ch, dN, dv))
            X = rd13_jvp_X(ch, x, eo, eu, Uo, dh) if RD13["jvp"] == "sternheimer" else None   # rd13 banded: the Sternheimer X (None: the eigenbasis lines below)
            if X is None:   # rd13 banded: v36nq's eigenbasis formula (the default), re-indented
                Hm = Uu.conj().T @ (dh @ Uo)   # rd13 banded
                den = eo[None, :] - eu[:, None]   # rd13 banded
                small = den.abs() < 1e-12   # rd13 banded
                JV["dropped_pairs"] += int(small.sum())   # rd13 banded
                M = torch.where(small, torch.zeros_like(Hm), Hm / torch.where(small, torch.ones_like(den), den).to(Hm.dtype))   # rd13 banded
                X = Uu @ M   # rd13 banded
''')))

EDITS.append(("E record", blk(r'''
        if PAIR["n"]:
            rec["v36b"] = {"pair_workers": PAIR["n"], "products_paired": PAIR["products"], "seconds": PAIR["seconds"], "serial_fallbacks": PAIR["fallbacks"],'''), blk(r'''
        if RD13["on"]:   # rd13 banded
            rec["rd13"] = rd13_record()   # rd13 banded
        if PAIR["n"]:
            rec["v36b"] = {"pair_workers": PAIR["n"], "products_paired": PAIR["products"], "seconds": PAIR["seconds"], "serial_fallbacks": PAIR["fallbacks"],''')))

EDITS.append(("F main", blk(r'''
    C5.backend(a.gpu)
    run(a)
'''), blk(r'''
    C5.backend(a.gpu)
    rd13_configure(a, ap)   # rd13 banded
    run(a)
''')))

EDITS.append(("G options", blk(r'''
    ap.add_argument("--nk-jvp", dest="nk_jvp", default=None, choices=["exact"], help="(v35) the Krylov Jacobian's product by the exact linearization")
'''), blk(r'''
    ap.add_argument("--nk-jvp", dest="nk_jvp", default=None, choices=["exact"], help="(v35) the Krylov Jacobian's product by the exact linearization")
    ap.add_argument("--eigh", dest="eigh", default="dense", choices=["dense", "banded"], help="(rd13) the clocked chain's eigendecomposition: dense = torch.linalg.eigh (v36nq); banded = engine_banded_eigh_v1 (the banded O(n^2) GPU solver, guarded, dense fallback)")   # rd13 banded
    ap.add_argument("--jvp", dest="jvp", default="eigenbasis", choices=["eigenbasis", "sternheimer"], help="(rd13; with --nk-jvp exact) the exact J.v's X: eigenbasis = v36nq's products; sternheimer = one shifted banded solve per occupied state")   # rd13 banded
''')))


def md5(b):
    return hashlib.md5(b).hexdigest()


def forward(text):
    for name, old, new in EDITS:
        c = text.count(old)
        if c != 1:
            raise SystemExit("edit %s: the OLD anchor occurs %d times (must be 1); it begins: %r" % (name, c, old.splitlines()[0][:100]))
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
    """every line a NEW block adds or changes carries MARK; a line of OLD is not new (nor is one moved out from behind the opening triple
    quote); blank lines are neutral."""
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
    ap.add_argument("--src", default="engine_collapse_v36nq.py")
    ap.add_argument("--dst", default=None, help="default: engine_collapse_v36nq_rd13.py beside --src")
    ap.add_argument("--reverse", action="store_true", help="src is engine_collapse_v36nq_rd13.py: restore v36nq into dst")
    ap.add_argument("--expect", default=None, help="(reverse) the file the restored text must equal byte for byte")
    ap.add_argument("--expect-src-md5", default=SRC_MD5_PREFIX, help="(build) the prefix the source's md5 must start with (default %s; '' to skip)" % SRC_MD5_PREFIX)
    ap.add_argument("--check-only", action="store_true")
    a = ap.parse_args()
    if a.dst is None:
        a.dst = os.path.join(os.path.dirname(os.path.abspath(a.src)), "engine_collapse_v36nq_rd13.py" if not a.reverse else "engine_collapse_v36nq_restored.py")
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
        raise SystemExit("the source's md5 does not start with %s: not engine_collapse_v36nq.py; nothing written" % a.expect_src_md5)
    if MARK in text:
        raise SystemExit("the source already carries %r; nothing written" % MARK)
    for name, old, new in EDITS:
        print("  anchor %-12s occurs %d time(s) in the input (%d -> %d lines)" % (name, text.count(old), len(old.splitlines()), len(new.splitlines())))
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
