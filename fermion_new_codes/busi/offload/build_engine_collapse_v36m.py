#!/usr/bin/env python
"""build_engine_collapse_v36m.py (busi, reviewer A; reorientation 5, the host-offload Newton engine): engine_collapse_v36m.py from
engine_collapse_v36j.py (md5 f1601080...) by marked replacement blocks.
Rules: every OLD anchor occurs exactly once in the input (checked in the input and again before each replacement); every line a NEW block
adds carries "v36m offload" (checked); the reverse pass (NEW -> OLD, each NEW once) restores the input byte for byte (checked after every
build; available as --reverse); the output compiles (py_compile).
    python build_engine_collapse_v36m.py --src .../engine_collapse_v36j.py --dst .../engine_collapse_v36m.py [--expect-src-md5 f1601080]
    python build_engine_collapse_v36m.py --reverse --src .../engine_collapse_v36m.py --dst .../restored.py --expect .../engine_collapse_v36j.py
"""
import argparse
import hashlib
import py_compile

MARK = "v36m offload"


def blk(s):
    return s[1:] if s.startswith("\n") else s


EDITS = []

# A: the lineage (the first line of the module docstring)
EDITS.append(("A lineage", blk(r'''
"""engine_collapse_v36j.py (busi, reviewer A; M3c_nk_oom_v1): version 36j = version 36d with one call and no option: at the end of every record'''), blk(r'''
"""engine_collapse_v36m.py (busi, reviewer A; reorientation 5, the host-offload Newton engine; built by build_engine_collapse_v36m.py) [v36m offload]
version 36m = version 36j with one option, off by default, so that the defaults are version 36j bitwise (with it off nothing of it runs): [v36m offload]
--nk-host-release on | test (needs --mix newton, --nk-line-search halving, --nk-jvp exact): in the set-resolving solve (--nk-set resolve) a [v36m offload]
    finished round's kept evaluation (e, U and the continued reference R2 of every channel, 96 kappa_sea nsites^2 bytes) leaves the device for [v36m offload]
    pinned host memory (pageable when pinning is refused) at the start of the next round, and the chosen round's comes back by exact copy at the [v36m offload]
    record's choice, every tensor with its size, strides and view relation (R2 a view of U stays a view of the restored U); a trial the halving [v36m offload]
    search rejects leaves the exact J.v's memo at once (jvp reads the memo only at the accepted iterate, which is always its last evaluation), and [v36m offload]
    the memo is emptied when a solve ends; so at most two evaluation caches are on the device at any moment (the kept iterate's and the one in [v36m offload]
    progress) in place of R + 2 on a record of R rounds (M3c_nk_oom_v1 section 4: 88.9 GiB at 4096 sites and three rounds; 43.9 GiB now). The data [v36m offload]
    come back by copy and never by recomputation (a recomputation would change the eigendecomposition and solve counts the json and the record line [v36m offload]
    carry). Nothing the solve computes changes, so every record, fields, spectra and slice number is version 36j's bitwise; the record carries a [v36m offload]
    "v36m" entry (rounds released and restored, bytes, seconds, pinned) only with the option on; "test" also sends the single round of a quiet record [v36m offload]
    through the host and back (the gate's exercise of the copy on every record); wall-clock seconds differ. Version 36j's docstring follows. [v36m offload]
engine_collapse_v36j.py (busi, reviewer A; M3c_nk_oom_v1): version 36j = version 36d with one call and no option: at the end of every record''')))

# B: the state dict and the helpers, after NSR
EDITS.append(("B helpers", blk(r'''
NSR = {"on": False, "prev_sig": None, "rec": None, "prev_m": None, "other": None}   # version 36c (--nk-set resolve); 36d: prev_m, other'''), blk(r'''
NSR = {"on": False, "prev_sig": None, "rec": None, "prev_m": None, "other": None}   # version 36c (--nk-set resolve); 36d: prev_m, other
OFF = {"on": False, "test": False, "pinned": None, "rec": None}   # v36m offload: --nk-host-release (on | test): the host release of finished rounds' evaluations
def _rec_v36m():   # v36m offload: the record's counters (offload_record_v36m hands them to the record and starts again)
    if OFF["rec"] is None:   # v36m offload
        OFF["rec"] = {"rounds_released": 0, "rounds_restored": 0, "bytes_released": 0, "bytes_restored": 0, "seconds_release": 0.0, "seconds_restore": 0.0, "memo_dropped": 0}   # v36m offload
    return OFF["rec"]   # v36m offload
def offload_record_v36m():   # v36m offload: the record's "v36m" json entry
    r_ = dict(_rec_v36m()); OFF["rec"] = None   # v36m offload
    r_["pinned"] = OFF["pinned"]   # v36m offload
    return r_   # v36m offload
def _host_v36m(t):   # v36m offload: an exact host copy of the tensor t with its size and strides, pinned where the driver grants it (decided once per run), else pageable
    torch = BK["torch"]   # v36m offload
    h = None   # v36m offload
    if OFF["pinned"] is not False:   # v36m offload
        try:   # v36m offload
            h = torch.empty_strided(tuple(t.size()), tuple(t.stride()), dtype=t.dtype, device="cpu", pin_memory=True)   # v36m offload
            OFF["pinned"] = True   # v36m offload
        except (RuntimeError, TypeError):   # v36m offload: no pinned memory (no CUDA runtime, or the host refused): pageable from here on
            OFF["pinned"] = False   # v36m offload
    if h is None:   # v36m offload
        h = torch.empty_strided(tuple(t.size()), tuple(t.stride()), dtype=t.dtype, device="cpu")   # v36m offload
    h.copy_(t)   # v36m offload
    return h   # v36m offload
def _dev_v36m(h):   # v36m offload: the host copy back on the run's device, size and strides kept (a fresh block, aligned as the caching allocator aligns every block)
    torch = BK["torch"]   # v36m offload
    d = torch.empty_strided(tuple(h.size()), tuple(h.stride()), dtype=h.dtype, device=BK["dev"])   # v36m offload
    d.copy_(h)   # v36m offload
    return d   # v36m offload
def _same_storage_v36m(x, y):   # v36m offload: True when x and y share one storage (R2 a view of U)
    try:   # v36m offload
        return x.untyped_storage().data_ptr() == y.untyped_storage().data_ptr()   # v36m offload
    except (AttributeError, RuntimeError):   # v36m offload: older torch
        return x.storage().data_ptr() == y.storage().data_ptr()   # v36m offload
class HostCache_v36m(object):   # v36m offload: a round's kept evaluation on the host: kappa -> (hm, e, U, R2 or its view description, cinfo)
    def __init__(self, cache):   # v36m offload
        torch = BK["torch"]   # v36m offload
        self.entries = {}; self.bytes = 0   # v36m offload
        for k_, (hm_, e_, U_, ref_, ci_) in cache.items():   # v36m offload
            eh = _host_v36m(e_) if torch.is_tensor(e_) else e_   # v36m offload
            Uh = _host_v36m(U_) if torch.is_tensor(U_) else U_   # v36m offload
            for q_ in (e_, U_):   # v36m offload
                if torch.is_tensor(q_):   # v36m offload
                    self.bytes += q_.numel() * q_.element_size()   # v36m offload
            if torch.is_tensor(ref_) and torch.is_tensor(U_) and _same_storage_v36m(ref_, U_):   # v36m offload: R2 a view of U (continue_fast_info's U[:, :nneg]): its size, strides and offset relative to U, not a second copy
                rh = ("view_v36m", tuple(ref_.size()), tuple(ref_.stride()), int(ref_.storage_offset() - U_.storage_offset()))   # v36m offload
            else:   # v36m offload
                rh = _host_v36m(ref_) if torch.is_tensor(ref_) else ref_   # v36m offload
                if torch.is_tensor(ref_):   # v36m offload
                    self.bytes += ref_.numel() * ref_.element_size()   # v36m offload
            self.entries[k_] = (hm_, eh, Uh, rh, ci_)   # v36m offload
    def to_device(self):   # v36m offload: the cache back on the device, every tensor with its size, strides and view relation
        torch = BK["torch"]   # v36m offload
        out = {}   # v36m offload
        for k_, (hm_, eh, Uh, rh, ci_) in self.entries.items():   # v36m offload
            e_ = _dev_v36m(eh) if torch.is_tensor(eh) else eh   # v36m offload
            U_ = _dev_v36m(Uh) if torch.is_tensor(Uh) else Uh   # v36m offload
            if isinstance(rh, tuple) and len(rh) == 4 and rh[0] == "view_v36m":   # v36m offload
                ref_ = torch.as_strided(U_, rh[1], rh[2], U_.storage_offset() + rh[3])   # v36m offload
            else:   # v36m offload
                ref_ = _dev_v36m(rh) if torch.is_tensor(rh) else rh   # v36m offload
            out[k_] = (hm_, e_, U_, ref_, ci_)   # v36m offload
        return out   # v36m offload
def release_round_v36m(b):   # v36m offload: a finished round's kept evaluation (nk_stage_v32's "best" dict b) to the host; the device tensors die with their last holder
    if b is None or b.get("cache") is None or isinstance(b.get("cache"), HostCache_v36m):   # v36m offload
        return   # v36m offload
    t_ = time.time()   # v36m offload
    hc = HostCache_v36m(b["cache"])   # v36m offload
    sync_dev()   # v36m offload: the copies complete before the device tensors can be released
    b["cache"] = hc   # v36m offload
    r_ = _rec_v36m(); r_["rounds_released"] += 1; r_["bytes_released"] += hc.bytes; r_["seconds_release"] += time.time() - t_   # v36m offload
def restore_round_v36m(b):   # v36m offload: the chosen round's kept evaluation back on the device by exact copy (a no-op when it never left)
    if b is None or not isinstance(b.get("cache"), HostCache_v36m):   # v36m offload
        return   # v36m offload
    t_ = time.time()   # v36m offload
    hc = b["cache"]   # v36m offload
    b["cache"] = hc.to_device()   # v36m offload
    sync_dev()   # v36m offload
    r_ = _rec_v36m(); r_["rounds_restored"] += 1; r_["bytes_restored"] += hc.bytes; r_["seconds_restore"] += time.time() - t_   # v36m offload
def memo_drop_v36m(F):   # v36m offload: the exact J.v's memo emptied through the hook nk_stage_v32 sets on F (a no-op without --nk-jvp exact)
    d_ = getattr(F, "v36m_drop", None)   # v36m offload
    if d_ is not None:   # v36m offload
        d_()   # v36m offload''')))

# C: the memo's hook in nk_stage_v32
EDITS.append(("C memo hook", blk(r'''
    jv_on = getattr(a, "nk_jvp", None) == "exact"
    memo = {"x": None, "cache": None, "srcs": None}
    if jv_on:
        F.jvp = jvp'''), blk(r'''
    jv_on = getattr(a, "nk_jvp", None) == "exact"
    memo = {"x": None, "cache": None, "srcs": None}
    if jv_on:
        F.jvp = jvp
        if OFF["on"]:   # v36m offload: the hook memo_drop_v36m calls on a rejected trial and at a solve's end
            def v36m_drop():   # v36m offload: the cache and sources leave; memo["x"] stays, so jvp re-evaluates F exactly where version 36j does (never after a drop: every jvp follows an accepted evaluation, which refills the memo)
                memo["cache"] = None; memo["srcs"] = None   # v36m offload
                _rec_v36m()["memo_dropped"] += 1   # v36m offload
            F.v36m_drop = v36m_drop   # v36m offload''')))

# D: nk_loop_v36c, a rejected trial
EDITS.append(("D v36c rejected trial", blk(r'''
                if rt < r0:
                    acc = (s, xt, Ft, L["gap"])
                    break
            s *= 0.5'''), blk(r'''
                if rt < r0:
                    acc = (s, xt, Ft, L["gap"])
                    break
            if OFF["on"]:   # v36m offload: this trial is rejected (crossing, or no fall): its evaluation leaves the memo now
                memo_drop_v36m(F)   # v36m offload
            s *= 0.5''')))

# E: nk_loop_v32's halving search, a rejected trial
EDITS.append(("E v32 rejected trial", blk(r'''
                if rt < r0:
                    acc = (s, xt, Ft)
                    break
                s *= 0.5'''), blk(r'''
                if rt < r0:
                    acc = (s, xt, Ft)
                    break
                if OFF["on"]:   # v36m offload: a rejected trial's evaluation leaves the memo (version 32's halving search)
                    memo_drop_v36m(F)   # v36m offload
                s *= 0.5''')))

# F: rounds_v36c, the start of a round
EDITS.append(("F round start", blk(r'''
    for k_round in range(1, 4):
        nk["best"] = None; nk["failed"] = None'''), blk(r'''
    for k_round in range(1, 4):
        if OFF["on"] and rounds:   # v36m offload: the finished round's kept evaluation leaves the device before the next round's first evaluation
            memo_drop_v36m(F); release_round_v36m(rounds[-1]["best"])   # v36m offload
        nk["best"] = None; nk["failed"] = None''')))

# G: rounds_v36c, the choice
EDITS.append(("G choice", blk(r'''
    for r in rounds:
        if r is not chosen and r["best"] is not None:
            r["best"] = {k: r["best"][k] for k in ("resid", "k_it", "app", "dE", "E")}
    nk["best"] = chosen["best"]; nk["failed"] = chosen["failed"]'''), blk(r'''
    for r in rounds:
        if r is not chosen and r["best"] is not None:
            r["best"] = {k: r["best"][k] for k in ("resid", "k_it", "app", "dE", "E")}
    if OFF["on"]:   # v36m offload: the solve is over: the memo emptied; the chosen round's evaluation back on the device by exact copy
        memo_drop_v36m(F)   # v36m offload
        if OFF["test"]:   # v36m offload: --nk-host-release test: the kept round goes through the host too (the gate exercises the copy on every record)
            release_round_v36m(chosen["best"])   # v36m offload
        restore_round_v36m(chosen["best"])   # v36m offload
    nk["best"] = chosen["best"]; nk["failed"] = chosen["failed"]''')))

# H: nk_stage_v32 without --nk-set resolve: the memo emptied after the loop
EDITS.append(("H plain loop end", blk(r'''
        status = "stopped: %s" % str(ex)[:120]
    out = {"name": name, "start": start, "status": status, "line_search": steps, "best": nk["best"], "failed": nk["failed"], "evalG": evalG,'''), blk(r'''
        status = "stopped: %s" % str(ex)[:120]
    if OFF["on"]:   # v36m offload: the solve is over; the memo's evaluation leaves (jvp is not called again)
        memo_drop_v36m(F)   # v36m offload
    out = {"name": name, "start": start, "status": status, "line_search": steps, "best": nk["best"], "failed": nk["failed"], "evalG": evalG,''')))

# I: run(): the record's entry
EDITS.append(("I record entry", blk(r'''
        if V36["fixes"] or V36["kcache"]:
            rec["v36"] = {"fixes": V36["fixes"], "gap_none_substeps": V36["gap_none_substeps"], "kernel_cache": dict(V36["kcache"]) if V36["kcache"] else None}'''), blk(r'''
        if OFF["on"]:   # v36m offload: the record's host-release entry (only with the option on); a line when something left the device
            rec["v36m"] = offload_record_v36m()   # v36m offload
            if rec["v36m"]["rounds_released"]:   # v36m offload
                print("  v36m offload: rounds released %d (%.2f GiB in %.1f s), restored %d (%.2f GiB in %.1f s), memo dropped %d times, pinned %s" % (   # v36m offload
                    rec["v36m"]["rounds_released"], rec["v36m"]["bytes_released"] / 2.0 ** 30, rec["v36m"]["seconds_release"], rec["v36m"]["rounds_restored"],   # v36m offload
                    rec["v36m"]["bytes_restored"] / 2.0 ** 30, rec["v36m"]["seconds_restore"], rec["v36m"]["memo_dropped"], rec["v36m"]["pinned"]), flush=True)   # v36m offload
        if V36["fixes"] or V36["kcache"]:
            rec["v36"] = {"fixes": V36["fixes"], "gap_none_substeps": V36["gap_none_substeps"], "kernel_cache": dict(V36["kcache"]) if V36["kcache"] else None}''')))

# J: the option
EDITS.append(("J argparse", blk(r'''
    ap.add_argument("--nk-set", dest="nk_set", default=None, choices=["resolve"], help="(v36c) the set-resolving Newton solve")'''), blk(r'''
    ap.add_argument("--nk-set", dest="nk_set", default=None, choices=["resolve"], help="(v36c) the set-resolving Newton solve")
    ap.add_argument("--nk-host-release", dest="nk_host_release", default="off", choices=["off", "on", "test"], help="(v36m) finished rounds' eigendecompositions and references to pinned host memory, the memo emptied of rejected trials; bitwise; test: the kept round round-trips too")   # v36m offload''')))

# K: the validation
EDITS.append(("K validation", blk(r'''
    if a.nk_set == "resolve":
        if a.nk_jvp != "exact" or a.nk_line_search != "halving":
            ap.error("version 36c: --nk-set resolve needs --nk-jvp exact and --nk-line-search halving")
        NSR["on"] = True'''), blk(r'''
    if a.nk_set == "resolve":
        if a.nk_jvp != "exact" or a.nk_line_search != "halving":
            ap.error("version 36c: --nk-set resolve needs --nk-jvp exact and --nk-line-search halving")
        NSR["on"] = True
    if a.nk_host_release != "off":   # v36m offload
        if a.mix != "newton" or a.nk_line_search != "halving" or a.nk_jvp != "exact":   # v36m offload: the memo exists under the exact J.v, and only the halving search never makes a rejected trial the next iterate
            ap.error("version 36m: --nk-host-release needs --mix newton, --nk-line-search halving and --nk-jvp exact")   # v36m offload
        OFF["on"] = True; OFF["test"] = (a.nk_host_release == "test")   # v36m offload''')))


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
    bad, n_marked = [], 0
    for name, old, new in EDITS:
        olds = set(old.splitlines())
        olds |= {l[3:] for l in old.splitlines() if l.startswith('"""')}   # busi fix: an old line moved out from behind the opening triple quote is not new
        for ln in new.splitlines():
            if ln in olds:
                continue
            if MARK in ln:
                n_marked += 1
            else:
                bad.append((name, ln))
    return bad, n_marked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--reverse", action="store_true", help="src is engine_collapse_v36m.py: restore v36j into dst")
    ap.add_argument("--expect", default=None, help="(reverse) the file the restored text must equal byte for byte")
    ap.add_argument("--expect-src-md5", default=None, help="(build) a prefix the source's md5 must start with (f1601080)")
    ap.add_argument("--check-only", action="store_true")
    a = ap.parse_args()
    src = open(a.src, "rb").read()
    text = src.decode("utf-8")
    if a.reverse:
        out = reverse(text).encode("utf-8")
        if not a.check_only:
            open(a.dst, "wb").write(out)
        print("reverse: wrote %s (md5 %s)" % (a.dst, md5(out)))
        if a.expect:
            ok = out == open(a.expect, "rb").read()
            print("restored equals %s byte for byte: %s" % (a.expect, ok))
            raise SystemExit(0 if ok else 1)
        return
    print("source %s: md5 %s, %d bytes" % (a.src, md5(src), len(src)))
    if a.expect_src_md5 and not md5(src).startswith(a.expect_src_md5):
        raise SystemExit("the source's md5 does not start with %s" % a.expect_src_md5)
    for name, old, new in EDITS:
        print("  anchor %-24s occurs %d time(s) in the input" % (name, text.count(old)))
    bad, n_marked = marker_check()
    if bad:
        for name, ln in bad:
            print("  UNMARKED new line in %s: %r" % (name, ln))
        raise SystemExit("new lines without the marker")
    print("every new line carries %r (%d marked lines)" % (MARK, n_marked))
    built = forward(text)
    out = built.encode("utf-8")
    back = reverse(built).encode("utf-8")
    print("reversible (NEW -> OLD restores the input byte for byte): %s" % (back == src))
    if back != src:
        raise SystemExit("the reverse pass does not restore the input")
    if a.check_only:
        print("check only: %s would be %d bytes, md5 %s" % (a.dst, len(out), md5(out)))
        return
    open(a.dst, "wb").write(out)
    py_compile.compile(a.dst, doraise=True)
    print("wrote %s: %d bytes, md5 %s; compiles" % (a.dst, len(out), md5(out)))
    print("check: grep -c '%s' %s  ->  %d" % (MARK, a.dst, n_marked))


if __name__ == "__main__":
    main()
