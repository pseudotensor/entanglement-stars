#!/usr/bin/env python
"""build_engine_collapse_v36x.py (busi, analyst; owner order #299, PERF tracker row ISO-A3, reviewer B perf_iso rank 3):
engine_collapse_v36x.py from engine_collapse_v36r.py (md5 7b6d0f4f...) by marked replacement blocks, in the form of build_engine_collapse_v36r.py.
Rules: the source's md5 must begin with 7b6d0f4f (--expect-src-md5); every OLD anchor is copied verbatim from v36r and occurs exactly once
(checked); every line a NEW block adds or changes carries "[v36x perf]" (checked); the reverse pass (NEW -> OLD, each NEW once) restores v36r
byte for byte (checked in memory before writing and again on the written file; available as --reverse); the output compiles; an existing
destination is never overwritten.
    python build_engine_collapse_v36x.py [--src .../engine_collapse_v36r.py] [--dst .../engine_collapse_v36x.py] [--check-only]
    python build_engine_collapse_v36x.py --reverse --src .../engine_collapse_v36x.py --dst .../restored.py --expect .../engine_collapse_v36r.py

The patch: two options, both off by default; with both off the exact J.v runs v36r's own loop, untouched (the only executed new lines are
the tests of JX["on"]), so the defaults are v36r bitwise.
  --nk-jvp-cache-gib G (G > 0): inside one exact-J.v evaluation point (the memo's eigensystems at x, keyed by the memo's cache object and the
      exact bytes of x), each channel's field-independent intermediates are computed at the first product and reused by the later Krylov
      products at the same x: the occupied/unoccupied basis selections U[:, mask], U[:, ~mask], the small-pair mask, its count and the safe
      denominator, and h @ Uo, Q @ Uo. Every arithmetic expression is v36r's on the same tensors (no reciprocal, no batching, no reordering),
      so the products are bitwise v36r's. At most G GiB are retained per evaluation point (channels in order until the budget; the rest are
      computed fresh at every product, as in v36r). The workspace is dropped at the start of every F evaluation (before its eigh), so it
      never outlives the memo's eigensystems.
  --nk-jvp-defer on: the 2K per-channel site sums of a product stay on the device and are copied to the host in one transfer after the
      channel loop; the host accumulation (deps -= w * s, dJ -= w * s) then runs in v36r's channel order on the same arrays: bitwise.
  Both need --nk-jvp exact. A record carries a "v36x" entry (the options and the per-record counters) only with an option on.
Blocks: A the docstring's lineage; B the JX state and jvp_chs_v36x (before JV); C F drops the workspace at its start (nk_stage_v32);
D the jvp's channel loop: jvp_chs_v36x with an option on, v36r's loop otherwise; E the record's "v36x" entry; F the two options;
G their validation.
"""
import argparse
import hashlib
import os
import py_compile

MARK = "[v36x perf]"
SRC_MD5_PREFIX = "7b6d0f4f"


def blk(s):
    return s[1:] if s.startswith("\n") else s


EDITS = []

# A: the lineage (the first line of v36r's docstring)
EDITS.append(("A lineage", blk(r'''
"""engine_collapse_v36r.py (busi, analyst; #303: the v36q plain-direction fallback ported onto v36n; built by build_engine_collapse_v36r.py over v36n, md5 0c84a446) [v36r fallback]'''), blk(r'''
"""engine_collapse_v36x.py (busi, analyst; #299 PERF ISO-A3, reviewer B perf_iso rank 3; built by build_engine_collapse_v36x.py over v36r, md5 7b6d0f4f) [v36x perf]
version 36x = version 36r with two options, off by default, so that the defaults are version 36r bitwise (with both off the exact J.v runs [v36x perf]
v36r's own channel loop): --nk-jvp-cache-gib G reuses, between the Krylov products at one evaluation point (the memo's eigensystems at x, [v36x perf]
keyed by the memo's cache object and the exact bytes of x), each channel's U[:, mask], U[:, ~mask], small-pair mask and count, safe [v36x perf]
denominator, h @ Uo and Q @ Uo (at most G GiB retained; the same expressions on the same tensors: bitwise); --nk-jvp-defer on copies the [v36x perf]
2K site sums of a product to the host in one transfer after the channel loop, the host accumulation in v36r's order (bitwise). Both need [v36x perf]
--nk-jvp exact; a record carries a "v36x" entry only with an option on. Version 36r's docstring follows. [v36x perf]
engine_collapse_v36r.py (busi, analyst; #303: the v36q plain-direction fallback ported onto v36n; built by build_engine_collapse_v36r.py over v36n, md5 0c84a446) [v36r fallback]''')))

# B: the state and the cached channel loop (before JV)
EDITS.append(("B JX and jvp_chs_v36x", blk(r'''
JV = {"products": 0, "seconds": 0.0, "dropped_pairs": 0, "extra_evaluations": 0, "ref_mismatch": None, "_chk": 0}   # version 35 (--nk-jvp exact)'''), blk(r'''
JX = {"on": False, "gib": 0.0, "defer": False, "hits": 0, "misses": 0, "uncached": 0, "workspaces": 0, "bytes_max": 0, "deferred": 0}   # [v36x perf]


def jvp_chs_v36x(chs, memo, x, dN, dv, deps, dJ):   # [v36x perf]
    """[v36x perf] v36r's exact-J.v channel loop with the evaluation point's invariants reused (--nk-jvp-cache-gib) and the site sums' [v36x perf]
    transfers deferred (--nk-jvp-defer on). The workspace memo["v36x"] belongs to memo["cache"] and the bytes of x; F drops it. [v36x perf]"""
    torch = BK["torch"]   # [v36x perf]
    cache_ = memo["cache"]   # [v36x perf]
    xb = np.asarray(x, dtype=float).tobytes()   # [v36x perf]
    ws = memo.get("v36x")   # [v36x perf]
    if not ws or ws["cache"] is not cache_ or ws["x"] != xb:   # [v36x perf] a new evaluation point: a fresh workspace
        ws = {"cache": cache_, "x": xb, "items": {}, "used": 0}   # [v36x perf]
        memo["v36x"] = ws   # [v36x perf]
        JX["workspaces"] += 1   # [v36x perf]
    lim = int(JX["gib"] * 1073741824.0)   # [v36x perf]
    sites = []   # [v36x perf]
    for ch in chs:   # [v36x perf]
        _, e, U, ref, cinfo = cache_[ch.kappa]   # [v36x perf]
        it = ws["items"].get(ch.kappa)   # [v36x perf]
        if it is None:   # [v36x perf] v36r's expressions
            mask = torch.as_tensor(np.asarray(cinfo["_mask"], bool), device=U.device)   # [v36x perf]
            Uo = U[:, mask]; Uu = U[:, ~mask]   # [v36x perf]
            eo = e[mask].to(torch.float64); eu = e[~mask].to(torch.float64)   # [v36x perf]
            JX["misses"] += 1   # [v36x perf]
        else:   # [v36x perf]
            Uo, Uu = it["Uo"], it["Uu"]   # [v36x perf]
            JX["hits"] += 1   # [v36x perf]
        dh = dev_csr(clocked_host_sparse(ch, dN, dv))   # [v36x perf]
        Hm = Uu.conj().T @ (dh @ Uo)   # [v36x perf]
        if it is None:   # [v36x perf]
            den = eo[None, :] - eu[:, None]   # [v36x perf]
            small = den.abs() < 1e-12   # [v36x perf]
            nsmall = int(small.sum())   # [v36x perf]
            safe = torch.where(small, torch.ones_like(den), den).to(Hm.dtype)   # [v36x perf]
            del den, eo, eu   # [v36x perf]
            AUs = [None, None]   # [v36x perf]
        else:   # [v36x perf]
            small, nsmall, safe, AUs = it["small"], it["nsmall"], it["safe"], it["AUo"]   # [v36x perf]
        JV["dropped_pairs"] += nsmall   # [v36x perf]
        M = torch.where(small, torch.zeros_like(Hm), Hm / safe)   # [v36x perf]
        X = Uu @ M   # [v36x perf]
        if JV["ref_mismatch"] is None or JV["_chk"] < len(chs):   # [v36x perf]
            dg = (ref.conj() * ref).real.sum(dim=1) - (Uo.conj() * Uo).real.sum(dim=1)   # [v36x perf]
            JV["ref_mismatch"] = max(JV["ref_mismatch"] or 0.0, float(dg.abs().max())); JV["_chk"] += 1   # [v36x perf]
        for ia, (A_, acc) in enumerate(((ch.h, deps), (ch.Q, dJ))):   # [v36x perf]
            AX = A_ @ X   # [v36x perf]
            if AUs[ia] is None:   # [v36x perf]
                AUs[ia] = A_ @ Uo   # [v36x perf]
            d_ = (Uo.conj() * AX).real.sum(dim=1) + (X.conj() * AUs[ia]).real.sum(dim=1)   # [v36x perf]
            s_ = d_.reshape(-1, 2).sum(dim=1)   # [v36x perf]
            if JX["defer"]:   # [v36x perf]
                sites.append(s_)   # [v36x perf]
            else:   # [v36x perf]
                acc -= ch.w * s_.cpu().numpy()   # [v36x perf]
        if it is None and lim > 0:   # [v36x perf] retained while the budget lasts
            item = {"Uo": Uo, "Uu": Uu, "small": small, "nsmall": nsmall, "safe": safe, "AUo": AUs}   # [v36x perf]
            cost = sum(t_.numel() * t_.element_size() for t_ in (Uo, Uu, small, safe, AUs[0], AUs[1]))   # [v36x perf]
            if ws["used"] + cost <= lim:   # [v36x perf]
                ws["items"][ch.kappa] = item; ws["used"] += cost   # [v36x perf]
                JX["bytes_max"] = max(JX["bytes_max"], ws["used"])   # [v36x perf]
            else:   # [v36x perf]
                JX["uncached"] += 1   # [v36x perf]
    if JX["defer"] and sites:   # [v36x perf] one transfer; the host accumulation in v36r's channel order
        host = torch.stack(sites).cpu().numpy()   # [v36x perf]
        for i_, ch in enumerate(chs):   # [v36x perf]
            deps -= ch.w * host[2 * i_]   # [v36x perf]
            dJ -= ch.w * host[2 * i_ + 1]   # [v36x perf]
        JX["deferred"] += 1   # [v36x perf]


JV = {"products": 0, "seconds": 0.0, "dropped_pairs": 0, "extra_evaluations": 0, "ref_mismatch": None, "_chk": 0}   # version 35 (--nk-jvp exact)''')))

# C: F drops the workspace before its eigh (nk_stage_v32's F; the block is unique with the memo line)
EDITS.append(("C F drops the workspace", blk(r'''
    def F(x):
        N_a = np.array(x[:n], dtype=float); v_a = np.array(x[n:], dtype=float)
        cache_, srcs_, sol_ = evalG(N_a, v_a)
        Nn_, vn_, Wn_, info_ = sol_
        ctr["n"] += 1
        if jv_on:
            memo["x"] = np.asarray(x, dtype=float).copy(); memo["cache"] = cache_; memo["srcs"] = srcs_'''), blk(r'''
    def F(x):
        if JX["on"]:   # [v36x perf] the J.v workspace belongs to the last evaluation: dropped before this one's eigh
            memo.pop("v36x", None)   # [v36x perf]
        N_a = np.array(x[:n], dtype=float); v_a = np.array(x[n:], dtype=float)
        cache_, srcs_, sol_ = evalG(N_a, v_a)
        Nn_, vn_, Wn_, info_ = sol_
        ctr["n"] += 1
        if jv_on:
            memo["x"] = np.asarray(x, dtype=float).copy(); memo["cache"] = cache_; memo["srcs"] = srcs_''')))

# D: the jvp's channel loop
EDITS.append(("D jvp loop", blk(r'''
        deps = np.zeros(n); dJ = np.zeros(n)
        for ch in chs:
            _, e, U, ref, cinfo = cache_[ch.kappa]'''), blk(r'''
        deps = np.zeros(n); dJ = np.zeros(n)
        if JX["on"]:   # [v36x perf]
            jvp_chs_v36x(chs, memo, x, dN, dv, deps, dJ)   # [v36x perf]
        for ch in (chs if not JX["on"] else ()):   # [v36x perf] v36r's loop, unchanged, with both options off
            _, e, U, ref, cinfo = cache_[ch.kappa]''')))

# E: the record's entry
EDITS.append(("E record entry", blk(r'''
            PAIR.update({"products": 0, "seconds": 0.0, "fallbacks": 0, "hits": 0, "misses": 0, "errors": []})   # per record, as JV'''), blk(r'''
            PAIR.update({"products": 0, "seconds": 0.0, "fallbacks": 0, "hits": 0, "misses": 0, "errors": []})   # per record, as JV
        if JX["on"]:   # [v36x perf] per record, as JV
            rec["v36x"] = {"jvp_cache_gib": JX["gib"], "jvp_defer": JX["defer"], "channel_hits": JX["hits"], "channel_misses": JX["misses"],   # [v36x perf]
                           "uncached": JX["uncached"], "workspaces": JX["workspaces"], "bytes_max": JX["bytes_max"], "deferred_products": JX["deferred"]}   # [v36x perf]
            JX.update({"hits": 0, "misses": 0, "uncached": 0, "workspaces": 0, "bytes_max": 0, "deferred": 0})   # [v36x perf]''')))

# F: the options
EDITS.append(("F argparse", blk(r'''
    ap.add_argument("--pair-workers", dest="pair_workers", type=int, default=0, choices=[0, 2], help="(v36b) the exact J.v's 2 field solves per product in 2 persistent worker processes")'''), blk(r'''
    ap.add_argument("--pair-workers", dest="pair_workers", type=int, default=0, choices=[0, 2], help="(v36b) the exact J.v's 2 field solves per product in 2 persistent worker processes")
    ap.add_argument("--nk-jvp-cache-gib", dest="nk_jvp_cache_gib", type=float, default=0.0, help="(v36x) the exact J.v's field-independent per-channel intermediates reused across the Krylov products at one evaluation point, at most this many GiB; 0: off (v36r bitwise); bitwise")   # [v36x perf]
    ap.add_argument("--nk-jvp-defer", dest="nk_jvp_defer", default="off", choices=["off", "on"], help="(v36x) the exact J.v's 2K site sums copied to the host in one transfer per product; bitwise")   # [v36x perf]''')))

# G: the validation (before the --nk-set resolve block)
EDITS.append(("G validation", blk(r'''
    if a.nk_set == "resolve":
        if a.nk_jvp != "exact" or a.nk_line_search != "halving":'''), blk(r'''
    if a.nk_jvp_cache_gib or a.nk_jvp_defer == "on":   # [v36x perf]
        if a.nk_jvp != "exact" or not (a.nk_jvp_cache_gib >= 0.0):   # [v36x perf]
            ap.error("version 36x: --nk-jvp-cache-gib (>= 0) and --nk-jvp-defer act inside the exact J.v (--nk-jvp exact)")   # [v36x perf]
        JX.update({"on": True, "gib": float(a.nk_jvp_cache_gib), "defer": a.nk_jvp_defer == "on"})   # [v36x perf]
    if a.nk_set == "resolve":
        if a.nk_jvp != "exact" or a.nk_line_search != "halving":''')))


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
        olds |= {l[3:] for l in old.splitlines() if l.startswith('"""')}   # a line moved out from behind the opening triple quote
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
    ap.add_argument("--src", default="engine_collapse_v36r.py")
    ap.add_argument("--dst", default=None, help="default: engine_collapse_v36x.py beside --src")
    ap.add_argument("--reverse", action="store_true", help="src is engine_collapse_v36x.py: restore v36r into dst")
    ap.add_argument("--expect", default=None, help="(reverse) the file the restored text must equal byte for byte")
    ap.add_argument("--expect-src-md5", default=SRC_MD5_PREFIX, help="(build) the prefix the source's md5 must start with (default %s; '' to skip)" % SRC_MD5_PREFIX)
    ap.add_argument("--check-only", action="store_true")
    a = ap.parse_args()
    if a.dst is None:
        a.dst = os.path.join(os.path.dirname(os.path.abspath(a.src)), "engine_collapse_v36x.py" if not a.reverse else "engine_collapse_v36r_restored.py")
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
        raise SystemExit("the source's md5 does not start with %s: not engine_collapse_v36r.py; nothing written" % a.expect_src_md5)
    if MARK in text:
        raise SystemExit("the source already carries %r; nothing written" % MARK)
    for name, old, new in EDITS:
        print("  anchor %-26s occurs %d time(s) in the input (%d -> %d lines)" % (name, text.count(old), len(old.splitlines()), len(new.splitlines())))
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
