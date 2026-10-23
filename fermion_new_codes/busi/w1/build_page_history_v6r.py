#!/usr/bin/env python
"""build_page_history_v6r.py (busi lane PERF build, owner order #299; final4/PERF/PERF_TRACKER.md rows PG-A1, PG-A2 and PG-A3; reviewer B perf_page
ranks 1 to 3, box lead_api/reviewb_perf_page_20261019_182307): writes page_history_v6r.py from page_history_v6q.py (md5 b4f4ea69, the kappa 64/96
production candidate) by exact string replacements in build_engine_collapse_v36u.py's form. Each OLD is copied exactly from page_history_v6q.py and
must occur exactly once (else the build aborts); each NEW keeps its OLD's text and adds or changes lines so that replacing every NEW by its OLD
restores the source byte for byte (checked here). Every inserted or changed line carries the marker "[v6r perf]" (checked here).

The changes (all bitwise at the defaults, so that v6r at v6q's arguments writes v6q's numbers; the summary's "args" gains three keys):
  PG-A1 (reviewer B rank 1) one B.apply per column block shared by every trace of the same operator on the same state (the collar trace and the ledger
        cuts' traces at each sub-step's start and end, and the work jump with its cuts, on the sector route and the full route), with the scalar
        transfers deferred: each trace keeps its own per-block torch.sum of the same expression, and its blocks are added on the host in the
        original left-to-right order, from one device-to-host copy per call in place of one per block per trace. --v6r-share-traces off restores
        v6q's calls (one tr_blocked per trace).
  PG-A2 (reviewer B rank 2) --prop-chunk P: the column chunk of the sea state's Chebyshev propagation alone (step_record's c.Phi, both routes), apart
        from --chunk (the D13 sources and probes, the archive, the ledgers) and --trace-chunk (every trace). 0 (the default) = --chunk, v6q's
        chunking. A sweep (2048, 256, 512, 128) is a timing experiment for the A/B; Banded.apply is column-separable, so a nonzero value is
        expected bitwise too, which the A/B checks.
  PG-A3 (reviewer B rank 3) the full-state assembly (W1S.assemble_full_state, one complete QR of the + sector block) at a checkpoint whose clock also
        has the probes, the saved columns or the reference's live columns: the checkpoint's assembly is kept as a private device copy, keyed by the
        sea tensor's identity, its version counter and the record, and every later consumer at the same clock gets a fresh clone of it in place
        of a second QR; cleared at the next step_record. Costs one extra full-columns copy (2 n x n complex128: 2 GiB at nsites 8192) on the
        device at those clocks only. --v6r-full-cache off restores v6q's assemblies.
Usage (box): python build_page_history_v6r.py [page_history_v6q.py path] [page_history_v6r.py path]   (defaults: v6q beside this script, v6r beside v6q)."""
import hashlib
import os
import py_compile
import sys

MARK = "[v6r perf]"
SRC_MD5 = "b4f4ea69c199ac9785e676e060f36a15"

REPLACEMENTS = [
("docstring",
r'''#!/usr/bin/env python
"""page_history_v6o.py (reviewer A, the busi lane, W1 of REVIEWA_REVIEW_web40i.md section 4; code defects 10 to 13 of its section 5): page_history_v6n.py
''',
r'''#!/usr/bin/env python
"""page_history_v6r.py [v6r perf] = page_history_v6q.py (md5 b4f4ea69) built by build_page_history_v6r.py (busi PERF, PERF_TRACKER rows PG-A1 to PG-A3): [v6r perf]
PG-A1 one B.apply per column block shared by the traces of one operator on one state, the scalar transfers deferred (the same per-block sums, [v6r perf]
added on the host in the original order; --v6r-share-traces off: v6q's calls); PG-A2 --prop-chunk, the sea propagation's column chunk apart [v6r perf]
from --chunk and --trace-chunk (0 = --chunk, v6q's); PG-A3 the checkpoint's full-state assembly reused by the probes, the saved columns and [v6r perf]
the live reference at the same clock (clones of a private copy; --v6r-full-cache off: v6q's assemblies). Bitwise against v6q at the defaults. [v6r perf]
[v6r perf] v6q's docstring follows: page_history_v6o.py (reviewer A, the busi lane, W1 of REVIEWA_REVIEW_web40i.md section 4; code defects 10 to 13 of its section 5): page_history_v6n.py
'''),
("the shared traces and the full-state cache (module level)",
r'''        tot += float(torch.sum((Xb.conj() * Y).real))
        del Y
    return tot


# ---------------------------------------------------------------------------------------------------------------- the state
''',
r'''        tot += float(torch.sum((Xb.conj() * Y).real))
        del Y
    return tot


PERF = {"share": True, "full_cache": True, "multi_calls": 0, "applies_saved": 0, "full_hits": 0, "full_misses": 0, "full_kept": 0}   # [v6r perf] the switches (HistRun sets them from the args) and the process's counters


def tr_blocked_multi(B, X, ws, chunk=None):   # [v6r perf] PG-A1
    """[v6r perf] PG-A1: [tr_blocked(B, X, w) for w in ws] with one B.apply per column block: every trace keeps tr_blocked's per-block expression [v6r perf]
    torch.sum((Xb.conj() * (w * Y)).real) on the same block and the same Y, and its blocks are added on the host left to right from 0.0 as [v6r perf]
    tr_blocked adds them, after one device-to-host copy of all the per-block sums (the stack copies the float64 scalars exactly). Bitwise. [v6r perf]"""
    ch = int(chunk or TRACE_CHUNK[0])   # [v6r perf]
    parts = [[] for _ in ws]   # [v6r perf]
    for j0 in range(0, X.shape[1], ch):   # [v6r perf]
        Xb = X[:, j0: j0 + ch]   # [v6r perf]
        Y = B.apply(Xb)   # [v6r perf]
        for i_, w in enumerate(ws):   # [v6r perf]
            Z = Y if w is None else w * Y   # [v6r perf]
            parts[i_].append(torch.sum((Xb.conj() * Z).real))   # [v6r perf]
            del Z   # [v6r perf]
        del Y   # [v6r perf]
    nb = len(parts[0]) if parts else 0   # [v6r perf]
    PERF["multi_calls"] += 1; PERF["applies_saved"] += nb * max(0, len(ws) - 1)   # [v6r perf]
    if nb == 0:   # [v6r perf]
        return [0.0 for _ in ws]   # [v6r perf]
    vals = torch.stack([p_ for ps_ in parts for p_ in ps_]).cpu().tolist()   # [v6r perf] one transfer
    out = []   # [v6r perf]
    for i_ in range(len(ws)):   # [v6r perf]
        tot = 0.0   # [v6r perf]
        for x_ in vals[i_ * nb: (i_ + 1) * nb]:   # [v6r perf] tr_blocked's order
            tot += x_   # [v6r perf]
        out.append(tot)   # [v6r perf]
    return out   # [v6r perf]


def tr_multi(B, X, ws):   # [v6r perf] PG-A1
    """[v6r perf] the traces of one operator on one state: tr_blocked_multi, or v6q's one tr_blocked per trace under --v6r-share-traces off."""   # [v6r perf]
    if not PERF["share"]:   # [v6r perf]
        return [tr_blocked(B, X, w) for w in ws]   # [v6r perf]
    return tr_blocked_multi(B, X, ws)   # [v6r perf]


def v6r_full_keep(c, k, X_full, keep):   # [v6r perf] PG-A3
    """[v6r perf] PG-A3: at a checkpoint, a private copy of the full columns just assembled, kept for the later consumers at the same clock (or dropped)."""   # [v6r perf]
    c._v6r_full = (c.Phi, c.Phi._version, int(k), X_full.detach().clone()) if keep else None   # [v6r perf]
    PERF["full_kept"] += int(bool(keep))   # [v6r perf]


def v6r_full(c, k):   # [v6r perf] PG-A3
    """[v6r perf] PG-A3: v6q's `W1S.assemble_full_state(c.Phi) if getattr(c, "sector_prop", False) else c.Phi`, except that a copy kept at this [v6r perf]
    record for this very sea tensor at its current version is cloned in place of a second QR."""   # [v6r perf]
    if not getattr(c, "sector_prop", False):   # [v6r perf]
        return c.Phi   # [v6r perf]
    ent = getattr(c, "_v6r_full", None)   # [v6r perf]
    if ent is not None and ent[0] is c.Phi and ent[1] == c.Phi._version and ent[2] == int(k):   # [v6r perf]
        PERF["full_hits"] += 1   # [v6r perf]
        return ent[3].clone()   # [v6r perf]
    PERF["full_misses"] += 1   # [v6r perf]
    return W1S.assemble_full_state(c.Phi)   # [v6r perf]


# ---------------------------------------------------------------------------------------------------------------- the state
'''),
("parse: the options",
r'''    ap.add_argument("--trace-chunk", dest="trace_chunk", type=int, default=1024, help="v6e: the column block of every trace on the device")
''',
r'''    ap.add_argument("--trace-chunk", dest="trace_chunk", type=int, default=1024, help="v6e: the column block of every trace on the device")
    ap.add_argument("--prop-chunk", dest="prop_chunk", type=int, default=0, help="[v6r perf] PG-A2: the column chunk of the sea state's propagation alone (0: --chunk, v6q's)")   # [v6r perf]
    ap.add_argument("--v6r-share-traces", dest="v6r_share_traces", default="on", choices=["on", "off"], help="[v6r perf] PG-A1: one B.apply per block shared by the traces of one operator on one state, the transfers deferred (bitwise); off: v6q's calls")   # [v6r perf]
    ap.add_argument("--v6r-full-cache", dest="v6r_full_cache", default="on", choices=["on", "off"], help="[v6r perf] PG-A3: the checkpoint's full-state assembly reused by the probes, the columns and the live reference at the same clock (bitwise); off: v6q's assemblies")   # [v6r perf]
'''),
("HistRun.__init__: the switches",
r'''        TRACE_CHUNK[0] = max(1, int(getattr(a, "trace_chunk", 1024)))
''',
r'''        TRACE_CHUNK[0] = max(1, int(getattr(a, "trace_chunk", 1024)))
        PERF["share"] = (getattr(a, "v6r_share_traces", "on") == "on"); PERF["full_cache"] = (getattr(a, "v6r_full_cache", "on") == "on")   # [v6r perf]
        if int(getattr(a, "prop_chunk", 0) or 0) < 0:   # [v6r perf]
            raise ValueError("--prop-chunk must be 0 (= --chunk) or positive")   # [v6r perf]
'''),
("step_record: the kept full columns dropped",
r'''        rec = {"k": k, "t": t0, "per_channel": {}}
        for c in self.chs:
''',
r'''        rec = {"k": k, "t": t0, "per_channel": {}}
        for c_ in self.chs:   # [v6r perf] PG-A3: the clock's kept full columns dropped before the state moves
            c_._v6r_full = None   # [v6r perf]
        for c in self.chs:
'''),
("step_record (full route): the sub-step's jump",
r'''                    jump += tr_blocked(Bdh, c.Phi)
                    for lab in chis_x:                                           # v6n (b): the sub-step's jump at the ledger cuts
                        jump_x[lab] += tr_blocked(Bdh, c.Phi, chis_x[lab])
''',
r'''                    tv_ = tr_multi(Bdh, c.Phi, [None] + [chis_x[lab] for lab in chis_x])   # [v6r perf] PG-A1
                    jump += tv_[0]   # [v6r perf]
                    for i_, lab in enumerate(chis_x):   # [v6r perf] v6n (b): the sub-step's jump at the ledger cuts
                        jump_x[lab] += tv_[1 + i_]   # [v6r perf]
'''),
("step_record (full route): the traces at the sub-step's start",
r'''                Eo0 = tr_blocked(Bh, c.Phi, chi)
                Ex0 = {lab: tr_blocked(Bh, c.Phi, chis_x[lab]) for lab in chis_x}      # v6n (b)
''',
r'''                tv_ = tr_multi(Bh, c.Phi, [chi] + [chis_x[lab] for lab in chis_x])   # [v6r perf] PG-A1
                Eo0 = tv_[0]   # [v6r perf]
                Ex0 = {lab: tv_[1 + i_] for i_, lab in enumerate(chis_x)}   # [v6r perf] v6n (b)
'''),
("step_record (full route): the propagation chunk",
r'''                c.Phi, K_, tl = propagate(Bh, c.Phi, sb - sa, a.cheb_tol, a.chunk)
''',
r'''                c.Phi, K_, tl = propagate(Bh, c.Phi, sb - sa, a.cheb_tol, int(getattr(a, "prop_chunk", 0) or 0) or a.chunk)   # [v6r perf] PG-A2
'''),
("step_record (full route): the traces at the sub-step's end",
r'''                Eo1 = tr_blocked(Bh, c.Phi, chi)
                flow += Eo1 - Eo0
                for lab in chis_x:                                               # v6n (b): the exact flow at the ledger cuts over the sub-step
                    flow_x[lab] += tr_blocked(Bh, c.Phi, chis_x[lab]) - Ex0[lab]
''',
r'''                tv_ = tr_multi(Bh, c.Phi, [chi] + [chis_x[lab] for lab in chis_x])   # [v6r perf] PG-A1
                Eo1 = tv_[0]   # [v6r perf]
                flow += Eo1 - Eo0
                for i_, lab in enumerate(chis_x):   # [v6r perf] v6n (b): the exact flow at the ledger cuts over the sub-step
                    flow_x[lab] += tv_[1 + i_] - Ex0[lab]   # [v6r perf]
'''),
("step_record (full route): the jump into the next record",
r'''                jump += tr_blocked(Bdh, c.Phi)
                for lab in chis_x:                                               # v6n (b): the jump at the ledger cuts into the next record's h
                    jump_x[lab] += tr_blocked(Bdh, c.Phi, chis_x[lab])
''',
r'''                tv_ = tr_multi(Bdh, c.Phi, [None] + [chis_x[lab] for lab in chis_x])   # [v6r perf] PG-A1
                jump += tv_[0]   # [v6r perf]
                for i_, lab in enumerate(chis_x):   # [v6r perf] v6n (b): the jump at the ledger cuts into the next record's h
                    jump_x[lab] += tv_[1 + i_]   # [v6r perf]
'''),
("w1_step_record_sector: the sub-step's jump",
r'''                jump += 2.0 * tr_blocked(Bd, c.Phi) - W1S.trace_diag(dh)                                                          # v6o W1
                for lab in chis_x:                                                                                                # v6o W1
                    jump_x[lab] += 2.0 * tr_blocked(Bd, c.Phi, masks_s[lab]) - W1S.trace_diag(dh, masks_np[lab])                   # v6o W1
                del Bd; hp_cur = hp                                                                                               # v6o W1
''',
r'''                tv_ = tr_multi(Bd, c.Phi, [None] + [masks_s[lab] for lab in chis_x])   # [v6r perf] PG-A1
                jump += 2.0 * tv_[0] - W1S.trace_diag(dh)   # [v6r perf] v6o W1
                for i_, lab in enumerate(chis_x):   # [v6r perf] v6o W1
                    jump_x[lab] += 2.0 * tv_[1 + i_] - W1S.trace_diag(dh, masks_np[lab])   # [v6r perf] v6o W1
                del Bd; hp_cur = hp                                                                                               # v6o W1
'''),
("w1_step_record_sector: the traces at the sub-step's start",
r'''            Eo0 = 2.0 * tr_blocked(Bp, c.Phi, chi_s) - trh_chi                                                                    # v6o W1
            Ex0 = {lab: 2.0 * tr_blocked(Bp, c.Phi, masks_s[lab]) - trh_x[lab] for lab in chis_x}                                 # v6o W1
''',
r'''            tv_ = tr_multi(Bp, c.Phi, [chi_s] + [masks_s[lab] for lab in chis_x])   # [v6r perf] PG-A1
            Eo0 = 2.0 * tv_[0] - trh_chi   # [v6r perf] v6o W1
            Ex0 = {lab: 2.0 * tv_[1 + i_] - trh_x[lab] for i_, lab in enumerate(chis_x)}   # [v6r perf] v6o W1
'''),
("w1_step_record_sector: the propagation chunk",
r'''            c.Phi, K_, tl = propagate(Bp, c.Phi, sb - sa, a.cheb_tol, a.chunk)                                                    # v6o W1
''',
r'''            c.Phi, K_, tl = propagate(Bp, c.Phi, sb - sa, a.cheb_tol, int(getattr(a, "prop_chunk", 0) or 0) or a.chunk)   # [v6r perf] PG-A2; v6o W1
'''),
("w1_step_record_sector: the traces at the sub-step's end",
r'''            Eo1 = 2.0 * tr_blocked(Bp, c.Phi, chi_s) - trh_chi                                                                    # v6o W1
            flow += Eo1 - Eo0                                                                                                     # v6o W1
            for lab in chis_x:                                                                                                    # v6o W1
                flow_x[lab] += (2.0 * tr_blocked(Bp, c.Phi, masks_s[lab]) - trh_x[lab]) - Ex0[lab]                                # v6o W1
''',
r'''            tv_ = tr_multi(Bp, c.Phi, [chi_s] + [masks_s[lab] for lab in chis_x])   # [v6r perf] PG-A1
            Eo1 = 2.0 * tv_[0] - trh_chi   # [v6r perf] v6o W1
            flow += Eo1 - Eo0                                                                                                     # v6o W1
            for i_, lab in enumerate(chis_x):   # [v6r perf] v6o W1
                flow_x[lab] += (2.0 * tv_[1 + i_] - trh_x[lab]) - Ex0[lab]   # [v6r perf] v6o W1
'''),
("w1_step_record_sector: the jump into the next record",
r'''            jump += 2.0 * tr_blocked(Bd, c.Phi) - W1S.trace_diag(dh)                                                              # v6o W1
            for lab in chis_x:                                                                                                    # v6o W1
                jump_x[lab] += 2.0 * tr_blocked(Bd, c.Phi, masks_s[lab]) - W1S.trace_diag(dh, masks_np[lab])                       # v6o W1
            del Bd                                                                                                                # v6o W1
''',
r'''            tv_ = tr_multi(Bd, c.Phi, [None] + [masks_s[lab] for lab in chis_x])   # [v6r perf] PG-A1
            jump += 2.0 * tv_[0] - W1S.trace_diag(dh)   # [v6r perf] v6o W1
            for i_, lab in enumerate(chis_x):   # [v6r perf] v6o W1
                jump_x[lab] += 2.0 * tv_[1 + i_] - W1S.trace_diag(dh, masks_np[lab])   # [v6r perf] v6o W1
            del Bd                                                                                                                # v6o W1
'''),
("d13_load_reference: the live columns",
r'''            Phi = W1S.assemble_full_state(c.Phi) if getattr(c, "sector_prop", False) else c.Phi                                  # v6o W1: the full columns from the + sector block
''',
r'''            Phi = v6r_full(c, self.k)   # [v6r perf] PG-A3; v6o W1: the full columns from the + sector block
'''),
("save_columns: the columns",
r'''            X_ = W1S.assemble_full_state(c.Phi) if getattr(c, "sector_prop", False) else c.Phi                                   # v6o W1: the full columns from the + sector block
''',
r'''            X_ = v6r_full(c, k)   # [v6r perf] PG-A3; v6o W1: the full columns from the + sector block
'''),
("checkpoint_lean: the assembly kept",
r'''            Phi_rd = W1S.assemble_full_state(Phi) if sp_ else Phi                                                                   # v6o W1: the full columns for the D13 readouts, the references and the columns
''',
r'''            Phi_rd = W1S.assemble_full_state(Phi) if sp_ else Phi                                                                   # v6o W1: the full columns for the D13 readouts, the references and the columns
            v6r_full_keep(c, k, Phi_rd, bool(sp_ and PERF["full_cache"] and Phi is c.Phi and (k in self.probe_k or k in self.col_k)))   # [v6r perf] PG-A3: kept (a private copy) when the probes or the columns follow at this clock
'''),
("d13_probes_if_due: the probes' columns",
r'''            Phi_now = W1S.assemble_full_state(c.Phi) if getattr(c, "sector_prop", False) else c.Phi                               # v6o W1: the full columns for the probes
''',
r'''            Phi_now = v6r_full(c, k)   # [v6r perf] PG-A3; v6o W1: the full columns for the probes
'''),
("run: the counters",
r'''        if self.frozen:
            self.run_frozen()
        self.finish()
''',
r'''        if self.frozen:
            self.run_frozen()
        log("v6r perf: shared-trace calls %d (B.apply per block saved %d); full-state assemblies kept %d, reused %d, assembled at a consumer %d; switches share %s, full cache %s, prop chunk %s (process totals)" % (   # [v6r perf]
            PERF["multi_calls"], PERF["applies_saved"], PERF["full_kept"], PERF["full_hits"], PERF["full_misses"], PERF["share"], PERF["full_cache"], int(getattr(self.a, "prop_chunk", 0) or 0) or self.a.chunk))   # [v6r perf]
        self.finish()
'''),
]


def build(text):
    """the forward replacements on the source text; every OLD must occur exactly once. Returns the built text."""
    for name, old, new in REPLACEMENTS:
        n = text.count(old)
        if n != 1:
            raise SystemExit("build_page_history_v6r: the OLD string of '%s' occurs %d times in the source (1 required): the source is not page_history_v6q.py" % (name, n))
        text = text.replace(old, new)
    return text


def reverse(text):
    """the reverse replacements (NEW -> OLD, in reverse order); every NEW must occur exactly once."""
    for name, old, new in reversed(REPLACEMENTS):
        n = text.count(new)
        if n != 1:
            raise SystemExit("build_page_history_v6r: the NEW text of '%s' occurs %d times in the built file (1 required)" % (name, n))
        text = text.replace(new, old)
    return text


def md5(b):
    return hashlib.md5(b).hexdigest()


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "page_history_v6q.py")
    dst = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(os.path.abspath(src)), "page_history_v6r.py")
    raw = open(src, "rb").read()
    if md5(raw) != SRC_MD5:
        raise SystemExit("build_page_history_v6r: the source md5 is %s, not v6q's %s" % (md5(raw), SRC_MD5))
    text = raw.decode("utf-8")
    out = build(text)
    back = reverse(out)
    if back != text:
        raise SystemExit("build_page_history_v6r: the reverse replacements do not restore the source (a NEW text overlaps another)")
    srcset = set(text.splitlines())
    unmarked = [l_ for l_ in out.splitlines() if l_ not in srcset and MARK not in l_]
    if unmarked:
        raise SystemExit("build_page_history_v6r: %d new lines without the marker, the first: %r" % (len(unmarked), unmarked[0][:200]))
    marked = sum(1 for l_ in out.splitlines() if MARK in l_)
    if os.path.exists(dst) and open(dst, "rb").read() != out.encode("utf-8"):
        raise SystemExit("build_page_history_v6r: %s exists with other contents: not overwritten" % dst)
    with open(dst, "wb") as fh:
        fh.write(out.encode("utf-8"))
    py_compile.compile(dst, doraise=True)
    print("built %s from %s: %d replacements, %d marked lines, every new line marked; source md5 %s; built md5 %s; the reverse replacements restore the source byte for byte; compiled" % (
        dst, src, len(REPLACEMENTS), marked, md5(raw), md5(out.encode("utf-8"))))


if __name__ == "__main__":
    main()
