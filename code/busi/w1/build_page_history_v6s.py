#!/usr/bin/env python
"""build_page_history_v6s.py (busi lane, inbox #309: the Page-curve verification plan final4/PAGE_VERIFY_PLAN.md, owner approved; reviewer A's
LEAD_reviewa_page_verify_routes_v1.md A2, A3, B2, B6): writes page_history_v6s.py from page_history_v6q.py (md5 b4f4ea69) by exact string
replacements in build_page_history_v6r.py's form. Each OLD is copied exactly from page_history_v6q.py and must occur exactly once (else the build
aborts); each NEW keeps its OLD's text and adds or changes lines so that replacing every NEW by its OLD restores the source byte for byte (checked
here). Every inserted or changed line carries the marker "[v6s verify]" (checked here). The new code lives in page_interior_v6s.py (a new module
imported only by v6s); no shared file is changed. v6s is v6q plus three options, each off by default (at the defaults every number is v6q's; the
summary's "args" gains the new keys):
  1. --kappa-zero (A2's conformal control): the channel's mass term kappa/R switched off (ChannelOps.kappa_mass = 0 in chain_sparse, the
     field-built banded operator and its check, the D13 readout's collar-split spinors); the label kappa, the tags and the weights 16 kappa kept.
  2. --interior-readout {off, on, check} (A3, B6): at every checkpoint the excess 2 [S_I(C_+) - S_I(P_{-,+})] on the interior block of each cut,
     the vacuum's block by the contour integral (no eigendecomposition); check adds the dense vacuum's interior and exterior blocks, the state's
     exterior block and the production readout's numbers beside (the purity identity's gate).
  3. --lean-interior (B6, B2's 32k chain): the + sector state by the production's sub-steps without the per-step traces, the readouts by the
     interior side only (page_interior_v6s.run_lean_interior), with --hold-fields-from T (B2: the record at T held to --T-end-rs; also usable
     without the lean mode) and the held records in Chebyshev steps of --hold-step-rs.
Usage (box): python build_page_history_v6s.py [page_history_v6q.py path] [page_history_v6s.py path]   (defaults: v6q beside this script, v6s beside v6q)."""
import hashlib
import os
import py_compile
import sys

MARK = "[v6s verify]"
SRC_MD5 = "b4f4ea69c199ac9785e676e060f36a15"

REPLACEMENTS = [
("docstring",
r'''#!/usr/bin/env python
"""page_history_v6o.py (reviewer A, the busi lane, W1 of REVIEWA_REVIEW_web40i.md section 4; code defects 10 to 13 of its section 5): page_history_v6n.py
''',
r'''#!/usr/bin/env python
"""page_history_v6s.py [v6s verify] = page_history_v6q.py (md5 b4f4ea69) built by build_page_history_v6s.py (busi lane, inbox #309, the Page-curve [v6s verify]
verification plan; new code in page_interior_v6s.py). Three options, each off by default (bitwise v6q at the defaults): --kappa-zero (A2's [v6s verify]
conformal control: the mass term kappa/R off, the label kept); --interior-readout {on, check} (A3/B6: the excess on the interior block by purity, [v6s verify]
the vacuum block by the contour); --lean-interior (B6/B2: the 32k chain's lean propagation, interior readouts only) with --hold-fields-from T [v6s verify]
(B2: the record at T held to --T-end-rs) and --hold-step-rs. [v6s verify]
[v6s verify] v6q's docstring follows: page_history_v6o.py (reviewer A, the busi lane, W1 of REVIEWA_REVIEW_web40i.md section 4; code defects 10 to 13 of its section 5): page_history_v6n.py
'''),
("the import",
r'''import page_w1_sector_v1d as W1S  # noqa: E402''',
r'''import page_interior_v6s as IR  # noqa: E402  [v6s verify] the interior readout, the lean mode, the hold, the mass view (a new module, imported only here)
import page_w1_sector_v1d as W1S  # noqa: E402'''),
("ChannelOps: the mass coefficient",
r'''    def __init__(self, kappa, R, device):
        self.kappa = int(kappa); self.w = 16.0 * self.kappa
        self.R = R; self.n2 = 2 * len(R); self.device = device
        self.h0, self.Q = OPS.chain_sparse(R, 0.0, float(kappa))
''',
r'''    def __init__(self, kappa, R, device, kappa_zero=False):   # [v6s verify] kappa_zero (--kappa-zero, A2's conformal control): the mass term kappa/R off, the label kappa kept
        self.kappa = int(kappa); self.w = 16.0 * self.kappa
        self.R = R; self.n2 = 2 * len(R); self.device = device
        self.kappa_mass = 0.0 if kappa_zero else float(kappa)   # [v6s verify] the coefficient of the on-site (kappa/R) sigma_y; float(kappa) at the defaults, v6q's value
        self.h0, self.Q = OPS.chain_sparse(R, 0.0, self.kappa_mass)   # [v6s verify] (v6q: float(kappa))
'''),
("HistRun: the hold",
r'''        self.tau = H.tau
''',
r'''        self.tau = H.tau
        self.hold_info = None   # [v6s verify]
        if getattr(a, "hold_fields_from", None) is not None:   # [v6s verify] B2: the record at --hold-fields-from held (copies at the spacing tau) to --T-end-rs, before the clocks are set
            self.hold_info = IR.hold_extend(H, float(a.hold_fields_from) * self.rs_nom, float(a.T_end_rs) * self.rs_nom, self.tau, log)   # [v6s verify]
'''),
("HistRun: the channels with the mass coefficient",
r'''        self.chs = [HistChannel(ChannelOps(k, H.R, self.dev)) for k in self.channels]
''',
r'''        self.chs = [HistChannel(ChannelOps(k, H.R, self.dev, kappa_zero=bool(getattr(a, "kappa_zero", False)))) for k in self.channels]   # [v6s verify] --kappa-zero
        if getattr(a, "kappa_zero", False):   # [v6s verify]
            log("v6s verify --kappa-zero: channel(s) %s run with the mass term kappa/R switched off (A2's conformal control: h = (1/2){N, h_0} + (1/2){v, Q} at kappa = 0 on the same fields, cuts and propagation); the labels, tags and weights 16 kappa keep kappa; the continuum comparator L_H is kappa's, not the control's" % self.channels)   # [v6s verify]
'''),
("start: the field-built operator's check with the mass view",
r'''fb = W1S.fast_banded_matches(c.ops, N, v, self.dev)''',
r'''fb = W1S.fast_banded_matches(IR.mass_view(c.ops), N, v, self.dev)  # [v6s verify] the channel's mass coefficient (the channel object itself at the defaults)'''),
("d13_init_channel: the readout's spinors at kappa = 0",
r'''            c.d13.ref_loader = (lambda k_ref, c=c: self.d13_load_reference(c, k_ref))
''',
r'''            c.d13.ref_loader = (lambda k_ref, c=c: self.d13_load_reference(c, k_ref))
            if getattr(c.ops, "kappa_mass", float(c.ops.kappa)) != float(c.ops.kappa):   # [v6s verify] --kappa-zero: the readout's local spinors (the collar split's D(q) = sin q sigma_x + (kappa/R_c) sigma_y) at the chain's mass 0
                c.d13.kappa = 0   # [v6s verify]
'''),
("d13_record_steps: the field-built operator with the mass coefficient",
r'''W1S.FastBanded(N, v, c.ops.kappa, c.ops.R, self.dev), sb - sa) for (sa, sb, mode, (N, v)) in sub]''',
r'''W1S.FastBanded(N, v, c.ops.kappa_mass, c.ops.R, self.dev), sb - sa) for (sa, sb, mode, (N, v)) in sub]  # [v6s verify] the mass coefficient (FastBanded takes float(kappa): v6q's at the defaults)'''),
("checkpoint_lean: the check with the mass view",
r'''not W1S.fast_banded_matches(c.ops, N, v, self.dev)["ok"]:''',
r'''not W1S.fast_banded_matches(IR.mass_view(c.ops), N, v, self.dev)["ok"]:  # [v6s verify] the mass view;'''),
("checkpoint_lean: the interior readout",
r'''            cr.release_vacuum()
''',
r'''            if getattr(self.a, "interior_readout", "off") != "off" and sp_:   # [v6s verify] item 2: the interior-side excess by purity with the contour vacuum block; 'check' adds the dense and exterior-side comparisons
                pc["interior"] = IR.interior_readout(Phi, hp, cr.cuts, np.asarray(c.ops.R, float), self.dev, threads=(self.a.interior_threads or self.a.threads), hs=self.a.interior_hs, U=self.a.interior_U,   # [v6s verify]
                                                     floor=cr.w1_floor, vac=vac, sc_cuts=(pc.get("d13") or {}).get("cuts"), check=(self.a.interior_readout == "check"), log=log, kappa=c.ops.kappa, trs=t / self.rs_nom)   # [v6s verify]
            cr.release_vacuum()
'''),
("parse: the options",
r'''    ap.add_argument("--record-smooth", dest="record_smooth", type=int, default=1, help="v6p T1 DIAGNOSTIC, never a production row: the engine's applied fields replaced by their centred moving average over this many records (1: off, bitwise)")  # v6p T1
''',
r'''    ap.add_argument("--record-smooth", dest="record_smooth", type=int, default=1, help="v6p T1 DIAGNOSTIC, never a production row: the engine's applied fields replaced by their centred moving average over this many records (1: off, bitwise)")  # v6p T1
    ap.add_argument("--kappa-zero", dest="kappa_zero", action="store_true", help="v6s verify (A2): the channel's mass term kappa/R switched off (the conformal control); the label kappa, the tags and the weights kept")   # [v6s verify]
    ap.add_argument("--interior-readout", dest="interior_readout", default="off", choices=["off", "on", "check"], help="v6s verify (A3/B6): the interior-side excess at every checkpoint (the contour vacuum block); check: the dense vacuum's interior and exterior blocks and the state's exterior block beside")   # [v6s verify]
    ap.add_argument("--interior-hs", dest="interior_hs", type=float, default=0.3, help="v6s verify: the contour's trapezoid step in s = ln t")   # [v6s verify]
    ap.add_argument("--interior-U", dest="interior_U", type=float, default=37.0, help="v6s verify: the contour's tails: s from ln(delta) - U to ln(w) + U")   # [v6s verify]
    ap.add_argument("--interior-threads", dest="interior_threads", type=int, default=0, help="v6s verify: host threads of the contour's banded solves (0: --threads)")   # [v6s verify]
    ap.add_argument("--lean-interior", dest="lean_interior", action="store_true", help="v6s verify (B6/B2): the lean interior mode: the + sector state by the production's sub-steps (no per-step traces), held records in Chebyshev steps of --hold-step-rs, readouts by the interior side only every --interior-every-rs")   # [v6s verify]
    ap.add_argument("--interior-every-rs", dest="interior_every_rs", type=float, default=10.0, help="v6s verify: the lean mode's readout clocks (r_s)")   # [v6s verify]
    ap.add_argument("--impurity-every-rs", dest="impurity_every_rs", type=float, default=0.0, help="v6s verify: the lean mode's trace-norm impurity (an n_+ x n_+ Gram) at most this often (0: every readout)")   # [v6s verify]
    ap.add_argument("--lean-flux", dest="lean_flux", default="on", choices=["on", "off"], help="v6s verify: the lean mode's collar current Tr[J C] once per record (or held step)")   # [v6s verify]
    ap.add_argument("--init-eig-device", dest="init_eig_device", default="auto", choices=["auto", "cpu", "device"], help="v6s verify: the lean mode's initial eigendecomposition of h_+ (auto: the host above 12000 sites)")   # [v6s verify]
    ap.add_argument("--init-cache", dest="init_cache", default=None, help="v6s verify: the lean mode's initial Phi_+ cached at this path (loaded when its h_+ md5 matches, else written)")   # [v6s verify]
    ap.add_argument("--hold-fields-from", dest="hold_fields_from", type=float, default=None, help="v6s verify (B2): the fields of the record at this clock (r_s) held to --T-end-rs (copies at the spacing tau)")   # [v6s verify]
    ap.add_argument("--hold-step-rs", dest="hold_step_rs", type=float, default=1.0, help="v6s verify: the lean mode's Chebyshev step over held records (r_s)")   # [v6s verify]
'''),
("parse: the checks",
r'''    if a.ledger_anchor_rs is not None and a.freeze_at_rs is not None:
''',
r'''    if (a.kappa_zero or a.interior_readout != "off" or a.lean_interior or a.hold_fields_from is not None) and a.collector == "on":   # [v6s verify]
        ap.error("the v6s verify options (--kappa-zero, --interior-readout, --lean-interior, --hold-fields-from) need --collector off")   # [v6s verify]
    if a.hold_fields_from is not None and (a.T_end_rs is None or a.record_ext or a.follow or a.T_end_rs < a.hold_fields_from):   # [v6s verify]
        ap.error("--hold-fields-from needs --T-end-rs at or after it and no continuation (--record-ext, --follow)")   # [v6s verify]
    if a.interior_readout != "off" and a.sector == "off":   # [v6s verify]
        ap.error("--interior-readout reads the + sector state: not with --sector off")   # [v6s verify]
    if a.lean_interior:   # [v6s verify] the lean mode writes no D13 store, no columns and no ksum table, and never extends the channel sequence
        a.no_ksum = True; a.kappa_extend = False; a.no_columns = True; a.probe_from_rs = 1e9; a.columns_every_rs = 0.0   # [v6s verify]
    if a.ledger_anchor_rs is not None and a.freeze_at_rs is not None:
'''),
("run_one: the lean mode",
r'''    run = HistRun(a)
    if a.resume and os.path.exists(run.prefix + "_state.pt"):
''',
r'''    run = HistRun(a)
    if getattr(a, "lean_interior", False):   # [v6s verify] item 3: the lean interior mode (page_interior_v6s.run_lean_interior)
        IR.run_lean_interior(run, {"propagate": propagate, "tr_blocked": tr_blocked, "log": log, "T_START": T_START}, resume=bool(a.resume))   # [v6s verify]
    elif a.resume and os.path.exists(run.prefix + "_state.pt"):   # [v6s verify] (v6q: if)
'''),
]


def build(text):
    """the forward replacements on the source text; every OLD must occur exactly once. Returns the built text."""
    for name, old, new in REPLACEMENTS:
        n = text.count(old)
        if n != 1:
            raise SystemExit("build_page_history_v6s: the OLD string of '%s' occurs %d times in the source (1 required): the source is not page_history_v6q.py" % (name, n))
        text = text.replace(old, new)
    return text


def reverse(text):
    """the reverse replacements (NEW -> OLD, in reverse order); every NEW must occur exactly once."""
    for name, old, new in reversed(REPLACEMENTS):
        n = text.count(new)
        if n != 1:
            raise SystemExit("build_page_history_v6s: the NEW text of '%s' occurs %d times in the built file (1 required)" % (name, n))
        text = text.replace(new, old)
    return text


def md5(b):
    return hashlib.md5(b).hexdigest()


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "page_history_v6q.py")
    dst = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(os.path.abspath(src)), "page_history_v6s.py")
    raw = open(src, "rb").read()
    if md5(raw) != SRC_MD5:
        raise SystemExit("build_page_history_v6s: the source md5 is %s, not v6q's %s" % (md5(raw), SRC_MD5))
    text = raw.decode("utf-8")
    out = build(text)
    back = reverse(out)
    if back != text:
        raise SystemExit("build_page_history_v6s: the reverse replacements do not restore the source (a NEW text overlaps another)")
    srcset = set(text.splitlines())
    unmarked = [l_ for l_ in out.splitlines() if l_ not in srcset and MARK not in l_]
    if unmarked:
        raise SystemExit("build_page_history_v6s: %d new lines without the marker, the first: %r" % (len(unmarked), unmarked[0][:200]))
    marked = sum(1 for l_ in out.splitlines() if MARK in l_)
    if os.path.exists(dst) and open(dst, "rb").read() != out.encode("utf-8"):
        raise SystemExit("build_page_history_v6s: %s exists with other contents: not overwritten" % dst)
    with open(dst, "wb") as fh:
        fh.write(out.encode("utf-8"))
    py_compile.compile(dst, doraise=True)
    print("built %s from %s: %d replacements, %d marked lines, every new line marked; source md5 %s; built md5 %s; the reverse replacements restore the source byte for byte; compiled" % (
        dst, src, len(REPLACEMENTS), marked, md5(raw), md5(out.encode("utf-8"))))


if __name__ == "__main__":
    main()
