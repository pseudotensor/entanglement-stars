#!/usr/bin/env python
"""build_page_history_v6o_v1d.py (reviewer A, the busi lane, W1; v1d of build_page_history_v6o.py, W1_fix_v1d.md): page_history_v6o.py from
page_history_v6n.py and page_d13_readout_v1d.py from page_d13_readout_v1b.py by exact replacement lists, in the form build_page_history_v6n.py used: every
OLD must occur exactly once in its source (the build aborts otherwise), every changed code line carries "v6o W1" (v6o) or "v1c"/"v1d" (the readout), the
v1b, v1c and v1d changes "v1b", "v1c" and "v1d" beside; page_w1_sector_v1d.py (the module both import as W1S) is copied beside them. The sources are not
touched. Both outputs are byte-compiled after the build; the md5 of every file is printed. The v1d changes against the v1c builder (W1_fix_v1d.md
section 3): the imports of page_w1_sector_v1d and the readout's name page_d13_readout_v1d.py; the channel carries T_init, the state's impurity at its first
checkpoint (set there, saved and restored), passed to the propagation floor whose a priori bound now has the construction allowance 4 n_+^(3/2) eps, so
that the premise is well posed at the first checkpoint (v1c's bound was 0 there and every run printed a FLOOR WARNING against the state's own
construction); the FLOOR WARNING prints the construction term; pc["w1"] holds T_init; the summary's floor label. Every other replacement is the v1c
builder's, verbatim.

Usage (pod 7): cd <internal>/page && python w1/build_page_history_v6o_v1d.py --src-v6n code_d13/page_history_v6n.py
               --src-v1b code_d13/page_d13_readout_v1b.py --sector-src w1/page_w1_sector_v1d.py --out-dir code_d13
       (box):   the same on the rsynced copies (RUN_W1_v1d.md section 1).
--check-only: verify the OLD anchors and print what would be written, writing nothing."""
import argparse
import hashlib
import os
import py_compile
import shutil
import sys

# ----------------------------------------------------------------------------------------------------- v6n -> v6o
REPL_V6O = [
# 1. the docstring header
(r'''"""page_history_v6n.py (reviewer A for the solve builder, the busi lane; reorientation 4, the solve entry's item 2): page_history_v6m.py''',
 r'''"""page_history_v6o.py (reviewer A, the busi lane, W1 of REVIEWA_REVIEW_web40i.md section 4; code defects 10 to 13 of its section 5): page_history_v6n.py
built by build_page_history_v6o's exact replacement list (every changed line carries "v6o W1"), with page_d13_readout_v1d.py (v1b plus the halved
entropies and the tail-majorant rule) and the new module page_w1_sector_v1d.py (v1b: W1_fix_v1b.md, the vacuum route into every channel readout's
configuration copy, the state's representation chosen after the sector decision, the identity audit's floor from Tr|h|; v1c: W1_fix_v1c.md, the
propagation's floor beside every excess; v1d: W1_fix_v1d.md, the construction allowance in the floor's a priori bound). Changes against v6n:
(1) the propagation (defect 11): --interp linear (the default, the production) propagates every record in --interp-sub sub-steps (8: one lattice
    unit) with the fields interpolated linearly between the record and the next and taken at the sub-step's midpoint, so the records' jump of h
    becomes a ramp; --interp none is v6n's piecewise-constant propagation (the control, v6n's code path), --interp cubic v6m's Catmull-Rom
    everywhere; the probes' back-propagation takes the same sub-steps (on the field-built banded operator, verified bitwise against the CSR's);
    the Chebyshev degree per record is summed and stored (cheb_K_sum: the cost statement).
(2) the held vacuum at every checkpoint and the halving: with --sector auto (the default) the massless sector symmetry [S_stag, h] = 0 and the
    pairing K h K = -h (page_w1_sector_v1d) give P_-(h) from the eigendecomposition of the + sector block h_+ (n x n, 1/8 of the dense cost), so
    the held vacuum's exterior-cut entropy, the excess S_X(C) - S_X(P_-(h)), the cross-cut bound t_X and ||C - P_-||_1 are read at every
    checkpoint and every cut, every dense entropy at half the dimension (S_X = 2 S_Xc(C_+)); with --sector-propagate on (the default) the state
    itself is propagated as its + sector block Phi_+ (n x n_+) under the banded h_+ (half-bandwidth 2; about 1/5 of the full apply's traffic), the
    - sector following exactly as conj(1_+ - C_+), and the full columns are assembled at every checkpoint for the D13 readouts, the references and
    the saved columns (one QR); used only when the operator's and the initial state's defects are below --sector-tol (printed; otherwise v6n's dense
    route); the - sector is computed beside at the check clocks (--sector-check-every-rs, 0 = every checkpoint) and the pairing defects print.
(3) the formed-energy majorant (defect 10): E_kappa = Tr[|h| (C - P_-)^2] = Tr[h (C - P_-)] = E_inv identically (page_w1_sector_v1d's identity), so the
    per-channel E_inv of every row is the majorant's term (times 16 kappa); the kappa sequence stops by page_d13_readout_v1d's tail-majorant rule
    (the energy terms falling geometrically over the last 3 channels, the excess's tail bounded by the entropy per formed energy times the energy
    tail, the collar flow's by its own majorant; --kappa-hard-frac the tolerance), never by the last channel's share (printed beside as retired).
(4) the audits (defect 13): every record's sparse h is checked for the pairing premise max|h + conj h| and max|[S_stag, h]| (exact); every banded
    eigensolve for Tr min(h, 0) = -Tr|h|/2; every dense vacuum for E_kappa's spectral form against E_inv, the occupation pairing n^- = 1 - n^+ and
    the state's off-sector weight; the field-built banded operator against the CSR's at every checkpoint.
(5) --lean (the control pass): no columns, no shells, no probes, no cross-cut norms; the entropies, the excess, E_inv and the collar flow only.
(6) labels: S_X prints as the raw exterior-cut entropy (regulated, not the escaping entropy) with the held vacuum's beside; the readout is the excess (Y13).
(7) v1c (W1_fix_v1c.md) and v1d (W1_fix_v1d.md): the propagation's floor beside every excess. The state's impurity ||Phi^dagger Phi - 1||_1 (the
    Gram's eigenvalues; the trace-norm distance of C from its range projector, the whole propagation error on a held record) is measured at every
    checkpoint and compared with the a priori bound 4 n_+^(3/2) eps (the construction allowance: the eigensolve's orthonormality defect before any
    sub-step, which v1c's bound omitted so that the first checkpoint failed the premise against the state's own construction, v1d) + 2 n_+ sum of the
    Chebyshev tails + the rounding allowance 16 K_max eps per sub-step per column (a FLOOR WARNING where it exceeds it); the impurity at the first
    checkpoint, T_init, is carried with the state (saved and restored) and the growth since it printed; every cut's excess prints with its low floor
    2 m h2(T_meas/m) (the impurity's entropy by the Lidskii pairing and the binary entropy's modulus, m the paired eigenvalues), its high floor
    2 m h2(T_cheb/m), the rounding term 2 m h2(8 eps) and the resolution (resolved above the high floor, marginal between, unresolved below the low
    floor); the sub-steps are counted (n_substeps) and the propagator's Gershgorin bound stored (h_bound) for the E_inv premise 2 w T. The held
    gate's tolerance is this floor, derived per channel, checkpoint and cut.
v6n's docstring follows.

page_history_v6n.py (reviewer A for the solve builder, the busi lane; reorientation 4, the solve entry's item 2): page_history_v6m.py'''),
# 2. the import (v1b: page_w1_sector_v1b; v1c: page_w1_sector_v1c; v1d: page_w1_sector_v1d and the readout page_d13_readout_v1d)
(r'''import page_d13_readout_v1b as D13  # noqa: E402  (v6n_b: v1b adds the cross-cut block trace norms, EVOLVEDCUT EC1)''',
 r'''import page_d13_readout_v1d as D13  # noqa: E402  (v6o W1 v1d: v1d = v1b plus the halved entropies, the tail-majorant rule and the propagation's floor)
import page_w1_sector_v1d as W1S  # noqa: E402  (v6o W1 v1d: the sector symmetry, the pairing audits, the interpolation sub-steps, the majorant, the propagation's floor with the construction allowance; W1_fix_v1d.md)'''),
# 3. the channel's representation flag, the sub-step count and the first checkpoint's impurity
(r'''        self.d13 = None                                 # v6n: the channel's D13 readout (its caches and source modes)''',
 r'''        self.d13 = None                                 # v6n: the channel's D13 readout (its caches and source modes)
        self.sector_prop = False                        # v6o W1: True when Phi is the + sector block Phi_+ (n x n_+), propagated by h_+
        self.n_substeps = 0                             # v6o W1 v1c: the sub-steps taken so far (the propagation floor's count)
        self.T_init = None                              # v6o W1 v1d: the state's impurity at its first checkpoint (the construction), the growth's origin; saved and restored'''),
# 4. the parser's W1 arguments
(r'''    ap.add_argument("--memory-figure-only", dest="memory_figure_only", action="store_true")''',
 r'''    ap.add_argument("--memory-figure-only", dest="memory_figure_only", action="store_true")
    ap.add_argument("--interp", default="linear", choices=["linear", "none", "cubic"], help="v6o W1: the propagation: linear sub-steps between records (production), none = v6n's piecewise-constant (control), cubic = v6m's everywhere")  # v6o W1
    ap.add_argument("--interp-sub", dest="interp_sub", type=int, default=8, help="v6o W1: sub-steps per record for --interp linear (8: one lattice unit at tau 8)")  # v6o W1
    ap.add_argument("--sector", default="auto", choices=["auto", "on", "off"], help="v6o W1: the massless sector halving of every dense entropy and of the vacuum; auto uses it when the operator and the initial state respect S_stag")  # v6o W1
    ap.add_argument("--sector-propagate", dest="sector_propagate", default="on", choices=["on", "off"], help="v6o W1: propagate the + sector block Phi_+ under h_+ (on) or the full columns (off); only with the sector route")  # v6o W1
    ap.add_argument("--sector-tol", dest="sector_tol", type=float, default=1e-8, help="v6o W1: the state's off-sector weight ||P_- C P_+||_F above which the halving is not used")  # v6o W1
    ap.add_argument("--sector-check-every-rs", dest="sector_check_every_rs", type=float, default=10.0, help="v6o W1: the clocks at which the - sector and the cross-cut norms are computed beside the + (the pairing check); 0 = every checkpoint")  # v6o W1
    ap.add_argument("--audit", default="on", choices=["on", "off"], help="v6o W1: the per-record and per-eigensolve audits (defect 13)")  # v6o W1
    ap.add_argument("--lean", action="store_true", help="v6o W1: the control pass: no columns, shells, probes or cross-cut norms (the entropies, the excess, E_inv and the collar flow only)")  # v6o W1'''),
# 5. the tolerance's help
(r'''    ap.add_argument("--kappa-hard-frac", dest="kappa_hard_frac", type=float, default=0.01, help="v6n: the hard part's share of the total below which the sum stops")''',
 r'''    ap.add_argument("--kappa-hard-frac", dest="kappa_hard_frac", type=float, default=0.01, help="v6o W1: the tolerance of the tail majorant (the excess's and the collar flow's bounded tail over the partial sum) below which the sum stops; never the last channel's share")  # v6o W1'''),
# 6. the parser's checks and the lean mode
(r'''    if a.follow and a.cubic_window:
        ap.error("--cubic-window is not combined with --follow")''',
 r'''    if a.follow and a.cubic_window:
        ap.error("--cubic-window is not combined with --follow")
    if a.cubic_window and a.interp != "none":                                                                                      # v6o W1
        ap.error("--cubic-window (v6m's control) is combined with --interp none only")                                             # v6o W1
    if a.interp_sub < 1:                                                                                                           # v6o W1
        ap.error("--interp-sub must be at least 1")                                                                                # v6o W1
    if a.collector == "on" and a.sector != "off":                                                                                  # v6o W1
        ap.error("--sector off is required with --collector on (v6m's code paths hold the full columns)")                          # v6o W1
    if a.lean:                                                                                                                     # v6o W1: the control pass
        a.no_columns = True; a.shells = "off"; a.cut_norm = "off"; a.cut_norm_full = "off"; a.probe_from_rs = 1e9; a.columns_every_rs = 0.0   # v6o W1'''),
# 7. the W1 setup after d13_setup
(r'''        self.anchor = None; self.anchor_started = False; self.follow_end = None
        self.d13_setup()                                                          # v6n''',
 r'''        self.anchor = None; self.anchor_started = False; self.follow_end = None
        self.d13_setup()                                                          # v6n
        self.w1_setup()                                                           # v6o W1: the interpolation, the sector decision's defaults, the audits, the check clocks'''),
# 8. the initial state (v1b: the decision before the state's representation is fixed)
(r'''        for c in self.chs:
            h = c.ops.h_csr(N, v)
            e, V = OPS.eigh_csr(h, self.dev)
            neg = e < 0
            c.Phi = V[:, neg].clone()
            c.F = torch.zeros((c.ops.n2, 0), dtype=CD, device=self.dev)
            log("kappa %d: the ground state of h(T_0): %d occupied of %d, %d levels within 1e-12 of zero" % (c.ops.kappa, int(neg.sum()), c.ops.n2, int((e.abs() < 1e-12).sum())))
            if getattr(a, "collector", "on") == "off":                            # v6n: the T_0 vacuum (the ground state's own columns) for the first checkpoint
                self.d13_init_channel(c)
                c.d13.set_vacuum(float(H.T[self.k0]), self.k0, e.cpu().numpy(), c.Phi, "the ground state's eigendecomposition at T_0 (dense, v6m's)", float(e[neg].sum()))
            del V, e
            if self.dev.type == "cuda":
                torch.cuda.empty_cache()
        self.min_N_minus_absv_T0 = mnv''',
 r'''        for c in self.chs:
            h = c.ops.h_csr(N, v)
            op_aud = W1S.operator_audit(h)                                                                                         # v6o W1: the operator's sector and pairing premises at T_0
            use_sector = bool(getattr(a, "sector", "auto") != "off" and op_aud["ok"])                                                # v6o W1
            vac0 = None                                                                                                            # v6o W1 v1b
            if use_sector:                                                                                                         # v6o W1: the ground state by the sector route (1/8 of the dense eigh)
                vac0 = W1S.killing_vacuum_sector(h, self.dev)                                                                      # v6o W1
                e = torch.as_tensor(vac0["e"], dtype=RD, device=self.dev); neg = e < 0; V = None                                     # v6o W1
                Vfull0 = vac0["Vneg"]                                                                                              # v6o W1
                E_ref0 = vac0["E_ref"]; route0 = vac0["route"]                                                                     # v6o W1 v1b
                extra0 = {"e_plus": vac0["e_plus"], "V_plus": vac0["V_plus"], "n_neg_plus": vac0["n_neg_plus"], "n_pos_plus": vac0["n_pos_plus"], "audit_plus": vac0["audit_plus"]}   # v6o W1
            else:                                                                                                                  # v6o W1: v6n's dense route
                e, V = OPS.eigh_csr(h, self.dev)
                neg = e < 0
                Vfull0 = V[:, neg].clone(); E_ref0 = float(e[neg].sum()); route0 = "the ground state's eigendecomposition at T_0 (dense, v6m's)"; extra0 = None   # v6o W1 v1b
            self.w1_sector_decide(c, h, Vfull0, use_sector)                                                                        # v6o W1 v1b: the decision (and c.sector_prop) before the state's representation is fixed
            if getattr(c, "sector_prop", False):                                                                                   # v6o W1 v1b: the + sector block Phi_+ only when the halving is used
                c.Phi = vac0["V_plus"][:, torch.as_tensor(vac0["e_plus"], dtype=RD, device=self.dev) < 0].clone()                  # v6o W1 v1b
                route0 += "; the state propagated as Phi_+"                                                                        # v6o W1 v1b
            else:                                                                                                                  # v6o W1 v1b
                c.Phi = Vfull0                                                                                                     # v6o W1 v1b: the full columns (v6n's representation and alias: the T_0 vacuum's columns are read at the first checkpoint only, before any step, then released)
            c.F = torch.zeros((c.ops.n2, 0), dtype=CD, device=self.dev)
            log("kappa %d: the ground state of h(T_0): %d occupied of %d, %d levels within 1e-12 of zero (%s)" % (c.ops.kappa, int(neg.sum()), c.ops.n2, int((e.abs() < 1e-12).sum()), route0))   # v6o W1
            fb = W1S.fast_banded_matches(c.ops, N, v, self.dev)                                                                    # v6o W1: the field-built banded operator (the probes') against the CSR's
            self.w1_fast = bool(fb["ok"])                                                                                          # v6o W1
            log("v6o W1 kappa %d: the field-built banded operator against the CSR's at T_0: %s (largest diagonal difference %.1e, bound relative difference %.1e)" % (c.ops.kappa, "bitwise" if fb["ok"] else "DIFFERS: the probes take the CSR route", fb["max_diag_diff"], fb["bound_rel_diff"]))   # v6o W1
            if getattr(a, "collector", "on") == "off":                            # v6n: the T_0 vacuum (the ground state's own columns) for the first checkpoint
                self.d13_init_channel(c)
                c.d13.set_vacuum(float(H.T[self.k0]), self.k0, e.cpu().numpy(), Vfull0, route0, E_ref0, extra=extra0 if self.sector_on else None)   # v6o W1
            del V, e
            if self.dev.type == "cuda":
                torch.cuda.empty_cache()
        self.min_N_minus_absv_T0 = mnv'''),
# 9. the record's sub-steps
(r'''        sub = [(t0, t1, None)]
        if self.cw is not None and self.cw[0] <= t0 < self.cw[1] and not held:     # v6j (i)
            nsub = max(1, int(round(t1 - t0)))
            sub = [(t0 + i * (t1 - t0) / nsub, t0 + (i + 1) * (t1 - t0) / nsub, "cubic") for i in range(nsub)]
        rec = {"k": k, "t": t0, "per_channel": {}}''',
 r'''        sub = W1S.record_substeps(H, k, t0, t1, held, self.cw, self.interp_mode, self.interp_sub)                              # v6o W1: linear sub-steps (production), v6n's record (control), v6m's cubic window
        rec = {"k": k, "t": t0, "per_channel": {}}'''),
# 10. the per-channel body's start: the sector dispatch and the audit
(r'''            flow = 0.0; L0 = None; Kmax = 0; tail = 0.0; jump = 0.0
            chis_x = self.d13_cut_masks(c); flow_x = {lab: 0.0 for lab in chis_x}; jump_x = {lab: 0.0 for lab in chis_x}   # v6n (b): the ledger cuts
            h_cur = c.ops.h_csr(*H.fields(k))                                    # v6f (1): the record's piecewise h (the checkpoint's)''',
 r'''            flow = 0.0; L0 = None; Kmax = 0; tail = 0.0; jump = 0.0; Ksum = 0                                                   # v6o W1: Ksum, the record's summed Chebyshev degree
            chis_x = self.d13_cut_masks(c); flow_x = {lab: 0.0 for lab in chis_x}; jump_x = {lab: 0.0 for lab in chis_x}   # v6n (b): the ledger cuts
            if getattr(c, "sector_prop", False):                                                                                 # v6o W1: the + sector propagation (page_w1_sector_v1d)
                self.w1_step_record_sector(c, k, sub, held, rec, chis_x, flow_x, jump_x)                                          # v6o W1
                continue                                                                                                         # v6o W1
            h_cur = c.ops.h_csr(*H.fields(k))                                    # v6f (1): the record's piecewise h (the checkpoint's)
            self.w1_audit_record(c, h_cur, k)                                                                                    # v6o W1 defect 13: the pairing premise and [S_stag, h] on the record's h (exact, sparse)'''),
# 11. the sub-step loop's header
(r'''            for (sa, sb, mode) in sub:
                if mode == "cubic":
                    N, v = H.cubic_fields(0.5 * (sa + sb))
                else:
                    N, v = H.fields(k)
                h = c.ops.h_csr(N, v)
                if mode == "cubic" and not held:                                 # v6f (1): the sub-step's jump on the state''',
 r'''            for (sa, sb, mode, (N, v)) in sub:                                                                                   # v6o W1: the sub-step's fields come with it
                h = c.ops.h_csr(N, v)
                if mode in ("cubic", "linear") and not held:                                                                     # v6o W1 (v6f (1)): the sub-step's jump on the state, the linear sub-steps included'''),
# 12. the summed degree
(r'''                Kmax = max(Kmax, K_); tail += tl''',
 r'''                Kmax = max(Kmax, K_); tail += tl; Ksum += int(K_)                                                               # v6o W1'''),
# 13. the record entry (v1c: the sub-step count)
(r'''            rec["per_channel"][str(c.ops.kappa)] = {"collar_flow": flow, "L_at_t": L0, "cheb_K": Kmax, "cheb_tail": tail, "work_jump": jump,''',
 r'''            c.n_substeps += len(sub)                                                                                            # v6o W1 v1c: the propagation floor's count (the full route)
            rec["per_channel"][str(c.ops.kappa)] = {"collar_flow": flow, "L_at_t": L0, "cheb_K": Kmax, "cheb_K_sum": Ksum, "cheb_tail": tail, "work_jump": jump,   # v6o W1: cheb_K_sum'''),
# 14. the probes' record steps
(r'''        sub = [(t0, t1, None)]
        if self.cw is not None and self.cw[0] <= t0 < self.cw[1]:
            nsub = max(1, int(round(t1 - t0)))
            sub = [(t0 + i * (t1 - t0) / nsub, t0 + (i + 1) * (t1 - t0) / nsub, "cubic") for i in range(nsub)]
        out = []
        for (sa, sb, mode) in sub:
            N, v = H.cubic_fields(0.5 * (sa + sb)) if mode == "cubic" else H.fields(k)
            out.append((c.ops.h_csr(N, v), sb - sa))
        return out''',
 r'''        sub = W1S.record_substeps(H, k, t0, t1, False, self.cw, self.interp_mode, self.interp_sub)                             # v6o W1: the same sub-steps as step_record
        if getattr(self, "w1_fast", False):                                                                                      # v6o W1: the field-built banded operator (verified bitwise against the CSR's at T_0 and every checkpoint)
            return [(W1S.FastBanded(N, v, c.ops.kappa, c.ops.R, self.dev), sb - sa) for (sa, sb, mode, (N, v)) in sub]             # v6o W1
        return [(c.ops.h_csr(N, v), sb - sa) for (sa, sb, mode, (N, v)) in sub]                                                 # v6o W1'''),
# 15. the W1 methods (v1b: w1_set_vac_route; v1c: the sub-step count in the sector step, the floor label in the summary; v1d: the construction allowance's label)
(r'''    # ------------------------------------------------------------------------------------------------- v6n: the collector-free run
    def d13_setup(self):''',
 r'''    # --------------------------------------------------------------------------------------------------------------- v6o W1
    def w1_setup(self):                                                                                                           # v6o W1
        """v6o W1: the interpolation mode and its sub-steps, the sector decision's defaults, the audit accumulators, the check clocks."""
        a = self.a
        self.interp_mode = getattr(a, "interp", "linear"); self.interp_sub = max(1, int(getattr(a, "interp_sub", 8)))              # v6o W1
        self.sector_on = False; self.sector_report = None; self.w1_fast = False                                                   # v6o W1
        self.audit_on = (getattr(a, "audit", "on") == "on")                                                                       # v6o W1
        self.w1_audit = {"max_h_plus_conj_h": 0.0, "max_commutator_S_stag": 0.0, "records_audited": 0, "records_failed": 0, "tol_rel": 1e-13,
                         "fast_banded_mismatches": 0, "floor_premise_failures": 0}                                                # v6o W1 v1c: the floor premise's failures counted
        x = float(getattr(a, "sector_check_every_rs", 10.0))                                                                      # v6o W1
        if x <= 0:                                                                                                                # v6o W1
            self.sector_check_k = set(self.ck)                                                                                    # v6o W1
        else:                                                                                                                     # v6o W1
            self.sector_check_k = set(self.vac_k)                                                                                 # v6o W1: the baseline clocks always
            T0r = float(self.H.T[self.k0]) / self.rs_nom; T1r = self._T_of_k(self.kend) / self.rs_nom                              # v6o W1
            y = math.ceil(T0r / x - 1e-9) * x                                                                                     # v6o W1
            while y <= T1r + 1e-9:                                                                                                # v6o W1
                self.sector_check_k.add(self._k_of(y * self.rs_nom)); y += x                                                      # v6o W1
        self.ck |= self.sector_check_k                                                                                            # v6o W1
        log("v6o W1: propagation %s; sector %s (the - sector and the cross-cut norms at %d check clocks); audits %s; the propagation's floor (W1_fix_v1c, the construction allowance of W1_fix_v1d) beside every excess" % (
            self.w1_propagation_label(), getattr(a, "sector", "auto"), len(self.sector_check_k), "on" if self.audit_on else "off"))   # v6o W1 v1d

    def w1_propagation_label(self):                                                                                               # v6o W1
        return W1S.propagation_label(self.interp_mode, self.interp_sub, self.a.cubic_window)                                     # v6o W1

    def w1_set_vac_route(self, route):                                                                                            # v6o W1 v1b
        """v6o W1 v1b: the vacuum route into the run's D13 configuration and into every channel readout's own copy of it (ChannelReadout keeps
        dict(cfg); the first build set the run's dict only, so the readouts kept 'dense' and the first vacuum formed after T_0 on the sector route
        carried no V_plus: the oracle's KeyError at the second checkpoint). A readout created afterwards copies the updated run dict."""
        if self.d13_cfg is not None:                                                                                              # v6o W1 v1b
            self.d13_cfg["vac_route"] = route                                                                                     # v6o W1 v1b
        for cc in self.chs:                                                                                                       # v6o W1 v1b
            if cc.d13 is not None:                                                                                                # v6o W1 v1b
                cc.d13.cfg["vac_route"] = route                                                                                   # v6o W1 v1b

    def w1_sector_decide(self, c, h, Phi_full, use_sector):                                                                       # v6o W1
        """v6o W1: the massless sector symmetry [S_stag, h] = 0 (S_stag = diag((-1)^j sigma_y)) and the antiunitary pairing K h K = -h at m = 0: the
        operator's defects (sparse, exact) and the initial state's off-sector weight ||P_- C P_+||_F; the halving (and the + sector propagation) is used
        only when both are below the tolerances; --sector on refuses otherwise, auto falls back to v6n's dense route (printed). v1b: sets c.sector_prop
        and writes the vacuum route into every readout's configuration (w1_set_vac_route)."""
        a = self.a
        if getattr(a, "sector", "auto") == "off" or not use_sector:                                                               # v6o W1
            rep = {"used": False, "reason": "--sector off" if getattr(a, "sector", "auto") == "off" else "the operator does not respect S_stag or the pairing at T_0",
                   "operator": W1S.operator_audit(h)}                                                                            # v6o W1
        else:                                                                                                                     # v6o W1
            rep = W1S.sector_report(h, Phi_full, float(a.sector_tol))                                                             # v6o W1
            rep["used"] = bool(rep["ok"]); rep["reason"] = "the operator and the state respect S_stag" if rep["ok"] else "the state's off-sector weight exceeds --sector-tol"   # v6o W1
            if a.sector == "on" and not rep["ok"]:                                                                                # v6o W1
                raise RuntimeError("--sector on but the state or the operator does not respect S_stag: %s" % rep)                  # v6o W1
        self.sector_on = bool(rep["used"]); self.sector_report = rep                                                              # v6o W1
        c.sector_prop = bool(self.sector_on and getattr(a, "sector_propagate", "on") == "on")                                     # v6o W1 v1b: the state's representation follows the decision (read by start())
        if self.sector_on:                                                                                                        # v6o W1 v1b
            self.w1_set_vac_route("sector")                                                                                       # v6o W1 v1b: the references' vacua and every checkpoint's by the sector route, in the run's configuration and in every readout's copy
        op = rep.get("operator") or {}; st = rep.get("state") or {}                                                               # v6o W1
        log("v6o W1 kappa %d: sector check: max|h + conj h| %.1e, max|[S_stag, h]| %.1e (relative to max|h| %.3e); the state's off-sector weight %s (Tr C_+ %s, Tr C_- %s); "
            "the halving %s%s" % (c.ops.kappa, op.get("max_h_plus_conj_h", 0.0), op.get("max_commutator_S_stag", 0.0), op.get("h_max", 0.0), _f(st.get("off_sector_F"), "%.1e"),
                                 _f(st.get("tr_C_plus"), "%.3f"), _f(st.get("tr_C_minus"), "%.3f"), "USED" if self.sector_on else "NOT used (%s)" % rep.get("reason"),
                                 "; the state propagated as its + sector block Phi_+" if getattr(c, "sector_prop", False) else ""))   # v6o W1

    def w1_audit_record(self, c, h, k):                                                                                           # v6o W1
        """v6o W1 defect 13: the record's h against the pairing premise and the sector commutator (exact sparse arithmetic); failures logged and counted."""
        if not getattr(self, "audit_on", False):                                                                                  # v6o W1
            return                                                                                                                # v6o W1
        aud = W1S.operator_audit(h, self.w1_audit["tol_rel"])                                                                     # v6o W1
        self.w1_audit["records_audited"] += 1                                                                                     # v6o W1
        self.w1_audit["max_h_plus_conj_h"] = max(self.w1_audit["max_h_plus_conj_h"], aud["max_h_plus_conj_h"])                     # v6o W1
        self.w1_audit["max_commutator_S_stag"] = max(self.w1_audit["max_commutator_S_stag"], aud["max_commutator_S_stag"])         # v6o W1
        if not aud["ok"]:                                                                                                         # v6o W1
            self.w1_audit["records_failed"] += 1                                                                                  # v6o W1
            log("v6o W1 AUDIT FAIL kappa %d record %d: max|h + conj h| %.3e, max|[S_stag, h]| %.3e against max|h| %.3e" % (c.ops.kappa, k, aud["max_h_plus_conj_h"], aud["max_commutator_S_stag"], aud["h_max"]))   # v6o W1

    def w1_step_record_sector(self, c, k, sub, held, rec, chis_x, flow_x, jump_x):                                                # v6o W1
        """v6o W1: step_record's per-channel body on the + sector (page_w1_sector_v1d): the state Phi_+ (n x n_+) propagated by the banded
        h_+ = W_+^dagger h W_+ (half-bandwidth 2); every site-weighted trace of the full pure state from the pairing C_- = conj(1_+ - C_+), h_- =
        -conj(h_+): Tr[W h C] = 2 Re Tr[W h_+ C_+] - Tr[W h_+] (W site-diagonal, 1 for Tr[h C]), the collar current Tr[J C] = 2 Re Tr[J_+ C_+] with
        J_+ = (i/2)[h_+^2, Pi]; the work's jumps at every operator switch and the ledger cuts' flows as v6f/v6n define them; the source modes (full
        vectors) take the full banded h of the sub-step."""
        a, H = self.a, self.H                                                                                                     # v6o W1
        dev = self.dev                                                                                                            # v6o W1
        Rs = np.asarray(c.ops.R, float)                                                                                           # v6o W1
        chi_np = (Rs > self.R_collar_abs).astype(float)                                                                           # v6o W1
        chi_s = torch.as_tensor(chi_np, dtype=RD, device=dev).to(CD)[:, None]                                                     # v6o W1
        masks_np = {lab: m[0::2].detach().cpu().numpy().real.ravel() for lab, m in chis_x.items()}                                # v6o W1: the ledger cuts on the sites
        masks_s = {lab: m[0::2] for lab, m in chis_x.items()}                                                                     # v6o W1
        flow = 0.0; L0 = None; Kmax = 0; tail = 0.0; jump = 0.0; Ksum = 0                                                         # v6o W1
        h_cur = c.ops.h_csr(*H.fields(k))                                                                                         # v6o W1: the record's piecewise h (the checkpoint's)
        self.w1_audit_record(c, h_cur, k)                                                                                         # v6o W1
        hp_cur = W1S.sector_plus_csr(h_cur)                                                                                       # v6o W1
        for (sa, sb, mode, (N, v)) in sub:                                                                                        # v6o W1
            h = c.ops.h_csr(N, v); hp = W1S.sector_plus_csr(h)                                                                    # v6o W1
            if mode in ("cubic", "linear") and not held:                                                                          # v6o W1: the sub-step's jump on the state
                dh = (hp - hp_cur).tocsr(); Bd = OPS.Banded(dh, dev)                                                              # v6o W1
                jump += 2.0 * tr_blocked(Bd, c.Phi) - W1S.trace_diag(dh)                                                          # v6o W1
                for lab in chis_x:                                                                                                # v6o W1
                    jump_x[lab] += 2.0 * tr_blocked(Bd, c.Phi, masks_s[lab]) - W1S.trace_diag(dh, masks_np[lab])                   # v6o W1
                del Bd; hp_cur = hp                                                                                               # v6o W1
            Bp = OPS.Banded(hp, dev)                                                                                              # v6o W1
            trh_chi = W1S.trace_diag(hp, chi_np); trh_x = {lab: W1S.trace_diag(hp, masks_np[lab]) for lab in chis_x}             # v6o W1
            Eo0 = 2.0 * tr_blocked(Bp, c.Phi, chi_s) - trh_chi                                                                    # v6o W1
            Ex0 = {lab: 2.0 * tr_blocked(Bp, c.Phi, masks_s[lab]) - trh_x[lab] for lab in chis_x}                                 # v6o W1
            if L0 is None:                                                                                                        # v6o W1
                Jp = OPS.Banded(W1S.current_out_plus_csr(hp, Rs, self.R_collar_abs), dev)                                         # v6o W1
                L0 = 2.0 * tr_blocked(Jp, c.Phi)                                                                                  # v6o W1
                del Jp                                                                                                            # v6o W1
            c.Phi, K_, tl = propagate(Bp, c.Phi, sb - sa, a.cheb_tol, a.chunk)                                                    # v6o W1
            if c.d13 is not None and c.d13.src:                                                                                   # v6o W1: the source modes travel with the full h
                Bh = c.ops.banded(h); c.d13.propagate_sources(Bh, sb - sa, a.cheb_tol, a.chunk); del Bh                            # v6o W1
            Eo1 = 2.0 * tr_blocked(Bp, c.Phi, chi_s) - trh_chi                                                                    # v6o W1
            flow += Eo1 - Eo0                                                                                                     # v6o W1
            for lab in chis_x:                                                                                                    # v6o W1
                flow_x[lab] += (2.0 * tr_blocked(Bp, c.Phi, masks_s[lab]) - trh_x[lab]) - Ex0[lab]                                # v6o W1
            Kmax = max(Kmax, K_); tail += tl; Ksum += int(K_)                                                                     # v6o W1
            del Bp                                                                                                                # v6o W1
        if not held and k + 1 < len(H):                                                                                           # v6o W1: the jump into the next record's h at its start
            hn = W1S.sector_plus_csr(c.ops.h_csr(*H.fields(k + 1))); dh = (hn - hp_cur).tocsr(); Bd = OPS.Banded(dh, dev)          # v6o W1
            jump += 2.0 * tr_blocked(Bd, c.Phi) - W1S.trace_diag(dh)                                                              # v6o W1
            for lab in chis_x:                                                                                                    # v6o W1
                jump_x[lab] += 2.0 * tr_blocked(Bd, c.Phi, masks_s[lab]) - W1S.trace_diag(dh, masks_np[lab])                       # v6o W1
            del Bd                                                                                                                # v6o W1
        c.n_substeps += len(sub)                                                                                                  # v6o W1 v1c: the propagation floor's count (the sector route)
        c.work_jump_since += jump                                                                                                 # v6o W1
        c.collar_flow_acc += flow                                                                                                 # v6o W1
        c.cheb_K_max = max(c.cheb_K_max, Kmax); c.cheb_tail_acc += tail                                                           # v6o W1
        rec["per_channel"][str(c.ops.kappa)] = {"collar_flow": flow, "L_at_t": L0, "cheb_K": Kmax, "cheb_K_sum": Ksum, "cheb_tail": tail, "work_jump": jump,   # v6o W1
                                                "cut_flow": dict(flow_x), "cut_jump": dict(jump_x), "sector_prop": True}         # v6o W1

    def w1_summary(self):                                                                                                         # v6o W1
        return {"propagation": self.w1_propagation_label(), "interp": self.interp_mode, "interp_sub": self.interp_sub, "sector": self.sector_report, "sector_used": self.sector_on,   # v6o W1
                "sector_propagate": [bool(getattr(c, "sector_prop", False)) for c in self.chs], "fast_banded_probes": bool(getattr(self, "w1_fast", False)), "audit": self.w1_audit,   # v6o W1
                "sector_check_clocks_over_rs": sorted(self._T_of_k(k_) / self.rs_nom for k_ in getattr(self, "sector_check_k", set())),   # v6o W1
                "E_kappa_identity": "E_kappa = Tr[|h| (C - P_-)^2] = Tr[h (C - P_-)] = E_inv for the pure state: every row's per-channel E_inv times 16 kappa is the formed-energy majorant's term",   # v6o W1
                "propagation_floor": ("v1c (W1_fix_v1c) and v1d (W1_fix_v1d): every row's pc['w1'] holds the state's impurity ||Phi^dagger Phi - 1||_1, its value at the first checkpoint T_init (the construction), "
                                      "the sub-step count, the propagator's Gershgorin bound and the a priori bound 4 n_+^(3/2) eps (the construction allowance) + 2 n_+ sum of the Chebyshev tails + the rounding "
                                      "allowance; every cut's excess carries excess_floor_low (the impurity's entropy, 2 m h2(T/m) + the rounding term), excess_floor_high (the a priori bound's) and "
                                      "excess_resolution; the held gate's tolerance is the low floor"),   # v6o W1 v1d
                "T_init": {str(c.ops.kappa): getattr(c, "T_init", None) for c in self.chs},                                       # v6o W1 v1d
                "cheb_tol": float(getattr(self.a, "cheb_tol", 1e-14)),                                                            # v6o W1 v1c: the propagation's tolerance, the truncation floor's parameter
                "labels": "S_X is the raw exterior-cut entropy (regulated); the readout is the excess over the held vacuum of the same fields at the same cut and record"}   # v6o W1

    # ------------------------------------------------------------------------------------------------- v6n: the collector-free run
    def d13_setup(self):'''),
# 16. the live reference's columns
(r'''        if k_ref == self.k and c.Phi is not None and not os.path.exists(path):
            Phi = c.Phi''',
 r'''        if k_ref == self.k and c.Phi is not None and not os.path.exists(path):
            Phi = W1S.assemble_full_state(c.Phi) if getattr(c, "sector_prop", False) else c.Phi                                  # v6o W1: the full columns from the + sector block'''),
# 17. the saved columns' meta
(r'''            meta = {"kappa": c.ops.kappa, "k": int(k), "t": t, "t_over_rs": trs, "nsites": int(self.H.nsites), "n2": int(c.ops.n2), "n_occ": int(c.Phi.shape[1]), "rs_nominal": self.rs_nom,''',
 r'''            X_ = W1S.assemble_full_state(c.Phi) if getattr(c, "sector_prop", False) else c.Phi                                   # v6o W1: the full columns from the + sector block
            meta = {"kappa": c.ops.kappa, "k": int(k), "t": t, "t_over_rs": trs, "nsites": int(self.H.nsites), "n2": int(c.ops.n2), "n_occ": int(X_.shape[1]), "rs_nominal": self.rs_nom,   # v6o W1'''),
(r'''                    "code": "page_history_v6n.py", "what": "the occupied columns Phi of the propagated pure Gaussian state (C = Phi Phi^dagger) at the record's clock, before step_record(k)",''',
 r'''                    "code": "page_history_v6o.py", "what": "the occupied columns Phi of the propagated pure Gaussian state (C = Phi Phi^dagger) at the record's clock, before step_record(k)",   # v6o W1'''),
(r'''                D13.save_columns(path, c.Phi, meta, key="Phi", fmt=self.d13_cfg["columns_format"])
                log("v6n: kappa %d: the occupied columns at T/r_s %.4f (record %d) saved to %s (%d x %d complex128, %.2f GiB)" % (c.ops.kappa, trs, k, path, c.Phi.shape[0], c.Phi.shape[1], c.Phi.numel() * 16 / 2.0 ** 30))''',
 r'''                D13.save_columns(path, X_, meta, key="Phi", fmt=self.d13_cfg["columns_format"])                                   # v6o W1
                log("v6o W1: kappa %d: the occupied columns at T/r_s %.4f (record %d) saved to %s (%d x %d complex128, %.2f GiB)" % (c.ops.kappa, trs, k, path, X_.shape[0], X_.shape[1], X_.numel() * 16 / 2.0 ** 30))   # v6o W1'''),
# 18. checkpoint_lean's energy readouts on either representation (v1c: the impurity's trace norm and the propagator's bound; v1d: T_init at the first checkpoint)
(r'''            Phi = c.Phi; n0 = Phi.shape[1]
            E_hC = c.trace_h(Bh, Phi)
            E_ref, mininfo = D13.tr_min_h0(h)
            E_inv = E_hC - E_ref
            Gs = Phi.mH @ Phi
            Gs.diagonal().sub_(1.0)
            unit = float(torch.linalg.norm(Gs)); del Gs
            c.unit_defect = max(c.unit_defect, unit)
            chi_c = (c.ops.R2 > self.R_collar_abs).to(CD)[:, None]
            E_out_C = tr_blocked(Bh, Phi, chi_c)
            vac = cr.vacuum(h, t, k, due=(k in self.vac_k))
            r_t = c.outer_root(N, v)
            E_out = None; W_hdot = None; E_ref_dense = None
            if vac is not None:
                Vneg = vac["Vneg"]
                E_ref_dense = vac["E_ref"]
                E_out = E_out_C - tr_blocked(Bh, Vneg, chi_c)
                Bd = OPS.Banded(c.ops.hdot_csr(dN, dv, self.tau), self.dev)
                W_hdot = c.trace_h(Bd, Phi) - c.trace_h(Bd, Vneg)
                del Bd''',
 r'''            Phi = c.Phi; n0 = Phi.shape[1]
            Gs = Phi.mH @ Phi
            Gs.diagonal().sub_(1.0)
            unit = float(torch.linalg.norm(Gs))                                                                                  # v6o W1 v1c: the Gram kept for its trace norm
            unit_tr = float(torch.linalg.eigvalsh(0.5 * (Gs + Gs.mH)).abs().sum()); del Gs                                       # v6o W1 v1c: ||Phi^dagger Phi - 1||_1, the state's impurity (its trace-norm distance from the nearest projector; W1_fix_v1c)
            if getattr(c, "T_init", None) is None:                                                                                 # v6o W1 v1d
                c.T_init = unit_tr                                                                                                 # v6o W1 v1d: the construction's impurity, measured at the first checkpoint before any sub-step (restored on resume; W1_fix_v1d)
            c.unit_defect = max(c.unit_defect, unit)
            chi_c = (c.ops.R2 > self.R_collar_abs).to(CD)[:, None]
            E_ref, mininfo = D13.tr_min_h0(h)
            sp_ = bool(getattr(c, "sector_prop", False))                                                                          # v6o W1: Phi is the + sector block Phi_+ (n x n_+)
            if sp_:                                                                                                                 # v6o W1
                hp = W1S.sector_plus_csr(h); Bp = OPS.Banded(hp, self.dev); chi_s = chi_c[0::2]                                        # v6o W1
                E_hC = 2.0 * c.trace_h(Bp, Phi) - W1S.trace_diag(hp)                                                                 # v6o W1: Tr[h C] = 2 Re Tr[h_+ C_+] - Tr h_+
                E_out_C = 2.0 * tr_blocked(Bp, Phi, chi_s) - W1S.trace_diag(hp, chi_s)                                               # v6o W1
                w1_bound = Bp.bound                                                                                                # v6o W1 v1c: the propagator's Gershgorin bound (the E_inv premise 2 w T)
            else:                                                                                                                   # v6o W1: v6n's full state
                E_hC = c.trace_h(Bh, Phi)
                E_out_C = tr_blocked(Bh, Phi, chi_c)
                w1_bound = Bh.bound                                                                                                # v6o W1 v1c
            E_inv = E_hC - E_ref
            vac = cr.vacuum(h, t, k, due=(k in self.vac_k) or self.sector_on)                                                     # v6o W1: the held vacuum at every checkpoint on the sector route
            Phi_rd = W1S.assemble_full_state(Phi) if sp_ else Phi                                                                   # v6o W1: the full columns for the D13 readouts, the references and the columns
            if self.w1_fast and not W1S.fast_banded_matches(c.ops, N, v, self.dev)["ok"]:                                           # v6o W1: the probes' operator re-verified on this record's fields
                self.w1_fast = False; self.w1_audit["fast_banded_mismatches"] += 1                                                  # v6o W1
                log("v6o W1 kappa %d T/r_s %.3f: the field-built banded operator differs from the CSR's: the probes take the CSR route from here" % (c.ops.kappa, t / self.rs_nom))   # v6o W1
            r_t = c.outer_root(N, v)
            E_out = None; W_hdot = None; E_ref_dense = None
            if vac is not None:
                Vneg = vac["Vneg"]
                E_ref_dense = vac["E_ref"]
                hd_ = c.ops.hdot_csr(dN, dv, self.tau)                                                                              # v6o W1
                if sp_:                                                                                                             # v6o W1
                    Vp_ = vac["V_plus"][:, torch.as_tensor(np.asarray(vac["e_plus"], float), dtype=RD, device=self.dev) < 0]            # v6o W1
                    E_out = E_out_C - (2.0 * tr_blocked(Bp, Vp_, chi_s) - W1S.trace_diag(hp, chi_s))                                   # v6o W1
                    hdp_ = W1S.sector_plus_csr(hd_); Bd = OPS.Banded(hdp_, self.dev)                                                  # v6o W1
                    W_hdot = 2.0 * (c.trace_h(Bd, Phi) - c.trace_h(Bd, Vp_))                                                          # v6o W1: Tr[h_dot (C - P)] = 2 Re Tr[h_dot_+ (C_+ - P_{-,+})]
                    del Vp_                                                                                                         # v6o W1
                else:                                                                                                               # v6o W1
                    E_out = E_out_C - tr_blocked(Bh, Vneg, chi_c)
                    Bd = OPS.Banded(hd_, self.dev)
                    W_hdot = c.trace_h(Bd, Phi) - c.trace_h(Bd, Vneg)
                del Bd'''),
# 19. the W1 checks of the checkpoint (v1b: the identity audit's threshold; v1c: the impurity, the floor, the readout's floor dict; v1d: T_init and the construction allowance)
(r'''            pc["work_jumps_since_last"] = c.work_jump_since
            if c.E_hC_last is not None:''',
 r'''            w1_fe = W1S.formed_energy(vac, Phi_rd) if (vac is not None and vac.get("V_plus") is not None) else None                  # v6o W1: E_kappa's spectral form, the occupation pairing, the off-sector weight
            w1_aud = mininfo.get("audit")                                                                                             # v6o W1 defect 13: Tr min(h, 0) = -Tr|h|/2 on the banded spectrum
            w1_warn, w1_tol, w1_res = W1S.identity_warning(E_inv, w1_fe, w1_aud)                                                     # v6o W1 v1b: the residual's floor follows Tr|h| (the first build's 1e-8 max(|E_inv|, 1e-6) flagged rounding at E_inv ~ 0)
            w1_q = (int(vac["n_neg_plus"]) if (sp_ and vac is not None and vac.get("n_neg_plus") is not None) else (int(n0) if sp_ else int(mininfo["n_neg"])))   # v6o W1 v1c: the vacuum's rank (the + sector's occupied count; the full chain's on the dense route)
            w1_fl = W1S.propagation_floor(int(n0) if sp_ else int(Phi.shape[1]), c.cheb_tail_acc, int(getattr(c, "n_substeps", 0)), int(c.cheb_K_max), unit_tr, T_init=getattr(c, "T_init", None))   # v6o W1 v1d: the state's trace-norm error, a priori (the construction allowance + 2 n_+ sum of the Chebyshev tails + the rounding allowance) and measured (||Phi^dagger Phi - 1||_1, the impurity), with the first checkpoint's impurity carried (W1_fix_v1d)
            cr.w1_floor = dict(w1_fl, q=w1_q, sector=sp_, h_bound=w1_bound)                                                          # v6o W1 v1c: the readout annotates every cut's excess with its floors (page_w1_sector_v1d.excess_floors)
            pc["w1"] = {"E_kappa_spectral": w1_fe, "banded_audit": w1_aud, "E_kappa_identity_residual": w1_res, "E_kappa_identity_tol": w1_tol,   # v6o W1 v1b
                        "n0_plus": int(n0) if sp_ else None, "n_occ_full": int(Phi_rd.shape[1]), "sector_prop": sp_, "E_ref_banded_minus_sector": (E_ref - E_ref_dense) if E_ref_dense is not None else None,   # v6o W1
                        "sea_unitarity_trace": unit_tr, "sea_unitarity_F": unit, "n_substeps": int(getattr(c, "n_substeps", 0)), "h_bound": w1_bound, "propagation_floor": w1_fl,   # v6o W1 v1c: the impurity and the floor's inputs
                        "T_init": getattr(c, "T_init", None)}                                                                           # v6o W1 v1d: the first checkpoint's impurity (the construction)
            if w1_warn:                                                                                                                 # v6o W1 v1b
                log("v6o W1 AUDIT WARNING kappa %d T/r_s %.3f: pairing residual %s, E_kappa identity residual %s against the tolerance %.1e" % (c.ops.kappa, t / self.rs_nom, _f((w1_aud or {}).get("pairing_residual_rel"), "%.2e"), _f(w1_res, "%.2e"), w1_tol))   # v6o W1 v1b
            if w1_fl.get("premise_ok") is False:                                                                                        # v6o W1 v1c
                self.w1_audit["floor_premise_failures"] += 1                                                                            # v6o W1 v1c
                log("v6o W1 FLOOR WARNING kappa %d T/r_s %.3f: the state's impurity %.3e exceeds the propagation's a priori bound %.3e (construction %.2e, truncation %.2e, rounding allowance %.2e; at the first checkpoint %s): the drift is not the construction's and the propagation's alone" % (c.ops.kappa, t / self.rs_nom, unit_tr, w1_fl["T_cheb"], w1_fl["T_cheb_construction"], w1_fl["T_cheb_truncation"], w1_fl["T_cheb_rounding_allowance"], _f(w1_fl.get("T_init"), "%.2e")))   # v6o W1 v1d
            log(W1S.format_w1_line(c.ops.kappa, t / self.rs_nom, E_inv, w1_fe, w1_aud, cr.w1_floor))                                 # v6o W1 v1c: the impurity and the floor on the line
            pc["work_jumps_since_last"] = c.work_jump_since
            if c.E_hC_last is not None:'''),
# 20. the reference's columns
(r'''            if k in self.ref_k:
                self.d13_reference(c, k, t, Phi, h, vac, N, v, E_k)''',
 r'''            if k in self.ref_k:
                self.d13_reference(c, k, t, Phi_rd, h, vac, N, v, E_k)                                                              # v6o W1: the full columns'''),
# 21. the readouts' call with the sector flags
(r'''            if self.d13_inline:
                sc, arrays = cr.checkpoint(Phi, h, Bh, t, k, N, v, E_k, rs_k, TH, vac, dens=dens, T_start=float(H.T[self.k0]),''',
 r'''            if self.d13_inline:
                cr.sector_on = self.sector_on; cr.sector_check_now = bool(k in self.sector_check_k)                                # v6o W1: the halved route and its check clocks
                sc, arrays = cr.checkpoint(Phi_rd, h, Bh, t, k, N, v, E_k, rs_k, TH, vac, dens=dens, T_start=float(H.T[self.k0]),   # v6o W1: the full columns'''),
# 22. the vacuum's release and the full columns'
(r'''            cr.release_vacuum()
            pc["seconds"] = time.time() - tc''',
 r'''            cr.release_vacuum()
            if sp_:                                                                                                                 # v6o W1
                del Phi_rd                                                                                                          # v6o W1
            pc["seconds"] = time.time() - tc'''),
# 23. the probes' live state
(r'''            res = c.d13.probes(t, k, c.Phi, h, lambda kk, c=c: self.d13_record_steps(c, kk), refs)''',
 r'''            Phi_now = W1S.assemble_full_state(c.Phi) if getattr(c, "sector_prop", False) else c.Phi                               # v6o W1: the full columns for the probes
            res = c.d13.probes(t, k, Phi_now, h, lambda kk, c=c: self.d13_record_steps(c, kk), refs)                              # v6o W1
            del Phi_now                                                                                                           # v6o W1'''),
# 24. the state's representation flag, the sub-step count and T_init saved
(r'''                                                 "d13": (c.d13.state_host() if c.d13 is not None else None)}''',
 r'''                                                 "d13": (c.d13.state_host() if c.d13 is not None else None), "sector_prop": bool(getattr(c, "sector_prop", False)),   # v6o W1
                                                 "n_substeps": int(getattr(c, "n_substeps", 0)), "T_init": getattr(c, "T_init", None)}   # v6o W1 v1d: the sub-step count (v1c) and the first checkpoint's impurity'''),
(r'''        st["v6g"] = {"anchor_started": self.anchor_started, "follow_end": self.follow_end}''',
 r'''        st["v6g"] = {"anchor_started": self.anchor_started, "follow_end": self.follow_end}
        st["w1"] = {"sector_on": bool(self.sector_on), "sector_report": self.sector_report, "w1_fast": bool(getattr(self, "w1_fast", False)), "audit": self.w1_audit}   # v6o W1'''),
# 25. the state's representation flag restored (v1b: the vacuum route into every readout's copy; v1c: the sub-step count; v1d: T_init)
(r'''            c.work_jump_since = s.get("work_jump_since", 0.0); c.E_hC_last = s.get("E_hC_last"); c.E_ref_last = s.get("E_ref_last")   # v6g: kept on resume''',
 r'''            c.work_jump_since = s.get("work_jump_since", 0.0); c.E_hC_last = s.get("E_hC_last"); c.E_ref_last = s.get("E_ref_last")   # v6g: kept on resume
            c.sector_prop = bool(s.get("sector_prop", False))                                                                    # v6o W1
            c.n_substeps = int(s.get("n_substeps", 0))                                                                           # v6o W1 v1c
            c.T_init = s.get("T_init")                                                                                           # v6o W1 v1d: the first checkpoint's impurity (a state without it sets T_init at its next checkpoint and prints the growth from there)'''),
(r'''        self.anchor_started = bool(v6g.get("anchor_started", False)); self.follow_end = v6g.get("follow_end")''',
 r'''        self.anchor_started = bool(v6g.get("anchor_started", False)); self.follow_end = v6g.get("follow_end")
        w1 = st.get("w1") or {}                                                                                                   # v6o W1
        self.sector_on = bool(w1.get("sector_on", False)); self.sector_report = w1.get("sector_report"); self.w1_fast = bool(w1.get("w1_fast", False))   # v6o W1
        if w1.get("audit"):                                                                                                       # v6o W1
            self.w1_audit = w1["audit"]                                                                                           # v6o W1
            self.w1_audit.setdefault("floor_premise_failures", 0)                                                                # v6o W1 v1c
        if self.sector_on:                                                                                                        # v6o W1 v1b
            self.w1_set_vac_route("sector")                                                                                       # v6o W1 v1b: the run's configuration and every channel readout's copy'''),
# 26. the summary's code and the W1 block
(r'''        summary = {"code": "page_history_v6n.py", "collector": getattr(a, "collector", "on"), "d13": self.d13_summary(), "settings": vars(a), "ledger_anchor": anc, "extension": extd,''',
 r'''        summary = {"code": "page_history_v6o.py", "w1": self.w1_summary(), "collector": getattr(a, "collector", "on"), "d13": self.d13_summary(), "settings": vars(a), "ledger_anchor": anc, "extension": extd,   # v6o W1'''),
# 27. the store's code
(r'''        self.d13_store = D13.Store(self.prefix + "_d13", settings={"code": "page_history_v6n.py", "args": vars(a), "d13": {k_: (list(v_) if isinstance(v_, (list, tuple)) else v_) for k_, v_ in cfg.items()}},''',
 r'''        self.d13_store = D13.Store(self.prefix + "_d13", settings={"code": "page_history_v6o.py", "args": vars(a), "d13": {k_: (list(v_) if isinstance(v_, (list, tuple)) else v_) for k_, v_ in cfg.items()}},   # v6o W1'''),
# 28. the saved vacua's code
(r'''                    D13.save_columns(vpath, Vneg, {"kappa": c.ops.kappa, "k": int(k_ref), "t": t, "t_over_rs": trs, "code": "page_history_v6n.py", "what": "the occupied columns of P_-(h(t)), the held Killing vacuum"}, key="Vneg")''',
 r'''                    D13.save_columns(vpath, Vneg, {"kappa": c.ops.kappa, "k": int(k_ref), "t": t, "t_over_rs": trs, "code": "page_history_v6o.py", "what": "the occupied columns of P_-(h(t)), the held Killing vacuum"}, key="Vneg")   # v6o W1'''),
(r'''                D13.save_columns(vpath, vac["Vneg"], {"kappa": c.ops.kappa, "k": int(k), "t": t, "t_over_rs": trs, "code": "page_history_v6n.py", "what": "the occupied columns of P_-(h(t)), the held Killing vacuum"}, key="Vneg")''',
 r'''                D13.save_columns(vpath, vac["Vneg"], {"kappa": c.ops.kappa, "k": int(k), "t": t, "t_over_rs": trs, "code": "page_history_v6o.py", "what": "the occupied columns of P_-(h(t)), the held Killing vacuum"}, key="Vneg")   # v6o W1'''),
# 29. the D13 tables: the raw entropy, the held vacuum's, the excess, the impurity and the floors (v1c)
(r'''                      "| T/r_s | E_inv | L (interval) | L/L_H(kappa_T) | collar out soft | collar out hard | shells hard out | shells hard in | above band N | " + " | ".join("S_X %s" % x for x in cl) + " | " + " | ".join("excess %s" % x for x in cl) + " | vacuum |",
                      "|---|" + "---|" * (9 + 2 * len(cl))]''',
 r'''                      "| T/r_s | E_kappa = E_inv (Tr|h|(C - P_-)^2 = Tr[h (C - P_-)], this channel) | impurity ||Phi^dagger Phi - 1||_1 | L (interval) | L/L_H(kappa_T) | collar out soft | collar out hard | shells hard out | shells hard in | above band N | " + " | ".join("S_X raw %s" % x for x in cl) + " | " + " | ".join("S_X held vacuum %s" % x for x in cl) + " | " + " | ".join("excess %s" % x for x in cl) + " | " + " | ".join("excess floor (high) %s" % x for x in cl) + " | vacuum route |",   # v6o W1 v1c: the impurity and the floors
                      "|---|" + "---|" * (10 + 4 * len(cl))]                                                                      # v6o W1 v1c'''),
(r'''                lines.append("| %.3f | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                    r["t_over_rs"], _f(r["E_inv"], "%.6e"), _f(r["L_interval"], "%.4e"), _f(r["L_over_LH"], "%.4f"), _f(cs.get("E_flux_out_soft"), "%.4e"), _f(cs.get("E_flux_out_hard"), "%.4e"),
                    _f(sh.get("hard_E_out"), "%.4e"), _f(sh.get("hard_E_in"), "%.4e"), _f(sh.get("above_N_total"), "%.3e"),
                    " | ".join(_f((cu.get(x) or {}).get("S_X"), "%.6f") for x in cl), " | ".join(_f((cu.get(x) or {}).get("excess"), "%.6f") for x in cl), "dense" if r["vacuum_dense"] else "-"))''',
 r'''                lines.append("| %.3f | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (                  # v6o W1 v1c: the impurity, the held vacuum's and the floors' columns
                    r["t_over_rs"], _f(r["E_inv"], "%.6e"), _f(((r.get("w1") or {}).get("sea_unitarity_trace")), "%.1e"), _f(r["L_interval"], "%.4e"), _f(r["L_over_LH"], "%.4f"), _f(cs.get("E_flux_out_soft"), "%.4e"), _f(cs.get("E_flux_out_hard"), "%.4e"),   # v6o W1 v1c
                    _f(sh.get("hard_E_out"), "%.4e"), _f(sh.get("hard_E_in"), "%.4e"), _f(sh.get("above_N_total"), "%.3e"),
                    " | ".join(_f((cu.get(x) or {}).get("S_X"), "%.6f") for x in cl), " | ".join(_f((cu.get(x) or {}).get("S_X_vac"), "%.6f") for x in cl), " | ".join(_f((cu.get(x) or {}).get("excess"), "%.6f") for x in cl), " | ".join(_f((cu.get(x) or {}).get("excess_floor_high"), "%.1e") for x in cl), ("sector" if self.sector_on else "dense") if r["vacuum_dense"] else "-"))   # v6o W1 v1c'''),
# 30. the tables' header line
(r'''             "History: %s; %d records, tau %g, ell %s, g %g, r_s nominal %g, the engine's %d sites, the chain %d sites; channels %s (weights 16 kappa); R_cut %g r_s, collar %g r_s; eps_loc %g, eps_form %g, the pair ledger %s; collection every %g r_s.\n" % (
                 self.H.label, len(self.H), self.tau, self.H.ell, self.g, self.rs_nom, self.H.ne, self.H.nsites, self.channels, a.R_cut_rs, a.R_collar_rs, a.eps_loc, a.eps_form,
                 a.pair_ledger, a.collect_every_rs),''',
 r'''             "History: %s; %d records, tau %g, ell %s, g %g, r_s nominal %g, the engine's %d sites, the chain %d sites; channels %s (weights 16 kappa); R_cut %g r_s, collar %g r_s; eps_loc %g, eps_form %g, the pair ledger %s; collection every %g r_s; propagation: %s (Chebyshev tolerance %g); the S_X columns of the D13 tables are the raw exterior-cut entropy (regulated), the readout is the excess over the held vacuum of the same fields at the same cut and record, printed with its propagation floor (v1c, v1d).\n" % (   # v6o W1 v1d
                 self.H.label, len(self.H), self.tau, self.H.ell, self.g, self.rs_nom, self.H.ne, self.H.nsites, self.channels, a.R_cut_rs, a.R_collar_rs, a.eps_loc, a.eps_form,
                 a.pair_ledger, a.collect_every_rs, self.w1_propagation_label(), float(a.cheb_tol)),                             # v6o W1 v1c'''),
# 31. the sequence's verdict log
(r'''        res = D13.ksum_table(a.out, a.tag, done, a.kappa_hard_frac, getattr(a, "probe_from_rs", 40.0))
        if res is not None:
            log("v6n: the convergence table in kappa after channels %s: the last channel's hard outgoing collar flow share %.4f of the total (rule: %s; %s)" % (
                done, res["last_channel_hard_share_max"], res["rule"], "the hard part still moves" if res["hard_moves_above_frac"] else "converged at this channel"))''',
 r'''        res = D13.ksum_table(a.out, a.tag, done, a.kappa_hard_frac, getattr(a, "probe_from_rs", 40.0))
        if res is not None:
            log("v6o W1: the convergence table in kappa after channels %s: %s (the retired rule's last-channel hard share %.4f prints beside and decides nothing)" % (   # v6o W1
                done, res["verdict_line"], res["last_channel_hard_share_max"]))                                                   # v6o W1'''),
# 32. the extension by the tail-majorant rule
(r'''        if i >= len(kappas):
            if not a.kappa_extend or res is None or not res["hard_moves_above_frac"]:
                break
            if done[-1] + 1 > a.kappa_hard_max:
                log("v6n: the hard part still moves at kappa %d but --kappa-hard-max %d is reached: the sum stops here (printed with the last channel's share)" % (done[-1], a.kappa_hard_max))
                break
            log("v6n: the hard part moves at kappa %d by more than %g of its total: the sum is extended to kappa %d" % (done[-1], a.kappa_hard_frac, done[-1] + 1))''',
 r'''        if i >= len(kappas):
            if not a.kappa_extend or res is None or not res["extend"]:                                                           # v6o W1: the tail-majorant rule decides the extension
                break
            if done[-1] + 1 > a.kappa_hard_max:
                log("v6o W1: the tail majorant is above the tolerance at kappa %d but --kappa-hard-max %d is reached: the sum stops here and prints with its tail bound (not converged)" % (done[-1], a.kappa_hard_max))   # v6o W1
                break
            log("v6o W1: the tail majorant at kappa %d exceeds %g of the partial sum: the sum is extended to kappa %d" % (done[-1], a.kappa_hard_frac, done[-1] + 1))   # v6o W1'''),
# 33. the D13 tables' rows carry the channel's W1 block (v1c: the impurity column)
(r'''                rows_ch.append({"t_over_rs": r["t_over_rs"], "two_gE": r["two_gE"], "vacuum_dense": bool(pc.get("vacuum_dense")), "inv": sc.get("inventory"),
                                "shells": (sc.get("shells") or {}).get("totals"), "E_inv": pc.get("E_inv"), "cuts": sc.get("cuts"), "collar_split": sc.get("collar_split"), "L_interval": r.get("L_interval"), "L_over_LH": r.get("L_over_LH_kappaT")})''',
 r'''                rows_ch.append({"t_over_rs": r["t_over_rs"], "two_gE": r["two_gE"], "vacuum_dense": bool(pc.get("vacuum_dense")), "inv": sc.get("inventory"),
                                "shells": (sc.get("shells") or {}).get("totals"), "E_inv": pc.get("E_inv"), "cuts": sc.get("cuts"), "collar_split": sc.get("collar_split"), "L_interval": r.get("L_interval"), "L_over_LH": r.get("L_over_LH_kappaT"), "w1": pc.get("w1")})   # v6o W1 v1c: the impurity for the tables'''),
]

# ----------------------------------------------------------------------------------------------------- v1b -> v1d
REPL_V1D = [
(r'''"""page_d13_readout_v1b.py: page_d13_readout_v1.py with (a'), the cross-cut block trace norms of hreduce's EVOLVEDCUT_theorem_v1 EC1''',
 r'''"""page_d13_readout_v1d.py (v6o W1): page_d13_readout_v1b.py with (a''), the massless-sector halving of every dense entropy and the tail-majorant rule
of the kappa sum (built from v1b by build_page_history_v6o_v1d.py's exact replacement list; v1b is not touched; the changed lines carry "v1c" or "v1d"):
    ChannelReadout.checkpoint uses page_w1_sector_v1d.exterior_entropies_sector and cross_cut_readout_sector when the caller sets sector_on (the +
    sector's blocks at half the dimension, the - sector beside at the check clocks, the pairing defects printed); killing_vacuum takes the route
    "sector" (P_-(h) from the eigendecomposition of h_+); set_vacuum carries the sector data; tr_min_h0 returns the pairing audit Tr min(h, 0) =
    -Tr|h|/2 (defect 13); the probes accept a banded operator per sub-step; ksum_table applies the formed-energy tail-majorant rule
    (page_w1_sector_v1d.convergence_rule, defect 10) and prints the retired last-channel share beside, deciding nothing by it; the S_X columns are
    labelled raw. W1_fix_v1c: the caller's w1_floor (the state's impurity, the a priori Chebyshev bound) annotates every cut's excess with its
    propagation floors and resolution on both routes (page_w1_sector_v1d.excess_floors); ksum_table carries the floors; format_line prints them.
    W1_fix_v1d: the a priori bound carries the construction allowance (page_w1_sector_v1d.propagation_floor); nothing of this file's arithmetic changes.
    v1b's docstring follows.

page_d13_readout_v1b.py: page_d13_readout_v1.py with (a'), the cross-cut block trace norms of hreduce's EVOLVEDCUT_theorem_v1 EC1'''),
(r'''CODE = "page_d13_readout_v1b.py"''',
 r'''CODE = "page_d13_readout_v1d.py"  # v1d (v6o W1)'''),
(r'''import page_ops_v6 as OPS  # noqa: E402''',
 r'''import page_ops_v6 as OPS  # noqa: E402
import page_w1_sector_v1d as W1S  # noqa: E402  (v1c, v6o W1 v1d: the sector route, the audits, the majorant rule, the propagation's floor with the construction allowance; W1_fix_v1d.md)'''),
(r'''    g.add_argument("--vac-route", dest="vac_route", default="dense", choices=["dense", "pair"],
                   help="dense: torch eigh of h (v6m's); pair: the massless chain's conjugate-pair structure h = i A (the real eigenproblem of h^2, half the work)")''',
 r'''    g.add_argument("--vac-route", dest="vac_route", default="dense", choices=["dense", "pair", "sector"],                                                     # v1c (v6o W1)
                   help="dense: torch eigh of h (v6m's); pair: the massless chain's conjugate-pair structure h = i A (the real eigenproblem of h^2, half the work); sector: the + sector block h_+ (v1c: 1/8 of the dense work; page_history_v6o sets it when the state respects S_stag)")   # v1c (v6o W1)'''),
(r'''    neg = e < 0
    return float(e[neg].sum()), {"n_neg": int(neg.sum()), "n_pos": int((e > 0).sum()), "n_zero_1e-12": int((np.abs(e) < 1e-12).sum()), "sum_e": float(e.sum()),''',
 r'''    neg = e < 0
    return float(e[neg].sum()), {"n_neg": int(neg.sum()), "n_pos": int((e > 0).sum()), "n_zero_1e-12": int((np.abs(e) < 1e-12).sum()), "sum_e": float(e.sum()), "audit": W1S.spectral_audit(e),   # v1c (v6o W1) defect 13: Tr min(h, 0) = -Tr|h|/2'''),
(r'''    if route == "pair":
        return killing_vacuum_pair(h_csr, device, cluster_tol)''',
 r'''    if route == "pair":
        return killing_vacuum_pair(h_csr, device, cluster_tol)
    if route == "sector":                                                                                                     # v1c (v6o W1)
        return W1S.killing_vacuum_sector(h_csr, device)                                                                     # v1c (v6o W1)'''),
(r'''        self._masks = None
        self.ref_loader = None''',
 r'''        self._masks = None
        self.ref_loader = None
        self.sector_on = False; self.sector_check_now = False                                                                 # v1c (v6o W1): set by page_history_v6o per checkpoint
        self.w1_floor = None                                                                                                  # v1c (W1_fix_v1c): the propagation's floor (the impurity, the a priori bound), set by page_history_v6o per checkpoint'''),
(r'''        if self.vac is not None:
            self.vac["Vneg"] = None
            if self.dev.type == "cuda":
                torch.cuda.empty_cache()
        d = killing_vacuum(h_csr, self.dev, self.cfg["vac_route"], self.cfg["vac_cluster_tol"])''',
 r'''        if self.vac is not None:
            self.vac["Vneg"] = None
            self.vac["V_plus"] = None                                                                                         # v1c (v6o W1)
            if self.dev.type == "cuda":
                torch.cuda.empty_cache()
        d = killing_vacuum(h_csr, self.dev, self.cfg["vac_route"], self.cfg["vac_cluster_tol"])'''),
(r'''    def set_vacuum(self, t, k, e_np, Vneg, route, E_ref):
        self.vac = {"t": float(t), "k": int(k), "e": e_np, "Vneg": Vneg, "n_neg": int(Vneg.shape[1]), "E_ref": float(E_ref), "route": route, "seconds": 0.0,
                    "n_zero_1e-12": int((np.abs(e_np) < 1e-12).sum())}''',
 r'''    def set_vacuum(self, t, k, e_np, Vneg, route, E_ref, extra=None):                                                        # v1c (v6o W1): extra carries the sector data
        self.vac = {"t": float(t), "k": int(k), "e": e_np, "Vneg": Vneg, "n_neg": int(Vneg.shape[1]), "E_ref": float(E_ref), "route": route, "seconds": 0.0,
                    "n_zero_1e-12": int((np.abs(e_np) < 1e-12).sum())}
        if extra:                                                                                                             # v1c (v6o W1)
            self.vac.update(extra)                                                                                            # v1c (v6o W1)'''),
(r'''    def release_vacuum(self):
        if self.vac is not None:
            self.vac["Vneg"] = None''',
 r'''    def release_vacuum(self):
        if self.vac is not None:
            self.vac["Vneg"] = None
            self.vac["V_plus"] = None                                                                                         # v1c (v6o W1)'''),
(r'''                    for (hk, dt) in reversed(record_steps(kk)):
                        W, K_, tl = propagate(OPS.Banded(hk, self.dev), W, -dt, cfg["cheb_tol"], cfg["chunk"])''',
 r'''                    for (hk, dt) in reversed(record_steps(kk)):
                        W, K_, tl = propagate(hk if isinstance(hk, OPS.Banded) else OPS.Banded(hk, self.dev), W, -dt, cfg["cheb_tol"], cfg["chunk"])   # v1c (v6o W1): a banded operator per sub-step accepted'''),
(r'''        ent = exterior_entropies(Phi, self.R2, self.cuts, Vneg)
        if Vneg is not None and self.cfg.get("cut_norm", "on") == "on":                                 # v1b: the cross-cut block trace norms (EVOLVEDCUT EC1)
            cross_cut_readout(Phi, Vneg, self.R2, self.cuts, ent, self.cfg, self.kappa, trs, sc, arrays)
        sc["cuts"] = {}
        for lab, d in ent.items():
            for key in ("eig_X", "eig_Xc", "eig_X_vac", "eig_Xc_vac"):
                if isinstance(d.get(key), np.ndarray):
                    arrays["%s_cut%s" % (key, lab)] = d.pop(key)
            sc["cuts"][lab] = d''',
 r'''        if getattr(self, "sector_on", False) and (vac is None or vac.get("V_plus") is not None):                                 # v1c (v6o W1): the halved route (the + sector; the - beside at the check clocks)
            ent = W1S.exterior_entropies_sector(Phi, self.R, self.cuts, vac, check_minus=bool(getattr(self, "sector_check_now", False)), floor=getattr(self, "w1_floor", None))   # v1c (v6o W1; W1_fix_v1c: the floors beside every excess)
            sc["sector"] = ent.pop("_sector")                                                                                   # v1c (v6o W1)
            if vac is not None and self.cfg.get("cut_norm", "on") == "on" and bool(getattr(self, "sector_check_now", False)):     # v1c (v6o W1): the cross-cut norms at the check clocks
                W1S.cross_cut_readout_sector(Phi, vac, self.R, self.cuts, ent, self.cfg, self.kappa, trs, sc, arrays)             # v1c (v6o W1)
            sc["sector_line"] = W1S.sector_line(sc["sector"], ent)                                                              # v1c (v6o W1)
        else:                                                                                                                   # v1c (v6o W1): v1b's full route
            ent = exterior_entropies(Phi, self.R2, self.cuts, Vneg)
            if Vneg is not None and self.cfg.get("cut_norm", "on") == "on":                                 # v1b: the cross-cut block trace norms (EVOLVEDCUT EC1)
                cross_cut_readout(Phi, Vneg, self.R2, self.cuts, ent, self.cfg, self.kappa, trs, sc, arrays)
            if getattr(self, "w1_floor", None):                                                                                 # v1c (W1_fix_v1c): the floors on the full route (modes, no doubling)
                W1S.excess_floors(ent, self.R, self.cuts, self.w1_floor, sector=False)                                          # v1c (W1_fix_v1c)
        sc["cuts"] = {}
        for lab, d in ent.items():
            for key in [k_ for k_, v_ in d.items() if isinstance(v_, np.ndarray)]:                                              # v1c (v6o W1): every spectrum array (the sector route's included)
                arrays["%s_cut%s" % (key, lab)] = d.pop(key)
            sc["cuts"][lab] = d'''),
(r'''    sx = ", ".join("%s r_s %.6f%s%s" % (lab, d["S_X"], (" (vacuum %.6f, excess %.6f)" % (d["S_X_vac"], d["excess"])) if d.get("S_X_vac") is not None else "",
                                        (", t_X %.6f %s" % (d["t_X"], "ok" if d.get("bound_ok") else ("FAIL" if d.get("bound_ok") is False else "-"))) if d.get("t_X") is not None else "") for lab, d in cuts.items())''',
 r'''    sx = ", ".join("%s r_s %.6f%s%s%s" % (lab, d["S_X"], (" (vacuum %.6f, excess %.6f)" % (d["S_X_vac"], d["excess"])) if d.get("S_X_vac") is not None else "",   # v1c (W1_fix_v1c): the floor and the resolution beside the excess
                                          (", t_X %.6f %s" % (d["t_X"], "ok" if d.get("bound_ok") else ("FAIL" if d.get("bound_ok") is False else "-"))) if d.get("t_X") is not None else "",   # v1c (W1_fix_v1c)
                                          (", floor %.1e (low %.1e) %s" % (d["excess_floor_high"], d.get("excess_floor_low") or 0.0, d.get("excess_resolution") or "-")) if d.get("excess_floor_high") is not None else "") for lab, d in cuts.items())   # v1c (W1_fix_v1c)'''),
(r'''    return ("D13 kappa %d T/r_s %.3f: E_inv %s; S_X [%s] nats (complement gap max %.1e); ||C - P_-||_1 %s; energy beyond the cut [%s]; shells %s; collar window %s; source modes %s; "
            "first return (r_s) [%s]; enclosed source %s; %.0f s" % (kappa, sc["t_over_rs"], _f(E_inv, "%.6e"), sx, gap, tns, iv, shs, css, srcs, frs, ens, sc.get("seconds", 0.0)))''',
 r'''    return ("D13 kappa %d T/r_s %.3f: E_inv %s; S_X raw [%s] nats (complement gap max %.1e); ||C - P_-||_1 %s; energy beyond the cut [%s]; shells %s; collar window %s; source modes %s; "   # v1c (v6o W1): 'raw'
            "first return (r_s) [%s]; enclosed source %s; sector %s; %.0f s" % (kappa, sc["t_over_rs"], _f(E_inv, "%.6e"), sx, gap, tns, iv, shs, css, srcs, frs, ens, sc.get("sector_line", "-"), sc.get("seconds", 0.0)))   # v1c (v6o W1)'''),
(r'''                for key in ("S_X", "excess", "S_X_vac", "t_X", "abs_dS"):''',
 r'''                for key in ("S_X", "excess", "S_X_vac", "t_X", "abs_dS", "excess_floor_high", "excess_floor_low"):             # v1c (W1_fix_v1c): the floors summed with the weights'''),
(r'''    smax = max([s["share"] for s in shares], default=0.0)
    res = {"kappas": kap_sorted, "clocks": clocks, "table": table, "last_channel": K, "last_channel_hard_share_max": smax, "hard_moves_above_frac": bool(smax > hard_frac),
           "hard_frac": hard_frac, "shares": shares, "cut_labels": cut_labels,
           "rule": "the sum is extended beyond the last channel while the hard outgoing collar flow (the windowed reading above 25 T_H^ref) of the last channel exceeds hard_frac of the total at any readout clock from %g r_s" % from_rs,
           "label": "partial sums over kappa <= K with the weights 16 kappa; E_inv, the collar flow accumulated, the interval current and its comparators, the collar flow's soft/hard split, the exterior-cut entropies and their excess, the hard shells, the enclosed source"}''',
 r'''    smax = max([s["share"] for s in shares], default=0.0)
    w1 = W1S.convergence_rule(per, kap_sorted, clocks, cut_labels, hard_frac, d13_of, from_rs)                                   # v1c (v6o W1): the tail-majorant rule (defect 10)
    res = {"kappas": kap_sorted, "clocks": clocks, "table": table, "last_channel": K, "last_channel_hard_share_max": smax, "hard_moves_above_frac": bool(smax > hard_frac),
           "hard_frac": hard_frac, "shares": shares, "cut_labels": cut_labels,
           "retired_rule": "v6n's: the sum extended while the last channel's hard outgoing collar flow exceeded hard_frac of the total at a readout clock from %g r_s; printed, deciding nothing (it passes on the flat held vacuum with an infinite remainder of the raw sum)" % from_rs,   # v1c (v6o W1)
           "rule": w1["rule"], "w1": w1, "extend": bool(w1["extend"]), "verdict_line": w1["verdict_line"],                      # v1c (v6o W1)
           "label": "partial sums over kappa <= K with the weights 16 kappa; E_inv (= E_kappa, the formed-energy majorant's term), the collar flow accumulated, the interval current and its comparators, the collar flow's soft/hard split, the raw exterior-cut entropies (regulated), the held vacuum's and their excess (the readout) with the excess's propagation floors (v1c), the hard shells, the enclosed source"}   # v1c (v6o W1)'''),
(r'''        L = ["# %s: the convergence table in kappa (channels %s; weights 16 kappa)" % (tag, kap_sorted), "", res["label"], "", res["rule"] + ": the last channel's largest share %.4f (%s)" % (smax, "moves" if res["hard_moves_above_frac"] else "converged"), ""]''',
 r'''        L = ["# %s: the convergence table in kappa (channels %s; weights 16 kappa)" % (tag, kap_sorted), "", res["label"], "", res["verdict_line"], "",
             "the retired rule (v6n): the last channel's hard collar-flow share %.4f (%s by its 1-percent criterion); it decides nothing here" % (smax, "moves" if res["hard_moves_above_frac"] else "passes"), ""]   # v1c (v6o W1)'''),
(r'''        with open(os.path.join(out_dir, tag + "_ksum.md"), "w") as f:''',
 r'''        L += W1S.convergence_markdown(res["w1"])                                                                              # v1c (v6o W1): the majorant's tables (with the floors, v1c)
        with open(os.path.join(out_dir, tag + "_ksum.md"), "w") as f:'''),
]


def apply(src, repl, name):
    for i, (old, new) in enumerate(repl):
        c = src.count(old)
        if c != 1:
            raise SystemExit("%s: replacement %d: OLD occurs %d times (must be exactly once); OLD starts: %r" % (name, i + 1, c, old[:160]))
        src = src.replace(old, new)
    return src


def marker_check(old_src, new_src, markers, name):
    """every code line of the output that is not in the source must carry a marker (the module docstring is exempt)."""
    old_lines = set(old_src.splitlines())
    lines = new_src.splitlines()
    unmarked = []
    in_doc = False; doc_done = False
    for ln in lines:
        s = ln.strip()
        if not doc_done:
            if not in_doc and s.startswith('"""'):
                in_doc = True
                if s.count('"""') >= 2 and len(s) > 3:
                    in_doc = False; doc_done = True
                continue
            if in_doc:
                if '"""' in s:
                    in_doc = False; doc_done = True
                continue
        if ln in old_lines or not s:
            continue
        if not any(m in ln for m in markers):
            unmarked.append(ln)
    if unmarked:
        print("%s: WARNING: %d new lines without a marker:" % (name, len(unmarked)))
        for u in unmarked[:40]:
            print("   " + u.rstrip())
    return len(unmarked)


def md5(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src-v6n", required=True); ap.add_argument("--src-v1b", required=True)
    ap.add_argument("--sector-src", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "page_w1_sector_v1d.py"))   # v1d
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--check-only", action="store_true")
    a = ap.parse_args()
    v6n = open(a.src_v6n).read(); v1b = open(a.src_v1b).read()
    v6o = apply(v6n, REPL_V6O, "v6n -> v6o")
    v1d = apply(v1b, REPL_V1D, "v1b -> v1d")                                                                                   # v1d
    n1 = marker_check(v6n, v6o, ("v6o W1",), "page_history_v6o.py")
    n2 = marker_check(v1b, v1d, ("v1c", "v1d"), "page_d13_readout_v1d.py")                                                     # v1d: the v1c lines keep their marks
    print("v6n -> v6o: %d replacements applied, %d unmarked new lines; v1b -> v1d: %d replacements, %d unmarked (builder v1d, W1_fix_v1d.md)" % (len(REPL_V6O), n1, len(REPL_V1D), n2))
    if a.check_only:
        print("check only: nothing written")
        return
    os.makedirs(a.out_dir, exist_ok=True)
    p6 = os.path.join(a.out_dir, "page_history_v6o.py"); p1 = os.path.join(a.out_dir, "page_d13_readout_v1d.py"); ps = os.path.join(a.out_dir, "page_w1_sector_v1d.py")   # v1d
    open(p6, "w").write(v6o); open(p1, "w").write(v1d)                                                                           # v1d
    if os.path.abspath(a.sector_src) != os.path.abspath(ps):
        shutil.copyfile(a.sector_src, ps)
    for p in (p6, p1, ps):
        py_compile.compile(p, doraise=True)
        print("%s  %s" % (md5(p), p))
    print("sources: %s %s; %s %s" % (md5(a.src_v6n), a.src_v6n, md5(a.src_v1b), a.src_v1b))


if __name__ == "__main__":
    main()
