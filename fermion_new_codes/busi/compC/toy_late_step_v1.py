#!/usr/bin/env python
"""toy_late_step_v1.py (busi lane builder, 20 Oct; ReviewA_busi_open_q12_v1.md section 1): the late-step two-sample toy beside the review's x
fixture (C64_v9c_review_v1 finding 9) of engine_twofield_v9c_selftest.py, in the same fixture class (the layer's 8-level heavy-like toys through
decide() -> v9c_decide -> assign_primitive follow_chord) and the same units (tau 2, levels 0 .. 7, the set the four lowest, member basis 3).

The ruling's q(s) = A [0.01 + (s - 0.9)^2 + c (s - 1/2)(s - 1)(s - 3/2)], A = the existing second passage's scale. In x's toy the second passage is
the member (slope 2.4) crossing the flat level 5 exactly, so its splitting is Delta = 2.4 |s - s*| and Delta^2 = 5.76 (s - s*)^2: A = 2.4^2 = 5.76.
The primitive fits the parabola Delta^2 = alpha_s^2 (s - s*)^2 + gap_min^2 (fit_pair: y = Delta^2), so q is mapped to the GAP SQUARED: the pair's
splitting is Delta(s) = sqrt(q(s)), and the fit's alpha_s^2 is the curvature A (alpha_s 2.4, as x's second passage) and gap_min^2 its minimum.

The Hamiltonian: levels 0, 1, 2, 5, 6, 7 flat; the (3, 4) block centred at 3.5 with H33 = 3.5 - D/2, H44 = 3.5 + D/2, H34 = V, where
D = sqrt(q) t(s), 2 V = sqrt(q) sqrt(1 - t^2), so the block's eigenvalues are 3.5 -/+ sqrt(q)/2 exactly (the splitting is sqrt(q) whatever t is)
and t(s) = cos(theta), the mixing angle, sets only the character: t = -tanh((s - s_sw)/w), w = 0.01. The member (basis 3, occupied) is the lower
level while t > 0 and the upper after the switch, so the set changes index 3 -> 4 at the switch (a passage). X and R switch at s_sw = 0.96875,
inside the last piece of every level 1 .. 4 (the points 1 - 2^-l are clean before, s = 1 clean after: the passage is completed inside the chord's
last piece at every level, the two-sample case); C switches at its minimum, s_sw = 0.375.
  X (extension): c = 0.4, the path extends (v9c_extends True: the s = 1.5 state supplied by the same h(s)).
  C (control):   c = 0, the minimum moved to s* = 0.375: q = A [0.01 + (s - 0.375)^2].
  R (refusal):   X's h(s) with the s > 1 states withheld (v9c_extends False and no fields: V9cSpectra._extend refuses).

Read where x reads (entries_of(info) of v9c_decide: the reported, i.e. first stable, level) and, beside it, every level's events of the
primitive (the Decision follow_chord returned, captured by wrapping the module's follow_chord for the call), whose 'samples' carry the
pair's (s, Delta) samples; Delta^2 / A is printed.

run_late_step_checks(C, L) is called by engine_twofield_v9c_selftest_v1y.py after x; standalone: python toy_late_step_v1.py [--engine PATH]
(the built engine's namespace instead of the builder's block) [--prim MODULE].
"""
import argparse
import importlib.util
import math
import os
import sys

import numpy as np

A_SCALE = 2.4 ** 2          # x's second passage: Delta^2 = 5.76 (s - s*)^2
TAU = 2.0                   # x's tau
SW_X = 0.96875              # the switch inside the last piece of levels 1 .. 4 (1 - 2^-5)
SW_W = 0.01
VARIANTS = {"X": dict(c=0.4, s0=0.9, sw=SW_X, extends=True), "C": dict(c=0.0, s0=0.375, sw=0.375, extends=True), "R": dict(c=0.4, s0=0.9, sw=SW_X, extends=False)}


def q_over_A(s, c, s0):
    return 0.01 + (s - s0) ** 2 + c * (s - 0.5) * (s - 1.0) * (s - 1.5)


def late_h(c, s0, sw, A=A_SCALE, w=SW_W):
    def h(s):
        q = A * q_over_A(s, c, s0)
        t = -math.tanh((s - sw) / w)
        r = math.sqrt(q)
        D = r * t; V = 0.5 * r * math.sqrt(max(0.0, 1.0 - t * t))
        E = np.array([0.0, 1.0, 2.0, 0.0, 0.0, 5.0, 6.0, 7.0])
        H = np.diag(E); H[3, 3] = 3.5 - 0.5 * D; H[4, 4] = 3.5 + 0.5 * D; H[3, 4] = V; H[4, 3] = V
        return H
    return h


def _toy_classes(ToyPath):
    class LoggedPath(ToyPath):
        """x's ToyPath with every eig(s) logged (the extension shows as an s > 1 call)"""
        def __init__(self, h, extends=True):
            ToyPath.__init__(self, h); self.calls = []; self._ext = bool(extends)

        @property
        def v9c_extends(self):
            return self._ext

        def eig(self, s):
            self.calls.append(float(s))
            return ToyPath.eig(self, s)
    return LoggedPath


def run_variant(L, ToyPath, decide, entries_of, idx_mask, name, mode):
    v = VARIANTS[name]
    LoggedPath = _toy_classes(ToyPath)
    path = LoggedPath(late_h(v["c"], v["s0"], v["sw"]), extends=v["extends"])
    prim = L["v9c_prim"](mode["prim_module"]); orig = prim.follow_chord; cap = {}

    def capture(*a, **k):
        d = orig(*a, **k); cap["dec"] = d
        return d
    prim.follow_chord = capture
    try:
        MASK = idx_mask(8, [0, 1, 2, 3])
        m, info, spec = decide(L, None, MASK, TAU, mode, path=path)
    finally:
        prim.follow_chord = orig
    dec = cap.get("dec")
    levels = []
    if dec is not None:
        ch = dec.channels[0]
        for res in dec.results[ch]:
            evs = []
            for ev in res["events"]:
                evs.append({k: ev.get(k) for k in ("kind", "s_p", "s_c", "s_star", "gap_min2", "gap_min2_raw", "alpha_s", "alpha_T", "A", "fit_kind", "fit_ok", "fit_why", "decision", "decided", "resolved", "block", "flags", "samples", "n_samples")})
            levels.append({"level": res["level"], "events": evs, "flags": list(res["flags"])})
    return {"name": name, "mask": np.asarray(m, bool), "info": info, "entries": entries_of(info), "levels": levels, "calls": list(path.calls), "spec": spec}


def _f(x, fmt="%.12g"):
    return "None" if x is None else (fmt % x)


def describe(r):
    out = []
    i = r["info"]
    out.append("y %s: reported level %s (stable %s), extension %r, path eig s > 1: %s, mask %s, n_spectra %s" % (
        r["name"], i.get("level_stable"), i.get("stable"), i.get("extension"), [s for s in r["calls"] if s > 1.0], np.asarray(r["mask"]).astype(int).tolist(), i.get("n_spectra")))
    for e in r["entries"]:
        out.append("y %s   entry (x's read): kind %s branch %s decided %s flagged %s fit %s s* %s gap_min^2/A %s alpha_s %s tau_lz %s exponent %s flags %s" % (
            r["name"], e.get("kind"), e.get("branch"), e.get("decided"), e.get("flagged"), e.get("fit_kind"), _f(e.get("s_star")),
            _f(None if e.get("gap_min") is None else e["gap_min"] ** 2 / A_SCALE), _f(e.get("alpha_s")), _f(e.get("tau_lz")), _f(e.get("exponent")), e.get("flags")))
    for lv in r["levels"]:
        for ev in lv["events"]:
            out.append("y %s   level %d event: s_p %s s_c %s samples (s, Delta^2/A) %s fit %s ok %s why %r s* %s gap_min^2/A %s (raw %s) alpha_s %s decision %s decided %s resolved %s flags %s" % (
                r["name"], lv["level"], ev["s_p"], ev["s_c"], [(round(s, 6), "%.15g" % (d * d / A_SCALE)) for s, d in (ev["samples"] or [])], ev["fit_kind"], ev["fit_ok"], ev["fit_why"],
                _f(ev["s_star"]), _f(None if ev["gap_min2"] is None else ev["gap_min2"] / A_SCALE), _f(None if ev["gap_min2_raw"] is None else ev["gap_min2_raw"] / A_SCALE), _f(ev["alpha_s"]),
                ev["decision"], ev["decided"], ev["resolved"], ev["flags"]))
    return out


def rel(a, b):
    return abs(a - b) / max(abs(b), 1e-300)


def sample_at(ev, s):
    for s_, d in (ev.get("samples") or []):
        if abs(s_ - s) <= 1e-15:
            return d
    return None


def judge(rX, rC, rR, at):
    """the ruling's pass conditions 2 to 4 at `at`: 'reported' (x's read: v9c_decide's entries, i.e. the first stable level; R and C events at
    their own reported levels, R's samples compared with X's at X's reported level) or a level number (that level's events of the primitive for
    all three). Returns [(name, ok, detail)]."""
    def lev(r):
        return (r["info"].get("level_stable") or (r["levels"][-1]["level"] if r["levels"] else None)) if at == "reported" else at

    def ev_at(r, lv):
        for L_ in r["levels"]:
            if L_["level"] == lv:
                ps = [e for e in L_["events"] if e["kind"] == "passage"]
                return ps[0] if ps else None
        return None
    lX, lC, lR = lev(rX), lev(rC), lev(rR)
    eX, eC = ev_at(rX, lX), ev_at(rC, lC)
    eR = ev_at(rR, lX)          # R at X's level: the same chord points
    rows = []
    entX = rX["entries"][0] if rX["entries"] else {}; entC = rC["entries"][0] if rC["entries"] else {}; entR = rR["entries"][0] if rR["entries"] else {}
    # 2. X
    if at == "reported":
        s_star = entX.get("s_star"); gm2 = None if entX.get("gap_min") is None else entX["gap_min"] ** 2
        carries = any("third sample at s=1.5000" in f for f in (entX.get("flags") or []))
        in_record = "samples" in entX
        smp = {"samples": entX.get("samples")} if in_record else (eX or {})
    else:
        s_star = eX and eX["s_star"]; gm2 = eX and eX["gap_min2"]
        carries = bool(eX) and any("third sample at s=1.5000" in f for f in (eX.get("flags") or []))
        in_record = None; smp = eX or {}
    d15 = sample_at(smp, 1.5)
    q15 = None if d15 is None else d15 * d15
    ok2a = d15 is not None and rel(q15, 0.37 * A_SCALE) <= 1e-12 and carries and in_record is not False
    ok2b = s_star is not None and gm2 is not None and rel(s_star, 0.9) <= 1e-10 and rel(gm2, 0.01 * A_SCALE) <= 1e-10
    rows.append(("y X 2a (%s, level %s) the passage record carries the s = 1.5 sample = 0.37 A to 1e-12" % (at, lX), ok2a,
                 "sample Delta^2/A at 1.5: %s (rel err %s); flag 'third sample at s=1.5000' %s; the json entry carries 'samples': %s; X's samples (s, Delta^2/A) at this level %s" % (
                     _f(None if q15 is None else q15 / A_SCALE, "%.15g"), _f(None if q15 is None else rel(q15, 0.37 * A_SCALE), "%.2e"), carries, in_record,
                     [(round(a_, 6), "%.12g" % (b_ * b_ / A_SCALE)) for a_, b_ in ((eX or {}).get("samples") or [])])))
    rows.append(("y X 2b (%s, level %s) the fitted vertex 0.9 and minimum 0.01 A to 1e-10" % (at, lX), ok2b,
                 "s* %s (rel %s) gap_min^2/A %s (rel %s)" % (_f(s_star, "%.15g"), _f(None if s_star is None else rel(s_star, 0.9), "%.2e"), _f(None if gm2 is None else gm2 / A_SCALE, "%.15g"), _f(None if gm2 is None else rel(gm2, 0.01 * A_SCALE), "%.2e"))))
    # 3. C
    if at == "reported":
        sC, gC, aC, brC, tC = entC.get("s_star"), (None if entC.get("gap_min") is None else entC["gap_min"] ** 2), entC.get("alpha_s"), entC.get("branch"), entC.get("tau_lz")
        gX, aX, brX, tX = (None if entX.get("gap_min") is None else entX["gap_min"] ** 2), entX.get("alpha_s"), entX.get("branch"), entX.get("tau_lz")
    else:
        sC, gC, aC, brC, tC = (eC and eC["s_star"]), (eC and eC["gap_min2"]), (eC and eC["alpha_s"]), (eC and eC["decision"]), (eC and eC["A"])
        gX, aX, brX, tX = (eX and eX["gap_min2"]), (eX and eX["alpha_s"]), (eX and eX["decision"]), (eX and eX["A"])
    no_ext = not any(s_ > 1.0 for s_ in rC["calls"]) and rC["info"].get("extension") is None and all(sample_at(e_, 1.5) is None for L_ in rC["levels"] for e_ in L_["events"])
    ok3a = no_ext and sC is not None and gC is not None and rel(sC, 0.375) <= 1e-10 and rel(gC, 0.01 * A_SCALE) <= 1e-10
    ok3b = (None not in (gC, gX, aC, aX, tC, tX)) and rel(gC, gX) <= 1e-10 and rel(aC, aX) <= 1e-10 and brC == brX and rel(tC, tX) <= 1e-9
    rows.append(("y C 3a (%s, level %s) no extension call, no s = 1.5 sample; vertex 0.375, minimum 0.01 A to 1e-10" % (at, lC), ok3a,
                 "path eig s > 1: %s; extension %r; s* %s gap_min^2/A %s" % ([s_ for s_ in rC["calls"] if s_ > 1.0], rC["info"].get("extension"), _f(sC, "%.15g"), _f(None if gC is None else gC / A_SCALE, "%.15g"))))
    rows.append(("y C 3b (%s, levels C %s / X %s) the same fitted minimum and curvature as X, the same rate and decision" % (at, lC, lX), ok3b,
                 "gap_min^2/A C %s X %s; alpha_s C %s X %s; tau_LZ C %s X %s; branch C %s X %s" % (_f(None if gC is None else gC / A_SCALE), _f(None if gX is None else gX / A_SCALE), _f(aC), _f(aX), _f(tC), _f(tX), brC, brX)))
    # 4. R
    refusal = rR["info"].get("extension")
    refused_flag = bool(eR) and any("refused by the oracle" in f for f in (eR.get("flags") or []))
    if at == "reported":
        refused_flag = refused_flag or any("refused by the oracle" in f for f in (entR.get("flags") or []))
        outR = (entR.get("branch"), entR.get("decided"), entR.get("flagged"), entR.get("fit_kind"), entR.get("s_star"), entR.get("gap_min"))
        outX = (entX.get("branch"), entX.get("decided"), entX.get("flagged"), entX.get("fit_kind"), entX.get("s_star"), entX.get("gap_min"))
    else:
        outR = eR and (eR["decision"], eR["decided"], eR["resolved"], eR["fit_kind"], eR["s_star"], eR["gap_min2"])
        outX = eX and (eX["decision"], eX["decided"], eX["resolved"], eX["fit_kind"], eX["s_star"], eX["gap_min2"])
    pts = [eX["s_p"], 1.0] if eX else []
    bitwise = bool(eR and eX) and all(sample_at(eR, p_) is not None and sample_at(eX, p_) is not None and sample_at(eR, p_) == sample_at(eX, p_) for p_ in pts)
    no15 = not any(s_ > 1.0 for s_ in rR["calls"]) and all(sample_at(e_, 1.5) is None for L_ in rR["levels"] for e_ in L_["events"]) and ("samples" not in entR or sample_at(entR, 1.5) is None)
    ok4 = isinstance(refusal, str) and refused_flag and no15 and outR != outX and bitwise
    rows.append(("y R 4 (%s, R level %s, samples at X's level %s) refusal text recorded, no s = 1.5 sample, the outcome differs from X's, the in-step samples at s_p and 1 bitwise X's" % (at, lR, lX), ok4,
                 "extension %r; 'refused by the oracle' flag %s; no 1.5 sample or s > 1 call %s; outcome R %s X %s; in-step points %s: R %s X %s; bitwise %s" % (
                     refusal, refused_flag, no15, outR, outX, pts, [None if eR is None else sample_at(eR, p_) for p_ in pts], [None if eX is None else sample_at(eX, p_) for p_ in pts], bitwise)))
    return rows


def x_predicates(rX, rR, hold_fn):
    """INFO: the review's x predicates (lines 447-449 and 453-456 of engine_twofield_v9c_selftest.py) transcribed onto the one-passage X and R,
    with x's toy-specific numbers (two passages, s* 0.4167 / 0.8333, gap 0, the band top 0.133, the set at index 5) dropped"""
    out = []
    e = rX["entries"][-1] if rX["entries"] else {}
    okx = (bool(e) and e.get("branch") == "diabatic" and e.get("decided") and not e.get("flagged") and e.get("fit_kind") == "parabola" and any("third sample at s=1.5000" in f for f in e.get("flags") or [])
           and rX["info"]["stable"] and rX["info"].get("extension") is None and not hold_fn(rX["info"]))
    out.append(("x-on-X: diabatic, decided, not flagged, fit parabola, 'third sample at s=1.5000', stable, extension None, no hold", okx,
                "branch %s decided %s flagged %s fit %s flags %s stable %s extension %r hold %s" % (e.get("branch"), e.get("decided"), e.get("flagged"), e.get("fit_kind"), e.get("flags"), rX["info"]["stable"], rX["info"].get("extension"), hold_fn(rX["info"]))))
    e = rR["entries"][-1] if rR["entries"] else {}
    okr = (bool(e) and e.get("branch") == "diabatic" and e.get("decided") and not e.get("flagged") and e.get("fit_kind") == "grid" and any("refused by the oracle" in f for f in e.get("flags") or [])
           and e.get("exponent") == 0.0 and isinstance(rR["info"].get("extension"), str) and rR["info"]["stable"] and not hold_fn(rR["info"]))
    out.append(("x-on-R: diabatic, decided, not flagged, fit grid, 'refused by the oracle', exponent 0, the refusal text, stable, no hold", okr,
                "branch %s decided %s flagged %s fit %s exponent %s flags %s extension %r stable %s hold %s" % (e.get("branch"), e.get("decided"), e.get("flagged"), e.get("fit_kind"), e.get("exponent"), e.get("flags"), rR["info"].get("extension"), rR["info"]["stable"], hold_fn(rR["info"]))))
    return out


def run_late_step_checks(C, L, ST):
    """ST: the self-test module (ToyPath, decide, entries_of, idx_mask, lz_mode, hold) so the toy is in x's class"""
    mode = ST.lz_mode(L)
    print("y late-step toy (ReviewA_busi_open_q12_v1 sec. 1): A = %.4g (x's second passage, slope 2.4), q mapped to the gap squared; tau %g; mode prim %s levels %s w_mix %s; switch X/R at s %.5f (w %.3g), C at 0.375" % (
        A_SCALE, TAU, mode["prim_module"], mode["prim_levels"], mode["prim_w_mix"], SW_X, SW_W), flush=True)
    for nm, v in VARIANTS.items():
        print("y %s: c %g, s0 %g; q/A at s = 0, 1/2, 1, 3/2: %s" % (nm, v["c"], v["s0"], ["%.6g" % q_over_A(s, v["c"], v["s0"]) for s in (0.0, 0.5, 1.0, 1.5)]), flush=True)
    R = {}
    for nm in ("X", "C", "R"):
        R[nm] = run_variant(L, ST.ToyPath, ST.decide, ST.entries_of, ST.idx_mask, nm, mode)
        for line in describe(R[nm]):
            print(line, flush=True)
    for name, ok, det in judge(R["X"], R["C"], R["R"], "reported"):
        C.ok(name, ok, det)
    for name, ok, det in judge(R["X"], R["C"], R["R"], 1):
        print("INFO %s %s: %s" % ("PASS" if ok else "FAIL", name, det), flush=True)
    hold_fn = lambda info: ST.hold(L, info)
    for name, ok, det in x_predicates(R["X"], R["R"], hold_fn):
        print("INFO %s %s: %s" % ("PASS" if ok else "FAIL", name, det), flush=True)
    return R


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default=None, help="a built engine (its module namespace is the layer); default: the builder's block")
    ap.add_argument("--selftest", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "engine_twofield_v9c_selftest.py"))
    ap.add_argument("--prim", default=None, help="the primitive module name (default the layer's V9C_DEFAULTS)")
    a = ap.parse_args(argv)
    here = os.path.dirname(os.path.abspath(a.selftest))
    if here not in sys.path:
        sys.path.insert(0, here)
    spec = importlib.util.spec_from_file_location("v9c_selftest_for_y", a.selftest); ST = importlib.util.module_from_spec(spec); spec.loader.exec_module(ST)
    if a.engine:
        spec = importlib.util.spec_from_file_location("engine_twofield_v9c", a.engine); mod = importlib.util.module_from_spec(spec); sys.modules["engine_twofield_v9c"] = mod; spec.loader.exec_module(mod)
        L = vars(mod); src = a.engine
    else:
        L = ST.B.load_block_namespace(); src = "the builder's block %s" % ST.B.__file__
    if a.prim:
        L["V9C_DEFAULTS"] = dict(L["V9C_DEFAULTS"], prim_module=a.prim)
    print("y on %s; primitive %s" % (src, L["v9c_prim"](L["V9C_DEFAULTS"]["prim_module"]).VERSION), flush=True)
    C = ST.Checks()
    run_late_step_checks(C, L, ST)
    failed = C.failed()
    print("toy_late_step_v1: %s (%d of %d checks failed)%s" % ("FAIL" if failed else "PASS", len(failed), len(C.rows), (": " + "; ".join(failed)) if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
