#!/usr/bin/env python
"""toy_late_step_v2.py (busi lane builder, 20 Oct; REVIEWA_C64_toy_readout_v1.md section 2): the late-step toy's pass lines y1 to y8, all C.ok lines,
over toy_late_step_v1.py (c58f0cbe), which supplies the fixture unchanged (the variants X, C, R, late_h, q_over_A, run_variant, describe): v1's judge
(the conditions at x's read and their level-1 INFO duplicates) and its transcribed x predicates are not used. Every expected number is computed here
from the toy's own q(s) (Delta = sqrt(q)): the parabola through q at the samples' s, the grid formula on sqrt(q); the rule's k and hbar from the
layer's v9c_rule. Tolerances relative. "Entry": x's read (v9c_decide's entries, the reported level). "Level-l event": the primitive's passage event
at level l from the captured Decision.

  y1 X, the extension at the reported level (2, stable): one passage entry, extension None, the path's s > 1 calls exactly [1 + 2^-1, 1 + 2^-2],
     the entry's 3 samples at 1 - h, 1, 1 + h (h = 2^-2; to 1e-15 in s), the s = 1 + h sample's Delta^2 = q(1 + h) to 1e-12, the extension flag
  y2 X, the fit at the reported level: parabola, diabatic, decided, not flagged; s*, gap_min^2, alpha_s the parabola through q at 1 - h, 1, 1 + h to
     1e-10; tau_lz = k gap_min^2 tau / (hbar alpha_s) to 1e-10
  y3 X, the fit at level 1: the samples (0.5, q), (1, q), (1.5, q) in Delta^2 to 1e-12; the flag "third sample at s=1.5000"; parabola, resolved,
     decided diabatic; s*, gap_min^2, alpha_s the parabola through q there to 1e-10 (q's cubic vanishes at those points: 0.9, 0.01 A, 2.4)
  y4 C: level 2; no s > 1 call, no s > 1 sample at any level, extension None; the entry parabola, diabatic, decided; s*, gap_min^2, alpha_s q's
     (an exact parabola) to 1e-10, tau_lz from them; the level-1 event the same s*, gap_min^2, alpha_s
  y5 C against X: at the reported level alpha_s equal, the branch equal, X's gap_min^2 over C's the ratio of the q-parabolas' minima; at level 1 the
     fitted minima equal and tau_LZ equal
  y6 R at its reported level l_R: the refusal text ("no fields to extend"), no s > 1 call or sample; the entry fit grid, gap_min 0, the event's
     gap_min2 0, exponent 0, diabatic, decided, not flagged, the refusal and grid-bound flags, 2 samples at 1 - 2^-l_R and 1, s* and alpha_s the
     grid formula on sqrt(q) to 1e-10; the outcome differs from X's
  y7 R against X: R's in-step samples bitwise X's at levels 2 (s = 0.75, 1) and 1 (s = 0.5, 1)
  y8 R at level 1: fit grid, the refusal flag at s=1.5000 and "two samples only", the samples (0.5, sqrt q) and (1, sqrt q) (s to 1e-15, Delta
     to 1e-12), gap_min2 0, s* and alpha_s the grid formula to 1e-10, decided diabatic, the outcome differs from X's level 1

run_late_step_checks(C, L, ST) is called by engine_twofield_v9c_selftest_v1z.py after x; standalone: python toy_late_step_v2.py [--engine PATH]
[--selftest engine_twofield_v9c_selftest_v1z.py] [--prim MODULE].
"""
import argparse
import importlib.util
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import toy_late_step_v1 as T1  # noqa: E402  the fixture, unchanged

A_SCALE = T1.A_SCALE
TAU = T1.TAU
VARIANTS = T1.VARIANTS


def q_of(name, s):
    v = VARIANTS[name]
    return A_SCALE * T1.q_over_A(float(s), v["c"], v["s0"])


def delta_of(name, s):
    return math.sqrt(q_of(name, s))


def rel_ok(a, b, tol):
    if a is None or b is None:
        return False
    a = float(a); b = float(b)
    return a == b or abs(a - b) <= tol * max(abs(b), 1e-300)


def parabola_q(name, svals):
    """the parabola y = alpha_s^2 (s - s*)^2 + gap_min^2 through (s, q(s)) at three s: (s*, gap_min^2 clamped at 0, alpha_s)"""
    (s0, s1, s2) = sorted(float(s) for s in svals)
    y0, y1, y2 = q_of(name, s0), q_of(name, s1), q_of(name, s2)
    a = ((y2 - y1) / (s2 - s1) - (y1 - y0) / (s1 - s0)) / (s2 - s0)
    b = (y1 - y0) / (s1 - s0) - a * (s0 + s1)
    c = y0 - a * s0 * s0 - b * s0
    return -b / (2.0 * a), max(c - b * b / (4.0 * a), 0.0), math.sqrt(a)


def three_around_min(name, svals):
    """the three s around the smallest sqrt(q) among svals (fit_pair's choice)"""
    pts = sorted(float(s) for s in svals)
    if len(pts) < 3:
        return None
    m = min(range(len(pts)), key=lambda i: (q_of(name, pts[i]), i))
    lo = min(max(m - 1, 0), len(pts) - 3)
    return pts[lo:lo + 3]


def grid_q(name, sp, sc):
    """the grid formula on Delta = sqrt(q): s* = (s_p D_c + s_c D_p)/(D_p + D_c), alpha_s = (D_p + D_c)/(s_c - s_p)"""
    dp, dc = delta_of(name, sp), delta_of(name, sc)
    return (sp * dc + sc * dp) / (dp + dc), (dp + dc) / (sc - sp)


def smp(x):
    return sorted((float(a), float(b)) for a, b in ((x or {}).get("samples") or []))


def ev_at(r, lv):
    for L_ in r["levels"]:
        if L_["level"] == lv:
            ps = [e for e in L_["events"] if e["kind"] == "passage"]
            return ps[0] if ps else None
    return None


def has_flag(x, text):
    return any(text in f for f in ((x or {}).get("flags") or []))


def any_sample_above_1(r, ent=None):
    if any(s > 1.0 for L_ in r["levels"] for e in L_["events"] for s, _d in (e.get("samples") or [])):
        return True
    return ent is not None and any(s > 1.0 for s, _d in smp(ent))


def gm2_entry(e):
    return None if e.get("gap_min") is None else e["gap_min"] ** 2


def judge_y(rX, rC, rR, rule):
    """the ruling's pass lines y1 to y8: [(name, ok, detail)]"""
    k, hbar = float(rule.k), float(rule.hbar)
    rows = []
    iX, iC, iR = rX["info"], rC["info"], rR["info"]
    eX = rX["entries"][0] if len(rX["entries"]) == 1 else {}
    eC = rC["entries"][0] if rC["entries"] else {}
    eR = rR["entries"][0] if rR["entries"] else {}
    lX = iX.get("level_stable"); hX = 2.0 ** -int(lX) if lX else None
    # y1
    calls_hi = [s for s in rX["calls"] if s > 1.0]
    want_calls = [1.0 + 2.0 ** -1, 1.0 + 2.0 ** -2]
    sX = smp(eX); want_s = [1.0 - hX, 1.0, 1.0 + hX] if hX else []
    s_ok = len(sX) == 3 and all(abs(a - b) <= 1e-15 for (a, _d), b in zip(sX, want_s))
    q_ext = q_of("X", 1.0 + hX) if hX else None
    d_ext = sX[2][1] if len(sX) == 3 else None
    flag1 = ("the pair's third sample at s=%.4f on the straight chord extended by one piece" % (1.0 + hX)) if hX else None
    ok1 = (lX == 2 and iX.get("stable") is True and len(rX["entries"]) == 1 and eX.get("kind") == "passage" and iX.get("extension") is None and calls_hi == want_calls
           and s_ok and d_ext is not None and rel_ok(d_ext * d_ext, q_ext, 1e-12) and flag1 is not None and has_flag(eX, flag1))
    rows.append(("y1 X, the extension at the reported level: level 2 stable, one passage entry, extension None, the s > 1 calls [1.5, 1.25], 3 samples at 0.75, 1, 1.25, the s = 1.25 sample's Delta^2 = q(1.25) to 1e-12, the extension flag", ok1,
                 "level %s stable %s entries %d kind %s extension %r; s > 1 calls %s (want %s); samples (s, Delta^2/A) %s; Delta^2/A at %s: %s against q/A %s (rel %s); flag %r %s" % (
                     lX, iX.get("stable"), len(rX["entries"]), eX.get("kind"), iX.get("extension"), calls_hi, want_calls, [(a, "%.15g" % (b * b / A_SCALE)) for a, b in sX],
                     None if hX is None else 1.0 + hX, None if d_ext is None else "%.15g" % (d_ext * d_ext / A_SCALE), None if q_ext is None else "%.15g" % (q_ext / A_SCALE),
                     None if (d_ext is None or q_ext is None) else "%.2e" % (abs(d_ext * d_ext - q_ext) / q_ext), flag1, has_flag(eX, flag1) if flag1 else None)))
    # y2
    pX = parabola_q("X", want_s) if hX else (None, None, None)
    tX = (k * pX[1] * TAU / (hbar * pX[2])) if pX[2] else None
    ok2 = (eX.get("fit_kind") == "parabola" and eX.get("branch") == "diabatic" and eX.get("decided") is True and eX.get("flagged") is False
           and rel_ok(eX.get("s_star"), pX[0], 1e-10) and rel_ok(gm2_entry(eX), pX[1], 1e-10) and rel_ok(eX.get("alpha_s"), pX[2], 1e-10) and rel_ok(eX.get("tau_lz"), tX, 1e-10))
    rows.append(("y2 X, the fit at the reported level: parabola, diabatic, decided, not flagged; s*, gap_min^2, alpha_s the parabola through q at 1 - h, 1, 1 + h to 1e-10; tau_lz = k gap_min^2 tau/(hbar alpha_s) to 1e-10", ok2,
                 "fit %s branch %s decided %s flagged %s; s* %s gap_min^2/A %s alpha_s %s tau_lz %s; q's parabola s* %s gap_min^2/A %s alpha_s %s tau_lz %s" % (
                     eX.get("fit_kind"), eX.get("branch"), eX.get("decided"), eX.get("flagged"), T1._f(eX.get("s_star"), "%.15g"), T1._f(None if gm2_entry(eX) is None else gm2_entry(eX) / A_SCALE, "%.15g"),
                     T1._f(eX.get("alpha_s"), "%.15g"), T1._f(eX.get("tau_lz"), "%.15g"), T1._f(pX[0], "%.15g"), T1._f(None if pX[1] is None else pX[1] / A_SCALE, "%.15g"), T1._f(pX[2], "%.15g"), T1._f(tX, "%.15g"))))
    # y3
    e1 = ev_at(rX, 1) or {}
    s1 = smp(e1); w1 = [0.5, 1.0, 1.5]
    smp1_ok = len(s1) == 3 and all(abs(a - b) <= 1e-15 and rel_ok(d * d, q_of("X", b), 1e-12) for (a, d), b in zip(s1, w1))
    p1 = parabola_q("X", w1)
    ok3 = (smp1_ok and has_flag(e1, "third sample at s=1.5000") and e1.get("fit_kind") == "parabola" and e1.get("resolved") is True and e1.get("decided") is True and e1.get("decision") == "diabatic"
           and rel_ok(e1.get("s_star"), p1[0], 1e-10) and rel_ok(e1.get("gap_min2"), p1[1], 1e-10) and rel_ok(e1.get("alpha_s"), p1[2], 1e-10))
    rows.append(("y3 X, the fit at level 1 (the ruling's numbers): samples (0.5, q), (1, q), (1.5, q) in Delta^2 to 1e-12, the flag 'third sample at s=1.5000', parabola, resolved, decided diabatic; s*, gap_min^2, alpha_s q's parabola there to 1e-10", ok3,
                 "samples (s, Delta^2/A) %s against q/A %s; flag %s; fit %s resolved %s decided %s decision %s; s* %s gap_min^2/A %s alpha_s %s; q's parabola %s %s %s" % (
                     [(a, "%.15g" % (d * d / A_SCALE)) for a, d in s1], ["%.15g" % (q_of("X", b) / A_SCALE) for b in w1], has_flag(e1, "third sample at s=1.5000"), e1.get("fit_kind"), e1.get("resolved"), e1.get("decided"), e1.get("decision"),
                     T1._f(e1.get("s_star"), "%.15g"), T1._f(None if e1.get("gap_min2") is None else e1["gap_min2"] / A_SCALE, "%.15g"), T1._f(e1.get("alpha_s"), "%.15g"), "%.15g" % p1[0], "%.15g" % (p1[1] / A_SCALE), "%.15g" % p1[2])))
    # y4
    lC = iC.get("level_stable")
    tri_C = three_around_min("C", [a for a, _d in smp(eC)])
    pC = parabola_q("C", tri_C) if tri_C else (None, None, None)
    tC = (k * pC[1] * TAU / (hbar * pC[2])) if pC[2] else None
    eC1 = ev_at(rC, 1) or {}
    no_ext_C = (not any(s > 1.0 for s in rC["calls"])) and (not any_sample_above_1(rC, eC)) and iC.get("extension") is None
    ok4 = (lC == 2 and no_ext_C and eC.get("fit_kind") == "parabola" and eC.get("branch") == "diabatic" and eC.get("decided") is True
           and rel_ok(eC.get("s_star"), pC[0], 1e-10) and rel_ok(gm2_entry(eC), pC[1], 1e-10) and rel_ok(eC.get("alpha_s"), pC[2], 1e-10) and rel_ok(eC.get("tau_lz"), tC, 1e-10)
           and rel_ok(eC1.get("s_star"), pC[0], 1e-10) and rel_ok(eC1.get("gap_min2"), pC[1], 1e-10) and rel_ok(eC1.get("alpha_s"), pC[2], 1e-10))
    rows.append(("y4 C: level 2, no s > 1 call or sample, extension None; the entry parabola, diabatic, decided, s*, gap_min^2, alpha_s q's to 1e-10, tau_lz from them to 1e-10; the level-1 event the same", ok4,
                 "level %s; no extension %s; entry fit %s branch %s decided %s s* %s gap_min^2/A %s alpha_s %s tau_lz %s; level-1 event s* %s gap_min^2/A %s alpha_s %s; q's parabola (at %s) s* %s gap_min^2/A %s alpha_s %s tau_lz %s" % (
                     lC, no_ext_C, eC.get("fit_kind"), eC.get("branch"), eC.get("decided"), T1._f(eC.get("s_star"), "%.15g"), T1._f(None if gm2_entry(eC) is None else gm2_entry(eC) / A_SCALE, "%.15g"), T1._f(eC.get("alpha_s"), "%.15g"), T1._f(eC.get("tau_lz"), "%.15g"),
                     T1._f(eC1.get("s_star"), "%.15g"), T1._f(None if eC1.get("gap_min2") is None else eC1["gap_min2"] / A_SCALE, "%.15g"), T1._f(eC1.get("alpha_s"), "%.15g"),
                     tri_C, T1._f(pC[0], "%.15g"), T1._f(None if pC[1] is None else pC[1] / A_SCALE, "%.15g"), T1._f(pC[2], "%.15g"), T1._f(tC, "%.15g"))))
    # y5
    ratio_want = (pX[1] / pC[1]) if (pX[1] and pC[1]) else None
    ratio = (gm2_entry(eX) / gm2_entry(eC)) if (gm2_entry(eX) and gm2_entry(eC)) else None
    tratio = (eX["tau_lz"] / eC["tau_lz"]) if (eX.get("tau_lz") and eC.get("tau_lz")) else None
    eX1, eC1_ = ev_at(rX, 1) or {}, ev_at(rC, 1) or {}
    ok5 = (rel_ok(eX.get("alpha_s"), eC.get("alpha_s"), 1e-10) and eX.get("branch") == eC.get("branch") == "diabatic" and rel_ok(ratio, ratio_want, 1e-10)
           and rel_ok(eX1.get("gap_min2"), eC1_.get("gap_min2"), 1e-10) and rel_ok(eX1.get("A"), eC1_.get("A"), 1e-10))
    rows.append(("y5 C against X: at the reported level alpha_s equal, the branch equal (diabatic), X's gap_min^2 over C's the ratio of q's parabola minima to 1e-10; at level 1 the fitted minima and tau_LZ equal to 1e-10", ok5,
                 "alpha_s X %s C %s; branch X %s C %s; gap_min^2 ratio %s (want %s), tau_lz ratio %s; level 1: gap_min^2/A X %s C %s, tau_LZ X %s C %s" % (
                     T1._f(eX.get("alpha_s"), "%.15g"), T1._f(eC.get("alpha_s"), "%.15g"), eX.get("branch"), eC.get("branch"), T1._f(ratio, "%.15g"), T1._f(ratio_want, "%.15g"), T1._f(tratio, "%.15g"),
                     T1._f(None if eX1.get("gap_min2") is None else eX1["gap_min2"] / A_SCALE, "%.15g"), T1._f(None if eC1_.get("gap_min2") is None else eC1_["gap_min2"] / A_SCALE, "%.15g"), T1._f(eX1.get("A"), "%.15g"), T1._f(eC1_.get("A"), "%.15g"))))
    # y6
    lR = iR.get("level_stable"); hR = 2.0 ** -int(lR) if lR else None
    evR = ev_at(rR, lR) or {} if lR else {}
    sR = smp(eR); want_R = [1.0 - hR, 1.0] if hR else []
    sR_ok = len(sR) == 2 and all(abs(a - b) <= 1e-15 for (a, _d), b in zip(sR, want_R))
    gR = grid_q("R", want_R[0], want_R[1]) if hR else (None, None)
    ext = iR.get("extension")
    outR = (eR.get("branch"), eR.get("decided"), eR.get("flagged"), eR.get("fit_kind"), eR.get("s_star"), eR.get("gap_min"))
    outX = (eX.get("branch"), eX.get("decided"), eX.get("flagged"), eX.get("fit_kind"), eX.get("s_star"), eX.get("gap_min"))
    grid_flag = "two samples only (the crossing inside the chord's last piece): the grid bound"
    ok6 = (isinstance(ext, str) and "no fields to extend" in ext and not any(s > 1.0 for s in rR["calls"]) and not any_sample_above_1(rR, eR)
           and eR.get("fit_kind") == "grid" and eR.get("gap_min") == 0.0 and evR.get("gap_min2") == 0.0 and eR.get("exponent") == 0.0 and eR.get("branch") == "diabatic" and eR.get("decided") is True and eR.get("flagged") is False
           and has_flag(eR, "refused by the oracle") and has_flag(eR, grid_flag) and sR_ok and rel_ok(eR.get("s_star"), gR[0], 1e-10) and rel_ok(eR.get("alpha_s"), gR[1], 1e-10) and outR != outX)
    rows.append(("y6 R at its reported level l_R: the refusal text ('no fields to extend'), no s > 1 call or sample; the entry fit grid, gap_min 0, gap_min2 0, exponent 0, diabatic, decided, not flagged, the refusal and grid-bound flags, 2 samples at 1 - 2^-l_R and 1, s* and alpha_s the grid formula on sqrt(q) to 1e-10; the outcome differs from X's", ok6,
                 "l_R %s; extension %r; s > 1 calls %s; fit %s gap_min %s event gap_min2 %s exponent %s branch %s decided %s flagged %s; flags refused %s grid %s; samples %s (want s %s); s* %s alpha_s %s against the grid formula %s %s; tau_lz band %s; outcome R %s X %s" % (
                     lR, ext, [s for s in rR["calls"] if s > 1.0], eR.get("fit_kind"), eR.get("gap_min"), evR.get("gap_min2"), eR.get("exponent"), eR.get("branch"), eR.get("decided"), eR.get("flagged"),
                     has_flag(eR, "refused by the oracle"), has_flag(eR, grid_flag), sR, want_R, T1._f(eR.get("s_star"), "%.15g"), T1._f(eR.get("alpha_s"), "%.15g"), T1._f(gR[0], "%.15g"), T1._f(gR[1], "%.15g"), eR.get("tau_lz_band"), outR, outX)))
    # y7
    def at(e, s):
        return T1.sample_at(e or {}, s)
    eR2, eX2, eR1, eX1b = ev_at(rR, 2), ev_at(rX, 2), ev_at(rR, 1), ev_at(rX, 1)
    pairs = [(eR2, eX2, 0.75), (eR2, eX2, 1.0), (eR1, eX1b, 0.5), (eR1, eX1b, 1.0)]
    vals = [(s, at(a, s), at(b, s)) for a, b, s in pairs]
    ok7 = all(r_ is not None and x_ is not None and r_ == x_ for _s, r_, x_ in vals)
    rows.append(("y7 R against X, the in-step samples: R's level-2 samples at 0.75 and 1 and level-1 samples at 0.5 and 1 bitwise X's", ok7,
                 "; ".join("s %s: R %r X %r" % (s, r_, x_) for s, r_, x_ in vals)))
    # y8
    eR1 = eR1 or {}
    s8 = smp(eR1); w8 = [0.5, 1.0]
    s8_ok = len(s8) == 2 and all(abs(a - b) <= 1e-15 and rel_ok(d, delta_of("R", b), 1e-12) for (a, d), b in zip(s8, w8))
    g8 = grid_q("R", 0.5, 1.0)
    out8 = (eR1.get("decision"), eR1.get("decided"), eR1.get("resolved"), eR1.get("fit_kind"), eR1.get("s_star"), eR1.get("gap_min2"))
    out8X = (eX1.get("decision"), eX1.get("decided"), eX1.get("resolved"), eX1.get("fit_kind"), eX1.get("s_star"), eX1.get("gap_min2"))
    ok8 = (eR1.get("fit_kind") == "grid" and has_flag(eR1, "the chord's extension to s=1.5000 refused by the oracle") and has_flag(eR1, "two samples only") and s8_ok and eR1.get("gap_min2") == 0.0
           and rel_ok(eR1.get("s_star"), g8[0], 1e-10) and rel_ok(eR1.get("alpha_s"), g8[1], 1e-10) and eR1.get("decision") == "diabatic" and eR1.get("decided") is True and out8 != out8X)
    rows.append(("y8 R at level 1: fit grid, the refusal flag at s=1.5000 and 'two samples only', samples (0.5, sqrt q) and (1, sqrt q), gap_min2 0, s* and alpha_s the grid formula to 1e-10, decided diabatic, the outcome differs from X's level 1", ok8,
                 "fit %s; flags refused-1.5 %s two-samples %s; samples %s against sqrt q %s; gap_min2 %s; s* %s alpha_s %s against %s %s; decision %s decided %s; outcome R %s X %s" % (
                     eR1.get("fit_kind"), has_flag(eR1, "the chord's extension to s=1.5000 refused by the oracle"), has_flag(eR1, "two samples only"), s8, [delta_of("R", b) for b in w8], eR1.get("gap_min2"),
                     T1._f(eR1.get("s_star"), "%.15g"), T1._f(eR1.get("alpha_s"), "%.15g"), "%.15g" % g8[0], "%.15g" % g8[1], eR1.get("decision"), eR1.get("decided"), out8, out8X)))
    return rows


def run_late_step_checks(C, L, ST):
    """ST: the self-test module (ToyPath, decide, entries_of, idx_mask, lz_mode) so the toy is in x's class"""
    mode = ST.lz_mode(L)
    print("y late-step toy v2 (REVIEWA_C64_toy_readout_v1 sec. 2): A = %.4g, q mapped to the gap squared; tau %g; mode prim %s levels %s w_mix %s; switch X/R at s %.5f (w %.3g), C at 0.375" % (
        A_SCALE, TAU, mode["prim_module"], mode["prim_levels"], mode["prim_w_mix"], T1.SW_X, T1.SW_W), flush=True)
    R = {}
    for nm in ("X", "C", "R"):
        R[nm] = T1.run_variant(L, ST.ToyPath, ST.decide, ST.entries_of, ST.idx_mask, nm, mode)
        for line in T1.describe(R[nm]):
            print(line, flush=True)
    rule = L["v9c_rule"](mode)
    for name, ok, det in judge_y(R["X"], R["C"], R["R"], rule):
        C.ok(name, ok, det)
    return R


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default=None, help="a built engine (its module namespace is the layer); default: the builder's block")
    ap.add_argument("--selftest", default=os.path.join(HERE, "engine_twofield_v9c_selftest_v1z.py"))
    ap.add_argument("--prim", default=None, help="the primitive module name (default the layer's V9C_DEFAULTS)")
    a = ap.parse_args(argv)
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
    print("toy_late_step_v2: %s (%d of %d checks failed)%s" % ("FAIL" if failed else "PASS", len(failed), len(C.rows), (": " + "; ".join(failed)) if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
