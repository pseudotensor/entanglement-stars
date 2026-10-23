#!/usr/bin/env python
"""build_engine_twofield_v9c_selftest_v1z.py (busi lane builder, 20 Oct; REVIEWA_C64_toy_readout_v1.md sections 2 and 3): engine_twofield_v9c_selftest_v1z.py
over engine_twofield_v9c_selftest_v1y.py (md5 4e5cc754dfafe09b5e6c846b77c84445) in the lane's patch form: two exact replacements, each OLD occurring once
(the docstring head; the x block's two C.ok lines through the y call), every new line carrying "[v1z]" (blank lines excepted; v1y's first
docstring line, kept after the new head, carries the mark too); --reverse restores v1y byte for byte; the source md5 is checked; an output
is never overwritten. The x-X and x-R expected numbers are computed in the test from x's toy's own h(s); the y lines live in toy_late_step_v2.py.

Usage: python build_engine_twofield_v9c_selftest_v1z.py engine_twofield_v9c_selftest_v1y.py [engine_twofield_v9c_selftest_v1z.py] [--check]
       python build_engine_twofield_v9c_selftest_v1z.py engine_twofield_v9c_selftest_v1z.py restored.py --reverse [--against engine_twofield_v9c_selftest_v1y.py]
"""
import argparse
import hashlib
import os
import sys

MARK = "[v1z]"
SOURCE_MD5 = "4e5cc754dfafe09b5e6c846b77c84445"
OLD_DOC = "\"\"\"engine_twofield_v9c_selftest_v1y.py: engine_twofield_v9c_selftest.py a7db616f with one addition marked [v1y], the group y (the late-step\n"
NEW_DOC = "\"\"\"engine_twofield_v9c_selftest_v1z.py: engine_twofield_v9c_selftest_v1y.py 4e5cc754 with the changes of REVIEWA_C64_toy_readout_v1 sections 2 and 3, every changed line marked [v1z]:  # [v1z]\nthe review's two x lines replaced by the corrected x-X and x-R (read at the reported level; every expected number from x's toy's own h(s));  # [v1z]\nthe group y now toy_late_step_v2.py (the pass lines y1 to y8, all C.ok lines; v1's INFO duplicates and transcribed x predicates gone); nothing else touched.  # [v1z]\n\nengine_twofield_v9c_selftest_v1y.py (v1y's head follows; [v1z]): engine_twofield_v9c_selftest.py a7db616f with one addition marked [v1y], the group y (the late-step\n"
OLD_X = "    okx = (len(evx) == 2 and all(e_[\"branch\"] == \"diabatic\" and e_[\"decided\"] and not e_[\"flagged\"] for e_ in evx) and abs(evx[0][\"s_star\"] - 1.0 / 2.4) < 1e-6 and abs(evx[1][\"s_star\"] - 2.0 / 2.4) < 1e-6\n           and evx[1][\"fit_kind\"] == \"parabola\" and any(\"third sample at s=1.5000\" in f_ for f_ in evx[1][\"flags\"]) and abs(evx[1][\"gap_min\"]) < 1e-6 and np.array_equal(mx, MASKX) and ix[\"stable\"] and ix.get(\"extension\") is None and not hold(L, ix))\n    C.ok(\"x extension: two diabatic passages at s* 0.4167 and 0.8333, the second fitted through the third sample on the chord extended to s = 1.5 (the toy extends, so the verification is not consulted), the set at index 5, stable, no hold reason\", okx,\n         \"s* %s fit %s flags %s extension %s mask %s\" % ([e_.get(\"s_star\") for e_ in evx], [e_.get(\"fit_kind\") for e_ in evx], evx[1].get(\"flags\") if len(evx) > 1 else None, ix.get(\"extension\"), mx.astype(int).tolist()))\n    mxn, ixn, _ = decide(L, None, MASK, 2.0, lz_mode(L), path=ToyPathNoExt(hx))\n    evxn = entries_of(ixn)\n    okxn = (len(evxn) == 2 and all(e_[\"branch\"] == \"diabatic\" and e_[\"decided\"] and not e_[\"flagged\"] for e_ in evxn) and evxn[1][\"fit_kind\"] == \"grid\" and any(\"refused by the oracle\" in f_ for f_ in evxn[1][\"flags\"])\n            and evxn[1][\"exponent\"] == 0.0 and evxn[1][\"tau_lz_band\"][1] is not None and abs(evxn[1][\"tau_lz_band\"][1] - 0.16 / 1.2) < 1e-6 and isinstance(ixn.get(\"extension\"), str) and \"not in scope\" in ixn[\"extension\"]\n            and np.array_equal(mxn, MASKX) and ixn[\"stable\"] and not hold(L, ixn))\n    C.ok(\"x extension refused (the verification at s = 1 fails: no engine eigendecomposition in the layer's scope): the second passage on the two-sample bound, decided diabatic (exponent 0, the band's top 0.133), flagged as information, the refusal in the passage block, the set at index 5, stable, no hold reason\", okxn,\n         \"fit %s exponent %s band %s flags %s extension %s\" % ([e_.get(\"fit_kind\") for e_ in evxn], evxn[1].get(\"exponent\") if len(evxn) > 1 else None, evxn[1].get(\"tau_lz_band\") if len(evxn) > 1 else None, evxn[1].get(\"flags\") if len(evxn) > 1 else None, ixn.get(\"extension\")))\n\n    # y. (ReviewA_busi_open_q12_v1 sec. 1; [v1y]) the late-step two-sample toy beside x, in x's class: toy_late_step_v1.py (variants X, C, R)\n    import toy_late_step_v1 as Y  # [v1y]\n    Y.run_late_step_checks(C, L, sys.modules[__name__])  # [v1y]\n"
NEW_X = "    # x-X and x-R [v1z]: the review's two x lines corrected by REVIEWA_C64_toy_readout_v1 sec. 3 (read at the reported level l = info[\"level_stable\"]; every expected number from x's toy's own h(s) here)  # [v1z]\n    def _x_split(h, s, a_=3, b_=5):  # [v1z] the pair's splitting of x's h(s) at s by the test's own eig: the member's (basis 3) and level 5's (basis 5) characters\n        e_, U_ = np.linalg.eigh(h(float(s))); w_ = np.abs(U_) ** 2  # [v1z]\n        return float(abs(e_[int(np.argmax(w_[a_]))] - e_[int(np.argmax(w_[b_]))]))  # [v1z]\n    def _x_par(pts):  # [v1z] the parabola Delta^2 = alpha_s^2 (s - s*)^2 + gap_min^2 through 3 (s, Delta) points: (s*, gap_min^2 clamped at 0, alpha_s)\n        (s0_, d0_), (s1_, d1_), (s2_, d2_) = sorted(pts); y0_, y1_, y2_ = d0_ * d0_, d1_ * d1_, d2_ * d2_  # [v1z]\n        a_ = ((y2_ - y1_) / (s2_ - s1_) - (y1_ - y0_) / (s1_ - s0_)) / (s2_ - s0_); b_ = (y1_ - y0_) / (s1_ - s0_) - a_ * (s0_ + s1_); c_ = y0_ - a_ * s0_ * s0_ - b_ * s0_  # [v1z]\n        return -b_ / (2.0 * a_), max(c_ - b_ * b_ / (4.0 * a_), 0.0), math.sqrt(a_)  # [v1z]\n    def _x_rel(a_, b_, tol):  # [v1z] relative agreement (both exactly zero agrees)\n        return a_ is not None and b_ is not None and (a_ == b_ or abs(a_ - b_) <= tol * max(abs(b_), 1e-300))  # [v1z]\n    def _x_smp(e_):  # [v1z]\n        return sorted((float(p_[0]), float(p_[1])) for p_ in (e_.get(\"samples\") or []))  # [v1z]\n    lx = ix.get(\"level_stable\"); hl = (2.0 ** -int(lx)) if lx else None  # [v1z]\n    sx = [1.0 - hl, 1.0, 1.0 + hl] if hl else []; smx = _x_smp(evx[1]) if len(evx) > 1 else []; dx = [_x_split(hx, s_) for s_ in sx]  # [v1z]\n    smp_ok = len(smx) == 3 and all(abs(p_[0] - s_) <= 1e-15 and _x_rel(p_[1], d_, 1e-12) for p_, s_, d_ in zip(smx, sx, dx))  # [v1z]\n    px = _x_par(list(zip(sx, dx))) if len(dx) == 3 else (None, None, None)  # [v1z]\n    gmx = None if (len(evx) < 2 or evx[1].get(\"gap_min\") is None) else evx[1][\"gap_min\"] ** 2  # [v1z]\n    fit_ok = len(evx) > 1 and _x_rel(evx[1].get(\"s_star\"), px[0], 1e-10) and _x_rel(gmx, px[1], 1e-10) and _x_rel(evx[1].get(\"alpha_s\"), px[2], 1e-10) and abs(evx[1][\"s_star\"] - 5.0 / 6.0) < 0.01  # [v1z]\n    flag_x = (\"the pair's third sample at s=%.4f on the straight chord extended by one piece\" % (1.0 + hl)) if hl else None  # [v1z]\n    okx = (len(evx) == 2 and all(e_[\"branch\"] == \"diabatic\" and e_[\"decided\"] and not e_[\"flagged\"] for e_ in evx) and abs(evx[0][\"s_star\"] - 1.0 / 2.4) < 1e-6  # [v1z] the first passage's predicates untouched\n           and evx[1][\"fit_kind\"] == \"parabola\" and flag_x is not None and any(flag_x in f_ for f_ in evx[1][\"flags\"]) and smp_ok and fit_ok and np.array_equal(mx, MASKX) and ix[\"stable\"] and ix.get(\"extension\") is None and not hold(L, ix))  # [v1z]\n    C.ok(\"x-X extension (REVIEWA_C64_toy_readout_v1 sec. 3): two diabatic passages, the first at s* 0.4167; the second at the reported level l fitted through its 3 samples at 1 - 2^-l, 1, 1 + 2^-l (the third on the chord extended by one piece; the toy extends, so the verification is not consulted), each the pair's splitting of x's h(s) to 1e-12, s*, gap_min^2 and alpha_s the parabola through them to 1e-10, s* within 0.01 of 5/6; the set at index 5, stable, no hold reason\", okx,  # [v1z]\n         \"level %s; s* %s fit %s; samples %s against the toy's %s (ok %s); fit s* %s gap_min^2 %s alpha_s %s against the test's parabola %s (ok %s); flag %r present %s; extension %s mask %s\" % (lx, [e_.get(\"s_star\") for e_ in evx], [e_.get(\"fit_kind\") for e_ in evx], smx, list(zip(sx, dx)), smp_ok,  # [v1z]\n             evx[1].get(\"s_star\") if len(evx) > 1 else None, gmx, evx[1].get(\"alpha_s\") if len(evx) > 1 else None, px, fit_ok, flag_x, bool(len(evx) > 1 and flag_x and any(flag_x in f_ for f_ in evx[1][\"flags\"])), ix.get(\"extension\"), mx.astype(int).tolist()))  # [v1z]\n    mxn, ixn, _ = decide(L, None, MASK, 2.0, lz_mode(L), path=ToyPathNoExt(hx))\n    evxn = entries_of(ixn)\n    lr = ixn.get(\"level_stable\"); hr = (2.0 ** -int(lr)) if lr else None  # [v1z]\n    sr = [1.0 - hr, 1.0] if hr else []; smr = _x_smp(evxn[1]) if len(evxn) > 1 else []; dr = [_x_split(hx, s_) for s_ in sr]  # [v1z]\n    smr_ok = len(smr) == 2 and all(abs(p_[0] - s_) <= 1e-15 and _x_rel(p_[1], d_, 1e-12) for p_, s_, d_ in zip(smr, sr, dr))  # [v1z]\n    if len(dr) == 2:  # [v1z] the grid formula on the toy's own splittings: s* = (s_p D_c + s_c D_p)/(D_p + D_c), alpha_s = (D_p + D_c)/(s_c - s_p); the band top D_grid^2 = min(D_p, D_c)^2, on tau_LZ times k/(hbar alpha_T), alpha_T = alpha_s / tau\n        g_s = (sr[0] * dr[1] + sr[1] * dr[0]) / (dr[0] + dr[1]); g_a = (dr[0] + dr[1]) / (sr[1] - sr[0]); g_d2 = min(dr) ** 2; rule_x = L[\"v9c_rule\"](lz_mode(L))  # [v1z]\n        g_top = rule_x.k * g_d2 / (rule_x.hbar * (g_a / 2.0))  # [v1z] tau 2.0, x's\n    else:  # [v1z]\n        g_s = g_a = g_d2 = g_top = None  # [v1z]\n    e1n = evxn[1] if len(evxn) > 1 else {}  # [v1z]\n    grid_ok = bool(e1n) and _x_rel(e1n.get(\"s_star\"), g_s, 1e-10) and _x_rel(e1n.get(\"alpha_s\"), g_a, 1e-10) and _x_rel(e1n.get(\"gap_min2_err\"), g_d2, 1e-10) and (e1n.get(\"tau_lz_band\") or [None, None])[1] is not None and _x_rel(e1n[\"tau_lz_band\"][1], g_top, 1e-10)  # [v1z]\n    okxn = (len(evxn) == 2 and all(e_[\"branch\"] == \"diabatic\" and e_[\"decided\"] and not e_[\"flagged\"] for e_ in evxn) and evxn[1][\"fit_kind\"] == \"grid\" and any(\"refused by the oracle\" in f_ for f_ in evxn[1][\"flags\"])  # [v1z]\n            and any(\"two samples only (the crossing inside the chord's last piece): the grid bound\" in f_ for f_ in evxn[1][\"flags\"]) and evxn[1][\"exponent\"] == 0.0 and evxn[1].get(\"gap_min\") == 0.0 and smr_ok and grid_ok  # [v1z]\n            and isinstance(ixn.get(\"extension\"), str) and \"not in scope\" in ixn[\"extension\"] and np.array_equal(mxn, MASKX) and ixn[\"stable\"] and not hold(L, ixn))  # [v1z]\n    C.ok(\"x-R extension refused (REVIEWA_C64_toy_readout_v1 sec. 3; the verification at s = 1 fails: no engine eigendecomposition in the layer's scope): the second passage on the two-sample grid bound at the reported level l (fit grid, exponent 0, gap_min 0, 'refused by the oracle', 'two samples only'), its 2 samples at 1 - 2^-l and 1 the toy's splitting to 1e-12, s*, alpha_s and the band top (D_grid^2 and on tau_LZ) the grid formula on them to 1e-10, decided diabatic, not flagged, the refusal text in the passage block, the set at index 5, stable, no hold reason\", okxn,  # [v1z]\n         \"level %s; fit %s exponent %s gap_min %s; samples %s against the toy's %s (ok %s); s* %s alpha_s %s D_grid^2 %s band %s against the test's s* %s alpha_s %s D_grid^2 %s band top %s (ok %s); flags %s extension %s\" % (lr, [e_.get(\"fit_kind\") for e_ in evxn], e1n.get(\"exponent\"), e1n.get(\"gap_min\"), smr, list(zip(sr, dr)), smr_ok,  # [v1z]\n             e1n.get(\"s_star\"), e1n.get(\"alpha_s\"), e1n.get(\"gap_min2_err\"), e1n.get(\"tau_lz_band\"), g_s, g_a, g_d2, g_top, grid_ok, e1n.get(\"flags\"), ixn.get(\"extension\")))  # [v1z]\n\n    # y. (ReviewA_busi_open_q12_v1 sec. 1; [v1y]; REVIEWA_C64_toy_readout_v1 sec. 2 [v1z]) the late-step two-sample toy beside x, in x's class: toy_late_step_v2.py (variants X, C, R; pass lines y1 to y8)  # [v1z]\n    import toy_late_step_v2 as Y  # [v1z]\n    Y.run_late_step_checks(C, L, sys.modules[__name__])  # [v1y]\n"


class BuildStop(Exception):
    pass


def replacements():
    return [("docstring head", OLD_DOC, NEW_DOC), ("the x lines and the y call", OLD_X, NEW_X)]


def apply(text, reps, reverse=False):
    bad = [(n, text.count(nw if reverse else o)) for n, o, nw in reps if text.count(nw if reverse else o) != 1]
    if bad:
        raise BuildStop("; ".join("%s: %s occurs %d time(s), not once" % (n, "NEW" if reverse else "OLD", c) for n, c in bad))
    for n, o, nw in (reversed(reps) if reverse else reps):
        a, b = (nw, o) if reverse else (o, nw)
        text = text.replace(a, b, 1)
    return text


def marks_check(src, out):
    old = set(src.split("\n"))
    return [(q, l) for q, l in enumerate(out.split("\n"), 1) if l.strip() and l not in old and MARK not in l]


def md5(text):
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def read(p):
    with open(p, "rb") as f:
        return f.read().decode("utf-8")


def write_new(p, text):
    if os.path.exists(p):
        raise BuildStop("the output %s exists; it is never overwritten" % p)
    with open(p, "xb") as f:
        f.write(text.encode("utf-8"))


def main(argv=None):
    ap = argparse.ArgumentParser(description="engine_twofield_v9c_selftest_v1z over v1y (exact replacements; fails closed)")
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
            print("STOP: the source md5 is %s, not v1y's %s" % (md5(src), SOURCE_MD5)); return 2
        out_text = apply(src, reps)
    except BuildStop as stop:
        print("STOP: %s; nothing written" % stop); return 2
    marks = marks_check(src, out_text)
    if marks:
        print("STOP: %d new line(s) lack %r: %s" % (len(marks), MARK, marks[:5])); return 2
    if apply(out_text, reps, reverse=True) != src:
        print("STOP: the reverse does not restore v1y byte for byte"); return 2
    compile(out_text, "engine_twofield_v9c_selftest_v1z.py", "exec")
    print("build: %s (md5 %s) -> v1z (md5 %s); %d replacements; %d lines carry %r; marks ok; reverse byte for byte; compiled" % (
        a.source, md5(src), md5(out_text), len(reps), sum(1 for l in out_text.split("\n") if MARK in l), MARK))
    if a.check:
        print("CHECK PASS (nothing written)"); return 0
    out = a.out or os.path.join(os.path.dirname(os.path.abspath(a.source)), "engine_twofield_v9c_selftest_v1z.py")
    try:
        write_new(out, out_text)
    except BuildStop as stop:
        print("STOP: %s" % stop); return 2
    print("written: %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
