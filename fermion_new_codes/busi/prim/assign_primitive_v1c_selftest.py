#!/usr/bin/env python
"""assign_primitive_v1c_selftest.py (reviewer A for the busi lane; PRIM_ruling_v1.md, PRIM_fix_v1b.md, PRIM_fix_v1c): the self-test of the assignment
primitive on a stand-in fixture with a known crossing sequence.

The fixture: ten diabatic levels d = (-5, -4.1, -3.2, -2.3, m0 + slope s, nm1, nm2, 2.4, 3.3, 4.6), the first five occupied at s = 0, the member
(index 4) rising through the empty levels nm1 and nm2 with couplings V1 and V2 (complex, phase 0.6 + 0.8i) and through nothing else, so that every
crossing is a known two-level crossing: exact when its coupling is zero, avoided with gap_min = 2 V otherwise, at s* = (nm - m0)/slope with the
relative slope alpha_s = slope. The mirror form blockdiag(H, -H*) has the engines' e <-> -e symmetry (index i <-> N - 1 - i), the occupied set the
ten negative levels (the Dirac sea), and the known sequence: the member meets its own mirror at e = 0 (a self-mirror crossing, exact), then each
crossing of the member appears with its mirror. Expected masks are the diabatic characters' occupations sorted by their energies (exact when the
couplings vanish; the character-following answer otherwise); where a crossing is taken adiabatically the member's and the partner's occupations
swap for s beyond it. The checks a to p are listed in PRIM_ruling_v1 section 3.1 and keep the expected values of the v1b self-test (PRIM_fix_v1b
section 4); e and i are expected as the ruling states (the wide avoided crossing is an adiabatic event with gap_min 0.4, alpha_s 3, s* 0.4 and
tau_LZ 2.667; the mirror fixture's wide pair is found, mirrored, adiabatic), e2 is e from level 3 (every piece's rotation below the window: the
box's probe regime), f2 is f at levels 5 to 6 (the narrow crossing stays diabatic once the grid resolves its window). PRIM_fix_v1c: the adiabatic
wide crossing followed by the member's character crossing the decoupled nm2 level in the same window is one adiabatic event, not a block (e, e2
pass); the two exact crossings 0.05 apart under overlap stay a block until the halving separates them (g, g2); the self-mirror crossing of the
mirror fixture under the energy rule is found in the first window (j stabilizes at level 2; the check asserts no level).

Usage: python assign_primitive_v1c_selftest.py [--all] [--torch] [--verbose]; exit code = the number of failures. Expected: 19 PASS, 0 FAIL (--torch: 21)."""
import argparse
import json
import math
import sys
import traceback

import numpy as np

import assign_primitive_v1c as P

QUERIES = tuple(P.GAUSS3) + (1.0,)
PHASE = complex(0.6, 0.8)
MEMBER, NM1, NM2 = 4, 5, 6
OCC = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0], bool)
BASE = {"m0": -1.0, "slope": 3.0, "nm1": 0.2, "nm2": 1.1, "V1": 0.0, "V2": 0.0}
TAU = 50.0


def params(**kw):
    p = dict(BASE); p.update(kw); return p


def diabats(s, p):
    return np.array([-5.0, -4.1, -3.2, -2.3, p["m0"] + p["slope"] * s, p["nm1"], p["nm2"], 2.4, 3.3, 4.6])


def h_plain(s, p):
    H = np.diag(diabats(s, p)).astype(complex)
    H[MEMBER, NM1] = p["V1"] * PHASE; H[NM1, MEMBER] = np.conj(H[MEMBER, NM1])
    H[MEMBER, NM2] = p["V2"] * PHASE; H[NM2, MEMBER] = np.conj(H[MEMBER, NM2])
    return H


def h_mirror(s, p):
    H = h_plain(s, p); N = H.shape[0]
    M = np.zeros((2 * N, 2 * N), complex); M[:N, :N] = H; M[N:, N:] = -H.conj()
    return M


def expected_mask(s, p, occ, mirror):
    """the diabatic characters' occupations sorted by their energies at s."""
    d = diabats(s, p)
    if mirror:
        energies = np.concatenate([d, -d]); flags = np.concatenate([occ, ~occ])
    else:
        energies, flags = d, occ
    return flags[np.argsort(energies, kind="stable")]


def swapped_occ():
    o = OCC.copy(); o[MEMBER] = False; o[NM1] = True; return o


def idx(m):
    return np.flatnonzero(np.asarray(m, bool)).tolist()


class NumpyOracle(object):
    def __init__(self, hfuns):
        self.hfuns = hfuns; self.n = 0

    def spectrum(self, s, channels):
        out = {}
        for ch in channels:
            e, U = np.linalg.eigh(self.hfuns[ch](s)); out[ch] = (e, U); self.n += 1
        return out

    def columns(self, U, cols):
        return U[:, list(cols)]

    def weights(self, A, B):
        return np.abs(A.conj().T @ B) ** 2

    def project(self, A, B):
        return self.weights(A, B).sum(axis=0)


class TorchOracle(object):
    def __init__(self, hfuns, device="cpu"):
        import torch
        self.torch = torch; self.hfuns = hfuns; self.device = device; self.n = 0

    def spectrum(self, s, channels):
        out = {}
        for ch in channels:
            H = self.torch.as_tensor(self.hfuns[ch](s), device=self.device)
            e, U = self.torch.linalg.eigh(H); out[ch] = (e.detach().cpu().numpy().astype(float), U); self.n += 1
        return out

    def columns(self, U, cols):
        return U[:, self.torch.as_tensor(list(cols), dtype=self.torch.int64, device=U.device)]

    def weights(self, A, B):
        return ((A.conj().transpose(0, 1) @ B).abs() ** 2).detach().cpu().numpy().astype(float)

    def project(self, A, B):
        return ((A.conj().transpose(0, 1) @ B).abs() ** 2).sum(0).detach().cpu().numpy().astype(float)


def run(p, rule, mirror=False, queries=QUERIES, tau=TAU, oracle_cls=NumpyOracle, **kw):
    hf = (lambda s: h_mirror(s, p)) if mirror else (lambda s: h_plain(s, p))
    orc = oracle_cls({"1": hf})
    m0 = {"1": expected_mask(0.0, p, OCC, mirror)}
    m0_copy = m0["1"].copy()
    dec = P.follow_chord(orc, m0, rule, queries=queries, tau=tau, weights_ch={"1": 1.0}, mirror=("auto" if mirror else None), **kw)
    return dec, orc, m0, m0_copy


def xs_of(dec):
    return sorted(dec.crossings, key=lambda ev: ev["s_star"])


def ev_text(ev):
    return "%d->%d%s %s %s s* %.6f gap_min %s alpha_s %s tau_LZ %s jump %s/%s%s" % (
        ev["i"], ev["n"], (" (mirrored)" if ev["mirrored"] else (" (self-mirror)" if ev["self_mirror"] else "")), ev.get("kind", "?"), ev["decision"], ev["s_star"],
        ("%.3e" % ev["gap_min"]) if ev["gap_min"] is not None else "-", ("%.4f" % ev["alpha_s"]) if ev["alpha_s"] is not None else "-",
        ("%.3g" % ev["A"]) if ev["A"] is not None else "-", ("%+.4f" % ev["jump_grid"]) if ev["jump_grid"] is not None else "-",
        ("%+.4f" % ev["jump_fit"]) if ev["jump_fit"] is not None else "-", (" " + "; ".join(ev["flags"])) if ev["flags"] else "")


class Acc(object):
    def __init__(self, verbose):
        self.fails = 0; self.verbose = verbose

    def __call__(self, name, ok, detail):
        print("%s %s: %s" % ("PASS" if ok else "FAIL", name, detail), flush=True)
        if not ok:
            self.fails += 1

    def guard(self, name, fn):
        try:
            fn()
        except Exception as ex:
            self.fails += 1
            print("FAIL %s: exception %s: %s" % (name, type(ex).__name__, ex), flush=True)
            if self.verbose:
                traceback.print_exc()


def check_a(acc):
    dec, orc, m0, m0c = run(params(slope=0.0), P.Rule("lz"))
    same = all(np.array_equal(dec.masks_at[s]["1"], m0c) for s in dec.masks_at)
    ok = dec.stable and dec.level_stable["1"] == 2 and not dec.crossings and dec.printable_set and same and dec.jump_fit_sum == 0.0 and dec.jump_grid_sum == 0.0
    acc("a. no crossing: the set unchanged bitwise at every point, stable at level 2, printable", ok,
        "stable %s at level %s, crossings %d, masks equal %s at %d points, spectra %d" % (dec.stable, dec.level_stable["1"], len(dec.crossings), same, len(dec.masks_at), orc.n))


def check_b(acc):
    p = params(nm2=3.5)
    dec, orc, m0, m0c = run(p, P.Rule("overlap"))
    xs = xs_of(dec); want = expected_mask(1.0, p, OCC, False)
    ok = (len(xs) == 1 and xs[0]["i"] == 4 and xs[0]["n"] == 5 and xs[0]["decision"] == "diabatic" and xs[0]["resolved"] and xs[0]["decided"]
          and abs(xs[0]["s_star"] - 0.4) < 1e-7 and xs[0]["gap_min"] < 1e-6 and abs(xs[0]["alpha_s"] - 3.0) < 1e-6 and np.array_equal(dec.masks_at[1.0]["1"], want)
          and dec.stable and dec.level_stable["1"] == 2 and abs(xs[0]["jump_grid"] + 0.3) < 1e-9 and abs(xs[0]["jump_fit"]) < 1e-6 and dec.jumps_fall and xs[0]["falls"] and xs[0]["jump_prev"] is not None)
    acc("b. exact crossing at s* 0.4, overlap: diabatic, 4 -> 5, final set {0,1,2,3,5}, stable at level 2, jump -0.3 at the grid falling to 0", ok,
        "%s; final %s (expected %s); stable %s at %s; jumps_fall %s" % ("; ".join(ev_text(ev) for ev in xs) or "no crossing", idx(dec.masks_at[1.0]["1"]), idx(want), dec.stable, dec.level_stable["1"], dec.jumps_fall))
    dec2, _o, _m, _c = run(p, P.Rule("overlap"), level_min=3, level_max=5)
    xs2 = xs_of(dec2)
    ok2 = (len(xs2) == 1 and dec2.stable and dec2.level_stable["1"] == 4 and abs(xs2[0]["jump_grid"] + 0.075) < 1e-9 and abs(xs2[0]["jump_prev"] + 0.075) < 1e-9 and xs2[0]["falls"] and abs(xs2[0]["s_star"] - 0.4) < 1e-7)
    acc("b2. the same from level 3: stable at level 4, the jump at the grid -0.075 (the point 0.375), falling from level 3's -0.075", ok2,
        "%s; stable %s at %s" % ("; ".join(ev_text(ev) for ev in xs2) or "no crossing", dec2.stable, dec2.level_stable["1"]))


def check_c(acc):
    p = params(nm2=3.5)
    dec, orc, m0, m0c = run(p, P.Rule("lz"))
    xs = xs_of(dec); want = expected_mask(1.0, p, OCC, False)
    ok = (len(xs) == 1 and xs[0]["decision"] == "diabatic" and xs[0]["decided"] and xs[0]["A"] is not None and xs[0]["A"] < 1e-6 and np.array_equal(dec.masks_at[1.0]["1"], want) and dec.printable_set)
    acc("c. exact crossing, lz: tau_LZ 0, diabatic, decided", ok, "%s; printable %s" % ("; ".join(ev_text(ev) for ev in xs) or "no crossing", dec.printable_set))


def check_d(acc):
    p = params(nm2=3.5)
    dec, orc, m0, m0c = run(p, P.Rule("energy"))
    xs = xs_of(dec)
    ok = (len(xs) == 1 and xs[0]["decision"] == "adiabatic" and all(np.array_equal(dec.masks_at[s]["1"], m0c) for s in dec.masks_at) and xs[0]["jump_grid"] == 0.0 and xs[0]["jump_fit"] == 0.0
          and abs(xs[0]["jump_alt"] + 0.3) < 1e-9 and dec.stable and dec.printable_set)
    acc("d. exact crossing, energy: adiabatic, the set unchanged, jump 0, the would-be jump -0.3", ok, "%s; final %s" % ("; ".join(ev_text(ev) for ev in xs) or "no crossing", idx(dec.masks_at[1.0]["1"])))


def check_e(acc):
    p = params(V1=0.2)
    dec, orc, m0, m0c = run(p, P.Rule("lz"))
    xs = xs_of(dec)
    ok = (len(xs) == 1 and xs[0]["decision"] == "adiabatic" and xs[0]["decided"] and abs(xs[0]["gap_min"] - 0.4) < 1e-6 and abs(xs[0]["alpha_s"] - 3.0) < 1e-6 and abs(xs[0]["A"] - 0.16 / 0.06) < 1e-4
          and abs(xs[0]["s_star"] - 0.4) < 1e-7 and np.array_equal(dec.masks_at[1.0]["1"], m0c) and np.array_equal(dec.masks_at[QUERIES[2]]["1"], m0c) and dec.stable and dec.printable_set)
    acc("e. avoided crossing gap 0.4 (tau 50, tau_LZ 2.667), lz: one adiabatic event recorded (gap_min 0.4, alpha_s 3, s* 0.4), the set unchanged, no event at nm2 afterwards", ok,
        "%s; final %s; stable %s at %s" % ("; ".join(ev_text(ev) for ev in xs) or "no crossing", idx(dec.masks_at[1.0]["1"]), dec.stable, dec.level_stable["1"]))


def check_e2(acc):
    p = params(V1=0.2)
    dec, orc, m0, m0c = run(p, P.Rule("lz"), level_min=3, level_max=4)
    xs = xs_of(dec)
    ok = (len(xs) == 1 and xs[0]["decision"] == "adiabatic" and xs[0]["decided"] and abs(xs[0]["gap_min"] - 0.4) < 1e-6 and abs(xs[0]["alpha_s"] - 3.0) < 1e-6 and abs(xs[0]["s_star"] - 0.4) < 1e-7
          and all(np.array_equal(dec.masks_at[s]["1"], m0c) for s in dec.masks_at) and dec.stable and dec.level_stable["1"] == 4 and dec.printable_set)
    acc("e2. the same from level 3 (every piece's rotation below the window, the box's probe): the event found by the accumulated character, adiabatic, stable at level 4", ok,
        "%s; final %s; stable %s at %s; notes %s" % ("; ".join(ev_text(ev) for ev in xs) or "no crossing", idx(dec.masks_at[1.0]["1"]), dec.stable, dec.level_stable["1"], dec.notes["1"]))


def check_f(acc):
    p = params(V1=0.02)
    dec, orc, m0, m0c = run(p, P.Rule("lz"))
    xs = xs_of(dec); want = expected_mask(1.0, p, OCC, False)
    ok = (len(xs) == 2 and all(ev["decision"] == "diabatic" and ev["decided"] for ev in xs) and abs(xs[0]["gap_min"] - 0.04) < 1e-6 and abs(xs[0]["s_star"] - 0.4) < 1e-7
          and abs(xs[0]["A"] - 0.0016 / 0.06) < 1e-5 and abs(xs[1]["s_star"] - 0.7) < 1e-3 and xs[1]["gap_min"] < 5e-3 and np.array_equal(dec.masks_at[1.0]["1"], want) and dec.stable and dec.jumps_fall)
    acc("f. avoided crossing gap 0.04 (tau_LZ 0.027) then the exact crossing at 0.7, lz: both diabatic, counted once each, final set {0,1,2,3,6}", ok,
        "%s; final %s (expected %s)" % ("; ".join(ev_text(ev) for ev in xs) or "no crossing", idx(dec.masks_at[1.0]["1"]), idx(want)))


def check_f2(acc):
    p = params(V1=0.02)
    dec, orc, m0, m0c = run(p, P.Rule("lz"), level_min=5, level_max=6)
    xs = xs_of(dec); want = expected_mask(1.0, p, OCC, False)
    jg = math.sqrt(0.0019515625)   # the pair's splitting at the sampled point 0.40625: sqrt((3 x 0.00625)^2 + 0.04^2)
    ok = (len(xs) == 2 and all(ev["decision"] == "diabatic" and ev["decided"] and ev["resolved"] for ev in xs) and abs(xs[0]["gap_min"] - 0.04) < 1e-6 and abs(xs[0]["s_star"] - 0.4) < 1e-7
          and abs(xs[0]["jump_grid"] + jg) < 1e-7 and xs[0]["falls"] and abs(xs[1]["s_star"] - 0.7) < 1e-3 and xs[1]["gap_min"] < 5e-3
          and np.array_equal(dec.masks_at[1.0]["1"], want) and dec.stable and dec.level_stable["1"] == 6 and dec.jumps_fall and dec.printable_set)
    acc("f2. the same at levels 5 to 6 (the grid resolves the narrow window): both crossings diabatic at both levels, stable at level 6, jump_grid -0.04418 at the point 0.40625 falling to -0.04, final set {0,1,2,3,6}", ok,
        "%s; final %s (expected %s); stable %s at %s; notes %s" % ("; ".join(ev_text(ev) for ev in xs) or "no crossing", idx(dec.masks_at[1.0]["1"]), idx(want), dec.stable, dec.level_stable["1"], dec.notes["1"]))


def check_g(acc):
    p = params(nm2=0.35)
    dec, orc, m0, m0c = run(p, P.Rule("overlap"), level_max=6)
    xs = xs_of(dec); want = expected_mask(1.0, p, OCC, False)
    ok = (dec.stable and dec.level_stable["1"] == 5 and len(xs) == 2 and abs(xs[0]["s_star"] - 0.4) < 1e-7 and abs(xs[1]["s_star"] - 0.45) < 1e-7 and np.array_equal(dec.masks_at[1.0]["1"], want) and dec.jumps_fall)
    acc("g. two crossings 0.05 apart (a block at levels 1 to 3): separated at level 4, stable at level 5, two crossings", ok,
        "stable %s at %s; %s; final %s" % (dec.stable, dec.level_stable["1"], "; ".join(ev_text(ev) for ev in xs) or "no crossing", idx(dec.masks_at[1.0]["1"])))
    dec3, _o, _m, _c = run(p, P.Rule("overlap"), level_max=3)
    xs3 = dec3.crossings
    ok3 = ((not dec3.stable) and dec3.level_stable["1"] is None and (not dec3.printable_set) and any(ev["block"] for ev in xs3) and any("block" in f for ev in xs3 for f in ev["flags"]))
    acc("g2. the same with level_max 3: unstable, not printable, the block flagged", ok3, "stable %s, printable %s; %s" % (dec3.stable, dec3.printable_set, "; ".join(ev_text(ev) for ev in xs3) or "no crossing"))


def check_h(acc, oracle_cls=NumpyOracle, label="h"):
    p = params(V1=0.02)
    dec, orc, m0, m0c = run(p, P.Rule("overlap"), mirror=True, oracle_cls=oracle_cls)
    xs = xs_of(dec)
    s_zero = (p["V1"] ** 2 / p["nm1"] - p["m0"]) / p["slope"]
    ok = (len(xs) == 3 and xs[0]["self_mirror"] and xs[0]["decision"] == "diabatic" and abs(xs[0]["s_star"] - s_zero) < 0.02
          and xs[1]["mirrored"] and abs(xs[1]["s_star"] - 0.4) < 1e-7 and abs(xs[1]["gap_min"] - 0.04) < 1e-6 and abs(xs[1]["jump_fit"] + 0.08) < 1e-6
          and xs[2]["mirrored"] and abs(xs[2]["s_star"] - 0.7) < 1e-3 and xs[2]["gap_min"] < 5e-3
          and np.array_equal(dec.masks_at[0.5]["1"], expected_mask(0.5, p, OCC, True)) and np.array_equal(dec.masks_at[1.0]["1"], expected_mask(1.0, p, OCC, True))
          and dec.mirror["1"] and dec.stable and dec.printable_set and dec.n_diabatic == 3)
    acc("%s. the mirror fixture, overlap: the known sequence (self-mirror at e = 0 near 1/3, then 0.4 and 0.7 each with its mirror merged, the merged jump twice the sector's), final set as expected" % label, ok,
        "mirror read %s; %s; at 0.5 %s; final %s; stable %s at %s" % (dec.mirror["1"], "; ".join(ev_text(ev) for ev in xs) or "no crossing", idx(dec.masks_at[0.5]["1"]), idx(dec.masks_at[1.0]["1"]), dec.stable, dec.level_stable["1"]))
    return dec


def check_i(acc):
    p = params(m0=-1.3, nm1=1.0, nm2=3.0, V1=0.2)
    dec, orc, m0, m0c = run(p, P.Rule("lz"), mirror=True, level_min=2, level_max=6)
    xs = xs_of(dec)
    s_zero = (p["V1"] ** 2 / p["nm1"] - p["m0"]) / p["slope"]; s_av = (p["nm1"] - p["m0"]) / p["slope"]
    ok = (len(xs) == 2 and xs[0]["self_mirror"] and xs[0]["decision"] == "diabatic" and abs(xs[0]["s_star"] - s_zero) < 0.01 and xs[0]["gap_converged"]
          and xs[1]["mirrored"] and xs[1]["decision"] == "adiabatic" and abs(xs[1]["s_star"] - s_av) < 1e-7 and abs(xs[1]["gap_min"] - 0.4) < 1e-6 and abs(xs[1]["A"] - 0.16 / 0.06) < 1e-4
          and np.array_equal(dec.masks_at[0.5]["1"], expected_mask(0.5, p, OCC, True)) and np.array_equal(dec.masks_at[1.0]["1"], expected_mask(1.0, p, swapped_occ(), True))
          and dec.stable and dec.printable_set)
    acc("i. the mirror fixture, lz: the self-mirror exact crossing diabatic (its fitted gap_min^2 converged against the threshold), the wide avoided pair found (its rotation incomplete at the chord's end), mirrored, adiabatic, the occupations swapped after it", ok,
        "%s; at 0.5 %s; final %s (expected %s); stable %s at %s; notes %s; flags %s" % ("; ".join(ev_text(ev) for ev in xs) or "no crossing", idx(dec.masks_at[0.5]["1"]), idx(dec.masks_at[1.0]["1"]),
                                                                                     idx(expected_mask(1.0, p, swapped_occ(), True)), dec.stable, dec.level_stable["1"], dec.notes["1"], dec.flags["1"]))


def check_j(acc):
    p = params(V1=0.02)
    dec, orc, m0, m0c = run(p, P.Rule("energy"), mirror=True)
    xs = xs_of(dec)
    ok = (len(xs) == 1 and xs[0]["self_mirror"] and xs[0]["decision"] == "adiabatic" and all(np.array_equal(dec.masks_at[s]["1"], m0c) for s in dec.masks_at) and dec.stable and dec.printable_set)
    acc("j. the mirror fixture, energy: one adiabatic crossing (the self-mirror), the set the sea at every point, no event afterwards", ok,
        "%s; final %s; stable %s at %s" % ("; ".join(ev_text(ev) for ev in xs) or "no crossing", idx(dec.masks_at[1.0]["1"]), dec.stable, dec.level_stable["1"]))


def check_k(acc):
    p = params(V1=0.02, nm2=3.5)
    dec, orc, m0, m0c = run(p, P.Rule("lz"), queries=(0.41, 1.0))
    xs = xs_of(dec); want = expected_mask(1.0, p, OCC, False)
    flagged = bool(dec.query_flags["1"].get(0.41))
    ok = (len(xs) == 1 and xs[0]["decision"] == "diabatic" and flagged and np.array_equal(dec.masks_at[0.41]["1"], want) and dec.stable and dec.printable_set)
    acc("k. a query point inside the crossing's window (s 0.41, weight 0.8): flagged, its set the decided one (swapped)", ok,
        "flagged %s; at 0.41 %s (expected %s); %s" % (flagged, idx(dec.masks_at[0.41]["1"]), idx(want), "; ".join(ev_text(ev) for ev in xs) or "no crossing"))


def check_l(acc):
    p = params(V1=0.02)
    d1, o1, m1, c1 = run(p, P.Rule("overlap"), mirror=True)
    d2, o2, m2, c2 = run(p, P.Rule("overlap"), mirror=True)
    j1 = json.dumps(d1.summary(), sort_keys=True); j2 = json.dumps(d2.summary(), sort_keys=True)
    masks_same = all(np.array_equal(d1.masks_at[s]["1"], d2.masks_at[s]["1"]) for s in d1.masks_at)
    input_kept = np.array_equal(m1["1"], c1) and np.array_equal(m2["1"], c2)
    copies = all(d1.masks_at[s]["1"] is not m1["1"] for s in d1.masks_at)
    ok = (j1 == j2) and masks_same and input_kept and copies
    acc("l. determinism and the input: two runs identical (summaries and masks), the start mask unmodified, the results copies", ok, "summaries equal %s, masks equal %s, input kept %s, copies %s" % (j1 == j2, masks_same, input_kept, copies))


def check_m(acc):
    r = P.Rule("lz", c=1.0)
    d1 = r.decide(0.06, 0.001, 0.06); d2 = r.decide(0.06, 0.0, 0.06); d3 = r.decide(0.03, 0.0, 0.06); d4 = r.decide(0.0, 0.0, 0.06); d5 = r.decide(0.06, 0.0, 0.0)
    d6 = P.Rule("overlap").decide(10.0, 0.0, 0.06); d7 = P.Rule("energy").decide(0.0, 0.0, 0.06)
    ok = ((not d1["decided"]) and d1["tie"] and d1["decision"] == "adiabatic" and d2["decided"] and d2["decision"] == "adiabatic" and d2["knife"] and d3["decided"] and d3["decision"] == "diabatic"
          and d4["decision"] == "diabatic" and d4["decided"] and d5["decision"] == "adiabatic" and d6["decision"] == "diabatic" and d6["decided"] and d7["decision"] == "adiabatic" and d7["decided"])
    acc("m. the rule's bands: undecided when the band contains c, knife-edge at c, diabatic below, exact crossing diabatic, no rate adiabatic, overlap and energy fixed", ok,
        "band [%.3g, %.3g] -> %s/%s; at c -> %s knife %s; below -> %s; exact -> %s; alpha 0 -> %s" % (d1["A_lo"], d1["A_hi"], d1["decision"], d1["decided"], d2["decision"], d2["knife"], d3["decision"], d4["decision"], d5["decision"]))


def check_n(acc):
    s_star, gm, al = 0.37, 0.04, 3.0
    samples = [(s, np.sqrt(al * al * (s - s_star) ** 2 + gm * gm)) for s in (0.125, 0.25, 0.375, 0.5, 0.625)]
    f = P.fit_pair(samples, eig_tol=1e-12)
    c = P.classify_pair(samples, P.Rule("lz"), tau=50.0, eig_tol=1e-12)
    g = P.fit_pair_grid([(0.875, 0.075), (0.9375, 0.1125)], P.Rule("lz"), tau=50.0)
    ok = (f["fit_ok"] and abs(f["s_star"] - s_star) < 1e-10 and abs(f["gap_min"] - gm) < 1e-10 and abs(f["alpha_s"] - al) < 1e-10 and f["misfit"] is not None and f["misfit"] < 1e-9
          and c["decision"] == "diabatic" and c["decided"] and abs(c["A"] - gm * gm * 50.0 / al) < 1e-9 and abs(c["alpha_T"] - al / 50.0) < 1e-12
          and g["fit_ok"] and g["fit_kind"] == "grid" and abs(g["s_star"] - 0.9) < 1e-12 and abs(g["alpha_s"] - 3.0) < 1e-12 and g["decision"] == "diabatic" and g["decided"] and abs(g["gap_min2_err"] - 0.075 ** 2) < 1e-15)
    acc("n. the fit's exactness on the two-level form: s*, gap_min, alpha_s to 1e-10, the other samples predicted, tau_LZ 0.0267 diabatic; the two-sample grid bound: s* 0.9, alpha_s 3, decided diabatic from the band [0, 0.075^2]", ok,
        "s* %s gap_min %s alpha_s %s misfit %s; %s tau_LZ %s; grid s* %s alpha %s %s/%s" % (f["s_star"], f["gap_min"], f["alpha_s"], f["misfit"], c["decision"], c["A"], g["s_star"], g["alpha_s"], g["decision"], g["decided"]))


def check_o(acc):
    ok = ((not P.jump_falls(-0.1, -0.2, 0.0, 1e-9, 1e-12)) and P.jump_falls(-0.2, -0.1, -0.05, 1e-9, 1e-12) and (not P.jump_falls(-0.1, -0.1, -0.2, 1e-9, 1e-12)) and P.jump_falls(-0.3, -0.3, 0.0, 1e-9, 1e-12))
    acc("o. the jumps-fall test's branches: a growing jump fails, an equal one holds, a resolved jump above the grid's fails", ok, "%s" % ok)


def check_p(acc):
    try:
        import torch  # noqa: F401
    except ImportError:
        acc("p. the torch oracle against numpy", True, "SKIP: torch not importable")
        return
    p = params(V1=0.02)
    dn, on, mn, cn = run(p, P.Rule("overlap"), mirror=True)
    dt, ot, mt, ct = run(p, P.Rule("overlap"), mirror=True, oracle_cls=TorchOracle)
    xn, xt = xs_of(dn), xs_of(dt)
    masks_same = all(np.array_equal(dn.masks_at[s]["1"], dt.masks_at[s]["1"]) for s in dn.masks_at)
    evs_same = len(xn) == len(xt) and all(a["decision"] == b["decision"] and a["mirrored"] == b["mirrored"] and a["self_mirror"] == b["self_mirror"] and abs(a["s_star"] - b["s_star"]) < 1e-8 for a, b in zip(xn, xt))
    ok = masks_same and evs_same and dn.level_stable["1"] == dt.level_stable["1"]
    acc("p. the torch oracle against numpy on the mirror fixture: the same masks, crossings and stable level", ok, "masks equal %s, crossings equal %s, levels %s/%s" % (masks_same, evs_same, dn.level_stable["1"], dt.level_stable["1"]))
    p2 = params(m0=-1.3, nm1=1.0, nm2=3.0, V1=0.2)
    dn2, _a, _b, _c = run(p2, P.Rule("lz"), mirror=True, level_min=2, level_max=6)
    dt2, _a, _b, _c = run(p2, P.Rule("lz"), mirror=True, level_min=2, level_max=6, oracle_cls=TorchOracle)
    xn2, xt2 = xs_of(dn2), xs_of(dt2)
    same2 = (len(xn2) == len(xt2) and all(a["decision"] == b["decision"] and a["kind"] == b["kind"] and abs(a["s_star"] - b["s_star"]) < 1e-8 for a, b in zip(xn2, xt2))
             and all(np.array_equal(dn2.masks_at[s]["1"], dt2.masks_at[s]["1"]) for s in dn2.masks_at) and dn2.level_stable["1"] == dt2.level_stable["1"])
    acc("p2. the torch oracle against numpy on i's fixture (the incomplete passage path)", same2, "crossings %d/%d, levels %s/%s" % (len(xn2), len(xt2), dn2.level_stable["1"], dt2.level_stable["1"]))


def main():
    ap = argparse.ArgumentParser(description="assign_primitive_v1c self-test on the stand-in fixture")
    ap.add_argument("--all", action="store_true", help="run every check (the default)")
    ap.add_argument("--torch", action="store_true", help="also run the torch oracle check (p)")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    acc = Acc(a.verbose)
    print("assign_primitive_v1c_selftest: %s on the stand-in fixture; queries %s; tau %g" % (P.VERSION, ", ".join("%.4f" % q for q in QUERIES), TAU), flush=True)
    for name, fn in (("a", check_a), ("b", check_b), ("c", check_c), ("d", check_d), ("e", check_e), ("e2", check_e2), ("f", check_f), ("f2", check_f2), ("g", check_g), ("h", check_h),
                     ("i", check_i), ("j", check_j), ("k", check_k), ("l", check_l), ("m", check_m), ("n", check_n), ("o", check_o)):
        acc.guard(name, lambda fn=fn: fn(acc))
    if a.torch:
        acc.guard("p", lambda: check_p(acc))
    print("selftest: %d failure(s)" % acc.fails, flush=True)
    return acc.fails


if __name__ == "__main__":
    sys.exit(main())
