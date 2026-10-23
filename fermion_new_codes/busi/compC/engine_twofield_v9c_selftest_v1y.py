#!/usr/bin/env python
"""engine_twofield_v9c_selftest_v1y.py: engine_twofield_v9c_selftest.py a7db616f with one addition marked [v1y], the group y (the late-step
toy of ReviewA_busi_open_q12_v1 section 1, toy_late_step_v1.py) called at the end of the layer group, after x; x and every other line unchanged.

engine_twofield_v9c_selftest.py (reviewer A for the busi lane; C64_v9c_ruling_v1 2(c): G0''; C64_G4s8_hold_ruling_v1 (4b); C64_transport_ruling_v1 4.4;
rewritten against the built engine_twofield_v9b.py, md5 301e9dfc, after the first self-test, written without the source, did not apply).

Three groups. The layer (always): the block of build_engine_twofield_v9c.py executed on its own, on 8-level toys (h(s) linear in s, so the two-level fit is
exact); the toy routes in v9b's own signatures through the wrapper. The build (--v9b): the build, the marks, the reverse round trip, equality with --v9c, the
built engine's import (its defaults, the four names wrapped, V9C_MISSING empty) and its arguments' defaults. The fixture (--v9b --fixture [--gpu G]): the
engine's real routes on engine_twofield_v9b_selftest's manufactured paths (engine_twofield_v8_selftest's 128-site fixture), floor mode against the unwrapped
routes bitwise and lz mode against the ruling's expectations; then the v9b self-test's fixture passed by v9c in v9b's place (--v9b-selftest; a copy of the
code directory is made for it). Every check prints PASS or FAIL with its detail; the last line is "engine_twofield_v9c_selftest: PASS (n checks)" or FAIL
with the failing names (exit 1).

The layer's toys. heavy-like: eight levels 0 .. 7, the member (basis 3) rising from e_m0 at the rate slope per chord through the empty partner (basis 4,
energy 4), a coupling V between them (gap 2 V), the set the four lowest. mirror: eight levels symmetric under e <-> -e, the member -0.9 + 1.8 s meeting
its own mirror 0.9 - 1.8 s at e = 0 with a coupling V, the set the four negative levels (one spectrum, the mirror read off it: the self-mirror crossing
counted once). light-like: the heavy-like toy through the light's spectrum object (sectors all +1: the S = +1 sector is every level). The record's tau is
the chord's physical length (alpha_T = alpha_s / tau).
  a  the exact crossing, no mirror: one diabatic event (gap 0 to the fit), the set swapped, inversions 1, the jump zero and falling, no hold reason
  b  the exact crossing with the mirror: self-mirror, counted once, the set swapped
  c  tau 20: gap 0.1 diabatic (tau_LZ 0.11), gap 0.4 adiabatic (1.78, the set kept, jump_alt recorded, no hold reason), gap 0.3 the knife edge
     (tau_LZ 1: undecided, held); the light-like spectrum object: gap 0.1 diabatic, gap 0.4 adiabatic at tau 20, gap 0.4 diabatic at tau 3 (0.27)
  d  4.4's fixture (d) moved to the primitive: tau 2, gap 1e-3 and 1e-8 diabatic; at the HOLD ruling's rate (alpha_T 0.0456 at tau 2) gap 1e-3 is
     diabatic with the exponent 3.44e-5 and gap 0.3 is skipped at the defaults with the previous reference (no spectrum), quiet without it (two
     spectra: the set's columns at 0 and the point 1/2) and quiet with the skip off (--lz-skip-gap 0; one spectrum)
  e  the inversion removed on a second record (the dived mode rising back: diabatic, delta -1, inversions 0)
  f  a block (four passages of one member in one chord): unresolved at level 2 and held; resolved at level 4 (four diabatic events, stable)
  g  quiet: no passage, one spectrum, the walk's mask object returned; the previous reference given: the same, still one spectrum
  h  skip: wide quiet boundary pairs and the previous reference clean at s = 1, no spectrum; no skip on an exact exchange of two levels (the mirror
     toy: the sorted spectrum's ends alike, the reference's character off its start indices at s = 1): the passage found
  i  a passage straddling two records: the first chord an approach (an entry without a decision, so wmin below 0.9 does not hold), the second the
     incomplete passage decided diabatic and the set swapped
  j  the G4s8 record form through the wrapper on the toy light route in v9b's signature: the walk's kept set replaced by the decided set, the route
     re-called once at x_n (V9cEndPath), anchors 3, the entry diabatic with its exponent and the primitive's version; the same on the toy heavy route
  k  floor mode through the wrapper: the route's return returned untouched (the same object), one call, nothing computed; the light caller's stash
     untouched in floor mode and set in lz mode
  l  the walk's own crossing merged with the event (no re-call, source walk+primitive, no hold); a bogus walk swap flagged, inconsistent, holding
  m  the arguments: the defaults equal V9C_DEFAULTS and leave V9C_MODE untouched; v9c_mode_set reads lz and lz_c 2 off a namespace; lz refuses the
     record transport and prim_levels 1
  n  tau_LZ halves from tau 2 to tau 1 on the same chord
  o  the torch spectrum object (when torch is present) agrees with numpy on decisions and masks
  p  the exponent convention: exponent = pi tau_LZ / (2 k); with k = pi/2 the two coincide
  q  determinism: two runs give identical entries
  r  v9c_hold_reasons on G4s8's 13.125 form (crossings empty, wmin 0.0027) names the unresolved passage; a walk overlap 0.8365 (block 2) holds by 4.6
  x  (C64_v9c_review_v1 finding 9) the chord's extension by one piece: two passages of one member in one chord, the second in the last piece after the
     first moved the anchor, so the anchor's side gives no sample: the third sample at s = 1.5 and the exact fit; with the extension refused (the
     verification at s = 1 fails: no engine eigendecomposition in the layer's scope) the two-sample bound, decided diabatic, flagged as information,
     the refusal recorded in the passage block
The fixture (F; the real engine, the real routes):
  F1 floor identity: the wrapped light route on (c)'s path returns what the unwrapped route returns (every number), no passage block
  F2 (c) the exact crossing (at s* 0.45, off the primitive's dyadic points; see pair_path) in lz: the walk's one crossing merged with the
     primitive's event (diabatic, exponent 0), no re-call, the mask (c)'s
  F3 (d 1e-3) in lz on the v9b fixture's own path, the G4s8 record form: the walk kept the set (crossings 0, the selected level's weight on the
     previous reference 1.2e-4, crossing_levels 4); the primitive decides diabatic with the exponent of that path's own geometry (its coupling grows
     along the chord, h0 + s D: the minimum gap c g / sqrt(4 c^2 + g^2), about g/2 = 5.0e-4, at s* 0.49994, alpha_s sqrt(4 c^2 + g^2): the exponent
     8.6e-6, derived), the route re-called at x_n, anchors 3, inversions 1, the final mask (c)'s, the cut check on it crossing_levels 0 with min weight
     above 0.99, no hold reason; the same with the previous reference stashed (the trigger's spectrum spared; the primitive needs the eigenbasis at 0)
  F3c (d 1e-3) with the constant coupling (the coupling in h0: the Landau-Zener form, gap g at s* 1/2, alpha_T 0.0456): the HOLD ruling's exponent
     3.44e-5, diabatic, re-called, min weight above 0.99
  F4 (d 1e-8) in lz: the walk's sub-floor crossing merged, diabatic, no re-call, the mask (c)'s
  F5 the adiabatic cases on the constant coupling: gap 0.3 at the fixture's rate skipped at the defaults with the previous reference stashed and quiet
     without it (the window wider than the chord, the set kept, no entry); gap 1e-3 at a sweep slower by 1e5 (tau 2e5): resolved adiabatic (tau_LZ 2.19),
     the set kept, the walk consistent, no re-call; at the knife edge (tau 9.12e4): flagged knife-edge or undecided and held
  F6 tau_LZ halves from tau 2 to tau 1 on (d 1e-3)
  F7 heavy (c) the exact crossing (at s* 0.501: off the dyadic points, inside the room the fixture's neighbouring levels leave) in lz: the walk's
     crossing merged, diabatic, no re-call, the occupation swapped; heavy (b) no crossing: no spectrum
  F8 the heavy avoided crossing of the constant gap 1e-3 (the walk kept the set by the floor): decided diabatic (exponent pi g^2 / (2 alpha_T)),
     re-called at x_n, anchors 3, inversions 1, the reference assembled by assign_occupation_v34 with min_occupied_weight above 0.99
  F9 determinism on F3
  F10 the callers: light_continue_v36 and continue_ref_v9 on the fixture's own window and a small field change, lz against floor: the reference and the
     mask bitwise, the entries equal but for the passage block (the light's, skipped or quiet; the heavy on the Weyl route has none)
  F11 (C64_v9c_review_v1 findings 2 and 9) the re-call on V9cEndPath over the real LightPathV9 through light_continue_v36 (v9c_decide forced to return
     the walk's own set as a change): the reference, the mask and the entry bitwise against floor but for the passage block, which carries the re-call;
     then the extension's field convention on that path: V9cSpectra._extend(1) reproduces the trial's eigenvalues and _extend(1/2) path.eig(1/2)'s
  u  the v9b self-test's fixture passes with v9c in v9b's place (floor mode; bitwise by its compared numbers)

Usage: python engine_twofield_v9c_selftest.py [--v9b engine_twofield_v9b.py] [--v9c engine_twofield_v9c.py] [--fixture] [--gpu G]
          [--v9b-selftest engine_twofield_v9b_selftest.py] [--skip-suite] [--suite-timeout SECONDS] [--skip-torch] [--code-dir DIR]
"""
import argparse
import functools
import importlib.util
import inspect
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace

import numpy as np

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
for _d in (os.environ.get("BUSI_ENGINE_DIR", "<internal>/engine"), os.environ.get("BUSI_PAGE_DIR", "<internal>/page")):
    if _d and os.path.isdir(_d) and _d not in sys.path:
        sys.path.append(_d)
import build_engine_twofield_v9c as B  # noqa: E402


class Checks(object):
    def __init__(self):
        self.rows = []

    def ok(self, name, cond, detail=""):
        cond = bool(cond)
        self.rows.append((name, cond, detail))
        print("%s %s%s" % ("PASS" if cond else "FAIL", name, (": " + str(detail)) if detail else ""), flush=True)
        return cond

    def failed(self):
        return [n for n, c, d in self.rows if not c]


# ------------------------------------------------------------------------------------------------------- the toys
def heavy_h(V, slope=1.8, e_m0=3.0, member=3, partner=4, N=8):
    def h(s):
        E = np.arange(N, dtype=float); E[member] = e_m0 + slope * s
        H = np.diag(E); H[member, partner] = V; H[partner, member] = V
        return H
    return h


def mirror_h(V, slope=1.8):
    def h(s):
        E = np.array([-4.0, -3.0, -2.0, -0.9 + slope * s, 0.9 - slope * s, 2.0, 3.0, 4.0])
        H = np.diag(E); H[3, 4] = V; H[4, 3] = V
        return H
    return h


class ToyPath(object):
    """h(s) dense numpy: eig(s) -> (e, U) (heavy-like) or (e, sig, U) with sig all +1 (light-like); extends beyond [0, 1]"""
    v9c_extends = True

    def __init__(self, h, light=False, torch_mod=None):
        self.h = h; self.light = light; self.eigh_n = 0; self.torch = torch_mod; self.trial = None

    def eig(self, s):
        self.eigh_n += 1
        H = self.h(float(s))
        if self.torch is not None:
            e, U = self.torch.linalg.eigh(self.torch.as_tensor(H, dtype=self.torch.float64)); e = e.numpy().astype(float)
        else:
            e, U = np.linalg.eigh(H)
        if self.light:
            return e, np.ones(e.size), U
        return e, U

    def h_host(self, s):
        return self.h(float(s))

    def eigvals(self, s):
        return np.linalg.eigvalsh(self.h(float(s)))


class ToyPathNoExt(ToyPath):
    """(C64_v9c_review_v1 finding 9) the toy without the extension protocol: v9c_extends False and dummy fields, so V9cSpectra._extend is reached and the verification at s = 1 fails in the layer (no engine eigendecomposition in scope): the extension's refusal path"""
    v9c_extends = False

    def __init__(self, h, light=False, torch_mod=None):
        ToyPath.__init__(self, h, light=light, torch_mod=torch_mod); self.f0 = (np.zeros(1), np.zeros(1)); self.f1 = (np.zeros(1), np.zeros(1)); self.ch = object()


def spec_of(L, path, light=False):
    if light:
        e0, s0, U0 = path.eig(0.0); trial = path.eig(1.0); path.trial = trial
        return L["V9cSpectra"](path, True, e0, s0, trial)
    e0, U0 = path.eig(0.0); trial = path.eig(1.0); path.trial = trial
    return L["V9cSpectra"](path, False, e0, None, trial)


def lz_mode(L, **over):
    M = dict(L["V9C_DEFAULTS"]); M["anchor_passage"] = "lz"; M.update(over)
    return M


def decide(L, h, mask0, tau, mode, mask_walk=None, crossings=None, light=False, path=None, ref0=None):
    p = path if path is not None else ToyPath(h, light=light)
    spec = spec_of(L, p, light=light)
    mw = mask0.copy() if mask_walk is None else mask_walk
    m, info = L["v9c_decide"]("light" if light else "heavy", 1, spec, mask0, mw, crossings or [], tau, mode=mode, ref0=ref0, say=False)
    return m, info, spec


def entries_of(info, kinds=("passage", "incomplete", "block")):
    return [c for c in info["entries"] if c.get("kind") in kinds]


def idx_mask(N, occ):
    m = np.zeros(N, bool); m[list(occ)] = True
    return m


def hold(L, info, wmin=0.999):
    return L["v9c_hold_reasons"]({"crossings": info["entries"], "min_occupied_weight": wmin, "passage": {k: v for k, v in info.items() if k != "entries"}})


def light_select_toy(e_np, sig, anchors):
    """light_select_sector_v9's rule, for the layer's toys (the engine's own is in scope in the built engine)"""
    sgn = np.where(np.asarray(sig, float) > 0.0, 1.0, -1.0); e_np = np.asarray(e_np, float)
    n_above = np.zeros(len(e_np), dtype=np.int64); gap = None
    for a_ in anchors:
        kv = e_np - a_ * sgn
        n_above += (kv >= 0.0).astype(np.int64)
        g_ = float(np.min(np.abs(kv))); gap = g_ if gap is None else min(gap, g_)
    return (n_above % 2 == 0), gap


def boundaries(m):
    m = np.asarray(m, bool); return np.flatnonzero(m[:-1] != m[1:])


def inversions(m):
    m = np.asarray(m, bool)
    if not m.any() or m.all():
        return 0
    return int(np.sum(m[int(np.argmax(~m)):]))


def toy_light_walk(path, S_host, e0_np, sig0, mask0, trial, n_sea, d_min, max_pieces, tol_block, mu_prev=None, step_scale=1.0, max_crossings=128, crossings=(), occp_end=None, calls=None):
    """v9b's light route in miniature, its signature and return: the + occupation kept (or the given end occupation: a sub-floor swap), the entry with the walk's counters"""
    if calls is not None:
        calls.append((type(path).__name__, mu_prev))
    sel0 = np.flatnonzero(np.asarray(sig0, float) > 0.0); order0 = sel0[np.argsort(np.asarray(e0_np, float)[sel0], kind="stable")]
    e1, sig1, U1 = trial; sel1 = np.flatnonzero(np.asarray(sig1, float) > 0.0); order1 = sel1[np.argsort(np.asarray(e1, float)[sel1], kind="stable")]; ep1 = np.asarray(e1, float)[order1]
    occp0 = np.asarray(mask0, bool)[order0].copy(); occp = occp0.copy() if occp_end is None else np.asarray(occp_end, bool).copy()
    bnds = boundaries(occp); anchors = [float(0.5 * (ep1[j] + ep1[j + 1])) for j in bnds]
    cr = [dict(c) for c in crossings]
    cert = {"norm_dK_G": 1.0, "mu0": anchors[0] if anchors else None, "mu1": anchors[0] if anchors else None, "gap_start": 1.0, "pieces": 3, "cholesky_calls": 9, "d_min_certified": 1e-7, "certified": True, "why": None,
            "smallest_piece": 0.3, "piece_list": None, "seconds": 0.0, "norm_K_G": 8.0, "rounding_F": 1e-28, "d_floor": 2e-14, "d_min_effective": 3.5e-7}
    sector = {"route": ("sector walk" if (len(bnds) > 1 or cr) else "mu S walk"), "pieces": 3, "cholesky": 9, "eigh": 0, "segments": 1, "probes_in_crossing": 0, "anchors_start": int(len(boundaries(occp0))), "anchors_end": int(len(bnds)),
              "anchors": anchors, "certified_fraction": 1.0, "crossings": cr, "flagged": int(sum(1 for c in cr if c.get("flagged"))), "inversions_plus_start": inversions(occp0), "inversions_plus_end": inversions(occp), "adiabatic_at_cut": 0,
              "fallback_why": None, "swaps": int(sum(1 for c in cr if c.get("swap")))}
    return cert, sector, anchors, occp


def toy_heavy_walk(path, prev_ref, e0_np, mask0, trial, n_sea, d_min, max_pieces, tol_block, step_scale=1.0, max_crossings=128, crossings=(), occ_end=None, calls=None):
    if calls is not None:
        calls.append((type(path).__name__,))
    occ0 = np.asarray(mask0, bool).copy(); occ = occ0.copy() if occ_end is None else np.asarray(occ_end, bool).copy()
    cr = [dict(c) for c in crossings]
    walk = {"route": "sector walk", "pieces": 3, "eigvals": 3, "eigh": 0, "probes_in_crossing": 0, "anchors_start": int(len(boundaries(occ0))), "anchors_end": int(len(boundaries(occ))), "crossings": cr, "flagged": 0, "certified_fraction": 1.0,
            "norm_dH_G": 1.0, "norm_H_G": 8.0, "d_min_effective": 3.5e-7, "gap_start": 1.0, "margin_min": 0.5, "smallest_piece": 0.3, "certified": True, "why": None, "seconds": 0.0, "inversions_start": inversions(occ0), "inversions_end": inversions(occ),
            "fallback_why": None, "swaps": int(sum(1 for c in cr if c.get("swap")))}
    return walk, occ


def walk_entry(s, swap=True, overlap=0.99995, delta=1, flagged=False, why=None):
    return {"s": s, "piece": [s - 3e-4, s + 3e-4], "s_reopen": s + 3e-4, "probes": 3, "anchor_energy": 3.5, "pair": [3.4, 3.6], "pair_start": [3.0, 4.0], "pair_gap_end": 0.2, "overlap": overlap, "inversion_delta": delta,
            "swap": swap, "block": 2, "flagged": flagged, "flagged_why": why, "intermediate": 0, "occupied_before": [0], "occupied_after": [1]}


# ------------------------------------------------------------------------------------------------------- the layer
def run_layer_checks(C, L, skip_torch=False):
    L["light_select_sector_v9"] = light_select_toy
    MASK = idx_mask(8, [0, 1, 2, 3]); SWAP = idx_mask(8, [0, 1, 2, 4])
    # a. the exact crossing, no mirror
    m_out, info, _ = decide(L, heavy_h(0.0), MASK, 2.0, lz_mode(L))
    ev = entries_of(info)
    C.ok("a exact crossing: one diabatic event", info["ran"] and len(ev) == 1 and ev[0]["branch"] == "diabatic" and ev[0]["decided"], "%d events, %s" % (len(ev), ev[0]["branch"] if ev else "-"))
    C.ok("a exact crossing: the fit", ev and abs(ev[0]["s_star"] - 1.0 / 1.8) < 1e-6 and ev[0]["gap_min"] < 1e-6 and abs(ev[0]["alpha_s"] - 1.8) < 1e-6, "s* %s gap_min %s alpha_s %s" % (ev[0]["s_star"] if ev else None, ev[0]["gap_min"] if ev else None, ev[0]["alpha_s"] if ev else None))
    C.ok("a exact crossing: the set swapped, inversions 1, stable, jumps fall, no hold reason", np.array_equal(m_out, SWAP) and info["inversions_end"] == 1 and info["stable"] and info["jumps_fall"] and ev and ev[0]["inversion_delta"] == 1 and ev[0]["swap"] and abs(ev[0]["jump_fit"]) < 1e-9 and ev[0]["falls"] is not False and not hold(L, info),
         "mask %s inv %s stable %s falls %s hold %s" % (m_out.astype(int).tolist(), info["inversions_end"], info["stable"], info["jumps_fall"], hold(L, info)))
    # b. the exact crossing with the mirror
    m_out, info, _ = decide(L, mirror_h(0.0), MASK, 2.0, lz_mode(L))
    ev = entries_of(info)
    C.ok("b mirror exact crossing: self-mirror, counted once, diabatic, the set swapped", len(ev) == 1 and ev[0]["self_mirror"] and ev[0]["branch"] == "diabatic" and np.array_equal(m_out, SWAP) and abs(ev[0]["s_star"] - 0.5) < 1e-6 and info["mirror"],
         "%d events self_mirror %s s* %s mirror %s" % (len(ev), ev[0]["self_mirror"] if ev else None, ev[0]["s_star"] if ev else None, info.get("mirror")))
    # c. the three branches at tau 20; the light-like spectrum object
    for label, h, tau, expect, A_expect, light in (("heavy gap 0.1", heavy_h(0.05), 20.0, "diabatic", 0.1111, False), ("heavy gap 0.4", heavy_h(0.2), 20.0, "adiabatic", 1.7778, False),
                                                   ("light-like gap 0.1", heavy_h(0.05), 20.0, "diabatic", 0.1111, True), ("light-like gap 0.4", heavy_h(0.2), 20.0, "adiabatic", 1.7778, True), ("light-like gap 0.4 at tau 3", heavy_h(0.2), 3.0, "diabatic", 0.2667, True)):
        m_out, info, _ = decide(L, h, MASK, tau, lz_mode(L), light=light)
        ev = entries_of(info); got = ev[0] if ev else {}
        want_mask = SWAP if expect == "diabatic" else MASK
        C.ok("c %s: %s" % (label, expect), len(ev) == 1 and got.get("branch") == expect and got.get("decided") and not got.get("flagged") and np.array_equal(m_out, want_mask) and abs(got.get("tau_lz", 0.0) - A_expect) < 2e-3 and info["stable"] and not hold(L, info),
             "branch %s tau_LZ %s mask %s stable %s hold %s" % (got.get("branch"), got.get("tau_lz"), m_out.astype(int).tolist(), info["stable"], hold(L, info)))
        if expect == "adiabatic":
            C.ok("c %s: jump_alt recorded, jump 0, overlap None, the character weight recorded" % label, got.get("jump_alt") is not None and abs(got.get("jump_alt")) > 0.1 and abs(got.get("jump_grid") or 0.0) < 1e-12 and got.get("overlap") is None and got.get("character_weight") is not None, "jump_alt %s character %s" % (got.get("jump_alt"), got.get("character_weight")))
    m_out, info, _ = decide(L, heavy_h(0.15), MASK, 20.0, lz_mode(L))
    ev = entries_of(info)
    C.ok("c heavy gap 0.3: the knife edge undecided and held", len(ev) == 1 and ev[0]["tie"] and not ev[0]["decided"] and ev[0]["flagged"] and info["unresolved"] and bool(hold(L, info, 0.5)),
         "tau_LZ %s tie %s decided %s" % (ev[0]["tau_lz"] if ev else None, ev[0]["tie"] if ev else None, ev[0]["decided"] if ev else None))
    # d. 4.4's fixture (d) moved to the primitive
    for g, label in ((1e-3, "1e-3"), (1e-8, "1e-8")):
        m_out, info, _ = decide(L, heavy_h(0.5 * g), MASK, 2.0, lz_mode(L))
        ev = entries_of(info)
        C.ok("d gap %s at tau 2: diabatic, decided, unflagged, the set swapped" % label, len(ev) == 1 and ev[0]["branch"] == "diabatic" and ev[0]["decided"] and not ev[0]["flagged"] and np.array_equal(m_out, SWAP), "branch %s tau_LZ %s flags %s" % (ev[0]["branch"] if ev else None, ev[0]["tau_lz"] if ev else None, ev[0]["flags"] if ev else None))
    m_out, info, _ = decide(L, heavy_h(5e-4, slope=0.0912, e_m0=3.955), MASK, 2.0, lz_mode(L))
    ev = entries_of(info)
    C.ok("d gap 1e-3 at alpha_T 0.0456: diabatic with the exponent 3.44e-5", len(ev) == 1 and ev[0]["branch"] == "diabatic" and abs(ev[0]["exponent"] / 3.4443e-5 - 1.0) < 1e-2 and abs(ev[0]["gap_min"] - 1e-3) < 1e-6 and abs(ev[0]["alpha_T"] - 0.0456) < 1e-4, "exponent %s gap_min %s alpha_T %s" % (ev[0]["exponent"] if ev else None, ev[0]["gap_min"] if ev else None, ev[0]["alpha_T"] if ev else None))
    pq = ToyPath(heavy_h(0.15, slope=0.0912, e_m0=3.955)); e0q, U0q = pq.eig(0.0)
    m_out, info, spec = decide(L, None, MASK, 2.0, lz_mode(L), path=pq, ref0=U0q[:, :4])
    C.ok("d gap 0.3 at alpha_T 0.0456 with the previous reference: skipped at the defaults (no spectrum), the set kept", info["skipped"] and not info["ran"] and np.array_equal(m_out, MASK) and info["n_spectra"] == 0 and spec.n_eigh == 0, "skipped %s spectra %d" % (info["skipped"], info["n_spectra"]))
    m_out, info, spec = decide(L, heavy_h(0.15, slope=0.0912, e_m0=3.955), MASK, 2.0, lz_mode(L))
    C.ok("d gap 0.3 at alpha_T 0.0456 without the previous reference: no skip, quiet (the window wider than the chord), the set's columns at 0 and the point 1/2, no entry", info["quiet"] and not info["skipped"] and not info["ran"] and np.array_equal(m_out, MASK) and info["n_spectra"] == 2 and not info["entries"], "quiet %s spectra %d" % (info["quiet"], info["n_spectra"]))
    m_out, info, spec = decide(L, None, MASK, 2.0, lz_mode(L, lz_skip_gap=0.0), path=pq, ref0=U0q[:, :4])
    C.ok("d gap 0.3 at alpha_T 0.0456, the skip off (--lz-skip-gap 0) with the previous reference: quiet, one spectrum (the point 1/2), no entry", info["quiet"] and not info["skipped"] and not info["ran"] and np.array_equal(m_out, MASK) and info["n_spectra"] == 1 and not info["entries"], "quiet %s spectra %d" % (info["quiet"], info["n_spectra"]))
    # e. the inversion removed
    m_out, info, _ = decide(L, heavy_h(0.0, slope=-1.8, e_m0=4.8), SWAP, 2.0, lz_mode(L))
    ev = entries_of(info)
    C.ok("e the inversion removed: diabatic, delta -1, inversions 0", len(ev) == 1 and ev[0]["branch"] == "diabatic" and ev[0]["inversion_delta"] == -1 and np.array_equal(m_out, MASK) and info["inversions_end"] == 0, "delta %s inv %s" % (ev[0]["inversion_delta"] if ev else None, info["inversions_end"]))
    # f. a block
    m_out, info, _ = decide(L, heavy_h(0.0, slope=10.0), MASK, 2.0, lz_mode(L, prim_levels=2))
    C.ok("f block at level 2: unresolved and held", info["ran"] and not info["stable"] and info["unresolved"] and bool(hold(L, info, 0.5)), "stable %s counts %s" % (info["stable"], info["counts"]))
    m_out, info, _ = decide(L, heavy_h(0.0, slope=10.0), MASK, 2.0, lz_mode(L, prim_levels=4))
    ev = entries_of(info)
    C.ok("f block at level 4: four diabatic events, stable, the set at index 7", info["stable"] and info["counts"]["diabatic"] == 4 and len(ev) == 4 and np.array_equal(m_out, idx_mask(8, [0, 1, 2, 7])) and info["inversions_end"] == 1, "stable %s counts %s mask %s" % (info["stable"], info["counts"], m_out.astype(int).tolist()))
    # g. quiet; h. skip
    mw = MASK.copy()
    m_out, info, spec = decide(L, heavy_h(0.0, slope=0.5), MASK, 2.0, lz_mode(L, lz_skip_gap=0.0), mask_walk=mw)
    C.ok("g quiet: no passage, the walk's mask object returned, two spectra (the set's columns at 0, the point 1/2)", info["quiet"] and info["n_spectra"] == 2 and m_out is mw, "quiet %s spectra %d same object %s" % (info["quiet"], info["n_spectra"], m_out is mw))
    p = ToyPath(heavy_h(0.0, slope=0.5)); e0, U0 = p.eig(0.0); ref0 = U0[:, :4]
    m_out, info, spec = decide(L, None, MASK, 2.0, lz_mode(L, lz_skip_gap=0.0), mask_walk=mw, path=p, ref0=ref0)
    C.ok("g quiet with the previous reference given: one spectrum (the point 1/2)", info["quiet"] and info["n_spectra"] == 1 and m_out is mw, "quiet %s spectra %d" % (info["quiet"], info["n_spectra"]))
    ph_ = ToyPath(heavy_h(0.0, slope=0.1)); e0h_, U0h_ = ph_.eig(0.0)
    m_out, info, spec = decide(L, None, MASK, 2.0, lz_mode(L), mask_walk=mw, path=ph_, ref0=U0h_[:, :4])
    C.ok("h skip: wide quiet boundary pairs and the previous reference clean at s = 1, no spectrum", info["skipped"] and info["n_spectra"] == 0 and m_out is mw, "skipped %s spectra %d" % (info["skipped"], info["n_spectra"]))
    pm_ = ToyPath(mirror_h(0.0)); e0m_, U0m_ = pm_.eig(0.0)
    m_out, info, spec = decide(L, None, MASK, 2.0, lz_mode(L), mask_walk=mw, path=pm_, ref0=U0m_[:, :4])
    C.ok("h no skip on an exact exchange (the mirror toy: the sorted spectrum's ends alike, the reference's character at s = 1 off its start indices): the passage found", not info["skipped"] and info["ran"] and len(entries_of(info)) == 1 and np.array_equal(m_out, SWAP), "skipped %s ran %s entries %d" % (info["skipped"], info["ran"], len(entries_of(info))))
    # i. a passage straddling two records
    h = heavy_h(0.05, slope=0.95)
    m1, info1, _ = decide(L, h, MASK, 20.0, lz_mode(L))
    appr = [c for c in info1["entries"] if c.get("kind") == "approach"]
    h2 = heavy_h(0.05, slope=0.95, e_m0=3.95)
    e1, U1 = np.linalg.eigh(h(1.0)); p_end = (np.abs(U1[:, :4].conj().T @ np.linalg.eigh(h(0.0))[1][:, :4]) ** 2).sum(axis=1).min()
    C.ok("i first chord: an approach entry, the set kept, stable, wmin below 0.9 with an entry of its own (no hold)", info1["ran"] and len(appr) == 1 and not entries_of(info1) and np.array_equal(m1, MASK) and info1["stable"] and p_end < 0.9 and not hold(L, info1, float(p_end)),
         "approaches %d wmin %.3f s* %s" % (len(appr), p_end, appr[0]["s"] if appr else None))
    m2, info2, _ = decide(L, h2, MASK, 20.0, lz_mode(L))
    ev = entries_of(info2)
    C.ok("i second chord: the incomplete passage decided diabatic, the set swapped", len(ev) == 1 and ev[0]["kind"] == "incomplete" and ev[0]["branch"] == "diabatic" and ev[0]["decided"] and np.array_equal(m2, SWAP) and info2["stable"] and abs(ev[0]["s_star"] - 0.05 / 0.95) < 2e-2,
         "kind %s branch %s s* %s mask %s" % (ev[0]["kind"] if ev else None, ev[0]["branch"] if ev else None, ev[0]["s_star"] if ev else None, m2.astype(int).tolist()))
    # j. the G4s8 record form through the wrapper (the toy light route in v9b's signature); the toy heavy route
    L["V9C_CTX"]["tau"] = 2.0
    p = ToyPath(heavy_h(5e-4, slope=0.0912, e_m0=3.955), light=True); e0, s0, U0 = p.eig(0.0); trial = p.eig(1.0); p.trial = trial
    calls = []
    W = L["v9c_wrap_walk"]("light", functools.partial(toy_light_walk, calls=calls))
    L["V9C_MODE"]["anchor_passage"] = "lz"
    try:
        cert, sec, anchors, occp = W(p, None, e0, s0, MASK, trial, 4, 1e-7, 4096, 1e-8, mu_prev=3.5)
    finally:
        L["V9C_MODE"]["anchor_passage"] = "floor"
    ent = sec.get("crossings") or []
    C.ok("j G4s8 form (light): the decided set, the route re-called once at x_n (V9cEndPath, mu_prev None), anchors 3, inversions 1", np.array_equal(occp, SWAP) and len(calls) == 2 and calls[1][0] == "V9cEndPath" and calls[1][1] is None and sec["passage"]["mask_changed"] and sec["passage"]["recall"] is not None and sec["anchors_end"] == 3 and len(anchors) == 3 and sec["inversions_plus_end"] == 1 and sec["route"] == "sector walk",
         "occp %s calls %s anchors %s inv %s route %s" % (occp.astype(int).tolist(), calls, sec.get("anchors_end"), sec.get("inversions_plus_end"), sec.get("route")))
    C.ok("j G4s8 form (light): the entry diabatic with its exponent, the primitive's version in the entry, no hold reason", len(ent) == 1 and ent[0]["branch"] == "diabatic" and abs(ent[0]["exponent"] / 3.4443e-5 - 1.0) < 1e-2 and sec["passage"]["prim"].startswith("assign_primitive") and not L["v9c_hold_reasons"](dict(sec, min_occupied_weight=0.999)) and sec["crossings_walk"] == [],
         "%s exponent %s hold %s" % (sec["passage"].get("prim"), ent[0]["exponent"] if ent else None, L["v9c_hold_reasons"](dict(sec, min_occupied_weight=0.999))))
    ph = ToyPath(heavy_h(5e-4, slope=0.0912, e_m0=3.955)); e0h, U0h = ph.eig(0.0); trialh = ph.eig(1.0); ph.trial = trialh
    callsh = []
    WH = L["v9c_wrap_walk"]("heavy", functools.partial(toy_heavy_walk, calls=callsh))
    L["V9C_MODE"]["anchor_passage"] = "lz"
    try:
        walk, occ = WH(ph, U0h[:, :4], e0h, MASK, trialh, 4, 1e-7, 4096, 1e-8)
    finally:
        L["V9C_MODE"]["anchor_passage"] = "floor"
    C.ok("j G4s8 form (heavy): the decided set, re-called once at x_n, anchors 3, inversions 1, the entry diabatic", np.array_equal(occ, SWAP) and len(callsh) == 2 and callsh[1][0] == "V9cEndPath" and walk["passage"]["mask_changed"] and walk["anchors_end"] == 3 and walk["inversions_end"] == 1 and len(walk["crossings"]) == 1 and walk["crossings"][0]["branch"] == "diabatic",
         "occ %s calls %d anchors %s inv %s" % (occ.astype(int).tolist(), len(callsh), walk.get("anchors_end"), walk.get("inversions_end")))
    # k. floor mode through the wrapper; the caller's stash
    calls = []
    W = L["v9c_wrap_walk"]("light", functools.partial(toy_light_walk, calls=calls))
    out0 = W(p, None, e0, s0, MASK, trial, 4, 1e-7, 4096, 1e-8, mu_prev=3.5)
    C.ok("k floor mode: the route's return untouched, one call, no passage block, the set the walk's", isinstance(out0, tuple) and len(calls) == 1 and "passage" not in out0[1] and np.array_equal(out0[3], MASK), "calls %d keys %s" % (len(calls), sorted(out0[1].keys())[:5]))
    seen = {}
    def toy_caller(a, ch, window, *rest, **kw):
        seen["ref"] = L["V9C_CTX"]["light_ref"]; return "ok"
    WC = L["v9c_wrap_light_caller"](toy_caller)
    r1 = WC(None, None, [{"ref": "REF"}]); s_floor = seen.get("ref")
    L["V9C_MODE"]["anchor_passage"] = "lz"
    try:
        r2 = WC(None, None, [{"ref": "REF"}]); s_lz = seen.get("ref")
    finally:
        L["V9C_MODE"]["anchor_passage"] = "floor"
    C.ok("k the light caller's stash: untouched in floor mode, the window's last reference in lz mode, cleared after", r1 == "ok" and r2 == "ok" and s_floor is None and s_lz == "REF" and L["V9C_CTX"]["light_ref"] is None, "floor %s lz %s after %s" % (s_floor, s_lz, L["V9C_CTX"]["light_ref"]))
    # l. the walk's crossing merged; a bogus one flagged
    pl = ToyPath(heavy_h(0.0), light=True); e0, s0, U0 = pl.eig(0.0); trial = pl.eig(1.0); pl.trial = trial
    calls = []
    W = L["v9c_wrap_walk"]("light", functools.partial(toy_light_walk, crossings=[walk_entry(1.0 / 1.8)], occp_end=SWAP, calls=calls))
    L["V9C_MODE"]["anchor_passage"] = "lz"
    try:
        cert, sec, anchors, occp = W(pl, None, e0, s0, MASK, trial, 4, 1e-7, 4096, 1e-8, mu_prev=None)
        calls2 = []
        W2 = L["v9c_wrap_walk"]("light", functools.partial(toy_light_walk, crossings=[walk_entry(0.9)], occp_end=SWAP, calls=calls2))
        cert2, sec2, anchors2, occp2 = W2(pl, None, e0, s0, MASK, trial, 4, 1e-7, 4096, 1e-8, mu_prev=None)
    finally:
        L["V9C_MODE"]["anchor_passage"] = "floor"
    cr = sec.get("crossings") or []
    C.ok("l walk crossing merged: source walk+primitive, the walk's keys kept (overlap, block 2), the decision added, no re-call, no hold reason", len(calls) == 1 and len(cr) == 1 and cr[0]["source"] == "walk+primitive" and cr[0]["branch"] == "diabatic" and cr[0]["overlap"] == 0.99995 and cr[0]["block"] == 2 and (cr[0].get("exponent") or 0.0) < 1e-9 and sec["passage"]["walk_consistent"] and not L["v9c_hold_reasons"](dict(sec, min_occupied_weight=0.999)) and np.array_equal(occp, SWAP) and not sec["passage"]["mask_changed"],
         "source %s reasons %s" % (cr[0]["source"] if cr else None, L["v9c_hold_reasons"](dict(sec, min_occupied_weight=0.999))))
    cr2 = sec2.get("crossings") or []
    C.ok("l bogus walk swap: flagged, inconsistent, holding; the primitive's event appended", not sec2["passage"]["walk_consistent"] and any(c.get("source") == "walk" and c.get("flagged") for c in cr2) and any(c.get("source") == "primitive" for c in cr2) and bool(L["v9c_hold_reasons"](sec2)), "reasons %s" % [r[0][:60] for r in L["v9c_hold_reasons"](sec2)])
    # m. the arguments
    ap = argparse.ArgumentParser(); L["v9c_add_arguments"](ap)
    before = dict(L["V9C_MODE"]); ns = ap.parse_args([])
    same_defaults = all(getattr(ns, k) == v for k, v in L["V9C_DEFAULTS"].items()) and L["V9C_MODE"] == before == L["V9C_DEFAULTS"]
    ns2 = ap.parse_args(["--anchor-passage", "lz", "--lz-c", "2.0", "--prim-levels", "3"]); ns2.tau = 2.0; ns2.cut_transport = "sector"
    M2 = L["v9c_mode_set"](ns2)
    reached = L["V9C_MODE"]["anchor_passage"] == "lz" and L["V9C_MODE"]["lz_c"] == 2.0 and L["V9C_MODE"]["prim_levels"] == 3 and M2["lz_c"] == 2.0 and L["V9C_CTX"]["tau"] == 2.0
    refused = []
    for bad in (dict(cut_transport="record"), dict(prim_levels=1), dict(tau=0.0)):
        ns3 = ap.parse_args(["--anchor-passage", "lz"]); ns3.tau = 2.0; ns3.cut_transport = "sector"
        for k_, v_ in bad.items():
            setattr(ns3, k_, v_)
        try:
            L["v9c_mode_set"](ns3); refused.append(False)
        except SystemExit:
            refused.append(True)
    L["V9C_MODE"].clear(); L["V9C_MODE"].update(L["V9C_DEFAULTS"]); L["V9C_CTX"]["tau"] = 2.0
    C.ok("m arguments: the defaults equal V9C_DEFAULTS and leave V9C_MODE untouched; lz, lz_c 2 and tau reach the mode; lz refuses the record transport, prim_levels 1 and tau 0", same_defaults and reached and all(refused), "defaults %s reached %s refused %s" % (same_defaults, reached, refused))
    # n. tau_LZ halves with tau
    _, i2, _ = decide(L, heavy_h(5e-4), MASK, 2.0, lz_mode(L)); _, i1, _ = decide(L, heavy_h(5e-4), MASK, 1.0, lz_mode(L))
    A2 = entries_of(i2)[0]["tau_lz"] if entries_of(i2) else None; A1 = entries_of(i1)[0]["tau_lz"] if entries_of(i1) else None
    C.ok("n tau_LZ at tau 1 is half of tau 2's", A1 is not None and A2 is not None and abs(A1 / A2 - 0.5) < 1e-6, "A(tau 1) %s A(tau 2) %s" % (A1, A2))
    # o. the torch spectrum object
    if skip_torch:
        print("SKIP o torch (--skip-torch)")
    else:
        try:
            import torch
            pt = ToyPath(heavy_h(0.05), torch_mod=torch); pn = ToyPath(heavy_h(0.05))
            m_t, info_t, _ = decide(L, None, MASK, 20.0, lz_mode(L), path=pt); m_n, info_n, _ = decide(L, None, MASK, 20.0, lz_mode(L), path=pn)
            ptl = ToyPath(heavy_h(0.05), light=True, torch_mod=torch)
            m_tl, info_tl, _ = decide(L, None, MASK, 20.0, lz_mode(L), path=ptl, light=True)
            C.ok("o torch agrees with numpy (the heavy's and the light's spectrum objects)", np.array_equal(np.asarray(m_t), m_n) and np.array_equal(np.asarray(m_tl), m_n) and entries_of(info_t)[0]["branch"] == entries_of(info_n)[0]["branch"] == entries_of(info_tl)[0]["branch"] and abs(entries_of(info_t)[0]["tau_lz"] - entries_of(info_n)[0]["tau_lz"]) < 1e-6 and abs(entries_of(info_tl)[0]["tau_lz"] - entries_of(info_n)[0]["tau_lz"]) < 1e-6,
                 "tau_LZ torch %s light %s numpy %s" % (entries_of(info_t)[0]["tau_lz"], entries_of(info_tl)[0]["tau_lz"], entries_of(info_n)[0]["tau_lz"]))
        except ImportError:
            print("SKIP o torch (torch not present)")
    # p. the exponent convention
    _, ia, _ = decide(L, heavy_h(0.05), MASK, 20.0, lz_mode(L)); _, ib, _ = decide(L, heavy_h(0.05), MASK, 20.0, lz_mode(L, lz_k=math.pi / 2.0))
    ea = entries_of(ia)[0]; eb = entries_of(ib)[0]
    C.ok("p exponent = pi tau_LZ / (2 k); with k = pi/2 they coincide", abs(ea["exponent"] - math.pi / 2.0 * ea["tau_lz"]) < 1e-12 and abs(eb["exponent"] - eb["tau_lz"]) < 1e-12 and abs(eb["tau_lz"] - math.pi / 2.0 * ea["tau_lz"]) < 1e-9, "k 1: %s / %s; k pi/2: %s / %s" % (ea["tau_lz"], ea["exponent"], eb["tau_lz"], eb["exponent"]))
    # q. determinism
    _, ic, _ = decide(L, mirror_h(0.05), MASK, 20.0, lz_mode(L)); _, idd, _ = decide(L, mirror_h(0.05), MASK, 20.0, lz_mode(L))
    C.ok("q determinism: identical entries", json.dumps(ic["entries"], sort_keys=True) == json.dumps(idd["entries"], sort_keys=True))
    # r. v9c_hold_reasons on G4s8's forms
    r1 = L["v9c_hold_reasons"]({"route": "sector walk", "crossings": [], "min_occupied_weight": 0.0027})
    r2 = L["v9c_hold_reasons"]({"route": "sector walk", "crossings": [walk_entry(0.9519, overlap=0.8365, swap=False, delta=0)], "min_occupied_weight": 0.9999})
    r3 = L["v9c_hold_reasons"]({"route": "sector walk", "crossings": [walk_entry(0.5)], "min_occupied_weight": 0.9999})
    C.ok("r hold reasons: 13.125's form names the unresolved passage; a walk overlap 0.8365 holds by 4.6; a clean walk swap (block 2) does not hold", len(r1) == 1 and r1[0][0].startswith("unresolved passage") and len(r2) == 1 and r2[0][0].startswith("passage overlap") and not r3, "%s | %s | %s" % ([r[0] for r in r1], [r[0] for r in r2], r3))
    # x. (C64_v9c_review_v1 finding 9) the chord's extension by one piece: the member (slope 2.4 from 3.0) passes level 4 (gap 0.1, s* 0.4167) and level 5 (exact, s* 0.8333) in one chord; at level 1 the first passage is seen at 0.5 and moves the anchor there, so the second, completed in the last piece, has two samples (0.5 and 1.0) and takes its third on the chord extended to s = 1.5
    hx = heavy_h(0.05, slope=2.4, e_m0=3.0); MASKX = idx_mask(8, [0, 1, 2, 5])
    mx, ix, _ = decide(L, hx, MASK, 2.0, lz_mode(L))
    evx = entries_of(ix)
    okx = (len(evx) == 2 and all(e_["branch"] == "diabatic" and e_["decided"] and not e_["flagged"] for e_ in evx) and abs(evx[0]["s_star"] - 1.0 / 2.4) < 1e-6 and abs(evx[1]["s_star"] - 2.0 / 2.4) < 1e-6
           and evx[1]["fit_kind"] == "parabola" and any("third sample at s=1.5000" in f_ for f_ in evx[1]["flags"]) and abs(evx[1]["gap_min"]) < 1e-6 and np.array_equal(mx, MASKX) and ix["stable"] and ix.get("extension") is None and not hold(L, ix))
    C.ok("x extension: two diabatic passages at s* 0.4167 and 0.8333, the second fitted through the third sample on the chord extended to s = 1.5 (the toy extends, so the verification is not consulted), the set at index 5, stable, no hold reason", okx,
         "s* %s fit %s flags %s extension %s mask %s" % ([e_.get("s_star") for e_ in evx], [e_.get("fit_kind") for e_ in evx], evx[1].get("flags") if len(evx) > 1 else None, ix.get("extension"), mx.astype(int).tolist()))
    mxn, ixn, _ = decide(L, None, MASK, 2.0, lz_mode(L), path=ToyPathNoExt(hx))
    evxn = entries_of(ixn)
    okxn = (len(evxn) == 2 and all(e_["branch"] == "diabatic" and e_["decided"] and not e_["flagged"] for e_ in evxn) and evxn[1]["fit_kind"] == "grid" and any("refused by the oracle" in f_ for f_ in evxn[1]["flags"])
            and evxn[1]["exponent"] == 0.0 and evxn[1]["tau_lz_band"][1] is not None and abs(evxn[1]["tau_lz_band"][1] - 0.16 / 1.2) < 1e-6 and isinstance(ixn.get("extension"), str) and "not in scope" in ixn["extension"]
            and np.array_equal(mxn, MASKX) and ixn["stable"] and not hold(L, ixn))
    C.ok("x extension refused (the verification at s = 1 fails: no engine eigendecomposition in the layer's scope): the second passage on the two-sample bound, decided diabatic (exponent 0, the band's top 0.133), flagged as information, the refusal in the passage block, the set at index 5, stable, no hold reason", okxn,
         "fit %s exponent %s band %s flags %s extension %s" % ([e_.get("fit_kind") for e_ in evxn], evxn[1].get("exponent") if len(evxn) > 1 else None, evxn[1].get("tau_lz_band") if len(evxn) > 1 else None, evxn[1].get("flags") if len(evxn) > 1 else None, ixn.get("extension")))

    # y. (ReviewA_busi_open_q12_v1 sec. 1; [v1y]) the late-step two-sample toy beside x, in x's class: toy_late_step_v1.py (variants X, C, R)
    import toy_late_step_v1 as Y  # [v1y]
    Y.run_late_step_checks(C, L, sys.modules[__name__])  # [v1y]


# ------------------------------------------------------------------------------------------------------- the build
def build_checks(C, a):
    src = B.read(a.v9b)
    names = (a.light_def, a.heavy_def, a.light_caller, a.run_def)
    reps = B.replacements(*names)
    bad = B.precheck(src, reps)
    defs_ok = all(src.count("def %s(" % n_) == 1 for n_ in names[1:])
    if not C.ok("s build: every hard anchor occurs once in v9b (md5 %s)" % B.md5(src), not bad and defs_ok, "; ".join("%s occurs %d" % (r.name, n) for r, n in bad) + ("" if defs_ok else "; a def line not once")):
        print("  (python build_engine_twofield_v9c.py %s --probe lists the candidate lines; the lane posts them)" % a.v9b)
        return None
    out, log = B.apply(src, reps)
    marks = B.marks_check(src, out)
    back, _ = B.apply(out, reps, reverse=True)
    n_marked = sum(1 for line in out.split("\n") if B.MARK in line)
    C.ok("s build: the marks (every new line carries v9c prim) and the reverse round trip byte for byte", not marks and back == src, "%d unmarked; %d marked lines; reverse %s; md5 %s" % (len(marks), n_marked, "ok" if back == src else "differs", B.md5(out)))
    if a.v9c:
        C.ok("s build: equals the given engine_twofield_v9c.py byte for byte", B.read(a.v9c) == out, "%s (md5 %s)" % (a.v9c, B.md5(B.read(a.v9c))))
    return out


def import_built(path):
    spec = importlib.util.spec_from_file_location("engine_twofield_v9c", path)
    mod = importlib.util.module_from_spec(spec); sys.modules["engine_twofield_v9c"] = mod
    spec.loader.exec_module(mod)
    return mod


def import_checks(C, a, mod):
    light = getattr(mod, a.light_def, None); heavy = getattr(mod, a.heavy_def, None); caller = getattr(mod, a.light_caller, None); run_ = getattr(mod, a.run_def, None)
    C.ok("t built engine: V9C_MODE equals V9C_DEFAULTS at import, the four names wrapped, V9C_MISSING empty",
         mod.V9C_MODE == mod.V9C_DEFAULTS and all(hasattr(f_, "__wrapped__") for f_ in (light, heavy, caller, run_)) and not mod.V9C_MISSING,
         "missing %s wrapped %s" % (mod.V9C_MISSING, [hasattr(f_, "__wrapped__") for f_ in (light, heavy, caller, run_)]))
    C.ok("t built engine: the version constant", getattr(mod, "ENGINE_VERSION", "engine_twofield_v9c").endswith("v9c") and mod.V9C_VERSION == "engine_twofield_v9c", "%s" % getattr(mod, "ENGINE_VERSION", "(no ENGINE_VERSION in v9b; the soft replacement skipped)"))
    for sp, fn in (("light", light), ("heavy", heavy)):
        params = set(inspect.signature(fn.__wrapped__).parameters)
        need = [mod.V9C_NAMES[sp][k] for k in mod.V9C_REQUIRED[sp]]
        C.ok("t built engine: the %s route binds the required names %s" % (sp, need), all(n in params for n in need), "signature %s" % str(inspect.signature(fn.__wrapped__)))
    ap = argparse.ArgumentParser(); mod.v9c_add_arguments(ap); ns = ap.parse_args([])
    C.ok("t built engine: the arguments' defaults equal V9C_DEFAULTS", all(getattr(ns, k) == v for k, v in mod.V9C_DEFAULTS.items()), str(vars(ns)))


# ------------------------------------------------------------------------------------------------------- the fixture (the real routes)
def pair_path(V, S9, chL, N0, v0, sgn_dev, tol_block, g, const=False, s_star=0.5):
    """the light channel's pair at the cut on a manufactured path, two forms, the diabats' half-rate cp = c / (2 s_star) (the crossing at s_star;
    the exact crossings are placed at 0.45, off the primitive's dyadic points: at a dyadic point the pair is degenerate below the sectorization's
    block tolerance, its eigenvectors there arbitrary, and a clean-point sample at the minimum itself leaves the two-level fit's vertex unbracketed
    by rounding, an unresolved event by the primitive's rule). const False: engine_twofield_v9b_selftest.light_pair_path's own construction, h(s) =
    h0 + s (X - X*) with the coupling (g/2)(|a><b| + |b><a|) inside X, so the coupling grows along the chord (g s / 2) and the pair's splitting is
    Delta^2 = 4 cp^2 (s - s_star)^2 + g^2 s^2: at s_star 1/2 the minimum c g / sqrt(4 c^2 + g^2) (about g/2) at 2 c^2 / (4 c^2 + g^2), alpha_s =
    sqrt(4 c^2 + g^2) (derived; the walk's checks read the same path). const True: the coupling in h0 (h0 + Y - Y*, Y = (g/2)(|a><b| + |b><a|) on
    the diabats of h0) and D = cp (|a><a| - |b><b|) - its conjugate: the Landau-Zener form with the constant gap g at s_star and alpha_s = 2 cp, the
    HOLD ruling's (4b) numbers (the exponent pi g^2 / (2 alpha_T), 3.44e-5 at g 1e-3 and alpha_T 0.0456). Returns (path, e0, sig0, occ0 (the mu S
    selection), ref0, n_sea, mu0, c, g_min, alpha_s)."""
    torch = V.BK["torch"]
    h0 = V.clocked_gpu(chL, N0, v0)
    path0 = S9.ManufacturedLightPath(V, h0, torch.zeros_like(h0), sgn_dev, tol_block)
    e0, sig0, U0 = path0.eig(0.0)
    n_plus0 = int(np.sum((sig0 > 0.0) & (e0 < 0.0)))
    order0, ep0 = V.light_plus_order_v9(e0, sig0)
    ia, ib = int(order0[n_plus0 - 1]), int(order0[n_plus0])
    ua = U0[:, ia].clone(); ub = U0[:, ib].clone(); c = float(ep0[n_plus0] - ep0[n_plus0 - 1])
    cp = c / (2.0 * float(s_star))
    X = cp * (torch.outer(ua, ua.conj()) - torch.outer(ub, ub.conj()))
    Y = (0.5 * g) * (torch.outer(ua, ub.conj()) + torch.outer(ub, ua.conj())) if g > 0.0 else None
    if const:
        hc = h0 + (Y - Y.conj()) if Y is not None else h0
        path = S9.ManufacturedLightPath(V, hc, X - X.conj(), sgn_dev, tol_block)
        e0, sig0, U0 = path.eig(0.0)
        n_plus0 = int(np.sum((sig0 > 0.0) & (e0 < 0.0)))
        g_min = float(g); alpha_s = 2.0 * cp
    else:
        path = path0
        path.D = (X + Y - (X + Y).conj()) if Y is not None else (X - X.conj())
        s_min = 4.0 * cp * cp * float(s_star) / (4.0 * cp * cp + g * g)
        g_min = math.sqrt(4.0 * cp * cp * (s_min - float(s_star)) ** 2 + g * g * s_min * s_min); alpha_s = math.sqrt(4.0 * cp * cp + g * g)
    mu0, _gp = V.light_mu_midgap_v36(e0, sig0, n_plus0)
    occ0, _gap0 = V.light_select_v36(e0, sig0, mu0)
    ref0 = U0[:, torch.as_tensor(occ0, device=U0.device)].clone()
    return path, e0, sig0, occ0, ref0, int(occ0.sum()), mu0, c, g_min, alpha_s


def run_fixture_checks(C, V9, S8, S9, gpu, n=128, kappa_sea=3):
    V9.C5.backend(gpu)
    V35 = V9.V35; V35.V36["fixes"] = True
    a = S9.args9(S8, n, kappa_sea, "lagged", "sector")
    FX = S8.make_fixture(V9, a)
    S8.sparse_probe_and_patch([V9], FX.chs[0])
    torch = V9.BK["torch"]
    chL = FX.chl[0]; chH = FX.chs[0]; N0, v0 = FX.f_ref; tol_block = a.light_tol_block; d_min = a.light_d_min
    CERT = V9.CERT
    light = getattr(V9, "light_sector_walk_v9"); heavy = getattr(V9, "heavy_sector_walk_v9")
    print("fixture: %d sites, heavy kappa 1..%d, light %s sourcing and %s the guard, device %s; the routes %s / %s (wrapped %s / %s)" % (n, kappa_sea, a.light_kappas, a.light_guard_kappas, V9.BK["dev"], light.__name__, heavy.__name__, hasattr(light, "__wrapped__"), hasattr(heavy, "__wrapped__")), flush=True)

    def lz(on=True, **over):
        tau = float(over.pop("tau", 2.0))
        V9.V9C_MODE.clear(); V9.V9C_MODE.update(V9.V9C_DEFAULTS)
        if on:
            V9.V9C_MODE["anchor_passage"] = "lz"; V9.V9C_MODE.update(over)
        V9.V9C_CTX["tau"] = tau if on else None; V9.V9C_CTX["light_ref"] = None

    def mask_of(e1, sig1, anchors):
        return V9.light_select_sector_v9(e1, sig1, anchors)[0] if anchors is not None else None

    def cut_of(ref0, e1, sig1, mask, U1, anchors, cert):
        return V9.light_cut_check_sector_v9(ref0, e1, sig1, mask, U1, anchors, cert["norm_dK_G"], V9.light_select_sector_v9(e1, sig1, anchors)[1])

    try:
        # (c) the exact crossing: F1 floor identity, F2 lz
        path, e0, sig0, occ0, ref0, n_sea, mu0, c, _gm, _as = pair_path(V9, S9, chL, N0, v0, FX.sgn_dev, tol_block, g=0.0, s_star=0.45)
        trial = path.eig(1.0); path.trial = trial; e1, sig1, U1 = trial
        lz(False)
        path.eigh_n = 0
        out_w = light(path, FX.S_host, e0, sig0, occ0, trial, n_sea, d_min, 4096, tol_block, mu_prev=mu0)
        path.eigh_n = 0
        out_r = light.__wrapped__(path, FX.S_host, e0, sig0, occ0, trial, n_sea, d_min, 4096, tol_block, mu_prev=mu0)
        diffs = []
        S8.deep_diff(strip_seconds(out_w[0]), strip_seconds(out_r[0]), "cert", diffs); S8.deep_diff(out_w[1], out_r[1], "sector", diffs); S8.deep_diff(list(out_w[2]), list(out_r[2]), "anchors", diffs); S8.deep_diff(np.asarray(out_w[3], bool), np.asarray(out_r[3], bool), "occp", diffs)
        C.ok("F1 floor identity: the wrapped light route returns what the unwrapped route returns on (c)'s path (every number but the seconds), no passage block", not diffs and "passage" not in out_w[1], "%d differences %s" % (len(diffs), diffs[:3]))
        mask_c = mask_of(e1, sig1, out_w[2]); cr_w = out_w[1]["crossings"]
        lz(True)
        cert, sec, anchors, occp = light(path, FX.S_host, e0, sig0, occ0, trial, n_sea, d_min, 4096, tol_block, mu_prev=mu0)
        cr = sec.get("crossings") or []; P = sec.get("passage") or {}
        mask_lz = mask_of(e1, sig1, anchors)
        C.ok("F2 (c) exact crossing in lz: the walk's crossing merged with the event (diabatic, exponent 0), no re-call, anchors 3, inversions 1, the mask (c)'s, no hold reason",
             cert["certified"] and len(cr) == 1 and cr[0]["source"] == "walk+primitive" and cr[0]["branch"] == "diabatic" and (cr[0].get("exponent") or 0.0) < 1e-6 and cr[0]["swap"] is True and P.get("ran") and not P.get("mask_changed") and P.get("recall") is None and P.get("walk_consistent") and sec["anchors_end"] == 3 and sec["inversions_plus_end"] == 1 and mask_lz is not None and np.array_equal(mask_lz, mask_c) and not V9.v9c_hold_reasons(dict(sec, min_occupied_weight=0.999)) and len(cr_w) == 1,
             "certified %s crossings %d source %s branch %s exponent %s gap_min %s s* %s (the walk's piece %s) recall %s spectra %s hold %s" % (cert["certified"], len(cr), cr[0].get("source") if cr else None, cr[0].get("branch") if cr else None, cr[0].get("exponent") if cr else None, cr[0].get("gap_min") if cr else None, cr[0].get("s_star") if cr else None, cr[0].get("piece") if cr else None, P.get("recall"), P.get("n_spectra"), V9.v9c_hold_reasons(dict(sec, min_occupied_weight=0.999))))
        # (d 1e-3), the v9b fixture's own path: F3 the G4s8 record form; F6 tau halving; F9 determinism
        pathd, e0d, sig0d, occ0d, ref0d, n_sead, mu0d, cd, gmd, asd = pair_path(V9, S9, chL, N0, v0, FX.sgn_dev, tol_block, g=1e-3)
        exp_d = math.pi * gmd * gmd / (2.0 * asd / 2.0)
        triald = pathd.eig(1.0); pathd.trial = triald; e1d, sig1d, U1d = triald
        lz(False)
        certf, secf, anchf, occpf = light(pathd, FX.S_host, e0d, sig0d, occ0d, triald, n_sead, d_min, 4096, tol_block, mu_prev=mu0d)
        maskf = mask_of(e1d, sig1d, anchf); cutf = cut_of(ref0d, e1d, sig1d, maskf, U1d, anchf, certf)
        results = []
        for stash in (False, True):
            lz(True)
            V9.V9C_CTX["light_ref"] = ref0d if stash else None
            try:
                certd, secd, anchd, occpd = light(pathd, FX.S_host, e0d, sig0d, occ0d, triald, n_sead, d_min, 4096, tol_block, mu_prev=mu0d)
            finally:
                V9.V9C_CTX["light_ref"] = None
            results.append((certd, secd, anchd, occpd))
        (certd, secd, anchd, occpd), (certd2, secd2, anchd2, occpd2) = results
        crd = secd.get("crossings") or []; Pd = secd.get("passage") or {}
        maskd = mask_of(e1d, sig1d, anchd); cutd = cut_of(ref0d, e1d, sig1d, maskd, U1d, anchd, certd) if maskd is not None else {}
        wsel = cutd.get("min_weight_selected")
        ok3 = (certf["certified"] and not (secf.get("crossings") or []) and cutf["min_weight_selected"] is not None and cutf["min_weight_selected"] < 0.01 and cutf["crossing_levels"] == 4
               and certd["certified"] and len(crd) == 1 and crd[0]["source"] == "primitive" and crd[0]["branch"] == "diabatic" and crd[0]["decided"] and not crd[0]["flagged"] and abs(crd[0]["exponent"] / exp_d - 1.0) < 1e-2 and abs(crd[0]["alpha_T"] / (asd / 2.0) - 1.0) < 1e-3 and abs(crd[0]["gap_min"] / gmd - 1.0) < 1e-3
               and Pd.get("mask_changed") and Pd.get("recall") is not None and Pd.get("walk_consistent") and Pd.get("stable") and secd["anchors_end"] == 3 and secd["inversions_plus_end"] == 1 and secd["route"] == "sector walk"
               and maskd is not None and np.array_equal(maskd, mask_c) and cutd.get("crossing_levels") == 0 and wsel is not None and wsel > 0.99 and not V9.v9c_hold_reasons(dict(secd, min_occupied_weight=float(wsel))))
        C.ok("F3 (d 1e-3) in lz on the v9b fixture's own path (the coupling grows along the chord: the minimum gap c g / sqrt(4 c^2 + g^2) = %.4e, alpha_s %.4f, the exponent %.3e at tau 2, derived), the G4s8 record form: the walk kept the set (crossings 0, weight %s below 1/2, crossing_levels %s); the primitive decides diabatic with that exponent, re-called at x_n, anchors 3, inversions 1, the final mask (c)'s, the cut check crossing_levels 0 with min weight above 0.99, no hold reason" % (
            gmd, asd, exp_d, ("%.1e" % cutf["min_weight_selected"]) if cutf.get("min_weight_selected") is not None else None, cutf.get("crossing_levels")), ok3,
            "certified %s crossings %d branch %s exponent %s alpha_T %s gap_min %s s* %s mask_changed %s recall %s anchors %s inversions %s route %s mask equal %s crossing_levels %s min weight %s spectra %s (path eigh %s) hold %s" % (
                certd["certified"], len(crd), crd[0].get("branch") if crd else None, crd[0].get("exponent") if crd else None, crd[0].get("alpha_T") if crd else None, crd[0].get("gap_min") if crd else None, crd[0].get("s_star") if crd else None, Pd.get("mask_changed"), Pd.get("recall"), secd.get("anchors_end"), secd.get("inversions_plus_end"), secd.get("route"),
                bool(np.array_equal(maskd, mask_c)) if maskd is not None else None, cutd.get("crossing_levels"), wsel, Pd.get("n_spectra"), Pd.get("eigh_path"), V9.v9c_hold_reasons(dict(secd, min_occupied_weight=float(wsel) if wsel is not None else 0.0))))
        Pd2 = secd2.get("passage") or {}
        C.ok("F3 with the previous reference stashed: the same decision and set, one spectrum less (the trigger from the stash)", np.array_equal(np.asarray(occpd2, bool), np.asarray(occpd, bool)) and json.dumps(secd2.get("crossings"), sort_keys=True) == json.dumps(crd, sort_keys=True) and Pd2.get("n_spectra") == Pd.get("n_spectra"),
             "spectra %s against %s (the primitive needs the eigenbasis at 0 once it runs; the stash spares it only on the trigger)" % (Pd2.get("n_spectra"), Pd.get("n_spectra")))
        lz(True)
        certd3, secd3, anchd3, occpd3 = light(pathd, FX.S_host, e0d, sig0d, occ0d, triald, n_sead, d_min, 4096, tol_block, mu_prev=mu0d)
        C.ok("F9 determinism on F3: identical entries and set", json.dumps(secd3.get("crossings"), sort_keys=True) == json.dumps(crd, sort_keys=True) and np.array_equal(np.asarray(occpd3, bool), np.asarray(occpd, bool)))
        lz(True, tau=1.0)
        certt, sect, ancht, occpt = light(pathd, FX.S_host, e0d, sig0d, occ0d, triald, n_sead, d_min, 4096, tol_block, mu_prev=mu0d)
        crt = sect.get("crossings") or []
        C.ok("F6 tau_LZ at tau 1 is half of tau 2's on (d 1e-3)", len(crt) == 1 and crd and abs(crt[0]["tau_lz"] / crd[0]["tau_lz"] - 0.5) < 1e-6, "tau 1: %s, tau 2: %s" % (crt[0].get("tau_lz") if crt else None, crd[0].get("tau_lz") if crd else None))
        # (d 1e-3) with the constant coupling (the Landau-Zener form, the HOLD ruling's numbers): F3c; F5 the slow sweep and the knife edge
        pathc, e0c, sig0c, occ0c, ref0c, n_seac, mu0c, cc, gmc, asc = pair_path(V9, S9, chL, N0, v0, FX.sgn_dev, tol_block, g=1e-3, const=True)
        exp_c = math.pi * gmc * gmc / (2.0 * asc / 2.0)
        trialc = pathc.eig(1.0); pathc.trial = trialc; e1c, sig1c, U1c = trialc
        lz(False)
        certcf, seccf, anchcf, occpcf = light(pathc, FX.S_host, e0c, sig0c, occ0c, trialc, n_seac, d_min, 4096, tol_block, mu_prev=mu0c)
        maskcf = mask_of(e1c, sig1c, anchcf); cutcf = cut_of(ref0c, e1c, sig1c, maskcf, U1c, anchcf, certcf)
        lz(True)
        certc, secc, anchc, occpc = light(pathc, FX.S_host, e0c, sig0c, occ0c, trialc, n_seac, d_min, 4096, tol_block, mu_prev=mu0c)
        crc = secc.get("crossings") or []; Pc = secc.get("passage") or {}
        maskc_ = mask_of(e1c, sig1c, anchc); cutc = cut_of(ref0c, e1c, sig1c, maskc_, U1c, anchc, certc) if maskc_ is not None else {}
        wselc = cutc.get("min_weight_selected")
        okc = (certcf["certified"] and not (seccf.get("crossings") or []) and cutcf.get("min_weight_selected") is not None and cutcf["min_weight_selected"] < 0.01
               and certc["certified"] and len(crc) == 1 and crc[0]["branch"] == "diabatic" and crc[0]["decided"] and not crc[0]["flagged"] and abs(crc[0]["exponent"] / exp_c - 1.0) < 1e-2 and abs(crc[0]["alpha_T"] - 0.0456) < 2e-4 and abs(crc[0]["gap_min"] / gmc - 1.0) < 1e-3 and abs(crc[0]["s_star"] - 0.5) < 1e-3
               and Pc.get("mask_changed") and Pc.get("recall") is not None and Pc.get("walk_consistent") and secc["anchors_end"] == 3 and secc["inversions_plus_end"] == 1 and wselc is not None and wselc > 0.99 and cutc.get("crossing_levels") == 0 and not V9.v9c_hold_reasons(dict(secc, min_occupied_weight=float(wselc))))
        C.ok("F3c (d 1e-3) with the constant coupling (the Landau-Zener form: gap %.1e, alpha_T %.4f, the exponent %.3e, the HOLD ruling's 3.4e-5): the walk kept the set (weight %s); the primitive diabatic with that exponent at s* 1/2, re-called, anchors 3, inversions 1, min weight above 0.99, no hold reason" % (gmc, asc / 2.0, exp_c, ("%.1e" % cutcf["min_weight_selected"]) if cutcf.get("min_weight_selected") is not None else None), okc,
             "crossings %d branch %s exponent %s alpha_T %s gap_min %s s* %s recall %s anchors %s inversions %s min weight %s hold %s" % (len(crc), crc[0].get("branch") if crc else None, crc[0].get("exponent") if crc else None, crc[0].get("alpha_T") if crc else None, crc[0].get("gap_min") if crc else None, crc[0].get("s_star") if crc else None, Pc.get("recall"), secc.get("anchors_end"), secc.get("inversions_plus_end"), wselc, V9.v9c_hold_reasons(dict(secc, min_occupied_weight=float(wselc) if wselc is not None else 0.0))))
        lz(True, tau=2e5)
        certs, secs, anchs, occps = light(pathc, FX.S_host, e0c, sig0c, occ0c, trialc, n_seac, d_min, 4096, tol_block, mu_prev=mu0c)
        crs = secs.get("crossings") or []; Ps = secs.get("passage") or {}
        masks = mask_of(e1c, sig1c, anchs)
        tlz_s = gmc * gmc * 2e5 / asc
        C.ok("F5 the constant coupling at a sweep slower by 1e5 (tau 2e5): resolved adiabatic (tau_LZ %.3g), the set kept, the walk consistent, no re-call, no hold reason" % tlz_s, len(crs) == 1 and crs[0]["branch"] == "adiabatic" and crs[0]["decided"] and not crs[0]["flagged"] and abs(crs[0]["tau_lz"] / tlz_s - 1.0) < 1e-2 and not Ps.get("mask_changed") and Ps.get("recall") is None and Ps.get("walk_consistent") and masks is not None and np.array_equal(masks, occ0c) and not V9.v9c_hold_reasons(dict(secs, min_occupied_weight=0.999)),
             "branch %s tau_LZ %s exponent %s mask kept %s hold %s" % (crs[0].get("branch") if crs else None, crs[0].get("tau_lz") if crs else None, crs[0].get("exponent") if crs else None, bool(np.array_equal(masks, occ0c)) if masks is not None else None, V9.v9c_hold_reasons(dict(secs, min_occupied_weight=0.999))))
        tau_k = asc / (gmc * gmc)
        lz(True, tau=tau_k)
        certk, seck, anchk, occpk = light(pathc, FX.S_host, e0c, sig0c, occ0c, trialc, n_seac, d_min, 4096, tol_block, mu_prev=mu0c)
        crk = seck.get("crossings") or []
        C.ok("F5 the knife edge (tau %.4g: tau_LZ 1): flagged knife-edge or undecided and held" % tau_k, len(crk) == 1 and (crk[0].get("knife") or crk[0].get("tie")) and crk[0]["flagged"] and bool(V9.v9c_hold_reasons(dict(seck, min_occupied_weight=0.999))), "tau_LZ %s knife %s tie %s flagged_why %s" % (crk[0].get("tau_lz") if crk else None, crk[0].get("knife") if crk else None, crk[0].get("tie") if crk else None, crk[0].get("flagged_why") if crk else None))
        # (d 1e-8): F4
        pathe, e0e, sig0e, occ0e, ref0e, n_seae, mu0e, ce, _gme, _ase = pair_path(V9, S9, chL, N0, v0, FX.sgn_dev, tol_block, g=1e-8, s_star=0.45)
        triale = pathe.eig(1.0); pathe.trial = triale; e1e, sig1e, U1e = triale
        lz(True)
        certe, sece, anche, occpe = light(pathe, FX.S_host, e0e, sig0e, occ0e, triale, n_seae, d_min, 4096, tol_block, mu_prev=mu0e)
        cre = sece.get("crossings") or []; Pe = sece.get("passage") or {}
        maske = mask_of(e1e, sig1e, anche)
        C.ok("F4 (d 1e-8) in lz: the walk's sub-floor crossing merged, diabatic, no re-call, the mask (c)'s, no hold reason", certe["certified"] and len(cre) == 1 and cre[0]["source"] == "walk+primitive" and cre[0]["branch"] == "diabatic" and cre[0]["swap"] is True and not Pe.get("mask_changed") and Pe.get("walk_consistent") and maske is not None and np.array_equal(maske, mask_c) and not V9.v9c_hold_reasons(dict(sece, min_occupied_weight=0.999)),
             "crossings %d source %s branch %s tau_LZ %s overlap %s (prim character %s) hold %s" % (len(cre), cre[0].get("source") if cre else None, cre[0].get("branch") if cre else None, cre[0].get("tau_lz") if cre else None, cre[0].get("overlap") if cre else None, cre[0].get("character_weight") if cre else None, V9.v9c_hold_reasons(dict(sece, min_occupied_weight=0.999))))
        # the adiabatic case at the fixture's rate (the constant coupling g 0.3, the window wider than the chord): F5 skipped / quiet
        pathq, e0q, sig0q, occ0q, ref0q, n_seaq, mu0q, cq, _gmq, _asq = pair_path(V9, S9, chL, N0, v0, FX.sgn_dev, tol_block, g=0.3, const=True)
        trialq = pathq.eig(1.0); pathq.trial = trialq
        lz(True)
        V9.V9C_CTX["light_ref"] = ref0q
        try:
            certq, secq, anchq, occpq = light(pathq, FX.S_host, e0q, sig0q, occ0q, trialq, n_seaq, d_min, 4096, tol_block, mu_prev=mu0q)
        finally:
            V9.V9C_CTX["light_ref"] = None
        Pq = secq.get("passage") or {}
        lz(True)
        certq2, secq2, anchq2, occpq2 = light(pathq, FX.S_host, e0q, sig0q, occ0q, trialq, n_seaq, d_min, 4096, tol_block, mu_prev=mu0q)
        Pq2 = secq2.get("passage") or {}
        C.ok("F5 gap 0.3 at the fixture's rate: skipped at the defaults with the previous reference stashed (no spectrum); quiet without it (the window wider than the chord; the set's columns at 0 and the point 1/2), the set kept, no entry", Pq.get("skipped") and Pq.get("n_spectra") == 0 and not (secq.get("crossings") or []) and np.array_equal(np.asarray(occpq, bool), np.asarray(occ0q, bool)[V9.light_plus_order_v9(e0q, sig0q)[0]]) and Pq2.get("quiet") and not (secq2.get("crossings") or []) and Pq2.get("n_spectra") == 2,
             "skipped %s spectra %s; quiet %s spectra %s" % (Pq.get("skipped"), Pq.get("n_spectra"), Pq2.get("quiet"), Pq2.get("n_spectra")))
        # the heavy: F7 (c) exact and (b) no crossing; F8 the avoided crossing
        h0h = V9.clocked_gpu(chH, N0, v0)
        pathh = S9.ManufacturedHeavyPath(V9, h0h, torch.zeros_like(h0h)); e0h, U0h = pathh.eig(0.0)
        n_seah = int(np.sum(e0h < 0.0)); occ0h = e0h < 0.0
        ua = U0h[:, n_seah - 1].clone(); ub = U0h[:, n_seah].clone(); ch_ = float(e0h[n_seah] - e0h[n_seah - 1])
        ref0h = U0h[:, torch.as_tensor(occ0h, device=U0h.device)].clone()
        pathh.D = (ch_ / 1.002) * (torch.outer(ua, ua.conj()) - torch.outer(ub, ub.conj()))   # the crossing at s* 0.501: off the dyadic points by 1e-3 (rounding-proof), the pair's ends inside the room its neighbours leave (0.0025 on this fixture)
        trialh = pathh.eig(1.0); pathh.trial = trialh; e1h, U1h = trialh
        lz(True)
        walkh, occh = heavy(pathh, ref0h, e0h, occ0h, trialh, n_seah, d_min, 4096, 1e-8)
        crh = walkh.get("crossings") or []; Ph = walkh.get("passage") or {}
        okh = walkh["certified"] and len(crh) == 1 and crh[0]["source"] == "walk+primitive" and crh[0]["branch"] == "diabatic" and crh[0]["swap"] is True and not Ph.get("mask_changed") and Ph.get("walk_consistent") and walkh["anchors_end"] == 3 and walkh["inversions_end"] == 1 and not V9.v9c_hold_reasons(dict(walkh, min_occupied_weight=0.999))
        C.ok("F7 heavy (c) exact crossing in lz: the walk's crossing merged, diabatic, no re-call, anchors 3, inversions 1, no hold reason", okh, "certified %s crossings %d source %s branch %s exponent %s spectra %s hold %s" % (walkh["certified"], len(crh), crh[0].get("source") if crh else None, crh[0].get("branch") if crh else None, crh[0].get("exponent") if crh else None, Ph.get("n_spectra"), V9.v9c_hold_reasons(dict(walkh, min_occupied_weight=0.999))))
        x = chL.R / float(len(chL.R)); dN = 2e-3 * np.exp(-((x - 0.45) / 0.15) ** 2); dv = -2e-3 * x * np.exp(-((x - 0.5) / 0.3) ** 2)
        Dbh = V9.clocked_gpu(chH, N0 + dN, v0 + dv) - h0h
        pathb = S9.ManufacturedHeavyPath(V9, h0h + pathh.D, Dbh); trialb = pathb.eig(1.0); pathb.trial = trialb
        refh = U1h[:, torch.as_tensor(occh, device=U1h.device)].clone()
        walkb, occb = heavy(pathb, refh, e1h, occh, trialb, n_seah, d_min, 4096, 1e-8)
        Pb = walkb.get("passage") or {}
        C.ok("F7 heavy (b) no crossing from the inverted state: skipped or quiet, the occupation unchanged, no entry", walkb["certified"] and not (walkb.get("crossings") or []) and (Pb.get("skipped") or Pb.get("quiet")) and np.array_equal(np.asarray(occb, bool), np.asarray(occh, bool)), "skipped %s quiet %s spectra %s" % (Pb.get("skipped"), Pb.get("quiet"), Pb.get("n_spectra")))
        h0a = h0h + 0.5e-3 * (torch.outer(ua, ub.conj()) + torch.outer(ub, ua.conj()))
        patha = S9.ManufacturedHeavyPath(V9, h0a, ch_ * (torch.outer(ua, ua.conj()) - torch.outer(ub, ub.conj()))); e0a, U0a = patha.eig(0.0)
        n_seaa = int(np.sum(e0a < 0.0)); occ0a = e0a < 0.0; ref0a = U0a[:, torch.as_tensor(occ0a, device=U0a.device)].clone()
        triala = patha.eig(1.0); patha.trial = triala; e1a, U1a = triala
        lz(False)
        walkaf, occaf = heavy(patha, ref0a, e0a, occ0a, triala, n_seaa, d_min, 4096, 1e-8)
        refaf, infoaf, maskaf = CERT.assign_occupation_v34(ref0a, torch.as_tensor(e1a, device=U1a.device), U1a, n_seaa, occaf)
        lz(True)
        walka, occa = heavy(patha, ref0a, e0a, occ0a, triala, n_seaa, d_min, 4096, 1e-8)
        cra = walka.get("crossings") or []; Pa = walka.get("passage") or {}
        refa, infoa, maska = CERT.assign_occupation_v34(ref0a, torch.as_tensor(e1a, device=U1a.device), U1a, n_seaa, occa)
        exp_a = math.pi * 1e-6 / (2.0 * ch_)
        oka = (n_seaa == n_seah and walkaf["certified"] and not (walkaf.get("crossings") or []) and np.array_equal(np.asarray(occaf, bool), occ0a) and infoaf.get("min_occupied_weight") is not None and infoaf["min_occupied_weight"] < 0.5
               and walka["certified"] and len(cra) == 1 and cra[0]["source"] == "primitive" and cra[0]["branch"] == "diabatic" and not cra[0]["flagged"] and abs(cra[0]["exponent"] / exp_a - 1.0) < 1e-2 and abs(cra[0]["gap_min"] / 1e-3 - 1.0) < 1e-3 and Pa.get("mask_changed") and Pa.get("recall") is not None and Pa.get("walk_consistent")
               and walka["anchors_end"] == 3 and walka["inversions_end"] == 1 and np.array_equal(np.asarray(occa, bool), np.asarray(occh, bool)) and infoa.get("min_occupied_weight") is not None and infoa["min_occupied_weight"] > 0.99 and not V9.v9c_hold_reasons(dict(walka, min_occupied_weight=infoa["min_occupied_weight"])))
        C.ok("F8 heavy avoided crossing of the constant gap 1e-3 (floor: the set kept, min_occupied_weight %s): lz decides diabatic (exponent %.3g), re-called at x_n, anchors 3, inversions 1, the occupation (c)'s, assign_occupation_v34's min_occupied_weight above 0.99, no hold reason" % (("%.1e" % infoaf["min_occupied_weight"]) if infoaf.get("min_occupied_weight") is not None else None, exp_a), oka,
             "crossings %d branch %s exponent %s alpha_T %s recall %s anchors %s inversions %s wmin %s spectra %s hold %s" % (len(cra), cra[0].get("branch") if cra else None, cra[0].get("exponent") if cra else None, cra[0].get("alpha_T") if cra else None, Pa.get("recall"), walka.get("anchors_end"), walka.get("inversions_end"), infoa.get("min_occupied_weight"), Pa.get("n_spectra"), V9.v9c_hold_reasons(dict(walka, min_occupied_weight=infoa.get("min_occupied_weight") or 0.0))))
        # F10 the callers on the fixture's own window and a small field change, lz against floor
        N1, v1 = N0 + dN, v0 + dv
        diffs = []; notes = []
        for chL_ in FX.chl:
            s_ = FX.st[chL_.key]
            e_r, U_r = V9.eigT(V9.clocked_gpu(chL_, N1, v1)); e_s, U_s = V9.eigT(V9.clocked_gpu(chL_, N1, v1))
            lz(False)
            ref_r, ci_r, m_r = V9.light_continue_v36(a, chL_, s_["window"], (N1, v1), e_r, U_r, s_["n_sea"], FX.sgn_dev, FX.S_host)
            lz(True)
            ref_s, ci_s, m_s = V9.light_continue_v36(a, chL_, s_["window"], (N1, v1), e_s, U_s, s_["n_sea"], FX.sgn_dev, FX.S_host)
            S8.deep_diff(ref_r, ref_s, "light %d ref" % chL_.kappa, diffs); S8.deep_diff(np.asarray(m_r, bool), np.asarray(m_s, bool), "light %d mask" % chL_.kappa, diffs)
            P_ = (ci_s.get("sector") or {}).get("passage") or {}
            S8.deep_diff(S9.strip_keys(ci_r, S9.STRIP_INFO | {"passage"}), S9.strip_keys(ci_s, S9.STRIP_INFO | {"passage"}), "light %d cinfo" % chL_.kappa, diffs)
            notes.append("light %d: route %s, passage %s (spectra %s)" % (chL_.kappa, (ci_s.get("sector") or {}).get("route"), "skipped" if P_.get("skipped") else ("quiet" if P_.get("quiet") else ("ran" if P_.get("ran") else "none")), P_.get("n_spectra")))
            if not (P_.get("skipped") or P_.get("quiet") or P_.get("ran")):
                diffs.append(("light %d passage block" % chL_.kappa, "missing", P_))
        for ch in FX.chs:
            s_ = FX.st[ch.kappa]
            e_r, U_r = V9.eigT(V9.clocked_gpu(ch, N1, v1))
            lz(False)
            ref_r, ci_r, m_r = V9.continue_ref_v9(a, ch, s_["window"], (N1, v1), e_r, U_r, s_["n_sea"])
            lz(True)
            ref_s, ci_s, m_s = V9.continue_ref_v9(a, ch, s_["window"], (N1, v1), e_r, U_r, s_["n_sea"])
            S8.deep_diff(ref_r, ref_s, "heavy %d ref" % ch.kappa, diffs); S8.deep_diff(np.asarray(m_r, bool), np.asarray(m_s, bool), "heavy %d mask" % ch.kappa, diffs)
            S8.deep_diff(S9.strip_keys(ci_r, S9.STRIP_INFO | {"passage"}), S9.strip_keys(ci_s, S9.STRIP_INFO | {"passage"}), "heavy %d cinfo" % ch.kappa, diffs)
            notes.append("heavy %d: route %s" % (ch.kappa, (ci_s.get("sector") or {}).get("route")))
        C.ok("F10 the callers (light_continue_v36, continue_ref_v9) lz against floor on the fixture's window: the reference and the mask bitwise, the entries equal but for the passage block", not diffs, "%d differences %s; %s" % (len(diffs), diffs[:3], "; ".join(notes)))
        # F11 (C64_v9c_review_v1 findings 2 and 9) the re-call on V9cEndPath over the real LightPathV9 through the real caller, and the extension's field convention on that path
        chL1 = FX.chl[0]; s1 = FX.st[chL1.key]
        e_fa, U_fa = V9.eigT(V9.clocked_gpu(chL1, N1, v1)); e_fb, U_fb = V9.eigT(V9.clocked_gpu(chL1, N1, v1))
        lz(False)
        ref_f, ci_f, m_f = V9.light_continue_v36(a, chL1, s1["window"], (N1, v1), e_fa, U_fa, s1["n_sea"], FX.sgn_dev, FX.S_host)
        caught = []; decide0 = V9.v9c_decide; Spec0 = V9.V9cSpectra

        class CapSpectra(Spec0):
            def __init__(self, path, *rest, **kw):
                Spec0.__init__(self, path, *rest, **kw); caught.append(self)

        def force_recall(species, channel, spec, mask0, mask_walk, crossings_walk, tau, floor=None, mode=None, ref0=None, say=None):
            m_out, info = decide0(species, channel, spec, mask0, mask_walk, crossings_walk, tau, floor=floor, mode=mode, ref0=ref0, say=say)
            if species == "light" and info is not None:
                info = dict(info); info["ran"] = True; info["mask_changed"] = True; info["flags"] = list(info.get("flags") or []) + ["F11: the re-call forced on the walk's own set (no passage on this chord)"]
                return np.asarray(mask_walk, bool).copy(), info
            return m_out, info
        V9.V9cSpectra = CapSpectra; V9.v9c_decide = force_recall
        try:
            lz(True)
            ref_l, ci_l, m_l = V9.light_continue_v36(a, chL1, s1["window"], (N1, v1), e_fb, U_fb, s1["n_sea"], FX.sgn_dev, FX.S_host)
        finally:
            V9.V9cSpectra = Spec0; V9.v9c_decide = decide0; lz(False)
        diffs = []
        S8.deep_diff(ref_f, ref_l, "light %d ref" % chL1.kappa, diffs); S8.deep_diff(np.asarray(m_f, bool), np.asarray(m_l, bool), "light %d mask" % chL1.kappa, diffs)
        S8.deep_diff(S9.strip_keys(ci_f, S9.STRIP_INFO | {"passage", "crossings_walk"}), S9.strip_keys(ci_l, S9.STRIP_INFO | {"passage", "crossings_walk"}), "light %d cinfo" % chL1.kappa, diffs)
        sec11 = (ci_l.get("sector") or {}); P11 = sec11.get("passage") or {}; R11 = P11.get("recall") or {}
        C.ok("F11 the re-call on V9cEndPath over the real LightPathV9 through light_continue_v36 (v9c_decide forced to return the walk's own set as a change): the reference, the mask and the entry bitwise against floor but for the passage block, which carries the re-call with the walk's own anchors_end and route",
             not diffs and P11.get("ran") and P11.get("mask_changed") is False and bool(R11) and R11.get("anchors_end") == sec11.get("anchors_end") and R11.get("route") == sec11.get("route") and len(caught) >= 1,
             "%d differences %s; passage ran %s mask_changed %s recall %s; route %s anchors_end %s; spectra objects %d" % (len(diffs), diffs[:3], P11.get("ran"), P11.get("mask_changed"), R11, sec11.get("route"), sec11.get("anchors_end"), len(caught)))
        ok_ext = False; det_ext = "no spectra object captured"
        if caught:
            spec11 = caught[0]; v11 = spec11._verify_extension()
            if v11 is True:
                e_h, sig_h, _U_h = spec11._extend(0.5); e_p, sig_p, _U_p = spec11.path.eig(0.5)
                e_h = np.asarray(e_h, float); e_p = np.asarray(e_p, float); dev11 = float(np.max(np.abs(e_h - e_p)))
                same_sect = bool(np.array_equal(np.asarray(sig_h, float) > 0.0, np.asarray(sig_p, float) > 0.0))
                ok_ext = dev11 <= 1e-8 * max(1.0, float(np.max(np.abs(e_p)))) and same_sect
                det_ext = "verified at s = 1 (the trial's eigenvalues reproduced); at s = 1/2 against path.eig the eigenvalues differ by %.2e and the sectors agree %s" % (dev11, same_sect)
            else:
                det_ext = "the verification at s = 1 refused: %s (post the LightPathV9 def line and its field attributes; the gates may run on the two-sample bound, the fix lands before S13)" % v11
        C.ok("F11b the extension's field convention on the real LightPathV9: V9cSpectra._extend(1) reproduces the trial's eigenvalues and _extend(1/2) path.eig(1/2)'s (the extrapolated fields are the chord's)", ok_ext, det_ext)
    except Exception:
        import traceback
        C.ok("F fixture: exception", False, traceback.format_exc())
    finally:
        lz(False)


def strip_seconds(d):
    return {k: v for k, v in d.items() if k != "seconds"}


def suite_check(C, a, out):
    """the v9b self-test's fixture with v9c in v9b's place (floor mode)"""
    work = tempfile.mkdtemp(prefix="v9c_suite_")
    try:
        code = os.path.join(work, "code")
        shutil.copytree(HERE, code, ignore=shutil.ignore_patterns("*.npz", "*.log", "*.pt", "*.json", "__pycache__", "results", "states", "g0*", "stage", "*.sh"))
        B.write(os.path.join(code, "engine_twofield_v9b.py"), out)
        shutil.copy(a.v9b_selftest, os.path.join(code, os.path.basename(a.v9b_selftest)))
        cmd = [sys.executable, os.path.basename(a.v9b_selftest), "fixture"] + (["--gpu", str(a.gpu)] if a.gpu is not None else [])
        env = dict(os.environ); env["PYTHONDONTWRITEBYTECODE"] = "1"
        t0 = time.time()
        try:
            r = subprocess.run(cmd, cwd=code, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, timeout=a.suite_timeout, env=env)
            lines = [l_ for l_ in r.stdout.split("\n") if l_.startswith("PASS") or l_.startswith("FAIL") or l_.startswith("SKIP") or l_.startswith("selftest")]
            C.ok("u the v9b self-test's fixture passes with v9c (floor) in v9b's place (%.0f s)" % (time.time() - t0), r.returncode == 0 and "selftest: 0 failure(s)" in r.stdout, "exit %d; %s" % (r.returncode, " | ".join(l_[:90] for l_ in lines)))
            if r.returncode != 0:
                print(r.stdout[-4000:])
        except subprocess.TimeoutExpired:
            C.ok("u the v9b self-test's fixture passes with v9c (floor) in v9b's place", False, "timeout after %d s" % a.suite_timeout)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description="G0'' of engine_twofield_v9c")
    ap.add_argument("--v9b", default=None, help="engine_twofield_v9b.py: the build, the marks, the reverse, the built engine's import and defaults")
    ap.add_argument("--v9c", default=None, help="engine_twofield_v9c.py as built: must equal the builder's output byte for byte; the fixture runs on it")
    ap.add_argument("--fixture", action="store_true", help="the real routes on the manufactured paths (needs --v9b; the engine's modules on BUSI_ENGINE_DIR)")
    ap.add_argument("--gpu", type=int, default=None, help="the fixture's device (default the CPU)")
    ap.add_argument("--v9b-selftest", default=None, help="engine_twofield_v9b_selftest.py: its fixture run with v9c in v9b's place in floor mode")
    ap.add_argument("--light-def", default=B.LIGHT_DEF_DEFAULT); ap.add_argument("--heavy-def", default=B.HEAVY_DEF_DEFAULT); ap.add_argument("--light-caller", default=B.LIGHT_CALLER_DEFAULT); ap.add_argument("--run-def", default=B.RUN_DEF_DEFAULT)
    ap.add_argument("--skip-torch", action="store_true"); ap.add_argument("--skip-suite", action="store_true"); ap.add_argument("--skip-layer", action="store_true")
    ap.add_argument("--suite-timeout", type=int, default=3600)
    ap.add_argument("--code-dir", default=HERE, help="where engine_twofield_v8_selftest.py and engine_twofield_v9b_selftest.py are imported from")
    a = ap.parse_args(argv)
    if a.code_dir and a.code_dir not in sys.path:
        sys.path.insert(0, a.code_dir)
    C = Checks()
    L = B.load_block_namespace(a.light_def, a.heavy_def, a.light_caller, a.run_def)
    C.ok("0 the layer loads from the builder's block; the primitive imports", L["v9c_prim"]().VERSION.startswith("assign_primitive"), L["v9c_prim"]().VERSION)
    if not a.skip_layer:
        run_layer_checks(C, L, skip_torch=a.skip_torch)
    if a.v9b:
        out = build_checks(C, a)
        if out is not None:
            tmp = tempfile.mkdtemp(prefix="v9c_selftest_")
            try:
                path = a.v9c if a.v9c else os.path.join(tmp, "engine_twofield_v9c.py")
                if not a.v9c:
                    B.write(path, out)
                try:
                    mod = import_built(path)
                    import_checks(C, a, mod)
                except Exception as exc:
                    import traceback
                    C.ok("t built engine imports", False, "%s: %s\n%s" % (type(exc).__name__, exc, traceback.format_exc()))
                    mod = None
                if a.fixture and mod is not None:
                    import engine_twofield_v8_selftest as S8
                    import engine_twofield_v9b_selftest as S9
                    run_fixture_checks(C, mod, S8, S9, a.gpu)
                elif a.fixture:
                    C.ok("F fixture", False, "the built engine did not import")
                if a.skip_suite or not a.v9b_selftest:
                    print("SKIP u the v9b self-test in v9b's place (%s)" % ("--skip-suite" if a.skip_suite else "no --v9b-selftest"))
                else:
                    suite_check(C, a, out)
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
    else:
        print("SKIP s, t, F, u (no --v9b given): the layer checks only")
    failed = C.failed()
    if failed:
        print("engine_twofield_v9c_selftest: FAIL (%d of %d checks): %s" % (len(failed), len(C.rows), "; ".join(failed)))
        return 1
    print("engine_twofield_v9c_selftest: PASS (%d checks)" % len(C.rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
