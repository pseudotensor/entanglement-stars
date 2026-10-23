#!/usr/bin/env python
"""engine_twofield_v9c_selftest.py (reviewer A for the busi lane; C64_v9c_ruling_v1 2(c): G0''; C64_G4s8_hold_ruling_v1 (4b); C64_transport_ruling_v1 4.4).

The layer of v9c (the block of build_engine_twofield_v9c.py) is tested on its own with the fixtures below; the built engine, when v9b is given, for
the build, the marks, the reverse, its defaults and the v9b suite in v9b's place. Every check prints PASS or FAIL with its detail; the last line is the
verdict "engine_twofield_v9c_selftest: PASS (n checks)" or FAIL with the failing names (exit 1).

The fixtures (h(s) linear in s, so K(s) = (1 - s) h0 + s h1 is h(s) and the two-level fit is exact). heavy-like: eight levels 0 .. 7, the member
(basis 3) rising from e_m0 at the rate slope per chord through the empty partner (basis 4, energy 4), a coupling V between them (gap 2 V), the set
the four lowest. light-like: eight levels symmetric under e <-> -e (the charge-conjugation mirror), the member -0.9 + 1.8 s meeting its own mirror
0.9 - 1.8 s at e = 0 with a coupling V, the set the four negative levels. The record's tau is the chord's physical length (alpha_T = alpha_s / tau).
  a  the exact crossing, no mirror: one diabatic event (gap 0 to the fit), the set swapped, inversions 1, the jump zero and falling
  b  the exact crossing with the mirror: self-mirror, counted once, the set swapped
  c  tau 20: gap 0.1 diabatic (tau_LZ 0.11), gap 0.4 adiabatic (1.78, the set kept, jump_alt recorded), gap 0.3 the knife edge (tau_LZ 1: undecided, held);
     the same three on the light fixture with gap 0.1, 0.6, 0.6 at tau 20 and tau 3 (the last the knife edge: tau_LZ 2.0 against the sweep's 0.3)
  d  4.4's fixture (d) moved to the primitive: tau 2, gap 1e-3 and 1e-8 diabatic; at the HOLD ruling's rate (alpha_T 0.0456 at tau 2) gap 1e-3 is
     diabatic with the exponent 3.4e-5 and gap 0.3 is quiet (its window wider than the chord: the adiabatic continuation, no entry)
  e  the inversion removed on a second record (the dived mode rising back: diabatic, delta -1, inversions 0)
  f  a block (four passages of one member in one chord): unresolved at level 2 and held; resolved at level 4 (four diabatic events, stable)
  g  quiet: no passage, one spectrum, the walk's mask object returned
  h  skip: wide quiet boundary pairs, no spectrum
  i  a passage straddling two records: the first chord an approach (an entry without a decision, so wmin below 0.9 does not hold), the second the
     incomplete passage decided diabatic and the set swapped
  j  the G4s8 record form through the wrapper: the walk's adiabatic set at wmin below 0.01 replaced by the decided set, the route re-called once at x_n,
     wmin recomputed above 0.99, adiabatic_at_cut 0, no hold reason
  k  floor mode through the wrapper: the route's return returned untouched (the same object), one call, nothing computed
  l  the walk's own crossing merged with the event (no re-call); a bogus walk crossing flagged and holding
  m  the arguments: the defaults equal V9C_DEFAULTS and V9C_MODE is untouched by a default parse; --anchor-passage lz and --lz-c 2 reach V9C_MODE
  n  tau_LZ halves from tau 2 to tau 1 on the same chord
  o  the torch oracle (when torch is present) agrees with numpy on decisions and masks
  p  the exponent convention: exponent = pi tau_LZ / (2 k); with k = pi/2 the two coincide
  q  determinism: two runs give identical entries
  r  v9c_hold_reasons on G4s8's 13.125 form (crossings empty, wmin 0.0027) names the unresolved passage; a walk overlap 0.8365 holds by 4.6
  with --v9b: s the build, the marks and the reverse round trip (and equality with --v9c when given); t the built engine imports with V9C_MODE equal to
  V9C_DEFAULTS, both routes wrapped, V9C_MISSING empty; u the v9b self-test passes with v9c in v9b's place in floor mode (--v9b-selftest; a copy of the
  code directory is made for it)

Usage: python engine_twofield_v9c_selftest.py [--v9b engine_twofield_v9b.py] [--v9c engine_twofield_v9c.py] [--v9b-selftest engine_twofield_v9b_selftest.py]
          [--light-def NAME] [--heavy-def NAME] [--skip-torch] [--skip-suite] [--suite-timeout SECONDS]
"""
import argparse
import functools
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
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


def eigh(H):
    return np.linalg.eigh(np.asarray(H))


def idx_mask(N, occ):
    m = np.zeros(N, bool); m[list(occ)] = True
    return m


def heavy_h(V, slope=1.8, e_m0=3.0, member=3, partner=4, N=8):
    def h(s):
        E = np.arange(N, dtype=float); E[member] = e_m0 + slope * s
        H = np.diag(E); H[member, partner] = V; H[partner, member] = V
        return H
    return h


def light_h(V, slope=1.8):
    def h(s):
        E = np.array([-4.0, -3.0, -2.0, -0.9 + slope * s, 0.9 - slope * s, 2.0, 3.0, 4.0])
        H = np.diag(E); H[3, 4] = V; H[4, 3] = V
        return H
    return h


def chord(h, s0=0.0, s1=1.0):
    h0, h1 = h(s0), h(s1)
    e0, U0 = eigh(h0); e1, U1 = eigh(h1)
    return h0, h1, e0, U0, e1, U1


def lz_mode(L, **over):
    M = dict(L["V9C_DEFAULTS"]); M["anchor_passage"] = "lz"; M.update(over)
    return M


def decide(L, species, h, mask0, tau, mode, mask_walk=None, crossings=None, channel=1, s0=0.0, s1=1.0, h0h1=None):
    if h0h1 is None:
        h0, h1, e0, U0, e1, U1 = chord(h, s0, s1)
    else:
        h0, h1 = h0h1; e0, U0 = eigh(h0); e1, U1 = eigh(h1)
    mw = mask0.copy() if mask_walk is None else mask_walk
    return L["v9c_decide"](species, channel, h0, h1, e0, U0, e1, U1, mask0, mw, crossings or [], tau, mode=mode, say=False)


def entries_of(info, kinds=("passage", "incomplete", "block")):
    return [c for c in info["entries"] if c.get("kind") in kinds]


def toy_walk(h0, h1, e0, U0, e1, U1, mask0, Vprev, anchors0, tau, args, kappa, floor, crossings=(), mask_end=None, calls=None):
    """v9b's certified continuation in miniature: the set's indices kept (or the given end mask: a sub-floor swap), the reference the eigenvectors at
    x_n, the entry with the walk's counters, wmin against the previous reference and the cut check's count"""
    if calls is not None:
        calls.append(1)
    m = np.asarray(mask0, bool).copy() if mask_end is None else np.asarray(mask_end, bool).copy()
    occ = np.flatnonzero(m); V1 = U1[:, occ]
    p = (np.abs(np.asarray(Vprev).conj().T @ V1) ** 2).sum(axis=0)
    ent = {"route": "sector walk", "pieces": 3, "cholesky": 9, "anchors_start": len(anchors0), "anchors_end": int(np.count_nonzero(m[:-1] != m[1:])),
           "certified_fraction": 1.0, "crossings": [dict(c) for c in crossings], "pockets": 1, "inversions_sector": 0, "min_occupied_weight": float(p.min()),
           "adiabatic_at_cut": int(np.count_nonzero(p < 0.5)), "gap": float(e1[4] - e1[3]), "d_min": 1e-7, "swaps": len(crossings)}
    return V1, m, ent


def run_layer_checks(C, L, skip_torch=False):
    MASK = idx_mask(8, [0, 1, 2, 3]); SWAP = idx_mask(8, [0, 1, 2, 4])
    # a. the exact crossing, no mirror
    m_out, info = decide(L, "heavy", heavy_h(0.0), MASK, 2.0, lz_mode(L))
    ev = entries_of(info)
    C.ok("a exact crossing: one diabatic event", info["ran"] and len(ev) == 1 and ev[0]["branch"] == "diabatic" and ev[0]["decided"], "%d events, %s" % (len(ev), ev[0]["branch"] if ev else "-"))
    C.ok("a exact crossing: the fit", ev and abs(ev[0]["s_star"] - 1.0 / 1.8) < 1e-6 and ev[0]["gap_min"] < 1e-6 and abs(ev[0]["alpha_s"] - 1.8) < 1e-6, "s* %s gap_min %s alpha_s %s" % (ev[0]["s_star"] if ev else None, ev[0]["gap_min"] if ev else None, ev[0]["alpha_s"] if ev else None))
    C.ok("a exact crossing: the set swapped, inversions 1, stable, jumps fall", np.array_equal(m_out, SWAP) and info["inversions_end"] == 1 and info["stable"] and info["jumps_fall"] and ev and ev[0]["delta"] == 1 and abs(ev[0]["jump_fit"]) < 1e-9 and ev[0]["falls"] is not False, "mask %s inv %s stable %s falls %s" % (m_out.astype(int).tolist(), info["inversions_end"], info["stable"], info["jumps_fall"]))
    C.ok("a exact crossing: no hold reason", not L["v9c_hold_reasons"]({"crossings": info["entries"], "min_occupied_weight": 0.999, "passage": {k: v for k, v in info.items() if k != "entries"}}))
    # b. the exact crossing with the mirror
    m_out, info = decide(L, "light", light_h(0.0), MASK, 2.0, lz_mode(L))
    ev = entries_of(info)
    C.ok("b mirror exact crossing: self-mirror, counted once, diabatic, the set swapped", len(ev) == 1 and ev[0]["self_mirror"] and ev[0]["branch"] == "diabatic" and np.array_equal(m_out, SWAP) and abs(ev[0]["s_star"] - 0.5) < 1e-6, "%d events self_mirror %s s* %s" % (len(ev), ev[0]["self_mirror"] if ev else None, ev[0]["s_star"] if ev else None))
    # c. the three branches at tau 20 (heavy) and the light fixture
    for label, h, mask, tau, expect, A_expect in (("heavy gap 0.1", heavy_h(0.05), MASK, 20.0, "diabatic", 0.1111), ("heavy gap 0.4", heavy_h(0.2), MASK, 20.0, "adiabatic", 1.7778),
                                                 ("light gap 0.1", light_h(0.05), MASK, 20.0, "diabatic", 0.0556), ("light gap 0.6", light_h(0.3), MASK, 20.0, "adiabatic", 2.0)):
        m_out, info = decide(L, "heavy" if label.startswith("heavy") else "light", h, mask, tau, lz_mode(L))
        ev = entries_of(info)
        got = ev[0] if ev else {}
        want_mask = SWAP if expect == "diabatic" else MASK
        C.ok("c %s: %s" % (label, expect), len(ev) == 1 and got.get("branch") == expect and got.get("decided") and not got.get("flagged") and np.array_equal(m_out, want_mask) and abs(got.get("tau_lz", 0.0) - A_expect) < 2e-3 and info["stable"],
             "branch %s tau_LZ %s mask %s stable %s" % (got.get("branch"), got.get("tau_lz"), m_out.astype(int).tolist(), info["stable"]))
        if expect == "adiabatic":
            C.ok("c %s: jump_alt recorded, jump 0" % label, got.get("jump_alt") is not None and abs(got.get("jump_alt")) > 0.1 and abs(got.get("jump_grid") or 0.0) < 1e-12, "jump_alt %s" % got.get("jump_alt"))
    m_out, info = decide(L, "heavy", heavy_h(0.15), MASK, 20.0, lz_mode(L))
    ev = entries_of(info)
    C.ok("c heavy gap 0.3: the knife edge undecided and held", len(ev) == 1 and ev[0]["tie"] and not ev[0]["decided"] and ev[0]["flagged"] and info["unresolved"] and bool(L["v9c_hold_reasons"]({"crossings": info["entries"], "min_occupied_weight": 0.5, "passage": {k: v for k, v in info.items() if k != "entries"}})),
         "tau_LZ %s tie %s decided %s" % (ev[0]["tau_lz"] if ev else None, ev[0]["tie"] if ev else None, ev[0]["decided"] if ev else None))
    m_out, info = decide(L, "light", light_h(0.3), MASK, 3.0, lz_mode(L))
    ev = entries_of(info)
    C.ok("c light gap 0.6 at tau 3: diabatic (tau_LZ 0.3)", len(ev) == 1 and ev[0]["branch"] == "diabatic" and ev[0]["decided"] and abs(ev[0]["tau_lz"] - 0.3) < 2e-3 and np.array_equal(m_out, SWAP), "tau_LZ %s" % (ev[0]["tau_lz"] if ev else None))
    # d. 4.4's fixture (d) moved to the primitive
    for g, label in ((1e-3, "1e-3"), (1e-8, "1e-8")):
        m_out, info = decide(L, "heavy", heavy_h(0.5 * g), MASK, 2.0, lz_mode(L))
        ev = entries_of(info)
        C.ok("d gap %s at tau 2: diabatic, decided, unflagged, the set swapped" % label, len(ev) == 1 and ev[0]["branch"] == "diabatic" and ev[0]["decided"] and not ev[0]["flagged"] and np.array_equal(m_out, SWAP), "branch %s tau_LZ %s flags %s" % (ev[0]["branch"] if ev else None, ev[0]["tau_lz"] if ev else None, ev[0]["flags"] if ev else None))
    m_out, info = decide(L, "heavy", heavy_h(5e-4, slope=0.0912, e_m0=3.955), MASK, 2.0, lz_mode(L))
    ev = entries_of(info)
    C.ok("d gap 1e-3 at alpha_T 0.0456: diabatic with the exponent 3.4e-5", len(ev) == 1 and ev[0]["branch"] == "diabatic" and abs(ev[0]["exponent"] / 3.4443e-5 - 1.0) < 1e-2 and abs(ev[0]["gap_min"] - 1e-3) < 1e-6, "exponent %s gap_min %s alpha_T %s" % (ev[0]["exponent"] if ev else None, ev[0]["gap_min"] if ev else None, ev[0]["alpha_T"] if ev else None))
    m_out, info = decide(L, "heavy", heavy_h(0.15, slope=0.0912, e_m0=3.955), MASK, 2.0, lz_mode(L))
    C.ok("d gap 0.3 at alpha_T 0.0456: quiet (the window wider than the chord), the set kept", info["quiet"] and not info["ran"] and m_out is not None and np.array_equal(m_out, MASK) and info["n_spectra"] == 1, "quiet %s spectra %d" % (info["quiet"], info["n_spectra"]))
    # e. the inversion removed
    m_out, info = decide(L, "heavy", heavy_h(0.0, slope=-1.8, e_m0=4.8), SWAP, 2.0, lz_mode(L))
    ev = entries_of(info)
    C.ok("e the inversion removed: diabatic, delta -1, inversions 0", len(ev) == 1 and ev[0]["branch"] == "diabatic" and ev[0]["delta"] == -1 and np.array_equal(m_out, MASK) and info["inversions_end"] == 0, "delta %s inv %s" % (ev[0]["delta"] if ev else None, info["inversions_end"]))
    # f. a block
    m_out, info = decide(L, "heavy", heavy_h(0.0, slope=10.0), MASK, 2.0, lz_mode(L, prim_levels=2))
    C.ok("f block at level 2: unresolved and held", info["ran"] and not info["stable"] and info["unresolved"] and bool(L["v9c_hold_reasons"]({"crossings": info["entries"], "min_occupied_weight": 0.5, "passage": {k: v for k, v in info.items() if k != "entries"}})), "stable %s counts %s" % (info["stable"], info["counts"]))
    m_out, info = decide(L, "heavy", heavy_h(0.0, slope=10.0), MASK, 2.0, lz_mode(L, prim_levels=4))
    ev = entries_of(info)
    C.ok("f block at level 4: four diabatic events, stable, the set at index 7", info["stable"] and info["counts"]["diabatic"] == 4 and len(ev) == 4 and np.array_equal(m_out, idx_mask(8, [0, 1, 2, 7])) and info["inversions_end"] == 1, "stable %s counts %s mask %s" % (info["stable"], info["counts"], m_out.astype(int).tolist()))
    # g. quiet; h. skip
    mw = MASK.copy()
    m_out, info = decide(L, "heavy", heavy_h(0.0, slope=0.5), MASK, 2.0, lz_mode(L), mask_walk=mw)
    C.ok("g quiet: no passage, one spectrum, the walk's mask object returned", info["quiet"] and info["n_spectra"] == 1 and m_out is mw, "quiet %s spectra %d same object %s" % (info["quiet"], info["n_spectra"], m_out is mw))
    m_out, info = decide(L, "heavy", heavy_h(0.0, slope=0.1), MASK, 2.0, lz_mode(L), mask_walk=mw)
    C.ok("h skip: wide quiet boundary pairs, no spectrum", info["skipped"] and info["n_spectra"] == 0 and m_out is mw, "skipped %s spectra %d" % (info["skipped"], info["n_spectra"]))
    # i. a passage straddling two records
    h = heavy_h(0.05, slope=0.95)
    m1, info1 = decide(L, "heavy", h, MASK, 20.0, lz_mode(L))
    appr = [c for c in info1["entries"] if c.get("kind") == "approach"]
    h0, h1 = h(1.0), heavy_h(0.05, slope=0.95, e_m0=3.95)(1.0)
    e1, U1 = eigh(h0)
    p_end = (np.abs(U1[:, :4].conj().T @ eigh(h(0.0))[1][:, :4]) ** 2).sum(axis=1).min()
    C.ok("i first chord: an approach entry, the set kept, stable, wmin below 0.9 with an entry of its own (no hold)", info1["ran"] and len(appr) == 1 and not entries_of(info1) and np.array_equal(m1, MASK) and info1["stable"] and p_end < 0.9 and not L["v9c_hold_reasons"]({"crossings": info1["entries"], "min_occupied_weight": float(p_end), "passage": {k: v for k, v in info1.items() if k != "entries"}}),
         "approaches %d wmin %.3f s* %s" % (len(appr), p_end, appr[0]["s"] if appr else None))
    m2, info2 = decide(L, "heavy", None, MASK, 20.0, lz_mode(L), h0h1=(h0, h1))
    ev = entries_of(info2)
    C.ok("i second chord: the incomplete passage decided diabatic, the set swapped", len(ev) == 1 and ev[0]["kind"] == "incomplete" and ev[0]["branch"] == "diabatic" and ev[0]["decided"] and np.array_equal(m2, SWAP) and info2["stable"] and abs(ev[0]["s_star"] - 0.05 / 0.95) < 2e-2,
         "kind %s branch %s s* %s mask %s" % (ev[0]["kind"] if ev else None, ev[0]["branch"] if ev else None, ev[0]["s_star"] if ev else None, m2.astype(int).tolist()))
    # j. the G4s8 record form through the wrapper; k. floor mode
    h0, h1, e0, U0, e1, U1 = chord(heavy_h(0.05))
    calls = []
    W = L["v9c_wrap_walk"]("light", functools.partial(toy_walk, calls=calls))
    L["V9C_MODE"]["anchor_passage"] = "lz"
    try:
        V1, m, ent = W(h0, h1, e0, U0, e1, U1, MASK, U0[:, :4], L["v9c_anchors"](e0, MASK), 20.0, None, 1, 3.5e-7)
    finally:
        L["V9C_MODE"]["anchor_passage"] = "floor"
    C.ok("j G4s8 form: the decided set, the route re-called once, the reference at x_n of the decided set", np.array_equal(m, SWAP) and len(calls) == 2 and np.allclose(V1, U1[:, [0, 1, 2, 4]]) and ent["passage"]["mask_changed"], "mask %s calls %d" % (m.astype(int).tolist(), len(calls)))
    C.ok("j G4s8 form: wmin recomputed above 0.99 from the walk's below 0.01, adiabatic_at_cut 0, no hold reason", ent["min_occupied_weight"] > 0.99 and ent["min_occupied_weight_walk"] < 0.01 and ent["adiabatic_at_cut"] == 0 and ent["adiabatic_at_cut_walk"] >= 1 and not L["v9c_hold_reasons"](ent),
         "wmin %.4f (walk %.4f) adiabatic_at_cut %d (walk %d) reasons %s" % (ent["min_occupied_weight"], ent["min_occupied_weight_walk"], ent["adiabatic_at_cut"], ent["adiabatic_at_cut_walk"], L["v9c_hold_reasons"](ent)))
    C.ok("j G4s8 form: the entry diabatic with its exponent and the primitive's version", ent["crossings"] and ent["crossings"][0]["branch"] == "diabatic" and ent["crossings"][0]["exponent"] is not None and ent["passage"]["prim"].startswith("assign_primitive"), "%s" % (ent["passage"].get("prim")))
    calls = []
    W = L["v9c_wrap_walk"]("light", functools.partial(toy_walk, calls=calls))
    out0 = W(h0, h1, e0, U0, e1, U1, MASK, U0[:, :4], L["v9c_anchors"](e0, MASK), 20.0, None, 1, 3.5e-7)
    C.ok("k floor mode: the route's return untouched, one call", isinstance(out0, tuple) and len(calls) == 1 and "passage" not in out0[2] and np.array_equal(out0[1], MASK), "calls %d keys %s" % (len(calls), sorted(out0[2].keys())[:4]))
    # l. the walk's crossing merged; a bogus one flagged
    h0, h1, e0, U0, e1, U1 = chord(heavy_h(0.0))
    walk_entry = {"s": 1.0 / 1.8, "piece": [0.5553, 0.5559], "probes": 6, "overlap": 0.99995, "delta": 1, "block": 2, "swap": True, "flagged": False}
    calls = []
    W = L["v9c_wrap_walk"]("light", functools.partial(toy_walk, crossings=[walk_entry], mask_end=SWAP, calls=calls))
    L["V9C_MODE"]["anchor_passage"] = "lz"
    try:
        V1, m, ent = W(h0, h1, e0, U0, e1, U1, MASK, U0[:, :4], L["v9c_anchors"](e0, MASK), 2.0, None, 1, 3.5e-7)
        calls2 = []
        W2 = L["v9c_wrap_walk"]("light", functools.partial(toy_walk, crossings=[dict(walk_entry, s=0.9)], mask_end=SWAP, calls=calls2))
        V1b, mb, entb = W2(h0, h1, e0, U0, e1, U1, MASK, U0[:, :4], L["v9c_anchors"](e0, MASK), 2.0, None, 1, 3.5e-7)
    finally:
        L["V9C_MODE"]["anchor_passage"] = "floor"
    C.ok("l walk crossing merged: source walk+primitive, no re-call, no hold reason", len(calls) == 1 and ent["crossings"] and ent["crossings"][0]["source"] == "walk+primitive" and ent["crossings"][0]["branch"] == "diabatic" and ent["passage"]["walk_consistent"] and not L["v9c_hold_reasons"](ent) and np.array_equal(m, SWAP),
         "source %s reasons %s" % (ent["crossings"][0]["source"] if ent["crossings"] else None, L["v9c_hold_reasons"](ent)))
    C.ok("l bogus walk crossing: flagged, inconsistent, holding", not entb["passage"]["walk_consistent"] and any(c.get("source") == "walk" and c.get("flagged") for c in entb["crossings"]) and bool(L["v9c_hold_reasons"](entb)), "reasons %s" % [r[0] for r in L["v9c_hold_reasons"](entb)])
    # m. the arguments
    ap = argparse.ArgumentParser(); L["v9c_add_arguments"](ap)
    before = dict(L["V9C_MODE"]); ns = ap.parse_args([])
    same_defaults = all(getattr(ns, k) == v for k, v in L["V9C_DEFAULTS"].items()) and L["V9C_MODE"] == before == L["V9C_DEFAULTS"]
    ns2 = ap.parse_args(["--anchor-passage", "lz", "--lz-c", "2.0", "--prim-levels", "3"])
    reached = L["V9C_MODE"]["anchor_passage"] == "lz" and L["V9C_MODE"]["lz_c"] == 2.0 and L["V9C_MODE"]["prim_levels"] == 3 and ns2.anchor_passage == "lz"
    L["V9C_MODE"].clear(); L["V9C_MODE"].update(L["V9C_DEFAULTS"])
    C.ok("m arguments: the defaults equal V9C_DEFAULTS and leave V9C_MODE untouched; lz and lz_c 2 reach V9C_MODE", same_defaults and reached, "defaults %s reached %s" % (same_defaults, reached))
    # n. tau_LZ halves with tau
    _, i2 = decide(L, "heavy", heavy_h(5e-4), MASK, 2.0, lz_mode(L)); _, i1 = decide(L, "heavy", heavy_h(5e-4), MASK, 1.0, lz_mode(L))
    A2 = entries_of(i2)[0]["tau_lz"] if entries_of(i2) else None; A1 = entries_of(i1)[0]["tau_lz"] if entries_of(i1) else None
    C.ok("n tau_LZ at tau 1 is half of tau 2's", A1 is not None and A2 is not None and abs(A1 / A2 - 0.5) < 1e-6, "A(tau 1) %s A(tau 2) %s" % (A1, A2))
    # o. the torch oracle
    if skip_torch:
        print("SKIP o torch oracle (--skip-torch)")
    else:
        try:
            import torch
            h0, h1, e0, U0, e1, U1 = chord(heavy_h(0.05))
            t = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float64)
            et0, Ut0 = torch.linalg.eigh(t(h0)); et1, Ut1 = torch.linalg.eigh(t(h1))
            m_t, info_t = L["v9c_decide"]("heavy", 1, t(h0), t(h1), et0, Ut0, et1, Ut1, MASK, MASK.copy(), [], 20.0, mode=lz_mode(L), say=False)
            m_n, info_n = decide(L, "heavy", heavy_h(0.05), MASK, 20.0, lz_mode(L))
            C.ok("o torch oracle agrees with numpy", np.array_equal(np.asarray(m_t), m_n) and entries_of(info_t)[0]["branch"] == entries_of(info_n)[0]["branch"] and abs(entries_of(info_t)[0]["tau_lz"] - entries_of(info_n)[0]["tau_lz"]) < 1e-6, "tau_LZ torch %s numpy %s" % (entries_of(info_t)[0]["tau_lz"], entries_of(info_n)[0]["tau_lz"]))
        except ImportError:
            print("SKIP o torch oracle (torch not present)")
    # p. the exponent convention
    _, ia = decide(L, "heavy", heavy_h(0.05), MASK, 20.0, lz_mode(L)); _, ib = decide(L, "heavy", heavy_h(0.05), MASK, 20.0, lz_mode(L, lz_k=math.pi / 2.0))
    ea = entries_of(ia)[0]; eb = entries_of(ib)[0]
    C.ok("p exponent = pi tau_LZ / (2 k); with k = pi/2 they coincide", abs(ea["exponent"] - math.pi / 2.0 * ea["tau_lz"]) < 1e-12 and abs(eb["exponent"] - eb["tau_lz"]) < 1e-12 and abs(eb["tau_lz"] - math.pi / 2.0 * ea["tau_lz"]) < 1e-9, "k 1: %s / %s; k pi/2: %s / %s" % (ea["tau_lz"], ea["exponent"], eb["tau_lz"], eb["exponent"]))
    # q. determinism
    _, ic = decide(L, "light", light_h(0.05), MASK, 20.0, lz_mode(L)); _, idd = decide(L, "light", light_h(0.05), MASK, 20.0, lz_mode(L))
    C.ok("q determinism: identical entries", json.dumps(ic["entries"], sort_keys=True) == json.dumps(idd["entries"], sort_keys=True))
    # r. v9c_hold_reasons on G4s8's forms
    r1 = L["v9c_hold_reasons"]({"route": "sector walk", "crossings": [], "min_occupied_weight": 0.0027})
    r2 = L["v9c_hold_reasons"]({"route": "sector walk", "crossings": [{"s": 0.9519, "overlap": 0.8365, "flagged": False, "source": "walk", "swap": False}], "min_occupied_weight": 0.9999})
    C.ok("r hold reasons: 13.125's form names the unresolved passage; a walk overlap 0.8365 holds by 4.6", len(r1) == 1 and r1[0][0].startswith("unresolved passage") and len(r2) == 1 and r2[0][0].startswith("passage overlap"), "%s | %s" % ([r[0] for r in r1], [r[0] for r in r2]))


def run_engine_checks(C, a):
    src = B.read(a.v9b)
    reps = B.replacements(a.light_def, a.heavy_def)
    bad = B.precheck(src, reps)
    heavy_ok = src.count("def %s(" % a.heavy_def) == 1
    if not C.ok("s build: every hard anchor occurs once in v9b", not bad and heavy_ok, "; ".join("%s occurs %d" % (r.name, n) for r, n in bad) + ("" if heavy_ok else "; heavy def %r occurs %d" % (a.heavy_def, src.count("def %s(" % a.heavy_def)))):
        print("  (python build_engine_twofield_v9c.py %s --probe lists the candidate lines; the lane posts them)" % a.v9b)
        return
    out, log = B.apply(src, reps)
    marks = B.marks_check(src, out)
    back, _ = B.apply(out, reps, reverse=True)
    C.ok("s build: the marks (every new line carries v9c prim) and the reverse round trip byte for byte", not marks and back == src, "%d unmarked; reverse %s" % (len(marks), "ok" if back == src else "differs"))
    if a.v9c:
        C.ok("s build: equals the given engine_twofield_v9c.py byte for byte", B.read(a.v9c) == out, "%s" % a.v9c)
    tmp = tempfile.mkdtemp(prefix="v9c_selftest_")
    try:
        path = os.path.join(tmp, "engine_twofield_v9c_built.py")
        B.write(path, out)
        sys.path.insert(0, HERE)
        spec = importlib.util.spec_from_file_location("engine_twofield_v9c_built", path)
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
            light = getattr(mod, a.light_def, None); heavy = getattr(mod, a.heavy_def, None)
            C.ok("t built engine: V9C_MODE equals V9C_DEFAULTS at import, both routes wrapped, V9C_MISSING empty",
                 mod.V9C_MODE == mod.V9C_DEFAULTS and hasattr(light, "__wrapped__") and hasattr(heavy, "__wrapped__") and not mod.V9C_MISSING,
                 "missing %s wrapped light %s heavy %s" % (mod.V9C_MISSING, hasattr(light, "__wrapped__"), hasattr(heavy, "__wrapped__")))
            C.ok("t built engine: the version constant", getattr(mod, "ENGINE_VERSION", "engine_twofield_v9c").endswith("v9c") and mod.V9C_VERSION == "engine_twofield_v9c", "%s" % getattr(mod, "ENGINE_VERSION", "(no ENGINE_VERSION; soft)"))
            for sp, fn in (("light", light), ("heavy", heavy)):
                if fn is None:
                    continue
                import inspect
                params = set(inspect.signature(fn.__wrapped__).parameters)
                need = [mod.V9C_NAMES[sp][k] for k in mod.V9C_REQUIRED]
                C.ok("t built engine: the %s route binds the required names %s" % (sp, need), all(n in params for n in need), "signature %s" % str(inspect.signature(fn.__wrapped__)))
        except Exception as exc:
            C.ok("t built engine imports", False, "%s: %s" % (type(exc).__name__, exc))
        if a.skip_suite or not a.v9b_selftest:
            print("SKIP u the v9b self-test in v9b's place (%s)" % ("--skip-suite" if a.skip_suite else "no --v9b-selftest"))
        else:
            work = os.path.join(tmp, "suite")
            shutil.copytree(HERE, work, ignore=shutil.ignore_patterns("*.npz", "*.json", "*.log", "__pycache__", "results", "states", "gate", "logs", "*.pt"))
            B.write(os.path.join(work, "engine_twofield_v9b.py"), out)
            shutil.copy(a.v9b_selftest, os.path.join(work, os.path.basename(a.v9b_selftest)))
            try:
                r = subprocess.run([sys.executable, os.path.basename(a.v9b_selftest)], cwd=work, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, timeout=a.suite_timeout)
                tail = "\n".join(r.stdout.strip().split("\n")[-5:])
                C.ok("u the v9b self-test passes with v9c (floor) in v9b's place", r.returncode == 0 and "FAIL" not in r.stdout, "exit %d; last lines:\n%s" % (r.returncode, tail))
            except subprocess.TimeoutExpired:
                C.ok("u the v9b self-test passes with v9c (floor) in v9b's place", False, "timeout after %d s" % a.suite_timeout)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description="G0'' of engine_twofield_v9c")
    ap.add_argument("--v9b", default=None, help="engine_twofield_v9b.py: the build, the marks, the reverse, the built engine's defaults")
    ap.add_argument("--v9c", default=None, help="engine_twofield_v9c.py as built: must equal the builder's output byte for byte")
    ap.add_argument("--v9b-selftest", default=None, help="engine_twofield_v9b_selftest.py: run with v9c in v9b's place in floor mode")
    ap.add_argument("--light-def", default=B.LIGHT_DEF_DEFAULT); ap.add_argument("--heavy-def", default=B.HEAVY_DEF_DEFAULT)
    ap.add_argument("--skip-torch", action="store_true"); ap.add_argument("--skip-suite", action="store_true")
    ap.add_argument("--suite-timeout", type=int, default=3600)
    a = ap.parse_args(argv)
    C = Checks()
    L = B.load_block_namespace(a.light_def, a.heavy_def)
    C.ok("0 the layer loads from the builder's block; the primitive imports", L["v9c_prim"]().VERSION.startswith("assign_primitive"), L["v9c_prim"]().VERSION)
    run_layer_checks(C, L, skip_torch=a.skip_torch)
    if a.v9b:
        run_engine_checks(C, a)
    else:
        print("SKIP s, t, u (no --v9b given): the layer checks only")
    failed = C.failed()
    if failed:
        print("engine_twofield_v9c_selftest: FAIL (%d of %d checks): %s" % (len(failed), len(C.rows), "; ".join(failed)))
        return 1
    print("engine_twofield_v9c_selftest: PASS (%d checks)" % len(C.rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
