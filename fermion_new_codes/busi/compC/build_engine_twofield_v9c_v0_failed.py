#!/usr/bin/env python
"""build_engine_twofield_v9c.py (reviewer A for the busi lane; C64_v9c_ruling_v1.md section 2; C64_G4s8_hold_ruling_v1 (4) and (4b); PRIM_ruling_v1 3.4):
engine_twofield_v9c.py over engine_twofield_v9b.py in the lane's patch form: exact replacements, each occurring once in v9b; every line of v9c not in
v9b carries "v9c prim"; --reverse restores v9b byte for byte; the defaults are bitwise v9b.

What v9c does. One run argument --anchor-passage {floor, lz} (default floor). floor: the two sector routes are called untouched, so every number of
v9b is reproduced in both --cut-transport modes. lz: after v9b's route has certified a channel's path and assembled its result, the assignment
primitive (--prim-module, default assign_primitive_v1d; its VERSION is written into every sector entry) follows the channel's occupied set from the
stored mask along the record's straight path K(s) = (1 - s) h(x_(n-1)) + s h(x_n) with Rule lz (tau_LZ = k gap_min^2 / |alpha_T| against c, k = 1,
c = 1, alpha_T = alpha_s / tau), levels 1 to --prim-levels (4), w_mix 0.9 and the charge-conjugation mirror read off the spectrum (PRIM 2.1 to 2.5).
One spectrum at s = 1/2 is the trigger: a channel whose set is clean at 1/2 and at 1 on its start indices has no passage on the chord and costs one
eigendecomposition; a channel whose boundary pairs are wider than --lz-skip-gap (2e-2) at both ends and whose adjacent levels move by less than half
of the smaller splitting costs none. Every completed or incomplete passage becomes an entry of the channel's "crossings": the branch (diabatic or
adiabatic), tau_LZ with its band, the Landau-Zener exponent pi tau_LZ / (2 k), gap_min, alpha_T, s*, the pair's indices at both ends, the overlap (the
pairing's character weight), the charge jumps (grid, fit, alt) and whether they fall under the halving; approaches and tails (the fitted minimum
beyond or before the chord) are entries without a decision, so a record whose end lies inside a window carries an entry of its own; blocks and
undecided or knife-edge passages are flagged. A sub-floor pair keeps v9b's crossing phase (the swap is 9b's) and its entry is annotated by the matched
event: PRIM 3.4 by the chord rather than the probes. Where the decided set at s = 1 differs from the walk's (a sub-threshold minimum decided diabatic:
G4s8's six passages), the route is re-called on the degenerate chord from h(x_n) to h(x_n) with the decided set, so the reference is assembled in
v9b's own form and the certificate's bookkeeping stays the walk's; min_occupied_weight and adiabatic_at_cut are then recomputed against the previous
reference. The gate's and the watch's read of a channel is v9c_hold_reasons(sector entry).

The patch. R1 version (soft; skipped with a notice when absent): ENGINE_VERSION = "engine_twofield_v9b" -> "engine_twofield_v9c". R2 args (hard): the
first line of the add_argument call of --cut-transport gains, before it, v9c_add_arguments(<parser>), the parser's name read off that line. R3 layer
(hard): the block V9C_BLOCK is inserted before def light_sector_walk_v9( (--light-def). R4 rebind (hard): before the main guard the two sector routes
(--light-def, --heavy-def, default heavy_sector_walk_v9) are rebound by name to their wrapper; a route not defined is listed in V9C_MISSING and
--anchor-passage lz then refuses to parse, naming it.

The contract with v9b (the wrapper reads the routes' arguments by name, V9C_NAMES, and their return). Arguments: h0, h1 (the channel's dense N x N
operators at x_(n-1) and x_n, numpy or torch, as the crossing phase forms K(s)), e1, U1 (the eigendecomposition at x_n) and mask0 (the stored occupied
index mask of the sorted spectrum) are required; e0, U0 (the eigendecomposition at x_(n-1); else K(0) is diagonalized), Vprev (the previous reference's
occupied columns, for min_occupied_weight), anchors0 (the anchor list; midpoints for the re-call), tau (else args.tau), args, kappa or ell (the
channel label) and floor (3.5e-7 when absent) are read when present. Return: a tuple holding the sector entry (a dict with "crossings" or "route") and
the end mask (a one-dimensional boolean array); the other items are the reference. A mismatch raises V9cContract naming the signature or the return;
the lane posts the line it names and edits V9C_NAMES, V9C_LIGHT_DEF or V9C_HEAVY_DEF here (or passes --light-def / --heavy-def); no other edit.

Usage. python build_engine_twofield_v9c.py engine_twofield_v9b.py [engine_twofield_v9c.py] [--light-def NAME] [--heavy-def NAME]
         --probe: the counts of the OLD strings in v9b and the candidate lines (nothing written)
         --check: the build in memory, the marks check and the reverse round trip (nothing written)
         --reverse engine_twofield_v9c.py [restored.py] [--against engine_twofield_v9b.py]: the reverse replacements, byte for byte
The builder stops without writing when any hard OLD string does not occur exactly once, naming it and the lines to post.
"""
import argparse
import hashlib
import os
import re
import sys

MARK = "v9c prim"
LIGHT_DEF_DEFAULT = "light_sector_walk_v9"
HEAVY_DEF_DEFAULT = "heavy_sector_walk_v9"
MAIN_GUARD = 'if __name__ == "__main__":'
OLD_VERSION = 'ENGINE_VERSION = "engine_twofield_v9b"'
NEW_VERSION = 'ENGINE_VERSION = "engine_twofield_v9c"  # v9c prim'
OLD_ARGS = '    ap.add_argument("--cut-transport", choices=("record", "sector"), default="record",'

V9C_BLOCK = r'''# >>> v9c prim block (light route @LIGHT_DEF@, heavy route @HEAVY_DEF@): the assignment primitive at every anchor passage of the sector walk;
# build_engine_twofield_v9c.py over engine_twofield_v9b.py; C64_v9c_ruling_v1 section 2, C64_G4s8_hold_ruling_v1 (4) and (4b), PRIM_ruling_v1 3.4.
# Defaults bitwise v9b: with --anchor-passage floor (the default) the routes are called untouched; with lz each channel's occupied set is followed
# along the record's straight path by the primitive named by --prim-module and every anchor passage is decided by its Landau-Zener rule.
import inspect as _v9c_inspect
import importlib as _v9c_importlib
import math as _v9c_math
import argparse as _v9c_argparse
import numpy as _v9c_np

V9C_VERSION = "engine_twofield_v9c"
V9C_LIGHT_DEF = "@LIGHT_DEF@"
V9C_HEAVY_DEF = "@HEAVY_DEF@"
V9C_DEFAULTS = {"anchor_passage": "floor", "prim_module": "assign_primitive_v1d", "lz_c": 1.0, "lz_k": 1.0, "lz_tie": "adiabatic", "lz_knife": 0.1,
                "prim_levels": 4, "prim_w_mix": 0.9, "lz_skip_gap": 2e-2}
V9C_MODE = dict(V9C_DEFAULTS)
V9C_NAMES = {"light": {"h0": "h0", "h1": "h1", "e0": "e0", "U0": "U0", "e1": "e1", "U1": "U1", "mask0": "mask0", "vprev": "Vprev", "anchors0": "anchors0",
                       "tau": "tau", "args": "args", "channel": "kappa", "floor": "floor"},
             "heavy": {"h0": "h0", "h1": "h1", "e0": "e0", "U0": "U0", "e1": "e1", "U1": "U1", "mask0": "mask0", "vprev": "Vprev", "anchors0": "anchors0",
                       "tau": "tau", "args": "args", "channel": "ell", "floor": "floor"}}
V9C_REQUIRED = ("h0", "h1", "e1", "U1", "mask0")
V9C_FLOOR = 3.5e-7
V9C_MISSING = []
V9C_INFO_FLAGS = ("exact crossing to this fit", "third sample at s=", "no set change, no event", "two samples only")
_V9C_PRIM = {}


class V9cContract(RuntimeError):
    """v9b's sector routes do not meet the wrapper's contract (V9C_NAMES, the return form); the message names the line the lane posts"""


class V9cStore(_v9c_argparse.Action):
    """stores the parsed value in the namespace and in V9C_MODE (the routes' wrapper reads the mode without the namespace in its scope); a default
    never passes through an Action, so V9C_MODE keeps floor unless --anchor-passage lz is on the command line"""

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)
        V9C_MODE[self.dest] = values
        if self.dest == "anchor_passage" and values == "lz" and V9C_MISSING:
            parser.error("--anchor-passage lz: the sector route(s) %s are not defined in this engine; post their def lines and rebuild with --light-def / --heavy-def" % ", ".join(V9C_MISSING))


def v9c_add_arguments(ap):
    """the v9c arguments (the builder's hook R2 places this call before --cut-transport); every default reproduces v9b"""
    ap.add_argument("--anchor-passage", dest="anchor_passage", choices=("floor", "lz"), default="floor", action=V9cStore,
                    help="floor: v9b, a certified passage keeps the adiabatic branch and the floor is the only threshold; lz: every anchor passage of the sector walk decided by the Landau-Zener rule of the assignment primitive along the record's straight path")
    ap.add_argument("--prim-module", dest="prim_module", default="assign_primitive_v1d", action=V9cStore,
                    help="the assignment primitive (assign_primitive_v1d or its successor); its VERSION is recorded in every sector entry")
    ap.add_argument("--lz-c", dest="lz_c", type=float, default=1.0, action=V9cStore,
                    help="c_LZ: adiabatic when tau_LZ = k gap_min^2 / |alpha_T| >= c on the whole band (the record convention's c_LZ = 1)")
    ap.add_argument("--lz-k", dest="lz_k", type=float, default=1.0, action=V9cStore,
                    help="k of tau_LZ (1: the lane's reading of the record line; pi/2 makes tau_LZ the Landau-Zener exponent)")
    ap.add_argument("--lz-tie", dest="lz_tie", choices=("adiabatic", "diabatic"), default="adiabatic", action=V9cStore,
                    help="the branch when the band on tau_LZ still contains c at the deepest level (the passage is then flagged undecided: a HOLD)")
    ap.add_argument("--lz-knife", dest="lz_knife", type=float, default=0.1, action=V9cStore,
                    help="a decided passage with |ln(tau_LZ / c)| below this is flagged knife-edge")
    ap.add_argument("--prim-levels", dest="prim_levels", type=int, default=4, action=V9cStore,
                    help="the primitive's deepest halving level (levels 1 .. this; stability needs two levels, so at least 2)")
    ap.add_argument("--prim-w-mix", dest="prim_w_mix", type=float, default=0.9, action=V9cStore,
                    help="the primitive's clean-point weight (a point with a set member below it lies inside a crossing's window)")
    ap.add_argument("--lz-skip-gap", dest="lz_skip_gap", type=float, default=2e-2, action=V9cStore,
                    help="a channel whose boundary pairs are wider than this at both ends of the record and whose adjacent levels move by less than half the smaller splitting is not examined (no passage can lie inside)")


def v9c_mode(args=None):
    """the mode: V9C_MODE (the parsed command line) overridden by the parsed namespace where it carries the keys"""
    M = dict(V9C_MODE)
    if args is not None:
        for key in V9C_DEFAULTS:
            if hasattr(args, key):
                M[key] = getattr(args, key)
    return M


def v9c_prim(name=None):
    """the primitive module, imported once by name"""
    name = name or V9C_MODE["prim_module"]
    mod = _V9C_PRIM.get(name)
    if mod is None:
        mod = _v9c_importlib.import_module(name)
        _V9C_PRIM[name] = mod
    return mod


def v9c_rule(M=None):
    """Rule lz of the primitive with the declared c_LZ and k"""
    M = M if M is not None else V9C_MODE
    prim = v9c_prim(M["prim_module"])
    return prim.Rule(kind="lz", c=float(M["lz_c"]), k=float(M["lz_k"]), hbar=1.0, tie=M["lz_tie"], knife=float(M["lz_knife"]))


def _v9c_is_torch(x):
    return hasattr(x, "device") and hasattr(x, "detach") and hasattr(x, "cpu")


def _v9c_to_np(x):
    if _v9c_is_torch(x):
        return x.detach().cpu().numpy()
    return _v9c_np.asarray(x)


def _v9c_dense(h):
    if hasattr(h, "toarray") and not _v9c_is_torch(h):
        return h.toarray()
    return h


def _v9c_K(h0, h1, s):
    """K(s) = (1 - s) h(x_(n-1)) + s h(x_n) on the record's straight path (s beyond [0, 1] is the primitive's one-piece extension)"""
    s = float(s)
    return (1.0 - s) * _v9c_dense(h0) + s * _v9c_dense(h1)


def _v9c_eigh(K):
    """the eigendecomposition of K in its own backend: numpy eigh or torch eigh on the device; the eigenvalues as numpy"""
    if _v9c_is_torch(K):
        import torch
        e, U = torch.linalg.eigh(K)
        return _v9c_np.asarray(e.detach().cpu().numpy(), float), U
    e, U = _v9c_np.linalg.eigh(_v9c_np.asarray(K))
    return _v9c_np.asarray(e, float), U


def _v9c_cols(U, cols):
    return U[:, [int(c) for c in cols]]


def _v9c_weights(A, B):
    """|A^H B|^2 as a numpy array, for numpy or torch columns (a numpy operand is moved beside a torch one)"""
    ta, tb = _v9c_is_torch(A), _v9c_is_torch(B)
    if ta or tb:
        import torch
        if not ta:
            A = torch.as_tensor(_v9c_np.asarray(A), device=B.device, dtype=B.dtype)
        if not tb:
            B = torch.as_tensor(_v9c_np.asarray(B), device=A.device, dtype=A.dtype)
        W = (A.conj().transpose(0, 1) @ B).abs() ** 2
        return _v9c_np.asarray(W.detach().cpu().numpy(), float)
    return _v9c_np.abs(_v9c_np.asarray(A).conj().T @ _v9c_np.asarray(B)) ** 2


def _v9c_mask_like(m, template):
    """the decided mask in the walk's own type (a torch mask when the walk returns one)"""
    if _v9c_is_torch(template):
        import torch
        return torch.as_tensor(_v9c_np.asarray(m, bool), device=template.device)
    return _v9c_np.asarray(m, bool)


def _v9c_py(x):
    """plain Python for json"""
    if isinstance(x, dict):
        return {str(k): _v9c_py(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_v9c_py(v) for v in x]
    if isinstance(x, _v9c_np.ndarray):
        return [_v9c_py(v) for v in x.tolist()]
    if isinstance(x, _v9c_np.bool_):
        return bool(x)
    if isinstance(x, _v9c_np.integer):
        return int(x)
    if isinstance(x, _v9c_np.floating):
        return float(x)
    return x


def v9c_boundaries(mask):
    """the indices j with exactly one of j, j + 1 occupied (the anchors' positions)"""
    m = _v9c_np.asarray(_v9c_to_np(mask), bool)
    return _v9c_np.flatnonzero(m[:-1] != m[1:])


def v9c_anchors(e, mask):
    """the anchors as midpoints a_j = (e_j + e_(j+1)) / 2 of the mask's boundaries"""
    e = _v9c_np.asarray(_v9c_to_np(e), float)
    return [float(0.5 * (e[j] + e[j + 1])) for j in v9c_boundaries(mask)]


def v9c_inversions(mask):
    """the occupied indices above the lowest empty one"""
    m = _v9c_np.asarray(_v9c_to_np(mask), bool)
    occ = _v9c_np.flatnonzero(m); emp = _v9c_np.flatnonzero(~m)
    if occ.size == 0 or emp.size == 0:
        return 0
    return int(_v9c_np.count_nonzero(occ > emp[0]))


def v9c_pockets(mask):
    """the maximal runs of occupied indices"""
    m = _v9c_np.asarray(_v9c_to_np(mask), bool)
    if m.size == 0:
        return 0
    d = _v9c_np.diff(_v9c_np.concatenate(([0], m.astype(int))))
    return int(_v9c_np.count_nonzero(d == 1))


def v9c_wmin(vprev, U1, mask):
    """the decided set's weight on the previous reference: (the minimum over its occupied vectors, the count below 1/2)"""
    occ = _v9c_np.flatnonzero(_v9c_np.asarray(_v9c_to_np(mask), bool))
    if occ.size == 0:
        return 1.0, 0
    p = _v9c_weights(vprev, _v9c_cols(U1, occ)).sum(axis=0)
    return float(p.min()), int(_v9c_np.count_nonzero(p < 0.5))


class V9cOracle(object):
    """the primitive's oracle on the record's straight path: the engine's eigendecompositions at s = 0 and s = 1 when given, its own (eigh of K(s)
    in the operators' backend) elsewhere; every spectrum of one call cached by s"""

    def __init__(self, h0, h1, e0=None, U0=None, e1=None, U1=None):
        self.h0 = h0; self.h1 = h1; self.cache = {}; self.n_eigh = 0; self.computed = []
        if e1 is not None and U1 is not None:
            self.cache[1.0] = (_v9c_np.asarray(_v9c_to_np(e1), float), U1)
        if e0 is not None and U0 is not None:
            self.cache[0.0] = (_v9c_np.asarray(_v9c_to_np(e0), float), U0)

    def spectrum(self, s, channels):
        s = float(s)
        if s not in self.cache:
            e, U = _v9c_eigh(_v9c_K(self.h0, self.h1, s))
            self.cache[s] = (e, U); self.n_eigh += 1; self.computed.append(s)
        return {ch: self.cache[s] for ch in channels}

    def columns(self, U, cols):
        return _v9c_cols(U, cols)

    def weights(self, A, B):
        return _v9c_weights(A, B)

    def project(self, A, B):
        return self.weights(A, B).sum(axis=0)

    def release(self, U):
        return None


def v9c_skip(e0, e1, mask, skip_gap):
    """True when no passage of the set can lie on the chord: at every boundary the pair's splitting at both ends exceeds skip_gap and neither level
    adjacent to the boundary moves by half the smaller splitting or more over the record (the diabats cannot meet inside)"""
    e0 = _v9c_np.asarray(e0, float); e1 = _v9c_np.asarray(e1, float); J = v9c_boundaries(mask)
    if J.size == 0:
        return True
    for j in J:
        d0 = float(e0[j + 1] - e0[j]); d1 = float(e1[j + 1] - e1[j]); dmin = min(d0, d1)
        if dmin <= float(skip_gap):
            return False
        move = max(abs(float(e1[j] - e0[j])), abs(float(e1[j + 1] - e0[j + 1])))
        if move >= 0.5 * dmin:
            return False
    return True


def v9c_precheck(oracle, channel, mask0, w_mix):
    """the trigger: with the set's eigenvectors at s = 0 as the reference, its character at s = 1/2 and at s = 1 is clean (every member's weight at
    least w_mix, every other level's at most 1 - w_mix) and on the start indices: no passage lies on the chord. One spectrum beyond the record's two."""
    m = _v9c_np.asarray(_v9c_to_np(mask0), bool); occ = _v9c_np.flatnonzero(m); n = int(occ.size); N = int(m.size)
    if n == 0 or n == N:
        return True
    e0, U0 = oracle.spectrum(0.0, [channel])[channel]
    ref = oracle.columns(U0, occ)
    for s in (0.5, 1.0):
        e, U = oracle.spectrum(s, [channel])[channel]
        p = oracle.project(ref, U)
        order = _v9c_np.argsort(-p, kind="stable"); top = _v9c_np.zeros(N, bool); top[order[:n]] = True
        if not (float(p[top].min()) >= float(w_mix) and float(p[~top].max()) <= 1.0 - float(w_mix) and _v9c_np.array_equal(top, m)):
            return False
    return True


def _v9c_exponent(A, k):
    """the Landau-Zener exponent pi gap_min^2 / (2 |alpha_T|) = pi tau_LZ / (2 k)"""
    if A is None:
        return None
    A = float(A)
    if not _v9c_math.isfinite(A):
        return float("inf")
    return (_v9c_math.pi / 2.0) * A / float(k)


def _v9c_holding_flags(flags):
    return [f for f in (flags or []) if not any(key in f for key in V9C_INFO_FLAGS)]


def _v9c_apply_swap(m, ev):
    """the running set after a diabatic event: the event's character mask when the primitive gives it, else the pair swap (and its mirror's)"""
    ma = ev.get("mask_after")
    if ma is not None:
        return _v9c_np.asarray(ma, bool).copy()
    m = _v9c_np.asarray(m, bool).copy()
    if ev.get("partner_c") is not None and ev.get("n") is not None:
        m[int(ev["partner_c"])] = False; m[int(ev["n"])] = True
    mr = ev.get("mirror")
    if mr and mr.get("partner_c") is not None and mr.get("n") is not None:
        m[int(mr["partner_c"])] = False; m[int(mr["n"])] = True
    return m


def _v9c_entry_from_event(ev, M, m_run):
    """one entry of "crossings" from one event of the primitive; returns (entry, the running set after it)"""
    k = float(M["lz_k"]); decision = ev.get("decision"); diabatic = decision == "diabatic"
    kind = "block" if ev.get("block") else ev.get("kind", "passage")
    overlap = ev.get("w_new") if ev.get("kind") == "passage" else ev.get("w_partner")
    holding = _v9c_holding_flags(ev.get("flags"))
    flagged = bool(holding) or not (ev.get("resolved") and ev.get("decided")) or bool(ev.get("knife")) or bool(ev.get("tie")) or bool(ev.get("block"))
    inv_before = v9c_inversions(m_run)
    m_after = _v9c_apply_swap(m_run, ev) if (diabatic and not ev.get("block")) else _v9c_np.asarray(m_run, bool).copy()
    inv_after = v9c_inversions(m_after)
    entry = {"s": ev.get("s_star"), "piece": [ev.get("s_p"), ev.get("s_c")], "probes": ev.get("n_samples"), "source": "primitive", "kind": kind, "level": ev.get("level"),
             "branch": decision, "decided": bool(ev.get("decided")), "resolved": bool(ev.get("resolved")), "swap": bool(diabatic), "flagged": bool(flagged),
             "tau_lz": ev.get("A"), "tau_lz_band": [ev.get("A_lo"), ev.get("A_hi")], "exponent": _v9c_exponent(ev.get("A"), k), "c": float(M["lz_c"]), "k": k,
             "gap_min": ev.get("gap_min"), "gap_min2_err": ev.get("gap_min2_err"), "alpha_s": ev.get("alpha_s"), "alpha_T": ev.get("alpha_T"), "s_star": ev.get("s_star"),
             "fit_kind": ev.get("fit_kind"), "pair_start": [ev.get("i"), ev.get("partner_p")], "pair_end": [ev.get("partner_c"), ev.get("n")],
             "overlap": overlap, "weight_end": ev.get("w_new"), "delta": int(inv_after - inv_before), "mirrored": bool(ev.get("mirrored")),
             "self_mirror": bool(ev.get("self_mirror")), "mirror": ev.get("mirror"), "jump_grid": ev.get("jump_grid"), "jump_fit": ev.get("jump_fit"),
             "jump_alt": ev.get("jump_alt"), "jump_prev": ev.get("jump_prev"), "falls": ev.get("falls"), "tie": bool(ev.get("tie")), "knife": bool(ev.get("knife")),
             "block": bool(ev.get("block")), "flags": list(ev.get("flags") or []), "holding_flags": holding}
    return _v9c_py(entry), m_after


def _v9c_entry_from_run(run, M, tau):
    """an approach or a tail: the fitted minimum lies beyond or before this chord; an entry without a decision"""
    alpha_s = run.get("alpha_s"); gm = run.get("gap_min")
    alpha_T = (float(alpha_s) / float(tau)) if alpha_s else None
    A = None if (gm is None or not alpha_T) else float(M["lz_k"]) * float(gm) * float(gm) / abs(alpha_T)
    kind = run.get("kind")
    entry = {"s": run.get("s_star"), "piece": None, "probes": len(run.get("samples") or []), "source": "primitive", "kind": kind,
             "branch": ("pending" if kind == "approach" else "before"), "decided": False, "resolved": True, "swap": False, "flagged": False,
             "tau_lz": A, "exponent": _v9c_exponent(A, M["lz_k"]), "gap_min": gm, "alpha_s": alpha_s, "alpha_T": alpha_T, "s_star": run.get("s_star"),
             "pair_start": [run.get("i"), None], "pair_end": list(run.get("pair") or []), "overlap": None, "delta": 0,
             "flags": ["the fitted minimum lies %s this chord: %s, no decision here" % ("after" if kind == "approach" else "before", kind)], "holding_flags": []}
    return _v9c_py(entry)


def _v9c_merge_walk(entries, crossings_walk, level, floor, info):
    """the walk's own crossing entries matched to the primitive's events by s (within a piece of the final level): matched ones carry both, an
    unmatched walk crossing and a walk swap against the primitive's decision are flagged; the primitive's events the walk did not see are appended"""
    tol = 2.0 ** (-max(int(level or 1), 1)) + 1e-9
    used = set(); out = []
    for c in (crossings_walk or []):
        s_w = c.get("s"); match = None
        if s_w is not None:
            best = None
            for q, ent in enumerate(entries):
                if q in used or ent.get("s") is None or ent.get("kind") in ("approach", "tail"):
                    continue
                d = abs(float(ent["s"]) - float(s_w))
                if d <= tol and (best is None or d < best[0]):
                    best = (d, q)
            if best is not None:
                match = best[1]
        merged = dict(c)
        merged["flags"] = list(c.get("flags") or [])
        if match is None:
            merged["source"] = "walk"; merged.setdefault("branch", "diabatic"); merged["flagged"] = True
            merged["flags"].append("the walk's crossing has no event of the primitive within %.3g of s = %s" % (tol, s_w))
            info["walk_consistent"] = False
        else:
            ent = entries[match]; used.add(match)
            for key, val in ent.items():
                if key in ("s", "piece", "probes", "overlap", "delta", "block", "swap", "flagged", "flags"):
                    merged["prim_" + key] = val
                else:
                    merged.setdefault(key, val)
            merged["source"] = "walk+primitive"
            merged["flagged"] = bool(c.get("flagged")) or bool(ent.get("flagged"))
            merged["flags"].extend(list(ent.get("flags") or []))
            if ent.get("branch") != "diabatic":
                merged["flagged"] = True; merged["flags"].append("the walk swapped a pair the primitive decided %s" % ent.get("branch")); info["walk_consistent"] = False
        out.append(merged)
    for q, ent in enumerate(entries):
        if q in used:
            continue
        ent = dict(ent)
        err = ent.get("gap_min2_err")
        if ent.get("branch") == "diabatic" and ent.get("kind") == "passage" and ent.get("gap_min") is not None and err is not None:
            if float(ent["gap_min"]) + _v9c_math.sqrt(max(float(err), 0.0)) < float(floor):
                ent["flagged"] = True; ent["flags"] = list(ent.get("flags") or []) + ["a sub-floor passage (gap_min %.2e with its band below the floor %.2e) the walk's certificate did not open" % (float(ent["gap_min"]), float(floor))]
                info["walk_consistent"] = False
        out.append(ent)
    out.sort(key=lambda d: (float("inf") if d.get("s") is None else float(d["s"])))
    return out


def _v9c_check_shapes(species, channel, h0, h1, e1, U1, N):
    for nm, h in (("h0", h0), ("h1", h1)):
        shp = tuple(int(x) for x in getattr(_v9c_dense(h), "shape", ()))
        if len(shp) != 2 or shp[0] != N or shp[1] != N:
            raise V9cContract("v9c prim: %s channel %s: %s is not the channel's dense N x N operator (shape %s against N = %d from the mask); V9C_NAMES[%r][%r] names it" % (species, channel, nm, shp, N, species, nm))
    shp = tuple(int(x) for x in getattr(U1, "shape", ()))
    if int(_v9c_np.asarray(e1).size) != N or shp != (N, N):
        raise V9cContract("v9c prim: %s channel %s: e1, U1 are not the eigendecomposition at x_n of the channel's operator (%d eigenvalues, U1 %s, N %d)" % (species, channel, int(_v9c_np.asarray(e1).size), shp, N))


def _v9c_say(text):
    print(text, flush=True)


def v9c_line(info):
    c = info.get("counts") or {}
    tl = info.get("tau_lz_min")
    tl_s = "-" if tl is None else ("%.3e" % tl if _v9c_math.isfinite(tl) else "inf")
    return "%s %s primitive (lz, %s, tau %g): level %s of %s, spectra %d, entries %d (diabatic %d adiabatic %d approach %d tail %d unresolved %d), tau_LZ min %s, exponent sum %.3e, set changes %d, inversions %d -> %d%s%s" % (
        info.get("species"), info.get("channel"), info.get("prim"), info.get("tau", 0.0), info.get("level_stable"), info.get("prim_levels"), info.get("n_spectra", 0), len(info.get("entries") or []),
        c.get("diabatic", 0), c.get("adiabatic", 0), c.get("approach", 0), c.get("tail", 0), c.get("unresolved", 0), tl_s, float(info.get("exponent_sum") or 0.0),
        int(info.get("set_changes") or 0), int(info.get("inversions_start") or 0), int(info.get("inversions_end") or 0),
        "" if info.get("walk_consistent", True) else "; INCONSISTENT with the walk", "; UNRESOLVED" if info.get("unresolved") else "")


def v9c_decide(species, channel, h0, h1, e0, U0, e1, U1, mask0, mask_walk, crossings_walk, tau, floor=None, mode=None, say=None):
    """The decision of one channel's record (the hook of both species). Returns (mask_out, info): mask_out is the occupied index mask at x_n (the walk's
    own object when nothing changed, else the primitive's decided set) and info the sector entry's passage block; (mask_walk, None) in floor mode,
    with nothing computed."""
    M = v9c_mode() if mode is None else dict(mode)
    if M.get("anchor_passage", "floor") != "lz":
        return mask_walk, None
    levels = int(M["prim_levels"])
    if levels < 2:
        raise ValueError("v9c prim: --prim-levels must be at least 2 (stability under halving needs two levels)")
    prim = v9c_prim(M["prim_module"]); rule = v9c_rule(M)
    m0 = _v9c_np.asarray(_v9c_to_np(mask0), bool).copy(); mw = _v9c_np.asarray(_v9c_to_np(mask_walk), bool).copy(); N = int(m0.size)
    floor = V9C_FLOOR if floor is None else float(floor); tau = float(tau); w_mix = float(M["prim_w_mix"])
    cw = list(crossings_walk or []); walk_changed = not _v9c_np.array_equal(m0, mw)
    info = {"mode": "lz", "engine": V9C_VERSION, "prim": prim.VERSION, "rule": rule.describe(), "species": str(species), "channel": str(channel), "tau": tau,
            "prim_levels": levels, "ran": False, "skipped": False, "quiet": False, "n_spectra": 0, "mask_changed": False, "entries": [], "flags": [], "notes": [],
            "stable": None, "level_stable": None, "unresolved": False, "walk_consistent": True, "walk_changed": bool(walk_changed), "walk_crossings": len(cw),
            "counts": {"diabatic": 0, "adiabatic": 0, "approach": 0, "tail": 0, "unresolved": 0}, "tau_lz_min": None, "exponent_sum": 0.0,
            "set_changes": 0, "inversions_start": v9c_inversions(m0), "inversions_end": v9c_inversions(mw)}
    if walk_changed and not cw:
        info["walk_consistent"] = False
        info["flags"].append("the walk changed the set (%d indices) with no crossing entry" % int(_v9c_np.count_nonzero(m0 != mw)))
    e1a = _v9c_np.asarray(_v9c_to_np(e1), float)
    _v9c_check_shapes(species, channel, h0, h1, e1a, U1, N)
    oracle = V9cOracle(h0, h1, e0, U0, e1a, U1)
    e0a = oracle.spectrum(0.0, [channel])[channel][0]
    if (not walk_changed) and v9c_skip(e0a, e1a, m0, float(M["lz_skip_gap"])):
        info["skipped"] = True; info["n_spectra"] = oracle.n_eigh
        return mask_walk, info
    if (not walk_changed) and v9c_precheck(oracle, channel, m0, w_mix):
        info["quiet"] = True; info["n_spectra"] = oracle.n_eigh
        return mask_walk, info
    dec = prim.follow_chord(oracle, {channel: m0}, rule=rule, queries=(), tau=tau, weights_ch={channel: 1.0}, level_min=1, level_max=levels,
                            w_mix=w_mix, mirror="auto", channels=[channel], log=None)
    m_prim = _v9c_np.asarray(dec.masks_end[channel], bool).copy()
    final = dec.results[channel][-1]
    entries = []; m_run = m0.copy()
    for ev in sorted(dec.crossings, key=lambda d: (float("inf") if d.get("s_star") is None else float(d["s_star"]))):
        ent, m_run = _v9c_entry_from_event(ev, M, m_run); entries.append(ent)
    for run in dec.end_runs.get(channel, []):
        entries.append(_v9c_entry_from_run(run, M, tau))
    merged = _v9c_merge_walk(entries, cw, final["level"], floor, info)
    counts = info["counts"]; tl = []
    for ent in merged:
        br = ent.get("branch"); kind = ent.get("kind")
        if kind in ("approach", "tail"):
            counts[kind] += 1
        elif br in ("diabatic", "adiabatic") and ent.get("decided", True) and not ent.get("block") and not ent.get("flagged"):
            counts[br] += 1
        else:
            counts["unresolved"] += 1
        if ent.get("tau_lz") is not None and kind not in ("approach", "tail"):
            tl.append(float(ent["tau_lz"]))
        if br == "diabatic" and kind not in ("approach", "tail") and ent.get("exponent") is not None and _v9c_math.isfinite(float(ent["exponent"])):
            info["exponent_sum"] += float(ent["exponent"])
    info.update({"ran": True, "n_spectra": oracle.n_eigh, "stable": bool(dec.stable), "level_stable": dec.level_stable.get(channel), "levels": dec.levels.get(channel),
                 "jumps_fall": bool(dec.jumps_fall), "printable_set": bool(dec.printable_set), "entries": merged, "end_in_window": bool(dec.end_in_window.get(channel)),
                 "mask_changed": bool(not _v9c_np.array_equal(m_prim, mw)), "set_changes": int(_v9c_np.count_nonzero(m_prim != m0)), "inversions_end": v9c_inversions(m_prim),
                 "tau_lz_min": (min(tl) if tl else None)})
    info["flags"].extend(list(dec.flags.get(channel, []))); info["notes"].extend(list(dec.notes.get(channel, [])))
    info["unresolved"] = bool((not info["stable"]) or counts["unresolved"] > 0 or (not info["jumps_fall"]))
    if say is None:
        say = _v9c_say
    if say and (merged or info["unresolved"] or not info["walk_consistent"]):
        say(v9c_line(info))
    return (m_prim if info["mask_changed"] else mask_walk), info


def v9c_annotate(ent, info, mask_final=None):
    """the sector entry in place: the passage block, the merged crossings, the counts of the decided set"""
    if info is None:
        return ent
    ent["passage"] = _v9c_py({k: v for k, v in info.items() if k != "entries"})
    if info.get("ran"):
        ent["crossings_walk"] = list(ent.get("crossings") or [])
        ent["crossings"] = list(info["entries"])
        if mask_final is not None:
            m = _v9c_np.asarray(_v9c_to_np(mask_final), bool)
            ent["inversions_walk"] = ent.get("inversions_sector"); ent["inversions_sector"] = v9c_inversions(m)
            ent["anchors_end_walk"] = ent.get("anchors_end"); ent["anchors_end"] = int(v9c_boundaries(m).size)
            ent["pockets_prim"] = v9c_pockets(m)
    return ent


def v9c_hold_reasons(sector, w_mix=0.9):
    """the gate's and the watch's v9c reads of one channel's sector entry (C64_G4s8_hold_ruling_v1 (4b); C64_v9c_ruling_v1 3): a list of (what, value,
    bound), empty when nothing holds"""
    S = sector if isinstance(sector, dict) else {}
    out = []
    cr = S.get("crossings") or []; wmin = S.get("min_occupied_weight")
    if (not cr) and wmin is not None and float(wmin) < float(w_mix):
        out.append(("unresolved passage (certified record, no entry of its own)", float(wmin), ">= %.2f or an entry" % w_mix))
    for c in cr:
        if c.get("kind") in ("approach", "tail"):
            continue
        kind = c.get("kind", "crossing"); s = c.get("s")
        if c.get("flagged") or c.get("block") or (c.get("source") != "walk" and c.get("source") is not None and (c.get("decided") is False or c.get("resolved") is False)):
            out.append(("flagged or undecided passage at s = %s (%s, %s)" % (s, kind, c.get("branch")), c.get("tau_lz"), "decided and unflagged"))
        if c.get("knife"):
            out.append(("knife-edge passage at s = %s" % s, c.get("tau_lz"), "|ln(tau_LZ / c)| >= knife"))
        if c.get("branch") == "diabatic" and c.get("falls") is False:
            out.append(("the jump does not fall under the halving at s = %s" % s, c.get("jump_grid"), "<= the previous level's"))
        ov = c.get("overlap")
        if ov is not None and float(ov) < float(w_mix):
            out.append(("passage overlap at s = %s" % s, float(ov), ">= %.2f" % w_mix))
    P = S.get("passage") or {}
    if P.get("ran") and P.get("stable") is False:
        out.append(("the primitive unstable at its deepest level", P.get("level_stable"), "stable"))
    if P.get("walk_consistent") is False:
        out.append(("the walk and the primitive disagree", P.get("flags"), "consistent"))
    if P.get("ran") and P.get("jumps_fall") is False:
        out.append(("the charge jumps do not fall under the halving", None, "fall"))
    return out


def _v9c_split_return(out, species, fn):
    """the route's return: a tuple holding the sector entry (a dict with 'crossings' or 'route') and the end mask (a one-dimensional boolean array)"""
    if not isinstance(out, (tuple, list)):
        raise V9cContract("v9c prim: the %s sector route %s returned a %s, not a tuple; post its return line" % (species, getattr(fn, "__name__", "?"), type(out).__name__))
    items = list(out)
    ient = [q for q, x in enumerate(items) if isinstance(x, dict) and ("crossings" in x or "route" in x)]
    imask = [q for q, x in enumerate(items) if str(getattr(x, "dtype", "")) in ("bool", "torch.bool") and int(getattr(x, "ndim", 0)) == 1]
    if len(ient) != 1 or len(imask) != 1:
        raise V9cContract("v9c prim: the %s sector route %s returns %s; one entry dict (with 'crossings' or 'route') and one boolean end mask are required; post its return line" % (
            species, getattr(fn, "__name__", "?"), [type(x).__name__ + (str(getattr(x, "dtype", "")) and ("[%s]" % getattr(x, "dtype", ""))) for x in items]))
    return items, ient[0], imask[0]


def _v9c_recall(fn, sig, ba, names, mask_out, e1, U1):
    """the route re-called on the degenerate chord at x_n with the decided set: the reference assembled in v9b's own form"""
    ba2 = sig.bind(*ba.args, **ba.kwargs); ba2.apply_defaults()
    A = ba2.arguments
    A[names["h0"]] = A[names["h1"]]
    for key0, key1 in (("e0", "e1"), ("U0", "U1")):
        if names.get(key0) in A and names.get(key1) in A:
            A[names[key0]] = A[names[key1]]
    A[names["mask0"]] = mask_out
    if names.get("anchors0") in A:
        A[names["anchors0"]] = v9c_anchors(e1, mask_out)
    if names.get("vprev") in A:
        A[names["vprev"]] = _v9c_cols(U1, _v9c_np.flatnonzero(_v9c_np.asarray(_v9c_to_np(mask_out), bool)))
    return fn(*ba2.args, **ba2.kwargs)


def v9c_passages(species, fn, sig, ba, out):
    """the lz-mode work on one route call: the decision, the re-call where the set changed, the entry's annotation"""
    names = V9C_NAMES[species]; A = ba.arguments
    missing = [key for key in V9C_REQUIRED if names.get(key) not in A]
    if missing:
        raise V9cContract("v9c prim: the %s sector route %s does not bind %s; its signature is %s; set V9C_NAMES[%r] in build_engine_twofield_v9c.py (the lane posts the def line) and rebuild" % (
            species, getattr(fn, "__name__", "?"), ", ".join("%r (V9C_NAMES %r)" % (names.get(key), key) for key in missing), str(sig), species))
    items, ie, im = _v9c_split_return(out, species, fn)
    ent = items[ie]; mask_walk = items[im]
    args = A.get(names.get("args")); M = v9c_mode(args)
    tau = A.get(names.get("tau"))
    if tau is None:
        tau = getattr(args, "tau", None)
    if tau is None:
        raise V9cContract("v9c prim: the record's tau is neither among the %s route's arguments (V9C_NAMES tau) nor args.tau; post the signature" % species)
    channel = A.get(names.get("channel"), species)
    floor = A.get(names.get("floor"))
    floor = V9C_FLOOR if floor is None else floor
    mask_out, info = v9c_decide(species, channel, A[names["h0"]], A[names["h1"]], A.get(names.get("e0")), A.get(names.get("U0")), A[names["e1"]], A[names["U1"]],
                                A[names["mask0"]], mask_walk, ent.get("crossings") or [], tau, floor=floor, mode=M)
    if info is None:
        return out
    if info["mask_changed"]:
        mask_like = _v9c_mask_like(mask_out, mask_walk)
        try:
            out2 = _v9c_recall(fn, sig, ba, names, mask_like, A[names["e1"]], A[names["U1"]])
        except Exception as exc:
            raise V9cContract("v9c prim: the re-call of the %s route on the degenerate chord at x_n failed (%s: %s); post the route's signature" % (species, type(exc).__name__, exc))
        items2, ie2, im2 = _v9c_split_return(out2, species, fn)
        m2 = _v9c_np.asarray(_v9c_to_np(items2[im2]), bool); mo = _v9c_np.asarray(_v9c_to_np(mask_out), bool)
        if not _v9c_np.array_equal(m2, mo):
            info["walk_consistent"] = False; info["flags"].append("the re-call at x_n returned a set other than the decided one (%d indices differ)" % int(_v9c_np.count_nonzero(m2 != mo)))
        if items2[ie2].get("crossings"):
            info["flags"].append("the re-call at x_n reported %d crossing(s): a pair degenerate at x_n" % len(items2[ie2]["crossings"]))
        info["recall"] = {"pieces": items2[ie2].get("pieces"), "cholesky": items2[ie2].get("cholesky"), "route": items2[ie2].get("route")}
        new_items = list(items2); new_items[ie2] = ent
        vprev = A.get(names.get("vprev"))
        if vprev is not None:
            wmin, nad = v9c_wmin(vprev, A[names["U1"]], mask_out)
            ent["min_occupied_weight_walk"] = ent.get("min_occupied_weight"); ent["min_occupied_weight"] = wmin
            if "adiabatic_at_cut" in ent:
                ent["adiabatic_at_cut_walk"] = ent["adiabatic_at_cut"]; ent["adiabatic_at_cut"] = nad
            info["wmin_recomputed"] = True
        else:
            info["wmin_recomputed"] = False
            info["flags"].append("the previous reference is not among the route's arguments (V9C_NAMES vprev): min_occupied_weight is the walk's value for its own set")
        out = tuple(new_items) if isinstance(out, tuple) else new_items
    v9c_annotate(ent, info, mask_out)
    return out


def v9c_wrap_walk(species, fn):
    """the sector route of one species wrapped by name: in floor mode the route's return passes through untouched (bitwise v9b); in lz mode
    v9c_passages runs on it"""
    sig = _v9c_inspect.signature(fn)
    names = V9C_NAMES[species]
    has_args = names.get("args") in sig.parameters

    def wrapped(*a, **k):
        out = fn(*a, **k)
        ba = None; args = None
        if V9C_MODE["anchor_passage"] == "lz" or has_args:
            try:
                ba = sig.bind(*a, **k); ba.apply_defaults(); args = ba.arguments.get(names.get("args"))
            except TypeError:
                ba = None
        if v9c_mode(args).get("anchor_passage", "floor") != "lz":
            return out
        if ba is None:
            raise V9cContract("v9c prim: the %s sector route's call could not be bound to its signature %s" % (species, str(sig)))
        return v9c_passages(species, fn, sig, ba, out)

    wrapped.__name__ = getattr(fn, "__name__", species); wrapped.__qualname__ = getattr(fn, "__qualname__", wrapped.__name__)
    wrapped.__doc__ = getattr(fn, "__doc__", None); wrapped.__wrapped__ = fn; wrapped.v9c_species = species
    return wrapped
# <<< v9c prim block'''

V9C_REBIND = r'''# >>> v9c prim rebind: the sector routes of both species rebound by name to their wrapper (v9b's functions kept as __wrapped__)
for _v9c_species, _v9c_name in (("light", V9C_LIGHT_DEF), ("heavy", V9C_HEAVY_DEF)):
    if _v9c_name in globals() and callable(globals()[_v9c_name]):
        globals()[_v9c_name] = v9c_wrap_walk(_v9c_species, globals()[_v9c_name])
    else:
        V9C_MISSING.append(_v9c_name)
# <<< v9c prim rebind'''


class BuildStop(Exception):
    def __init__(self, name, count, old):
        Exception.__init__(self, "replacement %s: the string occurs %d time(s), not once" % (name, count))
        self.name = name; self.count = count; self.old = old


class Rep(object):
    def __init__(self, name, old, new, soft=False):
        self.name = name; self.old = old; self.new = new; self.soft = soft


def mark(text):
    """every non-blank line without the mark gains it"""
    out = []
    for line in text.split("\n"):
        if line.strip() == "" or MARK in line:
            out.append(line)
        else:
            out.append(line + "  # " + MARK)
    return "\n".join(out)


def block_text(light_def=LIGHT_DEF_DEFAULT, heavy_def=HEAVY_DEF_DEFAULT):
    return V9C_BLOCK.replace("@LIGHT_DEF@", light_def).replace("@HEAVY_DEF@", heavy_def)


def rebind_text(light_def=LIGHT_DEF_DEFAULT, heavy_def=HEAVY_DEF_DEFAULT):
    return V9C_REBIND.replace("@LIGHT_DEF@", light_def).replace("@HEAVY_DEF@", heavy_def)


def load_block_namespace(light_def=LIGHT_DEF_DEFAULT, heavy_def=HEAVY_DEF_DEFAULT):
    """the layer executed on its own (the self-test's fixtures run on it without v9b)"""
    ns = {"__name__": "engine_twofield_v9c_block"}
    exec(compile(block_text(light_def, heavy_def), "<v9c prim block>", "exec"), ns)
    return ns


def replacements(light_def=LIGHT_DEF_DEFAULT, heavy_def=HEAVY_DEF_DEFAULT, old_args=OLD_ARGS):
    indent = old_args[:len(old_args) - len(old_args.lstrip())]
    head = old_args.strip()
    if ".add_argument(" not in head:
        raise ValueError("OLD_ARGS must be the first line of an add_argument call: %r" % old_args)
    var = head.split(".add_argument(")[0]
    new_args = indent + "v9c_add_arguments(%s)  # %s\n" % (var, MARK) + old_args
    light_old = "def %s(" % light_def
    return [Rep("version", OLD_VERSION, NEW_VERSION, soft=True),
            Rep("args", old_args, new_args),
            Rep("layer", light_old, mark(block_text(light_def, heavy_def)) + "\n\n\n" + light_old),
            Rep("rebind", MAIN_GUARD, mark(rebind_text(light_def, heavy_def)) + "\n\n\n" + MAIN_GUARD)]


def precheck(text, reps, reverse=False):
    """[(rep, count)] of the strings that do not occur exactly once (a soft one absent is allowed)"""
    bad = []
    for r in reps:
        s = r.new if reverse else r.old
        n = text.count(s)
        if n != 1 and not (r.soft and n == 0):
            bad.append((r, n))
    return bad


def apply(text, reps, reverse=False):
    bad = precheck(text, reps, reverse)
    if bad:
        r, n = bad[0]
        raise BuildStop(r.name, n, r.new if reverse else r.old)
    log = []
    for r in reps:
        old, new = (r.new, r.old) if reverse else (r.old, r.new)
        n = text.count(old)
        if n == 0 and r.soft:
            log.append("%s: skipped (soft; not present)" % r.name); continue
        text = text.replace(old, new, 1); log.append("%s: applied" % r.name)
    return text, log


def marks_check(src, out):
    """[(line number, line)] of the lines of out not in src that lack the mark (blank lines skipped)"""
    old = set(src.split("\n")); bad = []
    for q, line in enumerate(out.split("\n"), 1):
        if line.strip() == "" or line in old:
            continue
        if MARK not in line:
            bad.append((q, line))
    return bad


def header_names(text):
    m = re.search(r"\(light route (\S+), heavy route (\S+)\)", text)
    if m is None:
        return None
    return m.group(1), m.group(2)


def _def_lines(src):
    lines = src.split("\n"); out = []
    for q, line in enumerate(lines):
        if line.startswith("def "):
            sig = [line.rstrip()]; j = q
            while not sig[-1].rstrip().endswith(":") and j + 1 < len(lines) and len(sig) < 8:
                j += 1; sig.append(lines[j].rstrip())
            out.append((q + 1, "\n".join(sig)))
    return out


def probe(src, reps, light_def, heavy_def):
    print("probe of v9b (%s):" % md5(src))
    for r in reps:
        s = r.old
        print("  %-8s %-4s occurs %d time(s): %r" % (r.name, "soft" if r.soft else "hard", src.count(s), s if len(s) < 100 else s[:100] + "..."))
    print("  heavy route def line %r occurs %d time(s)" % ("def %s(" % heavy_def, src.count("def %s(" % heavy_def)))
    print("candidate lines:")
    for q, line in enumerate(src.split("\n"), 1):
        if "--cut-transport" in line and "add_argument" in line:
            print("  args     %6d: %s" % (q, line.rstrip()))
        if "VERSION" in line and ("v9b" in line or "v9" in line) and "=" in line:
            print("  version  %6d: %s" % (q, line.rstrip()))
    for q, sig in _def_lines(src):
        low = sig.lower()
        if any(key in low for key in ("sector", "walk", "weyl", "transport_references", "resolve_crossings", "continue")):
            print("  def      %6d: %s" % (q, sig.replace("\n", "\n                   ")))
    print("  main guard occurs %d time(s)" % src.count(MAIN_GUARD))
    print("the lane posts the lines of the hard anchors that do not occur once, and the two routes' signatures and return lines for V9C_NAMES")


def md5(text):
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def read(path):
    with open(path, "rb") as f:
        return f.read().decode("utf-8")


def write(path, text):
    with open(path, "wb") as f:
        f.write(text.encode("utf-8"))


def main(argv=None):
    ap = argparse.ArgumentParser(description="engine_twofield_v9c over engine_twofield_v9b (exact replacements; fails closed)")
    ap.add_argument("source", help="engine_twofield_v9b.py (or engine_twofield_v9c.py with --reverse)")
    ap.add_argument("out", nargs="?", default=None, help="the output path (default engine_twofield_v9c.py beside the source; with --reverse <source>.restored.py)")
    ap.add_argument("--light-def", default=LIGHT_DEF_DEFAULT, help="the light sector route's function name (default %(default)s)")
    ap.add_argument("--heavy-def", default=HEAVY_DEF_DEFAULT, help="the heavy sector route's function name (default %(default)s)")
    ap.add_argument("--probe", action="store_true", help="counts and candidate lines; nothing written")
    ap.add_argument("--check", action="store_true", help="the build in memory with the marks check and the reverse round trip; nothing written")
    ap.add_argument("--reverse", action="store_true", help="the reverse replacements: v9c -> v9b")
    ap.add_argument("--against", default=None, help="with --reverse: the v9b file the restored text must equal byte for byte")
    a = ap.parse_args(argv)
    src = read(a.source)
    if a.reverse:
        names = header_names(src) or (a.light_def, a.heavy_def)
        reps = replacements(names[0], names[1])
        try:
            back, log = apply(src, reps, reverse=True)
        except BuildStop as stop:
            print("STOP: %s (reverse)" % stop); return 2
        out = a.out or (os.path.splitext(a.source)[0] + ".restored.py")
        write(out, back)
        print("reverse: %s -> %s (md5 %s); %s" % (a.source, out, md5(back), "; ".join(log)))
        if a.against:
            ref = read(a.against)
            print("against %s: %s" % (a.against, "BYTE FOR BYTE" if ref == back else "DIFFERS (md5 %s)" % md5(ref)))
            return 0 if ref == back else 2
        return 0
    reps = replacements(a.light_def, a.heavy_def)
    if a.probe:
        probe(src, reps, a.light_def, a.heavy_def); return 0
    bad = precheck(src, reps)
    if bad or src.count("def %s(" % a.heavy_def) != 1:
        for r, n in bad:
            print("STOP: replacement %s: the OLD string occurs %d time(s), not once:\n%s" % (r.name, n, r.old))
        if src.count("def %s(" % a.heavy_def) != 1:
            print("STOP: the heavy route's def line %r occurs %d time(s); give --heavy-def" % ("def %s(" % a.heavy_def, src.count("def %s(" % a.heavy_def)))
        print("nothing written; the lane posts the lines named (python %s %s --probe lists the candidates)" % (os.path.basename(sys.argv[0]), a.source))
        return 2
    out_text, log = apply(src, reps)
    marks = marks_check(src, out_text)
    if marks:
        print("STOP: %d line(s) of v9c not in v9b lack the mark %r (a builder defect):" % (len(marks), MARK))
        for q, line in marks[:20]:
            print("  %6d: %s" % (q, line))
        return 2
    back, _ = apply(out_text, reps, reverse=True)
    if back != src:
        print("STOP: the reverse replacements do not restore v9b byte for byte (a builder defect)"); return 2
    print("build: %s (md5 %s) -> v9c (md5 %s); %s; marks ok; reverse byte for byte" % (a.source, md5(src), md5(out_text), "; ".join(log)))
    if a.check:
        print("CHECK PASS (nothing written)"); return 0
    out = a.out or os.path.join(os.path.dirname(os.path.abspath(a.source)), "engine_twofield_v9c.py")
    write(out, out_text)
    print("written: %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
