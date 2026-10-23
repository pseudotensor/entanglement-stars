#!/usr/bin/env python
"""build_engine_twofield_v9c1c.py: build_engine_twofield_v9c1b.py f275a637 (the v9c1b builder) with its three primitive-naming block lines changed,
each marked "[v9c1c x]" (busi builder, 20 Oct; REVIEWA_C64_toy_readout_v1.md section 1): the default primitive is assign_primitive_v1f (in
V9C_DEFAULTS and in argparse; build_assign_primitive_v1f.py: a last-piece passage whose extension gave no sample takes the two-sample grid bound) and
the import refusal names assign_primitive_v1f.py. The output (engine_twofield_v9c1c.py; staged under the name build_engine_twofield_v9c.py so the
self-test's layer runs this block) differs from engine_twofield_v9c1b.py only on lines carrying "[v9c1c x]" (V9C1C_LINES below: restoring those
lines gives v9c1b byte for byte, --to-v9c1b); over v9b it reverses byte for byte as v9c1 and v9c1b do. The v9c1b docstring follows.

build_engine_twofield_v9c1b.py: build_engine_twofield_v9c.py 8fe64350 (the v9c1 builder) with three marked block edits "[v9c1b x]" (busi builder,
20 Oct; ReviewA_busi_open_q12_v1.md section 1 and the late-step toy toy_late_step_v1.py): the default primitive is assign_primitive_v1e (the
last-piece passage takes its extension sample before the anchor's; build_assign_primitive_v1e.py), every primitive entry carries the pair's
(s, Delta) samples the fit used ("samples"), and the import refusal no longer names v1d. The output (engine_twofield_v9c1b.py; staged under the
name build_engine_twofield_v9c.py so the self-test's layer runs this block) differs from engine_twofield_v9c1.py only on lines carrying
"[v9c1b x]"; over v9b it reverses byte for byte as v9c1 does. The original docstring follows.

build_engine_twofield_v9c.py (reviewer A for the busi lane; C64_v9c_ruling_v1.md section 2; C64_G4s8_hold_ruling_v1 (4) and (4b); PRIM_ruling_v1 3.4;
rewritten against the built engine_twofield_v9b.py, md5 301e9dfc0f38f98bf9f528713ea6bf44, after the first builder, written without the source, did not
apply): engine_twofield_v9c.py over engine_twofield_v9b.py in the lane's patch form: exact replacements, each OLD occurring once in v9b (verbatim lines of
v9b); every line of v9c not in v9b carries "v9c prim"; --reverse restores v9b byte for byte; the defaults are bitwise v9b.

What v9c does. One run argument --anchor-passage {floor, lz} (default floor). floor: the two sector routes are called untouched (one dict lookup in a
wrapper), so every number of v9b is reproduced in both --cut-transport modes. lz: after v9b's route has certified a channel's path and assembled its
result, the assignment primitive (--prim-module, default assign_primitive_v1d; its VERSION is written into every sector entry the routes return)
follows the channel's occupied set from the stored mask along the record's straight path (the engine's own eigendecompositions path.eig(s) at the
primitive's dyadic points, h linear in the fields, so K(s) = (1 - s) h(x_(n-1)) + s h(x_n); the trial's own at s = 1) with Rule lz (tau_LZ = k gap_min^2 /
|alpha_T| against c, k = 1, c = 1, alpha_T = alpha_s / tau), levels 1 to --prim-levels (4), w_mix 0.9 and the charge-conjugation mirror read off the
spectrum (the primitive's mirror "auto"). v9b's contract, read off its source: the light route works in the S = +1 sector's sorted order (occp =
mask0[order0], one anchor per boundary, the S = -1 sector the conjugate mirror, slaved by the mirror rule at x_n), so the primitive follows the S = +1
sector's sorted levels and the walk's own occupation there (the - sector's levels never enter: an occupied + level passing an empty - level is a
sorted-index swap between decoupled sectors, not an anchor passage, and the mirror rule carries it); the heavy route works over every level. One
spectrum at s = 1/2 is the trigger: a channel whose set is clean at 1/2 and at 1 on its start indices (the set's eigenvectors at x_(n-1): the previous
reference, which the light caller stashes for the wrapper and the heavy route receives, projected on the spectra) has no passage on the chord and costs
one eigendecomposition; a channel whose boundary pairs are wider than --lz-skip-gap (2e-2) at both ends, whose adjacent levels move by less than
half of the smaller splitting, and whose previous reference is clean on its start indices in the trial's own eigenbasis (an exchange of two levels
leaves the sorted spectrum's ends alike; this catches it at no eigendecomposition) costs none. Every completed or incomplete passage becomes an entry
of the channel's "crossings" (the walk's key names kept: s, piece, probes, overlap (the pairing's character weight: the member's weight on its new
index on a completed swap; None on an adiabatic or incomplete entry, whose character weight is recorded beside), inversion_delta, swap, block (the
pair's size), flagged, flagged_why; plus branch, tau_lz with its band, the
Landau-Zener exponent pi tau_LZ / (2 k), gap_min, alpha_s, alpha_T, s_star, the pair's indices at both ends, the jumps (grid, fit, alt) and whether
they fall under the halving); approaches and tails (the fitted minimum beyond or before the chord) are entries without a decision, so a record whose
end lies inside a window carries an entry of its own; blocks and undecided or knife-edge passages are flagged. A sub-floor pair keeps v9b's crossing
phase (the swap is 9b's) and its entry is annotated by the matched event (PRIM 3.4 by the chord rather than the probes); a walk swap without an event,
or against an adiabatic decision, is flagged. Where the decided set at s = 1 differs from the walk's (a sub-threshold minimum decided diabatic: G4s8's
six passages) the route is re-called on the degenerate chord at x_n (V9cEndPath: h_host, eig and the fields at s = 1 for every s; no field change, no
piece) with the decided set (the light: the full mask from the decided set's anchors by the mirror rule, light_select_sector_v9), so the end
bookkeeping (anchors_end, anchors, inversions, the pairs within the floor at the end) is assembled in v9b's own form, the walk's certificate and
counters stay the walk's, and v9b's callers then build the reference from the returned anchors_end (the light) or occ_end (the heavy) exactly as for
the walk's own set: min_occupied_weight, the cut check and adiabatic_at_cut are recomputed against the previous reference by v9b's own statements. The
gate's and the watch's read of a channel is v9c_hold_reasons(sector entry); the engine prints one primitive line per channel that had an entry.

The patch. R1 version (soft; skipped with a notice when absent, as in v9b): ENGINE_VERSION = "engine_twofield_v9b" -> "engine_twofield_v9c". R2 args
(hard): v9b's --cut-transport add_argument line (one line, with its trailing "# v9 sector" comment) gains, before it, v9c_add_arguments(ap). R3 resume
(hard): v9b's two-line --cut-transport resume check in light_resume_v36 gains, after it, the same check on --anchor-passage and the lz arguments. R4
layer (hard): the block V9C_BLOCK is inserted before def light_sector_walk_v9( (--light-def). R5 rebind (hard): before the main guard the two sector
routes (--light-def, --heavy-def, default heavy_sector_walk_v9) are rebound by name to their wrapper, the light caller (--light-caller, default
light_continue_sector_v9) to the wrapper that stashes the previous reference, and run (--run-def) to the wrapper that reads the mode off the parsed
namespace before the run (the routes have no args of their own); a name not defined is listed in V9C_MISSING and --anchor-passage lz then refuses to
start, naming it.

Usage. python build_engine_twofield_v9c.py engine_twofield_v9b.py [engine_twofield_v9c.py] [--light-def NAME] [--heavy-def NAME] [--light-caller NAME] [--run-def NAME]
         --probe: the counts of the OLD strings in v9b and the candidate lines (nothing written)
         --check: the build in memory, the marks check and the reverse round trip (nothing written)
         --reverse engine_twofield_v9c.py [restored.py] [--against engine_twofield_v9b.py]: the reverse replacements, byte for byte
The builder stops without writing when any hard OLD string does not occur exactly once, naming it and the lines to post; the result is compiled before
it is written.
"""
import argparse
import hashlib
import os
import re
import sys

MARK = "v9c prim"
LIGHT_DEF_DEFAULT = "light_sector_walk_v9"
HEAVY_DEF_DEFAULT = "heavy_sector_walk_v9"
LIGHT_CALLER_DEFAULT = "light_continue_sector_v9"
RUN_DEF_DEFAULT = "run"
MAIN_GUARD = 'if __name__ == "__main__":'
OLD_VERSION = 'ENGINE_VERSION = "engine_twofield_v9b"'
NEW_VERSION = 'ENGINE_VERSION = "engine_twofield_v9c"  # v9c prim'


def _s(x):
    """the snippets below start on the line after the opening quotes: that first newline is not part of them."""
    return x[1:] if x.startswith("\n") else x


# v9b line 2771, verbatim (main(): the add_argument call of --cut-transport, one line)
OLD_ARGS = _s(r'''
    ap.add_argument("--cut-transport", dest="cut_transport", default="record", choices=["record", "sector"], help="(v9 sector, 9b) record: version 8 bitwise in both --reference-solve modes; sector: past a closed occupation cut the reference continued by the sector walk, the declared convention applied to each crossing at the scale of the crossing (the piece bracketed by certified probe points), the walk certified at every boundary of the occupied set, the record-scale convention the flagged fallback (busi/notes/C64_transport_ruling_v1.md section 4; busi/notes/C64_v9_fix_v1.md)")   # v9 sector''')

# v9b lines 1256 to 1257, verbatim (light_resume_v36(): the resume check of --cut-transport, a single if)
OLD_RESUME = _s(r'''
    if str(J["args"].get("cut_transport", "record")) != str(getattr(a, "cut_transport", "record")):      # v9 sector: the resume check gains the transport (4.5); a run without the argument (version 8's) is the record transport
        raise SystemExit("v9 sector: the run's --cut-transport is %r, this call's %r: a resume at a transport change goes through the launcher (pod_launch_C_v12b.sh rewrites the snapshot copy's args.cut_transport and keeps the old value as args.cut_transport_before)" % (J["args"].get("cut_transport", "record"), getattr(a, "cut_transport", "record")))   # v9 sector''')

RESUME_NEW_TAIL = _s(r'''
    for _v9c_key in V9C_RESUME_KEYS:
        if str(J["args"].get(_v9c_key, V9C_DEFAULTS[_v9c_key])) != str(getattr(a, _v9c_key, V9C_DEFAULTS[_v9c_key])):
            raise SystemExit("v9c prim: the run's %s is %r, this call's %r: a resume at an anchor-passage change goes through the launcher (pod_launch_C_v12c.sh rewrites the snapshot copy's args.anchor_passage and keeps the old value as args.anchor_passage_before)" % (_v9c_key, J["args"].get(_v9c_key, V9C_DEFAULTS[_v9c_key]), getattr(a, _v9c_key, V9C_DEFAULTS[_v9c_key])))''')

V9C_BLOCK = r'''# >>> v9c prim block (light route @LIGHT_DEF@, heavy route @HEAVY_DEF@, light caller @LIGHT_CALLER@, run @RUN_DEF@): the assignment primitive at every anchor passage of the sector walk;
# build_engine_twofield_v9c.py over engine_twofield_v9b.py; C64_v9c_ruling_v1 section 2, C64_G4s8_hold_ruling_v1 (4) and (4b), PRIM_ruling_v1 3.4.
# Defaults bitwise v9b: with --anchor-passage floor (the default) the routes are called untouched and nothing below runs beyond one dict lookup; with lz
# each certified channel's occupied set is followed along the record's straight path by the primitive named by --prim-module (the S = +1 sector's sorted
# levels on the light, the walk's own order, the S = -1 sector slaved by the mirror rule as in v9b; every level on the heavy) and every anchor passage is
# decided by its Landau-Zener rule; where the decided set differs from the walk's, the route is re-called on the degenerate chord at x_n with that set.
import inspect as _v9c_inspect
import importlib as _v9c_importlib
import math as _v9c_math
import numpy as _v9c_np

V9C_VERSION = "engine_twofield_v9c"
V9C_LIGHT_DEF = "@LIGHT_DEF@"
V9C_HEAVY_DEF = "@HEAVY_DEF@"
V9C_LIGHT_CALLER = "@LIGHT_CALLER@"
V9C_RUN_DEF = "@RUN_DEF@"
V9C_DEFAULTS = {"anchor_passage": "floor", "prim_module": "assign_primitive_v1f", "lz_c": 1.0, "lz_k": 1.0, "lz_tie": "adiabatic", "lz_knife": 0.1,  # [v9c1c x] v1f
                "prim_levels": 4, "prim_w_mix": 0.9, "lz_skip_gap": 2e-2}
V9C_RESUME_KEYS = tuple(V9C_DEFAULTS)
V9C_MODE = dict(V9C_DEFAULTS)
V9C_CTX = {"tau": None, "light_ref": None}
V9C_NAMES = {"light": {"path": "path", "e0": "e0_np", "sig0": "sig0", "mask0": "mask0", "trial": "trial", "d_min": "d_min", "tol_block": "tol_block", "mu_prev": "mu_prev"},
             "heavy": {"path": "path", "prev_ref": "prev_ref", "e0": "e0_np", "mask0": "mask0", "trial": "trial", "d_min": "d_min", "tol_block": "tol_block"}}
V9C_REQUIRED = {"light": ("path", "e0", "sig0", "mask0", "trial"), "heavy": ("path", "e0", "mask0", "trial")}
V9C_MISSING = []
V9C_FLOOR = 3.5e-7
V9C_INFO_FLAGS = ("exact crossing to this fit", "third sample at s=", "no set change, no event", "two samples only", "refused by the oracle", "continuation unclear there")
_V9C_PRIM = {}


class V9cContract(RuntimeError):
    """v9b's sector routes, their callers or their arguments do not meet the wrapper's contract; the message names what the lane posts"""


def v9c_add_arguments(ap):
    """the v9c arguments (the builder's R2 places this call before --cut-transport); every default reproduces v9b; the run's wrapper reads them off the parsed namespace"""
    ap.add_argument("--anchor-passage", dest="anchor_passage", choices=["floor", "lz"], default="floor",
                    help="(v9c prim) floor: v9b, a certified passage keeps the adiabatic branch and the floor is the only threshold; lz: every anchor passage of the sector walk decided by the Landau-Zener rule of the assignment primitive along the record's straight path (C64_v9c_ruling_v1 section 2)")
    ap.add_argument("--prim-module", dest="prim_module", default="assign_primitive_v1f", help="(v9c prim) the assignment primitive (assign_primitive_v1d or its successor), beside the engine; its VERSION is recorded in every sector entry")  # [v9c1c x] v1f
    ap.add_argument("--lz-c", dest="lz_c", type=float, default=1.0, help="(v9c prim) c_LZ: adiabatic when tau_LZ = k gap_min^2 / |alpha_T| >= c on the whole band (the record convention's c_LZ = 1)")
    ap.add_argument("--lz-k", dest="lz_k", type=float, default=1.0, help="(v9c prim) k of tau_LZ (1: the lane's reading of the record line; pi/2 makes tau_LZ the Landau-Zener exponent)")
    ap.add_argument("--lz-tie", dest="lz_tie", choices=["adiabatic", "diabatic"], default="adiabatic", help="(v9c prim) the branch when the band on tau_LZ still contains c at the deepest level (the passage is then flagged undecided: a HOLD)")
    ap.add_argument("--lz-knife", dest="lz_knife", type=float, default=0.1, help="(v9c prim) a decided passage with |ln(tau_LZ / c)| below this is flagged knife-edge")
    ap.add_argument("--prim-levels", dest="prim_levels", type=int, default=4, help="(v9c prim) the primitive's deepest halving level (levels 1 .. this; stability needs two levels, so at least 2)")
    ap.add_argument("--prim-w-mix", dest="prim_w_mix", type=float, default=0.9, help="(v9c prim) the primitive's clean-point weight (a point with a set member below it lies inside a crossing's window)")
    ap.add_argument("--lz-skip-gap", dest="lz_skip_gap", type=float, default=2e-2, help="(v9c prim) a channel whose boundary pairs are wider than this at both ends of the record, whose adjacent levels move by less than half the smaller splitting, and whose previous reference is clean on its start indices in the trial's eigenbasis is not examined (no passage can lie inside; no eigendecomposition); 0 turns the skip off")


def v9c_mode(args=None):
    """the mode: V9C_MODE (the run's parsed namespace through v9c_mode_set) overridden by a namespace where it carries the keys"""
    M = dict(V9C_MODE)
    if args is not None:
        for key in V9C_DEFAULTS:
            if hasattr(args, key):
                M[key] = getattr(args, key)
    return M


def v9c_prim(name=None):
    """the primitive module, imported once by name (beside the engine, or on sys.path)"""
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


def v9c_mode_set(a):
    """the mode from the parsed namespace (the run's wrapper, before run(a)): V9C_MODE and the record's tau in V9C_CTX; --anchor-passage lz checked at the start (sector mode, the levels, the primitive's import, the routes present)"""
    for key, val in V9C_DEFAULTS.items():
        V9C_MODE[key] = getattr(a, key, val)
    V9C_CTX["tau"] = getattr(a, "tau", None)
    if V9C_MODE["anchor_passage"] == "lz":
        if getattr(a, "cut_transport", "record") != "sector":
            raise SystemExit("v9c prim: --anchor-passage lz decides the sector walk's anchor passages: --cut-transport sector is required")
        if int(V9C_MODE["prim_levels"]) < 2:
            raise SystemExit("v9c prim: --prim-levels must be at least 2 (stability under halving needs two levels)")
        if not (0.5 < float(V9C_MODE["prim_w_mix"]) < 1.0):
            raise SystemExit("v9c prim: --prim-w-mix must lie in (0.5, 1)")
        if V9C_CTX["tau"] is None or not (float(V9C_CTX["tau"]) > 0.0):
            raise SystemExit("v9c prim: the record's tau (--tau) must be positive: alpha_T = alpha_s / tau")
        if V9C_MISSING:
            raise SystemExit("v9c prim: --anchor-passage lz: %s; post the def lines and rebuild with --light-def / --heavy-def / --light-caller / --run-def" % "; ".join(V9C_MISSING))
        try:
            prim = v9c_prim(V9C_MODE["prim_module"]); rule = v9c_rule(V9C_MODE)
        except Exception as exc:
            raise SystemExit("v9c prim: the primitive %r cannot be imported or its Rule built (%s: %s); assign_primitive_v1f.py belongs beside the engine" % (V9C_MODE["prim_module"], type(exc).__name__, exc))  # [v9c1c x]
        a.anchor_passage_engine = V9C_VERSION; a.anchor_passage_prim = prim.VERSION   # C64_v9c_review_v1 finding 5: an lz run's json names the engine and the primitive in its args (the json's script string stays v9b's, which the floor mode's bitwise identity needs); floor mode sets nothing
        print("v9c prim: --anchor-passage lz: %s at every anchor passage of the sector walk (%s); tau %g, levels 1 to %d, w_mix %g, skip gap %g" % (
            prim.VERSION, rule.describe(), float(V9C_CTX["tau"]), int(V9C_MODE["prim_levels"]), float(V9C_MODE["prim_w_mix"]), float(V9C_MODE["lz_skip_gap"])), flush=True)
    return dict(V9C_MODE)


def v9c_wrap_run(fn):
    """run(a) wrapped: the mode read off the parsed namespace first (the sector routes have no args of their own)"""
    def wrapped(a, *rest, **kw):
        v9c_mode_set(a)
        return fn(a, *rest, **kw)
    wrapped.__name__ = getattr(fn, "__name__", "run"); wrapped.__doc__ = getattr(fn, "__doc__", None); wrapped.__wrapped__ = fn
    return wrapped


def _v9c_is_torch(x):
    return hasattr(x, "device") and hasattr(x, "detach") and hasattr(x, "cpu")


def _v9c_to_np(x):
    if _v9c_is_torch(x):
        return x.detach().cpu().numpy()
    return _v9c_np.asarray(x)


def _v9c_index(U, idx):
    """the columns idx of U (numpy, or torch on its own device)"""
    idx = _v9c_np.asarray(idx, dtype=_v9c_np.int64).reshape(-1)
    if _v9c_is_torch(U):
        import torch
        return U[:, torch.as_tensor(idx, device=U.device)]
    return _v9c_np.asarray(U)[:, idx]


def _v9c_gram(A, B):
    """|A^H B|^2 as a numpy float array, for numpy or torch columns (a numpy operand is moved beside a torch one)"""
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


def _v9c_park(U):
    """a torch eigenbasis moved to the host between the primitive's levels (the device holds only what the primitive is using)"""
    if _v9c_is_torch(U) and str(U.device) != "cpu":
        return U.detach().cpu()
    return U


def _v9c_unpark(U, dev):
    if _v9c_is_torch(U) and dev is not None and str(U.device) != str(dev):
        return U.to(dev)
    return U


class V9cPlus(object):
    """the primitive's eigenvector object for the light: the whole eigenbasis U (numpy or torch, untouched) with the S = +1 sector's indices in energy order; the columns are taken through the order and the weights are gathered after the product"""

    def __init__(self, U, order):
        self.U = U; self.order = _v9c_np.asarray(order, dtype=_v9c_np.int64)

    def cols(self, cols):
        return _v9c_index(self.U, self.order[_v9c_np.asarray(cols, dtype=_v9c_np.int64).reshape(-1)])

    def full(self):
        return _v9c_index(self.U, self.order)

    def park(self):
        return V9cPlus(_v9c_park(self.U), self.order)

    def to(self, dev):
        return V9cPlus(_v9c_unpark(self.U, dev), self.order)


def _v9c_columns(U, cols):
    if isinstance(U, V9cPlus):
        return U.cols(cols)
    return _v9c_index(U, cols)


def _v9c_weights(A, B):
    if isinstance(A, V9cPlus):
        A = A.full()
    if isinstance(B, V9cPlus):
        return _v9c_gram(A, B.U)[:, B.order]
    return _v9c_gram(A, B)


def _v9c_plus_order(e, sig):
    """the S = +1 sector's levels in energy order (light_plus_order_v9's rule): (their indices into the level list, their energies)"""
    e = _v9c_np.asarray(_v9c_to_np(e), float); sel = _v9c_np.flatnonzero(_v9c_np.asarray(_v9c_to_np(sig), float) > 0.0)
    order = sel[_v9c_np.argsort(e[sel], kind="stable")]
    return order, e[order]


class V9cSpectra(object):
    """the record's straight path as the primitive sees it: spectrum(s) -> (e, U) with e the sorted eigenvalues (the S = +1 sector's on the light) as numpy and U the eigenvector object (a V9cPlus on the light): the engine's own eigendecompositions, at s = 1 the trial (given), at s = 0 path.eig(0) on first use, elsewhere path.eig(s) (counted in path.eigh_n by the path and in n_eigh here), parked on the host between the primitive's levels; s beyond [0, 1] (the primitive's extension by one piece) from the engine's paths' fields, or from a path that declares v9c_extends, refused otherwise (the primitive then keeps its two-sample bound)"""

    def __init__(self, path, light, e0, sig0=None, trial=None):
        self.path = path; self.light = bool(light); self.n_eigh = 0; self.cache = {}; self.parked = {}; self.order0 = None; self.order1 = None; self.ext_verified = None
        if self.light:
            if not (isinstance(trial, (tuple, list)) and len(trial) == 3):
                raise V9cContract("v9c prim: the light route's trial is not (e1, sig1, U1); post its signature")
            e1, sig1, U1 = trial
            self.order0, self.e0 = _v9c_plus_order(e0, sig0); self.order1, self.e1 = _v9c_plus_order(e1, sig1)
            self.cache[1.0] = (self.e1, V9cPlus(U1, self.order1))
        else:
            if not (isinstance(trial, (tuple, list)) and len(trial) == 2):
                raise V9cContract("v9c prim: the heavy route's trial is not (e1, U1); post its signature")
            e1, U1 = trial
            self.e0 = _v9c_np.asarray(_v9c_to_np(e0), float); self.e1 = _v9c_np.asarray(_v9c_to_np(e1), float)
            self.cache[1.0] = (self.e1, U1)
        self.dev = getattr(U1, "device", None)
        self.N = int(self.e0.size)
        if int(self.e1.size) != self.N:
            raise V9cContract("v9c prim: the level counts differ between the window's spectrum (%d) and the trial's (%d)" % (self.N, int(self.e1.size)))

    def _extend(self, s):
        """s outside [0, 1] on the engine's paths (LightPathV9, HeavyPathV9: their fields linear in s), as their eig computes it at the extrapolated fields"""
        p = self.path; G = globals()
        f0 = getattr(p, "f0", None); f1 = getattr(p, "f1", None); ch = getattr(p, "ch", None)
        if f0 is None or f1 is None or ch is None:
            raise ValueError("v9c prim: the path has no fields to extend the chord to s = %.4f" % s)
        N0, v0 = _v9c_np.asarray(f0[0], float), _v9c_np.asarray(f0[1], float); N1, v1 = _v9c_np.asarray(f1[0], float), _v9c_np.asarray(f1[1], float)
        N, v = N0 + s * (N1 - N0), v0 + s * (v1 - v0)
        if self.light:
            eigT_, gpu_, sect_, se_ = G.get("eigT"), G.get("clocked_gpu"), G.get("light_sectorize_v36"), G.get("sector_e_v9")
            if eigT_ is None or gpu_ is None or sect_ is None or se_ is None:
                raise ValueError("v9c prim: the engine's light eigendecomposition is not in scope for the extension to s = %.4f" % s)
            e, U = eigT_(gpu_(ch, N, v)); p.eigh_n = int(getattr(p, "eigh_n", 0)) + 1
            sigma, _n_rot, _imp = sect_(e, U, p.sgn_dev, p.tol_block)
            return e.cpu().numpy().astype(float), se_(sigma), U
        eig_at_, se_ = G.get("eig_at"), G.get("sector_e_v9")
        if eig_at_ is None or se_ is None:
            raise ValueError("v9c prim: the engine's heavy eigendecomposition is not in scope for the extension to s = %.4f" % s)
        e, U = eig_at_(ch, N, v); p.eigh_n = int(getattr(p, "eigh_n", 0)) + 1
        return se_(e), U

    def _verify_extension(self):
        """(C64_v9c_review_v1 finding 9) before the first extension of a spectra object, the extrapolation at s = 1 must reproduce the trial's eigenvalues: the field convention (f0, f1, ch) and the engine's eigendecomposition checked on the path itself; True, else the mismatch as text and every extension of this object is refused (the primitive then keeps its two-sample bound and flags it as information)"""
        try:
            out = self._extend(1.0)
        except Exception as exc:
            return "%s: %s" % (type(exc).__name__, exc)
        self.n_eigh += 1
        if self.light:
            e, sig, _U = out; ep = _v9c_plus_order(e, sig)[1]
        else:
            e, _U = out; ep = _v9c_np.asarray(_v9c_to_np(e), float)
        if int(ep.size) != self.N:
            return "the extrapolation at s = 1 has %d levels against the trial's %d" % (int(ep.size), self.N)
        scale = max(1.0, float(_v9c_np.max(_v9c_np.abs(self.e1)))); dev = float(_v9c_np.max(_v9c_np.abs(ep - self.e1)))
        if not (dev <= 1e-8 * scale):
            return "the extrapolation at s = 1 differs from the trial's eigenvalues by %.3e (scale %.3g): the path's fields are not the chord's" % (dev, scale)
        return True

    def _eig(self, s):
        if 0.0 <= s <= 1.0 or getattr(self.path, "v9c_extends", False):
            out = self.path.eig(s)
        else:
            if self.ext_verified is None:
                self.ext_verified = self._verify_extension()
            if self.ext_verified is not True:
                raise ValueError("v9c prim: the chord's extension to s = %.4f refused: %s" % (s, self.ext_verified))
            out = self._extend(s)
        self.n_eigh += 1
        if self.light:
            e, sig, U = out; order, ep = _v9c_plus_order(e, sig)
            if int(ep.size) != self.N:
                raise V9cContract("v9c prim: the S = +1 sector count changes along the path (%d at s %.6f against %d)" % (int(ep.size), s, self.N))
            return ep, V9cPlus(U, order)
        e, U = out; e = _v9c_np.asarray(_v9c_to_np(e), float)
        if int(e.size) != self.N:
            raise V9cContract("v9c prim: the level count changes along the path (%d at s %.6f against %d)" % (int(e.size), s, self.N))
        return e, U

    def spectrum(self, s):
        s = float(s)
        if s in self.cache:
            return self.cache[s]
        if s in self.parked:
            e, Uh = self.parked[s]
            return e, (Uh.to(self.dev) if isinstance(Uh, V9cPlus) else _v9c_unpark(Uh, self.dev))
        e, U = self._eig(s)
        if s == 0.0:
            self.cache[0.0] = (e, U)
        else:
            self.parked[s] = (e, (U.park() if isinstance(U, V9cPlus) else _v9c_park(U)))
        return e, U


class V9cOracle(object):
    """the primitive's oracle protocol over a V9cSpectra (or any object with spectrum(s) -> (e, U))"""

    def __init__(self, spec):
        self.spec = spec

    def spectrum(self, s, channels):
        e, U = self.spec.spectrum(s)
        return {ch: (e, U) for ch in channels}

    def columns(self, U, cols):
        return _v9c_columns(U, cols)

    def weights(self, A, B):
        return _v9c_weights(A, B)

    def project(self, A, B):
        return _v9c_weights(A, B).sum(axis=0)

    def release(self, U):
        return None


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
    """the indices j with exactly one of j, j + 1 occupied (boundaries_v9's rule)"""
    m = _v9c_np.asarray(_v9c_to_np(mask), bool)
    return _v9c_np.flatnonzero(m[:-1] != m[1:])


def v9c_anchors(e, mask):
    """the anchors as midpoints a_j = (e_j + e_(j+1)) / 2 of the mask's boundaries (midgaps_v9's rule)"""
    e = _v9c_np.asarray(_v9c_to_np(e), float)
    return [float(0.5 * (e[j] + e[j + 1])) for j in v9c_boundaries(mask)]


def v9c_inversions(mask):
    """the occupied levels above the lowest empty one (inversions_run_v9's rule)"""
    m = _v9c_np.asarray(_v9c_to_np(mask), bool)
    if not m.any() or m.all():
        return 0
    return int(_v9c_np.sum(m[int(_v9c_np.argmax(~m)):]))


def _v9c_clean_at(ref, U, m, w_mix):
    """the reference's character on the spectrum U is clean (every member's weight at least w_mix, every other level's at most 1 - w_mix) and on the indices m"""
    p = _v9c_weights(ref, U).sum(axis=0); n = int(_v9c_np.count_nonzero(m))
    order = _v9c_np.argsort(-p, kind="stable"); top = _v9c_np.zeros(int(m.size), bool); top[order[:n]] = True
    return bool(float(p[top].min()) >= float(w_mix) and float(p[~top].max()) <= 1.0 - float(w_mix) and _v9c_np.array_equal(top, m))


def v9c_skip(spec, mask, skip_gap, ref0=None, w_mix=0.9):
    """True when no passage of the set can lie on the chord and none completed, at no eigendecomposition: skip_gap positive (0 turns the skip off), at every boundary the pair's splitting at both ends exceeds skip_gap and neither level adjacent to the boundary moves by half the smaller splitting or more over the record (the diabats cannot meet inside), and the previous reference's character on the trial's own eigenbasis (s = 1) is clean on the start indices (an exchange of two levels leaves the sorted spectrum's ends alike and is caught here); without the previous reference no skip"""
    if not (float(skip_gap) > 0.0) or ref0 is None:
        return False
    m = _v9c_np.asarray(_v9c_to_np(mask), bool); n = int(m.sum()); N = int(m.size)
    if n == 0 or n == N:
        return True
    if int(getattr(ref0, "shape", (0, 0))[1]) != n:
        return False
    e0 = _v9c_np.asarray(spec.e0, float); e1 = _v9c_np.asarray(spec.e1, float); J = v9c_boundaries(m)
    for j in J:
        d0 = float(e0[j + 1] - e0[j]); d1 = float(e1[j + 1] - e1[j]); dmin = min(d0, d1)
        if dmin <= float(skip_gap):
            return False
        move = max(abs(float(e1[j] - e0[j])), abs(float(e1[j + 1] - e0[j + 1])))
        if move >= 0.5 * dmin:
            return False
    e1_, U1 = spec.spectrum(1.0)
    return _v9c_clean_at(ref0, U1, m, w_mix)


def v9c_precheck(spec, mask0, w_mix, ref0=None):
    """the trigger: with the set's eigenvectors at s = 0 as the reference (ref0 when given: the previous reference's columns of the set; else the set's columns of the spectrum at 0), its character at s = 1/2 and at s = 1 is clean (every member's weight at least w_mix, every other level's at most 1 - w_mix) and on the start indices: no passage lies on the chord. One spectrum beyond the record's two."""
    m = _v9c_np.asarray(_v9c_to_np(mask0), bool); occ = _v9c_np.flatnonzero(m); n = int(occ.size); N = int(m.size)
    if n == 0 or n == N:
        return True
    ref = ref0
    if ref is not None and int(getattr(ref, "shape", (0, 0))[1]) != n:
        ref = None
    if ref is None:
        e0, U0 = spec.spectrum(0.0)
        ref = _v9c_columns(U0, occ)
    for s in (0.5, 1.0):
        e, U = spec.spectrum(s)
        if not _v9c_clean_at(ref, U, m, w_mix):
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


def _v9c_holding(flags):
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


def _v9c_pair_energies(e1, pair):
    if e1 is None or pair is None or any(x is None for x in pair):
        return None, None
    lo, hi = sorted(int(x) for x in pair)
    if lo < 0 or hi >= int(_v9c_np.asarray(e1).size):
        return None, None
    return [float(e1[lo]), float(e1[hi])], float(e1[hi] - e1[lo])


def _v9c_entry_from_event(ev, M, m_run, e1):
    """one entry of "crossings" from one event of the primitive, in the walk's key names; returns (entry, the running set after it)"""
    k = float(M["lz_k"]); decision = ev.get("decision"); diabatic = decision == "diabatic"
    unresolved_block = bool(ev.get("block"))
    kind = "block" if unresolved_block else ev.get("kind", "passage")
    holding = _v9c_holding(ev.get("flags"))
    flagged = bool(holding) or not (ev.get("resolved") and ev.get("decided")) or bool(ev.get("knife")) or bool(ev.get("tie")) or unresolved_block
    inv_before = v9c_inversions(m_run)
    m_after = _v9c_apply_swap(m_run, ev) if (diabatic and not unresolved_block) else _v9c_np.asarray(m_run, bool).copy()
    inv_after = v9c_inversions(m_after)
    pair_end = [ev.get("partner_c"), ev.get("n")]
    pair_e, gap_end = _v9c_pair_energies(e1, pair_end)
    why = "; ".join(holding) if holding else None
    if flagged and not why:
        why = ("undecided: the band on tau_LZ contains c" if ev.get("tie") else ("knife-edge" if ev.get("knife") else ("an unresolved block" if unresolved_block else "unresolved")))
    entry = {"s": ev.get("s_star"), "piece": [ev.get("s_p"), ev.get("s_c")], "s_reopen": None, "probes": ev.get("n_samples"), "source": "primitive", "kind": kind, "level": ev.get("level"),
             "branch": decision, "decided": bool(ev.get("decided")), "resolved": bool(ev.get("resolved")), "swap": bool(diabatic and not unresolved_block), "flagged": bool(flagged), "flagged_why": why,
             "block": 2 + len(ev.get("further") or []), "tau_lz": ev.get("A"), "tau_lz_band": [ev.get("A_lo"), ev.get("A_hi")], "exponent": _v9c_exponent(ev.get("A"), k), "c": float(M["lz_c"]), "k": k,
             "gap_min": ev.get("gap_min"), "gap_min2_err": ev.get("gap_min2_err"), "alpha_s": ev.get("alpha_s"), "alpha_T": ev.get("alpha_T"), "s_star": ev.get("s_star"), "fit_kind": ev.get("fit_kind"),
             "pair_start": [ev.get("i"), ev.get("partner_p")], "pair_end": pair_end, "pair": pair_e, "pair_gap_end": gap_end, "anchor_energy": None,
             "overlap": (ev.get("w_new") if (diabatic and not unresolved_block and ev.get("kind") == "passage") else None), "character_weight": ev.get("w_new"), "w_partner": ev.get("w_partner"), "inversion_delta": int(inv_after - inv_before), "intermediate": 0, "occupied_before": None, "occupied_after": None,
             "mirrored": bool(ev.get("mirrored")), "self_mirror": bool(ev.get("self_mirror")), "mirror": ev.get("mirror"),
             "jump_grid": ev.get("jump_grid"), "jump_fit": ev.get("jump_fit"), "jump_alt": ev.get("jump_alt"), "jump_prev": ev.get("jump_prev"), "falls": ev.get("falls"),
             "tie": bool(ev.get("tie")), "knife": bool(ev.get("knife")), "unresolved_block": unresolved_block, "flags": list(ev.get("flags") or []), "holding_flags": holding}
    entry["samples"] = [[float(s_), float(d_)] for s_, d_ in (ev.get("samples") or [])]   # [v9c1b x] the pair's (s, Delta) samples the fit used (the extension's s > 1 among them)
    return _v9c_py(entry), m_after


def _v9c_entry_from_run(run, M, tau, e1):
    """an approach or a tail: the fitted minimum lies beyond or before this chord; an entry without a decision"""
    alpha_s = run.get("alpha_s"); gm = run.get("gap_min")
    alpha_T = (float(alpha_s) / float(tau)) if alpha_s else None
    A = None if (gm is None or not alpha_T) else float(M["lz_k"]) * float(gm) * float(gm) / abs(alpha_T)
    kind = run.get("kind"); pair = list(run.get("pair") or [None, None])
    pair_e, gap_end = _v9c_pair_energies(e1, pair)
    entry = {"s": run.get("s_star"), "piece": None, "s_reopen": None, "probes": len(run.get("samples") or []), "source": "primitive", "kind": kind,
             "branch": ("pending" if kind == "approach" else "before"), "decided": False, "resolved": True, "swap": False, "flagged": False, "flagged_why": None, "block": 2,
             "tau_lz": A, "tau_lz_band": None, "exponent": _v9c_exponent(A, M["lz_k"]), "c": float(M["lz_c"]), "k": float(M["lz_k"]), "gap_min": gm, "alpha_s": alpha_s, "alpha_T": alpha_T, "s_star": run.get("s_star"),
             "pair_start": [run.get("i"), None], "pair_end": pair, "pair": pair_e, "pair_gap_end": gap_end, "anchor_energy": None, "overlap": None, "inversion_delta": 0, "intermediate": 0,
             "flags": ["the fitted minimum lies %s this chord: %s, no decision here" % ("after" if kind == "approach" else "before", kind)], "holding_flags": []}
    return _v9c_py(entry)


def _v9c_merge_walk(entries, crossings_walk, level, floor, info):
    """the walk's own crossing entries matched to the primitive's events by s (inside the walk's piece, or within a piece of the final level): a matched one carries both (the walk's keys kept, the primitive's under prim_ where they collide, the decision's keys added); a walk swap without an event, or against an adiabatic decision, is flagged; a walk entry without a swap (a pair within the floor at the path's end, flagged by the walk) needs no event; the primitive's events the walk did not see are appended, a sub-floor one flagged"""
    tol = 2.0 ** (-max(int(level or 1), 1)) + 1e-9
    used = {}; out = []
    def cap(q):
        return 2 if entries[q].get("mirrored") else 1
    for c in (crossings_walk or []):
        s_w = c.get("s"); piece = c.get("piece") or [None, None]; best = None
        for q, ent in enumerate(entries):
            if used.get(q, 0) >= cap(q) or ent.get("kind") in ("approach", "tail") or ent.get("s") is None:
                continue
            cands = [float(ent["s"])]
            if ent.get("mirror") and ent["mirror"].get("s_star") is not None:
                cands.append(float(ent["mirror"]["s_star"]))
            for s_e in cands:
                inside = (piece[0] is not None and piece[1] is not None and float(piece[0]) - tol <= s_e <= float(piece[1]) + tol)
                d = 0.0 if inside else (abs(s_e - float(s_w)) if s_w is not None else float("inf"))
                if (inside or d <= tol) and (best is None or d < best[0]):
                    best = (d, q)
        merged = dict(c); merged["flags"] = list(c.get("flags") or [])
        swapped = bool(c.get("swap"))
        if best is None:
            merged["source"] = "walk"
            merged.setdefault("branch", "diabatic" if swapped else ("pending" if (c.get("flagged") and "path's end" in str(c.get("flagged_why"))) else "adiabatic"))
            if swapped:
                merged["flagged"] = True; merged["flags"].append("the walk swapped a pair with no event of the primitive inside its piece %s (s %s)" % (piece, s_w))
                merged["flagged_why"] = "; ".join([x for x in (c.get("flagged_why"), merged["flags"][-1]) if x])
                info["walk_consistent"] = False
        else:
            q = best[1]; ent = entries[q]; used[q] = used.get(q, 0) + 1
            for key, val in ent.items():
                if key in ("s", "piece", "probes", "overlap", "inversion_delta", "block", "swap", "flagged", "flagged_why", "flags", "source", "kind", "pair", "pair_gap_end", "s_reopen"):
                    merged["prim_" + key] = val
                else:
                    merged.setdefault(key, val)
            merged["source"] = "walk+primitive"
            merged["flagged"] = bool(c.get("flagged")) or bool(ent.get("flagged"))
            merged["flags"].extend(list(ent.get("flags") or []))
            whys = [x for x in (c.get("flagged_why"), ent.get("flagged_why")) if x]
            if swapped and ent.get("branch") != "diabatic":
                merged["flagged"] = True; whys.append("the walk swapped a pair the primitive decided %s" % ent.get("branch")); merged["flags"].append(whys[-1]); info["walk_consistent"] = False
            merged["flagged_why"] = "; ".join(whys) if whys else None
        out.append(merged)
    for q, ent in enumerate(entries):
        if used.get(q, 0) > 0:
            continue
        ent = dict(ent)
        err = ent.get("gap_min2_err")
        if ent.get("branch") == "diabatic" and ent.get("kind") == "passage" and ent.get("gap_min") is not None and err is not None and floor is not None:
            if float(ent["gap_min"]) + _v9c_math.sqrt(max(float(err), 0.0)) < float(floor):
                ent["flagged"] = True; why = "a sub-floor passage (gap_min %.2e with its band below the floor %.2e) the walk's certificate did not open" % (float(ent["gap_min"]), float(floor))
                ent["flags"] = list(ent.get("flags") or []) + [why]; ent["flagged_why"] = "; ".join([x for x in (ent.get("flagged_why"), why) if x]); info["walk_consistent"] = False
        out.append(ent)
    out.sort(key=lambda d: (float("inf") if d.get("s") is None else float(d["s"])))
    return out


def _v9c_say(text):
    print(text, flush=True)


def v9c_line(info):
    c = info.get("counts") or {}
    tl = info.get("tau_lz_min")
    tl_s = "-" if tl is None else ("%.3e" % tl if _v9c_math.isfinite(tl) else "inf")
    return "  v9c prim %s kappa %s (%s, lz, tau %g): level %s of %s, spectra %d (path eigh %s), entries %d (diabatic %d adiabatic %d approach %d tail %d unresolved %d), tau_LZ min %s, exponent sum %.3e, set changes %d, inversions %d -> %d%s%s%s" % (
        info.get("species"), info.get("channel"), info.get("prim"), float(info.get("tau") or 0.0), info.get("level_stable"), info.get("prim_levels"), int(info.get("n_spectra", 0)), info.get("eigh_path"), len(info.get("entries") or []),
        c.get("diabatic", 0), c.get("adiabatic", 0), c.get("approach", 0), c.get("tail", 0), c.get("unresolved", 0), tl_s, float(info.get("exponent_sum") or 0.0),
        int(info.get("set_changes") or 0), int(info.get("inversions_start") or 0), int(info.get("inversions_end") or 0),
        "" if info.get("walk_consistent", True) else "; INCONSISTENT with the walk", "; UNRESOLVED" if info.get("unresolved") else "", "; re-called at x_n" if info.get("recall") else "")


def v9c_decide(species, channel, spec, mask0, mask_walk, crossings_walk, tau, floor=None, mode=None, ref0=None, say=None):
    """The decision of one channel's record (both species). spec: a V9cSpectra (the spectrum the primitive sees: the S = +1 sector's sorted levels on the light, every level on the heavy); mask0, mask_walk: the occupied index masks of that spectrum at x_(n-1) and the walk's at x_n; crossings_walk: the walk's entries; ref0: the set's eigenvectors at x_(n-1) for the trigger (optional). Returns (mask_out, info): mask_out the occupied mask at x_n (the walk's own object when nothing changed, else the primitive's decided set) and info the sector entry's passage block; (mask_walk, None) in floor mode, with nothing computed."""
    M = v9c_mode() if mode is None else dict(mode)
    if M.get("anchor_passage", "floor") != "lz":
        return mask_walk, None
    levels = int(M["prim_levels"])
    if levels < 2:
        raise ValueError("v9c prim: --prim-levels must be at least 2 (stability under halving needs two levels)")
    if tau is None or not (float(tau) > 0.0):
        raise V9cContract("v9c prim: the record's tau is not set (args.tau through the run's wrapper, V9C_CTX['tau']); alpha_T = alpha_s / tau")
    prim = v9c_prim(M["prim_module"]); rule = v9c_rule(M); tau = float(tau); w_mix = float(M["prim_w_mix"])
    m0 = _v9c_np.asarray(_v9c_to_np(mask0), bool).copy(); N = int(m0.size)
    mw = m0.copy() if mask_walk is None else _v9c_np.asarray(_v9c_to_np(mask_walk), bool).copy()
    if int(mw.size) != N or N != int(spec.N):
        raise V9cContract("v9c prim: %s channel %s: the masks (%d, %d) and the spectrum (%d) differ in size" % (species, channel, N, int(mw.size), int(spec.N)))
    floor = V9C_FLOOR if floor is None else float(floor)
    cw = list(crossings_walk or []); walk_changed = not _v9c_np.array_equal(m0, mw)
    info = {"mode": "lz", "engine": V9C_VERSION, "prim": prim.VERSION, "rule": rule.describe(), "species": str(species), "channel": str(channel), "tau": tau, "floor": floor,
            "spectrum": ("the S = +1 sector's sorted levels (the entries' indices are S = +1 sorted indices; the S = -1 sector by the mirror rule at x_n)" if str(species) == "light" else "every level in energy order"),   # C64_v9c_review_v1 finding 1
            "prim_levels": levels, "ran": False, "skipped": False, "quiet": False, "n_spectra": 0, "eigh_path": None, "mask_changed": False, "entries": [], "flags": [], "notes": [],
            "stable": None, "level_stable": None, "unresolved": False, "walk_consistent": True, "walk_changed": bool(walk_changed), "walk_crossings": len(cw), "recall": None,
            "counts": {"diabatic": 0, "adiabatic": 0, "approach": 0, "tail": 0, "unresolved": 0}, "tau_lz_min": None, "exponent_sum": 0.0,
            "set_changes": 0, "inversions_start": v9c_inversions(m0), "inversions_end": v9c_inversions(mw)}
    if walk_changed and not any(c.get("swap") for c in cw):
        info["walk_consistent"] = False
        info["flags"].append("the walk changed the set (%d indices) with no swap entry" % int(_v9c_np.count_nonzero(m0 != mw)))
    if not walk_changed and not cw:
        if v9c_skip(spec, m0, float(M["lz_skip_gap"]), ref0, w_mix):
            info["skipped"] = True; info["n_spectra"] = int(spec.n_eigh)
            return mask_walk, info
        if v9c_precheck(spec, m0, w_mix, ref0):
            info["quiet"] = True; info["n_spectra"] = int(spec.n_eigh)
            return mask_walk, info
    oracle = V9cOracle(spec)
    dec = prim.follow_chord(oracle, {channel: m0}, rule=rule, queries=(), tau=tau, weights_ch={channel: 1.0}, level_min=1, level_max=levels,
                            w_mix=w_mix, mirror="auto", channels=[channel], log=None)
    m_prim = _v9c_np.asarray(dec.masks_end[channel], bool).copy()
    final = dec.results[channel][-1]
    entries = []; m_run = m0.copy()
    for ev in sorted(dec.crossings, key=lambda d: (float("inf") if d.get("s_star") is None else float(d["s_star"]))):
        ent, m_run = _v9c_entry_from_event(ev, M, m_run, spec.e1); entries.append(ent)
    for run_ in dec.end_runs.get(channel, []):
        entries.append(_v9c_entry_from_run(run_, M, tau, spec.e1))
    merged = _v9c_merge_walk(entries, cw, final["level"], floor, info)
    counts = info["counts"]; tl = []
    for ent in merged:
        br = ent.get("branch"); kind = ent.get("kind")
        if kind in ("approach", "tail"):
            counts[kind] += 1
        elif br in ("diabatic", "adiabatic") and ent.get("decided", True) and not ent.get("flagged"):
            counts[br] += 1
        else:
            counts["unresolved"] += 1
        if ent.get("tau_lz") is not None and kind not in ("approach", "tail"):
            tl.append(float(ent["tau_lz"]))
        if br == "diabatic" and kind not in ("approach", "tail") and ent.get("exponent") is not None and _v9c_math.isfinite(float(ent["exponent"])):
            info["exponent_sum"] += float(ent["exponent"])
    info.update({"ran": True, "n_spectra": int(spec.n_eigh), "stable": bool(dec.stable), "level_stable": dec.level_stable.get(channel), "levels": dec.levels.get(channel),
                 "jumps_fall": bool(dec.jumps_fall), "printable_set": bool(dec.printable_set), "entries": merged, "end_in_window": bool(dec.end_in_window.get(channel)),
                 "mask_changed": bool(not _v9c_np.array_equal(m_prim, mw)), "set_changes": int(_v9c_np.count_nonzero(m_prim != m0)), "inversions_end": v9c_inversions(m_prim),
                 "mirror": bool(dec.mirror.get(channel)), "tau_lz_min": (min(tl) if tl else None)})
    info["flags"].extend(list(dec.flags.get(channel, []))); info["notes"].extend(list(dec.notes.get(channel, [])))
    info["extension"] = spec.ext_verified if hasattr(spec, "ext_verified") else None   # C64_v9c_review_v1 finding 9: None when no extension was needed, True when verified at s = 1, else the refusal's text
    info["unresolved"] = bool((not info["stable"]) or counts["unresolved"] > 0 or (not info["jumps_fall"]))
    if say is None:
        say = _v9c_say
    if say and (merged or info["unresolved"] or not info["walk_consistent"]):
        say(v9c_line(info))
    return (m_prim if info["mask_changed"] else mask_walk), info


def v9c_annotate(sector, info, extra, mask_out, mask_walk):
    """the sector entry in place: the passage block (the primitive's VERSION in it), the merged crossings (the walk's own kept under crossings_walk), the counts"""
    if info is None:
        return sector
    sector["passage"] = _v9c_py({k: v for k, v in info.items() if k != "entries"})
    if info.get("ran"):
        sector["crossings_walk"] = list(sector.get("crossings") or [])
        cr = list(info["entries"]) + list(extra or [])
        sector["crossings"] = cr
        sector["flagged"] = int(sum(1 for c_ in cr if c_.get("flagged"))); sector["swaps"] = int(sum(1 for c_ in cr if c_.get("swap")))
    return sector


def v9c_hold_reasons(sector, w_mix=0.9):
    """the gate's and the watch's v9c reads of one channel's sector entry (C64_G4s8_hold_ruling_v1 (4b); C64_v9c_ruling_v1 2(a) and 3): a list of (what, value, bound), empty when nothing holds"""
    S = sector if isinstance(sector, dict) else {}
    out = []
    if S.get("route") == "record-scale fallback":
        out.append(("record-scale fallback", S.get("fallback_why"), "none"))
    cr = S.get("crossings") or []; wmin = S.get("min_occupied_weight")
    if (not cr) and wmin is not None and float(wmin) < float(w_mix):
        out.append(("unresolved passage (certified record, no entry of its own)", float(wmin), ">= %.2f or an entry" % w_mix))
    for c in cr:
        kind = c.get("kind") or "crossing"; s = c.get("s"); src = c.get("source") or "walk"
        if kind in ("approach", "tail"):
            continue
        if c.get("flagged"):
            out.append(("flagged passage at s = %s (%s, %s): %s" % (s, kind, c.get("branch"), c.get("flagged_why")), c.get("tau_lz"), "unflagged"))
        elif src != "walk" and not (c.get("decided", True) and c.get("resolved", True)):
            out.append(("undecided or unresolved passage at s = %s (%s, %s)" % (s, kind, c.get("branch")), c.get("tau_lz"), "decided and resolved"))
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


class V9cEndPath(object):
    """the degenerate chord at x_n for the route's re-call with the decided set: every s the path's end (h_host(1), eig(1): the trial, eigvals(1), the fields at 1); no field change, so the walk certifies with no piece and assembles its end bookkeeping (anchors, inversions, the pairs within the floor at the end) for the given set"""

    def __init__(self, path):
        self.path = path; self.ch = getattr(path, "ch", None); self.eigh_n = 0; self.eigvals_n = 0; self.trial = getattr(path, "trial", None)

    def fields(self, s):
        return self.path.fields(1.0) if hasattr(self.path, "fields") else None

    def h_host(self, s):
        return self.path.h_host(1.0)

    def eig(self, s):
        return self.path.eig(1.0)

    def eigvals(self, s):
        return self.path.eigvals(1.0)


def _v9c_label(path, species):
    ch = getattr(path, "ch", None); kappa = getattr(ch, "kappa", None)
    return str(kappa) if kappa is not None else str(species)


def _v9c_fallback_block(species, channel, M, why):
    return _v9c_py({"mode": "lz", "engine": V9C_VERSION, "prim": v9c_prim(M["prim_module"]).VERSION, "species": species, "channel": channel, "ran": False, "fallback": True, "why": why})


def _v9c_light_ref0(ref, mask0, sig0):
    """the previous reference's columns of the S = +1 sector's set (the light caller's stash), in the + sorted order; None when absent or of another count"""
    if ref is None:
        return None
    mask0 = _v9c_np.asarray(_v9c_to_np(mask0), bool); sig0 = _v9c_np.asarray(_v9c_to_np(sig0), float)
    cols = _v9c_np.flatnonzero(mask0)
    if int(getattr(ref, "shape", (0, 0))[1]) != int(cols.size):
        return None
    keep = _v9c_np.flatnonzero(sig0[cols] > 0.0)
    return _v9c_index(ref, keep)


def _v9c_recall(fn, sig, ba, updates):
    ba2 = sig.bind(*ba.args, **ba.kwargs); ba2.apply_defaults()
    for key, val in updates.items():
        if key in ba2.arguments:
            ba2.arguments[key] = val
    try:
        return fn(*ba2.args, **ba2.kwargs)
    except Exception as exc:
        raise V9cContract("v9c prim: the re-call of %s on the degenerate chord at x_n failed (%s: %s); post the route's signature" % (getattr(fn, "__name__", "?"), type(exc).__name__, exc))


def _v9c_light_passages(fn, sig, ba, out, M):
    """the lz-mode work on one light route call: the decision on the S = +1 sector's set, the re-call where the set changed, the entry's annotation; returns (cert, sector, anchors_end, occp_end) with the decided set's anchors and occupation"""
    names = V9C_NAMES["light"]; A = ba.arguments
    if not (isinstance(out, tuple) and len(out) == 4 and isinstance(out[1], dict) and "crossings" in out[1]):
        raise V9cContract("v9c prim: the light route %s returned %s; (cert, sector, anchors_end, occp_end) expected; post its return line" % (getattr(fn, "__name__", "?"), type(out).__name__))
    cert, sector, anchors_end, occp_end = out
    path = A[names["path"]]; channel = _v9c_label(path, "light")
    if not (isinstance(cert, dict) and cert.get("certified")) or occp_end is None:
        sector["passage"] = _v9c_fallback_block("light", channel, M, cert.get("why") if isinstance(cert, dict) else None)
        return out
    e0 = _v9c_np.asarray(_v9c_to_np(A[names["e0"]]), float); sig0 = _v9c_np.asarray(_v9c_to_np(A[names["sig0"]]), float); mask0 = _v9c_np.asarray(_v9c_to_np(A[names["mask0"]]), bool); trial = A[names["trial"]]
    spec = V9cSpectra(path, True, e0, sig0, trial)
    m0 = mask0[spec.order0]; mw = _v9c_np.asarray(_v9c_to_np(occp_end), bool)
    ref0 = _v9c_light_ref0(V9C_CTX.get("light_ref"), mask0, sig0)
    floor = cert.get("d_min_effective") or A.get(names.get("d_min"))
    mask_out, info = v9c_decide("light", channel, spec, m0, mw, sector.get("crossings"), V9C_CTX.get("tau"), floor=floor, mode=M, ref0=ref0)
    if info is None:
        return out
    info["eigh_path"] = int(getattr(path, "eigh_n", 0))
    anchors_new = anchors_end; occp_new = occp_end; extra = []
    if info["mask_changed"]:
        e1, sig1, U1 = trial; e1 = _v9c_np.asarray(_v9c_to_np(e1), float); sig1 = _v9c_np.asarray(_v9c_to_np(sig1), float)
        bnds = v9c_boundaries(mask_out); anchors_dec = [float(0.5 * (spec.e1[j] + spec.e1[j + 1])) for j in bnds]
        select = globals().get("light_select_sector_v9")
        if select is None:
            raise V9cContract("v9c prim: light_select_sector_v9 (the anchor mask) is not in scope for the re-call at x_n")
        if int(bnds.size) % 2 == 0:
            info["walk_consistent"] = False; info["flags"].append("the decided set has an even boundary count (%d) in the S = +1 sector, outside the mirror rule: the walk's set kept" % int(bnds.size)); mask_out = mw
        else:
            full = _v9c_np.asarray(select(e1, sig1, anchors_dec)[0], bool)
            if not _v9c_np.array_equal(full[spec.order1], mask_out):
                info["walk_consistent"] = False; info["flags"].append("the mirror rule's S = +1 part differs from the decided set at x_n: the walk's set kept"); mask_out = mw
            else:
                out2 = _v9c_recall(fn, sig, ba, {names["path"]: V9cEndPath(path), names["e0"]: e1, names["sig0"]: sig1, names["mask0"]: full, names.get("mu_prev", "mu_prev"): None})
                cert2, sector2, anchors2, occp2 = out2
                if not (isinstance(cert2, dict) and cert2.get("certified")) or occp2 is None:
                    info["walk_consistent"] = False; info["flags"].append("the re-call at x_n fell back (%s): the walk's set kept" % (cert2.get("why") if isinstance(cert2, dict) else None)); mask_out = mw
                elif not _v9c_np.array_equal(_v9c_np.asarray(_v9c_to_np(occp2), bool), mask_out):
                    info["walk_consistent"] = False; info["flags"].append("the re-call at x_n returned a set other than the decided one (%d indices differ): the walk's set kept" % int(_v9c_np.count_nonzero(_v9c_np.asarray(_v9c_to_np(occp2), bool) != mask_out))); mask_out = mw
                else:
                    anchors_new = anchors2; occp_new = occp2
                    sector.update({"anchors_end": sector2.get("anchors_end"), "anchors": sector2.get("anchors"), "inversions_plus_end": sector2.get("inversions_plus_end")})
                    if sector2.get("route") == "sector walk":
                        sector["route"] = "sector walk"
                    extra = [dict(c_, source="walk (re-call at x_n)") for c_ in (sector2.get("crossings") or [])]
                    info["recall"] = {"route": sector2.get("route"), "pieces": sector2.get("pieces"), "cholesky": sector2.get("cholesky"), "anchors_end": sector2.get("anchors_end"), "flagged": sector2.get("flagged"), "inversions_plus_end": sector2.get("inversions_plus_end")}
        info["eigh_path"] = int(getattr(path, "eigh_n", 0)); info["mask_changed"] = bool(not _v9c_np.array_equal(mask_out, mw))
    v9c_annotate(sector, info, extra, mask_out, mw)
    return cert, sector, anchors_new, occp_new


def _v9c_heavy_passages(fn, sig, ba, out, M):
    """the lz-mode work on one heavy route call: the decision over every level, the re-call where the set changed, the entry's annotation; returns (walk, occ_end) with the decided occupation"""
    names = V9C_NAMES["heavy"]; A = ba.arguments
    if not (isinstance(out, tuple) and len(out) == 2 and isinstance(out[0], dict) and "crossings" in out[0]):
        raise V9cContract("v9c prim: the heavy route %s returned %s; (walk, occ_end) expected; post its return line" % (getattr(fn, "__name__", "?"), type(out).__name__))
    walk, occ_end = out
    path = A[names["path"]]; channel = _v9c_label(path, "heavy")
    if not walk.get("certified") or occ_end is None:
        walk["passage"] = _v9c_fallback_block("heavy", channel, M, walk.get("why"))
        return out
    e0 = _v9c_np.asarray(_v9c_to_np(A[names["e0"]]), float); mask0 = _v9c_np.asarray(_v9c_to_np(A[names["mask0"]]), bool); trial = A[names["trial"]]
    spec = V9cSpectra(path, False, e0, None, trial)
    m0 = mask0; mw = _v9c_np.asarray(_v9c_to_np(occ_end), bool)
    prev_ref = A.get(names.get("prev_ref"))
    ref0 = prev_ref if (prev_ref is not None and int(getattr(prev_ref, "shape", (0, 0))[1]) == int(m0.sum())) else None
    floor = walk.get("d_min_effective") or A.get(names.get("d_min"))
    mask_out, info = v9c_decide("heavy", channel, spec, m0, mw, walk.get("crossings"), V9C_CTX.get("tau"), floor=floor, mode=M, ref0=ref0)
    if info is None:
        return out
    info["eigh_path"] = int(getattr(path, "eigh_n", 0))
    occ_new = occ_end; extra = []
    if info["mask_changed"]:
        out2 = _v9c_recall(fn, sig, ba, {names["path"]: V9cEndPath(path), names["e0"]: spec.e1, names["mask0"]: mask_out.copy()})
        walk2, occ2 = out2
        if not walk2.get("certified") or occ2 is None:
            info["walk_consistent"] = False; info["flags"].append("the re-call at x_n fell back (%s): the walk's set kept" % walk2.get("why")); mask_out = mw
        elif not _v9c_np.array_equal(_v9c_np.asarray(_v9c_to_np(occ2), bool), mask_out):
            info["walk_consistent"] = False; info["flags"].append("the re-call at x_n returned a set other than the decided one (%d indices differ): the walk's set kept" % int(_v9c_np.count_nonzero(_v9c_np.asarray(_v9c_to_np(occ2), bool) != mask_out))); mask_out = mw
        else:
            occ_new = occ2
            walk.update({"anchors_end": walk2.get("anchors_end"), "inversions_end": walk2.get("inversions_end")})
            extra = [dict(c_, source="walk (re-call at x_n)") for c_ in (walk2.get("crossings") or [])]
            info["recall"] = {"route": walk2.get("route"), "pieces": walk2.get("pieces"), "eigvals": walk2.get("eigvals"), "anchors_end": walk2.get("anchors_end"), "flagged": walk2.get("flagged"), "inversions_end": walk2.get("inversions_end")}
        info["eigh_path"] = int(getattr(path, "eigh_n", 0)); info["mask_changed"] = bool(not _v9c_np.array_equal(mask_out, mw))
    v9c_annotate(walk, info, extra, mask_out, mw)
    return walk, occ_new


def v9c_wrap_walk(species, fn):
    """the sector route of one species wrapped by name: in floor mode the route's return passes through untouched (bitwise v9b; one dict lookup); in lz mode the species' passages run on it. The route's arguments are read by name (V9C_NAMES); a route that does not bind the required names is listed in V9C_MISSING and lz refuses to start."""
    sig = _v9c_inspect.signature(fn); names = V9C_NAMES[species]
    missing = [key for key in V9C_REQUIRED[species] if names.get(key) not in sig.parameters]
    if missing:
        V9C_MISSING.append("the %s route %s does not bind %s (its signature %s; V9C_NAMES[%r] names them)" % (species, getattr(fn, "__name__", "?"), ", ".join("%r" % names.get(k_) for k_ in missing), str(sig), species))
    work = _v9c_light_passages if species == "light" else _v9c_heavy_passages

    def wrapped(*a, **k):
        out = fn(*a, **k)
        if V9C_MODE["anchor_passage"] != "lz":
            return out
        try:
            ba = sig.bind(*a, **k); ba.apply_defaults()
        except TypeError as exc:
            raise V9cContract("v9c prim: the %s route's call could not be bound to its signature %s (%s)" % (species, str(sig), exc))
        return work(fn, sig, ba, out, dict(V9C_MODE))

    wrapped.__name__ = getattr(fn, "__name__", species); wrapped.__qualname__ = getattr(fn, "__qualname__", wrapped.__name__)
    wrapped.__doc__ = getattr(fn, "__doc__", None); wrapped.__wrapped__ = fn; wrapped.v9c_species = species
    return wrapped


def v9c_wrap_light_caller(fn):
    """light_continue_sector_v9 wrapped: in lz mode the window's last entry's reference (the previous reference's occupied columns) is stashed for the light route's trigger, and cleared after"""
    def wrapped(a, ch, window, *rest, **kw):
        if V9C_MODE["anchor_passage"] != "lz":
            return fn(a, ch, window, *rest, **kw)
        V9C_CTX["light_ref"] = (window[-1].get("ref") if window else None)
        try:
            return fn(a, ch, window, *rest, **kw)
        finally:
            V9C_CTX["light_ref"] = None
    wrapped.__name__ = getattr(fn, "__name__", "light_continue_sector_v9"); wrapped.__doc__ = getattr(fn, "__doc__", None); wrapped.__wrapped__ = fn
    return wrapped
# <<< v9c prim block'''

V9C_REBIND = r'''# >>> v9c prim rebind: the sector routes of both species rebound by name to their wrapper, the light caller to its stash, run to the mode's reader (v9b's functions kept as __wrapped__)
for _v9c_species, _v9c_name in (("light", V9C_LIGHT_DEF), ("heavy", V9C_HEAVY_DEF)):
    if _v9c_name in globals() and callable(globals()[_v9c_name]):
        globals()[_v9c_name] = v9c_wrap_walk(_v9c_species, globals()[_v9c_name])
    else:
        V9C_MISSING.append("the %s route %s is not defined in this engine" % (_v9c_species, _v9c_name))
if V9C_LIGHT_CALLER in globals() and callable(globals()[V9C_LIGHT_CALLER]):
    globals()[V9C_LIGHT_CALLER] = v9c_wrap_light_caller(globals()[V9C_LIGHT_CALLER])
else:
    V9C_MISSING.append("the light caller %s is not defined in this engine" % V9C_LIGHT_CALLER)
if V9C_RUN_DEF in globals() and callable(globals()[V9C_RUN_DEF]):
    globals()[V9C_RUN_DEF] = v9c_wrap_run(globals()[V9C_RUN_DEF])
else:
    V9C_MISSING.append("the run %s is not defined in this engine" % V9C_RUN_DEF)
# <<< v9c prim rebind'''


V9C1C_LINES = [  # [v9c1c x] (the v9c1b line, the v9c1c line) as in the block; the built engine carries them with "  # v9c prim" appended unless the line already names it
    ('V9C_DEFAULTS = {"anchor_passage": "floor", "prim_module": "assign_primitive_v1e", "lz_c": 1.0, "lz_k": 1.0, "lz_tie": "adiabatic", "lz_knife": 0.1,  # [v9c1b x] v1e',
     'V9C_DEFAULTS = {"anchor_passage": "floor", "prim_module": "assign_primitive_v1f", "lz_c": 1.0, "lz_k": 1.0, "lz_tie": "adiabatic", "lz_knife": 0.1,  # [v9c1c x] v1f'),
    ('    ap.add_argument("--prim-module", dest="prim_module", default="assign_primitive_v1e", help="(v9c prim) the assignment primitive (assign_primitive_v1d or its successor), beside the engine; its VERSION is recorded in every sector entry")  # [v9c1b x] v1e',
     '    ap.add_argument("--prim-module", dest="prim_module", default="assign_primitive_v1f", help="(v9c prim) the assignment primitive (assign_primitive_v1d or its successor), beside the engine; its VERSION is recorded in every sector entry")  # [v9c1c x] v1f'),
    ('            raise SystemExit("v9c prim: the primitive %r cannot be imported or its Rule built (%s: %s); that module belongs beside the engine" % (V9C_MODE["prim_module"], type(exc).__name__, exc))  # [v9c1b x]',
     '            raise SystemExit("v9c prim: the primitive %r cannot be imported or its Rule built (%s: %s); assign_primitive_v1f.py belongs beside the engine" % (V9C_MODE["prim_module"], type(exc).__name__, exc))  # [v9c1c x]'),
]


def to_v9c1b(text):
    """engine_twofield_v9c1c.py -> engine_twofield_v9c1b.py: each [v9c1c x] line replaced by its v9c1b line (each occurring once)"""
    for old, new in V9C1C_LINES:
        suf = "" if MARK in new else "  # " + MARK
        if text.count(new + suf + "\n") != 1:
            raise BuildStop("to_v9c1b", text.count(new + suf + "\n"), new + suf)
        text = text.replace(new + suf + "\n", old + suf + "\n", 1)
    return text


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


def _names(light_def, heavy_def, light_caller, run_def):
    return {"@LIGHT_DEF@": light_def, "@HEAVY_DEF@": heavy_def, "@LIGHT_CALLER@": light_caller, "@RUN_DEF@": run_def}


def block_text(light_def=LIGHT_DEF_DEFAULT, heavy_def=HEAVY_DEF_DEFAULT, light_caller=LIGHT_CALLER_DEFAULT, run_def=RUN_DEF_DEFAULT):
    text = V9C_BLOCK
    for key, val in _names(light_def, heavy_def, light_caller, run_def).items():
        text = text.replace(key, val)
    return text


def rebind_text(light_def=LIGHT_DEF_DEFAULT, heavy_def=HEAVY_DEF_DEFAULT, light_caller=LIGHT_CALLER_DEFAULT, run_def=RUN_DEF_DEFAULT):
    text = V9C_REBIND
    for key, val in _names(light_def, heavy_def, light_caller, run_def).items():
        text = text.replace(key, val)
    return text


def load_block_namespace(light_def=LIGHT_DEF_DEFAULT, heavy_def=HEAVY_DEF_DEFAULT, light_caller=LIGHT_CALLER_DEFAULT, run_def=RUN_DEF_DEFAULT):
    """the layer executed on its own (the self-test's fixtures run on it without v9b)"""
    ns = {"__name__": "engine_twofield_v9c_block"}
    exec(compile(block_text(light_def, heavy_def, light_caller, run_def), "<v9c prim block>", "exec"), ns)
    return ns


def replacements(light_def=LIGHT_DEF_DEFAULT, heavy_def=HEAVY_DEF_DEFAULT, light_caller=LIGHT_CALLER_DEFAULT, run_def=RUN_DEF_DEFAULT, old_args=OLD_ARGS, old_resume=OLD_RESUME):
    indent = old_args[:len(old_args) - len(old_args.lstrip())]
    head = old_args.strip()
    if ".add_argument(" not in head:
        raise ValueError("OLD_ARGS must be the add_argument call of --cut-transport: %r" % old_args[:80])
    var = head.split(".add_argument(")[0]
    new_args = indent + "v9c_add_arguments(%s)  # %s\n" % (var, MARK) + old_args
    new_resume = old_resume + "\n" + mark(RESUME_NEW_TAIL)
    light_old = "def %s(" % light_def
    return [Rep("version", OLD_VERSION, NEW_VERSION, soft=True),
            Rep("args", old_args, new_args),
            Rep("resume", old_resume, new_resume),
            Rep("layer", light_old, mark(block_text(light_def, heavy_def, light_caller, run_def)) + "\n\n\n" + light_old),
            Rep("rebind", MAIN_GUARD, mark(rebind_text(light_def, heavy_def, light_caller, run_def)) + "\n\n\n" + MAIN_GUARD)]


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
    m = re.search(r"\(light route (\S+), heavy route (\S+), light caller (\S+), run (\S+)\)", text)
    if m is None:
        return None
    return m.group(1), m.group(2), m.group(3), m.group(4)


def _def_lines(src):
    lines = src.split("\n"); out = []
    for q, line in enumerate(lines):
        if line.startswith("def "):
            out.append((q + 1, line.rstrip()[:200]))
    return out


def probe(src, reps, names):
    print("probe of v9b (%s):" % md5(src))
    for r in reps:
        s = r.old
        print("  %-8s %-4s occurs %d time(s): %r" % (r.name, "soft" if r.soft else "hard", src.count(s), s if len(s) < 100 else s[:100] + "..."))
    for label, name in (("heavy route", names[1]), ("light caller", names[2]), ("run", names[3])):
        print("  %s def line %r occurs %d time(s)" % (label, "def %s(" % name, src.count("def %s(" % name)))
    print("candidate lines:")
    for q, line in enumerate(src.split("\n"), 1):
        if "--cut-transport" in line and "add_argument" in line:
            print("  args     %6d: %s" % (q, line.rstrip()[:200]))
        if 'get("cut_transport"' in line and "resume" in line.lower():
            print("  resume   %6d: %s" % (q, line.rstrip()[:200]))
        if "VERSION" in line and ("v9b" in line or "v9" in line) and "=" in line:
            print("  version  %6d: %s" % (q, line.rstrip()[:200]))
    for q, line in _def_lines(src):
        low = line.lower()
        if any(key in low for key in ("sector_walk", "continue_sector", "continue_ref_v9", "def run(", "resume_v36")):
            print("  def      %6d: %s" % (q, line))
    print("  main guard occurs %d time(s)" % src.count(MAIN_GUARD))
    print("the lane posts the lines of the hard anchors that do not occur once, and the routes' signatures and return lines for V9C_NAMES")


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
    ap.add_argument("--light-caller", default=LIGHT_CALLER_DEFAULT, help="the light route's caller, which holds the previous reference (default %(default)s)")
    ap.add_argument("--run-def", default=RUN_DEF_DEFAULT, help="the run function main() calls with the parsed namespace (default %(default)s)")
    ap.add_argument("--probe", action="store_true", help="counts and candidate lines; nothing written")
    ap.add_argument("--check", action="store_true", help="the build in memory with the marks check and the reverse round trip; nothing written")
    ap.add_argument("--reverse", action="store_true", help="the reverse replacements: v9c -> v9b")
    ap.add_argument("--against", default=None, help="with --reverse: the v9b file the restored text must equal byte for byte")
    ap.add_argument("--to-v9c1b", default=None, help="[v9c1c x] the source is engine_twofield_v9c1c.py: restore the [v9c1c x] lines and compare with this v9c1b file byte for byte")
    a = ap.parse_args(argv)
    src = read(a.source)
    names = (a.light_def, a.heavy_def, a.light_caller, a.run_def)
    if a.to_v9c1b:   # [v9c1c x]
        back = to_v9c1b(src); ref = read(a.to_v9c1b)
        print("to v9c1b: %s (md5 %s) -> md5 %s; against %s (md5 %s): %s" % (a.source, md5(src), md5(back), a.to_v9c1b, md5(ref), "BYTE FOR BYTE" if back == ref else "DIFFERS"))
        return 0 if back == ref else 2
    if a.reverse:
        names = header_names(src) or names
        reps = replacements(*names)
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
    reps = replacements(*names)
    if a.probe:
        probe(src, reps, names); return 0
    bad = precheck(src, reps)
    defs_bad = [(label, name) for label, name in (("heavy route", a.heavy_def), ("light caller", a.light_caller), ("run", a.run_def)) if src.count("def %s(" % name) != 1]
    if bad or defs_bad:
        for r, n in bad:
            print("STOP: replacement %s: the OLD string occurs %d time(s), not once:\n%s" % (r.name, n, r.old))
        for label, name in defs_bad:
            print("STOP: the %s def line %r occurs %d time(s); give --%s" % (label, "def %s(" % name, src.count("def %s(" % name), {"heavy route": "heavy-def", "light caller": "light-caller", "run": "run-def"}[label]))
        print("nothing written; the lane posts the lines named (python %s %s --probe lists the candidates)" % (os.path.basename(sys.argv[0]), a.source))
        return 2
    out_text, log = apply(src, reps)
    marks = marks_check(src, out_text)
    if marks:
        print("STOP: %d line(s) of v9c not in v9b lack the mark %r (a builder defect):" % (len(marks), MARK))
        for q, line in marks[:20]:
            print("  %6d: %s" % (q, line[:160]))
        return 2
    back, _ = apply(out_text, reps, reverse=True)
    if back != src:
        print("STOP: the reverse replacements do not restore v9b byte for byte (a builder defect)"); return 2
    try:
        compile(out_text, "engine_twofield_v9c.py", "exec")
    except SyntaxError as exc:
        print("STOP: the built text does not compile: %s" % exc); return 2
    n_marked = sum(1 for line in out_text.split("\n") if MARK in line)
    print("build: %s (md5 %s) -> v9c (md5 %s); %s; %d lines carry %r; marks ok; reverse byte for byte; compiled" % (a.source, md5(src), md5(out_text), "; ".join(log), n_marked, MARK))
    if a.check:
        print("CHECK PASS (nothing written)"); return 0
    out = a.out or os.path.join(os.path.dirname(os.path.abspath(a.source)), "engine_twofield_v9c.py")
    if os.path.abspath(out) == os.path.abspath(a.source):
        print("STOP: the output would overwrite the source"); return 2
    write(out, out_text)
    print("written: %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
