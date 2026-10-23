#!/usr/bin/env python
"""assign_primitive_v1.py (reviewer A for the busi lane; PRIM_ruling_v1.md): the one assignment primitive of reorientation 5, mechanism B, shared by the
1-field engine's chord transport (engine_collapse_v36p, --dg-assign prim), the Newton set-resolving rounds (engine_collapse_v36j rounds_v36c; the
plan in PRIM_ruling_v1 section 3) and the two-field sector walk's crossing phase (engine_twofield_v9b; the same plan). Pure Python and numpy; the
eigenvectors stay in the caller's backend (numpy or torch) behind a small oracle protocol.

The question. Given a chord s in [0, 1] of configurations (the accepted record's fields at s = 0, the candidate fields at s = 1) and the occupied set
at s = 0 (per channel a boolean index mask of the sorted spectrum), which index mask is the occupied set at s = 1 and at the caller's query points
(the three-node rule's nodes), which near-crossings of a set member with a level outside the set lie on the chord, what the declared rule decides
at each, and whether the answer is resolved: the same when the chord is halved, with charge jumps that fall with the halving.

The walk at one level l (per channel; channels are independent and stop at their own stable level). The points are k / 2^l for k = 0 .. 2^l with
the query points. The set's reference at the last clean point, the eigenvectors of its decided mask, is projected on the spectrum at the next
point: p_j, the set subspace's weight on level j; the character mask is the n largest p_j (ties to the lower index). A point is clean when every
character member has p >= w_mix and every other level p <= 1 - w_mix; otherwise it lies inside a crossing's window, the reference is not moved
and, for each member whose character is split, the two levels that share it and their splitting are recorded. Between consecutive clean points
the character mask is compared with the decided mask: an index that entered the set is the new position of one member whose character continued
there (the reference row with the largest weight on that level). The levels whose energy order against that member swapped between the two
points are found by continuing the characters of a small index window; the members among them are crossed members (no event) and exactly one
non-member among them is the partner of one passage. Two or more passages in one piece, an unclear continuation (weight below w_mix) or index
changes that are not the sum of single passages are a block, unresolved at this level. The pair's adiabatic splitting Delta(s) = |e_upper -
e_lower| is sampled at the clean point before (member and partner), at the window points between (the pair sharing the member's character), at
the clean point after (the swapped positions) and at the next point (by character continuation), and fitted by the two-level form Delta^2 =
alpha_s^2 (s - s*)^2 + gap_min^2 through the three samples around the smallest splitting (exact for linear diabats; converging to the minimum as
the samples approach it; a negative fitted minimum is an exact crossing to this fit and is clamped to zero). With the chord's physical length tau
the diabatic levels pass each other at the rate alpha_T = alpha_s / tau in the record's clock. The eigenvalue tolerance eig_tol bands gap_min^2.

The declared rule (Rule). lz: the Landau-Zener convention of the 1-field record solve: tau_LZ = k gap_min^2 / (hbar |alpha_T|) (k = 1: the traversal
time gap_min/alpha_T of the crossing in units of the pair's precession time hbar/gap_min; k = pi/2 gives the LZ exponent); adiabatic when tau_LZ >= c
on the whole band of gap_min^2 (the occupied set keeps the lower branch: its character changes, its index does not, the chart's charge is
continuous), diabatic when tau_LZ < c on the whole band (the set follows its character to the other branch: the index swaps at s*, the chart's
charge jumps by jump_sign w gap_min there), undecided when the band contains c (the next level decides it; at the last level the tie goes to
Rule.tie and the crossing is flagged, so the record is not printable). overlap: the maximum-overlap convention (Computation C's declared convention
and version 36k's chord): every crossing diabatic. energy: every crossing adiabatic (the energy-order control). The gap against the step's
resolution, made precise: the smallest sampled splitting bounds gap_min from above, so a crossing narrower than the grid with Delta_grid^2 < c hbar
alpha_T / k is diabatic without being resolved; a crossing wider than the grid is sampled near its minimum and its band decides it; what the grid
leaves undecided the halving resolves.

Halving stability. Levels level_min .. level_max (at least two). A channel is stable at level l when its decided masks at s = 1 and at every query
point equal level l - 1's, its crossings match level l - 1's one to one (decision, direction, mirror flags, s* within one piece of level l - 1),
under the lz rule every matched crossing's gap_min^2 moved by at most gap_conv times the threshold c hbar alpha_T / k plus the band (the
decision's input converged against the scale it is compared with), and none is a block, unfitted or undecided. The primitive stops at the first
stable level per channel; the accepted set is the decided masks there; a channel unstable at level_max returns its finest masks with stable False.

The charge jump. For a diabatic crossing: jump_grid = jump_sign w Delta_grid, the jump a switch placed at the walk's resolution makes (first order in
the levels' splitting at the assignment point), and jump_fit = jump_sign w gap_min, the resolved limit, zero at an exact crossing. jumps_fall is
True when at every matched diabatic crossing |jump_grid(l)| <= |jump_grid(l - 1)| and |jump_fit| <= |jump_grid(l)| within jump_tol; a jump that
does not fall is a wrong pairing (the two levels paired are not the partners of one crossing). An adiabatic crossing has jump 0 and its would-be
diabatic jump recorded as jump_alt. Mirror sectors (e <-> -e, index i <-> N - 1 - i; mirror="auto" reads the symmetry off the first spectrum): a
crossing and its mirror are one crossing, counted once (the primary is the one with the smaller old index), its jump the sum of both sectors; a
self-mirror crossing (a level meeting its own mirror at e = 0) counts once.

Printability (the caller's criterion): a record prints when its residual meets the stop rule, its accepted set is stable under halving
(Decision.stable) and its charge jumps fall (Decision.jumps_fall); Decision.printable_set is the conjunction of the last two.

The oracle protocol (duck typed): spectrum(s, channels) -> {channel: (e, U)} with e the eigenvalues sorted ascending as numpy and U the eigenvector
columns in the backend's array; columns(U, cols) -> the columns (default U[:, list(cols)]); weights(A, B) -> numpy |A^H B|^2 of shape (A.shape[1],
B.shape[1]) (default numpy; a torch oracle must provide it); project(A, B) -> numpy (B.shape[1],), the column sums of weights(A, B) (default from
weights); release(U) optional. The caller's keep(s, channel, e, U, mask) -> any is called once per query point and channel at that channel's final
level with the decided mask (the engine keeps its held reference there); its returns are in Decision.kept[s][channel]. Memory: per channel the walk
holds the eigenvectors of the last clean point, of the current point and of at most the pending query points; nothing is cached across levels
but the eigenvalues.

Bitwise. Where a channel's set does not change along the chord the primitive returns the input mask's values at every point (np.array_equal with
the input; the arrays are copies, the input is not modified) and no crossing; two runs on the same oracle give identical results (deterministic ties).
"""
import math

import numpy as np

VERSION = "assign_primitive_v1"
GAUSS3 = ((1.0 - math.sqrt(0.6)) / 2.0, 0.5, (1.0 + math.sqrt(0.6)) / 2.0)   # the three-node rule's nodes on [0, 1] (the engines' query points)
DEFAULTS = {"level_min": 1, "level_max": 4, "w_mix": 0.9, "eig_tol": 1e-10, "jump_tol": 1e-9, "jump_sign": -1.0, "gap_conv": 0.25}

__all__ = ["VERSION", "GAUSS3", "DEFAULTS", "Rule", "fit_pair", "classify_pair", "jump_falls", "dyadic_points", "follow_chord", "Decision"]


def dyadic_points(level, queries=()):
    """the chord's points at a level: k / 2^level for k = 0 .. 2^level with the query points; sorted, without duplicates (dyadic rationals are exact floats)."""
    n = 1 << int(level)
    pts = set(float(k) / n for k in range(n + 1))
    for q in queries:
        q = float(q)
        if not (0.0 <= q <= 1.0):
            raise ValueError("assign_primitive_v1: a query point outside [0, 1]: %r" % (q,))
        pts.add(q)
    return sorted(pts)


def _py(x):
    """plain Python for json: numpy scalars and arrays, tuples, dicts, lists."""
    if isinstance(x, dict):
        return {str(k): _py(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_py(v) for v in x]
    if isinstance(x, np.ndarray):
        return [_py(v) for v in x.tolist()]
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    return x


class Rule(object):
    """The declared rule at a near-crossing. kind: lz (adiabatic iff tau_LZ = k gap_min^2/(hbar |alpha_T|) >= c on the band), overlap (every crossing
    diabatic), energy (every crossing adiabatic). tie: the decision when the band on tau_LZ still contains c at the last level. knife: a decided lz
    crossing with |ln(tau_LZ / c)| below it is flagged knife-edge."""
    KINDS = ("lz", "overlap", "energy")

    def __init__(self, kind="lz", c=1.0, k=1.0, hbar=1.0, tie="adiabatic", knife=0.1):
        if kind not in self.KINDS:
            raise ValueError("assign_primitive_v1.Rule: kind must be one of %s, not %r" % (self.KINDS, kind))
        if tie not in ("adiabatic", "diabatic"):
            raise ValueError("assign_primitive_v1.Rule: tie must be adiabatic or diabatic, not %r" % (tie,))
        if not (c > 0.0 and k > 0.0 and hbar > 0.0):
            raise ValueError("assign_primitive_v1.Rule: c, k and hbar must be positive")
        self.kind = kind; self.c = float(c); self.k = float(k); self.hbar = float(hbar); self.tie = tie; self.knife = float(knife)

    def adiabaticity(self, gap_min2, alpha_T):
        """tau_LZ = k gap_min^2 / (hbar |alpha_T|); infinite when the levels do not pass each other."""
        a = abs(float(alpha_T)) if alpha_T is not None else 0.0
        if not (a > 0.0):
            return float("inf")
        return self.k * max(float(gap_min2), 0.0) / (self.hbar * a)

    def default(self):
        """the decision of a crossing the fit does not resolve (unresolved; the level is then unstable)."""
        return {"lz": self.tie, "overlap": "diabatic", "energy": "adiabatic"}[self.kind]

    def decide(self, gap_min2, gap_min2_err, alpha_T):
        """returns {decision, decided, A, A_lo, A_hi, c, kind, tie, knife}; A the adiabaticity, [A_lo, A_hi] its band from the band on gap_min^2."""
        gm2 = max(float(gap_min2), 0.0); err = abs(float(gap_min2_err)) if gap_min2_err is not None else 0.0
        A = self.adiabaticity(gm2, alpha_T); A_lo = self.adiabaticity(max(gm2 - err, 0.0), alpha_T); A_hi = self.adiabaticity(gm2 + err, alpha_T)
        out = {"A": A, "A_lo": A_lo, "A_hi": A_hi, "c": self.c, "kind": self.kind, "tie": False, "knife": False}
        if self.kind == "overlap":
            out.update({"decision": "diabatic", "decided": True}); return out
        if self.kind == "energy":
            out.update({"decision": "adiabatic", "decided": True}); return out
        if A_hi < self.c:
            out.update({"decision": "diabatic", "decided": True})
        elif A_lo >= self.c:
            out.update({"decision": "adiabatic", "decided": True})
        else:
            out.update({"decision": self.tie, "decided": False, "tie": True})
        if out["decided"] and A > 0.0 and math.isfinite(A) and abs(math.log(A / self.c)) < self.knife:
            out["knife"] = True
        return out

    def describe(self):
        if self.kind == "lz":
            return "lz: tau_LZ = %g gap_min^2 / (%g |alpha_T|) >= %g adiabatic, below diabatic, the band's tie %s, knife-edge flag at |ln(tau_LZ/c)| < %g" % (self.k, self.hbar, self.c, self.tie, self.knife)
        return {"overlap": "overlap: every crossing diabatic (the maximum-overlap convention)", "energy": "energy: every crossing adiabatic (the energy-order control)"}[self.kind]

    def __repr__(self):
        return "Rule(%s)" % self.describe()


def fit_pair(samples, eig_tol=1e-10):
    """The two-level fit of a pair's splitting. samples: (s, Delta) pairs. The parabola y = Delta^2 through the three samples around the smallest Delta
    gives s*, gap_min^2 (clamped to zero when negative: an exact crossing to the fit), alpha_s and a band on gap_min^2 from eig_tol; the remaining
    samples give the misfit (advisory). fit_ok needs three samples, a convex parabola and the vertex between the three samples."""
    pts = []
    for s, d in samples:
        if s is None or d is None:
            continue
        s = float(s); d = abs(float(d))
        if math.isfinite(s) and math.isfinite(d):
            pts.append((s, d))
    pts.sort()
    ded = []
    for s, d in pts:
        if ded and abs(ded[-1][0] - s) <= 1e-15:
            if d < ded[-1][1]:
                ded[-1] = (s, d)
        else:
            ded.append((s, d))
    pts = ded
    out = {"fit_ok": False, "fit_why": None, "n_samples": len(pts), "s_star": None, "gap_min": None, "gap_min2": None, "gap_min2_raw": None, "gap_min2_err": None,
           "alpha_s": None, "delta_grid": None, "s_grid": None, "bracket": False, "misfit": None, "fit_points": None}
    if not pts:
        out["fit_why"] = "no samples of the pair's splitting"
        return out
    m = min(range(len(pts)), key=lambda q: (pts[q][1], q))
    out["delta_grid"], out["s_grid"] = pts[m][1], pts[m][0]
    if len(pts) < 3:
        out["fit_why"] = "fewer than 3 samples of the pair's splitting"
        return out
    lo = min(max(m - 1, 0), len(pts) - 3)
    (s0, d0), (s1, d1), (s2, d2) = pts[lo], pts[lo + 1], pts[lo + 2]
    y0, y1, y2 = d0 * d0, d1 * d1, d2 * d2
    a = ((y2 - y1) / (s2 - s1) - (y1 - y0) / (s1 - s0)) / (s2 - s0)
    b = (y1 - y0) / (s1 - s0) - a * (s0 + s1)
    c = y0 - a * s0 * s0 - b * s0
    out["fit_points"] = [(s0, d0), (s1, d1), (s2, d2)]
    if not (a > 0.0):
        out["fit_why"] = "no convex fit: the three samples around the smallest splitting do not curve up (not one two-level crossing)"
        return out
    s_star = -b / (2.0 * a); gm2 = c - b * b / (4.0 * a); alpha_s = math.sqrt(a)
    L0 = (s_star - s1) * (s_star - s2) / ((s0 - s1) * (s0 - s2))
    L1 = (s_star - s0) * (s_star - s2) / ((s1 - s0) * (s1 - s2))
    L2 = (s_star - s0) * (s_star - s1) / ((s2 - s0) * (s2 - s1))
    err = sum(abs(L) * (4.0 * d * eig_tol + 4.0 * eig_tol * eig_tol) for L, d in ((L0, d0), (L1, d1), (L2, d2)))
    out["gap_min2_raw"] = gm2
    if gm2 < 0.0:
        gm2 = 0.0
    out.update({"s_star": s_star, "gap_min2": gm2, "gap_min": math.sqrt(gm2), "gap_min2_err": err, "alpha_s": alpha_s, "bracket": bool(s0 <= s_star <= s2)})
    others = [(s, d) for q, (s, d) in enumerate(pts) if q < lo or q >= lo + 3]
    if others:
        mis = 0.0
        for s, d in others:
            yp = a * s * s + b * s + c; yo = d * d
            mis = max(mis, abs(yp - yo) / max(yo, gm2 + a * (s - s_star) ** 2, 1e-300))
        out["misfit"] = mis
    if not out["bracket"]:
        out["fit_why"] = "the vertex outside the three samples around the smallest splitting"
        return out
    out["fit_ok"] = True
    return out


def classify_pair(samples, rule, tau=1.0, eig_tol=1e-10):
    """fit_pair plus the rule's decision: alpha_T = alpha_s / tau, A = tau_LZ, its band, decision, decided, tie, knife."""
    out = fit_pair(samples, eig_tol)
    out.update({"alpha_T": None, "A": None, "A_lo": None, "A_hi": None, "decision": rule.default(), "decided": False, "tie": False, "knife": False})
    if not out["fit_ok"]:
        return out
    alpha_T = out["alpha_s"] / float(tau)
    d = rule.decide(out["gap_min2"], out["gap_min2_err"], alpha_T)
    out.update({"alpha_T": alpha_T, "A": d["A"], "A_lo": d["A_lo"], "A_hi": d["A_hi"], "decision": d["decision"], "decided": d["decided"], "tie": d["tie"], "knife": d["knife"]})
    return out


def jump_falls(j_prev, j_now, j_fit, tol=1e-9, tol_abs=0.0):
    """the jumps-fall test of one matched diabatic crossing: |j_now| <= |j_prev| and |j_fit| <= |j_now| within the relative tol and the absolute tol_abs."""
    if j_prev is None or j_now is None:
        return True
    ok = abs(j_now) <= abs(j_prev) * (1.0 + tol) + tol_abs
    if j_fit is not None:
        ok = ok and (abs(j_fit) <= abs(j_now) * (1.0 + tol) + tol_abs)
    return bool(ok)


def _mirror_map(mirror, e):
    """the index mirror: None (off), True (i -> N - 1 - i), a callable, or "auto" (i -> N - 1 - i when e[i] + e[N - 1 - i] vanishes to 1e-9 of the scale)."""
    N = int(e.size)
    if mirror is None or mirror is False:
        return None
    if callable(mirror):
        return mirror
    if mirror is True:
        return lambda i, N=N: N - 1 - i
    if mirror == "auto":
        if N == 0:
            return None
        scale = max(1.0, float(np.max(np.abs(e))))
        if float(np.max(np.abs(e + e[::-1]))) <= 1e-9 * scale:
            return lambda i, N=N: N - 1 - i
        return None
    raise ValueError("assign_primitive_v1: mirror must be None, True, a callable or 'auto', not %r" % (mirror,))


class _Ctx(object):
    """the shared parameters and backend calls of one follow_chord call."""

    def __init__(self, oracle, rule, tau, w_mix, eig_tol, jump_tol, jump_sign, gap_conv, weights_ch, queries, keep, log):
        self.oracle = oracle; self.rule = rule; self.tau = float(tau); self.w_mix = float(w_mix); self.eig_tol = float(eig_tol)
        self.jump_tol = float(jump_tol); self.jump_sign = float(jump_sign); self.gap_conv = float(gap_conv); self.weights_ch = dict(weights_ch or {})
        self.queries = sorted(set(float(q) for q in queries) | {1.0}); self.query_set = set(self.queries); self.keep = keep; self.log = log

    def columns(self, U, cols):
        cols = [int(c) for c in cols]
        if hasattr(self.oracle, "columns"):
            return self.oracle.columns(U, cols)
        return U[:, cols]

    def weights(self, A, B):
        if hasattr(self.oracle, "weights"):
            return np.asarray(self.oracle.weights(A, B), float)
        return np.abs(np.asarray(A).conj().T @ np.asarray(B)) ** 2

    def project(self, A, B):
        if hasattr(self.oracle, "project"):
            return np.asarray(self.oracle.project(A, B), float)
        return self.weights(A, B).sum(axis=0)

    def release(self, U):
        if hasattr(self.oracle, "release"):
            self.oracle.release(U)

    def w(self, ch):
        return float(self.weights_ch.get(ch, 1.0))

    def say(self, text):
        if self.log is not None:
            self.log(text)


def _ev_key(ev):
    return (ev["s_star"], ev["i"], ev["n"])


class _Walk(object):
    """the walk of one channel at one level."""

    def __init__(self, ctx, ch, mask0, level, mirror):
        self.ctx = ctx; self.ch = ch; self.level = int(level); self.mirror = mirror
        m = np.asarray(mask0)
        if m.dtype != np.bool_ or m.ndim != 1:
            raise ValueError("assign_primitive_v1: the start set of channel %r must be a one-dimensional boolean mask of the sorted spectrum" % (ch,))
        self.mask0 = m.copy(); self.N = int(m.size); self.n = int(m.sum())
        self.mask = self.mask0.copy(); self.mask_before = self.mask0.copy()
        self.clean = None; self.order = []; self.e_at = {}; self.clean_at = {}
        self.window = []; self.pending = []; self.events = []; self.flags = []
        self.deferred = []; self.kept = {}; self.masks = {}; self.query_flags = {}; self.end_in_window = False

    # --- helpers
    def _defer(self, s, e, U):
        if s in self.ctx.query_set:
            self.deferred.append({"s": s, "e": e, "U": U})

    def _top(self, p):
        order = np.argsort(-p, kind="stable")
        top = np.zeros(self.N, bool); top[order[:self.n]] = True
        return top

    def _set_clean(self, s, e, U, mask):
        idx = np.flatnonzero(mask)
        self.clean = {"s": float(s), "e": e, "U": U, "idx": idx, "ref": (self.ctx.columns(U, idx) if idx.size else None)}
        self.mask = mask

    def _mask_at(self, s):
        m = self.mask0.copy()
        for ev in sorted(self.events, key=_ev_key):
            if ev["decision"] != "diabatic" or ev["s_star"] > s:
                continue
            if ev["chain"] or ev["partner_c"] is None:
                m = ev["mask_after"].copy(); continue
            m[ev["partner_c"]] = False; m[ev["n"]] = True
            mr = ev.get("mirror")
            if mr:
                m[mr["partner_c"]] = False; m[mr["n"]] = True
        return m

    def _flush(self, s_limit):
        rest = []
        for d in self.deferred:
            if d["s"] <= s_limit + 1e-15:
                m = self._mask_at(d["s"]); self.masks[d["s"]] = m
                if self.ctx.keep is not None:
                    self.kept[d["s"]] = self.ctx.keep(d["s"], self.ch, d["e"], d["U"], m)
                self.ctx.release(d["U"])
            else:
                rest.append(d)
        self.deferred = rest

    # --- the points
    def step(self, s, e, U, last):
        s = float(s); e = np.asarray(e, float); ctx = self.ctx
        if e.size != self.N:
            raise ValueError("assign_primitive_v1: channel %r: %d eigenvalues at s=%.6g against a mask of %d levels" % (self.ch, e.size, s, self.N))
        self.order.append(s); self.e_at[s] = e
        if self.clean is None:
            self._set_clean(s, e, U, self.mask0.copy()); self.clean_at[s] = True
            self._defer(s, e, U); self._flush(s)
            if last:
                self._finish()
            return
        if self.pending:
            self._decide_pending(s, e, U)
        self._flush(self.clean["s"])
        if self.n == 0 or self.n == self.N:
            top = self.mask0.copy(); w_in = 1.0; w_out = 0.0; clean = True; p = None
        else:
            p = ctx.project(self.clean["ref"], U)
            top = self._top(p)
            w_in = float(p[top].min()); w_out = float(p[~top].max())
            clean = bool(w_in >= ctx.w_mix and w_out <= 1.0 - ctx.w_mix)
        if clean:
            self._events(s, e, U, top)
            self._set_clean(s, e, U, top.copy()); self.window = []
        else:
            self._window_point(s, e, U, p)
            if s in ctx.query_set:
                self.query_flags.setdefault(s, []).append("the query point s=%.6g inside a crossing's window (min occupied weight %.4f, max empty %.4f): its set is the decided mask, not the point's character mask" % (s, w_in, w_out))
        self.clean_at[s] = clean
        self._defer(s, e, U)
        if last:
            if self.pending:
                self._decide_pending(s, e, U)
            if not clean:
                self.end_in_window = True
                self.flags.append("the chord's end inside a crossing's window (min occupied weight %.4f, max empty %.4f)" % (w_in, w_out))
            self._finish()

    def _window_point(self, s, e, U, p):
        wm = self.ctx.w_mix; pairs = {}
        if p is not None and self.clean["ref"] is not None:
            mixing = np.flatnonzero((p > 1.0 - wm) & (p < wm))
            if mixing.size:
                W = self.ctx.weights(self.clean["ref"], self.ctx.columns(U, mixing))
                for r in np.flatnonzero(W.sum(axis=1) > 1.0 - wm).tolist():
                    o = np.argsort(-W[r], kind="stable")
                    if o.size >= 2 and W[r, o[1]] > 1.0 - wm:
                        k1, k2 = int(mixing[o[0]]), int(mixing[o[1]])
                        pairs[r] = (k1, k2, float(abs(e[k1] - e[k2])))
        self.window.append({"s": float(s), "pairs": pairs})

    def _events(self, s, e, U, top):
        ctx = self.ctx
        old = np.flatnonzero(self.mask & ~top); new = np.flatnonzero(top & ~self.mask)
        if new.size == 0 and old.size == 0:
            return
        c = self.clean; self.mask_before = self.mask.copy()
        wsamp = [(w["s"], w["pairs"]) for w in self.window]
        evs = []
        for n_idx in new.tolist():
            col = ctx.weights(c["ref"], ctx.columns(U, [n_idx]))[:, 0]
            r = int(np.argmax(col)); i = int(c["idx"][r])
            lo_, hi_ = min(i, n_idx), max(i, n_idx); K = (hi_ - lo_) + 3
            cand = list(range(max(0, lo_ - K), min(self.N, hi_ + K + 1)))
            Wc = ctx.weights(ctx.columns(c["U"], cand), ctx.columns(U, cand))
            pi = {}
            for a_, q in enumerate(cand):
                j_ = int(np.argmax(Wc[a_])); pi[q] = (cand[j_], float(Wc[a_, j_]))
            swapped = [q for q in cand if q != i and (q - i) * (pi[q][0] - n_idx) < 0]
            unclear = [q for q in swapped if pi[q][1] < ctx.w_mix]
            passages = [q for q in swapped if not self.mask_before[q]]
            crossed = [q for q in swapped if self.mask_before[q]]
            flags = []
            if unclear:
                flags.append("the continuation of %d swapped level(s) unclear (weight below w_mix %.3g) between s=%.6g and %.6g" % (len(unclear), ctx.w_mix, c["s"], s))
            if len(passages) != 1:
                flags.append("%d passages of the member (%d -> %d, %d crossed members) in one piece between s=%.6g and %.6g: a block at this level" % (len(passages), i, n_idx, len(crossed), c["s"], s))
            single = bool(len(passages) == 1 and not unclear)
            evs.append({"channel": str(self.ch), "level": self.level, "row": r, "i": i, "n": int(n_idx), "w_new": float(col[r]), "s_p": float(c["s"]), "s_c": float(s),
                        "passages": len(passages), "crossed_members": len(crossed), "partner_p": (int(passages[0]) if single else None), "partner_c": (int(pi[passages[0]][0]) if single else None),
                        "w_partner": (pi[passages[0]][1] if single else None), "block": (not single), "chain": False, "partner_ok": single,
                        "window_samples": [(ws, pairs[r][2]) for ws, pairs in wsamp if r in pairs], "mask_after": top.copy(), "flags": flags})
        rows = [ev["row"] for ev in evs]; removed = [ev["partner_c"] for ev in evs if not ev["block"]]
        bad = (len(set(rows)) != len(rows)) or (not set(removed) <= set(old.tolist())) or (len(removed) != len(set(removed)))
        if bad:
            for ev in evs:
                ev["chain"] = True; ev["block"] = True; ev["partner_ok"] = False
                ev["flags"].append("the piece's index changes (left %s, entered %s) are not the sum of single passages at this level" % (old.tolist(), new.tolist()))
        self.pending.extend(evs)

    def _decide_pending(self, s_now, e_now, U_now):
        ctx = self.ctx; c = self.clean; revert_pairs = []; revert_all = False; w = ctx.w(self.ch)
        for ev in self.pending:
            i, n = ev["i"], ev["n"]; wsamp = ev.pop("window_samples")
            samples = []
            if not ev["block"]:
                ep = self.e_at[ev["s_p"]]; samples.append((ev["s_p"], float(abs(ep[ev["partner_p"]] - ep[i]))))
                samples.extend(wsamp)
                ec = self.e_at[ev["s_c"]]; samples.append((ev["s_c"], float(abs(ec[ev["partner_c"]] - ec[n]))))
                if s_now > ev["s_c"] and U_now is not None:
                    km = int(np.argmax(ctx.weights(ctx.columns(c["U"], [n]), U_now)[0])); kp = int(np.argmax(ctx.weights(ctx.columns(c["U"], [ev["partner_c"]]), U_now)[0]))
                    if km != kp:
                        samples.append((float(s_now), float(abs(e_now[km] - e_now[kp]))))
            else:
                samples.extend(wsamp)
            ev["samples"] = sorted(samples)
            res = classify_pair(samples, ctx.rule, tau=ctx.tau, eig_tol=ctx.eig_tol) if not ev["block"] else dict(fit_pair(samples, ctx.eig_tol), alpha_T=None, A=None, A_lo=None, A_hi=None, decision=ctx.rule.default(), decided=False, tie=False, knife=False)
            for key in ("s_star", "gap_min", "gap_min2", "gap_min2_raw", "gap_min2_err", "alpha_s", "alpha_T", "delta_grid", "s_grid", "A", "A_lo", "A_hi", "fit_ok", "fit_why", "misfit", "tie", "knife"):
                ev[key] = res.get(key)
            resolved = bool((not ev["block"]) and ev["partner_ok"] and res["fit_ok"])
            if resolved:
                decision, decided = res["decision"], bool(res["decided"])
            else:
                decision, decided = ctx.rule.default(), False
                if (not ev["block"]) and res.get("fit_why"):
                    ev["flags"].append("the pair's splitting does not fit one crossing: %s" % res["fit_why"])
            if ev["s_star"] is None:
                ev["s_star"] = 0.5 * (ev["s_p"] + ev["s_c"])
            if ev["gap_min2_raw"] is not None and ev["gap_min2_raw"] < -(ev["gap_min2_err"] or 0.0):
                ev["flags"].append("the fitted minimum below zero (%.2e): an exact crossing to this fit, clamped" % ev["gap_min2_raw"])
            if ev["tie"]:
                ev["flags"].append("undecided at this level: the band on tau_LZ [%.3g, %.3g] contains c %.3g; the tie %s applied" % (ev["A_lo"], ev["A_hi"], ctx.rule.c, decision))
            if ev["knife"]:
                ev["flags"].append("knife-edge: tau_LZ %.3g within a factor e^%.2g of c %.3g" % (ev["A"], ctx.rule.knife, ctx.rule.c))
            dg = ev["delta_grid"] if ev["delta_grid"] is not None else None
            j_grid = (ctx.jump_sign * w * dg) if dg is not None else 0.0
            j_fit = (ctx.jump_sign * w * ev["gap_min"]) if ev["gap_min"] is not None else None
            ev.update({"decision": decision, "decided": decided, "resolved": resolved, "mirrored": False, "self_mirror": False, "mirror": None, "jump_prev": None, "falls": None,
                       "gap_min2_prev": None, "gap_converged": None, "matched_prev": False})
            if decision == "diabatic":
                ev.update({"jump_grid": j_grid, "jump_fit": j_fit, "jump_alt": 0.0})
            else:
                ev.update({"jump_grid": 0.0, "jump_fit": 0.0, "jump_alt": j_grid})
                if ev["chain"] or ev["partner_c"] is None:
                    revert_all = True
                else:
                    revert_pairs.append((ev["partner_c"], n))
            self.events.append(ev)
        self.pending = []
        if revert_all:
            self.mask = self.mask_before.copy()
        elif revert_pairs:
            for kp, n in revert_pairs:
                self.mask[n] = False; self.mask[kp] = True
        if revert_all or revert_pairs:
            self._set_clean(c["s"], c["e"], c["U"], self.mask)

    def _merge_mirrors(self):
        mir = self.mirror
        if mir is None:
            return
        evs = sorted(self.events, key=_ev_key); used = set(); out = []
        tol = 2.0 ** (-self.level) + 1e-9
        for a_i, ev in enumerate(evs):
            if a_i in used:
                continue
            if ev["partner_p"] is None or ev["partner_c"] is None:
                out.append(ev); used.add(a_i); continue
            if ev["partner_p"] == mir(ev["i"]):
                ev["self_mirror"] = True; out.append(ev); used.add(a_i); continue
            mi, mn = mir(ev["partner_p"]), mir(ev["partner_c"])
            partner = None
            for b_i in range(a_i + 1, len(evs)):
                if b_i in used:
                    continue
                evb = evs[b_i]
                if evb["i"] == mi and evb["n"] == mn and abs(evb["s_star"] - ev["s_star"]) <= tol:
                    partner = b_i; break
            if partner is None:
                ev["flags"].append("the mirror of this crossing (%d -> %d) not seen at this level" % (mi, mn)); out.append(ev); used.add(a_i); continue
            evb = evs[partner]; used.update((a_i, partner))
            prim, sec = (ev, evb) if ev["i"] <= evb["i"] else (evb, ev)
            prim["mirrored"] = True
            prim["mirror"] = {"i": sec["i"], "n": sec["n"], "partner_p": sec["partner_p"], "partner_c": sec["partner_c"], "s_star": sec["s_star"], "decision": sec["decision"], "gap_min": sec["gap_min"]}
            for key in ("jump_grid", "jump_fit", "jump_alt"):
                if prim.get(key) is not None and sec.get(key) is not None:
                    prim[key] = prim[key] + sec[key]
            if sec["decision"] != prim["decision"]:
                prim["flags"].append("the mirror's decision differs (%s against %s)" % (sec["decision"], prim["decision"])); prim["decided"] = False; prim["resolved"] = False
            for f in sec["flags"]:
                if f not in prim["flags"]:
                    prim["flags"].append(f)
            out.append(prim)
        self.events = sorted(out, key=_ev_key)

    def _finish(self):
        self._merge_mirrors()
        self._flush(2.0)
        for s in self.order:
            self.masks[s] = self._mask_at(s)
        self.clean = None

    def result(self):
        return {"channel": self.ch, "level": self.level, "points": list(self.order), "masks": self.masks, "events": self.events, "flags": list(self.flags), "kept": self.kept,
                "query_flags": self.query_flags, "query_s": list(self.ctx.queries), "end_in_window": self.end_in_window}


def _compare(prev, cur, ctx):
    """the halving-stability and jumps-fall comparison of one channel's consecutive levels."""
    notes = []; stable = True; falls_all = True; matched = 0
    for s in [0.0] + list(cur["query_s"]):
        if (s not in prev["masks"]) or (s not in cur["masks"]) or (not np.array_equal(prev["masks"][s], cur["masks"][s])):
            stable = False; notes.append("the set at s=%.6g differs between levels %d and %d" % (s, prev["level"], cur["level"]))
    ea = sorted(prev["events"], key=_ev_key); eb = sorted(cur["events"], key=_ev_key)
    tol_s = 2.0 ** (-prev["level"]) + 1e-9
    if len(ea) != len(eb):
        stable = False; notes.append("crossings %d at level %d against %d at level %d" % (len(ea), prev["level"], len(eb), cur["level"]))
    else:
        w = ctx.w(cur["channel"])
        for a_, b_ in zip(ea, eb):
            same = ((a_["decision"], int(np.sign(a_["n"] - a_["i"])), bool(a_["mirrored"]), bool(a_["self_mirror"])) == (b_["decision"], int(np.sign(b_["n"] - b_["i"])), bool(b_["mirrored"]), bool(b_["self_mirror"]))
                    and abs(a_["s_star"] - b_["s_star"]) <= tol_s)
            if not same:
                stable = False
                notes.append("crossing %d -> %d (%s, s* %.4f) at level %d against %d -> %d (%s, s* %.4f) at level %d" % (a_["i"], a_["n"], a_["decision"], a_["s_star"], prev["level"], b_["i"], b_["n"], b_["decision"], b_["s_star"], cur["level"]))
                continue
            matched += 1; b_["matched_prev"] = True
            if ctx.rule.kind == "lz" and a_.get("gap_min2") is not None and b_.get("gap_min2") is not None and b_.get("alpha_T"):
                thr = ctx.rule.c * ctx.rule.hbar * abs(b_["alpha_T"]) / ctx.rule.k
                conv = abs(b_["gap_min2"] - a_["gap_min2"]) <= ctx.gap_conv * thr + 2.0 * (b_["gap_min2_err"] or 0.0)
                b_["gap_min2_prev"] = a_["gap_min2"]; b_["gap_converged"] = bool(conv)
                if not conv:
                    stable = False; notes.append("gap_min^2 of the crossing at s* %.4f moved from %.3e to %.3e against the threshold %.3e: not converged" % (b_["s_star"], a_["gap_min2"], b_["gap_min2"], thr))
            if b_["decision"] == "diabatic":
                tol_abs = 8.0 * abs(w) * ctx.eig_tol * (2.0 if b_["mirrored"] else 1.0)
                f = jump_falls(a_["jump_grid"], b_["jump_grid"], b_["jump_fit"], ctx.jump_tol, tol_abs)
                b_["jump_prev"] = a_["jump_grid"]; b_["falls"] = f
                if not f:
                    falls_all = False; notes.append("the jump of the crossing at s* %.4f does not fall: %.3e at level %d, %.3e at level %d, resolved %s" % (b_["s_star"], a_["jump_grid"], prev["level"], b_["jump_grid"], cur["level"], b_["jump_fit"]))
    unresolved = [ev for ev in eb if not (ev["resolved"] and ev["decided"])]
    if unresolved:
        stable = False; notes.append("%d crossing(s) unresolved or undecided at level %d" % (len(unresolved), cur["level"]))
    return {"stable": bool(stable), "jumps_fall": bool(falls_all), "notes": notes, "matched": matched}


class Decision(object):
    """the result of follow_chord: masks_at[s][channel], kept[s][channel], crossings, stable, level_stable, jumps_fall, printable_set, sums, flags, levels."""

    def __init__(self, chans, results, stable_at, comparisons, level_table, n_spectra, ctx, mirror_map, level_min, level_max):
        self.channels = list(chans); self.results = results; self.level_min = int(level_min); self.level_max = int(level_max)
        self.level_stable = {ch: stable_at.get(ch) for ch in chans}
        self.stable = bool(all(stable_at.get(ch) is not None for ch in chans))
        self.level_stable_max = max([stable_at.get(ch) or level_max for ch in chans] or [level_min])
        final = {ch: results[ch][-1] for ch in chans}
        self.query_s = list(ctx.queries)
        self.masks_at = {s: {ch: final[ch]["masks"][s] for ch in chans} for s in [0.0] + self.query_s}
        self.masks_end = self.masks_at[1.0]
        self.kept = {s: {ch: final[ch]["kept"][s] for ch in chans if s in final[ch]["kept"]} for s in self.query_s}
        self.crossings = [ev for ch in chans for ev in final[ch]["events"]]
        self.n_diabatic = sum(1 for ev in self.crossings if ev["decision"] == "diabatic")
        self.n_adiabatic = sum(1 for ev in self.crossings if ev["decision"] == "adiabatic")
        self.n_unresolved = sum(1 for ev in self.crossings if not (ev["resolved"] and ev["decided"]))
        self.comparisons = comparisons
        self.jumps_fall = bool(all(comparisons[ch]["jumps_fall"] for ch in chans if comparisons.get(ch) is not None))
        self.printable_set = bool(self.stable and self.jumps_fall)
        self.jump_grid_sum = float(sum((ev["jump_grid"] or 0.0) for ev in self.crossings))
        self.jump_fit_sum = float(sum((ev["jump_fit"] or 0.0) for ev in self.crossings))
        self.flags = {ch: list(final[ch]["flags"]) for ch in chans}
        self.query_flags = {ch: dict(final[ch]["query_flags"]) for ch in chans}
        self.level_table = level_table; self.n_spectra = int(n_spectra)
        self.mirror = {ch: bool(mirror_map.get(ch) is not None) for ch in chans}
        self.notes = {ch: (comparisons[ch]["notes"] if comparisons.get(ch) else []) for ch in chans}
        self.levels = {ch: [{"level": r["level"], "n_points": len(r["points"]), "n_crossings": len(r["events"]), "n_unresolved": sum(1 for ev in r["events"] if not (ev["resolved"] and ev["decided"])), "flags": list(r["flags"])} for r in results[ch]] for ch in chans}

    def summary(self, samples=True):
        evs = []
        for ev in self.crossings:
            d = {k: v for k, v in ev.items() if k != "mask_after"}
            if not samples:
                d.pop("samples", None)
            evs.append(d)
        return _py({"version": VERSION, "stable": self.stable, "level_stable": self.level_stable, "level_stable_max": self.level_stable_max, "level_min": self.level_min, "level_max": self.level_max,
                    "n_crossings": len(self.crossings), "n_diabatic": self.n_diabatic, "n_adiabatic": self.n_adiabatic, "n_unresolved": self.n_unresolved, "jumps_fall": self.jumps_fall,
                    "printable_set": self.printable_set, "jump_grid_sum": self.jump_grid_sum, "jump_fit_sum": self.jump_fit_sum, "crossings": evs, "flags": self.flags, "query_flags": self.query_flags,
                    "notes": self.notes, "levels": self.levels, "level_table": self.level_table, "n_spectra": self.n_spectra, "mirror": self.mirror, "query_s": self.query_s,
                    "set_changes_at_end": {ch: int(np.count_nonzero(self.masks_at[1.0][ch] != self.masks_at[0.0][ch])) for ch in self.channels}})

    def __repr__(self):
        return "Decision(stable=%s, levels=%s, crossings=%d (%d diabatic, %d adiabatic, %d unresolved), jumps_fall=%s, printable_set=%s, spectra=%d)" % (
            self.stable, self.level_stable, len(self.crossings), self.n_diabatic, self.n_adiabatic, self.n_unresolved, self.jumps_fall, self.printable_set, self.n_spectra)


def follow_chord(oracle, masks0, rule=None, queries=(), tau=1.0, weights_ch=None, level_min=None, level_max=None, w_mix=None, eig_tol=None, jump_tol=None,
                 jump_sign=None, gap_conv=None, mirror="auto", keep=None, channels=None, log=None):
    """The primitive. oracle: see the module docstring; masks0: {channel: boolean mask of the sorted spectrum at s = 0}; rule: a Rule (default lz, c = 1,
    k = 1); queries: the s at which the decided set is wanted (s = 1 always); tau: the chord's physical length (alpha_T = alpha_s / tau); weights_ch:
    {channel: w} for the jumps; level_min .. level_max: the halving levels; w_mix, eig_tol, jump_tol, jump_sign, gap_conv: DEFAULTS; mirror: None, True, a
    callable or "auto"; keep(s, channel, e, U, mask) -> any at the query points; channels: the order (default sorted by str). Returns a Decision."""
    d = DEFAULTS
    level_min = d["level_min"] if level_min is None else int(level_min); level_max = d["level_max"] if level_max is None else int(level_max)
    w_mix = d["w_mix"] if w_mix is None else float(w_mix); eig_tol = d["eig_tol"] if eig_tol is None else float(eig_tol)
    jump_tol = d["jump_tol"] if jump_tol is None else float(jump_tol); jump_sign = d["jump_sign"] if jump_sign is None else float(jump_sign)
    gap_conv = d["gap_conv"] if gap_conv is None else float(gap_conv)
    if level_min < 0:
        raise ValueError("assign_primitive_v1: level_min must be at least 0")
    if level_max < level_min + 1:
        raise ValueError("assign_primitive_v1: level_max must be at least level_min + 1 (stability under halving needs two levels)")
    if not (0.5 < w_mix < 1.0):
        raise ValueError("assign_primitive_v1: w_mix must lie in (0.5, 1)")
    rule = rule if rule is not None else Rule()
    chans = list(channels) if channels is not None else sorted(masks0, key=lambda k_: str(k_))
    for ch in chans:
        if ch not in masks0:
            raise KeyError("assign_primitive_v1: no start set for channel %r" % (ch,))
    ctx = _Ctx(oracle, rule, tau, w_mix, eig_tol, jump_tol, jump_sign, gap_conv, weights_ch, queries, keep, log)
    results = {ch: [] for ch in chans}; stable_at = {}; comparisons = {}; mirror_map = {}
    active = list(chans); n_spectra = 0; level_table = []
    for level in range(level_min, level_max + 1):
        if not active:
            break
        pts = dyadic_points(level, ctx.queries)
        walks = {}
        for q, s in enumerate(pts):
            spec = oracle.spectrum(s, list(active)); n_spectra += len(active)
            for ch in active:
                e, U = spec[ch]
                e = np.asarray(e, float)
                if ch not in mirror_map:
                    mirror_map[ch] = _mirror_map(mirror, e)
                if ch not in walks:
                    walks[ch] = _Walk(ctx, ch, masks0[ch], level, mirror_map[ch])
                walks[ch].step(s, e, U, last=(q == len(pts) - 1))
            spec = None
        row = {"level": level, "channels": [str(c_) for c_ in active], "n_points": len(pts), "n_crossings": 0, "n_unresolved": 0, "stable": []}
        for ch in active:
            res = walks[ch].result(); results[ch].append(res)
            row["n_crossings"] += len(res["events"]); row["n_unresolved"] += sum(1 for ev in res["events"] if not (ev["resolved"] and ev["decided"]))
            if len(results[ch]) >= 2:
                cmp_ = _compare(results[ch][-2], res, ctx); comparisons[ch] = cmp_
                if cmp_["stable"]:
                    stable_at[ch] = level; row["stable"].append(str(ch))
        level_table.append(row)
        ctx.say("assign_primitive_v1 level %d: channels %s at %d points; crossings %d (%d unresolved); stable: %s" % (level, ",".join(row["channels"]), row["n_points"], row["n_crossings"], row["n_unresolved"], ",".join(row["stable"]) or "-"))
        active = [ch for ch in active if ch not in stable_at]
    return Decision(chans, results, stable_at, comparisons, level_table, n_spectra, ctx, mirror_map, level_min, level_max)
