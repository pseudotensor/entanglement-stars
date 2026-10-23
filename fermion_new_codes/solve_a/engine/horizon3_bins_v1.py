#!/usr/bin/env python
"""horizon3_bins_v1.py (the solve lane; designed by reviewer A for the lane's builder; engine spec 49.1 (1)): the one binning of HORIZON
version 3's Killing spectra, imported by compose_horizon_v3_v2.py (its constants and the closure identities; that composer still computes
nothing beyond sums and differences of the files' numbers), export_horizon_refcol_v2.py (the reference column's 5 raw Killing bins and
threshold columns), dust_root_history_v2b.py (the dust columns' Killing bins in an eigenbasis) and horizon3_selftest_v2.py (the hand-counted,
level-by-level and edge-case checks). RULE is the paragraph the three scripts' md files print.
Conventions: lattice units (a = 1), omega in 1/a; b = sqrt(m^2 + 1) the sine chain's band top at the record's m; the 5 bins in BIN_LABELS'
order, each half-open (lo, hi] (a level exactly at an edge belongs to the bin below the edge); the 3 shares below -m = bins 1 + 2,
|omega| < m = bin 3, above m = bins 4 + 5; the threshold content at buffer mu the levels within mu x half-width (inclusive) of one of the 4
edges, half-width b r_s/(2 R_max) at +-b and m r_s/(2 R_max) at +-m; the bins at a buffer read on the remainder; the threshold term of every
bin and share |buffer 1 - buffer 2|. Pure numpy; no engine, no files, no dates.
"""
import math

import numpy as np

SHARES = ("w_below_minus_m", "w_in_gap", "w_above_m")
EDGE_NAMES = ("-b", "-m", "m", "b")
BIN_LABELS = ["(-inf, -b]", "(-b, -m]", "(-m, m]", "(m, b]", "(b, inf)"]
BIN_KEYS = ("w_below_minus_b", "w_below_minus_m", "w_in_gap", "w_above_m", "w_above_b")
DEFAULT_BUFFERS = (1.0, 2.0)

RULE = ("The binning rule (engine spec 49.1 (1); horizon3_bins_v1.py, the one function the export, the dust script and the composer's checks "
        "import): a column's Killing levels are sorted into 5 bins in the order (-inf, -b], (-b, -m], (-m, m], (m, b], (b, inf), with "
        "b = sqrt(m^2 + 1/a^2) the sine chain's band top at the record's m, each bin open at its lower edge and closed at its upper one (a level "
        "exactly at an edge belongs to the bin below it; this matters only for a level exactly at an edge, and the export's recomputation of the "
        "producer's stored spectrum_end on the box, flagged beyond 1e-9, is the check that the producer's binning agrees); the 3 shares are "
        "below -m = bins 1 + 2, |omega| < m = bin 3, above m = bins 4 + 5. The threshold content at buffer mu is every level within "
        "mu x half-width of one of the 4 edges, inclusive (|omega - edge| <= mu x half-width), half-width b r_s/(2 R_max) at +-b and "
        "m r_s/(2 R_max) at +-m with r_s the record's own 2GE and R_max the chain's end; the bins and shares at that buffer are read on the "
        "remaining levels, so 'shares at the buffer + threshold total = raw shares' is an identity, and the threshold term of each bin and share "
        "is |its value at buffer 1 - its value at buffer 2|. Nothing in this rule changed against the version 1 scripts: horizon3_selftest_v1's "
        "four failures (raw 5 bins, raw shares, buffer 1 bins and shares, buffer 2 bins and shares) were one hand-count error in the test's "
        "expectations, the level at omega = -0.5/a (weight 0.10) placed in the gap when at m = 0.2/a it lies in (-b, -m]; the code's "
        "[0.03, 0.17, 0.20, 0.48, 0.12] was the correct count, and the threshold weights, totals, closure and threshold terms passed because that "
        "level lies in no threshold interval. The self-test's version 2 carries the corrected counts, an independent level-by-level count on random "
        "spectra, and the edge cases (levels exactly at +-m and +-b, and at the buffer boundaries).")


def band_top(m):
    """b = sqrt(m^2 + 1/a^2), the sine chain's band top at the record's m (lattice units, a = 1)."""
    return math.sqrt(float(m) * float(m) + 1.0)


def buffer_key(mu):
    """the json key of a buffer: '1', '2' for the integers, else the float's text."""
    return str(int(mu)) if float(mu).is_integer() else str(mu)


def edge_values(m, b):
    """the 4 edges by name."""
    return {"-b": -float(b), "-m": -float(m), "m": float(m), "b": float(b)}


def half_widths(m, b, r_s, R_max):
    """49.1 (1): b r_s/(2 R_max) at +-b, m r_s/(2 R_max) at +-m."""
    return {"b": float(b) * float(r_s) / (2.0 * float(R_max)), "m": float(m) * float(r_s) / (2.0 * float(R_max))}


def half_width_of(name, hw):
    return hw["b"] if name in ("b", "-b") else hw["m"]


def bin_index(e, m, b):
    """the bin (0..4 in BIN_LABELS' order) of each level: (lo, hi], a level exactly at an edge in the bin below it."""
    e = np.asarray(e, float)
    return np.searchsorted(np.array([-float(b), -float(m), float(m), float(b)]), e, side="left")


def killing_bins_5(e, w, m, b):
    """the 5 bins' weights (BIN_LABELS' order) of the weights w at the levels e; w 1-d (one column) -> (5,); w 2-d (levels x columns) -> (5, columns)."""
    e = np.asarray(e, float); w = np.asarray(w, float)
    idx = bin_index(e, m, b)
    out = np.zeros((5,) + w.shape[1:], float)
    for i in range(5):
        out[i] = w[idx == i].sum(axis=0)
    return out


def shares_of_bins5(b5):
    """the 3 shares and the 2 outer bins from the 5 bins: below -m = bins 1 + 2, |omega| < m = bin 3, above m = bins 4 + 5."""
    b5 = np.asarray(b5, float)
    return {"w_below_minus_b": b5[0], "w_below_minus_m": b5[0] + b5[1], "w_in_gap": b5[2], "w_above_m": b5[3] + b5[4], "w_above_b": b5[4]}


def threshold_masks(e, m, b, hw, mu):
    """per edge the levels with |e - edge| <= mu x half-width (inclusive), and their union. The 4 intervals do not overlap on the records
    (b - m is hundreds of half-widths); where they would, a level counts once, at the edge nearest in half-width units."""
    e = np.asarray(e, float); ev = edge_values(m, b); names = list(EDGE_NAMES)
    inside = np.array([np.abs(e - ev[nm]) <= float(mu) * half_width_of(nm, hw) for nm in names]).reshape(len(names), -1)
    multi = inside.sum(axis=0) > 1
    if multi.any():
        units = np.array([np.abs(e - ev[nm]) / half_width_of(nm, hw) for nm in names]).reshape(len(names), -1)
        nearest = np.argmin(units, axis=0)
        for i in range(len(names)):
            inside[i, multi] &= (nearest[multi] == i)
    return {nm: inside[i] for i, nm in enumerate(names)}, inside.any(axis=0)


def weights_in_basis(U, X):
    """|U^dagger X|^2 per eigenvector without copying U: X 1-d (2n,) -> (levels,); X 2-d (2n x columns) -> (levels x columns)."""
    X = np.asarray(X)
    if X.ndim == 1:
        return np.abs((X.conj() @ U).conj()) ** 2
    return np.abs((X.conj().T @ U).conj().T) ** 2


def killing_bins(e, U, X, m, b):
    """the dust script's form: the 5 bins (5 x columns) of the columns X on the eigenvectors U at the levels e, and the mean Killing energy per column."""
    w = weights_in_basis(U, X); ee = np.asarray(e, float)
    if w.ndim == 1:
        w = w[:, None]
    return killing_bins_5(ee, w, m, b), (ee[:, None] * w).sum(axis=0)


def threshold_columns(e, w, m, b, r_s, R_max, buffers=DEFAULT_BUFFERS):
    """engine spec 49.1 (1) on one column's weights w at the levels e: the raw 5 bins and shares; at each buffer mu the threshold content
    (the levels within mu x half-width, inclusive, of each of the 4 edges), the 5 bins and the shares on the remainder; the threshold term of
    every bin and share = |its value at the first buffer - its value at the second|."""
    e = np.asarray(e, float); w = np.asarray(w, float); m = float(m); b = float(b)
    hw = half_widths(m, b, r_s, R_max)
    raw5 = killing_bins_5(e, w, m, b); tot = float(w.sum())
    raw = {"bins5": raw5.tolist()}
    raw.update({k: float(v) for k, v in shares_of_bins5(raw5).items()})
    raw.update({"total": tot, "w_negative": float(w[e < 0].sum()), "mean": float((e * w).sum() / tot) if tot > 0 else None})
    out = {"half_width_b": hw["b"], "half_width_m": hw["m"], "r_s": float(r_s), "R_max": float(R_max), "m": m, "b": b, "bin_labels": list(BIN_LABELS),
           "raw": raw, "buffers": {}, "threshold_error": {}}
    for mu in buffers:
        masks, union = threshold_masks(e, m, b, hw, mu)
        tw = {nm: float(w[inb].sum()) for nm, inb in masks.items()}
        keep = ~union
        b5 = killing_bins_5(e[keep], w[keep], m, b)
        B = {"mu": float(mu), "threshold_weight": tw, "threshold_total": float(sum(tw.values())), "bins5": b5.tolist(), "n_levels_in_threshold": int(union.sum())}
        B.update({k: float(v) for k, v in shares_of_bins5(b5).items()})
        out["buffers"][buffer_key(mu)] = B
    keys = [buffer_key(mu) for mu in buffers]
    if len(keys) >= 2:
        B1, B2 = out["buffers"][keys[0]], out["buffers"][keys[1]]
        for k in BIN_KEYS + ("threshold_total",):
            out["threshold_error"][k] = abs(B1[k] - B2[k])
        out["threshold_error"]["bins5"] = np.abs(np.array(B1["bins5"]) - np.array(B2["bins5"])).tolist()
        out["threshold_error"]["label"] = "the threshold term: |bin(buffer %s) - bin(buffer %s)|" % (keys[0], keys[1])
    return out


def closure_residual(raw_shares, buffer_shares, threshold_total):
    """(the 3 shares at a buffer) + the threshold total - (the 3 raw shares): zero when the bookkeeping closes."""
    return float(sum(float(x) for x in buffer_shares)) + float(threshold_total) - float(sum(float(x) for x in raw_shares))


def error_identity_residual(B1, B2, E, keys=SHARES + ("threshold_total",)):
    """max over the keys of | |B1[k] - B2[k]| - E[k] |: zero when the printed threshold term is |buffer 1 - buffer 2|; None where no key is numeric."""
    vals = []
    for k in keys:
        try:
            vals.append(abs(abs(float(B1[k]) - float(B2[k])) - float(E[k])))
        except (KeyError, TypeError, ValueError):
            pass
    return max(vals) if vals else None
