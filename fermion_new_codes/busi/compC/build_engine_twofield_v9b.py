#!/usr/bin/env python
"""build_engine_twofield_v9b.py (the busi lane, Computation C; the transport ruling busi/notes/C64_transport_ruling_v1.md section 4, order 5.2; the
G0' fix busi/notes/C64_v9_fix_v1.md): writes engine_twofield_v9b.py from engine_twofield_v8.py (input 03, as built and red-teamed) by exact string
replacements, the form of build_engine_twofield_v8.py. Each OLD below is copied from version 8; the only tolerance in the match is the run of spaces
the file uses to align a trailing comment (2 or more spaces before "#" match any such run; the code and the comment text are exact). Every OLD must
occur exactly once in version 8 or the build stops and writes nothing; the result is compiled (a syntax check) before it is written. Every line the
replacements add or change carries the mark "v9 sector" (engine_twofield_v9b_selftest.py marks checks it line by line, blank lines skipped).
The record transport keeps every statement of version 8: where a sector branch is inserted in front of version 8's `if route == "v3":` that line
becomes the `elif` of the same statements; the 4 heavy continuation calls go through continue_ref_v9, which returns continue_ref_v34's result
untouched in record mode; nothing else on the record path is touched (--cut-transport record: version 8 bitwise in both --reference-solve modes,
the ruling's 4.4(a); the json's "script" names this file and its args carry the argument).
Version 9b over version 9 (the same 30 replacements; the v9 block changed, the two window literals made device-safe, the names):
  the G0' self-test failed (c), (e), (d 1e-8) and heavy (c) with one physical crossing counted several times. Version 9 resolved a crossing over
  the piece [s_lo, 2 s1 - s_lo] about the probe point s1 where the ladder failed; by the Weyl step rule s1 lies before the crossing and the
  reflected point need not pass it, so the pair was still within the floor at the piece's end, the walk re-anchored on it, and the "anchor below
  the floor at the piece's start" branch resolved the same pair again over the floor's piece (delta 0, overlap 1), once more after the real
  swap: three entries for one crossing on the light, two on the heavy. Version 9b brackets a crossing by certified probe points: a ladder
  failure opens the anchor's crossing phase (it leaves the step rule, the other anchors' Weyl steps and ladder continue; alone, the probe
  advances from the reflected point by doubling); at every probe the pair is tested for reopening (the banded Cholesky at twice the floor, then
  the pair's gap from the eigendecomposition at the probe, which the assignment needs anyway); at the first probe where it has reopened the
  pair is assigned once over [s_lo, s_hi] and the walk re-anchors there; a pair still within the floor at s = 1 is assigned there and flagged;
  an anchor within the floor at a segment's start opens its crossing phase from the start. The heavy walk the same with its eigenvalues at the
  probes. The parity of the S = +1 sector's boundary count is checked (the mirror rule needs it odd). The crossing entries gain "s_reopen",
  "probes", "pair_gap_end" and "flagged_why"; the sector entry "probes_in_crossing" and "flagged"; the lines print the flagged count.
The changes, by function and place (the build prints each with its version 8 and version 9b line numbers):
  1. the module docstring (version 9b's paragraph before version 8's);
  2. the v9 block before the v8 block: sector_v9, sector_desc_v9, sector_e_v9, boundaries_v9, inversions_run_v9, light_plus_order_v9,
     midgaps_v9, light_select_sector_v9 (the anchor mask of 4.1(i)), light_cut_check_sector_v9, LightPathV9, HeavyPathV9,
     _resolve_crossings_v9 (4.1(ii): the pair/block assignment by OPS3.assign_max_overlap_v3 over the bracketed piece), light_sector_walk_v9
     (4.1: light_walk_v36's loop with one anchor per boundary of the S = +1 sector's occupation and the crossing phase; one anchor and no
     crossing bitwise), light_continue_sector_v9 (the banded route in sector mode), heavy_sector_walk_v9 (4.2: the Weyl bound at every boundary
     with the eigenvalues at the probe points, engine_ops_v34_cert.eigvals_host, and the crossing phase), continue_ref_v9 (continue_ref_v34,
     then the heavy walk when the record-scale Weyl test fails), light_resume_anchor_v9, light_channel_segment_v9 (the light line's per-channel
     text, version 8's in record mode), sector_record_segment_v9 (the record line's sector segment);
  3. light_continue_v36(): the sector dispatch before `if route == "v3":` (that line the `elif`);
  4. light_init_v36(), light_resume_v36() (the resume check gains --cut-transport; the window's gap/mu/n_plus from the anchors in sector
     mode; the window entry's "e" and "sigma"), light_follow_v36(): the window entries carry the spectrum ("e") and the sectors ("sigma");
  5. run(): the run label; the head; light_meta's note; the heavy T = 0 window ("e"); the plain iteration loop's heavy call; the heavy and
     light window entries after acceptance ("e", "sigma"); the heavy and light per-channel json entries gain "sector"; the record's key and
     label; the record line's sector segment; the light line's per-channel segment; the 3 json "script" names;
  6. light_follow_v36(): the json "script" name; light_slice_arrays_v36(): anchors_L<k> (sector mode);
  7. newton_record(), nk_stage_v32(): evalG's heavy call; transport_references_v8(): the heavy call;
  8. main(): the argument --cut-transport {record, sector} and its 2 checks (--reference R2; --light-transport banded with light channels).
Usage: python build_engine_twofield_v9b.py --v8 PATH/engine_twofield_v8.py --out PATH/engine_twofield_v9b.py"""
import argparse
import re
import sys

MARK = "v9 sector"
MARK8 = "v8 lagged (the ruling's v4)"


def _s(x):
    """the snippets below start on the line after the opening quotes: that first newline is not part of them."""
    return x[1:] if x.startswith("\n") else x


def pattern_of(old):
    """old as a regex: exact everywhere except that a run of 2 or more spaces before '#' matches any such run (the comment alignment)."""
    parts = re.split(r"( {2,}#)", old)
    out = []
    for p in parts:
        if re.fullmatch(r" {2,}#", p):
            out.append("[ ]{2,}#")
        else:
            out.append(re.escape(p))
    return re.compile("".join(out))


R = []   # (name, where, old, new): every old occurs exactly once in version 8

# 1. the docstring
R.append(("docstring", "the module docstring: version 9b's paragraph before version 8's", _s(r'''
"""engine_twofield_v8.py (the busi lane, Computation C; the fix ruling busi/notes/C64_fix_ruling_v4.md section I.2, inbox busi #193 item 2): the'''), _s(r'''
"""engine_twofield_v9b.py (the busi lane, Computation C; the transport ruling busi/notes/C64_transport_ruling_v1.md section 4, order 5.2; the G0' fix
busi/notes/C64_v9_fix_v1.md): version 8 with one run argument, --cut-transport {record, sector}, default record, under which every number this file
writes is version 8's in both --reference-solve modes (the json names this script and its args carry the argument; nothing else differs: the heavy
continuation calls go through continue_ref_v9, which returns continue_ref_v34's result untouched in record mode, and the light's banded route is
version 8's block, the sector branch inserted before `if route == "v3":`; the window entries carry the spectrum and the sectors under new keys that
version 8 never reads). Filed as 9b: version 9 as first built counted one physical crossing several times (its refined piece about the probe point
where the ladder failed need not pass the crossing, and the pair, still within the floor at the piece's end, was resolved again over the floor's
piece); 9b brackets a crossing by certified probe points and flags a pair still within the floor at the path's end. Every line the build adds or
changes carries the mark "v9 sector" (build_engine_twofield_v9b.py; engine_twofield_v9b_selftest.py marks checks it line by line). In sector mode
(4.1 to 4.3 of the ruling):
(a) the light: at record n the previous accepted reference's occupied set O in the S = +1 sector's sorted spectrum of h(x_(n-1)) (the stored
    mask, the window entry's "e" and "sigma") has one anchor per boundary, a_j = (e_j + e_(j+1))/2; the S = -1 sector is the conjugate mirror
    (h purely imaginary, [h, S] = 0, the spectrum symmetric), so its boundaries are the mirror images -a_j and the Cholesky certificate of
    K_j(s)^2 - d^2 covers both sectors at once, as the mu S walk's does for its single mu (the mirror rule needs an odd boundary count: the
    lowest + level occupied, the highest empty; an even count is the fallback). Along the straight path from h(x_(n-1)) to h(x_n) the anchors
    move linearly from their midgaps at the segment's start to their midgaps at the trial fields by the same sorted indices (the mu S walk's mu0
    to mu1 for one boundary); a piece is certified when, for every anchor in the step rule, the Cholesky ladder certifies the gap of K_j(s) at
    its probe point above the floor (the floor 3.5e-7, d_min, the ladder, the doubling check as light_walk_v36 has them; one anchor and no
    crossing: light_walk_v36 bitwise, its mu0 the previous record's own mu). A ladder failure at a probe point s1 opens that anchor's crossing
    phase: the anchor leaves the step rule and the ladder (the other anchors' Weyl steps and ladder continue, so the path stays certified for
    them; alone, the probe advances from the reflected point s1 + (s1 - s_lo) by doubling), and at every probe the pair is tested for reopening
    (pd_banded_v36 of K_j(s)^2 - d^2 at d = max(2 (d_eff + 8 eps ||K||_G), 4 tol_block), then the pair's gap from the eigendecomposition at the
    probe, which the assignment needs anyway); at the first probe s_hi where the pair has reopened, its two eigenvectors at s_lo (the last probe
    certified before the failure) and at s_hi are assigned by the declared convention (OPS3.assign_max_overlap_v3 of the block: the
    maximum-overlap rule with its degenerate-block rotation; the weight of the chosen assignment the crossing's overlap), the occupation by
    sorted index updated, the inversion count changing by one when the occupied eigenvector continues on the branch above the anchor or back,
    the boundaries and anchors recomputed from the spectrum at s_hi and the walk resumed there; a pair still within the floor at s = 1 is
    assigned there and flagged ("the pair within the floor at the path's end"); an anchor within the floor at a segment's start (such a
    record's end, or a stored occupation at a near-degenerate pair) opens its crossing phase from the start; more than two levels within the
    floor of a pair at either end of its piece, or two anchors' blocks touching, make one block assigned together and flagged; the reference at
    x_n is the trial eigenvectors under the anchor mask (occupied where the number of anchors a_j with e - a_j sgn(sigma) >= 0 is even:
    light_select_v36's rule for one anchor); the cut check is computed and recorded as now and in sector mode does not take the route (4.4(d)).
(b) the heavy: continue_ref_v34 as now; a record whose Weyl test passes (its gap already the smallest over every boundary, version 3's
    weyl_gap) keeps its result bitwise; where the record-scale test fails, heavy_sector_walk_v9 certifies pieces by the Weyl bound from the
    boundary gaps of the eigenvalues at each probe point (engine_ops_v34_cert.eigvals_host on the host), a boundary gap below the floor at a
    probe point opens the boundary's crossing phase (reopened when its gap less the slack is at least the floor), the crossing resolved as the
    light's over [the probe certified before it, the probe where it reopened] with the heavy's eigendecompositions at both, and the reference
    at x_n is version 3's assignment with the walk's occupation given (engine_ops_v34_cert.assign_occupation_v34: the weights against the
    previous reference keep min_occupied_weight's meaning); its "verified" is True, so the record is not a convention record.
(c) the fallback of last resort is the record-scale convention (OPS3.continue_sea_v3, version 8's fallback), taken when a walk cannot be
    certified (more than --light-max-pieces pieces or probes, more than 128 crossings in one record, the sector impurity, an inconsistent
    count, an even boundary count) and flagged "route": "record-scale fallback" (the heavy's then counts as version 8's LZ convention record).
(d) the records: each light channel's json entry and each heavy continuation entry gain "sector": {"route", "pieces", "cholesky" | "eigvals",
    "eigh", "probes_in_crossing", "anchors_start", "anchors_end", "anchors", "certified_fraction", "crossings": [{"s" (the probe where the
    sub-floor gap was first seen), "piece" [s_lo, s_hi], "s_reopen", "probes", "anchor_energy", "pair", "pair_start", "pair_gap_end", "overlap",
    "inversion_delta", "block", "flagged", "flagged_why", "intermediate", "occupied_before", "occupied_after"}], "flagged", "segments",
    "inversions_plus_start/end" (the heavy: "inversions_start/end"), "adiabatic_at_cut", "min_occupied_weight", "fallback_why"}; the record
    gains "cut_transport": "sector" and its label the sentence of 4.3; the counters TML and TM gain anchors and crossings; the light line's
    per-channel text reads "certified (mu S walk)" where one pocket sufficed and "sector walk: pieces P (Cholesky C) anchors A crossings K
    (min overlap w, max block m, flagged f)" otherwise, "RECORD-SCALE FALLBACK (...)" on a fallback; the record line gains a sector segment;
    "mu" of a channel with several anchors is the first anchor, its "gap" the minimum over anchors, its "n_plus" the + sector's occupied count;
    the slice stores anchors_L<k> (sector mode) beside the mask (mu_L the first anchor); the resume rebuilds the anchors from the mask and the
    spectrum at the slice's fields (one boundary: the slice's mu as version 8), and its argument check gains --cut-transport (a run without
    the argument is the record transport; a resume at a transport change goes through the launcher's rewrite, pod_launch_C_v12b.sh).
Sector mode needs --reference R2 and, with light channels, --light-transport banded. Version 8's docstring follows.
engine_twofield_v8.py (the busi lane, Computation C; the fix ruling busi/notes/C64_fix_ruling_v4.md section I.2, inbox busi #193 item 2): the''')))

# 2. the v9 block, before the v8 block (after light_continue_v36, whose helpers it calls)
BLOCK9 = r'''
# -------------------------------------------------- v9 sector: the sector walk (busi/notes/C64_transport_ruling_v1.md section 4: 4.1 the light, 4.2 the heavy, 4.3 the records, lines and slices); v9b: a crossing bracketed by certified probe points (busi/notes/C64_v9_fix_v1.md)
def sector_v9(a):                                                                                        # v9 sector
    """v9 sector: whether --cut-transport sector is on; False on every path version 8 ran, whose statements are then the ones executed (the record transport, bitwise)."""   # v9 sector
    return getattr(a, "cut_transport", "record") == "sector"                                            # v9 sector


def sector_desc_v9():                                                                                    # v9 sector
    """v9 sector: the scheme's sentence (the run label and the head)."""                                # v9 sector
    return ("where a reference's occupation cut has closed, the occupied subspace is continued piecewise along the straight path between the accepted fields with the "   # v9 sector
            "lane's declared maximum-overlap convention applied to each crossing at the scale of the crossing and the walk certified at every boundary of the occupied set "   # v9 sector
            "(the sector walk: on the light the mu S walk's banded Cholesky certificate at one anchor per boundary of the S = +1 sector's occupation, the anchors the midgaps, "   # v9 sector
            "linear in s from the previous record's to the trial fields' by the same sorted indices, the S = -1 sector the conjugate mirror; on the heavy the Weyl bound at one "   # v9 sector
            "anchor per boundary of the occupation with the eigenvalues at the probe points; a sub-floor approach at an anchor resolved by the convention on the crossing pair over "   # v9 sector
            "the piece bracketed by the last probe point certified before it and the first probe point at which the pair has reopened above the floor, the pair's eigenvectors at "   # v9 sector
            "the piece's ends, the occupation updated, the walk re-anchored and resumed; a block of more than two levels, two anchors' blocks touching, or a pair still within the "   # v9 sector
            "floor at the path's end flagged); the record-scale convention is the fallback of last resort and is flagged (engine_twofield_v9b --cut-transport sector; "   # v9 sector
            "busi/notes/C64_transport_ruling_v1.md 4.1 to 4.3; busi/notes/C64_v9_fix_v1.md)")             # v9 sector


def sector_e_v9(e):                                                                                      # v9 sector
    """v9 sector: a channel's eigenvalues (or sector signs) as a float array on the host (the window entries' "e" and "sigma"; a tensor on any device, or an array)."""   # v9 sector
    ev = e.cpu().numpy() if hasattr(e, "cpu") else np.asarray(e)                                         # v9 sector
    return np.asarray(ev.real if np.iscomplexobj(ev) else ev, float)                                     # v9 sector


def boundaries_v9(occ):                                                                                  # v9 sector
    """v9 sector, 4.1: the boundaries of an occupation over the levels in energy order: the indices j with exactly one of j, j + 1 occupied."""   # v9 sector
    occ = np.asarray(occ, bool)                                                                          # v9 sector
    return np.where(occ[:-1] != occ[1:])[0]                                                              # v9 sector


def inversions_run_v9(occ):                                                                              # v9 sector
    """v9 sector: occupied levels above the lowest empty level (one sector's, or the heavy's, inversion count over the levels in energy order)."""   # v9 sector
    occ = np.asarray(occ, bool)                                                                          # v9 sector
    if not occ.any() or occ.all():                                                                       # v9 sector
        return 0                                                                                         # v9 sector
    return int(np.sum(occ[int(np.argmax(~occ)):]))                                                      # v9 sector


def light_plus_order_v9(e_np, sig):                                                                      # v9 sector
    """v9 sector: the S = +1 sector's levels in energy order: (their indices into the level list, their energies)."""   # v9 sector
    e_np = np.asarray(e_np, float); sel = np.where(np.asarray(sig, float) > 0.0)[0]                        # v9 sector
    order = sel[np.argsort(e_np[sel], kind="stable")]                                                    # v9 sector
    return order, e_np[order]                                                                            # v9 sector


def midgaps_v9(ep, bnds):                                                                                # v9 sector
    """v9 sector, 4.1: the anchors at the boundaries bnds of the sorted spectrum ep, a_j = (e_j + e_(j+1))/2 (light_mu_midgap_v36's expression; one boundary: that mu, bitwise)."""   # v9 sector
    return [0.5 * (ep[j] + ep[j + 1]) for j in bnds]                                                     # v9 sector


def light_select_sector_v9(e_np, sig, anchors):                                                          # v9 sector
    """v9 sector, 4.1(i): the anchor mask: a level (e, sigma) is occupied when the number of anchors a_j with e - a_j sgn(sigma) >= 0 is even (one anchor: light_select_v36's e - mu sgn(sigma) < 0, bitwise; with an odd number of anchors the S = -1 level at -e is occupied exactly where the + level at e is empty, the mirror rule); the gap the minimum over anchors of min |e - a_j sgn(sigma)| over the levels (one anchor: light_select_v36's gap, bitwise)."""   # v9 sector
    sgn = np.where(np.asarray(sig, float) > 0.0, 1.0, -1.0)                                              # v9 sector
    e_np = np.asarray(e_np, float)                                                                       # v9 sector
    n_above = np.zeros(len(e_np), dtype=np.int64); gap = None                                            # v9 sector
    for a_ in anchors:                                                                                   # v9 sector
        kv = e_np - a_ * sgn                                                                             # v9 sector
        n_above += (kv >= 0.0).astype(np.int64)                                                          # v9 sector
        g_ = float(np.min(np.abs(kv)))                                                                   # v9 sector
        gap = g_ if gap is None else min(gap, g_)                                                        # v9 sector
    return (n_above % 2 == 0), gap                                                                       # v9 sector


def light_cut_check_sector_v9(ref_prev, e_np, sig, occ, U, anchors, nG, gap_end):                        # v9 sector
    """v9 sector: light_cut_check_v3 with the anchors in place of the single mu (the levels within reach of any anchor, |e - a_j sgn(sigma)| <= 4 ||dK||_G + 2 gap_end, in both sectors): the counts and the extreme weights; crossing_levels counts the levels whose weight on the previous reference disagrees with their occupation (a swapped pair counts 2 in each sector, 4 with its conjugate mirror); in sector mode the verdict is recorded (crossing_levels > 0: the certified walk passed an above-floor avoided crossing at an anchor adiabatically) and does not take the route (4.4(d))."""   # v9 sector
    torch = BK["torch"]                                                                                  # v9 sector
    sgn = np.where(np.asarray(sig, float) > 0.0, 1.0, -1.0); e_np = np.asarray(e_np, float)              # v9 sector
    kmin = None                                                                                          # v9 sector
    for a_ in anchors:                                                                                   # v9 sector
        kv = np.abs(e_np - a_ * sgn); kmin = kv if kmin is None else np.minimum(kmin, kv)                # v9 sector
    half = 4.0 * float(nG) + 2.0 * float(gap_end)                                                        # v9 sector
    idx = np.where(kmin <= half)[0]                                                                      # v9 sector
    out = {"window_levels": int(len(idx)), "window_half_width": half, "min_weight_selected": None, "max_weight_unselected": None, "crossing_levels": 0, "anchors": int(len(anchors))}   # v9 sector
    if len(idx) == 0:                                                                                    # v9 sector
        return out                                                                                       # v9 sector
    Uw = U[:, torch.as_tensor(idx, dtype=torch.int64, device=U.device)]                                  # v9 sector
    w = (ref_prev.conj().T @ Uw).abs().pow(2).sum(dim=0).cpu().numpy().astype(float)                     # v9 sector
    o = np.asarray(occ, bool)[idx]                                                                       # v9 sector
    if o.any():                                                                                          # v9 sector
        out["min_weight_selected"] = float(w[o].min())                                                   # v9 sector
    if (~o).any():                                                                                       # v9 sector
        out["max_weight_unselected"] = float(w[~o].max())                                                # v9 sector
    out["crossing_levels"] = int(np.sum(o & (w < 0.5)) + np.sum((~o) & (w > 0.5)))                       # v9 sector
    return out                                                                                           # v9 sector


class LightPathV9:                                                                                       # v9 sector
    """v9 sector: the straight path from the previous accepted fields to the trial fields (h_m linear in N, v): h_host(s) the host CSR of h(s) (clocked_host_sparse; the endpoints' own arrays at s = 0 and s = 1, as light_walk_v36 builds K0 and K1), eig(s) the sectorized eigendecomposition at s (the trial's own, rotated in place by light_sectorize_v36, at s = 1)."""   # v9 sector
    def __init__(self, ch, f_prev, f_cur, sgn_dev, tol_block, trial):                                     # v9 sector
        self.ch = ch; self.f0 = (f_prev[0], f_prev[1]); self.f1 = (f_cur[0], f_cur[1]); self.sgn_dev = sgn_dev; self.tol_block = tol_block; self.trial = trial; self.eigh_n = 0   # v9 sector
    def fields(self, s):                                                                                 # v9 sector
        if s <= 0.0:                                                                                     # v9 sector
            return self.f0                                                                               # v9 sector
        if s >= 1.0:                                                                                     # v9 sector
            return self.f1                                                                               # v9 sector
        N0, v0 = np.asarray(self.f0[0], float), np.asarray(self.f0[1], float); N1, v1 = np.asarray(self.f1[0], float), np.asarray(self.f1[1], float)   # v9 sector
        return (N0 + s * (N1 - N0), v0 + s * (v1 - v0))                                                  # v9 sector
    def h_host(self, s):                                                                                 # v9 sector
        N, v = self.fields(s)                                                                            # v9 sector
        return clocked_host_sparse(self.ch, N, v)                                                        # v9 sector
    def eig(self, s):                                                                                    # v9 sector
        if s >= 1.0 and self.trial is not None:                                                          # v9 sector
            return self.trial                                                                            # v9 sector
        N, v = self.fields(s)                                                                            # v9 sector
        e, U = eigT(clocked_gpu(self.ch, N, v)); self.eigh_n += 1                                        # v9 sector
        sigma, _n_rot, _imp = light_sectorize_v36(e, U, self.sgn_dev, self.tol_block)                    # v9 sector
        return e.cpu().numpy().astype(float), sector_e_v9(sigma), U                                      # v9 sector


class HeavyPathV9:                                                                                       # v9 sector
    """v9 sector: a heavy channel's straight path between the previous accepted fields and the trial fields: h_host(s) the host CSR, eigvals(s) the banded eigenvalues on the host (engine_ops_v34_cert.eigvals_host, sorted), eig(s) the eigendecomposition on the device (eig_at; the trial's own at s = 1)."""   # v9 sector
    def __init__(self, ch, f_prev, f_cur, trial):                                                        # v9 sector
        self.ch = ch; self.f0 = (f_prev[0], f_prev[1]); self.f1 = (f_cur[0], f_cur[1]); self.trial = trial; self.eigh_n = 0; self.eigvals_n = 0   # v9 sector
    def fields(self, s):                                                                                 # v9 sector
        if s <= 0.0:                                                                                     # v9 sector
            return self.f0                                                                               # v9 sector
        if s >= 1.0:                                                                                     # v9 sector
            return self.f1                                                                               # v9 sector
        N0, v0 = np.asarray(self.f0[0], float), np.asarray(self.f0[1], float); N1, v1 = np.asarray(self.f1[0], float), np.asarray(self.f1[1], float)   # v9 sector
        return (N0 + s * (N1 - N0), v0 + s * (v1 - v0))                                                  # v9 sector
    def h_host(self, s):                                                                                 # v9 sector
        N, v = self.fields(s)                                                                            # v9 sector
        return clocked_host_sparse(self.ch, N, v)                                                        # v9 sector
    def eigvals(self, s):                                                                                # v9 sector
        self.eigvals_n += 1                                                                              # v9 sector
        return np.asarray(CERT.eigvals_host(self.h_host(s)), float)                                      # v9 sector
    def eig(self, s):                                                                                    # v9 sector
        if s >= 1.0 and self.trial is not None:                                                          # v9 sector
            return self.trial                                                                            # v9 sector
        N, v = self.fields(s)                                                                            # v9 sector
        e, U = eig_at(self.ch, N, v); self.eigh_n += 1                                                   # v9 sector
        return sector_e_v9(e), U                                                                         # v9 sector


def _resolve_crossings_v9(path, light, bnds, failing_js, occ_sorted, s_lo, s_hi, s_at_js, d_eff_js, tol_block, eig_lo=None, eig_hi=None, probes_js=None, flag_js=None):   # v9 sector (v9b)
    """v9 sector, 4.1(ii) and 4.2 (v9b): the pairs at the failing anchors assigned by the declared convention over the piece [s_lo, s_hi]: s_lo the last probe point certified before the pair's gap fell below the floor, s_hi the first probe point at which the pair has reopened above it (the path's end, flagged by the caller, where it has not); the eigendecompositions at the piece's ends (given, or computed here); the block of a failing anchor its pair (j, j + 1 in the sorted order the walk uses: the S = +1 sector's on the light, every level's on the heavy) extended by the levels within the floor of it at either end and merged with a touching block of another failing anchor (a block larger than 2, or a merged one, is flagged); OPS3.assign_max_overlap_v3 of the block's previously occupied vectors at s_lo onto the block's eigenvectors at s_hi (the convention's maximum-overlap rule with its degenerate-block rotation; c_LZ enters the record-scale convention only through its sub-step count, the piece being the sub-step here); the weight of the chosen assignment is the crossing's overlap; the occupation by sorted index updated. Returns (occ_sorted, the crossing entries, (e_hi, sig_hi, U_hi, order_hi, ep_hi))."""   # v9 sector
    torch = BK["torch"]                                                                                  # v9 sector
    if light:                                                                                            # v9 sector
        e_lo, sig_lo, U_lo = path.eig(s_lo) if eig_lo is None else eig_lo                                # v9 sector
        e_hi, sig_hi, U_hi = path.eig(s_hi) if eig_hi is None else eig_hi                                # v9 sector
        order_lo, ep_lo = light_plus_order_v9(e_lo, sig_lo); order_hi, ep_hi = light_plus_order_v9(e_hi, sig_hi)   # v9 sector
    else:                                                                                                # v9 sector
        e_lo, U_lo = path.eig(s_lo) if eig_lo is None else eig_lo                                        # v9 sector
        e_hi, U_hi = path.eig(s_hi) if eig_hi is None else eig_hi; sig_hi = None                         # v9 sector
        ep_lo = np.asarray(e_lo, float); ep_hi = np.asarray(e_hi, float); order_lo = np.arange(len(ep_lo)); order_hi = np.arange(len(ep_hi))   # v9 sector
    occ = np.asarray(occ_sorted, bool).copy(); n = len(occ); nf = len(failing_js)                        # v9 sector
    s_at_js = list(s_at_js) if s_at_js is not None else [float(s_hi)] * nf                               # v9 sector
    probes_js = list(probes_js) if probes_js is not None else [0] * nf                                   # v9 sector
    flag_js = list(flag_js) if flag_js is not None else [None] * nf                                      # v9 sector
    blocks = []                                                                                          # v9 sector
    for k_ in range(nf):                                                                                 # v9 sector
        j = failing_js[k_]; d_eff = d_eff_js[k_]; b = int(bnds[j]); lo_i = b; hi_i = b + 1               # v9 sector
        while lo_i > 0 and (ep_hi[lo_i] - ep_hi[lo_i - 1] < d_eff or ep_lo[lo_i] - ep_lo[lo_i - 1] < d_eff):   # v9 sector
            lo_i -= 1                                                                                    # v9 sector
        while hi_i < n - 1 and (ep_hi[hi_i + 1] - ep_hi[hi_i] < d_eff or ep_lo[hi_i + 1] - ep_lo[hi_i] < d_eff):   # v9 sector
            hi_i += 1                                                                                    # v9 sector
        why = (["a block of %d levels within the floor" % (hi_i - lo_i + 1)] if hi_i - lo_i + 1 > 2 else []) + ([flag_js[k_]] if flag_js[k_] else [])   # v9 sector
        blocks.append([lo_i, hi_i, [k_], why])                                                           # v9 sector
    blocks.sort(key=lambda q: q[0]); merged = []                                                         # v9 sector
    for blk in blocks:                                                                                   # v9 sector
        if merged and blk[0] <= merged[-1][1] + 1:                                                       # v9 sector: two anchors' blocks touching within one piece: one block, flagged
            merged[-1][1] = max(merged[-1][1], blk[1]); merged[-1][2] += blk[2]; merged[-1][3] = merged[-1][3] + blk[3] + ["two anchors' crossings in one piece"]   # v9 sector
        else:                                                                                            # v9 sector
            merged.append(blk)                                                                           # v9 sector
    out = []                                                                                             # v9 sector
    for lo_i, hi_i, ks, why in merged:                                                                   # v9 sector
        idx = np.arange(lo_i, hi_i + 1); occ_b = occ[idx].copy(); n_occ_b = int(occ_b.sum()); size = int(len(idx))   # v9 sector
        inv_before = inversions_run_v9(occ); overlap = None; mask_b = occ_b.copy(); n_inter = 0           # v9 sector
        if 0 < n_occ_b < size:                                                                           # v9 sector
            cols_lo = torch.as_tensor(order_lo[idx][occ_b].astype(np.int64), device=U_lo.device)        # v9 sector
            cols_hi = torch.as_tensor(order_hi[idx].astype(np.int64), device=U_hi.device)                # v9 sector
            _ref_b, info_b, mask_b = OPS3.assign_max_overlap_v3(U_lo[:, cols_lo], ep_hi[idx], U_hi[:, cols_hi], n_occ_b, tol_block)   # v9 sector
            mask_b = np.asarray(mask_b, bool); overlap = info_b.get("min_occupied_weight"); n_inter = int(info_b.get("n_intermediate") or 0)   # v9 sector
            occ[idx] = mask_b                                                                            # v9 sector
        delta = int(inversions_run_v9(occ) - inv_before)                                                 # v9 sector
        for q_, k_ in enumerate(ks):                                                                     # v9 sector
            b = int(bnds[failing_js[k_]])                                                                # v9 sector
            out.append({"s": float(s_at_js[k_]), "piece": [float(s_lo), float(s_hi)], "s_reopen": float(s_hi), "probes": int(probes_js[k_]), "anchor_energy": float(0.5 * (ep_lo[b] + ep_lo[b + 1])),   # v9 sector
                        "pair": [float(ep_hi[b]), float(ep_hi[b + 1])], "pair_start": [float(ep_lo[b]), float(ep_lo[b + 1])], "pair_gap_end": float(ep_hi[b + 1] - ep_hi[b]),   # v9 sector
                        "overlap": (float(overlap) if overlap is not None else None), "inversion_delta": (delta if q_ == 0 else 0), "swap": (bool(not np.array_equal(occ_b, mask_b)) if q_ == 0 else False), "block": size, "flagged": bool(why),   # v9 sector (review): the swap of the block's occupation by sorted index, reorientation 5's crossing, apart from the inversion count's change
                        "flagged_why": ("; ".join(why) if why else None), "intermediate": n_inter, "occupied_before": [int(k) for k in idx[occ_b]], "occupied_after": [int(k) for k in idx[mask_b]]})   # v9 sector
    return occ, out, (e_hi, sig_hi, U_hi, order_hi, ep_hi)                                               # v9 sector


def light_sector_walk_v9(path, S_host, e0_np, sig0, mask0, trial, n_sea, d_min, max_pieces, tol_block, mu_prev=None, step_scale=1.0, max_crossings=128):   # v9 sector
    """v9 sector, 4.1 (v9b): the sector walk over K_j(s) = h(s) - a_j(s) S, one anchor per boundary of the S = +1 sector's occupation (mask0 over the levels of e0_np with sectors sig0), the anchors linear in s from their midgaps at the segment's start to their midgaps at the trial fields by the same sorted indices; light_walk_v36's piece loop (the Weyl step d_rig/||dK||_G, the Cholesky ladder from the certified d down to the floor, the doubling check) at every anchor in the step rule on every probe point, the step the smallest anchor's. A ladder failure at a probe point s1 opens that anchor's crossing phase: it leaves the step rule and the ladder (the other anchors' continue, so the path stays certified for them; alone, the probe advances from the reflected point s1 + (s1 - s_lo) by doubling), and at every probe the pair is tested for reopening (pd_banded_v36 at d = max(2 (d_eff + slack), 4 tol_block), then the pair's gap from the eigendecomposition at the probe, which the assignment needs anyway); at the first probe s_hi where it has reopened the crossing is resolved by _resolve_crossings_v9 over [s_lo, s_hi], s_lo the probe point certified before the failure, the walk re-anchored from the spectrum at s_hi and resumed; a pair still within the floor at s = 1 is resolved there, flagged; an anchor within the floor at a segment's start opens its crossing phase from the start. One anchor and no crossing: light_walk_v36 bitwise (mu_prev the previous record's own mu as its mu0; step_scale 1). Returns (cert, sector, anchors_end, occ_plus_end): cert light_walk_v36's certificate dict (its keys), sector the record entry of 4.3; on a failure cert["certified"] is False with its why and the caller takes the record-scale fallback."""   # v9 sector
    t0 = time.time(); eps = float(np.finfo(float).eps)                                                 # v9 sector
    e1_np, sig1, U1 = trial                                                                              # v9 sector
    e0_np = np.asarray(e0_np, float); sig0 = np.asarray(sig0, float)                                     # v9 sector
    order0, ep0 = light_plus_order_v9(e0_np, sig0); order1, ep1 = light_plus_order_v9(e1_np, sig1)      # v9 sector
    occp = np.asarray(mask0, bool)[order0].copy()                                                        # v9 sector
    cert = {"norm_dK_G": 0.0, "mu0": None, "mu1": None, "gap_start": None, "pieces": 0, "cholesky_calls": 0, "d_min_certified": None, "certified": False, "why": None, "smallest_piece": None, "piece_list": None, "seconds": 0.0, "norm_K_G": 0.0, "rounding_F": 0.0, "d_floor": 0.0, "d_min_effective": 0.0}   # v9 sector
    sector = {"route": "sector walk", "pieces": 0, "cholesky": 0, "eigh": 0, "segments": 0, "probes_in_crossing": 0, "anchors_start": int(len(boundaries_v9(occp))), "anchors_end": 0, "anchors": [], "certified_fraction": 0.0, "crossings": [], "flagged": 0, "inversions_plus_start": inversions_run_v9(occp), "inversions_plus_end": None, "adiabatic_at_cut": 0, "fallback_why": None}   # v9 sector
    pieces = []; crossings = []; nchol = 0; dmin_all = None; cross_len = 0.0; segments = 0; n_probe_cross = 0; end_resolved = set()   # v9 sector (review): the boundaries a crossing phase resolved at s = 1
    def done(why):                                                                                       # v9 sector
        cert["why"] = why; cert["seconds"] = time.time() - t0; cert["cholesky_calls"] = nchol            # v9 sector
        sector.update({"fallback_why": why, "route": "record-scale fallback", "crossings": crossings, "pieces": len(pieces), "cholesky": nchol, "probes_in_crossing": n_probe_cross, "eigh": int(getattr(path, "eigh_n", 0))})   # v9 sector
        return cert, sector, None, None                                                                  # v9 sector
    if len(ep1) != len(ep0) or len(ep0) == 0:                                                            # v9 sector
        return done("the S = +1 sector counts differ or are empty (%d at the start, %d at the trial fields)" % (len(ep0), len(ep1)))   # v9 sector
    if sector["anchors_start"] == 0:                                                                     # v9 sector
        return done("no boundary of the occupation in the S = +1 sector")                                # v9 sector
    if sector["anchors_start"] % 2 == 0:                                                                 # v9 sector: the mirror rule (the - sector occupied where the + sector is empty) needs an odd boundary count: the lowest + level occupied and the highest empty
        return done("an even number of boundaries (%d) in the S = +1 sector: the lowest level empty or the highest occupied, outside the mirror rule" % sector["anchors_start"])   # v9 sector
    h1 = path.h_host(1.0)                                                                                # v9 sector
    s_seg = 0.0; h_seg = path.h_host(0.0); e_seg = e0_np; sig_seg = sig0; ep_seg = ep0; seg = None; s_cur = 0.0   # v9 sector
    crossing = {}; cross_step = None                                                                     # v9 sector: the anchors in their crossing phase by their boundary's level index ({"s_lo", "s_fail", "probes"}); the doubling probe step while any anchor is crossing, in s
    while True:                                                                                          # v9 sector
        if seg is None:                                                                                  # v9 sector: a segment from s_seg (its spectrum, its h) to s = 1: one anchor per boundary of occp, linear in s from its midgap at s_seg to its midgap at the trial fields by the same + indices (the mu S walk's mu0 to mu1 for one boundary)
            bnds = boundaries_v9(occp)                                                                   # v9 sector
            if len(bnds) == 0:                                                                           # v9 sector
                return done("no boundary of the occupation in the S = +1 sector after a crossing")      # v9 sector
            if len(bnds) % 2 == 0:                                                                       # v9 sector
                return done("an even number of boundaries (%d) in the S = +1 sector after a crossing, outside the mirror rule" % len(bnds))   # v9 sector
            a_seg = midgaps_v9(ep_seg, bnds); a_end = midgaps_v9(ep1, bnds)                              # v9 sector
            if segments == 0 and len(bnds) == 1 and mu_prev is not None:                                 # v9 sector: one boundary: the previous record's own mu (light_walk_v36's mu0), bitwise
                a_seg = [mu_prev]                                                                        # v9 sector
            sgn_seg = np.where(sig_seg > 0.0, 1.0, -1.0)                                                 # v9 sector
            seg = {"bnds": [int(b) for b in bnds], "a0": a_seg, "a1": a_end, "K0": [], "dK": [], "nG": [], "GK": [], "F": [], "d_eff": [], "slack": [], "d": [], "from_eig": [], "g0": []}   # v9 sector
            for a0, a1 in zip(a_seg, a_end):                                                             # v9 sector
                K0 = (h_seg - a0 * S_host).tocsr(); K1 = (h1 - a1 * S_host).tocsr(); dK = (K1 - K0).tocsr()   # v9 sector
                nG = OPS3.gershgorin(dK); GK = max(OPS3.gershgorin(K0), OPS3.gershgorin(K1))              # v9 sector
                F = 64.0 * eps * GK * GK; d_eff = max(float(d_min), 2.0 * math.sqrt(F))                  # v9 sector
                g0 = float(np.min(np.abs(e_seg - a0 * sgn_seg)))                                         # v9 sector
                for k_, v_ in (("K0", K0), ("dK", dK), ("nG", nG), ("GK", GK), ("F", F), ("d_eff", d_eff), ("slack", 8.0 * eps * GK), ("g0", g0), ("d", g0 - 8.0 * eps * GK), ("from_eig", True)):   # v9 sector
                    seg[k_].append(v_)                                                                   # v9 sector
            if segments == 0:                                                                            # v9 sector
                cert.update({"norm_dK_G": max(seg["nG"]), "mu0": a_seg[0], "mu1": a_end[0], "gap_start": min(seg["g0"]), "norm_K_G": max(seg["GK"]), "rounding_F": max(seg["F"]), "d_floor": 2.0 * math.sqrt(max(seg["F"])), "d_min_effective": max(seg["d_eff"])})   # v9 sector
            segments += 1; s_cur = s_seg                                                                 # v9 sector
            crossing = {b: q for b, q in crossing.items() if b in seg["bnds"]}                           # v9 sector: the anchors still in their crossing phase, carried across the re-anchoring by their boundary index
        m = len(seg["bnds"]); bl = seg["bnds"]                                                           # v9 sector
        if all(nG_ == 0.0 for nG_ in seg["nG"]):                                                         # v9 sector: no field change (light_walk_v36's nG == 0: certified, no piece)
            if dmin_all is None:                                                                         # v9 sector
                dmin_all = min(seg["g0"])                                                                # v9 sector
            break                                                                                        # v9 sector
        for j in range(m):                                                                               # v9 sector: an anchor within the floor at this point (the segment's start: a pair still within the floor from the previous record's end, or a stored occupation at a near-degenerate pair) opens its crossing phase from here
            if bl[j] not in crossing and not (seg["d"][j] >= seg["d_eff"][j]):                           # v9 sector
                crossing[bl[j]] = {"s_lo": s_cur, "s_fail": s_cur, "probes": 0}                          # v9 sector
        if crossing and cross_step is None:                                                              # v9 sector
            cross_step = min(seg["d_eff"][j] / seg["nG"][j] for j in range(m) if seg["nG"][j] > 0.0) * (1.0 - s_seg)   # v9 sector: the floor's Weyl length, in s
        others = [j for j in range(m) if bl[j] not in crossing]                                          # v9 sector
        step = None                                                                                      # v9 sector
        if others:                                                                                       # v9 sector
            d_rig = [(seg["d"][j] if seg["from_eig"][j] else math.sqrt(max(seg["d"][j] * seg["d"][j] - seg["F"][j], 0.0))) for j in range(m)]   # v9 sector
            step = min(d_rig[j] / seg["nG"][j] for j in others if seg["nG"][j] > 0.0)                   # v9 sector
            if step_scale != 1.0:                                                                        # v9 sector
                step = step * step_scale                                                                 # v9 sector
            step = step * (1.0 - s_seg)                                                                  # v9 sector: the Weyl step in the segment's parameter, in s (a factor 1.0 on the first segment: bitwise)
        if crossing:                                                                                     # v9 sector
            if step is None or cross_step <= step:                                                      # v9 sector (review): the doubling only when the crossing probe governs the step; under the others' Weyl steps cross_step holds
                step = cross_step; cross_step = 2.0 * cross_step                                         # v9 sector
        s1 = min(1.0, s_cur + step)                                                                      # v9 sector
        if others:                                                                                       # v9 sector
            d_piece = min(seg["d"][j] for j in others)                                                   # v9 sector
            pieces.append((s_cur, s1, d_piece)); dmin_all = d_piece if dmin_all is None else min(dmin_all, d_piece)   # v9 sector
        if crossing:                                                                                     # v9 sector
            n_probe_cross += 1                                                                           # v9 sector
            for q in crossing.values():                                                                  # v9 sector
                q["probes"] += 1                                                                         # v9 sector
        if s1 >= 1.0 and not crossing:                                                                   # v9 sector: the last piece certified by the Weyl step (light_walk_v36)
            break                                                                                        # v9 sector
        if len(pieces) + n_probe_cross >= max_pieces:                                                    # v9 sector
            return done("more than %d pieces or probes (||dK||_G %.3e, d %.3e)" % (max_pieces, max(seg["nG"]), min(seg["d"])))   # v9 sector
        t1 = (s1 - s_seg) / (1.0 - s_seg)                                                                # v9 sector: the segment's parameter at the probe (s1 itself on the first segment: bitwise)
        new_d = list(seg["d"]); newly = []                                                               # v9 sector
        for j in others:                                                                                 # v9 sector: the ladder at the probe point for the anchors in the step rule
            Ks = (seg["K0"][j] + t1 * seg["dK"][j]).tocsr()                                              # v9 sector
            ab, _u = band_upper_v36(Ks @ Ks)                                                             # v9 sector
            d_new = None; d_try = seg["d"][j]                                                            # v9 sector
            while d_try >= seg["d_eff"][j]:                                                              # v9 sector
                nchol += 1                                                                               # v9 sector
                if pd_banded_v36(ab, d_try):                                                             # v9 sector
                    d_new = d_try                                                                        # v9 sector
                    break                                                                                # v9 sector
                d_try *= 0.5                                                                             # v9 sector
            if d_new is None:                                                                            # v9 sector: the gap at the probe point below the floor: the anchor's crossing phase opens, s_lo the probe point certified before it
                newly.append(j)                                                                          # v9 sector
                continue                                                                                 # v9 sector
            if d_new == seg["d"][j] and d_new * 2.0 < 1.0:                                               # v9 sector
                nchol += 1                                                                               # v9 sector
                if pd_banded_v36(ab, 2.0 * d_new):                                                       # v9 sector
                    d_new = 2.0 * d_new                                                                  # v9 sector
            new_d[j] = d_new                                                                             # v9 sector
        for j in newly:                                                                                  # v9 sector
            crossing[bl[j]] = {"s_lo": s_cur, "s_fail": s1, "probes": 0}                                 # v9 sector
        if newly and cross_step is None:                                                                 # v9 sector: the first probe of the crossing phase at the reflected point s1 + (s1 - s_lo) (4.1(ii)'s refined piece), then doubling
            cross_step = s1 - s_cur                                                                      # v9 sector
        at_end = s1 >= 1.0; ready = []                                                                   # v9 sector
        if crossing:                                                                                     # v9 sector: the reopening test for the anchors in their crossing phase
            cand = [j for j in range(m) if bl[j] in crossing and j not in newly]                         # v9 sector
            if at_end:                                                                                   # v9 sector
                ready = sorted(cand + newly)                                                             # v9 sector
            else:                                                                                        # v9 sector
                for j in cand:                                                                           # v9 sector
                    Ks = (seg["K0"][j] + t1 * seg["dK"][j]).tocsr()                                      # v9 sector
                    ab, _u = band_upper_v36(Ks @ Ks); nchol += 1                                         # v9 sector
                    if pd_banded_v36(ab, max(2.0 * (seg["d_eff"][j] + seg["slack"][j]), 4.0 * float(tol_block))):   # v9 sector: the nearest level at least twice the floor from the anchor: the pair may have reopened
                        ready.append(j)                                                                  # v9 sector
        if ready:                                                                                        # v9 sector
            eig_hi = path.eig(s1); e_hi, sig_hi, U_hi = eig_hi                                           # v9 sector: the eigendecomposition at the probe (the trial's own at s = 1)
            order_hi, ep_hi = light_plus_order_v9(e_hi, sig_hi)                                          # v9 sector
            if len(ep_hi) != len(ep_seg):                                                                # v9 sector
                return done("the S = +1 sector count changes along the path (%d at s %.6f against %d)" % (len(ep_hi), s1, len(ep_seg)))   # v9 sector
            opened = [j for j in ready if (ep_hi[bl[j] + 1] - ep_hi[bl[j]]) >= max(2.0 * (seg["d_eff"][j] + seg["slack"][j]), 4.0 * float(tol_block))]   # v9 sector: the pair's gap at the probe, exactly: reopened when its re-centred anchor would clear the floor
            ready = ready if at_end else opened                                                          # v9 sector: at the path's end every crossing anchor is resolved, a pair still within the floor flagged
            if ready:                                                                                    # v9 sector
                groups = {}                                                                              # v9 sector
                for j in ready:                                                                          # v9 sector
                    groups.setdefault(crossing[bl[j]]["s_lo"], []).append(j)                             # v9 sector
                for s_lo, js in sorted(groups.items()):                                                  # v9 sector
                    eig_lo = path.eig(s_lo)                                                              # v9 sector
                    occp, crs, _hi = _resolve_crossings_v9(path, True, seg["bnds"], js, occp, s_lo, s1, [crossing[bl[j]]["s_fail"] for j in js], [seg["d_eff"][j] for j in js], tol_block,   # v9 sector
                                                           eig_lo=eig_lo, eig_hi=eig_hi, probes_js=[crossing[bl[j]]["probes"] for j in js], flag_js=[(None if j in opened else "the pair within the floor at the path's end") for j in js])   # v9 sector
                    crossings += crs; cross_len += (s1 - s_lo)                                           # v9 sector
                    for j in js:                                                                         # v9 sector
                        del crossing[bl[j]]                                                              # v9 sector
                        if s1 >= 1.0:                                                                    # v9 sector (review)
                            end_resolved.add(bl[j])                                                      # v9 sector
                if len(crossings) > max_crossings:                                                       # v9 sector
                    return done("more than %d crossings in one record" % max_crossings)                  # v9 sector
                s_seg = s1; e_seg = np.asarray(e_hi, float); sig_seg = np.asarray(sig_hi, float); ep_seg = ep_hi; h_seg = h1 if s1 >= 1.0 else path.h_host(s1); seg_last = seg; seg = None   # v9 sector: re-anchored from the spectrum at the probe (busi fix: seg None after a crossing resolved at s = 1)
                if not crossing:                                                                         # v9 sector
                    cross_step = None                                                                    # v9 sector
                if s1 >= 1.0:                                                                            # v9 sector
                    break                                                                                # v9 sector
                continue                                                                                 # v9 sector
        if s1 >= 1.0:                                                                                    # v9 sector
            break                                                                                        # v9 sector
        s_cur = s1                                                                                       # v9 sector
        for j in others:                                                                                 # v9 sector
            if j not in newly:                                                                           # v9 sector
                seg["d"][j] = new_d[j]; seg["from_eig"][j] = False                                       # v9 sector
    bnds_end = boundaries_v9(occp); anchors_end = midgaps_v9(ep1, bnds_end)                              # v9 sector
    if len(bnds_end) % 2 == 0:                                                                           # v9 sector (review): the mirror rule re-checked after a block resolved at s = 1 (finding 7)
        return done("an even number of boundaries (%d) in the S = +1 sector at the path's end, outside the mirror rule" % len(bnds_end))   # v9 sector
    seg_f = seg if seg is not None else seg_last; g_end_floor = 2.0 * (max(seg_f["d_eff"]) + max(seg_f["slack"]))   # v9 sector (review): the floor as a pair gap, the crossing phase's reopening criterion (busi fix: seg None after a crossing resolved at s = 1)
    for b_ in [int(b_) for b_ in bnds_end if int(b_) not in end_resolved]:                               # v9 sector (review): a boundary pair within the floor at s = 1 that no crossing phase resolved (the Weyl step reached 1, or no field change) was continued by sorted index at a near-degenerate point: flagged, no swap
        if (ep1[b_ + 1] - ep1[b_]) < g_end_floor:                                                        # v9 sector
            crossings.append({"s": 1.0, "piece": [1.0, 1.0], "s_reopen": 1.0, "probes": 0, "anchor_energy": float(0.5 * (ep1[b_] + ep1[b_ + 1])), "pair": [float(ep1[b_]), float(ep1[b_ + 1])], "pair_start": None, "pair_gap_end": float(ep1[b_ + 1] - ep1[b_]), "overlap": None, "inversion_delta": 0, "swap": False, "block": 2, "flagged": True, "flagged_why": "the occupation cut within the floor at the path's end without a crossing phase (the Weyl step reached s = 1, or no field change): the pair continued by sorted index at a near-degenerate point", "intermediate": 0, "occupied_before": None, "occupied_after": None})   # v9 sector
    cert.update({"certified": True, "pieces": len(pieces), "cholesky_calls": nchol, "d_min_certified": dmin_all, "smallest_piece": (min(p[1] - p[0] for p in pieces) if pieces else None),   # v9 sector
                 "piece_list": ([[float(p[0]), float(p[1]), float(p[2])] for p in pieces] if pieces else None), "seconds": time.time() - t0, "why": None})   # v9 sector
    sector.update({"pieces": len(pieces), "cholesky": nchol, "eigh": int(getattr(path, "eigh_n", 0)), "segments": segments, "probes_in_crossing": n_probe_cross, "anchors_end": int(len(bnds_end)), "anchors": [float(x) for x in anchors_end],   # v9 sector
                   "certified_fraction": float(min(1.0, max(0.0, 1.0 - cross_len))), "crossings": crossings, "swaps": int(sum(1 for c_ in crossings if c_.get("swap"))), "flagged": int(sum(1 for c_ in crossings if c_.get("flagged"))), "inversions_plus_end": inversions_run_v9(occp),   # v9 sector
                   "route": ("mu S walk" if (sector["anchors_start"] == 1 and len(bnds_end) == 1 and not crossings) else "sector walk")})   # v9 sector
    return cert, sector, anchors_end, occp                                                               # v9 sector


def light_continue_sector_v9(a, ch, window, f_cur, e, U, n_sea, sgn_dev, S_host, e_np, sig, imp):       # v9 sector
    """v9 sector: light_continue_v36's banded route in sector mode (4.1): the anchors from the window's last entry (its mask, its spectrum "e" and sectors "sigma"), the sector walk, the anchor mask at the trial fields, the count check, the cut check (recorded); the record-scale convention (version 8's fallback, OPS3.continue_sea_v3) only when the walk cannot be certified, flagged. Returns (ref, info, mask) for light_continue_v36 to finish as it finishes the banded route."""   # v9 sector
    torch = BK["torch"]; t0 = time.time()                                                                # v9 sector
    prev = window[-1]                                                                                    # v9 sector
    e0 = prev.get("e"); sig0 = prev.get("sigma"); mask0 = prev.get("mask")                               # v9 sector
    if e0 is None or sig0 is None:                                                                       # v9 sector: a window entry without the stored spectrum (version 9 writes it in every entry; a foreign entry): recomputed at the previous accepted fields
        e0t, U0 = eigT(clocked_gpu(ch, prev["f"][0], prev["f"][1])); sg0, _n0, _i0 = light_sectorize_v36(e0t, U0, sgn_dev, a.light_tol_block)   # v9 sector
        e0 = e0t.cpu().numpy().astype(float); sig0 = sector_e_v9(sg0); del e0t, U0                        # v9 sector
    e0 = sector_e_v9(e0); sig0 = sector_e_v9(sig0)                                                       # v9 sector: on the host as float arrays whatever the entry holds (an array or a tensor on any device)
    if mask0 is None and prev.get("mu") is not None:                                                     # v9 sector
        mask0 = light_select_v36(e0, sig0, prev["mu"])[0]                                                # v9 sector
    why = None; cert = None; sector = None; occ = None; gap_end = None; cut = None; anchors_end = None; occp_end = None   # v9 sector
    if imp > a.light_sigma_tol:                                                                          # v9 sector
        why = "sector impurity %.2e" % imp                                                               # v9 sector
    elif mask0 is None:                                                                                  # v9 sector
        why = "no stored occupation of the previous reference"                                           # v9 sector
    else:                                                                                                # v9 sector
        path = LightPathV9(ch, prev["f"], f_cur, sgn_dev, a.light_tol_block, (e_np, sig, U))             # v9 sector
        cert, sector, anchors_end, occp_end = light_sector_walk_v9(path, S_host, e0, sig0, mask0, (e_np, sig, U), n_sea, a.light_d_min, a.light_max_pieces, a.light_tol_block, mu_prev=prev.get("mu"))   # v9 sector
        sector["eigh"] = int(path.eigh_n)                                                                # v9 sector
        if not cert["certified"]:                                                                        # v9 sector
            why = cert["why"]                                                                            # v9 sector
    if why is None:                                                                                      # v9 sector
        occ, gap_end = light_select_sector_v9(e_np, sig, anchors_end)                                    # v9 sector
        if int(occ.sum()) != n_sea:                                                                      # v9 sector
            why = "the anchor mask counts %d against n_sea %d" % (int(occ.sum()), n_sea)                 # v9 sector
    if why is None and cert["norm_dK_G"] > 0.0:                                                          # v9 sector: the cut check as now, recorded (its verdict does not take the route: 4.4(d))
        cut = light_cut_check_sector_v9(prev["ref"], e_np, sig, occ, U, anchors_end, cert["norm_dK_G"], gap_end)   # v9 sector
    if sector is not None:                                                                               # v9 sector
        TML["anchors"] = TML.get("anchors", 0) + int(sector.get("anchors_end") or 0); TML["crossings"] = TML.get("crossings", 0) + len(sector.get("crossings") or [])   # v9 sector
    info = {}                                                                                            # v9 sector
    if why is None:                                                                                      # v9 sector
        ref = U[:, torch.as_tensor(occ, device=U.device)]; mask = occ                                    # v9 sector
        _order1, ep1 = light_plus_order_v9(e_np, sig); b0 = int(boundaries_v9(occp_end)[0])             # v9 sector
        cr = sector["crossings"]; ov = [c_["overlap"] for c_ in cr if c_.get("overlap") is not None]; nfl = int(sum(1 for c_ in cr if c_.get("flagged")))   # v9 sector
        sector.update({"adiabatic_at_cut": (int(cut["crossing_levels"]) if cut is not None else 0), "inversions_sector": light_sector_inversions_v36(e_np, sig, mask), "n_plus": int(occp_end.sum()), "gap": gap_end})   # v9 sector
        mode = (("mu S selector, certified walk (%d pieces)" % cert["pieces"]) if sector["route"] == "mu S walk" else   # v9 sector
                ("sector walk: %d anchors, %d pieces, %d crossings (min overlap %s, max block %s, flagged %d)" % (sector["anchors_end"], cert["pieces"], len(cr), fmt(min(ov) if ov else None, "%.4f"), (max(c_["block"] for c_ in cr) if cr else "-"), nfl)))   # v9 sector
        info.update({"mode": mode, "verified": True, "isolation_verified": True, "certified": cert, "nsub": cert["pieces"], "fallback": False, "why": None, "mu": anchors_end[0], "gap": gap_end, "weyl_gap": gap_end,   # v9 sector
                     "gap_plus_half": 0.5 * (ep1[b0 + 1] - ep1[b0]), "n_plus": int(occp_end.sum()), "margin_min": cert["d_min_certified"], "norm_dH_G": cert["norm_dK_G"], "cut_check": cut,   # v9 sector
                     "cut_crossing": bool(cut is not None and cut["crossing_levels"] > 0), "sector": sector})   # v9 sector
    else:                                                                                                # v9 sector: the record-scale convention, the fallback of last resort, flagged (4.1(iii))
        ref, inff, mask = OPS3.continue_sea_v3(ch, window, f_cur, e, U, n_sea, a.tau, a.c_LZ, eig_at, clocked_host_sparse)   # v9 sector
        info.update(inff)                                                                                # v9 sector
        if sector is None:                                                                               # v9 sector
            sector = {"route": "record-scale fallback", "pieces": 0, "cholesky": 0, "eigh": 0, "segments": 0, "probes_in_crossing": 0, "anchors_start": None, "anchors_end": None, "anchors": [], "certified_fraction": 0.0, "crossings": [], "flagged": 0, "adiabatic_at_cut": 0}   # v9 sector
        sector.update({"route": "record-scale fallback", "fallback_why": why, "inversions_sector": light_sector_inversions_v36(e_np, sig, np.asarray(mask, bool))})   # v9 sector
        info.update({"mode": "RECORD-SCALE FALLBACK: the LZ convention, version 3's maximum-overlap continuation (%s) (%s)" % (inff.get("mode"), why), "verified": False, "isolation_verified": False,   # v9 sector
                     "certified": cert, "nsub": int(inff.get("nsub") or 1), "fallback": True, "why": why, "margin_min": None, "cut_check": cut, "cut_crossing": False, "sector": sector})   # v9 sector
    info["seconds_sector"] = time.time() - t0                                                            # v9 sector
    return ref, info, mask                                                                               # v9 sector


def heavy_sector_walk_v9(path, prev_ref, e0_np, mask0, trial, n_sea, d_min, max_pieces, tol_block, step_scale=1.0, max_crossings=128):   # v9 sector
    """v9 sector, 4.2 (v9b): the heavy's sector walk: the Weyl bound (each eigenvalue moves at most ||dh||_G per unit s) certifies the piece from s to s + g/(2 ||dh||_G) with g the smallest boundary gap, less the eigenvalue slack 8 eps ||h||_G, of the eigenvalues at s over the boundaries in the step rule (the occupation carried by sorted index: no occupied level meets an empty one inside the piece), the eigenvalues at the piece's end on the host (engine_ops_v34_cert.eigvals_host) the next probe's; a boundary gap below the floor (the same floor formula as the light's, max(d_min, 2 sqrt(64 eps) ||h||_G)) at a probe point opens that boundary's crossing phase (s_lo the probe point certified before it; alone, the probe advances by the floor's Weyl length, doubling), and at the first probe at which its gap less the slack is at least the floor again (and clear of the convention's block tolerance) the crossing is resolved as the light's (_resolve_crossings_v9 with the heavy's eigendecompositions at the piece's ends); a pair still within the floor at s = 1 is resolved there, flagged. Returns (walk, occ_end); walk["certified"] False with its why on a failure (the caller keeps version 8's record-scale result, flagged)."""   # v9 sector
    t0 = time.time(); eps = float(np.finfo(float).eps)                                                 # v9 sector
    e1_np, U1 = trial; e1_np = np.asarray(e1_np, float)                                                  # v9 sector
    occ = np.asarray(mask0, bool).copy(); e_s = np.asarray(e0_np, float)                                 # v9 sector
    pieces = []; crossings = []; cross_len = 0.0; margin = None; n_probe_cross = 0                      # v9 sector
    walk = {"route": "sector walk", "pieces": 0, "eigvals": 0, "eigh": 0, "probes_in_crossing": 0, "anchors_start": int(len(boundaries_v9(occ))), "anchors_end": 0, "crossings": [], "flagged": 0, "certified_fraction": 0.0, "norm_dH_G": 0.0, "norm_H_G": 0.0,   # v9 sector
            "d_min_effective": 0.0, "gap_start": None, "margin_min": None, "smallest_piece": None, "certified": False, "why": None, "seconds": 0.0, "inversions_start": inversions_run_v9(occ), "inversions_end": None, "fallback_why": None}   # v9 sector
    def done(why):                                                                                       # v9 sector
        walk.update({"why": why, "fallback_why": why, "route": "record-scale fallback", "seconds": time.time() - t0, "crossings": crossings, "pieces": len(pieces), "probes_in_crossing": n_probe_cross, "eigvals": int(getattr(path, "eigvals_n", 0)), "eigh": int(getattr(path, "eigh_n", 0))})   # v9 sector
        return walk, None                                                                                # v9 sector
    if len(e_s) != len(occ) or len(e1_np) != len(occ):                                                   # v9 sector
        return done("the level counts differ between the window's spectrum, its mask and the trial spectrum")   # v9 sector
    if walk["anchors_start"] == 0:                                                                       # v9 sector
        return done("no boundary of the occupation")                                                     # v9 sector
    h0 = path.h_host(0.0); h1 = path.h_host(1.0); dH = (h1 - h0).tocsr(); nG = OPS3.gershgorin(dH)       # v9 sector
    GH = max(OPS3.gershgorin(h0), OPS3.gershgorin(h1)); F = 64.0 * eps * GH * GH; d_eff = max(float(d_min), 2.0 * math.sqrt(F)); slack = 8.0 * eps * GH   # v9 sector
    b0 = boundaries_v9(occ)                                                                              # v9 sector
    walk.update({"norm_dH_G": nG, "norm_H_G": GH, "d_min_effective": d_eff, "gap_start": float(np.min(e_s[b0 + 1] - e_s[b0]))})   # v9 sector
    g_need = max(d_eff + slack, 4.0 * float(tol_block))                                                  # v9 sector: a boundary pair reopened: its gap less the slack at least the floor (the step rule's own criterion) and clear of the convention's degenerate-block tolerance
    s = 0.0; s_prev = None; crossing = {}; cross_step = None; closed = set(); s_occ = 0.0                # v9 sector
    if nG > 0.0:                                                                                         # v9 sector
        while True:                                                                                      # v9 sector
            bnds = boundaries_v9(occ)                                                                    # v9 sector
            if len(bnds) == 0:                                                                           # v9 sector
                return done("no boundary of the occupation after a crossing")                            # v9 sector
            bl = [int(b) for b in bnds]; crossing = {b: q for b, q in crossing.items() if b in bl}       # v9 sector
            gaps = e_s[bnds + 1] - e_s[bnds]                                                             # v9 sector
            for j in range(len(bl)):                                                                     # v9 sector: a boundary gap below the floor at this probe point: its crossing phase opens, s_lo the probe point certified before it (the start, if there is none)
                if bl[j] not in crossing and bl[j] not in closed and not (gaps[j] - slack >= d_eff):     # v9 sector
                    crossing[bl[j]] = {"s_lo": (s if s_prev is None else max(s_prev, s_occ)), "s_fail": s, "probes": 0}   # v9 sector
                    if cross_step is None:                                                               # v9 sector
                        cross_step = (d_eff / nG) if (s_prev is None or s - max(s_prev, s_occ) <= 0.0) else (s - max(s_prev, s_occ))   # v9 sector
            at_end = s >= 1.0                                                                            # v9 sector
            opened = [j for j in range(len(bl)) if bl[j] in crossing and gaps[j] >= g_need]              # v9 sector
            ready = [j for j in range(len(bl)) if bl[j] in crossing] if at_end else opened               # v9 sector: at the path's end every crossing boundary is resolved, a pair still within the floor flagged
            if ready:                                                                                    # v9 sector
                eig_hi = path.eig(s)                                                                     # v9 sector: the eigendecomposition at the probe (the trial's own at s = 1)
                groups = {}                                                                              # v9 sector
                for j in ready:                                                                          # v9 sector
                    groups.setdefault(crossing[bl[j]]["s_lo"], []).append(j)                             # v9 sector
                for s_lo, js in sorted(groups.items()):                                                  # v9 sector
                    eig_lo = path.eig(s_lo)                                                              # v9 sector
                    occ, crs, _hi = _resolve_crossings_v9(path, False, bnds, js, occ, s_lo, s, [crossing[bl[j]]["s_fail"] for j in js], [d_eff] * len(js), tol_block,   # v9 sector
                                                          eig_lo=eig_lo, eig_hi=eig_hi, probes_js=[crossing[bl[j]]["probes"] for j in js], flag_js=[(None if j in opened else "the pair within the floor at the path's end") for j in js])   # v9 sector
                    crossings += crs; cross_len += (s - s_lo)                                            # v9 sector
                    for j in js:                                                                         # v9 sector
                        del crossing[bl[j]]                                                              # v9 sector
                        if at_end:                                                                       # v9 sector
                            closed.add(bl[j])                                                            # v9 sector
                if len(crossings) > max_crossings:                                                       # v9 sector
                    return done("more than %d crossings in one record" % max_crossings)                  # v9 sector
                if not crossing:                                                                         # v9 sector
                    cross_step = None                                                                    # v9 sector
                s_occ = s                                                                                # v9 sector (review)
                continue                                                                                 # v9 sector: the boundaries at this probe recomputed with the occupation updated
            if at_end:                                                                                   # v9 sector
                break                                                                                    # v9 sector
            others = [j for j in range(len(bl)) if bl[j] not in crossing]                                # v9 sector
            step = None; g_min = None                                                                    # v9 sector
            if others:                                                                                   # v9 sector
                g_min = float(min(gaps[j] - slack for j in others)); step = g_min / (2.0 * nG)           # v9 sector: the Weyl half step: no occupied level meets an empty one inside the piece
                if step_scale != 1.0:                                                                    # v9 sector
                    step = step * step_scale                                                             # v9 sector
            if crossing:                                                                                 # v9 sector
                if step is None or cross_step <= step:                                                  # v9 sector (review): the doubling only when the crossing probe governs the step; under the others' Weyl steps cross_step holds
                    step = cross_step; cross_step = 2.0 * cross_step                                     # v9 sector
            s1 = min(1.0, s + step)                                                                      # v9 sector
            if others:                                                                                   # v9 sector
                pieces.append((s, s1, g_min)); mg = g_min - 2.0 * nG * (s1 - s); margin = mg if margin is None else min(margin, mg)   # v9 sector
            if crossing:                                                                                 # v9 sector
                n_probe_cross += 1                                                                       # v9 sector
                for q in crossing.values():                                                              # v9 sector
                    q["probes"] += 1                                                                     # v9 sector
            if len(pieces) + n_probe_cross >= max_pieces:                                                # v9 sector
                return done("more than %d pieces or probes (||dh||_G %.3e, the boundary gap %s)" % (max_pieces, nG, fmt(g_min, "%.3e")))   # v9 sector
            s_prev = s; s = s1; e_s = e1_np if s1 >= 1.0 else path.eigvals(s1)                           # v9 sector: the eigenvalues at the next probe point (the trial's own at s = 1)
    bnds_end = boundaries_v9(occ)                                                                        # v9 sector
    if nG == 0.0:                                                                                        # v9 sector (review): finding 2 on the heavy's no-field-change path
        for b_ in [int(b_) for b_ in bnds_end]:                                                          # v9 sector (review): a boundary pair within the floor at s = 1 with no field change was continued by sorted index at a near-degenerate point: flagged, no swap
            if (e1_np[b_ + 1] - e1_np[b_]) - slack < d_eff:                                              # v9 sector
                crossings.append({"s": 1.0, "piece": [1.0, 1.0], "s_reopen": 1.0, "probes": 0, "anchor_energy": float(0.5 * (e1_np[b_] + e1_np[b_ + 1])), "pair": [float(e1_np[b_]), float(e1_np[b_ + 1])], "pair_start": None, "pair_gap_end": float(e1_np[b_ + 1] - e1_np[b_]), "overlap": None, "inversion_delta": 0, "swap": False, "block": 2, "flagged": True, "flagged_why": "the occupation cut within the floor at the path's end without a crossing phase (no field change): the pair continued by sorted index at a near-degenerate point", "intermediate": 0, "occupied_before": None, "occupied_after": None})   # v9 sector
    walk.update({"certified": True, "pieces": len(pieces), "eigvals": int(getattr(path, "eigvals_n", 0)), "eigh": int(getattr(path, "eigh_n", 0)), "probes_in_crossing": n_probe_cross, "anchors_end": int(len(bnds_end)), "crossings": crossings,   # v9 sector
                 "flagged": int(sum(1 for c_ in crossings if c_.get("flagged"))), "swaps": int(sum(1 for c_ in crossings if c_.get("swap"))), "certified_fraction": float(min(1.0, max(0.0, 1.0 - cross_len))), "margin_min": margin, "smallest_piece": (min(p[1] - p[0] for p in pieces) if pieces else None), "inversions_end": inversions_run_v9(occ), "seconds": time.time() - t0})   # v9 sector
    return walk, occ                                                                                     # v9 sector


def continue_ref_v9(a, ch, window, f_cur, e, U, n_sea):                                                 # v9 sector
    """v9 sector: the heavy's continuation: continue_ref_v34 as version 8 (returned untouched in record mode, and kept bitwise in sector mode when its Weyl test passes: version 3's weyl_gap is already the smallest gap over every boundary); where the record-scale test fails in sector mode, heavy_sector_walk_v9 (4.2) and version 3's assignment with the walk's occupation given; the record-scale result kept and flagged when the walk cannot be certified."""   # v9 sector
    ref, cinfo, mask = continue_ref_v34(a, ch, window, f_cur, e, U, n_sea)                               # v9 sector: version 8's call
    if not sector_v9(a):                                                                                 # v9 sector
        return ref, cinfo, mask                                                                          # v9 sector
    prev = window[-1]                                                                                    # v9 sector
    mask0 = prev.get("mask")                                                                             # v9 sector
    if mask0 is None:                                                                                    # v9 sector: the T = 0 window of a resumed base-module run: the n_sea lowest levels occupied (engine_ops_v34_cert._prev_mask's rule)
        mask0 = np.zeros(int(e.shape[0]), bool); mask0[:n_sea] = True                                    # v9 sector
    n_b0 = int(len(boundaries_v9(mask0)))                                                                # v9 sector
    if cinfo.get("verified"):                                                                            # v9 sector
        cinfo["sector"] = {"route": "Weyl", "pieces": int(cinfo.get("nsub") or 1), "eigvals": 0, "eigh": 0, "probes_in_crossing": 0, "anchors_start": n_b0, "anchors_end": int(len(boundaries_v9(mask))), "crossings": [], "flagged": 0, "certified_fraction": 1.0,   # v9 sector
                           "inversions_start": inversions_run_v9(mask0), "inversions_end": int(cinfo.get("inversions") or 0), "n_intermediate": int(cinfo.get("n_intermediate") or 0), "min_occupied_weight": cinfo.get("min_occupied_weight"), "fallback_why": None}   # v9 sector
        TM["anchors"] = TM.get("anchors", 0) + cinfo["sector"]["anchors_end"]                            # v9 sector
        return ref, cinfo, mask                                                                          # v9 sector
    e0 = prev.get("e")                                                                                   # v9 sector
    if e0 is None:                                                                                       # v9 sector: a window entry without the stored spectrum (the base module's resume_slice_v34 entry): the eigenvalues at the previous accepted fields on the host
        e0 = CERT.eigvals_host(clocked_host_sparse(ch, prev["f"][0], prev["f"][1]))                      # v9 sector
    ev = sector_e_v9(e)                                                                                  # v9 sector
    path = HeavyPathV9(ch, prev["f"], f_cur, (ev, U))                                                    # v9 sector
    walk, occ_end = heavy_sector_walk_v9(path, prev["ref"], sector_e_v9(e0), mask0, (ev, U), n_sea, a.light_d_min, a.light_max_pieces, 1e-8)   # v9 sector
    TM["anchors"] = TM.get("anchors", 0) + int(walk.get("anchors_end") or 0); TM["crossings"] = TM.get("crossings", 0) + len(walk.get("crossings") or [])   # v9 sector
    if not walk["certified"]:                                                                            # v9 sector: the record-scale convention (version 8's result above), the fallback of last resort, flagged; its "verified" False keeps version 8's LZ-convention label
        cinfo["sector"] = dict(walk); cinfo["sector"].update({"route": "record-scale fallback", "n_intermediate": int(cinfo.get("n_intermediate") or 0), "min_occupied_weight": cinfo.get("min_occupied_weight"), "inversions_end": int(cinfo.get("inversions") or 0)})   # v9 sector
        cinfo["mode"] = "RECORD-SCALE FALLBACK (%s): %s" % (walk["why"], cinfo.get("mode"))               # v9 sector
        return ref, cinfo, mask                                                                          # v9 sector
    ref2, info2, mask2 = CERT.assign_occupation_v34(prev["ref"], e, U, n_sea, occ_end)                   # v9 sector: version 3's assignment with the walk's occupation given (the weights against the previous reference keep min_occupied_weight's meaning)
    s_T = walk["norm_dH_G"] / a.tau; tau_LZ = (1.0 / (a.c_LZ * math.sqrt(s_T))) if s_T > 0.0 else math.inf   # v9 sector
    cr = walk["crossings"]; ov = [c_["overlap"] for c_ in cr if c_.get("overlap") is not None]           # v9 sector
    info2.update({"norm_dH_G": walk["norm_dH_G"], "s_T": s_T, "tau_LZ": tau_LZ, "c_LZ": a.c_LZ, "m": 1, "m_eff": 1, "window_consistency": None,   # v9 sector
                  "mode": "sector walk (the Weyl bound at every boundary: %d pieces, %d crossings, min overlap %s, max block %s, flagged %d)" % (walk["pieces"], len(cr), fmt(min(ov) if ov else None, "%.4f"), (max(c_["block"] for c_ in cr) if cr else "-"), int(walk.get("flagged") or 0)),   # v9 sector
                  "nsub": max(1, int(walk["pieces"])), "verified": True, "isolation_verified": True, "margin_min": walk["margin_min"]})   # v9 sector
    walk["n_intermediate"] = int(info2.get("n_intermediate") or 0); walk["min_occupied_weight"] = info2.get("min_occupied_weight"); walk["inversions_end"] = int(info2.get("inversions") or 0)   # v9 sector
    info2["sector"] = walk                                                                               # v9 sector
    return ref2, info2, mask2                                                                            # v9 sector


def light_resume_anchor_v9(a, e_np, sig, mask, mu, n_plus):                                              # v9 sector
    """v9 sector: the resumed light window's (gap, mu, n_plus): record mode light_select_v36's gap at the slice's mu as version 8; sector mode with several boundaries the anchors rebuilt from the mask and the spectrum at the slice's fields (4.3): the gap the minimum over anchors, mu the first anchor, n_plus the + sector's occupied count (one boundary: the slice's mu and n_plus as version 8, bitwise on the same device)."""   # v9 sector
    if not sector_v9(a):                                                                                 # v9 sector
        return (light_select_v36(e_np, sig, mu)[1] if mu is not None else None), mu, n_plus              # v9 sector
    order, ep = light_plus_order_v9(e_np, sig); occp = np.asarray(mask, bool)[order]; bnds = boundaries_v9(occp)   # v9 sector
    if len(bnds) <= 1 and mu is not None:                                                                # v9 sector
        return light_select_v36(e_np, sig, mu)[1], mu, n_plus                                            # v9 sector
    if len(bnds) == 0:                                                                                   # v9 sector
        return None, mu, n_plus                                                                          # v9 sector
    anchors = midgaps_v9(ep, bnds); gap = light_select_sector_v9(e_np, sig, anchors)[1]                  # v9 sector
    return gap, anchors[0], int(occp.sum())                                                              # v9 sector


def light_channel_segment_v9(a, k_, q):                                                                  # v9 sector
    """v9 sector: the light line's per-channel text: version 8's in record mode (bitwise); in sector mode "certified (mu S walk)" where one pocket sufficed, "sector walk: pieces P (Cholesky C) anchors A crossings K (min overlap w, max block m, flagged f)" otherwise, "RECORD-SCALE FALLBACK (why)" on a fallback (4.3)."""   # v9 sector
    sec = q.get("sector") if sector_v9(a) else None                                                      # v9 sector
    if sec is None:                                                                                      # v9 sector
        route = ("certified" if q.get("verified") else "FALLBACK (%s)" % q.get("why"))                   # v9 sector
    elif sec.get("route") == "record-scale fallback":                                                    # v9 sector
        route = "RECORD-SCALE FALLBACK (%s)" % q.get("why")                                              # v9 sector
    elif sec.get("route") == "sector walk":                                                              # v9 sector
        cr = sec.get("crossings") or []; ov = [c_["overlap"] for c_ in cr if c_.get("overlap") is not None]   # v9 sector
        route = "sector walk: pieces %s (Cholesky %s) anchors %s crossings %d (swaps %d, min overlap %s, max block %s, flagged %d)%s" % (sec.get("pieces"), sec.get("cholesky"), sec.get("anchors_end"), len(cr), int(sum(1 for c_ in cr if c_.get("swap"))), fmt(min(ov) if ov else None, "%.4f"), (max(c_["block"] for c_ in cr) if cr else "-"), int(sec.get("flagged") or 0), ((" adiabatic at an anchor %d" % sec["adiabatic_at_cut"]) if sec.get("adiabatic_at_cut") else ""))   # v9 sector
    else:                                                                                                # v9 sector
        route = "certified (mu S walk)%s" % ((" adiabatic at the cut %d" % sec["adiabatic_at_cut"]) if sec.get("adiabatic_at_cut") else "")   # v9 sector
    return "kappa %s: %s mu %s gap %s pieces %s d_min %s pockets %d inv %d wmin %s E_inv %.3e" % (                  # v9 sector
        k_, route, fmt(q.get("mu"), "%.3e"), fmt(q.get("gap"), "%.2e"), q.get("nsub"),                   # v9 sector
        fmt((q.get("certified") or {}).get("d_min_certified") if isinstance(q.get("certified"), dict) else None, "%.2e"), q.get("n_pocket") or 0, q.get("inversions_sector") or 0,   # v9 sector
        fmt(q.get("min_occupied_weight"), "%.4f"), q.get("E_inv"))                                       # v9 sector


def sector_record_segment_v9(cont, contL):                                                               # v9 sector
    """v9 sector: the record line's sector segment (sector mode): per species the channels on the sector walk, their anchors, crossings, the smallest overlap, the largest block, the flagged crossings, the record-scale fallbacks, the light's adiabatic passages at an anchor."""   # v9 sector
    def part(entries, light):                                                                            # v9 sector
        n_sw = 0; n_cr = 0; ov = None; blk = 0; n_fb = 0; adia = 0; anc = 0; nfl = 0                     # v9 sector
        for q in entries.values():                                                                       # v9 sector
            sec = q.get("sector") or {}; r_ = sec.get("route")                                           # v9 sector
            if r_ == "record-scale fallback":                                                            # v9 sector
                n_fb += 1                                                                                # v9 sector
            elif r_ == "sector walk":                                                                    # v9 sector
                n_sw += 1                                                                                # v9 sector
            anc = max(anc, int(sec.get("anchors_end") or 0))                                             # v9 sector
            for c_ in (sec.get("crossings") or []):                                                      # v9 sector
                n_cr += 1; blk = max(blk, int(c_.get("block") or 0)); nfl += (1 if c_.get("flagged") else 0)   # v9 sector
                if c_.get("overlap") is not None:                                                        # v9 sector
                    ov = c_["overlap"] if ov is None else min(ov, c_["overlap"])                         # v9 sector
            if light:                                                                                    # v9 sector
                adia += int(sec.get("adiabatic_at_cut") or 0)                                            # v9 sector
        s_ = "%d on the sector walk, anchors max %d, crossings %d (min overlap %s, max block %d, flagged %d), record-scale fallbacks %d" % (n_sw, anc, n_cr, fmt(ov, "%.4f"), blk, nfl, n_fb)   # v9 sector
        return s_ + ((", adiabatic at an anchor %d" % adia) if light else "")                            # v9 sector
    return "; v9 sector [heavy: %s; light: %s]" % (part(cont, False), part(contL, True))                 # v9 sector
'''
R.append(("v9 block", "the module: the sector walk's functions before the v8 block (after light_continue_v36, whose helpers they call)",
          "\n\n# -------------------------------------------------- v8 lagged (the ruling's v4): the transported-reference scheme (busi/notes/C64_fix_ruling_v4.md I.2)\n",
          "\n\n" + BLOCK9.strip("\n") + "\n\n\n# -------------------------------------------------- v8 lagged (the ruling's v4): the transported-reference scheme (busi/notes/C64_fix_ruling_v4.md I.2)\n"))

# 3. light_continue_v36(): the sector dispatch
R.append(("light_continue_v36 dispatch", "light_continue_v36(): the sector branch before `if route == \"v3\":` (that line the `elif`)", _s(r'''
    if route == "v3":
        ref, inf3, mask = OPS3.continue_sea_v3(ch, window, f_cur, e, U, n_sea, a.tau, a.c_LZ, eig_at, clocked_host_sparse)'''), _s(r'''
    if route == "banded" and sector_v9(a):                                                               # v9 sector: 4.1, the sector walk in place of the banded block below and of its record-scale fallback
        ref, infs, mask = light_continue_sector_v9(a, ch, window, f_cur, e, U, n_sea, sgn_dev, S_host, e_np, sig, imp)   # v9 sector
        info.update(infs)                                                                                # v9 sector
    elif route == "v3":                                                                                  # v9 sector: version 8's `if` as the `elif` (the same statements on the record path)
        ref, inf3, mask = OPS3.continue_sea_v3(ch, window, f_cur, e, U, n_sea, a.tau, a.c_LZ, eig_at, clocked_host_sparse)''')))

# 4. the window entries: light_init_v36, light_resume_v36 (and its check and gap), light_follow_v36
R.append(("light_init window", "light_init_v36(): the T = 0 window entry carries the spectrum and the sectors", _s(r'''
                                                    "mask": occ, "mu": mu0, "n_plus": n_plus0}]}'''), _s(r'''
                                                    "mask": occ, "mu": mu0, "n_plus": n_plus0, "e": e_np, "sigma": sig}]}   # v9 sector: the spectrum and sectors at the accepted fields (the sector walk's anchors; version 8 reads no new key)''')))
R.append(("light_resume check", "light_resume_v36(): the resume check gains --cut-transport (a run without it is the record transport)", _s(r'''
    Z = np.load(a.resume_slice)
    Zf = np.load(os.path.join(rdir, "collapse_%s_fields.npz" % rtag), allow_pickle=True)'''), _s(r'''
    if str(J["args"].get("cut_transport", "record")) != str(getattr(a, "cut_transport", "record")):      # v9 sector: the resume check gains the transport (4.5); a run without the argument (version 8's) is the record transport
        raise SystemExit("v9 sector: the run's --cut-transport is %r, this call's %r: a resume at a transport change goes through the launcher (pod_launch_C_v12b.sh rewrites the snapshot copy's args.cut_transport and keeps the old value as args.cut_transport_before)" % (J["args"].get("cut_transport", "record"), getattr(a, "cut_transport", "record")))   # v9 sector
    Z = np.load(a.resume_slice)
    Zf = np.load(os.path.join(rdir, "collapse_%s_fields.npz" % rtag), allow_pickle=True)''')))
R.append(("light_resume gap", "light_resume_v36(): the window's gap, mu and n_plus (the anchors in sector mode)", _s(r'''
        gap = light_select_v36(e_np, sig, mu)[1] if mu is not None else None'''), _s(r'''
        gap, mu, n_plus = light_resume_anchor_v9(a, e_np, sig, mask, mu, n_plus)                       # v9 sector: version 8's gap at the slice's mu in record mode; the anchors rebuilt from the mask and the spectrum in sector mode (4.3)''')))
R.append(("light_resume window", "light_resume_v36(): the window entry carries the spectrum and the sectors", _s(r'''
                        "window": [{"ref": ref, "f": (N_s.copy(), v_s.copy()), "gap": gap, "weyl_gap": gap, "mask": mask, "mu": mu, "n_plus": n_plus}]}'''), _s(r'''
                        "window": [{"ref": ref, "f": (N_s.copy(), v_s.copy()), "gap": gap, "weyl_gap": gap, "mask": mask, "mu": mu, "n_plus": n_plus, "e": e_np, "sigma": sig}]}   # v9 sector''')))
R.append(("light_follow window", "light_follow_v36(): the window entry carries the spectrum and the sectors", _s(r'''
            s_["window"] = [{"ref": ref, "f": (N_app.copy(), v_app.copy()), "gap": ci.get("gap"), "weyl_gap": ci.get("weyl_gap", ci.get("gap")), "mask": mask, "mu": ci.get("mu"), "n_plus": ci.get("n_plus")}]'''), _s(r'''
            s_["window"] = [{"ref": ref, "f": (N_app.copy(), v_app.copy()), "gap": ci.get("gap"), "weyl_gap": ci.get("weyl_gap", ci.get("gap")), "mask": mask, "mu": ci.get("mu"), "n_plus": ci.get("n_plus"), "e": e_np, "sigma": sig}]   # v9 sector''')))

# 5. run()
R.append(("run label", "run(): the run label after version 8's", _s(r'''
    if lagged_v8(a):                                                                                   # v8 lagged (the ruling's v4): (f) the run label
        label += "; v8 lagged (the ruling's engine_twofield_v4 lagged): " + lagged_desc_v8()           # v8 lagged (the ruling's v4)'''), _s(r'''
    if lagged_v8(a):                                                                                   # v8 lagged (the ruling's v4): (f) the run label
        label += "; v8 lagged (the ruling's engine_twofield_v4 lagged): " + lagged_desc_v8()           # v8 lagged (the ruling's v4)
    if sector_v9(a):                                                                                   # v9 sector: the run label
        label += "; v9 sector: " + sector_desc_v9()                                                    # v9 sector''')))
R.append(("head entry", "run(): the head names the transport", _s(r'''
    if lagged_v8(a):                                                                                   # v8 lagged (the ruling's v4): the head names the scheme
        head["reference_solve_v8"] = {"mode": "lagged", "description": lagged_desc_v8()}                # v8 lagged (the ruling's v4)'''), _s(r'''
    if lagged_v8(a):                                                                                   # v8 lagged (the ruling's v4): the head names the scheme
        head["reference_solve_v8"] = {"mode": "lagged", "description": lagged_desc_v8()}                # v8 lagged (the ruling's v4)
    if sector_v9(a):                                                                                   # v9 sector: the head names the transport
        head["cut_transport_v9"] = {"mode": "sector", "description": sector_desc_v9()}                 # v9 sector''')))
R.append(("heavy T0 window", "run(): the heavy T = 0 window entry carries the spectrum", _s(r'''
            st[ch.kappa] = {"sea": U[:, neg].clone(), "dust": Phi_d, "f": f_d, "ref": ref0, "n_sea": _k, "window": [{"ref": ref0, "f": (N.copy(), v.copy()), "gap": gap}]}'''), _s(r'''
            st[ch.kappa] = {"sea": U[:, neg].clone(), "dust": Phi_d, "f": f_d, "ref": ref0, "n_sea": _k, "window": [{"ref": ref0, "f": (N.copy(), v.copy()), "gap": gap, "e": sector_e_v9(_en)}]}   # v9 sector: the spectrum at the accepted fields on the host (the heavy sector walk's start; version 8 reads no new key)''')))
R.append(("light_meta", "run(): light_meta's note in sector mode", _s(r'''
    if lagged_v8(a):                                                                                   # v8 lagged (the ruling's v4): the fields npz's two references named
        light_meta["reference_solve_v8"] = ("lagged: eps_tot, J_tot, eps_pairs, J_pairs are the solve's sources (the held reference P_-(n-1)); eps_light_fed, J_light_fed, eps_L<k>, J_L<k> "   # v8 lagged (the ruling's v4)
                                            "are the stored reference's P_-(n), transported after acceptance; eps_heavy = eps_tot - eps_light_fed holds up to the light lag, the record's lag.d_eps_total.light")   # v8 lagged (the ruling's v4)'''), _s(r'''
    if lagged_v8(a):                                                                                   # v8 lagged (the ruling's v4): the fields npz's two references named
        light_meta["reference_solve_v8"] = ("lagged: eps_tot, J_tot, eps_pairs, J_pairs are the solve's sources (the held reference P_-(n-1)); eps_light_fed, J_light_fed, eps_L<k>, J_L<k> "   # v8 lagged (the ruling's v4)
                                            "are the stored reference's P_-(n), transported after acceptance; eps_heavy = eps_tot - eps_light_fed holds up to the light lag, the record's lag.d_eps_total.light")   # v8 lagged (the ruling's v4)
    if sector_v9(a):                                                                                   # v9 sector: the fields npz's note
        light_meta["cut_transport_v9"] = "sector: where a channel's cut has closed its reference is continued by the sector walk (the record's light_v36.channels[k].sector and continuation[k].sector entries name the route, the anchors and the crossings; a record-scale fallback or a flagged crossing is marked there; busi/notes/C64_transport_ruling_v1.md 4.3)"   # v9 sector''')))
R.append(("plain loop heavy call", "run(): the plain iteration loop's heavy continuation through continue_ref_v9", _s(r'''
                elif a.reference == "R2":                                                               # v8 lagged (the ruling's v4): version 3's `if` as the `elif` (the same statements on the implicit path)
                    t_c = time.time()
                    s_ = st[ch.kappa]
                    ref, cinfo, mask = continue_ref_v34(a, ch, s_["window"], (N_app, v_app), e, U, s_["n_sea"])'''), _s(r'''
                elif a.reference == "R2":                                                               # v8 lagged (the ruling's v4): version 3's `if` as the `elif` (the same statements on the implicit path)
                    t_c = time.time()
                    s_ = st[ch.kappa]
                    ref, cinfo, mask = continue_ref_v9(a, ch, s_["window"], (N_app, v_app), e, U, s_["n_sea"])   # v9 sector: continue_ref_v34's result in record mode (bitwise), the heavy sector walk in sector mode''')))
R.append(("heavy window", "run(): the heavy window entry after acceptance carries the spectrum (sector mode)", _s(r'''
            st[ch.kappa]["window"].append({"ref": cache[ch.kappa][3], "f": (N_app.copy(), v_app.copy()), "gap": ci_.get("gap_between_blocks"),
                                           "weyl_gap": ci_.get("weyl_gap", ci_.get("gap_between_blocks")), "mask": masks.get(ch.kappa)})'''), _s(r'''
            st[ch.kappa]["window"].append({"ref": cache[ch.kappa][3], "f": (N_app.copy(), v_app.copy()), "gap": ci_.get("gap_between_blocks"),
                                           "weyl_gap": ci_.get("weyl_gap", ci_.get("gap_between_blocks")), "mask": masks.get(ch.kappa), "e": (sector_e_v9(cache[ch.kappa][1]) if sector_v9(a) else None)})   # v9 sector: the spectrum at the accepted fields (the heavy sector walk's start)''')))
R.append(("light window", "run(): the light window entry after acceptance carries the spectrum and the sectors", _s(r'''
            st[chL.key]["window"] = [{"ref": cache[chL.key][3], "f": (N_app.copy(), v_app.copy()), "gap": ci_.get("gap"), "weyl_gap": ci_.get("weyl_gap", ci_.get("gap")),
                                      "mask": masksL.get(chL.key), "mu": ci_.get("mu"), "n_plus": ci_.get("n_plus")}]'''), _s(r'''
            st[chL.key]["window"] = [{"ref": cache[chL.key][3], "f": (N_app.copy(), v_app.copy()), "gap": ci_.get("gap"), "weyl_gap": ci_.get("weyl_gap", ci_.get("gap")),
                                      "mask": masksL.get(chL.key), "mu": ci_.get("mu"), "n_plus": ci_.get("n_plus"), "e": sector_e_v9(cache[chL.key][1]), "sigma": sigL.get(chL.key)}]   # v9 sector: the spectrum and sectors at the accepted fields on the host (the sector walk's anchors)''')))
R.append(("heavy cont entry", "run(): the heavy continuation entry gains \"sector\" (sector mode)", _s(r'''
                cont[str(ch.kappa)]["mask_equals_negative_set"] = same'''), _s(r'''
                cont[str(ch.kappa)]["mask_equals_negative_set"] = same
                if sector_v9(a):                                                                         # v9 sector: 4.3's per-channel entry (the heavy)
                    cont[str(ch.kappa)]["sector"] = cinfo.get("sector")                                  # v9 sector''')))
R.append(("light cont entry", "run(): the light channel entry gains \"sector\" (sector mode), min_occupied_weight in its present meaning", _s(r'''
            c_["mask_equals_negative_set"] = bool(np.array_equal(mk, (e < 0).cpu().numpy()))'''), _s(r'''
            c_["mask_equals_negative_set"] = bool(np.array_equal(mk, (e < 0).cpu().numpy()))
            if sector_v9(a):                                                                             # v9 sector: 4.3's per-channel entry
                c_["sector"] = dict(ci.get("sector") or {}); c_["sector"]["min_occupied_weight"] = ci["min_occupied_weight"]   # v9 sector''')))
R.append(("record keys", "run(): the record's key and label (sector mode)", _s(r'''
        if lagged_v8(a):                                                                                # v8 lagged (the ruling's v4): (d) the record's keys, (f) the record label
            rec["reference_solve"] = "lagged"; rec["lag"] = lag_v8                                      # v8 lagged (the ruling's v4)
            rec["record_label"] += "; reference transported, held in the solve (engine_twofield_v8 --reference-solve lagged, the ruling's engine_twofield_v4 lagged)"   # v8 lagged (the ruling's v4)'''), _s(r'''
        if lagged_v8(a):                                                                                # v8 lagged (the ruling's v4): (d) the record's keys, (f) the record label
            rec["reference_solve"] = "lagged"; rec["lag"] = lag_v8                                      # v8 lagged (the ruling's v4)
            rec["record_label"] += "; reference transported, held in the solve (engine_twofield_v8 --reference-solve lagged, the ruling's engine_twofield_v4 lagged)"   # v8 lagged (the ruling's v4)
        if sector_v9(a):                                                                                # v9 sector: the record's key and label (4.3)
            rec["cut_transport"] = "sector"; rec["record_label"] += "; at closed cuts the reference continued by the sector walk (engine_twofield_v9b --cut-transport sector)"   # v9 sector''')))
R.append(("record line", "run(): the record line's sector segment after version 32's", _s(r'''
        print(_line + (v32_record_segment(solve32) if solve32 is not None else ""), flush=True)'''), _s(r'''
        print(_line + (v32_record_segment(solve32) if solve32 is not None else "") + (sector_record_segment_v9(cont, contL) if sector_v9(a) else ""), flush=True)   # v9 sector: the sector segment (sector mode)''')))
R.append(("light line", "run(): the light line's per-channel segment (version 8's text in record mode)", _s(r'''
                "; ".join("kappa %s: %s mu %s gap %s pieces %s d_min %s pockets %d inv %d wmin %s E_inv %.3e" % (
                    k_, ("certified" if q.get("verified") else "FALLBACK (%s)" % q.get("why")), fmt(q.get("mu"), "%.3e"), fmt(q.get("gap"), "%.2e"), q.get("nsub"),
                    fmt((q.get("certified") or {}).get("d_min_certified") if isinstance(q.get("certified"), dict) else None, "%.2e"), q.get("n_pocket") or 0, q.get("inversions_sector") or 0,
                    fmt(q.get("min_occupied_weight"), "%.4f"), q.get("E_inv")) for k_, q in recL["channels"].items()),'''), _s(r'''
                "; ".join(light_channel_segment_v9(a, k_, q) for k_, q in recL["channels"].items()),   # v9 sector: version 8's text in record mode (bitwise), the sector walk's in sector mode (4.3)''')))
R.append(("script periodic", "run(): the periodic json dump's script name", _s(r'''
                json.dump({"script": "engine_twofield_v8.py", "args": vars(a), "head": head, "records": recs}, fh, default=float)   # v3; v8 lagged (the ruling's v4): the script's name'''), _s(r'''
                json.dump({"script": "engine_twofield_v9b.py", "args": vars(a), "head": head, "records": recs}, fh, default=float)   # v3; v8 lagged (the ruling's v4); v9 sector: the script's name''')))
R.append(("script s11 stop", "run(): the P2 stop's json dump's script name", _s(r'''
                json.dump({"script": "engine_twofield_v8.py", "args": vars(a), "head": head, "records": recs, "end_s11": end33}, fh, default=float)   # v3; v8 lagged (the ruling's v4): the script's name'''), _s(r'''
                json.dump({"script": "engine_twofield_v9b.py", "args": vars(a), "head": head, "records": recs, "end_s11": end33}, fh, default=float)   # v3; v8 lagged (the ruling's v4); v9 sector: the script's name''')))
R.append(("script final", "run(): the final json dump's script name", _s(r'''
        json.dump(dict({"script": "engine_twofield_v8.py", "args": vars(a), "head": head, "records": recs}, **({"end_v32": end32} if end32 is not None else {}),   # v3; v8 lagged (the ruling's v4): the script's name'''), _s(r'''
        json.dump(dict({"script": "engine_twofield_v9b.py", "args": vars(a), "head": head, "records": recs}, **({"end_v32": end32} if end32 is not None else {}),   # v3; v8 lagged (the ruling's v4); v9 sector: the script's name''')))
R.append(("script light-follow", "light_follow_v36(): the json's script name", _s(r'''
                json.dump({"script": "engine_twofield_v8.py (light-follow)", "args": vars(a), "followed": a.light_follow, "label": label, "head": {"light_v36": headL, "subcritical_T0": tz0},   # v3; v8 lagged (the ruling's v4): the script's name'''), _s(r'''
                json.dump({"script": "engine_twofield_v9b.py (light-follow)", "args": vars(a), "followed": a.light_follow, "label": label, "head": {"light_v36": headL, "subcritical_T0": tz0},   # v3; v8 lagged (the ruling's v4); v9 sector: the script's name''')))
R.append(("slice anchors", "light_slice_arrays_v36(): anchors_L<k> in sector mode (mu_L the first anchor there)", _s(r'''
        arrs["nplus_L%d" % k] = int(nplusL[chL.key]) if nplusL[chL.key] is not None else -1'''), _s(r'''
        arrs["nplus_L%d" % k] = int(nplusL[chL.key]) if nplusL[chL.key] is not None else -1
        sec_ = _ci.get("sector") if isinstance(_ci, dict) else None                                       # v9 sector: the anchor list of 4.3 (sector mode only; the slice is otherwise version 8's)
        if sec_ is not None and sec_.get("anchors") is not None:                                          # v9 sector
            arrs["anchors_L%d" % k] = np.asarray(sec_["anchors"], float)                                  # v9 sector''')))

# 7. the heavy calls in newton_record, nk_stage_v32, transport_references_v8
R.append(("newton_record evalG", "newton_record(): evalG's heavy continuation through continue_ref_v9", _s(r'''
    nk = {"best": None, "n": 0, "E_prev": E0, "failed": None}

    def evalG(N_a, v_a):
        eig = eigh_all(chs, N_a, v_a, bool(a.streams))
        cache_ = dict(cache0)
        for ch in chs:
            e, U = eig.pop(ch.kappa)
            if lagged_v8(a):                                                                             # v8 lagged (the ruling's v4): (a) the heavy reference held through the evaluation, no continuation here
                ref = st[ch.kappa]["ref"]; cinfo = held_cinfo_v8()                                       # v8 lagged (the ruling's v4): P_-(n-1), the stored ref columns
            elif a.reference == "R2":                                                                   # v8 lagged (the ruling's v4): version 3's `if` as the `elif` (the same statements on the implicit path)
                t_c = time.time()
                s_ = st[ch.kappa]
                ref, cinfo, mask = continue_ref_v34(a, ch, s_["window"], (N_a, v_a), e, U, s_["n_sea"])'''), _s(r'''
    nk = {"best": None, "n": 0, "E_prev": E0, "failed": None}

    def evalG(N_a, v_a):
        eig = eigh_all(chs, N_a, v_a, bool(a.streams))
        cache_ = dict(cache0)
        for ch in chs:
            e, U = eig.pop(ch.kappa)
            if lagged_v8(a):                                                                             # v8 lagged (the ruling's v4): (a) the heavy reference held through the evaluation, no continuation here
                ref = st[ch.kappa]["ref"]; cinfo = held_cinfo_v8()                                       # v8 lagged (the ruling's v4): P_-(n-1), the stored ref columns
            elif a.reference == "R2":                                                                   # v8 lagged (the ruling's v4): version 3's `if` as the `elif` (the same statements on the implicit path)
                t_c = time.time()
                s_ = st[ch.kappa]
                ref, cinfo, mask = continue_ref_v9(a, ch, s_["window"], (N_a, v_a), e, U, s_["n_sea"])   # v9 sector: continue_ref_v34's result in record mode (bitwise), the heavy sector walk in sector mode''')))
R.append(("nk_stage_v32 evalG", "nk_stage_v32(): evalG's heavy continuation through continue_ref_v9", _s(r'''
    n_start = ctr["n"]

    def evalG(N_a, v_a):
        eig = eigh_all(chs, N_a, v_a, bool(a.streams))
        cache_ = dict(cache0)
        for ch in chs:
            e, U = eig.pop(ch.kappa)
            if lagged_v8(a):                                                                             # v8 lagged (the ruling's v4): (a) the heavy reference held through the evaluation, no continuation here
                ref = st[ch.kappa]["ref"]; cinfo = held_cinfo_v8()                                       # v8 lagged (the ruling's v4): P_-(n-1), the stored ref columns
            elif a.reference == "R2":                                                                   # v8 lagged (the ruling's v4): version 3's `if` as the `elif` (the same statements on the implicit path)
                t_c = time.time()
                s_ = st[ch.kappa]
                ref, cinfo, mask = continue_ref_v34(a, ch, s_["window"], (N_a, v_a), e, U, s_["n_sea"])'''), _s(r'''
    n_start = ctr["n"]

    def evalG(N_a, v_a):
        eig = eigh_all(chs, N_a, v_a, bool(a.streams))
        cache_ = dict(cache0)
        for ch in chs:
            e, U = eig.pop(ch.kappa)
            if lagged_v8(a):                                                                             # v8 lagged (the ruling's v4): (a) the heavy reference held through the evaluation, no continuation here
                ref = st[ch.kappa]["ref"]; cinfo = held_cinfo_v8()                                       # v8 lagged (the ruling's v4): P_-(n-1), the stored ref columns
            elif a.reference == "R2":                                                                   # v8 lagged (the ruling's v4): version 3's `if` as the `elif` (the same statements on the implicit path)
                t_c = time.time()
                s_ = st[ch.kappa]
                ref, cinfo, mask = continue_ref_v9(a, ch, s_["window"], (N_a, v_a), e, U, s_["n_sea"])   # v9 sector: continue_ref_v34's result in record mode (bitwise), the heavy sector walk in sector mode''')))
R.append(("transport heavy call", "transport_references_v8(): the heavy transport through continue_ref_v9", _s(r'''
        ref, cinfo, mask = continue_ref_v34(a, ch, s_["window"], (N_app, v_app), e, U, s_["n_sea"])      # v8 lagged (the ruling's v4)'''), _s(r'''
        ref, cinfo, mask = continue_ref_v9(a, ch, s_["window"], (N_app, v_app), e, U, s_["n_sea"])       # v8 lagged (the ruling's v4); v9 sector: continue_ref_v34's result in record mode (bitwise), the heavy sector walk in sector mode''')))

# 8. main(): the argument and the checks
R.append(("argument", "main(): the run argument --cut-transport {record, sector}", _s(r'''
    ap.add_argument("--reference-solve", dest="reference_solve", default="implicit", choices=["implicit", "lagged"], help="(v8 lagged, the ruling's engine_twofield_v4) implicit: version 3 bitwise; lagged: the references held through the record's solve at the previous accepted record's and transported once along the accepted path after acceptance, the lag residual printed (busi/notes/C64_fix_ruling_v4.md I.2)")   # v8 lagged (the ruling's v4)
    a = ap.parse_args()'''), _s(r'''
    ap.add_argument("--reference-solve", dest="reference_solve", default="implicit", choices=["implicit", "lagged"], help="(v8 lagged, the ruling's engine_twofield_v4) implicit: version 3 bitwise; lagged: the references held through the record's solve at the previous accepted record's and transported once along the accepted path after acceptance, the lag residual printed (busi/notes/C64_fix_ruling_v4.md I.2)")   # v8 lagged (the ruling's v4)
    ap.add_argument("--cut-transport", dest="cut_transport", default="record", choices=["record", "sector"], help="(v9 sector, 9b) record: version 8 bitwise in both --reference-solve modes; sector: past a closed occupation cut the reference continued by the sector walk, the declared convention applied to each crossing at the scale of the crossing (the piece bracketed by certified probe points), the walk certified at every boundary of the occupied set, the record-scale convention the flagged fallback (busi/notes/C64_transport_ruling_v1.md section 4; busi/notes/C64_v9_fix_v1.md)")   # v9 sector
    a = ap.parse_args()''')))
R.append(("checks", "main(): the sector mode's 2 checks (--reference R2; --light-transport banded with light channels)", _s(r'''
    if lagged_v8(a) and a.light_follow:                                                                  # v8 lagged (the ruling's v4)
        ap.error("v8 lagged (the ruling's v4): --light-follow has no record solve; --reference-solve lagged does not apply to it")   # v8 lagged (the ruling's v4)'''), _s(r'''
    if lagged_v8(a) and a.light_follow:                                                                  # v8 lagged (the ruling's v4)
        ap.error("v8 lagged (the ruling's v4): --light-follow has no record solve; --reference-solve lagged does not apply to it")   # v8 lagged (the ruling's v4)
    if sector_v9(a) and a.reference != "R2":                                                             # v9 sector
        ap.error("v9 sector: --cut-transport sector continues the sea R2 past its closed cut: --reference R2 is required")   # v9 sector
    if sector_v9(a) and (a.light_kappas or a.light_guard_kappas) and a.light_transport != "banded":      # v9 sector
        ap.error("v9 sector: the sector walk generalizes the banded (mu S) walk: --light-transport banded is required with --cut-transport sector")   # v9 sector''')))


def build(src):
    """every OLD exactly once in version 8's text (the line numbers taken there), then the replacements in order, each exactly once at its turn."""
    pos = []
    for name, where, old, new in R:
        hits = list(pattern_of(old).finditer(src))
        if len(hits) != 1:
            raise SystemExit("build: %r (%s) occurs %d times in version 8, expected exactly once; the version 8 text differs from input 03 at that place; nothing written" % (name, where, len(hits)))
        pos.append(src[:hits[0].start()].count("\n") + 1)
    out = src
    for name, where, old, new in R:
        hits = list(pattern_of(old).finditer(out))
        if len(hits) != 1:
            raise SystemExit("build: %r (%s) matched %d times at its turn, expected exactly once; nothing written" % (name, where, len(hits)))
        out = out[:hits[0].start()] + new + out[hits[0].end():]
    log = []
    for (name, where, old, new), l8 in zip(R, pos):
        k = out.find(new)
        log.append((name, where, l8, (out[:k].count("\n") + 1) if k >= 0 else -1))
    return out, log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v8", required=True, help="engine_twofield_v8.py (input 03, as built and red-teamed)")
    ap.add_argument("--out", required=True, help="engine_twofield_v9b.py to write")
    a = ap.parse_args()
    src = open(a.v8, encoding="utf-8").read()
    if "engine_twofield_v8.py (the busi lane, Computation C; the fix ruling" not in src or MARK8 not in src:
        raise SystemExit("build: %s is not the built engine_twofield_v8.py" % a.v8)
    if MARK in src:
        raise SystemExit("build: %s already carries the mark %r; the source must be version 8" % (a.v8, MARK))
    out, log = build(src)
    compile(out, a.out, "exec")
    with open(a.out, "w", encoding="utf-8") as fh:
        fh.write(out)
    n_marked = sum(1 for l_ in out.splitlines() if MARK in l_)
    print("built %s from %s: %d replacements, each matched exactly once; %d lines carry the mark %r; %d -> %d lines; compiled (version 9b: the crossing's piece bracketed by certified probe points, busi/notes/C64_v9_fix_v1.md)" % (
        a.out, a.v8, len(log), n_marked, MARK, len(src.splitlines()), len(out.splitlines())))
    for name, where, l8, l9 in log:
        print("  %-26s version 8 line %5d -> version 9b line %5d: %s" % (name, l8, l9, where))


if __name__ == "__main__":
    main()
