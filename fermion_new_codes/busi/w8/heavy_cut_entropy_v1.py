#!/usr/bin/env python
"""heavy_cut_entropy_v1.py (reviewer A at maximum effort, the busi lane's code author; W8 of REVIEWA_REVIEW_web40i.md section 4,
requirements_new.txt edit 25). A slices144 readout (READOUT "heavycut", the form of page_own_object_v1.py): on every stored slice the
driver selects (the production's 10 and the sigma/a 8 member's 8 in RUN_W8_v1.md), per channel kappa and summed over the channels with the
multiplicity 16 kappa, at the cuts 2, 16 and 24 r_s, the heavy field's exterior-cut entropy from the whole heavy covariance and every term
of the decomposition S_H = S_P + (S_Q - S_P) + S_d - K_sd with its bound, on the GPU (the A6000) where the dense eigensolves are large.

Definitions (the review's terms, REVIEWA_REVIEW_web40i F7 and edit 25; the request's provisional glosses "S_P the reference sea's entropy at
the cut, S_Q the physical sea's, S_d the dust's, K_sd the sea-dust correlation term" hold with the 2 precisions marked (*)):
  the cut X = {R > x} (the 2 rows of every site with R_n > x; Xc the rest), s(p) = -p ln p - (1 - p) ln(1 - p) in nats; the channel's stored
  occupied columns Phi = [Phi_s | Phi_d] (sea, dust) Loewdin-orthonormalized, the dust's stored occupations f_i; Q = Phi_s Phi_s^dagger,
  D = Phi_d f Phi_d^dagger, C = Q + D the whole heavy covariance of the channel (Q D = 0, the sea and the dust being orthogonal column
  sets, and C <= 1); P = P_-(h_m) = 1(h_m < 0) the held Killing vacuum at the slice's applied fields.
- S_H = Tr s(C_X): the heavy field's exterior-cut entropy from the whole heavy covariance, computed from the nonzero spectrum of
  Y_X^dagger Y_X with Y = [Phi_s | Phi_d f^(1/2)] (C_X = Y_X Y_X^dagger; the Gram route, the smaller of Y_X^dagger Y_X and Y_X Y_X^dagger as
  page_d13_readout_v1b.gram_eigs chooses it, never the n2 x n2 covariance).
- S_P = Tr s(P_X): the held Killing vacuum's entropy at the same cut (one dense eigendecomposition of h_m per channel). (*) The review's
  first term is the baseline of every excess it names, "the excess over the held Killing vacuum of the same fields at the same cut and
  record, S(C) - S(P_-(h))" (edits 7 to 9 and Y2): the held Killing vacuum, not the continued incoming sea (the 2 coincide where no mode has
  crossed zero energy). Where the slice's mask_k (the engine's occupied set in the eigenbasis of h_m) is present, the projector it selects
  prints beside as S_mask with its overlap with the physical sea: a diagnostic of the continued reference, not a term of the decomposition.
- S_Q = Tr s(Q_X): the physical sea's entropy at the cut (the sea columns alone, f = 1; a pure Gaussian state, S_X = S_Xc, the gap printed).
- S_Q - S_P: the sea's excess, signed (a created excitation can lower the cut entropy, edit 9); its bound |S_Q - S_P| <= t_X =
  2 ||Pi_X (Q - P) Pi_Xc||_1 (EVOLVEDCUT EC1, page_d13_readout_v1b.cross_cut_norms by name: the X-by-Xc block's singular values).
- S_d = Tr s(D_X): the dust's entropy at the cut (the dust columns alone with their f). Its bounds: the straddling-orbital bound
  S_d <= sum_i s(f_i w_i), w_i = ||Pi_X d_i||^2 the exterior weight of dust orbital i (Rotfel'd's trace inequality on the rank-1 terms; the
  requirements' N_B h(f_N) form), and the count bound min(n_d, n_X) ln 2.
- K_sd = S_Q + S_d - S_H: (*) the restriction term (the review's name). It is the quantity the request calls the sea-dust correlation term,
  with this reading: the 2 entropies are not additive because the cut does not respect the sea-dust mode split; the state itself is the
  product of the sea's and the dust's. The review's bounds 0 <= K_sd <= 2 min(S_Q, S_d) hold: with T = [Q_X^(1/2), D_X^(1/2)] the 2-block
  covariance Gamma = T^dagger T has the nonzero spectrum of T T^dagger = C_X, so Tr s(Gamma) = S_H; its diagonal blocks are Q_X and D_X and
  0 <= Gamma <= 1; the Gaussian state of Gamma on 2 fermion blocks obeys subadditivity, S_H <= S_Q + S_d (Rotfel'd's inequality for the
  concave s with s(0) = 0), and the triangle inequality S_H >= |S_Q - S_d|, which is K_sd <= 2 min(S_Q, S_d). Both are checked on every
  row; the auxiliary covariance is a device of the proof and is neither formed nor printed (D15: no mutual information).
- S_H = S_P + (S_Q - S_P) + S_d - K_sd holds by the definition of K_sd; the arithmetic residual prints as a check. The Page panel's heavy
  curve is the heavy excess S_H - S_P = (S_Q - S_P) + S_d - K_sd (Y13: the panel's axis is the excess); the raw S_H and S_P print beside it
  as regulated tables (the radial truncation's raw entropy has no limit in kappa); the bracket |S_H - S_P| <= t_X + S_d prints with it.
- the complement's entropy prints for every term: equal for the pure Q and P (the purity gap, a rounding check); for C and D the gap is
  bounded by the dust's own entropy sum_i s(f_i), 0 where every f_i is 1 (input 05: "Phi_k = [sea_k | dust_k] (f all 1)"; the f prints).

Assumptions in place of W13 (stated here and in every json; W13's STATUS lines supersede them; nothing weighted prints before W13 lands):
- the chain's dimension per channel: 2 nsites, 4096 for the production and the sigma/a 8 member (2048 two-component sites; input 05
  SliceReader: n2 = 2 nsites, rows 2j and 2j + 1 belong to site j), 8192 on B2; read from the slice and printed.
- the sea's column count: the slice's sea_k member, shape[1] (SliceReader.n_sea), with f = 1 on every sea column (SliceReader.channel:
  f = [ones(n_sea), f_dust]); printed per channel beside the held vacuum's occupied count n_neg and the mask's popcount.
- the dust's occupation across the 16 kappa copies: ASSUMED whole multiplets, every copy of channel kappa carrying the stored dust columns
  dust_k with the stored f_k, so the per-channel S_H, S_d and K_sd are multiplied by the slice's weights (16 kappa; SliceReader.weights)
  like the sea's terms. The check printed with every slice is the weighted dust count sum_kappa 16 kappa sum_i f_i against N_B (the records
  json's argument where one is named, else 640 at (0.2, 64) from requirements_new.txt) beside the unweighted count. Where W13 states a
  partially filled multiplet, --copy-pattern FILE (per channel the list of the 16 kappa copies' dust column subsets) recomputes the dust
  rows as the sum over the copies (S_Q and S_P are the same in every copy) and prints them as weighted_pattern; the per-channel
  (unweighted) rows stand under either answer.
- the g_n guard: ASSUMED to be the sources' rule (requirements section 6: no degeneracy weight on any source in the radial codes), which
  this readout does not touch (it weights entropies, never sources); its meaning for the dust sources, whether a stored dust column sources
  once or 16 kappa times, is W13's and is the same question as the count check above.

Per slice this module writes heavycut_<history>_T<clock>.json (every scalar) and .npz (the exterior and interior spectra of C, Q, D and P
per cut and channel, the dust orbitals' exterior weights, the cross-cut singular values); compose writes HEAVYCUT_W8_v1_<history>.md and
HEAVYCUT_W8_v1_all.json. No plotting here (heavy_cut_plot_v1.py draws from the composed json into a separate folder); no mutual
information (D15); no dates (elapsed seconds only). The matter flux through the cuts with the Killing split at m and b stays
page_own_object_v1's readout; where its json for the same slice is present in <out>/page the comparison prints: S_H against its S_X
(equal where every f is 1), S_P against its S_X^vac, the dust number beyond the cut against its N_out.

Driver: python heavy_cut_entropy_v1.py --history prod,s8 --clocks all --cuts-rs 2,16,24 --device cuda:0 --threads 16 --out DIR
[--slices-code DIR_OF_INPUT_05]; resumable per slice (a slice's json present = done; --force recomputes); --compose-only for the tables."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
READOUT = "heavycut"
TITLE = "HEAVYCUT_W8_v1"
CODE = "heavy_cut_entropy_v1.py"
MODULE_NAME = "heavy_cut_entropy_v1"
PRINT_STATUS = ("not for print until W13 states the dust's per-copy occupation across the 16 kappa copies of each channel and the lead has read the tables; "
                "the weighted sums assume whole multiplets (every copy of channel kappa carries the stored dust columns with the stored f) and carry the count check against N_B")
D15 = "no mutual information is computed or printed (D15); the auxiliary 2-block covariance of the bounds' proof is not formed"
C_STAR_DEFAULT = 1.325486838698363
IO = None


def _pre_threads():
    """the BLAS and torch thread count from --threads (16 where not given), set before numpy is imported (slices144_io_v1's rule)."""
    n = None
    argv = sys.argv[1:]
    for i, x in enumerate(argv):
        if x == "--threads" and i + 1 < len(argv):
            n = argv[i + 1]
        elif x.startswith("--threads="):
            n = x.split("=", 1)[1]
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        if n is not None:
            os.environ[var] = str(int(n))
        else:
            os.environ.setdefault(var, "16")


def bind_slices_code(d=None, required=False):
    """imports slices144_io_v1 (input 05) as IO from: sys.path, the argument, $SLICES144_CODE, this folder, the lane's usual folders."""
    global IO
    if IO is not None:
        return IO
    try:
        import slices144_io_v1 as _io
        IO = _io
        return IO
    except ImportError:
        pass
    cands = [d] if d else []
    if os.environ.get("SLICES144_CODE"):
        cands += os.environ["SLICES144_CODE"].split(os.pathsep)
    cands += [HERE, os.path.join(HERE, "..", "slices144"), os.path.join(HERE, "..", "..", "slices144"), "<internal>/slices144"]
    for c in cands:
        c = os.path.abspath(c)
        if os.path.isfile(os.path.join(c, "slices144_io_v1.py")):
            if c not in sys.path:
                sys.path.insert(0, c)
            import slices144_io_v1 as _io
            IO = _io
            return IO
    if required:
        raise ImportError("slices144_io_v1.py (input 05) was not found: pass --slices-code DIR or set SLICES144_CODE to the folder that holds it")
    return None


_pre_threads()
bind_slices_code(None, False)
import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

CD = torch.complex128
RD = torch.float64
LN2 = math.log(2.0)

DEFINITIONS = {
    "cut": "X = {R > x}: the 2 rows of every site with R_n > x (the slice's R, R_n = n + 1 in lattice units); Xc its complement; S_Z the entropy of the Z block",
    "S_H": "Tr s(C_X) of the whole heavy covariance C = Phi_s Phi_s^dagger + Phi_d f Phi_d^dagger of the channel, from the nonzero spectrum of Y_X^dagger Y_X, Y = [Phi_s | Phi_d f^(1/2)] (the Gram route)",
    "S_P": "Tr s(P_X) of the held Killing vacuum P = 1(h_m < 0) at the slice's applied fields (the baseline of every excess in REVIEWA_REVIEW_web40i edits 7 to 9 and Y2; not the continued reference)",
    "S_Q": "Tr s(Q_X) of the physical sea Q = Phi_s Phi_s^dagger alone (f = 1)",
    "excess_sea": "S_Q - S_P, the sea's excess, signed; |S_Q - S_P| <= t_X = 2 ||Pi_X (Q - P) Pi_Xc||_1 (EVOLVEDCUT EC1)",
    "S_d": "Tr s(D_X) of the dust D = Phi_d f Phi_d^dagger alone; S_d <= sum_i s(f_i w_i) with w_i the dust orbital's exterior weight (the straddling-orbital bound) and S_d <= min(n_d, n_X) ln 2",
    "K_sd": "S_Q + S_d - S_H, the restriction term; 0 <= K_sd <= 2 min(S_Q, S_d) (subadditivity and the triangle inequality of the 2-block Gaussian state whose covariance has the spectrum of C_X and the blocks Q_X, D_X)",
    "heavy_excess": "S_H - S_P = (S_Q - S_P) + S_d - K_sd, the Page panel's heavy curve; |S_H - S_P| <= t_X + S_d",
    "identity": "S_H = S_P + (S_Q - S_P) + S_d - K_sd by the definition of K_sd; decomposition_residual is its arithmetic residual",
    "complements": "every term's complement entropy printed; S_X = S_Xc for the pure Q and P; |S_X - S_Xc| <= sum_i s(f_i) for C and D (0 where every f_i is 1)",
    "S_mask": "the cut entropy of the projector the slice's mask_k selects in the ascending eigenbasis of h_m (a diagnostic of the continued reference, not a term); tr_Q_Pmask its overlap with the physical sea",
    "weights": "the per-channel terms multiplied by the slice's weights 16 kappa under the whole-multiplet assumption (W13 pending); the per-channel rows are unweighted",
}


def pick_device(name, log=None):
    if str(name).startswith("cuda") and torch.cuda.is_available():
        return torch.device(name)
    if str(name).startswith("cuda") and log:
        log("WARNING: %s is not available here: the CPU runs the dense eigensolves" % name)
    return torch.device("cpu")


# ---------------------------------------------------------------------------------------------------------------- the arithmetic
def h_t(x):
    """s(p) = -p ln p - (1 - p) ln(1 - p) elementwise on a float64 tensor clipped to [0, 1] (page_ops_v6.binary_entropy_t's expression)."""
    x = x.clamp(0.0, 1.0)
    return -(torch.special.xlogy(x, x) + torch.special.xlogy(1.0 - x, 1.0 - x))


def h_np(x):
    x = np.clip(np.asarray(x, float), 0.0, 1.0)
    out = np.zeros_like(x)
    msk = (x > 0.0) & (x < 1.0)
    out[msk] = -(x[msk] * np.log(x[msk]) + (1.0 - x[msk]) * np.log1p(-x[msk]))
    return out


def gram_eigs(X):
    """the nonzero spectrum of X X^dagger from the smaller of X^dagger X and X X^dagger (page_d13_readout_v1b.gram_eigs' rule); (ascending, route)."""
    if X.shape[0] == 0 or X.shape[1] == 0:
        return torch.zeros(0, dtype=RD, device=X.device), "empty"
    if X.shape[1] <= X.shape[0]:
        G = X.mH @ X
        route = "gram %d x %d" % (X.shape[1], X.shape[1])
    else:
        G = X @ X.mH
        route = "block %d x %d" % (X.shape[0], X.shape[0])
    G = 0.5 * (G + G.mH)
    ev = torch.linalg.eigvalsh(G)
    del G
    return ev, route


def entropy(ev):
    return float(h_t(ev).sum()) if ev.numel() else 0.0


def excursion(ev):
    return max(0.0, float(-ev.min()), float(ev.max()) - 1.0) if ev.numel() else 0.0


def block_entropy(X, mask):
    """S_X and S_Xc of the covariance X X^dagger (X orthonormal columns, scaled by sqrt f where the state is mixed) at the cut mask."""
    evX, rX = gram_eigs(X[mask])
    evC, rC = gram_eigs(X[~mask])
    return {"S_X": entropy(evX), "S_Xc": entropy(evC), "ev_X": evX, "ev_Xc": evC, "route_X": rX, "route_Xc": rC,
            "excursion": max(excursion(evX), excursion(evC)), "content_X": float(evX.sum()) if evX.numel() else 0.0,
            "content_Xc": float(evC.sum()) if evC.numel() else 0.0}


def spectrum_bound(ev, cstar=C_STAR_DEFAULT, ledger_fn=None):
    """page_own_object_v1.cross_cut_bound's arithmetic on an exterior spectrum: t = sum sqrt(lambda (1 - lambda)), S_X <= (pi C*/2) t (the
    singular values of the cross-cut block for a pure state; a spectrum statistic for a mixed one), the per-mode ledger where importable."""
    lam = np.clip(np.asarray(ev, float), 0.0, 1.0)
    s2 = lam * (1.0 - lam)
    t = float(np.sum(np.sqrt(s2)))
    out = {"t": t, "bound_pi_Cstar_over_2_t": 0.5 * math.pi * cstar * t, "modes_mixed_above_1e-12": int(np.sum(s2 > 1e-12)), "C_star": cstar}
    if ledger_fn is not None:
        try:
            out["bound_pair_ledger_sharp"] = float(ledger_fn(np.sqrt(np.minimum(lam, 1.0 - lam))))
        except Exception as ex:  # noqa: BLE001
            out["bound_pair_ledger_sharp"] = None
            out["ledger_error"] = "%s: %s" % (type(ex).__name__, ex)
    return out


def cut_terms(S, Dc, f_d, V, mask, lab, x_abs, cstar=C_STAR_DEFAULT, ledger_fn=None, tol=1e-9):
    """every term of the decomposition at one cut for one channel: S_H, S_P, S_Q, S_d, their complements, K_sd, the sea's excess, the heavy
    excess, the identity's residual, the bounds (K_sd in [0, 2 min(S_Q, S_d)], S_H in [|S_Q - S_d|, S_Q + S_d], the straddling-orbital and
    count bounds of S_d, the purity gaps against the dust's own entropy, the spectrum bounds of P, Q and H). S (n2 x n_s) and V (n2 x n_neg)
    orthonormal device columns, Dc (n2 x n_d) orthonormal dust columns with occupations f_d, mask the exterior rows (bool, n2).
    The cross-cut bound t_X is attached by attach_cross_cut. Returns (scalars, arrays)."""
    dev = S.device
    f_np = np.clip(np.asarray(f_d, float).reshape(-1), 0.0, 1.0)
    if Dc.shape[1]:
        Dsc = Dc * torch.sqrt(torch.as_tensor(f_np, dtype=RD, device=dev)).to(CD)[None, :]
    else:
        Dsc = Dc
    Y = torch.cat([S, Dsc], dim=1)
    nX = int(mask.sum())
    nC = int(mask.numel()) - nX
    d = {"label": lab, "R_abs": float(x_abs), "n_modes_X": nX, "n_modes_Xc": nC, "n_sea": int(S.shape[1]), "n_dust": int(Dc.shape[1]), "n_vac": int(V.shape[1])}
    arrays = {}
    for key, X in (("H", Y), ("Q", S), ("d", Dsc), ("P", V)):
        b = block_entropy(X, mask)
        d["S_" + key] = b["S_X"]
        d["S_%s_c" % key] = b["S_Xc"]
        d["content_X_" + key] = b["content_X"]
        d["content_Xc_" + key] = b["content_Xc"]
        d["route_" + key] = [b["route_X"], b["route_Xc"]]
        d["excursion_" + key] = b["excursion"]
        arrays["eig_X_" + key] = b["ev_X"].cpu().numpy()
        arrays["eig_Xc_" + key] = b["ev_Xc"].cpu().numpy()
    del Y, Dsc
    wX = (Dc[mask].abs() ** 2).sum(dim=0).cpu().numpy() if Dc.shape[1] else np.zeros(0)
    SH, SQ, Sd, SP = d["S_H"], d["S_Q"], d["S_d"], d["S_P"]
    SHc, SQc, Sdc, SPc = d["S_H_c"], d["S_Q_c"], d["S_d_c"], d["S_P_c"]
    K = SQ + Sd - SH
    Kc = SQc + Sdc - SHc
    d.update({"K_sd": K, "K_sd_c": Kc, "excess_sea": SQ - SP, "excess_sea_c": SQc - SPc, "heavy_excess": SH - SP, "heavy_excess_c": SHc - SPc,
              "decomposition_rhs": SP + (SQ - SP) + Sd - K, "decomposition_residual": (SP + (SQ - SP) + Sd - K) - SH})
    strad = float(np.sum(h_np(f_np * wX)))
    strad_c = float(np.sum(h_np(f_np * (1.0 - wX))))
    mixed = float(np.sum(h_np(f_np)))
    count_bound = min(int(Dc.shape[1]), nX) * LN2
    d["bounds"] = {
        "K_sd_lower": 0.0, "K_sd_upper": 2.0 * min(SQ, Sd), "K_sd_in_bounds": bool(-tol <= K <= 2.0 * min(SQ, Sd) + tol),
        "K_sd_c_upper": 2.0 * min(SQc, Sdc), "K_sd_c_in_bounds": bool(-tol <= Kc <= 2.0 * min(SQc, Sdc) + tol),
        "S_H_lower": abs(SQ - Sd), "S_H_upper": SQ + Sd, "S_H_in_bounds": bool(abs(SQ - Sd) - tol <= SH <= SQ + Sd + tol),
        "S_d_straddling_bound": strad, "S_d_c_straddling_bound": strad_c, "S_d_count_bound": count_bound,
        "S_d_in_bounds": bool(Sd <= strad + tol and Sd <= count_bound + tol and Sdc <= strad_c + tol),
        "dust_mixedness": mixed, "purity_gap_H": SH - SHc, "purity_gap_d": Sd - Sdc, "purity_gap_Q": SQ - SQc, "purity_gap_P": SP - SPc,
        "purity_gaps_in_bounds": bool(abs(SH - SHc) <= mixed + tol and abs(Sd - Sdc) <= mixed + tol and abs(SQ - SQc) <= tol and abs(SP - SPc) <= tol),
        "t_X": None, "abs_excess_sea": abs(SQ - SP), "excess_sea_bound_holds": None, "heavy_excess_bracket": None, "heavy_excess_in_bracket": None,
        "all_bounds_hold": None, "spectrum_bounds": {key: spectrum_bound(arrays["eig_X_" + key], cstar, ledger_fn) for key in ("P", "Q", "H")},
        "tolerance_nats": float(tol),
        "label": ("K_sd in [0, 2 min(S_Q, S_d)] and S_H in [|S_Q - S_d|, S_Q + S_d] (the review's bounds; subadditivity and the triangle inequality of the 2-block Gaussian state "
                  "with C_X's spectrum); S_d <= sum_i s(f_i w_i) (Rotfel'd on the rank-1 terms) and <= min(n_d, n_X) ln 2; |S_X - S_Xc| <= sum_i s(f_i) for C and D, 0 for Q and P; "
                  "|S_Q - S_P| <= t_X (EC1); |S_H - S_P| <= t_X + S_d")}
    d["dust_orbitals"] = {"exterior_weight_min": float(wX.min()) if wX.size else None, "exterior_weight_max": float(wX.max()) if wX.size else None,
                          "exterior_weight_mean": float(wX.mean()) if wX.size else None, "n_mostly_beyond_cut": int(np.sum(wX > 0.5)),
                          "number_beyond_cut": float(np.sum(f_np * wX)), "number_inside_cut": float(np.sum(f_np * (1.0 - wX))),
                          "label": "w_i = ||Pi_X d_i||^2 per Loewdin-orthonormalized dust column; number beyond the cut = sum_i f_i w_i = Tr D_X"}
    arrays["dust_exterior_weight"] = wX
    return d, arrays


def attach_cross_cut(d, t_X, route, tol=1e-9):
    """the EC1 bound of the sea's excess and the heavy excess's bracket into the cut dict; sets all_bounds_hold."""
    b = d["bounds"]
    b["t_X"] = float(t_X)
    b["cross_cut_route"] = route
    b["excess_sea_bound_holds"] = bool(abs(d["excess_sea"]) <= t_X + tol)
    b["heavy_excess_bracket"] = [-(t_X + d["S_d"]), t_X + d["S_d"]]
    b["heavy_excess_in_bracket"] = bool(abs(d["heavy_excess"]) <= t_X + d["S_d"] + tol)
    b["all_bounds_hold"] = bool(b["K_sd_in_bounds"] and b["K_sd_c_in_bounds"] and b["S_H_in_bounds"] and b["S_d_in_bounds"] and b["purity_gaps_in_bounds"]
                                and b["excess_sea_bound_holds"] and b["heavy_excess_in_bracket"])
    return d


def local_cross_cut(S, V, mask):
    """t_X = 2 ||Pi_X (Q - P) Pi_Xc||_1 from the columns: the X-by-Xc block S_X S_Xc^dagger - V_X V_Xc^dagger and its singular values
    (page_d13_readout_v1b.cross_cut_norms' arithmetic, used only where that module is not importable); (t_X, the singular values)."""
    K = S[mask] @ S[~mask].mH
    if V.shape[1]:
        K.addmm_(V[mask], V[~mask].mH, alpha=-1.0)
    sv = torch.linalg.svdvals(K)
    del K
    return 2.0 * float(sv.sum()), sv.cpu().numpy()


def pattern_terms(S, Dc, f_d, V, mask, copies, SQ, SP, SQc=None, SPc=None, tol=1e-9):
    """W13's alternative: the channel's 16 kappa copies carry the dust column subsets `copies` (a list of index lists); the dust rows as the
    sum over the copies (equal subsets computed once), S_Q and S_P the same in every copy."""
    f_np = np.clip(np.asarray(f_d, float).reshape(-1), 0.0, 1.0)
    cnt = Counter(tuple(sorted(int(i) for i in c)) for c in copies)
    tot = {"S_H": 0.0, "S_H_c": 0.0, "S_d": 0.0, "S_d_c": 0.0, "K_sd_upper": 0.0, "S_d_straddling_bound": 0.0}
    per = []
    ok = True
    for sub, mult in sorted(cnt.items()):
        idx = list(sub)
        if any(i < 0 or i >= int(Dc.shape[1]) for i in idx):
            raise ValueError("a copy names the dust column %s outside 0..%d" % (idx, int(Dc.shape[1]) - 1))
        Dsub = Dc[:, idx] if idx else Dc[:, :0]
        fsub = f_np[idx] if idx else np.zeros(0)
        Dsc = Dsub * torch.sqrt(torch.as_tensor(fsub, dtype=RD, device=Dc.device)).to(CD)[None, :] if idx else Dsub
        bH = block_entropy(torch.cat([S, Dsc], dim=1), mask)
        bd = block_entropy(Dsc, mask)
        wX = (Dsub[mask].abs() ** 2).sum(dim=0).cpu().numpy() if idx else np.zeros(0)
        Kc = SQ + bd["S_X"] - bH["S_X"]
        inb = bool(-tol <= Kc <= 2.0 * min(SQ, bd["S_X"]) + tol)
        ok = ok and inb and bool(abs(SQ - bd["S_X"]) - tol <= bH["S_X"] <= SQ + bd["S_X"] + tol)
        per.append({"columns": idx, "multiplicity": int(mult), "S_H": bH["S_X"], "S_H_c": bH["S_Xc"], "S_d": bd["S_X"], "S_d_c": bd["S_Xc"], "K_sd": Kc,
                    "K_sd_upper": 2.0 * min(SQ, bd["S_X"]), "K_sd_in_bounds": inb, "S_d_straddling_bound": float(np.sum(h_np(fsub * wX)))})
        tot["S_H"] += mult * bH["S_X"]
        tot["S_H_c"] += mult * bH["S_Xc"]
        tot["S_d"] += mult * bd["S_X"]
        tot["S_d_c"] += mult * bd["S_Xc"]
        tot["K_sd_upper"] += mult * 2.0 * min(SQ, bd["S_X"])
        tot["S_d_straddling_bound"] += mult * per[-1]["S_d_straddling_bound"]
    n_c = len(copies)
    out = dict(tot, n_copies=n_c, n_distinct_subsets=len(cnt), S_Q=n_c * SQ, S_P=n_c * SP, K_sd=n_c * SQ + tot["S_d"] - tot["S_H"], excess_sea=n_c * (SQ - SP),
               heavy_excess=tot["S_H"] - n_c * SP, all_bounds_hold=ok, per_subset=per,
               label="the sum over the copies of channel kappa of S_X(Q + D_c) and S_X(D_c), D_c the copy's dust columns; S_Q and S_P counted once per copy")
    if SQc is not None and SPc is not None:
        out["K_sd_c"] = n_c * SQc + tot["S_d_c"] - tot["S_H_c"]
        out["heavy_excess_c"] = tot["S_H_c"] - n_c * SPc
    return out


# ---------------------------------------------------------------------------------------------------------------- the channel
def loewdin_device(Phi, n_s):
    """X (X^dagger X)^(-1/2) on Phi's device (the step of engine_massive_channel_D_v2.loewdin, which runs on the CPU); returns
    (Phi orthonormalized, the stored defect max|G - 1|, the defect after, the stored sea-dust overlap max)."""
    G = Phi.mH @ Phi
    G = 0.5 * (G + G.mH)
    Gd = G.clone()
    Gd.diagonal().sub_(1.0)
    e0 = float(Gd.abs().max())
    del Gd
    ovl = float(G[:n_s, n_s:].abs().max()) if G.shape[0] > n_s else 0.0
    wv, U = torch.linalg.eigh(G)
    del G
    Phi = Phi @ ((U * (1.0 / torch.sqrt(wv.clamp_min(1e-300))).to(CD)[None, :]) @ U.mH)
    del U
    G1 = Phi.mH @ Phi
    G1.diagonal().sub_(1.0)
    e1 = float(G1.abs().max())
    del G1
    return Phi, e0, e1, ovl


def load_channel(rd, kap, dev, loewdin_dev, loewdin, log):
    """the channel's sea and dust columns on the device, Loewdin-orthonormalized together (the sea and the dust then exactly orthogonal),
    with the stored occupations, the multiplicity, the stored e_k and mask_k, and the orthonormality defects."""
    n_s, n_d = rd.n_sea[kap], rd.n_dust[kap]
    out = {"kappa": int(kap), "weight": float(rd.weights[int(kap)]), "n_sea": int(n_s), "n_dust": int(n_d), "complex64": bool(rd.complex64)}
    use_gpu = (loewdin_dev == "gpu") or (loewdin_dev == "auto" and dev.type == "cuda")
    if loewdin == "on" and not use_gpu:
        ch = rd.channel(kap, orthonormalize=True)
        Phi = torch.as_tensor(np.asarray(ch["Phi"]), dtype=CD, device=dev)
        f_d = np.asarray(ch["f_dust"], float).reshape(-1)
        out.update({"stored_orthonormality": ch["stored_orthonormality"], "loewdin_orthonormality": ch["loewdin_orthonormality"], "orthonormalization": ch["orthonormalization"],
                    "dust_sea_overlap_max_stored": ch["dust_sea_overlap_max"]})
        e_stored, mask = ch["e"], ch["mask"]
        del ch
    else:
        sea = rd.sea(kap)
        dust, f_d = rd.dust(kap)
        Phi = torch.empty((rd.n2, n_s + n_d), dtype=CD, device=dev)
        Phi[:, :n_s] = torch.as_tensor(sea, dtype=CD, device=dev)
        del sea
        if n_d:
            Phi[:, n_s:] = torch.as_tensor(dust, dtype=CD, device=dev)
        del dust
        if loewdin == "on":
            Phi, e0, e1, ovl = loewdin_device(Phi, n_s)
            out.update({"stored_orthonormality": e0, "loewdin_orthonormality": e1, "dust_sea_overlap_max_stored": ovl,
                        "orthonormalization": "Loewdin X (X^dagger X)^(-1/2) on %s (the step of engine_massive_channel_D_v2.loewdin, which runs on the CPU)" % dev})
        else:
            G = Phi.mH @ Phi
            ovl = float(G[:n_s, n_s:].abs().max()) if n_d else 0.0
            G.diagonal().sub_(1.0)
            out.update({"stored_orthonormality": float(G.abs().max()), "loewdin_orthonormality": None, "dust_sea_overlap_max_stored": ovl, "orthonormalization": "none (the stored columns; --loewdin off)"})
            del G
        e_stored, mask = rd.e(kap), rd.mask(kap)
        f_d = np.asarray(f_d, float).reshape(-1)
    out["S"] = Phi[:, :n_s]
    out["D"] = Phi[:, n_s:]
    out["f_dust"] = f_d
    out["e_stored"] = e_stored
    out["mask"] = mask
    out["sea_dust_cross_after"] = float((out["S"].mH @ out["D"]).abs().max()) if n_d else 0.0
    return out


def killing_vacuum(H, dev, e_stored=None, ops=None):
    """P_-(h_m) = 1(h_m < 0) by one dense eigendecomposition (page_ops_v6.eigh_csr: torch.linalg.eigh of the dense h_m on the device, the
    arithmetic of page_d13_readout_v1b.killing_vacuum's dense route), the full eigenbasis returned for the mask: (info, Vneg, Vall)."""
    ops = ops or IO.OPS
    t0 = time.time()
    e, Vall = ops.eigh_csr(H, dev)
    e_np = np.asarray(e.cpu().numpy(), float)
    neg = e < 0
    V = Vall[:, neg].clone()
    info = {"n_neg": int(neg.sum()), "n_pos": int((e > 0).sum()), "n_zero_1e-12": int(np.sum(np.abs(e_np) < 1e-12)), "E_ref": float(e_np[e_np < 0].sum()),
            "e_min": float(e_np.min()), "e_max": float(e_np.max()), "seconds": time.time() - t0,
            "route": "dense (page_ops_v6.eigh_csr: torch.linalg.eigh of h_m on %s; page_d13_readout_v1b.killing_vacuum's dense route, the full eigenbasis kept for the mask)" % dev,
            "eig_vs_stored": None}
    if e_stored is not None:
        try:
            info["eig_vs_stored"] = float(np.max(np.abs(np.sort(e_np) - np.sort(np.asarray(e_stored, float).reshape(-1)))))
        except ValueError:
            info["eig_vs_stored"] = None
    info["e"] = e_np
    return info, V, Vall


def find_N_B(ctx, a):
    if getattr(a, "N_B", None) is not None:
        return float(a.N_B), "--N-B"
    args = getattr(ctx, "args", None) or {}
    pat = re.compile(r"^(n_?b|nb|n_?dust|n_?part(icles)?|npart)$", re.IGNORECASE)
    for k, v in args.items():
        if pat.match(str(k)) and isinstance(v, (int, float)) and not isinstance(v, bool):
            return float(v), "the records json's argument %s" % k
    return 640.0, "requirements_new.txt: N_B 640 at (0.2, 64) (no dust-count argument named in the records json)"


def load_pattern(path):
    with open(path) as f:
        P = json.load(f)
    out = {}
    for k, v in P.items():
        kk = int(str(k).replace("kappa", "").replace("_", "").strip())
        out[kk] = v
    return out


def w13_assumptions(rd, N_B, nb_src):
    return {
        "status": "W13 is answered separately; these are the readout's assumptions with their origin in the inputs; W13's STATUS lines supersede them",
        "chain_dimension_per_channel": {"two_component_sites": int(rd.nsites), "dimension": int(rd.n2),
                                        "origin": "the slice's R array and input 05's row convention (SliceReader: n2 = 2 nsites; rows 2j and 2j + 1 belong to site j at R[j])"},
        "sea_columns": {"per_channel": {str(k): int(rd.n_sea[k]) for k in rd.kappas}, "f": "1 on every sea column",
                        "origin": "the slice's sea_k member (SliceReader.n_sea = sea_k.shape[1]); SliceReader.channel sets f = [ones(n_sea), f_dust]"},
        "dust_columns": {"per_channel": {str(k): int(rd.n_dust[k]) for k in rd.kappas},
                         "origin": "the slice's dust_k and f_k members (SliceReader.dust); input 05's docstring: Phi_k = [sea_k | dust_k] (f all 1); the f statistics print per channel"},
        "per_copy_occupation": {"assumed": "whole multiplets: every one of the 16 kappa copies of channel kappa carries the stored dust columns with the stored f",
                                "weights": {str(k): float(rd.weights[k]) for k in rd.kappas}, "origin_of_weights": "the slice's weights member (16 kappa; input 05)",
                                "check": "the weighted dust count sum_kappa 16 kappa sum_i f_i against N_B %g (%s) beside the unweighted count, in dust_count_check" % (N_B, nb_src),
                                "alternative": "--copy-pattern FILE with W13's per-copy column subsets recomputes the dust rows as the sum over the copies (weighted_pattern)"},
        "g_n_guard": {"assumed": ("the sources' rule, no degeneracy weight on any source in the radial codes (requirements_new.txt section 6), which this readout does not touch: "
                                  "it weights entropies and never sources; whether a stored dust column sources once or 16 kappa times is W13's and the same question as the count check")},
    }


# ----------------------------------------------------------------------------------------------------------------- the page json
def page_comparison(a, entry, rd, chans, labels):
    """page_own_object_v1's values for the same slice where its json is present: S_X (= S_H where every f is 1), S_X_vac (= S_P), the dust's N_out."""
    d = os.path.join(a.out, "page") if a.page_dir == "auto" else a.page_dir
    p = os.path.join(d, "page_%s_T%07.2f.json" % (entry["history"], rd.T_over_rs))
    if not os.path.isfile(p):
        return {"status": "no page_own_object_v1 json at %s" % p, "max_abs_dS_H": None, "max_abs_dS_P": None}
    try:
        with open(p) as f:
            J = json.load(f)
    except Exception as ex:  # noqa: BLE001
        return {"status": "page json unreadable (%s)" % ex, "max_abs_dS_H": None, "max_abs_dS_P": None}
    pc = {int(c["kappa"]): c for c in (J.get("channels") or [])}
    out = {"file": p, "per_channel": {}, "max_abs_dS_H": 0.0, "max_abs_dS_P": 0.0, "max_abs_dN_dust": 0.0, "n_compared": 0,
           "label": "dS_H = S_H - page's S_X (page treats every dust occupation as 1, so they agree where every f is 1); dS_P = S_P - page's S_X_vac; dN = the dust number beyond the cut minus page's N_out"}
    for c in chans:
        q = pc.get(int(c["kappa"]))
        if not q:
            continue
        for lab in labels:
            cu = (q.get("cuts") or {}).get(lab)
            if not cu or cu.get("S_X") is None:
                continue
            mine = c["cuts"][lab]
            rec = {"S_X_page": cu["S_X"], "S_H": mine["S_H"], "dS_H": mine["S_H"] - float(cu["S_X"]), "S_X_vac_page": cu.get("S_X_vac"), "S_P": mine["S_P"],
                   "dS_P": (mine["S_P"] - float(cu["S_X_vac"])) if cu.get("S_X_vac") is not None else None}
            du = cu.get("dust") or {}
            if du.get("N_out") is not None:
                rec["N_out_page"] = du["N_out"]
                rec["dust_number_beyond_cut"] = mine["dust_orbitals"]["number_beyond_cut"]
                rec["dN_dust"] = mine["dust_orbitals"]["number_beyond_cut"] - float(du["N_out"])
                out["max_abs_dN_dust"] = max(out["max_abs_dN_dust"], abs(rec["dN_dust"]))
            out["per_channel"]["k%d_cut%s" % (c["kappa"], lab)] = rec
            out["max_abs_dS_H"] = max(out["max_abs_dS_H"], abs(rec["dS_H"]))
            if rec["dS_P"] is not None:
                out["max_abs_dS_P"] = max(out["max_abs_dS_P"], abs(rec["dS_P"]))
            out["n_compared"] += 1
    out["status"] = "compared %d channel-cut pairs" % out["n_compared"] if out["n_compared"] else "the page json carries no matching channel or cut"
    return out


def partial_sums(chans, lab):
    keys = ("S_H", "S_P", "S_Q", "S_d", "K_sd", "excess_sea", "heavy_excess")
    acc = {k: 0.0 for k in keys}
    out = []
    for c in sorted(chans, key=lambda c_: c_["kappa"]):
        d = c["cuts"][lab]
        w = c["weight"]
        last = {k: w * float(d[k]) for k in keys}
        for k in keys:
            acc[k] += last[k]
        e = {"K": int(c["kappa"]), "weight": w}
        e.update({"sum_" + k: acc[k] for k in keys})
        e.update({"last_" + k: last[k] for k in keys})
        e["last_share_heavy_excess"] = (abs(last["heavy_excess"]) / abs(acc["heavy_excess"])) if acc["heavy_excess"] != 0.0 else None
        out.append(e)
    return out


# ---------------------------------------------------------------------------------------------------------------- the driver's hooks
def add_arguments(ap):
    g = ap.add_argument_group("heavy_cut_entropy_v1 (W8)")
    g.add_argument("--slices-code", dest="slices_code", default=None, help="the folder of slices144_io_v1.py (input 05) where it is not this one and not $SLICES144_CODE")
    g.add_argument("--device", default="cuda:0", help="the torch device of the dense eigensolves and the Gram spectra (cuda:0 under CUDA_VISIBLE_DEVICES set to the A6000; cpu otherwise)")
    g.add_argument("--loewdin-device", dest="loewdin_device", default="auto", choices=["auto", "gpu", "cpu"],
                   help="where the Loewdin step runs: gpu (X (X^dagger X)^(-1/2) on --device) or cpu (engine_massive_channel_D_v2.loewdin through SliceReader.channel); auto: gpu on a cuda device")
    g.add_argument("--cut-block-route", dest="cut_block_route", default="svd", choices=["svd", "gram"], help="the cross-cut block's singular values (page_d13_readout_v1b.cross_cut_norms)")
    g.add_argument("--cut-norm-full", dest="cut_norm_full", default="gram", choices=["gram", "svd", "bound", "off"], help="the full ||Q - P||_1 and n_pair per channel (page_d13_readout_v1b.full_trace_norm)")
    g.add_argument("--copy-pattern", dest="copy_pattern", default=None, help="W13's per-copy dust occupation: a json {kappa: 'whole' | {'copies': [[dust column indices], ... (16 kappa lists)]}}")
    g.add_argument("--mask-reference", dest="mask_reference", default="on", choices=["on", "off"],
                   help="S_mask: the cut entropy of the projector the slice's mask_k selects in the ascending eigenbasis of h_m (a diagnostic of the continued reference, not a term)")
    g.add_argument("--N-B", dest="N_B", type=float, default=None, help="the dust count for the count check (default: the records json's argument where one is named, else 640)")
    g.add_argument("--page-dir", dest="page_dir", default="auto", help="page_own_object_v1's outputs for the same slices (auto: <out>/page)")
    g.add_argument("--save-spectra", dest="save_spectra", default="on", choices=["on", "off"], help="the per-cut spectra and singular values into the npz")
    g.add_argument("--bound-tol", dest="bound_tol", type=float, default=1e-9, help="the rounding allowance of every bound check (nats)")


def is_done(entry, a):
    return os.path.exists(IO.out_paths(a, READOUT, entry)[0])


WEIGHTED_TERMS = ("S_H", "S_P", "S_Q", "S_d", "K_sd", "excess_sea", "heavy_excess")
WEIGHTED_BOUNDS = ("t_X", "K_sd_upper", "S_H_lower", "S_H_upper", "S_d_straddling_bound", "S_d_count_bound", "dust_mixedness")


def process_slice(rd, ctx, entry, a):
    t0 = time.time()
    log = ctx.log
    OPS, D13 = IO.OPS, IO.D13
    dev = pick_device(a.device, log)
    pj, pz = IO.out_paths(a, READOUT, entry)
    rec = ctx.record_at(entry, rd.T)
    m, g, rs = ctx.m, ctx.g, rd.rs
    R, N_app, v_app = rd.R, rd.N_applied, rd.v_applied
    if N_app is None or v_app is None:
        raise RuntimeError("the slice %s carries no applied fields (N_applied, v_applied): the held Killing vacuum needs them" % rd.path)
    cuts = [(lab, x * rs) for lab, x in IO.cuts_from_arg(a.cuts_rs)]
    labels = [lab for lab, _ in cuts]
    tol = float(a.bound_tol)
    N_B, nb_src = find_N_B(ctx, a)
    row = IO.slice_header(rd, ctx, entry, rec, READOUT, TITLE, CODE)
    row.update({"print_status": PRINT_STATUS, "d15": D15, "device": str(dev), "definitions": DEFINITIONS, "assumptions_W13": w13_assumptions(rd, N_B, nb_src),
                "cuts_rs": [float(x) for x in a.cuts_rs.split(",")]})
    E_rec = rec.get("E")
    T_H = 1.0 / (8.0 * math.pi * g * float(E_rec)) if E_rec else None
    row["record"] = {"E": E_rec, "two_gE": 2.0 * g * float(E_rec) if E_rec else None, "T_H": T_H, "m_over_TH": (m / T_H) if T_H else None, "R_t": rec.get("R_t"),
                     "trapping_zeros": rec.get("trapping_zeros"), "min_N_minus_absv": rec.get("min_N_minus_absv"), "E_over_E0": rec.get("E_over_E0")}
    form = ctx.formation(entry)
    row["formation"] = form
    row["first_return"] = {}
    for lab, x in cuts:
        if D13 is not None and hasattr(D13, "first_return"):
            row["first_return"][lab] = D13.first_return(N_app, v_app, R, x, rd.T, 0.0, rs, form["T"] if form else None, form["r_root"] if form else None)
        else:
            row["first_return"][lab] = {"first_return_rs": None, "before_first_return": None, "note": "page_d13_readout_v1b not importable: no first-return estimate"}
    pattern = load_pattern(a.copy_pattern) if a.copy_pattern else None
    ledger_fn = getattr(OPS, "pair_ledger_sharp", None)
    cstar = float(getattr(OPS, "C_STAR", C_STAR_DEFAULT))
    R2_t = torch.as_tensor(np.repeat(np.asarray(R, float), 2), dtype=RD, device=dev)
    masks = {lab: (R2_t > x) for lab, x in cuts}
    arrays = {"R": np.asarray(R, float), "N_applied": np.asarray(N_app, float), "v_applied": np.asarray(v_app, float)}
    chans = []
    wsum = {lab: {} for lab in labels}
    wpat = {lab: {} for lab in labels}
    count_w = 0.0
    count_u = 0.0
    pattern_used = False
    for kap in rd.kappas:
        t1 = time.time()
        chn = load_channel(rd, kap, dev, a.loewdin_device, a.loewdin, log)
        S, Dc, f_d, w = chn["S"], chn["D"], chn["f_dust"], chn["weight"]
        n_s, n_d = chn["n_sea"], chn["n_dust"]
        count_w += w * float(np.sum(f_d))
        count_u += float(np.sum(f_d))
        h0, Q, H = IO.operators(R, m, kap, N_app, v_app)
        vac, V, Vall = killing_vacuum(H, dev, chn["e_stored"], OPS)
        e_np = vac.pop("e")
        mask_info = None
        V_mask = None
        if chn["mask"] is not None:
            mb = np.asarray(chn["mask"], bool).reshape(-1)
            pop = int(mb.sum())
            mask_info = {"popcount": pop, "equals_n_sea": bool(pop == n_s), "equals_n_neg": bool(pop == vac["n_neg"]), "bits_differing_from_the_negative_set": int(np.sum(mb != (e_np < 0))),
                         "S_mask_computed": False,
                         "label": "mask_k read as the engine's occupied set over the eigenvectors of h_m in ascending order (the continued reference's candidate); the overlap with the physical sea is the check"}
            if a.mask_reference == "on" and pop > 0 and mb.size == int(Vall.shape[1]):
                V_mask = Vall[:, torch.as_tensor(mb, device=dev)].clone()
                ov = float(torch.linalg.norm(S.mH @ V_mask)) ** 2
                mask_info.update({"S_mask_computed": True, "tr_Q_Pmask": ov, "n_sea_minus_tr_Q_Pmask": n_s - ov})
        del Vall
        trQP = float(torch.linalg.norm(S.mH @ V)) ** 2
        overlap = {"tr_Q_Pminus": trQP, "n_pair": n_s - trQP, "n_hole": vac["n_neg"] - trQP,
                   "label": "Tr[Q P_-] = ||Phi_s^dagger V||_F^2; n_pair = Tr[(1 - P_-) Q] the sea's content outside the held vacuum, n_hole = Tr[P_- (1 - Q)]"}
        full = None
        if D13 is not None and hasattr(D13, "full_trace_norm") and a.cut_norm_full != "off":
            try:
                full = D13.full_trace_norm(S, V, a.cut_norm_full, 24.0, 1e-10, 1024)
                full.pop("sv", None)
            except Exception as ex:  # noqa: BLE001
                full = {"error": "%s: %s" % (type(ex).__name__, ex)}
        cc = None
        if D13 is not None and hasattr(D13, "cross_cut_norms"):
            try:
                cc = D13.cross_cut_norms(S, V, R2_t, cuts, a.cut_block_route)
            except Exception as ex:  # noqa: BLE001
                log("  kappa %d: page_d13_readout_v1b.cross_cut_norms failed (%s: %s): the local block route" % (kap, type(ex).__name__, ex))
                cc = None
        out = {k_: v_ for k_, v_ in chn.items() if k_ not in ("S", "D", "f_dust", "e_stored", "mask")}
        out.update({"f_dust": {"n": int(f_d.size), "min": float(f_d.min()) if f_d.size else None, "max": float(f_d.max()) if f_d.size else None, "sum": float(f_d.sum()),
                               "n_fractional": int(np.sum((f_d > 1e-12) & (f_d < 1.0 - 1e-12))), "all_one_within_1e-12": bool(np.all(np.abs(f_d - 1.0) < 1e-12)) if f_d.size else True},
                    "vacuum": vac, "mask": mask_info, "overlap_sea_vacuum": overlap, "full_trace_norm": full, "cuts": {}})
        pat = pattern.get(int(kap)) if pattern else None
        for lab, x in cuts:
            d, arr = cut_terms(S, Dc, f_d, V, masks[lab], lab, x, cstar, ledger_fn, tol)
            if cc is not None and lab in cc:
                c = cc[lab]
                arr["sv_block"] = c.pop("sv_block")
                attach_cross_cut(d, c["t_X"], "page_d13_readout_v1b.cross_cut_norms (%s)" % c.get("block_route"), tol)
                d["bounds"]["cross_cut_detail"] = {k_: v_ for k_, v_ in c.items() if k_ != "label"}
            else:
                tX, sv = local_cross_cut(S, V, masks[lab])
                arr["sv_block"] = sv
                attach_cross_cut(d, tX, "local: 2 x the sum of the singular values of the X-by-Xc block of Q - P (page_d13_readout_v1b not importable)", tol)
            if V_mask is not None:
                bm = block_entropy(V_mask, masks[lab])
                d["mask_reference"] = {"S_mask": bm["S_X"], "S_mask_c": bm["S_Xc"], "excess_sea_over_mask": d["S_Q"] - bm["S_X"], "heavy_excess_over_mask": d["S_H"] - bm["S_X"],
                                       "label": DEFINITIONS["S_mask"]}
                arr["eig_X_mask"] = bm["ev_X"].cpu().numpy()
            if isinstance(pat, dict) and pat.get("copies"):
                d["pattern"] = pattern_terms(S, Dc, f_d, V, masks[lab], pat["copies"], d["S_Q"], d["S_P"], d["S_Q_c"], d["S_P_c"], tol)
                d["pattern"]["n_copies_expected"] = int(round(w))
                d["pattern"]["n_copies_matches_16_kappa"] = bool(d["pattern"]["n_copies"] == int(round(w)))
                pattern_used = True
            out["cuts"][lab] = d
            if a.save_spectra == "on":
                for key, val in arr.items():
                    arrays["k%d_cut%s_%s" % (kap, lab, key)] = val
            ws = wsum[lab]
            for key in WEIGHTED_TERMS:
                ws[key] = ws.get(key, 0.0) + w * float(d[key])
            for key in WEIGHTED_BOUNDS:
                val = d["bounds"].get(key)
                if val is not None:
                    ws[key] = ws.get(key, 0.0) + w * float(val)
            ws["heavy_excess_bracket_halfwidth"] = ws.get("heavy_excess_bracket_halfwidth", 0.0) + w * (float(d["bounds"]["t_X"]) + float(d["S_d"]))
            ws["dust_number_beyond_cut"] = ws.get("dust_number_beyond_cut", 0.0) + w * float(d["dust_orbitals"]["number_beyond_cut"])
            ws["all_bounds_hold"] = bool(ws.get("all_bounds_hold", True) and d["bounds"]["all_bounds_hold"])
            for gk in ("purity_gap_H", "purity_gap_Q", "purity_gap_P", "purity_gap_d"):
                ws["max_abs_" + gk] = max(ws.get("max_abs_" + gk, 0.0), abs(float(d["bounds"][gk])))
            ws["max_abs_decomposition_residual"] = max(ws.get("max_abs_decomposition_residual", 0.0), abs(float(d["decomposition_residual"])))
            ws["max_excursion"] = max(ws.get("max_excursion", 0.0), max(float(d["excursion_" + k_]) for k_ in ("H", "Q", "d", "P")))
            if d.get("mask_reference"):
                ws["S_mask"] = ws.get("S_mask", 0.0) + w * float(d["mask_reference"]["S_mask"])
                ws["heavy_excess_over_mask"] = ws.get("heavy_excess_over_mask", 0.0) + w * float(d["mask_reference"]["heavy_excess_over_mask"])
            wp = wpat[lab]
            if d.get("pattern"):
                p_ = d["pattern"]
                for key in WEIGHTED_TERMS:
                    wp[key] = wp.get(key, 0.0) + float(p_[key])
                wp["K_sd_upper"] = wp.get("K_sd_upper", 0.0) + float(p_["K_sd_upper"])
                wp["S_d_straddling_bound"] = wp.get("S_d_straddling_bound", 0.0) + float(p_["S_d_straddling_bound"])
                wp["all_bounds_hold"] = bool(wp.get("all_bounds_hold", True) and p_["all_bounds_hold"])
                wp["channels_with_pattern"] = wp.get("channels_with_pattern", []) + [int(kap)]
            else:
                for key in WEIGHTED_TERMS:
                    wp[key] = wp.get(key, 0.0) + w * float(d[key])
                wp["K_sd_upper"] = wp.get("K_sd_upper", 0.0) + w * float(d["bounds"]["K_sd_upper"])
                wp["S_d_straddling_bound"] = wp.get("S_d_straddling_bound", 0.0) + w * float(d["bounds"]["S_d_straddling_bound"])
                wp["all_bounds_hold"] = bool(wp.get("all_bounds_hold", True) and d["bounds"]["all_bounds_hold"])
        out["seconds"] = time.time() - t1
        chans.append(out)
        c0 = out["cuts"][labels[0]]
        log("  kappa %d (w %g): sea %d, dust %d (f sum %.3f, fractional %d), vacuum %d occupied (eig vs stored %s, %.1f s), n_pair %.3e; beyond %s r_s: S_H %.6f = S_P %.6f + (S_Q - S_P) %+.6f + S_d %.6f - K_sd %.6f "
            "(residual %.1e; heavy excess %+.6f; K_sd upper %.6f; t_X %.4f; S_d straddling bound %.6f; gaps H %.1e Q %.1e P %.1e; bounds %s); %s%.0f s" % (
                kap, w, n_s, n_d, out["f_dust"]["sum"], out["f_dust"]["n_fractional"], vac["n_neg"], IO.fx(vac["eig_vs_stored"], "%.1e"), vac["seconds"], overlap["n_pair"], labels[0],
                c0["S_H"], c0["S_P"], c0["excess_sea"], c0["S_d"], c0["K_sd"], c0["decomposition_residual"], c0["heavy_excess"], c0["bounds"]["K_sd_upper"], c0["bounds"]["t_X"],
                c0["bounds"]["S_d_straddling_bound"], c0["bounds"]["purity_gap_H"], c0["bounds"]["purity_gap_Q"], c0["bounds"]["purity_gap_P"],
                "hold" if c0["bounds"]["all_bounds_hold"] else "FAIL", ("S_mask %.6f; " % c0["mask_reference"]["S_mask"]) if c0.get("mask_reference") else "", out["seconds"]))
        del S, Dc, V, V_mask, chn
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    row["channels"] = chans
    row["weighted"] = {lab: dict(wsum[lab], weights="16 kappa (the slice's weights; whole multiplets assumed, W13 pending)", channels=list(rd.kappas)) for lab in labels}
    if pattern_used:
        row["weighted_pattern"] = {lab: dict(wpat[lab], weights="the sum over the copies named by --copy-pattern where given, 16 kappa times the whole terms elsewhere", pattern_file=a.copy_pattern) for lab in labels}
    whole_ok = abs(count_w - N_B) < 1e-6 * max(1.0, N_B)
    stored_ok = abs(count_u - N_B) < 1e-6 * max(1.0, N_B)
    row["dust_count_check"] = {"N_B": N_B, "N_B_source": nb_src, "weighted_dust_count": count_w, "unweighted_dust_count": count_u, "whole_multiplets_consistent": bool(whole_ok),
                               "stored_columns_are_the_whole_set": bool(stored_ok),
                               "verdict": ("the weighted count equals N_B: the whole-multiplet assumption is consistent with the count (W13 confirms the pattern)" if whole_ok else
                                           ("the unweighted count equals N_B: the stored dust columns are the whole dust and the 16 kappa weight does not apply to the dust part; the weighted dust rows are to be recomputed with W13's pattern (--copy-pattern)" if stored_ok else
                                            "neither count equals N_B: the per-copy pattern is W13's to state; the weighted dust rows are provisional"))}
    row["partial_sums_in_kappa"] = {lab: partial_sums(chans, lab) for lab in labels}
    row["page_comparison"] = page_comparison(a, entry, rd, chans, labels)
    row["seconds"] = time.time() - t0
    IO.save_outputs(pj, pz, row, arrays)
    parts = []
    for lab in labels:
        w_ = row["weighted"][lab]
        parts.append("beyond %s r_s S_H %.6f = S_P %.6f + (S_Q - S_P) %+.6f + S_d %.6f - K_sd %.6f (heavy excess %+.6f, bracket %.4f, bounds %s)" % (
            lab, w_["S_H"], w_["S_P"], w_["excess_sea"], w_["S_d"], w_["K_sd"], w_["heavy_excess"], w_["heavy_excess_bracket_halfwidth"], "hold" if w_["all_bounds_hold"] else "FAIL"))
    log("%s [not for print] T/r_s %.3f (%s; %s; %s): %s; dust count weighted %.2f, unweighted %.2f against N_B %g (%s); page comparison: %s (max |dS_H| %s, |dS_P| %s); first return %s r_s; %.0f s" % (
        TITLE, rd.T_over_rs, rd.dtype_str, entry["history"], dev, "; ".join(parts), count_w, count_u, N_B,
        "whole multiplets consistent" if whole_ok else ("stored columns are the whole set" if stored_ok else "neither"), row["page_comparison"]["status"],
        IO.fx(row["page_comparison"].get("max_abs_dS_H"), "%.1e"), IO.fx(row["page_comparison"].get("max_abs_dS_P"), "%.1e"), IO.fx(row["first_return"][labels[0]].get("first_return_rs"), "%.1f"), row["seconds"]))
    return row


# ------------------------------------------------------------------------------------------------------------------ the tables
def compose(a, hists):
    rows = IO.load_rows(a, READOUT, hists)
    if not rows:
        return
    fx = IO.fx
    labels = sorted({lab for r in rows for lab in (r.get("weighted") or {})}, key=float)
    L = ["# %s: the heavy field's exterior-cut entropy from the whole heavy covariance on the stored slices (%s) -- NOT FOR PRINT" % (TITLE, CODE), "",
         "W8 (REVIEWA_REVIEW_web40i section 4; requirements edit 25). %s. %s." % (PRINT_STATUS, D15), "",
         "- S_H: Tr s(C_X) of the whole heavy covariance C = Q + D of the channel (the Loewdin-orthonormalized sea columns, the dust columns with their stored f) beyond the cut X = {R > x}, by the Gram route; per channel (unweighted) and summed with the slice's weights 16 kappa.",
         "- S_P: the held Killing vacuum P = 1(h_m < 0) at the slice's applied fields, its entropy at the same cut: the baseline of the excess (edits 7 to 9, Y2). S_Q: the physical sea Q alone. S_Q - S_P: the sea's excess, signed, with |S_Q - S_P| <= t_X = 2 ||Pi_X (Q - P) Pi_Xc||_1 (EC1).",
         "- S_d: the dust D alone, with S_d <= sum_i s(f_i w_i) (w_i the dust orbital's exterior weight; the straddling-orbital bound) and S_d <= min(n_d, n_X) ln 2.",
         "- K_sd = S_Q + S_d - S_H: the restriction term, 0 <= K_sd <= 2 min(S_Q, S_d); equivalently |S_Q - S_d| <= S_H <= S_Q + S_d. S_H = S_P + (S_Q - S_P) + S_d - K_sd holds by definition (the residual is the arithmetic check).",
         "- the heavy excess S_H - S_P = (S_Q - S_P) + S_d - K_sd is the Page panel's heavy curve (Y13); S_H and S_P are regulated tables (the radial truncation's raw entropy has no limit in kappa). |S_H - S_P| <= t_X + S_d.",
         "- the weighted sums assume whole multiplets (every one of the 16 kappa copies of channel kappa carries the stored dust columns with the stored f); the dust count check per slice compares sum 16 kappa sum f and sum f with N_B; W13 decides (edit 25: stated before any heavy entropy prints).",
         "- S_mask (where the slice's mask_k is present): the cut entropy of the projector the mask selects in the eigenbasis of h_m, a diagnostic of the continued reference beside the held vacuum's S_P; not a term.",
         "- page: page_own_object_v1's S_X and S_X^vac for the same slice and cut where its json exists (S_X equals S_H where every f is 1).",
         "- the first return: page_d13_readout_v1b.first_return in the slice's applied fields; a row past it reads the chain end's return.", ""]
    for hk in hists:
        rs_ = [r for r in rows if r["history"] == hk]
        if not rs_:
            continue
        r0 = rs_[0]
        asm = r0.get("assumptions_W13") or {}
        L += ["## %s: %s -- NOT FOR PRINT" % (hk, r0["history_label"]), "", "Engine: %s" % r0["engine"], "",
              "Assumptions in place of W13 on this history: the chain's dimension per channel %s (%s two-component sites); the sea's columns per channel %s; the dust's columns per channel %s; the weights %s; "
              "the dust count check per slice below (N_B %s, %s); the g_n guard read as the sources' rule, untouched here." % (
                  (asm.get("chain_dimension_per_channel") or {}).get("dimension"), (asm.get("chain_dimension_per_channel") or {}).get("two_component_sites"),
                  json.dumps((asm.get("sea_columns") or {}).get("per_channel")), json.dumps((asm.get("dust_columns") or {}).get("per_channel")),
                  json.dumps((asm.get("per_copy_occupation") or {}).get("weights")), fx((r0.get("dust_count_check") or {}).get("N_B"), "%g"), (r0.get("dust_count_check") or {}).get("N_B_source")), ""]
        for lab in labels:
            L += ["### Beyond %s r_s (summed over the channels with the multiplicity 16 kappa; whole multiplets assumed)" % lab, "",
                  "| T/r_s | dtype | E/E0 | S_H | S_P | S_H - S_P (the heavy excess) | S_Q - S_P (the sea's excess, signed) | t_X (|S_Q - S_P| <= t_X) | S_d | sum s(f w) (S_d bound) | K_sd | 2 min(S_Q, S_d) | all bounds | max purity gap H; Q; P | S_mask; S_H - S_mask | dust count w; u (N_B) | page S_X - S_H; S_X^vac - S_P (max abs) | first return (r_s) |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
            for r in rs_:
                w = (r.get("weighted") or {}).get(lab)
                if not w:
                    continue
                dc = r.get("dust_count_check") or {}
                pg = r.get("page_comparison") or {}
                fr = (r.get("first_return") or {}).get(lab) or {}
                L.append("| %.3f | %s | %s | %.6f | %.6f | %+.6f | %+.6f | %.4f (%s) | %.6f | %.6f | %.6f | %.6f | %s | %.1e; %.1e; %.1e | %s; %s | %.1f; %.1f (%s) | %s; %s | %s%s |" % (
                    r["T_over_rs"], r["dtype"], fx((r.get("record") or {}).get("E_over_E0"), "%.6f"), w["S_H"], w["S_P"], w["heavy_excess"], w["excess_sea"], w.get("t_X", 0.0),
                    "ok" if abs(w["excess_sea"]) <= w.get("t_X", 0.0) + 1e-6 else "FAIL", w["S_d"], w.get("S_d_straddling_bound", 0.0), w["K_sd"], w.get("K_sd_upper", 0.0),
                    "hold" if w.get("all_bounds_hold") else "FAIL", w.get("max_abs_purity_gap_H", 0.0), w.get("max_abs_purity_gap_Q", 0.0), w.get("max_abs_purity_gap_P", 0.0),
                    fx(w.get("S_mask"), "%.6f"), fx(w.get("heavy_excess_over_mask"), "%+.6f"), dc.get("weighted_dust_count", 0.0), dc.get("unweighted_dust_count", 0.0),
                    "w = N_B" if dc.get("whole_multiplets_consistent") else ("u = N_B" if dc.get("stored_columns_are_the_whole_set") else "neither"),
                    fx(pg.get("max_abs_dS_H"), "%.1e") if pg.get("n_compared") else "-", fx(pg.get("max_abs_dS_P"), "%.1e") if pg.get("n_compared") else "-",
                    fx(fr.get("first_return_rs"), "%.1f"), "" if fr.get("before_first_return") in (None, True) else " (past the first return)"))
            L.append("")
            if any(r.get("weighted_pattern") for r in rs_):
                L += ["#### Beyond %s r_s with the per-copy pattern of --copy-pattern (W13's alternative; the sum over the copies for the channels named, 16 kappa times the whole terms elsewhere)" % lab, "",
                      "| T/r_s | S_H | S_P | S_H - S_P | S_Q - S_P | S_d | sum s(f w) | K_sd | K_sd upper | bounds | channels with a pattern |", "|---|---|---|---|---|---|---|---|---|---|---|"]
                for r in rs_:
                    w = (r.get("weighted_pattern") or {}).get(lab)
                    if not w:
                        continue
                    L.append("| %.3f | %.6f | %.6f | %+.6f | %+.6f | %.6f | %.6f | %.6f | %.6f | %s | %s |" % (
                        r["T_over_rs"], w["S_H"], w["S_P"], w["heavy_excess"], w["excess_sea"], w["S_d"], w.get("S_d_straddling_bound", 0.0), w["K_sd"], w.get("K_sd_upper", 0.0),
                        "hold" if w.get("all_bounds_hold") else "FAIL", w.get("channels_with_pattern")))
                L.append("")
            L += ["### Per channel beyond %s r_s (unweighted; the multiplicity 16 kappa applies to the sums above)" % lab, "",
                  "| T/r_s | kappa | n_sea; n_dust; n_neg (eig vs stored) | S_H | S_P | S_Q | S_d | K_sd [0, 2 min(S_Q, S_d)] | S_Q - S_P (t_X) | S_H - S_P | S_H in [|S_Q - S_d|, S_Q + S_d] | S_d bound sum s(f w); count | gaps H; Q; P; d | n_pair | dust beyond the cut | S_mask | bounds |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
            for r in rs_:
                for c in r["channels"]:
                    d = c["cuts"].get(lab)
                    if not d:
                        continue
                    b = d["bounds"]
                    L.append("| %.3f | %d | %d; %d; %d (%s) | %.6f | %.6f | %.6f | %.6f | %.6f [0, %.6f] | %+.6f (%.4f) | %+.6f | [%.6f, %.6f] | %.6f; %.3f | %.1e; %.1e; %.1e; %.1e | %.3e | %.4e | %s | %s |" % (
                        r["T_over_rs"], c["kappa"], c["n_sea"], c["n_dust"], c["vacuum"]["n_neg"], fx(c["vacuum"].get("eig_vs_stored"), "%.1e"), d["S_H"], d["S_P"], d["S_Q"], d["S_d"], d["K_sd"], b["K_sd_upper"],
                        d["excess_sea"], b["t_X"], d["heavy_excess"], b["S_H_lower"], b["S_H_upper"], b["S_d_straddling_bound"], b["S_d_count_bound"], b["purity_gap_H"], b["purity_gap_Q"], b["purity_gap_P"],
                        b["purity_gap_d"], (c.get("overlap_sea_vacuum") or {}).get("n_pair", 0.0), d["dust_orbitals"]["number_beyond_cut"], fx((d.get("mask_reference") or {}).get("S_mask"), "%.6f"),
                        "hold" if b["all_bounds_hold"] else "FAIL"))
            L.append("")
            L += ["### The partial sums in kappa beyond %s r_s (weights 16 kappa; the raw S_H and S_P grow with the channel set, the heavy excess is the readout; the last channel's share is printed, no convergence is claimed from it)" % lab, "",
                  "| T/r_s | K | sum S_H | sum S_P | sum (S_H - S_P) | sum (S_Q - S_P) | sum S_d | sum K_sd | last channel's share of the heavy excess |", "|---|---|---|---|---|---|---|---|---|"]
            for r in rs_:
                for e in (r.get("partial_sums_in_kappa") or {}).get(lab) or []:
                    L.append("| %.3f | %d | %.6f | %.6f | %+.6f | %+.6f | %.6f | %.6f | %s |" % (r["T_over_rs"], e["K"], e["sum_S_H"], e["sum_S_P"], e["sum_heavy_excess"], e["sum_excess_sea"], e["sum_S_d"], e["sum_K_sd"],
                                                                                           fx(e.get("last_share_heavy_excess"), "%.4f")))
            L.append("")
        L += ["### Checks", ""]
        for r in rs_:
            cs = r["channels"]
            dc = r.get("dust_count_check") or {}
            pg = r.get("page_comparison") or {}
            fstat = [c["f_dust"] for c in cs]
            L.append("- T/r_s %.3f: %s; Loewdin defects after %s (stored %s; sea-dust cross after %s); vacuum eigenvalues against the stored e_k max %s; the vacuum's occupied counts %s against the sea's columns %s; "
                     "f: all 1 on every channel %s (fractional %d); mask popcounts %s (against n_sea %s); dust count weighted %.2f, unweighted %.2f against N_B %g: %s; every bound holds on every channel and cut: %s; "
                     "max |decomposition residual| %.1e; max eigenvalue excursion %.1e; page comparison: %s%s; %s; device %s; %.0f s." % (
                         r["T_over_rs"], r.get("print_status", PRINT_STATUS).split(";")[0], fx(max((c.get("loewdin_orthonormality") or 0.0) for c in cs), "%.1e"), fx(max(c["stored_orthonormality"] for c in cs), "%.1e"),
                         fx(max(c.get("sea_dust_cross_after", 0.0) for c in cs), "%.1e"), fx(max([c["vacuum"]["eig_vs_stored"] for c in cs if c["vacuum"].get("eig_vs_stored") is not None] or [float("nan")]), "%.1e"),
                         sorted({c["vacuum"]["n_neg"] for c in cs}), sorted({c["n_sea"] for c in cs}), all(f_["all_one_within_1e-12"] for f_ in fstat), sum(f_["n_fractional"] for f_ in fstat),
                         sorted({(c.get("mask") or {}).get("popcount") for c in cs if c.get("mask")}) or "absent", sorted({c["n_sea"] for c in cs}), dc.get("weighted_dust_count", 0.0), dc.get("unweighted_dust_count", 0.0),
                         dc.get("N_B", 0.0), dc.get("verdict"), all(d["bounds"]["all_bounds_hold"] for c in cs for d in c["cuts"].values()),
                         max(r["weighted"][l_].get("max_abs_decomposition_residual", 0.0) for l_ in labels), max(r["weighted"][l_].get("max_excursion", 0.0) for l_ in labels),
                         pg.get("status"), (" (max |dS_H| %s, |dS_P| %s, |dN| %s)" % (fx(pg.get("max_abs_dS_H"), "%.1e"), fx(pg.get("max_abs_dS_P"), "%.1e"), fx(pg.get("max_abs_dN_dust"), "%.1e"))) if pg.get("n_compared") else "",
                         ("complex64 slice: floor %s" % (r.get("floor") or {}).get("source_rel_range")) if r.get("complex64") else "complex128", r.get("device"), r.get("seconds", 0.0)))
        L.append("")
    L += ["- every number above is printed by the script from the stored slices and the records; %s; the reading is the lead's." % PRINT_STATUS]
    d = os.path.join(a.out, READOUT)
    os.makedirs(d, exist_ok=True)
    for hk in hists:
        if any(r["history"] == hk for r in rows):
            with open(os.path.join(d, "%s_%s.md" % (TITLE, hk)), "w") as f:
                f.write("\n".join(L) + "\n")
    with open(os.path.join(d, "%s_all.json" % TITLE), "w") as f:
        json.dump({"code": CODE, "title": TITLE, "print_status": PRINT_STATUS, "d15": D15, "definitions": DEFINITIONS, "histories": hists, "cuts_rs": labels, "rows": rows}, f, default=IO.jdefault, indent=1)


def main(argv=None):
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--slices-code", dest="slices_code", default=None)
    pa, _ = pre.parse_known_args(argv)
    bind_slices_code(pa.slices_code, required=True)
    sys.modules.setdefault(MODULE_NAME, sys.modules[__name__])
    IO.READOUT_MODULES[READOUT] = MODULE_NAME
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    IO.add_common_arguments(ap)
    add_arguments(ap)
    ap.set_defaults(cuts_rs="2,16,24")
    a = ap.parse_args(argv)
    a.readouts = READOUT
    IO.run(a)


if __name__ == "__main__":
    main()
