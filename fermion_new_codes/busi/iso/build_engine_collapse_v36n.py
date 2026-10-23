#!/usr/bin/env python
"""build_engine_collapse_v36n.py (busi, reviewer A; ISO_ruling_v1): the marked-patch builder of engine_collapse_v36n.py over engine_collapse_v36j.py.

The form of build_engine_collapse_v36e: a list of (OLD, NEW) blocks applied in order; each OLD occurs exactly once in the text at its application
and is replaced by its NEW; afterwards each NEW occurs exactly once in the result and replacing every NEW by its OLD in reverse order restores
the host byte for byte (both checked here before anything is written); the result is compiled (py_compile); the host's and the result's md5 and
every block go to <out>_patch.json beside the result. The blocks: the module docstring's first line (the version 36n paragraph prepended; the
anchor is the host's own first line, whatever its version letter), the module globals after the OPT line (the version 36n hooks: ISO, iso_*,
iter_cache_*), prepare_dust (the cavity keys at the start, the memory-lean spec entry, the ordering key before the levels line, the weights
before fill_lowest, the occupation scale after it, the rebuilt hw in the second loop, the catalog into the head), run() (the label after the
v34 lines, the printed catalog line after the initial-data print, the drop before the plain loop's eigh_all, the restore in the unconverged
block) and main() (the five arguments after --nk-set, iso_configure before C5.backend). Every hook returns its argument unchanged (the same
object) with the options at their defaults, so the default path is the host's: the gate is the self-test's J against N and G-16 of RUN_ISO_v1
(the member's first 16 records against the stored v36d records).
Independence of v36m (the host-offload engine built separately): v36m's blocks act inside the Newton solve (rounds_v36c, nk_stage_v32,
confirm_v32) and the record end's propagation; none of the anchors here lies there, and --iter-cache drop acts in the plain iteration loop that
the Newton path does not execute, so the two patch sets apply in either order: --host engine_collapse_v36m.py --out engine_collapse_v36o.py
gives the composite. The same over engine_collapse_v36e.py (the first-return end rule, which ISO_M3d_s4 needs) gives engine_collapse_v36ne.py.
Usage: python build_engine_collapse_v36n.py --code-dir DIR [--host engine_collapse_v36j.py] [--out engine_collapse_v36n.py] [--check-only] [--import-check]
  --check-only: re-derive the result from the host in memory and compare it with the file present (the build gate's second half).
  --import-check: import the result in a subprocess (needs the engine's environment: torch, scipy, the engine modules)."""
import argparse
import hashlib
import json
import os
import py_compile
import re
import subprocess
import sys

OPT_LINE = 'OPT = {"collar_policy": "flag", "support_rel": 1e-14, "site_fields": "dense"}'
NK_SET_LINE = '    ap.add_argument("--nk-set", dest="nk_set", default=None, choices=["resolve"], help="(v36c) the set-resolving Newton solve")'

DOC = '''"""@STEM@.py (busi, reviewer A; ISO_ruling_v1): version @OV@ = version @HV@ with options, each off by default, so that the defaults run
version @HV@'s code path (the preparation, the record loop and every number bitwise; the gates: the self-test's J against N on the fixture and the
member's first 16 records against the stored v36d records, G-16 of RUN_ISO_v1):
--dust-set isotropic [--dust-kappa-max K_d] [--dust-multiplicity 1|2]: the dust catalog of the isotropic Fermi ball at R_0 in place of the well's
    filling. In prepare_dust the ordering key of the well's level (kappa, j) in fill_lowest is the cavity's level: the j-th positive eigenvalue
    (at or above the band bottom m) of the free chain h_kappa (N = 1, v = 0, no well: Chan.clocked_dense(ones, zeros)) on the chain's sites
    R <= R_0, computed once per preparation (it does not depend on the fields); fill_lowest fills the lowest cavity levels across the sea set's
    channels with the chains' weights 16 kappa and f in [0, 1] (the last filled group fractional, uniform over its copies) to N_B exactly; the
    orbitals stay the well's levels Uwv[:, j] with the same (kappa, j) labels (the same radial node count), projected on the positive band of
    h_m[N, v] and Loewdin-orthonormalized as now; the self-consistency and everything after the preparation unchanged. The ruling (ISO_ruling_v1):
    the chains carry one parity of the 8-species field (16 kappa per channel), so the ball on them is the degeneracy-8 ball, kappa_max' =
    (9 pi N_B/16)^(1/3) = 1.209 N_B^(1/3) semiclassically (the discrete catalog's top channel is where a channel's first cavity level crosses E_F,
    lower than kappa_max' by the Airy term: 7 at N_B 640); the requirement's 0.960 N_B^(1/3) is the degeneracy-16 ball, realized on the chains only
    by --dust-multiplicity 2 (f <= 2 per level: the chain's orbital standing for both parities' orbitals of its radial label; labelled doubled).
    --dust-kappa-max K_d drops the channels above K_d (labelled capped when the uncapped filling reaches above K_d: the ball squeezed into
    kappa <= K_d takes higher radial levels, not isotropic); a catalog whose top channel is the sea set's top is labelled truncated by the sea
    set. The head carries dust_set and catalog (per channel the columns, labels and occupations, the top channel, the catalog's E_F and k_F R_0,
    the rule's kappa_max with the Weyl and surface-corrected continuum ball beside, f_kin on the orbitals and on the cavity key, the cavity's
    sites and dropped edge modes, the dust momentum, and (ISO_fix_v1b) the frame figures: the well's dust is at rest in the static frame, which
    moves through the chain's frame at the Eulerian speed |v|/N, so f_kin of the orbitals in the static frame is (sum_s sqrt(eps_s^2 - J_s^2) -
    N_B m)/(N_B m) and the dust momentum sum J equals the static frame's sum_s eps_s |v_s|/N_s, both carried beside the chain-frame figures), the
    run label "dust set: the isotropic cavity catalog at R_0 ..." and two printed lines after the initial data; nothing is added to the head or
    the label with the option off.
--iter-cache drop: in the plain iteration loop the previous iteration's eigenvectors and continued references leave the device before the next
    eigendecomposition (the entry keeps its eigenvalues and continuation record; the dict is mutated in place, so the smallest-residual
    iteration's entry is covered); a record that ends on such an iteration (unconverged, its smallest-residual iterate not the last) recomputes
    them at that iterate's applied fields from the record's window: the same matrices, the same cuSOLVER call, the same window, bitwise; the kept
    eigenvalues and masks are checked against the recomputed ones; TM is restored so the record's counts stay the record's (the seconds go to
    timing's iter_cache_recompute_s). The device peak falls from n^2 (256 K + about 150) bytes to n^2 (160 K + about 130) (ISO_ruling_v1 section 3).
--prep-memory lean: prepare_dust does not keep the dense well hamiltonian over the channel loop (rebuilt from the same fields where it is used:
    bitwise); the preparation's device set falls from 128 K n^2 to 64 K n^2 bytes.
Version @HV@'s docstring follows.
'''

HOOKS = r'''


# ------------------------------------------------------------------------------------------------------- version 36n (busi, reviewer A; ISO_ruling_v1)
ISO = {"set": "well", "kappa_max": None, "multiplicity": 1, "prep_memory": "full", "iter_cache": "keep", "desc": "", "keys": {}, "key_info": {},
       "n0": None, "R0": None, "m": None, "occ": None, "levels": None, "capped": False, "top_uncapped": None, "recomputes": 0}
C_ISO_CHAIN = (9.0 * math.pi / 16.0) ** (1.0 / 3.0)   # 1.209: the degeneracy-8 ball, one parity of 8 species: the chains' 16 kappa with f <= 1 (the ruling)
C_ISO_G16 = (9.0 * math.pi / 32.0) ** (1.0 / 3.0)     # 0.960: the degeneracy-16 ball (the requirement's rule), on the chains only with f <= 2


def iso_on():
    return ISO["set"] == "isotropic"


def iso_configure(a, ap):
    """main(): the version 36n options into ISO; with every option at its default the engine's code path is the host's."""
    ds = getattr(a, "dust_set", "well"); kd = getattr(a, "dust_kappa_max", None); mult = int(getattr(a, "dust_multiplicity", 1) or 1)
    if ds != "isotropic" and (kd is not None or mult != 1):
        ap.error("version 36n: --dust-kappa-max and --dust-multiplicity act on --dust-set isotropic")
    if ds == "isotropic" and (getattr(a, "massless_ball", False) or getattr(a, "shell", None)):
        ap.error("version 36n: --dust-set isotropic is the massive dust's cavity catalog at R_0 (not with --massless-ball or --shell)")
    if kd is not None and kd < 1:
        ap.error("version 36n: --dust-kappa-max must be at least 1")
    ISO.update({"set": ds, "kappa_max": kd, "multiplicity": mult, "prep_memory": getattr(a, "prep_memory", "full"), "iter_cache": getattr(a, "iter_cache", "keep")})
    if ds == "isotropic":
        s = ("the isotropic cavity catalog at R_0 = %g r_s (ISO_ruling_v1: the free chain's levels on the cavity's sites order the well's levels; "
             "one parity, 16 kappa per channel, f <= 1; kappa_max' = 1.209 N_B^(1/3))" % a.R0_over_rs)
        if mult != 1:
            s += ("; doubled: f <= %d (the chain's orbital standing for both parities' orbitals of its radial label, %d kappa per channel: the "
                  "requirement's degeneracy-16 count, kappa_max = 0.960 N_B^(1/3))" % (mult, 16 * mult))
        if kd is not None:
            s += "; the channel set capped at kappa <= %d" % kd
        ISO["desc"] = s


def iso_label(label, a):
    return label if not iso_on() else label + "; v36n: dust set: " + ISO["desc"]


def iso_begin(chs, R, R0, m):
    """prepare_dust's start (nothing on the default path): the isotropic set's ordering keys, once per preparation (they do not depend on the
    fields): per channel the positive levels of the free chain h_kappa (N = 1, v = 0, no well) on the cavity's sites R <= R_0, the lattice's
    hard wall; levels below the band bottom m (a hard cut's edge modes, if the chain had any) are dropped and counted."""
    if not iso_on():
        return
    torch = BK["torch"]
    Rr = np.asarray(R, float); nsites = len(Rr)
    n0 = int(np.sum(Rr <= R0 * (1.0 + 1e-12)))
    if n0 < 4:
        raise RuntimeError("version 36n: the cavity R <= R_0 = %g holds %d sites" % (R0, n0))
    ones = np.ones(nsites); zeros = np.zeros(nsites)
    keys, info = {}, {}
    for ch in chs:
        H = ch.clocked_dense(ones, zeros)
        ev = torch.linalg.eigvalsh(H[:2 * n0, :2 * n0]).cpu().numpy().astype(float)
        del H
        pos = np.sort(ev[ev > 0])
        inband = pos >= m * (1.0 - 1e-9)
        if not inband.any():
            raise RuntimeError("version 36n: the cavity spectrum of channel %d has no level at or above m" % ch.kappa)
        keys[ch.kappa] = pos[inband]
        info[ch.kappa] = {"positive_levels": int(len(pos)), "edge_modes_dropped": int(np.sum(~inband)), "lowest": float(pos[inband][0])}
    ISO.update({"keys": keys, "key_info": info, "n0": n0, "R0": float(R0), "m": float(m), "occ": None, "levels": None, "capped": False, "top_uncapped": None})


def iso_key(ch, ewp):
    """the ordering key of the well's level (kappa, j) in fill_lowest: the default returns the well's positive eigenvalues themselves (the same
    object: bitwise); the isotropic set returns the cavity's j-th positive level for j below both counts (the orbitals stay the well's)."""
    if not iso_on():
        return ewp
    key = ISO["keys"][ch.kappa]
    return np.asarray(key[:min(len(ewp), len(key))], float)


def iso_top(occ, chs):
    ks = [chs[ic].kappa for (ic, j), f in occ.items() if f > 0]
    return max(ks) if ks else 0


def iso_levels(levels, chs, NB):
    """before fill_lowest (the default returns the list itself): the isotropic set weights each level multiplicity x 16 kappa (f per copy stays in
    [0, 1]: the last filled group fractional and uniform as fill_lowest makes it) and under --dust-kappa-max K_d drops the channels above K_d
    (capped when the uncapped filling reaches above K_d); the filling is recorded for the head (the same call fill_lowest makes next)."""
    if not iso_on():
        return levels
    mult = ISO["multiplicity"]
    lv = [(e_, w_ * mult, ic, j) for (e_, w_, ic, j) in levels]
    kd = ISO["kappa_max"]
    if kd is not None:
        ISO["top_uncapped"] = iso_top(fill_lowest(list(lv), NB), chs)
        lv = [t for t in lv if chs[t[2]].kappa <= kd]
        ISO["capped"] = bool(ISO["top_uncapped"] > kd)
    ISO["occ"] = fill_lowest(list(lv), NB)
    ISO["levels"] = lv
    return lv


def iso_scale(fk):
    return fk if (not iso_on() or ISO["multiplicity"] == 1) else ISO["multiplicity"] * fk


def iso_spec(entry):
    """--prep-memory lean: the well's dense hamiltonian is not kept over the channel loop (rebuilt from the same fields where it is used: bitwise)."""
    return entry if ISO["prep_memory"] != "lean" else (entry[0], entry[1], entry[2], None)


def iso_hw(hw, ch, N, v, reading, Ud):
    return hw if hw is not None else ch.clocked_dense(N, v if reading == "A" else 0 * v) + Ud


def iso_wkb(kappa, kR):
    """the WKB count of a hard cavity's levels below k in channel kappa (L = kappa; the turning point's and the wall's quarter waves)."""
    x = kappa / kR
    if x >= 1.0:
        return 0
    return int(math.floor((kR / math.pi) * (math.sqrt(1.0 - x * x) - x * math.acos(x)) + 0.25))


def iso_ball(g, NB, R0, m):
    """the continuum ball of degeneracy g per momentum in the cavity R_0: at the leading Weyl order k_F R_0 = (9 pi N_B/(2 g))^(1/3) and with the
    Dirichlet surface term (2 g (k R_0)^3/(9 pi) - g (k R_0)^2/4 = N_B); per variant k_F R_0, p_F/m, E_kin/(N_B m) (relativistic, from the
    variant's density of states) and the cold form 0.3 (p_F/m)^2."""
    kR0 = (9.0 * math.pi * NB / (2.0 * g)) ** (1.0 / 3.0)
    kR1 = kR0
    for _ in range(100):
        f_ = 2.0 * g * kR1 ** 3 / (9.0 * math.pi) - g * kR1 ** 2 / 4.0 - NB
        d_ = 6.0 * g * kR1 ** 2 / (9.0 * math.pi) - g * kR1 / 2.0
        if d_ <= 0:
            break
        kR1 -= f_ / d_
        if abs(f_ / d_) < 1e-12:
            break
    tz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    out = {}
    for lab, kR, surf in (("weyl", kR0, 0.0), ("surface", kR1, g * R0 ** 2 / 2.0)):
        pF = kR / R0
        p = np.linspace(0.0, pF, 4001)
        dos = 2.0 * g * R0 ** 3 / (3.0 * math.pi) * p * p - surf * p
        ek = float(tz((np.sqrt(p * p + m * m) - m) * dos, p))
        out[lab] = {"k_F_R0": kR, "p_F_over_m": pF / m, "f_kin": ek / (NB * m), "f_kin_cold": 0.3 * (pF / m) ** 2}
    return out


def iso_frame(NB, m, N, v, eps, J):
    """ISO_fix_v1b: the dust's figures in the frame it is at rest in. The well's orbitals are stationary states of h_m[N, v] + U_w: at rest in
    the static frame, which moves through the chain's (Painleve-Gullstrand) frame at the Eulerian speed |v|/N; the chain's h and Q measure the
    river frame's energy and momentum per site (eps_s, J_s), so f_kin on the orbitals carries the bulk flow and sum J = sum_s eps_s |v_s|/N_s
    (the Eulerian relation, up to the pressure's few percent), not 0. sum_s sqrt(eps_s^2 - J_s^2) = sum_s eps_s sqrt(1 - u_s^2), u = J/eps, is
    the static frame's N_B m plus thermal energy (the relativistic form of the host's f_kin_bulk_frame; exact for a local boost to
    O(v^2 (T^rr/T^00)^2))."""
    eps_ = np.asarray(eps, float); J_ = np.asarray(J, float)
    rest = float(np.sum(np.sqrt(np.maximum(eps_ * eps_ - J_ * J_, 0.0))))
    out = {"f_kin_static_frame": (rest - NB * m) / (NB * m), "f_kin_chain_frame": (float(eps_.sum()) - NB * m) / (NB * m),
           "dust_momentum_total": float(J_.sum()), "dust_velocity_mean": float(J_.sum() / eps_.sum())}
    if N is not None and v is not None:
        uE = np.abs(np.asarray(v, float)) / np.asarray(N, float)
        out["static_frame_momentum"] = float(np.sum(eps_ * uE)); out["static_frame_speed_mean"] = out["static_frame_momentum"] / float(eps_.sum())
        out["static_frame_speed_max"] = float(uE.max())
        out["momentum_ratio"] = (abs(out["dust_momentum_total"]) / out["static_frame_momentum"]) if out["static_frame_momentum"] > 0 else None
    return out


def iso_head(head, chs, dust, m, N=None, v=None, eps=None, J=None):
    """the isotropic set's catalog into the head (the default adds nothing to the head); ISO_fix_v1b: eps, J are prepare_dust's dust site
    arrays (the sums head["f_kin"] and head["dust_momentum_total"] are made of) and N, v its final fields, for iso_frame."""
    if not iso_on():
        return
    NB = float(head["N_B"]); nb13 = NB ** (1.0 / 3.0); R0 = ISO["R0"]; mult = ISO["multiplicity"]; g = 8 * mult
    frame = iso_frame(NB, m, N, v, eps, J) if (eps is not None and J is not None) else {}
    occ = ISO["occ"] or {}
    ball = iso_ball(g, NB, R0, m)
    kR_w, kR_s = ball["weyl"]["k_F_R0"], ball["surface"]["k_F_R0"]
    cat, top, ekin_key, EF = {}, 0, 0.0, m
    for ic, ch in enumerate(chs):
        js = sorted(j for (i_, j), f in occ.items() if i_ == ic and f > 0)
        if not js:
            continue
        key = ISO["keys"][ch.kappa]
        f = np.array([mult * occ[(ic, j)] for j in js], float); E = np.array([key[j] for j in js], float)
        ekin_key += ch.w * float(f @ (E - m)); EF = max(EF, float(E.max())); top = max(top, ch.kappa)
        cat[str(ch.kappa)] = {"columns": len(js), "j": js, "f": [float(x) for x in f], "particles": float(ch.w * f.sum()), "lowest_f": float(f.min()),
                              "E_cavity": [float(x) for x in E], "n_WKB_surface": iso_wkb(ch.kappa, kR_s), "w": ch.w}
    kF = math.sqrt(max(EF * EF - m * m, 0.0))
    ktop = max(ch.kappa for ch in chs)
    c1 = (cat.get("1") or {}).get("columns", 0)
    head["dust_set"] = "isotropic"
    head["catalog"] = {"label": "dust set: " + ISO["desc"], "multiplicity": mult, "g_per_momentum": g, "weights": "16 kappa x f, f <= %d" % mult,
                       "kappa_max_coefficient": kR_w / nb13, "kappa_max_rule": kR_w, "kappa_max_chain_rule": C_ISO_CHAIN * nb13, "kappa_max_g16_rule": C_ISO_G16 * nb13,
                       "ball": ball, "k_F_R0_catalog": kF * R0, "E_F": EF, "top_channel": top, "sea_top": ktop, "truncated_by_sea_set": bool(top >= ktop),
                       "capped": ISO["capped"], "kappa_cap": ISO["kappa_max"], "top_uncapped": ISO["top_uncapped"],
                       "f_kin_orbitals": head["f_kin"], "f_kin_on_cavity_key": ekin_key / (NB * m), "columns_kappa_1": c1, "R_c_4a_columns_kappa_1": 4.0 * c1,
                       "cavity_sites": ISO["n0"], "R0": R0, "key_info": {str(k): v for k, v in ISO["key_info"].items()}, "channels": cat,
                       "dust_momentum_total": head["dust_momentum_total"], "frame": frame}


def iso_print(head):
    if not iso_on():
        return
    c = head["catalog"]
    print("dust set (v36n): %s; N_B %.6f; top channel %d of the sea set's %d%s%s; k_F R_0 of the catalog %.3f (E_F %.6f) against the ball's %.3f (Weyl) and %.3f (surface term); kappa_max' %.2f = %.4f N_B^(1/3) (the g = 16 rule's %.2f beside); f_kin %.5f on the orbitals, %.5f on the cavity key, the ball's %.5f (Weyl) and %.5f (surface); p_F/m %.4f (Weyl) %.4f (surface); 4 a x the kappa 1 columns %.0f a; dust momentum %.2e; cavity %d sites, edge modes dropped %d; per channel: %s" % (
        c["label"], head["N_B"], c["top_channel"], c["sea_top"], " TRUNCATED BY THE SEA SET" if c["truncated_by_sea_set"] else "",
        (" CAPPED at %d (uncapped top %d)" % (c["kappa_cap"], c["top_uncapped"])) if c["capped"] else "", c["k_F_R0_catalog"], c["E_F"],
        c["ball"]["weyl"]["k_F_R0"], c["ball"]["surface"]["k_F_R0"], c["kappa_max_rule"], c["kappa_max_coefficient"], c["kappa_max_g16_rule"],
        c["f_kin_orbitals"], c["f_kin_on_cavity_key"], c["ball"]["weyl"]["f_kin"], c["ball"]["surface"]["f_kin"], c["ball"]["weyl"]["p_F_over_m"],
        c["ball"]["surface"]["p_F_over_m"], c["R_c_4a_columns_kappa_1"], c["dust_momentum_total"], c["cavity_sites"],
        sum(v["edge_modes_dropped"] for v in c["key_info"].values()),
        "; ".join("kappa %s: %d columns (WKB %d), %.3f particles, lowest f %.4f" % (k, v["columns"], v["n_WKB_surface"], v["particles"], v["lowest_f"]) for k, v in c["channels"].items())), flush=True)
    fr = c.get("frame") or {}
    if fr:                                                                                     # ISO_fix_v1b: the frame line
        print("dust set (v36n) frames: f_kin %.5f in the chain's frame (the head's f_kin: the static frame's plus the bulk flow's), %.5f in the static frame (sum_s sqrt(eps_s^2 - J_s^2) - N_B m), the cavity key's %.5f (static/key %.3f: the well's wall against the hard cavity); dust momentum sum J %.4f against the static frame's sum eps |v|/N %s (ratio %s; zero J would be free fall with the river); sum J / sum eps %.4f, eps-weighted |v|/N %s, largest |v|/N %s" % (
            fr["f_kin_chain_frame"], fr["f_kin_static_frame"], c["f_kin_on_cavity_key"], (fr["f_kin_static_frame"] / c["f_kin_on_cavity_key"]) if c["f_kin_on_cavity_key"] else float("nan"),
            fr["dust_momentum_total"], ("%.4f" % fr["static_frame_momentum"]) if "static_frame_momentum" in fr else "-", ("%.4f" % fr["momentum_ratio"]) if fr.get("momentum_ratio") is not None else "-",
            fr["dust_velocity_mean"], ("%.4f" % fr["static_frame_speed_mean"]) if "static_frame_speed_mean" in fr else "-", ("%.4f" % fr["static_frame_speed_max"]) if "static_frame_speed_max" in fr else "-"), flush=True)


def iter_cache_drop(a, cache):
    """--iter-cache drop (the default keeps everything and returns the same dict): before the next eigendecomposition the previous iteration's
    eigenvectors and continued references leave the device; the entry keeps its eigenvalues and continuation record, the dict is mutated in
    place (the smallest-residual iteration's entry is this dict or an earlier one treated the same way); a record that ends on such an
    iteration recomputes them (iter_cache_restore). The lagged entry (no eigenvalues) is untouched."""
    if ISO["iter_cache"] != "drop":
        return cache
    for k_ in list(cache):
        t_ = cache[k_]
        if t_[1] is not None and (t_[2] is not None or t_[3] is not None):
            cache[k_] = (t_[0], t_[1], None, None, t_[4])
    return cache


def iter_cache_restore(a, chs, st, cache, N_a, v_a):
    """the record ends on an iteration whose eigenvectors were dropped (unconverged, its smallest-residual iterate not the last): its
    eigendecomposition at its applied fields and its continuation from the record's window, recomputed: the same matrices, the same cuSOLVER
    call, the same window (bitwise); the kept eigenvalues and masks are checked against the recomputed ones; TM is restored so that the record's
    counts stay the record's (the seconds go to timing's iter_cache_recompute_s)."""
    if ISO["iter_cache"] != "drop" or not any(t_[1] is not None and t_[2] is None for t_ in cache.values()):
        return cache
    torch = BK["torch"]
    tm0 = dict(TM); t_r = time.time()
    eig = eigh_all(chs, N_a, v_a, bool(a.streams))
    out = dict(cache)
    for ch in chs:
        e, U = eig.pop(ch.kappa)
        t_ = cache[ch.kappa]
        if a.reference == "R2":
            s_ = st[ch.kappa]
            ref, cinfo, mask = continue_ref_v34(a, ch, s_["window"], (N_a, v_a), e, U, s_["n_sea"])
            cinfo["_mask"] = mask
        else:
            ref = st[ch.kappa]["sea"]; cinfo = {"mode": "R1", "_mask": (e < 0).cpu().numpy()}
        if t_[1] is not None and not torch.equal(t_[1], e):
            raise RuntimeError("version 36n: the recomputed eigenvalues of channel %d differ from the record's" % ch.kappa)
        m_old = t_[4].get("_mask") if isinstance(t_[4], dict) else None
        if m_old is not None and not np.array_equal(np.asarray(m_old, bool), np.asarray(cinfo["_mask"], bool)):
            raise RuntimeError("version 36n: the recomputed occupation mask of channel %d differs from the record's" % ch.kappa)
        out[ch.kappa] = (None, e, U, ref, cinfo)
    del eig
    sec = time.time() - t_r
    TM.clear(); TM.update(tm0); TM["iter_cache_recompute_s"] = TM.get("iter_cache_recompute_s", 0.0) + sec
    ISO["recomputes"] += 1
    return out
'''

ARGS_NEW = '''
    ap.add_argument("--dust-set", dest="dust_set", default="well", choices=["well", "isotropic"], help="(v36n) the dust catalog: the well's filling (the default) or the isotropic cavity catalog at R_0 (ISO_ruling_v1)")
    ap.add_argument("--dust-kappa-max", dest="dust_kappa_max", type=int, default=None, help="(v36n) with isotropic: cap the catalog's channel set at this kappa (labelled capped when it binds)")
    ap.add_argument("--dust-multiplicity", dest="dust_multiplicity", type=int, default=1, choices=[1, 2], help="(v36n) with isotropic: 2 fills f <= 2 per level (the requirement's degeneracy-16 count on the one-parity chains; labelled doubled)")
    ap.add_argument("--iter-cache", dest="iter_cache", default="keep", choices=["keep", "drop"], help="(v36n) drop: release the previous iteration's eigenvectors and references before the next eigendecomposition (recomputed when the record ends on them); bitwise")
    ap.add_argument("--prep-memory", dest="prep_memory", default="full", choices=["full", "lean"], help="(v36n) lean: the preparation does not keep the dense well hamiltonians over the channel loop (rebuilt where used); bitwise")'''


def md5(b):
    return hashlib.md5(b).hexdigest()


def blocks(first_line, out_stem, hv, ov):
    doc_new = DOC.replace("@STEM@", out_stem).replace("@HV@", hv).replace("@OV@", ov) + first_line[3:]
    return [
        ("docstring", first_line, doc_new),
        ("hooks", OPT_LINE, OPT_LINE + HOOKS),
        ("prep_begin", "    U0 = 0.5 * m",
         "    U0 = 0.5 * m\n    iso_begin(chs, R, R0, m)                                                    # version 36n (nothing by default)"),
        ("prep_spec", "            spec[ch.kappa] = (ewp, Uwv[:, pos], Um[:, em > 0], hw)",
         "            spec[ch.kappa] = (ewp, Uwv[:, pos], Um[:, em > 0], hw)\n            spec[ch.kappa] = iso_spec(spec[ch.kappa])                                  # version 36n (--prep-memory lean; the same entry by default)"),
        ("prep_key", "            levels += [(float(x), ch.w, ic, j) for j, x in enumerate(ewp[:400])]",
         "            ewp = iso_key(ch, ewp)                                                       # version 36n (the ordering key; the same object by default)\n            levels += [(float(x), ch.w, ic, j) for j, x in enumerate(ewp[:400])]"),
        ("prep_fill", "        occ = fill_lowest(levels, NB)",
         "        levels = iso_levels(levels, chs, NB)                                             # version 36n (the same list by default)\n        occ = fill_lowest(levels, NB)"),
        ("prep_hw", "            ewp, Uwv, Pplus, hw = spec[ch.kappa]",
         "            ewp, Uwv, Pplus, hw = spec[ch.kappa]\n            hw = iso_hw(hw, ch, N, v, reading, Ud)                                        # version 36n (rebuilt only under --prep-memory lean)"),
        ("prep_fk", "            fk = np.array([occ[(ic, j)] for j in keep])",
         "            fk = np.array([occ[(ic, j)] for j in keep])\n            fk = iso_scale(fk)                                                           # version 36n (the same array by default)"),
        ("prep_head", '            "solver_iterations": info.get("it"), "dictionary": FLD.DICTIONARY}',
         '            "solver_iterations": info.get("it"), "dictionary": FLD.DICTIONARY}\n    iso_head(head, chs, dust, m, N, v, eps, J)                                                    # version 36n (nothing by default)'),
        ("run_label", '    if v34_on(a):\n        label += "; v34: " + v34_desc(a)',
         '    if v34_on(a):\n        label += "; v34: " + v34_desc(a)\n    label = iso_label(label, a)                                                     # version 36n (the same label by default)'),
        ("run_print", '            head["collar_margin"], head["seconds"]), flush=True)',
         '            head["collar_margin"], head["seconds"]), flush=True)\n        iso_print(head)                                                                 # version 36n (nothing by default)'),
        ("run_drop", "            eig = eigh_all(chs, N_app, v_app, bool(a.streams))",
         "            cache = iter_cache_drop(a, cache)                                           # version 36n (--iter-cache drop; the same dict by default)\n            eig = eigh_all(chs, N_app, v_app, bool(a.streams))"),
        ("run_restore", '            N_app, v_app = best["app"]',
         '            N_app, v_app = best["app"]\n            cache = iter_cache_restore(a, chs, st, cache, N_app, v_app)                 # version 36n (the same dict unless entries were dropped)'),
        ("main_args", NK_SET_LINE, NK_SET_LINE + ARGS_NEW),
        ("main_configure", "    C5.backend(a.gpu)",
         "    iso_configure(a, ap)                                                            # version 36n\n    C5.backend(a.gpu)"),
    ]


def derive(host_text, out_stem):
    first = next(l for l in host_text.split("\n") if not l.startswith("#!"))   # busi fix: the hosts begin with a shebang line; the docstring head is the first line after it
    mh = re.match(r'"""engine_collapse_v36(\w+)\.py', first)
    if not mh:
        raise SystemExit("the host's first line is not an engine_collapse_v36* docstring head: %r" % first[:80])
    mo = re.match(r"engine_collapse_v36(\w+)$", out_stem)
    if not mo:
        raise SystemExit("the output must be named engine_collapse_v36<letters>.py (got %s)" % out_stem)
    hv, ov = "36" + mh.group(1), "36" + mo.group(1)
    B = blocks(first, out_stem, hv, ov)
    text = host_text
    for name, old, new in B:
        c = text.count(old)
        if c != 1:
            raise SystemExit("block %s: its OLD occurs %d times in the text at its application (exactly 1 needed)" % (name, c))
        text = text.replace(old, new, 1)
    back = text
    for name, old, new in reversed(B):
        c = back.count(new)
        if c != 1:
            raise SystemExit("block %s: its NEW occurs %d times in the result (exactly 1 needed for the reverse)" % (name, c))
        back = back.replace(new, old, 1)
    if back != host_text:
        raise SystemExit("the reverse replacements do not restore the host byte for byte")
    return text, B, hv, ov


def main():
    ap = argparse.ArgumentParser(description="the marked-patch builder of engine_collapse_v36n.py (busi, ISO_ruling_v1)")
    ap.add_argument("--code-dir", required=True, help="the engine code tree (engine_collapse_v36j.py and its modules)")
    ap.add_argument("--host", default="engine_collapse_v36j.py")
    ap.add_argument("--out", default="engine_collapse_v36n.py")
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--import-check", action="store_true")
    a = ap.parse_args()
    hp = os.path.join(a.code_dir, a.host); op = os.path.join(a.code_dir, a.out)
    host = open(hp, "rb").read().decode("utf-8")
    stem = os.path.splitext(os.path.basename(a.out))[0]
    text, B, hv, ov = derive(host, stem)
    if a.check_only:
        if not os.path.exists(op):
            raise SystemExit("--check-only: %s is not present" % op)
        cur = open(op, "rb").read().decode("utf-8")
        same = cur == text
        print("check: %s %s the derivation from %s (host md5 %s; file md5 %s; derived md5 %s)" % (
            a.out, "equals" if same else "DIFFERS FROM", a.host, md5(host.encode("utf-8")), md5(cur.encode("utf-8")), md5(text.encode("utf-8"))))
        sys.exit(0 if same else 1)
    with open(op, "wb") as fh:
        fh.write(text.encode("utf-8"))
    py_compile.compile(op, doraise=True)
    info = {"host": a.host, "host_md5": md5(host.encode("utf-8")), "out": a.out, "out_md5": md5(text.encode("utf-8")), "host_version": hv, "out_version": ov,
            "blocks": [{"name": n_, "old": o_, "new": w_} for n_, o_, w_ in B],
            "invariants": ["each OLD occurs exactly once at its application", "each NEW occurs exactly once in the result",
                           "replacing every NEW by its OLD in reverse order restores the host byte for byte", "py_compile passes"]}
    with open(os.path.join(a.code_dir, stem + "_patch.json"), "w") as fh:
        json.dump(info, fh, indent=1)
    print("built %s from %s: %d blocks; host md5 %s; result md5 %s; compiled; the reverse restores the host byte for byte; %s written" % (
        a.out, a.host, len(B), info["host_md5"], info["out_md5"], stem + "_patch.json"))
    print("hook calls in the result: %d (iso_begin, iso_spec, iso_key, iso_levels, iso_hw, iso_scale, iso_head, iso_label, iso_print, iter_cache_drop, iter_cache_restore, iso_configure)" % sum(
        text.count(s_) for s_ in ("    iso_begin(chs, R, R0, m)", "iso_spec(spec[ch.kappa])", "ewp = iso_key(ch, ewp)", "levels = iso_levels(levels, chs, NB)",
                                   "hw = iso_hw(hw, ch, N, v, reading, Ud)", "fk = iso_scale(fk)", "    iso_head(head, chs, dust, m, N, v, eps, J)", "label = iso_label(label, a)",
                                   "        iso_print(head)", "cache = iter_cache_drop(a, cache)", "cache = iter_cache_restore(a, chs, st, cache, N_app, v_app)", "    iso_configure(a, ap)")))
    if a.import_check:
        rc = subprocess.call([sys.executable, "-c", "import sys; sys.path.insert(0, %r); import %s as E; print('import ok:', E.__name__, 'ISO defaults', E.ISO['set'], E.ISO['iter_cache'], E.ISO['prep_memory'])" % (os.path.abspath(a.code_dir), stem)],
                             cwd=a.code_dir)
        if rc != 0:
            raise SystemExit("the import check failed (the engine's environment is needed: torch, scipy, the engine modules)")


if __name__ == "__main__":
    main()
