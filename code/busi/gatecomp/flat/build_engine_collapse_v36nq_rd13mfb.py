#!/usr/bin/env python
"""build_engine_collapse_v36nq_rd13mfb.py (busi, analyst; GATECOMP, the flat preparation; ReviewA_gatecomp_v1 sec. 2 (ii), ReviewB_gatecomp_v1
sec. 3; busi/notes/GATECOMP_flat_build_v1.md): engine_collapse_v36nq_rd13mfb.py from the merged engine engine_collapse_v36nq_rd13mf.py (md5
248f25f3..., code16d) by marked replacement blocks, in the form of build_engine_collapse_v36nq_rd13.py (busi/code/rd13).
Rules (checked): the source's md5 begins with 248f25f3 (--expect-src-md5); every OLD anchor is copied verbatim from the source and occurs
exactly once; every line a NEW block adds or changes carries "flat prep"; the reverse pass (NEW -> OLD, each NEW once) restores the source
byte for byte (in memory before writing and again on the written file; available as --reverse); the output compiles; nothing is overwritten.
    python build_engine_collapse_v36nq_rd13mfb.py --src .../engine_collapse_v36nq_rd13mf.py [--dst .../engine_collapse_v36nq_rd13mfb.py] [--check-only]
    python build_engine_collapse_v36nq_rd13mfb.py --reverse --src .../engine_collapse_v36nq_rd13mfb.py --dst .../restored.py --expect .../engine_collapse_v36nq_rd13mf.py

The option (default --prep well: the parent's code path; nothing of the patch runs but one dictionary read per call site, so the records,
fields, spectra, handoff and slices are 248f25f3's bitwise; the json's args gain the keys prep, prep_R, prep_edge, prep_taper):
--prep flatbox [--prep-R R_b] [--prep-edge hard | taper] [--prep-taper W]: the dust orbitals are the lowest levels of a box of radius R_b
    (units of r_s; default --R0-over-rs, 4) in place of the levitating well's levels. Per channel, once per preparation, the free chain
    h_kappa (N = 1, v = 0, no well) on the sites R <= R_wall (the lattice's hard wall, as v36n's iso_begin cuts it), plus for --prep-edge
    taper the smooth wall U(R) = E_k ((R - R_in)/W)^2 for R > R_in (R_in = R_b - W, R_wall = R_b + W, W in r_s, E_k the hard box's Weyl
    Fermi kinetic energy at R_b on the degeneracy-8 chains, so the Thomas-Fermi density falls as (1 - x^2)^(3/2) and vanishes near R_b; the
    hard wall at R_b + W, where U = 4 E_k). Its positive levels at or above m (at most 400 per channel) are the ordering keys and its
    eigenvectors, embedded with zeros beyond R_wall, the orbitals; fill_lowest fills them across the sea set's channels with the chains'
    weights 16 kappa to N_B = r_s/(2 g m) exactly (the ladder's rest number); each pass projects the kept columns on the positive band of
    h_m[N, v] and Loewdin-orthonormalizes them as the well's filling is (the projector made only for channels that hold dust; the well's
    eigendecomposition is not made); the fields' self-consistency, E_0 = FLD.charge and everything after the preparation are the parent's.
    The levels are real standing waves, so the dust momentum is zero before the projection: the dust moves with the river (the marginally
    bound Oppenheimer-Snyder release of the paper's homogeneous control), not at coordinate rest. Stationarity is read against the cavity's
    hamiltonian (the block-diagonal free chain with the wall). hard is reviewer A's text (sec. 2 (ii)); taper is reviewer B's (sec. 3: "a smooth
    taper, not a sharp edge"); both are preparation-ladder members, not a change of model. The head carries prep and flatbox (the catalog
    per channel, the top channel, E_F, the cavity, the edge, the density profile and its uniformity, the frame figures); the run label gains
    "flat prep: ..."; nothing is added to the head or the label at the default.
Blocks:
  A  the module docstring's lineage lines;
  B  before iter_cache_drop: FLAT, flat_configure, flat_label, flat_begin, flat_channel, flat_hcav, flat_proj, flat_head, flat_print;
  C  prepare_dust: flat_begin after iso_begin;
  D  prepare_dust's first channel loop: the cavity's keys and orbitals (the well's two eigendecompositions skipped under flatbox);
  E  prepare_dust's second channel loop: flat_proj before the projection;
  F  prepare_dust: flat_head after iso_head;
  G  run(): flat_label after iso_label; flat_print after iso_print;
  H  the options (after --nk-plain-fallback);
  I  main(): flat_configure after iso_configure.
"""
import argparse
import hashlib
import os
import py_compile

MARK = "flat prep"
SRC_MD5_PREFIX = "248f25f3"
SRC_NAME = "engine_collapse_v36nq_rd13mf.py"
OUT_NAME = "engine_collapse_v36nq_rd13mfb.py"


def blk(s):
    return s[1:] if s.startswith("\n") else s


EDITS = []

EDITS.append(("A lineage", blk(r'''
#!/usr/bin/env python
"""engine_collapse_v36nq_rd13mf.py (busi, analyst; MERGE-1: RD-2's --fields wiring'''), blk(r'''
#!/usr/bin/env python
"""engine_collapse_v36nq_rd13mfb.py (busi, analyst; GATECOMP, the flat preparation; built by build_engine_collapse_v36nq_rd13mfb.py over v36nq_rd13mf, md5 248f25f3) [flat prep]
version 36nq_rd13mfb = version 36nq_rd13mf with one option, off by default, so that the defaults are 248f25f3 bitwise (the json's args gain 4 keys): [flat prep]
--prep flatbox [--prep-R R_b] [--prep-edge hard|taper] [--prep-taper W]: the dust orbitals are the lowest levels of a box of radius R_b r_s [flat prep]
    (default --R0-over-rs): the free chain's levels on R <= R_wall (hard: the lattice's wall at R_b; taper: U = E_k ((R - R_in)/W)^2 beyond [flat prep]
    R_in = R_b - W, the wall at R_b + W), filled by fill_lowest across the channels (16 kappa) to N_B = r_s/(2 g m), projected on the positive [flat prep]
    band of h_m[N, v] and Loewdin-orthonormalized each pass, the fields self-consistent as for the well's filling (ReviewA_gatecomp_v1 sec. 2 (ii): [flat prep]
    "a preparation-ladder member, not a change of model"; taper: ReviewB_gatecomp_v1 sec. 3). Version 36nq_rd13mf's docstring follows. [flat prep]
engine_collapse_v36nq_rd13mf.py (busi, analyst; MERGE-1: RD-2's --fields wiring''')))

EDITS.append(("B helpers", blk(r'''


def iter_cache_drop(a, cache):
'''), blk(r'''


# ------------------------------------------------------------------------------------------------------- flat prep (busi, analyst; GATECOMP)   [flat prep]
FLAT = {"on": False, "edge": "hard", "R_b": None, "W": None, "R_in": None, "R_wall": None, "E_k": None, "nc": None, "Ud": None, "desc": "",   # flat prep
        "keys": {}, "vecs": {}, "info": {}, "NB": None}   # flat prep: the option's state (the default: nothing of flat prep runs)


def flat_configure(a, ap):   # flat prep: main(), after iso_configure; --prep well (the default) returns at once
    p_ = getattr(a, "prep", "well"); e_ = getattr(a, "prep_edge", "hard"); W_ = getattr(a, "prep_taper", None); Rb_ = getattr(a, "prep_R", None)   # flat prep
    if p_ == "well":   # flat prep
        if e_ != "hard" or W_ is not None or Rb_ is not None:   # flat prep
            ap.error("flat prep: --prep-R, --prep-edge and --prep-taper act on --prep flatbox")   # flat prep
        return   # flat prep
    if getattr(a, "dust_set", "well") != "well" or getattr(a, "massless_ball", False) or getattr(a, "shell", None) or not a.m > 0:   # flat prep
        ap.error("flat prep: --prep flatbox replaces the well's filling (not with --dust-set isotropic, --massless-ball, --shell, or m = 0)")   # flat prep
    Rb_ = float(a.R0_over_rs if Rb_ is None else Rb_)   # flat prep
    if e_ == "hard" and W_ is not None:   # flat prep
        ap.error("flat prep: --prep-taper acts on --prep-edge taper")   # flat prep
    if e_ == "taper" and (W_ is None or not 0.0 < W_ < Rb_):   # flat prep
        ap.error("flat prep: --prep-edge taper needs --prep-taper W with 0 < W < R_b (units of r_s)")   # flat prep
    FLAT.update(on=True, edge=e_, R_b=Rb_ * a.rs, W=(W_ * a.rs if e_ == "taper" else None))   # flat prep
    if e_ == "hard":   # flat prep
        FLAT["desc"] = ("the flat box: the lowest levels of the free chain on R <= R_b = %g r_s (the lattice's hard wall; ReviewA_gatecomp_v1 sec. 2 (ii)), "   # flat prep
                        "16 kappa per channel, f <= 1, to N_B; projected on the positive band of h_m[N, v] each pass" % Rb_)   # flat prep
    else:   # flat prep
        FLAT["desc"] = ("the tapered box: the lowest levels of the free chain with U = E_k ((R - R_in)/W)^2 beyond R_in = %g r_s (W = %g r_s, E_k the hard box's "   # flat prep
                        "Weyl Fermi kinetic energy at R_b = %g r_s), the wall at %g r_s (ReviewB_gatecomp_v1 sec. 3), 16 kappa per channel, f <= 1, to N_B; "   # flat prep
                        "projected on the positive band of h_m[N, v] each pass" % (Rb_ - W_, W_, Rb_, Rb_ + W_))   # flat prep


def flat_label(label):   # flat prep: the run label (the same object by default)
    return label if not FLAT["on"] else label + "; flat prep: " + FLAT["desc"]   # flat prep


def flat_begin(chs, R, m, NB):   # flat prep: prepare_dust's start: per channel the cavity's levels and orbitals, once (they do not depend on the fields)
    if not FLAT["on"]:   # flat prep
        return   # flat prep
    torch = BK["torch"]   # flat prep
    Rr = np.asarray(R, float); ns = len(Rr); Rb = FLAT["R_b"]   # flat prep
    if FLAT["edge"] == "hard":   # flat prep
        R_in, R_wall, Ek = Rb, Rb, None   # flat prep
        U = np.zeros(ns)   # flat prep
    else:   # flat prep
        W = FLAT["W"]; R_in, R_wall = Rb - W, Rb + W   # flat prep
        Ek = math.sqrt((iso_ball(8, NB, Rb, m)["weyl"]["p_F_over_m"] * m) ** 2 + m * m) - m   # flat prep: the hard box's Weyl Fermi kinetic energy at R_b
        x = np.clip((Rr - R_in) / W, 0.0, None)   # flat prep
        U = Ek * x * x   # flat prep
    nc = int(np.sum(Rr <= R_wall * (1.0 + 1e-12)))   # flat prep
    if nc < 4 or nc >= ns:   # flat prep
        raise RuntimeError("flat prep: the box R <= %g holds %d of the chain's %d sites" % (R_wall, nc, ns))   # flat prep
    Ud = T_(np.diag(np.repeat(U[:nc], 2)))   # flat prep
    ones = np.ones(ns); zeros = np.zeros(ns)   # flat prep
    keys, vecs, info = {}, {}, {}   # flat prep
    for ch in chs:   # flat prep
        H = ch.clocked_dense(ones, zeros)   # flat prep
        Hc = H[:2 * nc, :2 * nc] + Ud   # flat prep
        del H   # flat prep
        e, V = torch.linalg.eigh(Hc)   # flat prep: the cavity is small (2 nc); dense under every --eigh
        del Hc   # flat prep
        e_ = e.real.cpu().numpy().astype(float)   # flat prep
        pos = e_ > 0   # flat prep
        inband = e_ >= m * (1.0 - 1e-9)   # flat prep: levels below the band bottom m (a hard cut's edge modes, if any) dropped and counted
        idx = np.where(inband)[0][:400]   # flat prep: the engine's 400 levels per channel
        if not len(idx):   # flat prep
            raise RuntimeError("flat prep: the box spectrum of channel %d has no level at or above m" % ch.kappa)   # flat prep
        Vk = torch.zeros((2 * ns, len(idx)), dtype=V.dtype, device=V.device)   # flat prep
        Vk[:2 * nc] = V[:, torch.as_tensor(idx, device=V.device)]   # flat prep: embedded with zeros beyond the wall
        del V   # flat prep
        keys[ch.kappa] = e_[idx]; vecs[ch.kappa] = Vk   # flat prep
        info[ch.kappa] = {"positive_levels": int(pos.sum()), "edge_modes_dropped": int(np.sum(pos & ~inband)), "lowest": float(e_[idx[0]]), "kept": int(len(idx))}   # flat prep
    FLAT.update(R_in=float(R_in), R_wall=float(R_wall), E_k=Ek, nc=nc, Ud=Ud, keys=keys, vecs=vecs, info=info, NB=float(NB))   # flat prep


def flat_channel(ch, spec):   # flat prep: prepare_dust's first channel loop: the cavity's keys and orbitals; the projector is made in flat_proj
    spec[ch.kappa] = (FLAT["keys"][ch.kappa], FLAT["vecs"][ch.kappa], None, "flat")   # flat prep: "flat" stands for the cavity's hamiltonian
    return FLAT["keys"][ch.kappa]   # flat prep


def flat_hcav(ch):   # flat prep: the cavity's hamiltonian on the whole chain (block diagonal: the box and the outside decoupled), for the stationarity read
    ns = len(ch.R); nc = FLAT["nc"]   # flat prep
    H = ch.clocked_dense(np.ones(ns), np.zeros(ns))   # flat prep
    H[:2 * nc, 2 * nc:] = 0; H[2 * nc:, :2 * nc] = 0   # flat prep
    H[:2 * nc, :2 * nc] += FLAT["Ud"]   # flat prep
    return H   # flat prep


def flat_proj(ch, N, v, Pplus, hw):   # flat prep: prepare_dust's second channel loop, for a channel that holds dust (the default returns the same objects)
    if not (isinstance(hw, str) and hw == "flat"):   # flat prep
        return Pplus, hw   # flat prep
    em, Um = eigT(ch.clocked_dense(N, v))   # flat prep: the positive band of h_m[N, v], as the well's filling projects on it
    Pplus = Um[:, em > 0]   # flat prep
    del Um   # flat prep
    return Pplus, flat_hcav(ch)   # flat prep


def flat_head(head, chs, dust, m, R, N, v, eps, J):   # flat prep: the head's prep and flatbox entries (nothing by default)
    if not FLAT["on"]:   # flat prep
        return   # flat prep
    Rr = np.asarray(R, float); NB = float(head["N_B"]); Rb = FLAT["R_b"]   # flat prep
    nsite = np.zeros(len(Rr)); cat = {}; top = 0; EF = m; ekin = 0.0   # flat prep
    for ic, ch in enumerate(chs):   # flat prep
        if ch.kappa not in dust:   # flat prep
            continue   # flat prep
        Phi, fk = dust[ch.kappa]   # flat prep
        p_ = (Phi.abs() ** 2).real.cpu().numpy()   # flat prep
        nsite += ch.w * (p_[0::2] + p_[1::2]) @ np.asarray(fk, float)   # flat prep: the count per site (both components)
        key = FLAT["keys"][ch.kappa]   # flat prep
        E_ = np.asarray(key[:len(fk)], float)   # flat prep: the kept columns are the lowest levels of the channel (fill_lowest fills a channel from its bottom)
        ekin += ch.w * float(np.asarray(fk, float) @ (E_ - m)); EF = max(EF, float(E_.max())); top = max(top, ch.kappa)   # flat prep
        cat[str(ch.kappa)] = {"columns": int(Phi.shape[1]), "f": [float(x) for x in fk], "particles": float(ch.w * np.sum(fk)), "E_box": [float(x) for x in E_],   # flat prep
                              "lowest_f": float(np.min(fk)), "w": ch.w, "n_WKB": iso_wkb(ch.kappa, iso_ball(8, NB, Rb, m)["weyl"]["k_F_R0"])}   # flat prep
    tot = float(nsite.sum()); cum = np.cumsum(nsite) / max(tot, 1e-300)   # flat prep
    rq = lambda q: float(Rr[min(int(np.searchsorted(cum, q)), len(Rr) - 1)])   # flat prep
    nvol = nsite / (4.0 * math.pi * Rr * Rr)   # flat prep: the count per unit volume (one site = one radial shell of thickness a)
    Rin = FLAT["R_in"]; win = (Rr >= 0.2 * Rin) & (Rr <= 0.7 * Rin)   # flat prep: the interior window
    n_in = float(np.mean(nvol[win])); n_TF = NB / (4.0 * math.pi * Rb ** 3 / 3.0)   # flat prep
    j_ = np.arange(len(Rr))   # flat prep
    r90 = float(Rr[int(j_[(nvol >= 0.9 * n_in) & (Rr <= FLAT["R_wall"])].max())])   # flat prep: the outermost site at 90 percent of the interior density
    r10 = float(Rr[int(j_[(nvol >= 0.1 * n_in) & (Rr <= FLAT["R_wall"])].max())])   # flat prep
    ball = iso_ball(8, NB, Rb, m); ktop = max(ch.kappa for ch in chs)   # flat prep
    fr = iso_frame(NB, m, N, v, eps, J)   # flat prep: the dust's frame figures (v36n ISO_fix_v1b)
    st = 4; pr = Rr <= 1.25 * FLAT["R_wall"]   # flat prep: the profile every 4 sites to 1.25 R_wall
    head["prep"] = "flatbox"   # flat prep
    head["flatbox"] = {"label": "flat prep: " + FLAT["desc"], "edge": FLAT["edge"], "R_b": Rb, "W": FLAT["W"], "R_in": Rin, "R_wall": FLAT["R_wall"], "E_k_taper": FLAT["E_k"],   # flat prep
                       "box_sites": FLAT["nc"], "key_info": {str(k): q for k, q in FLAT["info"].items()}, "channels": cat, "top_channel": top, "sea_top": ktop,   # flat prep
                       "truncated_by_sea_set": bool(top >= ktop), "E_F": EF, "k_F_R_b_catalog": math.sqrt(max(EF * EF - m * m, 0.0)) * Rb, "ball": ball,   # flat prep
                       "kappa_max_chain_rule": C_ISO_CHAIN * NB ** (1.0 / 3.0), "f_kin_on_box_key": ekin / (NB * m), "f_kin_orbitals": head["f_kin"],   # flat prep
                       "count": tot, "R_10": rq(0.1), "R_50": rq(0.5), "R_90": rq(0.9), "R_99": rq(0.99), "n_interior": n_in, "n_TF": n_TF, "n_interior_over_TF": n_in / n_TF,   # flat prep
                       "n_interior_rel_rms": float(np.std(nvol[win]) / n_in), "n_interior_min_max": [float(nvol[win].min() / n_in), float(nvol[win].max() / n_in)],   # flat prep
                       "interior_window": [0.2 * Rin, 0.7 * Rin], "R_n90": r90, "R_n10": r10, "edge_width_90_10": r10 - r90,   # flat prep
                       "count_beyond_R_b": float(nsite[Rr > Rb].sum()), "frame": fr, "dust_momentum_total": head["dust_momentum_total"],   # flat prep
                       "profile_R": [float(x) for x in Rr[pr][::st]], "profile_n_over_interior": [float(x) for x in (nvol / n_in)[pr][::st]]}   # flat prep


def flat_print(head):   # flat prep: one printed line after the initial data (nothing by default)
    if not FLAT["on"]:   # flat prep
        return   # flat prep
    c = head["flatbox"]   # flat prep
    print("flat prep: %s; N_B %.6f; top channel %d of the sea set's %d%s; k_F R_b %.3f (E_F %.6f) against the ball's %.3f (Weyl) and %.3f (surface); kappa_max' %.2f; f_kin %.5f on the orbitals, %.5f on the box key; interior density %.5f of Thomas-Fermi's (rel rms %.2e, min %.4f, max %.4f on [%.0f, %.0f] a); count radii R_50 %.1f R_90 %.1f R_99 %.1f a; density 90 percent at %.1f a, 10 percent at %.1f a (edge %.1f a); beyond R_b %.3e; box %d sites, edge modes %d; dust momentum %.3e; per channel: %s" % (   # flat prep
        c["label"], head["N_B"], c["top_channel"], c["sea_top"], " TRUNCATED BY THE SEA SET" if c["truncated_by_sea_set"] else "", c["k_F_R_b_catalog"], c["E_F"],   # flat prep
        c["ball"]["weyl"]["k_F_R0"], c["ball"]["surface"]["k_F_R0"], c["kappa_max_chain_rule"], c["f_kin_orbitals"], c["f_kin_on_box_key"], c["n_interior_over_TF"],   # flat prep
        c["n_interior_rel_rms"], c["n_interior_min_max"][0], c["n_interior_min_max"][1], c["interior_window"][0], c["interior_window"][1], c["R_50"], c["R_90"], c["R_99"],   # flat prep
        c["R_n90"], c["R_n10"], c["edge_width_90_10"], c["count_beyond_R_b"], c["box_sites"], sum(q["edge_modes_dropped"] for q in c["key_info"].values()), c["dust_momentum_total"],   # flat prep
        "; ".join("kappa %s: %d columns (WKB %d), %.3f particles, lowest f %.4f" % (k, q["columns"], q["n_WKB"], q["particles"], q["lowest_f"]) for k, q in c["channels"].items())), flush=True)   # flat prep


def iter_cache_drop(a, cache):
''')))

EDITS.append(("C begin", blk(r'''
    iso_begin(chs, R, R0, m)                                                    # version 36n (nothing by default)
'''), blk(r'''
    iso_begin(chs, R, R0, m)                                                    # version 36n (nothing by default)
    flat_begin(chs, R, m, NB)   # flat prep (nothing by default)
''')))

EDITS.append(("D first loop", blk(r'''
        for ic, ch in enumerate(chs):
            hw = ch.clocked_dense(N, v if reading == "A" else 0 * v) + Ud
'''), blk(r'''
        for ic, ch in enumerate(chs):
            if FLAT["on"]:   # flat prep: the box's keys and orbitals in place of the well's (the well's two eigendecompositions are not made)
                ewp = flat_channel(ch, spec)   # flat prep
                levels += [(float(x), ch.w, ic, j) for j, x in enumerate(ewp[:400])]   # flat prep: the well's line, on the box's levels
                continue   # flat prep
            hw = ch.clocked_dense(N, v if reading == "A" else 0 * v) + Ud
''')))

EDITS.append(("E second loop", blk(r'''
            fk = iso_scale(fk)                                                           # version 36n (the same array by default)
            raw = Pplus @ (Pplus.conj().T @ Uwv[:, keep])
'''), blk(r'''
            fk = iso_scale(fk)                                                           # version 36n (the same array by default)
            Pplus, hw = flat_proj(ch, N, v, Pplus, hw)   # flat prep (the same objects by default)
            raw = Pplus @ (Pplus.conj().T @ Uwv[:, keep])
''')))

EDITS.append(("F head", blk(r'''
    iso_head(head, chs, dust, m, N, v, eps, J)                                                    # version 36n (nothing by default)
'''), blk(r'''
    iso_head(head, chs, dust, m, N, v, eps, J)                                                    # version 36n (nothing by default)
    flat_head(head, chs, dust, m, R, N, v, eps, J)   # flat prep (nothing by default)
''')))

EDITS.append(("G label", blk(r'''
    label = iso_label(label, a)                                                     # version 36n (the same label by default)
'''), blk(r'''
    label = iso_label(label, a)                                                     # version 36n (the same label by default)
    label = flat_label(label)   # flat prep (the same label by default)
''')))

EDITS.append(("G print", blk(r'''
        iso_print(head)                                                                 # version 36n (nothing by default)
'''), blk(r'''
        iso_print(head)                                                                 # version 36n (nothing by default)
        flat_print(head)   # flat prep (nothing by default)
''')))

EDITS.append(("H options", blk(r'''
    a = ap.parse_args()
    if a.fields != "ivp" or a.fields_warm_start != "off":   # rd2 fields
'''), blk(r'''
    ap.add_argument("--prep", dest="prep", default="well", choices=["well", "flatbox"], help="(flat prep) the dust orbitals: the levitating well's levels (the default) or the lowest levels of the box of radius --prep-R")   # flat prep
    ap.add_argument("--prep-R", dest="prep_R", type=float, default=None, help="(flat prep) the box radius R_b in units of r_s (default --R0-over-rs)")   # flat prep
    ap.add_argument("--prep-edge", dest="prep_edge", default="hard", choices=["hard", "taper"], help="(flat prep) hard: the lattice's wall at R_b; taper: U = E_k ((R - R_b + W)/W)^2 beyond R_b - W, the wall at R_b + W")   # flat prep
    ap.add_argument("--prep-taper", dest="prep_taper", type=float, default=None, help="(flat prep) the taper width W in units of r_s (with --prep-edge taper)")   # flat prep
    a = ap.parse_args()
    if a.fields != "ivp" or a.fields_warm_start != "off":   # rd2 fields
''')))

EDITS.append(("I configure", blk(r'''
    iso_configure(a, ap)                                                            # version 36n
'''), blk(r'''
    iso_configure(a, ap)                                                            # version 36n
    flat_configure(a, ap)   # flat prep
''')))


def md5(b):
    return hashlib.md5(b).hexdigest()


def forward(text):
    for name, old, new in EDITS:
        c = text.count(old)
        if c != 1:
            raise SystemExit("edit %s: the OLD anchor occurs %d times (must be 1); it begins: %r" % (name, c, old.splitlines()[0][:100]))
        text = text.replace(old, new)
    return text


def reverse(text):
    for name, old, new in reversed(EDITS):
        c = text.count(new)
        if c != 1:
            raise SystemExit("reverse %s: the NEW block occurs %d times (must be 1)" % (name, c))
        text = text.replace(new, old)
    return text


def marker_check():
    """every line a NEW block adds or changes carries MARK; a line of OLD is not new (nor is one moved out from behind the opening triple
    quote); blank lines are neutral."""
    bad, n_marked = [], 0
    for name, old, new in EDITS:
        olds = set(old.splitlines())
        olds |= {l[3:] for l in old.splitlines() if l.startswith('"""')}
        for ln in new.splitlines():
            if ln in olds or ln.strip() == "":
                continue
            if MARK in ln:
                n_marked += 1
            else:
                bad.append((name, ln))
    return bad, n_marked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC_NAME)
    ap.add_argument("--dst", default=None, help="default: %s beside --src" % OUT_NAME)
    ap.add_argument("--reverse", action="store_true", help="src is %s: restore %s into dst" % (OUT_NAME, SRC_NAME))
    ap.add_argument("--expect", default=None, help="(reverse) the file the restored text must equal byte for byte")
    ap.add_argument("--expect-src-md5", default=SRC_MD5_PREFIX, help="(build) the prefix the source's md5 must start with (default %s; '' to skip)" % SRC_MD5_PREFIX)
    ap.add_argument("--check-only", action="store_true")
    a = ap.parse_args()
    if a.dst is None:
        a.dst = os.path.join(os.path.dirname(os.path.abspath(a.src)), OUT_NAME if not a.reverse else "engine_collapse_v36nq_rd13mf_restored.py")
    src = open(a.src, "rb").read()
    text = src.decode("utf-8")
    if a.reverse:
        out = reverse(text).encode("utf-8")
        if not a.check_only:
            if os.path.exists(a.dst):
                raise SystemExit("%s exists; nothing written (never overwritten)" % a.dst)
            open(a.dst, "wb").write(out)
        print("reverse: %s %s (md5 %s)" % ("would write" if a.check_only else "wrote", a.dst, md5(out)))
        if a.expect:
            ok = out == open(a.expect, "rb").read()
            print("restored equals %s byte for byte: %s" % (a.expect, ok))
            raise SystemExit(0 if ok else 1)
        return
    print("source %s: md5 %s, %d bytes" % (a.src, md5(src), len(src)))
    if a.expect_src_md5 and not md5(src).startswith(a.expect_src_md5):
        raise SystemExit("the source's md5 does not start with %s: not %s; nothing written" % (a.expect_src_md5, SRC_NAME))
    if MARK in text:
        raise SystemExit("the source already carries %r; nothing written" % MARK)
    for name, old, new in EDITS:
        print("  anchor %-14s occurs %d time(s) in the input (%d -> %d lines)" % (name, text.count(old), len(old.splitlines()), len(new.splitlines())))
    bad, n_marked = marker_check()
    if bad:
        for name, ln in bad:
            print("  UNMARKED new line in %s: %r" % (name, ln))
        raise SystemExit("new lines without the marker")
    print("every new or changed line carries %r (%d marked lines)" % (MARK, n_marked))
    built = forward(text)
    out = built.encode("utf-8")
    back = reverse(built).encode("utf-8")
    print("reversible (NEW -> OLD restores the input byte for byte): %s" % (back == src))
    if back != src:
        raise SystemExit("the reverse pass does not restore the input; nothing written")
    try:
        compile(built, a.dst, "exec")
    except SyntaxError as e:
        raise SystemExit("the result does not compile (%s); nothing written" % e)
    if a.check_only:
        print("check only: %s would be %d bytes, md5 %s" % (a.dst, len(out), md5(out)))
        return
    if os.path.exists(a.dst):
        if open(a.dst, "rb").read() == out:
            print("%s exists and equals the build byte for byte (md5 %s); not rewritten" % (a.dst, md5(out)))
            return
        raise SystemExit("%s exists and differs from the build; nothing written (never overwritten)" % a.dst)
    open(a.dst, "wb").write(out)
    written = open(a.dst, "rb").read()
    if written != out or reverse(written.decode("utf-8")).encode("utf-8") != src:
        raise SystemExit("the written file does not reverse to the input")
    py_compile.compile(a.dst, doraise=True)
    n_lines = sum(1 for ln in built.splitlines() if MARK in ln)
    print("wrote %s: %d bytes, md5 %s; compiles; the reverse of the written file equals the source byte for byte" % (a.dst, len(out), md5(out)))
    print("check: grep -c '%s' %s  ->  %d" % (MARK, a.dst, n_lines))


if __name__ == "__main__":
    main()
