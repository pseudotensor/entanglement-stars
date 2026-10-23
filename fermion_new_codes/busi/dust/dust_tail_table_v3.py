#!/usr/bin/env python
"""dust_tail_table_v3.py (busi lane, reviewer A): the delivery's table of the dust's second route, version 3, composed from the outputs only:
<out-dir>/<tag>_all.json (dust_tail_v3.py's) and the first route's jsons. v3: the first route's D v2 jsons (engine_massive_channel_D_v2.py's
layout, slices -> channels -> regions[cut]) are read at their nested paths (regions[cut].hull.S_hull, regions[cut].sector[eps_loc].dim,
.dim_plus, .dim_minus, .S_F, .E_F, .particles, .holes, .F_orthonormality, .outside_weight_min_*; regions[cut].S_F_loc_control,
.S_hull_loc_control, .particles_Pm, .holes_Pm, .particles_dust, .holes_dust, .dust_weight, .region_S_excess, .E_out_dust; the channel's
stored_orthonormality and loewdin_orthonormality; the slice's weighted[cut]); version 2's composer's pick() skipped dict-valued keys and
printed "-" for S_hull (defect 2 of its red team). Printed: the first route's S_hull, S_F, d_F, its sector count n_F = sector particles +
sector holes and the bound sum 16 kappa d_F h2(n_F/d_F) beside this route's at every slice of the run (V3-A, V3-C), the first route's own
numbers at every clock its jsons carry (V3-B; T/r_s 40 from Dv2_prod10.json when --first-route-t40-json names it: the dust's number, the
block-basis counts, the pairs' derived counts, what the dust's count permits in the sector), the floors and the floor-aware verdicts, version
2's localized set with the same verdict (V3-D), the checks (V3-E) and the Loewdin control when run (V3-F). Version 2's sections follow,
composed by dust_tail_table_v2.compose_v2 on the same json with the headers of V2-F, V2-G and V2-I relabelled (its "escaped sector F" is
version 2's localized set) and the first route's nested keys lifted to the flat names it looks up, so its first-route S_hull column prints.
Nothing is averaged. Usage: python dust_tail_table_v3.py --out-dir DIR --tag TAG [--first-route-tail-json F] [--first-route-dv2-jsons
a.json,b.json] [--first-route-t40-json Dv2_prod10.json] [--out FILE]; dust_tail_v3.run_main calls compose_from_outputs."""
import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dust_tail_v1 as V1  # noqa: E402  (load_first_route for the tail json and version 2's composer; fmt, utc)
import dust_tail_table_v2 as TBL2  # noqa: E402  (compose_v2: version 2's sections on the same json)

fmt = V1.fmt
utc = V1.utc
V2_SEP = ("--- version 2's sections follow, composed by dust_tail_table_v2.compose_v2 on the same json; the headers of V2-F, V2-G and V2-I are relabelled by "
          "dust_tail_table_v3 (version 2's 'escaped sector F' is its localized set F_loc, not the first route's escaped sector, which is V3-A above); the first "
          "route's nested keys hull.S_hull and sector[eps_loc].dim are lifted to the flat names S_hull and escaped_dim before version 2's composer runs, so its "
          "first-route S_hull column prints; the numbers are untouched ---")
RELABELS = (
    ("## V2-F. (ii) The escaped sector F (eps_loc",
     "## V2-F. (ii) Version 2's localized set F_loc: the individual eigenvectors of the whole chain's h_m with outside weight >= 1 - eps_loc beyond the cut (version 2's set; NOT the first route's escaped sector, which is table V3-A; eps_loc", True),
    ("the escaped sector's counts, S_F and bound, the region excess",
     "the localized set F_loc's counts, S_F and bound (version 2's set, not the first route's sector; its floor-aware verdict is table V3-D), the region excess", False),
    ("| F: F_+^dag P_- F_+ max; F_-^dag P_- F_- - 1 max;",
     "| F_loc (version 2's set): F_+^dag P_- F_+ max; F_-^dag P_- F_- - 1 max;", False),
)


# ------------------------------------------------------------------------------------------------------------------- helpers
def h2(p):
    """the binary entropy in nats (the same function as dust_tail_v2.h2; p clipped to [0, 1])."""
    p = min(max(float(p), 0.0), 1.0)
    out = 0.0
    if p > 0.0:
        out -= p * math.log(p)
    if p < 1.0:
        out -= (1.0 - p) * math.log(1.0 - p)
    return out


def bound_dh2(n, d):
    d = float(d)
    return d * h2(float(n) / d) if d > 0 else 0.0


def ck(key):
    return str(key).rstrip("0").rstrip(".")


def fi(x):
    return "-" if x is None else "%d" % int(x)


def f6(x):
    return fmt(x, "%.6e")


def f4(x):
    return fmt(x, "%.4f")


def f1(x):
    return fmt(x, "%.1e")


def diff(x, y, f="%+.3e"):
    return fmt((x - y) if (x is not None and y is not None) else None, f)


def minnone(a, b):
    vals = [v for v in (a, b) if v is not None]
    return min(vals) if vals else None


def num(x):
    if x is None or isinstance(x, (dict, list, str, bool)):
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def dig(d, *path):
    """descend a nested dict along path; each step tries the key as given, then its %g, float and trimmed forms ('16' and '16.0'; '0.01' and
    '1e-02'); None when any step is absent."""
    cur = d
    for p in path:
        if not isinstance(cur, dict):
            return None
        cands = [str(p)]
        try:
            fp = float(p)
            cands += ["%g" % fp, str(fp), ck(p)]
        except (TypeError, ValueError):
            pass
        found = False
        for c in cands:
            if c in cur:
                cur = cur[c]
                found = True
                break
        if not found:
            return None
    return cur


# ------------------------------------------------------------------------------------------------------------------- the first route's jsons
def read_first_route_raw(paths):
    """the first route's D v2 jsons (engine_massive_channel_D_v2.py's layout) keyed by T/r_s: {T: {"file", "weighted": slice.weighted,
    "channels": {kappa: channel dict}, "eps_loc", "eps_loc_control", "eps_rank", "eps_control" (its args), "E_G", "dtype"}}."""
    out = {}
    for pth in [p for p in (paths or "").split(",") if p]:
        with open(pth) as fh:
            J = json.load(fh)
        args = J.get("args") or {}
        for s in (J.get("slices") or []):
            T = round(float(s["T_over_rs"]), 6)
            out[T] = {"file": os.path.basename(pth), "weighted": s.get("weighted") or {}, "channels": {int(c["kappa"]): c for c in (s.get("channels") or [])},
                      "eps_loc": float(args.get("eps_loc", 0.01)), "eps_loc_control": float(args.get("eps_loc_control", 1e-3)), "eps_rank": float(args.get("eps_rank", 1e-9)),
                      "eps_control": float(args.get("eps_control", 1e-6)), "E_G": s.get("E_G"), "dtype": s.get("dtype")}
    return out


def first_channel_sector(ch, cut, eps_loc, eps_rank):
    """the first route's per-channel numbers at the nested keys of its json for one cut (None when the region is absent); the bound d_F h2(n_F/d_F),
    the ratio, the pairs' derived counts and what the dust's count permits are formed here from them."""
    if not isinstance(ch, dict):
        return None
    reg = dig(ch, "regions", cut)
    if not isinstance(reg, dict):
        return None
    sectors = reg.get("sector") if isinstance(reg.get("sector"), dict) else {}
    sec = dig(sectors, "%g" % eps_loc) or {}
    others = [sectors[q_] for q_ in sectors if q_ != ("%g" % eps_loc) and isinstance(sectors[q_], dict)]
    secc = others[0] if others else {}
    hull = reg.get("hull") if isinstance(reg.get("hull"), dict) else (dig(sec, "hulls", "%g" % eps_rank) or {})
    kappa = ch.get("kappa")
    d = {"kappa": kappa, "w": num(ch.get("w")) if num(ch.get("w")) is not None else (16.0 * float(kappa) if kappa is not None else None),
         "S_hull": num(hull.get("S_hull")), "hull_dim": hull.get("hull_dim"), "kept": hull.get("kept"), "dropped": hull.get("dropped"),
         "dropped_bound": num(hull.get("dropped_bound")), "vacuum_check": num(hull.get("vacuum_check")), "svd_fallback": hull.get("svd_fallback"),
         "S_F": num(sec.get("S_F", reg.get("S_F"))), "E_F": num(sec.get("E_F", reg.get("E_F"))), "d_F": sec.get("dim"), "d_plus": sec.get("dim_plus"), "d_minus": sec.get("dim_minus"),
         "sector_particles": num(sec.get("particles", reg.get("sector_particles"))), "sector_holes": num(sec.get("holes", reg.get("sector_holes"))),
         "F_orthonormality": num(sec.get("F_orthonormality")), "outside_weight_min": minnone(num(sec.get("outside_weight_min_plus")), num(sec.get("outside_weight_min_minus"))),
         "S_F_loc_control": num(reg.get("S_F_loc_control", secc.get("S_F"))), "S_hull_loc_control": num(reg.get("S_hull_loc_control")), "d_F_loc_control": secc.get("dim"),
         "particles_Pm": num(reg.get("particles_Pm")), "holes_Pm": num(reg.get("holes_Pm")), "particles_dust": num(reg.get("particles_dust")), "holes_dust": num(reg.get("holes_dust")),
         "dust_weight": num(reg.get("dust_weight")), "region_S_excess": num(reg.get("region_S_excess")), "E_out_dust": num(reg.get("E_out_dust")),
         "stored_orthonormality": num(ch.get("stored_orthonormality")), "loewdin_orthonormality": num(ch.get("loewdin_orthonormality"))}
    both = d["sector_particles"] is not None and d["sector_holes"] is not None
    d["n_F"] = (d["sector_particles"] + d["sector_holes"]) if both else None
    d["bound_F"] = bound_dh2(d["n_F"], d["d_F"]) if (d["n_F"] is not None and d["d_F"]) else None
    d["ratio"] = (d["S_hull"] / d["bound_F"]) if (d["S_hull"] is not None and d["bound_F"]) else None
    d["pairs_particles"] = (d["particles_Pm"] - d["particles_dust"]) if (d["particles_Pm"] is not None and d["particles_dust"] is not None) else None
    d["pairs_holes"] = (d["holes_Pm"] - d["holes_dust"]) if (d["holes_Pm"] is not None and d["holes_dust"] is not None) else None
    d["dust_net"] = (d["particles_dust"] + d["holes_dust"]) if (d["particles_dust"] is not None and d["holes_dust"] is not None) else None
    d["dust_permits"] = bound_dh2(d["dust_weight"], d["d_F"]) if (d["dust_weight"] is not None and d["d_F"]) else None
    return d


def first_slice_sector(frs, cut, eps_loc=None, eps_rank=None):
    """the first route's weighted numbers at one cut from its per-channel json entries (w = its channel's w), the bound as the per-channel sum,
    the ratio's range over channels, the one-dimension forms for the dust's count, and the json's own weighted keys with their difference from
    the per-channel sums (a check that the read is complete)."""
    if not frs:
        return None
    eps_loc = frs.get("eps_loc", 0.01) if eps_loc is None else eps_loc
    eps_rank = frs.get("eps_rank", 1e-9) if eps_rank is None else eps_rank
    chs = [c for c in (first_channel_sector(frs["channels"][kappa], cut, eps_loc, eps_rank) for kappa in sorted(frs["channels"])) if c is not None]
    if not chs:
        return None

    def W(name):
        vals = [(c["w"], c.get(name)) for c in chs]
        return float(sum(w * v for w, v in vals)) if all(v is not None for _, v in vals) and all(w is not None for w, _ in vals) else None

    out = {name: W(name) for name in ("S_hull", "S_F", "E_F", "sector_particles", "sector_holes", "n_F", "bound_F", "dropped_bound", "particles_Pm", "holes_Pm", "particles_dust",
                                       "holes_dust", "pairs_particles", "pairs_holes", "dust_net", "dust_weight", "region_S_excess", "S_F_loc_control", "S_hull_loc_control", "E_out_dust", "dust_permits")}
    out["n_channels"] = len(chs)
    alld = all(c["d_F"] is not None for c in chs)
    out["d_F_unw"] = int(sum(int(c["d_F"]) for c in chs)) if alld else None
    out["d_F_w"] = float(sum(c["w"] * int(c["d_F"]) for c in chs)) if alld else None
    out["hull_dim_unw"] = int(sum(int(c["hull_dim"]) for c in chs)) if all(c["hull_dim"] is not None for c in chs) else None
    out["d_F_loc_control_unw"] = int(sum(int(c["d_F_loc_control"]) for c in chs)) if all(c["d_F_loc_control"] is not None for c in chs) else None
    out["ratio"] = (out["S_hull"] / out["bound_F"]) if (out["S_hull"] is not None and out["bound_F"]) else None
    rat = [c["ratio"] for c in chs if c["ratio"] is not None]
    out["ratio_min"] = min(rat) if rat else None
    out["ratio_max"] = max(rat) if rat else None
    out["dust_permits_w_form"] = bound_dh2(out["dust_weight"], out["d_F_w"]) if (out["dust_weight"] is not None and out["d_F_w"]) else None
    out["dust_permits_review_form"] = bound_dh2(out["dust_weight"], out["d_F_unw"]) if (out["dust_weight"] is not None and out["d_F_unw"]) else None
    out["max_stored_orthonormality"] = max((c["stored_orthonormality"] for c in chs if c["stored_orthonormality"] is not None), default=None)
    out["max_loewdin_orthonormality"] = max((c["loewdin_orthonormality"] for c in chs if c["loewdin_orthonormality"] is not None), default=None)
    wj = dig(frs.get("weighted") or {}, cut) or {}
    out["json_weighted"] = {q_: num(wj.get(q_)) for q_ in ("S_hull", "S_F", "sector_particles", "sector_holes", "dust_weight", "particles_Pm", "holes_Pm", "E_F", "region_S_excess",
                                                           "dropped_bound", "S_F_loc_control", "S_hull_loc_control")}
    out["sum_minus_json"] = {q_: ((out.get(q_) - out["json_weighted"][q_]) if (out.get(q_) is not None and out["json_weighted"][q_] is not None) else None) for q_ in out["json_weighted"]}
    out["channels"] = chs
    out["eps_loc"] = eps_loc
    out["eps_rank"] = eps_rank
    return out


def lift_nested_keys(first, eps_loc=0.01):
    """v3: the first route's nested keys lifted to the flat names version 2's composer looks up (hull.S_hull -> S_hull; sector[eps_loc].dim ->
    escaped_dim), in the dict dust_tail_v1.load_first_route returned; returns the number of regions lifted."""
    n = 0
    for fd in (first.get("dv2") or {}).values():
        if not isinstance(fd, dict):
            continue
        for chd in fd.values():
            if not isinstance(chd, dict):
                continue
            for reg in (chd.get("regions") or {}).values():
                if not isinstance(reg, dict):
                    continue
                hull = reg.get("hull")
                if isinstance(hull, dict) and hull.get("S_hull") is not None and "S_hull" not in reg:
                    reg["S_hull"] = hull["S_hull"]
                    n += 1
                sec = dig(reg, "sector", "%g" % eps_loc)
                if isinstance(sec, dict) and sec.get("dim") is not None and "escaped_dim" not in reg:
                    reg["escaped_dim"] = sec["dim"]
    return n


def relabel_v2(md2, dev):
    """v3: the headers of version 2's sections relabelled; a loud failure when the expected header text is not found exactly once."""
    out = md2
    for old, new, always in RELABELS:
        n = out.count(old)
        if n != 1:
            if always or (dev and n == 0) or n > 1:
                raise RuntimeError("dust_tail_table_v3: version 2's header text %r occurs %d times; the relabelling of version 2's sections failed" % (old[:48], n))
            continue
        out = out.replace(old, new)
    return out


# ------------------------------------------------------------------------------------------------------------------- the v3 sections
def compose_v3(J, first, fr, all_json_name="<tag>_all.json", t40_name="", lifted=0):
    rows = sorted([r for r in J.get("rows", []) if "v3" in r], key=lambda r: r["T_over_rs"])
    prov = J.get("provenance") or {}
    if not rows:
        return "# dust_tail_table_v3: no rows with v3 readouts in the outputs\n"
    cuts_rs = J.get("cuts_rs") or [16.0, 24.0]
    keys = [str(float(c)) for c in cuts_rs]
    v30 = rows[0]["v3"]
    eps = float(v30["eps_loc"])
    eps_c = v30.get("eps_loc_control")
    eps_rank = float(v30["eps_rank"])
    eps_ctl = float(v30["eps_control"])
    dev2 = any((r.get("v2") or {}).get("deviation") for r in rows)
    lw = bool(v30.get("loewdin"))
    files = ", ".join(sorted({frs["file"] for frs in fr.values()})) if fr else "none"
    mism = sorted({frs["eps_loc"] for frs in fr.values() if abs(frs["eps_loc"] - eps) > 1e-12})
    L = ["# The shed dust's second route, version 3 (dust_tail_v3.py, busi lane; web40g task 1): the first route's escaped sector rebuilt per spectral half, S_F and S_hull with the bound d_F h2(n_F/d_F) and the floors, beside the first route's own S_hull, d_F, counts and bound at every clock its jsons carry", "",
         "Composed %s by dust_tail_table_v3.py from %s (the outputs of dust_tail_v3.py) and the first route's jsons (%s; read at their nested keys, named in the provenance); every number printed by the script; the first route's numbers stand beside this route's and are never averaged with them. Weighted = summed over the channels with w = 16 kappa; unweighted = the sum over the 10 chains without the copies. "
         "F = the first route's escaped sector: in each spectral half of h_m (e >= 0 and e < 0) the eigenvectors of the half's outside-weight operator V_h^dag Pi_Y V_h with eigenvalue >= 1 - eps_loc, eps_loc = %g (control %s), F = [F_+, F_-], d_F = d_+ + d_- per chain; C = S S^dag + D f D^dag (the stored columns as stored); M = F^dag C F; S_F = tr s(M); n_+ = Tr F_+^dag C F_+, n_- = Tr F_-^dag (1 - C) F_-; "
         "E_F = Re tr[(F^dag h_m F)(M - Pi)]; S_hull = the first route's hull step in the sector's coordinates (the eigenvectors of M - Pi with |lambda| > eps_rank = %g, every left singular vector of each half block; control eps %g). "
         "Floors: eta = the larger of the stored columns' Gram defect max|Phi^dag Phi - 1| (Phi = [S, D]) and the eigensolver's (max|F^dag F - 1|, F_+^dag P_- F_+, F_-^dag P_- F_- - 1); n_floor = d_F eta; S_floor = d_F h2(eta); 'respected' or 'EXCEEDED' is printed only when S_F > S_floor and n_F > n_floor, else 'below floor'. "
         "s(x) = -x ln x - (1 - x) ln(1 - x); entropies in nats; energies in units of 1/a. Version 2's set of single eigenvectors (its 'escaped sector F') is called the localized set F_loc here; it is not the first route's sector.%s" % (
             utc(), all_json_name, files, eps, fmt(eps_c, "%g"), eps_rank, eps_ctl, (" The first route's jsons carry eps_loc %s, this run used %g: the sector columns are then not the same construction." % (", ".join("%g" % x for x in mism), eps)) if mism else ""), ""]
    # V3-A
    L += ["## V3-A. The escaped sector F (the first route's construction) beyond each cut, weighted: the counts split into the dust's and the pairs', S_F, the bound sum 16k d_F h2(n_F,k/d_F,k), the weighted floors, the verdicts, S_hull and E_F; beside them the first route's own S_hull, S_F, d_F, sector count n_F = sector particles + sector holes, bound (sum 16k d_F h2(n_F/d_F) on its json keys) and ratio, and the differences this minus first", "",
          "| T/r_s | cut r_s | d_F: unw; w (d_+, d_- per chain; smallest kept outside weight) | n_+ = Tr F_+^dag C F_+ (dust; pairs) | n_- = Tr F_-^dag (1 - C) F_- (pairs; the dust's occupation of F_-, subtracted) | n_F; the pairs' share | S_F (nats) | bound sum 16k d_F h2(n_F,k/d_F,k) | S_F/bound (max over channels) | floors (weighted): n_floor; S_floor | verdict (channels respected / EXCEEDED / below floor) | S_hull (hull dim unw; dropped bound) | E_F | first route: S_hull; S_F; d_F unw; n_F; bound; S_hull/bound (channels min to max); dropped bound | this - first: S_F - S_hull; n_F - n_F; d_F - d_F (unw) |",
          "|" + "---|" * 15]
    for r in rows:
        T = round(r["T_over_rs"], 6)
        frs = fr.get(T)
        for key in keys:
            q = r["v3"]["weighted"][key]
            secs = [c3["sector"][key] for c3 in r["v3"]["channels"]]
            dp = ",".join(str(x) for x in sorted({s["dim_plus"] for s in secs}))
            dm = ",".join(str(x) for x in sorted({s["dim_minus"] for s in secs}))
            fs = first_slice_sector(frs, ck(key))
            if fs:
                fcell = "%s; %s; %s; %s; %s; %s (%s to %s); %s" % (f6(fs["S_hull"]), f6(fs["S_F"]), fi(fs["d_F_unw"]), f6(fs["n_F"]), f6(fs["bound_F"]), f4(fs["ratio"]), f4(fs["ratio_min"]), f4(fs["ratio_max"]), f1(fs["dropped_bound"]))
                dcell = "%s; %s; %s" % (diff(q["S_F"], fs["S_hull"]), diff(q["n_F"], fs["n_F"]), fi((q["d_F_unw"] - fs["d_F_unw"]) if fs["d_F_unw"] is not None else None))
            else:
                fcell, dcell = "-", "-"
            vd = q["verdicts"]
            L.append("| %.3f | %s | %d; %g (%s, %s; %s) | %.6e (%.6e; %.6e) | %.6e (%.6e; %.6e) | %.6e; %s | %.6e | %.6e | %s (%s) | %.1e; %.1e | %s (%d / %d / %d) | %.6e (%d; %.1e) | %+.6e | %s | %s |" % (
                r["T_over_rs"], ck(key), q["d_F_unw"], q["d_F_w"], dp, dm, f4(q["outside_weight_min"]), q["n_plus"], q["n_plus_dust"], q["n_plus_pairs"], q["n_minus"], q["n_minus_pairs"], q["n_minus_dust_occ"],
                q["n_F"], f4(q["pairs_share_of_n_F"]), q["S_F"], q["bound_F"], f4(q["S_F_over_bound_F"]), f4(q["max_S_F_over_bound_F"]), q["n_floor_w"], q["S_floor_w"], q["verdict"],
                vd["respected"], vd["EXCEEDED"], vd["below floor"], q["S_hull"], q["hull_dim_unw"], q["dropped_bound"], q["E_F"], fcell, dcell))
    L.append("")
    L.append("The block form sum 16k [d_+ h2(n_+/d_+) + d_- h2(n_-/d_-)], the one-dimension forms d_w h2(n_F/d_w) and d_unw h2(n_F/d_unw), the dust-only S_F,dust with its rank bound, the hull at the control eps and the eps_loc control are in the json (v3.weighted[cut].bound_F_blocks, bound_F_w_form, bound_F_review_form, S_F_dust, bound_F_dust_rank, S_hull_control, control).")
    # V3-B
    L += ["", "## V3-B. The first route's own numbers at every clock its jsons carry (weighted with its w; read at the nested keys named in the provenance; nothing of this route's enters): S_hull and S_F in its escaped sector of dimension d_F, its sector count n_F = sector particles + sector holes, the bound sum 16k d_F h2(n_F/d_F) and the ratio, its Killing-block particles and holes of the whole deviation and of the dust (holes_dust is printed negative by the engine: the dust's occupation of negative levels fills holes), the pairs' derived counts, the dust's number, and what the dust's count alone permits in the sector (the per-channel sum of 16k d_F h2(N_k/d_F), the one-dimension forms with d_w = sum 16k d_F and d_unw = sum d_F); the json's own weighted S_hull and the per-channel sum's difference from it", "",
          "| T/r_s | cut r_s | file | S_hull (json's own weighted; sum minus it) | S_F; S_F at eps_loc control | d_F: unw; w (hull dim unw) | sector particles; holes; n_F | bound sum 16k d_F h2(n_F/d_F) | S_hull/bound (channels min to max) | dropped bound | Killing block of C - P_-: particles_Pm; holes_Pm | the dust's: particles_dust; holes_dust (signed) | the pairs' derived: particles_Pm - particles_dust; holes_Pm - holes_dust | dust_weight (tail json) | the dust's count permits: per-channel sum; d_w form (d_w); d_unw form (d_unw) | region S excess (comparator) | stored Gram - 1 (max); Loewdin (max) |",
          "|" + "---|" * 17]
    for T in sorted(fr):
        frs = fr[T]
        for key in keys:
            cut = ck(key)
            fs = first_slice_sector(frs, cut)
            if not fs:
                continue
            tail = (((first or {}).get("tail") or {}).get(T) or {}).get("beyond_%s" % cut) or {}
            jw = fs["json_weighted"]
            L.append("| %.3f | %s | %s | %s (%s; %s) | %s; %s | %s; %s (%s) | %s; %s; %s | %s | %s (%s to %s) | %s | %s; %s | %s; %s | %s; %s | %s (%s) | %s; %s (%s); %s (%s) | %s | %s; %s |" % (
                T, cut, frs["file"], f6(fs["S_hull"]), f6(jw["S_hull"]), diff(fs["S_hull"], jw["S_hull"]), f6(fs["S_F"]), f6(fs["S_F_loc_control"]), fi(fs["d_F_unw"]), fmt(fs["d_F_w"], "%g"), fi(fs["hull_dim_unw"]),
                f6(fs["sector_particles"]), f6(fs["sector_holes"]), f6(fs["n_F"]), f6(fs["bound_F"]), f4(fs["ratio"]), f4(fs["ratio_min"]), f4(fs["ratio_max"]), f1(fs["dropped_bound"]),
                f6(fs["particles_Pm"]), f6(fs["holes_Pm"]), f6(fs["particles_dust"]), fmt(fs["holes_dust"], "%+.6e"), f6(fs["pairs_particles"]), f6(fs["pairs_holes"]), f6(fs["dust_weight"]), f6(num(tail.get("dust_weight"))),
                f6(fs["dust_permits"]), f6(fs["dust_permits_w_form"]), fmt(fs["d_F_w"], "%g"), f6(fs["dust_permits_review_form"]), fi(fs["d_F_unw"]), f6(fs["region_S_excess"]), f1(fs["max_stored_orthonormality"]), f1(fs["max_loewdin_orthonormality"])))
    if not fr:
        L.append("(no first-route json given)")
    # V3-C
    L += ["", "## V3-C. Per channel (kappa, w = 16 kappa; both cuts): the sector's dimensions and smallest kept outside weight, the counts, S_F, the bound and ratio, the floors (eta_cols = the stored columns' Gram defect; eta_eig = the largest of F orth, F_+^dag P_- F_+, F_-^dag P_- F_- - 1), the verdict, S_hull (hull dim; kept; L reduces Pi; smallest singular value kept), E_F, S_F,dust with its rank bound, the eps_loc control; beside them the first route's per-channel S_hull, S_F, dim, sector particles and holes, stored and Loewdin orthonormality, and this S_F minus its S_hull", "",
          "| T/r_s | kappa | w | dust columns | cut r_s | d_F (d_+, d_-); w_min | n_+ (dust; pairs) | n_- | n_F | S_F | bound d_F h2(n_F/d_F) | S_F/bound | eta_cols; eta_eig (F orth; vac+; vac- - 1) | n_floor; S_floor | verdict | S_hull (hull dim; kept; L reduces Pi; s_min) | E_F | S_F,dust; rank bound | control eps_loc %s: d_F; S_F | first route: S_hull; S_F; dim; particles; holes; stored Gram - 1; Loewdin | this S_F - first S_hull |" % fmt(eps_c, "%g"),
          "|" + "---|" * 21]
    for r in rows:
        T = round(r["T_over_rs"], 6)
        frs = fr.get(T)
        for c3 in r["v3"]["channels"]:
            for key in keys:
                s = c3["sector"][key]
                sc = (c3.get("sector_control") or {}).get(key) or {}
                fl = s["floors"]
                hull = (s.get("hulls") or {}).get("%g" % eps_rank) or {}
                fc = first_channel_sector(frs["channels"].get(int(c3["kappa"])), ck(key), frs["eps_loc"], frs["eps_rank"]) if (frs and int(c3["kappa"]) in frs["channels"]) else None
                fcell = ("%s; %s; %s; %s; %s; %s; %s" % (f6(fc["S_hull"]), f6(fc["S_F"]), fi(fc["d_F"]), f6(fc["sector_particles"]), f6(fc["sector_holes"]), f1(fc["stored_orthonormality"]), f1(fc["loewdin_orthonormality"]))) if fc else "-"
                L.append("| %.3f | %d | %g | %d | %s | %d (%d, %d); %s | %.6e (%.6e; %.6e) | %.6e | %.6e | %.6e | %.6e | %s | %.1e; %.1e (%.1e; %.1e; %.1e) | %.1e; %.1e | %s | %s (%s; %s; %s; %s) | %+.6e | %.6e; %.6e | %s; %s | %s | %s |" % (
                    r["T_over_rs"], c3["kappa"], c3["w"], c3["n_dust"], ck(key), s["dim"], s["dim_plus"], s["dim_minus"], f4(minnone(s.get("outside_weight_min_plus"), s.get("outside_weight_min_minus"))),
                    s["n_plus"], s["n_plus_dust"], s["n_plus_pairs"], s["n_minus"], s["n_F"], s["S_F"], s["bound_F"], f4(s["S_F_over_bound_F"]), fl["eta_columns"], fl["eta_eigensolver"], s["F_orthonormality"],
                    s["vacuum_plus_overlap_max"], s["vacuum_minus_identity_dev"], fl["n_floor"], fl["S_floor"], s["verdict"], f6(s["S_hull"]), fi(s["hull_dim"]), fi(hull.get("kept")), f1(hull.get("vacuum_check")), f1(hull.get("s_min_kept")),
                    s["E_F"], s["S_F_dust"], s["bound_F_dust_rank"], fi(sc.get("dim")), f6(sc.get("S_F")), fcell, (diff(s["S_F"], fc["S_hull"]) if fc else "-")))
    # V3-D
    L += ["", "## V3-D. Version 2's localized set F_loc (the eigenvectors of the whole chain's h_m with weight >= 1 - eps_loc beyond the cut; version 2's dev entries, relabelled: not the first route's escaped sector) with the floor-aware verdict recomputed from its own json entries (eta per channel = the larger of the stored columns' Gram defect, F_+^dag P_- F_+ and F_-^dag P_- F_- - 1; n_floor = d_F eta; S_floor = d_F h2(eta)); weighted, with the per-channel verdicts counted and version 2's printed flag beside; the sector F's d_F, S_F and n_F for comparison%s" % ("" if dev2 else " (not computed: version 2's --deviation was off)"), ""]
    if dev2:
        L += ["| T/r_s | cut r_s | F_loc: d_F unw; w | n_F | S_F | bound sum 16k d_F h2(n_F,k/d_F,k) | S_F/bound (max over channels) | floors (weighted): n_floor; S_floor | verdicts: respected / EXCEEDED / below floor (version 2 printed) | the sector F: d_F unw; S_F; n_F | per channel (kappa: d_F, n_F, S_F, bound, n_floor, S_floor, verdict) |", "|" + "---|" * 11]
        for r in rows:
            for key in keys:
                dev = (((r.get("v2") or {}).get("weighted") or {}).get(key) or {}).get("dev")
                q = r["v3"]["weighted"][key]
                ls = q.get("localized_set") or {}
                if not dev:
                    continue
                per = []
                for c3 in r["v3"]["channels"]:
                    lv = (c3.get("loc_verdict") or {}).get(key)
                    if lv:
                        per.append("%d: %d, %.2e, %.2e, %.2e, %.1e, %.1e, %s" % (c3["kappa"], lv["d_F"], lv["n_F"], lv["S_F"], lv["bound_F"], lv["n_floor"], lv["S_floor"], lv["verdict"]))
                vd = ls.get("verdicts") or {}
                L.append("| %.3f | %s | %d; %g | %.6e | %.6e | %.6e | %s | %s; %s | %s / %s / %s (%s) | %d; %.6e; %.6e | %s |" % (
                    r["T_over_rs"], ck(key), dev["d_F_unw"], dev["d_F_w"], dev["n_F"], dev["S_F"], dev["bound_F"], f4(dev.get("max_S_F_over_bound_F")), f1(ls.get("n_floor_w")), f1(ls.get("S_floor_w")),
                    fi(vd.get("respected")), fi(vd.get("EXCEEDED")), fi(vd.get("below floor")), "yes" if dev.get("all_bound_F_respected") else "NO", q["d_F_unw"], q["S_F"], q["n_F"], "; ".join(per) or "-"))
    # V3-E
    L += ["", "## V3-E. Checks per slice (maxima over channels and cuts): the stored columns' Gram defect, the sector's orthonormality and vacuum identities, the hull's identities, the sector count against the block-basis count of version 2 (they differ at the localization level), the eigenvalues of h_m against version 2's decomposition and against the stored e_k, the stored sea's number beyond the cut minus the vacuum's (signed, from version 1's key sea_excess_beyond when present; version 1's table C prints its magnitude with a + sign), and the seconds", "",
          "| T/r_s | stored Gram - 1 | F^dag F - 1 | F_+^dag P_- F_+ | F_-^dag P_- F_- - 1 | hull: L reduces Pi; L^dag L - 1; abs(S_hull - S_F) | n_F - block count (beyond %s; %s) | eigenvalues: v3 vs v2; v3 vs stored; spectrum symmetry | sea excess beyond %s; %s (signed, min over channels) | seconds: v1 pass; v2 total from the slice's start; v3 pass; slice total |" % (ck(keys[0]), ck(keys[-1]), ck(keys[0]), ck(keys[-1])),
          "|" + "---|" * 10]
    for r in rows:
        v3 = r["v3"]
        qs = [v3["weighted"][k] for k in keys]
        sea_exc = []
        for key in keys:
            vals = []
            for ch in r.get("channels", []):
                se = ch.get("sea_excess_beyond")
                if isinstance(se, dict):
                    x = se.get(key, se.get(ck(key)))
                    if x is not None:
                        vals.append(float(x))
            sea_exc.append(("%+.1e" % min(vals)) if vals else "-")
        L.append("| %.3f | %.1e | %.1e | %.1e | %.1e | %.1e; %.1e; %.1e | %s | %s; %s; %s | %s | %.0f; %.0f; %.0f; %.0f |" % (
            r["T_over_rs"], max(q["max_stored_orthonormality"] for q in qs), max(q["max_F_orthonormality"] for q in qs), max(q["max_vacuum_plus_overlap"] for q in qs), max(q["max_vacuum_minus_identity_dev"] for q in qs),
            max(q["max_hull_vacuum_check"] for q in qs), max(q["max_hull_orthonormality"] for q in qs), max(q["max_abs_S_hull_minus_S_F"] for q in qs),
            "; ".join(fmt(q.get("n_F_minus_block_count"), "%+.3e") for q in qs), f1(v3.get("max_e_vs_v2")), f1(v3.get("max_e_vs_stored")), f1(v3.get("max_spectrum_symmetry")), "; ".join(sea_exc),
            float(r.get("seconds", 0.0)), float((r.get("v2") or {}).get("seconds", 0.0)), float(v3.get("seconds", 0.0)), float(v3.get("seconds_total", 0.0))))
    # V3-F
    if lw:
        L += ["", "## V3-F. The Loewdin control (--loewdin): the sector readouts on Phi G^(-1/2) (the first route's step on the stored columns) beside the as-stored ones, weighted; the first route's S_hull is computed on Loewdin-orthonormalized columns", "",
              "| T/r_s | cut r_s | S_F as stored | S_F Loewdin | Loewdin - stored | n_F as stored; Loewdin | E_F as stored; Loewdin | Loewdin orthonormality (max over channels) | first route S_hull | S_F Loewdin - first S_hull |", "|" + "---|" * 10]
        for r in rows:
            T = round(r["T_over_rs"], 6)
            for key in keys:
                q = r["v3"]["weighted"][key]
                lo = q.get("loewdin")
                if not lo:
                    continue
                fs = first_slice_sector(fr.get(T), ck(key))
                L.append("| %.3f | %s | %.6e | %.6e | %+.3e | %.6e; %.6e | %+.6e; %+.6e | %.1e | %s | %s |" % (
                    r["T_over_rs"], ck(key), q["S_F"], lo["S_F"], lo["S_F_minus_stored"], q["n_F"], lo["n_F"], q["E_F"], lo["E_F"], lo["max_loewdin_orthonormality"],
                    f6(fs["S_hull"]) if fs else "-", diff(lo["S_F"], fs["S_hull"]) if fs else "-"))
    # provenance
    L += ["", "## Provenance", ""] + ["- %s: %s" % (k, v) for k, v in prov.items()]
    L.append("- first route's jsons read by dust_tail_table_v3.read_first_route_raw: %s%s" % (files, ("; T/r_s 40 and earlier from %s" % os.path.basename(t40_name)) if t40_name else ""))
    L.append("- first route's nested keys read: regions[cut].hull.{S_hull, hull_dim, kept, dropped, dropped_bound, vacuum_check}; regions[cut].sector[%g].{dim, dim_plus, dim_minus, S_F, E_F, particles, holes, F_orthonormality, outside_weight_min_plus, outside_weight_min_minus}; regions[cut].{S_F, E_F, sector_particles, sector_holes, S_F_loc_control, S_hull_loc_control, particles_Pm, holes_Pm, particles_dust, holes_dust, dust_weight, region_S_excess, E_out_dust}; channel.{w, kappa, stored_orthonormality, loewdin_orthonormality}; slice.weighted[cut].{S_hull, S_F, sector_particles, sector_holes, particles_Pm, holes_Pm, dust_weight, E_F, region_S_excess, dropped_bound}; the first route's regions are keyed '16' and '24' and its sectors by eps_loc as %%g" % eps)
    L.append("- formed by this script from those keys (the red team's arithmetic, now printed): the first route's n_F = sector particles + sector holes, its bound sum 16k d_F h2(n_F/d_F) and ratios, the pairs' derived counts particles_Pm - particles_dust and holes_Pm - holes_dust, the dust's net particles_dust + holes_dust, and what the dust's count permits")
    L.append("- regions lifted for version 2's composer (hull.S_hull -> S_hull, sector[eps_loc].dim -> escaped_dim): %d" % lifted)
    L.append("- <tag>_table_v2.md is written unchanged beside this file for the comparison with version 2's run r2b; in it 'escaped sector F' names version 2's localized set")
    return "\n".join(L) + "\n"


def compose_from_outputs(out_dir, tag, tail_json, dv2_jsons, t40_json="", out_path=None):
    with open(os.path.join(out_dir, "%s_all.json" % tag)) as fh:
        J = json.load(fh)
    first = V1.load_first_route(tail_json, dv2_jsons)
    fr = read_first_route_raw(",".join(p for p in (dv2_jsons, t40_json) if p))
    rows3 = [r for r in J.get("rows", []) if "v3" in r]
    eps_loc = float(rows3[0]["v3"]["eps_loc"]) if rows3 else 0.01
    lifted = lift_nested_keys(first, eps_loc)
    md3 = compose_v3(J, first, fr, all_json_name="%s_all.json" % tag, t40_name=t40_json, lifted=lifted)
    dev2 = any((r.get("v2") or {}).get("deviation") for r in J.get("rows", []))
    TBL2.FOUND.clear()
    md2 = relabel_v2(TBL2.compose_v2(J, first, all_json_name="%s_all.json" % tag), dev2)
    md = md3 + "\n" + V2_SEP + "\n\n" + md2
    out_path = out_path or os.path.join(out_dir, "%s_table_v3.md" % tag)
    with open(out_path, "w") as fh:
        fh.write(md)
    return out_path, md


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True); ap.add_argument("--tag", required=True); ap.add_argument("--first-route-tail-json", default=""); ap.add_argument("--first-route-dv2-jsons", default="")
    ap.add_argument("--first-route-t40-json", default=""); ap.add_argument("--out", default="")
    a = ap.parse_args()
    out_path, _ = compose_from_outputs(a.out_dir, a.tag, a.first_route_tail_json, a.first_route_dv2_jsons, a.first_route_t40_json, a.out or None)
    print("wrote %s" % out_path)


if __name__ == "__main__":
    main()
