#!/usr/bin/env python
"""page_own_object_v1.py (reviewer A for the solve builder, the busi lane). Inbox busi #144 item (6), word for word:
"PAGE_own_object_v1: on every stored slice of the r_s/a 64, τ 8, σ/a 4 history (12 to 40 and the continuation's slices) and on the σ/a 8
member's slices, the heavy field's exterior-cut entanglement entropy at the cuts 16 and 24 r_s per engine channel with multiplicity (the
exterior Gram matrix of the occupied columns through s, the reorientation's recipe), the held Killing vacuum's at the same cut beside it,
and the dust number and energy flux through the cuts with the Killing split at m; beside them the light channel's exterior-cut entropy from
page_history_v6n at matched clocks and the 2 fields' sum; the chain end's first-return estimate beside every row. Post the slice format for
second_route through the lead. Same folder; this does not print until OWNER item 2 is decided."

Every output of this module is marked "does not print until OWNER item 2 is decided": the json, the npz and the tables.

Per slice and per engine channel kappa (weight 16 kappa; the sums 'with multiplicity' beside the per-channel rows):
- S_X(C) at the cuts (16, 24 r_s by --cuts-rs): page_d13_readout_v1b.exterior_entropies on the Loewdin-orthonormalized occupied columns
  Phi (C = Phi Phi^dagger pure): Tr s(C_X) from the eigenvalues of the exterior Gram matrix Phi_X^dagger Phi_X (or the block Phi_X Phi_X^dagger,
  whichever is smaller), the complement's S_Xc beside it (equal on a pure state: the purity gap printed), the held Killing vacuum
  P_- = 1(h_m < 0) at the slice's applied fields (page_d13_readout_v1b.killing_vacuum, one dense eigendecomposition of h_m per channel; its
  eigenvalues checked against the stored e_k) and its S_X^vac at the same cut, the excess S_X - S_X^vac;
- the cross-cut bound t_X = ||Pi_X C (1 - Pi_X)||_1 = sum_i sqrt(lambda_i (1 - lambda_i)) over the exterior block's spectrum, with
  S_X <= (pi C*/2) t_X (h(p) <= C* arcsin sqrt(min(p, 1 - p)) <= (pi C*/2) sqrt(p(1 - p)), C* = page_ops_v6.C_STAR) and the sharper per-mode
  ledger page_ops_v6.pair_ledger_sharp(sqrt(min(lambda, 1 - lambda))); page_d13_readout_v1b.t_X is called where the module exposes it and
  its value printed beside (the route named);
- the Killing energy beyond the cut E_X^C = Re Tr[Pi_X h_m C] and the vacuum's, their difference (page_d13_readout_v1b.inventories);
- the dust number and energy flux through the cuts with the Killing split at m: the dust content beyond the cut in the exterior block's
  Killing eigenbasis (interior_continuation_v1.dust_killing_split: N_out, E_direct, the part at omega >= m (unbound content above m), below
  m, at omega <= -m) and the dust's instantaneous number and Killing-energy currents through the cut (page_ops_v6.number_current_csr,
  current_out_csr; outward positive); compose adds the coarse flux between consecutive slice clocks, Delta(content)/Delta T, split at m, as
  the mass-loss law against the clock at infinity; the deviation's (C - P_-) number current beside;
- the light channel's exterior-cut entropy at the matched clock from page_history_v6n's outputs (--light-dir: directories searched for the
  D13 store jsons *_d13.json; the clock within --match-tol-rs; the weighted sum over the light channel's channels where the store carries
  it, else the sum of 16 kappa S_X over its per_channel entries) where they exist, and the 2 fields' sum S_heavy + S_light (the product-state
  identity of the 2 independent fields at the Hartree level); 'not yet available' where no output matches;
- the chain end's first-return estimate (page_d13_readout_v1b.first_return: the light round trip cut -> chain end -> cut in the slice's
  applied fields from the run's start and from the formation's root) beside every row and cut.
Outputs per slice: page_<history>_T<clock>.json and .npz (the exterior spectra of C and P_- per cut and channel, the dust's Killing spectra);
compose writes PAGE_own_object_v1_<history>.md (marked not for print) and PAGE_own_object_v1_all.json. CPU only; no plotting; no mutual
information (D15); no dates."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import slices144_io_v1 as IO  # noqa: E402
import interior_continuation_v1 as IC  # noqa: E402
import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

READOUT = "page"
TITLE = "PAGE_own_object_v1"
CODE = "page_own_object_v1.py"
PRINT_STATUS = "does not print until OWNER item 2 is decided"
D15 = "no mutual information is computed or printed (D15)"


def add_arguments(ap):
    g = ap.add_argument_group("page_own_object_v1")
    g.add_argument("--light-dir", dest="light_dir", default="auto", help="comma list of directories holding page_history_v6n's *_d13.json stores (auto: D/page_v6n, D/page_v6m)")
    g.add_argument("--match-tol-rs", dest="match_tol_rs", type=float, default=0.0625, help="the light channel's clock must lie within this of the slice's (r_s)")
    g.add_argument("--vac-route", dest="vac_route", default="dense", choices=["dense"], help="the Killing vacuum's route (dense eigh; the massive chain has no pair structure)")
    g.add_argument("--trace-chunk", dest="trace_chunk", type=int, default=1024)


def is_done(entry, a):
    return os.path.exists(IO.out_paths(a, READOUT, entry)[0])


def cross_cut_bound(ev):
    """t_X and the 2 bounds from the exterior block's spectrum of a pure covariance."""
    lam = np.clip(np.asarray(ev, float), 0.0, 1.0)
    s2 = lam * (1.0 - lam)
    t = float(np.sum(np.sqrt(s2)))
    cstar = float(getattr(IO.OPS, "C_STAR", 1.325486838698363))
    small = np.minimum(lam, 1.0 - lam)
    ledger = float(IO.OPS.pair_ledger_sharp(np.sqrt(small))) if hasattr(IO.OPS, "pair_ledger_sharp") else None
    return {"t_X": t, "bound_pi_Cstar_over_2_t_X": 0.5 * math.pi * cstar * t, "bound_pair_ledger_sharp": ledger, "C_star": cstar, "modes_mixed_above_1e-12": int(np.sum(s2 > 1e-12)),
            "definition": "t_X = ||Pi_X C (1 - Pi_X)||_1 = sum_i sqrt(lambda_i (1 - lambda_i)) over the exterior block's spectrum (C pure); S_X = sum_i h(lambda_i) <= (pi C*/2) t_X; the per-mode ledger page_ops_v6.pair_ledger_sharp(sqrt(min(lambda, 1 - lambda))) is the sharper bound"}


def t_X_from_module(ev):
    fn = getattr(IO.D13, "t_X", None) if IO.D13 is not None else None
    if fn is None:
        return {"value": None, "route": "page_d13_readout_v1b exposes no t_X: the local definition only"}
    for args, lab in (((np.asarray(ev, float),), "t_X(eig_X numpy)"), ((IO.torch.as_tensor(np.asarray(ev, float)),), "t_X(eig_X tensor)")):
        try:
            r = fn(*args)
            if IO.torch.is_tensor(r):
                r = r.item() if r.numel() == 1 else r.cpu().numpy().tolist()
            elif isinstance(r, dict):
                r = {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in r.items()}
            else:
                r = float(r)
            return {"value": r, "route": "page_d13_readout_v1b.%s" % lab}
        except Exception:  # noqa: BLE001
            continue
    return {"value": None, "route": "page_d13_readout_v1b.t_X is present but its signature matched none of the tried calls: the local definition only"}


# ---------------------------------------------------------------------------------------------------------- the light channel
def light_index(dirs, log):
    idx = []
    for d in dirs:
        if not d or not os.path.isdir(d):
            continue
        for root, _, files in os.walk(d):
            for fn in files:
                if fn.endswith("_d13.json"):
                    p = os.path.join(root, fn)
                    try:
                        with open(p) as f:
                            J = json.load(f)
                    except Exception:  # noqa: BLE001
                        continue
                    for key, c in (J.get("clocks") or {}).items():
                        try:
                            t = float(c.get("t_over_rs", key))
                        except (TypeError, ValueError):
                            continue
                        idx.append({"t": t, "file": p, "clock": c})
    log("light channel: %d clock entries in %d directories (%s)" % (len(idx), len([d for d in dirs if d and os.path.isdir(d)]), ", ".join(d for d in dirs if d and os.path.isdir(d)) or "none present"))
    return idx


def light_at(idx, trs, tol, labels):
    cands = [x for x in idx if abs(x["t"] - trs) <= tol]
    if not cands:
        return None
    t_best = min(abs(x["t"] - trs) for x in cands)
    same = [x for x in cands if abs(abs(x["t"] - trs) - t_best) < 1e-12]
    out = {"t_over_rs": same[0]["t"], "files": sorted({x["file"] for x in same}), "cuts": {}, "channels": []}
    per = {}
    weighted = {}
    for x in same:
        c = x["clock"]
        w = c.get("weighted_over_state_channels") or {}
        for lab in labels:
            if ("S_X_cut%s" % lab) in w:
                weighted.setdefault(lab, {"S_X": 0.0, "excess": 0.0, "S_X_vac": 0.0, "route": "weighted_over_state_channels"})
                weighted[lab]["S_X"] = float(w["S_X_cut%s" % lab])
                weighted[lab]["excess"] = float(w.get("excess_cut%s" % lab, 0.0) or 0.0)
                weighted[lab]["S_X_vac"] = weighted[lab]["S_X"] - weighted[lab]["excess"]
        for kap, pc in (c.get("per_channel") or {}).items():
            per[str(kap)] = pc
    out["channels"] = sorted(per, key=lambda s: int(s) if s.isdigit() else 0)
    for lab in labels:
        if lab in weighted:
            out["cuts"][lab] = weighted[lab]
            continue
        sx = 0.0
        sv = 0.0
        ok = False
        for kap, pc in per.items():
            cu = (pc.get("cuts") or {}).get(lab)
            if cu and cu.get("S_X") is not None:
                w_ = 16.0 * int(kap) if kap.isdigit() else 1.0
                sx += w_ * float(cu["S_X"])
                sv += w_ * float(cu.get("S_X_vac") or 0.0)
                ok = True
        out["cuts"][lab] = {"S_X": sx, "S_X_vac": sv, "excess": sx - sv, "route": "sum of 16 kappa S_X over the per_channel entries"} if ok else None
    return out


# ------------------------------------------------------------------------------------------------------------ the slice
def process_slice(rd, ctx, entry, a):
    if IO.D13 is None:
        raise RuntimeError("page_own_object_v1 needs page_d13_readout_v1b (exterior_entropies, killing_vacuum, inventories, first_return): not importable (%s)" % IO.SETUP.get("D13"))
    t0 = time.time()
    log = ctx.log
    torch = IO.torch
    dev = torch.device("cpu")
    D13 = IO.D13
    D13.TRACE_CHUNK[0] = int(a.trace_chunk)
    pj, pz = IO.out_paths(a, READOUT, entry)
    rec = ctx.record_at(entry, rd.T)
    m, g, rs = ctx.m, ctx.g, rd.rs
    R, N_app, v_app = rd.R, rd.N_applied, rd.v_applied
    cuts = [(lab, x * rs) for lab, x in IO.cuts_from_arg(a.cuts_rs)]
    labels = [lab for lab, _ in cuts]
    row = IO.slice_header(rd, ctx, entry, rec, READOUT, TITLE, CODE)
    row["print_status"] = PRINT_STATUS
    row["d15"] = D15
    E_rec = rec.get("E")
    T_H = 1.0 / (8.0 * math.pi * g * float(E_rec)) if E_rec else None
    row["record"] = {"E": E_rec, "two_gE": 2.0 * g * float(E_rec) if E_rec else None, "T_H": T_H, "m_over_TH": (m / T_H) if T_H else None, "R_t": rec.get("R_t"), "trapping_zeros": rec.get("trapping_zeros"),
                     "min_N_minus_absv": rec.get("min_N_minus_absv"), "E_over_E0": rec.get("E_over_E0")}
    arrays = {"R": R, "N_applied": N_app, "v_applied": v_app}
    R2_t = torch.as_tensor(np.repeat(R, 2))
    form = ctx.formation(entry)
    row["formation"] = form
    row["first_return"] = {}
    for lab, x in cuts:
        fr = D13.first_return(N_app, v_app, R, x, rd.T, 0.0, rs, form["T"] if form else None, form["r_root"] if form else None)
        row["first_return"][lab] = fr
    chans = []
    wsum = {lab: {} for lab in labels}
    for kap in rd.kappas:
        t1 = time.time()
        ch = rd.channel(kap, orthonormalize=(a.loewdin == "on"))
        Phi, f = ch["Phi"], ch["f"]
        w = ch["weight"]
        h0, Q, H = IO.operators(R, m, kap, N_app, v_app)
        out = {"kappa": int(kap), "weight": w, "n_sea": ch["n_sea"], "n_dust": ch["n_dust"], "stored_orthonormality": ch["stored_orthonormality"], "loewdin_orthonormality": ch["loewdin_orthonormality"],
               "orthonormalization": ch["orthonormalization"], "dust_sea_overlap_max": ch["dust_sea_overlap_max"]}
        vac = D13.killing_vacuum(H, dev, a.vac_route)
        Vneg = vac["Vneg"]
        out["vacuum"] = {"n_neg": vac["n_neg"], "E_ref": vac["E_ref"], "route": vac["route"], "seconds": vac["seconds"], "n_zero_1e-12": vac.get("n_zero_1e-12"),
                         "eig_vs_stored": float(np.max(np.abs(np.sort(np.asarray(vac["e"], float)) - np.sort(ch["e"])))) if ch["e"] is not None else None}
        Phi_t = torch.as_tensor(Phi)
        ent = D13.exterior_entropies(Phi_t, R2_t, cuts, Vneg)
        Bh = IO.OPS.Banded(H, dev)
        inv = D13.inventories(Bh, Phi_t, R2_t, cuts, Vneg, a.trace_chunk)
        dust, f_d = rd.dust(kap)
        out["cuts"] = {}
        for lab, x in cuts:
            d = ent[lab]
            for key in ("eig_X", "eig_Xc", "eig_X_vac", "eig_Xc_vac"):
                if isinstance(d.get(key), np.ndarray):
                    arrays["k%d_cut%s_%s" % (kap, lab, key)] = d.pop(key)
            evX = arrays["k%d_cut%s_eig_X" % (kap, lab)]
            cc = cross_cut_bound(evX)
            cc["module"] = t_X_from_module(evX)
            cc["bound_holds"] = bool(d["S_X"] <= cc["bound_pi_Cstar_over_2_t_X"] + 1e-9)
            d["cross_cut"] = cc
            d["energy"] = inv[lab]
            d["dust"] = None
            if dust.shape[1]:
                sp, arr = IC.dust_killing_split(H, R, dust, f_d, x, m, a.m_split_at, T_H)
                sp["number_current"] = IC.cut_current(H, R, x, dust, f_d, "number")
                sp["energy_current"] = IC.cut_current(H, R, x, dust, f_d, "energy")
                d["dust"] = sp
                arrays["k%d_cut%s_dust_omega" % (kap, lab)] = arr["omega"]
                arrays["k%d_cut%s_dust_n" % (kap, lab)] = arr["n_dust"]
            d["currents"] = {"C_number": IC.cut_current(H, R, x, Phi, f, "number"), "vac_number": IC.cut_current(H, R, x, Vneg.numpy(), np.ones(Vneg.shape[1]), "number"),
                             "C_energy": IC.cut_current(H, R, x, Phi, f, "energy"), "vac_energy": IC.cut_current(H, R, x, Vneg.numpy(), np.ones(Vneg.shape[1]), "energy")}
            d["currents"]["deviation_number"] = d["currents"]["C_number"] - d["currents"]["vac_number"]
            d["currents"]["deviation_energy"] = d["currents"]["C_energy"] - d["currents"]["vac_energy"]
            out["cuts"][lab] = d
            ws = wsum[lab]
            for key, val in (("S_X", d["S_X"]), ("S_Xc", d["S_Xc"]), ("S_X_vac", d["S_X_vac"]), ("excess", d["excess"]), ("t_X", cc["t_X"]), ("bound_pi_Cstar_over_2_t_X", cc["bound_pi_Cstar_over_2_t_X"]),
                             ("bound_pair_ledger_sharp", cc["bound_pair_ledger_sharp"]), ("E_C", inv[lab]["E_C"]), ("E_vac", inv[lab]["E_vac"]), ("E_D", inv[lab]["E_D"]),
                             ("deviation_number_current", d["currents"]["deviation_number"]), ("deviation_energy_current", d["currents"]["deviation_energy"])):
                if val is not None:
                    ws[key] = ws.get(key, 0.0) + w * float(val)
            ws["max_abs_purity_gap"] = max(ws.get("max_abs_purity_gap", 0.0), abs(d["purity_gap"]))
            if d["dust"]:
                for key in ("N_out", "E_direct", "E_block", "N_above_m", "E_above_m", "N_below_m", "E_below_m", "N_neg_m", "E_neg_m", "number_current", "energy_current"):
                    ws["dust_" + key] = ws.get("dust_" + key, 0.0) + w * float(d["dust"][key])
                ws["dust_max_abs_completeness"] = max(ws.get("dust_max_abs_completeness", 0.0), abs(d["dust"]["completeness_check"]))
        out["seconds"] = time.time() - t1
        chans.append(out)
        c16 = out["cuts"][labels[0]]
        log("  kappa %d: vacuum %d occupied (eig vs stored %s, %.0f s); beyond %s r_s S_X %.6f (complement gap %.1e), vacuum %.6f, excess %+.6f, t_X %.4f (bound %.4f); E_D %+.4e; dust N_out %s (>= m %s), current %s; %.0f s" % (
            kap, vac["n_neg"], IO.fx(out["vacuum"]["eig_vs_stored"], "%.1e"), vac["seconds"], labels[0], c16["S_X"], c16["purity_gap"], c16["S_X_vac"], c16["excess"], c16["cross_cut"]["t_X"],
            c16["cross_cut"]["bound_pi_Cstar_over_2_t_X"], c16["energy"]["E_D"], IO.fx((c16["dust"] or {}).get("N_out"), "%.4e"), IO.fx((c16["dust"] or {}).get("N_above_m"), "%.4e"),
            IO.fx((c16["dust"] or {}).get("number_current"), "%+.3e"), out["seconds"]))
        del Phi, Phi_t, Vneg, vac, Bh, dust, ch
    row["channels"] = chans
    row["weighted"] = {lab: dict(wsum[lab], weights="16 kappa", channels=rd.kappas) for lab in labels}
    # the light channel at the matched clock
    dirs = [os.path.join(a.box_D, "page_v6n"), os.path.join(a.box_D, "page_v6m")] if a.light_dir == "auto" else [x for x in a.light_dir.split(",") if x]
    idx = light_index(dirs, log)
    lc = light_at(idx, rd.T_over_rs, a.match_tol_rs, labels)
    row["light_channel"] = lc if lc else {"status": "not yet available: no page_history_v6n output within %g r_s of T/r_s %.3f in %s" % (a.match_tol_rs, rd.T_over_rs, dirs)}
    row["two_fields_sum"] = {}
    for lab in labels:
        hv = row["weighted"][lab].get("S_X")
        lv = (lc or {}).get("cuts", {}).get(lab) if lc else None
        row["two_fields_sum"][lab] = {"heavy": hv, "light": lv["S_X"] if lv else None, "sum": (hv + lv["S_X"]) if (lv and hv is not None) else None,
                                      "heavy_excess": row["weighted"][lab].get("excess"), "light_excess": lv["excess"] if lv else None,
                                      "identity": "the 2 fields' exterior-cut entropies add (independent fields at the Hartree level: D16's product-state identity)"}
    row["seconds"] = time.time() - t0
    IO.save_outputs(pj, pz, row, arrays)
    w16 = row["weighted"][labels[0]]
    log("%s [%s] T/r_s %.3f (%s; %s): beyond %s r_s weighted S_X %.6f, vacuum %.6f, excess %+.6f (purity gap max %.1e), t_X bound %.4f; E_D %+.5e; dust N_out %s (>= m %s) E_direct %s current %s; light %s; first return %s r_s; %.0f s" % (
        TITLE, PRINT_STATUS, rd.T_over_rs, rd.dtype_str, entry["history"], labels[0], w16.get("S_X", 0.0), w16.get("S_X_vac", 0.0), w16.get("excess", 0.0), w16.get("max_abs_purity_gap", 0.0),
        w16.get("bound_pi_Cstar_over_2_t_X", 0.0), w16.get("E_D", 0.0), IO.fx(w16.get("dust_N_out"), "%.4e"), IO.fx(w16.get("dust_N_above_m"), "%.4e"), IO.fx(w16.get("dust_E_direct"), "%+.4e"),
        IO.fx(w16.get("dust_number_current"), "%+.3e"), IO.fx(((lc or {}).get("cuts", {}).get(labels[0]) or {}).get("S_X") if lc else None, "%.6f") if lc else "not yet available",
        IO.fx(row["first_return"][labels[0]].get("first_return_rs"), "%.1f"), row["seconds"]))
    return row


# ------------------------------------------------------------------------------------------------------------ the tables
def compose(a, hists):
    rows = IO.load_rows(a, READOUT, hists)
    if not rows:
        return
    labels = sorted({lab for r in rows for lab in (r.get("weighted") or {})}, key=float)
    L = ["# %s: the model's object's own exterior-cut entropy and mass loss on the stored slices (%s) -- %s" % (TITLE, CODE, PRINT_STATUS.upper()), "",
         "Inbox busi #144 item (6). %s. %s." % (PRINT_STATUS, D15), "",
         "- S_X: the heavy field's exterior-cut entanglement entropy Tr s(C_X) from the exterior Gram matrix of the Loewdin-orthonormalized occupied columns (page_d13_readout_v1b.exterior_entropies; the complement's S_Xc equals it on the pure state: the purity gap is the check); per channel and summed with 16 kappa; the held Killing vacuum P_- = 1(h_m < 0) at the slice's applied fields and its S_X^vac beside, the excess S_X - S_X^vac.",
         "- t_X = ||Pi_X C (1 - Pi_X)||_1 = sum sqrt(lambda(1 - lambda)) over the exterior block's spectrum; S_X <= (pi C*/2) t_X with C* = page_ops_v6.C_STAR (the sharper per-mode ledger page_ops_v6.pair_ledger_sharp in the json); page_d13_readout_v1b.t_X beside where exposed.",
         "- E_D: the Killing energy beyond the cut of C - P_- (page_d13_readout_v1b.inventories); the dust beyond the cut in the exterior block's Killing eigenbasis (N_out, E_direct, the part at omega >= %g m = the unbound content above m), its instantaneous number and energy currents through the cut (outward positive), and the coarse flux between consecutive slice clocks Delta(content)/Delta T (the mass-loss law against the clock at infinity)." % a.m_split_at,
         "- the light channel: page_history_v6n's exterior-cut entropy at the matched clock (within %g r_s) where its output exists; the 2 fields' sum by the product-state identity; 'not yet available' otherwise." % a.match_tol_rs,
         "- the first return: page_d13_readout_v1b.first_return in the slice's applied fields (the light round trip cut -> chain end -> cut from the run's start and from the formation's root; the earlier); an estimate, not a measured arrival; a row past it reads the chain end's return.", ""]
    for hk in hists:
        rs_ = [r for r in rows if r["history"] == hk]
        if not rs_:
            continue
        L += ["## %s: %s -- %s" % (hk, rs_[0]["history_label"], PRINT_STATUS), "", "Engine: %s" % rs_[0]["engine"], ""]
        for lab in labels:
            L += ["### Beyond %s r_s (summed with 16 kappa)" % lab, "",
                  "| T/r_s | dtype | E/E0 | S_X heavy | S_X vacuum | excess | purity gap (max over channels) | t_X; (pi C*/2) t_X | E_D | dust: N_out; E_direct | dust N (omega >= m); E (omega >= m) | dust currents: number; energy | coarse flux to the next slice: dN/dT; dE/dT; dN(>= m)/dT | light S_X (clock) | heavy + light | first return (r_s) |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
            for i, r in enumerate(rs_):
                w = r["weighted"].get(lab)
                if not w:
                    continue
                nxt = rs_[i + 1] if i + 1 < len(rs_) else None
                flux = "-"
                if nxt and nxt["weighted"].get(lab) and w.get("dust_N_out") is not None and nxt["weighted"][lab].get("dust_N_out") is not None:
                    dT = (nxt["T_over_rs"] - r["T_over_rs"]) * r["rs"]
                    w2 = nxt["weighted"][lab]
                    flux = "%+.3e; %+.3e; %+.3e (to %.1f)" % ((w2["dust_N_out"] - w["dust_N_out"]) / dT, (w2["dust_E_direct"] - w["dust_E_direct"]) / dT, (w2["dust_N_above_m"] - w["dust_N_above_m"]) / dT, nxt["T_over_rs"])
                lc = r.get("light_channel") or {}
                lcu = (lc.get("cuts") or {}).get(lab) if lc.get("cuts") else None
                fr = r["first_return"].get(lab, {})
                L.append("| %.3f | %s | %s | %.6f | %.6f | %+.6f | %.1e | %.4f; %.4f | %+.6e | %s; %s | %s; %s | %s; %s | %s | %s | %s | %s%s |" % (
                    r["T_over_rs"], r["dtype"], IO.fx(r["record"].get("E_over_E0"), "%.6f"), w.get("S_X", 0.0), w.get("S_X_vac", 0.0), w.get("excess", 0.0), w.get("max_abs_purity_gap", 0.0), w.get("t_X", 0.0),
                    w.get("bound_pi_Cstar_over_2_t_X", 0.0), w.get("E_D", 0.0), IO.fx(w.get("dust_N_out"), "%.5e"), IO.fx(w.get("dust_E_direct"), "%+.5e"), IO.fx(w.get("dust_N_above_m"), "%.5e"),
                    IO.fx(w.get("dust_E_above_m"), "%+.5e"), IO.fx(w.get("dust_number_current"), "%+.3e"), IO.fx(w.get("dust_energy_current"), "%+.3e"), flux,
                    ("%.6f (%.3f)" % (lcu["S_X"], lc["t_over_rs"])) if lcu else "not yet available", IO.fx(r["two_fields_sum"].get(lab, {}).get("sum"), "%.6f"),
                    IO.fx(fr.get("first_return_rs"), "%.1f"), "" if fr.get("before_first_return") in (None, True) else " (past the first return)"))
            L.append("")
        lab = labels[0]
        L += ["### Per channel beyond %s r_s (unweighted; the multiplicity 16 kappa applies to the sums above)" % lab, "",
              "| T/r_s | kappa | S_X | S_Xc | S_X vacuum | excess | t_X | n_neg (eig vs stored e_k) | Loewdin (stored) | E_C; E_vac; E_D | dust N_out; N (>= m); E_direct; current | deviation number current |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in rs_:
            for c in r["channels"]:
                d = c["cuts"][lab]
                du = d.get("dust") or {}
                L.append("| %.3f | %d | %.6f | %.6f | %.6f | %+.6f | %.4f | %d (%s) | %s (%.1e) | %+.5e; %+.5e; %+.5e | %s; %s; %s; %s | %+.3e |" % (
                    r["T_over_rs"], c["kappa"], d["S_X"], d["S_Xc"], d["S_X_vac"], d["excess"], d["cross_cut"]["t_X"], c["vacuum"]["n_neg"], IO.fx(c["vacuum"]["eig_vs_stored"], "%.1e"),
                    IO.fx(c["loewdin_orthonormality"], "%.1e"), c["stored_orthonormality"], d["energy"]["E_C"], d["energy"]["E_vac"], d["energy"]["E_D"],
                    IO.fx(du.get("N_out"), "%.4e"), IO.fx(du.get("N_above_m"), "%.4e"), IO.fx(du.get("E_direct"), "%+.4e"), IO.fx(du.get("number_current"), "%+.3e"), d["currents"]["deviation_number"]))
        L += ["", "### Checks", ""]
        for r in rs_:
            L.append("- T/r_s %.3f: purity gap max %.1e; vacuum eigenvalues against the stored e_k max %s; Loewdin orthonormality after %s (stored %s); dust completeness max %s; the cross-cut bound holds on every channel: %s; the vacuum's occupied count %s against the sea's %d columns; %s; %.0f s." % (
                r["T_over_rs"], max(r["weighted"][l_].get("max_abs_purity_gap", 0.0) for l_ in labels),
                IO.fx(max([c["vacuum"]["eig_vs_stored"] for c in r["channels"] if c["vacuum"].get("eig_vs_stored") is not None] or [None]), "%.1e") if any(c["vacuum"].get("eig_vs_stored") is not None for c in r["channels"]) else "-",
                IO.fx(max((c["loewdin_orthonormality"] or 0.0) for c in r["channels"]), "%.1e"), IO.fx(max(c["stored_orthonormality"] for c in r["channels"]), "%.1e"),
                IO.fx(max([r["weighted"][l_].get("dust_max_abs_completeness", 0.0) for l_ in labels]), "%.1e"),
                all(c["cuts"][l_]["cross_cut"]["bound_holds"] for c in r["channels"] for l_ in labels), sorted({c["vacuum"]["n_neg"] for c in r["channels"]}), r["channels"][0]["n_sea"],
                "complex64 slice: floor %s" % r["floor"]["source_rel_range"] if r["complex64"] else "complex128", r["seconds"]))
        L.append("")
    L += ["- every number above is printed by the script from the stored slices and the records; %s; the reading is the lead's." % PRINT_STATUS]
    d = os.path.join(a.out, READOUT)
    for hk in hists:
        if any(r["history"] == hk for r in rows):
            with open(os.path.join(d, "%s_%s.md" % (TITLE, hk)), "w") as f:
                f.write("\n".join(L) + "\n")
    with open(os.path.join(d, "%s_all.json" % TITLE), "w") as f:
        json.dump({"code": CODE, "title": TITLE, "print_status": PRINT_STATUS, "d15": D15, "histories": hists, "rows": rows}, f, default=IO.jdefault, indent=1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    IO.add_common_arguments(ap)
    IC.add_arguments(ap)
    add_arguments(ap)
    a = ap.parse_args(argv)
    a.readouts = READOUT
    IO.run(a)


if __name__ == "__main__":
    main()
