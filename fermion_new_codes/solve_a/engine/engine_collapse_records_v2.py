#!/usr/bin/env python
"""engine_collapse_records_v2.py (builder analyst; engine spec 24.1 (1), (11) and 24.2: COLLAPSE_records_v1.1 for a production made of
segments): version 1's tables (per record E/E0, the Gauss identity's error, per interval the Gauss balance dE against the evolution, reset
and lapse parts and its closure, the collar's Killing cut current; per record the stress checks, the evolution rows' residuals, the replay's
source differences and the masks) over consecutive segments of one history, each from its own run, handoff and rows: a segment is a
stored run (json and fields npz) with the handoff npz that carries its per-record shell quantities (an engine_replay_v2 replay for an
engine_collapse_v17 segment, the run's own for an engine_collapse_v19 one), the rows' json (engine_rows_v2) and the replay's json
(engine_replay_v2: the sources against the stored and the occupied set against the negative set; engine_replay_v3: also the masks bitwise
against the stored ones), taken over [from, to] in T/rs; the Gauss balance runs across the junction. Added: the segment column; the mask
column reads "stored: k of n" (replay v3, bitwise against the stored masks) or "derived" (replay v2 over a segment with no stored masks:
the replay's occupied set equals the negative set, as the stored run's Weyl-verified continuation implies for its own masks); the fields
check over the v19 segment (every record one integration segment, no step-failure zero, no stretch, no continued interval), counted from
the records; the label and the status line given by files (the lead's text, verbatim); the slices present (file, size). Writes the md and a
data npz (version 1's keys, the records of all segments in time order, a 'segment' index). Args: --config json --label-file --status-file
--slices-dir --md --npz. The config: {"segments": [{"name", "run", "handoff", "rows", "replay", "replay_kind": "v2"|"v3", "from", "to"}]}."""
import argparse
import json
import math
import os

import numpy as np


def shell_stress(H, k, part, R):
    F = np.asarray(H["F_face_%s" % part][k], float); fc = np.asarray(H["fc_%s" % part][k], float)
    Rf = np.concatenate([R[:-1] + 0.5 * np.diff(R), [R[-1] + 0.5]])
    Srf = F / (4.0 * math.pi * Rf ** 2)
    return 0.5 * (np.concatenate([[Srf[0]], Srf[:-1]]) + Srf), fc / (8.0 * math.pi * R)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--label-file", required=True)
    ap.add_argument("--status-file", required=True)
    ap.add_argument("--slices-dir", default=None)
    ap.add_argument("--crosshw", default=None, help="an engine_replay_crosshw_v2 md; its summary lines are quoted")
    ap.add_argument("--md", required=True)
    ap.add_argument("--npz", required=True)
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    label = open(a.label_file).read().strip(); status = open(a.status_file).read().strip()
    segs = []
    for s in cfg["segments"]:
        d, tag = os.path.split(s["run"])
        J = json.load(open(os.path.join(d, "collapse_%s.json" % tag)))
        Z = np.load(os.path.join(d, "collapse_%s_fields.npz" % tag), allow_pickle=True)
        H = np.load(s["handoff"], allow_pickle=True)
        rows = {round(r_["T_over_rs"], 6): r_ for r_ in json.load(open(s["rows"]))["rows"]} if s.get("rows") and os.path.exists(s["rows"]) else {}
        rep = {round(r_["T_over_rs"], 6): r_ for r_ in json.load(open(s["replay"]))["rows"]} if s.get("replay") and os.path.exists(s["replay"]) else {}
        segs.append(dict(s, J=J, Z=Z, H=H, rows_=rows, rep_=rep, tag=tag))
    J0 = segs[0]["J"]
    rs = float(J0["args"]["rs"]); E0 = float(J0["head"]["E0"]); R = np.asarray(segs[0]["Z"]["R"], float)
    L = ["## COLLAPSE_records_v1.1: %s (engine_collapse_records_v2.py)" % cfg.get("title", segs[-1]["tag"]), "",
         "Label (the lead's, verbatim): %s" % label, "", "Status: %s" % status, "",
         "Segments: " + "; ".join("%s: %s (%s), T/rs %g to %g, shell quantities from %s, rows %s, replay %s (%s)" % (
             s["name"], s["tag"], s["J"].get("script"), s["from"], s["to"], os.path.basename(s["handoff"]), os.path.basename(s.get("rows") or "-"),
             os.path.basename(s.get("replay") or "-"), s.get("replay_kind", "-")) for s in segs) + ".", ""]
    # the fields check over the v19 segments
    for s in segs:
        inr = [r for r in s["J"]["records"] if r.get("E") is not None and s["from"] - 1e-9 <= r["T"] / rs <= s["to"] + 1e-9]
        if not any(isinstance(r.get("gauge"), dict) and "segments" in r["gauge"] for r in inr):
            continue
        n = one = zeros = stretches = cont = 0
        for r in s["J"]["records"]:
            if r.get("E") is None or not (s["from"] - 1e-9 <= r["T"] / rs <= s["to"] + 1e-9):
                continue
            g = r.get("gauge") or {}
            n += 1
            one += int(g.get("segments", 1) == 1)
            zeros += int(g.get("step_failure_zeros", 0) or 0)
            stretches += int(g.get("t_stretches", 0) or 0)
            cont += int(bool(g.get("intervals")))
        L.append("- fields check, segment %s: %d records; one integration segment in %d; step-failure zeros %d; stretches %d; records with a continued interval %d." % (
            s["name"], n, one, zeros, stretches, cont))
    if a.crosshw and os.path.exists(a.crosshw):
        L += ["- the cross-hardware replay (%s): " % os.path.basename(a.crosshw) + " ".join(l[2:] for l in open(a.crosshw).read().split("\n") if l.startswith("- ") and "every number" not in l)]
    L += ["", "### The Gauss balance per interval (dE against the evolution, reset and lapse parts; the collar's Killing cut current over the interval beside)", "",
          "| T/rs | segment | E/E0 | Gauss identity error | dE | evolution | reset | lapse | closure | collar current x dT |", "|---|---|---|---|---|---|---|---|---|---|"]
    data = {k: [] for k in ("T", "segment", "S_r_dust", "S_t_dust", "S_r_pairs", "S_t_pairs", "b_dust", "b_pairs", "eps_tot", "J_tot", "eps_pairs", "J_pairs",
                            "N2avg", "eps_sea", "eps_R2", "eps_dust", "N", "v", "N_applied", "v_applied", "S_r_ell", "S_t_ell")}
    prev = None
    order = []
    for si, s in enumerate(segs):
        Z, H, J = s["Z"], s["H"], s["J"]
        recs = {round(r["T"], 6): r for r in J["records"] if r.get("E") is not None}
        Tz = np.asarray(Z["T"], float); TH = np.asarray(H["T"], float)
        for k in range(len(TH)):
            T = float(TH[k]); t = T / rs
            if not (s["from"] - 1e-9 <= t <= s["to"] + 1e-9):
                continue
            kz = int(np.argmin(np.abs(Tz - T)))
            if abs(Tz[kz] - T) > 1e-9:
                continue
            rec = recs.get(round(T, 6), {})
            eps = np.asarray(Z["eps_tot"][kz], float)
            N2 = np.asarray(H["N2avg"][k], float)
            eC = np.asarray(H["eps_sea"][k], float) + np.asarray(H["eps_dust"][k], float); eR = np.asarray(H["eps_R2"][k], float)
            E = float(rec.get("E", np.nan))
            G = float(np.sum(eps * N2))
            gerr = (E - G) / abs(E) if E == E else float("nan")
            line = "| %.3f | %s | %.7f | %+.2e |" % (t, s["name"], E / E0, gerr)
            if prev is not None:
                a_ = 0.5 * (prev["N2"] + N2)
                evo = float(np.sum(a_ * (eC - prev["eC"]))); rst = -float(np.sum(a_ * (eR - prev["eR"])))
                lap = float(np.sum((N2 - prev["N2"]) * 0.5 * (prev["eps"] + eps)))
                dE = E - prev["E"]
                mb = rec.get("M1_balance") or {}
                line += " %+.4e | %+.4e | %+.4e | %+.4e | %+.2e | %s |" % (dE, evo, rst, lap, dE - (evo + rst + lap), ("%+.4e" % mb["evolution_trapezoid"]) if "evolution_trapezoid" in mb else "-")
            else:
                line += " - | - | - | - | - | - |"
            L.append(line)
            prev = {"N2": N2, "eC": eC, "eR": eR, "eps": eps, "E": E}
            order.append((si, k, T))
            Srd, Std = shell_stress(H, k, "dust", R); Srp, Stp = shell_stress(H, k, "pairs", R)
            for key, val in (("T", T), ("segment", si), ("S_r_dust", Srd), ("S_t_dust", Std), ("S_r_pairs", Srp), ("S_t_pairs", Stp), ("b_dust", H["b_dust"][k]),
                             ("b_pairs", H["b_pairs"][k]), ("eps_tot", eps), ("J_tot", Z["J_tot"][kz]), ("eps_pairs", Z["eps_pairs"][kz]), ("J_pairs", Z["J_pairs"][kz]),
                             ("N2avg", N2), ("eps_sea", H["eps_sea"][k]), ("eps_R2", eR), ("eps_dust", H["eps_dust"][k]), ("N", Z["N"][kz]), ("v", Z["v"][kz]),
                             ("N_applied", Z["N_applied"][kz]), ("v_applied", Z["v_applied"][kz]), ("S_r_ell", H["S_r_ell"][k]), ("S_t_ell", H["S_t_ell"][k])):
                data[key].append(np.asarray(val, float) if key not in ("T", "segment") else val)
    L += ["", "- evolution: sum_n <N^2>_n Delta(eps of the propagated sea and dust); reset: -sum_n <N^2>_n Delta Tr[R2 h_n]; lapse: sum_n Delta<N^2>_n eps_n "
          "(the interval's means: the three parts sum exactly to the change of sum_n eps_n <N^2>_n); closure: dE minus their sum; collar current x dT: the "
          "inside Killing energy's cut current at the collar by the trapezoid (the record's M1).", "",
          "### The stress checks, the evolution rows and the replay (residual over the largest term; the matter's support and the exterior; at the radii)", "",
          "| T/rs | segment | identity dust, pairs | virial dust, pairs | replay eps, J | res_rr support | res_tt support | res_rr, res_tt exterior | at R_t | at R_sh | at r_in | stepping floor | masks |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    f_ = lambda x: ("%.1e" % x) if isinstance(x, (int, float)) and x == x else "-"
    for si, k, T in order:
        s = segs[si]; key = round(T / rs, 6)
        rp = s["rep_"].get(key, {}); rw = s["rows_"].get(key, {})
        rad = rw.get("radii", {})
        g2 = lambda lab: ("%.1e, %.1e" % tuple(rad[lab])) if lab in rad else "-"
        sf = rw.get("stepping_floor") or [None, None]
        rec = next((r for r in s["J"]["records"] if r.get("E") is not None and abs(r["T"] - T) < 1e-9), {})
        st = rec.get("stress") or {}
        if s.get("replay_kind") == "v3":
            idn = [None, None]; vir = [None, None]
            if st:
                idn = [st.get("identity_max_abs", {}).get("dust"), st.get("identity_max_abs", {}).get("pairs")]
                vir = [st.get("virial_rel", {}).get("dust"), st.get("virial_rel", {}).get("pairs")]
            masks = ("stored: %s of %d" % (rp["masks_equal_channels"], int(s["J"]["args"]["kappa_sea"]))) if rp else "-"
            eps_rel, J_rel = rp.get("eps_rel"), rp.get("J_rel")
        else:
            idn = rp.get("identity_max_abs") or [None, None]; vir = rp.get("virial_rel") or [None, None]
            masks = ("derived (%s)" % rp.get("mask_equals_negative_set")) if rp else "-"
            eps_rel, J_rel = rp.get("eps_rel"), rp.get("J_rel")
        L.append("| %.3f | %s | %s, %s | %s, %s | %s, %s | %s | %s | %s, %s | %s | %s | %s | %s, %s | %s |" % (
            T / rs, s["name"], f_(idn[0]), f_(idn[1]), f_(vir[0]), f_(vir[1]), f_(eps_rel), f_(J_rel), f_(rw.get("res_rr_support")), f_(rw.get("res_tt_support")),
            f_(rw.get("res_rr_exterior")), f_(rw.get("res_tt_exterior")), g2("R_t"), g2("R_sh"), g2("r_in"), f_(sf[0]), f_(sf[1]), masks))
    L += ["", "- identity: the lattice momentum balance d<Q_in>/dt = -F + sum f (exact at unit clock and zero flow; the largest absolute difference); virial: "
          "S_r + 2 S_t against rho - m b relative; replay: the replayed sources against the stored shell totals (acceptance 1e-8); masks: 'stored: k of n' are the "
          "channels whose replayed occupation mask equals the run's stored mask bit for bit (engine_replay_v3), 'derived' the replay's occupied set against the "
          "negative-energy set where the run stored no mask (engine_replay_v2); the rows' d_t k by the centred difference of successive records."]
    if a.slices_dir and os.path.isdir(a.slices_dir):
        fl = sorted(f for f in os.listdir(a.slices_dir) if f.endswith(".npz"))
        L += ["", "### The slices (%s)" % a.slices_dir, "", "| file | bytes |", "|---|---|"] + ["| %s | %d |" % (f, os.path.getsize(os.path.join(a.slices_dir, f))) for f in fl]
    L += ["", "- the data file %s holds per record and shell S_r, S_t (dust, pairs), b, eps and J (total, pairs), <N^2>_n, the sea's, R2's and the dust's shell energies, "
          "N and v (the record's solve) and the applied fields, with the smoothed S_r,ell and S_t,ell at the coarse nodes, the segment index per record; R, T, g, ell, tau, m beside." % os.path.basename(a.npz),
          "- every number above is printed by the script from the runs' jsons and npz, the handoff files, the rows' and the replays' jsons."]
    S0 = segs[0]
    np.savez_compressed(a.npz, R=R, r_nodes=np.asarray(S0["H"]["r_nodes"], float), g=float(J0["args"]["g"]), ell=float(J0["args"]["ell"]), tau=float(J0["args"]["tau"]),
                        m=float(J0["args"]["m"]), rs=rs, segment_names=np.array([s["name"] for s in segs]), **{k: np.array(v) for k, v in data.items()})
    open(a.md, "w").write("\n".join(L) + "\n")
    print("\n".join(L[:12]))


if __name__ == "__main__":
    main()
