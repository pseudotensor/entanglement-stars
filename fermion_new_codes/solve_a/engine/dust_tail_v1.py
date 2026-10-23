#!/usr/bin/env python
"""dust_tail_v1.py (the lane's analyst builder; engine spec 40.3 (3), 39.4 (7)): the dust's tail, one table, results/page/D_dust_tail_v1.md:
per stored slice from T/r_s 20 on (the production's slices 20 to 40, the continuation's from 42), weighted over the channels (16 kappa):
from D's stored jsons, beyond 16 and 24 r_s, the dust's weight (number), its Killing-basis particle number and energy (D's particles_dust,
E_particles_dust) and its energy per particle in units of m; from the slices, the dust's exact number current through 16 r_s,
F = -2 Im sum_c f_c phi_X^dagger h_XY phi_Y over the dust columns (X the sites R <= 16 r_s, Y beyond; outward positive; dust-only by the
structural fact of 40.3: h_m anticommutes with the on-site antiunitary, so the sea and the reference keep site density one), at the slice's
applied fields; the slices from which the paper's D numbers may be read are not decided here (the marker waits for turn 49), so every row is
printed with its clock. Read in turn 49 to say whether the collapse sheds unbound dust; nothing prints before. Every number printed by the
script. Usage: --slices a.npz,... --d-json a.json,... --rs 64 --m 0.2 --md --out-json."""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import engine_single_object_A_v2 as SA  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slices", required=True); ap.add_argument("--d-json", required=True)
    ap.add_argument("--rs", type=float, default=64.0); ap.add_argument("--m", type=float, default=0.2)
    ap.add_argument("--md", required=True); ap.add_argument("--out-json", required=True)
    a = ap.parse_args()
    for p in (a.md, a.out_json):
        if os.path.exists(p):
            sys.exit("refusing to overwrite %s" % p)
    dv = {}
    for p in [x for x in a.d_json.split(",") if x]:
        for sl in json.load(open(p)).get("slices", []):
            dv[(round(float(sl["T_over_rs"]), 6), os.path.basename(sl.get("slice", "")))] = sl
    rows = []
    for p in [x for x in a.slices.split(",") if x]:
        Z = np.load(p)
        T = float(Z["T"]) / a.rs; R = np.asarray(Z["R"], float)
        Napp = np.asarray(Z["N_applied"], float); vapp = np.asarray(Z["v_applied"], float)
        wts = {int(k): float(w) for k, w in zip(Z["kappas"], Z["weights"])}
        i0 = int(np.searchsorted(R, 16.0 * a.rs, side="right"))
        F = 0.0; Fk = {}
        for k in wts:
            dust = np.asarray(Z["dust_%d" % k], np.complex128); f = np.asarray(Z["f_%d" % k], float)
            if dust.shape[1] == 0:
                Fk[k] = 0.0; continue
            H = SA.hm_host(R, a.m, k, Napp, vapp).tocsr()
            hXY = H[: 2 * i0, 2 * i0:]
            z = np.sum(f[None, :] * (dust[: 2 * i0].conj() * (hXY @ dust[2 * i0:])), axis=0).sum()
            Fk[k] = float(-2.0 * np.imag(z)); F += wts[k] * Fk[k]
        sl = dv.get((round(T, 6), os.path.basename(p)))
        if sl is None:
            cands = [v for (t_, _), v in dv.items() if t_ == round(T, 6)]
            sl = cands[0] if cands else None
        row = {"T_over_rs": T, "slice": p, "F_dust_16_weighted": F, "F_dust_16_per_channel": Fk, "D_json_found": sl is not None}
        if sl is not None:
            for key in ("16", "24"):
                regs = [(c["w"], c["regions"][key]) for c in sl["channels"]]
                W_ = lambda f_: float(sum(w * f_(r) for w, r in regs))  # noqa: E731
                n_p = W_(lambda r: r["particles_dust"]); e_p = W_(lambda r: r["E_particles_dust"])
                row["beyond_%s" % key] = {"dust_weight": W_(lambda r: r["dust_weight"]), "particles_dust": n_p, "E_particles_dust": e_p,
                                          "holes_dust": W_(lambda r: r["holes_dust"]), "E_out_dust": W_(lambda r: r["E_out_dust"]),
                                          "E_per_particle_over_m": (e_p / n_p / a.m) if n_p > 0 else None}
        rows.append(row)
        Z.close()
        print("T/rs %.3f: F_dust(16 r_s) %+.6e; D json %s" % (T, F, "found" if sl is not None else "missing"), flush=True)
    rows.sort(key=lambda r: (r["T_over_rs"], "ext" in r["slice"]))
    L = ["# D_dust_tail_v1: the dust's tail beyond the collars (engine spec 40.3 (3), 39.4 (7))", "",
         "Composed %s by code/engine/dust_tail_v1.py; every number printed by the script. Weighted over the channels (16 kappa). The beyond-16 and beyond-24 r_s columns are D's stored readouts "
         "(engine_massive_channel_D_v2 on the stored slices: the dust's weight; its Killing-basis particle number and energy in the exterior block's eigenbasis); E/particle in units of m; "
         "F the dust's exact number current through 16 r_s at the slice (outward positive). The marker that limits the paper's D numbers waits for turn 49, so every slice is listed. "
         "A record for turn 49's reading (whether the collapse sheds unbound dust); nothing prints before." % time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()), "",
         "| T/r_s | slice | dust weight beyond 16 | particles beyond 16; E | E/particle beyond 16 (m) | dust weight beyond 24 | particles beyond 24; E | E/particle beyond 24 (m) | F through 16 r_s |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        b16 = r.get("beyond_16") or {}; b24 = r.get("beyond_24") or {}
        f_ = lambda d, k, fm: (fm % d[k]) if d.get(k) is not None else "-"  # noqa: E731
        L.append("| %.3f | %s | %s | %s; %s | %s | %s | %s; %s | %s | %+.4e |" % (
            r["T_over_rs"], "continuation" if "ext" in os.path.basename(r["slice"]) else "production", f_(b16, "dust_weight", "%.4e"), f_(b16, "particles_dust", "%.4e"),
            f_(b16, "E_particles_dust", "%.4e"), f_(b16, "E_per_particle_over_m", "%.4f"), f_(b24, "dust_weight", "%.4e"), f_(b24, "particles_dust", "%.4e"),
            f_(b24, "E_particles_dust", "%.4e"), f_(b24, "E_per_particle_over_m", "%.4f"), r["F_dust_16_weighted"]))
    open(a.md, "x").write("\n".join(L) + "\n")
    json.dump({"script": "dust_tail_v1.py", "rows": rows}, open(a.out_json, "x"), indent=1, default=float)
    print("wrote %s" % a.md)


if __name__ == "__main__":
    main()
