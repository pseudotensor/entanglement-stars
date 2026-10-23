"""dust_root_check_v1.py (the lane's analyst builder; engine spec 43.4 (1)): at the dust history's launch records T/r_s 12.125 and 15.0, side
by side: the producer v1's two tag roots (dense: B v1's dense_roots via B v2's outer_inner_of; 'applied': the record's trapping zeros), the dense
roots with their signed slopes and the routine's return path (found among the dense roots, or substituted by the record's R_t), the record's
stored trapping zeros, min over r of N + v and its sign-change count on the dense function's roots and on the applied site fields (v1b's
applied root), and delta_H = R - 2 g M_g(R) is not stored per root here (printed where the record carries the mass function)."""
import sys, json, numpy as np
sys.path.insert(0, "code")
import engine_single_object_B_v2 as BV2
import engine_dust_root_history_v1b as D1b
H = BV2.CombinedHistory("H100b/C3prod19/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11.json", "H100b/C3prod19/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11_fields.npz", 8192,
                        "C3ext19/collapse_c19_rs64_A_R2_tau8_ell4_avg_ext100.json", "C3ext19/collapse_c19_rs64_A_R2_tau8_ell4_avg_ext100_fields.npz", 40.0)
TT = H.T / H.rs_nom; out = []
for t in (12.125, 15.0):
    k = int(np.argmin(abs(TT - t))); rd = BV2.roots_at(H, k, "engine_fields_v14"); rec = H.rec[k]
    Ro, Ri, kin = BV2.outer_inner_of(rd, rec)
    pos = [q for q in rd if q["slope"] > 0]
    N, v = H.fields(k); f = (N + v)[: H.ne]
    sc = int(np.sum(np.sign(f[:-1]) != np.sign(f[1:])))
    o = {"T_over_rs": float(TT[k]), "dense_roots": [[q["R"], q["slope"]] for q in rd], "record_trapping_zeros": rec.get("trapping_zeros"), "record_R_t": rec.get("R_t"),
         "v1_dense_outer": Ro, "v1_dense_path": "found" if pos else "substituted by the record's R_t", "v1_applied_outer": max(rec.get("trapping_zeros") or [0]) or rec.get("R_t"),
         "applied_site_min_N_plus_v": float(f.min()), "applied_site_min_at_R": float(H.R[int(np.argmin(f))]), "applied_site_sign_changes": sc,
         "v1b_applied_outer": D1b.applied_outer_root(H, k)}
    out.append(o); print(json.dumps(o))
json.dump(out, open("DUST/dust_root_check_v1.json", "w"), indent=1)
