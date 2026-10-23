"""b2_read_check_v1.py (the lane's analyst builder; engine spec 42.2 (1)): page_history_v6i's History on B2's records at --nsites 8192: the engine's
domain, the join at 64 r_s, the appended exterior 64 to 128 r_s, the collar (16 r_s), R_cut (24 r_s) and the far collector (48 r_s) inside the domain;
the join check over 32 to 64 r_s per record; the production's first records (C3prod19 r11) beside B2's at the same clocks inside 32 r_s."""
import sys, json, numpy as np
sys.path.insert(0, "<internal>/code_v6")
import page_history_v6i as P
d = "<internal>/B2"; t = "c19_rs64_A_R2_tau8_ell4_avg_B2n4096"
H = P.History(d + "collapse_%s.json" % t, d + "collapse_%s_fields.npz" % t, 8192)
rs = H.rs_nom
print("records", len(H.T), "T/rs", H.T[0] / rs, "to", H.T[-1] / rs, "engine sites", H.ne, "= R_max/r_s", H.Re[-1] / rs, "chain", H.nsites, "=", H.nsites / rs, "r_s")
for nm, R in (("collar", 16), ("R_cut", 24), ("far collector", 48)):
    print(nm, R, "r_s inside the engine's domain:", R * rs < H.Re[-1])
jn = [H.join_check(k, 32 * rs, 64 * rs) for k in range(len(H.T))]
print("join over 32 to 64 r_s: max |N-1|", max(j[0] for j in jn), "max rel dv", max(j[1] for j in jn))
N, v = H.fields(len(H.T) - 1)
print("exterior beyond 64 r_s: N==1", bool(np.all(N[H.ne:] == 1)), "v at 64/65 r_s", v[H.ne - 1], v[H.ne])
P0 = H.__class__("<internal>/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11.json", "<internal>/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11_fields.npz", 8192)
m = min(len(P0.T), len(H.T)); ne0 = P0.ne
dE = max(abs(H.E[k] / H.E[0] - P0.E[k] / P0.E[0]) for k in range(m))
dN = max(float(np.max(np.abs(H.N_e[k][:ne0] - P0.N_e[k]))) for k in range(m))
print("against the production's first %d records (inside 32 r_s): max |dE/E0| %.3e, max |dN| %.3e" % (m, dE, dN))
