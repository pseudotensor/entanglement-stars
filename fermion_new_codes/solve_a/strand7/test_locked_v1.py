# builder's check of strand7_sea_page_v2.LockedBackground before the runs (SPEC_strand7_page.md sec. 12.2 and C-pg-14): r_0, the energy control,
# N_s at the lead's window estimates, the window, and the late law lambda r_h and u_H against frozen_schw E8 (S7a_run1) at the same T/r_h
import sys, math, json
sys.path.insert(0, "<internal>/code")
import numpy as np
import strand7_sea_page_v2 as P
rs0 = 32.0
A0 = 0.234465 * rs0
L = P.LockedBackground(A0, rs0, 4 * rs0, 40 * rs0 + 0.25 * rs0)
print("r0/A0 %.6f (spec: 14.99); energy control %s" % (L.r0 / A0, json.dumps(L.energy_control)))
for T in (0, 10, 16.5, 20, 24, 25.3, 30, 36, 40):
    s = L.s(T * rs0)
    print("T/rs0 %5.1f: s/A0 %.8f N_s %.6e R/r_h %.10f" % (T, s / A0, L.N_s(T * rs0), s * (1 + A0 / s) ** 2 / (4 * A0)))
cut = P.Cuts(None, 1.0, locked=L, T_end=40 * rs0)
TH = 1.0 / (4 * math.pi * 4 * A0)
w = cut.window(lambda t: TH)
print("window (r_s0):", {k: (v / rs0 if v else v) for k, v in w.items()})
rh = 4 * A0
Tg = np.arange(10 * rs0, 30 * rs0, 0.1 * rs0)
R = np.array([L.s(t) * (1 + A0 / L.s(t)) ** 2 for t in Tg])
d = R - rh
for (t1, t2) in ((14, 18), (18, 22), (22, 26)):
    m = (Tg >= t1 * rs0) & (Tg <= t2 * rs0)
    sl = np.polyfit(Tg[m], np.log(d[m]), 1)[0]
    print("lambda r_h over [%g, %g] r_s0: %.6f" % (t1, t2, -sl * rh))
fz = json.load(open("<internal>/s7a_frozen_schw_E8_A.json"))
m = fz["meta"]; recs = fz["run"]["records"]
print("frozen_schw E8: r_s %g, A0 %s, last T/r_s %g" % (m["r_s"], fz["run"].get("A0"), recs[-1]["T_over_rs"]))
