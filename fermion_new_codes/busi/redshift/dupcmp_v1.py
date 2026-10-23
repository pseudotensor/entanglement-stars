import sys, numpy as np
sys.path.insert(0, '<internal>/code')
import read_history_v1 as RH
reg = RH.registry()
groups = [["P64_s4_prod", "P64_s4_prod56", "P64_ref_c19r11", "P64_s4_c27", "P64_s4_ext200", "P64_s4_newton", "P64_s4_B2n4096"],
          ["P64_s8_prod", "P64_s8_c27"], ["P64_s2_nk", "P64_s2_to40"], ["ISO_128_s4", "ISO_128_s4_plain"], ["M3c_s4_96", "M3c_s4_96_nk"]]
for g in groups:
    H0 = RH.load_history(g[0], kind="nodes", check_md5=False, reg=reg)
    for lab in g[1:]:
        H = RH.load_history(lab, kind="nodes", check_md5=False, reg=reg)
        t0 = {round(t, 6): i for i, t in enumerate(H0["T"])}
        common = [(t0[round(t, 6)], j) for j, t in enumerate(H["T"]) if round(t, 6) in t0]
        nr = min(len(H0["r"]), len(H["r"]))
        if not np.allclose(H0["r"][:nr], H["r"][:nr]):
            print(g[0], lab, "different node grids"); continue
        dN = [float(np.max(np.abs(H0["N"][i, :nr] - H["N"][j, :nr]))) for i, j in common]
        dv = [float(np.max(np.abs(H0["v"][i, :nr] - H["v"][j, :nr]))) for i, j in common]
        first = next((H["T"][j] for (i, j), a, b in zip(common, dN, dv) if max(a, b) > 1e-8), None)
        upto = [max(a, b) for (i, j), a, b in zip(common, dN, dv) if H["T"][j] <= 12.5 + 1e-9]
        print("%-14s vs %-16s common %4d  max|dN,dv| all %.1e  to T 12.5 %.1e  first > 1e-8 at %s" % (
            g[0], lab, len(common), max(max(dN), max(dv)), max(upto) if upto else float('nan'), first))
