#!/usr/bin/env python
"""mk_v17_shell_input_v1.py (busi, analyst; inbox #343): a gate input for engine_fields_v17's empty-interior rule: R1 RUNG2c40b_srf's
T0012 mirror (R, g, T, eps_tot, J_tot only) with eps_tot and J_tot set to 0 at the sites R <= R_CUT (default 40 = 20 ell at ell 2), so the
smoothed sources vanish below 1e-14 on an interval [0, r_e] and v17 takes its own path (integrate_UL_v17, Piecewise of v16) instead of
delegating to v16's solve. A synthetic test input for the module swap, not a physical state.
usage: python mk_v17_shell_input_v1.py IN.npz OUT.npz [R_CUT]"""
import sys
import numpy as np

src, dst = sys.argv[1], sys.argv[2]
rcut = float(sys.argv[3]) if len(sys.argv) > 3 else 40.0
z = np.load(src, allow_pickle=False)
R = z["R"]; e = np.array(z["eps_tot"]); J = np.array(z["J_tot"])
m = R <= rcut
e[:, m] = 0.0; J[:, m] = 0.0
np.savez(dst, R=R, g=z["g"], T=z["T"], eps_tot=e, J_tot=J)
print("wrote %s: %d records, %d of %d sites zeroed (R <= %g)" % (dst, len(z["T"]), int(m.sum()), len(R), rcut))
