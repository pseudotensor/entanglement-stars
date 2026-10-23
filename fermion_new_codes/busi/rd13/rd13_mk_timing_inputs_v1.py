#!/usr/bin/env python
"""rd13_mk_timing_inputs_v1.py (busi, analyst): the small inputs of rd13_h100_timing_v1.py (fields, perturbations, kappa 1 masks) from the stored
W2 r2b T0013.50 slice and fields npz, the ISO_128 nkqh2 T0005.00 slice and fields npz, and the prototype's synthetic 8192-site profile.
    python rd13_mk_timing_inputs_v1.py OUT_NPZ CODE_DIR W2_SLICE W2_FIELDS ISO_SLICE ISO_FIELDS"""
import sys
import numpy as np
sys.path.insert(0, sys.argv[2])
import engine_vacflow_v1 as VF
out = {}
for case, sl, fl in (("w2", sys.argv[3], sys.argv[4]), ("iso", sys.argv[5], sys.argv[6])):
    z = np.load(sl, allow_pickle=False); zf = np.load(fl, allow_pickle=False)
    Ti = np.asarray(zf["T"], float); i1 = int(np.argmin(np.abs(Ti - float(z["T"])))); i0 = max(0, i1 - 1)
    out[case + "_R"] = np.asarray(z["R"], float); out[case + "_N"] = np.asarray(z["N_applied"], float); out[case + "_v"] = np.asarray(z["v_applied"], float)
    out[case + "_dN"] = np.asarray(zf["N_applied"][i1], float) - np.asarray(zf["N_applied"][i0], float)
    out[case + "_dv"] = np.asarray(zf["v_applied"][i1], float) - np.asarray(zf["v_applied"][i0], float)
    out[case + "_mask"] = np.asarray(z["mask_1"], np.uint8)
n_s = 8192
R = np.arange(n_s, dtype=float) + 1.0; rs = n_s / 32.0
out["syn_R"] = R; out["syn_v"] = VF.profile(R, rs, "exact"); out["syn_N"] = 1.0 - 0.3 * np.exp(-(R / (2 * rs)) ** 2)
bump = np.exp(-((R - 2 * rs) / rs) ** 2); out["syn_dN"] = 1e-3 * bump; out["syn_dv"] = 1e-3 * bump
np.savez(sys.argv[1], **out)
print("wrote", sys.argv[1], {k: v.shape for k, v in out.items()})
