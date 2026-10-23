"""h2_seg_scan_v1.py (busi; PERF H2-A/H2-B coverage): which stored records of a fields mirror exercise more than one U/L segment, stretches,
gauge intervals, zeros or near tangencies (engine_fields_v16, options irrelevant). usage: python h2_seg_scan_v1.py SRC_DIR FIELDS_NPZ ELL"""
import sys
import numpy as np
sys.path.insert(0, sys.argv[1])
import engine_fields_v16 as FLD, engine_fields_v4 as F4, engine_fields_v12 as F12, engine_fields_v14 as F14, engine_fields_kcache_v1 as KC  # noqa
KC.install([F4, F12, F14, FLD])
z = np.load(sys.argv[2], allow_pickle=False); R = z["R"]; g = float(z["g"]); ell = float(sys.argv[3])
for i in range(len(z["T"])):
    Nn, vn, Wn, info = FLD.solve_constraints_ivp(z["eps_tot"][i], z["J_tot"][i], g, R, ell, N0=None, rtol=1e-12, atol=1e-16, collar_policy="flag", support_rel=1e-14)
    gz = info.get("gauge") or {}
    print("rec %d T %.3f segments %s stretches %s intervals %d zeros %d near_tangencies %s failed %s" % (i, z["T"][i], (info.get("ivp") or {}).get("segments"),
          gz.get("stretches"), len(gz.get("intervals", [])), len(gz.get("zeros", [])), gz.get("near_tangencies"), info.get("failed")), flush=True)
