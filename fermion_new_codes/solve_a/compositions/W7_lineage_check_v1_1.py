#!/usr/bin/env python
"""W7_lineage_check_v1_1.py (COMP2 v1.1, ruling 1's gate): attributes the 48-to-56 same-set gap. The ext100 production restarted from the r11
T28 slice, so 28 to 40 was run twice (r11 and ext100). Lists the HF dataset's solve/B_ext100_slices/ files; if ext100's own T40 slice exists,
fetches it into --out-dir (md5 printed; token from HF_TOKEN, never printed) and compares its dust columns (kappa 1..5, as stored) with the r11
T40 slice's: per-column |<a|b>|^2 and the site-weight L1 difference. Read-only on every input; deletes nothing outside its own cache."""
import hashlib, json, os, shutil, sys
import numpy as np
REPO = "pseudotensor/qi2-fermion-bulk"
R11_40 = "<internal>/orbitals_c19_rs64_A_R2_tau8_ell4_avg_r11_T0040.00.npz"
NAME = "orbitals_c19_rs64_A_R2_tau8_ell4_avg_ext100_T0040.00.npz"


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 24), b""):
            h.update(c)
    return h.hexdigest()


out_dir = sys.argv[1]
tok = os.environ["HF_TOKEN"]
from huggingface_hub import HfApi, hf_hub_download
files = sorted(f for f in HfApi(token=tok).list_repo_files(REPO, repo_type="dataset") if f.startswith("solve/B_ext100_slices/"))
res = {"repo_files_B_ext100_slices": files}
rp = "solve/B_ext100_slices/" + NAME
if rp in files:
    dst = os.path.join(out_dir, NAME)
    if not os.path.exists(dst):
        p = hf_hub_download(REPO, rp, repo_type="dataset", token=tok, cache_dir=os.path.join(out_dir, ".hf_cache"))
        shutil.copyfile(p, dst + ".part"); os.replace(dst + ".part", dst)
        shutil.rmtree(os.path.join(out_dir, ".hf_cache"), ignore_errors=True)
    res["ext100_T40_md5"] = md5(dst); res["r11_T40_md5"] = md5(R11_40)
    A = np.load(dst); Bz = np.load(R11_40)
    res["kappas"] = {}
    for k in range(1, 6):
        a = np.asarray(A["dust_%d" % k], np.complex128); b = np.asarray(Bz["dust_%d" % k], np.complex128); n = a.shape[0] // 2
        ov = np.abs((a.conj() * b).sum(axis=0)) ** 2 / ((np.abs(a) ** 2).sum(axis=0) * (np.abs(b) ** 2).sum(axis=0))
        wa = (np.abs(a) ** 2).reshape(n, 2, -1).sum(axis=1); wb = (np.abs(b) ** 2).reshape(n, 2, -1).sum(axis=1)
        res["kappas"][str(k)] = {"one_minus_overlap_max": float((1 - ov).max()), "site_weight_L1_max": float(np.abs(wa - wb).sum(axis=0).max()),
                                 "fields_N_maxdiff": float(np.abs(A["N_applied"] - Bz["N_applied"]).max()), "fields_v_maxdiff": float(np.abs(A["v_applied"] - Bz["v_applied"]).max())}
json.dump(res, open(os.path.join(out_dir, "lineage_check_v1_1.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
