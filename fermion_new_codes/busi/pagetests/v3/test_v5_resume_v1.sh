#!/bin/bash
# test_v5_resume_v1.sh (busi; box CPU): run_bprime_v5's checkpoint/resume against an uninterrupted v5 run and against run_bprime_v4.
# The F1 s = 16 prep, 4 columns in batches of 2, clocks 0.25 and 0.5 L (several steps per batch per clock). C is killed by --stop-after-s
# right after a mid-batch checkpoint (exit 75) and resumed; then the npz arrays are compared.
set -u
PY=${QI2_ROOT}/python
CODE=${QI2_ROOT}/v3
P=${QI2_ROOT}/F1_s16_d0_prep.npz
OUT=${1:-${QI2_ROOT}/test_v5_resume}
rm -rf "$OUT"; mkdir -p "$OUT"; cd "$CODE"
C="--phase run --prep $P --device cpu --T-max-over-L 0.5 --extra-T-over-L 0.25 0.5 --dt 2 --dt-sample 2 --col-batch 2 --max-columns 4 --threads 4 --out $OUT"
$PY run_bprime_v4.py $C --tag A_v4 > $OUT/A.log 2>&1; echo "A rc $?"
$PY run_bprime_v5.py $C --tag B_v5 --ckpt-every-s 1e9 > $OUT/B.log 2>&1; echo "B rc $?"
$PY run_bprime_v5.py $C --tag C_v5 --ckpt-every-s 5 --stop-after-s 40 > $OUT/C1.log 2>&1; echo "C first leg rc $? (75 expected)"
$PY run_bprime_v5.py $C --tag C_v5 --ckpt-every-s 5 --resume > $OUT/C2.log 2>&1; echo "C resumed rc $?"
$PY - "$OUT" <<'PYEOF'
import sys, json, numpy as np
o = sys.argv[1]
A, B, C = (np.load("%s/%s.npz" % (o, t)) for t in ("A_v4", "B_v5", "C_v5"))
res = {}
for k in ("G", "R", "S_diag", "S_full_every16", "radial_profiles_every16", "absorbed_flux_sectors_every16", "S_t"):
    res[k] = {"v4_vs_v5": float(np.max(np.abs(A[k] - B[k]))), "resumed_vs_uninterrupted": float(np.max(np.abs(C[k] - B[k])))}
J = json.load(open("%s/C_v5.json" % o))
res["resumed_from"] = J["timing"]["resumed_from"]
res["PASS_v4_vs_v5_1e-12"] = all(v["v4_vs_v5"] < 1e-12 for k, v in res.items() if isinstance(v, dict))
res["PASS_resume_exact_1e-14"] = all(v["resumed_vs_uninterrupted"] < 1e-14 for k, v in res.items() if isinstance(v, dict)) and len(res["resumed_from"]) == 1
json.dump(res, open("%s/test_v5_resume.json" % o, "w"), indent=1)
print(json.dumps(res, indent=1))
PYEOF
