#!/bin/bash
# The gate of engine spec 23.6 for engine_star_v8 (the GPU levels) on box GPU 0, after the v23 diagnostic (PID given): the stored (0.1, 1/4)
# family's largest member (N_B 1136, 1536 sites, ell 8) from the continuum seed; then its E and occupations against the stored member's.
# Builder analyst.
cd ${QI2_ROOT}/engine || exit 1
while kill -0 $1 2>/dev/null; do sleep 20; done
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
mkdir -p STAT
echo "star v8 gate start $(date -u +%H:%M:%S)" >> STAT/status.txt
nice -n 10 ${QI2_ROOT}/python code/engine_star_v8.py --m 0.1 --g 0.25 --ell 8 --nsites 1536 --eos EOS/eos_v4_tables.npz --ovpg OVPG/ovpg_families.json \
  --NB-list 1136 --ell-control "" --gpu 0 --out STAT --tag gate8_m0.1_g0.25_NB1136 > STAT/log_gate8.log 2>&1
echo "star v8 gate exit $? $(date -u +%H:%M:%S)" >> STAT/status.txt
${QI2_ROOT}/python - > STAT/gate8_compare_v1.md 2>&1 <<'PY'
import json
a = [q for q in json.load(open("STAT/star_gate8_m0.1_g0.25_NB1136.json"))["members"] if not q.get("failed")]
b = [q for q in json.load(open("Sfam5/star_m0.1_g0.25_ell8.json"))["members"] if not q.get("failed") and abs(q["N_B"] - 1136.0) < 1e-6]
print("## engine_star_v8's gate (the GPU levels) against the stored (0.1, 1/4) member at N_B 1136 (engine_star_v6)")
print()
if not a or not b:
    print("- a member is missing: v8 %d, stored %d" % (len(a), len(b)))
else:
    x, y = a[0], b[0]
    print("- E: v8 %.15f, stored %.15f, relative difference %.2e (gate 1e-10)" % (x["E"], y["E"], abs(x["E"] - y["E"]) / abs(y["E"])))
    occ_same = x.get("levels") == y.get("levels")
    lx = {k: [(round(e, 9), round(f, 12)) for e, f in v] for k, v in x["levels"].items()}; ly = {k: [(round(e, 9), round(f, 12)) for e, f in v] for k, v in y["levels"].items()}
    print("- the occupied levels and occupations (rounded to 1e-9, 1e-12) %s; channels occupied v8 %s, stored %s" % ("identical" if lx == ly else "different", sorted(lx), sorted(ly)))
    print("- E_top %.12f against %.12f; R_surf %s against %s; passes %d against %d" % (x["E_top"], y["E_top"], x.get("R_surf"), y.get("R_surf"), x["passes"], y["passes"]))
    print("- verdict: %s" % ("pass" if (abs(x["E"] - y["E"]) / abs(y["E"]) < 1e-10 and lx == ly) else "fail"))
PY
echo "star v8 gate compare done $(date -u +%H:%M:%S)" >> STAT/status.txt
