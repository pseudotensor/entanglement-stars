#!/bin/bash
# GATECOMP (iii): gate profiles on the stored records of ISO_128_s4 (job (b)) and the three r_s/a 64 width members. Box CPU, nice 10,
# one thread each (4 processes; with the dust comparator 5 cores). Outputs: small JSONs in ${QI2_ROOT}/.
set -e
B=${QI2_ROOT}/iii; C=$B/code; O=$B/gates; mkdir -p $O
FM=${QI2_ROOT}/code        # engine_fields_v16.py = v16d 7a69faad (ivp route by default)
I=${QI2_ROOT}/collapse_lad_ISO_128_s4_rd_m0p2_rs128_n4096_R04_k10_tau8_ell4_prep4_A_R2_avg_newton_v36nq_rd2
P=${QI2_ROOT}/pod2_engine
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
PY="nice -n 10 ${QI2_ROOT}/python $C/gate_profiles_v1.py --fm-dir $FM"
md5sum $C/gate_profiles_v1.py $FM/engine_fields_v16.py ${I}_fields.npz ${I}.json $P/C3lad32r2b/collapse_c32_rs64_A_R2_tau8_ell2_prep4_avg_nk40r2b_fields.npz \
  $P/C3lad32r2b/collapse_c32_rs64_A_R2_tau8_ell2_prep4_avg_nk40r2b.json $P/C3lad27/collapse_c27_rs64_A_R2_tau8_ell4_prep4_avg_fields.npz \
  $P/C3lad27/collapse_c27_rs64_A_R2_tau8_ell4_prep4_avg.json $P/C3lad27/collapse_c27_rs64_A_R2_tau8_ell8_prep4_avg_fields.npz \
  $P/C3lad27/collapse_c27_rs64_A_R2_tau8_ell8_prep4_avg.json > $O/inputs_md5.txt
nohup $PY --data ${I}_fields.npz --json ${I}.json --label ISO_128_s4 --T-from 9.0 --T-to 12.0625 --R-birth 78.175 --uncertified 12.0625 \
  --out $O/gate_ISO_128_s4.json > $O/gate_ISO_128_s4.log 2>&1 < /dev/null &
nohup $PY --data $P/C3lad32r2b/collapse_c32_rs64_A_R2_tau8_ell2_prep4_avg_nk40r2b_fields.npz --json $P/C3lad32r2b/collapse_c32_rs64_A_R2_tau8_ell2_prep4_avg_nk40r2b.json \
  --label P64_s2_c32 --T-from 9.0 --T-to 13.0 --R-birth 29.207 --out $O/gate_P64_s2.json > $O/gate_P64_s2.log 2>&1 < /dev/null &
nohup $PY --data $P/C3lad27/collapse_c27_rs64_A_R2_tau8_ell4_prep4_avg_fields.npz --json $P/C3lad27/collapse_c27_rs64_A_R2_tau8_ell4_prep4_avg.json \
  --label P64_s4_c27 --T-from 9.0 --T-to 13.5 --R-birth 31.669 --out $O/gate_P64_s4.json > $O/gate_P64_s4.log 2>&1 < /dev/null &
nohup $PY --data $P/C3lad27/collapse_c27_rs64_A_R2_tau8_ell8_prep4_avg_fields.npz --json $P/C3lad27/collapse_c27_rs64_A_R2_tau8_ell8_prep4_avg.json \
  --label P64_s8_c27 --T-from 9.0 --T-to 14.0 --R-birth 37.105 --out $O/gate_P64_s8.json > $O/gate_P64_s8.log 2>&1 < /dev/null &
echo launched
