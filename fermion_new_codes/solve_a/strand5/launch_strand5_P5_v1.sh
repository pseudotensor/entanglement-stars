#!/bin/bash
# Strand 5 P5 (SPEC_strand5_covariant_row.md sec. 3 P5; the production chain, 2 threads, after P3): strand5_P5_remainders_v1.py on J1's reference
# members, the I2b, G1 and G3 tables (copies of RESULT_strand5_family.md and RESULT_strand5_kernel_eos.md in P5run1/inputs) and the J4 family
# records.  r2's physical g_4/k_a(0) waits for the lead (the physical rectangle's massless packet).  Builder analyst.
cd ${QI2_ROOT}/s5_star || exit 1
mkdir -p P5run1
date -u '+queued %Y-%m-%d %H:%M:%S UTC' > P5run1/status.txt
until grep -q DONE P3run1/status.txt 2>/dev/null; do sleep 30; done
date -u '+start %Y-%m-%d %H:%M:%S UTC' >> P5run1/status.txt
CP=""
for inv in m0.1_aH1 m0.1_aH0.5 m0.03_aH0.3 m0.01_aH0.1; do
  for op in kernel local_ka0 local_klat; do
    CP="$CP${CP:+,}${inv}_${op}=J4run2/lstar_${inv}_${op}_g0_0.json:J4run2/lstar_${inv}_${op}_g0.2_1.json"
  done
done
for op in kernel local_ka0; do
  CP="$CP,m0.01_aH0.1_${op}_pk=J4run4/lstar_m0.01_aH0.1_${op}_g0_0_pk.json:J4run4/lstar_m0.01_aH0.1_${op}_g0.2_1_pk.json"
done
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 nice -n 10 ${QI2_ROOT}/python code/strand5_P5_remainders_v1.py \
  --family-md P5run1/inputs/RESULT_strand5_family.md --kernel-eos-md P5run1/inputs/RESULT_strand5_kernel_eos.md \
  --local-families "m0.01_aH0.1_pk=J4run4/lstar_m0.01_aH0.1_local_ka0_g0_0_pk.json,m0.03_aH0.3=J4run2/lstar_m0.03_aH0.3_local_ka0_g0_0.json" \
  --coupling-pairs "$CP" --md P5run1/RESULT_P5_tables_v1.md --out-json P5run1/P5.json > P5run1/run.log 2>&1
echo "P5 exit $? $(date -u +%H:%M:%S)" >> P5run1/status.txt
echo DONE >> P5run1/status.txt
