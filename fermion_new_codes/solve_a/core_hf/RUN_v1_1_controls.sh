#!/bin/bash
# CORE_HF_full v1.1 controls: 4 CPU reruns with core_hf_slices_v1_1.py (--no-response; the card runs dressed betas through --ref-arrays), 2 queues x 6 threads
C=${QI2_ROOT}/qi2_corehf_v1_1_code; B=${QI2_ROOT}/core_hf_v1; L=$B/v1_1/logs; PY=${QI2_ROOT}/python
q1() {
  nice -n 10 $PY -B $C/core_hf_slices_v1_1.py --history prod --clocks 16 --no-response --threads 6 --dr-refine 0.05,0.025 \
     --ref-arrays $B/card_bench1b/prod/core_hf_prod_T0016.00_arrays.npz --out $B/v1_1/runs/c128 > $L/run_prod16_c128.log 2>&1
  nice -n 10 $PY -B $C/core_hf_slices_v1_1.py --history prod --clocks 12 --no-response --threads 6 \
     --ref-arrays $B/card_bench1b/prod/core_hf_prod_T0012.00_arrays.npz --out $B/v1_1/runs/p12 > $L/run_prod12.log 2>&1
}
q2() {
  nice -n 10 $PY -B $C/core_hf_slices_v1_1.py --history prod --clocks 16 --no-response --threads 6 --slice-dir $B/v1_1/c64r_slice \
     --ref-arrays $B/card_bench1b/prod/core_hf_prod_T0016.00_arrays.npz --out $B/v1_1/runs/c64r > $L/run_prod16_c64r.log 2>&1
  nice -n 10 $PY -B $C/core_hf_slices_v1_1.py --history s8 --clocks 12 --no-response --threads 6 --slice-dir $B/s8_slices \
     --ref-arrays $B/card_R2/s8/core_hf_s8_T0012.00_arrays.npz --out $B/v1_1/runs/s8 > $L/run_s8_12.log 2>&1
}
q1 & q2 & wait
echo ALL_DONE > $L/controls_done.flag
