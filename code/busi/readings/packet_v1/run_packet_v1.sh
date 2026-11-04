#!/bin/bash
# run_packet_v1.sh (busi lane, reviewer A slot, inbox #354): the packet completeness readings on the box (CPU nice 10, 8 threads; GPU 1 for the
# dense eigendecompositions only, free by nvidia-smi at launch). Usage: bash run_packet_v1.sh {quick|full|bounds|cov T0016.00|...}
set -u
PY=${QI2_ROOT}/python
CODE=${QI2_ROOT}/code
OUT=${QI2_ROOT}/packet_v1
D=${QI2_ROOT}/C3prod19
TAG=collapse_c19_rs64_A_R2_tau8_ell4_avg_r11
ENGINE=${QI2_ROOT}/code
STATES=${QI2_ROOT}/c19_rs64_A_R2_tau8_ell4_avg_r11
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
cd "$CODE" || exit 1
case "$1" in
quick)  nice -n 10 $PY packet_v1_propagate.py --fields $D/${TAG}_fields.npz --json $D/$TAG.json --out $OUT/quick --engine-code $ENGINE --gpu 1 --quick --channels 1 ;;
full)   nice -n 10 $PY packet_v1_propagate.py --fields $D/${TAG}_fields.npz --json $D/$TAG.json --out $OUT/full --engine-code $ENGINE --gpu 1 --channels 1 5 ;;
full2)  nice -n 10 $PY packet_v1_propagate.py --fields $D/${TAG}_fields.npz --json $D/$TAG.json --out $OUT/full2 --engine-code $ENGINE --gpu 1 --channels 1 5 ;;
tables) nice -n 10 $PY packet_v1_tables.py --summary $OUT/full2/packet_v1_summary.json --series $OUT/full2/packet_v1_series.npz --bounds $OUT/bounds/packet_v1_bounds.json --cov $OUT/cov --out $OUT/full2/packet_v1_tables.md ;;
fig)    nice -n 10 $PY packet_v1_fig.py --series $OUT/full2/packet_v1_series.npz --bounds $OUT/bounds/packet_v1_bounds.npz --out $OUT/full2/REVIEWA_packet_completeness_v1.png ;;
bounds) nice -n 10 $PY packet_v1_bounds.py --fields $D/${TAG}_fields.npz --json $D/$TAG.json --spectra $D/${TAG}_spectra.npz --out $OUT/bounds --engine-code $ENGINE ;;
cov)    nice -n 10 $PY packet_v1_covariance.py --slice $STATES/orbitals_${TAG#collapse_}_$2.npz --fields $D/${TAG}_fields.npz --json $D/$TAG.json --out $OUT/cov --gpu 1 ;;
*) echo "usage: $0 {quick|full|bounds|cov Txxxx.xx}"; exit 2 ;;
esac
