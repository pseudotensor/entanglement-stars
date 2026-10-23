#!/bin/bash
# rd13_chain_v1.sh (busi, analyst; RD-1/RD-3 self-test on the box, sequential, one GPU): (b) the per-call test, (a) the fixture gate (CPU
# pairs + the GPU smoke), (c) the ISO_128 replay arms B (banded + sternheimer, 3 records), D (defaults, 3 records), P (the parent v36nq,
# the first record), then the comparison. GPU = the box GPU index (never 0).
set -u
B=${QI2_ROOT}/RD1_build; PY=${QI2_ROOT}/python; export GPU=${GPU:-2}
W=${QI2_ROOT}/T0013.50
WT=lad_W2_dg_m0p2_rs64_n2048_R04_k10_tau8_ell4_prep4_A_R2_avg_plain_v36f_v36p_r2b
I=${QI2_ROOT}/iso128_h2; T=lad_ISO_128_s4_nkqh2_m0p2_rs128_n4096_R04_k10_tau8_ell4_prep4_A_R2_avg_newton_v36nq
L=$B/logs/chain_v1.log
say() { echo "[$(date -u +%FT%TZ)] $*" >> $L; }
cd $B
md5sum code/engine_banded_eigh_v1.py code/engine_collapse_v36nq_rd13.py code/engine_collapse_v36nq.py code/engine_fields_v16.py > $B/logs/chain_md5_v1.txt
step=${1:-all}
if [ "$step" = all ] || [ "$step" = b ]; then
  say "(b) per-call start"; mkdir -p $B/percall
  env CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 nice -n 10 $PY $B/build/rd13_percall_v1.py $B/percall $B/code \
    --w2 $W/orbitals_${WT}_T0013.50.npz --w2-fields $W/collapse_${WT}_fields.npz --iso $I/states_h2/orbitals_${T}_T0005.00.npz \
    --iso-fields $I/run_h2/collapse_${T}_fields.npz --synthetic 8192 > $B/logs/percall_v2.log 2>&1
  say "(b) per-call rc $?"
fi
if [ "$step" = all ] || [ "$step" = a ]; then
  say "(a) fixture gate start"
  env CUDA_VISIBLE_DEVICES=$GPU nice -n 10 $PY $B/build/rd13_fixture_gate_v1.py --code $B/code --out $B/fixture --threads 8 > $B/logs/fixture_gate_v1.log 2>&1
  say "(a) fixture gate rc $?"
fi
if [ "$step" = all ] || [ "$step" = c ]; then
  say "(c) arm B start"; bash $B/build/rd13_replay_v1.sh B engine_collapse_v36nq_rd13.py 5.1875 --eigh banded --jvp sternheimer; say "(c) arm B rc $?"
  say "(c) arm D start"; bash $B/build/rd13_replay_v1.sh D engine_collapse_v36nq_rd13.py 5.1875; say "(c) arm D rc $?"
  say "(c) arm P start"; bash $B/build/rd13_replay_v1.sh P engine_collapse_v36nq.py 5.0625; say "(c) arm P rc $?"
  nice -n 10 $PY $B/build/cmp_rd13_replay_v1.py --replay $B/replay --prod $I/run_h2/$T --arms D,B --parent P --t0 5.0 --out $B/replay/cmp_rd13_replay_v1 > $B/logs/cmp_v1.log 2>&1
  say "(c) cmp rc $?"
fi
say "chain done"
