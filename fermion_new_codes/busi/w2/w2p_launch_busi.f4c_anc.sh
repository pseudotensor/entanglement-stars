#!/bin/bash
# w2p_launch_busi.f4c_anc.sh (busi f52b1152, 2026-10-20 04:1x UTC): the NO-CODE FALLBACK of busi/notes/W2_pass12625_ruling_reviewa_v1.md sec. 4/5:
# v36v STOPped 04:03 UTC at 13.625 -> 13.75 ("the deepest level (6) reached", NOT CONVERGED), so the epoch runs under dg_assign=anchor with the
# comparator's (v36l's) solver from T0012.50 to 14.5; the switch to dg_assign=prim is made later at this run's 14.5 slice (a separate launch).
# Copy of w2p_launch_busi.f4c_r2b.sh (R2prime), changed only: DGK = the args snapshot's own dg settings (= v36l's: anderson m4, beta 1.0,
# max_change 2e-2, sub_max 3, sub_start 0, cap 40, chain_trigger off) plus dg_assign=anchor (the engine default, stated) and dg_attr=off
# (diagnostic only, no effect on the fixed point, ruling sec. 2 item 7); T_over_rs 14.5; tag suffix _anc; pid file W2_dgp_anc.pid.
# Engine fixv4c b623d3b0 (bitwise v36l at defaults). Usage: w2p_launch_busi.f4c_anc.sh main [print]
set -e
ROOT=${QI2_ROOT}/ladders; CODE=${QI2_ROOT}/code; PY=${QI2_ROOT}/python
TAG=lad_W2_dg_m0p2_rs64_n2048_R04_k10_tau8_ell4_prep4_A_R2_avg_plain_v36f; TAGP=${TAG}_v36p
STAGE=${QI2_ROOT}/stage
RUN=$STAGE/runs/$TAG; SNAP=$RUN/args_v36p; SLICE=$STAGE/w2_resume_v36k/orbitals_${TAG}_T0012.50.npz
ENGINE=$CODE/engine_collapse_v36p_fixv4c.py
SL128=0.0000000000,11.0000000000,11.5000000000,12.0000000000,12.1250000000,12.2500000000,12.5000000000,12.7500000000,12.8750000000,13.0000000000,13.2500000000,13.5000000000,13.7500000000,14.0000000000,14.2500000000,14.5000000000
mkdir -p $ROOT/logs $ROOT/pids
[ "$1" = main ] || { echo "usage: $0 main [print]"; exit 1; }
T_=${TAGP}_anc; DGK="dg_assign=anchor dg_attr=off dg_mix=anderson dg_anderson_m=4 dg_beta=1.0 dg_max_change=2e-2 dg_sub_max=3 dg_sub_start=0 dg_cap=40 dg_chain_trigger=off"; END=14.5; PIDF=$ROOT/pids/W2_dgp_anc.pid
OUT=$ROOT/runs/$T_; STDP=$ROOT/states/$T_; LOGP=$ROOT/logs/$T_.log
ARGS=$($PY $CODE/w2_cmd_from_args.py $SNAP/collapse_$TAG.json $DGK slices_c128=$SL128 T_over_rs=$END gpu=0 out=$OUT tag=$T_ slice_dir=$STDP)
echo "$ARGS" | tr ' ' '\n' | grep -A1 -E "^--(step|dg-|mix|tau|T-over-rs|resume|out|tag|slice-dir|gpu|slices-c128)" | paste -sd' ' | fold -w 250
echo "engine $ENGINE md5 $(md5sum < $ENGINE | cut -c1-8); slice $(md5sum < $SLICE | cut -c1-8)"
if [ "$2" != print ]; then
  for f in $OUT $LOGP $PIDF $PIDF.rc; do test ! -e $f || { echo "refusing: $f exists"; exit 1; }; done
  mkdir -p $OUT $STDP
  export NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  cd $OUT
  setsid nohup bash -c "nice -n 10 $PY $ENGINE $ARGS --resume-slice $SLICE --resume-run $SNAP/$TAG > $LOGP 2>&1 < /dev/null & echo \$! > $PIDF; wait \$!; echo \$? > $PIDF.rc" > /dev/null 2>&1 < /dev/null &
  sleep 3; echo "launched $T_ pid $(cat $PIDF) log $LOGP"
fi
