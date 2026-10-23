#!/bin/bash
# w2p_launch_busi.v36v_anc.sh (busi f52b1152, 2026-10-20 05:1x UTC): copy of w2p_launch_busi.v36v.sh (busi 9ae0d5c4, 2026-10-19): engine v36v (engine_collapse_v36v.py, md5 29111a34; built by build_engine_collapse_v36v.py over fixv4c b623d3b0)
# per busi/notes/W2_pass12625_ruling_reviewa_v1.md sec. 3 "v36v launch" and sec. 5. Copy of w2p_launch_busi.f4c_r2b.sh (R2prime, md5 53af637a), changed only:
#   ENGINE=engine_collapse_v36v.py; DGK = R2prime's DGK_BASE + dg_stall=8 dg_sub_counts=1,2,3,4,5,6,8, then dg_sub_max=6 dg_sub_start=1 dg_prim_halve=on;
#   tag suffix _v36v (plus optional $TAGX), pid file W2_dgp_v36v$TAGX.pid; the resume slice a parameter (SLICE_T, below).
# SLICE_T (env, default 12.50): 12.50 = the kit slice T0012.50 (4aa640af) with the args snapshot run (as R2prime); any other clock (12.75, 12.875, 13.00 ...)
#   = R2prime's own slice $ROOT/states/<TAGP>_r2b/orbitals_<TAGP>_r2b_T<clock>.npz with --resume-run $ROOT/runs/<TAGP>_r2b/<TAGP>_r2b (read only; R2prime's
#   files are never written). The slice's stored dg_level_v36k (if any) is the starting level and indexes the count list (printed below); the
#   T0012.50 slice stores none, so the start is dg_sub_start=1 = 2 pieces of tau 4.
# TAGX (env, default empty): an extra tag suffix for a second v36v launch (the launcher refuses an existing out dir, log or pid file).
# CHANGED ONLY: the resume source is the anchor fallback _anc (w2p_launch_busi.f4c_anc.sh) instead of R2prime _r2b: SLICE_T default 14.50 (ruling sec. 4: switch to dg_assign=prim at the 14.5 slice), slice $ROOT/states/<TAGP>_anc/..., run $ROOT/runs/<TAGP>_anc/<TAGP>_anc (read only); tag suffix _v36v_anc, pid W2_dgp_v36v_anc.pid.
# Usage: [SLICE_T=14.50] [TAGX=_b] w2p_launch_busi.v36v_anc.sh main [print]
set -e
ROOT=${QI2_ROOT}/ladders; CODE=${QI2_ROOT}/code; PY=${QI2_ROOT}/python
TAG=lad_W2_dg_m0p2_rs64_n2048_R04_k10_tau8_ell4_prep4_A_R2_avg_plain_v36f; TAGP=${TAG}_v36p
STAGE=${QI2_ROOT}/stage
RUN=$STAGE/runs/$TAG; SNAP=$RUN/args_v36p
SLICE_T=${SLICE_T:-14.50}; TAGX=${TAGX:-}
if false; then
  SLICE=$STAGE/w2_resume_v36k/orbitals_${TAG}_T0012.50.npz; RRUN=$SNAP/$TAG
else
  TR=${TAGP}_anc; SLICE=$ROOT/states/$TR/orbitals_${TR}_T$(printf %07.2f $SLICE_T).npz; RRUN=$ROOT/runs/$TR/$TR
fi
ENGINE=$CODE/engine_collapse_v36v.py
DGK_BASE="dg_chart=frozen dg_mix=anderson dg_anderson_m=4 dg_beta=0.5 dg_max_change=5e-3 dg_chain_tol=1e-8 dg_chain_trigger=off dg_attr=off dg_end_passes=2 dg_cap=100 dg_tol=1e-12 dg_assign=prim dg_prim_rule=lz dg_prim_c_lz=1 dg_prim_lz_k=1 dg_prim_tie=adiabatic dg_prim_level_min=1 dg_prim_level_max=5 dg_prim_wmix=0.9 dg_prim_eig_tol=1e-10 dg_prim_jump_tol=1e-9 dg_prim_knife=0.1 dg_prim_mirror=auto dg_prim_rounds=3 dg_prim_redecide=1e-6 dg_prim_settle=3 dg_prim_settle_max=10 dg_prim_flips=2"
DGK_BASE="$DGK_BASE dg_stall=8 dg_sub_counts=1,2,3,4,5,6,8"
SL128=0.0000000000,11.0000000000,11.5000000000,12.0000000000,12.1250000000,12.2500000000,12.5000000000,12.7500000000,12.8750000000,13.0000000000,13.5000000000,14.0000000000,14.5000000000,15.0000000000,15.5000000000,15.7500000000,15.8750000000,16.0000000000,16.1250000000,17.0000000000,17.3750000000,18.0000000000,18.3750000000,19.0000000000,20.0000000000,21.0000000000,22.0000000000,23.0000000000,24.0000000000,25.0000000000,26.0000000000,27.0000000000,28.0000000000,29.0000000000,30.0000000000,31.0000000000,32.0000000000,33.0000000000,34.0000000000,35.0000000000,36.0000000000,37.0000000000,38.0000000000,39.0000000000,40.0000000000
mkdir -p $ROOT/logs $ROOT/pids
[ "$1" = main ] || { echo "usage: [SLICE_T=14.50] [TAGX=_b] $0 main [print]"; exit 1; }
T_=${TAGP}_v36v_anc$TAGX; DGK="$DGK_BASE dg_sub_max=6 dg_sub_start=1 dg_prim_halve=on"; END=40; PIDF=$ROOT/pids/W2_dgp_v36v_anc$TAGX.pid
OUT=$ROOT/runs/$T_; STDP=$ROOT/states/$T_; LOGP=$ROOT/logs/$T_.log
ARGS=$($PY $CODE/w2_cmd_from_args.py $SNAP/collapse_$TAG.json $DGK slices_c128=$SL128 T_over_rs=$END gpu=0 out=$OUT tag=$T_ slice_dir=$STDP)
echo "$ARGS" | tr ' ' '\n' | grep -A1 -E "^--(step|dg-|mix|tau|T-over-rs|resume|out|tag|slice-dir|gpu|slices-c128)" | paste -sd' ' | fold -w 250
echo "engine $ENGINE md5 $(md5sum < $ENGINE | cut -c1-8)"
echo "resume: SLICE_T $SLICE_T slice $SLICE ($(test -e $SLICE && md5sum < $SLICE | cut -c1-8 || echo MISSING)) run $RRUN ($(test -e $(dirname $RRUN)/collapse_$(basename $RRUN).json && echo json present || echo json MISSING))"
test -e $SLICE && $PY -c "import numpy as np, sys; Z = np.load(sys.argv[1]); c = [1, 2, 3, 4, 5, 6, 8]; L = int(Z['dg_level_v36k']) if 'dg_level_v36k' in Z.files else None; s = (('stored dg_level_v36k %d -> the first attempt %d pieces of tau %g' % (L, c[L], 8.0 / c[L])) if L is not None else 'no stored level: dg_sub_start 1 -> the first attempt 2 pieces of tau 4'); print('slice T/rs', float(Z['T']) / 64.0, s)" $SLICE
echo "out $OUT log $LOGP pid file $PIDF"
if [ "$2" != print ]; then
  test -e $SLICE || { echo "refusing: slice $SLICE missing"; exit 1; }
  for f in $OUT $LOGP $PIDF $PIDF.rc; do test ! -e $f || { echo "refusing: $f exists"; exit 1; }; done
  mkdir -p $OUT $STDP
  export NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  cd $OUT
  setsid nohup bash -c "nice -n 10 $PY $ENGINE $ARGS --resume-slice $SLICE --resume-run $RRUN > $LOGP 2>&1 < /dev/null & echo \$! > $PIDF; wait \$!; echo \$? > $PIDF.rc" > /dev/null 2>&1 < /dev/null &
  sleep 3; echo "launched $T_ pid $(cat $PIDF) log $LOGP"
fi
