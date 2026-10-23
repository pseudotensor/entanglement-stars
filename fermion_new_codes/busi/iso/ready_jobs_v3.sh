#!/bin/bash
# ready_jobs_v3.sh MODE (busi 9ae0d5c4, 2026-10-19; early_failures_ruling_v3 1.2, 2.1, 2.2, 4.1; specs in busi/results/ready_jobs_v3.md).
# Card-queue entries: each mode blocks until its engine exits and never kills anything itself (2.2's and 4.1a's watch_kept4_v1.sh may end
# its engine by the ruled rule). NUMPY_MADVISE_HUGEPAGE=0. Engines unchanged: v36q 759c526c (RUNG2), v36n 0c84a446 (ISO). Own tags and dirs;
# stored runs only read (resume across a mix goes through a snapshot copy whose json has args.mix edited, the v34 resume check).
#   pod 8 (ruling 1.2, staged dir D8 copied from R3; 2.1):
#     r2i            0.375 -> 0.500 plain --maxit 60, slice at 0.5 (from R3's Newton 0.375 slice)
#     r2ii_plain     0.500 (r2i's slice) -> 0.625 plain --maxit 60
#     r2ii_anderson  same, --mix anderson
#     r2ii_damped    same, --mix damped --mix-alpha 0.5 (the engine's damping flag)
#     r2iii          0.500 (R3's Newton 0.500 slice) -> 0.625 --mix anderson --maxit 60
#     r2all          r2i, then the three r2ii, then r2iii (about 20 min)
#     isonewton      2.1: A(ii)'s ISO size control (0.05, 128, 1536 sites, kappa_sea 16, T/rs 0 to 0.5) with --mix newton and J2's NK flags
#   pod 7 (ruling 2.2):
#     m3d            ISO_M3d_s4 from T = 0 under --mix anderson --maxit 30 --end-above 0.001, early slices; watch 2 > 1e-5 or 1 > 1e-4
#   pod 6 (ruling 4.1, after ISO_128_s4 ends):
#     m3c_a          M3c's line fresh from T = 0 to 2.5833 (the last converged record before 2.667), slices 0 and 2.5833; watch 4 > 1e-3
#     m3c_b          three Newton records 2.667 to 2.833 from m3c_a's slice (J2's NK flags), slice 2.8333
#     m3c_c          ten plain records 2.917 to 3.667 from m3c_b's slice
#     m3c_all        m3c_a, m3c_b, m3c_c in order (each refuses when its input slice is missing)
export NUMPY_MADVISE_HUGEPAGE=0
MODE=${1:?mode}
R=${QI2_ROOT}/ladders; PY=${QI2_ROOT}/python; C=${QI2_ROOT}/code_v36d
D8=$R/diag_early_failures_v3; H=$D8/cmd_from_args_v1.py; WK=$R/watch_kept4_v1.sh
NK="--nk-krylov 40 --nk-halvings 6 --nk-inner 15 --nk-jvp exact --nk-line-search halving --nk-set resolve --nk-maxiter 30 --nk-restarts 2 --end-above 0.001"
now() { date -u +%FT%TZ; }
STF=${STF:-$R/STATUS_ready_v3}
st() { echo "$(now) $MODE: $*" | tee -a $STF; }
md5ok() { [ "$(md5sum $1 | cut -c1-32)" = $2 ] || { st "md5 mismatch on $1 (want $2): refused"; exit 1; }; }
snap() {   # $1 source prefix dir/tag, $2 snap dir, $3 mix: copy the run's four files, set args.mix in the copy only
  mkdir -p $2; cp $1.json $1_fields.npz $1_handoff.npz $1_spectra.npz $2/ || return 1
  $PY -c "
import json,sys
p=sys.argv[1]; d=json.load(open(p)); d['args']['mix']=sys.argv[2]
d['_busi_note']='busi ready_jobs_v3: snapshot copy; args.mix set to '+sys.argv[2]+' for the v34 resume check (original untouched)'
json.dump(d,open(p,'w'))" $2/$(basename $1).json $3; }
run() {   # $1 engine, $2 log, $3 member name for the pid file ('' for none), $4 watch args ('' for none), rest: engine arguments
  local E=$1 L=$2 M=$3 WA=$4; shift 4
  mkdir -p $R/pids $R/logs $(dirname $L)
  st "start: $E $* (log $L)"
  nice -n 10 $PY $E "$@" > $L 2>&1 < /dev/null &
  local P=$!; [ -n "$M" ] && echo $P > $R/pids/$M.pid
  if [ -n "$WA" ]; then sleep 5; setsid nohup bash $WK $M $WA > /dev/null 2>&1 < /dev/null & st "engine pid $P; watch_kept4_v1.sh $M $WA attached"; else st "engine pid $P (no watch)"; fi
  wait $P; local rc=$?; [ -n "$M" ] && echo $rc > $R/pids/$M.pid.rc
  st "end rc $rc; records $(grep -c '^T/rs [0-9.]* E/E0' $L); last $(grep -o '^T/rs [0-9.]*' $L | tail -1); unconverged $(grep -c UNCONVERGED $L)"
  return $rc; }

# ---- pod 8: RUNG2 probes (engine v36q) ----
E36Q=$C/engine_collapse_v36q.py
N375=diagEF_RUNG2_newton_to0p375_v36q; N500=diagEF_RUNG2_newton_to0p5_v36q
S375=$D8/rung2_newton_to0p375/states/orbitals_${N375}_T0000.38.npz
S500N=$D8/rung2_newton_to0p5/states/orbitals_${N500}_T0000.50.npz
TPL2=$D8/templates/collapse_diagEF_RUNG2_plain0p625_from0p375_v36q.json
r2i() {
  md5ok $E36Q 759c526c8713d926cd37572f59656e62; md5ok $S375 63920bd4e2af540bbdc79531d27e2502
  local W=$D8/r2i T=diagEF3_RUNG2_r2i_plain0p5_v36q
  test ! -e $W/run/collapse_$T.json || { st "refused: $W/run/collapse_$T.json exists"; return 1; }
  snap $D8/rung2_newton_to0p375/run/collapse_$N375 $W/snap plain || { st "snapshot failed"; return 1; }
  mkdir -p $W/run $W/states; cd $W/run
  run $E36Q $W/$T.log "" "" $($PY $H $TPL2 out=$W/run tag=$T slice_dir=$W/states slices_c128=0.5 slices_c64= maxit=60 mix=plain) \
      --resume-slice $S375 --resume-run $W/snap/$N375 --segment-end-T-over-rs 0.5; }
r2ii() {   # $1 mix
  md5ok $E36Q 759c526c8713d926cd37572f59656e62
  local X=$1 T0=diagEF3_RUNG2_r2i_plain0p5_v36q; local S=$D8/r2i/states/orbitals_${T0}_T0000.50.npz
  [ -s $S ] || { st "no r2i 0.500 slice ($S): refused"; return 1; }
  local W=$D8/r2ii_$X T=diagEF3_RUNG2_r2ii_${X}0p625_v36q
  test ! -e $W/run/collapse_$T.json || { st "refused: $W/run/collapse_$T.json exists"; return 1; }
  snap $D8/r2i/run/collapse_$T0 $W/snap $X || { st "snapshot failed"; return 1; }
  mkdir -p $W/run; cd $W/run
  run $E36Q $W/$T.log "" "" $($PY $H $TPL2 out=$W/run tag=$T slices_c128= slices_c64= maxit=60 mix=$X mix_alpha=0.5) \
      --resume-slice $S --resume-run $W/snap/$T0 --segment-end-T-over-rs 0.625; }
r2iii() {
  md5ok $E36Q 759c526c8713d926cd37572f59656e62; md5ok $S500N 1f3cd9e9af65d0c847d9ef504b192341
  local W=$D8/r2iii T=diagEF3_RUNG2_r2iii_anderson0p625_fromN0p5_v36q
  test ! -e $W/run/collapse_$T.json || { st "refused: $W/run/collapse_$T.json exists"; return 1; }
  snap $D8/rung2_newton_to0p5/run/collapse_$N500 $W/snap anderson || { st "snapshot failed"; return 1; }
  mkdir -p $W/run; cd $W/run
  run $E36Q $W/$T.log "" "" $($PY $H $TPL2 out=$W/run tag=$T slices_c128= slices_c64= maxit=60 mix=anderson) \
      --resume-slice $S500N --resume-run $W/snap/$N500 --segment-end-T-over-rs 0.625; }

# ---- pod 8: 2.1 the Newton size control (engine v36n) ----
E36N=$C/engine_collapse_v36n.py
isonewton() {
  md5ok $E36N 0c84a4466cd50a57621526faade1b2d3
  local TPL=$D8/templates/collapse_diagEF_ISO_size_m0p05_rs128_n1536_k16_v36n.json
  local W=$D8/iso_size_newton T=diagEF3_ISOnewton_m0p05_rs128_n1536_k16_v36n
  test ! -e $W/run/collapse_$T.json || { st "refused: $W/run/collapse_$T.json exists"; return 1; }
  mkdir -p $W/run; cd $W/run
  run $E36N $W/$T.log "" "" $($PY $H $TPL out=$W/run tag=$T mix=newton) $NK; }

# ---- pod 7: 2.2 the ISO_M3d_s4 retry (engine v36n) ----
m3d() {
  md5ok $E36N 0c84a4466cd50a57621526faade1b2d3
  local M=ISO_M3d_s4 T=lad_ISO_M3d_s4_m0p05_rs326_n3912_R04_k22_tau8_ell4_prep4_A_R2_avg_anderson_v36n
  local RUN=$R/runs/$T STD=$R/states/$T
  test ! -e $RUN/collapse_$T.json || { st "refused: $RUN/collapse_$T.json exists (resume is the launcher's, not this entry's)"; return 1; }
  mkdir -p $RUN $STD; cd $RUN
  run $E36N $R/logs/$T.log $M "2 1e-5 2 1e-4" --m 0.05 --g 0.25 --rs 326 --R0-over-rs 4 --reading A --nsites 3912 --kappa-sea 22 --reference R2 \
      --site-fields averaged --prep-ell 4 --ell 4 --tau 8 --T-over-rs 30 --gpu 0 --mix anderson --maxit 30 --end-above 0.001 --field-kernel-cache --v36-fixes \
      --slices-c128 0.0000000000,0.2453987730,0.4907975460,1.0061349693,1.4969325153,2.0122699387,2.5030674847,2.9938650307 \
      --dust-set isotropic --iter-cache drop --prep-memory lean --slice-dir $STD --out $RUN --tag $T; }

# ---- pod 6: 4.1 the M3c episode test (engine v36n) ----
M3CB="--m 0.1 --g 0.25 --rs 96 --R0-over-rs 4 --reading A --nsites 3072 --kappa-sea 12 --reference R2 --site-fields averaged --prep-ell 4 --ell 4 --tau 8 --T-over-rs 40 --gpu 0 --field-kernel-cache --v36-fixes --dust-set isotropic --iter-cache drop"
EP=$R/m3c_episode_v3
TA=lad_ISO_M3c_epA_m0p1_rs96_n3072_R04_k12_tau8_ell4_prep4_A_R2_avg_plain_v36n
TB=lad_ISO_M3c_epB_m0p1_rs96_n3072_R04_k12_tau8_ell4_prep4_A_R2_avg_newton_v36n
TC=lad_ISO_M3c_epC_m0p1_rs96_n3072_R04_k12_tau8_ell4_prep4_A_R2_avg_plain_v36n
m3c_a() {
  md5ok $E36N 0c84a4466cd50a57621526faade1b2d3
  test ! -e $EP/A/run/collapse_$TA.json || { st "refused: $EP/A/run/collapse_$TA.json exists"; return 1; }
  mkdir -p $EP/A/run $EP/A/states; cd $EP/A/run
  run $E36N $R/logs/$TA.log ISO_M3c_epA "4 1e-3 4" $M3CB --mix plain --slices-c128 0.0000000000,2.5833333333 \
      --slice-dir $EP/A/states --out $EP/A/run --tag $TA --segment-end-T-over-rs 2.5833333333; }
m3c_b() {
  md5ok $E36N 0c84a4466cd50a57621526faade1b2d3
  local S=$EP/A/states/orbitals_${TA}_T0002.58.npz
  [ -s $S ] || { st "no m3c_a 2.5833 slice ($S): refused"; return 1; }
  test ! -e $EP/B/run/collapse_$TB.json || { st "refused: $EP/B/run/collapse_$TB.json exists"; return 1; }
  snap $EP/A/run/collapse_$TA $EP/B/snap newton || { st "snapshot failed"; return 1; }
  mkdir -p $EP/B/run $EP/B/states; cd $EP/B/run
  run $E36N $R/logs/$TB.log "" "" $M3CB --mix newton $NK --slices-c128 2.8333333333 --slice-dir $EP/B/states --out $EP/B/run --tag $TB \
      --resume-slice $S --resume-run $EP/B/snap/$TA --segment-end-T-over-rs 2.8333333333; }
m3c_c() {
  md5ok $E36N 0c84a4466cd50a57621526faade1b2d3
  local S=$EP/B/states/orbitals_${TB}_T0002.83.npz
  [ -s $S ] || { st "no m3c_b 2.8333 slice ($S): refused"; return 1; }
  test ! -e $EP/C/run/collapse_$TC.json || { st "refused: $EP/C/run/collapse_$TC.json exists"; return 1; }
  snap $EP/B/run/collapse_$TB $EP/C/snap plain || { st "snapshot failed"; return 1; }
  mkdir -p $EP/C/run $EP/C/states; cd $EP/C/run
  run $E36N $R/logs/$TC.log "" "" $M3CB --mix plain --slices-c128 3.6666666667 --slice-dir $EP/C/states --out $EP/C/run --tag $TC \
      --resume-slice $S --resume-run $EP/C/snap/$TB --segment-end-T-over-rs 3.6666666667; }

case $MODE in
  r2i) r2i;; r2ii_plain) r2ii plain;; r2ii_anderson) r2ii anderson;; r2ii_damped) r2ii damped;; r2iii) r2iii;;
  r2all) r2i && { r2ii plain; r2ii anderson; r2ii damped; }; r2iii;;
  isonewton) isonewton;;
  m3d) m3d;;
  m3c_a) m3c_a;; m3c_b) m3c_b;; m3c_c) m3c_c;; m3c_all) m3c_a && m3c_b && m3c_c;;
  *) echo "unknown mode $MODE"; exit 2;;
esac
