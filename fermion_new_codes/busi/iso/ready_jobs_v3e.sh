#!/bin/bash
# v3e (busi f52b1152, 20 Oct, ISO_128_nk_route_v2): v3d plus the modes iso128_nk_q0, iso128_nk_q0_dry, iso128_nk_q and iso128_nk_q_dry (ISO_128_s4 on the composite v36nq 8549c542, pod6); nothing else changed.
# v3d (busi 9ae0d5c4, 01:2x UTC 20 Oct, #312): v3c plus the mode iso128_nk_r (ISO_128_s4 Newton restart on v36r 7b6d0f4f from the certified 11.5 slice, pod6) and its dry check iso128_nk_r_dry; nothing else changed.
# v3c (busi 9ae0d5c4, 19 Oct, #303): v3b plus the mode m3c_nk_r (m3c_nk on v36r 7b6d0f4f with --nk-plain-fallback on; own tag and dir N_r); nothing else changed.
# v3b (busi 9ae0d5c4, 21:2x UTC 19 Oct, inbox #303): v3 plus the mode m3c_nk (the M3c Newton continuation to 40); nothing else changed.
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

# ---- pod7b: #303 the M3c Newton continuation (engine v36n), busi 9ae0d5c4 21:2x UTC 19 Oct (v3b addition) ----
# From m3c_a's certified 2.5833 slice (B's own 2.8333 record ended UNCONVERGED at kept residual 1.08e-3, so its slice is not a certified start).
# J2's NK flags with --end-above 0.002 in place of 0.001: B's 2.8333 (1.08e-3) passes, a real failure still ends the member.
# The lead may restore 0.001 with END=0.001 in the environment. Slices 2.8333, 3, 3.5, 4, then every 2 r_s to 40 (resume points; about 7 GB each).
TN=lad_ISO_M3c_nk_m0p1_rs96_n3072_R04_k12_tau8_ell4_prep4_A_R2_avg_newton_v36n
m3c_nk() {
  md5ok $E36N 0c84a4466cd50a57621526faade1b2d3
  local S=$EP/A/states/orbitals_${TA}_T0002.58.npz E=${END:-0.002}
  [ -s $S ] || { st "no m3c_a 2.5833 slice ($S): refused"; return 1; }
  test ! -e $EP/N/run/collapse_$TN.json || { st "refused: $EP/N/run/collapse_$TN.json exists (resume it with its ladder resume instead)"; return 1; }
  snap $EP/A/run/collapse_$TA $EP/N/snap newton || { st "snapshot failed"; return 1; }
  mkdir -p $EP/N/run $EP/N/states; cd $EP/N/run
  run $E36N $R/logs/$TN.log ISO_M3c_nk "" $M3CB --mix newton ${NK% --end-above 0.001} --end-above $E \
      --slices-c128 2.8333333333,3.0000000000,3.5000000000,4.0000000000,6.0000000000,8.0000000000,10.0000000000,12.0000000000,14.0000000000,16.0000000000,18.0000000000,20.0000000000,22.0000000000,24.0000000000,26.0000000000,28.0000000000,30.0000000000,32.0000000000,34.0000000000,36.0000000000,38.0000000000,40.0000000000 \
      --slice-dir $EP/N/states --out $EP/N/run --tag $TN --resume-slice $S --resume-run $EP/N/snap/$TA; }


# ---- pod7b: #303 the M3c Newton continuation on v36r (the v36q plain-direction fallback on v36n), busi 9ae0d5c4 (v3c addition) ----
# m3c_nk_r = m3c_nk with the engine v36r (md5 7b6d0f4f; build_engine_collapse_v36r.py over v36n 0c84a446) and --nk-plain-fallback on;
# everything else as m3c_nk (A's certified 2.5833 slice, J2's NK flags, --end-above ${END:-0.002}, the same slices); own tag, dir N_r, pid name.
E36R=$C/engine_collapse_v36r.py
TNR=lad_ISO_M3c_nkr_m0p1_rs96_n3072_R04_k12_tau8_ell4_prep4_A_R2_avg_newton_v36r
m3c_nk_r() {
  md5ok $E36R 7b6d0f4f2db82a98780bdca78166f61b
  local S=$EP/A/states/orbitals_${TA}_T0002.58.npz E=${END:-0.002}
  [ -s $S ] || { st "no m3c_a 2.5833 slice ($S): refused"; return 1; }
  test ! -e $EP/N_r/run/collapse_$TNR.json || { st "refused: $EP/N_r/run/collapse_$TNR.json exists (resume it with its ladder resume instead)"; return 1; }
  snap $EP/A/run/collapse_$TA $EP/N_r/snap newton || { st "snapshot failed"; return 1; }
  mkdir -p $EP/N_r/run $EP/N_r/states; cd $EP/N_r/run
  run $E36R $R/logs/$TNR.log ISO_M3c_nkr "" $M3CB --mix newton ${NK% --end-above 0.001} --end-above $E --nk-plain-fallback on \
      --slices-c128 2.8333333333,3.0000000000,3.5000000000,4.0000000000,6.0000000000,8.0000000000,10.0000000000,12.0000000000,14.0000000000,16.0000000000,18.0000000000,20.0000000000,22.0000000000,24.0000000000,26.0000000000,28.0000000000,30.0000000000,32.0000000000,34.0000000000,36.0000000000,38.0000000000,40.0000000000 \
      --slice-dir $EP/N_r/states --out $EP/N_r/run --tag $TNR --resume-slice $S --resume-run $EP/N_r/snap/$TA; }


# ---- pod6: #312 the ISO_128_s4 Newton restart on v36r, busi 9ae0d5c4 (v3d addition) ----
# ISO_128_s4 (v36n plain) converged to 11.938; from 12.000 every record is UNCONVERGED at the 21-iteration cap (residuals 5e-3 to 4e-2).
# This member restarts from the certified 11.5 record (converged in 5 iterations; slice md5 b9d4b76e, staged in $I8/inputs from the box's mirror)
# with the run's exact base line (its json args), --mix newton, J2's NK flags with --end-above ${END:-0.002} (as m3c_nk), --nk-plain-fallback on.
# The resume reads the PRE-MADE snapshot $I8/snap (the live run's four files copied 01:22 UTC 20 Oct, args.mix set to newton in the copy only;
# 185 of its 201 records are at or before 11.5, which the v34 resume keeps). The live run is never read. Slices 12, 12.5, 13, 14, then every
# 2 r_s to 40 (16 slices, 10.7 GB each). SLICE_MD5=0 skips the 10.7 GB md5 (about 1 to 2 min) at start.
I8=$R/iso128_nk
T128=lad_ISO_128_s4_m0p2_rs128_n4096_R04_k10_tau8_ell4_prep4_A_R2_avg_plain_v36n
T128R=lad_ISO_128_s4_nkr_m0p2_rs128_n4096_R04_k10_tau8_ell4_prep4_A_R2_avg_newton_v36r
S128=$I8/inputs/orbitals_${T128}_T0011.50.npz
B128="--m 0.2 --g 0.25 --rs 128 --R0-over-rs 4 --reading A --nsites 4096 --kappa-sea 10 --reference R2 --site-fields averaged --prep-ell 4 --ell 4 --tau 8 --T-over-rs 40 --gpu 0 --field-kernel-cache --v36-fixes --dust-set isotropic --iter-cache drop --prep-memory lean"
SL128=12.0000000000,12.5000000000,13.0000000000,14.0000000000,16.0000000000,18.0000000000,20.0000000000,22.0000000000,24.0000000000,26.0000000000,28.0000000000,30.0000000000,32.0000000000,34.0000000000,36.0000000000,38.0000000000,40.0000000000
iso128_args() { echo $B128 --mix newton ${NK% --end-above 0.001} --end-above ${END:-0.002} --nk-plain-fallback on \
      --slices-c128 $SL128 --slice-dir $I8/states --out $I8/run --tag $T128R --resume-slice $S128 --resume-run $I8/snap/$T128; }
iso128_pre() {   # the checks both modes share
  md5ok $E36R 7b6d0f4f2db82a98780bdca78166f61b
  [ -s $S128 ] || { st "no 11.5 slice ($S128): refused"; return 1; }
  [ "${SLICE_MD5:-1}" = 0 ] || md5ok $S128 b9d4b76e43d27c1ee96ab3df9c07264f
  for f in .json _fields.npz _handoff.npz _spectra.npz; do [ -s $I8/snap/collapse_$T128$f ] || { st "no snapshot file $I8/snap/collapse_$T128$f: refused"; return 1; }; done
  grep -q '"mix": "newton"' $I8/snap/collapse_$T128.json || { st "snapshot json args.mix is not newton: refused"; return 1; }
  test ! -e $I8/run/collapse_$T128R.json || { st "refused: $I8/run/collapse_$T128R.json exists (resume it with its ladder resume instead)"; return 1; }; }
iso128_nk_r() {
  iso128_pre || return 1
  mkdir -p $I8/run $I8/states $I8/logs; cd $I8/run
  run $E36R $I8/logs/$T128R.log ISO_128_s4_nkr "" $(iso128_args); }
iso128_nk_r_dry() {   # all checks, then the engine's own argparse and checks with run() and the GPU backend stubbed: starts nothing, writes nothing but the STATUS line
  STF=${STF_DRY:-/dev/null}
  iso128_pre || return 1
  cd /tmp && $PY -I -c "
import sys; sys.path.insert(0, sys.argv[1]); sys.argv = [sys.argv[2]] + sys.argv[3:]
import engine_collapse_v36r as E
E.run = lambda a: print('DRY OK: engine args parsed and checked; would run with', sorted((k, v) for k, v in vars(a).items() if v not in (None, False)))
E.C5.backend = lambda g: print('DRY: backend(gpu %s) stubbed' % g)
E.main()" $C $E36R $(iso128_args); }


# ---- pod6: ISO_128_nk_route_v2 (secs. 3.3, 3.4) the composite v36nq: Q0, its bitwise test from 11.5, and the member's run from T = 0, busi f52b1152 (v3e addition) ----
# engine_collapse_v36nq.py md5 8549c542 (build_engine_collapse_v36n.py --host engine_collapse_v36q.py, v36q 759c526c; built on the box 20 Oct). Both lines are
# section 18's staged line with --nk-host-release on and --end-above 0.001 (fixed here; END is not read) and PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True.
# iso128_nk_q0: from the 11.5 slice (b9d4b76e), --T-over-rs 11.9375 (inclusive: the loop runs to step nsteps 191 and breaks after its record), no
#   --slices-c128 (the engine's default is the empty list: no slice is written). Backstop: the mode stops its own engine by PID if a T/rs 12.000 line, or a
#   round line after the 11.938 record line, appears in the log. The v34 resume's args check compares m, g, rs, nsites, ell, kappa_sea, tau, reference,
#   site_fields, c_LZ, mix, massless_ball and fields_v17 only (resume_slice_v34), so the end clock and the new flags pass with the shared snapshot $I8/snap;
#   SNAPQ0=1 makes and uses an own copy $I8/snap_q0 (copied from $I8/snap, args.T_over_rs set to 11.9375, a _busi_note) should the resume ever refuse.
# iso128_nk_q: from the plain's T = 0 slice (782db60d; the resume keeps 1 row), the same snapshot, 21 slices (3, 5, 11.5, 11.9375, then section 18's 17).
# Refusals (both): engine md5, slice missing or md5 (SLICE_MD5=0 skips the 10.7 GB md5), any snapshot file missing, snapshot args.mix not newton, own json present.
E36NQ=$C/engine_collapse_v36nq.py; M36NQ=8549c5429e0fcd2608c92434a3c26967
S128Z=$I8/inputs/orbitals_${T128}_T0000.00.npz; M128Z=782db60df2d7577f3fb93618fb2296b0
T128Q0=lad_ISO_128_s4_nkq0_m0p2_rs128_n4096_R04_k10_tau8_ell4_prep4_A_R2_avg_newton_v36nq
T128Q=lad_ISO_128_s4_nkq_m0p2_rs128_n4096_R04_k10_tau8_ell4_prep4_A_R2_avg_newton_v36nq
NKQ="--mix newton ${NK% --end-above 0.001} --nk-plain-fallback on --nk-host-release on --end-above 0.001"
SLQ=3.0000000000,5.0000000000,11.5000000000,11.9375000000,$SL128
SNQ0=$I8/snap; [ "${SNAPQ0:-0}" = 1 ] && SNQ0=$I8/snap_q0
iso128_q0_args() { echo ${B128/--T-over-rs 40/--T-over-rs 11.9375} $NKQ --resume-slice $S128 --resume-run $SNQ0/$T128 --slice-dir $I8/states --out $I8/run --tag $T128Q0; }
iso128_q_args() { echo $B128 $NKQ --resume-slice $S128Z --resume-run $I8/snap/$T128 --slices-c128 $SLQ --slice-dir $I8/states --out $I8/run --tag $T128Q; }
nkq_pre() {   # $1 slice, $2 its md5, $3 snapshot dir, $4 own tag: the checks every v3e mode shares
  md5ok $E36NQ $M36NQ
  [ -s $1 ] || { st "no slice ($1): refused"; return 1; }
  [ "${SLICE_MD5:-1}" = 0 ] || md5ok $1 $2
  for f in .json _fields.npz _handoff.npz _spectra.npz; do [ -s $3/collapse_$T128$f ] || { st "no snapshot file $3/collapse_$T128$f: refused"; return 1; }; done
  grep -q '"mix": "newton"' $3/collapse_$T128.json || { st "snapshot json args.mix is not newton: refused"; return 1; }
  test ! -e $I8/run/collapse_$4.json || { st "refused: $I8/run/collapse_$4.json exists (resume it with its ladder resume instead)"; return 1; }; }
snap_q0() {   # SNAPQ0=1 only: the copy of $I8/snap with args.T_over_rs 11.9375 (the copy only; $I8/snap untouched)
  [ -s $I8/snap_q0/collapse_$T128.json ] && return 0
  mkdir -p $I8/snap_q0; cp $I8/snap/collapse_$T128.json $I8/snap/collapse_${T128}_fields.npz $I8/snap/collapse_${T128}_handoff.npz $I8/snap/collapse_${T128}_spectra.npz $I8/snap_q0/ || return 1
  $PY -c "
import json,sys
p=sys.argv[1]; d=json.load(open(p)); d['args']['T_over_rs']=11.9375
d['_busi_note']=(d.get('_busi_note','')+'; ' if d.get('_busi_note') else '')+'busi ready_jobs_v3e: snap_q0 copy of iso128_nk/snap; args.T_over_rs set to 11.9375 for the v34 resume check (iso128_nk/snap untouched)'
json.dump(d,open(p,'w'))" $I8/snap_q0/collapse_$T128.json; }
q0_backstop() {   # $1 the engine pid this mode started, $2 its log: stop it if the 12.0 record starts
  local P=$1 L=$2
  while kill -0 $P 2>/dev/null; do
    if grep -q 'T/rs 12\.000' $L 2>/dev/null || awk '/^T\/rs 11\.938 E\/E0/{f=1;next} f && /^  v36[a-z]* (round|direct|fallback)/{x=1} END{exit !x}' $L 2>/dev/null; then
      st "backstop: the 12.0 record started in $L: engine pid $P stopped by the mode (TERM, then KILL after 30 s)"
      kill -TERM $P 2>/dev/null; sleep 30; kill -0 $P 2>/dev/null && kill -KILL $P 2>/dev/null; return 0; fi
    sleep 20; done; }
iso128_nk_q0() {
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  if [ "${SNAPQ0:-0}" = 1 ]; then snap_q0 || { st "snap_q0 copy failed: refused"; return 1; }; fi
  nkq_pre $S128 b9d4b76e43d27c1ee96ab3df9c07264f $SNQ0 $T128Q0 || return 1
  mkdir -p $I8/run $I8/states $I8/logs $R/pids; cd $I8/run
  local L=$I8/logs/$T128Q0.log
  st "start: $E36NQ $(iso128_q0_args) (log $L; PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF)"
  nice -n 10 $PY $E36NQ $(iso128_q0_args) > $L 2>&1 < /dev/null &
  local P=$!; echo $P > $R/pids/ISO_128_s4_nkq0.pid
  q0_backstop $P $L & local BS=$!
  st "engine pid $P; backstop pid $BS (stops it at a 12.0 line)"
  wait $P; local rc=$?; echo $rc > $R/pids/ISO_128_s4_nkq0.pid.rc; kill $BS 2>/dev/null
  st "end rc $rc; records $(grep -c '^T/rs [0-9.]* E/E0' $L); last $(grep -o '^T/rs [0-9.]*' $L | tail -1); unconverged $(grep -c UNCONVERGED $L)"
  return $rc; }
iso128_nk_q() {
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  nkq_pre $S128Z $M128Z $I8/snap $T128Q || return 1
  mkdir -p $I8/run $I8/states $I8/logs; cd $I8/run
  run $E36NQ $I8/logs/$T128Q.log ISO_128_s4_nkq "" $(iso128_q_args); }
nkq_dry() {   # $1 args function: the engine's own argparse and checks with run() and the GPU backend stubbed: starts nothing, writes nothing but the STATUS line
  cd /tmp && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY -I -c "
import sys; sys.path.insert(0, sys.argv[1]); sys.argv = [sys.argv[2]] + sys.argv[3:]
import engine_collapse_v36nq as E
E.run = lambda a: print('DRY OK: engine args parsed and checked; would run with', sorted((k, v) for k, v in vars(a).items() if v not in (None, False)))
E.C5.backend = lambda g: print('DRY: backend(gpu %s) stubbed' % g)
E.main()" $C $E36NQ $($1); }
iso128_nk_q0_dry() {
  STF=${STF_DRY:-/dev/null}
  [ "${SNAPQ0:-0}" = 1 ] && [ ! -s $I8/snap_q0/collapse_$T128.json ] && echo "DRY: SNAPQ0=1 and snap_q0 absent (the run mode makes it)"
  nkq_pre $S128 b9d4b76e43d27c1ee96ab3df9c07264f $SNQ0 $T128Q0 || return 1
  nkq_dry iso128_q0_args; }
iso128_nk_q_dry() {
  STF=${STF_DRY:-/dev/null}
  nkq_pre $S128Z $M128Z $I8/snap $T128Q || return 1
  nkq_dry iso128_q_args; }

case $MODE in
  r2i) r2i;; r2ii_plain) r2ii plain;; r2ii_anderson) r2ii anderson;; r2ii_damped) r2ii damped;; r2iii) r2iii;;
  r2all) r2i && { r2ii plain; r2ii anderson; r2ii damped; }; r2iii;;
  isonewton) isonewton;;
  m3d) m3d;;
  m3c_a) m3c_a;; m3c_b) m3c_b;; m3c_c) m3c_c;; m3c_all) m3c_a && m3c_b && m3c_c;; m3c_nk) m3c_nk;; m3c_nk_r) m3c_nk_r;;
  iso128_nk_r) iso128_nk_r;; iso128_nk_r_dry) iso128_nk_r_dry;;
  iso128_nk_q0) iso128_nk_q0;; iso128_nk_q0_dry) iso128_nk_q0_dry;; iso128_nk_q) iso128_nk_q;; iso128_nk_q_dry) iso128_nk_q_dry;;
  *) echo "unknown mode $MODE"; exit 2;;
esac
