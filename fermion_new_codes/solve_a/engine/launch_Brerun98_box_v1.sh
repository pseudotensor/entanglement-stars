#!/bin/bash
# launch_Brerun98_box_v1.sh (solve builder 6, analyst; engine spec 47.5 (1), 47.6 (1)): B ended at T/r_s 100 with no T100 slice (the box's /
# below B's --min-free-gb 60). B's last stretch is rerun from B's latest complex128 slice (T/r_s 98, states/..._ext100_T0098.00.npz) with the
# gated runner run_collapse_v19_from_slice_v1.py and B's arguments unchanged (list_gpu2_ext19_v1.txt), --resume-from a copy of B's files on
# /data (md5-checked), output directory and slices on /data, --min-free-gb 12, slices every 2 r_s as B's list (98 is the start, 100 the end),
# on box GPU 2 (an A6000) under its own guard: GPU 2's free memory below 8192 MiB stops this rerun's python by its PID (nothing else).
# The recomputed records 98.125 to 100 are compared with B's stored ones by compare_member_floor_v2 (within the floor expected, masks equal).
# Usage (box): setsid nohup bash ${QI2_ROOT}/launch_Brerun98_box_v1.sh > /dev/null 2>&1 &
set -u
ENG=${QI2_ROOT}/engine
BD=${QI2_ROOT}/Brerun98
TAG0=c19_rs64_A_R2_tau8_ell4_avg_ext100
TAG=c19_rs64_A_R2_tau8_ell4_avg_ext100_r98
ST=$BD/status.txt
mkdir -p $BD/Bfiles $BD/out $BD/states
cd $ENG || exit 1
echo "Brerun98 start $(date -u '+%Y-%m-%d %H:%M:%S UTC') pid $$" >> $ST
for f in C3ext19/collapse_${TAG0}.json C3ext19/collapse_${TAG0}_fields.npz C3ext19/collapse_${TAG0}_handoff.npz C3ext19/collapse_${TAG0}_spectra.npz; do
  cp -p $f $BD/Bfiles/ || { echo "copy FAIL $f" >> $ST; exit 2; }
  a=$(md5sum < $f); b=$(md5sum < $BD/Bfiles/$(basename $f))
  [ "$a" = "$b" ] || { echo "md5 FAIL $f" >> $ST; exit 2; }
done
echo "B's files copied to $BD/Bfiles md5-equal $(date -u '+%H:%M:%S')" >> $ST
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
${QI2_ROOT}/python code/run_collapse_v19_from_slice_v1.py --m 0.2 --g 0.25 --rs 64 --R0-over-rs 4 --reading A --nsites 2048 --ell 4 \
  --kappa-sea 10 --tau 8 --T-over-rs 100 --reference R2 --half-offset 0 --collar-policy flag --support-rel 1e-14 --out $BD/out --site-fields averaged \
  --stop-E 1e-7 --stop-fields 1e-6 --maxit 20 --mix plain --delta-rtol-every 10 --commutator-every 1 --streams 0 --m1-simpson-steps auto --c-LZ 1.0 \
  --window-max 4 --resume-from $BD/Bfiles/$TAG0 --resume-T-over-rs 98 --slices-c128 100 --slice-dir $BD/states \
  --slice states/c19_rs64_A_R2_tau8_ell4_avg_ext100/orbitals_${TAG0}_T0098.00.npz --min-free-gb 12 --gpu 2 --tag $TAG > $BD/Brerun98.log 2>&1 &
PY=$!
echo "Brerun98 python pid $PY $(date -u '+%H:%M:%S')" >> $ST
while kill -0 $PY 2>/dev/null; do
  fr=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 2 | tr -d ' ')
  if [ -n "$fr" ] && [ "$fr" -lt 8192 ]; then
    echo "GUARD $(date -u '+%H:%M:%S') GPU 2 free ${fr} MiB < 8192: stopping pid $PY" >> $ST
    kill $PY; break
  fi
  sleep 20
done
wait $PY; rc=$?
echo "Brerun98 exit $rc $(date -u '+%Y-%m-%d %H:%M:%S UTC')" >> $ST
if [ -f $BD/out/collapse_${TAG}.json ]; then
  ${QI2_ROOT}/python code/compare_member_floor_v2.py --run $BD/out/collapse_${TAG}.json --ref C3ext19/collapse_${TAG0}.json \
    --label-run "B rerun from 98" --label-ref "B (C3ext19)" --md $BD/Brerun98_vs_B.md --out-json $BD/Brerun98_vs_B.json --criterion floor >> $ST 2>&1
  echo "compare exit $? $(date -u '+%H:%M:%S')" >> $ST
fi
