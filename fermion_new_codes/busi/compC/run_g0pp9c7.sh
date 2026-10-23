#!/bin/bash
# G0'' of engine_twofield_v9c7 (PERF RD-1c) on the box CPU, nice 10, run_g0pp9c6.sh's steps with v9c7 in v9c6's place and v9c6 as REF:
# the build over v9c6 (md5 89b852eb) with the reverse byte for byte; v9c_selftest v1z's layer, import and fixture checks on v9c7 (the
# build-identity line "s build: equals the given engine" is expected to FAIL and is excluded, as for v9c2c..v9c6); the v9b self-test's fixture
# with v9c7 in v9b's place; the stand-in fixture (1024 sites, 4 records; the three forms: sector lagged lz with a slice at 0.0625, its resume to
# 0.1875, record implicit with a Simpson step). Arms: REF v9c6 at the defaults, OFF v9c7 at the defaults (bitwise REF). --heavy-eigh banded
# needs a GPU, so no option arm here (the GPU smoke is rd1c_fixture_gpu_v1.sh). Two arms at a time, 4 threads each; then a check that no
# "v9c7 perf" line is in the defaults' logs after the command line.
# Usage. run_g0pp9c7.sh   (outputs ${QI2_ROOT}/g0pp9c7)
ST=${QI2_ROOT}/v9c7
cd $ST || exit 2
export CUDA_VISIBLE_DEVICES="" BUSI_ENGINE_DIR=${QI2_ROOT}/compC_engine_v36 BUSI_PAGE_DIR=${QI2_ROOT}/page
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMBA_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
PY=${QI2_ROOT}/python
P="nice -n 10 $PY"
m() { md5sum < $1 | cut -c1-32; }
W=${QI2_ROOT}/g0pp9c7; mkdir -p $W
echo "# G0'' v9c7 on the box CPU (stage $ST) $(date -u +%FT%TZ)"
echo "== build $(date -u +%T)"
$P build_engine_twofield_v9c7.py engine_twofield_v9c6.py --check; echo "rc $?"
rm -f $W/v9c6_restored.py
$P build_engine_twofield_v9c7.py engine_twofield_v9c7.py $W/v9c6_restored.py --reverse --against engine_twofield_v9c6.py; echo "rc $?"
rm -f $W/v9c6_restored.py
$P -m py_compile engine_twofield_v9c7.py && echo "py_compile ok"
echo "# md5 engine_twofield_v9c6.py $(m engine_twofield_v9c6.py); engine_twofield_v9c7.py $(m engine_twofield_v9c7.py); build_engine_twofield_v9c7.py $(m build_engine_twofield_v9c7.py); run_g0pp9c7.sh $(m run_g0pp9c7.sh); lines marked [v9c7 perf] $(grep -c 'v9c7 perf\]' engine_twofield_v9c7.py)"
echo "== G0'' selftest on v9c7 (CPU): the layer, the import, the fixture $(date -u +%T)"
$P engine_twofield_v9c_selftest_v1z.py --v9b engine_twofield_v9b.py --v9c engine_twofield_v9c7.py --fixture --skip-suite > $W/selftest_v1z.log 2>&1; echo "rc $?"
grep -E "^(FAIL|engine_twofield_v9c_selftest)" $W/selftest_v1z.log
echo "  PASS lines $(grep -c '^PASS' $W/selftest_v1z.log), FAIL lines $(grep -c '^FAIL' $W/selftest_v1z.log), FAIL lines other than the build identity $(grep '^FAIL' $W/selftest_v1z.log | grep -vc '^FAIL s build: equals the given')"
echo "== the v9b self-test's fixture with v9c7 in v9b's place $(date -u +%T)"
SU=$W/suite_code; rm -rf $SU; mkdir -p $SU
cp engine_twofield_v9c7.py $SU/engine_twofield_v9b.py; cp engine_twofield_v9b_selftest.py engine_twofield_v8_selftest.py engine_twofield_v8.py assign_primitive_v1d.py assign_primitive_v1e.py assign_primitive_v1f.py $SU/
( cd $SU && $P engine_twofield_v9b_selftest.py fixture 2>&1 | grep -E "^(FAIL|SKIP|selftest)" ); echo "rc ${PIPESTATUS[0]}"
rm -rf $SU
E2E="--m 0.2 --g 0.25 --rs 32.0 --R0-over-rs 4 --reading A --nsites 1024 --prep-ell 4 --ell 4 --kappa-sea 10 --tau 2.0 --T-over-rs 0.1875 --reference R2 --half-offset 0 --collar-policy flag --support-rel 1e-14 --site-fields averaged --stop-E 1e-7 --stop-fields 1e-6 --maxit 20 --mix newton --nk-maxiter 30 --nk-krylov 40 --nk-restarts 0 --nk-line-search halving --nk-halvings 6 --source-continuation two-stage --end-above 1e-3 --nk-jvp exact --delta-rtol-every 10 --commutator-every 1 --streams 0 --c-LZ 1.0 --window-max 4 --light-kappas 1,2 --light-guard-kappas 3 --light-coupling on --light-transport banded --light-d-min 1e-7 --s11-collars 8,16,24 --s11-stop 1 --field-kernel-cache --v36-fixes --slices-c64 '' --slices-light-c128 ''"
run() { local log=$1; local eng=$2; shift 2; local t0=$(date +%s); eval $P $eng $E2E "$@" > $log 2>&1; local rc=$?; echo "rc $rc in $(( $(date +%s) - t0 )) s ($log)"; return $rc; }
SLZ="--reference-solve lagged --cut-transport sector --anchor-passage lz --m1-simpson-steps ''"
RI="--cut-transport record --m1-simpson-steps 1"
arm_run() {
  local arm=$1 eng=engine_twofield_v9c7.py opt="" forms="sLz res ri"
  case $arm in REF) eng=engine_twofield_v9c6.py;; OFF) eng=engine_twofield_v9c7.py;; esac
  local d=$W/$arm; rm -rf $d; mkdir -p $d/sLz/states $d/res/states $d/ri
  echo "== $arm ($eng $opt): sector lagged lz with a slice at 0.0625 $(date -u +%T)"
  run $d/sLz/run.log $eng $SLZ --slices-c128 0.0625 --slice-dir $d/sLz/states --out $d/sLz --tag e2e_v9cL $opt
  local SL=$(ls $d/sLz/states/orbitals_e2e_v9cL_T*.npz | head -1)
  echo "== $arm: its resume from $(basename $SL) to 0.1875 $(date -u +%T)"
  run $d/res/run.log $eng $SLZ --slices-c128 "''" --slice-dir $d/res/states --out $d/res --tag e2e_v9cL --resume-slice $SL --resume-run $d/sLz/e2e_v9cL --segment-end-T-over-rs 0.1875 $opt
  echo "== $arm: record implicit with a Simpson step at step 1 $(date -u +%T)"; run $d/ri/run.log $eng $RI --slices-c128 "''" --out $d/ri --tag e2e_v9ri $opt
}
for arm in REF OFF; do arm_run $arm > $W/arm_$arm.log 2>&1 & done
wait
for arm in REF OFF; do cat $W/arm_$arm.log; done
echo "== compares $(date -u +%T)"
echo "-- OFF against REF, bitwise"
for form in sLz/e2e_v9cL ri/e2e_v9ri; do $PY cmp_v9c3_runs.py $W/REF/$form $W/OFF/$form --args-equal; done
$PY - $W/REF/res/e2e_v9cL $W/OFF/res/e2e_v9cL "OFF against REF" <<'PYEOF'
import sys
sys.path.insert(0, "${QI2_ROOT}/v9c7")
import engine_twofield_v8_selftest as S8
import engine_twofield_v9b_selftest as S9
common, diffs = S9.compare_runs_stripped(S8, sys.argv[1], sys.argv[2], {"engine_v34", "label"})
print("%s %s res, label and engine_v34 stripped: %d common records, %d differences%s" % ("PASS" if (common and not diffs) else "FAIL", sys.argv[3], len(common), len(diffs), (" %s" % diffs[:4]) if diffs else ""))
PYEOF
SO=$(ls $W/OFF/sLz/states/*.npz | head -1); MO=$(m $SO); S=$W/REF/sLz/states/$(basename $SO)
if [ -f $S ] && [ "$(m $S)" = "$MO" ]; then echo "PASS slice REF $(basename $SO) md5 $MO equals OFF's"; else echo "FAIL slice REF $(basename $SO) against OFF's $MO"; fi
echo "-- no v9c7 perf line in the defaults' logs after the command line (line 1 of a log may name the engine file)"
for f in $W/OFF/sLz/run.log $W/OFF/res/run.log $W/OFF/ri/run.log; do echo "  $(basename $(dirname $f)): $(tail -n +2 $f | grep -c 'v9c7 perf') lines"; done
echo "-- prune (the box disk): slices and npz removed, logs and jsons kept"
find $W -name "*.npz" -delete; find $W -type d -name states -empty -delete
du -sh $W
echo "== done $(date -u +%T)"
