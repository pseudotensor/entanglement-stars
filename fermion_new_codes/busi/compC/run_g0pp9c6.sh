#!/bin/bash
# G0'' of engine_twofield_v9c6 (PERF RD-4) on the box CPU, nice 10, run_g0pp9c5.sh's steps with v9c6 in v9c5's place and v9c5 as REF:
# the build over v9c5 (md5 7ca18d71) with the reverse byte for byte; v9c_selftest v1z's layer, import and fixture checks on v9c6 (the
# build-identity line "s build: equals the given engine" is expected to FAIL and is excluded, as for v9c2c..v9c5); the v9b self-test's fixture
# with v9c6 in v9b's place; the stand-in fixture (1024 sites, 4 records; the three forms: sector lagged lz with a slice at 0.0625, its resume to
# 0.1875, record implicit with a Simpson step). Arms: REF v9c5 at the defaults, OFF v9c6 at the defaults (bitwise REF), and on the sector forms
# only (--light-walk tabulated needs --cut-transport sector): C2 v9c5 --perf-sym-ladder-sector, T2 v9c6 C2 + --light-walk tabulated (the
# table on the CPU), T0 v9c6 --light-walk tabulated (the Cholesky fallback on the full K). Two arms at a time, 4 threads each.
# Usage. run_g0pp9c6.sh   (outputs ${QI2_ROOT}/g0pp9c6)
ST=${QI2_ROOT}/v9c6
cd $ST || exit 2
export CUDA_VISIBLE_DEVICES="" BUSI_ENGINE_DIR=${QI2_ROOT}/compC_engine_v36 BUSI_PAGE_DIR=${QI2_ROOT}/page
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMBA_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
PY=${QI2_ROOT}/python
P="nice -n 10 $PY"
m() { md5sum < $1 | cut -c1-32; }
W=${QI2_ROOT}/g0pp9c6; mkdir -p $W
echo "# G0'' v9c6 on the box CPU (stage $ST) $(date -u +%FT%TZ)"
echo "== build $(date -u +%T)"
$P build_engine_twofield_v9c6.py engine_twofield_v9c5.py --check; echo "rc $?"
rm -f $W/v9c5_restored.py
$P build_engine_twofield_v9c6.py engine_twofield_v9c6.py $W/v9c5_restored.py --reverse --against engine_twofield_v9c5.py; echo "rc $?"
rm -f $W/v9c5_restored.py
$P -m py_compile engine_twofield_v9c6.py && echo "py_compile ok"
echo "# md5 engine_twofield_v9c5.py $(m engine_twofield_v9c5.py); engine_twofield_v9c6.py $(m engine_twofield_v9c6.py); build_engine_twofield_v9c6.py $(m build_engine_twofield_v9c6.py); run_g0pp9c6.sh $(m run_g0pp9c6.sh); lines marked [v9c6 perf] $(grep -c 'v9c6 perf\]' engine_twofield_v9c6.py)"
echo "== G0'' selftest on v9c6 (CPU): the layer, the import, the fixture $(date -u +%T)"
$P engine_twofield_v9c_selftest_v1z.py --v9b engine_twofield_v9b.py --v9c engine_twofield_v9c6.py --fixture --skip-suite > $W/selftest_v1z.log 2>&1; echo "rc $?"
grep -E "^(FAIL|engine_twofield_v9c_selftest)" $W/selftest_v1z.log
echo "  PASS lines $(grep -c '^PASS' $W/selftest_v1z.log), FAIL lines $(grep -c '^FAIL' $W/selftest_v1z.log), FAIL lines other than the build identity $(grep '^FAIL' $W/selftest_v1z.log | grep -vc '^FAIL s build: equals the given')"
echo "== the v9b self-test's fixture with v9c6 in v9b's place $(date -u +%T)"
SU=$W/suite_code; rm -rf $SU; mkdir -p $SU
cp engine_twofield_v9c6.py $SU/engine_twofield_v9b.py; cp engine_twofield_v9b_selftest.py engine_twofield_v8_selftest.py engine_twofield_v8.py assign_primitive_v1d.py assign_primitive_v1e.py assign_primitive_v1f.py $SU/
( cd $SU && $P engine_twofield_v9b_selftest.py fixture 2>&1 | grep -E "^(FAIL|SKIP|selftest)" ); echo "rc ${PIPESTATUS[0]}"
rm -rf $SU
E2E="--m 0.2 --g 0.25 --rs 32.0 --R0-over-rs 4 --reading A --nsites 1024 --prep-ell 4 --ell 4 --kappa-sea 10 --tau 2.0 --T-over-rs 0.1875 --reference R2 --half-offset 0 --collar-policy flag --support-rel 1e-14 --site-fields averaged --stop-E 1e-7 --stop-fields 1e-6 --maxit 20 --mix newton --nk-maxiter 30 --nk-krylov 40 --nk-restarts 0 --nk-line-search halving --nk-halvings 6 --source-continuation two-stage --end-above 1e-3 --nk-jvp exact --delta-rtol-every 10 --commutator-every 1 --streams 0 --c-LZ 1.0 --window-max 4 --light-kappas 1,2 --light-guard-kappas 3 --light-coupling on --light-transport banded --light-d-min 1e-7 --s11-collars 8,16,24 --s11-stop 1 --field-kernel-cache --v36-fixes --slices-c64 '' --slices-light-c128 ''"
run() { local log=$1; local eng=$2; shift 2; local t0=$(date +%s); eval $P $eng $E2E "$@" > $log 2>&1; local rc=$?; echo "rc $rc in $(( $(date +%s) - t0 )) s ($log)"; return $rc; }
SLZ="--reference-solve lagged --cut-transport sector --anchor-passage lz --m1-simpson-steps ''"
RI="--cut-transport record --m1-simpson-steps 1"
arm_run() {
  local arm=$1 eng=engine_twofield_v9c6.py opt forms="sLz res ri"
  case $arm in REF) eng=engine_twofield_v9c5.py; opt="";; OFF) opt="";; C2) eng=engine_twofield_v9c5.py; opt="--perf-sym-ladder-sector"; forms="sLz res";;
    T2) opt="--perf-sym-ladder-sector --light-walk tabulated"; forms="sLz res";; T0) opt="--light-walk tabulated"; forms="sLz res";; esac
  local d=$W/$arm; rm -rf $d; mkdir -p $d/sLz/states $d/res/states $d/ri
  echo "== $arm ($eng $opt): sector lagged lz with a slice at 0.0625 $(date -u +%T)"
  run $d/sLz/run.log $eng $SLZ --slices-c128 0.0625 --slice-dir $d/sLz/states --out $d/sLz --tag e2e_v9cL $opt
  local SL=$(ls $d/sLz/states/orbitals_e2e_v9cL_T*.npz | head -1)
  echo "== $arm: its resume from $(basename $SL) to 0.1875 $(date -u +%T)"
  run $d/res/run.log $eng $SLZ --slices-c128 "''" --slice-dir $d/res/states --out $d/res --tag e2e_v9cL --resume-slice $SL --resume-run $d/sLz/e2e_v9cL --segment-end-T-over-rs 0.1875 $opt
  if [[ $forms == *ri* ]]; then echo "== $arm: record implicit with a Simpson step at step 1 $(date -u +%T)"; run $d/ri/run.log $eng $RI --slices-c128 "''" --out $d/ri --tag e2e_v9ri $opt; fi
}
for pair in "REF OFF" "C2 T2" "T0"; do
  for arm in $pair; do arm_run $arm > $W/arm_$arm.log 2>&1 & done
  wait
done
for arm in REF OFF C2 T2 T0; do cat $W/arm_$arm.log; done
echo "== compares $(date -u +%T)"
echo "-- OFF against REF, bitwise"
for form in sLz/e2e_v9cL ri/e2e_v9ri; do $PY cmp_v9c3_runs.py $W/REF/$form $W/OFF/$form --args-equal; done
$PY - $W/REF/res/e2e_v9cL $W/OFF/res/e2e_v9cL "OFF against REF" <<'PYEOF'
import sys
sys.path.insert(0, "${QI2_ROOT}/v9c6")
import engine_twofield_v8_selftest as S8
import engine_twofield_v9b_selftest as S9
common, diffs = S9.compare_runs_stripped(S8, sys.argv[1], sys.argv[2], {"engine_v34", "label"})
print("%s %s res, label and engine_v34 stripped: %d common records, %d differences%s" % ("PASS" if (common and not diffs) else "FAIL", sys.argv[3], len(common), len(diffs), (" %s" % diffs[:4]) if diffs else ""))
PYEOF
SO=$(ls $W/OFF/sLz/states/*.npz | head -1); MO=$(m $SO); S=$W/REF/sLz/states/$(basename $SO)
if [ -f $S ] && [ "$(m $S)" = "$MO" ]; then echo "PASS slice REF $(basename $SO) md5 $MO equals OFF's"; else echo "FAIL slice REF $(basename $SO) against OFF's $MO"; fi
echo "-- the tabulated arms: T2 against C2 and T0 against OFF (cmp_symc_v1: labels exact, the Cholesky counters expected to differ; cmp_rd4_v1: the sec. 7 budgets)"
for pr in "C2 T2" "OFF T0"; do set -- $pr
  for form in sLz/e2e_v9cL; do $PY cmp_symc_v1.py $W/$1/$form $W/$2/$form --top 12; $PY cmp_rd4_v1.py $W/$1/$form $W/$2/$form; done
  $PY cmp_symc_v1.py $W/$1/res/e2e_v9cL $W/$2/res/e2e_v9cL --strip label,engine_v34 --top 12; $PY cmp_rd4_v1.py $W/$1/res/e2e_v9cL $W/$2/res/e2e_v9cL --strip label,engine_v34
  echo "  $2 sLz json args against $1's: $($PY cmp_v9c3_runs.py $W/$1/sLz/e2e_v9cL $W/$2/sLz/e2e_v9cL --args-equal | grep -o 'args differ in [^;]*')"
done
echo "-- the v9c6 perf lines (counters) per arm and form"
for arm in T2 T0; do for f in sLz res; do echo "  $arm $f: $(grep -h 'v9c6 perf' $W/$arm/$f/run.log | tr '\n' ' ' | cut -c1-700)"; done; done
echo "-- prune (the box disk): slices and npz removed, logs and jsons kept"
find $W -name "*.npz" -delete; find $W -type d -name states -empty -delete
du -sh $W
echo "== done $(date -u +%T)"
