#!/bin/bash
# G0'' of engine_twofield_v9c5 (PERF SYM-C1/SYM-C2; reviewer A symmetry audit secs. 1(b), 3) on the box CPU, nice 10, 8 threads per run, no card:
# the build over v9c4 (md5 152ec1c6) with the reverse byte for byte; v9c_selftest v1z's layer, import and fixture checks on v9c5 (the
# build-identity line "s build: equals the given engine" is expected to FAIL and is excluded, as for v9c2c/v9c3c/v9c4); the v9b self-test's
# fixture with v9c5 in v9b's place; the stand-in fixture (1024 sites, 4 records; run_g0pp9c4.sh's three forms: sector lagged lz with a slice at
# 0.0625, its resume to 0.1875, record implicit with a Simpson step). Arms: REF v9c4 at the defaults, OFF v9c5 at the defaults, C1
# --perf-sym-light-sector, C2 --perf-sym-ladder-sector, C12 both with H3's A1 band gather and B1 8 workers, ALL both with every H3 and v9c2
# option. Compares: OFF against REF bitwise (cmp_v9c3_runs --args-equal; the resumes with label/engine_v34 stripped; the slice md5); every
# SYM arm against OFF in the tolerance class with exact labels (cmp_symc_v1.py: every bool/int/string leaf and bool/int npz row equal, E and
# E/E0 to 1e-12 relative, the spectra npz eigenvalue rows to 1e-13 relative to their max |e|), and its json args against OFF's.
# Usage. run_g0pp9c5.sh   (outputs ${QI2_ROOT}/g0pp9c5)
ST=${QI2_ROOT}/v9c5
cd $ST || exit 2
export CUDA_VISIBLE_DEVICES="" BUSI_ENGINE_DIR=${QI2_ROOT}/compC_engine_v36 BUSI_PAGE_DIR=${QI2_ROOT}/page
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1
PY=${QI2_ROOT}/python
P="nice -n 10 $PY"
m() { md5sum < $1 | cut -c1-32; }
ALLV9C2="--perf-eig-cache-gib 4 --perf-dens-memo on --perf-uhx-cache-gib 2 --perf-writer on"
W=${QI2_ROOT}/g0pp9c5; mkdir -p $W
echo "# G0'' v9c5 on the box CPU (stage $ST) $(date -u +%FT%TZ)"
echo "== build $(date -u +%T)"
$P build_engine_twofield_v9c5.py engine_twofield_v9c4.py --check; echo "rc $?"
rm -f $W/v9c4_restored.py
$P build_engine_twofield_v9c5.py engine_twofield_v9c5.py $W/v9c4_restored.py --reverse --against engine_twofield_v9c4.py; echo "rc $?"
rm -f $W/v9c4_restored.py
$P -m py_compile engine_twofield_v9c5.py && echo "py_compile ok"
echo "# md5 engine_twofield_v9c4.py $(m engine_twofield_v9c4.py); engine_twofield_v9c5.py $(m engine_twofield_v9c5.py); build_engine_twofield_v9c5.py $(m build_engine_twofield_v9c5.py); cmp_symc_v1.py $(m cmp_symc_v1.py); unit_symc_v9c5.py $(m unit_symc_v9c5.py); run_g0pp9c5.sh $(m run_g0pp9c5.sh); lines marked [v9c5 perf] $(grep -c 'v9c5 perf\]' engine_twofield_v9c5.py)"
echo "== G0'' selftest on v9c5 (CPU): the layer, the import, the fixture $(date -u +%T)"
$P engine_twofield_v9c_selftest_v1z.py --v9b engine_twofield_v9b.py --v9c engine_twofield_v9c5.py --fixture --skip-suite > $W/selftest_v1z.log 2>&1; echo "rc $?"
grep -E "^(FAIL|engine_twofield_v9c_selftest)" $W/selftest_v1z.log
echo "  PASS lines $(grep -c '^PASS' $W/selftest_v1z.log), FAIL lines $(grep -c '^FAIL' $W/selftest_v1z.log), FAIL lines other than the build identity $(grep '^FAIL' $W/selftest_v1z.log | grep -vc '^FAIL s build: equals the given')"
echo "== the v9b self-test's fixture with v9c5 in v9b's place $(date -u +%T)"
SU=$W/suite_code; rm -rf $SU; mkdir -p $SU
cp engine_twofield_v9c5.py $SU/engine_twofield_v9b.py; cp engine_twofield_v9b_selftest.py engine_twofield_v8_selftest.py engine_twofield_v8.py assign_primitive_v1d.py assign_primitive_v1e.py assign_primitive_v1f.py $SU/
( cd $SU && $P engine_twofield_v9b_selftest.py fixture 2>&1 | grep -E "^(FAIL|SKIP|selftest)" ); echo "rc ${PIPESTATUS[0]}"
rm -rf $SU
E2E="--m 0.2 --g 0.25 --rs 32.0 --R0-over-rs 4 --reading A --nsites 1024 --prep-ell 4 --ell 4 --kappa-sea 10 --tau 2.0 --T-over-rs 0.1875 --reference R2 --half-offset 0 --collar-policy flag --support-rel 1e-14 --site-fields averaged --stop-E 1e-7 --stop-fields 1e-6 --maxit 20 --mix newton --nk-maxiter 30 --nk-krylov 40 --nk-restarts 0 --nk-line-search halving --nk-halvings 6 --source-continuation two-stage --end-above 1e-3 --nk-jvp exact --delta-rtol-every 10 --commutator-every 1 --streams 0 --c-LZ 1.0 --window-max 4 --light-kappas 1,2 --light-guard-kappas 3 --light-coupling on --light-transport banded --light-d-min 1e-7 --s11-collars 8,16,24 --s11-stop 1 --field-kernel-cache --v36-fixes --slices-c64 '' --slices-light-c128 ''"
run() { local log=$1; local eng=$2; shift 2; local t0=$(date +%s); eval $P $eng $E2E "$@" > $log 2>&1; local rc=$?; echo "rc $rc in $(( $(date +%s) - t0 )) s ($log)"; return $rc; }
SLZ="--reference-solve lagged --cut-transport sector --anchor-passage lz --m1-simpson-steps ''"
RI="--cut-transport record --m1-simpson-steps 1"
arm_run() {
  local arm=$1 eng=engine_twofield_v9c5.py opt
  case $arm in REF) eng=engine_twofield_v9c4.py; opt="";; OFF) opt="";; C1) opt="--perf-sym-light-sector";; C2) opt="--perf-sym-ladder-sector";;
    C12) opt="--perf-sym-light-sector --perf-sym-ladder-sector --perf-h3-band-gather --perf-h3-anchor-workers 8";;
    ALL) opt="--perf-sym-light-sector --perf-sym-ladder-sector --perf-h3-band-gather --perf-h3-anchor-workers 8 --perf-h3-weight-gather --perf-solve-memo on $ALLV9C2";; esac
  local d=$W/$arm; rm -rf $d; mkdir -p $d/sLz/states $d/res/states $d/ri
  echo "== $arm ($eng $opt): sector lagged lz with a slice at 0.0625 $(date -u +%T)"
  run $d/sLz/run.log $eng $SLZ --slices-c128 0.0625 --slice-dir $d/sLz/states --out $d/sLz --tag e2e_v9cL $opt
  local SL=$(ls $d/sLz/states/orbitals_e2e_v9cL_T*.npz | head -1)
  echo "== $arm: its resume from $(basename $SL) to 0.1875 $(date -u +%T)"
  run $d/res/run.log $eng $SLZ --slices-c128 "''" --slice-dir $d/res/states --out $d/res --tag e2e_v9cL --resume-slice $SL --resume-run $d/sLz/e2e_v9cL --segment-end-T-over-rs 0.1875 $opt
  echo "== $arm: record implicit with a Simpson step at step 1 $(date -u +%T)"
  run $d/ri/run.log $eng $RI --slices-c128 "''" --out $d/ri --tag e2e_v9ri $opt
}
for arm in REF OFF C1 C2; do arm_run $arm > $W/arm_$arm.log 2>&1 & done
wait
for arm in C12 ALL; do arm_run $arm > $W/arm_$arm.log 2>&1 & done
wait
for arm in REF OFF C1 C2 C12 ALL; do cat $W/arm_$arm.log; done
echo "== compares $(date -u +%T)"
echo "-- OFF against REF, bitwise"
for form in sLz/e2e_v9cL ri/e2e_v9ri; do $PY cmp_v9c3_runs.py $W/REF/$form $W/OFF/$form --args-equal; done
$PY - $W/REF/res/e2e_v9cL $W/OFF/res/e2e_v9cL "OFF against REF" <<'EOF'
import sys
sys.path.insert(0, "${QI2_ROOT}/v9c5")
import engine_twofield_v8_selftest as S8
import engine_twofield_v9b_selftest as S9
common, diffs = S9.compare_runs_stripped(S8, sys.argv[1], sys.argv[2], {"engine_v34", "label"})
print("%s %s res, label and engine_v34 stripped: %d common records, %d differences%s" % ("PASS" if (common and not diffs) else "FAIL", sys.argv[3], len(common), len(diffs), (" %s" % diffs[:4]) if diffs else ""))
EOF
SO=$(ls $W/OFF/sLz/states/*.npz | head -1); MO=$(m $SO); S=$W/REF/sLz/states/$(basename $SO)
if [ -f $S ] && [ "$(m $S)" = "$MO" ]; then echo "PASS slice REF $(basename $SO) md5 $MO equals OFF's"; else echo "FAIL slice REF $(basename $SO) against OFF's $MO"; fi
echo "-- every SYM arm against OFF: the tolerance class with exact labels (and the json args: the options never reach the namespace)"
for arm in C1 C2 C12 ALL; do
  for form in sLz/e2e_v9cL ri/e2e_v9ri; do $PY cmp_symc_v1.py $W/OFF/$form $W/$arm/$form --top 12; done
  $PY cmp_symc_v1.py $W/OFF/res/e2e_v9cL $W/$arm/res/e2e_v9cL --strip label,engine_v34 --top 12
  for form in sLz/e2e_v9cL ri/e2e_v9ri; do echo "  $arm $form json args against OFF's: $($PY cmp_v9c3_runs.py $W/OFF/$form $W/$arm/$form --args-equal | grep -o 'args differ in [^;]*')"; done
done
echo "-- the v9c5 perf lines (counters) per arm and form"
for arm in C1 C2 C12 ALL; do for f in sLz res ri; do echo "  $arm $f: $(grep -h 'v9c5 perf' $W/$arm/$f/run.log | tr '\n' ' ' | cut -c1-500)"; done; done
echo "-- record seconds per arm and form (indicative only: 4 arms side by side)"
for arm in REF OFF C1 C2 C12 ALL; do for form in sLz/collapse_e2e_v9cL res/collapse_e2e_v9cL ri/collapse_e2e_v9ri; do
$PY - $W/$arm/$form.json $arm $form <<'EOF'
import json, sys
J = json.load(open(sys.argv[1])); R = [r for r in J["records"] if r.get("E") is not None]
s = [r.get("seconds", 0) for r in R]
print("%s %s: record seconds %s sum %.1f" % (sys.argv[2], sys.argv[3], [round(x, 1) for x in s], sum(s)))
EOF
done; done
echo "-- prune (the box disk): slices and npz removed, logs and jsons kept"
find $W -name "*.npz" -delete; find $W -type d -name states -empty -delete
du -sh $W
echo "== done $(date -u +%T)"
