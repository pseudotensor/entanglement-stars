#!/bin/bash
# flat_dry_v1.sh (busi, GATECOMP flat build; ReviewA_flatq_v1 sec. 1: "the ten staged lines re-parse, fixtures PREP OK unchanged"):
# (1) parse: the kits' ten staged lines (lines/ banded and lines_dense/) by prep_dry_v1.py parse through the parent (248f25f3) and the
#     child (02a1729a); the child's vars(a) must equal the parent's plus the 4 new keys at their defaults (read by flat_dry_read_v1.py);
# (2) check: the ten dense lines through the child (CHECK OK);
# (3) prep: the kits' fixtures (dry/fx/*.args: FXP_cv, FXP_w fresh; FX_tau, FX_plain, FX_point, FX_tight restarts; FX_tau_noview must be
#     REFUSED) through the child, their printed initial-data / restart lines against the parent's stored prep.log (dry/fx/NAME/prep.log).
# Box CPU (prep_dry_v1 empties CUDA_VISIBLE_DEVICES), nice 10, 8 threads, run one at a time.
set -u
K=${QI2_ROOT}/kits; F=${QI2_ROOT}/flat; C=$F/code16d_flat; PY=${QI2_ROOT}/python
P=engine_collapse_v36nq_rd13mf.py; X=engine_collapse_v36nq_rd13mfb.py; D=$F/dry; rm -rf $D; mkdir -p $D
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 NUMPY_MADVISE_HUGEPAGE=0
for L in lines lines_dense; do
  for af in $K/$L/*.args; do n=$(basename $af .args)
    for e in P X; do eng=$P; [ $e = X ] && eng=$X; o=$D/parse/$L/$n/$e; mkdir -p $o
      $PY $K/code/prep_dry_v1.py parse $C $eng $af $o > $o/log 2>&1; echo "parse $L $n $e rc $? $(grep -c PARSED $o/log)"
    done
  done
done
for af in $K/lines_dense/*.args; do n=$(basename $af .args); o=$D/check/$n; mkdir -p $o
  nice -n 10 $PY $K/code/prep_dry_v1.py check $C $X $af $o > $o/log 2>&1; echo "check $n rc $? $(grep -E 'CHECK OK|rror' $o/log | tail -1 | cut -c1-160)"
done
for n in FXP_cv FXP_w FX_tau FX_plain FX_point FX_tight FX_tau_noview; do o=$D/prep/$n; mkdir -p $o; t0=$(date +%s)
  nice -n 10 $PY $K/code/prep_dry_v1.py prep $C $X $K/dry/fx/$n.args $o > $o/prep.log 2>&1; rc=$?
  echo "prep $n rc $rc ($(( $(date +%s) - t0 )) s): $(grep -E 'PREP OK|version 34|SystemExit|Error|error' $o/prep.log | tail -1 | cut -c1-200)"
done
echo DONE
