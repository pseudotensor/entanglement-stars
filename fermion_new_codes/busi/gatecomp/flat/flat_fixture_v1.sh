#!/bin/bash
# flat_fixture_v1.sh (busi, GATECOMP flat build; ReviewA_flatq_v1 sec. 1: "flat fixture at r_s/a 32 PREP OK with its catalog"). Box CPU,
# nice 10, 8 threads. The kits' FXP_w line (the W256 shape at r_s/a 32, 512 sites = 16 r_s) with K = 10 (the box catalog's top 5 + 5 at
# N_B 320, flat_profile_v1), ell 2 (sigma/r_s 1/16), prep-ell 4, and --prep flatbox:
#   FXF_hard (reviewer A's text), FXF_taper (--prep-edge taper --prep-taper 1.0, reviewer B's), each prep_dry_v1.py prep (PREP OK);
#   option errors (check mode, each must exit non-zero with the 'flat prep:' message): taper without flatbox, flatbox with --dust-set
#   isotropic, taper without W, a W on the hard edge;
#   EV_hard: FXF_hard evolved T 0 to T/r_s 0.125 (3 records; pair workers 0, the end slice), for the T = 0 six-term (forward difference).
set -u
K=${QI2_ROOT}/kits; F=${QI2_ROOT}/flat; C=$F/code16d_flat; X=engine_collapse_v36nq_rd13mfb.py
PY=${QI2_ROOT}/python; D=$F/fixture; [ -e $D ] && { echo "refusing: $D exists"; exit 1; }; mkdir -p $D
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 NUMBA_NUM_THREADS=8 NUMPY_MADVISE_HUGEPAGE=0
B=$(sed -e 's/--kappa-sea 14/--kappa-sea 10/' -e 's/--ell 8 /--ell 2 /' -e 's/--tag fxp_w/--tag fxf/' $K/dry/fx/FXP_w.args)
echo "$B --prep flatbox" > $D/FXF_hard.args
echo "$B --prep flatbox --prep-edge taper --prep-taper 1.0" > $D/FXF_taper.args
echo "$B --prep-edge taper --prep-taper 1.0" > $D/ERR_taper_nobox.args
echo "$(echo "$B" | sed 's/--dust-set well/--dust-set isotropic/') --prep flatbox" > $D/ERR_box_iso.args
echo "$B --prep flatbox --prep-edge taper" > $D/ERR_taper_noW.args
echo "$B --prep flatbox --prep-taper 1.0" > $D/ERR_hard_W.args
md5sum $C/$X $D/*.args > $D/md5_inputs.txt
for n in FXF_hard FXF_taper; do o=$D/$n; mkdir -p $o; t0=$(date +%s)
  nice -n 10 $PY $K/code/prep_dry_v1.py prep $C $X $D/$n.args $o > $o/prep.log 2>&1; rc=$?
  echo "prep $n rc $rc ($(( $(date +%s) - t0 )) s): $(grep -E 'PREP OK|Error|error' $o/prep.log | tail -1 | cut -c1-200)"
done
for n in ERR_taper_nobox ERR_box_iso ERR_taper_noW ERR_hard_W; do o=$D/$n; mkdir -p $o
  $PY $K/code/prep_dry_v1.py check $C $X $D/$n.args $o > $o/log 2>&1; rc=$?
  echo "check $n rc $rc (non-zero expected): $(grep -E 'flat prep:|CHECK OK' $o/log | tail -1 | cut -c1-200)"
done
o=$D/EV_hard; mkdir -p $o/states
A=$(sed -e 's/--gpu 0 //' -e 's/--pair-workers 2/--pair-workers 0/' -e 's/--T-over-rs 14/--T-over-rs 0.125 --slices-c128 0.125/' -e "s#--out X --slice-dir X --tag fxf#--out $o --slice-dir $o/states --tag ev_hard#" $D/FXF_hard.args)
echo "$A" > $o/cmd_args.txt; t0=$(date +%s)
env CUDA_VISIBLE_DEVICES= nice -n 10 $PY -u $C/$X $A > $o/log.txt 2>&1; echo "EV_hard rc $? ($(( $(date +%s) - t0 )) s)"
echo DONE
