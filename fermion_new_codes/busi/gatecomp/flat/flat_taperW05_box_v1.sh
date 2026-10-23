#!/bin/bash
# flat_taperW05_box_v1.sh (busi, GATECOMP flat; ReviewA_edge_v1: TAPER, W = 0.5 r_s; inbox #418): box CPU, nice 10, before any card.
#  EV_taperW05: the r_s/a 32 flat fixture (flat_fixture_v1's FXF line, K 10, ell 2, prep-ell 4) with --prep-edge taper --prep-taper 0.5,
#               evolved T 0 to 0.125 (3 records, pair workers 0), 8 threads;
#  P128_W05:    the engine's own projected preparation of W = 0.5 at r_s/a 128 (2048 sites, K 15; memcal Mt15's line with W 0.5) by
#               prep_dry_v1.py prep (CPU; stops before the records): discarded norm, stationarity, f_kin on the orbitals and the box key, the
#               flatbox line (column counts, edge, interior). 8 threads.
set -u
F=${QI2_ROOT}/flat; C=$F/code16d_flat; X=engine_collapse_v36nq_rd13mfb.py; PY=${QI2_ROOT}/python; K=${QI2_ROOT}/kits
D=$F/taperW05; [ -e $D ] && { echo "refusing: $D exists"; exit 1; }; mkdir -p $D
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMBA_NUM_THREADS=8 NUMPY_MADVISE_HUGEPAGE=0 CUDA_VISIBLE_DEVICES=
sed 's/--prep-taper 1.0/--prep-taper 0.5/' $F/fixture/FXF_taper.args > $D/FXF_taperW05.args
o=$D/EV_taperW05; mkdir -p $o/states
A=$(sed -e 's/--gpu 0 //' -e 's/--pair-workers 2/--pair-workers 0/' -e 's/--T-over-rs 14/--T-over-rs 0.125 --slices-c128 0.125/' -e "s#--out X --slice-dir X --tag fxf#--out $o --slice-dir $o/states --tag ev_taperW05#" $D/FXF_taperW05.args)
echo "$A" > $o/cmd_args.txt
( t0=$(date +%s); nice -n 10 $PY -u $C/$X $A > $o/log.txt 2>&1; echo "EV_taperW05 rc $? ($(( $(date +%s) - t0 )) s)" >> $D/done.txt ) &
sed -e 's/--prep-taper 1.0/--prep-taper 0.5/' -e 's/--tag mc_Mt15/--tag p128_w05/' $F/memcal/Mt15/args.txt > $D/P128_W05.args
o2=$D/P128_W05; mkdir -p $o2
( t0=$(date +%s); nice -n 10 $PY $K/code/prep_dry_v1.py prep $C $X $D/P128_W05.args $o2 > $o2/prep.log 2>&1; echo "P128_W05 rc $? ($(( $(date +%s) - t0 )) s)" >> $D/done.txt ) &
wait; md5sum $C/$X $D/*.args > $D/md5_inputs.txt; echo DONE >> $D/done.txt
