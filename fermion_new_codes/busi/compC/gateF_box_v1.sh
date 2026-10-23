#!/bin/bash
# Computation C, gate F (RUN_ORDER_C_v1.md section 4; busi/notes/C_gateD_judgment_v1.md section 7 step 3) on the box CPU: the light channels
# kappa 1, 2 (sourcing) and 3 (guard) followed on the sigma/a 4 member's 641 stored records (C3lad32, 1024 sites) by engine_twofield_v3. The reading
# of interest is the record of the first cut crossing per channel (the certified horizon of the light transport on the real history), the
# fallback count and E_rad0/E. No light slices are saved (the box holds 99 GB, and the solve lane needs 90 GB free to restore B's slices for this
# lane's dust route); the light-follow json and npz carry the per-record arrays (COMMS rule 10). No GPU is free to this lane (pod 3 stalled).
R=${QI2_ROOT}/compC; C=${QI2_ROOT}/compC; E=${QI2_ROOT}/compC_engine_v36
export BUSI_ENGINE_DIR=$E BUSI_PAGE_DIR=$E PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
mkdir -p $R/follow_s4
echo "gateF_s4 launch $(date -u +%H:%M:%S) load $(cut -d' ' -f1 /proc/loadavg)" >> $R/status.txt
cd $C && nice -n 10 ${QI2_ROOT}/python engine_twofield_v3.py --m 0.2 --g 0.25 --rs 32 --nsites 1024 --tau 2 --half-offset 0 --ell 4 \
  --light-kappas 1,2 --light-guard-kappas 3 --light-transport banded --s11-collars 8,16,24 \
  --light-follow ${QI2_ROOT}/c31_rs32_A_R2_tau2_ell4_prep4_avg --slices-light-c128 "" \
  --out $R/follow_s4 --tag follow_s4 > $R/gateF_s4.log 2>&1
echo "gateF_s4 exit $? $(date -u +%H:%M:%S)" >> $R/status.txt
