#!/bin/bash
# launch_single_object_B2_hist_dust_box_v1.sh (builder analyst, the HORIZON version 2 builder; engine spec 38.4 (1) (i) to (iv), 39.5 (1)):
# engine_single_object_B_v2.py on box GPU 0 (free until the certified S11 segment; nice 10, 4 CPU threads beside the slices job's 4), in turn:
# (1) the production's solved history (C3prod19 to T/r_s 40, then #36 B's continuation C3ext19 after 40) from T/r_s 15.0 and 30.0 to the last
#     record the continuation holds at the start (--T-end-rs 100 is clipped to it; later records extend it by --resume), kappa 1 and 2, the
#     18 columns, the cubic control on the first 8 r_s after 15.0, the signed branch readout every 1 r_s;
# (2) the sigma/a 8 history (C3lad27 c27 ell8, engine_fields_v16) from T/r_s 15.0 and 28.0 to 40, likewise;
# (3) mode dust on the production with the continuation: the B.5 version 2 columns of S4one/v2/B5v2_cols_prod.npz (T/r_s 12.125 and 15.0);
# (4) mode dust on the sigma/a 8 history: S4one/v2/B5v2_cols_s8.npz (T/r_s 13.0 and 16.0) to 40.
# Outputs in S4one/v2/; status lines in S4one/v2/status.txt. Usage (from ${QI2_ROOT}/engine): bash code/launch_single_object_B2_hist_dust_box_v1.sh
cd ${QI2_ROOT}/engine || exit 1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PAGE_V6_PATH=${QI2_ROOT}/v6
PY=${QI2_ROOT}/python
S=S4one/v2/status.txt
P1="--record H100b/C3prod19/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11.json --fields H100b/C3prod19/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11_fields.npz --fields-module engine_fields_v14"
P2="--record2 C3ext19/collapse_c19_rs64_A_R2_tau8_ell4_avg_ext100.json --fields2 C3ext19/collapse_c19_rs64_A_R2_tau8_ell4_avg_ext100_fields.npz --switch-T-rs 40"
S8="--record H100b/C3lad27/collapse_c27_rs64_A_R2_tau8_ell8_prep4_avg.json --fields H100b/C3lad27/collapse_c27_rs64_A_R2_tau8_ell8_prep4_avg_fields.npz --fields-module engine_fields_v16"
run() {
  tag=$1; shift
  echo "$tag start $(date -u +%H:%M:%S) UTC" >> $S
  nice -n 10 $PY code/engine_single_object_B_v2.py "$@" > S4one/v2/log_$tag.log 2>&1
  echo "$tag exit $? $(date -u +%H:%M:%S) UTC" >> $S
}
run B2_hist_prod --mode history --gpu 0 $P1 $P2 --launch-T 15.0,30.0 --kappas 1,2 --T-end-rs 100 --cubic-window-rs 8 --signed-every-rs 1 --out-json S4one/v2/B2_hist_prod.json --out-npz S4one/v2/B2_hist_prod.npz
run B2_hist_s8 --mode history --gpu 0 $S8 --launch-T 15.0,28.0 --kappas 1,2 --T-end-rs 40 --cubic-window-rs 8 --signed-every-rs 1 --out-json S4one/v2/B2_hist_s8.json --out-npz S4one/v2/B2_hist_s8.npz
run B2_dust_prod --mode dust --gpu 0 $P1 $P2 --columns-npz S4one/v2/B5v2_cols_prod.npz --kappas 1,2 --T-end-rs 100 --signed-every-rs 1 --out-json S4one/v2/B2_dust_prod.json --out-npz S4one/v2/B2_dust_prod.npz
run B2_dust_s8 --mode dust --gpu 0 $S8 --columns-npz S4one/v2/B5v2_cols_s8.npz --kappas 1,2 --T-end-rs 40 --signed-every-rs 1 --out-json S4one/v2/B2_dust_s8.json --out-npz S4one/v2/B2_dust_s8.npz
echo "B2 hist dust all done $(date -u +%H:%M:%S) UTC" >> $S
