#!/bin/bash
# launch_single_object_B2_retry_box_v1.sh (builder analyst, the HORIZON version 2 builder): the B2 runs that ran out of GPU memory on box GPU 0
# (each dense eigendecomposition of the 16384-dimension h asks 16 GiB of workspace; two such jobs beside the S11 gates do not fit the A6000),
# run one at a time after launch_single_object_B2_hist_prod_box_v2.sh's run has ended: (1) the sigma/a 8 history again (B2_hist_s8 exit 1 at
# 02:09:16 UTC, nothing written): T/r_s 15.0 and 28.0 to 40, kappa 1 and 2 -> S4one/v2/B2_hist_s8_b; (2) and (3) each dust run of the first
# chain again only if its status line shows a nonzero exit (B2_dust_prod with the continuation's snapshot S4one/v2/inputs/ext_snap1/ ->
# B2_dust_prod_b; B2_dust_s8 -> B2_dust_s8_b). nice 10, 4 CPU threads. Usage (from ${QI2_ROOT}/engine): nohup bash code/launch_single_object_B2_retry_box_v1.sh &
cd ${QI2_ROOT}/engine || exit 1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PAGE_V6_PATH=${QI2_ROOT}/v6
PY=${QI2_ROOT}/python
S=S4one/v2/status.txt
until grep -q "B2_hist_prod_b exit" $S 2>/dev/null; do sleep 30; done
S8="--record H100b/C3lad27/collapse_c27_rs64_A_R2_tau8_ell8_prep4_avg.json --fields H100b/C3lad27/collapse_c27_rs64_A_R2_tau8_ell8_prep4_avg_fields.npz --fields-module engine_fields_v16"
D=S4one/v2/inputs/ext_snap1
P1="--record H100b/C3prod19/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11.json --fields H100b/C3prod19/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11_fields.npz --fields-module engine_fields_v14"
P2="--record2 $D/collapse_c19_rs64_A_R2_tau8_ell4_avg_ext100.json --fields2 $D/collapse_c19_rs64_A_R2_tau8_ell4_avg_ext100_fields.npz --switch-T-rs 40"
run() {
  tag=$1; shift
  echo "$tag start $(date -u +%H:%M:%S) UTC" >> $S
  nice -n 10 $PY code/engine_single_object_B_v2.py "$@" > S4one/v2/log_$tag.log 2>&1
  echo "$tag exit $? $(date -u +%H:%M:%S) UTC" >> $S
}
run B2_hist_s8_b --mode history --gpu 0 $S8 --launch-T 15.0,28.0 --kappas 1,2 --T-end-rs 40 --cubic-window-rs 8 --signed-every-rs 1 --out-json S4one/v2/B2_hist_s8_b.json --out-npz S4one/v2/B2_hist_s8_b.npz
until grep -q "B2 hist dust all done" $S 2>/dev/null; do sleep 30; done
if ! grep -q "B2_dust_prod exit 0" $S; then
  run B2_dust_prod_b --mode dust --gpu 0 $P1 $P2 --columns-npz S4one/v2/B5v2_cols_prod.npz --kappas 1,2 --T-end-rs 100 --signed-every-rs 1 --out-json S4one/v2/B2_dust_prod_b.json --out-npz S4one/v2/B2_dust_prod_b.npz
fi
if ! grep -q "B2_dust_s8 exit 0" $S; then
  run B2_dust_s8_b --mode dust --gpu 0 $S8 --columns-npz S4one/v2/B5v2_cols_s8.npz --kappas 1,2 --T-end-rs 40 --signed-every-rs 1 --out-json S4one/v2/B2_dust_s8_b.json --out-npz S4one/v2/B2_dust_s8_b.npz
fi
echo "B2 retry all done $(date -u +%H:%M:%S) UTC" >> $S
