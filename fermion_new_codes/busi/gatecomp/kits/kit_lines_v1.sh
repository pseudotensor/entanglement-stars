#!/bin/bash
# kit_lines_v1.sh (busi lane, GATECOMP kits (i) and (ii), inbox #400; busi/results/ready_jobs_v3.md sec. 40): writes one ARGS file per
# launch line into OUT (default ./lines), exactly as the card runs it, plus LINES.tsv (name, engine, args md5, what it is). Nothing runs.
# Card layout (solve may change CW; every path below follows it):  $CW/code (the code16d dir, MD5SUMS_code16d.txt cdc9eb3f),
#   $CW/<line>/{run,states,logs}, the restart views in $CW/<line>_view.
# Placeholders for the post-birth lines (solve fills them from the reader's sec. 3 birth of the named member, before launch):
#   SLICE_T = the member's last stored slice at or below T_birth - 0.25 (one of the slice list below), END_T = T_birth + 0.75 (point) or
#   + 0.5 (tight, reint). Defaults are reviewer A's expected birth near 11.0 to 11.5: SLICE_T 10.50, END_T 12.25.
set -u
OUT=${OUT:-lines}; CW=${CW:-${QI2_ROOT}/gatecomp}
SLICE_T=${SLICE_T:-10.50}; END_PT=${END_PT:-12.25}; END_TI=${END_TI:-12.00}; SLT=$(printf "%07.2f" $SLICE_T)
mkdir -p $OUT
ENGINE=engine_collapse_v36nq_rd13mf.py   # 248f25f35e48a88a84b102ddd99a5713; field module engine_fields_v16.py = v16d 7a69faad (v16e HELD)
# the route: ReviewA_s3b_v1 sec. 1 (#402): RD-1 ADOPTED for 4096-site lines on the v16d collocation route, the gate set included:
# --eigh banded --fields collocation, cache off. ROUTE=dense gives the dense arm (the entry test's 4-record arm; the CPU check mode).
ROUTE=${ROUTE:-banded}
DENSE="--eigh $ROUTE --jvp eigenbasis --nk-cache-host off"
base() {  # $1 = kappa_sea, $2 = T_over_rs
echo "--m 0.2 --g 0.25 --R0-over-rs 4 --reading A --kappa-sea $1 --reference R2 --half-offset 0 --collar-policy flag --support-rel 1e-14 --gpu 0 \
--site-fields averaged --stop-E 1e-7 --stop-fields 1e-6 --maxit 20 --field-rtol 1e-12 --field-atol 1e-16 --delta-rtol-every 10 --commutator-every 1 \
--streams 0 --m1-simpson-steps auto --c-LZ 1.0 --window-max 4 --s11-collars 8,16 --s11-stop 1 --field-kernel-cache --v36-fixes --T-over-rs $2"; }
NK="--mix newton --mix-alpha 0.5 --nk-maxiter 30 --nk-inner 15 --nk-krylov 40 --nk-jvp exact --nk-line-search halving --nk-halvings 6 --nk-restarts 2 \
--nk-plain-fallback on --nk-host-release on --nk-set resolve --iter-cache drop --prep-memory lean --end-above 0.001"
PLAIN="--mix plain --mix-alpha 0.5 --iter-cache drop --prep-memory lean"
: > $OUT/LINES.tsv
put() { echo "$2" | tr -s ' ' > $OUT/$1.args; printf '%s\t%s\t%s\t%s\n' "$1" "$ENGINE" "$(md5sum < $OUT/$1.args | cut -c1-32)" "$3" >> $OUT/LINES.tsv; }
io() { echo "--slice-dir $CW/$1/states --out $CW/$1/run --tag $2"; }

# ---- kit (i): the chain validation, ISO at r_s/a 128 on a 2048-site chain (16 r_s), fresh from T = 0 to 13 (reviewer A sec. 2 (i))
T_CV=lad_ISO_128_s4_cv2048_m0p2_rs128_n2048_R04_k10_tau8_ell4_prep4_A_R2_avg_newton_v36nq_rd13mf
put CV2048 "$(base 10 13) --rs 128 --nsites 2048 --ell 4 --prep-ell 4 --tau 8 --dust-set isotropic $NK --pair-workers 2 \
--slices-c128 3.0000,6.0000,9.0000,11.0000,11.5000,12.0000,13.0000 $(io CV2048 $T_CV) --fields collocation $DENSE" \
"kit (i): job (b)'s args, nsites 4096 -> 2048, fresh (no resume), T 13, merged engine dense"

# ---- kit (ii): the well triple at r_s/a 256 on 4096 sites (16 r_s), K 14, tau 16 a, prep-ell 4, T 0 to 14 (reviewer A sec. 2 (ii))
SL256="0.0000,3.0000,6.0000,9.0000,10.0000,10.5000,10.7500,11.0000,11.2500,11.5000,11.7500,12.0000,12.5000,13.0000,14.0000"
for S in 4 8 16; do
  T=lad_W256_s${S}_m0p2_rs256_n4096_R04_k14_tau16_ell${S}_prep4_A_R2_avg_newton_v36nq_rd13mf
  eval T_W$S=$T
  put W256_s$S "$(base 14 14) --rs 256 --nsites 4096 --ell $S --prep-ell 4 --tau 16 --dust-set well $NK --pair-workers 2 \
--slices-c128 $SL256 $(io W256_s$S $T) --fields collocation $DENSE" "kit (ii) member sigma/a $S"
done
M8=$CW/W256_s8; SL9=$M8/states/orbitals_${T_W8}_T0009.00.npz
# controls (reviewer A): tau/r_s 1/32 from the 8a member's T = 9 slice (view tau=8); the plain mix on the same stretch (view mix=plain; ivp fields)
put C_tau32 "$(base 14 14) --rs 256 --nsites 4096 --ell 8 --prep-ell 4 --tau 8 --dust-set well $NK --pair-workers 2 \
--resume-slice $SL9 --resume-run $CW/C_tau32_view/$T_W8 --segment-end-T-over-rs 14 --slices-c128 12.0000,13.0000 \
$(io C_tau32 ${T_W8/_tau16_/_tau8_}_from9) --fields collocation $DENSE" "control: tau 8 a from W256_s8 T9 (resume_view_v1 --set tau=8), to 14"
put C_plain "$(base 14 14) --rs 256 --nsites 4096 --ell 8 --prep-ell 4 --tau 16 --dust-set well $PLAIN \
--resume-slice $SL9 --resume-run $CW/C_plain_view/$T_W8 --segment-end-T-over-rs 14 --slices-c128 12.0000,13.0000 \
$(io C_plain ${T_W8/_newton_/_plain_ivp_}_from9) --fields ivp $DENSE" "control: plain mix + ivp fields from W256_s8 T9 (view --set mix=plain), to 14"
# ReviewA_resview_v1's step control by local error: two tau/2 records from the 8a member's T 3 and T 6 slices through the view (tau=8),
# each ending on the parent's next record clock (3.0625, 6.0625); 4 records in all (about 15 min). KEEP W256_s8's T 3 and T 6 slices.
for T0 in 3 6; do
  put C_step$T0 "$(base 14 14) --rs 256 --nsites 4096 --ell 8 --prep-ell 4 --tau 8 --dust-set well $NK --pair-workers 2 \
--resume-slice $M8/states/orbitals_${T_W8}_T$(printf '%07.2f' $T0).npz --resume-run $CW/C_step${T0}_view/$T_W8 --segment-end-T-over-rs $T0.0625 --slices-c128 \"\" \
$(io C_step$T0 ${T_W8/_tau16_/_tau8_}_step$T0) --fields collocation $DENSE" "resview step control: two tau/2 records from W256_s8 T$T0 (view tau=8) to $T0.0625"
done
# the point-sampled control at the root (view site_fields=dense), 8a, from SLICE_T to END_PT
put C_point "$(base 14 14 | sed 's/--site-fields averaged/--site-fields dense/') --rs 256 --nsites 4096 --ell 8 --prep-ell 4 --tau 16 --dust-set well $NK --pair-workers 2 \
--resume-slice $M8/states/orbitals_${T_W8}_T${SLT}.npz --resume-run $CW/C_point_view/$T_W8 --segment-end-T-over-rs $END_PT --slices-c128 \"\" \
$(io C_point ${T_W8}_dense_from${SLICE_T}) --fields collocation $DENSE" "control: point-sampled site fields from W256_s8 SLICE_T (view --set site_fields=dense) to T_birth + 0.75"
# reviewer B: the tightening check (8a; stop rule 100x tighter; no view: stop_E is not a checked arg) and the local re-integration (4a, 16a; tau 8)
put C_tight "$(base 14 14 | sed 's/--stop-E 1e-7 --stop-fields 1e-6/--stop-E 1e-9 --stop-fields 1e-8/') --rs 256 --nsites 4096 --ell 8 --prep-ell 4 --tau 16 --dust-set well $NK --pair-workers 2 \
--resume-slice $M8/states/orbitals_${T_W8}_T${SLT}.npz --resume-run $M8/run/$T_W8 --segment-end-T-over-rs $END_TI --slices-c128 \"\" \
$(io C_tight ${T_W8}_tight_from${SLICE_T}) --fields collocation $DENSE" "reviewer B: stop-E 1e-9, stop-fields 1e-8 from W256_s8 SLICE_T to T_birth + 0.5"
for S in 4 16; do
  eval T=\$T_W$S
  put C_reint_s$S "$(base 14 14) --rs 256 --nsites 4096 --ell $S --prep-ell 4 --tau 8 --dust-set well $NK --pair-workers 2 \
--resume-slice $CW/W256_s$S/states/orbitals_${T}_T${SLT}.npz --resume-run $CW/C_reint_s${S}_view/$T --segment-end-T-over-rs $END_TI --slices-c128 \"\" \
$(io C_reint_s$S ${T/_tau16_/_tau8_}_from${SLICE_T}) --fields collocation $DENSE" "reviewer B: local re-integration, tau 8 a from W256_s$S SLICE_T (view --set tau=8) to T_birth + 0.5"
done
sed -i 's/ --slices-c128 "" / /' $OUT/*.args   # an empty slice list is the default (no slices)
for f in $OUT/*.args; do n=$(basename $f .args); m=$(md5sum < $f | cut -c1-32); awk -v n=$n -v m=$m 'BEGIN{FS=OFS="\t"} $1==n{$3=m}1' $OUT/LINES.tsv > $OUT/.t && mv $OUT/.t $OUT/LINES.tsv; done
cat $OUT/LINES.tsv
