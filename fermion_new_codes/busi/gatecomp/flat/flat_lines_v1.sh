#!/bin/bash
# flat_lines_v1.sh (busi, GATECOMP, the flat triple; ReviewA_flatq_v1 secs. 1 and 3; busi/notes/GATECOMP_flat_build_v1.md): writes one ARGS
# file per launch line into OUT (default ./lines_flat), exactly as the card runs it, plus LINES.tsv (name, engine, args md5, what). Nothing runs.
# STAGED: waits for the lead's word on the card (inbox #408 (4)); nothing on a card from this script.
# ReviewA_edge_v1 (#418): TAPER, W = 0.5 r_s is the member (EDGE=taper W=0.5 K=15 CACHE=off); W = 1 the fallback; the hard wall is OUT.
# The members: r_s/a 256 on 4096 sites (16 r_s), --prep flatbox (reviewer A's hard wall at R_b = 4 r_s), K = the box catalog's top + 5 = 16
# (measured: top channel 11 at N_B 2560, flat_profile_v1; reviewer A's estimate was 13 and K 18), tau 16 a, prep-ell 4, sigma/a 4, 8, 16,
# banded + collocation (RD-1 adopted for 4096-site lines), --nk-cache-host on by default (K 16 off projects 76 to 80 GB on an 80 GB card:
# notes sec. 6; CACHE=off writes the off lines), the reopen2 Newton block, pair workers 2, T 0 to T_END (default 6.5: the comparator's
# caustic is 5.35 hard and 4.39 to 4.78 taper; reviewer A's stop at T_gate + 1 record is the member's end; T_END is the cap).
# Slices (the lead's resume-audit rule, inbox #410/#412): T 0, T 0.5, then every 0.5 r_s (records at K 15/16 take 300 to 510 s, so
# 1 r_s would be 1.3 to 2.3 h, above the audit's 90 min), and every 0.25 r_s from 3.25 to 6.0 through birth and the caustic
# (comparator births 3.90 to 4.72), then 6.5: 20 slices, about 17 GB each at K 16 (340 GB per member at most; prune by the keep rule).
# The controls on the flat 8a member (ReviewA_flatq_v1 sec. 3), each from a slice of F256_s8 through resume_view_v1 where a checked
# arg changes (ReviewA_resview_v1: tau, mix and re-integration views admitted on #405's tests; the site_fields view is live):
#   C_tau32 (tau 8 a), C_plain (mix plain + ivp fields), from SLICE_T = the last slice at or below T_birth - 1 (plan: "from the last slice
#   >= 1 r_s before birth"; the ruling's "T 9" is the well's clock, translated so; for reviewer A), to END_G = T_gate + 1 record;
#   C_step1, C_step3: two tau/2 records from the T 1 and T 3 slices to the parent's next record clock (the resview local-error design);
#   C_point (site_fields dense; only if the free re-solve's shift exceeds the floor, resview (3)), C_tight (stop rule 100x tighter, no view),
#   C_reint_s4, C_reint_s16 (tau 8 a): from SLICE_B = the last slice at or below T_birth - 0.25, to T_birth + 0.5 (END_B).
# Placeholders (busi fills them from the 8a member's birth read before launch): SLICE_T, END_G, SLICE_B, END_B.
# Card layout: $CW/code = code16d_flat (code16d + engine_collapse_v36nq_rd13mfb.py 02a1729a), $CW/<line>/{run,states,logs}.
set -u
OUT=${OUT:-lines_flat}; CW=${CW:-${QI2_ROOT}/flat}; K=${K:-16}; CACHE=${CACHE:-on}; T_END=${T_END:-6.5}
SLICE_T=${SLICE_T:-3.5000}; END_G=${END_G:-6.0}; SLICE_B=${SLICE_B:-4.2500}; END_B=${END_B:-5.25}
EDGE=${EDGE:-hard}; EOPT="--prep flatbox"; ETAG=fbh
[ "$EDGE" = taper ] && { W=${W:-0.5}; EOPT="--prep flatbox --prep-edge taper --prep-taper $W"; ETAG=fbt$(echo $W | tr -d .); }   # the tag carries W (fbt05, fbt10)
mkdir -p $OUT
ENGINE=engine_collapse_v36nq_rd13mfb.py   # 02a1729ab808948d443eed9144b75b03 (over 248f25f3); engine_fields_v16.py = v16d 7a69faad
ROUTE="--eigh banded --jvp eigenbasis --nk-cache-host $CACHE"
base() { echo "--m 0.2 --g 0.25 --R0-over-rs 4 --reading A --kappa-sea $1 --reference R2 --half-offset 0 --collar-policy flag --support-rel 1e-14 --gpu 0 \
--site-fields averaged --stop-E 1e-7 --stop-fields 1e-6 --maxit 20 --field-rtol 1e-12 --field-atol 1e-16 --delta-rtol-every 10 --commutator-every 1 \
--streams 0 --m1-simpson-steps auto --c-LZ 1.0 --window-max 4 --s11-collars 8,16 --s11-stop 1 --field-kernel-cache --v36-fixes --T-over-rs $2"; }
NK="--mix newton --mix-alpha 0.5 --nk-maxiter 30 --nk-inner 15 --nk-krylov 40 --nk-jvp exact --nk-line-search halving --nk-halvings 6 --nk-restarts 2 \
--nk-plain-fallback on --nk-host-release on --nk-set resolve --iter-cache drop --prep-memory lean --end-above 0.001 --source-continuation two-stage"
# ReviewA_w256end_v1 sec. 3 (ii): every Newton line carries the source continuation as the fallback (stage 2 = the polish: the plain Newton on
# this record's map from the continuation's endpoint, judged by --end-above 1e-3); the member runs beside flat_stall_watch_v1.py (two
# consecutive path records, two in ten, or a decade's rise in the kept residual within 1 r_s stop it for a read). C_plain (mix plain) cannot carry it.
: > $OUT/LINES.tsv
put() { echo "$2" | tr -s ' ' > $OUT/$1.args; printf '%s\t%s\t%s\t%s\n' "$1" "$ENGINE" "$(md5sum < $OUT/$1.args | cut -c1-32)" "$3" >> $OUT/LINES.tsv; }
io() { echo "--slice-dir $CW/$1/states --out $CW/$1/run --tag $2"; }
SLF=${SLF:-0.0000,0.5000,1.0000,1.5000,2.0000,2.5000,3.0000,3.2500,3.5000,3.7500,4.0000,4.2500,4.5000,4.7500,5.0000,5.2500,5.5000,5.7500,6.0000,6.5000}
for S in 4 8 16; do
  T=lad_F256_s${S}_${ETAG}_m0p2_rs256_n4096_R04_k${K}_tau16_ell${S}_prep4_A_R2_avg_newton_v36nq_rd13mfb
  put F256_s$S "$(base $K $T_END) --rs 256 --nsites 4096 --ell $S --prep-ell 4 --tau 16 --dust-set well $EOPT $NK --pair-workers 2 \
--slices-c128 $SLF $(io F256_s$S $T) --fields collocation $ROUTE" "flat triple member sigma/a $S ($EDGE edge, K $K, cache $CACHE, T 0 to $T_END)"
done
T8=lad_F256_s8_${ETAG}_m0p2_rs256_n4096_R04_k${K}_tau16_ell8_prep4_A_R2_avg_newton_v36nq_rd13mfb; M8=$CW/F256_s8
sl() { echo "$M8/states/orbitals_${T8}_T$(printf '%07.2f' $1).npz"; }
B8="$(base $K $T_END) --rs 256 --nsites 4096 --ell 8 --prep-ell 4 --dust-set well $EOPT"
put C_tau32 "$B8 --tau 8 $NK --pair-workers 2 --resume-slice $(sl $SLICE_T) --resume-run $CW/C_tau32_view/$T8 --segment-end-T-over-rs $END_G \
--slices-c128 $END_G $(io C_tau32 ${T8/_tau16_/_tau8_}_from$SLICE_T) --fields collocation $ROUTE" "control on F256_s8: tau 8 a from SLICE_T (view tau=8) to END_G"
put C_plain "$B8 --tau 16 --mix plain --mix-alpha 0.5 --iter-cache drop --prep-memory lean --resume-slice $(sl $SLICE_T) --resume-run $CW/C_plain_view/$T8 \
--segment-end-T-over-rs $END_G --slices-c128 $END_G $(io C_plain ${T8/_newton_/_plain_ivp_}_from$SLICE_T) --fields ivp $(echo $ROUTE | sed 's/--nk-cache-host on/--nk-cache-host off/')" "control on F256_s8: plain mix + ivp fields from SLICE_T (view mix=plain) to END_G (cache off: the host cache is a Newton option)"
for T0 in 1 3; do
  put C_step$T0 "$B8 --tau 8 $NK --pair-workers 2 --resume-slice $(sl $T0) --resume-run $CW/C_step${T0}_view/$T8 --segment-end-T-over-rs $T0.0625 \
--slices-c128 9.0000 $(io C_step$T0 ${T8/_tau16_/_tau8_}_step$T0) --fields collocation $ROUTE" "resview step control on F256_s8: two tau/2 records from T $T0 (view tau=8) to $T0.0625"
done
put C_point "$(echo "$B8" | sed 's/--site-fields averaged/--site-fields dense/') --tau 16 $NK --pair-workers 2 --resume-slice $(sl $SLICE_B) --resume-run $CW/C_point_view/$T8 \
--segment-end-T-over-rs $END_B --slices-c128 9.0000 $(io C_point ${T8}_dense_from$SLICE_B) --fields collocation $ROUTE" "control on F256_s8: point-sampled site fields from SLICE_B (view site_fields=dense) to END_B; only if the free re-solve's shift exceeds the floor"
put C_tight "$(echo "$B8" | sed 's/--stop-E 1e-7 --stop-fields 1e-6/--stop-E 1e-9 --stop-fields 1e-8/') --tau 16 $NK --pair-workers 2 --resume-slice $(sl $SLICE_B) \
--resume-run $M8/run/$T8 --segment-end-T-over-rs $END_B --slices-c128 9.0000 $(io C_tight ${T8}_tight_from$SLICE_B) --fields collocation $ROUTE" "reviewer B: stop-E 1e-9, stop-fields 1e-8 on F256_s8 from SLICE_B to END_B"
for S in 4 16; do
  TS=lad_F256_s${S}_${ETAG}_m0p2_rs256_n4096_R04_k${K}_tau16_ell${S}_prep4_A_R2_avg_newton_v36nq_rd13mfb
  put C_reint_s$S "$(base $K $T_END) --rs 256 --nsites 4096 --ell $S --prep-ell 4 --dust-set well $EOPT --tau 8 $NK --pair-workers 2 \
--resume-slice $CW/F256_s$S/states/orbitals_${TS}_T$(printf '%07.2f' $SLICE_B).npz --resume-run $CW/C_reint_s${S}_view/$TS --segment-end-T-over-rs $END_B \
--slices-c128 9.0000 $(io C_reint_s$S ${TS/_tau16_/_tau8_}_from$SLICE_B) --fields collocation $ROUTE" "reviewer B: local re-integration, tau 8 a on F256_s$S from SLICE_B (view tau=8) to END_B"
done
# ReviewA_w256end_v1 sec. 3 (iii): tau = 16 a stays; the only step remedy is the local tau/32 continuation from the last slice before a stall,
# at most 6 records or 3 card-hours, labelled (not spliced): C_loc32 on F256_s8 from SLICE_L (placeholder, the stall's last slice), view tau=8.
SLICE_L=${SLICE_L:-3.0000}
put C_loc32 "$B8 --tau 8 $NK --pair-workers 2 --resume-slice $(sl $SLICE_L) --resume-run $CW/C_loc32_view/$T8 --segment-end-T-over-rs $(echo "$SLICE_L + 6 * 0.03125" | bc -l | sed 's/0*$//') \
--slices-c128 9.0000 $(io C_loc32 ${T8/_tau16_/_tau8_}_loc$SLICE_L) --fields collocation $ROUTE" "w256end 3 (iii): the local tau/32 continuation on F256_s8 from SLICE_L (view tau=8), 6 records, labelled; only after a stall"
echo "wrote $OUT: $(wc -l < $OUT/LINES.tsv) lines"; cat $OUT/LINES.tsv
