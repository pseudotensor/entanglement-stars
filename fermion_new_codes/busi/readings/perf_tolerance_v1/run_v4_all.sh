#!/bin/bash
# run_v4_all.sh (busi, analyst, 21 Oct): every stored gate read by tolcmp_v4 (ReviewA_s3b_v1 sec. 1, the mode read) and, beside it, by
# tolcmp_v3 (5ae7977b) on the same arms and arguments. Box CPU only, nice 10, 8 threads, reads only. Outputs in
# ${QI2_ROOT}/tolcmp_v4<gate>/{v3,v4}/ (pairwise and gate jsons) and <gate>/tolcmp_v{3,4}.txt; staging dirs are symlinks.
#   run_v4_all.sh [gate ...]     (default: all)
set -u
B=${QI2_ROOT}/perf_redesign; O=$B/tolcmp_v4; C=$O/code; PY=${QI2_ROOT}/python
E="env OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 nice -n 10"
ISO=${QI2_ROOT}/run_h2; TISO=lad_ISO_128_s4_nkqh2_m0p2_rs128_n4096_R04_k10_tau8_ell4_prep4_A_R2_avg_newton_v36nq
P64=${QI2_ROOT}/final; TP64=lad_P64_s4_newton_v36nq_h100_1; SP64=$P64/states/orbitals_${TP64}_T0012.50.npz
R1=${QI2_ROOT}/T0012.00; R2=${QI2_ROOT}/RUNG2c40_v36q
PR2=${QI2_ROOT}/segment_c64s8L_lagged_sector_v9c3_anchor_passage_lz_20261020T091433Z; TC64=C64_s8_ell8_tau2_twofield_v4L
PC_RD1ISO="RD-1 per call on ISO_128 nkqh2 T0005.00's own matrices (RD1_build/percall/rd13_percall_v1.txt, 2n 8192, kappa 1 and 10): eigenvalues 7.1e-15, residuals 1.9e-13, orthonormality 2.6e-13, tracked sets equal to the stored masks, route banded, 0 fallbacks (ruling v2 sec. 5)"
PC_RD1P64="RD-1 per call on stored production matrices (RD1_build/percall/rd13_percall_v1.txt: eigenvalues 7.1e-15, residuals 2.1e-14 ||h||, sets equal to the stored masks; ruling v2 sec. 5)"
PC_MERGE64="RD-1 per call on stored production matrices (RD1_build/percall/rd13_percall_v1.txt: eigenvalues 7.1e-15, residuals 2.1e-14 ||h||, sets equal to the stored masks; ruling v2 sec. 5); RD-2b field gate on stored negative-centre and intact records (RD2b_build/gate/: fields at most 2.0e-9, E 2.3e-10; ruling v2 sec. 3); MEM-1 bitwise by its gate"
PC_MERGEISO="RD-1 per call ISO_128 2n 8192 7.1e-15; field gate records 81-88 fields <= 5.1e-11 (MERGE_build/fieldgate)"
PC_RD2C="RD-2 field gate (rd2_field_gate_v1, RD2b_build/gate/rd2_field_gate_rung2.json, 20 RUNG2c40b records): E rel at most 2.7e-9, fields 4.2e-9 (ruling v1 sec. 3.4); RD-2b field gate over 47 negative-centre records fields at most 2.0e-9, E 2.3e-10 (ruling v2 sec. 3 (3))"
PC_RD2W="RD-5 warm start against cold at the stored sources: fields 4.4e-16, E 2.5e-16 (ruling v1 sec. 3.4); the collocation route as RD-2 C"
PC_RD2S2="route identity at 2.125 (C's 412 solves fell back to I's route; ruling v2 sec. 4); RD-2 / RD-2b field gates as for S1"
PC_RD2B="RD-2b notes sec. 4 (b): Ce against P12 on the I arm stored 2.125 to 3.0 sources, fields 1.6e-9, E 2.5e-11 (ReviewA_tolv2_v1 sec. 3 (3))"
PC_DP="none: D' is the parent route itself (rd13 defaults, 4 threads against D's 8); no change route to test (ReviewA_s3b_v1 sec. 1: D' reported, no longer gates)"
PC_MEM1="MEM-1: G1 bitwise G0 on every array (memlaw notes sec. 6); no change route to test per call"
PC_TWO="RD-1c / RD-4 per-call tests as in their notes (supporting; twofield records carry no Newton stop flip)"
# stage DIR TAGSRC_PREFIX : links collapse_g{,_fields,_spectra,_handoff,_s11}.{json,npz} in DIR to PREFIX{...}
stage() { mkdir -p $1; for s in "" _fields _spectra _handoff _s11; do x=npz; [ -z "$s" ] && x=json; [ -e $2$s.$x ] && ln -sfn $2$s.$x $1/collapse_g$s.$x; done; }
# run NAME X D P TAG AFTER RS MINREC PERCALL NAMES [slice args...]
run() {
  local N=$1 X=$2 D=$3 P=$4 T=$5 A=$6 RS=$7 MR=$8 PC=$9 NM=${10}; shift 10
  mkdir -p $O/$N
  $E $PY -I $C/tolcmp_v3.py $X $D $P --tag $T --after $A --rs $RS --names "$NM" --percall "$PC" --out $O/$N/v3 "$@" > $O/$N/tolcmp_v3.txt 2>&1; local r3=$?
  $E $PY -I $C/tolcmp_v4.py $X $D $P --tag $T --after $A --rs $RS --names "$NM" --percall "$PC" --min-records $MR --out $O/$N/v4 "$@" > $O/$N/tolcmp_v4.txt 2>&1; local r4=$?
  echo "$N v3 rc $r3 v4 rc $r4 $(date -u +%FT%TZ) | v3: $(grep -h '^######## GATE' $O/$N/tolcmp_v3.txt | grep -o '[A-Z]*$') | v4: $(grep -h '^######## GATE' $O/$N/tolcmp_v4.txt | grep -o '[A-Z]*$')"
}
S=$O/stage
GATES=${*:-"rd1_p64 rd1_entry dprime sec3b rd2_S1_C rd2_S1_W rd2_S2_C rd2_S2_W rd2_box_C rd2_box_W rd2b_S2 merge_p64 merge_iso_A merge_iso_AB merge_iso_Fd rd1c_card rd4_card mem1"}
for g in $GATES; do
  case $g in
    rd1_p64) run $g $B/RD1_build/tolcmp64/B $B/RD1_build/tolcmp64/D $B/RD1_build/tolcmp64/prod rd13 11.5 64 8 "$PC_RD1P64" "B,D,prod" \
               --slice-x $B/RD1_build/replay64/B/states/orbitals_rd13_B_T0012.50.npz --slice-d $B/RD1_build/replay64/D/states/orbitals_rd13_D_T0012.50.npz --slice-p $SP64;;
    rd1_entry) stage $S/entry/B $B/RD1_build/entry_iso128/B/collapse_rd13_B; stage $S/entry/D $B/RD1_build/entry_iso128/D/collapse_rd13_D; stage $S/entry/P $ISO/collapse_$TISO
               run $g $S/entry/B $S/entry/D $S/entry/P g 5.0 128 8 "$PC_RD1ISO" "B,D,prod" \
               --slice-x $B/RD1_build/entry_iso128/B/states/orbitals_rd13_B_T0005.50.npz --slice-d $B/RD1_build/entry_iso128/D/states/orbitals_rd13_D_T0005.50.npz;;
    dprime) stage $S/entry/Dp $B/RD1_build/dprime/out/Dp/collapse_rd13_Dp; stage $S/entry/D $B/RD1_build/entry_iso128/D/collapse_rd13_D; stage $S/entry/P $ISO/collapse_$TISO
               run $g $S/entry/Dp $S/entry/D $S/entry/P g 5.0 128 2 "$PC_DP" "Dp,D,prod";;
    sec3b) run $g $B/MERGE_build/gate115b/stage/ABd $B/MERGE_build/gate115b/stage/Fdd2 $B/MERGE_build/gate115b/stage/prod mf 11.5 128 2 "$PC_RD1ISO" "ABd,Fdd2,jobb";;
    rd2_S1_C|rd2_S1_W|rd2_box_C|rd2_box_W)
           seg=${g#rd2_}; seg=${seg%_*}; arm=${g##*_}; [ $seg = S1 ] && SD=$B/RD2_build/h100f/v2/S1 || SD=$B/RD2_build/replay
           stage $S/rd2_$seg/I $SD/I/collapse_RUNG2c40b_srf_rd2I; stage $S/rd2_$seg/$arm $SD/$arm/collapse_RUNG2c40b_srf_rd2$arm; stage $S/rd2_$seg/P $R1/collapse_RUNG2c40b_srf
           PC="$PC_RD2C"; [ $arm = W ] && PC="$PC_RD2W"
           SL=(); [ $seg = S1 ] && SL=(--slice-x $SD/$arm/states/orbitals_RUNG2c40b_srf_rd2${arm}_T0012.50.npz --slice-d $SD/I/states/orbitals_RUNG2c40b_srf_rd2I_T0012.50.npz)
           run $g $S/rd2_$seg/$arm $S/rd2_$seg/I $S/rd2_$seg/P g 12.0 64 8 "$PC" "$arm,I,prod" "${SL[@]}";;
    rd2_S2_C|rd2_S2_W)
           arm=${g##*_}; SD=$B/RD2_build/h100f/v2/S2
           stage $S/rd2_S2/I $SD/I/collapse_RUNG2c40_v36q_rd2I; stage $S/rd2_S2/$arm $SD/$arm/collapse_RUNG2c40_v36q_rd2$arm; stage $S/rd2_S2/P $R2/collapse_RUNG2c40_v36q
           PC="$PC_RD2S2"; [ $arm = W ] && PC="$PC_RD2W; $PC_RD2S2"
           run $g $S/rd2_S2/$arm $S/rd2_S2/I $S/rd2_S2/P g 2.0 64 8 "$PC" "$arm,I,prod" \
               --slice-x $SD/$arm/states/orbitals_RUNG2c40_v36q_rd2${arm}_T0003.00.npz --slice-d $SD/I/states/orbitals_RUNG2c40_v36q_rd2I_T0003.00.npz;;
    rd2b_S2) run $g $B/RD2b_build/tolcmp_v1a/C $B/RD2b_build/tolcmp_v1a/I $B/RD2b_build/tolcmp_v1a/prod rd2b 2.0 64 8 "$PC_RD2B" "C,I,prod" \
               --slice-x $B/RD2b_build/h100f/S2/C/states/orbitals_RUNG2c40_v36q_rd2bC_T0003.00.npz --slice-d $B/RD2b_build/h100f/S2/I/states/orbitals_RUNG2c40_v36q_rd2bI_T0003.00.npz;;
    merge_p64) G=$B/MERGE_build/gate64/stage
           run $g $G/GA $G/G0 $G/prod mf 11.5 64 8 "$PC_MERGE64" "GA,G0,prod" \
               --slice-x $B/MERGE_build/replay64/GA/states/orbitals_mf_GA_T0012.50.npz --slice-d $B/MERGE_build/replay64/G0/states/orbitals_mf_G0_T0012.50.npz --slice-p $SP64;;
    merge_iso_A|merge_iso_AB|merge_iso_Fd) arm=${g##*_}; G=$B/MERGE_build/gateiso/stage
           run $g $G/$arm $G/D $G/prod mf 5.0 128 8 "$PC_MERGEISO" "$arm,D,prod";;
    rd1c_card) G=$B/RD1c_build/h100f/gate
           run $g $G/RD1CAB_B7_20261021T175449Z $G/RD1CAB_A6_20261021T174101Z $PR2 $TC64 13 64 8 "$PC_TWO" "B7,A6,P_R2";;
    rd4_card) G=$B/RD4_build/h100f/gate
           run $g $G/RD4AB_T6_20261021T083004Z $G/RD4AB_B5_20261021T081537Z $PR2 $TC64 13 64 8 "$PC_TWO" "T6,B5,P_R2";;
    mem1) M=$B/memlaw/rd13m/replay64
           stage $S/mem1/G1 $M/G1/collapse_rd13m_G1; stage $S/mem1/G0 $M/G0/collapse_rd13m_G0; stage $S/mem1/P $P64/run/collapse_$TP64
           run $g $S/mem1/G1 $S/mem1/G0 $S/mem1/P g 11.5 64 8 "$PC_MEM1" "G1,G0,prod" \
               --slice-x $M/G1/states/orbitals_rd13m_G1_T0012.50.npz --slice-d $M/G0/states/orbitals_rd13m_G0_T0012.50.npz --slice-p $SP64;;
    *) echo "unknown gate $g"; exit 2;;
  esac
done
echo "ALL DONE $(date -u +%FT%TZ)"
