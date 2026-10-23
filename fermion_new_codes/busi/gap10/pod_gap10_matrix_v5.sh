#!/bin/bash
# GAP10 (B5) matrix rerun v5 (busi/notes/GAP10_T7_final_v2.md section 2; change 2 of the cold reading of version 1): the 10
# matrix runs (T7: the 4 matched-preparation twins, then the 4 fixed runs; then T6's 2 flat runs at production size with the
# Fourier-exact check at 1 and 2 r_s), then T5 (the Taylor solver against Chebyshev at r_s/a 24, [100] d 4 and [111] d 8) and
# T4 (the r_s/a 24 matrix at 10 positions and the parent ladder H 60, 88, 124 on common vectors clipped to half-width 60 at
# [100] d 4 and 16), per the gap note's section 7 and gap10_matrix_v2.py, one process at a time, on pod 3's container disk
# ${QI2_ROOT}/busi_local. v4 wrote to the network volume /workspace, dead since about 09:50 UTC ("Transport endpoint is not
# connected"): nothing here names /workspace (a stat of it can hang a shell), and gap10_matrix_v2.py is pointed at the
# container disk through GAP10_CODE and GAP10_RUNS; the script aborts if the generated matrix names /workspace.
#
# Status file $ST (matrix_status_v5.txt), 2 lines per run:
#   "<tier> run <label> exit <rc> path <stem> md5npz <m1> md5json <m2> napply <n> wall <s>s <HH:MM:SS>"
#   "<tier> acc <label> rs_file=.. reldiff=.. N_is_one=.. r_h=.. kappa_h=.. crossings=.. sweeps=.. norm_err_max=.. energy_err_max=..
#    cheb_excess_max=.. fft1=.. fft2=.. ms_per_app=.. need .. free .. | unitarity=PASS/FAIL energy=.. cheb_excess=.. rs_file=.. N_is_one=.. r_h=.. fft1=.."
# The acceptance items (the delivery note section 2): rs_file against --rs within 1 percent and N_is_one False (T7, from
# [record]); r_h within 0.01 a of 22.2741 a on [100] and 22.2416 a on [111] (T7, from [profile]; the inner crossing printed
# beside); |norm-1| < 1e-12 at every sweep end and |E-E0| < 1e-10 (from the [sweep] lines); the Chebyshev vector norm excess
# < 1e-11 (from [monitor]); the Fourier-exact difference < 1e-11 at 1 r_s (T6, from [fft check]; the 2 r_s value is the lattice
# front's tail at the faces, about 1e-5 to 1e-6, recorded and not a failure). T4's parent ladder and T5's solver are accepted
# from table F of the analysis (gap note section 8 checks 3 and 5), not from the logs. "na" = the item does not apply to the run.
#
# Resumable: a run whose "run" line carries exit 0 is skipped. One process at a time: the script waits while another
# gap10_run_v2 process of this lane is alive. Memory: before each run the card's free memory (nvidia-smi) is compared with the
# run's estimated need plus GAP10_MARGIN_GB (default 6); a run that does not fit is logged DEFERRED and the script goes on,
# so that the runs that fit beside Computation C's sigma/a 2 member (27 to 33 GB of the 79 GiB card for 8 to 12 hours) run now
# and the ones that do not wait for a window (rerun the script in the window: the exit-0 runs are skipped).
# Estimated needs (6 state buffers of B*4*L^3*16 bytes + about 200 bytes per site of fields and geometry + about 3 GB of receiver
# matrices and context; + 1 state buffer for the Fourier reference copy; the run's own [memory] line prints the 6-buffer part only):
#   T7  B 7, L 207: about 28 GB (fits beside C)      T6  B 4, L 225, FFT: about 25 GB (fits)
#   T5  B 7, L 177, Taylor: about 21 GB (fits)        T4 rs24  B 7, L 177: about 19 GB (fits);  H 60, L 121: about 7 GB (fits)
#   T4 parents H 124, L 249, B 7: about 49 GB: does NOT fit beside C's member (49 + 33 > 79 GiB with the 2 contexts); they run in a
#   window before C's sigma/a 2 launch or after the member ends (before HORIZON v3 takes the card), or beside C split into a site
#   process (about 29 GB) and a packet process (about 24 GB) with GAP10_T4_BIG=split (the analysis reads the 2 stems separately).
# Sharing the card slows C by the fraction of the time both compute (about 1.5 to 2.5 hours for everything that fits); the lead
# decides when to start. Optional tiers (GAP10_EXTRA_TIERS="T2 T1 T3 T7P"): T1 to T3 as gap10_matrix_v2.py prints them (nothing in
# the note rests on them); T7P = a parent on the record at [100] d 2 and d 8 (half-width 4.5 r_s + 16, L 293), the only parent
# control that would reach the gap note's decision rule at 3 r_s on [100], run as site + packet processes (about 47 and 37 GB).
# Outputs are not pushed (no key to the box): box_pull_gap10_rerun_v1.sh reads $ST over ssh, rsyncs each run's npz and json
# when its run line appears, verifies the md5s and touches <stem>.pulled here; with GAP10_PRUNE_PULLED=1 this script deletes
# pulled npz files when the disk falls under 10 GB (never anything unpulled).
# Start:  nohup setsid bash ${QI2_ROOT}/pod_gap10_matrix_v5.sh > ${QI2_ROOT}/matrix_v5_nohup.log 2>&1 &
set -u
R=${QI2_ROOT}/busi_local; G=$R/gap10; CODE=$R/code/gap10; REC=$R/records/record_slice.npz
ST=$G/matrix_status_v5.txt; LOGS=$G/logs_v5; OUT=$G/T7_rerun
MARGIN=${GAP10_MARGIN_GB:-6}; T4BIG=${GAP10_T4_BIG:-auto}; EXTRA=${GAP10_EXTRA_TIERS:-}; PRUNE=${GAP10_PRUNE_PULLED:-0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # the wavelength set's v4 setting; changes no arithmetic
mkdir -p $G $LOGS $OUT $G/runs; cd $G
stamp(){ date -u +%Y-%m-%dT%H:%M:%SZ; }
hms(){ date -u +%H:%M:%S; }
echo "v5 start $(stamp) margin=${MARGIN}GB t4big=$T4BIG extra='$EXTRA' prune=$PRUNE gpu=$(nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv,noheader 2>/dev/null | tr -d '\n')" >> $ST
for f in $REC $R/records/record_slice.json $CODE/gap10_run_v2.py $CODE/gap10_matrix_v2.py $CODE/gap10_core_v2.py; do
  [ -e "$f" ] || { echo "v5 ABORT missing $f $(stamp)" >> $ST; exit 2; }
done
GAP10_CODE=$CODE GAP10_RUNS=$G python $CODE/gap10_matrix_v2.py > $G/matrix_v5.sh 2>> $LOGS/matrix_gen.log \
  || { echo "v5 ABORT gap10_matrix_v2.py failed $(stamp)" >> $ST; exit 2; }
grep -q '/workspace' $G/matrix_v5.sh && { echo "v5 ABORT matrix_v5.sh names /workspace $(stamp)" >> $ST; exit 2; }

free_gb(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | awk '{printf "%.1f", $1/953.674}'; }
disk_gb(){ df --output=avail -BG / | tail -1 | tr -dc 0-9; }
# the estimate of the header: $1 rs $2 half_frac $3 margin $4 B $5 fft(0/1)
need_gb(){ awk -v rs="$1" -v hf="$2" -v mg="$3" -v B="$4" -v fft="$5" 'BEGIN{H=int(hf*rs+0.5)+mg; L=2*H+1; s=B*4*L*L*L*16/1e9; printf "%.1f", 6*s+(fft?s:0)+200*L*L*L/1e9+3}'; }
arg(){ echo "$1" | grep -o -- "--$2 [^ ]*" | head -1 | awk '{print $2}'; }
cmd_need(){ local c="$1"; local rs hf mg src tw dp B fft
  rs=$(arg "$c" rs); hf=$(arg "$c" half_frac); mg=$(arg "$c" margin); src=$(arg "$c" source); tw=$(arg "$c" inside_twins_max_depth); dp=$(arg "$c" depth)
  hf=${hf:-3.0}; mg=${mg:-16}; src=${src:-both}; B=7
  case $src in site) B=4;; packet) B=3;; esac
  if [ "$src" = both ] && [ "${tw:-4}" != "-1" ] && awk -v d="${dp:-4}" -v t="${tw:-4}" 'BEGIN{exit !(d<=t)}'; then B=10; fi
  fft=0; echo "$c" | grep -q -- '--fft_check_sweeps' && fft=1
  need_gb "$rs" "$hf" "$mg" "$B" "$fft"
}
cmd_label(){ echo "$1" | grep -o -- '--\(rs\|ma\|mode\|sampling\|direction\|depth\|half_frac\|solver\|source\|tag\) [^ ]*' | sed 's/^--//' | tr '\n' ' ' | sed 's/ $//'; }
agg_max(){ awk 'BEGIN{m=-1}{v=$1+0; if(v>m)m=v}END{if(m<0)print "na"; else printf "%.2e",m}'; }
pf(){ awk -v v="$1" -v t="$2" 'BEGIN{ if(v==""||v=="na"){print "na"} else if(v+0 < t+0){print "PASS"} else {print "FAIL"}}'; }
prune_pulled(){ [ "$PRUNE" = 1 ] || return 0; [ "$(disk_gb)" -lt 10 ] || return 0
  for f in $G/runs/*/*.npz; do [ -e "$f" ] || continue; [ -e "${f%.npz}.pulled" ] && { rm -f "$f"; echo "pruned $f (pulled, disk low) $(hms)" >> $ST; }; done; }

acceptance(){ # $1 tier $2 label $3 log -> 1 "acc" line on stdout
  local T="$1" L="$2" LOG="$3" rsf rel n1 rh kh cr dir ne ee cx f1 f2 ms mem ns vU vE vC vR vN vH vF ref
  rsf=$(sed -n 's/^\[record\].*rs_file = \([0-9.eE+-]*\) .*/\1/p' "$LOG" | head -1)
  rel=$(sed -n 's/^\[record\].*relative difference \([0-9.eE+-]*\)).*/\1/p' "$LOG" | head -1)
  n1=$(sed -n 's/^\[record\].*N_is_one=\([A-Za-z]*\).*/\1/p' "$LOG" | head -1)
  dir=$(sed -n 's/^\[profile\] allocated \([0-9]*\):.*/\1/p' "$LOG" | head -1)
  rh=$(sed -n 's/^\[profile\] allocated.*r_h=\([0-9.eEna+-]*\) .*/\1/p' "$LOG" | head -1)
  kh=$(sed -n 's/^\[profile\] allocated.*kappa_h=\([0-9.eEna+-]*\) .*/\1/p' "$LOG" | head -1)
  cr=$(sed -n 's/^\[profile\] allocated.*crossings (radius, inverse slope)=\(\[[^]]*\]\).*/\1/p' "$LOG" | head -1 | tr -d ' ')
  ne=$(grep -o '|norm-1|=[0-9.eE+-]*' "$LOG" | sed 's/.*=//' | agg_max)
  ee=$(grep -o '|E-E0|max=[0-9.eE+-]*' "$LOG" | sed 's/.*=//' | agg_max)
  cx=$(grep -o 'norm excess max over k: [0-9.eE+-]*' "$LOG" | sed 's/.*: //' | agg_max)
  f1=$(grep '^\[fft check\]' "$LOG" | sed -n '1s/.*= //p'); f2=$(grep '^\[fft check\]' "$LOG" | sed -n '2s/.*= //p')
  ms=$(sed -n 's/^\[sweep 1\/.*(\([0-9.]*\) ms\/application.*/\1/p' "$LOG" | head -1)
  mem=$(sed -n 's/^\[memory\].*= \([0-9.]*\) GB; free on [^:]*: \([0-9.]*\) GB.*/need \1 free \2/p' "$LOG" | head -1)
  ns=$(grep -c '^\[sweep ' "$LOG")
  vU=$(pf "$ne" 1e-12); vE=$(pf "$ee" 1e-10); vC=$(pf "$cx" 1e-11); vF=$(pf "$f1" 1e-11); vR=na; vN=na; vH=na
  if [ "$T" = T7 ]; then
    vR=$(pf "$rel" 1e-2)
    [ -n "$n1" ] && { [ "$n1" = False ] && vN=PASS || vN=FAIL; }
    ref=""; case "$dir" in 100) ref=22.2741;; 111) ref=22.2416;; esac
    [ -n "$rh" ] && [ -n "$ref" ] && vH=$(awk -v a="$rh" -v b="$ref" 'BEGIN{d=a-b; if(d<0)d=-d; print (d<0.01)?"PASS":"FAIL"}')
  fi
  echo "$T acc $L rs_file=${rsf:-na} reldiff=${rel:-na} N_is_one=${n1:-na} r_h=${rh:-na} kappa_h=${kh:-na} crossings=${cr:-na} sweeps=$ns norm_err_max=$ne energy_err_max=$ee cheb_excess_max=$cx fft1=${f1:-na} fft2=${f2:-na} ms_per_app=${ms:-na} ${mem:-need na free na} | unitarity=$vU energy=$vE cheb_excess=$vC rs_file=$vR N_is_one=$vN r_h=$vH fft1=$vF"
}

DEFERRED=0
run_one(){ # $1 tier $2 label $3 estimated need GB $4 command
  local T="$1" L="$2" N="$3" c="$4" F LOG RC P NA W M1 M2
  grep -qF -- "$T run $L exit 0 " $ST && return 0
  prune_pulled
  if [ "$(disk_gb)" -lt 5 ]; then echo "$T SKIPPED $L disk under 5 GB $(hms)" >> $ST; return 0; fi
  while pgrep -f 'gap10_run_v2(_wl)?\.py' > /dev/null; do sleep 60; done     # one process of this lane at a time
  F=$(free_gb)
  if awk -v f="$F" -v n="$N" -v m="$MARGIN" 'BEGIN{exit !(f+0 < n+m)}'; then
    echo "$T DEFERRED $L free ${F}GB below need ${N}GB + margin ${MARGIN}GB $(hms)" >> $ST; DEFERRED=$((DEFERRED+1)); return 0
  fi
  LOG=$LOGS/${T}_$(echo "$L" | tr -c 'A-Za-z0-9._-' '_').log
  echo "== $(stamp) free ${F}GB need ${N}GB :: $c" >> "$LOG"
  nice -n 10 bash -c "$c" >> "$LOG" 2>&1; RC=$?
  P=$(grep -o '^\[done\] [^ ]*' "$LOG" | tail -1 | awk '{print $2}'); P=${P:-none}
  NA=$(grep -o '^\[done\].*applications=[0-9]*' "$LOG" | tail -1 | sed 's/.*applications=//')
  W=$(grep -o 'wall=[0-9]*s$' "$LOG" | tail -1 | tr -dc 0-9)
  M1=none; M2=none
  [ -f "$P.npz" ] && M1=$(md5sum "$P.npz" | awk '{print $1}'); [ -f "$P.json" ] && M2=$(md5sum "$P.json" | awk '{print $1}')
  echo "$T run $L exit $RC path $P md5npz $M1 md5json $M2 napply ${NA:-na} wall ${W:-na}s $(hms)" >> $ST
  acceptance "$T" "$L" "$LOG" >> $ST
}

# ---- T7: the record, the 4 matched twins then the 4 fixed runs (the same driver, record file and arguments as v4; H 103, 20 sweeps)
B7="python $CODE/gap10_run_v2.py --rs 29 --ma 0.2 --ell 4 --Rb_frac 3.86 --profile file --profile_file $REC --sampling point --source both --out $G/runs/T7_record --inside_twins_max_depth -1"
N7=$(need_gb 29 3.0 16 7 0)
for PREP in matched fixed; do
  for DD in "100 2" "100 8" "111 2" "111 8"; do set -- $DD
    if [ $PREP = matched ]; then X="--prep matched --tag _matched"; else X="--prep fixed"; fi
    run_one T7 "$1 d$2 $PREP" "$N7" "$B7 --direction $1 --depth $2 $X"
  done
done
echo "T7 done $(stamp) deferred_so_far=$DEFERRED" >> $ST

# ---- tiers read from matrix_v5.sh (T6 with the Fourier check; T5; T4 with the parent ladder); the big T4 parents by GAP10_T4_BIG
tier_from_matrix(){ local T="$1" c L N hf c1 c2
  while IFS= read -r c; do
    L=$(cmd_label "$c"); N=$(cmd_need "$c"); hf=$(arg "$c" half_frac)
    if [ "$T" = T4 ] && [ "${hf:-3.0}" = "4.5" ] && [ "$T4BIG" = split ]; then
      c1=$(echo "$c" | sed 's/--source both/--source site/'); c2=$(echo "$c" | sed 's/--source both/--source packet/')
      run_one T4 "$(cmd_label "$c1")" "$(cmd_need "$c1")" "$c1"; run_one T4 "$(cmd_label "$c2")" "$(cmd_need "$c2")" "$c2"; continue
    fi
    run_one "$T" "$L" "$N" "$c"
  done < <(grep -v '^#' $G/matrix_v5.sh | grep gap10_run_v2 | grep -- "--out $G/runs/$T")
  echo "$T done $(stamp) deferred_so_far=$DEFERRED" >> $ST
}
for T in T6 T5 T4; do tier_from_matrix $T; done
for T in $EXTRA; do
  if [ "$T" = T7P ]; then   # the optional parent on the record: [100] d 2 and d 8 at half-width 4.5 r_s + 16, site + packet processes
    for DP in 2 8; do for S in site packet; do
      c="${B7/--source both/--source $S} --direction 100 --depth $DP --prep fixed --half_frac 4.5 --margin 16 --tag _P146"
      c="${c/--out $G\/runs\/T7_record/--out $G/runs/T7_parent}"
      run_one T7P "100 d$DP $S H146" "$(cmd_need "$c")" "$c"
    done; done
    echo "T7P done $(stamp) deferred_so_far=$DEFERRED" >> $ST
  else tier_from_matrix "$T"; fi
done

# ---- analysis: v3 if present (the wavelength set's), else v2 (the repairs of the interim's section 5), else v1; the status file names it
DIRS=""; for d in T7_record T6_flat T5_taylor T4_rs24 T4_box T7_parent T1 T2 T3_ell8 T3_Rb08 T3_cell T3_point; do [ -d $G/runs/$d ] && DIRS="$DIRS $G/runs/$d"; done
ARC=1
for A in gap10_analysis_v3.py gap10_analysis_v2.py gap10_analysis.py; do
  [ -f $CODE/$A ] || continue
  KC=""; [ "$A" = gap10_analysis_v3.py ] && KC="--kcsv $OUT/gap10_wavelength.csv"
  nice -n 10 python $CODE/$A $DIRS --csv $OUT/gap10_results.csv --tcsv $OUT/gap10_transmissivity.csv $KC --md $OUT/gap10_results.md --figs $OUT/figs > $OUT/analysis_$A.log 2>&1; ARC=$?
  echo "analysis $A exit $ARC $(hms)" >> $ST
  [ $ARC -eq 0 ] && break
done
if [ "$DEFERRED" -eq 0 ]; then echo "all done $(stamp)" >> $ST
else echo "pass done $(stamp) deferred=$DEFERRED (rerun this script in a window: runs logged exit 0 are skipped; then the analysis runs again)" >> $ST; fi
