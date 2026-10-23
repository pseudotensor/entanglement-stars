#!/bin/bash
# run_s16A_arm_v2.sh (busi; B' s = 16 STAGE A, ONE ARM per call, with checkpoint and resume: inbox #458 / #459; ReviewA_bpchk2_v1 sec. 5,
# ReviewB_bpchk2_v1 sec. 3B). START ONLY AFTER busi POSTS THE BENCHMARK PASS.
#   ARM=1   F1 d0, 256 columns, to 1e2 L, dt DTP a             prep F1_s16_d0           port=sink2L(onset 2 L, eta_max 1/a)
#   ARM=2a  eta_max x 2                                        prep F1_s16_d0_eta2_bb   (eta_max 2/a; the base input basis)
#   ARM=2c  dt = DTP/2 control                                  prep F1_s16_d0
#   ARM=2b  onset 2.25 L, half-side 2.75 L                     prep F1_s16_d0_onset225_bb (the base input basis)
# Driver run_bprime_v5.py (= the benchmarked v4 + checkpoint/resume + labels; CPU-tested: v4 = v5 and resume = uninterrupted).
# Checkpoint: every clock and every CKPT_S seconds (default 900) into $OUT/<tag>_ckpt (two batch states of 4.4 GB + meta).
# Re-running the same command RESUMES from the last checkpoint (on the same or another card: copy $OUT/<tag>_ckpt there first).
# WIN=sup|exit runs one window (128 columns); WIN=both (default) the full 256.
# Usage: CUDA_VISIBLE_DEVICES=<g> ARM=<arm> DTP=<dt> WIN=<win> PY=<venv python> OUT=<dir> bash run_s16A_arm_v2.sh       (DRY=1 DEV=cpu: the CPU dry run)
set -u
KIT=${KIT:-$(cd "$(dirname "$0")" && pwd)}
PY=${PY:-python}; DEV=${DEV:-cuda:0}; OUT=${OUT:-$KIT/out}; BATCH=${BATCH:-128}; CKPT_S=${CKPT_S:-900}; CAP_H=${CAP_H:-6}; DRY=${DRY:-0}
ARM=${ARM:?set ARM to 1, 2a, 2c or 2b}
DTP=${DTP:-4}          # the production step set by the dt ladder; the 2c control runs at DTP/2
DTC=$(awk -v d=$DTP 'BEGIN{print d/2}')
case $ARM in
  1)  TAG=s16A_1_F1_d0_T100;        PREP=F1_s16_d0;             DT=$DTP; PT="port=sink2L(onset 2 L, sin^2 ramp 0.5 L, eta_max 1/a, vacuum Markov bath)";;
  2a) TAG=s16A_2a_F1_eta2_T100;     PREP=F1_s16_d0_eta2_bb;     DT=$DTP; PT="port=sink2L(onset 2 L, sin^2 ramp 0.5 L, eta_max 2/a, vacuum Markov bath)";;
  2c) TAG=s16A_2c_F1_dthalf_T100;   PREP=F1_s16_d0;             DT=$DTC; PT="port=sink2L(onset 2 L, sin^2 ramp 0.5 L, eta_max 1/a, vacuum Markov bath)";;
  2b) TAG=s16A_2b_F1_onset225_T100; PREP=F1_s16_d0_onset225_bb; DT=$DTP; PT="port=sink2L(onset 2.25 L, sin^2 ramp 0.5 L, eta_max 1/a, half-side 2.75 L, vacuum Markov bath)";;
  *) echo "unknown ARM $ARM"; exit 1;;
esac
WIN=${WIN:-both}       # both | sup | exit: the column subset (the owner-box arms run one window each; see split_prep_v1.py)
if [ "$WIN" != "both" ]; then PREP=${PREP}_${WIN}; TAG=${TAG}_${WIN}; fi
HT="supply_filled (H1): supply occupied, exit at the Killing filling; vacuum sink bath per channel; holes as conjugate particles; declared hypothesis, no massless preparation claimed"
mkdir -p "$OUT"; LOG="$OUT/$TAG.driver.log"
TMAX=100; EXTRA=""
if [ "$DRY" = "1" ]; then TMAX=1; EXTRA="--max-columns 4"; BATCH=2; CKPT_S=30; echo "DRY run (CPU, 4 columns, T = 1 L)" | tee -a "$LOG"; fi
echo "start $(date -u +%FT%TZ) ARM $ARM tag $TAG prep $PREP dt $DT batch $BATCH ckpt ${CKPT_S}s cap ${CAP_H}h CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}" | tee -a "$LOG"
cd "$KIT" && md5sum -c --quiet MD5SUMS >> "$LOG" 2>&1 || { echo "STOP: kit md5 check FAILED" | tee -a "$LOG"; exit 3; }
if [ "$DRY" != "1" ]; then nvidia-smi --query-gpu=index,name,memory.used --format=csv -i ${CUDA_VISIBLE_DEVICES:-0} | tee -a "$LOG"; fi
cd "$KIT/v3"
timeout $(( CAP_H * 3600 )) $PY run_bprime_v5.py --phase run --prep ../prep/${PREP}_prep.npz --device $DEV --card --T-max-over-L $TMAX --extra-T-over-L $TMAX \
   --dt $DT --col-batch $BATCH --cin supply_filled --threads 4 --ckpt-every-s $CKPT_S --resume --port-tag "$PT" --hyp-tag "$HT" \
   --out "$OUT" --tag $TAG $EXTRA >> "$OUT/$TAG.log" 2>&1
RC=$?
echo "$(date -u +%FT%TZ) end $TAG rc $RC" | tee -a "$LOG"
if [ $RC -eq 0 ]; then cp "$KIT/prep/stageA_manifest_v1.json" "$OUT/" 2>/dev/null; (cd "$OUT" && md5sum $TAG.json $TAG.npz > $TAG.MD5SUMS); fi
exit $RC
