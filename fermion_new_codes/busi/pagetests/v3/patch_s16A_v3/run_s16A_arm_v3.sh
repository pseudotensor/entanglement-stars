#!/bin/bash
# run_s16A_arm_v3.sh (busi; B' s = 16 stage A at the ladder's dt (inbox #466: dt 0.5 a), ONE ARM AND ONE WINDOW per call; a PATCH on top of
# the kit_s16A_v2 already unpacked on the card (its MD5SUMS are re-checked), so only this 10 kB patch is copied: the owner-box link moves
# about 4.3 MB/s). First call makes the per-window preps from the kit's prep_v5 preps with split_prep_v1.py (a column subset of the same
# orthonormal basis) and checks their basis bytes against the box's (md5 of U_sub: sup 37be42d2bf86, exit a1ed5f1c7e49).
#   ARM=1  F1 d0        prep F1_s16_d0              port=sink2L(onset 2 L, eta_max 1/a)
#   ARM=2a eta_max x 2  prep F1_s16_d0_eta2_bb      (eta_max 2/a)
#   ARM=2b onset 2.25 L prep F1_s16_d0_onset225_bb  (half-side 2.75 L)
#   ARM=2c dt = DTP/2   prep F1_s16_d0              (the step control)
# WIN=sup|exit|both (128, 128 or 256 columns); DTP (default 0.5); TMAX in L (default 100; the owner-box calls use 69.32 = the theta 0.01
# clock, so they finish before 20:45); checkpoint every CKPT_S s into $OUT/<tag>_ckpt; re-running resumes.
# Usage: cd <kit_s16A_v2 dir> && CUDA_VISIBLE_DEVICES=<g> ARM=<a> WIN=<w> DTP=0.5 TMAX=<t> PY=<venv python> OUT=<dir> bash <patch>/run_s16A_arm_v3.sh
set -u
KIT=${KIT:-$PWD}; PATCH=$(cd "$(dirname "$0")" && pwd)
PY=${PY:-python}; DEV=${DEV:-cuda:0}; OUT=${OUT:-$KIT/out}; BATCH=${BATCH:-128}; CKPT_S=${CKPT_S:-900}; CAP_H=${CAP_H:-12}; DRY=${DRY:-0}
ARM=${ARM:?}; WIN=${WIN:-both}; DTP=${DTP:-0.5}; TMAX=${TMAX:-100}
DTC=$(awk -v d=$DTP 'BEGIN{print d/2}')
case $ARM in
  1)  TAG=s16A_1_F1_d0;        PREP=F1_s16_d0;             DT=$DTP; PT="port=sink2L(onset 2 L, sin^2 ramp 0.5 L, eta_max 1/a, vacuum Markov bath)";;
  2a) TAG=s16A_2a_F1_eta2;     PREP=F1_s16_d0_eta2_bb;     DT=$DTP; PT="port=sink2L(onset 2 L, sin^2 ramp 0.5 L, eta_max 2/a, vacuum Markov bath)";;
  2b) TAG=s16A_2b_F1_onset225; PREP=F1_s16_d0_onset225_bb; DT=$DTP; PT="port=sink2L(onset 2.25 L, sin^2 ramp 0.5 L, eta_max 1/a, half-side 2.75 L, vacuum Markov bath)";;
  2c) TAG=s16A_2c_F1_dthalf;   PREP=F1_s16_d0;             DT=$DTC; PT="port=sink2L(onset 2 L, sin^2 ramp 0.5 L, eta_max 1/a, vacuum Markov bath)";;
  *) echo "unknown ARM"; exit 1;;
esac
TAG=${TAG}_dt${DT}_T${TMAX}_${WIN}
HT="supply_filled (H1): supply occupied, exit at the Killing filling; vacuum sink bath per channel; holes as conjugate particles; declared hypothesis, no massless preparation claimed"
EXTRA=""; if [ "$DRY" = "1" ]; then TMAX=1; EXTRA="--max-columns 4"; BATCH=2; CKPT_S=30; fi
mkdir -p "$OUT"; LOG="$OUT/$TAG.driver.log"
echo "start $(date -u +%FT%TZ) ARM $ARM WIN $WIN dt $DT T_max ${TMAX} L tag $TAG CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}" | tee -a "$LOG"
(cd "$KIT" && md5sum -c --quiet MD5SUMS) >> "$LOG" 2>&1 || { echo "STOP: kit md5 check FAILED" | tee -a "$LOG"; exit 3; }
(cd "$PATCH" && md5sum -c --quiet PATCH_MD5SUMS) >> "$LOG" 2>&1 || { echo "STOP: patch md5 check FAILED" | tee -a "$LOG"; exit 3; }
P=$KIT/prep/${PREP}_prep.npz
if [ "$WIN" != "both" ]; then
  P=$KIT/prep/${PREP}_${WIN}_prep.npz
  [ -f "$P" ] || $PY "$PATCH/split_prep_v1.py" "$KIT/prep/${PREP}_prep.npz" >> "$LOG" 2>&1
  WANT=$([ "$WIN" = "sup" ] && echo 37be42d2bf86 || echo a1ed5f1c7e49)
  GOT=$($PY -c "import numpy as np,hashlib,sys; print(hashlib.md5(np.ascontiguousarray(np.load(sys.argv[1])['U_sub']).tobytes()).hexdigest()[:12])" "$P")
  [ "$GOT" = "$WANT" ] || { echo "STOP: split basis md5 $GOT != $WANT" | tee -a "$LOG"; exit 4; }
fi
cd "$KIT/v3"
timeout $(( CAP_H * 3600 )) $PY run_bprime_v5.py --phase run --prep "$P" --device $DEV --card --T-max-over-L $TMAX --extra-T-over-L $TMAX \
   --dt $DT --col-batch $BATCH --cin supply_filled --threads 4 --ckpt-every-s $CKPT_S --resume --port-tag "$PT" --hyp-tag "$HT" \
   --out "$OUT" --tag $TAG $EXTRA >> "$OUT/$TAG.log" 2>&1
RC=$?
echo "$(date -u +%FT%TZ) end $TAG rc $RC" | tee -a "$LOG"
if [ $RC -eq 0 ]; then cp "$KIT/prep/stageA_manifest_v1.json" "$OUT/" 2>/dev/null; (cd "$OUT" && md5sum $TAG.json $TAG.npz > $TAG.MD5SUMS); fi
exit $RC
