#!/usr/bin/env bash
# o5_cube_runs_H103_v1.sh (busi lane): o5_cube_runs_v1.sh with only its H 103 half (about 6 GB each, beside A2gen_B2 on pod 5 under its guard); the H 232 half runs by o5_cube_runs_H232_v1.sh when the card is free. o5_cube_runs_v1.sh -- the 8 cubic propagations of O5 (GAP10_T7_final_v1.1 section 9): 1 column, 2 sites (R_c 20 and 14 a), 2 boxes
# (H 232 for the 10 r_s window; H 103 for the cheap 3 r_s check), 2 channels (kappa 1, 2), each the lifted column as the single member
# (B 1) on the record's held fields, read at the same shells and sweep ends as the chain with the certificate band, the fixed-cut
# current through the chain's cut and the lifted detector packets (plus GAP10's axis packets at the chain's k_det).
# Resumable: a run whose npz exists is skipped unless O5_FORCE is set. Exact command lines; the paths are pod 3's (override by environment).
set -u
CODE=${O5_CODE:-${QI2_ROOT}/code}                       # holds gap10_run_v2_o5.py beside gap10_core_opt2.py and gap10_sources_wl.py
REC=${O5_REC:-${QI2_ROOT}/record_slice.npz}     # md5 334ecb67f5ed85811159e35db35866fe (record_slice.json md5_written_npz)
LIFT=${O5_LIFT:-${QI2_ROOT}/lift}                           # o5_lift_v1.py's outputs
RUNS=${O5_RUNS:-${QI2_ROOT}/runs}
LOGS=${O5_LOGS:-${QI2_ROOT}/logs}
RH=${O5_RH:-22.250079960858976}                                     # the chain's cut: record_slice.json roots_on_table[1].r; o5_chain_export_v1.py prints the same as r_h
DEVICE=${O5_DEVICE:-auto}
export GAP10_OPT2_BACKEND=${GAP10_OPT2_BACKEND:-compiled}
mkdir -p "$LOGS"

run_one () {   # H TMAX_RS RC DEPTH KAPPA
  local H=$1 TMAX=$2 RC=$3 DEPTH=$4 KAPPA=$5
  local SRC="$LIFT/o5_lift_Rc${RC}_k${KAPPA}_H${H}_src.npz"
  local DET="$LIFT/o5_lift_Rc${RC}_k${KAPPA}_H${H}_det.npz"
  local STEM="rs29_m0.2_ell4_Rb3.86_point_file_radial_H${H}_100_d${DEPTH}_o5_Rc${RC}_k${KAPPA}_cheb"
  local OUT="$RUNS/H${H}"
  if [ -f "$OUT/$STEM.npz" ] && [ -z "${O5_FORCE:-}" ]; then echo "[skip] $OUT/$STEM.npz exists"; return 0; fi
  if [ ! -f "$SRC" ] || [ ! -f "$DET" ]; then echo "[missing] $SRC or $DET: run o5_lift_v1.py --H $H for this chain first"; return 2; fi
  mkdir -p "$OUT"
  python "$CODE/gap10_run_v2_o5.py" --rs 29 --ma 0.2 --ell 4 --Rb_frac 3.86 --profile file --profile_file "$REC" --sampling point \
    --direction 100 --depth "$DEPTH" --prep matched --H "$H" --tmax_rs "$TMAX" --tau_frac 1 --dt_read 0.5 --regions 1.5 2.0 3.0 \
    --wl_components raw --o5_source_npz "$SRC" --o5_det_npz "$DET" --o5_axis_k -1 --o5_current --o5_rh "$RH" \
    --o5_tag "_Rc${RC}_k${KAPPA}" --out "$OUT" --device "$DEVICE" 2>&1 | tee "$LOGS/o5_${STEM}.log"
  return "${PIPESTATUS[0]}"
}

# the cheap 3 r_s checks first (H 103: about 6 GB, under 2 minutes each), then the 10 r_s window (H 232: about 63 GB, about 4 minutes each)
for KAPPA in 1 2; do
  run_one 103 3 20 2 "$KAPPA"
  run_one 103 3 14 8 "$KAPPA"
done
echo "[o5 cube runs H103] done"
