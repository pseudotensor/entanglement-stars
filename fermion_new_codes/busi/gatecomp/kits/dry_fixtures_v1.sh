#!/bin/bash
# dry_fixtures_v1.sh (busi, GATECOMP kits, #400): prep-mode CPU dry runs (prep_dry_v1.py prep: the preparation, or the restart's state
# rebuilt from its slice, then a stop before the record loop; no evolution) of the kit's line SHAPES on the box's small fixtures.
#  Fresh: FXP_cv = the CV2048 line at r_s/a 32 on 512 sites (16 r_s), K 10, isotropic, tau 2; FXP_w = a W256 member's line at r_s/a 32 on
#  512 sites, K 14, the well, ell 8, prep-ell 4, tau 2.
#  Restarts from MERGE-1's fixture parent F1/M0 (merged engine defaults on code16d, 1024 sites, r_s/a 32, ell 2, K 10, tau 2, records 0 to
#  0.125, slice 0.125; read only): FX_tau (view tau=1), FX_plain (view mix=plain, plain block, ivp fields), FX_point (view
#  site_fields=dense), FX_tight (stop rule 100x tighter, no view), and FX_tau_noview (tau=1 against the parent itself: must be REFUSED).
set -u
K=${QI2_ROOT}/kits; CODE=${QI2_ROOT}/code16d; ENG=engine_collapse_v36nq_rd13mf.py
PY=${QI2_ROOT}/python; P=${QI2_ROOT}/M0
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 NUMPY_MADVISE_HUGEPAGE=0
F=$K/dry/fx; rm -rf $F; mkdir -p $F
B="--m 0.2 --g 0.25 --R0-over-rs 4 --reading A --reference R2 --half-offset 0 --collar-policy flag --support-rel 1e-14 --gpu 0 \
--stop-E 1e-7 --stop-fields 1e-6 --maxit 20 --field-rtol 1e-12 --field-atol 1e-16 --delta-rtol-every 10 --commutator-every 1 \
--streams 0 --m1-simpson-steps auto --c-LZ 1.0 --window-max 4 --s11-collars 8,16 --s11-stop 1 --field-kernel-cache --v36-fixes"
NK="--mix newton --mix-alpha 0.5 --nk-maxiter 30 --nk-inner 15 --nk-krylov 40 --nk-jvp exact --nk-line-search halving --nk-halvings 6 --nk-restarts 2 \
--nk-plain-fallback on --nk-host-release on --nk-set resolve --iter-cache drop --prep-memory lean --end-above 0.001"
PLAIN="--mix plain --mix-alpha 0.5 --iter-cache drop --prep-memory lean"
DENSE="--eigh dense --jvp eigenbasis --nk-cache-host off"
mk() { echo "$2" | tr -s ' ' > $F/$1.args; }
mk FXP_cv "$B --kappa-sea 10 --site-fields averaged --T-over-rs 13 --rs 32 --nsites 512 --ell 4 --prep-ell 4 --tau 2 --dust-set isotropic $NK --pair-workers 2 --out X --slice-dir X --tag fxp_cv --fields collocation $DENSE"
mk FXP_w "$B --kappa-sea 14 --site-fields averaged --T-over-rs 14 --rs 32 --nsites 512 --ell 8 --prep-ell 4 --tau 2 --dust-set well $NK --pair-workers 2 --out X --slice-dir X --tag fxp_w --fields collocation $DENSE"
SL=$P/states/orbitals_fx_M0_T0000.12.npz
FX="--kappa-sea 10 --T-over-rs 0.25 --rs 32 --nsites 1024 --ell 2 --prep-ell 4 --resume-slice $SL --segment-end-T-over-rs 0.15625"
$PY $K/code/resume_view_v1.py $P/fx_M0 $F/view_tau --set tau=1 --at 0.125 --reason "dry run: the tau/32 control's shape" > $F/view_tau.log 2>&1
$PY $K/code/resume_view_v1.py $P/fx_M0 $F/view_plain --set mix=plain --at 0.125 --reason "dry run: the plain-mix control's shape" > $F/view_plain.log 2>&1
$PY $K/code/resume_view_v1.py $P/fx_M0 $F/view_point --set site_fields=dense --at 0.125 --reason "dry run: the point-sampled control's shape" > $F/view_point.log 2>&1
mk FX_tau "$B --site-fields averaged $FX --tau 1 $NK --pair-workers 0 --resume-run $F/view_tau/fx_M0 --out X --slice-dir X --tag fx_tau --fields ivp $DENSE"
mk FX_plain "$B --site-fields averaged $FX --tau 2 $PLAIN --resume-run $F/view_plain/fx_M0 --out X --slice-dir X --tag fx_plain --fields ivp $DENSE"
mk FX_point "$B --site-fields dense $FX --tau 2 $NK --pair-workers 0 --resume-run $F/view_point/fx_M0 --out X --slice-dir X --tag fx_point --fields ivp $DENSE"
mk FX_tight "$(echo "$B" | sed 's/--stop-E 1e-7 --stop-fields 1e-6/--stop-E 1e-9 --stop-fields 1e-8/') --site-fields averaged $FX --tau 2 $NK --pair-workers 0 --resume-run $P/fx_M0 --out X --slice-dir X --tag fx_tight --fields ivp $DENSE"
mk FX_tau_noview "$B --site-fields averaged $FX --tau 1 $NK --pair-workers 0 --resume-run $P/fx_M0 --out X --slice-dir X --tag fx_tau_nv --fields ivp $DENSE"
for n in FXP_cv FXP_w FX_tau FX_plain FX_point FX_tight FX_tau_noview; do
  D=$F/$n; mkdir -p $D; t0=$(date +%s)
  nice -n 10 $PY $K/code/prep_dry_v1.py prep $CODE $ENG $F/$n.args $D > $D/prep.log 2>&1; rc=$?
  echo "$n prep rc $rc ($(( $(date +%s) - t0 )) s): $(grep -E 'PREP OK|version 34|initial data|SystemExit|Error|error' $D/prep.log | tail -2 | cut -c1-260)"
done
