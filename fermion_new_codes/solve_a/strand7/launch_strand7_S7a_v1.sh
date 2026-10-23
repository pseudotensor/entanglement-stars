#!/bin/bash
# Strand 7 S7a (SPEC_strand7_collapse.md sec. 3): strand7_shell_v1.py over the spec's settings as six single-thread jobs at a time on slot B's share
# of the thread budget (J5mem2 h = 0.5 waits for this chain), after J5decomp3; a 0.5 r_s smoke run first (the chain stops if it fails); then
# strand7_shell_report_v1.py.  Settings: (i) self at E_0 = 8 (r_s = 16 h), R_0 = 4 r_s, cases A and B (fields kept for S7b), with the C3 pairs
# (dt = 0.005 r_s; h = 0.5 with n doubled, w_r = 2 held); R_0 = 8 r_s, cases A and B, run to 60 r_s (the builder's extension: the infall from
# 8 r_s takes about 35 r_s; the first 40 r_s are the spec's run); the scaling runs E_0 = 4 and 16, case A; (iii) held, cases A and B (fields kept);
# (ii) frozen_schw at E_0 = 8 (A, B) and 4 (A), frozen_ext and frozen_full at E_0 = 8 (A), and frozen_schw at E_0 = 8 (A) by the run (i) stepper;
# C1 static checks at z_c = 0.3 and at J1's maximum (0.83385376); C5 vacuum.  Builder analyst.
D=${QI2_ROOT}/s7_collapse
cd $D || exit 1
mkdir -p S7a_run1 S7a_smoke
date -u '+queued %Y-%m-%d %H:%M:%S UTC' > S7a_run1/status.txt
until grep -q DONE ${QI2_ROOT}/status.txt; do sleep 30; done
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PY=${QI2_ROOT}/python
date -u '+start %Y-%m-%d %H:%M:%S UTC' >> S7a_run1/status.txt
nice -n 10 $PY code/strand7_shell_v1.py --run self --T-over-rs 0.5 --out S7a_smoke --tag smoke_selfA > S7a_smoke/smoke_selfA.log 2>&1
rc=$?
echo "smoke exit $rc $(date -u +%H:%M:%S)" >> S7a_run1/status.txt
if [ $rc -ne 0 ]; then echo "smoke failed: chain stopped" >> S7a_run1/status.txt; echo DONE >> S7a_run1/status.txt; exit 1; fi
run1() { tag=$1; shift; nice -n 10 $PY code/strand7_shell_v1.py "$@" --out S7a_run1 --tag $tag > S7a_run1/$tag.log 2>&1; echo "$tag exit $? $(date -u +%H:%M:%S)" >> S7a_run1/status.txt; }
export -f run1
cat > S7a_run1/jobs.txt <<'JOBS'
self_E8_R4_A_dt --run self --E0 8 --R0-over-rs 4 --case A --dt-over-rs 0.005
self_E8_R4_B_dt --run self --E0 8 --R0-over-rs 4 --case B --dt-over-rs 0.005
self_E8_R8_A --run self --E0 8 --R0-over-rs 8 --case A --T-over-rs 60
self_E8_R8_B --run self --E0 8 --R0-over-rs 8 --case B --T-over-rs 60
self_E8_R4_A --run self --E0 8 --R0-over-rs 4 --case A --save-fields 1
self_E8_R4_B --run self --E0 8 --R0-over-rs 4 --case B --save-fields 1
vacuum --run vacuum --E0 8
static_zc0.3 --run static_check --zc 0.3
static_zcmax --run static_check --zc 0.83385376
frozen_schw_E8_A --run frozen_schw --E0 8 --case A
frozen_schw_E8_B --run frozen_schw --E0 8 --case B
frozen_schw_E4_A --run frozen_schw --E0 4 --case A
frozen_ext_E8_A --run frozen_ext --E0 8 --case A
frozen_full_E8_A --run frozen_full --E0 8 --case A
frozen_schw_E8_A_verlet --run frozen_schw --E0 8 --case A --integrator verlet
self_E8_R4_A_h05 --run self --E0 8 --R0-over-rs 4 --case A --h 0.5
self_E8_R4_B_h05 --run self --E0 8 --R0-over-rs 4 --case B --h 0.5
self_E4_R4_A --run self --E0 4 --R0-over-rs 4 --case A
self_E16_R4_A --run self --E0 16 --R0-over-rs 4 --case A
held_E8_R4_A --run held --E0 8 --R0-over-rs 4 --case A --save-fields 1
held_E8_R4_B --run held --E0 8 --R0-over-rs 4 --case B --save-fields 1
JOBS
xargs -P 6 -L 1 bash -c 'run1 "$@"' _ < S7a_run1/jobs.txt
nice -n 10 $PY code/strand7_shell_report_v1.py --dir S7a_run1 --md S7a_run1/RESULT_S7a_tables_v1.md \
  --c3 "A dt=self_E8_R4_A:self_E8_R4_A_dt,A h=self_E8_R4_A:self_E8_R4_A_h05,B dt=self_E8_R4_B:self_E8_R4_B_dt,B h=self_E8_R4_B:self_E8_R4_B_h05" \
  > S7a_run1/report.log 2>&1
echo "report exit $? $(date -u +%H:%M:%S)" >> S7a_run1/status.txt
echo DONE >> S7a_run1/status.txt
