#!/bin/bash
# engine_capture_v10's box-CPU tests before any pod run (engine spec 38.5 (3); the brief: "build and test the port on the box CPU first"):
# (1) the regression: version 9 and version 10 --no-outer-port on version 9's smoke configuration (slice 302, --control --absorber,
# --kappa-cap 3, reading times to 8 r_s, --workers 2), then version 10 with the port on the same configuration, one after the other;
# (2) test_capture_v10_port_v1 per probe (omega/m 1.1 with the propagator timing, 1.25, 1.5, 2), four single-threaded processes in
# parallel. At most 7 busy cores, nice 10; outputs CAP10TEST/. Builder analyst.
cd ${QI2_ROOT}/engine || exit 1
mkdir -p CAP10TEST
PY=${QI2_ROOT}/python
F="--fields H100b/C3prod19/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11_fields.npz --records H100b/C3prod19/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11.json --fields-key applied"
S="--slice 302 --control --absorber --workers 2 --kappa-cap 3 --T-max-over-rs 8"
echo "box tests start $(date -u +%H:%M:%S) UTC" > CAP10TEST/status.txt
(
  OMP_NUM_THREADS=2 nice -n 10 $PY code/engine_capture_v9.py $F $S --out-json CAP10TEST/smoke_v9_box.json --md CAP10TEST/smoke_v9_box.md > CAP10TEST/smoke_v9_box.log 2>&1
  echo "smoke_v9_box exit $? $(date -u +%H:%M:%S) UTC" >> CAP10TEST/status.txt
  OMP_NUM_THREADS=2 nice -n 10 $PY code/engine_capture_v10.py $F $S --no-outer-port --out-json CAP10TEST/smoke_v10_noport_box.json --md CAP10TEST/smoke_v10_noport_box.md > CAP10TEST/smoke_v10_noport_box.log 2>&1
  echo "smoke_v10_noport_box exit $? $(date -u +%H:%M:%S) UTC" >> CAP10TEST/status.txt
  OMP_NUM_THREADS=2 nice -n 10 $PY code/engine_capture_v10.py $F $S --out-json CAP10TEST/smoke_v10_port_box.json --md CAP10TEST/smoke_v10_port_box.md > CAP10TEST/smoke_v10_port_box.log 2>&1
  echo "smoke_v10_port_box exit $? $(date -u +%H:%M:%S) UTC" >> CAP10TEST/status.txt
) &
for x in 1.1 1.25 1.5 2; do
  if [ "$x" = "1.1" ]; then TK="--timing-kappas 1 50 100 200"; else TK="--timing-kappas"; fi
  ( OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 10 $PY code/test_capture_v10_port_v1.py --fields H100b/C3prod19/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11_fields.npz \
      --omega-over-m $x $TK --json CAP10TEST/port_test_om$x.json > CAP10TEST/port_test_om$x.log 2>&1
    echo "port_test_om$x exit $? $(date -u +%H:%M:%S) UTC" >> CAP10TEST/status.txt ) &
done
wait
echo "box tests end $(date -u +%H:%M:%S) UTC" >> CAP10TEST/status.txt
