#!/bin/bash
# Strand S diagnostic D-a2 (SPEC sec. 9.1; strandS_dos_flow_v1.py): the flow chart's density of states on the prescribed PG background at
# r_s = 16, 32, 64 (R_m = r_s/4, R_max = 8 r_s), the massless packet and the light rectangle's masses at aM = 0.1.  One CPU thread, sequential,
# gated on the lane's busy processes (at most 15).  Label: prescribed flow (simpler version); PG radial; v declared.  Builder analyst.
cd ${QI2_ROOT}/sS_entropy || exit 1
mkdir -p Da2
export FLOW_CODE=${QI2_ROOT}/code OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for rs in 16 32 64; do
  until [ "$(bash ${QI2_ROOT}/lane_threads_v1.sh)" -le 15 ]; do sleep 30; done
  date -u "+Da2 rs$rs start %H:%M:%S UTC" >> Da2/status.txt
  nice -n 10 ${QI2_ROOT}/python code/strandS_dos_flow_v1.py --rs $rs --out-json Da2/dos_flow_rs$rs.json > Da2/run_rs$rs.log 2>&1
  echo "Da2 rs$rs exit $? $(date -u +%H:%M:%S)" >> Da2/status.txt
done
echo DONE >> Da2/status.txt
