#!/bin/bash
# Strand 6 (b): split the remaining queue over two GPUs.  Stops the run2 launcher shell only (its running python process, order 4 L 2048,
# continues); GPU 1 then runs the remaining L = 2048 settings after that process exits; GPU 2 (free at launch) runs L = 4096 and the spacing
# trend.  Same output folder omega/run2; status in omega/run2/status_gpu1.txt and status_gpu2.txt.
set -u
D=${QI2_ROOT}/s6_validation
cd $D || exit 1
O=omega/run2
for p in $(ps -u jon -o pid=,args= | awk '/launch_strand6b_omega_v2.sh/ && !/awk/ {print $1}'); do kill $p 2>/dev/null && echo "stopped launcher $p" >> $O/status.txt; done
CUR=$(ps -u jon -o pid=,args= | awk '/strand6b_omega_v4.py/ && /--L 2048/ && /--order 4/ && !/awk/ {print $1}' | head -1)
echo "requeue $(date -u +%H:%M:%S): current process $CUR (order 4 L 2048 mult 16) left running" >> $O/status.txt
run() {  # gpu order aOmega L mult extra...
  g=$1; shift
  CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 nice -n 10 ${QI2_ROOT}/python code/strand6b_omega_v4.py --out $O --order $1 --aOmega $2 --L $3 --mult $4 --gpu $g "${@:5}" >> $O/run_gpu$g.log 2>&1
  echo "order $1 aOmega $2 L $3 mult $4 ${*:5} exit $? $(date -u +%H:%M:%S)" >> $O/status_gpu$g.txt
}
(
  if [ -n "$CUR" ]; then while kill -0 $CUR 2>/dev/null; do sleep 20; done; fi
  echo "order 4 L 2048 mult 16 process ended $(date -u +%H:%M:%S)" >> $O/status_gpu1.txt
  run 1 6 0.004 2048 16 --spec --oneblock
  for o in 4 6; do run 1 $o 0.004 2048 32 --oneblock; done
  echo DONE >> $O/status_gpu1.txt
) &
(
  for o in 4 6; do run 2 $o 0.004 4096 16 --oneblock; done
  for aO in 0.008 0.002; do
    for o in 4 6; do run 2 $o $aO 1024 16 --spec; run 2 $o $aO 2048 16 --oneblock; done
  done
  echo DONE >> $O/status_gpu2.txt
) &
wait
