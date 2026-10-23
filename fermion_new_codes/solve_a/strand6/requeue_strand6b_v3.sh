#!/bin/bash
# Strand 6 (b): split the spacing trend over GPUs 1 and 2.  Stops requeue_strand6b_v2.sh's shell (its running python, order 4 a Omega 0.008
# L 1024, continues); GPU 1 then runs the rest of a Omega 0.008 after that process; GPU 2 (free since order 6 L 4096 ended) runs a Omega
# 0.002.  Output omega/run2; status_gpu1.txt, status_gpu2.txt.
set -u
D=${QI2_ROOT}/s6_validation
cd $D || exit 1
O=omega/run2
for p in $(ps -u jon -o pid=,args= | awk '/requeue_strand6b_v2.sh/ && !/awk/ {print $1}'); do kill $p 2>/dev/null && echo "requeue v3 $(date -u +%H:%M:%S): stopped requeue v2 shell $p" >> $O/status_gpu1.txt; done
CUR=$(ps -u jon -o pid=,args= | awk '/strand6b_omega_v4.py/ && /--gpu 1/ && !/awk/ {print $1}' | head -1)
run() {  # gpu order aOmega L mult extra...
  g=$1; shift
  CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 nice -n 10 ${QI2_ROOT}/python code/strand6b_omega_v4.py --out $O --order $1 --aOmega $2 --L $3 --mult $4 --gpu $g "${@:5}" >> $O/run_gpu$g.log 2>&1
  echo "order $1 aOmega $2 L $3 mult $4 ${*:5} exit $? $(date -u +%H:%M:%S)" >> $O/status_gpu$g.txt
}
(
  if [ -n "$CUR" ]; then while kill -0 $CUR 2>/dev/null; do sleep 20; done; fi
  echo "order 4 aOmega 0.008 L 1024 mult 16 --spec process $CUR ended $(date -u +%H:%M:%S)" >> $O/status_gpu1.txt
  run 1 4 0.008 2048 16 --oneblock
  run 1 6 0.008 1024 16 --spec
  run 1 6 0.008 2048 16 --oneblock
  echo DONE3 >> $O/status_gpu1.txt
) &
(
  for o in 4 6; do run 2 $o 0.002 1024 16 --spec; run 2 $o 0.002 2048 16 --oneblock; done
  echo DONE3 >> $O/status_gpu2.txt
) &
wait
