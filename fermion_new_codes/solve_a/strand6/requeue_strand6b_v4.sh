#!/bin/bash
# Strand 6 (b): the spacing trend at a Omega = 1/500 to L = 4096 (L = 2048 not converged there: the heaviest sea's correlation length is about
# 500 sites): order 4 on GPU 1 (free after requeue v3), order 6 on GPU 2 after its current process (order 6, a Omega 0.002, L 2048) ends.
# Output omega/run2.
set -u
D=${QI2_ROOT}/s6_validation
cd $D || exit 1
O=omega/run2
run() {  # gpu order aOmega L mult extra...
  g=$1; shift
  CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 nice -n 10 ${QI2_ROOT}/python code/strand6b_omega_v4.py --out $O --order $1 --aOmega $2 --L $3 --mult $4 --gpu $g "${@:5}" >> $O/run_gpu$g.log 2>&1
  echo "order $1 aOmega $2 L $3 mult $4 ${*:5} exit $? $(date -u +%H:%M:%S)" >> $O/status_gpu$g.txt
}
CUR=$(ps -u jon -o pid=,args= | awk '/strand6b_omega_v4.py/ && /--gpu 2/ && !/awk/ {print $1}' | head -1)
( run 1 4 0.002 4096 16 --oneblock; echo DONE4 >> $O/status_gpu1.txt ) &
( if [ -n "$CUR" ]; then while kill -0 $CUR 2>/dev/null; do sleep 20; done; fi; run 2 6 0.002 4096 16 --oneblock; echo DONE4 >> $O/status_gpu2.txt ) &
wait
