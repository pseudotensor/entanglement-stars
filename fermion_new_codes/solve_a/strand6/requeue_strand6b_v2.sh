#!/bin/bash
# Strand 6 (b): the spacing trend on GPU 1 (free after its queue; the two litellm service contexts there hold no compute): a Omega in
# {0.008, 0.002}, orders 4 and 6, L = 1024 with the spec grid and L = 2048 with one block, as the GPU 2 chain of requeue_strand6b_v1.sh
# would have run them after L = 4096 order 6 (that chain stopped by stop_requeue_gpu2_v1.sh).  Output omega/run2; status_gpu1.txt.
set -u
D=${QI2_ROOT}/s6_validation
cd $D || exit 1
O=omega/run2
run() {  # gpu order aOmega L mult extra...
  g=$1; shift
  CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 nice -n 10 ${QI2_ROOT}/python code/strand6b_omega_v4.py --out $O --order $1 --aOmega $2 --L $3 --mult $4 --gpu $g "${@:5}" >> $O/run_gpu$g.log 2>&1
  echo "order $1 aOmega $2 L $3 mult $4 ${*:5} exit $? $(date -u +%H:%M:%S)" >> $O/status_gpu$g.txt
}
echo "requeue v2 start $(date -u +%H:%M:%S): spacing trend on GPU 1" >> $O/status_gpu1.txt
for aO in 0.008 0.002; do
  for o in 4 6; do run 1 $o $aO 1024 16 --spec; run 1 $o $aO 2048 16 --oneblock; done
done
echo DONE2 >> $O/status_gpu1.txt
