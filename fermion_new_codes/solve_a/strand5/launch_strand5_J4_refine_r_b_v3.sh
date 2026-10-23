#!/bin/bash
# Strand 5 sec. 10(r)(b), the second crossings of the (0.1, 0.5) local families and the (0.1, 1) kernel families located (builder's extension,
# stated: the second pass at z_c = 2.45 to 2.75 still had the second eigenvalue positive): z_c = 2.80 to 3.00 in steps of 0.02; stage 1 members
# and stage 2 version 10 with the full-node Hessian at every member. One at a time, 2 threads; waits for the J5 pilot's Gaussian run to finish
# (thread budget). Builder analyst.
set -u
D=${QI2_ROOT}/s5_star
cd $D || exit 1
O=J4refine4
mkdir -p $O
date -u '+queued %Y-%m-%d %H:%M:%S UTC' > $O/status.txt
until grep -q "gauss_h1 exit" J5pilot2/status.txt; do sleep 30; done
date -u '+start %Y-%m-%d %H:%M:%S UTC' >> $O/status.txt
K=${QI2_ROOT}/kernel_L256.json
PY=${QI2_ROOT}/python
ZL=$($PY -c "print(','.join('%.2f' % (2.80 + 0.02 * i) for i in range(11)))")
{ for op in local_ka0 local_klat; do for g in g0_0 g0.2_1; do echo "J4run2/lstar_m0.1_aH0.5_${op}_$g.json $ZL ref2ndc"; done; done
  for g in g0_0 g0.2_1; do echo "J4run2/lstar_m0.1_aH1_kernel_$g.json $ZL ref2ndc"; done; } > $O/list.txt
job() {  # family zcs suffix
  tag=$(basename $1 .json | sed 's/^lstar_//')
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 nice -n 10 $PY code/strand5_J4_members_v1.py --family $1 --kernel $K --zc $2 --suffix $3 --out $O > $O/log_members_${tag}_$3.txt 2>&1
  echo "members $tag $3 exit $? $(date -u +%H:%M:%S)" >> $O/status.txt
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 nice -n 10 $PY code/strand5_J4_stage2_v10.py --family $O/lstar_${tag}_$3.json --kernel $K --out $O --full-node all > $O/log_stage2_${tag}_$3.txt 2>&1
  echo "stage2 $tag $3 exit $? $(date -u +%H:%M:%S)" >> $O/status.txt
}
export -f job
export O K PY
cat $O/list.txt | xargs -P 1 -L 1 bash -c 'job $0 $1 $2'
date -u '+end %Y-%m-%d %H:%M:%S UTC' >> $O/status.txt
echo DONE >> $O/status.txt
