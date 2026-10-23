#!/bin/bash
# Strand 5 sec. 10(r)(b), the second crossings located (builder's extension, stated: the first pass's members at z_c = 2.06, 2.17, 2.29 did not
# bracket them): the (0.1, 0.5) kernel families at z_c = 2.290 to 2.330 in steps of 0.005, the (0.1, 0.5) local families and the (0.1, 1) kernel
# families at 2.45 to 2.75 in steps of 0.025; stage 1 members (strand5_J4_members_v1.py) and stage 2 version 10 with the full-node Hessian at
# every member. Two at a time, 2 threads each; builder analyst.
set -u
D=${QI2_ROOT}/s5_star
cd $D || exit 1
O=J4refine3
mkdir -p $O
date -u '+start %Y-%m-%d %H:%M:%S UTC' > $O/status.txt
K=${QI2_ROOT}/kernel_L256.json
PY=${QI2_ROOT}/python
ZK=$($PY -c "print(','.join('%.3f' % (2.29 + 0.005 * i) for i in range(9)))")
ZL=$($PY -c "print(','.join('%.3f' % (2.45 + 0.025 * i) for i in range(13)))")
{ for g in g0_0 g0.2_1; do echo "J4run2/lstar_m0.1_aH0.5_kernel_$g.json $ZK ref2ndb"; done
  for op in local_ka0 local_klat; do for g in g0_0 g0.2_1; do echo "J4run2/lstar_m0.1_aH0.5_${op}_$g.json $ZL ref2ndb"; done; done
  for g in g0_0 g0.2_1; do echo "J4run2/lstar_m0.1_aH1_kernel_$g.json $ZL ref2ndb"; done; } > $O/list.txt
job() {  # family zcs suffix
  tag=$(basename $1 .json | sed 's/^lstar_//')
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 nice -n 10 $PY code/strand5_J4_members_v1.py --family $1 --kernel $K --zc $2 --suffix $3 --out $O > $O/log_members_${tag}_$3.txt 2>&1
  echo "members $tag $3 exit $? $(date -u +%H:%M:%S)" >> $O/status.txt
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 nice -n 10 $PY code/strand5_J4_stage2_v10.py --family $O/lstar_${tag}_$3.json --kernel $K --out $O --full-node all > $O/log_stage2_${tag}_$3.txt 2>&1
  echo "stage2 $tag $3 exit $? $(date -u +%H:%M:%S)" >> $O/status.txt
}
export -f job
export O K PY
cat $O/list.txt | xargs -P 2 -L 1 bash -c 'job $0 $1 $2'
date -u '+end %Y-%m-%d %H:%M:%S UTC' >> $O/status.txt
echo DONE >> $O/status.txt
