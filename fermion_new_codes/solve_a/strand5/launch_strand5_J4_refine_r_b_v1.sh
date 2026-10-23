#!/bin/bash
# Strand 5 sec. 10(r)(b): new stage-1 members (strand5_J4_members_v1.py) and stage 2 with the full-node Hessian at every member
# (strand5_J4_stage2_v10.py --full-node all). Refined maximum: the (0.1, 0.5) kernel families (0, 0) and (0.2, 1) at z_c = 0.8174, 0.8194 (builder's
# additions: a stage-1 scan put the parabola vertex at 0.8215 and 0.8219, below the specification's 0.8234) and 0.8214, 0.8234, 0.8254. Second
# crossing: the six (0.1, 0.5) families and the two (0.1, 1) kernel families at z_c = 2.06, 2.17, 2.29 (between 1.949609 and 2.418435, equal in
# log z_c, rounded). Two at a time, 2 threads each; builder analyst.
set -u
D=${QI2_ROOT}/s5_star
cd $D || exit 1
O=J4refine2
mkdir -p $O
date -u '+start %Y-%m-%d %H:%M:%S UTC' > $O/status.txt
K=${QI2_ROOT}/kernel_L256.json
PY=${QI2_ROOT}/python
{ for g in g0_0 g0.2_1; do echo "J4run2/lstar_m0.1_aH0.5_kernel_$g.json 0.8174,0.8194,0.8214,0.8234,0.8254 refmax"; done
  for op in kernel local_ka0 local_klat; do for g in g0_0 g0.2_1; do echo "J4run2/lstar_m0.1_aH0.5_${op}_$g.json 2.06,2.17,2.29 ref2nd"; done; done
  for g in g0_0 g0.2_1; do echo "J4run2/lstar_m0.1_aH1_kernel_$g.json 2.06,2.17,2.29 ref2nd"; done; } > $O/list.txt
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
