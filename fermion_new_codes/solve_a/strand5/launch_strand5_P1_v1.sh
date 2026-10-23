#!/bin/bash
# Strand 5 production P1, phase 1 (SPEC_strand5_covariant_row.md sec. 3, P1 (1), (2), (6) and the controls; the production chain, 2 threads):
#   (a) new stage-1 members of J4run4's (0.01, 0.1) local_ka0 couplings (0, 0) family (the packet kernel at L = 1024, as J4run4 and J4s2run9b) at
#       z_c = z_max - 0.002, z_max, z_max + 0.002 with z_max = 0.833894 (RESULT_strand5_family.md revision 2, I2b: the refined maximum) and at
#       z_c = 2.8 and 2.9 (the second crossing), by strand5_J4_members_v1.py;
#   (b) stage 2 version 10 with the full matter-node basis at every member (--full-node all): on (a)'s members, on the 20-member family, and on its
#       local_klat twin (J4run2's (0.01, 0.1) local_klat (0, 0) family with the kernel J4s2run9 used).
# Phase 2 (strand5_P1_cert_v1.py: the crossing, the derivative, the tangent overlap, the margin table) follows on these records.
# Waits for P2 (the chain is sequential).  Builder analyst.
set -u
D=${QI2_ROOT}/s5_star
cd $D || exit 1
O=P1run1
mkdir -p $O
date -u '+queued %Y-%m-%d %H:%M:%S UTC' > $O/status.txt
until grep -q "^P2 exit" P2run1/status.txt 2>/dev/null; do sleep 20; done
date -u '+start %Y-%m-%d %H:%M:%S UTC' >> $O/status.txt
KP=$D/J4kernels_run1
P1024="1:$D/J4kernels_run2/packets_C_m0.01_L1024.json,-1:$KP/packets_C_m0.10049876_L256.json,-1:$KP/packets_C_m0.10049876_L256.json,1:$KP/packets_C_m0.14177447_L256.json"
PY=${QI2_ROOT}/python
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
FAM=J4run4/lstar_m0.01_aH0.1_local_ka0_g0_0_pk.json
nice -n 10 $PY code/strand5_J4_members_v1.py --family $FAM --packets $P1024 --zc 0.831894,0.833894,0.835894,2.8,2.9 --suffix p1new --out $O > $O/members.log 2>&1
echo "members exit $? $(date -u +%H:%M:%S)" >> $O/status.txt
nice -n 10 $PY code/strand5_J4_stage2_v10.py --family $O/lstar_m0.01_aH0.1_local_ka0_g0_0_pk_p1new.json --packets $P1024 --out $O --full-node all > $O/s2_new.log 2>&1
echo "stage2 new members exit $? $(date -u +%H:%M:%S)" >> $O/status.txt
nice -n 10 $PY code/strand5_J4_stage2_v10.py --family $FAM --packets $P1024 --out $O --full-node all > $O/s2_family.log 2>&1
echo "stage2 family exit $? $(date -u +%H:%M:%S)" >> $O/status.txt
nice -n 10 $PY code/strand5_J4_stage2_v10.py --family J4run2/lstar_m0.01_aH0.1_local_klat_g0_0.json --kernel $D/J2b_run1/kernel_J4_L256.json --out $O --full-node all > $O/s2_twin.log 2>&1
echo "stage2 twin exit $? $(date -u +%H:%M:%S)" >> $O/status.txt
date -u '+end %Y-%m-%d %H:%M:%S UTC' >> $O/status.txt
echo DONE >> $O/status.txt
