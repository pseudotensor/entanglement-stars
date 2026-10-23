#!/bin/bash
# launch_single_object_B5_orbitals_p2cpu_v1.sh (builder analyst; inbox #41's revised S4 part B, B.5): engine_single_object_B_v1.py --mode
# orbitals on the production's solved history (C3prod19/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11) with the columns of
# engine_single_object_B5_prep_v1.py (S4one/B5_orbitals_v1.npz, md5 478daafd6fac09ac3f3815a777b3db46: the state's own interior orbitals at
# T/r_s 12.125 and 15.0, kappa 1 and 2), propagated to T/r_s 40: the weight beyond R_t(t) and 2GE(t), the outward current through R_t(t),
# the launch spectrum on the launch record's frozen h. On pod 2's CPU (--gpu -1; 16 threads, nice 10), for the reason in
# launch_single_object_B_s8_p2cpu_v1.sh. Started directly by nohup. Output S4one/B_orbitals.json; status lines in S4one/status_B.txt.
cd ${QI2_ROOT}/engine || exit 1
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
S=S4one/status_B.txt
du=$(du -sm /workspace | tail -1 | cut -f1)
if [ "$du" -ge 40000 ]; then echo "B_orbitals skipped: /workspace $du MB $(date -u +%H:%M:%S) UTC" >> $S; exit 1; fi
echo "B_orbitals start $(date -u +%H:%M:%S) UTC (/workspace $du MB; CPU 16 threads)" >> $S
nice -n 10 python code/engine_single_object_B_v1.py --mode orbitals --gpu -1 --record C3prod19/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11.json \
    --fields C3prod19/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11_fields.npz --orbitals-npz S4one/B5_orbitals_v1.npz --kappas 1,2 --T-end-rs 40 \
    --out-json S4one/B_orbitals.json > S4one/log_B_orbitals.log 2>&1
echo "B_orbitals exit $? $(date -u +%H:%M:%S) UTC" >> $S
