#!/bin/bash
# run_init32k_mrrr_v6s.sh (busi lane, #309 item 3): the 32768-site initial + sector vacuum by the all-levels zheevr (MRRR) route, 16 threads, box CPU,
# written to page_verify/cache/B2_n32768_k1_T0_Phiplus_mrrr.pt; the same arguments as the running subset-route job (pid 2793944).
W=${QI2_ROOT}/page_verify; S=${QI2_ROOT}/code_d13; PY=${QI2_ROOT}/python; B2=${QI2_ROOT}/B2_history
cd $W
export CUDA_VISIBLE_DEVICES= PYTHONPATH=${QI2_ROOT}/code_v6:$S PAGE_CODE_V6=${QI2_ROOT}/code_v6 NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16
echo "init32k_mrrr launch $(date -u +%FT%TZ) $(md5sum $S/page_interior_init_mrrr_v6s.py | cut -c1-8)" >> status.txt
/usr/bin/time -v nice -n 10 $PY $S/page_interior_init_mrrr_v6s.py --record $B2/collapse_c19_rs64_A_R2_tau8_ell4_avg_B2n4096.json --fields $B2/collapse_c19_rs64_A_R2_tau8_ell4_avg_B2n4096_fields.npz --fields-key applied --nsites 32768 --kappas 1-1 --interp linear --interp-sub 8 --cheb-tol 1e-16 --cuts-rs 2,16,24,30,48 --R-cut-rs 24 --R-collar-rs 16 --join-rs 32,64 --T-end-rs 500 --hold-fields-from 100 --lean-interior --init-cache $W/cache/B2_n32768_k1_T0_Phiplus_mrrr.pt --device cpu --threads 16 --out runs/init32k_mrrr --tag B2_32k > logs/init32k_k1_mrrr.log 2>&1
rc=$?
echo "init32k_mrrr exit $rc $(date -u +%FT%TZ)" >> status.txt
