cd ${QI2_ROOT}/engine
for t in C3/primary_rs64_A_R2_tau8_v9 C3/c9_rs64_A_R2_tau8_ell8 C3/c11_rs32_A_R2_tau8_ell4 C3/c12_rs32_A_R2_tau8_ell4 C3/primary_rs64_A_R2_tau8_v13 C3/c13_rs64_A_R2_tau8_ell8 C3/primary_rs64_A_R2_tau8_v15 C3/c15_rs64_A_R2_tau8_ell8 H100/C3/primary_rs64_A_R2_tau8_v15_h100; do
  [ -f ${t%/*}/collapse_${t##*/}_fields.npz ] || continue
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 nice -n 10 ${QI2_ROOT}/python code/engine_relabel_v1.py --runs $t --out RELABEL >> RELABEL/relabel_v1_run2.log 2>&1 < /dev/null
done
echo DONE >> RELABEL/relabel_v1_run2.log
