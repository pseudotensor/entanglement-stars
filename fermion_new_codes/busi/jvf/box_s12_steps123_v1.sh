#!/bin/bash
# S12 RUN_ORDER steps 1 to 3 on the box CPU (builder): self-test, structure dumps, main post-processing run.
PY=${QI2_ROOT}/python
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
RUN=${QI2_ROOT}/s12_v1; mkdir -p $RUN
cd ${QI2_ROOT}/jvf
E=${QI2_ROOT}/engine
VF10=$E/VF10/vacflow10_vf10_rs32_exact_m0_full_R2.json; VF9=$E/VF9/vacflow9_vf9_rs32_exact_m0_full_R2.json
VF3_32=$E/VF/vacflow_vf3_rs32_exact_m0.json; VF3_16=$E/VF/vacflow_vf3_rs16_exact_m0_R2.json; VF7=$E/VF/vacflow_vf7_rs32_exact_m0_R2aware.json
ls -l $VF10 $VF9 $VF3_32 $VF3_16 $VF7 > $RUN/inputs_ls.txt
nice -n 10 $PY s12_postprocess.py --selftest > $RUN/selftest.log 2>&1; echo "selftest exit $?" >> $RUN/status.txt
for f in $VF10 $VF9; do nice -n 10 $PY s12_postprocess.py --dump-keys $f --depth 2; done > $RUN/dump_vf.txt 2>&1
for f in $VF3_32 $VF3_16 $VF7; do nice -n 10 $PY s12_postprocess.py --dump-keys $f --depth 3; done > $RUN/dump_old.txt 2>&1
echo "dumps done" >> $RUN/status.txt
/usr/bin/time -v nice -n 10 $PY s12_postprocess.py --vf10 $VF10 --vf9 $VF9 --vf3-32 $VF3_32 --vf3-16 $VF3_16 --vf7 $VF7 --out $RUN > $RUN/main.log 2> $RUN/main_time.txt; echo "main exit $?" >> $RUN/status.txt
ls -l $E/engine_vacflow_v7.py $E/code/engine_vacflow_v7.py >> $RUN/status.txt 2>&1
