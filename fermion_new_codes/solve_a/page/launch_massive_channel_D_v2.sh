#!/bin/bash
# launch_massive_channel_D_v2.sh (builder analyst; version 1 with engine_massive_channel_D_v2.py and new output names; S1 D of the quasi-frozen ruling, brief lead/BRIEF_D_S11_builder.md): on the box CPU,
# nice 10, 16 threads. (1) engine_massive_channel_D_v2.py on the production's 10 stored slices (T/r_s 12 to 40) against its record json and
# fields npz; (2) the control: the complex128 slices T/r_s 16 and 28 with the columns rounded to complex64 (--round-c64), the floor that the
# five complex64 slices carry; (3) the tables composed. Run from ${QI2_ROOT}/engine. Writes Dmassive/Dv2_prod10.*,
# Dmassive/Dv2_prod10_c64ctrl.* and Dmassive/status_Dv2_prod10.txt.
set -u
cd ${QI2_ROOT}/engine || exit 2
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
PY=${QI2_ROOT}/python
RUN=H100b/C3prod19/c19_rs64_A_R2_tau8_ell4_avg_r11
SD=H100b/states/c19_rs64_A_R2_tau8_ell4_avg_r11
S=""
for t in 0012 0015 0016 0020 0024 0025 0028 0030 0035 0040; do S="$S${S:+,}$SD/orbitals_c19_rs64_A_R2_tau8_ell4_avg_r11_T$t.00.npz"; done
ST=Dmassive/status_Dv2_prod10.txt
echo "main start $(date -u '+%Y-%m-%d %H:%M:%S UTC') pid $$" >> $ST
nice -n 10 $PY code/engine_massive_channel_D_v2.py --run $RUN --slices "$S" --threads 16 --out-json Dmassive/Dv2_prod10.json > Dmassive/Dv2_prod10.log 2>&1
rc=$?
echo "main exit $rc $(date -u '+%Y-%m-%d %H:%M:%S UTC')" >> $ST
nice -n 10 $PY code/engine_massive_channel_D_v2.py --compose Dmassive/Dv2_prod10.json --md Dmassive/Dv2_prod10.md --out-json Dmassive/Dv2_prod10_summary.json >> Dmassive/Dv2_prod10.log 2>&1
echo "main compose exit $? $(date -u '+%Y-%m-%d %H:%M:%S UTC')" >> $ST
C="$SD/orbitals_c19_rs64_A_R2_tau8_ell4_avg_r11_T0016.00.npz,$SD/orbitals_c19_rs64_A_R2_tau8_ell4_avg_r11_T0028.00.npz"
echo "c64ctrl start $(date -u '+%Y-%m-%d %H:%M:%S UTC')" >> $ST
nice -n 10 $PY code/engine_massive_channel_D_v2.py --run $RUN --slices "$C" --round-c64 --threads 16 --out-json Dmassive/Dv2_prod10_c64ctrl.json > Dmassive/Dv2_prod10_c64ctrl.log 2>&1
echo "c64ctrl exit $? $(date -u '+%Y-%m-%d %H:%M:%S UTC')" >> $ST
nice -n 10 $PY code/engine_massive_channel_D_v2.py --compose Dmassive/Dv2_prod10_c64ctrl.json --md Dmassive/Dv2_prod10_c64ctrl.md --out-json Dmassive/Dv2_prod10_c64ctrl_summary.json >> Dmassive/Dv2_prod10_c64ctrl.log 2>&1
echo "c64ctrl compose exit $? $(date -u '+%Y-%m-%d %H:%M:%S UTC')" >> $ST
echo "done $(date -u '+%Y-%m-%d %H:%M:%S UTC')" >> $ST
