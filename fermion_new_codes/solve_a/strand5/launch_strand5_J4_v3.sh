#!/bin/bash
# Strand 5 J4 stage 1, run 3 (spec sec. 10(g)): the (0.01, 0.1) families with the kernel built from per-packet symbols, the light
# packet's L = 512 symbol and the heavy packets' L = 256 symbols (J4kernels_run1), by strand5_lattice_star_v3.py --packets; operators
# kernel and local_ka0 (local_klat does not depend on the kernel; J4run2's stands); couplings (0, 0) and (0.2, 1) as two parallel
# chains (CPU, 2 threads each); R_c = 274 by sec. 10(e)'s rule (the summed kernel's |K_n|/|K_0| stays above 1e-10 up to n = 254, plus
# 20 sites); 20 members z_c in [0.05, 3].  Output s5_star/J4run3.
set -u
D=${QI2_ROOT}/s5_star
cd $D || exit 1
O=J4run3
mkdir -p $O
date -u '+start %Y-%m-%d %H:%M:%S UTC' > $O/status.txt
K=$D/J4kernels_run1
P="1:$K/packets_C_m0.01_L512.json,-1:$K/packets_C_m0.10049876_L256.json,-1:$K/packets_C_m0.10049876_L256.json,1:$K/packets_C_m0.14177447_L256.json"
run() {  # operator couplings
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 nice -n 10 ${QI2_ROOT}/python code/strand5_lattice_star_v3.py --out $O --packets $P --mlight 0.01 --aH 0.1 --Rc 274 --operator $1 --couplings $2 --zmin 0.05 --zmax 3 --nmembers 20 >> $O/run_g$2.log 2>&1
  echo "m 0.01 aH 0.1 $1 g $2 exit $? $(date -u +%H:%M:%S)" >> $O/status.txt
}
chain() { run kernel $1; run local_ka0 $1; }
chain 0,0 &
chain 0.2,1 &
wait
date -u '+end %Y-%m-%d %H:%M:%S UTC' >> $O/status.txt
echo DONE >> $O/status.txt
