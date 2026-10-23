#!/bin/bash
# Strand 2b item 6 (SPEC sec. 15, after the second reviewer B red team): pass 2's 45 R4 cases rerun with driver v13 (library v8: complete
# per-start records at the returned vector, the enclosure's skipped and failed counts with the zero-failure pass rule, the continuation's
# cluster gap and sigma_min per accepted step, the warm b-solve statuses, R_free by the closed form beside the differenced value), 12
# single-thread workers, into pass2v13/; and the cube's two enclosure cases ((2,0), m 0.1, (0.2, 1, 1), k 1 and 1e4) with driver v7
# (library v6: Newton never executes a rejected step; the same records), two single-thread processes, into c8encl_v7/.
set -u
D=${QI2_ROOT}/s2b_row
cd $D || exit 1
PY=${QI2_ROOT}/python
mkdir -p pass2v13 c8encl_v7
date -u '+start %Y-%m-%d %H:%M:%S UTC (R4 driver v13, 12 workers; cube driver v7, 2 processes)' > pass2v13/status.txt
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for k in 1 10000; do
  (nice -n 10 $PY code/strand2b_row_C8_v7.py --out c8encl_v7 --block 2,0 --m 0.1 --gM 0.2 --gB 1 --Lam 1 --k $k --enclosure --unique > c8encl_v7/case_k$k.log 2>&1; echo "cube enclosure k$k exit $? $(date -u +%H:%M:%S)" >> pass2v13/status.txt) &
done
cp pass2v12/cases.txt pass2v13/cases.txt
cat pass2v13/cases.txt | xargs -P 12 -L 1 bash -c 'nice -n 10 ${QI2_ROOT}/python code/strand2b_row_R4_v13.py --out pass2v13 --block $0 --m $1 --gM $2 --gB $3 --Lam $4 --k $5 >> pass2v13/run.log 2>&1; echo "$0 $1 $2 $3 $4 $5 exit $? $(date -u +%H:%M:%S)" >> pass2v13/status.txt'
echo "R4 cases done $(date -u +%H:%M:%S)" >> pass2v13/status.txt
wait
date -u '+end %Y-%m-%d %H:%M:%S UTC' >> pass2v13/status.txt
echo DONE >> pass2v13/status.txt
