#!/bin/bash
# box_j4m_a4_cases_v1.sh (busi lane, J4 massive's A_4 follow-up; J4_massive_judgment_v1.1 (c)): the 80 massless boxes that give every
# radius one rung set 6R, 8R, 10R, 12R (L 120, 144 at R 12; 160 at R 16; 200, 240 at R 20; 240, 288 at R 24; 168, 280, 336 at R 28), on the
# production's balls (n2max and offset k from the records), mu 0, kind ladder2, into the same records folder (the done-key skips anything
# complete). 4 workers of 4 threads (16 in all), nice 10; the L 288 and 336 kernel builds need 3 to 5 GB each. Resumable: rerun.
B=${QI2_ROOT}/j4m_v1; C=${QI2_ROOT}/j4m; PY=${QI2_ROOT}/python
echo "a4 cases start $(date -u +%FT%TZ) load $(cut -d' ' -f1 /proc/loadavg)" >> $B/status.txt
cd $C || exit 1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
xargs -P 4 -L 1 sh -c 'nice -n 10 '"$PY"' -B j4m_case_v1.py --n2max $1 --L $3 --mu 0 --window-centre $0 --offset $2 --kind ladder2 --outdir '"$B"'/records_box --cache-dir '"$B"'/cache_box --threads 4 >> '"$B"'/logs/a4_cases.log 2>&1; echo "R $0 n $1 k $2 L $3 exit $?" >> '"$B"'/logs/a4_cases_status.txt' < $B/a4_cases_v1.txt
echo "a4 cases end $(date -u +%FT%TZ) ok $(grep -c 'exit 0' $B/logs/a4_cases_status.txt) of $(wc -l < $B/a4_cases_v1.txt)" >> $B/status.txt
