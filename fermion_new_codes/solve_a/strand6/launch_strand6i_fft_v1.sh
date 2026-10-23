#!/bin/bash
# Strand 6 (i), spec (2) (SPEC_strand6_validation.md, final3 (i)): strand6i_kernel_fft_v1.py, the control rectangles by the quadrature and the
# grid route (L = 64 and 128 at H = 1 and 0.5) on strand 6's two-thread chain after (h); the grid route on the free GPU among 1 and 2
# (nvidia-smi at launch: utilization below 10 percent and under 2 GB in use; GPU 0 carries another lane's job), else the CPU.  Builder analyst.
D=${QI2_ROOT}/s6_val
mkdir -p $D/s6i_fft1
cd $D || exit 1
date -u '+queued %Y-%m-%d %H:%M:%S UTC' > s6i_fft1/status.txt
until grep -q DONE s6h_run1/status.txt 2>/dev/null; do sleep 30; done
DEV=cpu
for g in 1 2; do
  read util mem <<< "$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits -i $g | tr -d ',')"
  if [ "$util" -lt 10 ] && [ "$mem" -lt 2000 ]; then DEV=cuda:$g; break; fi
done
date -u "+start %Y-%m-%d %H:%M:%S UTC device $DEV" >> s6i_fft1/status.txt
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 nice -n 10 ${QI2_ROOT}/python code/strand6i_kernel_fft_v1.py \
  --device $DEV --md s6i_fft1/RESULT_s6i_fft_tables_v1.md --out-json s6i_fft1/s6i_fft.json > s6i_fft1/run.log 2>&1
echo "fft exit $? $(date -u +%H:%M:%S)" >> s6i_fft1/status.txt
echo DONE >> s6i_fft1/status.txt
