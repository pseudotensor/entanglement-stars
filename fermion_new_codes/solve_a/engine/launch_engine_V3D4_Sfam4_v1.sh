#!/bin/bash
# GPU 0 sequence per the lead's sec. 12.8: V3D version 4 (engine_validate3d_v4.py: the smooth mass wall in both operators; L = 12, 16 about a
# cube-centre point and L = 13, 17 about a site; B = 0 and 0.3), then the (0.1, 1/4) family on engine_star_v4 (zero-temperature filling at
# fixed N_B on the fraction grid with bisection, ell 8, the ell-control 4, 8, 16, the closed counts).  One GPU, nice 10, 1 thread.
GPU=${1:-0}
cd ${QI2_ROOT}/engine || exit 1
mkdir -p V3D4 Sfam4
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for spec in "12 cube" "16 cube" "13 site" "17 site"; do set -- $spec; for B in 0 0.3; do
  tag=L$1_$2_B$B
  echo "V3D4 $tag start $(date -u +%H:%M:%S) UTC (gpu $GPU)" >> status.txt
  nice -n 10 ${QI2_ROOT}/python code/engine_validate3d_v4.py --L $1 --centre $2 --B $B --gpu $GPU --out V3D4 --tag $tag > V3D4/$tag.log 2>&1
  echo "V3D4 $tag exit $? $(date -u +%H:%M:%S)" >> status.txt
done; done
echo "S-fam4 (0.1, 1/4) ell 8 start $(date -u +%H:%M:%S) UTC (gpu $GPU)" >> status.txt
nice -n 10 ${QI2_ROOT}/python code/engine_star_v4.py --m 0.1 --g 0.25 --ell 8 --nsites 1536 --eos EOS/eos_v4_tables.npz \
  --ovpg OVPG/ovpg_families.json --ell-control 4,8,16 --closed-count --gpu $GPU --out Sfam4 --tag m0.1_g0.25_ell8 > Sfam4/m0.1_g0.25_ell8.out 2>&1
echo "S-fam4 (0.1, 1/4) exit $? $(date -u +%H:%M:%S) UTC" >> status.txt
