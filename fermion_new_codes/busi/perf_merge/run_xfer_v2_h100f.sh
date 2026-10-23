#!/bin/bash
# run_xfer_v2_h100f.sh (busi, MERGE-1; ReviewA_v16e_v1 sec. 4 (a)): xfer_bw_v2.py on h100-f with the GPU idle, under the shared lock,
# nvidia-smi (PCIe link, processes) logged before and after, lspci link status where readable.
W=${QI2_ROOT}/busi_merge; O=$W/out/xfer2; mkdir -p $O
exec 9>>${QI2_ROOT}/busi_gpu.lock; flock 9; echo "$(date -u +%FT%TZ) xfer2: lock held" >> $W/out/iso115_wait.log
while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do sleep 15; done
{ date -u +%FT%TZ; nvidia-smi; nvidia-smi -q | grep -A12 "PCI$"; nvidia-smi -q | grep -B2 -A30 "PCIe Generation"; nvidia-smi topo -m; } > $O/before.txt 2>&1
(lspci -vvv 2>/dev/null | grep -B3 -A40 -i nvidia | grep -i "LnkSta\|LnkCap\|Kernel driver\|NUMA" ) > $O/lspci.txt 2>&1
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu,pcie.link.gen.current,pcie.link.width.current --format=csv,noheader -lms 1000 > $O/smi.csv 2>&1 &
smi=$!
${QI2_ROOT}/python -B $W/xfer_bw_v2.py --reps 5 --out $O/xfer_bw_v2.json > $O/xfer_bw_v2.txt 2>&1
echo "rc $?" >> $O/xfer_bw_v2.txt
kill $smi
{ date -u +%FT%TZ; nvidia-smi; } > $O/after.txt 2>&1
flock -u 9; echo "$(date -u +%FT%TZ) xfer2: lock released" >> $W/out/iso115_wait.log
