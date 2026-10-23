#!/bin/bash
# Free pod 1's disk quota (exhausted at 16:43 UTC, the Page pilot and the lane 4 member stopped writing): remove pod 1's copies of the two
# fixed-preparation ladder members' slice states, after their 16 files were checked md5-identical to the box's mirror
# (${QI2_ROOT}/states, 16:57 UTC); their consumers (the records chains, S8 version 2) ran on the box.
# Each directory is checked to hold exactly its 8 files before removal. Builder analyst.
cd ${QI2_ROOT}/states || exit 1
for d in c27_rs64_A_R2_tau8_ell4_prep4_avg c27_rs64_A_R2_tau8_ell8_prep4_avg; do
  n=$(find "$d" -type f | wc -l)
  [ "$n" -eq 8 ] || { echo "$d holds $n files, not 8: left in place"; continue; }
  rm -r -- "$d" && echo "removed $d (8 files; the box mirror holds the identical copy)"
done
echo "# pod 1's disk quota freed: the two ladder members' slice states removed (identical copies on the box, H100b/states/) $(date -u +%H:%M:%S) UTC" >> ${QI2_ROOT}/h100b_status.txt
du -sh ${QI2_ROOT}/states 2>/dev/null
