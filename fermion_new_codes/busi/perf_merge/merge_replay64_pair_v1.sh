#!/bin/bash
# merge_replay64_pair_v1.sh (busi, MERGE-1 self-test (c)): waits for the F1 fixture set's 8 arms to end (thread budget 8 on the box), then
# runs the P64_s4 replay's two arms concurrently on box GPU 1, THR 1 per process, code16e: G0 = the merged engine at its defaults,
# GA = --eigh banded --nk-cache-host on --fields collocation (every adopted option).
M=${QI2_ROOT}/MERGE_build; B=$M/build
{
  until [ "$(grep -h " end " $M/fixture/F1/*/arm.log 2>/dev/null | wc -l)" -ge 8 ]; do sleep 60; done
  echo "[$(date -u +%FT%TZ)] F1 ended; replay64 G0 and GA start" >> $M/replay64_pair.log
  GPU=1 THR=1 $B/merge_replay64_v1.sh G0 12.5 &
  GPU=1 THR=1 $B/merge_replay64_v1.sh GA 12.5 --eigh banded --nk-cache-host on --fields collocation &
  wait
  echo "[$(date -u +%FT%TZ)] replay64 G0 and GA done" >> $M/replay64_pair.log
  exit 0
}
