#!/bin/bash
# waits for the fixture G0 (pid in g0pp9c4.pid) and the REF segment (seg_REF.pid), then the timed segment arms one at a time
cd ${QI2_ROOT}/H3_build
for f in g0pp9c4.pid seg_REF.pid; do p=$(cat $f); while kill -0 $p 2>/dev/null; do sleep 30; done; done
echo "timed chain start $(date -u +%FT%TZ) load $(cut -d" " -f1-3 /proc/loadavg)"
for arm in OFF A1 B8 AB8 OFF2; do echo "load before $arm: $(cut -d" " -f1-3 /proc/loadavg)"; bash ${QI2_ROOT}/run_h3seg_v9c4.sh $arm; done
echo "timed chain done $(date -u +%FT%TZ)"
