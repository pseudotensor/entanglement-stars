#!/bin/bash
# Inbox #36 (the owner's approval of reviewer A's quasi-frozen ruling): the v5 controls of 36.1 (4) are reduced to the baseline, point, ell8 and
# ramp, the remaining 8 cancelled. On pod 1 lane 1 the one still queued and not started, page_v5_ctrl32_n4096, is marked started so the
# runner skips it (no queue line edited); the running page_v5_ctrl32_Rb0_8 finishes as a record. Builder analyst.
cd ${QI2_ROOT}/engine || exit 1
grep -qxF page_v5_ctrl32_n4096 queue_lane1.txt.started && { echo "already marked"; exit 0; }
echo page_v5_ctrl32_n4096 >> queue_lane1.txt.started
echo "# page_v5_ctrl32_n4096 cancelled before its start (inbox #36: the controls reduced to baseline, point, ell8 and ramp); page_v5_ctrl32_Rb0_8, running, finishes as a record $(date -u +%H:%M:%S) UTC" >> h100b_status.txt
tail -1 queue_lane1.txt.started
