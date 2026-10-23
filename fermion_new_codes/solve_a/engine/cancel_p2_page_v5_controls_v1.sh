#!/bin/bash
# Inbox #36: the v5 controls reduced to the baseline, point, ell8 and ramp, the remaining 8 cancelled. On the second H100's p2lane2 the five
# queued and not started (kappa3, coll6, coll4, taus2, cut48) are marked started so the runner skips them (no queue line edited); the running
# p2_page_v5_ctrl32_Rb08 finishes as a record. Builder analyst.
cd ${QI2_ROOT}/engine || exit 1
for t in p2_page_v5_ctrl32_kappa3 p2_page_v5_ctrl32_coll6 p2_page_v5_ctrl32_coll4 p2_page_v5_ctrl32_taus2 p2_page_v5_ctrl32_cut48; do
  grep -qxF "$t" queue_p2b.txt.started || echo "$t" >> queue_p2b.txt.started
done
echo "# p2_page_v5_ctrl32_kappa3, coll6, coll4, taus2, cut48 cancelled before their start (inbox #36); p2_page_v5_ctrl32_Rb08, running, finishes as a record $(date -u +%H:%M:%S) UTC" >> h100b_status.txt
tail -5 queue_p2b.txt.started
