#!/bin/bash
# compose_page_v2_final_v1.sh (builder analyst, 2026-10-17; engine spec 39.1 (1), (3), (4)): PAGE_solved_history version 2 from the landed runs,
# in one pass, local and light (json reading and one matplotlib figure; no solver): the filtered fetch (fetch_page_v6_v6.sh, no mutual
# information), the composer page_solved_history_report_v5.py (its draft label "A to T/r_s 40 with its fixed-field controls and variants;
# the extension to 100 pending"), the figure item7_fig_page_draft_v4.py with version 2's marks, and FILE 2's sentence filled from version 2's
# json by fill_item7_sentence_v1.py (it prints only after the icloud lane's R2 has read version 2; the lane delivers). Every output's md5
# printed. It stops when a variant has not landed final (s4far, s4loc3, s4k3, s4cubic), when C or s4 is not final, or when the sec. 4
# words appear in the composed text. Usage (from the worktree's solve/): bash code/page/compose_page_v2_final_v1.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
bash code/page/fetch_page_v6_v6.sh > /dev/null
python3 -I - <<'EOF'
import json, sys
base = "results/page/v6bA_runs/"
need = {"s4": "v6bA_s4", "s4tr": "v6eA_s4tr", "s4far": "v6dA_s4far", "s4loc3": "v6dA_s4loc3", "s4k3": "v6fA_s4k3", "s4cubic": "v6fA_s4cubic",
        "c15": "v6eA_c15", "c30": "v6eA_c30", "s8": "v6fA_s8"}
bad = []
for n, t in need.items():
    try:
        d = json.load(open(base + "%s/%s_summary.json" % (t, t)))
        if not d.get("final"):
            bad.append("%s not final" % n)
    except FileNotFoundError:
        bad.append("%s not landed" % n)
if bad:
    sys.exit("version 2 waits: " + "; ".join(bad))
print("every run of version 2 landed final")
EOF
R=results/page
python3 code/page/page_solved_history_report_v5.py $R $R/PAGE_solved_history_v2.md $R/PAGE_solved_history_v2_summary.json
python3 code/page/item7_fig_page_draft_v4.py $R/PAGE_solved_history_v2_summary.json $R/ITEM7_fig_page_draft_v2.png $R/ITEM7_fig_page_draft_v2_SOURCES.txt --marks v2
python3 code/page/fill_item7_sentence_v1.py $R/PAGE_solved_history_v2_summary.json ${QI2_ROOT}/ITEM7_solved_history_sentence_v1_template.md $R/ITEM7_solved_history_sentence_v1.md
for f in $R/PAGE_solved_history_v2.md $R/ITEM7_fig_page_draft_v2_SOURCES.txt $R/ITEM7_solved_history_sentence_v1.md; do
  if grep -qiE "frozen|freeze|prediction|dark|defect" "$f"; then echo "a sec. 4 word in $f:"; grep -noiE ".{40}(frozen|freeze|prediction|dark|defect).{40}" "$f" | head; exit 4; fi
done
md5sum $R/PAGE_solved_history_v2.md $R/PAGE_solved_history_v2_summary.json $R/ITEM7_fig_page_draft_v2.png $R/ITEM7_fig_page_draft_v2_SOURCES.txt $R/ITEM7_solved_history_sentence_v1.md
