#!/bin/bash
# merge_fixture_chain_v1.sh (busi, MERGE-1): the fixture sets run one after the other (each set's arms concurrently), on code16e (the
# deployed dir, engine_fields_v16.py = v16e 83fe8b2e), THR 1, GPU 2. Read once at start: the whole loop is one bash compound command.
M=${QI2_ROOT}/MERGE_build; B=$M/build
{
  GPU=2 THR=1 CODE=code16e $B/merge_fixture_v1.sh F2a "Rf engine_collapse_v36nq_rd2.py --fields collocation" "Mf engine_collapse_v36nq_rd13mf.py --fields collocation"
  GPU=2 THR=1 CODE=code16e $B/merge_fixture_v1.sh F2b "Rw engine_collapse_v36nq_rd2.py --fields collocation --fields-warm-start on" "Mw engine_collapse_v36nq_rd13mf.py --fields collocation --fields-warm-start on"
  GPU=2 THR=1 CODE=code16e $B/merge_fixture_v1.sh F2c "Rs engine_collapse_v36nq_rd13.py --jvp sternheimer" "Ms engine_collapse_v36nq_rd13mf.py --jvp sternheimer"
  GPU=2 THR=1 CODE=code16e $B/merge_fixture_v1.sh F2d "Mbf engine_collapse_v36nq_rd13mf.py --eigh banded --fields collocation" "Mall engine_collapse_v36nq_rd13mf.py --eigh banded --nk-cache-host on --fields collocation"
  echo "[$(date -u +%FT%TZ)] chain F2a-F2d done" >> $M/fixture/chain.log
  exit 0
}
