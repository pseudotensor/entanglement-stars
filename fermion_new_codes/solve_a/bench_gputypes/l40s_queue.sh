#!/bin/bash
# l40s_queue.sh (solve bench_gputypes v1): the L40S arms after kit b, one at a time on GPU 0, NT 14 (inbox #199), the instance's own venvs
# (${QI2_ROOT}/v36d = the cards' ladder_env/venv_v36d stack, ${QI2_ROOT}/np1263 = busi_compC/venv_np1263). Waits for any running kit first.
W=${QI2_ROOT}/bench_gputypes; cd $W
while pgrep -f "bench_exec.py" > /dev/null; do sleep 30; done
echo "$(date -u +%FT%TZ) a start"; BENCH_PY=${QI2_ROOT}/python BENCH_TAG=l40s bash kits/a_rung2_v36q/run.sh 0 14; echo "$(date -u +%FT%TZ) a end rc $?"
echo "$(date -u +%FT%TZ) c start"; BENCH_PY=${QI2_ROOT}/python BENCH_TAG=l40s bash kits/c_twofield_v9c2/run.sh 0 14; echo "$(date -u +%FT%TZ) c end rc $?"
