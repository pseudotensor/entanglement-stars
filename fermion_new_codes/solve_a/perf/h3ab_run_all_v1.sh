#!/bin/bash
# solve 8a7b5274, PERF H3 A/B on h100-f: the arms one at a time on the one card, H3_0 first (busi ready_jobs_v3 sec. 35; bench_protocol 4.2).
# ABC is dropped when threadpoolctl is absent from the twofield venv (spec: install nothing). Then the reads (cmp_v9c3_runs.py --args-equal, h3ab_read_v1.py).
BASE=${QI2_ROOT}/busi_compC; cd $BASE; PY=${QI2_ROOT}/python
ARMS="H3_0 AC"; $PY -c "import threadpoolctl" 2>/dev/null && ARMS="H3_0 AC ABC"
echo "arms $ARMS $(date -u +%FT%TZ)"
for A in $ARMS; do bash $BASE/h3ab_arm_v1.sh $A; echo "arm $A exit $? $(date -u +%FT%TZ)"; done
R=$(ls -d $BASE/gate/H3AB_H3_0_* | tail -1); TAG=C64_s8_ell8_tau2_twofield_v4L
set -a; . $BASE/threads.env; set +a; export BUSI_ENGINE_DIR=$BASE/engine BUSI_PAGE_DIR=$BASE/engine CUDA_VISIBLE_DEVICES=""
for A in AC ABC; do D=$(ls -d $BASE/gate/H3AB_${A}_* 2>/dev/null | tail -1); [ -n "$D" ] || continue
  echo "== cmp $A"; $PY $BASE/code/cmp_v9c3_runs.py $R/$TAG $D/$TAG --b-log $D/segment_run.log --args-equal; echo "cmp exit $?"; done
$PY $BASE/h3ab_read_v1.py $BASE/H3AB_read_v1.json $R $(ls -d $BASE/gate/H3AB_AC_* $BASE/gate/H3AB_ABC_* 2>/dev/null | tail -2)
echo "ALL DONE $(date -u +%FT%TZ)"
