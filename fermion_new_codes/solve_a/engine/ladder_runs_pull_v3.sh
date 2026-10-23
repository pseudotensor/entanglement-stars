#!/bin/bash
# ladder_runs_pull_v3.sh: version 2 with the pod root as an optional 4th argument (default ${QI2_ROOT}/ladders; pod 1 runs on ${QI2_ROOT}/ladders). Version 2 follows.
# ladder_runs_pull_v2.sh: version 1 with the remote tar's exit status ignored for absent optional names (queue_*, policy_* exist on some
# cards only; version 1 then failed every pull) and the archive checked by tar -tf instead. Version 1 follows.
# ladder_runs_pull_v1.sh (busi lane as solve builder; LADDERS_spec_v1 section 6): the box's pull of a ladder card's run files and logs (not slices;
# the slices go by mirror_slices_v2.sh), every 30 min, by tar over ssh because the pods have no rsync: ${QI2_ROOT}/{runs,logs,STATUS,
# status_*.txt} into ${QI2_ROOT}/ladders<card>/. The engine rewrites its json and npz every 5 records, so a pull is at most 5
# records and 30 min behind. Never touches the pod's processes or files. Usage (box): setsid nohup bash ladder_runs_pull_v1.sh CARD HOST PORT &
CARD=${1:?card}; HOST=${2:?host}; PORT=${3:?port}; PR=${4:-${QI2_ROOT}/ladders}; D=${QI2_ROOT}/$CARD; mkdir -p $D
while true; do
  ssh -n -i ${QI2_ROOT}/id_rsa -o BatchMode=yes -o ConnectTimeout=30 -p $PORT $HOST "cd $PR && tar -cf - --exclude='*.part' runs logs STATUS status_*.txt queue_* policy_* 2>/dev/null; exit 0" > $D/.pull.tar.part 2>/dev/null; tar -tf $D/.pull.tar.part > /dev/null 2>&1 \
    && mv $D/.pull.tar.part $D/.pull.tar && tar --no-same-owner -C $D -xf $D/.pull.tar && rm -f "${D:?}/.pull.tar" \
    && echo "$(date -u +%FT%TZ) pulled $(du -sh $D/runs 2>/dev/null | cut -f1)" >> $D/pull.log \
    || echo "$(date -u +%FT%TZ) pull failed" >> $D/pull.log
  sleep 1800
done
