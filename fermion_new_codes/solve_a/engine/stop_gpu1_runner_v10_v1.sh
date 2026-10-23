#!/bin/bash
# Stop the GPU 1 runner for list_gpu1_v10.txt (PID 3953232) before it launches the v11 primary: the owner asked for the GPU jobs to be
# optimized for speed and memory first. Exact PID, args checked. Builder analyst.
P=3953232
A=$(ps -o args= -p $P 2>/dev/null)
case "$A" in
  "bash code/launch_engine_gpu_v4.sh 1 code/list_gpu1_v10.txt 3893958 none") kill $P && echo "stopped $P ($A)";;
  *) echo "PID $P args do not match (got: '$A'); nothing stopped";;
esac
