#!/bin/bash
# Stops the unguarded J3' waiter launch_obs_J3v3_primary_v3.sh (replaced by v4 with the history-length guard); exact PID by its arguments.
# Builder analyst.
for pid in $(ps -eo pid,args | awk '$2=="bash" && $3=="code/launch_obs_J3v3_primary_v3.sh" {print $1}'); do
  kill $pid && echo "stopped $pid: $(ps -o args= -p $pid 2>/dev/null) (launch_obs_J3v3_primary_v3.sh)"
done
