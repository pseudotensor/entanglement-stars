#!/bin/bash
# stop_obs_ball_cut_all_v1.sh (builder analyst): inbox #2 of 2026-10-16 moved S6 (the three-dimensional ball cut, J4) to the busi lane.
# Stops this lane's S6 chains on the box: each PID is killed only when its command line still matches the expected launcher or script.
# The launchers first (so no next item starts), then their python children. No pattern kills.
cd ${QI2_ROOT}/J4BC 2>/dev/null || cd ${QI2_ROOT}/obs
log=stop_obs_ball_cut_all_v1.log
stop_one() {
  local pid=$1 want=$2
  local args
  args=$(ps -o args= -p "$pid" 2>/dev/null)
  if [ -z "$args" ]; then echo "$(date -u +%H:%M:%S) pid $pid gone" >> "$log"; return; fi
  case "$args" in
    *"$want"*) kill "$pid" && echo "$(date -u +%H:%M:%S) killed $pid ($want)" >> "$log" ;;
    *) echo "$(date -u +%H:%M:%S) pid $pid kept: args do not match $want: $args" >> "$log" ;;
  esac
}
stop_one 1761097 "launch_obs_ball_cut_ctrl_v2.sh"
stop_one 1761100 "launch_obs_ball_cut_ctrl_v2.sh"
stop_one 1787609 "launch_obs_ball_cut_v1.sh"
stop_one 1991945 "launch_obs_ball_cut_window_v2.sh"
sleep 1
stop_one 1761104 "balls.py measure --radius 5.99 --box 128 --method dense"
stop_one 2087105 "obs_ball_cut_v1.py run --R 40 --solver gram"
stop_one 2136302 "obs_ball_cut_window_v1.py --R 32 --box-ratio 6"
echo "$(date -u +%H:%M:%S) S6 stopped (inbox #2: S6 moved to the busi lane)" >> "$log"
cat "$log"
