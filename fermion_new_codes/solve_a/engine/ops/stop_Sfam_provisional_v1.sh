#!/bin/bash
# Stop the provisional (0.1, 1/4) family on engine_star_v3 (launch_engine_Sfam_v1.sh): the lead's sec. 12 rulings (turn 19) specify the
# family on engine_star_v4 (the seed at the same N_B, the fraction grid with bisection, the flags, R_surf) and put V3D first on GPU 0; the
# finished members stay as the provisional record.  PIDs by ps/awk on the exact arguments, never pkill -f.
ps -eo pid,args | awk '/bash launch_engine_Sfam_v1.sh/ && !/awk/ {print $1}' | while read p; do kill "$p" && echo "killed launcher $p"; done
ps -eo pid,args | awk '/engine_star_v3.py --m 0.1 --g 0.25 --ell 8 --filling zeroT_NB --NB-list 200/ && !/awk/ {print $1}' | while read p; do kill "$p" && echo "killed $p"; done
echo "$(date -u +%H:%M:%S) UTC: the provisional (0.1, 1/4) family on engine_star_v3 stopped (members so far kept); V3D v4 then the v4 family on GPU 0" >> ${QI2_ROOT}/status.txt
