#!/bin/bash
# ASYMPTOTICS_checks_v1.2 (engine_asymptotics_table_v4.py; engine spec 30.6 (7): row (6) when C-capture version 6 lands): version 1.1's
# inputs with C-capture version 6's md for row (6) and the goal file copied fresh (the inbox #7 rulings changed the goal's text). Run on the
# box from engine/ after H100b/CAP/capture_v6_prod19_last.md exists. Builder analyst.
cd ${QI2_ROOT}/engine || exit 1
[ -f H100b/CAP/capture_v6_prod19_last.md ] || { echo "C-capture version 6's md is not on this host yet"; exit 2; }
cp ${QI2_ROOT}/GOAL_lane-alt.md ASYM/GOAL_lane-alt_v1.2.md
P=${QI2_ROOT}/python
H="the production=H100b/C3prod/collapse_c17_rs64_A_R2_tau8_ell4_avg.json,H100b/C3prod19/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11.json|64|4|8|engine_collapse_v17 then v19;the tau 4 control=H100b/C3prod19/collapse_c19_rs64_A_R2_tau4_ell4_avg.json|64|4|4|engine_collapse_v19;the tau 2 member (to T/rs 21.5)=H100b/C3prod25/collapse_c25_rs64_A_R2_tau2_ell4_avg.json|64|4|2|engine_collapse_v25;the ell 8 record=H100b/C3prod24/collapse_c24_rs64_A_R2_tau8_ell8_avg.json|64|8|8|engine_collapse_v24;the ell 16 record=H100b/C3prod25/collapse_c25_rs64_A_R2_tau8_ell16_avg.json|64|16|8|engine_collapse_v25;the r_s 32 tau 2 member=H100b/C3set24/collapse_c24_rs32_A_R2_tau2_ell4_avg.json|32|4|2|engine_collapse_v24;the r_s 32 tau 1 member=H100b/C3set25/collapse_c25_rs32_A_R2_tau1_ell4_avg.json|32|4|1|engine_collapse_v25"
S="the production (ell 4, tau 8)=J3S/j3p_scope_v2_c19_ell4.json;the tau 4 control (ell 4)=J3S/j3p_scope_v2_c19_tau4.json;the ell 8 record (tau 8)=J3S/j3p_scope_v2_c24_ell8.json"
F="(0.2, 1/4)=Sfam6/star_m0.2_g0.25_ell8.json,Sfam6/star_m0.2_g0.25_ell8_ext.json,Sfam6/star_m0.2_g0.25_ell8_ext2.json;(0.1, 1/4)=Sfam5/star_m0.1_g0.25_ell8.json;(0.05, 1/4)=Sfam7/star_m0.05_g0.25_ell8.json;(0.2, 1/16)=Sfam6/star_m0.2_g0.0625_ell8.json,Sfam6/star_m0.2_g0.0625_ell8_ext.json;(0.1, 1/16)=Sfam6/star_m0.1_g0.0625_ell8.json,Sfam6/star_m0.1_g0.0625_ell8_ext.json;(0.1, 1/64)=Sfam8/star_star8box_m0.1_g0.015625_max.json"
OMP_NUM_THREADS=2 nice -n 10 $P code/engine_asymptotics_table_v4.py --title ASYMPTOTICS_checks_v1.2 --histories "$H" --scope "$S" --jpl ASYM/RESULT_JPL_v1.md --goal ASYM/GOAL_lane-alt_v1.2.md --families "$F" --ovpg3 OVPG3/ovpg3_v1.json --capture H100b/CAP/capture_v6_prod19_last.md --md ASYM/ASYMPTOTICS_checks_v1.2.md > ASYM/log_v4.log 2>&1
echo "exit $?"
