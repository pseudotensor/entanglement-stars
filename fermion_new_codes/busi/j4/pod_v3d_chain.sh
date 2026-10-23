#!/bin/bash
# J4 version 2, RUN_ORDER_v3d sections 5 to 7 on the pod CPU (builder): the gate-D fit when stage D ends, the level-5
# preview 2.5 hours into stage E, the final fit when stage E ends. Reads the stages' status file only; touches no stage.
cd ${QI2_ROOT}/j4
J4=${QI2_ROOT}/j4; ST=$J4/v3d_chain_status.txt; SS=$J4/v2_stageDE_status.txt
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
D="--dirs $J4/prod $J4/prod_v2 --wide-upto 40"
until grep -q "stageD production exit" $SS 2>/dev/null; do sleep 120; done
TD=$(date +%s)
python j4_fit_v3d.py $D --out $J4/J4_fit_v3d_gateD.json --plot $J4/J4_fit_v3d_gateD > $J4/J4_fit_v3d_gateD.log 2>&1; echo "fit_v3d_gateD exit $? $(date -u +%H:%M:%S)" >> $ST
while [ $(( $(date +%s) - TD )) -lt 9000 ] && ! grep -q "stageE production exit" $SS; do sleep 120; done
python j4_fit_v3d.py $D --allow-incomplete --out $J4/J4_fit_v3d_preview62.json > $J4/J4_fit_v3d_preview62.log 2>&1; echo "fit_v3d_preview62 exit $? $(date -u +%H:%M:%S)" >> $ST
until grep -q "stageE production exit" $SS 2>/dev/null; do sleep 120; done
python j4_fit_v3d.py $D --out $J4/J4_fit_v3d_final.json --plot $J4/J4_fit_v3d_final > $J4/J4_fit_v3d_final.log 2>&1; echo "fit_v3d_final exit $? $(date -u +%H:%M:%S)" >> $ST
echo "all done $(date -u +%H:%M:%S)" >> $ST
