#!/bin/bash
# gate_P3_ptcfb_card.sh (busi #331): run ON THE CARD by the solve lane before queueing P3_ptcfb_RUNG2c (R1 layout); read-only checks, prints KIT_OK or FAIL.
# 1. the engine copied to ${QI2_ROOT}/engine_collapse_v36z.py with md5 924eefed; 2. the member's inputs at their md5s;
# 3. the engine's --help lists --nk-ptc {off,fallback,test} (imports resolve in code_v36d with venv_v36d). No GPU is touched.
L=${QI2_ROOT}/ladders; C=${QI2_ROOT}/code_v36d; PY=${QI2_ROOT}/python; ok=1
chk() { m=$(md5sum "$1" 2>/dev/null | cut -c1-32); if [ "$m" = "$2" ]; then echo "OK   $2  $1"; else echo "FAIL $1: md5 ${m:-missing}, want $2"; ok=0; fi; }
chk $C/engine_collapse_v36z.py 924eefed596c9a32fe029294a2dae498
chk $L/runs/RUNG2c40_v36q/collapse_RUNG2c40_v36q.json e4c601be377c85ac2e9c3162ecc37c74
chk $L/runs/RUNG2c40_v36q/collapse_RUNG2c40_v36q_fields.npz 10eaad2fadaaea35ab603ebe3ffd721e
chk $L/runs/RUNG2c40_v36q/collapse_RUNG2c40_v36q_handoff.npz 65c6643a035afef10b0fac02ff5f6546
chk $L/runs/RUNG2c40_v36q/collapse_RUNG2c40_v36q_spectra.npz cffcd71e82d8e58484c697a489e6d545
chk $L/states/RUNG2c40_v36q/orbitals_RUNG2c40_v36q_T0002.00.npz 0fd5783f9f7223315d6c42034f2196b9
if [ -e $L/runs/pilot3_ptcfb_RUNG2c/collapse_pilot3_ptcfb_RUNG2c.json ]; then echo "FAIL the pilot's run directory already holds a json (never overwritten)"; ok=0; fi
cd $C && $PY engine_collapse_v36z.py --help 2>&1 | grep -q '{off,fallback,test}' && echo "OK   engine_collapse_v36z.py --help lists --nk-ptc {off,fallback,test}" || { echo "FAIL engine_collapse_v36z.py --help (imports?)"; ok=0; }
[ $ok = 1 ] && echo KIT_OK || echo FAIL
