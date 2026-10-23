#!/bin/bash
# Waits on the box until the synced copy of the resumed production (H100b/C3prod19/..._r11) holds its last record (T/rs 40) in the json and
# in the spectra file, then writes the marker REPLAY3/prod_final.marker (the GPU 1 queue's condition for engine_replay_v3). Builder analyst.
cd ${QI2_ROOT}/engine || exit 1
P=H100b/C3prod19/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11
until ${QI2_ROOT}/python -c "
import json, numpy as np, sys
J = json.load(open('$P.json')); rs = float(J['args']['rs'])
last = max(r['T'] / rs for r in J['records'] if r.get('E') is not None)
S = np.load('${P}_spectra.npz'); H = np.load('${P}_handoff.npz')
ok = abs(last - 40.0) < 1e-9 and abs(float(S['T'][-1]) / rs - 40.0) < 1e-9 and abs(float(H['T'][-1]) / rs - 40.0) < 1e-9
sys.exit(0 if ok else 1)
" 2>/dev/null; do sleep 60; done
sleep 30
touch REPLAY3/prod_final.marker
echo "REPLAY3 the production's synced copy is final $(date -u +%H:%M:%S) UTC" >> REPLAY3/status.txt
