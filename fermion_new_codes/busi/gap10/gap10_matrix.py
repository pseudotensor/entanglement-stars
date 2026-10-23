#!/usr/bin/env python
"""gap10_matrix.py -- prints the run matrix (1 shell command per propagation), tier by tier, with the pod paths.
Roots: GAP10_CODE (default <internal>/gap10) and GAP10_RUNS (default <internal>/gap10)."""
import os

CODE = os.environ.get('GAP10_CODE', '<internal>/gap10')
RUNS = os.environ.get('GAP10_RUNS', '<internal>/gap10')
PY = os.environ.get('GAP10_PY', 'python')


def cmd(**kw):
    parts = [f"{PY} {CODE}/gap10_run.py"]
    for k, v in kw.items():
        if isinstance(v, (list, tuple)):
            parts.append(f"--{k} " + " ".join(str(x) for x in v))
        elif v is True:
            parts.append(f"--{k}")
        else:
            parts.append(f"--{k} {v}")
    return " ".join(parts)


def tier(name, runs):
    print(f"# ---- {name} ({len(runs)} runs)")
    for r in runs:
        print(r)


DIRS = ('100', '111')
DEPTHS = (0, 2, 4, 8, 16)
NOTWIN = dict(inside_twins_max_depth=-1)
tier("T0 tests", [f"{PY} {CODE}/gap10_tests.py"])
tier("T1 primary: rs=32, ma=0, ell=4, Rb=0.6 rs, back-averaged, tau=rs/2, source=both (inside twins at d<=4), plus the helicity check",
     [cmd(rs=32, ma=0, ell=4, Rb_frac=0.6, tau_frac=0.5, direction=d, depth=dp, source='both', out=f'{RUNS}/runs/T1') for d in DIRS for dp in DEPTHS]
     + [cmd(rs=32, ma=0, ell=4, Rb_frac=0.6, tau_frac=0.5, direction='100', depth=4, source='packet', spins=2, tag='_hel2', out=f'{RUNS}/runs/T1', **NOTWIN)])
tier("T2 primary: rs=32, ma=0.2 (packets at a w = 0.25, 0.3, 0.4), tau=rs/2",
     [cmd(rs=32, ma=0.2, ell=4, Rb_frac=0.6, tau_frac=0.5, direction=d, depth=dp, source='both', out=f'{RUNS}/runs/T2') for d in DIRS for dp in DEPTHS])
tier("T3 controls at rs=32, ma=0 (fixed preparation throughout): ell=8; Rb=0.8 rs; cell-only sampling (the ell -> 0 end); point sampling",
     [cmd(rs=32, ma=0, ell=8, Rb_frac=0.6, direction=d, depth=dp, source='both', out=f'{RUNS}/runs/T3_ell8', **NOTWIN) for d in DIRS for dp in (4, 16)]
     + [cmd(rs=32, ma=0, ell=4, Rb_frac=0.8, direction=d, depth=dp, source='both', out=f'{RUNS}/runs/T3_Rb08', **NOTWIN) for d in DIRS for dp in (4, 16)]
     + [cmd(rs=32, ma=0, ell=4, Rb_frac=0.6, sampling='cell', direction='100', depth=dp, source='both', out=f'{RUNS}/runs/T3_cell', **NOTWIN) for dp in (4, 16)]
     + [cmd(rs=32, ma=0, ell=4, Rb_frac=0.6, sampling='point', direction='100', depth=dp, source='both', out=f'{RUNS}/runs/T3_point', **NOTWIN) for dp in (4, 16)])
tier("T4 resolution: rs=24 full matrix at ma=0; parent control at [100] d=4, 16 on common vectors clipped to half-width 60: H=60 (margin 0), 88, 124",
     [cmd(rs=24, ma=0, ell=4, Rb_frac=0.6, direction=d, depth=dp, source='both', out=f'{RUNS}/runs/T4_rs24', **NOTWIN) for d in DIRS for dp in DEPTHS]
     + [cmd(rs=24, ma=0, ell=4, Rb_frac=0.6, half_frac=2.5, margin=0, common_half=60, direction='100', depth=dp, source='both', tag='_c60', out=f'{RUNS}/runs/T4_box', **NOTWIN) for dp in (4, 16)]
     + [cmd(rs=24, ma=0, ell=4, Rb_frac=0.6, half_frac=3.0, margin=16, common_half=60, direction='100', depth=dp, source='both', tag='_c60', out=f'{RUNS}/runs/T4_box', **NOTWIN) for dp in (4, 16)]
     + [cmd(rs=24, ma=0, ell=4, Rb_frac=0.6, half_frac=4.5, margin=16, common_half=60, direction='100', depth=dp, source='both', tag='_c60', out=f'{RUNS}/runs/T4_box', **NOTWIN) for dp in (4, 16)])
tier("T5 solver cross-check: Taylor stepping against Chebyshev at rs=24 (readouts at the step ends, dt_read = 1)",
     [cmd(rs=24, ma=0, ell=4, Rb_frac=0.6, direction='100', depth=4, source='both', solver='taylor', dt_read=1.0, out=f'{RUNS}/runs/T5_taylor', **NOTWIN),
      cmd(rs=24, ma=0, ell=4, Rb_frac=0.6, direction='111', depth=8, source='both', solver='taylor', dt_read=1.0, out=f'{RUNS}/runs/T5_taylor', **NOTWIN)])
tier("T6 flat-space control at production size (v = 0), with the exact Fourier comparison at t = 1 and 2 rs",
     [cmd(rs=32, ma=0, mode='flat', direction='100', depth=4, source='site', fft_check_sweeps=[1, 2], out=f'{RUNS}/runs/T6_flat'),
      cmd(rs=32, ma=0.2, mode='flat', direction='100', depth=4, source='site', fft_check_sweeps=[1, 2], out=f'{RUNS}/runs/T6_flat')])
print("# ---- T7 record profile (fill RS, MA, ELL and the record table written from the solve lane's delivered collapse slice; skip if no trapped slice exists)")
for d in DIRS:
    for dp in (2, 8):
        print(cmd(rs='RS', ma='MA', ell='ELL', Rb_frac='RBFRAC', profile='file', profile_file=f'{RUNS}/records/record_slice.npz', sampling='point',
                  direction=d, depth=dp, source='both', out=f'{RUNS}/runs/T7_record', **NOTWIN))
print("# ---- analysis")
print(f"{PY} {CODE}/gap10_analysis.py {RUNS}/runs/T1 {RUNS}/runs/T2 {RUNS}/runs/T3_ell8 {RUNS}/runs/T3_Rb08 {RUNS}/runs/T3_cell {RUNS}/runs/T3_point "
      f"{RUNS}/runs/T4_rs24 {RUNS}/runs/T4_box {RUNS}/runs/T5_taylor {RUNS}/runs/T6_flat {RUNS}/runs/T7_record "
      f"--csv {RUNS}/gap10_results.csv --tcsv {RUNS}/gap10_transmissivity.csv --md {RUNS}/gap10_results.md --figs {RUNS}/figs")
