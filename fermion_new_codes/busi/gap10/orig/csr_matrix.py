#!/usr/bin/env python
"""csr_matrix.py -- prints the run matrix (1 shell command per propagation), tier by tier."""


def cmd(**kw):
    parts = ["python csr_run.py"]
    for k, v in kw.items():
        if isinstance(v, (list, tuple)):
            parts.append(f"--{k} " + " ".join(str(x) for x in v))
        else:
            parts.append(f"--{k} {v}")
    return " ".join(parts)


def tier(name, runs):
    print(f"# ---- {name} ({len(runs)} runs)")
    for r in runs:
        print(r)


DIRS = ('100', '111')
DEPTHS = (2, 4, 8, 16)
NOTWIN = dict(inside_twins_max_depth=0)
tier("T0 tests", ["python csr_tests.py"])
tier("T1 primary: rs=32, ma=0, ell=4, Rb=0.6 rs, back-averaged, source=both (inside twins at d<=4), plus the helicity check",
     [cmd(rs=32, ma=0, ell=4, Rb_frac=0.6, direction=d, depth=dp, source='both', out='runs/T1') for d in DIRS for dp in DEPTHS]
     + [cmd(rs=32, ma=0, ell=4, Rb_frac=0.6, direction='100', depth=4, source='packet', spins=2, tag='_hel2', out='runs/T1', **NOTWIN)])
tier("T2 primary: rs=32, ma=0.2 (packets at a w = 0.25, 0.3, 0.4)",
     [cmd(rs=32, ma=0.2, ell=4, Rb_frac=0.6, direction=d, depth=dp, source='both', out='runs/T2') for d in DIRS for dp in DEPTHS])
tier("T3 controls at rs=32, ma=0: ell=8; Rb=0.8 rs; point sampling (fixed preparation profile throughout)",
     [cmd(rs=32, ma=0, ell=8, Rb_frac=0.6, direction=d, depth=dp, source='both', out='runs/T3_ell8', **NOTWIN) for d in DIRS for dp in (4, 16)]
     + [cmd(rs=32, ma=0, ell=4, Rb_frac=0.8, direction=d, depth=dp, source='both', out='runs/T3_Rb08', **NOTWIN) for d in DIRS for dp in (4, 16)]
     + [cmd(rs=32, ma=0, ell=4, Rb_frac=0.6, sampling='point', direction='100', depth=dp, source='both', out='runs/T3_point', **NOTWIN) for dp in (4, 16)])
tier("T4 resolution: rs=24 full matrix at ma=0; parent controls (half-width 2.5 rs margin 0; half-width 4.5 rs + 16)",
     [cmd(rs=24, ma=0, ell=4, Rb_frac=0.6, direction=d, depth=dp, source='both', out='runs/T4_rs24', **NOTWIN) for d in DIRS for dp in DEPTHS]
     + [cmd(rs=24, ma=0, ell=4, Rb_frac=0.6, half_frac=2.5, margin=0, direction='100', depth=dp, source='both', out='runs/T4_box', **NOTWIN) for dp in (4, 16)]
     + [cmd(rs=24, ma=0, ell=4, Rb_frac=0.6, half_frac=4.5, margin=16, direction='100', depth=dp, source='both', out='runs/T4_box', **NOTWIN) for dp in (4, 16)])
tier("T5 solver cross-check: Taylor stepping against Chebyshev at rs=24",
     [cmd(rs=24, ma=0, ell=4, Rb_frac=0.6, direction='100', depth=4, source='both', solver='taylor', out='runs/T5_taylor', **NOTWIN),
      cmd(rs=24, ma=0, ell=4, Rb_frac=0.6, direction='111', depth=8, source='both', solver='taylor', out='runs/T5_taylor', **NOTWIN)])
tier("T6 flat-space control at production size (v = 0), with the exact Fourier comparison at t = 1 and 2 rs",
     [cmd(rs=32, ma=0, mode='flat', direction='100', depth=4, source='both', fft_check_sweeps=[1, 2], out='runs/T6_flat'),
      cmd(rs=32, ma=0.2, mode='flat', direction='100', depth=4, source='both', fft_check_sweeps=[1, 2], out='runs/T6_flat')])
print("# ---- analysis")
print("python csr_analysis.py runs/T1 runs/T2 runs/T3_ell8 runs/T3_Rb08 runs/T3_point runs/T4_rs24 runs/T4_box runs/T5_taylor runs/T6_flat "
      "--csv csr_results.csv --md csr_results.md --figs figs")
