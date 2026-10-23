#!/usr/bin/env python
"""s24_run_v1.py (busi, inbox #420, NOTCOMP item 2.4: the channel tail's common mode against the records' time interpolation; ReviewA_notcomp_v1
sec. 2.4). A wrapper around the production's page_history_v6s.main (611d190f, read-only, unchanged). With S24_SPLINE=1 it replaces
History.cubic_fields (the --interp cubic path: Catmull-Rom, C1) by a NATURAL CUBIC SPLINE through every record's chain fields (C2 in time), per
site, for N and v (scipy CubicSpline, bc natural); the sub-steps stay the cubic path's 1-unit sub-steps (tau = 8 a: 8 per record, the same count as
the production's linear ramp). Without S24_SPLINE the wrapper is the production engine itself (linear, or --interp cubic = Catmull-Rom).
"""
import os
import sys

import numpy as np

import page_history_v6s as PH

if os.environ.get("S24_SPLINE", "0") == "1":
    from scipy.interpolate import CubicSpline

    def _spline_fields(self, t):
        sp = getattr(self, "_s24_spline", None)
        if sp is None or sp[2] != len(self.T):
            FN = np.stack([self.fields(k)[0] for k in range(len(self.T))]); FV = np.stack([self.fields(k)[1] for k in range(len(self.T))])
            sp = (CubicSpline(self.T, FN, axis=0, bc_type="natural"), CubicSpline(self.T, FV, axis=0, bc_type="natural"), len(self.T))
            self._s24_spline = sp
            # the spline passes through every record: check at the records
            eN = max(float(np.max(np.abs(sp[0](self.T[k]) - FN[k]))) for k in range(0, len(self.T), 50))
            print("[s24] natural cubic spline (C2) through %d records x %d sites; max |spline - record| at every 50th record %.2e" % (len(self.T), FN.shape[1], eN), flush=True)
        return sp[0](t), sp[1](t)

    PH.History.cubic_fields = _spline_fields
    print("[s24] History.cubic_fields replaced by the natural cubic spline (C2); --interp cubic now means C2", flush=True)

if __name__ == "__main__":
    PH.main()
