#!/usr/bin/env python
"""Strand 6 (i) revision 4, the builder's panel check (labelled): F_0 direct at the diagnostic and window-edge rectangles with the (S, u) panel
densities 3/2, 6/4, 12/8 (revision 3's) and 24/16 per decade (strand6i_rev4_v1.F0_direct_panels), to show the quadrature's convergence. Prints."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import strand6i_kernel_sign_v2 as KS  # noqa: E402
import strand6i_rev4_v1 as R4  # noqa: E402

mu = KS.MU
mu_e = mu * 0.0055 / 0.0213
for name, m in (("diagnostic", mu), ("window edge", mu_e)):
    for sp, up in ((3, 2), (6, 4), (12, 8), (24, 16)):
        print("%s: S %d per decade, u %d per decade: F_0 direct = %.15e" % (name, sp, up, R4.F0_direct_panels(0.0, m ** 2, m ** 2, sp, up)), flush=True)
