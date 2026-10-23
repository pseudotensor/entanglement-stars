#!/usr/bin/env python
"""Strand 5 sec. 10(e)'s R_c rule for a packet-built kernel (builder analyst): build the summed kernel with strand5_lattice_star_v3's
setup_kernel_packets (truncation 1e-12), report its moment, half-width, the decay |K_n|/|K_0| at a few n, the last n with |K_n|/|K_0| >= 1e-10
and R_c = that n + 20 (printed last, alone, for the launcher)."""
import sys
import types

sys.path.insert(0, sys.argv[1])
import strand5_lattice_star_v3 as S  # noqa: E402

a = types.SimpleNamespace(mlight=float(sys.argv[3]), aH=float(sys.argv[4]), operator="kernel", packets=sys.argv[2])
st, k, info = S.setup_kernel(a)
K0 = abs(st[0])
last = max(n for n in st if abs(st[n]) >= 1e-10 * K0)
sys.stderr.write("moment %.10e half-width %d decay %s last n above 1e-10: %d\n" % (
    k, info["stencil_half_width"], " ".join("n%d %.2e" % (n, abs(st.get(n, 0)) / K0) for n in (100, 200, 255, 300, 400, 500) if n in st), last))
print(last + 20)
