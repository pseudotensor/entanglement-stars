#!/usr/bin/env python
"""tolcmp_v4_sens_sig.py (busi, 21 Oct): a SENSITIVITY read, not a verdict. Runs tolcmp_v4.main() unchanged except the number of significant
figures at which the arms' kept residuals must agree for mode A (tolcmp_v4.SIG, 4 by the ruling's "to four figures"). Usage:
    tolcmp_v4_sens_sig.py N <tolcmp_v4 arguments>      (N = 3 reads the kept residuals to three figures)"""
import importlib.util
import os
import sys

sp = importlib.util.spec_from_file_location("tolcmp_v4", os.path.join(os.path.dirname(os.path.abspath(__file__)), "tolcmp_v4.py"))
m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
m.SIG = int(sys.argv[1])
print("SENSITIVITY (not a verdict): tolcmp_v4 with the kept residuals read to %d significant figures" % m.SIG)
sys.argv = [sys.argv[0]] + sys.argv[2:]
m.main()
