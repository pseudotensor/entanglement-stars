#!/usr/bin/env python
"""run_slices144_v1.py (the busi lane as solve builder): a launcher for slices144_io_v1 that imports it as a module and calls its main(), so that
the readout modules' "import slices144_io_v1" and the driver share one module object (run as a script, the driver was __main__ and the readouts
imported a second copy whose OPS stayed None: AttributeError on chain_sparse). --core-version 2 maps the 'core' readout to core_kinematics_v2
(the reduced-zone convention, reviewer A's fix_slices144). Every other argument is slices144_io_v1's, passed through unchanged."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
argv = sys.argv[1:]
core_v = "2"
if "--core-version" in argv:
    i = argv.index("--core-version"); core_v = argv[i + 1]; del argv[i:i + 2]
import slices144_io_v1 as IO  # noqa: E402
if core_v == "2":
    IO.READOUT_MODULES["core"] = "core_kinematics_v2"
sys.exit(IO.main(argv))
