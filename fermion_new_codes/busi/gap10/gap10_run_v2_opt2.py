#!/usr/bin/env python
"""gap10_run_v2_opt2.py -- gap10_run_v2.py unchanged, with the core gap10_core_opt2 bound to the name gap10_core_v2 before
the driver's `import gap10_core_v2 as C`. Same command line, same outputs; the [run] line prints "core gap10_core_opt2"
and the json 'core' field records it. The backend is chosen by GAP10_OPT2_BACKEND (compiled | eager | auto); for a
production run set the backend that passed the gate explicitly, so that a compile failure raises instead of falling back
(busi/notes/GPU_opt_judgment_v2.md, the corrected adoption rule). The file name keeps 'gap10_run_v2' so that the matrix
chain's grep on that string still selects the commands. The same binding runs the test file on the opt2 core:
    GAP10_OPT2_BACKEND=compiled python -c "import sys, runpy; sys.path.insert(0, '.'); import gap10_core_opt2 as C; \
        sys.modules['gap10_core_v2'] = C; sys.argv = ['gap10_tests_v3.py', '--device', 'cuda']; \
        runpy.run_path('gap10_tests_v3.py', run_name='__main__')"
Note for reading the log: the first [sweep] line of a run includes the compile of the step (about 3 to 6 s, printed on
the [opt2] line); the production-size ms per application is the second [sweep] line.
"""
import os
import runpy
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import gap10_core_opt2 as C   # noqa: E402  (imports the real gap10_core_v2 first, then takes its name)

sys.modules['gap10_core_v2'] = C
runpy.run_path(os.path.join(_HERE, 'gap10_run_v2.py'), run_name='__main__')
