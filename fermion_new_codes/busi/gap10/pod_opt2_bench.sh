#!/bin/bash
# GPU optimization (inbox #34), opt2's gate and timing on the pod (builder; busi/notes/GPU_opt_judgment_v2.md section 4), beside
# whatever runs on the GPU: the bench at L 65 (gate) and L 129 (timing), then the full test file on opt2 with the compiled backend.
cd ${QI2_ROOT}/gap10; ST=${QI2_ROOT}/opt2_status.txt
python ${QI2_ROOT}/gap10_opt_bench_v2.py --device cuda --size 65 --timing_size 129 --json opt_bench_pod_v2.json > opt_bench_pod_v2.log 2>&1; echo "bench exit $? $(date -u +%H:%M:%S)" >> $ST
cd ${QI2_ROOT}/gap10 && GAP10_OPT2_BACKEND=compiled python -c "import sys, runpy; sys.path.insert(0, '.'); import gap10_core_opt2 as C; sys.modules['gap10_core_v2'] = C; sys.argv = ['gap10_tests_v3.py', '--device', 'cuda']; runpy.run_path('gap10_tests_v3.py', run_name='__main__')" > ${QI2_ROOT}/tests_pod_opt2.log 2>&1; echo "tests_opt2 exit $? $(date -u +%H:%M:%S)" >> $ST
echo "all done $(date -u +%H:%M:%S)" >> $ST
