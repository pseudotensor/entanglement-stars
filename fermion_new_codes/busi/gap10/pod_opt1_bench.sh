#!/bin/bash
# GPU optimization (inbox #34), the pod's gate and timing for gap10_core_opt1 beside J4's stage E (builder): the bench at
# L 65 (gate) and L 129 (timing), then the full test file on the new core with the compiled backend (busi/notes/GPU_opt_judgment_v1.md section 7).
cd ${QI2_ROOT}/gap10; ST=${QI2_ROOT}/opt1_status.txt
python ${QI2_ROOT}/gap10_opt_bench_v1.py --device cuda --size 65 --timing_size 129 --json opt_bench_pod_v1.json > opt_bench_pod_v1.log 2>&1; echo "bench exit $? $(date -u +%H:%M:%S)" >> $ST
cd ${QI2_ROOT}/gap10 && GAP10_OPT1_BACKEND=compiled python -c "import sys, runpy; sys.path.insert(0, '.'); import gap10_core_opt1 as C; sys.modules['gap10_core_v2'] = C; sys.argv = ['gap10_tests_v3.py', '--device', 'cuda']; runpy.run_path('gap10_tests_v3.py', run_name='__main__')" > ${QI2_ROOT}/tests_pod_opt1.log 2>&1; echo "tests_opt1 exit $? $(date -u +%H:%M:%S)" >> $ST
echo "all done $(date -u +%H:%M:%S)" >> $ST
