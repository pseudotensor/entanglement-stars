# b4_threads_v1.sh (busi f52b1152, 20 Oct): sourced by the B4 launchers. H1's thread limits: OMP/MKL/OPENBLAS_NUM_THREADS = the card's CPU
# quota minus 2, the quota read at launch from /sys/fs/cgroup/cpu.max (cgroup v2) or cpu.cfs_quota_us / cpu.cfs_period_us (v1).
# No quota (the box): B4_NOQUOTA_THREADS (default 8). B4_THREADS overrides everything. Sets B4_NT and B4_QUOTA_LINE.
b4_threads() {
  local q="" a b
  if [ -r /sys/fs/cgroup/cpu.max ]; then read a b < /sys/fs/cgroup/cpu.max; [ "$a" != "max" ] && [ -n "$b" ] && q=$(( a / b )); B4_QUOTA_LINE="cpu.max $a $b"; fi
  if [ -z "$q" ] && [ -r /sys/fs/cgroup/cpu/cpu.cfs_quota_us ]; then a=$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us); b=$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us); [ "$a" -gt 0 ] 2>/dev/null && q=$(( a / b )); B4_QUOTA_LINE="cfs_quota $a $b"; fi
  [ -z "$B4_QUOTA_LINE" ] && B4_QUOTA_LINE="no quota"
  if [ -n "$B4_THREADS" ]; then B4_NT=$B4_THREADS; elif [ -n "$q" ]; then B4_NT=$(( q - 2 )); else B4_NT=${B4_NOQUOTA_THREADS:-8}; fi
  [ "$B4_NT" -lt 1 ] && B4_NT=1
  export OMP_NUM_THREADS=$B4_NT MKL_NUM_THREADS=$B4_NT OPENBLAS_NUM_THREADS=$B4_NT
}
