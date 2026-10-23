#!/bin/bash
# install_h2default_v1.sh (busi, analyst; inbox #343; ready_jobs_v3 sec. 33): makes a NEW code dir for a Newton line's next restart: a copy
# of the line's current code dir with engine_fields_v16.py replaced by engine_fields_v16c's content (fa84dad3; H2 Options A and B on by
# default). The line's current code dir is only read (never written, never touched). Engines, v17 and every other file are copied unchanged
# (checked by md5). Refuses when the new dir exists, when the old engine_fields_v16.py is neither v16 (1ffcf5e8) nor v16b (d7250a37, the
# sec. 31 swap), or when the copy would exceed 2 GB.
#   usage: install_h2default_v1.sh OLD_CODE_DIR NEW_CODE_DIR        (KIT defaults to this script's directory)
set -u
KIT=${KIT:-$(cd "$(dirname "$0")" && pwd)}
OLD=${1:?old code dir}; NEW=${2:?new code dir}
V16=1ffcf5e88c7953d7bfee025d57f4fb86; V16B=d7250a370c0f083f5a8c740c108d49e7; V16C=fa84dad3183704f4b908cba6c09c49ca
die() { echo "install_h2default_v1: $*; nothing written"; exit 1; }
die2() { echo "install_h2default_v1: $*; the incomplete $NEW is left for inspection (remove it before a retry); $OLD untouched"; exit 1; }
[ -d "$OLD" ] || die "$OLD is not a directory"
OLD=$(cd "$OLD" && pwd)
[ -e "$NEW" ] && die "$NEW exists"
[ "$(md5sum < "$KIT/engine_fields_v16.py" | cut -c1-32)" = "$V16C" ] || die "the kit's engine_fields_v16.py is not v16c"
m=$(md5sum < "$OLD/engine_fields_v16.py" | cut -c1-32)
case $m in $V16) was=v16;; $V16B) was=v16b;; $V16C) die "$OLD already runs v16c";; *) die "$OLD/engine_fields_v16.py md5 $m is neither v16 nor v16b";; esac
kb=$(du -sk --exclude=__pycache__ "$OLD" | cut -f1); [ "$kb" -lt 2097152 ] || die "$OLD holds $kb kB (over 2 GB): copy the code files by hand"
mkdir -p "$NEW" || die "cannot create $NEW"
NEW=$(cd "$NEW" && pwd)
( cd "$OLD" && tar --exclude=__pycache__ -cf - . ) | ( cd "$NEW" && tar -xpf - ) || die2 "copy failed"
cp -p "$NEW/engine_fields_v16.py" "$NEW/engine_fields_v16.py.${was}_${m:0:8}"
cp "$KIT/engine_fields_v16.py" "$NEW/engine_fields_v16.py"
[ "$(md5sum < "$NEW/engine_fields_v16.py" | cut -c1-32)" = "$V16C" ] || die2 "the new engine_fields_v16.py is not v16c"
bad=$(cd "$OLD" && find . -name __pycache__ -prune -o -type f ! -name engine_fields_v16.py -print | while read -r f; do cmp -s "$f" "$NEW/$f" || echo "$f"; done)
[ -z "$bad" ] || die2 "files differ after the copy: $(echo $bad | head -c 400)"
( cd "$NEW" && md5sum *.py > MD5SUMS_h2default.txt )
echo "install_h2default_v1: $NEW = $OLD (engine_fields_v16.py was $was ${m:0:8}) with engine_fields_v16.py -> v16c fa84dad3; every other file equal"
grep -E "  (engine_collapse_v3(6q_srf|6r|6nq|6y|6z|7a|7n)|engine_fields_v1[67])\.py$" "$NEW/MD5SUMS_h2default.txt"
echo "restart the line from $NEW with its own command line unchanged (H2_* unset = A and B on); revert: restart from $OLD (or export H2_PW_ONE=0 H2_LOCAL_SPLINE=0)"
