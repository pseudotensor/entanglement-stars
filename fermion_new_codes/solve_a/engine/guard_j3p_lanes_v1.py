"""Builder analyst: guarded copies of lane 1's J3' and C-capture lines in lane 3's queue (whichever lane frees first runs each; the other
records a skip). Run on the pod from <internal>/engine; rewrites queue_lane1.txt (guards on the four unstarted lines) and appends to
queue_lane3.txt; each file written to a temporary name and moved."""
import os
tags = ["j3p_prod19_main", "j3p_prod19_kjm1", "j3p_prod19_j1.5", "ccap_prod19_last"]
L1 = open("queue_lane1.txt").read().split("\n")
started1 = set(open("queue_lane1.txt.started").read().split())
cmds = {}
out1 = []
for l in L1:
    t = l.split("\t", 1)[0]
    if t in tags:
        assert t not in started1, t
        cmd = l.split("\t", 1)[1]
        assert not cmd.startswith("if grep"), "already guarded"
        cmds[t] = cmd
        l = "%s\tif grep -q \"lane3 %s_l3 start\" h100b_status.txt; then echo \"lane1 %s skipped (run by lane 3) $(date -u +%%H:%%M:%%S) UTC\" >> h100b_status.txt; exit 0; fi; %s" % (t, t, t, cmd)
    out1.append(l)
assert len(cmds) == 4, cmds.keys()
L3 = open("queue_lane3.txt").read().rstrip("\n").split("\n")
for t in tags:
    L3.append("%s_l3\tif grep -q \"lane1 %s start\" h100b_status.txt; then echo \"lane3 %s_l3 skipped (run by lane 1) $(date -u +%%H:%%M:%%S) UTC\" >> h100b_status.txt; exit 0; fi; %s" % (t, t, t, cmds[t]))
open("queue_lane1.txt.new", "w").write("\n".join(out1)); os.replace("queue_lane1.txt.new", "queue_lane1.txt")
open("queue_lane3.txt.new", "w").write("\n".join(L3) + "\n"); os.replace("queue_lane3.txt.new", "queue_lane3.txt")
print("guarded", len(cmds))
