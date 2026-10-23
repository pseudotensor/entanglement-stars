import sys, os, time, json, numpy as np, torch
sys.path.insert(0, "<internal>/perf_redesign"); sys.path.insert(0, "<internal>/code")
import banded_eigh_gpu_v1 as BE, engine_vacflow_v1 as VF, banded_eigh_proto_v1 as PR
dev = torch.device("cuda:0"); torch.cuda.set_device(dev)
lines = []; out = []
for n_s in (8192,):
    R = np.arange(n_s, dtype=float) + 1.0; rs = n_s / 32.0
    v = VF.profile(R, rs, "exact"); N = 1.0 - 0.3 * np.exp(-(R / (2 * rs)) ** 2)
    h0, Q = VF.chain_sparse(R, 0.2, 1.0)
    PR.run_case("synthetic_%d" % n_s, h0, Q, N, v, dev, lines, out, dense=True, reps=1)
    print(lines[-2], flush=True); print(lines[-1], flush=True)
json.dump(out, open("<internal>/banded_eigh_synth16k.json", "w"), indent=1, default=float)
open("<internal>/banded_eigh_synth16k.txt", "w").write("\n".join(lines) + "\n")
