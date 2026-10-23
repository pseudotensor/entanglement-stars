"""FP64 timing of the dense kernels the ball-cut production needs, on the lane's H100 (task 1, J4).
Singular values of a real square block (torch.linalg.svdvals, cuSOLVER) against the Gram route
(eigvalsh of K K^T), and complex Hermitian eigvalsh, at the block sizes of radii 24 to 64.
Writes a JSON with the seconds per call (the second of 2 calls, after warm-up) and the peak memory."""
import json, sys, time
import torch

sizes = [int(s) for s in sys.argv[1].split(",")] if len(sys.argv) > 1 else [4096, 8192, 16384, 24576, 32768]
out = sys.argv[2] if len(sys.argv) > 2 else "gpu_bench_v1.json"
dev = torch.device("cuda")
rec = {"device": torch.cuda.get_device_name(0), "torch": torch.__version__, "rows": []}
for n in sizes:
    row = {"n": n}
    g = torch.Generator(device=dev).manual_seed(n)
    K = torch.randn(n, n, generator=g, dtype=torch.float64, device=dev) / n ** 0.5
    for name, fn in [("svdvals", lambda: torch.linalg.svdvals(K)),
                     ("gram_eigvalsh", lambda: torch.linalg.eigvalsh(K @ K.T)),
                     ("sym_eigvalsh", lambda: torch.linalg.eigvalsh(K + K.T))]:
        try:
            torch.cuda.reset_peak_memory_stats()
            for rep in range(2):
                torch.cuda.synchronize(); t0 = time.time()
                v = fn(); torch.cuda.synchronize()
                dt = time.time() - t0
            row[name + "_s"] = round(dt, 3)
            row[name + "_peak_GB"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
            del v
        except Exception as exc:  # noqa: BLE001
            row[name + "_error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
        torch.cuda.empty_cache()
    if n <= 16384:
        try:
            Z = torch.complex(K, K.T.contiguous()); H = Z + Z.conj().T; del Z
            torch.cuda.synchronize(); t0 = time.time(); w = torch.linalg.eigvalsh(H); torch.cuda.synchronize()
            row["complex_eigvalsh_s"] = round(time.time() - t0, 3); del H, w
        except Exception as exc:  # noqa: BLE001
            row["complex_eigvalsh_error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
    del K; torch.cuda.empty_cache()
    rec["rows"].append(row)
    print(json.dumps(row), flush=True)
    json.dump(rec, open(out, "w"), indent=1)
