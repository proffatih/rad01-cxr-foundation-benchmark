"""
latency.py — inference-latency protocol.

What is timed: the encoder forward pass only, on an input tensor that is already
pre-processed and resident on the target device (image decoding, resizing and
host-to-device transfer are excluded). Linear-probe cost is negligible (one D x 15
matrix product) and excluded.

GPU: NVIDIA RTX 3070 Laptop (8 GB), CUDA events + synchronize; 20 warm-up and 100 timed
iterations; reported = median iteration time / batch size, for batch 1 and 32,
fp32 (primary) and fp16 autocast.
CPU: batch 1, fp32, torch threads = physical cores; 5 warm-up, 30 timed iterations.
Output: results/latency.json
"""
from __future__ import annotations
import os, sys, time, json, platform, subprocess
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(__file__))
import common as rc
import extract_full as ex


def gpu_time(f, x, iters=100, warm=20, half=False):
    ctx = torch.autocast("cuda", dtype=torch.float16) if half else torch.autocast("cuda", enabled=False)
    with torch.inference_mode(), ctx:
        for _ in range(warm):
            f(x)
        torch.cuda.synchronize()
        ts = []
        for _ in range(iters):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record(); f(x); e.record(); torch.cuda.synchronize()
            ts.append(s.elapsed_time(e))
    return float(np.median(ts) / x.shape[0])


def cpu_time(f, x, iters=30, warm=5):
    with torch.inference_mode():
        for _ in range(warm):
            f(x)
        ts = []
        for _ in range(iters):
            t = time.perf_counter(); f(x); ts.append((time.perf_counter() - t) * 1000)
    return float(np.median(ts) / x.shape[0])


def n_params(f):
    for cell in (f.__closure__ or []):
        obj = cell.cell_contents
        if isinstance(obj, torch.nn.Module):
            return round(sum(p.numel() for p in obj.parameters()) / 1e6, 1)
    return None


def main():
    cpu_name = [l.split(":", 1)[1].strip() for l in subprocess.check_output(["lscpu"], text=True).splitlines()
                if l.startswith("Model name")][0]
    phys = int(subprocess.check_output("lscpu -p=Core | grep -v '^#' | sort -u | wc -l", shell=True, text=True))
    torch.set_num_threads(phys)
    res = {"gpu": torch.cuda.get_device_name(0), "cpu": cpu_name, "cpu_threads": phys,
           "torch": torch.__version__, "protocol": __doc__, "configs": {}}
    for cid, (build, size, _, _) in ex.CONFIGS.items():
        ex.DEV = "cuda"
        f = build()
        r = {"input": size, "params_M": n_params(f)}
        for bs in (1, 32):
            x = torch.randn(bs, 3, size, size, device="cuda")
            r[f"gpu_fp32_b{bs}_ms"] = gpu_time(f, x)
            r[f"gpu_fp16_b{bs}_ms"] = gpu_time(f, x, half=True)
        del f; torch.cuda.empty_cache()
        ex.DEV = "cpu"
        fc = build()
        r["cpu_fp32_b1_ms"] = cpu_time(fc, torch.randn(1, 3, size, size))
        del fc
        res["configs"][cid] = r
        print(cid, {k: round(v, 2) if isinstance(v, float) else v for k, v in r.items()}, flush=True)
        rc.save_json(res, rc.RESULTS / "latency.partial.json")
    rc.save_json(res, rc.RESULTS / "latency.json")


if __name__ == "__main__":
    main()
