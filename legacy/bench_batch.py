# -*- coding: utf-8 -*-
"""
bench_batch.py -- forward and forward+backward wall time vs batch size for the
PINN network, plus achieved FLOP rate.  This is the measurement that decides
whether a GPU could help: it shows whether the CPU is already in the
throughput-bound regime (large batches, high GFLOP/s) or in the latency-bound
regime (small batches, low GFLOP/s), and gives the arithmetic intensity.

Run as an isolated process (thread count fixed at import).
"""
import argparse
import json
import time

import torch

from common import MLP, Params, pinn_losses, env_info


def flops_fwd(width, depth, n_in=1, n_out=3):
    """multiply-add pairs counted as 2 FLOP, per sample."""
    f = 2 * n_in * width
    f += 2 * width * width * (depth - 1)
    f += 2 * width * n_out
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--depth", type=int, default=5)
    ap.add_argument("--json", default="batch.json")
    ap.add_argument("--max-n", type=int, default=100_000,
                    help="largest batch size; N=1e6 f64 needs ~40 s/step and "
                         "several GB -- do NOT run it on the 4-core box")
    ap.add_argument("--dtypes", default="float64,float32")
    a = ap.parse_args()
    torch.set_num_threads(min(a.threads, 4))          # load limit
    P = Params()
    want = [getattr(torch, s.strip()) for s in a.dtypes.split(",") if s.strip()]

    rows = []
    for dtype in want:
        net = MLP(width=a.width, depth=a.depth, seed=0)
        net = net.double() if dtype == torch.float64 else net.float()
        for n in [k for k in (1, 10, 100, 1000, 10_000, 100_000, 1_000_000)
                  if k <= a.max_n]:
            x = torch.rand(n, 1, dtype=dtype)
            # --- pure forward (inference) ---
            with torch.no_grad():
                for _ in range(3):
                    net(x)
                r = max(3, min(200, int(2e6 / n)))
                t0 = time.perf_counter()
                for _ in range(r):
                    net(x)
                t_fwd = (time.perf_counter() - t0) / r

            # --- full PINN loss fwd+bwd (residual only, cheapest realistic) ---
            z = x.clone().requires_grad_(True)
            def step():
                net.zero_grad(set_to_none=True)
                Lr, LH, Ls, Lic = pinn_losses(net, z, P, use_H=False,
                                              use_sym=False)
                (Lr + Lic).backward()
            try:
                for _ in range(2):
                    step()
                r2 = max(2, min(50, int(2e5 / n)))
                t0 = time.perf_counter()
                for _ in range(r2):
                    step()
                t_step = (time.perf_counter() - t0) / r2
            except RuntimeError as e:
                t_step = float("nan")
                print("  step failed:", e)

            fpS = flops_fwd(a.width, a.depth)
            gflops_fwd = n * fpS / t_fwd / 1e9
            rows.append(dict(dtype=str(dtype).replace("torch.", ""), n=n,
                             t_fwd_s=t_fwd, us_fwd=t_fwd * 1e6,
                             gflops_fwd=gflops_fwd,
                             t_step_s=t_step, ms_step=t_step * 1e3,
                             flops_per_sample_fwd=fpS))
            print(f"{str(dtype):<16s} N={n:<9d} fwd={t_fwd*1e6:10.1f} us "
                  f"({gflops_fwd:7.2f} GFLOP/s)   res-loss fwd+bwd="
                  f"{t_step*1e3:9.3f} ms", flush=True)

    with open(a.json, "w", encoding="utf-8") as f:
        json.dump(dict(env=env_info(), width=a.width, depth=a.depth,
                       threads=a.threads, rows=rows), f, indent=2)
    print("written", a.json)


if __name__ == "__main__":
    main()
