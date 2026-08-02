# -*- coding: utf-8 -*-
"""
bench_scaling.py -- per-Adam-iteration cost of the PINN as a function of
  * dtype (float64 / float32)
  * number of collocation points
  * network width / depth
  * which loss terms are active (cost of the 2nd-order L_sym term)
  * number of CPU threads

Every entry is a real timed loop (warmup + N timed iterations).
"""
import argparse
import json
import time

import torch

from common import MLP, Params, pinn_losses, env_info


def time_iter(width, depth, ncol, dtype, use_H, use_sym, n_iter=30, warmup=5,
              threads=4, seed=0):
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    P = Params()
    net = MLP(width=width, depth=depth, seed=seed)
    net = net.double() if dtype == torch.float64 else net.float()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    z = (2.0 * torch.rand(ncol, 1, dtype=dtype) - 1.0).requires_grad_(True)

    def step():
        opt.zero_grad(set_to_none=True)
        Lr, LH, Ls, Lic = pinn_losses(net, z, P, use_H=use_H, use_sym=use_sym)
        (Lr + LH + Ls + 100.0 * Lic).backward()
        opt.step()

    for _ in range(warmup):
        step()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        step()
    dt = (time.perf_counter() - t0) / n_iter
    return dt * 1e3, net.n_params()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--niter", type=int, default=30)
    ap.add_argument("--json", default="scaling.json")
    args = ap.parse_args()

    rows = []

    def add(label, **kw):
        n_iter = kw.pop("n_iter", args.niter)
        ms, npar = time_iter(n_iter=n_iter, **kw)
        r = dict(label=label, ms_per_iter=ms, n_params=npar,
                 dtype=str(kw["dtype"]).replace("torch.", ""), **{
                     k: v for k, v in kw.items() if k != "dtype"})
        rows.append(r)
        print(f"{label:<44s} {ms:9.2f} ms/iter  params={npar}", flush=True)
        return ms

    print("=== A. dtype (5x128, ncol=2048, full loss, 4 threads) ===")
    for dt in (torch.float64, torch.float32):
        add(f"dtype={dt}", width=128, depth=5, ncol=2048, dtype=dt,
            use_H=True, use_sym=True)

    print("\n=== B. loss-term breakdown (5x128, ncol=2048, f64) ===")
    for uh, us, nm in ((False, False, "res+ic only"), (True, False, "res+ic+H"),
                       (False, True, "res+ic+sym"), (True, True, "full")):
        add(f"loss={nm}", width=128, depth=5, ncol=2048, dtype=torch.float64,
            use_H=uh, use_sym=us)

    print("\n=== C. collocation points (5x128, f64, full loss) ===")
    for n in (256, 512, 1024, 2048, 4096, 8192):
        add(f"ncol={n}", width=128, depth=5, ncol=n, dtype=torch.float64,
            use_H=True, use_sym=True, n_iter=max(10, args.niter // 2))

    print("\n=== D. width (depth=5, ncol=2048, f64, full loss) ===")
    for w in (32, 64, 128, 256):
        add(f"width={w}", width=w, depth=5, ncol=2048, dtype=torch.float64,
            use_H=True, use_sym=True, n_iter=max(10, args.niter // 2))

    print("\n=== E. depth (width=128, ncol=2048, f64, full loss) ===")
    for d in (3, 4, 5, 6, 8):
        add(f"depth={d}", width=128, depth=d, ncol=2048, dtype=torch.float64,
            use_H=True, use_sym=True, n_iter=max(10, args.niter // 2))

    print("\n=== F. threads (5x128, ncol=2048, f64, full loss) ===")
    for t in (1, 2, 4, 8):
        add(f"threads={t}", width=128, depth=5, ncol=2048, dtype=torch.float64,
            use_H=True, use_sym=True, threads=t, n_iter=max(10, args.niter // 2))

    print("\n=== G. residual-only (cheap ablation), f32, thin nets ===")
    for w, d, n in ((64, 4, 1024), (64, 4, 2048), (128, 5, 1024)):
        add(f"f32 {d}x{w} ncol={n} res-only", width=w, depth=d, ncol=n,
            dtype=torch.float32, use_H=False, use_sym=False)

    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(dict(env=env_info(), rows=rows), f, indent=2)
    print(f"\nwritten {args.json}")


if __name__ == "__main__":
    main()
