# -*- coding: utf-8 -*-
"""
bench_ansatz.py -- does the PINN actually converge?

bench_pinn.py (soft initial conditions, L_ic weighted by 100, StepLR halving
every adam//4) plateaus at residual ~5e-2 and L2-relative error ~0.4 even after
8000 Adam iterations (see pinn_res_long.log).  That is a *training-configuration*
failure, not a compute-budget shortfall, so it must be diagnosed before any
budget for Fig. 2 / Sec. S6 can be trusted.

This script changes exactly two things and nothing else:
  1. HARD initial conditions via the ansatz
         y(z) = y0 + (1 + z) * N_Theta(z),      z in [-1, 1], z = -1 <=> tau_min
     so y(-1) = y0 identically and the L_ic term disappears from the loss;
  2. a smooth cosine learning-rate decay 1e-3 -> 1e-4 instead of StepLR, which
     otherwise leaves the optimizer at lr = 6e-5 long before convergence.

Everything else (architecture, dtype, collocation sampling, residual loss) is
identical to common.pinn_losses, so the comparison is clean.

LOAD LIMIT: defaults are ~3000 iters at ncol=256 (~20 ms/iter measured) ~= 60 s.
"""
import argparse
import json
import math
import time

import numpy as np
import torch
from scipy.integrate import solve_ivp

from common import MLP, Params, Y0, rhs_t, rhs_np, to_phys_tau, grad1, env_info


def solution(net, z, P):
    """hard-IC ansatz: y(-1) = Y0 exactly"""
    y0 = torch.tensor(Y0, dtype=z.dtype).view(1, 3)
    return y0 + (1.0 + z) * net(z)


def residual_loss(net, z, P):
    s = P.tau_cut
    tau = to_phys_tau(z, P)
    y = solution(net, z, P)
    th, p, Sg = y[:, 0:1], y[:, 1:2], y[:, 2:3]
    dth = grad1(th, z) / s
    dp = grad1(p, z) / s
    dSg = grad1(Sg, z) / s
    f_th, f_p, f_Sg = rhs_t(tau, th, p, P)
    return ((dth - f_th) ** 2).mean() + ((dp - f_p) ** 2).mean() \
        + ((dSg - f_Sg) ** 2).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adam", type=int, default=3000)
    ap.add_argument("--lbfgs", type=int, default=200)
    ap.add_argument("--ncol", type=int, default=256)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--depth", type=int, default=5)
    ap.add_argument("--resample", type=int, default=200)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default="ansatz.json")
    a = ap.parse_args()
    torch.set_num_threads(min(a.threads, 4))
    torch.manual_seed(a.seed)

    P = Params()
    tau_ref = np.linspace(-P.tau_cut, P.tau_cut, 2001)
    ref = solve_ivp(rhs_np, (-P.tau_cut, P.tau_cut), Y0, args=(P,),
                    method="DOP853", rtol=1e-10, atol=1e-12, t_eval=tau_ref)
    y_ref = ref.y.T

    net = MLP(width=a.width, depth=a.depth, seed=a.seed).double()

    def sample(n):
        z = (2.0 * torch.rand(n, 1, dtype=torch.float64) - 1.0)
        return z.requires_grad_(True)

    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.adam,
                                                       eta_min=1e-4)
    z = sample(a.ncol)
    t0 = time.perf_counter()
    for it in range(a.adam):
        if a.resample and it % a.resample == 0 and it > 0:
            z = sample(a.ncol)
        opt.zero_grad(set_to_none=True)
        L = residual_loss(net, z, P)
        L.backward()
        opt.step()
        sched.step()
        if it % max(1, a.adam // 10) == 0 or it == a.adam - 1:
            print(f"  adam {it:6d}  res={float(L.detach()):.3e}", flush=True)
    t_adam = time.perf_counter() - t0

    ncl = [0]
    t_lb = 0.0
    if a.lbfgs > 0:
        z = sample(a.ncol)
        opt2 = torch.optim.LBFGS(net.parameters(), max_iter=a.lbfgs,
                                 history_size=50, tolerance_grad=1e-14,
                                 tolerance_change=1e-16,
                                 line_search_fn="strong_wolfe")

        def closure():
            ncl[0] += 1
            opt2.zero_grad(set_to_none=True)
            L = residual_loss(net, z, P)
            L.backward()
            return L
        t0 = time.perf_counter()
        opt2.step(closure)
        t_lb = time.perf_counter() - t0
        print(f"  lbfgs closures={ncl[0]}  res="
              f"{float(residual_loss(net, z, P).detach()):.3e}", flush=True)

    zz = torch.tensor(tau_ref / P.tau_cut, dtype=torch.float64).view(-1, 1)
    zz.requires_grad_(True)
    y = solution(net, zz, P).detach().numpy()
    err = np.abs(y - y_ref)
    rec = dict(adam=a.adam, lbfgs_closures=ncl[0], ncol=a.ncol,
               width=a.width, depth=a.depth,
               t_adam_s=t_adam, t_lbfgs_s=t_lb,
               ms_per_adam_iter=1e3 * t_adam / max(1, a.adam),
               ms_per_lbfgs_closure=1e3 * t_lb / max(1, ncl[0]),
               final_res=float(residual_loss(net, sample(4096), P).detach()),
               max_abs_all=float(err.max()),
               l2_rel_all=float(np.linalg.norm(y - y_ref) / np.linalg.norm(y_ref)),
               dSigma_pinn=float(y[-1, 2] - y[0, 2]),
               dSigma_ref=float(y_ref[-1, 2] - y_ref[0, 2]))
    print(json.dumps(rec, indent=2))
    with open(a.json, "w", encoding="utf-8") as f:
        json.dump(dict(env=env_info(), params=P.as_dict(), result=rec), f, indent=2)


if __name__ == "__main__":
    main()
