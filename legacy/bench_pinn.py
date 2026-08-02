# -*- coding: utf-8 -*-
"""
bench_pinn.py -- cost of ONE PINN training run on CPU + accuracy vs DOP853.

usage: python bench_pinn.py [--adam N] [--lbfgs N] [--ncol N] [--width W] [--depth D]
                            [--ablation full|noH|nosym|res] [--tag NAME] [--json OUT]
"""
import argparse
import json
import math
import time

import numpy as np
import torch
from scipy.integrate import solve_ivp

from common import (MLP, Params, Y0, pinn_losses, to_phys_tau, rhs_np,
                    peak_rss_mb, env_info)


def reference(P, n_eval=2001):
    t0 = time.perf_counter()
    tau_eval = np.linspace(-P.tau_cut, P.tau_cut, n_eval)
    sol = solve_ivp(rhs_np, (-P.tau_cut, P.tau_cut), Y0, args=(P,),
                    method="DOP853", rtol=1e-10, atol=1e-12, t_eval=tau_eval,
                    dense_output=False)
    dt = time.perf_counter() - t0
    assert sol.success, sol.message
    return tau_eval, sol.y.T, dt, sol.nfev


def train(P, args, seed=0, verbose=True):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(args.threads)

    use_H = args.ablation in ("full", "nosym")
    use_sym = args.ablation in ("full", "noH")

    net = MLP(width=args.width, depth=args.depth, seed=seed).double()
    npar = net.n_params()

    lam_H_max, lam_sym_max, lam_ic = 1.0, 1.0, 100.0

    def sample(n):
        z = (2.0 * torch.rand(n, 1, dtype=torch.float64) - 1.0)
        z.requires_grad_(True)
        return z

    def total(z, it, n_it):
        # annealed weights: 0 -> max over the first half of Adam
        frac = min(1.0, 2.0 * it / max(1, n_it))
        lH = lam_H_max * frac if use_H else 0.0
        ls = lam_sym_max * frac if use_sym else 0.0
        Lr, LH, Ls, Lic = pinn_losses(net, z, P, use_H=use_H, use_sym=use_sym)
        return Lr + lH * LH + ls * Ls + lam_ic * Lic, (Lr, LH, Ls, Lic)

    hist = []
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, args.adam // 4),
                                            gamma=0.5)
    z = sample(args.ncol)
    t_adam0 = time.perf_counter()
    for it in range(args.adam):
        if args.resample and it % args.resample == 0 and it > 0:
            z = sample(args.ncol)
        opt.zero_grad(set_to_none=True)
        L, parts = total(z, it, args.adam)
        L.backward()
        opt.step()
        sched.step()
        if verbose and (it % max(1, args.adam // 10) == 0 or it == args.adam - 1):
            hist.append((it, float(L)))
            print(f"  adam {it:6d}  L={float(L):.3e}  res={float(parts[0]):.2e} "
                  f"H={float(parts[1]):.2e} sym={float(parts[2]):.2e} "
                  f"ic={float(parts[3]):.2e}", flush=True)
    t_adam = time.perf_counter() - t_adam0

    t_lb = 0.0
    n_closure = [0]
    if args.lbfgs > 0:
        z = sample(args.ncol)
        opt2 = torch.optim.LBFGS(net.parameters(), max_iter=args.lbfgs,
                                 history_size=50, tolerance_grad=1e-12,
                                 tolerance_change=1e-14,
                                 line_search_fn="strong_wolfe")

        def closure():
            n_closure[0] += 1
            opt2.zero_grad(set_to_none=True)
            L, _ = total(z, args.adam, args.adam)
            L.backward()
            return L

        t0 = time.perf_counter()
        opt2.step(closure)
        t_lb = time.perf_counter() - t0
        L, parts = total(z, args.adam, args.adam)
        if verbose:
            print(f"  lbfgs  closures={n_closure[0]}  L={float(L):.3e} "
                  f"res={float(parts[0]):.2e} H={float(parts[1]):.2e} "
                  f"sym={float(parts[2]):.2e} ic={float(parts[3]):.2e}", flush=True)

    return net, npar, t_adam, t_lb, n_closure[0], hist


def evaluate(net, P, tau_ref, y_ref):
    with torch.no_grad():
        z = torch.tensor(tau_ref / P.tau_cut, dtype=torch.float64).view(-1, 1)
        y = net(z).numpy()
    err = np.abs(y - y_ref)
    return dict(max_abs_theta=float(err[:, 0].max()),
                max_abs_p=float(err[:, 1].max()),
                max_abs_Sigma=float(err[:, 2].max()),
                max_abs_all=float(err.max()),
                l2_rel_all=float(np.linalg.norm(y - y_ref) / np.linalg.norm(y_ref)),
                dSigma_pinn=float(y[-1, 2] - y[0, 2]),
                dSigma_ref=float(y_ref[-1, 2] - y_ref[0, 2]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adam", type=int, default=20000)
    ap.add_argument("--lbfgs", type=int, default=500)
    ap.add_argument("--ncol", type=int, default=2048)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--depth", type=int, default=5)
    ap.add_argument("--resample", type=int, default=500)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ablation", default="full",
                    choices=["full", "noH", "nosym", "res"])
    ap.add_argument("--tag", default="main")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    P = Params()
    tau_ref, y_ref, t_ref, nfev = reference(P)
    print(f"[ref] DOP853 rtol=1e-10 : {t_ref*1000:.2f} ms, nfev={nfev}", flush=True)

    t0 = time.perf_counter()
    net, npar, t_adam, t_lb, ncl, hist = train(P, args, seed=args.seed)
    t_tot = time.perf_counter() - t0
    acc = evaluate(net, P, tau_ref, y_ref)

    rec = dict(tag=args.tag, ablation=args.ablation, n_params=npar,
               adam_iters=args.adam, lbfgs_iters=args.lbfgs,
               lbfgs_closures=ncl, ncol=args.ncol, width=args.width,
               depth=args.depth, threads=args.threads, seed=args.seed,
               t_adam_s=t_adam, t_lbfgs_s=t_lb, t_total_s=t_tot,
               ms_per_adam_iter=1e3 * t_adam / max(1, args.adam),
               ms_per_lbfgs_closure=1e3 * t_lb / max(1, ncl),
               t_ref_ivp_s=t_ref, ivp_nfev=nfev,
               peak_rss_mb=peak_rss_mb(), **acc)
    print(json.dumps(rec, indent=2))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(dict(env=env_info(), params=P.as_dict(), result=rec), f,
                      indent=2)


if __name__ == "__main__":
    main()
