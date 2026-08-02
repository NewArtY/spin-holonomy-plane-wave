# -*- coding: utf-8 -*-
"""
bench_ivp.py -- cost of the reference / production ODE solver.

Measures:
  1. one trajectory with solve_ivp DOP853 at several tolerances (numpy RHS)
  2. the same with a hand-written fixed-step RK4 and a 4th-order symplectic
     (Forest-Ruth / Yoshida) integrator, as used for FIG. 2 and Table S1
  3. serial cost of one point of a (phi0 x tau_p x E0) sweep
  4. multiprocessing speed-up of the sweep over 1/2/4 worker processes
  5. how solve_ivp cost scales with the DIMENSION of the state vector -- proxy
     for the classical-Cartesian (8 comps) and Fermi-Walker (20 comps)
     formulations of Sec. S3, which are not implemented yet.

LOAD LIMITS (this box is a 4-core i7-4770, must stay usable):
  * worker counts are hard-capped at (1, 2, 4) -- never oversubscribe;
  * --mp-points defaults to 64;
  * this module imports ONLY physics_np (no torch), so spawned workers start in
    ~0.2 s instead of re-importing torch (~2 s, ~300 MB) each.  Importing torch
    in 8 spawned workers is the most likely cause of the earlier machine hang.
  * any section can be skipped: --skip dop853,fixed,sweep,mp,dim

usage:  python bench_ivp.py [--mp-points 64] [--skip mp] [--json ivp.json]
"""
import argparse
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor

# keep BLAS/OpenMP single-threaded: the RHS is scalar python, threads only hurt
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
from scipy.integrate import solve_ivp

from physics_np import Params, Y0, rhs_np, field_np

MAX_WORKERS = (1, 2, 4)          # hard cap, do not raise on this machine


# ------------------------------------------------------------------ integrators
def rk4_fixed(P, n_steps):
    h = 2.0 * P.tau_cut / n_steps
    t = -P.tau_cut
    y = np.array(Y0, dtype=float)
    for _ in range(n_steps):
        k1 = np.array(rhs_np(t, y, P))
        k2 = np.array(rhs_np(t + h / 2, y + h / 2 * k1, P))
        k3 = np.array(rhs_np(t + h / 2, y + h / 2 * k2, P))
        k4 = np.array(rhs_np(t + h, y + h * k3, P))
        y = y + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
    return y


_FR = 1.0 / (2.0 - 2.0 ** (1.0 / 3.0))
_C = [_FR / 2, (1 - _FR) / 2, (1 - _FR) / 2, _FR / 2]
_D = [_FR, -2.0 ** (1.0 / 3.0) * _FR, _FR, 0.0]


def symplectic4(P, n_steps):
    """Forest-Ruth 4th-order splitting for H = cos(th)cos(p) + a(t) sin(th)cos(p).
    Non-separable H -> we use the standard 4th-order composition of the
    (explicit) Euler-A / Euler-B half maps evaluated at frozen a(t); this is the
    scheme referenced as the '4th-order symplectic integrator' in the plan."""
    h = 2.0 * P.tau_cut / n_steps
    t = -P.tau_cut
    th, p, Sg = Y0
    for _ in range(n_steps):
        for c, d in zip(_C, _D):
            a = field_np(t + c * h, P)
            th = th - c * h * math.sin(p) * (math.cos(th) + a * math.sin(th))
            if d != 0.0:
                a = field_np(t + c * h, P)
                p = p + d * h * math.cos(p) * (math.sin(th) - a * math.cos(th))
            Sg = Sg + c * h * (P.kT * math.sin(th) * math.cos(p)
                               + (P.g - 1.0) * a * math.cos(th))
        t += h
    return np.array([th, p, Sg])


# --------------------------------------------------- dimension-scaling proxy RHS
def rhs_dim(tau, y, P, nd):
    """Structurally identical driven-trig RHS with nd coupled components.
    Used ONLY to measure how solve_ivp wall time scales with state dimension,
    as a proxy for the 8-component Cartesian and 20-component Fermi-Walker
    systems of Sec. S3 (which are not implemented yet)."""
    a = field_np(tau, P)
    y = np.asarray(y)
    ysh = np.roll(y, 1)
    return (-np.sin(ysh) * (np.cos(y) + a * np.sin(y)) + 0.1 * a * np.cos(y)).tolist()


# ------------------------------------------------------------------ sweep worker
def one_point(arg):
    phi0, tau_p, E0, rtol = arg
    P = Params(E0=E0, tau_p=tau_p, phi0=phi0, tau_cut=max(2.0, 3.4 * tau_p))
    sol = solve_ivp(rhs_np, (-P.tau_cut, P.tau_cut), Y0, args=(P,),
                    method="DOP853", rtol=rtol, atol=rtol * 1e-2)
    dS = sol.y[2, -1] - sol.y[2, 0]
    return phi0, tau_p, E0, dS, math.sin(dS / 2.0) ** 2, sol.nfev


def sweep_serial(points, rtol):
    t0 = time.perf_counter()
    out = [one_point((p, tp, e, rtol)) for (p, tp, e) in points]
    return time.perf_counter() - t0, out


def sweep_mp(points, rtol, nworkers):
    args = [(p, tp, e, rtol) for (p, tp, e) in points]
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=nworkers) as ex:
        out = list(ex.map(one_point, args,
                          chunksize=max(1, len(args) // (2 * nworkers))))
    return time.perf_counter() - t0, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="ivp.json")
    ap.add_argument("--mp-points", type=int, default=64,
                    help="points in the sweep benchmark (capped at 128)")
    ap.add_argument("--sweep-rtol", type=float, default=1e-10)
    ap.add_argument("--skip", default="",
                    help="comma list of: dop853,fixed,sweep,mp,dim")
    a = ap.parse_args()
    skip = {s.strip() for s in a.skip.split(",") if s.strip()}
    n_mp = min(a.mp_points, 128)                      # hard cap
    from common import env_info                       # torch import: parent only
    rec = dict(env=env_info(), mp_points=n_mp, sweep_rtol=a.sweep_rtol)

    P = Params()
    ref = solve_ivp(rhs_np, (-P.tau_cut, P.tau_cut), Y0, args=(P,),
                    method="DOP853", rtol=1e-13, atol=1e-15)
    yref = ref.y[:, -1]
    rec["reference"] = dict(rtol=1e-13, nfev=int(ref.nfev), y_end=yref.tolist())

    if "dop853" not in skip:
        print("=== 1. solve_ivp DOP853, single trajectory, vs tolerance ===")
        tol_rows = []
        for rtol in (1e-6, 1e-8, 1e-10, 1e-12):
            for _ in range(2):
                solve_ivp(rhs_np, (-P.tau_cut, P.tau_cut), Y0, args=(P,),
                          method="DOP853", rtol=rtol, atol=rtol * 1e-2)
            n_rep = 20
            t0 = time.perf_counter()
            for _ in range(n_rep):
                s = solve_ivp(rhs_np, (-P.tau_cut, P.tau_cut), Y0, args=(P,),
                              method="DOP853", rtol=rtol, atol=rtol * 1e-2)
            dt = (time.perf_counter() - t0) / n_rep
            err = float(np.abs(s.y[:, -1] - yref).max())
            tol_rows.append(dict(rtol=rtol, ms=dt * 1e3, nfev=int(s.nfev),
                                 nsteps=int(len(s.t)), max_abs_err_end=err))
            print(f"  rtol={rtol:.0e}  {dt*1e3:8.3f} ms  nfev={s.nfev:6d} "
                  f"steps={len(s.t):5d}  |err|_end={err:.2e}", flush=True)
        rec["dop853"] = tol_rows

    if "fixed" not in skip:
        print("\n=== 2. fixed-step RK4 / symplectic-4 (python loop) ===")
        fix_rows = []
        for n in (500, 1000, 2000, 5000, 10000):
            t0 = time.perf_counter()
            y1 = rk4_fixed(P, n)
            t_rk = time.perf_counter() - t0
            t0 = time.perf_counter()
            y2 = symplectic4(P, n)
            t_sy = time.perf_counter() - t0
            fix_rows.append(dict(n_steps=n, rk4_ms=t_rk * 1e3, symp4_ms=t_sy * 1e3,
                                 rk4_err=float(np.abs(y1 - yref).max()),
                                 symp4_err=float(np.abs(y2 - yref).max())))
            print(f"  n={n:6d}  RK4 {t_rk*1e3:8.2f} ms err={np.abs(y1-yref).max():.2e}"
                  f"   SYMP4 {t_sy*1e3:8.2f} ms err={np.abs(y2-yref).max():.2e}",
                  flush=True)
        rec["fixed_step"] = fix_rows

    pts = None
    t_ser = None
    if "sweep" not in skip:
        print("\n=== 3. sweep cost (serial) ===")
        rng = np.random.default_rng(0)
        pts = [(float(rng.uniform(0, 2 * math.pi)),
                float(rng.uniform(0.3, 3.0)),
                float(rng.uniform(0.1, 1.0))) for _ in range(n_mp)]
        t_ser, out = sweep_serial(pts, a.sweep_rtol)
        per_pt = t_ser / len(pts)
        nfev_mean = float(np.mean([o[5] for o in out]))
        print(f"  {len(pts)} random (phi0,tau_p,E0) points serial: {t_ser:.2f} s "
              f"=> {per_pt*1e3:.2f} ms/point  <nfev>={nfev_mean:.0f}", flush=True)
        rec["sweep_serial"] = dict(n=len(pts), t_s=t_ser,
                                   ms_per_point=per_pt * 1e3, nfev_mean=nfev_mean)

    if "mp" not in skip and pts is not None:
        print("\n=== 4. multiprocessing scaling (max 4 workers) ===")
        mp_rows = []
        for nw in MAX_WORKERS:
            t, _ = sweep_mp(pts, a.sweep_rtol, nw)
            mp_rows.append(dict(workers=nw, t_s=t, speedup=t_ser / t,
                                ms_per_point=t / len(pts) * 1e3))
            print(f"  workers={nw}: {t:7.2f} s  speedup vs serial="
                  f"{t_ser/t:5.2f}x", flush=True)
        rec["sweep_mp"] = mp_rows

    if "dim" not in skip:
        print("\n=== 5. solve_ivp cost vs state dimension (S3 proxy) ===")
        dim_rows = []
        for nd in (3, 8, 20):
            y0 = np.linspace(0.1, 0.5, nd)
            for _ in range(2):
                solve_ivp(rhs_dim, (-P.tau_cut, P.tau_cut), y0, args=(P, nd),
                          method="DOP853", rtol=1e-10, atol=1e-12)
            n_rep = 10
            t0 = time.perf_counter()
            for _ in range(n_rep):
                s = solve_ivp(rhs_dim, (-P.tau_cut, P.tau_cut), y0, args=(P, nd),
                              method="DOP853", rtol=1e-10, atol=1e-12)
            dt = (time.perf_counter() - t0) / n_rep
            dim_rows.append(dict(ndim=nd, ms=dt * 1e3, nfev=int(s.nfev)))
            print(f"  ndim={nd:3d}  {dt*1e3:8.2f} ms  nfev={s.nfev:6d}", flush=True)
        rec["dim_scaling"] = dim_rows

    with open(a.json, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=2)
    print("written", a.json)


if __name__ == "__main__":
    main()
