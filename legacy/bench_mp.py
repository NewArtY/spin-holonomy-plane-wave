# -*- coding: utf-8 -*-
"""
bench_mp.py -- decompose the multiprocessing cost of a parameter sweep into
(a) pool start-up (Windows 'spawn': one fresh interpreter per worker) and
(b) actual parallel throughput.

bench_ivp.py section 4 measures wall time only; with 64 points (~1.9 s of work)
the ~1 s pool start-up hides the real speed-up.  Here we measure start-up
separately so the production sweeps of Fig. 1 / Fig. S2 (10^3-10^4 points, where
start-up amortizes to nothing) can be budgeted honestly.

LOAD LIMIT: workers capped at (1,2,4) on this 4-core box.  Torch is never
imported (worker module = bench_ivp -> physics_np only).
"""
import json
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from bench_ivp import one_point, sweep_serial

WORKERS = (1, 2, 4)
N_POINTS = 128
RTOL = 1e-10


def _noop(x):
    return x


def pool_startup_s(nw):
    """time to create the pool and get one trivial task back"""
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=nw) as ex:
        list(ex.map(_noop, range(nw)))
    return time.perf_counter() - t0


def main():
    rng = np.random.default_rng(1)
    pts = [(float(rng.uniform(0, 2 * np.pi)),
            float(rng.uniform(0.3, 3.0)),
            float(rng.uniform(0.1, 1.0))) for _ in range(N_POINTS)]

    t_ser, _ = sweep_serial(pts, RTOL)
    print(f"serial: {N_POINTS} points in {t_ser:.2f} s "
          f"=> {t_ser/N_POINTS*1e3:.2f} ms/point", flush=True)

    rows = []
    for nw in WORKERS:
        t_up = pool_startup_s(nw)
        args = [(p, tp, e, RTOL) for (p, tp, e) in pts]
        t0 = time.perf_counter()
        with ProcessPoolExecutor(max_workers=nw) as ex:
            list(ex.map(one_point, args, chunksize=max(1, N_POINTS // (2 * nw))))
        t_all = time.perf_counter() - t0
        t_comp = max(1e-9, t_all - t_up)
        rows.append(dict(workers=nw, startup_s=t_up, wall_s=t_all,
                         compute_s=t_comp, speedup_wall=t_ser / t_all,
                         speedup_amortized=t_ser / t_comp))
        print(f"  workers={nw}: startup={t_up:5.2f} s  wall={t_all:6.2f} s "
              f"(speedup {t_ser/t_all:4.2f}x)  compute-only={t_comp:6.2f} s "
              f"(amortized speedup {t_ser/t_comp:4.2f}x)", flush=True)

    with open("mp.json", "w", encoding="utf-8") as f:
        json.dump(dict(n_points=N_POINTS, rtol=RTOL, serial_s=t_ser,
                       ms_per_point_serial=t_ser / N_POINTS * 1e3, rows=rows),
                  f, indent=2)
    print("written mp.json")


if __name__ == "__main__":
    main()
