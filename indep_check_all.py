# -*- coding: utf-8 -*-
r"""
indep_check_all.py -- driver for the four independent verification scripts.

QUESTION
--------
The results of the article rest on one code base (spin_magnitude.py for the plane
wave, focused_beam.py for the focused pulse).  A null result obtained from a
single implementation is worth little, so the theorem and the holonomy law were
re-derived and re-implemented from scratch, in covariant bivector form, in
indep_check.py and its three companions:

  indep_check.py    T1  g = 2 exactly: the net map must be the identity for every
                        pulse shape (linear, elliptical, circular, unipolar
                        potential, rotating polarisation plane, figure-eight) and
                        every initial four-velocity.
  indep_check2.py   T2  linear polarisation at ARBITRARY anomaly, including
                        a_e = 1, a_e = -0.7: still the identity.
                    T3  the broken hypothesis alpha(+inf) != alpha(-inf): the net
                        map is NOT the identity, which shows the test has power.
  indep_check3.py   T4  the a_e^2 law: Theta_net/a_e^2 constant, and
                        Theta_net = -(1/2) a_e^2 A against the measured area.
  indep_check4.py   T5  reparameterisation invariance: eta -> s(eta) at fixed
                        curve leaves Theta_net unchanged, which is the holonomy
                        statement itself.

The four scripts print a report; this driver runs them with a fixed seed,
captures the report into indep_check.log, and stores the numeric records that
indep_check.report() collects, together with the environment, in a JSON file so
that the numbers quoted in the Supplemental Material can be traced.

Usage
    python indep_check_all.py --out indep_check.json
Runtime: ~90 s on one core (measured).  Single process.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import platform
import runpy
import sys
import time

import numpy as np
import scipy

import indep_check

SEED = 20260730
SCRIPTS = ("indep_check.py", "indep_check2.py", "indep_check3.py",
           "indep_check4.py")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--out", default="indep_check.json")
    ap.add_argument("--log", default="indep_check.log")
    ap.add_argument("--scripts", default=",".join(SCRIPTS))
    args = ap.parse_args(argv)

    here = os.path.dirname(os.path.abspath(__file__))
    res = dict(meta=dict(script="indep_check_all.py", seed=SEED,
                         python=sys.version.split()[0], numpy=np.__version__,
                         scipy=scipy.__version__, os=platform.platform()),
               runs={})
    text = []
    for name in [s.strip() for s in args.scripts.split(",") if s.strip()]:
        np.random.seed(SEED)
        indep_check.RESULTS.clear()
        buf = io.StringIO()
        t0 = time.perf_counter()
        with contextlib.redirect_stdout(buf):
            if name == "indep_check.py":
                # call the imported module, so that report() fills the RESULTS
                # list this driver reads; runpy would create a second copy of it
                indep_check.main_t1()
            else:
                runpy.run_path(os.path.join(here, name), run_name="__main__")
        dt = time.perf_counter() - t0
        out = buf.getvalue()
        text.append(f"### {name}   ({dt:.1f} s)\n{out}")
        res["runs"][name] = dict(wall_s=dt, n_records=len(indep_check.RESULTS),
                                 records=list(indep_check.RESULTS),
                                 stdout=out)
        print(f"[{name}] {dt:.1f} s, {len(indep_check.RESULTS)} recorded rows",
              flush=True)

    res["wall_s"] = sum(r["wall_s"] for r in res["runs"].values())
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    with open(args.log, "w", encoding="utf-8") as fh:
        fh.write("\n".join(text))
    print("written", args.out, "and", args.log, flush=True)
    return res


if __name__ == "__main__":
    main()
