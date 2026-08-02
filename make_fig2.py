# -*- coding: utf-8 -*-
r"""
make_fig2.py -- FIG. 2 of
"Net electron spin rotation in a plane-wave pulse: Holonomy set by the
anomalous magnetic moment" (Physical Review A, Regular Article).

WHAT THE FIGURE SHOWS
---------------------
The boundary of validity of the holonomy law.  A focused Gaussian pulse is not a
plane wave, and the departure from Eq. (10) is plotted against the diffraction
parameter eps = 1/(k w0) on log-log axes for a0 = 1, N = 1, circular
polarisation, head-on collision at gamma = 10.  Three quantities are shown:

  * filled circles  relative deviation of the second-order coefficient r_2 from
                    its plane-wave value -(1/2) A nhat;
  * squares         |r_0|, the g-independent rotation, identically zero in a
                    plane wave (it is the Thomas-Wigner rotation that comes with
                    the ponderomotive kick);
  * triangles       |r_1|, the first order in the anomaly a_e, also identically zero
                    in a plane wave;
  * open circles    the deviation of r_2 for LINEAR polarisation, which stays at
                    the round-off floor for every eps.

The dashed guide has slope 2: every violation enters at order eps^2.  The upper
axis converts eps into the waist in wavelengths, w0/lambda = 1/(2 pi eps).

The rotation vector of the spin relative to the momentum is expanded as
r = r_0 + a_e r_1 + a_e^2 r_2 by five-point differences over five triads
carried along one shared orbit with anomalies a_e in {0, +-h, +-2h}, h = 0.05.

DATA
----
code/focused_beam.json, stage `eps`, series `g10_a1_cir` and `g10_a1_lin`
(produced by `python focused_beam.py --stages eps`).  Nothing is computed here.

Usage
    python make_fig2.py --data focused_beam.json --outdir ../manuscript
"""
from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np

import plotstyle as ps


def series(rows, key):
    e = np.array([r["beam"]["eps"] for r in rows])
    if key == "r2dev":
        y = np.array([abs(r["r2_dev_rel"]) for r in rows])
    else:
        y = np.array([r[key] for r in rows])
    o = np.argsort(e)
    return e[o], y[o]


def build(data):
    ps.use_style()
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(ps.COLW, 2.95))
    ax = fig.add_axes([0.180, 0.148, 0.795, 0.730])

    cir = data["eps"]["g10_a1_cir"]
    lin = data["eps"]["g10_a1_lin"]

    e, y2 = series(cir, "r2dev")
    _, y0 = series(cir, "r0_abs")
    _, y1 = series(cir, "r1_abs")
    el, y2l = series(lin, "r2dev")

    # slope-2 guide, anchored on the r_2 deviation at the smallest eps
    eg = np.array([0.02, 0.4])
    ax.plot(eg, y2[0] * (eg / e[0]) ** 2, "--", color=ps.GREY, lw=ps.MIN_LW_PT, zorder=1)
    ax.plot(eg, y0[0] * (eg / e[0]) ** 2, "--", color=ps.GREY, lw=ps.MIN_LW_PT, zorder=1)

    ax.plot(e, y2, ls="-", lw=0.8, marker="o", ms=3.2, color=ps.ORANGE,
            mfc=ps.ORANGE, mec=ps.BLACK, mew=ps.MIN_LW_PT, zorder=4,
            label=r"$|r_2/r_2^{\rm pw}-1|$")
    ax.plot(e, y0, ls="-.", lw=0.8, marker="s", ms=3.2, color=ps.BLUE,
            mfc=ps.BLUE, mec=ps.BLACK, mew=ps.MIN_LW_PT, zorder=3, label=r"$|r_0|$")
    ax.plot(e, y1, ls=":", lw=0.9, marker="^", ms=3.4, color=ps.GREEN,
            mfc=ps.GREEN, mec=ps.BLACK, mew=ps.MIN_LW_PT, zorder=3, label=r"$|r_1|$")
    ax.plot(el, y2l, ls="none", marker="o", ms=3.4, mfc="none",
            mec=ps.ORANGE, mew=0.7, zorder=4,
            label=r"linear pol.")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.02, 0.42)
    ax.set_ylim(8e-12, 2.5)
    ax.set_xlabel(r"diffraction parameter $\varepsilon=1/(kw_0)$", labelpad=1.0)
    ax.set_ylabel(r"departure from the plane-wave holonomy", labelpad=1.0)
    ax.set_yticks([1e-11, 1e-9, 1e-7, 1e-5, 1e-3, 1e-1])
    ax.set_xticks([0.02, 0.05, 0.1, 0.2, 0.4])
    ax.set_xticklabels(["0.02", "0.05", "0.1", "0.2", "0.4"])
    ax.set_xticks([], minor=True)

    ax.text(0.052, 3.0e-4, r"slope 2", fontsize=ps.MIN_FONT_PT, color=ps.GREY,
            rotation=32, ha="left", va="bottom", rotation_mode="anchor")
    ax.text(0.0215, 5.0e-11, r"round-off floor", fontsize=ps.MIN_FONT_PT, color=ps.GREY,
            ha="left", va="bottom")

    ax.legend(loc="upper left", ncol=2, borderaxespad=0.35,
              handletextpad=0.5, labelspacing=0.2, columnspacing=1.0)

    # ------------------------------------------------- upper axis: w0/lambda
    axt = ax.twiny()
    axt.set_xscale("log")
    axt.set_xlim(*ax.get_xlim())
    ticks = [0.025, 0.05, 0.1, 0.2, 0.4]
    axt.set_xticks(ticks)
    axt.set_xticklabels([f"{1.0 / (2 * math.pi * t):.2g}" for t in ticks])
    axt.set_xticks([], minor=True)
    axt.set_xlabel(r"waist $w_0/\lambda$", labelpad=2.0)
    axt.tick_params(direction="in")
    return fig


def main(argv=None):
    ap = argparse.ArgumentParser(description="Draw FIG. 2.")
    ap.add_argument("--data", default="focused_beam.json")
    ap.add_argument("--outdir", default=os.path.join("..", "manuscript"))
    ap.add_argument("--png-dir", default="figproofs")
    args = ap.parse_args(argv)

    with open(args.data, encoding="utf-8") as fh:
        data = json.load(fh)
    fig = build(data)
    for p in ps.save(fig, "fig2", args.outdir, args.png_dir):
        print("written", p)
    sc = data["eps"]["g10_a1_cir_scaling"]
    print("local log-log slopes  r2dev:",
          ", ".join(f"{s[2]:.2f}" for s in sc["r2dev"]["local"]))
    print("local log-log slopes  r0   :",
          ", ".join(f"{s[2]:.2f}" for s in sc["r0"]["local"]))
    print("linear-polarisation floor:",
          max(abs(r["r2_dev_rel"]) for r in data["eps"]["g10_a1_lin"]))


if __name__ == "__main__":
    main()
