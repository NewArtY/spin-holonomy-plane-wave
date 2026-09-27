# -*- coding: utf-8 -*-
r"""
make_figS2.py -- FIG. S2 of the Supplemental Material of
"Net electron spin rotation in a plane-wave pulse: Holonomy set by the
anomalous magnetic moment" (Physical Review D).

WHAT THE FIGURE SHOWS
---------------------
In a plane wave the net rotation carries no carrier-envelope-phase dependence at
all.  Finite focusing brings it back.  For linear polarisation at
eps = 0.15 (w0 = 1.06 lambda), gamma = 10, N = 1 and an electron on axis, the
spin rotation relative to the momentum is a pure first harmonic in phi0, and the
figure gives the amplitude of that harmonic against the intensity parameter a0:

  * filled circles  A_1(r_0), the g-independent coefficient.  It is linear in a0
                    up to a0 ~ 2, passes through a minimum near a0 ~ 4 that we
                    cannot explain, and then grows as a0^2.6 -- a0^3.2, reaching
                    0.781 rad at a0 = 24;
  * open squares    A_1(r_1), the first order in the anomaly a_e, linear in a0
                    up to a0 ~ 12 (measured local slopes 0.99 down to 0.85) and
                    steepening to 2.0 over the last interval;
  * right-hand axis the ponderomotive kick |Delta u_perp| of the same runs, which
                    is what the g-independent rotation follows.

The minimum near a0 = 4 is marked as unexplained.

DATA
----
code/focused_beam.json, stage `a0big`, series `a0` (the eps = 0.15 CEP scans at
a0 = 1..24; the a0 = 1 point is shared with stages `cep` and `cepscan`).

Usage
    python make_figS2.py --data focused_beam.json --outdir ../manuscript
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

import plotstyle as ps


def build(data):
    ps.use_style()
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(ps.COLW, 2.85))
    ax = fig.add_axes([0.170, 0.150, 0.690, 0.820])

    rows = sorted(data["a0big"]["a0"], key=lambda r: r["a0"])
    a0 = np.array([r["a0"] for r in rows])
    A0 = np.array([r["r0_A1_abs"] for r in rows])
    A1 = np.array([r["r1_A1_abs"] for r in rows])
    du = np.array([r["du_perp"] for r in rows])

    axr = ax.twinx()
    axr.plot(a0, du, ls="--", lw=0.8, marker="v", ms=2.8, color=ps.GREY,
             mfc="none", mec=ps.GREY, mew=0.6, zorder=2)
    axr.set_yscale("log")
    axr.set_ylim(5e-5, 5.0)
    axr.set_ylabel(r"$|\Delta u_\perp|$", color=ps.GREY, labelpad=1.0)
    axr.tick_params(axis="y", colors=ps.GREY, direction="in")
    axr.spines["right"].set_color(ps.GREY)

    ax.plot(a0, A0, ls="-", lw=0.9, marker="o", ms=3.4, color=ps.ORANGE,
            mfc=ps.ORANGE, mec=ps.BLACK, mew=ps.MIN_LW_PT, zorder=4,
            label=r"$A_1(r_0)$")
    ax.plot(a0, A1, ls=":", lw=0.9, marker="s", ms=3.4, color=ps.BLUE,
            mfc="none", mec=ps.BLUE, mew=0.7, zorder=4, label=r"$A_1(r_1)$")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.85, 32.0)
    ax.set_ylim(3e-4, 3.0)
    ax.set_xlabel(r"intensity parameter $a_0$", labelpad=1.0)
    ax.set_ylabel(r"first-harmonic CEP amplitude  (rad)", labelpad=1.0)
    ax.set_xticks([1, 2, 4, 8, 16, 24])
    ax.set_xticklabels(["1", "2", "4", "8", "16", "24"])
    ax.set_xticks([], minor=True)
    ax.set_zorder(axr.get_zorder() + 1)
    ax.patch.set_visible(False)

    imin = int(np.argmin(A0))
    ax.annotate("unexplained\nminimum", xy=(a0[imin], A0[imin] * 0.72),
                xytext=(2.0, 6.0e-4), fontsize=ps.MIN_FONT_PT, color=ps.GREY,
                ha="center", va="center", linespacing=1.1,
                arrowprops=dict(arrowstyle="->", lw=0.5, color=ps.GREY,
                                mutation_scale=6))
    ax.plot([a0[imin]], [A0[imin]], ls="none", marker="o", ms=6.0,
            mfc="none", mec=ps.GREY, mew=0.6, zorder=5)

    from matplotlib.lines import Line2D
    handles = [Line2D([], [], ls="-", lw=0.9, marker="o", ms=3.4,
                      color=ps.ORANGE, mfc=ps.ORANGE, mec=ps.BLACK, mew=ps.MIN_LW_PT,
                      label=r"$A_1(r_0)$"),
               Line2D([], [], ls=":", lw=0.9, marker="s", ms=3.4,
                      color=ps.BLUE, mfc="none", mec=ps.BLUE, mew=0.7,
                      label=r"$A_1(r_1)$"),
               Line2D([], [], ls="--", lw=0.8, marker="v", ms=2.8,
                      color=ps.GREY, mfc="none", mec=ps.GREY, mew=0.6,
                      label=r"$|\Delta u_\perp|$ (right)")]
    ax.legend(handles=handles, loc="upper left",
              borderaxespad=0.35, handletextpad=0.5, labelspacing=0.2)
    return fig


def main(argv=None):
    ap = argparse.ArgumentParser(description="Draw FIG. S2 of the Supplement.")
    ap.add_argument("--data", default="focused_beam.json")
    ap.add_argument("--outdir", default=os.path.join("..", "manuscript"))
    ap.add_argument("--png-dir", default="figproofs")
    args = ap.parse_args(argv)

    with open(args.data, encoding="utf-8") as fh:
        data = json.load(fh)
    fig = build(data)
    for p in ps.save(fig, "figS2", args.outdir, args.png_dir):
        print("written", p)
    sc = data["a0big"]["a0_scaling"]
    print("local slopes A1(r0):",
          ", ".join(f"{s[2]:.2f}" for s in sc["r0"]["local"]))
    print("local slopes A1(r1):",
          ", ".join(f"{s[2]:.2f}" for s in sc["r1"]["local"]))


if __name__ == "__main__":
    main()
