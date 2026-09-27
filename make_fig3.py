# -*- coding: utf-8 -*-
r"""
make_fig3.py -- FIG. 3 of
"Net electron spin rotation in a plane-wave pulse: Holonomy set by the
anomalous magnetic moment" (Physical Review D).

(This script was make_fig2.py up to deposit v1.0.1, when the figure was FIG. 2.
From v1.1.0 every curve is expressed relative to the plane-wave signal.)

WHAT THE FIGURE SHOWS
---------------------
How a finite waist spoils the holonomy law, in units of the signal itself.  A
focused Gaussian pulse (a0 = 1, N = 1, circular polarisation, head-on collision
at gamma = 10, electron on axis) is compared with the plane-wave signal
|Theta_law| = (1/2) a_e^2 |A| at the PHYSICAL anomaly, against the diffraction
parameter eps = 1/(k w0):

  * squares    |r_0| / |Theta_law|: the g-independent rotation (background).
               It must be below 1 for the signal to dominate.  The dashed
               continuation is the eps^2 law anchored on the smallest computed
               eps; it crosses 1 at the waist quoted in Table II (extrapolated).
  * diamonds   Theta_anom / |Theta_law| - 1, the relative error of the area law
               for the anomalous part Theta_anom = |a_e r_1 + a_e^2 r_2|
               (Eq. (13) of the article).  It crosses the 1% line inside the
               computed range.
  * circles    |r_2 - r_2^pw| / |r_2^pw|, the deviation of the second-order
               coefficient alone.

Horizontal lines mark the signal level (1) and 1% of it.  The upper axis gives
the waist in wavelengths, w0/lambda = 1/(2 pi eps).

The rotation vector of the spin relative to the momentum is expanded as
r = r_0 + a_e r_1 + a_e^2 r_2 by five-point differences over five triads
carried along one shared orbit with anomalies a_e in {0, +-h, +-2h}, h = 0.05.

DATA
----
code/focused_beam.json, stage `eps`, series `g10_a1_cir` (produced by
`python focused_beam.py --stages eps`).  Nothing is integrated here; the
physical anomaly a_e = 1.15965218e-3 is inserted into the stored r_1, r_2.

Usage
    python make_fig3.py --data focused_beam.json --outdir ../manuscript
"""
from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np

import plotstyle as ps

AE = 1.15965218e-3          # physical electron anomaly


def curves(rows):
    rows = sorted(rows, key=lambda r: r["beam"]["eps"])
    e = np.array([r["beam"]["eps"] for r in rows])
    law = np.array([AE**2 * abs(r["r2_pw_z"]) for r in rows])
    r0 = np.array([np.linalg.norm(r["r0"]) for r in rows])
    anom = np.array([np.linalg.norm(AE * np.array(r["r1"]) + AE**2 * np.array(r["r2"]))
                     for r in rows])
    r2dev = np.array([abs(r["r2_dev_rel"]) for r in rows])
    return e, r0 / law, anom / law - 1.0, r2dev


def build(data):
    ps.use_style()
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(ps.COLW, 2.95))
    ax = fig.add_axes([0.165, 0.148, 0.810, 0.730])

    e, bg, anom, r2dev = curves(data["eps"]["g10_a1_cir"])
    assert np.all(anom > 0)

    # reference levels
    for lev, txt in ((1.0, "signal"), (0.01, "1% of signal")):
        ax.axhline(lev, color=ps.GREY, lw=ps.MIN_LW_PT, zorder=1)
        ax.text(0.0043, lev * 1.35, txt, fontsize=ps.MIN_FONT_PT, color=ps.GREY,
                ha="left", va="bottom")

    # eps^2 continuation of the background below the computed range
    ex = np.array([0.0045, e[0]])
    ax.plot(ex, bg[0] * (ex / e[0]) ** 2, "--", color=ps.BLUE, lw=0.8, zorder=2)
    ecross = e[0] * math.sqrt(1.0 / bg[0])
    ax.text(0.0060, 0.10, r"$\varepsilon^2$ continuation", fontsize=ps.MIN_FONT_PT,
            color=ps.BLUE, rotation=19, ha="left", va="top", rotation_mode="anchor")

    ax.plot(e, bg, ls="-", lw=0.8, marker="s", ms=3.2, color=ps.BLUE,
            mfc=ps.BLUE, mec=ps.BLACK, mew=ps.MIN_LW_PT, zorder=4,
            label=r"$|r_0|/|\Theta_{\rm law}|$")
    ax.plot(e, anom, ls="-", lw=0.8, marker="D", ms=2.9, color=ps.GREEN,
            mfc=ps.GREEN, mec=ps.BLACK, mew=ps.MIN_LW_PT, zorder=4,
            label=r"$\Theta_{\rm anom}/|\Theta_{\rm law}|-1$")
    ax.plot(e, r2dev, ls="-", lw=0.8, marker="o", ms=3.2, color=ps.ORANGE,
            mfc=ps.ORANGE, mec=ps.BLACK, mew=ps.MIN_LW_PT, zorder=4,
            label=r"$|r_2-r_2^{\rm pw}|/|r_2^{\rm pw}|$")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.004, 0.42)
    ax.set_ylim(1e-4, 1e5)
    ax.set_xlabel(r"diffraction parameter $\varepsilon=1/(kw_0)$", labelpad=1.0)
    ax.set_ylabel(r"relative size (see legend)", labelpad=1.0)
    ax.set_yticks([1e-4, 1e-2, 1e0, 1e2, 1e4])
    ax.set_xticks([0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4])
    ax.set_xticklabels(["0.005", "0.01", "0.02", "0.05", "0.1", "0.2", "0.4"])
    ax.set_xticks([], minor=True)

    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.0), ncol=1,
              borderaxespad=0.35, handletextpad=0.5, labelspacing=0.2)

    # ------------------------------------------------- upper axis: w0/lambda
    axt = ax.twiny()
    axt.set_xscale("log")
    axt.set_xlim(*ax.get_xlim())
    ticks = [0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.4]
    axt.set_xticks(ticks)
    axt.set_xticklabels([f"{1.0 / (2 * math.pi * t):.2g}" for t in ticks])
    axt.set_xticks([], minor=True)
    axt.set_xlabel(r"waist $w_0/\lambda$", labelpad=2.0)
    axt.tick_params(direction="in")
    return fig, dict(eps=e, background=bg, anomalous=anom, r2dev=r2dev,
                     eps_background_equals_signal=ecross)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Draw FIG. 3.")
    ap.add_argument("--data", default="focused_beam.json")
    ap.add_argument("--outdir", default=os.path.join("..", "manuscript"))
    ap.add_argument("--png-dir", default="figproofs")
    args = ap.parse_args(argv)

    with open(args.data, encoding="utf-8") as fh:
        data = json.load(fh)
    fig, c = build(data)
    for p in ps.save(fig, "fig3_focusing", args.outdir, args.png_dir):
        print("written", p)
    for i, e in enumerate(c["eps"]):
        print(f"eps={e:6.3f}  |r0|/law={c['background'][i]:9.3e}  "
              f"anom/law-1={c['anomalous'][i]:9.3e}  r2dev={c['r2dev'][i]:9.3e}")
    ec = c["eps_background_equals_signal"]
    print(f"background = signal (eps^2 continuation): eps = {ec:.4f}, "
          f"w0 = {1 / (2 * math.pi * ec):.1f} lambda")


if __name__ == "__main__":
    main()
