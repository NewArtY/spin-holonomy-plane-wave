# -*- coding: utf-8 -*-
r"""
make_figS1.py -- FIG. S1 of the Supplemental Material of
"Net electron spin rotation in a plane-wave pulse: Holonomy set by the
anomalous magnetic moment" (Physical Review D).

WHAT THE FIGURE SHOWS
---------------------
Every null result quoted in the paper is limited by arithmetic, not by physics,
and this figure makes that visible.  The measured net spin rotation is plotted
against the Lorentz factor for three families:

  * filled circles  max |Theta_net| over the g = 2 scan at each gamma (two
                    envelopes, a0 = 0.1..10, N = 1 and 4, three CEPs).  Zero by
                    the theorem -- what is plotted is round-off, and it is FLAT
                    at 2.3-2.6e-13 rad over four decades in gamma;
  * filled squares  max |Theta_net| for LINEAR polarisation with the anomaly
                    inflated up to a_e = 10, i.e. four orders of magnitude above
                    the physical value.  Also identically zero;
  * grey triangles  max ||R^T R - 1||, the measured violation of orthogonality of
                    the transported triad in the same g = 2 runs: this is the
                    quantity that actually grows with gamma and that sets the
                    resolution;
  * dashed line     the gamma^2 cancellation ESTIMATE, anchored on the measured
                    invariant error at gamma = 1: the laboratory components of S
                    and u grow as gamma and the invariants are built from
                    cancellations among them, so a naive count gives gamma^2;
  * dotted line     what the invariant error MEASURES, gamma^p with p read off
                    the endpoints of the same series;
  * open circles    the circularly polarised SIGNAL at a0 = 5, N = 1, which is
                    flat to six digits over four decades in gamma.

ARTWORK FIXED 2026-08-02
------------------------
The gamma^2 guide is an estimate and the data do not follow it: it runs above
the measured invariant error by about two decades at the top of the range, and
the effective exponent of the measurement is p = 1.44, not 2.  The earlier
version of this figure drew the estimate alone, so the reader had no way to see
that the two disagree.  Both lines are now drawn and labelled, and the crossing
of the ESTIMATE with the circular signal -- which is what fixes the gamma at
which the laboratory formulation stops resolving the effect -- is marked at its
measured position, gamma = 5.8e3.  (An earlier caption said 2e4; that number
came from anchoring the guide on 1e-14 instead of on the measured 1.65e-12.)

DATA
----
code/spin_magnitude.json:
  key t1, sub-test (a)  ->  filled circles, grey triangles  (also code/sm_t1.json)
  key t1, sub-test (b)  ->  filled squares   (the anomaly-inflated linear scan;
                            the tail comment of supplement.tex points at key t2,
                            but t2 was run at the PHYSICAL anomaly, so the
                            "inflated" series has to come from t1/anyg_cases)
  key t3, gamma_scan_d1 ->  open circles     (equivalently code/sm_t3.json)

Usage
    python make_figS1.py --data spin_magnitude.json --outdir ../manuscript
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os

import numpy as np

import plotstyle as ps


def max_per_gamma(cases, key="theta"):
    d = collections.defaultdict(float)
    for c in cases:
        d[c["gamma"]] = max(d[c["gamma"]], abs(c[key]))
    g = np.array(sorted(d))
    return g, np.array([d[x] for x in g])


def endpoint_exponent(x, y):
    """Effective log-log exponent between the first and the last point."""
    return math.log(y[-1] / y[0]) / math.log(x[-1] / x[0])


def build(data):
    ps.use_style()
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(ps.COLW, 2.95))
    ax = fig.add_axes([0.178, 0.150, 0.800, 0.820])

    g2, y2 = max_per_gamma(data["t1"]["g2_cases"])
    _, yo = max_per_gamma(data["t1"]["g2_cases"], "orth_err")
    ga, ya = max_per_gamma(data["t1"]["anyg_cases"])
    sig = data["t3"]["gamma_scan_d1"]
    gs = np.array([r["gamma"] for r in sig])
    ys = np.array([abs(r["theta"]) for r in sig])

    p_eff = endpoint_exponent(g2, yo)
    gx = float((ys.mean() / yo[0]) ** 0.5)        # crossing of the gamma^2 guide

    # ---- the estimate (gamma^2) and what the invariant error actually does ---
    gg = np.logspace(0, 4.5, 80)
    ax.plot(gg, yo[0] * gg ** 2, ls=(0, (5, 2)), color=ps.GREY, lw=0.7, zorder=1)
    ax.plot(gg, yo[0] * gg ** p_eff, ls=(0, (1.2, 1.2)), color=ps.GREY, lw=0.7,
            zorder=1)

    ax.plot(gs, ys, ls=":", lw=0.9, marker="o", ms=3.6, mfc="none",
            mec=ps.ORANGE, mew=0.7, color=ps.ORANGE, zorder=4,
            label=r"circular signal, $a_0=5$")
    ax.plot(ga, ya, ls="-.", lw=0.8, marker="s", ms=3.2, color=ps.PURPLE,
            mfc=ps.PURPLE, mec=ps.BLACK, mew=ps.MIN_LW_PT, zorder=3,
            label=r"linear pol., $a_{e}\leq 10$")
    ax.plot(g2, y2, ls="-", lw=0.8, marker="o", ms=3.2, color=ps.BLUE,
            mfc=ps.BLUE, mec=ps.BLACK, mew=ps.MIN_LW_PT, zorder=3,
            label=r"$g=2$ scan")
    ax.plot(g2, yo, ls="none", marker="^", ms=3.4, color=ps.GREY,
            mfc="none", mec=ps.GREY, mew=0.7, zorder=2,
            label=r"$\|R^{\top}\!R-\mathbb{1}\|$")

    # the crossing that fixes the resolution limit
    ax.plot([gx], [yo[0] * gx ** 2], ls="none", marker="x", ms=4.0,
            mec=ps.BLACK, mew=0.7, zorder=5)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.6, 3.0e4)
    ax.set_ylim(1e-17, 3e-2)
    ax.set_xlabel(r"Lorentz factor $\gamma$", labelpad=1.0)
    ax.set_ylabel(r"$|\Theta_{\rm net}|$  (rad)", labelpad=1.0)
    ax.set_yticks([1e-16, 1e-13, 1e-10, 1e-7, 1e-4])

    ax.text(2.2, yo[0] * 2.2 ** 2 * 1.8, r"$\gamma^{2}$ estimate",
            fontsize=ps.MIN_FONT_PT, color=ps.GREY, rotation=28,
            ha="left", va="bottom", rotation_mode="anchor")
    ax.text(1.3e3, yo[0] * 1.3e3 ** p_eff * 2.6,
            rf"$\gamma^{{{p_eff:.2f}}}$ measured",
            fontsize=ps.MIN_FONT_PT, color=ps.GREY, rotation=21,
            ha="left", va="bottom", rotation_mode="anchor")
    ax.annotate(rf"$\gamma\simeq{gx / 1e3:.1f}\times10^{{3}}$",
                xy=(gx, yo[0] * gx ** 2), xytext=(6.0e2, 2.0e-2),
                fontsize=ps.MIN_FONT_PT, color=ps.BLACK, ha="center", va="top",
                arrowprops=dict(arrowstyle="->", lw=ps.MIN_LW_PT,
                                color=ps.BLACK, mutation_scale=6))
    ax.legend(loc="lower left", ncol=1, borderaxespad=0.35,
              handletextpad=0.5, labelspacing=0.2)
    return fig, dict(p_eff=p_eff, gx=gx, g2=g2, y2=y2, yo=yo, ys=ys)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Draw FIG. S1 of the Supplement.")
    ap.add_argument("--data", default="spin_magnitude.json")
    ap.add_argument("--outdir", default=os.path.join("..", "manuscript"))
    ap.add_argument("--png-dir", default="figproofs")
    args = ap.parse_args(argv)

    with open(args.data, encoding="utf-8") as fh:
        data = json.load(fh)
    fig, d = build(data)
    for p in ps.save(fig, "figS1", args.outdir, args.png_dir):
        print("written", p)
    y2, yo, ys = d["y2"], d["yo"], d["ys"]
    print(f"g = 2 angle floor: {y2.min():.4e} .. {y2.max():.4e} rad "
          f"(flat: last two {y2[-2]:.4e}, {y2[-1]:.4e})")
    print(f"invariant error:   {yo[0]:.4e} .. {yo[-1]:.4e}, "
          f"effective exponent {d['p_eff']:.3f}")
    print(f"gamma^2 estimate exceeds the measured invariant error by "
          f"{yo[0] * d['g2'][-1] ** 2 / yo[-1]:.1f}x "
          f"({math.log10(yo[0] * d['g2'][-1] ** 2 / yo[-1]):.2f} decades) "
          f"at gamma = {d['g2'][-1]:g}")
    print(f"circular signal:   {ys.min():.7e} .. {ys.max():.7e} rad "
          f"(flat to {abs(ys.max() / ys.min() - 1):.1e})")
    print(f"estimate meets the signal at gamma = {d['gx']:.3e}")


if __name__ == "__main__":
    main()
