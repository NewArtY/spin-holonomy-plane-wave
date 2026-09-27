# -*- coding: utf-8 -*-
r"""
make_fig2.py -- FIG. 2 of
"Net electron spin rotation in a plane-wave pulse: Holonomy set by the
anomalous magnetic moment" (Physical Review D).

(This script was make_fig1.py up to deposit v1.0.1, when the figure was FIG. 1.
Since v1.1.0 the inset abscissa is labelled a_e rho, rho being the radius of the
circular plateau, and the theory line (1/2) a_e^2 |A|; data and layout are
unchanged.)

WHAT THE FIGURE SHOWS
---------------------
Main panel: the measured net spin rotation |Theta_net| of every pulse
configuration against the signed area A swept by the vector potential, with the
holonomy law of Eq. (10),

    Theta_net = -(1/2) a_e^2 A,

overlaid as a solid line with no fitted parameter.  The x axis is symmetric-log,
so both helicities (A > 0 and A < 0) are on the same panel and the two families
with A = 0 have a place to sit; the line therefore appears as a "V" whose vertex
is at A = 0.  The two open families are the ones the law predicts to vanish
identically: linear polarisation, and the zero-area curve a_y ~ a_x^2 traced out
and back.  They are drawn at their MEASURED magnitude.

VERTICAL RANGE -- FIXED 2026-08-02
----------------------------------
The earlier version of this script clipped the panel at ylim = 2e-13.  That
removed 32 of the 42 zero-area points from the figure, i.e. it hid exactly the
evidence the figure exists to present: the smallest measured floor is
4.07e-17 rad, the largest 9.95e-12 rad.  The panel now runs from 1e-17 to 0.3 rad
and every point of the deposit is on it.  The cost is a vertical range of 17.7
decades, which is paid for by putting the inset, the colour bar and the legend
into the empty band between the null floor (below 1e-11) and the signal (above
1e-5) -- a band that is empty because the separation between the two is eleven
orders of magnitude, which is itself the result.

The solid line is drawn only over the range of |A| that was actually measured
(0.33 to 1294), so that it does not run down into the column of null points and
suggest a comparison that is not being made: for those families the law predicts
zero exactly, not a small number.

COLOUR
------
The colour of every filled symbol is the effective ellipticity |delta| of the
pulse, 0 for linear and 1 for circular.  For the polarisation-gated family,
whose instantaneous ellipticity sweeps through the pulse, the colour is the
ellipticity |(1-q)/(1+q)| of the residual net circular component, q being the
amplitude ratio of the two counter-rotating pulses.  The colour carries no
information beyond the parameters of the scan; it is there to show that the
collapse onto the line is not driven by polarisation.

Inset: the ratio Theta_net/Theta_law, Theta_law = -(1/2) a_e^2 A, for a flat-top
circularly polarised pulse against x = a_e a0, together with the resummation of
Eq. (12), (sqrt(1+x^2)-1)/(x^2/2).  It shows where the leading-order law starts
to lose accuracy: the expansion parameter is a_e a0, not a_e.

DATA
----
code/holonomy_scan.json, stages `main`, `null`, `inset`
(produced by `python holonomy_scan.py`).  Nothing in this script computes
physics; it only reads and draws.

Usage
    python make_fig2.py --data holonomy_scan.json --outdir ../manuscript
"""
from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np

import plotstyle as ps

ANO = r"a_{e}"                      # the anomaly, as it is set in the manuscript


def delta_eff(rec):
    """Ellipticity used for the colour scale."""
    if rec["family"] == "gate":                 # two counter-rotating circulars
        q = rec["p_q"]
        return abs((1.0 - q) / (1.0 + q))
    return abs(rec.get("p_delta", 1.0))


MARKERS = {          # envelope / shape  ->  (marker, label)
    "gauss": ("o", r"Gaussian"),
    "cos2": ("s", r"$\cos^{2}$"),
    "chirp": ("D", r"chirped"),
    "gate": ("^", r"gated"),
}
# families that are Gaussian-enveloped scans of something else
AS_GAUSS = ("gauss", "Nscan", "gamma", "cep")

YLO, YHI = 1.0e-17, 0.3             # every measured point is inside this range


def build(data):
    ps.use_style()
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib import cm, colors

    fig = plt.figure(figsize=(ps.COLW, 4.75))
    ax = fig.add_axes([0.180, 0.250, 0.800, 0.665])
    axr = fig.add_axes([0.180, 0.070, 0.800, 0.150], sharex=ax)

    main, null, ins = data["main"], data["null"], data["inset"]
    anom = main["anom"]

    # ---------------------------------------------------------------- theory
    # Drawn over the measured range of |A| only: outside it the line would run
    # into the null column, where the prediction is an exact zero.  The two
    # signs of A are separate branches; joining them would draw a spurious
    # horizontal floor across A = 0.
    aabs = [abs(r["area"]) for r in main["rows"]]
    lo, hi = min(aabs), max(aabs) * 1.6
    branch = np.logspace(math.log10(lo), math.log10(hi), 400)
    for sgn in (-1.0, 1.0):
        ax.plot(sgn * branch, 0.5 * anom ** 2 * branch, "-", color=ps.BLACK,
                lw=0.9, zorder=1)

    # ---------------------------------------------------------------- data
    norm = colors.Normalize(vmin=0.0, vmax=1.0)
    cmap = plt.get_cmap("viridis")

    groups = {"gauss": [], "cos2": [], "chirp": [], "gate": []}
    for r in main["rows"]:
        key = "gauss" if r["family"] in AS_GAUSS else r["family"]
        groups[key].append(r)

    for key, rows in groups.items():
        mk = MARKERS[key][0]
        x = [r["area"] for r in rows]
        y = [abs(r["rz"]) for r in rows]
        c = [cmap(norm(delta_eff(r))) for r in rows]
        ax.scatter(x, y, s=15 if mk != "D" else 13, marker=mk, c=c,
                   edgecolors=ps.BLACK, linewidths=ps.MIN_LW_PT, zorder=3)
        axr.scatter(x, [-r["rel_resid"] for r in rows],
                    s=13 if mk != "D" else 11, marker=mk, c=c,
                    edgecolors=ps.BLACK, linewidths=ps.MIN_LW_PT, zorder=3)

    # ------------------------------------------------------- the A = 0 sets
    ylin = [r["theta"] for r in null["linear"]]
    y8 = [r["theta"] for r in null["fig8"]]
    ax.scatter(np.zeros(len(ylin)), ylin, s=16, marker="o",
               facecolors="none", edgecolors=ps.ORANGE, linewidths=0.7, zorder=4)
    ax.scatter(np.zeros(len(y8)), y8, s=16, marker="s",
               facecolors="none", edgecolors=ps.BLUE, linewidths=0.7, zorder=4)

    # ---------------------------------------------------------------- axes
    ax.set_xscale("symlog", linthresh=1.0, linscale=0.40,
                  subs=[2, 3, 4, 5, 6, 7, 8, 9])
    ax.set_yscale("log")
    ax.set_xlim(-4.0e3, 4.0e3)
    ax.set_ylim(YLO, YHI)
    ax.tick_params(labelbottom=False)
    axr.set_xlabel(r"area functional $\mathcal{A}=\int(a_xa_y'-a_ya_x')\,d\eta$",
                   labelpad=1.0)
    # every run falls short of the law (rel_resid < 0), so plot -rel_resid
    assert all(r["rel_resid"] < 0 for r in main["rows"])
    axr.set_yscale("log")
    axr.set_ylim(2e-6, 5e-3)
    axr.set_yticks([1e-5, 1e-4, 1e-3])
    axr.set_ylabel(r"$1-\Theta_{\rm net}/\Theta_{\rm law}$",
                   labelpad=1.0, fontsize=ps.MIN_FONT_PT)
    ax.set_ylabel(r"$|\Theta_{\rm net}|$  (rad)", labelpad=1.0)
    ax.set_xticks([-1e3, -1e1, 0, 1e1, 1e3])
    ax.set_yticks([1e-16, 1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2])

    ax.annotate(r"$\mathcal{A}=0$ families:" "\n" "numerical floor",
                xy=(0.522, 0.105), xytext=(0.60, 0.115),
                xycoords="axes fraction", textcoords="axes fraction",
                fontsize=ps.MIN_FONT_PT, color=ps.GREY,
                ha="left", va="center", linespacing=1.15,
                arrowprops=dict(arrowstyle="->", lw=ps.MIN_LW_PT,
                                color=ps.GREY, mutation_scale=6))

    # ---------------------------------------------------------------- legend
    handles = [Line2D([], [], ls="-", lw=0.9, color=ps.BLACK,
                      label=r"$\frac{1}{2}" + ANO + r"^{2}|\mathcal{A}|$")]
    for key in ("gauss", "cos2", "chirp", "gate"):
        mk, lab = MARKERS[key]
        handles.append(Line2D([], [], ls="none", marker=mk, ms=3.2,
                              mfc=cmap(0.62), mec=ps.BLACK,
                              mew=ps.MIN_LW_PT, label=lab))
    handles += [
        Line2D([], [], ls="none", marker="o", ms=3.4, mfc="none",
               mec=ps.ORANGE, mew=0.7, label="linear"),
        Line2D([], [], ls="none", marker="s", ms=3.4, mfc="none",
               mec=ps.BLUE, mew=0.7, label=r"$a_y\!\propto\!a_x^{2}$"),
    ]
    ax.legend(handles=handles, loc="center right", bbox_to_anchor=(1.005, 0.545),
              ncol=1, borderaxespad=0.0, handletextpad=0.4, labelspacing=0.22)

    # ---------------------------------------------------------------- colourbar
    cax = fig.add_axes([0.222, 0.2856, 0.230, 0.0122])
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal",
                      ticks=[0.0, 0.5, 1.0])
    cb.outline.set_linewidth(ps.MIN_LW_PT)
    cb.dividers.set_visible(False)      # empty, but matplotlib gives it 0.3 pt
    cax.tick_params(labelsize=ps.MIN_FONT_PT, length=1.6,
                    width=ps.MIN_LW_PT, pad=1.2)
    cax.set_title(r"ellipticity $|\delta|$", fontsize=ps.MIN_FONT_PT, pad=2.0)

    # ---------------------------------------------------------------- inset
    axi = fig.add_axes([0.302, 0.5437, 0.220, 0.118])
    xs = np.logspace(-2, math.log10(1.3), 300)
    axi.plot(xs, (np.sqrt(1.0 + xs ** 2) - 1.0) / (0.5 * xs ** 2), "-",
             color=ps.BLACK, lw=0.8, zorder=1)
    axi.plot([r["x"] for r in ins["rows"]], [r["ratio"] for r in ins["rows"]],
             ls="none", marker="o", ms=2.8, mfc=ps.ORANGE, mec=ps.BLACK,
             mew=ps.MIN_LW_PT, zorder=3)
    axi.set_xscale("log")
    axi.set_xlim(1.4e-2, 1.6)
    axi.set_ylim(0.82, 1.05)
    axi.set_yticks([0.9, 1.0])
    axi.set_xticks([1e-2, 1e-1, 1e0])
    axi.tick_params(labelsize=ps.MIN_FONT_PT, pad=1.2)
    axi.set_xlabel(ANO.join(("$", r"\,\rho$")), fontsize=ps.MIN_FONT_PT,
                   labelpad=0.5)
    axi.set_ylabel(r"$|r_{\rm net}|/|\Theta_{\rm law}|$",
                   fontsize=ps.MIN_FONT_PT, labelpad=1.5)
    for s in axi.spines.values():
        s.set_linewidth(ps.MIN_LW_PT)
    return fig


def main(argv=None):
    ap = argparse.ArgumentParser(description="Draw FIG. 2.")
    ap.add_argument("--data", default="holonomy_scan.json")
    ap.add_argument("--outdir", default=os.path.join("..", "manuscript"))
    ap.add_argument("--png-dir", default="figproofs")
    args = ap.parse_args(argv)

    with open(args.data, encoding="utf-8") as fh:
        data = json.load(fh)
    fig = build(data)
    out = ps.save(fig, "fig2_holonomy", args.outdir, args.png_dir)
    m, n = data["main"], data["null"]
    nulls = [r["theta"] for r in n["linear"]] + [r["theta"] for r in n["fig8"]]
    print(f"{m['n']} configurations, max relative residual vs the holonomy law: "
          f"{m['max_rel_resid']:.2e}")
    print(f"A = 0 families: {len(nulls)} points, floors from {min(nulls):.2e} "
          f"to {max(nulls):.2e} rad "
          f"(linear {n['max_theta_linear']:.3e}, "
          f"figure-eight {n['max_theta_fig8']:.3e})")
    print(f"points outside the plotted range "
          f"[{YLO:.0e}, {YHI:.0e}]: "
          f"{sum(1 for y in nulls if not (YLO <= y <= YHI))}")
    print(f"inset max deviation from the resummation: "
          f"{data['inset']['max_dev_from_eq7']:.2e}")
    for p in out:
        print("written", p)


if __name__ == "__main__":
    main()
