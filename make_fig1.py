# -*- coding: utf-8 -*-
r"""
make_fig1.py -- FIG. 1 of
"Net electron spin rotation in a plane-wave pulse: Holonomy set by the
anomalous magnetic moment" (Physical Review D): geometry of the curve C traced
by the transverse vector potential in the polarization plane, and of its
signed area.

PANELS
------
(a) Why an area appears.  A small rectangular loop in the (a_x, a_y) plane.
    Along each leg the connection of Eq. (7) rotates the rest-frame spin about
    the in-plane axis  nhat x da_perp  (dashed arrows), i.e. da_perp turned by
    +90 deg about nhat.  Rotations about different axes do not commute, and the
    closed loop leaves a rotation about nhat through -a_e^2 times the enclosed
    area (second-order group commutator; the sign is the one of Eq. (10)).
(b) An elliptically polarized Gaussian pulse (delta = 0.5, N = 1, a0 = 1, CEP
    0), drawn with the pulse parameterization of Eq. (S2).  The curve leaves
    the origin, spirals out and back in, and closes by Eq. (1).  Grey levels
    are the winding number ell(x) of C about each point; the signed area is
    counted with that multiplicity, and  A = 2 * integral(ell d^2a).  The script
    checks this against the line integral A = int (a_x a_y' - a_y a_x') d eta
    and against the closed form delta/(1+delta^2) a0^2 sigma sqrt(pi).
(c) Linear polarization: C degenerates to a segment traced back and forth,
    A = 0.
(d) The zero-area family of the numerical scan, a_x = a0 h, a_y = a0 h^2
    with h the Gaussian envelope (class FigureEight in holonomy_scan.py): a
    parabolic arc traced out and back, A = 0.

Unlike the other figure scripts this one reads no data file: every curve is
computed here from the pulse formulas, and the signed area of panel (b) is
checked three ways before anything is drawn (the script stops if they
disagree).  The figure is produced by code only (APS forbids generative-AI
artwork) and passes the lettering/line-width audit of plotstyle.py.

Usage
    python make_fig1.py --outdir ../manuscript
"""
from __future__ import annotations

import argparse
import math
import os

import numpy as np

import plotstyle as ps

SIG_PER_N = math.pi / math.sqrt(math.log(2.0))   # sigma = 3.7734 N, as in the scan


# ----------------------------------------------------------------- pulses
def elliptical(eta, a0=1.0, N=1.0, delta=0.5, phi0=0.0):
    s = SIG_PER_N * N
    f = np.exp(-eta ** 2 / (2 * s ** 2))
    nrm = a0 / math.sqrt(1 + delta ** 2)
    return nrm * f * np.cos(eta + phi0), nrm * delta * f * np.sin(eta + phi0)


def linear(eta, a0=1.0, N=1.0):
    s = SIG_PER_N * N
    f = np.exp(-eta ** 2 / (2 * s ** 2))
    return a0 * f * np.cos(eta), 0 * eta


def parabola(eta, a0=1.0, N=1.0, curv=1.0):
    s = SIG_PER_N * N
    f = np.exp(-eta ** 2 / (2 * s ** 2))
    return a0 * f, curv * a0 * f ** 2


def signed_area(x, y, eta):
    """A = int (a_x a_y' - a_y a_x') d eta, i.e. twice the signed area."""
    dx, dy = np.gradient(x, eta), np.gradient(y, eta)
    return np.trapezoid(x * dy - y * dx, eta)


def winding(x, y, gx, gy):
    """Winding number of the closed polyline (x, y) about every grid point."""
    w = np.zeros_like(gx)
    for i in range(len(x) - 1):
        a1 = np.arctan2(y[i] - gy, x[i] - gx)
        a2 = np.arctan2(y[i + 1] - gy, x[i + 1] - gx)
        d = a2 - a1
        d = (d + np.pi) % (2 * np.pi) - np.pi
        w += d
    return np.rint(w / (2 * np.pi))


# ------------------------------------------------------------------ drawing
def arrow_along(ax, x, y, idx, color, size=7):
    """Direction arrowhead on the curve at sample idx."""
    ax.annotate("", xy=(x[idx + 1], y[idx + 1]), xytext=(x[idx], y[idx]),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=0.9,
                                mutation_scale=size, shrinkA=0, shrinkB=0))


def panel_label(ax, s):
    ax.text(0.03, 0.97, s, transform=ax.transAxes, ha="left", va="top",
            fontsize=9, fontweight="bold")


def build():
    ps.use_style()
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(ps.COLW, 3.40))
    # 2 x 2 grid, square axes, hand-placed so that nothing is rescaled
    W, H = fig.get_size_inches()
    side = 1.33                                    # inches
    x0, x1 = 0.36 / W, (0.36 + side + 0.33) / W
    y1, y0 = (H - 0.08 - side) / H, (H - 0.08 - 2 * side - 0.36) / H
    axa = fig.add_axes([x0, y1, side / W, side / H])
    axb = fig.add_axes([x1, y1, side / W, side / H])
    axc = fig.add_axes([x0, y0, side / W, side / H])
    axd = fig.add_axes([x1, y0, side / W, side / H])

    # ------------------------------------------------------------ panel (a)
    ax = axa
    L = 1.0
    corners = [(0, 0), (L, 0), (L, L), (0, L), (0, 0)]
    ax.fill([0, L, L, 0], [0, 0, L, L], color="#D9D9D9", lw=0, zorder=0)
    for (xa, ya), (xb, yb) in zip(corners[:-1], corners[1:]):
        ax.annotate("", xy=(xb, yb), xytext=(xa, ya),
                    arrowprops=dict(arrowstyle="-|>", color=ps.BLACK, lw=1.0,
                                    mutation_scale=8, shrinkA=0, shrinkB=0))
        # rotation axis nhat x da = da turned by +90 deg, drawn from mid-leg
        mx, my = 0.5 * (xa + xb), 0.5 * (ya + yb)
        dx, dy = (xb - xa) / L, (yb - ya) / L
        rx, ry = -dy, dx
        ax.annotate("", xy=(mx + 0.22 * rx, my + 0.22 * ry), xytext=(mx, my),
                    arrowprops=dict(arrowstyle="-|>", color=ps.ORANGE, lw=0.9,
                                    ls=(0, (2.2, 1.4)), mutation_scale=7,
                                    shrinkA=0, shrinkB=0))
    ax.text(0.5, 0.5, "area\n" r"$\mathcal{A}/2$", ha="center", va="center",
            fontsize=8)
    ax.text(0.5, -0.17, r"$d\mathbf{a}_{\perp}$", ha="center", va="center",
            fontsize=8, color=ps.BLACK)
    ax.text(1.04, 0.17, r"$\hat{\mathbf{n}}\times d\mathbf{a}_{\perp}$", ha="left",
            va="center", fontsize=8, color=ps.ORANGE)
    ax.set_xlim(-0.40, 1.40)
    ax.set_ylim(-0.62, 1.40)
    ax.set_aspect("equal")
    ax.axis("off")
    panel_label(ax, "(a)")
    # propagation direction nhat: out of the page
    ax.add_patch(plt.Circle((1.22, 1.22), 0.075, fill=False, lw=0.7, color=ps.BLACK))
    ax.plot([1.22], [1.22], "o", ms=1.6, color=ps.BLACK)
    ax.text(1.12, 1.22, r"$\hat{\mathbf{n}}$", ha="right", va="center", fontsize=8)
    ax.text(0.5, -0.45, r"net: $-a_{e}^{2}\mathcal{A}/2$ about $\hat{\mathbf{n}}$",
            ha="center", va="center", fontsize=8)

    # ------------------------------------------------------------ panel (b)
    ax = axb
    N, delta = 1.0, 0.5
    s = SIG_PER_N * N
    eta = np.linspace(-8 * s, 8 * s, 40001)
    x, y = elliptical(eta, N=N, delta=delta)
    A_line = signed_area(x, y, eta)
    A_closed = delta / (1 + delta ** 2) * s * math.sqrt(math.pi)
    # winding numbers on a grid (coarse polyline is enough for w)
    sub = slice(None, None, 20)
    xs, ys = x[sub], y[sub]
    xs = np.append(xs, xs[0]); ys = np.append(ys, ys[0])
    g = np.linspace(-1.0, 1.0, 501)
    gx, gy = np.meshgrid(g, g)
    w = winding(xs, ys, gx, gy)
    cell = (g[1] - g[0]) ** 2
    A_wind = 2 * w.sum() * cell
    print(f"panel (b): A line integral = {A_line:.6f}, closed form = "
          f"{A_closed:.6f}, 2*sum(w)*dA = {A_wind:.4f}, max w = {w.max():.0f}")
    if not (abs(A_line - A_closed) < 1e-6 * abs(A_closed)
            and abs(A_wind - A_line) < 2e-2 * abs(A_line)):
        raise SystemExit("signed-area cross-check failed")
    WCAP = 3                                   # levels >= 3 share one grey
    wmax = WCAP
    wshow = np.minimum(w, WCAP)
    greys = ["#FFFFFF", "#E3E3E3", "#BDBDBD", "#8F8F8F"]
    from matplotlib.colors import ListedColormap, BoundaryNorm
    cmap = ListedColormap(greys[: wmax + 1])
    norm = BoundaryNorm(np.arange(-0.5, wmax + 1.5), cmap.N)
    ax.pcolormesh(gx, gy, wshow, cmap=cmap, norm=norm, shading="auto",
                  rasterized=True, zorder=0)
    ax.plot(x, y, "-", color=ps.BLUE, lw=0.8, zorder=2)
    # direction arrows at a few phases on the outer turns
    for e0 in (-3.2, 0.4, 3.6):
        i = int(np.searchsorted(eta, e0))
        arrow_along(ax, x, y, i, ps.BLUE, size=7)
    ax.plot([0], [0], "o", ms=2.2, color=ps.BLACK, zorder=3)
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_aspect("equal")
    ax.set_xticks([-0.8, 0, 0.8]); ax.set_yticks([-0.8, 0, 0.8])
    ax.set_xlabel(r"$a_{x}$", labelpad=1)
    ax.set_ylabel(r"$a_{y}$", labelpad=0)
    panel_label(ax, "(b)")
    ax.text(0.97, 0.03, rf"$\mathcal{{A}}={A_line:.2f}$", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8)
    # winding-number key
    for k in range(1, wmax + 1):
        ax.add_patch(plt.Rectangle((0.58 + 0.13 * (k - 1), 0.84), 0.12, 0.12,
                                   transform=ax.transAxes, facecolor=greys[k],
                                   edgecolor=ps.BLACK, lw=0.6, zorder=4))
        ax.text(0.64 + 0.13 * (k - 1), 0.90, str(k) if k < WCAP else f"{k}+", transform=ax.transAxes,
                ha="center", va="center", fontsize=8, zorder=5,
                color="black")
    ax.text(0.56, 0.90, r"$\ell$", transform=ax.transAxes, ha="right",
            va="center", fontsize=8)

    # ------------------------------------------------------------ panel (c)
    ax = axc
    xl, yl = linear(eta, N=N)
    Al = signed_area(xl, yl, eta)
    ax.plot(xl, yl, "-", color=ps.BLUE, lw=0.9)
    ax.annotate("", xy=(0.75, 0.06), xytext=(-0.75, 0.06),
                arrowprops=dict(arrowstyle="<|-|>", color=ps.BLUE, lw=0.7,
                                mutation_scale=7))
    ax.plot([0], [0], "o", ms=2.2, color=ps.BLACK)
    ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1)
    ax.set_aspect("equal")
    ax.set_xticks([-1, 0, 1]); ax.set_yticks([-1, 0, 1])
    ax.set_xlabel(r"$a_{x}$", labelpad=1)
    ax.set_ylabel(r"$a_{y}$", labelpad=0)
    panel_label(ax, "(c)")
    ax.text(0.5, 0.25, "linear:\n" r"segment, $\mathcal{A}=0$",
            transform=ax.transAxes, ha="center", va="center", fontsize=8)
    print(f"panel (c): A = {Al:.2e}")

    # ------------------------------------------------------------ panel (d)
    ax = axd
    xp, yp = parabola(eta, N=N)
    Ap = signed_area(xp, yp, eta)
    ax.plot(xp, yp, "-", color=ps.BLUE, lw=0.9)
    i1 = int(np.searchsorted(eta, -1.2 * s))
    i2 = int(np.searchsorted(eta, 1.2 * s))
    ax.annotate("", xy=(xp[i1 + 400] + 0.0, yp[i1 + 400] + 0.07),
                xytext=(xp[i1] + 0.0, yp[i1] + 0.07),
                arrowprops=dict(arrowstyle="-|>", color=ps.BLUE, lw=0.8,
                                mutation_scale=7, shrinkA=0, shrinkB=0))
    ax.annotate("", xy=(xp[i2 + 400] + 0.0, yp[i2 + 400] - 0.07),
                xytext=(xp[i2] + 0.0, yp[i2] - 0.07),
                arrowprops=dict(arrowstyle="-|>", color=ps.BLUE, lw=0.8,
                                mutation_scale=7, shrinkA=0, shrinkB=0))
    ax.plot([0], [0], "o", ms=2.2, color=ps.BLACK)
    ax.set_xlim(-0.35, 1.2); ax.set_ylim(-0.35, 1.2)
    ax.set_aspect("equal")
    ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel(r"$a_{x}$", labelpad=1)
    ax.set_ylabel(r"$a_{y}$", labelpad=0)
    panel_label(ax, "(d)")
    ax.text(0.97, 0.03, "out and back:\n" r"$\mathcal{A}=0$",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8)
    print(f"panel (d): A = {Ap:.2e}")
    return fig


def main(argv=None):
    ap = argparse.ArgumentParser(description="Draw FIG. 1.")
    ap.add_argument("--outdir", default=os.path.join("..", "manuscript"))
    ap.add_argument("--png-dir", default="figproofs")
    args = ap.parse_args(argv)
    fig = build()
    for p in ps.save(fig, "fig1_curve", args.outdir, args.png_dir):
        print("written", p)


if __name__ == "__main__":
    main()
