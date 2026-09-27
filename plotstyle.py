# -*- coding: utf-8 -*-
r"""
plotstyle.py -- one place for the APS figure conventions used by the five
make_fig*.py scripts of this deposit.

Constraints taken from the APS/REVTeX author guidelines and from
plan/06-figures-and-tables.md:

  * single-column artwork is 86 mm wide (8.6 cm = one column of the APS
    two-column layout used by Physical Review D); the figures
    are produced at exactly that width and included with
    \includegraphics[width=\columnwidth], so there is no reduction and the
    on-page font size equals the font size set here;
  * lettering >= 2 mm high: 8 pt has a cap height of ~2.0 mm, so 8 pt is the
    FLOOR for every piece of lettering, including tick labels, legend entries,
    colour-bar labels, in-axes annotations and inset axes.  Axis labels are
    9 pt.  Nothing in these figures is set below 8 pt;
  * no rule thinner than 0.5 pt anywhere.  The floor enforced here is 0.6 pt,
    which is the value this docstring has always claimed, and it now applies to
    marker edges, tick marks (major and minor), guide lines and axes frames as
    well as to the data lines;
  * the figures must survive greyscale printing, therefore every series is
    distinguished by MARKER SHAPE or LINE STYLE as well as by colour;
  * no titles inside the artwork (the caption lives in the LaTeX source);
  * no transparency anywhere: EPS has no alpha channel, and matplotlib would
    silently rasterise or flatten it.

The two floors are not left to the discipline of the calling scripts.  `audit()`
walks the finished figure and raises on any text below MIN_FONT_PT or any
visible stroke below MIN_LW_PT, and `save()` calls it before writing a file.  A
figure that violates the APS minima therefore cannot be produced by this
deposit at all.

Both PDF and EPS are written for every figure.  APS lists .ps/.eps/.tif/.jpg
as the preferred production formats, PDF is accepted for review; fonts are
embedded as Type 42 (TrueType) in both.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

MM = 1.0 / 25.4
COLW = 86.0 * MM                  # 86 mm = one APS column = 3.386 in

# APS minima, enforced by audit() below.  Do not lower them.
MIN_FONT_PT = 8.0                 # >= 2 mm cap height at 1:1 reproduction
MIN_LW_PT = 0.6                   # APS asks for >= 0.5 pt; 0.6 is our floor

# Colour-blind-safe qualitative set (Okabe-Ito), all of which also separate in
# greyscale when combined with the marker shapes used below.
BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
PURPLE = "#7B3294"
BLACK = "#000000"
GREY = "#595959"                  # dark enough to stay legible at 0.6 pt

RC = {
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "font.size": MIN_FONT_PT,
    "axes.labelsize": 9.0,
    "axes.titlesize": 9.0,
    "legend.fontsize": MIN_FONT_PT,
    "xtick.labelsize": MIN_FONT_PT,
    "ytick.labelsize": MIN_FONT_PT,
    "axes.linewidth": MIN_LW_PT,
    "grid.linewidth": MIN_LW_PT,
    "lines.linewidth": 0.9,
    "lines.markersize": 3.2,
    "lines.markeredgewidth": MIN_LW_PT,
    "xtick.major.width": MIN_LW_PT,
    "ytick.major.width": MIN_LW_PT,
    "xtick.minor.width": MIN_LW_PT,
    "ytick.minor.width": MIN_LW_PT,
    "xtick.major.size": 2.6,
    "ytick.major.size": 2.6,
    "xtick.minor.size": 1.5,
    "ytick.minor.size": 1.5,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "legend.frameon": False,
    "legend.handlelength": 1.4,
    "legend.handletextpad": 0.5,
    "legend.labelspacing": 0.25,
    "legend.borderpad": 0.2,
    "legend.columnspacing": 0.9,
    "savefig.dpi": 600,
    "figure.dpi": 110,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "pdf.compression": 9,
    "path.simplify": False,
    "axes.unicode_minus": False,
}


def use_style():
    plt.rcParams.update(RC)


# --------------------------------------------------------------------- audit
def _viol_text(obj):
    if not obj.get_visible() or not str(obj.get_text()).strip():
        return None
    fs = float(obj.get_fontsize())
    if fs < MIN_FONT_PT - 1e-9:
        return f"text {str(obj.get_text())[:28]!r} at {fs:.2f} pt"
    return None


def _viol_line(obj):
    if not obj.get_visible():
        return None
    out = []
    ls = obj.get_linestyle()
    lw = float(obj.get_linewidth())
    if ls not in ("None", "none", " ", "") and lw > 0.0 and lw < MIN_LW_PT - 1e-9:
        out.append(f"line lw {lw:.2f} pt")
    mk = obj.get_marker()
    mew = float(obj.get_markeredgewidth())
    if mk not in ("None", "none", " ", "", None) and 0.0 < mew < MIN_LW_PT - 1e-9:
        out.append(f"marker edge {mew:.2f} pt")
    return "; ".join(out) or None


def _viol_patch(obj):
    if not obj.get_visible():
        return None
    lw = float(obj.get_linewidth())
    if 0.0 < lw < MIN_LW_PT - 1e-9:
        return f"patch edge {lw:.2f} pt"
    return None


def _viol_coll(obj):
    if not obj.get_visible():
        return None
    for lw in obj.get_linewidths():
        if 0.0 < float(lw) < MIN_LW_PT - 1e-9:
            return f"collection edge {float(lw):.2f} pt"
    return None


def audit(fig, stem=""):
    """Raise unless every stroke is >= MIN_LW_PT and every glyph >= MIN_FONT_PT.

    The figure has to be fully laid out, so call this after the last artist has
    been added.  Returns the number of artists inspected.
    """
    import matplotlib.collections as mcoll
    import matplotlib.lines as mlines
    import matplotlib.patches as mpatches
    import matplotlib.text as mtext

    fig.canvas.draw()                      # realise tick artists and layout
    bad, seen = [], 0
    for obj in fig.findobj():
        seen += 1
        if isinstance(obj, mtext.Text):
            v = _viol_text(obj)
        elif isinstance(obj, mlines.Line2D):
            v = _viol_line(obj)
        elif isinstance(obj, mcoll.Collection):
            v = _viol_coll(obj)
        elif isinstance(obj, mpatches.Patch):
            v = _viol_patch(obj)
        else:
            v = None
        if v:
            bad.append(v)
    if bad:
        uniq = sorted(set(bad))
        raise ValueError(
            f"{stem or 'figure'} violates the APS minima "
            f"({MIN_FONT_PT:g} pt lettering, {MIN_LW_PT:g} pt rules): "
            + "; ".join(uniq[:12])
            + (f" ... and {len(uniq) - 12} more" if len(uniq) > 12 else ""))
    return seen


def save(fig, stem, outdir, png_dir=None):
    """Write <stem>.pdf and <stem>.eps into outdir, and a PNG proof.

    The canvas is saved at its declared size (no bbox_inches='tight'), so the
    file really is 86 mm wide and the on-page font sizes are the ones set in RC.
    Margins are the responsibility of the calling script.  `audit()` runs first,
    so no file is written unless the APS minima are met.
    """
    import os
    audit(fig, stem)
    os.makedirs(outdir, exist_ok=True)
    paths = []
    for ext in ("pdf", "eps"):
        p = os.path.join(outdir, f"{stem}.{ext}")
        fig.savefig(p, format=ext)
        paths.append(p)
    if png_dir:
        os.makedirs(png_dir, exist_ok=True)
        p = os.path.join(png_dir, f"{stem}.png")
        fig.savefig(p, format="png")
        paths.append(p)
    return paths
