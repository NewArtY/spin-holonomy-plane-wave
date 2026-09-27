# Changelog

All notable changes to this deposit. Versions follow the Zenodo releases; the
concept DOI [10.5281/zenodo.21758393](https://doi.org/10.5281/zenodo.21758393)
always resolves to the latest one.

## v1.1.0 — unreleased (prepared 2026-09-26)

Figures, documentation and one new check for the revised article, now under
consideration at **Physical Review D**. **No earlier computed result changed**:
every `*.json` and `*.log` file present in v1.0.1 is identical in v1.1.0; the
only new result files are `rr_ikt_check.json` and `rr_ikt_check.log`.

* **New `rr_ikt_check.py`** (stage `rr` of `reproduce_all.py`, 7 s). Two checks
  behind new statements of the article. (i) Radiation reaction: the Lorentz
  and spin equations with the Landau-Lifshitz force and Fermi-Walker transport,
  at `tau0` inflated to 1e-4..1e-3; the leading term leaves the rotation about
  the propagation direction unchanged (relative change < 1e-11) and adds an
  in-plane rotation equal to the transverse kick; the Schott term adds
  `-(1/2)(tau0 kappa)^2 A'`, with `A'` the signed area of the hodograph
  `a_perp'`, reproduced to five digits for an electron at rest and for one
  with `kappa` = 3 at the same `tau0 kappa`. (ii) The first-order integral of
  a field-dressed anomaly, `|int (da/max da) a_perp' d eta| / max|a_perp|`,
  for circular, linear and elliptical pulses of 0.5-4 cycles, maximized over
  the carrier-envelope phase (at most 0.94).
  This replaces the heuristic radiation-reaction estimate of the first
  version of the article (Sec. S7).

* **New FIG. 1.** `make_fig1.py` draws the geometry of the curve traced by the
  transverse vector potential and of its signed area (four panels). It reads no
  data file; the signed area of panel (b) is computed three ways (line
  integral, closed form, winding-number sum) and the script stops if they
  disagree.
* **Renumbered figure scripts**, to follow the article: the former
  `make_fig1.py` (holonomy scan) is now `make_fig2.py`, the former
  `make_fig2.py` (focused beam) is now `make_fig3.py`. Output files are named
  after the article's figure files: `fig1_curve`, `fig2_holonomy`,
  `fig3_focusing`; the PNG proofs in `figproofs/` follow.
* **FIG. 2 labels.** The inset abscissa is now `a_e rho`, `rho` being the radius
  of the circular plateau (the article's notation; the argument is still called
  `a0` in `holonomy_scan.FlatTop`), and the theory line is labelled
  `(1/2) a_e^2 |A|`, since the ordinate is `|Theta_net|`; the abscissa is
  labelled by the definition of `A`. **New lower panel**: the relative shortfall
  `1 - Theta_net/Theta_law` on a log scale, which shows the three bands, one per
  `a0`, of the next-order term. The theory line is drawn as two branches, one
  per sign of `A`; joined, they produced a spurious horizontal segment near
  `A = 0`. The annotation of the zero-area column reads "numerical floor".
  The inset ordinate is labelled `|r_net|/|Theta_law|`. Data unchanged.
* **FIG. 3 redrawn relative to the signal.** Every curve is now divided by the
  plane-wave signal |Theta_law| = (1/2) a_e^2 |A| at the physical anomaly:
  the background |r_0|/|Theta_law| (with its eps^2 continuation, which crosses
  the signal at w0 = 16 lambda), the relative error of the anomalous part
  |a_e r_1 + a_e^2 r_2|/|Theta_law| - 1, and the r_2 deviation, with reference
  lines at the signal and at 1% of it. The linear-polarisation points (no
  signal to normalise to) are no longer plotted. The ordinate reads
  "relative size (see legend)". Data unchanged.
* `reproduce_all.py` draws five figures; `make_fig1.py` takes no `--data`
  argument, also in `--quick` mode. New stage tag `rr`.
* Docstrings, `README.md` and `plotstyle.py`: journal changed from Physical
  Review A to Physical Review D; figure numbers updated; the README's pointer to
  a nonexistent "Sec. S3.7" for stage `t4` replaced by what `t4` actually
  checks; the note on the independent implementation now points to Sec. S4.1.
  The README's scope section no longer says that no radiation reaction is
  integrated: `rr_ikt_check.py` integrates it (classically, at inflated
  `tau0`); section pointers of `rr_ikt_check.py` follow the article (Secs.
  III G, VII, S7).
* All five figures redrawn with the pinned environment of `requirements.txt`
  (Matplotlib 3.10.9).
* `MANIFEST.sha256` regenerated.

## v1.0.1 — 2026-08-02

Tooling and documentation only. **No computed result changed**: every `*.json`
and `*.log` file is identical to v1.0.0, and so is every figure.

* `reproduce_all.py` — `--verify` no longer fails on a working copy. It walked
  every directory under `code/`, so a virtual environment created inside the
  package (which is what most IDEs do by default) made it report several hundred
  `EXTRA` files and exit nonzero. `MANIFEST_SKIP_DIRS` now leaves out two kinds
  of tree: what this package generates (`__pycache__/`, `quickrun/`) and what
  belongs to whoever is running it (`.venv/`, `venv/`, `.idea/`, `.vscode/`,
  `.git/`, and the usual tool caches). This was the only defect that could make
  an untouched copy of the deposit look corrupted.
* `README.md` — added the Zenodo badge and the concept DOI; corrected the
  description of what the manifest covers; pointed at this file.
* `CHANGELOG.md` — added.
* `MANIFEST.sha256` — regenerated (72 files) to cover `.gitignore` and the
  edited files above.

Anyone holding the v1.0.0 archive can keep using its results as they are; only
`--verify` and the README differ.

## v1.0.0 — 2026-08-02

First release, deposited with the article

> N. S. Akintsov, A. P. Nevecheria, S. N. Andreev, Q.-H. Qin,
> *Net electron spin rotation in a plane-wave pulse: Holonomy set by the
> anomalous magnetic moment*, submitted to Physical Review A.

Contains the plane-wave and focused-beam calculations, the four independent
re-derivations, the figure scripts, and the computed results behind every number
quoted in the article and its Supplemental Material.
