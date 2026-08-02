# Changelog

All notable changes to this deposit. Versions follow the Zenodo releases; the
concept DOI [10.5281/zenodo.21758393](https://doi.org/10.5281/zenodo.21758393)
always resolves to the latest one.

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
