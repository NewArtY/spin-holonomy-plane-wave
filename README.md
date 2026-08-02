# Net electron spin rotation in a plane-wave pulse — code and data

Computational supplement to

> N. S. Akintsov, A. P. Nevecheria, S. N. Andreev, Q.-H. Qin,
> *Net electron spin rotation in a plane-wave pulse: Holonomy set by the
> anomalous magnetic moment*,
> submitted to **Physical Review A** (Regular Article).
> Zenodo DOI: **to be inserted on deposit** — cite this deposit, not this file.

This directory contains everything needed to reproduce every number quoted in
the article and in its Supplemental Material, and to redraw FIG. 1, FIG. 2,
FIG. S1 and FIG. S2 from scratch. No figure in the paper was produced by hand,
by a drawing program, or by a generative model; each one is the output of the
`make_fig*.py` script listed below, reading a JSON file produced by one of the
computation scripts.

## What the code computes

An electron crosses a finite laser pulse. Its rest-frame spin is transported by
the Thomas–Bargmann–Michel–Telegdi equation along the exact Volkov orbit, and
the quantity of interest is the rotation that survives after the pulse has
passed. For a plane wave the code measures

    Theta_net = -(1/2) a_e^2 A + O(a_e^4),
    A         = int (a_x a_y' - a_y a_x') deta,

with `a_e = (g-2)/2` the anomalous magnetic moment and `A` the signed area that
the transverse potential sweeps in the polarisation plane, and it measures the
two families for which `A` vanishes identically. For a focused Gaussian pulse
the code measures how that law fails, as a function of the diffraction parameter
`eps = 1/(k w0)`.

**Notation.** `a_e` is the anomaly throughout (the manuscript macro is `\ano`);
`a_x`, `a_y`, `a_perp` are the components of the normalised vector potential and
`a0` is its amplitude. In the source, the anomaly is the argument named `anom`.

The measurements are all *numerical experiments*: an ODE is integrated to a
prescribed tolerance and the answer is read off. There is no fitting anywhere,
and every "zero" reported is a measured round-off floor, not an assumption.

## Layout

```
code/
  README.md                 this file
  LICENSE                   MIT (see "Licence" below)
  MANIFEST.sha256           integrity manifest; see "Checking the deposit"
  requirements.txt          exact package versions used
  reproduce_all.py          ONE ENTRY POINT: recompute everything, redraw everything

  spin_magnitude.py         plane-wave scans, stages t1..t7
  holonomy_scan.py          the (signed area, net rotation) data set of FIG. 1
  focused_beam.py           focused Gaussian pulse, Lax-Louisell-McKnight fields
  focused_beam_summary.py   pretty-prints the tables of focused_beam.json
  indep_check.py            independent re-implementation, T1
  indep_check2.py             "         "                 T2, T3
  indep_check3.py             "         "                 T4 and the a_e^2 law
  indep_check4.py             "         "                 T5, reparameterisation
  indep_check_all.py        driver: runs the four above, writes JSON + log

  plotstyle.py              APS figure conventions shared by the four make_fig*
  make_fig1.py              -> manuscript/fig1.pdf, fig1.eps
  make_fig2.py              -> manuscript/fig2.pdf, fig2.eps
  make_figS1.py             -> manuscript/figS1.pdf, figS1.eps
  make_figS2.py             -> manuscript/figS2.pdf, figS2.eps

  *.json                    the deposited results (see the table below)
  *.log                     human-readable transcripts of the same runs
  figproofs/                PNG proofs of the four figures, for screen reading
  quickrun/                 created by `reproduce_all.py --quick`; safe to delete

  legacy/                   a discarded line of work, kept as a record;
                            see legacy/README.md.  NOTHING in the paper uses it.
```

## Which script produces what

| Script | Output | Used by |
| :-- | :-- | :-- |
| `spin_magnitude.py --stages t1` | `sm_t1.json` | TABLE S4 rows 1–3; FIG. S1 filled circles, filled squares and grey triangles |
| `spin_magnitude.py --stages t2 --deltas 0,1` | `sm_t2.json` | the CEP bounds of Sec. S3.6 (the deposited `sm_t2lin.json` and `sm_t2cir.json` are the same grid split by polarisation, from the original session) |
| `spin_magnitude.py --stages t3` | `sm_t3.json` | TABLE S5 (scaling laws); FIG. S1 open circles |
| `spin_magnitude.py --stages t4` | `sm_t4.json` | the antisymmetry test of Sec. S3.7 |
| `spin_magnitude.py --stages t5` | `sm_t5.json` | TABLE S9 (applicability window), the working point of Sec. VII |
| `spin_magnitude.py --stages t6` | `sm_t6.json` | **TABLE S3** (integrator-tolerance scan, Sec. S4.3) |
| `spin_magnitude.py --stages t7` | `sm_t7.json` | **the `Delta S_par = O(a_e^6)` measurement** behind the helicity statement of Sec. III.E |
| `spin_magnitude.py --merge ...` | `spin_magnitude.json` | the merged file cited in the article and the Supplement |
| `holonomy_scan.py --stages main,null,inset` | `holonomy_scan.json` | **FIG. 1** (all three panels of data) |
| `holonomy_scan.py --stages wrap` | same file, key `wrap` | **the angle-wrapping case of Sec. S2.3** (4.96 rad predicted, 1.307 rad read from the matrix) |
| `focused_beam.py --stages field` | `focused_beam.json`, stage `field` | TABLE S6 (Maxwell residuals) |
| `focused_beam.py --stages eps,xcheck` | stages `eps`, `xcheck` | **FIG. 2**, TABLE S7 |
| `focused_beam.py --stages cep,cepscan,a0big` | stages `cep`, `cepscan`, `a0big` | **FIG. S2**, Sec. S5.6 |
| `focused_beam.py --stages conv,order,scans,window` | remaining stages | convergence, field-model systematics, TABLE S8 |
| `indep_check_all.py` | `indep_check.json`, `indep_check.log` | the independent-verification claims of Sec. S3 |
| `make_fig1.py` … `make_figS2.py` | `../manuscript/fig*.pdf`, `fig*.eps`, `figproofs/*.png` | the four figures |

`focused_beam_summary.py focused_beam.json` reprints the numeric tables that
were transcribed into the Supplemental Material; it is a reader, not a
computation.

## How to reproduce everything from nothing

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows;  source .venv/bin/activate elsewhere
pip install -r requirements.txt

python reproduce_all.py --quick    # ~6 min smoke test, writes only into quickrun/
python reproduce_all.py            # ~21 min, the published numbers and figures
```

`reproduce_all.py` runs the steps **strictly one at a time**, one child process
at a time, with BLAS threading pinned to one thread; none of the scripts uses
multiprocessing.

`--quick` **runs every stage of every script and all four independent checks**,
with the grids reduced wherever the script exposes a knob for it
(`spin_magnitude.py` stages t1–t7, `holonomy_scan.py --quick`, and
`focused_beam.py --eps-list --n-phi --field-eps --a0-list --a0big-list
--gamma-list --N-list --b-list`). `indep_check2/3/4` are standalone scripts with
no grid parameter, so they run at full size; they are most of the quick-mode
wall time. Everything `--quick` writes goes into `code/quickrun/`, so it can
never overwrite a deposited result, and the numbers and figures it produces are
a smoke test rather than the published results.

`--only` and `--skip` take the tags `plane`, `holonomy`, `focused`, `indep`,
`figs`; `python reproduce_all.py --only figs` redraws the four figures from the
JSON files already present, in about ten seconds.

Individual pieces can be run directly, for example

```bash
python spin_magnitude.py --stages t1 --out sm_t1.json
python spin_magnitude.py --stages t7 --out sm_t7.json
python holonomy_scan.py --stages main,null,inset,wrap --out holonomy_scan.json
python focused_beam.py --stages eps,xcheck --out fb_A.json
python make_fig1.py --data holonomy_scan.json --outdir ../manuscript
```

Every script has `--help`, a docstring that states the question it answers, and
a fixed seed (`SEED = 20260730`). The seed only affects reporting order: the
computation is deterministic, the ODE solver is DOP853 with prescribed
tolerances, and re-running reproduces the deposited JSON bit for bit on the same
platform. (Checked on 2026-08-02: stage t1 re-run after the grids were made
configurable returns all 258 records bit for bit, and `holonomy_scan.py` stages
`main`, `null` and `inset` likewise.) Across platforms the physical numbers agree
to the quoted digits and the round-off floors move in the last one or two.

One tolerance caveat is worth knowing before reading `sm_t6.json`: SciPy's
Runge–Kutta drivers refuse an `rtol` below `100 eps` and silently raise it, so
the `rtol = 1e-14` column of the tolerance scan is executed at `2.22e-14`. The
field `rtol_effective` in the JSON records what was actually used.

## Checking the deposit

```bash
python reproduce_all.py --verify     # check every file against MANIFEST.sha256
python reproduce_all.py --manifest   # regenerate it after an intentional change
```

`MANIFEST.sha256` lists the SHA-256 digest and the size of every deposited file
in `code/`, excluding the generated trees `__pycache__/` and `quickrun/`.
`--verify` reports every file that is changed, missing or unexpected, and exits
nonzero if any are. Run it after downloading the archive and before running
anything: it is the only way to tell a truncated or edited copy from the
deposited one.

## Measured run times

Machine: Windows 10 Pro 22H2, Intel i7 laptop, CPython 3.13.2, one core in use.
All figures are wall-clock, taken from the `wall_s` fields of the deposited JSON
files.

| Step | Time |
| :-- | --: |
| `spin_magnitude.py` t1 | 84 s |
| `spin_magnitude.py` t2 (linear + circular) | 237 s |
| `spin_magnitude.py` t3 | 276 s |
| `spin_magnitude.py` t4 | 45 s |
| `spin_magnitude.py` t5 | < 1 s (closed-form estimates) |
| `spin_magnitude.py` t6 (tolerance scan) | 15 s |
| `spin_magnitude.py` t7 (`Delta S_par`) | 5 s |
| `holonomy_scan.py` (main + null + inset + wrap) | 53 s |
| `focused_beam.py` (ten stages) | 281 s |
| `indep_check_all.py` (four scripts; `indep_check3.py` alone is 140 s) | 165 s |
| the four `make_fig*.py` | 10 s |
| **total, full run** | **≈ 21 min** |
| `reproduce_all.py --quick` | ≈ 6 min (of which `indep` 165 s, `focused` 139 s) |

The dominant cost is the number of DOP853 steps at `rtol = 1e-13`,
`atol = 1e-16` over pulses up to `N = 16` cycles, i.e. a phase interval of
`|eta| <= 8 sigma ≈ 483`. Nothing here needs a GPU, and none of it benefits from
more than one core per process.

## Environment

`requirements.txt` pins the versions actually installed on the machine that
produced the deposit: NumPy 2.4.4, SciPy 1.17.1, Matplotlib 3.10.9, SymPy 1.14.0
on CPython 3.13.2. SymPy is needed only by `focused_beam.py`, which derives the
Lax–Louisell–McKnight expansion symbolically and checks the Coulomb-gauge
closure before evaluating it numerically. Matplotlib is needed only by the four
plotting scripts. No compiled extension, no GPU, no PyTorch: `torch` is listed
in `requirements.txt` as a comment and is required only by three files in
`legacy/`.

Figures are written as **both PDF and EPS** at exactly 86 mm width (one PRA
column). APS lists PostScript formats among the preferred production formats, so
the EPS files are the ones intended for production and the PDFs are for the
review PDF. Fonts are embedded as Type 42 in both. Every series is distinguished
by marker shape or line style as well as by colour, so the figures survive
greyscale printing.

The two APS minima — **no lettering below 8 pt** (about 2 mm of cap height at
1:1, which is what these figures are reproduced at) and **no rule below
0.6 pt** — are not left to the discipline of the plotting scripts.
`plotstyle.audit()` walks the finished figure, inspects every text, line, patch
and collection, and raises before anything is written; `plotstyle.save()` calls
it. A figure that violates either minimum cannot be produced by this deposit.

## Licence

The files in this directory are released under the **MIT Licence** (see
`LICENSE`).

Rationale, for the authors to confirm or overrule: MIT is the customary choice
for research code that is meant to be run, modified and folded into other
people's pipelines, which is what this is — a set of integrators and plotting
drivers. CC-BY-4.0 is the better fit for *data and text*, and is what Zenodo
proposes by default; it is a poor fit for source code, because it says nothing
about patents, warranties or the status of derivative binaries. If the authors
prefer a single licence for the whole deposit, the usual compromise is MIT for
the `.py` files and CC-BY-4.0 for the `.json` result files; that split can be
recorded in the Zenodo metadata without changing anything here. **The final
choice is the authors'.**

## Scope, and what this code does not claim

The plane-wave part is exact within its stated model: an exact plane wave, the
classical T-BMT equation with an anomalous moment, no radiation reaction, no
photon emission, no Stern–Gerlach force, no electric dipole moment. The focused
part uses a Lax–Louisell–McKnight expansion carried to third order in `eps`,
whose Maxwell residuals are measured and reported in TABLE S6; the amplitudes
extracted from it carry the systematic error quantified in Sec. S5.7, while the
`eps^2` exponent does not. The applicability estimates of stage `t5` are
analytic scalings taken from the review literature — no radiation is simulated
anywhere in this repository.

Three limits of the deposited numbers are stated in the code and are repeated
here because they are easy to overread:

* the plane-wave zero at `a0 = 10` is a round-off floor, not a converged value.
  Stage `t6` shows it fluctuating over `(1.0–2.6)e-13` rad while the tolerance is
  tightened by two decades and the step count triples. The defensible statement
  is the bound, not three significant figures;
* `Delta S_par` is measured with the anomaly inflated by two to three orders of
  magnitude, because the physical value is `~1e-20` and no double-precision
  integration can see it. What is measured is the *exponent*, and it is 6.00 to
  5.98 over the small-anomaly end of the scan;
* the `wrap` stage exists because the angle read from an SO(3) matrix is not the
  holonomy once the accumulated angle passes `pi`. Every scaling exponent in this
  deposit is read below that fold, and the stage shows where the fold is.

The `legacy/` subdirectory is a record of a discarded approach and supports none
of the claims above; see `legacy/README.md`.
