# -*- coding: utf-8 -*-
r"""
reproduce_all.py -- one entry point that recomputes every number and redraws
every figure of

    "Net electron spin rotation in a plane-wave pulse:
     Holonomy set by the anomalous magnetic moment"
    N. S. Akintsov, A. P. Nevecheria, S. N. Andreev, Q.-H. Qin
    (Physical Review D)

WHAT IT RUNS, IN ORDER
----------------------
  1. spin_magnitude.py   stages t1..t7, one JSON per stage, then merged into
                         spin_magnitude.json          (plane-wave scans)
                         t6 = integrator-tolerance scan behind TABLE S3
                         t7 = longitudinal spin projection, Delta S_par
  2. holonomy_scan.py    stages main,null,inset,wrap -> holonomy_scan.json
                         (the (area, rotation) data set of FIG. 2, and the
                          angle-wrapping case of Sec. S2.3)
  3. focused_beam.py     all ten stages, three JSON files, then merged into
                         focused_beam.json            (focused Gaussian pulse)
  4. indep_check_all.py  the four from-scratch verification scripts
                         -> indep_check.json / .log
  5. make_fig1.py, make_fig2.py, make_fig3.py, make_figS1.py, make_figS2.py
                         -> ../manuscript/fig*.pdf and fig*.eps, PNG proofs in
                         ./figproofs  (make_fig1.py reads no data file: FIG. 1
                         is computed from the pulse formulas)

WHAT --quick DOES, AND WHAT IT DOES NOT
---------------------------------------
--quick runs EVERY stage of every script and all four independent checks.  What
it reduces is the size of the grids, and only where the script exposes a knob
for it: spin_magnitude t1..t7, holonomy_scan (its own --quick), and
focused_beam (--eps-list, --n-phi, --field-eps, --a0-list,
--a0big-list, --gamma-list, --N-list, --b-list).
indep_check2/3/4 are standalone scripts with no grid parameter and therefore run
at full size; that is most of the quick-mode wall time.

Everything --quick writes goes into code/quickrun/, so a quick run can never
overwrite a deposited result.  The numbers and figures it produces are a smoke
test, not the published results.

Before 2026-08-02 this was not true: --quick passed no reduction to stage t1
(83.9 s, i.e. the full grid), skipped stages t2 and t4 entirely, ran three of
the ten focused-beam stages and two of the four independent checks.  The README
nevertheless said it exercised every code path.  It does now.

INTEGRITY
---------
    python reproduce_all.py --manifest      write MANIFEST.sha256
    python reproduce_all.py --verify        check every file against it
The manifest covers every deposited file in code/ (sources, results, logs,
figure proofs, licence, README); it excludes what this package generates
(__pycache__/, quickrun/) and what belongs to the person running it rather than
to the deposit (a virtual environment, an IDE directory, a VCS database), so a
venv created inside code/ does not make the check fail.  See
MANIFEST_SKIP_DIRS.  --verify exits nonzero on any difference and names it, which is
what a repository check needs and what re-downloading a Zenodo archive cannot
otherwise establish.

LOAD LIMITS
-----------
Every step is a separate Python process and the steps run STRICTLY ONE AT A
TIME: this driver never has more than one child process alive, and none of the
scripts it calls uses multiprocessing or threading.  BLAS threading is pinned to
one thread per process as well, because the arrays here are 4x4 and threaded
BLAS only adds contention.  Nothing runs in the background.

RUNTIME  (measured on the machine in code/README.md: Windows 10, i7 laptop,
single core actually used)
    --quick        ~6 min    every stage, reduced grids, figures NOT the
                             published ones
    full           ~21 min   the published numbers and figures
        spin_magnitude   ~11.5 min (t1 84 s, t2 237 s, t3 276 s, t4 45 s,
                                    t5 <1 s, t6 15 s, t7 5 s)
        holonomy_scan    ~53 s      (main 28 s, null 16 s, inset 2 s, wrap 5 s)
        focused_beam     ~5 min     (281 s over ten stages)
        indep_check_all  ~2.8 min   (165 s; indep_check3.py alone is 140 s)
        figures          ~10 s

Usage
    python reproduce_all.py                 # full run, writes into ./ and ../manuscript
    python reproduce_all.py --quick         # fast smoke run, into ./quickrun
    python reproduce_all.py --only figs     # redraw the figures from existing JSON
    python reproduce_all.py --skip focused  # everything except the focused-beam scans
    python reproduce_all.py --outdir .      # keep the figures next to the code
    python reproduce_all.py --manifest      # (re)write MANIFEST.sha256
    python reproduce_all.py --verify        # check the deposit against it
"""
from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
MANIFEST = "MANIFEST.sha256"

# Steps: (tag, description, list of argv tails, list of argv tails in quick mode)
STEPS = [
    ("plane", "plane-wave scans (spin_magnitude.py, stages t1-t7)", [
        ["spin_magnitude.py", "--stages", "t1", "--out", "sm_t1.json"],
        # one run over both polarisations: --merge is a dict update, so running
        # delta=0 and delta=1 into two files would leave only one of them in the
        # merged t2.  The deposited sm_t2lin.json / sm_t2cir.json are that same
        # grid split by polarisation, from the original session.
        ["spin_magnitude.py", "--stages", "t2", "--deltas", "0,1",
         "--out", "sm_t2.json"],
        ["spin_magnitude.py", "--stages", "t3", "--out", "sm_t3.json"],
        ["spin_magnitude.py", "--stages", "t4", "--out", "sm_t4.json"],
        ["spin_magnitude.py", "--stages", "t5", "--out", "sm_t5.json"],
        ["spin_magnitude.py", "--stages", "t6", "--out", "sm_t6.json"],
        ["spin_magnitude.py", "--stages", "t7", "--out", "sm_t7.json"],
        ["spin_magnitude.py", "--merge",
         "sm_t1.json,sm_t2.json,sm_t3.json,sm_t4.json,sm_t5.json,"
         "sm_t6.json,sm_t7.json",
         "--out", "spin_magnitude.json"],
    ], [
        ["spin_magnitude.py", "--stages", "t1",
         "--t1-envs", "gauss", "--t1-gammas", "1,100", "--t1-a0s", "1,10",
         "--t1-Ns", "1", "--t1-phis", "0,0.7", "--t1-anoms", "1e-2,1",
         "--t1-cf-gammas", "1", "--out", "quickrun/sm_t1.json"],
        ["spin_magnitude.py", "--stages", "t2", "--deltas", "0,1",
         "--gammas", "1,100", "--a0s", "1", "--Ns", "1",
         "--n-phi-lin", "2", "--n-phi-cir", "4",
         "--out", "quickrun/sm_t2.json"],
        ["spin_magnitude.py", "--stages", "t3,t5", "--gammas", "1,100",
         "--a0s", "1", "--Ns", "1", "--n-phi-cir", "4",
         "--t3-sigmas", "1,3.7745", "--t3-nphi", "4",
         "--out", "quickrun/sm_t3.json"],
        ["spin_magnitude.py", "--stages", "t4",
         "--t4-a0s", "0.1,1", "--t4-sigmas", "1", "--t4-nphi", "4",
         "--out", "quickrun/sm_t4.json"],
        ["spin_magnitude.py", "--stages", "t6",
         "--t6-rtols", "1e-9,1e-12,1e-13", "--t1-phis", "0,0.7",
         "--out", "quickrun/sm_t6.json"],
        ["spin_magnitude.py", "--stages", "t7",
         "--t7-x", "0.05,0.1,0.2", "--t7-a0s", "1",
         "--out", "quickrun/sm_t7.json"],
        ["spin_magnitude.py", "--merge",
         "quickrun/sm_t1.json,quickrun/sm_t2.json,quickrun/sm_t3.json,"
         "quickrun/sm_t4.json,quickrun/sm_t6.json,quickrun/sm_t7.json",
         "--out", "quickrun/spin_magnitude.json"],
    ]),
    ("holonomy", "signed-area scan for FIG. 2 (holonomy_scan.py)", [
        ["holonomy_scan.py", "--stages", "main,null,inset,wrap",
         "--out", "holonomy_scan.json"],
    ], [
        ["holonomy_scan.py", "--stages", "main,null,inset,wrap", "--quick",
         "--out", "quickrun/holonomy_scan.json"],
    ]),
    ("focused", "focused Gaussian pulse (focused_beam.py, ten stages)", [
        ["focused_beam.py", "--stages",
         "field,conv,order,eps,xcheck,scans,window", "--out", "fb_A.json"],
        ["focused_beam.py", "--stages", "cep,cepscan", "--out", "fb_B.json"],
        ["focused_beam.py", "--stages", "a0big", "--out", "fb_C.json"],
        ["focused_beam.py", "--merge", "fb_A.json,fb_B.json,fb_C.json",
         "--out", "focused_beam.json"],
    ], [
        ["focused_beam.py", "--stages",
         "field,conv,order,eps,xcheck,scans,window,cep,cepscan,a0big",
         "--eps-list", "0.30,0.15,0.05", "--n-phi", "4",
         "--field-eps", "0.30,0.10,0.025", "--a0-list", "0.5,1,2",
         "--a0big-list", "1,4,12,24", "--gamma-list", "1,10,1000",
         "--N-list", "0.5,2", "--b-list", "0,0.5",
         "--out", "quickrun/focused_beam.json"],
    ]),
    ("indep", "independent verification (indep_check_all.py, four scripts)", [
        ["indep_check_all.py", "--out", "indep_check.json",
         "--log", "indep_check.log"],
    ], [
        # no grid knob exists in these four; they run in full, which is the
        # honest way to keep the claim "every code path" true.
        ["indep_check_all.py", "--out", "quickrun/indep_check.json",
         "--log", "quickrun/indep_check.log"],
    ]),
    ("rr", "radiation reaction and dressed anomaly (rr_ikt_check.py)", [
        ["rr_ikt_check.py", "--out", "rr_ikt_check.json",
         "--log", "rr_ikt_check.log"],
    ], [
        # a few seconds; runs in full
        ["rr_ikt_check.py", "--out", "quickrun/rr_ikt_check.json",
         "--log", "quickrun/rr_ikt_check.log"],
    ]),
]

FIGS = ["make_fig1.py", "make_fig2.py", "make_fig3.py", "make_figS1.py",
        "make_figS2.py"]

# What the checksum manifest covers.  Two kinds of directory are excluded:
# trees this package generates itself (__pycache__, quickrun), and trees that
# belong to whoever is running the code rather than to the deposit -- a virtual
# environment, an IDE project directory, a version-control database.  Without
# the second group, `--verify` reports several hundred EXTRA files as soon as
# someone creates a venv inside code/, which is what most IDEs do by default.
MANIFEST_SKIP_DIRS = {
    # generated by this package
    "__pycache__", "quickrun", ".ipynb_checkpoints",
    # virtual environments
    ".venv", "venv", "env", "ENV", ".virtualenv",
    # version control
    ".git", ".hg", ".svn",
    # IDE and editor project data
    ".idea", ".vscode", ".vs", ".spyproject", ".history",
    # tool caches
    ".pytest_cache", ".mypy_cache", ".ruff_cache", ".tox", ".eggs",
}
MANIFEST_SKIP_FILES = {MANIFEST}
MANIFEST_SKIP_EXT = {".pyc", ".pyo"}


# ----------------------------------------------------------------- integrity
def deposit_files(root=HERE):
    """Every deposited file, as paths relative to code/, sorted."""
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in MANIFEST_SKIP_DIRS)
        for fn in sorted(filenames):
            if fn in MANIFEST_SKIP_FILES:
                continue
            if os.path.splitext(fn)[1].lower() in MANIFEST_SKIP_EXT:
                continue
            p = os.path.join(dirpath, fn)
            out.append(os.path.relpath(p, root).replace(os.sep, "/"))
    return sorted(out)


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_manifest(root=HERE):
    rels = deposit_files(root)
    lines = ["# MANIFEST.sha256 -- integrity manifest of this deposit.",
             "# Columns: sha256, size in bytes, path relative to code/.",
             "# Regenerate with `python reproduce_all.py --manifest`,",
             "# check with   `python reproduce_all.py --verify`.",
             f"# files: {len(rels)}"]
    for rel in rels:
        p = os.path.join(root, rel.replace("/", os.sep))
        lines.append(f"{sha256_of(p)}  {os.path.getsize(p)}  {rel}")
    path = os.path.join(root, MANIFEST)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {MANIFEST}: {len(rels)} files")
    return path


def read_manifest(root=HERE):
    path = os.path.join(root, MANIFEST)
    if not os.path.exists(path):
        raise SystemExit(f"{MANIFEST} is missing; run --manifest first")
    rec = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            digest, size, rel = line.split(None, 2)
            rec[rel] = (digest, int(size))
    return rec


def verify_manifest(root=HERE):
    want = read_manifest(root)
    have = set(deposit_files(root))
    changed, missing, extra = [], [], sorted(have - set(want))
    for rel, (digest, size) in sorted(want.items()):
        p = os.path.join(root, rel.replace("/", os.sep))
        if not os.path.exists(p):
            missing.append(rel)
            continue
        if os.path.getsize(p) != size or sha256_of(p) != digest:
            changed.append(rel)
    ok = not (changed or missing or extra)
    print(f"verify: {len(want)} files in {MANIFEST}, "
          f"{len(changed)} changed, {len(missing)} missing, {len(extra)} extra")
    for tag, lst in (("CHANGED", changed), ("MISSING", missing),
                     ("EXTRA", extra)):
        for rel in lst:
            print(f"  {tag:8s} {rel}")
    print("verify: OK" if ok else "verify: FAILED")
    return ok


# --------------------------------------------------------------------- run
def run(argv, env):
    print("  $ python " + " ".join(argv), flush=True)
    t0 = time.perf_counter()
    r = subprocess.run([sys.executable] + argv, cwd=HERE, env=env)
    dt = time.perf_counter() - t0
    if r.returncode != 0:
        raise SystemExit(f"FAILED ({r.returncode}): {' '.join(argv)}")
    return dt


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Recompute and redraw everything, one process at a time.")
    ap.add_argument("--quick", action="store_true",
                    help="every stage, reduced grids (~6 min); the figures "
                         "produced are NOT the published ones")
    ap.add_argument("--only", default="",
                    help="comma-separated subset of: "
                         + ",".join(t for t, *_ in STEPS) + ",figs")
    ap.add_argument("--skip", default="", help="same tags, but excluded")
    ap.add_argument("--outdir", default=None,
                    help="where the figure .pdf and .eps files are written "
                         "(default: ../manuscript, or ./quickrun with --quick)")
    ap.add_argument("--manifest", action="store_true",
                    help="write MANIFEST.sha256 for the deposit and exit")
    ap.add_argument("--verify", action="store_true",
                    help="check the deposit against MANIFEST.sha256 and exit")
    args = ap.parse_args(argv)

    if args.manifest:
        write_manifest()
        return
    if args.verify:
        raise SystemExit(0 if verify_manifest() else 1)

    only = {s.strip() for s in args.only.split(",") if s.strip()}
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}

    # In quick mode nothing published is touched: every output, JSON and figure
    # alike, goes to code/quickrun/.
    qdir = os.path.join(HERE, "quickrun")
    if args.quick:
        os.makedirs(qdir, exist_ok=True)
    if args.outdir is None:
        args.outdir = ("quickrun" if args.quick
                       else os.path.join("..", "manuscript"))
    fig_extra = ([] if not args.quick else ["--png-dir", "quickrun"])
    fig_data = {             # make_fig1.py takes no --data
        "make_fig2.py": "quickrun/holonomy_scan.json",
        "make_fig3.py": "quickrun/focused_beam.json",
        "make_figS1.py": "quickrun/spin_magnitude.json",
        "make_figS2.py": "quickrun/focused_beam.json",
    }

    env = dict(os.environ)
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env[v] = "1"

    print(f"reproduce_all: {'QUICK' if args.quick else 'FULL'} run, "
          f"one process at a time, figures -> {args.outdir}", flush=True)
    total, times = time.perf_counter(), {}
    for tag, desc, full, quick in STEPS:
        if (only and tag not in only) or tag in skip:
            print(f"[{tag}] skipped", flush=True)
            continue
        print(f"[{tag}] {desc}", flush=True)
        times[tag] = sum(run(a, env) for a in (quick if args.quick else full))

    if not ((only and "figs" not in only) or "figs" in skip):
        print("[figs] drawing FIG. 1, 2, 3, S1, S2", flush=True)
        times["figs"] = sum(
            run([f, "--outdir", args.outdir] + fig_extra
                + (["--data", fig_data[f]]
                   if args.quick and f in fig_data else []), env)
            for f in FIGS)

    print("\n---- wall time ----", flush=True)
    for k, v in times.items():
        print(f"  {k:10s} {v:8.1f} s")
    print(f"  {'TOTAL':10s} {time.perf_counter() - total:8.1f} s")
    if args.quick:
        print("\nQUICK MODE: the numbers and the figures above are a smoke "
              "test, not the published results.  Re-run without --quick.")
    else:
        print("\nRegenerate the integrity manifest with "
              "`python reproduce_all.py --manifest`.")


if __name__ == "__main__":
    main()
