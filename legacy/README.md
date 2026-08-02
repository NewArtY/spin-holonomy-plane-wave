# `legacy/` — a discarded line of work, kept as a record

**Nothing in this subdirectory supports any claim in the paper.** No number, no
table and no figure of the Letter or of the Supplemental Material is computed
here. The files are kept because deleting the record of a rejected approach is
worse than keeping it: they document what was tried, what it cost, and why it
was dropped.

## What this was

The project began as a *methods* paper: a physics-informed neural network (PINN)
was to be trained on a coupled orbit–spin system and benchmarked against
conventional ODE integration, and the physics content was a "rapidity–BMT
closure" written in terms of the proper time `tau`.

Two things ended that plan.

1. **The model was not a solution of Maxwell's equations.** `physics_np.py` and
   `common.py` encode a vector potential that is a function of the proper time,
   `A_mu(tau)`. A real plane wave depends on the light-front phase,
   `A_mu(eta)` with `eta = k.x`. Under the proper-time parameterisation the pulse
   does not return the four-velocity to its initial value, the transverse field
   acquires a dc component, and a "carrier-envelope-phase effect" appears that is
   an artefact of the parameterisation and not of the physics. Everything
   downstream of that model inherited the artefact. The correct treatment, and
   the reason the artefact appears, is written out in the docstring of
   `../spin_magnitude.py`, section 1.
2. **The benchmark had no result to report.** Once the physics was redone
   correctly, the spin problem turned out to have a closed-form answer — the
   holonomy of the Letter — and a 4×4 linear ODE integrated by DOP853 solves it
   to machine precision in milliseconds. A neural surrogate for a problem with an
   exact solution is not a contribution, and the timing comparisons below were
   measuring the cost of approximating something that does not need
   approximating.

The paper was rewritten around the physical result. The PINN, the benchmarks and
the rapidity model were dropped in full.

## Files

| File | What it was |
| :-- | :-- |
| `common.py` | the model system and the PINN definition (PyTorch) shared by the benchmarks; contains the discarded `A_mu(tau)` model |
| `physics_np.py` | torch-free copy of the same model, split out so that Windows `spawn` workers would not each import PyTorch |
| `bench_pinn.py` | PINN training runs; results in `pinn_res*.json`, `pinn_short.*` |
| `bench_ansatz.py` | hard-constraint ansatz variant of the same network; `ansatz.json` |
| `bench_batch.py` | batch-size and float32/float64 sweeps; `batch_f32.json`, `batch.log` |
| `bench_ivp.py` | reference ODE integration used as the benchmark baseline; `ivp.json` |
| `bench_mp.py` | multiprocessing scaling of that baseline; `mp.json` |
| `bench_scaling.py` | problem-size scaling study; `scaling.json` |
| `*.json`, `*.log` | the outputs of the runs above, kept unedited |

`bench_pinn.py`, `bench_batch.py` and `bench_ansatz.py` need PyTorch
(`torch==2.12.0+cpu` was installed on the machine used). The rest need only
NumPy and SciPy. They are **not** covered by `../reproduce_all.py` and are not
required by `../requirements.txt`.

## Reading these files

Treat the numbers in them as a timing record on one specific laptop, and treat
the physics in `common.py` / `physics_np.py` as superseded. If you want the
physics, read `../spin_magnitude.py` and `../focused_beam.py`, which were written
from scratch and deliberately share no code with this subdirectory.
