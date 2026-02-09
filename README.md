# GEM-Style Constraints for PEFT: Dual Gradient Projection in LoRA

Reference implementation of **I-GEM**, a fixed-budget dual projected-gradient
approximation to Gradient Episodic Memory that runs entirely inside the LoRA
adapter subspace.

GEM keeps continual learning stable by projecting each update so it does not
increase the loss on replayed samples from earlier tasks. It does this by
solving a quadratic program at every step, on the host, over all model
parameters — which does not survive contact with an LLM-sized model. I-GEM
keeps the constraint and replaces the solver: the same dual problem is solved
with a handful of projected-gradient steps, on the GPU, over the adapter
parameters only.

## Method

With `g` the current adapter gradient and `G` the stacked per-task replay
gradients, the projection is

```
min ½‖g̃ − g‖²   s.t.   G g̃ ≥ 0
```

whose dual is a non-negative QP over multipliers `λ`:

```
min_{λ ≥ 0}  ½ λᵀ(GGᵀ)λ + (Gg)ᵀλ,      g̃ = g + Gᵀλ
```

I-GEM runs `K` projected-gradient steps on `λ` instead of calling a solver:

```
λ ← max(0, λ − η_λ[(GGᵀ)λ + Gg])          repeated K times
η_λ = c / σ̂_max(GGᵀ),  c ∈ [0.5, 0.9]
```

`σ̂_max` comes from a few steps of power iteration, `λ` is warm-started across
minibatches within a task and reset at task boundaries, and the rows of `G` are
normalised so one large constraint cannot dominate the dual geometry. A single
step is two matrix–vector products, so a projection costs `O(K·t·d_φ)` and
storage is `O(t·d_φ)` — with `d_φ` the adapter dimension rather than the full
parameter count `d_θ`.

## Setup

```bash
uv sync
uv run python scripts/train.py --config configs/config.yaml
uv run python scripts/plot_results.py --results-dir ./results/agnews/
```

Training writes one `.npy` (the R matrix), one `_baseline.npy` (the R₀ row),
one `.csv` (per-experience metrics) and one `.json` (run summary, including the
projection count and mean overhead) per run. The plotting script reads the JSON
summaries and emits both LaTeX tables and the figures.

Run a single method with `--plugin igem`. Add `--keep-going` to let a sweep
continue past a failing run instead of stopping at it.

## Benchmark

AG News, recast as a domain-incremental stream. Rather than splitting by class —
which makes the task trivially separable — each experience draws a different
class prior (one label at 0.7, the rest at 0.1) and is biased towards a distinct
lexical corner of each label. Experiences are built from **disjoint** rows and
then split 80/20 into train and test. The partition is fixed across runs; the
order the model meets the experiences in is shuffled per seed.

The backbone is GPT-2 (355M) with the base weights frozen and LoRA adapters
(`r=8`, `α=32`, dropout 0.05) on `c_attn` and `c_proj`, which covers both the
attention and MLP projections. Classification is by scoring: each item is
presented as four `(headline, label word)` continuations and the mean
log-likelihood of the label word becomes that class's logit.

## Layout

```
src/igem/
  plugins/     base (shared hooks) + gem, agem, igem
  benchmarks/  the AG News domain-drift stream
  models/      GPT-2 + LoRA and the option-scoring wrapper
  utils/       CL metrics, seeding, the train/eval loop
scripts/       train.py, plot_results.py
configs/       hyperparameters
tests/         projection maths, benchmark construction, metrics, end-to-end
```

## Hyperparameters

| | |
|---|---|
| seeds | 0, 2, 5, 7, 11 |
| experiences (T) | 3 |
| epochs / experience | 1 |
| learning rate | 1e-3 (AdamW, weight decay 1e-2) |
| train / eval batch | 32 / 50 |
| patterns per experience | 100 |
| memory strength | 0.3 |
| projection interval | 1 |
| PGD iterations (K) | 3 |
| max sequence length | 512 |

## Metrics

With `R[j, i]` the accuracy on task `i` after training through task `j`, and
`R₀` the accuracies before any continual training:

- **AvgAcc** — mean of the final row.
- **BWT** — mean of `R[T-1, i] − R[i, i]`; negative means forgetting.
- **FWT** — mean of `R[i-1, i] − R₀[i]`; zero-shot gain on a task not yet seen.
- **Forgetting** — mean of `max_{t<T} R[t, i] − R[T-1, i]`.
- **MPO** — total projection time divided by the number of projections.

`R₀` is measured at the start of every run rather than assumed to be chance, so
FWT reflects the actual starting point of the adapters.

## Implementation notes

A few places where the code makes a choice the paper leaves open, or where the
two could be read as disagreeing:

- **Parameter space.** `param_space: adapter` (the default) restricts the
  gradient vector — and therefore `G` — to trainable adapter parameters, which
  is the setting the projection is defined in. `param_space: full` keeps the
  frozen base weights in the vector, which is the configuration the exact-GEM
  cost comparison refers to. The setting matters a great deal for the measured
  overhead, since in `full` the solver marshals ~354M zeros per projection, so
  it is explicit rather than implied.
- **`memory_strength`** is the margin in GEM's QP (`λ ≥ memory_strength`,
  following the reference implementation). I-GEM projects onto `λ ≥ 0` as in
  Algorithm 1, and A-GEM projects onto the constraint boundary exactly, so
  neither uses it.
- **`memory_size`** is carried in the config for compatibility, but the buffer
  is governed by `patterns_per_exp`; each experience is visited once, so no
  eviction ever happens.
- **Step size.** `σ̂_max` is estimated by power iteration with `c = 0.9` rather
  than computed exactly. Short power iterations under-estimate the spectral
  norm, so `c < 1` keeps the step below `1/L`.

## Tests

```bash
uv run pytest
```

The projection routines are checked against an exact QP solve rather than only
for self-consistency: given enough iterations the fixed-budget PGD must reach
the same point the solver does, the result must satisfy `G g̃ ≥ 0`, and a
gradient that already satisfies the constraints must be left alone. The
benchmark tests assert that experiences never share a row, and the metric tests
check every definition above against values worked out by hand.
