# GEM-Style Constraints for PEFT: Dual Gradient Projection in LoRA

Reference implementation of **I-GEM**, a fixed-budget dual projected-gradient
approximation to Gradient Episodic Memory that runs inside the LoRA
adapter subspace.

## Setup

```bash
uv sync
uv run python scripts/train.py --config configs/config.yaml
uv run python scripts/plot_results.py --results-dir ./results/agnews/
```
Run a single method with `--plugin igem`. Add `--keep-going` to let a sweep
continue past a failing run instead of stopping at it.

## Benchmark

AG News, recast as a domain-incremental stream. Rather than splitting by class, each experience draws a different
class prior  and is biased towards a distinct
 corner of each label. Experiences are built from disjoint rows and
then split 80/20 into train and test. The partition is fixed across runs, but the
order the model meets the experiences in is shuffled per seed.

The backbone is GPT-2 (355M) with the base weights frozen and LoRA adapters
(`r=8`, `α=32`, dropout 0.05) on `c_attn` and `c_proj`, which covers both the
attention and MLP projections. Classification is by scoring: each item is
presented as four `(headline, label word)` continuations and the mean
log-likelihood of the label word becomes that class's logit.

## Implementation notes

- `param_space: adapter`  restricts the
  gradient vector to trainable adapter parameters, which
  is the setting the projection is defined in. `param_space: full` keeps the
  frozen base weights in the vector. 
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

