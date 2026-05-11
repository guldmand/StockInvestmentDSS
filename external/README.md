# External Dependencies

This folder documents external repositories and frameworks used by the PoC.

External repositories should not be cloned or vendored into this repository unless explicitly needed.

## Strategy

- Prefer package manager installation when possible.
- Pin external repositories by commit when they are cloned or used directly.
- Keep heavy research/RL dependencies out of the fast backend image.
- Define container dependencies in the relevant Dockerfile, requirements file or environment file.
- Local Conda environments are for development only and must not be assumed by containers, CI or k3s.

## Dependency split

Fast application/backend:

- should stay lightweight
- may consume stored outputs, metrics and decisions
- should avoid FinRL/PyTorch unless required

Slow research/worker layer:

- may use FinRL, PyTorch, Stable-Baselines3, Gymnasium and ObjectRL
- may run notebooks, training, backtests and evaluation jobs
- should write outputs to `research/results/`, DuckDB, Parquet or CSV

## Manifest

The external dependency manifest is:

```txt
external/external-repos.lock
```

Each entry should include:

- name
- URL
- commit pin or `pin-later`
- role
- usage layer

## Rule

Do not clone external repositories into `external/` unless a task explicitly requires it.
