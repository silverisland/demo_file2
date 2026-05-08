# Repository Guidelines

## Project Structure & Module Organization

This repository is a PyTorch time-series fusion project for photovoltaic power forecasting. Core experiment flow lives in `run_longExp.py` and `exp/`. Data loading is in `data_provider/fusion_dataset.py`; metrics and training utilities are in `utils/`. Fusion models are grouped under `models/fusion/`, with version selection handled by `models/factory.py`. Configuration placeholders for expert models are in `configs/`, and design notes are in `docs/`.

There is currently no dedicated `tests/` directory. Use focused smoke tests and `py_compile` checks before committing.

## Build, Test, and Development Commands

Use the Pixi environment Python:

```bash
.pixi/envs/default/bin/python -m py_compile run_longExp.py exp/*.py models/**/*.py utils/*.py
```

Checks Python syntax across the main project files.

```bash
.pixi/envs/default/bin/python run_longExp.py --help
```

Verifies CLI options, including `--fusion_version` and `--fusion_expert_name`.

```bash
.pixi/envs/default/bin/python -c "from models.factory import fusion_version_choices; print(fusion_version_choices())"
```

Confirms fusion model registry imports correctly.

Training uses `run_longExp.py`, for example:

```bash
.pixi/envs/default/bin/python run_longExp.py --is_training 1 --model_id exp1 --model FusionModel --fusion_version expert_head --fusion_expert_name m1 --data custom
```

Real training also requires valid data paths and expert model dependencies.

## Coding Style & Naming Conventions

Use 4-space indentation and standard Python naming: `snake_case` for functions and variables, `PascalCase` for classes. Keep fusion experiments as separate files under `models/fusion/`, such as `v6.py` or `expert_head.py`. Register new versions in `models/factory.py` by adding an import and one `FUSION_REGISTRY` entry.

Keep model outputs shaped as `(B, n_features, pred_len)` unless a file clearly documents otherwise.

## Testing Guidelines

Before submitting changes, run syntax checks and at least one minimal smoke test for the touched model. For fusion models, verify both `flag="test"` and `flag="train"` paths when applicable. Test names are not standardized yet; if adding tests, prefer `tests/test_<module>.py` with small tensor inputs and no external data dependency.

## Commit & Pull Request Guidelines

Recent history uses short imperative messages, sometimes with Conventional Commit prefixes, for example `feat:` and `refactor:`. Prefer concise messages like `feat: add expert head reconstruction` or `refactor: simplify fusion registry`.

Pull requests should describe the experiment or behavior changed, list validation commands run, note any required data/config changes, and call out compatibility breaks in model signatures or output shapes.

## Configuration & Safety Notes

Do not commit private datasets, checkpoints, or machine-specific paths. Keep `configs/*.yaml` portable, and document required external expert model packages when a fusion version depends on them.
