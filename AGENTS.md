# Repository Guidelines

## What can this project do
`light_mappo` is a lightweight MAPPO training stack centered on a Multi-UAV pursuit-evasion environment. It provides configurable training/evaluation runners, a pursuit scenario with optional target patrol routes, and TensorBoard logging plus saved artifacts under `results/`.

## Project Structure & Module Organization
`algorithms/` contains MAPPO core implementations and utilities (policy, actor-critic, RNN/CNN/MLP helpers). `envs/` holds environment wrappers and example discrete/continuous envs, plus `uav_pursuit_env.py`. `runner/` orchestrates training loops (shared vs separated policies). `train/` contains the entry script `train_uav_pursuit.py`. `config/` stores YAML configs (including patrol routes). `scripts/` includes helpers like the patrol route editor and MPE rendering. Outputs go to `results/`. Common utilities live in `utils/`, and CLI defaults are defined in `parameters.py`.

## Build, Test, and Development Commands
This project is run directly with Python (no build step) and should use YAML configs rather than long CLI arg lists.
- `python train/train_uav_pursuit.py --config config/pursuit3v1.yaml` is the standard training command.
- `python train/train_uav_pursuit.py --config config/minimal_test.yaml` is the quick testing command.



## Coding Style & Naming Conventions
Python code follows standard PEP 8 conventions: 4-space indentation, `snake_case` for functions/variables, and `CamelCase` for classes. Keep imports grouped (stdlib, third-party, local). There is no enforced formatter/linter; keep changes consistent with surrounding code and avoid large reformat-only diffs. YAML configs live in `config/` and should use clear, lower_snake_case keys.

## Testing Guidelines
There is no dedicated unit test suite. Validate changes by running a short config-based training job (e.g., `config/minimal_test.yaml`) and confirming a new run folder appears under `results/` with logs and (if enabled) GIFs. If you touch environment logic, run the UAV pursuit script with a full config to catch runtime regressions.

## Commit & Pull Request Guidelines
Recent commits use short, imperative subject lines (e.g., “Update config”, “Add patrol routes control”). Follow that pattern and keep subjects under ~60 characters. PRs should include: a concise summary, configs/CLI args used, and any artifacts (log snippets, GIFs in `results/`). Link related issues when applicable and call out breaking changes or new dependencies.

## Configuration & Outputs
Prefer YAML configs via `--config` for reproducibility. Training artifacts are written to `results/<env>/<scenario>/<algorithm>/<experiment_name>/run*/`. Keep large generated files out of PRs unless explicitly requested.
