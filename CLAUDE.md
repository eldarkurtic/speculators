# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Speculators is a library for training speculative-decoding draft models ("speculators") and exporting them in a standardized, Hugging Face-compatible format that deploys directly into vLLM (`vllm serve <speculator_model>`). It supports the EAGLE-3 and DFlash algorithms, plus converters that import external research checkpoints (EAGLE v1/v2/v3, HASS) into the speculators format.

## Common commands

```bash
pip install -e ".[dev]"          # dev install

make quality                     # ruff check + ruff format --check + mdformat --check + mypy --check-untyped-defs (run before pushing)
make style                       # auto-fix: ruff format, ruff check --fix, mdformat

python -m pytest tests/unit                       # unit tests (mocked, fast)
python -m pytest tests/integration                # integration tests
python -m pytest tests/e2e                         # end-to-end
python -m pytest tests/unit/models/test_foo.py::test_bar   # single test
python -m pytest tests/unit --cov=speculators --cov-report=html
```

CI (`.github/workflows/`) runs exactly `make quality`, then `pytest -ra tests/unit`, then `tests/integration`. Commits require a DCO signoff (`git commit --signoff`). PRs/issues go upstream to `vllm-project/speculators`.

Note: `pytest` is configured verbose-by-default (`addopts = '-s -vvv --cache-clear'` in `pyproject.toml`), so output is noisy and the cache is cleared each run.

`pytest` markers (see `pyproject.toml`): `smoke`, `sanity`, `regression`, `e2e`, `slow`, `multi_gpu`.

## Architecture

**Registry-driven, algorithm-pluggable.** The core abstraction is a pair of registries built on `ClassRegistryMixin` (`src/speculators/utils/registry.py`):

- `SpeculatorModel` (`src/speculators/model.py`) — base `PreTrainedModel`. Concrete models register with `@SpeculatorModel.register("eagle3")` / `@SpeculatorModel.register("dflash")`.
- `SpeculatorModelConfig` (`src/speculators/config.py`) — base Pydantic + `PretrainedConfig`. Configs register the same way.

`speculators/__init__.py` calls `reload_schemas()` at import to populate the registries. The `speculators_model_type` field in a saved `config.json` is the key used to look up the right class on load. This is why the training script and CLI never hard-code algorithm names — they resolve them through the registry. To add an algorithm, follow `docs/developer/add_algorithm.md`: create a self-contained dir under `src/speculators/models/<algo>/` with `config.py` + `core.py`, register both classes, add training factory classmethods on the model, and add CLI args to `scripts/train.py`.

**Source layout (`src/speculators/`):**

- `models/eagle3/`, `models/dflash/` — self-contained algorithm implementations (each owns `config.py`, `core.py`, `attention.py`, `metrics.py`, `model_definitions.py`). Both reuse `DraftVocabMixin` (in `model.py`) for draft↔target vocab mapping, embeddings, and LM heads.
- `convert/eagle/` — converters from external EAGLE/HASS checkpoints; entry is `convert/entrypoints.py::convert_model`, exposed as `speculators convert`.
- `train/` — training engine: `trainer.py` (`Trainer`/`TrainerConfig`), `data.py` (datasets + collate), `distributed_batch_sampler.py`, `checkpointer.py`, `vocab_mapping.py`, `noise_transforms.py`.
- `data_generation/` — offline/online hidden-state generation via a vLLM client (`vllm_client.py`).
- `proposals/` — token proposal methods (`greedy.py`), also registry-based via `TokenProposalConfig`.
- `__main__.py` — the `speculators` Typer CLI (`convert`, `version`).

**Training is run via scripts, not the installed CLI.** `scripts/train.py` is an argparse-based, `torchrun`-launched distributed trainer (it is *not* a `speculators` subcommand). The full pipeline is: `scripts/prepare_data.py` → `scripts/launch_vllm.py` (server for hidden states) → `torchrun ... scripts/train.py`. See `examples/train/*.sh` for complete, runnable online/offline pipelines and `docs/user_guide/tutorials/` for walkthroughs.

**Two training modes:** *offline* generates hidden states to disk first (`scripts/data_generation_offline.py`); *online* generates them on-the-fly from a live vLLM server during training (`--vllm-endpoint`, `--on-missing generate`).

## Conventions

- Line length 88, double quotes, ruff + mypy enforced. Per-directory ruff ignores already relax rules for `tests/`, `scripts/`, `examples/`, `convert/`, and `data_generation/` (see `[tool.ruff.lint.extend-per-file-ignores]`) — don't fight those.
- `make quality` runs `mdformat --check` on all Markdown; run `make style` after editing any `.md` or it will fail CI.

## Branch note

The current branch `dflash-ablation` carries an `ablation/` directory (DFlash layer-selection experiment scaffolding: `gen_cache.sh`, `run.sh`, `eal.py`, handoff docs) and a second venv `.venv_vllm` used for two-venv cache generation. This is experiment tooling, not part of the shipped package.
