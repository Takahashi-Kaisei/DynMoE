# Repository Guidelines

## Project Structure & Module Organization

DynMoE is a research monorepo with three main work areas. `EMoE/` contains language and vision experiments built around the local `tutel/` fork, including `Language/`, `Vision/domainbed/`, and their test suites. `MoE-LLaVA/` contains the vision-language training and serving code under `moellava/`, plus task-specific shell scripts in `scripts/` and docs in `docs/`. `Examples/DeepSpeed-MoE/` is the smallest end-to-end reference for DynMoE on top of the vendored `DeepSpeed-0.9.5/` implementation. Top-level `assets/` holds paper visuals. Colab-specific operational notes are consolidated in the single top-level `COLAB_HANDOFF.md`; keep future Colab setup, troubleshooting, and session handoff updates there instead of creating additional `COLAB*.md` files.

## Build, Test, and Development Commands

Use the commands that match the component you are editing:

- `cd EMoE/tutel && pip install ./` installs the local Tutel fork required by `EMoE`.
- `cd MoE-LLaVA && pip install -e . && pip install -e ".[train]"` installs the VLM package and training extras.
- `cd Examples/DeepSpeed-MoE && bash train.sh` runs the reference ImageNet training entrypoint.
- `cd Examples/DeepSpeed-MoE && bash eval.sh` evaluates a saved checkpoint.
- `cd EMoE/tutel && python3 -m pytest -v -s tests/` runs Tutel regression tests.
- `cd EMoE/Vision/domainbed && python -m pytest test/` runs DomainBed-style vision tests.
- For the tested MoE-LLaVA StableLM Colab sanity check, follow `COLAB_HANDOFF.md`: use a GPU runtime, create `/content/DynMoE/.venv310`, install the pinned Python 3.10 stack there, set `HF_HOME=/content/hf_cache`, and run DeepSpeed from `/content/DynMoE/MoE-LLaVA`.

## Coding Style & Naming Conventions

Python is the primary language. Follow existing conventions: 4-space indentation, `snake_case` for functions and variables, `PascalCase` for classes, and descriptive script names such as `finetune_dynmoe.sh` or `test_glue_no_trainer.py`. Keep changes localized to the relevant stack instead of introducing cross-cutting abstractions. `EMoE/environment.yml` includes `black==23.3.0`; format Python consistently with Black-style line wrapping even though no repo-wide formatter target is defined.

## Testing Guidelines

Prefer the narrowest test scope that covers your change. For kernel or routing changes in `EMoE/tutel/`, run its `pytest` suite. For vision training logic, use `EMoE/Vision/domainbed/test/`. `MoE-LLaVA/` primarily relies on script-driven validation, so include the exact training, eval, or CLI command you used in the PR. Name new tests `test_<feature>.py` and keep them adjacent to the existing suite for that module.

For Colab workflow changes, update `COLAB_HANDOFF.md` and record the exact runtime assumptions, package versions, command, prompt, and observed result. The StableLM MoE inference path is CUDA/NCCL-oriented, so local Apple Silicon import checks are useful but do not replace a GPU-backed Colab sanity check.

## Commit & Pull Request Guidelines

Recent history favors short, imperative commit messages such as `fix forward in experts.py.` or `update readme: add usage examples.` Keep subject lines concise and specific to one change. PRs should include: a short problem statement, the affected subproject (`EMoE`, `MoE-LLaVA`, or `Examples`), commands used for validation, and any dataset or GPU assumptions. Add screenshots only for UI or visualization changes.
