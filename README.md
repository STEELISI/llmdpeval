# LLM DP Evaluation

This repository tests whether LLMs can write differentially private (DP) Python code. It generates code for DP
tasks, with or without retrieved context from a corpus of DP outlines, runs the code in a sandbox, and supports
grading the results by hand. The original design (prompts, data, generation, the runnable check and the first
evaluation) is by Pithayuth Charnsethikul in the upstream STEELISI/llmdpeval repository; this version replaces
its retrieval and evaluation.

- [docs/method.md](docs/method.md): the pipeline step by step.
- [docs/architecture.md](docs/architecture.md): a diagram of one generation and the module list.
- [docs/grading_rubric.md](docs/grading_rubric.md): the grades and the ground truth.
- [sphere/README.md](sphere/README.md): running the evaluation on a SPHERE experiment node.

## Setup

Use Python 3.11 or newer. Run every command from the repo root.

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements-lock.txt
pip check
pip install --no-deps -e .        # or set PYTHONPATH=src
python -m llmdpeval.verify_env
```

`requirements-lock.txt` is the exact environment the grids ran in; `run_grid` refuses to start if the installed
packages differ from it. It also pins `weave`, which the code does not import.

API keys are read from the environment variable, or else from a key file:

- `sphere_key.txt` (`SPHERE_LLM_API_KEY`): sphere-llm.net models and embeddings
- `openai_key.txt` (`OPENAI_API_KEY`), `anthropic_key.txt` (`ANTHROPIC_API_KEY`), `gemini_key.txt` (`GOOGLE_API_KEY`)

Key files live in `~/.llmdpeval_keys/` (Windows: `%USERPROFILE%\.llmdpeval_keys\`) or in the folder
`LLMDPEVAL_KEY_DIR` names. If neither exists, the repo root (gitignored) is read, with a warning. Keep them outside
the repo: generated code never gets the keys in its environment, but on a local machine it runs with your user
rights and could read a key file by its path.

## Corpus and index

```bash
python -m llmdpeval.check_corpus --workers 4     # outlines, skeleton tests, theory cards
python -m llmdpeval.rag build                    # embed outlines + theory cards (needs the sphere key)
python -m llmdpeval.rag eval --query fields --filter framework --k 1 3 5
```

`corpus/` holds `opendp/`, `diffprivlib/`, `numpy/` (outlines `.md` and verified skeletons `.py`), `theory/`
(theory cards) and `api_index/` (API lists used while writing; not retrieved). The format is in
`corpus/FORMAT.md`, sources and licences in `corpus/SOURCES.md`. The index is saved under `storage/`. When the
corpus, the embedded metadata, the embedding model or the node format change, `rag build` rebuilds it;
generation and repair stop with a message to run `rag build` first, because several generations run at once and
must not write the same index.

## Generate

```bash
python -m llmdpeval.run_grid --grid-dir generated_code/grid_A     # every setting from pipeline.toml
python -m llmdpeval.run_grid --grid-dir generated_code/grid_B --runs 4 \
    --conditions OpenDP-rag-k3 Diffprivlib-rag-k3 None-rag-k3 --theory --code --think off
```

- `--conditions`: `None`, `Diffprivlib`, `OpenDP` (task prompt only) and `<framework>-rag-k<K>` (K retrieved
  outlines). K values come from `rag_k` in `pipeline.toml`.
- `--template guided|plain`, `--theory`, `--code`, `--holdout`, `--repair N`, `--think on|off`, `--hard-cap SECONDS`:
  see docs/method.md. `pipeline.toml` sets `repair = 0`, so generation only calls the model and can run on an XDC.
- `--models` (default: `models` in `pipeline.toml`), `--tasks` (default: stat_count, selection_topk,
  ml_regression; `app_<name>` for a paper application).

Each cell is a folder `<model>_<rag part>_<framework><flags>` with `<task>_code_<run>.py` and a generation log.
Existing files are skipped, so the same command resumes a stopped grid. A generation killed by the hard cap is
marked `<task>_code_<run>.timeout` and is not generated again. One generation can also be run alone:
`python -m llmdpeval.generate --llm gemma4:latest --task stat_count --framework OpenDP --k 3 --output-dir DIR`.

Two prompts differ from the upstream versions: `stat_composition` used `Master` (no row matches; the data has
`Masters`), and `ml_regression` took its bounds from the data. Grids made with the upstream prompts are not
comparable on these two tasks.

## Where code runs

Generated code is untrusted and never runs on an XDC. On Linux it runs only on a SPHERE experiment node as a
separate user with no network (`sphere/README.md`, `LLMDPEVAL_SANDBOX_USER`); on a local machine the sandbox is a
scratch folder, an allowlisted environment and a timeout, which is not isolation (docs/method.md, section 8).
`run_grid --evaluate`, `run_grid --repair`, `generate --repair`, `repair` and `grade median` refuse to start on
Linux when the hostname contains `xdc`, or anywhere when `LLMDPEVAL_NO_EXEC=1` (set it on an XDC whose name has
no `xdc`).

## Evaluate and repair

```bash
python -m llmdpeval.run_grid --grid-dir generated_code/grid_A --evaluate --eval-workers 4
python -m llmdpeval.repair generated_code/grid_A generated_code/grid_A_fix2 --rounds 2
python -m llmdpeval.run_grid --grid-dir generated_code/grid_A_fix2 --evaluate --eval-workers 4
python -m llmdpeval.repair --selftest
```

`--evaluate` writes `<grid>/results.jsonl`, one line per script. Code that `sandbox.is_unsafe` flags (network,
processes, file deletion) is not run (`"status": "unsafe_not_run"`) and is graded by hand; an answer with no code
is graded N. `repair` picks the crashed and the privacy-flagged scripts from `results.jsonl` and sends one kind of
feedback per round (docs/method.md, section 7). Evaluate the repaired grid afterwards.

## Grade

```bash
python -m llmdpeval.grade triage generated_code/grid_A           # flags to check first -> triage.csv
python -m llmdpeval.grade table generated_code/grid_A --runs 4   # hand_grades.csv -> grades.csv + summary
python -m llmdpeval.grade median generated_code/grid_A --n 20    # G vs U on the median of 20 re-runs (P stays P)
python -m llmdpeval.app_grade generated_code/grid_A              # suggested grades for paper applications
```

Hand grades (`hand_grades.csv`: cell, task, run, grade, reason) stay next to the grid outputs, not in the repo.
`grade median` re-runs only the three toy tasks; paper applications keep their hand grade.

## Checks

```bash
python -m pytest tests/
python -m llmdpeval.helper && python -m llmdpeval.sandbox && python -m llmdpeval.repair --selftest
```

## Paper applications

`applications/<app>/` holds `original.py` (the analysis without DP), `prompt.txt`, `spec.md` (privacy setup,
expected releases, utility target), `reference_dp.py` and `ceiling.py`. The data goes in `applications/<app>/data/`
and the reference runs in `runs/`; both are gitignored. Run an application as a grid task, e.g.
`--tasks app_gopher_sqf --hard-cap 3600`.

## Reproductions

`reproductions/<paper>/` holds the recipe to re-run a published paper's own code: `fetch.sh` (pinned commit),
`env/` (locks and setup), `run.sh`, `compare.py` and `expected/` where the paper's numbers had to be typed in.
`RESULT.md` records what matched. The papers' code, data and run outputs are not in git.

## Layout

- `src/llmdpeval/`: the package (module list in docs/architecture.md).
- `prompts/`: toy task prompts. `data/`: `adult_clean.csv` (UCI Adult) and `auto_mpg_clean.csv` (UCI Auto MPG).
- `corpus/`: outlines, skeletons and theory cards. `rag_eval/`: dev tasks and relevance labels for retrieval metrics.
- `applications/`, `reproductions/`: paper applications and paper reproductions.
- `pipeline.toml`: every generation setting in one place.
- `sphere/`: the SPHERE node model and setup script.
- `tests/`: regression checks (no LLM, no network).
