# Method

This file describes the pipeline in plain words. The code is in `src/llmdpeval/`.

The question: given a task, can an LLM write Python code that computes a correct answer and a valid
differentially private (DP) answer, with the DP framework the task asks for (OpenDP, diffprivlib, or none)?

## 1. Tasks

- **Toy tasks** (`prompts/*.txt`): one query on a public dataset (`data/`). The grid uses three:
  `stat_count` (count of `income == ">50K"`), `selection_topk` (top 5 occupations by share of >50K) and
  `ml_regression` (mpg from weight, predict 4 points). Requirement 4 of each prompt names the framework;
  `helper.set_prompt_framework` rewrites it, and the `None` condition removes it.
- **Paper applications** (`applications/<app>/`): existing, non-private analysis code from a published paper.
  The task is to add DP to it. `prompt.txt` holds the task, with `{code}` replaced by `original.py`.
  `spec.md` states the privacy setup and the expected releases. `reference_dp.py` is a reference DP version, and
  `ceiling.py` runs it 20 times to set the utility bar. The data is fetched separately and is not in git.

## 2. Corpus (`corpus/`)

The corpus is pinned to framework versions: OpenDP 0.16.0, diffprivlib 0.6.6, and NumPy. Each item says which
version it was written for (`framework_version`). The format is in `corpus/FORMAT.md`.

- **Outlines** (`corpus/<framework>/<id>.md`): one file per technique. Front matter (framework, mechanism,
  task type, API calls, source, licence) plus four sections: goal, steps, privacy accounting, pitfalls. Steps are
  pseudo-code, not runnable code. They name the real functions and the order to call them.
- **Verified skeletons** (`corpus/<framework>/<id>.py`): a short runnable function next to most outlines. Each has
  a `_test()` that runs it on made-up data and returns the epsilon the framework reports. `check_corpus` runs
  every `_test()` and fails if one crashes or reports epsilon above 1.
- **Theory cards** (`corpus/theory/*.md`): one DP idea per card (e.g. sensitivity, composition, the exponential
  mechanism, Top-K selection). Each card is tagged with the task types it helps.

`check_corpus` also checks that API names resolve in the pinned environment, and that no item names the test
tasks' data (leak words such as `income` or `mpg`). `retrieval.corpus_hash` hashes the outlines and the cards, so a run
log records exactly which corpus it used.

## 3. Retrieval (`retrieval.py`)

- **Index:** outlines and theory cards are embedded separately with the embedder set in `pipeline.toml`
  (`qwen3-embedding:4b`), served by sphere-llm.
  The index is rebuilt when anything it depends on changes: the corpus hash, the metadata keys that are embedded
  (`retrieval.EMBED_KEYS`), the embedding model or the node format (`retrieval.index_stamp`).
- **Query ("fields"):** the task's `Query type` and `Query definition` lines, plus the privacy-parameter line and
  the framework line. With no framework named, the query says "Use plain NumPy".
- **Framework filter:** only outlines of the requested framework are searched (`None` searches the NumPy outlines).
- **Top K** outlines (K = 1 or 3) go into the prompt, each followed by its skeleton when `--code` is set.
  `--holdout` hides the skeletons of the three grid tasks' own outlines (a leakage check).
- **Theory routing (`--theory`):** one card, in its own slot before the outlines. Among the cards tagged for the
  task type of the top outline, the best-scoring card is chosen. If none is tagged, or the top outline is generic,
  the best card overall is used.
- `python -m llmdpeval.rag eval` measures hit@K and MRR against `rag_eval/labels.yaml`, one pipeline step at a
  time (query, search, filter, rerank). The three grid tasks are held out from tuning.

## 4. Paper applications: the analysis step

A paper application has no `Query type` line. So the model first reads the code and lists every value it
releases (prints, returns or saves) that comes from the private data, as JSON with a query type, definition,
target columns and conditions. Each release becomes its own fields query. The top results of all releases are
merged round-robin (the best of each release first), without duplicates, up to 6 outlines. The model's own list
of releases is shown in the prompt before the outlines.

## 5. Prompt

The default template ("guided") asks the model to adapt the outlines, not copy their constants, and to start the
code with a short plan: the sensitivity of each released value, the mechanism, how epsilon is split, and how it
checks the epsilon actually spent. The code must print the epsilon the framework's own accounting reports.
The "plain" template (`--template plain`) only puts the context after the task, as in the original design.

## 6. Model calls and the think switch

The sphere-llm models are reasoning models. Without `--think`, calls go to the OpenAI-style endpoint. With
`--think on|off`, calls go to the Ollama-native endpoint, which takes an explicit think flag per request. Run 1 of each cell uses temperature 0; later runs use 0.7. Each generation has a
wall-clock cap (`--hard-cap`, 1800 s by default). A run with no answer in time is recorded as a timeout
(`<task>_code_<n>.timeout`) and is not generated again when the grid is resumed.

The grid's conditions (`pipeline.toml`) are `None`, `Diffprivlib`, `OpenDP` and the same three with 3 retrieved
outlines (`<framework>-rag-k3`). `rag_k` lists the K values offered; add 1 to it for `-rag-k1` conditions.

## 7. Repair loop

With `repair = 0` (`pipeline.toml`) generation never runs code. Repair is a separate step after the grid is
evaluated: `python -m llmdpeval.repair SRC OUT` (2 rounds by default), then `run_grid --grid-dir OUT --evaluate`.
`--repair N` on `run_grid` or `generate` runs the same loop inline, where code may run.

The generated code is run in the sandbox. Each round can send one kind of feedback back to the same model, with
the same prompt, up to N rounds in total:

- **Crash:** the script exits with an error. The tail of stderr is sent back.
- **Fake fix:** a "fix" that exits 0 with no output, or that adds a catch-all `except`. It is sent back with a note.
- **Privacy feedback:** the script exits 0, but its own output shows an accounting problem. The checks use only
  the printed output and the budget stated in the prompt (no LLM judge, no code execution tricks):
  1. the prompt asks for the epsilon to be printed (paper applications; the guided RAG template) but no line of
     the output gives a value labelled as epsilon;
  2. a printed epsilon is 0, negative or NaN;
  3. the epsilon spent is above the budget: a printed total (a line with "total" or "overall", or the JSON key
     `epsilon_total`) if there is one, else the sum of per-release lines ("epsilon for X: ..."), else the largest
     printed epsilon. The budget is the first requirement line of the prompt that says `epsilon = X`, so a value
     inside a retrieved outline is never read as the budget. Printed values are rounded, so 0.1% over is allowed.

  The value of an epsilon line is the first number after the first `:` or `=` that follows the word epsilon,
  after dropping parentheses that do not mention epsilon (so "noise scale (b=2): 1.0" reads 1.0). The feedback
  names the check that failed and asks the model to fix the accounting without changing the parts that are
  correct. Comparing a printed value with the framework's own accounting call (OpenDP `map`, diffprivlib
  accountant) would need to run parts of the code, so that check is not done.

The budget, delta and neighbouring relation come from the task prompt itself, before the RAG template and the
outlines are added (the outlines mention other epsilons, delta = 0 and replace-one). Replace-one is read from the
prompt's "(replace-one)" (paper applications); the toy prompts say "(add/remove)". Whether epsilon must be
printed is read from the full prompt, since the guided template asks for it.

Code that imports or calls something that reaches the network, starts processes or deletes files
(`sandbox.is_unsafe`, on the code's syntax tree) is never run (a triage filter, not isolation; see section 8).
Every round is kept as `<task>_code_<n>.round<r>.py`, and `<task>_code_<n>.repair.json` logs each round's status
(`error`, `fake_fix`, `static_privacy`, `privacy_feedback` with the problem text, `ok`, `unsafe_not_run`). If a
repair call fails (an API timeout, ...), the last executed round is copied back to `<task>_code_<n>.py` and the
log ends with `repair_failed`, the error and that round's run result. With inline `--repair`, a run counts as done
only once its `.repair.json` exists; a run killed by the hard cap is marked `.timeout` and keeps the code it had.
`python -m llmdpeval.repair SRC OUT` repairs the crashed and the privacy-flagged scripts of a finished grid with
the same prompts.

## 8. Evaluation sandbox

Each script runs from a copy in a fresh scratch folder, never in the repo. The folder holds copies of only the
data the task needs: `data/*.csv` for the toy tasks, `applications/<app>/data/*` for a paper application. The
data folder is passed as the first argument (`.`, the scratch folder itself). The timeout is 120 s. Results go to
`results.jsonl` in the grid folder, one line per script as soon as it finishes. Code that
`sandbox.is_unsafe` flags is not run; its line has `"status": "unsafe_not_run"` and it is graded by hand. An answer
with no code is not run either and is graded N.

The script gets only an allowlisted set of environment variables (`sandbox.ENV_ALLOW`: paths, temp folders,
locale, thread counts), so the API keys that generation and repair load are not in its environment. On a local
machine this is not isolation. The script runs with the user's rights and has network access, so it could still
read any file the user can, including a key file, by its absolute path. The key files therefore belong outside the
repo folder: `helper.load_api_key` reads them from `$LLMDPEVAL_KEY_DIR`, else `~/.llmdpeval_keys` (the repo root,
with a warning, only if that folder does not exist). `sandbox.is_unsafe` is a triage filter, not a security
boundary.

On the SPHERE evaluation node (`LLMDPEVAL_SANDBOX_USER`), the script runs as a separate user with no network and
no DNS, in a scratch folder owned by that user with mode 700, under a memory limit. Each run starts in its own
process group, which is killed (as the sandbox user) when the script ends or times out, so a background child does
not outlive its run while other runs go on in parallel. A process that leaves its group is removed by
`pkill -u <sandbox user>` at the end of each `run_grid --evaluate` and `grade median` pass, and after each script
in `repair` (which runs one script at a time).

Generated code never runs on an XDC; it runs on a SPHERE experiment node (`sphere/README.md`) or a local
machine. `sandbox.check_exec_allowed` refuses to run it on Linux
when the hostname contains `xdc` (SPHERE XDC hostnames are `<xdc>-<project>`), or anywhere when
`LLMDPEVAL_NO_EXEC=1`; `run_grid --evaluate`, `run_grid --repair`, `generate --repair`, `repair` and
`grade median` check it before they start.

## 9. Grading

Every runnable script is read by hand and graded with the rubric in `docs/grading_rubric.md` (G, P, L, U, W, E,
N). Epsilon is read from the framework, not from comments. `grade triage` flags things to look at first (framework
imports, hand-rolled noise, fixed seeds, unsafe code, true answers in the output); it decides nothing.
`grade table` joins the hand grades with the run results. `grade median` re-runs each valid script 20 times and
decides useful (G) or too noisy (U) on the median error, not on one DP draw (a P stays P, with a note when the median
is not useful). Only the three toy tasks have an
automatic metric; paper applications and code flagged by `sandbox.is_unsafe` are not re-run and keep their hand
grade. The Top-K metric counts how many of the true top 5 are in the printed private top-5 list (the first
bracketed list of 5 names, else the first 5 occupations named in the private part of the output).
