"""Prompt building: the templates, the analysis step (call 1 for paper applications) and the final RAG prompt."""
import functools
import json
import re

import yaml
from llama_index.core import VectorStoreIndex

from llmdpeval.helper import CORPUS, FRAMEWORK_DIRS
from llmdpeval.retrieval import (LABELS, PIPELINE, TEST_TASKS, fallback_release, plan_nodes, release_prompt,
                                 retrieve_nodes, retrieve_theory)


PROMPT_TEMPLATES = {
    # The original pipeline's template: retrieval only (the ablation).
    "plain": ("Use the documentation context to answer the user prompt.\n\n"
             "User prompt:\n{prompt}\n\nDocumentation context:\n{context}"),
    # The default: adapt-don't-copy + plan before code + print the accounted epsilon.
    "guided": (
        "You are writing differentially private code for the task below. The outlines below show the library's "
        "real functions and the order to call them. Adapt them to this task: do not copy constants, noise scales "
        "or epsilon formulas from them. Use the neighbouring relation the task states; the outlines default to "
        "replace-one (the number of rows is public) and note the add/remove alternative where the two differ.\n"
        "Start the code with a short plan in comments: (1) the sensitivity of each released value under that "
        "relation and why, "
        "(2) the mechanism, (3) how the total epsilon is split across releases, (4) how you check the epsilon "
        "actually spent.\n"
        "The code must print the epsilon the framework's own accounting reports (for example OpenDP's "
        "measurement.map(d_in) or a diffprivlib accountant's spent budget); with plain NumPy, print the "
        "epsilon implied by your sensitivity and noise scale.\n\n"
        "User prompt:\n{prompt}\n\nOutlines:\n{context}"),
}


def task_prompt_of(final_prompt: str) -> str:
    """The task prompt inside a final RAG prompt: the text between a template's "User prompt:" header and its
    context header ("Outlines:" or "Documentation context:"). A prompt that matches no template is returned as is
    (a prompt sent without retrieval is the task prompt itself)."""
    for template in PROMPT_TEMPLATES.values():
        head, tail = template.split("{prompt}")
        marker = tail.split("{context}")[0]
        if final_prompt.startswith(head) and marker in final_prompt[len(head):]:
            return final_prompt[len(head):].split(marker, 1)[0]
    return final_prompt


# The corpus's task_type vocabulary (metadata on every outline), offered to the analysis step so each release can be
# matched to corpus items by metadata, not only by embedding similarity.
TASK_TYPES = {
    "count": "number of rows, or a share of rows over a public total",
    "sum": "sum of a bounded column",
    "mean": "average of a bounded column, overall or per group",
    "variance": "variance or standard deviation",
    "quantile": "median, quartiles, percentiles",
    "histogram": "counts or rates per category or bin, including per-group rates",
    "selection": "choosing the best or top-k items",
    "regression": "linear regression and its predictions",
    "classification": "training a classifier (e.g. logistic regression) and evaluating it",
    "pca": "principal components",
    "inference": "standard errors, p-values or significance tests on DP results",
    "composition": "splitting one privacy budget across several releases",
    "other": "anything else",
}
ANALYSIS_PROMPT = (
    "Below is a task that asks you to add differential privacy to existing analysis code. Do not write code yet.\n"
    "Read the code and list every value it releases (prints, returns or saves) that is computed from the private "
    "data, including trained models and metrics computed on private rows. For each release give: query_type "
    "(a short statistic or model name, e.g. Count, Proportion by group, Mean, Linear regression, Logistic "
    "regression, Top-K), definition (one sentence: what is computed, on which rows), target_columns (list), "
    "conditions (list of row filters, may be empty) and task_types (a list of one or two of these names: "
    + "; ".join(f"{name} = {meaning}" for name, meaning in TASK_TYPES.items()) + ").\n"
    "Answer with only a JSON list of objects with exactly those five keys, in a ```json fenced block.\n\n"
    "Task:\n{prompt}\n\n"
    # Repeated after the task: the task text ends with "write the code" instructions, which models otherwise
    # follow (they answer with DP code and no list of releases).
    "End of task. Do not write code. Answer with only the JSON list of releases (keys query_type, definition, "
    "target_columns, conditions, task_types) in a ```json fenced block.")


def analyze_code(prompt: str) -> tuple[list[dict], str]:
    """Step 1 for paper applications: the generation model itself (Settings.llm) names the releases.
    Returns (releases, raw answer); releases is [] if the answer has no parsable JSON list."""
    from llama_index.core import Settings
    raw = Settings.llm.complete(ANALYSIS_PROMPT.format(prompt=prompt)).text
    match = re.search(r"```(?:json)?\s*(.*?)```", raw, re.S)
    try:
        releases = json.loads(match.group(1) if match else raw)
    except json.JSONDecodeError:
        return [], raw
    keys = ("query_type", "definition", "target_columns", "conditions")
    if not isinstance(releases, list):
        return [], raw
    releases = [r for r in releases if isinstance(r, dict) and all(k in r for k in keys)]
    for r in releases:  # task_types are optional; keep only names from the corpus vocabulary
        tags = r.get("task_types") or []
        r["task_types"] = [t for t in (tags if isinstance(tags, list) else [tags]) if t in TASK_TYPES][:2]
    return releases, raw


@functools.lru_cache(maxsize=1)
def heldout_skeletons() -> frozenset[str]:
    """Leakage check: the outlines labelled correct for the 3 grid tasks (rag_eval/labels.yaml), read once."""
    labels = yaml.safe_load(LABELS.read_text())
    # The Top-K rate skeletons are hidden even if labels.yaml changes (they solve the grid's Top-K task).
    rate_topk = {f"{fw}-topk-rate-two-histograms" for fw in ("opendp", "diffprivlib", "numpy")}
    return frozenset({i for task in TEST_TASKS for ids in labels[task].values() for i in ids} | rate_topk)


def skeleton_for(outline_id: str, holdout: bool = False) -> str | None:
    """The verified code skeleton next to an outline (check_corpus ran its _test), without _test.
    holdout=True hides the skeletons of the grid tasks' own outlines (leakage check)."""
    from llmdpeval.check_corpus import skeleton_without_test
    if holdout and outline_id in heldout_skeletons():
        return None
    hits = [CORPUS / fw / f"{outline_id}.py" for fw in FRAMEWORK_DIRS if (CORPUS / fw / f"{outline_id}.py").exists()]
    return skeleton_without_test(hits[0]) if hits else None


def build_rag_prompt(index: VectorStoreIndex, prompt: str, framework: str, k: int,
                     template: str = "guided", theory_index: VectorStoreIndex | None = None,
                     code: bool = False, holdout: bool = False) -> tuple[str, list[str]]:
    """Final prompt for generation: the top-k outlines as context, in one of PROMPT_TEMPLATES.
    With theory_index, the single best theory card goes first, in its own slot (it never displaces an outline).
    With code, each outline is followed by its verified code skeleton.
    A prompt without a 'Query type:' line (paper applications) is planned first: analyze_code names the releases,
    each release gets its own fields query, and the plan is shown to the model with the context."""
    plan = None
    if "Query type:" in prompt:
        nodes = retrieve_nodes(index, prompt, framework, k, PIPELINE["query"], PIPELINE["filter"])
    else:
        releases, raw = analyze_code(prompt)
        print(f"analysis: {len(releases)} releases\n{raw}", flush=True)
        if not releases:  # unparsable analysis: retrieve on the whole prompt rather than fail the run
            releases = [fallback_release(prompt)]
        nodes = plan_nodes(index, releases, prompt, framework, k, PIPELINE["query"], PIPELINE["filter"])
        plan = json.dumps(releases, indent=1)
    blocks = []
    for i, n in enumerate(nodes, 1):
        block = f"[Context {i}] {n.node.metadata['title']}\n{n.node.text}"
        skeleton = skeleton_for(n.node.node_id, holdout) if code else None
        if skeleton:
            block += ("\n\nVerified code skeleton (real imports and calls; fill in the parameters from the task, "
                      f"do not copy values):\n```python\n{skeleton}```")
        blocks.append(block)
    ids = [n.node.node_id for n in nodes]
    if theory_index is not None:
        # paper apps have no "Query type:" line, so the card query uses the first release's fields
        query_prompt = prompt if plan is None else release_prompt(releases[0], prompt)
        card = retrieve_theory(theory_index, query_prompt, framework, nodes)
        blocks.insert(0, f"[Theory] {card.node.metadata['title']}\n{card.node.text}")
        ids.insert(0, card.node.node_id)
    if plan is not None:
        blocks.insert(0, f"[Your analysis of the code: the releases to make private]\n{plan}")
    return PROMPT_TEMPLATES[template].format(prompt=prompt, context="\n\n".join(blocks)), ids
