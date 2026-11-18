"""Offline retrieval metrics: hit@K / MRR (eval), theory-card routing (theory-eval), paper-app coverage (app-eval)
and the models' own code analyses (analyze-apps). Run through `python -m llmdpeval.rag`."""
import json
from pathlib import Path

import yaml

from llmdpeval.helper import NO_FRAMEWORK, REPO_ROOT
from llmdpeval.prompts import analyze_code
from llmdpeval.retrieval import (FRAMEWORK_TAG, LABELS, PIPELINE, TEST_TASKS, corpus_hash, load_index, plan_nodes,
                                 prompt_for, rerank_device, retrieve, retrieve_nodes, retrieve_theory)

THEORY_LABELS = REPO_ROOT / "rag_eval" / "theory_labels.yaml"  # {query_id: [acceptable theory card ids]}


def run_theory_eval(args) -> None:
    """Which theory card each eval query gets vs rag_eval/theory_labels.yaml (DEV and TEST separately)."""
    labels = yaml.safe_load(THEORY_LABELS.read_text())
    index, theory = load_index(args.embed), load_index(args.embed, corpus="theory")
    rows = []
    for qid, good in labels.items():
        for fw in FRAMEWORK_TAG:
            prompt = prompt_for(qid, fw)
            nodes = retrieve_nodes(index, prompt, fw, 1, PIPELINE["query"], PIPELINE["filter"])
            card = retrieve_theory(theory, prompt, fw, nodes, args.mode).node.node_id
            rows.append((qid, fw, card, card in good))
            print(f"{qid:<17} {fw:<12} {'ok  ' if card in good else 'MISS'} {card}  (outline task_type: "
                  f"{nodes[0].node.metadata.get('task_type') if nodes else None})")
    for name, sel in (("DEV", lambda q: q not in TEST_TASKS), ("TEST", lambda q: q in TEST_TASKS)):
        part = [r for r in rows if sel(r[0])]
        print(f"{name:<5} theory routing {sum(r[3] for r in part)}/{len(part)} correct (mode={args.mode})")


def score(ranked: list[str], relevant: set[str], k: int) -> tuple[int, float]:
    """hit@k (any relevant id in the top k) and reciprocal rank of the first relevant id."""
    hit = int(any(i in relevant for i in ranked[:k]))
    rr = next((1 / (r + 1) for r, i in enumerate(ranked) if i in relevant), 0.0)
    return hit, rr


def summary(rows: list[dict], ks: list[int], name: str) -> str:
    """One line of mean hit@k and MRR over `rows`."""
    n = len(rows) or 1
    return (f"{name:<5} " + " ".join(f"hit@{k}={sum(r[f'hit@{k}'] for r in rows) / n:.2f}" for k in ks)
            + f" mrr={sum(r['mrr'] for r in rows) / n:.2f}  (n={len(rows)})")


def run_eval(args) -> None:
    """hit@K and MRR of the outline retrieval for every labelled task x framework (rag_eval/labels.yaml),
    reported separately for the dev tasks and the held-out grid tasks."""
    labels = yaml.safe_load(LABELS.read_text())
    index = None if args.local_embed else load_index(args.embed)
    depth = max(args.k)
    rows = []
    for task, by_fw in labels.items():
        for fw, relevant in by_fw.items():
            if not relevant:  # corpus gap: nothing to find
                print(f"GAP  {task:<16} {fw}")
                continue
            ranked = retrieve(index, task, fw, depth, args.query, args.filter, args.search, args.rerank, args.local_embed)
            hits = {k: score(ranked, set(relevant), k)[0] for k in args.k}
            rows.append({"task": task, "framework": fw, "mrr": score(ranked, set(relevant), depth)[1],
                         **{f"hit@{k}": v for k, v in hits.items()}, "top": ranked[:depth]})
    for r in rows:
        print(f"{r['task']:<16} {r['framework']:<12} " +
              " ".join(f"hit@{k}={r[f'hit@{k}']}" for k in args.k) + f" mrr={r['mrr']:.2f}  {r['top'][:3]}")
    print(summary([r for r in rows if r["task"] not in TEST_TASKS], args.k, "DEV"))
    print(summary([r for r in rows if r["task"] in TEST_TASKS], args.k, "TEST"))
    if args.out:
        Path(args.out).write_text(json.dumps({"config": {**vars(args), "device": rerank_device() if (args.rerank != "none" or args.local_embed) else None}, "corpus": corpus_hash(), "rows": rows}, indent=2))


APP_LABELS = REPO_ROOT / "rag_eval" / "app_labels.yaml"
APP_VARIANTS = {"dense": (False, False), "boost": (True, False), "composition": (False, True), "boost+composition": (True, True)}


def run_app_eval(args) -> None:
    """Paper-app retrieval: is a correct item for each expected release in the final (merged) context, and is a
    budget-splitting item there? Releases come from app_labels.yaml (the spec), or from a JSON file of analyses
    ({app: {model: [releases]}}) so the models' own task identification is measured too."""
    labels = yaml.safe_load(APP_LABELS.read_text(encoding="utf-8"))
    sources = {"spec": {app: rels for app, rels in labels["apps"].items()}}
    if args.analyses:
        for app, by_model in json.loads(Path(args.analyses).read_text(encoding="utf-8")).items():
            for model, rels in by_model.items():
                sources.setdefault(model, {})[app] = rels
    index = load_index(args.embed)
    rows = []
    for source, apps in sources.items():
        for app, rels in apps.items():
            expected = labels["apps"][app]
            for fw in ("OpenDP", "Diffprivlib", NO_FRAMEWORK):
                prompt = f"x\n3. Use the following privacy parameters: epsilon = 1.\n" + (
                    "" if fw == NO_FRAMEWORK else f"4. Use the following DP framework: {fw}.\n")
                for variant, (boost, comp) in APP_VARIANTS.items():
                    ids = [n.node.node_id for n in plan_nodes(index, rels, prompt, fw, args.k, PIPELINE["query"],
                                                              PIPELINE["filter"], boost, comp)]
                    for e in expected:
                        if e["relevant"][fw]:
                            rows.append({"source": source, "app": app, "fw": fw, "variant": variant,
                                         "item": e["name"], "hit": bool(set(e["relevant"][fw]) & set(ids))})
                    rows.append({"source": source, "app": app, "fw": fw, "variant": variant, "item": "composition",
                                 "hit": bool(set(labels["composition"][fw]) & set(ids))})
    for source in sources:
        print(f"\nreleases from: {source}")
        for variant in APP_VARIANTS:
            rel = [r["hit"] for r in rows if r["source"] == source and r["variant"] == variant and r["item"] != "composition"]
            comp = [r["hit"] for r in rows if r["source"] == source and r["variant"] == variant and r["item"] == "composition"]
            print(f"  {variant:18} releases covered {sum(rel)}/{len(rel)}   budget item present {sum(comp)}/{len(comp)}")
    misses = [r for r in rows if not r["hit"] and r["variant"] == "boost+composition" and r["source"] == "spec"]
    print("\nspec misses with boost+composition:", [(r["app"], r["fw"], r["item"]) for r in misses])
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=1), encoding="utf-8")


def run_analyze_apps(args) -> None:
    """Ask each model for its own code analysis of each paper app (the same ANALYSIS_PROMPT the pipeline uses),
    and save {app: {model: [releases]}} for app-eval --analyses, plus the raw answers."""
    import os
    from llmdpeval.helper import configure_models, load_prompt
    if args.think:
        os.environ["SPHERE_THINK"] = args.think
    labels = yaml.safe_load(APP_LABELS.read_text(encoding="utf-8"))
    out, raw_out = {}, {}
    for model in args.models:
        configure_models(llm_model_name=model, temperature=0.0)
        for app in labels["apps"]:
            releases, raw = analyze_code(load_prompt(f"app_{app}", args.framework))
            out.setdefault(app, {})[model] = releases
            raw_out.setdefault(app, {})[model] = raw
            print(f"{model} {app}: {len(releases)} releases, task_types {[r['task_types'] for r in releases]}", flush=True)
        Path(args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")  # after each model: a stopped run keeps its part
        Path(args.out).with_suffix(".raw.json").write_text(json.dumps(raw_out, indent=1), encoding="utf-8")
