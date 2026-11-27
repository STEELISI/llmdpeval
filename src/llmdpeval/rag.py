"""Command line for retrieval: build the index and run the offline retrieval metrics.
The code lives in retrieval.py (index, search), prompts.py (prompt building) and rag_eval.py (the metrics).

  python -m llmdpeval.rag build                 # embed outlines + theory cards (needed before generation)
  python -m llmdpeval.rag eval --k 1 3 5        # hit@K / MRR per task x framework against rag_eval/labels.yaml
  python -m llmdpeval.rag theory-eval           # which theory card each eval query gets

Each pipeline step (query, search, filter, rerank) is a --flag on `eval`, so steps are compared one at a time.
"""
import argparse

from llmdpeval.helper import DEFAULT_EMBED_MODEL
# build_rag_prompt, PIPELINE and corpus_hash are imported here for generate and repair (rag.<name>).
from llmdpeval.prompts import build_rag_prompt  # noqa: F401
from llmdpeval.rag_eval import run_analyze_apps, run_app_eval, run_eval, run_theory_eval
from llmdpeval.retrieval import (LOCAL_EMBEDDERS, PIPELINE, RERANKERS, THEORY_ROUTING, corpus_hash,  # noqa: F401
                                 load_index)


def main() -> None:
    """Dispatch the build / eval / theory-eval / app-eval / analyze-apps subcommands."""
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--embed", default=DEFAULT_EMBED_MODEL)
    e = sub.add_parser("eval")
    e.add_argument("--embed", default=DEFAULT_EMBED_MODEL)
    e.add_argument("--k", type=int, nargs="+", default=[1, 3, 5])
    e.add_argument("--query", choices=["will", "full", "fields"], default="will", help="how the retrieval query is built")
    e.add_argument("--search", choices=["dense", "bm25", "hybrid"], default="dense", help="retriever")
    e.add_argument("--local-embed", choices=list(LOCAL_EMBEDDERS), help="local embedder instead of the sphere index")
    e.add_argument("--rerank", choices=["none", *RERANKERS], default="none", help="cross-encoder re-ranking")
    e.add_argument("--filter", choices=["none", "framework"], default="none", help="metadata filter on retrieval")
    e.add_argument("--out", help="write config + per-row results as JSON")
    t = sub.add_parser("theory-eval")
    t.add_argument("--embed", default=DEFAULT_EMBED_MODEL)
    t.add_argument("--mode", choices=["tagged", "top1"], default=THEORY_ROUTING)
    a = sub.add_parser("app-eval")
    a.add_argument("--embed", default=DEFAULT_EMBED_MODEL)
    a.add_argument("--k", type=int, default=3)
    a.add_argument("--analyses", help="JSON {app: {model: [releases]}} from the models' own code analysis")
    a.add_argument("--out", help="write per-row results as JSON")
    z = sub.add_parser("analyze-apps")
    z.add_argument("--models", nargs="+", default=["gemma4:latest", "granite4.2:30b", "qwen3.6:27b", "qwen3:32b"])
    z.add_argument("--framework", default="Diffprivlib")
    z.add_argument("--think", choices=["on", "off"], default="on")
    z.add_argument("--out", required=True)
    args = p.parse_args()
    if args.cmd == "analyze-apps":
        run_analyze_apps(args)
        return
    if args.cmd == "theory-eval":
        run_theory_eval(args)
    elif args.cmd == "app-eval":
        run_app_eval(args)
    elif args.cmd == "build":  # both indexes: generation and repair only load them, never rebuild
        load_index(args.embed)
        load_index(args.embed, "theory")
    else:
        run_eval(args)


if __name__ == "__main__":
    main()
