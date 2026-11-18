"""Retrieval: the outline corpus (corpus/<framework>/) and theory cards (corpus/theory/), their indexes, query
building, dense/BM25/hybrid search, re-ranking, task-type boost and the per-release plan for paper applications."""
import functools
import json
import re

import yaml
from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.schema import TextNode

from llmdpeval.helper import (CORPUS, DEFAULT_EMBED_MODEL, FRAMEWORK_DIRS, NO_FRAMEWORK, PIPELINE_CONFIG, REPO_ROOT,
                              STORAGE, TASKS, THEORY, configure_models, lf_hash, load_prompt)

LABELS = REPO_ROOT / "rag_eval" / "labels.yaml"  # {query_id: {framework: [relevant outline ids]}}
DEV_TASKS = REPO_ROOT / "rag_eval" / "dev_tasks.yaml"
TEST_TASKS = TASKS  # the grid tasks: held out, never tuned on
# Metadata shown to the embedder/LLM; the rest (source_url, license, api_calls) stays out of the vector text.
EMBED_KEYS = ["title", "framework", "mechanism", "task_type", "topic"]

QUERY_TYPE_PATTERN = re.compile(r"^Query type:\s*(.+)$", flags=re.MULTILINE)
QUERY_DEFINITION_PATTERN = re.compile(r"^Query definition:\s*(.+)$", flags=re.MULTILINE)


def build_retrieval_query(prompt: str) -> str:
    """The original pipeline's retrieval query: the Query type and Query definition lines only."""
    query_type = QUERY_TYPE_PATTERN.search(prompt)
    query_definition = QUERY_DEFINITION_PATTERN.search(prompt)
    if query_type is None or query_definition is None:
        raise ValueError("Prompt is missing Query type or Query definition.")
    return f"Query type: {query_type.group(1)}\nQuery definition: {query_definition.group(1)}"


# Theory cards (corpus/theory) have their own index and their own slot in the prompt.
# The outline glob lists the framework folders explicitly, so theory cards never enter the outline index.
CORPORA = {"outlines": (CORPUS, FRAMEWORK_DIRS), "theory": (THEORY, ["."])}


def corpus_files(corpus: str = "outlines") -> list:
    """The .md files of a corpus (\"outlines\" or \"theory\"), sorted."""
    root, folders = CORPORA[corpus]
    return sorted(p for folder in folders for p in (root / folder).glob("*.md"))


def load_outlines(corpus: str = "outlines") -> list[TextNode]:
    """One TextNode per corpus file: the body as text, the YAML front matter as metadata (lists joined to strings);
    only EMBED_KEYS are shown to the embedder and the LLM."""
    nodes = []
    for path in corpus_files(corpus):
        _, front, body = path.read_text(encoding="utf-8").split("---\n", 2)
        meta = yaml.safe_load(front)
        for key, value in meta.items():  # flat values only in vector stores
            if isinstance(value, list):
                meta[key] = ", ".join(map(str, value))
        nodes.append(TextNode(
            id_=meta["id"], text=body.strip(), metadata=meta,
            excluded_embed_metadata_keys=[k for k in meta if k not in EMBED_KEYS],
            excluded_llm_metadata_keys=[k for k in meta if k not in EMBED_KEYS],
        ))
    return nodes


def corpus_hash(corpus: str = "outlines") -> str:
    """Hash of the corpus files: relative path + contents with LF line endings, in sorted order."""
    return lf_hash(corpus_files(corpus), CORPORA[corpus][0])


def skeleton_hash() -> str:
    """Hash of the code skeletons (*.py next to the outlines). Skeletons are not embedded, so they are not part of
    corpus_hash or the index stamp, but a grid must not continue after a skeleton changes."""
    return lf_hash(sorted(p for folder in FRAMEWORK_DIRS for p in (CORPUS / folder).glob("*.py")), CORPUS)


NODE_FORMAT = "1"  # bump when load_outlines changes the text or metadata that gets embedded


def index_stamp(embed: str, corpus: str = "outlines") -> str:
    """What an index depends on: corpus contents, embedded metadata keys, embed model and node format."""
    return json.dumps({"corpus": corpus_hash(corpus), "embed": embed, "embed_keys": EMBED_KEYS,
                       "node_format": NODE_FORMAT}, sort_keys=True)


def load_index(embed: str = DEFAULT_EMBED_MODEL, corpus: str = "outlines", build: bool = True) -> VectorStoreIndex:
    """Load the index of `corpus` for `embed`, rebuilding it if anything in index_stamp changed since it was built.
    build=False (generation and repair, which can run several processes at once) never rebuilds: a missing or
    stale index stops the run with a message to run `python -m llmdpeval.rag build` first, so parallel workers
    never write the same index folder."""
    configure_models(embed_model_name=embed)
    persist = STORAGE / f"{corpus}_{re.sub(r'[^A-Za-z0-9]+', '_', embed)}"
    stamp = persist / "corpus_hash.txt"
    if stamp.exists() and stamp.read_text() == index_stamp(embed, corpus):
        return load_index_from_storage(StorageContext.from_defaults(persist_dir=persist))
    if not build:
        raise SystemExit(f"The {corpus} index for {embed} in {persist} is missing or out of date. Build it first: "
                         f"python -m llmdpeval.rag build --embed {embed}")
    nodes = load_outlines(corpus)
    print(f"Embedding {len(nodes)} {corpus} with {embed} -> {persist}")
    index = VectorStoreIndex(nodes, show_progress=True)
    index.storage_context.persist(persist_dir=persist)
    stamp.write_text(index_stamp(embed, corpus))
    return index


def render_dev_prompt(spec: dict, framework: str) -> str:
    """A dev task in the same template as prompts/*.txt, so every pipeline step sees the same kind of input."""
    fw = "" if framework == NO_FRAMEWORK else f"4. Use the following DP framework: {framework}.\n"
    return (f"You are given a csv file `data.csv`.\n\n"
            f"Write executable Python code to compute one differentially private query answer for the following task:\n\n"
            f"Query type: {spec['type']}\nQuery definition: {spec['definition']}\n"
            f"Target columns: {spec['columns']}\n\nRequirements:\n"
            f"1. Compute the true, non-private answer.\n2. Compute a differentially private version.\n"
            f"3. Use the following privacy parameters: {spec['params']}.\n{fw}"
            f"Print the true answer and the private answer. Return only executable Python code.\n")


def prompt_for(qid: str, framework: str) -> str:
    """The prompt of a retrieval-eval query: a grid task's own prompt, or a dev task rendered in the same form."""
    if qid in TEST_TASKS:
        return load_prompt(qid, framework)
    return render_dev_prompt(yaml.safe_load(DEV_TASKS.read_text())[qid], framework)


REQUIREMENT_FIELDS = re.compile(r"^\d\. Use the following (privacy parameters|DP framework)")
BULLET = re.compile(r"^\s+-\s")


def requirement_lines(prompt: str) -> list[str]:
    """The privacy-parameter and framework requirement lines, each followed by its indented bullet lines (a prompt
    can list its parameters as "3. Use the following privacy parameters:" plus "    - epsilon = 1" lines)."""
    keep, in_fields = [], False
    for line in prompt.splitlines():
        if REQUIREMENT_FIELDS.match(line):
            keep.append(line)
            in_fields = True
        elif in_fields and BULLET.match(line):
            keep.append(line)
        else:
            in_fields = False
    return keep


def make_query(prompt: str, mode: str) -> str:
    """The retrieval query for a prompt. Modes: "will" = the Query type and Query definition lines (the original
    pipeline's query); "full" = the whole prompt; "fields" = "will" plus the privacy-parameter and framework
    requirement lines ("Use plain NumPy" when no framework is named)."""
    if mode == "full":
        return prompt
    query = build_retrieval_query(prompt)
    if mode == "fields":
        keep = [l.split(". ", 1)[1] if REQUIREMENT_FIELDS.match(l) else l.strip() for l in requirement_lines(prompt)]
        query += "\n" + "\n".join(keep)
        if not any("DP framework" in l for l in keep):
            query += "\nUse plain NumPy, no DP framework."
    return query


FRAMEWORK_TAG = {"OpenDP": "opendp", "Diffprivlib": "diffprivlib", NO_FRAMEWORK: "numpy"}


@functools.lru_cache(maxsize=None)
def bm25_retriever(depth: int, framework_tag: str | None):
    """BM25 over the outline corpus, built once per (depth, framework filter) in a process."""
    from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters
    from llama_index.retrievers.bm25 import BM25Retriever
    filters = (MetadataFilters(filters=[ExactMatchFilter(key="framework", value=framework_tag)])
               if framework_tag else None)
    return BM25Retriever.from_defaults(nodes=load_outlines(), similarity_top_k=depth, filters=filters)


RRF_K = 60  # standard reciprocal-rank-fusion constant (Cormack et al. 2009)
FUSION_DEPTH = 20  # candidates taken from each retriever before fusing


def retrieve_nodes(index: VectorStoreIndex, prompt: str, framework: str, k: int, query_mode: str = "will",
                   filter_mode: str = "none", search: str = "dense"):
    """search = dense (embeddings) | bm25 (keywords) | hybrid (both, reciprocal rank fusion).
    filter=framework keeps only outlines of the prompt's framework (None -> numpy), in both retrievers."""
    from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters
    filters = (MetadataFilters(filters=[ExactMatchFilter(key="framework", value=FRAMEWORK_TAG[framework])])
               if filter_mode == "framework" else None)
    query = make_query(prompt, query_mode)
    depth = k if search != "hybrid" else max(k, FUSION_DEPTH)
    runs = []
    if search in ("dense", "hybrid"):
        runs.append(index.as_retriever(similarity_top_k=depth, filters=filters).retrieve(query))
    if search in ("bm25", "hybrid"):
        tag = FRAMEWORK_TAG[framework] if filter_mode == "framework" else None
        runs.append(bm25_retriever(depth, tag).retrieve(query))
    if len(runs) == 1:
        return runs[0][:k]
    fused, by_id = {}, {}
    for run in runs:
        for rank, n in enumerate(run):
            fused[n.node.node_id] = fused.get(n.node.node_id, 0.0) + 1 / (RRF_K + rank + 1)
            by_id[n.node.node_id] = n
    return [by_id[i] for i in sorted(fused, key=fused.get, reverse=True)[:k]]


RERANKERS = {  # local cross-encoders on CPU, pinned to exact Hugging Face revisions
    "minilm": ("cross-encoder/ms-marco-MiniLM-L-6-v2", "233902d25c440f23af6f7d6e94d2946bac0bee0a"),
    "bge": ("BAAI/bge-reranker-base", "2cfc18c9415c912f9d8155881c133215df768a70"),
}
RERANK_DEPTH = 10  # candidates the cross-encoder re-scores
_cross_encoders = {}


def rerank_device() -> str:
    """GPU when present (speed only); RAG_DEVICE=cpu forces the reproducible CPU path."""
    import os
    import torch
    return os.environ.get("RAG_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")


def rerank(prompt: str, query_mode: str, nodes, name: str, k: int):
    """Re-score `nodes` against the query with a pinned cross-encoder (RERANKERS) and keep the best k."""
    from sentence_transformers import CrossEncoder
    if name not in _cross_encoders:
        model, revision = RERANKERS[name]
        _cross_encoders[name] = CrossEncoder(model, revision=revision, device=rerank_device())
    query = make_query(prompt, query_mode)
    scores = _cross_encoders[name].predict([(query, n.node.get_content()) for n in nodes])
    return [n for _, n in sorted(zip(scores, nodes), key=lambda p: -p[0])][:k]


LOCAL_EMBEDDERS = {  # local sentence-transformers embedders (no trust_remote_code), pinned revisions
    "bge-base": ("BAAI/bge-base-en-v1.5", "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a"),
    "bge-large": ("BAAI/bge-large-en-v1.5", "d4aa6901d3a41ba39fb536a557fa166f842b0e09"),
    "mpnet": ("sentence-transformers/all-mpnet-base-v2", "e8c3b32edf5434bc2275fc9bab85f82640a19130"),
}
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "  # from the BGE model card
_local = {}


def local_retrieve(name: str, prompt: str, framework: str, k: int, query_mode: str, filter_mode: str) -> list[str]:
    """Dense retrieval with a local embedder (GPU via rerank_device()); same text and filter as the index path."""
    import numpy as np
    from llama_index.core.schema import MetadataMode
    from sentence_transformers import SentenceTransformer
    if name not in _local:
        model_id, revision = LOCAL_EMBEDDERS[name]
        model = SentenceTransformer(model_id, revision=revision, device=rerank_device())
        nodes = load_outlines()
        docs = model.encode([n.get_content(metadata_mode=MetadataMode.EMBED) for n in nodes], normalize_embeddings=True)
        _local[name] = (model, nodes, docs)
    model, nodes, docs = _local[name]
    query = make_query(prompt, query_mode)
    q = model.encode([(BGE_QUERY_PREFIX if name.startswith("bge") else "") + query], normalize_embeddings=True)[0]
    order = np.argsort(-(docs @ q))
    keep = [i for i in order if filter_mode != "framework" or nodes[i].metadata["framework"] == FRAMEWORK_TAG[framework]]
    return [nodes[i].node_id for i in keep[:k]]


def retrieve(index: VectorStoreIndex, qid: str, framework: str, k: int, query_mode: str = "will",
             filter_mode: str = "none", search: str = "dense", reranker: str = "none",
             local_embed: str | None = None) -> list[str]:
    """Outline ids for an eval query, best first."""
    prompt = prompt_for(qid, framework)
    if local_embed:
        return local_retrieve(local_embed, prompt, framework, k, query_mode, filter_mode)
    if reranker == "none":
        nodes = retrieve_nodes(index, prompt, framework, k, query_mode, filter_mode, search)
    else:
        nodes = retrieve_nodes(index, prompt, framework, max(k, RERANK_DEPTH), query_mode, filter_mode, search)
        nodes = rerank(prompt, query_mode, nodes, reranker, k)
    return [n.node.node_id for n in nodes]


# The retrieval settings used for generation (pipeline.toml `query` and `filter`).
PIPELINE = {"query": PIPELINE_CONFIG["query"], "filter": PIPELINE_CONFIG["filter"]}


MAX_PLAN_CONTEXTS = 6  # cap on outlines in a planned (multi-release) prompt, so long code still fits
BOOST_DEPTH = 10  # dense candidates per release; those whose task_type matches the release's move to the front
COMPOSITION_QUERY = ("Query type: Composition\nQuery definition: basic sequential composition under pure "
                     "epsilon-DP (delta = 0): split one total epsilon budget across several Laplace or objective-"
                     "perturbation releases so the epsilons add up to the total, and check the total spent")


def fallback_release(prompt: str) -> dict:
    """One release holding the prompt itself, on one line: the fields query reads only the first line of
    'Query definition:', so newlines would cut it to the prompt's first sentence."""
    return {"query_type": "", "definition": " ".join(prompt[:2000].split()), "target_columns": [], "conditions": []}


def release_prompt(release: dict, prompt: str) -> str:
    """One release in the grid prompt's field format, plus the task's privacy/framework lines, so make_query
    builds the same 'fields' query the grid tasks use."""
    keep = requirement_lines(prompt)
    return "\n".join([f"Query type: {release['query_type']}", f"Query definition: {release['definition']}",
                      f"Target columns: {release['target_columns']}", f"Conditions: {release['conditions']}", *keep])


def boost_by_task_type(nodes: list, task_types: list[str], k: int) -> list:
    """Stable re-order: candidates whose task_type is one of the release's task_types first, then the rest,
    each in dense order; keep k. No (valid) task_types -> the dense top-k unchanged."""
    if not task_types:
        return nodes[:k]
    match = [n for n in nodes if n.node.metadata.get("task_type") in task_types]
    return (match + [n for n in nodes if n not in match])[:k]


def plan_nodes(index: VectorStoreIndex, releases: list[dict], prompt: str, framework: str, k: int,
               query_mode: str, filter_mode: str, boost: bool = True, composition: bool = True) -> list:
    """Top-k per release (task-type boosted within the dense top BOOST_DEPTH), merged round-robin (best of each
    release first), deduplicated, capped. With 2+ releases one extra slot holds a pure-DP budget-splitting item."""
    depth = BOOST_DEPTH if boost else k
    runs = [retrieve_nodes(index, release_prompt(r, prompt), framework, depth, query_mode, filter_mode)
            for r in releases]
    runs = [boost_by_task_type(run, r.get("task_types", []) if boost else [], k) for run, r in zip(runs, releases)]
    merged, seen = [], set()
    for rank in range(k):
        for run in runs:
            if rank < len(run) and run[rank].node.node_id not in seen:
                seen.add(run[rank].node.node_id)
                merged.append(run[rank])
    cap = MAX_PLAN_CONTEXTS
    if composition and len(releases) >= 2:
        comp = [n for n in retrieve_nodes(index, COMPOSITION_QUERY, framework, BOOST_DEPTH, "full", filter_mode)
                if n.node.metadata.get("task_type") == "composition"
                and n.node.metadata.get("dp_notion") == "pure"]  # the tasks state epsilon with delta = 0
        if comp and comp[0].node.node_id not in {n.node.node_id for n in merged[:cap]}:
            return merged[:cap] + [comp[0]]  # an extra slot: it never displaces a release's item
    return merged[:cap]


THEORY_ROUTING = "tagged"  # "tagged": cards whose task_types include the top outline's task_type; "top1": plain dense


def retrieve_theory(theory_index: VectorStoreIndex, prompt: str, framework: str, outline_nodes, mode: str = None):
    """The one theory card for a prompt. tagged: the best-scoring card among those tagged for the task type of the
    top retrieved outline (its task_type metadata); falls back to the best card overall if none is tagged."""
    mode = mode or THEORY_ROUTING
    cards = retrieve_nodes(theory_index, prompt, framework, 50 if mode == "tagged" else 1, PIPELINE["query"])
    task = outline_nodes[0].node.metadata.get("task_type") if outline_nodes else None
    if task == "other":  # a generic top outline says nothing about the task: rank all cards
        return cards[0]
    tagged = [c for c in cards if task in {t.strip() for t in str(c.node.metadata.get("task_types", "")).split(",")}]
    return (tagged or cards)[0] if mode == "tagged" else cards[0]
