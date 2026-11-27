"""Shared configuration and utilities: repo paths, pipeline settings, model clients, API keys, prompt loading and
code extraction from model answers."""
import ast
import hashlib
import json
import os
import re
import sys
import time
import tomllib
import urllib.error
import urllib.request
from pathlib import Path

from llama_index.core import Settings
from llama_index.core.llms import CompletionResponse, CustomLLM, LLMMetadata
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.anthropic import Anthropic
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI
from llama_index.llms.openai_like import OpenAILike

# Every path is relative to the repo root, so the scripts work from any cwd.
REPO_ROOT = Path(__file__).resolve().parents[2]
PROMPT_DIR = REPO_ROOT / "prompts"
APP_DIR = REPO_ROOT / "applications"  # paper applications: <app>/prompt.txt with {code} = <app>/original.py
DATA_DIR = REPO_ROOT / "data"
CORPUS = REPO_ROOT / "corpus"
FRAMEWORK_DIRS = ["opendp", "diffprivlib", "numpy"]  # outline folders; corpus/theory and corpus/api_index are not outlines
THEORY = CORPUS / "theory"
STORAGE = REPO_ROOT / "storage"
LOCK_FILE = REPO_ROOT / "requirements-lock.txt"
PIPELINE_FILE = REPO_ROOT / "pipeline.toml"


def load_pipeline(path: Path = PIPELINE_FILE) -> dict:
    """The generation pipeline settings (pipeline.toml): the defaults for helper, retrieval and run_grid."""
    return tomllib.loads(path.read_text(encoding="utf-8"))


PIPELINE_CONFIG = load_pipeline()

TASKS = ["stat_count", "selection_topk", "ml_regression"]  # the grid tasks: held out, never tuned on
NO_FRAMEWORK = "None"
FRAMEWORKS = ["OpenDP", "Diffprivlib", NO_FRAMEWORK]

DP_FRAMEWORK_REQUIREMENT_PATTERN = re.compile(
    r"^4\. Use the following DP framework: .+\.$",
    flags=re.MULTILINE,
)
API_KEY_ENV_VARS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "aws": "AWS_BEARER_TOKEN_BEDROCK",
    "azure_openai": "AZURE_OPENAI_API_KEY",
    "gemini": "GOOGLE_API_KEY",
    "openai": "OPENAI_API_KEY",
    "sphere": "SPHERE_LLM_API_KEY",
}
MAX_OUTPUT_TOKENS = PIPELINE_CONFIG["max_output_tokens"]
# sphere-llm models are reasoning models; completion tokens include the reasoning.
SPHERE_MAX_OUTPUT_TOKENS = PIPELINE_CONFIG["sphere_max_output_tokens"]
SPHERE_API_BASE = "https://sphere-llm.net:8194/api"
# sphere-llm also proxies Ollama, which serves the embeddings (the OpenAI-style /api has none).
SPHERE_OLLAMA_BASE = "https://sphere-llm.net:8194/ollama"

# Default: pipeline.toml `embedder`, served by sphere-llm, so RAG needs no local Ollama.
DEFAULT_EMBED_MODEL = PIPELINE_CONFIG["embedder"]
EMBED_MODELS = {
    "nomic-embed-text:latest": "sphere",
    "bge-m3:latest": "sphere",  # full tag required
    "qwen3-embedding:4b": "sphere",
    "nomic-embed-text": "ollama",
    "text-embedding-3-large": "openai",
}
LLM_MODELS = {
    "gemma4:latest": "sphere",
    "granite4.2:30b": "sphere",
    "qwen3.6:27b": "sphere",
    "qwen3:32b": "sphere",
    "qwen3.6:35b": "sphere",
    "gpt-5-mini": "azure_openai",
    "gemini-3.5-flash-lite": "gemini",
    "gemini-4-hydrogen": "gemini",
    "qwen.qwen3-coder-480b-a35b-instruct": "aws",
    "gpt-5.4": "openai",
    "gpt-5.4-nano": "openai",
    "gemini-3.1-flash-lite": "gemini",
    "claude-haiku-4-5": "anthropic",
    "llama3.2": "ollama",
    "qwen2.5-coder:7b": "ollama",
    "codellama:7b": "ollama",
}
SPHERE_MODELS = [m for m, p in LLM_MODELS.items() if p == "sphere"]


DEFAULT_KEY_DIR = Path.home() / ".llmdpeval_keys"
_warned_key_dir = False


def key_dir() -> Path:
    """Folder with the <provider>_key.txt files: $LLMDPEVAL_KEY_DIR if set, else ~/.llmdpeval_keys. If that
    folder does not exist, the repo root is used, with a warning. Keys belong outside the repo: generated code
    runs with your rights and could read a file inside it."""
    global _warned_key_dir
    if os.environ.get("LLMDPEVAL_KEY_DIR"):
        return Path(os.environ["LLMDPEVAL_KEY_DIR"])
    if DEFAULT_KEY_DIR.is_dir():
        return DEFAULT_KEY_DIR
    if not _warned_key_dir:
        _warned_key_dir = True
        print(f"warning: {DEFAULT_KEY_DIR} does not exist; reading key files from the repo root. Move them out of "
              "the repo (or set LLMDPEVAL_KEY_DIR).", file=sys.stderr)
    return REPO_ROOT


def load_api_key(provider: str | None) -> None:
    """Put the provider's key in its env var: an already set env var wins, else <provider>_key.txt in key_dir()."""
    env_var = API_KEY_ENV_VARS.get(provider)
    if env_var and not os.environ.get(env_var):
        os.environ[env_var] = (key_dir() / f"{provider}_key.txt").read_text(encoding="utf-8").strip()


def build_llm(model_name: str, temperature: float = 0.0):
    """A llama-index LLM client for a model in LLM_MODELS, with its provider's API key loaded."""
    if model_name not in LLM_MODELS:
        raise ValueError(f"Unsupported LLM model: {model_name}. Choose from: {', '.join(LLM_MODELS)}")
    provider = LLM_MODELS[model_name]
    load_api_key(provider)
    builders = {
        "anthropic": build_anthropic_llm,
        "aws": build_bedrock_llm,
        "azure_openai": build_azure_llm,
        "gemini": build_gemini_llm,
        "ollama": build_ollama_llm,
        "openai": build_openai_llm,
        "sphere": build_sphere_llm,
    }
    return builders[provider](model_name, temperature)


def build_embedding(model_name: str):
    """A llama-index embedding client for a model in EMBED_MODELS, with its provider's API key loaded."""
    if model_name not in EMBED_MODELS:
        raise ValueError(f"Unsupported embedding model: {model_name}. Choose from: {', '.join(EMBED_MODELS)}")
    provider = EMBED_MODELS[model_name]
    load_api_key(provider)
    if provider == "ollama":
        return OllamaEmbedding(model_name=model_name)
    if provider == "openai":
        return OpenAIEmbedding(model=model_name)
    return build_sphere_embedding(model_name)


def configure_models(embed_model_name: str | None = None, llm_model_name: str | None = None,
                     temperature: float = 0.0) -> None:
    """Set llama-index's global embedding model and/or LLM (Settings.embed_model, Settings.llm)."""
    if embed_model_name:
        Settings.embed_model = build_embedding(embed_model_name)
        print(f"Using {EMBED_MODELS[embed_model_name]} embedding model: {embed_model_name}")
    if llm_model_name:
        Settings.llm = build_llm(llm_model_name, temperature)
        print(f"Using {LLM_MODELS[llm_model_name]} LLM model: {llm_model_name}")


def build_openai_llm(model_name: str, temperature: float = 0.0):
    """OpenAI client."""
    return OpenAI(model=model_name, temperature=temperature, max_tokens=MAX_OUTPUT_TOKENS)


def build_ollama_llm(model_name: str, temperature: float = 0.0):
    """Local Ollama client."""
    return Ollama(
        model=model_name,
        temperature=temperature,
        request_timeout=float(PIPELINE_CONFIG["ollama_timeout"]),
        additional_kwargs={"num_predict": MAX_OUTPUT_TOKENS},
    )


class PatientGoogleGenAI(GoogleGenAI):
    """GoogleGenAI that waits out a 503 (model overloaded) or a 429 (rate limit) once. The library's own retries
    give up within 60 s."""

    def complete(self, prompt: str, formatted: bool = False, **kwargs) -> CompletionResponse:
        from google.genai.errors import APIError
        try:
            return super().complete(prompt, formatted=formatted, **kwargs)
        except APIError as e:
            if e.code not in (429, 503):
                raise
            print(f"{e.code} from {self.model}; retrying once in {OVERLOAD_WAIT_S} s", flush=True)
            time.sleep(OVERLOAD_WAIT_S)
            return super().complete(prompt, formatted=formatted, **kwargs)


def build_gemini_llm(model_name: str, temperature: float = 0.0):
    """Google Gemini client. Gemini 3.5+ models reason: high thinking level and the reasoning output budget."""
    if model_name in THINKING_GEMINI_MODELS:
        from google.genai import types
        out = GEMINI_MAX_OUTPUT_TOKENS.get(model_name, SPHERE_MAX_OUTPUT_TOKENS)
        config = types.GenerateContentConfig(
            temperature=temperature, max_output_tokens=out,
            thinking_config=types.ThinkingConfig(thinking_level="HIGH"))
        cls = PatientGoogleGenAI if model_name in PATIENT_GEMINI_MODELS else GoogleGenAI
        return cls(model=model_name, temperature=temperature, max_tokens=out, generation_config=config)
    return GoogleGenAI(model=model_name, temperature=temperature, max_tokens=MAX_OUTPUT_TOKENS)


THINKING_GEMINI_MODELS = {"gemini-3.5-flash-lite", "gemini-4-hydrogen"}
# gemini-4-hydrogen reasons past 32768 tokens on the paper applications; 65536 is its output limit.
GEMINI_MAX_OUTPUT_TOKENS = {"gemini-4-hydrogen": 65536}
PATIENT_GEMINI_MODELS = {"gemini-4-hydrogen"}
OVERLOAD_WAIT_S = 300
# Azure AI Foundry resource, e.g. https://<resource>.services.ai.azure.com (the Responses API path is added).
AZURE_ENDPOINT_ENV = "AZURE_OPENAI_ENDPOINT"
AZURE_RETRY_WAITS_S = (30, 120, 300)


class AzureResponsesLLM(CustomLLM):
    """Azure OpenAI (AI Foundry) Responses API, text in and text out. Reasoning models take no temperature;
    the reasoning effort is set instead."""
    model: str
    effort: str = "high"
    max_output_tokens: int = 32768
    timeout_s: float = 1800.0

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(model_name=self.model, num_output=self.max_output_tokens, is_chat_model=False)

    def complete(self, prompt: str, formatted: bool = False, **kwargs) -> CompletionResponse:
        body = {"model": self.model, "input": prompt, "reasoning": {"effort": self.effort},
                "max_output_tokens": self.max_output_tokens}
        if not os.environ.get(AZURE_ENDPOINT_ENV):
            raise RuntimeError(f"set {AZURE_ENDPOINT_ENV} to the Azure AI Foundry resource URL")
        url = os.environ[AZURE_ENDPOINT_ENV].rstrip("/") + "/openai/v1/responses"
        req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={
            "Content-Type": "application/json", "api-key": os.environ["AZURE_OPENAI_API_KEY"]})
        for attempt, wait in enumerate(AZURE_RETRY_WAITS_S + (None,)):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
                    data = json.load(r)
                break
            except urllib.error.HTTPError as e:  # rate limit or transient server error: wait and retry
                if e.code not in (429, 500, 502, 503, 504) or wait is None:
                    raise
                print(f"{e.code} from {self.model}; retry {attempt + 1} in {wait} s", flush=True)
                time.sleep(wait)
        text = "".join(c.get("text", "") for o in data.get("output", []) if o.get("type") == "message"
                       for c in o.get("content", []) if c.get("type") == "output_text")
        if not text and data.get("status") != "completed":  # e.g. the output budget ran out while reasoning
            text = f"No answer: response status {data.get('status')} {data.get('incomplete_details')}"
        return CompletionResponse(text=text, raw=data.get("usage"))

    def stream_complete(self, prompt: str, formatted: bool = False, **kwargs):
        raise NotImplementedError("streaming is not used by the pipeline")


def build_azure_llm(model_name: str, temperature: float = 0.0):
    """Azure OpenAI reasoning model (temperature does not apply)."""
    return AzureResponsesLLM(model=model_name, max_output_tokens=SPHERE_MAX_OUTPUT_TOKENS,
                             timeout_s=float(PIPELINE_CONFIG["sphere_timeout"]))


# AWS Bedrock's OpenAI-compatible endpoint (Bedrock API key as a bearer token).
BEDROCK_API_BASE = "https://bedrock-mantle.us-east-1.api.aws/v1"


def build_bedrock_llm(model_name: str, temperature: float = 0.0):
    """AWS Bedrock open-weight model. Same output budget and timeout as the other paper-app models; the OpenAI
    client retries rate limits and server errors (max_retries)."""
    return OpenAILike(model=model_name, api_base=BEDROCK_API_BASE, api_key=os.environ["AWS_BEARER_TOKEN_BEDROCK"],
                      is_chat_model=True, temperature=temperature, max_tokens=SPHERE_MAX_OUTPUT_TOKENS,
                      timeout=float(PIPELINE_CONFIG["sphere_timeout"]), max_retries=3)


def build_anthropic_llm(model_name: str, temperature: float = 0.0):
    """Anthropic client."""
    return Anthropic(model=model_name, temperature=temperature, max_tokens=MAX_OUTPUT_TOKENS)


def build_sphere_llm(model_name: str, temperature: float = 0.0):
    """A sphere-llm.net client. SPHERE_THINK=on|off selects the Ollama-native endpoint with an explicit think
    flag (the only per-request switch these reasoning models honour); unset, the OpenAI-style /api endpoint."""
    think = os.environ.get("SPHERE_THINK")
    if think:
        if think not in ("on", "off"):
            raise ValueError(f"SPHERE_THINK must be on or off, got {think!r}")
        return Ollama(
            model=model_name,
            base_url=SPHERE_OLLAMA_BASE,
            headers={"Authorization": f"Bearer {os.environ['SPHERE_LLM_API_KEY']}"},
            thinking=think == "on",
            temperature=temperature,
            request_timeout=float(PIPELINE_CONFIG["sphere_timeout"]),
            additional_kwargs={"num_predict": SPHERE_MAX_OUTPUT_TOKENS},
        )
    return OpenAILike(
        model=model_name,
        api_base=SPHERE_API_BASE,
        api_key=os.environ["SPHERE_LLM_API_KEY"],
        is_chat_model=True,
        temperature=temperature,
        max_tokens=SPHERE_MAX_OUTPUT_TOKENS,
        # qwen3.6 can reason past 10 minutes on Top-K; a timeout would be scored as a failure.
        timeout=float(PIPELINE_CONFIG["sphere_timeout"]),
        # No silent retries: a call over the timeout is recorded once as a timeout.
        max_retries=0,
    )


class RetryingOllamaEmbedding(OllamaEmbedding):
    """Ollama embeddings that retry a query that times out or fails to connect. The shared GPU can queue an
    embedding behind long generation calls; a retry returns the same vector, so results do not change."""

    def get_query_embedding(self, query: str):
        for attempt in range(EMBED_RETRIES):
            try:
                return super().get_query_embedding(query)
            except Exception as e:  # httpx timeout / connection errors from the Ollama client
                if attempt == EMBED_RETRIES - 1 or not any(s in type(e).__name__ for s in ("Timeout", "Connect")):
                    raise
                print(f"embedding retry {attempt + 1}/{EMBED_RETRIES - 1} after {type(e).__name__}", flush=True)
                time.sleep(EMBED_RETRY_WAIT_S)


EMBED_RETRIES = 4  # one try plus up to three retries
EMBED_RETRY_WAIT_S = 30


def build_sphere_embedding(model_name: str):
    """sphere-llm.net embedding client (its Ollama proxy; the OpenAI-style /api serves no embeddings)."""
    return RetryingOllamaEmbedding(
        model_name=model_name,
        base_url=SPHERE_OLLAMA_BASE,
        client_kwargs={
            "headers": {"Authorization": f"Bearer {os.environ['SPHERE_LLM_API_KEY']}"},
            "timeout": PIPELINE_CONFIG["embed_timeout"],  # long: generation calls can hold the shared GPU
        },
    )


def load_prompt(task: str, framework: str = "OpenDP") -> str:
    """The task prompt with its framework requirement set (or removed for "None"). `app_<name>` loads
    applications/<name>/prompt.txt with {code} replaced by original.py; any other task loads prompts/<task>.txt."""
    if task.startswith("app_"):  # e.g. app_gopher_sqf -> applications/gopher_sqf/
        app = APP_DIR / task.removeprefix("app_")
        prompt = (app / "prompt.txt").read_text(encoding="utf-8")
        prompt = prompt.replace("{code}", (app / "original.py").read_text(encoding="utf-8"))
        return set_prompt_framework(prompt, framework)
    prompt_path = PROMPT_DIR / f"{task}.txt"
    if not prompt_path.is_file():
        raise ValueError(f"Unsupported task: {task}. Choose from: {available_prompts()}")
    return set_prompt_framework(prompt_path.read_text(encoding="utf-8"), framework)


def set_prompt_framework(prompt: str, framework: str) -> str:
    """Rewrite requirement 4 ("Use the following DP framework: X.") to name `framework`, or drop it for "None"."""
    if framework == NO_FRAMEWORK:
        return remove_prompt_framework(prompt)
    prompt, replacement_count = DP_FRAMEWORK_REQUIREMENT_PATTERN.subn(
        f"4. Use the following DP framework: {framework}.", prompt, count=1)
    if replacement_count == 0:
        raise ValueError("Prompt is missing requirement 4 for the DP framework.")
    return prompt


def remove_prompt_framework(prompt: str) -> str:
    """Drop requirement 4 and renumber the later requirements (5 -> 4, ...). Only text after requirement 4 is
    renumbered, so numbered lines inside inlined code above it are left alone."""
    match = re.search(DP_FRAMEWORK_REQUIREMENT_PATTERN.pattern + r"\r?\n", prompt, flags=re.MULTILINE)
    if match is None:
        raise ValueError("Prompt is missing requirement 4 for the DP framework.")
    tail = re.sub(r"^(\d+)\. ", lambda m: f"{int(m.group(1)) - 1}. ", prompt[match.end():], flags=re.MULTILINE)
    return prompt[:match.start()] + tail


def available_prompts() -> list[str]:
    """Names of the toy tasks in prompts/."""
    return sorted(p.stem for p in PROMPT_DIR.glob("*.txt"))


def print_prompt(prompt: str, title: str = "Filled prompt") -> None:
    """Print a prompt under a title rule."""
    print(f"\n{title}")
    print("=" * 128)
    print(prompt)


def safe_path_part(value: str) -> str:
    """`value` reduced to letters, digits, '.' and '-', for use in a folder or file name."""
    return re.sub(r"[^A-Za-z0-9.-]+", "-", value).strip(".-") or "unknown"


NO_CODE = "# llmdpeval: no_code"  # first line of a saved answer that had no code (graded N, never run)
TRUNCATED = "# llmdpeval: truncated (no closing code fence)"  # the answer was cut off inside the code block
CODE_START = re.compile(r"^\s*(import |from \S+ import |def |class |print\(|if __name__)", re.M)


PYTHON_TAGS = {"", "python", "py", "python3"}


def response_to_code(response) -> str:
    """The first fenced block tagged python/py (any case) or untagged; if every block has another tag (bash,
    json, ...), the first block. An opening fence with no closing fence (a truncated answer): the text after it,
    marked TRUNCATED. No fence: the whole text if it parses as Python or has code-like lines, else the text
    commented out under NO_CODE (an answer that is only prose)."""
    text = getattr(response, "text", None) or str(response)
    if not text.strip():  # an empty answer has no code (ast.parse would accept the empty string)
        return NO_CODE + "\n"
    blocks = re.findall(r"```([A-Za-z0-9_+-]*)[ \t]*\n?(.*?)```", text, flags=re.DOTALL)
    if blocks:
        code = next((body for tag, body in blocks if tag.lower() in PYTHON_TAGS), blocks[0][1])
        return code.strip() + "\n" if code.strip() else NO_CODE + "\n"
    opening = re.search(r"```[A-Za-z0-9_+-]*[ \t]*\n", text)
    if opening:
        return f"{TRUNCATED}\n{text[opening.end():].strip()}\n"
    try:
        ast.parse(text)
        return text.strip() + "\n"
    except SyntaxError:
        pass
    if CODE_START.search(text):
        return text.strip() + "\n"
    return NO_CODE + "\n" + "".join(f"# {line}\n" for line in text.strip().splitlines())


def code_status(code: str) -> str:
    """ok, truncated or no_code, from the marker response_to_code puts on the first line."""
    first = code.split("\n", 1)[0]
    return "no_code" if first == NO_CODE else "truncated" if first == TRUNCATED else "ok"


def file_key(path: str | Path) -> tuple[str, str, int]:
    """(cell, task, run) from .../<cell>/<task>_code_<run>.py, whatever the path prefix or separator."""
    p = Path(str(path).replace("\\", "/"))
    task, run = p.stem.rsplit("_code_", 1)
    return p.parent.name, task, int(run)


def code_files(grid: Path) -> list[Path]:
    """The final generated scripts of a grid; repair keeps each round as <task>_code_<n>.round<r>.py."""
    return [f for f in sorted(Path(grid).glob("*/*_code_*.py")) if ".round" not in f.name]


def lf_hash(paths, root: Path | None = None) -> str:
    """First 16 hex digits of a sha256 over the files in order, each with LF line endings (the same on Windows and
    Linux checkouts). With root, each file's path relative to root (POSIX form) is hashed before its contents."""
    h = hashlib.sha256()
    for path in paths:
        prefix = Path(path).relative_to(root).as_posix().encode() if root else b""
        h.update(prefix + Path(path).read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()[:16]


def keep_awake() -> None:
    """Keep Windows awake while this process runs: Modern Standby freezes long model calls (they fail with
    "Server disconnected") and pauses the hard cap. Sets ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED;
    no-op elsewhere."""
    if os.name == "nt":
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000003)


if __name__ == "__main__":  # self-check of the pure helpers
    assert response_to_code("```py\nprint(1)\n```") == "print(1)\n"
    assert response_to_code("```python\nx = 1\n```") == "x = 1\n"
    assert response_to_code("```\nx = 1\n```") == "x = 1\n"
    assert response_to_code("x = 1") == "x = 1\n"
    assert response_to_code("```bash\npip install x\n```\n```python\nprint(2)\n```") == "print(2)\n"
    assert response_to_code("```json\n{}\n```") == "{}\n"
    assert safe_path_part("gemma4:latest") == "gemma4-latest"
    assert "qwen3.6:35b" in SPHERE_MODELS and len(SPHERE_MODELS) == 5, SPHERE_MODELS
    p = load_prompt("stat_count", NO_FRAMEWORK)
    assert "DP framework" not in p and "\n4. Print" in p
    assert file_key("generated_code\\g\\cell_x_OpenDP\\stat-count_code_3.py") == ("cell_x_OpenDP", "stat-count", 3)
    os.environ["LLMDPEVAL_KEY_DIR"] = "/somewhere/else"
    assert key_dir() == Path("/somewhere/else")
    print("helper self-check ok")
