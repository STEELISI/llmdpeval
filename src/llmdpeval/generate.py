"""Generate code for one task: load the prompt, optionally add retrieved context, call the LLM, save the code.

  python -m llmdpeval.generate --llm gemma4:latest --task stat_count --framework OpenDP --k 3 --theory --code \
      --output-dir generated_code/try

--k 0 (the default) sends the task prompt alone. run_grid calls this once per generation, in a subprocess.
The log format (the last "Filled prompt" banner is the final prompt) is read back by `repair`.
"""
import argparse
import os
from pathlib import Path

from llama_index.core import Settings

from llmdpeval.helper import (DEFAULT_EMBED_MODEL, FRAMEWORKS, LLM_MODELS, PIPELINE_CONFIG, code_status,
                              configure_models, load_prompt, print_prompt, response_to_code, safe_path_part)

# --reask: sent once after the prompt when the answer had no code block (off by default: the first answer is graded)
REASK = ("\n\nYour previous answer contained no code. Return the complete program as executable Python code "
         "in one ```python block.")


def parse_args() -> argparse.Namespace:
    """Command-line options for one generation."""
    p = argparse.ArgumentParser()
    p.add_argument("--llm", required=True, choices=list(LLM_MODELS))
    p.add_argument("--embed", default=DEFAULT_EMBED_MODEL, help="embedding model for retrieval")
    p.add_argument("--task", default="stat_count", help="prompts/<task>.txt, or app_<name> for applications/<name>/")
    p.add_argument("--framework", default="OpenDP", choices=FRAMEWORKS,
                   help="DP framework named in requirement 4 of the prompt. None removes requirement 4.")
    p.add_argument("--k", type=int, default=0, help="outlines retrieved into the prompt; 0 = no retrieval")
    p.add_argument("--template", choices=["guided", "plain"], default=PIPELINE_CONFIG["template"],
                   help="guided = plan + adapt-don't-copy + print the accounted epsilon; plain = context only")
    p.add_argument("--theory", action="store_true", help="add the single best DP theory card before the outlines")
    p.add_argument("--code", action="store_true", help="follow each outline with its verified code skeleton")
    p.add_argument("--holdout", action="store_true", help="with --code: hide the grid tasks' own skeletons")
    p.add_argument("--repair", type=int, default=0, help="up to N rounds of running the code and sending crashes back")
    p.add_argument("--think", choices=["on", "off"], help="sphere models: explicit think flag (Ollama-native endpoint)")
    p.add_argument("--reask", action="store_true", help="ask once more if the answer has no code (default off)")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--run-index", type=int, default=1, help="run number in the file name")
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def generate_code(final_prompt: str, task: str, output_dir: Path, run_index: int, reask: bool = False) -> Path:
    """Send the final prompt to Settings.llm and save the extracted code as <task>_code_<run_index>.py.
    With reask, a prose-only answer is asked once more for code. Returns the code file."""
    print(f"\nGeneration run {run_index}")
    response = Settings.llm.complete(final_prompt)
    if reask and code_status(response_to_code(response)) == "no_code":
        print("code status: no_code; re-asking once (--reask)", flush=True)
        response = Settings.llm.complete(final_prompt + REASK)
    raw = response.raw if isinstance(response.raw, dict) else {}  # Ollama-native path only (--think)
    if raw:
        message = raw.get("message") or {}
        thinking = message.get("thinking") if isinstance(message, dict) else getattr(message, "thinking", None)
        print(f"usage: thinking_chars={len(thinking or '')} eval_count={raw.get('eval_count')} "
              f"prompt_eval_count={raw.get('prompt_eval_count')} done_reason={raw.get('done_reason')}", flush=True)
    code_file = output_dir / f"{safe_path_part(task)}_code_{run_index}.py"
    code = response_to_code(response)
    print(f"code status: {code_status(code)}", flush=True)  # ok | truncated | no_code (helper.response_to_code)
    code_file.write_text(code, encoding="utf-8")
    return code_file


def main() -> None:
    """Build the prompt (with retrieval when --k > 0), generate, and optionally repair."""
    args = parse_args()
    if args.repair:
        from llmdpeval.sandbox import check_exec_allowed
        check_exec_allowed()  # repair runs the generated code: never on the XDC
    if args.think:
        os.environ["SPHERE_THINK"] = args.think  # read by helper.build_sphere_llm
    configure_models(embed_model_name=args.embed if args.k else None, llm_model_name=args.llm,
                     temperature=args.temperature)
    print(f"Using task: {args.task}")
    print(f"Using DP framework: {args.framework}")
    prompt = load_prompt(args.task, framework=args.framework)
    print_prompt(prompt)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving generated code to: {output_dir}")

    final_prompt = prompt
    if args.k:
        from llmdpeval import rag
        # never rebuilt here: run_grid starts several of these at once (python -m llmdpeval.rag build first)
        theory_index = rag.load_index(args.embed, "theory", build=False) if args.theory else None
        final_prompt, ids = rag.build_rag_prompt(rag.load_index(args.embed, build=False), prompt, args.framework, args.k,
                                                 args.template, theory_index, args.code, args.holdout)
        print(f"rag pipeline {rag.PIPELINE}, template {args.template}, theory {args.theory}, code {args.code}, "
              f"holdout {args.holdout}, corpus {rag.corpus_hash()}, outlines {ids}")
        print_prompt(final_prompt)
    elif args.theory or args.code or args.holdout:
        print("note: --theory/--code/--holdout need --k > 0; ignored")

    code_file = generate_code(final_prompt, args.task, output_dir, args.run_index, args.reask)
    if args.repair:
        from llmdpeval.repair import repair_loop
        print(f"repair log: {repair_loop(Settings.llm, code_file, final_prompt, args.repair, task_prompt=prompt)}")


if __name__ == "__main__":
    main()
