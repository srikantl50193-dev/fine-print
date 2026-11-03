"""Compare models and prompts on the tuning questions, and pick by a rule fixed in advance.

The rule: the highest correct-citation rate, among runs that refuse at least 90% of unanswerable
questions and have a p95 latency under 30 seconds. Ties go to the lower latency.

This only reads the tuning questions. Each run's results are saved in eval/runs/tuning/, and a
run that is already saved is skipped, so an interrupted comparison can be resumed.

Usage: uv run python scripts/compare_answers.py [--model NAME]... [--prompt NAME]... [--force]
"""

import argparse
import statistics
import time
from pathlib import Path

import httpx

from fine_print.config import PROJECT_ROOT, get_settings
from fine_print.evaluate import (
    AnswerResult,
    EvalQuestion,
    OutOfScopeQuestion,
    evaluate_answers,
    load_jsonl,
)
from fine_print.prompts import list_prompts, load_prompt
from fine_print.rag import call_ollama

MODELS = ("llama3.2", "llama3.1:8b", "qwen3:8b", "gemma4:e4b")
RUNS_DIR = PROJECT_ROOT / "eval" / "runs" / "tuning"
MIN_UNANSWERABLE_REFUSED = 0.9
MAX_P95_LATENCY_S = 30.0
SPEED_PROMPT = "Explain in about 100 words what a privacy policy is."
LOAD_TIMEOUT_S = 600


def run_path(model: str, prompt: str) -> Path:
    return RUNS_DIR / f"{model.replace(':', '-')}__{prompt}.json"


def unload(model: str) -> None:
    """Free a model's memory, so that each run starts from the same state."""
    url = f"{get_settings().ollama_url}/api/generate"
    httpx.post(url, json={"model": model, "keep_alive": 0}, timeout=LOAD_TIMEOUT_S)


def measure_model(model: str) -> dict[str, float]:
    """Measure a model's load time, loaded memory and generation speed."""
    url = get_settings().ollama_url
    unload(model)
    options = {"temperature": 0, "num_ctx": get_settings().ollama_num_ctx, "num_predict": 200}
    speeds = []
    load_s = 0.0
    for attempt in range(3):
        body = httpx.post(
            f"{url}/api/generate",
            json={
                "model": model,
                "prompt": SPEED_PROMPT,
                "stream": False,
                "think": False,
                "options": options,
            },
            timeout=LOAD_TIMEOUT_S,
        ).json()
        if attempt == 0:
            load_s = body["load_duration"] / 1e9
        speeds.append(body["eval_count"] / (body["eval_duration"] / 1e9))
    loaded = httpx.get(f"{url}/api/ps", timeout=30).json()["models"]
    size = next(m["size"] for m in loaded if m["name"].startswith(model))
    return {
        "load_s": round(load_s, 1),
        "memory_gb": round(size / 1e9, 1),
        "tokens_per_s": round(statistics.median(speeds), 1),
    }


def run(model: str, prompt_name: str, force: bool) -> AnswerResult:
    path = run_path(model, prompt_name)
    if path.exists() and not force:
        return AnswerResult.model_validate_json(path.read_text(encoding="utf-8"))
    settings = get_settings()
    questions = load_jsonl(settings.questions_path, EvalQuestion)
    off_topic = load_jsonl(settings.out_of_scope_path, OutOfScopeQuestion)
    prompt = load_prompt(prompt_name)
    # Load the model first, so that its load time isn't counted as latency.
    call_ollama("Reply with: ok", prompt.system, model)
    start = time.perf_counter()
    result = evaluate_answers(settings.default_strategy, questions, off_topic, model, prompt)
    print(f"  {model} / {prompt_name}: {time.perf_counter() - start:.0f}s", flush=True)
    path.write_text(result.model_dump_json(indent=1) + "\n", encoding="utf-8")
    return result


def eligible(result: AnswerResult) -> bool:
    return (
        result.unanswerable_refused >= MIN_UNANSWERABLE_REFUSED
        and result.p95_latency_s < MAX_P95_LATENCY_S
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", dest="models")
    parser.add_argument("--prompt", action="append", dest="prompts")
    parser.add_argument("--force", action="store_true", help="repeat runs that are already saved")
    args = parser.parse_args()

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    models = args.models or MODELS
    prompts = args.prompts or list_prompts()

    results = []
    for model in models:
        measured = measure_model(model)
        print(f"{model}: {measured}", flush=True)
        results.extend(run(model, prompt, args.force) for prompt in prompts)
        unload(model)

    print(
        "\n| Model | Prompt | Answered | Correct citation | Wrong refusals "
        "| Unanswerable refused | p95 latency (s) | Eligible |"
    )
    print("|" + " --- |" * 8)
    ordered = sorted(results, key=lambda r: (-r.cited_relevant, r.p95_latency_s))
    for r in ordered:
        print(
            f"| {r.model} | {r.prompt} | {r.answered:.0%} | {r.cited_relevant:.0%} "
            f"| {len(r.wrongly_refused)} | {r.unanswerable_refused:.0%} "
            f"| {r.p95_latency_s:.1f} | {'yes' if eligible(r) else 'no'} |"
        )
    winners = [r for r in ordered if eligible(r)]
    print(f"\nPick: {winners[0].model} / {winners[0].prompt}" if winners else "\nNo eligible run")


if __name__ == "__main__":
    main()
