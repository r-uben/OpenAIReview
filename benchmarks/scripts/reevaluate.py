"""Re-evaluate saved benchmark results using the LLM judge.

Loads a results JSONL produced by run_benchmark.py, runs the LLM judge
for each (predicted, ground-truth) pair, and saves an updated JSONL with
the additional llm_judge metrics. This avoids re-running the expensive
review API calls.

Usage:
    uv run python scripts/reevaluate.py results/results_<timestamp>.jsonl
    uv run python scripts/reevaluate.py results/results_<timestamp>.jsonl --judge-model openai/gpt-4o-mini
"""

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT.parent / "src"))

from reviewer.evaluate import evaluate, llm_judge_is_match, print_report
from reviewer.models import Comment, ReviewResult

BENCHMARK_FILE = ROOT / "data" / "benchmark.jsonl"


def load_ground_truth() -> dict[str, list[dict]]:
    gt = {}
    with open(BENCHMARK_FILE) as f:
        for line in f:
            paper = json.loads(line)
            gt[paper["slug"]] = paper["comments"]
    return gt


def result_from_dict(d: dict) -> ReviewResult:
    r = ReviewResult(
        method=d["method"],
        paper_slug=d["paper_slug"],
        model=d.get("model", ""),
        total_prompt_tokens=d.get("total_prompt_tokens", 0),
        total_completion_tokens=d.get("total_completion_tokens", 0),
    )
    for c in d.get("comments", []):
        r.comments.append(Comment(
            title=c.get("title", ""),
            quote=c.get("quote", ""),
            explanation=c.get("explanation", ""),
            comment_type=c.get("comment_type", "logical"),
            paragraph_index=c.get("paragraph_index"),
        ))
    return r


def main():
    parser = argparse.ArgumentParser(description="Re-evaluate results with LLM judge")
    parser.add_argument("results_file", help="Path to results JSONL from run_benchmark.py")
    parser.add_argument(
        "--judge-model",
        default=os.environ.get("JUDGE_MODEL", "anthropic/claude-haiku-4-5"),
        help="Model to use as LLM judge",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path (default: <input>_llm_judge.jsonl)",
    )
    args = parser.parse_args()

    input_path = Path(args.results_file)
    output_path = Path(args.output) if args.output else input_path.with_stem(
        input_path.stem + "_llm_judge"
    )

    gt_by_slug = load_ground_truth()

    records = []
    with open(input_path) as f:
        for line in f:
            records.append(json.loads(line))

    print(f"Re-evaluating {len(records)} records with LLM judge ({args.judge_model})")
    print(f"Output: {output_path}\n")

    all_results = []
    with open(output_path, "w") as out:
        for i, record in enumerate(records):
            method = record["method"]
            slug = record["paper_slug"]
            ground_truth = gt_by_slug.get(slug, [])
            result = result_from_dict(record["result"])

            print(f"[{i+1}/{len(records)}] {method} / {slug} "
                  f"({result.num_comments} predicted, {len(ground_truth)} GT)")

            metrics = evaluate(
                result, ground_truth,
                use_llm_judge=True,
                judge_model=args.judge_model,
            )

            print(f"  sim_recall={metrics['recall']:.2f}  "
                  f"llm_recall={metrics['recall_llm']:.2f}  "
                  f"llm_precision={metrics['precision_llm']:.2f}")

            record["metrics"] = metrics
            out.write(json.dumps(record) + "\n")
            all_results.append((method, slug, metrics))

    print_report(all_results)
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
