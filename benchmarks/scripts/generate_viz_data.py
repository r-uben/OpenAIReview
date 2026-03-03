"""Generate viz/data.json from benchmark + results for the HTML viewer."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT.parent / "src"))

from reviewer.utils import split_into_paragraphs, locate_comment_in_document

BENCHMARK = ROOT / "data" / "benchmark.jsonl"
RESULTS = ROOT / "results" / "full_benchmark_with_location.jsonl"
OUTPUT = ROOT / "viz" / "data.json"

METHOD_LABELS = {
    "ground_truth": "Ground Truth (Expert)",
    "zero_shot": "Zero-Shot",
    "few_shot": "Few-Shot",
    "rag_local": "RAG Local",
    "rag_retrieved": "RAG Retrieved",
    "rag_retrieved_cot": "RAG Retrieved CoT",
    "rag_top_k_filter": "RAG Top-K Filter",
}

METHOD_ORDER = [
    "ground_truth",
    "zero_shot",
    "few_shot",
    "rag_local",
    "rag_retrieved",
    "rag_retrieved_cot",
    "rag_top_k_filter",
]


def build_gt_method(paper: dict) -> dict:
    """Build the ground_truth method entry from benchmark data."""
    comments = []
    for i, c in enumerate(paper["comments"]):
        comments.append({
            "id": f"gt_{i}",
            "title": c.get("title", ""),
            "quote": c.get("quote", ""),
            "explanation": c.get("message", ""),
            "comment_type": c.get("comment_type", "technical"),
            "paragraph_index": c.get("paragraph_index"),
            "score": c.get("score"),
        })
    return {
        "label": METHOD_LABELS["ground_truth"],
        "metrics": None,
        "overall_feedback": paper.get("overall_feedback", ""),
        "comments": comments,
    }


def build_predicted_method(record: dict, paragraphs: list[str]) -> tuple[str, dict]:
    """Build a predicted method entry from a results record.

    Re-locates each comment's paragraph_index against the current paragraph
    split to avoid stale indices from older runs.
    """
    # Use result.method for RAG variants, fallback to top-level method
    method_key = record["result"].get("method", record["method"])
    comments = []
    for i, c in enumerate(record["result"]["comments"]):
        quote = c.get("quote", "")
        # Re-locate against current paragraphs
        para_idx = locate_comment_in_document(quote, paragraphs) if quote else None
        comments.append({
            "id": f"{method_key}_{i}",
            "title": c.get("title", ""),
            "quote": quote,
            "explanation": c.get("explanation", ""),
            "comment_type": c.get("comment_type", "technical"),
            "paragraph_index": para_idx,
        })
    return method_key, {
        "label": METHOD_LABELS.get(method_key, method_key),
        "metrics": record.get("metrics"),
        "overall_feedback": record["result"].get("overall_feedback", ""),
        "comments": comments,
    }


def main():
    # Load benchmark papers
    papers_raw = []
    with open(BENCHMARK) as f:
        for line in f:
            papers_raw.append(json.loads(line))

    # Load results, indexed by paper slug
    results_by_slug: dict[str, list[dict]] = {}
    with open(RESULTS) as f:
        for line in f:
            record = json.loads(line)
            slug = record["paper_slug"]
            results_by_slug.setdefault(slug, []).append(record)

    # Build output
    papers_out = []
    for paper in papers_raw:
        slug = paper["slug"]
        paragraphs = split_into_paragraphs(paper["document_content"])

        methods = {}
        # Ground truth
        methods["ground_truth"] = build_gt_method(paper)

        # Predicted methods
        for record in results_by_slug.get(slug, []):
            method_key, method_data = build_predicted_method(record, paragraphs)
            methods[method_key] = method_data

        papers_out.append({
            "slug": slug,
            "title": paper["title"],
            "authors": paper.get("authors", ""),
            "field": paper.get("field", ""),
            "paragraphs": [{"index": i, "text": t} for i, t in enumerate(paragraphs)],
            "methods": methods,
        })

    output = {
        "papers": papers_out,
        "method_order": [m for m in METHOD_ORDER if any(m in p["methods"] for p in papers_out)],
    }

    OUTPUT.parent.mkdir(exist_ok=True)
    with open(OUTPUT, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Wrote {OUTPUT}")
    print(f"  {len(papers_out)} papers")
    for p in papers_out:
        print(f"  {p['slug']}: {len(p['paragraphs'])} paragraphs, {len(p['methods'])} methods")


if __name__ == "__main__":
    main()
