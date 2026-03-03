"""CLI entry point for openaireview."""

import argparse
import json
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


DEFAULT_LARGE_MODEL = os.environ.get("LARGE_MODEL", "anthropic/claude-opus-4-5")
DEFAULT_SMALL_MODEL = os.environ.get("SMALL_MODEL", "anthropic/claude-haiku-4-5")


def slugify(name: str) -> str:
    """Convert a name to a URL-friendly slug."""
    s = name.lower().strip()
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"[\s_]+", "-", s)
    s = re.sub(r"-+", "-", s)
    return s.strip("-")[:80]


def cmd_review(args: argparse.Namespace) -> None:
    """Run a review on a document."""
    from .method_incremental import review_incremental
    from .method_rag import review_rag
    from .method_zero_shot import review_zero_shot
    from .parsers import parse_document
    from .utils import split_into_paragraphs

    file_path = Path(args.file)
    if not file_path.exists():
        print(f"Error: file not found: {file_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Parsing {file_path.name}...")
    title, content = parse_document(file_path)
    print(f"  Title: {title}")

    slug = args.name or slugify(Path(file_path).stem)
    paragraphs = split_into_paragraphs(content)
    print(f"  {len(paragraphs)} paragraphs")

    method = args.method
    print(f"Running method: {method}...")

    if method == "zero_shot":
        result = review_zero_shot(slug, content, model=args.large_model)
    elif method == "rag_local":
        result = review_rag(
            slug, content,
            small_model=args.small_model,
            large_model=args.large_model,
            variant="rag_local",
        )
    elif method in ("incremental", "incremental_full"):
        consolidated, full = review_incremental(
            slug, content,
            model=args.large_model,
            small_model=args.small_model,
        )
        result = full if method == "incremental_full" else consolidated
    else:
        print(f"Error: unknown method: {method}", file=sys.stderr)
        sys.exit(1)

    print(f"  Found {result.num_comments} comments")

    # Build output JSON
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{slug}.json"

    # Build viz-compatible data
    paper_data = _build_paper_json(
        slug, title, content, paragraphs, method, result
    )

    # Merge with existing file if present
    if output_file.exists():
        try:
            existing = json.loads(output_file.read_text())
            existing["methods"][method] = paper_data["methods"][method]
            paper_data = existing
        except (json.JSONDecodeError, KeyError):
            pass

    output_file.write_text(json.dumps(paper_data, indent=2))
    print(f"Results saved to: {output_file}")


def _build_paper_json(
    slug: str,
    title: str,
    content: str,
    paragraphs: list[str],
    method: str,
    result,
) -> dict:
    """Build viz-compatible JSON structure for a paper."""
    para_list = [{"index": i, "text": p} for i, p in enumerate(paragraphs)]

    comments = []
    for i, c in enumerate(result.comments):
        comments.append({
            "id": f"{method}_{i}",
            "title": c.title,
            "quote": c.quote,
            "explanation": c.explanation,
            "comment_type": c.comment_type,
            "paragraph_index": c.paragraph_index,
        })

    method_data = {
        "label": method.replace("_", " ").title(),
        "overall_feedback": result.overall_feedback,
        "comments": comments,
    }

    return {
        "slug": slug,
        "title": title,
        "paragraphs": para_list,
        "methods": {method: method_data},
    }


def cmd_serve(args: argparse.Namespace) -> None:
    """Start the visualization server."""
    from .serve import run_server
    run_server(results_dir=args.results_dir, port=args.port)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="openaireview",
        description="AI-powered academic paper reviewer",
    )
    subparsers = parser.add_subparsers(dest="command")

    # review subcommand
    review_parser = subparsers.add_parser(
        "review", help="Review an academic paper"
    )
    review_parser.add_argument("file", help="Path to the paper file")
    review_parser.add_argument(
        "--method",
        choices=["zero_shot", "rag_local", "incremental", "incremental_full"],
        default="incremental",
        help="Review method (default: incremental)",
    )
    review_parser.add_argument(
        "--large-model", default=DEFAULT_LARGE_MODEL,
        help="Large model for deep analysis",
    )
    review_parser.add_argument(
        "--small-model", default=DEFAULT_SMALL_MODEL,
        help="Small model for filtering/summarization",
    )
    review_parser.add_argument(
        "--output-dir", default="./review_results",
        help="Directory for output JSON files (default: ./review_results)",
    )
    review_parser.add_argument(
        "--name", default=None,
        help="Paper slug name (default: derived from filename)",
    )

    # serve subcommand
    serve_parser = subparsers.add_parser(
        "serve", help="Start visualization server"
    )
    serve_parser.add_argument(
        "--results-dir", default="./review_results",
        help="Directory containing result JSON files (default: ./review_results)",
    )
    serve_parser.add_argument(
        "--port", type=int, default=8080,
        help="Server port (default: 8080)",
    )

    args = parser.parse_args()
    if args.command == "review":
        cmd_review(args)
    elif args.command == "serve":
        cmd_serve(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
