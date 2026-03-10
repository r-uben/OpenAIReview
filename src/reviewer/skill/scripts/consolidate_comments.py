#!/usr/bin/env python3
"""Gather all sub-agent comment JSON files from a review workspace.

Usage:
    python3 ~/.claude/commands/openaireview/scripts/consolidate_comments.py <review_dir>

Reads every .json file in <review_dir>/comments/, annotates each issue with
its index and source file, then:
  - Writes <review_dir>/comments/all_comments.json  (full text, indexed)
  - Prints a compact title list to stdout (~one line per issue, never truncated)

The orchestrator reads the title list inline to identify duplicates, then
fetches full text for specific issues on demand:

    python3 -c "
    import json
    items = json.load(open('<review_dir>/comments/all_comments.json'))
    print(items[N]['explanation'])   # read issue N in full
    "
"""

import json
import sys
from pathlib import Path


def main():
    if len(sys.argv) < 2:
        print("Usage: consolidate_comments.py <review_dir>", file=sys.stderr)
        sys.exit(1)

    comments_dir = Path(sys.argv[1]) / "comments"
    if not comments_dir.exists():
        print("No comments found.", file=sys.stderr)
        return

    all_issues = []
    for f in sorted(comments_dir.glob("*.json")):
        if f.name == "all_comments.json":
            continue
        try:
            issues = json.loads(f.read_text())
        except json.JSONDecodeError:
            print(f"WARNING: Could not parse {f.name}", file=sys.stderr)
            continue
        if not isinstance(issues, list):
            issues = [issues]
        for issue in issues:
            issue["_source_file"] = f.name
            issue["_index"] = len(all_issues)
            all_issues.append(issue)

    # Write full-text file for on-demand reads
    out = comments_dir / "all_comments.json"
    out.write_text(json.dumps(all_issues, indent=2))

    # Print compact title list to stdout — always small, never truncated
    print(f"Total: {len(all_issues)} comments from {comments_dir}\n")
    print(f"{'IDX':<4}  {'SOURCE':<35}  TITLE")
    print("-" * 100)
    for it in all_issues:
        src = it["_source_file"].replace(".json", "")
        print(f"[{it['_index']:<3}]  {src:<35}  {it['title'][:60]}")

    print(f"\nFull text written to: {out}")
    print("To read issue N in full:")
    print(f"  python3 -c \"import json; it=json.load(open('{out}'))[N]; print(it['title']); print(it['explanation'])\"")


if __name__ == "__main__":
    main()
