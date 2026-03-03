"""Parse refine.ink example HTML files into a benchmark JSONL file."""

import json
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

RAW_HTML_DIR = Path(__file__).parent.parent / "data" / "raw_html"
OUTPUT_FILE = Path(__file__).parent.parent / "data" / "benchmark.jsonl"

PAPER_SLUGS = [
    "inference-molecular",
    "coset-codes",
    "targeting-interventions-networks",
    "chaotic-balanced-state",
]


def extract_js_string(text: str, start: int) -> tuple[str, int]:
    """Extract a JS string value starting at the opening quote after start."""
    i = text.index('"', start)
    i += 1  # skip opening quote
    result = []
    while i < len(text):
        ch = text[i]
        if ch == '"':
            return "".join(result), i + 1
        if ch == "\\" and i + 1 < len(text):
            nxt = text[i + 1]
            escapes = {"n": "\n", "t": "\t", "r": "\r", '"': '"', "\\": "\\", "/": "/"}
            if nxt in escapes:
                result.append(escapes[nxt])
                i += 2
            elif nxt == "x" and i + 3 < len(text):
                # \xNN hex escape
                hex_str = text[i + 2 : i + 4]
                try:
                    result.append(chr(int(hex_str, 16)))
                except ValueError:
                    result.append(nxt)
                i += 4
            elif nxt == "u" and i + 5 < len(text):
                # \uNNNN unicode escape
                hex_str = text[i + 2 : i + 6]
                try:
                    result.append(chr(int(hex_str, 16)))
                except ValueError:
                    result.append(nxt)
                i += 6
            else:
                result.append(nxt)
                i += 2
        else:
            result.append(ch)
            i += 1
    raise ValueError("Unterminated JS string")


def extract_field(big: str, field: str, from_idx: int) -> tuple[str, int]:
    """Extract a string field value from JS object starting near from_idx."""
    pattern = field + ':"'
    idx = big.index(pattern, from_idx)
    val, end = extract_js_string(big, idx + len(pattern) - 1)
    return val, end


def parse_comments(big: str) -> list[dict]:
    """Extract all comment objects from the JS bundle."""
    comments = []
    # Find each comment by id field
    for m in re.finditer(r'id:"(comment[_-]\w+)"', big):
        start = m.start()
        try:
            comment_id = m.group(1)
            title, _ = extract_field(big, "title", start)
            paragraph, _ = extract_field(big, "paragraph", start)
            quote, _ = extract_field(big, "quote", start)
            message, _ = extract_field(big, "message", start)
            # score may be a number or string
            score_m = re.search(r'score:"?([\d.]+)"?', big[start : start + 3000])
            score = float(score_m.group(1)) if score_m else None

            comments.append(
                {
                    "id": comment_id,
                    "title": title,
                    "paragraph": paragraph,
                    "quote": quote,
                    "message": message,
                    "score": score,
                    "comment_type": classify_comment(title, message),
                }
            )
        except (ValueError, AttributeError) as e:
            print(f"  Warning: could not parse comment near {start}: {e}", file=sys.stderr)
    return comments


def classify_comment(title: str, message: str) -> str:
    """Classify a mistake as 'technical' (math) or 'logical' (reasoning/clarity)."""
    text = (title + " " + message).lower()
    technical_keywords = [
        "formula", "equation", "incorrect", "error", "typo", "sign", "factor",
        "variance", "proof", "theorem", "lemma", "derivation", "calculation",
        "incorrect formula", "math", "matrix", "probability", "integral",
        "normalization", "index", "exponent", "coefficient", "prefactor",
        "parameter mismatch", "incorrect example", "wrong", "missing",
        "standard deviation", "inconsistent definition",
    ]
    for kw in technical_keywords:
        if kw in text:
            return "technical"
    return "logical"


def split_into_paragraphs(text: str, min_chars: int = 100) -> list[str]:
    """Split document into paragraphs, merging short ones with the next."""
    raw = [p.strip() for p in text.split("\n\n") if p.strip()]
    paragraphs: list[str] = []
    carry = ""
    for p in raw:
        if carry:
            p = carry + "\n\n" + p
            carry = ""
        if len(p) < min_chars:
            carry = p
        else:
            paragraphs.append(p)
    if carry:
        if paragraphs:
            paragraphs[-1] = paragraphs[-1] + "\n\n" + carry
        else:
            paragraphs.append(carry)
    return paragraphs


def locate_paragraph_index(paragraph_text: str, paragraphs: list[str]) -> int | None:
    """Find the best-matching paragraph index for a comment's paragraph text."""
    if not paragraph_text:
        return None
    target = paragraph_text.lower().strip()
    # Fast exact-substring check
    for i, p in enumerate(paragraphs):
        if target[:80] in p.lower():
            return i
    # Fallback to fuzzy matching
    best_idx, best_score = None, 0.0
    for i, p in enumerate(paragraphs):
        score = SequenceMatcher(None, target[:500], p.lower()[:600]).ratio()
        if score > best_score:
            best_score = score
            best_idx = i
    return best_idx if best_score >= 0.3 else None


def assign_paragraph_indices(comments: list[dict], doc_content: str) -> None:
    """Set paragraph_index on each comment dict."""
    paragraphs = split_into_paragraphs(doc_content)
    for comment in comments:
        comment["paragraph_index"] = locate_paragraph_index(
            comment.get("paragraph", ""), paragraphs
        )


def parse_paper(slug: str) -> dict:
    """Parse a single paper HTML file."""
    html_path = RAW_HTML_DIR / f"{slug}.html"
    print(f"Parsing {slug}...", file=sys.stderr)

    with open(html_path) as f:
        content = f.read()

    scripts = re.findall(r"<script[^>]*>(.*?)</script>", content, re.DOTALL)
    big = max(scripts, key=len)

    # Extract paper metadata
    title, _ = extract_field(big, "title", 0)
    authors, _ = extract_field(big, "authors", 0)

    # Extract document content (the full paper text as markdown)
    doc_content, _ = extract_field(big, "documentContent", 0)

    # Extract badge/field
    try:
        badge, _ = extract_field(big, "badge", 0)
    except ValueError:
        badge = ""

    # Extract overall AI feedback
    overall_feedback = ""
    try:
        overall_feedback, _ = extract_field(big, "overallFeedback", 0)
    except ValueError:
        pass

    comments = parse_comments(big)
    assign_paragraph_indices(comments, doc_content)
    print(f"  Found {len(comments)} comments", file=sys.stderr)

    return {
        "slug": slug,
        "title": title,
        "authors": authors,
        "field": badge,
        "document_content": doc_content,
        "overall_feedback": overall_feedback,
        "comments": comments,
        "num_comments": len(comments),
    }


def main():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w") as out:
        for slug in PAPER_SLUGS:
            paper = parse_paper(slug)
            out.write(json.dumps(paper) + "\n")

    print(f"\nWrote {len(PAPER_SLUGS)} papers to {OUTPUT_FILE}")

    # Print summary
    with open(OUTPUT_FILE) as f:
        for line in f:
            p = json.loads(line)
            technical = sum(1 for c in p["comments"] if c["comment_type"] == "technical")
            logical = sum(1 for c in p["comments"] if c["comment_type"] == "logical")
            print(f"  {p['title'][:50]}: {p['num_comments']} comments ({technical} technical, {logical} logical)")


if __name__ == "__main__":
    main()
