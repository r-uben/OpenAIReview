"""Utilities for parsing reviewer output and chunking text."""

import json
import re
from difflib import SequenceMatcher

import tiktoken

from .models import Comment


def count_tokens(text: str, model: str = "gpt-4o") -> int:
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


def chunk_text(text: str, max_tokens: int = 6000, overlap_tokens: int = 200) -> list[str]:
    """Split text into chunks of at most max_tokens with overlap."""
    try:
        enc = tiktoken.get_encoding("cl100k_base")
    except Exception:
        chars_per_chunk = max_tokens * 4
        overlap_chars = overlap_tokens * 4
        chunks = []
        i = 0
        while i < len(text):
            chunks.append(text[i : i + chars_per_chunk])
            i += chars_per_chunk - overlap_chars
        return chunks

    tokens = enc.encode(text)
    chunks = []
    i = 0
    while i < len(tokens):
        chunk_tokens = tokens[i : i + max_tokens]
        chunks.append(enc.decode(chunk_tokens))
        i += max_tokens - overlap_tokens
    return chunks


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


def locate_comment_in_document(
    quote: str,
    paragraphs: list[str],
    threshold: float = 0.3,
) -> int | None:
    """Find the paragraph index that best matches a quote.

    Uses fuzzy substring matching. Returns the index of the best-matching
    paragraph, or None if no paragraph scores above *threshold*.
    """
    if not quote or not paragraphs:
        return None

    quote_lower = quote.lower().strip()[:500]
    best_idx = None
    best_score = 0.0

    for i, para in enumerate(paragraphs):
        para_lower = para.lower().strip()
        # Fast exact-substring check first
        if quote_lower[:80] in para_lower:
            return i
        score = SequenceMatcher(None, quote_lower, para_lower[:600]).ratio()
        if score > best_score:
            best_score = score
            best_idx = i

    return best_idx if best_score >= threshold else None


def assign_paragraph_indices(
    comments: list[Comment],
    document_content: str,
) -> None:
    """Set paragraph_index on each comment by locating its quote in the document."""
    paragraphs = split_into_paragraphs(document_content)
    for comment in comments:
        comment.paragraph_index = locate_comment_in_document(
            comment.quote, paragraphs
        )


def parse_comments_from_list(items: list[dict]) -> list[Comment]:
    """Convert a list of raw dicts into Comment objects."""
    comments = []
    for item in items:
        if not isinstance(item, dict):
            continue
        title = item.get("title", item.get("name", "Untitled"))
        quote = item.get("quote", item.get("flagged_text", item.get("text", "")))
        explanation = item.get("explanation", item.get("message", item.get("comment", "")))
        comment_type = item.get("type", item.get("comment_type", "logical")).lower()
        if comment_type not in ("technical", "logical"):
            comment_type = "technical" if any(
                kw in (title + explanation).lower()
                for kw in ["formula", "equation", "math", "proof", "calculation",
                           "theorem", "incorrect", "wrong", "error", "sign", "factor",
                           "variance", "derivation", "typo", "parameter"]
            ) else "logical"
        paragraph_index = item.get("paragraph_index", None)
        if paragraph_index is not None:
            paragraph_index = int(paragraph_index)
        comments.append(Comment(
            title=title,
            quote=quote,
            explanation=explanation,
            comment_type=comment_type,
            paragraph_index=paragraph_index,
        ))
    return comments


def parse_review_response(response: str) -> tuple[str, list[Comment]]:
    """Parse LLM response returning (overall_feedback, comments).

    Handles two formats:
    - {"overall_feedback": "...", "comments": [...]}  (preferred)
    - [...]  (bare array fallback)
    """
    # Try to extract first valid JSON object or array using raw_decode
    text = response.strip()
    # Strip markdown code fences if present
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    decoder = json.JSONDecoder()
    # Scan for the first '[' or '{' and try to parse from there
    for i, ch in enumerate(text):
        if ch in ("{", "["):
            try:
                obj, _ = decoder.raw_decode(text, i)
                break
            except json.JSONDecodeError:
                continue
    else:
        # Nothing parseable found
        return "", []

    if isinstance(obj, dict):
        overall_feedback = obj.get("overall_feedback", "")
        items = obj.get("comments", [])
        return overall_feedback, parse_comments_from_list(items)
    elif isinstance(obj, list):
        return "", parse_comments_from_list(obj)
    return "", []


def parse_comments_from_response(response: str) -> list[Comment]:
    """Parse a JSON array of comments from an LLM response (legacy wrapper)."""
    _, comments = parse_review_response(response)
    return comments
