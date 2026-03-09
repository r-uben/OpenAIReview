"""Unit tests for utility functions (no API calls)."""

import json

from reviewer.models import Comment
from reviewer.utils import (
    chunk_text,
    count_tokens,
    locate_comment_in_document,
    parse_comments_from_list,
    parse_review_response,
    split_into_paragraphs,
)


def test_count_tokens_nonempty():
    assert count_tokens("hello world") > 0


def test_count_tokens_empty():
    assert count_tokens("") == 0


def test_chunk_text_short():
    text = "Short text."
    chunks = chunk_text(text, max_tokens=1000, overlap_tokens=100)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_chunk_text_splits_long():
    text = "word " * 5000
    chunks = chunk_text(text, max_tokens=500, overlap_tokens=50)
    assert len(chunks) > 1
    # Each chunk should be within limits (roughly)
    for c in chunks:
        assert count_tokens(c) <= 510  # small tolerance


def test_split_into_paragraphs_basic():
    text = "First paragraph.\n\nSecond paragraph that is long enough to stand on its own easily."
    paras = split_into_paragraphs(text, min_chars=10)
    assert len(paras) == 2


def test_split_into_paragraphs_merges_short():
    text = "Hi.\n\nThis is a much longer paragraph that should absorb the short one above."
    paras = split_into_paragraphs(text, min_chars=100)
    assert len(paras) == 1


def test_locate_comment_exact_match():
    paragraphs = [
        "The cat sat on the mat and looked around the room with great curiosity.",
        "The dog chased the ball across the wide green field on a sunny afternoon.",
    ]
    idx = locate_comment_in_document("dog chased the ball", paragraphs)
    assert idx == 1


def test_locate_comment_no_match():
    paragraphs = ["AAAA BBBB CCCC DDDD EEEE FFFF GGGG HHHH IIII JJJJ KKKK LLLL MMMM NNNN"]
    idx = locate_comment_in_document("xxxx yyyy zzzz wwww 1234 5678 9012 3456", paragraphs)
    assert idx is None


def test_parse_comments_from_list():
    items = [
        {
            "title": "Wrong sign",
            "quote": "x = -y",
            "explanation": "Should be positive.",
            "type": "technical",
        },
        {
            "title": "Overclaim",
            "quote": "we prove",
            "explanation": "Not actually proven.",
            "type": "logical",
        },
    ]
    comments = parse_comments_from_list(items)
    assert len(comments) == 2
    assert comments[0].title == "Wrong sign"
    assert comments[0].comment_type == "technical"
    assert comments[1].comment_type == "logical"


def test_parse_comments_from_list_infers_type():
    # When type is unrecognized, it infers from keywords in title+explanation
    items = [{"title": "Wrong formula", "quote": "x", "explanation": "bad", "type": "other"}]
    comments = parse_comments_from_list(items)
    assert comments[0].comment_type == "technical"


def test_parse_review_response_json_object():
    response = json.dumps({
        "overall_feedback": "Good paper.",
        "comments": [
            {"title": "Issue", "quote": "text", "explanation": "problem", "type": "technical"}
        ],
    })
    feedback, comments = parse_review_response(response)
    assert feedback == "Good paper."
    assert len(comments) == 1


def test_parse_review_response_json_array():
    response = json.dumps([
        {"title": "Issue", "quote": "text", "explanation": "problem", "type": "logical"}
    ])
    feedback, comments = parse_review_response(response)
    assert feedback == ""
    assert len(comments) == 1


def test_parse_review_response_markdown_fenced():
    response = '```json\n[{"title": "Bug", "quote": "x", "explanation": "y", "type": "technical"}]\n```'
    _, comments = parse_review_response(response)
    assert len(comments) == 1


def test_parse_review_response_empty():
    feedback, comments = parse_review_response("No JSON here at all.")
    assert comments == []
