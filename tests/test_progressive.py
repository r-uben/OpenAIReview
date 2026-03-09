"""Unit tests for progressive method helpers (no API calls)."""

from reviewer.method_progressive import merge_into_passages, split_into_paragraphs


def test_split_and_merge_roundtrip():
    """Paragraphs fed into merge should cover all text."""
    text = "\n\n".join(f"Paragraph {i} with enough text to pass the minimum character threshold easily." for i in range(20))
    paragraphs = split_into_paragraphs(text, min_chars=50)
    passages = merge_into_passages(paragraphs, target_chars=500)
    # Every paragraph index should appear in exactly one passage
    all_indices = []
    for indices, _ in passages:
        all_indices.extend(indices)
    assert sorted(all_indices) == list(range(len(paragraphs)))


def test_merge_respects_target():
    paragraphs = ["x" * 200 for _ in range(10)]
    passages = merge_into_passages(paragraphs, target_chars=500)
    # Should produce multiple passages
    assert len(passages) > 1
    # No passage text (excluding separators) should massively exceed target
    for _, text in passages:
        assert len(text) < 1000
