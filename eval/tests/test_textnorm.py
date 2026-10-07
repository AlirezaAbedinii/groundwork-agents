"""Quote matching: whitespace-insensitive, case-sensitive, any quote covers its item."""

from schemas import EvidenceItem, Quote
from textnorm import covered_items, normalize_ws


def item(*texts: str) -> EvidenceItem:
    return EvidenceItem(quotes=[Quote(source="a.md", text=t) for t in texts])


def test_normalize_ws_collapses_runs_and_strips():
    assert normalize_ws("  Set\n\tUV_CACHE_DIR   to\r\nmove it  ") == "Set UV_CACHE_DIR to move it"
    assert normalize_ws("Keep CASE") == "Keep CASE"
    assert normalize_ws(" \n ") == ""


def test_a_quote_matches_across_a_line_wrap_and_indentation():
    hit = "Set `UV_CACHE_DIR` to move the\n    cache somewhere else."
    assert covered_items(hit, [item("to move the cache somewhere")]) == frozenset({0})


def test_matching_is_case_sensitive():
    hit = "Set `UV_CACHE_DIR` to move the cache."
    assert covered_items(hit, [item("set `uv_cache_dir` to move")]) == frozenset()


def test_any_one_quote_covers_its_item():
    evidence = [item("not in this hit at all", "UV_CACHE_DIR to move")]
    assert covered_items("Set UV_CACHE_DIR to move the cache.", evidence) == frozenset({0})


def test_one_hit_can_cover_several_items():
    hit = "Run `uv lock` to write uv.lock. Run `uv sync` to install from it."
    evidence = [
        item("Run `uv lock` to write"),
        item("not here, nowhere"),
        item("`uv sync` to install"),
    ]
    assert covered_items(hit, evidence) == frozenset({0, 2})


def test_no_evidence_covers_nothing():
    assert covered_items("any text at all", []) == frozenset()
