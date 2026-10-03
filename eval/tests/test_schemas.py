"""The golden-set and agent-task rules the schema enforces on its own."""

import dataclasses

import pytest
from pydantic import ValidationError

from schemas import (
    PRF,
    AgentTask,
    CorrectnessVerdict,
    EvidenceItem,
    ExpectedCall,
    FaithfulnessVerdict,
    GoldenQuestion,
    Quote,
    TaskVerdict,
    ToolCall,
)


def item(*texts: str, source: str = "uv/concepts/cache.md") -> dict:
    return {"quotes": [{"source": source, "text": t} for t in texts]}


def row(**overrides) -> dict:
    base = {
        "id": "q001",
        "question": "Where does uv keep its cache?",
        "reference_answer": "In the cache directory, which UV_CACHE_DIR moves.",
        "category": "lookup",
        "evidence": [item("UV_CACHE_DIR moves the cache")],
        "origin": "targeted",
        "split": "dev",
        "verified": True,
    }
    return base | overrides


def test_a_lookup_row_parses():
    q = GoldenQuestion.model_validate(row())
    assert q.evidence[0].quotes[0].source == "uv/concepts/cache.md"
    assert q.notes == ""


@pytest.mark.parametrize(
    ("category", "n_items", "ok"),
    [
        ("no_answer", 0, True),
        ("no_answer", 1, False),
        ("lookup", 0, False),
        ("lookup", 1, True),
        ("lookup", 2, True),
        ("multi_hop", 1, False),
        ("multi_hop", 2, True),
        ("ambiguous", 1, False),
        ("ambiguous", 2, True),
    ],
)
def test_evidence_count_fits_the_category(category, n_items, ok):
    evidence = [item(f"evidence item number {i}") for i in range(n_items)]
    data = row(category=category, evidence=evidence)
    if ok:
        GoldenQuestion.model_validate(data)
    else:
        with pytest.raises(ValidationError, match=category):
            GoldenQuestion.model_validate(data)


@pytest.mark.parametrize(("length", "ok"), [(11, False), (12, True), (120, True), (121, False)])
def test_quote_length_bounds(length, ok):
    data = {"source": "a.md", "text": "x" * length}
    if ok:
        Quote.model_validate(data)
    else:
        with pytest.raises(ValidationError):
            Quote.model_validate(data)


def test_whitespace_doesnt_count_toward_the_minimum():
    with pytest.raises(ValidationError, match="besides whitespace"):
        Quote(source="a.md", text="   too  short   ")


@pytest.mark.parametrize("source", ["/abs/a.md", "../a.md", "uv/../a.md", "uv\\a.md", ""])
def test_quote_source_must_be_relative_posix(source):
    with pytest.raises(ValidationError, match="relative to the corpus root"):
        Quote(source=source, text="a long enough quote")


def test_an_evidence_item_needs_a_quote():
    with pytest.raises(ValidationError):
        EvidenceItem(quotes=[])


@pytest.mark.parametrize("rid", ["q1", "q0001", "x001", "Q001"])
def test_id_pattern(rid):
    with pytest.raises(ValidationError):
        GoldenQuestion.model_validate(row(id=rid))


@pytest.mark.parametrize(("rid", "origin"), [("s001", "targeted"), ("q001", "synthetic")])
def test_id_prefix_matches_origin(rid, origin):
    with pytest.raises(ValidationError, match="ids start with"):
        GoldenQuestion.model_validate(row(id=rid, origin=origin))


def test_a_synthetic_row_parses():
    q = GoldenQuestion.model_validate(row(id="s001", origin="synthetic"))
    assert q.origin == "synthetic"


def test_split_is_required_and_closed():
    data = row()
    del data["split"]
    with pytest.raises(ValidationError, match="split"):
        GoldenQuestion.model_validate(data)
    with pytest.raises(ValidationError):
        GoldenQuestion.model_validate(row(split="train"))


def test_a_misspelled_key_is_an_error():
    with pytest.raises(ValidationError, match="note"):
        GoldenQuestion.model_validate(row(note="typo for notes"))


@pytest.mark.parametrize("field", ["question", "reference_answer"])
def test_text_fields_cant_be_empty(field):
    with pytest.raises(ValidationError):
        GoldenQuestion.model_validate(row(**{field: ""}))


@pytest.mark.parametrize("rating", [0, 6])
def test_correctness_rating_is_1_to_5(rating):
    with pytest.raises(ValidationError):
        CorrectnessVerdict(reasoning="r", rating=rating)


def test_verdicts_forbid_extra_keys_for_structured_output():
    for model in (CorrectnessVerdict, FaithfulnessVerdict, TaskVerdict):
        assert model.model_json_schema()["additionalProperties"] is False


def test_expected_call_args_keep_their_json_types():
    call = ExpectedCall.model_validate(
        {"tool": "db_query", "args": {"a": "3", "b": 3, "c": 3.5, "d": True}}
    )
    assert call.args == {"a": "3", "b": 3, "c": 3.5, "d": True}
    assert [type(v) for v in call.args.values()] == [str, int, float, bool]


def test_agent_task_defaults():
    task = AgentTask.model_validate(
        {
            "id": "t01",
            "request": "Summarize the cache docs into notes.md",
            "category": "docs",
            "must_call": [{"tool": "rag_search"}],
            "rubric": [{"id": "c1", "description": "Names UV_CACHE_DIR"}],
        }
    )
    assert task.may_call == [] and task.forbidden == [] and task.notes == ""
    assert task.must_call[0].args == {}
    assert task.rubric[0].critical is True


def test_tool_call_keeps_the_log_fields_it_needs():
    logged = {
        "created_at": "2026-10-02T10:00:00Z",
        "subtask_sid": "s1",
        "specialist": "researcher",
        "tool_name": "rag_search",
        "status": "rate_limited",
        "arguments": {"query": "cache"},
        "output": None,
        "error": "429",
        "latency_ms": 12,
        "sensitive": False,
    }
    call = ToolCall.model_validate(logged)
    assert call.model_dump() == {
        "tool_name": "rag_search",
        "specialist": "researcher",
        "arguments": {"query": "cache"},
        "status": "rate_limited",
    }
    with pytest.raises(ValidationError):
        ToolCall.model_validate(logged | {"status": "timeout"})


def test_metric_results_are_frozen():
    prf = PRF(precision=None, recall=0.0, f1=None, tp=0, fp=0, fn=3, tn=5)
    with pytest.raises(dataclasses.FrozenInstanceError):
        prf.tp = 1  # type: ignore[misc]
