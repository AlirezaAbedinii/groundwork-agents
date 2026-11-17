import pytest
from pydantic import ValidationError

from orchestrator.planning.schemas import ExecutionPlan, Subtask


def _subtask(sid: str, specialist: str = "research", depends_on: list[str] | None = None) -> dict:
    return {
        "id": sid,
        "description": f"do {sid}",
        "specialist": specialist,
        "depends_on": depends_on or [],
    }


def test_valid_plan_and_waves():
    plan = ExecutionPlan(
        subtasks=[
            _subtask("s1"),
            _subtask("s2", "analysis"),
            _subtask("s3", "writing", depends_on=["s1", "s2"]),
        ],
        confidence=0.8,
    )
    assert plan.topological_waves() == [["s1", "s2"], ["s3"]]
    assert plan.subtask("s3").specialist == "writing"


def test_duplicate_ids_rejected():
    with pytest.raises(ValidationError, match="Duplicate subtask ids"):
        ExecutionPlan(subtasks=[_subtask("s1"), _subtask("s1")], confidence=0.8)


def test_unknown_dependency_rejected():
    with pytest.raises(ValidationError, match="unknown ids"):
        ExecutionPlan(subtasks=[_subtask("s1", depends_on=["nope"])], confidence=0.8)


def test_self_dependency_rejected():
    with pytest.raises(ValidationError, match="depends on itself"):
        ExecutionPlan(subtasks=[_subtask("s1", depends_on=["s1"])], confidence=0.8)


def test_cycle_rejected():
    with pytest.raises(ValidationError, match="cycle"):
        ExecutionPlan(
            subtasks=[_subtask("s1", depends_on=["s2"]), _subtask("s2", depends_on=["s1"])],
            confidence=0.8,
        )


def test_confidence_bounds():
    with pytest.raises(ValidationError):
        ExecutionPlan(subtasks=[_subtask("s1")], confidence=1.2)
    with pytest.raises(ValidationError):
        ExecutionPlan(subtasks=[_subtask("s1")], confidence=-0.1)


def test_unknown_specialist_rejected():
    with pytest.raises(ValidationError):
        Subtask(id="s1", description="x", specialist="astrologer")


def test_at_least_one_subtask():
    with pytest.raises(ValidationError):
        ExecutionPlan(subtasks=[], confidence=0.5)


def test_a_subtask_lists_its_tools_and_defaults_to_none():
    named = Subtask.model_validate({**_subtask("s1"), "tools": ["web_search", "file_write"]})
    assert named.tools == ["web_search", "file_write"]
    assert Subtask.model_validate(_subtask("s2", "writing")).tools == []
    assert "tools" in ExecutionPlan.model_json_schema()["$defs"]["Subtask"]["properties"]


def test_a_plan_lists_no_deliverables_unless_it_says_so():
    plan = ExecutionPlan(subtasks=[_subtask("s1")], confidence=0.8)

    assert plan.deliverables == []
    assert "deliverables" in ExecutionPlan.model_json_schema()["properties"]


def test_a_deliverable_names_its_path_and_producer():
    plan = ExecutionPlan(
        subtasks=[_subtask("s1"), _subtask("s2", "writing", depends_on=["s1"])],
        deliverables=[{"path": "memo.md", "producer": "s2"}],
        confidence=0.8,
    )

    assert [(d.path, d.producer) for d in plan.deliverables] == [("memo.md", "s2")]


def test_a_deliverable_from_an_unknown_subtask_rejected():
    with pytest.raises(ValidationError, match="Deliverable 'memo.md' has unknown producer 's9'"):
        ExecutionPlan(
            subtasks=[_subtask("s1")], deliverables=[{"path": "memo.md", "producer": "s9"}], confidence=0.8
        )


def test_duplicate_deliverable_paths_rejected():
    with pytest.raises(ValidationError, match="Duplicate deliverable paths"):
        ExecutionPlan(
            subtasks=[_subtask("s1")],
            deliverables=[{"path": "memo.md", "producer": "s1"}, {"path": "memo.md", "producer": "s1"}],
            confidence=0.8,
        )
