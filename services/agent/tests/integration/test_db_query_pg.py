"""db_query against the seeded Postgres: the schema text the model is shown, and what
`search_path` does and does not confine (ADR 0016)."""

import pytest

from orchestrator.db.session import get_engine
from orchestrator.tools.base import InMemoryInvocationStore, ToolContext, ToolExecutionError
from orchestrator.tools.db_query import describe
from orchestrator.tools.defaults import build_default_registry

SEEDED = "demo.vector_db_stats(name text, github_stars integer, license text)"


@pytest.fixture()
def registry():
    return build_default_registry(InMemoryInvocationStore())


def _query(registry, sql: str) -> dict:
    return registry.invoke("db_query", {"sql": sql}, ToolContext(task_id="t1", specialist="analysis"))


def test_the_description_names_the_seeded_table_and_its_columns(registry):
    assert SEEDED in describe(get_engine())
    assert SEEDED in registry.get("db_query").description  # what the analysis specialist is shown


def test_an_unqualified_name_resolves_to_the_demo_schema(registry):
    result = _query(registry, "SELECT name FROM vector_db_stats ORDER BY name")
    assert [row["name"] for row in result["rows"]] == ["chroma", "qdrant", "weaviate"]


def test_an_unqualified_agent_table_is_not_found(registry):
    with pytest.raises(ToolExecutionError, match="UndefinedTable|does not exist"):
        _query(registry, "SELECT * FROM plans")


def test_a_qualified_agent_table_still_runs_search_path_is_not_a_boundary(registry):
    result = _query(registry, "SELECT count(*) AS n FROM public.plans")
    assert result["row_count"] == 1


def test_the_search_path_lasts_only_for_the_query(registry):
    _query(registry, "SELECT 1 AS one")
    with get_engine().connect() as connection:
        assert "public" in connection.exec_driver_sql("SHOW search_path").scalar()
