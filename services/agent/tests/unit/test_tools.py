import pytest

from orchestrator.tools.api_call import ApiCallInput, SPEC as API_SPEC, handle as api_handle
from orchestrator.tools.base import ToolContext, ToolExecutionError
from orchestrator.tools.code_exec import NOTHING_PRINTED, CodeExecInput, handle as code_handle
from orchestrator.tools.db_query import STATIC_DESCRIPTION, DbQueryInput, describe, handle as db_handle
from orchestrator.tools.file_io import (
    FileReadInput,
    FileWriteInput,
    handle_read,
    handle_write,
)
from orchestrator.tools.web_search import WebSearchInput, handle as search_handle


@pytest.fixture()
def ctx(tmp_path):
    return ToolContext(task_id="t1", specialist="analysis", workspace=tmp_path / "ws")


def test_file_write_then_read_roundtrip(ctx):
    written = handle_write(FileWriteInput(path="notes/a.txt", content="hello"), ctx)
    assert written.bytes_written == 5
    read = handle_read(FileReadInput(path="notes/a.txt"), ctx)
    assert read.content == "hello"


def test_file_path_traversal_is_blocked(ctx):
    with pytest.raises(ToolExecutionError, match="escapes the task workspace"):
        handle_write(FileWriteInput(path="../evil.txt", content="x"), ctx)


def test_file_read_missing_file(ctx):
    with pytest.raises(ToolExecutionError, match="does not exist"):
        handle_read(FileReadInput(path="nope.txt"), ctx)


def test_code_exec_subprocess_runs_python(ctx):
    result = code_handle(CodeExecInput(code="print(6 * 7)"), ctx)
    assert result.exit_code == 0
    assert result.stdout.strip() == "42"


def test_code_exec_reports_success_and_says_nothing_when_output_was_printed(ctx):
    result = code_handle(CodeExecInput(code="print(6 * 7)"), ctx)
    assert result.ok is True
    assert result.note is None


def test_code_exec_notes_code_that_ran_and_printed_nothing(ctx):
    result = code_handle(CodeExecInput(code="def answer():\n    return 42\n"), ctx)
    assert (result.exit_code, result.ok, result.stdout, result.stderr) == (0, True, "", "")
    assert result.note == NOTHING_PRINTED


def test_code_exec_reports_a_non_zero_exit_as_not_ok(ctx):
    result = code_handle(CodeExecInput(code="raise ValueError('bad input')"), ctx)
    assert result.exit_code != 0
    assert result.ok is False
    assert "bad input" in result.stderr
    assert result.note is None  # it printed a traceback


def test_code_exec_times_out(ctx):
    with pytest.raises(ToolExecutionError, match="timed out"):
        code_handle(CodeExecInput(code="import time; time.sleep(5)", timeout_s=1), ctx)


def test_code_exec_refuses_when_disabled(ctx, monkeypatch):
    from orchestrator.config import get_settings

    monkeypatch.setattr(get_settings(), "code_exec_backend", "disabled")
    with pytest.raises(ToolExecutionError, match="disabled"):
        code_handle(CodeExecInput(code="print(6 * 7)"), ctx)


def test_db_query_rejects_writes(ctx):
    with pytest.raises(ToolExecutionError, match="read-only"):
        db_handle(DbQueryInput(sql="DELETE FROM demo.vector_db_stats"), ctx)


def test_db_query_rejects_multi_statement(ctx):
    with pytest.raises(ToolExecutionError, match="single SQL statement"):
        db_handle(DbQueryInput(sql="SELECT 1; SELECT 2"), ctx)


def test_api_call_rejects_non_allowlisted_host(ctx):
    with pytest.raises(ToolExecutionError, match="not in the API allowlist"):
        api_handle(ApiCallInput(method="GET", url="https://evil.example.org/x"), ctx)


def test_api_call_mock_response_for_allowlisted_host(ctx):
    result = api_handle(ApiCallInput(method="GET", url="https://api.github.com/repos"), ctx)
    assert result.status_code == 200
    assert "[mock]" in result.body


def test_api_call_post_is_sensitive_get_is_not():
    assert API_SPEC.is_sensitive(ApiCallInput(method="POST", url="https://api.github.com/x"))
    assert not API_SPEC.is_sensitive(ApiCallInput(method="GET", url="https://api.github.com/x"))


def test_web_search_mock_returns_canned_results(ctx):
    output = search_handle(WebSearchInput(query="chroma db"), ctx)
    assert output.results[0].snippet == "[mock] result for chroma db"


class _FakeEngine:
    """Answers describe()'s information_schema query with fixed rows, or raises."""

    def __init__(self, rows=(), error: Exception | None = None):
        self.rows, self.error = list(rows), error

    def connect(self):
        if self.error:
            raise self.error
        engine = self

        class Connection:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def execute(self, statement, params):
                assert params == {"schema": "demo"}
                return type("Result", (), {"all": lambda _: engine.rows})()

        return Connection()


def test_db_query_describe_lists_each_table_with_its_columns_in_order():
    rows = [
        ("licenses", "spdx_id", "text"),
        ("vector_db_stats", "name", "text"),
        ("vector_db_stats", "github_stars", "integer"),
        ("vector_db_stats", "license", "text"),
    ]
    assert describe(_FakeEngine(rows)) == (
        f"{STATIC_DESCRIPTION}. Unqualified table names resolve to demo. Tables: demo.licenses(spdx_id text); "
        "demo.vector_db_stats(name text, github_stars integer, license text)"
    )


def test_db_query_describe_falls_back_on_a_database_error_or_an_empty_schema():
    assert describe(_FakeEngine(error=OSError("connection refused"))) == STATIC_DESCRIPTION
    assert describe(_FakeEngine(rows=[])) == STATIC_DESCRIPTION
