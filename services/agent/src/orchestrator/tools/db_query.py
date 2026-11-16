"""Read-only SQL tool against the seeded `demo` schema in Postgres.

The model is shown the schema's tables and columns, read from
``information_schema`` when the registry is built (``describe``), so its first
query can name a real table. Each query runs with ``search_path`` set to
``demo``, so an unqualified name resolves there and not to the agent's own
tables in ``public``. That is hygiene, not a security boundary: a qualified
``public.plans`` still runs (ADR 0016).
"""

from __future__ import annotations

import logging
from typing import Any

import sqlalchemy as sa
from pydantic import BaseModel, Field
from sqlalchemy import Engine

from orchestrator.db.session import get_engine
from orchestrator.tools.base import ToolContext, ToolExecutionError, ToolSpec

logger = logging.getLogger(__name__)

SCHEMA = "demo"
_ALLOWED_PREFIXES = ("SELECT", "WITH")
STATIC_DESCRIPTION = "Run a read-only SQL SELECT against the demo database schema"

_COLUMNS = sa.text(
    "SELECT table_name, column_name, data_type FROM information_schema.columns"
    " WHERE table_schema = :schema ORDER BY table_name, ordinal_position"
)


def describe(engine: Engine) -> str:
    """The tool's description with the schema's tables and columns, e.g.
    "... Tables: demo.vector_db_stats(name text, github_stars integer, license text)".

    Falls back to the static description when the database can't be read or the
    schema has no tables."""
    try:
        with engine.connect() as connection:
            rows = connection.execute(_COLUMNS, {"schema": SCHEMA}).all()
    except Exception as exc:  # any database error: unreachable, auth, missing catalog access
        logger.warning("db_query: could not read the %s schema, using the static description: %s", SCHEMA, exc)
        return STATIC_DESCRIPTION
    tables: dict[str, list[str]] = {}
    for table, column, data_type in rows:
        tables.setdefault(table, []).append(f"{column} {data_type}")
    if not tables:
        return STATIC_DESCRIPTION
    listed = "; ".join(f"{SCHEMA}.{table}({', '.join(columns)})" for table, columns in tables.items())
    return f"{STATIC_DESCRIPTION}. Unqualified table names resolve to {SCHEMA}. Tables: {listed}"


class DbQueryInput(BaseModel):
    sql: str = Field(min_length=1, description="A single read-only SELECT statement")
    max_rows: int = Field(default=50, ge=1, le=200)


class DbQueryOutput(BaseModel):
    rows: list[dict]
    row_count: int


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def handle(args: DbQueryInput, ctx: ToolContext) -> DbQueryOutput:
    statement = args.sql.strip().rstrip(";").strip()
    if ";" in statement:
        raise ToolExecutionError("Only a single SQL statement is allowed")
    if not statement.upper().startswith(_ALLOWED_PREFIXES):
        raise ToolExecutionError("Only read-only SELECT/WITH queries are allowed")

    engine = get_engine()
    with engine.begin() as connection:
        connection.exec_driver_sql("SET TRANSACTION READ ONLY")
        connection.exec_driver_sql(f"SET LOCAL search_path TO {SCHEMA}")  # hygiene, not a boundary
        result = connection.execute(sa.text(statement))
        rows = [
            {key: _jsonable(value) for key, value in mapping.items()}
            for mapping in result.mappings().fetchmany(args.max_rows)
        ]
    return DbQueryOutput(rows=rows, row_count=len(rows))


SPEC = ToolSpec(
    name="db_query",
    description=STATIC_DESCRIPTION,  # the registry replaces it with describe()'s
    input_schema=DbQueryInput,
    output_schema=DbQueryOutput,
    owners=frozenset({"analysis"}),
    rate_limit=10,
    handler=handle,
)
