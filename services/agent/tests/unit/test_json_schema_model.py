"""Input models built from MCP tools' JSON Schema keep the server's contract exactly."""

import copy
import re

import pytest
from pydantic import ValidationError

from orchestrator.tools.json_schema_model import model_from_json_schema

MODE = "hybrid: keywords and meaning, reranked; dense: meaning only. Leave it out to use the service's default."

# The input schemas the RAG MCP server (mcp/server.py) publishes, as an MCP client receives them.
SEARCH = {
    "type": "object",
    "properties": {
        "query": {
            "description": "What to find.",
            "maxLength": 2000,
            "minLength": 1,
            "title": "Query",
            "type": "string",
        },
        "top_k": {
            "default": 3,
            "description": "How many passages to use.",
            "maximum": 20,
            "minimum": 1,
            "title": "Top K",
            "type": "integer",
        },
        "mode": {
            "anyOf": [{"enum": ["dense", "hybrid"], "type": "string"}, {"type": "null"}],
            "default": None,
            "description": MODE,
            "title": "Mode",
        },
    },
    "required": ["query"],
    "title": "searchArguments",
}
ASK = {
    "type": "object",
    "properties": {
        "question": {
            "description": "The question.",
            "maxLength": 2000,
            "minLength": 1,
            "title": "Question",
            "type": "string",
        },
        "mode": SEARCH["properties"]["mode"],
        "top_k": {
            # Pydantic puts an Optional[Annotated[...]]'s description inside the non-null branch.
            "anyOf": [
                {"description": "How many passages to use.", "maximum": 20, "minimum": 1, "type": "integer"},
                {"type": "null"},
            ],
            "default": None,
            "title": "Top K",
        },
    },
    "required": ["question"],
    "title": "askArguments",
}
LIST_SOURCES = {"type": "object", "properties": {}, "title": "list_sourcesArguments"}


def test_required_fields_have_no_default_and_optional_ones_keep_theirs():
    Search = model_from_json_schema("rag_search_input", SEARCH)

    args = Search(query="FERRY-429")

    assert (args.query, args.top_k, args.mode) == ("FERRY-429", 3, None)
    # What the adapter sends: the defaults the server declared, and no nulls.
    assert args.model_dump(mode="json", exclude_none=True) == {"query": "FERRY-429", "top_k": 3}
    with pytest.raises(ValidationError, match="query"):
        Search()


def test_an_enum_becomes_a_literal():
    Search = model_from_json_schema("rag_search_input", SEARCH)

    assert Search(query="x", mode="dense").mode == "dense"
    with pytest.raises(ValidationError, match="mode"):
        Search(query="x", mode="sparse")


def test_any_of_with_null_is_optional_and_keeps_the_inner_bounds():
    Ask = model_from_json_schema("rag_ask_input", ASK)

    assert Ask(question="q").top_k is None
    assert Ask(question="q", top_k=None).top_k is None
    assert Ask(question="q", top_k=20).top_k == 20
    with pytest.raises(ValidationError, match="top_k"):
        Ask(question="q", top_k=21)


@pytest.mark.parametrize(
    "arguments",
    [
        {"query": ""},
        {"query": "x" * 2001},
        {"query": "x", "top_k": 0},
        {"query": "x", "top_k": 21},
    ],
    ids=["empty-query", "long-query", "top_k-0", "top_k-21"],
)
def test_bounds_and_lengths_are_enforced(arguments):
    Search = model_from_json_schema("rag_search_input", SEARCH)

    with pytest.raises(ValidationError):
        Search.model_validate(arguments)


def test_exclusive_bounds_and_patterns_are_enforced():
    Model = model_from_json_schema(
        "m",
        {
            "type": "object",
            "properties": {
                "ratio": {"type": "number", "exclusiveMinimum": 0, "exclusiveMaximum": 1},
                "code": {"type": "string", "pattern": "^FERRY-[0-9]+$"},
            },
            "required": ["ratio", "code"],
        },
    )

    assert Model(ratio=0.5, code="FERRY-429").code == "FERRY-429"
    with pytest.raises(ValidationError):
        Model(ratio=1, code="FERRY-429")
    with pytest.raises(ValidationError):
        Model(ratio=0.5, code="ferry")


def test_unknown_arguments_are_rejected():
    Search = model_from_json_schema("rag_search_input", SEARCH)

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        Search.model_validate({"query": "x", "delete_everything": True})
    with pytest.raises(ValidationError):
        model_from_json_schema("rag_list_sources_input", LIST_SOURCES).model_validate({"anything": 1})


def _contract(schema: dict) -> dict:
    """A schema without titles, with branch descriptions lifted to their property."""
    properties = {}
    for name, prop in copy.deepcopy(schema)["properties"].items():
        prop.pop("title", None)
        for branch in prop.get("anyOf", []):
            if "description" in branch:
                prop.setdefault("description", branch.pop("description"))
        properties[name] = prop
    return {"properties": properties, "required": sorted(schema.get("required", []))}


@pytest.mark.parametrize("schema", [SEARCH, ASK, LIST_SOURCES], ids=["search", "ask", "list_sources"])
def test_the_llm_sees_the_servers_contract(schema):
    """model_json_schema() is what tool_definitions_for shows the LLM."""
    generated = model_from_json_schema("tool_input", schema).model_json_schema()

    assert _contract(generated) == _contract(schema)
    assert generated["additionalProperties"] is False


@pytest.mark.parametrize(
    ("properties", "needle"),
    [
        ({"filter": {"$ref": "#/$defs/Filter"}}, "filter"),
        ({"filter": {"type": "object", "properties": {"x": {"type": "string"}}}}, "filter"),
        ({"ids": {"type": "array", "items": {"type": "string"}}}, "ids"),
        ({"mode": {"type": ["string", "null"]}}, "mode"),
        ({"mode": {"const": "dense", "type": "string"}}, "mode"),
        ({"value": {"anyOf": [{"type": "string"}, {"type": "integer"}]}}, "value"),
        ({"top-k": {"type": "integer"}}, "top-k"),
        ({"model_name": {"type": "string"}}, "model_name"),
    ],
    ids=["ref", "nested-object", "array", "type-list", "const", "union", "not-an-identifier", "pydantic-name"],
)
def test_constructs_outside_the_subset_fail_at_discovery_naming_the_property(properties, needle):
    with pytest.raises(ValueError, match=re.escape(f"tool_input.{needle}")):
        model_from_json_schema("tool_input", {"type": "object", "properties": properties})


@pytest.mark.parametrize(
    "schema",
    [
        {"type": "array", "items": {"type": "string"}},
        {"type": "object", "properties": {}, "$defs": {}},
        {"type": "object", "properties": {}, "required": ["query"]},
    ],
    ids=["not-an-object", "defs", "required-undescribed"],
)
def test_malformed_object_schemas_fail_at_discovery(schema):
    with pytest.raises(ValueError, match="tool_input"):
        model_from_json_schema("tool_input", schema)
