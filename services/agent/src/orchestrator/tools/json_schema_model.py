"""Pydantic input models from JSON Schema, for tools discovered over MCP.

An MCP server describes each tool's arguments as JSON Schema. The registry needs
a Pydantic model instead: it validates arguments with it, and
``tool_definitions_for`` shows the LLM the model's ``model_json_schema()``. This
builds that model for the subset of JSON Schema that MCPServer generates from
Python type hints, so the LLM and the registry both see the server's contract.
Anything outside the subset raises ValueError at discovery instead of being
loosened into arguments the server would reject.

Supported: an object of string/integer/number/boolean properties, ``enum``,
``anyOf`` of one type and null, ``default``, ``required``, numeric bounds,
string lengths and ``pattern``, plus ``title``/``description``. Unknown
arguments are rejected (``extra="forbid"``).
"""

from __future__ import annotations

import keyword
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, create_model

_TYPES: dict[str, type] = {"string": str, "integer": int, "number": float, "boolean": bool}
# JSON Schema keyword -> the Field() argument that produces it.
_CONSTRAINTS = {
    "minimum": "ge",
    "maximum": "le",
    "exclusiveMinimum": "gt",
    "exclusiveMaximum": "lt",
    "minLength": "min_length",
    "maxLength": "max_length",
    "pattern": "pattern",
}
_METADATA = {"title", "description"}
_OBJECT_KEYS = {"type", "properties", "required"} | _METADATA
_NULL = {"type": "null"}


def model_from_json_schema(name: str, schema: dict[str, Any]) -> type[BaseModel]:
    """A model named ``name`` with the schema's fields, constraints, defaults and descriptions."""
    if schema.get("type") != "object":
        raise ValueError(f"{name}: expected an object schema, got type {schema.get('type')!r}")
    if unsupported := set(schema) - _OBJECT_KEYS:
        raise ValueError(f"{name}: unsupported JSON Schema keywords {sorted(unsupported)}")
    properties: dict[str, Any] = schema.get("properties", {})
    required = set(schema.get("required", []))
    if missing := required - set(properties):
        raise ValueError(f"{name}: required but not described: {sorted(missing)}")

    fields: dict[str, Any] = {}
    for prop, prop_schema in properties.items():
        where = f"{name}.{prop}"
        if not prop.isidentifier() or keyword.iskeyword(prop) or prop.startswith("model_") or hasattr(BaseModel, prop):
            raise ValueError(f"{where}: can't be a model field name")
        annotation, description = _annotation(where, prop_schema)
        # Required: no default. Optional: the schema's default, or None when it has none.
        default = ... if prop in required else prop_schema.get("default")
        fields[prop] = (annotation, Field(default, description=description))
    return create_model(name, __config__=ConfigDict(extra="forbid"), **fields)


def _annotation(where: str, schema: dict[str, Any]) -> tuple[Any, str | None]:
    """The Python type for one property, and its description."""
    if "anyOf" not in schema:
        return _scalar(where, schema, outer=True), schema.get("description")
    variants = schema["anyOf"]
    types = [variant for variant in variants if variant != _NULL]
    if set(schema) - {"anyOf", "default"} - _METADATA or len(variants) != 2 or len(types) != 1:
        raise ValueError(f"{where}: anyOf is supported only as [<one type>, null]")
    # Pydantic writes the description of an Optional[Annotated[...]] inside the non-null branch.
    description = schema.get("description") or types[0].get("description")
    return _scalar(where, types[0], outer=False) | None, description


def _scalar(where: str, schema: dict[str, Any], *, outer: bool) -> Any:
    """str/int/float/bool with its constraints, or a Literal for an enum."""
    allowed = {"type", "enum"} | set(_CONSTRAINTS) | _METADATA | ({"default"} if outer else set())
    if unsupported := set(schema) - allowed:
        raise ValueError(f"{where}: unsupported JSON Schema keywords {sorted(unsupported)}")
    type_ = schema.get("type")
    base = _TYPES.get(type_) if isinstance(type_, str) else None  # not a list like ["string", "null"]
    if base is None:
        raise ValueError(f"{where}: unsupported type {schema.get('type')!r}")
    constraints = {_CONSTRAINTS[key]: value for key, value in schema.items() if key in _CONSTRAINTS}
    if "enum" in schema:
        values = schema["enum"]
        if not values or constraints or not all(isinstance(value, base) for value in values):
            raise ValueError(f"{where}: enum must be non-empty {schema['type']} values, without other constraints")
        return Literal[tuple(values)]
    return Annotated[base, Field(**constraints)] if constraints else base
