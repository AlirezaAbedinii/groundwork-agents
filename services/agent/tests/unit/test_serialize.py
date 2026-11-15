"""render_result: a tool result as valid JSON within its budget, saying what it left out."""

import json
import random
import re

import pytest
from pydantic import BaseModel

from orchestrator.tools.base import ToolSpec
from orchestrator.tools.serialize import MIN_RESULT_BUDGET_CHARS, render_result

MARKER = re.compile(r"…\[\+(\d+) chars\]$")


def says_what_it_left_out(value) -> bool:
    """A truncated count or a shortened string anywhere in ``value``."""
    if isinstance(value, str):
        return MARKER.search(value) is not None
    if isinstance(value, dict):
        count = value.get("truncated")
        if isinstance(count, dict) and count.keys() == {"shown", "of"} and count["shown"] < count["of"]:
            return True
        return any(says_what_it_left_out(item) for item in value.values())
    if isinstance(value, list):
        return any(says_what_it_left_out(item) for item in value)
    return False


# Escapes of every width: quote and backslash (2), newline (2), "é" and "—" (6), an emoji (12).
ALPHABET = "abc xyz " * 6 + '"\\\n' + "é—" + "\U0001f600"
KEYS = ["hits", "rows", "content", "snippet", "answer", "truncated", "score", "ключ"]


def random_text(rng: random.Random) -> str:
    length = rng.choice([0, 3, 40, 300, 1500, 6000])
    return "".join(rng.choice(ALPHABET) for _ in range(rng.randint(0, length)))


def random_value(rng: random.Random, depth: int):
    kinds = ["text", "number", "flag", "none"] + (["list", "dict"] * 2 if depth < 3 else [])
    kind = rng.choice(kinds)
    if kind == "text":
        return random_text(rng)
    if kind == "number":
        return rng.choice([rng.randint(-10**6, 10**6), rng.random()])
    if kind == "flag":
        return rng.random() < 0.5
    if kind == "none":
        return None
    if kind == "list":
        return [random_value(rng, depth + 1) for _ in range(rng.choice([0, 1, 2, 10, 40]))]
    return random_dict(rng, depth + 1)


def random_dict(rng: random.Random, depth: int) -> dict:
    return {rng.choice(KEYS) + str(n % 3 or ""): random_value(rng, depth) for n in range(rng.randint(0, 6))}


def random_output(rng: random.Random) -> dict:
    """Shaped like the registry's outputs: local tools' fields, or an MCP tool's ``{"content": ...}``."""
    if rng.random() < 0.5:
        return {"content": rng.choice([random_dict(rng, 1), random_value(rng, 1), random_text(rng)])}
    return random_dict(rng, 0)


def test_every_result_parses_and_fits_and_a_shortened_one_says_so():
    rng = random.Random(20261115)
    shortened = 0
    for case in range(400):
        output, budget = random_output(rng), rng.randint(200, 5000)

        text = render_result(output, budget)

        assert len(text) <= budget, f"case {case}"
        parsed = json.loads(text)
        if json.dumps(output) == text:
            continue
        shortened += 1
        assert says_what_it_left_out(parsed), f"case {case}: {text[:200]}"
    assert shortened > 100  # the generator does reach the shortening paths


@pytest.mark.parametrize(
    "output",
    [
        {"msg": "x" * 1980},
        {"query": "Ferry — retries", "hits": [{"snippet": "é" * 100, "score": 0.5}] * 2},
        {"content": "plain text"},
        {"rows": [], "row_count": 0},
    ],
)
def test_a_result_that_fits_is_exactly_its_json(output):
    assert render_result(output, 2000) == json.dumps(output)


def test_a_result_exactly_at_the_budget_is_unchanged():
    output = {"msg": "x" * (2000 - len('{"msg": ""}'))}
    assert len(json.dumps(output)) == 2000
    assert render_result(output, 2000) == json.dumps(output)


HIT = {
    "chunk_id": "9d2b7c5e0f1a4836",
    "source_file": "03-configuration.md",
    "section_heading": "Ferry — Configuration Reference",
    "score": 0.8123456789,
    "snippet": ("Ferry retries a job with backoff — see ferry.retry.* → the DLQ. " * 5)[:300],
}


def test_ten_hits_keep_whole_hits_and_say_how_many_are_shown():
    hits = [{**HIT, "chunk_id": f"chunk-{n}"} for n in range(10)]
    output = {"content": {"query": "retry policy", "mode": "hybrid", "hits": hits}}
    assert len(json.dumps(output)) > 4000

    content = json.loads(render_result(output, 4000))["content"]

    shown = content["truncated"]["shown"]
    assert content["truncated"] == {"shown": shown, "of": 10} and 0 < shown < 10
    assert content["hits"] == hits[:shown]  # whole hits, best first
    assert list(content) == ["query", "mode", "hits", "truncated"]  # the count right after the list
    one_more = {**output["content"], "hits": hits[: shown + 1], "truncated": {"shown": shown + 1, "of": 10}}
    assert len(json.dumps({"content": one_more})) > 4000  # as many as fit


def test_rows_at_the_top_level_are_dropped_from_the_end():
    rows = [{"name": f"repo-{n}", "stars": n} for n in range(200)]

    result = json.loads(render_result({"rows": rows, "row_count": 200}, 2000))

    assert result["rows"] == rows[: result["truncated"]["shown"]] and result["truncated"]["of"] == 200
    assert result["row_count"] == 200


def test_a_list_that_is_the_whole_content_gets_its_count_beside_it():
    result = json.loads(render_result({"content": [f"line {n}" for n in range(500)]}, 1000))

    assert result["content"] == [f"line {n}" for n in range(result["truncated"]["shown"])]
    assert result["truncated"]["of"] == 500


def test_a_single_long_string_is_shortened_with_the_marker():
    text = "".join(chr(ord("a") + n % 26) for n in range(20_000))

    rendered = render_result({"content": text}, 2000)

    assert len(rendered) <= 2000
    shortened = json.loads(rendered)["content"]
    kept, marker = shortened.split("…", 1)
    assert text.startswith(kept) and len(kept) > 1900
    assert marker == f"[+{20_000 - len(kept)} chars]"


def test_the_longest_string_is_shortened_first():
    output = {"stdout": "o" * 10_000, "stderr": "Traceback: short", "exit_code": 1}

    result = json.loads(render_result(output, 2000))

    assert MARKER.search(result["stdout"])
    assert (result["stderr"], result["exit_code"]) == ("Traceback: short", 1)


def test_when_no_whole_item_fits_strings_are_shortened_and_every_item_kept():
    citations = [{"index": n, "chunk_id": f"chunk-{n}", "supported": True} for n in range(1, 6)]
    output = {"content": {"answer": "Ferry retries. " * 400, "refused": False, "citations": citations}}

    content = json.loads(render_result(output, 4000))["content"]

    assert content["citations"] == citations and "truncated" not in content
    assert MARKER.search(content["answer"])


def test_a_structure_too_big_to_shorten_becomes_one_shortened_string():
    output = {f"field_{n}": n for n in range(1000)}  # no list, no string: nothing else to cut

    rendered = render_result(output, 300)

    assert len(rendered) <= 300
    preview = json.loads(rendered)
    assert json.dumps(output).startswith(preview.split("…")[0]) and MARKER.search(preview)


def test_a_budget_below_the_minimum_is_refused():
    with pytest.raises(ValueError, match="at least 200"):
        render_result({"msg": "hi"}, MIN_RESULT_BUDGET_CHARS - 1)


class _In(BaseModel):
    msg: str


def _spec(**budget) -> ToolSpec:
    return ToolSpec(name="echo", description="Echo", input_schema=_In, output_schema=_In,
                    owners=frozenset({"research"}), rate_limit=1, handler=lambda args, ctx: args, **budget)


def test_a_tool_spec_has_a_result_budget_of_2000_unless_given_one():
    assert _spec().result_budget_chars == 2000
    assert _spec(result_budget_chars=4000).result_budget_chars == 4000
    with pytest.raises(ValueError, match="echo: result_budget_chars must be at least 200"):
        _spec(result_budget_chars=100)
