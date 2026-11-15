"""Tool results as the model reads them: valid JSON within a character budget.

``render_result`` returns ``json.dumps(output)`` unchanged when it fits. A longer
result is shortened in the first of these ways that fits:

1. Whole items are dropped from the end of the longest list, at the top level or
   under ``content`` (where MCP tools put their results), and
   ``"truncated": {"shown": n, "of": m}`` goes right after it.
2. The longest strings are shortened, longest first, each ending in ``…[+N chars]``.
   This is the way when there is no list, or when not even one whole item fits.
3. Both: the longest list cut to its first item, and strings shortened.
4. The JSON text itself as one shortened JSON string, for a result whose
   structure alone is over the budget.

Every result parses, and a shortened one says what it left out. Sizes are
characters of ``json.dumps`` output with its defaults, which writes each
non-ASCII character as six.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from typing import Any

DEFAULT_RESULT_BUDGET_CHARS = 2000
# Enough for the shortest form any shortened result can take.
MIN_RESULT_BUDGET_CHARS = 200

TRUNCATED_KEY = "truncated"


def render_result(output: dict, budget: int) -> str:
    """``output`` as JSON of at most ``budget`` characters."""
    if budget < MIN_RESULT_BUDGET_CHARS:
        raise ValueError(f"a result budget must be at least {MIN_RESULT_BUDGET_CHARS} chars, not {budget}")
    text = json.dumps(output)
    if len(text) <= budget:
        return text

    path = _longest_list(output)
    if path is not None:
        items = len(_at(output, path))
        if fitted := _largest_fitting(1, items - 1, budget, lambda shown: json.dumps(_trimmed(output, path, shown))):
            return fitted
    if fitted := _shortened(output, budget):
        return fitted
    if path is not None and (fitted := _shortened(_trimmed(output, path, 1), budget)):
        return fitted
    fitted = _largest_fitting(0, len(text), budget, lambda limit: json.dumps(_shorten(text, limit)))
    assert fitted is not None, "MIN_RESULT_BUDGET_CHARS leaves room for the shortest form"
    return fitted


def _longest_list(output: dict) -> tuple[str, ...] | None:
    """Where the non-empty list with the longest JSON is: ``(key,)``, ``("content",)`` or ``("content", key)``."""
    paths = [(key,) for key, value in output.items() if isinstance(value, list)]
    if isinstance(content := output.get("content"), dict):
        paths += [("content", key) for key, value in content.items() if isinstance(value, list)]
    paths = [path for path in paths if path[-1] != TRUNCATED_KEY and _at(output, path)]
    return max(paths, key=lambda path: len(json.dumps(_at(output, path))), default=None)


def _at(output: dict, path: tuple[str, ...]) -> Any:
    value: Any = output
    for key in path:
        value = value[key]
    return value


def _trimmed(output: dict, path: tuple[str, ...], shown: int) -> dict:
    """``output`` with the list at ``path`` cut to its first ``shown`` items, and the count after it."""
    *outer, key = path
    holder = output[outer[0]] if outer else output
    items = holder[key]
    trimmed: dict = {}
    for name, value in holder.items():
        if name == TRUNCATED_KEY:
            continue  # a field of that name gives way to the count
        trimmed[name] = items[:shown] if name == key else value
        if name == key:
            trimmed[TRUNCATED_KEY] = {"shown": shown, "of": len(items)}
    return {**output, outer[0]: trimmed} if outer else trimmed


def _shortened(output: dict, budget: int) -> str | None:
    """``output`` as JSON with each string over the largest limit that fits shortened to it, or None."""
    longest = max(_string_lengths(output), default=0)
    return _largest_fitting(0, longest, budget, lambda limit: json.dumps(_shorten_strings(output, limit)))


def _string_lengths(value: Any) -> Iterator[int]:
    if isinstance(value, str):
        yield len(value)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _string_lengths(item)
    elif isinstance(value, list):
        for item in value:
            yield from _string_lengths(item)


def _shorten_strings(value: Any, limit: int) -> Any:
    if isinstance(value, str):
        return _shorten(value, limit)
    if isinstance(value, dict):
        return {key: _shorten_strings(item, limit) for key, item in value.items()}
    if isinstance(value, list):
        return [_shorten_strings(item, limit) for item in value]
    return value


def _shorten(text: str, limit: int) -> str:
    """``text`` cut to ``limit`` characters plus a marker, when that is shorter as JSON."""
    if len(text) <= limit:
        return text
    cut = f"{text[:limit]}…[+{len(text) - limit} chars]"
    # A cut that saves nothing is not made, so the JSON never gets shorter as the
    # limit grows, which the binary search in _largest_fitting relies on.
    return cut if len(json.dumps(cut)) < len(json.dumps(text)) else text


def _largest_fitting(low: int, high: int, budget: int, render: Callable[[int], str]) -> str | None:
    """``render(n)`` for the largest n in [low, high] whose text fits ``budget``, or None if none does.

    The text must not get shorter as n grows: one more whole item, or one more
    character kept, never shortens a result."""
    best = None
    while low <= high:
        middle = (low + high) // 2
        text = render(middle)
        if len(text) <= budget:
            best, low = text, middle + 1
        else:
            high = middle - 1
    return best
