"""A live sanity check of the judge prompts: 10 hand-made cases with known verdicts.

Paid (about $0.02 on claude-haiku-4-5, capped at $0.10) and deselected by default:

    .venv/bin/python -m pytest -m live tests/live/test_judge_sanity.py -s

The cases target the ways a judge goes wrong: rewarding length or confidence over
content, missing a partial answer, accepting a refusal of an answerable question, and
counting outside knowledge or a contradiction as support. The judge passes at 9 of 10:
one miss can be noise, two point at a prompt to fix. Verdicts are cached like any other
judge call, so re-running unchanged prompts is free.

Questions and reference answers come from the golden set; the contexts quote the pinned
uv and FastAPI docs.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import pytest

import metrics
from budget import Budget
from judge import DEFAULT_PROVIDER, Judge
from llm import KEY_ENV, load_env
from schemas import Context

pytestmark = pytest.mark.live

MAX_COST_USD = 0.10
REFUSAL = (
    "I don't know based on the provided documentation. "
    "The retrieved context did not contain enough information to answer this question."
)


def ctx(source: str, heading: str, text: str) -> Context:
    return Context(
        chunk_id=source,
        text=text,
        score=0.9,
        metadata={"source_file": source, "section_heading": heading},
    )


CACHE_DIR_CONTEXT = ctx(
    "uv/concepts/cache.md",
    "Cache directory",
    "uv determines the cache directory according to, in order:\n\n"
    "1. A temporary cache directory, if `--no-cache` was requested.\n"
    "2. The specific cache directory specified via `--cache-dir`, `UV_CACHE_DIR`, or\n"
    "   [`tool.uv.cache-dir`](../reference/settings.md#cache-dir).\n"
    "3. A system-appropriate cache directory, e.g., `$XDG_CACHE_HOME/uv` or `$HOME/.cache/uv` "
    "on Unix and\n   `%LOCALAPPDATA%\\uv\\cache` on Windows\n\n"
    "!!! note\n\n"
    "    uv _always_ requires a cache directory. When `--no-cache` is requested, uv will still "
    "use\n    a temporary cache for sharing data within that single invocation.",
)
FASTAPI_RUN_CONTEXT = ctx(
    "fastapi/fastapi-cli.md",
    "`fastapi run`",
    "Executing `fastapi run` starts FastAPI in production mode.\n\n"
    "By default, **auto-reload** is disabled. It also listens on the IP address `0.0.0.0`, "
    "which means all the available IP addresses, this way it will be publicly accessible to "
    "anyone that can communicate with the machine. This is how you would normally run it in "
    "production, for example, in a container.",
)
CORS_CONTEXT = ctx(
    "fastapi/tutorial/cors.md",
    "Use `CORSMiddleware`",
    "* `allow_credentials` - Indicate that cookies should be supported for cross-origin "
    "requests. Defaults to `False`.\n\n"
    "    None of `allow_origins`, `allow_methods` and `allow_headers` can be set to `['*']` "
    "if `allow_credentials` is set to `True`. All of them must be explicitly specified.",
)


@dataclass(frozen=True)
class Correctness:
    name: str
    question: str
    reference: str
    answer: str
    passes: bool  # expected metrics.correctness_pass


@dataclass(frozen=True)
class Faithfulness:
    name: str
    answer: str
    faithful: bool  # expected: no unsupported claim (score 1.0, or None with no claims)
    contexts: list[Context] = field(default_factory=list)


CORS_Q = (
    "Can I use `allow_origins=['*']` together with `allow_credentials=True` in FastAPI's "
    "CORS middleware?"
)
CORS_REF = (
    "No. If `allow_credentials` is `True`, none of `allow_origins`, `allow_methods` and "
    "`allow_headers` can be set to `['*']`."
)

CASES: list[Correctness | Faithfulness] = [
    Correctness(
        "terse but right",
        CORS_Q,
        CORS_REF,
        "No: with credentials allowed, origins, methods and headers can't be `['*']`.",
        passes=True,
    ),
    Correctness(
        "verbose but wrong",
        "If I pass `--no-cache` to uv, does it run without any cache directory at all?",
        "No. When `--no-cache` is requested, uv still uses a temporary cache to share data "
        "within that single invocation.",
        "Yes. Passing `--no-cache` tells uv to bypass caching entirely: it neither creates nor "
        "reads a cache directory, so every package is downloaded and built from scratch for "
        "that invocation. This is handy in CI, where you want clean, reproducible installs, and "
        "in Docker builds, where a cache would only bloat the image layers. The trade-off is "
        "speed: without any cache, repeated installs redo all the work, so for day-to-day "
        "development you should keep the cache and run `uv cache clean` when you need a fresh "
        "start. In short, `--no-cache` means uv runs with no cache directory whatsoever.",
        passes=False,
    ),
    Correctness(
        "right, with extra correct detail",
        "How do `fastapi dev` and `fastapi run` differ in auto-reload and in the IP address "
        "they listen on?",
        "`fastapi dev` has auto-reload enabled by default and listens on `127.0.0.1`, reachable "
        "only from your own machine. `fastapi run` has auto-reload disabled by default and "
        "listens on `0.0.0.0`.",
        "`fastapi dev` reloads on code changes by default and binds to `127.0.0.1`, so only "
        "your machine can reach it; it also sets `FASTAPI_ENV=development` unless it's already "
        "set. `fastapi run` has auto-reload off and binds to `0.0.0.0`, all available "
        "addresses, which is what you want in a container.",
        passes=True,
    ),
    Correctness(
        "one part of three",
        "If I set `model_config` on a base Pydantic model, do subclasses get it, what happens "
        "when a subclass sets its own, and does a model used as a field type pick up the outer "
        "model's config?",
        "Subclasses inherit it, and configuration given on a subclass is merged with the "
        "parent's. It is not propagated to a nested model: each model has its own "
        "configuration boundary.",
        "Yes, subclasses inherit the base model's `model_config`.",
        passes=False,
    ),
    Correctness(
        "refuses an answerable question",
        "How do I turn on uv's malware check when syncing a project?",
        "Set `audit.malware-check = true` in your uv settings, or set `UV_MALWARE_CHECK=1` in "
        "your environment.",
        REFUSAL,
        passes=False,
    ),
    Correctness(
        "right words, wrong order",
        "I stack an `AfterValidator`, a `BeforeValidator` and a `WrapValidator` on one field "
        "with `Annotated`. In what order does Pydantic run them?",
        "Before and wrap validators run from right to left, and after validators then run "
        "from left to right.",
        "Before and wrap validators run from left to right, and after validators then run "
        "from right to left.",
        passes=False,
    ),
    Faithfulness(
        "faithful paraphrase",
        "uv picks its cache directory in order: a temporary one when `--no-cache` is passed "
        "[1]; otherwise the one set by `--cache-dir`, `UV_CACHE_DIR` or `tool.uv.cache-dir` "
        "[1]; failing that, a system default such as `$HOME/.cache/uv` on Unix [1].",
        faithful=True,
        contexts=[CACHE_DIR_CONTEXT],
    ),
    Faithfulness(
        "true aside the context doesn't state",
        "`fastapi run` starts the app in production mode with auto-reload disabled and listens "
        "on `0.0.0.0` [1]. Under the hood, it serves the app with Uvicorn.",
        faithful=False,
        contexts=[FASTAPI_RUN_CONTEXT],
    ),
    Faithfulness(
        "contradicts the context",
        "Yes. With `allow_credentials=True` you can keep `allow_origins=['*']` [1]; only "
        "`allow_methods` has to be listed explicitly.",
        faithful=False,
        contexts=[CORS_CONTEXT],
    ),
    Faithfulness(
        "a refusal",
        REFUSAL,
        faithful=True,
        contexts=[CACHE_DIR_CONTEXT],
    ),
]


def test_the_judge_gets_at_least_9_of_10_right():
    pytest.importorskip("judge_prompts", reason="the judge prompts aren't written yet")
    load_env(Path(__file__).parents[2] / ".env")
    provider = os.environ.get("JUDGE_PROVIDER") or DEFAULT_PROVIDER
    if not os.environ.get(KEY_ENV.get(provider, "")):
        pytest.skip(f"no {KEY_ENV.get(provider, provider)} for the judge")
    judge = Judge.from_env(budget=Budget(MAX_COST_USD))

    lines, right = [], 0
    for case in CASES:
        if isinstance(case, Correctness):
            verdict = judge.correctness(case.question, case.reference, case.answer)
            got = metrics.correctness_pass(verdict)
            expected, shown = case.passes, f"rating {verdict.rating}"
            why = verdict.reasoning
        else:
            verdict = judge.faithfulness(case.contexts, case.answer)
            score = metrics.faithfulness_score(verdict)
            got = score is None or score == 1.0
            expected = case.faithful
            shown = "no claims" if score is None else f"{score:.2f} of {len(verdict.claims)}"
            why = "; ".join(f"{c.claim} -> {c.supported}" for c in verdict.claims)
        right += got == expected
        mark = "ok  " if got == expected else "MISS"
        lines.append(f"{mark} {type(case).__name__.lower()}: {case.name} ({shown}): {why}")

    report = "\n".join(
        [
            *lines,
            f"{right}/{len(CASES)} right · {judge.cache_hits} cached · {judge.budget.summary()}",
        ]
    )
    print(f"\n{judge.provider}:{judge.model} prompts {judge.prompt_version}\n{report}")
    assert right >= 9, report
