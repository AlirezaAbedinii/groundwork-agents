"""The judge client: templates in, verdicts out, cached, under a budget.

The real SDKs run over a mocked transport, with stand-in templates, so these tests
pin the judge's mechanics and not the wording of any prompt.
"""

import json
from types import SimpleNamespace

import anthropic
import httpx
import httpx2  # the Anthropic SDK's fork of httpx
import openai
import pytest

from budget import Budget, BudgetExceeded
from judge import Judge, cache_key, render_contexts
from judge_fake import FakeJudge
from llm import LLMError
from schemas import (
    ClaimVerdict,
    Context,
    CorrectnessVerdict,
    CriterionVerdict,
    FaithfulnessVerdict,
    RubricCriterion,
    TaskVerdict,
)

PROMPTS = SimpleNamespace(
    PROMPT_VERSION="test-1",
    CORRECTNESS_SYSTEM="Grade correctness.",
    CORRECTNESS_USER="Q: {question}\nReference: {reference}\nAnswer: {answer}",
    FAITHFULNESS_SYSTEM="Grade faithfulness.",
    FAITHFULNESS_USER="Contexts:\n{contexts}\n\nAnswer: {answer}",
    TASK_SYSTEM="Grade the task.",
    TASK_USER="{request}\n---\n{final_output}\n---\n{subtask_outputs}\n---\n{criteria}",
)

CORRECT = {"reasoning": "Matches the reference.", "rating": 5}
CONTEXTS = [
    Context(
        chunk_id="a",
        text="uv _always_ requires a cache directory.",
        score=0.91,
        metadata={"source_file": "uv/concepts/cache.md", "section_heading": "Cache directory"},
    ),
    Context(chunk_id="b", text="By default, **auto-reload** is disabled.", score=0.5, metadata={}),
]


class Server:
    """Replies in order and records every request body."""

    def __init__(self, *replies: dict, http=httpx2) -> None:
        self.replies = list(replies)
        self.requests: list[dict] = []
        self.http = http

    def __call__(self, request):
        self.requests.append(json.loads(request.content))
        return self.http.Response(200, json=self.replies.pop(0))


def tool_reply(verdict: dict) -> dict:
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-haiku-4-5",
        "content": [{"type": "tool_use", "id": "toolu_1", "name": "verdict", "input": verdict}],
        "stop_reason": "tool_use",
        "stop_sequence": None,
        "usage": {"input_tokens": 400, "output_tokens": 60},
    }


def make_judge(server: Server, tmp_path, budget=None, prompts=PROMPTS) -> Judge:
    client = anthropic.Anthropic(
        api_key="test",
        max_retries=0,
        http_client=httpx2.Client(transport=httpx2.MockTransport(server)),
    )
    return Judge(
        "anthropic",
        "claude-haiku-4-5",
        cache_dir=tmp_path / "judge",
        budget=budget if budget is not None else Budget(1.0),
        prompts=prompts,
        client=client,
    )


# --- each kind: rendered templates in, its verdict model out ------------------------


def test_correctness_renders_the_templates_and_returns_a_verdict(tmp_path):
    server = Server(tool_reply(CORRECT))
    judge = make_judge(server, tmp_path)
    verdict = judge.correctness("Where is the cache?", "In ~/.cache/uv.", "~/.cache/uv")
    assert verdict == CorrectnessVerdict(**CORRECT)
    body = server.requests[0]
    assert body["system"] == "Grade correctness."
    assert body["messages"] == [
        {
            "role": "user",
            "content": "Q: Where is the cache?\nReference: In ~/.cache/uv.\nAnswer: ~/.cache/uv",
        }
    ]
    # The verdict model is the schema of the tool the model must call, reasoning first.
    assert body["tool_choice"] == {"type": "tool", "name": "verdict"}
    assert body["tools"][0]["input_schema"]["required"] == ["reasoning", "rating"]
    # 400 input and 60 output tokens at $1 and $5 per million
    assert judge.budget.calls == 1
    assert judge.budget.spent_usd == pytest.approx(400e-6 + 60 * 5e-6)


def test_values_with_braces_go_in_verbatim(tmp_path):
    server = Server(tool_reply(CORRECT))
    answer = "Declare it as `@app.get('/items/{item_id}')` and use {{double}} braces."
    make_judge(server, tmp_path).correctness("q", "r", answer)
    assert server.requests[0]["messages"][0]["content"].endswith(answer)


def test_faithfulness_numbers_and_labels_the_contexts_like_the_generator(tmp_path):
    assert render_contexts(CONTEXTS) == (
        "[1] (uv/concepts/cache.md § Cache directory)\nuv _always_ requires a cache directory."
        "\n\n[2] (unknown)\nBy default, **auto-reload** is disabled."
    )
    claims = {
        "claims": [
            {
                "claim": "uv needs a cache directory.",
                "reasoning": "[1] says so.",
                "supported": True,
            },
            {"claim": "It is in /tmp.", "reasoning": "Not stated.", "supported": False},
        ]
    }
    server = Server(tool_reply(claims))
    verdict = make_judge(server, tmp_path).faithfulness(CONTEXTS, "uv needs one, in /tmp [1].")
    assert verdict == FaithfulnessVerdict(**claims)
    assert server.requests[0]["messages"][0]["content"] == (
        f"Contexts:\n{render_contexts(CONTEXTS)}\n\nAnswer: uv needs one, in /tmp [1]."
    )


def test_task_lists_the_criteria_without_saying_which_are_critical(tmp_path):
    criteria = [
        RubricCriterion(id="c1", description="Names the cache directory"),
        RubricCriterion(id="c2", description="Cites a source", critical=False),
    ]
    reply = {
        "criteria": [
            {"id": "c1", "reasoning": "It does.", "passed": True},
            {"id": "c2", "reasoning": "No source.", "passed": False},
        ]
    }
    server = Server(tool_reply(reply))
    verdict = make_judge(server, tmp_path).task("req", "final", "sub", criteria)
    assert verdict == TaskVerdict(**reply)
    user = server.requests[0]["messages"][0]["content"]
    assert (
        user
        == "req\n---\nfinal\n---\nsub\n---\n- c1: Names the cache directory\n- c2: Cites a source"
    )
    assert "critical" not in user


@pytest.mark.parametrize(
    "call",
    [
        lambda j: j.faithfulness([], "an answer"),
        lambda j: j.task("req", "final", "sub", []),
    ],
)
def test_nothing_to_judge_against_raises_before_any_call(tmp_path, call):
    server = Server()
    with pytest.raises(ValueError):
        call(make_judge(server, tmp_path))
    assert server.requests == []


# --- the cache ----------------------------------------------------------------------


def test_a_cache_hit_costs_nothing_and_needs_no_budget(tmp_path):
    first = make_judge(Server(tool_reply(CORRECT)), tmp_path)
    verdict = first.correctness("q", "r", "a")
    # A new judge on the same cache directory, with no budget and no replies scripted.
    server = Server()
    second = make_judge(server, tmp_path, budget=Budget(0.0))
    assert second.correctness("q", "r", "a") == verdict
    assert server.requests == []
    assert second.cache_hits == 1
    assert (second.budget.calls, second.budget.spent_usd) == (0, 0.0)


def test_a_cache_entry_records_what_the_judge_saw_and_paid(tmp_path):
    judge = make_judge(Server(tool_reply(CORRECT)), tmp_path)
    judge.correctness("q", "r", "a")
    [entry_path] = (tmp_path / "judge").glob("*.json")
    entry = json.loads(entry_path.read_text())
    assert entry_path.stem == cache_key(
        "anthropic", "claude-haiku-4-5", "test-1", "Grade correctness.", entry["user"]
    )
    assert entry["kind"] == "correctness" and entry["prompt_version"] == "test-1"
    assert entry["verdict"] == CORRECT
    assert entry["usage"]["cost_usd"] == pytest.approx(judge.budget.spent_usd)


def test_a_new_prompt_version_asks_again(tmp_path):
    make_judge(Server(tool_reply(CORRECT)), tmp_path).correctness("q", "r", "a")
    server = Server(tool_reply(CORRECT))
    bumped = SimpleNamespace(**{**vars(PROMPTS), "PROMPT_VERSION": "test-2"})
    make_judge(server, tmp_path, prompts=bumped).correctness("q", "r", "a")
    assert len(server.requests) == 1


@pytest.mark.parametrize("field", range(5))
def test_the_cache_key_covers_provider_model_version_and_text(field):
    base = ["anthropic", "claude-haiku-4-5", "v1", "system", "user"]
    changed = list(base)
    changed[field] += "x"
    assert cache_key(*changed) != cache_key(*base)


def test_an_unreadable_cache_entry_is_asked_again_and_replaced(tmp_path):
    make_judge(Server(tool_reply(CORRECT)), tmp_path).correctness("q", "r", "a")
    [entry_path] = (tmp_path / "judge").glob("*.json")
    entry_path.write_text("{")
    server = Server(tool_reply(CORRECT))
    judge = make_judge(server, tmp_path)
    assert judge.correctness("q", "r", "a") == CorrectnessVerdict(**CORRECT)
    assert len(server.requests) == 1
    assert json.loads(entry_path.read_text())["verdict"] == CORRECT


# --- the budget and bad replies -----------------------------------------------------


def test_the_budget_stops_a_call_before_it_is_sent(tmp_path):
    server = Server(tool_reply(CORRECT))
    judge = make_judge(server, tmp_path, budget=Budget(0.001))  # a call may cost ~$0.011
    with pytest.raises(BudgetExceeded):
        judge.correctness("q", "r", "a")
    assert server.requests == []
    assert not (tmp_path / "judge").exists()


def test_a_reply_that_isnt_a_verdict_is_billed_but_not_cached(tmp_path):
    server = Server(tool_reply({"reasoning": "Great.", "rating": 7}), tool_reply(CORRECT))
    judge = make_judge(server, tmp_path)
    with pytest.raises(LLMError):
        judge.correctness("q", "r", "a")
    assert judge.correctness("q", "r", "a") == CorrectnessVerdict(**CORRECT)
    assert len(server.requests) == 2 and judge.budget.calls == 2
    assert judge.cache_hits == 0


# --- either provider ----------------------------------------------------------------


def test_an_openai_judge_sends_the_verdict_as_a_strict_schema(tmp_path):
    reply = {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 0,
        "model": "gpt-4o",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": json.dumps(CORRECT), "refusal": None},
            }
        ],
        "usage": {"prompt_tokens": 400, "completion_tokens": 60, "total_tokens": 460},
    }
    server = Server(reply, http=httpx)
    client = openai.OpenAI(
        api_key="test",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(server)),
    )
    judge = Judge(
        "openai", "gpt-4o", cache_dir=tmp_path, budget=Budget(1.0), prompts=PROMPTS, client=client
    )
    assert judge.correctness("q", "r", "a") == CorrectnessVerdict(**CORRECT)
    fmt = server.requests[0]["response_format"]["json_schema"]
    assert fmt["name"] == "verdict" and fmt["strict"] is True


def test_from_env_defaults_to_claude_haiku(monkeypatch, tmp_path):
    monkeypatch.delenv("JUDGE_PROVIDER", raising=False)
    monkeypatch.delenv("JUDGE_MODEL", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    judge = Judge.from_env(budget=Budget(1.0), cache_dir=tmp_path, prompts=PROMPTS)
    assert (judge.provider, judge.model, judge.prompt_version) == (
        "anthropic",
        "claude-haiku-4-5",
        "test-1",
    )


def test_from_env_reads_the_provider_and_model(monkeypatch, tmp_path):
    monkeypatch.setenv("JUDGE_PROVIDER", "openai")
    monkeypatch.setenv("JUDGE_MODEL", "gpt-4o")
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    judge = Judge.from_env(budget=Budget(1.0), cache_dir=tmp_path, prompts=PROMPTS)
    assert (judge.provider, judge.model) == ("openai", "gpt-4o")
    monkeypatch.setenv("JUDGE_PROVIDER", "gemini")
    with pytest.raises(ValueError):
        Judge.from_env(budget=Budget(1.0), cache_dir=tmp_path, prompts=PROMPTS)


# --- the fake -----------------------------------------------------------------------


def test_the_fake_passes_everything_by_default():
    fake = FakeJudge()
    criteria = [
        RubricCriterion(id="c1", description="d"),
        RubricCriterion(id="c2", description="e"),
    ]
    assert fake.correctness("q", "r", "a").rating == 5
    assert fake.faithfulness(CONTEXTS, "a").claims == [
        ClaimVerdict(claim="a", reasoning="scripted", supported=True)
    ]
    assert fake.task("req", "final", "sub", criteria).criteria == [
        CriterionVerdict(id="c1", reasoning="scripted", passed=True),
        CriterionVerdict(id="c2", reasoning="scripted", passed=True),
    ]
    assert [kind for kind, _ in fake.requests] == ["correctness", "faithfulness", "task"]
    assert fake.budget.spent_usd == 0.0


def test_the_fake_follows_its_script_and_checks_inputs_like_the_judge():
    def by_answer(question, reference, answer):
        return CorrectnessVerdict(reasoning="scripted", rating=5 if answer == reference else 2)

    fake = FakeJudge(correctness=by_answer)
    assert fake.correctness("q", "yes", "yes").rating == 5
    assert fake.correctness("q", "yes", "no").rating == 2
    with pytest.raises(ValueError):
        fake.faithfulness([], "a")
