"""The contract for judge_prompts.py: what the judge fills in, and what it must never see.

The judge fills each template with ``str.format``, so the placeholders are parsed the
same way here: a stray brace, a misspelled or missing field, a format spec or a
positional ``{}`` fails. The task-rubric templates are checked once they exist.
"""

import re
import string

import pytest

import judge_prompts

PLACEHOLDERS = {
    "CORRECTNESS_SYSTEM": set(),
    "CORRECTNESS_USER": {"question", "reference", "answer"},
    "FAITHFULNESS_SYSTEM": set(),
    "FAITHFULNESS_USER": {"contexts", "answer"},
    "TASK_SYSTEM": set(),
    "TASK_USER": {"request", "final_output", "subtask_outputs", "criteria"},
}
MAX_CHARS = 2500

_no_task_templates = pytest.mark.skipif(
    not hasattr(judge_prompts, "TASK_SYSTEM"),
    reason="the task-rubric templates aren't written yet",
)
TEMPLATES = [
    pytest.param(name, marks=_no_task_templates) if name.startswith("TASK_") else name
    for name in PLACEHOLDERS
]


def fields(template: str) -> list[tuple[str, str, str | None]]:
    """(name, format spec, conversion) of every replacement field, as str.format sees them."""
    return [
        (name, spec, conversion)
        for _, name, spec, conversion in string.Formatter().parse(template)
        if name is not None
    ]


def test_the_prompt_version_is_set():
    assert isinstance(judge_prompts.PROMPT_VERSION, str)
    assert judge_prompts.PROMPT_VERSION.strip()


@pytest.mark.parametrize("name", TEMPLATES)
def test_placeholders_are_exactly_the_contract(name):
    found = fields(getattr(judge_prompts, name))
    assert {field for field, _, _ in found} == PLACEHOLDERS[name]
    assert all(spec == "" and conversion is None for _, spec, conversion in found), (
        "plain {name} fields only"
    )


@pytest.mark.parametrize("name", TEMPLATES)
def test_templates_render_and_stay_short(name):
    template = getattr(judge_prompts, name)
    rendered = template.format(**{field: f"<{field}>" for field in PLACEHOLDERS[name]})
    assert all(f"<{field}>" in rendered for field in PLACEHOLDERS[name])
    assert len(template) <= MAX_CHARS


def test_faithfulness_never_mentions_the_reference_answer():
    # Faithfulness is judged against the retrieved contexts alone.
    text = judge_prompts.FAITHFULNESS_SYSTEM + judge_prompts.FAITHFULNESS_USER
    assert "reference" not in text.lower()


@pytest.mark.parametrize("rating", range(1, 6))
def test_the_correctness_scale_defines_every_rating(rating):
    # Each level of the 1-5 scale starts its own line ("5: ...", "- **5** ..."), so a
    # missing level is visible.
    assert re.search(rf"(?m)^\W*{rating}\b", judge_prompts.CORRECTNESS_SYSTEM)
