"""Rules versions (orchestrator/rules.py): one function maps a version to its switches."""

import dataclasses
import re
from pathlib import Path

import pytest

import orchestrator
from orchestrator.rules import CURRENT_RULES_VERSION, Rules, effective_version, rules_for


def test_no_version_is_version_1_the_rules_from_before_the_lint():
    assert effective_version(None) == 1
    assert rules_for(None) == rules_for(1) == Rules(
        lint_plans=False, deliver_single_output=False, check_deliverables=False, require_tool_results=False,
        only_requested_files=False,
    )


def test_version_2_lints_plans_and_delivers_a_single_output():
    assert rules_for(2) == Rules(
        lint_plans=True, deliver_single_output=True, check_deliverables=False, require_tool_results=False,
        only_requested_files=False,
    )


def test_version_3_also_checks_deliverables():
    assert rules_for(3) == Rules(
        lint_plans=True, deliver_single_output=True, check_deliverables=True, require_tool_results=False,
        only_requested_files=False,
    )


def test_version_4_also_requires_a_result_from_a_subtasks_tools():
    assert rules_for(4) == Rules(
        lint_plans=True, deliver_single_output=True, check_deliverables=True, require_tool_results=True,
        only_requested_files=False,
    )


def test_version_5_also_plans_only_the_files_the_request_names():
    assert CURRENT_RULES_VERSION == 5
    assert rules_for(5) == Rules(
        lint_plans=True, deliver_single_output=True, check_deliverables=True, require_tool_results=True,
        only_requested_files=True,
    )


def test_an_unknown_version_is_refused():
    with pytest.raises(ValueError, match="Unknown rules version 99"):
        rules_for(99)


def test_the_switches_are_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        rules_for(2).lint_plans = False


def test_only_the_rules_module_compares_version_numbers():
    # everything else reads the switches rules_for returns
    comparison = re.compile(r"rules_version\W*(==|!=|<=|>=|<|>)|(==|!=|<=|>=|<|>)\W*\w*rules_version", re.I)
    package = Path(orchestrator.__file__).parent
    offenders = [
        f"{path.relative_to(package)}:{number}"
        for path in package.rglob("*.py")
        if path.name != "rules.py"
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if comparison.search(line)
    ]
    assert offenders == []
