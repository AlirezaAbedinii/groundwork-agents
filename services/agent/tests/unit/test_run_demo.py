"""The showcase script (scripts/run_demo.py) stops up front on a stack that can't run it."""

import argparse
import importlib.util
from pathlib import Path

import httpx
import pytest

_spec = importlib.util.spec_from_file_location("run_demo", Path(__file__).parents[2] / "scripts" / "run_demo.py")
run_demo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_demo)

COMPOSE = {"code_exec": "docker", "max_plan_subtasks": 5, "plan_lint": "enforce"}


def demo_against(config: dict):
    demo = run_demo.Demo(argparse.Namespace(api="http://agent.test"))
    demo.api = httpx.Client(base_url="http://agent.test", transport=httpx.MockTransport(
        lambda request: httpx.Response(200, json=config)
    ))
    return demo


def test_the_compose_stack_passes():
    demo_against(COMPOSE).check_stack()


@pytest.mark.parametrize(("config", "reason"), [
    pytest.param({**COMPOSE, "code_exec": "disabled", "max_plan_subtasks": 4},
                 "code_exec is disabled (no sandbox, as in the Azure deployment), and both demo plans run code; "
                 "plans are limited to 4 subtasks", id="the Azure deployment"),
    pytest.param({**COMPOSE, "max_plan_subtasks": 4}, "plans are limited to 4 subtasks", id="make dev at the default"),
])
def test_a_stack_that_cant_run_it_stops_with_the_reason(config, reason, capsys):
    with pytest.raises(SystemExit):
        demo_against(config).check_stack()

    assert reason in capsys.readouterr().err
