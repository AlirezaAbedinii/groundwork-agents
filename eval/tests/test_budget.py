"""The spending cap."""

import pytest

from budget import Budget, BudgetExceeded


def test_a_call_that_fits_is_allowed_up_to_the_cap():
    budget = Budget(0.10)
    budget.check(0.10)
    budget.record(0.04)
    budget.check(0.06)
    with pytest.raises(BudgetExceeded, match="up to"):
        budget.check(0.0601)


def test_record_counts_calls_and_spend():
    budget = Budget(1.0)
    budget.record(0.25)
    budget.record(0.0)
    assert budget.calls == 2
    assert budget.spent_usd == pytest.approx(0.25)
    assert budget.remaining_usd == pytest.approx(0.75)
    assert budget.summary() == "2 calls · $0.250000 of a $1.00 cap"


def test_a_negative_cap_is_rejected():
    with pytest.raises(ValueError):
        Budget(-0.01)
