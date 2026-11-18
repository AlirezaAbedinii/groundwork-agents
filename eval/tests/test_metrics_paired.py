"""Acceptance tests for refusals of answerable questions and the paired mode comparison.

Expected values are worked out by hand in the comments.
"""

import pytest

import metrics
from schemas import FalseRefusals, Share

T, F = True, False


# --- refusals of answerable questions -----------------------------------------------


def test_false_refusals_count_each_refuser():
    # 5 answerable questions: 2 refused by the gate, 1 by the model, 2 answered.
    found = metrics.false_refusals(["gate", None, "model", "gate", None])
    assert found == FalseRefusals(refused=Share(3, 5), gate=Share(2, 5), model=Share(1, 5))
    assert found.refused.rate == pytest.approx(0.6)


def test_false_refusals_with_nothing_refused_or_no_questions():
    assert metrics.false_refusals([None, None]).refused == Share(0, 2)
    empty = metrics.false_refusals([])
    assert empty.refused == Share(0, 0) and empty.refused.rate is None


def test_false_refusals_reject_an_unknown_refuser():
    # A typo or a new refuser would otherwise count as a refusal by nobody.
    with pytest.raises(ValueError, match="unknown refusers"):
        metrics.false_refusals(["gate", "judge"])


# --- strata -------------------------------------------------------------------------


def test_per_stratum_slices_every_column_alike():
    strata = ["b", "a", "b", "a", "c"]
    left = [1, 2, 3, 4, 5]
    right = [10, 20, 30, 40, 50]
    # a: rows 1 and 3 -> [2, 4] with [20, 40]; b: rows 0 and 2; c: row 4. Sorted keys.
    out = metrics.per_stratum(strata, lambda x, y: (x, y), left, right)
    assert list(out) == ["a", "b", "c"]
    assert out["a"] == ([2, 4], [20, 40])
    assert out["b"] == ([1, 3], [10, 30])
    assert out["c"] == ([5], [50])


def test_per_stratum_applies_a_metric():
    out = metrics.per_stratum(["x", "y", "x"], metrics.false_refusals, ["gate", None, None])
    assert out["x"].refused == Share(1, 2)
    assert out["y"].refused == Share(0, 1)


def test_per_stratum_needs_aligned_columns():
    with pytest.raises(ValueError, match="rows for 2 strata"):
        metrics.per_stratum(["a", "b"], len, [1, 2, 3])
