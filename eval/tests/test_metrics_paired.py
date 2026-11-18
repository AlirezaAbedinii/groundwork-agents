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


# --- exact McNemar ------------------------------------------------------------------


def test_mcnemar_on_the_published_discordant_counts():
    # 8 pass only in a, 6 only in b, 3 pass in both, 2 fail in both: d = 14.
    a = [T] * 8 + [F] * 6 + [T] * 3 + [F] * 2
    b = [F] * 8 + [T] * 6 + [T] * 3 + [F] * 2
    # Lower tail to 6: C(14, 0..6) = 1 + 14 + 91 + 364 + 1001 + 2002 + 3003 = 6476.
    # p = 2 · 6476 / 2^14 = 12952 / 16384 = 0.79052734375
    found = metrics.mcnemar_exact(a, b)
    assert (found.only_a, found.only_b, found.n) == (8, 6, 19)
    assert found.p_value == pytest.approx(12952 / 16384)


def test_mcnemar_one_sided_discordance():
    # 0 against 5: p = 2 · C(5, 0) / 2^5 = 2 / 32
    found = metrics.mcnemar_exact([F] * 5, [T] * 5)
    assert (found.only_a, found.only_b) == (0, 5)
    assert found.p_value == pytest.approx(0.0625)


def test_mcnemar_is_symmetric_and_capped_at_one():
    # 3 against 3: 2 · (1 + 6 + 15 + 20) / 64 = 1.3125, capped to 1.0.
    a, b = [T, T, T, F, F, F], [F, F, F, T, T, T]
    assert metrics.mcnemar_exact(a, b).p_value == 1.0
    one = metrics.mcnemar_exact([T, F, F], [F, T, T])
    other = metrics.mcnemar_exact([F, T, T], [T, F, F])
    assert (one.only_a, one.only_b) == (other.only_b, other.only_a) == (1, 2)
    # 1 against 2: 2 · (C(3, 0) + C(3, 1)) / 2^3 = 2 · 4 / 8 = 1.0
    assert one.p_value == other.p_value == 1.0
    # Nothing discordant: no evidence either way.
    assert metrics.mcnemar_exact([T, F], [T, F]).p_value == 1.0


@pytest.mark.parametrize(("a", "b"), [([], []), ([T], [T, F])])
def test_mcnemar_needs_paired_results(a, b):
    with pytest.raises(ValueError, match="equally long"):
        metrics.mcnemar_exact(a, b)


# --- paired bootstrap ---------------------------------------------------------------


def test_bootstrap_of_a_constant_difference_has_no_width():
    # Every pair differs by +0.5, so every resample's mean is 0.5.
    found = metrics.paired_bootstrap([0.0, 0.25, 0.5], [0.5, 0.75, 1.0], resamples=200)
    assert (found.difference, found.low, found.high) == pytest.approx((0.5, 0.5, 0.5))
    assert (found.n, found.resamples, found.seed) == (3, 200, metrics.BOOTSTRAP_SEED)


def test_bootstrap_of_two_pairs():
    # Differences b - a are 0 and 1. A resample of two draws has mean 0, 0.5 or 1 with
    # probability 1/4, 1/2, 1/4. At 95 % the 2.5th percentile is 0 (a quarter of the
    # resamples are 0) and the 97.5th is 1 (a quarter are 1). At 40 % the 30th and 70th
    # percentiles both fall among the 0.5s (cumulative 1/4 < 0.3 and 3/4 >= 0.7). With
    # 10,000 resamples each count is within a few standard errors (~0.004) of its share.
    a, b = [0.0, 0.0], [0.0, 1.0]
    found = metrics.paired_bootstrap(a, b)
    assert (found.difference, found.low, found.high) == (0.5, 0.0, 1.0)
    narrow = metrics.paired_bootstrap(a, b, confidence=0.4)
    assert (narrow.low, narrow.high) == (0.5, 0.5)


def test_bootstrap_difference_is_b_minus_a():
    found = metrics.paired_bootstrap([1.0, 1.0], [0.0, 0.0], resamples=10)
    assert (found.difference, found.low, found.high) == (-1.0, -1.0, -1.0)


def test_bootstrap_is_reproducible_from_its_seed():
    a = [0.0, 0.5, 1.0, 1.0, 0.0, 0.25]
    b = [1.0, 0.5, 0.0, 1.0, 0.5, 0.25]
    assert metrics.paired_bootstrap(a, b, seed=7) == metrics.paired_bootstrap(a, b, seed=7)
    assert metrics.paired_bootstrap(a, b, seed=7).seed == 7


def test_nearest_rank_picks_the_250th_and_9750th_of_10000():
    # (1 - 0.95) / 2 is 0.025000000000000022 in floating point; without rounding first,
    # the ceiling would pick the 251st value.
    ordered = [float(i) for i in range(1, 10_001)]
    tail = (1 - 0.95) / 2
    assert metrics._nearest_rank(ordered, tail) == 250.0
    assert metrics._nearest_rank(ordered, 1 - tail) == 9750.0


@pytest.mark.parametrize(
    ("a", "b", "kwargs", "match"),
    [
        ([], [], {}, "equally long"),
        ([1.0], [1.0, 2.0], {}, "equally long"),
        ([1.0], [2.0], {"resamples": 0}, "resamples"),
        ([1.0], [2.0], {"confidence": 1.0}, "confidence"),
        ([1.0], [float("nan")], {}, "finite"),
    ],
)
def test_bootstrap_rejects_bad_input(a, b, kwargs, match):
    with pytest.raises(ValueError, match=match):
        metrics.paired_bootstrap(a, b, **kwargs)
