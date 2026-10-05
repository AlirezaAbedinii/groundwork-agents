"""Acceptance tests for the refusal, answer and agreement metrics (metrics.py part B).

The positive class for refusals is "refused": a question should be refused if and
only if it is no_answer. A metric whose denominator is 0 is None, not 0.0.
Expected values are worked out in the comments.
"""

import math

import pytest

import metrics
from schemas import PRF, ClaimVerdict, CorrectnessVerdict, FaithfulnessVerdict, SweepPoint

T, F = True, False


# --- refusal precision, recall and F1 -----------------------------------------------


def test_refusal_prf_counts_a_toy_set():
    refused = [T, T, F, F, T]
    should = [T, F, T, F, T]
    # tp: 0, 4 · fp: 1 · fn: 2 · tn: 3
    # precision 2/3, recall 2/3, F1 = 2·tp / (2·tp + fp + fn) = 4/6
    prf = metrics.refusal_prf(refused, should)
    assert (prf.tp, prf.fp, prf.fn, prf.tn) == (2, 1, 1, 1)
    assert prf.precision == pytest.approx(2 / 3)
    assert prf.recall == pytest.approx(2 / 3)
    assert prf.f1 == pytest.approx(2 / 3)


def test_no_predicted_refusals():
    # Nothing refused, one should have been: precision 0/0 -> None, recall 0/1 = 0.0,
    # F1 = 0 / (0 + 0 + 1) = 0.0
    prf = metrics.refusal_prf([F, F], [T, F])
    assert prf.precision is None
    assert prf.recall == 0.0
    assert prf.f1 == 0.0


def test_nothing_to_refuse_and_nothing_refused():
    # Every metric's denominator is 0.
    prf = metrics.refusal_prf([F, F], [F, F])
    assert (prf.precision, prf.recall, prf.f1) == (None, None, None)
    assert prf.tn == 2


def test_refusal_prf_needs_equal_lengths():
    with pytest.raises(ValueError):
        metrics.refusal_prf([T], [T, F])


# --- the threshold sweep ------------------------------------------------------------


def test_sweep_covers_every_distinct_score_and_infinity():
    # Refuse iff score < threshold. Thresholds: the distinct scores ascending, then +inf.
    scores = [0.9, 0.5, 0.5, 0.2]
    should = [F, T, F, T]
    points = metrics.refusal_sweep(scores, should)
    assert [p.threshold for p in points] == [0.2, 0.5, 0.9, math.inf]
    # t = 0.2 refuses nothing           -> tp 0 fp 0 fn 2 tn 2: precision None, recall 0
    # t = 0.5 refuses {0.2}             -> tp 1 fp 0 fn 1 tn 2: precision 1,    recall 1/2
    # t = 0.9 refuses {0.5, 0.5, 0.2}   -> tp 2 fp 1 fn 0 tn 1: precision 2/3,  recall 1
    # t = inf refuses everything        -> tp 2 fp 2 fn 0 tn 0: precision 1/2,  recall 1
    assert [(p.prf.tp, p.prf.fp, p.prf.fn, p.prf.tn) for p in points] == [
        (0, 0, 2, 2),
        (1, 0, 1, 2),
        (2, 1, 0, 1),
        (2, 2, 0, 0),
    ]
    assert points[0].prf.precision is None
    recalls = [p.prf.recall for p in points]
    assert recalls == sorted(recalls)  # refusing more never lowers recall


def test_sweep_ends_never_and_always_refuse():
    points = metrics.refusal_sweep([0.3, 0.7], [T, F])
    assert points[0].prf.tp + points[0].prf.fp == 0  # lowest threshold: nothing refused
    assert points[-1].prf.fn + points[-1].prf.tn == 0  # +inf: everything refused


# --- choosing a threshold -----------------------------------------------------------


def test_choose_threshold_returns_a_swept_threshold_with_the_best_f1():
    points = metrics.refusal_sweep([0.9, 0.5, 0.5, 0.2], [F, T, F, T])
    # F1 = 2tp / (2tp + fp + fn) at 0.2: 0/2 = 0 · 0.5: 2/3 = 0.667 · 0.9: 4/5 = 0.8
    #   · inf: 4/6 = 0.667 -> 0.9
    assert metrics.choose_threshold(points) == 0.9


def test_choose_threshold_separates_a_separable_set():
    # no_answer questions score 0.1 and 0.2, answerable ones 0.6 and 0.8.
    points = metrics.refusal_sweep([0.1, 0.2, 0.6, 0.8], [T, T, F, F])
    t = metrics.choose_threshold(points)
    refused = [s < t for s in [0.1, 0.2, 0.6, 0.8]]
    assert refused == [T, T, F, F]
    assert t == 0.6  # the lowest swept threshold that gets F1 = 1


def test_choose_threshold_breaks_ties_toward_fewer_refusals():
    same = PRF(precision=0.5, recall=1.0, f1=2 / 3, tp=1, fp=1, fn=0, tn=0)
    points = [SweepPoint(0.7, same), SweepPoint(0.4, same)]
    assert metrics.choose_threshold(points) == 0.4


def test_choose_threshold_needs_something_to_refuse():
    # Nothing should be refused: F1 is None where nothing is refused, 0.0 elsewhere.
    points = metrics.refusal_sweep([0.3, 0.7], [F, F])
    with pytest.raises(ValueError):
        metrics.choose_threshold(points)


# --- Wilson intervals ---------------------------------------------------------------


def test_wilson_interval_for_8_of_10():
    # p = 0.8, z = 1.96: centre (0.8 + 0.192) / 1.384 = 0.7167,
    # half-width 1.96·sqrt(0.016 + 0.0096) / 1.384 = 0.2266 -> (0.490, 0.943)
    low, high = metrics.wilson_interval(8, 10)
    assert low == pytest.approx(0.490, abs=1e-3)
    assert high == pytest.approx(0.943, abs=1e-3)


def test_wilson_interval_at_the_edges():
    assert metrics.wilson_interval(0, 0) == (0.0, 1.0)  # no data: anything is possible
    low, high = metrics.wilson_interval(0, 10)
    # centre = half-width = 0.192 / 1.384 = 0.1388 -> (0.0, 0.2775)
    assert low == pytest.approx(0.0, abs=1e-12)
    assert high == pytest.approx(0.2775, abs=1e-3)
    low, high = metrics.wilson_interval(10, 10)
    assert (low, high) == (pytest.approx(0.7225, abs=1e-3), pytest.approx(1.0, abs=1e-12))


@pytest.mark.parametrize(("successes", "n"), [(11, 10), (-1, 10), (0, -1)])
def test_wilson_interval_rejects_impossible_counts(successes, n):
    with pytest.raises(ValueError):
        metrics.wilson_interval(successes, n)


# --- Cohen's kappa ------------------------------------------------------------------


def test_kappa_is_one_for_identical_raters():
    assert metrics.cohens_kappa([T, F, T], [T, F, T]) == pytest.approx(1.0)


def test_kappa_matches_a_textbook_table():
    # 50 items: both yes 20, A yes/B no 5, A no/B yes 10, both no 15.
    # p_o = 35/50 = 0.7; A says yes 25/50, B 30/50 -> p_e = 0.5·0.6 + 0.5·0.4 = 0.5
    # kappa = (0.7 - 0.5) / (1 - 0.5) = 0.4
    a = [T] * 20 + [T] * 5 + [F] * 10 + [F] * 15
    b = [T] * 20 + [F] * 5 + [T] * 10 + [F] * 15
    assert metrics.cohens_kappa(a, b) == pytest.approx(0.4)


def test_kappa_with_constant_raters():
    # Both always "pass": p_o = p_e = 1, so kappa is 0/0 -> None (undefined, not 1.0).
    assert metrics.cohens_kappa([T, T, T], [T, T, T]) is None
    # Constant but opposite: p_o = 0, p_e = 1·0 + 0·1 = 0 -> kappa = 0.
    assert metrics.cohens_kappa([T, T], [F, F]) == 0.0


@pytest.mark.parametrize(("a", "b"), [([], []), ([T], [T, F])])
def test_kappa_needs_paired_ratings(a, b):
    with pytest.raises(ValueError):
        metrics.cohens_kappa(a, b)


# --- correctness and faithfulness ---------------------------------------------------


@pytest.mark.parametrize(("rating", "passes"), [(1, F), (3, F), (4, T), (5, T)])
def test_correctness_passes_at_4_and_above(rating, passes):
    assert metrics.correctness_pass(CorrectnessVerdict(reasoning="r", rating=rating)) is passes


def claims(*supported: bool) -> FaithfulnessVerdict:
    return FaithfulnessVerdict(
        claims=[
            ClaimVerdict(claim=f"c{i}", reasoning="r", supported=s) for i, s in enumerate(supported)
        ]
    )


def test_faithfulness_is_the_share_of_supported_claims():
    assert metrics.faithfulness_score(claims(T, T, T, F)) == 0.75
    assert metrics.faithfulness_score(claims(T)) == 1.0


def test_faithfulness_without_claims_is_undefined():
    assert metrics.faithfulness_score(claims()) is None
