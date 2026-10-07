"""Acceptance tests for the retrieval metrics (metrics.py part A).

A ranking is one entry per retrieved hit, best first: the set of evidence items the
hit covers. The unit of relevance is the evidence item, so an item covered by
several hits counts once, at its first rank. Expected values are worked out in the
comments; log2(3) = 1.585, log2(4) = 2, log2(5) = 2.322, log2(11) = 3.459.
"""

import math

import pytest

import metrics

NONE = frozenset()


def items(*ids: int) -> frozenset[int]:
    return frozenset(ids)


def ideal(n: int) -> float:
    """DCG of one new item at each rank 1..n."""
    return sum(1 / math.log2(rank + 1) for rank in range(1, n + 1))


# --- a perfect ranking --------------------------------------------------------------


def test_a_perfect_ranking_scores_one_everywhere():
    ranked = [items(0), items(1)]  # each item at the earliest rank it could have
    assert metrics.recall_at_k(ranked, n_items=2, k=2) == 1.0
    assert metrics.mrr_at_k(ranked, k=10) == 1.0
    assert metrics.ndcg_at_k(ranked, n_items=2, k=10) == pytest.approx(1.0)


# --- MRR ----------------------------------------------------------------------------


def test_mrr_is_the_reciprocal_rank_of_the_first_relevant_hit():
    ranked = [NONE, NONE, items(0)]
    assert metrics.mrr_at_k(ranked, k=10) == pytest.approx(1 / 3)


def test_mrr_is_zero_when_the_first_relevant_hit_is_beyond_k():
    assert metrics.mrr_at_k([NONE, items(0)], k=1) == 0.0


# --- recall and nDCG with items at ranks 1 and 4 ------------------------------------


def test_items_at_ranks_1_and_4():
    ranked = [items(0), NONE, NONE, items(1), NONE]
    # Recall@3: only item 0 is in the top 3 -> 1/2.  Recall@5: both -> 2/2.
    assert metrics.recall_at_k(ranked, n_items=2, k=3) == 0.5
    assert metrics.recall_at_k(ranked, n_items=2, k=5) == 1.0
    # nDCG@5 = (1/log2 2 + 1/log2 5) / (1/log2 2 + 1/log2 3)
    #        = (1 + 0.4307) / (1 + 0.6309) = 0.8772
    expected = (1 + 1 / math.log2(5)) / (1 + 1 / math.log2(3))
    assert metrics.ndcg_at_k(ranked, n_items=2, k=5) == pytest.approx(expected)
    assert expected == pytest.approx(0.8772, abs=1e-4)


# --- several items in one hit, one item in several hits -----------------------------


def test_one_hit_covering_both_items():
    # Two items stated in one chunk: everything is found at rank 1. The second item
    # is credited at rank 2, where the ideal puts it, so nDCG is exactly 1, not
    # (1 + 1) / 1.6309 = 1.23.
    ranked = [items(0, 1), NONE]
    assert metrics.recall_at_k(ranked, n_items=2, k=1) == 1.0
    assert metrics.mrr_at_k(ranked, k=10) == 1.0
    assert metrics.ndcg_at_k(ranked, n_items=2, k=10) == pytest.approx(1.0)


def test_one_hit_covering_both_items_at_rank_2():
    # Both items first found at rank 2: credited at ranks max(2, 1) and max(2, 2).
    # nDCG = (1/log2 3 + 1/log2 3) / (1 + 1/log2 3) = 1.2619 / 1.6309 = 0.7737,
    # better than finding them at ranks 2 and 3 (0.6309 + 0.5) / 1.6309 = 0.6934.
    both_at_2 = metrics.ndcg_at_k([NONE, items(0, 1)], n_items=2, k=10)
    at_2_and_3 = metrics.ndcg_at_k([NONE, items(0), items(1)], n_items=2, k=10)
    assert both_at_2 == pytest.approx(2 / math.log2(3) / ideal(2))
    assert at_2_and_3 == pytest.approx((1 / math.log2(3) + 0.5) / ideal(2))
    assert both_at_2 > at_2_and_3


def test_the_same_item_in_two_overlapping_chunks_counts_once():
    ranked = [items(0), items(0), items(1)]  # overlapping windows both hold item 0
    # Recall@2: still only item 0 -> 1/2, not 2/2.
    assert metrics.recall_at_k(ranked, n_items=2, k=2) == 0.5
    # nDCG: item 1 is first found at rank 3 -> (1 + 1/log2 4) / 1.6309 = 1.5 / 1.6309 = 0.9197
    assert metrics.ndcg_at_k(ranked, n_items=2, k=10) == pytest.approx(1.5 / ideal(2))


def test_a_late_third_item_still_costs_when_two_share_rank_1():
    # Items 0 and 1 at rank 1 (credited at 1 and 2), item 2 at rank 5.
    # nDCG = (1 + 0.6309 + 1/log2 6) / (1 + 0.6309 + 0.5) = 2.0177 / 2.1309 = 0.9469
    ranked = [items(0, 1), NONE, NONE, NONE, items(2)]
    expected = (1 + 1 / math.log2(3) + 1 / math.log2(6)) / ideal(3)
    assert metrics.ndcg_at_k(ranked, n_items=3, k=10) == pytest.approx(expected)
    assert expected < 1


# --- nothing relevant, short lists, n_items > k -------------------------------------


def test_nothing_relevant_scores_zero():
    ranked = [NONE] * 5
    assert metrics.recall_at_k(ranked, n_items=2, k=5) == 0.0
    assert metrics.mrr_at_k(ranked, k=5) == 0.0
    assert metrics.ndcg_at_k(ranked, n_items=2, k=5) == 0.0


def test_k_beyond_the_list_uses_what_there_is():
    assert metrics.recall_at_k([items(0)], n_items=1, k=10) == 1.0
    assert metrics.mrr_at_k([items(0)], k=10) == 1.0
    assert metrics.ndcg_at_k([items(0)], n_items=1, k=10) == pytest.approx(1.0)
    assert metrics.recall_at_k([], n_items=1, k=10) == 0.0


def test_more_items_than_k():
    # 3 items, k = 2: at most 2 can be found, and the ideal has only 2 ranks.
    ranked = [items(0), items(1), items(2)]
    assert metrics.recall_at_k(ranked, n_items=3, k=2) == pytest.approx(2 / 3)
    assert metrics.ndcg_at_k(ranked, n_items=3, k=2) == pytest.approx(1.0)
    # All three in the first hit: only min(3, 2) = 2 are credited, so nDCG is 1, not more.
    assert metrics.ndcg_at_k([items(0, 1, 2)], n_items=3, k=2) == pytest.approx(1.0)
    assert metrics.recall_at_k([items(0, 1, 2)], n_items=3, k=2) == 1.0


# --- bad input ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "call",
    [
        lambda: metrics.recall_at_k([items(0)], n_items=0, k=1),
        lambda: metrics.ndcg_at_k([items(0)], n_items=0, k=1),
        lambda: metrics.recall_at_k([items(0)], n_items=1, k=0),
        lambda: metrics.mrr_at_k([items(0)], k=0),
        lambda: metrics.ndcg_at_k([items(0)], n_items=1, k=0),
        lambda: metrics.recall_at_k([items(1)], n_items=1, k=1),  # item 1 doesn't exist
        lambda: metrics.ndcg_at_k([items(-1)], n_items=1, k=1),
    ],
)
def test_bad_arguments_raise(call):
    with pytest.raises(ValueError):
        call()


# --- when MRR and nDCG disagree (the L3.1 exercise) ---------------------------------


def test_mrr_and_ndcg_can_prefer_different_rankings():
    # A finds item 0 at rank 1 and never finds item 1; B finds both, at ranks 2 and 3.
    a = [items(0)] + [NONE] * 9
    b = [NONE, items(0), items(1)] + [NONE] * 7
    # MRR only looks at the first relevant hit: A 1/1 = 1.0, B 1/2 = 0.5 -> prefers A.
    assert metrics.mrr_at_k(a, k=10) == 1.0
    assert metrics.mrr_at_k(b, k=10) == 0.5
    # nDCG rewards finding everything: A 1/1.6309 = 0.6131, B (0.6309 + 0.5)/1.6309 = 0.6934.
    ndcg_a = metrics.ndcg_at_k(a, n_items=2, k=10)
    ndcg_b = metrics.ndcg_at_k(b, n_items=2, k=10)
    assert ndcg_a == pytest.approx(1 / ideal(2))
    assert ndcg_b == pytest.approx((1 / math.log2(3) + 0.5) / ideal(2))
    assert ndcg_b > ndcg_a
