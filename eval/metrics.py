"""The evaluation's metrics: pure functions over what the harnesses recorded.

Stdlib only, no I/O. The harnesses turn raw runs into the arguments below and the
reports print the results, so a metric can be fixed and every run re-scored for free.

Conventions every function follows:

- **The unit of relevance is the evidence item**, not the chunk. A ranked hit is the
  set of items it covers; an item covered by several hits counts once, at its first
  rank, and ``n_items`` is the number of items, never the number of relevant chunks.
- **Undefined is None, not 0.0**: a ratio with a zero denominator is None, which the
  reports print as "n/a" and leave out of means.
- **Bad input raises ValueError** instead of producing a number that looks valid.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

# One entry per ranked hit, best first: the evidence items (indexes into the question's
# evidence list) that the hit covers.
Ranked = Sequence[frozenset[int]]


# --- retrieval ----------------------------------------------------------------------


def _check_k(k: int) -> None:
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")


def _check_items(ranked: Ranked, n_items: int) -> None:
    if n_items <= 0:
        raise ValueError(f"n_items must be >= 1, got {n_items} (no_answer rows have no ranking)")
    for rank, hit in enumerate(ranked, start=1):
        if bad := sorted(i for i in hit if not 0 <= i < n_items):
            raise ValueError(f"hit {rank} covers items {bad}, outside range({n_items})")


def _first_ranks(ranked: Ranked, k: int) -> list[int]:
    """The 1-based rank at which each distinct item is first covered in the top k, in order.

    One entry per item, so an item seen again lower down adds nothing, and a hit that
    covers two new items adds its rank twice.
    """
    seen: set[int] = set()
    ranks: list[int] = []
    for rank, hit in enumerate(ranked[:k], start=1):
        new = hit - seen
        ranks.extend([rank] * len(new))
        seen |= new
    return ranks


def _discount(rank: int) -> float:
    return 1 / math.log2(rank + 1)


def recall_at_k(ranked: Ranked, n_items: int, k: int) -> float:
    """The share of the question's evidence items covered anywhere in the top k.

    Rewards finding everything the answer needs and ignores order within the top k,
    so it measures what the generator is given, not how well it's ranked. Duplicates
    don't help: two chunks holding the same item count as one item found.
    """
    _check_items(ranked, n_items)
    _check_k(k)
    return len(_first_ranks(ranked, k)) / n_items


def mrr_at_k(ranked: Ranked, k: int = 10) -> float:
    """1 / rank of the first hit that covers any evidence item, or 0.0 if none is in the top k.

    Rewards putting something relevant at the very top and nothing else: it can't
    tell whether the other items were found, so on multi-item questions it is a
    "first useful hit" score, not a completeness score.
    """
    _check_k(k)
    for rank, hit in enumerate(ranked[:k], start=1):
        if hit:
            return 1 / rank
    return 0.0


def ndcg_at_k(ranked: Ranked, n_items: int, k: int = 10) -> float:
    """DCG of the items found in the top k over the DCG of the best possible ranking.

    Rewards finding every item and finding it early: each item is worth
    1 / log2(rank + 1), so a late item still counts, but less. The j-th distinct item
    found is credited at rank max(r_j, j), where r_j is the rank of the hit that first
    covered it: a hit covering two new items is worth what two hits at the top would
    be, never more, so the score stays in [0, 1]. The ideal puts one new item at each
    of ranks 1..min(n_items, k), and at most that many items are credited.
    """
    _check_items(ranked, n_items)
    _check_k(k)
    creditable = min(n_items, k)
    found = _first_ranks(ranked, k)[:creditable]
    dcg = sum(_discount(max(rank, j)) for j, rank in enumerate(found, start=1))
    ideal = sum(_discount(j) for j in range(1, creditable + 1))
    return dcg / ideal
