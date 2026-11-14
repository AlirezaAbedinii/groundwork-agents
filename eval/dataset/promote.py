"""Promote the checked targeted candidates into the golden set.

    python -m dataset.promote [--candidates golden/candidates_targeted.jsonl]
                              [--golden golden/golden_set.jsonl]

Only rows with ``verified: true`` move; the others stay in the candidates file. Each
promoted row gets a split by ``split_by_stratum``: within its stratum (category x
phrasing) of the promoted rows, the rows are ordered by a hash of ``SPLIT_SEED`` and
their id, and the first ``round(0.4 * n)`` go to ``dev``, the rest to ``test``. The
rows already in the golden set are not read for this and their lines are not
rewritten: the promoted rows are appended after them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

from dataset.validate import (
    DEV_SHARE,
    Findings,
    _round_half_up,
    _with_split,
    check_golden,
    read_rows,
)

SPLIT_SEED = "pk-20-targeted-2026-11"


def _order(row_id: str, seed: str) -> str:
    return hashlib.sha256(f"{seed}:{row_id}".encode()).hexdigest()


def split_by_stratum(rows: list[dict], seed: str = SPLIT_SEED) -> dict[str, str]:
    """Each row's split, keyed by id: ``round(DEV_SHARE * n)`` dev rows per stratum, chosen
    by the seeded hash order. The result depends only on the ids, strata and seed, not
    on the order of ``rows``."""
    strata: dict[tuple[str, str | None], list[str]] = defaultdict(list)
    for row in rows:
        strata[row["category"], row.get("phrasing")].append(row["id"])
    splits = {}
    for ids in strata.values():
        n_dev = _round_half_up(DEV_SHARE * len(ids))
        for rank, row_id in enumerate(sorted(ids, key=lambda i: _order(i, seed))):
            splits[row_id] = "dev" if rank < n_dev else "test"
    return splits


def _lines(path: Path) -> list[str]:
    if not path.is_file():
        return []
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def promote(candidates: Path, golden: Path, seed: str = SPLIT_SEED) -> list[dict]:
    """Move the verified rows of ``candidates`` to the end of ``golden`` with their splits;
    returns the promoted rows. Raises ``ValueError``, writing nothing, if a promoted row
    would not be a valid golden row or its id is already taken."""
    rows = [json.loads(line) for line in _lines(candidates)]
    ready = [r for r in rows if r.get("verified") is True]
    if not ready:
        return []
    existing = _lines(golden)
    taken = {json.loads(line)["id"] for line in existing}
    if clash := sorted(r["id"] for r in ready if r["id"] in taken):
        raise ValueError(f"already in {golden.name}: {', '.join(clash)}")
    if "split" in {k for r in ready for k in r}:
        raise ValueError("a candidate has a split already; candidates get one here")
    splits = split_by_stratum(ready, seed)
    promoted = [_with_split(r, splits[r["id"]]) for r in ready]

    findings = Findings()
    check_golden(read_rows([json.dumps(r) for r in promoted], findings), findings)
    if findings.errors:
        raise ValueError("; ".join(f"{where}: {message}" for where, message in findings.errors))

    new_lines = [json.dumps(r, ensure_ascii=False) for r in promoted]
    golden.write_text("\n".join(existing + new_lines) + "\n", encoding="utf-8")
    left = [r for r in rows if r.get("verified") is not True]
    candidates.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in left), encoding="utf-8"
    )
    return promoted


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m dataset.promote",
        description="Move the verified targeted candidates into the golden set.",
    )
    parser.add_argument("--candidates", type=Path, default=Path("golden/candidates_targeted.jsonl"))
    parser.add_argument("--golden", type=Path, default=Path("golden/golden_set.jsonl"))
    args = parser.parse_args(argv)
    if not args.candidates.is_file():
        parser.error(f"no candidates at {args.candidates}")
    try:
        promoted = promote(args.candidates, args.golden)
    except ValueError as exc:
        print(f"nothing promoted: {exc}")
        return 1
    n_dev = sum(r["split"] == "dev" for r in promoted)
    print(f"promoted {len(promoted)} rows ({n_dev} dev, {len(promoted) - n_dev} test)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
