"""Promoting checked targeted candidates, and the split rule inside each stratum."""

import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

from dataset.promote import main, promote, split_by_stratum

GOLDEN = Path(__file__).parent.parent / "golden" / "golden_set.jsonl"

# The 75 rows the set had before the expansion, byte for byte (ids, text, splits).
ORIGINAL_ROWS = 75
ORIGINAL_SHA256 = "6411de51afc2ae2835b592781f3740ecb04285d4311756ffd7297785e7545474"


def row(n: int, category: str = "lookup", phrasing: str | None = "exact_token", **extra) -> dict:
    out = {
        "id": f"q{n:03}",
        "question": f"Question {n}?",
        "reference_answer": "An answer.",
        "category": category,
        "evidence": []
        if category == "no_answer"
        else [{"quotes": [{"source": "uv/a.md", "text": f"a quote for row {n}"}]}]
        * (1 if category == "lookup" else 2),
        "origin": "targeted",
        "verified": True,
        "notes": "",
    }
    if phrasing is not None:
        out["phrasing"] = phrasing
    return out | extra


def write(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    return path


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.fixture
def golden(tmp_path) -> Path:
    old = [row(1, split="dev"), row(2, "no_answer", None, split="test")]
    for r in old:
        r.pop("phrasing", None)  # the rows before q051 have no phrasing
    return write(tmp_path / "golden.jsonl", old)


def mixed_rows() -> list[dict]:
    rows = [row(n, "lookup", "exact_token") for n in range(51, 81)]  # 30
    rows += [row(n, "lookup", "paraphrase") for n in range(81, 112)]  # 31
    rows += [row(n, "multi_hop", "paraphrase") for n in range(112, 126)]  # 14
    rows += [row(n, "ambiguous", None) for n in range(126, 135)]  # 9
    rows += [row(n, "no_answer", None) for n in range(135, 158)]  # 23
    return rows


def test_each_stratum_gets_40_percent_dev_rounded_half_up():
    rows = mixed_rows()
    splits = split_by_stratum(rows)
    dev = Counter((r["category"], r.get("phrasing")) for r in rows if splits[r["id"]] == "dev")
    assert dev == {
        ("lookup", "exact_token"): 12,
        ("lookup", "paraphrase"): 12,
        ("multi_hop", "paraphrase"): 6,
        ("ambiguous", None): 4,
        ("no_answer", None): 9,
    }
    assert set(splits.values()) == {"dev", "test"} and len(splits) == len(rows)


def test_the_split_rule_is_deterministic_and_ignores_row_order():
    rows = mixed_rows()
    first = split_by_stratum(rows)
    assert split_by_stratum(list(reversed(rows))) == first
    assert split_by_stratum(rows, seed="another seed") != first  # the seed picks the rows
    # dev rows are not simply the first ones written
    lookups = [r["id"] for r in rows[:30]]
    assert [i for i in lookups if first[i] == "dev"] != lookups[:12]


def test_promote_appends_verified_rows_and_leaves_existing_lines_alone(tmp_path, golden):
    before = golden.read_bytes()
    pending = row(53, verified=False)
    candidates = write(tmp_path / "c.jsonl", [row(51), row(52, "no_answer", None), pending])

    promoted = promote(candidates, golden)

    assert [r["id"] for r in promoted] == ["q051", "q052"]
    assert golden.read_bytes().startswith(before)
    assert [r["id"] for r in read(golden)] == ["q001", "q002", "q051", "q052"]
    assert all(r["split"] in ("dev", "test") and r["verified"] for r in read(golden)[2:])
    assert list(read(golden)[2])[:7] == [
        "id",
        "question",
        "reference_answer",
        "category",
        "evidence",
        "origin",
        "split",
    ]
    assert read(candidates) == [pending]  # unverified rows stay where they are


def test_promote_refuses_an_id_already_in_the_golden_set(tmp_path, golden):
    before = golden.read_bytes()
    candidates = write(tmp_path / "c.jsonl", [row(1), row(51)])
    with pytest.raises(ValueError, match="already in golden.jsonl: q001"):
        promote(candidates, golden)
    assert golden.read_bytes() == before


def test_promote_refuses_an_invalid_row_and_writes_nothing(tmp_path, golden, capsys):
    before = golden.read_bytes()
    bad = row(51, "multi_hop", "paraphrase") | {"evidence": []}
    candidates = write(tmp_path / "c.jsonl", [bad])
    assert main(["--candidates", str(candidates), "--golden", str(golden)]) == 1
    assert "nothing promoted" in capsys.readouterr().out
    assert golden.read_bytes() == before
    assert read(candidates) == [bad]


def test_the_original_golden_rows_are_unchanged():
    lines = GOLDEN.read_text(encoding="utf-8").splitlines()
    head = "".join(line + "\n" for line in lines[:ORIGINAL_ROWS])
    assert hashlib.sha256(head.encode()).hexdigest() == ORIGINAL_SHA256


def test_the_promoted_rows_follow_the_split_rule():
    rows = [json.loads(line) for line in GOLDEN.read_text(encoding="utf-8").splitlines()]
    promoted = rows[ORIGINAL_ROWS:]
    assert promoted and all(r["origin"] == "targeted" and r["verified"] for r in promoted)
    expected = split_by_stratum([{k: v for k, v in r.items() if k != "split"} for r in promoted])
    assert {r["id"]: r["split"] for r in promoted} == expected
