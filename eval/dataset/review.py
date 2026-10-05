"""Review synthetic candidates by hand, one at a time.

    python -m dataset.review

Each candidate shows its chunk, the draft and its flags. Keys: ``a`` accept, ``e``
edit the draft's JSON in ``$EDITOR`` (the flags are re-checked), ``r`` reject with a
reason, ``s`` skip, ``q`` quit. An accepted draft becomes a golden row with the next
``s0NN`` id, ``origin: synthetic``, ``verified: true`` and a split (40/60 per
category); a rejected one moves to ``golden/synthetic_rejects.jsonl`` with its
reason. Both leave ``golden/candidates_unverified.jsonl``; skipped ones stay.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Callable, Sequence
from datetime import date
from pathlib import Path

from pydantic import ValidationError

from dataset.synthesize import append_jsonl, flag_draft, read_jsonl
from dataset.validate import Findings, read_rows, write_splits
from schemas import GoldenQuestion

EDITABLE = ("question", "reference_answer", "quote")
KEYS = "[a]ccept  [e]dit  [r]eject  [s]kip  [q]uit > "


def open_in_editor(text: str) -> str:
    """Edit ``text`` in ``$EDITOR`` (``vi`` if unset) and return the result."""
    with tempfile.NamedTemporaryFile("w+", suffix=".json", delete=False) as f:
        f.write(text)
        path = f.name
    try:
        subprocess.run([os.environ.get("EDITOR", "vi"), path], check=True)
        return Path(path).read_text(encoding="utf-8")
    finally:
        os.unlink(path)


def show(candidate: dict, out: Callable[[str], None]) -> None:
    out("")
    out(f"== {candidate['candidate_id']}  {candidate['source']}  § {candidate['heading'] or '-'}")
    out(candidate["chunk_text"])
    out("--")
    out(f"Q: {candidate['question']}")
    out(f"A: {candidate['reference_answer']}")
    out(f"quote: {candidate['quote']}")
    for flag in candidate["flags"]:
        out(f"FLAG: {flag}")


def golden_row(candidate: dict, row_id: str) -> dict:
    heading = f" § {candidate['heading']}" if candidate["heading"] else ""
    return {
        "id": row_id,
        "question": candidate["question"],
        "reference_answer": candidate["reference_answer"],
        "category": "lookup",
        "evidence": [{"quotes": [{"source": candidate["source"], "text": candidate["quote"]}]}],
        "origin": "synthetic",
        "verified": True,
        "notes": (
            f"synthetic: drafted by {candidate['model']} from one chunk of "
            f"{candidate['source']}{heading} (chunk {candidate['chunk_id']}), checked by hand."
        ),
    }


def next_synthetic_id(golden: Path) -> str:
    ids = [int(r["id"][1:]) for r in read_jsonl(golden) if r["id"].startswith("s")]
    return f"s{max(ids, default=0) + 1:03}"


def accept(candidate: dict, golden: Path) -> tuple[str | None, list[str]]:
    """Append the candidate as a golden row and give it a split; returns its id, or the
    schema's objections and no id."""
    row = golden_row(candidate, next_synthetic_id(golden))
    try:
        GoldenQuestion.model_validate(row | {"split": "test"})  # the split is assigned below
    except ValidationError as exc:
        return None, [f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()]
    lines = golden.read_text(encoding="utf-8").splitlines() if golden.is_file() else []
    lines = [line for line in lines if line.strip()] + [json.dumps(row, ensure_ascii=False)]
    golden.write_text("\n".join(lines) + "\n", encoding="utf-8")
    rows = read_rows(lines, Findings())
    write_splits(golden, lines, rows)
    return row["id"], []


def review(
    candidates: Path,
    golden: Path,
    rejects: Path,
    *,
    ask: Callable[[str], str] = input,
    edit: Callable[[str], str] = open_in_editor,
    out: Callable[[str], None] = print,
) -> dict[str, int]:
    pending = read_jsonl(candidates)
    tally = {"accepted": 0, "rejected": 0, "skipped": 0}

    def save() -> None:
        candidates.write_text(
            "".join(json.dumps(c, ensure_ascii=False) + "\n" for c in pending), encoding="utf-8"
        )

    for candidate in list(pending):
        show(candidate, out)
        while True:
            key = ask(KEYS).strip().lower()
            if key == "a":
                row_id, problems = accept(candidate, golden)
                if problems:
                    out("not accepted; edit the draft first: " + "; ".join(problems))
                    continue
                pending.remove(candidate)
                save()
                tally["accepted"] += 1
                out(f"accepted as {row_id}")
                break
            if key == "e":
                fields = {k: candidate[k] for k in EDITABLE}
                try:
                    edited = json.loads(edit(json.dumps(fields, indent=2, ensure_ascii=False)))
                    candidate.update({k: str(edited[k]) for k in EDITABLE})
                except (json.JSONDecodeError, KeyError, TypeError) as exc:
                    out(f"edit ignored: {exc}")
                    continue
                existing = {r["id"]: r["question"] for r in read_jsonl(golden)} | {
                    c["candidate_id"]: c["question"] for c in pending if c is not candidate
                }
                candidate["flags"] = flag_draft(
                    candidate["question"], candidate["quote"], candidate["chunk_text"], existing
                )
                save()
                show(candidate, out)
                continue
            if key == "r":
                reason = ask("reason > ").strip()
                if not reason:
                    out("a reject needs a reason")
                    continue
                append_jsonl(
                    rejects, candidate | {"reason": reason, "rejected_on": date.today().isoformat()}
                )
                pending.remove(candidate)
                save()
                tally["rejected"] += 1
                break
            if key == "s":
                tally["skipped"] += 1
                break
            if key == "q":
                out(_summary(tally, pending))
                return tally
            out("unknown key")
    out(_summary(tally, pending))
    return tally


def _summary(tally: dict[str, int], pending: list[dict]) -> str:
    return (
        f"{tally['accepted']} accepted · {tally['rejected']} rejected · "
        f"{tally['skipped']} skipped · {len(pending)} still pending"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m dataset.review", description="Review synthetic candidates by hand."
    )
    parser.add_argument(
        "--candidates", type=Path, default=Path("golden/candidates_unverified.jsonl")
    )
    parser.add_argument("--golden", type=Path, default=Path("golden/golden_set.jsonl"))
    parser.add_argument("--rejects", type=Path, default=Path("golden/synthetic_rejects.jsonl"))
    args = parser.parse_args(argv)
    if not args.candidates.is_file():
        parser.error(f"no candidates at {args.candidates}; run python -m dataset.synthesize first")
    review(args.candidates, args.golden, args.rejects)
    return 0


if __name__ == "__main__":
    sys.exit(main())
