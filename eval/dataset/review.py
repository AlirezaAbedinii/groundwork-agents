"""Review candidates by hand, one at a time.

    python -m dataset.review
    python -m dataset.review --targeted [--corpus ../services/rag/data/raw/toolchain_docs]

Each candidate shows its chunk, the draft and its flags. Keys: ``a`` accept, ``e``
edit the draft's JSON in ``$EDITOR`` (the flags are re-checked), ``r`` reject with a
reason, ``s`` skip, ``q`` quit. An accepted draft becomes a golden row with the next
``s0NN`` id, ``origin: synthetic``, ``verified: true`` and a split (40/60 per
category); a rejected one moves to ``golden/synthetic_rejects.jsonl`` with its
reason. Both leave ``golden/candidates_unverified.jsonl``; skipped ones stay.

With ``--targeted`` it walks the golden-format rows in
``golden/candidates_targeted.jsonl`` that aren't verified yet, showing each quote
with its lines in the corpus file. ``a`` sets ``verified: true`` in place (nothing is
promoted into the golden set; that is a separate step), ``e`` edits the row's JSON
and saves it only if it still validates, ``r`` moves the row to
``golden/targeted_rejects.jsonl`` with a reason, ``s`` and ``q`` as above.
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
from dataset.validate import (
    Corpus,
    Findings,
    Row,
    check_candidates,
    check_corpus,
    read_rows,
    write_splits,
)
from schemas import GoldenQuestion
from textnorm import normalize_ws

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


# --- targeted candidates (golden-format rows) ---------------------------------------

CONTEXT_LINES = 1  # lines shown around a quote
MAX_QUOTE_LINES = 12  # a 120-character quote never spans more


def quote_lines(lines: list[str], quote: str) -> tuple[int, int] | None:
    """1-based first and last line of the shortest run of ``lines`` that holds ``quote``."""
    needle = normalize_ws(quote)
    for end in range(len(lines)):
        for start in range(end, max(-1, end - MAX_QUOTE_LINES), -1):
            if needle in normalize_ws("\n".join(lines[start : end + 1])):
                return start + 1, end + 1
    return None


def show_targeted(row: dict, corpus: Corpus | None, out: Callable[[str], None]) -> None:
    phrasing = f" × {row['phrasing']}" if row.get("phrasing") else ""
    out("")
    out(f"== {row.get('id')}  {row.get('category')}{phrasing}")
    out(f"Q: {row.get('question')}")
    out(f"A: {row.get('reference_answer')}")
    out(f"notes: {row.get('notes') or '-'}")
    evidence = row.get("evidence") or []
    if not evidence:
        out("evidence: none")
    for i, item in enumerate(evidence):
        for j, quote in enumerate(item.get("quotes", [])):
            source, text = quote.get("source", ""), quote.get("text", "")
            out(f"-- evidence[{i}] quote {j + 1}: {source}")
            out(f"   {text!r}")
            doc = corpus.file(source) if corpus else None
            if corpus is None:
                continue
            span = quote_lines(doc.lines, text) if doc else None
            if span is None:
                out(f"   NOT FOUND in {source}")
                continue
            first = max(1, span[0] - CONTEXT_LINES)
            last = min(len(doc.lines), span[1] + CONTEXT_LINES)
            for n in range(first, last + 1):
                mark = ">" if span[0] <= n <= span[1] else " "
                out(f"  {mark}{n:>5} | {doc.lines[n - 1]}")


def _problems(row: dict, golden: Path, corpus_root: Path | None) -> list[str]:
    findings = Findings()
    questions = check_candidates([Row(line=1, data=row)], golden, findings)
    if corpus_root is not None:
        check_corpus(questions, corpus_root, findings)
    return [message for _, message in findings.errors]


def review_targeted(
    candidates: Path,
    golden: Path,
    rejects: Path,
    corpus_root: Path | None,
    *,
    ask: Callable[[str], str] = input,
    edit: Callable[[str], str] = open_in_editor,
    out: Callable[[str], None] = print,
) -> dict[str, int]:
    """Check the unverified rows of ``candidates`` in place; returns the tally."""
    rows = read_jsonl(candidates)
    corpus = Corpus(corpus_root) if corpus_root is not None else None
    tally = {"accepted": 0, "rejected": 0, "skipped": 0}

    def save() -> None:
        candidates.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
        )

    for row in [r for r in rows if not r.get("verified")]:
        show_targeted(row, corpus, out)
        while True:
            key = ask(KEYS).strip().lower()
            if key == "a":
                row["verified"] = True
                save()
                tally["accepted"] += 1
                out(f"verified {row.get('id')}")
                break
            if key == "e":
                fields = {k: v for k, v in row.items() if k not in ("id", "verified")}
                try:
                    edited = json.loads(edit(json.dumps(fields, indent=2, ensure_ascii=False)))
                    if not isinstance(edited, dict):
                        raise TypeError("the row must stay a JSON object")
                except (json.JSONDecodeError, TypeError) as exc:
                    out(f"edit ignored: {exc}")
                    continue
                updated = {"id": row.get("id")} | edited | {"verified": row.get("verified")}
                problems = _problems(updated, golden, corpus_root)
                if problems:
                    out("edit not saved: " + "; ".join(problems))
                    continue
                row.clear()
                row.update(updated)
                save()
                show_targeted(row, corpus, out)
                continue
            if key == "r":
                reason = ask("reason > ").strip()
                if not reason:
                    out("a reject needs a reason")
                    continue
                append_jsonl(
                    rejects, row | {"reason": reason, "rejected_on": date.today().isoformat()}
                )
                rows.remove(row)
                save()
                tally["rejected"] += 1
                break
            if key == "s":
                tally["skipped"] += 1
                break
            if key == "q":
                out(_summary(tally, [r for r in rows if not r.get("verified")]))
                return tally
            out("unknown key")
    out(_summary(tally, [r for r in rows if not r.get("verified")]))
    return tally


def _summary(tally: dict[str, int], pending: list[dict]) -> str:
    return (
        f"{tally['accepted']} accepted · {tally['rejected']} rejected · "
        f"{tally['skipped']} skipped · {len(pending)} still pending"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m dataset.review", description="Review candidates by hand."
    )
    parser.add_argument(
        "--targeted", action="store_true", help="check the golden-format targeted candidates"
    )
    parser.add_argument("--candidates", type=Path)
    parser.add_argument("--golden", type=Path, default=Path("golden/golden_set.jsonl"))
    parser.add_argument("--rejects", type=Path)
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("../services/rag/data/raw/toolchain_docs"),
        help="with --targeted: the corpus root, to show each quote in its file",
    )
    args = parser.parse_args(argv)
    if args.targeted:
        candidates = args.candidates or Path("golden/candidates_targeted.jsonl")
        rejects = args.rejects or Path("golden/targeted_rejects.jsonl")
        if not candidates.is_file():
            parser.error(f"no candidates at {candidates}")
        corpus = args.corpus if args.corpus.is_dir() else None
        if corpus is None:
            print(f"no corpus at {args.corpus}: quotes are shown without their lines")
        review_targeted(candidates, args.golden, rejects, corpus)
        return 0
    args.candidates = args.candidates or Path("golden/candidates_unverified.jsonl")
    args.rejects = args.rejects or Path("golden/synthetic_rejects.jsonl")
    if not args.candidates.is_file():
        parser.error(f"no candidates at {args.candidates}; run python -m dataset.synthesize first")
    review(args.candidates, args.golden, args.rejects)
    return 0


if __name__ == "__main__":
    sys.exit(main())
