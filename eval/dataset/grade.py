"""Grade a sample of judged answers by hand, blind to the judge, to measure the judge.

    python -m dataset.grade --run-id my-run

Samples ``--n`` answers (30) that the judge rated for correctness in a run's answer
stage, stratified by mode and category with a fixed seed, and shows them in shuffled
order: the question, the reference answer and the system's answer, never the judge's
rating or reasoning, nor the mode. Keys: ``p`` pass, ``f`` fail, ``s`` skip, ``q``
quit; a pass or a fail takes an optional note. Pass means what the judge's pass means:
correct in substance, so a reader following the answer gets everything the question
asked for right (a 4 or 5 on the judge's scale).

Grades are appended to ``golden/human_grades.jsonl`` with a hash of the answer, so a
grade only ever counts against the answer it was given for, and already graded answers
are skipped when grading resumes. ``rag_eval score`` then reports Cohen's kappa between
the judge and these grades.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from datetime import date
from pathlib import Path

from dataset.synthesize import append_jsonl, read_jsonl
from rag_eval import answer_sha, latest, load_golden

KEYS = "[p]ass  [f]ail  [s]kip  [q]uit > "
GUIDE = (
    "Pass if a reader following the answer gets everything the question asked for right "
    "(different wording or a missing minor detail is fine); fail otherwise."
)


def judged_answers(run_dir: Path, golden: Path) -> list[dict]:
    """The run's answers that carry a correctness verdict, with what a grader needs."""
    questions = {q.id: q for q in load_golden(golden)}
    items = []
    for (qid, mode), r in sorted(latest(run_dir / "answers.jsonl").items()):
        if "error" in r or not r.get("correctness") or qid not in questions:
            continue
        q = questions[qid]
        items.append(
            {
                "id": qid,
                "mode": mode,
                "category": q.category,
                "question": q.question,
                "reference": q.reference_answer,
                "answer": r["answer"],
                "answer_sha": answer_sha(r["answer"]),
            }
        )
    return items


def allocate(sizes: dict, n: int) -> dict:
    """Split ``n`` across strata in proportion to their sizes (largest remainder)."""
    total = sum(sizes.values())
    if n >= total:
        return dict(sizes)
    exact = {k: n * size / total for k, size in sizes.items()}
    quotas = {k: math.floor(v) for k, v in exact.items()}
    by_remainder = sorted(exact, key=lambda k: (-(exact[k] - quotas[k]), str(k)))
    for k in by_remainder[: n - sum(quotas.values())]:
        quotas[k] += 1
    return quotas


def sample(items: Sequence[dict], n: int, seed: int) -> list[dict]:
    """A stratified sample by mode and category, in an order that mixes the strata."""
    rng = random.Random(seed)
    strata: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for item in sorted(items, key=lambda i: (i["mode"], i["id"])):
        strata[(item["mode"], item["category"])].append(item)
    quotas = allocate({k: len(v) for k, v in strata.items()}, n)
    picked = [item for k in sorted(strata) for item in rng.sample(strata[k], quotas[k])]
    rng.shuffle(picked)
    return picked


def grade(
    items: Sequence[dict],
    grades_path: Path,
    run_id: str,
    *,
    ask: Callable[[str], str] = input,
    out: Callable[[str], None] = print,
) -> Counter[str]:
    graded = {(g["id"], g["mode"], g["answer_sha"]) for g in read_jsonl(grades_path)}
    todo = [i for i in items if (i["id"], i["mode"], i["answer_sha"]) not in graded]
    tally: Counter[str] = Counter()
    out(f"{len(items) - len(todo)} of {len(items)} already graded. {GUIDE}")
    for n, item in enumerate(todo, start=1):
        out("")
        out(f"== {n}/{len(todo)}  {item['id']}")
        out(f"Q: {item['question']}")
        out(f"Reference: {item['reference']}")
        out("Answer:")
        out(item["answer"])
        while True:
            key = ask(KEYS).strip().lower()
            if key in ("p", "f"):
                note = ask("note (optional) > ").strip()
                append_jsonl(
                    grades_path,
                    {
                        "id": item["id"],
                        "mode": item["mode"],
                        "answer_sha": item["answer_sha"],
                        "pass": key == "p",
                        "note": note,
                        "run_id": run_id,
                        "graded_on": date.today().isoformat(),
                    },
                )
                tally["passed" if key == "p" else "failed"] += 1
                break
            if key == "s":
                tally["skipped"] += 1
                break
            if key == "q":
                out(_summary(tally))
                return tally
            out("unknown key")
    out(_summary(tally))
    return tally


def _summary(tally: Counter[str]) -> str:
    return f"{tally['passed']} passed · {tally['failed']} failed · {tally['skipped']} skipped"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m dataset.grade", description="Grade judged answers by hand, blind."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--golden", type=Path, default=Path("golden/golden_set.jsonl"))
    parser.add_argument("--grades", type=Path, default=Path("golden/human_grades.jsonl"))
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    items = judged_answers(args.runs / args.run_id, args.golden)
    if not items:
        parser.error(f"run {args.run_id} has no judged answers; run its answer stage first")
    grade(sample(items, args.n, args.seed), args.grades, args.run_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
