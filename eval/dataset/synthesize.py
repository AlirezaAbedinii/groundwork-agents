"""Draft synthetic golden candidates, one chunk each, for review by hand.

    python -m dataset.synthesize --n 40 --model gpt-4o-mini --rag http://localhost:8000 \\
        --max-cost-usd 0.10

Chunks come from the collection the RAG API serves: each top-level source (uv,
pydantic, fastapi) gets a share of ``--n`` in proportion to its chunks, and a file
gives at most one chunk. A chunk is skipped if it's under 300 characters, more than
60 % fenced code, or already holds a golden quote; files that already have a
candidate, a rejected draft or a synthetic row are skipped too. The model drafts a
question, a reference answer and a quote from the chunk alone. Automatic checks
flag a draft and never drop it: the quote isn't in the chunk or has the wrong
length, the question shares a 6-word run with the chunk (leakage), or its words
overlap an existing question's (token Jaccard > 0.6).

Drafts are appended to ``golden/candidates_unverified.jsonl`` (not committed) as they
are paid for; ``python -m dataset.review`` turns them into golden rows or rejects.
The key comes from ``OPENAI_API_KEY`` or ``ANTHROPIC_API_KEY``, in the environment
or in ``eval/.env``.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import httpx
from pydantic import BaseModel

from budget import Budget, BudgetExceeded
from llm import LLM, LLMError, load_env
from textnorm import normalize_ws

MIN_CHARS = 300
MAX_CODE_SHARE = 0.6
LEAK_RUN = 6
MAX_JACCARD = 0.6

SYSTEM = """\
You write test questions for a question-answering system over the documentation of \
uv, Pydantic and FastAPI.

You get one chunk of a documentation page. Write:
- question: something a developer using these tools might really ask, answerable \
from this chunk alone. Use your own words: don't copy phrases from the chunk, and \
don't mention "the chunk", "the text" or "the documentation".
- reference_answer: a correct and complete answer in one or two sentences, using \
only facts stated in the chunk.
- quote: the 12 to 120 characters of the chunk that state the key fact, copied \
exactly (same spelling, punctuation and markdown), from running text rather than a \
heading or a code block.

Pick a fact a user would care about, not trivia such as link targets, the names \
used in an example or the order of a list."""

USER = """\
Page: {source}
Section: {heading}

Chunk:
<<<
{text}
>>>"""


class Draft(BaseModel):
    question: str
    reference_answer: str
    quote: str


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    source: str
    heading: str | None
    text: str


# --- selection ----------------------------------------------------------------------


def code_share(text: str) -> float:
    """The share of characters inside fenced code. A chunk can start or end inside a
    fence, so with an odd number of fence lines both readings are tried and the larger
    share is kept."""
    lines = text.splitlines(keepends=True)
    fences = [i for i, line in enumerate(lines) if line.lstrip().startswith(("```", "~~~"))]

    def share(start_inside: bool) -> float:
        inside, code = start_inside, 0
        for i, line in enumerate(lines):
            if i in fences:
                code += len(line)
                inside = not inside
            elif inside:
                code += len(line)
        return code / max(len(text), 1)

    return max(share(False), share(True)) if len(fences) % 2 else share(False)


def allocate(chunks_per_source: dict[str, int], n: int) -> dict[str, int]:
    """Split ``n`` across sources in proportion to their chunks (largest remainder)."""
    total = sum(chunks_per_source.values())
    exact = {s: n * c / total for s, c in chunks_per_source.items()}
    share = {s: int(x) for s, x in exact.items()}
    by_remainder = sorted(exact, key=lambda s: (exact[s] - share[s], s), reverse=True)
    for s in by_remainder[: n - sum(share.values())]:
        share[s] += 1
    return share


def eligible(chunk: Chunk, golden_quotes: set[str]) -> bool:
    if len(chunk.text) < MIN_CHARS or code_share(chunk.text) > MAX_CODE_SHARE:
        return False
    text = normalize_ws(chunk.text)
    return not any(quote in text for quote in golden_quotes)


def select_chunks(
    rag: httpx.Client, n: int, rng: random.Random, skip_files: set[str], golden_quotes: set[str]
) -> list[Chunk]:
    documents = rag.get("/v1/documents").raise_for_status().json()["documents"]
    files: dict[str, list[str]] = defaultdict(list)
    chunks_per_source: dict[str, int] = defaultdict(int)
    for doc in documents:
        top = doc["source_file"].split("/")[0]
        chunks_per_source[top] += doc["chunks"]
        if doc["source_file"] not in skip_files:
            files[top].append(doc["source_file"])

    selected: list[Chunk] = []
    for top, wanted in sorted(allocate(chunks_per_source, n).items()):
        candidates = sorted(files[top])
        rng.shuffle(candidates)
        taken = 0
        for source in candidates:
            if taken == wanted:
                break
            body = rag.get("/v1/chunks", params={"source_file": source}).raise_for_status().json()
            chunks = [
                Chunk(c["chunk_id"], source, c["section_heading"], c["text"])
                for c in body["chunks"]
            ]
            rng.shuffle(chunks)
            pick = next((c for c in chunks if eligible(c, golden_quotes)), None)
            if pick:
                selected.append(pick)
                taken += 1
        if taken < wanted:
            print(f"note: {top} has only {taken} eligible files of the {wanted} wanted")
    return selected


# --- checks -------------------------------------------------------------------------


def words(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def shared_run(question: str, chunk_text: str, length: int = LEAK_RUN) -> str | None:
    """The first run of ``length`` words the question shares with the chunk, if any."""
    q, c = words(question), words(chunk_text)
    runs = {tuple(c[i : i + length]) for i in range(len(c) - length + 1)}
    for i in range(len(q) - length + 1):
        if tuple(q[i : i + length]) in runs:
            return " ".join(q[i : i + length])
    return None


def jaccard(a: str, b: str) -> float:
    sa, sb = set(words(a)), set(words(b))
    return len(sa & sb) / len(sa | sb) if sa | sb else 0.0


def flag_draft(question: str, quote: str, chunk_text: str, existing: dict[str, str]) -> list[str]:
    """Reasons to look twice at a draft. ``existing`` maps ids to questions it mustn't echo."""
    flags = []
    if normalize_ws(quote) not in normalize_ws(chunk_text):
        flags.append("quote not in chunk")
    length = len(normalize_ws(quote))
    if not 12 <= length <= 120:
        flags.append(f"quote is {length} characters (12-120 allowed)")
    run = shared_run(question, chunk_text)
    if run:
        flags.append(f"leakage: question shares {run!r} with the chunk")
    for qid, other in existing.items():
        score = jaccard(question, other)
        if score > MAX_JACCARD:
            flags.append(f"near-duplicate of {qid} (token Jaccard {score:.2f})")
    return flags


# --- files --------------------------------------------------------------------------


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def append_jsonl(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def next_candidate_id(*groups: list[dict]) -> int:
    ids = [int(r["candidate_id"][1:]) for g in groups for r in g if "candidate_id" in r]
    return max(ids, default=0) + 1


# --- command line -------------------------------------------------------------------


def run(
    *,
    n: int,
    llm: LLM,
    rag: httpx.Client,
    golden: Path,
    out: Path,
    rejects: Path,
    seed: int = 0,
) -> int:
    rows = read_jsonl(golden)
    pending, rejected = read_jsonl(out), read_jsonl(rejects)
    golden_quotes = {
        normalize_ws(q["text"]) for r in rows for item in r["evidence"] for q in item["quotes"]
    }
    skip_files = {c["source"] for c in pending + rejected} | {
        q["source"]
        for r in rows
        if r["origin"] == "synthetic"
        for item in r["evidence"]
        for q in item["quotes"]
    }
    existing = {r["id"]: r["question"] for r in rows} | {
        c["candidate_id"]: c["question"] for c in pending
    }

    chunks = select_chunks(rag, n, random.Random(seed), skip_files, golden_quotes)
    print(f"selected {len(chunks)} chunks; drafting with {llm.model}")
    cid, drafted, failed = next_candidate_id(pending, rejected), 0, 0
    try:
        for chunk in chunks:
            user = USER.format(source=chunk.source, heading=chunk.heading or "-", text=chunk.text)
            try:
                draft, usage = llm.complete(SYSTEM, user, Draft, name="draft")
            except LLMError as exc:
                failed += 1
                print(f"  {chunk.source}: no draft ({exc})")
                continue
            candidate_id = f"c{cid:03}"
            flags = flag_draft(draft.question, draft.quote, chunk.text, existing)
            append_jsonl(
                out,
                {
                    "candidate_id": candidate_id,
                    "source": chunk.source,
                    "heading": chunk.heading,
                    "chunk_id": chunk.chunk_id,
                    "chunk_text": chunk.text,
                    **draft.model_dump(),
                    "flags": flags,
                    "model": llm.model,
                    "cost_usd": round(usage.cost_usd, 8),
                    "drafted_on": date.today().isoformat(),
                },
            )
            existing[candidate_id] = draft.question
            cid += 1
            drafted += 1
            print(f"  {candidate_id} {chunk.source}" + (f"  [{'; '.join(flags)}]" if flags else ""))
    except BudgetExceeded as exc:
        print(f"stopped: {exc}")
        print(f"{drafted} drafts written to {out} · {llm.budget.summary()}")
        return 1
    print(f"{drafted} drafts written to {out} · {failed} failed · {llm.budget.summary()}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m dataset.synthesize", description="Draft synthetic golden candidates."
    )
    parser.add_argument("--n", type=int, required=True, help="how many chunks to draft from")
    parser.add_argument("--model", required=True, help="e.g. gpt-4o-mini")
    parser.add_argument("--provider", choices=["openai", "anthropic"], default="openai")
    parser.add_argument(
        "--rag", required=True, metavar="URL", help="the RAG API serving the corpus"
    )
    parser.add_argument("--max-cost-usd", type=float, required=True, help="spending cap")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--golden", type=Path, default=Path("golden/golden_set.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("golden/candidates_unverified.jsonl"))
    parser.add_argument("--rejects", type=Path, default=Path("golden/synthetic_rejects.jsonl"))
    args = parser.parse_args(argv)

    load_env(Path(".env"))
    budget = Budget(args.max_cost_usd)
    llm = LLM(args.provider, args.model, budget=budget, max_tokens=400)
    with httpx.Client(base_url=args.rag, timeout=30) as rag:
        return run(
            n=args.n,
            llm=llm,
            rag=rag,
            golden=args.golden,
            out=args.out,
            rejects=args.rejects,
            seed=args.seed,
        )


if __name__ == "__main__":
    sys.exit(main())
