"""The judge: grades answers and agent outputs with a model from another provider.

Each kind of verdict has a system and a user template in ``judge_prompts``, filled in
with ``str.format`` (values go in verbatim, braces and all). The verdict's Pydantic
model travels as the structured-output schema through ``llm.LLM``, so the prompts never
describe JSON and every reply is validated into its model.

Verdicts are cached on disk, keyed by a hash of everything that decides them: provider,
model, prompt version, and the rendered system and user text. A cached verdict costs
nothing and needs no budget. A new one is priced and recorded in the ``Budget``, which
refuses the call before it's sent if it could pass the cap. A reply that isn't a valid
verdict is still billed but never cached.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any, get_args

from pydantic import BaseModel, ValidationError

from budget import Budget
from llm import LLM, Provider
from schemas import Context, CorrectnessVerdict, FaithfulnessVerdict, RubricCriterion, TaskVerdict

DEFAULT_PROVIDER: Provider = "anthropic"
DEFAULT_MODEL = "claude-haiku-4-5"
CACHE_DIR = Path(__file__).parent / ".cache" / "judge"
# Room for a faithfulness verdict on a long answer (one entry per claim, each with its
# reasoning). The budget reserves all of it before every call.
MAX_TOKENS = 2048


def source_label(metadata: dict) -> str:
    """``source § heading (p.N)``, as the RAG service labels a context for its generator."""
    label = str(metadata.get("source_file", "unknown"))
    if heading := metadata.get("section_heading"):
        label += f" § {heading}"
    page = metadata.get("page", -1)
    if isinstance(page, int) and page > 0:
        label += f" (p.{page})"
    return label


def render_contexts(contexts: Sequence[Context]) -> str:
    """Number and label the contexts exactly as the generator saw them, so an answer's
    ``[n]`` points at the same passage for the judge."""
    return "\n\n".join(
        f"[{i}] ({source_label(c.metadata)})\n{c.text}" for i, c in enumerate(contexts, start=1)
    )


def render_criteria(criteria: Sequence[RubricCriterion]) -> str:
    """One ``- id: description`` line per criterion. Whether a criterion is critical is a
    scoring rule, so the judge doesn't see it and grades each one on its own."""
    return "\n".join(f"- {c.id}: {c.description}" for c in criteria)


def cache_key(provider: str, model: str, prompt_version: str, system: str, user: str) -> str:
    payload = json.dumps(
        {
            "provider": provider,
            "model": model,
            "prompt_version": prompt_version,
            "system": system,
            "user": user,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class Judge:
    """Correctness, faithfulness and task-rubric verdicts from one provider and model.

    ``prompts`` is any object with ``PROMPT_VERSION`` and the ``*_SYSTEM``/``*_USER``
    templates (tests pass their own); it defaults to the ``judge_prompts`` module.
    ``client`` is passed to ``LLM`` (tests pass an SDK client on a mock transport).
    """

    def __init__(
        self,
        provider: Provider,
        model: str,
        *,
        cache_dir: Path,
        budget: Budget,
        prompts: Any = None,
        client: Any = None,
    ) -> None:
        if prompts is None:
            import judge_prompts as prompts
        self.provider = provider
        self.model = model
        self.prompts = prompts
        self.prompt_version: str = prompts.PROMPT_VERSION
        self.cache_dir = cache_dir
        self.cache_hits = 0
        self._llm = LLM(provider, model, budget=budget, max_tokens=MAX_TOKENS, client=client)

    @classmethod
    def from_env(cls, *, budget: Budget, cache_dir: Path = CACHE_DIR, **kwargs: Any) -> Judge:
        """The judge named by ``JUDGE_PROVIDER`` and ``JUDGE_MODEL`` (Anthropic's
        claude-haiku-4-5 when unset)."""
        provider = os.environ.get("JUDGE_PROVIDER") or DEFAULT_PROVIDER
        if provider not in get_args(Provider):
            raise ValueError(
                f"JUDGE_PROVIDER must be one of {get_args(Provider)}, got {provider!r}"
            )
        model = os.environ.get("JUDGE_MODEL") or DEFAULT_MODEL
        return cls(provider, model, cache_dir=cache_dir, budget=budget, **kwargs)

    @property
    def budget(self) -> Budget:
        return self._llm.budget

    def correctness(self, question: str, reference: str, answer: str) -> CorrectnessVerdict:
        return self._verdict(
            "CORRECTNESS", CorrectnessVerdict, question=question, reference=reference, answer=answer
        )

    def faithfulness(self, contexts: Sequence[Context], answer: str) -> FaithfulnessVerdict:
        if not contexts:
            raise ValueError("faithfulness needs the contexts the answer was written from")
        return self._verdict(
            "FAITHFULNESS", FaithfulnessVerdict, contexts=render_contexts(contexts), answer=answer
        )

    def task(
        self,
        request: str,
        final_output: str,
        subtask_outputs: str,
        criteria: Sequence[RubricCriterion],
    ) -> TaskVerdict:
        if not criteria:
            raise ValueError("a task verdict needs at least one rubric criterion")
        return self._verdict(
            "TASK",
            TaskVerdict,
            request=request,
            final_output=final_output,
            subtask_outputs=subtask_outputs,
            criteria=render_criteria(criteria),
        )

    def _verdict[V: BaseModel](self, kind: str, schema: type[V], **values: str) -> V:
        system = getattr(self.prompts, f"{kind}_SYSTEM").format()
        user = getattr(self.prompts, f"{kind}_USER").format(**values)
        key = cache_key(self.provider, self.model, self.prompt_version, system, user)
        path = self.cache_dir / f"{key}.json"
        if (cached := self._load(path, schema)) is not None:
            self.cache_hits += 1
            return cached
        verdict, usage = self._llm.complete(system, user, schema, name="verdict")
        entry = {
            "kind": kind.lower(),
            "provider": self.provider,
            "model": self.model,
            "prompt_version": self.prompt_version,
            "system": system,
            "user": user,
            "verdict": verdict.model_dump(),
            "usage": {
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "cost_usd": usage.cost_usd,
            },
        }
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(entry, indent=1, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)  # a run killed mid-write never leaves a half entry behind
        return verdict

    @staticmethod
    def _load[V: BaseModel](path: Path, schema: type[V]) -> V | None:
        """The cached verdict, or None if there is none or it no longer fits ``schema``
        (then it is asked again and overwritten)."""
        if not path.is_file():
            return None
        try:
            return schema.model_validate(json.loads(path.read_text(encoding="utf-8"))["verdict"])
        except (json.JSONDecodeError, KeyError, TypeError, ValidationError):
            return None
