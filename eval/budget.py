"""A spending cap for paid commands.

Every command that calls a provider takes ``--max-cost-usd`` (no default) and builds
one ``Budget`` from it. Before each call the caller passes that call's worst case
(its input estimate plus the whole output allowance); the budget refuses the call if
it could take spending past the cap, so the cap holds even when every call runs long.
"""

from __future__ import annotations

from dataclasses import dataclass


class BudgetExceeded(RuntimeError):
    """The next call could take spending past ``--max-cost-usd``."""


@dataclass
class Budget:
    max_cost_usd: float
    spent_usd: float = 0.0
    calls: int = 0

    def __post_init__(self) -> None:
        if self.max_cost_usd < 0:
            raise ValueError("max_cost_usd must be >= 0")

    @property
    def remaining_usd(self) -> float:
        return self.max_cost_usd - self.spent_usd

    def check(self, worst_case_usd: float) -> None:
        """Raise ``BudgetExceeded`` if a call costing up to ``worst_case_usd`` could pass it."""
        if self.spent_usd + worst_case_usd > self.max_cost_usd:
            raise BudgetExceeded(
                f"the next call could cost up to ${worst_case_usd:.6f}; "
                f"${self.spent_usd:.6f} of the ${self.max_cost_usd:.2f} cap is spent"
            )

    def record(self, cost_usd: float) -> None:
        self.spent_usd += cost_usd
        self.calls += 1

    def summary(self) -> str:
        calls = f"{self.calls} call{'s' * (self.calls != 1)}"
        return f"{calls} · ${self.spent_usd:.6f} of a ${self.max_cost_usd:.2f} cap"
