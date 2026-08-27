"""Evaluation strategies as data: which units of work exist, and how their
results combine. See ``src/eval_plans/base.py`` for the protocol and its known
boundary, and ``docs/eval_schemes.md`` for which config selects which strategy.
"""

from src.eval_plans.base import (
    CohortCache,
    EvalPlan,
    RunContext,
    RunUnit,
    UnitResult,
    default_step_for,
)
from src.eval_plans.runner import LOG_STEP_OFFSET, apply_overrides, run_plan, run_unit
from src.eval_plans.single import SingleSplitPlan
from src.eval_plans.user_cv import UserCVPlan
from src.eval_plans.walk_forward import WalkForwardPlan

__all__ = [
    "CohortCache",
    "EvalPlan",
    "LOG_STEP_OFFSET",
    "RunContext",
    "RunUnit",
    "SingleSplitPlan",
    "UnitResult",
    "UserCVPlan",
    "WalkForwardPlan",
    "apply_overrides",
    "default_step_for",
    "run_plan",
    "run_unit",
]
