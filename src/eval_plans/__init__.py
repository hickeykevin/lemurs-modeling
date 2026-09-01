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
from src.eval_plans.cv_aggregation import aggregate_cv_metrics
from src.eval_plans.fold_identity import fold_identity, user_hash
from src.eval_plans.pooled_metrics import compute_pooled_metrics, fold_breakdown_table
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
    "aggregate_cv_metrics",
    "apply_overrides",
    "compute_pooled_metrics",
    "default_step_for",
    "fold_breakdown_table",
    "fold_identity",
    "run_plan",
    "run_unit",
    "user_hash",
]

