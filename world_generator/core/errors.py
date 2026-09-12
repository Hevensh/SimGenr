"""Failure categories describe the attempted model, not feasibility of reality."""
from __future__ import annotations


class PhysicalInfeasibilityError(RuntimeError):
    """A solver explicitly certified the configured constraints infeasible."""

    def __init__(self, message: str, *, stage: str | None = None) -> None:
        super().__init__(message)
        self.stage = stage


class SolverError(RuntimeError):
    """Dependency, numerical or termination failure; infeasibility is unproven."""

    def __init__(self, message: str, *, stage: str | None = None) -> None:
        super().__init__(message)
        self.stage = stage


def generation_failure_category(error: BaseException, *, stage: str | None = None) -> str:
    if isinstance(error, PhysicalInfeasibilityError):
        return "PHYSICAL_INFEASIBILITY"
    # Existing public configuration and array contracts use ValueError. An
    # unknown dataclass configuration property uses TypeError; arbitrary
    # TypeErrors later in generation remain implementation failures.
    if isinstance(error, ValueError) or (stage == "configuration" and isinstance(error, TypeError)):
        return "INVALID_INPUT"
    if isinstance(error, SystemExit) and error.code == 2:
        return "INVALID_INPUT"
    return "GENERATION_ERROR"
