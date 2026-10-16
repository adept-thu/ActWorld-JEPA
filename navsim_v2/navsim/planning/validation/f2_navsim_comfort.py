"""Strict NAVSIM comfort adapter for authoritative F2 causal validation.

Drive-JEPA calibrates a candidate at the current token by comparing its
``simulation_v2`` state with the independently selected standard-simulator
carry from the preceding token.  This module freezes that small but critical
piece of runtime semantics without importing the heavy NAVSIM stack until an
official adapter is explicitly requested.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from navsim.planning.validation.f2_causal_validator import (
    MAX_CONTINUITY_DELTA_US,
)


STATE_TIME_STEPS = 41
STATE_DIMENSION = 11
COMFORT_INTERVAL_US = 100_000
COMFORT_BRANCHES = ("critic", "base")
_OFFICIAL_COMFORT_SEAL = object()


class F2NavsimComfortError(RuntimeError):
    """Raised when the frozen two-frame comfort contract is violated."""


ComfortKernel = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]


def _state_array(value: Any, name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise F2NavsimComfortError(f"{name} must be a numpy.ndarray")
    if value.dtype != np.dtype("<f8") or value.dtype.str != "<f8":
        raise F2NavsimComfortError(f"{name} must have exact little-endian float64 dtype")
    if value.shape != (STATE_TIME_STEPS, STATE_DIMENSION):
        raise F2NavsimComfortError(
            f"{name} must have shape ({STATE_TIME_STEPS}, {STATE_DIMENSION})"
        )
    if not value.flags.c_contiguous:
        raise F2NavsimComfortError(f"{name} must be C-contiguous")
    if not np.isfinite(value).all():
        raise F2NavsimComfortError(f"{name} contains NaN or infinity")
    return value


class DriveJepaExtendedComfort:
    """Callable implementing the exact Drive-JEPA two-frame overlap rule.

    The overlap index is the floor of the positive observation interval at the
    frozen 0.1-second sampling rate.  Authoritative inputs are the 41x11
    float64 states stored in the sealed artifact.  The official NAVSIM kernel
    returns one boolean, which is exposed to the causal state machine as the
    exact scalar 0.0 or 1.0.
    """

    def __init__(
        self, kernel: ComfortKernel, *, _official_seal: object | None = None
    ) -> None:
        if not callable(kernel):
            raise F2NavsimComfortError("comfort kernel must be callable")
        self._kernel = kernel
        self._official = _official_seal is _OFFICIAL_COMFORT_SEAL

    @property
    def is_official(self) -> bool:
        """True only for the adapter constructed from NAVSIM's live kernel."""

        return self._official

    @classmethod
    def official(cls) -> "DriveJepaExtendedComfort":
        """Load the same NAVSIM comfort kernel imported by Drive-JEPA."""

        from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_comfort_metrics import (
            ego_is_two_frame_extended_comfort,
        )

        return cls(
            ego_is_two_frame_extended_comfort,
            _official_seal=_OFFICIAL_COMFORT_SEAL,
        )

    def __call__(
        self,
        branch_name: str,
        past_carry_state: Any,
        candidate_calibration_state: Any,
        delta_time_us: int,
    ) -> float:
        if branch_name not in COMFORT_BRANCHES:
            raise F2NavsimComfortError(
                f"comfort branch must be one of {COMFORT_BRANCHES!r}"
            )
        if (
            isinstance(delta_time_us, bool)
            or not isinstance(delta_time_us, int)
            or not (0 < delta_time_us <= MAX_CONTINUITY_DELTA_US)
        ):
            raise F2NavsimComfortError(
                "comfort delta_time_us must satisfy the frozen continuity boundary"
            )
        overlap_start = delta_time_us // COMFORT_INTERVAL_US
        if not (1 <= overlap_start < STATE_TIME_STEPS):
            raise F2NavsimComfortError(
                "continuous comfort interval does not contain a complete 0.1-second step"
            )

        past = _state_array(past_carry_state, "past carry state")
        current = _state_array(
            candidate_calibration_state, "candidate calibration state"
        )
        past_overlap = past[overlap_start:]
        current_overlap = current[:-overlap_start]
        if past_overlap.shape != current_overlap.shape or not past_overlap.size:
            raise F2NavsimComfortError("two-frame comfort overlap is empty or mismatched")
        time_point_s = (
            np.arange(past_overlap.shape[0], dtype=np.float64)
            * (COMFORT_INTERVAL_US / 1_000_000.0)
        )
        try:
            raw = self._kernel(
                past_overlap[None, ...],
                current_overlap[None, ...],
                time_point_s,
            )
        except Exception as error:
            raise F2NavsimComfortError(
                f"NAVSIM two-frame comfort kernel failed: {type(error).__name__}: {error}"
            ) from error
        result = np.asarray(raw)
        if result.shape != (1,) or result.dtype != np.dtype(np.bool_):
            raise F2NavsimComfortError(
                "NAVSIM two-frame comfort kernel must return one boolean"
            )
        return float(bool(result[0]))


__all__ = (
    "COMFORT_BRANCHES",
    "COMFORT_INTERVAL_US",
    "DriveJepaExtendedComfort",
    "F2NavsimComfortError",
    "STATE_DIMENSION",
    "STATE_TIME_STEPS",
)
