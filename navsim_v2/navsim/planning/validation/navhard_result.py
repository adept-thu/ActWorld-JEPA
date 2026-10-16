"""Fail-closed NAVHard result parsing for Drive/ActWorld comparisons.

The NAVSIM two-stage CSV contains three aggregate rows.  Drive-JEPA's
published NAVHard comparison value corresponds to the Stage-2 EPDMS row,
while the devkit also emits the pseudo-closed-loop Stage-1 x Stage-2 product.
Both are retained, but only the explicitly named Stage-2 row is compared with
the published 52.88 baseline and the preregistered 53.88 target.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping


STAGE_ONE_TOKEN = "extended_pdm_score_stage_one"
PUBLISHED_COMPARISON_TOKEN = "extended_pdm_score_stage_two"
COMBINED_TOKEN = "extended_pdm_score_combined"
SUMMARY_TOKENS = frozenset(
    {STAGE_ONE_TOKEN, PUBLISHED_COMPARISON_TOKEN, COMBINED_TOKEN}
)


class NavHardResultError(ValueError):
    """Raised when a NAVHard result cannot support a benchmark claim."""


@dataclass(frozen=True)
class NavHardResult:
    csv_path: str
    csv_sha256: str
    scenario_count: int
    failed_count: int
    ordered_token_sha256: str
    stage_one_epdms: float
    published_comparison_stage_two_epdms: float
    combined_pseudo_closed_loop_epdms: float


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_bool(value: str, *, token: str) -> bool:
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise NavHardResultError(f"invalid valid={value!r} for token {token!r}")


def _parse_score(value: str, *, token: str) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError) as error:
        raise NavHardResultError(
            f"invalid score={value!r} for token {token!r}"
        ) from error
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise NavHardResultError(
            f"score for token {token!r} must be finite and in [0, 1]"
        )
    return score


def _read_rows(path: Path) -> Iterable[Mapping[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"token", "valid", "score"}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise NavHardResultError(
                f"NAVHard CSV is missing required columns: {sorted(missing)}"
            )
        yield from reader


def load_navhard_result(
    path: Path, *, expected_scenarios: int = 5912
) -> NavHardResult:
    """Validate complete coverage and extract unambiguous aggregate scores."""

    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if (
        isinstance(expected_scenarios, bool)
        or not isinstance(expected_scenarios, int)
        or expected_scenarios <= 0
    ):
        raise ValueError("expected_scenarios must be a positive integer")

    scenarios = []
    summaries: Dict[str, float] = {}
    failed = []
    for row in _read_rows(path):
        token = str(row["token"])
        if not token:
            raise NavHardResultError("NAVHard CSV contains an empty token")
        valid = _parse_bool(str(row["valid"]), token=token)
        score = _parse_score(str(row["score"]), token=token)
        if token in SUMMARY_TOKENS:
            if token in summaries:
                raise NavHardResultError(f"duplicate NAVHard summary row {token!r}")
            if not valid:
                raise NavHardResultError(f"NAVHard summary row {token!r} is invalid")
            summaries[token] = score
        else:
            scenarios.append(token)
            if not valid:
                failed.append(token)

    missing_summaries = SUMMARY_TOKENS.difference(summaries)
    if missing_summaries:
        raise NavHardResultError(
            f"NAVHard CSV is missing summary rows: {sorted(missing_summaries)}"
        )
    if len(scenarios) != expected_scenarios:
        raise NavHardResultError(
            "NAVHard scenario coverage mismatch: "
            f"expected {expected_scenarios}, found {len(scenarios)}"
        )
    if len(set(scenarios)) != len(scenarios):
        raise NavHardResultError("NAVHard CSV contains duplicate scenario tokens")
    if failed:
        raise NavHardResultError(
            f"NAVHard CSV contains {len(failed)} failed scenarios"
        )

    ordered_digest = hashlib.sha256()
    for token in scenarios:
        ordered_digest.update(token.encode("utf-8"))
        ordered_digest.update(b"\n")

    return NavHardResult(
        csv_path=str(path),
        csv_sha256=_sha256(path),
        scenario_count=len(scenarios),
        failed_count=len(failed),
        ordered_token_sha256=ordered_digest.hexdigest(),
        stage_one_epdms=summaries[STAGE_ONE_TOKEN],
        published_comparison_stage_two_epdms=summaries[
            PUBLISHED_COMPARISON_TOKEN
        ],
        combined_pseudo_closed_loop_epdms=summaries[COMBINED_TOKEN],
    )


def write_result_record(result: NavHardResult, path: Path) -> None:
    """Atomically write a normalized evidence record."""

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(asdict(result), indent=2, sort_keys=True) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent, prefix=f".{destination.name}.", suffix=".partial"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument("--expected-scenarios", type=int, default=5912)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = load_navhard_result(
        args.csv, expected_scenarios=args.expected_scenarios
    )
    if args.output is not None:
        write_result_record(result, args.output)
    print(json.dumps(asdict(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
