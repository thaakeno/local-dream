#!/usr/bin/env python3
"""MOSS-only deterministic regression for the SM8850 HTP partition plan.

Checks that the staged pinned source actually contains the scoped headroom
adjustment, then replays the observed alpha.72 Qwen weight split with synthetic
per-block weights. This is NOT an on-device graph/quality benchmark.
"""
from __future__ import annotations

import pathlib
import sys

RUNTIME_BUDGET_MIB = 2.60 * 1024
WORKSPACE_RESERVE_MIB = 320
OBSERVED_WORKSPACE_MIB = 2855.86 - 2580.0
# Reproduces the observed 17 + 17 grouping, preserving the exact 2580/2031
# MiB totals (not a claim that all real Qwen blocks have identical sizes).
QWEN_SEGMENTS_MIB = [2580.0 / 17] * 17 + [2031.3 / 17] * 17


def greedy_first_split(weights: list[float], cap: float) -> tuple[float, float]:
    first = 0.0
    for weight in weights:
        if first + weight > cap:
            break
        first += weight
    return first, sum(weights) - first


def test_alpha72_original_failure() -> None:
    first, second = greedy_first_split(QWEN_SEGMENTS_MIB, RUNTIME_BUDGET_MIB)
    assert abs(first - 2580.0) < 0.01
    assert first + OBSERVED_WORKSPACE_MIB > RUNTIME_BUDGET_MIB
    assert second < RUNTIME_BUDGET_MIB


def test_reserved_partition_budget() -> None:
    planner_cap = RUNTIME_BUDGET_MIB - WORKSPACE_RESERVE_MIB
    first, second = greedy_first_split(QWEN_SEGMENTS_MIB, planner_cap)
    assert first < 2580
    assert first <= planner_cap
    assert second <= planner_cap, (first, second, planner_cap)
    assert max(first, second) + OBSERVED_WORKSPACE_MIB <= RUNTIME_BUDGET_MIB


def test_scoped_upstream_patch(source: pathlib.Path) -> None:
    data = source.read_text()
    assert "htp_workspace_reserve_bytes = 320ll * 1024 * 1024" in data
    assert "backends.size() > 1" in data
    assert 'rfind("HTP", 0) == 0' in data
    assert "limit - htp_workspace_reserve_bytes" in data
    assert "MOSS HTP split planner:" in data


if __name__ == "__main__":
    test_alpha72_original_failure()
    test_reserved_partition_budget()
    if len(sys.argv) != 2:
        raise SystemExit("Pass staged source path to verify the scoped patch")
    test_scoped_upstream_patch(pathlib.Path(sys.argv[1]))
    print("MOSS SM8850 partition regression: PASS (synthetic); staged patch verified")
