#!/usr/bin/env python3
"""MOSS-only deterministic regression for the SM8850 HTP partition plan.

Checks that the staged pinned source actually contains the scoped headroom
adjustment, then replays the observed alpha.72 Qwen weight split with synthetic
per-block weights. This is NOT an on-device graph/quality benchmark.
"""
from __future__ import annotations

import pathlib
import sys

PREVIOUS_RUNTIME_BUDGET_MIB = 2.60 * 1024
RUNTIME_BUDGET_MIB = 2.75 * 1024
WORKSPACE_RESERVE_MIB = 320
HEXAGON_MAPPING_WINDOW_MIB = 3337998336 / (1024 * 1024)
TURBO_DIFFUSION_ESTIMATE_MIB = 4.59 * 1024
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
    first, second = greedy_first_split(QWEN_SEGMENTS_MIB, PREVIOUS_RUNTIME_BUDGET_MIB)
    assert abs(first - 2580.0) < 0.01
    assert first + OBSERVED_WORKSPACE_MIB > PREVIOUS_RUNTIME_BUDGET_MIB
    assert second < PREVIOUS_RUNTIME_BUDGET_MIB


def test_reserved_partition_budget() -> None:
    planner_cap = RUNTIME_BUDGET_MIB - WORKSPACE_RESERVE_MIB
    first, second = greedy_first_split(QWEN_SEGMENTS_MIB, planner_cap)
    assert first < 2580
    assert first <= planner_cap
    assert second <= planner_cap, (first, second, planner_cap)
    assert max(first, second) + OBSERVED_WORKSPACE_MIB <= RUNTIME_BUDGET_MIB


def test_turbo_diffusion_capacity() -> None:
    # Previous cap worked for the 4.61 GiB Qwen encoder, but the separately
    # loaded, LoRA-augmented diffusion graph needs ~4.59 GiB plus placement
    # granularity. This synthetic test checks *planner capacity*, not runtime.
    old_total_weight_capacity = 2 * (PREVIOUS_RUNTIME_BUDGET_MIB - WORKSPACE_RESERVE_MIB)
    new_per_session = RUNTIME_BUDGET_MIB - WORKSPACE_RESERVE_MIB
    assert old_total_weight_capacity < TURBO_DIFFUSION_ESTIMATE_MIB
    assert 2 * new_per_session > TURBO_DIFFUSION_ESTIMATE_MIB
    assert HEXAGON_MAPPING_WINDOW_MIB - RUNTIME_BUDGET_MIB > 350

    segments = [TURBO_DIFFUSION_ESTIMATE_MIB / 42] * 42
    first, second = greedy_first_split(segments, new_per_session)
    assert first <= new_per_session
    assert second <= new_per_session
    assert first + second > old_total_weight_capacity


def test_moss_native_budget(source: pathlib.Path) -> None:
    bridge = pathlib.Path(__file__).resolve().parents[1] / "MossEngine.cpp"
    native = bridge.read_text()
    assert 'HTP0:0=2.75,HTP0:1=2.75' in native
    assert 'p.auto_fit = false' in native
    assert 'p.max_vram = kMossHtpV81Budgets' in native

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
    test_turbo_diffusion_capacity()
    if len(sys.argv) != 2:
        raise SystemExit("Pass staged source path to verify the scoped patch")
    test_scoped_upstream_patch(pathlib.Path(sys.argv[1]))
    test_moss_native_budget(pathlib.Path(sys.argv[1]))
    print("MOSS SM8850 Qwen/Turbo partition regressions: PASS (synthetic); staged patch verified")
