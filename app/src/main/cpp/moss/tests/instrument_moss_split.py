#!/usr/bin/env python3
"""Crash-localization and invalid split-budget preflight for the pinned MOSS copy.

This instruments ONLY the staged MOSS engine. It does not edit the upstream
submodule or any other Local Dream backend. Each source replacement is strict.
"""
from pathlib import Path
import sys

if len(sys.argv) != 2:
    raise SystemExit("usage: instrument_moss_split.py STAGED_SOURCE")
root = Path(sys.argv[1])
part = root / "src/core/layer_split_partition.cpp"
runner = root / "src/core/ggml_runner.cpp"


def insert_once(path: Path, anchor: str, insertion: str) -> None:
    data = path.read_text()
    if data.count(anchor) != 1:
        raise SystemExit(f"nonunique MOSS probe anchor in {path}: {anchor[:60]!r}")
    path.write_text(data.replace(anchor, insertion + anchor, 1))


# Separate stage checkpoints disambiguate a C++ graph/param pointer fault
# from backend weight residency or HTP allocation. Previous logs ended after
# budget display and therefore never identified which stage was crashing.
insert_once(
    part,
    "        // Existing placements may already occupy the reported free VRAM.",
    """        if (split_backends.size() == 2 &&
            layer_split_backend_device_display_name(split_backends[0]).rfind("HTP", 0) == 0) {
            LOG_INFO("[MOSS_SPLIT_TRACE] stage=capacity desc=%s segments=%zu "
                     "weights=%.1f MiB cap0=%.1f MiB cap1=%.1f MiB",
                     desc, plan.segments.size(), total_param_bytes / (1024.0 * 1024.0),
                     backend_capacities[0] / (1024.0 * 1024.0),
                     backend_capacities[1] / (1024.0 * 1024.0));
            // Treat cumulative capacity overflow as a regular error. Do NOT
            // attempt an over-budget native assignment that could abort the
            // entire Android process.
            if (total_param_bytes > backend_capacities[0] + backend_capacities[1]) {
                LOG_ERROR("[MOSS_SPLIT_TRACE] invalid total weight budget; "
                          "needed %.1f MiB, available %.1f MiB",
                          total_param_bytes / (1024.0 * 1024.0),
                          (backend_capacities[0] + backend_capacities[1]) / (1024.0 * 1024.0));
                return false;
            }
        }
""",
)
insert_once(
    part,
    "        const int n_nodes = ggml_graph_n_nodes(gf);",
    """        if (split_backends.size() == 2 &&
            layer_split_backend_device_display_name(split_backends[0]).rfind("HTP", 0) == 0) {
            LOG_INFO("[MOSS_SPLIT_TRACE] stage=segment-assignments desc=%s "
                     "split0=%.1f MiB split1=%.1f MiB",
                     desc,
                     assignment.bytes_by_backend[0] / (1024.0 * 1024.0),
                     assignment.bytes_by_backend[1] / (1024.0 * 1024.0));
        }
""",
)
insert_once(
    part,
    "        *assignment_out = std::move(assignment);",
    """        if (split_backends.size() == 2 &&
            layer_split_backend_device_display_name(split_backends[0]).rfind("HTP", 0) == 0) {
            LOG_INFO("[MOSS_SPLIT_TRACE] stage=node-assignments desc=%s nodes=%zu",
                     desc, assignment.node_assignments.size());
        }
""",
)
insert_once(
    runner,
    "    for (size_t i = 0; i < split_backends.size(); i++) {\n        if (assignment.tensors_by_backend[i].empty()) {",
    """    if (split_backends.size() == 2 &&
        sd::layer_split_backend_device_display_name(split_backends[0]).rfind("HTP", 0) == 0) {
        LOG_INFO("[MOSS_SPLIT_TRACE] stage=partition-return desc=%s", get_desc().c_str());
    }
""",
)
insert_once(
    runner,
    "        if (!manager->assign_compute_backend(assignment.tensors_by_backend[i], split_backends[i])) {",
    """        if (split_backends.size() == 2 &&
            sd::layer_split_backend_device_display_name(split_backends[0]).rfind("HTP", 0) == 0) {
            LOG_INFO("[MOSS_SPLIT_TRACE] stage=manager-enter desc=%s backend=%zu tensors=%zu",
                     get_desc().c_str(), i, assignment.tensors_by_backend[i].size());
        }
""",
)
insert_once(
    runner,
    "    graph_cut_layer_split_node_assignments_ = std::move(assignment.node_assignments);",
    """    if (split_backends.size() == 2 &&
        sd::layer_split_backend_device_display_name(split_backends[0]).rfind("HTP", 0) == 0) {
        LOG_INFO("[MOSS_SPLIT_TRACE] stage=manager-complete desc=%s", get_desc().c_str());
    }
""",
)
print("MOSS staged partition preflight and native crash breadcrumbs installed")
