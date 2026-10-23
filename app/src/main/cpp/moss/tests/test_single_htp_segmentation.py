#!/usr/bin/env python3
"""Assert MOSS uses upstream, budgeted single-NPU segmented execution."""
from pathlib import Path
import sys

if len(sys.argv) != 2:
    raise SystemExit("usage: test_single_htp_segmentation.py STAGED_SOURCE")

stage = Path(sys.argv[1])
moss = Path(__file__).resolve().parents[1]
main = (moss.parent / "src/main.cpp").read_text()
bridge = (moss / "MossEngine.cpp").read_text()
svc = (moss.parent.parent / "java/io/github/xororz/localdream/service/BackendService.kt").read_text()
runner = (stage / "src/core/ggml_runner.cpp").read_text()
hexagon = (stage / "ggml/src/ggml-hexagon/ggml-hexagon.cpp").read_text()
partition = (stage / "src/core/layer_split_partition.cpp").read_text()
assert 'moss ? "diffusion=HTP0:0,te=HTP0:0,vae=HTP0:0"' in main
assert 'moss ? "all=disk"' in main
assert 'const std::string moss_htp = "HTP0:0&HTP0:1"' not in main
assert 'kMossSingleHtpBudget = "HTP0:0=2.60"' in bridge
assert 'std::strcmp(req->params_backend, "all=disk")' in bridge
assert 'max_vram=HTP0:0=2.60GiB params_backend=all=disk' in svc
assert 'env["GGML_HEXAGON_DEVICES"] = "HTP0:0,HTP0:1"' in svc
# The native driver defaults must be restored: on the old crash workaround
# NHMX=0 blocked HMX matmul; OPFUSION=0 disabled the normal NPU graph fusions.
assert 'env["GGML_HEXAGON_NHMX"] =' not in svc
assert 'env["GGML_HEXAGON_OPFUSION"] =' not in svc
assert 'env["GGML_HEXAGON_OPBATCH"] =' not in svc
assert 'env["GGML_HEXAGON_OPQUEUE"] =' not in svc
assert 'opt_nhmx    = 1' in hexagon
assert 'opt_opfusion = 1' in hexagon

assert '!is_multi_device() && !sd_backend_is_cpu(runtime_backend)' in runner
assert 'manager->segmented_compute_enabled()' in runner
assert 'ggml_graph_cut::build_segment_graph' in runner
assert '[MOSS_HTP_COPY]' not in hexagon
assert 'MOSS HTP split planner:' not in partition
print("MOSS single HTP segmented graph cut and pristine ggml source: PASS")
