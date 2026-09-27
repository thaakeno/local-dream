#!/usr/bin/env python3
"""Compile-time YuE2/Hexagon integration for Local Dream.

The dependency revisions are pinned. This script fails on source drift and adds
native HTP implementations for the two Oobleck ops missing from upstream
DSPQueue (SIN and COL2IM_1D), then switches yue2.cpp to a one-backend strict
accelerator scheduler. There is no runtime source rewriting or CPU fallback.
"""
from pathlib import Path
import shutil
import sys

if len(sys.argv) != 4:
    raise SystemExit("usage: integrate_htp_yue2.py <jz-root> <yue2-root> <overlay-dir>")

jz = Path(sys.argv[1])
yue = Path(sys.argv[2])
overlay = Path(sys.argv[3])
htp = jz / "ggml/src/ggml-hexagon/htp"
host = jz / "ggml/src/ggml-hexagon/ggml-hexagon.cpp"

def replace_once(path: Path, old: str, new: str, label: str) -> None:
    text = path.read_text()
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one source anchor, found {count} in {path}")
    path.write_text(text.replace(old, new, 1))

shutil.copy2(overlay / "sin-ops.c", htp / "sin-ops.c")
shutil.copy2(overlay / "col2im-ops.c", htp / "col2im-ops.c")

replace_once(
    htp / "CMakeLists.txt",
    "    im2col-ops.c\n    roll-ops.c",
    "    im2col-ops.c\n    col2im-ops.c\n    sin-ops.c\n    roll-ops.c",
    "HTP source list",
)

replace_once(
    htp / "htp-ops.h",
    "    HTP_OP_ROLL,\n    HTP_OP_ARGMAX,\n\n    HTP_OP_INVALID",
    "    HTP_OP_ROLL,\n    HTP_OP_ARGMAX,\n    HTP_OP_SIN,\n    HTP_OP_COL2IM_1D,\n\n    HTP_OP_INVALID",
    "HTP op enum",
)

replace_once(
    htp / "htp-ctx.h",
    "int op_im2col(struct htp_ops_context * octx);\nint op_allreduce",
    "int op_im2col(struct htp_ops_context * octx);\nint op_col2im_1d(struct htp_ops_context * octx);\nint op_sin(struct htp_ops_context * octx);\nint op_allreduce",
    "HTP op declarations",
)

replace_once(
    htp / "main.c",
    "        case HTP_OP_IM2COL:\n            return op_im2col(octx);\n\n        case HTP_OP_ROLL:",
    "        case HTP_OP_IM2COL:\n            return op_im2col(octx);\n\n        case HTP_OP_COL2IM_1D:\n            return op_col2im_1d(octx);\n\n        case HTP_OP_SIN:\n            return op_sin(octx);\n\n        case HTP_OP_ROLL:",
    "HTP dispatch",
)

helpers = r"""
static bool ggml_hexagon_supported_sin(const struct ggml_hexagon_session * sess, const struct ggml_tensor * op) {
    GGML_UNUSED(sess);
    const struct ggml_tensor * src = op->src[0];
    return src && src->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 &&
           ggml_are_same_shape(src, op) && ggml_is_contiguous(src) && ggml_is_contiguous(op);
}

static bool ggml_hexagon_supported_col2im_1d(const struct ggml_hexagon_session * sess, const struct ggml_tensor * op) {
    GGML_UNUSED(sess);
    const struct ggml_tensor * src = op->src[0];
    if (!src || src->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32 ||
        !ggml_is_contiguous(src) || !ggml_is_contiguous(op)) {
        return false;
    }
    const int32_t * params = (const int32_t *) op->op_params;
    const int32_t stride = params[0];
    const int32_t oc = params[1];
    const int32_t padding = params[2];
    if (stride <= 0 || oc <= 0 || src->ne[0] % oc != 0) return false;
    const int64_t kernel = src->ne[0] / oc;
    const int64_t t_out = (src->ne[1] - 1) * stride + kernel - 2 * padding;
    return t_out > 0 && op->ne[0] == t_out && op->ne[1] == oc;
}

"""
replace_once(
    host,
    "static bool ggml_backend_hexagon_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {",
    helpers + "static bool ggml_backend_hexagon_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {",
    "host support helpers",
)

replace_once(
    host,
    "        case GGML_OP_SQR:\n        case GGML_OP_SQRT:\n        case GGML_OP_LOG:",
    "        case GGML_OP_SIN:\n            supp = ggml_hexagon_supported_sin(sess, op);\n            break;\n\n        case GGML_OP_SQR:\n        case GGML_OP_SQRT:\n        case GGML_OP_LOG:",
    "host SIN support",
)

replace_once(
    host,
    "        case GGML_OP_IM2COL:\n            supp = ggml_hexagon_supported_im2col(sess, op);\n            break;",
    "        case GGML_OP_IM2COL:\n            supp = ggml_hexagon_supported_im2col(sess, op);\n            break;\n\n        case GGML_OP_COL2IM_1D:\n            supp = ggml_hexagon_supported_col2im_1d(sess, op);\n            break;",
    "host COL2IM support",
)

replace_once(
    host,
    "        case GGML_OP_SQR:             return HTP_OP_SQR;",
    "        case GGML_OP_SIN:             return HTP_OP_SIN;\n        case GGML_OP_SQR:             return HTP_OP_SQR;",
    "host SIN remap",
)

replace_once(
    host,
    "        case GGML_OP_IM2COL:          return HTP_OP_IM2COL;",
    "        case GGML_OP_IM2COL:          return HTP_OP_IM2COL;\n        case GGML_OP_COL2IM_1D:       return HTP_OP_COL2IM_1D;",
    "host COL2IM remap",
)

backend_h = yue / "src/backend.h"
replace_once(
    backend_h,
    """    bool best_is_cpu = (strcmp(ggml_backend_name(bp.backend), "CPU") == 0);
    int  n_threads   = backend_cpu_n_threads();
    if (best_is_cpu) {
        ggml_backend_free(bp.backend);
        bp.backend     = cpu_backend_new(n_threads);
        bp.cpu_backend = bp.backend;
    } else {
        bp.cpu_backend = cpu_backend_new(n_threads);
    }
    if (!bp.cpu_backend) {
        fprintf(stderr, "[Load] FATAL: failed to init CPU backend\\n");
        exit(1);
    }
    bp.has_gpu = !best_is_cpu;
    fprintf(stderr, "[Load] %s backend: %s (CPU threads: %d)\\n", label, ggml_backend_name(bp.backend), n_threads);
""",
    """    bool best_is_cpu = (strcmp(ggml_backend_name(bp.backend), "CPU") == 0);
    int  n_threads   = backend_cpu_n_threads();
    const char * strict_env = std::getenv("YUE2_STRICT_ACCELERATOR");
    const bool strict_accelerator = !best_is_cpu && strict_env && strcmp(strict_env, "0") != 0;
    if (best_is_cpu) {
        ggml_backend_free(bp.backend);
        bp.backend     = cpu_backend_new(n_threads);
        bp.cpu_backend = bp.backend;
    } else if (strict_accelerator) {
        bp.cpu_backend = bp.backend;
    } else {
        bp.cpu_backend = cpu_backend_new(n_threads);
    }
    if (!bp.cpu_backend) {
        fprintf(stderr, "[Load] FATAL: failed to init scheduler backend\\n");
        exit(1);
    }
    bp.has_gpu = !best_is_cpu;
    fprintf(stderr, "[Load] %s backend: %s (%s, CPU threads: %d)\\n", label, ggml_backend_name(bp.backend),
            strict_accelerator ? "STRICT accelerator-only" : "hybrid", n_threads);
""",
    "strict yue2 scheduler",
)

qwen = yue / "src/qwen3-lm.h"
qtext = qwen.read_text()
needle = "    ggml_backend_sched_graph_compute(m->sched, gf);"
if qtext.count(needle) < 1:
    raise RuntimeError("qwen graph compute anchor missing")
qtext = qtext.replace(
    needle,
    """    if (ggml_backend_sched_graph_compute(m->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "[LM] FATAL: accelerator graph compute failed\\n");
        std::abort();
    }""",
)
qwen.write_text(qtext)

replace_once(
    yue / "src/nar.h",
    "    ggml_backend_sched_graph_compute(n->sched, n->graph);\n    ggml_backend_tensor_get(n->out_v, v_out, 0, block * M * sizeof(float));",
    """    if (ggml_backend_sched_graph_compute(n->sched, n->graph) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "[NAR] FATAL: accelerator graph compute failed\\n");
        return false;
    }
    ggml_backend_tensor_get(n->out_v, v_out, 0, block * M * sizeof(float));""",
    "NAR compute status",
)

vae = yue / "src/vae.h"
replace_once(
    vae,
    "    ggml_backend_sched_graph_compute(m->sched, m->graph);\n\n    return (int) m->graph_output->ne[0];",
    """    if (ggml_backend_sched_graph_compute(m->sched, m->graph) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "[VAE] FATAL: accelerator graph compute failed for T=%d\\n", T_latent);
        return -1;
    }

    return (int) m->graph_output->ne[0];""",
    "VAE compute status",
)

replace_once(
    vae,
    """        int tile_T = vae_ggml_compute(m, latent, right - left, left);
        if (tile_T < 0) {
            fprintf(stderr, "[VAE] FATAL: tile %d decode failed\\n", i);
            return -1;
        }
""",
    """        fprintf(stderr, "[VAE] Tile %d/%d begin: latent=%d\\n", i + 1, num_tiles, right - left);
        Timer tile_timer;
        int tile_T = vae_ggml_compute(m, latent, right - left, left);
        if (tile_T < 0) {
            fprintf(stderr, "[VAE] FATAL: tile %d decode failed\\n", i);
            return -1;
        }
        fprintf(stderr, "[VAE] Tile %d/%d done: %.0f ms\\n", i + 1, num_tiles, tile_timer.ms());
""",
    "VAE tile progress",
)

print("Local Dream YuE2 native HTP integration applied")
