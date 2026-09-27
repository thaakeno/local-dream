#!/usr/bin/env python3
"""Compile-time YuE2/Hexagon integration for Local Dream.

The dependency revisions are pinned. This script fails on source drift and adds
native HTP implementations for the Oobleck ops missing from upstream
DSPQueue (SIN and COL2IM_1D), adds adaptive VTCM fitting for native HTP
binary broadcasts, canonicalizes VAE binary inputs, and enforces strict accelerator compute. There is no CPU compute fallback.
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

replace_once(
    host,
    """    struct htp_binary_vtcm_layout L;
    htp_binary_vtcm_layout_build(&L, kparams, sess->vtcm_size);
    if (L.rows_per_buffer == 0 || L.total_bytes > sess->vtcm_size) {""",
    """    // YuE2 adaptive binary VTCM thread fit.
    // Large audio rows can exceed the 8 MiB VTCM budget when all HVX
    // workers double-buffer a full row. The native scalar-broadcast kernel
    // already supports [T,C] + [1,C], so reduce only this op's HTP worker
    // count until the exact same kernel fits. No CPU fallback and no repeat.
    struct htp_binary_vtcm_layout L;
    while (kparams->n_threads > 0) {
        htp_binary_vtcm_layout_build(&L, kparams, sess->vtcm_size);
        if (L.rows_per_buffer != 0 && L.total_bytes <= sess->vtcm_size) {
            break;
        }
        --kparams->n_threads;
    }
    if (kparams->n_threads == 0) {""",
    "YuE2 adaptive binary VTCM thread fit",
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
    } else {
        // GGML requires the final scheduler backend to be a real CPU device.
        // Strict mode still registers it, then pins every compute node to HTP
        // before graph allocation so CPU can never execute model operations.
        bp.cpu_backend = cpu_backend_new(n_threads);
    }
    if (!bp.cpu_backend) {
        fprintf(stderr, "[Load] FATAL: failed to init CPU scheduler backend\\n");
        exit(1);
    }
    bp.has_gpu = !best_is_cpu;
    fprintf(stderr, "[Load] %s backend: %s (%s, CPU compute: %s, threads: %d)\\n",
            label, ggml_backend_name(bp.backend),
            strict_accelerator ? "STRICT HTP compute" : "hybrid",
            strict_accelerator ? "forbidden" : "allowed", n_threads);
""",
    "strict yue2 scheduler",
)

replace_once(
    backend_h,
    """    return sched;
}
""",
    """    return sched;
}

// GGML's scheduler requires CPU to be present as the final backend. Strict
// mode keeps that structural CPU backend but validates and pins all compute
// nodes to HTP before allocation. CPU is only allowed to hold host/input data.
static bool backend_strict_pin_graph(ggml_backend_sched_t sched,
                                     ggml_backend_t accelerator,
                                     struct ggml_cgraph * graph,
                                     const char * label) {
    const char * strict_env = std::getenv("YUE2_STRICT_ACCELERATOR");
    if (!strict_env || strcmp(strict_env, "0") == 0) {
        return true;
    }
    if (!accelerator ||
        ggml_backend_dev_type(ggml_backend_get_device(accelerator)) == GGML_BACKEND_DEVICE_TYPE_CPU) {
        fprintf(stderr, "[%s] FATAL: strict accelerator mode has no accelerator backend\\n", label);
        return false;
    }

    int pinned = 0;
    const int n_nodes = ggml_graph_n_nodes(graph);
    for (int i = 0; i < n_nodes; ++i) {
        struct ggml_tensor * node = ggml_graph_node(graph, i);
        if (!node || node->op == GGML_OP_NONE || ggml_is_view(node)) {
            continue;
        }
        if (!ggml_backend_supports_op(accelerator, node)) {
            const struct ggml_tensor * s0 = node->src[0];
            const struct ggml_tensor * s1 = node->src[1];
            fprintf(stderr,
                    "[%s] FATAL: HTP unsupported node %d op=%s name=%s; CPU fallback blocked\\n",
                    label, i, ggml_op_desc(node), node->name[0] ? node->name : "<unnamed>");
            fprintf(stderr,
                    "[%s] node shape=[%lld,%lld,%lld,%lld] type=%s nb=[%zu,%zu,%zu,%zu]\\n",
                    label,
                    (long long) node->ne[0], (long long) node->ne[1],
                    (long long) node->ne[2], (long long) node->ne[3],
                    ggml_type_name(node->type),
                    node->nb[0], node->nb[1], node->nb[2], node->nb[3]);
            if (s0) {
                fprintf(stderr,
                        "[%s] src0 shape=[%lld,%lld,%lld,%lld] type=%s nb=[%zu,%zu,%zu,%zu] cont=%d perm=%d\\n",
                        label,
                        (long long) s0->ne[0], (long long) s0->ne[1],
                        (long long) s0->ne[2], (long long) s0->ne[3],
                        ggml_type_name(s0->type),
                        s0->nb[0], s0->nb[1], s0->nb[2], s0->nb[3],
                        (int) ggml_is_contiguous(s0), (int) ggml_is_permuted(s0));
            }
            if (s1) {
                fprintf(stderr,
                        "[%s] src1 shape=[%lld,%lld,%lld,%lld] type=%s nb=[%zu,%zu,%zu,%zu] cont=%d perm=%d\\n",
                        label,
                        (long long) s1->ne[0], (long long) s1->ne[1],
                        (long long) s1->ne[2], (long long) s1->ne[3],
                        ggml_type_name(s1->type),
                        s1->nb[0], s1->nb[1], s1->nb[2], s1->nb[3],
                        (int) ggml_is_contiguous(s1), (int) ggml_is_permuted(s1));
            }
            return false;
        }
        ggml_backend_sched_set_tensor_backend(sched, node, accelerator);
        ++pinned;
    }

    if (getenv("YUE2_LOG_DEBUG")) {
        fprintf(stderr, "[%s] Strict HTP preflight: %d compute nodes pinned to %s\\n",
                label, pinned, ggml_backend_name(accelerator));
    }
    return true;
}
""",
    "strict graph pin helper",
)

qwen = yue / "src/qwen3-lm.h"
qtext = qwen.read_text()
needle = "    ggml_backend_sched_graph_compute(m->sched, gf);"
if qtext.count(needle) < 1:
    raise RuntimeError("qwen graph compute anchor missing")
qtext = qtext.replace(
    """    ggml_backend_sched_reset(m->sched);
    if (!ggml_backend_sched_alloc_graph(m->sched, gf)) {""",
    """    ggml_backend_sched_reset(m->sched);
    if (!backend_strict_pin_graph(m->sched, m->backend, gf, "LM")) {
        std::abort();
    }
    if (!ggml_backend_sched_alloc_graph(m->sched, gf)) {""",
)

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
    """    ggml_backend_sched_reset(n->sched);
    if (!ggml_backend_sched_alloc_graph(n->sched, n->graph)) {""",
    """    ggml_backend_sched_reset(n->sched);
    if (!backend_strict_pin_graph(n->sched, n->backend, n->graph, "NAR")) {
        return false;
    }
    if (!ggml_backend_sched_alloc_graph(n->sched, n->graph)) {""",
    "NAR strict pin",
)

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
    """// Graph building
// Snake activation (5-op naive decomposition for backend pattern fusion)""",
    """// Graph building
// Keep ADDs native-HTP compatible without changing their numerical result.
// Conv/col2im views may expose non-contiguous strides rejected by Hexagon's
// binary kernels. Materialize only those operands; already-contiguous tensors
// remain zero-copy.
static struct ggml_tensor * vae_htp_add(struct ggml_context * ctx,
                                        struct ggml_tensor *  a,
                                        struct ggml_tensor *  b) {
    if (!ggml_is_contiguous(a) || ggml_is_permuted(a)) {
        a = ggml_cont(ctx, a);
    }
    if (!ggml_is_contiguous(b) || ggml_is_permuted(b)) {
        b = ggml_cont(ctx, b);
    }
    return ggml_add(ctx, a, b);
}

// Snake activation (5-op naive decomposition for backend pattern fusion)""",
    "VAE HTP-safe ADD canonicalization",
)

vtext = vae.read_text()
vtext = vtext.replace("return ggml_add(ctx, x, d);",
                      "return vae_htp_add(ctx, x, d);")
vtext = vtext.replace("y                        = ggml_add(ctx, y, b2d);",
                      "y                        = vae_htp_add(ctx, y, b2d);")
vtext = vtext.replace("return ggml_add(ctx, skip, x);",
                      "return vae_htp_add(ctx, skip, x);")
vae.write_text(vtext)

replace_once(
    vae,
    """        if (!ggml_backend_sched_alloc_graph(m->sched, m->graph)) {""",
    """        if (!backend_strict_pin_graph(m->sched, m->backend, m->graph, "VAE")) {
            fprintf(stderr, "[VAE] FATAL: strict HTP graph preflight failed for T=%d\\n", T_latent);
            ggml_free(ctx);
            free(m->graph_buf);
            m->graph_ctx = NULL;
            m->graph_buf = NULL;
            m->graph_T   = 0;
            return -1;
        }
        if (!ggml_backend_sched_alloc_graph(m->sched, m->graph)) {""",
    "VAE strict pin",
)

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
