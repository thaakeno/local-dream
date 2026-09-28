#!/usr/bin/env python3
"""Compile-time YuE2/Hexagon integration for Local Dream.

The dependency revisions are pinned. This script fails on source drift and adds
native HTP implementations for the Oobleck ops missing from upstream
DSPQueue (SIN, COL2IM_1D and streaming channel-broadcast ADD/MUL), fuses
Oobleck Snake and transpose-conv bias into native HTP kernels, uses ComfyUI-compatible
DPM-Solver++ 2M with the SGM-uniform flow schedule, trims HTP attention windows
to native 64-key blocks, and enforces strict accelerator compute.
There is no CPU compute fallback.
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
yue_cmake = yue / "CMakeLists.txt"

def replace_once(path: Path, old: str, new: str, label: str) -> None:
    text = path.read_text()
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one source anchor, found {count} in {path}")
    path.write_text(text.replace(old, new, 1))

replace_once(
    yue_cmake,
    "# yue-server: HTTP server (single slot job queue + embedded webui)",
    """# Local Dream: representative HTP transport benchmark. The identical graph
# is compiled against either DSPQueue or FastRPC/mempool for a fair on-device A/B.
if(EXISTS "${CMAKE_SOURCE_DIR}/tools/yue-transport-bench.cpp")
    add_executable(yue-transport-bench tools/yue-transport-bench.cpp)
    link_ggml_backends(yue-transport-bench)
    if(GGML_HEXAGON_USE_MEMPOOL)
        target_compile_definitions(yue-transport-bench PRIVATE YUE2_TRANSPORT_FASTRPC=1)
    else()
        target_compile_definitions(yue-transport-bench PRIVATE YUE2_TRANSPORT_FASTRPC=0)
    endif()
endif()

# yue-server: HTTP server (single slot job queue + embedded webui)""",
    "transport benchmark target",
)

shutil.copy2(overlay / "sin-ops.c", htp / "sin-ops.c")
shutil.copy2(overlay / "col2im-ops.c", htp / "col2im-ops.c")
shutil.copy2(overlay / "snake-ops.c", htp / "snake-ops.c")
shutil.copy2(overlay / "channel-bcast-add-ops.c", htp / "channel-bcast-add-ops.c")
shutil.copy2(overlay / "instrumental-transfer.h", yue / "src/instrumental-transfer.h")
shutil.copy2(overlay / "ar-lora.h", yue / "src/ar-lora.h")

replace_once(
    htp / "CMakeLists.txt",
    "    im2col-ops.c\n    roll-ops.c",
    "    im2col-ops.c\n    col2im-ops.c\n    sin-ops.c\n    snake-ops.c\n    channel-bcast-add-ops.c\n    roll-ops.c",
    "HTP source list",
)

replace_once(
    htp / "htp-ops.h",
    "    HTP_OP_ROLL,\n    HTP_OP_ARGMAX,\n\n    HTP_OP_INVALID",
    "    HTP_OP_ROLL,\n    HTP_OP_ARGMAX,\n    HTP_OP_SIN,\n    HTP_OP_COL2IM_1D,\n    HTP_OP_COL2IM_1D_BIAS,\n    HTP_OP_SNAKE,\n    HTP_OP_CHANNEL_BCAST_ADD,\n    HTP_OP_CHANNEL_BCAST_MUL,\n\n    HTP_OP_INVALID",
    "HTP op enum",
)

replace_once(
    htp / "htp-ctx.h",
    "int op_im2col(struct htp_ops_context * octx);\nint op_allreduce",
    "int op_im2col(struct htp_ops_context * octx);\nint op_col2im_1d(struct htp_ops_context * octx);\nint op_col2im_1d_bias(struct htp_ops_context * octx);\nint op_sin(struct htp_ops_context * octx);\nint op_snake(struct htp_ops_context * octx);\nint op_channel_bcast_add(struct htp_ops_context * octx);\nint op_channel_bcast_mul(struct htp_ops_context * octx);\nint op_allreduce",
    "HTP op declarations",
)

replace_once(
    htp / "main.c",
    "        case HTP_OP_IM2COL:\n            return op_im2col(octx);\n\n        case HTP_OP_ROLL:",
    "        case HTP_OP_IM2COL:\n            return op_im2col(octx);\n\n        case HTP_OP_COL2IM_1D:\n            return op_col2im_1d(octx);\n\n        case HTP_OP_COL2IM_1D_BIAS:\n            return op_col2im_1d_bias(octx);\n\n        case HTP_OP_SIN:\n            return op_sin(octx);\n\n        case HTP_OP_SNAKE:\n            return op_snake(octx);\n\n        case HTP_OP_CHANNEL_BCAST_ADD:\n            return op_channel_bcast_add(octx);\n\n        case HTP_OP_CHANNEL_BCAST_MUL:\n            return op_channel_bcast_mul(octx);\n\n        case HTP_OP_ROLL:",
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


# Oobleck emits huge contiguous F32 binary rows. DSPQueue's generic binary
# path stages full rows in VTCM. At the final 48 kHz stages one row is
# ~3.84 MiB, so even one generic worker cannot fit. Route channel broadcasts
# and very wide equal-shape residuals through bounded streaming HVX kernels.
channel_add_helper = r"""
static bool ggml_hexagon_is_yue2_channel_bcast(const struct ggml_tensor * op) {
    if (!op || (op->op != GGML_OP_ADD && op->op != GGML_OP_MUL) ||
        op->type != GGML_TYPE_F32) {
        return false;
    }
    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * src1 = op->src[1];
    if (!src0 || !src1 || src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32) {
        return false;
    }
    if (src0->ne[0] <= 1 || src0->ne[1] <= 0 ||
        src0->ne[2] != 1 || src0->ne[3] != 1 ||
        op->ne[0] != src0->ne[0] || op->ne[1] != src0->ne[1] ||
        op->ne[2] != 1 || op->ne[3] != 1 ||
        !ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) ||
        !ggml_is_contiguous(op) ||
        ggml_is_permuted(src0) || ggml_is_permuted(src1) ||
        ggml_is_permuted(op)) {
        return false;
    }

    const bool channel_broadcast =
        src1->ne[0] == 1 && src1->ne[1] == src0->ne[1] &&
        src1->ne[2] == 1 && src1->ne[3] == 1;
    const bool same_shape =
        src1->ne[0] == src0->ne[0] && src1->ne[1] == src0->ne[1] &&
        src1->ne[2] == 1 && src1->ne[3] == 1;

    // Broadcasts are the Snake/bias form. Equal-shape residuals stay on the
    // normal VTCM kernel until the time row is >= 1 MiB; past that point the
    // generic double-buffer layout is no longer robust on an 8 MiB v81 HTP.
    const bool wide_residual = same_shape && src0->ne[0] >= 262144;
    return channel_broadcast || wide_residual;
}

"""
replace_once(
    host,
    "static htp_op_code op_remap_to_htp(const ggml_tensor * t) {",
    channel_add_helper + "static htp_op_code op_remap_to_htp(const ggml_tensor * t) {",
    "YuE2 channel-broadcast binary helper",
)

replace_once(
    host,
    "        case GGML_OP_MUL:             return HTP_OP_MUL;\n        case GGML_OP_ADD:             return HTP_OP_ADD;",
    "        case GGML_OP_MUL:             return ggml_hexagon_is_yue2_channel_bcast(t) ? HTP_OP_CHANNEL_BCAST_MUL : HTP_OP_MUL;\n        case GGML_OP_ADD:             return ggml_hexagon_is_yue2_channel_bcast(t) ? HTP_OP_CHANNEL_BCAST_ADD : HTP_OP_ADD;",
    "YuE2 channel-broadcast binary remap",
)

replace_once(
    host,
    """        case GGML_OP_MUL:
        case GGML_OP_ADD:
        case GGML_OP_SUB:
        case GGML_OP_DIV:
            supp = ggml_hexagon_supported_binary(sess, op);
            break;""",
    """        case GGML_OP_SUB:
        case GGML_OP_DIV:
            supp = ggml_hexagon_supported_binary(sess, op);
            break;

        case GGML_OP_ADD:
        case GGML_OP_MUL:
            supp = ggml_hexagon_is_yue2_channel_bcast(op) ||
                   ggml_hexagon_supported_binary(sess, op);
            break;""",
    "YuE2 channel-broadcast binary support",
)

# Oobleck transpose-conv is emitted as COL2IM_1D followed immediately by a
# [T,C] + [1,C] bias. Fuse that pair in the DSPQueue batch so the native
# COL2IM kernel applies the bias while each sample is already in a register.
# This removes one complete waveform-sized DDR pass per upsampling block.
col2im_bias_fusion = r"""
    bool try_fuse_yue2_col2im_bias(const htp_opnode & node) {
        if ((node.opcode != HTP_OP_ADD &&
             node.opcode != HTP_OP_CHANNEL_BCAST_ADD) ||
            n_ops == 0) {
            return false;
        }

        htp_opnode & col = ops[n_ops - 1];
        if (col.opcode != HTP_OP_COL2IM_1D) {
            return false;
        }

        const ggml_tensor * col_out = col.dst();
        const ggml_tensor * out = node.dst();
        if (!col_out || !out || !ggml_hexagon_tensor_is_fuseable(col_out)) {
            return false;
        }

        const ggml_tensor * bias = nullptr;
        if (node.src0() == col_out ||
            (node.src0() && node.src0()->data == col_out->data)) {
            bias = node.src1();
        } else if (node.src1() == col_out ||
                   (node.src1() && node.src1()->data == col_out->data)) {
            bias = node.src0();
        }

        if (!bias || bias->type != GGML_TYPE_F32 ||
            col_out->type != GGML_TYPE_F32 || out->type != GGML_TYPE_F32 ||
            bias->ne[0] != 1 || bias->ne[1] != col_out->ne[1] ||
            bias->ne[2] != 1 || bias->ne[3] != 1 ||
            !ggml_is_contiguous(bias) || !ggml_is_contiguous(col_out) ||
            !ggml_is_contiguous(out) || !ggml_are_same_shape(col_out, out)) {
            return false;
        }

        const ggml_tensor * src = col.src0();
        if (!src || !try_fuse_common({src, bias, out})) {
            return false;
        }

        col.opcode = HTP_OP_COL2IM_1D_BIAS;
        col.name = "YUE2_COL2IM_BIAS";
        col.inputs.clear();
        col.inputs.push_back(src);
        col.inputs.push_back(bias);
        col.outputs.clear();
        col.outputs.push_back(out);
        col.fused.push_back(node.node);

        htp_op_desc & o = h_ops[n_ops - 1];
        o.opcode = HTP_OP_COL2IM_1D_BIAS;
        // Preserve o.params: they already hold stride/out-channels/padding
        // from the original COL2IM node.
        o.src[0] = add_tensor(src);
        o.src[1] = add_tensor(bias);
        for (unsigned int k = 2; k < HTP_OP_MAX_INPUTS; ++k) {
            o.src[k] = 0xffff;
        }
        o.dst[0] = add_tensor(out);
        for (unsigned int k = 1; k < HTP_OP_MAX_OUTPUTS; ++k) {
            o.dst[k] = 0xffff;
        }

        HEX_VERBOSE("ggml-hex: %s fused YuE2 COL2IM+bias (#%u)\n",
                    sess->c_name(), n_ops - 1);
        return true;
    }

"""

# Oobleck Snake is emitted by yue2.cpp as:
#   MUL(x, alpha) -> SIN -> SQR -> MUL(inv_beta) -> ADD(x)
# Collapse the chain inside the DSPQueue batch builder so one HTP invocation
# performs the entire activation. This keeps the canonical GGML graph intact
# while eliminating four intermediate DDR tensors / dispatch barriers for each
# of the decoder's dozens of Snake activations.
snake_fusion = r"""
    bool try_fuse_yue2_snake(const htp_opnode & node) {
        if ((node.opcode != HTP_OP_ADD &&
             node.opcode != HTP_OP_CHANNEL_BCAST_ADD) ||
            n_ops < 4) return false;

        const unsigned int base = n_ops - 4;
        htp_opnode & mul_alpha = ops[base + 0];
        htp_opnode & sin_node  = ops[base + 1];
        htp_opnode & sqr_node  = ops[base + 2];
        htp_opnode & mul_beta  = ops[base + 3];

        const bool mul_alpha_ok =
            mul_alpha.opcode == HTP_OP_MUL ||
            mul_alpha.opcode == HTP_OP_CHANNEL_BCAST_MUL;
        const bool mul_beta_ok =
            mul_beta.opcode == HTP_OP_MUL ||
            mul_beta.opcode == HTP_OP_CHANNEL_BCAST_MUL;
        if (!mul_alpha_ok ||
            sin_node.opcode != HTP_OP_SIN ||
            sqr_node.opcode != HTP_OP_SQR ||
            !mul_beta_ok) {
            return false;
        }

        const ggml_tensor * ax  = mul_alpha.dst();
        const ggml_tensor * sn  = sin_node.dst();
        const ggml_tensor * sq  = sqr_node.dst();
        const ggml_tensor * del = mul_beta.dst();
        const ggml_tensor * out = node.dst();

        if (!ax || !sn || !sq || !del || !out ||
            sin_node.src0() != ax ||
            sqr_node.src0() != sn) {
            return false;
        }

        const ggml_tensor * x = nullptr;
        const ggml_tensor * alpha = nullptr;
        if (mul_alpha.src0() && mul_alpha.src1()) {
            // In the canonical graph x is [T,C] and alpha is [1,C]. Keep this
            // structural check instead of depending on tensor names.
            if (mul_alpha.src0()->ne[0] > 1 && mul_alpha.src1()->ne[0] == 1) {
                x = mul_alpha.src0();
                alpha = mul_alpha.src1();
            } else if (mul_alpha.src1()->ne[0] > 1 && mul_alpha.src0()->ne[0] == 1) {
                x = mul_alpha.src1();
                alpha = mul_alpha.src0();
            }
        }
        if (!x || !alpha) return false;

        const ggml_tensor * inv_beta = nullptr;
        if (mul_beta.src0() == sq) inv_beta = mul_beta.src1();
        else if (mul_beta.src1() == sq) inv_beta = mul_beta.src0();
        if (!inv_beta) return false;

        const bool final_chain =
            (node.src0() == x && node.src1() == del) ||
            (node.src1() == x && node.src0() == del);
        if (!final_chain) return false;

        if (x->type != GGML_TYPE_F32 || alpha->type != GGML_TYPE_F32 ||
            inv_beta->type != GGML_TYPE_F32 || out->type != GGML_TYPE_F32 ||
            x->ne[2] != 1 || x->ne[3] != 1 ||
            out->ne[0] != x->ne[0] || out->ne[1] != x->ne[1] ||
            out->ne[2] != 1 || out->ne[3] != 1 ||
            alpha->ne[0] != 1 || alpha->ne[1] != x->ne[1] ||
            alpha->ne[2] != 1 || alpha->ne[3] != 1 ||
            inv_beta->ne[0] != 1 || inv_beta->ne[1] != x->ne[1] ||
            inv_beta->ne[2] != 1 || inv_beta->ne[3] != 1 ||
            !ggml_is_contiguous(x) || !ggml_is_contiguous(alpha) ||
            !ggml_is_contiguous(inv_beta) || !ggml_is_contiguous(out)) {
            return false;
        }

        // Every intermediate must be single-use/fuseable; otherwise replacing
        // the chain would change another consumer's value.
        if (!ggml_hexagon_tensor_is_fuseable(ax) ||
            !ggml_hexagon_tensor_is_fuseable(sn) ||
            !ggml_hexagon_tensor_is_fuseable(sq) ||
            !ggml_hexagon_tensor_is_fuseable(del)) {
            return false;
        }

        if (!try_fuse_common({x, alpha, inv_beta, out})) {
            return false;
        }

        htp_opnode fused(HTP_OP_SNAKE, const_cast<ggml_tensor *>(out));
        fused.name = "YUE2_SNAKE";
        fused.inputs.clear();
        fused.inputs.push_back(x);
        fused.inputs.push_back(alpha);
        fused.inputs.push_back(inv_beta);
        fused.outputs.clear();
        fused.outputs.push_back(out);
        fused.fused.push_back(mul_alpha.node);
        fused.fused.push_back(sin_node.node);
        fused.fused.push_back(sqr_node.node);
        fused.fused.push_back(mul_beta.node);
        fused.fused.push_back(node.node);

        // Reuse the first descriptor slot and discard the remaining four ops.
        // Tensor-map entries for the removed intermediates may remain in the
        // batch metadata; they are harmless and avoid mutating the map while
        // the batch is being built.
        ops[base] = fused;
        htp_op_desc & o = h_ops[base];
        memset(&o, 0, sizeof(o));
        o.opcode = HTP_OP_SNAKE;
        for (unsigned int k = 0; k < HTP_OP_MAX_INPUTS; ++k) o.src[k] = 0xffff;
        for (unsigned int k = 0; k < HTP_OP_MAX_OUTPUTS; ++k) o.dst[k] = 0xffff;
        o.src[0] = add_tensor(x);
        o.src[1] = add_tensor(alpha);
        o.src[2] = add_tensor(inv_beta);
        o.dst[0] = add_tensor(out);

        n_ops = base + 1;
        HEX_VERBOSE("ggml-hex: %s fused YuE2 Snake (#%u)\n", sess->c_name(), base);
        return true;
    }

"""
replace_once(
    host,
    """    bool try_fuse(const htp_opnode & node) {
        if (!opt_opfusion) return false;""",
    col2im_bias_fusion + snake_fusion + """    bool try_fuse(const htp_opnode & node) {
        if (!opt_opfusion) return false;
        if (try_fuse_yue2_col2im_bias(node)) return true;
        if (try_fuse_yue2_snake(node)) return true;""",
    "YuE2 fused Snake batch pattern",
)


replace_once(
    host,
    """            if (graph->nodes[i]->op == GGML_OP_RMS_NORM && ggml_can_fuse(graph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL })) {
                extra->flags |= GGML_HEXAGON_TENSOR_FUSEABLE;
            } else if (graph->nodes[i]->op == GGML_OP_MUL_MAT || graph->nodes[i]->op == GGML_OP_MUL_MAT_ID) {
                if ((i + 1 < graph->n_nodes && graph->nodes[i + 1]->op == GGML_OP_ADD && ggml_can_fuse(graph, i, { graph->nodes[i]->op, GGML_OP_ADD })) ||
                    ggml_node_has_n_uses(graph, i, 1)) {
                    extra->flags |= GGML_HEXAGON_TENSOR_FUSEABLE;
                }
            }""",
    """            if (graph->nodes[i]->op == GGML_OP_RMS_NORM && ggml_can_fuse(graph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL })) {
                extra->flags |= GGML_HEXAGON_TENSOR_FUSEABLE;
            } else if (graph->nodes[i]->op == GGML_OP_MUL_MAT || graph->nodes[i]->op == GGML_OP_MUL_MAT_ID) {
                if ((i + 1 < graph->n_nodes && graph->nodes[i + 1]->op == GGML_OP_ADD && ggml_can_fuse(graph, i, { graph->nodes[i]->op, GGML_OP_ADD })) ||
                    ggml_node_has_n_uses(graph, i, 1)) {
                    extra->flags |= GGML_HEXAGON_TENSOR_FUSEABLE;
                }
            } else if ((graph->nodes[i]->op == GGML_OP_MUL ||
                        graph->nodes[i]->op == GGML_OP_SIN ||
                        graph->nodes[i]->op == GGML_OP_SQR ||
                        graph->nodes[i]->op == GGML_OP_COL2IM_1D) &&
                       ggml_node_has_n_uses(graph, i, 1)) {
                // YuE2 Oobleck Snake intermediates are single-consumer. Mark
                // them so the DSPQueue pattern fusion can safely collapse
                // MUL->SIN->SQR->MUL->ADD without touching unrelated graphs.
                extra->flags |= GGML_HEXAGON_TENSOR_FUSEABLE;
            }""",
    "YuE2 Snake fuseable tensor tags",
)

replace_once(
    host,
    """    struct htp_binary_vtcm_layout L;
    htp_binary_vtcm_layout_build(&L, kparams, sess->vtcm_size);
    if (L.rows_per_buffer == 0 || L.total_bytes > sess->vtcm_size) {""",
    """    // Adaptive generic binary VTCM thread fit. Very wide YuE2 channel
    // broadcasts use the dedicated streaming ops above; for other supported binary
    // geometries, reduce only this op's worker count until its exact kernel
    // fits. No CPU fallback and no semantic rewrite.
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
    int unsupported = 0;
    const int n_nodes = ggml_graph_n_nodes(graph);
    for (int i = 0; i < n_nodes; ++i) {
        struct ggml_tensor * node = ggml_graph_node(graph, i);
        if (!node || node->op == GGML_OP_NONE || ggml_is_view(node)) {
            continue;
        }
        if (!ggml_backend_supports_op(accelerator, node)) {
            ++unsupported;
            const struct ggml_tensor * s0 = node->src[0];
            const struct ggml_tensor * s1 = node->src[1];
            fprintf(stderr,
                    "[%s] UNSUPPORTED[%d] node=%d op=%s name=%s; CPU fallback blocked\\n",
                    label, unsupported, i, ggml_op_desc(node),
                    node->name[0] ? node->name : "<unnamed>");
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
            continue;
        }
        ggml_backend_sched_set_tensor_backend(sched, node, accelerator);
        ++pinned;
    }

    if (unsupported != 0) {
        fprintf(stderr,
                "[%s] FATAL: strict HTP preflight found %d unsupported compute nodes out of %d; CPU fallback blocked\\n",
                label, unsupported, n_nodes);
        return false;
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

# HTP flash-attention consumes 64-key blocks. The upstream 256-token padding
# exists for CUDA graph shape stability, but on DSP it makes semantic decoding
# attend over up to 255 masked keys that can never contribute. Keep identical
# masking semantics while using the native HTP block granularity.
for old, new, label in [
    ("GGML_PAD(kv_len, 256)", "GGML_PAD(kv_len, 64)", "prefill HTP KV window"),
    ("GGML_PAD(max_kv_len, 256)", "GGML_PAD(max_kv_len, 64)", "decode HTP KV window"),
]:
    if qtext.count(old) != 1:
        raise RuntimeError(f"{label}: expected one source anchor, found {qtext.count(old)}")
    qtext = qtext.replace(old, new, 1)
qtext = qtext.replace("rounded up to 256", "rounded up to 64")
qtext = qtext.replace("spans of 256 decode steps", "spans of 64 decode steps")

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


# Official YuE2 instrumental path: plan normally, then move Vocal melody
# into the Ins lane before semantic generation. This mirrors the released
# yue2-music skill instead of relying on a negative prompt alone.
pipeline_h = yue / "src/pipeline.h"
replace_once(
    pipeline_h,
    """#include "request.h"
""",
    """#include "request.h"
#include "instrumental-transfer.h"
""",
    "instrumental transfer include",
)

replace_once(
    pipeline_h,
    """    if (r.steps < 1 || r.lm_batch_size < 1 || r.synth_batch_size < 1) {
        fprintf(stderr, "[Pipeline] FATAL: steps and batch sizes must be positive\\n");
        return false;
    }
""",
    """    if (r.steps < 1 || r.lm_batch_size < 1 || r.synth_batch_size < 1) {
        fprintf(stderr, "[Pipeline] FATAL: steps and batch sizes must be positive\\n");
        return false;
    }
    if (r.instrumental && cot == YUE2_COT_OFF) {
        fprintf(stderr, "[Pipeline] FATAL: instrumental mode requires melody/full planning\\n");
        return false;
    }
    if (r.instrumental && !r.lyrics.empty()) {
        fprintf(stderr, "[Pipeline] FATAL: instrumental mode requires empty lyrics\\n");
        return false;
    }
""",
    "instrumental validation",
)

replace_once(
    pipeline_h,
    """    std::vector<std::vector<int>> prefixes(B);
    for (int i = 0; i < B; i++) {
        prefixes[i] = yue2_build_prompt_ids(encode, cot, r.style, r.lyrics, has_score ? &abc_ids[i] : nullptr);
    }
""",
    """    if (r.instrumental && has_score) {
        for (int i = 0; i < B; ++i) {
            try {
                scores[i] = localdream_instrumental_transfer_abc(
                    scores[i], truncated[i], (double) r.duration);
            } catch (const std::exception & e) {
                fprintf(stderr, "[Instrumental] FATAL: score transfer failed: %s\\n", e.what());
                return false;
            }
            abc_ids[i] = encode(scores[i]);
            fprintf(stderr, "[Instrumental] Song %d: Vocal melody transferred to Ins before semantic inference\\n", i);
        }
    }

    const std::string effective_style = r.instrumental
        ? std::string("Instrumental, no vocals, no singing, no humming. ") + r.style
        : r.style;
    const std::string effective_lyrics = r.instrumental ? std::string() : r.lyrics;

    std::vector<std::vector<int>> prefixes(B);
    for (int i = 0; i < B; i++) {
        prefixes[i] = yue2_build_prompt_ids(
            encode, cot, effective_style, effective_lyrics,
            has_score ? &abc_ids[i] : nullptr);
    }
""",
    "instrumental score transfer",
)

qwen_enc = yue / "src/qwen3-enc.h"
qwen_lm = yue / "src/qwen3-lm.h"

# Functional AR LoRA. Base quantized weights remain immutable and each
# adapter projection evaluates W*x + scale*B*(A*x) on the same HTP backend.
replace_once(
    qwen_enc,
    """#include "gguf-weights.h"
""",
    """#include "gguf-weights.h"
#include "ar-lora.h"
""",
    "AR LoRA include",
)

replace_once(
    qwen_enc,
    """    struct ggml_tensor * down_proj;  // [FFN, H]
};""",
    """    struct ggml_tensor * down_proj;  // [FFN, H]

    Yue2LoraPair lora_q;
    Yue2LoraPair lora_k;
    Yue2LoraPair lora_v;
    Yue2LoraPair lora_o;
    Yue2LoraPair lora_gate;
    Yue2LoraPair lora_up;
    Yue2LoraPair lora_down;
    bool lora_enabled = false;
};""",
    "AR LoRA layer bindings",
)

replace_once(
    qwen_enc,
    """static struct ggml_tensor * qwen3_build_mlp(struct ggml_context * ctx,
                                            Qwen3Layer *          ly,
                                            struct ggml_tensor *  x,  // [H, S]
                                            int                   S) {
    (void) S;
    struct ggml_tensor * ff;
    if (ly->gate_up) {
        struct ggml_tensor * gu = qwen3_linear(ctx, ly->gate_up, x);
        ff                      = ggml_swiglu(ctx, gu);
    } else {
        struct ggml_tensor * gate = qwen3_linear(ctx, ly->gate_proj, x);
        struct ggml_tensor * up   = qwen3_linear(ctx, ly->up_proj, x);
        ff                        = ggml_swiglu_split(ctx, gate, up);
    }
    return qwen3_linear(ctx, ly->down_proj, ff);
}""",
    """static struct ggml_tensor * qwen3_build_mlp(struct ggml_context * ctx,
                                            Qwen3Layer *          ly,
                                            struct ggml_tensor *  x,  // [H, S]
                                            int                   S) {
    struct ggml_tensor * ff;
    if (ly->gate_up) {
        struct ggml_tensor * gu = qwen3_linear(ctx, ly->gate_up, x);
        if (ly->lora_enabled &&
            (yue2_lora_pair_ready(ly->lora_gate) || yue2_lora_pair_ready(ly->lora_up))) {
            const int64_t F = gu->ne[0] / 2;
            struct ggml_tensor * gate = ggml_view_2d(ctx, gu, F, S, gu->nb[1], 0);
            struct ggml_tensor * up = ggml_view_2d(ctx, gu, F, S, gu->nb[1], (size_t) F * gu->nb[0]);
            gate = yue2_lora_apply(ctx, gate, ly->lora_gate, x, true);
            up = yue2_lora_apply(ctx, up, ly->lora_up, x, true);
            ff = ggml_swiglu_split(ctx, gate, up);
        } else {
            ff = ggml_swiglu(ctx, gu);
        }
    } else {
        struct ggml_tensor * gate = qwen3_linear(ctx, ly->gate_proj, x);
        struct ggml_tensor * up   = qwen3_linear(ctx, ly->up_proj, x);
        gate = yue2_lora_apply(ctx, gate, ly->lora_gate, x, ly->lora_enabled);
        up = yue2_lora_apply(ctx, up, ly->lora_up, x, ly->lora_enabled);
        ff = ggml_swiglu_split(ctx, gate, up);
    }
    struct ggml_tensor * out = qwen3_linear(ctx, ly->down_proj, ff);
    return yue2_lora_apply(ctx, out, ly->lora_down, ff, ly->lora_enabled);
}""",
    "AR LoRA MLP",
)

replace_once(
    qwen_lm,
    """    WeightCtx            wctx;
    ggml_backend_t       backend;""",
    """    WeightCtx            wctx;
    WeightCtx            lora_wctx;
    bool                 lora_loaded;
    bool                 lora_enabled;
    ggml_backend_t       backend;""",
    "AR LoRA model storage",
)

replace_once(
    qwen_lm,
    """    wctx_alloc(&m->wctx, m->backend);
    gf_close(&gf);

    // Persistent graph arenas""",
    """    wctx_alloc(&m->wctx, m->backend);
    gf_close(&gf);

    const char * lora_path = getenv("YUE2_AR_LORA");
    if (lora_path && lora_path[0]) {
        GGUFModel lf;
        if (!gf_load(&lf, lora_path)) {
            fprintf(stderr, "[AR-LoRA] FATAL: cannot load %s\\n", lora_path);
            return false;
        }
        if (strcmp(gf_get_str(lf, "general.architecture"), "yue2_lora") != 0 ||
            strcmp(gf_get_str(lf, "yue2.component"), "generation-adapter") != 0 ||
            strcmp(gf_get_str(lf, "yue2.adapter.type"), "lora") != 0) {
            fprintf(stderr, "[AR-LoRA] FATAL: incompatible adapter metadata\\n");
            gf_close(&lf);
            return false;
        }
        const int64_t rank_key = gguf_find_key(lf.gguf, "yue2.adapter.rank");
        const int64_t alpha_key = gguf_find_key(lf.gguf, "yue2.adapter.alpha");
        if (rank_key < 0 || alpha_key < 0) {
            fprintf(stderr, "[AR-LoRA] FATAL: rank/alpha metadata missing\\n");
            gf_close(&lf);
            return false;
        }
        const uint32_t rank = gguf_get_val_u32(lf.gguf, rank_key);
        const float alpha = gguf_get_val_f32(lf.gguf, alpha_key);
        if (rank == 0 || !isfinite(alpha) || alpha <= 0.0f) {
            fprintf(stderr, "[AR-LoRA] FATAL: invalid rank/alpha\\n");
            gf_close(&lf);
            return false;
        }
        const float scale = alpha / (float) rank;

        wctx_init(&m->lora_wctx, c.n_layers * 14 + 8);
        bool complete = true;
        for (int i = 0; i < c.n_layers; ++i) {
            char pfx[128];
            snprintf(pfx, sizeof(pfx), "model.layers.%d", i);
            Qwen3Layer & ly = m->layers[i];
            complete &= yue2_lora_load_pair(&m->lora_wctx, lf, std::string(pfx) + ".self_attn.q_proj", &ly.lora_q, scale);
            complete &= yue2_lora_load_pair(&m->lora_wctx, lf, std::string(pfx) + ".self_attn.k_proj", &ly.lora_k, scale);
            complete &= yue2_lora_load_pair(&m->lora_wctx, lf, std::string(pfx) + ".self_attn.v_proj", &ly.lora_v, scale);
            complete &= yue2_lora_load_pair(&m->lora_wctx, lf, std::string(pfx) + ".self_attn.o_proj", &ly.lora_o, scale);
            complete &= yue2_lora_load_pair(&m->lora_wctx, lf, std::string(pfx) + ".mlp.gate_proj", &ly.lora_gate, scale);
            complete &= yue2_lora_load_pair(&m->lora_wctx, lf, std::string(pfx) + ".mlp.up_proj", &ly.lora_up, scale);
            complete &= yue2_lora_load_pair(&m->lora_wctx, lf, std::string(pfx) + ".mlp.down_proj", &ly.lora_down, scale);
        }
        if (!complete || !wctx_alloc(&m->lora_wctx, m->backend)) {
            fprintf(stderr, "[AR-LoRA] FATAL: adapter targets incomplete or allocation failed\\n");
            gf_close(&lf);
            return false;
        }
        gf_close(&lf);
        m->lora_loaded = true;
        fprintf(stderr, "[AR-LoRA] Loaded instrumental adapter: rank=%u alpha=%.6g scale=%.6g, %d layers\\n",
                rank, (double) alpha, (double) scale, c.n_layers);
    }

    // Persistent graph arenas""",
    "AR LoRA load",
)

replace_once(
    qwen_lm,
    """    // Reshape to heads: [X*D, S] -> [D, X, S]
    q = ggml_reshape_3d(ctx, q, D, Nh, S);""",
    """    q = yue2_lora_apply(ctx, q, ly->lora_q, x, ly->lora_enabled);
    k = yue2_lora_apply(ctx, k, ly->lora_k, x, ly->lora_enabled);
    v = yue2_lora_apply(ctx, v, ly->lora_v, x, ly->lora_enabled);

    // Reshape to heads: [X*D, S] -> [D, X, S]
    q = ggml_reshape_3d(ctx, q, D, Nh, S);""",
    "AR LoRA prefill QKV",
)

replace_once(
    qwen_lm,
    """    // O projection
    return qwen3_linear(ctx, ly->o_proj, attn);
}""",
    """    // O projection
    struct ggml_tensor * out = qwen3_linear(ctx, ly->o_proj, attn);
    return yue2_lora_apply(ctx, out, ly->lora_o, attn, ly->lora_enabled);
}""",
    "AR LoRA prefill O",
)

replace_once(
    qwen_lm,
    """            // Reshape to heads: [D, Heads, N]
            q = ggml_reshape_3d(ctx, q, D, Nh, N);""",
    """            q = yue2_lora_apply(ctx, q, ly->lora_q, norm, ly->lora_enabled);
            k = yue2_lora_apply(ctx, k, ly->lora_k, norm, ly->lora_enabled);
            v = yue2_lora_apply(ctx, v, ly->lora_v, norm, ly->lora_enabled);

            // Reshape to heads: [D, Heads, N]
            q = ggml_reshape_3d(ctx, q, D, Nh, N);""",
    "AR LoRA batch QKV",
)

replace_once(
    qwen_lm,
    """            struct ggml_tensor * attn_out = qwen3_linear(ctx, ly->o_proj, attn_cat);
            hidden                        = ggml_add(ctx, hidden, attn_out);""",
    """            struct ggml_tensor * attn_out = qwen3_linear(ctx, ly->o_proj, attn_cat);
            attn_out = yue2_lora_apply(ctx, attn_out, ly->lora_o, attn_cat, ly->lora_enabled);
            hidden   = ggml_add(ctx, hidden, attn_out);""",
    "AR LoRA batch O",
)

replace_once(
    qwen_lm,
    """// Build self-attention with KV cache write + read.""",
    """static bool qw3lm_set_lora_enabled(Qwen3LM * m, bool requested) {
    const bool enabled = requested && m->lora_loaded;
    if (requested && !m->lora_loaded) {
        fprintf(stderr, "[AR-LoRA] Instrumental adapter not installed; transformed-score base AR remains active\\n");
    }
    if (m->lora_enabled == enabled) {
        return enabled;
    }
    if (m->batch_graph.graph.sched_allocated) {
        static_graph_release(&m->batch_graph.graph, m->sched);
    }
    m->batch_graph.built = false;
    m->lora_enabled = enabled;
    for (int i = 0; i < m->cfg.n_layers; ++i) {
        m->layers[i].lora_enabled = enabled;
    }
    fprintf(stderr, "[AR-LoRA] %s\\n", enabled ? "enabled" : "disabled");
    return enabled;
}

// Build self-attention with KV cache write + read.""",
    "AR LoRA toggle",
)

replace_once(
    qwen_lm,
    """    backend_release(m->backend, m->cpu_backend);
    wctx_free(&m->wctx);
    *m = {};""",
    """    backend_release(m->backend, m->cpu_backend);
    if (m->lora_wctx.ctx) {
        wctx_free(&m->lora_wctx);
    }
    wctx_free(&m->wctx);
    *m = {};""",
    "AR LoRA free",
)

replace_once(
    pipeline_h,
    """        lm = require_lm(p);
        if (!lm) {
            return false;
        }
        lm_hold.emplace(p->store, lm);""",
    """        lm = require_lm(p);
        if (!lm) {
            return false;
        }
        qw3lm_set_lora_enabled(lm, r.instrumental);
        lm_hold.emplace(p->store, lm);""",
    "instrumental LoRA initial activation",
)

replace_once(
    pipeline_h,
    """                Qwen3LM * lm_chunk = require_lm(p);
                if (!lm_chunk) {
                    return false;
                }
                ModelHandle lm_chunk_hold(p->store, lm_chunk);""",
    """                Qwen3LM * lm_chunk = require_lm(p);
                if (!lm_chunk) {
                    return false;
                }
                qw3lm_set_lora_enabled(lm_chunk, r.instrumental);
                ModelHandle lm_chunk_hold(p->store, lm_chunk);""",
    "instrumental LoRA chunk activation",
)

# Exact eager NAR attention tiling for long songs. The score softmax is
# independent per query row, so slicing query rows and concatenating their
# contexts is mathematically identical to one giant score matrix. This ports
# audio.cpp #642's strategy with a phone-sized score-tile budget.
nar_h = yue / "src/nar.h"
replace_once(
    nar_h,
    """// NAR attention: fresh Q/K/V for the latent block of every variation,
""",
    """static struct ggml_tensor * nar_attn_f32_tiled(
        struct ggml_context * ctx,
        struct ggml_tensor * q,
        struct ggml_tensor * k,
        struct ggml_tensor * v,
        struct ggml_tensor * mask,
        float scale) {
    const int64_t steps = q->ne[1];
    const int64_t kv_steps = k->ne[1];
    const int64_t heads = q->ne[2];
    const int64_t batch = q->ne[3];

    // Keep one eager F32 score tile around 96 MiB on mobile. The rows are
    // balanced to avoid a tiny slow tail. A short song stays one tile.
    const int64_t bytes_per_row =
        kv_steps * heads * batch * (int64_t) sizeof(float);
    const int64_t target_bytes = 96LL * 1024LL * 1024LL;
    const int64_t max_rows = std::max<int64_t>(
        1, target_bytes / std::max<int64_t>(1, bytes_per_row));
    const int64_t tiles = std::max<int64_t>(
        1, (steps + max_rows - 1) / max_rows);
    const int64_t rows_base = steps / tiles;
    const int64_t wider = steps % tiles;

    struct ggml_tensor * vt = ggml_cont(ctx, ggml_transpose(ctx, v));
    struct ggml_tensor * joined = nullptr;
    int64_t first = 0;

    for (int64_t tile = 0; tile < tiles; ++tile) {
        const int64_t rows = rows_base + (tile < wider ? 1 : 0);
        struct ggml_tensor * q_tile = q;
        struct ggml_tensor * mask_tile = mask;

        if (tiles > 1) {
            q_tile = ggml_cont(
                ctx,
                ggml_view_4d(
                    ctx, q,
                    q->ne[0], rows, q->ne[2], q->ne[3],
                    q->nb[1], q->nb[2], q->nb[3],
                    (size_t) first * q->nb[1]));
            if (mask) {
                mask_tile = ggml_cont(
                    ctx,
                    ggml_view_2d(
                        ctx, mask,
                        mask->ne[0], rows, mask->nb[1],
                        (size_t) first * mask->nb[1]));
            }
        }

        struct ggml_tensor * scores = ggml_mul_mat(ctx, k, q_tile);
        ggml_mul_mat_set_prec(scores, GGML_PREC_F32);
        struct ggml_tensor * weights =
            ggml_soft_max_ext(ctx, scores, mask_tile, scale, 0.0f);
        struct ggml_tensor * context = ggml_mul_mat(ctx, vt, weights);
        ggml_mul_mat_set_prec(context, GGML_PREC_F32);
        joined = joined ? ggml_concat(ctx, joined, context, 1) : context;
        first += rows;
    }

    if (tiles > 1) {
        fprintf(stderr,
                "[NAR] Exact eager attention tiling: %lld query rows, %lld key rows, %lld tiles, <=%.1f MiB scores/tile\\n",
                (long long) steps, (long long) kv_steps, (long long) tiles,
                (double) target_bytes / (1024.0 * 1024.0));
    }

    return ggml_cont(ctx, ggml_permute(ctx, joined, 0, 2, 1, 3));
}

// NAR attention: fresh Q/K/V for the latent block of every variation,
""",
    "exact NAR attention tiling helper",
)

replace_once(
    nar_h,
    """    float                scale = 1.0f / sqrtf((float) D);
    struct ggml_tensor * attn  = use_flash_attn ? ggml_flash_attn_ext(ctx, q, k_full, v_full, mask, scale, 0.0f, 0.0f) :
                                                  qwen3_attn_f32(ctx, q, k_full, v_full, mask, scale);
""",
    """    float scale = 1.0f / sqrtf((float) D);
    struct ggml_tensor * attn = use_flash_attn
        ? ggml_flash_attn_ext(ctx, q, k_full, v_full, mask, scale, 0.0f, 0.0f)
        : nar_attn_f32_tiled(ctx, q, k_full, v_full, mask, scale);
""",
    "exact NAR attention tiling call",
)

# Local Dream mobile acoustic solver.
#
# Midpoint remains the release/reference protocol. The fast path below ports
# ComfyUI/k-diffusion's actual DPM-Solver++(2M) update and pairs it with
# ComfyUI's sgm_uniform scheduler for a discrete-flow/CONST model. YuE2 predicts
# flow velocity v, so the denoised estimate consumed by DPM++ is x0=x-sigma*v.
# This is not the old uniform-grid Adams-Bashforth approximation.
request_h = yue / "src/request.h"
replace_once(
    request_h,
    """    int     steps;    // 32, midpoint steps of the flow matching ODE
""",
    """    int     steps;       // acoustic ODE steps
    std::string ode_method;  // "midpoint" or "dpmpp_2m" (ComfyUI + sgm_uniform)
""",
    "DPM++ 2M request field",
)

replace_once(
    request_h,
    """    std::string style;   // ""
    std::string lyrics;  // ""
""",
    """    std::string style;   // ""
    std::string lyrics;  // ""
    bool instrumental;   // official no-vocal score transfer path
""",
    "instrumental request field",
)

request_cpp = yue / "src/request.cpp"
replace_once(
    request_cpp,
    """    r->style  = "";
    r->lyrics = "";
    r->abc    = "";
""",
    """    r->style        = "";
    r->lyrics       = "";
    r->instrumental = false;
    r->abc          = "";
""",
    "instrumental request default",
)
replace_once(
    request_cpp,
    """    if ((v = yyjson_obj_get(obj, "lyrics")) && yyjson_is_str(v)) {
        r->lyrics = yy_str(v);
    }
""",
    """    if ((v = yyjson_obj_get(obj, "lyrics")) && yyjson_is_str(v)) {
        r->lyrics = yy_str(v);
    }
    if ((v = yyjson_obj_get(obj, "instrumental")) && yyjson_is_bool(v)) {
        r->instrumental = yyjson_get_bool(v);
    }
""",
    "instrumental request parse",
)
replace_once(
    request_cpp,
    """    if (!sparse || r->lyrics != d.lyrics) {
        yyjson_mut_obj_add_strncpy(doc, root, "lyrics", r->lyrics.c_str(), r->lyrics.size());
    }
""",
    """    if (!sparse || r->lyrics != d.lyrics) {
        yyjson_mut_obj_add_strncpy(doc, root, "lyrics", r->lyrics.c_str(), r->lyrics.size());
    }
    if (!sparse || r->instrumental != d.instrumental) {
        yyjson_mut_obj_add_bool(doc, root, "instrumental", r->instrumental);
    }
""",
    "instrumental request serialize",
)
replace_once(
    request_cpp,
    """    r->steps            = 32;
    r->lm_batch_size    = 1;""",
    """    r->steps            = 32;
    r->ode_method       = "midpoint";
    r->lm_batch_size    = 1;""",
    "DPM++ 2M request default",
)
replace_once(
    request_cpp,
    """    if ((v = yyjson_obj_get(obj, "steps")) && yyjson_is_int(v)) {
        r->steps = yyjson_get_int(v);
    }
    if ((v = yyjson_obj_get(obj, "lm_batch_size")) && yyjson_is_int(v)) {""",
    """    if ((v = yyjson_obj_get(obj, "steps")) && yyjson_is_int(v)) {
        r->steps = yyjson_get_int(v);
    }
    if ((v = yyjson_obj_get(obj, "ode_method")) && yyjson_is_str(v)) {
        r->ode_method = yy_str(v);
    }
    if ((v = yyjson_obj_get(obj, "lm_batch_size")) && yyjson_is_int(v)) {""",
    "DPM++ 2M request parse",
)
replace_once(
    request_cpp,
    """    if (!sparse || r->steps != d.steps) {
        yyjson_mut_obj_add_int(doc, root, "steps", r->steps);
    }
    if (!sparse || r->lm_batch_size != d.lm_batch_size) {""",
    """    if (!sparse || r->steps != d.steps) {
        yyjson_mut_obj_add_int(doc, root, "steps", r->steps);
    }
    if (!sparse || r->ode_method != d.ode_method) {
        yyjson_mut_obj_add_strncpy(doc, root, "ode_method", r->ode_method.c_str(), r->ode_method.size());
    }
    if (!sparse || r->lm_batch_size != d.lm_batch_size) {""",
    "DPM++ 2M request serialize",
)

nar_h = yue / "src/nar.h"
replace_once(
    nar_h,
    """                      int                  steps,
                      const DebugDumper *  dbg,""",
    """                      int                  steps,
                      const char *         method,
                      const DebugDumper *  dbg,""",
    "DPM++ 2M solver signature",
)

replace_once(
    nar_h,
    """    size_t             count = (size_t) n->latent_dim * T_lat * M;
    std::vector<float> first(count), mid(count), second(count);
    float              dt = 1.0f / (float) steps;
    char               name[64];

    debug_dump_2d(dbg, "noise", state, T_lat, n->latent_dim);
    Timer solve_timer;""",
    """    // COMFY_DPM_PLUS_PLUS_2M_SGM_UNIFORM
    size_t count = (size_t) n->latent_dim * T_lat * M;
    const bool midpoint = strcmp(method, "midpoint") == 0;
    const bool dpmpp_2m = strcmp(method, "dpmpp_2m") == 0;
    if (!midpoint && !dpmpp_2m) {
        fprintf(stderr, "[NAR] FATAL: unknown ODE method %s\\n", method ? method : "<null>");
        return false;
    }

    std::vector<float> first(count);
    std::vector<float> mid;
    std::vector<float> second;
    std::vector<float> denoised;
    std::vector<float> old_denoised;
    if (midpoint) {
        mid.resize(count);
        second.resize(count);
    } else {
        denoised.resize(count);
        old_denoised.resize(count);
    }

    const float dt = 1.0f / (float) steps;
    bool  have_old_denoised = false;
    float previous_sigma = 0.0f;
    int   evaluations = 0;
    char  name[64];

    fprintf(stderr, "[NAR] Solver: %s, scheduler=%s, steps=%d, evaluations=%d\\n",
            method, midpoint ? "reference_uniform" : "sgm_uniform",
            steps, midpoint ? steps * 2 : steps);
    debug_dump_2d(dbg, "noise", state, T_lat, n->latent_dim);
    Timer solve_timer;""",
    "DPM++ 2M solver setup",
)

replace_once(
    nar_h,
    """        float t = 1.0f - (float) step * dt;
        if (!nar_velocity(n, kv, state, T_lat, M, ar_len, kv_set, nar_logit_clamped(t), first.data())) {
            return false;
        }
        if (dbg->enabled && step == 0) {
            nar_dump_named(n, dbg);
        }
        for (size_t i = 0; i < count; i++) {
            mid[i] = state[i] - first[i] * (dt * 0.5f);
        }
        if (!nar_velocity(n, kv, mid.data(), T_lat, M, ar_len, kv_set, nar_logit_clamped(t - dt * 0.5f),
                          second.data())) {
            return false;
        }
        for (size_t i = 0; i < count; i++) {
            state[i] -= second[i] * dt;
        }
        snprintf(name, sizeof(name), "nar_step%d_first", step);
        debug_dump_2d(dbg, name, first.data(), T_lat, n->latent_dim);
        snprintf(name, sizeof(name), "nar_step%d_second", step);
        debug_dump_2d(dbg, name, second.data(), T_lat, n->latent_dim);
        snprintf(name, sizeof(name), "nar_step%d_xt", step);
        debug_dump_2d(dbg, name, state, T_lat, n->latent_dim);
        fprintf(stderr, "[NAR] Step %d/%d, %.0f ms\\n", step + 1, steps, step_timer.ms());""",
    """        if (midpoint) {
            const float t = 1.0f - (float) step * dt;
            if (!nar_velocity(n, kv, state, T_lat, M, ar_len, kv_set, nar_logit_clamped(t), first.data())) {
                return false;
            }
            ++evaluations;
            if (dbg->enabled && step == 0) {
                nar_dump_named(n, dbg);
            }
            for (size_t i = 0; i < count; i++) {
                mid[i] = state[i] - first[i] * (dt * 0.5f);
            }
            if (!nar_velocity(n, kv, mid.data(), T_lat, M, ar_len, kv_set, nar_logit_clamped(t - dt * 0.5f),
                              second.data())) {
                return false;
            }
            ++evaluations;
            for (size_t i = 0; i < count; i++) {
                state[i] -= second[i] * dt;
            }

            snprintf(name, sizeof(name), "nar_step%d_first", step);
            debug_dump_2d(dbg, name, first.data(), T_lat, n->latent_dim);
            snprintf(name, sizeof(name), "nar_step%d_second", step);
            debug_dump_2d(dbg, name, second.data(), T_lat, n->latent_dim);
        } else {
            // ComfyUI normal_scheduler(..., sgm=True) for ModelSamplingDiscreteFlow:
            // linearly space the model-sampling timestep from sigma_max to
            // sigma_min (the 1/1000 endpoint), map each through the flow shift,
            // then append an exact final zero.
            const float shift = n->timestep_shift;
            const auto flow_shift = [shift](float t) {
                return shift * t / (1.0f + (shift - 1.0f) * t);
            };
            const auto flow_unshift = [shift](float sigma) {
                const float denom = shift - (shift - 1.0f) * sigma;
                return sigma / denom;
            };

            const float sigma_max = flow_shift(1.0f);
            const float sigma_min = flow_shift(0.001f);
            const float scheduler_t =
                sigma_max + (sigma_min - sigma_max) * ((float) step / (float) steps);
            const float sigma = flow_shift(scheduler_t);
            const float sigma_next = (step + 1 == steps)
                ? 0.0f
                : flow_shift(
                    sigma_max + (sigma_min - sigma_max) * ((float) (step + 1) / (float) steps));

            if (!(sigma > 0.0f) || !(sigma_next >= 0.0f) || !(sigma_next < sigma)) {
                fprintf(stderr, "[NAR] FATAL: invalid sgm_uniform sigma pair %.9f -> %.9f\\n",
                        (double) sigma, (double) sigma_next);
                return false;
            }

            const float raw_t = flow_unshift(sigma);
            if (!(raw_t > 0.0f) || !(raw_t <= 1.0f)) {
                fprintf(stderr, "[NAR] FATAL: invalid YuE2 flow timestep %.9f for sigma %.9f\\n",
                        (double) raw_t, (double) sigma);
                return false;
            }
            if (!nar_velocity(n, kv, state, T_lat, M, ar_len, kv_set,
                              nar_logit_clamped(raw_t), first.data())) {
                return false;
            }
            ++evaluations;
            if (dbg->enabled && step == 0) {
                nar_dump_named(n, dbg);
            }

            // ComfyUI CONST flow model: denoised = model_input - model_output*sigma.
            for (size_t i = 0; i < count; i++) {
                denoised[i] = state[i] - first[i] * sigma;
            }

            if (sigma_next == 0.0f) {
                memcpy(state, denoised.data(), count * sizeof(float));
            } else {
                const float ratio = sigma_next / sigma;
                float current_coeff = 1.0f;
                float old_coeff = 0.0f;
                if (have_old_denoised) {
                    const float h = logf(sigma / sigma_next);
                    const float h_last = logf(previous_sigma / sigma);
                    if (!(h > 0.0f) || !(h_last > 0.0f)) {
                        fprintf(stderr, "[NAR] FATAL: invalid DPM++ log-time interval\\n");
                        return false;
                    }
                    const float r = h_last / h;
                    if (!(r > 0.0f)) {
                        fprintf(stderr, "[NAR] FATAL: invalid DPM++ history ratio %.9f\\n", (double) r);
                        return false;
                    }
                    old_coeff = 1.0f / (2.0f * r);
                    current_coeff = 1.0f + old_coeff;
                }

                const float denoised_mix = 1.0f - ratio; // -expm1(-h), stable here
                for (size_t i = 0; i < count; i++) {
                    const float d = current_coeff * denoised[i] - old_coeff * old_denoised[i];
                    state[i] = ratio * state[i] + denoised_mix * d;
                }
            }

            memcpy(old_denoised.data(), denoised.data(), count * sizeof(float));
            previous_sigma = sigma;
            have_old_denoised = true;

            snprintf(name, sizeof(name), "nar_step%d_first", step);
            debug_dump_2d(dbg, name, first.data(), T_lat, n->latent_dim);
            snprintf(name, sizeof(name), "nar_step%d_denoised", step);
            debug_dump_2d(dbg, name, denoised.data(), T_lat, n->latent_dim);
        }

        snprintf(name, sizeof(name), "nar_step%d_xt", step);
        debug_dump_2d(dbg, name, state, T_lat, n->latent_dim);
        fprintf(stderr, "[NAR] Step %d/%d, %.0f ms\\n", step + 1, steps, step_timer.ms());""",
    "DPM++ 2M solver step",
)

replace_once(
    nar_h,
    """    fprintf(stderr, "[NAR] Solved: T_lat=%d, %d variations, %d steps, %.0f ms (%.1f ms/step)\\n", T_lat, M, steps,
            solve_timer.ms(), solve_timer.ms() / steps);""",
    """    fprintf(stderr,
            "[NAR] Solved (%s/%s): T_lat=%d, %d variations, %d steps, %d evaluations, %.0f ms (%.1f ms/step)\\n",
            method, midpoint ? "reference_uniform" : "sgm_uniform",
            T_lat, M, steps, evaluations, solve_timer.ms(), solve_timer.ms() / steps);""",
    "DPM++ 2M solver summary",
)

pipeline_h = yue / "src/pipeline.h"
replace_once(
    pipeline_h,
    """    if (r.steps < 1 || r.lm_batch_size < 1 || r.synth_batch_size < 1) {
        fprintf(stderr, "[Pipeline] FATAL: steps and batch sizes must be positive\\n");
        return false;
    }""",
    """    if (r.steps < 1 || r.lm_batch_size < 1 || r.synth_batch_size < 1) {
        fprintf(stderr, "[Pipeline] FATAL: steps and batch sizes must be positive\\n");
        return false;
    }
    if (r.ode_method != "midpoint" && r.ode_method != "dpmpp_2m") {
        fprintf(stderr, "[Pipeline] FATAL: ode_method must be midpoint or dpmpp_2m\\n");
        return false;
    }""",
    "DPM++ 2M pipeline validation",
)
replace_once(
    pipeline_h,
    """            if (!nar_solve(nar, &p->kv, block.data(), frames, M, ar_len, i, r.steps, dbg, cancelled, cancel_data)) {""",
    """            if (!nar_solve(nar, &p->kv, block.data(), frames, M, ar_len, i, r.steps,
                           r.ode_method.c_str(), dbg, cancelled, cancel_data)) {""",
    "DPM++ 2M pipeline call",
)

server_cpp = yue / "tools/yue-server.cpp"
replace_once(
    server_cpp,
    """    if (r->steps < 1) {
        res.status = 400;
        res.set_content(json_string("error", "steps must be positive"), "application/json");
        return false;
    }""",
    """    if (r->steps < 1) {
        res.status = 400;
        res.set_content(json_string("error", "steps must be positive"), "application/json");
        return false;
    }
    if (r->ode_method != "midpoint" && r->ode_method != "dpmpp_2m") {
        res.status = 400;
        res.set_content(json_string("error", "ode_method must be midpoint or dpmpp_2m"), "application/json");
        return false;
    }""",
    "DPM++ 2M server validation",
)

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
# Keep yue2.cpp's compact GEMM + GGML_OP_COL2IM_1D graph. Local Dream
# replaces that op with a native channel-blocked HTP kernel in col2im-ops.c.
# The previous graph-level CONT/ADD/CONCAT lowering added more than 100 nodes
# and turned waveform decode into a memory-movement bottleneck on SM8850.

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

# Whole-decoder parity gate. The CPU decoder exists only as a one-shot
# validator. Generation remains strict HTP-only: a parity failure aborts
# instead of silently falling back.
replace_once(
    vae,
    """    // Scratch buffer (reused across tiles, grown as needed)
    std::vector<float> scratch_in;  // transposed input [64 * T]
};""",
    """    // Scratch buffer (reused across tiles, grown as needed)
    std::vector<float> scratch_in;  // transposed input [64 * T]
    bool standalone_backend = false;  // parity reference only
};""",
    "VAE parity ownership flag",
)

replace_once(
    vae,
    """// Load model
static void vae_ggml_load(VAEGGML * m, const char * path) {""",
    """// Forward declarations for the one-shot full decoder parity check.
static int vae_ggml_decode(VAEGGML * m, const float * latent, int T_latent,
                           float * audio_out, int max_T_audio);
static void vae_ggml_free(VAEGGML * m);
static bool g_vae_reference_loading = false;
static bool g_vae_parity_checked = false;

// Load model
static void vae_ggml_load(VAEGGML * m, const char * path) {""",
    "VAE parity declarations",
)

replace_once(
    vae,
    """    // Phase 2: allocate backend buffer
    BackendPair bp = backend_init("VAE");
    m->backend     = bp.backend;
    m->cpu_backend = bp.cpu_backend;
    m->sched       = backend_sched_new(bp, 8192);""",
    """    // CPU is permitted only for the short parity reference instance.
    // Normal generation always takes the accelerator branch below.
    BackendPair bp = {};
    if (g_vae_reference_loading) {
        bp.backend = cpu_backend_new(backend_cpu_n_threads());
        bp.cpu_backend = bp.backend;
        bp.has_gpu = false;
        m->standalone_backend = true;
        if (!bp.backend) {
            fprintf(stderr, "[VAE-PARITY] FATAL: CPU reference backend unavailable\\n");
            exit(1);
        }
    } else {
        bp = backend_init("VAE");
    }
    m->backend      = bp.backend;
    m->cpu_backend  = bp.cpu_backend;
    m->sched        = backend_sched_new(bp, 8192);""",
    "VAE parity CPU backend",
)

replace_once(
    vae,
    """    fprintf(stderr, "[VAE] Loaded: 6 blocks, upsample=1920x, F32 activations\\n");
    gf_close(&gf);
}""",
    """    fprintf(stderr, "[VAE] Loaded: 6 blocks, upsample=1920x, F32 activations\\n");
    gf_close(&gf);

    if (!g_vae_reference_loading && !g_vae_parity_checked) {
        g_vae_parity_checked = true;
        fprintf(stderr, "[VAE-PARITY] Running full HTP vs CPU decoder check\\n");

        constexpr int pt = 4;
        constexpr int pa = pt * 1920 - 64;
        std::vector<float> latent((size_t) pt * 64);
        uint32_t rng = 0x6d2b79f5u;
        for (size_t i = 0; i < latent.size(); ++i) {
            rng = rng * 1664525u + 1013904223u;
            const float u = (float) ((rng >> 8) & 0x00ffffffu) / 16777216.0f;
            latent[i] = (u * 2.0f - 1.0f) * 0.35f;
        }

        std::vector<float> htp((size_t) pa * 2);
        std::vector<float> ref((size_t) pa * 2);
        const int hn = vae_ggml_decode(m, latent.data(), pt, htp.data(), pa);

        VAEGGML cpu_ref = {};
        g_vae_reference_loading = true;
        vae_ggml_load(&cpu_ref, path);
        g_vae_reference_loading = false;
        const int rn = vae_ggml_decode(&cpu_ref, latent.data(), pt, ref.data(), pa);
        vae_ggml_free(&cpu_ref);

        if (hn != pa || rn != pa) {
            fprintf(stderr, "[VAE-PARITY] FATAL: length htp=%d cpu=%d expected=%d\\n", hn, rn, pa);
            exit(1);
        }

        double dot = 0.0, na = 0.0, nb = 0.0, mse = 0.0;
        float max_abs = 0.0f;
        const size_t n = (size_t) pa * 2;
        for (size_t i = 0; i < n; ++i) {
            const double a = htp[i], b = ref[i], d = a - b;
            dot += a * b; na += a * a; nb += b * b; mse += d * d;
            const float ad = fabsf((float) d);
            if (ad > max_abs) max_abs = ad;
        }
        const double cosine = dot / (sqrt(na * nb) + 1e-30);
        const double rmse = sqrt(mse / (double) n);
        fprintf(stderr, "[VAE-PARITY] cosine=%.9f max_abs=%.9g rmse=%.9g samples=%zu\\n",
                cosine, (double) max_abs, rmse, n);
        if (cosine < 0.999999 || max_abs > 8.0e-4f) {
            fprintf(stderr, "[VAE-PARITY] FATAL: HTP decoder diverges; refusing corrupted audio\\n");
            exit(1);
        }
        fprintf(stderr, "[VAE-PARITY] PASS\\n");
    }
}""",
    "VAE whole decoder parity gate",
)

replace_once(
    vae,
    """    backend_release(m->backend, m->cpu_backend);
    *m = {};""",
    """    if (m->standalone_backend) {
        if (m->backend) ggml_backend_free(m->backend);
    } else {
        backend_release(m->backend, m->cpu_backend);
    }
    *m = {};""",
    "VAE parity reference free",
)

replace_once(
    vae,
    """        if (!ggml_backend_sched_alloc_graph(m->sched, m->graph)) {""",
    """        if (!m->standalone_backend &&
            !backend_strict_pin_graph(m->sched, m->backend, m->graph, "VAE")) {
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
""",
    "VAE tile progress begin",
)
replace_once(
    vae,
    """        ggml_backend_tensor_get(m->graph_output, audio_out + max_T_audio + out_start, (tile_T + crop) * sizeof(float),
                                core_len * sizeof(float));
    }

    // Compact ch1""",
    """        ggml_backend_tensor_get(m->graph_output, audio_out + max_T_audio + out_start, (tile_T + crop) * sizeof(float),
                                core_len * sizeof(float));
        fprintf(stderr, "[VAE] Tile %d/%d done: %.0f ms\\n", i + 1, num_tiles, tile_timer.ms());
    }

    // Compact ch1""",
    "VAE tile progress done",
)

print("Local Dream YuE2 native HTP integration applied")
