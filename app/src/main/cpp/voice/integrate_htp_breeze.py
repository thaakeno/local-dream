#!/usr/bin/env python3
"""Compile-time Breeze / ggml-hexagon integration for Local Dream.

Pinned upstream revisions only. This adds the Breeze vocoder ops that the
generic Hexagon binary path cannot execute efficiently at waveform scale:
streaming channel-broadcast ADD/MUL, SIN, COL2IM_1D, fused COL2IM+bias,
fused Snake, and adaptive VTCM sizing. No CPU/GPU compute fallback.
"""
from pathlib import Path
import shutil
import sys

if len(sys.argv) != 4:
    raise SystemExit("usage: integrate_htp_breeze.py <hexagon-root> <breeze-root> <overlay-dir>")

root = Path(sys.argv[1])
breeze = Path(sys.argv[2])
overlay = Path(sys.argv[3])
htp = root / "ggml/src/ggml-hexagon/htp"
host = root / "ggml/src/ggml-hexagon/ggml-hexagon.cpp"
codec = breeze / "src/codec_decoder.cpp"
backbone_h = breeze / "include/breeze/backbone.h"
backbone_cpp = breeze / "src/backbone.cpp"
generation = breeze / "src/generation.cpp"

def replace_once(path: Path, old: str, new: str, label: str) -> None:
    text = path.read_text()
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one source anchor, found {count} in {path}")
    path.write_text(text.replace(old, new, 1))

for src in (
    "breeze-sin-ops.c",
    "breeze-col2im-ops.c",
    "breeze-channel-bcast-ops.c",
    "breeze-snake-ops.c",
):
    shutil.copy2(overlay / src, htp / src)

replace_once(
    htp / "CMakeLists.txt",
    "    im2col-ops.c\n    roll-ops.c",
    "    im2col-ops.c\n"
    "    breeze-col2im-ops.c\n"
    "    breeze-sin-ops.c\n"
    "    breeze-snake-ops.c\n"
    "    breeze-channel-bcast-ops.c\n"
    "    roll-ops.c",
    "Breeze HTP source list",
)

replace_once(
    htp / "htp-ops.h",
    "    HTP_OP_ARGMAX,\n    HTP_OP_UNARY_GELU_ERF,",
    "    HTP_OP_ARGMAX,\n"
    "    HTP_OP_SIN,\n"
    "    HTP_OP_COL2IM_1D,\n"
    "    HTP_OP_COL2IM_1D_BIAS,\n"
    "    HTP_OP_SNAKE,\n"
    "    HTP_OP_CHANNEL_BCAST_ADD,\n"
    "    HTP_OP_CHANNEL_BCAST_MUL,\n"
    "    HTP_OP_UNARY_GELU_ERF,",
    "Breeze HTP op enum",
)

replace_once(
    htp / "htp-ctx.h",
    "int op_im2col(struct htp_ops_context * octx);\nint op_allreduce",
    "int op_im2col(struct htp_ops_context * octx);\n"
    "int op_col2im_1d(struct htp_ops_context * octx);\n"
    "int op_col2im_1d_bias(struct htp_ops_context * octx);\n"
    "int op_sin(struct htp_ops_context * octx);\n"
    "int op_snake(struct htp_ops_context * octx);\n"
    "int op_channel_bcast_add(struct htp_ops_context * octx);\n"
    "int op_channel_bcast_mul(struct htp_ops_context * octx);\n"
    "int op_allreduce",
    "Breeze HTP declarations",
)

replace_once(
    htp / "main.c",
    """        case HTP_OP_IM2COL:
            return op_im2col(octx);

        case HTP_OP_ROLL:""",
    """        case HTP_OP_IM2COL:
            return op_im2col(octx);

        case HTP_OP_COL2IM_1D:
            return op_col2im_1d(octx);

        case HTP_OP_COL2IM_1D_BIAS:
            return op_col2im_1d_bias(octx);

        case HTP_OP_SIN:
            return op_sin(octx);

        case HTP_OP_SNAKE:
            return op_snake(octx);

        case HTP_OP_CHANNEL_BCAST_ADD:
            return op_channel_bcast_add(octx);

        case HTP_OP_CHANNEL_BCAST_MUL:
            return op_channel_bcast_mul(octx);

        case HTP_OP_ROLL:""",
    "Breeze HTP dispatch",
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
    return t_out > 0 && op->ne[0] == t_out && op->ne[1] == oc &&
           op->ne[2] == 1 && op->ne[3] == 1;
}

static bool ggml_hexagon_is_breeze_channel_binary(const struct ggml_tensor * op) {
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

    // Once a waveform row reaches ~1 MiB, streaming directly from mapped DDR
    // is both faster and avoids the generic binary kernel's VTCM row staging.
    const bool wide_residual = same_shape && src0->ne[0] >= 262144;
    return channel_broadcast || wide_residual;
}

"""
replace_once(
    host,
    "static bool ggml_hexagon_supported_sum(const struct ggml_hexagon_session * sess, const struct ggml_tensor * op) {",
    helpers + "static bool ggml_hexagon_supported_sum(const struct ggml_hexagon_session * sess, const struct ggml_tensor * op) {",
    "Breeze host helpers",
)

replace_once(
    host,
    "        case GGML_OP_MUL:             return HTP_OP_MUL;\n        case GGML_OP_ADD:             return HTP_OP_ADD;",
    "        case GGML_OP_MUL:             return ggml_hexagon_is_breeze_channel_binary(t) ? HTP_OP_CHANNEL_BCAST_MUL : HTP_OP_MUL;\n"
    "        case GGML_OP_ADD:             return ggml_hexagon_is_breeze_channel_binary(t) ? HTP_OP_CHANNEL_BCAST_ADD : HTP_OP_ADD;",
    "Breeze channel binary remap",
)
replace_once(
    host,
    "        case GGML_OP_SQR:             return HTP_OP_SQR;",
    "        case GGML_OP_SIN:             return HTP_OP_SIN;\n        case GGML_OP_SQR:             return HTP_OP_SQR;",
    "Breeze SIN remap",
)
replace_once(
    host,
    "        case GGML_OP_IM2COL:          return HTP_OP_IM2COL;",
    "        case GGML_OP_IM2COL:          return HTP_OP_IM2COL;\n        case GGML_OP_COL2IM_1D:       return HTP_OP_COL2IM_1D;",
    "Breeze COL2IM remap",
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
            supp = ggml_hexagon_is_breeze_channel_binary(op) ||
                   ggml_hexagon_supported_binary(sess, op);
            break;""",
    "Breeze channel binary support",
)
replace_once(
    host,
    """        case GGML_OP_SQR:
        case GGML_OP_SQRT:
        case GGML_OP_LOG:""",
    """        case GGML_OP_SIN:
            supp = ggml_hexagon_supported_sin(sess, op);
            break;

        case GGML_OP_SQR:
        case GGML_OP_SQRT:
        case GGML_OP_LOG:""",
    "Breeze SIN support",
)
replace_once(
    host,
    """        case GGML_OP_IM2COL:
            supp = ggml_hexagon_supported_im2col(sess, op);
            break;""",
    """        case GGML_OP_IM2COL:
            supp = ggml_hexagon_supported_im2col(sess, op);
            break;

        case GGML_OP_COL2IM_1D:
            supp = ggml_hexagon_supported_col2im_1d(sess, op);
            break;""",
    "Breeze COL2IM support",
)

col2im_fusion = r"""
    bool try_fuse_breeze_col2im_bias(const htp_opnode & node) {
        if ((node.opcode != HTP_OP_ADD &&
             node.opcode != HTP_OP_CHANNEL_BCAST_ADD) ||
            n_ops == 0) {
            return false;
        }

        htp_opnode & col = ops[n_ops - 1];
        if (col.opcode != HTP_OP_COL2IM_1D) return false;

        const ggml_tensor * col_out = col.dst();
        const ggml_tensor * out = node.dst();
        if (!col_out || !out || !ggml_hexagon_tensor_is_fuseable(col_out)) return false;

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
        if (!src || !try_fuse_common({src, bias, out})) return false;

        col.opcode = HTP_OP_COL2IM_1D_BIAS;
        col.name = "BREEZE_COL2IM_BIAS";
        col.inputs.clear();
        col.inputs.push_back(src);
        col.inputs.push_back(bias);
        col.outputs.clear();
        col.outputs.push_back(out);
        col.fused.push_back(node.node);

        htp_op_desc & o = h_ops[n_ops - 1];
        o.opcode = HTP_OP_COL2IM_1D_BIAS;
        o.src[0] = add_tensor(src);
        o.src[1] = add_tensor(bias);
        for (unsigned int k = 2; k < HTP_OP_MAX_INPUTS; ++k) o.src[k] = 0xffff;
        o.dst[0] = add_tensor(out);
        for (unsigned int k = 1; k < HTP_OP_MAX_OUTPUTS; ++k) o.dst[k] = 0xffff;

        HEX_VERBOSE("ggml-hex: %s fused Breeze COL2IM+bias (#%u)\n",
                    sess->c_name(), n_ops - 1);
        return true;
    }

"""
snake_fusion = r"""
    bool try_fuse_breeze_snake(const htp_opnode & node) {
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
        if (!mul_alpha_ok || sin_node.opcode != HTP_OP_SIN ||
            sqr_node.opcode != HTP_OP_SQR || !mul_beta_ok) {
            return false;
        }

        const ggml_tensor * ax = mul_alpha.dst();
        const ggml_tensor * sn = sin_node.dst();
        const ggml_tensor * sq = sqr_node.dst();
        const ggml_tensor * del = mul_beta.dst();
        const ggml_tensor * out = node.dst();
        if (!ax || !sn || !sq || !del || !out ||
            sin_node.src0() != ax || sqr_node.src0() != sn) return false;

        const ggml_tensor * x = nullptr;
        const ggml_tensor * alpha = nullptr;
        if (mul_alpha.src0() && mul_alpha.src1()) {
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
            inv_beta->ne[0] != 1 || inv_beta->ne[1] != x->ne[1] ||
            !ggml_is_contiguous(x) || !ggml_is_contiguous(alpha) ||
            !ggml_is_contiguous(inv_beta) || !ggml_is_contiguous(out)) {
            return false;
        }

        if (!ggml_hexagon_tensor_is_fuseable(ax) ||
            !ggml_hexagon_tensor_is_fuseable(sn) ||
            !ggml_hexagon_tensor_is_fuseable(sq) ||
            !ggml_hexagon_tensor_is_fuseable(del) ||
            !try_fuse_common({x, alpha, inv_beta, out})) {
            return false;
        }

        htp_opnode fused(HTP_OP_SNAKE, const_cast<ggml_tensor *>(out));
        fused.name = "BREEZE_SNAKE";
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
        HEX_VERBOSE("ggml-hex: %s fused Breeze Snake (#%u)\n", sess->c_name(), base);
        return true;
    }

"""
replace_once(
    host,
    """    bool try_fuse(const htp_opnode & node) {
        if (!opt_opfusion) return false;""",
    col2im_fusion + snake_fusion + """    bool try_fuse(const htp_opnode & node) {
        if (!opt_opfusion) return false;
        if (try_fuse_breeze_col2im_bias(node)) return true;
        if (try_fuse_breeze_snake(node)) return true;""",
    "Breeze batch fusions",
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
                extra->flags |= GGML_HEXAGON_TENSOR_FUSEABLE;
            }""",
    "Breeze fuseable tensor tags",
)

replace_once(
    host,
    """    struct htp_binary_vtcm_layout L;
    htp_binary_vtcm_layout_build(&L, kparams, sess->vtcm_size);
    if (L.rows_per_buffer == 0 || L.total_bytes > sess->vtcm_size) {""",
    """    // Waveform-scale binaries can exceed an 8 MiB v81 VTCM layout when
    // every worker gets a full row. Reduce only this op's worker count until
    // the exact layout fits. Dedicated Breeze channel broadcasts above bypass
    // row staging entirely.
    struct htp_binary_vtcm_layout L;
    while (kparams->n_threads > 0) {
        htp_binary_vtcm_layout_build(&L, kparams, sess->vtcm_size);
        if (L.rows_per_buffer != 0 && L.total_bytes <= sess->vtcm_size) break;
        --kparams->n_threads;
    }
    if (kparams->n_threads == 0) {""",
    "Breeze adaptive binary VTCM",
)

replace_once(
    codec,
    """    ggml_tensor * alpha = ggml_reshape_2d(ctx, ggml_exp(ctx, la), 1, la->ne[0]);
    ggml_tensor * beta = ggml_reshape_2d(ctx, ggml_exp(ctx, lb), 1, lb->ne[0]);
    ggml_tensor * s = ggml_sin(ctx, ggml_mul(ctx, x, alpha));
    return ggml_add(ctx, x, ggml_div(ctx, ggml_sqr(ctx, s), beta));""",
    """    ggml_tensor * alpha = ggml_reshape_2d(ctx, ggml_exp(ctx, la), 1, la->ne[0]);
    // Exact algebraic lowering: sqr(sin(x*a)) / exp(lb) ==
    // sqr(sin(x*a)) * exp(-lb). This turns the waveform-sized DIV broadcast
    // into the streaming MUL path and enables the fused HTP Snake kernel.
    ggml_tensor * inv_beta = ggml_reshape_2d(ctx, ggml_exp(ctx, ggml_neg(ctx, lb)), 1, lb->ne[0]);
    ggml_tensor * s = ggml_sin(ctx, ggml_mul(ctx, x, alpha));
    return ggml_add(ctx, x, ggml_mul(ctx, ggml_sqr(ctx, s), inv_beta));""",
    "Breeze Snake reciprocal lowering",
)


# The per-frame audio embedding shape is constant. Upstream rebuilt and
# reallocated this graph for every 80 ms audio frame, then immediately freed
# it. Keep one prepared graph resident and only update the 16 codebook ids.
replace_once(
    backbone_h,
    """struct StepOut {
    std::vector<float> hidden; // [hidden_size]
    std::vector<float> logits; // [audio_vocab_size + 1]
};""",
    """struct StepOut {
    std::vector<float> hidden; // [hidden_size]
    std::vector<float> logits; // [audio_vocab_size + 1]
};

struct AudioEmbedRunner {
    Graph graph{256};
    ggml_tensor * ids = nullptr;
    ggml_tensor * out = nullptr;
    int n_codebooks = 0;
    int vocab = 0;

    void init(BreezeModel & m);
    std::vector<float> run(BreezeModel & m, const std::vector<int> & codes);
};""",
    "persistent audio embedding runner declaration",
)

replace_once(
    backbone_cpp,
    """static ggml_tensor * build_audio_embed(ggml_context * ctx, BreezeModel & m, ggml_tensor * idx, int n) {
    const int nc = m.cfg.num_codebooks;
    const int hidden = m.cfg.hidden_size;
    // get_rows only indexes along one axis, so the frames stay flat until after the lookup
    ggml_tensor * rows = ggml_get_rows(ctx, m.w("audio_embd.weight"), idx);    // [hidden, nc*n]
    rows = ggml_reshape_3d(ctx, rows, hidden, nc, n);
    ggml_tensor * perm = ggml_cont(ctx, ggml_permute(ctx, rows, 1, 0, 2, 3));  // [nc, hidden, n]
    ggml_tensor * summed = ggml_sum_rows(ctx, perm);                           // [1, hidden, n]
    return ggml_reshape_2d(ctx, summed, hidden, n);
}
""",
    """static ggml_tensor * build_audio_embed(ggml_context * ctx, BreezeModel & m, ggml_tensor * idx, int n) {
    const int nc = m.cfg.num_codebooks;
    const int hidden = m.cfg.hidden_size;
    // get_rows only indexes along one axis, so the frames stay flat until after the lookup
    ggml_tensor * rows = ggml_get_rows(ctx, m.w("audio_embd.weight"), idx);    // [hidden, nc*n]
    rows = ggml_reshape_3d(ctx, rows, hidden, nc, n);
    ggml_tensor * perm = ggml_cont(ctx, ggml_permute(ctx, rows, 1, 0, 2, 3));  // [nc, hidden, n]
    ggml_tensor * summed = ggml_sum_rows(ctx, perm);                           // [1, hidden, n]
    return ggml_reshape_2d(ctx, summed, hidden, n);
}

void AudioEmbedRunner::init(BreezeModel & m) {
    n_codebooks = m.cfg.num_codebooks;
    vocab = m.cfg.audio_vocab_size;
    std::vector<int32_t> zeros((size_t) n_codebooks, 0);
    ids = graph.input_i32(zeros, n_codebooks);
    out = build_audio_embed(graph.ctx, m, ids, 1);
    graph.prepare(m.backend, out);
}

std::vector<float> AudioEmbedRunner::run(BreezeModel & m, const std::vector<int> & codes) {
    if (!ids || !out) init(m);
    if ((int) codes.size() != n_codebooks) {
        throw std::runtime_error("audio embedding frame has wrong codebook count");
    }
    std::vector<int32_t> idx((size_t) n_codebooks);
    for (int cb = 0; cb < n_codebooks; ++cb) idx[(size_t) cb] = codes[(size_t) cb] + cb * vocab;
    ggml_backend_tensor_set(ids, idx.data(), 0, idx.size() * sizeof(int32_t));
    graph.replay(m.backend);
    return tensor_to_f32(out);
}
""",
    "persistent audio embedding runner implementation",
)

replace_once(
    generation,
    """    DepthRunner depth;
    depth.init(m, use_cfg ? 2 : 1);

    SampleParams bp;""",
    """    DepthRunner depth;
    depth.init(m, use_cfg ? 2 : 1);
    AudioEmbedRunner audio_embed;
    audio_embed.init(m);

    SampleParams bp;""",
    "generation audio embedding cache",
)

replace_once(
    generation,
    """        std::vector<float> ae = audio_embed_forward(m, frame, 1);
        if (use_cfg) {""",
    """        std::vector<float> ae = audio_embed.run(m, frame);
        if (use_cfg) {""",
    "generation cached audio embedding",
)

replace_once(
    generation,
    """    DepthRunner depth;
    depth.init(m, use_cfg ? 2 : 1);

    std::vector<int> out((size_t) src_T * nc);""",
    """    DepthRunner depth;
    depth.init(m, use_cfg ? 2 : 1);
    AudioEmbedRunner audio_embed;
    audio_embed.init(m);

    std::vector<int> out((size_t) src_T * nc);""",
    "conversion audio embedding cache",
)

replace_once(
    generation,
    """        std::vector<float> ae = audio_embed_forward(m, frame, 1);
        if (use_cfg) {""",
    """        std::vector<float> ae = audio_embed.run(m, frame);
        if (use_cfg) {""",
    "conversion cached audio embedding",
)
