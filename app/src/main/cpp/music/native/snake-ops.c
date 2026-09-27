// Local Dream YuE2 fused Oobleck Snake activation for Qualcomm Hexagon HTP.
//
// Computes y = x + sin(alpha * x)^2 * inv_beta in one HTP op instead of
// materializing the five GGML elementwise nodes separately. YuE2 stores audio
// activations as [T, C], so each channel is one contiguous row and alpha /
// inv_beta are [1, C] row-broadcast scalars. The hot path runs 32 fp32 samples
// per HVX vector and never leaves accelerator memory.
#include <HAP_farf.h>
#include <math.h>
#include <stdint.h>

#include "hex-common.h"
#include "hex-profile.h"
#include "hvx-utils.h"
#include "htp-ctx.h"
#include "htp-ops.h"
#include "htp-tensor.h"

struct htp_snake_context {
    struct htp_ops_context * octx;
    uint32_t channels;
    uint32_t channels_per_thread;
    uint32_t time;
};

static void snake_thread(unsigned int nth, unsigned int ith, void * data) {
    (void) nth;
    const struct htp_snake_context * sctx =
        (const struct htp_snake_context *) data;
    struct htp_ops_context * octx = sctx->octx;

    const struct htp_tensor * x_t     = octx->src[0];
    const struct htp_tensor * alpha_t = octx->src[1];
    const struct htp_tensor * beta_t  = octx->src[2];
    const struct htp_tensor * dst_t   = octx->dst;

    const float * x = (const float *) (uintptr_t) x_t->data;
    const float * alpha = (const float *) (uintptr_t) alpha_t->data;
    const float * inv_beta = (const float *) (uintptr_t) beta_t->data;
    float * dst = (float *) (uintptr_t) dst_t->data;

    const uint32_t c0 = sctx->channels_per_thread * ith;
    const uint32_t c1 = MIN(c0 + sctx->channels_per_thread, sctx->channels);
    if (c0 >= c1) return;

    struct htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ith);

    for (uint32_t c = c0; c < c1; ++c) {
        const float a = alpha[c];
        const float b = inv_beta[c];
        const HVX_Vector va = hvx_vec_splat_f32(a);
        const HVX_Vector vb = hvx_vec_splat_f32(b);

        const float * row = x + (size_t) c * sctx->time;
        float * out = dst + (size_t) c * sctx->time;

        uint32_t t = 0;
        for (; t + VLEN_FP32 <= sctx->time; t += VLEN_FP32) {
            const HVX_Vector vx = *(const HVX_UVector *) (row + t);
            const HVX_Vector vax = HVX_OP_MUL_F32(vx, va);
            const HVX_Vector vs = hvx_vec_sin_f32(vax);
            const HVX_Vector vs2 = HVX_OP_MUL_F32(vs, vs);
            const HVX_Vector vd = HVX_OP_MUL_F32(vs2, vb);
            const HVX_Vector vy = HVX_OP_ADD_F32(vx, vd);
            *(HVX_UVector *) (out + t) = vy;
        }
        for (; t < sctx->time; ++t) {
            const float s = sinf(a * row[t]);
            out[t] = row[t] + s * s * b;
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ith);
}

int op_snake(struct htp_ops_context * octx) {
    const struct htp_tensor * x = octx->src[0];
    const struct htp_tensor * alpha = octx->src[1];
    const struct htp_tensor * inv_beta = octx->src[2];
    const struct htp_tensor * dst = octx->dst;

    if (!x || !alpha || !inv_beta || !dst ||
        x->type != HTP_TYPE_F32 || alpha->type != HTP_TYPE_F32 ||
        inv_beta->type != HTP_TYPE_F32 || dst->type != HTP_TYPE_F32) {
        return HTP_STATUS_NO_SUPPORT;
    }
    if (x->ne[2] != 1 || x->ne[3] != 1 ||
        dst->ne[0] != x->ne[0] || dst->ne[1] != x->ne[1] ||
        dst->ne[2] != 1 || dst->ne[3] != 1 ||
        alpha->ne[0] != 1 || alpha->ne[1] != x->ne[1] ||
        inv_beta->ne[0] != 1 || inv_beta->ne[1] != x->ne[1]) {
        return HTP_STATUS_INVAL_PARAMS;
    }
    if (x->nb[0] != sizeof(float) || dst->nb[0] != sizeof(float) ||
        x->nb[1] != x->ne[0] * sizeof(float) ||
        dst->nb[1] != dst->ne[0] * sizeof(float) ||
        alpha->nb[0] != sizeof(float) || inv_beta->nb[0] != sizeof(float)) {
        return HTP_STATUS_NO_SUPPORT;
    }

    const uint32_t channels = x->ne[1];
    const uint32_t time = x->ne[0];
    if (channels == 0 || time == 0) return HTP_STATUS_INVAL_PARAMS;

    const uint32_t n_threads = MIN(octx->ctx->n_threads, channels);
    if (n_threads == 0) return HTP_STATUS_INVAL_PARAMS;

    struct htp_snake_context sctx = {
        .octx = octx,
        .channels = channels,
        .channels_per_thread = (channels + n_threads - 1) / n_threads,
        .time = time,
    };
    work_queue_run(octx->ctx->work_queue, snake_thread, &sctx, n_threads);
    return HTP_STATUS_OK;
}
