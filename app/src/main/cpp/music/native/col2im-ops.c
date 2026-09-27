// Local Dream YuE2 native HTP COL2IM_1D op.
// Scatter-add half of Oobleck's GEMM-based ConvTranspose1d.
// Runs entirely in HTP-mapped DDR, so audio tensors never bounce through CPU.
#include <HAP_farf.h>
#include <stdint.h>

#include "hex-common.h"
#include "hex-profile.h"
#include "htp-ctx.h"
#include "htp-ops.h"
#include "htp-tensor.h"

struct htp_col2im_context {
    struct htp_ops_context * octx;
    uint32_t total;
    uint32_t per_thread;
    int32_t stride;
    int32_t out_channels;
    int32_t padding;
    int32_t kernel;
    int32_t k_oc;
    int32_t t_in;
    int32_t t_out;
};

static void col2im_thread(unsigned int nth, unsigned int ith, void * data) {
    (void) nth;
    const struct htp_col2im_context * c = (const struct htp_col2im_context *) data;
    const struct htp_tensor * src_t = c->octx->src[0];
    const struct htp_tensor * dst_t = c->octx->dst;
    const float * src = (const float *) (uintptr_t) src_t->data;
    float * dst = (float *) (uintptr_t) dst_t->data;

    uint32_t begin = c->per_thread * ith;
    uint32_t end = MIN(begin + c->per_thread, c->total);
    if (begin >= end) return;

    struct htp_thread_trace * tr = &c->octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ith);

    for (uint32_t work = begin; work < end; ++work) {
        int32_t oc = (int32_t) (work / (uint32_t) c->t_out);
        int32_t t_out = (int32_t) (work - (uint32_t) oc * (uint32_t) c->t_out);
        int32_t t_abs = t_out + c->padding;

        const int32_t a = t_abs - c->kernel + 1;
        int32_t t_min = a >= 0 ? (a + c->stride - 1) / c->stride : a / c->stride;
        if (t_min < 0) t_min = 0;
        int32_t t_max = t_abs / c->stride;
        if (t_max >= c->t_in) t_max = c->t_in - 1;

        float sum = 0.0f;
        for (int32_t ti = t_min; ti <= t_max; ++ti) {
            int32_t k = t_abs - ti * c->stride;
            if ((uint32_t) k < (uint32_t) c->kernel) {
                sum += src[(size_t) ti * (size_t) c->k_oc +
                           (size_t) oc * (size_t) c->kernel + (size_t) k];
            }
        }
        dst[(size_t) oc * (size_t) c->t_out + (size_t) t_out] = sum;
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ith);
}

int op_col2im_1d(struct htp_ops_context * octx) {
    const struct htp_tensor * src = octx->src[0];
    const struct htp_tensor * dst = octx->dst;
    if (!src || !dst || src->type != HTP_TYPE_F32 || dst->type != HTP_TYPE_F32) {
        return HTP_STATUS_NO_SUPPORT;
    }
    if (src->nb[0] != sizeof(float) || dst->nb[0] != sizeof(float) ||
        src->nb[1] != src->ne[0] * sizeof(float) ||
        dst->nb[1] != dst->ne[0] * sizeof(float)) {
        return HTP_STATUS_NO_SUPPORT;
    }

    int32_t stride = octx->op_params[0];
    int32_t oc = octx->op_params[1];
    int32_t padding = octx->op_params[2];
    if (stride <= 0 || oc <= 0 || src->ne[0] % (uint32_t) oc != 0) {
        return HTP_STATUS_INVAL_PARAMS;
    }

    int32_t k_oc = (int32_t) src->ne[0];
    int32_t kernel = k_oc / oc;
    int32_t t_in = (int32_t) src->ne[1];
    int32_t t_out = (t_in - 1) * stride + kernel - 2 * padding;
    if (t_out <= 0 || dst->ne[0] != (uint32_t) t_out || dst->ne[1] != (uint32_t) oc) {
        return HTP_STATUS_INVAL_PARAMS;
    }

    uint64_t total64 = (uint64_t) oc * (uint64_t) t_out;
    if (total64 == 0 || total64 > UINT32_MAX) return HTP_STATUS_INVAL_PARAMS;
    uint32_t total = (uint32_t) total64;
    uint32_t n_threads = octx->ctx->n_threads;
    struct htp_col2im_context ctx = {
        .octx = octx,
        .total = total,
        .per_thread = (total + n_threads - 1) / n_threads,
        .stride = stride,
        .out_channels = oc,
        .padding = padding,
        .kernel = kernel,
        .k_oc = k_oc,
        .t_in = t_in,
        .t_out = t_out,
    };
    work_queue_run(octx->ctx->work_queue, col2im_thread, &ctx, n_threads);
    return HTP_STATUS_OK;
}
