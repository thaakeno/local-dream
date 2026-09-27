// Local Dream YuE2 native HTP sine op.
// Runs GGML_OP_SIN entirely on the Hexagon DSP; no CPU scheduler handoff.
#include <HAP_farf.h>
#include <math.h>
#include <stdint.h>

#include "hex-common.h"
#include "hex-profile.h"
#include "hvx-sin-cos.h"
#include "htp-ctx.h"
#include "htp-ops.h"
#include "htp-tensor.h"

struct htp_sin_context {
    struct htp_ops_context * octx;
    uint32_t start;
    uint32_t count;
    uint32_t per_thread;
};

static void sin_thread(unsigned int nth, unsigned int ith, void * data) {
    (void) nth;
    const struct htp_sin_context * sctx = (const struct htp_sin_context *) data;
    struct htp_ops_context * octx = sctx->octx;
    const struct htp_tensor * src = octx->src[0];
    const struct htp_tensor * dst = octx->dst;

    uint32_t begin = sctx->start + sctx->per_thread * ith;
    uint32_t end = MIN(begin + sctx->per_thread, sctx->start + sctx->count);
    if (begin >= end) return;

    const float * in = (const float *) (uintptr_t) src->data;
    float * out = (float *) (uintptr_t) dst->data;

    struct htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ith);

    uint32_t i = begin;
    for (; i + 32 <= end; i += 32) {
        HVX_Vector x = *(const HVX_UVector *) (in + i);
        HVX_Vector y = hvx_vec_sin_f32(x);
        *(HVX_UVector *) (out + i) = y;
    }
    for (; i < end; ++i) {
        out[i] = sinf(in[i]);
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ith);
}

int op_sin(struct htp_ops_context * octx) {
    const struct htp_tensor * src = octx->src[0];
    const struct htp_tensor * dst = octx->dst;
    if (!src || !dst || src->type != HTP_TYPE_F32 || dst->type != HTP_TYPE_F32) {
        return HTP_STATUS_NO_SUPPORT;
    }
    if (src->ne[0] != dst->ne[0] || src->ne[1] != dst->ne[1] ||
        src->ne[2] != dst->ne[2] || src->ne[3] != dst->ne[3]) {
        return HTP_STATUS_INVAL_PARAMS;
    }
    if (src->nb[0] != sizeof(float) || dst->nb[0] != sizeof(float) ||
        src->nb[1] != src->ne[0] * sizeof(float) ||
        dst->nb[1] != dst->ne[0] * sizeof(float)) {
        return HTP_STATUS_NO_SUPPORT;
    }

    uint64_t total64 = (uint64_t) src->ne[0] * src->ne[1] * src->ne[2] * src->ne[3];
    if (total64 == 0 || total64 > UINT32_MAX) return HTP_STATUS_INVAL_PARAMS;
    uint32_t total = (uint32_t) total64;
    uint32_t n_threads = octx->ctx->n_threads;
    struct htp_sin_context sctx = {
        .octx = octx,
        .start = 0,
        .count = total,
        .per_thread = (total + n_threads - 1) / n_threads,
    };
    work_queue_run(octx->ctx->work_queue, sin_thread, &sctx, n_threads);
    return HTP_STATUS_OK;
}
