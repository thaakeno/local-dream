// Local Dream Breeze native HTP channel-broadcast binary ops.
//
// Handles Breeze vocoder's huge contiguous F32 binary rows without
// staging a multi-megabyte row in VTCM. It supports [T,C] op [1,C] channel
// broadcasts plus wide [T,C] op [T,C] residuals. ADD covers conv/residual
// bias paths and MUL covers Snake alpha/inv-beta. The tensors stay in HTP-mapped
// memory and each worker streams bounded time chunks with HVX. These are
// first-class HTP ops: unsupported shapes are rejected and strict accelerator
// mode fails rather than falling back to CPU.
#include <HAP_farf.h>
#include <stdint.h>

#include "hex-common.h"
#include "hex-profile.h"
#include "hvx-utils.h"
#include "htp-ctx.h"
#include "htp-ops.h"
#include "htp-tensor.h"

#define BREEZE_CHANNEL_ADD_CHUNK_ELEMS 8192u

static inline HVX_Vector breeze_add_f32(HVX_Vector a, HVX_Vector b) {
#if __HVX_ARCH__ < 79
    return Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(a, b));
#else
    return Q6_Vsf_vadd_VsfVsf(a, b);
#endif
}

static inline HVX_Vector breeze_mul_f32(HVX_Vector a, HVX_Vector b) {
#if __HVX_ARCH__ < 79
    return Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(a, b));
#else
    return Q6_Vsf_vmpy_VsfVsf(a, b);
#endif
}

enum breeze_channel_binary_kind {
    BREEZE_CHANNEL_BINARY_ADD = 0,
    BREEZE_CHANNEL_BINARY_MUL = 1,
};

struct htp_channel_add_context {
    struct htp_ops_context * octx;
    uint32_t time;
    uint32_t channels;
    uint32_t chunks_per_row;
    uint32_t total_jobs;
    enum breeze_channel_binary_kind kind;
    int channel_broadcast;
};

static void channel_add_thread(unsigned int nth, unsigned int ith, void * data) {
    const struct htp_channel_add_context * c =
        (const struct htp_channel_add_context *) data;
    struct htp_ops_context * octx = c->octx;

    const struct htp_tensor * src0_t = octx->src[0];
    const struct htp_tensor * src1_t = octx->src[1];
    const struct htp_tensor * dst_t  = octx->dst;

    const float * src0 = (const float *) (uintptr_t) src0_t->data;
    const float * bias = (const float *) (uintptr_t) src1_t->data;
    float * dst = (float *) (uintptr_t) dst_t->data;

    struct htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ith);

    for (uint32_t job = ith; job < c->total_jobs; job += nth) {
        const uint32_t ch = job / c->chunks_per_row;
        const uint32_t chunk = job - ch * c->chunks_per_row;
        const uint32_t start = chunk * BREEZE_CHANNEL_ADD_CHUNK_ELEMS;
        const uint32_t end = MIN(start + BREEZE_CHANNEL_ADD_CHUNK_ELEMS, c->time);
        if (start >= end) {
            continue;
        }

        const float * in = src0 + (size_t) ch * c->time;
        const float * rhs = c->channel_broadcast
            ? NULL
            : bias + (size_t) ch * c->time;
        float * out = dst + (size_t) ch * c->time;
        const HVX_Vector vb = c->channel_broadcast
            ? hvx_vec_splat_f32(bias[ch])
            : hvx_vec_splat_f32(0.0f);

        uint32_t t = start;
        if (c->kind == BREEZE_CHANNEL_BINARY_ADD) {
            for (; t + VLEN_FP32 <= end; t += VLEN_FP32) {
                const HVX_Vector vx = *(const HVX_UVector *) (in + t);
                const HVX_Vector vr = c->channel_broadcast
                    ? vb
                    : *(const HVX_UVector *) (rhs + t);
                *(HVX_UVector *) (out + t) = breeze_add_f32(vx, vr);
            }
            for (; t < end; ++t) {
                out[t] = in[t] + (c->channel_broadcast ? bias[ch] : rhs[t]);
            }
        } else {
            for (; t + VLEN_FP32 <= end; t += VLEN_FP32) {
                const HVX_Vector vx = *(const HVX_UVector *) (in + t);
                const HVX_Vector vr = c->channel_broadcast
                    ? vb
                    : *(const HVX_UVector *) (rhs + t);
                *(HVX_UVector *) (out + t) = breeze_mul_f32(vx, vr);
            }
            for (; t < end; ++t) {
                out[t] = in[t] * (c->channel_broadcast ? bias[ch] : rhs[t]);
            }
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ith);
}

static int op_channel_bcast_binary(struct htp_ops_context * octx,
                                   enum breeze_channel_binary_kind kind) {
    const struct htp_tensor * src0 = octx->src[0];
    const struct htp_tensor * src1 = octx->src[1];
    const struct htp_tensor * dst  = octx->dst;

    if (!src0 || !src1 || !dst ||
        src0->type != HTP_TYPE_F32 || src1->type != HTP_TYPE_F32 ||
        dst->type != HTP_TYPE_F32) {
        return HTP_STATUS_NO_SUPPORT;
    }

    if (src0->ne[0] == 0 || src0->ne[1] == 0 ||
        src0->ne[2] != 1 || src0->ne[3] != 1 ||
        dst->ne[0] != src0->ne[0] || dst->ne[1] != src0->ne[1] ||
        dst->ne[2] != 1 || dst->ne[3] != 1) {
        return HTP_STATUS_INVAL_PARAMS;
    }

    const int channel_broadcast =
        src1->ne[0] == 1 &&
        src1->ne[1] == src0->ne[1] &&
        src1->ne[2] == 1 &&
        src1->ne[3] == 1;
    const int same_shape =
        src1->ne[0] == src0->ne[0] &&
        src1->ne[1] == src0->ne[1] &&
        src1->ne[2] == 1 &&
        src1->ne[3] == 1;
    if (!channel_broadcast && !same_shape) {
        return HTP_STATUS_INVAL_PARAMS;
    }

    if (src0->nb[0] != sizeof(float) || dst->nb[0] != sizeof(float) ||
        src1->nb[0] != sizeof(float) ||
        src0->nb[1] != src0->ne[0] * sizeof(float) ||
        dst->nb[1] != dst->ne[0] * sizeof(float) ||
        (channel_broadcast
            ? src1->nb[1] != sizeof(float)
            : src1->nb[1] != src1->ne[0] * sizeof(float))) {
        return HTP_STATUS_NO_SUPPORT;
    }

    const uint32_t time = (uint32_t) src0->ne[0];
    const uint32_t channels = (uint32_t) src0->ne[1];
    const uint32_t chunks =
        (time + BREEZE_CHANNEL_ADD_CHUNK_ELEMS - 1) / BREEZE_CHANNEL_ADD_CHUNK_ELEMS;
    const uint64_t total64 = (uint64_t) channels * chunks;
    if (chunks == 0 || total64 == 0 || total64 > UINT32_MAX) {
        return HTP_STATUS_INVAL_PARAMS;
    }

    const uint32_t total_jobs = (uint32_t) total64;
    const uint32_t n_threads = MIN(octx->ctx->n_threads, total_jobs);
    if (n_threads == 0) {
        return HTP_STATUS_INVAL_PARAMS;
    }

    struct htp_channel_add_context c = {
        .octx = octx,
        .time = time,
        .channels = channels,
        .chunks_per_row = chunks,
        .total_jobs = total_jobs,
        .kind = kind,
        .channel_broadcast = channel_broadcast,
    };
    work_queue_run(octx->ctx->work_queue, channel_add_thread, &c, n_threads);
    return HTP_STATUS_OK;
}

int op_channel_bcast_add(struct htp_ops_context * octx) {
    return op_channel_bcast_binary(octx, BREEZE_CHANNEL_BINARY_ADD);
}

int op_channel_bcast_mul(struct htp_ops_context * octx) {
    return op_channel_bcast_binary(octx, BREEZE_CHANNEL_BINARY_MUL);
}
