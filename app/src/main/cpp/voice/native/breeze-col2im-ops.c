// Local Dream Breeze native HTP COL2IM_1D and fused COL2IM+bias ops.
//
// Oobleck's transposed convolutions always use kernel = 2 * stride.  In that
// case the overlap pattern is exact and tiny: each output block is either one
// half-kernel at an edge, or the sum of the second half from t-1 and the first
// half from t.  Compute that layout directly instead of doing one division,
// range search and scattered load per output sample.
//
// The op stays entirely on HTP-mapped memory, partitions by output channel so
// workers never contend for the same destination row, and writes each row
// sequentially. The fused form applies the per-channel transpose-conv bias
// while the sample is already in a register, eliminating a full waveform-sized
// ADD pass. Unexpected shapes keep a correctness fallback.
#include <HAP_farf.h>
#include <stdint.h>

#include "hex-common.h"
#include "hex-profile.h"
#include "htp-ctx.h"
#include "htp-ops.h"
#include "htp-tensor.h"

struct htp_col2im_context {
    struct htp_ops_context * octx;
    uint32_t channel_start;
    uint32_t channel_count;
    uint32_t channels_per_thread;
    int32_t stride;
    int32_t out_channels;
    int32_t padding;
    int32_t kernel;
    int32_t k_oc;
    int32_t t_in;
    int32_t t_out;
    int32_t fast_overlap;
    int32_t has_bias;
};

static inline void col2im_fast_channel(const float * src,
                                       float * dst,
                                       int32_t oc,
                                       float bias,
                                       const struct htp_col2im_context * c) {
    const int32_t s = c->stride;
    const int32_t edge = s - c->padding;
    const size_t oc_off = (size_t) oc * (size_t) c->kernel;
    size_t out = 0;

    // Full (uncropped) transpose-conv blocks are:
    //   A0,
    //   B0 + A1,
    //   B1 + A2,
    //   ...
    //   B(T-2) + A(T-1),
    //   B(T-1)
    // Padding simply crops p values from both outer blocks.
    const float * a0 = src + oc_off;
    for (int32_t r = c->padding; r < s; ++r) {
        dst[out++] = a0[r] + bias;
    }

    for (int32_t ti = 1; ti < c->t_in; ++ti) {
        const float * prev_b =
            src + (size_t) (ti - 1) * (size_t) c->k_oc + oc_off + (size_t) s;
        const float * cur_a =
            src + (size_t) ti * (size_t) c->k_oc + oc_off;

        if (ti + 1 < c->t_in) {
            __builtin_prefetch(
                src + (size_t) (ti + 1) * (size_t) c->k_oc + oc_off,
                0,
                2);
        }

        // stride is one of 6,5,4,4,2,2 for Breeze.  Keeping this tiny loop
        // branch-free is substantially cheaper than reconstructing t_min /
        // t_max for every output sample.
        for (int32_t r = 0; r < s; ++r) {
            dst[out++] = prev_b[r] + cur_a[r] + bias;
        }
    }

    const float * tail =
        src + (size_t) (c->t_in - 1) * (size_t) c->k_oc + oc_off + (size_t) s;
    for (int32_t r = 0; r < edge; ++r) {
        dst[out++] = tail[r] + bias;
    }
}

static inline void col2im_generic_channel(const float * src,
                                          float * dst,
                                          int32_t oc,
                                          float bias,
                                          const struct htp_col2im_context * c) {
    for (int32_t t_out = 0; t_out < c->t_out; ++t_out) {
        const int32_t t_abs = t_out + c->padding;
        const int32_t a = t_abs - c->kernel + 1;
        int32_t t_min = a >= 0
            ? (a + c->stride - 1) / c->stride
            : a / c->stride;
        if (t_min < 0) t_min = 0;

        int32_t t_max = t_abs / c->stride;
        if (t_max >= c->t_in) t_max = c->t_in - 1;

        float sum = 0.0f;
        for (int32_t ti = t_min; ti <= t_max; ++ti) {
            const int32_t k = t_abs - ti * c->stride;
            if ((uint32_t) k < (uint32_t) c->kernel) {
                sum += src[(size_t) ti * (size_t) c->k_oc +
                           (size_t) oc * (size_t) c->kernel +
                           (size_t) k];
            }
        }
        dst[t_out] = sum + bias;
    }
}

static void col2im_thread(unsigned int nth, unsigned int ith, void * data) {
    (void) nth;
    const struct htp_col2im_context * c =
        (const struct htp_col2im_context *) data;
    const struct htp_tensor * src_t = c->octx->src[0];
    const struct htp_tensor * bias_t = c->has_bias ? c->octx->src[1] : NULL;
    const struct htp_tensor * dst_t = c->octx->dst;
    const float * src = (const float *) (uintptr_t) src_t->data;
    const float * bias = bias_t ? (const float *) (uintptr_t) bias_t->data : NULL;
    float * dst = (float *) (uintptr_t) dst_t->data;

    const uint32_t begin =
        c->channel_start + c->channels_per_thread * ith;
    const uint32_t end = MIN(
        begin + c->channels_per_thread,
        c->channel_start + c->channel_count);
    if (begin >= end) return;

    struct htp_thread_trace * tr = &c->octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ith);

    for (uint32_t ch = begin; ch < end; ++ch) {
        float * out = dst + (size_t) ch * (size_t) c->t_out;
        const float b = bias ? bias[ch] : 0.0f;
        if (c->fast_overlap) {
            col2im_fast_channel(src, out, (int32_t) ch, b, c);
        } else {
            col2im_generic_channel(src, out, (int32_t) ch, b, c);
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ith);
}

static int op_col2im_1d_impl(struct htp_ops_context * octx, int has_bias) {
    const struct htp_tensor * src = octx->src[0];
    const struct htp_tensor * bias = has_bias ? octx->src[1] : NULL;
    const struct htp_tensor * dst = octx->dst;
    if (!src || !dst || src->type != HTP_TYPE_F32 || dst->type != HTP_TYPE_F32 ||
        (has_bias && (!bias || bias->type != HTP_TYPE_F32))) {
        return HTP_STATUS_NO_SUPPORT;
    }
    if (src->nb[0] != sizeof(float) || dst->nb[0] != sizeof(float) ||
        src->nb[1] != src->ne[0] * sizeof(float) ||
        dst->nb[1] != dst->ne[0] * sizeof(float)) {
        return HTP_STATUS_NO_SUPPORT;
    }

    const int32_t stride = octx->op_params[0];
    const int32_t oc = octx->op_params[1];
    const int32_t padding = octx->op_params[2];
    if (stride <= 0 || oc <= 0 || src->ne[0] % (uint32_t) oc != 0) {
        return HTP_STATUS_INVAL_PARAMS;
    }

    const int32_t k_oc = (int32_t) src->ne[0];
    const int32_t kernel = k_oc / oc;
    const int32_t t_in = (int32_t) src->ne[1];
    const int32_t t_out =
        (t_in - 1) * stride + kernel - 2 * padding;
    if (t_in <= 0 || t_out <= 0 ||
        dst->ne[0] != (uint32_t) t_out ||
        dst->ne[1] != (uint32_t) oc) {
        return HTP_STATUS_INVAL_PARAMS;
    }
    if (has_bias &&
        (bias->ne[0] != 1 || bias->ne[1] != (uint32_t) oc ||
         bias->ne[2] != 1 || bias->ne[3] != 1 ||
         bias->nb[0] != sizeof(float) || bias->nb[1] != sizeof(float))) {
        return HTP_STATUS_INVAL_PARAMS;
    }

    const uint32_t n_threads = MIN(
        octx->ctx->n_threads,
        (uint32_t) oc);
    if (n_threads == 0) return HTP_STATUS_INVAL_PARAMS;

    struct htp_col2im_context ctx = {
        .octx = octx,
        .channel_start = 0,
        .channel_count = (uint32_t) oc,
        .channels_per_thread =
            ((uint32_t) oc + n_threads - 1) / n_threads,
        .stride = stride,
        .out_channels = oc,
        .padding = padding,
        .kernel = kernel,
        .k_oc = k_oc,
        .t_in = t_in,
        .t_out = t_out,
        .fast_overlap =
            kernel == 2 * stride &&
            padding >= 0 &&
            padding < stride,
        .has_bias = has_bias,
    };

    work_queue_run(
        octx->ctx->work_queue,
        col2im_thread,
        &ctx,
        n_threads);
    return HTP_STATUS_OK;
}

int op_col2im_1d(struct htp_ops_context * octx) {
    return op_col2im_1d_impl(octx, 0);
}

int op_col2im_1d_bias(struct htp_ops_context * octx) {
    return op_col2im_1d_impl(octx, 1);
}
