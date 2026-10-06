#ifndef HVX_ERF_H
#define HVX_ERF_H

#include "hvx-base.h"
#include "hvx-exp.h"
#include "hvx-inverse.h"
#include <math.h>

// Maximum error is about 1.5e-7 for the Abramowitz-Stegun approximation.
static __attribute__((noinline)) HVX_Vector hvx_vec_erf_f32(HVX_Vector x) {
    const HVX_Vector zero = hvx_vec_splat_f32(0.0f);
    const HVX_Vector ax = hvx_vec_abs_f32(x);
    HVX_Vector t = hvx_vec_inverse_f32(hvx_vec_add_f32_f32(
        hvx_vec_splat_f32(1.0f), hvx_vec_mul_f32_f32(hvx_vec_splat_f32(0.3275911f), ax)));

    HVX_Vector poly = hvx_vec_mul_f32_f32(hvx_vec_splat_f32(1.061405429f), t);
    poly = hvx_vec_add_f32_f32(hvx_vec_splat_f32(-1.453152027f), hvx_vec_mul_f32_f32(poly, t));
    poly = hvx_vec_add_f32_f32(hvx_vec_splat_f32(1.421413741f), hvx_vec_mul_f32_f32(poly, t));
    poly = hvx_vec_add_f32_f32(hvx_vec_splat_f32(-0.284496736f), hvx_vec_mul_f32_f32(poly, t));
    poly = hvx_vec_add_f32_f32(hvx_vec_splat_f32(0.254829592f), hvx_vec_mul_f32_f32(poly, t));

    const HVX_Vector exp_term = hvx_vec_exp_f32(hvx_vec_neg_f32(hvx_vec_mul_f32_f32(ax, ax)));
    HVX_Vector result = hvx_vec_sub_f32_f32(hvx_vec_splat_f32(1.0f),
        hvx_vec_mul_f32_f32(hvx_vec_mul_f32_f32(poly, t), exp_term));

    const HVX_VectorPred neg = Q6_Q_vcmp_gt_VsfVsf(zero, x);
    result = Q6_V_vmux_QVV(neg, hvx_vec_neg_f32(result), result);
    return result;
}

static inline HVX_Vector hvx_vec_gelu_erf_f32(HVX_Vector x) {
    const HVX_Vector scale = hvx_vec_splat_f32(0.7071067811865475f);
    const HVX_Vector half  = hvx_vec_splat_f32(0.5f);
    const HVX_Vector one   = hvx_vec_splat_f32(1.0f);
    const HVX_Vector max_x = hvx_vec_splat_f32(10.0f);
    const HVX_Vector min_x = hvx_vec_splat_f32(-10.0f);
    const HVX_VectorPred neg_large = Q6_Q_vcmp_gt_VsfVsf(min_x, x);
    const HVX_VectorPred pos_large = Q6_Q_vcmp_gt_VsfVsf(x, max_x);
    HVX_Vector x_calc = Q6_V_vmux_QVV(neg_large, min_x, x);
    x_calc = Q6_V_vmux_QVV(pos_large, max_x, x_calc);
    const HVX_Vector erf = hvx_vec_erf_f32(hvx_vec_mul_f32_f32(x_calc, scale));

    HVX_Vector result = hvx_vec_mul_f32_f32(hvx_vec_mul_f32_f32(half, x_calc), hvx_vec_add_f32_f32(one, erf));

    result = Q6_V_vmux_QVV(neg_large, hvx_vec_splat_f32(0.0f), result);
    result = Q6_V_vmux_QVV(pos_large, x, result);
    return result;
}

static inline void hvx_gelu_erf_f32_aa(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    // SM8850 / Hexagon v81: the recently-added vector GELU_ERF path is
    // numerically wrong on-device (the LocalDream startup test reproducibly
    // sees ~5.6e-2 downstream error). Breeze's codec transformer/decoder uses
    // GELU_ERF directly, so that error compounds until the vocoder becomes NaN.
    //
    // Keep execution on the DSP, but use Hexagon libm erff() per element for
    // the exact reference formula. This is a correctness-first v81 workaround;
    // only GELU_ERF is scalarized, all matmuls/attention/convolutions remain HVX.
    const float * restrict s = (const float *) src;
    float * restrict d = (float *) dst;
    const float inv_sqrt2 = 0.7071067811865475244f;

    for (uint32_t i = 0; i < n; ++i) {
        const float x = s[i];
        if (x < -10.0f) {
            d[i] = 0.0f;
        } else if (x > 10.0f) {
            d[i] = x;
        } else {
            d[i] = 0.5f * x * (1.0f + erff(x * inv_sqrt2));
        }
    }
}

#endif /* HVX_ERF_H */
