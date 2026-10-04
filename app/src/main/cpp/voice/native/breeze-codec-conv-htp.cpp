#include "breeze/codec.h"

#include <string>
#include <vector>

namespace breeze {
namespace codec_detail {

ggml_tensor * conv1d_causal(ggml_context * ctx, ggml_tensor * w, ggml_tensor * b,
                            ggml_tensor * x, int stride, int dilation) {
    const int K = (int) w->ne[0];
    int pad = (K - 1) * dilation - (stride - 1);
    if (pad < 0) pad = 0;
    if (!ggml_is_contiguous(x)) x = ggml_cont(ctx, x);
    const int64_t n_out = (x->ne[0] + pad - (int64_t) dilation * (K - 1) - 1) / stride + 1;
    ggml_tensor * y = ggml_conv_1d(ctx, w, x, stride, pad, dilation);
    if (y->ne[0] != n_out) {
        y = ggml_cont(ctx, ggml_view_2d(ctx, y, n_out, y->ne[1], y->nb[1], 0));
    }
    if (b) y = ggml_add(ctx, y, ggml_reshape_2d(ctx, b, 1, b->ne[0]));
    return y;
}


static ggml_tensor * elu_htp(ggml_context * ctx, ggml_tensor * x) {
    // Exact ELU(x, alpha=1) lowered to Hexagon-supported primitives:
    // relu(x) - relu(1 - exp(x)).
    ggml_tensor * zero = ggml_sub(ctx, x, x);
    ggml_tensor * one = ggml_exp(ctx, zero);
    ggml_tensor * negative = ggml_relu(ctx, ggml_sub(ctx, one, ggml_exp(ctx, x)));
    return ggml_sub(ctx, ggml_relu(ctx, x), negative);
}

ggml_tensor * convtr1d_causal(ggml_context * ctx, ggml_tensor * w, ggml_tensor * b,
                              ggml_tensor * x, int stride) {
    // Equivalent transpose-conv lowering:
    // [K, OC, IC] x [T, IC] -> GEMM [K*OC, T] -> COL2IM_1D.
    // This avoids GGML_OP_CONV_TRANSPOSE_1D, which the strict Hexagon backend
    // does not implement, without changing the model math.
    const int64_t kernel = w->ne[0];
    const int64_t out_channels = w->ne[1];
    const int64_t in_channels = w->ne[2];

    ggml_tensor * w2 = ggml_reshape_2d(ctx, w, kernel * out_channels, in_channels);
    w2 = ggml_cont(ctx, ggml_transpose(ctx, w2));
    ggml_tensor * xt = ggml_cont(ctx, ggml_transpose(ctx, x));
    ggml_tensor * projected = ggml_cont(ctx, ggml_mul_mat(ctx, w2, xt));
    ggml_tensor * y = ggml_col2im_1d(ctx, projected, stride, (int) out_channels, 0);

    const int keep = (int) x->ne[0] * stride;
    y = ggml_cont(ctx, ggml_view_2d(ctx, y, keep, y->ne[1], y->nb[1], 0));
    if (b) y = ggml_add(ctx, y, ggml_reshape_2d(ctx, b, 1, b->ne[0]));
    return y;
}

ggml_tensor * resnet_block(ggml_context * ctx, BreezeModel & m, const std::string & p, ggml_tensor * x) {
    ggml_tensor * h = elu_htp(ctx, x);
    h = conv1d_causal(ctx, m.w(p + ".block.1.conv.weight"), m.w(p + ".block.1.conv.bias"), h, 1, 1);
    h = elu_htp(ctx, h);
    h = conv1d_causal(ctx, m.w(p + ".block.3.conv.weight"), m.w(p + ".block.3.conv.bias"), h, 1, 1);
    return ggml_add(ctx, x, h);
}

// depthwise conv as shift and multiply, ggml conv ops have no group support
// x is [L, C], w is [C, K]
ggml_tensor * depthwise1d_causal(ggml_context * ctx, ggml_tensor * w, ggml_tensor * b,
                                 ggml_tensor * x, int K) {
    const int64_t L = x->ne[0];
    const int64_t C = x->ne[1];
    ggml_tensor * xp = ggml_pad_ext(ctx, x, 0, K - 1, 0, 0, 0, 0, 0, 0);
    xp = ggml_roll(ctx, xp, K - 1, 0, 0, 0);
    ggml_tensor * acc = nullptr;
    for (int k = 0; k < K; k++) {
        ggml_tensor * seg = ggml_cont(ctx, ggml_view_2d(ctx, xp, L, C, xp->nb[1], (size_t) k * xp->nb[0]));
        ggml_tensor * wk = ggml_cont(ctx, ggml_transpose(ctx,
            ggml_view_2d(ctx, w, C, 1, w->nb[1], (size_t) k * w->nb[1])));
        ggml_tensor * t = ggml_mul(ctx, seg, wk);
        acc = acc ? ggml_add(ctx, acc, t) : t;
    }
    if (b) acc = ggml_add(ctx, acc, ggml_reshape_2d(ctx, b, 1, b->ne[0]));
    return acc;
}

ggml_tensor * seanet_encoder(ggml_context * ctx, BreezeModel & m, ggml_tensor * x) {
    const std::vector<int> & up = m.cfg.codec.upsampling_ratios;
    auto conv = [&](int i, ggml_tensor * in, int s) {
        const std::string p = "codec.enc." + std::to_string(i);
        return conv1d_causal(ctx, m.w(p + ".conv.weight"), m.w(p + ".conv.bias"), in, s, 1);
    };
    const int conv_idx[4] = { 3, 6, 9, 12 };
    const int res_idx[4] = { 1, 4, 7, 10 };
    const int ratios[4] = { up[3], up[2], up[1], up[0] };
    ggml_tensor * h = conv(0, x, 1);
    for (int s = 0; s < 4; s++) {
        h = resnet_block(ctx, m, "codec.enc." + std::to_string(res_idx[s]), h);
        h = elu_htp(ctx, h);
        h = conv(conv_idx[s], h, ratios[s]);
    }
    h = elu_htp(ctx, h);
    return conv(14, h, 1);
}

}
}
