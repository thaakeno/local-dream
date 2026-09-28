#include "ggml.h"
#include "ggml-backend.h"
#include "src/backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static double now_ms() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}

int main() {
    setenv("GGML_BACKEND", "HTP0", 1);

    BackendPair bp = backend_init("TRANSPORT-BENCH");
    if (!bp.backend || std::strcmp(ggml_backend_name(bp.backend), "HTP0") != 0) {
        std::fprintf(stderr, "BENCH_ERROR backend=%s\n", bp.backend ? ggml_backend_name(bp.backend) : "null");
        return 2;
    }

    constexpr int K = 512;
    constexpr int N = 64;
    constexpr int LAYERS = 12;
    constexpr int WARMUP = 2;
    constexpr int RUNS = 9;

    const size_t ctx_size =
        ggml_tensor_overhead() * 128 + ggml_graph_overhead_custom(256, false) + 4096;
    std::vector<unsigned char> arena(ctx_size);
    ggml_init_params params = { ctx_size, arena.data(), true };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) return 3;

    ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, K);
    ggml_tensor * bias = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, 1);
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);
    ggml_set_name(w, "bench_weight");
    ggml_set_name(bias, "bench_bias");
    ggml_set_name(x, "bench_input");
    ggml_set_input(w);
    ggml_set_input(bias);
    ggml_set_input(x);

    ggml_tensor * y = x;
    for (int i = 0; i < LAYERS; ++i) {
        y = ggml_mul_mat(ctx, w, y);
        y = ggml_add(ctx, y, bias);
        y = ggml_tanh(ctx, y);
    }
    ggml_set_name(y, "bench_output");
    ggml_set_output(y);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 256, false);
    ggml_build_forward_expand(graph, y);

    ggml_backend_sched_t sched = backend_sched_new(bp, 256);
    ggml_backend_sched_set_tensor_backend(sched, w, bp.backend);
    ggml_backend_sched_set_tensor_backend(sched, bias, bp.backend);
    ggml_backend_sched_set_tensor_backend(sched, x, bp.backend);
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        ggml_backend_sched_set_tensor_backend(sched, ggml_graph_node(graph, i), bp.backend);
    }

    if (!ggml_backend_sched_alloc_graph(sched, graph)) {
        std::fprintf(stderr, "BENCH_ERROR alloc_graph\n");
        return 4;
    }

    std::vector<float> wdata((size_t) K * K);
    std::vector<float> bdata(K);
    std::vector<float> xdata((size_t) K * N);
    uint32_t rng = 0x12345678u;
    auto next = [&]() {
        rng = rng * 1664525u + 1013904223u;
        return ((float) ((rng >> 8) & 0xffffffu) / 16777216.0f - 0.5f) * 0.08f;
    };
    for (float & v : wdata) v = next();
    for (float & v : bdata) v = next();
    for (float & v : xdata) v = next();

    ggml_backend_tensor_set(w, wdata.data(), 0, wdata.size() * sizeof(float));
    ggml_backend_tensor_set(bias, bdata.data(), 0, bdata.size() * sizeof(float));
    ggml_backend_tensor_set(x, xdata.data(), 0, xdata.size() * sizeof(float));

    for (int i = 0; i < WARMUP; ++i) {
        if (ggml_backend_sched_graph_compute(sched, graph) != GGML_STATUS_SUCCESS) return 5;
        ggml_backend_synchronize(bp.backend);
    }

    std::vector<double> times;
    times.reserve(RUNS);
    for (int i = 0; i < RUNS; ++i) {
        const double t0 = now_ms();
        const enum ggml_status st = ggml_backend_sched_graph_compute(sched, graph);
        ggml_backend_synchronize(bp.backend);
        const double dt = now_ms() - t0;
        if (st != GGML_STATUS_SUCCESS) return 6;
        times.push_back(dt);
    }

    std::sort(times.begin(), times.end());
    const double median = times[times.size() / 2];
    double sum = 0.0;
    for (double t : times) sum += t;
    const double mean = sum / times.size();

    std::vector<float> out(16);
    ggml_backend_tensor_get(y, out.data(), 0, out.size() * sizeof(float));
    double checksum = 0.0;
    for (float v : out) checksum += v;

#if defined(GGML_HEXAGON_USE_MEMPOOL)
    const char * transport = "fastrpc";
#else
    const char * transport = "dspqueue";
#endif

    std::printf(
        "BENCH_JSON {\"transport\":\"%s\",\"median_ms\":%.3f,"
        "\"mean_ms\":%.3f,\"runs\":%d,\"nodes\":%d,\"checksum\":%.9f}\n",
        transport, median, mean, RUNS, ggml_graph_n_nodes(graph), checksum);
    std::fflush(stdout);

    ggml_backend_sched_free(sched);
    ggml_free(ctx);
    backend_release(bp.backend, bp.cpu_backend);
    return 0;
}
