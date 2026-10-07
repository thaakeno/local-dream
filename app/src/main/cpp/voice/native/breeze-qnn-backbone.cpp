#include "breeze/qnn_backbone.h"

#include "breeze/model.h"

#include <HTP/QnnHtpDevice.h>
#include <QnnSampleApp.hpp>
#include <QnnTypeMacros.hpp>

#include "DynamicLoadUtil.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <memory>
#include <string>
#include <vector>

using namespace qnn::tools::sample_app;

namespace breeze {
namespace {

constexpr int kBackboneBucket = 512;

struct GraphIo {
    uint32_t graph_index = 0;
    Qnn_Tensor_t * inputs = nullptr;
    Qnn_Tensor_t * outputs = nullptr;
    bool valid = false;
};

struct NativeBlob {
    Qnn_DataType_t type = QNN_DATATYPE_UNDEFINED;
    std::vector<uint32_t> dims;
    std::vector<unsigned char> bytes;
};

struct PrefillCache {
    NativeBlob key;
    NativeBlob value;
    int valid_tokens = 0;
};

static bool name_contains(const char * name, const char * needle) {
    return name && needle && std::string(name).find(needle) != std::string::npos;
}

static int tensor_index(
    Qnn_Tensor_t * tensors,
    uint32_t count,
    const char * needle,
    int fallback = -1
) {
    for (uint32_t i = 0; i < count; ++i) {
        if (name_contains(QNN_TENSOR_GET_NAME(tensors[i]), needle)) return (int) i;
    }
    return fallback >= 0 && fallback < (int) count ? fallback : -1;
}

static size_t tensor_elements(const Qnn_Tensor_t & tensor) {
    const uint32_t rank = QNN_TENSOR_GET_RANK(tensor);
    const uint32_t * dims = QNN_TENSOR_GET_DIMENSIONS(tensor);
    size_t n = 1;
    for (uint32_t i = 0; i < rank; ++i) n *= dims ? dims[i] : 1u;
    return n;
}

static bool put_i32_array(Qnn_Tensor_t & tensor, const int32_t * values, size_t count) {
    auto buf = QNN_TENSOR_GET_CLIENT_BUF(tensor);
    const size_t bytes = count * sizeof(int32_t);
    if (!buf.data || buf.dataSize < bytes) return false;
    std::memcpy(buf.data, values, bytes);
    return true;
}

static bool capture_native(const Qnn_Tensor_t & tensor, NativeBlob & out) {
    const uint32_t rank = QNN_TENSOR_GET_RANK(tensor);
    const uint32_t * dims = QNN_TENSOR_GET_DIMENSIONS(tensor);
    const auto buf = QNN_TENSOR_GET_CLIENT_BUF(tensor);
    if (!buf.data || !buf.dataSize || !dims || !rank) return false;
    out.type = QNN_TENSOR_GET_DATA_TYPE(tensor);
    out.dims.assign(dims, dims + rank);
    out.bytes.resize(buf.dataSize);
    std::memcpy(out.bytes.data(), buf.data, buf.dataSize);
    return true;
}

static bool finite_nonzero(const std::vector<float> & values) {
    if (values.empty()) return false;
    bool nonzero = false;
    for (float x : values) {
        if (!std::isfinite(x)) return false;
        nonzero = nonzero || std::fabs(x) > 1.0e-9f;
    }
    return nonzero;
}

class BreezeQnnBackboneApp final : public QnnSampleApp {
public:
    void * model_handle = nullptr;
    uint32_t power_config_id = 0;
    bool power_config_active = false;
    GraphIo graph;
    std::string expected_graph;

    BreezeQnnBackboneApp(
        QnnFunctionPointers qnnFunctionPointers,
        void * backendHandle,
        const std::string & cachedBinaryPath,
        std::string expected
    ) : QnnSampleApp(
            qnnFunctionPointers,
            "",
            "",
            backendHandle,
            "",
            false,
            qnn::tools::iotensor::OutputDataType::FLOAT_ONLY,
            qnn::tools::iotensor::InputDataType::FLOAT,
            ProfilingLevel::OFF,
            false,
            cachedBinaryPath,
            ""
        ),
        expected_graph(std::move(expected)) {}

    ~BreezeQnnBackboneApp() {
        tear_down();
        if (m_graphsInfo) freeContext();
        release_power_vote();
        freeDevice();
        terminateBackend();
        if (model_handle) {
            dlclose(model_handle);
            model_handle = nullptr;
        }
    }

    void tear_down() {
        if (!graph.valid || (!graph.inputs && !graph.outputs) ||
            !m_graphsInfo || graph.graph_index >= m_graphsCount) {
            graph.inputs = nullptr;
            graph.outputs = nullptr;
            graph.valid = false;
            return;
        }
        auto & g = (*m_graphsInfo)[graph.graph_index];
        m_ioTensor.tearDownInputAndOutputTensors(
            graph.inputs,
            graph.outputs,
            g.numInputTensors,
            g.numOutputTensors
        );
        graph.inputs = nullptr;
        graph.outputs = nullptr;
        graph.valid = false;
    }

    bool setup_graph() {
        if (!m_graphsInfo || m_graphsCount < 1) return false;
        int found = -1;
        for (uint32_t i = 0; i < m_graphsCount; ++i) {
            const char * name = (*m_graphsInfo)[i].graphName;
            std::fprintf(
                stderr,
                "[BREEZE_QNN_BACKBONE] context=%s graph[%u]=%s\n",
                expected_graph.c_str(),
                i,
                name ? name : "<unnamed>"
            );
            if (name_contains(name, expected_graph.c_str())) found = (int) i;
        }
        if (found < 0 && m_graphsCount == 1) {
            // Direct AI Hub qnn_context_binary compilation may normalize the
            // graph name. Each v3 file intentionally contains exactly one graph.
            found = 0;
        }
        if (found < 0) return false;
        graph.graph_index = (uint32_t) found;
        auto & g = (*m_graphsInfo)[graph.graph_index];
        const auto rc = m_ioTensor.setupInputAndOutputTensors(
            &graph.inputs,
            &graph.outputs,
            g
        );
        graph.valid = rc == qnn::tools::iotensor::StatusCode::SUCCESS;
        return graph.valid;
    }

    void release_power_vote() {
        if (!power_config_active) return;
        auto qnn = m_qnnFunctionPointers.qnnInterface;
        QnnDevice_Infrastructure_t deviceInfra = nullptr;
        if (
            qnn.deviceGetInfrastructure &&
            qnn.deviceGetInfrastructure(&deviceInfra) == QNN_SUCCESS &&
            deviceInfra
        ) {
            auto * htp = static_cast<QnnHtpDevice_Infrastructure_t *>(deviceInfra);
            htp->perfInfra.destroyPowerConfigId(power_config_id);
        }
        power_config_id = 0;
        power_config_active = false;
    }

    // Balanced latency/power policy for sustained on-device TTS.
    // The old "burst" vote disabled DCVS + HTP sleep, pinned bus/core at the
    // maximum voltage corner and busy-polled FastRPC for 9.999 ms. That is a
    // benchmark profile, not a sensible long-running mobile inference policy.
    // Keep PERFORMANCE_MODE, but let DCVS scale clocks, allow HTP sleep between
    // graphs, and use a 200 us RPC control-latency vote instead of polling.
    bool set_adaptive_power() {
        auto qnn = m_qnnFunctionPointers.qnnInterface;
        QnnDevice_Infrastructure_t deviceInfra = nullptr;
        if (
            !qnn.deviceGetInfrastructure ||
            qnn.deviceGetInfrastructure(&deviceInfra) != QNN_SUCCESS ||
            !deviceInfra
        ) {
            return false;
        }

        auto * htp = static_cast<QnnHtpDevice_Infrastructure_t *>(deviceInfra);
        auto perf = htp->perfInfra;
        if (power_config_active) return true;
        if (perf.createPowerConfigId(0, 0, &power_config_id) != QNN_SUCCESS) {
            power_config_id = 0;
            return false;
        }
        power_config_active = true;

        QnnHtpPerfInfrastructure_PowerConfig_t dcvs{};
        dcvs.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3;
        dcvs.dcvsV3Config.contextId = power_config_id;
        dcvs.dcvsV3Config.powerMode =
            QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE;
        dcvs.dcvsV3Config.setDcvsEnable = 1;
        dcvs.dcvsV3Config.dcvsEnable = 1;
        dcvs.dcvsV3Config.setSleepDisable = 1;
        dcvs.dcvsV3Config.sleepDisable = 0;
        dcvs.dcvsV3Config.setSleepLatency = 1;
        dcvs.dcvsV3Config.sleepLatency = 200;
        dcvs.dcvsV3Config.setBusParams = 0;
        dcvs.dcvsV3Config.setCoreParams = 0;

        QnnHtpPerfInfrastructure_PowerConfig_t rpc{};
        rpc.option =
            QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_CONTROL_LATENCY;
        rpc.rpcControlLatencyConfig = 200;

        const QnnHtpPerfInfrastructure_PowerConfig_t * configs[] = {
            &dcvs,
            &rpc,
            nullptr,
        };
        const bool ok =
            perf.setPowerConfig(power_config_id, configs) == QNN_SUCCESS;
        if (!ok) release_power_vote();
        std::fprintf(
            stderr,
            "[BREEZE_QNN_POWER] profile=adaptive-performance dcvs=1 sleep=1 "
            "rpc_poll=0 rpc_latency_us=200 max_corner_pin=0 ok=%d\n",
            ok ? 1 : 0
        );
        return ok;
    }

    bool execute(double & ms) {
        auto & g = (*m_graphsInfo)[graph.graph_index];
        const auto t0 = std::chrono::steady_clock::now();
        const auto rc = m_qnnFunctionPointers.qnnInterface.graphExecute(
            g.graph,
            graph.inputs,
            g.numInputTensors,
            graph.outputs,
            g.numOutputTensors,
            m_profileBackendHandle,
            nullptr
        );
        ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0
        ).count();
        return rc == QNN_GRAPH_NO_ERROR;
    }

    bool run_prefill(
        const std::vector<float> & embeddings,
        int tokens,
        int hidden,
        StepOut & out,
        PrefillCache & cache,
        double & ms
    ) {
        auto & g = (*m_graphsInfo)[graph.graph_index];
        if (g.numInputTensors != 3 || g.numOutputTensors != 4) return false;
        if (tokens <= 0 || tokens > kBackboneBucket) return false;
        if ((int) embeddings.size() != tokens * hidden) return false;

        const int ei = tensor_index(graph.inputs, g.numInputTensors, "inputs_embeds", 0);
        const int mi = tensor_index(graph.inputs, g.numInputTensors, "attention_mask", 1);
        const int pi = tensor_index(graph.inputs, g.numInputTensors, "positions", 2);
        const int ho = tensor_index(graph.outputs, g.numOutputTensors, "hidden", 0);
        const int lo = tensor_index(graph.outputs, g.numOutputTensors, "logits", 1);
        const int ko = tensor_index(graph.outputs, g.numOutputTensors, "key_cache", 2);
        const int vo = tensor_index(graph.outputs, g.numOutputTensors, "value_cache", 3);
        if (ei < 0 || mi < 0 || pi < 0 || ho < 0 || lo < 0 || ko < 0 || vo < 0) {
            return false;
        }

        const int pad = kBackboneBucket - tokens;
        std::vector<float> padded(
            (size_t) kBackboneBucket * (size_t) hidden,
            0.0f
        );
        std::memcpy(
            padded.data() + (size_t) pad * (size_t) hidden,
            embeddings.data(),
            embeddings.size() * sizeof(float)
        );

        std::vector<float> mask(
            (size_t) kBackboneBucket * (size_t) kBackboneBucket,
            -10000.0f
        );
        for (int q = 0; q < kBackboneBucket; ++q) {
            if (q < pad) {
                mask[(size_t) q * kBackboneBucket + q] = 0.0f;
            } else {
                for (int k = pad; k <= q; ++k) {
                    mask[(size_t) q * kBackboneBucket + k] = 0.0f;
                }
            }
        }

        std::vector<int32_t> positions((size_t) kBackboneBucket, 0);
        for (int i = 0; i < tokens; ++i) positions[(size_t) pad + i] = i;

        if (m_ioTensor.copyFromFloatToNative(
                padded.data(),
                &graph.inputs[ei]) != qnn::tools::iotensor::StatusCode::SUCCESS) {
            return false;
        }
        if (m_ioTensor.copyFromFloatToNative(
                mask.data(),
                &graph.inputs[mi]) != qnn::tools::iotensor::StatusCode::SUCCESS) {
            return false;
        }
        if (!put_i32_array(graph.inputs[pi], positions.data(), positions.size())) {
            return false;
        }

        if (!execute(ms)) return false;

        out.hidden.assign(tensor_elements(graph.outputs[ho]), 0.0f);
        out.logits.assign(tensor_elements(graph.outputs[lo]), 0.0f);
        if (m_ioTensor.convertToFloatInto(
                out.hidden.data(),
                &graph.outputs[ho]) != qnn::tools::iotensor::StatusCode::SUCCESS) {
            return false;
        }
        if (m_ioTensor.convertToFloatInto(
                out.logits.data(),
                &graph.outputs[lo]) != qnn::tools::iotensor::StatusCode::SUCCESS) {
            return false;
        }
        cache.valid_tokens = tokens;
        return finite_nonzero(out.hidden) &&
            finite_nonzero(out.logits) &&
            capture_native(graph.outputs[ko], cache.key) &&
            capture_native(graph.outputs[vo], cache.value);
    }

    bool zero_step_cache() {
        auto & g = (*m_graphsInfo)[graph.graph_index];
        if (g.numInputTensors != 5 || g.numOutputTensors != 4) return false;
        const int ki = tensor_index(graph.inputs, g.numInputTensors, "key_cache", 2);
        const int vi = tensor_index(graph.inputs, g.numInputTensors, "value_cache", 3);
        if (ki < 0 || vi < 0) return false;
        for (int index : {ki, vi}) {
            auto buf = QNN_TENSOR_GET_CLIENT_BUF(graph.inputs[index]);
            if (!buf.data || !buf.dataSize) return false;
            std::memset(buf.data, 0, buf.dataSize);
        }
        return true;
    }

    bool seed_branch(const PrefillCache & source, int branch) {
        auto & g = (*m_graphsInfo)[graph.graph_index];
        const int ki = tensor_index(graph.inputs, g.numInputTensors, "key_cache", 2);
        const int vi = tensor_index(graph.inputs, g.numInputTensors, "value_cache", 3);
        if (ki < 0 || vi < 0) return false;

        auto copy = [&](const NativeBlob & src, Qnn_Tensor_t & dst) {
            if (src.type != QNN_TENSOR_GET_DATA_TYPE(dst)) return false;
            const uint32_t rank = QNN_TENSOR_GET_RANK(dst);
            const uint32_t * dd = QNN_TENSOR_GET_DIMENSIONS(dst);
            if (!dd || rank != 4 || src.dims.size() != 4) return false;

            const int src_batch = (int) src.dims[0];
            const int lkv = (int) src.dims[1];
            const int src_seq = (int) src.dims[2];
            const int head_dim = (int) src.dims[3];
            const int dst_batch = (int) dd[0];
            const int dst_lkv = (int) dd[1];
            const int dst_seq = (int) dd[2];
            const int dst_dim = (int) dd[3];
            if (
                src_batch != 1 ||
                src_seq != kBackboneBucket ||
                lkv != dst_lkv ||
                head_dim != dst_dim ||
                branch < 0 ||
                branch >= dst_batch ||
                source.valid_tokens <= 0 ||
                source.valid_tokens > src_seq ||
                source.valid_tokens > dst_seq
            ) {
                return false;
            }

            auto db = QNN_TENSOR_GET_CLIENT_BUF(dst);
            if (!db.data || !db.dataSize) return false;
            const size_t src_elements =
                (size_t) src_batch * lkv * src_seq * head_dim;
            const size_t dst_elements =
                (size_t) dst_batch * dst_lkv * dst_seq * dst_dim;
            if (
                !src_elements ||
                !dst_elements ||
                src.bytes.size() % src_elements ||
                db.dataSize % dst_elements
            ) {
                return false;
            }
            const size_t bytes_per = src.bytes.size() / src_elements;
            if (bytes_per != db.dataSize / dst_elements) return false;

            const int pad = src_seq - source.valid_tokens;
            const size_t row = (size_t) head_dim * bytes_per;
            const auto * sp = src.bytes.data();
            auto * dp = static_cast<unsigned char *>(db.data);
            for (int h = 0; h < lkv; ++h) {
                const size_t src_offset =
                    ((size_t) h * src_seq + (size_t) pad) * row;
                const size_t dst_offset =
                    (((size_t) branch * lkv + (size_t) h) * dst_seq) * row;
                std::memcpy(
                    dp + dst_offset,
                    sp + src_offset,
                    (size_t) source.valid_tokens * row
                );
            }
            return true;
        };

        return copy(source.key, graph.inputs[ki]) &&
            copy(source.value, graph.inputs[vi]);
    }

    int step_batch_size() const {
        if (!graph.valid || !m_graphsInfo || graph.graph_index >= m_graphsCount) return 0;
        auto & g = (*m_graphsInfo)[graph.graph_index];
        const int ei = tensor_index(graph.inputs, g.numInputTensors, "input_embed", 0);
        if (ei < 0) return 0;
        const uint32_t * dims = QNN_TENSOR_GET_DIMENSIONS(graph.inputs[ei]);
        return dims && QNN_TENSOR_GET_RANK(graph.inputs[ei]) >= 1 ? (int) dims[0] : 0;
    }

    bool copy_new_cache(const std::vector<int32_t> & positions) {
        auto & g = (*m_graphsInfo)[graph.graph_index];
        const int ki = tensor_index(graph.inputs, g.numInputTensors, "key_cache", 2);
        const int vi = tensor_index(graph.inputs, g.numInputTensors, "value_cache", 3);
        const int ko = tensor_index(graph.outputs, g.numOutputTensors, "new_key", 2);
        const int vo = tensor_index(graph.outputs, g.numOutputTensors, "new_value", 3);
        if (ki < 0 || vi < 0 || ko < 0 || vo < 0) return false;

        auto copy = [&](Qnn_Tensor_t & src, Qnn_Tensor_t & dst, const char * label) {
            const uint32_t * sd = QNN_TENSOR_GET_DIMENSIONS(src);
            const uint32_t * dd = QNN_TENSOR_GET_DIMENSIONS(dst);
            if (!sd || !dd || QNN_TENSOR_GET_RANK(src) != 4 ||
                QNN_TENSOR_GET_RANK(dst) != 4) {
                std::fprintf(stderr, "[BREEZE_QNN_BACKBONE] %s cache rank mismatch\n", label);
                return false;
            }

            const int batch = (int) sd[0];
            const int lkv = (int) sd[1];
            const int src_seq = (int) sd[2];
            const int head_dim = (int) sd[3];
            const int dst_seq = (int) dd[2];
            if (
                batch <= 0 ||
                src_seq != 1 ||
                positions.size() != (size_t) batch ||
                (int) dd[0] != batch ||
                (int) dd[1] != lkv ||
                (int) dd[3] != head_dim
            ) {
                std::fprintf(
                    stderr,
                    "[BREEZE_QNN_BACKBONE] %s cache shape mismatch "
                    "src=%dx%dx%dx%d dst=%ux%ux%ux%u positions=%zu\n",
                    label, batch, lkv, src_seq, head_dim,
                    dd[0], dd[1], dd[2], dd[3], positions.size()
                );
                return false;
            }
            if (QNN_TENSOR_GET_DATA_TYPE(src) != QNN_TENSOR_GET_DATA_TYPE(dst)) {
                std::fprintf(stderr, "[BREEZE_QNN_BACKBONE] %s cache dtype mismatch src=%d dst=%d\n",
                             label, (int) QNN_TENSOR_GET_DATA_TYPE(src),
                             (int) QNN_TENSOR_GET_DATA_TYPE(dst));
                return false;
            }

            auto sb = QNN_TENSOR_GET_CLIENT_BUF(src);
            auto db = QNN_TENSOR_GET_CLIENT_BUF(dst);
            const size_t src_elements = tensor_elements(src);
            const size_t dst_elements = tensor_elements(dst);
            if (
                !sb.data ||
                !db.data ||
                !src_elements ||
                !dst_elements ||
                sb.dataSize % src_elements ||
                db.dataSize % dst_elements
            ) {
                std::fprintf(stderr, "[BREEZE_QNN_BACKBONE] %s cache buffer layout invalid src_bytes=%u dst_bytes=%u\n",
                             label, sb.dataSize, db.dataSize);
                return false;
            }
            const size_t bytes_per = sb.dataSize / src_elements;
            if (bytes_per != db.dataSize / dst_elements) {
                std::fprintf(stderr, "[BREEZE_QNN_BACKBONE] %s cache element width mismatch\n", label);
                return false;
            }

            const size_t row = (size_t) head_dim * bytes_per;
            auto * sp = static_cast<const unsigned char *>(sb.data);
            auto * dp = static_cast<unsigned char *>(db.data);
            for (int b = 0; b < batch; ++b) {
                const int p = positions[(size_t) b];
                if (p < 0 || p >= dst_seq) return false;
                for (int h = 0; h < lkv; ++h) {
                    const size_t src_offset =
                        ((size_t) b * lkv + (size_t) h) * row;
                    const size_t dst_offset =
                        (((size_t) b * lkv + (size_t) h) * dst_seq + (size_t) p) * row;
                    std::memcpy(dp + dst_offset, sp + src_offset, row);
                }
            }
            return true;
        };

        return copy(graph.outputs[ko], graph.inputs[ki], "key") &&
            copy(graph.outputs[vo], graph.inputs[vi], "value");
    }

    bool run_step(
        int hidden,
        const std::vector<float> & embedding,
        std::vector<int32_t> & positions,
        std::vector<StepOut> & outs,
        double & ms
    ) {
        auto & g = (*m_graphsInfo)[graph.graph_index];
        if (g.numInputTensors != 5 || g.numOutputTensors != 4) return false;
        const int graph_batch = step_batch_size();
        if ((int) embedding.size() != hidden || graph_batch <= 0 ||
            positions.size() != (size_t) graph_batch) return false;
        for (int p : positions) if (p < 0 || p >= kBackboneBucket) return false;

        const int ei = tensor_index(graph.inputs, g.numInputTensors, "input_embed", 0);
        const int pi = tensor_index(graph.inputs, g.numInputTensors, "positions", 1);
        const int mi = tensor_index(graph.inputs, g.numInputTensors, "attention_mask", 4);
        const int ho = tensor_index(graph.outputs, g.numOutputTensors, "hidden", 0);
        const int lo = tensor_index(graph.outputs, g.numOutputTensors, "logits", 1);
        if (ei < 0 || pi < 0 || mi < 0 || ho < 0 || lo < 0) return false;

        std::vector<float> embeds((size_t) graph_batch * (size_t) hidden);
        for (int b = 0; b < graph_batch; ++b) {
            std::copy(
                embedding.begin(),
                embedding.end(),
                embeds.begin() + (size_t) b * (size_t) hidden
            );
        }

        std::vector<float> mask((size_t) graph_batch * 513u, -10000.0f);
        for (int b = 0; b < graph_batch; ++b) {
            float * row = mask.data() + (size_t) b * 513u;
            for (int i = 0; i < positions[(size_t) b]; ++i) row[i] = 0.0f;
            row[512] = 0.0f;
        }

        if (m_ioTensor.copyFromFloatToNative(
                embeds.data(),
                &graph.inputs[ei]) != qnn::tools::iotensor::StatusCode::SUCCESS) {
            return false;
        }
        if (!put_i32_array(graph.inputs[pi], positions.data(), positions.size())) {
            return false;
        }
        if (m_ioTensor.copyFromFloatToNative(
                mask.data(),
                &graph.inputs[mi]) != qnn::tools::iotensor::StatusCode::SUCCESS) {
            return false;
        }

        if (!execute(ms)) return false;

        std::vector<float> hidden_all(tensor_elements(graph.outputs[ho]), 0.0f);
        std::vector<float> logits_all(tensor_elements(graph.outputs[lo]), 0.0f);
        if (m_ioTensor.convertToFloatInto(
                hidden_all.data(),
                &graph.outputs[ho]) != qnn::tools::iotensor::StatusCode::SUCCESS) {
            return false;
        }
        if (m_ioTensor.convertToFloatInto(
                logits_all.data(),
                &graph.outputs[lo]) != qnn::tools::iotensor::StatusCode::SUCCESS) {
            return false;
        }
        if (!finite_nonzero(hidden_all) || !finite_nonzero(logits_all)) return false;
        if (hidden_all.size() != (size_t) graph_batch * (size_t) hidden ||
            logits_all.size() % (size_t) graph_batch != 0) {
            return false;
        }

        const size_t logits_per_branch = logits_all.size() / (size_t) graph_batch;
        outs.assign((size_t) graph_batch, StepOut{});
        for (int b = 0; b < graph_batch; ++b) {
            outs[(size_t) b].hidden.assign(
                hidden_all.begin() + (size_t) b * hidden,
                hidden_all.begin() + (size_t) (b + 1) * hidden
            );
            outs[(size_t) b].logits.assign(
                logits_all.begin() + (size_t) b * logits_per_branch,
                logits_all.begin() + (size_t) (b + 1) * logits_per_branch
            );
        }

        if (!copy_new_cache(positions)) return false;
        for (auto & p : positions) ++p;
        return true;
    }
};

static bool load_backbone_app(
    const std::string & lib_dir,
    const std::string & context_path,
    const std::string & graph_name,
    std::unique_ptr<BreezeQnnBackboneApp> & app
) {
    QnnFunctionPointers systemFuncs;
    const std::string systemPath = lib_dir + "/libQnnSystem.so";
    const std::string backendPath = lib_dir + "/libQnnHtp.so";

    if (qnn::tools::dynamicloadutil::getQnnSystemFunctionPointers(
            systemPath,
            &systemFuncs) != qnn::tools::dynamicloadutil::StatusCode::SUCCESS) {
        return false;
    }

    QnnFunctionPointers funcs;
    void * backendHandle = nullptr;
    void * modelHandle = nullptr;
    if (qnn::tools::dynamicloadutil::getQnnFunctionPointers(
            backendPath,
            context_path,
            &funcs,
            &backendHandle,
            false,
            &modelHandle) != qnn::tools::dynamicloadutil::StatusCode::SUCCESS) {
        return false;
    }
    funcs.qnnSystemInterface = systemFuncs.qnnSystemInterface;

    auto candidate = std::make_unique<BreezeQnnBackboneApp>(
        funcs,
        backendHandle,
        context_path,
        graph_name
    );
    candidate->model_handle = modelHandle;
    auto fail = [&]() {
        candidate.reset();
        return false;
    };

    if (candidate->initialize() != StatusCode::SUCCESS) return fail();
    if (candidate->initializeBackend() != StatusCode::SUCCESS) return fail();
    if (candidate->createDevice() != StatusCode::SUCCESS) return fail();
    if (candidate->initializeProfiling() != StatusCode::SUCCESS) return fail();
    if (candidate->registerOpPackages() != StatusCode::SUCCESS) return fail();
    if (candidate->createFromBinary() != StatusCode::SUCCESS) return fail();
    if (!candidate->setup_graph()) return fail();
    candidate->set_adaptive_power();

    app = std::move(candidate);
    return true;
}

} // namespace

struct QnnBackboneRunner::Impl {
    std::unique_ptr<BreezeQnnBackboneApp> step;
    std::string lib_dir;
    std::string prefill_path;
    std::string step_path;
    bool enabled = false;
    int branches = 1;
    int max_seq = kBackboneBucket;
    std::vector<int32_t> positions;
    size_t frames = 0;
    double total_ms = 0.0;
};

QnnBackboneRunner::QnnBackboneRunner() : impl_(std::make_unique<Impl>()) {}
QnnBackboneRunner::~QnnBackboneRunner() = default;

bool QnnBackboneRunner::init(BreezeModel & m, int branches) {
    (void) m;
    if (branches != 1 && branches != 2) return false;
    const char * prefill_path = std::getenv("BREEZE_QNN_BACKBONE_PREFILL_PATH");
    const char * step_path = std::getenv("BREEZE_QNN_BACKBONE_STEP_PATH");
    const char * lib = std::getenv("BREEZE_QNN_LIB_DIR");
    if (
        !prefill_path || !*prefill_path ||
        !step_path || !*step_path ||
        !lib || !*lib
    ) {
        return false;
    }

    impl_->lib_dir = lib;
    impl_->prefill_path = prefill_path;
    impl_->step_path = step_path;
    impl_->branches = branches;
    impl_->positions.assign(2, 0);
    impl_->enabled = true;
    std::fprintf(
        stderr,
        "[BREEZE_QNN_BACKBONE] configured logical_branches=%d "
        "prefill_context=separate step_context=separate\n",
        branches
    );
    return true;
}

bool QnnBackboneRunner::ready() const {
    return impl_ && impl_->enabled;
}

int QnnBackboneRunner::max_seq() const {
    return impl_ ? impl_->max_seq : 0;
}

void QnnBackboneRunner::disable() {
    if (!impl_) return;
    impl_->enabled = false;
    impl_->step.reset();
    impl_->positions.clear();
}

bool QnnBackboneRunner::prefill(
    BreezeModel & m,
    const std::vector<float> & cond_embeddings,
    int cond_tokens,
    const std::vector<float> * uncond_embeddings,
    int uncond_tokens,
    StepOut & cond_out,
    StepOut & uncond_out
) {
    if (!ready()) return false;
    const int hidden = m.cfg.hidden_size;
    if (
        cond_tokens <= 0 ||
        cond_tokens > impl_->max_seq ||
        (int) cond_embeddings.size() != cond_tokens * hidden
    ) {
        return false;
    }
    if (
        impl_->branches == 2 &&
        (!uncond_embeddings ||
         uncond_tokens <= 0 ||
         uncond_tokens > impl_->max_seq ||
         (int) uncond_embeddings->size() != uncond_tokens * hidden)
    ) {
        return false;
    }

    std::unique_ptr<BreezeQnnBackboneApp> prefill_app;
    if (!load_backbone_app(
            impl_->lib_dir,
            impl_->prefill_path,
            "backbone_prefill_512",
            prefill_app)) {
        std::fprintf(stderr, "[BREEZE_QNN_BACKBONE] prefill context load failed\n");
        disable();
        return false;
    }

    PrefillCache cond_cache;
    PrefillCache uncond_cache;
    double ms = 0.0;
    double prefill_ms = 0.0;
    if (!prefill_app->run_prefill(
            cond_embeddings,
            cond_tokens,
            hidden,
            cond_out,
            cond_cache,
            ms)) {
        std::fprintf(stderr, "[BREEZE_QNN_BACKBONE] conditional prefill failed\n");
        disable();
        return false;
    }
    prefill_ms += ms;

    if (impl_->branches == 2) {
        if (!prefill_app->run_prefill(
                *uncond_embeddings,
                uncond_tokens,
                hidden,
                uncond_out,
                uncond_cache,
                ms)) {
            std::fprintf(stderr, "[BREEZE_QNN_BACKBONE] unconditional prefill failed\n");
            disable();
            return false;
        }
        prefill_ms += ms;
    } else {
        uncond_cache = cond_cache;
        uncond_out = cond_out;
        uncond_tokens = cond_tokens;
    }

    // Release the large prompt context before creating the AR1 context so the
    // backbone weights are never duplicated in resident memory.
    prefill_app.reset();

    if (!load_backbone_app(
            impl_->lib_dir,
            impl_->step_path,
            "backbone_step_b2",
            impl_->step) ||
        !impl_->step->zero_step_cache() ||
        !impl_->step->seed_branch(cond_cache, 0) ||
        !impl_->step->seed_branch(uncond_cache, 1)) {
        std::fprintf(stderr, "[BREEZE_QNN_BACKBONE] AR1 context load/seed failed\n");
        disable();
        return false;
    }

    impl_->positions[0] = cond_tokens;
    impl_->positions[1] = uncond_tokens;

    std::fprintf(
        stderr,
        "[BREEZE_QNN_BACKBONE] prefill_ms=%.2f cond=%d uncond=%d "
        "prompt_context_released=1 ar1_ready=1\n",
        prefill_ms,
        cond_tokens,
        impl_->branches == 2 ? uncond_tokens : 0
    );
    return true;
}

bool QnnBackboneRunner::step(
    BreezeModel & m,
    const std::vector<float> & audio_embedding,
    StepOut & cond_out,
    StepOut & uncond_out
) {
    if (!ready() || !impl_->step) return false;
    if ((int) audio_embedding.size() != m.cfg.hidden_size) return false;
    for (int p : impl_->positions) {
        if (p < 0 || p >= impl_->max_seq) return false;
    }

    std::vector<StepOut> outs;
    double ms = 0.0;
    if (!impl_->step->run_step(
            m.cfg.hidden_size,
            audio_embedding,
            impl_->positions,
            outs,
            ms) ||
        outs.size() != 2) {
        std::fprintf(stderr, "[BREEZE_QNN_BACKBONE] AR1 step failed\n");
        disable();
        return false;
    }

    cond_out = std::move(outs[0]);
    if (impl_->branches == 2) {
        uncond_out = std::move(outs[1]);
    }

    impl_->frames++;
    impl_->total_ms += ms;
    if (impl_->frames == 1 || impl_->frames % 4 == 0) {
        std::fprintf(
            stderr,
            "[BREEZE_QNN_BACKBONE] frames=%zu step_ms=%.2f avg_ms=%.2f\n",
            impl_->frames,
            ms,
            impl_->total_ms / (double) impl_->frames
        );
    }
    return true;
}

} // namespace breeze
