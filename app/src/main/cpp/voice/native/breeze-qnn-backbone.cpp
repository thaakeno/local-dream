#include "breeze/qnn_backbone.h"

#include "breeze/model.h"

#include <HTP/QnnHtpDevice.h>
#include <QnnSampleApp.hpp>
#include <QnnTypeMacros.hpp>

#include "DynamicLoadUtil.hpp"

#include <algorithm>
#include <array>
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

struct GraphIo {
    uint32_t graph_index = 0;
    Qnn_Tensor_t * inputs = nullptr;
    Qnn_Tensor_t * outputs = nullptr;
    bool valid = false;
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
    std::array<GraphIo, 4> prefill{};
    GraphIo step_b1;
    GraphIo step_b2;

    BreezeQnnBackboneApp(
        QnnFunctionPointers qnnFunctionPointers,
        void * backendHandle,
        const std::string & cachedBinaryPath
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
        ) {}

    ~BreezeQnnBackboneApp() {
        for (auto & io : prefill) tear_down(io);
        tear_down(step_b1);
        tear_down(step_b2);
        if (m_graphsInfo) freeContext();
        freeDevice();
        terminateBackend();
        if (model_handle) {
            dlclose(model_handle);
            model_handle = nullptr;
        }
    }

    void tear_down(GraphIo & io) {
        if (!io.valid || (!io.inputs && !io.outputs) ||
            !m_graphsInfo || io.graph_index >= m_graphsCount) {
            io.inputs = nullptr;
            io.outputs = nullptr;
            io.valid = false;
            return;
        }
        auto & graph = (*m_graphsInfo)[io.graph_index];
        m_ioTensor.tearDownInputAndOutputTensors(
            io.inputs,
            io.outputs,
            graph.numInputTensors,
            graph.numOutputTensors
        );
        io.inputs = nullptr;
        io.outputs = nullptr;
        io.valid = false;
    }

    bool setup_one(GraphIo & io, uint32_t index) {
        io.graph_index = index;
        auto & graph = (*m_graphsInfo)[index];
        const auto rc = m_ioTensor.setupInputAndOutputTensors(
            &io.inputs, &io.outputs, graph
        );
        io.valid = rc == qnn::tools::iotensor::StatusCode::SUCCESS;
        return io.valid;
    }

    bool setup_graphs() {
        if (!m_graphsInfo || m_graphsCount < 6) {
            std::fprintf(
                stderr,
                "[BREEZE_QNN_BACKBONE] expected >=6 graphs, got %u\n",
                (unsigned) m_graphsCount
            );
            return false;
        }

        const int buckets[4] = {64, 128, 256, 512};
        std::array<int, 4> pidx = {-1, -1, -1, -1};
        int s1 = -1;
        int s2 = -1;
        for (uint32_t i = 0; i < m_graphsCount; ++i) {
            const char * name = (*m_graphsInfo)[i].graphName;
            std::fprintf(
                stderr,
                "[BREEZE_QNN_BACKBONE] graph[%u]=%s\n",
                i,
                name ? name : "<unnamed>"
            );
            for (int b = 0; b < 4; ++b) {
                const std::string needle =
                    "backbone_prefill_" + std::to_string(buckets[b]);
                if (name_contains(name, needle.c_str())) pidx[b] = (int) i;
            }
            if (name_contains(name, "backbone_step_b1")) s1 = (int) i;
            if (name_contains(name, "backbone_step_b2")) s2 = (int) i;
        }
        for (int i : pidx) if (i < 0) return false;
        if (s1 < 0 || s2 < 0) return false;

        for (int b = 0; b < 4; ++b) {
            if (!setup_one(prefill[b], (uint32_t) pidx[b])) return false;
        }
        if (!setup_one(step_b1, (uint32_t) s1)) return false;
        if (!setup_one(step_b2, (uint32_t) s2)) return false;

        for (auto & io : prefill) {
            auto & g = (*m_graphsInfo)[io.graph_index];
            if (g.numInputTensors != 3 || g.numOutputTensors != 4) {
                std::fprintf(
                    stderr,
                    "[BREEZE_QNN_BACKBONE] bad prefill IO %u/%u\n",
                    g.numInputTensors,
                    g.numOutputTensors
                );
                return false;
            }
        }
        for (GraphIo * io : {&step_b1, &step_b2}) {
            auto & g = (*m_graphsInfo)[io->graph_index];
            if (g.numInputTensors != 5 || g.numOutputTensors != 4) {
                std::fprintf(
                    stderr,
                    "[BREEZE_QNN_BACKBONE] bad step IO %u/%u\n",
                    g.numInputTensors,
                    g.numOutputTensors
                );
                return false;
            }
        }
        return true;
    }

    bool set_burst_power() {
        auto qnn = m_qnnFunctionPointers.qnnInterface;
        QnnDevice_Infrastructure_t deviceInfra = nullptr;
        if (!qnn.deviceGetInfrastructure ||
            qnn.deviceGetInfrastructure(&deviceInfra) != QNN_SUCCESS ||
            !deviceInfra) {
            return false;
        }
        auto * htp = static_cast<QnnHtpDevice_Infrastructure_t *>(deviceInfra);
        auto perf = htp->perfInfra;
        uint32_t id = 0;
        if (perf.createPowerConfigId(0, 0, &id) != QNN_SUCCESS) return false;

        QnnHtpPerfInfrastructure_PowerConfig_t rpc{};
        rpc.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_POLLING_TIME;
        rpc.rpcPollingTimeConfig = 9999;
        const QnnHtpPerfInfrastructure_PowerConfig_t * p1[] = {&rpc, nullptr};
        if (perf.setPowerConfig(id, p1) != QNN_SUCCESS) return false;

        QnnHtpPerfInfrastructure_PowerConfig_t dcvs{};
        dcvs.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3;
        dcvs.dcvsV3Config.contextId = id;
        dcvs.dcvsV3Config.powerMode =
            QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE;
        dcvs.dcvsV3Config.setDcvsEnable = 1;
        dcvs.dcvsV3Config.dcvsEnable = 0;
        dcvs.dcvsV3Config.setSleepDisable = 1;
        dcvs.dcvsV3Config.sleepDisable = 1;
        dcvs.dcvsV3Config.setBusParams = 1;
        dcvs.dcvsV3Config.busVoltageCornerMin =
            DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
        dcvs.dcvsV3Config.busVoltageCornerTarget =
            DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
        dcvs.dcvsV3Config.busVoltageCornerMax =
            DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
        dcvs.dcvsV3Config.setCoreParams = 1;
        dcvs.dcvsV3Config.coreVoltageCornerMin =
            DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
        dcvs.dcvsV3Config.coreVoltageCornerTarget =
            DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
        dcvs.dcvsV3Config.coreVoltageCornerMax =
            DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
        const QnnHtpPerfInfrastructure_PowerConfig_t * p2[] = {&dcvs, nullptr};
        return perf.setPowerConfig(id, p2) == QNN_SUCCESS;
    }

    bool execute_graph(GraphIo & io, double & ms) {
        auto & graph = (*m_graphsInfo)[io.graph_index];
        const auto t0 = std::chrono::steady_clock::now();
        const auto rc = m_qnnFunctionPointers.qnnInterface.graphExecute(
            graph.graph,
            io.inputs,
            graph.numInputTensors,
            io.outputs,
            graph.numOutputTensors,
            m_profileBackendHandle,
            nullptr
        );
        ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0
        ).count();
        return rc == QNN_GRAPH_NO_ERROR;
    }

    GraphIo * prefill_for(int tokens, int & bucket) {
        const int buckets[4] = {64, 128, 256, 512};
        for (int i = 0; i < 4; ++i) {
            if (tokens <= buckets[i]) {
                bucket = buckets[i];
                return &prefill[i];
            }
        }
        bucket = 0;
        return nullptr;
    }

    GraphIo & step_for(int branches) {
        return branches == 2 ? step_b2 : step_b1;
    }

    bool zero_step_cache(int branches) {
        GraphIo & io = step_for(branches);
        auto & g = (*m_graphsInfo)[io.graph_index];
        const int ki = tensor_index(io.inputs, g.numInputTensors, "key_cache");
        const int vi = tensor_index(io.inputs, g.numInputTensors, "value_cache");
        if (ki < 0 || vi < 0) return false;
        for (int index : {ki, vi}) {
            auto buf = QNN_TENSOR_GET_CLIENT_BUF(io.inputs[index]);
            if (!buf.data || buf.dataSize == 0) return false;
            std::memset(buf.data, 0, buf.dataSize);
        }
        return true;
    }

    bool seed_cache(
        GraphIo & source,
        int valid_tokens,
        int bucket,
        GraphIo & step,
        int branch
    ) {
        auto & pg = (*m_graphsInfo)[source.graph_index];
        auto & sg = (*m_graphsInfo)[step.graph_index];
        const int pki = tensor_index(source.outputs, pg.numOutputTensors, "key_cache");
        const int pvi = tensor_index(source.outputs, pg.numOutputTensors, "value_cache");
        const int ski = tensor_index(step.inputs, sg.numInputTensors, "key_cache");
        const int svi = tensor_index(step.inputs, sg.numInputTensors, "value_cache");
        if (pki < 0 || pvi < 0 || ski < 0 || svi < 0) return false;

        auto copy = [&](Qnn_Tensor_t & src, Qnn_Tensor_t & dst) {
            const uint32_t * sd = QNN_TENSOR_GET_DIMENSIONS(src);
            const uint32_t * dd = QNN_TENSOR_GET_DIMENSIONS(dst);
            if (!sd || !dd || QNN_TENSOR_GET_RANK(src) != 4 ||
                QNN_TENSOR_GET_RANK(dst) != 4) return false;
            const int lkv = (int) sd[1];
            const int src_seq = (int) sd[2];
            const int head_dim = (int) sd[3];
            const int branches = (int) dd[0];
            const int dst_lkv = (int) dd[1];
            const int dst_seq = (int) dd[2];
            const int dst_dim = (int) dd[3];
            if (
                src_seq != bucket ||
                lkv != dst_lkv ||
                head_dim != dst_dim ||
                branch < 0 ||
                branch >= branches ||
                valid_tokens <= 0 ||
                valid_tokens > bucket ||
                valid_tokens > dst_seq
            ) return false;
            if (QNN_TENSOR_GET_DATA_TYPE(src) != QNN_TENSOR_GET_DATA_TYPE(dst)) {
                return false;
            }
            auto sb = QNN_TENSOR_GET_CLIENT_BUF(src);
            auto db = QNN_TENSOR_GET_CLIENT_BUF(dst);
            if (!sb.data || !db.data) return false;
            const size_t se = tensor_elements(src);
            const size_t de = tensor_elements(dst);
            if (se == 0 || de == 0 || sb.dataSize % se || db.dataSize % de) {
                return false;
            }
            const size_t bytes = sb.dataSize / se;
            if (bytes != db.dataSize / de) return false;
            const int pad = bucket - valid_tokens;
            auto * sp = static_cast<const unsigned char *>(sb.data);
            auto * dp = static_cast<unsigned char *>(db.data);
            const size_t row = (size_t) head_dim * bytes;
            for (int h = 0; h < lkv; ++h) {
                const size_t src_offset =
                    ((size_t) h * bucket + (size_t) pad) * row;
                const size_t dst_offset =
                    (((size_t) branch * lkv + (size_t) h) * dst_seq) * row;
                std::memcpy(
                    dp + dst_offset,
                    sp + src_offset,
                    (size_t) valid_tokens * row
                );
            }
            return true;
        };

        return copy(source.outputs[pki], step.inputs[ski]) &&
            copy(source.outputs[pvi], step.inputs[svi]);
    }

    bool prefill_branch(
        const std::vector<float> & embeddings,
        int tokens,
        int hidden,
        GraphIo & step,
        int branch,
        StepOut & out,
        double & ms
    ) {
        int bucket = 0;
        GraphIo * io = prefill_for(tokens, bucket);
        if (!io || !io->valid) return false;
        auto & g = (*m_graphsInfo)[io->graph_index];

        const int ei = tensor_index(io->inputs, g.numInputTensors, "inputs_embeds", 0);
        const int mi = tensor_index(io->inputs, g.numInputTensors, "attention_mask", 1);
        const int pi = tensor_index(io->inputs, g.numInputTensors, "positions", 2);
        const int ho = tensor_index(io->outputs, g.numOutputTensors, "hidden", 0);
        const int lo = tensor_index(io->outputs, g.numOutputTensors, "logits", 1);
        if (ei < 0 || mi < 0 || pi < 0 || ho < 0 || lo < 0) return false;
        if ((int) embeddings.size() != tokens * hidden) return false;

        const int pad = bucket - tokens;
        std::vector<float> padded((size_t) bucket * hidden, 0.0f);
        std::memcpy(
            padded.data() + (size_t) pad * hidden,
            embeddings.data(),
            embeddings.size() * sizeof(float)
        );

        std::vector<float> mask((size_t) bucket * bucket, -10000.0f);
        for (int q = 0; q < bucket; ++q) {
            if (q < pad) {
                mask[(size_t) q * bucket + q] = 0.0f;
            } else {
                for (int k = pad; k <= q; ++k) {
                    mask[(size_t) q * bucket + k] = 0.0f;
                }
            }
        }
        std::vector<int32_t> positions((size_t) bucket, 0);
        for (int i = 0; i < tokens; ++i) positions[(size_t) pad + i] = i;

        if (m_ioTensor.copyFromFloatToNative(
                padded.data(), &io->inputs[ei]) !=
            qnn::tools::iotensor::StatusCode::SUCCESS) return false;
        if (m_ioTensor.copyFromFloatToNative(
                mask.data(), &io->inputs[mi]) !=
            qnn::tools::iotensor::StatusCode::SUCCESS) return false;
        if (!put_i32_array(io->inputs[pi], positions.data(), positions.size())) {
            return false;
        }

        if (!execute_graph(*io, ms)) return false;
        out.hidden.assign(tensor_elements(io->outputs[ho]), 0.0f);
        out.logits.assign(tensor_elements(io->outputs[lo]), 0.0f);
        if (m_ioTensor.convertToFloatInto(
                out.hidden.data(), &io->outputs[ho]) !=
            qnn::tools::iotensor::StatusCode::SUCCESS) return false;
        if (m_ioTensor.convertToFloatInto(
                out.logits.data(), &io->outputs[lo]) !=
            qnn::tools::iotensor::StatusCode::SUCCESS) return false;
        if (!finite_nonzero(out.hidden) || !finite_nonzero(out.logits)) return false;
        return seed_cache(*io, tokens, bucket, step, branch);
    }

    bool copy_new_cache(GraphIo & io, const std::vector<int32_t> & positions) {
        auto & g = (*m_graphsInfo)[io.graph_index];
        const int ki = tensor_index(io.inputs, g.numInputTensors, "key_cache");
        const int vi = tensor_index(io.inputs, g.numInputTensors, "value_cache");
        const int ko = tensor_index(io.outputs, g.numOutputTensors, "new_key");
        const int vo = tensor_index(io.outputs, g.numOutputTensors, "new_value");
        if (ki < 0 || vi < 0 || ko < 0 || vo < 0) return false;

        auto copy = [&](Qnn_Tensor_t & src, Qnn_Tensor_t & dst) {
            const uint32_t * sd = QNN_TENSOR_GET_DIMENSIONS(src);
            const uint32_t * dd = QNN_TENSOR_GET_DIMENSIONS(dst);
            if (!sd || !dd || QNN_TENSOR_GET_RANK(src) != 4 ||
                QNN_TENSOR_GET_RANK(dst) != 4) return false;
            const int batch = (int) sd[0];
            const int lkv = (int) sd[1];
            const int head_dim = (int) sd[3];
            const int dst_seq = (int) dd[2];
            if (
                (int) positions.size() != batch ||
                (int) dd[0] != batch ||
                (int) dd[1] != lkv ||
                (int) dd[3] != head_dim ||
                (int) sd[2] != 1
            ) return false;
            if (QNN_TENSOR_GET_DATA_TYPE(src) != QNN_TENSOR_GET_DATA_TYPE(dst)) {
                return false;
            }
            auto sb = QNN_TENSOR_GET_CLIENT_BUF(src);
            auto db = QNN_TENSOR_GET_CLIENT_BUF(dst);
            const size_t se = tensor_elements(src);
            const size_t de = tensor_elements(dst);
            if (!sb.data || !db.data || se == 0 || de == 0 ||
                sb.dataSize % se || db.dataSize % de) return false;
            const size_t bytes = sb.dataSize / se;
            if (bytes != db.dataSize / de) return false;
            const size_t row = (size_t) head_dim * bytes;
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

        return copy(io.outputs[ko], io.inputs[ki]) &&
            copy(io.outputs[vo], io.inputs[vi]);
    }

    bool step_once(
        int branches,
        int hidden,
        int max_seq,
        const std::vector<float> & embedding,
        std::vector<int32_t> & positions,
        std::vector<StepOut> & outs,
        double & ms
    ) {
        GraphIo & io = step_for(branches);
        auto & g = (*m_graphsInfo)[io.graph_index];
        const int ei = tensor_index(io.inputs, g.numInputTensors, "input_embed", 0);
        const int pi = tensor_index(io.inputs, g.numInputTensors, "positions", 1);
        const int mi = tensor_index(io.inputs, g.numInputTensors, "attention_mask", 4);
        const int ho = tensor_index(io.outputs, g.numOutputTensors, "hidden", 0);
        const int lo = tensor_index(io.outputs, g.numOutputTensors, "logits", 1);
        if (ei < 0 || pi < 0 || mi < 0 || ho < 0 || lo < 0) return false;
        if ((int) embedding.size() != hidden || (int) positions.size() != branches) {
            return false;
        }
        for (int p : positions) if (p < 0 || p >= max_seq) return false;

        std::vector<float> embeds((size_t) branches * hidden);
        for (int b = 0; b < branches; ++b) {
            std::copy(
                embedding.begin(),
                embedding.end(),
                embeds.begin() + (size_t) b * hidden
            );
        }
        std::vector<float> mask(
            (size_t) branches * (max_seq + 1),
            -10000.0f
        );
        for (int b = 0; b < branches; ++b) {
            const int p = positions[(size_t) b];
            float * row = mask.data() + (size_t) b * (max_seq + 1);
            for (int i = 0; i < p; ++i) row[i] = 0.0f;
            row[max_seq] = 0.0f; // current token is concatenated at the final slot
        }

        if (m_ioTensor.copyFromFloatToNative(
                embeds.data(), &io.inputs[ei]) !=
            qnn::tools::iotensor::StatusCode::SUCCESS) return false;
        if (!put_i32_array(io.inputs[pi], positions.data(), positions.size())) {
            return false;
        }
        if (m_ioTensor.copyFromFloatToNative(
                mask.data(), &io.inputs[mi]) !=
            qnn::tools::iotensor::StatusCode::SUCCESS) return false;

        if (!execute_graph(io, ms)) return false;

        std::vector<float> hidden_all(tensor_elements(io.outputs[ho]), 0.0f);
        std::vector<float> logits_all(tensor_elements(io.outputs[lo]), 0.0f);
        if (m_ioTensor.convertToFloatInto(
                hidden_all.data(), &io.outputs[ho]) !=
            qnn::tools::iotensor::StatusCode::SUCCESS) return false;
        if (m_ioTensor.convertToFloatInto(
                logits_all.data(), &io.outputs[lo]) !=
            qnn::tools::iotensor::StatusCode::SUCCESS) return false;
        if (!finite_nonzero(hidden_all) || !finite_nonzero(logits_all)) return false;

        const int logits_per_branch = (int) logits_all.size() / branches;
        outs.assign((size_t) branches, StepOut{});
        for (int b = 0; b < branches; ++b) {
            outs[(size_t) b].hidden.assign(
                hidden_all.begin() + (size_t) b * hidden,
                hidden_all.begin() + (size_t) (b + 1) * hidden
            );
            outs[(size_t) b].logits.assign(
                logits_all.begin() + (size_t) b * logits_per_branch,
                logits_all.begin() + (size_t) (b + 1) * logits_per_branch
            );
        }

        if (!copy_new_cache(io, positions)) return false;
        for (int & p : positions) ++p;
        return true;
    }
};

static bool load_backbone_app(
    const std::string & lib_dir,
    const std::string & context_path,
    std::unique_ptr<BreezeQnnBackboneApp> & app
) {
    QnnFunctionPointers systemFuncs;
    const std::string systemPath = lib_dir + "/libQnnSystem.so";
    const std::string backendPath = lib_dir + "/libQnnHtp.so";

    if (qnn::tools::dynamicloadutil::getQnnSystemFunctionPointers(
            systemPath, &systemFuncs) !=
        qnn::tools::dynamicloadutil::StatusCode::SUCCESS) {
        std::fprintf(
            stderr,
            "[BREEZE_QNN_BACKBONE] failed to load %s\n",
            systemPath.c_str()
        );
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
            &modelHandle) !=
        qnn::tools::dynamicloadutil::StatusCode::SUCCESS) {
        std::fprintf(stderr, "[BREEZE_QNN_BACKBONE] failed to load QNN HTP backend\n");
        return false;
    }
    funcs.qnnSystemInterface = systemFuncs.qnnSystemInterface;

    auto candidate = std::make_unique<BreezeQnnBackboneApp>(
        funcs,
        backendHandle,
        context_path
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
    if (!candidate->setup_graphs()) return fail();
    candidate->set_burst_power();

    app = std::move(candidate);
    return true;
}

} // namespace

struct QnnBackboneRunner::Impl {
    std::unique_ptr<BreezeQnnBackboneApp> app;
    bool enabled = false;
    int branches = 1;
    int max_seq = 512;
    std::vector<int32_t> positions;
    size_t frames = 0;
    double total_ms = 0.0;
};

QnnBackboneRunner::QnnBackboneRunner() : impl_(std::make_unique<Impl>()) {}
QnnBackboneRunner::~QnnBackboneRunner() = default;

bool QnnBackboneRunner::init(BreezeModel & m, int branches) {
    (void) m;
    if (branches != 1 && branches != 2) return false;
    const char * path = std::getenv("BREEZE_QNN_BACKBONE_PATH");
    const char * lib = std::getenv("BREEZE_QNN_LIB_DIR");
    if (!path || !*path || !lib || !*lib) return false;

    if (!load_backbone_app(lib, path, impl_->app)) {
        std::fprintf(
            stderr,
            "[BREEZE_QNN_BACKBONE] context load failed; using ggml backbone fallback\n"
        );
        impl_->app.reset();
        return false;
    }
    impl_->branches = branches;
    impl_->enabled = true;
    impl_->positions.assign((size_t) branches, 0);
    std::fprintf(
        stderr,
        "[BREEZE_QNN_BACKBONE] ready branches=%d prefill=64/128/256/512 "
        "step_cache=512 kv_update=new-row-only\n",
        branches
    );
    return true;
}

bool QnnBackboneRunner::ready() const {
    return impl_ && impl_->enabled && impl_->app;
}

int QnnBackboneRunner::max_seq() const {
    return impl_ ? impl_->max_seq : 0;
}

void QnnBackboneRunner::disable() {
    if (!impl_) return;
    impl_->enabled = false;
    impl_->app.reset();
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
    ) return false;
    if (
        impl_->branches == 2 &&
        (!uncond_embeddings ||
         uncond_tokens <= 0 ||
         uncond_tokens > impl_->max_seq ||
         (int) uncond_embeddings->size() != uncond_tokens * hidden)
    ) return false;

    auto & step = impl_->app->step_for(impl_->branches);
    if (!impl_->app->zero_step_cache(impl_->branches)) return false;

    double ms = 0.0;
    double total = 0.0;
    if (!impl_->app->prefill_branch(
            cond_embeddings,
            cond_tokens,
            hidden,
            step,
            0,
            cond_out,
            ms)) {
        std::fprintf(stderr, "[BREEZE_QNN_BACKBONE] conditional prefill failed\n");
        disable();
        return false;
    }
    total += ms;
    impl_->positions[0] = cond_tokens;

    if (impl_->branches == 2) {
        if (!impl_->app->prefill_branch(
                *uncond_embeddings,
                uncond_tokens,
                hidden,
                step,
                1,
                uncond_out,
                ms)) {
            std::fprintf(stderr, "[BREEZE_QNN_BACKBONE] unconditional prefill failed\n");
            disable();
            return false;
        }
        total += ms;
        impl_->positions[1] = uncond_tokens;
    }

    std::fprintf(
        stderr,
        "[BREEZE_QNN_BACKBONE] prefill_ms=%.2f cond=%d uncond=%d\n",
        total,
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
    if (!ready()) return false;
    if ((int) audio_embedding.size() != m.cfg.hidden_size) return false;
    for (int p : impl_->positions) {
        if (p >= impl_->max_seq) {
            std::fprintf(
                stderr,
                "[BREEZE_QNN_BACKBONE] cache capacity reached at %d\n",
                p
            );
            return false;
        }
    }

    std::vector<StepOut> outs;
    double ms = 0.0;
    if (!impl_->app->step_once(
            impl_->branches,
            m.cfg.hidden_size,
            impl_->max_seq,
            audio_embedding,
            impl_->positions,
            outs,
            ms)) {
        std::fprintf(stderr, "[BREEZE_QNN_BACKBONE] step failed\n");
        disable();
        return false;
    }
    if (outs.empty()) return false;
    cond_out = std::move(outs[0]);
    if (impl_->branches == 2) uncond_out = std::move(outs[1]);

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
