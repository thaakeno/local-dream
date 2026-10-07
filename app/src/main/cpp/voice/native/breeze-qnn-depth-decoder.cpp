#include "breeze/qnn_depth_decoder.h"

#include "breeze/model.h"
#include "breeze/sampling.h"

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

static bool restore_native(const NativeBlob & src, Qnn_Tensor_t & dst) {
    if (src.type != QNN_TENSOR_GET_DATA_TYPE(dst)) return false;
    const uint32_t rank = QNN_TENSOR_GET_RANK(dst);
    const uint32_t * dims = QNN_TENSOR_GET_DIMENSIONS(dst);
    if (!dims || src.dims.size() != rank) return false;
    for (uint32_t i = 0; i < rank; ++i) {
        if (src.dims[i] != dims[i]) return false;
    }
    auto buf = QNN_TENSOR_GET_CLIENT_BUF(dst);
    if (!buf.data || buf.dataSize != src.bytes.size()) return false;
    std::memcpy(buf.data, src.bytes.data(), src.bytes.size());
    return true;
}

static bool copy_native_tensor(const Qnn_Tensor_t & src, Qnn_Tensor_t & dst) {
    NativeBlob blob;
    return capture_native(src, blob) && restore_native(blob, dst);
}

static bool finite_logits(const std::vector<float> & logits) {
    if (logits.empty()) return false;
    bool nonzero = false;
    for (float x : logits) {
        if (!std::isfinite(x)) return false;
        nonzero = nonzero || std::fabs(x) > 1.0e-8f;
    }
    return nonzero;
}

static std::vector<float> cfg_logits(
    const std::vector<float> & logits,
    int logical_branches,
    float scale
) {
    if (logits.size() % 2 != 0) return {};
    const size_t vocab = logits.size() / 2;
    std::vector<float> out(vocab);
    if (logical_branches == 1) {
        std::copy(logits.begin(), logits.begin() + vocab, out.begin());
        return out;
    }
    for (size_t i = 0; i < vocab; ++i) {
        const float cond = logits[i];
        const float uncond = logits[vocab + i];
        out[i] = uncond + scale * (cond - uncond);
    }
    return out;
}

class BreezeQnnDepthApp final : public QnnSampleApp {
public:
    void * model_handle = nullptr;
    GraphIo graph;
    std::string expected_graph;

    BreezeQnnDepthApp(
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

    ~BreezeQnnDepthApp() {
        tear_down();
        if (m_graphsInfo) freeContext();
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
                "[BREEZE_QNN_DEPTH] context=%s graph[%u]=%s\n",
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

    bool prefill(
        const float * hidden,
        size_t hidden_count,
        int32_t cb0,
        std::vector<float> & logits,
        NativeBlob & key_cache,
        NativeBlob & value_cache,
        double & ms
    ) {
        auto & g = (*m_graphsInfo)[graph.graph_index];
        if (g.numInputTensors != 2 || g.numOutputTensors != 3) return false;
        const int h = tensor_index(graph.inputs, g.numInputTensors, "backbone_hidden", 0);
        const int t = tensor_index(graph.inputs, g.numInputTensors, "first_codebook", 1);
        const int l = tensor_index(graph.outputs, g.numOutputTensors, "logits", 0);
        const int k = tensor_index(graph.outputs, g.numOutputTensors, "key_cache", 1);
        const int v = tensor_index(graph.outputs, g.numOutputTensors, "value_cache", 2);
        if (h < 0 || t < 0 || l < 0 || k < 0 || v < 0) return false;
        if (tensor_elements(graph.inputs[h]) != hidden_count) return false;

        if (m_ioTensor.copyFromFloatToNative(
                const_cast<float *>(hidden),
                &graph.inputs[h]) != qnn::tools::iotensor::StatusCode::SUCCESS) {
            return false;
        }
        int32_t first[2] = {cb0, cb0};
        if (!put_i32_array(graph.inputs[t], first, 2)) return false;
        if (!execute(ms)) return false;

        logits.assign(tensor_elements(graph.outputs[l]), 0.0f);
        if (m_ioTensor.convertToFloatInto(
                logits.data(),
                &graph.outputs[l]) != qnn::tools::iotensor::StatusCode::SUCCESS) {
            return false;
        }
        return capture_native(graph.outputs[k], key_cache) &&
            capture_native(graph.outputs[v], value_cache);
    }

    bool seed_step(const NativeBlob & key_cache, const NativeBlob & value_cache) {
        auto & g = (*m_graphsInfo)[graph.graph_index];
        if (g.numInputTensors != 4 || g.numOutputTensors != 3) return false;
        const int k = tensor_index(graph.inputs, g.numInputTensors, "key_cache", 2);
        const int v = tensor_index(graph.inputs, g.numInputTensors, "value_cache", 3);
        return k >= 0 && v >= 0 &&
            restore_native(key_cache, graph.inputs[k]) &&
            restore_native(value_cache, graph.inputs[v]);
    }

    bool step(
        int32_t token,
        int32_t position,
        std::vector<float> & logits,
        double & ms
    ) {
        auto & g = (*m_graphsInfo)[graph.graph_index];
        if (g.numInputTensors != 4 || g.numOutputTensors != 3) return false;
        const int ti = tensor_index(graph.inputs, g.numInputTensors, "token", 0);
        const int pi = tensor_index(graph.inputs, g.numInputTensors, "position", 1);
        const int ki = tensor_index(graph.inputs, g.numInputTensors, "key_cache", 2);
        const int vi = tensor_index(graph.inputs, g.numInputTensors, "value_cache", 3);
        const int lo = tensor_index(graph.outputs, g.numOutputTensors, "logits", 0);
        const int ko = tensor_index(graph.outputs, g.numOutputTensors, "key_cache", 1);
        const int vo = tensor_index(graph.outputs, g.numOutputTensors, "value_cache", 2);
        if (ti < 0 || pi < 0 || ki < 0 || vi < 0 || lo < 0 || ko < 0 || vo < 0) {
            return false;
        }

        int32_t tokens[2] = {token, token};
        if (!put_i32_array(graph.inputs[ti], tokens, 2) ||
            !put_i32_array(graph.inputs[pi], &position, 1)) {
            return false;
        }
        if (!execute(ms)) return false;

        logits.assign(tensor_elements(graph.outputs[lo]), 0.0f);
        if (m_ioTensor.convertToFloatInto(
                logits.data(),
                &graph.outputs[lo]) != qnn::tools::iotensor::StatusCode::SUCCESS) {
            return false;
        }
        return copy_native_tensor(graph.outputs[ko], graph.inputs[ki]) &&
            copy_native_tensor(graph.outputs[vo], graph.inputs[vi]);
    }
};

static bool load_depth_app(
    const std::string & lib_dir,
    const std::string & context_path,
    const std::string & graph_name,
    std::unique_ptr<BreezeQnnDepthApp> & app
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

    auto candidate = std::make_unique<BreezeQnnDepthApp>(
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
    candidate->set_burst_power();

    app = std::move(candidate);
    return true;
}

} // namespace

struct QnnDepthRunner::Impl {
    std::unique_ptr<BreezeQnnDepthApp> prefill;
    std::unique_ptr<BreezeQnnDepthApp> step;
    bool enabled = false;
    int branches = 1;
    size_t frames = 0;
    double total_ms = 0.0;
};

QnnDepthRunner::QnnDepthRunner() : impl_(std::make_unique<Impl>()) {}
QnnDepthRunner::~QnnDepthRunner() = default;

bool QnnDepthRunner::init(BreezeModel & m, int branches) {
    if (branches != 1 && branches != 2) return false;
    const char * prefill_path = std::getenv("BREEZE_QNN_DEPTH_PREFILL_PATH");
    const char * step_path = std::getenv("BREEZE_QNN_DEPTH_STEP_PATH");
    const char * lib = std::getenv("BREEZE_QNN_LIB_DIR");
    if (!prefill_path || !*prefill_path || !step_path || !*step_path || !lib || !*lib) {
        return false;
    }

    if (!load_depth_app(
            lib,
            prefill_path,
            "depth_prefill_b2",
            impl_->prefill) ||
        !load_depth_app(
            lib,
            step_path,
            "depth_step_b2",
            impl_->step)) {
        std::fprintf(
            stderr,
            "[BREEZE_QNN_DEPTH] separate context load failed; using ggml fallback\n"
        );
        impl_->prefill.reset();
        impl_->step.reset();
        return false;
    }

    std::vector<float> hidden((size_t) 2 * (size_t) m.cfg.hidden_size, 0.0f);
    std::vector<float> logits;
    NativeBlob k;
    NativeBlob v;
    double ms_prefill = 0.0;
    double ms_step = 0.0;
    if (!impl_->prefill->prefill(
            hidden.data(),
            hidden.size(),
            0,
            logits,
            k,
            v,
            ms_prefill) ||
        !finite_logits(logits) ||
        !impl_->step->seed_step(k, v) ||
        !impl_->step->step(0, 2, logits, ms_step) ||
        !finite_logits(logits)) {
        std::fprintf(
            stderr,
            "[BREEZE_QNN_DEPTH] separate-context selftest failed; using ggml fallback\n"
        );
        impl_->prefill.reset();
        impl_->step.reset();
        return false;
    }

    impl_->branches = branches;
    impl_->enabled = true;
    std::fprintf(
        stderr,
        "[BREEZE_QNN_DEPTH] ready logical_branches=%d contexts=2 "
        "prefill_ms=%.2f step_ms=%.2f sampling=host-native\n",
        branches,
        ms_prefill,
        ms_step
    );
    return true;
}

bool QnnDepthRunner::ready() const {
    return impl_ && impl_->enabled && impl_->prefill && impl_->step;
}

void QnnDepthRunner::disable() {
    if (!impl_) return;
    impl_->enabled = false;
    impl_->prefill.reset();
    impl_->step.reset();
}

bool QnnDepthRunner::run(
    BreezeModel & m,
    const std::vector<std::vector<float>> & backbone_hiddens,
    int first_codebook,
    float cfg_scale,
    std::mt19937 & rng,
    std::vector<int> & residual_codebooks
) {
    if (!ready() || (int) backbone_hiddens.size() != impl_->branches) return false;
    for (const auto & hidden : backbone_hiddens) {
        if (hidden.size() != (size_t) m.cfg.hidden_size) return false;
    }

    std::vector<float> flat;
    flat.reserve((size_t) 2 * (size_t) m.cfg.hidden_size);
    flat.insert(flat.end(), backbone_hiddens[0].begin(), backbone_hiddens[0].end());
    if (impl_->branches == 2) {
        flat.insert(flat.end(), backbone_hiddens[1].begin(), backbone_hiddens[1].end());
    } else {
        flat.insert(flat.end(), backbone_hiddens[0].begin(), backbone_hiddens[0].end());
    }

    SampleParams sp;
    sp.temperature = m.cfg.depth_temperature;
    sp.top_k = m.cfg.depth_top_k;
    sp.top_p = m.cfg.depth_top_p;

    residual_codebooks.clear();
    residual_codebooks.reserve((size_t) m.cfg.num_codebooks - 1u);

    std::vector<float> logits_all;
    NativeBlob k;
    NativeBlob v;
    double graph_ms = 0.0;
    double frame_ms = 0.0;
    if (!impl_->prefill->prefill(
            flat.data(),
            flat.size(),
            (int32_t) first_codebook,
            logits_all,
            k,
            v,
            graph_ms) ||
        !finite_logits(logits_all) ||
        !impl_->step->seed_step(k, v)) {
        std::fprintf(stderr, "[BREEZE_QNN_DEPTH] prefill/seed failed; disabling\n");
        disable();
        return false;
    }
    frame_ms += graph_ms;

    std::vector<float> logits = cfg_logits(
        logits_all,
        impl_->branches,
        cfg_scale
    );
    if (!finite_logits(logits)) return false;
    int token = sample_token(logits, sp, rng);
    residual_codebooks.push_back(token);

    for (int position = 2; position < m.cfg.num_codebooks; ++position) {
        if (!impl_->step->step(
                (int32_t) token,
                (int32_t) position,
                logits_all,
                graph_ms) ||
            !finite_logits(logits_all)) {
            std::fprintf(
                stderr,
                "[BREEZE_QNN_DEPTH] AR1 step failed position=%d; disabling\n",
                position
            );
            disable();
            residual_codebooks.clear();
            return false;
        }
        frame_ms += graph_ms;
        logits = cfg_logits(logits_all, impl_->branches, cfg_scale);
        if (!finite_logits(logits)) {
            disable();
            residual_codebooks.clear();
            return false;
        }
        token = sample_token(logits, sp, rng);
        residual_codebooks.push_back(token);
    }

    impl_->frames++;
    impl_->total_ms += frame_ms;
    if (impl_->frames == 1 || impl_->frames % 4 == 0) {
        std::fprintf(
            stderr,
            "[BREEZE_QNN_DEPTH] frames=%zu frame_ms=%.2f avg_ms=%.2f "
            "graphs_per_frame=15 contexts=2\n",
            impl_->frames,
            frame_ms,
            impl_->total_ms / (double) impl_->frames
        );
    }
    return residual_codebooks.size() == (size_t) m.cfg.num_codebooks - 1u;
}

} // namespace breeze
