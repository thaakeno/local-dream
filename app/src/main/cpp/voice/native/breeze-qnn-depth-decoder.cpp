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

static bool copy_native_tensor(const Qnn_Tensor_t & src, Qnn_Tensor_t & dst) {
    const auto sb = QNN_TENSOR_GET_CLIENT_BUF(src);
    const auto db = QNN_TENSOR_GET_CLIENT_BUF(dst);
    if (!sb.data || !db.data || sb.dataSize != db.dataSize) return false;
    if (QNN_TENSOR_GET_DATA_TYPE(src) != QNN_TENSOR_GET_DATA_TYPE(dst)) return false;
    std::memcpy(db.data, sb.data, sb.dataSize);
    return true;
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
    int branches,
    float scale
) {
    if (branches == 1) return logits;
    if (branches != 2 || logits.size() % 2 != 0) return {};
    const size_t vocab = logits.size() / 2;
    std::vector<float> out(vocab);
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
    GraphIo prefill_b1;
    GraphIo prefill_b2;
    GraphIo step_b1;
    GraphIo step_b2;

    BreezeQnnDepthApp(
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

    ~BreezeQnnDepthApp() {
        tear_down(prefill_b1);
        tear_down(prefill_b2);
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
        if (!m_graphsInfo || m_graphsCount < 4) {
            std::fprintf(
                stderr,
                "[BREEZE_QNN_DEPTH] expected 4 v2 graphs, got %u\n",
                (unsigned) m_graphsCount
            );
            return false;
        }
        int p1 = -1, p2 = -1, s1 = -1, s2 = -1;
        for (uint32_t i = 0; i < m_graphsCount; ++i) {
            const char * name = (*m_graphsInfo)[i].graphName;
            std::fprintf(
                stderr,
                "[BREEZE_QNN_DEPTH] graph[%u]=%s\n",
                i,
                name ? name : "<unnamed>"
            );
            if (name_contains(name, "depth_prefill_b1")) p1 = (int) i;
            if (name_contains(name, "depth_prefill_b2")) p2 = (int) i;
            if (name_contains(name, "depth_step_b1")) s1 = (int) i;
            if (name_contains(name, "depth_step_b2")) s2 = (int) i;
        }
        if (p1 < 0 || p2 < 0 || s1 < 0 || s2 < 0) return false;
        if (!setup_one(prefill_b1, (uint32_t) p1)) return false;
        if (!setup_one(prefill_b2, (uint32_t) p2)) return false;
        if (!setup_one(step_b1, (uint32_t) s1)) return false;
        if (!setup_one(step_b2, (uint32_t) s2)) return false;

        for (GraphIo * io : {&prefill_b1, &prefill_b2}) {
            auto & g = (*m_graphsInfo)[io->graph_index];
            if (g.numInputTensors != 2 || g.numOutputTensors != 3) return false;
        }
        for (GraphIo * io : {&step_b1, &step_b2}) {
            auto & g = (*m_graphsInfo)[io->graph_index];
            if (g.numInputTensors != 4 || g.numOutputTensors != 3) return false;
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

    GraphIo & prefill_for(int branches) {
        return branches == 2 ? prefill_b2 : prefill_b1;
    }

    GraphIo & step_for(int branches) {
        return branches == 2 ? step_b2 : step_b1;
    }

    bool prefill_depth(
        int branches,
        const float * hidden,
        size_t hidden_count,
        int32_t cb0,
        std::vector<float> & logits,
        double & ms
    ) {
        GraphIo & io = prefill_for(branches);
        GraphIo & step = step_for(branches);
        auto & g = (*m_graphsInfo)[io.graph_index];
        const int h = tensor_index(io.inputs, g.numInputTensors, "backbone_hidden", 0);
        const int t = tensor_index(io.inputs, g.numInputTensors, "first_codebook", 1);
        const int l = tensor_index(io.outputs, g.numOutputTensors, "logits", 0);
        if (h < 0 || t < 0 || l < 0) return false;
        if (tensor_elements(io.inputs[h]) != hidden_count) return false;

        if (m_ioTensor.copyFromFloatToNative(
                const_cast<float *>(hidden), &io.inputs[h]) !=
            qnn::tools::iotensor::StatusCode::SUCCESS) return false;
        std::vector<int32_t> first((size_t) branches, cb0);
        if (!put_i32_array(io.inputs[t], first.data(), first.size())) return false;
        if (!execute_graph(io, ms)) return false;

        logits.assign(tensor_elements(io.outputs[l]), 0.0f);
        if (m_ioTensor.convertToFloatInto(logits.data(), &io.outputs[l]) !=
            qnn::tools::iotensor::StatusCode::SUCCESS) return false;

        const int pk = tensor_index(io.outputs, g.numOutputTensors, "key_cache", 1);
        const int pv = tensor_index(io.outputs, g.numOutputTensors, "value_cache", 2);
        auto & sg = (*m_graphsInfo)[step.graph_index];
        const int sk = tensor_index(step.inputs, sg.numInputTensors, "key_cache", 2);
        const int sv = tensor_index(step.inputs, sg.numInputTensors, "value_cache", 3);
        return pk >= 0 && pv >= 0 && sk >= 0 && sv >= 0 &&
            copy_native_tensor(io.outputs[pk], step.inputs[sk]) &&
            copy_native_tensor(io.outputs[pv], step.inputs[sv]);
    }

    bool step_depth(
        int branches,
        int32_t token,
        int32_t position,
        std::vector<float> & logits,
        double & ms
    ) {
        GraphIo & io = step_for(branches);
        auto & g = (*m_graphsInfo)[io.graph_index];
        const int ti = tensor_index(io.inputs, g.numInputTensors, "token", 0);
        const int pi = tensor_index(io.inputs, g.numInputTensors, "position", 1);
        const int ki = tensor_index(io.inputs, g.numInputTensors, "key_cache", 2);
        const int vi = tensor_index(io.inputs, g.numInputTensors, "value_cache", 3);
        const int lo = tensor_index(io.outputs, g.numOutputTensors, "logits", 0);
        const int ko = tensor_index(io.outputs, g.numOutputTensors, "key_cache", 1);
        const int vo = tensor_index(io.outputs, g.numOutputTensors, "value_cache", 2);
        if (ti < 0 || pi < 0 || ki < 0 || vi < 0 || lo < 0 || ko < 0 || vo < 0) {
            return false;
        }

        std::vector<int32_t> tokens((size_t) branches, token);
        if (!put_i32_array(io.inputs[ti], tokens.data(), tokens.size()) ||
            !put_i32_array(io.inputs[pi], &position, 1)) {
            return false;
        }
        if (!execute_graph(io, ms)) return false;

        logits.assign(tensor_elements(io.outputs[lo]), 0.0f);
        if (m_ioTensor.convertToFloatInto(logits.data(), &io.outputs[lo]) !=
            qnn::tools::iotensor::StatusCode::SUCCESS) return false;

        return copy_native_tensor(io.outputs[ko], io.inputs[ki]) &&
            copy_native_tensor(io.outputs[vo], io.inputs[vi]);
    }
};

static bool load_depth_app(
    const std::string & lib_dir,
    const std::string & context_path,
    std::unique_ptr<BreezeQnnDepthApp> & app
) {
    QnnFunctionPointers systemFuncs;
    const std::string systemPath = lib_dir + "/libQnnSystem.so";
    const std::string backendPath = lib_dir + "/libQnnHtp.so";

    if (qnn::tools::dynamicloadutil::getQnnSystemFunctionPointers(
            systemPath, &systemFuncs) !=
        qnn::tools::dynamicloadutil::StatusCode::SUCCESS) {
        std::fprintf(stderr, "[BREEZE_QNN_DEPTH] failed to load %s\n", systemPath.c_str());
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
        std::fprintf(stderr, "[BREEZE_QNN_DEPTH] failed to load QNN HTP backend\n");
        return false;
    }
    funcs.qnnSystemInterface = systemFuncs.qnnSystemInterface;

    auto candidate = std::make_unique<BreezeQnnDepthApp>(
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

struct QnnDepthRunner::Impl {
    std::unique_ptr<BreezeQnnDepthApp> app;
    bool enabled = false;
    int branches = 1;
    size_t frames = 0;
    double total_ms = 0.0;
};

QnnDepthRunner::QnnDepthRunner() : impl_(std::make_unique<Impl>()) {}
QnnDepthRunner::~QnnDepthRunner() = default;

bool QnnDepthRunner::init(BreezeModel & m, int branches) {
    if (branches != 1 && branches != 2) return false;
    const char * path = std::getenv("BREEZE_QNN_DEPTH_PATH");
    if (!path || !*path) path = std::getenv("BREEZE_QNN_GENERATOR_PATH");
    const char * lib = std::getenv("BREEZE_QNN_LIB_DIR");
    if (!path || !*path || !lib || !*lib) return false;

    if (!load_depth_app(lib, path, impl_->app)) {
        std::fprintf(
            stderr,
            "[BREEZE_QNN_DEPTH] context load failed; using ggml depth fallback\n"
        );
        impl_->app.reset();
        return false;
    }

    // Cheap batch-1 real-device smoke test catches incompatible context binaries.
    std::vector<float> hidden((size_t) m.cfg.hidden_size, 0.0f);
    std::vector<float> logits;
    double ms = 0.0;
    if (!impl_->app->prefill_depth(
            1,
            hidden.data(),
            hidden.size(),
            0,
            logits,
            ms) ||
        !finite_logits(logits)) {
        std::fprintf(stderr, "[BREEZE_QNN_DEPTH] device selftest failed; using ggml fallback\n");
        impl_->app.reset();
        return false;
    }

    impl_->branches = branches;
    impl_->enabled = true;
    std::fprintf(
        stderr,
        "[BREEZE_QNN_DEPTH] ready branches=%d graphs=b1+b2 selftest_ms=%.2f "
        "sampling=host-native cache=native-persistent\n",
        branches,
        ms
    );
    return true;
}

bool QnnDepthRunner::ready() const {
    return impl_ && impl_->enabled && impl_->app;
}

void QnnDepthRunner::disable() {
    if (!impl_) return;
    impl_->enabled = false;
    impl_->app.reset();
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
    flat.reserve((size_t) impl_->branches * m.cfg.hidden_size);
    for (const auto & hidden : backbone_hiddens) {
        flat.insert(flat.end(), hidden.begin(), hidden.end());
    }

    SampleParams sp;
    sp.temperature = m.cfg.depth_temperature;
    sp.top_k = m.cfg.depth_top_k;
    sp.top_p = m.cfg.depth_top_p;

    residual_codebooks.clear();
    residual_codebooks.reserve((size_t) m.cfg.num_codebooks - 1u);

    std::vector<float> logits_all;
    double graph_ms = 0.0;
    double frame_ms = 0.0;
    if (!impl_->app->prefill_depth(
            impl_->branches,
            flat.data(),
            flat.size(),
            (int32_t) first_codebook,
            logits_all,
            graph_ms) ||
        !finite_logits(logits_all)) {
        std::fprintf(stderr, "[BREEZE_QNN_DEPTH] prefill failed; disabling accelerator\n");
        disable();
        return false;
    }
    frame_ms += graph_ms;
    std::vector<float> logits = cfg_logits(logits_all, impl_->branches, cfg_scale);
    if (!finite_logits(logits)) return false;
    int token = sample_token(logits, sp, rng);
    residual_codebooks.push_back(token);

    for (int position = 2; position < m.cfg.num_codebooks; ++position) {
        if (!impl_->app->step_depth(
                impl_->branches,
                (int32_t) token,
                (int32_t) position,
                logits_all,
                graph_ms) ||
            !finite_logits(logits_all)) {
            std::fprintf(
                stderr,
                "[BREEZE_QNN_DEPTH] step failed position=%d; disabling accelerator\n",
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
            "graphs_per_frame=15 branches=%d\n",
            impl_->frames,
            frame_ms,
            impl_->total_ms / (double) impl_->frames,
            impl_->branches
        );
    }
    return residual_codebooks.size() == (size_t) m.cfg.num_codebooks - 1u;
}

} // namespace breeze
