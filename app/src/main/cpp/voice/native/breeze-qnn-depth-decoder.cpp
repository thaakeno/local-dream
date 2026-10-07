#include "breeze/qnn_depth_decoder.h"

#include "breeze/breeze.h"
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
#include <stdexcept>
#include <string>
#include <vector>

using namespace qnn::tools::sample_app;

namespace breeze {
namespace {

struct GraphIo {
    uint32_t graph_index = 0;
    Qnn_Tensor_t * inputs = nullptr;
    Qnn_Tensor_t * outputs = nullptr;
};

static bool name_contains(const char * name, const char * needle) {
    return name && needle && std::string(name).find(needle) != std::string::npos;
}

static int tensor_index(Qnn_Tensor_t * tensors, uint32_t count, const char * needle, int fallback) {
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

static bool put_i32(Qnn_Tensor_t & tensor, int32_t value) {
    auto buf = QNN_TENSOR_GET_CLIENT_BUF(tensor);
    if (!buf.data || buf.dataSize < sizeof(value)) return false;
    std::memcpy(buf.data, &value, sizeof(value));
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

class BreezeQnnDepthApp final : public QnnSampleApp {
public:
    void * model_handle = nullptr;
    GraphIo prefill;
    GraphIo step;

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
        tear_down(prefill);
        tear_down(step);
        if (m_graphsInfo) freeContext();
        freeDevice();
        terminateBackend();
        if (model_handle) {
            dlclose(model_handle);
            model_handle = nullptr;
        }
    }

    void tear_down(GraphIo & io) {
        if ((!io.inputs && !io.outputs) || !m_graphsInfo || io.graph_index >= m_graphsCount) {
            io.inputs = nullptr;
            io.outputs = nullptr;
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
    }

    bool setup_graphs() {
        if (!m_graphsInfo || m_graphsCount != 2) {
            std::fprintf(
                stderr,
                "[BREEZE_QNN_DEPTH] expected 2 graphs (prefill,step), got %u\n",
                (unsigned) m_graphsCount
            );
            return false;
        }

        int prefill_idx = -1;
        int step_idx = -1;
        for (uint32_t i = 0; i < m_graphsCount; ++i) {
            const char * name = (*m_graphsInfo)[i].graphName;
            std::fprintf(stderr, "[BREEZE_QNN_DEPTH] graph[%u]=%s\n",
                         i, name ? name : "<unnamed>");
            if (name_contains(name, "prefill")) prefill_idx = (int) i;
            if (name_contains(name, "step")) step_idx = (int) i;
        }
        if (prefill_idx < 0) prefill_idx = 0;
        if (step_idx < 0) step_idx = prefill_idx == 0 ? 1 : 0;
        if (prefill_idx == step_idx) return false;

        prefill.graph_index = (uint32_t) prefill_idx;
        step.graph_index = (uint32_t) step_idx;

        auto setup = [&](GraphIo & io) {
            auto & graph = (*m_graphsInfo)[io.graph_index];
            return qnn::tools::iotensor::StatusCode::SUCCESS ==
                m_ioTensor.setupInputAndOutputTensors(
                    &io.inputs, &io.outputs, graph
                );
        };
        if (!setup(prefill) || !setup(step)) return false;

        auto & pg = (*m_graphsInfo)[prefill.graph_index];
        auto & sg = (*m_graphsInfo)[step.graph_index];
        if (pg.numInputTensors != 2 || pg.numOutputTensors != 3 ||
            sg.numInputTensors != 4 || sg.numOutputTensors != 3) {
            std::fprintf(
                stderr,
                "[BREEZE_QNN_DEPTH] bad graph IO prefill=%u/%u step=%u/%u\n",
                pg.numInputTensors, pg.numOutputTensors,
                sg.numInputTensors, sg.numOutputTensors
            );
            return false;
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
        dcvs.dcvsV3Config.busVoltageCornerMin = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
        dcvs.dcvsV3Config.busVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
        dcvs.dcvsV3Config.busVoltageCornerMax = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
        dcvs.dcvsV3Config.setCoreParams = 1;
        dcvs.dcvsV3Config.coreVoltageCornerMin = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
        dcvs.dcvsV3Config.coreVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
        dcvs.dcvsV3Config.coreVoltageCornerMax = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
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

    bool prefill_depth(
        const float * hidden,
        size_t hidden_count,
        int32_t cb0,
        std::vector<float> & logits,
        double & ms
    ) {
        auto & g = (*m_graphsInfo)[prefill.graph_index];
        const int h = tensor_index(prefill.inputs, g.numInputTensors, "backbone_hidden", 0);
        const int t = tensor_index(prefill.inputs, g.numInputTensors, "first_codebook", 1);
        const int l = tensor_index(prefill.outputs, g.numOutputTensors, "logits", 0);
        if (h < 0 || t < 0 || l < 0) return false;
        if (tensor_elements(prefill.inputs[h]) != hidden_count) return false;
        if (m_ioTensor.copyFromFloatToNative(
                const_cast<float *>(hidden), &prefill.inputs[h]) !=
            qnn::tools::iotensor::StatusCode::SUCCESS) return false;
        if (!put_i32(prefill.inputs[t], cb0)) return false;
        if (!execute_graph(prefill, ms)) return false;

        logits.assign(tensor_elements(prefill.outputs[l]), 0.0f);
        if (m_ioTensor.convertToFloatInto(logits.data(), &prefill.outputs[l]) !=
            qnn::tools::iotensor::StatusCode::SUCCESS) return false;

        const int pk = tensor_index(prefill.outputs, g.numOutputTensors, "key_cache", 1);
        const int pv = tensor_index(prefill.outputs, g.numOutputTensors, "value_cache", 2);
        auto & sg = (*m_graphsInfo)[step.graph_index];
        const int sk = tensor_index(step.inputs, sg.numInputTensors, "key_cache", 2);
        const int sv = tensor_index(step.inputs, sg.numInputTensors, "value_cache", 3);
        return pk >= 0 && pv >= 0 && sk >= 0 && sv >= 0 &&
            copy_native_tensor(prefill.outputs[pk], step.inputs[sk]) &&
            copy_native_tensor(prefill.outputs[pv], step.inputs[sv]);
    }

    bool step_depth(
        int32_t token,
        int32_t position,
        std::vector<float> & logits,
        double & ms
    ) {
        auto & g = (*m_graphsInfo)[step.graph_index];
        const int ti = tensor_index(step.inputs, g.numInputTensors, "token", 0);
        const int pi = tensor_index(step.inputs, g.numInputTensors, "position", 1);
        const int ki = tensor_index(step.inputs, g.numInputTensors, "key_cache", 2);
        const int vi = tensor_index(step.inputs, g.numInputTensors, "value_cache", 3);
        const int lo = tensor_index(step.outputs, g.numOutputTensors, "logits", 0);
        const int ko = tensor_index(step.outputs, g.numOutputTensors, "key_cache", 1);
        const int vo = tensor_index(step.outputs, g.numOutputTensors, "value_cache", 2);
        if (ti < 0 || pi < 0 || ki < 0 || vi < 0 || lo < 0 || ko < 0 || vo < 0) {
            return false;
        }
        if (!put_i32(step.inputs[ti], token) || !put_i32(step.inputs[pi], position)) {
            return false;
        }
        if (!execute_graph(step, ms)) return false;

        logits.assign(tensor_elements(step.outputs[lo]), 0.0f);
        if (m_ioTensor.convertToFloatInto(logits.data(), &step.outputs[lo]) !=
            qnn::tools::iotensor::StatusCode::SUCCESS) return false;

        // Carry the native FP16/quantized cache forward without a host-float roundtrip.
        return copy_native_tensor(step.outputs[ko], step.inputs[ki]) &&
            copy_native_tensor(step.outputs[vo], step.inputs[vi]);
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
            backendPath, context_path, &funcs, &backendHandle,
            false, &modelHandle) != qnn::tools::dynamicloadutil::StatusCode::SUCCESS) {
        std::fprintf(stderr, "[BREEZE_QNN_DEPTH] failed to load QNN HTP backend\n");
        return false;
    }
    funcs.qnnSystemInterface = systemFuncs.qnnSystemInterface;

    auto candidate = std::make_unique<BreezeQnnDepthApp>(
        funcs, backendHandle, context_path
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

static bool finite_logits(const std::vector<float> & logits) {
    if (logits.empty()) return false;
    bool nonzero = false;
    for (float x : logits) {
        if (!std::isfinite(x)) return false;
        nonzero = nonzero || std::fabs(x) > 1.0e-8f;
    }
    return nonzero;
}

} // namespace

struct QnnDepthRunner::Impl {
    std::unique_ptr<BreezeQnnDepthApp> app;
    bool enabled = false;
    size_t frames = 0;
    double total_ms = 0.0;
};

QnnDepthRunner::QnnDepthRunner() : impl_(std::make_unique<Impl>()) {}
QnnDepthRunner::~QnnDepthRunner() = default;

bool QnnDepthRunner::init(BreezeModel & m) {
    (void) m;
    const char * path = std::getenv("BREEZE_QNN_GENERATOR_PATH");
    const char * lib = std::getenv("BREEZE_QNN_LIB_DIR");
    if (!path || !*path || !lib || !*lib) return false;

    if (!load_depth_app(lib, path, impl_->app)) {
        std::fprintf(stderr, "[BREEZE_QNN_DEPTH] context load failed; using ggml depth fallback\n");
        impl_->app.reset();
        return false;
    }

    // Cheap real-device smoke test. It catches incompatible context binaries
    // before synthesis without requiring a separate reference file.
    std::vector<float> hidden(2048, 0.0f);
    std::vector<float> logits;
    double ms = 0.0;
    if (!impl_->app->prefill_depth(hidden.data(), hidden.size(), 0, logits, ms) ||
        !finite_logits(logits)) {
        std::fprintf(stderr, "[BREEZE_QNN_DEPTH] device selftest failed; using ggml fallback\n");
        impl_->app.reset();
        return false;
    }

    impl_->enabled = true;
    std::fprintf(
        stderr,
        "[BREEZE_QNN_DEPTH] ready graphs=depth_prefill,depth_step selftest_ms=%.2f "
        "sampling=host-native cache=native-persistent\n",
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
    const std::vector<float> & backbone_hidden,
    int first_codebook,
    std::mt19937 & rng,
    std::vector<int> & residual_codebooks
) {
    if (!ready() || backbone_hidden.size() != (size_t) m.cfg.hidden_size) return false;

    SampleParams sp;
    sp.temperature = m.cfg.depth_temperature;
    sp.top_k = m.cfg.depth_top_k;
    sp.top_p = m.cfg.depth_top_p;

    residual_codebooks.clear();
    residual_codebooks.reserve((size_t) m.cfg.num_codebooks - 1u);

    std::vector<float> logits;
    double graph_ms = 0.0;
    double frame_ms = 0.0;
    if (!impl_->app->prefill_depth(
            backbone_hidden.data(), backbone_hidden.size(),
            (int32_t) first_codebook, logits, graph_ms) ||
        !finite_logits(logits)) {
        std::fprintf(stderr, "[BREEZE_QNN_DEPTH] prefill failed; disabling accelerator\n");
        disable();
        return false;
    }
    frame_ms += graph_ms;
    int token = sample_token(logits, sp, rng);
    residual_codebooks.push_back(token);

    // Positions 2..15 consume codebooks 1..14 and predict codebooks 2..15.
    for (int position = 2; position < m.cfg.num_codebooks; ++position) {
        if (!impl_->app->step_depth(
                (int32_t) token, (int32_t) position, logits, graph_ms) ||
            !finite_logits(logits)) {
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
        token = sample_token(logits, sp, rng);
        residual_codebooks.push_back(token);
    }

    impl_->frames++;
    impl_->total_ms += frame_ms;
    if (impl_->frames == 1 || impl_->frames % 4 == 0) {
        std::fprintf(
            stderr,
            "[BREEZE_QNN_DEPTH] frames=%zu frame_ms=%.2f avg_ms=%.2f graphs_per_frame=15\n",
            impl_->frames,
            frame_ms,
            impl_->total_ms / (double) impl_->frames
        );
    }
    return residual_codebooks.size() == (size_t) m.cfg.num_codebooks - 1u;
}

} // namespace breeze
