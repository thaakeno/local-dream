#include "breeze/qnn_depth_decoder.h"

#include "breeze/model.h"
#include "breeze/sampling.h"

#include <HTP/QnnHtpDevice.h>
#include <QnnOwnedRuntime.hpp>
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

static bool put_i32(Qnn_Tensor_t & tensor, int32_t value) {
    auto buf = QNN_TENSOR_GET_CLIENT_BUF(tensor);
    if (!buf.data || buf.dataSize < sizeof(value)) return false;
    std::memcpy(buf.data, &value, sizeof(value));
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

static bool finite_logits(const std::vector<float> & logits) {
    if (logits.empty()) return false;
    bool nonzero = false;
    for (float x : logits) {
        if (!std::isfinite(x)) return false;
        nonzero = nonzero || std::fabs(x) > 1.0e-8f;
    }
    return nonzero;
}

static std::vector<float> combine_cfg(
    const std::vector<float> & cond,
    const std::vector<float> & uncond,
    int branches,
    float scale
) {
    if (branches == 1) return cond;
    if (cond.size() != uncond.size() || cond.empty()) return {};
    std::vector<float> out(cond.size());
    for (size_t i = 0; i < out.size(); ++i) {
        out[i] = uncond[i] + scale * (cond[i] - uncond[i]);
    }
    return out;
}

class BreezeQnnDepthApp final : public QnnOwnedRuntime {
public:
    void * model_handle = nullptr;
    uint32_t power_config_id = 0;
    bool power_config_active = false;
    GraphIo prefill;
    GraphIo step;

    BreezeQnnDepthApp(
        QnnFunctionPointers qnnFunctionPointers,
        void * backendHandle,
        const std::string & cachedBinaryPath
    ) : QnnOwnedRuntime(
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
        release_power_vote();
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

    bool setup_graphs() {
        if (!m_graphsInfo || m_graphsCount != 2) {
            std::fprintf(
                stderr,
                "[BREEZE_QNN_DEPTH] expected proven 2-graph batch1 context, got %u\n",
                (unsigned) m_graphsCount
            );
            return false;
        }

        int prefill_idx = -1;
        int step_idx = -1;
        for (uint32_t i = 0; i < m_graphsCount; ++i) {
            const char * name = (*m_graphsInfo)[i].graphName;
            std::fprintf(
                stderr,
                "[BREEZE_QNN_DEPTH] graph[%u]=%s\n",
                i,
                name ? name : "<unnamed>"
            );
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
            const auto rc = m_ioTensor.setupInputAndOutputTensors(
                &io.inputs,
                &io.outputs,
                graph
            );
            io.valid = rc == qnn::tools::iotensor::StatusCode::SUCCESS;
            return io.valid;
        };
        if (!setup(prefill) || !setup(step)) return false;

        auto & pg = (*m_graphsInfo)[prefill.graph_index];
        auto & sg = (*m_graphsInfo)[step.graph_index];
        if (pg.numInputTensors != 2 || pg.numOutputTensors != 3 ||
            sg.numInputTensors != 4 || sg.numOutputTensors != 3) {
            std::fprintf(
                stderr,
                "[BREEZE_QNN_DEPTH] bad graph IO prefill=%u/%u step=%u/%u\n",
                pg.numInputTensors,
                pg.numOutputTensors,
                sg.numInputTensors,
                sg.numOutputTensors
            );
            return false;
        }
        return true;
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

    bool execute(GraphIo & io, double & ms) {
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

    bool prefill_branch(
        const float * hidden,
        size_t hidden_count,
        int32_t cb0,
        std::vector<float> & logits,
        NativeBlob & key_cache,
        NativeBlob & value_cache,
        double & ms
    ) {
        auto & g = (*m_graphsInfo)[prefill.graph_index];
        const int h = tensor_index(prefill.inputs, g.numInputTensors, "backbone_hidden", 0);
        const int t = tensor_index(prefill.inputs, g.numInputTensors, "first_codebook", 1);
        const int l = tensor_index(prefill.outputs, g.numOutputTensors, "logits", 0);
        const int k = tensor_index(prefill.outputs, g.numOutputTensors, "key_cache", 1);
        const int v = tensor_index(prefill.outputs, g.numOutputTensors, "value_cache", 2);
        if (h < 0 || t < 0 || l < 0 || k < 0 || v < 0) return false;
        if (tensor_elements(prefill.inputs[h]) != hidden_count) return false;

        if (m_ioTensor.copyFromFloatToNative(
                const_cast<float *>(hidden),
                &prefill.inputs[h]) != qnn::tools::iotensor::StatusCode::SUCCESS) {
            return false;
        }
        if (!put_i32(prefill.inputs[t], cb0) || !execute(prefill, ms)) return false;

        logits.assign(tensor_elements(prefill.outputs[l]), 0.0f);
        if (m_ioTensor.convertToFloatInto(
                logits.data(),
                &prefill.outputs[l]) != qnn::tools::iotensor::StatusCode::SUCCESS) {
            return false;
        }
        return capture_native(prefill.outputs[k], key_cache) &&
            capture_native(prefill.outputs[v], value_cache);
    }

    bool step_branch(
        int32_t token,
        int32_t position,
        NativeBlob & key_cache,
        NativeBlob & value_cache,
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

        if (!restore_native(key_cache, step.inputs[ki]) ||
            !restore_native(value_cache, step.inputs[vi]) ||
            !put_i32(step.inputs[ti], token) ||
            !put_i32(step.inputs[pi], position) ||
            !execute(step, ms)) {
            return false;
        }

        logits.assign(tensor_elements(step.outputs[lo]), 0.0f);
        if (m_ioTensor.convertToFloatInto(
                logits.data(),
                &step.outputs[lo]) != qnn::tools::iotensor::StatusCode::SUCCESS) {
            return false;
        }

        return capture_native(step.outputs[ko], key_cache) &&
            capture_native(step.outputs[vo], value_cache);
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
    candidate->set_adaptive_power();

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
    const char * context = std::getenv("BREEZE_QNN_DEPTH_PATH");
    const char * lib = std::getenv("BREEZE_QNN_LIB_DIR");
    if (!context || !*context || !lib || !*lib) return false;

    if (!load_depth_app(lib, context, impl_->app)) {
        std::fprintf(
            stderr,
            "[BREEZE_QNN_DEPTH] proven batch1 context load failed; using ggml fallback\n"
        );
        impl_->app.reset();
        return false;
    }

    std::vector<float> hidden((size_t) m.cfg.hidden_size, 0.0f);
    std::vector<float> logits;
    NativeBlob key;
    NativeBlob value;
    double prefill_ms = 0.0;
    double step_ms = 0.0;
    if (!impl_->app->prefill_branch(
            hidden.data(),
            hidden.size(),
            0,
            logits,
            key,
            value,
            prefill_ms) ||
        !finite_logits(logits) ||
        !impl_->app->step_branch(
            0,
            2,
            key,
            value,
            logits,
            step_ms) ||
        !finite_logits(logits)) {
        std::fprintf(
            stderr,
            "[BREEZE_QNN_DEPTH] batch1 device selftest failed; using ggml fallback\n"
        );
        impl_->app.reset();
        return false;
    }

    impl_->branches = branches;
    impl_->enabled = true;
    std::fprintf(
        stderr,
        "[BREEZE_QNN_DEPTH] ready physical_batch=1 logical_branches=%d "
        "cfg_serial=%d context=proven-v1 prefill_ms=%.2f step_ms=%.2f\n",
        branches,
        branches == 2 ? 1 : 0,
        prefill_ms,
        step_ms
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

    SampleParams sp;
    sp.temperature = m.cfg.depth_temperature;
    sp.top_k = m.cfg.depth_top_k;
    sp.top_p = m.cfg.depth_top_p;

    residual_codebooks.clear();
    residual_codebooks.reserve((size_t) m.cfg.num_codebooks - 1u);

    std::vector<NativeBlob> keys((size_t) impl_->branches);
    std::vector<NativeBlob> values((size_t) impl_->branches);
    std::vector<std::vector<float>> branch_logits((size_t) impl_->branches);
    double frame_ms = 0.0;

    for (int b = 0; b < impl_->branches; ++b) {
        double ms = 0.0;
        const auto & hidden = backbone_hiddens[(size_t) b];
        if (!impl_->app->prefill_branch(
                hidden.data(),
                hidden.size(),
                (int32_t) first_codebook,
                branch_logits[(size_t) b],
                keys[(size_t) b],
                values[(size_t) b],
                ms) ||
            !finite_logits(branch_logits[(size_t) b])) {
            std::fprintf(
                stderr,
                "[BREEZE_QNN_DEPTH] serial prefill failed branch=%d; disabling\n",
                b
            );
            disable();
            return false;
        }
        frame_ms += ms;
    }

    std::vector<float> logits = combine_cfg(
        branch_logits[0],
        impl_->branches == 2 ? branch_logits[1] : branch_logits[0],
        impl_->branches,
        cfg_scale
    );
    if (!finite_logits(logits)) {
        disable();
        return false;
    }

    int token = sample_token(logits, sp, rng);
    residual_codebooks.push_back(token);

    for (int position = 2; position < m.cfg.num_codebooks; ++position) {
        for (int b = 0; b < impl_->branches; ++b) {
            double ms = 0.0;
            if (!impl_->app->step_branch(
                    (int32_t) token,
                    (int32_t) position,
                    keys[(size_t) b],
                    values[(size_t) b],
                    branch_logits[(size_t) b],
                    ms) ||
                !finite_logits(branch_logits[(size_t) b])) {
                std::fprintf(
                    stderr,
                    "[BREEZE_QNN_DEPTH] serial AR1 failed position=%d branch=%d; disabling\n",
                    position,
                    b
                );
                disable();
                residual_codebooks.clear();
                return false;
            }
            frame_ms += ms;
        }

        logits = combine_cfg(
            branch_logits[0],
            impl_->branches == 2 ? branch_logits[1] : branch_logits[0],
            impl_->branches,
            cfg_scale
        );
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
            "physical_batch=1 logical_branches=%d cfg_serial=%d\n",
            impl_->frames,
            frame_ms,
            impl_->total_ms / (double) impl_->frames,
            impl_->branches,
            impl_->branches == 2 ? 1 : 0
        );
    }
    return residual_codebooks.size() == (size_t) m.cfg.num_codebooks - 1u;
}

} // namespace breeze
