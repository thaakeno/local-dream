#include "breeze/qnn_vocoder.h"

#include <HTP/QnnHtpDevice.h>
#include <QnnSampleApp.hpp>
#include <QnnTypeMacros.hpp>

#include "DynamicLoadUtil.hpp"
#include "Logger.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

using namespace qnn::tools::sample_app;

namespace breeze {
namespace {

class BreezeQnnApp final : public QnnSampleApp {
public:
    Qnn_Tensor_t * inputs = nullptr;
    Qnn_Tensor_t * outputs = nullptr;
    void * model_handle = nullptr;

    BreezeQnnApp(
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
            qnn::tools::iotensor::InputDataType::NATIVE,
            ProfilingLevel::OFF,
            false,
            cachedBinaryPath,
            ""
        ) {}

    ~BreezeQnnApp() {
        if ((inputs || outputs) && m_graphsInfo && m_graphsCount > 0) {
            m_ioTensor.tearDownInputAndOutputTensors(
                inputs, outputs,
                (*m_graphsInfo)[0].numInputTensors,
                (*m_graphsInfo)[0].numOutputTensors
            );
        }
        inputs = nullptr;
        outputs = nullptr;
        if (m_graphsInfo) freeContext();
        freeDevice();
        terminateBackend();
        if (model_handle) {
            dlclose(model_handle);
            model_handle = nullptr;
        }
    }

    bool setup_io() {
        if (inputs && outputs) return true;
        if (!m_graphsInfo || m_graphsCount != 1) {
            std::fprintf(stderr, "[BREEZE_QNN] expected exactly one vocoder graph, got %u\n",
                         (unsigned) m_graphsCount);
            return false;
        }
        return qnn::tools::iotensor::StatusCode::SUCCESS ==
            m_ioTensor.setupInputAndOutputTensors(inputs ? nullptr : &inputs,
                                                  outputs ? nullptr : &outputs,
                                                  (*m_graphsInfo)[0]);
    }

    bool execute(const int32_t * codes, size_t code_count, float * audio, size_t sample_count) {
        if (!setup_io()) return false;
        auto & graph = (*m_graphsInfo)[0];
        if (graph.numInputTensors != 1 || graph.numOutputTensors != 1) {
            std::fprintf(stderr,
                         "[BREEZE_QNN] graph IO mismatch inputs=%u outputs=%u\n",
                         graph.numInputTensors, graph.numOutputTensors);
            return false;
        }

        auto & in = inputs[0];
        auto & out = outputs[0];
        const auto in_buf = QNN_TENSOR_GET_CLIENT_BUF(in);
        const auto out_buf = QNN_TENSOR_GET_CLIENT_BUF(out);
        const auto in_type = QNN_TENSOR_GET_DATA_TYPE(in);
        const auto out_type = QNN_TENSOR_GET_DATA_TYPE(out);

        if (in_type != QNN_DATATYPE_INT_32 || out_type != QNN_DATATYPE_FLOAT_32) {
            std::fprintf(stderr,
                         "[BREEZE_QNN] unsupported IO dtype input=%d output=%d\n",
                         (int) in_type, (int) out_type);
            return false;
        }
        if (in_buf.dataSize != code_count * sizeof(int32_t) ||
            out_buf.dataSize != sample_count * sizeof(float)) {
            std::fprintf(stderr,
                         "[BREEZE_QNN] IO byte mismatch in=%u/%zu out=%u/%zu\n",
                         in_buf.dataSize, code_count * sizeof(int32_t),
                         out_buf.dataSize, sample_count * sizeof(float));
            return false;
        }

        std::memcpy(in_buf.data, codes, code_count * sizeof(int32_t));
        const auto t0 = std::chrono::steady_clock::now();
        const auto st = m_qnnFunctionPointers.qnnInterface.graphExecute(
            graph.graph,
            inputs, graph.numInputTensors,
            outputs, graph.numOutputTensors,
            m_profileBackendHandle, nullptr
        );
        const double ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0).count();
        if (st != QNN_GRAPH_NO_ERROR) {
            std::fprintf(stderr, "[BREEZE_QNN] graphExecute failed err=%d\n", (int) st);
            return false;
        }
        std::memcpy(audio, out_buf.data, sample_count * sizeof(float));
        std::fprintf(stderr, "[BREEZE_QNN] graph64_ms=%.2f\n", ms);
        return true;
    }

    // Short speech flushes benefit from race-to-idle. QNN owns the HTP clock
    // vote for the native vocoder context and releases it with the context.
    bool set_burst_power() {
        auto qnn = m_qnnFunctionPointers.qnnInterface;
        QnnDevice_Infrastructure_t deviceInfra = nullptr;
        if (qnn.deviceGetInfrastructure(&deviceInfra) != QNN_SUCCESS || !deviceInfra) {
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
};

struct QnnHandles {
    QnnFunctionPointers funcs;
    void * backend = nullptr;
    void * model = nullptr;
};

static bool load_qnn(
    const std::string & lib_dir,
    const std::string & context_path,
    std::unique_ptr<BreezeQnnApp> & app
) {
    QnnFunctionPointers systemFuncs;
    const std::string systemPath = lib_dir + "/libQnnSystem.so";
    const std::string backendPath = lib_dir + "/libQnnHtp.so";

    if (qnn::tools::dynamicloadutil::getQnnSystemFunctionPointers(
            systemPath, &systemFuncs) !=
        qnn::tools::dynamicloadutil::StatusCode::SUCCESS) {
        std::fprintf(stderr, "[BREEZE_QNN] failed to load %s\n", systemPath.c_str());
        return false;
    }

    QnnFunctionPointers funcs;
    void * backendHandle = nullptr;
    void * modelHandle = nullptr;
    if (qnn::tools::dynamicloadutil::getQnnFunctionPointers(
            backendPath, context_path, &funcs, &backendHandle,
            false, &modelHandle) != qnn::tools::dynamicloadutil::StatusCode::SUCCESS) {
        std::fprintf(stderr, "[BREEZE_QNN] failed to load QNN HTP backend\n");
        return false;
    }
    funcs.qnnSystemInterface = systemFuncs.qnnSystemInterface;

    auto candidate = std::make_unique<BreezeQnnApp>(
        funcs, backendHandle, context_path);
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

    candidate->set_burst_power();
    app = std::move(candidate);
    return true;
}

} // namespace

struct BreezeQnnVocoder::Impl {
    std::unique_ptr<BreezeQnnApp> app;
    std::vector<int> history;
    int fixed_frames = 64;
    int left_context = 25;
    int n_codebooks = 16;
    int samples_per_frame = 1920;
};

BreezeQnnVocoder::BreezeQnnVocoder() : impl_(std::make_unique<Impl>()) {}
BreezeQnnVocoder::~BreezeQnnVocoder() = default;

bool BreezeQnnVocoder::init_from_environment() {
    const char * path = std::getenv("BREEZE_QNN_VOCODER_PATH");
    const char * libdir = std::getenv("BREEZE_QNN_LIB_DIR");
    if (!path || !*path || !libdir || !*libdir) return false;
    if (!std::filesystem::is_regular_file(path)) {
        std::fprintf(stderr, "[BREEZE_QNN] context missing: %s\n", path);
        return false;
    }
    if (!load_qnn(libdir, path, impl_->app)) {
        std::fprintf(stderr, "[BREEZE_QNN] accelerator init failed; using ggml vocoder\n");
        return false;
    }
    impl_->history.clear();
    std::fprintf(stderr,
                 "[BREEZE_QNN] ready path=%s graph_frames=%d left_context=%d backend=QNN-HTP\n",
                 path, impl_->fixed_frames, impl_->left_context);
    return true;
}

bool BreezeQnnVocoder::ready() const {
    return impl_ && impl_->app != nullptr;
}

void BreezeQnnVocoder::reset() {
    if (impl_) impl_->history.clear();
}

std::vector<float> BreezeQnnVocoder::decode_stream(
    const std::vector<int> & codes,
    int T,
    int n_cb,
    int spf
) {
    if (!ready() || T <= 0) return {};
    if (n_cb != impl_->n_codebooks || spf != impl_->samples_per_frame ||
        codes.size() != (size_t) T * (size_t) n_cb) {
        throw std::runtime_error("Breeze QNN vocoder shape mismatch");
    }

    const int history_frames = (int) impl_->history.size() / n_cb;
    const int ctx = std::min(history_frames, impl_->left_context);
    if (ctx + T > impl_->fixed_frames) {
        throw std::runtime_error("Breeze QNN vocoder chunk exceeds fixed graph");
    }

    std::vector<int32_t> input(
        (size_t) n_cb * (size_t) impl_->fixed_frames, 0);
    // QNN graph contract is [1, codebook, time].
    for (int cb = 0; cb < n_cb; ++cb) {
        for (int t = 0; t < ctx; ++t) {
            const int src_frame = history_frames - ctx + t;
            input[(size_t) cb * impl_->fixed_frames + t] =
                static_cast<int32_t>(impl_->history[(size_t) src_frame * n_cb + cb]);
        }
        for (int t = 0; t < T; ++t) {
            input[(size_t) cb * impl_->fixed_frames + ctx + t] =
                static_cast<int32_t>(codes[(size_t) t * n_cb + cb]);
        }
    }

    std::vector<float> full(
        (size_t) impl_->fixed_frames * (size_t) spf);
    if (!impl_->app->execute(
            input.data(), input.size(), full.data(), full.size())) {
        throw std::runtime_error("Breeze QNN vocoder execution failed");
    }

    const size_t begin = (size_t) ctx * spf;
    const size_t count = (size_t) T * spf;
    std::vector<float> out(full.begin() + begin, full.begin() + begin + count);
    for (float v : out) {
        if (!std::isfinite(v)) {
            throw std::runtime_error("Breeze QNN vocoder produced non-finite PCM");
        }
    }

    // Keep at most the official 25-frame causal left context.
    std::vector<int> merged;
    merged.reserve((size_t) (ctx + T) * n_cb);
    if (ctx > 0) {
        const auto first = impl_->history.end() - (size_t) ctx * n_cb;
        merged.insert(merged.end(), first, impl_->history.end());
    }
    merged.insert(merged.end(), codes.begin(), codes.end());
    const int merged_frames = (int) merged.size() / n_cb;
    const int keep = std::min(merged_frames, impl_->left_context);
    impl_->history.assign(
        merged.end() - (size_t) keep * n_cb, merged.end());

    return out;
}

} // namespace breeze
