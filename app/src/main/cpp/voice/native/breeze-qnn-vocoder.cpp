#include "breeze/qnn_vocoder.h"

#include <HTP/QnnHtpDevice.h>
#include <QnnOwnedRuntime.hpp>
#include <QnnTypeMacros.hpp>

#include "DynamicLoadUtil.hpp"
#include "Logger.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <limits>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace qnn::tools::sample_app;

namespace breeze {
namespace {

class BreezeQnnApp final : public QnnOwnedRuntime {
public:
    struct GraphIo {
        Qnn_Tensor_t * inputs = nullptr;
        Qnn_Tensor_t * outputs = nullptr;
    };

    std::vector<GraphIo> graph_io;
    void * model_handle = nullptr;
    uint32_t power_config_id = 0;
    bool power_config_active = false;

    BreezeQnnApp(
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

    ~BreezeQnnApp() {
        if (m_graphsInfo) {
            const size_t count = std::min<size_t>(graph_io.size(), m_graphsCount);
            for (size_t i = 0; i < count; ++i) {
                auto & slot = graph_io[i];
                if (!slot.inputs && !slot.outputs) continue;
                m_ioTensor.tearDownInputAndOutputTensors(
                    slot.inputs,
                    slot.outputs,
                    (*m_graphsInfo)[i].numInputTensors,
                    (*m_graphsInfo)[i].numOutputTensors
                );
                slot.inputs = nullptr;
                slot.outputs = nullptr;
            }
            freeContext();
        }
        release_power_vote();
        freeDevice();
        terminateBackend();
        if (model_handle) {
            dlclose(model_handle);
            model_handle = nullptr;
        }
    }

    // The pinned QAIRT metadata can describe an exported vocoder graph as a
    // flattened tensor or as NFC/NCF. Shape *element counts*, not an assumed
    // rank of exactly 3, determine whether an artifact contains an 8/32/64
    // frame graph. The trusted PCM self-test still has to pass before any
    // QNN output reaches playback.
    static size_t tensor_elements(const Qnn_Tensor_t & tensor) {
        const uint32_t rank = QNN_TENSOR_GET_RANK(tensor);
        const uint32_t * dims = QNN_TENSOR_GET_DIMENSIONS(tensor);
        if (!rank || rank > 8 || !dims) return 0;
        size_t count = 1;
        for (uint32_t i = 0; i < rank; ++i) {
            if (!dims[i] || count > SIZE_MAX / dims[i]) return 0;
            count *= dims[i];
        }
        return count;
    }

    static std::string describe_shape(const Qnn_Tensor_t & tensor) {
        const uint32_t rank = QNN_TENSOR_GET_RANK(tensor);
        const uint32_t * dims = QNN_TENSOR_GET_DIMENSIONS(tensor);
        std::string desc;
        for (uint32_t i = 0; i < rank && i < 8; ++i) {
            if (i) desc += "x";
            desc += dims ? std::to_string(dims[i]) : "?";
        }
        return desc.empty() ? "<scalar>" : desc;
    }

    bool setup_io(size_t index) {
        if (!m_graphsInfo || index >= m_graphsCount) return false;
        const auto & graph = (*m_graphsInfo)[index];
        if (graph.numInputTensors != 1 || graph.numOutputTensors != 1) {
            std::fprintf(stderr,
                "[BREEZE_QNN_GRAPH] idx=%zu name=%s inputs=%u outputs=%u incompatible\n",
                index, graph.graphName ? graph.graphName : "<unnamed>",
                (unsigned)graph.numInputTensors, (unsigned)graph.numOutputTensors);
            return false;
        }
        if (graph_io.size() < m_graphsCount) graph_io.resize(m_graphsCount);
        auto & slot = graph_io[index];
        if (slot.inputs && slot.outputs) return true;
        const auto status = m_ioTensor.setupInputAndOutputTensors(
            slot.inputs ? nullptr : &slot.inputs,
            slot.outputs ? nullptr : &slot.outputs,
            graph
        );
        if (status != qnn::tools::iotensor::StatusCode::SUCCESS ||
            !slot.inputs || !slot.outputs) {
            std::fprintf(stderr,
                "[BREEZE_QNN_GRAPH] idx=%zu name=%s tensor-setup-failed status=%d input=%p output=%p\n",
                index, graph.graphName ? graph.graphName : "<unnamed>",
                (int)status, (void *)slot.inputs, (void *)slot.outputs);
            return false;
        }
        return true;
    }

    int graph_frames(size_t index) {
        if (!setup_io(index)) return 0;
        auto & graph = (*m_graphsInfo)[index];
        const auto & in = graph_io[index].inputs[0];
        const auto & out = graph_io[index].outputs[0];
        const size_t feature_elements = tensor_elements(in);
        const size_t pcm_elements = tensor_elements(out);
        const uint32_t rank = QNN_TENSOR_GET_RANK(in);
        const uint32_t * dims = QNN_TENSOR_GET_DIMENSIONS(in);
        if (rank == 3 && dims) {
            const int frames = dims[1] == 512 && dims[2] > 0 ? (int)dims[2] :
                               dims[2] == 512 && dims[1] > 0 ? (int)dims[1] : 0;
            if (frames > 0 &&
                feature_elements == (size_t)frames * 512u &&
                pcm_elements == (size_t)frames * 1920u) return frames;
        }
        // Older exported contexts may flatten [1, frames, 512]. This is a
        // strict compatibility path, NOT permission to accept arbitrary graphs.
        if (feature_elements && feature_elements % 512u == 0u) {
            const size_t frames = feature_elements / 512u;
            if ((frames == 8u || frames == 32u || frames == 64u) &&
                pcm_elements == frames * 1920u) {
                std::fprintf(stderr,
                    "[BREEZE_QNN_GRAPH] idx=%zu name=%s matched=element-count "
                    "frames=%zu input=%s output=%s\n",
                    index, graph.graphName ? graph.graphName : "<unnamed>",
                    frames, describe_shape(in).c_str(), describe_shape(out).c_str());
                return (int) frames;
            }
        }
        std::fprintf(stderr,
            "[BREEZE_QNN_GRAPH] idx=%zu name=%s rejected input_rank=%u input=%s "
            "in_elems=%zu output_rank=%u output=%s out_elems=%zu\n",
            index, graph.graphName ? graph.graphName : "<unnamed>",
            (unsigned)rank, describe_shape(in).c_str(), feature_elements,
            (unsigned)QNN_TENSOR_GET_RANK(out),
            describe_shape(out).c_str(), pcm_elements);
        return 0;
    }

    int find_graph(int frames) {
        if (!m_graphsInfo || m_graphsCount == 0) return -1;
        for (size_t i = 0; i < m_graphsCount; ++i) {
            if (graph_frames(i) == frames) return (int) i;
        }
        return -1;
    }

    bool supports_frames(int frames) {
        return find_graph(frames) >= 0;
    }

    bool execute(
        float * features,
        size_t feature_count,
        float * audio,
        size_t sample_count,
        int frames
    ) {
        const int graph_index = find_graph(frames);
        if (graph_index < 0) {
            std::fprintf(
                stderr,
                "[BREEZE_QNN] no vocoder graph for %d frames (graphs=%u)\n",
                frames,
                (unsigned) m_graphsCount
            );
            return false;
        }
        auto & graph = (*m_graphsInfo)[(size_t) graph_index];
        auto & slot = graph_io[(size_t) graph_index];
        if (graph.numInputTensors != 1 || graph.numOutputTensors != 1) return false;

        auto & in = slot.inputs[0];
        auto & out = slot.outputs[0];

        const uint32_t rank = QNN_TENSOR_GET_RANK(in);
        const uint32_t * dims = QNN_TENSOR_GET_DIMENSIONS(in);
        const size_t input_elems = tensor_elements(in);
        if (input_elems != feature_count) {
            std::fprintf(
                stderr,
                "[BREEZE_QNN] input element mismatch frames=%d graph=%zu host=%zu\n",
                frames,
                input_elems,
                feature_count
            );
            return false;
        }

        float * src_features = features;
        std::vector<float> repacked;
        const char * layout = "NFC";
        if (rank == 3 && dims) {
            if (dims[1] == (uint32_t) frames && dims[2] == 512) {
                layout = "NFC";
            } else if (dims[1] == 512 && dims[2] == (uint32_t) frames) {
                layout = "NCF";
                repacked.resize(feature_count);
                for (size_t t = 0; t < (size_t) frames; ++t) {
                    for (size_t ch = 0; ch < 512; ++ch) {
                        repacked[ch * (size_t) frames + t] =
                            features[t * 512u + ch];
                    }
                }
                src_features = repacked.data();
            } else {
                std::fprintf(
                    stderr,
                    "[BREEZE_QNN] unexpected input shape rank=3 dims=%u,%u,%u frames=%d\n",
                    dims[0], dims[1], dims[2], frames
                );
                return false;
            }
        }

        if (
            m_ioTensor.copyFromFloatToNative(src_features, &in) !=
            qnn::tools::iotensor::StatusCode::SUCCESS
        ) {
            std::fprintf(
                stderr,
                "[BREEZE_QNN] failed to convert FP32 host features to native input type=%d\n",
                (int) QNN_TENSOR_GET_DATA_TYPE(in)
            );
            return false;
        }

        if (rank == 3 && dims) {
            std::fprintf(
                stderr,
                "[BREEZE_QNN_IO] graph=%d input=%ux%ux%u layout=%s native_type=%d bytes=%u\n",
                frames,
                dims[0], dims[1], dims[2], layout,
                (int) QNN_TENSOR_GET_DATA_TYPE(in),
                QNN_TENSOR_GET_CLIENT_BUF(in).dataSize
            );
        }

        const auto t0 = std::chrono::steady_clock::now();
        const auto st = m_qnnFunctionPointers.qnnInterface.graphExecute(
            graph.graph,
            slot.inputs,
            graph.numInputTensors,
            slot.outputs,
            graph.numOutputTensors,
            m_profileBackendHandle,
            nullptr
        );
        const double ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0
        ).count();
        if (st != QNN_GRAPH_NO_ERROR) {
            std::fprintf(
                stderr,
                "[BREEZE_QNN] graphExecute failed frames=%d err=%d\n",
                frames,
                (int) st
            );
            return false;
        }

        if (
            m_ioTensor.convertToFloatInto(audio, &out) !=
            qnn::tools::iotensor::StatusCode::SUCCESS
        ) {
            std::fprintf(
                stderr,
                "[BREEZE_QNN] failed to convert native PCM output type=%d to FP32\n",
                (int) QNN_TENSOR_GET_DATA_TYPE(out)
            );
            return false;
        }

        double sum = 0.0;
        double sq = 0.0;
        float peak = 0.0f;
        size_t bad = 0;
        for (size_t i = 0; i < sample_count; ++i) {
            const float v = audio[i];
            if (!std::isfinite(v)) {
                ++bad;
                continue;
            }
            sum += v;
            sq += (double) v * v;
            peak = std::max(peak, std::fabs(v));
        }
        const double rms = sample_count ? std::sqrt(sq / sample_count) : 0.0;
        std::fprintf(
            stderr,
            "[BREEZE_QNN] graph%d_ms=%.2f input_type=%d output_type=%d checksum=%.7g peak=%.7g rms=%.7g nonfinite=%zu\n",
            frames,
            ms,
            (int) QNN_TENSOR_GET_DATA_TYPE(in),
            (int) QNN_TENSOR_GET_DATA_TYPE(out),
            sum,
            peak,
            rms,
            bad
        );
        return bad == 0;
    }

    // Short speech flushes benefit from race-to-idle. QNN owns the HTP clock
    // vote for the native vocoder context and releases it with the context.
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

    candidate->set_adaptive_power();
    app = std::move(candidate);
    return true;
}

static bool read_exact_floats(
    const std::string & path,
    size_t count,
    std::vector<float> & out
) {
    std::error_code ec;
    const auto bytes = std::filesystem::file_size(path, ec);
    if (ec || bytes != count * sizeof(float)) {
        std::fprintf(
            stderr,
            "[BREEZE_QNN_SELFTEST] bad reference size path=%s got=%llu expected=%zu\n",
            path.c_str(),
            (unsigned long long) bytes,
            count * sizeof(float)
        );
        return false;
    }
    out.resize(count);
    std::ifstream in(path, std::ios::binary);
    return (bool) in.read(
        reinterpret_cast<char *>(out.data()),
        (std::streamsize) (count * sizeof(float))
    );
}

static bool run_qnn_reference_selftest(
    BreezeQnnApp & app,
    const std::string & features_path,
    const std::string & audio_path
) {
    constexpr size_t kFeatures = 64u * 512u;
    constexpr size_t kSamples = 64u * 1920u;
    std::vector<float> features;
    std::vector<float> expected;
    if (!read_exact_floats(features_path, kFeatures, features) ||
        !read_exact_floats(audio_path, kSamples, expected)) {
        return false;
    }

    std::vector<float> got(kSamples, 0.0f);
    if (!app.execute(features.data(), features.size(), got.data(), got.size(), 64)) {
        std::fprintf(stderr, "[BREEZE_QNN_SELFTEST] graph execution failed\n");
        return false;
    }

    double abs_sum = 0.0;
    double dot = 0.0;
    double got_sq = 0.0;
    double ref_sq = 0.0;
    float max_abs = 0.0f;
    float peak = 0.0f;
    size_t bad = 0;
    for (size_t i = 0; i < kSamples; ++i) {
        const float a = got[i];
        const float b = expected[i];
        if (!std::isfinite(a) || !std::isfinite(b)) {
            ++bad;
            continue;
        }
        const float d = std::fabs(a - b);
        max_abs = std::max(max_abs, d);
        abs_sum += d;
        peak = std::max(peak, std::fabs(a));
        dot += (double) a * b;
        got_sq += (double) a * a;
        ref_sq += (double) b * b;
    }
    const double mean_abs = abs_sum / (double) kSamples;
    const double got_rms = std::sqrt(got_sq / (double) kSamples);
    const double ref_rms = std::sqrt(ref_sq / (double) kSamples);
    const double corr = (got_sq > 0.0 && ref_sq > 0.0)
        ? dot / std::sqrt(got_sq * ref_sq)
        : 0.0;

    std::fprintf(
        stderr,
        "[BREEZE_QNN_SELFTEST] max_abs=%.7g mean_abs=%.7g peak=%.7g "
        "rms=%.7g ref_rms=%.7g corr=%.7g nonfinite=%zu\n",
        max_abs,
        mean_abs,
        peak,
        got_rms,
        ref_rms,
        corr,
        bad
    );

    // FP16 HTP math can differ slightly from PyTorch/ONNX FP32, but a valid
    // vocoder must have real signal and strongly agree with the reference.
    return bad == 0 &&
           peak > 1e-3f &&
           got_rms > 5e-5 &&
           ref_rms > 5e-5 &&
           corr > 0.80 &&
           mean_abs < 0.05;
}

} // namespace

struct BreezeQnnVocoder::Impl {
    std::unique_ptr<BreezeQnnApp> app;
    std::vector<int> history;
    std::vector<float> lut;
    std::vector<int> graph_frames{8, 32, 64};
    int left_context=25,n_codebooks=16,codebook_size=2048,feature_channels=512,samples_per_frame=1920;
};
BreezeQnnVocoder::BreezeQnnVocoder():impl_(std::make_unique<Impl>()){}
BreezeQnnVocoder::~BreezeQnnVocoder()=default;
bool BreezeQnnVocoder::init_from_environment(){
    const char *path=std::getenv("BREEZE_QNN_VOCODER_PATH"),
               *lp=std::getenv("BREEZE_QNN_VOCODER_LUT_PATH"),
               *lib=std::getenv("BREEZE_QNN_LIB_DIR"),
               *stf=std::getenv("BREEZE_QNN_SELFTEST_FEATURES_PATH"),
               *sta=std::getenv("BREEZE_QNN_SELFTEST_AUDIO_PATH"),
               *skip_st=std::getenv("BREEZE_QNN_SKIP_SELFTEST");
    const bool skip_selftest=skip_st&&*skip_st&&std::atoi(skip_st)!=0;
    if(!path||!*path||!lp||!*lp||!lib||!*lib) return false;
    if(!skip_selftest&&(!stf||!*stf||!sta||!*sta)) return false;
    if(!std::filesystem::is_regular_file(path)||
       !std::filesystem::is_regular_file(lp)) return false;
    if(!skip_selftest&&
       (!std::filesystem::is_regular_file(stf)||
        !std::filesystem::is_regular_file(sta))) return false;
    const size_t elems=(size_t)impl_->n_codebooks*impl_->codebook_size*impl_->feature_channels, bytes=elems*sizeof(float);
    std::error_code ec; const auto actual=std::filesystem::file_size(lp,ec);
    if(ec||actual!=bytes){std::fprintf(stderr,"[BREEZE_QNN] LUT size got=%llu expected=%zu\n",(unsigned long long)actual,bytes);return false;}
    impl_->lut.resize(elems); std::ifstream in(lp,std::ios::binary);
    if(!in.read(reinterpret_cast<char*>(impl_->lut.data()),(std::streamsize)bytes)){impl_->lut.clear();return false;}
    if(!load_qnn(lib,path,impl_->app)){impl_->lut.clear();return false;}
    const char * selftest_state="passed";
    if(skip_selftest){
        selftest_state="cached";
        std::fprintf(stderr,"[BREEZE_QNN_SELFTEST] skipped cached=1\n");
    }else if(!run_qnn_reference_selftest(*impl_->app, stf, sta)){
        std::fprintf(
            stderr,
            "[BREEZE_QNN_SELFTEST] FAILED; refusing broken QNN artifact and falling back\n"
        );
        impl_->app.reset();
        impl_->lut.clear();
        return false;
    }
    // v4 contains 8/32/64-frame graphs. The proven v3 fallback contains
    // only 64. Keep one runtime implementation that accepts either artifact.
    if(!impl_->app->supports_frames(64)){
        std::fprintf(stderr,"[BREEZE_QNN] required 64-frame vocoder graph missing\n");
        impl_->app.reset();
        impl_->lut.clear();
        return false;
    }
    std::string available;
    for(int frames:impl_->graph_frames){
        if(impl_->app->supports_frames(frames)){
            if(!available.empty()) available += ",";
            available += std::to_string(frames);
        }
    }
    impl_->history.clear();
    std::fprintf(
        stderr,
        "[BREEZE_QNN] ready path=%s lut=%s graph_frames=%s left_context=25 "
        "features=512 layout=NFC backend=QNN-HTP-flex-sm8850-v81 selftest=%s\n",
        path,lp,available.c_str(),selftest_state
    );
    return true;
}
bool BreezeQnnVocoder::ready() const{return impl_&&impl_->app&&!impl_->lut.empty();}
void BreezeQnnVocoder::reset(){if(impl_)impl_->history.clear();}
std::vector<float> BreezeQnnVocoder::decode_stream(
    const std::vector<int>&codes,int T,int ncb,int spf
){
    if(!ready()||T<=0)return {};
    if(
        ncb!=impl_->n_codebooks||
        spf!=impl_->samples_per_frame||
        codes.size()!=(size_t)T*ncb
    )throw std::runtime_error("Breeze QNN vocoder shape mismatch");

    const int hf=(int)impl_->history.size()/ncb;
    const int ctx=std::min(hf,impl_->left_context);
    const int required=ctx+T;
    int graph_frames=0;
    for(int candidate:impl_->graph_frames){
        if(candidate>=required && impl_->app->supports_frames(candidate)){
            graph_frames=candidate;
            break;
        }
    }
    if(graph_frames<=0){
        throw std::runtime_error("Breeze QNN vocoder chunk exceeds largest graph");
    }
    std::vector<float> features(
        (size_t)impl_->feature_channels*(size_t)graph_frames,
        0.0f
    );
    auto add=[&](int dt,const int*fc){
        for(int cb=0;cb<ncb;cb++){
            int code=fc[cb];
            if(code<0||code>=impl_->codebook_size){
                throw std::runtime_error("Breeze QNN code id out of range");
            }
            size_t row=(
                (size_t)cb*impl_->codebook_size+(size_t)code
            )*impl_->feature_channels;
            const float*src=impl_->lut.data()+row;
            for(int ch=0;ch<impl_->feature_channels;ch++){
                features[(size_t)dt*impl_->feature_channels+ch]+=src[ch];
            }
        }
    };
    for(int t=0;t<ctx;t++){
        int sf=hf-ctx+t;
        add(t,impl_->history.data()+(size_t)sf*ncb);
    }
    for(int t=0;t<T;t++){
        add(ctx+t,codes.data()+(size_t)t*ncb);
    }

    double fsum=0;
    float fpeak=0;
    for(float v:features){
        fsum+=v;
        fpeak=std::max(fpeak,std::fabs(v));
    }
    std::fprintf(
        stderr,
        "[BREEZE_QNN_INPUT] graph=%d ctx=%d new=%d checksum=%.7g peak=%.7g\n",
        graph_frames,ctx,T,fsum,fpeak
    );

    std::vector<float> full((size_t)graph_frames*(size_t)spf);
    if(!impl_->app->execute(
        features.data(),features.size(),
        full.data(),full.size(),
        graph_frames
    )){
        throw std::runtime_error("Breeze QNN vocoder execution failed");
    }

    const size_t begin=(size_t)ctx*(size_t)spf;
    const size_t count=(size_t)T*(size_t)spf;
    if(begin+count>full.size()){
        throw std::runtime_error("Breeze QNN vocoder output slice overflow");
    }
    std::vector<float> out(
        full.begin()+begin,
        full.begin()+begin+count
    );
    for(float v:out){
        if(!std::isfinite(v)){
            throw std::runtime_error("Breeze QNN vocoder produced non-finite PCM");
        }
    }

    std::vector<int> merged;
    merged.reserve((size_t)(ctx+T)*ncb);
    if(ctx>0){
        auto first=impl_->history.end()-(size_t)ctx*ncb;
        merged.insert(merged.end(),first,impl_->history.end());
    }
    merged.insert(merged.end(),codes.begin(),codes.end());
    const int mf=(int)merged.size()/ncb;
    const int keep=std::min(mf,impl_->left_context);
    impl_->history.assign(
        merged.end()-(size_t)keep*ncb,
        merged.end()
    );
    return out;
}

} // namespace breeze
