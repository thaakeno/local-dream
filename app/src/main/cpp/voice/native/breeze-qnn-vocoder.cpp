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
#include <fstream>
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
            qnn::tools::iotensor::InputDataType::FLOAT,
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

    bool execute(float * features, size_t feature_count, float * audio, size_t sample_count) {
        if (!setup_io()) return false;
        auto & graph = (*m_graphsInfo)[0];
        if (graph.numInputTensors != 1 || graph.numOutputTensors != 1) return false;

        auto & in = inputs[0];
        auto & out = outputs[0];

        const uint32_t rank = QNN_TENSOR_GET_RANK(in);
        const uint32_t * dims = QNN_TENSOR_GET_DIMENSIONS(in);
        size_t input_elems = 1;
        for (uint32_t i = 0; i < rank; ++i) input_elems *= dims ? dims[i] : 1;
        if (input_elems != feature_count) {
            std::fprintf(
                stderr,
                "[BREEZE_QNN] input element mismatch graph=%zu host=%zu\n",
                input_elems,
                feature_count
            );
            return false;
        }

        float * src_features = features;
        std::vector<float> repacked;
        const char * layout = "NFC";
        if (rank == 3 && dims) {
            if (dims[1] == 64 && dims[2] == 512) {
                layout = "NFC";
            } else if (dims[1] == 512 && dims[2] == 64) {
                layout = "NCF";
                repacked.resize(feature_count);
                for (size_t t = 0; t < 64; ++t) {
                    for (size_t ch = 0; ch < 512; ++ch) {
                        repacked[ch * 64 + t] = features[t * 512 + ch];
                    }
                }
                src_features = repacked.data();
            } else {
                std::fprintf(
                    stderr,
                    "[BREEZE_QNN] unexpected input shape rank=3 dims=%u,%u,%u\n",
                    dims[0], dims[1], dims[2]
                );
                return false;
            }
        }

        // The compiled SM8850 context now exposes native FP16 graph IO.
        // Use QAIRT's conversion helpers instead of memcpy so FP32 host LUT
        // features are converted exactly to the graph's native tensor type.
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
                "[BREEZE_QNN_IO] input=%ux%ux%u layout=%s native_type=%d bytes=%u\n",
                dims[0], dims[1], dims[2], layout,
                (int) QNN_TENSOR_GET_DATA_TYPE(in),
                QNN_TENSOR_GET_CLIENT_BUF(in).dataSize
            );
        }

        const auto t0 = std::chrono::steady_clock::now();
        const auto st = m_qnnFunctionPointers.qnnInterface.graphExecute(
            graph.graph,
            inputs,
            graph.numInputTensors,
            outputs,
            graph.numOutputTensors,
            m_profileBackendHandle,
            nullptr
        );
        const double ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0
        ).count();
        if (st != QNN_GRAPH_NO_ERROR) {
            std::fprintf(stderr, "[BREEZE_QNN] graphExecute failed err=%d\n", (int) st);
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
            "[BREEZE_QNN] graph64_ms=%.2f input_type=%d output_type=%d checksum=%.7g peak=%.7g rms=%.7g nonfinite=%zu\n",
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
    if (!app.execute(features.data(), features.size(), got.data(), got.size())) {
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
    int fixed_frames=64,left_context=25,n_codebooks=16,codebook_size=2048,feature_channels=512,samples_per_frame=1920;
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
    impl_->history.clear();
    std::fprintf(stderr,"[BREEZE_QNN] ready path=%s lut=%s graph_frames=64 left_context=25 features=512 layout=NFC backend=QNN-HTP-v3-sm8850-v81 selftest=%s\n",path,lp,selftest_state);
    return true;
}
bool BreezeQnnVocoder::ready() const{return impl_&&impl_->app&&!impl_->lut.empty();}
void BreezeQnnVocoder::reset(){if(impl_)impl_->history.clear();}
std::vector<float> BreezeQnnVocoder::decode_stream(const std::vector<int>&codes,int T,int ncb,int spf){
    if(!ready()||T<=0)return {};
    if(ncb!=impl_->n_codebooks||spf!=impl_->samples_per_frame||codes.size()!=(size_t)T*ncb)throw std::runtime_error("Breeze QNN vocoder shape mismatch");
    const int hf=(int)impl_->history.size()/ncb, ctx=std::min(hf,impl_->left_context);
    if(ctx+T>impl_->fixed_frames)throw std::runtime_error("Breeze QNN vocoder chunk exceeds fixed graph");
    std::vector<float> features((size_t)impl_->feature_channels*impl_->fixed_frames,0.0f);
    auto add=[&](int dt,const int*fc){
        for(int cb=0;cb<ncb;cb++){int code=fc[cb];if(code<0||code>=impl_->codebook_size)throw std::runtime_error("Breeze QNN code id out of range");
            size_t row=((size_t)cb*impl_->codebook_size+(size_t)code)*impl_->feature_channels; const float*src=impl_->lut.data()+row;
            for(int ch=0;ch<impl_->feature_channels;ch++)features[(size_t)dt*impl_->feature_channels+ch]+=src[ch];
        }
    };
    for(int t=0;t<ctx;t++){int sf=hf-ctx+t;add(t,impl_->history.data()+(size_t)sf*ncb);}
    for(int t=0;t<T;t++)add(ctx+t,codes.data()+(size_t)t*ncb);
    double fsum=0;float fpeak=0;for(float v:features){fsum+=v;fpeak=std::max(fpeak,std::fabs(v));}
    std::fprintf(stderr,"[BREEZE_QNN_INPUT] ctx=%d new=%d checksum=%.7g peak=%.7g\n",ctx,T,fsum,fpeak);
    std::vector<float> full((size_t)impl_->fixed_frames*spf);
    if(!impl_->app->execute(features.data(),features.size(),full.data(),full.size()))throw std::runtime_error("Breeze QNN vocoder execution failed");
    const size_t begin=(size_t)ctx*spf,count=(size_t)T*spf;std::vector<float> out(full.begin()+begin,full.begin()+begin+count);
    for(float v:out)if(!std::isfinite(v))throw std::runtime_error("Breeze QNN vocoder produced non-finite PCM");
    std::vector<int> merged;merged.reserve((size_t)(ctx+T)*ncb);
    if(ctx>0){auto first=impl_->history.end()-(size_t)ctx*ncb;merged.insert(merged.end(),first,impl_->history.end());}
    merged.insert(merged.end(),codes.begin(),codes.end());int mf=(int)merged.size()/ncb,keep=std::min(mf,impl_->left_context);
    impl_->history.assign(merged.end()-(size_t)keep*ncb,merged.end());return out;
}

} // namespace breeze
