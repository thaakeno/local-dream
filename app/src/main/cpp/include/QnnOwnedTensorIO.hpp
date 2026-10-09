#pragma once
// Local Dream-owned, patch-free tensor adapter. Qualcomm's SDK allocation
// helpers are reused unchanged; conversions and validation live in our code.
#include <IOTensor.hpp>
#include <QnnSampleAppUtils.hpp>
#include <cstdlib>
#include <QnnTypeMacros.hpp>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

namespace localdream::qnn {
class QnnTensorIO final : public ::qnn::tools::iotensor::IOTensor {
    using Result = ::qnn::tools::iotensor::StatusCode;
    static bool elements(const Qnn_Tensor_t &t, size_t &n) {
        n = 1;
        const auto rank = QNN_TENSOR_GET_RANK(t);
        const auto *dims = QNN_TENSOR_GET_DIMENSIONS(t);
        if (rank && !dims) return false;
        for (uint32_t i=0;i<rank;++i) {
            if (!dims[i] || n>std::numeric_limits<size_t>::max()/dims[i]) return false;
            n*=dims[i];
        }
        return true;
    }
    static bool scale(const Qnn_Tensor_t &t,float &s,int32_t &z) {
        auto q=QNN_TENSOR_GET_QUANT_PARAMS(t);
        if (q.quantizationEncoding!=QNN_QUANTIZATION_ENCODING_SCALE_OFFSET) return false;
        s=q.scaleOffsetEncoding.scale; z=q.scaleOffsetEncoding.offset;
        return std::isfinite(s)&&s>0;
    }
    template<typename T> static bool capacity(const Qnn_ClientBuffer_t &b,size_t n) {
        return b.data && n<=std::numeric_limits<size_t>::max()/sizeof(T) &&
            b.dataSize>=n*sizeof(T);
    }
    template<typename T> static void encode(const float *src,T *dst,size_t n,float s,int32_t z) {
        const double lo=static_cast<double>(std::numeric_limits<T>::min());
        const double hi=static_cast<double>(std::numeric_limits<T>::max());
        for(size_t i=0;i<n;++i) {
            if (!std::isfinite(src[i])) { dst[i]=static_cast<T>(0); continue; }
            dst[i]=static_cast<T>(std::clamp(
                std::nearbyint(static_cast<double>(src[i])/s-z),lo,hi));
        }
    }
    template<typename T> static void decode(const T *src,float *dst,size_t n,float s,int32_t z) {
        for(size_t i=0;i<n;++i) dst[i]=(static_cast<float>(src[i])+z)*s;
    }
public:
    // SDK 2.50.0.260828's IOTensor::allocateBuffer omits FLOAT_16. This
    // first-party class owns FP16 graph buffers rather than modifying any
    // Qualcomm SDK source file. Non-FP16 graphs retain the unmodified SDK path.
    // The native SDK teardown can free these buffers because the metadata is
    // cloned with Qualcomm's public deepCopyQnnTensorInfo contract and every
    // client allocation uses calloc()/free().
    static constexpr size_t element_bytes(Qnn_DataType_t type) {
        switch(type) {
        case QNN_DATATYPE_BOOL_8:
        case QNN_DATATYPE_UINT_8:
        case QNN_DATATYPE_INT_8:
        case QNN_DATATYPE_UFIXED_POINT_8:
        case QNN_DATATYPE_SFIXED_POINT_8: return 1;
        case QNN_DATATYPE_FLOAT_16:
        case QNN_DATATYPE_UINT_16:
        case QNN_DATATYPE_INT_16:
        case QNN_DATATYPE_UFIXED_POINT_16:
        case QNN_DATATYPE_SFIXED_POINT_16: return 2;
        case QNN_DATATYPE_FLOAT_32:
        case QNN_DATATYPE_UINT_32:
        case QNN_DATATYPE_INT_32:
        case QNN_DATATYPE_UFIXED_POINT_32:
        case QNN_DATATYPE_SFIXED_POINT_32: return 4;
        case QNN_DATATYPE_UINT_64:
        case QNN_DATATYPE_INT_64: return 8;
        default: return 0;
        }
    }

private:
    static bool has_fp16(const Qnn_Tensor_t * tensors, uint32_t count) {
        if (count && !tensors) return false;
        for(uint32_t i=0;i<count;++i)
            if(QNN_TENSOR_GET_DATA_TYPE(tensors[i])==QNN_DATATYPE_FLOAT_16)
                return true;
        return false;
    }

    static Result create_tensors(Qnn_Tensor_t ** destination,
                                  const Qnn_Tensor_t * source,
                                  uint32_t count) {
        if (!destination || (count && !source)) return Result::FAILURE;
        *destination=nullptr;
        if (!count) return Result::SUCCESS;
        // Copy names, rank, dimensions and quantization metadata through
        // Qualcomm's public helper, which understands the current QNN ABI.
        auto * tensors=static_cast<Qnn_Tensor_t*>(
            std::calloc((size_t)count,sizeof(Qnn_Tensor_t)));
        if(!tensors) return Result::FAILURE;
        *destination=tensors;
        for(uint32_t i=0;i<count;++i) {
            tensors[i]=QNN_TENSOR_INIT;
            if(!::qnn::tools::sample_app::deepCopyQnnTensorInfo(
                    &tensors[i],const_cast<Qnn_Tensor_t*>(&source[i])))
                return Result::FAILURE;
            QNN_TENSOR_SET_MEM_TYPE(&tensors[i],QNN_TENSORMEMTYPE_RAW);
            size_t n=0;
            const size_t stride=element_bytes(QNN_TENSOR_GET_DATA_TYPE(tensors[i]));
            if(!stride || !elements(tensors[i],n) || !n ||
                n>std::numeric_limits<size_t>::max()/stride)
                return Result::FAILURE;
            const size_t bytes=n*stride;
            Qnn_ClientBuffer_t buffer=QNN_CLIENT_BUFFER_INIT;
            if(bytes>std::numeric_limits<decltype(buffer.dataSize)>::max())
                return Result::FAILURE;
            buffer.data=std::calloc(1,bytes);
            if(!buffer.data) return Result::FAILURE;
            buffer.dataSize=static_cast<decltype(buffer.dataSize)>(bytes);
            QNN_TENSOR_SET_CLIENT_BUF(&tensors[i],buffer);
        }
        return Result::SUCCESS;
    }

public:
    Result setupInputAndOutputTensors(
            Qnn_Tensor_t **inputs,Qnn_Tensor_t **outputs,
            ::qnn_wrapper_api::GraphInfo_t graph) {
        if(!inputs||!outputs)return Result::FAILURE;
        // Using the SDK's tested fast path for other types minimizes changes
        // to image generation, while owned FP16 allocation fixes Breeze v3.
        if(!has_fp16(graph.inputTensors,graph.numInputTensors) &&
           !has_fp16(graph.outputTensors,graph.numOutputTensors))
            return ::qnn::tools::iotensor::IOTensor::setupInputAndOutputTensors(
                inputs,outputs,graph);

        *inputs=nullptr;
        *outputs=nullptr;
        const auto a=create_tensors(inputs,graph.inputTensors,graph.numInputTensors);
        if(a==Result::SUCCESS) {
            const auto b=create_tensors(outputs,graph.outputTensors,graph.numOutputTensors);
            if(b==Result::SUCCESS) return Result::SUCCESS;
        }
        // The SDK's public teardown accepts the exact same deep-copy and
        // calloc ownership contract and releases partially built tensors.
        ::qnn::tools::iotensor::IOTensor::tearDownInputAndOutputTensors(
            *inputs,*outputs,graph.numInputTensors,graph.numOutputTensors);
        *inputs=nullptr;
        *outputs=nullptr;
        return Result::FAILURE;
    }

    Result copyFromFloatToNative(const float *src,Qnn_Tensor_t *t) {
        if (!src||!t) return Result::FAILURE;
        size_t n=0; if(!elements(*t,n)) return Result::FAILURE;
        auto b=QNN_TENSOR_GET_CLIENT_BUF(*t);
        switch(QNN_TENSOR_GET_DATA_TYPE(*t)) {
        case QNN_DATATYPE_FLOAT_32:
            if(!capacity<float>(b,n)) return Result::FAILURE;
            std::memcpy(b.data,src,n*sizeof(float));return Result::SUCCESS;
        case QNN_DATATYPE_FLOAT_16: {
            if(!capacity<_Float16>(b,n)) return Result::FAILURE;
            auto *out=static_cast<_Float16*>(b.data);
            for(size_t i=0;i<n;++i) out[i]=static_cast<_Float16>(src[i]);
            return Result::SUCCESS;
        }
        case QNN_DATATYPE_UFIXED_POINT_8:
        case QNN_DATATYPE_SFIXED_POINT_8:
        case QNN_DATATYPE_UFIXED_POINT_16:
        case QNN_DATATYPE_SFIXED_POINT_16: {
            float s;int32_t z;if(!scale(*t,s,z)) return Result::FAILURE;
            switch(QNN_TENSOR_GET_DATA_TYPE(*t)) {
            case QNN_DATATYPE_UFIXED_POINT_8:
                if(!capacity<uint8_t>(b,n)) return Result::FAILURE;
                encode(src,static_cast<uint8_t*>(b.data),n,s,z);break;
            case QNN_DATATYPE_SFIXED_POINT_8:
                if(!capacity<int8_t>(b,n)) return Result::FAILURE;
                encode(src,static_cast<int8_t*>(b.data),n,s,z);break;
            case QNN_DATATYPE_UFIXED_POINT_16:
                if(!capacity<uint16_t>(b,n)) return Result::FAILURE;
                encode(src,static_cast<uint16_t*>(b.data),n,s,z);break;
            default:
                if(!capacity<int16_t>(b,n)) return Result::FAILURE;
                encode(src,static_cast<int16_t*>(b.data),n,s,z);break;
            }
            return Result::SUCCESS;
        }
        default:return Result::FAILURE;
        }
    }
    Result convertToFloatInto(float *dst,Qnn_Tensor_t *t) {
        if(!dst||!t) return Result::FAILURE;
        size_t n=0;if(!elements(*t,n)) return Result::FAILURE;
        auto b=QNN_TENSOR_GET_CLIENT_BUF(*t);
        switch(QNN_TENSOR_GET_DATA_TYPE(*t)) {
        case QNN_DATATYPE_FLOAT_32:
            if(!capacity<float>(b,n))return Result::FAILURE;
            std::memcpy(dst,b.data,n*sizeof(float));return Result::SUCCESS;
        case QNN_DATATYPE_FLOAT_16: {
            if(!capacity<_Float16>(b,n)) return Result::FAILURE;
            const auto *src=static_cast<const _Float16*>(b.data);
            for(size_t i=0;i<n;++i)dst[i]=static_cast<float>(src[i]);
            return Result::SUCCESS;
        }
        case QNN_DATATYPE_UFIXED_POINT_8:
        case QNN_DATATYPE_SFIXED_POINT_8:
        case QNN_DATATYPE_UFIXED_POINT_16:
        case QNN_DATATYPE_SFIXED_POINT_16: {
            float s;int32_t z;if(!scale(*t,s,z)) return Result::FAILURE;
            switch(QNN_TENSOR_GET_DATA_TYPE(*t)) {
            case QNN_DATATYPE_UFIXED_POINT_8:
                if(!capacity<uint8_t>(b,n))return Result::FAILURE;
                decode(static_cast<const uint8_t*>(b.data),dst,n,s,z);break;
            case QNN_DATATYPE_SFIXED_POINT_8:
                if(!capacity<int8_t>(b,n))return Result::FAILURE;
                decode(static_cast<const int8_t*>(b.data),dst,n,s,z);break;
            case QNN_DATATYPE_UFIXED_POINT_16:
                if(!capacity<uint16_t>(b,n))return Result::FAILURE;
                decode(static_cast<const uint16_t*>(b.data),dst,n,s,z);break;
            default:
                if(!capacity<int16_t>(b,n))return Result::FAILURE;
                decode(static_cast<const int16_t*>(b.data),dst,n,s,z);break;
            }
            return Result::SUCCESS;
        }
        default:return Result::FAILURE;
        }
    }
};
} // namespace localdream::qnn
