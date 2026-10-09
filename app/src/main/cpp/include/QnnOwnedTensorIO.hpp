#pragma once
// Local Dream-owned, patch-free tensor adapter. Qualcomm's SDK allocation
// helpers are reused unchanged; conversions and validation live in our code.
#include <IOTensor.hpp>
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
