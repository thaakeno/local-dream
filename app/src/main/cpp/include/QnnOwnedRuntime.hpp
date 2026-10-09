#pragma once
// Owned RAII QNN context, zero SDK source patches or private-member access.
// Qualcomm's public SampleApp utilities provide graph metadata cloning and
// tensor storage; all lifecycle and model-specific policy stays in Local Dream.
#include <QnnSampleApp.hpp>
#include <QnnSampleAppUtils.hpp>
#include <QnnOwnedTensorIO.hpp>
#include <QnnTypeMacros.hpp>
#include <cstdint>
#include <fcntl.h>
#include <limits>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
namespace localdream::qnn {
class QnnOwnedRuntime {
public:
    using StatusCode = ::qnn::tools::sample_app::StatusCode;
    using QnnFunctionPointers = ::qnn::tools::sample_app::QnnFunctionPointers;
    using ProfilingLevel = ::qnn::tools::sample_app::ProfilingLevel;
    using OutputDataType = ::qnn::tools::iotensor::OutputDataType;
    using InputDataType = ::qnn::tools::iotensor::InputDataType;
protected:
    QnnFunctionPointers m_qnnFunctionPointers;
    Qnn_BackendHandle_t m_backendHandle = nullptr;
    Qnn_DeviceHandle_t m_deviceHandle = nullptr;
    Qnn_LogHandle_t m_logHandle = nullptr;
    Qnn_ProfileHandle_t m_profileBackendHandle = nullptr;
    Qnn_ContextHandle_t m_context = nullptr;
    QnnContext_Config_t **m_contextConfig = nullptr;
    qnn_wrapper_api::GraphInfo_t **m_graphsInfo = nullptr;
    uint32_t m_graphsCount = 0;
    QnnTensorIO m_ioTensor;
    std::string m_cachedBinaryPath;
    ProfilingLevel m_profilingLevel = ProfilingLevel::OFF;
    bool m_isContextCreated = false;
public:
    QnnOwnedRuntime(QnnFunctionPointers pointers, const std::string &, const std::string &,
        void *, const std::string &, bool, OutputDataType, InputDataType,
        ProfilingLevel level, bool, const std::string &path, const std::string &)
        :m_qnnFunctionPointers(pointers),m_cachedBinaryPath(path),m_profilingLevel(level){}
    QnnOwnedRuntime(const QnnOwnedRuntime&)=delete;
    QnnOwnedRuntime& operator=(const QnnOwnedRuntime&)=delete;
    virtual ~QnnOwnedRuntime(){freeContext();freeDevice();terminateBackend();}
    StatusCode initialize(){return StatusCode::SUCCESS;}
    StatusCode initializeBackend(){
        auto f=m_qnnFunctionPointers.qnnInterface.backendCreate;
        return f&&f(m_logHandle,nullptr,&m_backendHandle)==QNN_SUCCESS
            ?StatusCode::SUCCESS:StatusCode::FAILURE;
    }
    StatusCode createDevice(){
        auto f=m_qnnFunctionPointers.qnnInterface.deviceCreate;
        return !f||f(m_logHandle,nullptr,&m_deviceHandle)==QNN_SUCCESS
            ?StatusCode::SUCCESS:StatusCode::FAILURE;
    }
    StatusCode initializeProfiling(){
        return m_profilingLevel==ProfilingLevel::OFF?StatusCode::SUCCESS:StatusCode::FAILURE;
    }
    StatusCode registerOpPackages(){return StatusCode::SUCCESS;}
    StatusCode createFromBinary(){
        if(m_cachedBinaryPath.empty()||m_isContextCreated)return StatusCode::FAILURE;
        const int fd=::open(m_cachedBinaryPath.c_str(),O_RDONLY|O_CLOEXEC);
        if(fd<0)return StatusCode::FAILURE;
        struct stat st{};
        if(::fstat(fd,&st)!=0||st.st_size<=0||
            static_cast<uint64_t>(st.st_size)>std::numeric_limits<size_t>::max()){
            ::close(fd);return StatusCode::FAILURE;
        }
        const size_t size=static_cast<size_t>(st.st_size);
        void* map=::mmap(nullptr,size,PROT_READ,MAP_PRIVATE,fd,0);
        ::close(fd);
        if(map==MAP_FAILED)return StatusCode::FAILURE;
        struct Mapping{
            void *p;size_t n;
            ~Mapping(){if(p!=MAP_FAILED)::munmap(p,n);}
        }owner{map,size};
        ::madvise(map,size,MADV_SEQUENTIAL);
        auto &qnn=m_qnnFunctionPointers.qnnInterface;
        auto &sys=m_qnnFunctionPointers.qnnSystemInterface;
        if(!sys.systemContextCreate||!sys.systemContextGetBinaryInfo||
           !sys.systemContextFree||!qnn.contextCreateFromBinary||!qnn.graphRetrieve)
            return StatusCode::FAILURE;
        QnnSystemContext_Handle_t handle=nullptr;
        if(sys.systemContextCreate(&handle)!=QNN_SUCCESS)return StatusCode::FAILURE;
        const QnnSystemContext_BinaryInfo_t *info=nullptr;
        Qnn_ContextBinarySize_t infoSize=0;
        const auto metaResult=sys.systemContextGetBinaryInfo(handle,map,size,&info,&infoSize);
        bool ok=metaResult==QNN_SUCCESS && info &&
           ::qnn::tools::sample_app::copyMetadataToGraphsInfo(info,m_graphsInfo,m_graphsCount);
        sys.systemContextFree(handle);
        if(!ok)return StatusCode::FAILURE;
        if(qnn.contextCreateFromBinary(m_backendHandle,m_deviceHandle,
              (const QnnContext_Config_t**)m_contextConfig,
              map,size,&m_context,m_profileBackendHandle)!=QNN_SUCCESS)
            return StatusCode::FAILURE;
        m_isContextCreated=true;
        for(uint32_t i=0;i<m_graphsCount;++i)
            if(qnn.graphRetrieve(m_context,(*m_graphsInfo)[i].graphName,
                                 &(*m_graphsInfo)[i].graph)!=QNN_SUCCESS)
                return StatusCode::FAILURE;
        return StatusCode::SUCCESS;
    }
    StatusCode extractBackendProfilingInfo(Qnn_ProfileHandle_t,QnnSystemProfile_ProfileData_t*){
        return StatusCode::FAILURE; // profiling intentionally disabled; do not fake data
    }
    StatusCode freeContext(){
        if(m_graphsInfo){
            ::qnn_wrapper_api::freeGraphsInfo(&m_graphsInfo,m_graphsCount);
            m_graphsInfo=nullptr;m_graphsCount=0;
        }
        if(m_context){
            auto f=m_qnnFunctionPointers.qnnInterface.contextFree;
            if(f)f(m_context,nullptr);
            m_context=nullptr;
        }
        m_isContextCreated=false;
        return StatusCode::SUCCESS;
    }
    StatusCode freeDevice(){
        if(m_deviceHandle){
            auto f=m_qnnFunctionPointers.qnnInterface.deviceFree;
            if(f)f(m_deviceHandle);
            m_deviceHandle=nullptr;
        }
        return StatusCode::SUCCESS;
    }
    StatusCode terminateBackend(){
        if(m_backendHandle){
            auto f=m_qnnFunctionPointers.qnnInterface.backendFree;
            if(f)f(m_backendHandle);
            m_backendHandle=nullptr;
        }
        if(m_logHandle){
            auto f=m_qnnFunctionPointers.qnnInterface.logFree;
            if(f)f(m_logHandle);
            m_logHandle=nullptr;
        }
        return StatusCode::SUCCESS;
    }
};
} // namespace localdream::qnn
using localdream::qnn::QnnOwnedRuntime;
