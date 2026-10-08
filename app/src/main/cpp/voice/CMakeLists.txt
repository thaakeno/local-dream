cmake_minimum_required(VERSION 3.19)
project(localdream_breeze_htp LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT BREEZE_SOURCE_DIR OR NOT EXISTS "${BREEZE_SOURCE_DIR}/include/breeze/breeze.h")
    message(FATAL_ERROR "BREEZE_SOURCE_DIR must point at Breeze-TTS-2.cpp")
endif()
if(NOT GGML_SOURCE_DIR OR NOT EXISTS "${GGML_SOURCE_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "GGML_SOURCE_DIR must point at the pinned ggml-hexagon ggml tree")
endif()
if(NOT QNN_SDK_ROOT)
    set(QNN_SDK_ROOT "$ENV{QNN_SDK_ROOT}")
endif()
if(NOT QNN_SDK_ROOT OR NOT EXISTS "${QNN_SDK_ROOT}/include/QNN/QnnInterface.h")
    message(FATAL_ERROR "QNN_SDK_ROOT must point at the QAIRT/QNN SDK")
endif()

# Reuse the same QAIRT SampleApp runtime Local Dream already ships for image
# models. This keeps the speech path on official QNN APIs without a second
# inference framework.
set(QNN_SAMPLE_APP_DIR "${CMAKE_CURRENT_BINARY_DIR}/3rdparty/SampleApp")
file(COPY "${QNN_SDK_ROOT}/examples/QNN/SampleApp/SampleApp/"
     DESTINATION "${QNN_SAMPLE_APP_DIR}")
execute_process(
    COMMAND git apply --unsafe-paths --directory=${CMAKE_CURRENT_BINARY_DIR}
            "${CMAKE_CURRENT_SOURCE_DIR}/../SampleApp.patch"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../../../../.."
    RESULT_VARIABLE QNN_PATCH_RESULT
    ERROR_VARIABLE QNN_PATCH_ERROR
)
if(NOT QNN_PATCH_RESULT EQUAL 0)
    message(FATAL_ERROR "Failed to apply Local Dream QNN SampleApp patch: ${QNN_PATCH_ERROR}")
endif()

set(QNN_SAMPLE_SRC "${QNN_SAMPLE_APP_DIR}/src")
file(GLOB QNN_SAMPLE_SOURCES
    "${QNN_SAMPLE_SRC}/QnnSampleApp.cpp"
    "${QNN_SAMPLE_SRC}/Log/*.cpp"
    "${QNN_SAMPLE_SRC}/PAL/src/linux/*.cpp"
    "${QNN_SAMPLE_SRC}/PAL/src/common/*.cpp"
    "${QNN_SAMPLE_SRC}/Utils/*.cpp"
    "${QNN_SAMPLE_SRC}/WrapperUtils/*.cpp"
)
add_library(breeze_qnn_runtime STATIC ${QNN_SAMPLE_SOURCES})
target_compile_definitions(breeze_qnn_runtime PRIVATE
    QNN_API=__attribute__\(\(visibility\("default"\)\)\)
)
target_include_directories(breeze_qnn_runtime PUBLIC
    "${QNN_SDK_ROOT}/include/QNN"
    "${QNN_SAMPLE_SRC}"
    "${QNN_SAMPLE_SRC}/CachingUtil"
    "${QNN_SAMPLE_SRC}/Log"
    "${QNN_SAMPLE_SRC}/PAL/include"
    "${QNN_SAMPLE_SRC}/Utils"
    "${QNN_SAMPLE_SRC}/WrapperUtils"
)
target_link_libraries(breeze_qnn_runtime PUBLIC log android dl m c)

set(BREEZE_OVERLAY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/overlay/breeze")

set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(GGML_STATIC ON CACHE BOOL "" FORCE)
set(GGML_HEXAGON ON CACHE BOOL "" FORCE)
set(GGML_HEXAGON_USE_MEMPOOL OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_CPU OFF CACHE BOOL "" FORCE)
set(GGML_LLAMAFILE OFF CACHE BOOL "" FORCE)
set(GGML_VULKAN ON CACHE BOOL "" FORCE)
set(GGML_CUDA OFF CACHE BOOL "" FORCE)
set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)

add_subdirectory("${GGML_SOURCE_DIR}" ggml-build)

set(BREEZE_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/native/breeze-common-htp.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/native/breeze-gpu-depth.cpp"
    "${BREEZE_SOURCE_DIR}/src/audio.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/native/breeze-qnn-vocoder.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/native/breeze-qnn-depth-decoder.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/native/breeze-qnn-backbone.cpp"
    "${BREEZE_SOURCE_DIR}/src/config.cpp"
    "${BREEZE_OVERLAY_DIR}/src/gguf_loader.cpp"
    "${BREEZE_SOURCE_DIR}/src/tokenizer.cpp"
    "${BREEZE_SOURCE_DIR}/src/text_encoder.cpp"
    "${BREEZE_OVERLAY_DIR}/src/backbone.cpp"
    "${BREEZE_OVERLAY_DIR}/src/depth_decoder.cpp"
    "${BREEZE_OVERLAY_DIR}/src/codec.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/native/breeze-codec-conv-htp.cpp"
    "${BREEZE_OVERLAY_DIR}/src/codec_decoder.cpp"
    "${BREEZE_OVERLAY_DIR}/src/codec_transformer.cpp"
    "${BREEZE_SOURCE_DIR}/src/sampling.cpp"
    "${BREEZE_OVERLAY_DIR}/src/generation.cpp"
    "${BREEZE_SOURCE_DIR}/src/text_split.cpp"
    "${BREEZE_SOURCE_DIR}/src/voice.cpp"
    "${BREEZE_SOURCE_DIR}/src/model.cpp"
)

add_library(breeze_core STATIC ${BREEZE_SOURCES})
target_include_directories(
    breeze_core
    PUBLIC
        "${BREEZE_OVERLAY_DIR}/include"
        "${BREEZE_SOURCE_DIR}/include"
)
target_include_directories(breeze_core PRIVATE
    "${QNN_SDK_ROOT}/include/QNN"
    "${QNN_SAMPLE_SRC}"
    "${QNN_SAMPLE_SRC}/CachingUtil"
    "${QNN_SAMPLE_SRC}/Log"
    "${QNN_SAMPLE_SRC}/PAL/include"
    "${QNN_SAMPLE_SRC}/Utils"
    "${QNN_SAMPLE_SRC}/WrapperUtils"
)
target_link_libraries(breeze_core PUBLIC ggml breeze_qnn_runtime)
target_compile_definitions(breeze_core PRIVATE LOCALDREAM_VULKAN_DEPTH=1)

set(BREEZE_WEBUI_HEADER "${CMAKE_CURRENT_BINARY_DIR}/generated/breeze_webui_assets.h")
add_custom_command(
    OUTPUT "${BREEZE_WEBUI_HEADER}"
    COMMAND ${CMAKE_COMMAND}
        -DWEBUI_DIR=${BREEZE_SOURCE_DIR}/webui
        -DOUT_HEADER=${BREEZE_WEBUI_HEADER}
        -P ${BREEZE_SOURCE_DIR}/cmake/embed_webui.cmake
    DEPENDS
        "${BREEZE_SOURCE_DIR}/webui/index.html"
        "${BREEZE_SOURCE_DIR}/webui/style.css"
        "${BREEZE_SOURCE_DIR}/webui/app.js"
        "${BREEZE_SOURCE_DIR}/cmake/embed_webui.cmake"
)
add_custom_target(breeze_webui DEPENDS "${BREEZE_WEBUI_HEADER}")

add_executable(
    breeze-server
    "${BREEZE_SOURCE_DIR}/apps/server/main.cpp"
    "${BREEZE_SOURCE_DIR}/apps/server/server.cpp"
    "${BREEZE_SOURCE_DIR}/apps/server/voices.cpp"
    "${BREEZE_SOURCE_DIR}/apps/server/ws.cpp"
    "${BREEZE_SOURCE_DIR}/apps/server/ws_api.cpp"
    "${BREEZE_SOURCE_DIR}/apps/server/openai.cpp"
    "${BREEZE_SOURCE_DIR}/apps/server/encode.cpp"
)
add_dependencies(breeze-server breeze_webui)
target_include_directories(
    breeze-server
    PRIVATE
        "${CMAKE_CURRENT_BINARY_DIR}/generated"
        "${BREEZE_SOURCE_DIR}/third_party"
        "${BREEZE_SOURCE_DIR}/apps/server"
)
find_package(Threads REQUIRED)
target_link_libraries(breeze-server PRIVATE breeze_core Threads::Threads)
target_link_options(breeze-server PRIVATE "-Wl,-z,defs")


add_executable(
    breeze-htp-selftest
    "${CMAKE_CURRENT_SOURCE_DIR}/native/breeze-htp-selftest.cpp"
)
target_link_libraries(breeze-htp-selftest PRIVATE breeze_core Threads::Threads)
target_link_options(breeze-htp-selftest PRIVATE "-Wl,-z,defs")
