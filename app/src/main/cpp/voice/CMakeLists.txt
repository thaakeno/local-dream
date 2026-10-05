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

set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(GGML_STATIC ON CACHE BOOL "" FORCE)
set(GGML_HEXAGON ON CACHE BOOL "" FORCE)
set(GGML_HEXAGON_USE_MEMPOOL ON CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_CPU OFF CACHE BOOL "" FORCE)
set(GGML_LLAMAFILE OFF CACHE BOOL "" FORCE)
set(GGML_VULKAN OFF CACHE BOOL "" FORCE)
set(GGML_CUDA OFF CACHE BOOL "" FORCE)
set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)

add_subdirectory("${GGML_SOURCE_DIR}" ggml-build)

set(BREEZE_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/native/breeze-common-htp.cpp"
    "${BREEZE_SOURCE_DIR}/src/audio.cpp"
    "${BREEZE_SOURCE_DIR}/src/config.cpp"
    "${BREEZE_SOURCE_DIR}/src/gguf_loader.cpp"
    "${BREEZE_SOURCE_DIR}/src/tokenizer.cpp"
    "${BREEZE_SOURCE_DIR}/src/text_encoder.cpp"
    "${BREEZE_SOURCE_DIR}/src/backbone.cpp"
    "${BREEZE_SOURCE_DIR}/src/depth_decoder.cpp"
    "${BREEZE_SOURCE_DIR}/src/codec.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/native/breeze-codec-conv-htp.cpp"
    "${BREEZE_SOURCE_DIR}/src/codec_decoder.cpp"
    "${BREEZE_SOURCE_DIR}/src/codec_transformer.cpp"
    "${BREEZE_SOURCE_DIR}/src/sampling.cpp"
    "${BREEZE_SOURCE_DIR}/src/generation.cpp"
    "${BREEZE_SOURCE_DIR}/src/text_split.cpp"
    "${BREEZE_SOURCE_DIR}/src/voice.cpp"
    "${BREEZE_SOURCE_DIR}/src/model.cpp"
)

add_library(breeze_core STATIC ${BREEZE_SOURCES})
target_include_directories(breeze_core PUBLIC "${BREEZE_SOURCE_DIR}/include")
target_link_libraries(breeze_core PUBLIC ggml)

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
