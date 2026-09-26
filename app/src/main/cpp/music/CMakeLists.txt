cmake_minimum_required(VERSION 3.18)
project(yue2_mobile C CXX)

# YuE2 is built as a separate Android executable, exactly like the existing
# Local Dream core process.  We deliberately use yue2.cpp's supported
# GGML_SOURCE_DIR hook instead of carrying source patches.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(YUE2_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../3rdparty/yue2.cpp")
set(SDCPP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../3rdparty/stable-diffusion.cpp")

# Reuse the exact GGML + Hexagon tree already validated by Local Dream's
# Qwen/DiT runtime.  HTP is the primary backend; CPU stays available to the
# GGML scheduler for the small unsupported housekeeping ops.
set(GGML_SOURCE_DIR "${SDCPP_DIR}/ggml" CACHE PATH "Local Dream GGML tree" FORCE)
set(GGML_HEXAGON ON CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_LLAMAFILE OFF CACHE BOOL "" FORCE)
set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(PREBUILT_LIB_DIR "android_aarch64" CACHE STRING "" FORCE)

add_subdirectory("${YUE2_DIR}" yue2 EXCLUDE_FROM_ALL)

# Android treats native executables shipped in jniLibs as private native
# binaries.  Keep a .so-looking filename so the package installer preserves it,
# but it remains an executable with yue-server's normal main().
set_target_properties(yue-server PROPERTIES
    OUTPUT_NAME "libyue2_server.so"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/${CMAKE_ANDROID_ARCH_ABI}"
)

target_link_libraries(yue-server PRIVATE log android dl m)

# FastRPC loads the DSP-side skels at runtime.  Build the same pair used by the
# image DiT engine so the APK has one coherent HTP runtime for both features.
foreach(htp_version v79 v81)
    if(TARGET htp-${htp_version})
        add_dependencies(yue-server htp-${htp_version})
    endif()
endforeach()
