cmake_minimum_required(VERSION 3.18)
project(moss_engine C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# A second stable-diffusion.cpp version is intentionally linked to a DIFFERENT
# shared object. No in-place patching, no overriding the working Qwen/SDXL engine.
set(MOSS_SDCPP "${CMAKE_CURRENT_SOURCE_DIR}/../3rdparty/moss-stable-diffusion.cpp")
if(NOT EXISTS "${MOSS_SDCPP}/include/stable-diffusion.h")
    message(FATAL_ERROR "Initialize the pinned MOSS git submodule recursively")
endif()

set(GGML_HEXAGON ON CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_LLAMAFILE OFF CACHE BOOL "" FORCE)
set(SD_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(SD_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(PREBUILT_LIB_DIR "android_aarch64" CACHE STRING "" FORCE)

add_subdirectory("${MOSS_SDCPP}" moss-sdcpp EXCLUDE_FROM_ALL)
add_library(moss_engine SHARED MossEngine.cpp)
target_include_directories(moss_engine PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../include"
    "${MOSS_SDCPP}/include"
)
target_link_libraries(moss_engine PRIVATE stable-diffusion log)
target_compile_options(moss_engine PRIVATE -fvisibility=hidden)
target_link_options(moss_engine PRIVATE
    "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/../dit/dit_engine.map"
    "-Wl,-s"
)
set_target_properties(moss_engine PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib/${CMAKE_ANDROID_ARCH_ABI}"
)
foreach(htp_version v79 v81)
    if(TARGET htp-${htp_version})
        add_dependencies(moss_engine htp-${htp_version})
    endif()
endforeach()
