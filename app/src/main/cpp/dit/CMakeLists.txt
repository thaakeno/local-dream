cmake_minimum_required(VERSION 3.18)
project(dit_engine C CXX)

# Built as its own project, not a subdirectory of the core: stable-diffusion.cpp
# and the ggml Hexagon backend want a newer -march and a higher API level than
# the core, which still has to run on every device we support. The two only
# share include/DitEngine.h.

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Source-of-truth is a pinned, READ-ONLY git submodule. Prior revisions
# ran git apply against it during CMake configure, dirtying its checkout.
# Copy its already-initialized sources into this build directory, stamp the
# patch inputs, and patch only that disposable copy.
set(SDCPP_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../3rdparty/stable-diffusion.cpp")
set(SDCPP_DIR "${CMAKE_CURRENT_BINARY_DIR}/staged-stable-diffusion.cpp")
if(NOT EXISTS "${SDCPP_SOURCE_DIR}/ggml/CMakeLists.txt")
    message(FATAL_ERROR "Initialize stable-diffusion.cpp/ggml recursively first")
endif()
execute_process(COMMAND git -C "${SDCPP_SOURCE_DIR}" rev-parse HEAD
    OUTPUT_VARIABLE SDCPP_SOURCE_REV OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE SDCPP_GIT_RESULT)
execute_process(COMMAND git -C "${SDCPP_SOURCE_DIR}/ggml" rev-parse HEAD
    OUTPUT_VARIABLE GGML_SOURCE_REV OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE GGML_GIT_RESULT)
if(NOT SDCPP_GIT_RESULT EQUAL 0 OR NOT GGML_GIT_RESULT EQUAL 0)
    message(FATAL_ERROR "Both pinned diffusion and ggml submodules need valid git HEADs")
endif()
file(SHA256 "${CMAKE_CURRENT_SOURCE_DIR}/stable-diffusion.cpp.patch" SDCPP_PATCH_SHA)
file(SHA256 "${CMAKE_CURRENT_SOURCE_DIR}/ggml.patch" GGML_PATCH_SHA)
set(SDCPP_STAGE_ID "${SDCPP_SOURCE_REV}-${GGML_SOURCE_REV}-${SDCPP_PATCH_SHA}-${GGML_PATCH_SHA}")
set(SDCPP_STAGE_STAMP "${SDCPP_DIR}/.localdream-staged")
set(SDCPP_OLD_STAGE_ID "")
if(EXISTS "${SDCPP_STAGE_STAMP}")
    file(READ "${SDCPP_STAGE_STAMP}" SDCPP_OLD_STAGE_ID)
endif()
if(NOT "${SDCPP_OLD_STAGE_ID}" STREQUAL "${SDCPP_STAGE_ID}")
    file(REMOVE_RECURSE "${SDCPP_DIR}")
    file(COPY "${SDCPP_SOURCE_DIR}/" DESTINATION "${SDCPP_DIR}"
        PATTERN ".git" EXCLUDE PATTERN "build" EXCLUDE
        PATTERN ".venv" EXCLUDE PATTERN "__pycache__" EXCLUDE)
    # git apply also works outside a Git working tree; neither patch can
    # write back into the original, pinned SDK or upstream checkout.
    function(localdream_stage_patch patch_file work_dir)
        execute_process(COMMAND git apply --reverse --check "${patch_file}"
            WORKING_DIRECTORY "${work_dir}"
            RESULT_VARIABLE already_applied OUTPUT_QUIET ERROR_QUIET)
        if(NOT already_applied EQUAL 0)
            execute_process(COMMAND git apply --check "${patch_file}"
                WORKING_DIRECTORY "${work_dir}"
                RESULT_VARIABLE valid
                ERROR_VARIABLE patch_error)
            if(NOT valid EQUAL 0)
                message(FATAL_ERROR "${patch_file}: ${patch_error}")
            endif()
            execute_process(COMMAND git apply "${patch_file}"
                WORKING_DIRECTORY "${work_dir}"
                RESULT_VARIABLE applied ERROR_VARIABLE patch_error)
            if(NOT applied EQUAL 0)
                message(FATAL_ERROR "${patch_file}: ${patch_error}")
            endif()
        endif()
    endfunction()
    localdream_stage_patch("${CMAKE_CURRENT_SOURCE_DIR}/stable-diffusion.cpp.patch" "${SDCPP_DIR}")
    localdream_stage_patch("${CMAKE_CURRENT_SOURCE_DIR}/ggml.patch" "${SDCPP_DIR}/ggml")
    file(WRITE "${SDCPP_STAGE_STAMP}" "${SDCPP_STAGE_ID}")
endif()

set(GGML_HEXAGON ON CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_LLAMAFILE OFF CACHE BOOL "" FORCE)
set(SD_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(SD_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
# The skels are cross-compiled for the DSP by the Hexagon toolchain; keep the
# prebuilt FastRPC libraries for the Android host side.
set(PREBUILT_LIB_DIR "android_aarch64" CACHE STRING "" FORCE)

add_subdirectory("${SDCPP_DIR}" sdcpp EXCLUDE_FROM_ALL)

add_library(dit_engine SHARED DitEngine.cpp)

target_include_directories(dit_engine PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../include"
    "${SDCPP_DIR}/include"
)

target_link_libraries(dit_engine PRIVATE stable-diffusion log)

# One exported symbol; everything else stays internal so the core cannot drift
# into calling stable-diffusion.cpp directly. -fvisibility only covers this
# file's own objects, so the version script is what actually hides the symbols
# pulled in from libstable-diffusion.a and ggml.
target_compile_options(dit_engine PRIVATE -fvisibility=hidden)
target_link_options(dit_engine PRIVATE
    "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/dit_engine.map"
    "-Wl,-s"
)

set_target_properties(dit_engine PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib/${CMAKE_ANDROID_ARCH_ABI}"
)

# The DSP-side skels are loaded by FastRPC at run time, so nothing links them
# and EXCLUDE_FROM_ALL would otherwise leave them unbuilt. They ship in the
# same package as the engine, so tie them to this target. The Android catalog
# starts at SM8750 (HTP v79), so older skels would only add APK size.
foreach(htp_version v79 v81 v85)
    if(TARGET htp-${htp_version})
        add_dependencies(dit_engine htp-${htp_version})
    endif()
endforeach()
