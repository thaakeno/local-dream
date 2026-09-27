# Local Dream Android integration for the native YuE2 runtime.
#
# The pinned FastRPC Hexagon backend logs through Android's liblog. Keep that
# platform dependency at the integration boundary instead of rewriting either
# yue2.cpp or ggml-hexagon during the build.
if(ANDROID)
    link_libraries(log)
endif()
