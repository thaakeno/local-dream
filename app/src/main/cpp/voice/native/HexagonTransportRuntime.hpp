#pragma once

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <sys/stat.h>

// Owned integration point: upstream FastRPC resets ADSP_LIBRARY_PATH to
// /data/local/tmp during registry initialization. That directory is unusable
// by normal Android apps. Restore app-private DSP skel loading after
// ggml_backend_hexagon_reg() but before ggml_backend_dev_init().
// Nothing in the Qualcomm SDK or ggml source tree is modified here.
namespace breeze {
class HexagonTransportRuntime final {
public:
    static void prepareAfterRegistry() {
        const char * dir = std::getenv("BREEZE_FASTRPC_SKEL_DIR");
        if (!dir || !*dir) return;
        const std::string skel = std::string(dir) + "/libggml-htp-v81.so";
        struct stat st{};
        if (::stat(skel.c_str(), &st) != 0 || !S_ISREG(st.st_mode) || st.st_size < 65536) {
            throw std::runtime_error("FastRPC v81 skel missing: " + skel);
        }
        const char * existing = std::getenv("ADSP_LIBRARY_PATH");
        const std::string paths = std::string(dir) + ";" +
            (existing ? existing : "/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp;/dsp");
        if (::setenv("ADSP_LIBRARY_PATH", paths.c_str(), 1) != 0) {
            throw std::runtime_error("FastRPC DSP library path setup failed");
        }
        std::fprintf(stderr,
            "[BREEZE_TRANSPORT] backend=fastrpc-ion-mempool skel=%s path_restored=1\n",
            skel.c_str());
    }
};
}
