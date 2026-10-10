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
        // The pinned upstream registry resets BOTH path variables to its
        // desktop/test location. Recover the app's original library search
        // paths after registry construction, before any CDSP session starts.
        // This avoids modifying the Qualcomm SDK or the upstream backend.
        const char * hostLd = std::getenv("BREEZE_FASTRPC_HOST_LD_PATH");
        const char * hostAdsp = std::getenv("BREEZE_FASTRPC_HOST_ADSP_PATH");
        if (!hostLd || !*hostLd || !hostAdsp || !*hostAdsp) {
            throw std::runtime_error("Missing FastRPC app-private loader path configuration");
        }
        const std::string adsp = std::string(dir) + ";" + hostAdsp;
        if (::setenv("ADSP_LIBRARY_PATH", adsp.c_str(), 1) != 0 ||
            ::setenv("LD_LIBRARY_PATH", hostLd, 1) != 0) {
            throw std::runtime_error("FastRPC app-private loader path restoration failed");
        }
        std::fprintf(stderr,
            "[BREEZE_TRANSPORT] phase=loader-ready backend=fastrpc-ion-mempool "
            "skel=%s adsp_restored=1 ld_restored=1\n", skel.c_str());
    }
};
}
