#pragma once

#include <cstdio>
#include <cstdint>
#include <dlfcn.h>
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

        // Keep Qualcomm's upstream backend intact. The working DSPQueue
        // implementation enables unsigned modules for the resolved CDSP
        // domain specifically, while the mempool implementation uses -1
        // (all domains). Establish the known-good domain policy before the
        // FastRPC IDL stub tries to open the v81 skel.
        probeDomain3();
    }

private:
    static void probeDomain3() {
        using SessionControl = int (*)(uint32_t, void *, uint32_t);
        using Open = int (*)(const char *, uint64_t *);
        using Close = int (*)(uint64_t);
        struct UnsignedModule { int domain; int enable; };

        void * library = ::dlopen("libcdsprpc.so", RTLD_NOW | RTLD_LOCAL);
        if (!library) {
            std::fprintf(stderr, "[BREEZE_FASTRPC_PROBE] phase=dlopen success=0 reason=%s\n",
                         ::dlerror());
            return;
        }
        auto control = reinterpret_cast<SessionControl>(::dlsym(library, "remote_session_control"));
        auto open = reinterpret_cast<Open>(::dlsym(library, "remote_handle64_open"));
        auto close = reinterpret_cast<Close>(::dlsym(library, "remote_handle64_close"));
        if (!control || !open || !close) {
            std::fprintf(stderr, "[BREEZE_FASTRPC_PROBE] phase=symbols control=%d open=%d close=%d\n",
                control != nullptr, open != nullptr, close != nullptr);
            ::dlclose(library);
            return;
        }
        UnsignedModule unsignedModule{3, 1};
        const int policy = control(2u, &unsignedModule, sizeof(unsignedModule));
        std::fprintf(stderr, "[BREEZE_FASTRPC_PROBE] phase=unsigned-domain3 result=0x%x\n", policy);
        // The probe uses the pinned mempool IDL ABI (ggml_htp / 0.0.2).
        // Diagnostic only: no graph execution, resource allocations, or
        // changes to the working DSPQueue backend.
        static constexpr const char * uri =
            "file:///libggml-htp-v81.so?ggml_htp_skel_handle_invoke"
            "&_modver=1.0&_idlver=0.0.2&_dom=cdsp&_session=0";
        uint64_t handle = 0;
        const int result = open(uri, &handle);
        std::fprintf(stderr,
            "[BREEZE_FASTRPC_PROBE] phase=skel-open domain=3 result=0x%x handle=%d\n",
            result, handle != 0);
        if (result == 0 && handle != 0) {
            const int closed = close(handle);
            std::fprintf(stderr, "[BREEZE_FASTRPC_PROBE] phase=skel-close result=0x%x\n", closed);
        }
        ::dlclose(library);
    }
};
}
