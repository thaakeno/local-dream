#!/usr/bin/env python3
"""Use ggml's existing synchronized tensor copy on MOSS virtual HTP sessions.

This is intentionally a narrow, pinned-source patch. The crash in alpha.74
occurred in clone_buffer() shared_ptr refcount on a cross-session async copy.
No change is made to the external shared upstream submodule or other engines.
"""
from pathlib import Path
import sys

if len(sys.argv) != 2:
    raise SystemExit("usage: fix_htp_virtual_copy.py STAGED_SOURCE")
path = Path(sys.argv[1]) / "ggml/src/ggml-hexagon/ggml-hexagon.cpp"
source = path.read_text()
needle = """static bool ggml_backend_hexagon_cpy_tensor_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, const ggml_tensor * src, ggml_tensor * dst) {
    if (!ggml_backend_is_hexagon(backend_src) || !ggml_backend_is_hexagon(backend_dst)) {
        return false;
    }

    // FIXME: ggml-meta needs to call init_tensor on auxiliary tensors"""
replacement = """static bool ggml_backend_hexagon_cpy_tensor_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, const ggml_tensor * src, ggml_tensor * dst) {
    if (!ggml_backend_is_hexagon(backend_src) || !ggml_backend_is_hexagon(backend_dst)) {
        return false;
    }

    // MOSS SM8850: different virtual FastRPC sessions on the same physical HTP
    // must not execute the unsafe cross-session async copy path, which calls
    // clone_buffer() on a stale shared-buffer context and SIGSEGVs in shared_ptr
    // refcounting. Let upstream's scheduler synchronize both backends and use
    // ggml_backend_tensor_copy() for a regular host-memory transfer instead.
    // This is DATA MOVEMENT ONLY; all graph ops remain Hexagon NPU-only.
    auto * async_src_sess = static_cast<ggml_hexagon_session *>(backend_src->context);
    auto * async_dst_sess = static_cast<ggml_hexagon_session *>(backend_dst->context);
    if (async_src_sess != async_dst_sess &&
        async_src_sess->phys_idx == async_dst_sess->phys_idx) {
        static std::atomic_flag noted = ATOMIC_FLAG_INIT;
        if (!noted.test_and_set()) {
            GGML_LOG_INFO("[MOSS_HTP_COPY] using synchronized tensor transfers between virtual Hexagon sessions\\n");
        }
        return false;
    }

    // FIXME: ggml-meta needs to call init_tensor on auxiliary tensors"""
if source.count(needle) != 1:
    raise SystemExit("Pinned ggml async-copy entry point changed; refuse unsafe patch")
if "#include <atomic>" not in source:
    replacement_source=source.replace(needle,replacement,1)
    # std::atomic_flag needs its own explicit header, independently of
    # transitive includes; preserve original file's include order.
    if source.count("#include <algorithm>")==1:
        replacement_source=replacement_source.replace("#include <algorithm>", "#include <algorithm>\n#include <atomic>",1)
    else:
        raise SystemExit("Unable to insert <atomic> in pinned ggml header block")
else:
    replacement_source=source.replace(needle,replacement,1)
assert replacement_source.count("[MOSS_HTP_COPY]")==1
assert replacement_source.find("[MOSS_HTP_COPY]")<replacement_source.find("ggml_hexagon_cpy_tensor_async_virt(backend_src, backend_dst, src, dst);")
path.write_text(replacement_source)
print("MOSS-only virtual-session sync-copy guard: PASS (pinned upstream validated)")
