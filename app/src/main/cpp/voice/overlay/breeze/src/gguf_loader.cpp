#include "breeze/gguf_loader.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <unordered_map>

namespace breeze {

static bool is_decoder_codebook(const char * name, const ggml_tensor * t) {
    if (!name || !t || t->type != GGML_TYPE_F32) return false;
    const std::string n(name);
    return n.rfind("codec.dq.", 0) == 0 &&
           n.size() >= 6 &&
           n.compare(n.size() - 6, 6, ".embed") == 0;
}

static bool validate_f32_blob(const char * name, const std::vector<uint8_t> & raw) {
    if ((raw.size() % sizeof(float)) != 0) {
        std::fprintf(stderr, "[BREEZE_MODEL] invalid F32 byte count for %s: %zu\n", name, raw.size());
        return false;
    }
    for (size_t i = 0; i < raw.size() / sizeof(float); ++i) {
        float v = 0.0f;
        std::memcpy(&v, raw.data() + i * sizeof(float), sizeof(float));
        if (!std::isfinite(v)) {
            std::fprintf(stderr, "[BREEZE_MODEL] non-finite GGUF codebook value name=%s index=%zu\n", name, i);
            return false;
        }
    }
    return true;
}

static bool verify_decoder_codebook_mirrors(
    GGUFModel & model,
    Backend & be,
    const std::unordered_map<std::string, std::vector<float>> & expected,
    const std::vector<int32_t> & row_ids
) {
    float global_worst = 0.0f;
    size_t checked = 0;

    for (const auto & it : expected) {
        ggml_tensor * book = model.find(it.first);
        if (!book) {
            std::fprintf(stderr, "[BREEZE_MODEL] missing mirrored codebook %s\n", it.first.c_str());
            return false;
        }

        Graph g(128);
        auto * ids = g.input_i32(row_ids, (int) row_ids.size());
        auto * out = ggml_get_rows(g.ctx, book, ids);
        g.compute(be, out);
        const std::vector<float> got = tensor_to_f32(out);

        if (got.size() != it.second.size()) {
            std::fprintf(
                stderr,
                "[BREEZE_MODEL] mirrored codebook size mismatch name=%s got=%zu expected=%zu\n",
                it.first.c_str(), got.size(), it.second.size()
            );
            return false;
        }

        for (size_t i = 0; i < got.size(); ++i) {
            if (!std::isfinite(got[i])) {
                std::fprintf(
                    stderr,
                    "[BREEZE_MODEL] mirrored HTP codebook non-finite name=%s index=%zu\n",
                    it.first.c_str(), i
                );
                return false;
            }
            global_worst = std::max(global_worst, std::fabs(got[i] - it.second[i]));
        }
        checked++;
    }

    if (global_worst > 1.0e-6f) {
        std::fprintf(stderr, "[BREEZE_MODEL] mirrored HTP codebook mismatch worst=%.8g\n", global_worst);
        return false;
    }

    std::fprintf(
        stderr,
        "[BREEZE_MODEL] ordinary HTP codebook mirrors verified count=%zu rows=%zu width=256 worst=%.8g\n",
        checked, row_ids.size(), global_worst
    );
    return true;
}

#ifdef _WIN32
#define breeze_fseek _fseeki64
#else
#define breeze_fseek fseeko
#endif

bool GGUFModel::load(const std::string & path, Backend & be) {
    gguf_init_params gp{ /*no_alloc=*/true, /*ctx=*/&meta };
    gguf = gguf_init_from_file(path.c_str(), gp);
    if (!gguf) return false;

    // Keep the primary model allocation exactly like the device-proven v153
    // loader. In particular, DO NOT mark the full aggregate allocation as
    // GGML_BACKEND_BUFFER_USAGE_WEIGHTS: doing that changes the Hexagon mapping
    // and repack policy for the whole model.
    buffer = ggml_backend_alloc_ctx_tensors(meta, be.backend);
    if (!buffer) {
        free();
        return false;
    }

    const int64_t n = gguf_get_n_tensors(gguf);
    size_t codebook_count = 0;
    for (int64_t i = 0; i < n; ++i) {
        const char * name = gguf_get_tensor_name(gguf, i);
        ggml_tensor * t = ggml_get_tensor(meta, name);
        if (is_decoder_codebook(name, t)) codebook_count++;
    }
    if (codebook_count == 0) {
        std::fprintf(stderr, "[BREEZE_MODEL] no decoder codebooks found\n");
        free();
        return false;
    }

    // The v153 graph completed end-to-end on SM8850, but its decoder F32
    // codebooks lived inside the large aggregate model mapping. Mirror only
    // those small tables into a dedicated ordinary HTP buffer. This preserves
    // v153 scheduling/kernels for every op while keeping codebook GET_ROWS off
    // the fragile large-buffer/weight mapping path.
    ggml_init_params cp{
        ggml_tensor_overhead() * (codebook_count + 8) + 4096,
        nullptr,
        true,
    };
    codebook_meta = ggml_init(cp);
    if (!codebook_meta) {
        std::fprintf(stderr, "[BREEZE_MODEL] failed to allocate codebook metadata context\n");
        free();
        return false;
    }

    std::unordered_map<std::string, ggml_tensor *> mirrors;
    mirrors.reserve(codebook_count);
    for (int64_t i = 0; i < n; ++i) {
        const char * name = gguf_get_tensor_name(gguf, i);
        ggml_tensor * src = ggml_get_tensor(meta, name);
        if (!is_decoder_codebook(name, src)) continue;

        ggml_tensor * dst = ggml_new_tensor_4d(
            codebook_meta,
            src->type,
            src->ne[0],
            src->ne[1],
            src->ne[2],
            src->ne[3]
        );
        ggml_set_name(dst, name);
        mirrors.emplace(name, dst);
    }

    codebook_buffer = ggml_backend_alloc_ctx_tensors(codebook_meta, be.backend);
    if (!codebook_buffer) {
        std::fprintf(stderr, "[BREEZE_MODEL] failed to allocate ordinary HTP codebook mirror buffer\n");
        free();
        return false;
    }

    FILE * f = fopen(path.c_str(), "rb");
    if (!f) {
        free();
        return false;
    }

    const size_t data_off = gguf_get_data_offset(gguf);
    std::vector<uint8_t> buf;
    const std::vector<int32_t> verify_rows = { 31, 219, 1221, 1938, 2047 };
    std::unordered_map<std::string, std::vector<float>> expected;
    expected.reserve(codebook_count);

    size_t mirrored_codebooks = 0;
    for (int64_t i = 0; i < n; i++) {
        const char * name = gguf_get_tensor_name(gguf, i);
        ggml_tensor * t = ggml_get_tensor(meta, name);
        const size_t off = data_off + gguf_get_tensor_offset(gguf, i);
        const size_t sz = ggml_nbytes(t);
        buf.resize(sz);

        if (breeze_fseek(f, (long long) off, SEEK_SET) != 0 ||
            fread(buf.data(), 1, sz, f) != sz) {
            fclose(f);
            free();
            return false;
        }

        // Exact v153 upload for the primary model tensor.
        ggml_backend_tensor_set(t, buf.data(), 0, sz);

        auto mit = mirrors.find(name);
        if (mit == mirrors.end()) {
            tensors[name] = t;
            continue;
        }

        if (!validate_f32_blob(name, buf)) {
            fclose(f);
            free();
            return false;
        }
        if (t->ne[0] != 256 || t->ne[1] < 2048) {
            std::fprintf(
                stderr,
                "[BREEZE_MODEL] unexpected decoder codebook shape name=%s shape=%lldx%lld\n",
                name, (long long) t->ne[0], (long long) t->ne[1]
            );
            fclose(f);
            free();
            return false;
        }

        ggml_tensor * mirror = mit->second;
        ggml_backend_tensor_set(mirror, buf.data(), 0, sz);
        tensors[name] = mirror;
        mirrored_codebooks++;

        std::vector<float> rows((size_t) t->ne[0] * verify_rows.size());
        for (size_t r = 0; r < verify_rows.size(); ++r) {
            const size_t src_off = (size_t) verify_rows[r] * t->nb[1];
            const size_t row_bytes = (size_t) t->ne[0] * sizeof(float);
            if (src_off + row_bytes > buf.size()) {
                std::fprintf(stderr, "[BREEZE_MODEL] codebook verification row out of bounds name=%s\n", name);
                fclose(f);
                free();
                return false;
            }
            std::memcpy(rows.data() + r * (size_t) t->ne[0], buf.data() + src_off, row_bytes);
        }
        expected.emplace(name, std::move(rows));
    }
    fclose(f);

    if (mirrored_codebooks != codebook_count || expected.size() != codebook_count) {
        std::fprintf(
            stderr,
            "[BREEZE_MODEL] incomplete codebook mirroring mirrored=%zu expected=%zu\n",
            mirrored_codebooks, codebook_count
        );
        free();
        return false;
    }

    std::fprintf(
        stderr,
        "[BREEZE_MODEL] mirrored %zu decoder codebooks into dedicated ordinary HTP buffer; primary model map remains exact-v153\n",
        mirrored_codebooks
    );

    if (!verify_decoder_codebook_mirrors(*this, be, expected, verify_rows)) {
        free();
        return false;
    }

    return true;
}

void GGUFModel::free() {
    tensors.clear();

    if (codebook_buffer) ggml_backend_buffer_free(codebook_buffer);
    if (codebook_meta) ggml_free(codebook_meta);
    codebook_buffer = nullptr;
    codebook_meta = nullptr;

    if (buffer) ggml_backend_buffer_free(buffer);
    if (meta) ggml_free(meta);
    if (gguf) gguf_free(gguf);
    buffer = nullptr;
    meta = nullptr;
    gguf = nullptr;
}

ggml_tensor * GGUFModel::find(const std::string & name) const {
    auto it = tensors.find(name);
    return it == tensors.end() ? nullptr : it->second;
}

ggml_tensor * GGUFModel::get(const std::string & name) const {
    ggml_tensor * t = find(name);
    if (!t) throw std::runtime_error("missing tensor: " + name);
    return t;
}

bool GGUFModel::has(const std::string & name) const {
    return tensors.count(name) > 0;
}

int GGUFModel::kv_u32(const char * key, int def) const {
    const int64_t id = gguf_find_key(gguf, key);
    if (id < 0) return def;
    switch (gguf_get_kv_type(gguf, id)) {
        case GGUF_TYPE_UINT32: return (int) gguf_get_val_u32(gguf, id);
        case GGUF_TYPE_INT32:  return (int) gguf_get_val_i32(gguf, id);
        case GGUF_TYPE_UINT64: return (int) gguf_get_val_u64(gguf, id);
        case GGUF_TYPE_INT64:  return (int) gguf_get_val_i64(gguf, id);
        default: return def;
    }
}

float GGUFModel::kv_f32(const char * key, float def) const {
    const int64_t id = gguf_find_key(gguf, key);
    if (id < 0) return def;
    const gguf_type t = gguf_get_kv_type(gguf, id);
    if (t == GGUF_TYPE_FLOAT32) return gguf_get_val_f32(gguf, id);
    if (t == GGUF_TYPE_FLOAT64) return (float) gguf_get_val_f64(gguf, id);
    return def;
}

bool GGUFModel::kv_bool(const char * key, bool def) const {
    const int64_t id = gguf_find_key(gguf, key);
    if (id < 0) return def;
    if (gguf_get_kv_type(gguf, id) == GGUF_TYPE_BOOL) return gguf_get_val_bool(gguf, id);
    return def;
}

std::string GGUFModel::kv_str(const char * key, const std::string & def) const {
    const int64_t id = gguf_find_key(gguf, key);
    if (id < 0 || gguf_get_kv_type(gguf, id) != GGUF_TYPE_STRING) return def;
    return gguf_get_val_str(gguf, id);
}

std::vector<int> GGUFModel::kv_i32_array(const char * key) const {
    std::vector<int> out;
    const int64_t id = gguf_find_key(gguf, key);
    if (id < 0 || gguf_get_kv_type(gguf, id) != GGUF_TYPE_ARRAY) return out;
    const size_t n = gguf_get_arr_n(gguf, id);
    const void * data = gguf_get_arr_data(gguf, id);
    const gguf_type at = gguf_get_arr_type(gguf, id);
    out.resize(n);
    for (size_t i = 0; i < n; i++) {
        if (at == GGUF_TYPE_INT32) out[i] = ((const int32_t *) data)[i];
        else if (at == GGUF_TYPE_UINT32) out[i] = (int) ((const uint32_t *) data)[i];
    }
    return out;
}

std::vector<std::string> GGUFModel::kv_str_array(const char * key) const {
    std::vector<std::string> out;
    const int64_t id = gguf_find_key(gguf, key);
    if (id < 0 || gguf_get_kv_type(gguf, id) != GGUF_TYPE_ARRAY) return out;
    const size_t n = gguf_get_arr_n(gguf, id);
    out.reserve(n);
    for (size_t i = 0; i < n; i++) out.emplace_back(gguf_get_arr_str(gguf, id, i));
    return out;
}

}
