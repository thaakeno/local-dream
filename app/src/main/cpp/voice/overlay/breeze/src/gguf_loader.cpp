#include "breeze/gguf_loader.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace breeze {

static bool is_decoder_codebook(const char * name, const ggml_tensor * t) {
    if (!name || !t || t->type != GGML_TYPE_F32) return false;
    const std::string n(name);
    return n.rfind("codec.dq.", 0) == 0 &&
           n.size() >= 6 &&
           n.compare(n.size() - 6, 6, ".embed") == 0;
}

static bool validate_f32_blob(
    const char * name,
    const std::vector<uint8_t> & raw
) {
    if ((raw.size() % sizeof(float)) != 0) {
        std::fprintf(
            stderr,
            "[BREEZE_MODEL] invalid F32 byte count for %s: %zu\n",
            name,
            raw.size()
        );
        return false;
    }
    for (size_t i = 0; i < raw.size() / sizeof(float); ++i) {
        float v = 0.0f;
        std::memcpy(&v, raw.data() + i * sizeof(float), sizeof(float));
        if (!std::isfinite(v)) {
            std::fprintf(
                stderr,
                "[BREEZE_MODEL] non-finite GGUF codebook value name=%s index=%zu\n",
                name,
                i
            );
            return false;
        }
    }
    return true;
}

static bool verify_decoder_codebook_on_backend(GGUFModel & model, Backend & be) {
    ggml_tensor * book = model.find("codec.dq.first.0.embed");
    if (!book) {
        std::fprintf(stderr, "[BREEZE_MODEL] missing decoder codebook verification tensor\n");
        return false;
    }
    if (book->type != GGML_TYPE_F32 || book->ne[0] != 256 || book->ne[1] < 2048) {
        std::fprintf(
            stderr,
            "[BREEZE_MODEL] unexpected decoder codebook layout type=%s shape=%lldx%lld\n",
            ggml_type_name(book->type),
            (long long) book->ne[0],
            (long long) book->ne[1]
        );
        return false;
    }

    const std::vector<int32_t> row_ids = { 31, 219, 1221, 1938, 2047 };
    const size_t width = (size_t) book->ne[0];
    std::vector<float> expected(width * row_ids.size());
    for (size_t r = 0; r < row_ids.size(); ++r) {
        ggml_backend_tensor_get(
            book,
            expected.data() + r * width,
            (size_t) row_ids[r] * book->nb[1],
            width * sizeof(float)
        );
    }
    for (size_t i = 0; i < expected.size(); ++i) {
        if (!std::isfinite(expected[i])) {
            std::fprintf(
                stderr,
                "[BREEZE_MODEL] uploaded codebook rows are non-finite index=%zu\n",
                i
            );
            return false;
        }
    }

    Graph g(128);
    auto * ids = g.input_i32(row_ids, (int) row_ids.size());
    auto * out = ggml_get_rows(g.ctx, book, ids);
    g.compute(be, out);
    const std::vector<float> got = tensor_to_f32(out);

    if (got.size() != expected.size()) {
        std::fprintf(
            stderr,
            "[BREEZE_MODEL] HTP codebook verification size mismatch got=%zu expected=%zu\n",
            got.size(),
            expected.size()
        );
        return false;
    }

    float worst = 0.0f;
    size_t worst_i = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        if (!std::isfinite(got[i])) {
            std::fprintf(
                stderr,
                "[BREEZE_MODEL] HTP codebook verification non-finite index=%zu\n",
                i
            );
            return false;
        }
        const float err = std::fabs(got[i] - expected[i]);
        if (err > worst) {
            worst = err;
            worst_i = i;
        }
    }
    if (worst > 1.0e-6f) {
        std::fprintf(
            stderr,
            "[BREEZE_MODEL] HTP codebook verification mismatch index=%zu err=%.8g\n",
            worst_i,
            worst
        );
        return false;
    }

    std::fprintf(
        stderr,
        "[BREEZE_MODEL] HTP decoder codebook rows verified count=%zu width=%zu worst=%.8g\n",
        row_ids.size(),
        width,
        worst
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

    buffer = ggml_backend_alloc_ctx_tensors(meta, be.backend);
    if (!buffer) return false;

    // This entire allocation is immutable model data. Mark it as WEIGHTS
    // before the first upload. The Hexagon backend uses this bit to select its
    // v81 extended read-only mapping for large model buffers and to tag/repack
    // quantized tensors correctly.
    ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    FILE * f = fopen(path.c_str(), "rb");
    if (!f) return false;

    const size_t data_off = gguf_get_data_offset(gguf);
    const int64_t n = gguf_get_n_tensors(gguf);
    std::vector<uint8_t> buf;
    size_t verified_codebooks = 0;
    for (int64_t i = 0; i < n; i++) {
        const char * name = gguf_get_tensor_name(gguf, i);
        ggml_tensor * t = ggml_get_tensor(meta, name);
        const size_t off = data_off + gguf_get_tensor_offset(gguf, i);
        const size_t sz = ggml_nbytes(t);
        buf.resize(sz);
        if (breeze_fseek(f, (long long) off, SEEK_SET) != 0) { fclose(f); return false; }
        if (fread(buf.data(), 1, sz, f) != sz) { fclose(f); return false; }
        if (is_decoder_codebook(name, t)) {
            if (!validate_f32_blob(name, buf)) {
                fclose(f);
                return false;
            }
            verified_codebooks++;
        }
        ggml_backend_tensor_set(t, buf.data(), 0, sz);
        tensors[name] = t;
    }
    fclose(f);

    if (verified_codebooks == 0) {
        std::fprintf(stderr, "[BREEZE_MODEL] no decoder codebooks were verified\n");
        return false;
    }
    std::fprintf(
        stderr,
        "[BREEZE_MODEL] verified %zu finite decoder codebooks from GGUF before HTP execution\n",
        verified_codebooks
    );

    // Exercise the real uploaded model weight, not a synthetic compute tensor.
    // This catches large-buffer / extended-map failures before generation.
    if (!verify_decoder_codebook_on_backend(*this, be)) {
        return false;
    }

    return true;
}

void GGUFModel::free() {
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
