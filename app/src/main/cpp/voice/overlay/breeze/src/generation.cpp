#include "breeze/generation.h"
#include "breeze/sampling.h"
#include "breeze/text_encoder.h"
#include "breeze/qnn_depth_decoder.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <stdexcept>

namespace breeze {

struct Seg {
    bool is_text;
    std::vector<int> tokens;
    std::vector<int> codes; // frame-major, for audio segments
    int n_frames = 0;
    bool eos = false;
};

static Seg text_seg(BreezeModel & m, const std::string & s) {
    Seg seg;
    seg.is_text = true;
    seg.tokens = m.tok.encode(s, true);
    return seg;
}

static std::vector<Seg> build_segments(BreezeModel & m, const GenRequest & r, const std::string & text,
                                       bool has_ref, const std::string & ref_text,
                                       const std::vector<int> & ref_codes, int ref_T, bool cond) {
    std::vector<Seg> segs;
    const std::string spk = "[S0]";
    if (has_ref) {
        segs.push_back(text_seg(m, spk + ref_text));
        Seg a;
        a.is_text = false;
        a.codes = ref_codes;
        a.n_frames = ref_T;
        a.eos = true;
        segs.push_back(a);
    }
    std::string tail = cond ? spk + "<ins_bos>" + r.instruction + "<ins_eos>" + text : spk + text;
    segs.push_back(text_seg(m, tail));
    return segs;
}

static std::vector<float> assemble(BreezeModel & m, const std::vector<Seg> & segs, int & total) {
    const int H = m.cfg.hidden_size;
    std::vector<float> out;
    total = 0;
    for (const Seg & s : segs) {
        if (s.is_text) {
            std::vector<float> e = text_encoder_forward(m, s.tokens);
            out.insert(out.end(), e.begin(), e.end());
            total += (int) s.tokens.size();
        } else {
            std::vector<float> e = audio_embed_forward(m, s.codes, s.n_frames);
            out.insert(out.end(), e.begin(), e.end());
            total += s.n_frames;
            std::vector<int> eos_frame(m.cfg.num_codebooks, m.cfg.codebook_eos_token_id);
            std::vector<float> ee = audio_embed_forward(m, eos_frame, 1);
            out.insert(out.end(), ee.begin(), ee.end());
            total += 1;
        }
    }
    (void) H;
    return out;
}

static std::vector<float> combine_logits(const std::vector<float> & cond, const std::vector<float> & unc,
                                         bool use_cfg, float scale) {
    if (!use_cfg) return cond;
    std::vector<float> out(cond.size());
    for (size_t i = 0; i < out.size(); i++) out[i] = unc[i] + scale * (cond[i] - unc[i]);
    return out;
}

// what a finished piece leaves behind so the next one can keep the same voice
struct ChunkRef {
    std::vector<int> codes;
    int n_frames = 0;
    std::string text;
};

struct AudioStats {
    size_t samples = 0;
    size_t nonfinite = 0;
    size_t pcm_nonzero = 0;
    double rms = 0.0;
    float peak = 0.0f;
};

static AudioStats audio_stats(const std::vector<float> & audio) {
    AudioStats st;
    st.samples = audio.size();
    double sumsq = 0.0;
    for (float v : audio) {
        if (!std::isfinite(v)) {
            st.nonfinite++;
            continue;
        }
        const float a = std::fabs(v);
        st.peak = std::max(st.peak, a);
        sumsq += (double) v * (double) v;
        if (a * 32767.0f >= 1.0f) st.pcm_nonzero++;
    }
    const size_t finite = st.samples - st.nonfinite;
    if (finite > 0) st.rms = std::sqrt(sumsq / (double) finite);
    return st;
}

// Use the upstream Breeze vocoder graph for correctness. The custom stateful
// decode_stream path was fast enough to finish, but it was never an upstream
// path and on SM8850 it produced a WAV whose samples quantized to silence.
// For short clips we decode once. Long clips use the same 40-frame,
// left-context windowing strategy as upstream voice conversion.
static std::vector<float> decode_reference_audio(
    BreezeModel & m,
    MimiCodec & codec,
    const std::vector<int> & codes,
    int n_frames
) {
    const int nc = m.cfg.num_codebooks;
    const int spf = m.cfg.samples_per_frame;
    if (n_frames <= 0 || codes.size() != (size_t) n_frames * (size_t) nc) return {};

    constexpr int step = 40;
    if (n_frames <= step) {
        return codec.decode(codes, n_frames);
    }

    const int ctx = m.cfg.voc.sliding_window + 16;
    std::vector<float> audio;
    audio.reserve((size_t) n_frames * (size_t) spf);

    for (int start = 0; start < n_frames; start += step) {
        const int count = std::min(step, n_frames - start);
        const int ctx_start = start > ctx ? start - ctx : 0;
        const int sub_frames = start + count - ctx_start;
        std::vector<int> sub(
            codes.begin() + (size_t) ctx_start * (size_t) nc,
            codes.begin() + (size_t) (start + count) * (size_t) nc
        );
        std::vector<float> part = codec.decode(sub, sub_frames);
        const size_t skip = (size_t) (start - ctx_start) * (size_t) spf;
        const size_t want = (size_t) count * (size_t) spf;
        if (part.size() < skip + want) {
            throw std::runtime_error("Breeze reference vocoder returned a short PCM window");
        }
        audio.insert(audio.end(), part.begin() + skip, part.begin() + skip + want);
    }
    return audio;
}

static bool generate_chunk(BreezeModel & m, MimiCodec & codec, const GenRequest & req,
                           const std::string & text, const ChunkRef & ref, uint32_t seed,
                           const AudioCallback & cb, GenTimings & tm,
                           std::chrono::steady_clock::time_point t_start, ChunkRef & out) {
    const auto clock_now = [] { return std::chrono::steady_clock::now(); };
    const auto since = [](std::chrono::steady_clock::time_point t) {
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
    };

    std::mt19937 rng(seed);
    const int nc = m.cfg.num_codebooks;
    const int spf = m.cfg.samples_per_frame;
    const bool has_ref = !ref.codes.empty() && !ref.text.empty();
    const bool use_cfg = req.cfg_scale != 1.0f;

    const std::vector<int> & ref_codes = ref.codes;
    const int ref_T = ref.n_frames;

    auto t0 = clock_now();
    int total_c = 0, total_u = 0;
    std::vector<float> emb_c = assemble(m, build_segments(m, req, text, has_ref, ref.text, ref_codes, ref_T, true), total_c);
    std::vector<float> emb_u;
    if (use_cfg) emb_u = assemble(m, build_segments(m, req, text, has_ref, ref.text, ref_codes, ref_T, false), total_u);
    tm.prompt += since(t0);

    const int configured_max =
        req.max_new_tokens > 0 ? req.max_new_tokens : m.cfg.max_new_tokens;
    const double estimated_seconds = std::max(0.8, estimate_seconds(text));
    const double frames_per_second =
        (double) m.cfg.sample_rate / (double) m.cfg.samples_per_frame;
    const int estimated_frames =
        (int) (estimated_seconds * frames_per_second + 0.999);

    // The text estimate is useful for UI/progress, not as a stopping rule.
    // Breeze decides when speech is complete via EOS. The previous adaptive
    // cap could chop slow/expressive delivery before the model emitted EOS.
    const int adaptive_slack = std::max(12, estimated_frames / 2);
    const int soft_target = std::min(
        configured_max,
        std::max(32, estimated_frames + adaptive_slack)
    );
    const int max_new = configured_max;
    std::fprintf(
        stderr,
        "[BREEZE_LIMIT] estimate=%.2fs estimated_frames=%d configured=%d soft=%d hard=%d eos_first=1\n",
        estimated_seconds,
        estimated_frames,
        configured_max,
        soft_target,
        max_new
    );

    BackboneState st_c, st_u;
    st_c.init(m, total_c + max_new + 8);
    if (use_cfg) st_u.init(m, total_u + max_new + 8);

    t0 = clock_now();
    StepOut o_c = backbone_run(m, st_c, emb_c, total_c);
    StepOut o_u;
    if (use_cfg) o_u = backbone_run(m, st_u, emb_u, total_u);
    tm.prefill += since(t0);

    DepthRunner depth;
    depth.init(m, use_cfg ? 2 : 1);

    // Optional device-specific QNN depth decoder. V1 is batch-1, so CFG keeps
    // the proven ggml path. A failed QNN load/run disables itself and falls
    // back per frame without changing synthesis semantics.
    QnnDepthRunner qnn_depth;
    const bool qnn_depth_requested = !use_cfg;
    const bool qnn_depth_ready = qnn_depth_requested && qnn_depth.init(m);
    std::fprintf(
        stderr,
        "[BREEZE_DEPTH] backend=%s cfg=%.2f\n",
        qnn_depth_ready ? "qnn-htp" : "ggml-hexagon",
        req.cfg_scale
    );

    // This one-frame graph is shape-stable. Replaying it avoids rebuilding and
    // repartitioning an HTP graph for every generated codec frame.
    AudioEmbedRunner audio_embed;
    audio_embed.init(m);

    SampleParams bp;
    bp.temperature = req.temperature > 0.0f ? req.temperature : m.cfg.temperature;
    bp.top_k = req.top_k > 0 ? req.top_k : m.cfg.top_k;
    bp.top_p = req.top_p > 0.0f ? req.top_p : m.cfg.top_p;
    bp.repetition_penalty = req.repetition_penalty > 0.0f ? req.repetition_penalty : m.cfg.repetition_penalty;
    std::vector<int> suppress;
    for (int t = m.cfg.codec_codebook_size; t < m.cfg.audio_vocab_size; t++) suppress.push_back(t);

    std::vector<int> hist;
    std::vector<float> comb = combine_logits(o_c.logits, o_u.logits, use_cfg, req.cfg_scale);
    int cb0 = sample_token(comb, bp, rng, &hist, &suppress);

    std::vector<int> frames;
    int emitted = 0;
    int submitted = 0;
    bool stopped = false;
    codec.stream_reset();

    const bool qnn_pipeline = codec.uses_qnn_vocoder();
    // QNN and ggml-hexagon share the same physical HTP. Running them at the
    // same time made a ~2 s QNN graph take ~8 s on SM8850. Serialize HTP work:
    // short lines generate fully then decode once; long lines alternate large
    // generation bursts with 64-frame/39-new-frame vocoder calls.
    const bool qnn_streaming = qnn_pipeline && estimated_frames > 64;
    const int qnn_first_new = 64;
    const int qnn_steady_new = 39;

    const int chunk_max = std::max(1, req.chunk_max);
    int fallback_chunk = std::min(std::max(1, req.chunk_first), chunk_max);

    size_t streamed_samples = 0;
    size_t streamed_nonfinite = 0;
    size_t streamed_pcm_nonzero = 0;
    double streamed_sumsq = 0.0;
    float streamed_peak = 0.0f;

    struct DecodeResult {
        std::vector<float> audio;
        double ms = 0.0;
        int count = 0;
        int start = 0;
    };

    auto consume_audio = [&](DecodeResult result) {
        const size_t want = (size_t) result.count * (size_t) spf;
        if (result.audio.size() != want) {
            throw std::runtime_error("Breeze streaming vocoder returned the wrong PCM chunk length");
        }

        tm.vocoder += result.ms;
        tm.flushes++;
        const AudioStats ast = audio_stats(result.audio);
        streamed_samples += ast.samples;
        streamed_nonfinite += ast.nonfinite;
        streamed_pcm_nonzero += ast.pcm_nonzero;
        streamed_peak = std::max(streamed_peak, ast.peak);
        streamed_sumsq += ast.rms * ast.rms * (double) ast.samples;

        std::fprintf(
            stderr,
            "[BREEZE_VOCODER_STREAM] flush=%d new_frames=%d samples=%zu ms=%.2f "
            "total_ms=%.2f peak=%.6g rms=%.6g qnn=%d scheduler=serialized\n",
            tm.flushes, result.count, result.audio.size(), result.ms,
            tm.vocoder, ast.peak, ast.rms, qnn_pipeline ? 1 : 0
        );

        if (!tm.first_audio) {
            tm.first_vocoder = result.ms;
            tm.first_frames = result.count;
            tm.first_audio = since(t_start);
        }
        if (!cb(result.audio.data(), result.audio.size())) return false;
        emitted += result.count;
        return true;
    };

    int qnn_next_new = qnn_first_new;

    auto run_qnn = [&](int count) {
        if (count <= 0) return true;
        const int start = submitted;
        std::vector<int> sub(
            frames.begin() + (size_t) start * (size_t) nc,
            frames.begin() + (size_t) (start + count) * (size_t) nc
        );
        submitted += count;
        std::fprintf(
            stderr,
            "[BREEZE_SCHEDULER] qnn start=%d frames=%d generated=%d mode=serialized-htp\n",
            start, count, (int) frames.size() / nc
        );
        const auto tv = clock_now();
        DecodeResult result;
        result.audio = codec.decode_stream(sub, count);
        result.ms = since(tv);
        result.count = count;
        result.start = start;
        return consume_audio(std::move(result));
    };

    auto pump_qnn = [&]() {
        if (!qnn_pipeline || !qnn_streaming) return true;
        int have = (int) frames.size() / nc;
        int pending = have - submitted;
        while (pending >= qnn_next_new) {
            if (!run_qnn(qnn_next_new)) return false;
            qnn_next_new = qnn_steady_new;
            have = (int) frames.size() / nc;
            pending = have - submitted;
        }
        return true;
    };

    auto flush_fallback = [&](bool final_flush) {
        const int have = (int) frames.size() / nc;
        while (have - emitted >= fallback_chunk || (final_flush && have > emitted)) {
            const int count = final_flush
                ? std::min(chunk_max, have - emitted)
                : fallback_chunk;
            const int start = emitted;
            std::vector<int> sub(
                frames.begin() + (size_t) start * (size_t) nc,
                frames.begin() + (size_t) (start + count) * (size_t) nc
            );
            const auto tv = clock_now();
            DecodeResult result;
            result.audio = codec.decode_stream(sub, count);
            result.ms = since(tv);
            result.count = count;
            result.start = start;
            if (!consume_audio(std::move(result))) return false;
            fallback_chunk = chunk_max;
        }
        return true;
    };

    int generated_steps = 0;
    for (int step = 0; step < max_new; step++) {
        generated_steps = step + 1;
        if (cb0 == m.cfg.backbone_eos_token_id) break;
        std::vector<std::vector<float>> hiddens = { o_c.hidden };
        if (use_cfg) hiddens.push_back(o_u.hidden);
        auto td = clock_now();
        std::vector<int> depth_codes;
        bool used_qnn_depth = false;
        if (!use_cfg && qnn_depth.ready()) {
            used_qnn_depth = qnn_depth.run(m, o_c.hidden, cb0, rng, depth_codes);
        }
        if (!used_qnn_depth) {
            depth_codes = depth.run(m, hiddens, cb0, req.cfg_scale, rng);
        }
        tm.depth += since(td);
        std::vector<int> frame = { cb0 };
        frame.insert(frame.end(), depth_codes.begin(), depth_codes.end());

        bool pad = true;
        for (int c : frame) if (c != m.cfg.codebook_pad_token_id) { pad = false; break; }
        if (!pad) {
            frames.insert(frames.end(), frame.begin(), frame.end());
            tm.frames++;
            if (qnn_pipeline) {
                if (!pump_qnn()) {
                    stopped = true;
                    break;
                }
            } else if (!flush_fallback(false)) {
                stopped = true;
                break;
            }
        }
        hist.push_back(cb0);

        auto tb = clock_now();
        std::vector<float> ae = audio_embed.run(m, frame);
        if (use_cfg) {
            auto pair = backbone_run_cfg(m, st_c, st_u, ae);
            o_c = std::move(pair[0]);
            o_u = std::move(pair[1]);
        } else {
            o_c = backbone_run(m, st_c, ae, 1);
        }
        tm.backbone += since(tb);
        comb = combine_logits(o_c.logits, o_u.logits, use_cfg, req.cfg_scale);
        cb0 = sample_token(comb, bp, rng, &hist, &suppress);

        if (qnn_pipeline && !pump_qnn()) {
            stopped = true;
            break;
        }

        if (tm.frames == 1 || (tm.frames > 0 && tm.frames % 4 == 0)) {
            const double denom = std::max(1, tm.frames);
            std::fprintf(
                stderr,
                "[BREEZE_STAGE] frames=%d depth_ms_per_frame=%.2f backbone_ms_per_frame=%.2f "
                "vocoder_total_ms=%.2f first_audio_ms=%.2f\n",
                tm.frames,
                tm.depth / denom,
                tm.backbone / denom,
                tm.vocoder,
                tm.first_audio
            );
        }
    }

    const bool reached_eos = cb0 == m.cfg.backbone_eos_token_id;
    const int generated_frames = (int) frames.size() / nc;
    std::fprintf(
        stderr,
        "[BREEZE_GENERATION_DONE] frames=%d eos=%d hard_limit=%d\n",
        generated_frames,
        reached_eos ? 1 : 0,
        max_new
    );
    if (generated_steps >= max_new && !reached_eos) {
        std::fprintf(
            stderr,
            "[BREEZE_LIMIT] hard safety ceiling reached after %d steps; "
            "audio may end at the configured frame cap\n",
            generated_steps
        );
    }

    if (!stopped) {
        if (qnn_pipeline) {
            // No overlapping HTP jobs here: QAIRT explicitly warns that
            // simultaneous graph execution can significantly degrade
            // performance. Drain with the largest valid fixed-graph chunks.
            while (!stopped) {
                const int have = (int) frames.size() / nc;
                const int pending = have - submitted;
                if (pending <= 0) break;
                const int count = submitted == 0
                    ? std::min(qnn_first_new, pending)
                    : std::min(qnn_steady_new, pending);
                if (!run_qnn(count)) stopped = true;
            }
        } else if (!flush_fallback(true)) {
            stopped = true;
        }
    }

    st_c.free();
    if (use_cfg) st_u.free();
    depth.free();

    const int total_frames = (int) frames.size() / nc;
    if (total_frames <= 0) {
        throw std::runtime_error("Breeze generated no codec frames");
    }

    if (!stopped) {
        const size_t expected_samples = (size_t) total_frames * (size_t) spf;
        const double rms = streamed_samples > 0
            ? std::sqrt(streamed_sumsq / (double) streamed_samples)
            : 0.0;
        const double dbfs = rms > 0.0 ? 20.0 * std::log10(rms) : -240.0;
        const size_t min_nonzero = std::max<size_t>(32, streamed_samples / 1000);

        std::fprintf(
            stderr,
            "[BREEZE_AUDIO] path=stateful-stream frames=%d samples=%zu vocoder_ms=%.2f "
            "finite=%zu nonfinite=%zu pcm_nonzero=%zu peak=%.8g rms=%.8g dbfs=%.2f "
            "flushes=%d first_audio_ms=%.2f\n",
            total_frames,
            streamed_samples,
            tm.vocoder,
            streamed_samples - streamed_nonfinite,
            streamed_nonfinite,
            streamed_pcm_nonzero,
            streamed_peak,
            rms,
            dbfs,
            tm.flushes,
            tm.first_audio
        );

        if (streamed_samples != expected_samples) {
            throw std::runtime_error("Breeze streaming vocoder emitted the wrong total PCM length");
        }
        if (streamed_nonfinite != 0) {
            throw std::runtime_error("Breeze streaming vocoder produced non-finite PCM");
        }
        if (streamed_peak < 1.0e-3f || rms < 5.0e-5 ||
            streamed_pcm_nonzero < min_nonzero) {
            throw std::runtime_error(
                "Breeze streaming vocoder produced effectively silent PCM"
            );
        }
    }

    out.codes = std::move(frames);
    out.n_frames = (int) out.codes.size() / nc;
    out.text = text;
    return !stopped;
}

void GenSession::begin(BreezeModel & m, MimiCodec & codec, const GenRequest & req, GenTimings * tm) {
    m_model = &m;
    m_codec = &codec;
    m_req = req;
    m_piece = 0;
    m_start = std::chrono::steady_clock::now();
    m_codes.clear();
    m_frames = 0;
    m_text.clear();

    if (!req.ref_codes.empty() && req.ref_frames > 0 && !req.ref_text.empty()) {
        m_codes = req.ref_codes;
        m_frames = req.ref_frames;
        m_text = req.ref_text;
    } else if (!req.ref_audio.empty() && !req.ref_text.empty()) {
        const auto t0 = std::chrono::steady_clock::now();
        m_codes = codec.encode(req.ref_audio, m_frames);
        m_text = req.ref_text;
        if (tm) tm->encode_ref =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    }
}

bool GenSession::speak(const std::string & text, const AudioCallback & cb, GenTimings * tm) {
    if (!m_model || !m_codec) return false;
    GenTimings sink;
    GenTimings & t = tm ? *tm : sink;

    ChunkRef anchor;
    anchor.codes = m_codes;
    anchor.n_frames = m_frames;
    anchor.text = m_text;

    ChunkRef made;
    const bool ok = generate_chunk(*m_model, *m_codec, m_req, text, anchor,
                                   (uint32_t) m_req.seed + m_piece, cb, t, m_start, made);
    m_piece++;
    // the opening piece stands in as the reference when there was no clip to clone
    if (ok && m_codes.empty() && made.n_frames > 0) {
        m_codes = std::move(made.codes);
        m_frames = made.n_frames;
        m_text = made.text;
    }
    return ok;
}

// long text is generated piece by piece, each one carrying the same reference so the voice does
// not change at the seams
void generate(BreezeModel & m, MimiCodec & codec, const GenRequest & req, const AudioCallback & cb,
              GenTimings * timings) {
    GenTimings sink;
    GenTimings & tm = timings ? *timings : sink;

    GenSession s;
    s.begin(m, codec, req, &tm);

    // a half minute of reference makes the model skip whole sentences of whatever comes next, so
    // when the first piece has to double as the reference it stays near the usual clip length
    const int anchor_chars = 200;
    const std::vector<std::string> parts =
        split_text(req.text, req.split_chars, s.needs_anchor() ? anchor_chars : 0);

    for (const std::string & part : parts)
        if (!s.speak(part, cb, &tm)) return;
}

// keeps the source's semantic codes and rebuilds the acoustic ones in the reference voice. the words
// and their timing survive, the pitch contour does not, it gets replaced by the reference's own
// the backbone goes degenerate with nothing to read, and forcing codes against that state comes out
// mumbled. what the filler says does not matter, only that there is roughly a clip's worth of it
static std::string filler_text(double secs) {
    static const char * lines[] = {
        "This is a recording of ordinary speech made in a quiet room. ",
        "The words themselves do not matter very much at all here. ",
        "It simply carries on for a little while longer than that. ",
        "Nothing in particular is being described at this point. ",
    };
    const size_t want = (size_t) (secs * 17.0) + 16;
    std::string s;
    for (int i = 0; s.size() < want; i++) s += lines[i % 4];
    return s;
}

std::vector<float> convert_voice(BreezeModel & m, MimiCodec & codec, const std::vector<int> & src_codes,
                                 int src_T, const std::vector<float> & ref_audio,
                                 const std::string & ref_text, const ConvertOptions & opt) {
    const int nc = m.cfg.num_codebooks;
    const bool use_cfg = opt.cfg_scale != 1.0f;
    std::mt19937 rng((uint32_t) opt.seed);

    SampleParams sp;
    sp.temperature = opt.temperature;
    sp.top_k = opt.top_k;

    int ref_T = 0;
    std::vector<int> ref_codes;
    if (!opt.ref_codes.empty() && opt.ref_frames > 0) {
        ref_codes = opt.ref_codes;
        ref_T = opt.ref_frames;
    } else {
        ref_codes = codec.encode(ref_audio, ref_T);
    }

    const std::string text =
        opt.src_text.empty()
            ? filler_text(src_T * (double) m.cfg.samples_per_frame / m.cfg.sample_rate)
            : opt.src_text;
    GenRequest req;
    int total_c = 0, total_u = 0;
    std::vector<float> emb_c =
        assemble(m, build_segments(m, req, text, true, ref_text, ref_codes, ref_T, false), total_c);
    // the negative branch drops the reference, so guidance pushes toward the target voice
    std::vector<float> emb_u;
    if (use_cfg) emb_u = assemble(m, build_segments(m, req, text, false, "", {}, 0, false), total_u);

    BackboneState st_c, st_u;
    st_c.init(m, total_c + src_T + 8);
    if (use_cfg) st_u.init(m, total_u + src_T + 8);
    StepOut o_c = backbone_run(m, st_c, emb_c, total_c);
    StepOut o_u;
    if (use_cfg) o_u = backbone_run(m, st_u, emb_u, total_u);

    DepthRunner depth;
    depth.init(m, use_cfg ? 2 : 1);
    AudioEmbedRunner audio_embed;
    audio_embed.init(m);

    std::vector<int> out((size_t) src_T * nc);
    const int keep = opt.keep_acoustic < nc - 1 ? opt.keep_acoustic : nc - 1;
    for (int t = 0; t < src_T; t++) {
        const int cb0 = src_codes[(size_t) t * nc];
        std::vector<std::vector<float>> hiddens = { o_c.hidden };
        if (use_cfg) hiddens.push_back(o_u.hidden);
        std::vector<int> rest =
            depth.run(m, hiddens, cb0, opt.cfg_scale, rng, &sp,
                      keep > 0 ? &src_codes[(size_t) t * nc + 1] : nullptr, keep);
        out[(size_t) t * nc] = cb0;
        for (int c = 1; c < nc; c++) out[(size_t) t * nc + c] = rest[c - 1];

        const int * from = opt.feed_source ? &src_codes[(size_t) t * nc] : &out[(size_t) t * nc];
        std::vector<int> frame(from, from + nc);
        std::vector<float> ae = audio_embed.run(m, frame);
        if (use_cfg) {
            auto pair = backbone_run_cfg(m, st_c, st_u, ae);
            o_c = std::move(pair[0]);
            o_u = std::move(pair[1]);
        } else {
            o_c = backbone_run(m, st_c, ae, 1);
        }
        if (t % 25 == 0) { printf("\rconverting %d/%d frames", t, src_T); fflush(stdout); }
    }
    printf("\rconverted %d frames        \n", src_T);

    st_c.free();
    if (use_cfg) st_u.free();
    depth.free();

    // the vocoder upsamples 1920x, so decoding a long clip in one graph asks for gigabytes at once.
    // walk it in windows with enough left context for the convolutions to reach back over
    const int spf = m.cfg.samples_per_frame;
    const int ctx = m.cfg.voc.sliding_window + 16;
    const int step = 40;
    std::vector<float> audio;
    audio.reserve((size_t) src_T * spf);
    for (int start = 0; start < src_T; start += step) {
        const int count = std::min(step, src_T - start);
        const int cs = start > ctx ? start - ctx : 0;
        std::vector<int> sub(out.begin() + (size_t) cs * nc, out.begin() + (size_t) (start + count) * nc);
        std::vector<float> part = codec.decode(sub, start + count - cs);
        const size_t skip = (size_t) (start - cs) * spf;
        const size_t want = (size_t) count * spf;
        if (part.size() < skip + want) break;
        audio.insert(audio.end(), part.begin() + skip, part.begin() + skip + want);
    }
    return audio;
}

}
