#ifndef PIPELINESDXL_HPP
#define PIPELINESDXL_HPP

#include <MNN/Interpreter.hpp>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <future>
#include <memory>
#include <stdexcept>
#include <string>

#include "Config.hpp"
#include "MnnUtils.hpp"
#include "PipelineQnn.hpp"

// sdxl: QNN (HTP) UNet/VAE at a fixed 1024x1024, dual MNN CLIP encoders
// (CLIP-L + CLIP-G) with pooled output, plus SDXL micro-conditioning
// (time_ids). In lowram mode every stage model is loaded right before use and
// released right after, trading latency for peak memory.
class PipelineSdxl : public PipelineQnn {
 public:
  PipelineSdxl(TextEncoder &text_encoder, const std::string &model_dir,
               std::string clip_path, std::string clip2_path,
               std::string unet_path, std::string vae_decoder_path,
               std::string vae_encoder_path, bool use_v_pred, bool lowram)
      : PipelineQnn(text_encoder, model_dir, /*sdxl=*/true, use_v_pred),
        clip_path_(std::move(clip_path)),
        clip2_path_(std::move(clip2_path)),
        unet_path_(std::move(unet_path)),
        vae_decoder_path_(std::move(vae_decoder_path)),
        vae_encoder_path_(std::move(vae_encoder_path)),
        lowram_(lowram) {}

  ~PipelineSdxl() override { releaseClips(); }

  bool initialize() override {
    if (lowram_) {
      QNN_INFO(
          "[lowram] SDXL low-RAM mode: skipping pre-load of CLIP/UNET/VAE "
          "models");
      return true;
    }

    try {
      loadClipsIfNeeded();
    } catch (const std::exception &e) {
      QNN_ERROR("%s", e.what());
      return false;
    }
    QNN_INFO("Persistent SDXL MNN CLIP1/CLIP2 sessions created.");

    unet_ = qnn_runtime::createModel(unet_path_, "unet");
    if (!unet_) {
      QNN_ERROR("Failed create QNN UNET model.");
      return false;
    }
    vae_decoder_ = qnn_runtime::createModel(vae_decoder_path_, "vae_decoder");
    if (!vae_decoder_) {
      QNN_ERROR("Failed create QNN VAE Decoder model.");
      return false;
    }
    if (vae_encoder_path_.empty()) {
      QNN_INFO("img2img disabled: VAE encoder not available");
    } else if (getenv("LOCALDREAM_SDXL_SPILL_FILL_PROBE")) {
      // Probe mode needs a real encoder context so the backend can report its
      // standalone spill/fill requirement. Normal mode keeps it lazy instead.
      vae_encoder_ = qnn_runtime::createModel(vae_encoder_path_, "vae_encoder");
      if (!vae_encoder_) QNN_WARN("Failed create QNN VAE Enc model.");
    } else {
      QNN_INFO(
          "SDXL fast mode: VAE encoder is lazy; UNET + decoder stay resident");
    }

    // Probe mode: getProperty(MAX_SPILLFILL_BUFFER_SIZE) returns 0 on this HTP
    // version and pre-2.35 binaries carry no metadata, so the only way to learn
    // each model's real spill-fill need is the backend's own validation error.
    // Register each context as its own group head with a 1-byte budget:
    // creation fails and the backend logs "...is smaller than required
    // spill-fill size N" for each. Read those N from logcat, take the max, then
    // set LOCALDREAM_SDXL_SPILL_FILL_BYTES to it and unset the probe var.
    if (getenv("LOCALDREAM_SDXL_SPILL_FILL_PROBE")) {
      QNN_INFO(
          "[spill-fill] PROBE: forcing each context to log its required size; "
          "generation will NOT run this launch");
      unet_->setSpillFillGroup(1, nullptr);
      qnn_runtime::initializeApp("UNET", unet_);
      vae_decoder_->setSpillFillGroup(1, nullptr);
      qnn_runtime::initializeApp("VAEDecoder", vae_decoder_);
      if (vae_encoder_) {
        vae_encoder_->setSpillFillGroup(1, nullptr);
        qnn_runtime::initializeApp("VAEEncoder", vae_encoder_);
      }
      QNN_INFO(
          "[spill-fill] PROBE done; grep logcat for 'required spill-fill "
          "size'");
      return false;
    }

    // Fast mode still keeps UNet + VAE decoder resident, but by default
    // each QNN context owns the exact spill/fill allocation requested by its
    // binary. The previous fixed 1.5 GiB shared map could fail FastRPC/SMMU
    // buffer mapping (8003) even on devices with plenty of system RAM.
    //
    // Advanced users can opt back into sharing by setting an exact measured
    // LOCALDREAM_SDXL_SPILL_FILL_BYTES value; there is no guessed default.
    const uint64_t sf_bytes = configuredSpillFillGroupBytes();
    Qnn_ContextHandle_t group_head = nullptr;
    if (sf_bytes) {
      QNN_INFO("[spill-fill] explicit SDXL group size: %llu bytes",
               (unsigned long long)sf_bytes);
      unet_->setSpillFillGroup(sf_bytes, nullptr);
    } else {
      QNN_INFO("[spill-fill] SDXL adaptive mode: independent QNN context allocations");
    }

    // Ask QAIRT for its createFromBinary concurrent-resource group first.
    // This option is specifically intended for same-priority graphs that may
    // execute concurrently. If the target runtime rejects it, rebuild the
    // exact same primary context without the hint and keep the portable
    // independent-context concurrency path.
    unet_->setConcurrentResourceGroup(sf_bytes, nullptr);
    if (qnn_runtime::initializeApp("UNET", unet_) == EXIT_SUCCESS) {
      concurrent_group_enabled_ = true;
      QNN_INFO("[SDXL parallel] HTP concurrent-resource group accepted");
    } else {
      QNN_WARN(
          "[SDXL parallel] HTP concurrent-resource group unavailable; "
          "retrying normal context");
      unet_.reset();
      unet_ = qnn_runtime::createModel(unet_path_, "unet");
      if (!unet_) return false;
      if (sf_bytes) unet_->setSpillFillGroup(sf_bytes, nullptr);
      if (qnn_runtime::initializeApp("UNET", unet_) != EXIT_SUCCESS)
        return false;
      concurrent_group_enabled_ = false;
    }

    unet_->logSdxlGraphProbeOnce("UNET");
    unet_tokens_ = preloadedContextLength();
    if (sf_bytes) group_head = unet_->getContextHandle();
    logSpillFill("UNET", unet_);

    // Preload a second independent UNET context for exact CFG branch
    // concurrency. When supported, join it to the primary HTP concurrent
    // resource group; otherwise two ordinary contexts are still submitted
    // concurrently and the on-device overlap benchmark tells us what the
    // scheduler actually allowed.
    if (!lowram_) {
      try {
        unet_parallel_ =
            qnn_runtime::createModel(unet_path_, "unet_cfg_parallel");
        if (unet_parallel_) {
          if (sf_bytes)
            unet_parallel_->setSpillFillGroup(sf_bytes, group_head);
          if (concurrent_group_enabled_)
            unet_parallel_->setConcurrentResourceGroup(
                sf_bytes, unet_->getContextHandle());
        }
        if (unet_parallel_ &&
            qnn_runtime::initializeApp("UNET-CFG-PAR", unet_parallel_) ==
                EXIT_SUCCESS) {
          unet_parallel_tokens_ = unet_tokens_;
          QNN_INFO(
              "[SDXL parallel] second UNET context ready resource_group=%d",
              concurrent_group_enabled_ ? 1 : 0);
        } else {
          // A device may support two contexts but not the resource-sharing
          // registration. Retry the second context normally before giving up.
          unet_parallel_.reset();
          if (concurrent_group_enabled_) {
            concurrent_group_enabled_ = false;
            unet_parallel_ =
                qnn_runtime::createModel(unet_path_, "unet_cfg_parallel");
            if (unet_parallel_ && sf_bytes)
              unet_parallel_->setSpillFillGroup(sf_bytes, group_head);
            if (unet_parallel_ &&
                qnn_runtime::initializeApp("UNET-CFG-PAR",
                                           unet_parallel_) == EXIT_SUCCESS) {
              unet_parallel_tokens_ = unet_tokens_;
              QNN_INFO(
                  "[SDXL parallel] second normal context ready after "
                  "resource-group fallback");
            } else {
              unet_parallel_.reset();
            }
          }
          if (!unet_parallel_)
            QNN_WARN(
                "[SDXL parallel] second UNET context unavailable; "
                "serial fallback");
        }
      } catch (const std::exception &e) {
        unet_parallel_.reset();
        concurrent_group_enabled_ = false;
        QNN_WARN("[SDXL parallel] preload failed: %s; serial fallback",
                 e.what());
      }
    }

    if (sf_bytes) vae_decoder_->setSpillFillGroup(sf_bytes, group_head);
    if (qnn_runtime::initializeApp("VAEDecoder", vae_decoder_) != EXIT_SUCCESS)
      return false;
    logSpillFill("VAEDecoder", vae_decoder_);

    return true;
  }

  bool supportsImg2Img() const override {
    return !vae_encoder_path_.empty();
  }

 protected:
  // Per-stage decode previews would force a VAE decoder load/release per
  // step in lowram mode; disable them there.
  bool previewSupported() const override { return !lowram_; }
  // Normal SDXL generation is exactly 1024 so tiling never triggers there;
  // only ultrafix inputs exceed the fixed graph size.
  bool vaeTilingSupported() const override { return true; }
  int vaeTilePixelSize() const override { return 1024; }

  void encodeText(const ProcessedPromptPair &prompts, bool need_negative,
                  bool need_positive, Conditioning &cond) override {
    if (lowram_) loadClipsIfNeeded();
    if (!clip_interpreter_ || !clip2_interpreter_)
      throw std::runtime_error("SDXL CLIP interpreters not initialized!");

    std::vector<float> pooled(text_embedding_size_2);
    for (int chunk = 0; chunk < cond.seq_len / 77; ++chunk) {
      const int offset = chunk * 77;
      if (need_negative) {
        runDualClip(prompts.negative_embeddings.data() + offset * text_embedding_size,
                    prompts.negative_embeddings_2.data() + offset * text_embedding_size_2,
                    prompts.eos_positions[chunk],
                    cond.negHidden() + offset * cond.hidden_dim, pooled.data());
        // Take the pooled vector from the last chunk that holds actual prompt
        // text, not the last padded one: the two sides can need a different
        // number of chunks, and a short negative next to a long positive would
        // otherwise be pooled from an empty chunk.
        if (chunk + 1 == cond.negative_chunks)
          std::copy(pooled.begin(), pooled.end(), cond.negPooled());
      }
      if (need_positive) {
        runDualClip(prompts.positive_embeddings.data() + offset * text_embedding_size,
                    prompts.positive_embeddings_2.data() + offset * text_embedding_size_2,
                    prompts.eos_positions[cond.seq_len / 77 + chunk],
                    cond.posHidden() + offset * cond.hidden_dim, pooled.data());
        if (chunk + 1 == cond.positive_chunks)
          std::copy(pooled.begin(), pooled.end(), cond.posPooled());
      }
    }

    if (lowram_) releaseClips();
  }

  // Lowram model lifetimes are stage-scoped, not call-scoped: a stage model
  // loads on first use and is released only when the next stage needs the
  // memory (or by releaseTransientModels on exit). Tiled ultrafix passes
  // call vaeEncode/vaeDecode/runUnetStep dozens of times per stage, so a
  // per-call load/release would reload a multi-GB model once per tile.
  void vaeEncode(const GenerationRequest &, const float *image, float *mean,
                 float *std_dev) override {
    loadVaeEncoderIfNeeded();
    if (!vae_encoder_) throw std::runtime_error("QNN VAE Enc missing");
    if (StatusCode::SUCCESS != vae_encoder_->executeVaeEncoderGraphsSDXL(
                                   const_cast<float *>(image), mean, std_dev))
      throw std::runtime_error("QNN VAE enc SDXL exec failed");
  }

  void beginDenoise(const GenerationRequest &req) override {
    const int tokens =
        text_encoder_.contextLength(req.prompt, req.negative_prompt);

    parallel_run_enabled_ = false;
    parallel_pairs_ = 0;
    parallel_wall_ms_sum_ = 0.0;
    parallel_branch_ms_sum_ = 0.0;
    const bool want_parallel =
        !req.legacy_path && !lowram_ && req.cfg > 1.000001f;

    // The encoder has no work during sampling.
    if (vae_encoder_) releaseVaeEncoder();

    // Keep/rebuild the primary exact-CFG UNET for the requested token shape.
    if (!(unet_ && unet_tokens_ == tokens)) {
      vae_encoder_.reset();
      vae_decoder_.reset();
      unet_parallel_.reset();
      unet_.reset();

      const uint64_t sf_bytes =
          !lowram_ ? configuredSpillFillGroupBytes() : 0;
      auto make_primary = [&](bool concurrent_hint) {
        auto model = qnn_runtime::createModel(unet_path_, "unet");
        if (!model) return std::unique_ptr<QnnModel>{};
        if (sf_bytes) model->setSpillFillGroup(sf_bytes, nullptr);
        if (concurrent_hint)
          model->setConcurrentResourceGroup(sf_bytes, nullptr);
        return model;
      };

      std::unique_ptr<qnn_runtime::PatchedModelBuffer> patched;
      if (tokens > 77 && !text_encoder_.fixed_chunks_) {
        patched = qnn_runtime::applyZstdPatchToBuffer(
            unet_path_, model_dir_ + "/" + std::to_string(tokens) + ".patch");
        if (!patched) throw std::runtime_error(unet_path_);
      }

      concurrent_group_enabled_ = false;
      auto unet = make_primary(want_parallel);
      if (!unet) throw std::runtime_error("Failed create QNN UNET");
      int init_status = qnn_runtime::initializeApp(
          "UNET", unet, patched ? patched->buffer.get() : nullptr,
          patched ? patched->size : 0);
      if (init_status != EXIT_SUCCESS && want_parallel) {
        QNN_WARN(
            "[SDXL parallel] concurrent primary rejected for token patch; "
            "retrying normal context");
        unet.reset();
        unet = make_primary(false);
        if (!unet) throw std::runtime_error("Failed recreate QNN UNET");
        init_status = qnn_runtime::initializeApp(
            "UNET", unet, patched ? patched->buffer.get() : nullptr,
            patched ? patched->size : 0);
      } else if (init_status == EXIT_SUCCESS && want_parallel) {
        concurrent_group_enabled_ = true;
      }
      if (init_status != EXIT_SUCCESS)
        throw std::runtime_error("Failed init QNN UNET");

      unet->logSdxlGraphProbeOnce("UNET");
      unet_ = std::move(unet);
      unet_tokens_ = tokens;
      QNN_INFO(
          "[SDXL] primary UNET loaded for %d tokens concurrent_group=%d",
          tokens, concurrent_group_enabled_ ? 1 : 0);
    }
    unet_->resetSdxlStaticInputCache();

    // Legacy path is intentionally the original serial reference. Otherwise,
    // prepare an independent second context so exact cond/uncond branches can
    // overlap on HTP. Low-RAM mode stays serial because duplicating the UNET
    // would defeat its purpose.
    if (want_parallel) {
      if (!(unet_parallel_ && unet_parallel_tokens_ == tokens)) {
        unet_parallel_.reset();
        try {
          const uint64_t sf_bytes = configuredSpillFillGroupBytes();
          auto make_second = [&](bool concurrent_hint) {
            auto model =
                qnn_runtime::createModel(unet_path_, "unet_cfg_parallel");
            if (!model) return std::unique_ptr<QnnModel>{};
            if (sf_bytes)
              model->setSpillFillGroup(sf_bytes, unet_->getContextHandle());
            if (concurrent_hint)
              model->setConcurrentResourceGroup(sf_bytes,
                                                unet_->getContextHandle());
            return model;
          };

          std::unique_ptr<qnn_runtime::PatchedModelBuffer> patched2;
          if (tokens > 77 && !text_encoder_.fixed_chunks_) {
            patched2 = qnn_runtime::applyZstdPatchToBuffer(
                unet_path_,
                model_dir_ + "/" + std::to_string(tokens) + ".patch");
            if (!patched2) throw std::runtime_error("patch second UNET failed");
          }

          auto second = make_second(concurrent_group_enabled_);
          if (!second) throw std::runtime_error("create second UNET failed");
          int second_status = qnn_runtime::initializeApp(
              "UNET-CFG-PAR", second,
              patched2 ? patched2->buffer.get() : nullptr,
              patched2 ? patched2->size : 0);
          if (second_status != EXIT_SUCCESS && concurrent_group_enabled_) {
            QNN_WARN(
                "[SDXL parallel] concurrent second rejected; retrying "
                "independent context");
            concurrent_group_enabled_ = false;
            second.reset();
            second = make_second(false);
            if (!second)
              throw std::runtime_error("recreate second UNET failed");
            second_status = qnn_runtime::initializeApp(
                "UNET-CFG-PAR", second,
                patched2 ? patched2->buffer.get() : nullptr,
                patched2 ? patched2->size : 0);
          }
          if (second_status != EXIT_SUCCESS)
            throw std::runtime_error("init second UNET failed");

          unet_parallel_ = std::move(second);
          unet_parallel_tokens_ = tokens;
          QNN_INFO(
              "[SDXL parallel] second context loaded for %d tokens "
              "resource_group=%d",
              tokens, concurrent_group_enabled_ ? 1 : 0);
        } catch (const std::exception &e) {
          unet_parallel_.reset();
          QNN_WARN("[SDXL parallel] setup failed: %s; using serial exact CFG",
                   e.what());
        }
      }
      if (unet_parallel_) {
        unet_parallel_->resetSdxlStaticInputCache();
        parallel_run_enabled_ = true;
      }
    }

    QNN_INFO(
        "[SDXL path] requested=%s resolved=%s apg=%d resource_group=%d",
        req.legacy_path ? "legacy-path" : "parallel-exact",
        parallel_run_enabled_ ? "parallel-exact" : "serial-exact",
        req.apg_quality ? 1 : 0, concurrent_group_enabled_ ? 1 : 0);
  }

  void runUnetStep(const GenerationRequest &req,
                   const float *latents_batch2, float timestep,
                   bool skip_uncond, Conditioning &cond,
                   float *out_batch2) override {
    if (!unet_) throw std::runtime_error("QNN UNET missing");

    const int single_latent_size = 1 * 4 * sample_width * sample_height;
    const int ts = static_cast<int>(timestep);
    float *latents_in = const_cast<float *>(latents_batch2);
    float *time_ids = cond.time_ids.data();

    const bool do_parallel =
        !skip_uncond && !req.legacy_path && parallel_run_enabled_ &&
        unet_parallel_;

    if (do_parallel) {
      const auto pair_start = std::chrono::high_resolution_clock::now();
      double uncond_ms = 0.0;
      double cond_ms = 0.0;

      auto uncond_future = std::async(std::launch::async, [&]() {
        const auto t0 = std::chrono::high_resolution_clock::now();
        auto st = unet_->executeUnetGraphsSDXL(
            latents_in, ts, cond.negHidden(), cond.negPooled(), time_ids,
            out_batch2, cond.seq_len, cond.negative_chunks);
        uncond_ms = std::chrono::duration<double, std::milli>(
                        std::chrono::high_resolution_clock::now() - t0)
                        .count();
        return st;
      });

      auto cond_future = std::async(std::launch::async, [&]() {
        const auto t0 = std::chrono::high_resolution_clock::now();
        auto st = unet_parallel_->executeUnetGraphsSDXL(
            latents_in + single_latent_size, ts, cond.posHidden(),
            cond.posPooled(), time_ids + 6,
            out_batch2 + single_latent_size, cond.seq_len,
            cond.positive_chunks);
        cond_ms = std::chrono::duration<double, std::milli>(
                      std::chrono::high_resolution_clock::now() - t0)
                      .count();
        return st;
      });

      const auto uncond_status = uncond_future.get();
      const auto cond_status = cond_future.get();
      const double wall_ms =
          std::chrono::duration<double, std::milli>(
              std::chrono::high_resolution_clock::now() - pair_start)
              .count();

      if (uncond_status != StatusCode::SUCCESS ||
          cond_status != StatusCode::SUCCESS) {
        QNN_WARN(
            "[SDXL parallel] execution failed at t=%d; retrying serial exact CFG",
            ts);
        parallel_run_enabled_ = false;
      } else {
        ++parallel_pairs_;
        parallel_wall_ms_sum_ += wall_ms;
        parallel_branch_ms_sum_ += uncond_ms + cond_ms;
        const double overlap_pct =
            (uncond_ms + cond_ms) > 0.0
                ? 100.0 *
                      std::max(0.0, 1.0 - wall_ms / (uncond_ms + cond_ms))
                : 0.0;
        QNN_INFO(
            "[SDXL parallel] t=%d uncond=%.3fms cond=%.3fms wall=%.3fms "
            "overlap=%.1f%%",
            ts, uncond_ms, cond_ms, wall_ms, overlap_pct);
        return;
      }
    }

    // Serial exact CFG reference / fallback.
    if (!skip_uncond) {
      const auto uncond_start = std::chrono::high_resolution_clock::now();
      if (StatusCode::SUCCESS != unet_->executeUnetGraphsSDXL(
                                     latents_in, ts, cond.negHidden(),
                                     cond.negPooled(), time_ids, out_batch2,
                                     cond.seq_len, cond.negative_chunks))
        throw std::runtime_error("QNN UNET SDXL exec failed (uncond)");
      const double uncond_ms =
          std::chrono::duration<double, std::milli>(
              std::chrono::high_resolution_clock::now() - uncond_start)
              .count();
      QNN_INFO("[SDXL CFG] t=%d uncond=%.3fms", ts, uncond_ms);
    }

    const auto cond_start = std::chrono::high_resolution_clock::now();
    if (StatusCode::SUCCESS != unet_->executeUnetGraphsSDXL(
                                   latents_in + single_latent_size, ts,
                                   cond.posHidden(), cond.posPooled(),
                                   time_ids + 6,
                                   out_batch2 + single_latent_size,
                                   cond.seq_len, cond.positive_chunks))
      throw std::runtime_error("QNN UNET SDXL exec failed (cond)");
    const double cond_ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - cond_start)
            .count();
    QNN_INFO("[SDXL CFG] t=%d cond=%.3fms skip_uncond=%d", ts, cond_ms,
             skip_uncond ? 1 : 0);
  }

  void endDenoise() override {
    if (parallel_pairs_ > 0) {
      const double avg_wall = parallel_wall_ms_sum_ / parallel_pairs_;
      const double avg_branch_sum =
          parallel_branch_ms_sum_ / parallel_pairs_;
      const double overlap_pct =
          avg_branch_sum > 0.0
              ? 100.0 * std::max(0.0, 1.0 - avg_wall / avg_branch_sum)
              : 0.0;
      QNN_INFO(
          "[SDXL parallel summary] pairs=%d avg_wall=%.3fms "
          "avg_branch_sum=%.3fms overlap=%.1f%%",
          parallel_pairs_, avg_wall, avg_branch_sum, overlap_pct);
    }
    if (!lowram_ || !unet_) return;
    unet_parallel_.reset();
    unet_.reset();
    QNN_INFO("[lowram] SDXL UNET released");
  }

  void vaeDecode(const GenerationRequest &, const float *latents,
                 float *pixels) override {
    if (!vae_decoder_) {
      vae_decoder_ = createVaeModel(vae_decoder_path_, "vae_decoder");
      QNN_INFO("SDXL VAE Decoder loaded");
    }
    if (!vae_decoder_) throw std::runtime_error("QNN VAE Dec missing");
    if (StatusCode::SUCCESS != vae_decoder_->executeVaeDecoderGraphsSDXL(
                                   const_cast<float *>(latents), pixels))
      throw std::runtime_error("QNN VAE dec SDXL exec failed");
    // Lowram: stays loaded for the rest of the decode stage; released by
    // releaseTransientModels when generate() exits.
  }

  // Catch-all for lowram: release whatever stage model is still loaded when
  // generate() exits (normal return or exception).
  void releaseTransientModels() override {
    if (!lowram_) return;
    if (clip_interpreter_ || clip2_interpreter_) releaseClips();
    if (unet_) {
      unet_.reset();
      QNN_INFO("[lowram] SDXL UNET released");
    }
    if (vae_decoder_) {
      vae_decoder_.reset();
      QNN_INFO("[lowram] SDXL VAE Decoder released");
    }
    if (vae_encoder_) releaseVaeEncoder();
  }

 private:
  // Context length of the unpatched unet binary on disk.
  int preloadedContextLength() const {
    return text_encoder_.fixed_chunks_ ? 77 * text_encoder_.max_chunks_ : 77;
  }

  std::unique_ptr<QnnModel> createVaeModel(const std::string &path,
                                          const std::string &name) {
    auto model = qnn_runtime::createModel(path, name);
    if (!model) throw std::runtime_error("Failed create QNN model: " + name);
    if (!lowram_ && unet_) {
      const uint64_t sf_bytes = configuredSpillFillGroupBytes();
      if (sf_bytes)
        model->setSpillFillGroup(sf_bytes, unet_->getContextHandle());
    }
    if (qnn_runtime::initializeApp(name, model) != EXIT_SUCCESS)
      throw std::runtime_error("Failed init QNN model: " + name);
    return model;
  }

  // No guessed default. Different custom SDXL binaries have different HTP
  // scratch requirements and FastRPC virtual-address limits. With no override,
  // QNN allocates each context's required spill/fill buffer independently.
  static uint64_t configuredSpillFillGroupBytes() {
    const char *e = getenv("LOCALDREAM_SDXL_SPILL_FILL_BYTES");
    if (!e || !*e) return 0;
    return strtoull(e, nullptr, 10);
  }

  // Diagnostic: log a model's real HTP spill-fill requirement so the right
  // group buffer size can be chosen. Run once with sharing disabled
  // (LOCALDREAM_SDXL_SPILL_FILL_BYTES=0) to read all three clean values.
  static void logSpillFill(const char *name,
                           const std::unique_ptr<QnnModel> &model) {
    if (!model) return;
    uint64_t bytes = model->querySpillFillSize();
    QNN_INFO("[spill-fill] %s requires %llu bytes (%.1f MB)", name,
             (unsigned long long)bytes, bytes / (1024.0 * 1024.0));
  }

 protected:
  void loadClipsIfNeeded() {
    if (!clip_interpreter_) {
      clip_interpreter_ = createMnnInterpreterMmap(clip_path_.c_str());
      if (!clip_interpreter_)
        throw std::runtime_error("Failed load SDXL CLIP1 MNN");
    }
    if (!clip2_interpreter_) {
      clip2_interpreter_ = createMnnInterpreterMmap(clip2_path_.c_str());
      if (!clip2_interpreter_)
        throw std::runtime_error("Failed load SDXL CLIP2 MNN");
    }
    MnnSessionOptions opts;  // CLIP always runs on CPU
    if (!clip_session_) {
      clip_session_ = createMnnSession(clip_interpreter_, opts);
      if (!clip_session_)
        throw std::runtime_error("Failed create SDXL CLIP1 session");
      auto in1 =
          clip_interpreter_->getSessionInput(clip_session_, "input_embedding");
      clip_interpreter_->resizeTensor(in1, {1, 77, text_embedding_size});
      clip_interpreter_->resizeSession(clip_session_);
      clip_interpreter_->releaseModel();
    }
    if (!clip2_session_) {
      clip2_session_ = createMnnSession(clip2_interpreter_, opts);
      if (!clip2_session_)
        throw std::runtime_error("Failed create SDXL CLIP2 session");
      auto in2 = clip2_interpreter_->getSessionInput(clip2_session_,
                                                     "input_embedding");
      clip2_interpreter_->resizeTensor(in2, {1, 77, text_embedding_size_2});
      clip2_interpreter_->resizeSession(clip2_session_);
      clip2_interpreter_->releaseModel();
    }
    if (lowram_) QNN_INFO("[lowram] SDXL CLIP MNN loaded");
  }

  void releaseClips() {
    if (clip_session_ && clip_interpreter_) {
      clip_interpreter_->releaseSession(clip_session_);
    }
    clip_session_ = nullptr;
    if (clip2_session_ && clip2_interpreter_) {
      clip2_interpreter_->releaseSession(clip2_session_);
    }
    clip2_session_ = nullptr;
    delete clip_interpreter_;
    clip_interpreter_ = nullptr;
    delete clip2_interpreter_;
    clip2_interpreter_ = nullptr;
    if (lowram_) QNN_INFO("[lowram] SDXL CLIP MNN released");
  }

  void loadVaeEncoderIfNeeded() {
    if (vae_encoder_) return;
    if (vae_encoder_path_.empty())
      throw std::runtime_error("SDXL VAE Encoder path missing");
    vae_encoder_ = createVaeModel(vae_encoder_path_, "vae_encoder");
    QNN_INFO("SDXL VAE Encoder loaded on demand");
  }

  void releaseVaeEncoder() {
    if (!vae_encoder_) return;
    vae_encoder_.reset();
    QNN_INFO("SDXL VAE Encoder released before denoise");
  }

  // Encoder 1 (CLIP-L): 77x768 -> last_hidden_state 77x768.
  // Encoder 2 (CLIP-G): 77x1280 -> last_hidden_state 77x1280 + pooled_output
  // 77x1280 (exported without pooling; we select the EOS row here as the true
  // pooled embedding). Hidden states are concatenated along the feature dim:
  // [77, 768] + [77, 1280] = [77, 2048].
  void runDualClip(const float *emb1,
                   const float *emb2, int eos_pos,
                   float *out_hidden_concat, float *out_pooled) {
    const int concat_dim = text_embedding_size + text_embedding_size_2;

    auto in1 =
        clip_interpreter_->getSessionInput(clip_session_, "input_embedding");
    memcpy(in1->host<float>(), emb1,
           77 * text_embedding_size * sizeof(float));
    clip_interpreter_->runSession(clip_session_);
    auto out1 =
        clip_interpreter_->getSessionOutput(clip_session_, "last_hidden_state");
    const float *out1_data = out1->host<float>();

    auto in2 =
        clip2_interpreter_->getSessionInput(clip2_session_, "input_embedding");
    memcpy(in2->host<float>(), emb2,
           77 * text_embedding_size_2 * sizeof(float));
    clip2_interpreter_->runSession(clip2_session_);
    auto out2_hidden = clip2_interpreter_->getSessionOutput(
        clip2_session_, "last_hidden_state");
    auto out2_pool =
        clip2_interpreter_->getSessionOutput(clip2_session_, "pooled_output");
    const float *out2_hidden_data = out2_hidden->host<float>();
    const float *out2_pool_data = out2_pool->host<float>();

    for (int t = 0; t < 77; t++) {
      memcpy(out_hidden_concat + t * concat_dim,
             out1_data + t * text_embedding_size,
             text_embedding_size * sizeof(float));
      memcpy(out_hidden_concat + t * concat_dim + text_embedding_size,
             out2_hidden_data + t * text_embedding_size_2,
             text_embedding_size_2 * sizeof(float));
    }
    // Use the true EOS row; textual-inversion slots also carry ID 49407.
    memcpy(out_pooled, out2_pool_data + eos_pos * text_embedding_size_2,
           text_embedding_size_2 * sizeof(float));
  }

  const std::string clip_path_;
  const std::string clip2_path_;
  const std::string unet_path_;
  const std::string vae_decoder_path_;
  const std::string vae_encoder_path_;
  const bool lowram_;
  int unet_tokens_ = 77;

  // Independent second QNN context for simultaneous cond/uncond execution.
  std::unique_ptr<QnnModel> unet_parallel_;
  int unet_parallel_tokens_ = 0;
  bool parallel_run_enabled_ = false;
  bool concurrent_group_enabled_ = false;
  int parallel_pairs_ = 0;
  double parallel_wall_ms_sum_ = 0.0;
  double parallel_branch_ms_sum_ = 0.0;

  MNN::Interpreter *clip_interpreter_ = nullptr;
  MNN::Interpreter *clip2_interpreter_ = nullptr;
  MNN::Session *clip_session_ = nullptr;
  MNN::Session *clip2_session_ = nullptr;
};

#endif  // PIPELINESDXL_HPP
