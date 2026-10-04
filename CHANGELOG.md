# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.8] - 2026-10-04

### Changed
- **Upstream Submodule Bump:** Synchronized `vendor/stable-diffusion.cpp` to latest upstream `master` (`3f8527a` / `master-929-3f8527a`) and updated nested submodules including `ggml` (`89c4413f`).
- **Memory Lifecycle & C API Signatures:**
  - Migrated `generate_image`, `generate_video`, and `upscale` bindings to double-pointer output arguments (`sd_image_t** images_out, int* num_images_out`).
  - Added explicit resource deallocation via `free_sd_images` and `free_sd_audio` inside `try...finally` blocks in high-level Python wrappers, eliminating memory leaks and cross-CRT allocation/free issues.
- **Low-Level Struct Alignment (`stable_diffusion_cpp.py`):**
  - Updated `sd_ctx_params_t`: Added multi-backend routing options (`backend`, `params_backend`, `split_mode`, `auto_fit`), `max_vram` GiB string budget, model input paths (`uncond_diffusion_model_path`, `embeddings_connectors_path`, `audio_vae_path`, `audio_encoder_path`, `ip_adapter_path`, `motion_module_path`, `pulid_weights_path`), and `vae_format`. Removed deprecated CPU offload and circular flags.
  - Updated `sd_tiling_params_t`: Aligned spatial dimensions to `tile_size_w`/`tile_size_h` and `rel_size_w`/`rel_size_h`; added `temporal_tiling` and `extra_tiling_args`.
  - Updated `sd_sample_params_t`: Added `extra_sample_args`.
  - Updated `sd_hires_params_t`: Added `custom_sigmas` and `custom_sigmas_count`.
  - Updated `sd_img_gen_params_t` & `sd_vid_gen_params_t`: Added IP-Adapter, PuLID, Qwen-Image layer controls, circular generation flags, and unified `ref_image_args`.
- **High-Level Wrapper (`stable_diffusion.py` & `_internals.py`):**
  - Refactored `_StableDiffusionModel` to populate updated `sd_ctx_params_t` struct cleanly.
  - Updated `_UpscalerModel` to call new 6-argument `new_upscaler_ctx` signature.
  - Updated `generate_image`, `generate_video`, `upscale`, and `_image_slice` to handle new return pointers and manage memory freeing.
  - Updated `SAMPLE_METHOD_MAP`, `SCHEDULER_MAP`, `PREDICTION_MAP`, and `GGML_TYPE_MAP`.

### Added
- **New Structs:** `sd_audio_t`, `sd_ref_video_t`, `sd_pulid_params_t`, `sd_image_preprocess_params_t`, `sd_adetailer_params_t`.
- **New Enums:** `SDVAEFormat`, `SDCancelMode`, `SDLogLevel`.
- **New Samplers:** `EULER_CFG_PP`, `EULER_A_CFG_PP`, `EULER_GE`, `DPMPP2M_SDE`, `DPMPP2M_SDE_BT`, `LMS`.
- **New Schedulers:** `LTX2`, `LOGIT_NORMAL`, `FLUX2`, `FLUX`, `BETA`, `LLADA_IMAGE`.
- **New Quantization Types:** `SD_TYPE_Q1_0`, `SD_TYPE_Q2_0`, `SD_TYPE_F8_E4M3`, `SD_TYPE_F8_E5M2`.
- **New C API Functions:**
  - ControlNet hot-swapping: `sd_ctx_load_control_net`, `sd_ctx_unload_control_net`, `sd_ctx_has_control_net`.
  - ADetailer pipeline: `new_adetailer_ctx`, `free_adetailer_ctx`, `adetail_image`.
  - Generation cancellation: `sd_cancel_generation`.
  - System and device introspection: `sd_list_devices`, `sd_get_model_version_name`, `get_upscaler_model_scale`.
  - IMatrix quantization APIs: `load_imatrix`, `save_imatrix`, `enable_imatrix_collection`, `disable_imatrix_collection`.
  - Parameter initializers: `sd_ctx_params_init`, `sd_sample_params_init`, `sd_img_gen_params_init`, `sd_vid_gen_params_init`, `sd_cache_params_init`, `sd_hires_params_init`.
  - Multi-component conversion: `convert_with_components`.

## [0.4.7] - 2026-05-12

### Changed
- Updated submodule and resolved missing third-party module build errors on MSVC.
