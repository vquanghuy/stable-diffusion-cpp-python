# Agent Guidelines for stable-diffusion-cpp-python

This document provides operational guidelines, architecture context, and standard procedures for AI coding agents working in this repository.

---

## 1. Project Overview & Architecture

This repository is a Python ctypes wrapper for [`leejet/stable-diffusion.cpp`](https://github.com/leejet/stable-diffusion.cpp).
- **Personal Fork:** [`vquanghuy/stable-diffusion-cpp-python`](https://github.com/vquanghuy/stable-diffusion-cpp-python)
- **Upstream Python Wrapper:** [`william-murray1204/stable-diffusion-cpp-python`](https://github.com/william-murray1204/stable-diffusion-cpp-python)
- **Primary Objective:** Provide zero-compilation, multi-architecture CUDA wheels for Google Colab and downstream AI research projects (e.g., `omni-diffusion-study`), while tracking bleeding-edge model architectures (FLUX, SD 3.5, SDXL, SD 1.5, SD 2.1) supported by `stable-diffusion.cpp`.

### Directory Layout
```text
stable-diffusion-cpp-python/
├── vendor/
│   └── stable-diffusion.cpp/     # Upstream C++ engine submodule (includes nested ggml)
├── stable_diffusion_cpp/         # Python wrapper package
│   ├── stable_diffusion_cpp.py   # Low-level ctypes bindings (C structs, enums, function prototypes)
│   ├── _internals.py             # Intermediate OOP wrappers (_StableDiffusionModel, _UpscalerModel)
│   ├── stable_diffusion.py       # High-level public API (StableDiffusion, Upscaler)
│   └── __init__.py               # Package exports and version metadata
├── notebooks/                    # Colab multi-arch builder and inference notebooks
├── tests/                        # Unit and integration test suites
├── scratch/                      # Temporary read-only scratch scripts (gitignored)
├── CHANGELOG.md                  # Release and sync history (Keep a Changelog format)
└── pyproject.toml / CMakeLists.txt
```

---

## 2. Upstream Synchronization Lifecycle

When synchronizing with newer commits or releases from `leejet/stable-diffusion.cpp`, always follow this strict protocol:

### Step 1: Update Submodules
```bash
git submodule update --init --recursive
cd vendor/stable-diffusion.cpp
git fetch origin master
git checkout origin/master
cd ../..
```

### Step 2: Audit C API Header Changes
Compare the C header before and after updating:
```bash
git -C vendor/stable-diffusion.cpp diff HEAD@{1}..HEAD -- stable-diffusion.h
```

Inspect for:
1. **Struct field additions, reordering, or type changes** (e.g., `sd_ctx_params_t`, `sd_sample_params_t`, `sd_img_gen_params_t`, `sd_tiling_params_t`).
2. **New structs or types** (e.g., `sd_audio_t`, `sd_ref_video_t`, `sd_pulid_params_t`).
3. **Function prototype changes** (e.g., functions returning `sd_image_t**` instead of single pointers, newly introduced memory deallocators like `free_sd_images`).
4. **New enums or enum values** (samplers, schedulers, quantization types, cancel modes).

### Step 3: Align Python ctypes Layer
- Update `stable_diffusion_cpp/stable_diffusion_cpp.py`:
  - Ensure `ctypes.Structure._fields_` matches `stable-diffusion.h` byte-for-byte in exact order and size.
  - Define all new function signatures with accurate `argtypes` and `restype`.
- Update `stable_diffusion_cpp/_internals.py` and `stable_diffusion_cpp/stable_diffusion.py`:
  - Adapt wrapper calls to new struct fields and function parameters.
  - **Memory Management Rule:** Always manage C memory allocations properly. If the C API allocates an image array, ensure it is freed with `sd_cpp.free_sd_images(...)` (or `free_sd_audio(...)`).

### Step 4: Verification & Smoke Test
1. Verify bytecode compilation across all Python files:
   ```bash
   python3 -m py_compile stable_diffusion_cpp/*.py
   ```
2. Verify mapping dictionaries and enum alignments using a mock or tests:
   ```bash
   pytest tests/
   ```

### Step 5: Version Bump, Changelog, and Tagging
1. Bump version in `stable_diffusion_cpp/__init__.py`.
2. Document all additions, changes, and fixes in `CHANGELOG.md` following [Keep a Changelog](https://keepachangelog.com/).
3. Commit and tag:
   - **Git Source Tags:** Use clean SemVer (e.g., `v0.4.8`).
   - **Wheel Artifacts:** Append PEP 440 local version identifier (e.g., `0.4.8+cu122`) when compiling binary wheels specifically for Colab CUDA 12.2.
   ```bash
   git add vendor/stable-diffusion.cpp stable_diffusion_cpp/ CHANGELOG.md
   git commit -m "feat(upstream): sync leejet/stable-diffusion.cpp to <commit-hash> and update bindings"
   git tag -a v0.4.8 -m "Release v0.4.8 with upstream submodule sync and updated bindings"
   git push origin main --tags
   ```

---

## 3. Build & Packaging Commands

### Local Development (CPU)
```bash
# Editable install
pip install -e .

# Or build wheel locally
python3 -m pip wheel . --no-deps -w dist/
```

### Multi-Architecture CUDA Wheel Compilation (Google Colab)
Used to generate fat binaries supporting Tesla T4 (`sm_75`), Ampere A100 (`sm_80`), Ada Lovelace L4 (`sm_89`), and forward-compatible PTX (`sm_89-virtual`):
```bash
export CMAKE_ARGS="-DSD_CUDA=ON -DCMAKE_CUDA_ARCHITECTures=75;80;89;89-virtual -DCMAKE_CUDA_RUNTIME_LIBRARY=Static"
export FORCE_CMAKE=1
export CMAKE_BUILD_PARALLEL_LEVEL=2

python3 -m pip wheel . --no-deps -w /content/dist/ -v
```

---

## 4. Agent Safety & Working Conventions

- **Plan Before Edit:** Propose diagnosis or plans before making changes to source code.
- **Destructive Operations:** Never run destructive git commands (`git reset --hard`, force push, `git clean -f`) or delete repository files (`rm`, `rm -rf`) without explicit user confirmation.
- **Scratch Directory:** If analytical or diagnostic scripts are needed, write them in `./scratch/` within the repository root. Scratch scripts must be strictly read-only and analytical.
- **Documentation Integrity:** Preserve existing docstrings, typing annotations, and comments when editing Python bindings.
- **Language Preference:** Keep communication in English unless explicitly requested otherwise.
