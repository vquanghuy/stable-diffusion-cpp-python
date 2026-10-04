from __future__ import annotations

import os
import re
import sys
import ctypes
import pathlib
import functools
from enum import IntEnum
from typing import (
    TYPE_CHECKING,
    Any,
    List,
    Union,
    Generic,
    NewType,
    TypeVar,
    Callable,
    Optional,
)

try:
    from typing_extensions import TypeAlias
except ImportError:
    from typing import TypeAlias  # type: ignore


# Load the library
def _load_shared_library(lib_base_name: str):
    # Construct the paths to the possible shared library names
    _base_path = pathlib.Path(os.path.abspath(os.path.dirname(__file__))) / "lib"
    # Searching for the library in the current directory under the name "libstable-diffusion" (default name
    # for stable-diffusion-cpp) and "stable-diffusion" (default name for this repo)
    _lib_paths: List[pathlib.Path] = []

    # Determine the file extension based on the platform
    if sys.platform.startswith("linux"):
        _lib_paths += [
            _base_path / f"lib{lib_base_name}.so",
        ]
    elif sys.platform == "darwin":
        _lib_paths += [
            _base_path / f"lib{lib_base_name}.so",
            _base_path / f"lib{lib_base_name}.dylib",
        ]
    elif sys.platform == "win32":
        # Load the HIP runtime if present in the lib directory
        def extract_version(p: pathlib.Path):
            m = re.search(r"amdhip64[_\-]?(\d+)", p.name)
            return int(m.group(1)) if m else 0

        hip_dlls = sorted(_base_path.glob("amdhip64*.dll"), key=extract_version, reverse=True)
        if hip_dlls:
            ctypes.CDLL(str(hip_dlls[0]), winmode=0x8)

        _lib_paths += [
            _base_path / f"{lib_base_name}.dll",
            _base_path / f"lib{lib_base_name}.dll",
        ]
    else:
        raise RuntimeError("Unsupported platform")

    if "STABLE_DIFFUSION_CPP_LIB" in os.environ:
        lib_base_name = os.environ["STABLE_DIFFUSION_CPP_LIB"]
        _lib = pathlib.Path(lib_base_name)
        _base_path = _lib.parent.resolve()
        _lib_paths = [_lib.resolve()]

    cdll_args = dict()  # type: ignore
    # Add the library directory to the DLL search path on Windows (if needed)
    if sys.platform == "win32" and sys.version_info >= (3, 8):
        os.add_dll_directory(str(_base_path))
        if "CUDA_PATH" in os.environ:
            os.add_dll_directory(os.path.join(os.environ["CUDA_PATH"], "bin"))
            os.add_dll_directory(os.path.join(os.environ["CUDA_PATH"], "lib"))
        if "HIP_PATH" in os.environ:
            os.add_dll_directory(os.path.join(os.environ["HIP_PATH"], "bin"))
            os.add_dll_directory(os.path.join(os.environ["HIP_PATH"], "lib"))
        cdll_args["winmode"] = ctypes.RTLD_GLOBAL

    # Try to load the shared library, handling potential errors
    for _lib_path in _lib_paths:
        if _lib_path.exists():
            try:
                return ctypes.CDLL(str(_lib_path), **cdll_args)  # type: ignore
            except Exception as e:
                raise RuntimeError(f"Failed to load shared library '{_lib_path}': {e}")

    raise FileNotFoundError(f"Shared library with base name '{lib_base_name}' not found")


# Specify the base name of the shared library to load
_lib_base_name = "stable-diffusion"

# Load the library
_lib = _load_shared_library(_lib_base_name)

# ctypes sane type hint helpers
#
# - Generic Pointer and Array types
# - PointerOrRef type with a type hinted byref function
#
# NOTE: Only use these for static type checking not for runtime checks
# no good will come of that

if TYPE_CHECKING:
    CtypesCData = TypeVar("CtypesCData", bound=ctypes._CData)  # type: ignore

    CtypesArray: TypeAlias = ctypes.Array[CtypesCData]  # type: ignore

    CtypesPointer: TypeAlias = ctypes._Pointer[CtypesCData]  # type: ignore

    CtypesVoidPointer: TypeAlias = ctypes.c_void_p

    class CtypesRef(Generic[CtypesCData]):
        pass

    CtypesPointerOrRef: TypeAlias = Union[CtypesPointer[CtypesCData], CtypesRef[CtypesCData]]

    CtypesFuncPointer: TypeAlias = ctypes._FuncPointer  # type: ignore

F = TypeVar("F", bound=Callable[..., Any])


def ctypes_function_for_shared_library(lib: ctypes.CDLL):
    def ctypes_function(name: str, argtypes: List[Any], restype: Any, enabled: bool = True):
        def decorator(f: F) -> F:
            if enabled:
                func = getattr(lib, name)
                func.argtypes = argtypes
                func.restype = restype
                functools.wraps(f)(func)
                return func
            else:
                return f

        return decorator

    return ctypes_function


ctypes_function = ctypes_function_for_shared_library(_lib)


def byref(obj: CtypesCData, offset: Optional[int] = None) -> CtypesRef[CtypesCData]:
    """Type-annotated version of ctypes.byref"""
    ...


byref = ctypes.byref  # type: ignore


# // Abort callback
# // If not NULL, called before ggml computation
# // If it returns true, the computation is aborted
# typedef bool (*ggml_abort_callback)(void * data);
ggml_abort_callback = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.c_void_p)


# ===========================================
# include/stable-diffusion.h bindings
# ===========================================


# enum rng_type_t {
#     STD_DEFAULT_RNG,
#     CUDA_RNG,
#     CPU_RNG,
#     RNG_TYPE_COUNT
# };
class RNGType(IntEnum):
    STD_DEFAULT_RNG = 0
    CUDA_RNG = 1
    CPU_RNG = 2
    RNG_TYPE_COUNT = 3


# enum sample_method_t {
#     EULER_SAMPLE_METHOD,
#     EULER_A_SAMPLE_METHOD,
#     HEUN_SAMPLE_METHOD,
#     DPM2_SAMPLE_METHOD,
#     DPMPP2S_A_SAMPLE_METHOD,
#     DPMPP2M_SAMPLE_METHOD,
#     DPMPP2Mv2_SAMPLE_METHOD,
#     IPNDM_SAMPLE_METHOD,
#     IPNDM_V_SAMPLE_METHOD,
#     LCM_SAMPLE_METHOD,
#     DDIM_TRAILING_SAMPLE_METHOD,
#     TCD_SAMPLE_METHOD,
#     RES_MULTISTEP_SAMPLE_METHOD,
#     RES_2S_SAMPLE_METHOD,
#     ER_SDE_SAMPLE_METHOD,
#     SAMPLE_METHOD_COUNT
# };
class SampleMethod(IntEnum):
    EULER_SAMPLE_METHOD = 0
    EULER_A_SAMPLE_METHOD = 1
    HEUN_SAMPLE_METHOD = 2
    DPM2_SAMPLE_METHOD = 3
    DPMPP2S_A_SAMPLE_METHOD = 4
    DPMPP2M_SAMPLE_METHOD = 5
    DPMPP2Mv2_SAMPLE_METHOD = 6
    IPNDM_SAMPLE_METHOD = 7
    IPNDM_V_SAMPLE_METHOD = 8
    LCM_SAMPLE_METHOD = 9
    DDIM_TRAILING_SAMPLE_METHOD = 10
    TCD_SAMPLE_METHOD = 11
    RES_MULTISTEP_SAMPLE_METHOD = 12
    RES_2S_SAMPLE_METHOD = 13
    ER_SDE_SAMPLE_METHOD = 14
    EULER_CFG_PP_SAMPLE_METHOD = 15
    EULER_A_CFG_PP_SAMPLE_METHOD = 16
    EULER_GE_SAMPLE_METHOD = 17
    DPMPP2M_SDE_SAMPLE_METHOD = 18
    DPMPP2M_SDE_BT_SAMPLE_METHOD = 19
    LMS_SAMPLE_METHOD = 20
    SAMPLE_METHOD_COUNT = 21


# enum scheduler_t {
#     DISCRETE_SCHEDULER,
#     KARRAS_SCHEDULER,
#     EXPONENTIAL_SCHEDULER,
#     AYS_SCHEDULER,
#     GITS_SCHEDULER,
#     SGM_UNIFORM_SCHEDULER,
#     SIMPLE_SCHEDULER,
#     SMOOTHSTEP_SCHEDULER,
#     KL_OPTIMAL_SCHEDULER,
#     LCM_SCHEDULER,
#     BONG_TANGENT_SCHEDULER,
#     LTX2_SCHEDULER,
#     LOGIT_NORMAL_SCHEDULER,
#     FLUX2_SCHEDULER,
#     FLUX_SCHEDULER,
#     BETA_SCHEDULER,
#     LLADA_IMAGE_SCHEDULER,
#     SCHEDULER_COUNT
# };
class Scheduler(IntEnum):
    DISCRETE_SCHEDULER = 0
    KARRAS_SCHEDULER = 1
    EXPONENTIAL_SCHEDULER = 2
    AYS_SCHEDULER = 3
    GITS_SCHEDULER = 4
    SGM_UNIFORM_SCHEDULER = 5
    SIMPLE_SCHEDULER = 6
    SMOOTHSTEP_SCHEDULER = 7
    KL_OPTIMAL_SCHEDULER = 8
    LCM_SCHEDULER = 9
    BONG_TANGENT_SCHEDULER = 10
    LTX2_SCHEDULER = 11
    LOGIT_NORMAL_SCHEDULER = 12
    FLUX2_SCHEDULER = 13
    FLUX_SCHEDULER = 14
    BETA_SCHEDULER = 15
    LLADA_IMAGE_SCHEDULER = 16
    SCHEDULER_COUNT = 17


# enum prediction_t {
#     EPS_PRED,
#     V_PRED,
#     EDM_V_PRED,
#     FLOW_PRED,
#     FLUX_FLOW_PRED,
#     SEFI_FLOW_PRED,
#     MINIT2I_FLOW_PRED,
#     SENSENOVA_U1_FLOW_PRED,
#     PREDICTION_COUNT
# };
class Prediction(IntEnum):
    EPS_PRED = 0
    V_PRED = 1
    EDM_V_PRED = 2
    FLOW_PRED = 3
    FLUX_FLOW_PRED = 4
    SEFI_FLOW_PRED = 5
    MINIT2I_FLOW_PRED = 6
    SENSENOVA_U1_FLOW_PRED = 7
    PREDICTION_COUNT = 8


# // same as enum ggml_type
# enum sd_type_t {
#     SD_TYPE_F32  = 0,
#     SD_TYPE_F16  = 1,
#     SD_TYPE_Q4_0 = 2,
#     SD_TYPE_Q4_1 = 3,
#     // SD_TYPE_Q4_2 = 4, support has been removed
#     // SD_TYPE_Q4_3 = 5, support has been removed
#     SD_TYPE_Q5_0    = 6,
#     SD_TYPE_Q5_1    = 7,
#     SD_TYPE_Q8_0    = 8,
#     SD_TYPE_Q8_1    = 9,
#     SD_TYPE_Q2_K    = 10,
#     SD_TYPE_Q3_K    = 11,
#     SD_TYPE_Q4_K    = 12,
#     SD_TYPE_Q5_K    = 13,
#     SD_TYPE_Q6_K    = 14,
#     SD_TYPE_Q8_K    = 15,
#     SD_TYPE_IQ2_XXS = 16,
#     SD_TYPE_IQ2_XS  = 17,
#     SD_TYPE_IQ3_XXS = 18,
#     SD_TYPE_IQ1_S   = 19,
#     SD_TYPE_IQ4_NL  = 20,
#     SD_TYPE_IQ3_S   = 21,
#     SD_TYPE_IQ2_S   = 22,
#     SD_TYPE_IQ4_XS  = 23,
#     SD_TYPE_I8      = 24,
#     SD_TYPE_I16     = 25,
#     SD_TYPE_I32     = 26,
#     SD_TYPE_I64     = 27,
#     SD_TYPE_F64     = 28,
#     SD_TYPE_IQ1_M   = 29,
#     SD_TYPE_BF16    = 30,
#     // SD_TYPE_Q4_0_4_4 = 31, support has been removed from gguf files
#     // SD_TYPE_Q4_0_4_8 = 32,
#     // SD_TYPE_Q4_0_8_8 = 33,
#     SD_TYPE_TQ1_0 = 34,
#     SD_TYPE_TQ2_0 = 35,
#     // SD_TYPE_IQ4_NL_4_4 = 36,
#     // SD_TYPE_IQ4_NL_4_8 = 37,
#     // SD_TYPE_IQ4_NL_8_8 = 38,
#     SD_TYPE_MXFP4   = 39,  // MXFP4 (1 block)
#     SD_TYPE_NVFP4   = 40,  // NVFP4 (4 blocks, E4M3 scale)
#     SD_TYPE_Q1_0    = 41,
#     SD_TYPE_Q2_0    = 42,
#     SD_TYPE_F8_E4M3 = 43,
#     SD_TYPE_F8_E5M2 = 44,
#     SD_TYPE_COUNT   = 45,
# };
class GGMLType(IntEnum):
    SD_TYPE_F32 = 0
    SD_TYPE_F16 = 1
    SD_TYPE_Q4_0 = 2
    SD_TYPE_Q4_1 = 3
    # SD_TYPE_Q4_2 = 4 support has been removed
    # SD_TYPE_Q4_3 = 5 support has been removed
    SD_TYPE_Q5_0 = 6
    SD_TYPE_Q5_1 = 7
    SD_TYPE_Q8_0 = 8
    SD_TYPE_Q8_1 = 9
    # // k-quantizations
    SD_TYPE_Q2_K = 10
    SD_TYPE_Q3_K = 11
    SD_TYPE_Q4_K = 12
    SD_TYPE_Q5_K = 13
    SD_TYPE_Q6_K = 14
    SD_TYPE_Q8_K = 15
    SD_TYPE_IQ2_XXS = 16
    SD_TYPE_IQ2_XS = 17
    SD_TYPE_IQ3_XXS = 18
    SD_TYPE_IQ1_S = 19
    SD_TYPE_IQ4_NL = 20
    SD_TYPE_IQ3_S = 21
    SD_TYPE_IQ2_S = 22
    SD_TYPE_IQ4_XS = 23
    SD_TYPE_I8 = 24
    SD_TYPE_I16 = 25
    SD_TYPE_I32 = 26
    SD_TYPE_I64 = 27
    SD_TYPE_F64 = 28
    SD_TYPE_IQ1_M = 29
    SD_TYPE_BF16 = 30
    # SD_TYPE_Q4_0_4_4 = 31 # support has been removed from gguf files
    # SD_TYPE_Q4_0_4_8 = 32
    # SD_TYPE_Q4_0_8_8 = 33
    SD_TYPE_TQ1_0 = 34
    SD_TYPE_TQ2_0 = 35
    # SD_TYPE_IQ4_NL_4_4 = 36,
    # SD_TYPE_IQ4_NL_4_8 = 37,
    # SD_TYPE_IQ4_NL_8_8 = 38,
    SD_TYPE_MXFP4 = 39  # MXFP4 (1 block)
    SD_TYPE_NVFP4 = 40  # NVFP4 (4 blocks, E4M3 scale)
    SD_TYPE_Q1_0 = 41
    SD_TYPE_Q2_0 = 42
    SD_TYPE_F8_E4M3 = 43
    SD_TYPE_F8_E5M2 = 44
    SD_TYPE_COUNT = 45


# enum sd_log_level_t {
#     SD_LOG_DEBUG,
#     SD_LOG_VERBOSE,
#     SD_LOG_INFO,
#     SD_LOG_WARN,
#     SD_LOG_ERROR
# };
class SDLogLevel(IntEnum):
    SD_LOG_DEBUG = 0
    SD_LOG_VERBOSE = 1
    SD_LOG_INFO = 2
    SD_LOG_WARN = 3
    SD_LOG_ERROR = 4


# enum sd_vae_format_t {
#     SD_VAE_FORMAT_AUTO = -1,
#     SD_VAE_FORMAT_FLUX,
#     SD_VAE_FORMAT_SD3,
#     SD_VAE_FORMAT_FLUX2,
#     SD_VAE_FORMAT_WAN,
#     SD_VAE_FORMAT_COUNT,
# };
class SDVAEFormat(IntEnum):
    SD_VAE_FORMAT_AUTO = -1
    SD_VAE_FORMAT_FLUX = 0
    SD_VAE_FORMAT_SD3 = 1
    SD_VAE_FORMAT_FLUX2 = 2
    SD_VAE_FORMAT_WAN = 3
    SD_VAE_FORMAT_COUNT = 4


# enum sd_cancel_mode_t {
#     SD_CANCEL_ALL,
#     SD_CANCEL_NEW_LATENTS,
#     SD_CANCEL_RESET
# };
class SDCancelMode(IntEnum):
    SD_CANCEL_ALL = 0
    SD_CANCEL_NEW_LATENTS = 1
    SD_CANCEL_RESET = 2


# enum preview_t {
#     PREVIEW_NONE,
#     PREVIEW_PROJ,
#     PREVIEW_TAE,
#     PREVIEW_VAE,
#     PREVIEW_COUNT
# };
class Preview(IntEnum):
    PREVIEW_NONE = 0
    PREVIEW_PROJ = 1
    PREVIEW_TAE = 2
    PREVIEW_VAE = 3
    PREVIEW_COUNT = 4


# enum lora_apply_mode_t {
#     LORA_APPLY_AUTO,
#     LORA_APPLY_IMMEDIATELY,
#     LORA_APPLY_AT_RUNTIME,
#     LORA_APPLY_MODE_COUNT,
# };
class LoraApplyMode(IntEnum):
    LORA_APPLY_AUTO = 0
    LORA_APPLY_IMMEDIATELY = 1
    LORA_APPLY_AT_RUNTIME = 2
    LORA_APPLY_MODE_COUNT = 3


# enum sd_cache_mode_t {
#     SD_CACHE_DISABLED = 0,
#     SD_CACHE_EASYCACHE,
#     SD_CACHE_UCACHE,
#     SD_CACHE_DBCACHE,
#     SD_CACHE_TAYLORSEER,
#     SD_CACHE_CACHE_DIT,
#     SD_CACHE_SPECTRUM,
# };
class SDCacheMode(IntEnum):
    SD_CACHE_DISABLED = 0
    SD_CACHE_EASYCACHE = 1
    SD_CACHE_UCACHE = 2
    SD_CACHE_DBCACHE = 3
    SD_CACHE_TAYLORSEER = 4
    SD_CACHE_CACHE_DIT = 5
    SD_CACHE_SPECTRUM = 6


# enum sd_hires_upscaler_t {
#     SD_HIRES_UPSCALER_NONE,
#     SD_HIRES_UPSCALER_LATENT,
#     SD_HIRES_UPSCALER_LATENT_NEAREST,
#     SD_HIRES_UPSCALER_LATENT_NEAREST_EXACT,
#     SD_HIRES_UPSCALER_LATENT_ANTIALIASED,
#     SD_HIRES_UPSCALER_LATENT_BICUBIC,
#     SD_HIRES_UPSCALER_LATENT_BICUBIC_ANTIALIASED,
#     SD_HIRES_UPSCALER_LANCZOS,
#     SD_HIRES_UPSCALER_NEAREST,
#     SD_HIRES_UPSCALER_MODEL,
#     SD_HIRES_UPSCALER_COUNT,
# };
class SDHiresUpscaler(IntEnum):
    SD_HIRES_UPSCALER_NONE = 0
    SD_HIRES_UPSCALER_LATENT = 1
    SD_HIRES_UPSCALER_LATENT_NEAREST = 2
    SD_HIRES_UPSCALER_LATENT_NEAREST_EXACT = 3
    SD_HIRES_UPSCALER_LATENT_ANTIALIASED = 4
    SD_HIRES_UPSCALER_LATENT_BICUBIC = 5
    SD_HIRES_UPSCALER_LATENT_BICUBIC_ANTIALIASED = 6
    SD_HIRES_UPSCALER_LANCZOS = 7
    SD_HIRES_UPSCALER_NEAREST = 8
    SD_HIRES_UPSCALER_MODEL = 9
    SD_HIRES_UPSCALER_COUNT = 10


# ===========================================
# Inference
# ===========================================


# -------------------------------------------
# sd_embedding_t
# -------------------------------------------


# typedef struct { const char* name; const char* path; } sd_embedding_t;
class sd_embedding_t(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char_p),
        ("path", ctypes.c_char_p),
    ]


# -------------------------------------------
# sd_ctx_params_t
# -------------------------------------------


class sd_ctx_params_t(ctypes.Structure):
    _fields_ = [
        ("model_path", ctypes.c_char_p),
        ("clip_l_path", ctypes.c_char_p),
        ("clip_g_path", ctypes.c_char_p),
        ("clip_vision_path", ctypes.c_char_p),
        ("t5xxl_path", ctypes.c_char_p),
        ("llm_path", ctypes.c_char_p),
        ("llm_vision_path", ctypes.c_char_p),
        ("diffusion_model_path", ctypes.c_char_p),
        ("high_noise_diffusion_model_path", ctypes.c_char_p),
        ("uncond_diffusion_model_path", ctypes.c_char_p),
        ("embeddings_connectors_path", ctypes.c_char_p),
        ("vae_path", ctypes.c_char_p),
        ("audio_vae_path", ctypes.c_char_p),
        ("audio_encoder_path", ctypes.c_char_p),
        ("taesd_path", ctypes.c_char_p),
        ("control_net_path", ctypes.c_char_p),
        ("ip_adapter_path", ctypes.c_char_p),
        ("motion_module_path", ctypes.c_char_p),
        ("embeddings", ctypes.POINTER(sd_embedding_t)),
        ("embedding_count", ctypes.c_uint32),
        ("photo_maker_path", ctypes.c_char_p),
        ("pulid_weights_path", ctypes.c_char_p),
        ("tensor_type_rules", ctypes.c_char_p),
        ("n_threads", ctypes.c_int),
        ("wtype", ctypes.c_int),  # GGMLType
        ("rng_type", ctypes.c_int),  # RNGType
        ("sampler_rng_type", ctypes.c_int),  # RNGType
        ("prediction", ctypes.c_int),  # Prediction
        ("lora_apply_mode", ctypes.c_int),  # LoraApplyMode
        ("enable_mmap", ctypes.c_bool),
        ("flash_attn", ctypes.c_bool),
        ("diffusion_flash_attn", ctypes.c_bool),
        ("tae_preview_only", ctypes.c_bool),
        ("diffusion_conv_direct", ctypes.c_bool),
        ("vae_conv_direct", ctypes.c_bool),
        ("force_sdxl_vae_conv_scale", ctypes.c_bool),
        ("vae_format", ctypes.c_int),  # SDVAEFormat
        ("max_vram", ctypes.c_char_p),
        ("disable_prefetch", ctypes.c_bool),
        ("eager_load", ctypes.c_bool),
        ("backend", ctypes.c_char_p),
        ("params_backend", ctypes.c_char_p),
        ("split_mode", ctypes.c_char_p),
        ("auto_fit", ctypes.c_bool),
        ("rpc_servers", ctypes.c_char_p),
        ("model_args", ctypes.c_char_p),
        ("disable_segmented_compute", ctypes.c_bool),
        ("linear_scale", ctypes.c_float),
        ("attn_scale", ctypes.c_float),
        ("tokenizer", ctypes.c_char_p),
        ("sage_attn", ctypes.c_bool),
        ("conditioning_cache_size", ctypes.c_int),
    ]


# -------------------------------------------
# sd_ctx_params_init
# -------------------------------------------


# SD_API void sd_ctx_params_init(sd_ctx_params_t* sd_ctx_params);
@ctypes_function(
    "sd_ctx_params_init",
    [
        ctypes.POINTER(sd_ctx_params_t),  # sd_ctx_params
    ],
    None,
)
def sd_ctx_params_init(
    sd_ctx_params: sd_ctx_params_t,
    /,
) -> None: ...


# -------------------------------------------
# sd_ctx_t
# -------------------------------------------


# typedef struct sd_ctx_t sd_ctx_t;
class sd_ctx_t(ctypes.Structure):
    pass


# struct sd_ctx;
sd_ctx_t_p = NewType("sd_ctx_t_p", int)
sd_ctx_t_p_ctypes = ctypes.POINTER(sd_ctx_t)


# -------------------------------------------
# new_sd_ctx
# -------------------------------------------


# SD_API sd_ctx_t* new_sd_ctx(const sd_ctx_params_t* sd_ctx_params);
@ctypes_function(
    "new_sd_ctx",
    [
        ctypes.POINTER(sd_ctx_params_t),  # sd_ctx_params
    ],
    sd_ctx_t_p_ctypes,
)
def new_sd_ctx(
    sd_ctx_params: sd_ctx_params_t,
    /,
) -> Optional[sd_ctx_t_p]: ...


# -------------------------------------------
# free_sd_ctx
# -------------------------------------------


# SD_API void free_sd_ctx(sd_ctx_t* sd_ctx);
@ctypes_function(
    "free_sd_ctx",
    [
        sd_ctx_t_p_ctypes,  # sd_ctx
    ],
    None,
)
def free_sd_ctx(
    sd_ctx: sd_ctx_t_p,
    /,
): ...


# -------------------------------------------
# sd_audio_t
# -------------------------------------------


# typedef struct { uint32_t sample_rate; uint32_t channels; uint64_t sample_count; float* data; } sd_audio_t;
class sd_audio_t(ctypes.Structure):
    _fields_ = [
        ("sample_rate", ctypes.c_uint32),
        ("channels", ctypes.c_uint32),
        ("sample_count", ctypes.c_uint64),
        ("data", ctypes.POINTER(ctypes.c_float)),
    ]


# -------------------------------------------
# free_sd_audio
# -------------------------------------------


# SD_API void free_sd_audio(sd_audio_t* audio);
@ctypes_function(
    "free_sd_audio",
    [
        ctypes.POINTER(sd_audio_t),  # audio
    ],
    None,
)
def free_sd_audio(
    audio: ctypes.POINTER(sd_audio_t),
    /,
) -> None: ...


# -------------------------------------------
# sd_image_t
# -------------------------------------------


# typedef struct { uint32_t width; uint32_t height; uint32_t channel; uint8_t* data; } sd_image_t;
class sd_image_t(ctypes.Structure):
    _fields_ = [
        ("width", ctypes.c_uint32),
        ("height", ctypes.c_uint32),
        ("channel", ctypes.c_uint32),
        ("data", ctypes.POINTER(ctypes.c_uint8)),
    ]


# -------------------------------------------
# sd_pm_params_t
# -------------------------------------------


# typedef struct { sd_image_t* id_images; int id_images_count; const char* id_embed_path; float style_strength; } sd_pm_params_t;  // photo maker
class sd_pm_params_t(ctypes.Structure):
    _fields_ = [
        ("id_images", ctypes.POINTER(sd_image_t)),
        ("id_images_count", ctypes.c_int),
        ("id_embed_path", ctypes.c_char_p),
        ("style_strength", ctypes.c_float),
    ]  # photo maker


# -------------------------------------------
# sd_tiling_params_t
# -------------------------------------------


class sd_tiling_params_t(ctypes.Structure):
    _fields_ = [
        ("enabled", ctypes.c_bool),
        ("temporal_tiling", ctypes.c_bool),
        ("tile_size_w", ctypes.c_int),
        ("tile_size_h", ctypes.c_int),
        ("target_overlap", ctypes.c_float),
        ("rel_size_w", ctypes.c_float),
        ("rel_size_h", ctypes.c_float),
        ("extra_tiling_args", ctypes.c_char_p),
    ]


# -------------------------------------------
# sd_image_preprocess_params_t
# -------------------------------------------


class sd_image_preprocess_params_t(ctypes.Structure):
    _fields_ = [
        ("rules", ctypes.c_char_p),
    ]


# -------------------------------------------
# sd_ref_video_t
# -------------------------------------------


class sd_ref_video_t(ctypes.Structure):
    _fields_ = [
        ("frames", ctypes.POINTER(sd_image_t)),
        ("frame_count", ctypes.c_int),
        ("fps", ctypes.c_int),
        ("audio", sd_audio_t),
    ]


# -------------------------------------------
# sd_slg_params_t
# -------------------------------------------


# typedef struct { int* layers; size_t layer_count; float layer_start; float layer_end; float scale; } sd_slg_params_t;
class sd_slg_params_t(ctypes.Structure):
    _fields_ = [
        ("layers", ctypes.POINTER(ctypes.c_int)),
        ("layer_count", ctypes.c_size_t),
        ("layer_start", ctypes.c_float),
        ("layer_end", ctypes.c_float),
        ("scale", ctypes.c_float),
    ]


# -------------------------------------------
# sd_guidance_params_t
# -------------------------------------------


# typedef struct { float txt_cfg; float img_cfg; float distilled_guidance; sd_slg_params_t slg; } sd_guidance_params_t;
class sd_guidance_params_t(ctypes.Structure):
    _fields_ = [
        ("txt_cfg", ctypes.c_float),
        ("img_cfg", ctypes.c_float),
        ("distilled_guidance", ctypes.c_float),
        ("slg", sd_slg_params_t),
    ]


# -------------------------------------------
# sd_sample_params_t
# -------------------------------------------


class sd_sample_params_t(ctypes.Structure):
    _fields_ = [
        ("guidance", sd_guidance_params_t),
        ("scheduler", ctypes.c_int),  # Scheduler
        ("sample_method", ctypes.c_int),  # SampleMethod
        ("sample_steps", ctypes.c_int),
        ("eta", ctypes.c_float),
        ("shifted_timestep", ctypes.c_int),
        ("custom_sigmas", ctypes.POINTER(ctypes.c_float)),
        ("custom_sigmas_count", ctypes.c_int),
        ("flow_shift", ctypes.c_float),
        ("extra_sample_args", ctypes.c_char_p),
    ]


# -------------------------------------------
# sd_pulid_params_t
# -------------------------------------------


class sd_pulid_params_t(ctypes.Structure):
    _fields_ = [
        ("id_embedding_path", ctypes.c_char_p),
        ("id_weight", ctypes.c_float),
    ]


# -------------------------------------------
# sd_cache_params_t
# -------------------------------------------


# typedef struct { enum sd_cache_mode_t mode; float reuse_threshold; float start_percent; float end_percent; float error_decay_rate; bool use_relative_threshold; bool reset_error_on_compute; int Fn_compute_blocks; int Bn_compute_blocks; float residual_diff_threshold; int max_warmup_steps; int max_cached_steps; int max_continuous_cached_steps; int taylorseer_n_derivatives; int taylorseer_skip_interval; const char* scm_mask; bool scm_policy_dynamic; float spectrum_w; int spectrum_m; float spectrum_lam; int spectrum_window_size; float spectrum_flex_window; int spectrum_warmup_steps; float spectrum_stop_percent; } sd_cache_params_t;
class sd_cache_params_t(ctypes.Structure):
    _fields_ = [
        ("mode", ctypes.c_int),  # SDCacheMode
        ("reuse_threshold", ctypes.c_float),
        ("start_percent", ctypes.c_float),
        ("end_percent", ctypes.c_float),
        ("error_decay_rate", ctypes.c_float),
        ("use_relative_threshold", ctypes.c_bool),
        ("reset_error_on_compute", ctypes.c_bool),
        ("Fn_compute_blocks", ctypes.c_int),
        ("Bn_compute_blocks", ctypes.c_int),
        ("residual_diff_threshold", ctypes.c_float),
        ("max_warmup_steps", ctypes.c_int),
        ("max_cached_steps", ctypes.c_int),
        ("max_continuous_cached_steps", ctypes.c_int),
        ("taylorseer_n_derivatives", ctypes.c_int),
        ("taylorseer_skip_interval", ctypes.c_int),
        ("scm_mask", ctypes.c_char_p),
        ("scm_policy_dynamic", ctypes.c_bool),
        ("spectrum_w", ctypes.c_float),
        ("spectrum_m", ctypes.c_int),
        ("spectrum_lam", ctypes.c_float),
        ("spectrum_window_size", ctypes.c_int),
        ("spectrum_flex_window", ctypes.c_float),
        ("spectrum_warmup_steps", ctypes.c_int),
        ("spectrum_stop_percent", ctypes.c_float),
    ]


# -------------------------------------------
# sd_lora_t
# -------------------------------------------


# typedef struct { bool is_high_noise; float multiplier; const char* path; } sd_lora_t;
class sd_lora_t(ctypes.Structure):
    _fields_ = [
        ("is_high_noise", ctypes.c_bool),
        ("multiplier", ctypes.c_float),
        ("path", ctypes.c_char_p),
    ]


# -------------------------------------------
# sd_hires_params_t
# -------------------------------------------


class sd_hires_params_t(ctypes.Structure):
    _fields_ = [
        ("enabled", ctypes.c_bool),
        ("upscaler", ctypes.c_int),  # SDHiresUpscaler
        ("model_path", ctypes.c_char_p),
        ("scale", ctypes.c_float),
        ("target_width", ctypes.c_int),
        ("target_height", ctypes.c_int),
        ("steps", ctypes.c_int),
        ("denoising_strength", ctypes.c_float),
        ("upscale_tile_size", ctypes.c_int),
        ("custom_sigmas", ctypes.POINTER(ctypes.c_float)),
        ("custom_sigmas_count", ctypes.c_int),
    ]


# -------------------------------------------
# sd_img_gen_params_t
# -------------------------------------------


class sd_img_gen_params_t(ctypes.Structure):
    _fields_ = [
        ("loras", ctypes.POINTER(sd_lora_t)),
        ("lora_count", ctypes.c_uint32),
        ("prompt", ctypes.c_char_p),
        ("negative_prompt", ctypes.c_char_p),
        ("clip_skip", ctypes.c_int),
        ("init_image", sd_image_t),
        ("ref_images", ctypes.POINTER(sd_image_t)),
        ("ref_images_count", ctypes.c_int),
        ("ref_image_args", ctypes.c_char_p),
        ("mask_image", sd_image_t),
        ("width", ctypes.c_int),
        ("height", ctypes.c_int),
        ("sample_params", sd_sample_params_t),
        ("strength", ctypes.c_float),
        ("seed", ctypes.c_int64),
        ("batch_count", ctypes.c_int),
        ("control_image", sd_image_t),
        ("control_strength", ctypes.c_float),
        ("ip_adapter_image", sd_image_t),
        ("ip_adapter_strength", ctypes.c_float),
        ("pm_params", sd_pm_params_t),
        ("pulid_params", sd_pulid_params_t),
        ("vae_tiling_params", sd_tiling_params_t),
        ("cache", sd_cache_params_t),
        ("hires", sd_hires_params_t),
        ("qwen_image_layers", ctypes.c_int),
        ("circular_x", ctypes.c_bool),
        ("circular_y", ctypes.c_bool),
        ("image_preprocess", sd_image_preprocess_params_t),
    ]


# -------------------------------------------
# generate_image
# -------------------------------------------


# SD_API bool generate_image(sd_ctx_t* sd_ctx, const sd_img_gen_params_t* sd_img_gen_params, sd_image_t** images_out, int* num_images_out);
@ctypes_function(
    "generate_image",
    [
        sd_ctx_t_p_ctypes,  # sd_ctx
        ctypes.POINTER(sd_img_gen_params_t),  # sd_img_gen_params
        ctypes.POINTER(ctypes.POINTER(sd_image_t)),  # images_out
        ctypes.POINTER(ctypes.c_int),  # num_images_out
    ],
    ctypes.c_bool,
)
def generate_image(
    sd_ctx: sd_ctx_t_p,
    sd_img_gen_params: sd_img_gen_params_t,
    images_out: ctypes.POINTER(ctypes.POINTER(sd_image_t)),
    num_images_out: ctypes.POINTER(ctypes.c_int),
    /,
) -> bool: ...


# -------------------------------------------
# free_sd_images
# -------------------------------------------


# SD_API void free_sd_images(sd_image_t* result_images, int num_images);
@ctypes_function(
    "free_sd_images",
    [
        ctypes.POINTER(sd_image_t),  # result_images
        ctypes.c_int,  # num_images
    ],
    None,
)
def free_sd_images(
    result_images: ctypes.POINTER(sd_image_t),
    num_images: int,
    /,
) -> None: ...


# -------------------------------------------
# sd_vid_gen_params_t
# -------------------------------------------


class sd_vid_gen_params_t(ctypes.Structure):
    _fields_ = [
        ("loras", ctypes.POINTER(sd_lora_t)),
        ("lora_count", ctypes.c_uint32),
        ("prompt", ctypes.c_char_p),
        ("negative_prompt", ctypes.c_char_p),
        ("clip_skip", ctypes.c_int),
        ("init_image", sd_image_t),
        ("end_image", sd_image_t),
        ("ref_images", ctypes.POINTER(sd_image_t)),
        ("ref_images_count", ctypes.c_int),
        ("ref_videos", ctypes.POINTER(sd_ref_video_t)),
        ("ref_videos_count", ctypes.c_int),
        ("ref_audios", ctypes.POINTER(sd_audio_t)),
        ("ref_audios_count", ctypes.c_int),
        ("control_frames", ctypes.POINTER(sd_image_t)),
        ("control_frames_size", ctypes.c_int),
        ("width", ctypes.c_int),
        ("height", ctypes.c_int),
        ("sample_params", sd_sample_params_t),
        ("high_noise_sample_params", sd_sample_params_t),
        ("moe_boundary", ctypes.c_float),
        ("strength", ctypes.c_float),
        ("seed", ctypes.c_int64),
        ("video_frames", ctypes.c_int),
        ("fps", ctypes.c_int),
        ("vace_strength", ctypes.c_float),
        ("vae_tiling_params", sd_tiling_params_t),
        ("cache", sd_cache_params_t),
        ("hires", sd_hires_params_t),
        ("circular_x", ctypes.c_bool),
        ("circular_y", ctypes.c_bool),
        ("image_preprocess", sd_image_preprocess_params_t),
    ]


# -------------------------------------------
# generate_video
# -------------------------------------------


# SD_API bool generate_video(sd_ctx_t* sd_ctx, const sd_vid_gen_params_t* sd_vid_gen_params, sd_image_t** frames_out, int* num_frames_out, sd_audio_t** audio_out, int* fps_out);
@ctypes_function(
    "generate_video",
    [
        sd_ctx_t_p_ctypes,  # sd_ctx
        ctypes.POINTER(sd_vid_gen_params_t),  # sd_vid_gen_params
        ctypes.POINTER(ctypes.POINTER(sd_image_t)),  # frames_out
        ctypes.POINTER(ctypes.c_int),  # num_frames_out
        ctypes.POINTER(ctypes.POINTER(sd_audio_t)),  # audio_out
        ctypes.POINTER(ctypes.c_int),  # fps_out
    ],
    ctypes.c_bool,
)
def generate_video(
    sd_ctx: sd_ctx_t_p,
    sd_vid_gen_params: sd_vid_gen_params_t,
    frames_out: ctypes.POINTER(ctypes.POINTER(sd_image_t)),
    num_frames_out: ctypes.POINTER(ctypes.c_int),
    audio_out: ctypes.POINTER(ctypes.POINTER(sd_audio_t)),
    fps_out: ctypes.POINTER(ctypes.c_int),
    /,
) -> bool: ...


# -------------------------------------------
# sd_cancel_generation
# -------------------------------------------


# SD_API void sd_cancel_generation(sd_ctx_t* sd_ctx, enum sd_cancel_mode_t mode);
@ctypes_function(
    "sd_cancel_generation",
    [
        sd_ctx_t_p_ctypes,  # sd_ctx
        ctypes.c_int,  # mode
    ],
    None,
)
def sd_cancel_generation(
    sd_ctx: sd_ctx_t_p,
    mode: int,
    /,
) -> None: ...


# -------------------------------------------
# sd_get_model_version_name
# -------------------------------------------


# SD_API const char* sd_get_model_version_name(const sd_ctx_t* sd_ctx);
@ctypes_function(
    "sd_get_model_version_name",
    [
        sd_ctx_t_p_ctypes,  # sd_ctx
    ],
    ctypes.c_char_p,
)
def sd_get_model_version_name(
    sd_ctx: sd_ctx_t_p,
    /,
) -> bytes: ...


# -------------------------------------------
# sd_get_default_sample_method
# -------------------------------------------


# SD_API enum sample_method_t sd_get_default_sample_method(const sd_ctx_t* sd_ctx);
@ctypes_function(
    "sd_get_default_sample_method",
    [
        sd_ctx_t_p_ctypes,  # sd_ctx
    ],
    ctypes.c_int,  # SampleMethod
)
def sd_get_default_sample_method(
    sd_ctx: sd_ctx_t_p,
    /,
) -> Optional[SampleMethod]: ...


# -------------------------------------------
# sd_get_default_scheduler
# -------------------------------------------


# SD_API enum scheduler_t sd_get_default_scheduler(const sd_ctx_t* sd_ctx, enum sample_method_t sample_method);
@ctypes_function(
    "sd_get_default_scheduler",
    [
        sd_ctx_t_p_ctypes,  # sd_ctx
        ctypes.c_int,  # sample_method
    ],
    ctypes.c_int,  # Scheduler
)
def sd_get_default_scheduler(
    sd_ctx: sd_ctx_t_p,
    sample_method: SampleMethod,
    /,
) -> Optional[Scheduler]: ...


# -------------------------------------------
# sd_ctx ControlNet hot-swap APIs
# -------------------------------------------


# SD_API bool sd_ctx_load_control_net(sd_ctx_t* sd_ctx, const char* path);
@ctypes_function(
    "sd_ctx_load_control_net",
    [
        sd_ctx_t_p_ctypes,  # sd_ctx
        ctypes.c_char_p,  # path
    ],
    ctypes.c_bool,
)
def sd_ctx_load_control_net(
    sd_ctx: sd_ctx_t_p,
    path: bytes,
    /,
) -> bool: ...


# SD_API bool sd_ctx_unload_control_net(sd_ctx_t* sd_ctx);
@ctypes_function(
    "sd_ctx_unload_control_net",
    [
        sd_ctx_t_p_ctypes,  # sd_ctx
    ],
    ctypes.c_bool,
)
def sd_ctx_unload_control_net(
    sd_ctx: sd_ctx_t_p,
    /,
) -> bool: ...


# SD_API bool sd_ctx_has_control_net(const sd_ctx_t* sd_ctx);
@ctypes_function(
    "sd_ctx_has_control_net",
    [
        sd_ctx_t_p_ctypes,  # sd_ctx
    ],
    ctypes.c_bool,
)
def sd_ctx_has_control_net(
    sd_ctx: sd_ctx_t_p,
    /,
) -> bool: ...


# -------------------------------------------
# upscaler_ctx_t
# -------------------------------------------


# typedef struct upscaler_ctx_t upscaler_ctx_t;
class upscaler_ctx_t(ctypes.Structure):
    pass


# struct upscaler_ctx;
upscaler_ctx_t_p = NewType("upscaler_ctx_t_p", int)
upscaler_ctx_t_p_ctypes = ctypes.POINTER(upscaler_ctx_t)


# -------------------------------------------
# new_upscaler_ctx
# -------------------------------------------


# SD_API upscaler_ctx_t* new_upscaler_ctx(const char* esrgan_path, bool direct, int n_threads, int tile_size, const char* backend, const char* params_backend);
@ctypes_function(
    "new_upscaler_ctx",
    [
        ctypes.c_char_p,  # esrgan_path
        ctypes.c_bool,  # direct
        ctypes.c_int,  # n_threads
        ctypes.c_int,  # tile_size
        ctypes.c_char_p,  # backend
        ctypes.c_char_p,  # params_backend
    ],
    upscaler_ctx_t_p_ctypes,
)
def new_upscaler_ctx(
    esrgan_path: bytes,
    direct: bool,
    n_threads: int,
    tile_size: int,
    backend: Optional[bytes] = None,
    params_backend: Optional[bytes] = None,
    /,
) -> upscaler_ctx_t_p: ...


# -------------------------------------------
# free_upscaler_ctx
# -------------------------------------------


# SD_API void free_upscaler_ctx(upscaler_ctx_t* upscaler_ctx);
@ctypes_function(
    "free_upscaler_ctx",
    [
        upscaler_ctx_t_p_ctypes,  # upscaler_ctx
    ],
    None,
)
def free_upscaler_ctx(
    upscaler_ctx: upscaler_ctx_t_p,
    /,
) -> None: ...


# -------------------------------------------
# upscale
# -------------------------------------------


# SD_API bool upscale(upscaler_ctx_t* upscaler_ctx, sd_image_t input_image, uint32_t upscale_factor, sd_image_t** images_out, int* num_images_out);
@ctypes_function(
    "upscale",
    [
        upscaler_ctx_t_p_ctypes,  # upscaler_ctx
        sd_image_t,  # input_image
        ctypes.c_uint32,  # upscale_factor
        ctypes.POINTER(ctypes.POINTER(sd_image_t)),  # images_out
        ctypes.POINTER(ctypes.c_int),  # num_images_out
    ],
    ctypes.c_bool,
)
def upscale(
    upscaler_ctx: upscaler_ctx_t_p,
    input_image: sd_image_t,
    upscale_factor: int,
    images_out: ctypes.POINTER(ctypes.POINTER(sd_image_t)),
    num_images_out: ctypes.POINTER(ctypes.c_int),
    /,
) -> bool: ...


# -------------------------------------------
# get_upscale_factor
# -------------------------------------------


# SD_API int get_upscale_factor(upscaler_ctx_t* upscaler_ctx);
@ctypes_function(
    "get_upscale_factor",
    [
        upscaler_ctx_t_p_ctypes,  # upscaler_ctx
    ],
    ctypes.c_int,
)
def get_upscale_factor(
    upscaler_ctx: upscaler_ctx_t_p,
    /,
) -> int: ...


# -------------------------------------------
# get_upscaler_model_scale
# -------------------------------------------


# SD_API int get_upscaler_model_scale(const char* model_path);
@ctypes_function(
    "get_upscaler_model_scale",
    [
        ctypes.c_char_p,  # model_path
    ],
    ctypes.c_int,
)
def get_upscaler_model_scale(
    model_path: bytes,
    /,
) -> int: ...


# -------------------------------------------
# ADetailer
# -------------------------------------------


class adetailer_ctx_t(ctypes.Structure):
    pass


adetailer_ctx_t_p = NewType("adetailer_ctx_t_p", int)
adetailer_ctx_t_p_ctypes = ctypes.POINTER(adetailer_ctx_t)


class sd_adetailer_params_t(ctypes.Structure):
    _fields_ = [
        ("prompt", ctypes.c_char_p),
        ("negative_prompt", ctypes.c_char_p),
        ("extra_ad_args", ctypes.c_char_p),
    ]


# SD_API adetailer_ctx_t* new_adetailer_ctx(const char* detector_path, int n_threads, const char* backend, const char* params_backend);
@ctypes_function(
    "new_adetailer_ctx",
    [
        ctypes.c_char_p,  # detector_path
        ctypes.c_int,  # n_threads
        ctypes.c_char_p,  # backend
        ctypes.c_char_p,  # params_backend
    ],
    adetailer_ctx_t_p_ctypes,
)
def new_adetailer_ctx(
    detector_path: bytes,
    n_threads: int,
    backend: Optional[bytes] = None,
    params_backend: Optional[bytes] = None,
    /,
) -> adetailer_ctx_t_p: ...


# SD_API void free_adetailer_ctx(adetailer_ctx_t* adetailer_ctx);
@ctypes_function(
    "free_adetailer_ctx",
    [
        adetailer_ctx_t_p_ctypes,  # adetailer_ctx
    ],
    None,
)
def free_adetailer_ctx(
    adetailer_ctx: adetailer_ctx_t_p,
    /,
) -> None: ...


# SD_API bool adetail_image(adetailer_ctx_t* adetailer_ctx, sd_ctx_t* sd_ctx, sd_image_t input_image, const sd_adetailer_params_t* adetailer_params, const sd_img_gen_params_t* inpaint_params, sd_image_t** images_out, int* num_images_out);
@ctypes_function(
    "adetail_image",
    [
        adetailer_ctx_t_p_ctypes,  # adetailer_ctx
        sd_ctx_t_p_ctypes,  # sd_ctx
        sd_image_t,  # input_image
        ctypes.POINTER(sd_adetailer_params_t),  # adetailer_params
        ctypes.POINTER(sd_img_gen_params_t),  # inpaint_params
        ctypes.POINTER(ctypes.POINTER(sd_image_t)),  # images_out
        ctypes.POINTER(ctypes.c_int),  # num_images_out
    ],
    ctypes.c_bool,
)
def adetail_image(
    adetailer_ctx: adetailer_ctx_t_p,
    sd_ctx: sd_ctx_t_p,
    input_image: sd_image_t,
    adetailer_params: sd_adetailer_params_t,
    inpaint_params: sd_img_gen_params_t,
    images_out: ctypes.POINTER(ctypes.POINTER(sd_image_t)),
    num_images_out: ctypes.POINTER(ctypes.c_int),
    /,
) -> bool: ...


# -------------------------------------------
# convert
# -------------------------------------------


# SD_API bool convert(const char* input_path, const char* vae_path, const char* output_path, enum sd_type_t output_type, const char* tensor_type_rules, bool convert_name);
@ctypes_function(
    "convert",
    [
        ctypes.c_char_p,  # input_path
        ctypes.c_char_p,  # vae_path
        ctypes.c_char_p,  # output_path
        ctypes.c_int,  # output_type
        ctypes.c_char_p,  # tensor_type_rules
        ctypes.c_bool,  # convert_name
    ],
    ctypes.c_bool,
)
def convert(
    input_path: bytes,
    vae_path: bytes,
    output_path: bytes,
    output_type: int,
    tensor_type_rules: bytes,
    convert_name: bool,
    /,
) -> bool: ...


# -------------------------------------------
# convert_with_components
# -------------------------------------------


# SD_API bool convert_with_components(const char* model_path, const char* clip_l_path, const char* clip_g_path, const char* t5xxl_path, const char* diffusion_model_path, const char* vae_path, const char* output_path, enum sd_type_t output_type, const char* tensor_type_rules, bool convert_name, int n_threads);
@ctypes_function(
    "convert_with_components",
    [
        ctypes.c_char_p,  # model_path
        ctypes.c_char_p,  # clip_l_path
        ctypes.c_char_p,  # clip_g_path
        ctypes.c_char_p,  # t5xxl_path
        ctypes.c_char_p,  # diffusion_model_path
        ctypes.c_char_p,  # vae_path
        ctypes.c_char_p,  # output_path
        ctypes.c_int,  # output_type
        ctypes.c_char_p,  # tensor_type_rules
        ctypes.c_bool,  # convert_name
        ctypes.c_int,  # n_threads
    ],
    ctypes.c_bool,
)
def convert_with_components(
    model_path: bytes,
    clip_l_path: bytes,
    clip_g_path: bytes,
    t5xxl_path: bytes,
    diffusion_model_path: bytes,
    vae_path: bytes,
    output_path: bytes,
    output_type: int,
    tensor_type_rules: bytes,
    convert_name: bool,
    n_threads: int,
    /,
) -> bool: ...


# -------------------------------------------
# preprocess_canny
# -------------------------------------------


# SD_API bool preprocess_canny(sd_image_t image, float high_threshold, float low_threshold, float weak, float strong, bool inverse);
@ctypes_function(
    "preprocess_canny",
    [
        sd_image_t,  # image
        ctypes.c_float,  # high_threshold
        ctypes.c_float,  # low_threshold
        ctypes.c_float,  # weak
        ctypes.c_float,  # strong
        ctypes.c_bool,  # inverse
    ],
    ctypes.c_bool,
)
def preprocess_canny(
    image: sd_image_t,
    high_threshold: float,
    low_threshold: float,
    weak: float,
    strong: float,
    inverse: bool,
    /,
) -> bool: ...


# -------------------------------------------
# IMatrix APIs
# -------------------------------------------


# SD_API bool load_imatrix(const char* imatrix_path);
@ctypes_function(
    "load_imatrix",
    [
        ctypes.c_char_p,
    ],
    ctypes.c_bool,
)
def load_imatrix(imatrix_path: bytes, /) -> bool: ...


# SD_API void save_imatrix(const char* imatrix_path);
@ctypes_function(
    "save_imatrix",
    [
        ctypes.c_char_p,
    ],
    None,
)
def save_imatrix(imatrix_path: bytes, /) -> None: ...


# SD_API void enable_imatrix_collection(void);
@ctypes_function(
    "enable_imatrix_collection",
    [],
    None,
)
def enable_imatrix_collection() -> None: ...


# SD_API void disable_imatrix_collection(void);
@ctypes_function(
    "disable_imatrix_collection",
    [],
    None,
)
def disable_imatrix_collection() -> None: ...


# -------------------------------------------
# Device Listing
# -------------------------------------------


# SD_API size_t sd_list_devices(char* buffer, size_t buffer_size);
@ctypes_function(
    "sd_list_devices",
    [
        ctypes.c_char_p,
        ctypes.c_size_t,
    ],
    ctypes.c_size_t,
)
def sd_list_devices(buffer: Optional[ctypes.c_char_p], buffer_size: int, /) -> int: ...


# -------------------------------------------
# Parameter Inits
# -------------------------------------------


# SD_API void sd_sample_params_init(sd_sample_params_t* sample_params);
@ctypes_function(
    "sd_sample_params_init",
    [
        ctypes.POINTER(sd_sample_params_t),
    ],
    None,
)
def sd_sample_params_init(sample_params: sd_sample_params_t, /) -> None: ...


# SD_API void sd_img_gen_params_init(sd_img_gen_params_t* sd_img_gen_params);
@ctypes_function(
    "sd_img_gen_params_init",
    [
        ctypes.POINTER(sd_img_gen_params_t),
    ],
    None,
)
def sd_img_gen_params_init(sd_img_gen_params: sd_img_gen_params_t, /) -> None: ...


# SD_API void sd_vid_gen_params_init(sd_vid_gen_params_t* sd_vid_gen_params);
@ctypes_function(
    "sd_vid_gen_params_init",
    [
        ctypes.POINTER(sd_vid_gen_params_t),
    ],
    None,
)
def sd_vid_gen_params_init(sd_vid_gen_params: sd_vid_gen_params_t, /) -> None: ...


# SD_API void sd_cache_params_init(sd_cache_params_t* cache_params);
@ctypes_function(
    "sd_cache_params_init",
    [
        ctypes.POINTER(sd_cache_params_t),
    ],
    None,
)
def sd_cache_params_init(cache_params: sd_cache_params_t, /) -> None: ...


# SD_API void sd_hires_params_init(sd_hires_params_t* hires_params);
@ctypes_function(
    "sd_hires_params_init",
    [
        ctypes.POINTER(sd_hires_params_t),
    ],
    None,
)
def sd_hires_params_init(hires_params: sd_hires_params_t, /) -> None: ...


# ===========================================
# SD Context Information
# ===========================================

# -------------------------------------------
# sd_ctx_supports_image_generation
# -------------------------------------------


# SD_API bool sd_ctx_supports_image_generation(const sd_ctx_t* sd_ctx);
@ctypes_function(
    "sd_ctx_supports_image_generation",
    [
        sd_ctx_t_p_ctypes,  # sd_ctx
    ],
    ctypes.c_bool,
)
def sd_ctx_supports_image_generation(
    sd_ctx: sd_ctx_t_p,
    /,
) -> bool:
    """Check if the given Stable Diffusion context supports image generation."""
    ...


# -------------------------------------------
# sd_ctx_supports_video_generation
# -------------------------------------------


# SD_API bool sd_ctx_supports_video_generation(const sd_ctx_t* sd_ctx);
@ctypes_function(
    "sd_ctx_supports_video_generation",
    [
        sd_ctx_t_p_ctypes,  # sd_ctx
    ],
    ctypes.c_bool,
)
def sd_ctx_supports_video_generation(
    sd_ctx: sd_ctx_t_p,
    /,
) -> bool:
    """Check if the given Stable Diffusion context supports video generation."""
    ...


# ===========================================
# System Information
# ===========================================

# -------------------------------------------
# sd_get_num_physical_cores
# -------------------------------------------


# SD_API int32_t sd_get_num_physical_cores();
@ctypes_function(
    "sd_get_num_physical_cores",
    [],
    ctypes.c_int32,
)
def sd_get_num_physical_cores() -> int:
    """Get the number of physical cores"""
    ...


# -------------------------------------------
# sd_get_system_info
# -------------------------------------------


# SD_API const char* sd_get_system_info();
@ctypes_function(
    "sd_get_system_info",
    [],
    ctypes.c_char_p,
)
def sd_get_system_info() -> bytes:
    """Get the Stable diffusion system information"""
    ...


# -------------------------------------------
# sd_commit
# -------------------------------------------


# SD_API const char* sd_commit(void);
@ctypes_function(
    "sd_commit",
    [],
    ctypes.c_char_p,
)
def sd_commit() -> bytes:
    """Get the Stable diffusion commit hash"""
    ...


# -------------------------------------------
# sd_version
# -------------------------------------------


# SD_API const char* sd_version(void);
@ctypes_function(
    "sd_version",
    [],
    ctypes.c_char_p,
)
def sd_version() -> bytes:
    """Get the Stable diffusion version string"""
    ...


# ===========================================
# Progression
# ===========================================

# typedef void (*sd_progress_cb_t)(int step, int steps, float time, void* data);
sd_progress_callback = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_void_p)


# SD_API void sd_set_progress_callback(sd_progress_cb_t cb, void* data);
@ctypes_function(
    "sd_set_progress_callback",
    [
        ctypes.c_void_p,  # cb
        ctypes.c_void_p,  # data
    ],
    None,
)
def sd_set_progress_callback(
    callback: Optional[CtypesFuncPointer],
    data: ctypes.c_void_p,
    /,
):
    """Set callback for diffusion progression events."""
    ...


# ===========================================
# Preview
# ===========================================

# typedef void (*sd_preview_cb_t)(int step, int frame_count, sd_image_t* frames, bool is_noisy, void* data);
sd_preview_callback = ctypes.CFUNCTYPE(
    None, ctypes.c_int, ctypes.c_int, ctypes.POINTER(sd_image_t), ctypes.c_bool, ctypes.c_void_p
)


# SD_API void sd_set_preview_callback(sd_preview_cb_t cb, enum preview_t mode, int interval, bool denoised, bool noisy, void* data);
@ctypes_function(
    "sd_set_preview_callback",
    [
        ctypes.c_void_p,  # cb
        ctypes.c_int,  # mode
        ctypes.c_int,  # interval
        ctypes.c_bool,  # denoised
        ctypes.c_bool,  # noisy
        ctypes.c_void_p,  # data
    ],
    None,
)
def sd_set_preview_callback(
    callback: Optional[CtypesFuncPointer],
    mode: int,
    interval: int,
    denoised: bool,
    noisy: bool,
    data: ctypes.c_void_p,
    /,
):
    """Set callback for preview images during generation."""
    ...


# ===========================================
# Logging
# ===========================================

sd_log_callback = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_char_p, ctypes.c_void_p)


# SD_API void sd_set_log_callback(sd_log_cb_t sd_log_cb, void* data);
@ctypes_function(
    "sd_set_log_callback",
    [
        ctypes.c_void_p,  # sd_log_cb
        ctypes.c_void_p,  # data
    ],
    None,
)
def sd_set_log_callback(
    callback: Optional[CtypesFuncPointer],
    data: ctypes.c_void_p,
    /,
):
    """Set callback for all future logging events.
    If this is not called, or NULL is supplied, everything is output on stderr."""
    ...
