import os
import ctypes
from typing import Any, Optional
from contextlib import ExitStack

import stable_diffusion_cpp.stable_diffusion_cpp as sd_cpp
from ._utils import suppress_stdout_stderr

# ===========================================
# Stable Diffusion Model
# ===========================================


class _StableDiffusionModel:
    """Intermediate Python wrapper for a stable-diffusion.cpp stable_diffusion_model."""

    _free_sd_ctx = None
    # NOTE: this must be "saved" here to avoid exceptions when calling __del__

    def __init__(
        self,
        model_path: str,
        clip_l_path: str,
        clip_g_path: str,
        clip_vision_path: str,
        t5xxl_path: str,
        llm_path: str,
        llm_vision_path: str,
        diffusion_model_path: str,
        high_noise_diffusion_model_path: str,
        vae_path: str,
        taesd_path: str,
        control_net_path: str,
        embeddings: ctypes.Array[sd_cpp.sd_embedding_t],
        embedding_count: int,
        photo_maker_path: str,
        tensor_type_rules: str,
        vae_decode_only: bool = False,
        n_threads: int = 8,
        wtype: int = 0,
        rng_type: int = 0,
        sampler_rng_type: int = 0,
        prediction: int = 0,
        lora_apply_mode: int = 0,
        offload_params_to_cpu: bool = False,
        enable_mmap: bool = True,
        keep_clip_on_cpu: bool = False,
        keep_control_net_on_cpu: bool = False,
        keep_vae_on_cpu: bool = False,
        flash_attn: bool = False,
        diffusion_flash_attn: bool = False,
        tae_preview_only: bool = False,
        diffusion_conv_direct: bool = False,
        vae_conv_direct: bool = False,
        circular_x: bool = False,
        circular_y: bool = False,
        force_sdxl_vae_conv_scale: bool = False,
        chroma_use_dit_mask: bool = False,
        chroma_use_t5_mask: bool = False,
        chroma_t5_mask_pad: int = 0,
        qwen_image_zero_cond_t: bool = False,
        max_vram: Any = None,
        verbose: bool = False,
        uncond_diffusion_model_path: str = "",
        embeddings_connectors_path: str = "",
        audio_vae_path: str = "",
        audio_encoder_path: str = "",
        ip_adapter_path: str = "",
        motion_module_path: str = "",
        pulid_weights_path: str = "",
        vae_format: int = -1,
        disable_prefetch: bool = False,
        eager_load: bool = False,
        backend: str = "",
        params_backend: str = "",
        split_mode: str = "",
        auto_fit: bool = False,
        rpc_servers: str = "",
        model_args: str = "",
        disable_segmented_compute: bool = False,
        linear_scale: float = 0.0,
        attn_scale: float = 0.0,
        tokenizer: str = "",
        sage_attn: bool = False,
        conditioning_cache_size: int = 4,
    ):
        self._exit_stack = ExitStack()
        self.model = None

        def _encode_str(s: Optional[str]) -> Optional[bytes]:
            return s.encode("utf-8") if s else None

        self.params = sd_cpp.sd_ctx_params_t(
            model_path=_encode_str(model_path),
            clip_l_path=_encode_str(clip_l_path),
            clip_g_path=_encode_str(clip_g_path),
            clip_vision_path=_encode_str(clip_vision_path),
            t5xxl_path=_encode_str(t5xxl_path),
            llm_path=_encode_str(llm_path),
            llm_vision_path=_encode_str(llm_vision_path),
            diffusion_model_path=_encode_str(diffusion_model_path),
            high_noise_diffusion_model_path=_encode_str(high_noise_diffusion_model_path),
            uncond_diffusion_model_path=_encode_str(uncond_diffusion_model_path),
            embeddings_connectors_path=_encode_str(embeddings_connectors_path),
            vae_path=_encode_str(vae_path),
            audio_vae_path=_encode_str(audio_vae_path),
            audio_encoder_path=_encode_str(audio_encoder_path),
            taesd_path=_encode_str(taesd_path),
            control_net_path=_encode_str(control_net_path),
            ip_adapter_path=_encode_str(ip_adapter_path),
            motion_module_path=_encode_str(motion_module_path),
            embeddings=embeddings,
            embedding_count=embedding_count,
            photo_maker_path=_encode_str(photo_maker_path),
            pulid_weights_path=_encode_str(pulid_weights_path),
            tensor_type_rules=_encode_str(tensor_type_rules),
            n_threads=n_threads,
            wtype=wtype,
            rng_type=rng_type,
            sampler_rng_type=sampler_rng_type,
            prediction=prediction,
            lora_apply_mode=lora_apply_mode,
            enable_mmap=enable_mmap,
            flash_attn=flash_attn,
            diffusion_flash_attn=diffusion_flash_attn,
            tae_preview_only=tae_preview_only,
            diffusion_conv_direct=diffusion_conv_direct,
            vae_conv_direct=vae_conv_direct,
            force_sdxl_vae_conv_scale=force_sdxl_vae_conv_scale,
            vae_format=vae_format,
            max_vram=_encode_str(str(max_vram)) if max_vram else None,
            disable_prefetch=disable_prefetch,
            eager_load=eager_load,
            backend=_encode_str(backend),
            params_backend=_encode_str(params_backend),
            split_mode=_encode_str(split_mode),
            auto_fit=auto_fit,
            rpc_servers=_encode_str(rpc_servers),
            model_args=_encode_str(model_args),
            disable_segmented_compute=disable_segmented_compute,
            linear_scale=linear_scale,
            attn_scale=attn_scale,
            tokenizer=_encode_str(tokenizer),
            sage_attn=sage_attn,
            conditioning_cache_size=conditioning_cache_size,
        )

        # Load the free_sd_ctx function
        self._free_sd_ctx = sd_cpp._lib.free_sd_ctx

        # Load the model from the file if the path is provided
        if model_path:
            if not os.path.exists(model_path):
                raise ValueError(f"Model path does not exist: '{model_path}'")

        if diffusion_model_path:
            if not os.path.exists(diffusion_model_path):
                raise ValueError(f"Diffusion model path does not exist: '{diffusion_model_path}'")

        if model_path or diffusion_model_path:
            with suppress_stdout_stderr(disable=verbose):
                # Call function with a pointer to params
                self.model = sd_cpp.new_sd_ctx(ctypes.pointer(self.params))

            # Check if the model was loaded successfully
            if self.model is None:
                raise ValueError(f"Failed to load model from file: '{model_path}'")

        def free_ctx():
            """Free the model from memory."""
            if self.model is not None and self._free_sd_ctx is not None:
                self._free_sd_ctx(self.model)
                self.model = None

        self._exit_stack.callback(free_ctx)

    def close(self):
        """Closes the exit stack, ensuring all context managers are exited."""
        self._exit_stack.close()

    def __del__(self):
        """Free memory when the object is deleted."""
        self.close()


# ===========================================
# Upscaler Model
# ===========================================


class _UpscalerModel:
    """Intermediate Python wrapper for an Esrgan image upscaling model."""

    _free_upscaler_ctx = None
    # NOTE: this must be "saved" here to avoid exceptions when calling __del__

    def __init__(
        self,
        upscaler_path: str,
        offload_params_to_cpu: bool,
        direct: bool,
        n_threads: int,
        tile_size: int,
        verbose: bool,
    ):
        self.upscaler_path = upscaler_path
        self.offload_params_to_cpu = offload_params_to_cpu
        self.direct = direct
        self.n_threads = n_threads
        self.tile_size = tile_size
        self.verbose = verbose
        self._exit_stack = ExitStack()

        self.upscaler = None

        # Load the model from the file if the path is provided
        if upscaler_path:

            # Load the free_upscaler_ctx function
            self._free_upscaler_ctx = sd_cpp._lib.free_upscaler_ctx

            if not os.path.exists(upscaler_path):
                raise ValueError(f"Upscaler model path does not exist: '{upscaler_path}'")

            # Load the image upscaling model ctx
            self.upscaler = sd_cpp.new_upscaler_ctx(
                upscaler_path.encode("utf-8"),
                self.direct,
                self.n_threads,
                self.tile_size,
                None,
                None,
            )

            # Check if the model was loaded successfully
            if self.upscaler is None:
                raise ValueError(f"Failed to load upscaler model from file: '{upscaler_path}'")

        def free_ctx():
            """Free the model from memory."""
            if self.upscaler is not None and self._free_upscaler_ctx is not None:
                self._free_upscaler_ctx(self.upscaler)
                self.upscaler = None

        self._exit_stack.callback(free_ctx)

    def close(self):
        """Closes the exit stack, ensuring all context managers are exited."""
        self._exit_stack.close()

    def __del__(self):
        """Free memory when the object is deleted."""
        self.close()
