# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import inspect
from typing import Callable, List, Optional, Union
import random

import torch
import numpy as np
from tqdm.auto import tqdm
from PIL import Image
import PIL

from diffusers.utils import is_accelerate_available
from packaging import version
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

from ...configuration_utils import FrozenDict
from ...models import AutoencoderKL, UNet2DConditionModel
from ...pipeline_utils import DiffusionPipeline
from ...schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from ...utils import deprecate, logging
from . import StableDiffusionPipelineOutput
from .safety_checker import StableDiffusionSafetyChecker


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def preprocess_mask(mask: PIL.Image.Image, scale_factor: int = 8) -> torch.Tensor:
    """
    Preprocess a mask for the model.
    """
    mask = mask.convert("L")
    w, h = mask.size
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    mask = mask.resize(
        (w // scale_factor, h // scale_factor), resample=PIL.Image.NEAREST
    )
    mask = np.array(mask).astype(np.float32) / 255.0
    mask = np.tile(mask, (4, 1, 1))
    mask = mask[None].transpose(0, 1, 2, 3)  # what does this step do?
    mask = 1 - mask  # repaint white, keep black
    mask = torch.from_numpy(mask)

    return mask


def dynamic_thresholding_(img, quantile):
    # Dynamic thresholding from Imagen paper (May 2022)
    s = np.quantile(np.abs(img.cpu()), quantile, axis=tuple(range(1,img.ndim)))
    s = np.max(np.append(s, 1.0))
    torch.clamp_(img, -1*s, s)
    torch.FloatTensor.div_(img, s)


def dynamic_thresholding_torch(imgs, quantile):
    # Dynamic thresholding from Imagen paper (May 2022)
    #s = torch.quantile(torch.abs(imgs), quantile, dim=tuple(range(1, imgs.ndim)))
    #s = torch.max(torch.cat((s, torch.ones(1, device=imgs.device)), dim=0), dim=0)[0]
    #return torch.clamp(imgs, -s, s) / s
    quant = torch.quantile(imgs.float().abs(), quantile)
    clipped_latent = torch.clip(imgs, -quant, quant) / quant
    return clipped_latent


def minmax(a):
    max_val = a.max()
    min_val = a.min()
    a = (a - min_val) / (max_val - min_val)
    return a


class StableDiffusionPipeline(DiffusionPipeline):
    r"""
    Pipeline for text-to-image generation using Stable Diffusion.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder. Stable Diffusion uses the text portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please, refer to the [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) for details.
        feature_extractor ([`CLIPFeatureExtractor`]):
            Model that extracts features from generated images to be used as inputs for the `safety_checker`.
    """
    _optional_components = ["safety_checker", "feature_extractor"]

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
        ],
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPFeatureExtractor,
        requires_safety_checker: bool = True,
    ):
        super().__init__()

        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is True:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
                " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
            )
            deprecate("clip_sample not set", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)

        if safety_checker is None and requires_safety_checker:
            logger.warning(
                f"You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`. Ensure"
                " that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered"
                " results in services or applications open to the public. Both the diffusers team and Hugging Face"
                " strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling"
                " it only for use-cases that involve analyzing network behavior or auditing its results. For more"
                " information, please have a look at https://github.com/huggingface/diffusers/pull/254 ."
            )

        if safety_checker is not None and feature_extractor is None:
            raise ValueError(
                "Make sure to define a feature extractor when loading {self.__class__} if you want to use the safety"
                " checker. If you do not want to use the safety checker, you can pass `'safety_checker=None'` instead."
            )

        is_unet_version_less_0_9_0 = hasattr(unet.config, "_diffusers_version") and version.parse(
            version.parse(unet.config._diffusers_version).base_version
        ) < version.parse("0.9.0.dev0")
        is_unet_sample_size_less_64 = hasattr(unet.config, "sample_size") and unet.config.sample_size < 64
        if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
            deprecation_message = (
                "The configuration file of the unet has set the default `sample_size` to smaller than"
                " 64 which seems highly unlikely. If your checkpoint is a fine-tuned version of any of the"
                " following: \n- CompVis/stable-diffusion-v1-4 \n- CompVis/stable-diffusion-v1-3 \n-"
                " CompVis/stable-diffusion-v1-2 \n- CompVis/stable-diffusion-v1-1 \n- runwayml/stable-diffusion-v1-5"
                " \n- runwayml/stable-diffusion-inpainting \n you should change 'sample_size' to 64 in the"
                " configuration file. Please make sure to update the config accordingly as leaving `sample_size=32`"
                " in the config might lead to incorrect results in future versions. If you have downloaded this"
                " checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for"
                " the `unet/config.json` file"
            )
            deprecate("sample_size<64", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(unet.config)
            new_config["sample_size"] = 64
            unet._internal_dict = FrozenDict(new_config)

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.register_to_config(requires_safety_checker=requires_safety_checker)
        
        self.compile_dir = None
        self.in_channels = self.unet.in_channels
        
        
    def compile_models(self, compile_dir, width=512, height=512):
        self.compile_dir = compile_dir
        if self.compile_dir is not None:
            if not os.path.exists(self.compile_dir) or not os.path.exists(os.path.join(self.compile_dir, "UNet2DConditionModel")):
                from .compile import compile_diffusers
                compile_diffusers("", width, height, 77, 1, save_path=compile_dir, pipe=self)
                                                                          
            try:
                self.clip_ait_exe = self.init_ait_module(
                    model_name="CLIPTextModel", workdir=self.compile_dir
                )
                self.unet_ait_exe = self.init_ait_module(
                    model_name="UNet2DConditionModel", workdir=self.compile_dir
                )
                self.vae_ait_exe = self.init_ait_module(
                    model_name="AutoencoderKL", workdir=self.compile_dir
                )
            except OSError as e:
                print("Compiling models as they could not be loaded correctly...")
                from .compile import compile_diffusers
                compile_diffusers("", width, height, 77, 1, save_path=compile_dir)
                
                self.clip_ait_exe = self.init_ait_module(
                    model_name="CLIPTextModel", workdir=self.compile_dir
                )
                self.unet_ait_exe = self.init_ait_module(
                    model_name="UNet2DConditionModel", workdir=self.compile_dir
                )
                self.vae_ait_exe = self.init_ait_module(
                    model_name="AutoencoderKL", workdir=self.compile_dir
                )
            
    def del_pt_models(self):
        # delete models. keep vae for encoding
        self.unet.to("cpu")
        self.text_encoder.to("cpu")
        self.vae.decoder.to("cpu")
        del self.unet
        del self.text_encoder
        del self.vae.decoder
            
    def init_ait_module(
        self,
        model_name,
        workdir,
    ):
        from aitemplate.compiler import Model
        
        mod = Model(os.path.join(workdir, model_name, "test.so"))
        return mod

    def unet_inference(self, latent_model_input, timesteps, encoder_hidden_states):
        exe_module = self.unet_ait_exe
        timesteps_pt = timesteps.expand(latent_model_input.shape[0])
        inputs = {
            "input0": latent_model_input.permute((0, 2, 3, 1))
            .contiguous()
            .cuda()
            .half(),
            "input1": timesteps_pt.cuda().half(),
            "input2": encoder_hidden_states.cuda().half(),
        }
        ys = []
        num_ouputs = len(exe_module.get_output_name_to_index_map())
        for i in range(num_ouputs):
            shape = exe_module.get_output_maximum_shape(i)
            ys.append(torch.empty(shape).cuda().half())
        exe_module.run_with_tensors(inputs, ys, graph_mode=False)
        noise_pred = ys[0].permute((0, 3, 1, 2)).float()
        return noise_pred

    def clip_inference(self, input_ids, seqlen=77):
        exe_module = self.clip_ait_exe
        bs = input_ids.shape[0]
        position_ids = torch.arange(seqlen).expand((bs, -1)).cuda()
        inputs = {
            "input0": input_ids,
            "input1": position_ids,
        }
        ys = []
        num_ouputs = len(exe_module.get_output_name_to_index_map())
        for i in range(num_ouputs):
            shape = exe_module.get_output_maximum_shape(i)
            ys.append(torch.empty(shape).cuda().half())
        exe_module.run_with_tensors(inputs, ys, graph_mode=False)
        return ys[0].float()

    def vae_inference(self, vae_input):
        exe_module = self.vae_ait_exe
        inputs = [torch.permute(vae_input, (0, 2, 3, 1)).contiguous().cuda().half()]
        ys = []
        num_ouputs = len(exe_module.get_output_name_to_index_map())
        for i in range(num_ouputs):
            shape = exe_module.get_output_maximum_shape(i)
            ys.append(torch.empty(shape).cuda().half())
        exe_module.run_with_tensors(inputs, ys, graph_mode=False)
        vae_out = ys[0].permute((0, 3, 1, 2)).float()
        return vae_out        

    def enable_vae_slicing(self):
        r"""
        Enable sliced VAE decoding.

        When this option is enabled, the VAE will split the input tensor in slices to compute decoding in several
        steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        r"""
        Disable sliced VAE decoding. If `enable_vae_slicing` was previously invoked, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_slicing()

    def enable_sequential_cpu_offload(self, gpu_id=0):
        r"""
        Offloads all models to CPU using accelerate, significantly reducing memory usage. When called, unet,
        text_encoder, vae and safety checker have their state dicts saved to CPU and then are moved to a
        `torch.device('meta') and loaded to GPU only when their specific submodule has its `forward` method called.
        """
        if is_accelerate_available():
            from accelerate import cpu_offload
        else:
            raise ImportError("Please install accelerate via `pip install accelerate`")


        device = torch.device(f"cuda:{gpu_id}")

        for cpu_offloaded_model in [self.unet, self.text_encoder, self.vae]:
            if cpu_offloaded_model is not None:
                cpu_offload(cpu_offloaded_model, device)

        if self.safety_checker is not None:
            # TODO(Patrick) - there is currently a bug with cpu offload of nn.Parameter in accelerate
            # fix by only offloading self.safety_checker for now
            cpu_offload(self.safety_checker.vision_model, device)

    @property
    def _execution_device(self):
        r"""
        Returns the device on which the pipeline's models will be executed. After calling
        `pipeline.enable_sequential_cpu_offload()` the execution device can only be inferred from Accelerate's module
        hooks.
        """
        if self.device != torch.device("meta") or not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def _encode_prompt(self, prompt, text_embeddings=None, device="cuda", num_images_per_prompt=1, do_classifier_free_guidance=False, negative_prompt=""):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `list(int)`):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
        """
        batch_size = len(prompt) if isinstance(prompt, list) else 1

        if text_embeddings is None:
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(prompt, padding="max_length", return_tensors="pt").input_ids

            if not torch.equal(text_input_ids, untruncated_ids):
                removed_text = self.tokenizer.batch_decode(untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1])
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = text_inputs.attention_mask.to(device)
            else:
                attention_mask = None

            text_embeddings = self.text_encoder(
                text_input_ids.to(device),
                attention_mask=attention_mask,
            )
            text_embeddings = text_embeddings[0]

            # duplicate text embeddings for each generation per prompt, using mps friendly method
            bs_embed, seq_len, _ = text_embeddings.shape
            text_embeddings = text_embeddings.repeat(1, num_images_per_prompt, 1)
            text_embeddings = text_embeddings.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            uncond_embeddings = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            uncond_embeddings = uncond_embeddings[0]

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = uncond_embeddings.shape[1]
            uncond_embeddings = uncond_embeddings.repeat(1, num_images_per_prompt, 1)
            uncond_embeddings = uncond_embeddings.view(batch_size * num_images_per_prompt, seq_len, -1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        return text_embeddings

    def run_safety_checker(self, image, device, dtype):
        if self.safety_checker is not None:
            safety_checker_input = self.feature_extractor(self.numpy_to_pil(image), return_tensors="pt").to(device)
            image, has_nsfw_concept = self.safety_checker(
                images=image, clip_input=safety_checker_input.pixel_values.to(dtype)
            )
        else:
            has_nsfw_concept = None
        return image, has_nsfw_concept

    def decode_latents(self, latents):
        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        image = image.cpu().float()
        return image

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(self, prompt, height, width, callback_steps):
        #if not isinstance(prompt, str) and not isinstance(prompt, list):
        #    raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if latents is None:
            if device.type == "mps":
                # randn does not work reproducibly on mps
                latents = torch.randn(shape, generator=generator, device="cpu", dtype=dtype).to(device)
            else:
                latents = torch.randn(shape, generator=generator, device=device, dtype=dtype)
        else:
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        text_embeddings: Optional[torch.Tensor] = None,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        
        start_img: Optional[torch.Tensor] = None,
        img2img_strength: Optional[float] = None,
        seed=None,
        verbose=True,
        noise: Optional[torch.Tensor] = None,
        loss_callbacks: Optional[List] = None,
        mask_img: Optional = None,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide the image generation.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator`, *optional*):
                A [torch generator](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation
                deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """
        # 0.0 set seed
        if seed is not None:
            self.set_seed(seed)
        
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(prompt, height, width, callback_steps)

        # 2. Define call parameters
        if prompt is None:
            batch_size = len(text_embeddings)
        else:
            batch_size = 1 if isinstance(prompt, str) else len(prompt)
        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        text_embeddings = self._encode_prompt(
            prompt, text_embeddings, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
        )

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        #num_channels_latents = self.unet.in_channels
        #latents = self.prepare_latents(
        #    batch_size * num_images_per_prompt,
        #    num_channels_latents,
        #    height,
        #    width,
        #    text_embeddings.dtype,
        #    device,
        #    generator,
        #    latents,
        #)
        latents, init_latents, timesteps, noise = self.get_start_latents(width, height, 
                                                    batch_size * num_images_per_prompt, 
                                                    generator, text_embeddings, device, 
                                                    start_img, noise, img2img_strength, 
                                                    latents, num_inference_steps
                                                   )
        
        # 5.5 prepare mask
        # Prepare mask latent
        if mask_img:
            vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
            mask_img = preprocess_mask(mask_img, scale_factor=vae_scale_factor)
            mask = mask_img.to(device=self.device, dtype=latents.dtype)
        else:
            mask = None
        

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self.set_progress_bar_config(disable=not verbose)
        
        
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                if self.compile_dir is None:
                    noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
                else:
                    # predict the noise residual
                    noise_pred = self.unet_inference(latent_model_input, t, encoder_hidden_states=text_embeddings)
                
                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)
                   
                # mask latents
                if mask is not None:
                    init_latents_proper = self.scheduler.add_noise(init_latents, noise, torch.tensor([t]))
                    # import ipdb; ipdb.set_trace()
                    latents = (init_latents_proper * mask) + (latents * (1 - mask)) 

        # 8. Post-processing
        image = self.decode_latents(latents.to(text_embeddings.dtype))

        # 9. Run safety checker
        image, has_nsfw_concept = self.run_safety_checker(image, device, text_embeddings.dtype)

        # 10. Convert to PIL
        if output_type == "pil":
            image = image.permute(0, 2, 3, 1).numpy()
            image = self.numpy_to_pil(image)
        elif output_type == "numpy":
            image = image.permute(0, 2, 3, 1).numpy()

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)

    def get_start_latents(self, width, height, 
                          batch_size, generator, 
                          text_embeddings, device, 
                          start_img, noise, 
                          img2img_strength, latents, 
                          num_inference_steps):    
        # get the initial random noise unless the user supplied it

        # Unlike in other pipelines, latents need to be generated in the target device
        # for 1-to-1 results reproducibility with the CompVis implementation.
        # However this currently doesn't work in `mps`.
        offset = self.scheduler.config.get("steps_offset", 0)
        # create starting image/noise
        if noise is None:
            noise = self.sample_noise(width, height, batch_size=batch_size, generator=generator, dtype=text_embeddings.dtype, device=device)
            
        t_start = 0
        init_latents = latents
        if latents is None:
            if start_img is None:
                latents = noise
                # if we use LMSDiscreteScheduler, let's make sure latents are mulitplied by sigmas
                if isinstance(self.scheduler, LMSDiscreteScheduler):
                    latents = latents * self.scheduler.init_noise_sigma#((self.scheduler.sigmas[t_start]**2 + 1) ** 0.5)  
                init_latents = latents
                
            else:
                # encode start img with vae
                # make it torch tensor first
                if torch.is_tensor(start_img) and start_img.shape[-2] == noise.shape[-2]:
                    latents = start_img
                else:
                    latents = self.encode_image(start_img)
                    
                init_latents = latents
                
                # add noise
                if img2img_strength is not None and img2img_strength != 0:
                    # with img2img we skip the first (1-strength) * num_inference_steps steps, so we increase the total step count
                    num_inference_steps = int(np.ceil(num_inference_steps / img2img_strength))
                    # set schedule again according to updated steps
                    self.scheduler.set_timesteps(num_inference_steps)
                    
                    # now we calculate how many skeps are skipped (t_start) and at what point the noise is initialized (init_timestep)
                    init_timestep = int(num_inference_steps * img2img_strength) + offset
                    t_start = num_inference_steps - init_timestep
                    
                    timesteps = self.scheduler.timesteps[t_start]
                    timesteps = torch.tensor([timesteps] * batch_size, device=device)
                    # add noise to latents using the timesteps      
                    latents = self.scheduler.add_noise(latents, noise, timesteps)
        timesteps_tensor = self.scheduler.timesteps[t_start:].to(device)
        return latents, init_latents, timesteps_tensor, noise
    
    def sample_noise(self, width=512, height=512, batch_size=1, generator=None, dtype=None, device="cpu"):
        latents = torch.randn(batch_size, self.in_channels,
                                  height // 8, width // 8, generator=generator, device=device, dtype=dtype)
        return latents
    
    @torch.no_grad()
    def embed_prompts(self, prompts, weights=None, device="cpu"):
        if not isinstance(prompts, list):
            prompts = [prompts]
        text_embeddings = []
        for prompt in prompts:
            text_input = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            )
            text_input_ids = text_input.input_ids
            # check for truncation
            if text_input_ids.shape[-1] > self.tokenizer.model_max_length:
                removed_text = self.tokenizer.batch_decode(text_input_ids[:, self.tokenizer.model_max_length :])
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                )
                text_input_ids = text_input_ids[:, : self.tokenizer.model_max_length]
            
            if self.compile_dir is None:
                text_embedding = self.text_encoder(text_input_ids.to(device))[0]
            else:
                #text_embeddings = self.clip_inference(text_input.input_ids.to(self.device))
                text_embedding = self.clip_inference(text_input_ids.to(device))
            
            text_embeddings.append(text_embedding)
        if weights is None:
            text_embeddings = torch.mean(torch.stack(text_embeddings), 0)
        else:
            weights = torch.tensor(weights, device=text_embedding.device)
            normed_weights = weights / torch.sum(weights)
            text_embeddings = torch.sum(torch.stack(text_embeddings) * normed_weights, 0)
        return text_embeddings
    
    @torch.no_grad()
    def encode_image(self, image, torch_device=None, noise_strength_before_encode=None):
        if torch_device is None:
            torch_device = "cuda" if torch.cuda.is_available() else "cpu"
        if not torch.is_tensor(image):
            if not isinstance(image, np.ndarray):
                image = np.array(image)
            image = torch.from_numpy(image)
        if image.ndim == 3:
            image = image.unsqueeze(0)
        # reverse channel dimension
        if image.shape[-1] == 3:
            image = image.permute(0, 3, 1, 2)
        # make it float tensor between 0 and 1
        image = minmax(image)
        # add noise
        if noise_strength_before_encode is not None and noise_strength_before_encode > 0:
            image = (1 - noise_strength_before_encode) * image + noise_strength_before_encode * torch.randn_like(image)
        # clip to [0, 1]
        image = torch.clamp(image, 0, 1)
        # move to -1 to 1 for vae
        image = (image - 0.5) * 2
        # encode image
        latents = self.vae.encode(image.to(torch_device).to(self.vae.dtype)).latent_dist
        # encoded img is DiagonalGaussianDistribution, need to sample from it or we take the mean instead
        #latents = latents.sample()
        latents = latents.mean
        # norm latents
        latents = latents * 0.18215
        return latents
    
    def decode_image(self, latents, output_type="pil", device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if output_type == "latent":
            return latents.detach().cpu()
        latents = latents / 0.18215
        if self.compile_dir is None:
            image = self.vae.decode(latents.to(device))["sample"]
        else:
            image = self.vae_inference(latents.to(device))
        image = (image / 2 + 0.5).float().clamp(0, 1)
        image = image.cpu()
        
        if output_type == "pil":
            image = self.numpy_to_pil(image.permute(0, 2, 3, 1).numpy())
        elif output_type == "numpy":
            image = image.permute(0, 2, 3, 1).numpy()
        return image
    
    def set_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        
        
def extra_loss_guidance():
    if loss_callbacks is not None and len(loss_callbacks) > 0:
                    with torch.enable_grad():
                        grads = torch.zeros_like(latents)
                        step_index = self.scheduler.get_current_step(t)
                        sigma = self.scheduler.sigmas[step_index]
                        #denoised_images = None
                        for callback_dict in loss_callbacks:
                            if callback_dict["frequency"] is not None and i % callback_dict["frequency"] == 0:
                                # Requires grad on the latents
                                latents = latents.detach().requires_grad_()
                                if callback_dict["apply_to_image"]:
                                    # Get the predicted x0:
                                    if scheduler_step_before_callbacks:
                                        latents_x0 = latents
                                    else:
                                        if use_callbacks_simple_step:
                                            # do simple step
                                            latents_x0 = latents - sigma * noise_pred
                                        else:
                                            # actually use the scheduler step
                                            latents_x0 = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                                    # Decode to image space
                                    #denoised_images = self.vae.decode((1 / 0.18215) * latents_x0) / 2 + 0.5  # (0, 1)
                                    #if denoised_images is None:  
                                    denoised_images = self.vae.decode(latents_x0 / 0.18215)["sample"] / 2 + 0.5  # (0, 1)

                                    # Calculate loss
                                    loss = callback_dict["loss_function"](denoised_images)
                                else:
                                    loss = callback_dict["loss_function"](latents)
                                # Get gradient
                                cond_grad = -torch.autograd.grad(loss * callback_dict["weight"], latents)[0] 
                                # Modify the latents based on this gradient
                                grads += cond_grad * callback_dict["lr"]

                        latents = latents.detach() + grads * sigma**2