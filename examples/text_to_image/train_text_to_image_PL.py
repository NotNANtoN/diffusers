import argparse
import logging
import math
import os
import random
from pathlib import Path
from typing import Iterable, Optional
import sys
sys.path.append("../../src")

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint

from datasets import load_dataset
from diffusers import AutoencoderKL, DDPMScheduler, PNDMScheduler, StableDiffusionPipeline, UNet2DConditionModel, DPMSolverMultistepScheduler
from diffusers.optimization import get_scheduler
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from huggingface_hub import HfFolder, Repository, whoami
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=1,
        required=False,
        help="Num GPUs to use for PL",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=10,
        help="Number of parquet files that are opened and used to train for train_data_dir_var_aspect (each contains 10k images).",
    )
    parser.add_argument(
        "--max_width",
        type=int,
        default=768,
        help="Max width for train images",
    )
    parser.add_argument(
        "--max_height",
        type=int,
        default=768,
        help="Max height for train imges",
    )
    parser.add_argument(
        "--min_width",
        type=int,
        default=128,
        help="Max width for train images",
    )
    parser.add_argument(
        "--min_height",
        type=int,
        default=128,
        help="Max height for train imges",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--train_data_dir_var_aspect",
        type=str,
        default=None,
        help=(
            "A folder containing the training data, with variable aspect ratios"
        ),
    )
    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing an image."
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-model-finetuned",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        action="store_true",
        help="Whether to center crop images before resizing to resolution (if not set, random crop will be used)",
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"` and `"comet_ml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # Sanity checks
    if args.dataset_name is None and args.train_data_dir is None and args.train_data_dir_var_aspect is None:
        raise ValueError("Need either a dataset name or a training folder.")

    return args


def get_full_repo_name(model_id: str, organization: Optional[str] = None, token: Optional[str] = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"


dataset_name_mapping = {
    "lambdalabs/pokemon-blip-captions": ("image", "text"),
}


# Adapted from torch-ema https://github.com/fadel/pytorch_ema/blob/master/torch_ema/ema.py#L14
class EMAModel:
    """
    Exponential Moving Average of models weights
    """

    def __init__(self, parameters: Iterable[torch.nn.Parameter], decay=0.9999):
        parameters = list(parameters)
        self.shadow_params = [p.clone().detach() for p in parameters]

        self.decay = decay
        self.optimization_step = 0

    def get_decay(self, optimization_step):
        """
        Compute the decay factor for the exponential moving average.
        """
        value = (1 + optimization_step) / (10 + optimization_step)
        return 1 - min(self.decay, value)

    @torch.no_grad()
    def step(self, parameters):
        parameters = list(parameters)

        self.optimization_step += 1
        self.decay = self.get_decay(self.optimization_step)

        for s_param, param in zip(self.shadow_params, parameters):
            if param.requires_grad:
                tmp = self.decay * (s_param - param)
                s_param.sub_(tmp)
            else:
                s_param.copy_(param)

        torch.cuda.empty_cache()

    def copy_to(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        """
        Copy current averaged parameters into given collection of parameters.

        Args:
            parameters: Iterable of `torch.nn.Parameter`; the parameters to be
                updated with the stored moving averages. If `None`, the
                parameters with which this `ExponentialMovingAverage` was
                initialized will be used.
        """
        parameters = list(parameters)
        for s_param, param in zip(self.shadow_params, parameters):
            param.data.copy_(s_param.data)

    def to(self, device=None, dtype=None) -> None:
        r"""Move internal buffers of the ExponentialMovingAverage to `device`.

        Args:
            device: like `device` argument to `torch.Tensor.to`
        """
        # .to() on the tensors handles None correctly
        self.shadow_params = [
            p.to(device=device, dtype=dtype) if p.is_floating_point() else p.to(device=device)
            for p in self.shadow_params
        ]

        
@torch.inference_mode()  
def eval_model(args, torch_precision, unet, ema_unet, text_encoder, vae, tokenizer, global_step):
    gen_dir = os.path.join(args.output_dir, "generations", f"{global_step}")
    os.makedirs(gen_dir, exist_ok=True)

    pipe = models_to_pipe(args.use_ema,
                          unet, ema_unet, 
                          text_encoder, vae, 
                          tokenizer, 
                          args.pretrained_model_name_or_path,
                          torch_precision)


    prompts = ["full body portrait of Dilraba, slight smile, diffuse natural sun lights, autumn lights, highly detailed, digital painting, artstation, concept art, sharp focus, illustration",
               "cyborg woman| with a visible detailed brain| muscles cable wires| biopunk| cybernetic| cyberpunk| white marble bust| canon m50| 100mm| sharp focus| smooth| hyperrealism| highly detailed| intricate details| carved by michelangelo",
               "Goldorg, demonic orc from Moria, new leader of the Gundabad, strong muscular body, ugly figure, dirty grey skin, burned wrinkled face, body interlaced with frightening armor, metal coatings crossing head, heavy muscular figure, cinematic shot, detailed, trending on Artstation, dark blueish environment, demonic backlight, unreal engine, 8k",
               "Boris Johnson as smiling Rick Sanchez from Rick and Morty, unibrow, white robe, big eyes, realistic portrait, symmetrical, highly detailed, digital painting, artstation, concept art, smooth, sharp focus, illustration, cinematic lighting, art by artgerm and greg rutkowski and alphonse mucha",
               "Slavic dog head man, woolen torso in medieval clothes, characteristic of cynocephaly, oil painting, hyperrealism, beautiful, high resolution, trending on artstation",
               "Jesus taking a selfie, instagram, high detail, glamorous photograph, natural lighting",
              ]

    import wandb
    out_dict = {}

    if args.max_width > 256:
        resolutions = [(512, 512), (256, 256), (1024, 512), (512, 1024)]
    else:
        resolutions = [(128, 128), (64, 64), (128, 64), (64, 128)]
    for width, height in resolutions:
        for i, p in enumerate(prompts):
            pil_img = pipe(p, output_type="pil", num_inference_steps=30,
                    guidance_scale=10, seed=11,
                    width=width, height=height,
              )[0][0]
            img_name = f"{i}_{width}x{height}.jpg"
            img_path = os.path.join(gen_dir, img_name)

            out_dict[f"{width}x{height}/{i}"] = wandb.Image(pil_img)
            pil_img.save(img_path)

    wandb.log(out_dict)
    del pipe
                        

@torch.inference_mode()
def models_to_pipe(use_ema, unet, ema_unet, text_encoder, vae, tokenizer, pretrained_path, weight_dtype):
    state_dict = unet.state_dict()
    
    unet = UNet2DConditionModel.from_pretrained(pretrained_path, subfolder="unet")
    unet.load_state_dict(state_dict)
    unet.to(weight_dtype)
    unet.eval()
    unet.to("cuda")
    
    if use_ema:
        ema_unet.copy_to(unet.parameters())

    pipe = StableDiffusionPipeline(
        text_encoder=text_encoder,
        vae=vae,
        unet=unet,
        tokenizer=tokenizer,
        scheduler=DPMSolverMultistepScheduler.from_config(pretrained_path, subfolder="scheduler"),
        safety_checker=None,
        feature_extractor=None,
    )
    return pipe
    
    
from pytorch_lightning import Trainer, LightningModule


class LitModel(LightningModule):
    def __init__(self, args, unet, vae, text_encoder, ema_unet, tokenizer):
        super().__init__()
        self.args = args
        self.unet = unet
        self.vae = vae
        self.text_encoder = text_encoder
        self.ema_unet = ema_unet
        self.tokenizer = tokenizer
        self.noise_scheduler = DDPMScheduler.from_config(args.pretrained_model_name_or_path, subfolder="scheduler")


        self.torch_precision = args.mixed_precision
        if self.torch_precision == "fp16":
            self.torch_precision = torch.float16
            self.text_encoder = self.text_encoder.half()
            self.vae = self.vae.half()
        elif self.torch_precision == "bf16":
            self.torch_precision = torch.bfloat16
            self.text_encoder = self.text_encoder.to(torch.bfloat16)
            self.vae = self.vae.to(torch.bfloat16)
        self.step = 0

    def training_step(self, batch, batch_idx):
        latents = self.vae.encode(batch["pixel_values"].to(self.torch_precision)).latent_dist.sample()
        latents = latents * 0.18215

        #print(batch["pixel_values"].shape, batch["pixel_values"].dtype, batch["pixel_values"].requires_grad)
        #print(latents.shape, latents.dtype, latents.requires_grad)

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(0, self.noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
        timesteps = timesteps.long()

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        # Get the text embedding for conditioning
        encoder_hidden_states = self.text_encoder(batch["input_ids"])[0]

        # Predict the noise residual and compute loss
        unet_dtype = self.unet.conv_in.bias.dtype
        noisy_latents = noisy_latents.to(unet_dtype)
        encoder_hidden_states = encoder_hidden_states.to(unet_dtype)
        noise = noise.to(unet_dtype)
        #print(unet_dtype)
        noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample
        #loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
        loss = F.mse_loss(noise_pred, noise, reduction="mean")

        if self.args.use_ema:
            self.ema_unet.step(self.unet.parameters())

        self.step += 1
        return loss


    def val_step(self):
        eval_model(self.args, self.torch_precision, self.unet, self.ema_unet, self.text_encoder, self.vae, self.tokenizer, self.step)

    def configure_optimizers(self):
        # Initialize the optimizer
        if self.args.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
                )

            optimizer_cls = bnb.optim.AdamW8bit
            print("using 8bit")
        else:
            optimizer_cls = torch.optim.AdamW
        optimizer = optimizer_cls(
            self.unet.parameters(),
            lr=self.args.learning_rate,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            weight_decay=self.args.adam_weight_decay,
            eps=self.args.adam_epsilon,
        )
        return optimizer
        
        
def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    input_ids = torch.stack([example["input_ids"] for example in examples])
    masks = torch.stack([example["attention_mask"] for example in examples])
    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": masks,
    }
            
    
def main():
    import torch.multiprocessing
    torch.multiprocessing.set_sharing_strategy('file_system')
    
    
    args = parse_args()
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # If passed along, set the training seed now.
    #if args.seed is not None:
    #    set_seed(args.seed)

    # Load models and create wrapper for stable diffusion
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")

    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * args.gpus
        )

    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
        )
    else:
        if args.train_data_dir is not None:
            data_files = {}
            data_files["train"] = os.path.join(args.train_data_dir, "**")
            dataset = load_dataset(
                "imagefolder",
                data_files=data_files,
                cache_dir=args.cache_dir,
            )

            # See more about loading custom images at
            # https://huggingface.co/docs/datasets/v2.4.0/en/image_load#imagefolder
        elif args.train_data_dir_var_aspect is not None:
            from varying_aspect_ratio_dataset import BucketSampler, BucketDataset
            from torch.utils.data import DataLoader
            from varying_aspect_ratio_dataset import create_df_from_parquets, assign_to_buckets

            cache = f"laion_aesthetics_{args.max_files}.parquet"

            import pandas as pd
            if os.path.exists(cache):
                df = pd.read_parquet(cache)
            else:

                df = create_df_from_parquets(args.train_data_dir_var_aspect, min_width=args.min_width, min_height=args.min_height, max_files=args.max_files)
                df = assign_to_buckets(df, 
                                       bucket_step_size=64, 
                                       max_width=args.max_width, max_height=args.max_height,
                                       min_bucket_count=64)
                df.to_parquet(cache)
                
            bucket_sampler = BucketSampler(df["bucket"], batch_size=args.train_batch_size) 
            train_dataset = BucketDataset(df, tokenizer)
            train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, 
                                          sampler=bucket_sampler, 
                                          pin_memory=True,
                                          shuffle=False, 
                                          collate_fn=collate_fn,
                                          num_workers=16,
                                          drop_last=False)
                

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    
    #lr_scheduler = get_scheduler(
    #    args.lr_scheduler,
    #    optimizer=optimizer,
    #    num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
    #    num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    #)

    # Create EMA for the unet.
    if args.use_ema:
        ema_unet = EMAModel(unet.parameters())
    else:
        ema_unet = None

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
            
            
    # Train!
    total_batch_size = args.train_batch_size * args.gradient_accumulation_steps * args.gpus

    print("***** Running training *****")
    print(f"  Num examples = {len(train_dataset)}")
    print(f"  Num Epochs = {args.num_train_epochs}")
    print(f"  Instantaneous batch size per device = {args.train_batch_size}")
    print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    print(f"  Total optimization steps = {args.max_train_steps}")
        
    precision = args.mixed_precision
    if precision == "fp16":
        precision = 16
    elif precision == "fp32":
        precision = 32
        
    # from: https://forums.pytorchlightning.ai/t/gradient-checkpointing-ddp-nan/398/6
    import pytorch_lightning
    from pytorch_lightning.overrides import LightningDistributedModule
    class CustomDDPPlugin(pytorch_lightning.plugins.training_type.DDPPlugin):
        def configure_ddp(self):
            self.pre_configure_ddp()
            self._model = self._setup_model(LightningDistributedModule(self.model))
            self._register_ddp_hooks()
            self._model._set_static_graph() # THIS IS THE MAGIC LINE to fix DDP+gradient_checkpointing of unet
            # note: does not really work well. Increases memory load and maybe completely negates gradient checkpointing. Also, if it sets a static_graph that is not compatible with our various aspect ratios
    
    trainer = Trainer(#precision=32, # to make PT lightning work with 8bit adam need fp32
                      precision=precision,
                      devices=args.gpus,
                      accelerator="cuda",
                      #strategy="dp",
                      #strategy=CustomDDPPlugin(),
                      accumulate_grad_batches=args.gradient_accumulation_steps,
                      max_epochs=args.num_train_epochs,
                      max_steps=args.max_train_steps,
                      benchmark=False,
                      gradient_clip_val=args.max_grad_norm,
                      val_check_interval=100,
                     )
    
    lit_model = LitModel(args, unet, vae, text_encoder, ema_unet, tokenizer)
    
    trainer.fit(lit_model, train_dataloader)
      

    # Create the pipeline using the trained modules and save it.
    pipe = models_to_pipe(args.use_ema,
                          unet, ema_unet, 
                          text_encoder, vae, 
                          tokenizer, 
                          args.pretrained_model_name_or_path,
                          lit_model.torch_precision)
    pipe.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
