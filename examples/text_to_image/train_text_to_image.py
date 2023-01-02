import argparse
import logging
import math
import os
import random
from pathlib import Path
from typing import Iterable, Optional
import itertools
import sys
sys.path.append("../../src")

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import wandb

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel, DPMSolverMultistepScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from huggingface_hub import HfFolder, Repository, whoami
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer


try:
    from lora_diffusion import (
        extract_lora_ups_down,
        inject_trainable_lora,
        save_lora_weight,
        save_safeloras,
    )
except Exception:
    print("Could not load LoRA")

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.10.0.dev0")

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--num_experts",
        type=int,
        default=10,
        help="Number of experts to use with LoRA",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=0,
        help="Rank of LoRA approximation. If 0, no LoRA is used",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--compile",
        default=False,
        action="store_true",
        required=False,
        help="Whether to use new torch.compile from PT 2.0",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
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
    parser.add_argument("--ema_deacy", type=float, default=0.9999, help="EMA decay value")
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
                tmp = self.decay * (s_param - param.cpu())
                s_param.sub_(tmp)
            #else:
            #    s_param.copy_(param)

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
            param.data.copy_(s_param.data.to(param.data.device))

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
def models_to_pipe(accelerator, use_ema, unet, ema_unet, text_encoder, vae, pretrained_path, weight_dtype, revision, lora_rank, num_experts):
    # get state dict
    state_dict = unet.state_dict()
    #print(unet.conv_in.bias[:5])
    # rename for multi-gpu
    new_state_dict = {}
    for key in state_dict:
        if key.startswith("module."):
            cleaned_key = key.replace("module.", "")
        else:
            cleaned_key = key
        new_state_dict[cleaned_key] = state_dict[key].clone()
    del state_dict
    # create new model and load state dict into it
    new_unet = UNet2DConditionModel.from_pretrained(pretrained_path, subfolder="unet")
    if lora_rank > 0:
        from lora_utils import UNetLORAMoDE
        new_unet = UNetLORAMoDE(new_unet, num_experts=num_experts)
        #save_lora_weight(unet, "tmp/load_weights")
        #unet_lora_params, _ = inject_trainable_lora(
        #        new_unet, r=lora_rank#, loras=args.resume_unet
        #    )
    #else:
    
    if use_ema:
        ema_unet.copy_to(new_unet.parameters())
    else:
        new_unet.load_state_dict(new_state_dict, strict=True)
        
    new_unet.to(weight_dtype)
    new_unet.eval()
    new_unet.to(accelerator.device)
    
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_path, subfolder="tokenizer", revision=revision)
    scheduler = DPMSolverMultistepScheduler.from_pretrained(pretrained_path, subfolder="scheduler", revision=revision)

    pipe = StableDiffusionPipeline(
        text_encoder=text_encoder,
        vae=vae,
        unet=new_unet,
        tokenizer=tokenizer,
        scheduler=scheduler,
        safety_checker=None,
        feature_extractor=None,     
        requires_safety_checker=False,
    ).to(weight_dtype).to(accelerator.device)
    
    return pipe
    
    
def main():
    args = parse_args()
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    
    # save args
    import json
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'args.json'), 'w+', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        logging_dir=logging_dir,
    )
    
    if accelerator.is_local_main_process:
        # wandb init
        run = wandb.init(entity="finetuners")

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(Path(args.output_dir).name, 
                                               token=args.hub_token)
            else:
                repo_name = args.hub_model_id
            repo = Repository(args.output_dir, clone_from=repo_name)

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load models and create wrapper for stable diffusion
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        revision=args.revision,
    )

    if is_xformers_available():
        try:
            unet.enable_xformers_memory_efficient_attention()
        except Exception as e:
            logger.warning(
                "Could not enable memory efficient attention. Make sure xformers is installed"
                f" correctly and a GPU is available: {e}"
            )

    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        
           
    # add LoRA
    if args.lora_rank > 0:
        from lora_utils import UNetLORAMoDE
        unet = UNetLORAMoDE(unet, num_experts=args.num_experts)
        opt_params = itertools.chain(*itertools.chain(*unet.expert_weights))
        
        #unet.requires_grad_(False)
        #unet_lora_params, _ = inject_trainable_lora(
        #    unet, r=args.lora_rank#, loras=args.resume_unet
        #)
        #opt_params = itertools.chain(*unet_lora_params)
    else:
        opt_params = unet.parameters()

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        opt_params,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

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
                with accelerator.main_process_first():
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
                                          num_workers=16,
                                          drop_last=False)
        
    if args.train_data_dir_var_aspect is None:
        # Preprocessing the datasets.
        # We need to tokenize inputs and targets.
        column_names = dataset["train"].column_names

        # 6. Get the column names for input/target.
        dataset_columns = dataset_name_mapping.get(args.dataset_name, None)
        if args.image_column is None:
            image_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
        else:
            image_column = args.image_column
            if image_column not in column_names:
                raise ValueError(
                    f"--image_column' value '{args.image_column}' needs to be one of: {', '.join(column_names)}"
                )
        if args.caption_column is None:
            caption_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
        else:
            caption_column = args.caption_column
            if caption_column not in column_names:
                raise ValueError(
                    f"--caption_column' value '{args.caption_column}' needs to be one of: {', '.join(column_names)}"
                )

        # Preprocessing the datasets.
        # We need to tokenize input captions and transform the images.
        def tokenize_captions(examples, is_train=True):
            captions = []
            for caption in examples[caption_column]:
                if isinstance(caption, str):
                    captions.append(caption)
                elif isinstance(caption, (list, np.ndarray)):
                    # take a random caption if there are multiple
                    captions.append(random.choice(caption) if is_train else caption[0])
                else:
                    raise ValueError(
                        f"Caption column `{caption_column}` should contain either strings or lists of strings."
                    )
            inputs = tokenizer(captions, max_length=tokenizer.model_max_length, padding="do_not_pad", truncation=True)
            input_ids = inputs.input_ids
            return input_ids

        train_transforms = transforms.Compose(
            [
                transforms.Resize((args.resolution, args.resolution), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
                transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        def preprocess_train(examples):
            images = [image.convert("RGB") for image in examples[image_column]]
            examples["pixel_values"] = [train_transforms(image) for image in images]
            examples["input_ids"] = tokenize_captions(examples)

            return examples

        with accelerator.main_process_first():
            if args.max_train_samples is not None:
                dataset["train"] = dataset["train"].shuffle(seed=args.seed).select(range(args.max_train_samples))
            # Set the training transforms
            train_dataset = dataset["train"].with_transform(preprocess_train)

        def collate_fn(examples):
            pixel_values = torch.stack([example["pixel_values"] for example in examples])
            pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
            input_ids = [example["input_ids"] for example in examples]
            padded_tokens = tokenizer.pad({"input_ids": input_ids}, padding=True, return_tensors="pt")
            return {
                "pixel_values": pixel_values,
                "input_ids": padded_tokens.input_ids,
                "attention_mask": padded_tokens.attention_mask,
            }

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, shuffle=True, collate_fn=collate_fn, batch_size=args.train_batch_size
        )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move text_encode and vae to gpu.
    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    
    if args.compile:
        unet = torch.compile(unet)
        vae = torch.compile(vae)
        text_encoder = torch.compile(text_encoder)

    # Create EMA for the unet.
    if args.use_ema:
        ema_unet = EMAModel(unet.parameters(), args.ema_decay)
        ema_unet.to("cpu")
    else:
        ema_unet = None

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("text2image-fine-tune", config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  Learning rate at start = {args.learning_rate}")
    
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    global_step = 0
    
    eval_every = 20 * args.gradient_accumulation_steps
    save_every = 1000 * args.gradient_accumulation_steps
    
    
    for epoch in range(args.num_train_epochs):
        unet.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # Convert images to latent space
                latents = vae.encode(batch["pixel_values"].to(weight_dtype)).latent_dist.sample()
                latents = latents * 0.18215

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                if args.lora_rank > 0:
                    # when using mixture of denoising experts then we need to pass the same timestep for all batch elemenets as only one expert operates on them
                    timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (1,), device=latents.device, dtype=torch.int64).repeat(bsz)
                    timesteps = timesteps.long()
                else:
                    timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
                    timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                # Predict the noise residual and compute loss
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
                
                current_lr = lr_scheduler.get_last_lr()[0]
                logs = {"step_loss": loss.detach().item(), "lr": current_lr}
                progress_bar.set_postfix(**logs)
                    
                if accelerator.sync_gradients:
                    if args.use_ema:
                        ema_unet.step(unet.parameters())
                    progress_bar.update(1)
                    global_step += 1
                    # log learning rate
                    accelerator.log({"train_loss": train_loss, "lr": current_lr}, step=global_step)
                    train_loss = 0.0

                

                if global_step >= args.max_train_steps:
                    break
                
            # evaluate model regularly
            if accelerator.is_local_main_process:            
                if step % eval_every == 0:
                    gen_dir = os.path.join(args.output_dir, "generations", f"{global_step}")
                    os.makedirs(gen_dir, exist_ok=True)
        
                    pipe = models_to_pipe(accelerator, args.use_ema,
                                          unet, ema_unet, 
                                          text_encoder, vae,  
                                          args.pretrained_model_name_or_path,
                                          weight_dtype,
                                          args.revision,
                                          args.lora_rank,
                                          args.num_experts
                                         )


                    prompts = ["full body portrait of Dilraba, slight smile, diffuse natural sun lights, autumn lights, highly detailed, digital painting, artstation, concept art, sharp focus, illustration",
                               "cyborg woman| with a visible detailed brain| muscles cable wires| biopunk| cybernetic| cyberpunk| white marble bust| canon m50| 100mm| sharp focus| smooth| hyperrealism| highly detailed| intricate details| carved by michelangelo",
                               "Goldorg, demonic orc from Moria, new leader of the Gundabad, strong muscular body, ugly figure, dirty grey skin, burned wrinkled face, body interlaced with frightening armor, metal coatings crossing head, heavy muscular figure, cinematic shot, detailed, trending on Artstation, dark blueish environment, demonic backlight, unreal engine, 8k",
                               "Boris Johnson as smiling Rick Sanchez from Rick and Morty, unibrow, white robe, big eyes, realistic portrait, symmetrical, highly detailed, digital painting, artstation, concept art, smooth, sharp focus, illustration, cinematic lighting, art by artgerm and greg rutkowski and alphonse mucha",
                               "Slavic dog head man, woolen torso in medieval clothes, characteristic of cynocephaly, oil painting, hyperrealism, beautiful, high resolution, trending on artstation",
                               "Jesus taking a selfie, instagram, high detail, glamorous photograph, natural lighting",
                              ]

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

                    accelerator.log(out_dict, step=global_step)
                    del pipe
        
                if step > 5 and step % save_every == 0:
                    save_folder = os.path.join(args.output_dir, f"model_at_{global_step}")
                    
                    if args.lora_rank > 0:
                        save_lora_weight(unet, save_folder.replace("model_at", "lora_at"))
                    else:
                        pipe = models_to_pipe(accelerator, args.use_ema,
                                              unet, ema_unet, 
                                              text_encoder, vae,  
                                              args.pretrained_model_name_or_path,
                                              weight_dtype,
                                              args.revision,
                                              args.lora_rank,
                                              args.num_experts
                                             )

                        os.makedirs(save_folder, exist_ok=True)
                        pipe.save_pretrained(save_folder)
                        del pipe
                    
        
            
    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        pipe = models_to_pipe(accelerator, args.use_ema,
                              unet, ema_unet, 
                              text_encoder, vae,  
                              args.pretrained_model_name_or_path,
                              weight_dtype,
                              args.revision,
                              args.lora_rank,
                              args.num_experts
                             )
        save_folder = os.path.join(args.output_dir, f"final_model_at_{global_step}")
        os.makedirs(save_folder, exist_ok=True)
        pipe.save_pretrained(save_folder)


        if args.push_to_hub:
            repo.push_to_hub(commit_message="End of training", blocking=False, auto_lfs_prune=True)

    accelerator.end_training()
    # finish logging
    if accelerator.is_local_main_process:
        run.finish()


if __name__ == "__main__":
    main()
