accelerate launch --mixed_precision="bf16" train_text_to_image.py \
  --pretrained_model_name_or_path="../../../mus2vid/models/stable-diffusion-v1-5" \
  --train_data_dir_var_aspect="/hdd/data/finetune_SD/laion_aesthetics"  \
  --train_batch_size=1 \
  --gradient_accumulation_steps=8 \
  --gradient_checkpointing \
  --max_train_steps=2000 \
  --learning_rate=8e-07 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=100 \
  --output_dir="sdv1-5-var-aspect-3" \
  --use_8bit_adam \
  --mixed_precision="bf16" \
  --max_files=100 \
  --max_width=768 \
  --max_height=768 \

#  --use_ema \
