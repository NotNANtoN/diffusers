accelerate launch --mixed_precision="bf16" train_text_to_image.py \
  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
  --train_data_dir_var_aspect="PATH_TO_DATA"  \
  --train_batch_size=8 \
  --gradient_accumulation_steps=32 \
  --gradient_checkpointing \
  --max_train_steps=20001 \
  --learning_rate=1e-06 \
  --max_grad_norm=1 \
  --lr_scheduler="constant_with_warmup" --lr_warmup_steps=500 \
  --output_dir="sdv1-5-var-aspect-1" \
  --use_8bit_adam \
  --mixed_precision="bf16" \
  --max_files=2000 \
  --max_width=1024 \
  --max_height=768 \
  --hub_token="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" \
  --lora_rank 0


#  --use_ema \
