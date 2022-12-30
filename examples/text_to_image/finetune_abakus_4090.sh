accelerate launch --mixed_precision="bf16" train_text_to_image.py \
  --pretrained_model_name_or_path="../../../mus2vid/models/stable-diffusion-v1-5" \
  --train_data_dir_var_aspect="/hdd/data/LAION/finetune_SD/laion_aesthetics/"  \
  --train_batch_size=1 \
  --gradient_accumulation_steps=512 \
  --gradient_checkpointing \
  --max_train_steps=3000 \
  --learning_rate=1e-06 \
  --max_grad_norm=1 \
  --lr_scheduler="cosine_with_restarts" --lr_warmup_steps=300 \
  --output_dir="sdv1-5-var-aspect-7" \
  --use_8bit_adam \
  --mixed_precision="bf16" \
  --max_files=150 \
  --max_width=768 \
  --max_height=768 \
  --hub_token="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"

#  --use_ema \
#  --compile \
#  --lr_scheduler="constant_with_warmup" --lr_warmup_steps=300 \
