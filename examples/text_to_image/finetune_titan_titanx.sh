CUDA_VISIBLE_DEVICES=0 TORCH_DISTRIBUTED_DEBUG="INFO" accelerate launch train_text_to_image.py \
  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
  --train_data_dir_var_aspect="../../../finetune_SD/laion_aesthetics_2/"  \
  --train_batch_size=1 \
  --gradient_accumulation_steps=8 \
  --gradient_checkpointing \
  --max_train_steps=2000 \
  --learning_rate=2e-06 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=100 \
  --output_dir="sdv1-5-var-aspect-2" \
  --use_8bit_adam \
  --mixed_precision="fp16" \
  --max_files=102 \
  --max_width=64 \
  --max_height=64 \
  --min_height=64 \
  --min_width=64 \
  --hub_token="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"

#  --use_ema \
# --mixed_precision="fp16" --num_processes=2 