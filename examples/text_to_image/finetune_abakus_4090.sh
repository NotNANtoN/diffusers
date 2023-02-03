accelerate launch --mixed_precision="bf16" train_text_to_image.py \
  --pretrained_model_name_or_path="../../../mus2vid/models/stable-diffusion-v1-5" \
  --train_data_dir_var_aspect="/hdd/data/LAION/finetune_SD/laion_aesthetics/"  \
  --train_batch_size=2 \
  --gradient_accumulation_steps=512 \
  --max_train_steps=1500 \
  --lr_scheduler="cosine_with_restarts" --lr_warmup_steps=100 \
  --max_grad_norm=1 \
  --output_dir="sdv1-5-var-aspect-18" \
  --use_8bit_adam \
  --mixed_precision="bf16" \
  --max_files=500 \
  --max_width=768 \
  --max_height=768 \
  --hub_token="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" \
  --gradient_checkpointing \
  --lora_rank 0 \
  --learning_rate=5e-06 \
  --num_experts 0 \
  --min_width 512 \
  --min_height 368
  
  
  
#  --learning_rate=3e-04 \
#  --use_ema \
#  --ema_decay 0.9
#  --compile \
#--lr_scheduler="constant_with_warmup" --lr_warmup_steps=400 \
#  --learning_rate=1e-06 \ # for non-lora
#--pretrained_model_name_or_path="../../../mus2vid/models/stable-diffusion-v1-5" \
#  --pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1-base" \
