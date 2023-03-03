#!/bin/bash

OUT_DIR="./"
OUT_NAME="microsoft/deberta-v3-large"

export CUDA_VISIBLE_DEVICES=1
export TOKENIZERS_PARALLELISM=true

for SEED in 0; do

  python run_mlm.py \
    --output_dir ${OUT_DIR}/models/${SEED}/deberta-adapted \
    --overwrite_output_dir \
    --train_file ./mlm_data.csv \
    --seed ${SEED} \
    --model_name_or_path "microsoft/deberta-v3-large" \
    --tokenizer_name "microsoft/deberta-v3-large" \
    --do_train \
    --do_eval \
    --max_seq_length 512 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 128 \
    --per_device_eval_batch_size 8 \
    --optim adamw_torch \
    --learning_rate 1e-5 \
    --lr_scheduler_type constant_with_warmup \
    --warmup_steps 100 \
    --weight_decay 0.01 \
    --num_train_epochs 100 \
    --report_to wandb \
    --load_best_model_at_end \
    --evaluation_strategy steps --eval_steps 200 \
    --save_total_limit 1 --save_strategy steps --save_steps 200 \
    --logging_steps 25 \
    --dataloader_num_workers 8 \
    --bf16 \
    --push_to_hub 

done
