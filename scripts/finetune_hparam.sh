#!/bin/bash

OUT_DIR="./"

OUT_NAME="roberta-large"

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=true

for SEED in {0..4}; do

  python finetune.py \
    --seed ${SEED} \
    --output_dir ${OUT_DIR}/models/${SEED}/${OUT_NAME} \
    --model_name_or_path "roberta-large" \
    --tokenizer_name "roberta-large" \
    --do_train \
    --do_eval \
    --max_seq_length 512 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 32 \
    --per_device_eval_batch_size 8 \
    --optim adamw_torch \
    --lr_scheduler_type linear \
    --warmup_ratio 0.1 \
    --num_train_epochs 20 \
    --load_best_model_at_end \
    --evaluation_strategy steps --eval_steps 50 \
    --save_total_limit 0 --save_strategy steps --save_steps 50 \
    --logging_steps 50 \
    --hparam_search \
    --target "label_sexist" \
    --patience 3 \
    --hparam_trials 20 \
    --report_to none \
    --bf16 \
    --dataloader_num_workers 8 \
    --metric_for_best_model f1

  echo "------ Cleaning run files! ------"
  rm -r ${OUT_DIR}/models/${TARGET}/${COUNTRY}/${SEED}/${OUT_NAME}/run-*
  rm -r ${OUT_DIR}/models/${TARGET}/${COUNTRY}/${SEED}/${OUT_NAME}/checkpoint-*

done
