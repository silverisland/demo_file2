#!/bin/bash

# Train base models
for model in DLinear PatchTST iTransformer TimesNet
do
    python -u run_longExp.py \
      --is_training 1 \
      --model_id dummy_test \
      --model $model \
      --data custom \
      --features M \
      --seq_len 96 \
      --pred_len 24 \
      --enc_in 1 \
      --des 'test' \
      --itr 1 \
      --batch_size 32 \
      --learning_rate 0.001 \
      --train_epochs 5
done

# Train Fusion model
python -u run_longExp.py \
  --is_training 1 \
  --model_id dummy_test \
  --model FusionModel \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 24 \
  --enc_in 1 \
  --des 'test' \
  --itr 1 \
  --batch_size 32 \
  --learning_rate 0.001 \
  --train_epochs 10
