#!/bin/bash
echo "[DEBUG] main_linear.sh called with env: DATA_DIR=$DATA_DIR CKPT=$CKPT TASK=$TASK"

set -x  # これ以降、実行されるコマンドを全部ログに出す

python3 main_linear.py \
    --dataset wisdm2019 \
    --max_epochs 50 \
    --encoder tpn \
    --data_dir "$DATA_DIR" \
    --split_strategy class \
    --num_classes 18 \
    --num_tasks 6 \
    --task_idx "$TASK" \
    --input_dim 3 \
    --feature_dim 128 \
    --num_layers 2 \
    --hidden_dim 256 \
    --dropout 0.1 \
    --batch_size 256 \
    --num_workers 2 \
    --optimizer sgd \
    --lr 0.1 \
    --weight_decay 1e-4 \
    --scheduler cosine \
    --pretrained_feature_extractor "$CKPT" \
    --name "wisdm-linear" \
    --project Kaizen_HAR_LINEAR_TEST \
    --entity crazy-sonnet-ambl \
    --wandb \
    --offline \
    --checkpoint_dir ./experiments/linear_eval
