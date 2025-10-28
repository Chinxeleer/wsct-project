#!/bin/bash

set -e

python wsct_transformer.py \
    --data dataset.txt \
    --epochs 50 \
    --batch_size 32 \
    --lr 0.0001 \
    --d_model 128 \
    --n_layers 4 \
    --n_heads 8 \
    --d_ff 512 \
    --max_len 128 \
    --dropout 0.1 \
    --device cuda \
    --save_path wcst_model.pt
