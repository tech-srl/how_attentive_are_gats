#!/bin/bash

function run_mag() {
    SCRIPT="arxiv_mag_exp.py"
    echo "model $1"
    echo "$2 heads"
    python ${SCRIPT} --log_steps 1 \
                     --eval_steps 1 \
                     --type $1 \
                     --num_layers 2 \
                     --num_heads $2 \
                     --batch_size 20000 \
                     --num_workers 7 \
                     --dataset "ogbn-mag" \
                     --hidden_channels 256 \
                     --dropout 0.5 \
                     --lr 0.01 \
                     --epochs 100 \
                     --runs 10 \
                     --use_layer_norm \
                     --use_residual | tee -a "mag_${1}_${2}_heads_results.txt"
}

run_mag "GAT" "1"
run_mag "GAT2" "1"
run_mag "DPGAT" "1"

run_mag "GAT" "8"
run_mag "GAT2" "8"
run_mag "DPGAT" "8"


