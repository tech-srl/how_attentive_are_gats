#!/bin/bash

function run_arxiv() {
    SCRIPT="arxiv_mag_exp.py"
    echo "model $1"
    echo "$2 heads"
    python ${SCRIPT} --log_steps 1 \
                    --eval_steps 1 \
                    --type $1 \
                    --num_layers 3 \
                    --num_heads $2 \
                    --batch_size 20000 \
                    --num_workers 7 \
                    --dataset "ogbn-arxiv" \
                    --hidden_channels 256 \
                    --dropout 0.25 \
                    --lr 0.01 \
                    --epochs 50 \
                    --runs 10 \
                    --use_saint \
                    --num_steps 30 \
                    --walk_length 3 \
                    --use_layer_norm \
                    --use_residual | tee -a "arxiv_${1}_${2}_heads_results.txt"
}


run_arxiv "GAT" "1"
run_arxiv "GAT2" "1"
run_arxiv "DPGAT" "1"

run_arxiv "GAT" "8"
run_arxiv "GAT2" "8"
run_arxiv "DPGAT" "8"


