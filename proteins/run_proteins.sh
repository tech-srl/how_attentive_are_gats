#!/bin/bash

SCRIPT="proteins_exp.py"

echo "model GAT"
echo "1 head"
python ${SCRIPT} --type GAT --n-heads 1 | tee -a "proteins_GAT_1_heads_results.txt"
echo "model GAT2"
echo "1 head"
python ${SCRIPT} --type GAT2 --n-heads 1 | tee -a "proteins_GAT2_1_heads_results.txt"
echo "model DPGAT"
echo "1 head"
python ${SCRIPT} --type DPGAT --n-heads 1 --max_loss 0.32 --patient 100 --lr 0.001 | tee -a "proteins_DPGAT_1_heads_results.txt"

echo "model GAT"
echo "8 heads"
python ${SCRIPT} --type GAT --n-heads 8 | tee -a "proteins_GAT_8_heads_results.txt"
echo "model GAT2"
echo "8 heads"
python ${SCRIPT} --type GAT2 --n-heads 8 | tee -a "proteins_GAT2_1_heads_results.txt"
echo "model DPGAT"
echo "8 heads"
python ${SCRIPT} --type DPGAT --n-heads 8 --max_loss 0.32 --patient 100 --lr 0.001 | tee -a "proteins_DPGAT_1_heads_results.txt"



