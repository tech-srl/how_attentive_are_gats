#!/bin/bash

SCRIPT="citation2_exp.py"

echo "model GAT"
echo "1 head"
python ${SCRIPT} --type GAT --num_heads 1 | tee -a "citation2_GAT_1_heads_results.txt"
echo "model GAT2"
echo "1 head"
python ${SCRIPT} --type GAT2 --num_heads 1 | tee -a "citation2_GAT2_1_heads_results.txt"
echo "model DPGAT"
echo "1 head"
python ${SCRIPT} --type DPGAT --num_heads 1 | tee -a "citation2_DPGAT_1_heads_results.txt"

echo "model GAT"
echo "8 heads"
python ${SCRIPT} --type GAT --num_heads 8 | tee -a "citation2_GAT_8_heads_results.txt"
echo "model GAT2"
echo "8 heads"
python ${SCRIPT} --type GAT2 --num_heads 8 | tee -a "citation2_GAT2_8_heads_results.txt"
echo "model DPGAT"
echo "8 heads"
python ${SCRIPT} --type DPGAT --num_heads 8 | tee -a "citation2_DPGAT_8_heads_results.txt"
