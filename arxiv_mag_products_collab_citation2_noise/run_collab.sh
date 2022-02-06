#!/bin/bash

SCRIPT="collab_exp.py"

echo "model GAT"
echo "1 head"
echo "w/ val edges"
python ${SCRIPT} --type GAT --num_heads 1 --use_valedges_as_input | tee -a "collab_with_valedges_GAT_1_heads_results.txt"

echo "model GAT"
echo "1 head"
echo "w/o val edges"
python ${SCRIPT} --type GAT --num_heads 1 | tee -a "collab_without_valedges_GAT_1_heads_results.txt"

echo "model GAT2"
echo "1 head"
echo "w/ val edges"
python ${SCRIPT} --type GAT2 --num_heads 1 --use_valedges_as_input | tee -a "collab_with_valedges_GAT2_1_heads_results.txt"

echo "model GAT2"
echo "1 head"
echo "w/o val edges"
python ${SCRIPT} --type GAT2 --num_heads 1 | tee -a "collab_without_valedges_GAT2_1_heads_results.txt"

echo "model DPGAT"
echo "1 head"
echo "w/ val edges"
python ${SCRIPT} --type DPGAT --num_heads 1 --use_valedges_as_input | tee -a "collab_with_valedges_DPGAT_1_heads_results.txt"

echo "model DPGAT"
echo "1 head"
echo "w/o val edges"
python ${SCRIPT} --type DPGAT --num_heads 1 | tee -a "collab_without_valedges_DPGAT_1_heads_results.txt"


echo "model GAT"
echo "8 heads"
echo "w/ val edges"
python ${SCRIPT} --type GAT --num_heads 8 --use_valedges_as_input | tee -a "collab_with_valedges_GAT_8_heads_results.txt"

echo "model GAT"
echo "8 heads"
echo "w/o val edges"
python ${SCRIPT} --type GAT --num_heads 8 | tee -a "collab_without_valedges_GAT_8_heads_results.txt"

echo "model GAT2"
echo "8 heads"
echo "w/ val edges"
python ${SCRIPT} --type GAT2 --num_heads 8 --use_valedges_as_input | tee -a "collab_with_valedges_GAT2_8_heads_results.txt"

echo "model GAT2"
echo "8 heads"
echo "w/o val edges"
python ${SCRIPT} --type GAT2 --num_heads 8 | tee -a "collab_without_valedges_GAT2_8_heads_results.txt"

echo "model DPGAT"
echo "8 heads"
echo "w/ val edges"
python ${SCRIPT} --type DPGAT --num_heads 8 --use_valedges_as_input | tee -a "collab_with_valedges_DPGAT_8_heads_results.txt"

echo "model DPGAT"
echo "8 heads"
echo "w/o val edges"
python ${SCRIPT} --type DPGAT --num_heads 8 | tee -a "collab_without_valedges_DPGAT_8_heads_results.txt"
