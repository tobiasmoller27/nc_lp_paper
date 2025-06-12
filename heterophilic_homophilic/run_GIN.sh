#!/bin/bash

DATASETS=("CiteSeer" "Cora_ML" "Chameleon" "Roman_Empire")

for ds_name in "${DATASETS[@]}"; do
    python models/GIN_base.py --num_epochs 10000 \
                    --num_seeds 10 \
                    --save_results True \
                    --ds "$ds_name"
done