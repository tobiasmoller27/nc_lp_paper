#!/bin/bash

DATASETS=("CiteSeer" "Cora_ML" "Chameleon" "Roman_Empire")

for ds_name in "${DATASETS[@]}"; do
    python models/LPA_base.py --num_seeds 10 \
                    --save_results True \
                    --ds "$ds_name"
done
                                           