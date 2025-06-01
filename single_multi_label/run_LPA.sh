#!/bin/bash

DATASETS=("DBLP" "blog" "Yelp")

for ds_name in "${DATASETS[@]}"; do
    python models/LPA_base.py --num_seeds 10 \
                    --save_results True \
                    --ds "$ds_name"
done
                                           