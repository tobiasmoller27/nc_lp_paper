#!/bin/bash

python models/GIN_base_batches.py --num_epochs 500 \
                         --num_seeds 3 \
                         --save_results True \
                         --ds blog \
                         --batch_size 64 \
                         --print_epochs True \
                         --neg_pos_ratio 1


python models/GIN_base_batches.py --num_epochs 50 \
                         --num_seeds 3 \
                         --save_results True \
                         --ds Yelp \
                         --batch_size 512 \
                          --print_epochs True \
                         --neg_pos_ratio 1