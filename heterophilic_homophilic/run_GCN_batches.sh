#!/bin/bash

python models/GCN_base_batches.py --num_epochs 1000 \
                         --num_seeds 5 \
                         --save_results True \
                         --ds OGBN \
                         --batch_size 8192 \
                         --print_epochs True \
                         --neg_pos_ratio 1

python models/GCN_base_batches.py --num_epochs 1000 \
                         --num_seeds 5 \
                         --save_results True \
                         --ds Squirrel \
                         --print_epochs True \
                         --batch_size 1024
