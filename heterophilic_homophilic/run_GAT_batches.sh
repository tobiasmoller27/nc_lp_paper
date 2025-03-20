#!/bin/bash

python models/GAT_base.py --num_epochs 10000 \
                         --num_seeds 5 \
                         --save_results True \
                         --ds OGBN \
                         --batch_size 8192 \
                         --neg_pos_ratio 1


python models/GAT_base.py --num_epochs 10000 \
                         --num_seeds 5 \
                         --save_results True \
                         --ds Squirrel \
                         --batch_size 1024
