#!/bin/bash

python models/GAT_base.py --num_epochs 10000 \
                         --num_seeds 3 \
                         --save_results True \
                         --ds blog \
                         --batch_size 64 \
                         --neg_pos_ratio 1


python models/GAT_base.py --num_epochs 50 \
                         --num_seeds 3 \
                         --save_results True \
                         --ds Yelp \
                         --batch_size 512 \
                         --neg_pos_ratio 1
